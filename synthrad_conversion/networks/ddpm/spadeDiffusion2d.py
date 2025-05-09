import sys
sys.path.append('./networks/ddpm')

import os
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from monai.utils import first, set_determinism
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from monai.transforms.utils import allow_missing_keys_mode
from monai.transforms import Rotate90

from generative_custom.inferers import transformDiffusionInferer
from generative_custom.networks.nets import DiffusionModelUNet
from generative_custom.networks.schedulers import DDPMScheduler
from generative.networks.nets import SPADEDiffusionModelUNet

from utils.evaluate import (
    val_log,compare_imgs, 
    reverse_transforms,
    arrange_3_histograms,
    calculate_mask_metrices,
    Postprocessfactory,
    reverse_normalize_data,)

def checkdata(loader,inputtransforms,output_for_check=1,save_folder='./logs/test_images'):
    from PIL import Image
    import matplotlib
    matplotlib.use('Qt5Agg')
    for i, batch in enumerate(loader):
        images = batch["target"]
        labels = batch["target"]
        
        images=images[:,:,:,:,None]
        try:
            volume=torch.cat((volume,images),-1)
        except:
            volume=images

    volume = volume[0,:,:,:,:] #(B,C,H,W,D)    
    # the input into reverse transform should be in form: 20 is the cropped depth
    # (1, 512, 512, 20) -> (1, 452, 315, 5) C,H,W,D
    print (volume.shape)
    val_output_dict = {"target": volume}
    with allow_missing_keys_mode(inputtransforms):
        reversed_images_dict=inputtransforms.inverse(val_output_dict)

    for i in range(images.shape[0]):
        print(images.shape)
        imgformat='png'
        dpi=300
        os.makedirs(save_folder,exist_ok=True)
        if output_for_check == 1:
            # save images to file
            for j in range(images.shape[-1]):
                saved_name=os.path.join(save_folder,f"{i}_{j}.{imgformat}")
                img = images[:,:,:,j]
                #img =img.squeeze().cpu().numpy()
                img = img.permute(1,2,0).squeeze().cpu().numpy()
                img = (img * 255).astype(np.uint8)
                img = Image.fromarray(img)
                #img.save(saved_name)

                fig_ct = plt.figure()
                plt.gca().set_axis_off()
                plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                            hspace = 0, wspace = 0)
                plt.margins(0,0)
                plt.imshow(img, cmap='gray') #.squeeze()
                plt.savefig(saved_name.replace(f'.{imgformat}',f'_reversed.{imgformat}'), format=f'{imgformat}'
                            , bbox_inches='tight', pad_inches=0, dpi=dpi)
                plt.close(fig_ct)

import torch.nn as nn
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)
def load_pretrained_model(model, opt, pretrained_path=None):
    if pretrained_path is not None and os.path.exists(pretrained_path):
        latest_ckpt=pretrained_path
        loaded_state = torch.load(latest_ckpt)
        print(f'use pretrained model: {latest_ckpt}') 
        if 'epoch' in loaded_state:
            init_epoch=loaded_state["epoch"] # load or manually set
            print(f'continue from epoch {init_epoch}') 
            #init_epoch = int(input('Enter epoch number: '))
        else:
            print('no epoch information in the checkpoint file')
            init_epoch = int(input('Enter epoch number: '))
        model.load_state_dict(loaded_state["model"]) #
        opt.load_state_dict(loaded_state["opt"])
    else:
        init_epoch=0
        #model = model.apply(weights_init)
        print(f'start new training') 
    return model, opt, init_epoch
def arrange_images_assemble(img_assemble,
                   titles,
                   saved_name,
                   imgformat='jpg',
                   dpi = 500):
        image_number=len(img_assemble)
        fig, axs = plt.subplots(1, image_number, figsize=(16, 5)) # 
        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                    hspace = 0, wspace = 0.1)
        plt.margins(0,0)
        axs = axs.flatten()
        for i in range(image_number):
            axs[i].imshow(img_assemble[i], cmap='gray')
            axs[i].set_title(titles[i])
            axs[i].axis('off')
        # save image as png
        fig.savefig(saved_name, format=f'{imgformat}', bbox_inches='tight', pad_inches=0, dpi=dpi)
        #plt.show()
        plt.close(fig)

class LossTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.generator_losses = []
        self.ssim = []
        self.psnr = []
        self.mae = []

    def update(self, gen_loss):
        self.generator_losses.append(gen_loss)
    
    def update_metrics(self, ssim, psnr, mae):
        self.ssim.append(ssim)
        self.psnr.append(psnr)
        self.mae.append(mae)

    def get_mean_losses(self):
        mean_gen_loss = sum(self.generator_losses) / len(self.generator_losses)
        return mean_gen_loss
    def get_mean_metrics(self):
        mean_ssim = sum(self.ssim) / len(self.ssim)
        mean_psnr = sum(self.psnr) / len(self.psnr)
        mean_mae = sum(self.mae) / len(self.mae)
        return mean_ssim, mean_psnr, mean_mae
    
    def get_epoch_losses(self):
        return self.generator_losses

class DiffusionModel: #(nn.Module)
    def __init__(self,config,my_paths):
        #super(DiffusionModel, self).__init__()
        num_train_timesteps=config.ddpm.num_train_timesteps
        num_inference_steps=config.ddpm.num_inference_steps
        lr=config.train.learning_rate
        self.config=config
        self.device = torch.device(f'cuda:{config.GPU_ID}' if torch.cuda.is_available() else 'cpu')
        patch_depth = self.config.dataset.patch_size[-1]
        spatial_dims = 2 if patch_depth==1 else 3

        self.model = SPADEDiffusionModelUNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            num_res_blocks=4,
            label_nc=119,
            num_channels=(64, 128, 256, 512),
            attention_levels=(False, False, False, True),
            norm_num_groups=16,
            cross_attention_dim=1,
            with_conditioning=True,
            resblock_updown=True,
        )
        self.model.to(self.device)
        self.scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr) # 2.5e-5
        self.inferer = transformDiffusionInferer(self.scheduler)
        self.paths=my_paths
        self.saved_results_name=my_paths["saved_img_folder"] #f'./logs/{args.run_name}/results'
        self.saved_models_name=my_paths["saved_model_folder"] #f'./logs/{args.run_name}/models'
        self.saved_runs_name=my_paths["tensorboard_log_dir"] #f'./logs/{args.run_name}/runs'
        self.output_dir=os.path.join(my_paths['saved_img_folder'])   
        #os.makedirs(self.saved_results_name, exist_ok=True)
        #os.makedirs(self.saved_models_name, exist_ok=True)
        self.num_inference_steps = num_inference_steps
    
    def train(self,train_loader,val_loader):
        use_pretrained = False
        device = torch.device(f'cuda:{self.config.GPU_ID}' if torch.cuda.is_available() else 'cpu')
        model = self.model
        scheduler = self.scheduler
        optimizer = self.optimizer
        inferer = self.inferer
        logger = SummaryWriter(self.saved_runs_name)
        writeTensorboard=self.config.train.writeTensorboard

        if use_pretrained:
            model = torch.hub.load("marksgraham/pretrained_generative_models:v0.2", model="ddpm_2d", verbose=True).to(device)
        else:
            n_epochs = self.config.train.num_epochs
            val_interval = self.config.train.val_epoch_interval
            pretrained_path = self.config.ckpt_path
            epoch_loss_list = []
            model, optimizer, init_epoch = load_pretrained_model(model, optimizer, pretrained_path)
            
            self.loss_tracker = LossTracker()
            scaler = GradScaler()
            total_start = time.time()
            global_step=0
            for continue_epoch in range(n_epochs):
                epoch = continue_epoch + init_epoch + 1
                self.epoch = epoch
                epoch_num_total = n_epochs + init_epoch
                print("-" * 10)
                print(f"epoch {epoch}/{epoch_num_total}")
                model.train()
                epoch_loss = 0
                #progress_bar = tqdm(enumerate(train_loader), total=batch_number, ncols=70)
                #progress_bar.set_description(f"Epoch {epoch}")
                step = 0
                for batch in tqdm(train_loader): #progress_bar
                    step += 1
                    global_step += 1
                    sources = batch["source"].to(device) # MRI image
                    targets = batch["target"].to(device) # CT image
                    if step==1 and continue_epoch==0:
                        print('input shape:', sources.shape)
                    print('batch:',step,', input shape:', sources.shape)
                    optimizer.zero_grad(set_to_none=True)
                    with autocast(enabled=True):
                        # Generate random noise
                        noise = torch.randn_like(targets).to(device)
                        timesteps = torch.randint(
                            0, inferer.scheduler.num_train_timesteps, (targets.shape[0],), device=targets.device
                        ).long()
                        # Get model prediction
                        noise_pred = inferer(inputs=targets, append_image=sources, diffusion_model=model, noise=noise, timesteps=timesteps)
                        loss = F.mse_loss(noise_pred.float(), noise.float())
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    epoch_loss += loss.item()
                    self.loss_tracker.update(loss.item())
                    self.plot_learning_curves()
                    
                    output_loss_to_txt=False
                    if output_loss_to_txt:
                        printout=(
                                    "[Epoch %d/%d] [Batch %d] [loss: %f] \n"
                                    % (
                                        epoch,
                                        epoch_num_total,
                                        step,
                                        loss.item(),                  
                                    )
                                )
                        with open(self.paths["train_loss_file"], 'a') as f: # append mode
                            f.write(printout)
                    
                    if writeTensorboard:
                        logger.add_scalar("train_loss_MSE", loss.item(), global_step=global_step)
                

                #progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})
                #epoch_loss_list.append(epoch_loss / (step + 1))
                if writeTensorboard:
                    logger.add_scalar("train_epoch_loss", epoch_loss / (step + 1), global_step=epoch)
                
                if (epoch) % self.config.train.save_ckpt_interval == 0:
                    model_save_path = os.path.join(self.saved_models_name, f"model_{epoch % self.config.train.save_last}.pt")
                    torch.save({'epoch': epoch,
                        'model': model.state_dict(),
                        'opt': optimizer.state_dict()}, model_save_path
                        )
                        
                val_epoch_loss_list = []
                if (epoch) % val_interval == 0:
                    model.eval()
                    val_epoch_loss = 0
                   
                    val_epoch_loss_list.append(val_epoch_loss / (step + 1))
                    self._test_nifti(val_loader, load_pretrained=False)
                    if writeTensorboard:
                        logger.add_scalar("val_epoch_loss", val_epoch_loss / (step + 1), global_step=epoch)
            total_time = time.time() - total_start
            print(f"train completed, total time: {total_time}.")    
    
    def _sample(self, model,
                targets, inputs, 
                inferer, scheduler,
                num_inference_steps=1000,
                epoch=0, step=0, 
                device="cuda", i=0, 
                save_imgs=True):
        # Sampling
        # targets: CT image, which is  the target
        # inputs: MRI image, which is to be converted to CT image
        noise = torch.randn_like(targets).to(device)
        scheduler.set_timesteps(num_inference_steps=num_inference_steps)
        targets_single=targets
        inputs_single=inputs

        '''
        noise = torch.randn((1, targets.shape[1], targets.shape[2], targets.shape[3])) # B,C,H,W
        noise = noise.to(device)
        scheduler.set_timesteps(num_inference_steps=num_inference_steps)
        targets_single = targets[i,:,:,:]
        targets_single = targets_single[None,:,:,:]                
        inputs_single = inputs[i,:,:,:]
        inputs_single = inputs_single[None,:,:,:]
        '''

        with autocast(enabled=True):
            image = inferer.sample(input_noise=noise, input_image=inputs_single, diffusion_model=model, scheduler=scheduler)
        image_loss = F.mse_loss(image,targets_single)
        saved_name=os.path.join(self.saved_results_name,f"{epoch}_{step}.jpg")
        targets_single = targets_single.detach().cpu()
        inputs_single = inputs_single.detach().cpu()
        image = image.detach().cpu()

        '''if save_imgs:                         
            compare_imgs(input_imgs=inputs_single, 
                         target_imgs=targets_single,
                         fake_imgs=image,
                         saved_name=saved_name)'''
            
            
        return image_loss,inputs_single,targets_single,image
     
    def _test_nifti(self, val_loader, load_pretrained=True):
        
        device = self.device
        model = self.model
        scheduler = self.scheduler
        optimizer = self.optimizer
        inferer = self.inferer
        pretrained_path = self.config.ckpt_path
        if load_pretrained:
            model, optimizer, init_epoch = load_pretrained_model(model, optimizer, pretrained_path)
        else:
            init_epoch=self.epoch
        self.saved_logs_name=self.saved_results_name.replace('results','datalogs')
        

        epoch = init_epoch
        model.eval()
        
        self.loss_tracker = LossTracker()

        for step, batch in enumerate(val_loader): 
            if step >= self.config.train.sample_range_lower and step <= self.config.train.sample_range_upper: 
                print(step)
                targets = batch["target"].to(device) # CT image
                inputs = batch["source"].to(device) # MRI image
        
                print("original image shape:", targets.shape)
                val_loss, input_image, orig_image, generated_image_batch = self._sample(model=model, 
                                                                    targets=targets, 
                                                                    inputs=inputs, 
                                                                    inferer=inferer,
                                                                    scheduler=scheduler,
                                                                    num_inference_steps = self.num_inference_steps,
                                                                    epoch=epoch, step=step, 
                                                                    device=device, i=0, # i=0 because val_loader batch_size=1
                                                                    save_imgs=True) 
                if len(inputs.shape)==4:
                    self.evaluate2dBatch(val_loss, input_image, orig_image, generated_image_batch, inputs, targets,epoch,step)
                elif len(inputs.shape)==5:
                    self.evaluate25dBatch(val_loss, input_image, orig_image, generated_image_batch, inputs, targets,epoch,step)

    def evaluateSlice(self, image_losses, val_loss, 
                      input_image_slice, orig_image_slice, generated_image_slice, inputs_volume,
                      epoch,step,batch_idx=0,slice_idx=0):
        input_image=input_image_slice
        orig_image=orig_image_slice
        generated_image=generated_image_slice
        restore_transforms=True
        if restore_transforms:
            orig_image = reverse_normalize_data(orig_image, mode=self.config.dataset.normalize)
            generated_image = reverse_normalize_data(generated_image, mode=self.config.dataset.normalize)
        if self.config.dataset.rotate:
            img_assemble = [
                input_image.squeeze().cpu().detach(), #.permute(1,0)
                orig_image.squeeze().cpu().detach(),  #.permute(1,0)
                generated_image.squeeze().cpu().detach(), #.permute(1,0)
                ]
        else:
            img_assemble = [
                input_image.squeeze().permute(1,0).cpu().detach(),
                orig_image.squeeze().permute(1,0).cpu().detach(), 
                generated_image.squeeze().permute(1,0).cpu().detach(),
                ]
        self.loss_tracker.update(val_loss.item())
        metrics=calculate_mask_metrices(img_assemble[-1],  #*self.config.dataset.scalefactor
                                        img_assemble[1],  #*self.config.dataset.scalefactor
                                        None, 
                                        self.paths["train_metrics_file"], 
                                        f"{epoch}, {step}", 
                                        printoutput=False)
        ssim=metrics['ssim']
        psnr=metrics['psnr']
        mae=metrics['mae']
        self.loss_tracker.update_metrics(ssim, psnr, mae)
        self.plot_learning_curves()
        self.plot_metrics()
        printout=(
            "[Epoch %d] [Batch step %d] [batch_idx %d] [slice_idx %d] [loss: %f] [ssim: %f] [psnr: %f] [mae: %f] \n"
            % (
                epoch,
                step,
                batch_idx,
                slice_idx,
                val_loss.item(),
                ssim,
                psnr,
                mae,                    
            )
        )
        print(printout)
        with open(self.paths["val_log_file"], 'a') as f: # append mode
            f.write(printout)

    
        imgformat = 'jpg'
        dpi = 500
        titles = ['MRI', 'CT', self.config.model_name]
        img_folder=os.path.join(self.paths["saved_img_folder"], "img")
        os.makedirs(img_folder, exist_ok=True)
        hist_folder=os.path.join(self.paths["saved_img_folder"], "hist")
        os.makedirs(hist_folder, exist_ok=True)
        arrange_images_assemble(img_assemble, 
                        titles,
                        saved_name=os.path.join(img_folder, 
                            f"compare_epoch_{epoch}_{step}_{batch_idx}_{slice_idx}.{imgformat}"), 
                        imgformat=imgformat, dpi=dpi)
        arrange_3_histograms(img_assemble[0].numpy(), 
                                img_assemble[1].numpy(), 
                                img_assemble[-1].numpy(), 
                                saved_name=os.path.join(hist_folder, 
                                f"histograms_epoch_{epoch}_{step}_{batch_idx}_{slice_idx}.{imgformat}"),
                                x_lower_limit=self.config.visualize.x_lower_limit, 
                                x_upper_limit=self.config.visualize.x_upper_limit,
                                )

        generated_image=generated_image.unsqueeze(-1)
        #print("unsqueezed image shape:", generated_image.shape)
        image_losses.append(val_loss)
        return generated_image
    
    def evaluate2dBatch(self, val_loss, input_image, orig_image, generated_image_batch, inputs, targets,epoch,step):
        image_losses=[]
        output_file=os.path.join(self.saved_results_name, f"test")
        for batch_idx in range(targets.shape[0]):
            generated_image=generated_image_batch[batch_idx]
            input_image=inputs[batch_idx]
            orig_image=targets[batch_idx]
            restore_transforms=True
            if restore_transforms:
                orig_image = reverse_normalize_data(orig_image, mode=self.config.dataset.normalize)
                generated_image = reverse_normalize_data(generated_image, mode=self.config.dataset.normalize)
            if self.config.dataset.rotate:
                img_assemble = [
                    input_image.squeeze().cpu().detach(), #.permute(1,0)
                    orig_image.squeeze().cpu().detach(),  #.permute(1,0)
                    generated_image.squeeze().cpu().detach(), #.permute(1,0)
                    ]
            else:
                img_assemble = [
                    input_image.squeeze().permute(1,0).cpu().detach(),
                    orig_image.squeeze().permute(1,0).cpu().detach(), 
                    generated_image.squeeze().permute(1,0).cpu().detach(),
                    ]
            self.loss_tracker.update(val_loss.item())
            metrics=calculate_mask_metrices(img_assemble[-1],  #*self.config.dataset.scalefactor
                                            img_assemble[1],  #*self.config.dataset.scalefactor
                                            None, 
                                            self.paths["train_metrics_file"], 
                                            f"{epoch}, {step}", 
                                            printoutput=False)
            ssim=metrics['ssim']
            psnr=metrics['psnr']
            mae=metrics['mae']
            self.loss_tracker.update_metrics(ssim, psnr, mae)
            self.plot_learning_curves()
            self.plot_metrics()
            printout=(
                "[Epoch %d] [Batch %d] [idx %d] [loss: %f] [ssim: %f] [psnr: %f] [mae: %f] \n"
                % (
                    epoch,
                    step,
                    batch_idx,
                    val_loss.item(),
                    ssim,
                    psnr,
                    mae,                    
                )
            )
            print(printout)
            with open(self.paths["val_log_file"], 'a') as f: # append mode
                f.write(printout)

        
            imgformat = 'jpg'
            dpi = 500
            titles = ['MRI', 'CT', self.config.model_name]
            img_folder=os.path.join(self.paths["saved_img_folder"], "img")
            os.makedirs(img_folder, exist_ok=True)
            hist_folder=os.path.join(self.paths["saved_img_folder"], "hist")
            os.makedirs(hist_folder, exist_ok=True)
            arrange_images_assemble(img_assemble, 
                            titles,
                            saved_name=os.path.join(img_folder, 
                                f"compare_epoch_{epoch}_{step}_{batch_idx}.{imgformat}"), 
                            imgformat=imgformat, dpi=dpi)
            arrange_3_histograms(img_assemble[0].numpy(), 
                                    img_assemble[1].numpy(), 
                                    img_assemble[-1].numpy(), 
                                    saved_name=os.path.join(hist_folder, 
                                    f"histograms_epoch_{epoch}_{step}_{batch_idx}.{imgformat}"),
                                    x_lower_limit=self.config.visualize.x_lower_limit, 
                                    x_upper_limit=self.config.visualize.x_upper_limit,
                                    )

            generated_image=generated_image.unsqueeze(-1)
            print("unsqueezed image shape:", generated_image.shape)
            image_losses.append(val_loss)

            try:
                image_volume=torch.cat((image_volume,generated_image),-1)#pay attention to the order!
            except:
                image_volume=generated_image
            
            image_volume.meta = inputs[0].meta
            '''
            ["filename_or_obj", "affine", 
            "original_affine", 
            "spatial_shape", 
            "original_channel_dim", 
            "filename_or_obj", 
            "original_affine", 
            "spatial_shape", 
            "original_channel_dim"]
            '''
            image_volume.meta["filename_or_obj"] = "prediction"
        
        from utils.spacing import resample_ct_volume
        image_volume = resample_ct_volume(image_volume.unsqueeze(0), original_spacing=(1.0, 1.0, 1.0), new_spacing=(1.0, 1.0, 2.5))

        mean_ssim, mean_psnr, mean_mae = self.loss_tracker.get_mean_metrics()
        printout_conclusion=(
                    "[Epoch %d] [conclusion] [mean ssim: %f] [mean psnr: %f] [mean mae: %f] \n"
                    % (
                        epoch,
                        mean_ssim,
                        mean_psnr,
                        mean_mae,                    
                    )
                )
        print(printout_conclusion)
        with open(self.paths["val_log_file"], 'a') as f: # append mode
            f.write(printout_conclusion)
        reversed_image = image_volume
        '''reversed_image = reverse_transforms(output_images=image_volume, 
                                            orig_images=inputs, # output reverse should according to the original inputs MRI
                                            transforms=val_transforms)'''
        '''reversed_image = Rotate90(k=1)(image_volume)'''

        
        #reversed_image = reversed_image.unsqueeze(0)
        #print("reversed image shape:", reversed_image.shape)
        #SaveImage(output_dir=output_file, resample=True)(reversed_image.detach().cpu())
        
        # Create a NIfTI image object
        reversed_image = torch.squeeze(reversed_image.detach().cpu()).numpy()
        reversed_image = np.rot90(reversed_image, axes=(0, 1), k=2)
        import nibabel as nib
        nifti_img = nib.Nifti1Image(reversed_image, affine=np.eye(4))  # You can customize the affine transformation matrix if needed
        # Save the NIfTI image to a file
        output_file_name = output_file + '.nii.gz'
        nib.save(nifti_img, output_file_name)

    def evaluate25dBatch(self, val_loss, input_image, orig_image, generated_image_batch, inputs, targets,epoch,step):
        image_losses=[]
        output_file=os.path.join(self.saved_results_name, f"test")
        for batch_idx in range(targets.shape[0]):
            for slice_idx in range(targets.shape[-1]):
                print(f"evaluate {batch_idx+1}/{targets.shape[0]} inside batch {step}")
                print(f"for 2.5D image, this is the {slice_idx+1}/{targets.shape[-1]} th slice")
                input_image_slice=inputs[batch_idx,:,:,:,slice_idx]
                orig_image_slice=targets[batch_idx,:,:,:,slice_idx]
                generated_image_slice=generated_image_batch[batch_idx,:,:,:,slice_idx] # B,C,W,H,N
                generated_image = self.evaluateSlice(image_losses, val_loss, 
                      input_image_slice, orig_image_slice, generated_image_slice, inputs,
                      epoch,step,batch_idx,slice_idx)
                try:
                    image_volume=torch.cat((image_volume,generated_image),-1)#pay attention to the order!
                except:
                    image_volume=generated_image
        
                image_volume.meta = inputs[0].meta
        '''
        ["filename_or_obj", "affine", 
        "original_affine", 
        "spatial_shape", 
        "original_channel_dim", 
        "filename_or_obj", 
        "original_affine", 
        "spatial_shape", 
        "original_channel_dim"]
        '''
        image_volume.meta["filename_or_obj"] = "prediction"
            
        from utils.spacing import resample_ct_volume
        image_volume = resample_ct_volume(image_volume.unsqueeze(0), original_spacing=(1.0, 1.0, 1.0), new_spacing=(1.0, 1.0, 2.5))

        mean_ssim, mean_psnr, mean_mae = self.loss_tracker.get_mean_metrics()
        printout_conclusion=(
                    "[Epoch %d] [conclusion] [mean ssim: %f] [mean psnr: %f] [mean mae: %f] \n"
                    % (
                        epoch,
                        mean_ssim,
                        mean_psnr,
                        mean_mae,                    
                    )
                )
        print(printout_conclusion)
        with open(self.paths["val_log_file"], 'a') as f: # append mode
            f.write(printout_conclusion)
        reversed_image = image_volume
        '''reversed_image = reverse_transforms(output_images=image_volume, 
                                            orig_images=inputs, # output reverse should according to the original inputs MRI
                                            transforms=val_transforms)'''
        '''reversed_image = Rotate90(k=1)(image_volume)'''

        
        #reversed_image = reversed_image.unsqueeze(0)
        #print("reversed image shape:", reversed_image.shape)
        #SaveImage(output_dir=output_file, resample=True)(reversed_image.detach().cpu())
        
        # Create a NIfTI image object
        reversed_image = torch.squeeze(reversed_image.detach().cpu()).numpy()
        reversed_image = np.rot90(reversed_image, axes=(0, 1), k=2)
        import nibabel as nib
        nifti_img = nib.Nifti1Image(reversed_image, affine=np.eye(4))  # You can customize the affine transformation matrix if needed
        # Save the NIfTI image to a file
        output_file_name = output_file + '.nii.gz'
        nib.save(nifti_img, output_file_name)

    def postprocess(self, epoch,epoch_num_total, step, loss):
        printout=(
                                "[Epoch %d/%d] [Batch %d] [loss: %f] \n"
                                % (
                                    epoch,
                                    epoch_num_total,
                                    step,
                                    loss.item(),                  
                                )
                            )
        with open(self.paths["train_loss_file"], 'a') as f: # append mode
            f.write(printout)

    def testdata(self, train_loader, output_for_check=0,save_folder='test_images'):
        from PIL import Image
        for i, data in enumerate(train_loader):
            targets=data['target']
            inputs=data['source']
            print(i, ' targets: ',targets.shape)
            print(i, ' inputs: ',inputs.shape)
            os.makedirs(save_folder,exist_ok=True)
            if output_for_check == 1:
                # save images to file
                for j in range(targets.shape[0]):
                    img = targets[j,:,:,:]
                    img = img.permute(1,2,0).squeeze().cpu().numpy()
                    img = (img * 255).astype(np.uint8)
                    img = Image.fromarray(img)
                    img.save(f'{save_folder}/'+str(i)+'_'+str(j)+'.png')
            with open(f'{save_folder}/parameter.txt', 'a') as f:
                f.write('targets batch:' + str(targets.shape)+'\n')
                f.write('inputs batch:' + str(inputs.shape)+'\n')
                f.write('\n')
    
    def plotchain(self):
        device = self.device
        model = self.model
        scheduler = self.scheduler
        inferer = self.inferer
        model.eval()
        noise = torch.randn((1, 1, 64, 64))
        noise = noise.to(device)
        scheduler.set_timesteps(num_inference_steps=1000)
        with autocast(enabled=True):
            image, intermediates = inferer.sample(
                input_noise=noise,   input_image=noise,  diffusion_model=model, scheduler=scheduler, save_intermediates=True, intermediate_steps=100
            )

        chain = torch.cat(intermediates, dim=-1)
        num_images = chain.shape[-1]

        plt.style.use("default")
        plt.imshow(chain[0, 0].cpu(), vmin=0, vmax=1, cmap="gray")
        plt.tight_layout()
        plt.axis("off")
        plt.show()

    def plot_learning_curves(self):
        epoch_gen_loss_list=self.loss_tracker.generator_losses
        plt.style.use("ggplot")
        plt.title("generator_loss", fontsize=20)
        plt.plot(epoch_gen_loss_list, label="training loss")
        plt.yticks(fontsize=12)
        plt.xticks(fontsize=12)
        plt.xlabel("Batches", fontsize=16)
        plt.ylabel("Loss", fontsize=16)
        plt.legend(prop={"size": 14})
        plt.savefig(f"{self.output_dir}/generator_loss.png")
        plt.close()
        #plt.show()

    def plot_metrics(self):
        epoch_ssim_list=self.loss_tracker.ssim
        epoch_psnr_list=self.loss_tracker.psnr
        epoch_mae_list=self.loss_tracker.mae
        
        fig, axes = plt.subplots(3, 1, figsize=(10, 15))
        
        axes[0].plot(epoch_ssim_list, color="C0", linewidth=2.0, label="SSIM")
        axes[0].set_title("SSIM", fontsize=20)
        axes[0].set_xlabel("Batches", fontsize=16)
        axes[0].set_ylabel("SSIM", fontsize=16)
        axes[0].legend(prop={"size": 14})
        axes[0].tick_params(axis='both', which='major', labelsize=12)
        
        axes[1].plot(epoch_psnr_list, color="C1", linewidth=2.0, label="PSNR")
        axes[1].set_title("PSNR", fontsize=20)
        axes[1].set_xlabel("Batches", fontsize=16)
        axes[1].set_ylabel("PSNR", fontsize=16)
        axes[1].legend(prop={"size": 14})
        axes[1].tick_params(axis='both', which='major', labelsize=12)
        
        axes[2].plot(epoch_mae_list, color="C2", linewidth=2.0, label="MAE")
        axes[2].set_title("MAE", fontsize=20)
        axes[2].set_xlabel("Batches", fontsize=16)
        axes[2].set_ylabel("MAE", fontsize=16)
        axes[2].legend(prop={"size": 14})
        axes[2].tick_params(axis='both', which='major', labelsize=12)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/metrics.png")
        plt.close()
        #plt.show()



if __name__ == "__main__":
    set_determinism(42)
    import argparse
    from utils.my_configs_yacs import config_path
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="DDPM")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--n_epochs", type=int, default=50) 
    parser.add_argument("--val_interval", type=int, default=5)
    parser.add_argument("--train_number", type=int, default=170)
    parser.add_argument("--normalize", type=str, default="minmax")
    parser.add_argument("--pad", type=str, default="minimum")
    parser.add_argument("--val_number", type=int, default=1)
    parser.add_argument("--center_crop", type=int, default=20) # set to 0 or -1 means no cropping
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--pretrained_path", type=str, default=None)
    parser.add_argument("--lr", type=float, default=2.5e-5)
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--num_inference_steps", type=int, default=1000)
    parser.add_argument("--dataset_path", type=str, default=r"F:\yang_Projects\Datasets\Task1\pelvis")
    # r"F:\yang_Projects\Datasets\Task1\pelvis" 
    # r"C:\Users\56991\Projects\Datasets\Task1\pelvis" 
    # r"D:\Projects\data\Task1\pelvis" 
    parser.add_argument("--GPU_ID", type=int, default=0)

    args = parser.parse_args()
    args.device = f'cuda:{args.GPU_ID}' if torch.cuda.is_available() else 'cpu'
    print(torch.cuda.get_device_name(args.GPU_ID))
    train_loader,batch_number,val_loader,train_transforms=setupdata(args)
    logs_name=f'./logs/{args.run_name}'	
    os.makedirs(logs_name,exist_ok=True)
    device=args.device
    num_train_timesteps=args.num_train_timesteps
    num_inference_steps=args.num_inference_steps
    lr=args.lr
    

    if args.mode == "train":
        my_paths=config_path(args.run_name)
        Diffuser=DiffusionModel(device,
                 num_train_timesteps, num_inference_steps,
                 lr, my_paths)
        Diffuser.train(args.n_epochs,args.val_interval, 
              args.pretrained_path, 
              train_loader,batch_number ,val_loader)
    elif args.mode == "checkdata":
        #checkdata(train_loader)
        checkdata(val_loader,train_transforms)
    elif args.mode == "test":
        current_time=time.time()
        my_paths=config_path(args.run_name)
        Diffuser=DiffusionModel(device,
                 num_train_timesteps, num_inference_steps,
                 lr, my_paths)
        pretrained_path = args.pretrained_path
        Diffuser._test_nifti(pretrained_path,val_loader,train_transforms)