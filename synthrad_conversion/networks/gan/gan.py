# define the pix2pix network, copyright from deeplearning.ai, GAN course
from torch import nn
import torch.nn.functional as F
import torch
from tqdm import tqdm
import os
import datetime
from torch.utils.tensorboard import SummaryWriter
import monai
import glob
import time
import os
import torch
from monai.transforms import ThresholdIntensity
from monai.inferers import SliceInferer, Inferer
from monai.transforms import SaveImage

from synthrad_conversion.networks.basefunc import EarlyStopping, LossTracker, print_min_max_mean_std_value
from synthrad_conversion.networks.gan.my_net_pix2pix_blocks import pix2pix_Discriminator
from synthrad_conversion.utils.evaluate import (
                            save_single_image,
                            arrange_images, 
                            arrange_histograms,
                            arrange_3_histograms,
                            calculate_mask_metrices, 
                            InferenceMetrics, 
                            InferenceLogger,
                            Postprocessfactory,
                            save_image_slice,
                        )
from synthrad_conversion.utils.loss import WeightedMSELoss, FocalLoss, EnhancedMSELoss, EnhancedWeightedMSELoss, DualEnhancedWeightedMSELoss
from synthrad_conversion.networks.results_eval import evaluate2dBatch
from dataprocesser.step3_build_patch_dataset import patch_2d_from_single_volume, decode_dataset_from_single_volume_batch

class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.config=config
        self.input_dim = config.input_dim
        self.real_dim = config.real_dim
        self.hidden_channels_D = config.disc.hidden_channels_D
        self.learning_rate_D = config.train.learning_rate_D
        self.model_name=config.model_name
        self.device=torch.device(
            f'cuda:{config.GPU_ID[0]}' if torch.cuda.is_available() else 'cpu')
        self.WINDOW_WIDTH=config.dataset.WINDOW_WIDTH
        self.WINDOW_LEVEL=config.dataset.WINDOW_LEVEL
        if config.model_name == 'monai_pix2pix' \
            or config.model_name == 'pix2pix'\
                or config.model_name == 'monai_pix2pix_att':
            if config.disc.type=='monaiDisc': 
                self.disc = monai.networks.nets.Discriminator(
                    in_shape=config.disc.in_shape,
                    channels=config.disc.channels, 
                    strides=config.disc.strides, 
                    kernel_size=config.disc.kernel_size, 
                    num_res_units=config.disc.num_res_units, 
                    act=config.disc.act, 
                    norm=config.disc.norm,  # INSTANCE
                    dropout=config.disc.dropout, 
                    bias=config.disc.bias, 
                    last_act=config.disc.last_act)
            elif config.disc.type=='pix2pixDisc':
                self.disc = pix2pix_Discriminator(
                    self.input_dim + self.real_dim, 
                    self.hidden_channels_D
                    ).to(self.device)
        else:
            '''self.disc = Discriminator(
                in_shape=(1, 1, 1),
                channels=(1, 1, 1, 1),
                strides=(1, 1, 1, 1),
                num_res_units=1,
                kernel_size=1,
            ).to(self.device)'''
            raise NotImplementedError
        print('The number of parameters of discriminator is: ', 
              sum(p.numel() for p in self.disc.parameters() if p.requires_grad))
    
    def forward(self, x, y):
        # min max normalization to x and y
        if self.config.disc.adv_criterion=='BCE':
            # the input value should be manually normalized for BCE loss
            min_val,max_val = 0, 1
            min_x,max_x = torch.min(x),torch.max(x)
            min_y,max_y = torch.min(y),torch.max(y)
            x = (x - min_x) / (max_x - min_x) * (max_val - min_val) + min_val
            y = (y - min_y) / (max_y - min_y) * (max_val - min_val) + min_val

        x = torch.cat([x, y], dim=1)
        disc_output=self.disc(x)
        return disc_output


def train_gan(gen: nn.Module,
              disc: nn.Module,
              gen_opt,
              disc_opt,
              train_loader,
              val_loader,
              train_patient_IDs,
              config,
              paths):
    total_start = time.time()
    loss_functions = {
    "MSE": nn.MSELoss(),
    "BCEwithlogits": nn.BCEWithLogitsLoss(), # should be default selected
    "SSIM": monai.losses.SSIMLoss(spatial_dims=2),
    "BCE": nn.BCELoss(),
    "L1": nn.L1Loss(),
    "DiceLoss": monai.losses.DiceLoss(sigmoid=True),
    "FocalLoss": FocalLoss(),
    "WeightedMSELoss": WeightedMSELoss(weight_bone=10),
    "EnhancedMSELoss": EnhancedMSELoss(power=2, scale=10.0),
    "EnhancedWeightedMSELoss": EnhancedWeightedMSELoss(weight_bone=5, power=2, scale=10.0),
    "DualEnhancedWeightedMSELoss": DualEnhancedWeightedMSELoss(weight_bone=10, weight_soft_tissue=5, power=2, scale=10.0),

    # Add more loss functions as needed
    }
    device=torch.device(
            f'cuda:{config.GPU_ID[0]}' if torch.cuda.is_available() else 'cpu')
    print('use Recon loss: ', config.gen.recon_criterion)
    print('use Adv loss: ', config.disc.adv_criterion)
    adv_criterion =  loss_functions[config.disc.adv_criterion]# Loss for GAN
    recon_criterion = loss_functions[config.gen.recon_criterion] # Loss for reconstruction
    pretrained =config.ckpt_path
    if pretrained is not None and os.path.exists(pretrained):
        latest_ckpt=pretrained
        loaded_state = torch.load(latest_ckpt)
        print(f'use pretrained model: {latest_ckpt}') 
        if 'epoch' in loaded_state:
            init_epoch=loaded_state["epoch"] # load or manually set
            print(f'continue training from epoch {init_epoch}') 
            #init_epoch = int(input('Enter epoch number: '))
        else:
            print('no epoch information in the checkpoint file')
            init_epoch = int(input('Enter epoch number: '))
        
        gen.load_state_dict(loaded_state["gen"])
        gen_opt.load_state_dict(loaded_state["gen_opt"])
        load_disc=True
        if load_disc:
            disc.load_state_dict(loaded_state["disc"])
            disc_opt.load_state_dict(loaded_state["disc_opt"])
        else:
            disc = disc.apply(my_weights_init)
    else:
        gen = gen.apply(my_weights_init)
        disc = disc.apply(my_weights_init)
        print(f'start new training') 
        init_epoch=0

    writer =  SummaryWriter(paths["tensorboard_log_dir"])
    writeTensorboard=config.train.writeTensorboard

    
    log_step = 0
    loss_tracker = LossTracker(track_discriminator=True)
    early_stopping = EarlyStopping(patience=config.train.earlystopping_patience, min_delta=config.train.earlystopping_delta) # 0.1% threshold for early stopping

    for continue_epoch in range(config.train.num_epochs):
        #train_patient_IDs = train_patient_IDs
        total_patients = len(set(train_patient_IDs)) # Unique number of patients
        processed_patients = []
        pbar = tqdm(total=total_patients, desc="Training", ncols=100) #, unit="patient"
        
        epoch=continue_epoch+init_epoch+1
        epoch_num_total=config.train.num_epochs+init_epoch

        gen.train()
        disc.train()
        step = 0
        for data in tqdm(train_loader):
            step+=1
            log_step+=1
            source_images = data["source"].to(device)
            real_images = data["target"].to(device)
            if log_step==1:
                print_min_max_mean_std_value("target", real_images)
                print_min_max_mean_std_value("source", source_images)
                
            if config.gen.recon_criterion == "EnhancedWeightedMSELoss" or config.gen.recon_criterion == "DualEnhancedWeightedMSELoss":
                background=config.dataset.background
                bone_images = ThresholdIntensity(threshold=config.dataset.bone_min, above=True, cval=background)(real_images)
                bone_images = ThresholdIntensity(threshold=config.dataset.bone_max, above=False, cval=background)(bone_images)
                
                bone_labels = ThresholdIntensity(threshold=config.dataset.bone_min, above=True, cval=0)(bone_images)
                bone_labels = ThresholdIntensity(threshold=config.dataset.bone_min, above=False, cval=1)(bone_labels)

            #real_images=real_images/config.dataset.scalefactor
            
            
            # -----------------
            #  Train Generator
            # -----------------
            gen_opt.zero_grad()
            # Generate a batch of images
            if config.gen.in_channels_G == 2:
                mask_images = data["mask"].to(device)
                input_images = torch.cat((source_images, mask_images), dim=1)
            elif config.gen.in_channels_G == 1:
                input_images = source_images
            fake = gen(input_images)
            disc_fake_hat = disc(fake, source_images)
            gen_adv_output = disc_fake_hat
            gen_adv_loss = adv_criterion(disc_fake_hat, torch.ones_like(disc_fake_hat))

            '''
            if config.gen.recon_criterion == "EnhancedWeightedMSELoss":
                gen_rec_loss = recon_criterion(fake, real_images, bone_labels)
            else:
                gen_rec_loss = recon_criterion(fake, real_images)
                '''
            
            gen_rec_loss = recon_criterion(fake, real_images)
            gen_loss = gen_adv_loss + config.gen.lambda_recon * gen_rec_loss
            gen_loss.backward()
            gen_opt.step()
            generator_loss = {'gen_loss': gen_loss.item(), 
                              'gen_adv_loss': gen_adv_loss.item(), 
                              'gen_rec_loss': gen_rec_loss.item(),
                              'gen_adv_output': torch.mean(gen_adv_output),}
            '''generator_loss=train_pix2pix_gen_onestep(gen, disc, gen_opt, 
                    source_images, real_images, 
                    adv_criterion, recon_criterion,
                    config.gen.lambda_recon,)'''
            
            if writeTensorboard:
                writer.add_scalar("Generator (U-Net) loss", generator_loss['gen_loss'], log_step)
                writer.add_scalar("Generator (U-Net) adversarial loss", generator_loss['gen_adv_loss'], log_step)
                writer.add_scalar("Generator (U-Net) reconstruction loss", generator_loss['gen_rec_loss'], log_step)

            if step % config.train.update_D_interval == 0: # update discriminator by interval
                # ---------------------
                #  Train Discriminator
                # ---------------------
                with torch.no_grad():
                    fake = gen(source_images).detach()
                disc_opt.zero_grad()
                disc_fake_hat = disc(fake.detach(), source_images) # Detach generator
                disc_fake_loss = adv_criterion(disc_fake_hat, torch.zeros_like(disc_fake_hat))
                disc_real_hat = disc(real_images, source_images)
                disc_real_loss = adv_criterion(disc_real_hat, torch.ones_like(disc_real_hat))
                disc_loss = (disc_fake_loss + disc_real_loss) / 2
                disc_loss.backward() # Update gradients
                disc_opt.step() # Update optimizer
                discriminator_loss= {'disc_loss': disc_loss.item(),
                        'disc_fake_loss':disc_fake_loss.item(), 
                        'disc_real_loss':disc_real_loss.item(), 
                        'disc_fake_output': torch.mean(disc_fake_hat),
                        'disc_real_output': torch.mean(disc_real_hat),}
                
                '''discriminator_loss, fake=train_pix2pix_disc_onestep(gen, disc, 
                        source_images, real_images, 
                        disc_opt,
                        adv_criterion)'''
                if writeTensorboard:
                    writer.add_scalar('Discriminator loss', discriminator_loss['disc_loss'], log_step)
                    writer.add_scalar('Discriminator fake loss', discriminator_loss['disc_fake_loss'], log_step)
                    writer.add_scalar('Discriminator real loss', discriminator_loss['disc_real_loss'], log_step)
                
                loss_tracker.update(generator_loss['gen_loss'], discriminator_loss['disc_loss'])
            else:
                loss_tracker.update(generator_loss['gen_loss'])

            if log_step % config.train.sample_interval == 0:
                img_folder=os.path.join(paths["saved_img_folder"], f"step_{log_step}", "img")
                output_path=paths["saved_img_folder"]
                hist_folder=os.path.join(paths["saved_img_folder"], f"step_{log_step}", "hist")
                imgformat='png'
                dpi=100
                mse_loss = gen_rec_loss
                evaluate2dBatch(
                            source_images, 
                            real_images, 
                            fake, 
                            step,
                            config.validation.evaluate_restore_transforms,
                            config.dataset.normalize, output_path,
                            epoch, img_folder, imgformat, dpi,
                            config.dataset.rotate, 
                            paths["train_metrics_file"], 
                            loss_tracker,
                            mse_loss.item(),
                            hist_folder, 
                            x_lower_limit=config.validation.x_lower_limit, 
                            x_upper_limit=config.validation.x_upper_limit,
                            y_lower_limit=config.validation.y_lower_limit, 
                            y_upper_limit=config.validation.y_upper_limit,
                            val_log_file=paths["val_log_file"],
                            val_log_conclusion_file=paths["val_log_file"].replace('val_log','val_conclusion_log'),
                            model_name=config.model_name,
                            savenifti=False,
                        )
            
            # If this patient hasn't been processed yet, add to set and update the progress bar
            patient_ID_batch = data['patient_ID']
            for patient_ID in patient_ID_batch:
                if patient_ID not in processed_patients:
                    processed_patients.append(patient_ID)
                    if step % config.train.update_D_interval == 0:
                        postfix={
                                'pID': f"{patient_ID}", 
                                'epoch': f"{epoch}/{epoch_num_total}", 
                                'gen loss': f"{(generator_loss['gen_loss']):.3f}",
                                'disc loss': f"{(discriminator_loss['disc_loss']):.3f}"
                                } 
                    else:
                        postfix={
                            'pID': f"{patient_ID}", 
                            'epoch': f"{epoch}/{epoch_num_total}", 
                            'gen loss': f"{(generator_loss['gen_loss']):.3f}",
                            'disc loss': None
                            }
                    pbar.set_postfix(postfix)
                    pbar.update(1)
        pbar.close()
            
        mean_gen_loss, mean_disc_loss = loss_tracker.get_mean_losses()
        #epoch_gen_losses, epoch_disc_losses = loss_tracker.get_epoch_losses()
        #mean_ssim, mean_psnr, mean_mae = loss_tracker.get_mean_metrics()
        epoch_print_out=(
                    "[Epoch %d/%d] [mean G loss: %f] [mean D loss: %f] \n"
                    % (
                        epoch,
                        epoch_num_total,
                        mean_gen_loss, 
                        mean_disc_loss,               
                    )
                )
        print(epoch_print_out)
        with open(paths["epoch_loss_file"], 'a') as f: # append mode
            f.write(epoch_print_out)
        loss_tracker.reset()
        
        if epoch % config.train.val_epoch_interval == 0:
            inference_gan(gen,
                  val_loader,
                  #untransformed_ds_val,
                  #val_transforms,
                  config,
                  paths,
                  epoch=epoch,
                  )
            saved_model_name=os.path.join(paths["saved_model_folder"], f"epoch_{epoch}.pth")
            torch.save({'epoch': epoch,
                'gen': gen.state_dict(),
                'gen_opt': gen_opt.state_dict(),
                'disc': disc.state_dict(),
                'disc_opt': disc_opt.state_dict()
            }, saved_model_name)

        
        early_stopping(mean_gen_loss)
        if early_stopping.best_loss_updated:
            saved_model_name = os.path.join(paths["saved_model_folder"], f"best.pt")
            torch.save({'epoch': epoch,
                'gen': gen.state_dict(),
                'gen_opt': gen_opt.state_dict(),
                'disc': disc.state_dict(),
                'disc_opt': disc_opt.state_dict()
            }, saved_model_name)
            
        if early_stopping.early_stop:
            print("Early stopping triggered")
            total_time = time.time() - total_start
            saved_model_name = os.path.join(paths["saved_model_folder"], f"model_earlystopp.pt")
            torch.save({'epoch': epoch,
                'gen': gen.state_dict(),
                'gen_opt': gen_opt.state_dict(),
                'disc': disc.state_dict(),
                'disc_opt': disc_opt.state_dict()
            }, saved_model_name)

            print(f"train completed, total time: {total_time}.")    
            break
    total_time = time.time() - total_start
    print(f"train completed, total time: {total_time}.") 

def evaluate_intermediate_training_results(source_images, real_images, fake):
    source = source_images[-1,0,:,:].squeeze().permute(1,0).cpu().detach()
    real=real_images[-1,0,:,:].squeeze().permute(1,0).cpu().detach()
    fake=fake[-1,0,:,:].squeeze().permute(1,0).cpu().detach()
    img_assemble = [
        source, 
        real, 
        #bone_labels[-1,0,:,:].squeeze().permute(1,0).cpu().detach(),
        fake,
        ]
    metrics=calculate_mask_metrices(img_assemble[-1]*config.dataset.scalefactor, img_assemble[1]*config.dataset.scalefactor, None, 
                                    paths["train_metrics_file"], f"{epoch}/{epoch_num_total}, {step}", printoutput=True)
    ssim=metrics['ssim']
    psnr=metrics['psnr']
    mae=metrics['mae']
    loss_tracker.update_metrics(ssim, psnr, mae)

    print_output=("[Epoch %d/%d] [Batch %d] [D fake output: %f] [D real output: %f] [D loss: %f] [D fake loss: %f] [D real loss: %f] [G loss: %f] [G adv loss: %f] [G rec loss: %f] [G adv output: %f] \n "
        % (
            epoch,
            config.train.num_epochs,
            step,
            discriminator_loss["disc_fake_output"],
            discriminator_loss["disc_real_output"],
            discriminator_loss["disc_loss"],
            discriminator_loss["disc_fake_loss"],
            discriminator_loss["disc_real_loss"],
            generator_loss['gen_loss'],
            generator_loss['gen_adv_loss'],
            generator_loss['gen_rec_loss'],
            generator_loss['gen_adv_output'],
        ))
    print(print_output)
    with open(paths["train_loss_file"], 'a') as f: # append mode
        f.write(
                print_output
            )
    
    imgformat = 'jpg'
    dpi = 100
    titles = ['MRI', 'CT', 'mask', config.model_name]

    imgformat = 'jpg'
    dpi = 100
    arrange_images_with_mask(
                            img_assemble, 
                            titles,
                            saved_name=os.path.join(paths["saved_img_folder"], 
                                f"compare_epoch_{epoch}_{step}.{imgformat}"), 
                            imgformat=imgformat, dpi=dpi)
    
    arrange_3_histograms(img_assemble[0].numpy(), 
                            img_assemble[1].numpy(), 
                            img_assemble[-1].numpy(), 
                            saved_name=os.path.join(paths["saved_img_folder"], 
                            f"histograms_epoch_{epoch}_{step}.{imgformat}"))

def inference_gan(gen,
                  val_loader,
                  #untransformed_ds_val,
                  #val_transforms,
                  config,
                  paths,
                  epoch=0,
                  ):
    device=torch.device(
            f'cuda:{config.GPU_ID[0]}' if torch.cuda.is_available() else 'cpu')
    latest_ckpt=config.ckpt_path
    if latest_ckpt is not None and os.path.exists(latest_ckpt):
            print(f'use pretrained model: {latest_ckpt}')     
            loaded_state = torch.load(latest_ckpt, map_location=device)
            try:
                init_epoch=loaded_state["epoch"] # load or manually set
                print('inference with model of epoch %d' % init_epoch)
                epoch=init_epoch
            except:
                print('no epoch information in the checkpoint file')
                epoch = epoch
            gen.load_state_dict(loaded_state["gen"])
    else:
        print('no pretrained model/validation during training')
    
    gen.eval()
    with torch.no_grad():
        '''
        metrics = InferenceMetrics()
        infer_logger = InferenceLogger(paths["saved_logs_folder"])
        postprocessFactory = Postprocessfactory(untransformed_ds_val,val_transforms)
        saved_folder=paths["saved_inference_folder"]
        '''
        eval_loss_tracker = LossTracker()

        '''for volume_batch in tqdm(val_loader):
            val_set_idx = 0

            debatched_volume = decode_dataset_from_single_volume_batch(volume_batch)
            volume_batch_dataset = monai.data.Dataset([debatched_volume])

            if config.gen.in_channels_G == 1:
                keys = ["source", "target"]
            elif config.gen.in_channels_G == 2:
                keys = ["source", "target", "mask"]
            val_loader_batch = patch_2d_from_single_volume(keys, volume_batch_dataset, config.dataset.batch_size, config.dataset.num_workers)

            for val_data in val_loader_batch:'''

        for val_set_idx, val_data in enumerate(val_loader):
            val_inputs, val_labels, val_masks = val_data["source"].to(device), val_data["target"].to(device), val_data["mask"].to(device)
            patient_ID_batch=val_data["patient_ID"] 
            print('validation input image shape: ',val_inputs.shape) if val_set_idx == 0 else print('')
            slice_inferer = SliceInferer(
                roi_size=(-1,-1),
                sw_batch_size=1,
                spatial_dim=2,  # Spatial dim to slice along is defined here
                device=device,
                padding_mode="replicate",
            )
            val_output = slice_inferer(val_inputs, gen)

            print('validation input image shape: ',val_output.shape) if val_set_idx == 0 else print('')

            val_inputs = val_inputs.detach().cpu() # source
            val_labels = val_labels.detach().cpu()
            val_output = val_output.detach().cpu() # fake
            val_masks = val_masks.detach().cpu() # mask
            
            mse_loss = F.mse_loss(val_output, val_labels)
            img_folder=os.path.join(paths["saved_img_folder"], f"epoch_{epoch}", "img")
            output_path=paths["saved_img_folder"]
            hist_folder=os.path.join(paths["saved_img_folder"], f"epoch_{epoch}", "hist")
            imgformat='png'
            dpi=100
            if len(val_output.shape)==4:
                    # B, N, H, W, D
                    evaluate2dBatch(
                        val_inputs, 
                        val_labels, 
                        val_output, 
                        patient_ID_batch,
                        val_set_idx,
                        config.dataset.val_batch_size,
                        config.validation.evaluate_restore_transforms,
                        config.dataset.normalize, output_path,
                        epoch, img_folder, imgformat, dpi,
                        config.dataset.rotate, 
                        paths["train_metrics_file"], 
                        eval_loss_tracker,
                        mse_loss.item(),
                        hist_folder, 
                        x_lower_limit=config.validation.x_lower_limit, 
                        x_upper_limit=config.validation.x_upper_limit,
                        y_lower_limit=config.validation.y_lower_limit, 
                        y_upper_limit=config.validation.y_upper_limit,
                        val_log_file=paths["val_log_file"],
                        val_log_conclusion_file=paths["val_log_file"].replace('val_log','val_conclusion_log'),
                        model_name=config.model_name,
                        save_nifti_Batch3D=True,
                        save_nifti_Slice2D=False,
                        save_png_images=False,
                    )
            elif len(val_output.shape)==5:
                # B, N, H, W, D -> N, H, W, D -> D, N, H, W (change D as the batch dimension so we can use evaluate2dBatch)
                evaluate_2d_slices = True
                if evaluate_2d_slices:
                    for batch_idx in range(val_output.shape[0]):
                        val_input_idx=val_inputs[batch_idx].permute(3, 0, 1, 2)
                        val_output_idx=val_output[batch_idx].permute(3, 0, 1, 2)
                        val_label_idx=val_labels[batch_idx].permute(3, 0, 1, 2)
                        evaluate2dBatch(
                            val_input_idx, 
                            val_label_idx, 
                            val_output_idx, 
                            patient_ID_batch,
                            batch_idx,
                            config.dataset.val_batch_size,
                            config.validation.evaluate_restore_transforms,
                            config.dataset.normalize, output_path,
                            epoch, img_folder, imgformat, dpi,
                            config.dataset.rotate, 
                            paths["train_metrics_file"], 
                            eval_loss_tracker,
                            mse_loss.item(),
                            hist_folder, 
                            x_lower_limit=config.validation.x_lower_limit, 
                            x_upper_limit=config.validation.x_upper_limit,
                            y_lower_limit=config.validation.y_lower_limit, 
                            y_upper_limit=config.validation.y_upper_limit,
                            val_log_file=paths["val_log_file"],
                            val_log_conclusion_file=paths["val_log_file"].replace('val_log','val_conclusion_log'),
                            model_name=config.model_name,
                            save_nifti_Batch3D=True,
                            save_nifti_Slice2D=True,
                            save_png_images=False,
                        )
                else:
                    si = SaveImage(output_dir=f'{output_path}',
                    separate_folder=False,
                    output_postfix=f'{patient_ID_batch}_epoch_{epoch}_synth_{val_set_idx}',
                    resample=False)
                    si_input = SaveImage(output_dir=f'{output_path}',
                                separate_folder=False,
                                output_postfix=f'{patient_ID_batch}_epoch_{epoch}_img_{val_set_idx}',
                                resample=False)
                    si_seg = SaveImage(output_dir=f'{output_path}',
                                separate_folder=False,
                                output_postfix=f'{patient_ID_batch}_epoch_{epoch}_seg_{val_set_idx}',
                                resample=False)
                    si(val_output[0])
                    si_input(val_labels[0])
                    si_seg(val_inputs[0])

            
            '''
            val_metrices_unreversed=calculate_mask_metrices(
                                                val_output[0,0,:,:,:], 
                                                val_labels[0,0,:,:,:],   
                                                 val_masks[0,0,:,:,:], 
                                                infer_logger.get_log_file_total_sets_path(epoch, unreversed=True),
                                                val_set_idx)
            '''
            '''
            unreversed_val_source=val_images.squeeze()
            unreversed_val_targets=val_labels.squeeze()
            unreversed_val_output=val_output.squeeze()
            # reverse steps to fit output
            val_output,val_masks = postprocessFactory.reverseTransform(val_output,val_labels,val_images,val_masks)
            val_output = postprocessFactory.reverseNormalization(val_output, config.dataset.normalize,val_set_idx)
            
            # postprocessing to output
            reversed_ct = val_output*config.dataset.scalefactor
            window_min = config.dataset.WINDOW_LEVEL-(config.dataset.WINDOW_WIDTH/2)
            reversed_ct = ThresholdIntensity(threshold=window_min,above=True, cval=config.dataset.background)(reversed_ct)

            input_ct = postprocessFactory.get_reverse_info()['CT_data'][val_set_idx]
            input_ct = ThresholdIntensity(threshold=window_min,above=True, cval=config.dataset.background)(input_ct)
            
            Original_MRI = postprocessFactory.get_reverse_info()['MRI_data'][val_set_idx]
            print("reversed_ct shape:",reversed_ct.shape, "input_ct shape:",input_ct.shape, "Original_MRI shape:",Original_MRI.shape)

            # calculate_val_metrices requires shape [H, W, D]     
            val_metrices_reversed=calculate_mask_metrices(reversed_ct.squeeze(), 
                                                input_ct.squeeze(), 
                                                val_masks.squeeze(),
                                                infer_logger.get_log_file_total_sets_path(epoch, unreversed=False),
                                                val_set_idx)
            
            file_name_prex=f'Inference_valset_{val_set_idx}_epoch_{epoch}'
            SaveImage(output_dir=saved_folder, output_postfix=file_name_prex,resample=True)(reversed_ct.detach().cpu())

            # rotate the image to plot right ordered images
            reversed_ct = postprocessFactory.reverseRotate(reversed_ct)
            input_ct = postprocessFactory.reverseRotate(input_ct)
            Original_MRI = postprocessFactory.reverseRotate(Original_MRI)
            masks = postprocessFactory.reverseRotate(val_masks)

            reversed_ct = postprocessFactory.resizeOutput(reversed_ct)
            input_ct = postprocessFactory.resizeOutput(input_ct)
            Original_MRI = postprocessFactory.resizeOutput(Original_MRI)
            masks = postprocessFactory.resizeOutput(masks)

            input_imgs=Original_MRI.squeeze()
            label_imgs=input_ct.squeeze()
            fake_imgs=reversed_ct.squeeze()
            mask_imgs=masks.squeeze()

            postprocessFactory.compareInfo(fake_imgs,val_set_idx)
            slice_num=input_imgs.shape[-1]
            slice_range ={"min":50,"max":52}

            save_folder_index= os.path.join(saved_folder,f"valset_{val_set_idx}")
            os.makedirs(save_folder_index, exist_ok=True)
            for slice_idx in range(slice_range["min"], slice_range["max"]):
                save_image_slice(input_imgs[:,:,slice_idx], 
                                label_imgs[:,:,slice_idx], 
                                fake_imgs[:,:,slice_idx], 
                                slice_idx, val_set_idx, epoch, 
                                config.model_name, save_folder_index,
                                unreversed=False,
                                x_lower_limit=-1100, 
                                x_upper_limit=3000, 
                                y_lower_limit=0, 
                                y_upper_limit=15000,
                                dpi=config.save_dpi,)
                save_image_slice(unreversed_val_source[:,:,slice_idx], 
                                unreversed_val_targets[:,:,slice_idx], 
                                unreversed_val_output[:,:,slice_idx], 
                                slice_idx, val_set_idx, epoch, 
                                config.model_name, save_folder_index, 
                                unreversed=True,
                                x_lower_limit=-1, 
                                x_upper_limit=3, 
                                y_lower_limit=0, 
                                y_upper_limit=15000,
                                dpi=config.save_dpi,)

            metrics.update(val_metrices_reversed['ssim'], val_metrices_reversed['mae'], val_metrices_reversed['psnr'])
        
        overall_ssim = metrics.get_averages()['ssim']
        overall_mae = metrics.get_averages()['mae']
        overall_psnr = metrics.get_averages()['psnr']
        print("overall val_ssim: %.4f" % (metrics.get_averages()['ssim']),
                "overall val_mae: %.4f" % (metrics.get_averages()['mae']),
                "overall val_psnr: %.4f" % (metrics.get_averages()['psnr']))
        with open(infer_logger.get_log_file_total_sets_path(epoch, unreversed=False), 'a') as f:
            f.write(f'overall mean metrics, SSIM: {overall_ssim}, MAE: {overall_mae}, PSNR: {overall_psnr}\n')
            '''

def inference_refinenet(gen, refinenet,
                                          val_loader,
                  untransformed_ds_val,
                  val_transforms,
                  config,
                  paths,
                  ):
    device=torch.device(
            f'cuda:{config.GPU_ID}' if torch.cuda.is_available() else 'cpu')
    latest_ckpt=config.ckpt_path
    if latest_ckpt is not None:
            print(f'use pretrained model: {latest_ckpt}')     
            loaded_state = torch.load(latest_ckpt,map_location=device)
            try:
                init_epoch=loaded_state["epoch"] # load or manually set
                print('inference with model of epoch %d' % init_epoch)
                epoch=init_epoch
            except:
                print('no epoch information in the checkpoint file')
                epoch = 0
            gen.load_state_dict(loaded_state["gen"])
            refinenet.load_state_dict(loaded_state["refineNet"])
    else:
        print('no pretrained model/validation during training')

    gen.eval()
    with torch.no_grad():
        metrics = InferenceMetrics()
        infer_logger = InferenceLogger(paths["saved_logs_folder"])
        postprocessFactory = Postprocessfactory(untransformed_ds_val,val_transforms)
        saved_folder=paths["saved_inference_folder"]

        for val_set_idx, val_data in enumerate(val_loader):
            val_images, val_labels,val_masks = val_data["source"].to(device), val_data["target"].to(device), val_data["mask"].to(device)
            slice_inferer = SliceInferer(
                roi_size=(-1,-1),
                sw_batch_size=5,
                spatial_dim=2,  # Spatial dim to slice along is defined here
                device=device,
                padding_mode="replicate",
            )
            val_output = slice_inferer(val_images, gen)
            # refine the output
            val_output = ThresholdIntensity(threshold=config.dataset.tissue_min, above=True, cval=config.dataset.background)(val_output)
            val_output = slice_inferer(val_output, refinenet)
            val_output = ThresholdIntensity(threshold=config.dataset.tissue_min, above=True, cval=config.dataset.background)(val_output)
            
            val_images = val_images.detach().cpu() # source
            val_labels = val_labels.detach().cpu()/config.dataset.scalefactor # target
            val_output = val_output.detach().cpu() # fake
            val_masks = val_masks.detach().cpu() # mask

            unreversed_val_source=val_images.squeeze()
            unreversed_val_targets=val_labels.squeeze()
            unreversed_val_output=val_output.squeeze()
            # reverse steps to fit output
            val_output,val_masks = postprocessFactory.reverseTransform(val_output,val_labels,val_images,val_masks)
            val_output = postprocessFactory.reverseNormalization(val_output, config.dataset.normalize,val_set_idx)
            
            # postprocessing to output
            reversed_ct = val_output*config.dataset.scalefactor
            reversed_ct = ThresholdIntensity(threshold=config.dataset.tissue_min,above=True, cval=config.dataset.background)(reversed_ct)

            input_ct = postprocessFactory.get_reverse_info()['CT_data'][val_set_idx]
            input_ct = ThresholdIntensity(threshold=config.dataset.tissue_min,above=True, cval=config.dataset.background)(input_ct)
            Original_MRI = postprocessFactory.get_reverse_info()['MRI_data'][val_set_idx]
            print("reversed_ct shape:",reversed_ct.shape, "input_ct shape:",input_ct.shape, "Original_MRI shape:",Original_MRI.shape)

            file_name_prex=f'Inference_valset_{val_set_idx}_epoch_{epoch}'
            SaveImage(output_dir=saved_folder, output_postfix=file_name_prex,resample=True)(reversed_ct.detach().cpu())

            # calculate_val_metrices requires shape [H, W, D]     
            val_metrices_reversed=calculate_mask_metrices(reversed_ct.squeeze(), 
                                                input_ct.squeeze(), 
                                                val_masks.squeeze(),
                                                infer_logger.get_log_file_total_sets_path(epoch, unreversed=False),
                                                val_set_idx)
            
            # rotate the image to plot right ordered images
            reversed_ct = postprocessFactory.reverseRotate(reversed_ct)
            input_ct = postprocessFactory.reverseRotate(input_ct)
            Original_MRI = postprocessFactory.reverseRotate(Original_MRI)

            reversed_ct = postprocessFactory.resizeOutput(reversed_ct)
            input_ct = postprocessFactory.resizeOutput(input_ct)
            Original_MRI = postprocessFactory.resizeOutput(Original_MRI)

            input_imgs=Original_MRI.squeeze()
            label_imgs=input_ct.squeeze()
            fake_imgs=reversed_ct.squeeze()
            postprocessFactory.compareInfo(fake_imgs,val_set_idx)
            slice_num=input_imgs.shape[-1]
            slice_range ={"min":50,"max":52}

            save_folder_index= os.path.join(saved_folder,f"valset_{val_set_idx}")
            os.makedirs(save_folder_index, exist_ok=True)
            for slice_idx in range(slice_range["min"], slice_range["max"]):
                save_image_slice(input_imgs[:,:,slice_idx], 
                                label_imgs[:,:,slice_idx], 
                                fake_imgs[:,:,slice_idx], 
                                slice_idx, val_set_idx, epoch, 
                                config.model_name, 
                                save_folder_index,
                                unreversed=False,
                                x_lower_limit=-1100, 
                                x_upper_limit=3000, 
                                y_lower_limit=0, 
                                y_upper_limit=15000,
                                dpi=config.save_dpi,
                                )
                save_image_slice(unreversed_val_source[:,:,slice_idx], 
                                unreversed_val_targets[:,:,slice_idx], 
                                unreversed_val_output[:,:,slice_idx], 
                                slice_idx, val_set_idx, epoch, 
                                config.model_name, save_folder_index, 
                                unreversed=True,
                                x_lower_limit=-1, 
                                x_upper_limit=3, 
                                y_lower_limit=0, 
                                y_upper_limit=15000,
                                dpi=config.save_dpi,)

            metrics.update(val_metrices_reversed['ssim'], val_metrices_reversed['mae'], val_metrices_reversed['psnr'])
        overall_ssim = metrics.get_averages()['ssim']
        overall_mae = metrics.get_averages()['mae']
        overall_psnr = metrics.get_averages()['psnr']
        print("overall val_ssim: %.4f" % (metrics.get_averages()['ssim']),
                "overall val_mae: %.4f" % (metrics.get_averages()['mae']),
                "overall val_psnr: %.4f" % (metrics.get_averages()['psnr']))
        with open(infer_logger.get_log_file_total_sets_path(epoch, unreversed=False), 'a') as f:
            f.write(f'overall mean metrics, SSIM: {overall_ssim}, MAE: {overall_mae}, PSNR: {overall_psnr}\n')

import matplotlib.pyplot as plt
def arrange_images_with_mask(img_assemble,
                   titles,
                   saved_name,
                   imgformat='jpg',
                   dpi = 500):
        #fig, axs = plt.subplots(int(len(img_assemble)/4), 4, figsize=(16, 5)) 
        fig, axs = plt.subplots(1, len(img_assemble), figsize=(16, 5)) 
        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                    hspace = 0, wspace = 0.1)
        plt.margins(0,0)
        axs = axs.flatten()
        for i in range(len(axs)):
            axs[i].imshow(img_assemble[i], cmap='gray')
            axs[i].set_title(titles[i])
            axs[i].axis('off')
        # save image as png
        fig.savefig(saved_name, format=f'{imgformat}', bbox_inches='tight', pad_inches=0, dpi=dpi)
        #plt.show()
        plt.close(fig)
    
def my_weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        # Initialize convolutional and linear layers
        nn.init.normal_(m.weight, 0.0, 0.04)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.BatchNorm2d):
        # Initialize batch normalization layers
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0.0)




