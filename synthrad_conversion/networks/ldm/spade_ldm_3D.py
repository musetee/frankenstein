
import sys
import time
from abc import abstractmethod, ABC

import os
import numpy as np
import torch
import torch.nn
from generative.inferers import DiffusionInferer, LatentDiffusionInferer
from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.nets import SPADEDiffusionModelUNet, SPADEAutoencoderKL, PatchDiscriminator
from generative.networks.schedulers import DDPMScheduler
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn import L1Loss
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch.nn.functional as F
from monai.utils import first
from monai.losses.ssim_loss import SSIMLoss
from monai.transforms import SaveImage
from utils.evaluate import reverse_normalize_data
from dataprocesser.reconstruct_patch_to_volume import reconstruct_volume, initialize_collection

from synthrad_conversion.networks.model_registry import register_model

@register_model('spadeldm3d')
@register_model('spadeldm')  # 支持两个别名
class SpadeLDM3DRunner:
    def __init__(self, opt, paths, train_loader, val_loader):
        self.model = SPADEldm3D(opt, paths, train_loader, val_loader)
        self.opt = opt

    def train(self):
        self.model.train(
            n_epochs_autoencoder=self.opt.ldm.n_epochs_autoencoder,
            n_epochs_diffusion=self.opt.ldm.n_epochs_diffusion
        )

    def test(self):
        self.model.val()

    def test_ae(self):
        self.model.val_only_autoencoder()

def KL_loss(z_mu, z_sigma):
            kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3, 4])
            return torch.sum(kl_loss) / kl_loss.shape[0]
def convert_to_one_hot_3d(x, number_classes):
    if x.dtype != torch.long:
                x = x.long()
    # Create the one-hot encoded tensor
    b, c, h, w, d = x.size()
    one_hot = torch.zeros(b, number_classes, h, w, d, device=x.device)
    one_hot.scatter_(1, x, 1)
    return one_hot

class SPADEldm3D:
    def __init__(self, opt, my_paths, train_loader, val_loader):
        self.device ='cuda' if torch.cuda.is_available() else 'cpu'  
        #self.device = torch.device(f'cuda:{opt.GPU_ID}' if torch.cuda.is_available() else 'cpu')
        self.opt = opt
        self.my_paths=my_paths
        model_path = os.path.join(my_paths['saved_model_folder'])  
        output_path = os.path.join(my_paths['saved_img_folder'])   
        self.model_path = model_path
        self.output_path = output_path
        self.output_dir=output_path
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.ckpt_idx=self.opt.ldm.ckpt_idx
        self.number_classes=119

        # initial the learning rates
        self.lr_autoencoder = 5e-6
        self.lr_discriminator = 5e-6
        self.lr_diffusion = 1e-5

        #model_path_old = '../LiverTumors/autoencoder_v2.2/'
        #network = torch.load(f'{model_path}/diffusion3.pt', map_location=self.device)
        #torch.save(network.state_dict(), f'{model_path}/diffusion3dict.pt')
        autoencoder = SPADEAutoencoderKL(
            label_nc=self.number_classes,
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            num_channels=self.opt.ldm.num_channels_ae, #(64, 128, 256, 512)
            latent_channels=self.opt.ldm.latent_channels, #32
            num_res_blocks=self.opt.ldm.num_res_blocks_ae,
            norm_num_groups=self.opt.ldm.norm_num_groups,
            attention_levels=self.opt.ldm.attention_levels_ae,
        )
        
        discriminator = PatchDiscriminator(spatial_dims=3, num_layers_d=3, num_channels=16, in_channels=1, out_channels=1)
        # parallel computing
        #discriminator = nn.DataParallel(discriminator,device_ids=[opt.GPU_ID])
        autoencoder = autoencoder.to(self.device)
        discriminator = discriminator.to(self.device)

        optimizer_g = torch.optim.Adam(params=autoencoder.parameters(), lr=self.lr_autoencoder)
        optimizer_d = torch.optim.Adam(params=discriminator.parameters(), lr=self.lr_discriminator)

        if self.opt.server=='helixMultiple':
            print('using DataParallel on server', self.opt.server)
            
            autoencoder = nn.DataParallel(autoencoder) #, device_ids = [0,1]
            discriminator = nn.DataParallel(discriminator)

            '''
            autoencoder = nn.parallel.DistributedDataParallel(autoencoder)
            discriminator = nn.parallel.DistributedDataParallel(discriminator)
            unet = nn.parallel.DistributedDataParallel(unet)
            '''
        ckpt_path = opt.ckpt_path 

        if ckpt_path is not None and os.path.exists(ckpt_path):
            ckpt_path = os.path.join(opt.ckpt_path,"saved_models")
            ae = torch.load(f'{ckpt_path}/autoencoder_{self.ckpt_idx}.pt')
            d = torch.load(f'{ckpt_path}/discriminator_{self.ckpt_idx}.pt')

            try:
                opt_g = torch.load(f'{ckpt_path}/optimizer_g_{self.ckpt_idx}.pt')
                opt_d = torch.load(f'{ckpt_path}/optimizer_d_{self.ckpt_idx}.pt')
            except:
                print('Optimizer state_dict not found, training from scratch')
            try:
                # If we load the checkpoints saved in the same mode last time, e.g. both in multiple-GPU mode, then we can simply load state_dict
                autoencoder.load_state_dict(ae)
                discriminator.load_state_dict(d)
                print(f'Loaded checkpoints in {ckpt_path}.')
            except RuntimeError as e_single:
                try:
                    # If we use single-GPU mode this time, but the checkpoints were saved in multiple-GPU mode, we need to load state_dict from the modules
                    autoencoder.module.load_state_dict(ae)
                    discriminator.module.load_state_dict(d)
                    print(f'Loaded checkpoints in {ckpt_path}.')
                except RuntimeError as e_multi:
                    # If both loading attempts fail, print an error message
                    print('Something wrong with loading checkpoints, please check.')
                    print(f'Single GPU load error: {e_single}')
                    print(f'DataParallel load error: {e_multi}')

            try: 
                optimizer_g.load_state_dict(opt_g)
                optimizer_d.load_state_dict(opt_d)
            except:
                print('Optimizer state_dict not found, training from scratch')
        else:
            # parallel computing
            # autoencoder = nn.DataParallel(autoencoder,device_ids=[opt.GPU_ID])
            print('No model found, training from scratch')
        
        l1_loss = L1Loss()
        ssim = SSIMLoss(spatial_dims=3)
        adv_loss = PatchAdversarialLoss(criterion="least_squares", no_activation_leastsq=False)
        loss_perceptual = PerceptualLoss(spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2)
        loss_perceptual.to(self.device)
        
        scheduler = DDPMScheduler(num_train_timesteps=self.opt.ldm.num_train_timesteps, schedule="scaled_linear_beta", beta_start=0.0015, #0.0015, 0.0195
                                beta_end=0.0195)

        check_data = first(self.train_loader)
        with torch.no_grad():
            with autocast(enabled=True):
                # parallel computing
                if self.opt.server=='helixMultiple': 
                    z = autoencoder.module.encode_stage_2_inputs(check_data["img"].to(self.device))
                else:
                    z = autoencoder.encode_stage_2_inputs(check_data["img"].to(self.device))
        print(f"Scaling factor set to {1 / torch.std(z)}")
        scale_factor = 1 / torch.std(z)

        inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)
        
        
        self.autoencoder = autoencoder
        self.discriminator = discriminator
        self.l1_loss = l1_loss
        self.ssim = ssim
        self.adv_loss = adv_loss
        self.loss_perceptual = loss_perceptual
        self.discriminator = discriminator
        self.inferer = inferer
        self.scheduler = scheduler
        self.ckpt_path = ckpt_path
        self.optimizer_g = optimizer_g
        self.optimizer_d = optimizer_d

        
    def train(self, n_epochs_autoencoder=50, n_epochs_diffusion = 50):
        autoencoder = self.autoencoder
        discriminator = self.discriminator
        l1_loss = self.l1_loss
        ssim = self.ssim
        adv_loss = self.adv_loss
        loss_perceptual = self.loss_perceptual
        optimizer_g = self.optimizer_g
        optimizer_d = self.optimizer_d


        adv_weight = 0.02
        perceptual_weight = 0.001
        kl_weight = 1e-6
        ssim_wegiht = 0.2

        

        autoencoder_warm_up_n_epochs = -1
        epoch_recon_loss_list = []
        epoch_gen_loss_list = []
        epoch_disc_loss_list = []
        val_recon_epoch_loss_list = []
        intermediary_images = []
        n_example_images = 4

        data_loader = self.train_loader
        
        for epoch in range(n_epochs_autoencoder):
            autoencoder.train()
            discriminator.train()
            epoch_loss = 0
            gen_epoch_loss = 0
            disc_epoch_loss = 0
            progress_bar = tqdm(enumerate(data_loader), ncols=110) ##, total=len(data_loader)
            progress_bar.set_description(f"Epoch {epoch}")
            for step, batch in progress_bar:
                images = batch["img"].to(self.device) 
                seg = batch['seg'].to(self.device)

                # convert seg to one-hot code
                seg_one_hot = convert_to_one_hot_3d(seg, number_classes=self.number_classes)

                if step == 0 and epoch == 0: # print the first image in the first batch, just for debugging
                    print('input seg size:', seg.shape)
                    print('input one hot seg size:', seg_one_hot.shape)
                seg = seg_one_hot

                # Generator part
                optimizer_g.zero_grad(set_to_none=True)
                #print(images.device, seg.device, autoencoder.device)
                reconstruction, z_mu, z_sigma = autoencoder(images, seg)

                #print('seg shape', seg.shape, 'img shape', images.shape)
                #print('reconstruction shape', reconstruction.shape)
                kl_loss = KL_loss(z_mu, z_sigma)
                recons_loss = l1_loss(reconstruction.float(), images.float())
                ssim_loss = ssim(reconstruction.float(), images.float()) * ssim_wegiht
                p_loss = loss_perceptual(reconstruction.float(), images.float())
                loss_g = recons_loss + kl_weight * kl_loss + perceptual_weight * p_loss + ssim_loss


                if epoch > autoencoder_warm_up_n_epochs:
                    logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                    generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                    loss_g += adv_weight * generator_loss

                loss_g.backward()
                optimizer_g.step()


                if epoch > autoencoder_warm_up_n_epochs:
                    # Discriminator part
                    optimizer_d.zero_grad(set_to_none=True)
                    logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                    loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                    logits_real = discriminator(images.contiguous().detach())[-1]
                    loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                    discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

                    loss_d = adv_weight * discriminator_loss

                    loss_d.backward()
                    optimizer_d.step()

                epoch_loss += recons_loss.item()
                if epoch > autoencoder_warm_up_n_epochs:
                    gen_epoch_loss += generator_loss.item()
                    disc_epoch_loss += discriminator_loss.item()

                progress_bar.set_postfix(
                    {
                        "recons_loss": epoch_loss / (step + 1),
                        "gen_loss": gen_epoch_loss / (step + 1),
                        "disc_loss": disc_epoch_loss / (step + 1),
                    }
                )
            
            printout=(
                "[AE Epoch %d/%d] [recons_loss: %f] [gen_loss: %f] [disc_loss: %f]\n"
                % (
                    epoch,
                    n_epochs_autoencoder,
                    epoch_loss / (step + 1), 
                    gen_epoch_loss / (step + 1),   
                    disc_epoch_loss / (step + 1),              
                )
            )
            self.save_training_log(printout)

            epoch_recon_loss_list.append(epoch_loss / (step + 1))
            epoch_gen_loss_list.append(gen_epoch_loss / (step + 1))
            epoch_disc_loss_list.append(disc_epoch_loss / (step + 1))

            self.reconstruction=reconstruction
            self.epoch_recon_loss_list=epoch_recon_loss_list
            self.epoch_gen_loss_list=epoch_gen_loss_list
            self.epoch_disc_loss_list=epoch_disc_loss_list
            self.plot_ae_learning_curves()

            torch.save(autoencoder.state_dict(), f'{self.model_path}/autoencoder_{epoch % 5}.pt')
            torch.save(optimizer_g.state_dict(), f'{self.model_path}/optimizer_g_{epoch % 5}.pt')
            torch.save(discriminator.state_dict(), f'{self.model_path}/discriminator_{epoch % 5}.pt')
            torch.save(optimizer_d.state_dict(), f'{self.model_path}/optimizer_d_{epoch % 5}.pt')


        #self.reconstruction=reconstruction

        del discriminator
        del loss_perceptual
        torch.cuda.empty_cache()
        
        unet = SPADEDiffusionModelUNet(
            spatial_dims=3,
            in_channels=self.opt.ldm.latent_channels,
            out_channels=self.opt.ldm.latent_channels,
            num_res_blocks=self.opt.ldm.num_res_blocks_diff, # 4
            label_nc=self.number_classes,
            num_channels=self.opt.ldm.num_channels_diff,
            attention_levels=self.opt.ldm.attention_levels_diff,
            norm_num_groups=self.opt.ldm.norm_num_groups,
            cross_attention_dim=self.opt.ldm.cross_attention_dim_diff,
            with_conditioning=True,
            resblock_updown=True
        )
        unet=unet.to(self.device)
        optimizer_diff = torch.optim.Adam(params=unet.parameters(), lr=self.lr_diffusion)
        
        ckpt_path = self.ckpt_path
        if ckpt_path is not None and os.path.exists(ckpt_path) and not self.opt.ldm.train_new_different_diff:
            d = torch.load(f'{ckpt_path}/unet_{self.ckpt_idx}.pt', map_location=self.device)
            unet.load_state_dict(d)
            try:
                opt_diff = torch.load(f'{ckpt_path}/optimizer_diff_{self.ckpt_idx}.pt')
                optimizer_diff.load_state_dict(opt_diff)
            except:
                print('Optimizer state_dict not found, training from scratch')
        unet.to(self.device)

        '''
        scheduler = DDPMScheduler(num_train_timesteps=self.opt.ldm.num_train_timesteps, schedule="scaled_linear_beta", beta_start=0.0015, #0.0015, 0.0195
                                beta_end=0.0195)

        check_data = first(data_loader)
        with torch.no_grad():
            with autocast(enabled=True):
                # parallel computing
                # z = autoencoder.module.encode_stage_2_inputs(check_data["img"].to(self.device))
                z = autoencoder.encode_stage_2_inputs(check_data["img"].to(self.device))
        print(f"Scaling factor set to {1 / torch.std(z)}")
        scale_factor = 1 / torch.std(z)

        inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)'''

        scheduler = self.scheduler
        inferer = self.inferer
        
        epoch_loss_list = []
        autoencoder.eval()
        scaler = GradScaler()

        #first_batch = first(data_loader)
        #z = autoencoder.module.encode_stage_2_inputs(first_batch["img"].to(self.device))

        for epoch in range(n_epochs_diffusion):
            unet.train()
            epoch_loss = 0
            progress_bar = tqdm(enumerate(data_loader), ncols=70) #, total=len(data_loader)
            progress_bar.set_description(f"Epoch {epoch}")
            for step, batch in progress_bar:
                images = batch["img"].to(self.device)
                #if step == 0 or step == len(progress_bar) - 1:
                #    with torch.no_grad():
                #        z = autoencoder.module.encode_stage_2_inputs(images)
                seg = batch["seg"].to(self.device)
                # convert seg to one-hot code
                seg_one_hot = convert_to_one_hot_3d(seg, number_classes=self.number_classes)
                seg = seg_one_hot

                Aorta_diss = batch['Aorta_diss'].to(torch.float32).to(self.device)
                Aorta_diss=Aorta_diss.unsqueeze(-1).unsqueeze(-1)
                #print(Aorta_diss.shape)
                #print(seg.shape, seg.device)
                optimizer_diff.zero_grad(set_to_none=True)

                with autocast(enabled=True):
                    # parallel computing

                    if self.opt.server=='helixMultiple': 
                        z_mu, z_sigma = autoencoder.module.encode(images)
                        z = autoencoder.module.sampling(z_mu, z_sigma)
                    else:
                        z_mu, z_sigma = autoencoder.encode(images)
                        z = autoencoder.sampling(z_mu, z_sigma).to(self.device)

                    # Generate random noise
                    noise = torch.randn_like(z).to(self.device)
                    #print('input images size:', images.size())
                    #print('input latent size:', noise.size()) # ([2, 32, 32, 32, 4]) for 32
                    self.noise=noise
                    #print('\n input noise shape:', noise.shape)
                    # Create timesteps
                    timesteps = torch.randint(
                        0, inferer.scheduler.num_train_timesteps, (z.shape[0],), device=z.device
                    ).long()
                    
                    
                    #print("\n check input shape of inferer:", images.shape,seg.shape,noise.shape)

                    # Get model prediction
                    # parallel computing
                    if self.opt.server=='helixMultiple': 
                        noise_pred = inferer(
                            inputs=images, autoencoder_model=autoencoder.module, diffusion_model=unet,  #.module
                            noise=noise, timesteps=timesteps, seg=seg, condition=Aorta_diss
                        )
                    else:
                        
                        noise_pred = inferer(
                            inputs=images, autoencoder_model=autoencoder, diffusion_model=unet, 
                            noise=noise, timesteps=timesteps, seg=seg, condition=Aorta_diss
                        )
                    #test = torch.ones(2, 3, 128, 128, 128)
                    #print(noise_pred.float().shape)
                    #print(noise.float().shape)
                    loss = F.mse_loss(noise_pred.float(), noise.float())# + l1_loss(noise_pred.float(), noise.float())

                scaler.scale(loss).backward()
                scaler.step(optimizer_diff)
                scaler.update()

                epoch_loss += loss.item()

                progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})
            
            printout=(
                "[Diff Epoch %d/%d] [diff_loss: %f]\n"
                % (
                    epoch,
                    n_epochs_diffusion,
                    epoch_loss / (step + 1),              
                )
            )
            self.save_training_log(printout)
            epoch_loss_list.append(epoch_loss / (step + 1))
            torch.save(unet.state_dict(), f'{self.model_path}/unet_{epoch % 5}.pt')
            torch.save(optimizer_diff.state_dict(), f'{self.model_path}/optimizer_diff_{epoch % 5}.pt')

            self.unet = unet
            self.inferer = inferer
            self.scheduler = scheduler
            self.autoencoder = autoencoder
            self.epoch_loss_list=epoch_loss_list
            self.plot_diffusion_learning_curves()
            if (epoch + 1) % self.opt.train.val_epoch_interval == 0:
                self.val()

    def val_only_autoencoder(self):
        autoencoder = self.autoencoder
        autoencoder.eval()
        data_loader = self.val_loader
        si = SaveImage(output_dir=f'{self.output_path}',
                    separate_folder=False,
                    output_postfix=f'reconstructed',
                    resample=False)
        si_input = SaveImage(output_dir=f'{self.output_path}',
                    separate_folder=False,
                    output_postfix=f'input',
                    resample=False)
        new_initialization = True
        with torch.no_grad():
            for data in data_loader:
                images = data["img"].to(self.device)
                seg = data['seg'].to(self.device)
                seg_one_hot = convert_to_one_hot_3d(seg, number_classes=self.number_classes)
                seg = seg_one_hot
                if new_initialization:
                    print('new initialization for the reconstructed volume')
                    collected_patches_ground_truth, collected_coords, reconstructed_volume_ground_truth, count_volume = initialize_collection(data)
                    collected_patches_predicted, _, reconstructed_volume_predicted, _ = initialize_collection(data)
                reconstruction, _, _ = autoencoder(images, seg)
                reconstructed_image = reverse_normalize_data(reconstruction, mode=self.opt.dataset.normalize)
                images = reverse_normalize_data(images, mode=self.opt.dataset.normalize)
                collected_patches_predicted.append(reconstructed_image.detach().cpu())
                collected_patches_ground_truth.append(images.detach().cpu())
                collected_coords.append(data['patch_coords'])
                reconstructed_volume_predicted, count_volume = reconstruct_volume(collected_patches_predicted, collected_coords, reconstructed_volume_predicted, count_volume)
                reconstructed_volume_ground_truth, _ = reconstruct_volume(collected_patches_ground_truth, collected_coords, reconstructed_volume_ground_truth, None)
                finished_criteria = torch.all(count_volume == 1)
                if finished_criteria:
                    new_initialization = True
                    si(reconstructed_volume_predicted.unsqueeze(0), data['img'].meta)
                    si_input(reconstructed_volume_ground_truth.unsqueeze(0), data['img'].meta)
                else:
                    new_initialization = False



    def val(self):
        if self.opt.mode=='test':
            unet = SPADEDiffusionModelUNet(
                spatial_dims=3,
                in_channels=self.opt.ldm.latent_channels,
                out_channels=self.opt.ldm.latent_channels,
                num_res_blocks=self.opt.ldm.num_res_blocks_diff, # 4
                label_nc=self.number_classes,
                num_channels=self.opt.ldm.num_channels_diff,
                attention_levels=self.opt.ldm.attention_levels_diff,
                norm_num_groups=self.opt.ldm.norm_num_groups,
                cross_attention_dim=self.opt.ldm.cross_attention_dim_diff,
                with_conditioning=True,
                resblock_updown=True
            )
            unet=unet.to(self.device)
            ckpt_path = self.ckpt_path
            if ckpt_path is not None and os.path.exists(ckpt_path):
                d = torch.load(f'{ckpt_path}/unet_{self.ckpt_idx}.pt', map_location=self.device)
                unet.load_state_dict(d)
                #unet = nn.DataParallel(unet)
                unet.to(self.device)
        elif self.opt.mode=='train':
            unet=self.unet
        else:
            print('validation only for test or train mode')
        autoencoder=self.autoencoder
        inferer=self.inferer
        scheduler=self.scheduler
        
        
        val_loader = self.val_loader
        print("Validation")
        unet.eval()
        step=0
        
        si = SaveImage(output_dir=f'{self.output_path}',
                    separate_folder=False,
                    output_postfix=f'synthesized',
                    resample=False)
        si_input = SaveImage(output_dir=f'{self.output_path}',
                    separate_folder=False,
                    output_postfix=f'input',
                    resample=False)
        si_seg = SaveImage(output_dir=f'{self.output_path}',
                    separate_folder=False,
                    output_postfix=f'seg',
                    resample=False)
        
        new_initialization = True
        with torch.no_grad():
            for data in val_loader:
                step += 1
                image_v = data['img'].to(self.device)
                seg = data['seg'].to(self.device)
                patch_coords = data['patch_coords']
                # convert seg to one-hot code
                seg_one_hot = convert_to_one_hot_3d(seg, number_classes=self.number_classes)
                seg_orig = seg
                seg = seg_one_hot
                if new_initialization:
                    print('new initialization for the reconstructed volume')
                    collected_patches_ground_truth, collected_coords, reconstructed_volume_ground_truth, count_volume = initialize_collection(data)
                    collected_patches_predicted, _, reconstructed_volume_predicted, _ = initialize_collection(data)
                    collected_patches_seg, _, reconstructed_volume_seg, _ = initialize_collection(data)
                            
                if self.opt.ldm.manual_aorta_diss >= 0:
                    Aorta_diss_refer = data['Aorta_diss'].to(torch.float32).to(self.device)
                    #print(Aorta_diss_refer.dtype)
                    #print(Aorta_diss_refer.shape)
                    aorta_diss_length = len(Aorta_diss_refer)
                    #
                    Aorta_diss = torch.full((aorta_diss_length,), self.opt.ldm.manual_aorta_diss)
                    Aorta_diss = Aorta_diss.to(torch.float32).to(self.device)
                    print(f"set aorta dissertion as {self.opt.ldm.manual_aorta_diss} manually")
                else:
                    Aorta_diss = data['Aorta_diss'].to(torch.float32).to(self.device)
                Aorta_diss=Aorta_diss.unsqueeze(-1).unsqueeze(-1)
                with autocast(enabled=True):
                    if self.opt.server=='helixMultiple': 
                        z_mu, z_sigma = autoencoder.module.encode(image_v)
                        z = autoencoder.module.sampling(z_mu, z_sigma)
                        noise = torch.randn_like(z).to(self.device)
                        image = inferer.sample(input_noise=noise, diffusion_model=unet, scheduler=scheduler,
                                            autoencoder_model=autoencoder.module, seg=seg, conditioning=Aorta_diss)
                    else:
                        z_mu, z_sigma = autoencoder.encode(image_v)
                        z = autoencoder.sampling(z_mu, z_sigma)
                        noise = torch.randn_like(z).to(self.device)
                        image = inferer.sample(input_noise=noise, diffusion_model=unet, scheduler=scheduler,
                                            autoencoder_model=autoencoder, seg=seg, conditioning=Aorta_diss)
                    predicted_reverse = reverse_normalize_data(image, mode=self.opt.dataset.normalize)
                    ground_truth_image_reversed = reverse_normalize_data(image_v, mode=self.opt.dataset.normalize)
                    seg_reversed = seg_orig

                collected_patches_predicted.append(predicted_reverse.detach().cpu())
                collected_patches_ground_truth.append(ground_truth_image_reversed.detach().cpu())
                collected_patches_seg.append(seg_reversed.detach().cpu())
                collected_coords.append(patch_coords)
                reconstructed_volume_predicted, count_volume = reconstruct_volume(collected_patches_predicted, collected_coords, reconstructed_volume_predicted, count_volume)
                reconstructed_volume_ground_truth, _ = reconstruct_volume(collected_patches_ground_truth, collected_coords, reconstructed_volume_ground_truth, None)
                reconstructed_volume_seg, _ = reconstruct_volume(collected_patches_seg, collected_coords, reconstructed_volume_seg, None)

                finished_criteria = torch.all(count_volume == 1)
                if finished_criteria:
                    new_initialization = True
                    si(reconstructed_volume_predicted.unsqueeze(0), data['img'].meta) #, data['original_affine'][0], data['original_affine'][1]
                    si_input(reconstructed_volume_ground_truth.unsqueeze(0), data['img'].meta)
                    si_seg(reconstructed_volume_seg.unsqueeze(0), data['seg'].meta)
                else:
                    new_initialization = False
    def plot_ae_learning_curves(self):
        epoch_recon_loss_list=self.epoch_recon_loss_list
        epoch_gen_loss_list=self.epoch_gen_loss_list
        epoch_disc_loss_list=self.epoch_disc_loss_list
        
        plt.style.use("ggplot")
        plt.title("Learning Curves", fontsize=20)
        plt.plot(epoch_recon_loss_list, label="Reconstruction Loss", color="C0", linewidth=2.0)
        plt.yticks(fontsize=12)
        plt.xticks(fontsize=12)
        plt.xlabel("Epochs", fontsize=16)
        plt.ylabel("Loss", fontsize=16)
        plt.legend(prop={"size": 14})
        plt.savefig(f"{self.output_dir}/learning_curves.png")
        plt.close()
        #plt.show()

        plt.title("Adversarial Training Curves", fontsize=20)
        plt.plot(epoch_gen_loss_list, color="C0", linewidth=2.0, label="Generator")
        plt.plot(epoch_disc_loss_list, color="C1", linewidth=2.0, label="Discriminator")
        plt.yticks(fontsize=12)
        plt.xticks(fontsize=12)
        plt.xlabel("Epochs", fontsize=16)
        plt.ylabel("Loss", fontsize=16)
        plt.legend(prop={"size": 14})
        plt.savefig(f"{self.output_dir}/adversarial_training_curves.png")
        plt.close()
        #plt.show()
    
    def plot_reconstruction_example(self):
        reconstruction=self.reconstruction
        # ### Visualise reconstructions

        # Plot axial, coronal and sagittal slices of a training sample
        idx = 0
        channel = 0 
        img = reconstruction[idx, channel].detach().cpu().numpy()
        fig, axs = plt.subplots(nrows=1, ncols=3)
        for ax in axs:
            ax.axis("off")
        ax = axs[0]
        ax.imshow(img[..., img.shape[2] // 2], cmap="gray")
        ax = axs[1]
        ax.imshow(img[:, img.shape[1] // 2, ...], cmap="gray")
        ax = axs[2]
        ax.imshow(img[img.shape[0] // 2, ...], cmap="gray")
        plt.savefig(f"{self.output_dir}/reconstructions.png")
        plt.close()
        #plt.show()
    
    def plot_diffusion_learning_curves(self):
        epoch_loss_list=self.epoch_loss_list
        plt.plot(epoch_loss_list)
        plt.title("Learning Curves", fontsize=20)
        plt.plot(epoch_loss_list, label="diffusion_loss")
        plt.yticks(fontsize=12)
        plt.xticks(fontsize=12)
        plt.xlabel("Epochs", fontsize=16)
        plt.ylabel("Loss", fontsize=16)
        plt.legend(prop={"size": 14})
        plt.savefig(f"{self.output_dir}/diffusion_learning_curves.png")
        plt.close()
        #plt.show()

    def plot_diffusion_sampling_example(self):
        device=self.device
        num_inference_steps=1000
        # ### Plotting sampling example
        #
        # Finally, we generate an image with our LDM. For that, we will initialize a latent representation with just noise. Then, we will use the `unet` to perform 1000 denoising steps. In the last step, we decode the latent representation and plot the sampled image.

        # +
        autoencoder = self.autoencoder
        inferer = self.inferer
        scheduler = self.scheduler

        unet = self.unet
        autoencoder.eval()
        unet.eval()

        noise = torch.randn((1, 3, 24, 24, 16))
        noise = noise.to(device)
        scheduler.set_timesteps(num_inference_steps=num_inference_steps)
        synthetic_images = inferer.sample(
            input_noise=noise, autoencoder_model=autoencoder, diffusion_model=unet, scheduler=scheduler,
            conditioning=0
        )
        # -

        # ### Visualise synthetic data

        idx = 0
        channel = 0
        img = synthetic_images[idx, channel].detach().cpu().numpy()  # images
        fig, axs = plt.subplots(nrows=1, ncols=3)
        for ax in axs:
            ax.axis("off")
        ax = axs[0]
        ax.imshow(img[..., img.shape[2] // 2], cmap="gray")
        ax = axs[1]
        ax.imshow(img[:, img.shape[1] // 2, ...], cmap="gray")
        ax = axs[2]
        ax.imshow(img[img.shape[0] // 2, ...], cmap="gray")
        plt.savefig(f"{self.output_dir}/diffusion_sampling.png")
        plt.close()
    
    def save_training_log(self,printout):
        with open(self.my_paths["train_loss_file"], 'a') as f: # append mode
            f.write(printout)
        