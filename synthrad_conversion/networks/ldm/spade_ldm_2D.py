
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
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from monai.utils import first
from monai.losses.ssim_loss import SSIMLoss
from monai.transforms import SaveImage
from utils.evaluate import reverse_normalize_data
from dataprocesser.reconstruct_patch_to_volume import reconstruct_volume, initialize_collection

from utils.evaluate import (
    arrange_3_histograms,
    calculate_mask_metrices,
    save_single_image,
    reverse_normalize_data,)
from networks.ddpm.ddpm_mri2ct import DiffusionModel, LossTracker, arrange_images_assemble
VERBOSE = False

from synthrad_conversion.networks.model_registry import register_model
@register_model('spadeldm2d')
class SpadeLDM2DRunner:
    def __init__(self, opt, paths, train_loader, val_loader, **kwargs):
        self.model = SPADEldm2D(opt, paths, train_loader, val_loader)
        self.opt = opt

    def train(self):
        self.model.train()

    def test(self):
        self.model.val()

    def test_ae(self):
        self.model.val_only_autoencoder()

def KL_loss(z_mu, z_sigma):
            kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3])
            return torch.sum(kl_loss) / kl_loss.shape[0]
def convert_to_one_hot(x, number_classes):
    if x.dtype != torch.long:
                x = x.long()
    # Create the one-hot encoded tensor
    b, c, h, w = x.size()
    one_hot = torch.zeros(b, number_classes, h, w, device=x.device)
    one_hot.scatter_(1, x, 1)
    return one_hot

class SPADEldm2D(DiffusionModel):
    def __init__(self, config, paths, train_loader, val_loader):
        super().__init__(config,paths, train_loader, val_loader)
        self.ckpt_idx=self.config.ldm.ckpt_idx
        self.number_classes=119
        # initial the learning rates
        self.lr_autoencoder = 5e-6
        self.lr_discriminator = 5e-6
        self.lr_diffusion = 1e-5
        self.spatial_dims=2
        self.ckpt_path = self.config.ckpt_path 

    def init_autoencoder(self):
        autoencoder = SPADEAutoencoderKL(
            label_nc=self.number_classes,
            spatial_dims=self.spatial_dims,
            in_channels=1,
            out_channels=1,
            num_channels=self.config.ldm.num_channels_ae, #(64, 128, 256, 512)
            latent_channels=self.config.ldm.latent_channels, #32
            num_res_blocks=self.config.ldm.num_res_blocks_ae,
            norm_num_groups=self.config.ldm.norm_num_groups,
            attention_levels=self.config.ldm.attention_levels_ae,
        )
        
        discriminator = PatchDiscriminator(spatial_dims=self.spatial_dims, num_layers_d=3, num_channels=16, in_channels=1, out_channels=1)
        # parallel computing
        #discriminator = nn.DataParallel(discriminator,device_ids=[opt.GPU_ID])
        autoencoder = autoencoder.to(self.device)
        discriminator = discriminator.to(self.device)

        optimizer_g = torch.optim.Adam(params=autoencoder.parameters(), lr=self.lr_autoencoder)
        optimizer_d = torch.optim.Adam(params=discriminator.parameters(), lr=self.lr_discriminator)

        if self.config.server=='helixMultiple':
            print('using DataParallel on server', self.config.server)
            
            autoencoder = nn.DataParallel(autoencoder) #, device_ids = [0,1]
            discriminator = nn.DataParallel(discriminator)
        ckpt_path = self.config.ckpt_path 

        if ckpt_path is not None and os.path.exists(ckpt_path):
            ckpt_path = os.path.join(self.config.ckpt_path,"saved_models")
            ae = torch.load(f'{ckpt_path}/autoencoder_{self.ckpt_idx}.pt')
            d = torch.load(f'{ckpt_path}/discriminator_{self.ckpt_idx}.pt')
            
            try:
                opt_g = torch.load(f'{ckpt_path}/optimizer_g_{self.ckpt_idx}.pt')
                opt_d = torch.load(f'{ckpt_path}/optimizer_d_{self.ckpt_idx}.pt')
                optimizer_g.load_state_dict(opt_g)
                optimizer_d.load_state_dict(opt_d)
            except:
                print('Optimizer state_dict not found, training from scratch')
            
            try:
                step_information = torch.load(f'{ckpt_path}/step_information_{self.ckpt_idx}.pt')
                self.init_epoch=step_information["epoch"]
                self.init_step=step_information["global_step"]
                print(f'continue from, epoch {self.init_epoch}, global step {self.init_step}') 
            except:
                print('step informaiton not found')

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
        else:
            # parallel computing
            # autoencoder = nn.DataParallel(autoencoder,device_ids=[opt.GPU_ID])
            print('No model found, training from scratch')
        
        self.l1_loss = L1Loss()
        self.ssim = SSIMLoss(spatial_dims=self.spatial_dims)
        self.adv_loss = PatchAdversarialLoss(criterion="least_squares", no_activation_leastsq=False)
        self.loss_perceptual = PerceptualLoss(spatial_dims=self.spatial_dims, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2)
        self.loss_perceptual = self.loss_perceptual.to(self.device)
        self.autoencoder = autoencoder
        self.discriminator = discriminator
        self.optimizer_g = optimizer_g
        self.optimizer_d = optimizer_d

    def init_diffusion(self):
        check_data = first(self.train_loader)
        with torch.no_grad():
            with autocast(enabled=True):
                # parallel computing
                if self.config.server=='helixMultiple': 
                    z = self.autoencoder.module.encode_stage_2_inputs(check_data[self.config.dataset.indicator_B].to(self.device))
                else:
                    z = self.autoencoder.encode_stage_2_inputs(check_data[self.config.dataset.indicator_B].to(self.device))
        print(f"Scaling factor set to {1 / torch.std(z)}")
        scale_factor = 1 / torch.std(z)

        self.scheduler = DDPMScheduler(num_train_timesteps=self.config.ldm.num_train_timesteps, schedule="scaled_linear_beta", beta_start=0.0015, #0.0015, 0.0195
                                beta_end=0.0195)
        self.inferer = LatentDiffusionInferer(self.scheduler, scale_factor=scale_factor)

        unet = SPADEDiffusionModelUNet(
            spatial_dims=self.spatial_dims,
            in_channels=self.config.ldm.latent_channels,
            out_channels=self.config.ldm.latent_channels,
            num_res_blocks=self.config.ldm.num_res_blocks_diff, # 4
            label_nc=self.number_classes,
            num_channels=self.config.ldm.num_channels_diff,
            attention_levels=self.config.ldm.attention_levels_diff,
            norm_num_groups=self.config.ldm.norm_num_groups,
            cross_attention_dim=self.config.ldm.cross_attention_dim_diff,
            with_conditioning=True,
            resblock_updown=True
        )
        self.unet=unet.to(self.device)

        self.optimizer_diff = torch.optim.Adam(params=unet.parameters(), lr=self.lr_diffusion)

        
        if self.config.ckpt_path is not None and os.path.exists(self.config.ckpt_path) and not self.config.ldm.train_new_different_diff:
            ckpt_path = os.path.join(self.config.ckpt_path,"saved_models")
            unet_path = f'{ckpt_path}/unet_{self.ckpt_idx}.pt'
            if os.path.exists(unet_path):
                d = torch.load(unet_path, map_location=self.device)
                self.unet.load_state_dict(d)
                print('load diffusion unet from', unet_path)
                try:
                    opt_diff = torch.load(f'{ckpt_path}/optimizer_diff_{self.ckpt_idx}.pt')
                    self.optimizer_diff.load_state_dict(opt_diff)
                except:
                    print('Optimizer state_dict not found, training from scratch')

    def train(self):
        n_epochs_autoencoder = self.config.ldm.n_epochs_autoencoder
        n_epochs_diffusion = self.config.n_epochs_diffusion
        self.logger = SummaryWriter(self.saved_runs_name)
        self.train_autoencoder(n_epochs_autoencoder)
        self.train_diffusion(n_epochs_diffusion)
    
    def train_autoencoder(self,n_epochs_autoencoder):
        self.init_autoencoder()
        adv_weight = 0.02
        perceptual_weight = 0.001
        kl_weight = 1e-6
        ssim_wegiht = 0.2
        autoencoder_warm_up_n_epochs = -1
        self.epoch_recon_loss_list = []
        self.epoch_gen_loss_list = []
        self.epoch_disc_loss_list = []
        for epoch in range(n_epochs_autoencoder):
            self.autoencoder.train()
            self.discriminator.train()
            epoch_loss = 0
            gen_epoch_loss = 0
            disc_epoch_loss = 0
            progress_bar = tqdm(self.train_loader, ncols=110) ##, total=len(data_loader)
            progress_bar.set_description(f"Epoch {epoch}/{n_epochs_autoencoder}")
            step=0
            for batch in progress_bar:
                step+=1
                images = batch[self.config.dataset.indicator_B].to(self.device) 
                seg = batch[self.config.dataset.indicator_A].to(self.device)

                # convert seg to one-hot code
                seg_one_hot = convert_to_one_hot(seg, number_classes=self.number_classes)

                if step == 1 and epoch == 0: # print the first image in the first batch, just for debugging
                    print('seg shape', seg.shape, 'img shape', images.shape)
                    print('input one hot seg size:', seg_one_hot.shape)
                seg = seg_one_hot

                # Generator part
                self.optimizer_g.zero_grad(set_to_none=True)
                if VERBOSE:
                    print(images.device, seg.device, self.autoencoder.device)
                reconstruction, z_mu, z_sigma = self.autoencoder(images, seg)
                
                if VERBOSE:
                    print('reconstruction shape', reconstruction.shape)
                kl_loss = KL_loss(z_mu, z_sigma)
                recons_loss = self.l1_loss(reconstruction.float(), images.float())
                ssim_loss = self.ssim(reconstruction.float(), images.float()) * ssim_wegiht
                p_loss = self.loss_perceptual(reconstruction.float(), images.float())
                loss_g = recons_loss + kl_weight * kl_loss + perceptual_weight * p_loss + ssim_loss


                if epoch > autoencoder_warm_up_n_epochs:
                    logits_fake = self.discriminator(reconstruction.contiguous().float())[-1]
                    generator_loss = self.adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                    loss_g += adv_weight * generator_loss

                loss_g.backward()
                self.optimizer_g.step()


                if epoch > autoencoder_warm_up_n_epochs:
                    # Discriminator part
                    self.optimizer_d.zero_grad(set_to_none=True)
                    logits_fake = self.discriminator(reconstruction.contiguous().detach())[-1]
                    loss_d_fake = self.adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                    logits_real = self.discriminator(images.contiguous().detach())[-1]
                    loss_d_real = self.adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                    discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

                    loss_d = adv_weight * discriminator_loss

                    loss_d.backward()
                    self.optimizer_d.step()

                epoch_loss += recons_loss.item()
                if epoch > autoencoder_warm_up_n_epochs:
                    gen_epoch_loss += generator_loss.item()
                    disc_epoch_loss += discriminator_loss.item()

                progress_bar.set_postfix(
                    {
                        "recons_loss": epoch_loss / (step),
                        "gen_loss": gen_epoch_loss / (step),
                        "disc_loss": disc_epoch_loss / (step),
                    }
                )
            
            printout=(
                "[AE Epoch %d/%d] [recons_loss: %f] [gen_loss: %f] [disc_loss: %f]\n"
                % (
                    epoch,
                    n_epochs_autoencoder,
                    epoch_loss / (step), 
                    gen_epoch_loss / (step),   
                    disc_epoch_loss / (step),              
                )
            )
            self.save_training_log(printout)
            self.ae_epoch = epoch

            self.epoch_recon_loss_list.append(epoch_loss / (step))
            self.epoch_gen_loss_list.append(gen_epoch_loss / (step))
            self.epoch_disc_loss_list.append(disc_epoch_loss / (step))
            
            self.reconstruction=reconstruction
            self.plot_ae_learning_curves()
            self.plot_reconstruction_example()
            
            torch.save(self.autoencoder.state_dict(), f'{self.model_path}/autoencoder_{epoch % 5}.pt')
            torch.save(self.optimizer_g.state_dict(), f'{self.model_path}/optimizer_g_{epoch % 5}.pt')
            torch.save(self.discriminator.state_dict(), f'{self.model_path}/discriminator_{epoch % 5}.pt')
            torch.save(self.optimizer_d.state_dict(), f'{self.model_path}/optimizer_d_{epoch % 5}.pt')

        #self.reconstruction=reconstruction
        del self.discriminator
        del self.loss_perceptual
        torch.cuda.empty_cache()    
    def train_diffusion(self, n_epochs_diffusion):
        self.init_diffusion()
        epoch_loss_list = []
        self.autoencoder.eval()
        scaler = GradScaler()
        for epoch in range(n_epochs_diffusion):
            self.unet.train()
            epoch_loss = 0
            progress_bar = tqdm(self.train_loader, ncols=70) #, total=len(data_loader)
            progress_bar.set_description(f"Epoch {epoch}/{n_epochs_diffusion}")
            step = 0
            for batch in progress_bar:
                step += 1
                images = batch[self.config.dataset.indicator_B].to(self.device)
                seg = batch[self.config.dataset.indicator_A].to(self.device)
                # convert seg to one-hot code
                seg_one_hot = convert_to_one_hot(seg, number_classes=self.number_classes)
                seg = seg_one_hot

                Aorta_diss = batch['Aorta_diss'].to(torch.float32).to(self.device)
                Aorta_diss=Aorta_diss.unsqueeze(-1).unsqueeze(-1)
                if VERBOSE:
                    print(Aorta_diss.shape)
                    print(seg.shape, seg.device)
                self.optimizer_diff.zero_grad(set_to_none=True)

                with autocast(enabled=True):
                    # parallel computing
                    if self.config.server=='helixMultiple': 
                        z_mu, z_sigma = self.autoencoder.module.encode(images)
                        z = self.autoencoder.module.sampling(z_mu, z_sigma)
                    else:
                        z_mu, z_sigma = self.autoencoder.encode(images)
                        z = self.autoencoder.sampling(z_mu, z_sigma).to(self.device)

                    # Generate random noise
                    noise = torch.randn_like(z).to(self.device)
                    self.noise=noise
                    #print('\n input noise shape:', noise.shape)
                    # Create timesteps
                    timesteps = torch.randint(
                        0, self.inferer.scheduler.num_train_timesteps, (z.shape[0],), device=z.device
                    ).long()
                    
                    
                    #print("\n check input shape of inferer:", images.shape,seg.shape,noise.shape)

                    # Get model prediction
                    # parallel computing
                    if self.config.server=='helixMultiple': 
                        noise_pred = self.inferer(
                            inputs=images, autoencoder_model=self.autoencoder.module, diffusion_model=self.unet,  #.module
                            noise=noise, timesteps=timesteps, seg=seg, condition=Aorta_diss
                        )
                    else:
                        
                        noise_pred = self.inferer(
                            inputs=images, autoencoder_model=self.autoencoder, diffusion_model=self.unet, 
                            noise=noise, timesteps=timesteps, seg=seg, condition=Aorta_diss
                        )
                    #test = torch.ones(2, 3, 128, 128, 128)
                    #print(noise_pred.float().shape)
                    #print(noise.float().shape)
                    loss = F.mse_loss(noise_pred.float(), noise.float())# + l1_loss(noise_pred.float(), noise.float())

                scaler.scale(loss).backward()
                scaler.step(self.optimizer_diff)
                scaler.update()

                epoch_loss += loss.item()

                progress_bar.set_postfix({"loss": loss.item()})
            
            printout=(
                "[Diff Epoch %d/%d] [diff_loss: %f]\n"
                % (
                    epoch,
                    n_epochs_diffusion,
                    epoch_loss / (step),              
                )
            )
            self.save_training_log(printout)
            epoch_loss_list.append(epoch_loss / (step))
            torch.save(self.unet.state_dict(), f'{self.model_path}/unet_{epoch % 5}.pt')
            torch.save(self.optimizer_diff.state_dict(), f'{self.model_path}/optimizer_diff_{epoch % 5}.pt')

            self.epoch = epoch
            self.epoch_loss_list=epoch_loss_list
            self.plot_diffusion_learning_curves()
            if (epoch + 1) % self.config.train.val_epoch_interval == 0:
                self.val()
    
    def prepare_model_input(self, source):
        return convert_to_one_hot(source, number_classes=self.number_classes)
    
    def _sample(self, data):
        image_v = data[self.config.dataset.indicator_B].to(self.device)
        seg = data[self.config.dataset.indicator_A].to(self.device)
        # convert seg to one-hot code
        seg_one_hot = convert_to_one_hot(seg, number_classes=self.number_classes)
        seg = seg_one_hot
        self.scheduler.set_timesteps(num_inference_steps=self.config.ldm.num_inference_steps)
        if self.config.ldm.manual_aorta_diss >= 0:
            Aorta_diss_refer = data['Aorta_diss'].to(torch.float32).to(self.device)
            #print(Aorta_diss_refer.dtype)
            #print(Aorta_diss_refer.shape)
            aorta_diss_length = len(Aorta_diss_refer)
            #
            Aorta_diss = torch.full((aorta_diss_length,), self.config.ldm.manual_aorta_diss)
            Aorta_diss = Aorta_diss.to(torch.float32).to(self.device)
            print(f"set aorta dissertion as {self.config.ldm.manual_aorta_diss} manually")
        else:
            Aorta_diss = data['Aorta_diss'].to(torch.float32).to(self.device)
        Aorta_diss=Aorta_diss.unsqueeze(-1).unsqueeze(-1)
        with autocast(enabled=True):
            if self.config.server=='helixMultiple': 
                z_mu, z_sigma = self.autoencoder.module.encode(image_v)
                z = self.autoencoder.module.sampling(z_mu, z_sigma)
                noise = torch.randn_like(z).to(self.device)
                
                image = self.inferer.sample(input_noise=noise, diffusion_model=self.unet, scheduler=self.scheduler,
                autoencoder_model=self.autoencoder.module, seg=seg, conditioning=Aorta_diss)
            else:
                z_mu, z_sigma = self.autoencoder.encode(image_v)
                z = self.autoencoder.sampling(z_mu, z_sigma)
                noise = torch.randn_like(z).to(self.device)
                image = self.inferer.sample(input_noise=noise, diffusion_model=self.unet, scheduler=self.scheduler,
                                    autoencoder_model=self.autoencoder, seg=seg, conditioning=Aorta_diss)
            self.noise=noise
            self.mse_loss = F.mse_loss(image, image_v)
            return image
    def set_inference_parameters(self):
        self.manual_aorta_diss = 1
        self.save_intermediates=False
        self.intermediate_steps=50
        self.img_folder=os.path.join(self.paths["saved_img_folder"], f"epoch_{self.epoch}", "img")
        os.makedirs(self.img_folder, exist_ok=True)
        self.hist_folder=os.path.join(self.paths["saved_img_folder"], f"epoch_{self.epoch}", "hist")
        os.makedirs(self.hist_folder, exist_ok=True)
        self.imgformat = 'jpg'
        self.dpi = 100
    def val(self):
        if self.config.mode=='test':
            self.init_autoencoder()
            self.init_diffusion()
            self.epoch=0
        elif self.config.mode=='train':
            unet=self.unet
        else:
            print('validation only for test or train mode')
        print("Validation")
        self.set_inference_parameters()
        self.loss_tracker = LossTracker()
        self.unet.eval()
        step=0
        new_initialization = True
        with torch.no_grad():
            for batch in self.val_loader:
                if step >= self.config.train.sample_range_lower and step <= self.config.train.sample_range_upper: 
                    outputs_batch=self._sample(batch)
                    self.evaluate2dBatch(batch, outputs_batch, step)
                step += 1
    def val_only_autoencoder(self):
        self.init_autoencoder()
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
                images = data[self.config.dataset.indicator_B].to(self.device)
                seg = data[self.config.dataset.indicator_A].to(self.device)
                seg_one_hot = convert_to_one_hot(seg, number_classes=self.number_classes)
                seg = seg_one_hot
                if new_initialization:
                    print('new initialization for the reconstructed volume')
                    collected_patches_ground_truth, collected_coords, reconstructed_volume_ground_truth, count_volume = initialize_collection(data)
                    collected_patches_predicted, _, reconstructed_volume_predicted, _ = initialize_collection(data)
                reconstruction, _, _ = autoencoder(images, seg)
                reconstructed_image = reverse_normalize_data(reconstruction, mode=self.config.dataset.normalize)
                images = reverse_normalize_data(images, mode=self.config.dataset.normalize)
                collected_patches_predicted.append(reconstructed_image.detach().cpu())
                collected_patches_ground_truth.append(images.detach().cpu())
                collected_coords.append(data['patch_coords'])
                reconstructed_volume_predicted, count_volume = reconstruct_volume(collected_patches_predicted, collected_coords, reconstructed_volume_predicted, count_volume)
                reconstructed_volume_ground_truth, _ = reconstruct_volume(collected_patches_ground_truth, collected_coords, reconstructed_volume_ground_truth, None)
                finished_criteria = torch.all(count_volume == 1)
                if finished_criteria:
                    new_initialization = True
                    si(reconstructed_volume_predicted.unsqueeze(0), data[self.config.dataset.indicator_B].meta)
                    si_input(reconstructed_volume_ground_truth.unsqueeze(0), data[self.config.dataset.indicator_B].meta)
                else:
                    new_initialization = False
    def val_patchs_reverse(self):
        if self.config.mode=='test':
            unet = SPADEDiffusionModelUNet(
                spatial_dims=self.spatial_dims,
                in_channels=self.config.ldm.latent_channels,
                out_channels=self.config.ldm.latent_channels,
                num_res_blocks=self.config.ldm.num_res_blocks_diff, # 4
                label_nc=self.number_classes,
                num_channels=self.config.ldm.num_channels_diff,
                attention_levels=self.config.ldm.attention_levels_diff,
                norm_num_groups=self.config.ldm.norm_num_groups,
                cross_attention_dim=self.config.ldm.cross_attention_dim_diff,
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
        elif self.config.mode=='train':
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
                image_v = data[self.config.dataset.indicator_B].to(self.device)
                seg = data[self.config.dataset.indicator_A].to(self.device)
                patch_coords = data['patch_coords']
                # convert seg to one-hot code
                seg_one_hot = convert_to_one_hot(seg, number_classes=self.number_classes)
                seg_orig = seg
                seg = seg_one_hot
                if new_initialization:
                    print('new initialization for the reconstructed volume')
                    collected_patches_ground_truth, collected_coords, reconstructed_volume_ground_truth, count_volume = initialize_collection(data)
                    collected_patches_predicted, _, reconstructed_volume_predicted, _ = initialize_collection(data)
                    collected_patches_seg, _, reconstructed_volume_seg, _ = initialize_collection(data)
                            
                if self.config.ldm.manual_aorta_diss >= 0:
                    Aorta_diss_refer = data['Aorta_diss'].to(torch.float32).to(self.device)
                    #print(Aorta_diss_refer.dtype)
                    #print(Aorta_diss_refer.shape)
                    aorta_diss_length = len(Aorta_diss_refer)
                    #
                    Aorta_diss = torch.full((aorta_diss_length,), self.config.ldm.manual_aorta_diss)
                    Aorta_diss = Aorta_diss.to(torch.float32).to(self.device)
                    print(f"set aorta dissertion as {self.config.ldm.manual_aorta_diss} manually")
                else:
                    Aorta_diss = data['Aorta_diss'].to(torch.float32).to(self.device)
                Aorta_diss=Aorta_diss.unsqueeze(-1).unsqueeze(-1)
                with autocast(enabled=True):
                    if self.config.server=='helixMultiple': 
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
                    predicted_reverse = reverse_normalize_data(image, mode=self.config.dataset.normalize)
                    ground_truth_image_reversed = reverse_normalize_data(image_v, mode=self.config.dataset.normalize)
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
                    si(reconstructed_volume_predicted.unsqueeze(0), data[self.config.dataset.indicator_B].meta) #, data['original_affine'][0], data['original_affine'][1]
                    si_input(reconstructed_volume_ground_truth.unsqueeze(0), data[self.config.dataset.indicator_B].meta)
                    si_seg(reconstructed_volume_seg.unsqueeze(0), data[self.config.dataset.indicator_A].meta)
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
        batch_size = reconstruction.shape[0]
        # ### Visualise reconstructions
        dpi = 100
        # Plot axial, coronal and sagittal slices of a training sample
        channel = 0 
        for idx in range(batch_size):
            img = reconstruction[idx, channel].detach().cpu().numpy()
            plt.figure() #, figsize=(5, 4))
            plt.gca().set_axis_off()
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                        hspace = 0, wspace = 0)
            plt.margins(0,0)
            plt.imshow(img, cmap='gray')
            plt.savefig(f"{self.output_dir}/reconstructions_{self.ae_epoch}_{idx}.png"
                        , bbox_inches='tight', pad_inches=0, dpi=dpi)
            plt.close()

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
        with open(self.paths["train_loss_file"], 'a') as f: # append mode
            f.write(printout)
        