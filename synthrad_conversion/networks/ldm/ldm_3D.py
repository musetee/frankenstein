# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -

# # 3D Latent Diffusion Model
# In this tutorial, we will walk through the process of using the MONAI Generative Models package to generate synthetic data using Latent Diffusion Models (LDM)  [1, 2]. Specifically, we will focus on training an LDM to create synthetic brain images from the Brats dataset.
#
# [1] - Rombach et al. "High-Resolution Image Synthesis with Latent Diffusion Models" https://arxiv.org/abs/2112.10752
#
# [2] - Pinaya et al. "Brain imaging generation with latent diffusion models" https://arxiv.org/abs/2209.07162

# ### Set up imports

# +
import os
import shutil
import tempfile

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from monai import transforms
from monai.apps import DecathlonDataset
from monai.config import print_config
from monai.data import DataLoader
from monai.utils import first, set_determinism
from torch.cuda.amp import GradScaler, autocast
from torch.nn import L1Loss
from tqdm import tqdm

from generative.inferers import LatentDiffusionInferer
from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet, PatchDiscriminator
from generative.networks.schedulers import DDPMScheduler
from synthrad_conversion.networks.model_registry import register_model
@register_model('ldm3d')
class LDM3DRunner:
    def __init__(self, opt, paths, train_loader, val_loader):
        self.model = ldm_3D(opt, paths, train_loader, val_loader)
        self.opt = opt

    def train(self):
        self.model.train_autoencoder(n_epochs=self.opt.ldm.n_epochs_autoencoder)
        self.model.train_diffusion(n_epochs=self.opt.ldm.n_epochs_diffusion)

    def test(self):
        self.model.val()

    def test_ae(self):
        self.model.val_only_autoencoder()

# for reproducibility purposes set a seed
set_determinism(42)

# ### Setup a data directory and download dataset
# Specify a MONAI_DATA_DIRECTORY variable, where the data will be downloaded. If not specified a temporary directory will be used.

import torch.nn as nn
from torch.nn import Conv3d
class adapt_condition_shape(nn.Module):
    def __init__(self):
        super(adapt_condition_shape, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1)  # Output: [32, 64, 64, 64]
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1)  # Output: [64, 32, 32, 64]
        self.conv3 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1)  # Output: [64, 16, 16, 64]
        #self.fc = nn.Linear(256 * 8 * 8, 2048 * 64)  # Flatten and map to the final size

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        #print('shape of x',x.shape)
        batch, channel, height, width, depth = x.shape
        inner_dim = x.shape[1]
        x = x.permute(0, 2, 3, 4, 1).reshape(batch, height * width * depth, inner_dim)
        #print('shape of x',x.shape)
        #x = self.fc(x)
        #x = x.view(-1, 2048, 64)  # Reshape to the desired output shape
        return x
    
def KL_loss(z_mu, z_sigma):
    kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3, 4])
    return torch.sum(kl_loss) / kl_loss.shape[0]

class ldm_3D:   
    def __init__(self,opt,my_paths,train_loader, val_loader=None):
        self.opt=opt
        self.my_paths=my_paths
        self.device = torch.device(f'cuda:{opt.GPU_ID}' if torch.cuda.is_available() else 'cpu')
        self.train_loader=train_loader
        self.check_data  = first(train_loader)
        self.indicator_A = opt.dataset.indicator_A
        self.indicator_B = opt.dataset.indicator_B
        print(f'Image shape {self.check_data [self.indicator_A].shape}')
        self.val_epoch_interval=opt.train.val_epoch_interval

        if val_loader is None:
            self.val_loader=train_loader
        else:
            self.val_loader=val_loader

        device = self.device
        autoencoder = AutoencoderKL(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            num_channels=(32, 64, 64, 128),
            latent_channels=3,
            num_res_blocks=1,
            norm_num_groups=16,
            attention_levels=(False, False, False, True),
        )
        autoencoder.to(device)
        unet = DiffusionModelUNet(
            spatial_dims=3,
            in_channels=3,
            out_channels=3,
            num_res_blocks=1,
            num_channels=(32, 64, 64, 64),
            attention_levels=(False, True, True, True),
            num_head_channels=(0, 64, 64, 64),
            with_conditioning=True,
            cross_attention_dim=64, #self.opt.dataset.resized_size[-1]
        ).to(device)
        condition_adapter=adapt_condition_shape().to(device)

        num_train_timesteps=1000 # self.opt.ldm.num_train_timesteps
        scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps, 
                                  schedule="scaled_linear_beta", 
                                  beta_start=0.0015, 
                                  beta_end=0.0195)
        # -
        
        # ### Scaling factor
        #
        # As mentioned in Rombach et al. [1] Section 4.3.2 and D.1, the signal-to-noise ratio (induced by the scale of the latent space) can affect the results obtained with the LDM, if the standard deviation of the latent space distribution drifts too much from that of a Gaussian. For this reason, it is best practice to use a scaling factor to adapt this standard deviation.
        #
        # _Note: In case where the latent space is close to a Gaussian distribution, the scaling factor will be close to one, and the results will not differ from those obtained when it is not used._
        #

        # +
        with torch.no_grad():
            with autocast(enabled=True):
                z = autoencoder.encode_stage_2_inputs(self.check_data["target"].to(device))

        print(f"Scaling factor set to {1/torch.std(z)}")
        scale_factor = 1 / torch.std(z)
        inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)

        # load checkpoint
        if opt.ckpt_path is not None and os.path.exists(opt.ckpt_path):
            print('loading checkpoint from',os.path.join(opt.ckpt_path,"saved_models"))
            autoencoder.load_state_dict(torch.load(os.path.join(opt.ckpt_path,"saved_models", "autoencoder.pth")))
            unet.load_state_dict(torch.load(os.path.join(opt.ckpt_path,"saved_models", "diffusion.pth")))
            condition_adapter.load_state_dict(torch.load(os.path.join(opt.ckpt_path,"saved_models", "condition_adapter.pth")))
        else:
            print('No checkpoint found, training from scratch')
        self.unet=unet
        self.condition_adapter=condition_adapter
        self.autoencoder=autoencoder
        self.inferer=inferer
        self.scheduler=scheduler
        
    def train_autoencoder(self,n_epochs = 1):
        train_loader=self.train_loader
        device=self.device
        
        autoencoder=self.autoencoder


        discriminator = PatchDiscriminator(spatial_dims=3, num_layers_d=3, num_channels=32, in_channels=1, out_channels=1)
        discriminator.to(device)
        # -

        # ### Defining Losses
        #
        # We will also specify the perceptual and adversarial losses, including the involved networks, and the optimizers to use during the training process.

        # +
        l1_loss = L1Loss()
        adv_loss = PatchAdversarialLoss(criterion="least_squares")
        loss_perceptual = PerceptualLoss(spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2)
        loss_perceptual.to(device)

        adv_weight = 0.01
        perceptual_weight = 0.001
        kl_weight = 1e-6
        # -

        optimizer_g = torch.optim.Adam(params=autoencoder.parameters(), lr=1e-4)
        optimizer_d = torch.optim.Adam(params=discriminator.parameters(), lr=1e-4)

        
        autoencoder_warm_up_n_epochs = 5
        val_interval = self.val_epoch_interval
        epoch_recon_loss_list = []
        epoch_gen_loss_list = []
        epoch_disc_loss_list = []
        val_recon_epoch_loss_list = []
        intermediary_images = []
        n_example_images = 4

        for epoch in range(n_epochs):
            autoencoder.train()
            discriminator.train()
            epoch_loss = 0
            gen_epoch_loss = 0
            disc_epoch_loss = 0
            progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=110)
            progress_bar.set_description(f"Epoch {epoch}")

            for step, batch in progress_bar:
                images = batch["target"].to(device)  # choose only one of Brats channels

                # Generator part
                optimizer_g.zero_grad(set_to_none=True)
                reconstruction, z_mu, z_sigma = autoencoder(images)
                kl_loss = KL_loss(z_mu, z_sigma)

                recons_loss = l1_loss(reconstruction.float(), images.float())
                p_loss = loss_perceptual(reconstruction.float(), images.float())
                loss_g = recons_loss + kl_weight * kl_loss + perceptual_weight * p_loss

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
            if epoch % val_interval == 0:
            # save checkpoint
                torch.save(
                    autoencoder.state_dict(),
                    os.path.join(self.my_paths['saved_model_folder'], f"autoencoder.pth"), # _{epoch}
                )
                torch.save(
                    discriminator.state_dict(),
                    os.path.join(self.my_paths['saved_model_folder'], f"discriminator.pth"), # _{epoch}
                )

                    
            epoch_recon_loss_list.append(epoch_loss / (step + 1))
            epoch_gen_loss_list.append(gen_epoch_loss / (step + 1))
            epoch_disc_loss_list.append(disc_epoch_loss / (step + 1))
        
        self.epoch_recon_loss_list=epoch_recon_loss_list
        self.epoch_gen_loss_list=epoch_gen_loss_list
        self.epoch_disc_loss_list=epoch_disc_loss_list
        self.reconstruction=reconstruction
        self.autoencoder = autoencoder

        del discriminator
        del loss_perceptual
        torch.cuda.empty_cache()
        self.output_dir=self.my_paths['saved_img_folder']
        self.plot_ae_learning_curves()
        
    def plot_ae_learning_curves(self):
        epoch_recon_loss_list=self.epoch_recon_loss_list
        epoch_gen_loss_list=self.epoch_gen_loss_list
        epoch_disc_loss_list=self.epoch_disc_loss_list
        reconstruction=self.reconstruction
        plt.style.use("ggplot")
        plt.title("Learning Curves", fontsize=20)
        plt.plot(epoch_recon_loss_list)
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

    def train_diffusion(self, n_epochs = 1):
        autoencoder = self.autoencoder
        train_loader=self.train_loader
        inferer=self.inferer
        device = self.device
        scheduler=self.scheduler
        val_interval = self.val_epoch_interval
        # ## Diffusion Model
        #
        # ### Define diffusion model and scheduler
        #
        # In this section, we will define the diffusion model that will learn data distribution of the latent representation of the autoencoder. Together with the diffusion model, we define a beta scheduler responsible for defining the amount of noise tahat is added across the diffusion's model Markov chain.

        # +
        unet=self.unet
        condition_adapter=self.condition_adapter
        optimizer_diff = torch.optim.Adam(params=unet.parameters(), lr=1e-4)
        epoch_loss_list = []
        autoencoder.eval()
        scaler = GradScaler()

        first_batch = first(train_loader)
        z = autoencoder.encode_stage_2_inputs(first_batch["target"].to(device))

        for epoch in range(n_epochs):
            unet.train()
            epoch_loss = 0
            progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=70)
            progress_bar.set_description(f"Epoch {epoch}")
            for step, batch in progress_bar:
                condition_image = batch["source"].to(device)
                images = batch["target"].to(device)
                optimizer_diff.zero_grad(set_to_none=True)
                latent_condition = condition_adapter(condition_image) 
                #condition_adapter(condition_image) 
                #autoencoder.encode_stage_2_inputs(condition_image)
                #print('latent_condition shape',latent_condition.shape)

                with autocast(enabled=True):
                    # Generate random noise
                    noise = torch.randn_like(z).to(device)
                    # Create timesteps
                    timesteps = torch.randint(
                        0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
                    ).long()

                    # Get model prediction
                    noise_pred = inferer(
                        inputs=images, autoencoder_model=autoencoder, diffusion_model=unet, noise=noise, timesteps=timesteps,
                        condition=latent_condition
                    )

                    loss = F.mse_loss(noise_pred.float(), noise.float())

                scaler.scale(loss).backward()
                scaler.step(optimizer_diff)
                scaler.update()

                epoch_loss += loss.item()

                progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})
            epoch_loss_list.append(epoch_loss / (step + 1))
            
            if epoch % val_interval == 0:
                torch.save(
                    unet.state_dict(),
                    os.path.join(self.my_paths['saved_model_folder'], f"diffusion.pth"),  # _{epoch}
                )
                torch.save(
                    condition_adapter.state_dict(),
                    os.path.join(self.my_paths['saved_model_folder'], f"condition_adapter.pth"), # _{epoch}
                )
        self.unet=unet
        self.inferer=inferer
        self.scheduler=scheduler
        self.epoch_loss_list=epoch_loss_list

        self.plot_diffusion_learning_curves()
        self.plot_diffusion_sampling_example()
        
    def plot_diffusion_learning_curves(self):
        epoch_loss_list=self.epoch_loss_list
        plt.plot(epoch_loss_list)
        plt.title("Learning Curves", fontsize=20)
        plt.plot(epoch_loss_list)
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
            conditioning=None
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
        # ## Clean-up data

        #if directory is None:
        #    shutil.rmtree(root_dir)

    def validation(self):
        device=self.device
        unet=self.unet
        scheduler=self.scheduler
        inferer=self.inferer
        autoencoder=self.autoencoder
        condition_adapter=self.condition_adapter
        num_inference_steps=1000
        val_loader=self.val_loader
        self.output_dir=self.my_paths['saved_img_folder']

        first_batch = first(val_loader)
        z = autoencoder.encode_stage_2_inputs(first_batch["target"].to(device))

        unet.eval()
        autoencoder.eval()
        condition_adapter.eval()

        progress_bar = tqdm(enumerate(val_loader), total=len(val_loader), ncols=70)
        with torch.no_grad():
            for step, batch in progress_bar:
                condition_image = batch["source"].to(device)
                latent_condition = condition_adapter(condition_image) 
                noise = torch.randn_like(z).to(device)
                    # Create timesteps
                scheduler.set_timesteps(num_inference_steps=num_inference_steps)
                synthetic_images = inferer.sample(
                    input_noise=noise, autoencoder_model=autoencoder, diffusion_model=unet, scheduler=scheduler
                )
        # Decode latent representation of the intermediary images
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