import sys
sys.path.append('./networks/ddpm')

import os
import time
import torch
from monai.utils import first, set_determinism

from generative.inferers import DiffusionInferer
from generative.networks.schedulers import DDPMScheduler
from generative.networks.nets import SPADEDiffusionModelUNet
from synthrad_conversion.networks.model_registry import register_model
from synthrad_conversion.networks.ddpm.ddpm_mri2ct import DiffusionModel


@register_model('spade_ddpm2d_seg2med')
class SpadeDDPM2DSeg2MedRunner:
    def __init__(self, opt, paths, train_loader, val_loader, **kwargs):
        self.model = spadeDiffusionModel(opt, paths, train_loader=train_loader, val_loader=val_loader)
        self.opt = opt

    def train(self):
        self.model.train()

    def test(self):
        self.model.val()

VERBOSE = False
def convert_to_one_hot_2d(x, number_classes):
    if x.dtype != torch.long:
                x = x.long()
    # Create the one-hot encoded tensor
    b, c, h, w = x.size()
    one_hot = torch.zeros(b, number_classes, h, w, device=x.device)
    one_hot.scatter_(1, x, 1)
    return one_hot

class spadeDiffusionModel(DiffusionModel): #(nn.Module)
    def __init__(self,config,paths, train_loader, val_loader):
        #super(DiffusionModel, self).__init__()
        super().__init__(config,paths, train_loader, val_loader)
        self.number_classes=119

    def init_diffusion_model(self):
        patch_depth = self.config.dataset.patch_size[-1]
        spatial_dims = 2 if patch_depth==1 else 3
        self.model = SPADEDiffusionModelUNet(
            spatial_dims=spatial_dims,
            in_channels=1,
            out_channels=1,
            num_channels= self.config.ddpm.num_channels,  # (128, 256, 256), (32, 64, 64, 64) (64, 128, 256, 256)
            label_nc=self.number_classes,
            norm_num_groups=self.config.ddpm.norm_num_groups,
            attention_levels= self.config.ddpm.attention_levels,
            num_res_blocks= self.config.ddpm.num_res_units, # 2
            num_head_channels=self.config.ddpm.num_head_channels, # 256
            with_conditioning=True,
            cross_attention_dim = 1,
            resblock_updown=True, 
        )
        self.inferer = DiffusionInferer(self.scheduler)

    def prepare_model_input(self, source):
        return convert_to_one_hot_2d(source, number_classes=self.number_classes)
    
    def model_inferer_predict(self, targets, sources, model, noise, timesteps, condition):
        return self.inferer(inputs=targets, diffusion_model=model, noise=noise, timesteps=timesteps, condition=condition, seg=sources)
        
    def model_inferer_sample(self, Aorta_diss):
        if self.save_intermediates:
            image, intermediates = self.inferer.sample(input_noise=self.noise, 
                                    diffusion_model=self.model, 
                                    scheduler=self.scheduler,
                                    save_intermediates=self.save_intermediates,
                                    intermediate_steps=self.intermediate_steps,
                                    conditioning=Aorta_diss,
                                    seg=self.inputs_batch_prepared # different from original DDPM
                                    )
        else:
            image = self.inferer.sample(input_noise=self.noise, 
                                    diffusion_model=self.model, 
                                    scheduler=self.scheduler,
                                    save_intermediates=self.save_intermediates,
                                    intermediate_steps=self.intermediate_steps,
                                    conditioning=Aorta_diss,
                                    seg=self.inputs_batch_prepared)
            intermediates = None
        return image, intermediates