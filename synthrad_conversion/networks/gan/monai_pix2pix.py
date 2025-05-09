# runners/monai_pix2pix_runner.py
from synthrad_conversion.networks.model_registry import register_model
import torch
from synthrad_conversion.networks.unet.unet import UNET
from synthrad_conversion.networks.gan.gan import Discriminator, train_gan, inference_gan, inference_refinenet

@register_model('monai_pix2pix')
@register_model('monai_pix2pix_att')
class MonaiPix2PixRunner:
    def __init__(self, opt, paths, train_loader, val_loader, train_patient_IDs, device, **kwargs):
        self.opt = opt
        self.paths = paths
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_patient_IDs = train_patient_IDs
        self.device = device
        self.train_gan = train_gan
        self.inference_gan = inference_gan

        # 初始化生成器和判别器
        self.gen = UNET(opt).to(device)
        self.disc = Discriminator(opt).to(device)

        beta_1 = opt.train.beta_1
        beta_2 = opt.train.beta_2
        self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=opt.train.learning_rate_G, betas=(beta_1, beta_2))
        self.disc_opt = torch.optim.Adam(self.disc.parameters(), lr=opt.train.learning_rate_D, betas=(beta_1, beta_2))

        # 保存生成器的结构
        for name, module in self.gen.named_modules():
            with open(self.paths["model_layer_file"], 'a') as f:  # append模式
                f.write(f'{name}:\n{module}\n\n')

    def train(self):
        self.train_gan(self.gen, self.disc, self.gen_opt, self.disc_opt,
                       self.train_loader, self.val_loader, self.train_patient_IDs,
                       self.opt, self.paths)

    def test(self):
        self.inference_gan(self.gen, self.val_loader, self.opt, self.paths)
