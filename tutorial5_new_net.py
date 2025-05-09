from generative.inferers import ControlNetDiffusionInferer, DiffusionInferer
from generative.networks.nets import DiffusionModelUNet, ControlNet
from generative.networks.schedulers import DDPMScheduler
import torch
from synthrad_conversion.networks.ddpm.diffusion_unet_modality import DiffusionModelUNet_modality
device = torch.device("cuda:1")

model = DiffusionModelUNet_modality(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    num_channels=(128, 256, 256),
    attention_levels=(False, True, True),
    num_res_blocks=1,
    num_head_channels=32,
    norm_num_groups=8,
    num_modalities=6,
    modality_emb_dim=32,
    with_conditioning=True,
    cross_attention_dim=32,
)
model.to(device)

# test the input and output of this model
model.eval() 
# Batch大小
batch_size = 2
H, W = 128, 128  # 2D图像大小

# 输入图像
x = torch.randn(batch_size, 1, H, W).to(device)  # (N, C=1, H, W)

# 时间步
timesteps = torch.randint(0, 1000, (batch_size,), dtype=torch.long).to(device)

# source modality labels 和 target modality labels
source_modality_labels = torch.randint(0, 6, (batch_size,1,), dtype=torch.long).to(device)
target_modality_labels = torch.randint(0, 6, (batch_size,1,), dtype=torch.long).to(device)

print('timesteps shape', timesteps.shape)
print('input label shape', source_modality_labels.shape)
print('input label', source_modality_labels)
print('target label', target_modality_labels)
# context (可以是None，因为我们在forward里自己根据modality生成)
context = None

# class_labels (如果你模型有开num_class_embeds的话，要准备；如果没开，就不用)
class_labels = None


with torch.no_grad():
    output = model(
        x=x,
        timesteps=timesteps,
        context=context,
        class_labels=class_labels,
        source_modality_labels=source_modality_labels,
        target_modality_labels=target_modality_labels,
    )

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")