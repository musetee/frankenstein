"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
import argparse

import torch
import torch.nn as nn
import numpy as np
from torchvision import models
from scipy import linalg

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x): return x


class InceptionV3(nn.Module):
    def __init__(self):
        super().__init__()
        inception = models.inception_v3(pretrained=True)
        self.block1 = nn.Sequential(
            inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.block2 = nn.Sequential(
            inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.block3 = nn.Sequential(
            inception.Mixed_5b, inception.Mixed_5c,
            inception.Mixed_5d, inception.Mixed_6a,
            inception.Mixed_6b, inception.Mixed_6c,
            inception.Mixed_6d, inception.Mixed_6e)
        self.block4 = nn.Sequential(
            inception.Mixed_7a, inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d(output_size=(1, 1)))

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x.view(x.size(0), -1)


def frechet_distance(mu, cov, mu2, cov2):
    cc, _ = linalg.sqrtm(np.dot(cov, cov2), disp=False)
    dist = np.sum((mu -mu2)**2) + np.trace(cov + cov2 - 2*cc)
    return np.real(dist)


@torch.no_grad()
def calculate_fid_given_loaders(dataloader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inception = InceptionV3().eval().to(device)
    
    mu, cov = [], []
    actvs_synth = []
    actvs_target = []
    
    for synth, target in tqdm(dataloader, total=len(dataloader)):
        
        synth_actv = inception(synth.to(device))
        target_actv = inception(target.to(device))
        actvs_synth.append(synth_actv)
        actvs_target.append(target_actv)
    
    actvs_synth = torch.cat(actvs_synth, dim=0).cpu().detach().numpy()
    actvs_target = torch.cat(actvs_target, dim=0).cpu().detach().numpy()
    
    mu_synth = np.mean(actvs_synth, axis=0)
    cov_synth = np.cov(actvs_synth, rowvar=False)
    mu_target = np.mean(actvs_target, axis=0)
    cov_target = np.cov(actvs_target, rowvar=False)
    
    fid_value = frechet_distance(mu_synth, cov_synth, mu_target, cov_target)
    return fid_value

import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.nn.functional as F

def normalize_range(img):
    img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
    return img

def pad_to_desired_size(input_array, desired_size = (512, 512)):
    desired_height, desired_width = desired_size
    # Calculate padding for height and width
    height_padding = desired_height - input_array.shape[0]
    width_padding = desired_width - input_array.shape[1]

    # Split padding equally (top-bottom and left-right)
    padding_top = height_padding // 2
    padding_bottom = height_padding - padding_top
    padding_left = width_padding // 2
    padding_right = width_padding - padding_left

    # Apply padding
    # Padding format: ((pad_top, pad_bottom), (pad_left, pad_right))
    padded_array = np.pad(
        input_array,
        pad_width=((padding_top, padding_bottom), (padding_left, padding_right)),
        mode="constant",
        constant_values=0  # Padding with zeros
    )
    return padded_array

class NiftiDataset(Dataset):
    def __init__(self, folder_paths,transform=None):
        self.file_pairs = []
        self.transform = transform
        
        for folder in folder_paths:
            files = os.listdir(folder)
            synth_files = sorted([f for f in files if "synthesized" in f])
            target_files = sorted([f for f in files if "target" in f])

            

            for synth, target in zip(synth_files, target_files):
                synth_modified_filename = synth.replace("_seg", "")
                synth_modified_filename = synth_modified_filename.replace("synthesized", "")
                target_modified_filename = target.replace("target", "")
                assert synth_modified_filename == target_modified_filename, f"something wrong with file {synth_modified_filename} and {target_modified_filename}, from {synth} and {target}"

                self.file_pairs.append((os.path.join(folder, synth), os.path.join(folder, target)))
   
    
    def __len__(self):
        return len(self.file_pairs)
    
    def __getitem__(self, idx):
        synth_path, target_path = self.file_pairs[idx]
        #print(f"evaluate {synth_path} and {target_path}")
        synth_img = nib.load(synth_path).get_fdata()
        synth_img = normalize_range(synth_img)
        #synth_img = pad_to_desired_size(synth_img)
        synth_img = np.repeat(synth_img[np.newaxis, :, :], 3, axis=0)
        synth_img = torch.tensor(synth_img, dtype=torch.float32) #astype(np.float32)
        
        #synth_img = synth_img.unsqueeze(0)

        target_img = nib.load(target_path).get_fdata()
        target_img = normalize_range(target_img)
        #synth_img = pad_to_desired_size(synth_img)
        target_img = np.repeat(target_img[np.newaxis, :, :], 3, axis=0)
        target_img = torch.tensor(target_img, dtype=torch.float32) #astype(np.float32)

        # Optionally normalize and transform
        if self.transform:
            synth_img = self.transform(synth_img)
            target_img = self.transform(target_img)
        
        #print('transformed img shape:', (synth_img.shape))
        return synth_img,target_img  # torch.tensor(img, dtype=torch.float32)

def get_dataloaders(folder_paths, batch_size=8, transform=None):
    dataset = NiftiDataset(folder_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader


if __name__ == '__main__':
    from torchvision.transforms import Compose, Resize, Normalize, ToPILImage, ToTensor

    # Example usage
    DDPM_folder_paths = [
        #r"D:\Project\seg2med_Project\synthrad_results\results_all_eval\phase1_ct_anika_4_models\Infer_ddpm2d_seg2med_anika_512_all\saved_outputs\slice_output",
        r"D:\Project\seg2med_Project\synthrad_results\results_all_eval\phase1_ct_synthrad_anish_4_models\20241119_1142_Infer_ddpm2d_seg2med_anish_512\saved_outputs\slice_output",
        r"D:\Project\seg2med_Project\synthrad_results\results_all_eval\phase1_ct_synthrad_anish_4_models\20241119_0052_Infer_ddpm2d_seg2med_synthrad_512\saved_outputs\slice_output",
                         ]
    UNET_folder_paths = [
        #r"D:\Project\seg2med_Project\synthrad_results\results_all_eval\baseline_512\AttentionUnet_anika\saved_outputs",
        r"D:\Project\seg2med_Project\synthrad_results\results_all_eval\baseline_512\AttentionUnet_anish\saved_outputs",
        r"D:\Project\seg2med_Project\synthrad_results\results_all_eval\baseline_512\AttentionUnet_synthrad\saved_outputs",
    ]
    CYCLEGAN_folder_paths = [
        #r"D:\Project\seg2med_Project\synthrad_results\results_all_eval\baseline_512\cycle_gan_anika\saved_outputs",
        r"D:\Project\seg2med_Project\synthrad_results\results_all_eval\baseline_512\cycle_gan_anish\saved_outputs",
        r"D:\Project\seg2med_Project\synthrad_results\results_all_eval\baseline_512\cycle_gan_synthrad\saved_outputs",
    ]
    PIX2PIX_folder_paths = [
        #r"D:\Project\seg2med_Project\synthrad_results\results_all_eval\baseline_512\pix2pix_anika\saved_outputs",
        r"D:\Project\seg2med_Project\synthrad_results\results_all_eval\baseline_512\pix2pix_anish\saved_outputs",
        r"D:\Project\seg2med_Project\synthrad_results\results_all_eval\baseline_512\pix2pix_synthrad\saved_outputs",
    ]

    MR_folder_paths = [
        #r"D:\Project\seg2med_Project\synthrad_results\results_all_eval\phase1_mr\20241121_1055_Infer_ddpm2d_seg2med_mr_512\saved_outputs\slice_output",
        r"D:\Project\seg2med_Project\synthrad_results\results_all_eval\zhilin_results_testset10\slice_output",
    ]

    #folder_paths = [r"D:\Project\seg2med_Project\synthrad_results\results_all_eval\baseline_512\cycle_gan_synthrad\saved_outputs"]
    folder_paths = PIX2PIX_folder_paths

    


    transform = Compose([
        #ToPILImage(),
        Resize((410, 307)),  # Resize the volume slices or images
        #ToTensor(),
        #Normalize(mean=0.5, std=0.5)
    ])
    transform = None
    dataloader = get_dataloaders(folder_paths, batch_size=4, transform=transform)

    fid_value = calculate_fid_given_loaders(dataloader)
    print("FID Value:", fid_value)