import os
from dataprocesser.customized_transforms import CreateMaskTransformd, MergeMasksTransformd
IMG_EXTENSIONS = [
    #'.jpg', '.JPG', '.jpeg', '.JPEG',
    #'.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', 
    '.nrrd', '.nii.gz'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
import torch
from dataprocesser.list_dataset_synthrad_seg import synthrad_seg_loader
from dataprocesser.list_dataset_Anish_seg import anish_seg_loader
from dataprocesser.list_dataset_base import BaseDataLoader

