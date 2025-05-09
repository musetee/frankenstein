from torch.utils.data import DataLoader, Dataset, random_split
import torch.utils.data as data
import os.path
import random
from torchvision import transforms
from PIL import Image
import torch
from PIL import ImageFile
#from utils.MattingLaplacian import compute_laplacian

import nibabel as nib
import numpy as np
import os
import csv
import pandas as pd
#from transformers import CLIPTokenizer
from tqdm import tqdm
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Disable OSError: image file is truncated

import os
import numpy as np

import torch

IMG_EXTENSIONS = [
    #'.jpg', '.JPG', '.jpeg', '.JPEG',
    #'.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', 
    '.nrrd', '.nii.gz'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)




from dataprocesser.customized_transforms import CreateMaskTransformd, MergeMasksTransformd
from dataprocesser.list_dataset_base import BaseDataLoader



