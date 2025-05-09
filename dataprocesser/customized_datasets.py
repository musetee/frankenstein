import torch.utils.data as data
import nibabel as nib
import torch
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm

VERBOSE = False


def volume_slicer(volume_tensor, transform, all_slices=None):
    # Convert numpy array to PyTorch tensor
    # Note: You might need to add channel dimension or perform other adjustments
    volume_tensor = volume_tensor.permute(2, 1, 0) # [H, W, D] -> [D, H, W]
    volume_tensor = volume_tensor.unsqueeze(1)  # Add channel dimension [D, H, W] -> [D, 1, H, W]
    if transform is not None:
        volume_tensor = transform(volume_tensor)

    #print('stacking volume tensor:',volume_tensor.shape)
    if all_slices is None:
        all_slices = volume_tensor
    else:
        all_slices = torch.cat((all_slices, volume_tensor), 0)
    return all_slices

def infinite_loader(loader):
    """Yield batches indefinitely from a DataLoader."""
    while True:
        for batch in loader:
            yield batch
        # This explicitly resets the iterator
        loader.dataset.reset()
        
class csvDataset_3D(Dataset):
    def __init__(self, csv_file, transform=None, load_patient_number=1):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_frame = pd.read_csv(csv_file)
        # control the length of the dataset
        self.data_frame = self.data_frame[:load_patient_number]
        self.transform = transform
        
    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.data_frame.iloc[idx, -1]
        image = nib.load(img_path).get_fdata()
        image = torch.tensor(image, dtype=torch.float32)
        
        # Example: Using the 'Aorta_diss' column as a label
        label = self.data_frame.iloc[idx, -3]
        #label = torch.tensor(label, dtype=torch.float32)
        
        # If more processing is needed (e.g., normalization, adding channel dimension), do it here
        image = image.unsqueeze(0)  # Add channel dimension if it's a single channel image
        
        sample = {'image': image, 'label': label}
        
        return sample
    
class csvDataset_2D(Dataset):
    def __init__(self, csv_file, transform=None, load_patient_number=1):
        self.csv_file = csv_file
        self.transform = transform
        self.load_patient_number = load_patient_number
        self.data_frame = pd.read_csv(csv_file)
        if len(self.data_frame) == 0:
            raise RuntimeError(f"Found 0 images in: {csv_file}")
        
        # Initialize dataset
        self.initialize_dataset()

    def initialize_dataset(self):
        print('Loading dataset...')
        self.data_frame = self.data_frame[:self.load_patient_number]
        all_slices = None
        all_labels = []
        
        for idx in tqdm(range(len(self.data_frame))):
            img_path = self.data_frame.iloc[idx, -1]
            volume = nib.load(img_path)
            volume_data = volume.get_fdata()  # Load as [H, W, D]
            volume_tensor = torch.tensor(volume_data, dtype=torch.float32)
            all_slices = volume_slicer(volume_tensor, self.transform, all_slices)  # -> [D, 1, H, W] and pile up all the slices
            label = self.data_frame.iloc[idx, -3]
            all_labels = all_labels + [label] * volume_tensor.shape[0]
        
        print('All stacked slices:', all_slices.shape)
        self.all_slices = all_slices
        self.all_labels = all_labels

    def __len__(self):
        return self.all_slices.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.all_slices[idx]
        label = self.all_labels[idx]
        sample = {'source': image, 'target': label}
        return sample

    def reset(self):
        print('Resetting dataset...')
        self.initialize_dataset()
