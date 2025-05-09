import SimpleITK as sitk
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
from typing import (
    TYPE_CHECKING,
    Any,
    BinaryIO,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
    overload,
)
import h5py
######################
# base dataset for ict-UNET
def Load_from_HDF5(file_path=None, file_format= 'hdf5'):
    if file_format == 'hdf5':
        # read hdf5
        # Replace 'your_file.h5' with the path to your HDF5 file
        # file_path = r'E:\LoDoPaB\ground_truth_train\ground_truth_train_000.hdf5'
        if file_path is None:
            raise ValueError("Please provide a file path to the HDF5 file.")
        # Open the HDF5 file and load the dataset
        with h5py.File(file_path, 'r') as f:
            dataset = f['data'][:]
    elif file_format == 'dicom':
        # read dicom 
        if file_path is None:
            raise ValueError("Please provide a file path to the DICOM file.")
        else:
            patient_folder = file_path
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(patient_folder)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        # Added a call to PermuteAxes to change the axes of the data
        #image = sitk.PermuteAxes(image, [2, 1, 0])
        dataset = sitk.GetArrayFromImage(image)
    return dataset

import json
VERBOSE = False
class basejsonDataset(Dataset):
    def __init__(self, json_path, mode='train', transform_list=None, do_normalize=False, slice_axis=2, use_saved_slice_info=False, slice_info_path="./data_table./slice_info.json"):
        """
        Args:
            file_ids (list): List of file ids to load data from.
            mode (str): 'train' or 'test'. Determines if augmentation is applied.
            transform_list (list of callable, optional): List of transforms to be applied on a sample.
            slice_axis (int): The axis along which to slice the 3D volumes (0, 1, or 2).
        """

        self.mode = mode
        self.transform_list = transform_list
        self.do_normalize = do_normalize
        self.json_path = json_path
        self.slice_axis = slice_axis
        self.data_info = self._load_json()
        self.slice_info_file=slice_info_path
        if use_saved_slice_info and os.path.exists(self.slice_info_file):
            self.slice_info = self._load_slice_info()
        else:
            self.slice_info = self._calculate_slice_info()
        
        
    def __len__(self):
        return len(self.slice_info)
    
    def _load_json(self):
        with open(self.json_path, 'r') as file:
            data_info = json.load(file)
        return data_info
    
    def _load_slice_info(self):
        with open(self.slice_info_file, 'r') as f:
            slice_info = json.load(f)
        return slice_info
    
    def _calculate_slice_info(self):
        slice_info = []
        for entry in tqdm(self.data_info, desc="Calculating slice info"):
            data_img = self._load_file(entry['ground_truth'])
            num_slices = data_img.shape[self.slice_axis]
            for i in range(num_slices):
                slice_info.append((entry, i))
        with open(self.slice_info_file, 'w') as f:
            json.dump(slice_info, f, indent=4)
        return slice_info

    def __getitem__(self, idx: Union[int, List[int]]) -> Union[Dict, List[Dict]]:
        '''
        for huggingface dataset, batch should be a dictionary:
        batch = {
            "original_image": [img1, img2, img3],
            "ground_truth_image": [edited_img1, edited_img2, edited_img3],
            "edit_prompt": ["prompt1", "prompt2", "prompt3"]
        }
        '''
        if isinstance(idx, int):
            return self.__get_single_item__(idx)
        elif isinstance(idx, list):
            return self.__get_batch_items__(idx)
        else:
            raise TypeError(f"Invalid index type: {type(idx)}. Expected int or list of int.")
    
        
    def __get_single_item__(self, idx: int) -> Dict:
        entry, slice_idx = self.slice_info[idx]
        
        # original = ground_truth = label = img
        # edited = oberservation = data = sino
        input_image = self._load_file(entry['observation'])
        input_slice = self._slice_volume(input_image, slice_idx)

        ground_truth_image = self._load_file(entry['ground_truth'])
        ground_truth_slice = self._slice_volume(ground_truth_image, slice_idx)
        
        # Normalize
        if self.do_normalize:
            scale_factor=3000
            ground_truth_slice = (ground_truth_slice - np.min(ground_truth_slice)) / scale_factor

        # Expand dimensions to include channel dimension
        input_slice = np.expand_dims(input_slice, axis=0)
        ground_truth_slice = np.expand_dims(ground_truth_slice, axis=0)
        
        # Convert to torch tensors
        input_slice = torch.from_numpy(input_slice).float()
        ground_truth_slice = torch.from_numpy(ground_truth_slice).float()
        
        # Resize
        #resize = ResizeWithPadOrCrop(spatial_size=(512, 512), mode="minimum")
        #input_slice = resize(input_slice)
        #ground_truth_slice = resize(ground_truth_slice)

        single_item = {"data": input_slice, 
                 "label": ground_truth_slice}
        return single_item

    def __get_batch_items__(self, indices: List[int]) -> Dict[str, List]:
        batch = {"input_image": [], "ground_truth_image": []}
        for idx in indices:
            item = self.__get_single_item__(idx)
            for key in batch.keys():
                batch[key].append(item[key])
        return batch
    

    def _load_file(self, file_id):
        if file_id.endswith('.nrrd') or file_id.endswith('.nii.gz'):
            data_img = sitk.ReadImage(file_id)
            data_img = sitk.GetArrayFromImage(data_img)
        elif file_id.endswith('.hdf5'):
            data_img = Load_from_HDF5(file_path=file_id, file_format= 'hdf5')
        
        data_img = np.moveaxis(data_img, 0, -1)

        if VERBOSE:
            self._check_images(data_img)
        return data_img

    def _slice_volume(self, data_img, slice_idx):
        if self.slice_axis == 0:
            data_slice = data_img[slice_idx, :, :]
        elif self.slice_axis == 1:
            data_slice = data_img[:, slice_idx, :]
        elif self.slice_axis == 2:
            data_slice = data_img[:, :, slice_idx]
        else:
            raise ValueError(f"Invalid axis: {self.slice_axis}. Axis must be 0, 1, or 2.")

        return data_slice

    def _preprocess(self, data):
        if self.mode == 'train':
            for sample in range(data.shape[0]):
                interval = 10
                variation = np.random.randint(-interval, interval)
                data[sample, :, :, 0] = data[sample, :, :, 0] + variation
                interval = 2
                variation = np.random.randint(-interval, interval)
                data[sample, :, :, 1] = data[sample, :, :, 1] + variation

        data = self.normalize(data)
        if data.ndim < 4:
            data = np.expand_dims(data, axis=-1)
        return data

    @staticmethod
    def adapt_to_task(data_img, label_img):
        return data_img, label_img

    def _check_images(self, data, lbl):
        print('            Data :     ', data.shape, np.max(data), np.min(data))
        print('            Label:       ', lbl.shape, np.max(lbl), np.min(lbl))
        print('-------------------------------------------')
        pass


def example_json_dataset():
    dataset_name = 'xcat'
    json_path = f"./data_table/{dataset_name}_dataset.json"
    slice_info_path = f"./data_table/{dataset_name}_slice_info.json"
    dataset = basejsonDataset(json_path=json_path, 
                              mode='train', 
                              transform_list=None, 
                              slice_axis=2, 
                              use_saved_slice_info=True,
                              slice_info_path=slice_info_path)
    dataloader=DataLoader(dataset, batch_size=4, shuffle=True)
    
    print("Length of dataset:", len(dataset))
    for batch in dataloader:
        data = batch["data"]
        label = batch["label"]
        print(data.shape)
        print(label.shape)
        break
    