import SimpleITK as sitk
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from configs import config as cfg
VERBOSE = cfg.verbose

class CtrecoDataset(Dataset):
    def __init__(self, file_ids, mode='train', transform_list=None, slice_axis=2):
        """
        Args:
            file_ids (list): List of file ids to load data from.
            mode (str): 'train' or 'test'. Determines if augmentation is applied.
            transform_list (list of callable, optional): List of transforms to be applied on a sample.
            slice_axis (int): The axis along which to slice the 3D volumes (0, 1, or 2).
        """
        self.file_ids = file_ids
        self.mode = mode
        self.transform_list = transform_list
        self.slice_axis = slice_axis
        self.data_slices, self.label_slices = self._load_and_slice_all_files()

    def __len__(self):
        return len(self.data_slices)

    def __getitem__(self, idx):
        data_slice = self.data_slices[idx]
        label_slice = self.label_slices[idx]

        # Apply additional transforms if specified
        if self.transform_list:
            for transform in self.transform_list:
                data_slice, label_slice = transform(data_slice, label_slice)

        # Expand dimensions to include channel dimension
        data_slice = np.expand_dims(data_slice, axis=0)
        label_slice = np.expand_dims(label_slice, axis=0)
        
        # Convert to torch tensors
        data_slice = torch.from_numpy(data_slice).float()
        label_slice = torch.from_numpy(label_slice).float()
        
        batch = {"data": data_slice, "label": label_slice}
        return batch

    def _load_and_slice_all_files(self):
        data_slices = []
        label_slices = []
        for file_id in self.file_ids:
            data_img, label_img = self._load_file(file_id)
            slices_data, slices_label = self._slice_volume(data_img, label_img)
            data_slices.extend(slices_data)
            label_slices.extend(slices_label)
        return data_slices, label_slices

    def _load_file(self, file_id):
        _, file_number = os.path.split(file_id)

        data_img = sitk.ReadImage(os.path.join(file_id + '_sino_Metal.nrrd'))
        label_img = sitk.ReadImage(os.path.join(file_id + '_img_GT_noNoise.nrrd'))

        data_img = sitk.GetArrayFromImage(data_img)
        label_img = sitk.GetArrayFromImage(label_img)

        data_img = np.moveaxis(data_img, 0, -1)
        label_img = np.moveaxis(label_img, 0, -1)

        data_img, label_img = self.adapt_to_task(data_img, label_img)
        self._check_images(data_img, label_img)
        return data_img, label_img

    def _slice_volume(self, data_img, label_img):
        if self.slice_axis == 0:
            data_slices = [data_img[i, :, :] for i in range(data_img.shape[0])]
            label_slices = [label_img[i, :, :] for i in range(label_img.shape[0])]
        elif self.slice_axis == 1:
            data_slices = [data_img[:, i, :] for i in range(data_img.shape[1])]
            label_slices = [label_img[:, i, :] for i in range(label_img.shape[1])]
        elif self.slice_axis == 2:
            data_slices = [data_img[:, :, i] for i in range(data_img.shape[2])]
            label_slices = [label_img[:, :, i] for i in range(label_img.shape[2])]
        else:
            raise ValueError(f"Invalid axis: {self.slice_axis}. Axis must be 0, 1, or 2.")

        return data_slices, label_slices

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

class SliceTransform:
    def __init__(self, axis=2):
        """
        Args:
            axis (int): The axis along which to slice the 3D volume (0, 1, or 2).
        """
        self.axis = axis

    def __call__(self, data_img, label_img):
        """
        Slices the 3D volumes into 2D slices along the specified axis.

        Args:
            data_img (numpy.ndarray): 3D data volume.
            label_img (numpy.ndarray): 3D label volume.

        Returns:
            data_slices (list of numpy.ndarray): List of 2D data slices.
            label_slices (list of numpy.ndarray): List of 2D label slices.
        """
        if self.axis == 0:
            data_slices = [data_img[i, :, :] for i in range(data_img.shape[0])]
            label_slices = [label_img[i, :, :] for i in range(label_img.shape[0])]
        elif self.axis == 1:
            data_slices = [data_img[:, i, :] for i in range(data_img.shape[1])]
            label_slices = [label_img[:, i, :] for i in range(label_img.shape[1])]
        elif self.axis == 2:
            data_slices = [data_img[:, :, i] for i in range(data_img.shape[2])]
            label_slices = [label_img[:, :, i] for i in range(label_img.shape[2])]
        else:
            raise ValueError(f"Invalid axis: {self.axis}. Axis must be 0, 1, or 2.")

        return data_slices, label_slices

class Normalize:
    def __init__(self, type):
        self.type = type
    
    def __call__(self, data_img, label_img):
        if self.type == "window":
            #print("windowing")
            label_img = self.windowing(label_img)
        elif self.type == "scale":
            #print("scaling")
            label_img = self.scaling(label_img)
        return data_img, label_img
    def windowing(self, img):
        img_min=-100
        img_max=500
        img = np.clip(img, img_min, img_max)
        img = (img - img_min) / (img_max-img_min)
        return img
    def scaling(self, img):
        scale_factor=3000
        img = (img - np.min(img)) / scale_factor
        return img
    
def create_dataloader(file_ids, batch_size=4, mode='train', transform_list=None, shuffle=True):
    dataset = CtrecoDataset(file_ids, mode=mode, transform_list=transform_list)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

if __name__ == "__main__":
    test_csv = r"./data_table/test.csv"
    test_data = pd.read_csv(test_csv, dtype=object)
    test_filelist = test_data.iloc[:, -1].tolist()
    print(test_filelist)
    transform_list = []
    transform_list.append(Normalize(type="scale"))
    
    dataset = CtrecoDataset(file_ids=test_filelist, mode='train', transform_list=transform_list, slice_axis=2)
    dataloader=DataLoader(dataset, batch_size=4, shuffle=True)
    
    print("Length of dataset:", len(dataset))
    for batch in dataloader:
        data = batch["data"]
        label = batch["label"]
        print(data.shape)
        print(label.shape)
        break
    