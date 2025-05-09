from dataprocesser.dataset_registry import register_dataset
from dataprocesser.step0_dataset_base import BaseDataLoader
from dataprocesser.customized_transforms import (
    MaskHUAssigmentd, 
    CreateMaskWithBonesTransformd)
import os 
from torch.utils.data import DataLoader
import torch
from monai.transforms import (
    ScaleIntensityd,
    ThresholdIntensityd,
    NormalizeIntensityd,
    ShiftIntensityd,
)
import pandas as pd
@register_dataset('csv_slice')
def load_csv_slice(opt, my_paths):
    return csv_slices_DataLoader(opt, my_paths, dimension=2)

@register_dataset('csv_slice_assigned')
def csv_slices_assigned(opt, my_paths):
    return csv_slices_DataLoader(opt, my_paths, dimension=2)

class csv_slices_DataLoader(BaseDataLoader):
    def __init__(self,configs,paths=None,dimension=2, **kwargs): 
        super().__init__(configs, paths, dimension, **kwargs)

    def get_dataset_list(self):
        print('use csv dataset:',self.configs.dataset.data_dir)
        if self.configs.dataset.data_dir is not None and os.path.exists(self.configs.dataset.data_dir):
            csv_file_root = self.configs.dataset.data_dir
        else:
            raise ValueError('please check the data dir in config file!')
        folder_train = os.path.join(csv_file_root, 'train')
        folder__val = os.path.join(csv_file_root, 'val')

        self.train_ds = list_from_slice_csv(folder_train, self.indicator_A, self.indicator_B)
        self.val_ds = list_from_slice_csv(folder__val, self.indicator_A, self.indicator_B)
        
    def create_patch_dataset_and_dataloader(self, dimension=2):
        train_batch_size=self.configs.dataset.batch_size
        val_batch_size=self.configs.dataset.val_batch_size
        self.train_loader = DataLoader(
            self.train_volume_ds, 
            num_workers=self.num_workers, 
            batch_size=train_batch_size,
            shuffle=True,
            pin_memory=torch.cuda.is_available())
        
        self.val_loader = DataLoader(
            self.val_volume_ds, 
            num_workers=self.num_workers, 
            batch_size=val_batch_size,
            shuffle=False,
            pin_memory=torch.cuda.is_available())
        
class csv_slices_assigned_DataLoader(csv_slices_DataLoader):
    def __init__(self,configs,paths=None,dimension=2, **kwargs): 
        super().__init__(configs, paths, dimension, **kwargs)

    def get_pretransforms(self, transform_list):
        transform_list.append(MaskHUAssigmentd(keys=[self.indicator_A], csv_file=r'synthrad_conversion\TA2_anatomy.csv'))
        return transform_list
    
    def get_intensity_transforms(self, transform_list):
        threshold_low=self.configs.dataset.WINDOW_LEVEL - self.configs.dataset.WINDOW_WIDTH / 2
        threshold_high=self.configs.dataset.WINDOW_LEVEL + self.configs.dataset.WINDOW_WIDTH / 2
        offset=(-1)*threshold_low
        # if filter out the pixel with values below threshold1, set above=True, and the cval1>=threshold1, otherwise there will be problem
        # mask = img > self.threshold if self.above else img < self.threshold
        # res = where(mask, img, self.cval)
        transform_list.append(ThresholdIntensityd(keys=[self.indicator_A,self.indicator_B], threshold=threshold_low, above=True, cval=threshold_low))
        transform_list.append(ThresholdIntensityd(keys=[self.indicator_A,self.indicator_B], threshold=threshold_high, above=False, cval=threshold_high))
        transform_list.append(ShiftIntensityd(keys=[self.indicator_A,self.indicator_B], offset=offset))
        return transform_list
    
    def get_normlization(self, transform_list):
        normalize=self.configs.dataset.normalize
        # offset = self.configs.dataset.offset
        # we don't need normalization for segmentation mask
        if normalize=='zscore':
            transform_list.append(NormalizeIntensityd(keys=[self.indicator_A,self.indicator_B], nonzero=False, channel_wise=True))
            print('zscore normalization')

        elif normalize=='scale2000':
            transform_list.append(ScaleIntensityd(keys=[self.indicator_A,self.indicator_B], minv=None, maxv=None, factor=-0.9995))
            print('scale2000 normalization')

        elif normalize=='none' or normalize=='nonorm':
            print('no normalization')
            
        return transform_list

def list_from_slice_csv(train_or_val_folder, indicator_A, indicator_B):
    # it works for information saved in a json file:
    patient_info_csv = os.path.join(train_or_val_folder, 'patient_info.csv')
    slice_dataset_info = os.path.join(train_or_val_folder, 'dataset.csv')

    data_frame_patient_info = pd.read_csv(patient_info_csv)
    data_frame_slice = pd.read_csv(slice_dataset_info)
    if len(data_frame_patient_info) == 0:
        raise RuntimeError(f"Found 0 images in: {patient_info_csv}")
    
    required_patient_IDs = data_frame_patient_info.iloc[:, -2].tolist()
    Aorta_diss = data_frame_patient_info.iloc[:, -1].tolist()

    # Initialize lists to store the required information
    target_file_list_all = data_frame_slice.iloc[:, 0].tolist()  # img
    source_file_list_all = data_frame_slice.iloc[:, 1].tolist() # seg
    patient_IDs_all = data_frame_slice.iloc[:, 2].tolist() # patient_ID
    Aorta_diss_list_all = data_frame_slice.iloc[:, 3].tolist() # Aorta_diss
    
    source_file_list = []
    target_file_list = []
    patient_IDs_list = []
    Aorta_diss_list = []
    # Iterate over the dataset and fill the lists
    for idx in range(len(data_frame_slice)):
        if patient_IDs_all[idx] in required_patient_IDs:
            # Append to the respective lists
            target_file_list.append(target_file_list_all[idx])
            source_file_list.append(source_file_list_all[idx])
            patient_IDs_list.append(patient_IDs_all[idx])
            Aorta_diss_list.append(Aorta_diss_list_all[idx])
    
    dataset = [{indicator_A: i, indicator_B: j, 'mask': k, 
                'A_paths': i, 'B_paths': j, 'mask_path': k, 
                'Aorta_diss':ad, 'patient_ID': pID} 
                    for i, j, k, ad, pID in zip(source_file_list, target_file_list, target_file_list, Aorta_diss_list, patient_IDs_list)]
    return dataset
