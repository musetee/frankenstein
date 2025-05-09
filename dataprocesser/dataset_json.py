from dataprocesser.step0_dataset_base import BaseDataLoader
from dataprocesser.dataset_registry import register_dataset
import os
from torch.utils.data import DataLoader
import torch

@register_dataset('json_slice')
def load_json_slice(opt, my_paths):
    return slices_nifti_DataLoader(opt,my_paths, dimension=2)

class slices_nifti_DataLoader(BaseDataLoader):
    def __init__(self,configs,paths=None,dimension=2, **kwargs): 
        super().__init__(configs, paths, dimension, **kwargs)

    def get_dataset_list(self):
        print('use json dataset:',self.configs.dataset.data_dir)
        if self.configs.dataset.data_dir is not None and os.path.exists(self.configs.dataset.data_dir):
            json_file_root = self.configs.dataset.data_dir
        else:
            raise ValueError('please check the data dir in config file!')
        json_file_train = os.path.join(json_file_root, 'train', 'dataset.json')
        json_file_val = os.path.join(json_file_root, 'val', 'dataset.json')

        self.train_ds = list_from_json(json_file_train, self.indicator_A, self.indicator_B)
        self.val_ds = list_from_json(json_file_val, self.indicator_A, self.indicator_B)
        
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
        
def list_from_json(json_file, indicator_A, indicator_B):
    import json
    # it works for information saved in a json file:
    with open(json_file, 'r') as f:
            data = json.load(f)

    # Initialize lists to store the required information
    source_file_list = []
    target_file_list = []
    patient_IDs = []
    Aorta_diss_list = []
    
    # Iterate over the dataset and fill the lists
    for entry in data:
        # Extract the required fields
        source_file = entry['observation']  # Source file (observation)
        target_file = entry['ground_truth']  # Target file (ground truth)
        aorta_diss = entry['Aorta_diss']  # Aorta_diss value
        patient_id = entry['patient_ID']
        
        # Append to the respective lists
        source_file_list.append(source_file)
        target_file_list.append(target_file)
        patient_IDs.append(patient_id)
        Aorta_diss_list.append(aorta_diss)
    
    dataset = [{indicator_A: i, indicator_B: j, 'mask': k, 
                'A_paths': i, 'B_paths': j, 'mask_path': k, 
                'Aorta_diss':ad, 'patient_ID': pID} 
                    for i, j, k, ad, pID in zip(source_file_list, target_file_list, target_file_list, Aorta_diss_list, patient_IDs)]
    return dataset

        
