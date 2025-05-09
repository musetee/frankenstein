import monai
import os
import numpy as np
from monai.transforms import (
    Compose,
    LoadImaged,
    Rotate90d,
    ScaleIntensityd,
    EnsureChannelFirstd,
    ResizeWithPadOrCropd,
    DivisiblePadd,
    ThresholdIntensityd,
    NormalizeIntensityd,
    SqueezeDimd,
    Identityd,
    CenterSpatialCropd,
)

from monai.data import Dataset
from torch.utils.data import DataLoader
import torch
from .basics import get_file_list, get_transforms

def transform_datasets_to_2d(train_ds, val_ds, saved_name_train, saved_name_val, batch_size=8,ifsave=True):
    # Load 2D slices of CT images
    train_ds_2d = []
    val_ds_2d = []
    shape_list_train = []
    shape_list_val = []
    all_slices_train=0
    all_slices_val=0

    # Load 2D slices for training
    for sample in train_ds:
        train_ds_2d_image = LoadImaged(keys=["image", "label"],image_only=True, ensure_channel_first=False, simple_keys=True)(sample)
        train_ds_2d_image=DivisiblePadd(["image", "label"], (-1,batch_size), mode="minimum")(train_ds_2d_image)
        name = os.path.basename(os.path.dirname(sample['image']))
        num_slices = train_ds_2d_image["image"].shape[-1]
        #print(train_ds_2d_image["image"].shape)
        #print(num_slices)
        shape_list_train.append({'patient': name, 'shape': train_ds_2d_image["image"].shape})
        for i in range(num_slices):
            train_ds_2d.append({'image': train_ds_2d_image['image'][:,:,i], 'label': train_ds_2d_image['label'][:,:,i]})
        all_slices_train += num_slices

    # Load 2D slices for validation
    for sample in val_ds:
        val_ds_2d_image = LoadImaged(keys=["image", "label"],image_only=True, ensure_channel_first=False, simple_keys=True)(sample)
        val_ds_2d_image=DivisiblePadd(["image", "label"], (-1, batch_size), mode="minimum")(val_ds_2d_image)
        name = os.path.basename(os.path.dirname(sample['image']))
        shape_list_val.append({'patient': name, 'shape': val_ds_2d_image["image"].shape})
        num_slices = val_ds_2d_image["image"].shape[-1]
        for i in range(num_slices):
            val_ds_2d.append({'image': val_ds_2d_image['image'][:,:,i], 'label': val_ds_2d_image['label'][:,:,i]})
        all_slices_val += num_slices
    # Save shape list to csv
    if ifsave:
        np.savetxt(saved_name_train,shape_list_train,delimiter=',',fmt='%s',newline='\n') # f means format, r means raw string
        np.savetxt(saved_name_val,shape_list_val,delimiter=',',fmt='%s',newline='\n') # f means format, r means raw string
    return train_ds_2d, val_ds_2d, all_slices_train, all_slices_val, shape_list_train, shape_list_val
def get_train_val_loaders(train_ds_2d, val_ds_2d, batch_size, val_batch_size,resized_size=(256,256)):
    # Define transforms
    '''
    normalize='zscore'
    div_size=(16,16,None)
    train_transforms = get_transforms(normalize,resized_size,div_size)
    '''

    train_transforms = Compose(
        [
            EnsureChannelFirstd(keys=["image", "label"]),
            NormalizeIntensityd(keys=["image", "label"], nonzero=False, channel_wise=True), # z-score normalization
            ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=resized_size,mode="minimum"),
            Rotate90d(keys=["image", "label"], k=3),
            DivisiblePadd(["image", "label"], 16, mode="minimum"),
        ]
    )
    train_transforms_list=train_transforms.__dict__['transforms']
    # Create training dataset and data loader
    train_dataset = Dataset(data=train_ds_2d, transform=train_transforms)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    val_batch_size = val_batch_size
    # Create validation dataset and data loader
    val_dataset = Dataset(data=val_ds_2d, transform=train_transforms)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, val_loader, train_transforms_list,train_transforms
def mydataloader(data_pelvis_path,
                normalize='zscore',
                pad='minimum',
                train_number=1,
                val_number=1,
                batch_size=8,
                val_batch_size=1,
                saved_name_train='./train_ds_2d.csv',saved_name_val='./val_ds_2d.csv',
                resized_size=(512,512),
                div_size=(16,16,None),
                center_crop=20,): 
    #train_transforms = get_transforms(normalize,pad,resized_size,div_size)
    train_ds, val_ds = get_file_list(data_pelvis_path, 
                                     train_number, 
                                     val_number)
    train_ds_2d, val_ds_2d,\
    all_slices_train,all_slices_val,\
    shape_list_train,shape_list_val = transform_datasets_to_2d(train_ds, val_ds, 
                                                            saved_name_train, 
                                                            saved_name_val,
                                                            batch_size=batch_size,
                                                            ifsave=False)
    train_loader, val_loader, \
    train_transforms_list,train_transforms = get_train_val_loaders(train_ds_2d, 
                                                                val_ds_2d, 
                                                                batch_size=batch_size, 
                                                                val_batch_size=val_batch_size,
                                                                resized_size=resized_size)
    return train_loader,val_loader,\
            train_transforms_list,train_transforms,\
            all_slices_train,all_slices_val,\
            shape_list_train,shape_list_val