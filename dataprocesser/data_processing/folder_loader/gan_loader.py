import monai
import os
import numpy as np
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    SqueezeDimd,
    CenterSpatialCropd,
)

from monai.data import Dataset
from torch.utils.data import DataLoader
import torch
from .checkdata import check_volumes, save_volumes, check_batch_data
from .basics import get_file_list,crop_volumes, load_volumes, get_transforms

def load_batch_slices(train_volume_ds,val_volume_ds, train_batch_size=5,val_batch_size=1,window_width=1,ifcheck=True):
    patch_func = monai.data.PatchIterd(
        keys=["source", "target"],
        patch_size=(None, None, window_width),  # dynamic first two dimensions
        start_pos=(0, 0, 0)
    )
    if window_width==1:
        patch_transform = Compose(
            [
                SqueezeDimd(keys=["source", "target"], dim=-1),  # squeeze the last dim
            ]
        )
    else:
        patch_transform = None
    # for training
    train_patch_ds = monai.data.GridPatchDataset(
        data=train_volume_ds, patch_iter=patch_func, transform=patch_transform, with_coordinates=False)
    train_loader = DataLoader(
        train_patch_ds,
        batch_size=train_batch_size,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    # for validation
    val_loader = DataLoader(
        val_volume_ds, 
        num_workers=1, 
        batch_size=val_batch_size,
        pin_memory=torch.cuda.is_available())
    
    if ifcheck:
        check_batch_data(train_loader,val_loader,train_patch_ds,val_volume_ds,train_batch_size,val_batch_size)
    return train_loader,val_loader

def load_batch_slices3D(train_volume_ds,val_volume_ds, train_batch_size=5,val_batch_size=1,ifcheck=True):
    patch_func = monai.data.PatchIterd(
        keys=["source", "target"],
        patch_size=(None, None,32),  # dynamic first two dimensions
        start_pos=(0, 0, 0)
    )

    # for training
    train_patch_ds = monai.data.GridPatchDataset(
        data=train_volume_ds, patch_iter=patch_func, with_coordinates=False)
    train_loader = DataLoader(
        train_patch_ds,
        batch_size=train_batch_size,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    # for validation
    val_loader = DataLoader(
        val_volume_ds, 
        num_workers=1, 
        batch_size=val_batch_size,
        pin_memory=torch.cuda.is_available())
    
    if ifcheck:
        check_batch_data(train_loader,val_loader,train_patch_ds,val_volume_ds,train_batch_size,val_batch_size)
    return train_loader,val_loader

def myslicesloader(configs,paths):
    data_path=configs.dataset.data_dir
    train_number=configs.dataset.train_number
    val_number=configs.dataset.val_number
    train_batch_size=configs.dataset.batch_size
    val_batch_size=configs.dataset.val_batch_size
    saved_name_train=paths["saved_name_train"]
    saved_name_val=paths["saved_name_val"]
    center_crop=configs.dataset.center_crop
    source=configs.dataset.source
    target=configs.dataset.target
    ifcheck_volume=False
    ifcheck_sclices=False
    # volume-level transforms for both image and label
    train_transforms = get_transforms(configs,mode='train')
    val_transforms = get_transforms(configs,mode='val')

    #list all files in the folder
    file_list=[i for i in os.listdir(data_path) if 'overview' not in i]
    file_list_path=[os.path.join(data_path,i) for i in file_list]
    #list all ct and mr files in folder
    mask='mask'
    source_file_list=[os.path.join(j,f'{source}.nii.gz') for j in file_list_path]
    target_file_list=[os.path.join(j,f'{target}.nii.gz') for j in file_list_path]
    mask_file_list=[os.path.join(j,f'{mask}.nii.gz') for j in file_list_path]
    train_ds = [{'source': i, 'target': j, 'mask': k, 'A_paths': i, 'B_paths': j, 'mask_path': k} 
                for i, j, k in zip(source_file_list[0:train_number], target_file_list[0:train_number], mask_file_list[0:train_number])]
    val_ds = [{'source': i, 'target': j, 'mask': k, 'A_paths': i, 'B_paths': j, 'mask_path': k} 
              for i, j, k in zip(source_file_list[-val_number:], target_file_list[-val_number:], mask_file_list[-val_number:])]
    print('all files in dataset:',len(file_list))

    # load volumes and center crop
    if center_crop>0:
        crop=Compose([LoadImaged(keys=["source", "target", "mask"]),
                    EnsureChannelFirstd(keys=["source", "target", "mask"]),
                    CenterSpatialCropd(keys=["source", "target", "mask"], roi_size=(-1,-1,center_crop)),
                    
                    ])
        train_crop_ds = monai.data.Dataset(data=train_ds, transform=crop)
        val_crop_ds = monai.data.Dataset(data=val_ds, transform=crop)
        print('center crop:',center_crop)
    else:
        crop=Compose([LoadImaged(keys=["source", "target", "mask"]),
            EnsureChannelFirstd(keys=["source", "target", "mask"]),
            ])
        train_crop_ds = monai.data.Dataset(data=train_ds, transform=crop)
        val_crop_ds = monai.data.Dataset(data=val_ds, transform=crop)

    # load volumes
    train_volume_ds = monai.data.Dataset(data=train_crop_ds, transform=train_transforms) 
    val_volume_ds = monai.data.Dataset(data=val_crop_ds, transform=val_transforms)
    ifsave,ifcheck=False,False
    if ifsave:
        save_volumes(train_ds, val_ds, saved_name_train, saved_name_val)
    if ifcheck:
        check_volumes(train_ds, train_volume_ds, val_volume_ds, val_ds)

    # batch-level slicer for both image and label
    window_width=1
    patch_func = monai.data.PatchIterd(
        keys=["source", "target", "mask"],
        patch_size=(None, None, window_width),  # dynamic first two dimensions
        start_pos=(0, 0, 0)
    )
    if window_width==1:
        patch_transform = Compose(
            [
                SqueezeDimd(keys=["source", "target", "mask"], dim=-1),  # squeeze the last dim
            ]
        )
    else:
        patch_transform = None
        
    # for training
    train_patch_ds = monai.data.GridPatchDataset(
        data=train_volume_ds, patch_iter=patch_func, transform=patch_transform, with_coordinates=False)
    train_loader = DataLoader(
        train_patch_ds,
        batch_size=train_batch_size,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    # for validation
    val_loader = DataLoader(
        val_volume_ds, 
        num_workers=1, 
        batch_size=val_batch_size,
        pin_memory=torch.cuda.is_available())
    
    if ifcheck:
        check_batch_data(train_loader,val_loader,train_patch_ds,val_volume_ds,train_batch_size,val_batch_size)
    
    return train_crop_ds,val_crop_ds,train_loader,val_loader,train_transforms,val_transforms

def mydataloader_3d(data_pelvis_path,
                   train_number,
                   val_number,
                   train_batch_size,
                   val_batch_size,
                   saved_name_train='./train_ds_2d.csv',
                   saved_name_val='./val_ds_2d.csv',
                   resized_size=(600,400,150),
                   div_size=(16,16,16),
                   ifcheck_volume=True,):
    # volume-level transforms for both image and segmentation
    normalize='zscore'
    train_transforms = get_transforms(normalize,resized_size,div_size)
    
    train_ds, val_ds = get_file_list(data_pelvis_path, 
                                     train_number, 
                                     val_number)
    #train_volume_ds, val_volume_ds 
    
    train_volume_ds,val_volume_ds = load_volumes(train_transforms=train_transforms, 
                                                train_ds=train_ds, 
                                                val_ds=val_ds, 
                                                saved_name_train=saved_name_train, 
                                                saved_name_val=saved_name_train,
                                                ifsave=True,
                                                ifcheck=ifcheck_volume)
    '''
    train_loader = DataLoader(train_volume_ds, batch_size=train_batch_size)
    val_loader = DataLoader(val_volume_ds, batch_size=val_batch_size)
    '''
    ifcheck_sclices=False
    train_loader,val_loader = load_batch_slices3D(train_volume_ds, 
                                                val_volume_ds, 
                                                train_batch_size,
                                                val_batch_size=val_batch_size,
                                                ifcheck=ifcheck_sclices)
                                                
    return train_loader,val_loader,train_transforms


from torchvision.utils import save_image
def save_dataset_as_png(train_ds, train_volume_ds,saved_img_folder,saved_label_folder):    
    train_loader = DataLoader(train_volume_ds, batch_size=1)
    for idx, train_check_data in enumerate(train_loader):
        image_volume = train_check_data['image']
        label_volume = train_check_data['label']
        current_item = train_ds[idx]
        file_name_prex = os.path.basename(os.path.dirname(current_item['image']))
        slices_num=image_volume.shape[-1]
        for i in range(slices_num):
            image_i=image_volume[0,0,:,:,i]
            label_i=label_volume[0,0,:,:,i]
            #print(label_volume.shape)
            #SaveImage(output_dir=saved_img_folder, output_postfix=f'{file_name_prex}_image', output_ext='.png', resample=True)(image_volume[0,:,:,:,0])
            save_image(image_i, f'{saved_img_folder}\{file_name_prex}_image_{i}.png')
            save_image(label_i, f'{saved_label_folder}\{file_name_prex}_label_{i}.png')

def pre_dataset_for_stylegan(data_pelvis_path,
                            normalize,
                            train_number,
                            val_number,
                            saved_img_folder,
                            saved_label_folder,
                            saved_name_train='./train_ds_2d.csv',
                            saved_name_val='./val_ds_2d.csv',
                            resized_size=(600,400,None),
                            div_size=(16,16,None),):
    train_transforms = get_transforms(normalize,resized_size,div_size)
    train_ds, val_ds = get_file_list(data_pelvis_path, 
                                     train_number, 
                                     val_number)
    train_volume_ds, _ = load_volumes(train_transforms, 
                                                train_ds, 
                                                val_ds, 
                                                saved_name_train, 
                                                saved_name_val,
                                                ifsave=False,
                                                ifcheck=False)
    save_dataset_as_png(train_ds, train_volume_ds,saved_img_folder,saved_label_folder)
    return train_ds,train_volume_ds

def sum_slices(data_pelvis_path, num=180):
    train_ds, val_ds=get_file_list(data_pelvis_path, 0, num)
    train_ds_2d, val_ds_2d,\
    all_slices_train,all_slices_val,\
    shape_list_train,shape_list_val = transform_datasets_to_2d(train_ds, val_ds, 
                                                            saved_name_train='./train_ds_2d.csv', 
                                                            saved_name_val='./val_ds_2d.csv', 
                                                            ifsave=False)
    print(all_slices_val)
    return all_slices_val

def transform_datasets_to_2d(train_ds, val_ds, saved_name_train, saved_name_val,ifsave=True):
    # Load 2D slices of CT images
    train_ds_2d = []
    val_ds_2d = []
    shape_list_train = []
    shape_list_val = []
    all_slices_train=0
    all_slices_val=0

    # Load 2D slices for training
    for sample in train_ds:
        train_ds_2d_image = LoadImaged(keys=["source","target"],image_only=True, ensure_channel_first=False, simple_keys=True)(sample)
        name = os.path.basename(os.path.dirname(sample['image']))
        num_slices = train_ds_2d_image["source"].shape[-1]
        shape_list_train.append({'patient': name, 'shape': train_ds_2d_image["image"].shape})
        for i in range(num_slices):
            train_ds_2d.append({'image': train_ds_2d_image['image'][:,:,i], 'label': train_ds_2d_image['label'][:,:,i]})
        all_slices_train += num_slices

    # Load 2D slices for validation
    for sample in val_ds:
        val_ds_2d_image = LoadImaged(keys=["source","target"],image_only=True, ensure_channel_first=False, simple_keys=True)(sample)
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

def get_train_val_loaders(train_ds_2d, val_ds_2d, batch_size, val_batch_size,normalize, resized_size=(600,400), div_size=(16,16,None),):
    # Define transforms
    train_transforms = get_transforms(normalize,resized_size,div_size)
    train_transforms_list=train_transforms.__dict__['transforms']
    batch_size = batch_size
    # Create training dataset and data loader
    train_dataset = Dataset(data=train_ds_2d, transform=train_transforms)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)

    val_batch_size = val_batch_size
    # Create validation dataset and data loader
    val_dataset = Dataset(data=val_ds_2d, transform=train_transforms)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=1, pin_memory=True)
    return train_loader, val_loader, train_transforms_list,train_transforms

def mydataloader(data_pelvis_path,
                 train_number,
                 val_number,
                 batch_size,
                 val_batch_size,
                 saved_name_train='./train_ds_2d.csv',
                 saved_name_val='./val_ds_2d.csv',
                 resized_size=(600,400)): 
    train_ds, val_ds = get_file_list(data_pelvis_path, 
                                     train_number, 
                                     val_number)
    train_ds_2d, val_ds_2d,\
    all_slices_train,all_slices_val,\
    shape_list_train,shape_list_val = transform_datasets_to_2d(train_ds, val_ds, 
                                                            saved_name_train, 
                                                            saved_name_val,ifsave=True)
    
    train_loader, val_loader, \
    train_transforms_list,train_transforms = get_train_val_loaders(train_ds_2d, 
                                                                val_ds_2d, 
                                                                batch_size=batch_size, 
                                                                val_batch_size=val_batch_size,
                                                                normalize='zscore',
                                                                resized_size=resized_size,
                                                                div_size=(16,16,None),)
    return train_loader,val_loader,\
            train_transforms_list,train_transforms,\
            all_slices_train,all_slices_val,\
            shape_list_train,shape_list_val