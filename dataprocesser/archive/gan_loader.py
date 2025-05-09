import monai
import os
import numpy as np
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    SqueezeDimd,
    CenterSpatialCropd,
    Rotate90d,
    ScaleIntensityd,
    ResizeWithPadOrCropd,
    DivisiblePadd,
    ThresholdIntensityd,
    NormalizeIntensityd,
    ShiftIntensityd,
    Identityd,
    ScaleIntensityRanged,
    Spacingd,
)

from monai.data import Dataset
from torch.utils.data import DataLoader
import torch
from .checkdata import check_volumes, save_volumes, check_batch_data, test_volumes_pixdim

def get_transforms(configs, mode='train'):
    normalize=configs.dataset.normalize
    pad=configs.dataset.pad
    resized_size=configs.dataset.resized_size
    WINDOW_WIDTH=configs.dataset.WINDOW_WIDTH
    WINDOW_LEVEL=configs.dataset.WINDOW_LEVEL
    prob=configs.dataset.augmentationProb
    background=configs.dataset.background

    transform_list=[]
    min, max=WINDOW_LEVEL-(WINDOW_WIDTH/2), WINDOW_LEVEL+(WINDOW_WIDTH/2)
    #transform_list.append(ThresholdIntensityd(keys=["target"], threshold=min, above=True, cval=background))
    #transform_list.append(ThresholdIntensityd(keys=["target"], threshold=max, above=False, cval=-1000))
    # filter the source images
    # transform_list.append(ThresholdIntensityd(keys=["source"], threshold=configs.dataset.MRImax, above=False, cval=0))
    if normalize=='zscore':
        transform_list.append(NormalizeIntensityd(keys=["source", "target"], nonzero=False, channel_wise=True))
        print('zscore normalization')
    elif normalize=='minmax':
        transform_list.append(ScaleIntensityd(keys=["source", "target"], minv=-1.0, maxv=1.0))
        print('minmax normalization')

    elif normalize=='scale4000':
        transform_list.append(ScaleIntensityd(keys=["source"], minv=0, maxv=1))
        transform_list.append(ScaleIntensityd(keys=["target"], minv=0))
        transform_list.append(ScaleIntensityd(keys=["target"], factor=-0.99975)) # x=x(1+factor)
        print('scale1000 normalization')

    elif normalize=='scale1000':
        transform_list.append(ScaleIntensityd(keys=["source"], minv=0, maxv=1))
        transform_list.append(ScaleIntensityd(keys=["target"], minv=0))
        transform_list.append(ScaleIntensityd(keys=["target"], factor=-0.999)) 
        print('scale1000 normalization')

    elif normalize=='inputonlyzscore':
        transform_list.append(NormalizeIntensityd(keys=["source"], nonzero=False, channel_wise=True))
        print('only normalize input MRI images')

    elif normalize=='inputonlyminmax':
        transform_list.append(ScaleIntensityd(keys=["source"], minv=configs.dataset.normmin, maxv=configs.dataset.normmax))
        print('only normalize input MRI images')
    
    elif normalize=='none' or normalize=='nonorm':
        print('no normalization')

    spaceXY=0
    if spaceXY>0:
        transform_list.append(Spacingd(keys=["source"], pixdim=(spaceXY, spaceXY, 2.5), mode="bilinear")) # 
        transform_list.append(Spacingd(keys=["target", "mask"], pixdim=(spaceXY, spaceXY , 2.5), mode="bilinear")) #
    transform_list.append(ResizeWithPadOrCropd(keys=["source", "target"], spatial_size=resized_size,mode=pad))
    # transform_list.append(ScaleIntensityRanged(keys=["target"],a_min=WINDOW_LEVEL-(WINDOW_WIDTH/2), a_max=WINDOW_LEVEL+(WINDOW_WIDTH/2),b_min=0, b_max=1, clip=True))
    
    if configs.dataset.rotate:
        transform_list.append(Rotate90d(keys=["source",  "target"], k=3))

    if mode == 'train':
        from monai.transforms import (
            # data augmentation
            RandRotated,
            RandZoomd,
            RandBiasFieldd,
            RandAffined,
            RandGridDistortiond,
            RandGridPatchd,
            RandShiftIntensityd,
            RandGibbsNoised,
            RandAdjustContrastd,
            RandGaussianSmoothd,
            RandGaussianSharpend,
            RandGaussianNoised,
        )
        Aug=True
        if Aug:
            transform_list.append(RandRotated(keys=["source", "target", "mask"], range_x = 0.1, range_y = 0.1, range_z = 0.1, prob=prob, padding_mode="border", keep_size=True))
            transform_list.append(RandZoomd(keys=["source", "target", "mask"], prob=prob, min_zoom=0.9, max_zoom=1.3,padding_mode= "minimum" ,keep_size=True))
            transform_list.append(RandAffined(keys=["source", "target"],padding_mode="border" , prob=prob))
            #transform_list.append(Rand3DElasticd(keys=["source", "target"], prob=prob, sigma_range=(5, 8), magnitude_range=(100, 200), spatial_size=None, mode='bilinear'))
        intensityAug=False
        if intensityAug:
            print('intensity data augmentation is used')
            transform_list.append(RandBiasFieldd(keys=["source"], degree=3, coeff_range=(0.0, 0.1), prob=prob)) # only apply to MRI images
            transform_list.append(RandGaussianNoised(keys=["source"], prob=prob, mean=0.0, std=0.01))
            transform_list.append(RandAdjustContrastd(keys=["source"], prob=prob, gamma=(0.5, 1.5)))
            transform_list.append(RandShiftIntensityd(keys=["source"], prob=prob, offsets=20))
            transform_list.append(RandGaussianSharpend(keys=["source"], alpha=(0.2, 0.8), prob=prob))
        
    #transform_list.append(Rotate90d(keys=["source", "target"], k=3))
    #transform_list.append(DivisiblePadd(keys=["source", "target"], k=div_size, mode="minimum"))
    #transform_list.append(Identityd(keys=["source", "target"]))  # do nothing for the no norm case
    train_transforms = Compose(transform_list)
    return train_transforms
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
    ifsave,ifcheck,iftest=False,False,False
    if ifsave:
        save_volumes(train_ds, val_ds, saved_name_train, saved_name_val)
    if ifcheck:
        check_volumes(train_ds, train_volume_ds, val_volume_ds, val_ds)
    if iftest:
        test_volumes_pixdim(train_volume_ds)

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

def ddpmloader(configs,paths):
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
        crop=Compose([LoadImaged(keys=["source", "target"]),
                    EnsureChannelFirstd(keys=["source", "target"]),
                    CenterSpatialCropd(keys=["source", "target"], roi_size=(-1,-1,center_crop)),
                    
                    ])
        train_crop_ds = monai.data.Dataset(data=train_ds, transform=crop)
        val_crop_ds = monai.data.Dataset(data=val_ds, transform=crop)
        print('center crop:',center_crop)
    else:
        crop=Compose([LoadImaged(keys=["source", "target"]),
            EnsureChannelFirstd(keys=["source", "target"]),
            ])
        train_crop_ds = monai.data.Dataset(data=train_ds, transform=crop)
        val_crop_ds = monai.data.Dataset(data=val_ds, transform=crop)

    # load volumes
    train_volume_ds = monai.data.Dataset(data=train_crop_ds, transform=train_transforms) 
    val_volume_ds = monai.data.Dataset(data=val_crop_ds, transform=val_transforms)
    ifsave,ifcheck,iftest=False,False,False
    if ifsave:
        save_volumes(train_ds, val_ds, saved_name_train, saved_name_val)
    if ifcheck:
        check_volumes(train_ds, train_volume_ds, val_volume_ds, val_ds)
    if iftest:
        test_volumes_pixdim(train_volume_ds)

    # batch-level slicer for both image and label
    window_width=1
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
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    # for validation
    val_patch_ds = monai.data.GridPatchDataset(
        data=val_volume_ds, patch_iter=patch_func, transform=patch_transform, with_coordinates=False)
    val_loader = DataLoader(
        val_patch_ds, #val_volume_ds, 
        num_workers=0, 
        batch_size=val_batch_size,
        pin_memory=torch.cuda.is_available())
    
    if ifcheck:
        check_batch_data(train_loader,val_loader,train_patch_ds,val_volume_ds,train_batch_size,val_batch_size)
    
    return train_crop_ds,val_crop_ds,train_loader,val_loader,train_transforms,val_transforms
