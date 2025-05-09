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
    ShiftIntensityd,
    Identityd,
    CenterSpatialCropd,
    ScaleIntensityRanged,
    Spacingd,
)
from torch.utils.data import DataLoader
from .checkdata import check_volumes, save_volumes
def get_file_list(data_pelvis_path, train_number, val_number, source='mr', target='ct'):
    #list all files in the folder
    file_list=[i for i in os.listdir(data_pelvis_path) if 'overview' not in i]
    file_list_path=[os.path.join(data_pelvis_path,i) for i in file_list]
    #list all ct and mr files in folder
    source_file_list=[os.path.join(j,f'{source}.nii.gz') for j in file_list_path]
    target_file_list=[os.path.join(j,f'{target}.nii.gz') for j in file_list_path] #mr
    # Dict Version
    # source -> image
    # target -> label
    train_ds = [{'source': i, 'target': j, 'A_paths': i, 'B_paths': j} for i, j in zip(source_file_list[0:train_number], target_file_list[0:train_number])]
    val_ds = [{'source': i, 'target': j, 'A_paths': i, 'B_paths': j} for i, j in zip(source_file_list[-val_number:], target_file_list[-val_number:])]
    print('all files in dataset:',len(file_list))
    return train_ds, val_ds

def load_volumes(train_transforms,val_transforms,
                 train_crop_ds, val_crop_ds, 
                 train_ds, val_ds, 
                 saved_name_train=None, saved_name_val=None,
                 ifsave=False,ifcheck=False):
    train_volume_ds = monai.data.Dataset(data=train_crop_ds, transform=train_transforms) 
    val_volume_ds = monai.data.Dataset(data=val_crop_ds, transform=val_transforms)
    if ifsave:
        save_volumes(train_ds, val_ds, saved_name_train, saved_name_val)
    if ifcheck:
        check_volumes(train_ds, train_volume_ds, val_volume_ds, val_ds)
    return train_volume_ds,val_volume_ds

def crop_volumes(train_ds, val_ds,center_crop,resized_size=(512,512,None),pad='minimum'):
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
    return train_crop_ds, val_crop_ds 

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
    transform_list.append(ThresholdIntensityd(keys=["target"], threshold=min, above=True, cval=background))
    #transform_list.append(ThresholdIntensityd(keys=["target"], threshold=max, above=False, cval=-1000))
    # filter the source images
    # transform_list.append(ThresholdIntensityd(keys=["source"], threshold=configs.dataset.MRImax, above=False, cval=0))
    if normalize=='zscore':
        transform_list.append(NormalizeIntensityd(keys=["source", "target"], nonzero=False, channel_wise=True))
        print('zscore normalization')
    elif normalize=='minmax':
        transform_list.append(ScaleIntensityd(keys=["source", "target"], minv=-1, maxv=1.0))
        print('minmax normalization')

    elif normalize=='scale4000':
        transform_list.append(ScaleIntensityd(keys=["source"], minv=-1, maxv=1))
        transform_list.append(ScaleIntensityd(keys=["target"], minv=0))
        transform_list.append(ScaleIntensityd(keys=["target"], factor=-0.99975)) # x=x(1+factor)
        print('scale1000 normalization')

    elif normalize=='scale1000':
        transform_list.append(ScaleIntensityd(keys=["source"], minv=0, maxv=1))
        transform_list.append(ScaleIntensityd(keys=["target"], minv=0))
        transform_list.append(ScaleIntensityd(keys=["target"], factor=-0.99975)) 
        print('scale1000 normalization')

    elif normalize=='inputonlyzscore':
        transform_list.append(NormalizeIntensityd(keys=["source"], nonzero=False, channel_wise=True))
        print('only normalize input MRI images')

    elif normalize=='inputonlyminmax':
        transform_list.append(ScaleIntensityd(keys=["source"], minv=configs.dataset.normmin, maxv=configs.dataset.normmax))
        print('only normalize input MRI images')
    elif normalize=='none':
        print('no normalization')
    transform_list.append(Spacingd(keys=["source"], pixdim=(1.0, 1.0, 1.0), mode="bilinear")) # 
    transform_list.append(Spacingd(keys=["target", "mask"], pixdim=(1.0, 1.0 , 2.5), mode="bilinear")) #
    transform_list.append(ResizeWithPadOrCropd(keys=["source", "target", "mask"], spatial_size=resized_size,mode=pad))
    # transform_list.append(ScaleIntensityRanged(keys=["target"],a_min=WINDOW_LEVEL-(WINDOW_WIDTH/2), a_max=WINDOW_LEVEL+(WINDOW_WIDTH/2),b_min=0, b_max=1, clip=True))

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
            transform_list.append(RandAffined(keys=["source", "target", "mask"],padding_mode="border" , prob=prob))
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

def get_length(dataset, patch_batch_size):
    loader=DataLoader(dataset, batch_size=1)
    iterator = iter(loader)
    sum_nslices=0
    for idx in range(len(loader)):
        check_data = next(iterator)
        nslices=check_data['source'].shape[-1]
        sum_nslices+=nslices
    if sum_nslices%patch_batch_size==0:
        return sum_nslices//patch_batch_size
    else:
        return sum_nslices//patch_batch_size+1


