import monai
from monai.data import Dataset
from torch.utils.data import DataLoader
import torch 
from .basics import get_file_list, check_batch_data, get_transforms,crop_volumes, load_volumes
def load_batch_slices3D(train_volume_ds,val_volume_ds, train_batch_size=5,val_batch_size=1,ifcheck=True):
    patch_func = monai.data.PatchIterd(
        keys=["image", "label"],
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

def mydataloader_3d(data_pelvis_path,
                    normalize='zscore',
                    pad='minimum',
                    train_number=10,
                    val_number=2,
                    train_batch_size=1,
                    val_batch_size=1,
                    saved_name_train='./train_ds_2d.csv',
                    saved_name_val='./val_ds_2d.csv',
                    resized_size=(512,512,128),
                    div_size=(16,16,16),
                    ifcheck_volume=True,):
    # volume-level transforms for both image and segmentation
    
    train_transforms = get_transforms(normalize,pad,resized_size,div_size)
    
    train_ds, val_ds = get_file_list(data_pelvis_path, 
                                     train_number, 
                                     val_number)
    #train_volume_ds, val_volume_ds 
    
    train_crop_ds, val_crop_ds = crop_volumes(train_ds, val_ds,center_crop=0)
    train_volume_ds, val_volume_ds = load_volumes(train_transforms, 
                                                train_crop_ds, val_crop_ds, 
                                                train_ds, val_ds, 
                                                saved_name_train, saved_name_val,
                                                ifsave=True,
                                                ifcheck=ifcheck_volume)
    
    train_loader = DataLoader(train_volume_ds, batch_size=train_batch_size)
    val_loader = DataLoader(val_volume_ds, batch_size=val_batch_size)
    '''
    ifcheck_sclices=False
    
    train_loader,val_loader = load_batch_slices3D(train_volume_ds, 
                                                val_volume_ds, 
                                                train_batch_size,
                                                val_batch_size=val_batch_size,
                                                ifcheck=ifcheck_sclices)
    '''                                      
    return train_loader,val_loader,train_transforms