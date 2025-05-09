import monai
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

from .basics import get_file_list, get_transforms, load_volumes, crop_volumes
from .checkdata import check_batch_data, check_volumes, save_volumes
##### slices #####
def load_batch_slices(train_volume_ds,val_volume_ds, train_batch_size=8,val_batch_size=1,window_width=1,ifcheck=True):
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
    return train_loader,val_loader
def myslicesloader(data_pelvis_path,
                   normalize='minmax',
                   pad='minimum',
                   train_number=1,
                   val_number=1,
                   train_batch_size=8,
                   val_batch_size=1,
                   saved_name_train='./train_ds_2d.csv',
                   saved_name_val='./val_ds_2d.csv',
                   resized_size=(512,512,None),
                   div_size=(16,16,None),
                   center_crop=20,
                   ifcheck_volume=True,
                   ifcheck_sclices=False,):
    
    # volume-level transforms for both image and label
    train_transforms = get_transforms(normalize,pad,resized_size,div_size,mode='train',prob=0.8)
    val_transforms = get_transforms(normalize,pad,resized_size,div_size,mode='val')
    train_ds, val_ds = get_file_list(data_pelvis_path, 
                                     train_number, 
                                     val_number)
    train_crop_ds, val_crop_ds = crop_volumes(train_ds, val_ds,center_crop)
    train_ds, val_ds = load_volumes(train_transforms, val_transforms,
                                                train_crop_ds, val_crop_ds, 
                                                train_ds, val_ds, 
                                                saved_name_train, saved_name_val,
                                                ifsave=True,
                                                ifcheck=ifcheck_volume)
    train_loader,val_loader = load_batch_slices(train_ds, 
                                                val_ds, 
                                                train_batch_size,
                                                val_batch_size=val_batch_size,
                                                window_width=1,
                                                ifcheck=ifcheck_sclices)
    return train_ds, val_ds, train_loader,val_loader,train_transforms,val_transforms

def len_patchloader(train_volume_ds,train_batch_size):
    slice_number=sum(train_volume_ds[i]['source'].shape[-1] for i in range(len(train_volume_ds)))
    print('total slices in training set:',slice_number)

    import math
    batch_number=sum(math.ceil(train_volume_ds[i]['source'].shape[-1]/train_batch_size) for i in range(len(train_volume_ds)))
    print('total batches in training set:',batch_number)
    return slice_number,batch_number

if __name__ == '__main__':
    dataset_path=r"F:\yang_Projects\Datasets\Task1\pelvis"
    train_volume_ds,_,train_loader,_,_,_ = myslicesloader(dataset_path,
                    normalize='none',
                    train_number=2,
                    val_number=1,
                    train_batch_size=4,
                    val_batch_size=1,
                    saved_name_train='./train_ds_2d.csv',
                    saved_name_val='./val_ds_2d.csv',
                    resized_size=(512, 512, None),
                    div_size=(16,16,None),
                    ifcheck_volume=False,
                    ifcheck_sclices=False,)
    from tqdm import tqdm
    parameter_file=r'.\test.txt'
    for data in tqdm(train_loader):
         with open(parameter_file, 'a') as f:
            f.write('image batch:' + str(data["image"].shape)+'\n')
            f.write('label batch:' + str(data["label"].shape)+'\n')
            f.write('\n')