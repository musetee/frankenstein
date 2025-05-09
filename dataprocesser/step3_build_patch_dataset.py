import monai
from monai.transforms import (
    Compose,
    SqueezeDimd,
    ResizeWithPadOrCropd,
)
from monai.data import PatchIterd, PatchDataset, GridPatchDataset
import torch
from monai.transforms import RandSpatialCropSamplesd
from torch.utils.data import DataLoader, DistributedSampler
def decode_dataset_from_single_volume_batch(data):
    decoded_data = {}
    for key, value in data.items():
        decoded_data[key] = value[0]  # Extract the single element from the batch of size 1
    return decoded_data

class SlicingTransformd:
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, volume_data):
        # 输入 volume_data: 一个 dict，对应一个 volume
        sliced_list = []
        depth = volume_data[self.keys[0]].shape[-1]  # (C, H, W, D)
        for d in range(depth):
            slice_dict = {}
            for key in volume_data:
                if key in self.keys:
                    slice_dict[key] = volume_data[key][..., d]  # shape: (C, H, W)
                else:
                    slice_dict[key] = volume_data[key]  # keep metadata like modality
            sliced_list.append(slice_dict)
        return sliced_list

def patch_2d_from_single_volume(keys, 
                                 train_volume_ds, 
                                 train_batch_size,
                                 num_workers):
    # 每个 volume 变成 slice list
    slicing = SlicingTransformd(keys)
    sliced_data = []
    for item in train_volume_ds:
        sliced = slicing(item)
        sliced_data.extend(sliced)
    
    # 使用普通的 DataLoader
    train_loader = DataLoader(
        sliced_data,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    return train_loader

'''def patch_2d_from_single_volume(keys, 
                    train_volume_ds, 
                    train_batch_size,
                    num_workers):
    # batch-level slicer for both image and label
    patch_func = PatchIterd(
        keys=keys,
        patch_size=(None, None, 1),  # dynamic first two dimensions
        start_pos=(0, 0, 0)
    )
    # must squeezedim, otherwise will produce torch.Size([2, 1, 8, 16, 1])
    patch_transform = Compose(
            [
                SqueezeDimd(keys=keys, dim=-1),  # squeeze the last dim
            ]
        )
    # for training
    #print('\n training use grid patch dataset')
    train_patch_ds = GridPatchDataset(
        data=train_volume_ds, patch_iter=patch_func, transform=patch_transform, with_coordinates=False)
    train_loader = DataLoader(
        train_patch_ds,
        batch_size=train_batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader'''

def patch_2d_from_volume(keys, 
                    train_volume_ds, 
                    train_batch_size,
                    num_workers):
    # batch-level slicer for both image and label
    patch_func = PatchIterd(
        keys=keys,
        patch_size=(None, None, None, 1),  # dynamic first two dimensions
        start_pos=(0, 0, 0)
    )
    # must squeezedim, otherwise will produce torch.Size([2, 1, 8, 16, 1])
    patch_transform = Compose(
            [
                SqueezeDimd(keys=keys, dim=-1),  # squeeze the last dim
                SqueezeDimd(keys=keys, dim=1),  # squeeze the intermediate dim
            ]
        )
    # for training
    #print('\n training use grid patch dataset')
    train_patch_ds = GridPatchDataset(
        data=train_volume_ds, patch_iter=patch_func, transform=patch_transform, with_coordinates=False)
    train_loader = DataLoader(
        train_patch_ds,
        batch_size=train_batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader

def create_patch_2d(keys, 
                    train_volume_ds, val_volume_ds, 
                    train_batch_size,val_batch_size,
                    num_workers, val_use_patch):
    # batch-level slicer for both image and label
    patch_func = PatchIterd(
        keys=keys,
        patch_size=(None, None, 1),  # dynamic first two dimensions
        start_pos=(0, 0, 0)
    )
    # must squeezedim, otherwise will produce torch.Size([2, 1, 8, 16, 1])
    patch_transform = Compose(
            [
                SqueezeDimd(keys=keys, dim=-1),  # squeeze the last dim
            ]
        )
    # for training
    print('training use grid patch dataset')
    train_patch_ds = GridPatchDataset(
        data=train_volume_ds, patch_iter=patch_func, transform=patch_transform, with_coordinates=False)
    train_loader = DataLoader(
        train_patch_ds,
        batch_size=train_batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # for validation
    #if model_name=='ddpm' or 'ddpm2d_seg2med' or 'ddpm2d':
    if val_use_patch:
        print('validation use grid patch dataset')
        val_patch_ds = GridPatchDataset(
        data=val_volume_ds, patch_iter=patch_func, transform=patch_transform, with_coordinates=False)
        val_loader = DataLoader(
            val_patch_ds, #val_volume_ds, 
            num_workers=num_workers, 
            batch_size=val_batch_size,
            pin_memory=torch.cuda.is_available())
    else:
        print('validation use volume dataset')
        val_patch_ds = None
        val_loader = DataLoader(
            val_volume_ds, 
            num_workers=num_workers, 
            batch_size=1,
            pin_memory=torch.cuda.is_available())
    return train_loader, val_loader, train_patch_ds, val_patch_ds

def create_grid_patch_2d(keys, 
                    train_volume_ds, val_volume_ds, 
                    train_batch_size,val_batch_size,
                    num_workers, val_use_patch, 
                    is_ddp=False, rank=0, world_size=1):
    # batch-level slicer for both image and label
    window_width=1
    patch_func = monai.data.PatchIterd(
        keys=keys,
        patch_size=(None, None, window_width),  # dynamic first two dimensions
        start_pos=(0, 0, 0)
    )
    if window_width==1:
        patch_transform = Compose(
            [
                SqueezeDimd(keys=keys, dim=-1),  # squeeze the last dim
            ]
        )
    else:
        patch_transform = None
        
    # for training
    print('training use grid patch dataset')
    train_patch_ds = GridPatchDataset(
        data=train_volume_ds, patch_iter=patch_func, transform=patch_transform, with_coordinates=False)
    ds_for_loader_train = train_patch_ds
    # for validation
    #if model_name=='ddpm' or 'ddpm2d_seg2med' or 'ddpm2d':
    if val_use_patch:
        print('validation use grid patch dataset')
        val_patch_ds = GridPatchDataset(
        data=val_volume_ds, patch_iter=patch_func, transform=patch_transform, with_coordinates=False)
        ds_for_loader_val = val_patch_ds

    else:
        print('validation use volume dataset')
        val_patch_ds = None
        ds_for_loader_val = val_volume_ds
    
    if is_ddp:
        # 多GPU使用 DistributedSampler
        train_sampler = DistributedSampler(ds_for_loader_train, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(ds_for_loader_val, num_replicas=world_size, rank=rank, shuffle=False)

        train_loader = DataLoader(
            ds_for_loader_train,
            batch_size=train_batch_size,
            sampler=train_sampler,
            shuffle=False,  # sampler 不允许 shuffle=True
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        val_loader = DataLoader(
            ds_for_loader_val,
            batch_size=val_batch_size,
            sampler=val_sampler,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
    else:
        # 单GPU默认随机打乱
        train_loader = DataLoader(
            ds_for_loader_train,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        val_loader = DataLoader(
            ds_for_loader_val,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )

    return train_loader, val_loader, train_patch_ds, val_patch_ds

def create_patch_25d(keys, patch_size, 
                    train_volume_ds, val_volume_ds, 
                    train_batch_size,val_batch_size,
                    num_workers, val_use_patch, 
                    is_ddp=False, rank=0, world_size=1):
    # batch-level slicer for both image and label
    # 2.5 means stack slices together as a small volume patch
    # if window_width>1, means we train a 2.5D network
    window_width=patch_size[-1]
    patch_func = monai.data.PatchIterd(
        keys=keys,
        patch_size=patch_size,  # dynamic first two dimensions: (None, None, window_width)
        start_pos=(0, 0, 0)
    )
    if window_width==1:
        print(f"slice patch is 1, we use 2D-training")
        patch_transform = Compose(
            [
                SqueezeDimd(keys=keys, dim=-1),  # squeeze the last dim
            ]
        )
    else:
        print(f"use consecutive {window_width} slices for 2.5D-training")
        # there would be an error if original size < patch_size during training, so we should pad it in this case
        patch_transform = ResizeWithPadOrCropd(keys=keys, 
                                            spatial_size=patch_size, mode='minimum')
        
    # for training
    train_patch_ds = monai.data.GridPatchDataset(
        data=train_volume_ds, patch_iter=patch_func, transform=patch_transform, with_coordinates=False)
    ds_for_loader_train =  train_patch_ds

    # for validation
    if val_use_patch:
        val_patch_ds = monai.data.GridPatchDataset(
        data=val_volume_ds, patch_iter=patch_func, transform=patch_transform, with_coordinates=False)
        ds_for_loader_val = val_patch_ds
    else:
        ds_for_loader_val = val_volume_ds
    
    if is_ddp:
        # 多GPU使用 DistributedSampler
        train_sampler = DistributedSampler(ds_for_loader_train, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(ds_for_loader_val, num_replicas=world_size, rank=rank, shuffle=False)

        train_loader = DataLoader(
            ds_for_loader_train,
            batch_size=train_batch_size,
            sampler=train_sampler,
            shuffle=False,  # sampler 不允许 shuffle=True
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        val_loader = DataLoader(
            ds_for_loader_val,
            batch_size=val_batch_size,
            sampler=val_sampler,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
    else:
        # 单GPU默认随机打乱
        train_loader = DataLoader(
            ds_for_loader_train,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        val_loader = DataLoader(
            ds_for_loader_val,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
    
    return train_loader, val_loader
        
def create_patch_3d(
                    train_volume_ds, val_volume_ds, 
                    train_batch_size,
                    val_batch_size,
                    num_workers, 
                    is_ddp=False, rank=0, world_size=1
                    ):
    if is_ddp:
        # DistributedSampler
        train_sampler = DistributedSampler(train_volume_ds, num_replicas=world_size, rank=rank, shuffle=True) if is_ddp else None
        val_sampler = DistributedSampler(val_volume_ds, num_replicas=world_size, rank=rank, shuffle=False) if is_ddp else None

        # DataLoader
        train_loader = DataLoader(train_volume_ds, batch_size=train_batch_size, num_workers=num_workers,
                                    sampler=train_sampler, persistent_workers=True, pin_memory=True, shuffle=(train_sampler is None))
        val_loader = DataLoader(val_volume_ds, batch_size=val_batch_size, num_workers=num_workers,
                                    sampler=val_sampler, persistent_workers=True, pin_memory=True, shuffle=(val_sampler is None))
    else:
        # 3 means use the whole input volume for training 
        train_loader = DataLoader(
            train_volume_ds, 
            num_workers=num_workers, 
            batch_size=train_batch_size,
            shuffle=True,
            pin_memory=torch.cuda.is_available())
        val_loader = DataLoader(
            val_volume_ds, 
            num_workers=num_workers, 
            batch_size=val_batch_size,
            pin_memory=torch.cuda.is_available())
    return train_loader, val_loader

def create_patch_35d(keys, patch_size, 
                    train_volume_ds, val_volume_ds, 
                    train_batch_size,val_batch_size,
                    num_workers):
    # 3.5 means create patch from the original volume
    patch_func = monai.data.PatchIterd(
        keys=keys,
        patch_size=patch_size,  # dynamic first two dimensions
        start_pos=(0, 0, 0),
        mode="replicate",
    )
    patch_transform = None
        
    # for training
    train_patch_ds = monai.data.GridPatchDataset(
        data=train_volume_ds, patch_iter=patch_func, transform=patch_transform, with_coordinates=False)
    train_loader = DataLoader(
        train_patch_ds,
        batch_size=train_batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_patch_ds = monai.data.GridPatchDataset(
        data=val_volume_ds, patch_iter=patch_func, transform=patch_transform, with_coordinates=False)
    val_loader = DataLoader(
        val_patch_ds, #val_volume_ds, 
        num_workers=num_workers, 
        batch_size=val_batch_size,
        pin_memory=torch.cuda.is_available())
    return train_loader, val_loader