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
    Zoomd,
    ThresholdIntensityd,
    NormalizeIntensityd,
    ShiftIntensityd,
    Identityd,
    ScaleIntensityRanged,
    Spacingd,
    SaveImage,
)
from dataprocesser.customized_transforms import DivideTransformd

from torch.utils.data import ConcatDataset

from abc import ABC, abstractmethod
from dataprocesser import customized_transform_list
from dataprocesser.step3_build_patch_dataset import create_patch_2d, create_patch_3d
from dataprocesser.customized_transforms import (
    MaskHUAssigmentd, 
    CreateMaskWithBonesTransformd)
VERBOSE = False

def make_dataset_modality():
    images = []
    return images

class ABCLoader(ABC):
    @abstractmethod
    def __init__(self):
        """Subclass must implement this method."""
        pass

    def get_dataset_list(self):
        """Subclass must implement this method."""
        pass

    def create_dataset(self):
        """Subclass must implement this method."""
        pass

    def get_transforms(self):
        """Subclass must implement this method."""
        pass

    def get_normlization(self):
        """Subclass must implement this method."""
        pass

    def get_shape_transform(self):
        """Subclass must implement this method."""
        print("no shape transform here!!!!!!!!!!!!!!!!!!!!!!")
        pass

    def get_augmentation(self):
        """Subclass must implement this method."""
        pass

class BaseDataLoader(ABCLoader):
    def __init__(self,configs,paths=None,dimension=2, **kwargs): 
        print('create base dataset')
        self.configs=configs
        self.is_ddp = self.configs.is_ddp
        self.rank = self.configs.rank
        self.world_size = self.configs.world_size
        
        self.paths=paths
        self.indicator_A=self.configs.dataset.indicator_A	
        self.indicator_B=self.configs.dataset.indicator_B
        self.train_number=self.configs.dataset.train_number
        self.val_number=self.configs.dataset.val_number
        self.train_batch_size=self.configs.dataset.batch_size
        self.val_batch_size=self.configs.dataset.val_batch_size
        self.num_workers=self.configs.dataset.num_workers
        self.init_keys()
        
        self.train_transforms = self.get_transforms()
        self.val_transforms = self.get_transforms()
        
        if self.paths is not None:
            self.saved_name_train=self.paths["saved_name_train"]
            self.saved_name_val=self.paths["saved_name_val"]

        self.get_dataset_list()
        #print('all files in dataset:',len(self.source_file_list))
        self.rotation_level = kwargs.get('rotation_level', 0) # Default to no rotation (0)
        self.zoom_level = kwargs.get('zoom_level', 1.0)  # Default to no zoom (1.0)
        self.flip = kwargs.get('flip', 0)  # Default to no flip
        self.create_dataset()
        np.savetxt(self.saved_name_train,self.train_ds,delimiter=',',fmt='%s',newline='\n') # f means format, r means raw string
        np.savetxt(self.saved_name_val,self.val_ds,delimiter=',',fmt='%s',newline='\n') # f means format, r means raw string
        '''ifsave = None if paths is None else True
        finalcheck(self.train_ds, self.val_ds, 
               self.train_volume_ds, self.val_volume_ds,
               self.train_loader, self.val_loader,
               self.train_patch_ds, 
               self.train_batch_size, self.val_batch_size,
               self.saved_name_train, self.saved_name_val, 
               self.indicator_A, self.indicator_B,
            ifsave=ifsave, ifcheck=False,iftest_volumes_pixdim=False)'''


    def get_dataset_list(self):
        self.source_file_list = []
        self.train_ds=[]
        self.val_ds=[]
        
    def init_keys(self):
        print('base dataset use keys:', [self.indicator_A, self.indicator_B])
        self.keys = [self.indicator_A, self.indicator_B]

    def create_volume_dataset(self):
        # load volumes and center crop
        center_crop = self.configs.dataset.center_crop
        print('use keys for creating volume dataset: ', self.keys)
        transformations_crop = [
            LoadImaged(keys=self.keys),
            EnsureChannelFirstd(keys=self.keys),
        ]
        if center_crop>0:
            transformations_crop.append(CenterSpatialCropd(keys=self.keys, roi_size=(-1,-1,center_crop)))
        transformations_crop=Compose(transformations_crop)
        self.train_crop_ds = monai.data.Dataset(data=self.train_ds, transform=transformations_crop)
        self.val_crop_ds = monai.data.Dataset(data=self.val_ds, transform=transformations_crop)

        # load volumes
        self.train_volume_ds = monai.data.Dataset(data=self.train_crop_ds, transform=self.train_transforms) 
        self.val_volume_ds = monai.data.Dataset(data=self.val_crop_ds, transform=self.val_transforms)
    
    def create_dataset(self, dimension=None):
        self.create_volume_dataset()

        if self.configs.model_name in ['pix2pix','cycle_gan','syndiff', 'AttentionUnet']:
            use_patch_loader = True 
        else:
            use_patch_loader = False
            
        if self.configs.model_name in ['pix2pix','cycle_gan','syndiff']:
            val_use_patch = True 
        else:
            val_use_patch = False
        print('model name: ', self.configs.model_name)
        print('val_use_patch: ', val_use_patch)
        if use_patch_loader:
            self.train_loader, self.val_loader, self.train_patch_ds, self.val_patch_ds = create_patch_2d(self.keys, 
                        self.train_volume_ds, self.val_volume_ds, 
                        self.train_batch_size, self.val_batch_size,
                        self.num_workers, val_use_patch, self.is_ddp, self.rank, self.world_size)
        else:
            self.train_loader, self.val_loader = create_patch_3d( 
                        self.train_volume_ds, self.val_volume_ds, 
                        train_batch_size=1, val_batch_size=1,
                        num_workers = self.num_workers, 
                        is_ddp = self.is_ddp, rank=self.rank, world_size=self.world_size)
        
    def get_transforms(self):
        transform_list=[]
        transform_list = self.get_pretransforms(transform_list)
        transform_list = self.get_intensity_transforms(transform_list)
        transform_list = self.get_normlization(transform_list)
        transform_list = self.get_shape_transform(transform_list)
        train_transforms = Compose(transform_list)
        return train_transforms
    
    def get_pretransforms(self, transform_list):
        #print("customized transforms")
        return transform_list
    
    def get_intensity_transforms(self, transform_list):
        WINDOW_LEVEL=self.configs.dataset.WINDOW_LEVEL
        WINDOW_WIDTH=self.configs.dataset.WINDOW_WIDTH
        transform_list = customized_transform_list.add_Windowing_ZeroShift_ContourFilter_single_B_transforms(transform_list, WINDOW_LEVEL, WINDOW_WIDTH, self.indicator_B)
        return transform_list
    
    def get_normlization(self, transform_list):
        normalize=self.configs.dataset.normalize
        transform_list=customized_transform_list.add_normalization_transform_single_B(transform_list, self.indicator_B, normalize)

        return transform_list
    
    # is the same with all the same
    def get_shape_transform(self, transform_list):
        spaceXY=self.configs.dataset.spaceXY
        pad_value=0 #offset*(-1
        if spaceXY>0:
            transform_list.append(Spacingd(keys=self.keys, pixdim=(spaceXY, spaceXY, 2.5), mode="bilinear", ensure_same_shape=True)) 
        
        transform_list.append(Zoomd(keys=self.keys, 
                                    zoom=self.configs.dataset.zoom, keep_size=False, mode='area', padding_mode="constant", value=pad_value))
        transform_list.append(DivisiblePadd(keys=self.keys,
                                            k=self.configs.dataset.div_size, mode="constant", value=pad_value))
        transform_list.append(ResizeWithPadOrCropd(keys=self.keys, 
                                                  spatial_size=self.configs.dataset.resized_size,mode="constant", value=pad_value))

        if self.configs.dataset.rotate:
            transform_list.append(Rotate90d(keys=self.keys, k=3))
        return transform_list

