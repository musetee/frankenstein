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
)
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
import torch
from abc import ABC, abstractmethod

from datetime import datetime
import json
from tqdm import tqdm

from step1_init_data_list import (
    list_img_ad_from_anish_csv, 
    list_img_ad_pIDs_from_anish_csv,
    list_img_pID_from_synthrad_folder,
    list_from_anika_dataset,
    list_from_json,
    list_from_slice_csv,
    )
from step5_data_check_and_log import finalcheck
VERBOSE = False

def make_dataset_modality():
    images = []
    return images

class ABCLoader(ABC):
    @abstractmethod
    def __init__(self):
        """Subclass must implement this method."""
        pass

    def get_loader(self):
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
        self.configs=configs
        self.paths=paths
        self.init_parameters_and_transforms()
        self.get_loader()
        #print('all files in dataset:',len(self.source_file_list))
        

        self.rotation_level = kwargs.get('rotation_level', 0) # Default to no rotation (0)
        self.zoom_level = kwargs.get('zoom_level', 1.0)  # Default to no zoom (1.0)
        self.flip = kwargs.get('flip', 0)  # Default to no flip

        self.create_dataset(dimension=dimension)

        ifsave = None if paths is None else True
        finalcheck(self.train_ds, self.val_ds, 
               self.train_volume_ds, self.val_volume_ds,
               self.train_loader, self.val_loader,
               self.train_patch_ds, 
               self.train_batch_size, self.val_batch_size,
               self.saved_name_train, self.saved_name_val, 
               self.indicator_A, self.indicator_B,
            ifsave=ifsave, ifcheck=False,iftest_volumes_pixdim=False)

    def get_loader(self):
        self.source_file_list = []
        self.train_ds=[]
        self.val_ds=[]
        

    def init_parameters_and_transforms(self):
        self.indicator_A=self.configs.dataset.indicator_A	
        self.indicator_B=self.configs.dataset.indicator_B
        self.train_number=self.configs.dataset.train_number
        self.val_number=self.configs.dataset.val_number
        self.train_batch_size=self.configs.dataset.batch_size
        self.val_batch_size=self.configs.dataset.val_batch_size
        self.load_masks=self.configs.dataset.load_masks

        self.keys = [self.indicator_A, self.indicator_B, "mask"] if self.load_masks else [self.indicator_A, self.indicator_B]

        if self.configs.model_name=='augmentation':
            # Fixed parameters for rotation and zooming
            self.train_transforms = self.get_augmentation(transform_list=[], flip=self.flip, rotation_level=self.rotation_level, zoom_level=self.zoom_level)
        else:
            self.train_transforms = self.get_transforms(mode='train')
        self.val_transforms = self.get_transforms(mode='val')
        
        if self.paths is not None:
            self.saved_name_train=self.paths["saved_name_train"]
            self.saved_name_val=self.paths["saved_name_val"]

    def create_volume_dataset(self):
        # load volumes and center crop
        center_crop = self.configs.dataset.center_crop
        transformations_crop = [
            LoadImaged(keys=self.keys),
            EnsureChannelFirstd(keys=self.keys),
        ]
        if center_crop>0:
            transformations_crop.append(CenterSpatialCropd(keys=self.keys, roi_size=(-1,-1,center_crop)))
        transformations_crop=Compose(transformations_crop)
        train_crop_ds = monai.data.Dataset(data=self.train_ds, transform=transformations_crop)
        val_crop_ds = monai.data.Dataset(data=self.val_ds, transform=transformations_crop)

        # load volumes
        self.train_volume_ds = monai.data.Dataset(data=train_crop_ds, transform=self.train_transforms) 
        self.val_volume_ds = monai.data.Dataset(data=val_crop_ds, transform=self.val_transforms)
    
    def create_patch_dataset_and_dataloader(self, dimension=2):
        train_batch_size=self.configs.dataset.batch_size
        val_batch_size=self.configs.dataset.val_batch_size
        if dimension==2:
            # batch-level slicer for both image and label
            window_width=1
            patch_func = monai.data.PatchIterd(
                keys=self.keys,
                patch_size=(None, None, window_width),  # dynamic first two dimensions
                start_pos=(0, 0, 0)
            )
            if window_width==1:
                patch_transform = Compose(
                    [
                        SqueezeDimd(keys=self.keys, dim=-1),  # squeeze the last dim
                    ]
                )
            else:
                patch_transform = None
                
            # for training
            train_patch_ds = monai.data.GridPatchDataset(
                data=self.train_volume_ds, patch_iter=patch_func, transform=patch_transform, with_coordinates=False)
            train_loader = DataLoader(
                train_patch_ds,
                batch_size=train_batch_size,
                num_workers=self.configs.dataset.num_workers,
                pin_memory=torch.cuda.is_available(),
            )

            # for validation
            if self.configs.model_name=='ddpm' or 'ddpm2d_seg2med' or 'ddpm2d':
                val_patch_ds = monai.data.GridPatchDataset(
                data=self.val_volume_ds, patch_iter=patch_func, transform=patch_transform, with_coordinates=False)
                val_loader = DataLoader(
                    val_patch_ds, #val_volume_ds, 
                    num_workers=self.configs.dataset.num_workers, 
                    batch_size=val_batch_size,
                    pin_memory=torch.cuda.is_available())
            else:
                val_loader = DataLoader(
                    self.val_volume_ds, 
                    num_workers=self.configs.dataset.num_workers, 
                    batch_size=val_batch_size,
                    pin_memory=torch.cuda.is_available())
            self.train_patch_ds=train_patch_ds

        elif dimension==2.5:
            # batch-level slicer for both image and label
            # 2.5 means stack slices together as a small volume patch
            # if window_width>1, means we train a 2.5D network
            patch_size=self.configs.dataset.patch_size # (None, None, window_width)
            window_width=patch_size[-1]
            patch_func = monai.data.PatchIterd(
                keys=self.keys,
                patch_size=patch_size,  # dynamic first two dimensions: (None, None, window_width)
                start_pos=(0, 0, 0)
            )
            if window_width==1:
                print(f"slice patch is 1, we use 2D-training")
                patch_transform = Compose(
                    [
                        SqueezeDimd(keys=self.keys, dim=-1),  # squeeze the last dim
                    ]
                )
            else:
                print(f"use consecutive {window_width} slices for 2.5D-training")
                # there would be an error if original size < patch_size during training, so we should pad it in this case
                patch_transform = ResizeWithPadOrCropd(keys=self.keys, 
                                                  spatial_size=patch_size, mode='minimum')
                
            # for training
            train_patch_ds = monai.data.GridPatchDataset(
                data=self.train_volume_ds, patch_iter=patch_func, transform=patch_transform, with_coordinates=False)
            train_loader = DataLoader(
                train_patch_ds,
                batch_size=train_batch_size,
                num_workers=2,
                pin_memory=torch.cuda.is_available(),
            )

            # for validation
            if self.configs.model_name=='ddpm':
                val_patch_ds = monai.data.GridPatchDataset(
                data=self.val_volume_ds, patch_iter=patch_func, transform=patch_transform, with_coordinates=False)
                val_loader = DataLoader(
                    val_patch_ds, #val_volume_ds, 
                    num_workers=0, 
                    batch_size=val_batch_size,
                    pin_memory=torch.cuda.is_available())
            else:
                val_loader = DataLoader(
                    self.val_volume_ds, 
                    num_workers=1, 
                    batch_size=val_batch_size,
                    pin_memory=torch.cuda.is_available())
            self.train_patch_ds=train_patch_ds
            
        elif dimension==3:
            # 3 means use the whole input volume for training 
            train_loader = DataLoader(
                self.train_volume_ds, 
                num_workers=self.configs.dataset.num_workers, 
                batch_size=train_batch_size,
                pin_memory=torch.cuda.is_available())
            val_loader = DataLoader(
                self.val_volume_ds, 
                num_workers=self.configs.dataset.num_workers, 
                batch_size=val_batch_size,
                pin_memory=torch.cuda.is_available())
                
        elif dimension==3.5: 
            # 3.5 means create patch from the original volume
            patch_func = monai.data.PatchIterd(
                keys=[self.indicator_A, self.indicator_B],
                patch_size=self.configs.dataset.patch_size,  # dynamic first two dimensions
                start_pos=(0, 0, 0),
                mode="replicate",
            )
            patch_transform = None
                
            # for training
            train_patch_ds = monai.data.GridPatchDataset(
                data=self.train_volume_ds, patch_iter=patch_func, transform=patch_transform, with_coordinates=False)
            train_loader = DataLoader(
                train_patch_ds,
                batch_size=train_batch_size,
                num_workers=self.configs.dataset.num_workers,
                pin_memory=torch.cuda.is_available(),
            )

            val_patch_ds = monai.data.GridPatchDataset(
                data=self.val_volume_ds, patch_iter=patch_func, transform=patch_transform, with_coordinates=False)
            val_loader = DataLoader(
                val_patch_ds, #val_volume_ds, 
                num_workers=self.configs.dataset.num_workers, 
                batch_size=val_batch_size,
                pin_memory=torch.cuda.is_available())
        else:
            print('dimension of input data must be 2 or 2.5 or 3 or 3.5!')

        

        self.train_batch_size=train_batch_size
        self.val_batch_size=val_batch_size
        self.train_loader=train_loader
        self.val_loader=val_loader

    def create_dataset(self,dimension=2):
        self.create_volume_dataset()
        self.create_patch_dataset_and_dataloader(dimension=dimension)

    def get_transforms(self, mode='train'):
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
        threshold_low=self.configs.dataset.WINDOW_LEVEL - self.configs.dataset.WINDOW_WIDTH / 2
        threshold_high=self.configs.dataset.WINDOW_LEVEL + self.configs.dataset.WINDOW_WIDTH / 2
        offset=(-1)*threshold_low
        # if filter out the pixel with values below threshold1, set above=True, and the cval1>=threshold1, otherwise there will be problem
        # mask = img > self.threshold if self.above else img < self.threshold
        # res = where(mask, img, self.cval)
        transform_list.append(ThresholdIntensityd(keys=[self.indicator_B], threshold=threshold_low, above=True, cval=threshold_low))
        transform_list.append(ThresholdIntensityd(keys=[self.indicator_B], threshold=threshold_high, above=False, cval=threshold_high))
        transform_list.append(ShiftIntensityd(keys=[self.indicator_B], offset=offset))
        return transform_list
    
    def get_normlization(self, transform_list):
        normalize=self.configs.dataset.normalize
        indicator_A=self.configs.dataset.indicator_A
        indicator_B=self.configs.dataset.indicator_B
        # offset = self.configs.dataset.offset
        # we don't need normalization for segmentation mask
        if normalize=='zscore':
            transform_list.append(NormalizeIntensityd(keys=[indicator_B], nonzero=False, channel_wise=True))
            print('zscore normalization')
        elif normalize=='minmax':
            transform_list.append(ScaleIntensityd(keys=[indicator_B], minv=-1.0, maxv=1.0))
            print('minmax normalization')

        elif normalize=='scale1000_wrongbutworks':
            transform_list.append(ScaleIntensityd(keys=[indicator_B], minv=0))
            transform_list.append(ScaleIntensityd(keys=[indicator_B], factor=-0.999)) 
            print('scale1000 normalization')

        elif normalize=='scale4000':
            transform_list.append(ScaleIntensityd(keys=[indicator_B], minv=None, maxv=None, factor=-0.99975))
            print('scale4000 normalization')

        elif normalize=='scale2000':
            transform_list.append(ScaleIntensityd(keys=[indicator_B], minv=None, maxv=None, factor=-0.9995))
            print('scale2000 normalization')

        elif normalize=='scale1000':
            transform_list.append(ScaleIntensityd(keys=[indicator_B], minv=None, maxv=None, factor=-0.999)) 
            print('scale1000 normalization')
        
        elif normalize=='scale100':
            transform_list.append(ScaleIntensityd(keys=[indicator_B], minv=None, maxv=None,factor=-0.99)) 
            print('scale10 normalization')

        elif normalize=='scale10':
            transform_list.append(ScaleIntensityd(keys=[indicator_B], minv=None, maxv=None,factor=-0.9)) 
            print('scale10 normalization')

        elif normalize=='inputonlyzscore':
            transform_list.append(NormalizeIntensityd(keys=[indicator_A], nonzero=False, channel_wise=True))
            print('only normalize input MRI images')

        elif normalize=='inputonlyminmax':
            transform_list.append(ScaleIntensityd(keys=[indicator_A], minv=self.configs.dataset.normmin, maxv=self.configs.dataset.normmax))
            print('only normalize input MRI images')

        elif normalize == 'nonegative':
            transform_list.append(ShiftIntensityd(keys=[indicator_B], offset=self.configs.dataset.offset))
            print('none negative normalization')

        elif normalize=='none' or normalize=='nonorm':
            print('no normalization')

        return transform_list
    
    def get_shape_transform(self, transform_list):
        spaceXY=self.configs.dataset.spaceXY
        load_masks=self.configs.dataset.load_masks

        indicator_A=self.configs.dataset.indicator_A
        indicator_B=self.configs.dataset.indicator_B
        pad_value=0 #offset*(-1)
        keys = self.keys #[indicator_A, indicator_B, "mask"] if load_masks else [indicator_A, indicator_B]
        if spaceXY>0:
            transform_list.append(Spacingd(keys=[indicator_A], pixdim=(spaceXY, spaceXY, 2.5), mode="bilinear", ensure_same_shape=True)) # 
            transform_list.append(Spacingd(keys=[indicator_B, "mask"] if load_masks else [indicator_B], 
                                           pixdim=(spaceXY, spaceXY , 2.5), mode="bilinear", ensure_same_shape=True))
        
        transform_list.append(Zoomd(keys=keys, 
                                    zoom=self.configs.dataset.zoom, keep_size=False, mode='area', padding_mode="constant", value=pad_value))
        transform_list.append(DivisiblePadd(keys=keys,
                                            k=self.configs.dataset.div_size, mode="constant", value=pad_value))
        transform_list.append(ResizeWithPadOrCropd(keys=keys, 
                                                  spatial_size=self.configs.dataset.resized_size,mode="constant", value=pad_value))

        if self.configs.dataset.rotate:
            transform_list.append(Rotate90d(keys=keys, k=3))
        return transform_list

class anish_loader(BaseDataLoader):
    def __init__(self,configs,paths,dimension=2): 
        self.configs=configs
        self.paths=paths
        self.get_loader()
        super().create_dataset(dimension=dimension)
        self.finalcheck(ifsave=True,ifcheck=False,iftest_volumes_pixdim=False)

    def get_loader(self):
        indicator_A=self.configs.dataset.indicator_A	
        indicator_B=self.configs.dataset.indicator_B
        self.indicator_A=indicator_A
        self.indicator_B=indicator_B
        train_number=self.configs.dataset.train_number
        val_number=self.configs.dataset.val_number
        train_batch_size=self.configs.dataset.batch_size
        val_batch_size=self.configs.dataset.val_batch_size
        load_masks=self.configs.dataset.load_masks

        
        #source_file_list=[os.path.join(j,f'{self.configs.dataset.source_name}.nii.gz') for j in file_list_path] # "ct" for example
        #target_file_list=[os.path.join(j,f'{self.configs.dataset.target_name}.nii.gz') for j in file_list_path] # "mr" for example
        #mask_file_list=[os.path.join(j,f'{self.configs.dataset.mask_name}.nii.gz') for j in file_list_path]
        if self.configs.dataset.data_dir is not None and os.path.exists(self.configs.dataset.data_dir):
            # check if import data is csv file
            if self.configs.dataset.data_dir.endswith('.csv'):
                csv_file = self.configs.dataset.data_dir
            else:
                raise ValueError('The data directory in this case must be a csv file!')
        else:
            if self.configs.server == 'helix' or self.configs.server == 'helixSingle' or self.configs.server=='helixMultiple':
                csv_file = './healthy_dissec_helix.csv'
            else:
                csv_file = './healthy_dissec.csv'

        if self.configs.dataset.input_is_mask:
            load_seg=True
        else:
            load_seg=False
        source_file_list, source_Aorta_diss_list=list_img_ad_from_anish_csv(csv_file, load_seg)
        target_file_list, target_Aorta_diss_list=list_img_ad_from_anish_csv(csv_file)
        mask_file_list, mask_Aorta_diss_list=list_img_ad_from_anish_csv(csv_file)
        if load_masks:  
            train_ds = [{indicator_A: i, indicator_B: j, 'mask': k, 'A_paths': i, 'B_paths': j, 'mask_path': k} 
                        for i, j, k in zip(source_file_list[0:train_number], target_file_list[0:train_number], mask_file_list[0:train_number])]
            val_ds = [{indicator_A: i, indicator_B: j, 'mask': k, 'A_paths': i, 'B_paths': j, 'mask_path': k} 
                    for i, j, k in zip(source_file_list[-val_number:], target_file_list[-val_number:], mask_file_list[-val_number:])]
        else:
            train_ds = [{indicator_A: i, indicator_B: j, 'A_paths': i, 'B_paths': j, 'Aorta_diss':ad} 
                        for i, j, ad in zip(source_file_list[0:train_number], target_file_list[0:train_number], source_Aorta_diss_list[0:train_number])]
            val_ds = [{indicator_A: i, indicator_B: j, 'A_paths': i, 'B_paths': j, 'Aorta_diss':ad} 
                    for i, j, ad in zip(source_file_list[-val_number:], target_file_list[-val_number:], source_Aorta_diss_list[-val_number:])]
        self.train_ds=train_ds
        self.val_ds=val_ds
        self.source_file_list=source_file_list
        self.target_file_list=target_file_list
        self.mask_file_list=mask_file_list  

    def get_pretransforms(self, transform_list):
        normalize=self.configs.dataset.normalize
        indicator_A=self.configs.dataset.indicator_A
        indicator_B=self.configs.dataset.indicator_B
        load_masks=self.configs.dataset.load_masks
        input_is_mask=self.configs.dataset.input_is_mask
        if not input_is_mask:
            transform_list.append(CreateMaskTransformd(keys=[indicator_A],
                                                    tissue_min=self.configs.dataset.tissue_min,
                                                    tissue_max=self.configs.dataset.tissue_max,
                                                    bone_min=self.configs.dataset.bone_min,
                                                    bone_max=self.configs.dataset.bone_max))

from dataprocesser.customized_transforms import CreateMaskTransformd, MergeMasksTransformd

class synthrad_seg_loader(BaseDataLoader):
    def __init__(self,configs,paths,dimension=2,**kwargs): 
        super().__init__(configs,paths,dimension,**kwargs)
        
    def get_loader(self):
        # volume-level transforms for both image and label
        indicator_A=self.configs.dataset.indicator_A	
        indicator_B=self.configs.dataset.indicator_B
        train_number=self.configs.dataset.train_number
        val_number=self.configs.dataset.val_number
        self.indicator_A=indicator_A
        self.indicator_B=indicator_B
        load_masks=self.configs.dataset.load_masks
        # Conditional dictionary keys based on whether masks are loaded
        
        #list all files in the folder
        file_list=[i for i in os.listdir(self.configs.dataset.data_dir) if 'overview' not in i]
        file_list_path=[os.path.join(self.configs.dataset.data_dir,i) for i in file_list]
        #list all ct and mr files in folder
        
        
        # mask file means the images are used for extracting body contour, see get_pretransforms() below
        source_file_list, patient_IDs=list_img_pID_from_synthrad_folder(self.configs.dataset.data_dir, accepted_modalities=self.configs.dataset.source_name, saved_name=os.path.join(self.paths["saved_logs_folder"],"source_filenames.txt"))
        target_file_list, _=list_img_pID_from_synthrad_folder(self.configs.dataset.data_dir, accepted_modalities=self.configs.dataset.target_name, saved_name=os.path.join(self.paths["saved_logs_folder"],"target_filenames.txt"))
        mask_file_list, _=list_img_pID_from_synthrad_folder(self.configs.dataset.data_dir, accepted_modalities=self.configs.dataset.target_name, saved_name=os.path.join(self.paths["saved_logs_folder"],"mask_filenames.txt"))

        self.source_file_list=source_file_list
        self.target_file_list=target_file_list
        self.mask_file_list=mask_file_list
        
        Manual_Set_Aorta_Diss = 0
        ad = Manual_Set_Aorta_Diss
        train_ds = [{indicator_A: i, indicator_B: j, 'mask': k, 'A_paths': i, 'B_paths': j, 'mask_path': k, 'Aorta_diss':ad, 'patient_ID': pID} 
                    for i, j, k, pID in zip(source_file_list[0:train_number], target_file_list[0:train_number], mask_file_list[0:train_number], patient_IDs[0:train_number])]
        val_ds = [{indicator_A: i, indicator_B: j, 'mask': k, 'A_paths': i, 'B_paths': j, 'mask_path': k, 'Aorta_diss':ad, 'patient_ID': pID} 
                for i, j, k, pID in zip(source_file_list[-val_number:], target_file_list[-val_number:], mask_file_list[-val_number:], patient_IDs[-val_number:])]
        self.train_ds=train_ds
        self.val_ds=val_ds

    def get_pretransforms(self, transform_list):
        indicator_A=self.configs.dataset.indicator_A
        indicator_B=self.configs.dataset.indicator_B

        transform_list.append(CreateMaskTransformd(keys=['mask'],
                                                        body_threshold=-500,
                                                        body_mask_value=1,
                                                        ))
        transform_list.append(MergeMasksTransformd(keys=[indicator_A, 'mask']))
        return transform_list
    

from dataprocesser.customized_transforms import CreateMaskTransformd, MergeMasksTransformd, MaskHUAssigmentd

from monai.transforms import (
    ScaleIntensityd,
    ThresholdIntensityd,
    NormalizeIntensityd,
    ShiftIntensityd,
)

class anish_seg_loader(BaseDataLoader):
    def __init__(self,configs,paths=None,dimension=2, **kwargs): 
        super().__init__(configs,paths,dimension, **kwargs)

    def get_loader(self):
        indicator_A=self.configs.dataset.indicator_A	
        indicator_B=self.configs.dataset.indicator_B
        self.indicator_A=indicator_A
        self.indicator_B=indicator_B
        train_number=self.configs.dataset.train_number
        val_number=self.configs.dataset.val_number
        train_batch_size=self.configs.dataset.batch_size
        val_batch_size=self.configs.dataset.val_batch_size
        load_masks=self.configs.dataset.load_masks

        
        #source_file_list=[os.path.join(j,f'{self.configs.dataset.source_name}.nii.gz') for j in file_list_path] # "ct" for example
        #target_file_list=[os.path.join(j,f'{self.configs.dataset.target_name}.nii.gz') for j in file_list_path] # "mr" for example
        #mask_file_list=[os.path.join(j,f'{self.configs.dataset.mask_name}.nii.gz') for j in file_list_path]
        print('use csv dataset:',self.configs.dataset.data_dir)
        if self.configs.dataset.data_dir is not None and os.path.exists(self.configs.dataset.data_dir):
            # check if import data is csv file
            if self.configs.dataset.data_dir.endswith('.csv'):
                csv_file = self.configs.dataset.data_dir
            else:
                raise ValueError('The data directory in this case must be a csv file!')
        else:
            if self.configs.server == 'helix' or self.configs.server == 'helixSingle' or self.configs.server=='helixMultiple':
                csv_file = './healthy_dissec_helix.csv'
            else:
                csv_file = './healthy_dissec.csv'

        if self.configs.dataset.input_is_mask:
            load_seg=True
        else:
            load_seg=False
        source_file_list, source_Aorta_diss_list, patient_IDs=list_img_ad_pIDs_from_anish_csv(csv_file, load_seg)
        target_file_list, _, _ =list_img_ad_pIDs_from_anish_csv(csv_file)
        mask_file_list, _, _=list_img_ad_pIDs_from_anish_csv(csv_file)

        # here the original CT images are loaded as mask because they will be further processed as body contour and merged into mask.

        if load_masks:  
            train_ds = [{indicator_A: i, indicator_B: j, 'mask': k, 'A_paths': i, 'B_paths': j, 'mask_path': k, 'Aorta_diss':ad, 'patient_ID': pID} 
                        for i, j, k, ad, pID in zip(source_file_list[0:train_number], target_file_list[0:train_number], mask_file_list[0:train_number], source_Aorta_diss_list[0:train_number], patient_IDs[0:train_number])]
            
            val_ds = [{indicator_A: i, indicator_B: j, 'mask': k, 'A_paths': i, 'B_paths': j, 'mask_path': k, 'Aorta_diss':ad, 'patient_ID': pID} 
                    for i, j, k, ad, pID in zip(source_file_list[-val_number:], target_file_list[-val_number:], mask_file_list[-val_number:], source_Aorta_diss_list[-val_number:], patient_IDs[-val_number:])]
        else:
            train_ds = [{indicator_A: i, indicator_B: j, 'A_paths': i, 'B_paths': j, 'Aorta_diss':ad} 
                        for i, j, ad in zip(source_file_list[0:train_number], target_file_list[0:train_number], source_Aorta_diss_list[0:train_number])]
            val_ds = [{indicator_A: i, indicator_B: j, 'A_paths': i, 'B_paths': j, 'Aorta_diss':ad} 
                    for i, j, ad in zip(source_file_list[-val_number:], target_file_list[-val_number:], source_Aorta_diss_list[-val_number:])]
        print('train_ds: \n')
        for i in train_ds:
            print(i)
            print('\n')
        self.train_ds=train_ds
        self.val_ds=val_ds
        self.source_file_list=source_file_list
        self.target_file_list=target_file_list
        self.mask_file_list=mask_file_list
        
    def get_pretransforms(self, transform_list):
        indicator_A=self.configs.dataset.indicator_A
        indicator_B=self.configs.dataset.indicator_B

        transform_list.append(CreateMaskTransformd(keys=['mask'],
                                                        body_threshold=-500,
                                                        body_mask_value=1,
                                                        ))
        transform_list.append(MergeMasksTransformd(keys=[indicator_A, 'mask']))
        return transform_list
    

class combined_seg_loader(BaseDataLoader):
    def __init__(self,configs,paths,dimension=2,**kwargs): 
        self.dimension = dimension
        self.train_number_1 = kwargs.get('train_number_1', 170) 
        self.train_number_2 = kwargs.get('train_number_2', 152)  
        self.val_number_1 = kwargs.get('val_number_1', 10) 
        self.val_number_2 = kwargs.get('val_number_2', 10)  
        self.data_dir_1 = kwargs.get('data_dir_1', 'E:\Projects\yang_proj\data\synthrad\Task1\pelvis')
        self.data_dir_2 = kwargs.get('data_dir_2', 'E:\Projects\yang_proj\SynthRad_GAN\synthrad_conversion\healthy_dissec.csv')
        super().__init__(configs,paths,dimension,**kwargs)
        

    def get_loader(self):
        # define the dataset sizes for the dataset 1
        self.configs.dataset.data_dir = self.data_dir_1
        self.configs.dataset.train_number = self.train_number_1
        self.configs.dataset.val_number = self.val_number_1
        self.configs.dataset.source_name = ["ct_seg"]
        self.configs.dataset.target_name = ["ct"]
        self.configs.dataset.offset = 1024
        loader1 = synthrad_seg_loader(self.configs,self.paths,self.dimension)
        source_file_list1 = loader1.source_file_list

        # define the dataset sizes for the dataset 2
        self.configs.dataset.data_dir = self.data_dir_2
        self.configs.dataset.train_number = self.train_number_2
        self.configs.dataset.val_number = self.val_number_2
        self.configs.dataset.offset = 1000
        loader2 = anish_seg_loader(self.configs,self.paths,self.dimension)
        source_file_list2 = loader2.source_file_list

        train_ds1 = loader1.train_ds
        train_ds2 = loader2.train_ds

        val_ds1 = loader1.val_ds
        val_ds2 = loader2.val_ds

        self.train_ds = ConcatDataset([train_ds1, train_ds2])
        self.val_ds = ConcatDataset([val_ds1, val_ds2])
        self.source_file_list = source_file_list1+source_file_list2
    def get_pretransforms(self, transform_list):
        indicator_A=self.configs.dataset.indicator_A
        indicator_B=self.configs.dataset.indicator_B

        transform_list.append(CreateMaskTransformd(keys=['mask'],
                                                        body_threshold=-500,
                                                        body_mask_value=1,
                                                        ))
        transform_list.append(MergeMasksTransformd(keys=[indicator_A, 'mask']))
        return transform_list

    def save_nifti(self, save_output_path, case=0):
        from monai.transforms import SaveImage
        step = 0
        with torch.no_grad():
            for data in self.train_loader:
                si_input = SaveImage(output_dir=f'{save_output_path}',
                    separate_folder=False,
                    output_postfix=f'', # aug_{step}
                    resample=False)
                si_seg = SaveImage(output_dir=f'{save_output_path}',
                    separate_folder=False,
                    output_postfix=f'', # aug_{step}
                    resample=False)
                
                image_batch = data['img'].squeeze()
                seg_batch = data['seg'].squeeze()
                file_path_batch = data['B_paths']
                Aorta_diss = data['Aorta_diss']

                batch_size = len(file_path_batch)

                for i in range(batch_size):
                    step += 1

                    file_path = file_path_batch[i]
                    image = image_batch[i]
                    seg = seg_batch[i]

                    patient_ID = os.path.splitext(os.path.basename(file_path))[0]
                    save_name_img = patient_ID + str(case) + '_' + str(step)
                    save_name_img = os.path.join(save_output_path, save_name_img)

                    save_name_seg = patient_ID + str(case) + '_' + str(step) + '_seg'
                    save_name_seg = os.path.join(save_output_path, save_name_seg)
                    
                    si_input(image.unsqueeze(0), data['img'].meta, filename=save_name_img)
                    si_seg(seg.unsqueeze(0), data['seg'].meta, filename=save_name_seg)

class combined_seg_assigned_loader(combined_seg_loader):
    def __init__(self,configs,paths=None,dimension=2, **kwargs): 
        self.anatomy_list = kwargs.get('anatomy_list', 'synthrad_conversion/TA2_anatomy.csv')
        super().__init__(configs, paths, dimension, **kwargs)
        
    def get_pretransforms(self, transform_list):
        indicator_A=self.configs.dataset.indicator_A
        indicator_B=self.configs.dataset.indicator_B

        transform_list.append(CreateMaskTransformd(keys=['mask'],
                                                        body_threshold=-500,
                                                        body_mask_value=1,
                                                        ))
        transform_list.append(MergeMasksTransformd(keys=[indicator_A, 'mask']))
        transform_list.append(MaskHUAssigmentd(keys=[self.indicator_A], csv_file=self.anatomy_list))
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
    
class slices_nifti_DataLoader(BaseDataLoader):
    def __init__(self,configs,paths=None,dimension=2, **kwargs): 
        super().__init__(configs, paths, dimension, **kwargs)

    def get_loader(self):
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
            num_workers=self.configs.dataset.num_workers, 
            batch_size=train_batch_size,
            shuffle=True,
            pin_memory=torch.cuda.is_available())
        
        self.val_loader = DataLoader(
            self.val_volume_ds, 
            num_workers=self.configs.dataset.num_workers, 
            batch_size=val_batch_size,
            shuffle=False,
            pin_memory=torch.cuda.is_available())
        
class csv_slices_DataLoader(BaseDataLoader):
    def __init__(self,configs,paths=None,dimension=2, **kwargs): 
        super().__init__(configs, paths, dimension, **kwargs)

    def get_loader(self):
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
            num_workers=self.configs.dataset.num_workers, 
            batch_size=train_batch_size,
            shuffle=True,
            pin_memory=torch.cuda.is_available())
        
        self.val_loader = DataLoader(
            self.val_volume_ds, 
            num_workers=self.configs.dataset.num_workers, 
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

# for MRI -> CT task

class synthrad_mr2ct_loader(BaseDataLoader):
    def __init__(self,configs,paths=None,dimension=2): 
        super().__init__(configs,paths,dimension)
        
    def get_loader(self):
        # volume-level transforms for both image and label
        indicator_A=self.configs.dataset.indicator_A	
        indicator_B=self.configs.dataset.indicator_B
        train_number=self.configs.dataset.train_number
        val_number=self.configs.dataset.val_number
        self.indicator_A=indicator_A
        self.indicator_B=indicator_B
        load_masks=self.configs.dataset.load_masks
        # Conditional dictionary keys based on whether masks are loaded
        
        #list all files in the folder
        file_list=[i for i in os.listdir(self.configs.dataset.data_dir) if 'overview' not in i]
        file_list_path=[os.path.join(self.configs.dataset.data_dir,i) for i in file_list]
        #list all ct and mr files in folder
        
        
        #source_file_list=[os.path.join(j,f'{self.configs.dataset.source_name}.nii.gz') for j in file_list_path] # "ct" for example
        #target_file_list=[os.path.join(j,f'{self.configs.dataset.target_name}.nii.gz') for j in file_list_path] # "mr" for example
        #mask_file_list=[os.path.join(j,f'{self.configs.dataset.mask_name}.nii.gz') for j in file_list_path]
        source_file_list,_=list_img_pID_from_synthrad_folder(self.configs.dataset.data_dir, accepted_modalities=self.configs.dataset.source_name,saved_name=None)
        target_file_list,_=list_img_pID_from_synthrad_folder(self.configs.dataset.data_dir, accepted_modalities=self.configs.dataset.target_name,saved_name=None)
        mask_file_list,_=list_img_pID_from_synthrad_folder(self.configs.dataset.data_dir, accepted_modalities=self.configs.dataset.mask_name,saved_name=None)

        def write_write_file(images, file):
            with open(file,"w") as file:
                for image in images:
                    file.write(f'{image} \n')
                    
        if self.paths is not None:
            write_write_file(source_file_list, os.path.join(self.paths["saved_logs_folder"],"source_filenames.txt"))
            write_write_file(target_file_list, os.path.join(self.paths["saved_logs_folder"],"target_filenames.txt"))
            write_write_file(mask_file_list, os.path.join(self.paths["saved_logs_folder"],"mask_filenames.txt"))

        self.source_file_list=source_file_list
        self.target_file_list=target_file_list
        self.mask_file_list=mask_file_list
        
        if load_masks:  
            train_ds = [{indicator_A: i, indicator_B: j, 'mask': k, 'A_paths': i, 'B_paths': j, 'mask_path': k} 
                        for i, j, k in zip(source_file_list[0:train_number], target_file_list[0:train_number], mask_file_list[0:train_number])]
            val_ds = [{indicator_A: i, indicator_B: j, 'mask': k, 'A_paths': i, 'B_paths': j, 'mask_path': k} 
                    for i, j, k in zip(source_file_list[-val_number:], target_file_list[-val_number:], mask_file_list[-val_number:])]
        else:
            train_ds = [{indicator_A: i, indicator_B: j, 'A_paths': i, 'B_paths': j} 
                        for i, j in zip(source_file_list[0:train_number], target_file_list[0:train_number])]
            val_ds = [{indicator_A: i, indicator_B: j, 'A_paths': i, 'B_paths': j} 
                    for i, j in zip(source_file_list[-val_number:], target_file_list[-val_number:])]
        self.train_ds=train_ds
        self.val_ds=val_ds

    def get_normlization(self, transform_list):
        normalize=self.configs.dataset.normalize
        indicator_A=self.configs.dataset.indicator_A
        indicator_B=self.configs.dataset.indicator_B
        load_masks=self.configs.dataset.load_masks
        if normalize=='zscore':
            transform_list.append(NormalizeIntensityd(keys=[indicator_A, indicator_B], nonzero=False, channel_wise=True))
            print('zscore normalization')
        elif normalize=='minmax':
            transform_list.append(ScaleIntensityd(keys=[indicator_A, indicator_B], minv=-1.0, maxv=1.0))
            print('minmax normalization')

        elif normalize=='scale4000':
            transform_list.append(ScaleIntensityd(keys=[indicator_A], minv=0, maxv=1))
            transform_list.append(ShiftIntensityd(keys=[indicator_B], offset=1024))
            transform_list.append(ScaleIntensityd(keys=[indicator_B], minv=None, maxv=None, factor=-0.99975)) # x=x(1+factor)
            print('scale4000 normalization')

        elif normalize=='scale1000_wrongbutworks':
            transform_list.append(ScaleIntensityd(keys=[indicator_A], minv=0, maxv=1))
            transform_list.append(ScaleIntensityd(keys=[indicator_B], minv=0))
            transform_list.append(ScaleIntensityd(keys=[indicator_B], factor=-0.999)) 
            print('scale1000 normalization')

        elif normalize=='scale1000':
            transform_list.append(ScaleIntensityd(keys=[indicator_A], minv=0, maxv=1))
            transform_list.append(ShiftIntensityd(keys=[indicator_B], offset=1024))
            transform_list.append(ScaleIntensityd(keys=[indicator_B], minv=None, maxv=None, factor=-0.999)) 
            print('scale1000 normalization')
        
        elif normalize=='scale10':
            transform_list.append(ScaleIntensityd(keys=[indicator_A], minv=0, maxv=1))
            #transform_list.append(ScaleIntensityd(keys=[indicator_B], minv=0))
            transform_list.append(ScaleIntensityd(keys=[indicator_B], minv=None, maxv=None,factor=-0.9)) 
            print('scale10 normalization')

        elif normalize=='inputonlyzscore':
            transform_list.append(NormalizeIntensityd(keys=[indicator_A], nonzero=False, channel_wise=True))
            print('only normalize input MRI images')

        elif normalize=='inputonlyminmax':
            transform_list.append(ScaleIntensityd(keys=[indicator_A], minv=self.configs.dataset.normmin, maxv=self.configs.dataset.normmax))
            print('only normalize input MRI images')
        
        elif normalize=='none' or normalize=='nonorm':
            print('no normalization')
        return transform_list

class anika_registrated_mr2ct_loader(synthrad_mr2ct_loader):
    def __init__(self,configs,paths,dimension): 
        super().__init__(configs,paths,dimension)

    def get_loader(self):
        indicator_A=self.configs.dataset.indicator_A	
        indicator_B=self.configs.dataset.indicator_B
        self.indicator_A=indicator_A
        self.indicator_B=indicator_B
        train_number=self.configs.dataset.train_number
        val_number=self.configs.dataset.val_number
        train_batch_size=self.configs.dataset.batch_size
        val_batch_size=self.configs.dataset.val_batch_size
        load_masks=self.configs.dataset.load_masks

        # Conditional dictionary keys based on whether masks are loaded
        keys = [indicator_A, indicator_B, "mask"] if load_masks else [indicator_A, indicator_B]

        ct_dir = r'E:\Datasets\M2olie_Patientdata\CT'
        mri_dir = r'E:\Results\MultistepReg\M2olie_Patientdata\Multistep_network_A\predict'
        
        ct_dir = self.configs.dataset.ct_dir #'E:\Datasets\M2olie_Patientdata\CT'
        mri_dir = self.configs.dataset.mri_dir #'E:\Results\MultistepReg\M2olie_Patientdata\Multistep_network_A\predict'
        matched_pairs = list_from_anika_dataset(ct_dir, mri_dir, self.configs.dataset.mri_mode)
        for patient_id, paths in matched_pairs.items():
            print(f"Patient ID: {patient_id}, CT: {paths['CT']}, MRI: {paths['MRI']}")

        # use the matched pairs to form the dataset
        train_ds = [{indicator_A: paths['MRI'], indicator_B: paths['CT']} for patient_id, paths in list(matched_pairs.items())[:train_number]]
        val_ds = [{indicator_A: paths['MRI'], indicator_B: paths['CT']} for patient_id, paths in list(matched_pairs.items())[-val_number:]]
