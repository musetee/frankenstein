
from totalsegmentator.python_api import totalsegmentator
class CreateMaskTransformd: 
    def __init__(self, keys, tissue_min, tissue_max, bone_min, bone_max, mask_value_bones=2,
                 if_use_total_seg=False, organ_label_id=52, mask_value_organ=2, fast=True):
        self.keys = keys
        self.tissue_min = tissue_min
        self.tissue_max = tissue_max
        self.bone_min = bone_min
        self.bone_max = bone_max
        self.mask_value_bones = mask_value_bones

        self.if_use_total_seg = if_use_total_seg
        self.organ_label_id = organ_label_id
        self.mask_value_organ = mask_value_organ
        self.fast = fast

    def extract_organ_mask(self, input_img, organ_label_id, mask_value):
        # aorta = 52
        """
        Extracts a binary mask for a specific organ from a labeled NIFTI image.
        
        img_in: NIFTI image with segmentation labels.
        organ_name: Name of the organ to extract.
        label_map: Dictionary mapping label IDs to organ names.
        
        returns: Binary mask as a NIFTI image.
        """
        img_in = totalsegmentator(input=input_img, task='total',fast=self.fast)
        data = img_in.get_fdata()

        # Create a binary mask for the specified organ
        organ_mask_data = np.zeros_like(data)
        organ_mask_data[data == organ_label_id] = mask_value

        # Create a new NIFTI image for the binary mask
        organ_mask_img = nib.Nifti1Image(organ_mask_data, img_in.affine, img_in.header)
        return organ_mask_img
    
    def __call__(self, data):
        for key in self.keys:
            x = data[key]
            
            mask = torch.zeros_like(x)
            # [B, H, W, D]
            # create a mask for each slice in the batch
            for i in range(x.shape[0]):
                if self.if_use_total_seg:
                    mask_batch_i = self.extract_organ_mask(x[i,:,:,:], organ_label_id=self.organ_label_id, mask_value=self.mask_value_organ)
                    mask[i,:,:,:] = mask_batch_i
                for j in range(x.shape[-1]):
                    mask_slice = create_body_mask(x[i,:,:,j], body_threshold=self.tissue_min)
                    mask[i,:,:, j] = mask_slice
            #mask = torch.zeros_like(x)
            #mask[(x > self.tissue_min) & (x <= self.tissue_max)] = 1
            mask[(x >= self.bone_min) & (x <= self.bone_max)] = self.mask_value_bones
            data[key] = mask
            #print("input and mask shape: ",x.shape,data[key].shape)
        return data

class CreateSegTransformd:
    # create a mask by segmenting the input image using totalsegmentator
    def __init__(self, keys, organ_label_id=52, mask_value=2, fast=True):
        self.keys = keys
        self.organ_label_id = organ_label_id
        self.mask_value = mask_value
        self.fast = fast

    def extract_organ_mask(self, input_img, organ_label_id, mask_value):
        # aorta = 52
        """
        Extracts a binary mask for a specific organ from a labeled NIFTI image.
        
        img_in: NIFTI image with segmentation labels.
        organ_name: Name of the organ to extract.
        label_map: Dictionary mapping label IDs to organ names.
        
        returns: Binary mask as a NIFTI image.
        """
        img_in = totalsegmentator(input=input_img, task='total',fast=self.fast)
        data = img_in.get_fdata()

        if organ_label_id>0:
            # Create a binary mask for the specified organ
            organ_mask_data = np.zeros_like(data)
            organ_mask_data[data == organ_label_id] = mask_value
        else:
            organ_mask_data=data

        # Create a new NIFTI image for the binary mask
        organ_mask_img = nib.Nifti1Image(organ_mask_data, img_in.affine, img_in.header)
        return organ_mask_img
    
    def __call__(self, data):
        for key in self.keys:
            x = data[key]
            mask = torch.zeros_like(x)
            # [B, H, W, D]
            for i in range(x.shape[0]):
                mask_batch_i = self.extract_organ_mask(x[i,:,:,:], organ_label_id=self.organ_label_id, mask_value=self.mask_value)
                mask[i,:,:,:] = mask_batch_i
            data[key] = mask
        return data


class CreateTotalSegTransformd:
    # create a mask by segmenting the input image using totalsegmentator
    def __init__(self, keys, fast=True):
        self.keys = keys
        self.fast = fast

    def extract_organ_mask(self, input_img):
        # aorta = 52
        """
        Extracts a binary mask for a specific organ from a labeled NIFTI image.
        
        img_in: NIFTI image with segmentation labels.
        organ_name: Name of the organ to extract.
        label_map: Dictionary mapping label IDs to organ names.
        
        returns: Binary mask as a NIFTI image.
        """
        #print(input_img.meta)
        input_affine = input_img.meta['affine']

        input_img = torch_tensor_to_nifti(input_img, affine=input_affine)
        img_in = totalsegmentator(input=input_img, task='total', fast=self.fast)
        data = img_in.get_fdata()
        organ_mask_data=data
        # Create a new NIFTI image for the binary mask
        organ_mask_img = nib.Nifti1Image(organ_mask_data, img_in.affine, img_in.header)
        return organ_mask_img
    
    def __call__(self, data):
        for key in self.keys:
            x = data[key]
            mask = torch.zeros_like(x)
            # [B, H, W, D]
            for i in range(x.shape[0]):
                mask_batch_i = self.extract_organ_mask(x[i,:,:,:])
                numpy_data = mask_batch_i.get_fdata()
    
                # Convert the NumPy array to a PyTorch tensor
                tensor_data = torch.from_numpy(numpy_data).float()
                mask[i,:,:,:] = tensor_data
            data[key] = mask
        return data

    def get_transforms(self, transform_list):
        normalize=configs.dataset.normalize
        pad=configs.dataset.pad
        resized_size=configs.dataset.resized_size
        WINDOW_WIDTH=configs.dataset.WINDOW_WIDTH
        WINDOW_LEVEL=configs.dataset.WINDOW_LEVEL
        prob=configs.dataset.augmentationProb
        background=configs.dataset.background
        indicator_A=configs.dataset.indicator_A
        indicator_B=configs.dataset.indicator_B
        load_masks=configs.dataset.load_masks
        transform_list=[]
        input_is_mask=configs.dataset.input_is_mask
        # normally we input CT images and here we create masks for CT images
        if not input_is_mask:
            if not configs.dataset.use_all_masks:
                transform_list.append(CreateMaskTransformd(keys=[indicator_A],
                                                        tissue_min=configs.dataset.tissue_min,
                                                        tissue_max=configs.dataset.tissue_max,
                                                        bone_min=configs.dataset.bone_min,
                                                        bone_max=configs.dataset.bone_max,
                                                        mask_value_bones=2,
                                                        ))
            else:  # use all masks from the totalsegmentator
                transform_list.append(CreateTotalSegTransformd(keys=[indicator_A],
                                                        fast=True))
        min, max=WINDOW_LEVEL-(WINDOW_WIDTH/2), WINDOW_LEVEL+(WINDOW_WIDTH/2)
        #transform_list.append(ThresholdIntensityd(keys=[indicator_B], threshold=min, above=True, cval=background))
        #transform_list.append(ThresholdIntensityd(keys=[indicator_B], threshold=max, above=False, cval=-1000))
        # filter the source images
        # transform_list.append(ThresholdIntensityd(keys=[indicator_A], threshold=configs.dataset.MRImax, above=False, cval=0))
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

        elif normalize=='scale1000':
            transform_list.append(ShiftIntensityd(keys=[indicator_B], offset=1024))
            transform_list.append(ScaleIntensityd(keys=[indicator_B], minv=None, maxv=None, factor=-0.999)) 
            print('scale1000 normalization')
        
        elif normalize=='scale4000':
            transform_list.append(ShiftIntensityd(keys=[indicator_B], offset=1024))
            transform_list.append(ScaleIntensityd(keys=[indicator_B], minv=None, maxv=None, factor=-0.99975))
            print('scale4000 normalization')
            
        elif normalize=='scale10':
            #transform_list.append(ScaleIntensityd(keys=[indicator_B], minv=0))
            transform_list.append(ScaleIntensityd(keys=[indicator_B], minv=None, maxv=None,factor=-0.9)) 
            print('scale10 normalization')

        elif normalize=='inputonlyzscore':
            transform_list.append(NormalizeIntensityd(keys=[indicator_A], nonzero=False, channel_wise=True))
            print('only normalize input MRI images')

        elif normalize=='inputonlyminmax':
            transform_list.append(ScaleIntensityd(keys=[indicator_A], minv=configs.dataset.normmin, maxv=configs.dataset.normmax))
            print('only normalize input MRI images')
        
        elif normalize=='none' or normalize=='nonorm':
            print('no normalization')

        spaceXY=self.configs.dataset.spaceXY
        if spaceXY>0:
            transform_list.append(Spacingd(keys=[indicator_A], pixdim=(spaceXY, spaceXY, 2.5), mode="bilinear", ensure_same_shape=True)) # 
            transform_list.append(Spacingd(keys=[indicator_B, "mask"] if load_masks else [indicator_B], 
                                           pixdim=(spaceXY, spaceXY , 2.5), mode="bilinear", ensure_same_shape=True))
        
        transform_list.append(Zoomd(keys=[indicator_A, indicator_B,"mask"] if load_masks 
                                                   else [indicator_A, indicator_B], 
                                                  zoom=configs.dataset.zoom, keep_size=False, mode='area',padding_mode='minimum'))


        transform_list.append(DivisiblePadd(keys=[indicator_A, indicator_B,"mask"] if load_masks else [indicator_A, indicator_B],
                                            k=self.configs.dataset.div_size, mode="minimum"))
        transform_list.append(ResizeWithPadOrCropd(keys=[indicator_A, indicator_B,"mask"] if load_masks else [indicator_A, indicator_B], 
                                                  spatial_size=resized_size,mode=pad))

        if configs.dataset.rotate:
            transform_list.append(Rotate90d(keys=[indicator_A, indicator_B, "mask"] if load_masks else [indicator_A, indicator_B], k=3))

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
            shapeAug=configs.dataset.shapeAug
            if shapeAug:
                #transform_list.append(RandRotated(keys=[indicator_A, indicator_B, "mask"] if load_masks else [indicator_A, indicator_B],
                #                                  range_x = 0.0, range_y = 1.0, range_z = 1.0, 
                #                                  prob=prob, padding_mode="border", keep_size=False))
                transform_list.append(RandZoomd(keys=[indicator_A, indicator_B, "mask"] if load_masks else [indicator_A, indicator_B], 
                                                prob=prob, min_zoom=self.configs.dataset.rand_min_zoom, max_zoom=self.configs.dataset.rand_max_zoom,
                                                padding_mode= "minimum" ,keep_size=False))
                #transform_list.append(RandAffined(keys=[indicator_A, indicator_B], padding_mode="border" , prob=prob))
                #transform_list.append(Rand3DElasticd(keys=[indicator_A, indicator_B], prob=prob, sigma_range=(5, 8), magnitude_range=(100, 200), spatial_size=None, mode='bilinear'))
            intensityAug=configs.dataset.intensityAug
            if intensityAug:
                print('intensity data augmentation is used')
                transform_list.append(RandBiasFieldd(keys=[indicator_A], degree=3, coeff_range=(0.0, 0.1), prob=prob)) # only apply to MRI images
                transform_list.append(RandGaussianNoised(keys=[indicator_A], prob=prob, mean=0.0, std=0.01))
                transform_list.append(RandAdjustContrastd(keys=[indicator_A], prob=prob, gamma=(0.5, 1.5)))
                transform_list.append(RandShiftIntensityd(keys=[indicator_A], prob=prob, offsets=20))
                transform_list.append(RandGaussianSharpend(keys=[indicator_A], alpha=(0.2, 0.8), prob=prob))
            
        #transform_list.append(Rotate90d(keys=[indicator_A, indicator_B], k=3))
        #transform_list.append(DivisiblePadd(keys=[indicator_A, indicator_B], k=div_size, mode="minimum"))
        #transform_list.append(Identityd(keys=[indicator_A, indicator_B]))  # do nothing for the no norm case
        train_transforms = Compose(transform_list)
        return train_transforms
