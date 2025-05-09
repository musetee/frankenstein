from dataprocesser.customized_transforms import (
    CreateBodyContourTransformd, 
    MergeMasksTransformd, 
    UseContourToFilterImaged, 
    MaskHUAssigmentd, 
    MergeSegTissueTransformd,
    NormalizationMultimodal,
    CreateMaskWithBonesTransformd)
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
## intensity transforms
def add_normalization_transform_single_B(transform_list, indicator_B, normalize):
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

    elif normalize == 'nonegative':
        offset=1000
        transform_list.append(ShiftIntensityd(keys=[indicator_B], offset=offset))
        print('none negative normalization')

    elif normalize=='none' or normalize=='nonorm':
        print('no normalization')
    
    return transform_list

def add_normalization_multimodal(transform_list, indicator_A, indicator_B):
    transform_list.append(NormalizationMultimodal(keys=[indicator_A,indicator_B]))
    return transform_list

def add_normalization_transform_A_B(transform_list, normalize, indicator_A, indicator_B):
    if normalize=='zscore':
            transform_list.append(NormalizeIntensityd(keys=[indicator_A,indicator_B], nonzero=False, channel_wise=True))
            print('zscore normalization')
    elif normalize=='scale2000':
        transform_list.append(ScaleIntensityd(keys=[indicator_A,indicator_B], minv=None, maxv=None, factor=-0.9995))
        print('scale2000 normalization')
    elif normalize=='none' or normalize=='nonorm':
        print('no normalization')
    return transform_list

def add_normalization_transform_input_only(transform_list, indicator_A, normalize):
    if normalize=='inputonlyzscore':
        transform_list.append(NormalizeIntensityd(keys=[indicator_A], nonzero=False, channel_wise=True))
        print('only normalize input MRI images')

    elif normalize=='inputonlyminmax':
        normmin=0
        normmax=1
        transform_list.append(ScaleIntensityd(keys=[indicator_A], minv=normmin, maxv=normmax))
        print('only normalize input MRI images')

def add_CreateContour_MergeMask_transforms(transform_list, indicator_A):
    transform_list.append(CreateBodyContourTransformd(keys=['mask'],
                                                        body_threshold=-500,
                                                        body_mask_value=1,
                                                        ))
    transform_list.append(MergeMasksTransformd(keys=[indicator_A, 'mask']))
    return transform_list

def add_CreateContour_MergeMask_MaskHUAssign_transforms(transform_list, indicator_A, anatomy_list_csv):
    transform_list.append(CreateBodyContourTransformd(keys=['mask'],
                                                        body_threshold=-500,
                                                        body_mask_value=1,
                                                        )) # image -> contour
    transform_list.append(MergeMasksTransformd(keys=[indicator_A, 'mask'])) # seg+contour -> seg
    transform_list.append(MaskHUAssigmentd(keys=[indicator_A], csv_file=anatomy_list_csv))
    return transform_list

def add_CreateContour_MergeSegTissue_MergeMask_MaskHUAssign_transforms(transform_list, indicator_A, anatomy_list_csv, anatomy_list_csv_mr):
    transform_list.append(CreateBodyContourTransformd(keys=['mask'],
                                                    body_threshold=-500,
                                                    body_mask_value=1,
                                                    )) # image -> contour
    transform_list.append(MergeSegTissueTransformd(keys=[indicator_A, 'seg_tissue'])) # seg+seg_tissue -> seg
    transform_list.append(MergeMasksTransformd(keys=[indicator_A, 'mask'])) # seg+contour -> seg
    transform_list.append(MaskHUAssigmentd(keys=[indicator_A], csv_file=anatomy_list_csv))
    return transform_list

def add_Windowing_ZeroShift_ContourFilter_A_B_transforms(transform_list, WINDOW_LEVEL, WINDOW_WIDTH, indicator_A, indicator_B):
    threshold_low=WINDOW_LEVEL - WINDOW_WIDTH / 2
    threshold_high=WINDOW_LEVEL + WINDOW_WIDTH / 2
    offset=(-1)*threshold_low
    # if filter out the pixel with values below threshold1, set above=True, and the cval1>=threshold1, otherwise there will be problem
    # mask = img > self.threshold if self.above else img < self.threshold
    # res = where(mask, img, self.cval)
    transform_list.append(ThresholdIntensityd(keys=[indicator_A,indicator_B], threshold=threshold_low, above=True, cval=threshold_low))
    transform_list.append(ThresholdIntensityd(keys=[indicator_A,indicator_B], threshold=threshold_high, above=False, cval=threshold_high))
    transform_list.append(ShiftIntensityd(keys=[indicator_A,indicator_B], offset=offset))
    transform_list.append(UseContourToFilterImaged(keys=[indicator_B, 'mask'])) # image*contour -> image
    return transform_list

def add_Windowing_ZeroShift_ContourFilter_single_B_transforms(transform_list, WINDOW_LEVEL, WINDOW_WIDTH, indicator_B):
    threshold_low=WINDOW_LEVEL - WINDOW_WIDTH / 2
    threshold_high=WINDOW_LEVEL + WINDOW_WIDTH / 2
    offset=(-1)*threshold_low
    # if filter out the pixel with values below threshold1, set above=True, and the cval1>=threshold1, otherwise there will be problem
    # mask = img > self.threshold if self.above else img < self.threshold
    # res = where(mask, img``, self.cval)
    transform_list.append(ThresholdIntensityd(keys=[indicator_B], threshold=threshold_low, above=True, cval=threshold_low))
    transform_list.append(ThresholdIntensityd(keys=[indicator_B], threshold=threshold_high, above=False, cval=threshold_high))
    transform_list.append(ShiftIntensityd(keys=[indicator_B], offset=offset))
    transform_list.append(UseContourToFilterImaged(keys=[indicator_B, 'mask'])) # image*contour -> image
    return transform_list
