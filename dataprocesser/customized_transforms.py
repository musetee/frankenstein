import cv2
import numpy as np
import torch
from typing import List

VERBOSE = False

def get_data_scaler(config):
  """Data normalizer. Assume data are always in [0, 1]."""
  if config.data.centered:
    # Rescale to [-1, 1]
    return lambda x: x * 2. - 1.
  else:
    return lambda x: x

def get_data_inverse_scaler(config):
  """Inverse data normalizer."""
  if config.data.centered:
    # Rescale [-1, 1] to [0, 1]
    return lambda x: (x + 1.) / 2.
  else:
    return lambda x: x
  
def separate_maps(real_images,
                    tissue_min, tissue_max,
                    bone_min, bone_max):
    mask = torch.zeros_like(real_images)
    # Assign label 1 to tissue regions
    mask[(real_images > tissue_min) & (real_images <= tissue_max)] = 1
    # Assign label 2 to bone regions
    mask[(real_images >= bone_min) & (real_images <= bone_max)] = 2
    return mask

def create_body_contour_old(tensor_img, body_threshold=-500):
    """
    Create a binary body mask from a CT image tensor, using a specific threshold for the body parts.
    There would be problem if more body parts are presented (like two arms)
    Args:
    tensor_img (torch.Tensor): A tensor representation of a grayscale CT image, with intensity values from -1024 to 1500.

    Returns:
    torch.Tensor: A binary mask tensor where the entire body region is 1 and the background is 0.
    """
    # Convert tensor to numpy array
    numpy_img = tensor_img.numpy().astype(np.int16)  # Ensure we can handle negative values correctly

    # Threshold the image at -500 to separate potential body from the background
    binary_img = np.where(numpy_img > body_threshold, 1, 0).astype(np.uint8)
    #print(binary_img.shape)
    #print(binary_img)
    # Find contours from the binary image
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Create an empty mask and fill the largest contour
    mask = np.zeros_like(binary_img)
    if contours:
        # Assume the largest contour is the body contour
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest_contour], -1, 1, thickness=cv2.FILLED)

    # Convert the mask back to a tensor
    mask_tensor = torch.tensor(mask, dtype=torch.int32)

    return mask_tensor

def create_body_contour(tensor_img, body_threshold=-500, min_contour_area=10000):
    """
    Create a binary body mask from a CT image tensor, using a specific threshold for the body parts.
    Solve problem that more body parts are presented (like two arms)

    Args:
    tensor_img (torch.Tensor): A tensor representation of a grayscale CT image, with intensity values from -1024 to 1500.

    Returns:
    torch.Tensor: A binary mask tensor where the entire body region is 1 and the background is 0.
    """
    # Convert tensor to numpy array
    if isinstance(tensor_img, torch.Tensor):
        numpy_img = tensor_img.numpy().astype(np.int16)  # Ensure we can handle negative values correctly
    elif isinstance(tensor_img, np.ndarray):
        numpy_img = np.ascontiguousarray(tensor_img.astype(np.int16)) 
    else:
        print("This is not a PyTorch tensor or a NumPy array. Please Check!")
    # Threshold the image at -500 to separate potential body from the background
    binary_img = np.where(numpy_img > body_threshold, 1, 0).astype(np.uint8)

    # Find contours from the binary image
    contours, _ = cv2.findContours(binary_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty mask
    mask = np.zeros_like(binary_img)

    # Fill all detected body contours
    if contours:
        for contour in contours:
            if cv2.contourArea(contour) >= min_contour_area:
                if VERBOSE:
                    print('current contour area: ', cv2.contourArea(contour), 'threshold: ', min_contour_area)
                cv2.drawContours(mask, [contour], -1, 1, thickness=cv2.FILLED)

    # Convert the mask back to a tensor
    mask_tensor = torch.tensor(mask, dtype=torch.int32)

    return mask_tensor

import numpy as np
import cv2

def create_body_contour_by_seg_tissue(binary_mask: np.ndarray, area_threshold=1000) -> np.ndarray:
    """
    提取组织分割图中的身体轮廓（保留最大连通域/多个大区域），输出二值 mask。
    
    参数:
        binary_mask: np.ndarray, 2D 输入图，非 0 为组织区域
        area_threshold: int, 保留的最小连通域面积
    返回:
        contour_mask: np.uint8, 2D binary mask (0 or 1)
    """
    mask_uint8 = (binary_mask > 0).astype(np.uint8).copy() * 255

    # 找轮廓（忽略空洞）
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_mask = np.zeros_like(mask_uint8, dtype=np.uint8)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > area_threshold:
            cv2.drawContours(contour_mask, [cnt], -1, 1, thickness=-1)  # 填充轮廓

    return (contour_mask > 0).astype(np.uint8)

import pandas as pd

def HU_assignment(mask, csv_file):
    if isinstance(mask, torch.Tensor):
        hu_mask = torch.zeros_like(mask)
    elif isinstance(mask, np.ndarray):
        hu_mask = np.zeros_like(mask)
    
    df = pd.read_csv(csv_file)
    hu_values = dict(zip(df['Order Number'], df['HU Value']))
    order_begin_from_0 = True if df['Order Number'].min()==0 else False
    # Value Assigment
    hu_mask[mask == 0] = -1000 # background
    for organ_index, hu_value in hu_values.items():
        assert isinstance(hu_value, int), f"Expected mask value an integer, but got {hu_value}. Ensure the mask is created by fine mode of totalsegmentator"
        assert isinstance(organ_index, int), f"Expected organ_index an integer, but got {organ_index}. Ensure the mask is created by fine mode of totalsegmentator"
        if order_begin_from_0:
            hu_mask[mask == (organ_index+1)] = hu_value # mask value begin from 1 as body value, other than 0 in TA2 table, so organ_index+1
        else:
            hu_mask[mask == (organ_index)] = hu_value
    return hu_mask

class MaskHUAssigmentd: 
    def __init__(self, keys, csv_file):
        self.keys = keys
        # Read the CSV into a DataFrame
        self.df = pd.read_csv(csv_file)
        #print(self.hu_values)

    def __call__(self, data):
        # Create a dictionary to map organ index to HU values
        for key in self.keys:
            mask = data[key]

            self.hu_values = dict(zip(self.df['Order Number'], self.df['HU Value']))
            self.order_begin_from_0 = True if self.df['Order Number'].min()==0 else False
            hu_mask = torch.zeros_like(mask)
            # Value Assigment
            hu_mask[mask == 0] = -1000 # background
            for organ_index, hu_value in self.hu_values.items():
                assert isinstance(hu_value, int), f"Expected mask value an integer, but got {hu_value}. Ensure the mask is created by fine mode of totalsegmentator"
                assert isinstance(organ_index, int), f"Expected organ_index an integer, but got {organ_index}. Ensure the mask is created by fine mode of totalsegmentator"
                if self.order_begin_from_0:
                    hu_mask[mask == (organ_index+1)] = hu_value # mask value begin from 1 as body value, other than 0 in TA2 table, so organ_index+1
                else:
                    hu_mask[mask == (organ_index)] = hu_value
            data[key] = hu_mask
        return data
import pandas as pd
import numpy as np

def convert_segmentation_mask(source_mask, source_csv, target_csv, body_contour_value=1):
    """
    Converts segmentation mask values from source modality to target modality based on organ name mapping.

    Parameters:
    - source_mask (ndarray): The source segmentation mask array.
    - source_csv (str): Path to the CSV file of the source modality (CT or MR).
    - target_csv (str): Path to the CSV file of the target modality (MR or CT).
    - body_contour_value (int): The class value for "body contour" in the target modality.

    Returns:
    - target_mask (ndarray): The converted segmentation mask.
    """
    # Load the source and target anatomy lists
    source_df = pd.read_csv(source_csv)
    target_df = pd.read_csv(target_csv)

    # Create dictionaries mapping class values to organ names and vice versa
    source_mapping = {row['Organ Name']: row.iloc[0] for _, row in source_df.iterrows()}
    target_mapping = {row['Organ Name']: row.iloc[0] for _, row in target_df.iterrows()}

    # Initialize the target mask
    target_mask = np.full_like(source_mask, body_contour_value, dtype=source_mask.dtype)

    # Convert each unique class in the source mask
    for class_value in np.unique(source_mask):
        # Find the corresponding organ name in the source modality
        organ_name = {v: k for k, v in source_mapping.items()}.get(class_value, None)

        # If organ name exists, find the target class value
        if organ_name and organ_name in target_mapping:
            target_value = target_mapping[organ_name]
        else:
            # Use body contour class value for unmapped organs
            target_value = body_contour_value

        # Replace class values in the target mask
        target_mask[source_mask == class_value] = target_value

    return target_mask

class CreateBodyContourTransformd: 
    def __init__(self, keys, body_threshold,body_mask_value):
        self.keys = keys
        self.body_threshold = body_threshold
        self.body_mask_value = body_mask_value

    def __call__(self, data):
        # input medical image (CT) and create body contour, then replace the image by contour
        for key in self.keys:
            x = data[key]
            #print(x)
            mask = torch.zeros_like(x)
            # [B, H, W, D]
            # create a mask for each slice in the batch
            for i in range(x.shape[0]):
                for j in range(x.shape[-1]):
                    mask_slice = create_body_contour(x[i,:,:,j], body_threshold=self.body_threshold)
                    mask[i,:,:, j] = mask_slice
            mask[mask == 1] = self.body_mask_value
            if VERBOSE:
                print("created mask shape:", mask.shape)
            data[key] = mask
        return data
    
class CreateBodyContourMultiModalTransformd: 
    def __init__(self, keys, body_threshold,body_mask_value):
        self.keys = keys
        self.body_threshold = body_threshold
        self.body_mask_value = body_mask_value

    def __call__(self, data):
        # input medical image (CT) and create body contour, then replace the image by contour
        for key in self.keys:
            x = data[key]
            #print(x)
            mask = torch.zeros_like(x)
            # [B, H, W, D]
            # create a mask for each slice in the batch
            for i in range(x.shape[0]):
                for j in range(x.shape[-1]):
                    mask_slice = create_body_contour(x[i,:,:,j], body_threshold=self.body_threshold)
                    mask[i,:,:, j] = mask_slice
            mask[mask == 1] = self.body_mask_value
            if VERBOSE:
                print("created mask shape:", mask.shape)
            data[key] = mask
        return data
    
def convert_xcat_to_ct_mask(xcat_image, mapping_csv, tolerance=0.5):
    """
    Converts XCAT CT digital phantom images to simulated CT masks.

    Parameters:
    - xcat_image (torch.Tensor): The XCAT CT image tensor (in HU values).
    - mapping_csv (str): Path to the CSV file containing organ, HU value, and mask value mappings.
    - tolerance (float): Tolerance for HU value matching (default is ±0.5).

    Returns:
    - ct_mask (torch.Tensor): The converted CT mask tensor.
    """
    # Load the mapping CSV
    mapping_df = pd.read_csv(mapping_csv)

    # Initialize the CT mask as a tensor filled with zeros (or another default background value)
    if isinstance(xcat_image, np.ndarray):
        ct_mask = np.zeros_like(xcat_image, dtype=np.int32)
    elif isinstance(xcat_image, torch.Tensor):
        ct_mask = torch.zeros_like(xcat_image, dtype=torch.int32)
    else:
        raise TypeError("xcat_image must be a NumPy ndarray or a PyTorch tensor.")

    # Iterate over the mapping and replace pixel values
    for _, row in mapping_df.iterrows():
        organ = row['Organ']
        hu_value = row['HU_Value']
        mask_value = row['Mask_Value']

        # Apply the tolerance range for matching
        lower_bound = hu_value - tolerance
        upper_bound = hu_value + tolerance

        # Replace matching pixels with the mask value
        match_condition = (xcat_image >= lower_bound) & (xcat_image <= upper_bound)
        ct_mask[match_condition] = mask_value

        print(f"Processed {organ} with HU range [{lower_bound}, {upper_bound}] to mask value {mask_value}")
    return ct_mask

class UseContourToFilterImaged:
    def __init__(self, 
                 keys: List[str]
                 ):
        if len(keys) != 2:
            raise ValueError("Keys must be a list with exactly two string elements.")
        self.image_key = keys[0]
        self.contour_key = keys[1]
    def __call__(self, data):
        image = data[self.image_key]
        contour = data[self.contour_key]
        data[self.image_key] = image*contour
        return data
    
class MergeMasksTransformd: 
    def __init__(self, 
                 keys: List[str]):
        if len(keys) != 2:
            raise ValueError("Keys must be a list with exactly two string elements.")
        self.seg_key = keys[0]
        self.contour_key = keys[1]

    def __call__(self, data):
        seg = data[self.seg_key]
        contour = data[self.contour_key]
        merged_mask = seg + contour

        data[self.seg_key] = merged_mask
        return data
    
class MergeSegTissueTransformd: 
    def __init__(self, 
                 keys: List[str]):
        if len(keys) != 2:
            raise ValueError("Keys must be a list with exactly two string elements.")
        self.seg_key = keys[0]
        self.tissue_key = keys[1]

    def __call__(self, data):
        seg = data[self.seg_key]
        tissue = data[self.tissue_key]
        tissue += 100 # keep the tissue value always higher as segmentation organs
        # Create a mask for overlapping areas
        overlap_mask = (seg > 0) & (tissue > 0)
        
        # For overlapping areas, keep the lower value (organ values in seg)
        merged_mask = tissue.copy()
        merged_mask[overlap_mask] = seg[overlap_mask]
        
        # Keep all non-overlapping areas
        merged_mask[seg > 0] = seg[seg > 0]


        data[self.seg_key] = merged_mask
        return data
    
class DivideTransformd: 
    def __init__(self, 
                 keys: List[str],
                 divide_factor):
        self.keys=keys
        self.divide_factor=divide_factor
    def __call__(self, data):
        for key in self.keys:
            data[key] = data[key]/self.divide_factor
        return data

class MergeMasksTransformOldd: 
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        #print('check MergeMasksTransformd:', data)
        merged_mask = torch.zeros_like(data[self.keys[0]], dtype=torch.int32)

        for key in self.keys:
            merged_mask += data[key].to(torch.int32)
        for key in self.keys:
            data[key] = merged_mask
        return data
    
# convert the integer segemented labels to one-hot codes for training
class ConvertToOneHotd:
    def __init__(self, keys, number_classes):
        self.keys = keys
        self.nc = number_classes

    def __call__(self, data):
        for key in self.keys:
            x = data[key]
            # Ensure the tensor is of the correct type
            if x.dtype != torch.long:
                x = x.long()
            # Create the one-hot encoded tensor
            one_hot = torch.zeros(x.size(0), self.nc, x.size(1), x.size(2), device=x.device)
            one_hot.scatter_(1, x.unsqueeze(1), 1)
            data[key] = one_hot
        return data

# Example usage
# Assuming `ct_image_tensor` is a PyTorch tensor of a CT image
# ct_image_tensor = torch.tensor(img_array, dtype=torch.float32)
# mask_tensor = create_body_contour(ct_image_tensor)

class CreateMaskWithBonesTransform:
    def __init__(self,tissue_min,tissue_max,bone_min,bone_max):
        # You can add initialization parameters if needed
        self.tissue_min = tissue_min
        self.tissue_max = tissue_max
        self.bone_min = bone_min
        self.bone_max = bone_max

    def __call__(self, x):
        # x is the input tensor
        # Initialize mask with zeros (background)
        mask = torch.zeros_like(x)

        # Assign label 1 to tissue regions (-500 to 200)
        mask[(x > self.tissue_min) & (x <= self.tissue_max)] = 1

        # Assign label 2 to bone regions (200 to 1500)
        mask[(x >= self.bone_min) & (x <= self.bone_max)] = 2

        return mask

class CreateMaskWithBonesTransformd: 
    def __init__(self, keys, tissue_min, tissue_max, bone_min, bone_max):
        self.keys = keys
        self.tissue_min = tissue_min
        self.tissue_max = tissue_max
        self.bone_min = bone_min
        self.bone_max = bone_max
    def __call__(self, data):
        for key in self.keys:
            x = data[key]
            
            mask = torch.zeros_like(x)
            # [B, H, W, D]
            for i in range(x.shape[0]):
                for j in range(x.shape[-1]):
                    mask_slice = create_body_contour(x[i,:,:,j], body_threshold=self.tissue_min)
                    mask[i,:,:, j] = mask_slice
            #mask = torch.zeros_like(x)
            #mask[(x > self.tissue_min) & (x <= self.tissue_max)] = 1
            mask[(x >= self.bone_min) & (x <= self.bone_max)] = 2
            data[key] = mask
            #print("input and mask shape: ",x.shape,data[key].shape)
        return data

class NormalizationMultimodal:
    def __init__(self, keys):
        if len(keys) != 2:
            raise ValueError("Keys must be a list with exactly two string elements.")
        self.prior_key = keys[0]
        self.target_key = keys[1]

        self.prior_modality_norm_dict = {
            0: {'min': -300, 'max': 700},   # CT WW=1000, WL=200
            1: {'min': 0, 'max': 9},       # T1
            2: {'min': 0, 'max': 28},       # T2
            3: {'min': 0, 'max': 9},       # VIBE-IN
            4: {'min': 0, 'max': 10},       # VIBE-OPP
            5: {'min': 0, 'max': 6},       # DIXON
        }

        self.target_modality_norm_dict = {
            0: {'min': -300, 'max': 700},   # CT
            1: {'min': 0, 'max': 800},       # T1
            2: {'min': 0, 'max': 160},       # T2
            3: {'min': 0, 'max': 500},       # VIBE-IN
            4: {'min': 0, 'max': 520},       # VIBE-OPP
            5: {'min': 0, 'max': 560},       # DIXON
        }
    
    def __call__(self, data):
        modality = int(data['modality'])  
        
        if modality not in self.target_modality_norm_dict:
            raise ValueError(f"Unsupported modality id: {modality}")
        
        # Normalize target
        x_target = data[self.target_key]
        target_params = self.target_modality_norm_dict[modality]
        x_target = torch.clamp(x_target, target_params['min'], target_params['max'])
        x_target = (x_target - target_params['min']) / (target_params['max'] - target_params['min'])
        data[self.target_key] = x_target

        # Normalize prior
        x_prior = data[self.prior_key]
        prior_params = self.prior_modality_norm_dict[modality]
        x_prior = torch.clamp(x_prior, prior_params['min'], prior_params['max'])
        x_prior = (x_prior - prior_params['min']) / (prior_params['max'] - prior_params['min'])
        data[self.prior_key] = x_prior

        return data

