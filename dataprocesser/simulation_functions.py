from dataprocesser.customized_transforms import create_body_contour, create_body_contour_by_seg_tissue
from dataprocesser.Preprocess_MR_Mask_generation import process_segmentation
import numpy as np
import pandas as pd
import nibabel as nib

def _create_body_contour(x, body_threshold, body_mask_value):
    mask = np.zeros_like(x)
    # [H, W, D]
    # create a mask for each slice in the batch
    for j in range(x.shape[-1]):
        mask_slice = create_body_contour(x[:,:,j], body_threshold=body_threshold)
        mask[:,:, j] = mask_slice
    mask[mask == 1] = body_mask_value
    return mask

def _create_body_contour_by_tissue_seg(x, body_mask_value=1, area_threshold=1000):
    mask = np.zeros_like(x)
    # [H, W, D]
    # create a mask for each slice in the batch
    for j in range(x.shape[-1]):
        mask_slice = create_body_contour_by_seg_tissue(x[:,:,j], area_threshold=area_threshold)
        mask[:,:, j] = mask_slice
    mask[mask == 1] = body_mask_value
    return mask

def _merge_seg_tissue(seg, tissue):
    # because tissue seg from totalsegmentator is 1, 2, 3
    # to avoid ambiguity
    tissue[tissue > 0] += 200 # 1,2,3 + 200 = 201,202,203
    # Create a mask for overlapping areas
    overlap_mask = (seg > 0) & (tissue > 0)
    
    # For overlapping areas, keep the lower value (organ values in seg)
    merged_mask = tissue.copy()
    merged_mask[overlap_mask] = seg[overlap_mask]
    
    # Keep all non-overlapping areas
    merged_mask[seg > 0] = seg[seg > 0]
    return merged_mask

def _merge_seg_contour(seg, contour):
    merged_mask = seg + contour
    return merged_mask

def _assign_value_ct(csv_file, mask, key_id='Order Number', key_hu='HU Value'):
    print('assign hu value to ct image prior')
    df = pd.read_csv(csv_file)
    hu_values = dict(zip(df[key_id], df[key_hu]))
    order_begin_from_0 = True if df[key_id].min()==0 else False
    hu_mask = np.zeros_like(mask)
    # Value Assigment
    hu_mask[mask == 0] = -1000 # background
    for organ_index, hu_value in hu_values.items():
        assert isinstance(hu_value, int), f"Expected mask value an integer, but got {hu_value}. Ensure the mask is created by fine mode of totalsegmentator"
        assert isinstance(organ_index, int), f"Expected organ_index an integer, but got {organ_index}. Ensure the mask is created by fine mode of totalsegmentator"
        if order_begin_from_0:
            #print("order in csv begin from 0")
            hu_mask[mask == (organ_index+1)] = hu_value # mask value begin from 1 as body value, other than 0 in TA2 table, so organ_index+1
        else:
            #print("order in csv begin from 1")
            hu_mask[mask == (organ_index)] = hu_value
    return hu_mask

def _assign_value_mr(csv_file, mask, mr_signal_formula):
    csv_simulation_values = pd.read_csv(csv_file, header=None).to_numpy()
    assign_value_mask = process_segmentation(mask, csv_simulation_values, mr_signal_formula)
    return assign_value_mask