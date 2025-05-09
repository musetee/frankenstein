import nrrd
import sys
import os
import numpy as np
import pandas as pd
def HU_assignment(mask, csv_file):
    if isinstance(mask, np.ndarray):
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

#images = [r"D:\Project\seg2med_Project\new_synthetic\onlyliver.nii.gz"]
images_folder = r'D:\Project\seg2med_Project\new_synthetic\with_without_liver'
images = [os.path.join(images_folder,file) for file in os.listdir(images_folder)] 
output_folder = images_folder
mapping_csv = r'synthrad_conversion\TA2_CT_from1.csv'
#input_path = r'E:\Projects\chen_proj\aorta_XCAT2CT\CT_XCAT_aorta\CT_Model71_Energy90\CT_Model71_Energy90_atn_1.nrrd'
#output_path1 = r'E:\Projects\yang_proj\data\xcat\CT_Model71_Energy90_atn_1_seg.nrrd'
from tqdm import tqdm
import nibabel as nib
dataset_list=[]
step=0
for input_path in tqdm(images):
    if input_path.endswith('.nii.gz'):
        if ('withoutliver' in input_path) or ('onlyliver' in input_path):
            # Extract the file name from the input path
            file_name = os.path.basename(input_path)
            output_file_name = file_name.replace('.nii.gz', '_hu.nii.gz')
            output_file = os.path.join(output_folder, output_file_name)
            print(f'convert synthetic to hu mask, from {input_path} to {output_file}')
            
            img_metadata = nib.load(input_path)
            img = img_metadata.get_fdata()
            affine = img_metadata.affine

            hu_mask = HU_assignment(img, mapping_csv)

            img_processed = nib.Nifti1Image(hu_mask, affine)
            nib.save(img_processed, output_file)

            step += 1