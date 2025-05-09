import nibabel as nib
import numpy as np
import cv2
from scipy.ndimage import zoom
import random
from tqdm import tqdm
import os 
import numpy as np
from utils import load_nii, set_offset, crop_or_pad_like
def create_synthetic():
    body_contour_file = r"D:\Project\seg2med_Project\new_synthetic\contour\ct-volume-42553013_i-Spiral  1.5  B30f - 9_42553013_9_mask_9.nii.gz"

    body_contour_file = r"D:\Project\seg2med_Project\new_synthetic\contour\ct-volume-35167831_i-Spiral  1.5  B30f - 14_35167831_14_mask_1.nii.gz"
    total_seg_file = r'D:\Project\seg2med_Project\new_synthetic\merge\ct-volume-35167831_i-Spiral  1.5  B30f - 14_seg.nii.gz'
    liver_file = r'D:\Project\seg2med_Project\new_synthetic\merge\ct-volume-42538338_i-Spiral  1.5  B30f - 5 liver.nii.gz'

    total_seg, seg_affine = load_nii(total_seg_file)
    total_seg = total_seg.astype(np.uint8)
    total_seg[total_seg==5]=0
    total_seg[total_seg==4]=0

    body_contour, body_contour_affine = load_nii(body_contour_file)
    body_contour = body_contour.astype(np.uint8)
    body_contour = set_offset(body_contour, x_offset=10, y_offset=0, z_offset=0)
    factor = 1.0
    scale_factor_sequence = (factor, factor, 1.0)
    body_contour = zoom(body_contour, scale_factor_sequence, order=1)

    liver, _ = load_nii(liver_file)
    liver = liver.astype(np.uint8)
    liver[liver==1]=5
    liver = set_offset(liver, x_offset=5, y_offset=-20, z_offset=-20)
    factor = 0.90
    scale_factor_sequence = (factor, factor, 1.0)
    liver = zoom(liver, scale_factor_sequence, order=1)

    body_contour = crop_or_pad_like(body_contour, total_seg)
    liver = crop_or_pad_like(liver, total_seg)
    print(liver.shape)

    new_synthetic = body_contour + total_seg  + liver
    # Save the synthetic volume
    new_synthetic_file = r"D:\Project\seg2med_Project\new_synthetic\synthetic_patient.nii.gz"
    nib.save(nib.Nifti1Image(new_synthetic, seg_affine), new_synthetic_file)
    
    
def create_with_without_liver(patient_ID, body_contour_file, total_seg_file, liver_file, ct_img_file):
    total_seg, seg_affine = load_nii(total_seg_file)
    total_seg = total_seg.astype(np.uint8)
    liver = np.copy(total_seg)
    total_seg[total_seg==5]=0

    body_contour, body_contour_affine = load_nii(body_contour_file)
    body_contour = body_contour.astype(np.uint8)

    #liver, _ = load_nii(liver_file)
    #liver = liver.astype(np.uint8)
    liver[liver!=5]=0 
    liver[liver==5]=1
    liver_mask = np.copy(liver)
    inverted_liver_mask = np.copy(liver) ^ 1
    liver[liver==1]=5+1

    
    patient_without_liver = body_contour + total_seg
    patient_without_liver_file = f"D:/Project/seg2med_Project/new_synthetic/{patient_ID}_withoutliver.nii.gz"
    nib.save(nib.Nifti1Image(patient_without_liver, seg_affine), patient_without_liver_file)

    onlyliver_file = f"D:/Project/seg2med_Project/new_synthetic/{patient_ID}_onlyliver.nii.gz"
    nib.save(nib.Nifti1Image(liver, seg_affine), onlyliver_file)

    ct_img, _ = load_nii(ct_img_file)
    origliver = np.copy(ct_img)
    origliver[liver_mask==0]=-1000
    
    origwithoutliver = np.copy(ct_img)
    origwithoutliver[liver_mask==1]=-1000
    origliver_file = f"D:/Project/seg2med_Project/new_synthetic/{patient_ID}_origliver.nii.gz"
    nib.save(nib.Nifti1Image(origliver, seg_affine), origliver_file)
    
    origwithoutliver_file = f"D:/Project/seg2med_Project/new_synthetic/{patient_ID}_origwithoutliver.nii.gz"
    nib.save(nib.Nifti1Image(origwithoutliver, seg_affine), origwithoutliver_file)
if __name__ == '__main__':
    patient_ID_list = [
        '34438427_5',
        '35167831_14',
        '40094015_4',
        '40293225_16',
        '40501834_14',
        '41045510_19',
        '42487603_10',
        '42538338_5',
        '42540248_5',
        '42553013_9',
    ]
    
    body_contour_root = r"D:\Project\seg2med_Project\new_synthetic\contour"
    body_contour_file_list = [
    os.path.join(body_contour_root, r"ct-volume-34438427_i-Spiral  1.5  B30f - 5_34438427_5_mask_0.nii.gz"),
    os.path.join(body_contour_root, r"ct-volume-35167831_i-Spiral  1.5  B30f - 14_35167831_14_mask_1.nii.gz"),
    os.path.join(body_contour_root, r"ct-volume-40094015_i-Spiral  1.5  B30f - 4_40094015_4_mask_2.nii.gz"),
    os.path.join(body_contour_root, r"ct-volume-40293225_i-Spiral  3.0  B30f - 16_40293225_16_mask_3.nii.gz"),
    os.path.join(body_contour_root, r"ct-volume-40501834_i-Spiral  1.5  B30f - 14_40501834_14_mask_4.nii.gz"),
    os.path.join(body_contour_root, r"ct-volume-41045510_i-Spiral  1.5  B30f - 19_41045510_19_mask_5.nii.gz"),
    os.path.join(body_contour_root, r"ct-volume-42487603_i-Spiral  1.5  B30f - 10_42487603_10_mask_6.nii.gz"),
    os.path.join(body_contour_root, r"ct-volume-42538338_i-Spiral  1.5  B30f - 5_42538338_5_mask_7.nii.gz"),
    os.path.join(body_contour_root, r"ct-volume-42540248_i-Spiral  1.5  B30f - 5_42540248_5_mask_8.nii.gz"),
    os.path.join(body_contour_root, r"ct-volume-42553013_i-Spiral  1.5  B30f - 9_42553013_9_mask_9.nii.gz"),
    ]

    total_seg_root=r"D:\Project\seg2med_Project\new_synthetic\totalseg"
    total_seg_file_list =[ 
    os.path.join(total_seg_root, r'ct-volume-34438427_i-Spiral  1.5  B30f - 5_seg.nii.gz'),
    os.path.join(total_seg_root, r'ct-volume-35167831_i-Spiral  1.5  B30f - 14_seg.nii.gz'),
    os.path.join(total_seg_root, r'ct-volume-40094015_i-Spiral  1.5  B30f - 4_seg.nii.gz'),
    os.path.join(total_seg_root, r'ct-volume-40293225_i-Spiral  3.0  B30f - 16_seg.nii.gz'),
    os.path.join(total_seg_root, r'ct-volume-40501834_i-Spiral  1.5  B30f - 14_seg.nii.gz'),
    os.path.join(total_seg_root, r'ct-volume-41045510_i-Spiral  1.5  B30f - 19_seg.nii.gz'),
    os.path.join(total_seg_root, r'ct-volume-42487603_i-Spiral  1.5  B30f - 10_seg.nii.gz'),
    os.path.join(total_seg_root, r'ct-volume-42538338_i-Spiral  1.5  B30f - 5_seg.nii.gz'),
    os.path.join(total_seg_root, r'ct-volume-42540248_i-Spiral  1.5  B30f - 5_seg.nii.gz'),
    os.path.join(total_seg_root, r'ct-volume-42553013_i-Spiral  1.5  B30f - 9_seg.nii.gz'),
    ]

    ct_img_root = r"D:\Project\seg2med_Project\new_synthetic"

    liver_file_list = [
        os.path.join(ct_img_root, '1', 'seg', 'liver.nii.gz'),
        os.path.join(ct_img_root, '2', 'seg', 'liver.nii.gz'),
        os.path.join(ct_img_root, '3', 'seg', 'liver.nii.gz'),
        os.path.join(ct_img_root, '4', 'seg', 'liver.nii.gz'),
        os.path.join(ct_img_root, '5', 'seg', 'liver.nii.gz'),
        os.path.join(ct_img_root, '6', 'seg', 'liver.nii.gz'),
        os.path.join(ct_img_root, '7', 'seg', 'liver.nii.gz'),
        os.path.join(ct_img_root, '8', 'seg', 'liver.nii.gz'),
        os.path.join(ct_img_root, '9', 'seg', 'liver.nii.gz'),
        os.path.join(ct_img_root, '10', 'seg', 'liver.nii.gz'),
    ]

    ct_img_file_list = [ 
    os.path.join(ct_img_root, '1', r'ct-volume-34438427_i-Spiral  1.5  B30f - 5.nii.gz'),
    os.path.join(ct_img_root, '2', r'ct-volume-35167831_i-Spiral  1.5  B30f - 14.nii.gz'),
    os.path.join(ct_img_root, '3', r'ct-volume-40094015_i-Spiral  1.5  B30f - 4.nii.gz'),
    os.path.join(ct_img_root, '4', r'ct-volume-40293225_i-Spiral  3.0  B30f - 16.nii.gz'),
    os.path.join(ct_img_root, '5', r'ct-volume-40501834_i-Spiral  1.5  B30f - 14.nii.gz'),
    os.path.join(ct_img_root, '6', r'ct-volume-41045510_i-Spiral  1.5  B30f - 19.nii.gz'),
    os.path.join(ct_img_root, '7', r'ct-volume-42487603_i-Spiral  1.5  B30f - 10.nii.gz'),
    os.path.join(ct_img_root, '8', r'ct-volume-42538338_i-Spiral  1.5  B30f - 5.nii.gz'),
    os.path.join(ct_img_root, '9', r'ct-volume-42540248_i-Spiral  1.5  B30f - 5.nii.gz'),
    os.path.join(ct_img_root, '10', r'ct-volume-42553013_i-Spiral  1.5  B30f - 9.nii.gz'),
    ]

    for patient_ID in patient_ID_list:
        index = patient_ID_list.index(patient_ID)
        body_contour_file = body_contour_file_list[index]
        total_seg_file = total_seg_file_list[index]
        liver_file = liver_file_list[index]
        ct_img_file = ct_img_file_list[index]
        create_with_without_liver(patient_ID, body_contour_file, total_seg_file, liver_file, ct_img_file)