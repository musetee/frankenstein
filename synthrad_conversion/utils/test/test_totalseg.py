import torch
from torch.utils.data import DataLoader
import monai
import os
import glob
import SimpleITK as sitk

import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F
import argparse

from pkg_resources import require
from pathlib import Path
from totalsegmentator.python_api import totalsegmentator

def main():
    parser = argparse.ArgumentParser(description="Segment 104 anatomical structures in CT images.",
                                     epilog="Written by Jakob Wasserthal. If you use this tool please cite https://arxiv.org/abs/2208.05868")

    parser.add_argument("-i", metavar="filepath", dest="input",
                        help="CT nifti image or folder of dicom slices", 
                        type=lambda p: Path(p).absolute(), required=True)

    parser.add_argument("-o", metavar="directory", dest="output",
                        help="Output directory for segmentation masks", 
                        type=lambda p: Path(p).absolute(), required=True)

    parser.add_argument("-ot", "--output_type", choices=["nifti", "dicom"],
                    help="Select if segmentations shall be saved as Nifti or as Dicom RT Struct image.",
                    default="nifti")
                    
    parser.add_argument("-ml", "--ml", action="store_true", help="Save one multilabel image for all classes",
                        default=False)

    parser.add_argument("-nr", "--nr_thr_resamp", type=int, help="Nr of threads for resampling", default=1)

    parser.add_argument("-ns", "--nr_thr_saving", type=int, help="Nr of threads for saving segmentations", 
                        default=6)

    parser.add_argument("-f", "--fast", action="store_true", help="Run faster lower resolution model",
                        default=False)

    parser.add_argument("-t", "--nora_tag", type=str, 
                        help="tag in nora as mask. Pass nora project id as argument.",
                        default="None")

    parser.add_argument("-p", "--preview", action="store_true", 
                        help="Generate a png preview of segmentation",
                        default=False)

    # cerebral_bleed: Intracerebral hemorrhage 
    # liver_vessels: hepatic vessels
    parser.add_argument("-ta", "--task", choices=["total", "lung_vessels", "cerebral_bleed", 
                        "hip_implant", "coronary_arteries", "body", "pleural_pericard_effusion", 
                        "liver_vessels", "bones_extremities", "tissue_types",
                        "heartchambers_highres", "head", "aortic_branches", "heartchambers_test", 
                        "bones_tissue_test", "aortic_branches_test", "test"],
                        help="Select which model to use. This determines what is predicted.",
                        default="total")

    parser.add_argument("-rs", "--roi_subset", type=str, nargs="+",
                        help="Define a subset of classes to save (space separated list of class names). If running 1.5mm model, will only run the appropriate models for these rois.")

    parser.add_argument("-s", "--statistics", action="store_true", 
                        help="Calc volume (in mm3) and mean intensity. Results will be in statistics.json",
                        default=False)

    parser.add_argument("-r", "--radiomics", action="store_true", 
                        help="Calc radiomics features. Requires pyradiomics. Results will be in statistics_radiomics.json",
                        default=False)

    parser.add_argument("-cp", "--crop_path", help="Custom path to masks used for cropping. If not set will use output directory.", 
                        type=lambda p: Path(p).absolute(), default=None)

    parser.add_argument("-bs", "--body_seg", action="store_true", 
                        help="Do initial rough body segmentation and crop image to body region",
                        default=False)
    
    parser.add_argument("-fs", "--force_split", action="store_true", help="Process image in 3 chunks for less memory consumption",
                        default=False)

    parser.add_argument("-q", "--quiet", action="store_true", help="Print no intermediate outputs",
                        default=False)

    parser.add_argument("-v", "--verbose", action="store_true", help="Show more intermediate output",
                        default=False)

    # Tests:
    # 0: no testing behaviour activated
    # 1: total normal
    # 2: total fast -> removed because can run normally with cpu
    # 3: lung_vessels
    parser.add_argument("--test", metavar="0|1|3", choices=[0, 1, 3], type=int,
                        help="Only needed for unittesting.",
                        default=0)

    parser.add_argument('--version', action='version', version=require("TotalSegmentator")[0].version)

    args = parser.parse_args()

    seg_img = totalsegmentator(args.input, args.output, args.ml, args.nr_thr_resamp, args.nr_thr_saving,
                     args.fast, args.nora_tag, args.preview, args.task, args.roi_subset,
                     args.statistics, args.radiomics, args.crop_path, args.body_seg,
                     args.force_split, args.output_type, args.quiet, args.verbose, args.test)
    
    return seg_img

import nibabel as nib  
import numpy as np
def extract_organ_mask(input_img, organ_label_id, mask_value=2):
    # aorta = 52
    """
    Extracts a binary mask for a specific organ from a labeled NIFTI image.
    
    img_in: NIFTI image with segmentation labels.
    organ_name: Name of the organ to extract.
    label_map: Dictionary mapping label IDs to organ names.
    
    returns: Binary mask as a NIFTI image.
    """
    img_in = totalsegmentator(input=input_img, task='total', fast=True)
    data = img_in.get_fdata()

    # Find the label ID for the organ
    '''organ_label_id = None
    for label_id, name in label_map.items():
        if name == organ_name:
            organ_label_id = label_id
            break
    
    if organ_label_id is None:
        raise ValueError(f"Organ '{organ_name}' not found in the label map.")
    '''
    if organ_label_id>0:
    # Create a binary mask for the specified organ
        organ_mask_data = np.zeros_like(data)
        organ_mask_data[data == organ_label_id] = mask_value
    else:
        organ_mask_data=data
    # Create a new NIFTI image for the binary mask
    organ_mask_img = nib.Nifti1Image(organ_mask_data, img_in.affine, img_in.header)

    return organ_mask_img



if __name__ == '__main__':
   #seg_img = main()
    parser = argparse.ArgumentParser(description="spacing configuration.")
    parser.add_argument('--input_path', default='./logs/ct-volume-301.nii') # ct-volume-301.nii
    args = parser.parse_args()
    #image = r"E:\Projects\yang_proj\Task1\pelvis\1PA001\ct.nii.gz"
    input_path=args.input_path
    output_path=input_path.replace('.nii','_seg.nii')
    input_img = nib.load(input_path)
    print(input_img.shape)
    aorta_mask = extract_organ_mask(input_img, organ_label_id=0)
    nib.save(aorta_mask, output_path)


    
