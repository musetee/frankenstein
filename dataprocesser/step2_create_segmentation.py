from dataprocesser.dataset_anish import list_img_ad_from_anish_csv
from dataprocesser.dataset_synthrad import list_img_pID_from_synthrad_folder  

from dataprocesser.dataset_anika import pair_list_from_anika_dataset
from dataprocesser.dataset_anika import all_list_from_anika_dataset
from dataprocesser.dataset_combined_csv import list_img_seg_ad_pIDs_from_new_simplified_csv
from dataprocesser.dataset_anika import all_list_from_anika_dataset_include_duplicate
from dataprocesser.dataset_dominik import all_list_from_dominik_dataset
from dataprocesser.dataset_xcat import list_img_pID_from_XCAT_folder
    
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def run(target_file_list=None, task='total', dataset='synthrad', device="gpu"):
    if target_file_list is None:
        target_file_list=create_dataset_list(dataset)
    multi_label_image=False if dataset == 'anika_newsynthetic' else True
    multi_label_image=False if dataset == 'synthesized' else True
    create_segmentation(target_file_list, task, device, multi_label_image)
    

def create_segmentation(dataset_list, task='total', device="gpu", multi_label_image=True):
    # task:
    # total 
    # total_mr
    # tissue_types_mr
    
    import nibabel as nib
    import nrrd
    import numpy as np
    from totalsegmentator.python_api import totalsegmentator
    for sample in dataset_list:
        input_path=sample
        print(f'create segmentation mask for {input_path}')
        if input_path.endswith('.nii') or input_path.endswith('.nii.gz'):
            if task == 'tissue_types_mr' or task == 'tissue_types':
                output_path=input_path.replace('.nii','_seg_tissue.nii')
            else:
                output_path=input_path.replace('.nii','_seg.nii')
            input_img = nib.load(input_path)

        elif input_path.endswith('.nrrd'):
            if task == 'tissue_types_mr' or task == 'tissue_types':
                output_path=input_path.replace('.nrrd','_seg_tissue.nii.gz')
            else:
                output_path=input_path.replace('.nrrd','_seg.nii.gz')
            np_img, header = nrrd.read(input_path)
            # Extract metadata for affine transformation
            spacing = header.get('space directions', None)
            if spacing is None:
                spacing = np.eye(3)  # Default to identity matrix if not available
            else:
                spacing = np.array(spacing)
            origin = header.get('space origin', [0, 0, 0])
            origin = np.array(origin)
            affine = np.zeros([4,4])
            affine[:3, :3] = spacing  # Set voxel dimensions
            affine[:3, 3] = origin  # Set the origin
            print('space directions', spacing)
            print('space origin', origin)
            print('affine', affine)
            input_img = nib.Nifti1Image(np_img, affine)

        totalsegmentator(input=input_img, output=output_path, task=task, fast=False, ml=multi_label_image, device=device)
        print(f'segmentation mask is saved as {output_path}')
    '''try:  
        pass
    except:
        print("An exception occurred")'''

from dataprocesser.step1_init_data_list import appart_img_and_seg
from dataprocesser.dataset_anika import (
    all_list_from_anika_dataset, 
    extract_patientID_from_Anika_dataset, 
    all_list_from_anika_dataset_include_duplicate)
from dataprocesser.dataset_synthrad import list_img_pID_from_synthrad_folder
from dataprocesser.dataset_anish import list_img_seg_ad_pIDs_from_anish_csv
from dataprocesser.dataset_dominik import all_list_from_dominik_dataset
from dataprocesser.dataset_combined_csv import list_img_seg_ad_pIDs_from_new_simplified_csv
from dataprocesser.dataset_xcat import list_img_pID_from_XCAT_folder
def create_dataset_list(dataset='anika_all'):
    def get_synthrad_files(data_dir, modality, saved_name):
        return list_img_pID_from_synthrad_folder(data_dir, accepted_modalities=modality, saved_name=saved_name)[0]

    def get_anika_pairs(ct_dir, mri_dir, mri_mode='t1_vibe_in'):
        matched_pairs = all_list_from_anika_dataset(ct_dir, mri_dir, mri_mode)
        return [paths['CT'] for paths in matched_pairs.values()], [paths['MRI'] for paths in matched_pairs.values()]

    def load_synthetic_folder(synthetic_folder, extract_id_func):
        assert os.path.isdir(synthetic_folder), f'{synthetic_folder} is not a valid directory'
        images, patient_IDs = [], []
        for roots, _, files in sorted(os.walk(synthetic_folder)):
            for file in files:
                if "seg_volume" not in file:
                    path = os.path.join(roots, file)
                    patient_IDs.append(extract_id_func(path))
                    images.append(path)
        return images

    dataset_handlers = {
        'anish': lambda: list_img_seg_ad_pIDs_from_new_simplified_csv(
            r'E:\Projects\yang_proj\SynthRad_GAN\synthrad_conversion\datacsv\healthy_dissec_newserver_new.csv'
        )[3],

        'synthrad_ct': lambda: get_synthrad_files(r'E:\Projects\yang_proj\data\synthrad\Task1\pelvis', 'ct', 'target_filenames.txt'),

        'synthrad_mr': lambda: get_synthrad_files(r'E:\Projects\yang_proj\data\synthrad\Task1_val\pelvis', 'mr', 'target_filenames.txt'),

        'anika': lambda: get_anika_pairs(
            r'E:\Projects\yang_proj\data\anika\CT',
            r'E:\Projects\yang_proj\data\anika\MR_registrated'
        )[0],

        'anika_all_ct': lambda: appart_img_and_seg(
            all_list_from_anika_dataset_include_duplicate(
                r'E:\Projects\yang_proj\data\anika\CT',
                r'E:\Projects\yang_proj\data\anika\MR_registrated'
            )[0]
        )[0],

        'anika_all_mr': lambda: appart_img_and_seg(
            all_list_from_anika_dataset_include_duplicate(
                r'E:\Projects\yang_proj\data\anika\CT',
                r'E:\Projects\yang_proj\data\anika\MR_registrated'
            )[1]
        )[0],

        'dominik': lambda: all_list_from_dominik_dataset(r'E:\Projects\yang_proj\data\Dominik_MR_VIBE'),

        'xcat_ct': lambda: list_img_pID_from_XCAT_folder(
            r'E:\Projects\chen_proj\aorta_XCAT2CT\CT_XCAT_aorta', saved_name=None
        )[0],

        'anika_newsynthetic': lambda: appart_img_and_seg(load_synthetic_folder(
            r'E:\Projects\yang_proj\data\anika\new_synthetic',
            lambda path: os.path.basename(os.path.dirname(path))
        ))[0],

        'synthesized': lambda: load_synthetic_folder(
            r'E:\Projects\yang_proj\data\ddpm_anika_more_512_for_2_seg\volume_output',
            lambda path: '_'.join(os.path.basename(path).split('_')[:2])
        )
    }

    if dataset not in dataset_handlers:
        raise ValueError(f"Unsupported dataset '{dataset}', please choose an available one!")

    return dataset_handlers[dataset]()

if __name__=='__main__':
    run()