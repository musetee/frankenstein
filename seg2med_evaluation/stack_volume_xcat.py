import os
import nibabel as nib
import numpy as np
from glob import glob
import nrrd

def rename(folder_path):
    ## rename file as the format pID_type_sliceID
    # 1 delete the prefix, prefix: ct, ct_seg, ct_volume
    for filename in os.listdir(folder_path):
        old_file_path = os.path.join(folder_path, filename)
        parts = filename.split('_')
        if len(parts)>=3:
            patient_ID = parts[-6] + '_' + parts[-5] + '_' +parts[-4] + '_' + parts[-3]
            sign_name = parts[-2]
            slice_ID = parts[-1]
            
            if sign_name == 'seg':
                new_filename = f'{patient_ID}_seg_{slice_ID}'
            elif sign_name == 'input':
                new_filename = f'{patient_ID}_seg_{slice_ID}'
            elif sign_name == 'target':
                new_filename = f'{patient_ID}_target_{slice_ID}'
            else: # for synthesized and mask
                new_filename = f'{patient_ID}_{sign_name}_{slice_ID}'
            if not new_filename.endswith('.nii.gz'):
                new_filename += '.nii.gz'
            new_file_path = os.path.join(folder_path, new_filename)
            os.rename(old_file_path, new_file_path)
            print(f'Renamed {old_file_path} to {new_file_path}')

def renameV244(folder_path):
    for filename in os.listdir(folder_path):
        old_file_path = os.path.join(folder_path, filename)
        parts = filename.split('_')
        if len(parts)>=3:
            patient_ID = parts[-3]
            sign_name = parts[-2]
            slice_ID = int(parts[-1].split('.')[0])
            if patient_ID == 'V244':
                if slice_ID>=2499 and slice_ID<=2549:
                    slice_ID=slice_ID-2499+3136
                    new_filename = f'{patient_ID}_{sign_name}_{slice_ID}'
                    if not new_filename.endswith('.nii.gz'):
                        new_filename += '.nii.gz'
                    new_file_path = os.path.join(folder_path, new_filename)
                    os.rename(old_file_path, new_file_path)
                    print(f'Renamed {old_file_path} to {new_file_path}')
import re
from tqdm import tqdm

# Define a function to extract patient ID and slice number
def extract_patient_and_slice_info(file_path):
    filename = os.path.basename(file_path)
    parts = filename.split('_')
    patient_id = f'{parts[0]}_{parts[1]}_{parts[2]}_{parts[3]}_{parts[4]}'  # Assuming the patient ID format like '1PC082'
    slice_number = int(parts[-1].split('.')[0])  # Assuming slice number is in the last part of the filename
    return patient_id, slice_number

# Create a key function for sorting: by patient ID first, then by slice number
def sorting_key(file_path):
    patient_id, slice_number = extract_patient_and_slice_info(file_path)
    return patient_id, slice_number

def stack_volume(folder_path):
    # Get all file paths in the folder
    file_paths = sorted(glob(os.path.join(folder_path, '*.nii.gz')))

    seg_paths = [seg for seg in file_paths if 'seg' in os.path.basename(seg)]
    synthesized_paths = [synthesized for synthesized in file_paths if 'synthesized' in os.path.basename(synthesized)]
    target_paths = [target for target in file_paths if 'target' in os.path.basename(target)]

    
    print('len(seg_paths), len(synthesized_paths), len(target_paths): ')
    print(len(seg_paths), len(synthesized_paths), len(target_paths))
    stack_volume_for_one_type(seg_paths)
    stack_volume_for_one_type(synthesized_paths)
    stack_volume_for_one_type(target_paths)

def stack_volume_for_one_type(file_paths):
    # Sort files by slice number
    sorted_file_paths = sorted(file_paths, key=sorting_key)

    # Initialize variables
    current_patient_id = None
    slices = []

    # Process each file
    for file_path in tqdm(sorted_file_paths):
        # Extract patient ID from filename
        filename = os.path.basename(file_path)
        parts = filename.split('_')
        patient_id, _ = extract_patient_and_slice_info(file_path)

        # print(patient_id)
        # Load the image slice
        img = nib.load(file_path)
        data = img.get_fdata()

        if current_patient_id is None:
            current_patient_id = patient_id

        if patient_id != current_patient_id:
            # Save the volume for the previous patient
            # print(img.affine)
            save_volume(slices, current_patient_id, folder_path, img.affine)
            # Reset slices for the new patient
            slices = []
            current_patient_id = patient_id

        # Append the current slice to the list
        slices.append(data)

    # Save the last patient's volume
    if slices:
        save_volume(slices, current_patient_id, folder_path, img.affine)

def save_volume(slices, patient_id, folder_path, affine):
    # Stack slices into a 3D volume
    volume = np.stack(slices, axis=-1)
    # Create output folder
    save_folder_path = os.path.join(folder_path, 'volume_output')
    os.makedirs(save_folder_path, exist_ok=True)
    
    # Define output path
    output_format = 'nii.gz'
    output_path = os.path.join(save_folder_path, f'{patient_id}_volume.{output_format}')
    
    # Save volume based on format
    if output_format == 'nrrd':
        nrrd.write(output_path, volume)
    elif output_format == 'nii.gz':
        volume_img = nib.Nifti1Image(volume, affine)
        nib.save(volume_img, output_path)
    #print(f'Saved volume for patient {patient_id} at {output_path}')

# Set the path to the folder containing your slices
folder_path = r'D:\Project\seg2med_Project\synthrad_results\results_all_eval\20241119_0028_Infer_ddpm2d_seg2med_XCAT_CT_56Models_64slices\saved_outputs'
V244_folder_path = r'D:\Project\seg2med_Project\synthrad_results\results_all_eval\Results_Infer_Synthrad_Anish_CT\20241031_1129_Infer_ddpm2d_seg2med_noManualAorta\v244'

rename(folder_path)
# renameV244(V244_folder_path)
# stack_volume(folder_path)