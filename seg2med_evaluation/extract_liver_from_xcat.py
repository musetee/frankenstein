import os
import nibabel as nib
import numpy as np

def extract_liver_mask(input_folder, output_folder, liver_value=52.98, liver_blood_vessel_value=54.41, tolerance=0.5):
    """
    Extract liver masks from XCAT volumes and save them as .nii.gz files.
    
    Parameters:
        input_folder (str): Path to the folder containing the input XCAT volumes.
        output_folder (str): Path to the folder where liver masks will be saved.
        liver_value (float): Pixel value corresponding to the liver in XCAT volumes.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  # Create output folder if it doesn't exist

    # List all .nii.gz files in the input folder
    xcat_files = [f for f in os.listdir(input_folder) if f.endswith('target_volume.nii.gz')]

    for file_name in xcat_files:
        input_path = os.path.join(input_folder, file_name)

        # Load the XCAT volume
        xcat_volume = nib.load(input_path)
        xcat_data = xcat_volume.get_fdata()

        # Create a binary mask for the liver
        liver_lower = liver_value - tolerance
        liver_upper = liver_value + tolerance
        
        liver_blood_vessel_lower = liver_blood_vessel_value - tolerance
        liver_blood_vessel_upper = liver_blood_vessel_value + tolerance

        liver_mask = ((xcat_data >= liver_lower) & (xcat_data <= liver_upper)).astype(np.uint8)
        #liver_blood_vessel_mask = ((xcat_data >= liver_blood_vessel_lower) & (xcat_data <= liver_blood_vessel_upper)).astype(np.uint8)

        liver_mask = liver_mask  #+ liver_blood_vessel_mask
        # Save the liver mask as a .nii.gz file
        liver_mask_nii = nib.Nifti1Image(liver_mask, affine=xcat_volume.affine, header=xcat_volume.header)
        output_file_name = file_name.replace("target_volume.nii.gz", "liver_mask_volume.nii.gz")
        output_path = os.path.join(output_folder, output_file_name)
        nib.save(liver_mask_nii, output_path)

        print(f"Liver mask saved: {output_path}")

# Specify the input folder containing XCAT volumes and output folder for liver masks
input_folder = r"D:\Project\seg2med_Project\synthrad_results\results_all_eval\phase2_xcat_synthetic\20241120_2348_Infer_ddpm2d_seg2med_synthetic_512_ct\saved_outputs\volume_output"  # Replace with the actual path to your XCAT volumes
output_folder = r"D:\Project\seg2med_Project\synthrad_results\results_all_eval\phase2_xcat_synthetic\20241120_2348_Infer_ddpm2d_seg2med_synthetic_512_ct\saved_outputs\volume_output"  # Replace with the desired path to save liver masks

# Extract liver masks
extract_liver_mask(input_folder, output_folder, liver_value=52.98, liver_blood_vessel_value=54.41, tolerance=0.05)
