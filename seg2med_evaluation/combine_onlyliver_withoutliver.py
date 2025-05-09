import nibabel as nib
import os
import sys
sys.path.append('./synthetic_patient')
from utils import crop_or_pad_like

# Input file paths
withoutliver_file = r'D:\Project\seg2med_Project\synthrad_results\results_all_eval\phase2_xcat_synthetic\20241121_1110_Infer_ddpm2d_seg2med_without_liver\saved_outputs\volume_output\withoutliver_synthesized_volume.nii.gz'
onlyliver_file = r'D:\Project\seg2med_Project\synthrad_results\results_all_eval\phase2_xcat_synthetic\20241121_1324_Infer_ddpm2d_seg2med_only_liver\saved_outputs\volume_output\onlyliver_synthesized_volume.nii.gz'

# Load the NIfTI files
withoutliver_nifti = nib.load(withoutliver_file)
onlyliver_nifti = nib.load(onlyliver_file)

# Extract the voxel data
withoutliver_data = withoutliver_nifti.get_fdata()
onlyliver_data = onlyliver_nifti.get_fdata()


# Check if the dimensions match
if withoutliver_data.shape != onlyliver_data.shape:
    print(withoutliver_data.shape)
    print(onlyliver_data.shape)
    onlyliver_data = crop_or_pad_like(onlyliver_data, withoutliver_data)
    #raise ValueError("The dimensions of the two NIfTI files do not match. Cannot add them together.")

# Add the two volumes
combined_data = withoutliver_data + onlyliver_data

# Create a new NIfTI image
combined_nifti = nib.Nifti1Image(combined_data, affine=withoutliver_nifti.affine, header=withoutliver_nifti.header)

# Define the output file path
output_file = os.path.join(os.path.dirname(withoutliver_file), "combined_volume.nii.gz")

# Save the new NIfTI file
nib.save(combined_nifti, output_file)

print(f"Combined NIfTI file saved at: {output_file}")
