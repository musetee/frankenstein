import os
import nrrd
import nibabel as nib
import numpy as np

# Define the input folder containing .nrrd files and output folder for .nii.gz files
input_folder = r"D:\Project\seg2med_Project\synthrad_results\results_all_eval\zhilin_results_testset10\volume_output"  # Replace with the path to your folder
output_folder = r"D:\Project\seg2med_Project\synthrad_results\results_all_eval\zhilin_results_testset10\slice_output\target"  # Replace with the desired output folder path

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)
file_name_criterian = "synthesis"
file_name_criterian2 = "mask"
replace_file_name = "target"
# Iterate through all files in the input folder
for file_name in os.listdir(input_folder):
    if file_name.endswith(".nrrd") and file_name_criterian not in file_name and file_name_criterian2 not in file_name:
        file_path = os.path.join(input_folder, file_name)
        
        # Read the .nrrd file
        data, header = nrrd.read(file_path)
        
        # Loop through slices and save each as a .nii.gz
        for i in range(data.shape[2]):  # Assuming slices are along the third axis
            slice_data = data[:, :, i]  # Extract a single slice
            slice_nii = nib.Nifti1Image(slice_data, affine=np.eye(4))  # Create a NIfTI object
            
            # Construct output file name
            output_file_name = f"{os.path.splitext(file_name)[0]}_target_{i:03d}.nii.gz" # .replace(file_name_criterian, replace_file_name)
            output_file_path = os.path.join(output_folder, output_file_name)
            
            # Save the slice as .nii.gz
            nib.save(slice_nii, output_file_path)
        
        print(f"Processed {file_name}: Saved {data.shape[2]} slices.")

print("Processing complete!")
