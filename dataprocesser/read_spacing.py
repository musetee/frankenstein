import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
import matplotlib.pyplot as plt

def print_nifti_spacing(file_path):
    """
    Reads a NIfTI file and prints the spacing information.

    Parameters:
    - file_path: Path to the NIfTI file (.nii or .nii.gz)
    """
    # Load the NIfTI file
    nifti_img = nib.load(file_path)
    
    # Get the header from the loaded NIfTI file
    header = nifti_img.header
    
    # Extract the voxel spacing information. 
    # The zooms attribute of the header contains the voxel sizes (spacing) in the x, y, and z dimensions
    spacing = header.get_zooms()
    
    print(f"Spacing information (in mm):\n X: {spacing[0]}mm\n Y: {spacing[1]}mm\n Z: {spacing[2]}mm")


def resample_volume(nifti_img, new_spacing=(1.0, 1.0, 1.0)):
    """
    Resamples the given MRI volume to the new spacing.

    Parameters:
    - nifti_img: The loaded NIfTI image.
    - new_spacing: The desired spacing as a tuple (x, y, z) in mm.

    Returns:
    - resampled_img: The resampled NIfTI image.
    """
    # Get the current spacing from the NIfTI header
    current_spacing = nifti_img.header.get_zooms()
    
    # Calculate the resampling factor for each dimension
    resampling_factor = [current / new for current, new in zip(current_spacing, new_spacing)]
    
    # Resample the volume using zoom function
    resampled_data = zoom(nifti_img.get_fdata(), resampling_factor, order=3)
    
    # Create a new NIfTI image from the resampled data
    resampled_img = nib.Nifti1Image(resampled_data, affine=nifti_img.affine)
    
    # Update the header to reflect the new spacing
    resampled_img.header.set_zooms(new_spacing)
    
    return resampled_img

def display_slice(nifti_img, slice_index, axis=2):
    """
    Displays a slice from the given MRI volume.

    Parameters:
    - nifti_img: The loaded NIfTI image.
    - slice_index: The index of the slice to display.
    - axis: The axis to slice along. 0 = x, 1 = y, 2 = z.
    """
    data = nifti_img.get_fdata()
    
    if axis == 0:
        slice_data = data[slice_index, :, :]
    elif axis == 1:
        slice_data = data[:, slice_index, :]
    else:
        slice_data = data[:, :, slice_index]
    
    plt.imshow(slice_data.T, cmap="gray", origin="lower")
    plt.axis("off")
    plt.show()

# Load your MRI volume
# nifti_img = nib.load('path_to_your_nifti_file.nii.gz')

# Resample the volume to new spacing, for example, (2.0, 2.0, 2.0) mm
# new_spacing = (2.0, 2.0, 2.0)
# resampled_img = resample_volume(nifti_img, new_spacing=new_spacing)

# Display a slice from the resampled volume, for example, the 50th slice along the z-axis
# display_slice(resampled_img, slice_index=50, axis=2)

# Example usage:
# Replace 'path_to_your_nifti_file.nii.gz' with the actual path to your NIfTI file
# print_nifti_spacing('path_to_your_nifti_file.nii.gz')
if __name__ == "__main__":
    path_to_your_nifti_file = 'E:\Results\MultistepReg\M2olie_Patientdata\Baseline\moved_images\moved_mr-volume-42553013_t1_vibe_opp_tra - 3.nii.gz'
    print_nifti_spacing(path_to_your_nifti_file)
    nifti_img = nib.load(path_to_your_nifti_file)
    new_spacing = (0.7, 0.8, 1.5)
    resampled_img = resample_volume(nifti_img, new_spacing=new_spacing)
    display_slice(resampled_img, slice_index=50, axis=2)