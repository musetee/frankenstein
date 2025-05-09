import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import nrrd
def image_fov_from_volume_slice(nii_file_path, output_png_path, slice_index, center_x, center_y, fov_size=100):
    # Load the .nii.gz file
    if nii_file_path.endswith('.nii') or nii_file_path.endswith('.nii.gz'):
        nii_data = nib.load(nii_file_path)
        volume = nii_data.get_fdata()
    elif nii_file_path.endswith('.nrrd'):
    # Load the NRRD files
        volume, _ = nrrd.read(nii_file_path)

    # Select a slice (e.g., the middle slice along the third dimension)
    slice_data = volume[:, :, slice_index]

    # Rotate the slice clockwise by 90 degrees
    rotated_slice = np.rot90(slice_data, k=-1)  # Rotate clockwise

    # Normalize the rotated slice data for visualization (optional)
    rotated_slice = (rotated_slice - np.min(rotated_slice)) / (np.max(rotated_slice) - np.min(rotated_slice)) * 255
    rotated_slice = rotated_slice.astype(np.uint8)

    # Define the center and size of the square FOV

    # Calculate the boundaries of the square FOV
    x_start = max(0, center_x - fov_size // 2)
    x_end = min(rotated_slice.shape[1], center_x + fov_size // 2)
    y_start = max(0, center_y - fov_size // 2)
    y_end = min(rotated_slice.shape[0], center_y + fov_size // 2)

    # Crop the square FOV
    cropped_fov = rotated_slice[y_start:y_end, x_start:x_end]

    # Save the cropped FOV as a PNG image
    plt.figure(figsize=(8, 8))  # Set figure size (adjust as needed)
    plt.imshow(cropped_fov, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')  # Turn off axes
    plt.savefig(output_png_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()  # Close the figure to free memory

    print(f"Cropped FOV saved as {output_png_path}")
