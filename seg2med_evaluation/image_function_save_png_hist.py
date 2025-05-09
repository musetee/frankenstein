import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TKAgg')
import os
import nrrd
from utils import pad_image
def pad_image(image, desired_size=[512,512],pad_value=0):
    # Calculate the required padding
    current_size = image.shape
    pad_rows = (desired_size[0] - current_size[0]) // 2
    pad_cols = (desired_size[1] - current_size[1]) // 2

    # Ensure padding is evenly distributed, with extra pixel on the "end" if needed
    padding = (
        (pad_rows, desired_size[0] - current_size[0] - pad_rows),  # Top, Bottom
        (pad_cols, desired_size[1] - current_size[1] - pad_cols)   # Left, Right
    )

    # Pad the image
    padded_image = np.pad(image, padding, mode='constant', constant_values=pad_value)  # Zero-padding
    return padded_image
import numpy as np

def pad_image_3D(image, desired_size=[None, 512, 512], pad_value=0):
    """
    Pads a 3D image to the desired size, with an option to skip padding on specific dimensions.

    Parameters:
        image (numpy.ndarray): Input 3D image array (depth, height, width).
        desired_size (list): Target size as [depth, height, width]. Use `None` to skip padding for a dimension.
        pad_value (int or float): Value to pad with (default: 0).

    Returns:
        numpy.ndarray: Padded 3D image.
    """
    # Current size of the image
    current_size = image.shape

    # Calculate padding for each dimension
    padding = []
    for i in range(3):  # Loop over depth, height, and width
        if desired_size[i] is None:
            # Skip padding for this dimension
            padding.append((0, 0))
        else:
            # Calculate padding
            pad_before = (desired_size[i] - current_size[i]) // 2
            pad_after = desired_size[i] - current_size[i] - pad_before
            padding.append((pad_before, pad_after))

    # Pad the image
    padded_image = np.pad(image, padding, mode='constant', constant_values=pad_value)
    return padded_image


def load_volume(file_path):
        if file_path.endswith('.nii') or file_path.endswith('.nii.gz'):
            nii_data = nib.load(file_path)
            return nii_data.get_fdata(), nii_data.affine
        elif file_path.endswith('.nrrd'):
            volume, header = nrrd.read(file_path)
            return volume, header
        else:
            raise ValueError("Unsupported file format. Please use .nii, .nii.gz, or .nrrd files.")
        
def save_volume(volume, output_path, output_format, header=None, affine=None):
    # Save volume based on format
    if output_format == 'nrrd':
        nrrd.write(output_path, volume, header)
    elif output_format == 'nii.gz':
        volume_img = nib.Nifti1Image(volume, affine)
        nib.save(volume_img, output_path)

def load_nii_and_save_png(nii_file_path, output_png_path, output_fov_png_path, slice_index, dpi=300, 
                          fov_box=False, center_x=0, center_y=0, fov_size=60, pad_size=[512,512],
                          ):
    # Load the .nii.gz file
    #nii_file_path = 'path/to/your/file.nii.gz'  # Update with your file path
    if nii_file_path.endswith('.nii') or nii_file_path.endswith('.nii.gz'):
        nii_data = nib.load(nii_file_path)
        volume = nii_data.get_fdata()
    elif nii_file_path.endswith('.nrrd'):
    # Load the NRRD files
        volume, _ = nrrd.read(nii_file_path)

    #print(volume.shape)
    #print(np.min(volume))
    # Select a slice (e.g., the middle slice along the third dimension)
    slice_data = volume[:, :, slice_index]
    slice_data = np.rot90(slice_data, k=-1)  # Rotate clockwise
    slice_data = np.fliplr(slice_data)
    #
    # Normalize the slice data for visualization (optional)
    slice_data = (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data)) * 255
    slice_data = pad_image(slice_data, desired_size=pad_size,pad_value=0)

    slice_data = slice_data.astype(np.uint8)

    # Save the slice as a PNG image
    # output_png_path = 'output_slice.png'  # Update with your desired output path
    # plt.imsave(output_png_path, slice_data, cmap='gray')
    # Create a rectangle patch
    
    fig, ax = plt.subplots(figsize=(8, 8))  # Set figure size
    ax.imshow(slice_data, cmap='gray', vmin=0, vmax=255)

    if fov_box:
        x_start = max(0, center_x - fov_size // 2)
        y_start = max(0, center_y - fov_size // 2)
        rect = plt.Rectangle((x_start, y_start), fov_size, fov_size, edgecolor='red', facecolor='none', linewidth=2)
        ax.add_patch(rect)  # Add the rectangle to the plot

        # Calculate the boundaries of the square FOV
        x_start = max(0, center_x - fov_size // 2)
        x_end = min(slice_data.shape[1], center_x + fov_size // 2)
        y_start = max(0, center_y - fov_size // 2)
        y_end = min(slice_data.shape[0], center_y + fov_size // 2)

        # Crop the square FOV
        cropped_fov = slice_data[y_start:y_end, x_start:x_end]

        # Save the cropped FOV as a PNG image
        plt.figure(figsize=(8, 8))  # Set figure size (adjust as needed)
        plt.imshow(cropped_fov, cmap='gray', vmin=0, vmax=255)
        plt.axis('off')  # Turn off axes
        plt.savefig(output_fov_png_path, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()  # Close the figure to free memory

        print(f"Cropped FOV saved as {output_fov_png_path}")

    plt.axis('off')  # Turn off axes
    plt.savefig(output_png_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()  # Close the figure to free memory
    print(f"Slice saved as {output_png_path}")

def load_nii_and_save_hist(nii_file_path, output_hist_path, slice_index=None):
    # Load the volume
    if nii_file_path.endswith('.nii') or nii_file_path.endswith('.nii.gz'):
        nii_data = nib.load(nii_file_path)
        volume = nii_data.get_fdata()
    elif nii_file_path.endswith('.nrrd'):
        volume, _ = nrrd.read(nii_file_path)
    else:
        raise ValueError("Unsupported file format. Please use .nii, .nii.gz, or .nrrd files.")
    
    # If a specific slice is provided, extract that slice
    if slice_index is not None:
        slice_data = volume[:, :, slice_index]
    else:
        slice_data = volume.flatten()
    
    # Flatten the data for histogram generation
    slice_data = slice_data.flatten()
    
    # Generate histogram
    plt.figure(figsize=(8, 6))
    plt.hist(slice_data, bins=256, color='blue', alpha=0.7)
    plt.title("Histogram of Image Intensities")
    plt.xlabel("Intensity Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    
    # Save histogram as PNG
    plt.savefig(output_hist_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Histogram saved as {output_hist_path}")

def load_nii_and_save_gt_synth_hist(gt_nii_file_path, synthesized_nii_file_path, output_hist_path, slice_index=None, range=[-500,500]):
    """
    Generate and save histograms of ground truth and synthesized volumes on the same diagram.

    Parameters:
        gt_nii_file_path (str): Path to the ground truth .nii or .nrrd file.
        synthesized_nii_file_path (str): Path to the synthesized .nii or .nrrd file.
        output_hist_path (str): Path to save the histogram.
        slice_index (int, optional): Slice index to process. If None, use the entire volume.
    """
    
    # Load the volumes
    gt_volume,_ = load_volume(gt_nii_file_path)
    synthesized_volume,_  = load_volume(synthesized_nii_file_path)
    
    # Extract the specific slice or flatten the entire volume
    if slice_index is not None:
        gt_data = gt_volume[:, :, slice_index].flatten()
        synthesized_data = synthesized_volume[:, :, slice_index].flatten()
    else:
        gt_data = gt_volume.flatten()
        synthesized_data = synthesized_volume.flatten()
    
    # Filter data to the range [-500, 500]
    lower_range, upper_range = range[0], range[1]
    gt_data = gt_data[(gt_data >= lower_range) & (gt_data <= upper_range)]
    synthesized_data = synthesized_data[(synthesized_data >= lower_range) & (synthesized_data <= upper_range)]
    
    # Generate histograms
    plt.figure(figsize=(10, 6))
    plt.hist(gt_data, bins=256, range=(lower_range, upper_range), color='blue', alpha=0.6, label='Ground Truth')
    plt.hist(synthesized_data, bins=256, range=(lower_range, upper_range), color='orange', alpha=0.6, label='Synthesized')
    
    # Add titles and labels
    plt.title("Histogram of Image Intensities", fontsize=14)
    plt.xlabel("Intensity Value", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Save the histogram as PNG
    plt.savefig(output_hist_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Histogram saved as {output_hist_path}")

def load_nii_and_save_gt_synth_percentage_hist(gt_nii_file_path, synthesized_nii_file_path, output_hist_path, slice_index=None, bins=256, range=[-500,500]):
    """
    Generate and save histograms of ground truth and synthesized volumes with percentage on the y-axis.

    Parameters:
        gt_nii_file_path (str): Path to the ground truth .nii or .nrrd file.
        synthesized_nii_file_path (str): Path to the synthesized .nii or .nrrd file.
        output_hist_path (str): Path to save the histogram.
        slice_index (int, optional): Slice index to process. If None, use the entire volume.
    """

    
    # Load the volumes
    gt_volume,_ = load_volume(gt_nii_file_path)
    synthesized_volume,_ = load_volume(synthesized_nii_file_path)
    
    # Extract the specific slice or flatten the entire volume
    if slice_index is not None:
        gt_data = gt_volume[:, :, slice_index].flatten()
        synthesized_data = synthesized_volume[:, :, slice_index].flatten()
    else:
        gt_data = gt_volume.flatten()
        synthesized_data = synthesized_volume.flatten()
    
    # Filter data to the range [-500, 500]
    lower_range, upper_range = range[0], range[1]
    gt_data = gt_data[(gt_data >= lower_range) & (gt_data <= upper_range)]
    synthesized_data = synthesized_data[(synthesized_data >= lower_range) & (synthesized_data <= upper_range)]
    
    # Normalize histogram counts to percentage
    total_gt = len(gt_data)
    total_synthesized = len(synthesized_data)
    
    plt.figure(figsize=(10, 6))
    plt.hist(gt_data, bins=bins, range=(lower_range, upper_range), weights=np.ones_like(gt_data) / total_gt * 100,
             color='blue', alpha=0.6, label='Ground Truth')
    plt.hist(synthesized_data, bins=bins, range=(lower_range, upper_range), weights=np.ones_like(synthesized_data) / total_synthesized * 100,
             color='orange', alpha=0.6, label='Synthesized')
    
    # Add titles and labels
    plt.title("Histogram of Image Intensities", fontsize=14)
    plt.xlabel("Intensity Value", fontsize=12)
    plt.ylabel("Percentage (%)", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Save the histogram as PNG
    plt.savefig(output_hist_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Histogram saved as {output_hist_path}")

import numpy as np

def calculate_histcc(gt_nii_file_path, synthesized_nii_file_path, slice_index, bins=256, range=[-500,1000]):
    """
    Calculate the Histogram Correlation Coefficient (HistCC) between two images.
    
    Parameters:
        image1 (numpy.ndarray): First image (grayscale).
        image2 (numpy.ndarray): Second image (grayscale).
        bins (int): Number of bins for the histograms (default: 256).
    
    Returns:
        float: Histogram Correlation Coefficient.
    """
    # Flatten the images into 1D arrays
    # Load the volumes
    gt_volume,_ = load_volume(gt_nii_file_path)
    synthesized_volume,_ = load_volume(synthesized_nii_file_path)
    
    # Extract the specific slice or flatten the entire volume
    if slice_index is not None:
        gt_data = gt_volume[:, :, slice_index].flatten()
        synthesized_data = synthesized_volume[:, :, slice_index].flatten()
    else:
        gt_data = gt_volume.flatten()
        synthesized_data = synthesized_volume.flatten()


    # Calculate histograms (with the same bins and range for both images)
    hist1, _ = np.histogram(gt_data, bins=bins, range=(range[0], range[1]), density=True)
    hist2, _ = np.histogram(synthesized_data, bins=bins, range=(range[0], range[1]), density=True)

    # Normalize histograms to sum to 1 (probability distributions)
    hist1 = hist1 / np.sum(hist1)
    hist2 = hist2 / np.sum(hist2)

    # Calculate the mean of the histograms
    mean1 = np.mean(hist1)
    mean2 = np.mean(hist2)

    # Calculate the numerator (cross-covariance)
    numerator = np.sum((hist1 - mean1) * (hist2 - mean2))

    # Calculate the denominator (product of standard deviations)
    denominator = np.sqrt(np.sum((hist1 - mean1)**2) * np.sum((hist2 - mean2)**2))

    # Compute the HistCC
    histcc = numerator / denominator if denominator != 0 else 0

    return histcc


