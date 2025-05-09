import numpy as np
import nibabel as nib
import nrrd

def extract_roi(ct_image, center_x=256, center_y=256, length=300, width=300):
    """
    Extract a Region of Interest (ROI) from the CT image.

    Parameters:
        ct_image (numpy.ndarray): The CT image data as a 3D NumPy array.
        center_x (int): X-coordinate of the center of the ROI.
        center_y (int): Y-coordinate of the center of the ROI.
        length (int): Length of the square ROI.
        width (int): Width of the square ROI.

    Returns:
        numpy.ndarray: The ROI extracted from the CT image.
    """
    half_length = length // 2
    half_width = width // 2

    start_x = max(0, center_x - half_length)
    end_x = min(ct_image.shape[0], center_x + half_length)
    start_y = max(0, center_y - half_width)
    end_y = min(ct_image.shape[1], center_y + half_width)

    return ct_image[:, start_x:end_x, start_y:end_y]

def calculate_contrast(ct_image):
    """
    Calculate the contrast of a CT image.

    Parameters:
        ct_image (numpy.ndarray): The CT image data as a 3D NumPy array.

    Returns:
        float: The contrast of the CT image.
    """
    # Assuming the CT image data ranges from -1024 to 3071 (typical Hounsfield Units range for CT scans)
    min_value = -1024.0
    max_value = 3071.0
    #ct_image_roi=extract_roi(ct_image, center_x=0, center_y=0, length=300, width=300)
    #ct_image_roi_mean=np.mean(ct_image_roi)
    contrast = np.abs((np.max(ct_image) - np.min(ct_image))) / (max_value - min_value)
    return contrast

def calculate_standard_deviation(ct_image):
    """
    Calculate the standard deviation of CT values in the image.

    Parameters:
        ct_image (numpy.ndarray): The CT image data as a 3D NumPy array.

    Returns:
        float: The standard deviation of CT values.
    """
    return np.std(ct_image)

import matplotlib.pyplot as plt

def plot_ct_value_distribution(ct_image):
    """
    Plot the distribution of CT values in the image.

    Parameters:
        ct_image (numpy.ndarray): The CT image data as a 3D NumPy array.
    """
    # Flatten the 3D array to a 1D array to get all CT values.
    ct_values = ct_image.flatten()

    # Create the histogram of CT values.
    plt.hist(ct_values, bins=100, range=(-1024, 3071), color='blue', alpha=0.7)
    plt.xlabel('CT Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of CT Values')
    plt.grid(True)
    plt.show()

def ct_windowing(ct_image, window_width, window_level):
    """
    Apply CT windowing to the CT image.

    Parameters:
        ct_image (numpy.ndarray): The CT image data as a 3D NumPy array.
        window_width (float): The window width.
        window_level (float): The window level.

    Returns:
        numpy.ndarray: The CT image data after applying windowing.
    """
    # Calculate the lower and upper bounds of the window.
    lower_bound = window_level - window_width / 2.0
    upper_bound = window_level + window_width / 2.0

    # Clip the CT values within the window bounds.
    ct_image_windowed = np.clip(ct_image, lower_bound, upper_bound)

    return ct_image_windowed

def main():
    # Replace 'your_ct_image.nii' with the path to your NIfTI CT image file.
    pcct_path = r'D:\Data\dataNeaotomAlpha\Nifti\2511\2511_2.nii.gz'
    cbct_path = r'D:\Data\M2OLIE_Phantom\pre_cbct.nrrd'
    nifti_file_path = pcct_path
    nrrd_file_path = cbct_path

    # Load the NIfTI CT image data using nibabel.
    ct_image_nifti = nib.load(nifti_file_path)
    ct_image_data = ct_image_nifti.get_fdata()

    #ct_image_data, header = nrrd.read(nrrd_file_path)

    window_width = 150
    window_level = 30
    #plot_ct_value_distribution(ct_image_data)
    ct_image_data = ct_windowing(ct_image_data, window_width, window_level)
    #plot_ct_value_distribution(ct_image_data)
    
    # cut roi
    center_x = ct_image_data.shape[0] // 2
    center_y = ct_image_data.shape[1] // 2
    ct_image_roi=extract_roi(ct_image_data, center_x=center_x, center_y=center_y, length=300, width=300)
    ct_image_roi_mean=np.mean(ct_image_roi)

    # Calculate contrast and standard deviation of CT values.
    contrast = calculate_contrast(ct_image_data)
    std_deviation = calculate_standard_deviation(ct_image_data)
    print(ct_image_data.shape)

    print("size of ROI:", ct_image_roi.shape)
    print("Mean of CT values in ROI:", ct_image_roi_mean)
    print("Contrast of CT image:", contrast)
    print("Standard Deviation of CT values:", std_deviation)
    

if __name__ == "__main__":
    main()
