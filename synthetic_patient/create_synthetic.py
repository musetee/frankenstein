import nibabel as nib
import numpy as np
import cv2
from scipy.ndimage import zoom
import random
from tqdm import tqdm
def load_nii(file_path):
    """Load a .nii.gz file and return the image array."""
    nii = nib.load(file_path)
    return nii.get_fdata(), nii.affine

def resize_volume(volume, target_shape):
    """Resize a 3D volume to a target shape using zoom."""
    factors = [t / s for t, s in zip(target_shape, volume.shape)]
    return zoom(volume, factors, order=1)  # Linear interpolation

import numpy as np

def pad_to_target_size(array, target_shape, pad_value=0):
    """
    Pads a 3D NumPy array to the target shape with a specified value.

    Parameters:
    - array (ndarray): The input 3D array.
    - target_shape (tuple): The target shape (z, y, x) for the padded array.
    - pad_value (int or float): The value used for padding. Default is 0.

    Returns:
    - padded_array (ndarray): The padded array with the target shape.
    """
    # Calculate the required padding for each dimension
    z_pad = max(target_shape[0] - array.shape[0], 0)
    y_pad = max(target_shape[1] - array.shape[1], 0)
    x_pad = max(target_shape[2] - array.shape[2], 0)

    # Determine padding for each side
    z_pad_before, z_pad_after = z_pad // 2, z_pad - z_pad // 2
    y_pad_before, y_pad_after = y_pad // 2, y_pad - y_pad // 2
    x_pad_before, x_pad_after = x_pad // 2, x_pad - x_pad // 2

    # Apply padding
    padded_array = np.pad(
        array,
        pad_width=((z_pad_before, z_pad_after), (y_pad_before, y_pad_after), (x_pad_before, x_pad_after)),
        mode='constant',
        constant_values=pad_value
    )

    return padded_array

def create_synthetic_patient(organ_files, body_contour_file, target_shape=None, new_patient_file='synthetic_patient.nii.gz'):
    """
    Create a synthetic patient segmentation by combining organ segmentations.
    
    Parameters:
    - organ_files (dict): Dictionary where keys are organ names and values are file paths.
    - body_contour_file (str): File path to the body contour.
    - target_shape (tuple): The target shape of the synthetic segmentation.
    
    Returns:
    - synthetic_volume (ndarray): Synthetic 3D segmentation volume.
    """
    # Load the body contour
    body_contour, _ = load_nii(body_contour_file)
    if target_shape == None:
        target_shape = body_contour.shape
    else:
        body_contour = resize_volume(body_contour, target_shape)

    # Initialize the synthetic volume
    # synthetic_volume = np.zeros(target_shape, dtype=np.uint8)
    body_contour = body_contour.astype(np.uint8)
    # Loop through each organ and add to the synthetic volume
    synthetic_volume = np.zeros_like(body_contour, dtype=np.uint8)
    for organ_name, organ_file in tqdm(organ_files.items()):
        print(f'process {organ_name}')
        organ_seg, _ = load_nii(organ_file)
        organ_seg = organ_seg.astype(np.uint8)
        if np.max(organ_seg)==1:
            # Resize organ segmentation to target shape
            #organ_seg = resize_volume(organ_seg, target_shape)

            # Find a random position within the body contour
            # Define z_offset_range based on the body contour
            z_indices = np.where(body_contour > 0)[2]
            z_offset_range = (z_indices.min(), z_indices.max())

            # Select a random position within the body contour
            # z_min, z_max = body_indices[:, 2].min(), body_indices[:, 2].max()
            # print('z_min, z_max, organ_seg.shape[2]', z_min, z_max, organ_seg.shape[2])

            '''
            organ_indices = np.where(organ_seg==1)
            organ_shape_x = organ_indices[0].max() - organ_indices[0].min() + 1
            organ_shape_y = organ_indices[1].max() - organ_indices[1].min() + 1
            organ_shape_z = organ_indices[2].max() - organ_indices[2].min() + 1
            organ_min_x = organ_indices[0].min()
            organ_max_x = organ_indices[0].max()
            organ_min_y = organ_indices[1].min()
            organ_max_y = organ_indices[1].max()
            organ_min_z = organ_indices[2].min()
            organ_max_z = organ_indices[2].max()

            cropped_organ_seg = organ_seg[
                organ_min_x:organ_max_x + 1,
                organ_min_y:organ_max_y + 1,
                organ_min_z:organ_max_z + 1
                ]          
            
            print(organ_file, ': organ_shape_x, organ_shape_y, organ_shape_z', organ_shape_x, organ_shape_y, organ_shape_z)
            ''' 
            synthetic_volume, _ = random_zoom_organ_to_fit(synthetic_volume, body_contour, organ_seg, max_zoom_attempts=10)
            nib.save(nib.Nifti1Image(synthetic_volume, np.eye(4)), new_patient_file)

    return synthetic_volume
from scipy.ndimage import zoom
import numpy as np
import random



def random_zoom_organ_to_fit(synthetic_volume, body_contour, organ_seg, max_zoom_attempts=10):
    """
    Places an organ segmentation within the body contour using random zooming.

    Parameters:
    - body_contour (ndarray): 3D array of the body contour mask (binary).
    - organ_seg (ndarray): 3D array of the organ segmentation (binary mask).
    - max_zoom_attempts (int): Maximum attempts to randomly zoom and place the organ.

    Returns:
    - synthetic_volume (ndarray): The updated synthetic volume with the placed organ.
    - success (bool): Whether the organ was placed successfully.
    """
    # Ensure the body contour and organ segmentation are compatible
    
    if (
            organ_seg.shape[0] > body_contour.shape[0]
            or organ_seg.shape[1] > body_contour.shape[1]
            or organ_seg.shape[2] > body_contour.shape[2]
        ):
            organ_seg = crop_organ_to_fit_volume(organ_seg, synthetic_volume.shape)

    for attempt in range(max_zoom_attempts):
        
        scale_factor = 1.0 - 0.05*attempt
        scale_factor_sequence = (scale_factor, scale_factor, 1.0)
        zoomed_organ = zoom(organ_seg, scale_factor_sequence, order=1)  # Linear interpolation
        zoomed_organ = pad_to_target_size(zoomed_organ, synthetic_volume.shape, 0)
        
        print('zoomed_organ:', zoomed_organ.shape[0], zoomed_organ.shape[1], zoomed_organ.shape[2])
        print('body_contour:', body_contour.shape[0], body_contour.shape[1], body_contour.shape[2])
        
        synthetic_volume = synthetic_volume + zoomed_organ
        print('np.max(intermediate_volume)', np.max(synthetic_volume))
        if np.any(synthetic_volume >= 2):
            print(f"Attempt {attempt + 1} failed with scale factor {scale_factor:.2f}. Retrying...")
        else:
            print(f"Organ placed successfully with scale factor {scale_factor}.")
            return synthetic_volume, True

    print("Could not place the organ after maximum attempts.")
    return synthetic_volume, False


def crop_organ_to_fit_volume(organ_seg, synthetic_volume_shape):
    """
    Crops the organ segmentation to fit within the synthetic volume dimensions.

    Parameters:
    - organ_seg (ndarray): 3D array of the organ segmentation (binary mask).
    - synthetic_volume_shape (tuple): Shape of the synthetic volume (z, y, x).

    Returns:
    - cropped_organ (ndarray): Cropped 3D organ segmentation.
    """
    # Determine the cropping limits for each dimension
    crop_x = min(organ_seg.shape[0], synthetic_volume_shape[0])
    crop_y = min(organ_seg.shape[1], synthetic_volume_shape[1])
    crop_z = min(organ_seg.shape[2], synthetic_volume_shape[2])

    # Crop the organ segmentation
    cropped_organ = organ_seg[:crop_x, :crop_y, :crop_z]

    return cropped_organ

import os
import random

def get_random_organ_files(patient_dirs, organ_list=None):
    """
    Randomly selects one segmentation file for each organ from multiple patients.

    Parameters:
    - patient_dirs (list): List of patient directories containing organ segmentations.
    - organ_list (list, optional): List of organ names to select. Defaults to all organs in the first patient.

    Returns:
    - organ_files (dict): Dictionary where keys are organ names and values are file paths.
    """
    organ_files = {}

    # Ensure the input directories are valid
    for patient_dir in patient_dirs:
        if not os.path.isdir(patient_dir):
            raise ValueError(f"Invalid directory: {patient_dir}")

    # Use the first patient's directory to determine the organ list if not provided
    if organ_list is None:
        first_patient = patient_dirs[0]
        organ_list = [
            f.replace('.nii.gz', '') for f in os.listdir(os.path.join(first_patient, 'seg'))
            if f.endswith('.nii.gz')
        ]

    # Randomly select one file for each organ
    for organ in organ_list:
        candidates = []
        for patient_dir in patient_dirs:
            organ_path = os.path.join(patient_dir, 'seg', f"{organ}.nii.gz")
            if os.path.exists(organ_path):
                candidates.append(organ_path)

        if not candidates:
            raise ValueError(f"Organ {organ} not found in any patient directories.")

        # Randomly choose one file for the current organ
        organ_files[organ] = random.choice(candidates)

    return organ_files


# List of patient directories
patient_root = r'D:\Project\seg2med_Project\new_synthetic'
patient_IDs = [
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
]
patient_dirs = [os.path.join(patient_root, patient_ID) for patient_ID in patient_IDs]

# Get random organ segmentations
organ_files = get_random_organ_files(patient_dirs)

import shutil
destination = r'D:\Project\seg2med_Project\new_synthetic\synthetic_seg'
copy = False
# Print selected files
for organ, file_path in organ_files.items():
    print(f"{organ}: {file_path}")
    if copy:
        organ_seg, _ = load_nii(file_path)
        organ_seg = organ_seg.astype(np.uint8)
        if np.max(organ_seg)==1:
            shutil.copy(file_path, destination)

body_contour_file = r"D:\Project\seg2med_Project\new_synthetic\contour\ct-volume-34438427_i-Spiral  1.5  B30f - 5_34438427_5_mask_0.nii.gz"

# Save the synthetic volume
new_patient_file = r"D:\Project\seg2med_Project\new_synthetic\synthetic_patient.nii.gz"
synthetic_segmentation = create_synthetic_patient(organ_files, body_contour_file, target_shape=None, new_patient_file=new_patient_file)



