import numpy as np
import nibabel as nib
def pad_to_target(array, target_shape, pad_value=0):
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

def load_nii(file_path):
    """Load a .nii.gz file and return the image array."""
    nii = nib.load(file_path)
    return nii.get_fdata(), nii.affine

def crop_or_pad_like(input_file, target_file):
    if input_file.shape[0] > target_file.shape[0]:
        print('crop input shape:', input_file.shape,' to target shape:', target_file.shape)
        input_file = crop_organ_to_fit_volume(input_file, target_file.shape)
    elif input_file.shape[0] < target_file.shape[0]:
        print('pad input shape:', input_file.shape,' to target shape:', target_file.shape)
        input_file = pad_to_target(input_file, target_file.shape)

    if input_file.shape[1] > target_file.shape[1]:
        print('crop input shape:', input_file.shape,' to target shape:', target_file.shape)
        input_file = crop_organ_to_fit_volume(input_file, target_file.shape)
    elif input_file.shape[1] < target_file.shape[1]:
        print('pad input shape:', input_file.shape,' to target shape:', target_file.shape)
        input_file = pad_to_target(input_file, target_file.shape)

    if input_file.shape[2] > target_file.shape[2]:
        print('crop input shape:', input_file.shape,' to target shape:', target_file.shape)
        input_file = crop_organ_to_fit_volume(input_file, target_file.shape)
    elif input_file.shape[2] < target_file.shape[2]:
        print('pad input shape:', input_file.shape,' to target shape:', target_file.shape)
        input_file = pad_to_target(input_file, target_file.shape)
    return input_file

def set_offset(volume, x_offset=0, y_offset=0, z_offset=0, pad_value=0):
    """
    Applies an offset to a 3D NumPy image volume, maintaining the original size.

    Parameters:
    - volume (ndarray): 3D NumPy array representing the image volume.
    - x_offset (int): Offset along the x-axis (depth).
    - y_offset (int): Offset along the y-axis (height).
    - z_offset (int): Offset along the z-axis (width).
    - pad_value (int or float): Value used to pad the shifted volume.

    Returns:
    - shifted_volume (ndarray): The offset volume with the same size as the input.
    """
    # Initialize a volume of the same shape with the padding value
    shifted_volume = np.full_like(volume, pad_value)

    # Calculate the ranges for source (input) and destination (shifted) regions
    x_start_src, x_end_src = max(0, -x_offset), min(volume.shape[0], volume.shape[0] - x_offset)
    y_start_src, y_end_src = max(0, -y_offset), min(volume.shape[1], volume.shape[1] - y_offset)
    z_start_src, z_end_src = max(0, -z_offset), min(volume.shape[2], volume.shape[2] - z_offset)

    x_start_dst, x_end_dst = max(0, x_offset), min(volume.shape[0], volume.shape[0] + x_offset)
    y_start_dst, y_end_dst = max(0, y_offset), min(volume.shape[1], volume.shape[1] + y_offset)
    z_start_dst, z_end_dst = max(0, z_offset), min(volume.shape[2], volume.shape[2] + z_offset)

    # Copy the data from the source to the shifted destination
    shifted_volume[x_start_dst:x_end_dst, y_start_dst:y_end_dst, z_start_dst:z_end_dst] = \
        volume[x_start_src:x_end_src, y_start_src:y_end_src, z_start_src:z_end_src]

    return shifted_volume