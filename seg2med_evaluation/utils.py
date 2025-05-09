import numpy as np
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