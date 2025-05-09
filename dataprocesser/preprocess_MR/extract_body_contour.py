import numpy as np
import nrrd
import cv2
import os

# python extract_body_contour.py --input_folder *** --output_folder ***

def create_body_mask(numpy_img, body_threshold=50):
    """
    Create a binary body mask and bone mask from a CT image numpy array.

    Args:
    numpy_img (np.array): A numpy array representation of a grayscale CT image, with intensity values from -1024 to 1500.

    Returns:
    np.array: A binary mask array where the entire body region is 1 and the background is 0.
    np.array: A binary mask array where the bone region is 2 and the rest is 0.
    """
    # Ensure we can handle negative values correctly
    numpy_img = numpy_img.astype(np.int16)

    # Threshold the image to separate body from the background
    body_mask = np.where(numpy_img > body_threshold, 1, 0).astype(np.uint8)

    # Find contours from the binary image
    contours, _ = cv2.findContours(body_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty mask and fill the largest contour
    mask = np.zeros_like(body_mask, dtype=np.uint8)  # Ensure mask is a numpy array
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        largest_contour = np.array(largest_contour).astype(np.int32)
        mask = np.ascontiguousarray(mask)  # Ensure mask is contiguous in memory
        print(f"Contour type: {type(largest_contour)}, Contour shape: {largest_contour.shape}")
        print(f"Mask type: {type(mask)}, Mask shape: {mask.shape}, Mask is contiguous: {mask.flags['C_CONTIGUOUS']}")
        cv2.drawContours(mask, [largest_contour], -1, 1, thickness=cv2.FILLED)

    combined_mask = mask

    return combined_mask

def process_volume(volume_path, save_path, body_threshold=50):
    """
    Process the entire 3D volume to extract body and bone masks.

    Args:
    volume_path (str): Path to the input nrrd file containing the CT volume.
    save_path (str): Path to save the output nrrd file with the extracted masks.
    """
    # Load the volume
    volume_data, header = nrrd.read(volume_path)
    
    # Initialize an empty array for the combined masks
    combined_masks = np.zeros_like(volume_data, dtype=np.int16)

    # Process each slice
    for i in range(volume_data.shape[2]):
        slice_data = volume_data[:, :, i]
        combined_mask = create_body_mask(slice_data, body_threshold)
        combined_masks[:, :, i] = combined_mask

    # Save the result as a new nrrd file
    nrrd.write(save_path, combined_masks, header)

def process_folder(input_folder, output_folder, body_threshold=50):
    """
    Process all nrrd files in the input folder to extract body and bone masks and save them to the output folder.

    Args:
    input_folder (str): Path to the folder containing input nrrd files.
    output_folder (str): Path to the folder to save the output nrrd files.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".nrrd"):
            input_path = os.path.join(input_folder, filename)
            output_filename = f"{filename}"
            #output_filename = f"extract_body&bone_{filename}"
            output_path = os.path.join(output_folder, output_filename)
            print(f"Processing {input_path} and saving to {output_path}")
            process_volume(input_path, output_path, body_threshold)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract body and bone masks from 3D CT volumes in a folder.")
    parser.add_argument('--input_folder', required=True, help="Path to the folder containing input nrrd files.")
    parser.add_argument('--output_folder', required=True, help="Path to the folder to save the output nrrd files.")
    parser.add_argument('--body_threshold', type=int, default=50, help="Threshold to separate body from background.")
    args = parser.parse_args()

    process_folder(args.input_folder, args.output_folder, args.body_threshold)