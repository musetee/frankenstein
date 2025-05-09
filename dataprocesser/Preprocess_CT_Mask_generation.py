import os
import pandas as pd
import numpy as np
import nrrd
import SimpleITK as sitk
import cv2

import numpy as np

def shift_to_min_zero(arr):
    """
    Shifts the input NumPy array so that the minimum value becomes 0.
    
    Parameters:
        arr (numpy.ndarray): The input array to shift.
    
    Returns:
        numpy.ndarray: The shifted array with the minimum value as 0.
    """
    min_value = np.min(arr)  # Find the minimum value
    shifted_array = arr - min_value  # Subtract the minimum value from all elements
    return shifted_array


def create_body_mask(numpy_img, body_threshold=-500, min_contour_area=10000):
    """
    Create a binary body mask from a CT image tensor, using a specific threshold for the body parts.

    Args:
    tensor_img (torch.Tensor): A tensor representation of a grayscale CT image, with intensity values from -1024 to 1500.

    Returns:
    torch.Tensor: A binary mask tensor where the entire body region is 1 and the background is 0.
    """
    # Convert tensor to numpy array
    numpy_img = np.ascontiguousarray(numpy_img.astype(np.int16))  # Ensure we can handle negative values correctly
    #numpy_img = numpy_img.astype(np.int16)

    # Threshold the image at -500 to separate potential body from the background
    binary_img = np.where(numpy_img > body_threshold, 1, 0).astype(np.uint8)

    # Find contours from the binary image
    contours, _ = cv2.findContours(binary_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty mask
    mask = np.zeros_like(binary_img)

    VERBOSE = False
    # Fill all detected body contours
    if contours:
        for contour in contours:
            if cv2.contourArea(contour) >= min_contour_area:
                if VERBOSE:
                    print('current contour area: ', cv2.contourArea(contour), 'threshold: ', min_contour_area)
                cv2.drawContours(mask, [contour], -1, 1, thickness=cv2.FILLED)

    return mask

def apply_mask(normalized_image_array, mask_array):
    return normalized_image_array * mask_array

def print_all_info(data, title):
    print(f'min, max of {title}:', np.min(data), np.max(data))

def process_CT_segmentation_numpy(mask, csv_simulation_values):
    #df = pd.read_csv(csv_file)
    df = csv_simulation_values
    # Create a dictionary to map organ index to HU values
    hu_values = dict(zip(df['Order Number'], df['HU Value']))
    order_begin_from_0 = True if df['Order Number'].min()==0 else False
    
    hu_mask = np.zeros_like(mask)
    # Value Assigment
    hu_mask[mask == 0] = -1000 # background
    for organ_index, hu_value in hu_values.items():
        assert isinstance(hu_value, int), f"Expected mask value an integer, but got {hu_value}. Ensure the mask is created by fine mode of totalsegmentator"
        assert isinstance(organ_index, int), f"Expected organ_index an integer, but got {organ_index}. Ensure the mask is created by fine mode of totalsegmentator"
        if order_begin_from_0:
            hu_mask[mask == (organ_index+1)] = hu_value # mask value begin from 1 as body value, other than 0 in TA2 table, so organ_index+1
        else:
            hu_mask[mask == (organ_index)] = hu_value
    return hu_mask


# 处理单个图像和分割图
def process_image(input_path, contour_path, seg_path, seg_tissue_path, csv_simulation_values, output_path1, output_path2, output_path3, body_threshold):
    # 读取原始 MR 图像和分割图
    if input_path.endswith('.nrrd'):
        img, header = nrrd.read(input_path)
        segmentation_img, header_seg = nrrd.read(seg_path)
        seg_tissue_img, header_seg_tissue = nrrd.read(seg_tissue_path)
    elif input_path.endswith('.nii.gz') or input_path.endswith('.nii'):
        import nibabel as nib
        img_metadata = nib.load(input_path)
        img = img_metadata.get_fdata()
        affine = img_metadata.affine

        seg_metadata = nib.load(seg_path)
        segmentation_img = seg_metadata.get_fdata()
        affine_seg = seg_metadata.affine

        seg_tissue_metadata = nib.load(seg_tissue_path)
        seg_tissue_img = seg_tissue_metadata.get_fdata()
        
    # extract contour
    body_contour = np.zeros_like(img, dtype=np.int16)
    for i in range(img.shape[-1]):
        slice_data = img[:, :, i]
        body_contour[:, :, i] = create_body_mask(slice_data, body_threshold=body_threshold)
    
    # CT images don't need additional normalization
    # 
    
    # normalize to 0-1
    img_normalized = shift_to_min_zero(img)
    # img_normalized = img_normalized/2000 # scale factor
    
    # apply mask to ct img
    masked_image = apply_mask(img_normalized, body_contour)
    
    # process the mask image
    seg = segmentation_img
    tissue = seg_tissue_img
    tissue[tissue!=0] += 200
    # Create a mask for overlapping areas
    overlap_mask = (seg > 0) & (tissue > 0)
    
    # For overlapping areas, keep the lower value (organ values in seg)
    merged_mask = tissue.copy()
    merged_mask[overlap_mask] = seg[overlap_mask]
    
    # Keep all non-overlapping areas
    merged_mask[seg > 0] = seg[seg > 0]

    combined_array = merged_mask + body_contour
    
    processed_segmentation = combined_array

    # assign simulation value to ct segmentation mask
    assigned_segmentation = process_CT_segmentation_numpy(combined_array, csv_simulation_values)
    
    if input_path.endswith('.nrrd'):
        # 保存处理后的 MR 图像
        nrrd.write(output_path1, masked_image, header)
        
        # 保存处理后的分割图
        nrrd.write(output_path2, processed_segmentation, header_seg)

        # save the body contour mask

    elif input_path.endswith('.nii.gz') or input_path.endswith('.nii'):
        img_processed = nib.Nifti1Image(masked_image, affine)
        nib.save(img_processed, output_path1)
        seg_processed = nib.Nifti1Image(processed_segmentation, affine_seg)
        nib.save(seg_processed, output_path2)
        contour_processed = nib.Nifti1Image(body_contour, affine_seg)
        assigned_segmentation_processed  = nib.Nifti1Image(assigned_segmentation, affine_seg)
        # Split the path into directory and filename
        directory, filename = os.path.split(output_path2)
        contour_filename = filename.replace('_seg_merged', '_contour')
        contour_path = os.path.join(directory, contour_filename)
        nib.save(contour_processed, contour_path)

        nib.save(assigned_segmentation_processed, output_path3)
        
    return processed_segmentation

def analyse_hist(input_path):
    if input_path.endswith('.nrrd'):
        img, header = nrrd.read(input_path)
    elif input_path.endswith('.nii.gz'):
        import nibabel as nib
        img_metadata = nib.load(input_path)
        img = img_metadata.get_fdata()
        affine = img_metadata.affine
    import numpy as np
    import matplotlib.pyplot as plt

    # Plot the histogram
    print('shape of img: ', img.shape)
    plt.hist(img[:, :, 50], bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Value Distribution')
    plt.show()


def process_csv(csv_file, output_root, csv_simulation_file, body_threshold=-500):
    # read csv to get simulation value
    csv_simulation_values = pd.read_csv(csv_simulation_file) #.to_numpy()
    #csv_simulation_values = pd.read_csv(csv_simulation_file)

    # check 2-dimensional csv_simulation_values 
    if csv_simulation_values.ndim == 1:
        raise ValueError("CSV should contain two columns: organ_index and simulation_value")

    if not os.path.exists(csv_file):
        print('csv:', csv_file)
        raise ValueError('csv_file must input a available csv file in simplified form: id, Aorta_diss, seg, img!')
    else:
        print(f'use csv: {csv_file}')
    
    data_frame = pd.read_csv(csv_file)
    if len(data_frame) == 0:
        raise RuntimeError(f"Found 0 images in: {csv_file}")
    patient_IDs = data_frame.iloc[:, 0].tolist()
    Aorta_diss = data_frame.iloc[:, 1].tolist()
    segs =  data_frame.iloc[:, 2].tolist()
    images = data_frame.iloc[:, 3].tolist()

    from tqdm import tqdm
    dataset_list = []
    for idx in tqdm(range(len(images))):
        if (images[idx].endswith('.nii.gz') and segs[idx].endswith('.nii.gz')) or \
            (images[idx].endswith('.nii') and segs[idx].endswith('.nii')):
            input_file_path = images[idx]
            seg_file_path = segs[idx]
            patient_id = patient_IDs[idx]
            ad = Aorta_diss[idx]
            seg_tissue_file_path = seg_file_path.replace("_seg","_seg_tissue")

            root_dir = os.path.dirname(input_file_path)
            
            # Get root path (directory path)
            root_path = os.path.dirname(seg_file_path)
            ct_processed_file_name = f"{patient_id}_ct_processed.nii.gz"
            seg_merged_file_name = f"{patient_id}_ct_seg_merged.nii.gz"
            seg_merged_assigned_mask_file_name = f"{patient_id}_ct_seg_merged_assigned_mask.nii.gz"
            
            os.makedirs(output_root, exist_ok=True)
            output_file_path1 = os.path.join(output_root, ct_processed_file_name)
            output_file_path2 = os.path.join(output_root, seg_merged_file_name)
            output_file_path3 = os.path.join(output_root, seg_merged_assigned_mask_file_name)
            print(f"Processing {input_file_path} with segmentation {seg_file_path}")
            print(f"Save results to {output_file_path1} and {output_file_path2} and {output_file_path3} \n")
            
            
            processed_seg = process_image(input_file_path, None, seg_file_path, seg_tissue_file_path, csv_simulation_values, output_file_path1, output_file_path2, output_file_path3, body_threshold)

            # processed_mr_csv_file = ...
            csv_mr_line = [patient_id,ad, output_file_path2, output_file_path1, output_file_path3]
            dataset_list.append(csv_mr_line)

    import csv
    output_csv_file=os.path.join(output_root, 'processed_csv_file.csv')
    with open(output_csv_file, 'w', newline='') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(['id', 'Aorta_diss', 'seg', 'img', 'seg_mask']) 
        csvwriter.writerows(dataset_list) 

if __name__ == "__main__":
    import argparse
    csv_file = r'E:\Projects\yang_proj\SynthRad_GAN\synthrad_conversion\datacsv\ct_synthrad_test_newserver.csv'
    output_root = r'E:\Projects\yang_proj\data\synthrad\processed'
    csv_simulation_file = r'E:\Projects\yang_proj\SynthRad_GAN\synthrad_conversion\TA2_CT_from1.csv'
    process_csv(csv_file, output_root, csv_simulation_file, body_threshold=-500)

    '''parser = argparse.ArgumentParser(description="Process MR images and segmentation maps, apply masks and replace grayscale values.")
    parser.add_argument('--input_folder1', required=True, help="Path to the folder containing input MR .nrrd files.")
    parser.add_argument('--input_folder2', required=True, help="Path to the folder containing segmentation .nrrd files.")
    parser.add_argument('--output_folder1', required=True, help="Path to the folder to save the output MR files.")
    parser.add_argument('--output_folder2', required=True, help="Path to the folder to save the output segmentation files.")
    parser.add_argument('--csv_simulation_file', required=True, help="CSV file containing simulated CT grayscale values.")
    parser.add_argument('--body_threshold', type=int, default=50, help="Threshold to separate body from background.")
    args = parser.parse_args()

    process_folder(args.input_folder1, args.input_folder2, args.output_folder1, args.output_folder2, args.csv_simulation_file, args.body_threshold)'''
