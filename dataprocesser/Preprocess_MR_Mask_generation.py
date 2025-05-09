import os
import pandas as pd
import numpy as np
import nrrd
import SimpleITK as sitk
import cv2

from dataprocesser.preprocess_MR import step3_vibe_resetsignal
"""
该代码用于处理一组 MR 图像和对应的分割图，应用掩膜、进行归一化，并根据 CSV 文件中的仿真 MR 灰度值对分割图进行替换。最后将处理后的 MR 图像和分割图保存。

主要步骤：
1. **读取数据**：从指定的文件夹中读取 MR 图像和对应的分割图。
2. **归一化处理**：对 MR 图像进行归一化，将其值范围映射到 0 到 255 之间。
3. **轮廓提取**：从归一化后的 MR 图像中提取出主体区域的轮廓（根据给定的阈值分割），创建掩膜。
4. **掩膜应用**：将提取出的掩膜应用到归一化后的 MR 图像上，保留主体区域，抑制背景。
5. **分割图处理**：读取对应的分割图，并与提取出的轮廓进行叠加，之后根据 CSV 文件中的仿真 CT 值替换分割图中的灰度值。
6. **图像保存**：将处理后的 MR 图像和修改后的分割图保存到指定的输出文件夹中，保证其空间属性和几何信息与输入图像一致。
7. **输出**：在 ITK-SNAP 等医学图像工具中打开时, MR 图像和分割图能够保持同步和正确的比例显示。

函数简介：
- `normalize`: 对 MR 图像进行归一化处理，将像素值范围映射到 [0, 255]。
- `create_body_mask`: 从图像中提取出身体的轮廓，生成二值掩膜。
- `apply_mask`: 将提取的掩膜应用到 MR 图像上，保留轮廓内部的区域。
- `process_segmentation`: 读取分割图，并根据 CSV 文件中的仿真 CT 值对其灰度值进行替换。
- `process_image`: 处理单个 MR 图像及其对应的分割图，包括归一化、轮廓提取、掩膜应用、分割图处理等。
- `process_folder`: 处理整个文件夹中的 MR 图像和分割图，逐一处理所有图像并保存结果。
"""

# 归一化函数
def normalize(img, vmin_out=0, vmax_out=1, norm_min_v=None, norm_max_v=None, epsilon=1e-5):
    if norm_min_v is None and norm_max_v is None:
        norm_min_v = np.min(img)
        norm_max_v = np.max(img)
    img = np.clip(img, norm_min_v, norm_max_v)
    img = (img - norm_min_v) / (norm_max_v - norm_min_v + epsilon)
    img = img * (vmax_out - vmin_out) + vmin_out
    return img

# 创建轮廓掩膜
def create_body_mask_simple(numpy_img, body_threshold=50):
    numpy_img = numpy_img.astype(np.int16)
    body_mask = np.where(numpy_img > body_threshold, 1, 0).astype(np.uint8)
    contours, _ = cv2.findContours(body_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(body_mask, dtype=np.uint8)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        mask = np.ascontiguousarray(mask)
        largest_contour = np.ascontiguousarray(largest_contour)
        cv2.drawContours(mask, [largest_contour], -1, 1, thickness=cv2.FILLED)

    return mask

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

# process the segmentation, replace the classes with simulated MR values
def process_segmentation(combined_array, csv_simulation_values, mr_signal_formula=step3_vibe_resetsignal.calculate_signal_vibe):
    combined_array = combined_array.astype(np.int16)
    print_all_info(combined_array, 'combine')
    # two columns of  unique value 和 simulation value
    # the first element will not be included
    organ_indexs = csv_simulation_values[1:, 0]  # first column: organ index
    T1_values = csv_simulation_values[1:, 1]  # second column: simulate MRI value
    T2_values = csv_simulation_values[1:, 2]
    Rho_values = csv_simulation_values[1:, 3]
    order_begin_from_0 = True if organ_indexs.astype(int).min()==0 else False
    #print('organ order number begin from 0:', order_begin_from_0)
    #print(organ_indexs)
    assign_value_mask = np.zeros_like(combined_array)

    step=0
    for step in range(len(organ_indexs)):
        organ_index = organ_indexs[step] # in csv file, organs begin with 1       
        t1 = float(T1_values[step])
        t2 = float(T2_values[step])
        rho = float(Rho_values[step])

        simulation_value = mr_signal_formula(t1, t2, rho)
        organ_index = int(organ_index)
        if order_begin_from_0:
            #print("order in csv begin from 0")
            assign_value_mask[combined_array == organ_index+1] = simulation_value #  organ_index+ 1
        else:
            #print("order in csv begin from 1")
            assign_value_mask[combined_array == organ_index] = simulation_value
        step+=1
    print_all_info(assign_value_mask, 'assignment')
    return assign_value_mask

# 处理单个图像和分割图
def process_image(input_path, contour_path, seg_path, csv_simulation_values, output_path1, output_path2, body_threshold):
    # 读取原始 MR 图像和分割图
    if input_path.endswith('.nrrd'):
        img, header = nrrd.read(input_path)
        segmentation_img, header_seg = nrrd.read(seg_path)
    elif input_path.endswith('.nii.gz') or input_path.endswith('.nii'):
        import nibabel as nib
        img_metadata = nib.load(input_path)
        img = img_metadata.get_fdata()
        affine = img_metadata.affine
        
        seg_metadata = nib.load(seg_path)
        segmentation_img = seg_metadata.get_fdata()
    
    # 归一化处理
    norm_max=255 #255
    low_percentile = 5
    high_percentile = 90
    img_normalized = normalize(img, 0, norm_max, np.percentile(img, low_percentile), np.percentile(img, high_percentile), epsilon=0)
    
    # 提取轮廓图
    body_contour = np.zeros_like(img, dtype=np.int16)
    for i in range(img.shape[2]):
        slice_data = img[:, :, i]
        body_contour[:, :, i] = create_body_mask(slice_data, body_threshold=body_threshold)
    
    # 应用掩膜到归一化 MR 图像
    masked_image = apply_mask(img_normalized, body_contour)
    
    # 处理分割图
    # add contour background to the segmentation (all region inside body + 1)
    combined_array = segmentation_img + body_contour
    combined_array = np.clip(combined_array, 0, np.max(segmentation_img) + 1)
    print_all_info(segmentation_img, 'seg')
    processed_segmentation = process_segmentation(combined_array, csv_simulation_values)
    
    # normalize to 0-1
    # masked_image = masked_image/norm_max
    # processed_segmentation = processed_segmentation/norm_max

    if input_path.endswith('.nrrd'):
        # 保存处理后的 MR 图像
        nrrd.write(output_path1, masked_image, header)
        
        # 保存处理后的分割图
        nrrd.write(output_path2, processed_segmentation, header_seg)

        # save the body contour mask

    elif input_path.endswith('.nii.gz') or input_path.endswith('.nii'):
        img_processed = nib.Nifti1Image(masked_image, affine)
        nib.save(img_processed, output_path1)
        seg_processed = nib.Nifti1Image(processed_segmentation, affine)
        nib.save(seg_processed, output_path2)
        contour_processed = nib.Nifti1Image(body_contour, affine)
        
        # Split the path into directory and filename
        directory, filename = os.path.split(output_path2)
        new_filename = filename.replace('seg', 'contour')
        contour_path = os.path.join(directory, new_filename)

        nib.save(contour_processed, contour_path)
    return processed_segmentation

# 处理文件夹
def process_folder(input_folder1, input_folder2, output_folder1, output_folder2, csv_simulation_file, body_threshold=50):
    # 读取CSV文件获取仿真CT灰度值 (两列)
    csv_simulation_values = pd.read_csv(csv_simulation_file, header=None).to_numpy()

    # 检查 csv_simulation_values 是否是二维数组
    if csv_simulation_values.ndim == 1:
        raise ValueError("CSV 文件格式不正确，应该包含两列：organ_index 和 simulation_value")

    # 确保输出文件夹存在
    os.makedirs(output_folder1, exist_ok=True)
    os.makedirs(output_folder2, exist_ok=True)

    for filename in os.listdir(input_folder1):
        if filename.endswith('.nrrd'):
            input_file_path = os.path.join(input_folder1, filename)
            seg_file_path = os.path.join(input_folder2, filename)
            output_file_path1 = os.path.join(output_folder1, filename)
            output_file_path2 = os.path.join(output_folder2, filename)

            print(f"Processing {input_file_path} with segmentation {seg_file_path}")
            processed_seg = process_image(input_file_path, None, seg_file_path, csv_simulation_values, output_file_path1, output_file_path2, body_threshold)

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


def process_csv(csv_file, output_folder1, output_folder2, csv_simulation_file, body_threshold=50, output_mr_csv_file='processed_mr_csv_file.csv'):
    # 读取CSV文件获取仿真CT灰度值 (两列)
    csv_simulation_values = pd.read_csv(csv_simulation_file, header=None).to_numpy()
    #csv_simulation_values = pd.read_csv(csv_simulation_file)

    # 检查 csv_simulation_values 是否是二维数组
    if csv_simulation_values.ndim == 1:
        raise ValueError("CSV 文件格式不正确，应该包含两列：organ_index 和 simulation_value")

    # 确保输出文件夹存在
    os.makedirs(output_folder1, exist_ok=True)
    os.makedirs(output_folder2, exist_ok=True)
    
    from step1_init_data_list import list_img_seg_ad_pIDs_from_new_simplified_csv
    patient_IDs, Aorta_diss, segs, images = list_img_seg_ad_pIDs_from_new_simplified_csv(csv_file)
    from tqdm import tqdm
    dataset_list = []
    for idx in tqdm(range(len(images))):
        if (images[idx].endswith('.nii.gz') and segs[idx].endswith('.nii.gz')) or \
            (images[idx].endswith('.nii') and segs[idx].endswith('.nii')):
            input_file_path = images[idx]
            seg_file_path = segs[idx]
            patient_id = patient_IDs[idx]
            ad = Aorta_diss[idx]
            root_dir = os.path.dirname(input_file_path)
            
            output_file_path1 = os.path.join(output_folder1, os.path.relpath(input_file_path, start=root_dir))
            
            synthrad_basic_mr_name = 'mr'
            synthrad_basic_seg_name = 'mr_merged_seg'
            if os.path.basename(output_file_path1) == f'{synthrad_basic_mr_name}.nii.gz' or \
                os.path.basename(output_file_path1) == f'{synthrad_basic_mr_name}.nii':
                # Insert the patient ID in the filename
                output_file_path1 = output_file_path1.replace(f'{synthrad_basic_mr_name}', f'mr_{patient_id}')

            output_file_path2 = os.path.join(output_folder2, os.path.relpath(seg_file_path, start=root_dir))
            
            if os.path.basename(output_file_path2) == f'{synthrad_basic_seg_name}.nii.gz' or \
                os.path.basename(output_file_path2) == f'{synthrad_basic_seg_name}.nii':
                # Insert the patient ID in the filename
                output_file_path2 = output_file_path2.replace(f'{synthrad_basic_seg_name}', f'mr_seg_{patient_id}')

            print(f"Processing {input_file_path} with segmentation {seg_file_path}")
            print(f"Save results to {output_file_path1} and {output_file_path2}")
            
            processed_seg = process_image(input_file_path, None, seg_file_path, csv_simulation_values, output_file_path1, output_file_path2, body_threshold)

            # processed_mr_csv_file = ...
            csv_mr_line = [patient_id,ad,output_file_path2,output_file_path1]
            dataset_list.append(csv_mr_line)

    import csv
    with open(output_mr_csv_file, 'w', newline='') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(['id', 'Aorta_diss', 'seg', 'img'])
        csvwriter.writerows(dataset_list) 

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process MR images and segmentation maps, apply masks and replace grayscale values.")
    parser.add_argument('--input_folder1', required=True, help="Path to the folder containing input MR .nrrd files.")
    parser.add_argument('--input_folder2', required=True, help="Path to the folder containing segmentation .nrrd files.")
    parser.add_argument('--output_folder1', required=True, help="Path to the folder to save the output MR files.")
    parser.add_argument('--output_folder2', required=True, help="Path to the folder to save the output segmentation files.")
    parser.add_argument('--csv_simulation_file', required=True, help="CSV file containing simulated CT grayscale values.")
    parser.add_argument('--body_threshold', type=int, default=50, help="Threshold to separate body from background.")
    args = parser.parse_args()

    process_folder(args.input_folder1, args.input_folder2, args.output_folder1, args.output_folder2, args.csv_simulation_file, args.body_threshold)