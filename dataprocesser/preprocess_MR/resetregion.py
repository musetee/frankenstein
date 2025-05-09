import os
import SimpleITK as sitk

def apply_mask(segmentation_path, contour_path, output_path):
    # 读取分割图和轮廓图
    segmentation_image = sitk.ReadImage(segmentation_path)
    contour_image = sitk.ReadImage(contour_path)

    contour_array = sitk.GetArrayFromImage(contour_image)
    segmentation_array = sitk.GetArrayFromImage(segmentation_image)

    # 应用掩膜
    masked_array = segmentation_array * contour_array

    # 将结果转换回SimpleITK图像
    masked_image = sitk.GetImageFromArray(masked_array)
    masked_image.SetSpacing(segmentation_image.GetSpacing())
    masked_image.SetOrigin(segmentation_image.GetOrigin())
    masked_image.SetDirection(segmentation_image.GetDirection())

    # 保存处理后的图像
    sitk.WriteImage(masked_image, output_path)
    print(f"Processed and saved: {output_path}")

def process_folders(segmentation_folder, contour_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    segmentation_files = os.listdir(segmentation_folder)
    contour_files = os.listdir(contour_folder)

    for seg_file in segmentation_files:
        if seg_file in contour_files:
            segmentation_path = os.path.join(segmentation_folder, seg_file)
            contour_path = os.path.join(contour_folder, seg_file)
            output_path = os.path.join(output_folder, seg_file)

            apply_mask(segmentation_path, contour_path, output_path)

# 输入和输出文件夹路径
segmentation_folder = 'MR_VIBE_seg2_nrrd_24'
contour_folder = 'MR_VIBE_contour_nrrd_24'
output_folder = 'MR_VIBE_seg2_nrrd_24_resetregion'

# 处理文件夹中的文件
process_folders(segmentation_folder, contour_folder, output_folder)