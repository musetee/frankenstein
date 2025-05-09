import os
import SimpleITK as sitk

# 输入和输出文件夹路径
input_folder = 'MR_VIBE_nrrd'
output_folder = 'MR_VIBE_niigz'

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 遍历输入文件夹中的所有文件
for filename in os.listdir(input_folder):
    if filename.endswith('.nrrd'):
        # 构建完整的文件路径
        input_filepath = os.path.join(input_folder, filename)
        output_filepath = os.path.join(output_folder, filename.replace('.nrrd', '.nii.gz'))
        
        # 读取 NRRD 文件
        nrrd_image = sitk.ReadImage(input_filepath)
        
        # 保存为 NIFTI 文件
        sitk.WriteImage(nrrd_image, output_filepath)
        
        print(f"Converted {input_filepath} to {output_filepath}")

print("All files have been converted.")