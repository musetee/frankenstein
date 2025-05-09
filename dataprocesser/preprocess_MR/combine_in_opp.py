import os
import numpy as np
import SimpleITK as sitk

def extract_water_and_fat(in_phase_path, opp_phase_path):
    # 读取in-phase和opposed-phase图像
    in_phase_image = sitk.ReadImage(in_phase_path)
    opp_phase_image = sitk.ReadImage(opp_phase_path)

    in_phase_image = sitk.Cast(in_phase_image, sitk.sitkFloat32)
    opp_phase_image = sitk.Cast(opp_phase_image, sitk.sitkFloat32)

    in_phase_array = sitk.GetArrayFromImage(in_phase_image)
    opp_phase_array = sitk.GetArrayFromImage(opp_phase_image)

    # 计算水信号和脂肪信号
    water_signal = (in_phase_array + opp_phase_array) / 2.0
    fat_signal = (in_phase_array - opp_phase_array) / 2.0

    return water_signal, fat_signal, in_phase_image

def combine_signals(water_signal, fat_signal, weight_water=1, weight_fat=0):
    # 组合水信号和脂肪信号
    combined_signal = weight_water * water_signal + weight_fat * fat_signal
    return combined_signal

def process_and_save(in_phase_path, opp_phase_path, output_dir):
    # 提取水信号和脂肪信号
    water_signal, fat_signal, ref_image = extract_water_and_fat(in_phase_path, opp_phase_path)

    # 组合信号
    combined_signal = combine_signals(water_signal, fat_signal)

    # 将组合后的信号转换为SimpleITK图像
    combined_image = sitk.GetImageFromArray(combined_signal)

    # 设置图像属性（以in-phase图像为基准）
    combined_image.SetSpacing(ref_image.GetSpacing())
    combined_image.SetOrigin(ref_image.GetOrigin())
    combined_image.SetDirection(ref_image.GetDirection())

    # 构建输出路径
    base_name = os.path.basename(in_phase_path).replace('_in_tra', '').replace('.nrrd', '')
    combined_output_path = os.path.join(output_dir, base_name + '_combined_signal.nrrd')

    # 保存图像
    sitk.WriteImage(combined_image, combined_output_path)
    print(f"Processed and saved: {combined_output_path}")

def process_directory(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历目录中的文件，查找成对文件
    pairs = {}
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.nrrd'):
                key = file.split('_')[2]  # 获取唯一的标识符（例如：34438427）
                if key not in pairs:
                    pairs[key] = {}
                if '_in_tra' in file:
                    pairs[key]['in'] = os.path.join(root, file)
                elif '_opp_tra' in file:
                    pairs[key]['opp'] = os.path.join(root, file)

    # 处理成对文件
    for key, paths in pairs.items():
        if 'in' in paths and 'opp' in paths:
            process_and_save(paths['in'], paths['opp'], output_dir)
        else:
            print(f"Missing pair for {key}")

# 输入和输出目录
input_dir = r'E:\XCATproject\SynthRad_GAN\trainingdata_new\test0'
output_dir = r'E:\XCATproject\SynthRad_GAN\trainingdata_new\output'

# 处理目录中的文件
process_directory(input_dir, output_dir)