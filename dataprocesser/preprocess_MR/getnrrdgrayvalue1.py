import nrrd
import numpy as np
import os
import matplotlib.pyplot as plt

def plot_histogram(data, output_file_path, max_gray_value):
    # 计算灰度值直方图
    hist, bins = np.histogram(data, bins=range(int(data.min()), int(data.max()) + 2))
    
    # 创建图形
    plt.figure()
    plt.bar(bins[:-1], hist, width=1.0, edgecolor='black')
    plt.xlabel('Gray Value')
    plt.ylabel('Frequency')
    plt.title('Gray Value Distribution')
    
    # 在图中显示最大灰度值
    plt.text(0.95, 0.95, f'Max Gray Value: {max_gray_value}', 
             horizontalalignment='right', verticalalignment='top', 
             transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    
    # 保存图形
    plt.savefig(output_file_path)
    plt.close()

def generate_histograms_from_folder(folder_path, output_folder_path):
    # 确保输出文件夹存在
    os.makedirs(output_folder_path, exist_ok=True)
    
    # 遍历文件夹中的所有文件
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.nrrd'):
                file_path = os.path.join(root, file)
                try:
                    # 读取nrrd文件
                    data, header = nrrd.read(file_path)
                    
                    # 将所有切片的数据展开为一个一维数组
                    flattened_data = data.flatten()

                    # 获取最大灰度值
                    max_gray_value = flattened_data.max()
                    
                    # 生成并保存灰度值直方图
                    output_file_path = os.path.join(output_folder_path, f"{os.path.splitext(file)[0]}.png")
                    plot_histogram(flattened_data, output_file_path, max_gray_value)
                except Exception as e:
                    print(f"Error reading the NRRD file {file_path}: {e}")

# 示例用法
folder_path = r'E:\XCATproject\SynthRad_GAN\trainingdata_new\MR_VIBE_nrrd_new'
output_folder_path = r'E:\XCATproject\SynthRad_GAN\trainingdata_new\MR_VIBE_nrrd_new_histograms'
generate_histograms_from_folder(folder_path, output_folder_path)