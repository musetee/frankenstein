import pandas as pd
import numpy as np
import nrrd

# 读取Excel文件
excel_file = 'T1T2.xlsx'
df = pd.read_excel(excel_file)

# 提取表格数据
gray_values = df.iloc[:, 0].to_numpy()
T1_values = df.iloc[:, 1].to_numpy()
T2_values = df.iloc[:, 2].to_numpy()
rho_values = df.iloc[:, 3].to_numpy()

# 定义参数
TE = 4.54;		#in ms
TR = 7.25;		#in ms
alpha = 10;		#in deg

# 将 alpha 转换为弧度
alpha_rad = alpha * 2 * np.pi / 360

# 定义用于计算信号的函数
def calculate_signal(rho, alpha_rad, TR, T1, TE, T2):
    signal = (rho * np.sin(alpha_rad) * (1 - np.exp(-TR / T1)) * 
              np.exp(-TE / T2) / (1 - np.cos(alpha_rad) * np.exp(-TR / T1)))
    return signal

# 计算信号值
signal_values = np.array([calculate_signal(rho, alpha_rad, TR, T1, TE, T2) 
                          for rho, T1, T2 in zip(rho_values, T1_values, T2_values)])

# 创建新的DataFrame
new_df = df.copy()
new_df['Signal'] = signal_values

# 保存新表格为Excel文件
output_excel_file = 'output_T1T2_with_signal.xlsx'
new_df.to_excel(output_excel_file, index=False)

# 读取nrrd文件
nrrd_file = r'E:\XCATproject\SynthRad_GAN\trainingdata\MR_VIBE_contour&seg_nrrd\38647809.nrrd'  # 替换为你的实际路径
data, header = nrrd.read(nrrd_file)

# 将数据类型转换为浮点数以保留小数部分
data = data.astype(np.float64)

# 遍历每一个像素点并替换灰度值
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        for k in range(data.shape[2]):
            gray_value = data[i, j, k]
            if gray_value in gray_values:
                idx = np.where(gray_values == gray_value)[0][0]
                T1 = T1_values[idx]
                T2 = T2_values[idx]
                rho = rho_values[idx]
                signal = calculate_signal(rho, alpha_rad, TR, T1, TE, T2)
                data[i, j, k] = signal

# 保存处理后的nrrd文件
output_nrrd_file = 'processed_output.nrrd'
nrrd.write(output_nrrd_file, data, header)