import os
import pandas as pd
import numpy as np
import nrrd

def read_table():
    # 定义输入和输出文件夹
    input_folder = 'MR_VIBE_contour_seg1_seg2_nrrd_24_resetregion&greyvalue'  # 替换为实际的输入文件夹路径
    output_folder = 'MR_VIBE_contour_seg1_seg2_nrrd_24_resetregion&greyvalue_signal'  # 替换为实际的输出文件夹路径

    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 读取Excel文件
    excel_file = 'T1T2.xlsx'
    df = pd.read_excel(excel_file)


    # 提取表格数据
    gray_values = df.iloc[:, 0].to_numpy()
    T1_values = df.iloc[:, 1].to_numpy()
    T2_values = df.iloc[:, 2].to_numpy()
    rho_values = df.iloc[:, 3].to_numpy()

    # 遍历输入文件夹中的所有 .nrrd 文件
    for filename in os.listdir(input_folder):
        if filename.endswith('.nrrd'):
            # 构建完整的文件路径
            input_file_path = os.path.join(input_folder, filename)
            output_file_path = os.path.join(output_folder, filename)
            
            # 读取 nrrd 文件
            data, header = nrrd.read(input_file_path)
            
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
                            signal = calculate_signal_vibe(rho, alpha_rad, TR, T1, TE, T2)
                            data[i, j, k] = signal
            
            # 保存处理后的 nrrd 文件
            nrrd.write(output_file_path, data, header)

    print("All files have been processed.")


# 1.5 T setting
# vibe sequence intensity, default using "VIBE_in"
def calculate_signal_vibe(T1, T2, rho):
    # parameters for sequence
    TE = 4.54		#in ms
    TR = 7.25		#in ms
    alpha = 10		#in deg

    # convert alpha to radial

    alpha_rad = alpha * 2 * np.pi / 360
    signal = (rho * np.sin(alpha_rad) * (1 - np.exp(-TR / T1)) * 
              np.exp(-TE / T2) / (1 - np.cos(alpha_rad) * np.exp(-TR / T1)))
    return signal

def calculate_signal_vibe_opp(T1, T2, rho):
    # parameters for sequence
    TE = 2.30		#in ms
    TR = 7.25		#in ms
    alpha = 10		#in deg

    # convert alpha to radial

    alpha_rad = alpha * 2 * np.pi / 360
    signal = (rho * np.sin(alpha_rad) * (1 - np.exp(-TR / T1)) * 
              np.exp(-TE / T2) / (1 - np.cos(alpha_rad) * np.exp(-TR / T1)))
    return signal

def calculate_signal_vibe_dixon(T1, T2, rho):
    # parameters for sequence
    TE = 2.3		#in ms
    TR = 4.6		#in ms
    alpha = 10		#in deg
    W = 0.9
    F = 0.1
    delta_omega=220.0
    # convert alpha to radial
    alpha_rad = alpha * 2 * np.pi / 360

    # 信号前因子：包含T1恢复、T2*衰减、flip角影响
    A = rho*np.sin(alpha_rad) * \
        (1 - np.exp(-TR / T1)) / (1 - np.cos(alpha_rad) * np.exp(-TR / T1)) * \
        np.exp(-TE / T2)

    # 水脂信号：复数形式，脂肪成分有相位偏移
    phase_shift = np.exp(1j * 2 * np.pi * delta_omega * TE / 1000)  # ms → s
    signal = A * np.abs(W + F * phase_shift)
    return signal


import numpy as np

def calculate_signal_GRE_T1(T1, T2_star, rho):
    # S = M_0 \cdot \sin(\alpha) \cdot \frac{1 - e^{-TR / T_1}}{1 - \cos(\alpha) \cdot e^{-TR / T_1}} \cdot e^{-TE / T_2^*}
    """
    模拟 Gradient Echo (T1-weighted GRE) MR 图像的信号强度

    参数：
    T1      : 纵向弛豫时间（单位：ms）
    T2_star : 横向有效弛豫时间（T2*，单位：ms）
    rho     : 质子密度（单位：任意）

    返回：
    模拟信号强度
    """
    # 序列参数（你可以根据你自己的实际设置进行修改）
    TE = 4.0   # echo time in ms
    TR = 7  # repetition time in ms
    alpha_deg = 10  # flip angle in degrees

    # 转换为弧度
    alpha_rad = alpha_deg * np.pi / 180

    # GRE T1-weighted signal equation
    signal = rho * np.sin(alpha_rad) * \
             (1 - np.exp(-TR / T1)) / (1 - np.cos(alpha_rad) * np.exp(-TR / T1)) * \
             np.exp(-TE / T2_star)

    return signal

import numpy as np

def calculate_signal_T2_SPACE(T1, T2, rho=1.0):
    """
    仿真 T2-weighted SPACE/FSE 序列下的图像强度

    参数：
        T2  : T2 弛豫时间，单位 ms
        rho : 质子密度，默认 1.0
        TE  : 回波时间（Echo Time），单位 ms（默认 150ms）

    返回：
        模拟信号强度（float）
    """
    TE=150.0
    TR=2000.0
    signal = rho * (1 - np.exp(-TR / T1)) * np.exp(-TE / T2)
    return signal
