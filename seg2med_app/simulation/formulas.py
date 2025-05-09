import numpy as np

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
    TR=2000.
    signal = rho * (1 - np.exp(-TR / T1)) * np.exp(-TE / T2)
    return signal
