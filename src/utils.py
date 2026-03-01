
import numpy as np

# 物理常数
F = 96485.3329  # 法拉第常数 [C/mol]
R = 8.314462618 # 气体常数 [J/(mol K)]
T = 298.15      # 温度 [K] (25 deg C)

# LCO (钴酸锂) 和 Graphite (石墨) 的 OCV 函数
def OCV_p_corrected(theta, params=None):
    """
    LCO (钴酸锂) 正极 OCV 函数
    theta: 化学计量比 (嵌锂度)，范围 0-1
    
    满充时: theta_p ≈ 0.4 (贫锂), U_p ≈ 4.1-4.2V
    放电时: theta_p ≈ 0.99 (富锂), U_p ≈ 3.5-3.7V
    
    参考: Ramadass et al. (2004), Safari & Delacourt (2011)
    """
    theta = np.clip(theta, 0.001, 0.998)
    
    # 标准 LCO OCV 公式 (来自电化学文献)
    # 这个公式在 theta ∈ [0.4, 0.99] 范围内给出 3.5-4.3V
    u = (4.06279 + 0.0677504 * np.tanh(-21.8502 * theta + 12.8268)
         - 0.105734 * (1.0 / ((1.00167 - theta) ** 0.379571) - 1.576)
         + 0.012637 * (1.0 / ((1.00167 - theta) ** 0.379571) - 1.576) ** 2
         - 0.002803 * (1.0 / ((1.00167 - theta) ** 0.379571) - 1.576) ** 3
         + 0.000237 * (1.0 / ((1.00167 - theta) ** 0.379571) - 1.576) ** 4
         - 0.0000072 * (1.0 / ((1.00167 - theta) ** 0.379571) - 1.576) ** 5)
    
    return u

def OCV_n_corrected(theta, params=None):
    # 石墨
    # 更好的石墨 OCV:
    u_graphite = 0.6379 + 0.5416 * np.exp(-305.5309 * theta) + 0.044 * np.tanh(-(theta - 0.1958) / 0.1088) - \
                 0.1978 * np.tanh((theta - 1.0571) / 0.0854) - 0.6875 * np.tanh((theta + 0.0617) / 0.0529) - \
                 0.0175 * np.tanh((theta - 0.6117) / 0.0335)
    return u_graphite

