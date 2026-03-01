
import numpy as np
from src.utils import F, R, T, OCV_p_corrected, OCV_n_corrected

class DualElectrodeSPM:
    def __init__(self, params=None, initial_soc=1.0):
        """
        双电极单粒子模型 (SPM)
        
        参数:
        params: 包含8个参数的字典或列表:
        [D_p, D_n, k_p, k_n, R_p, R_n, c_max_p, c_max_n]
        """
        # 默认几何参数 (针对 18650 电池近似)
        self.L_n = 80e-6  # 负极厚度 [m]
        self.L_p = 70e-6  # 正极厚度 [m]
        self.A_geo = 0.042 # 几何面积 [m^2] (根据 ~2.0Ah 18650 调整)
        self.eps_s_n = 0.6 # 负极活性物质体积分数
        self.eps_s_p = 0.5 # 正极活性物质体积分数
        
        # 导出活性面积 (近似)
        # a_s = 3 * eps / R_s
        # 总面积 = A_geo * L * a_s = A_geo * L * 3 * eps / R_s
        
        self.update_params(params)
        
        # 初始状态
        # 需要根据 SOC 设置初始浓度
        # 对于满充 (SOC=1):
        # theta_n ~ 0.8-0.9, theta_p ~ 0.4-0.5 (取决于平衡)
        self.soc = initial_soc
        self.reset_state(initial_soc)

    def update_params(self, params):
        if params is None:
            # 默认参数 (近似值)
            self.D_p = 1e-14 # 正极扩散系数 [m^2/s]
            self.D_n = 3e-14 # 负极扩散系数 [m^2/s]
            self.k_p = 2e-11 # 正极反应速率常数
            self.k_n = 2e-11 # 负极反应速率常数
            self.R_p = 8e-6  # 正极颗粒半径 [m] (根据18650典型值调整)
            self.R_n = 10e-6 # 负极颗粒半径 [m] (根据18650典型值调整)
            self.c_max_p = 51385 # 正极最大浓度 [mol/m^3] (LCO理论值)
            self.c_max_n = 33133 # 负极最大浓度 [mol/m^3] (石墨理论值)
            self.R_f = 0.05 # 膜电阻 (固定或额外?)
        else:
            if isinstance(params, (list, np.ndarray)):
                if len(params) == 4:
                     # 仅辨识核心动力学参数 [D_p, D_n, k_p, k_n]
                     self.D_p, self.D_n, self.k_p, self.k_n = params
                     # 其余保持默认常量
                     if not hasattr(self, 'R_p'): self.R_p = 8e-6
                     if not hasattr(self, 'R_n'): self.R_n = 10e-6
                     if not hasattr(self, 'c_max_p'): self.c_max_p = 51385
                     if not hasattr(self, 'c_max_n'): self.c_max_n = 33133
                     if not hasattr(self, 'R_f'): self.R_f = 0.05
                elif len(params) == 5:
                     # 辨识 [D_p, D_n, k_p, k_n, R_f]
                     self.D_p, self.D_n, self.k_p, self.k_n, self.R_f = params
                     # 其余保持默认常量
                     if not hasattr(self, 'R_p'): self.R_p = 8e-6
                     if not hasattr(self, 'R_n'): self.R_n = 10e-6
                     if not hasattr(self, 'c_max_p'): self.c_max_p = 51385
                     if not hasattr(self, 'c_max_n'): self.c_max_n = 33133
                elif len(params) == 8:
                     self.D_p, self.D_n, self.k_p, self.k_n, self.R_p, self.R_n, self.c_max_p, self.c_max_n = params
                     # 保持 R_f 默认或从之前的设置继承
                     if not hasattr(self, 'R_f'): self.R_f = 0.05
                elif len(params) == 9:
                     self.D_p, self.D_n, self.k_p, self.k_n, self.R_p, self.R_n, self.c_max_p, self.c_max_n, self.R_f = params
            else:
                self.__dict__.update(params)
        
        # 更新依赖于 R_p, R_n 的导出几何参数
        self.a_s_n = 3 * self.eps_s_n / self.R_n
        self.a_s_p = 3 * self.eps_s_p / self.R_p
        self.Area_total_n = self.A_geo * self.L_n * self.a_s_n
        self.Area_total_p = self.A_geo * self.L_p * self.a_s_p

    def compute_ocv_voltage(self, soc):
        """计算给定 SOC 下的开路电压"""
        # NASA B0005 典型 SOC-Theta 映射
        theta_n = 0.01 + soc * (0.85 - 0.01)
        theta_p = 0.99 - soc * (0.99 - 0.40)
        
        # 限制范围避免 OCV 计算溢出
        theta_n = np.clip(theta_n, 0.001, 0.999)
        theta_p = np.clip(theta_p, 0.001, 0.999)
        
        U_n = OCV_n_corrected(theta_n)
        U_p = OCV_p_corrected(theta_p)
        
        return U_p - U_n

    def reset_state(self, soc):
        # 根据 SOC 初始化平均浓度
        # 此映射取决于电池平衡。
        # NASA B0005 大致为:
        # 0% SOC: theta_n ~ 0.01, theta_p ~ 0.99
        # 100% SOC: theta_n ~ 0.85, theta_p ~ 0.40
        
        # 简单线性映射用于初始化
        theta_n = 0.01 + soc * (0.85 - 0.01)
        theta_p = 0.99 - soc * (0.99 - 0.40)
        
        self.c_avg_n = theta_n * self.c_max_n
        self.c_avg_p = theta_p * self.c_max_p

    def simulate_cycle(self, t_array, I_array):
        # 模拟完整循环
        # 向量化实现以提高速度
        
        dt_array = np.diff(t_array, prepend=t_array[0])
        dt_array[0] = dt_array[1] if len(dt_array) > 1 else 1.0
        
        # 1. 累积电荷 (Ah 或 C)
        # 积分 I * dt
        q_cum = np.cumsum(I_array * dt_array) # Coulombs
        
        # 2. 平均浓度
        # c_n(t) = c_n(0) + (3 / (An * F * R_n)) * Q_cum
        
        delta_c_n = (3.0 / (self.Area_total_n * F * self.R_n)) * q_cum
        delta_c_p = -(3.0 / (self.Area_total_p * F * self.R_p)) * q_cum 
        
        # 假设 self.c_avg_n 是 t=0 时的初始值
        c_avg_n_vec = self.c_avg_n + delta_c_n
        c_avg_p_vec = self.c_avg_p + delta_c_p
        
        # 限制平均浓度
        c_avg_n_vec = np.clip(c_avg_n_vec, 100, self.c_max_n - 100)
        c_avg_p_vec = np.clip(c_avg_p_vec, 100, self.c_max_p - 100)
        
        # 3. 表面浓度
        # j_n = -I / (An * F)
        j_n_vec = -I_array / (self.Area_total_n * F)
        j_p_vec = I_array / (self.Area_total_p * F)
        
        c_surf_n_vec = c_avg_n_vec - (self.R_n / (5 * self.D_n)) * j_n_vec
        c_surf_p_vec = c_avg_p_vec - (self.R_p / (5 * self.D_p)) * j_p_vec
        
        # 限制表面浓度
        c_surf_n_vec = np.clip(c_surf_n_vec, 1e-3, self.c_max_n - 1e-3)
        c_surf_p_vec = np.clip(c_surf_p_vec, 1e-3, self.c_max_p - 1e-3)
        
        # 4. OCV
        theta_n_vec = c_surf_n_vec / self.c_max_n
        theta_p_vec = c_surf_p_vec / self.c_max_p
        
        U_n_vec = OCV_n_corrected(theta_n_vec)
        U_p_vec = OCV_p_corrected(theta_p_vec)
        
        # 5. 过电位
        c_e = 1000
        # i0 = k * sqrt(ce * cs * (cmax - cs))
        i0_n_vec = self.k_n * np.sqrt(c_e * c_surf_n_vec * (self.c_max_n - c_surf_n_vec))
        i0_p_vec = self.k_p * np.sqrt(c_e * c_surf_p_vec * (self.c_max_p - c_surf_p_vec))
        
        i0_n_vec = np.maximum(i0_n_vec, 1e-6)
        i0_p_vec = np.maximum(i0_p_vec, 1e-6)
        
        # eta = 2RT/F * asinh(j / 2i0)
        term_n = j_n_vec / (2 * i0_n_vec)
        term_p = j_p_vec / (2 * i0_p_vec)
        
        eta_n_vec = (2 * R * T / F) * np.arcsinh(term_n)
        eta_p_vec = (2 * R * T / F) * np.arcsinh(term_p)
        
        # 6. 端电压
        # V = (Up + etap) - (Un + etan) - I * Rf
        R_f = getattr(self, 'R_f', 0.05) 
        V_pred = (U_p_vec + eta_p_vec) - (U_n_vec + eta_n_vec) - I_array * R_f
        
        # 准备返回的微观变量
        micro_vars = {
            'c_surf_n': c_surf_n_vec,
            'c_surf_p': c_surf_p_vec,
            'eta_n': eta_n_vec,
            'eta_p': eta_p_vec,
            'theta_n': theta_n_vec,
            'theta_p': theta_p_vec
        }
            
        return V_pred, micro_vars

