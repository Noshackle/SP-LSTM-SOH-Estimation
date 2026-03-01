
import numpy as np
from scipy.signal import savgol_filter

class FeatureExtractor:
    def __init__(self):
        pass
        
    def extract_external_only(self, cycle_data):
        """
        仅提取3个外部特征 - 用于SVR、Random Forest、Base LSTM
        
        返回：包含3个外部特征的字典
        - Energy: 放电能量
        - t_rise: 恒流充电时间
        - IC_peak: 容量增量峰值
        """
        feats = {}
        
        d_data = cycle_data['discharge']
        c_data = cycle_data['charge']
        
        # 1. 能量 (放电阶段)
        t_d = d_data['t']
        V_d = d_data['V']
        I_d = d_data['I']
        dt_d = np.diff(t_d, prepend=t_d[0])
        E = np.sum(V_d * I_d * dt_d) / 3600.0  # Wh
        feats['Energy'] = abs(E)
        
        # 2. 恒流充电时间
        if c_data is not None:
            t_c = c_data['t']
            mask_cc = c_data['I'] > 1.0  # CC阶段阈值
            if np.any(mask_cc):
                t_rise = np.sum(np.diff(t_c, prepend=t_c[0])[mask_cc])
            else:
                t_rise = 0.0
            feats['t_rise'] = t_rise
            
            # 3. 容量增量峰值
            feats['IC_peak'] = self._compute_ic_peak(c_data)
        else:
            feats['t_rise'] = 0.0
            feats['IC_peak'] = 0.0
            
        return feats
    
    def extract_full(self, cycle_data, spm_params=None, Re=0.0):
        """
        提取完整的特征 - 用于SP-LSTM
        
        特征清单 (7维):
        1-3. 外部特征: t_rise, Energy, IC_peak
        4-7. SPM物理参数: log_D_n, log_D_p, log_k_p, log_ratio_k (kp/kn)
        
        注意：
        1. 已移除 Re (欧姆内阻)，因为它在参数辨识中未被重点关注，且与 SOH 的相关性不如 D/k 强。
        2. 已移除 log_k_n，因为其 P 值不显著 (>0.05)。
        3. 新增 log_ratio_k (kp/kn)，因其 P 值显著 (<0.05) 且具有物理意义。
        """
        # 先提取外部特征
        feats = self.extract_external_only(cycle_data)
        
        # 移除 Re 特征
        # feats['Re'] = Re
        
        # 添加 SPM 辨识出的核心动力学参数 (D_n, D_p, k_p, k_n)
        # 统一处理 spm_params 的长度差异
        if spm_params is not None:
            # 辨识顺序通常为 [D_p, D_n, k_p, k_n, (R_f)]
            D_p = max(spm_params[0], 1e-20)
            D_n = max(spm_params[1], 1e-20)
            k_p = max(spm_params[2], 1e-20)
            k_n = max(spm_params[3], 1e-20)
            
            feats['log_D_p'] = np.log10(D_p)
            feats['log_D_n'] = np.log10(D_n)
            feats['log_k_p'] = np.log10(k_p)
            # feats['log_k_n'] = np.log10(k_n) # 移除 log_k_n
            
            # 计算衍生特征 kp/kn 的对数
            # log(kp/kn) = log(kp) - log(kn)
            feats['log_ratio_k'] = np.log10(k_p / k_n)
        else:
            # 默认值
            feats['log_D_p'] = -14.0
            feats['log_D_n'] = -14.0
            feats['log_k_p'] = -11.0
            # feats['log_k_n'] = -11.0
            feats['log_ratio_k'] = 0.0
            
        return feats
    
    def _compute_ic_peak(self, c_data):
        """
        改进的容量增量峰值 (IC Peak) 计算
        - 只在充电阶段计算（电压上升）
        - 更稳健的平滑和异常值处理
        """
        try:
            V_c = np.asarray(c_data['V'])
            t_c = np.asarray(c_data['t'])
            I_c = np.asarray(c_data['I'])
            
            # 数据验证
            if len(V_c) < 20 or len(t_c) < 20 or len(I_c) < 20:
                return 0.5  # 默认值
            
            # 计算累积容量
            dt_c = np.diff(t_c, prepend=t_c[0])
            Q_c = np.cumsum(I_c * dt_c) / 3600.0  # Ah
            
            # 只在充电阶段（电压上升且电流足够）计算
            dV_c = np.diff(V_c, prepend=V_c[0])
            charging_mask = np.logical_and(dV_c > 0, I_c > 0.5)
            
            if np.sum(charging_mask) > 15:
                V_charge = V_c[charging_mask]
                Q_charge = Q_c[charging_mask]
                # 平滑处理
                window = min(21, len(V_charge))
                if window % 2 == 0:
                    window -= 1
                if window >= 5:
                    V_smooth = savgol_filter(V_charge, window, 3)
                    Q_smooth = savgol_filter(Q_charge, window, 3)
                    
                    # 计算 IC = dQ/dV
                    dQ = np.gradient(np.asarray(Q_smooth))
                    dV = np.gradient(np.asarray(V_smooth))
                    # 避免除零：将接近零的值替换为小常数
                    dV = np.where(np.abs(dV) < 1e-6, 1e-6, dV)
                    IC = dQ / dV
                    
                    # 在合理范围内找峰值 [3.7V, 4.15V]
                    mask1 = np.asarray(V_smooth >= 3.7)
                    mask2 = np.asarray(V_smooth <= 4.15)
                    mask3 = np.asarray(IC > 0)
                    mask4 = np.asarray(IC < 15)
                    mask_range = mask1 & mask2 & mask3 & mask4
                    if np.any(mask_range):
                        return float(np.max(IC[mask_range]))
            
            return 0.5  # 默认值
        except Exception as e:
            print(f"IC Peak计算警告: {e}")
            return 0.5  # 异常时返回默认值
    
    def extract(self, cycle_data, spm_params=None, Re=0.0):
        """
        兼容旧接口
        """
        return self.extract_full(cycle_data, spm_params=spm_params, Re=Re)
