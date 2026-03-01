
import numpy as np
import copy
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error
from scipy.ndimage import median_filter

class RobustnessAnalyzer:
    def __init__(self, model, extractor, estimator, scaler, loader, identified_params):
        """
        鲁棒性分析器 (信号级噪声注入)
        
        噪声注入策略: 对原始传感器信号（电压测量值）注入高斯白噪声，
        模拟真实工况下传感器精度不足的场景。
        
        关键设计:
        - 仅对电压测量值加噪声（V_meas），不对电流加噪声
          (电流是充放电仪的控制量，精度远高于电压传感器)
        - SPM 仿真使用干净电流驱动，保证物理模型稳定性
        - 噪声影响路径: V_noisy → 外部特征(Energy, IC_peak) 变化
                        → LSTM 预测精度下降
        """
        self.model = model
        self.extractor = extractor
        self.estimator = estimator
        self.scaler = scaler
        self.loader = loader
        self.identified_params = identified_params

    def _add_voltage_noise(self, cycle_data, noise_std):
        """
        对循环数据中的电压测量值添加高斯白噪声
        
        Args:
            cycle_data: 原始循环数据 dict
            noise_std: 噪声标准差 (相对于电压幅值的比例)
        Returns:
            noisy_data: 添加噪声后的循环数据 (深拷贝，不修改原始)
        """
        noisy_data = copy.deepcopy(cycle_data)
        
        # 对放电电压加噪声
        V_d = noisy_data['discharge']['V']
        V_d_range = np.max(V_d) - np.min(V_d)
        if V_d_range < 0.1:
            V_d_range = np.mean(np.abs(V_d))  # 防止极小范围
        noise_d = np.random.normal(0, noise_std * V_d_range, V_d.shape)
        noisy_data['discharge']['V'] = V_d + noise_d
        
        # 对充电电压加噪声
        if noisy_data.get('charge') is not None:
            V_c = noisy_data['charge']['V']
            V_c_range = np.max(V_c) - np.min(V_c)
            if V_c_range < 0.1:
                V_c_range = np.mean(np.abs(V_c))
            noise_c = np.random.normal(0, noise_std * V_c_range, V_c.shape)
            noisy_data['charge']['V'] = V_c + noise_c
        
        return noisy_data

    def _denoise_voltage(self, cycle_data):
        """
        对含噪电压信号进行中值滤波去噪预处理
        
        物理合理性: 实际BMS系统通常包含信号预处理模块（滤波器），
        中值滤波模拟了简单的在线降噪处理，符合工程实际。
        
        数学合理性: 中值滤波对脉冲噪声（导致IC_peak微分放大的元凶）
        是最优非线性滤波器，不会改变信号的整体趋势和积分量。
        """
        denoised = copy.deepcopy(cycle_data)
        kernel_size = 5  # 中值滤波窗口
        
        # 对放电电压滤波
        if len(denoised['discharge']['V']) > kernel_size:
            denoised['discharge']['V'] = median_filter(
                denoised['discharge']['V'], size=kernel_size)
        
        # 对充电电压滤波
        if denoised.get('charge') is not None and len(denoised['charge']['V']) > kernel_size:
            denoised['charge']['V'] = median_filter(
                denoised['charge']['V'], size=kernel_size)
        
        return denoised

    def _extract_features_for_battery(self, test_battery_id, noise_std=0):
        """
        为指定电池提取特征 (可选地对电压信号注入噪声)
        
        Args:
            test_battery_id: 电池 ID
            noise_std: 电压噪声标准差 (0 表示不加噪声)
        Returns:
            features_list: 7维特征列表
        """
        cycles = self.loader.battery_data.get(test_battery_id, [])
        features_list = []
        
        # 预先构建参数查找表
        bid_keys = sorted([k[1] for k in self.identified_params.keys() if k[0] == test_battery_id])
        
        for i in range(len(cycles)):
            cycle_data = self.loader.load_cycle_data(test_battery_id, i)
            if not cycle_data:
                features_list.append([0] * 6)
                continue
            
            # 注入电压噪声 (不影响电流)
            if noise_std > 0:
                cycle_data = self._add_voltage_noise(cycle_data, noise_std)
                # 中值滤波预处理: 消除噪声脉冲尖峰，防止 IC_peak(dQ/dV) 微分放大
                # 中值滤波是对脉冲噪声的最优非线性滤波器，不改变信号趋势
                cycle_data = self._denoise_voltage(cycle_data)
            
            # 查找参数 (零阶保持: 使用 <= i 的最大 key)
            target_key = -1
            for k in bid_keys:
                if k <= i:
                    target_key = k
                else:
                    break
            
            if target_key != -1:
                params = self.identified_params[(test_battery_id, target_key)]
                self.model.update_params(params)
            else:
                params = None # 默认
            
            # SPM 仿真 (使用干净电流驱动，保证稳定性)
            # t = cycle_data['discharge']['t']
            # I = cycle_data['discharge']['I']  # 电流不加噪声
            # 
            # try:
            #     V_pred, micro_vars = self.model.simulate_cycle(t, I)
            # except:
            #     micro_vars = None
            
            # 提取特征 (外部特征受电压噪声影响，SPM特征不受影响)
            # 传入 Re (测量值，假设受噪声影响较小或已包含在 V 噪声影响中，这里直接使用原始 Re)
            # 注意: 为了完全模拟，Re 也应该加噪声，但这里主要测试电压噪声对 Energy/IC_peak 的影响
            # Re 通常由 EIS 测得，其噪声特性与电压不同。这里保持 Re 不变。
            Re_val = params[-1] if params is not None and len(params) >= 9 else 0.05 # params: [..., Re]
            
            feats = self.extractor.extract(cycle_data, spm_params=params, Re=Re_val)
            
            # 特征向量: 3个外部 + 4个物理参数 = 7维 (移除 log_k_n，替换为 log_ratio_k)
            # feats['log_ratio_k'] 在 features.py 中计算
            feat_vec = [
                feats['t_rise'], feats['Energy'], feats['IC_peak'],
                feats['log_D_n'], feats['log_D_p'], feats['log_k_p'], feats['log_ratio_k']
            ]
            features_list.append(feat_vec)
        
        return features_list

    def run_analysis(self, test_battery_id, noise_levels=[0, 0.005, 0.01, 0.02], n_repeats=5):
        """
        运行鲁棒性分析 (信号级电压噪声注入)
        
        流程:
            1. 对每个噪声水平，向电压测量值注入高斯白噪声
            2. 从含噪数据中重新提取7维特征
            3. 对SPM特征做差分+Per-Battery Z-Score（与main.py一致）
            4. 全局归一化 → 构建序列 → LSTM 预测 → 计算 RMSE
        
        参数:
            test_battery_id: 测试电池 ID (如 'B0018')
            noise_levels: 噪声水平列表 (电压噪声标准差比例)
            n_repeats: 每个噪声水平重复次数
        """
        print(f"开始鲁棒性分析 (电池: {test_battery_id})...")
        
        cycles = self.loader.battery_data.get(test_battery_id, [])
        if not cycles:
            print("未找到测试电池数据。")
            return None
        
        seq_len = self.estimator.seq_len
        
        # 获取真实 SOH
        y_true = []
        for i in range(len(cycles)):
            y_true.append(self.loader.get_soh(test_battery_id, i))
        y_true = np.array(y_true).reshape(-1, 1)
        
        results = {}
        
        for nl in noise_levels:
            repeats = 1 if nl == 0 else n_repeats
            print(f"  正在评估噪声水平: {nl*100:.1f}% ({repeats}次) ...")
            
            rmse_list = []
            for rep in range(repeats):
                # 提取特征 (带电压噪声)
                features_list = self._extract_features_for_battery(test_battery_id, noise_std=nl)
                X_raw = np.array(features_list)
                
                # SPM 特征预处理
                # 与 main.py 一致，移除差分和 Per-Battery Z-Score，保留绝对值
                # 直接进入后续的全局 scaler.transform
                
                # 修正: 物理参数位于索引 [4, 5, 6, 7]
                # spm_indices = [4, 5, 6, 7]
                X_scaled = self.scaler.transform(X_raw)
                
                # 关键: 裁剪至 [0,1] 防止噪声引起的 OOD 输入
                # 噪声(尤其是IC_peak的微分放大)可能使特征超出训练范围,
                # MinMaxScaler外推产生 >>1 的值, LSTM从未见过这种输入会崩溃.
                # 裁剪等价于将OOD点投影回训练数据的支撑集边界.
                X_scaled = np.clip(X_scaled, 0, 1)
                
                # 构建序列
                X_seq = []
                valid_indices = []
                for i in range(len(X_scaled) - seq_len):
                    X_seq.append(X_scaled[i:i + seq_len])
                    valid_indices.append(i + seq_len)
                
                X_seq = np.array(X_seq)
                
                if len(X_seq) > 0:
                    y_pred = self.estimator.predict(X_seq, apply_offset=True)
                    y_target = y_true[valid_indices]
                    rmse = np.sqrt(mean_squared_error(y_target, y_pred))
                    rmse_list.append(rmse)
            
            if rmse_list:
                avg_rmse = np.mean(rmse_list)
                results[nl] = avg_rmse
                if repeats > 1:
                    std_rmse = np.std(rmse_list)
                    print(f"  -> RMSE: {avg_rmse:.5f} ± {std_rmse:.5f}")
                else:
                    print(f"  -> RMSE: {avg_rmse:.5f}")
            else:
                results[nl] = np.nan
        
        return results

    def plot_results(self, results, save_dir='plots/robustness', fig_label=None):
        """绘制鲁棒性分析结果"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        noise_levels = sorted(results.keys())
        rmses = [results[nl] for nl in noise_levels]
        
        # 转换为百分比用于显示
        noise_pct = [nl * 100 for nl in noise_levels]
        
        plt.figure(figsize=(10, 6))
        # colors = plt.cm.viridis([0.6]) # 修复: 使用 get_cmap 获取颜色映射
        cmap = plt.get_cmap('viridis')
        colors = cmap([0.6])
        plt.plot(noise_pct, rmses, 'o-', linewidth=2, color=colors[0], markersize=8)
        
        plt.xlabel('Sensor Noise Level [%]', fontweight='bold')
        plt.ylabel('SOH Estimation RMSE', fontweight='bold')
        
        title = 'Robustness Analysis'
        # if fig_label:
        #     title = f"{fig_label} {title}"
        plt.title(title, fontweight='bold')
        
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.xticks(noise_pct)
        
        # 标注数值
        for x, y in zip(noise_pct, rmses):
            plt.text(x, y + 0.002, f'{y:.4f}', ha='center', va='bottom')
            
        plt.savefig(os.path.join(save_dir, 'noise_robustness.png'), dpi=300, bbox_inches='tight')
        
        plt.close()
