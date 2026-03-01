
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from SALib.sample import saltelli
from SALib.analyze import sobol
from src.spm_model import DualElectrodeSPM
import os

class SensitivityAnalyzer:
    def __init__(self, model):
        self.model = model
        # 定义参数问题 (SALib格式)
        # 参数顺序: D_p, D_n, k_p, k_n, R_p, R_n, c_max_p, c_max_n
        # 注：R_f（欧姆内阻）可通过EIS直接测量，不作为待辨识参数，因此不纳入灵敏度分析
        self.problem = {
            'num_vars': 8,
            'names': ['D_p', 'D_n', 'k_p', 'k_n', 'R_p', 'R_n', 'c_max_p', 'c_max_n'],
            'bounds': [
                [1e-15, 1e-12], # D_p
                [1e-15, 1e-12], # D_n
                [1e-13, 1e-9],  # k_p
                [1e-13, 1e-9],  # k_n
                [1e-6, 20e-6],  # R_p
                [1e-6, 20e-6],  # R_n
                [30000, 60000], # c_max_p
                [20000, 50000]  # c_max_n
            ]
        }
        
    def run_analysis(self, t, I, V_ref, sample_size=128):
        """
        运行Sobol灵敏度分析
        目标函数: RMSE (模拟电压与参考电压的均方根误差)
        """
        print(f"开始灵敏度分析 (样本数: {sample_size})...")
        
        # 1. 生成样本
        # Saltelli采样器生成 N * (2D + 2) 个样本
        param_values = saltelli.sample(self.problem, sample_size)
        
        Y = np.zeros([param_values.shape[0]])
        
        # 2. 运行模型
        for i, X in enumerate(param_values):
            # 更新参数
            self.model.update_params(X)
            
            # 必须重置状态以确保公平比较
            # 假设从 100% SOC 开始 (或根据 V_ref[0] 估算)
            # 这里简单起见，假设满充开始放电
            self.model.reset_state(1.0) 
            
            try:
                V_sim, _ = self.model.simulate_cycle(t, I)
                
                # 计算 RMSE
                rmse = np.sqrt(np.mean((V_sim - V_ref)**2))
                Y[i] = rmse
            except:
                Y[i] = 1e6 # 异常惩罚
            
            if i % 100 == 0:
                print(f"已评估样本 {i}/{len(param_values)}")
                
        # 3. 分析结果
        Si = sobol.analyze(self.problem, Y, print_to_console=False)
        
        return Si

    def plot_sensitivity(self, Si, save_dir='plots/sensitivity', fig_label=None):
        """绘制灵敏度分析结果"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # 提取一阶指数 (S1) 和 总效应指数 (ST)
        names = self.problem['names']
        
        # 中文标签映射
        labels_map = {
            'D_p': 'Cathode Diffusion ($D_p$)',
            'D_n': 'Anode Diffusion ($D_n$)',
            'k_p': 'Cathode Reaction Rate ($k_p$)',
            'k_n': 'Anode Reaction Rate ($k_n$)',
            'R_p': 'Cathode Particle Radius ($R_p$)',
            'R_n': 'Anode Particle Radius ($R_n$)',
            'c_max_p': 'Cathode Max Conc. ($c_{max,p}$)',
            'c_max_n': 'Anode Max Conc. ($c_{max,n}$)'
        }
        labels = [labels_map[n] for n in names]
        
        x = np.arange(len(labels))
        width = 0.35
        
        # 将负值截断为0（负值是样本量不足导致的数值噪声）
        S1_vals = np.clip(Si['S1'], 0, None)
        ST_vals = np.clip(Si['ST'], 0, None)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 1. 绘制柱状图
        x = np.arange(len(labels))
        width = 0.35
        
        # 准备数据 (clip 负值)
        s1_plot = [max(0, val) for val in Si['S1']]
        st_plot = [max(0, val) for val in Si['ST']]
        
        rects1 = ax.bar(x - width/2, s1_plot, width, label='First Order (S1)', color='tab:blue', alpha=0.8)
        rects2 = ax.bar(x + width/2, st_plot, width, label='Total Effect (ST)', color='tab:orange', alpha=0.8)
        
        # 2. 添加数值标签
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                if height > 0.01:  # 只显示大于 0.01 的值
                    ax.annotate(f'{height:.3f}',
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom', fontsize=9)

        autolabel(rects1)
        autolabel(rects2)

        # 3. 设置轴标签和标题
        ax.set_ylabel('Sensitivity Index')
        ax.set_title('Parameter Sensitivity Analysis (Sobol Method)')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend()
        
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        fig.tight_layout()
        
        # 4. 保存图片
        fname = 'sensitivity_analysis.png'
            
        try:
            plt.savefig(os.path.join(save_dir, fname), dpi=300, bbox_inches='tight')
        except Exception as e:
            print(f"Warning: Failed to save with bbox_inches='tight', retrying without it. Error: {e}")
            plt.savefig(os.path.join(save_dir, fname), dpi=300)
            
        plt.close()
        
        # 保存数据
        df = pd.DataFrame({
            'Parameter': labels,
            'S1': Si['S1'],
            'S1_conf': Si['S1_conf'],
            'ST': Si['ST'],
            'ST_conf': Si['ST_conf']
        })
        df.to_csv(os.path.join(save_dir, 'sensitivity_results.csv'), index=False)
        print("灵敏度分析完成，结果已保存。")
