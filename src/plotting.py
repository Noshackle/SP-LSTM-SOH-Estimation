
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import matplotlib
import warnings

# 忽略字体警告
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def set_style():
    """设置学术海报级绘图风格"""
    # 使用 seaborn-whitegrid 样式（新版本兼容）
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        plt.style.use('seaborn-whitegrid')
    
    # 学术海报配置
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'SimHei', 'Microsoft YaHei', 'DejaVu Sans'],
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'axes.labelweight': 'bold',  # 坐标轴标签加粗
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 13,
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': 'gray',
        'figure.figsize': (10, 6),
        'lines.linewidth': 2,
        'lines.markersize': 8,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
    })

def plot_full_cycle_validation(validation_data, save_dir='plots/validation', fig_label=None):
    """
    绘制多电池全周期参数辨识验证图 - 仅RMSE趋势图
    
    参数:
        validation_data: dict, 格式为 {battery_id: {'cycles': [...], 'rmse': [...]}}
        save_dir: 保存目录
        fig_label: 图标签
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 准备数据
    batteries = list(validation_data.keys())
    n_batteries = len(batteries)
    
    if n_batteries == 0:
        return
    
    # 使用 viridis 色系
    n_batt = len(batteries)
    colors_list = plt.get_cmap('viridis')(np.linspace(0, 0.9, n_batt))
    color_map = {bid: colors_list[i] for i, bid in enumerate(batteries)}
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
    marker_map = {bid: markers[i % len(markers)] for i, bid in enumerate(batteries)}
    
    # 各电池各循环的RMSE趋势
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for bid in batteries:
        data = validation_data[bid]
        cycles = data['cycles']
        rmse_vals = data['rmse']
        color = color_map[bid]
        marker = marker_map[bid]
        ax.plot(cycles, rmse_vals, marker=marker, markersize=6, linewidth=2, 
                color=color, label=f'{bid}', alpha=0.85)
    
    ax.set_xlabel('Cycle Number', fontweight='bold')
    ax.set_ylabel('RMSE [V]', fontweight='bold')
    title = 'Parameter Identification Validation - Voltage RMSE per Cycle'
    if fig_label:
        title = f'{fig_label} {title}'
    ax.set_title(title, fontweight='bold')
    ax.legend(loc='upper right', frameon=True, framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_ylim(bottom=0)
    
    # 添加平均RMSE注释
    for bid in batteries:
        data = validation_data[bid]
        avg_rmse = np.mean(data['rmse'])
        ax.axhline(y=avg_rmse, color=color_map[bid], linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    
    fname = 'full_cycle_validation_rmse.png'
    
    plt.savefig(os.path.join(save_dir, fname), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 打印统计信息
    print("\n=== 参数辨识验证统计 ===")
    for bid in batteries:
        data = validation_data[bid]
        rmse_vals = np.array(data['rmse'])
        print(f"  {bid}: 平均RMSE={np.mean(rmse_vals):.4f}V, "
              f"最大RMSE={np.max(rmse_vals):.4f}V, 最小RMSE={np.min(rmse_vals):.4f}V")


def plot_battery_voltage_fitting(validation_data, save_dir='plots/validation', fig_label_prefix='Fig 4'):
    """
    为每个电池绘制全周期电压拟合图（从第0次到最终循环）
    
    参数:
        validation_data: dict, 格式为 {battery_id: {'cycles': [...], 'rmse': [...], 'voltage_samples': {...}}}
        save_dir: 保存目录
        fig_label_prefix: 图标签前缀
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    batteries = list(validation_data.keys())
    
    for idx, bid in enumerate(batteries):
        data = validation_data[bid]
        voltage_samples = data.get('voltage_samples', {})
        
        if not voltage_samples:
            continue
        
        # 获取所有采样的循环，按循环号排序
        sample_cycles = sorted(voltage_samples.keys())
        n_cycles = len(sample_cycles)
        
        if n_cycles == 0:
            continue
        
        # 计算子图布局：每行最多4个子图
        n_cols = min(4, n_cycles)
        n_rows = (n_cycles + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3.5*n_rows))
        
        # 确保 axes 是二维数组
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i, cyc in enumerate(sample_cycles):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            
            vdata = voltage_samples[cyc]
            t = vdata['t']
            V_meas = vdata['V_meas']
            V_sim = vdata['V_sim']
            rmse = vdata['rmse']
            
            # 使用 viridis 色系
            colors = plt.get_cmap('viridis')([0.2, 0.8])
            
            # 下采样标记点以避免过密
            markevery = max(1, len(t)//15)
            
            ax.plot(t, V_meas, 'o-', color=colors[0], linewidth=2, markersize=3, 
                   markevery=markevery, label='Measured', alpha=0.85)
            ax.plot(t, V_sim, 's--', color=colors[1], linewidth=2, markersize=3, 
                   markevery=markevery, label='Simulated', alpha=0.85)
            ax.set_xlabel('Time [s]', fontsize=10, fontweight='bold')
            ax.set_ylabel('Voltage [V]', fontsize=10, fontweight='bold')
            ax.set_title(f'Cycle {cyc} (RMSE={rmse:.3f}V)', fontsize=11, fontweight='bold')
            ax.legend(loc='upper right', fontsize=8, frameon=True, framealpha=0.9)
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.tick_params(labelsize=9)
        
        # 隐藏多余的子图
        for i in range(n_cycles, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)
        
        # 计算该电池的平均RMSE
        avg_rmse = np.mean(data['rmse'])
        
        fig.suptitle(f'{bid} Full Cycle Voltage Fitting (Avg RMSE={avg_rmse:.4f}V)', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # 保存文件
        fname = f"full_cycle_voltage_fitting_{bid}.png"
        
        plt.savefig(os.path.join(save_dir, fname), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  已生成 {bid} 全周期电压拟合图: {fname}")


def plot_feature_correlation(X, feature_names, save_dir='plots/features', fig_label=None):
    """绘制特征相关性矩阵热力图"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    plt.figure(figsize=(12, 10))
    corr = np.corrcoef(X.T)
    
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5},
                annot=True, fmt=".2f", xticklabels=feature_names, yticklabels=feature_names)
    
    title = 'Feature Correlation Matrix'
    if fig_label:
        title = f"{fig_label} {title}"
    plt.title(title)
    
    plt.tight_layout()
    
    fname = 'feature_correlation.png'
        
    plt.savefig(os.path.join(save_dir, fname), dpi=300, bbox_inches='tight')
    plt.close()

def plot_feature_evolution(X, soh, feature_names, save_dir='plots/features', fig_label=None):
    """绘制特征随SOH演变图"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    n_feats = X.shape[1]
    n_names = len(feature_names)
    
    # 动态计算布局
    cols = 4
    rows = (n_feats + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    axes = axes.flatten()
    
    for i in range(n_feats):
        ax = axes[i]
        cycles = range(len(X))
        sc = ax.scatter(soh, X[:, i], c=cycles, cmap='viridis', s=20, alpha=0.7)
        ax.set_xlabel('SOH')
        
        # 安全获取特征名称
        fname = feature_names[i] if i < n_names else f'Feature {i}'
        ax.set_ylabel(fname)
        ax.set_title(f'{fname} vs SOH')
        
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.invert_xaxis()
    
    # 隐藏多余的子图
    for i in range(n_feats, rows * cols):
        axes[i].axis('off')
    
    # 添加颜色条表示循环数
    cbar_ax = fig.add_axes((0.92, 0.15, 0.02, 0.7))
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, len(X)))  # type: ignore
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Cycle Number')
    
    title = 'Feature Evolution vs SOH'
    if fig_label:
        title = f"{fig_label} {title}"
    plt.suptitle(title, y=0.98, fontsize=14)
    
    # 使用 constrained_layout 替代 tight_layout 避免警告
    try:
        plt.tight_layout(rect=(0, 0, 0.9, 0.95))
    except:
        pass  # 忽略布局警告
    
    fname = 'feature_evolution.png'
        
    plt.savefig(os.path.join(save_dir, fname), dpi=300, bbox_inches='tight')
    plt.close()

def plot_soh_results(y_true, y_pred, train_len, save_dir='plots/results', fig_label=None):
    """绘制SOH估计结果"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # 1. SOH 曲线
    plt.figure(figsize=(10, 6))
    colors = plt.get_cmap('viridis')([0.2, 0.8])
    plt.plot(range(len(y_true)), y_true, 'o-', color=colors[0], linewidth=2, 
             markersize=5, markevery=max(1, len(y_true)//30), label='Reference SOH')
    
    if train_len > 0:
        plt.axvline(x=train_len, color='gray', linestyle='--', linewidth=1.5, label='Train/Test Split')
    
    if len(y_pred) == len(y_true):
        plt.plot(range(len(y_pred)), y_pred, 's--', color=colors[1], linewidth=2, 
                 markersize=4, markevery=max(1, len(y_pred)//30), label='Estimated SOH')
    
    plt.xlabel('Cycle Number', fontweight='bold')
    plt.ylabel('State of Health (SOH)', fontweight='bold')
    
    title = 'SOH Estimation Results'
    if fig_label:
        title = f"{fig_label} {title}"
    plt.title(title, fontweight='bold')
    
    plt.legend(loc='upper right', frameon=True, framealpha=0.9)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'soh_estimation_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 误差分析
    if len(y_pred) == len(y_true):
        # 仅计算测试集的误差
        y_true_test = y_true[train_len:]
        y_pred_test = y_pred[train_len:]
        
        error = y_true_test - y_pred_test
        rmse = np.sqrt(np.mean(error**2))
        mae = np.mean(np.abs(error))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 误差直方图
        sns.histplot(error, kde=True, ax=ax1, color='tab:blue')
        title1 = f'Test Error Dist.\nRMSE={rmse:.4f}, MAE={mae:.4f}'
        if fig_label:
             title1 = f"{fig_label} (Error Analysis) {title1}"
             
        ax1.set_title(title1)
        ax1.set_xlabel('Error [SOH]')
        
        # 散点图
        ax2.scatter(y_true_test, y_pred_test, alpha=0.5, c='tab:green')
        ax2.plot([0, 1], [0, 1], 'k--')
        ax2.set_xlabel('True SOH')
        ax2.set_ylabel('Estimated SOH')
        ax2.set_title('Parity Plot')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'error_analysis.png'))
        plt.close()

def plot_parameter_trends(identified_params, save_dir='plots/params', fig_labels=None):
    """
    绘制辨识参数的演变趋势
    fig_labels: list of 2 strings for the 2 plots, e.g. ['图 4-5', '图 4-6']
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    cycles = sorted(identified_params.keys())
    if not cycles: return
    
    # 参数名称更新为 4 个核心动力学参数 + 1 个可选 R_f
    param_names = ['D_p', 'D_n', 'k_p', 'k_n', 'R_f']
    
    # 检查 key 类型
    sample_key = cycles[0]
    if isinstance(sample_key, tuple):
        # 过滤 B0005
        target_bid = 'B0005'
        cycles = [c for b, c in cycles if b == target_bid]
        cycles.sort()
        params_array = np.array([identified_params[(target_bid, c)] for c in cycles])
    else:
        params_array = np.array([identified_params[c] for c in cycles])
    
    # 归一化
    params_norm = np.zeros_like(params_array)
    for i in range(params_array.shape[1]):
        p_min = params_array[:, i].min()
        p_max = params_array[:, i].max()
        if p_max - p_min > 1e-20:
            params_norm[:, i] = (params_array[:, i] - p_min) / (p_max - p_min)
        else:
            params_norm[:, i] = 0.5
    
    # 默认标签
    # if not fig_labels or len(fig_labels) < 2:
    #     fig_labels = ['Fig 4-5', 'Fig 4-6']
    
    # 1. 扩散系数 (D_p, D_n)
    plt.figure(figsize=(10, 6))
    colors = plt.get_cmap('viridis')([0.3, 0.7])
    # D_p, D_n 变化范围约 1e-16 到 1e-12，设置 y 轴范围以显示清晰
    plt.semilogy(cycles, params_array[:, 0], 'o-', color=colors[0], linewidth=2, markersize=6, label=r'Cathode Diffusion $D_p$')
    plt.semilogy(cycles, params_array[:, 1], 's-', color=colors[1], linewidth=2, markersize=6, label=r'Anode Diffusion $D_n$')
    plt.xlabel('Cycle', fontweight='bold')
    plt.ylabel(r'Diffusion Coefficient [$m^2/s$]', fontweight='bold')
    title = 'Diffusion Coeff. Evolution'
    # if fig_labels[0]: title = f"{fig_labels[0]} {title}"
    plt.title(title, fontweight='bold')
    plt.legend(loc='upper right', frameon=True, framealpha=0.9)
    plt.grid(True, which='both', linestyle='--', alpha=0.3)
    # 根据实际数据调整 Y 轴范围，确保不超出边界
    y_min = min(params_array[:, 0].min(), params_array[:, 1].min()) * 0.1
    y_max = max(params_array[:, 0].max(), params_array[:, 1].max()) * 10
    plt.ylim(max(y_min, 1e-17), min(y_max, 1e-11))
    
    fname1 = 'diffusion_coeff_evolution.png'
    plt.savefig(os.path.join(save_dir, fname1), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 反应速率 (k_p, k_n)
    plt.figure(figsize=(10, 6))
    plt.semilogy(cycles, params_array[:, 2], 'o-', color=colors[0], linewidth=2, markersize=6, label=r'Cathode Reaction Rate $k_p$')
    plt.semilogy(cycles, params_array[:, 3], 's-', color=colors[1], linewidth=2, markersize=6, label=r'Anode Reaction Rate $k_n$')
    plt.xlabel('Cycle', fontweight='bold')
    plt.ylabel(r'Reaction Rate Constant [$m^{2.5}/(mol^{0.5} \cdot s)$]', fontweight='bold')
    title = 'Reaction Rate Evolution'
    # if fig_labels[1]: title = f"{fig_labels[1]} {title}"
    plt.title(title, fontweight='bold')
    plt.legend(loc='center right', frameon=True, framealpha=0.9)
    plt.grid(True, which='both', linestyle='--', alpha=0.3)
    # 调整 Y 轴范围
    y_min_k = min(params_array[:, 2].min(), params_array[:, 3].min()) * 0.1
    y_max_k = max(params_array[:, 2].max(), params_array[:, 3].max()) * 10
    plt.ylim(max(y_min_k, 1e-15), min(y_max_k, 1e-7))
    
    fname2 = 'reaction_rate_evolution.png'
    plt.savefig(os.path.join(save_dir, fname2), dpi=300, bbox_inches='tight')
    plt.close()


def plot_soh_param_correlation(soh_list, params_list, param_names, save_dir='plots/params'):
    """绘制 SOH 与辨识参数的相关性散点图，包括衍生特征"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    n_params = len(param_names)
    
    # 计算衍生特征: D_n/D_p 和 k_p/k_n
    # 假设 params_list 列顺序: D_p, D_n, k_p, k_n
    # 注意避免除零错误
    D_p = params_list[:, 0]
    D_n = params_list[:, 1]
    k_p = params_list[:, 2]
    k_n = params_list[:, 3]
    
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio_D = D_n / D_p
        ratio_k = k_p / k_n # 注意有些文献用 k_n/k_p，这里按用户要求 k_p/k_n
        
    # 将衍生特征拼接到列表末尾
    params_aug = np.column_stack([params_list, ratio_D, ratio_k])
    names_aug = param_names + ['Ratio_Dn_Dp', 'Ratio_kp_kn']
    
    n_total = params_aug.shape[1]
    cols = 3
    rows = (n_total + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()
    
    from scipy.stats import pearsonr
    
    for i in range(n_total):
        ax = axes[i]
        p_vals = params_aug[:, i].flatten()
        soh_vals = soh_list.flatten()
        
        # 确保长度一致且有效
        valid_mask = ~np.isnan(p_vals) & ~np.isnan(soh_vals) & ~np.isinf(p_vals)
        p_vals = p_vals[valid_mask]
        soh_vals = soh_vals[valid_mask]
        
        if len(p_vals) < 2: continue

        # 处理对数坐标或原始值
        name = names_aug[i]
        # 物理参数和衍生参数 (Ratio) 都应该取对数
        is_log = name.startswith('D') or name.startswith('k') or name.startswith('Ratio')
        
        if is_log:
            # 过滤掉非正值
            valid_log = p_vals > 0
            if np.sum(valid_log) > 1:
                y_vals = np.log10(p_vals[valid_log])
                x_vals = soh_vals[valid_log]
                ylabel = f'log10({name})'
            else:
                continue
        else:
            y_vals = p_vals
            x_vals = soh_vals
            ylabel = name
            
        # 绘制散点
        ax.scatter(x_vals, y_vals, alpha=0.6, color='tab:blue', label='Data')
        
        # 添加趋势线
        if len(x_vals) > 1:
            z = np.polyfit(x_vals, y_vals, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min(x_vals), max(x_vals), 100)
            ax.plot(x_trend, p(x_trend), "r--", linewidth=2, label='Trend')
            
            # 计算相关系数
            corr, p_val = pearsonr(x_vals, y_vals)
            
            # 确保 p_val 是标量
            if hasattr(p_val, 'item'): p_val = p_val.item()
            elif isinstance(p_val, (list, tuple)) and len(p_val) > 0: p_val = p_val[0]
            p_val = float(p_val)
            
            sig_text = " (Satisfied)" if p_val < 0.05 else ""
            if p_val < 0.001:
                title_str = f'{name} vs SOH\nr={corr:.3f}, p={p_val:.2e}{sig_text}'
            else:
                title_str = f'{name} vs SOH\nr={corr:.3f}, p={p_val:.3f}{sig_text}'
        else:
            title_str = f'{name} vs SOH'
        
        ax.set_xlabel('SOH')
        ax.set_ylabel(ylabel)
        ax.set_title(title_str, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.invert_xaxis() # SOH 从 1 到 0 演变
        # ax.legend() # 图太小，legend可能遮挡，暂不显示
        
    for i in range(n_total, len(axes)):
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'soh_parameter_correlation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("已生成 SOH 与参数相关性分析图 (含衍生特征)。")

def plot_Re_from_metadata(metadata_path, battery_ids=['B0005', 'B0006', 'B0007', 'B0018'],
                          save_dir='plots/params', fig_label='Fig 4-8', identified_params=None):
    """
    直接从 metadata.csv 读取所有阻抗测量的 Re 值并绘图
    x 轴使用循环编号（根据阻抗测量前的放电次数推算）
    
    参数:
        metadata_path: metadata.csv 文件路径
        battery_ids: 要绘制的电池 ID 列表
        save_dir: 保存目录
        fig_label: 图表标签
        identified_params: 辨识的参数字典，格式为 {(battery_id, cycle_idx): params}
    """
    import pandas as pd
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 读取 metadata
    df = pd.read_csv(metadata_path)
    
    plt.figure(figsize=(12, 7))
    
    # 使用 viridis 色系
    n_batt = len(battery_ids)
    colors_list = plt.get_cmap('viridis')(np.linspace(0, 0.9, n_batt))
    color_map = {bid: colors_list[i] for i, bid in enumerate(battery_ids)}
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
    marker_map = {bid: markers[i % len(markers)] for i, bid in enumerate(battery_ids)}
    
    for bid in battery_ids:
        # 获取该电池的所有记录，按 test_id 排序
        bid_all = df[df['battery_id'] == bid].copy().sort_values('test_id')
        if bid_all.empty:
            print(f"  警告: {bid} 无数据")
            continue
        
        # 遍历所有记录，统计放电次数以推算循环编号
        discharge_count = 0
        cycle_list = []
        re_list = []
        
        for _, row in bid_all.iterrows():
            if row['type'] == 'discharge':
                discharge_count += 1
            elif row['type'] == 'impedance' and pd.notna(row.get('Re')):
                cycle_list.append(discharge_count)
                re_list.append(float(row['Re']) * 1000)  # Ohm -> mΩ
        
        if not re_list:
            print(f"  警告: {bid} 无阻抗测量数据")
            continue
        
        cycle_arr = np.array(cycle_list)
        re_arr = np.array(re_list)
        
        color = color_map[bid]
        marker = marker_map[bid]
        
        plt.plot(cycle_arr, re_arr, marker=marker, linestyle='-', 
                color=color, label=bid, linewidth=2, markersize=5, alpha=0.85)
        
        print(f"  {bid}: {len(re_arr)} 个阻抗测量点, 循环范围 {cycle_arr.min()}-{cycle_arr.max()}, "
              f"Re 范围 {re_arr.min():.1f}-{re_arr.max():.1f} mΩ")
    
    # 添加 B0018 的辨识电阻数据
    if identified_params and 'B0018' in battery_ids:
        # 提取 B0018 的辨识参数
        b0018_params = {k[1]: v for k, v in identified_params.items() if k[0] == 'B0018'}
        if b0018_params:
            # 按循环号排序
            sorted_cycles = sorted(b0018_params.keys())
            cycle_list = []
            re_identified_list = []
            
            for cycle in sorted_cycles:
                params = b0018_params[cycle]
                if len(params) >= 9:
                    # 第9个参数是 R_f (辨识的电阻)
                    re_identified = params[8] * 1000  # Ohm -> mΩ
                    cycle_list.append(cycle)
                    re_identified_list.append(re_identified)
            
            if re_identified_list:
                cycle_arr = np.array(cycle_list)
                re_arr = np.array(re_identified_list)
                
                # 使用与 B0018 相同的颜色，但使用虚线和不同的标记
                color = color_map.get('B0018', 'yellow')
                plt.plot(cycle_arr, re_arr, marker='x', linestyle='--', 
                        color=color, label='B0018 (Identified)', linewidth=2, markersize=5, alpha=0.85)
                print(f"  B0018 (Identified): {len(re_arr)} 个辨识点, 循环范围 {cycle_arr.min()}-{cycle_arr.max()}, "
                      f"Re 范围 {re_arr.min():.1f}-{re_arr.max():.1f} mΩ")
    
    plt.xlabel('Cycle Number', fontweight='bold')
    plt.ylabel(r'Ohmic Resistance $R_e$ [m$\Omega$]', fontweight='bold')
    title = 'Ohmic Resistance Evolution (EIS Measured vs Identified)'
    # if fig_label:
    #     title = f"{fig_label} {title}"
    plt.title(title, fontweight='bold')
    plt.legend(loc='upper right', frameon=True, framealpha=0.9)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    fname = 'param_ohmic_resistance.png'
    plt.savefig(os.path.join(save_dir, fname), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已生成多电池欧姆内阻演变图: {fname}")