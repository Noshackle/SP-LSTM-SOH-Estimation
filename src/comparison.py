
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor  # 新增 XGBoost
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.soh_estimator import SOHEstimator
import torch
import torch.nn as nn

# 新增 Dual-Stream GRN 模型
class GRN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.2):
        super(GRN, self).__init__()
        
        # 确定特征分割
        self.n_surface = 3
        self.n_sp = input_size - self.n_surface
        
        # ---- Stream 1: Surface GRN (快变量) ----
        # 仅输入表象特征
        self.gru = nn.GRU(self.n_surface, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc_base = nn.Linear(hidden_size, 1)
        
        # ---- Stream 2: Physics MLP (慢变量) ----
        if self.n_sp > 0:
            # 1. 特征门控 (仅对物理特征)
            self.sp_gate = nn.Parameter(torch.ones(self.n_sp))
            
            # 2. 物理特征处理 MLP
            self.sp_fc = nn.Sequential(
                nn.Linear(self.n_sp, 16),
                nn.ReLU(),
                nn.Linear(16, 8)
            )
            
            # 3. 融合层
            self.fc_final = nn.Linear(hidden_size + 8, 1)
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        
        # 拆分特征
        x_surface = x[:, :, :self.n_surface]
        
        # Stream 1
        out_gru, _ = self.gru(x_surface)
        feat_gru = out_gru[:, -1, :] # (batch, hidden)
        
        if self.n_sp > 0:
            # Stream 2
            x_sp = x[:, -1, self.n_surface:] # (batch, n_sp) 取最后时刻
            
            # 应用门控
            gate = torch.sigmoid(self.sp_gate)
            x_sp = x_sp * gate
            
            feat_sp = self.sp_fc(x_sp) # (batch, 8)
            
            # Fusion
            combined = torch.cat([feat_gru, feat_sp], dim=1)
            out = self.fc_final(combined)
        else:
            out = self.fc_base(feat_gru)
            
        return out

# 新增 Dual-Stream Transformer 模型
class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, nhead=2, dropout=0.3):
        super(TransformerModel, self).__init__()
        
        self.n_surface = 3
        self.n_sp = input_size - self.n_surface
        
        # ---- Stream 1: Surface Transformer ----
        self.input_proj = nn.Linear(self.n_surface, hidden_size)
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, 
                                                  dim_feedforward=hidden_size*2,
                                                  dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc_base = nn.Linear(hidden_size, 1)
        
        # ---- Stream 2: Physics MLP ----
        if self.n_sp > 0:
            self.sp_gate = nn.Parameter(torch.ones(self.n_sp))
            self.sp_fc = nn.Sequential(
                nn.Linear(self.n_sp, 16),
                nn.ReLU(),
                nn.Linear(16, 8)
            )
            self.fc_final = nn.Linear(hidden_size + 8, 1)
        
    def forward(self, x):
        # Stream 1
        x_surface = x[:, :, :self.n_surface]
        x_emb = self.input_proj(x_surface)
        out_trans = self.transformer_encoder(x_emb)
        feat_trans = out_trans[:, -1, :]
        
        if self.n_sp > 0:
            # Stream 2
            x_sp = x[:, -1, self.n_surface:]
            
            # Gate
            gate = torch.sigmoid(self.sp_gate)
            x_sp = x_sp * gate
            
            feat_sp = self.sp_fc(x_sp)
            
            # Fusion
            combined = torch.cat([feat_trans, feat_sp], dim=1)
            out = self.fc_final(combined)
        else:
            out = self.fc_base(feat_trans)
            
        return out

# 包装器类，用于统一 PyTorch 模型的训练和预测接口
class PyTorchWrapper:
    def __init__(self, model_class, input_size, hidden_size=64, num_layers=2, learning_rate=0.001):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model_class(input_size, hidden_size, num_layers).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
    def fit(self, X_train, y_train, epochs=200, batch_size=32):
        self.model.train()
        X_tensor = torch.FloatTensor(X_train).to(self.device)
        y_tensor = torch.FloatTensor(y_train).to(self.device)
        
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
    def predict(self, X_test):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_test).to(self.device)
            outputs = self.model(X_tensor)
            return outputs.cpu().numpy()

class ModelComparator:
    def __init__(self, X_train, y_train, X_test, y_test, scaler):
        """
        X_train, X_test: 形状为 (samples, seq_len, features) 的序列数据
        y_train, y_test: 形状为 (samples, 1) 的标签
        scaler: 用于反归一化或处理特征的缩放器
        """
        self.X_train_seq = X_train
        self.y_train = y_train
        self.X_test_seq = X_test
        self.y_test = y_test
        self.scaler = scaler
        
        # 展平时序特征用于传统机器学习模型 (XGBoost)
        # Base (3特征): (samples, seq_len*3)
        n_samples_train, seq_len, _ = X_train.shape
        self.X_train_flat_base = X_train[:, :, :3].reshape(n_samples_train, -1)
        
        n_samples_test = X_test.shape[0]
        self.X_test_flat_base = X_test[:, :, :3].reshape(n_samples_test, -1)
        
        # SP (7特征): (samples, seq_len*7)
        self.X_train_flat_sp = X_train.reshape(n_samples_train, -1)
        self.X_test_flat_sp = X_test.reshape(n_samples_test, -1)
        
        # 结果字典
        self.results = {}
        
    def train_evaluate_all(self, epochs=200, include_sp_lstm_retrain=False):
        """
        训练所有对比模型，成对比较 (Base vs SP)
        Base: 仅使用 3 维表象特征 [t_rise, Energy, IC_peak]
        SP: 使用 7 维特征 (3表象 + 4物理)
        """
        print("开始训练对比模型...")
        
        # 定义模型列表
        models_config = [
            ("XGBoost", None), # XGBoost 不需要 PyTorch wrapper
            ("GRN", GRN),
            ("Transformer", TransformerModel),
            ("LSTM", None) # LSTM 使用现有的 SOHEstimator
        ]
        
        seq_len = self.X_train_seq.shape[1]
        
        for name, model_cls in models_config:
            # 强制设置随机种子以确保每次运行结果一致
            torch.manual_seed(42)
            np.random.seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(42)
                
            print(f"\n>>> 正在评估 {name} 模型...")
            
            # --- Base Model (3 features) ---
            print(f"  训练 Base-{name} (3 features)...")
            if name == "XGBoost":
                model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
                model.fit(self.X_train_flat_base, self.y_train.ravel())
                y_pred = model.predict(self.X_test_flat_base).reshape(-1, 1)
            elif name == "LSTM":
                # 使用 SOHEstimator
                estimator = SOHEstimator(input_size=3, seq_len=seq_len)
                estimator.train(self.X_train_seq[:, :, :3], self.y_train, epochs=epochs, verbose=False)
                y_pred_raw = estimator.predict(self.X_test_seq[:, :, :3], apply_offset=False)
                # 应用校正
                estimator.calibrate_initial_state(y_pred_raw, self.y_test)
                y_pred = estimator.predict(self.X_test_seq[:, :, :3], apply_offset=True)
            else:
                # GRN / Transformer
                wrapper = PyTorchWrapper(model_cls, input_size=3)
                wrapper.fit(self.X_train_seq[:, :, :3], self.y_train, epochs=epochs)
                y_pred = wrapper.predict(self.X_test_seq[:, :, :3])
                y_pred = self._apply_initial_correction(y_pred, f"Base-{name}")
                
            self._log_result(f"Base-{name}", self.y_test, y_pred)
            
            # --- SP Model (7 features) ---
            print(f"  训练 SP-{name} (7 features)...")
            if name == "XGBoost":
                model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
                model.fit(self.X_train_flat_sp, self.y_train.ravel())
                y_pred = model.predict(self.X_test_flat_sp).reshape(-1, 1)
            elif name == "LSTM":
                if include_sp_lstm_retrain:
                    estimator = SOHEstimator(input_size=7, seq_len=seq_len)
                    estimator.train(self.X_train_seq, self.y_train, epochs=epochs, verbose=False)
                    y_pred_raw = estimator.predict(self.X_test_seq, apply_offset=False)
                    estimator.calibrate_initial_state(y_pred_raw, self.y_test)
                    y_pred = estimator.predict(self.X_test_seq, apply_offset=True)
                else:
                    print("  跳过 SP-LSTM 重新训练，使用外部传入的 'SP-LSTM (Ours)' 结果")
                    y_pred = None
            else:
                # GRN / Transformer
                wrapper = PyTorchWrapper(model_cls, input_size=7)
                wrapper.fit(self.X_train_seq, self.y_train, epochs=epochs)
                y_pred = wrapper.predict(self.X_test_seq)
                y_pred = self._apply_initial_correction(y_pred, f"SP-{name}")
                
            if y_pred is not None:
                self._log_result(f"SP-{name}", self.y_test, y_pred)
    
    def _apply_initial_correction(self, y_pred, model_name=""):
        """
        应用初始状态校正机制（与SP-LSTM的calibrate_initial_state完全一致）
        使用前N个样本的平均偏差进行校正
        """
        if len(y_pred) > 0 and len(self.y_test) > 0:
            n_init = min(5, len(y_pred))
            initial_offset = np.mean(self.y_test[:n_init] - y_pred[:n_init])
            y_pred_corrected = y_pred + initial_offset
            print(f"  -> {model_name} 初始状态校正偏移量: {initial_offset:.6f}")
            return y_pred_corrected
        return y_pred
    
    def _log_result(self, name, y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        self.results[name] = {
            'RMSE': rmse,
            'MAE': mae,
            'y_pred': y_pred
        }
        print(f"模型: {name} | RMSE: {rmse:.5f} | MAE: {mae:.5f}")

    def add_result(self, name, y_pred):
        """手动添加外部模型结果 (如 SP-LSTM)"""
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        mae = mean_absolute_error(self.y_test, y_pred)
        self.results[name] = {
            'RMSE': rmse,
            'MAE': mae,
            'y_pred': y_pred
        }
        
    def plot_comparison(self, save_dir='plots/comparison', fig_labels=None):
        """
        绘制算法对比图
        fig_labels: list of 2 strings, e.g. ['图 6-3', '图 6-4']
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        if not fig_labels or len(fig_labels) < 2:
            fig_labels = ['', '']
            
        # 1. 柱状图对比 RMSE
        names = list(self.results.keys())
        model_types = ['XGBoost', 'GRN', 'Transformer', 'LSTM']
        ordered_names = []
        for mt in model_types:
            base_name = f'Base-{mt}'
            sp_name = f'SP-{mt}' if mt != 'LSTM' else ('SP-LSTM (Ours)' if 'SP-LSTM (Ours)' in names else 'SP-LSTM')
            if base_name in names:
                ordered_names.append(base_name)
            if sp_name in names:
                ordered_names.append(sp_name)
        names = ordered_names + [n for n in names if n not in ordered_names]
        rmses = [self.results[n]['RMSE'] for n in names]
        
        plt.figure(figsize=(14, 7))
        # 使用 viridis 色系，但分组着色
        # 假设 names 是成对的: Base-XGB, SP-XGB, Base-GRN, SP-GRN...
        # 我们使用两种主色调区分 Base 和 SP
        
        x = np.arange(len(names))
        width = 0.6
        
        # 自动分配颜色: Base 用浅色，SP 用深色
        colors = []
        base_color = plt.get_cmap('Blues')(0.5)
        sp_color = plt.get_cmap('Oranges')(0.5)
        
        # 更好的配色方案：每组用不同的色系，但 Base 浅 SP 深
        cmaps = ['Blues', 'Greens', 'Purples', 'Reds']
        model_types = ['XGBoost', 'GRN', 'Transformer', 'LSTM']
        
        for name in names:
            is_sp = name.startswith('SP-') or ('SP-LSTM' in name)
            model_type = name.replace('Base-', '').replace('SP-', '').replace(' (Ours)', '')
            
            # 找到对应的色系
            cmap_name = 'Greys' # 默认
            for i, mt in enumerate(model_types):
                if mt in model_type:
                    cmap_name = cmaps[i % len(cmaps)]
                    break
            
            intensity = 0.7 if is_sp else 0.4
            colors.append(plt.get_cmap(cmap_name)(intensity))
            
        bars = plt.bar(names, rmses, color=colors, width=width)
        
        plt.ylabel('RMSE [dimensionless]', fontweight='bold')
        title = 'Performance Comparison: Base (Surface) vs SP (Physics-Informed)'
        plt.title(title, fontweight='bold', fontsize=14)
        
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.xticks(rotation=15, ha='right') # 旋转标签防止重叠
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.4f}',
                     ha='center', va='bottom', fontsize=10, fontweight='bold')
                     
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'model_rmse_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. SOH 预测曲线对比 (仅展示 SP 模型以避免混乱)
        plt.figure(figsize=(12, 7))
        cmap = plt.get_cmap('tab10')
        
        plt.plot(self.y_test, 'k-', linewidth=2, label='Ground Truth')
        
        # 筛选 SP 模型进行绘制
        sp_models = [n for n in names if n.startswith('SP-') or ('SP-LSTM' in n)]
        
        for i, name in enumerate(sp_models):
            marker = ['o', 's', '^', 'D'][i % 4]
            plt.plot(self.results[name]['y_pred'], marker=marker, linestyle='--', 
                     linewidth=2, markersize=5, 
                     markevery=max(1, len(self.y_test)//20), 
                     label=name, alpha=0.8)
            
        plt.xlabel('Test Samples', fontweight='bold')
        plt.ylabel('State of Health (SOH)', fontweight='bold')
        
        title = 'SOH Estimation Curve Comparison'
        plt.title(title, fontweight='bold')
        
        plt.legend(loc='upper right', frameon=True, framealpha=0.9)
        plt.grid(True, linestyle='--', alpha=0.3)
        
        fname = 'soh_curve_comparison.png'
            
        plt.savefig(os.path.join(save_dir, fname), dpi=300, bbox_inches='tight')
        plt.close()
