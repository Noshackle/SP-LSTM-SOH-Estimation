
import torch
import torch.nn as nn
import numpy as np

class SP_LSTM(nn.Module):
    def __init__(self, input_size=7, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        """
        SP-LSTM (双流架构): 物理增强 LSTM 用于 SOH 估计
        
        特征输入清单 (7维):
          - Stream 1 (Surface): 前3维 [t_rise, Energy, IC_peak]
          - Stream 2 (Physics): 后4维 [log_D_n, log_D_p, log_k_p, log_ratio_k]
        """
        super(SP_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 表象特征维度: [t_rise, Energy, IC_peak] (移除 Re)
        self.n_surface = 3
        # SPM 核心动力学特征维度: [log_D_n, log_D_p, log_k_p, log_ratio_k]
        self.n_sp = input_size - self.n_surface
        
        # ---- Stream 1: Surface LSTM (主预测流) ----
        self.lstm = nn.LSTM(self.n_surface, hidden_size, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0,
                            bidirectional=True)
        
        # 输出层（双向LSTM输出维度翻倍）
        self.bn = nn.BatchNorm1d(hidden_size * 2)
        self.dropout_layer = nn.Dropout(dropout)
        self.fc_base = nn.Linear(hidden_size * 2, output_size)
        
        # ---- Stream 2: Physics Correction (物理校正流) ----
        if self.n_sp > 0:
            # 使用更简单的特征融合方式
            self.sp_fc = nn.Sequential(
                nn.Linear(self.n_sp, 16),
                nn.ReLU(),
                nn.Linear(16, 8)
            )
            # 融合后的全连接层
            self.fc_final = nn.Linear(hidden_size * 2 + 8, output_size)
            
            # 可学习的融合权重
            self.fusion_bias = nn.Parameter(torch.tensor(-1.0))
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_size)
        """
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # ---- Stream 1: Surface LSTM ----
        x_surface = x[:, :, :self.n_surface]
        lstm_out, _ = self.lstm(x_surface)
        lstm_feat = lstm_out[:, -1, :]  # 取最后一个时间步
        lstm_feat = self.bn(lstm_feat)
        lstm_feat = self.dropout_layer(lstm_feat)
        
        if self.n_sp > 0:
            # ---- Stream 2: Physics Features ----
            # 取最后一个时间步的物理参数作为当前状态特征
            x_sp = x[:, -1, self.n_surface:]
            sp_feat = self.sp_fc(x_sp)
            
            # 特征拼接融合
            combined = torch.cat([lstm_feat, sp_feat], dim=1)
            soh = self.fc_final(combined)
        else:
            soh = self.fc_base(lstm_feat)
        
        return soh

class SOHEstimator:
    def __init__(self, input_size=7, seq_len=10, hidden_size=64, num_layers=2, dropout=0.2):
        self.model = SP_LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        self.seq_len = seq_len
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=5e-5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=100, T_mult=2, eta_min=1e-6
        )
        self.initial_offset = 0.0
        
    def train(self, X, y, epochs=800, verbose=True):
        self.model.train()
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        best_loss = float('inf')
        patience = 200
        patience_counter = 0
        best_state = None
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            outputs = self.model(X_tensor)
            
            mse_loss = self.criterion(outputs, y_tensor)
            
            # 趋势损失
            if len(outputs) > 1:
                pred_diff = outputs[1:] - outputs[:-1]
                true_diff = y_tensor[1:] - y_tensor[:-1]
                trend_loss = torch.mean((pred_diff - true_diff) ** 2)
                loss = mse_loss + 0.1 * trend_loss
            else:
                loss = mse_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                break
            
            if verbose and (epoch+1) % 50 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')
        
        if best_state is not None:
            self.model.load_state_dict(best_state)
                
    def predict(self, X, apply_offset=True):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            outputs = self.model(X_tensor)
        
        predictions = outputs.numpy()
        if apply_offset and self.initial_offset != 0.0:
            predictions = predictions + self.initial_offset
        return predictions
    
    def calibrate_initial_state(self, y_pred, y_true):
        if len(y_pred) > 0 and len(y_true) > 0:
            n_init = min(5, len(y_pred))
            self.initial_offset = np.mean(y_true[:n_init] - y_pred[:n_init])
            print(f"初始状态校正偏移量: {self.initial_offset:.6f}")
