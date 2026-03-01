
import numpy as np
from scipy.optimize import minimize, brentq
from src.spm_model import DualElectrodeSPM

class HybridOptimizer:
    def __init__(self, model: DualElectrodeSPM, r_f_bounds=(0.01, 0.2)):
        self.model = model
        # 参数边界 (根据文献优化：仅辨识 D_p, D_n, k_p, k_n)
        # c_max 和 R 设为常量，不参与辨识
        self.bounds = [
            (1e-16, 1e-12), # D_p
            (1e-16, 1e-12), # D_n
            (1e-13, 1e-8),  # k_p
            (1e-13, 1e-8),  # k_n
        ]
        
        # 如果需要辨识 R_f (测试电池)
        self.r_f_bounds = r_f_bounds
        
        # 正则化: 上一轮辨识参数 (用于平滑约束)
        self.prev_params = None
        # 降低正则化强度，优先保证电压拟合精度
        self.reg_lambda = 0.0 # 完全移除正则化，以追求 RMSE < 0.01 的极限拟合
        
    def _solve_initial_soc(self, V_target):
        """
        使用二分法求解初始 SOC，使得 OCV(SOC) = V_target
        """
        def error_func(soc):
            return self.model.compute_ocv_voltage(soc) - V_target
            
        try:
            # 假设 SOC 在 0 到 1 之间
            # 考虑到 OCV 可能不单调或有噪声，限制范围
            soc_sol = brentq(error_func, 0.0, 1.0, xtol=1e-3)
            return soc_sol
        except ValueError:
            # 如果解不在 [0, 1] 范围内 (例如电压过高或过低)
            if error_func(1.0) * error_func(0.0) > 0:
                # 同号，说明在范围外
                if abs(error_func(1.0)) < abs(error_func(0.0)):
                    return 1.0
                else:
                    return 0.0
            return 0.5 # 默认

    def objective_function(self, params, t, I, V_meas):
        # 更新模型参数
        self.model.update_params(params)
        
        R_f = getattr(self.model, 'R_f', 0.05)
        V_ocv_target = V_meas[0] + I[0] * R_f
        V_ocv_target = np.clip(V_ocv_target, 3.0, 4.3)
        
        init_soc = self._solve_initial_soc(V_ocv_target)
        self.model.reset_state(init_soc)
            
        # 模拟
        try:
            V_pred, _ = self.model.simulate_cycle(t, I)
        except Exception as e:
            return 1e6
            
        # 加权 RMSE
        # 为了进一步降低整体 RMSE (目标 0.01V)，我们移除局部加权，
        # 让优化器专注于全局最小均方误差 (MSE)
        weights = np.ones_like(V_meas)
        # mask = (V_meas >= 3.8) & (V_meas <= 4.1)
        # weights[mask] = 2.0
        
        mse = np.average((V_pred - V_meas)**2, weights=weights)
        rmse = np.sqrt(mse)
        
        # 正则化惩罚: 强力约束参数随老化的单调性与平滑性
        reg_penalty = 0.0
        if self.prev_params is not None:
            n_comp = min(len(params), len(self.prev_params))
            for i in range(n_comp):
                if params[i] > 0 and self.prev_params[i] > 0:
                    # 1. 平滑约束 (L2 惩罚对数比值)
                    log_ratio = np.log(params[i] / self.prev_params[i])
                    # 降低平滑约束权重，允许参数根据数据需要进行调整
                    reg_penalty += (log_ratio ** 2) * 0.1 * self.reg_lambda
                    
                    # 2. 物理单调性约束: D 和 k 应该随老化 (循环增加) 而下降
                    # params[i] (当前) 应该 <= self.prev_params[i] (上一轮)
                    # 如果当前值比上一轮大，施加极重的惩罚
                    if i < 4: # D_p, D_n, k_p, k_n
                        if params[i] > self.prev_params[i] * 1.02: # 允许 2% 的波动
                            diff_ratio = (params[i] - self.prev_params[i]) / self.prev_params[i]
                            # 增大惩罚权重以确保物理一致性 (考虑到 lambda=1e-6)
                            reg_penalty += (diff_ratio ** 2) * 2e5 * self.reg_lambda
            
            # 3. 特殊处理 R_f (如果存在，它是索引 4)
            if len(params) == 5 and len(self.prev_params) >= 5:
                # R_f 应该随老化单调上升
                if params[4] < self.prev_params[4] * 0.98:
                    diff_ratio = (self.prev_params[4] - params[4]) / self.prev_params[4]
                    reg_penalty += (diff_ratio ** 2) * 2e5 * self.reg_lambda
        
        # 额外的R_f范围惩罚
        if len(params) == 5:
            r_f = params[4]
            r_f_low, r_f_high = self.r_f_bounds
            if r_f < (r_f_low + (r_f_high - r_f_low) * 0.2):
                distance = (r_f - r_f_low) / (r_f_high - r_f_low)
                reg_penalty += (1.0 - distance) * 10.0 * self.reg_lambda
            elif r_f > (r_f_high - (r_f_high - r_f_low) * 0.2):
                distance = (r_f_high - r_f) / (r_f_high - r_f_low)
                reg_penalty += (1.0 - distance) * 10.0 * self.reg_lambda
        
        return rmse + reg_penalty

    def pso_search(self, t, I, V_meas, n_particles=30, n_iterations=20):
        # 自适应 PSO 实现
        dim = len(self.bounds)
        
        # 初始化粒子
        positions = np.zeros((n_particles, dim))
        velocities = np.zeros((n_particles, dim))
        pbest_pos = np.zeros((n_particles, dim))
        pbest_val = np.full(n_particles, np.inf)
        gbest_pos = np.zeros(dim)
        gbest_val = np.inf
        
        # 在边界内初始化位置
        # 如果有上一轮参数，将一半粒子初始化在其附近（局部搜索）
        n_local = n_particles // 2 if self.prev_params is not None else 0
        
        for i in range(dim):
            low, high = self.bounds[i]
            # 全局随机粒子
            positions[n_local:, i] = np.random.uniform(low, high, n_particles - n_local)
            velocities[:, i] = np.random.uniform(-1, 1, n_particles) * (high - low) * 0.1
        
        if self.prev_params is not None:
            # 局部粒子: 在上一轮参数附近对数扰动 (±0.3个数量级)
            for i in range(dim):
                low, high = self.bounds[i]
                center = self.prev_params[i]
                for j in range(n_local):
                    log_perturb = np.random.normal(0, 0.2)  # ~±0.2个数量级，减小扰动范围
                    val = center * (10 ** log_perturb)
                    positions[j, i] = np.clip(val, low, high)
        else:
            # 首次初始化时，对于 R_f 参数（第8个参数，索引8），使用边界的中间值
            # 确保初始值在训练电池电阻范围内
            if dim >= 9:
                r_f_idx = 8  # R_f 是第9个参数（索引8）
                low, high = self.bounds[r_f_idx]
                # 将所有粒子的 R_f 初始值设置在边界中间
                for j in range(n_particles):
                    positions[j, r_f_idx] = (low + high) / 2
            
        # 自适应参数
        w_max, w_min = 0.9, 0.4
        c1_max, c1_min = 2.5, 0.5
        c2_max, c2_min = 0.5, 2.5
        
        # 循环
        for it in range(n_iterations):
            # 线性递减惯性权重
            w = w_max - (w_max - w_min) * (it / n_iterations)
            # 时变加速系数
            # c1 (认知) 减少, c2 (社会) 增加
            c1 = c1_max - (c1_max - c1_min) * (it / n_iterations)
            c2 = c2_min + (c2_max - c2_min) * (it / n_iterations)
            
            for i in range(n_particles):
                # 评估
                val = self.objective_function(positions[i], t, I, V_meas)
                
                # 更新 PBest
                if val < pbest_val[i]:
                    pbest_val[i] = val
                    pbest_pos[i] = positions[i].copy()
                    
                # 更新 GBest
                if val < gbest_val:
                    gbest_val = val
                    gbest_pos = positions[i].copy()
            
            # 更新速度和位置
            r1 = np.random.rand(n_particles, dim)
            r2 = np.random.rand(n_particles, dim)
            
            velocities = w * velocities + c1 * r1 * (pbest_pos - positions) + c2 * r2 * (gbest_pos - positions)
            positions = positions + velocities
            
            # 限制边界
            for d in range(dim):
                positions[:, d] = np.clip(positions[:, d], self.bounds[d][0], self.bounds[d][1])
                
        return gbest_pos

    def run(self, t, I, V_meas, prev_params=None, identify_rf=False):
        """
        运行 PSO-LBFGS 混合优化
        
        参数:
            t, I, V_meas: 时间、电流、测量电压
            prev_params: 上一轮辨识参数, 用于正则化平滑约束
            identify_rf: 是否同时辨识 R_f (测试电池)
        """
        # 设置正则化参考
        self.prev_params = prev_params
        
        # 备份原始边界
        original_bounds = self.bounds.copy()
        
        # 如果需要辨识 R_f，临时修改边界
        if identify_rf:
            self.bounds = original_bounds + [self.r_f_bounds]
        
        # 1. PSO 阶段
        # 针对首次辨识 (prev_params is None)，使用更强的搜索力度以避免陷入局部最优
        # 这可以解决初期 RMSE 过大的问题
        if self.prev_params is None:
            n_part = 300
            n_iter = 200
        else:
            n_part = 100
            n_iter = 100
            
        initial_guess = self.pso_search(t, I, V_meas, n_particles=n_part, n_iterations=n_iter)
        
        # 如果是首次辨识，也增加 L-BFGS-B 的最大迭代次数
        max_bfgs_iter = 2000 if self.prev_params is None else 800
        
        # 2. L-BFGS-B 阶段
        result = minimize(
            self.objective_function, 
            initial_guess, 
            args=(t, I, V_meas), 
            method='L-BFGS-B', 
            bounds=self.bounds,
            options={'ftol': 1e-12, 'maxiter': max_bfgs_iter} # 增加迭代次数以进一步降低 RMSE
        )
        
        # 返回纯 RMSE (不含正则化) 用于报告
        self.prev_params = None  # 临时关闭正则化
        pure_rmse = self.objective_function(result.x, t, I, V_meas)
        self.prev_params = prev_params  # 恢复
        
        # 恢复原始边界
        self.bounds = original_bounds
        
        return result.x, pure_rmse

