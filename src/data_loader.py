
import os
import glob
import pandas as pd
import numpy as np

class DataLoader:
    def __init__(self, data_dir, metadata_path=None, target_batteries=None):
        """
        数据加载器
        data_dir: CSV 文件所在目录
        metadata_path: metadata.csv 文件路径
        target_batteries: 目标电池 ID 列表，如 ['B0005', 'B0006']
        """
        self.data_dir = data_dir
        
        # 加载元数据
        if metadata_path and os.path.exists(metadata_path):
            print(f"读取元数据: {metadata_path}...")
            self.metadata = pd.read_csv(metadata_path)
        else:
            self.metadata = None
            print("未提供元数据路径，将使用所有文件。")
            
        self.target_batteries = target_batteries
        
        # 缓存各电池的初始容量 Q_initial
        self._initial_capacities = {}
        
        # 扫描文件并分组
        self.battery_data = self._group_cycles_by_battery()
        
        print(f"已加载电池: {list(self.battery_data.keys())}")
        for bid, cycles in self.battery_data.items():
            print(f"  - {bid}: {len(cycles)} 个循环")
        
    def _group_cycles_by_battery(self):
        """按电池 ID 分组循环数据"""
        battery_data = {}
        
        if self.metadata is not None and self.target_batteries:
            # 过滤元数据
            # 确保 battery_id 匹配
            # metadata.csv 列: ..., battery_id, filename, ...
            
            for bid in self.target_batteries:
                print(f"正在处理电池 {bid}...")
                battery_df = self.metadata[self.metadata['battery_id'] == bid]
                
                # 获取该电池的所有文件，按时间排序 (metadata 应该已经是排序的，或者有 start_time)
                # 假设 metadata 按顺序排列，或者我们需要排序
                # metadata 中 filename 是 '00001.csv' 格式
                
                # 提取循环
                # 逻辑: 遍历该电池的所有记录，找到 Discharge，并关联最近的 Charge
                
                cycles = []
                charge_file = None
                last_Re = 0.05  # 默认内阻 50mΩ
                
                # 遍历行
                for _, row in battery_df.iterrows():
                    ftype = row['type']
                    fname = row['filename']
                    fpath = os.path.join(self.data_dir, fname)
                    
                    if not os.path.exists(fpath):
                        continue
                    
                    # 记录最近一次阻抗测量的 Re 值
                    if ftype == 'impedance':
                        re_val = row.get('Re', np.nan)
                        if pd.notna(re_val):
                            last_Re = float(re_val)
                        
                    if ftype == 'charge':
                        charge_file = fpath
                    elif ftype == 'discharge':
                        # 找到一个放电过程，构成一个循环
                        # 关联最近的充电文件和阻抗测量的 Re
                        cycles.append({
                            'discharge': fpath,
                            'charge': charge_file,
                            'Re': last_Re  # 使用最近一次阻抗测量的 Re
                        })
                        
                battery_data[bid] = cycles
                
        else:
            # 旧逻辑：不使用元数据，直接扫描目录 (不推荐用于多电池)
            print("警告: 未使用元数据过滤，默认处理所有文件为单个序列 (可能混合多个电池)。")
            files = sorted(glob.glob(os.path.join(self.data_dir, "*.csv")))
            # ... (这里可以保留旧逻辑用于兼容，或者直接强制要求 metadata)
            # 为了简单，如果没 metadata，就当做一个未知电池
            cycles = []
            charge_files = []
            for f in files[:200]: # 限制数量
                 try:
                    with open(f, 'r') as file: header = file.readline().strip()
                    if "Voltage_measured" in header:
                        df = pd.read_csv(f, nrows=10)
                        current = df['Current_measured'].mean()
                        if current < -0.1:
                            cycles.append({'discharge': f, 'charge': charge_files[-1] if charge_files else None})
                        elif current > 0.1:
                            charge_files.append(f)
                 except: pass
            battery_data['unknown'] = cycles
            
        return battery_data

    def load_cycle_data(self, battery_id, cycle_idx):
        """加载特定电池的特定循环数据"""
        if battery_id not in self.battery_data:
            return None
        
        cycles = self.battery_data[battery_id]
        if cycle_idx >= len(cycles):
            return None
            
        group = cycles[cycle_idx]
        d_file = group['discharge']
        c_file = group['charge']
        
        data = {}
        
        # 加载放电数据
        df_d = pd.read_csv(d_file)
        data['discharge'] = {
            't': df_d['Time'].values,
            'V': df_d['Voltage_measured'].values,
            'I': df_d['Current_measured'].values,
            'T': df_d['Temperature_measured'].values
        }
        
        # 加载充电数据 (如果存在)
        if c_file:
            df_c = pd.read_csv(c_file)
            data['charge'] = {
                't': df_c['Time'].values,
                'V': df_c['Voltage_measured'].values,
                'I': df_c['Current_measured'].values
            }
        else:
            data['charge'] = None
            
        return data

    def _compute_capacity(self, t, I):
        """计算放电容量 Q [Ah]"""
        # 按时间排序以防万一
        idx = np.argsort(t)
        t = t[idx]
        I = I[idx]
        
        dt = np.diff(t, prepend=t[0])
        dt[0] = dt[1] if len(dt) > 1 else 0
        
        # 放电电流为负，积分取负号得到正容量
        Q = np.sum(-I * dt) / 3600.0  # [Ah]
        return Q

    def get_initial_capacity(self, battery_id):
        """获取电池首次循环的实测容量 Q_initial [Ah]"""
        if battery_id not in self._initial_capacities:
            # 计算首次循环容量
            data = self.load_cycle_data(battery_id, 0)
            if data:
                t = data['discharge']['t']
                I = data['discharge']['I']
                Q_initial = self._compute_capacity(t, I)
                self._initial_capacities[battery_id] = Q_initial
            else:
                # 如果无法计算，使用标称容量作为后备
                self._initial_capacities[battery_id] = 2.0
        return self._initial_capacities[battery_id]

    def get_soh(self, battery_id, cycle_idx):
        """
        计算SOH = Q_k / Q_initial × 100%
        使用首次循环实测容量作为基准，消除电池个体差异
        """
        data = self.load_cycle_data(battery_id, cycle_idx)
        if not data: return None
        
        # 计算当前循环容量
        t = data['discharge']['t']
        I = data['discharge']['I']
        Q_k = self._compute_capacity(t, I)
        
        # 获取该电池的初始容量
        Q_initial = self.get_initial_capacity(battery_id)
        
        # SOH = Q_k / Q_initial
        SOH = Q_k / Q_initial
        
        return SOH

    def get_Re(self, battery_id, cycle_idx):
        """
        获取指定循环的欧姆内阻 Re [Ohm]
        该值来自 metadata.csv 中最近一次阻抗测量
        """
        if battery_id not in self.battery_data:
            return 0.05  # 默认值
        
        cycles = self.battery_data[battery_id]
        if cycle_idx >= len(cycles):
            return 0.05
            
        return cycles[cycle_idx].get('Re', 0.05)
    
    def get_battery_Re_range(self, battery_ids):
        """
        计算指定电池列表的电阻范围
        
        参数:
            battery_ids: 电池ID列表
            
        返回:
            (min_Re, max_Re): 电阻的最小值和最大值
        """
        all_Re = []
        
        for bid in battery_ids:
            if bid in self.battery_data:
                cycles = self.battery_data[bid]
                for cycle in cycles:
                    Re = cycle.get('Re', 0.05)
                    all_Re.append(Re)
        
        if not all_Re:
            return 0.01, 0.2  # 默认范围
        
        min_Re = min(all_Re)
        max_Re = max(all_Re)
        
        # 添加一些余量，确保范围足够大
        margin = (max_Re - min_Re) * 0.1
        min_Re = max(0.01, min_Re - margin)
        max_Re = min(0.2, max_Re + margin)
        
        return min_Re, max_Re
