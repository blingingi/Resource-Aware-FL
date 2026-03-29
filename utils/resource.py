import numpy as np

class ResourceManager:
    def __init__(self, num_users, dict_users, local_ep=5, model_size_mb=2.5, limit_ratio=0.8):
        """
        初始化资源管理器 (完美契合联邦学习主流物理能耗模型)
        :param local_ep: 本地训练轮数，计算量与此成正比
        """
        self.num_users = num_users
        self.profiles = {}
        self.selection_counts = np.zeros(num_users)
        
        # --- 物理学常数设置 (根据顶级会议的经典 MEC 参数归一化) ---
        # cycles_per_sample: 处理单个样本所需的 CPU 周期数 (单位: 兆周期 Mcycles)
        self.cycles_per_sample = 20.0 
        # kappa: CPU 能量有效性系数 (底层芯片的电容特征)
        self.kappa = 0.05
        self.local_ep = local_ep

        # === 1. 初始化物理设备 ===
        device_types = ['High', 'Mid', 'Low']
        np.random.seed(42) 
        
        for i in range(num_users):
            d_type = np.random.choice(device_types, p=[0.2, 0.5, 0.3])
            
            # 使用更严谨的学术指标：CPU频率(GHz), 带宽(Mbps), 传输功率(Watt)
            if d_type == 'High':
                f_i = np.random.uniform(2.0, 2.5)      # CPU频率 (GHz)
                R_i = np.random.uniform(10.0, 20.0)    # 通信带宽/传输速率 (Mbps)
                p_i = np.random.uniform(1.0, 1.5)      # 通信发射功率 (W)
            elif d_type == 'Mid':
                f_i = np.random.uniform(1.2, 1.8)
                R_i = np.random.uniform(4.0, 8.0)
                p_i = np.random.uniform(0.5, 0.8)
            else: # Low
                f_i = np.random.uniform(0.8, 1.0)
                R_i = np.random.uniform(1.0, 2.0)
                p_i = np.random.uniform(0.1, 0.3)
                
            self.profiles[i] = {
                'type': d_type,
                'f_i': f_i,
                'R_i': R_i,
                'p_i': p_i
            }

        # === 2. 静态预计算所有客户端的开销 ===
        self.time_costs = np.zeros(num_users)
        self.energy_costs = np.zeros(num_users)
        
        for i in range(num_users):
            data_samples = len(dict_users[i])
            profile = self.profiles[i]
            
            # ---------------------------------------------------------
            # [核心公式 1: 计算阶段 Computation]
            # 总 CPU 周期 (Mcycles) = 样本数 * Epoch * 每个样本的周期
            # ---------------------------------------------------------
            total_cycles = data_samples * self.local_ep * self.cycles_per_sample
            
            # 计算时延 = 总周期 / CPU频率 (因频率单位是GHz，周期是Mcycles，需除以1000换算秒)
            t_comp = total_cycles / (profile['f_i'] * 1000)
            
            # 计算能耗 = kappa * 总周期 * (CPU频率)^2
            e_comp = self.kappa * total_cycles * (profile['f_i'] ** 2)

            # ---------------------------------------------------------
            # [核心公式 2: 通信阶段 Communication]
            # ---------------------------------------------------------
            # 通信时延 = 模型大小 / 传输速率
            t_comm = model_size_mb / profile['R_i']
            
            # 通信能耗 = 传输功率 * 通信时延 (线性)
            e_comm = profile['p_i'] * t_comm

            # ---------------------------------------------------------
            # 总开销汇总
            # ---------------------------------------------------------
            self.time_costs[i] = t_comp + t_comm
            self.energy_costs[i] = e_comp + e_comm

        # === 3. 生成符合客观物理规律的 Lyapunov 红线 ===
        self.T_limit = np.mean(self.time_costs) * limit_ratio
        self.E_limit = np.mean(self.energy_costs) * limit_ratio
        
        print(f"[ResourceManager] 初始化完毕. Lya 约束红线设为 -> Time: {self.T_limit:.3f}, Energy: {self.E_limit:.3f}")

        # === 4. 初始化 Lyapunov 虚拟队列 ===
        self.Q_time = np.zeros(num_users)
        self.Q_energy = np.zeros(num_users)

    def get_penalty(self, user_idx):
        """
        返回某个客户端当前的资源惩罚值 (Q * Cost)
        供贪心算法直接调用
        """
        e_penalty = self.Q_energy[user_idx] * self.energy_costs[user_idx]
        t_penalty = self.Q_time[user_idx] * self.time_costs[user_idx]
        return e_penalty + t_penalty

    def update_queues_and_counts(self, selected_users):
        """
        每轮结束时调用，演进李雅普诺夫队列并更新公平性计数
        :return: (Avg_Q_time, Avg_Q_energy) 用于日志监控
        """
        for i in range(self.num_users):
            if i in selected_users:
                # 选中者：排队长度增加 (积累资源债)
                self.selection_counts[i] += 1
                self.Q_time[i] = max(0.0, self.Q_time[i] + self.time_costs[i] - self.T_limit)
                self.Q_energy[i] = max(0.0, self.Q_energy[i] + self.energy_costs[i] - self.E_limit)
            else:
                # 未选中者：排队长度缩减（偿还资源债）
                self.Q_time[i] = max(0.0, self.Q_time[i] - self.T_limit)
                self.Q_energy[i] = max(0.0, self.Q_energy[i] - self.E_limit)
                
        return np.mean(self.Q_time), np.mean(self.Q_energy)