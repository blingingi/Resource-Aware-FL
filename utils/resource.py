import numpy as np

class ResourceManager:
    def __init__(self, num_users, dict_users, model_size_mb=2.5, limit_ratio=0.8):
        """
        初始化资源管理器并完美适配 Lyapunov 队列
        :param dict_users: 全局数据划分字典，用于获取真实数据量
        :param limit_ratio: 限制系数（如 0.8 表示要求系统平均开销只能达到理论均值的 80%
        """
        self.num_users = num_users
        self.profiles = {}
        self.selection_counts = np.zeros(num_users)
        
        # === 1. 初始化物理设备 ===
        device_types = ['High', 'Mid', 'Low']
        np.random.seed(42) 
        
        for i in range(num_users):
            d_type = np.random.choice(device_types, p=[0.2, 0.5, 0.3])
            
            if d_type == 'High':
                compute_speed = np.random.uniform(8.0, 10.0) 
                bandwidth = np.random.uniform(8.0, 10.0)    
                power = np.random.uniform(4.0, 5.0)          
            elif d_type == 'Mid':
                compute_speed = np.random.uniform(4.0, 6.0)
                bandwidth = np.random.uniform(4.0, 6.0)
                power = np.random.uniform(2.0, 3.0)
            else: # Low
                compute_speed = np.random.uniform(1.0, 2.0)
                bandwidth = np.random.uniform(1.0, 2.0)
                power = np.random.uniform(0.5, 1.0)
                
            self.profiles[i] = {
                'type': d_type,
                'compute': compute_speed,
                'bandwidth': bandwidth,
                'power': power
            }

        # === 2. 静态预计算所有客户端的开销 ===
        self.time_costs = np.zeros(num_users)
        self.energy_costs = np.zeros(num_users)
        
        for i in range(num_users):
            data_samples = len(dict_users[i])
            profile = self.profiles[i]
            
            # 时延 = 计算时间 + 通信时间
            t_comp = (data_samples * 0.01) / profile['compute']
            t_comm = model_size_mb / profile['bandwidth']
            self.time_costs[i] = t_comp + t_comm
            
            # 能耗 = 功率 * 时间
            self.energy_costs[i] = self.time_costs[i] * profile['power']

        # === 3. 生成符合客观物理规律的 Lyapunov 红线 ===
        # 以全体客户端的平均真实开销为基准，乘以限制比例
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
                # 选中者：排队长度增加
                self.selection_counts[i] += 1
                self.Q_time[i] = max(0.0, self.Q_time[i] + self.time_costs[i] - self.T_limit)
                self.Q_energy[i] = max(0.0, self.Q_energy[i] + self.energy_costs[i] - self.E_limit)
            else:
                # 未选中者：排队长度缩减（偿还资源债）
                self.Q_time[i] = max(0.0, self.Q_time[i] - self.T_limit)
                self.Q_energy[i] = max(0.0, self.Q_energy[i] - self.E_limit)
                
        return np.mean(self.Q_time), np.mean(self.Q_energy)