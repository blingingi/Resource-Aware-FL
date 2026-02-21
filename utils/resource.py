import numpy as np

class ResourceManager:
    def __init__(self, num_users):
        self.num_users = num_users
        self.profiles = {}
        # 记录每个用户被选中的次数 (用于公平性)
        self.selection_counts = np.zeros(num_users)
        
        # 模拟三种不同类型的设备
        # 1. High-End (算力强，带宽大，但可能耗电高)
        # 2. Mid-Range (普通)
        # 3. Low-End (IoT设备，算得慢，带宽小，必须节能)
        device_types = ['High', 'Mid', 'Low']
        
        np.random.seed(42) # 固定随机种子方便复现
        
        for i in range(num_users):
            d_type = np.random.choice(device_types, p=[0.2, 0.5, 0.3])
            
            if d_type == 'High':
                compute_speed = np.random.uniform(8.0, 10.0) # FLOPs per cycle (相对值)
                bandwidth = np.random.uniform(8.0, 10.0)     # MB/s
                power = np.random.uniform(4.0, 5.0)          # Watt
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

    def calculate_cost(self, user_idx, data_samples, model_size_mb=2.5):
        """
        计算该用户的预估时延和能耗
        :param user_idx: 用户ID
        :param data_samples: 数据量
        :param model_size_mb: 模型大小 (CIFAR CNN 约为 2~3MB)
        :return: (time_cost, energy_cost)
        """
        profile = self.profiles[user_idx]
        
        # 1. 计算时延 (Computation Time) = (Data * Complexity) / Speed
        # 假设处理一张图需要 0.01 相对算力单位
        comp_load = data_samples * 0.01
        t_comp = comp_load / profile['compute']
        
        # 2. 通信时延 (Communication Time) = ModelSize / Bandwidth
        t_comm = model_size_mb / profile['bandwidth']
        
        total_time = t_comp + t_comm
        
        # 3. 能耗 (Energy) = Power * Time
        # 简单起见，假设计算和通信功率差不多 (也可以分开设)
        total_energy = total_time * profile['power']
        
        return total_time, total_energy

    def update_selection(self, selected_users):
        """更新被选次数，用于公平性约束"""
        for i in selected_users:
            self.selection_counts[i] += 1
            
    def get_fairness_weights(self):
        """
        计算公平性权重：选得越少，权重越大
        w = 1 / (count + 1)
        """
        return 1.0 / (self.selection_counts + 1.0)