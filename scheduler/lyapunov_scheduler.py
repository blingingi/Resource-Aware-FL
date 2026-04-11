# 文件路径: scheduler/lyapunov_scheduler.py

import numpy as np
import random

class LyapunovScheduler:
    def __init__(self, num_clients, num_classes, V_penalty, beta_target, max_energy):
        self.num_clients = num_clients
        self.num_classes = num_classes
        self.V = V_penalty          # 数据平衡权值 V
        self.beta = beta_target     # 目标参与率 (例如 0.2 表示长期来看每个设备参与 20% 的轮次)
        self.E_avg = max_energy     # 目标平均能耗阈值
        
        # 初始化虚拟队列
        self.Z = {k: 0.0 for k in range(num_clients)}  # 公平性队列
        self.q = {k: 0.0 for k in range(num_clients)}  # 能耗队列

    def calculate_skewness(self, selected_clients, client_data_sizes, client_label_distributions):
        """
        计算选中客户端的数据偏斜度 G(M)。
        严格按照 Xie 等人 2025 论文的公式 (14) 实现：
        G(M) = sum_{q=1}^{Q} ( (sum_{u in M} D_u * a_{u,q}) / (sum_{u in M} D_u) - 1/Q )^2
        """
        if not selected_clients:
            return float('inf') # 避免空集，空集的偏斜度视为无穷大
        
        total_D = sum(client_data_sizes[u] for u in selected_clients)
        if total_D == 0:
            return float('inf')

        skewness = 0.0
        # 遍历每一个类别 q
        for q in range(self.num_classes):
            # 计算选中集合中类别 q 的实际占比
            actual_q_ratio = sum(client_data_sizes[u] * client_label_distributions[u][q] for u in selected_clients) / total_D
            # 理想占比是 1/Q
            ideal_q_ratio = 1.0 / self.num_classes
            
            skewness += (actual_q_ratio - ideal_q_ratio) ** 2
            
        return skewness

    def objective_function(self, selected_clients, client_data_sizes, client_label_distributions, energy_profiles):
        """
        计算每轮的联合优化目标: V * G(S) + sum(q_k * e_k - Z_k)
        """
        if not selected_clients:
            return float('inf')
            
        # 1. 空间维度：数据不平衡惩罚
        g_penalty = self.V * self.calculate_skewness(selected_clients, client_data_sizes, client_label_distributions)
        
        # 2. 时间维度：队列漂移惩罚/奖励
        queue_penalty = sum([
            self.q[k] * energy_profiles[k] - self.Z[k] 
            for k in selected_clients
        ])
        
        return g_penalty + queue_penalty

    def double_greedy_selection(self, client_data_sizes, client_label_distributions, energy_profiles):
        """
        核心调度算法：双贪婪算法 (Double Greedy)
        """
        S1 = set()                              # 初始空集
        S2 = set(range(self.num_clients))       # 初始全集
        
        clients = list(range(self.num_clients))
        random.shuffle(clients) # 随机化遍历顺序，增加探索性
        
        for k in clients:
            # --- 方案 X: 尝试将 k 加入 S1 ---
            obj_S1 = self.objective_function(S1, client_data_sizes, client_label_distributions, energy_profiles)
            obj_S1_add = self.objective_function(S1.union({k}), client_data_sizes, client_label_distributions, energy_profiles)
            gain_x = max(obj_S1 - obj_S1_add, 0) # 收益：加入 k 后目标函数下降了多少
            
            # --- 方案 Y: 尝试将 k 从 S2 剔除 ---
            obj_S2 = self.objective_function(S2, client_data_sizes, client_label_distributions, energy_profiles)
            obj_S2_remove = self.objective_function(S2.difference({k}), client_data_sizes, client_label_distributions, energy_profiles)
            gain_y = max(obj_S2 - obj_S2_remove, 0) # 收益：剔除 k 后目标函数下降了多少
            
            # 计算保留 k 的概率
            if gain_x == 0 and gain_y == 0:
                prob_keep = 1.0 # 如果都没收益，保守起见保留在集合中
            else:
                prob_keep = gain_x / (gain_x + gain_y)
                
            # 依概率更新集合
            if np.random.rand() <= prob_keep:
                S1.add(k)
            else:
                S2.remove(k)
                
        # S1 和 S2 最终重合，返回列表形式的选中客户端 ID
        return list(S1)

    def update_queues(self, selected_clients, energy_profiles):
        """
        在每一轮联邦学习聚合结束后调用，动态演进队列
        """
        for k in range(self.num_clients):
            # 指示变量：选中为 1，未选中为 0
            s_k = 1.0 if k in selected_clients else 0.0
            
            # 1. 更新公平性队列 Z (长期未选中则积压增大)
            self.Z[k] = max(0.0, self.Z[k] + self.beta - s_k)
            
            # 2. 更新能耗队列 q (单轮能耗超过阈值则积压增大)
            actual_energy = energy_profiles[k] if k in selected_clients else 0.0
            self.q[k] = max(0.0, self.q[k] + actual_energy - self.E_avg)