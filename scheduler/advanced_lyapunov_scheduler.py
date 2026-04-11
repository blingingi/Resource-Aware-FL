# 文件路径: scheduler/advanced_lyapunov_scheduler.py

import torch
import numpy as np

class AdvancedLyapunovScheduler:
    def __init__(self, num_clients, num_classes, V_skewness, V_latency, max_energy, device='cpu'):
        self.num_clients = num_clients
        self.num_classes = num_classes
        self.V1 = V_skewness        # 数据偏斜度惩罚权重
        self.V2 = V_latency         # 全局延迟惩罚权重
        self.E_avg = max_energy     # 单轮能耗限额
        self.device = device
        
        # 核心状态跟踪 (时间维度)
        self.A = torch.ones(num_clients, device=self.device)   # 梯度年龄 (Age of Gradient)，初始为 1
        self.q = torch.zeros(num_clients, device=self.device)  # 能耗虚拟队列

    def calculate_continuous_skewness(self, w, D_tensor, A_tensor):
        """计算空间偏斜度 G(w)，w 为 Batch Size 比例"""
        weighted_D = w * D_tensor
        total_weighted_D = torch.sum(weighted_D) + 1e-9
        weighted_A = weighted_D.unsqueeze(1) * A_tensor
        actual_q_ratio = torch.sum(weighted_A, dim=0) / total_weighted_D
        ideal_q_ratio = 1.0 / self.num_classes
        skewness = torch.sum((actual_q_ratio - ideal_q_ratio) ** 2)
        return skewness

    def optimize_batch_size_ratio(self, D_tensor, A_tensor, comp_latency_base, comm_latency_fixed, energy_profiles, lr=0.01, num_steps=200):
        """
        利用 PGD 求解本轮最佳的 Batch Size 比例 w*
        comp_latency_base: 客户端处理其全部本地数据 (w=1.0) 时的计算延迟
        comm_latency_fixed: 固定的模型/砸碎数据传输延迟
        """
        # 初始化比例为 0.5
        w = torch.full((self.num_clients,), 0.5, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([w], lr=lr)
        
        for step in range(num_steps):
            optimizer.zero_grad()
            
            # 1. 偏斜度惩罚 (Training Loss 代理)
            skewness_penalty = self.V1 * self.calculate_continuous_skewness(w, D_tensor, A_tensor)
            
            # 2. 最大延迟惩罚 (Training Latency)
            # 实际计算延迟与 Batch Size 比例 w 成正比
            actual_latency = w * comp_latency_base + comm_latency_fixed
            # 系统整体延迟由最慢的那个设备决定
            latency_penalty = self.V2 * torch.max(actual_latency)
            
            # 3. 队列与梯度年龄惩罚
            # 能耗消耗正比于计算量 (w) + 通信基数
            actual_energy = w * energy_profiles['comp'] + energy_profiles['comm']
            # q * E - A * w: 能耗越高惩罚越大，梯度年龄越高（越陈旧）给予的奖励越大（促使 w 变大）
            lyapunov_penalty = torch.sum(self.q * actual_energy - self.A * w)
            
            # 联合损失函数
            loss = skewness_penalty + latency_penalty + lyapunov_penalty
            
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                w.clamp_(0.0, 1.0)
                
        return w.detach().cpu().numpy()

    def update_queues(self, w_optimal, energy_actual):
        """
        联邦聚合后更新梯度年龄 A 和能耗队列 q
        """
        w_tensor = torch.tensor(w_optimal, dtype=torch.float32, device=self.device)
        E_tensor = torch.tensor(energy_actual, dtype=torch.float32, device=self.device)
        
        # 1. 更新梯度年龄: 如果 w 接近 1 (全量 batch)，年龄重置为 1；否则年龄增长
        # 结合 Li et al. (2024) 的思想演化
        self.A = self.A * (1.0 - w_tensor) + 1.0
        
        # 2. 更新能耗队列
        self.q = torch.clamp(self.q + E_tensor - self.E_avg, min=0.0)