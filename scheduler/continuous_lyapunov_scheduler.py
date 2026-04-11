# 文件路径: scheduler/continuous_lyapunov_scheduler.py

import torch

class ContinuousLyapunovScheduler:
    def __init__(self, num_clients, num_classes, V_penalty, beta_target, max_energy, device='cpu'):
        self.num_clients = num_clients
        self.num_classes = num_classes
        self.V = V_penalty          # 数据平衡权值 V
        self.beta = beta_target     # 目标长期参与率
        self.E_avg = max_energy     # 单轮最大平均能耗
        self.device = device
        
        # 将虚拟队列设为 PyTorch Tensor 以便加速运算
        self.Z = torch.zeros(num_clients, device=self.device)  # 公平性队列
        self.q = torch.zeros(num_clients, device=self.device)  # 能耗队列

    def calculate_continuous_skewness(self, w, D_tensor, A_tensor):
        """
        计算连续偏斜度 G(w)
        w: 客户端的参与比例向量, shape [K]
        D_tensor: 数据量大小, shape [K]
        A_tensor: 标签分布比例, shape [K, Q]
        """
        # 计算有效参与的数据量 (w_k * D_k)
        weighted_D = w * D_tensor  # shape [K]
        
        # 分母：总聚合数据量 (加入 1e-9 防止除零崩溃)
        total_weighted_D = torch.sum(weighted_D) + 1e-9
        
        # 分子：计算每个类别 q 的有效聚合数量
        # w_k * D_k * a_{k,q}
        weighted_A = weighted_D.unsqueeze(1) * A_tensor  # shape [K, Q]
        
        # 当前全局各类别占比
        actual_q_ratio = torch.sum(weighted_A, dim=0) / total_weighted_D  # shape [Q]
        
        # 理想状态下的类别占比
        ideal_q_ratio = 1.0 / self.num_classes
        
        # 计算 L2 差异
        skewness = torch.sum((actual_q_ratio - ideal_q_ratio) ** 2)
        return skewness

    def optimize_participation_rates(self, D_tensor, A_tensor, E_tensor, lr=0.01, num_steps=200):
        """
        使用投影梯度下降 (PGD) 寻找当前轮次最优的参与比例 w*
        """
        # 初始化 w，均匀设为 0.5 作为起点
        w = torch.full((self.num_clients,), 0.5, device=self.device, requires_grad=True)
        
        # 使用 Adam 优化器加速收敛
        optimizer = torch.optim.Adam([w], lr=lr)
        
        for step in range(num_steps):
            optimizer.zero_grad()
            
            # 1. 计算空间维度惩罚 (偏斜度)
            g_penalty = self.V * self.calculate_continuous_skewness(w, D_tensor, A_tensor)
            
            # 2. 计算时间维度惩罚 (队列漂移)
            # sum( q_k * w_k * e_k - Z_k * w_k )
            queue_penalty = torch.sum(self.q * w * E_tensor - self.Z * w)
            
            # 联合目标函数
            loss = g_penalty + queue_penalty
            
            # 反向传播计算梯度
            loss.backward()
            optimizer.step()
            
            # 投影步骤 (Projection)：将 w 强行裁剪回合法的 [0, 1] 区间
            with torch.no_grad():
                w.clamp_(0.0, 1.0)
                
        # 返回优化后的 numpy 数组形式
        return w.detach().cpu().numpy()

    def update_queues(self, w_optimal, E_tensor_cpu):
        """
        演进虚拟队列
        w_optimal: PGD 解出的最佳参与比例 (numpy array)
        E_tensor_cpu: 设备能耗 (numpy array)
        """
        w_tensor = torch.tensor(w_optimal, dtype=torch.float32, device=self.device)
        E_tensor = torch.tensor(E_tensor_cpu, dtype=torch.float32, device=self.device)
        
        # 1. 更新公平性队列 Z_{t+1} = max(0, Z_t + \beta - w_k)
        self.Z = torch.clamp(self.Z + self.beta - w_tensor, min=0.0)
        
        # 2. 更新能耗队列 q_{t+1} = max(0, q_t + w_k * e_k - E_avg)
        self.q = torch.clamp(self.q + w_tensor * E_tensor - self.E_avg, min=0.0)