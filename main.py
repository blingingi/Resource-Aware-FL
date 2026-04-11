import torch
import numpy as np
from scheduler.continuous_lyapunov_scheduler import ContinuousLyapunovScheduler

def main():
    # 参数设置
    NUM_CLIENTS = 100
    NUM_CLASSES = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化你的连续调度器
    scheduler = ContinuousLyapunovScheduler(
        num_clients=NUM_CLIENTS, 
        num_classes=NUM_CLASSES, 
        V_penalty=50.0,      # 需要调优
        beta_target=0.2,     # 长期期望参与率
        max_energy=5.0,      # 单轮能耗限额
        device=device
    )
    
    # 假设这是你每个客户端的固定数据 (需转为 Tensor 放进 GPU/CPU)
    # D: 数据量, A: 标签分布占比, E: 单轮总能耗评估
    D_tensor = torch.tensor(client_data_sizes, dtype=torch.float32, device=device)
    A_tensor = torch.tensor(client_label_distributions, dtype=torch.float32, device=device)
    E_tensor = torch.tensor(energy_profiles, dtype=torch.float32, device=device)
    
    for round_idx in range(50):
        # 1. 求解最优参与比例向量 w*
        w_optimal = scheduler.optimize_participation_rates(D_tensor, A_tensor, E_tensor)
        
        # --------- 【至关重要的一步：在 PyTorch 的 FL 训练中应用 w】 ---------
        # 你不能仅仅解出 w 就完事了，必须让它切实控制训练过程！
        for k in range(NUM_CLIENTS):
            participation_ratio = w_optimal[k]
            
            if participation_ratio < 0.05:
                # 剔除噪音：如果参与比例极低（例如不到 5%），直接不分配算力，跳过该客户端
                continue 
                
            # 【应用方式 1：控制采样规模】
            # 计算该客户端本轮需要抽取多少样本
            # local_batch_size = int(client_data_sizes[k] * participation_ratio)
            # train_local_model(client_id=k, batch_size=local_batch_size)
            
            # 【应用方式 2：控制聚合权重】
            # 在 FedAvg 的聚合阶段，用 w_k 取代原来的平均权重
            # global_model += local_model_update * participation_ratio
        
        # ---------------------------------------------------------------------

        # 3. 更新虚拟队列状态
        scheduler.update_queues(w_optimal, energy_profiles)
        
        print(f"Round {round_idx+1} | 平均参与率: {np.mean(w_optimal):.4f} | 队列Z积压: {torch.mean(scheduler.Z).item():.2f}")