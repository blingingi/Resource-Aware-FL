import torch
import numpy as np

# 导入我们刚刚写好的高级调度器
from scheduler.advanced_lyapunov_scheduler import AdvancedLyapunovScheduler

def generate_advanced_dummy_data(num_clients, num_classes, device):
    """生成阶段一所需的模拟物理数据和网络数据"""
    # 1. 模拟数据量 D_k: 每个客户端拥有 500 到 2000 个样本
    D_cpu = np.random.randint(500, 2000, size=num_clients)
    D_tensor = torch.tensor(D_cpu, dtype=torch.float32, device=device)
    
    # 2. 模拟 Non-IID 标签分布 A_k: 使用 Dirichlet 分布 (alpha=0.1 意味着高度偏斜)
    A_cpu = np.random.dirichlet(np.ones(num_classes) * 0.1, size=num_clients)
    A_tensor = torch.tensor(A_cpu, dtype=torch.float32, device=device)
    
    # 3. 模拟计算延迟基数 (处理全部数据的耗时, 单位: 秒): 假设在 0.5s 到 3.0s 之间
    comp_latency_base = torch.empty(num_clients, device=device).uniform_(0.5, 3.0)
    
    # 4. 模拟通信延迟固定值 (传输模型参数的耗时, 单位: 秒): 假设在 0.1s 到 1.0s 之间
    comm_latency_fixed = torch.empty(num_clients, device=device).uniform_(0.1, 1.0)
    
    # 5. 模拟能耗基数: 计算满载能耗(comp)和固定通信能耗(comm)
    energy_profiles = {
        'comp': torch.empty(num_clients, device=device).uniform_(2.0, 8.0),
        'comm': torch.empty(num_clients, device=device).uniform_(0.5, 2.0)
    }
    
    return D_tensor, A_tensor, comp_latency_base, comm_latency_fixed, energy_profiles, D_cpu

def main():
    print("=== 🚀 SFL 高级调度系统 | 阶段一测试启动 ===")
    
    # 实验超参数
    NUM_CLIENTS = 100
    NUM_CLASSES = 10
    GLOBAL_ROUNDS = 50
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"当前运行设备: {DEVICE}")
    
    # 初始化我们的包工头 (调度器)
    # 这里的 V1 和 V2 是你需要重点观察的惩罚权重
    scheduler = AdvancedLyapunovScheduler(
        num_clients=NUM_CLIENTS, 
        num_classes=NUM_CLASSES, 
        V_skewness=50.0,     # 控制数据平衡的权重
        V_latency=10.0,      # 控制最大延迟的权重
        max_energy=4.0,      # 目标单轮平均最大能耗
        device=DEVICE
    )
    
    # 生成假数据
    D_tensor, A_tensor, comp_latency_base, comm_latency_fixed, energy_profiles, D_cpu = \
        generate_advanced_dummy_data(NUM_CLIENTS, NUM_CLASSES, DEVICE)
    
    # 开始模拟联邦学习轮次
    for round_idx in range(GLOBAL_ROUNDS):
        # 1. 包工头下达任务：计算这一轮最佳的 Batch Size 比例 w*
        w_optimal = scheduler.optimize_batch_size_ratio(
            D_tensor, A_tensor, 
            comp_latency_base, comm_latency_fixed, 
            energy_profiles,
            lr=0.05, num_steps=100  # PGD 优化参数
        )
        
        # 2. 将计算结果转化为物理意义上的变量
        w_tensor = torch.tensor(w_optimal, dtype=torch.float32, device=DEVICE)
        
        # 计算本轮的实际整体延迟 (取决于最慢的那个设备)
        actual_latencies = w_tensor * comp_latency_base + comm_latency_fixed
        round_latency = torch.max(actual_latencies).item()
        
        # 计算本轮的实际总能耗和平均能耗
        actual_energies = w_tensor * energy_profiles['comp'] + energy_profiles['comm']
        avg_energy = torch.mean(actual_energies).item()
        
        # 过滤掉真正参与的设备 (比例大于 5% 才算有效参与)
        active_clients = [i for i, w in enumerate(w_optimal) if w > 0.05]
        
        # 3. 打印本轮统计信息
        print(f"\n[Round {round_idx+1:02d}] 参与设备数: {len(active_clients):03d}/100 | 最大延迟: {round_latency:.2f}s | 平均能耗: {avg_energy:.2f}")
        
        # 挑选前 3 个设备看看它们具体被分配了多少数据量 (Batch Size)
        sample_info = []
        for i in range(3):
            bs = max(1, int(w_optimal[i] * D_cpu[i]))
            sample_info.append(f"设备{i}({w_optimal[i]:.0%}=>{bs}个样本)")
        print(f"  👉 抽样分配: {', '.join(sample_info)}")
        
        # 4. 包工头记账：演进系统的年龄与能耗队列
        scheduler.update_queues(w_optimal, actual_energies.detach().cpu().numpy())
        
        # 打印队列状态：A(年龄) 越小越新鲜，q(能耗积压) 越小越健康
        avg_Age = torch.mean(scheduler.A).item()
        avg_q = torch.mean(scheduler.q).item()
        print(f"  📊 系统状态: 平均梯度年龄(A) = {avg_Age:.2f} | 平均能耗积压(q) = {avg_q:.2f}")

if __name__ == "__main__":
    main()