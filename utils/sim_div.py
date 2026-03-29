import numpy as np
import torch
import torch.nn.functional as F

# ===================================================================
# 模块 1: 静态多样性计算 (Diversity Metric - 基于 KL 散度)
# 用途: 用于在训练前评估各个客户端本地数据标签分布与全局均匀分布的差异
# ===================================================================
def calculate_diversity(dataset, dict_users, num_classes):
    diversity_scores = []
    P_uniform = np.ones(num_classes) / num_classes 
    
    print("=> 正在计算所有客户端的静态数据多样性 (KL Divergence)...")
    
    for idx in range(len(dict_users)):
        user_indices = dict_users[idx]
        # 兼容不同版本 torchvision 的 targets 获取方式
        if hasattr(dataset, 'targets'):
            labels = np.array(dataset.targets)[list(user_indices)]
        else:
            labels = dataset.train_labels.numpy()[list(user_indices)]
            
        label_counts = np.zeros(num_classes)
        for label in labels:
            label_counts[label] += 1
        
        # 加上平滑项 epsilon=1e-5，防止除以 0 或 log(0)
        P_client = (label_counts + 1e-5) / (sum(label_counts) + 1e-5 * num_classes)
        kl_div = np.sum(P_uniform * np.log(P_uniform / P_client))
        diversity_scores.append(kl_div)
        
    return np.array(diversity_scores)


# ===================================================================
# 模块 2: 动态相似度与散度计算 (SIM / DIV Metric - 基于梯度余弦方向)
# 用途: 提取模型更新向量，并通过夹角余弦判断本地更新是否与全局收敛方向一致
# ===================================================================
def get_weight_difference(w_new, w_old):
    """
    计算模型权重的差值（即本轮的伪梯度/更新向量）
    将其展平为一维向量，并强制在 CPU 上计算以避免爆显存
    """
    diff = []
    for k in w_new.keys():
        # 忽略批归一化层中的追踪批次统计（非学习参数）
        if 'num_batches_tracked' in k:
            continue
        
        # 强制将新老权重拉到 CPU，转为 float 后再相减，避免 GPU 显存堆积
        t_new = w_new[k].cpu().float()
        t_old = w_old[k].cpu().float()
        diff.append((t_new - t_old).view(-1).detach())
        
    return torch.cat(diff)

def compute_cosine_similarity(vec1, vec2):
    """
    计算两个一维张量的余弦相似度 (Cosine Similarity)
    范围在 [-1, 1] 之间。
    1 表示方向完全一致 (有益)，-1 表示方向完全相反 (有毒/漂移)
    """
    vec1 = vec1.cpu()
    vec2 = vec2.cpu()
    
    # 防止因为模型没有更新（零向量）导致分母为 0 出现 NaN
    if vec1.norm() == 0 or vec2.norm() == 0:
        return 0.0
        
    # 余弦相似度计算
    sim = F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0)).item()
    return sim