import numpy as np
import torch
import torch
import torch.nn.functional as F
# ===================================================================
# 辅助函数：计算 KL 散度 (Diversity Metric)
# ===================================================================
def calculate_diversity(dataset, dict_users, num_classes):
    diversity_scores = []
    P_uniform = np.ones(num_classes) / num_classes 
    
    print("正在计算所有客户端的数据多样性 (Diversity)...")
    
    for idx in range(len(dict_users)):
        user_indices = dict_users[idx]
        if hasattr(dataset, 'targets'):
            labels = np.array(dataset.targets)[list(user_indices)]
        else:
            labels = dataset.train_labels.numpy()[list(user_indices)]
            
        label_counts = np.zeros(num_classes)
        for label in labels:
            label_counts[label] += 1
        
        P_client = (label_counts + 1e-5) / (sum(label_counts) + 1e-5 * num_classes)
        kl_div = np.sum(P_uniform * np.log(P_uniform / P_client))
        diversity_scores.append(kl_div)
        
    return np.array(diversity_scores)

# ===================================================================
# 辅助函数：计算相似性 (Similarity Metric)
# 只计算最后一层 (fc2) 的参数距离，防止距离过大导致分数归零
# ===================================================================
def calculate_similarity_score(w_global, w_local, k1=10, k2=0.01):
    diff_sum = 0
    target_layer = 'fc2' 
    
    layer_found = False
    for k in w_global.keys():
        if target_layer in k:
            diff_sum += torch.sum(torch.abs(w_global[k] - w_local[k])).item()
            layer_found = True
            
    if not layer_found:
        total_diff = 0
        total_params = 0
        for k in w_global.keys():
            total_diff += torch.sum(torch.abs(w_global[k] - w_local[k])).item()
            total_params += w_global[k].numel()
        diff_sum = total_diff
    
    rho = diff_sum
    sim = k1 * np.exp(-k2 * rho)
    return sim

def get_weight_difference(w_new, w_old):
    """计算模型权重的差值（更新量），并强制在 CPU 上计算以避免设备冲突"""
    diff = []
    for k in w_new.keys():
        # 强制将新老权重都拉到 CPU 上，转为 float 后再相减
        t_new = w_new[k].cpu().float()
        t_old = w_old[k].cpu().float()
        diff.append((t_new - t_old).view(-1).detach())
    return torch.cat(diff)

def compute_cosine_similarity(vec1, vec2):
    """计算两个一维张量的余弦相似度，强制在 CPU 运算"""
    # 强制拉到 CPU
    vec1 = vec1.cpu()
    vec2 = vec2.cpu()
    
    if vec1.norm() == 0 or vec2.norm() == 0:
        return 0.0
    return F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0)).item()
