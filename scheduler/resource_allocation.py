# 文件路径: scheduler/resource_allocation.py

import numpy as np

def calculate_communication_latency(s_decision, n_u, B_bandwidth, p_u, h_u, sigma_sq, data_size_smashed, data_size_model):
    """
    计算客户端的通信延迟 (上行传输 smashed data 和子模型)
    基于论文公式 (7) 和 (8)
    
    参数:
    s_decision: 切分层级 (Cut layer)
    n_u: 分配给该客户端的带宽比例 (0 到 1)
    B_bandwidth: 系统总带宽 (Hz)
    p_u: 客户端发射功率
    h_u: 信道增益
    sigma_sq: 噪声功率
    data_size_smashed: 根据切分层 s 决定的 smashed data 大小
    data_size_model: 根据切分层 s 决定的客户端子模型大小
    """
    if n_u <= 0:
        return float('inf')
        
    # 香农公式计算传输速率
    r_u = n_u * B_bandwidth * np.log2(1 + (p_u * h_u) / sigma_sq)
    
    # 避免除零错误
    if r_u == 0:
         return float('inf')
         
    # 计算传输延迟
    t_comm = (data_size_smashed + data_size_model) / r_u
    return t_comm

def calculate_computation_latency(s_decision, f_u, workload_client):
    """
    计算客户端的本地计算延迟
    
    参数:
    s_decision: 切分层级
    f_u: 客户端 CPU 频率 (flops/s)
    workload_client: 根据切分层 s 决定的客户端本地计算负载 (FLOPs)
    """
    if f_u <= 0:
        return float('inf')
        
    t_comp = workload_client / f_u
    return t_comp

def optimal_bandwidth_allocation(M_subset, s_decision_dict, B_bandwidth, p_dict, h_dict, sigma_sq, data_size_smashed_dict, data_size_model_dict, f_dict, workload_client_dict, epsilon=1e-3):
    """
    Algorithm 1: Optimal Bandwidth Allocation using Bisection (基于二分法的最优带宽分配)
    
    参数:
    M_subset: 当前选中的客户端列表 (list of IDs)
    s_decision_dict: 字典, 记录每个客户端当前的切分层级 s
    其他参数为包含每个客户端特定硬件/网络特征的字典
    """
    if not M_subset:
        return 0, {}

    # 1. 估算目标延迟 T 的下界和上界
    T_low = 0.0
    T_high = 0.0
    
    for u in M_subset:
        # 假设给予极端最小带宽 (非常小的比例) 计算理论最大可能延迟作为上界
        min_n = 1e-5
        t_comm_max = calculate_communication_latency(s_decision_dict[u], min_n, B_bandwidth, p_dict[u], h_dict[u], sigma_sq, data_size_smashed_dict[u], data_size_model_dict[u])
        t_comp = calculate_computation_latency(s_decision_dict[u], f_dict[u], workload_client_dict[u])
        T_high = max(T_high, t_comm_max + t_comp)
        
    # 2. 二分查找寻找最优全局延迟 T*
    T_mid = T_low
    n_policy = {}
    
    while (T_high - T_low) > epsilon:
        T_mid = (T_low + T_high) / 2.0
        sum_n = 0.0
        possible = True
        
        for u in M_subset:
            t_comp = calculate_computation_latency(s_decision_dict[u], f_dict[u], workload_client_dict[u])
            
            # 如果本地计算延迟已经超过了设定的目标全局延迟 T_mid，说明这个 T_mid 设得太严苛了
            if t_comp >= T_mid:
                possible = False
                break
                
            # 计算为了达到剩余的时间 (T_mid - t_comp) 用于通信，该客户端需要的带宽比例 n_u
            # 反推香农公式: T_comm = Data / (n_u * B * log2(1 + SNR))
            # 所以 n_u = Data / (T_comm * B * log2(1 + SNR))
            required_t_comm = T_mid - t_comp
            rate_factor = B_bandwidth * np.log2(1 + (p_dict[u] * h_dict[u]) / sigma_sq)
            
            n_u = (data_size_smashed_dict[u] + data_size_model_dict[u]) / (required_t_comm * rate_factor)
            
            n_policy[u] = n_u
            sum_n += n_u
            
        if not possible or sum_n > 1.0:
            # T_mid 太小，导致部分设备计算超时，或者需要的总带宽超过 100%
            T_low = T_mid
        elif sum_n < 1.0 - epsilon:
            # 还有带宽盈余，目标延迟还可以进一步压缩
            T_high = T_mid
        else:
            # 刚好分完
            break
            
    # 确保 n_policy 包含所有选中设备的带宽分配 (正常结束循环时，最后一次计算的 n_policy 就是近似最优的)
    for u in M_subset:
        if u not in n_policy:
             n_policy[u] = 1.0 / len(M_subset) # 后备兜底方案
             
    return T_mid, n_policy

def optimal_model_splitting(M_subset, n_policy_dict, S_options, B_bandwidth, p_dict, h_dict, sigma_sq, data_size_smashed_fn, data_size_model_fn, f_dict, workload_client_fn):
    """
    Algorithm 2: Optimal Model Splitting (最优模型拆分)
    遍历所有可能的切分点 s，为每个客户端独立选择总延迟最小的策略。
    
    参数注意: data_size_smashed_fn, data_size_model_fn, workload_client_fn 需要是函数，
    能够接收切分层 s 作为输入并返回对应的数据大小或计算负载。
    """
    s_optimal = {}
    
    for u in M_subset:
        best_s = None
        min_expected_latency = float('inf')
        n_u = n_policy_dict.get(u, 1.0 / len(M_subset)) # 如果没有前置策略，默认平分
        
        for s in S_options:
            # 根据切分点 s 获取当前客户端的数据和计算量
            data_size_smashed = data_size_smashed_fn(s)
            data_size_model = data_size_model_fn(s)
            workload_client = workload_client_fn(s)
            
            # 计算总延迟
            t_comm = calculate_communication_latency(s, n_u, B_bandwidth, p_dict[u], h_dict[u], sigma_sq, data_size_smashed, data_size_model)
            t_comp = calculate_computation_latency(s, f_dict[u], workload_client)
            
            total_latency = t_comm + t_comp
            
            if total_latency < min_expected_latency:
                min_expected_latency = total_latency
                best_s = s
                
        s_optimal[u] = best_s
        
    return s_optimal