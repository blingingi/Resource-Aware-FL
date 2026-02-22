#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
import torch
import os
from torchvision import datasets, transforms
from torch import nn

# 引入项目依赖
from utils.sampling import mnist_iid, mnist_noniid, mnist_dirichlet,cifar_iid, cifar_noniid, cifar_dirichlet
from utils.options import args_parser
from models.Update import LocalUpdate
# 确保 Nets 里的 CNNCifar 已经是你修改过的 3层宽体版本
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img
from utils.resource import ResourceManager

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

if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

   # ================= [Load Dataset] =================
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        
        # 兼容不同的划分方式
        if args.partition == 'iid':
            dict_users = mnist_iid(dataset_train, args.num_users)
        elif args.partition == 'shard':
            dict_users = mnist_noniid(dataset_train, args.num_users)
        elif args.partition == 'dirichlet':
            # 如果你的 sampling.py 里有 mnist_dirichlet 就用它，没有的话直接用 cifar_dirichlet 处理 MNIST 标签也是一样的
            try:
                from utils.sampling import mnist_dirichlet
                dict_users = mnist_dirichlet(dataset_train, args.num_users, args.alpha)
            except ImportError:
                dict_users = cifar_dirichlet(dataset_train, args.num_users, args.alpha)
        else:
            exit('Error: unrecognized partition strategy for MNIST')
            
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        
        if args.partition == 'iid':
            dict_users = cifar_iid(dataset_train, args.num_users)
        elif args.partition == 'shard':
            dict_users = cifar_noniid(dataset_train, args.num_users)
        elif args.partition == 'dirichlet':
            dict_users = cifar_dirichlet(dataset_train, args.num_users, args.alpha)
        else:
            exit('Error: unrecognized partition strategy for CIFAR')
    else:
        exit('Error: unrecognized dataset')

    # ================= [Build Model] =================
    if args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    
    print(net_glob)
    net_glob.train()
    w_glob = net_glob.state_dict()

    # ================= [策略修正] =================
    # 1. Diversity
    div_scores = calculate_diversity(dataset_train, dict_users, args.num_classes)
    div_min, div_max = div_scores.min(), div_scores.max()
    div_norm = (div_scores - div_min) / (div_max - div_min + 1e-8)
    
    # 2. Similarity
    sim_scores = np.ones(args.num_users) * 10.0 # 初始给个高分
    
    # 3. 权重参数
    alpha_1 = 0.2  # Similarity 权重
    alpha_2 = 0.8  # Diversity 权重
    # ============================================
    resource_mgr = ResourceManager(args.num_users)
    loss_train = []
    acc_test_history = [] 

    if args.all_clients: 
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]

    for iter in range(args.epochs):
        loss_locals = []
        len_locals = [] # 【修复1】初始化当前轮次的数据量记录列表
        
        if not args.all_clients:
            w_locals = []
            
        m = max(int(args.frac * args.num_users), 1)

      # ================= [Selection Strategy (Random + Resource Constraints)] =================
        # 1. 彻底放弃数据价值打分，直接纯随机打乱所有客户端
        all_clients = list(range(args.num_users))
        np.random.shuffle(all_clients)
        
        # 2. 资源约束下的随机装箱 (Random Knapsack Selection)
        selected_users = []
        current_energy_sum = 0.0
        current_max_time = 0.0
        
        for client_id in all_clients:
            data_len = len(dict_users[client_id])
            t, e = resource_mgr.calculate_cost(client_id, data_len)
            
            potential_energy = current_energy_sum + e
            potential_time = max(current_max_time, t)
            
            # 严格检查红线：只要不超时、不超电，并且是随机碰上的，就直接拉入队伍
            if potential_energy <= args.max_energy and potential_time <= args.max_time:
                selected_users.append(client_id)
                current_energy_sum = potential_energy
                current_max_time = potential_time
                
            # 招满 m 个人即可停止
            if len(selected_users) >= m:
                break
                
        # 3. 学术兜底机制：防止参数设得太严导致本轮无人可用
        if len(selected_users) == 0:
            print("⚠️ 警告：本轮系统资源约束过于严苛，触发兜底！强行选择 1 个最高效节点。")
            times = [resource_mgr.calculate_cost(i, len(dict_users[i]))[0] for i in range(args.num_users)]
            fastest_client = np.argmin(times)
            selected_users = [fastest_client]
            current_max_time, current_energy_sum = resource_mgr.calculate_cost(fastest_client, len(dict_users[fastest_client]))
            
        # 记录被选中的节点
        resource_mgr.update_selection(selected_users)
        
        print(f"Round {iter} | 随机约束装箱 | 选中 {len(selected_users)} 人 | 时延: {current_max_time:.2f}s | 耗能: {current_energy_sum:.2f}J")
        # ========================================================
        # ========================================================================

        # 【修复完成】只保留一个循环，且严格遍历 selected_users
        for idx in selected_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            
            new_sim = calculate_similarity_score(w_glob, w)
            sim_scores[idx] = new_sim
            
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
            
            # 严格记录被选客户端的真实样本数
            len_locals.append(len(dict_users[idx]))
            
        # 将权重传递给加权聚合函数
        w_glob = FedAvg(w_locals, len_locals)
        net_glob.load_state_dict(w_glob)

        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)

        # Evaluation
        net_glob.eval()
        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        acc_test_history.append(acc_test)
        print('Round {:3d}, Average loss {:.3f}, Test Acc {:.2f}%'.format(iter, loss_avg, acc_test))
        net_glob.train()
        args.lr = args.lr * 0.99


    # ================= [绘图与保存结果] =================
    import datetime

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    script_name = os.path.basename(__file__).split('.')[0]
    
    file_id = 'fed_{}_{}_{}_alpha{}_ep{}_time{}_energy{}_{}'.format(
        script_name, 
        args.dataset, 
        args.partition, 
        args.alpha, 
        args.epochs, 
        args.max_time,   # 记录时间红线
        args.max_energy, # 记录能耗红线
        timestamp
    )

    # 【修复5】增加防崩溃目录检查
    save_dir = './save'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, '{}_acc.npy'.format(file_id))
    np.save(save_path, acc_test_history)
    
    print(f"🎉 实验结束！数据已绝对安全地保存到: {save_path}")
