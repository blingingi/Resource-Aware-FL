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
from utils.sampling import mnist_iid, mnist_noniid, mnist_dirichlet, cifar_iid, cifar_noniid, cifar_dirichlet
from utils.options import args_parser
from models.Update import LocalUpdate
# 确保 Nets 里的 CNNCifar 已经是你修改过的 3层宽体版本
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img
from utils.resource import ResourceManager
from utils.sim_div import calculate_diversity, calculate_similarity_score

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
            try:
                from utils.sampling import mnist_dirichlet
                dict_users = mnist_dirichlet(dataset_train, args.num_users, args.alpha, args.local_bs)
            except ImportError:
                dict_users = cifar_dirichlet(dataset_train, args.num_users, args.alpha, args.local_bs)
        else:
            exit('Error: unrecognized partition strategy for MNIST')
            
    elif args.dataset == 'cifar':
        trans_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),     # 随机裁剪（标准CIFAR增强）
            transforms.RandomHorizontalFlip(),        # 随机水平翻转
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        trans_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_train)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_test)
        
        if args.partition == 'iid':
            dict_users = cifar_iid(dataset_train, args.num_users)
        elif args.partition == 'shard':
            dict_users = cifar_noniid(dataset_train, args.num_users)
        elif args.partition == 'dirichlet':
            dict_users = cifar_dirichlet(dataset_train, args.num_users, args.alpha, args.local_bs)
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

    # ================= [策略初始化] =================
    print("正在计算所有客户端的数据多样性 (Diversity)...")
    div_scores = calculate_diversity(dataset_train, dict_users, args.num_classes)
    div_min, div_max = div_scores.min(), div_scores.max()
    div_norm = (div_scores - div_min) / (div_max - div_min + 1e-8)
    
    sim_scores = np.ones(args.num_users) * 10.0 # 初始给个高分
    
    resource_mgr = ResourceManager(args.num_users)
    loss_train = []
    acc_test_history = [] 

    if args.all_clients: 
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]

    # ================= [联邦学习主循环] =================
    for iter in range(args.epochs):
        loss_locals = []
        len_locals = [] 
        
        if not args.all_clients:
            w_locals = []
            
        m = max(int(args.frac * args.num_users), 1)

        # ================= [Selection Strategy (OURS - 贪心背包 + 弹性降级)] =================
        sim_min, sim_max = sim_scores.min(), sim_scores.max()
        sim_norm = (sim_scores - sim_min) / (sim_max - sim_min + 1e-8)
        
        # 使用稳健的权重比例：多样性 0.5，相似性 0.5 (在 Dir=0.1 下防止极端偏向)
        alpha_div_dynamic = 0.5  
        alpha_sim_dynamic = 0.5
        data_utility_scores = alpha_sim_dynamic * sim_norm + alpha_div_dynamic * (1 - div_norm)
        
        if not hasattr(resource_mgr, 'wait_times'):
            resource_mgr.wait_times = np.zeros(args.num_users)
            
        valid_candidates = []
        roi_scores_dict = {}
        client_workload_ratio = {} # 【新增】：记录每个客户端本轮被允许完成的工作量比例
        
        # 1. 计算弹性比例与性价比 (ROI)
        for i in range(args.num_users):
            t, e = resource_mgr.calculate_cost(i, len(dict_users[i]))
            
            # 【核心突破：弹性降级 Partial Work】
            if t > args.max_time:
                # 超时了？不踢人！按比例缩减工作量、耗时和能耗
                ratio = args.max_time / t
                t_adjusted = args.max_time
                e_adjusted = e * ratio
            else:
                ratio = 1.0
                t_adjusted = t
                e_adjusted = e
                
            client_workload_ratio[i] = ratio
            valid_candidates.append(i) # 所有人都能进候选池，保留所有特征！
            
            # 线性温和补偿（防饿死机制），绝对封顶 2.5 倍
            if resource_mgr.wait_times[i] > 10:
                raw_bonus = 1.0 + 0.1 * (resource_mgr.wait_times[i] - 10)
                wait_bonus = min(raw_bonus, 2.5) 
            else:
                wait_bonus = 1.0
            
            # 计算终极性价比 (ROI)：由于训练数据打了折扣，Utility也要打折扣！
            roi = ((data_utility_scores[i] * ratio) / (e_adjusted + 1e-5)) * wait_bonus
            roi_scores_dict[i] = roi

        # 2. 贪心排序：按性价比 (ROI) 从高到低降序排列
        valid_candidates.sort(key=lambda x: roi_scores_dict[x], reverse=True)
        
        # 3. 能耗约束背包：从最高性价比开始装箱
        selected_users = []
        current_energy_sum = 0.0
        current_max_time = 0.0
        
        for client_id in valid_candidates:
            if len(selected_users) >= m:
                break # 已经选满了 m 个人
                
            # 拿到打折后的真实能耗和时延
            _, original_e = resource_mgr.calculate_cost(client_id, len(dict_users[client_id]))
            ratio = client_workload_ratio[client_id]
            e_adjusted = original_e * ratio
            
            # 检查打折后的能耗预算
            if current_energy_sum + e_adjusted <= args.max_energy:
                selected_users.append(client_id)
                current_energy_sum += e_adjusted
                
        # 兜底机制：如果能耗极度紧张，导致连一个人都装不下
        if len(selected_users) == 0:
             print("⚠️ 能耗预算极其严苛，触发兜底！寻找打折后能耗最小的客户端。")
             best_fallback = min(valid_candidates, key=lambda x: resource_mgr.calculate_cost(x, len(dict_users[x]))[1] * client_workload_ratio[x])
             selected_users.append(best_fallback)
             current_energy_sum += resource_mgr.calculate_cost(best_fallback, len(dict_users[best_fallback]))[1] * client_workload_ratio[best_fallback]

        # 统计本轮真实最大时延
        for su in selected_users:
            t, _ = resource_mgr.calculate_cost(su, len(dict_users[su]))
            t_adj = t * client_workload_ratio[su] if t > args.max_time else t
            current_max_time = max(current_max_time, t_adj)

        # 4. 更新等待陈旧度
        resource_mgr.wait_times += 1 
        for su in selected_users:
            resource_mgr.wait_times[su] = 0 
            
        resource_mgr.update_selection(selected_users)
        
        print(f"Round {iter:3d} | 选中 {len(selected_users):2d} 人 | "
              f"时延: {current_max_time:.2f}s | 耗能: {current_energy_sum:.2f}J")
        
        # ================= [Local Training] =================
        for idx in selected_users:
            ratio = client_workload_ratio[idx]
            # 【截断数据】：根据 ratio 决定给这个客户端真实派发多少数据
            actual_sample_num = max(int(len(dict_users[idx]) * ratio), 1)
            actual_idxs = dict_users[idx][:actual_sample_num] # 只取前 N 个样本
            
            # 这里传进去的是截断后的数据索引 actual_idxs！
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=actual_idxs)
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            
            new_sim = calculate_similarity_score(w_glob, w)
            sim_scores[idx] = new_sim
            
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
            
            # 【极其重要】：聚合权重必须按照它“实际训练的样本数”来算！
            len_locals.append(actual_sample_num)
            
        # 将权重传递给加权聚合函数
        w_glob = FedAvg(w_locals, len_locals)
        net_glob.load_state_dict(w_glob)

        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)

        # ================= [Evaluation] =================
        net_glob.eval()
        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        acc_test_history.append(acc_test)
        print('  ==> Average loss {:.3f}, Test Acc {:.2f}%'.format(loss_avg, acc_test))
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
        args.max_time,   
        args.max_energy, 
        timestamp
    )

    save_dir = './save'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, '{}_acc.npy'.format(file_id))
    np.save(save_path, acc_test_history)
    
    print(f"🎉 实验结束！数据已绝对安全地保存到: {save_path}")