#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import torch.nn.functional as F
import os
import datetime

# 引入 cifar_noniid
from utils.sampling import mnist_iid, mnist_noniid, mnist_dirichlet, cifar_iid, cifar_noniid, cifar_dirichlet
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img

# 导入你的资源管理器和计算工具
from utils.resource import ResourceManager
from utils.sim_div import get_weight_difference, compute_cosine_similarity
from utils.seed import set_seed

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    set_seed(42)
    
    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        if args.partition == 'iid':
            dict_users = mnist_iid(dataset_train, args.num_users)
        elif args.partition == 'shard':
            dict_users = mnist_noniid(dataset_train, args.num_users)
        elif args.partition == 'dirichlet':
            dict_users = mnist_dirichlet(dataset_train, args.num_users, args.alpha)
        else:
            exit('Error: unrecognized partition strategy')
            
    elif args.dataset == 'cifar':
        # 移除 RandomCrop 和 RandomHorizontalFlip，仅保留基础预处理
        trans_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        trans_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_train)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_test)
        
        if args.partition=='iid':
            dict_users = cifar_iid(dataset_train, args.num_users)
        elif args.partition=='shard':
            dict_users = cifar_noniid(dataset_train, args.num_users)
        elif args.partition == 'dirichlet':
            dict_users = cifar_dirichlet(dataset_train, args.num_users, args.alpha, args.local_bs)
        else:
            exit('Error: unrecognized partition strategy')
    else:
        exit('Error: unrecognized dataset')

    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
        
    print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    acc_test_history = [] 

    # === 初始化资源管理器 ===
    resource_mgr = ResourceManager(args.num_users, dict_users, local_ep=args.local_ep, limit_ratio=0.8)
    
    # === [核心创新] 初始化全局更新方向与历史梯度缓存 ===
    flat_w_glob = get_weight_difference(w_glob, w_glob) 
    global_update_dir = torch.zeros_like(flat_w_glob).cpu()
    historical_updates = {i: torch.zeros_like(flat_w_glob).cpu() for i in range(args.num_users)}

    # === [李雅普诺夫核心] 初始化公平性虚拟队列 ===
    m_expected = max(int(args.frac * args.num_users), 1)
    rho = (m_expected / float(args.num_users)) * 0.8  # 长期公平性阈值
    Z_queues = {i: 0.0 for i in range(args.num_users)}

    # === [关键参数] ===
    V = 5.0  # 静态适中常数，依靠 Z_queues 进行后期的完美制衡
    weight_sim = 0.8  # 对应你原代码的 alpha
    weight_div = 0.2  # 对应你原代码的 beta

    for iter in range(args.epochs):
        
        m = max(int(args.frac * args.num_users), 1)
        pool_size = min(m * 2, args.num_users) 
        candidate_idxs = np.random.choice(range(args.num_users), pool_size, replace=False)

        print(f"\n--- Round {iter} ---")
        
        # ==========================================================
        # 终极 Ours 挑选阶段：Utility (效用) + 李雅普诺夫虚拟队列 (公平与资源)
        # ==========================================================
        selected_idxs = []
        remaining_candidates = list(candidate_idxs)
        
        # 提前计算 SIM 分数，节省内部循环开销
        sim_scores = {}
        if iter > 0 and global_update_dir.norm() > 0:
            for idx in candidate_idxs:
                sim = compute_cosine_similarity(historical_updates[idx], global_update_dir)
                sim_scores[idx] = sim if not torch.isnan(torch.tensor(sim)) else 0.0
                
        for _ in range(m):
            best_score = -float('inf')
            best_idx = -1
            
            for idx in remaining_candidates:
                # 1. 获取资源惩罚项
                resource_penalty = resource_mgr.get_penalty(idx)
                
                # 2. 计算你的核心创新 Utility (SIM - weight_div * DIV)
                utility = 0.0
                if iter > 0 and global_update_dir.norm() > 0:
                    utility = weight_sim * sim_scores[idx]
                    div_penalty = 0.0
                    
                    if len(selected_idxs) > 0:
                        for selected_idx in selected_idxs:
                            sim_with_selected = compute_cosine_similarity(
                                historical_updates[idx], historical_updates[selected_idx]
                            )
                            if not torch.isnan(torch.tensor(sim_with_selected)):
                                div_penalty += sim_with_selected
                        div_penalty = div_penalty / len(selected_idxs)
                    
                    utility = utility - weight_div * div_penalty
                
                # 3. [终极得分公式：Ours (Utility) + Lya (Resource + Fairness)]
                # V_t 不再需要人工干预衰减，因为 Z_queues[idx] 的激增会强行打破任何长期的过度贪婪！
                final_score = (V * utility) - resource_penalty + Z_queues[idx]
                
                if final_score > best_score:
                    best_score = final_score
                    best_idx = idx
            
            selected_idxs.append(best_idx)
            remaining_candidates.remove(best_idx)
            
        print(f"Selected clients: {selected_idxs}")
        
        # ==========================================================
        # [李雅普诺夫核心] 严格更新虚拟队列 (替代了你之前的丑陋打折逻辑)
        # ==========================================================
        for idx in range(args.num_users):
            x_i = 1.0 if idx in selected_idxs else 0.0
            # 没有被选的节点，其 Z_queues 会一直增长，总有一刻它的 Z 会大到超越 Utility 强行被选中！
            # 被选中的瞬间，节点参与训练，历史梯度自动被刷新成最新的！
            Z_queues[idx] = max(0.0, Z_queues[idx] - x_i + rho)

        # ==========================================================
        # 结算阶段与消耗阶段
        # ==========================================================
        avg_q_t, avg_q_e = resource_mgr.update_queues_and_counts(selected_idxs)

        w_locals = []
        loss_locals = []
        len_locals = []
        
        for idx in selected_idxs:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(
                net=copy.deepcopy(net_glob).to(args.device),
                global_net=net_glob
            )
            
            # === [关键步骤] 把最新梯度存入缓存，供下一轮挑选时使用 ===
            update_vec = get_weight_difference(w, w_glob).cpu()
            historical_updates[idx] = update_vec
            
            w_locals.append({k: v.cpu() for k, v in w.items()})
            loss_locals.append(loss)
            len_locals.append(len(dict_users[idx]))
            
            torch.cuda.empty_cache()

        # ==========================================================
        # 模型聚合与全局更新
        # ==========================================================
        w_glob_new = FedAvg(w_locals, len_locals)
        
        current_global_update = get_weight_difference(w_glob_new, w_glob)
        global_update_dir = 0.9 * global_update_dir + 0.1 * current_global_update

        w_glob = w_glob_new
        net_glob.load_state_dict(w_glob)

        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)

        net_glob.eval() 
        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        acc_test_history.append(acc_test)
        
        print('Round {:3d}, Loss {:.3f}, Acc {:.2f}%, Avg Q_E: {:.3f}, Avg Q_T: {:.3f}'.format(
            iter, loss_avg, acc_test, avg_q_e, avg_q_t))
        net_glob.train()
        args.lr = args.lr * 0.99

    # ================= [绘图与保存结果] =================
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    script_name = "ours"
    
    file_id = 'fed_{}_{}_{}_alpha{}_ep{}_{}'.format(
        script_name, args.dataset, args.partition, args.alpha, args.epochs, timestamp)

    save_dir = './save'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, '{}_acc.npy'.format(file_id))
    np.save(save_path, acc_test_history)
    
    print(f"🎉 实验结束！数据已绝对安全地保存到: {save_path}")