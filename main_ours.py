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
            print("=> 正在使用 IID 均匀划分 MNIST 数据...")
            dict_users = mnist_iid(dataset_train, args.num_users)
        elif args.partition == 'shard':
            print("=> 正在使用 Shard 分片划分 MNIST 数据...")
            dict_users = mnist_noniid(dataset_train, args.num_users)
        elif args.partition == 'dirichlet':
            print(f"=> 正在使用 Dirichlet 划分 MNIST 数据, alpha={args.alpha}...")
            dict_users = mnist_dirichlet(dataset_train, args.num_users, args.alpha)
        else:
            exit('Error: unrecognized partition strategy')
            
    elif args.dataset == 'cifar':
        trans_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
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
            print("=> 正在使用 IID 均匀划分数据...")
            dict_users = cifar_iid(dataset_train, args.num_users)
        elif args.partition=='shard':
            print("=> 正在使用 Shard 分片划分数据...")
            dict_users = cifar_noniid(dataset_train, args.num_users)
        elif args.partition == 'dirichlet':
            print(f"=> 正在使用 Dirichlet 划分数据, alpha={args.alpha}...")
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

    # === [核心重构] 初始化资源管理器与李雅普诺夫队列 ===
    # 必须传入 dict_users，资源管理器才能根据真实数据量静态计算所有节点的开销
    resource_mgr = ResourceManager(args.num_users, dict_users, local_ep=args.local_ep, limit_ratio=0.8)
    
    # 权衡参数 V：越大越看重模型准确率 (SIM/DIV)，越小越看重资源限制。
    # 建议值：5.0~50.0 视具体分数数量级而定
    V = 5.0  

    # === 初始化全局更新方向参考 ===
    flat_w_glob = get_weight_difference(w_glob, w_glob) 
    global_update_dir = torch.zeros_like(flat_w_glob).cpu()
    
    # === [新增] 初始化历史梯度缓存 ===
    historical_updates = {i: torch.zeros_like(flat_w_glob).cpu() for i in range(args.num_users)}

    # === [核心修正] 初始化被选次数记录器，强制探索未被选择的节点 ===
    selection_counts = {i: 0 for i in range(args.num_users)}
    gamma = 2.0  # 频率惩罚系数。若依然出现死磕某几个节点的情况，可调大至 5.0 或 10.0

    alpha = 0.8  # SIM的权重
    beta = 0.2   # DIV的权重

    for iter in range(args.epochs):
        
        m = max(int(args.frac * args.num_users), 1)
        pool_size = min(m * 2, args.num_users) 
        
        candidate_idxs = np.random.choice(range(args.num_users), pool_size, replace=False)

        print(f"\n--- Round {iter} ---")
        
        selected_idxs = []
        
        if iter == 0 or global_update_dir.norm() == 0:
            selected_idxs = np.random.choice(candidate_idxs, m, replace=False).tolist()
        else:
            sim_scores = {}
            for idx in candidate_idxs:
                sim = compute_cosine_similarity(historical_updates[idx], global_update_dir)
                sim_scores[idx] = sim if not torch.isnan(torch.tensor(sim)) else 0.0
            
            remaining_candidates = list(candidate_idxs)
            
            for _ in range(m):
                best_score = -float('inf')
                best_idx = -1
                
                for idx in remaining_candidates:
                    # (1) 准确率收益项 (Utility)
                    utility = alpha * sim_scores[idx]
                    div_penalty = 0.0
                    
                    if len(selected_idxs) > 0:
                        for selected_idx in selected_idxs:
                            sim_with_selected = compute_cosine_similarity(
                                historical_updates[idx], historical_updates[selected_idx]
                            )
                            if not torch.isnan(torch.tensor(sim_with_selected)):
                                div_penalty += sim_with_selected
                        
                        div_penalty = div_penalty / len(selected_idxs)
                    
                    utility = utility - beta * div_penalty
                    
                    # (2) 获取真实的底层资源惩罚项
                    resource_penalty = resource_mgr.get_penalty(idx)
                    
                    # (3) [核心修正] Lya 得分计算：加入 gamma * selection_counts[idx] 的公平性惩罚
                    final_score = V * utility - resource_penalty - gamma * selection_counts[idx]
                    
                    if final_score > best_score:
                        best_score = final_score
                        best_idx = idx
                
                selected_idxs.append(best_idx)
                remaining_candidates.remove(best_idx)
        
        # === [核心修正] 选人结束后，更新被选节点的计数值 ===
        for idx in selected_idxs:
            selection_counts[idx] += 1
                
        print(f"Selected clients: {selected_idxs}")
        
        # ==========================================================
        # 核心重构 2: 更新李雅普诺夫资源队列 (结算阶段)
        # ==========================================================
        # 必须在确定选了谁之后，立刻更新队列，扣除对应资源
        avg_q_t, avg_q_e = resource_mgr.update_queues_and_counts(selected_idxs)

        # ==========================================================
        # 核心重构 3: 仅让被选中的客户端进行训练 (消耗阶段)
        # ==========================================================
        w_locals = []
        loss_locals = []
        len_locals = []
        
        for idx in selected_idxs:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            # 只有被选中的节点才真刀真枪地消耗算力和电量去 train
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            
            # 训练完成后，计算最新的更新梯度
            update_vec = get_weight_difference(w, w_glob).cpu()
            
            # === [关键步骤] 把最新梯度存入缓存，供下一轮挑选时使用 ===
            historical_updates[idx] = update_vec
            
            w_locals.append({k: v.cpu() for k, v in w.items()})
            loss_locals.append(loss)
            len_locals.append(len(dict_users[idx]))
            
            torch.cuda.empty_cache()

        # ==========================================================
        # 核心重构 4: 模型聚合与全局更新
        # ==========================================================
        w_glob_new = FedAvg(w_locals, len_locals)
        
        current_global_update = get_weight_difference(w_glob_new, w_glob)
        global_update_dir = 0.9 * global_update_dir + 0.1 * current_global_update

        w_glob = w_glob_new
        net_glob.load_state_dict(w_glob)

        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)

        # ... (后续的测试和保存代码保持不变) ...

        net_glob.eval() 
        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        acc_test_history.append(acc_test)
        
        # 打印当前轮次的平均队列长度，用于监控系统资源状态
        print('Round {:3d}, Loss {:.3f}, Acc {:.2f}%, Avg Q_E: {:.3f}, Avg Q_T: {:.3f}'.format(
            iter, loss_avg, acc_test, avg_q_e, avg_q_t))
        net_glob.train()
        # 全局学习率衰减：每轮乘 0.99
        # 跑到 200 轮时，学习率大约会平滑衰减到初始值的 13%，这是十分经典的设定。
        args.lr = args.lr * 0.99

    # ================= [绘图与保存结果] =================
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    script_name = os.path.basename(__file__).split('.')[0]
    
    file_id = 'fed_{}_{}_{}_alpha{}_ep{}_V{}_{}'.format(
        script_name, args.dataset, args.partition, args.alpha, args.epochs, V, timestamp)

    # 增加防崩溃目录检查，确保 save 文件夹存在
    save_dir = './save'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, '{}_acc.npy'.format(file_id))
    np.save(save_path, acc_test_history)
    
    print(f"🎉 实验结束！数据已绝对安全地保存到: {save_path}")


