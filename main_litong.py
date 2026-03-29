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

# 导入自定义模块
from utils.sampling import mnist_iid, mnist_noniid, mnist_dirichlet, cifar_iid, cifar_noniid, cifar_dirichlet
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img

# 导入余弦相似度计算与种子固定
from utils.sim_div import get_weight_difference, compute_cosine_similarity
from utils.seed import set_seed  # 【必须导入】

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    
    # 【致命修复 1】：固定随机种子，保证和 Base 实验完全一致的数据异质性切片
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
            exit('Error: unrecognized partition strategy for MNIST.')
            
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
            dict_users = cifar_iid(dataset_train, args.num_users)
        elif args.partition=='shard':
            dict_users = cifar_noniid(dataset_train, args.num_users)
        elif args.partition == 'dirichlet':
            dict_users = cifar_dirichlet(dataset_train, args.num_users, args.alpha,args.local_bs)
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

    # === 初始化全局更新方向参考 ===
    flat_w_glob = get_weight_difference(w_glob, w_glob) 
    global_update_dir = torch.zeros_like(flat_w_glob).cpu()

    # 超参数：SIM与DIV的权重
    alpha = 0.8  # SIM的权重
    beta = 0.2   # DIV的权重

    for iter in range(args.epochs):
        m = max(int(args.frac * args.num_users), 1)
        pool_size = min(m * 2, args.num_users) 
        
        # 1. 随机挑选候选池
        candidate_idxs = np.random.choice(range(args.num_users), pool_size, replace=False)
        
        candidate_updates = {}
        candidate_w_locals = {}
        candidate_losses = {}
        candidate_lens = {}

        print(f"\n--- Round {iter} ---")
        print(f"Training candidates ({pool_size} clients)...")
        
        # 2. 候选池节点进行本地训练
        for idx in candidate_idxs:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            
            # 【致命修复 2】：补充 global_net 传参，防止 Update.py 报错
            w, loss = local.train(
                net=copy.deepcopy(net_glob).to(args.device),
                global_net=net_glob
            )
            
            update_vec = get_weight_difference(w, w_glob).cpu()
            candidate_updates[idx] = update_vec
            
            candidate_w_locals[idx] = {k: v.cpu() for k, v in w.items()}
            candidate_losses[idx] = loss
            candidate_lens[idx] = len(dict_users[idx])
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 3. 基于 SIM + DIV 贪心挑选最终的 m 个节点
        selected_idxs = []
        
        if iter == 0 or global_update_dir.norm() == 0:
            selected_idxs = np.random.choice(candidate_idxs, m, replace=False).tolist()
        else:
            sim_scores = {}
            for idx in candidate_idxs:
                sim = compute_cosine_similarity(candidate_updates[idx], global_update_dir)
                # 【优化】：新版函数安全可靠，直接取值即可，剔除繁琐的 tensor 转化
                sim_scores[idx] = sim
            
            remaining_candidates = list(candidate_idxs)
            
            for _ in range(m):
                best_score = -float('inf')
                best_idx = -1
                
                for idx in remaining_candidates:
                    utility = alpha * sim_scores[idx]
                    div_penalty = 0.0
                    
                    if len(selected_idxs) > 0:
                        for selected_idx in selected_idxs:
                            sim_with_selected = compute_cosine_similarity(
                                candidate_updates[idx], candidate_updates[selected_idx]
                            )
                            div_penalty += sim_with_selected
                        
                        div_penalty = div_penalty / len(selected_idxs)
                        
                    final_score = utility - beta * div_penalty
                    
                    if final_score > best_score:
                        best_score = final_score
                        best_idx = idx
                
                selected_idxs.append(best_idx)
                remaining_candidates.remove(best_idx)
                
        print(f"Selected clients: {selected_idxs}")

        # 4. 提取被选中节点的数据准备聚合
        w_locals = [candidate_w_locals[idx] for idx in selected_idxs]
        len_locals = [candidate_lens[idx] for idx in selected_idxs]
        loss_locals = [candidate_losses[idx] for idx in selected_idxs]

        # 5. 加权聚合
        w_glob_new = FedAvg(w_locals, len_locals)
        
        # 6. 更新全局参考方向 (平滑动量)
        current_global_update = get_weight_difference(w_glob_new, w_glob)
        global_update_dir = 0.9 * global_update_dir + 0.1 * current_global_update

        w_glob = w_glob_new
        net_glob.load_state_dict(w_glob)

        # 7. 评估与日志
        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)

        net_glob.eval() 
        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        acc_test_history.append(acc_test)
        print('Round {:3d}, Average loss {:.3f}, Test Acc {:.2f}%'.format(iter, loss_avg, acc_test))
        net_glob.train()
        args.lr = args.lr * 0.99

    # ================= [绘图与保存结果] =================
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    script_name = os.path.basename(__file__).split('.')[0]
    
    file_id = 'fed_{}_{}_{}_alpha{}_ep{}_{}'.format(
        script_name, args.dataset, args.partition, args.alpha, args.epochs, timestamp)

    save_dir = './save'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, '{}_acc.npy'.format(file_id))
    np.save(save_path, acc_test_history)
    print(f"🎉 实验结束！数据已绝对安全地保存到: {save_path}")