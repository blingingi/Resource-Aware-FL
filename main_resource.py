#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
#设计理念：在严格的物理红线内做随机选择
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
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img
from utils.seed import set_seed
from utils.resource import ResourceManager
# 【修复1】：剔除了过期且未使用的 sim_div 导入

if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    set_seed(42)

    # ================= [Load Dataset] =================
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        
        if args.partition == 'iid':
            dict_users = mnist_iid(dataset_train, args.num_users)
        elif args.partition == 'shard':
            dict_users = mnist_noniid(dataset_train, args.num_users)
        elif args.partition == 'dirichlet':
            try:
                from utils.sampling import mnist_dirichlet
                dict_users = mnist_dirichlet(dataset_train, args.num_users, args.alpha)
            except ImportError:
                dict_users = cifar_dirichlet(dataset_train, args.num_users, args.alpha, args.local_bs)
        else:
            exit('Error: unrecognized partition strategy for MNIST')
            
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
    elif args.model == 'mlp':
        img_size = dataset_train[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    
    print(net_glob)
    net_glob.train()
    w_glob = net_glob.state_dict()

    # ================= [只保留资源管理器，删掉所有数据打分] =================
    w_glob = net_glob.state_dict()
    # 必须传入 dict_users 和 local_ep 才能初始化新的物理能耗模型
    resource_mgr = ResourceManager(args.num_users, dict_users, local_ep=args.local_ep)
    
    # 【修复2】：动态提取公平的系统红线 (单轮系统最大容忍耗时，和 m个节点的总耗能预算)
    m_target = max(int(args.frac * args.num_users), 1)
    SYS_MAX_TIME = resource_mgr.T_limit * 1.5   # 允许单轮最大时延(可适当放宽防止过于苛刻)
    SYS_MAX_ENERGY = resource_mgr.E_limit * m_target # 本轮总电量预算
    
    loss_train = []
    acc_test_history = [] 

    for iter in range(args.epochs):
        loss_locals = []
        len_locals = [] 
        w_locals = []
            
        m = max(int(args.frac * args.num_users), 1)

        # ================= [Selection Strategy (Random + Resource Constraints)] =================
        all_clients = list(range(args.num_users))
        np.random.shuffle(all_clients)
        
        selected_users = []
        current_energy_sum = 0.0
        current_max_time = 0.0
        
        for client_id in all_clients:
            t = resource_mgr.time_costs[client_id]
            e = resource_mgr.energy_costs[client_id]
            
            potential_energy = current_energy_sum + e
            potential_time = max(current_max_time, t)
            
            # 严格检查客观物理红线
            if potential_energy <= SYS_MAX_ENERGY and potential_time <= SYS_MAX_TIME:
                selected_users.append(client_id)
                current_energy_sum = potential_energy
                current_max_time = potential_time
                
            if len(selected_users) >= m:
                break
                
        # 3. 学术兜底机制
        if len(selected_users) == 0:
            print("⚠️ 警告：本轮系统资源约束过于严苛，触发兜底！强行选择 1 个最高效节点。")
            times = resource_mgr.time_costs
            fastest_client = np.argmin(times)
            selected_users = [fastest_client]
            current_max_time = resource_mgr.time_costs[fastest_client]
            current_energy_sum = resource_mgr.energy_costs[fastest_client]
            
        print(f"Round {iter} | 随机约束装箱 | 选中 {len(selected_users)} 人 | 时延: {current_max_time:.2f}s | 耗能: {current_energy_sum:.2f}J")
        # ========================================================================================

        for idx in selected_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            
            w, loss = local.train(
                net=copy.deepcopy(net_glob).to(args.device),
                global_net=net_glob
            )
            
            w_locals.append({k: v.cpu() for k, v in w.items()})
            loss_locals.append(loss)
            len_locals.append(len(dict_users[idx]))
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
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
    
    file_id = 'fed_{}_{}_{}_alpha{}_ep{}_{}'.format(
        script_name, 
        args.dataset, 
        args.partition, 
        args.alpha, 
        args.epochs, 
        timestamp
    )

    save_dir = './save'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, '{}_acc.npy'.format(file_id))
    np.save(save_path, acc_test_history)
    print(f"🎉 实验结束！数据已绝对安全地保存到: {save_path}")