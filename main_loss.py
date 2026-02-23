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

from utils.sampling import mnist_iid, mnist_noniid,mnist_dirichlet, cifar_iid,cifar_noniid, cifar_dirichlet
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # [可选] 固定随机种子 (为了复现实验)
    # import random
    # if args.seed is not None:
    #     random.seed(args.seed)
    #     torch.manual_seed(args.seed)
    #     np.random.seed(args.seed)

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
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
    
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    if args.all_clients: 
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]

    # [策略准备] 
    # 1. 初始化 Loss (用于计算概率)
    client_losses = np.ones(args.num_users) * 100.0
    # 2. 初始化计数器 (用于画“频率分布图”)
    client_selection_count = np.zeros(args.num_users)

    for iter in range(args.epochs):
        loss_locals = []
        if not args.all_clients:
            w_locals = []
            
        # ================= [Client Selection Strategy] =================
        m = max(int(args.frac * args.num_users), 1)
        
        # 基于 Loss 计算概率 (Loss 越大，概率越大)
        p_values = np.abs(client_losses) + 1e-8
        p_values = p_values / np.sum(p_values)
        
        try:
            idxs_users = np.random.choice(range(args.num_users), m, replace=False, p=p_values)
        except:
            # 容错处理：如果概率计算数值不稳定，回退到随机
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)
            
        # [统计] 记录本轮谁被选中了
        for i in idxs_users:
            client_selection_count[i] += 1
        # ===============================================================

        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            
            # [反馈] 记录该用户的 Loss，供下一轮选择使用
            client_losses[idx] = loss
            
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
            
        # update global weights
        w_glob = FedAvg(w_locals)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)

        # [评估] 每一轮结束时，跑一次测试并记录 Accuracy
        net_glob.eval() 
        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        acc_test_history.append(acc_test)
        print('Round {:3d}, Average loss {:.3f}, Test Acc {:.2f}%'.format(iter, loss_avg, acc_test))
        net_glob.train() 
        args.lr = args.lr * 0.99

 

    # ================= [绘图与保存结果] =================
    import os
    import datetime

    # 1. 获取精确到秒的时间戳，确保绝对不覆盖
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 2. 获取当前运行脚本的名字 (自动识别是 baseline 还是 litong_v2)
    script_name = os.path.basename(__file__).split('.')[0]
    
    # 3. 【核心修复】文件名里加上 脚本名、alpha 和 timestamp
    # 推荐修改你的保存命名逻辑，加入 partition 字段
    file_id = 'fed_{}_{}_{}_alpha{}_ep{}_{}'.format(
        script_name, args.dataset, args.partition, args.alpha, args.epochs, timestamp)

    # 4. 保存原始数据 (.npy)
    save_path = './save/{}_acc.npy'.format(file_id)
    np.save(save_path, acc_test_history)
    
    print(f"🎉 实验结束！数据已绝对安全地保存到: {save_path}")