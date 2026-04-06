#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
#无任何优化策略，作为性能基准
#不考虑资源消耗、数据质量、公平性
#实现最简单，但收敛慢且可能违反资源预算

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import os

# 引入 cifar_noniid
from utils.sampling import mnist_iid, mnist_noniid, mnist_dirichlet, cifar_iid, cifar_noniid, cifar_dirichlet
from utils.options import args_parser
from utils.seed import set_seed
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img


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
        # ================= [MNIST 数据划分逻辑] =================
        if args.partition == 'iid':
            print("=> 正在使用 IID 均匀划分 MNIST 数据...")
            dict_users = mnist_iid(dataset_train, args.num_users)
            
        elif args.partition == 'shard':
            print("=> 正在使用 Shard 分片划分 MNIST 数据 (每个客户端 2 种标签)...")
            # 调用你 sampling.py 中的 mnist_noniid 函数
            dict_users = mnist_noniid(dataset_train, args.num_users)
            
        elif args.partition == 'dirichlet':
            print(f"=> 正在使用 Dirichlet 划分 MNIST 数据, alpha={args.alpha}...")
            dict_users = mnist_dirichlet(dataset_train, args.num_users, args.alpha)
            
        else:
            # 严密的错误拦截
            exit('Error: unrecognized partition strategy for MNIST. Please choose from [iid, shard, dirichlet]')
    elif args.dataset == 'cifar':
        trans_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),     # 随机裁剪（标准CIFAR增强）
        transforms.RandomHorizontalFlip(),        # 随机水平翻转
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), 
                             (0.5, 0.5, 0.5))
        ])
        trans_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), 
                             (0.5, 0.5, 0.5))
        ])
        dataset_train = datasets.CIFAR10(
        '../data/cifar', 
        train=True, 
        download=True, 
        transform=trans_train
        )
        dataset_test = datasets.CIFAR10(
        '../data/cifar', 
        train=False, 
        download=True, 
        transform=trans_test
        )
        
        if args.partition=='iid':
            print("=> 正在使用 IID 均匀划分数据...")
            dict_users = cifar_iid(dataset_train, args.num_users)
        elif args.partition=='shard':
            print("=> 正在使用 Shard 分片划分数据...")
            dict_users = cifar_noniid(dataset_train, args.num_users)
        elif args.partition == 'dirichlet':
            print(f"=> 正在使用 Dirichlet 划分数据, alpha={args.alpha}...")
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


    for iter in range(args.epochs):
        loss_locals = []
        len_locals = [] 
        w_locals = []
            
        m = max(int(args.frac * args.num_users), 1)
        
        # 纯随机选择
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            
            # 【修复4：去除冗余的 global_net 深拷贝】
            # global_net 仅作为只读参考，直接传入 net_glob 本体即可，Update.py 中并不会修改它
            w, loss = local.train(
                net=copy.deepcopy(net_glob).to(args.device),
                global_net=net_glob 
            )
            
            # 【修复2：彻底解决 VRAM 显存泄漏】
            # 必须将权重逐个拉回 CPU，抛弃无脑的 deepcopy
            w_locals.append({k: v.cpu() for k, v in w.items()})
            loss_locals.append(loss)
            len_locals.append(len(dict_users[idx]))
            
            # 显式清空当前客户端训练产生的显存碎片
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        # 传递包含真实数据量权重的 len_locals
        w_glob = FedAvg(w_locals, len_locals)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)

        # 评估
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
        script_name, args.dataset, args.partition, args.alpha, args.epochs, timestamp)

    # 【修复4】增加防崩溃目录检查，确保 save 文件夹存在
    save_dir = './save'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, '{}_acc.npy'.format(file_id))
    np.save(save_path, acc_test_history)
    
    print(f"🎉 实验结束！数据已绝对安全地保存到: {save_path}")