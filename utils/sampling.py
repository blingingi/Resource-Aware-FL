#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users

def mnist_dirichlet(dataset, num_users, alpha):
    """
    Sample non-I.I.D client data from MNIST dataset using Dirichlet distribution
    包含防死锁机制与最小样本量保底策略
    """
    K = 10 # MNIST 有 10 个类别
    min_require_size = 10 # 保证每个客户端至少有 10 张图（一个 batch），防止 PyTorch 训练崩溃
    
    # 兼容不同版本的 PyTorch MNIST 标签命名
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, 'train_labels'):
        labels = np.array(dataset.train_labels)
    else:
        labels = np.array([dataset[i][1] for i in range(len(dataset))])
        
    N = len(labels)
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    idx_batch = [[] for _ in range(num_users)]
    
    # 【修复核心】增加最大尝试次数 (max_iters)，强行打破死循环死锁
    max_iters = 10 
    iters = 0
    min_size = 0
    
    while min_size < min_require_size and iters < max_iters:
        idx_batch = [[] for _ in range(num_users)]
        for k in range(K):
            idx_k = np.where(labels == k)[0]
            np.random.shuffle(idx_k)
            
            # 根据 alpha 生成分布比例
            proportions = np.random.dirichlet(np.repeat(alpha, num_users))
            # 限制每个客户端的数据量不能过多，保持整体平衡
            proportions = np.array([p * (len(idx_j) < N / num_users) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            
            # 分配数据索引
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            
        min_size = min([len(idx_j) for idx_j in idx_batch])
        iters += 1

    # 【兜底策略】对于 alpha=0.1 的极端情况，绝大多数客户端注定分不到数据
    # 为防止后续 PyTorch 出现 ValueError 崩溃，强行给不足 10 张图的贫困节点补发随机数据
    for j in range(num_users):
        if len(idx_batch[j]) < min_require_size:
            extra_idx = np.random.choice(N, min_require_size, replace=False)
            idx_batch[j] = idx_batch[j] + extra_idx.tolist()

    for j in range(num_users):
        np.random.shuffle(idx_batch[j])
        dict_users[j] = np.array(idx_batch[j], dtype='int64')
        
    return dict_users

def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def cifar_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 250
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    # labels = dataset.train_labels.numpy() # 旧版 PyTorch 写法
    # 新版 PyTorch (如 1.x/2.x) 写法:
    labels = np.array(dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users



def cifar_dirichlet(dataset, num_users, alpha):
    """
    Sample non-I.I.D client data from CIFAR10 dataset using Dirichlet distribution
    """
    K = 10 # CIFAR-10 有 10 个类别
    min_require_size = 10 # 保证每个客户端至少有 10 张图（一个 batch），防止 PyTorch 训练崩溃
    
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    else:
        labels = np.array(dataset.train_labels)
        
    N = len(labels)
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    idx_batch = [[] for _ in range(num_users)]
    
    # 【修复核心】增加最大尝试次数 (max_iters)，强行打破死循环死锁
    max_iters = 10 
    iters = 0
    min_size = 0
    
    while min_size < min_require_size and iters < max_iters:
        idx_batch = [[] for _ in range(num_users)]
        for k in range(K):
            idx_k = np.where(labels == k)[0]
            np.random.shuffle(idx_k)
            # 根据 alpha 生成分布比例
            proportions = np.random.dirichlet(np.repeat(alpha, num_users))
            # 限制每个客户端的数据量不能过多，保持整体平衡
            proportions = np.array([p * (len(idx_j) < N / num_users) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            
            # 分配数据索引
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            
        min_size = min([len(idx_j) for idx_j in idx_batch])
        iters += 1

    # 【兜底策略】对于 alpha=0.01 的极端情况，绝大多数客户端注定分不到数据
    # 为防止后续 PyTorch 出现 ZeroDivisionError 崩溃，强行给不足 10 张图的贫困节点补发随机数据
    for j in range(num_users):
        if len(idx_batch[j]) < min_require_size:
            extra_idx = np.random.choice(N, min_require_size, replace=False)
            idx_batch[j] = idx_batch[j] + extra_idx.tolist()

    for j in range(num_users):
        np.random.shuffle(idx_batch[j])
        dict_users[j] = np.array(idx_batch[j])
        
    return dict_users





if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
