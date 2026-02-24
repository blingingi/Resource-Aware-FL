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

def mnist_dirichlet(dataset, num_users, alpha, min_require_size):
    """
    Sample non-I.I.D client data from MNIST dataset using Dirichlet distribution
    支持通过 min_require_size (通常传入 args.local_bs) 动态控制最小样本量
    """
    K = 10 # MNIST 有 10 个类别
    
    # 1. 兼容不同版本的 PyTorch MNIST 标签命名
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, 'train_labels'):
        labels = np.array(dataset.train_labels)
    else:
        labels = np.array([dataset[i][1] for i in range(len(dataset))])
        
    N = len(labels)
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idx_batch = [[] for _ in range(num_users)]
    
    # 预先提取每个类别的索引池，提高 while 循环效率
    idx_k_list = [np.where(labels == k)[0] for k in range(K)]

    # 【已删除硬编码】现在 min_require_size 完全由函数参数决定
    
    max_iters = 10 
    iters = 0
    min_size = 0
    
    # 2. 分配主循环
    while min_size < min_require_size and iters < max_iters:
        idx_batch = [[] for _ in range(num_users)]
        for k in range(K):
            idx_k = idx_k_list[k].copy()
            np.random.shuffle(idx_k)
            
            # 根据 alpha 生成分布比例
            proportions = np.random.dirichlet(np.repeat(alpha, num_users))
            
            # 平衡性修正：防止单个客户端分到过多数据
            # 增加 np.sum(eff_probs) > 0 判断，防止 alpha 极小时全为 0 导致除以 0 崩溃
            eff_probs = np.array([p * (len(idx_j) < N / num_users) for p, idx_j in zip(proportions, idx_batch)])
            if eff_probs.sum() > 0:
                proportions = eff_probs / eff_probs.sum()
            
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            
            # 分配数据索引
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            
        min_size = min([len(idx_j) for idx_j in idx_batch])
        iters += 1

    # 3. 【无损兜底策略】确保每个客户端至少有 min_require_size 个样本
    for j in range(num_users):
        current_len = len(idx_batch[j])
        if current_len < min_require_size:
            if current_len > 0:
                # 方案 A: 客户端有少量数据，按现有分布放回重复采样
                need_size = min_require_size - current_len
                extra_idx = np.random.choice(idx_batch[j], need_size, replace=True)
                idx_batch[j].extend(extra_idx.tolist())
            else:
                # 方案 B: 极端空节点，随机分配一个主导类别并采样补齐
                random_class = np.random.randint(0, K)
                class_indices = idx_k_list[random_class]
                extra_idx = np.random.choice(class_indices, min_require_size, replace=False)
                idx_batch[j] = extra_idx.tolist()
                
    # 4. 打乱并输出结果
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



def cifar_dirichlet(dataset, num_users, alpha, min_require_size):
    """
    使用 Dirichlet 分布分配数据，支持动态最小样本量 (min_require_size)
    """
    K = 10 
    
    # 1. 兼容性获取标签
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, 'train_labels'):
        labels = np.array(dataset.train_labels)
    else:
        labels = np.array([dataset[i][1] for i in range(len(dataset))])
        
    N = len(labels)
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idx_batch = [[] for _ in range(num_users)]
    
    # 2. 预先获取每个类别的索引池
    idx_k_list = [np.where(labels == k)[0] for k in range(K)]
    
    # 【修复】删除函数内部硬编码的 min_require_size = 32，直接使用传入参数
    
    max_iters = 10 
    iters = 0
    min_size = 0
    
    # 3. 分配循环
    while min_size < min_require_size and iters < max_iters:
        idx_batch = [[] for _ in range(num_users)]
        for k in range(K):
            idx_k = idx_k_list[k].copy()
            np.random.shuffle(idx_k)
            
            # 生成分布比例
            proportions = np.random.dirichlet(np.repeat(alpha, num_users))
            
            # 这里的平衡逻辑在极低 alpha 下可能失效，做了安全处理
            effective_probs = np.array([p * (len(idx_j) < N / num_users) for p, idx_j in zip(proportions, idx_batch)])
            if effective_probs.sum() > 0:
                proportions = effective_probs / effective_probs.sum()
            
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            split_idxs = np.split(idx_k, proportions)
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, split_idxs)]
            
        min_size = min([len(idx_j) for idx_j in idx_batch])
        iters += 1

    # 4. 【核心兜底策略】确保每个客户端拥有至少一个完整的 Batch 量
    for j in range(num_users):
        current_len = len(idx_batch[j])
        if current_len < min_require_size:
            if current_len > 0:
                # 方案 A: 放回重复采样（保持该客户端已有的 Non-IID 分布）
                need_size = min_require_size - current_len
                extra_idx = np.random.choice(idx_batch[j], need_size, replace=True)
                idx_batch[j].extend(extra_idx.tolist())
            else:
                # 方案 B: 极端空节点补齐（随机选一个类别进行采样）
                random_class = np.random.randint(0, K)
                class_indices = idx_k_list[random_class]
                extra_idx = np.random.choice(class_indices, min_require_size, replace=False)
                idx_batch[j] = extra_idx.tolist()

    # 5. 打乱并输出
    for j in range(num_users):
        np.random.shuffle(idx_batch[j])
        dict_users[j] = np.array(idx_batch[j], dtype='int64')
        
    return dict_users



if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
