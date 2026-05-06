#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

class LocalUpdate(object):
    # 【修改】：在参数列表里增加一个 adaptive_ratio (即 w_k)
    def __init__(self, args, dataset=None, idxs=None, adaptive_ratio=1.0):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        
        # 核心逻辑：计算这轮该用多少真实的图片！
        # 确保哪怕比例再小，也至少用 1 个样本，否则代码会崩
        min_samples_required = max(10, self.args.local_bs) 
        actual_num_samples = max(min_samples_required, int(len(idxs) * adaptive_ratio))
        
        # 截断数据！这就是自适应 Batch Size 的物理落地！
        adaptive_idxs = list(idxs)[:actual_num_samples]
        
        self.ldr_train = DataLoader(
            DatasetSplit(dataset, adaptive_idxs), 
            batch_size=self.args.local_bs, 
            shuffle=True
        )
        
    # ... 下面的 train 函数保持你原来的逻辑不变即可 ...
    def train(self, net, global_net):
        net.train()
        # 将 global_net 的参数固化，切断它的梯度图，防止内存爆炸
        global_weight_collector = list(global_net.parameters())
        
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss = []
        
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                
                # 1. 正常的前向传播
                log_probs = net(images)
                
                # 2. 执行标准交叉熵损失
                loss = self.loss_func(log_probs, labels)
                
                # === [Proximal 约束项保留] ===
                if self.args.use_proximal:
                    proximal_term = 0.0
                    # 遍历本地模型的每一个参数矩阵 w，和全局模型的对应矩阵 w_t
                    for w, w_t in zip(net.parameters(), global_weight_collector):
                        # 计算 L2 距离的平方：||w - w_t||^2
                        proximal_term += torch.sum((w - w_t) ** 2)
                    
                    # 将惩罚项按权重 mu/2 叠加到原损失上
                    loss = loss + (self.args.mu / 2) * proximal_term
                # ==========================================================

                # 3. 带着被惩罚过的 Loss 进行反向传播
                loss.backward()
                optimizer.step()
                
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)