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
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        
        # === [新增代码：计算本地数据分布 pi_y] ===
        if self.args.use_logits:
            self.pi_y = torch.zeros(args.num_classes)
            for _, labels in self.ldr_train:
                self.pi_y += torch.bincount(labels, minlength=args.num_classes).cpu()
            
            total_samples = self.pi_y.sum()
            if total_samples > 0:
                self.pi_y = self.pi_y / total_samples
                
            self.pi_y = self.pi_y.to(self.args.device)
        # ==========================================

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
                
                # === [致命修复 2：补齐 Logit Adjustment 真实计算代码] ===
                if self.args.use_logits:
                    tau = 1.0
                    epsilon = 1e-8
                    # 计算惩罚项并减去
                    adjustment = tau * torch.log(self.pi_y + epsilon)
                    adjusted_logits = log_probs - adjustment
                    loss = self.loss_func(adjusted_logits, labels)
                else:
                    # 不开 Logits 时，执行标准交叉熵
                    loss = self.loss_func(log_probs, labels)
                # ==========================================================
                
                # === [致命修复 1：统一变量名为 use_proximal] ===
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