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
                
            # 转移到设备上备用
            self.pi_y = self.pi_y.to(self.args.device)
            # ==========================================

    def train(self, net):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                
                log_probs = net(images) 
                
                # === [根据开关决定是否进行 Logits 惩罚] ===
                if self.args.use_logits:
                    tau = 1.0
                    epsilon = 1e-8
                    adjustment = tau * torch.log(self.pi_y + epsilon)
                    adjusted_logits = log_probs - adjustment
                    loss = self.loss_func(adjusted_logits, labels)
                else:
                    # 如果没开开关，就走最标准的交叉熵计算
                    loss = self.loss_func(log_probs, labels)
                # ============================================

                loss.backward()
                optimizer.step()
                
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

