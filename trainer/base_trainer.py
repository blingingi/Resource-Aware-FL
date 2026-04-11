#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
联邦学习基础训练器 - 封装所有公共逻辑
"""
import copy
import numpy as np
import torch
from torchvision import datasets, transforms
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img
from utils.seed import set_seed
import os
import datetime


class BaseTrainer:
    """基础训练器：封装数据加载、模型构建、训练循环"""
    
    def __init__(self, args):
        self.args = args
        self.args.device = torch.device(
            'cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu'
        )
        set_seed(42)
        
        # 数据集和数据划分
        self.dataset_train, self.dataset_test, self.dict_users = self._load_and_split_data()
        
        # 模型构建
        self.net_glob = self._build_model()
        self.w_glob = self.net_glob.state_dict()
        
        # 训练记录
        self.loss_train = []
        self.acc_test_history = []
    
    def _load_and_split_data(self):
        """加载数据集并进行划分"""
        from utils.sampling import (
            mnist_iid, mnist_noniid, mnist_dirichlet,
            cifar_iid, cifar_noniid, cifar_dirichlet
        )
        
        if self.args.dataset == 'mnist':
            trans_mnist = transforms.Compose([
                transforms.ToTensor(), 
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            dataset_train = datasets.MNIST(
                '../data/mnist/', train=True, download=True, transform=trans_mnist
            )
            dataset_test = datasets.MNIST(
                '../data/mnist/', train=False, download=True, transform=trans_mnist
            )
            
            if self.args.partition == 'iid':
                dict_users = mnist_iid(dataset_train, self.args.num_users)
            elif self.args.partition == 'shard':
                dict_users = mnist_noniid(dataset_train, self.args.num_users)
            elif self.args.partition == 'dirichlet':
                try:
                    dict_users = mnist_dirichlet(dataset_train, self.args.num_users, self.args.alpha)
                except:
                    dict_users = cifar_dirichlet(
                        dataset_train, self.args.num_users, self.args.alpha, self.args.local_bs
                    )
            else:
                raise ValueError('Error: unrecognized partition strategy for MNIST')
                
        elif self.args.dataset == 'cifar':
            trans_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            trans_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            dataset_train = datasets.CIFAR10(
                '../data/cifar', train=True, download=True, transform=trans_train
            )
            dataset_test = datasets.CIFAR10(
                '../data/cifar', train=False, download=True, transform=trans_test
            )
            
            if self.args.partition == 'iid':
                dict_users = cifar_iid(dataset_train, self.args.num_users)
            elif self.args.partition == 'shard':
                dict_users = cifar_noniid(dataset_train, self.args.num_users)
            elif self.args.partition == 'dirichlet':
                dict_users = cifar_dirichlet(
                    dataset_train, self.args.num_users, self.args.alpha, self.args.local_bs
                )
            else:
                raise ValueError('Error: unrecognized partition strategy for CIFAR')
        else:
            raise ValueError('Error: unrecognized dataset')
        
        return dataset_train, dataset_test, dict_users
    
    def _build_model(self):
        """构建全局模型"""
        img_size = self.dataset_train[0][0].shape
        
        if self.args.model == 'cnn' and self.args.dataset == 'cifar':
            net_glob = CNNCifar(args=self.args).to(self.args.device)
        elif self.args.model == 'cnn' and self.args.dataset == 'mnist':
            net_glob = CNNMnist(args=self.args).to(self.args.device)
        elif self.args.model == 'mlp':
            len_in = 1
            for x in img_size:
                len_in *= x
            net_glob = MLP(
                dim_in=len_in, dim_hidden=200, dim_out=self.args.num_classes
            ).to(self.args.device)
        else:
            raise ValueError('Error: unrecognized model')
        
        print(net_glob)
        net_glob.train()
        return net_glob
    
    def select_clients(self, iter):
        """
        客户端选择策略（子类必须实现）
        返回: selected_idxs (list), 额外信息 (dict, 可选)
        """
        raise NotImplementedError("子类必须实现 select_clients 方法")
    
    def train(self):
        """执行完整的训练流程"""
        for iter in range(self.args.epochs):
            # 1. 选择客户端
            result = self.select_clients(iter)
            if isinstance(result, tuple):
                selected_idxs, extra_info = result
            else:
                selected_idxs = result
                extra_info = {}
            
            # 2. 本地训练
            w_locals, loss_locals, len_locals = self._local_train(selected_idxs)
            
            # 3. 模型聚合
            w_glob_new = FedAvg(w_locals, len_locals)
            self.w_glob = w_glob_new
            self.net_glob.load_state_dict(w_glob_new)
            
            # 4. 计算平均损失
            loss_avg = sum(loss_locals) / len(loss_locals)
            self.loss_train.append(loss_avg)
            
            # 5. 评估
            self.net_glob.eval()
            acc_test, loss_test = test_img(self.net_glob, self.dataset_test, self.args)
            self.acc_test_history.append(acc_test)
            
            # 6. 打印日志
            log_msg = 'Round {:3d}, Loss {:.3f}, Acc {:.2f}%'.format(
                iter, loss_avg, acc_test
            )
            if 'avg_q_e' in extra_info and 'avg_q_t' in extra_info:
                log_msg += ', Avg Q_E: {:.3f}, Avg Q_T: {:.3f}'.format(
                    extra_info['avg_q_e'], extra_info['avg_q_t']
                )
            print(log_msg)
            
            self.net_glob.train()
            self.args.lr = self.args.lr * 0.99
            
            # 7. 子类钩子（可选）
            self.on_round_end(iter, extra_info)
        
        # 保存结果
        self._save_results()
    
    def _local_train(self, selected_idxs):
        """对选中的客户端进行本地训练"""
        w_locals = []
        loss_locals = []
        len_locals = []
        
        for idx in selected_idxs:
            local = LocalUpdate(
                args=self.args, dataset=self.dataset_train, idxs=self.dict_users[idx]
            )
            w, loss = local.train(
                net=copy.deepcopy(self.net_glob).to(self.args.device),
                global_net=self.net_glob
            )
            
            w_locals.append({k: v.cpu() for k, v in w.items()})
            loss_locals.append(loss)
            len_locals.append(len(self.dict_users[idx]))
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return w_locals, loss_locals, len_locals
    
    def on_round_end(self, iter, extra_info):
        """每轮结束后的钩子函数（子类可重写）"""
        pass
    
    def _save_results(self):
        """保存实验结果"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        script_name = self.__class__.__name__.lower().replace('strategy', '')
        
        file_id = 'fed_{}_{}_{}_alpha{}_ep{}_{}'.format(
            script_name, 
            self.args.dataset, 
            self.args.partition, 
            self.args.alpha, 
            self.args.epochs, 
            timestamp
        )
        
        save_dir = './save'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        save_path = os.path.join(save_dir, '{}_acc.npy'.format(file_id))
        np.save(save_path, self.acc_test_history)
        
        print(f"🎉 实验结束！数据已保存到: {save_path}")