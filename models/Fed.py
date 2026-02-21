#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6



import copy
import torch

def FedAvg(w, dict_len):
    """
    加权平均聚合 (严格遵照 FedAvg 论文公式)
    :param w: 各个客户端的模型权重列表
    :param dict_len: 对应客户端的本地数据样本数列表
    """
    total_data_points = sum(dict_len)
    # 计算每个客户端的加权比例
    weights = [len_k / total_data_points for len_k in dict_len]
    
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] = w_avg[k] * weights[0] # 初始化加上第一项的权重
        for i in range(1, len(w)):
            w_avg[k] += w[i][k] * weights[i]
    return w_avg