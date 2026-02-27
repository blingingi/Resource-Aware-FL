#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6



import copy
import torch

def FedAvg(w, dict_len):
    total_data_points = sum(dict_len)
    weights = [len_k / total_data_points for len_k in dict_len]
    
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        # 强制转为 float 计算，避免 int64 乘 float 报错
        w_avg[k] = w_avg[k].float() * weights[0] 
        for i in range(1, len(w)):
            w_avg[k] += w[i][k].float() * weights[i]
        # 计算完后再转回该层原本的数据类型
        w_avg[k] = w_avg[k].to(w[0][k].dtype)
    return w_avg