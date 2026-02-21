#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse#内置的标准库，用于编写用户友好的命令行接口，通过ArgumentParser类
#创建的parser对象可以定义期望的命令行参数，自动解析命令行输入，生成帮助信息，处理错误
#
def args_parser():
    parser = argparse.ArgumentParser()#参数解析器，用于解析命令行参数和选项
    # federated arguments
    parser.add_argument('--epochs', type=int, default=10, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=10, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.0, help="SGD momentum (default: 0.5)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")

    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    #parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--partition', type=str, default='iid', 
                        choices=['iid', 'shard', 'dirichlet'], 
                        help="Data partitioning strategy: 'iid', 'shard', or 'dirichlet'")
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--all_clients', action='store_true', help='aggregation over all clients')
    # 时延约束 (单位: 秒)，默认设为 2.0s
    parser.add_argument('--max_time', type=float, default=500, help='maximum latency constraint per round (seconds)')

    # 能耗约束 (单位: 焦耳)，默认设为 50.0J
    parser.add_argument('--max_energy', type=float, default=50.0, help='maximum energy budget per round (joules)')
    # 添加 Dirichlet 分布的 alpha 参数，默认值设为 0.1
    parser.add_argument('--alpha', type=float, default=0.1, help='The value of alpha for Dirichlet distribution')
    args = parser.parse_args()
    return args
