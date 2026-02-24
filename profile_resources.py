#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from torchvision import datasets, transforms
from utils.sampling import cifar_dirichlet
from utils.resource import ResourceManager

def profile_system_resources():
    print("="*50)
    print("🚀 启动系统资源摸底测试 (CIFAR-10, Dir=0.1)")
    print("="*50)

    # 1. 模拟参数
    num_users = 100
    alpha = 0.1
    local_bs = 32
    
    # 2. 加载数据集
    print("加载 CIFAR-10 数据集...")
    trans_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_train)

    # 3. 划分数据 (严格使用你刚刚修好的带 min_require_size 的采样函数)
    print(f"按 Dirichlet (alpha={alpha}) 划分数据中...")
    dict_users = cifar_dirichlet(dataset_train, num_users, alpha, local_bs)
    
    # 4. 初始化资源管理器
    resource_mgr = ResourceManager(num_users)
    
    # 5. 计算并收集所有客户端的资源开销
    times = []
    energies = []
    data_sizes = []
    
    for i in range(num_users):
        data_size = len(dict_users[i])
        # 假设 local_ep = 2
        # 注意：如果你的 resource_mgr.calculate_cost 没有考虑 epoch，那这里的成本就是单轮成本。
        # 这里我们直接透视它返回的值
        t, e = resource_mgr.calculate_cost(i, data_size)
        times.append(t)
        energies.append(e)
        data_sizes.append(data_size)

    # 6. 统计分析
    times = np.array(times)
    energies = np.array(energies)
    data_sizes = np.array(data_sizes)
    
    print("\n📊 --- [数据量分布] ---")
    print(f"最小数据量: {data_sizes.min()} 张")
    print(f"最大数据量: {data_sizes.max()} 张")
    print(f"平均数据量: {data_sizes.mean():.1f} 张")
    
    print("\n⏱️ --- [时延 (Latency) 分布] ---")
    print(f"最快完成: {times.min():.2f} s")
    print(f"最慢完成: {times.max():.2f} s")
    print(f"平均耗时: {times.mean():.2f} s")
    print(f"中位数 (50%的人小于): {np.median(times):.2f} s")
    print(f"75% 分位数 (75%的人小于): {np.percentile(times, 75):.2f} s")
    print(f"90% 分位数 (90%的人小于): {np.percentile(times, 90):.2f} s")

    print("\n🔋 --- [单客户端能耗 (Energy) 分布] ---")
    print(f"最低能耗: {energies.min():.2f} J")
    print(f"最高能耗: {energies.max():.2f} J")
    print(f"平均能耗: {energies.mean():.2f} J")
    
    print("\n💡 --- [关于 max_energy 的设定建议] ---")
    # 假设我们一轮选 10 个人 (frac=0.1)
    avg_round_energy = energies.mean() * 10
    print(f"如果随机选 10 个人，一轮的平均总能耗约为: {avg_round_energy:.2f} J")
    
    print("\n🎯 --- [学长的终极设定建议] ---")
    print("1. 【挑战性 max_time】: 建议设在 '中位数' 和 '75% 分位数' 之间。")
    print("   这会逼迫你的 Ours 算法对那 25% 最慢的节点使用'弹性降级(Partial Work)'。")
    print(f"2. 【挑战性 max_energy】: 建议设为平均总能耗 ({avg_round_energy:.2f} J) 的 70% ~ 80%。")
    print("   这会逼迫你的贪心背包算法精打细算，绝不浪费哪怕 1 焦耳的电量！")
    print("="*50)

if __name__ == '__main__':
    profile_system_resources()