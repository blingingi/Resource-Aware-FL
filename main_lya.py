import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import os

# 引入 cifar_noniid
from utils.sampling import mnist_iid, mnist_noniid, mnist_dirichlet, cifar_iid, cifar_noniid, cifar_dirichlet
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img

# 导入你的资源管理器
from utils.resource import ResourceManager
from utils.sim_div import get_weight_difference, compute_cosine_similarity


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # ================= [MNIST 数据划分逻辑] =================
        if args.partition == 'iid':
            print("=> 正在使用 IID 均匀划分 MNIST 数据...")
            dict_users = mnist_iid(dataset_train, args.num_users)
            
        elif args.partition == 'shard':
            print("=> 正在使用 Shard 分片划分 MNIST 数据 (每个客户端 2 种标签)...")
            # 调用你 sampling.py 中的 mnist_noniid 函数
            dict_users = mnist_noniid(dataset_train, args.num_users)
            
        elif args.partition == 'dirichlet':
            print(f"=> 正在使用 Dirichlet 划分 MNIST 数据, alpha={args.alpha}...")
            dict_users = mnist_dirichlet(dataset_train, args.num_users, args.alpha)
            
        else:
            # 严密的错误拦截
            exit('Error: unrecognized partition strategy for MNIST. Please choose from [iid, shard, dirichlet]')
    elif args.dataset == 'cifar':
        trans_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),     # 随机裁剪（标准CIFAR增强）
        transforms.RandomHorizontalFlip(),        # 随机水平翻转
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), 
                             (0.5, 0.5, 0.5))
        ])
        trans_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), 
                             (0.5, 0.5, 0.5))
        ])
        dataset_train = datasets.CIFAR10(
        '../data/cifar', 
        train=True, 
        download=True, 
        transform=trans_train
        )
        dataset_test = datasets.CIFAR10(
        '../data/cifar', 
        train=False, 
        download=True, 
        transform=trans_test
        )
        
        if args.partition=='iid':
            print("=> 正在使用 IID 均匀划分数据...")
            dict_users = cifar_iid(dataset_train, args.num_users)
        elif args.partition=='shard':
            print("=> 正在使用 Shard 分片划分数据...")
            dict_users = cifar_noniid(dataset_train, args.num_users)
        elif args.partition == 'dirichlet':
            print(f"=> 正在使用 Dirichlet 划分数据, alpha={args.alpha}...")
            dict_users = cifar_dirichlet(dataset_train, args.num_users, args.alpha,args.local_bs)
        else:
            exit('Error: unrecognized partition strategy')
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    acc_test_history = [] 

    # === [新增] 初始化资源管理器与李雅普诺夫队列 ===
    resource_mgr = ResourceManager(args.num_users)
    
    # 虚拟队列，记录100个客户端的资源超支情况 (初始为0)
    Q_energy = np.zeros(args.num_users)
    Q_time = np.zeros(args.num_users)
    
    # 【核心超参数：资源红线与 Lyapunov V 值】
    # E_limit 和 T_limit 是系统允许的“每轮平均最高消耗”。
    # 你需要根据打印出的实际开销来调整这两个值，使得 High/Mid 轻松达标，Low 容易超标。
    E_limit = 2.0  
    T_limit = 1.0  
    
    V = 5.0  # 权衡参数：越大越看重模型准确率，越小越看重资源限制

    # === 初始化全局更新方向参考 ===
    flat_w_glob = get_weight_difference(w_glob, w_glob) 
    global_update_dir = torch.zeros_like(flat_w_glob).cpu()

    alpha = 0.8  # SIM的权重
    beta = 0.2   # DIV的权重

    for iter in range(args.epochs):
        
        m = max(int(args.frac * args.num_users), 1)
        pool_size = min(m * 2, args.num_users) 
        
        candidate_idxs = np.random.choice(range(args.num_users), pool_size, replace=False)
        
        candidate_updates = {}
        candidate_w_locals = {}
        candidate_losses = {}
        candidate_lens = {}
        candidate_costs = {} # [新增] 记录候选节点的资源开销

        print(f"\n--- Round {iter} ---")
        
        for idx in candidate_idxs:
            # 获取数据量，提前计算该节点的时延和能耗
            data_size = len(dict_users[idx])
            t_cost, e_cost = resource_mgr.calculate_cost(idx, data_size)
            candidate_costs[idx] = {'time': t_cost, 'energy': e_cost}

            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            
            update_vec = get_weight_difference(w, w_glob).cpu()
            candidate_updates[idx] = update_vec
            
            cpu_w = {k: v.cpu() for k, v in w.items()}
            candidate_w_locals[idx] = cpu_w
            
            candidate_losses[idx] = loss
            candidate_lens[idx] = data_size
            
            torch.cuda.empty_cache()

        # 3. 基于 Lyapunov (SIM + DIV - 资源队列惩罚) 贪心挑选
        selected_idxs = []
        
        if iter == 0 or global_update_dir.norm() == 0:
            selected_idxs = np.random.choice(candidate_idxs, m, replace=False).tolist()
        else:
            sim_scores = {}
            for idx in candidate_idxs:
                sim_scores[idx] = compute_cosine_similarity(candidate_updates[idx], global_update_dir)
            
            remaining_candidates = list(candidate_idxs)
            
            for _ in range(m):
                best_score = -float('inf')
                best_idx = -1
                
                for idx in remaining_candidates:
                    # (1) 准确率收益项
                    utility = alpha * sim_scores[idx]
                    div_penalty = 0.0
                    if len(selected_idxs) > 0:
                        for selected_idx in selected_idxs:
                            div_penalty += compute_cosine_similarity(
                                candidate_updates[idx], candidate_updates[selected_idx]
                            )
                    utility -= beta * div_penalty
                    
                    # (2) 资源惩罚项 (排队越长，开销越大，惩罚越重)
                    e_cost = candidate_costs[idx]['energy']
                    t_cost = candidate_costs[idx]['time']
                    resource_penalty = Q_energy[idx] * e_cost + Q_time[idx] * t_cost
                    
                    # (3) 结合 V 值的最终得分
                    final_score = V * utility - resource_penalty
                    
                    if final_score > best_score:
                        best_score = final_score
                        best_idx = idx
                
                selected_idxs.append(best_idx)
                remaining_candidates.remove(best_idx)
                
        print(f"Selected clients: {selected_idxs}")
        
        # === [新增] 李雅普诺夫队列的演进 (Queue Evolution) ===
        # 无论是否在候选池中，所有 100 个客户端的队列都要更新
        for i in range(args.num_users):
            if i in selected_idxs:
                # 选中的节点：排队长度 = 原长度 + 本轮消耗 - 平均限额
                e_cost = candidate_costs[i]['energy']
                t_cost = candidate_costs[i]['time']
                Q_energy[i] = max(0.0, Q_energy[i] + e_cost - E_limit)
                Q_time[i]   = max(0.0, Q_time[i] + t_cost - T_limit)
            else:
                # 没选中的节点：趁机休息，排队长度缩减
                Q_energy[i] = max(0.0, Q_energy[i] - E_limit)
                Q_time[i]   = max(0.0, Q_time[i] - T_limit)
                
        # 记录公平性 (可选)
        resource_mgr.update_selection(selected_idxs)

        # 4. 提取被选中节点的数据并聚合
        w_locals = [candidate_w_locals[idx] for idx in selected_idxs]
        len_locals = [candidate_lens[idx] for idx in selected_idxs]
        loss_locals = [candidate_losses[idx] for idx in selected_idxs]

        w_glob_new = FedAvg(w_locals, len_locals)
        
        current_global_update = get_weight_difference(w_glob_new, w_glob)
        global_update_dir = 0.9 * global_update_dir + 0.1 * current_global_update

        w_glob = w_glob_new
        net_glob.load_state_dict(w_glob)

        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)

        net_glob.eval() 
        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        acc_test_history.append(acc_test)
        
        # 打印当前轮次的平均队列长度，用于监控系统资源状态
        avg_q_e = np.mean(Q_energy)
        avg_q_t = np.mean(Q_time)
        print('Round {:3d}, Loss {:.3f}, Acc {:.2f}%, Avg Q_E: {:.2f}, Avg Q_T: {:.2f}'.format(
            iter, loss_avg, acc_test, avg_q_e, avg_q_t))
        net_glob.train()