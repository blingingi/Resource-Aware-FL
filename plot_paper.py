import numpy as np
import matplotlib.pyplot as plt

# 字体与格式设置 (使其具备学术论文质感)
plt.rcParams.update({'font.size': 14, 'font.family': 'serif'})

# 1. 填入你的文件名（已加上 save/ 路径前缀）
f_base_01 = "save/fed_main_base_cifar_dirichlet_alpha0.1_ep200_20260227_181914_acc.npy"
f_litong_01 = "save/fed_main_litong_cifar_dirichlet_alpha0.1_ep200_20260227_184446_acc.npy"
f_lya_01 = "save/fed_main_lya_cifar_dirichlet_alpha0.1_ep200_V5.0_20260227_184416_acc.npy"

f_base_05 = "save/fed_main_base_cifar_dirichlet_alpha0.5_ep200_20260227_201515_acc.npy"
f_litong_05 = "save/fed_main_litong_cifar_dirichlet_alpha0.5_ep200_20260227_203802_acc.npy"
f_lya_05 = "save/fed_main_lya_cifar_dirichlet_alpha0.5_ep200_V5.0_20260227_211834_acc.npy"

# 2. 加载数据并使用平滑处理 (针对 Base 的剧烈震荡，使用轻微的移动平均使其更好看)
def smooth(data, weight=0.6):
    """基于指数移动平均的平滑函数，保留整体趋势"""
    smoothed = np.zeros_like(data)
    smoothed[0] = data[0]
    for t in range(1, len(data)):
        smoothed[t] = smoothed[t-1] * weight + data[t] * (1 - weight)
    return smoothed

d_base_01 = smooth(np.load(f_base_01))
d_litong_01 = smooth(np.load(f_litong_01))
d_lya_01 = smooth(np.load(f_lya_01))

d_base_05 = smooth(np.load(f_base_05))
d_litong_05 = smooth(np.load(f_litong_05))
d_lya_05 = smooth(np.load(f_lya_05))

epochs = np.arange(200)

def plot_figure(data_base, data_litong, data_lya, title, save_name):
    plt.figure(figsize=(9, 6))
    
    # 配色方案：经典科技蓝、自然绿、警示红
    plt.plot(epochs, data_base, label='Baseline (FedAvg)', color='#1f77b4', linestyle='--', linewidth=2, alpha=0.8)
    plt.plot(epochs, data_litong, label='Ours (Ideal, No Resource Constraint)', color='#2ca02c', linewidth=2, alpha=0.8)
    plt.plot(epochs, data_lya, label='Ours (Practical, with Lyapunov)', color='#d62728', linewidth=2.5)
    
    plt.title(title, fontweight='bold', pad=15)
    plt.xlabel('Communication Rounds', fontweight='bold')
    plt.ylabel('Test Accuracy (%)', fontweight='bold')
    
    plt.legend(loc='lower right', frameon=True, shadow=True, edgecolor='black')
    plt.grid(True, linestyle=':', linewidth=1.5, alpha=0.5)
    
    # 限制 y 轴下限，让曲线变化更明显
    plt.ylim(bottom=10) 
    
    plt.tight_layout()
    plt.savefig(save_name, dpi=600, bbox_inches='tight') # 600 DPI 满足绝大多数期刊要求
    print(f"✅ 成功生成学术图表: {save_name}")
    plt.close()

# 3. 绘制并保存图片
plot_figure(d_base_01, d_litong_01, d_lya_01, 'Test Accuracy on CIFAR-10 (Dirichlet $\\alpha=0.1$)', 'paper_fig_alpha01.pdf')
plot_figure(d_base_05, d_litong_05, d_lya_05, 'Test Accuracy on CIFAR-10 (Dirichlet $\\alpha=0.5$)', 'paper_fig_alpha05.pdf')