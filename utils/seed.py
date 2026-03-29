import numpy as np
import random
import torch

def set_seed(seed):
    """固定所有随机种子，确保实验可绝对复现"""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # 保证卷积运算的确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False