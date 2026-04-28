"""
实验脚本公共工具函数
[INPUT]: torch, numpy
[OUTPUT]: set_seed, get_device
[POS]: scripts/ 的共享工具模块，被所有实验脚本消费
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import torch
import numpy as np


def set_seed(seed: int):
    """设置全局随机种子，保证实验可复现"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def get_device():
    """获取最优计算设备 (GPU优先)"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print(f"[INFO] Using CPU")
    return device
