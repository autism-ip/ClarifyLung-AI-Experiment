"""
 * [INPUT]: 依赖 models.hybrid_model 的 HybridModel, CLASS_NAMES
 * [OUTPUT]: 对外提供 HybridModel, CLASS_NAMES, device, hybrid_model 实例
 * [POS]: 根目录兼容入口，从 models 包 re-export，保留历史调用兼容性
 * [PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import torch
from models.hybrid_model import HybridModel, CLASS_NAMES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

__all__ = ['HybridModel', 'CLASS_NAMES', 'device']
