"""
 * [INPUT]: 依赖 models.hybrid_model 的 HybridModel
 * [OUTPUT]: 对外提供 HybridModel, CLASS_NAMES
 * [POS]: models/ 包的入口，统一暴露模型符号
 * [PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from .hybrid_model import HybridModel, CLASS_NAMES
from .configurable_hybrid import ConfigurableHybrid

__all__ = ['HybridModel', 'ConfigurableHybrid', 'CLASS_NAMES']
