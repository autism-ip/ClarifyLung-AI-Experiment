"""
 * [INPUT]: 依赖 training/trainer.py 的 Trainer 与 TrainingConfig
 * [OUTPUT]: 对外提供 Trainer、TrainingConfig 符号
 * [POS]: training/ 的模块入口，被外部训练脚本导入
 * [PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from .trainer import Trainer, TrainingConfig

__all__ = ["Trainer", "TrainingConfig"]
