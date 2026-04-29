"""
数据管理模块 - 负责数据加载、增强与可视化
[INPUT]: 原始X光片数据集
[OUTPUT]: DataLoader实例、统计图表、增强后图像
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from .custom_dataset import (
    CustomLungDataset,
    merge_datasets,
    split_dataset,
    create_weighted_sampler,
    LabelSchema,
)
from .augmentation import get_train_augmentation, get_val_augmentation
from .visualization import plot_class_distribution, plot_sample_grid

__all__ = [
    'CustomLungDataset',
    'merge_datasets',
    'split_dataset',
    'create_weighted_sampler',
    'LabelSchema',
    'get_train_augmentation',
    'get_val_augmentation',
    'plot_class_distribution',
    'plot_sample_grid',
]
