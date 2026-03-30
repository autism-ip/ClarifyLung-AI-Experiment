"""
configs/ - 配置文件模块
[INPUT]: 环境变量、默认路径
[OUTPUT]: 配置对象和常量
[POS]: configs/ 配置集中管理
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from .dataset_config import (
    DATASET_PATHS,
    DATASET_INFO,
    LABEL_TO_INDEX,
    INDEX_TO_LABEL,
    CLASS_NAMES,
    validate_paths,
)

__all__ = [
    "DATASET_PATHS",
    "DATASET_INFO",
    "LABEL_TO_INDEX",
    "INDEX_TO_LABEL",
    "CLASS_NAMES",
    "validate_paths",
]
