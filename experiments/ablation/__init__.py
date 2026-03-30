# =============================================================================
# ablation Package
# =============================================================================
"""
[INPUT]: 依赖 torch, typing, dataclass
[OUTPUT]: AblationStudy, AblationConfig, ABLATION_CONFIGS
[POS]: experiments/ablation/ 消融实验框架
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from .configs import (
    AblationConfig,
    ABLATION_CONFIGS,
    get_ablation_config,
    list_available_configs,
)
from .ablator import AblationStudy

__all__ = [
    'AblationStudy',
    'AblationConfig',
    'ABLATION_CONFIGS',
    'get_ablation_config',
    'list_available_configs',
]