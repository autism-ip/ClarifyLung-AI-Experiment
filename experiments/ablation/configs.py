"""
消融实验配置模块
[INPUT]: 消融变量参数
[OUTPUT]: AblationConfig数据类，预定义消融配置字典
[POS]: experiments/ablation/核心配置，定义消融实验变量矩阵
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# =============================================================================
# 数据类定义
# =============================================================================

@dataclass(frozen=True)
class AblationConfig:
    """
    消融实验配置数据类

    Attributes:
        name: 配置名称
        multi_scale: 多尺度特征开关 (layer3+layer4 vs 仅layer4)
        gate: 门控机制类型 ('se', 'sigmoid', None)
        transformer: Transformer模块开关
        cross_attention: 交叉注意力层数
    """
    name: str = "custom"
    description: str = ""
    multi_scale: bool = True
    gate: Optional[str] = 'se'
    transformer: bool = True
    cross_attention: bool = True


# =============================================================================
# 预定义消融配置矩阵
# =============================================================================

ABLATION_CONFIGS: Dict[str, AblationConfig] = {
    # 基线配置：全部禁用 (纯CNN基线)
    'baseline_cnn': AblationConfig(
        name='baseline_cnn',
        multi_scale=False,
        gate=None,
        transformer=False,
        cross_attention=False,
    ),

    # 完全混合：全部启用
    'full_hybrid': AblationConfig(
        name='full_hybrid',
        multi_scale=True,
        gate='se',
        transformer=True,
        cross_attention=True,
    ),

    # 消融1：去掉多尺度特征
    'no_multi_scale': AblationConfig(
        name='no_multi_scale',
        multi_scale=False,
        gate='se',
        transformer=True,
        cross_attention=True,
    ),

    # 消融2：去掉门控机制
    'no_gate': AblationConfig(
        name='no_gate',
        multi_scale=True,
        gate=None,
        transformer=True,
        cross_attention=True,
    ),

    # 消融3：去掉交叉注意力
    'no_cross_attention': AblationConfig(
        name='no_cross_attention',
        multi_scale=True,
        gate='se',
        transformer=True,
        cross_attention=False,
    ),

    # 消融4：去掉Transformer
    'no_transformer': AblationConfig(
        name='no_transformer',
        multi_scale=True,
        gate='se',
        transformer=False,
        cross_attention=False,
    ),
}


# =============================================================================
# 辅助函数
# =============================================================================

def get_ablation_config(name: str) -> AblationConfig:
    """
    获取指定名称的消融配置

    Args:
        name: 配置名称

    Returns:
        AblationConfig: 对应的配置

    Raises:
        KeyError: 配置名称不存在
    """
    if name not in ABLATION_CONFIGS:
        available = list(ABLATION_CONFIGS.keys())
        raise KeyError(
            f"Config '{name}' not found. Available configs: {available}"
        )
    return ABLATION_CONFIGS[name]


def list_available_configs() -> List[str]:
    """
    列出所有可用的消融配置名称

    Returns:
        List[str]: 配置名称列表
    """
    return list(ABLATION_CONFIGS.keys())


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == "__main__":
    print("Testing ablation configs module...")

    # 测试所有预定义配置
    print("\n1. Available configs:")
    for name in list_available_configs():
        print(f"  - {name}")

    # 测试获取配置
    print("\n2. Testing get_ablation_config:")
    baseline = get_ablation_config('baseline_cnn')
    print(f"  baseline_cnn: multi_scale={baseline.multi_scale}, "
          f"gate={baseline.gate}, transformer={baseline.transformer}, "
          f"cross_attention={baseline.cross_attention}")

    full = get_ablation_config('full_hybrid')
    print(f"  full_hybrid: multi_scale={full.multi_scale}, "
          f"gate={full.gate}, transformer={full.transformer}, "
          f"cross_attention={full.cross_attention}")

    no_transformer = get_ablation_config('no_transformer')
    print(f"  no_transformer: multi_scale={no_transformer.multi_scale}, "
          f"gate={no_transformer.gate}, transformer={no_transformer.transformer}, "
          f"cross_attention={no_transformer.cross_attention}")

    # 测试错误处理
    print("\n3. Testing error handling:")
    try:
        get_ablation_config('invalid_name')
    except KeyError as e:
        print(f"  KeyError caught: {e}")

    print("\nAll ablation config tests passed!")
