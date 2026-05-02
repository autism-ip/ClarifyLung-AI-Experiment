"""
消融实验配置模块 - 扩展版本
[INPUT]: 消融变量参数
[OUTPUT]: AblationConfig数据类，预定义消融配置字典
[POS]: experiments/ablation/核心配置，定义消融实验变量矩阵
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md

扩展内容:
1. 正向消融: 逐步添加组件，展示增量贡献
2. 反向消融: 逐步移除组件，验证组件必要性
3. 超参数消融: Transformer层数、注意力头数、模型维度
4. 门控类型消融: SE/Sigmoid/None对比
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# =============================================================================
# 数据类定义
# =============================================================================

@dataclass(frozen=True)
class AblationConfig:
    """
    消融实验配置数据类 - 扩展版本

    Attributes:
        name: 配置名称
        description: 配置描述
        multi_scale: 多尺度特征开关 (layer3+layer4 vs 仅layer4)
        gate: 门控机制类型 ('se', 'sigmoid', None)
        transformer: Transformer模块开关
        cross_attention: 交叉注意力开关
        model_dim: Transformer维度
        nhead: 注意力头数
        num_layers: Transformer层数
        dropout: Dropout率
        category: 配置类别 ('forward', 'reverse', 'hyperparam', 'gate_type')
    """
    name: str = "custom"
    description: str = ""
    multi_scale: bool = True
    gate: Optional[str] = 'se'
    transformer: bool = True
    cross_attention: bool = True
    model_dim: int = 512
    nhead: int = 8
    num_layers: int = 6
    dropout: float = 0.1
    category: str = "custom"


# =============================================================================
# 预定义消融配置矩阵
# =============================================================================

ABLATION_CONFIGS: Dict[str, AblationConfig] = {
    
    # =========================================================================
    # 1. 正向消融: 逐步添加组件，展示增量贡献
    # =========================================================================
    'baseline_cnn': AblationConfig(
        name='baseline_cnn',
        description='仅CNN基线，无Transformer',
        multi_scale=False,
        gate=None,
        transformer=False,
        cross_attention=False,
        category='forward',
    ),
    'multiscale_only': AblationConfig(
        name='multiscale_only',
        description='多尺度特征，无门控',
        multi_scale=True,
        gate=None,
        transformer=False,
        cross_attention=False,
        category='forward',
    ),
    'gating_added': AblationConfig(
        name='gating_added',
        description='多尺度+SE门控',
        multi_scale=True,
        gate='se',
        transformer=False,
        cross_attention=False,
        category='forward',
    ),
    'transformer_added': AblationConfig(
        name='transformer_added',
        description='完整双流架构',
        multi_scale=True,
        gate='se',
        transformer=True,
        cross_attention=False,
        category='forward',
    ),
    'full_hybrid': AblationConfig(
        name='full_hybrid',
        description='完整混合架构+交叉注意力',
        multi_scale=True,
        gate='se',
        transformer=True,
        cross_attention=True,
        category='forward',
    ),

    # =========================================================================
    # 2. 反向消融: 逐步移除组件，验证组件必要性
    # =========================================================================
    'remove_cross_attention': AblationConfig(
        name='remove_cross_attention',
        description='完整模型去掉交叉注意力',
        multi_scale=True,
        gate='se',
        transformer=True,
        cross_attention=False,
        category='reverse',
    ),
    'remove_transformer': AblationConfig(
        name='remove_transformer',
        description='完整模型去掉Transformer',
        multi_scale=True,
        gate='se',
        transformer=False,
        cross_attention=False,
        category='reverse',
    ),
    'remove_gate': AblationConfig(
        name='remove_gate',
        description='完整模型去掉门控',
        multi_scale=True,
        gate=None,
        transformer=True,
        cross_attention=True,
        category='reverse',
    ),
    'remove_multiscale': AblationConfig(
        name='remove_multiscale',
        description='完整模型去掉多尺度',
        multi_scale=False,
        gate='se',
        transformer=True,
        cross_attention=True,
        category='reverse',
    ),

    # =========================================================================
    # 3. 超参数消融: Transformer层数
    # =========================================================================
    'transformer_2layers': AblationConfig(
        name='transformer_2layers',
        description='Transformer 2层',
        multi_scale=True,
        gate='se',
        transformer=True,
        cross_attention=True,
        num_layers=2,
        category='hyperparam',
    ),
    'transformer_4layers': AblationConfig(
        name='transformer_4layers',
        description='Transformer 4层',
        multi_scale=True,
        gate='se',
        transformer=True,
        cross_attention=True,
        num_layers=4,
        category='hyperparam',
    ),
    'transformer_8layers': AblationConfig(
        name='transformer_8layers',
        description='Transformer 8层',
        multi_scale=True,
        gate='se',
        transformer=True,
        cross_attention=True,
        num_layers=8,
        category='hyperparam',
    ),

    # =========================================================================
    # 4. 超参数消融: 注意力头数
    # =========================================================================
    'nhead_4': AblationConfig(
        name='nhead_4',
        description='注意力头数=4',
        multi_scale=True,
        gate='se',
        transformer=True,
        cross_attention=True,
        nhead=4,
        category='hyperparam',
    ),
    'nhead_16': AblationConfig(
        name='nhead_16',
        description='注意力头数=16',
        multi_scale=True,
        gate='se',
        transformer=True,
        cross_attention=True,
        nhead=16,
        category='hyperparam',
    ),

    # =========================================================================
    # 5. 超参数消融: 模型维度
    # =========================================================================
    'model_dim_256': AblationConfig(
        name='model_dim_256',
        description='模型维度=256',
        multi_scale=True,
        gate='se',
        transformer=True,
        cross_attention=True,
        model_dim=256,
        category='hyperparam',
    ),
    'model_dim_768': AblationConfig(
        name='model_dim_768',
        description='模型维度=768',
        multi_scale=True,
        gate='se',
        transformer=True,
        cross_attention=True,
        model_dim=768,
        category='hyperparam',
    ),

    # =========================================================================
    # 6. 门控类型消融
    # =========================================================================
    'gate_sigmoid': AblationConfig(
        name='gate_sigmoid',
        description='Sigmoid门控',
        multi_scale=True,
        gate='sigmoid',
        transformer=True,
        cross_attention=True,
        category='gate_type',
    ),
    'gate_none': AblationConfig(
        name='gate_none',
        description='无门控机制',
        multi_scale=True,
        gate=None,
        transformer=True,
        cross_attention=True,
        category='gate_type',
    ),
}


# =============================================================================
# 配置分组（方便实验选择）
# =============================================================================

ABLATION_GROUPS = {
    'forward': ['baseline_cnn', 'multiscale_only', 'gating_added', 'transformer_added', 'full_hybrid'],
    'reverse': ['full_hybrid', 'remove_cross_attention', 'remove_transformer', 'remove_gate', 'remove_multiscale'],
    'transformer_layers': ['transformer_2layers', 'transformer_4layers', 'full_hybrid', 'transformer_8layers'],
    'attention_heads': ['nhead_4', 'full_hybrid', 'nhead_16'],
    'model_dim': ['model_dim_256', 'full_hybrid', 'model_dim_768'],
    'gate_type': ['gate_none', 'gate_sigmoid', 'full_hybrid'],
    'quick_test': ['baseline_cnn', 'full_hybrid'],
}


# =============================================================================
# 辅助函数
# =============================================================================

def get_ablation_config(name: str) -> AblationConfig:
    """获取指定名称的消融配置"""
    if name not in ABLATION_CONFIGS:
        available = list(ABLATION_CONFIGS.keys())
        raise KeyError(f"Config '{name}' not found. Available configs: {available}")
    return ABLATION_CONFIGS[name]


def get_configs_by_group(group: str) -> List[AblationConfig]:
    """获取指定分组的所有配置"""
    if group not in ABLATION_GROUPS:
        available = list(ABLATION_GROUPS.keys())
        raise KeyError(f"Group '{group}' not found. Available groups: {available}")
    return [ABLATION_CONFIGS[name] for name in ABLATION_GROUPS[group]]


def get_configs_by_category(category: str) -> List[AblationConfig]:
    """获取指定类别的所有配置"""
    return [cfg for cfg in ABLATION_CONFIGS.values() if cfg.category == category]


def list_available_configs() -> List[str]:
    """列出所有可用的消融配置名称"""
    return list(ABLATION_CONFIGS.keys())


def list_available_groups() -> List[str]:
    """列出所有可用的配置分组"""
    return list(ABLATION_GROUPS.keys())


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == "__main__":
    print("Testing ablation configs module...")

    # 测试所有预定义配置
    print("\n1. Available configs by category:")
    for category in ['forward', 'reverse', 'hyperparam', 'gate_type']:
        configs = get_configs_by_category(category)
        print(f"  {category}: {[c.name for c in configs]}")

    # 测试配置分组
    print("\n2. Available groups:")
    for group in list_available_groups():
        print(f"  {group}: {ABLATION_GROUPS[group]}")

    # 测试获取配置
    print("\n3. Testing get_ablation_config:")
    baseline = get_ablation_config('baseline_cnn')
    print(f"  baseline_cnn: multi_scale={baseline.multi_scale}, "
          f"gate={baseline.gate}, transformer={baseline.transformer}, "
          f"cross_attention={baseline.cross_attention}")

    full = get_ablation_config('full_hybrid')
    print(f"  full_hybrid: multi_scale={full.multi_scale}, "
          f"gate={full.gate}, transformer={full.transformer}, "
          f"cross_attention={full.cross_attention}, "
          f"num_layers={full.num_layers}")

    print("\nAll ablation config tests passed!")