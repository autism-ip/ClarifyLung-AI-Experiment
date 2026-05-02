"""
 * [INPUT]: 依赖 models.hybrid_model 的 HybridModel
 * [OUTPUT]: 对外提供 ConfigurableHybrid —— 支持消融实验配置的参数化包装器
 * [POS]: models/ 的消融实验适配器，将 AblationConfig 映射到 HybridModel 参数
 * [PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import torch.nn as nn

from .hybrid_model import HybridModel


class ConfigurableHybrid(nn.Module):
    """
    消融实验专用参数化混合模型

    将 AblationStudy 所需的布尔开关参数 (multi_scale, gate, transformer,
    cross_attention) 映射为 HybridModel 的具体构造参数，确保消融实验
    在真实组件上进行，而非独立实现的简化模型。

    Args:
        multi_scale: 是否使用多尺度特征 (layer3+layer4)
        gate: 门控类型 ('se', 'sigmoid', None)
        transformer: 是否启用 Transformer 模块
        cross_attention: 是否启用交叉注意力
        num_classes: 分类数
        model_dim: Transformer 维度
        nhead: 注意力头数
        num_layers: Transformer 层数
        dropout: Dropout 率
    """

    def __init__(
        self,
        multi_scale: bool = True,
        gate: str = 'se',
        transformer: bool = True,
        cross_attention: bool = True,
        num_classes: int = 3,
        model_dim: int = 768,
        nhead: int = 12,
        num_layers: int = 12,
        dropout: float = 0.1,
        pretrained: bool = True
    ):
        super().__init__()

        feature_layers = ['layer3', 'layer4'] if multi_scale else ['layer4']
        gate_type = gate if gate is not None else None

        self.model = HybridModel(
            feature_layers=feature_layers,
            gate_type=gate_type,
            model_dim=model_dim,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
            num_classes=num_classes,
            use_transformer=transformer,
            use_cross_attention=cross_attention,
            pretrained=pretrained
        )

    def forward(self, x):
        return self.model(x)
