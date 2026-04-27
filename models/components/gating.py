"""
 * [INPUT]: 依赖 torch.nn
 * [OUTPUT]: 对外提供 GatingMechanism
 * [POS]: models/components 的门控机制模块，被 hybrid_model.HybridModel 消费
 * [PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import torch
import torch.nn as nn


class GatingMechanism(nn.Module):
    """特征门控机制：SE 风格或 Sigmoid 风格通道注意力"""

    def __init__(
        self,
        gate_type='se',
        reduction_ratio=16,
        backbone='resnet50',
        feature_layers=None
    ):
        super().__init__()
        if feature_layers is None:
            feature_layers = ['layer3', 'layer4']

        self.gate_type = gate_type

        # channel maps for common ResNets
        if backbone == 'resnet18':
            layer_channels = {'layer1': 64, 'layer2': 128, 'layer3': 256, 'layer4': 512}
        elif backbone == 'resnet50':
            layer_channels = {'layer1': 256, 'layer2': 512, 'layer3': 1024, 'layer4': 2048}
        else:
            raise ValueError("Unsupported backbone")

        # input_dim is channel count (sum of selected layers)
        self.input_dim = sum(layer_channels[l] for l in feature_layers)

        if gate_type == 'sigmoid':
            self.gate_linear = nn.Linear(self.input_dim, self.input_dim)
            self.sigmoid = nn.Sigmoid()
        else:
            self.reduce_dim = max(self.input_dim // reduction_ratio, 1)
            self.excitation = nn.Sequential(
                nn.Linear(self.input_dim, self.reduce_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.reduce_dim, self.input_dim),
                nn.Sigmoid()
            )

    def forward(self, x):
        # x: [B, C, H, W]
        s = torch.mean(x, dim=[2, 3])  # [B, C]

        if self.gate_type == 'sigmoid':
            g = self.sigmoid(self.gate_linear(s))
            return x * g.unsqueeze(2).unsqueeze(3)
        else:
            scale = self.excitation(s)  # [B, C]
            return x * scale.unsqueeze(2).unsqueeze(3)
