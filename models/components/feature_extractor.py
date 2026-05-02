"""
 * [INPUT]: 依赖 torch.nn, torchvision.models, collections.OrderedDict, torch.nn.functional
 * [OUTPUT]: 对外提供 FeatureFusion, MutilScaleFeatureExtractor
 * [POS]: models/components 的特征提取器，被 hybrid_model.HybridModel 消费
 * [PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md

 公平对比实验: 所有模型都从头训练，不使用预训练权重
"""

import torch
import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict
import torch.nn.functional as F


class FeatureFusion(nn.Module):
    """多尺度特征融合：将不同层特征 resize 到统一空间尺寸后通道拼接"""

    def __init__(self, target_spatial_size=(7, 7)):
        super().__init__()
        self.target_spatial_size = target_spatial_size

    def forward(self, feature_list):
        # feature_list: list of [B, C_i, H_i, W_i]
        resized = []
        for f in feature_list:
            resized.append(
                F.interpolate(
                    f,
                    size=self.target_spatial_size,
                    mode='bilinear',
                    align_corners=False
                )
            )
        # concat on channel dim
        return torch.cat(resized, dim=1)  # [B, sum(C_i), H, W]


class MutilScaleFeatureExtractor(nn.Module):
    """CNN 多尺度特征提取器，基于 ResNet backbone，支持多尺度特征融合"""

    def __init__(
        self,
        model=None,
        feature_layers=None,
        use_multi_scale=True,
        target_spatial_size=(7, 7),
        pretrained=False  # 默认不使用预训练权重（公平对比）
    ):
        super().__init__()
        if feature_layers is None:
            feature_layers = ['layer3', 'layer4']
        if model is None:
            model = models.resnet50(pretrained=pretrained)

        self.feature_layers = feature_layers
        self.model = model
        self.use_multi_scale = use_multi_scale
        self.target_spatial_size = target_spatial_size

        self._feature_extractor = nn.ModuleDict(OrderedDict([
            ('conv1', nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)),
            ('layer1', model.layer1),
            ('layer2', model.layer2),
            ('layer3', model.layer3),
            ('layer4', model.layer4)
        ]))

        if use_multi_scale:
            self.fusion = FeatureFusion(target_spatial_size)

    def forward(self, x):
        feats = []
        out = x
        for name, layer in self._feature_extractor.items():
            out = layer(out)
            if name in self.feature_layers:
                feats.append(out)
        if self.use_multi_scale:
            return self.fusion(feats)  # [B, C_sum, H, W]
        else:
            return feats[0]
