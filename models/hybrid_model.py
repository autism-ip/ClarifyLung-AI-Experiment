"""
 * [INPUT]: 依赖 models.components 的 feature_extractor, gating, transformer, cross_attention, classification
 * [OUTPUT]: 对外提供 HybridModel
 * [POS]: models/ 的核心模型组装器，组合所有组件构成完整 CNN-Transformer 混合模型
 * [PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import torch
import torch.nn as nn
import torchvision.models as models

from .components.feature_extractor import MutilScaleFeatureExtractor
from .components.gating import GatingMechanism
from .components.transformer import ImagePatchExtractor, TransformerEncoder
from .components.cross_attention import CrossAttention
from .components.classification import ClassificationHead


CLASS_NAMES = ["normal", "malignant", "benign"]


def _get_layer_channels(backbone):
    """根据 backbone 名称返回各层通道数映射"""
    if backbone == 'resnet18':
        return {'layer1': 64, 'layer2': 128, 'layer3': 256, 'layer4': 512}
    # 默认 resnet50
    return {'layer1': 256, 'layer2': 512, 'layer3': 1024, 'layer4': 2048}


class HybridModel(nn.Module):
    """CNN-Transformer 混合模型：多尺度 CNN 特征 + Transformer 全局建模 + 交叉注意力融合"""

    def __init__(
        self,
        backbone_model=None,
        feature_layers=None,
        gate_type='se',
        model_dim=512,
        nhead=8,
        num_layers=6,
        dropout=0.1,
        num_classes=3,
        backbone_name='resnet50'
    ):
        super().__init__()
        if feature_layers is None:
            feature_layers = ['layer3', 'layer4']
        if backbone_model is None:
            backbone_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        self.backbone = backbone_model
        self.mutil_scale_extractor = MutilScaleFeatureExtractor(
            model=self.backbone,
            feature_layers=feature_layers,
            use_multi_scale=True
        )

        layer_channels = _get_layer_channels(backbone_name)
        self.fused_channels = sum(layer_channels[l] for l in feature_layers)

        self.gate_mechanism = GatingMechanism(
            gate_type=gate_type,
            backbone=backbone_name,
            feature_layers=feature_layers
        )

        # Transformer patch stream
        self.image_patch_extractor = ImagePatchExtractor(
            patch_size=16,
            model_dim=model_dim,
            image_size=224
        )
        self.transformer_encoder = TransformerEncoder(
            model_dim=model_dim,
            dropout=dropout,
            nhead=nhead,
            num_layers=num_layers
        )

        self.cnn_to_seq = nn.Linear(self.fused_channels, model_dim)

        self.cross_attention = CrossAttention(
            num_layers=2,
            nhead=nhead,
            model_dim=model_dim,
            dropout=dropout,
            cnn_in_channels=self.fused_channels
        )

        self.classifier = ClassificationHead(
            input_dim=2 * model_dim,
            num_classes=num_classes,
            dropout=dropout
        )

    def forward(self, x):
        # x: [B, 3, H, W]
        B = x.size(0)

        # CNN multi-scale -> [B, fused_channels, Hf, Wf]
        scnn = self.mutil_scale_extractor(x)
        scnn = self.gate_mechanism(scnn)  # gated [B, C, H, W]

        # Convert CNN map -> sequence [B, T_cnn, C] then project to model_dim
        B, C, Hf, Wf = scnn.shape
        scnn_seq = scnn.view(B, C, Hf * Wf).permute(0, 2, 1)  # [B, T_cnn, C]
        scnn_seq_proj = self.cnn_to_seq(scnn_seq)             # [B, T_cnn, model_dim]

        # Transformer path
        stf = self.image_patch_extractor(x)   # [B, T_patch, model_dim]
        stf = self.transformer_encoder(stf)   # [B, T_patch, model_dim]

        # Cross attention: treat queries as sequences
        attncnn, _ = self.cross_attention(scnn_seq_proj, stf, stf)
        attntf, _ = self.cross_attention(stf, scnn_seq_proj, scnn_seq_proj)

        # Pool each branch (global mean over sequence)
        pooled_cnn = attncnn.mean(dim=1)   # [B, model_dim]
        pooled_tf = attntf.mean(dim=1)     # [B, model_dim]

        fused = torch.cat([pooled_cnn, pooled_tf], dim=1)  # [B, 2*model_dim]
        out = self.classifier(fused)                       # [B, num_classes]
        return out
