"""
[INPUT]: torch, timm
[OUTPUT]: BaselineModelFactory for creating benchmark models
[POS]: experiments/benchmark/ model factory
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import torch
import torch.nn as nn
import timm


# =============================================================================
# ResNet50 - 经典CNN基准模型
# =============================================================================


def create_resnet50(num_classes: int = 3) -> nn.Module:
    """
    创建预训练 ResNet50 模型

    Args:
        num_classes: 输出类别数，默认3 (normal/malignant/benign)

    Returns:
        预训练 ResNet50 模型
    """
    model = timm.create_model(
        "resnet50",
        pretrained=True,
        num_classes=num_classes,
    )
    return model


# =============================================================================
# ViT - Vision Transformer 基准模型
# =============================================================================


def create_vit(num_classes: int = 3) -> nn.Module:
    """
    创建预训练 Vision Transformer (ViT-B/16) 模型

    Args:
        num_classes: 输出类别数，默认3

    Returns:
        预训练 ViT-B/16 模型
    """
    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=True,
        num_classes=num_classes,
    )
    return model


# =============================================================================
# Hybrid Basic - 简化的CNN-Transformer混合模型
# =============================================================================


class HybridBasic(nn.Module):
    """
    简化版CNN-Transformer混合模型
    - CNN: ResNet50 局部特征提取
    - Transformer: 单层自注意力
    - 直接拼接后分类
    """

    def __init__(self, num_classes: int = 3, model_dim: int = 256):
        super().__init__()
        # CNN特征提取器
        self.cnn = timm.create_model("resnet50", pretrained=True, features_only=True)
        cnn_channels = 2048  # ResNet50 layer4 output channels

        # 投影层
        self.proj = nn.Linear(cnn_channels, model_dim)

        # 简单Transformer (仅自注意力)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=8,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(model_dim * 2, num_classes),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN分支
        cnn_feat = self.cnn(x)[-1]  # [B, 2048, 7, 7]
        B, C, H, W = cnn_feat.shape
        cnn_seq = cnn_feat.view(B, C, H * W).permute(0, 2, 1)  # [B, 49, 2048]
        cnn_proj = self.proj(cnn_seq)  # [B, 49, 256]

        # 全局池化
        cnn_pooled = cnn_proj.mean(dim=1)  # [B, 256]

        # Transformer分支
        trans_feat = self.transformer(cnn_proj)  # [B, 49, 256]
        trans_pooled = trans_feat.mean(dim=1)  # [B, 256]

        # 拼接融合
        fused = torch.cat([cnn_pooled, trans_pooled], dim=1)  # [B, 512]

        return self.classifier(fused)


def create_hybrid_basic(num_classes: int = 3) -> nn.Module:
    """
    创建简化版混合模型

    Args:
        num_classes: 输出类别数，默认3

    Returns:
        HybridBasic 模型实例
    """
    return HybridBasic(num_classes=num_classes)


# =============================================================================
# Hybrid Advanced - 完整CNN-Transformer混合模型 (参考 model.py)
# =============================================================================


class HybridAdvanced(nn.Module):
    """
    完整版CNN-Transformer混合模型
    - 多尺度CNN特征提取
    - 门控机制
    - 交叉注意力融合
    - 预训练backbone
    """

    def __init__(
        self,
        num_classes: int = 3,
        model_dim: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        # 预训练ResNet50 backbone
        self.backbone = timm.create_model("resnet50", pretrained=True)
        # timm uses act1 instead of relu in newer versions
        act_layer = getattr(self.backbone, 'relu', getattr(self.backbone, 'act1', None))
        self.feature_extractor = nn.ModuleDict(
            {
                "conv1": nn.Sequential(
                    self.backbone.conv1,
                    self.backbone.bn1,
                    act_layer,
                    self.backbone.maxpool,
                ) if act_layer else nn.Sequential(
                    self.backbone.conv1,
                    self.backbone.bn1,
                    self.backbone.maxpool,
                ),
                "layer1": self.backbone.layer1,
                "layer2": self.backbone.layer2,
                "layer3": self.backbone.layer3,
                "layer4": self.backbone.layer4,
            }
        )

        # 多尺度特征融合
        self.layer_channels = {"layer1": 256, "layer2": 512, "layer3": 1024, "layer4": 2048}
        self.fused_channels = sum(self.layer_channels.values())
        self.target_spatial_size = (7, 7)

        # 门控机制 (SE-Net风格)
        self.gate_reduce = max(self.fused_channels // 16, 1)
        self.gate_excitation = nn.Sequential(
            nn.Linear(self.fused_channels, self.gate_reduce),
            nn.ReLU(inplace=True),
            nn.Linear(self.gate_reduce, self.fused_channels),
            nn.Sigmoid(),
        )

        # Transformer路径
        self.patch_proj = nn.Linear(16 * 16 * 3, model_dim)
        self.pos_encoder = nn.Parameter(torch.randn(1, 196, model_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=nhead,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 投影层
        self.cnn_proj = nn.Linear(self.fused_channels, model_dim)

        # 交叉注意力
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(model_dim)

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)

        # ---- CNN 多尺度特征提取 ----
        feats = []
        out = x
        for name, layer in self.feature_extractor.items():
            out = layer(out)
            if name in self.layer_channels:
                feats.append(out)

        # 统一尺寸并拼接
        resized = []
        for f in feats:
            resized.append(
                torch.nn.functional.interpolate(
                    f, size=self.target_spatial_size, mode="bilinear", align_corners=False
                )
            )
        fused = torch.cat(resized, dim=1)  # [B, fused_channels, 7, 7]

        # 门控
        gate_input = fused.mean(dim=[2, 3])  # [B, fused_channels]
        gate_weights = self.gate_excitation(gate_input)  # [B, fused_channels]
        gated = fused * gate_weights.unsqueeze(2).unsqueeze(3)

        # CNN序列表示
        B, C, H, W = gated.shape
        cnn_seq = gated.view(B, C, H * W).permute(0, 2, 1)  # [B, 49, fused_channels]
        cnn_proj = self.cnn_proj(cnn_seq)  # [B, 49, model_dim]

        # ---- Transformer 路径 ----
        # 图像分块
        patches = torch.nn.functional.unfold(x, kernel_size=16, stride=16)  # [B, 768, 196]
        patches = patches.permute(0, 2, 1)  # [B, 196, 768]
        trans_seq = self.patch_proj(patches)  # [B, 196, model_dim]
        trans_seq = trans_seq + self.pos_encoder

        # Transformer编码
        trans_feat = self.transformer(trans_seq)  # [B, 196, model_dim]

        # ---- 交叉注意力融合 ----
        # CNN关注Transformer
        attn_cnn, _ = self.cross_attn(cnn_proj, trans_feat, trans_feat)
        cnn_fused = self.cross_norm(cnn_proj + attn_cnn)

        # Transformer关注CNN
        attn_trans, _ = self.cross_attn(trans_feat, cnn_proj, cnn_proj)
        trans_fused = self.cross_norm(trans_feat + attn_trans)

        # 全局池化
        cnn_pooled = cnn_fused.mean(dim=1)  # [B, model_dim]
        trans_pooled = trans_fused.mean(dim=1)  # [B, model_dim]

        # 拼接分类
        fused_out = torch.cat([cnn_pooled, trans_pooled], dim=1)  # [B, 2*model_dim]

        return self.classifier(fused_out)


def create_hybrid_advanced(num_classes: int = 3) -> nn.Module:
    """
    创建完整版混合模型

    Args:
        num_classes: 输出类别数，默认3

    Returns:
        HybridAdvanced 模型实例
    """
    return HybridAdvanced(num_classes=num_classes)
