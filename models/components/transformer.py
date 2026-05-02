"""
 * [INPUT]: 依赖 torch.nn, math
 * [OUTPUT]: 对外提供 PositionEncoder, ImagePatchExtractor, TransformerEncoder
 * [POS]: models/components 的 Transformer 编码器模块，被 hybrid_model.HybridModel 消费
 * [PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionEncoder(nn.Module):
    """正弦位置编码，为序列注入位置信息"""

    def __init__(self, model_dim=512, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = model_dim

        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, model_dim, 2).float()
            * (-math.log(10000.0) / model_dim)
        ).unsqueeze(0)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, model_dim]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [B, T, D]
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return self.dropout(x)


class ImagePatchExtractor(nn.Module):
    """图像分块与线性投影：将图像切分为 patch 并投影到模型维度"""

    def __init__(self, patch_size=16, model_dim=512, image_size=224):
        super().__init__()
        self.patch_size = patch_size
        self.model_dim = model_dim
        self.image_size = image_size
        self.num_patches = (image_size // patch_size) ** 2
        self.projection = nn.Linear(patch_size * patch_size * 3, model_dim)

    def _extract_patches(self, x):
        # x: [B, 3, H, W]
        unfolded = F.unfold(
            x,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )  # [B, patch_flat, num_patches]
        unfolded = unfolded.permute(0, 2, 1)  # [B, num_patches, patch_flat]
        return unfolded

    def forward(self, x):
        patches = self._extract_patches(x)       # [B, num_patches, patch_flat]
        projected = self.projection(patches)     # [B, num_patches, model_dim]
        return projected


class TransformerEncoder(nn.Module):
    """Transformer 编码器：位置编码 + 多层 TransformerEncoderLayer"""

    def __init__(
        self,
        model_dim=512,
        dropout=0.1,
        nhead=8,
        dim_feedforward=3072,
        num_layers=6
    ):
        super().__init__()
        self.model_dim = model_dim
        self.pos_encoder = PositionEncoder(model_dim=model_dim, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )

    def forward(self, x):
        # x: [B, T, D]
        x = self.pos_encoder(x)
        return self.transformer(x)  # [B, T, D]
