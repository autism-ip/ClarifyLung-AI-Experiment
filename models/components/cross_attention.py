"""
 * [INPUT]: 依赖 torch.nn
 * [OUTPUT]: 对外提供 CrossAttentionLayer, CrossAttention
 * [POS]: models/components 的交叉注意力模块，被 hybrid_model.HybridModel 消费
 * [PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import torch
import torch.nn as nn


class CrossAttentionLayer(nn.Module):
    """单层交叉注意力：支持序列 [B, T, D] 或 CNN 特征图 [B, C, H, W] 输入"""

    def __init__(
        self,
        nhead=8,
        model_dim=256,
        dropout=0.1,
        cnn_in_channels=None
    ):
        super().__init__()
        self.nhead = nhead
        self.model_dim = model_dim
        self.head_dim = model_dim // nhead
        assert self.head_dim * nhead == model_dim, "model_dim must be divisible by nhead"
        self.cnn_in_channels = cnn_in_channels

        if cnn_in_channels is not None:
            self.cnn_proj = nn.Linear(cnn_in_channels, model_dim)

        self.w_q = nn.Linear(model_dim, model_dim)
        self.w_k = nn.Linear(model_dim, model_dim)
        self.w_v = nn.Linear(model_dim, model_dim)
        self.w_o = nn.Linear(model_dim, model_dim)
        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(model_dim)

    def _prepare(self, x):
        # Accept sequence [B, T, D] or CNN map [B, C, H, W]
        if x.dim() == 3:
            # [B, T, D]
            if x.size(-1) != self.model_dim:
                proj = nn.Linear(x.size(-1), self.model_dim).to(x.device)
                x = proj(x)
            return x
        elif x.dim() == 4:
            B, C, H, W = x.shape
            x_seq = x.view(B, C, H * W).permute(0, 2, 1)  # [B, H*W, C]
            if self.cnn_in_channels is None:
                raise RuntimeError(
                    "CrossAttentionLayer expects cnn_in_channels for 4D inputs"
                )
            x_seq = self.cnn_proj(x_seq)  # [B, H*W, model_dim]
            return x_seq
        else:
            raise ValueError("Unsupported input dim for CrossAttentionLayer")

    def forward(self, q, k, v):
        q_p = self._prepare(q)
        k_p = self._prepare(k)
        v_p = self._prepare(v)
        B = q_p.size(0)

        q_lin = self.w_q(q_p)
        k_lin = self.w_k(k_p)
        v_lin = self.w_v(v_p)

        qh = q_lin.view(B, -1, self.nhead, self.head_dim).transpose(1, 2)
        kh = k_lin.view(B, -1, self.nhead, self.head_dim).transpose(1, 2)
        vh = v_lin.view(B, -1, self.nhead, self.head_dim).transpose(1, 2)

        scores = torch.matmul(qh, kh.transpose(-2, -1)) * self.scale
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, vh)  # [B, nhead, Tq, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, -1, self.nhead * self.head_dim)
        out = self.w_o(out)
        out = self.norm(q_lin + out)
        return out, attn


class CrossAttention(nn.Module):
    """多层交叉注意力：堆叠多个 CrossAttentionLayer"""

    def __init__(
        self,
        num_layers=2,
        nhead=8,
        model_dim=256,
        dropout=0.1,
        cnn_in_channels=None
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossAttentionLayer(
                nhead=nhead,
                model_dim=model_dim,
                dropout=dropout,
                cnn_in_channels=cnn_in_channels
            )
            for _ in range(num_layers)
        ])

    def forward(self, query, key, value):
        out = query
        attn = None
        for layer in self.layers:
            out, attn = layer(out, key, value)
        return out, attn
