# =============================================================================
# Attention Map Visualization Module
# =============================================================================
"""
[INPUT]: torch, numpy, matplotlib
[OUTPUT]: AttentionVisualizer, visualize_attention
[POS]: experiments/visualization/ Attention map extraction and visualization
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import List, Optional


# =============================================================================
# Attention Visualizer
# =============================================================================
class AttentionVisualizer:
    """
    注意力图可视化工具类
    用于提取和可视化Transformer的注意力权重

    [功能]:
    - 提取MultiheadAttention层的注意力权重
    - 支持自定义注意力池化策略
    - 生成可解释的注意力可视化
    """

    def __init__(self, model: nn.Module):
        """
        初始化注意力可视化器

        Args:
            model: 包含MultiheadAttention的PyTorch模型
        """
        self.model = model
        self.model.eval()
        self.attention_layers = self._find_attention_layers()

    def has_attention(self) -> bool:
        """
        检查模型是否包含注意力层

        Returns:
            bool: 是否有注意力层
        """
        return len(self.attention_layers) > 0

    def _find_attention_layers(self) -> List[str]:
        """
        查找模型中所有的MultiheadAttention层

        Returns:
            注意力层名称列表
        """
        attention_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                attention_layers.append(name)
        return attention_layers

    def extract_attention(
        self,
        input_tensor: torch.Tensor,
        layer_idx: int = 0
    ) -> List[np.ndarray]:
        """
        从指定注意力层提取注意力权重

        [机制]:
        - 使用hook机制注册前向传播钩子
        - 捕获MultiheadAttention的原始注意力权重
        - 返回所有head的注意力权重平均值

        Args:
            input_tensor: 输入张量，形状 (seq_len, batch, embed_dim)
            layer_idx: 要提取的注意力层索引

        Returns:
            注意力权重列表，每个元素形状 (seq_len, seq_len)
        """
        if not self.attention_layers:
            raise ValueError("Model does not contain any MultiheadAttention layer")

        if layer_idx >= len(self.attention_layers):
            raise ValueError(f"Layer index {layer_idx} out of range")

        layer_name = self.attention_layers[layer_idx]
        attention_weights = []

        def hook_fn(module, input, output):
            # output: (attn_output, attn_weights)
            # attn_weights: (batch, num_heads, seq_len, seq_len)
            _, attn_weights = output
            # 取第一个batch的平均
            avg_weights = attn_weights[0].mean(dim=0).cpu().numpy()
            attention_weights.append(avg_weights)

        # 获取目标层
        target_layer = dict(self.model.named_modules())[layer_name]

        # 注册钩子
        handle = target_layer.register_forward_hook(hook_fn)

        try:
            with torch.no_grad():
                # 前向传播
                _ = self.model(input_tensor)
        finally:
            # 移除钩子
            handle.remove()

        if not attention_weights:
            raise RuntimeError("Failed to extract attention weights")

        return attention_weights

    def extract_all_attention(
        self,
        input_tensor: torch.Tensor
    ) -> List[List[np.ndarray]]:
        """
        提取所有注意力层的权重

        Args:
            input_tensor: 输入张量

        Returns:
            所有层的注意力权重列表
        """
        all_attention = []

        for i in range(len(self.attention_layers)):
            try:
                weights = self.extract_attention(input_tensor, layer_idx=i)
                all_attention.append(weights)
            except Exception:
                continue

        return all_attention


# =============================================================================
# Attention Visualization Functions
# =============================================================================
def visualize_attention(
    attention_matrix: np.ndarray,
    figsize: tuple = (8, 8),
    cmap: str = 'viridis',
    save_path: Optional[str] = None
) -> np.ndarray:
    """
    可视化单个注意力矩阵

    [参数]:
    - attention_matrix: 注意力权重矩阵，形状 (seq_len, seq_len)
    - figsize: 图像尺寸
    - cmap: 色彩映射
    - save_path: 保存路径，None则不保存

    [返回]:
    - 可视化后的图像数组
    """
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(attention_matrix, cmap=cmap, aspect='auto')

    ax.set_xlabel('Key Position')
    ax.set_ylabel('Query Position')
    ax.set_title('Attention Weights')

    plt.colorbar(im, ax=ax)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    # 转换为numpy数组 (使用buffer而非tostring_rgb，兼容macOS)
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)
    img = img[:, :, :3]  # 移除alpha通道，只保留RGB

    plt.close(fig)

    return img


def visualize_multihead_attention(
    attention_weights: np.ndarray,
    num_heads: int,
    figsize: tuple = (16, 4),
    save_path: Optional[str] = None
) -> List[np.ndarray]:
    """
    可视化多头注意力的各个头

    [参数]:
    - attention_weights: 多头注意力权重 (num_heads, seq_len, seq_len)
    - num_heads: 注意力头数量
    - figsize: 单个子图的尺寸
    - save_path: 保存路径

    [返回]:
    - 各头可视化的图像列表
    """
    seq_len = attention_weights.shape[1]
    images = []

    fig, axes = plt.subplots(1, num_heads, figsize=figsize)

    if num_heads == 1:
        axes = [axes]

    for head in range(num_heads):
        im = axes[head].imshow(attention_weights[head], cmap='viridis', aspect='auto')
        axes[head].set_title(f'Head {head + 1}')
        axes[head].set_xlabel('Key')
        axes[head].set_ylabel('Query')
        plt.colorbar(im, ax=axes[head])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    # 转换为numpy数组 (使用buffer而非tostring_rgb，兼容macOS)
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)
    img = img[:, :, :3]  # 移除alpha通道，只保留RGB
    images.append(img)

    plt.close(fig)

    return images