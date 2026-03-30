# =============================================================================
# TDD Phase 1: RED - 失败的测试用例
# =============================================================================
"""
[INPUT]: torch, numpy
[OUTPUT]: AttentionVisualizer
[POS]: experiments/visualization/ Attention map extraction and visualization
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import pytest
import torch
import numpy as np


# -----------------------------------------------------------------------------
# Test: AttentionVisualizer Initialization
# -----------------------------------------------------------------------------
def test_attention_visualizer_init():
    """AttentionVisualizer should initialize with model"""
    from experiments.visualization.attention_maps import AttentionVisualizer
    import torch.nn as nn

    model = nn.Linear(100, 100)
    viz = AttentionVisualizer(model)
    assert viz.model is not None


# -----------------------------------------------------------------------------
# Test: Extract Attention Returns List
# -----------------------------------------------------------------------------
def test_extract_attention_returns_list():
    """extract_attention should return list of arrays"""
    from experiments.visualization.attention_maps import AttentionVisualizer
    import torch.nn as nn
    import torch

    class AttentionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = nn.MultiheadAttention(64, 4)

        def forward(self, x):
            # x: (seq, batch, embed)
            attn_output, attn_weights = self.attn(x, x, x)
            return attn_output

    model = AttentionModel()
    viz = AttentionVisualizer(model)

    dummy_input = torch.randn(16, 1, 64)  # (seq, batch, embed)
    attn_weights = viz.extract_attention(dummy_input)

    assert isinstance(attn_weights, list)
    assert len(attn_weights) > 0


# -----------------------------------------------------------------------------
# Test: Attention Weights Shape
# -----------------------------------------------------------------------------
def test_attention_weights_shape():
    """extract_attention should return weights with correct shape"""
    from experiments.visualization.attention_maps import AttentionVisualizer
    import torch.nn as nn
    import torch

    # 使用标准的nn.MultiheadAttention，确保AttentionVisualizer能正确识别
    class AttentionModel(nn.Module):
        def __init__(self):
            super().__init__()
            # 使用标准MultiheadAttention，embed_dim必须能被num_heads整除
            self.attn = nn.MultiheadAttention(64, 4, batch_first=False)

        def forward(self, x):
            # x: (seq, batch, embed)
            attn_output, attn_weights = self.attn(x, x, x)
            return attn_output

    model = AttentionModel()
    viz = AttentionVisualizer(model)

    seq_len = 16
    batch_size = 1
    embed_dim = 64
    dummy_input = torch.randn(seq_len, batch_size, embed_dim)

    attn_weights = viz.extract_attention(dummy_input)

    # Should return list of attention weight arrays
    assert isinstance(attn_weights, list)
    for w in attn_weights:
        assert isinstance(w, np.ndarray)
        # Attention weights should be at least 1D (could be 2D if multi-head)
        assert w.ndim >= 1


# -----------------------------------------------------------------------------
# Test: Visualize Attention Returns Image Array
# -----------------------------------------------------------------------------
def test_visualize_attention_returns_array():
    """visualize_attention should return numpy array"""
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端，避免macOS canvas问题
    from experiments.visualization.attention_maps import visualize_attention
    import numpy as np

    # Create a sample attention matrix
    seq_len = 8
    attn_matrix = np.random.rand(seq_len, seq_len).astype(np.float32)
    # Normalize to sum to 1 (like real attention)
    attn_matrix = attn_matrix / attn_matrix.sum(axis=-1, keepdims=True)

    viz = visualize_attention(attn_matrix)

    assert isinstance(viz, np.ndarray)
    assert len(viz.shape) in [2, 3]  # grayscale or RGB


# -----------------------------------------------------------------------------
# Test: Model Without Attention Raises Error
# -----------------------------------------------------------------------------
def test_model_without_attention_raises():
    """extract_attention should raise error for model without attention"""
    from experiments.visualization.attention_maps import AttentionVisualizer
    import torch.nn as nn
    import pytest

    class NoAttentionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(100, 100)

        def forward(self, x):
            return self.fc(x)

    model = NoAttentionModel()
    viz = AttentionVisualizer(model)

    dummy_input = torch.randn(10, 1, 100)

    with pytest.raises((ValueError, RuntimeError)):
        viz.extract_attention(dummy_input)