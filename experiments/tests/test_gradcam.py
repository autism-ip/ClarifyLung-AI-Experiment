# =============================================================================
# TDD Phase 1: RED - 失败的测试用例
# =============================================================================
"""
[INPUT]: torch, gradcam, cv2
[OUTPUT]: GradCAMVisualizer
[POS]: experiments/visualization/ Grad-CAM heatmap generation
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import pytest
import torch
import numpy as np


# -----------------------------------------------------------------------------
# Test: GradCAMVisualizer Initialization
# -----------------------------------------------------------------------------
def test_gradcam_visualizer_init():
    """GradCAMVisualizer should initialize with model"""
    from experiments.visualization.gradcam import GradCAMVisualizer
    import torch.nn as nn

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, 3)
            self.fc = nn.Linear(16, 3)

        def forward(self, x):
            x = self.conv(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)

    model = SimpleModel()
    viz = GradCAMVisualizer(model)
    assert viz.model is not None


# -----------------------------------------------------------------------------
# Test: Generate Heatmap Returns Array
# -----------------------------------------------------------------------------
def test_generate_heatmap_returns_array():
    """generate_heatmap should return numpy array"""
    from experiments.visualization.gradcam import GradCAMVisualizer
    import torch.nn as nn

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, 3, padding=1)  # 32->32
            self.pool1 = nn.MaxPool2d(2, 2)  # 32->16
            self.pool2 = nn.MaxPool2d(2, 2)  # 16->8
            self.fc = nn.Linear(16 * 8 * 8, 3)

        def forward(self, x):
            x = self.conv(x)  # (1, 16, 32, 32)
            x = torch.relu(x)
            x = self.pool1(x)  # (1, 16, 16, 16)
            x = self.pool2(x)  # (1, 16, 8, 8)
            x = x.view(x.size(0), -1)  # (1, 1024)
            return self.fc(x)

    model = SimpleModel()
    viz = GradCAMVisualizer(model)

    # 传入3D张量，generate_heatmap会unsqueeze(0)添加batch维度
    dummy_input = torch.randn(3, 32, 32)
    heatmap = viz.generate_heatmap(dummy_input, target_category=0)

    assert isinstance(heatmap, np.ndarray)
    assert heatmap.shape == (32, 32) or heatmap.shape == (1, 32, 32)
    assert heatmap.min() >= 0 and heatmap.max() <= 1


# -----------------------------------------------------------------------------
# Test: Overlay Returns Uint8 Image
# -----------------------------------------------------------------------------
def test_overlay_returns_uint8_image():
    """overlay_heatmap should return uint8 image"""
    from experiments.visualization.gradcam import overlay_heatmap
    import numpy as np

    heatmap = np.random.rand(224, 224).astype(np.float32)
    image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    overlay = overlay_heatmap(heatmap, image)

    assert overlay.dtype == np.uint8
    assert overlay.shape == (224, 224, 3)


# -----------------------------------------------------------------------------
# Test: Generate Heatmap with Different Target Categories
# -----------------------------------------------------------------------------
def test_generate_heatmap_different_categories():
    """generate_heatmap should work with different target categories"""
    from experiments.visualization.gradcam import GradCAMVisualizer
    import torch.nn as nn

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, 3, padding=1)  # 32->32
            self.pool1 = nn.MaxPool2d(2, 2)  # 32->16
            self.pool2 = nn.MaxPool2d(2, 2)  # 16->8
            self.fc = nn.Linear(16 * 8 * 8, 3)

        def forward(self, x):
            x = self.conv(x)  # (1, 16, 32, 32)
            x = torch.relu(x)
            x = self.pool1(x)  # (1, 16, 16, 16)
            x = self.pool2(x)  # (1, 16, 8, 8)
            x = x.view(x.size(0), -1)  # (1, 1024)
            return self.fc(x)

    model = SimpleModel()
    viz = GradCAMVisualizer(model)
    # 传入3D张量，generate_heatmap会unsqueeze(0)添加batch维度
    dummy_input = torch.randn(3, 32, 32)

    # Test all three categories
    for target in [0, 1, 2]:
        heatmap = viz.generate_heatmap(dummy_input, target_category=target)
        assert isinstance(heatmap, np.ndarray)
        assert heatmap.shape == (32, 32) or heatmap.shape == (1, 32, 32)


# -----------------------------------------------------------------------------
# Test: Overlay with Custom Alpha
# -----------------------------------------------------------------------------
def test_overlay_with_custom_alpha():
    """overlay_heatmap should respect alpha parameter"""
    from experiments.visualization.gradcam import overlay_heatmap
    import numpy as np

    heatmap = np.random.rand(224, 224).astype(np.float32)
    image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    overlay = overlay_heatmap(heatmap, image, alpha=0.6)

    assert overlay.dtype == np.uint8
    assert overlay.shape == (224, 224, 3)
    # With higher alpha, the heatmap should contribute more
    # We can't easily test this, but at least verify output is valid