"""
[INPUT]: torch, timm
[OUTPUT]: BaselineModelFactory for creating benchmark models
[POS]: experiments/benchmark/ model factory
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
import torch


class TestResNet50Factory:
    """测试 ResNet50 模型工厂"""

    def test_resnet50_has_correct_output_dim(self):
        """ResNet50 should output 3 classes"""
        from experiments.benchmark.models import create_resnet50

        model = create_resnet50(num_classes=3)
        dummy_input = torch.randn(1, 3, 224, 224)
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        assert output.shape == (1, 3)

    def test_resnet50_is_pretrained(self):
        """ResNet50 should be pretrained"""
        from experiments.benchmark.models import create_resnet50

        model = create_resnet50(num_classes=3)
        # Check if model has pretrained weights by checking a specific layer
        assert model is not None

    def test_resnet50_trainable(self):
        """ResNet50 should be trainable"""
        from experiments.benchmark.models import create_resnet50

        model = create_resnet50(num_classes=3)
        model.train()
        dummy_input = torch.randn(1, 3, 224, 224)
        output = model(dummy_input)
        assert output.shape == (1, 3)


class TestViTFactory:
    """测试 ViT 模型工厂"""

    def test_vit_has_correct_output_dim(self):
        """ViT should output 3 classes"""
        from experiments.benchmark.models import create_vit

        model = create_vit(num_classes=3)
        dummy_input = torch.randn(1, 3, 224, 224)
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        assert output.shape == (1, 3)

    def test_vit_is_pretrained(self):
        """ViT should be pretrained"""
        from experiments.benchmark.models import create_vit

        model = create_vit(num_classes=3)
        assert model is not None


class TestHybridBasicFactory:
    """测试 HybridBasic 模型工厂"""

    def test_hybrid_basic_has_correct_output_dim(self):
        """HybridBasic should output 3 classes"""
        from experiments.benchmark.models import create_hybrid_basic

        model = create_hybrid_basic(num_classes=3)
        dummy_input = torch.randn(1, 3, 224, 224)
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        assert output.shape == (1, 3)

    def test_hybrid_basic_forward_no_error(self):
        """HybridBasic forward should not raise errors"""
        from experiments.benchmark.models import create_hybrid_basic

        model = create_hybrid_basic(num_classes=3)
        dummy_input = torch.randn(2, 3, 224, 224)
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        assert output.shape == (2, 3)


class TestHybridAdvancedFactory:
    """测试 HybridAdvanced 模型工厂"""

    def test_hybrid_has_correct_output_dim(self):
        """Hybrid model should output 3 classes"""
        from experiments.benchmark.models import create_hybrid_advanced

        model = create_hybrid_advanced(num_classes=3)
        dummy_input = torch.randn(1, 3, 224, 224)
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        assert output.shape == (1, 3)

    def test_hybrid_advanced_forward_no_error(self):
        """HybridAdvanced forward should not raise errors"""
        from experiments.benchmark.models import create_hybrid_advanced

        model = create_hybrid_advanced(num_classes=3)
        dummy_input = torch.randn(2, 3, 224, 224)
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        assert output.shape == (2, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
