"""
测试模型前向传播与训练流程
[INPUT]: 模型配置、模拟数据
[OUTPUT]: 测试报告、前向/反向传播验证
[POS]: tests/模型训练测试，验证端到端流程
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
import torch.nn as nn
import tempfile
from PIL import Image
import numpy as np

from model import HybridModel
from data.custom_dataset import CustomLungDataset
from data.augmentation import get_train_augmentation


class TestModelForward:
    """测试模型前向传播"""

    @pytest.fixture
    def model_config(self):
        """模型配置"""
        return {
            'num_classes': 3,
            'model_dim': 256,
            'nhead': 8,
            'num_layers': 4,
            'dropout': 0.1,
        }

    @pytest.fixture
    def batch_size(self):
        return 2

    @pytest.fixture
    def image_size(self):
        return 224

    @pytest.fixture
    def sample_batch(self, batch_size, image_size):
        """创建模拟图像批次"""
        return torch.randn(batch_size, 3, image_size, image_size)

    def test_model_initialization(self, model_config):
        """测试模型初始化"""
        model = HybridModel(**model_config)
        assert model is not None
        assert isinstance(model, nn.Module)

    def test_forward_pass(self, model_config, sample_batch):
        """测试前向传播"""
        model = HybridModel(**model_config)
        model.eval()

        with torch.no_grad():
            output = model(sample_batch)

        # 验证输出形状
        assert output.shape == (sample_batch.size(0), model_config['num_classes'])
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_backward_pass(self, model_config, sample_batch):
        """测试反向传播"""
        model = HybridModel(**model_config)
        model.train()

        # 前向传播
        output = model(sample_batch)

        # 计算损失
        target = torch.randint(0, model_config['num_classes'], (sample_batch.size(0),))
        loss = nn.CrossEntropyLoss()(output, target)

        # 反向传播
        loss.backward()

        # 验证梯度存在
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters() if p.requires_grad)
        assert has_grad, "Model should have non-zero gradients after backward"


class TestEndToEndTraining:
    """测试端到端训练流程"""

    @pytest.fixture
    def mock_dataset(self):
        """创建模拟数据集"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建模拟数据集结构
            for class_name in ['Normal cases', 'Malignant cases', 'Benign cases']:
                class_dir = Path(tmpdir) / class_name
                class_dir.mkdir(parents=True, exist_ok=True)
                for i in range(5):
                    img = Image.new('RGB', (512, 512), color=(i*40, 100, 150))
                    img.save(class_dir / f"image_{i}.jpg")

            yield tmpdir

    def test_data_loading(self, mock_dataset):
        """测试数据加载"""
        dataset = CustomLungDataset(mock_dataset, 'dataset1')
        assert len(dataset) == 15

        # 测试获取样本
        image, label = dataset[0]
        assert image is not None
        assert label in [0, 1, 2]

    def test_training_step(self, mock_dataset):
        """测试单步训练"""
        from torch.utils.data import DataLoader
        from data.augmentation import get_val_augmentation

        # 创建数据集和加载器
        transform = get_val_augmentation(224)
        dataset = CustomLungDataset(mock_dataset, 'dataset1', transform=transform)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

        # 创建模型
        model = HybridModel(num_classes=3, model_dim=128, nhead=4, num_layers=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        # 单步训练
        model.train()
        images, labels = next(iter(dataloader))

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        assert loss.item() > 0
        print(f"Training step completed. Loss: {loss.item():.4f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
