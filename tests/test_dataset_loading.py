"""
TDD Phase 1: 数据集加载与标签映射测试（RED阶段）
[INPUT]: 数据集路径、标签映射配置
[OUTPUT]: 测试报告（预期失败 → 实现后通过）
[POS]: tests/数据集测试，验证CustomLungDataset
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
import tempfile
import shutil
from PIL import Image
import pandas as pd
import os

from torch.utils.data import ConcatDataset, random_split


# =============================================================================
# Fixtures: 模拟数据集
# =============================================================================

@pytest.fixture(scope="function")
def mock_dataset1_dir():
    """模拟IQ-OTHNCCD数据集结构"""
    tmpdir = tempfile.mkdtemp()

    # dataset1结构: dataset1/Normal cases/, dataset1/Malignant cases/, dataset1/Benign cases/
    for class_name in ['Normal cases', 'Malignant cases', 'Benign cases']:
        class_dir = Path(tmpdir) / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        # 创建5张模拟图像
        for i in range(5):
            img = Image.new('RGB', (512, 512), color=(i*40, 100, 150))
            img.save(class_dir / f"image_{i}.jpg")

    yield tmpdir
    shutil.rmtree(tmpdir)


@pytest.fixture(scope="function")
def mock_dataset2_dir():
    """模拟Lung and Colon Cancer数据集结构"""
    tmpdir = tempfile.mkdtemp()

    # dataset2结构: dataset2/lung_n/, dataset2/lung_aca/, dataset2/lung_scc/
    class_mapping = {
        'lung_n': 'normal',
        'lung_aca': 'malignant',
        'lung_scc': 'benign'
    }

    for class_name in class_mapping.keys():
        class_dir = Path(tmpdir) / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        for i in range(5):
            img = Image.new('RGB', (768, 768), color=(i*30, 120, 180))
            img.save(class_dir / f"img_{i}.jpg")

    yield tmpdir
    shutil.rmtree(tmpdir)


@pytest.fixture(scope="function")
def mock_dataset3_dir():
    """模拟Lung Cancer 4 Types数据集结构 (train/val/test划分)"""
    tmpdir = tempfile.mkdtemp()

    # dataset3结构: dataset3/{train,val,test}/{normal,large.cell.carcinoma,adenocarcinoma,squamous.cell.carcinoma}/
    splits = ['train', 'val', 'test']
    classes = ['normal', 'large.cell.carcinoma', 'adenocarcinoma', 'squamous.cell.carcinoma']

    for split in splits:
        for class_name in classes:
            class_dir = Path(tmpdir) / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            # 每个类别创建3张图
            for i in range(3):
                img = Image.new('RGB', (512, 512), color=(i*50, 80, 200))
                img.save(class_dir / f"image_{i}.png")

    yield tmpdir
    shutil.rmtree(tmpdir)


# =============================================================================
# 测试类: CustomLungDataset
# =============================================================================

class TestCustomLungDataset:
    """测试自定义肺癌数据集类"""

    @pytest.mark.skip(reason="待实现: CustomLungDataset类")
    def test_initialization_with_dataset1(self, mock_dataset1_dir):
        """测试使用dataset1初始化"""
        from data.custom_dataset import CustomLungDataset

        dataset = CustomLungDataset(
            root_path=mock_dataset1_dir,
            dataset_type='dataset1'
        )

        # dataset1: 3 classes x 5 images = 15 samples
        assert len(dataset) == 15
        assert hasattr(dataset, 'labels')
        assert hasattr(dataset, 'images_path')

        # 验证标签映射: Normal cases -> normal (0), Malignant cases -> malignant (1), Benign cases -> benign (2)
        label_set = set(dataset.labels)
        assert label_set == {0, 1, 2}

    @pytest.mark.skip(reason="待实现: CustomLungDataset类")
    def test_initialization_with_dataset2(self, mock_dataset2_dir):
        """测试使用dataset2初始化"""
        from data.custom_dataset import CustomLungDataset

        dataset = CustomLungDataset(
            root_path=mock_dataset2_dir,
            dataset_type='dataset2'
        )

        # dataset2: 3 classes x 5 images = 15 samples
        assert len(dataset) == 15

        # 验证标签映射: lung_n -> normal (0), lung_aca -> malignant (1), lung_scc -> benign (2)
        label_counts = {0: 0, 1: 0, 2: 0}
        for label in dataset.labels:
            label_counts[label] += 1

        assert label_counts[0] == 5  # normal
        assert label_counts[1] == 5  # malignant
        assert label_counts[2] == 5  # benign

    @pytest.mark.skip(reason="待实现: CustomLungDataset类")
    def test_initialization_with_dataset3(self, mock_dataset3_dir):
        """测试使用dataset3初始化"""
        from data.custom_dataset import CustomLungDataset

        dataset = CustomLungDataset(
            root_path=mock_dataset3_dir,
            dataset_type='dataset3'
        )

        # dataset3: 3 splits x 4 classes x 3 images = 36 samples
        assert len(dataset) == 36

        # 验证标签映射:
        # normal -> normal (0)
        # large.cell.carcinoma -> malignant (1)
        # adenocarcinoma -> malignant (1)
        # squamous.cell.carcinoma -> benign (2)

    @pytest.mark.skip(reason="待实现: CustomLungDataset类")
    def test_getitem(self, mock_dataset1_dir):
        """测试__getitem__方法"""
        from data.custom_dataset import CustomLungDataset

        dataset = CustomLungDataset(
            root_path=mock_dataset1_dir,
            dataset_type='dataset1'
        )

        # 获取第一个样本
        image, label = dataset[0]

        # 验证返回类型
        assert isinstance(image, torch.Tensor)
        assert isinstance(label, int)
        assert label in [0, 1, 2]

        # 验证图像形状 [C, H, W]
        assert image.dim() == 3
        assert image.shape[0] == 3  # RGB

    @pytest.mark.skip(reason="待实现: CustomLungDataset类")
    def test_with_transform(self, mock_dataset1_dir):
        """测试带变换的数据集"""
        from data.custom_dataset import CustomLungDataset
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        dataset = CustomLungDataset(
            root_path=mock_dataset1_dir,
            dataset_type='dataset1',
            transform=transform
        )

        image, label = dataset[0]

        # 验证变换后尺寸
        assert image.shape == torch.Size([3, 224, 224])


# =============================================================================
# 测试类: 三数据集合并
# =============================================================================

class TestDatasetMerging:
    """测试三数据集合并"""

    @pytest.mark.skip(reason="待实现: 数据集合并逻辑")
    def test_concat_datasets(self, mock_dataset1_dir, mock_dataset2_dir, mock_dataset3_dir):
        """测试使用ConcatDataset合并"""
        from data.custom_dataset import CustomLungDataset

        # 创建三个数据集
        dataset1 = CustomLungDataset(mock_dataset1_dir, 'dataset1')
        dataset2 = CustomLungDataset(mock_dataset2_dir, 'dataset2')
        dataset3 = CustomLungDataset(mock_dataset3_dir, 'dataset3')

        # 合并
        combined = ConcatDataset([dataset1, dataset2, dataset3])

        # 验证总样本数
        expected_total = 15 + 15 + 36  # 66
        assert len(combined) == expected_total

    @pytest.mark.skip(reason="待实现: 标签映射验证")
    def test_unified_label_mapping(self, mock_dataset1_dir, mock_dataset2_dir, mock_dataset3_dir):
        """测试统一标签映射"""
        from data.custom_dataset import CustomLungDataset

        # 创建并合并数据集
        dataset1 = CustomLungDataset(mock_dataset1_dir, 'dataset1')
        dataset2 = CustomLungDataset(mock_dataset2_dir, 'dataset2')
        dataset3 = CustomLungDataset(mock_dataset3_dir, 'dataset3')

        combined = ConcatDataset([dataset1, dataset2, dataset3])

        # 验证所有标签都在[0, 1, 2]范围内
        all_labels = []
        for i in range(len(combined)):
            _, label = combined[i]
            all_labels.append(label)
            assert label in [0, 1, 2], f"Invalid label {label} at index {i}"

        # 验证每个类别都有样本
        label_counts = {0: 0, 1: 0, 2: 0}
        for label in all_labels:
            label_counts[label] += 1

        assert label_counts[0] > 0, "No normal samples"
        assert label_counts[1] > 0, "No benign samples"
        assert label_counts[2] > 0, "No malignant samples"


# =============================================================================
# 测试类: 数据划分
# =============================================================================

class TestDataSplitting:
    """测试数据集划分"""

    @pytest.mark.skip(reason="待实现: 数据集划分")
    def test_train_val_test_split(self, mock_dataset1_dir, mock_dataset2_dir, mock_dataset3_dir):
        """测试70/15/15划分"""
        from data.custom_dataset import CustomLungDataset

        # 创建并合并数据集
        dataset1 = CustomLungDataset(mock_dataset1_dir, 'dataset1')
        dataset2 = CustomLungDataset(mock_dataset2_dir, 'dataset2')
        dataset3 = CustomLungDataset(mock_dataset3_dir, 'dataset3')

        combined = ConcatDataset([dataset1, dataset2, dataset3])
        total_size = len(combined)

        # 计算划分大小
        train_size = int(0.7 * total_size)
        val_size = int(0.15 * total_size)
        test_size = total_size - train_size - val_size

        # 划分
        train_dataset, val_dataset, test_dataset = random_split(
            combined, [train_size, val_size, test_size]
        )

        # 验证划分比例
        assert len(train_dataset) == train_size
        assert len(val_dataset) == val_size
        assert len(test_dataset) == test_size

        # 验证总和
        assert len(train_dataset) + len(val_dataset) + len(test_dataset) == total_size

    @pytest.mark.skip(reason="待实现: DataLoader测试")
    def test_dataloader_creation(self, mock_dataset1_dir):
        """测试DataLoader创建"""
        from data.custom_dataset import CustomLungDataset
        from torch.utils.data import DataLoader

        dataset = CustomLungDataset(mock_dataset1_dir, 'dataset1')

        # 创建DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0  # 测试时用0
        )

        # 验证可以迭代
        batch_count = 0
        for images, labels in dataloader:
            batch_count += 1

            # 验证批次形状
            assert images.shape[0] <= 4  # 批次大小
            assert images.dim() == 4  # [B, C, H, W]
            assert labels.dim() == 1  # [B]

            # 只验证第一个批次
            break

        assert batch_count > 0


# =============================================================================
# 主测试入口
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
