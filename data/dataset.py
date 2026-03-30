"""
数据集定义模块
[INPUT]: 图像路径列表、标签、transform配置
[OUTPUT]: PyTorch Dataset/DataLoader实例
[POS]: data/核心组件，被训练脚本直接调用
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import os
from pathlib import Path
from typing import Optional, Callable, Tuple, List, Dict
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T


class XRayDataset(Dataset):
    """
    X光片数据集类

    支持从CSV或目录结构加载，自动处理三分类标签

    Attributes:
        image_paths: 图像文件路径列表
        labels: 整数标签列表 (0=normal, 1=malignant, 2=benign)
        class_names: 类别名称列表
        transform: 图像变换管道
    """

    CLASS_NAMES = ["normal", "malignant", "benign"]

    def __init__(
        self,
        data_source: str,
        transform: Optional[Callable] = None,
        mode: str = "csv",  # 'csv' or 'folder'
    ):
        """
        Args:
            data_source: CSV文件路径或数据目录路径
            transform: 可选的图像变换
            mode: 数据加载模式 ('csv' 或 'folder')
        """
        super().__init__()
        self.data_source = data_source
        self.transform = transform
        self.mode = mode
        self.image_paths: List[str] = []
        self.labels: List[int] = []

        if mode == "csv":
            self._load_from_csv()
        elif mode == "folder":
            self._load_from_folder()
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self._validate_data()

    def _load_from_csv(self):
        """从CSV加载数据，期望列: image_path, label (或class_name)"""
        df = pd.read_csv(self.data_source)

        # 自动检测列名
        path_col = None
        for col in ["image_path", "path", "filepath", "image"]:
            if col in df.columns:
                path_col = col
                break

        label_col = None
        for col in ["label", "class", "class_name", "category"]:
            if col in df.columns:
                label_col = col
                break

        if path_col is None:
            raise ValueError(f"CSV must contain an image path column. Found: {df.columns.tolist()}")
        if label_col is None:
            raise ValueError(f"CSV must contain a label column. Found: {df.columns.tolist()}")

        # 转换标签为整数
        for _, row in df.iterrows():
            self.image_paths.append(row[path_col])
            label_val = row[label_col]
            if isinstance(label_val, str):
                label_idx = self.CLASS_NAMES.index(label_val.lower())
            else:
                label_idx = int(label_val)
            self.labels.append(label_idx)

    def _load_from_folder(self):
        """从文件夹结构加载: data_source/class_name/image.jpg"""
        data_path = Path(self.data_source)

        for class_idx, class_name in enumerate(self.CLASS_NAMES):
            class_path = data_path / class_name
            if not class_path.exists():
                continue

            for img_path in class_path.glob("*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                    self.image_paths.append(str(img_path))
                    self.labels.append(class_idx)

    def _validate_data(self):
        """验证数据完整性"""
        if len(self.image_paths) == 0:
            raise ValueError("No images found in dataset")

        # 检查所有图像是否存在
        missing = []
        for path in self.image_paths[:100]:  # 抽样检查前100个
            if not Path(path).exists():
                missing.append(path)

        if len(missing) > 0:
            print(f"Warning: {len(missing)} images not found (showing first 5): {missing[:5]}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Returns:
            image: 变换后的图像张量 [C, H, W]
            label: 整数标签
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # 加载图像
        image = Image.open(img_path).convert('RGB')

        # 应用变换
        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_distribution(self) -> Dict[int, int]:
        """获取类别分布统计"""
        from collections import Counter
        return dict(Counter(self.labels))

    def get_weighted_sampler(self) -> WeightedRandomSampler:
        """获取类别平衡采样器，用于处理类别不平衡"""
        class_counts = self.get_class_distribution()
        total = len(self.labels)

        # 计算每个样本的权重
        weights = []
        for label in self.labels:
            class_weight = total / (len(class_counts) * class_counts[label])
            weights.append(class_weight)

        return WeightedRandomSampler(weights, len(weights), replacement=True)


def get_data_loaders(
    train_source: str,
    val_source: str,
    test_source: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    use_augmentation: bool = True,
    balance_sampling: bool = False,
) -> Dict[str, DataLoader]:
    """
    创建数据加载器工厂函数

    Args:
        train_source: 训练集路径(CSV或目录)
        val_source: 验证集路径
        test_source: 测试集路径(可选)
        batch_size: 批次大小
        num_workers: 数据加载线程数
        image_size: 图像尺寸
        use_augmentation: 是否使用数据增强
        balance_sampling: 是否使用类别平衡采样

    Returns:
        字典，包含'train'、'val'、(可选)'test'的DataLoader
    """
    from .augmentation import get_train_augmentation, get_val_augmentation

    # 选择变换
    train_transform = get_train_augmentation(image_size) if use_augmentation else get_val_augmentation(image_size)
    val_transform = get_val_augmentation(image_size)

    # 创建数据集
    train_dataset = XRayDataset(train_source, transform=train_transform, mode="auto")
    val_dataset = XRayDataset(val_source, transform=val_transform, mode="auto")

    # 创建采样器
    train_sampler = train_dataset.get_weighted_sampler() if balance_sampling else None

    # 创建DataLoader
    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    }

    # 测试集(可选)
    if test_source:
        test_dataset = XRayDataset(test_source, transform=val_transform, mode="auto")
        loaders["test"] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    # 打印数据信息
    print("=" * 50)
    print("数据集加载完成")
    print("=" * 50)
    print(f"训练集: {len(train_dataset)} 样本")
    print(f"  类别分布: {train_dataset.get_class_distribution()}")
    print(f"验证集: {len(val_dataset)} 样本")
    if test_source:
        print(f"测试集: {len(test_dataset)} 样本")
    print("=" * 50)

    return loaders
