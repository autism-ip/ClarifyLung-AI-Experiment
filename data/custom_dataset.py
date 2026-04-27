"""
自定义肺癌数据集加载模块
[INPUT]: 数据集路径、数据集类型(dataset1/2/3)、可选变换
[OUTPUT]: CustomLungDataset实例，统一标签(normal=0, benign=1, malignant=2)
[POS]: data/核心组件，处理三个数据集的加载与标签映射
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import os
from pathlib import Path
from typing import Optional, Callable, List, Dict, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset, ConcatDataset


# =============================================================================
# 数据集标签映射配置
# =============================================================================

# Dataset 1 (IQ-OTHNCCD) 映射
DATASET1_MAPPING = {
    'Normal cases': 'normal',
    'Malignant cases': 'malignant',
    'Benign cases': 'benign'
}

# Dataset 2 (Lung and Colon Cancer) 映射
DATASET2_MAPPING = {
    'lung_n': 'normal',
    'lung_aca': 'malignant',
    'lung_scc': 'benign'
}

# Dataset 3 (Lung Cancer 4 Types) 映射
DATASET3_MAPPING = {
    'normal': 'normal',
    'large.cell.carcinoma': 'malignant',
    'adenocarcinoma': 'malignant',
    'squamous.cell.carcinoma': 'benign'
}

# 统一标签到整数编码
LABEL_TO_INDEX = {
    'normal': 0,
    'benign': 1,
    'malignant': 2
}

# 反向映射（用于显示）
INDEX_TO_LABEL = {v: k for k, v in LABEL_TO_INDEX.items()}


# =============================================================================
# 自定义肺癌数据集类
# =============================================================================

class CustomLungDataset(Dataset):
    """
    自定义肺癌数据集类

    支持加载三个不同的肺癌数据集，并统一映射到标准标签：
    - 0: normal (正常)
    - 1: benign (良性)
    - 2: malignant (恶性)

    Attributes:
        root_path: 数据集根目录路径
        dataset_type: 数据集类型 ('dataset1', 'dataset2', 'dataset3')
        transform: 可选的图像变换
        images_path: 图像路径列表
        labels: 整数标签列表
    """

    def __init__(
        self,
        root_path: str,
        dataset_type: str,
        transform: Optional[Callable] = None
    ):
        """
        初始化数据集

        Args:
            root_path: 数据集根目录路径
            dataset_type: 数据集类型 ('dataset1', 'dataset2', 'dataset3')
            transform: 可选的图像变换函数
        """
        super().__init__()

        self.root_path = Path(root_path)
        self.dataset_type = dataset_type.lower()
        self.transform = transform

        # 存储图像路径和标签
        self.images_path: List[str] = []
        self.labels: List[int] = []

        # 根据数据集类型加载数据
        if self.dataset_type == 'dataset1':
            self._load_dataset1()
        elif self.dataset_type == 'dataset2':
            self._load_dataset2()
        elif self.dataset_type == 'dataset3':
            self._load_dataset3()
        else:
            raise ValueError(f"Unknown dataset_type: {dataset_type}. "
                           "Choose from 'dataset1', 'dataset2', 'dataset3'")

        # 验证数据加载成功
        if len(self.images_path) == 0:
            raise RuntimeError(f"No images found in {root_path} for {dataset_type}")

        if len(self.images_path) != len(self.labels):
            raise RuntimeError(f"Mismatch: {len(self.images_path)} images but {len(self.labels)} labels")

    def _load_dataset1(self):
        """
        加载Dataset 1 (IQ-OTHNCCD)
        期望结构: root/Normal cases/, root/Malignant cases/, root/Benign cases/
        实际Kaggle结构: root/Augmented IQ-OTHNCCD lung cancer dataset/{Normal cases,...}/
        策略: 自动探测根目录或子目录下的类别文件夹
        """
        classes = ['Normal cases', 'Malignant cases', 'Benign cases']

        for class_name in classes:
            # 尝试直接路径
            class_path = self.root_path / class_name
            if not class_path.exists():
                # 尝试在子目录中查找（适配Kaggle下载后的嵌套结构）
                for subdir in self.root_path.iterdir():
                    if subdir.is_dir():
                        candidate = subdir / class_name
                        if candidate.exists():
                            class_path = candidate
                            break

            if not class_path.exists():
                print(f"Warning: {class_name} not found under {self.root_path}, skipping...")
                continue

            # 映射到统一标签
            unified_label = DATASET1_MAPPING[class_name]
            label_idx = LABEL_TO_INDEX[unified_label]

            # 遍历目录中的图像文件
            for image_name in sorted(class_path.iterdir()):
                if image_name.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']:
                    self.images_path.append(str(image_name))
                    self.labels.append(label_idx)

    def _load_dataset2(self):
        """
        加载Dataset 2 (Lung and Colon Cancer)
        期望结构: root/lung_n/, root/lung_aca/, root/lung_scc/
        实际Kaggle结构: root/lung_colon_image_set/lung_image_sets/{lung_n,lung_aca,lung_scc}/
        策略: 自动探测根目录或嵌套子目录下的类别文件夹
        """
        classes = ['lung_n', 'lung_aca', 'lung_scc']

        for class_name in classes:
            # 尝试直接路径
            class_path = self.root_path / class_name
            if not class_path.exists():
                # 深度探测: 最多向下递归2层查找类别目录
                found = False
                for subdir in self.root_path.rglob('*'):
                    if subdir.is_dir() and subdir.name == class_name:
                        class_path = subdir
                        found = True
                        break
                if not found:
                    print(f"Warning: {class_name} not found under {self.root_path}, skipping...")
                    continue

            # 映射到统一标签
            unified_label = DATASET2_MAPPING[class_name]
            label_idx = LABEL_TO_INDEX[unified_label]

            for image_name in sorted(class_path.iterdir()):
                if image_name.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']:
                    self.images_path.append(str(image_name))
                    self.labels.append(label_idx)

    def _load_dataset3(self):
        """
        加载Dataset 3 (Lung Cancer 4 Types)
        期望结构: root/{train,valid,test}/{class_subdir}/  (class_subdir以class_name开头)
        实际Kaggle结构: root/Data/{train,valid,test}/{class_subdir}/
        策略: 自动探测 root/ 或 root/Data/ 下的 split 目录
        """
        splits = ['train', 'val', 'test', 'valid']  # 兼容 'val' 和 'valid'
        # 原始类名前缀 -> 统一标签映射
        class_prefixes = {
            'normal': 'normal',
            'large.cell.carcinoma': 'malignant',
            'adenocarcinoma': 'malignant',
            'squamous.cell.carcinoma': 'benign'
        }

        # 探测 split 根目录: 优先 root/{split}/, 其次 root/Data/{split}/
        split_roots = [self.root_path]
        data_subdir = self.root_path / 'Data'
        if data_subdir.exists() and data_subdir.is_dir():
            split_roots.append(data_subdir)

        for split in splits:
            split_path = None
            for root in split_roots:
                candidate = root / split
                if candidate.exists():
                    split_path = candidate
                    break

            if split_path is None:
                continue

            # 遍历split下的所有子目录
            for subdir in split_path.iterdir():
                if not subdir.is_dir():
                    continue

                # 根据子目录名前缀匹配类别
                matched_label = None
                for prefix, unified_label in class_prefixes.items():
                    if subdir.name.startswith(prefix):
                        matched_label = unified_label
                        break

                if matched_label is None:
                    continue

                label_idx = LABEL_TO_INDEX[matched_label]

                for image_name in sorted(subdir.iterdir()):
                    if image_name.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']:
                        self.images_path.append(str(image_name))
                        self.labels.append(label_idx)

    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.images_path)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        获取单个样本

        Args:
            idx: 样本索引

        Returns:
            image: 图像张量 [C, H, W]
            label: 整数标签 (0=normal, 1=benign, 2=malignant)
        """
        img_path = self.images_path[idx]
        label = self.labels[idx]

        # 加载图像
        image = Image.open(img_path).convert('RGB')

        # 应用变换
        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_distribution(self) -> Dict[int, int]:
        """
        获取类别分布统计

        Returns:
            字典: {label_index: count}
        """
        from collections import Counter
        return dict(Counter(self.labels))

    def get_class_names(self) -> List[str]:
        """
        获取类别名称列表

        Returns:
            类别名称列表
        """
        return [INDEX_TO_LABEL[i] for i in range(len(INDEX_TO_LABEL))]


# =============================================================================
# 便捷函数
# =============================================================================

def merge_datasets(
    dataset1_path: str,
    dataset2_path: str,
    dataset3_path: str,
    transform: Optional[Callable] = None,
) -> ConcatDataset:
    """
    合并三个数据集

    Args:
        dataset1_path: Dataset 1 (IQ-OTHNCCD) 路径
        dataset2_path: Dataset 2 (Lung and Colon) 路径
        dataset3_path: Dataset 3 (Lung Cancer 4 Types) 路径
        transform: 可选的图像变换

    Returns:
        ConcatDataset: 合并后的数据集
    """
    dataset1 = CustomLungDataset(dataset1_path, 'dataset1', transform)
    dataset2 = CustomLungDataset(dataset2_path, 'dataset2', transform)
    dataset3 = CustomLungDataset(dataset3_path, 'dataset3', transform)

    return ConcatDataset([dataset1, dataset2, dataset3])


def split_dataset(
    dataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
):
    """
    划分数据集为train/val/test

    Args:
        dataset: 输入数据集
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子

    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    # 设置随机种子保证可复现
    torch.manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    return train_dataset, val_dataset, test_dataset


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == "__main__":
    print("Testing CustomLungDataset...")
    print("=" * 60)

    # 创建临时测试数据集
    with tempfile.TemporaryDirectory() as tmpdir:
        # 模拟Dataset 1结构
        for class_name in ['Normal cases', 'Malignant cases', 'Benign cases']:
            class_dir = Path(tmpdir) / 'dataset1' / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            for i in range(3):
                img = Image.new('RGB', (512, 512), color=(i*60, 100, 150))
                img.save(class_dir / f"img_{i}.jpg")

        # 测试Dataset 1加载
        print("\n1. Testing Dataset 1 (IQ-OTHNCCD) loading:")
        try:
            dataset1 = CustomLungDataset(
                root_path=str(Path(tmpdir) / 'dataset1'),
                dataset_type='dataset1'
            )
            print(f"   ✓ Loaded {len(dataset1)} samples")
            print(f"   ✓ Class distribution: {dataset1.get_class_distribution()}")

            # 测试__getitem__
            image, label = dataset1[0]
            print(f"   ✓ Sample image shape: {image.shape}, label: {label}")
        except Exception as e:
            print(f"   ✗ Error: {e}")

        # 测试合并功能
        print("\n2. Testing dataset merging:")
        try:
            # 创建更多模拟数据
            for class_name in ['lung_n', 'lung_aca', 'lung_scc']:
                class_dir = Path(tmpdir) / 'dataset2' / class_name
                class_dir.mkdir(parents=True, exist_ok=True)
                for i in range(2):
                    img = Image.new('RGB', (512, 512), color=(i*80, 120, 180))
                    img.save(class_dir / f"img_{i}.jpg")

            dataset1 = CustomLungDataset(str(Path(tmpdir) / 'dataset1'), 'dataset1')
            dataset2 = CustomLungDataset(str(Path(tmpdir) / 'dataset2'), 'dataset2')

            combined = ConcatDataset([dataset1, dataset2])
            print(f"   ✓ Combined dataset size: {len(combined)}")
            print(f"   ✓ Dataset 1: {len(dataset1)}, Dataset 2: {len(dataset2)}")
        except Exception as e:
            print(f"   ✗ Error: {e}")

        # 测试划分功能
        print("\n3. Testing dataset splitting:")
        try:
            train_dataset, val_dataset, test_dataset = split_dataset(
                combined,
                train_ratio=0.7,
                val_ratio=0.15,
                test_ratio=0.15,
                seed=42
            )
            print(f"   ✓ Train: {len(train_dataset)} ({len(train_dataset)/len(combined)*100:.1f}%)")
            print(f"   ✓ Val: {len(val_dataset)} ({len(val_dataset)/len(combined)*100:.1f}%)")
            print(f"   ✓ Test: {len(test_dataset)} ({len(test_dataset)/len(combined)*100:.1f}%)")
        except Exception as e:
            print(f"   ✗ Error: {e}")

    print("\n" + "=" * 60)
    print("CustomLungDataset test completed!")
    print("=" * 60)
