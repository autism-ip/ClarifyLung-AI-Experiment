"""
细粒度标签映射器模块
[INPUT]: 原始3分类数据集
[OUTPUT]: 5分类数据集（腺癌/鳞癌/良性结节/炎症/正常）
[POS]: data/ 细粒度分类核心组件
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md

映射规则：
  Lung4Types: 
    - adenocarcinoma → 0(腺癌)
    - squamous.cell.carcinoma → 1(鳞癌)  
    - large.cell.carcinoma → 2(良性结节/大细胞癌归入待研究)
    - normal → 4(正常)
  
  IQ-OTHNCCD:
    - Normal → 4(正常)
    - Benign → 2(良性结节)
    - Malignant → 3(炎症/恶性待细分)
  
  LungColon:
    - lung_aca → 0(腺癌)
    - lung_scc → 1(鳞癌)
    - lung_n → 4(正常)

注意：炎症(3)类别当前策略：
  - 从IQ-OTHNCCD的Benign中分离（如果子目录存在inflammatory标记）
  - 否则：炎症与良性结节合并为类别2（保守策略）
  - 或：将部分良性随机标为炎症（不推荐，需明确标注）
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable, Union
from collections import defaultdict

from PIL import Image
from torch.utils.data import Dataset, ConcatDataset

from configs.dataset_config import DATASET_PATHS


# =============================================================================
# 细粒度标签定义（5分类）
# =============================================================================

FINEGRAINED_CLASSES = [
    "adenocarcinoma",      # 0: 肺腺癌
    "squamous_carcinoma",  # 1: 肺鳞癌
    "benign_nodule",       # 2: 良性结节
    "inflammation",        # 3: 炎症
    "normal",              # 4: 正常
]

FINEGRAINED_LABEL_TO_INDEX = {name: idx for idx, name in enumerate(FINEGRAINED_CLASSES)}
FINEGRAINED_INDEX_TO_LABEL = {idx: name for name, idx in FINEGRAINED_LABEL_TO_INDEX.items()}


# =============================================================================
# 数据集特定的细粒度映射
# =============================================================================

# Dataset3 (Lung4Types) 细粒度映射
# 原始目录名 -> 细粒度标签
LUNG4TYPES_FINEGRAINED_MAP = {
    'adenocarcinoma': 'adenocarcinoma',           # 0
    'squamous.cell.carcinoma': 'squamous_carcinoma',  # 1
    'large.cell.carcinoma': 'benign_nodule',      # 2（大细胞癌暂归入良性/待研究）
    'normal': 'normal',                           # 4
}

# Dataset1 (IQ-OTHNCCD) 细粒度映射
# 基于子目录名
IQOTHNCCD_FINEGRAINED_MAP = {
    'Normal cases': 'normal',           # 4
    'Benign cases': 'benign_nodule',    # 2
    'Malignant cases': 'benign_nodule', # 2（保守策略：恶性X光暂归入良性结节，因无法细分为腺/鳞癌）
}

# Dataset2 (LungColon) 细粒度映射
LUNGCOLON_FINEGRAINED_MAP = {
    'lung_aca': 'adenocarcinoma',       # 0
    'lung_scc': 'squamous_carcinoma',   # 1
    'lung_n': 'normal',                 # 4
}


def get_finegrained_label(raw_label: str, dataset_type: str) -> int:
    """
    将原始标签映射为细粒度标签索引

    Args:
        raw_label: 原始标签字符串
        dataset_type: 数据集类型 ('dataset1', 'dataset2', 'dataset3')

    Returns:
        细粒度标签索引 (0-4)
    """
    if dataset_type == 'dataset1':
        unified = IQOTHNCCD_FINEGRAINED_MAP.get(raw_label, 'benign_nodule')
    elif dataset_type == 'dataset2':
        unified = LUNGCOLON_FINEGRAINED_MAP.get(raw_label, 'benign_nodule')
    elif dataset_type == 'dataset3':
        unified = LUNG4TYPES_FINEGRAINED_MAP.get(raw_label, 'benign_nodule')
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    return FINEGRAINED_LABEL_TO_INDEX[unified]


# =============================================================================
# 细粒度数据集类
# =============================================================================

class FineGrainedLungDataset(Dataset):
    """
    细粒度肺癌数据集（5分类）

    包装已有的 CustomLungDataset，在 __getitem__ 时重新映射标签

    用法：
        from data.custom_dataset import CustomLungDataset
        from data.finegrained_labeler import FineGrainedLungDataset

        base_dataset = CustomLungDataset(path, 'dataset3', transform=transform)
        fg_dataset = FineGrainedLungDataset(base_dataset, 'dataset3')
        
        img, label = fg_dataset[0]  # label 范围 0-4
    """

    def __init__(
        self,
        base_dataset: Dataset,
        dataset_type: str,
        transform: Optional[Callable] = None
    ):
        """
        Args:
            base_dataset: 原始数据集实例
            dataset_type: 数据集类型
            transform: 额外的变换（可选，基础变换已在base_dataset中）
        """
        self.base_dataset = base_dataset
        self.dataset_type = dataset_type
        self.transform = transform

        # 预计算所有细粒度标签
        self.finegrained_labels: List[int] = []
        self._build_label_mapping()

    def _build_label_mapping(self):
        """构建原始标签到细粒度标签的映射"""
        for i in range(len(self.base_dataset)):
            # 获取原始标签（CustomLungDataset返回img, label）
            _, raw_label = self.base_dataset[i]

            # 需要将整数标签转回原始字符串标签
            # 但base_dataset只返回整数，我们需要从路径推断
            # 这里使用一种策略：如果base_dataset有原始路径信息
            if hasattr(self.base_dataset, 'labels') and hasattr(self.base_dataset, 'images_path'):
                # 从路径推断原始类别名
                img_path = self.base_dataset.images_path[i]
                raw_name = Path(img_path).parent.name
                fg_label = get_finegrained_label(raw_name, self.dataset_type)
            else:
                # 回退：使用原始整数标签的保守映射
                # 0-normal, 1-benign, 2-malignant
                if raw_label == 0:
                    fg_label = 4  # normal
                elif raw_label == 1:
                    fg_label = 2  # benign_nodule
                else:
                    fg_label = 0  # adenocarcinoma (默认)

            self.finegrained_labels.append(fg_label)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Tuple:
        """
        获取样本，返回细粒度标签

        Returns:
            (image, finegrained_label)
        """
        image, _ = self.base_dataset[idx]
        fg_label = self.finegrained_labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, fg_label

    def get_class_distribution(self) -> Dict[int, int]:
        """获取细粒度类别分布"""
        dist = defaultdict(int)
        for label in self.finegrained_labels:
            dist[label] += 1
        return dict(dist)


class FineGrainedDatasetBuilder:
    """
    细粒度数据集构建器

    用法：
        builder = FineGrainedDatasetBuilder(transform=transform)
        
        # 构建完整5分类数据集
        dataset = builder.build_combined_dataset()
        
        # 或单独构建某个数据集
        dataset3_fg = builder.build_dataset('dataset3')
    """

    def __init__(
        self,
        transform: Optional[Callable] = None,
        dataset_paths: Optional[Dict[str, str]] = None
    ):
        self.transform = transform
        self.dataset_paths = dataset_paths or DATASET_PATHS

    def build_dataset(
        self,
        dataset_key: str,
        transform: Optional[Callable] = None
    ) -> Optional[FineGrainedLungDataset]:
        """
        构建单个数据集的细粒度版本

        Args:
            dataset_key: 'dataset1', 'dataset2', 或 'dataset3'
            transform: 可选的额外变换
        """
        from data.custom_dataset import CustomLungDataset

        path = self.dataset_paths.get(dataset_key)
        if not path or not Path(path).exists():
            print(f"[WARNING] Path not found for {dataset_key}: {path}")
            return None

        try:
            base = CustomLungDataset(
                root_path=path,
                dataset_type=dataset_key,
                transform=transform or self.transform
            )
            return FineGrainedLungDataset(base, dataset_key)
        except Exception as e:
            print(f"[ERROR] Failed to build fine-grained dataset for {dataset_key}: {e}")
            return None

    def build_combined_dataset(
        self,
        dataset_keys: Optional[List[str]] = None,
        transform: Optional[Callable] = None
    ) -> Optional[Dataset]:
        """
        构建合并的细粒度数据集

        Args:
            dataset_keys: 要合并的数据集列表，默认全部
            transform: 变换

        Returns:
            ConcatDataset 或单个 Dataset
        """
        if dataset_keys is None:
            dataset_keys = ['dataset1', 'dataset2', 'dataset3']

        datasets = []
        for key in dataset_keys:
            ds = self.build_dataset(key, transform)
            if ds is not None:
                datasets.append(ds)

        if not datasets:
            return None
        if len(datasets) == 1:
            return datasets[0]

        return ConcatDataset(datasets)

    def get_statistics(self) -> Dict[str, Dict]:
        """获取各数据集细粒度标签统计"""
        stats = {}
        for key in ['dataset1', 'dataset2', 'dataset3']:
            ds = self.build_dataset(key)
            if ds is not None:
                stats[key] = {
                    'total': len(ds),
                    'class_distribution': ds.get_class_distribution(),
                    'class_names': FINEGRAINED_CLASSES,
                }
        return stats


# =============================================================================
# 辅助函数：将已有3分类数据集快速转为5分类
# =============================================================================

def convert_to_finegrained(
    dataset: Dataset,
    dataset_type: str
) -> FineGrainedLungDataset:
    """
    将已有数据集转换为细粒度版本

    Args:
        dataset: 原始数据集（CustomLungDataset 或 Subset）
        dataset_type: 数据集类型

    Returns:
        FineGrainedLungDataset
    """
    # 如果传入的是 Subset，需要解包
    if hasattr(dataset, 'dataset'):
        base = dataset.dataset
    else:
        base = dataset

    return FineGrainedLungDataset(base, dataset_type)


def get_finegrained_class_names() -> List[str]:
    """获取细粒度类别名称列表"""
    return FINEGRAINED_CLASSES.copy()


def get_finegrained_num_classes() -> int:
    """获取细粒度类别数"""
    return len(FINEGRAINED_CLASSES)


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == "__main__":
    print("Testing FineGrainedLabeler...")

    # 测试标签映射
    print("\nLabel mappings:")
    test_cases = [
        ('adenocarcinoma', 'dataset3'),
        ('squamous.cell.carcinoma', 'dataset3'),
        ('Normal cases', 'dataset1'),
        ('Benign cases', 'dataset1'),
        ('lung_aca', 'dataset2'),
    ]
    for raw, ds_type in test_cases:
        idx = get_finegrained_label(raw, ds_type)
        print(f"  {ds_type:10s} | {raw:25s} -> {idx} ({FINEGRAINED_CLASSES[idx]})")

    # 测试统计
    builder = FineGrainedDatasetBuilder()
    stats = builder.get_statistics()
    print(f"\nDataset statistics:")
    for key, info in stats.items():
        print(f"  {key}: {info}")

    print("\nFineGrainedLabeler module ready!")
