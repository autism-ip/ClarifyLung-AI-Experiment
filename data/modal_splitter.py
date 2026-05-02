"""
跨模态数据集分离器模块
[INPUT]: 合并后的数据集路径或Dataset实例
[OUTPUT]: X光子集、切片子集、按模态划分的train/val/test
[POS]: data/ 跨模态实验核心组件
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md

功能：将混合数据集分离为X光子集和切片子集
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
from collections import defaultdict

from torch.utils.data import Dataset, Subset, ConcatDataset

from configs.dataset_config import DATASET_PATHS, DATASET_INFO
from data.custom_dataset import CustomLungDataset


# =============================================================================
# 模态定义
# =============================================================================

MODALITY_XRAY = "xray"
MODALITY_HISTOPATHOLOGY = "histopathology"

# 数据集到模态的映射
DATASET_TO_MODALITY = {
    'dataset1': MODALITY_XRAY,           # IQ-OTHNCCD: X光
    'dataset2': MODALITY_HISTOPATHOLOGY, # LungColon: 组织切片
    'dataset3': MODALITY_HISTOPATHOLOGY, # Lung4Types: 组织切片（虽然含X光类数据，但源是切片集）
}

# 数据集名称映射
DATASET_NAMES = {
    'dataset1': 'IQ-OTHNCCD',
    'dataset2': 'LungColon',
    'dataset3': 'Lung4Types',
}


# =============================================================================
# 数据集分离
# =============================================================================

def load_datasets_by_modality(
    transform: Optional[Callable] = None,
    dataset_paths: Optional[Dict[str, str]] = None
) -> Dict[str, Dict[str, Dataset]]:
    """
    按模态加载数据集

    Args:
        transform: 图像变换
        dataset_paths: 自定义数据集路径，默认使用configs中的路径

    Returns:
        {
            'xray': {
                'dataset1': CustomLungDataset(...),
            },
            'histopathology': {
                'dataset2': CustomLungDataset(...),
                'dataset3': CustomLungDataset(...),
            }
        }
    """
    if dataset_paths is None:
        dataset_paths = DATASET_PATHS

    result = {
        MODALITY_XRAY: {},
        MODALITY_HISTOPATHOLOGY: {},
    }

    for dataset_key, path in dataset_paths.items():
        if not Path(path).exists():
            print(f"[WARNING] Dataset path not found: {path}, skipping {dataset_key}")
            continue

        modality = DATASET_TO_MODALITY.get(dataset_key)
        if modality is None:
            print(f"[WARNING] Unknown dataset key: {dataset_key}")
            continue

        try:
            dataset = CustomLungDataset(
                root_path=path,
                dataset_type=dataset_key,
                transform=transform
            )
            result[modality][dataset_key] = dataset
            print(f"[INFO] Loaded {dataset_key}: {len(dataset)} samples ({modality})")
        except Exception as e:
            print(f"[ERROR] Failed to load {dataset_key}: {e}")

    return result


def get_xray_datasets(
    transform: Optional[Callable] = None,
    dataset_paths: Optional[Dict[str, str]] = None
) -> Dict[str, Dataset]:
    """获取X光数据集（仅dataset1）"""
    by_modality = load_datasets_by_modality(transform, dataset_paths)
    return by_modality[MODALITY_XRAY]


def get_histopathology_datasets(
    transform: Optional[Callable] = None,
    dataset_paths: Optional[Dict[str, str]] = None
) -> Dict[str, Dataset]:
    """获取组织切片数据集（dataset2 + dataset3）"""
    by_modality = load_datasets_by_modality(transform, dataset_paths)
    return by_modality[MODALITY_HISTOPATHOLOGY]


def merge_modal_datasets(
    datasets_dict: Dict[str, Dataset]
) -> Optional[Dataset]:
    """
    合并同一模态下的多个数据集

    Args:
        datasets_dict: {dataset_key: Dataset}

    Returns:
        ConcatDataset 或 None
    """
    if not datasets_dict:
        return None

    datasets = list(datasets_dict.values())
    if len(datasets) == 1:
        return datasets[0]

    return ConcatDataset(datasets)


def split_by_modality(
    transform: Optional[Callable] = None,
    dataset_paths: Optional[Dict[str, str]] = None
) -> Tuple[Optional[Dataset], Optional[Dataset]]:
    """
    将数据集按模态分离

    Returns:
        (xray_dataset, histopathology_dataset)
    """
    by_modality = load_datasets_by_modality(transform, dataset_paths)

    xray = merge_modal_datasets(by_modality[MODALITY_XRAY])
    histo = merge_modal_datasets(by_modality[MODALITY_HISTOPATHOLOGY])

    return xray, histo


# =============================================================================
# 按模态划分 train/val/test
# =============================================================================

def split_modal_dataset_stratified(
    dataset: Dataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    num_classes: int = 3,
    seed: int = 42
) -> Tuple[Subset, Subset, Subset]:
    """
    对单个模态数据集进行分层划分

    使用 data_scarcity_sampler 中的分层逻辑
    """
    from data.data_scarcity_sampler import split_by_ratio_with_seed
    return split_by_ratio_with_seed(
        dataset, train_ratio, val_ratio, test_ratio, num_classes, seed
    )


class ModalSplitter:
    """
    跨模态数据集分离器

    用法：
        splitter = ModalSplitter(transform=train_transform)
        
        # 获取按模态分离的数据集
        xray_data, histo_data = splitter.get_modal_datasets()
        
        # 获取按模态+划分的数据集
        xray_train, xray_val, xray_test = splitter.get_xray_split()
        histo_train, histo_val, histo_test = splitter.get_histopathology_split()
        
        # 获取跨模态迁移配置
        # 实验A: X光训练 -> 切片测试
        exp_a = splitter.get_cross_modal_config('xray_to_histopathology')
        # 实验B: 切片训练 -> X光测试
        exp_b = splitter.get_cross_modal_config('histopathology_to_xray')
    """

    def __init__(
        self,
        transform: Optional[Callable] = None,
        dataset_paths: Optional[Dict[str, str]] = None,
        num_classes: int = 3,
        seed: int = 42
    ):
        self.transform = transform
        self.dataset_paths = dataset_paths or DATASET_PATHS
        self.num_classes = num_classes
        self.seed = seed

        # 懒加载缓存
        self._by_modality: Optional[Dict[str, Dict[str, Dataset]]] = None
        self._xray_dataset: Optional[Dataset] = None
        self._histo_dataset: Optional[Dataset] = None

    def _load(self):
        """懒加载数据集"""
        if self._by_modality is None:
            self._by_modality = load_datasets_by_modality(
                self.transform, self.dataset_paths
            )
            self._xray_dataset = merge_modal_datasets(
                self._by_modality[MODALITY_XRAY]
            )
            self._histo_dataset = merge_modal_datasets(
                self._by_modality[MODALITY_HISTOPATHOLOGY]
            )

    def get_modal_datasets(self) -> Tuple[Optional[Dataset], Optional[Dataset]]:
        """获取按模态分离的数据集"""
        self._load()
        return self._xray_dataset, self._histo_dataset

    def get_xray_dataset(self) -> Optional[Dataset]:
        """获取X光数据集"""
        self._load()
        return self._xray_dataset

    def get_histopathology_dataset(self) -> Optional[Dataset]:
        """获取组织切片数据集"""
        self._load()
        return self._histo_dataset

    def get_xray_split(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Tuple[Optional[Subset], Optional[Subset], Optional[Subset]]:
        """获取X光数据集的train/val/test划分"""
        dataset = self.get_xray_dataset()
        if dataset is None:
            return None, None, None
        return split_modal_dataset_stratified(
            dataset, train_ratio, val_ratio, test_ratio,
            self.num_classes, self.seed
        )

    def get_histopathology_split(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Tuple[Optional[Subset], Optional[Subset], Optional[Subset]]:
        """获取组织切片数据集的train/val/test划分"""
        dataset = self.get_histopathology_dataset()
        if dataset is None:
            return None, None, None
        return split_modal_dataset_stratified(
            dataset, train_ratio, val_ratio, test_ratio,
            self.num_classes, self.seed
        )

    def get_cross_modal_config(
        self,
        experiment_type: str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.2
    ) -> Dict:
        """
        获取跨模态迁移实验配置

        Args:
            experiment_type: 'xray_to_histopathology' 或 'histopathology_to_xray'
            train_ratio: 源模态训练集比例（源模态无test，全部用于训练）
            val_ratio: 源模态验证集比例

        Returns:
            实验配置字典，包含 train_loader, val_loader, test_loader
        """
        self._load()

        if experiment_type == 'xray_to_histopathology':
            # 实验A: X光训练 -> 切片测试
            source = self.get_xray_dataset()
            target = self.get_histopathology_dataset()
            source_name = "X-ray"
            target_name = "Histopathology"
        elif experiment_type == 'histopathology_to_xray':
            # 实验B: 切片训练 -> X光测试
            source = self.get_histopathology_dataset()
            target = self.get_xray_dataset()
            source_name = "Histopathology"
            target_name = "X-ray"
        else:
            raise ValueError(f"Unknown experiment_type: {experiment_type}")

        if source is None or target is None:
            raise RuntimeError(f"Missing dataset for {experiment_type}")

        # 源模态：划分为 train/val（用于训练和验证）
        source_train, source_val, _ = split_modal_dataset_stratified(
            source, train_ratio, val_ratio, 0.0, self.num_classes, self.seed
        )

        # 目标模态：全部作为测试集（或划分为test）
        target_train, target_val, target_test = split_modal_dataset_stratified(
            target, 0.0, 0.0, 1.0, self.num_classes, self.seed
        )

        return {
            'experiment_type': experiment_type,
            'source_name': source_name,
            'target_name': target_name,
            'source_train': source_train,
            'source_val': source_val,
            'target_test': target_test,
            'source_dataset': source,
            'target_dataset': target,
        }

    def get_statistics(self) -> Dict:
        """获取各模态统计信息"""
        self._load()

        stats = {
            'xray': {},
            'histopathology': {},
        }

        # X光统计
        if self._xray_dataset:
            stats['xray']['total'] = len(self._xray_dataset)
            # 统计各类别数量
            class_counts = defaultdict(int)
            for i in range(len(self._xray_dataset)):
                _, label = self._xray_dataset[i]
                class_counts[int(label)] += 1
            stats['xray']['class_distribution'] = dict(class_counts)

        # 切片统计
        if self._histo_dataset:
            stats['histopathology']['total'] = len(self._histo_dataset)
            class_counts = defaultdict(int)
            for i in range(len(self._histo_dataset)):
                _, label = self._histo_dataset[i]
                class_counts[int(label)] += 1
            stats['histopathology']['class_distribution'] = dict(class_counts)

        return stats


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == "__main__":
    print("Testing ModalSplitter...")

    # 检查数据集路径
    print("\nDataset paths:")
    for k, v in DATASET_PATHS.items():
        exists = "EXISTS" if Path(v).exists() else "NOT FOUND"
        print(f"  {k}: {v} [{exists}]")

    # 测试分离器（如果数据集存在）
    if any(Path(p).exists() for p in DATASET_PATHS.values()):
        splitter = ModalSplitter(num_classes=3, seed=42)

        xray, histo = splitter.get_modal_datasets()
        print(f"\nX-ray dataset: {len(xray) if xray else 'None'} samples")
        print(f"Histopathology dataset: {len(histo) if histo else 'None'} samples")

        stats = splitter.get_statistics()
        print(f"\nStatistics:")
        for modality, info in stats.items():
            print(f"  {modality}: {info}")
    else:
        print("\nNo datasets found locally, skipping integration test.")

    print("\nModalSplitter module ready!")
