"""
数据稀缺采样器模块
[INPUT]: Dataset, ratio(0.01-1.0), num_classes, random_seed
[OUTPUT]: Subset（确保每类都有样本）
[POS]: data/ 数据工程核心组件，支持分层稀缺采样
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md

功能：从完整数据集分层采样指定比例的数据
要求：
  - 分层采样：每类样本数 = 原始该类样本数 * ratio（向上取整，至少1张）
  - 支持随机种子（可复现）
  - 返回采样后的索引列表
"""

import random
from typing import List, Dict, Tuple, Optional, Union
from collections import defaultdict

import numpy as np
from torch.utils.data import Dataset, Subset


def stratified_sample_indices(
    labels: List[int],
    ratio: float,
    num_classes: int,
    random_seed: int = 42,
    min_samples_per_class: int = 1
) -> List[int]:
    """
    分层采样索引

    Args:
        labels: 完整数据集的标签列表
        ratio: 采样比例 (0.01 ~ 1.0)
        num_classes: 类别总数
        random_seed: 随机种子
        min_samples_per_class: 每类最少保留样本数

    Returns:
        采样后的索引列表

    示例:
        >>> labels = [0, 0, 1, 1, 2, 2]
        >>> idx = stratified_sample_indices(labels, 0.5, 3, seed=42)
        >>> # 每类至少保留 ceil(2*0.5)=1 个
    """
    if not 0 < ratio <= 1.0:
        raise ValueError(f"ratio must be in (0, 1.0], got {ratio}")

    # 按类别分组索引
    class_to_indices: Dict[int, List[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        class_to_indices[label].append(idx)

    # 验证所有类别都存在
    for c in range(num_classes):
        if c not in class_to_indices:
            raise ValueError(f"Class {c} not found in dataset. Found classes: {list(class_to_indices.keys())}")

    # 设置随机种子
    rng = random.Random(random_seed)

    sampled_indices: List[int] = []
    for class_id in range(num_classes):
        indices = class_to_indices[class_id]
        n_total = len(indices)

        # 计算该类需要采样的数量（向上取整，至少min_samples_per_class个）
        n_sample = max(min_samples_per_class, int(np.ceil(n_total * ratio)))
        # 确保不超过总数
        n_sample = min(n_sample, n_total)

        # 随机采样（不放回）
        sampled = rng.sample(indices, n_sample)
        sampled_indices.extend(sampled)

    # 按原始顺序排序，保持数据顺序一致性
    sampled_indices.sort()
    return sampled_indices


def create_stratified_subset(
    dataset: Dataset,
    ratio: float,
    num_classes: int,
    random_seed: int = 42,
    min_samples_per_class: int = 1
) -> Subset:
    """
    创建分层采样的子集

    Args:
        dataset: 原始数据集
        ratio: 采样比例
        num_classes: 类别数
        random_seed: 随机种子
        min_samples_per_class: 每类最少样本数

    Returns:
        torch.utils.data.Subset 实例

    示例:
        >>> subset = create_stratified_subset(dataset, ratio=0.1, num_classes=3, random_seed=42)
        >>> print(f"原始: {len(dataset)}, 采样后: {len(subset)}")
    """
    # 提取所有标签
    labels = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        labels.append(int(label))

    indices = stratified_sample_indices(
        labels=labels,
        ratio=ratio,
        num_classes=num_classes,
        random_seed=random_seed,
        min_samples_per_class=min_samples_per_class
    )

    return Subset(dataset, indices)


class DataScarcitySampler:
    """
    数据稀缺采样器 - 支持多比例、多种子、可复现实验

    用法：
        sampler = DataScarcitySampler(dataset, num_classes=3)
        
        # 获取不同比例的子集
        for ratio in [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]:
            subset = sampler.sample(ratio=ratio, seed=42)
            print(f"Ratio {ratio}: {len(subset)} samples")
    """

    # 预定义的标准采样比例（对应Feature 2中的7个数据比例）
    STANDARD_RATIOS = [0.01, 0.05, 0.10, 0.25, 0.50, 1.0]

    def __init__(
        self,
        dataset: Dataset,
        num_classes: int,
        labels: Optional[List[int]] = None
    ):
        """
        初始化采样器

        Args:
            dataset: 原始数据集
            num_classes: 类别数
            labels: 可选，预计算的标签列表（避免重复遍历）
        """
        self.dataset = dataset
        self.num_classes = num_classes

        if labels is None:
            self.labels = []
            for i in range(len(dataset)):
                _, label = dataset[i]
                self.labels.append(int(label))
        else:
            self.labels = [int(l) for l in labels]

        # 预计算每类样本数
        self.class_counts: Dict[int, int] = defaultdict(int)
        for label in self.labels:
            self.class_counts[label] += 1

    def sample(
        self,
        ratio: float,
        seed: int = 42,
        min_per_class: int = 1
    ) -> Subset:
        """
        采样指定比例的子集

        Args:
            ratio: 采样比例 (0, 1.0]
            seed: 随机种子
            min_per_class: 每类最少保留样本数

        Returns:
            Subset实例
        """
        indices = stratified_sample_indices(
            labels=self.labels,
            ratio=ratio,
            num_classes=self.num_classes,
            random_seed=seed,
            min_samples_per_class=min_per_class
        )
        return Subset(self.dataset, indices)

    def sample_all_ratios(
        self,
        ratios: Optional[List[float]] = None,
        seeds: Optional[List[int]] = None
    ) -> Dict[Tuple[float, int], Subset]:
        """
        批量采样多个比例和种子组合

        Args:
            ratios: 采样比例列表，默认使用STANDARD_RATIOS
            seeds: 随机种子列表，默认[42]

        Returns:
            字典: {(ratio, seed): Subset}
        """
        if ratios is None:
            ratios = self.STANDARD_RATIOS
        if seeds is None:
            seeds = [42]

        results = {}
        for ratio in ratios:
            for seed in seeds:
                subset = self.sample(ratio=ratio, seed=seed)
                results[(ratio, seed)] = subset

        return results

    def get_class_distribution(self) -> Dict[int, int]:
        """获取原始数据集的类别分布"""
        return dict(self.class_counts)

    def get_sampled_distribution(self, ratio: float) -> Dict[int, int]:
        """计算采样后的类别分布（理论值）"""
        dist = {}
        for class_id, count in self.class_counts.items():
            dist[class_id] = max(1, int(np.ceil(count * ratio)))
        return dist


def split_by_ratio_with_seed(
    dataset: Dataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    num_classes: int = 3,
    seed: int = 42
) -> Tuple[Subset, Subset, Subset]:
    """
    按分层比例划分 train/val/test（基于稀缺采样器）

    注意：此函数用于在采样后的子集上做进一步划分，
    或直接在完整数据集上做标准划分。

    Args:
        dataset: 原始数据集
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        num_classes: 类别数
        seed: 随机种子

    Returns:
        (train_subset, val_subset, test_subset)
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    # 获取标签
    labels = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        labels.append(int(label))

    # 按类别分组
    class_to_indices: Dict[int, List[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        class_to_indices[label].append(idx)

    rng = random.Random(seed)
    train_idx, val_idx, test_idx = [], [], []

    for class_id in range(num_classes):
        indices = class_to_indices[class_id].copy()
        rng.shuffle(indices)

        n_total = len(indices)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        # 测试集取剩余

        train_idx.extend(indices[:n_train])
        val_idx.extend(indices[n_train:n_train + n_val])
        test_idx.extend(indices[n_train + n_val:])

    return (
        Subset(dataset, sorted(train_idx)),
        Subset(dataset, sorted(val_idx)),
        Subset(dataset, sorted(test_idx))
    )


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == "__main__":
    print("Testing DataScarcitySampler...")

    # 创建一个模拟数据集
    class DummyDataset(Dataset):
        def __init__(self, n_per_class=100, num_classes=3):
            self.data = []
            for c in range(num_classes):
                for i in range(n_per_class):
                    self.data.append((f"img_{c}_{i}", c))

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    dataset = DummyDataset(n_per_class=100, num_classes=3)
    print(f"Original dataset: {len(dataset)} samples")

    sampler = DataScarcitySampler(dataset, num_classes=3)
    print(f"Class distribution: {sampler.get_class_distribution()}")

    # 测试不同比例
    for ratio in [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]:
        subset = sampler.sample(ratio=ratio, seed=42)
        print(f"  Ratio {ratio:4.0%}: {len(subset):4d} samples")

    # 测试可复现性
    s1 = sampler.sample(ratio=0.1, seed=123)
    s2 = sampler.sample(ratio=0.1, seed=123)
    assert [dataset[i] for i in s1.indices] == [dataset[i] for i in s2.indices]
    print("\nReproducibility check: PASSED")

    # 测试分层split
    train, val, test = split_by_ratio_with_seed(dataset, 0.7, 0.15, 0.15, 3, 42)
    print(f"\nSplit: train={len(train)}, val={len(val)}, test={len(test)}")

    print("\nAll DataScarcitySampler tests passed!")
