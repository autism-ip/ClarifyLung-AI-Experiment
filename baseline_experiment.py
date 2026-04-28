#!/usr/bin/env python3
"""
Baseline Model Comparison Experiment (Small Scale)
Trains 3 baseline models (ResNet50, ViT-B/16, HybridModel) on a stratified subset

[INPUT]: configs.DATASET_PATHS, torch, timm
[OUTPUT]: 3 model validation accuracies
[POS]: 根目录快速验证脚本，小规模真实数据端到端验证
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, ConcatDataset, WeightedRandomSampler
import numpy as np
import timm

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.custom_dataset import CustomLungDataset, create_weighted_sampler
from data.augmentation import get_train_augmentation, get_val_augmentation
from configs import DATASET_PATHS
from model import HybridModel


# =============================================================================
# 配置
# =============================================================================

SAMPLE_SIZE = 200
TRAIN_RATIO = 0.8
EPOCHS = 5
BATCH_SIZE = 16
SEED = 42


# =============================================================================
# 工具函数
# =============================================================================

def stratified_sample_from_dataset(dataset, n_samples, generator):
    """
    从单个数据集中按类别比例分层抽样

    Returns:
        List[int]: 抽样的索引列表
    """
    # 获取所有标签
    if hasattr(dataset, 'labels'):
        labels = np.array(dataset.labels)
    elif hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'labels'):
        labels = np.array(dataset.dataset.labels)
    else:
        labels = np.array([dataset[i][1] for i in range(len(dataset))])

    classes = np.unique(labels)
    indices_per_class = {c: np.where(labels == c)[0] for c in classes}

    # 按类别比例分配样本数
    total = len(labels)
    sampled_indices = []
    for c in classes:
        class_count = len(indices_per_class[c])
        class_ratio = class_count / total
        n_class_samples = max(1, int(n_samples * class_ratio))
        n_class_samples = min(n_class_samples, class_count)

        perm = torch.randperm(class_count, generator=generator).numpy()
        selected = indices_per_class[c][perm[:n_class_samples]]
        sampled_indices.extend(selected.tolist())

    return sampled_indices


def create_stratified_subset(datasets, total_samples, generator):
    """
    从多个数据集中分层抽样创建子集

    Args:
        datasets: 数据集列表
        total_samples: 总样本数
        generator: torch.Generator

    Returns:
        ConcatDataset, dict (各数据集抽样信息)
    """
    n_datasets = len(datasets)
    samples_per_dataset = total_samples // n_datasets
    remainder = total_samples % n_datasets

    all_indices = []
    info = {}

    for i, ds in enumerate(datasets):
        n = samples_per_dataset + (1 if i < remainder else 0)
        sampled = stratified_sample_from_dataset(ds, n, generator)
        all_indices.extend([(i, idx) for idx in sampled])
        info[f'dataset{i+1}'] = {
            'sampled': len(sampled),
            'distribution': {label: sum(1 for _, idx in all_indices[-len(sampled):]
                                       if datasets[i][idx][1] == label)
                           for label in set(datasets[i][idx][1] for _, idx in all_indices[-len(sampled):])}
        }

    return all_indices, info


def create_subset_from_indices(datasets, tuple_indices):
    """从 (dataset_idx, sample_idx) 元组列表创建 ConcatDataset"""
    subsets = []
    for i, ds in enumerate(datasets):
        idxs = [idx for d_idx, idx in tuple_indices if d_idx == i]
        if idxs:
            subsets.append(Subset(ds, idxs))
    return ConcatDataset(subsets)


# =============================================================================
# 训练函数
# =============================================================================

def train_model(model, name, train_loader, val_loader, epochs, device):
    """训练单个模型并返回验证准确率"""
    print(f"\n--- Training {name} ---")
    print(f"  Device: {device}")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch+1}/{epochs}: loss = {avg_loss:.4f}")

    # Evaluate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total if total > 0 else 0.0
    print(f"  Validation accuracy: {accuracy:.2%}")
    return accuracy


# =============================================================================
# 主流程
# =============================================================================

print("=" * 60)
print("Baseline Model Comparison Experiment (Small Scale)")
print("=" * 60)

# 独立的随机生成器，不污染全局状态
generator = torch.Generator().manual_seed(SEED)

# 加载数据集
train_transform = get_train_augmentation(224)
val_transform = get_val_augmentation(224)

print("\nLoading datasets...")
ds1 = CustomLungDataset(DATASET_PATHS['dataset1'], 'dataset1', transform=train_transform)
ds2 = CustomLungDataset(DATASET_PATHS['dataset2'], 'dataset2', transform=train_transform)
ds3 = CustomLungDataset(DATASET_PATHS['dataset3'], 'dataset3', transform=train_transform)

print(f"  Dataset 1 (IQ-OTHNCCD): {len(ds1)} samples, dist={ds1.get_class_distribution()}")
print(f"  Dataset 2 (LungColon):  {len(ds2)} samples, dist={ds2.get_class_distribution()}")
print(f"  Dataset 3 (Lung4Types): {len(ds3)} samples, dist={ds3.get_class_distribution()}")

# 分层抽样
tuple_indices, sample_info = create_stratified_subset([ds1, ds2, ds3], SAMPLE_SIZE, generator)
small_dataset = create_subset_from_indices([ds1, ds2, ds3], tuple_indices)

print(f"\nStratified sample: {len(small_dataset)} total")
for key, val in sample_info.items():
    print(f"  {key}: {val['sampled']} samples, dist={val['distribution']}")

# 合并数据集用于验证加载
train_size = int(TRAIN_RATIO * len(small_dataset))
val_size = len(small_dataset) - train_size
train_subset, val_subset = torch.utils.data.random_split(
    small_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED)
)

print(f"\nTrain subset: {len(train_subset)}, Val subset: {len(val_subset)}")

# 检查类别不平衡并应用加权采样
all_labels = []
for i in range(len(train_subset)):
    _, label = train_subset[i]
    all_labels.append(label)

class_counts = {c: all_labels.count(c) for c in set(all_labels)}
print(f"Train class distribution: {class_counts}")

# 如果类别不平衡严重，使用 WeightedRandomSampler
max_count = max(class_counts.values())
imbalance_ratio = max_count / min(class_counts.values()) if min(class_counts.values()) > 0 else 1.0

if imbalance_ratio > 2.0:
    print(f"[WARNING] Class imbalance detected (ratio={imbalance_ratio:.1f}), using WeightedRandomSampler")
    sampler = create_weighted_sampler(train_subset)
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, sampler=sampler)
else:
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)

val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. ResNet50
print("\n[1/3] ResNet50 Baseline")
resnet = timm.create_model('resnet50', pretrained=True, num_classes=3)
resnet_accuracy = train_model(resnet, "ResNet50", train_loader, val_loader, EPOCHS, device)

# 2. ViT
print("\n[2/3] ViT-B/16 Baseline")
vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=3)
vit_accuracy = train_model(vit, "ViT-B/16", train_loader, val_loader, EPOCHS, device)

# 3. HybridModel
print("\n[3/3] HybridModel (CNN-Transformer)")
hybrid = HybridModel(num_classes=3, model_dim=128, nhead=4, num_layers=2, dropout=0.1)
hybrid_accuracy = train_model(hybrid, "HybridModel", train_loader, val_loader, EPOCHS, device)

# Summary
print("\n" + "=" * 60)
print("Baseline Comparison Summary")
print("=" * 60)
print(f"ResNet50:     {resnet_accuracy:.2%}")
print(f"ViT-B/16:     {vit_accuracy:.2%}")
print(f"HybridModel:  {hybrid_accuracy:.2%}")
print("=" * 60)
print(f"\nNOTE: Small-scale experiment on {SAMPLE_SIZE} samples.")
print("      Results may not reflect full-dataset performance.")
