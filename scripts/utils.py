"""
实验脚本公共工具函数
[INPUT]: torch, numpy, data.custom_dataset, training.trainer
[OUTPUT]: set_seed, get_device, split_dataset_with_transforms, train_model
[POS]: scripts/ 的共享工具模块，被所有实验脚本消费
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import random
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np


def set_seed(seed: int):
    """设置全局随机种子，保证实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # 确定性行为：牺牲性能换取完全可复现
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    """获取最优计算设备 (GPU优先)"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print(f"[INFO] Using CPU")
    return device


def split_dataset_with_transforms(
    dataset1_path: str,
    dataset2_path: str,
    dataset3_path: str,
    train_transform,
    val_transform,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
):
    """
    加载三个数据集并划分为 train/val/test，确保验证/测试集使用正确的变换

    策略：分别创建带不同 transform 的数据集版本，用相同种子做 random_split，
    保证划分索引一致，然后取对应子集。
    """
    from data.custom_dataset import merge_datasets

    # 分别创建带不同 transform 的数据集
    dataset_train = merge_datasets(dataset1_path, dataset2_path, dataset3_path, transform=train_transform)
    dataset_val = merge_datasets(dataset1_path, dataset2_path, dataset3_path, transform=val_transform)
    dataset_test = merge_datasets(dataset1_path, dataset2_path, dataset3_path, transform=val_transform)

    total_size = len(dataset_train)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    # 使用相同种子分别划分，保证索引一致
    g = torch.Generator().manual_seed(seed)
    train_raw, _, _ = random_split(dataset_train, [train_size, val_size, test_size], generator=g)
    g = torch.Generator().manual_seed(seed)
    _, val_raw, _ = random_split(dataset_val, [train_size, val_size, test_size], generator=g)
    g = torch.Generator().manual_seed(seed)
    _, _, test_raw = random_split(dataset_test, [train_size, val_size, test_size], generator=g)

    return train_raw, val_raw, test_raw


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    epochs: int,
    device: torch.device,
    learning_rate: float = 1e-4,
    transformer_lr: float = 5e-4,
    weight_decay: float = 0.01,
    save_path: Optional[str] = None,
    log_prefix: str = "",
) -> dict:
    """
    通用训练函数，封装训练循环、验证、检查点保存

    Args:
        model: 待训练模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器（可为 None）
        epochs: 训练轮数
        device: 计算设备
        learning_rate: CNN/其他参数学习率
        transformer_lr: Transformer参数学习率（若模型不支持差分学习率，则退化为统一LR）
        weight_decay: 权重衰减
        save_path: 最佳模型保存路径（None则不保存）
        log_prefix: 日志前缀（如 "Fold 1, "）

    Returns:
        {
            'train_history': {'train_loss': [...], 'train_acc': [...], 'val_loss': [...], 'val_acc': [...]},
            'best_val_acc': float,
        }
    """
    from training.trainer import TrainingConfig, get_optimizer

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    # 差分学习率：尝试前缀匹配分组，失败则回退统一LR
    try:
        trainer_config = TrainingConfig(
            learning_rate=learning_rate,
            transformer_lr=transformer_lr,
            weight_decay=weight_decay,
        )
        optimizer = get_optimizer(model, trainer_config)
    except Exception:
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    train_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total

        # 验证
        val_loss = 0.0
        val_acc = 0.0
        if val_loader is not None:
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()

            val_loss /= len(val_loader)
            val_acc = val_correct / val_total

        scheduler.step()

        train_history['train_loss'].append(train_loss)
        train_history['train_acc'].append(train_acc)
        train_history['val_loss'].append(val_loss)
        train_history['val_acc'].append(val_acc)

        # 保存最佳模型
        if val_loader is not None and val_acc > best_val_acc:
            best_val_acc = val_acc
            if save_path is not None:
                torch.save(model.state_dict(), save_path)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            log = f"{log_prefix}Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}"
            if val_loader is not None:
                log += f" | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}"
            print(log)

    return {
        'train_history': train_history,
        'best_val_acc': best_val_acc,
    }
