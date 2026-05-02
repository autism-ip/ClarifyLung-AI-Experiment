"""
实验脚本公共工具函数
[INPUT]: torch, numpy, data.custom_dataset, training.trainer
[OUTPUT]: set_seed, get_device, split_dataset_with_transforms, train_model
[POS]: scripts/ 的共享工具模块，被所有实验脚本消费
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import random
from pathlib import Path
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


def create_quick_test_datasets(
    train_dataset,
    val_dataset,
    test_dataset,
    max_samples: int = 200,
    seed: int = 42,
):
    """
    创建快速测试用的数据集子集

    策略：按 70/15/15 比例从原数据集分层抽取，
    保证 train/val/test 结构不变，仅缩小规模。

    Returns:
        (train_subset, val_subset, test_subset)
    """
    from torch.utils.data import Subset

    g = torch.Generator().manual_seed(seed)

    train_n = int(max_samples * 0.70)
    val_n = int(max_samples * 0.15)
    test_n = max_samples - train_n - val_n

    def _subset(ds, n):
        if len(ds) <= n:
            return ds
        indices = torch.randperm(len(ds), generator=g)[:n].tolist()
        return Subset(ds, indices)

    return _subset(train_dataset, train_n), _subset(val_dataset, val_n), _subset(test_dataset, test_n)


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
    # 早停参数
    early_stopping_patience: int = 10,
    early_stopping_delta: float = 0.001,
    # 学习率预热参数
    warmup_epochs: int = 5,
    # 梯度裁剪参数
    max_grad_norm: float = 1.0,
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
        early_stopping_patience: 早停耐心值（默认10）
        early_stopping_delta: 早停最小改善（默认0.001）
        warmup_epochs: 学习率预热轮数（默认5）
        max_grad_norm: 梯度裁剪阈值（默认1.0）

    Returns:
        {
            'train_history': {'train_loss': [...], 'train_acc': [...], 'val_loss': [...], 'val_acc': [...]},
            'best_val_acc': float,
            'best_epoch': int,
            'early_stopped': bool,
        }
    """
    from training.trainer import TrainingConfig, get_optimizer

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    # AMP 混合精度训练 (GPU 环境下加速)
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler(device.type) if use_amp else None

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

    # 学习率调度：预热 + CosineAnnealing
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs  # 线性预热
        return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    early_stopped = False
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
            if use_amp and scaler is not None:
                with torch.amp.autocast(device.type):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                # 梯度裁剪
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
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
                    if use_amp and scaler is not None:
                        with torch.amp.autocast(device.type):
                            outputs = model(images)
                            loss = criterion(outputs, labels)
                    else:
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

        # 早停检查
        if val_loader is not None:
            if val_acc > best_val_acc + early_stopping_delta:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                patience_counter = 0
                if save_path is not None:
                    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), save_path)
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                print(f"{log_prefix}Early stopping at epoch {epoch+1} (patience={early_stopping_patience})")
                early_stopped = True
                break

        if (epoch + 1) % 10 == 0 or epoch == 0:
            log = f"{log_prefix}Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}"
            if val_loader is not None:
                log += f" | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}"
            print(log)

    return {
        'train_history': train_history,
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'early_stopped': early_stopped,
    }
