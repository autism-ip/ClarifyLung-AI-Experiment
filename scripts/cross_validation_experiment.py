#!/usr/bin/env python3
"""
交叉验证实验脚本
5折分层交叉验证 + 统计显著性检验

[INPUT]: 配置文件路径, K折数
[OUTPUT]: 各折结果, 统计检验结果, 置信区间
[POS]: scripts/ 交叉验证实验执行脚本
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import os
# 设置HuggingFace镜像（解决网络问题）
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.custom_dataset import merge_datasets
from data.augmentation import get_train_augmentation, get_val_augmentation
from configs import DATASET_PATHS
from models import HybridModel
from training.trainer import TrainingConfig, get_optimizer
from experiments.cross_validation import KFoldCrossValidator
from experiments.metrics import compute_metrics
from experiments.visualization import plot_training_curves
from scripts.utils import set_seed, get_device


# =============================================================================
# 实验配置
# =============================================================================

class CrossValidationConfig:
    """交叉验证实验配置"""

    # 数据集配置
    dataset1_path: str = DATASET_PATHS['dataset1']
    dataset2_path: str = DATASET_PATHS['dataset2']
    dataset3_path: str = DATASET_PATHS['dataset3']

    # 交叉验证配置
    n_folds: int = 5
    seed: int = 42

    # 训练配置
    batch_size: int = 32
    num_epochs: int = 30
    num_workers: int = 4

    # 优化器配置
    learning_rate: float = 1e-4
    transformer_lr: float = 5e-4
    weight_decay: float = 0.01

    # 其他
    image_size: int = 224
    num_classes: int = 3

    # 输出配置
    output_dir: str = "outputs/cross_validation"
    save_checkpoints: bool = True

    # 快速测试模式
    quick_test: bool = False


# =============================================================================
# 训练和评估函数
# =============================================================================

def train_fold(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    device: torch.device,
    config: CrossValidationConfig,
    fold: int,
    # 早停参数
    early_stopping_patience: int = 10,
    early_stopping_delta: float = 0.001,
    # 学习率预热参数
    warmup_epochs: int = 5,
    # 梯度裁剪参数
    max_grad_norm: float = 1.0,
) -> Tuple[Dict, Dict, float]:
    """训练单个折，返回训练历史、验证指标、训练时间"""

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    # AMP 混合精度训练
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler(device.type) if use_amp else None

    # 使用差分学习率（CNN小LR，Transformer大LR）
    trainer_config = TrainingConfig(
        learning_rate=config.learning_rate,
        transformer_lr=config.transformer_lr,
        weight_decay=config.weight_decay,
    )
    optimizer = get_optimizer(model, trainer_config)
    
    # 学习率调度：预热 + CosineAnnealing
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    train_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    early_stopped = False
    start_time = time.time()

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
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total

        # 验证
        model.eval()
        val_loss = 0.0
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
        if val_acc > best_val_acc + early_stopping_delta:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print(f"    Fold {fold+1}: Early stopping at epoch {epoch+1}")
            early_stopped = True
            break

        if (epoch + 1) % 10 == 0:
            print(f"    Fold {fold+1}, Epoch {epoch+1}/{epochs}: "
                  f"Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}")

    training_time = time.time() - start_time

    # 最终验证：在验证集上计算真实指标（非简化）
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # 保存预测数组供可视化CLI复用
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"fold_{fold+1}_y_true.npy", all_labels)
    np.save(out_dir / f"fold_{fold+1}_y_pred.npy", all_preds)
    np.save(out_dir / f"fold_{fold+1}_y_prob.npy", all_probs)

    # 计算完整指标（包含曲线数据）
    metrics = compute_metrics(all_labels, all_preds, all_probs)
    
    # 保存曲线数据和混淆矩阵
    metrics.save_curves(str(out_dir), prefix=f"fold_{fold+1}_")

    val_metrics = {
        'accuracy': metrics.accuracy,
        'macro_f1': metrics.f1_macro,
        'weighted_f1': metrics.f1_weighted,
        'auc_roc': metrics.auc_roc_ovr,
        'auc_roc_ovo': metrics.auc_roc_ovo,
        'precision_per_class': metrics.precision,
        'recall_per_class': metrics.sensitivity,
        'specificity_per_class': metrics.specificity,
        'auc_per_class': metrics.auc_per_class,
    }

    return train_history, val_metrics, training_time


# =============================================================================
# 主实验流程
# =============================================================================

def run_cross_validation_experiment(config: CrossValidationConfig):
    """运行交叉验证实验"""

    print("\n" + "="*60)
    print("交叉验证实验")
    print("="*60)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"设备: {get_device()}")
    print(f"K折数: {config.n_folds}")
    print(f"输出目录: {config.output_dir}")

    # 设置随机种子
    set_seed(config.seed)

    # 创建输出目录
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 准备数据
    print("\n[1/5] 加载数据集...")
    train_transform = get_train_augmentation(config.image_size)
    val_transform = get_val_augmentation(config.image_size)

    # 用于分层划分的数据集（只需标签，使用 val_transform 避免训练增强干扰）
    dataset_for_split = merge_datasets(
        config.dataset1_path,
        config.dataset2_path,
        config.dataset3_path,
        transform=val_transform
    )
    # 用于训练的数据集（带训练增强）
    dataset_train = merge_datasets(
        config.dataset1_path,
        config.dataset2_path,
        config.dataset3_path,
        transform=train_transform
    )

    # 快速测试模式
    if config.quick_test:
        print("\n[QUICK TEST MODE] 启用快速测试: 200样本子集, 3 folds, 1 epoch, batch=8, workers=0")
        from torch.utils.data import Subset
        g = torch.Generator().manual_seed(config.seed)
        indices = torch.randperm(len(dataset_for_split), generator=g)[:200].tolist()
        dataset_for_split = Subset(dataset_for_split, indices)
        dataset_train = Subset(dataset_train, indices)
        config.n_folds = 3
        config.num_epochs = 1
        config.batch_size = min(config.batch_size, 8)
        config.num_workers = 0
        config.save_checkpoints = False

    print(f"  合并数据集大小: {len(dataset_for_split)}")

    device = get_device()

    # 创建K折验证器
    validator = KFoldCrossValidator(num_folds=config.n_folds, random_seed=config.seed)

    # 创建所有折（基于 val_transform 数据集做分层）
    all_folds = validator.create_folds(dataset_for_split)

    # 存储结果
    fold_results = []
    all_histories = []

    print(f"\n[2/5] 开始 {config.n_folds} 折交叉验证...")

    for fold_idx in range(config.n_folds):
        print(f"\n{'='*40}")
        print(f"Fold {fold_idx + 1}/{config.n_folds}")
        print(f"{'='*40}")

        # 获取当前折的训练和验证索引
        train_indices, val_indices = all_folds[fold_idx]

        from torch.utils.data import Subset

        # 训练子集使用 train_transform，验证子集使用 val_transform
        train_subset = Subset(dataset_train, train_indices.tolist())
        val_subset = Subset(dataset_for_split, val_indices.tolist())

        print(f"  训练集: {len(train_subset)}, 验证集: {len(val_subset)}")

        # 创建数据加载器
        train_loader = DataLoader(
            train_subset, batch_size=config.batch_size,
            shuffle=True, num_workers=config.num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            val_subset, batch_size=config.batch_size,
            shuffle=False, num_workers=config.num_workers, pin_memory=True
        )

        # 创建模型 (使用统一 HybridModel，加载预训练权重)
        model = HybridModel(
            num_classes=config.num_classes,
            model_dim=768,
            nhead=12,
            num_layers=12,
            dropout=0.1,
            pretrained=True
        )

        # 训练
        train_history, val_metrics, training_time = train_fold(
            model, train_loader, val_loader,
            config.num_epochs, device, config, fold_idx
        )

        # K折交叉验证中，验证指标即该折的泛化结果
        # 不重复用验证集作为"测试集"，避免数据泄漏
        fold_result = {
            'fold': fold_idx + 1,
            'val_accuracy': val_metrics['accuracy'],
            'val_f1': val_metrics['macro_f1'],
            'val_weighted_f1': val_metrics['weighted_f1'],
            'val_auc': val_metrics['auc_roc'],
            'val_auc_ovo': val_metrics['auc_roc_ovo'],
            'val_precision_per_class': val_metrics['precision_per_class'],
            'val_recall_per_class': val_metrics['recall_per_class'],
            'val_specificity_per_class': val_metrics['specificity_per_class'],
            'val_auc_per_class': val_metrics['auc_per_class'],
            'training_time': training_time
        }
        fold_results.append(fold_result)
        all_histories.append({f'fold_{fold_idx+1}': train_history})

        print(f"  Fold {fold_idx+1} 结果:")
        print(f"    Val Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['macro_f1']:.4f}")

        # 保存checkpoint
        if config.save_checkpoints:
            torch.save(model.state_dict(), output_dir / f"fold_{fold_idx+1}_best.pth")

    # 计算汇总统计
    print("\n[3/5] 计算汇总统计...")

    val_accs = [r['val_accuracy'] for r in fold_results]
    val_f1s = [r['val_f1'] for r in fold_results]
    val_weighted_f1s = [r['val_weighted_f1'] for r in fold_results]
    val_aucs = [r['val_auc'] for r in fold_results]
    val_aucs_ovo = [r['val_auc_ovo'] for r in fold_results]
    times = [r['training_time'] for r in fold_results]

    # 计算每类指标的均值和标准差
    num_classes = len(fold_results[0]['val_precision_per_class'])
    precision_per_class = {i: [] for i in range(num_classes)}
    recall_per_class = {i: [] for i in range(num_classes)}
    specificity_per_class = {i: [] for i in range(num_classes)}
    auc_per_class = {i: [] for i in range(num_classes)}
    
    for r in fold_results:
        for i in range(num_classes):
            precision_per_class[i].append(r['val_precision_per_class'][i])
            recall_per_class[i].append(r['val_recall_per_class'][i])
            specificity_per_class[i].append(r['val_specificity_per_class'][i])
            auc_per_class[i].append(r['val_auc_per_class'][i])

    summary = {
        'val_accuracy': {
            'mean': np.mean(val_accs),
            'std': np.std(val_accs),
            'ci95': (np.mean(val_accs) - 1.96 * np.std(val_accs) / np.sqrt(config.n_folds),
                     np.mean(val_accs) + 1.96 * np.std(val_accs) / np.sqrt(config.n_folds))
        },
        'val_f1': {
            'mean': np.mean(val_f1s),
            'std': np.std(val_f1s)
        },
        'val_weighted_f1': {
            'mean': np.mean(val_weighted_f1s),
            'std': np.std(val_weighted_f1s)
        },
        'val_auc': {
            'mean': np.mean(val_aucs),
            'std': np.std(val_aucs)
        },
        'val_auc_ovo': {
            'mean': np.mean(val_aucs_ovo),
            'std': np.std(val_aucs_ovo)
        },
        'per_class_metrics': {
            'precision': {str(i): {'mean': np.mean(v), 'std': np.std(v)} for i, v in precision_per_class.items()},
            'recall': {str(i): {'mean': np.mean(v), 'std': np.std(v)} for i, v in recall_per_class.items()},
            'specificity': {str(i): {'mean': np.mean(v), 'std': np.std(v)} for i, v in specificity_per_class.items()},
            'auc': {str(i): {'mean': np.mean(v), 'std': np.std(v)} for i, v in auc_per_class.items()},
        },
        'total_training_time': np.sum(times),
        'avg_fold_time': np.mean(times)
    }

    print("\n交叉验证结果汇总:")
    print("-" * 60)
    print(f"{'Fold':<6} {'Val Acc':<12} {'Val F1':<12} {'Val AUC':<12} {'Time (s)':<10}")
    print("-" * 60)
    for r in fold_results:
        print(f"{r['fold']:<6} {r['val_accuracy']:<12.4f} {r['val_f1']:<12.4f} "
              f"{r['val_auc']:<12.4f} {r['training_time']:<10.1f}")
    print("-" * 60)
    print(f"{'Mean':<6} {summary['val_accuracy']['mean']:<12.4f} {summary['val_f1']['mean']:<12.4f} "
          f"{summary['val_auc']['mean']:<12.4f} {summary['avg_fold_time']:<10.1f}")
    print(f"{'Std':<6} {summary['val_accuracy']['std']:<12.4f} {summary['val_f1']['std']:<12.4f} "
          f"{summary['val_auc']['std']:<12.4f}")
    print("-" * 60)

    # 打印置信区间
    ci = summary['val_accuracy']['ci95']
    print(f"95% CI for Val Accuracy: [{ci[0]:.4f}, {ci[1]:.4f}]")

    # 统计显著性检验 (Shapiro-Wilk正态性检验)
    print("\n[4/5] 统计显著性检验...")

    # 正态性检验
    shapiro_stat, shapiro_p = stats.shapiro(val_accs)
    print(f"  Shapiro-Wilk检验: stat={shapiro_stat:.4f}, p={shapiro_p:.4f}")

    # 置信区间
    t_stat, t_p = stats.ttest_1samp(val_accs, 0.5)
    print(f"  One-sample t-test (vs 0.5): t={t_stat:.4f}, p={t_p:.4f}")

    # 保存结果
    print("\n[5/5] 保存结果...")

    results_dict = {
        'fold_results': fold_results,
        'summary': summary,
        'statistical_tests': {
            'shapiro_wilk': {'statistic': shapiro_stat, 'p_value': shapiro_p},
            'ttest': {'statistic': t_stat, 'p_value': t_p}
        }
    }

    results_path = output_dir / "cross_validation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=2, default=str)
    print(f"  结果已保存: {results_path}")

    # Markdown报告
    report_path = output_dir / "cross_validation_report.md"
    with open(report_path, 'w') as f:
        f.write(f"# 交叉验证实验报告\n\n")
        f.write(f"**实验时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**K折数**: {config.n_folds}\n\n")
        f.write(f"**模型**: HybridModel (CNN-Transformer)\n\n")
        f.write(f"## 各折结果\n\n")
        f.write(f"| Fold | Val Accuracy | Val F1 | Val AUC | Training Time (s) |\n")
        f.write(f"|------|---------------|--------|---------|-------------------|\n")
        for r in fold_results:
            f.write(f"| {r['fold']} | {r['val_accuracy']:.4f} | {r['val_f1']:.4f} | "
                    f"{r['val_auc']:.4f} | {r['training_time']:.1f} |\n")

        f.write(f"\n## 汇总统计\n\n")
        f.write(f"- **Val Accuracy**: {summary['val_accuracy']['mean']:.4f} ± {summary['val_accuracy']['std']:.4f}\n")
        f.write(f"- **Val F1**: {summary['val_f1']['mean']:.4f} ± {summary['val_f1']['std']:.4f}\n")
        f.write(f"- **Val AUC**: {summary['val_auc']['mean']:.4f} ± {summary['val_auc']['std']:.4f}\n")
        f.write(f"- **95% CI**: [{ci[0]:.4f}, {ci[1]:.4f}]\n\n")

        f.write(f"## 统计检验\n\n")
        f.write(f"- **Shapiro-Wilk**: W={shapiro_stat:.4f}, p={shapiro_p:.4f}\n")
        f.write(f"- **One-sample t-test**: t={t_stat:.4f}, p={t_p:.4f}\n")

    print(f"  报告已保存: {report_path}")

    # 绘制训练曲线（各折平均）
    combined_history = {}
    if all_histories:
        # 提取所有折的训练历史（去掉fold键）
        histories = [list(h.values())[0] for h in all_histories]
        if histories:
            # 获取所有指标键
            all_keys = histories[0].keys()
            for key in all_keys:
                # 找到最短的历史长度
                min_len = min(len(h[key]) for h in histories)
                if min_len > 0:
                    combined_history[key] = [
                        sum(h[key][i] for h in histories) / len(histories)
                        for i in range(min_len)
                    ]

    if combined_history:
        fig = plot_training_curves(
            metrics_dict=combined_history,
            save_path=str(output_dir / "cross_validation_curves.png")
        )

    print("\n" + "="*60)
    print("交叉验证实验完成!")
    print("="*60)

    return fold_results, summary


# =============================================================================
# 命令行入口
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='交叉验证实验')

    # 数据集路径
    parser.add_argument('--dataset1', type=str, help='Dataset1路径')
    parser.add_argument('--dataset2', type=str, help='Dataset2路径')
    parser.add_argument('--dataset3', type=str, help='Dataset3路径')

    # 交叉验证参数
    parser.add_argument('--folds', type=int, default=5, help='K折数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=32, help='批大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')

    # 输出配置
    parser.add_argument('--output-dir', type=str, default='outputs/cross_validation', help='输出目录')
    parser.add_argument('--no-checkpoint', action='store_true', help='不保存模型权重')
    parser.add_argument('--quick-test', action='store_true', help='快速测试模式: 200样本, 3 folds, 1epoch')

    args = parser.parse_args()

    # 创建配置
    config = CrossValidationConfig()

    if args.dataset1:
        config.dataset1_path = args.dataset1
    if args.dataset2:
        config.dataset2_path = args.dataset2
    if args.dataset3:
        config.dataset3_path = args.dataset3

    config.n_folds = args.folds
    config.seed = args.seed
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.output_dir = args.output_dir
    config.save_checkpoints = not args.no_checkpoint
    config.quick_test = args.quick_test

    # 运行实验
    run_cross_validation_experiment(config)


if __name__ == '__main__':
    main()
