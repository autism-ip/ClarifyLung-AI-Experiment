#!/usr/bin/env python3
"""
消融实验脚本
系统性评估 HybridModel 各组件的贡献度

消融变量:
1. multi_scale: 多尺度特征融合 (layer3+layer4 vs 仅layer4)
2. gate: 门控机制 (SE/Sigmoid/None)
3. transformer: Transformer模块 (6层/4层/2层/None)
4. cross_attention: 交叉注意力 (2层/1层/0层)

[INPUT]: 配置文件路径, 消融配置
[OUTPUT]: 消融实验结果表格, 各组件贡献度分析
[POS]: scripts/ 消融实验执行脚本
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Callable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.custom_dataset import merge_datasets
from data.augmentation import get_train_augmentation, get_val_augmentation
from configs import DATASET_PATHS
from models import ConfigurableHybrid
from experiments.ablation import AblationStudy, ABLATION_CONFIGS, AblationConfig
from experiments.metrics import compute_metrics
from experiments.visualization import plot_model_comparison
from scripts.utils import set_seed, get_device


# =============================================================================
# 实验配置
# =============================================================================

class AblationExperimentConfig:
    """消融实验配置"""

    # 数据集配置
    dataset1_path: str = DATASET_PATHS['dataset1']
    dataset2_path: str = DATASET_PATHS['dataset2']
    dataset3_path: str = DATASET_PATHS['dataset3']

    # 训练配置
    batch_size: int = 32
    num_epochs: int = 30
    num_workers: int = 4

    # 优化器配置
    learning_rate: float = 1e-4
    transformer_lr: float = 5e-4
    weight_decay: float = 0.01

    # 数据划分
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # 其他
    seed: int = 42
    image_size: int = 224
    num_classes: int = 3

    # 输出配置
    output_dir: str = "outputs/ablation"
    save_checkpoints: bool = True


# =============================================================================
# 训练和评估函数
# =============================================================================

def train_model_fn(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    device: torch.device,
    learning_rate: float = 1e-4,
    transformer_lr: float = 5e-4,
    weight_decay: float = 0.01
):
    """训练函数"""

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

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

        scheduler.step()

        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")


def evaluate_model_fn(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """评估函数"""

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

    metrics = compute_metrics(all_labels, all_preds, all_probs)

    # 转换为字典格式以兼容compute_summary
    return {
        'accuracy': metrics.accuracy,
        'macro_f1': metrics.f1_macro,
        'auc_roc': metrics.auc_roc_ovr,
        'precision': metrics.precision,
        'recall': metrics.recall,
    }


# =============================================================================
# 主实验流程
# =============================================================================

def run_ablation_experiment(config: AblationExperimentConfig):
    """运行消融实验"""

    print("\n" + "="*60)
    print("消融实验")
    print("="*60)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"设备: {get_device()}")
    print(f"输出目录: {config.output_dir}")

    # 设置随机种子
    set_seed(config.seed)

    # 创建输出目录
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 准备数据
    print("\n[1/4] 加载数据集...")
    train_transform = get_train_augmentation(config.image_size)

    dataset = merge_datasets(
        config.dataset1_path,
        config.dataset2_path,
        config.dataset3_path,
        transform=train_transform
    )

    print(f"  合并数据集大小: {len(dataset)}")

    # 划分数据集
    total_size = len(dataset)
    train_size = int(config.train_ratio * total_size)
    val_size = int(config.val_ratio * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config.seed)
    )

    print(f"  训练集: {len(train_dataset)}")
    print(f"  验证集: {len(val_dataset)}")
    print(f"  测试集: {len(test_dataset)}")

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        shuffle=True, num_workers=config.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size,
        shuffle=False, num_workers=config.num_workers, pin_memory=True
    )

    device = get_device()

    # 创建消融研究
    print("\n[2/4] 运行消融实验...")

    # 定义要测试的配置
    ablation_configs = [
        AblationConfig(
            name='baseline_cnn',
            multi_scale=False,
            gate=None,
            transformer=False,
            cross_attention=False,
            description='仅CNN，无Transformer'
        ),
        AblationConfig(
            name='multiscale_only',
            multi_scale=True,
            gate=None,
            transformer=False,
            cross_attention=False,
            description='多尺度特征，无门控'
        ),
        AblationConfig(
            name='gating_added',
            multi_scale=True,
            gate='se',
            transformer=False,
            cross_attention=False,
            description='多尺度+SE门控'
        ),
        AblationConfig(
            name='transformer_added',
            multi_scale=True,
            gate='se',
            transformer=True,
            cross_attention=False,
            description='完整双流架构'
        ),
        AblationConfig(
            name='full_hybrid',
            multi_scale=True,
            gate='se',
            transformer=True,
            cross_attention=True,
            description='完整混合架构+交叉注意力'
        ),
    ]

    results = {}

    for cfg in ablation_configs:
        print(f"\n--- Config: {cfg.name} ---")
        print(f"  {cfg.description}")

        start_time = time.time()

        # 创建模型 (使用 ConfigurableHybrid，基于真实 HybridModel 组件)
        model = ConfigurableHybrid(
            multi_scale=cfg.multi_scale,
            gate=cfg.gate,
            transformer=cfg.transformer,
            cross_attention=cfg.cross_attention,
            num_classes=config.num_classes
        )

        # 训练
        train_model_fn(
            model, train_loader, config.num_epochs, device,
            config.learning_rate, config.transformer_lr, config.weight_decay
        )

        # 评估
        metrics = evaluate_model_fn(model, val_loader, device)

        elapsed = time.time() - start_time
        metrics['training_time'] = elapsed
        metrics['config'] = cfg.name

        results[cfg.name] = metrics

        print(f"  Acc: {metrics['accuracy']:.4f}, F1: {metrics['macro_f1']:.4f}, "
              f"AUC: {metrics['auc_roc']:.4f}, Time: {elapsed:.1f}s")

        # 保存checkpoint
        if config.save_checkpoints:
            torch.save(model.state_dict(), output_dir / f"{cfg.name}_best.pth")

    # 计算消融摘要
    print("\n[3/4] 计算消融摘要...")

    summary = AblationStudy.compute_summary(results)

    # 生成报告
    print("\n消融实验结果:")
    print("-" * 80)
    print(f"{'Config':<20} {'Accuracy':<10} {'Macro F1':<10} {'AUC-ROC':<10} {'ΔAcc':<10}")
    print("-" * 80)

    baseline_acc = results.get('baseline_cnn', {}).get('accuracy', 0)

    for name, res in results.items():
        delta = res['accuracy'] - baseline_acc
        delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
        print(f"{name:<20} {res['accuracy']:<10.4f} {res['macro_f1']:<10.4f} "
              f"{res['auc_roc']:<10.4f} {delta_str:<10}")

    print("-" * 80)

    # 保存结果
    print("\n[4/4] 保存结果...")

    # JSON结果
    results_path = output_dir / "ablation_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            'results': {k: {kk: vv for kk, vv in v.items() if kk != 'config'}
                        for k, v in results.items()},
            'summary': summary
        }, f, indent=2, default=str)
    print(f"  结果已保存: {results_path}")

    # Markdown表格
    table_path = output_dir / "ablation_table.md"
    with open(table_path, 'w') as f:
        f.write(f"# 消融实验结果\n\n")
        f.write(f"实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"| Config | Accuracy | Macro F1 | AUC-ROC | ΔAcc |\n")
        f.write(f"|--------|----------|----------|---------|------|\n")
        for name, res in results.items():
            delta = res['accuracy'] - baseline_acc
            f.write(f"| {name} | {res['accuracy']:.4f} | {res['macro_f1']:.4f} | "
                    f"{res['auc_roc']:.4f} | {delta:+.4f} |\n")
    print(f"  表格已保存: {table_path}")

    # 保存实验配置
    config_path = output_dir / "experiment_config.json"
    config_dict = {k: v for k, v in vars(config).items() if not k.startswith('_')}
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)
    print(f"  配置已保存: {config_path}")

    print("\n" + "="*60)
    print("消融实验完成!")
    print("="*60)

    return results


# =============================================================================
# 命令行入口
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='消融实验')

    # 数据集路径
    parser.add_argument('--dataset1', type=str, help='Dataset1路径')
    parser.add_argument('--dataset2', type=str, help='Dataset2路径')
    parser.add_argument('--dataset3', type=str, help='Dataset3路径')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=32, help='批大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='CNN学习率')
    parser.add_argument('--transformer-lr', type=float, default=5e-4, help='Transformer学习率')

    # 其他参数
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--output-dir', type=str, default='outputs/ablation', help='输出目录')
    parser.add_argument('--no-checkpoint', action='store_true', help='不保存模型权重')

    args = parser.parse_args()

    # 创建配置
    config = AblationExperimentConfig()

    if args.dataset1:
        config.dataset1_path = args.dataset1
    if args.dataset2:
        config.dataset2_path = args.dataset2
    if args.dataset3:
        config.dataset3_path = args.dataset3

    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.transformer_lr = args.transformer_lr
    config.seed = args.seed
    config.output_dir = args.output_dir
    config.save_checkpoints = not args.no_checkpoint

    # 运行实验
    run_ablation_experiment(config)


if __name__ == '__main__':
    main()
