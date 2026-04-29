#!/usr/bin/env python3
"""
基准模型对比实验脚本
对比 ResNet50, ViT-B/16, HybridModel 在全量数据集上的性能

[INPUT]: 配置文件路径, 超参数
[OUTPUT]: 模型checkpoints, 训练日志, 结果对比表格
[POS]: scripts/ 基准实验执行脚本
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.augmentation import get_train_augmentation, get_val_augmentation
from configs import DATASET_PATHS
from models import HybridModel
from experiments.benchmark import create_resnet50, create_vit, create_hybrid_basic
from experiments.benchmark import generate_comparison_table, save_results
from experiments.visualization import plot_model_comparison, plot_training_curves
from experiments.metrics import compute_metrics
from scripts.utils import set_seed, get_device, split_dataset_with_transforms, train_model


# =============================================================================
# 实验配置
# =============================================================================

class BenchmarkExperimentConfig:
    """基准实验配置"""

    # 数据集配置
    dataset1_path: str = DATASET_PATHS['dataset1']
    dataset2_path: str = DATASET_PATHS['dataset2']
    dataset3_path: str = DATASET_PATHS['dataset3']

    # 训练配置
    batch_size: int = 32
    num_epochs: int = 50
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
    output_dir: str = "outputs/benchmark"
    save_checkpoints: bool = True
    save_plots: bool = True


# =============================================================================
# 模型工厂
# =============================================================================

def create_model(model_name: str, num_classes: int, pretrained: bool = True):
    """创建基准模型"""
    if model_name == 'resnet50':
        return create_resnet50(num_classes=num_classes)
    elif model_name == 'vit':
        return create_vit(num_classes=num_classes)
    elif model_name == 'hybrid_basic':
        return create_hybrid_basic(num_classes=num_classes)
    elif model_name == 'hybrid_advanced':
        # 使用 models/HybridModel 作为完整混合架构
        return HybridModel(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")


# =============================================================================
# 训练和评估函数
# =============================================================================

def train_single_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: BenchmarkExperimentConfig,
    device: torch.device,
    model_name: str
) -> dict:
    """训练单个模型并返回结果"""

    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"{'='*60}")

    save_path = None
    if config.save_checkpoints:
        save_path = str(Path(config.output_dir) / f"{model_name}_best.pth")

    return train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.num_epochs,
        device=device,
        learning_rate=config.learning_rate,
        transformer_lr=config.transformer_lr,
        weight_decay=config.weight_decay,
        save_path=save_path,
    )


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device
) -> dict:
    """在测试集上评估模型"""

    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
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

    # 计算各项指标
    metrics = compute_metrics(all_labels, all_preds, all_probs)

    return metrics


# =============================================================================
# 主实验流程
# =============================================================================

def run_benchmark_experiment(config: BenchmarkExperimentConfig):
    """运行基准对比实验"""

    print("\n" + "="*60)
    print("基准模型对比实验")
    print("="*60)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"设备: {get_device()}")
    print(f"数据集: dataset1={config.dataset1_path}")
    print(f"         dataset2={config.dataset2_path}")
    print(f"         dataset3={config.dataset3_path}")
    print(f"训练配置: epochs={config.num_epochs}, batch_size={config.batch_size}")
    print(f"输出目录: {config.output_dir}")

    # 设置随机种子
    set_seed(config.seed)

    # 创建输出目录
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 准备数据
    print("\n[1/4] 加载数据集...")
    train_transform = get_train_augmentation(config.image_size)
    val_transform = get_val_augmentation(config.image_size)

    train_dataset, val_dataset, test_dataset = split_dataset_with_transforms(
        config.dataset1_path,
        config.dataset2_path,
        config.dataset3_path,
        train_transform=train_transform,
        val_transform=val_transform,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        seed=config.seed,
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
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size,
        shuffle=False, num_workers=config.num_workers, pin_memory=True
    )

    device = get_device()

    # 定义要对比的模型
    models_to_compare = [
        ('resnet50', 'ResNet50', True),
        ('vit', 'ViT-B/16', True),
        ('hybrid_basic', 'Hybrid-Basic', False),
        ('hybrid_advanced', 'Hybrid-Advanced', False),
    ]

    results = []
    all_histories = {}

    # 训练每个模型
    print("\n[2/4] 训练基准模型...")

    for model_key, model_name, use_pretrained in models_to_compare:
        print(f"\n--- Training {model_name} ---")

        model = create_model(model_key, config.num_classes, use_pretrained)

        start_time = time.time()
        result = train_single_model(
            model, train_loader, val_loader, config, device, model_name
        )
        training_time = time.time() - start_time

        # 加载最佳权重并在测试集上评估
        if config.save_checkpoints:
            checkpoint_path = Path(config.output_dir) / f"{model_name}_best.pth"
            if checkpoint_path.exists():
                model.load_state_dict(torch.load(checkpoint_path, map_location=device))

        test_metrics = evaluate_model(model, test_loader, device)

        # 保存结果
        model_result = {
            'model_name': model_name,
            'accuracy': test_metrics.accuracy,
            'macro_f1': test_metrics.f1_macro,
            'auc_roc': test_metrics.auc_roc_ovr,
            'precision': float(np.mean(test_metrics.precision)) if test_metrics.precision else None,
            'recall': float(np.mean(test_metrics.sensitivity)) if test_metrics.sensitivity else None,
            'training_time': training_time,
        }
        results.append(model_result)
        all_histories[model_name] = result['train_history']

        print(f"  {model_name} Test Acc: {test_metrics.accuracy:.4f}, "
              f"F1: {test_metrics.f1_macro:.4f}, AUC: {test_metrics.auc_roc_ovr:.4f}")

    # 保存结果
    print("\n[3/4] 保存结果...")

    # 保存JSON结果
    results_path = output_dir / "benchmark_results.json"
    save_results(results, str(results_path))
    print(f"  结果已保存: {results_path}")

    # 生成对比表格
    table = generate_comparison_table(results)
    print("\n基准模型对比结果:")
    print(table)

    # 保存对比表格
    table_path = output_dir / "benchmark_table.md"
    with open(table_path, 'w') as f:
        f.write(f"# 基准模型对比实验结果\n\n")
        f.write(f"实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(table)
    print(f"  表格已保存: {table_path}")

    # 绘制对比图
    if config.save_plots:
        print("\n[4/4] 生成可视化图表...")

        # 模型对比柱状图
        fig = plot_model_comparison(
            benchmark_results=results,
            metrics=['accuracy', 'macro_f1', 'auc_roc'],
            save_path=str(output_dir / "model_comparison.png")
        )

        # 训练曲线
        fig = plot_training_curves(
            metrics_dict=all_histories,
            save_path=str(output_dir / "training_curves.png")
        )

        print(f"  图表已保存到: {output_dir}")

    # 保存实验配置
    config_path = output_dir / "experiment_config.json"
    config_dict = {k: v for k, v in vars(config).items() if not k.startswith('_')}
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)
    print(f"  配置已保存: {config_path}")

    print("\n" + "="*60)
    print("基准实验完成!")
    print("="*60)

    return results


# =============================================================================
# 命令行入口
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='基准模型对比实验')

    # 数据集路径
    parser.add_argument('--dataset1', type=str, help='Dataset1路径')
    parser.add_argument('--dataset2', type=str, help='Dataset2路径')
    parser.add_argument('--dataset3', type=str, help='Dataset3路径')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=32, help='批大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='CNN学习率')
    parser.add_argument('--transformer-lr', type=float, default=5e-4, help='Transformer学习率')

    # 其他参数
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--output-dir', type=str, default='outputs/benchmark', help='输出目录')
    parser.add_argument('--no-checkpoint', action='store_true', help='不保存模型权重')
    parser.add_argument('--no-plot', action='store_true', help='不生成图表')

    args = parser.parse_args()

    # 创建配置
    config = BenchmarkExperimentConfig()

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
    config.save_plots = not args.no_plot

    # 运行实验
    run_benchmark_experiment(config)


if __name__ == '__main__':
    main()
