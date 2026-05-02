#!/usr/bin/env python3
"""
消融实验脚本 - 扩展版本
系统性评估 HybridModel 各组件的贡献度

消融维度:
1. 正向消融: 逐步添加组件，展示增量贡献
2. 反向消融: 逐步移除组件，验证组件必要性
3. 超参数消融: Transformer层数、注意力头数、模型维度
4. 门控类型消融: SE/Sigmoid/None对比

[INPUT]: 配置文件路径, 消融配置
[OUTPUT]: 消融实验结果表格, 各组件贡献度分析, 完整曲线数据
[POS]: scripts/ 消融实验执行脚本
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
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.augmentation import get_train_augmentation, get_val_augmentation
from configs import DATASET_PATHS
from models import ConfigurableHybrid
from experiments.ablation import AblationConfig, AblationStudy
from experiments.ablation.configs import (
    ABLATION_CONFIGS, ABLATION_GROUPS,
    get_ablation_config, get_configs_by_group, list_available_groups
)
from experiments.metrics import compute_metrics
from experiments.visualization import plot_model_comparison
from scripts.utils import set_seed, get_device, split_dataset_with_transforms, train_model


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

    # 快速测试模式
    quick_test: bool = False
    
    # 消融组选择
    ablation_group: str = "forward"


# =============================================================================
# 评估函数
# =============================================================================

def evaluate_model_fn(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    save_dir: str = None,
    prefix: str = ""
) -> Dict:
    """评估函数，保存完整实验数据"""

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

    # 保存预测数组供可视化CLI自动读取
    if save_dir:
        sp = Path(save_dir)
        sp.mkdir(parents=True, exist_ok=True)
        np.save(sp / f"{prefix}y_true.npy", all_labels)
        np.save(sp / f"{prefix}y_pred.npy", all_preds)
        np.save(sp / f"{prefix}y_prob.npy", all_probs)

    # 计算完整指标（包含曲线数据）
    metrics = compute_metrics(all_labels, all_preds, all_probs)
    
    # 保存曲线数据和混淆矩阵
    if save_dir:
        metrics.save_curves(save_dir, prefix=prefix)

    # 转换为字典格式
    return {
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
    print(f"消融组: {config.ablation_group}")
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

    # 快速测试模式
    if config.quick_test:
        print("\n[QUICK TEST MODE] 启用快速测试: 200样本子集, 1 epoch, batch=8, workers=0")
        from scripts.utils import create_quick_test_datasets
        train_dataset, val_dataset, test_dataset = create_quick_test_datasets(
            train_dataset, val_dataset, test_dataset, max_samples=200, seed=config.seed
        )
        config.num_epochs = 1
        config.batch_size = min(config.batch_size, 8)
        config.num_workers = 0
        config.save_checkpoints = False

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

    # 获取消融配置
    print("\n[2/4] 运行消融实验...")

    if config.quick_test:
        ablation_configs = get_configs_by_group('quick_test')
    else:
        try:
            ablation_configs = get_configs_by_group(config.ablation_group)
        except KeyError:
            print(f"[WARNING] Unknown group '{config.ablation_group}', using 'forward'")
            ablation_configs = get_configs_by_group('forward')

    print(f"  配置数量: {len(ablation_configs)}")
    for cfg in ablation_configs:
        print(f"    - {cfg.name}: {cfg.description}")

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
            num_classes=config.num_classes,
            model_dim=768,
            nhead=12,
            num_layers=12,
            dropout=cfg.dropout,
            pretrained=True
        )
        
        # 收集模型信息
        model_info = {
            'model_name': cfg.name,
            'parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'config': {
                'multi_scale': cfg.multi_scale,
                'gate': cfg.gate,
                'transformer': cfg.transformer,
                'cross_attention': cfg.cross_attention,
                'model_dim': cfg.model_dim,
                'nhead': cfg.nhead,
                'num_layers': cfg.num_layers,
                'dropout': cfg.dropout,
            }
        }

        # 训练
        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=None,
            epochs=config.num_epochs,
            device=device,
            learning_rate=config.learning_rate,
            transformer_lr=config.transformer_lr,
            weight_decay=config.weight_decay,
        )

        # 评估（保存完整数据）
        metrics = evaluate_model_fn(
            model, val_loader, device,
            save_dir=str(config.output_dir),
            prefix=f"{cfg.name}_"
        )
        
        # 计算推理时间
        inference_start = time.time()
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224).to(device)
            for _ in range(100):
                model(dummy_input)
        inference_time = (time.time() - inference_start) / 100 * 1000  # ms per sample

        elapsed = time.time() - start_time
        metrics['training_time'] = elapsed
        metrics['inference_time_ms'] = inference_time
        metrics['config'] = cfg.name
        metrics['category'] = cfg.category
        metrics['model_info'] = model_info

        results[cfg.name] = metrics

        # 保存模型信息
        model_info_path = output_dir / f"{cfg.name}_model_info.json"
        with open(model_info_path, 'w') as f:
            json.dump(model_info, f, indent=2)

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
    print("-" * 100)
    print(f"{'Config':<25} {'Category':<12} {'Accuracy':<10} {'Macro F1':<10} {'AUC-ROC':<10} {'Params':<12} {'Time':<10}")
    print("-" * 100)

    for name, res in results.items():
        params = res.get('model_info', {}).get('parameters', 0)
        params_str = f"{params/1e6:.1f}M" if params > 1e6 else f"{params/1e3:.0f}K"
        print(f"{name:<25} {res.get('category', 'N/A'):<12} {res['accuracy']:<10.4f} "
              f"{res['macro_f1']:<10.4f} {res['auc_roc']:<10.4f} {params_str:<12} "
              f"{res['training_time']:<10.1f}")

    print("-" * 100)

    # 保存结果
    print("\n[4/4] 保存结果...")

    # JSON结果（完整数据）
    results_path = output_dir / "ablation_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            'results': {k: {kk: vv for kk, vv in v.items() if kk != 'config'}
                        for k, v in results.items()},
            'summary': summary,
            'ablation_group': config.ablation_group,
            'timestamp': datetime.now().isoformat(),
        }, f, indent=2, default=str)
    print(f"  结果已保存: {results_path}")

    # Markdown表格
    table_path = output_dir / "ablation_table.md"
    with open(table_path, 'w') as f:
        f.write(f"# 消融实验结果\n\n")
        f.write(f"实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"消融组: {config.ablation_group}\n\n")
        f.write(f"| Config | Category | Accuracy | Macro F1 | AUC-ROC | Params | Time (s) |\n")
        f.write(f"|--------|----------|----------|----------|---------|--------|----------|\n")
        for name, res in results.items():
            params = res.get('model_info', {}).get('parameters', 0)
            params_str = f"{params/1e6:.1f}M" if params > 1e6 else f"{params/1e3:.0f}K"
            f.write(f"| {name} | {res.get('category', 'N/A')} | {res['accuracy']:.4f} | "
                    f"{res['macro_f1']:.4f} | {res['auc_roc']:.4f} | {params_str} | "
                    f"{res['training_time']:.1f} |\n")
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

    # 消融参数
    parser.add_argument('--group', type=str, default='forward', 
                        choices=['forward', 'reverse', 'transformer_layers', 'attention_heads', 
                                 'model_dim', 'gate_type', 'quick_test', 'all'],
                        help='消融配置组')

    # 其他参数
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--output-dir', type=str, default='outputs/ablation', help='输出目录')
    parser.add_argument('--no-checkpoint', action='store_true', help='不保存模型权重')
    parser.add_argument('--quick-test', action='store_true', help='快速测试模式: 200样本, 1epoch, 仅跑2个配置')

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
    config.quick_test = args.quick_test
    config.ablation_group = args.group

    # 运行实验
    run_ablation_experiment(config)


if __name__ == '__main__':
    main()