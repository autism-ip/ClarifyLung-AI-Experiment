#!/usr/bin/env python3
"""
实验结果可视化 CLI
自动读取实验输出目录中的 JSON 报告和 .npy 预测数组，生成所有可视化图表。

用法:
    # 自动检测实验类型并生成全部可视化
    python scripts/visualize_experiment_results.py --experiment-dir outputs/benchmark

    # 强制指定实验类型
    python scripts/visualize_experiment_results.py --experiment-dir outputs/ablation --type ablation

    # 仅生成混淆矩阵（需要 .npy 预测文件）
    python scripts/visualize_experiment_results.py --experiment-dir outputs/benchmark --only confusion

输出:
    在实验目录下生成 *_revisualized.png 图表（避免覆盖实验运行时自动生成的图）
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use('Agg')

from experiments.visualization import (
    plot_training_curves,
    plot_model_comparison,
    plot_confusion_matrix,
    plot_class_distribution,
    plot_pie_chart,
)
from configs import CLASS_NAMES


# =============================================================================
# 实验类型自动检测
# =============================================================================

def detect_experiment_type(exp_dir: Path) -> str:
    """根据目录中的 JSON 文件自动检测实验类型"""
    if (exp_dir / 'benchmark_results.json').exists():
        return 'benchmark'
    if (exp_dir / 'ablation_results.json').exists():
        return 'ablation'
    if (exp_dir / 'cross_validation_results.json').exists():
        return 'crossval'
    raise ValueError(f"无法识别实验类型: {exp_dir}。未找到 benchmark/ablation/crossval 的结果 JSON。")


# =============================================================================
# 各实验类型的可视化生成
# =============================================================================

def visualize_benchmark(exp_dir: Path, only: str = None):
    """基准实验可视化: 模型对比图 + 各模型混淆矩阵"""
    results_path = exp_dir / 'benchmark_results.json'
    with open(results_path) as f:
        results = json.load(f)

    # 1. 模型对比柱状图
    if only in (None, 'comparison'):
        comparison_data = {}
        for r in results:
            comparison_data[r['model_name']] = {
                'accuracy': r['accuracy'],
                'macro_f1': r['macro_f1'],
                'auc_roc': r.get('auc_roc', 0),
            }
        fig = plot_model_comparison(
            comparison_data,
            metrics=['accuracy', 'macro_f1', 'auc_roc'],
            save_path=str(exp_dir / 'benchmark_comparison_revisualized.png')
        )
        print(f"  [OK] 模型对比图: {exp_dir / 'benchmark_comparison_revisualized.png'}")

    # 2. 各模型混淆矩阵（需要 .npy 文件）
    if only in (None, 'confusion'):
        for r in results:
            model_name = r['model_name']
            yt_path = exp_dir / f"{model_name}_y_true.npy"
            yp_path = exp_dir / f"{model_name}_y_pred.npy"
            if yt_path.exists() and yp_path.exists():
                y_true = np.load(yt_path)
                y_pred = np.load(yp_path)
                fig = plot_confusion_matrix(
                    y_true, y_pred,
                    class_names=CLASS_NAMES,
                    normalize=True,
                    save_path=str(exp_dir / f"{model_name}_confusion_revisualized.png")
                )
                print(f"  [OK] {model_name} 混淆矩阵: {exp_dir / f'{model_name}_confusion_revisualized.png'}")
            else:
                print(f"  [SKIP] {model_name}: 未找到 y_true/y_pred.npy，跳过混淆矩阵")


def visualize_ablation(exp_dir: Path, only: str = None):
    """消融实验可视化: 配置对比图 + 各配置混淆矩阵"""
    results_path = exp_dir / 'ablation_results.json'
    with open(results_path) as f:
        data = json.load(f)

    results = data.get('results', {})

    # 1. 消融对比柱状图
    if only in (None, 'comparison'):
        comparison_data = {}
        for cfg_name, metrics in results.items():
            comparison_data[cfg_name] = {
                'accuracy': metrics['accuracy'],
                'macro_f1': metrics['macro_f1'],
                'auc_roc': metrics.get('auc_roc', 0),
            }
        fig = plot_model_comparison(
            comparison_data,
            metrics=['accuracy', 'macro_f1', 'auc_roc'],
            save_path=str(exp_dir / 'ablation_comparison_revisualized.png')
        )
        print(f"  [OK] 消融对比图: {exp_dir / 'ablation_comparison_revisualized.png'}")

    # 2. 各配置混淆矩阵
    if only in (None, 'confusion'):
        for cfg_name in results.keys():
            yt_path = exp_dir / f"{cfg_name}_y_true.npy"
            yp_path = exp_dir / f"{cfg_name}_y_pred.npy"
            if yt_path.exists() and yp_path.exists():
                y_true = np.load(yt_path)
                y_pred = np.load(yp_path)
                fig = plot_confusion_matrix(
                    y_true, y_pred,
                    class_names=CLASS_NAMES,
                    normalize=True,
                    save_path=str(exp_dir / f"{cfg_name}_confusion_revisualized.png")
                )
                print(f"  [OK] {cfg_name} 混淆矩阵: {exp_dir / f'{cfg_name}_confusion_revisualized.png'}")
            else:
                print(f"  [SKIP] {cfg_name}: 未找到 y_true/y_pred.npy，跳过混淆矩阵")


def visualize_crossval(exp_dir: Path, only: str = None):
    """交叉验证可视化: 各折混淆矩阵 + 汇总统计"""
    results_path = exp_dir / 'cross_validation_results.json'
    with open(results_path) as f:
        data = json.load(f)

    fold_results = data.get('fold_results', [])

    # 1. 各折混淆矩阵
    if only in (None, 'confusion'):
        for i, fold in enumerate(fold_results, 1):
            yt_path = exp_dir / f"fold_{i}_y_true.npy"
            yp_path = exp_dir / f"fold_{i}_y_pred.npy"
            if yt_path.exists() and yp_path.exists():
                y_true = np.load(yt_path)
                y_pred = np.load(yp_path)
                fig = plot_confusion_matrix(
                    y_true, y_pred,
                    class_names=CLASS_NAMES,
                    normalize=True,
                    save_path=str(exp_dir / f"fold_{i}_confusion_revisualized.png")
                )
                print(f"  [OK] Fold {i} 混淆矩阵: {exp_dir / f'fold_{i}_confusion_revisualized.png'}")
            else:
                print(f"  [SKIP] Fold {i}: 未找到 y_true/y_pred.npy，跳过混淆矩阵")

    # 2. 汇总对比（如果有多个 fold）
    if only in (None, 'comparison') and len(fold_results) > 0:
        # 生成一个简化的 boxplot 数据
        comparison_data = {}
        for metric in ['accuracy', 'macro_f1', 'auc_roc']:
            values = [fold.get(metric, 0) for fold in fold_results]
            comparison_data[f'Fold {metric}'] = {f'Fold {i+1}': v for i, v in enumerate(values)}
        # 这里简化处理：不画 boxplot，而是打印统计
        print(f"  [INFO] {len(fold_results)} 折交叉验证完成，各折混淆矩阵已生成")


# =============================================================================
# 主入口
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='实验结果可视化 CLI')
    parser.add_argument('--experiment-dir', type=str, required=True,
                        help='实验输出目录，如 outputs/benchmark')
    parser.add_argument('--type', type=str, choices=['benchmark', 'ablation', 'crossval'],
                        help='强制指定实验类型（可选，默认自动检测）')
    parser.add_argument('--only', type=str, choices=['comparison', 'confusion'],
                        help='仅生成指定类型的图（comparison=对比图, confusion=混淆矩阵）')

    args = parser.parse_args()

    exp_dir = Path(args.experiment_dir)
    if not exp_dir.exists():
        print(f"[ERROR] 实验目录不存在: {exp_dir}")
        sys.exit(1)

    # 自动检测或强制指定实验类型
    exp_type = args.type or detect_experiment_type(exp_dir)
    print(f"=" * 60)
    print(f"实验结果可视化: {exp_type}")
    print(f"目录: {exp_dir}")
    print(f"=" * 60)

    if exp_type == 'benchmark':
        visualize_benchmark(exp_dir, only=args.only)
    elif exp_type == 'ablation':
        visualize_ablation(exp_dir, only=args.only)
    elif exp_type == 'crossval':
        visualize_crossval(exp_dir, only=args.only)

    print("\n可视化完成!")


if __name__ == '__main__':
    main()
