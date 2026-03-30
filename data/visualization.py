"""
数据可视化模块
[INPUT]: 数据集、标签列表、统计信息
[OUTPUT]: 可视化图表(分布图/样本网格/统计表格)
[POS]: data/可视化工具，支持数据集探索性分析
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import os
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image


# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 设置默认样式
sns.set_style("whitegrid")
plt.style.use("seaborn-v0_8-whitegrid")


def plot_class_distribution(
    labels: Union[List[int], np.ndarray, Dict[int, int]],
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None,
    show_values: bool = True,
) -> plt.Figure:
    """
    绘制类别分布图(柱状图+饼图)

    Args:
        labels: 标签列表或类别计数字典
        class_names: 类别名称列表
        figsize: 图像尺寸
        save_path: 保存路径
        show_values: 是否在柱状图上显示数值

    Returns:
        matplotlib Figure对象
    """
    # 处理输入
    if isinstance(labels, dict):
        class_counts = labels
        # 确保所有类别都存在
        if class_names:
            for i in range(len(class_names)):
                if i not in class_counts:
                    class_counts[i] = 0
    else:
        labels = np.array(labels)
        unique, counts = np.unique(labels, return_counts=True)
        class_counts = dict(zip(unique, counts))

    # 设置类别名称
    if class_names is None:
        class_names = [f"Class {i}" for i in sorted(class_counts.keys())]

    # 确保名称与数量匹配
    sorted_classes = sorted(class_counts.keys())
    counts = [class_counts[c] for c in sorted_classes]

    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # 柱状图
    ax1 = axes[0]
    bars = ax1.bar(class_names, counts, color=['#3498db', '#e74c3c', '#2ecc71'], alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax1.set_title('Class Distribution (Bar Chart)', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # 在柱状图上显示数值
    if show_values:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 饼图
    ax2 = axes[1]
    total = sum(counts)
    percentages = [c/total*100 for c in counts]
    colors = ['#3498db', '#e74c3c', '#2ecc71']

    wedges, texts, autotexts = ax2.pie(counts, labels=class_names, autopct='%1.1f%%',
                                        colors=colors, startangle=90,
                                        explode=[0.02]*len(class_names),
                                        shadow=True)
    ax2.set_title('Class Distribution (Pie Chart)', fontsize=14, fontweight='bold')

    # 美化饼图文字
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(11)

    plt.tight_layout()

    # 保存
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved class distribution plot to: {save_path}")

    return fig


def plot_sample_grid(
    dataset,
    num_samples: int = 16,
    ncols: int = 4,
    figsize: Tuple[int, int] = (12, 12),
    save_path: Optional[str] = None,
    class_names: Optional[List[str]] = None,
) -> plt.Figure:
    """
    绘制样本网格图

    Args:
        dataset: PyTorch Dataset实例
        num_samples: 样本数量
        ncols: 每行显示数量
        figsize: 图像尺寸
        save_path: 保存路径
        class_names: 类别名称列表

    Returns:
        matplotlib Figure对象
    """
    if class_names is None:
        class_names = ["normal", "malignant", "benign"]

    nrows = (num_samples + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if nrows > 1 else [axes] if ncols == 1 else axes.flatten()

    # 随机选择样本索引
    indices = torch.randperm(len(dataset))[:num_samples]

    for idx, ax in enumerate(axes):
        if idx < num_samples:
            # 获取样本
            image, label = dataset[indices[idx].item()]

            # 转换tensor为numpy显示
            if isinstance(image, torch.Tensor):
                # 反归一化 (假设使用ImageNet统计)
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                image = image * std + mean
                image = image.clamp(0, 1)
                image = image.permute(1, 2, 0).numpy()

            ax.imshow(image)
            ax.set_title(f"{class_names[label]}", fontsize=10, fontweight='bold',
                        color='green' if label == 0 else 'red' if label == 1 else 'orange')
            ax.axis('off')
        else:
            ax.axis('off')

    plt.suptitle(f"Sample Images (n={num_samples})", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved sample grid to: {save_path}")

    return fig


def create_summary_table(
    train_labels: List[int],
    val_labels: List[int],
    test_labels: Optional[List[int]] = None,
    class_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    创建数据集统计摘要表

    Args:
        train_labels: 训练集标签
        val_labels: 验证集标签
        test_labels: 测试集标签(可选)
        class_names: 类别名称

    Returns:
        统计摘要DataFrame
    """
    if class_names is None:
        class_names = ["normal", "malignant", "benign"]

    # 统计各集合
    data = {
        'Class': class_names,
        'Train Count': [sum(1 for l in train_labels if l == i) for i in range(len(class_names))],
        'Val Count': [sum(1 for l in val_labels if l == i) for i in range(len(class_names))],
    }

    if test_labels:
        data['Test Count'] = [sum(1 for l in test_labels if l == i) for i in range(len(class_names))]
        data['Total'] = [data['Train Count'][i] + data['Val Count'][i] + data['Test Count'][i]
                        for i in range(len(class_names))]
    else:
        data['Total'] = [data['Train Count'][i] + data['Val Count'][i]
                        for i in range(len(class_names))]

    # 计算百分比
    total_samples = sum(data['Total'])
    data['Percentage'] = [f"{count/total_samples*100:.2f}%" for count in data['Total']]

    df = pd.DataFrame(data)

    # 添加总计行
    total_row = {'Class': 'TOTAL'}
    for col in df.columns[1:]:
        if col == 'Percentage':
            total_row[col] = '100.00%'
        elif col in ['Train Count', 'Val Count', 'Test Count', 'Total']:
            total_row[col] = df[col].sum()
        else:
            total_row[col] = '-'

    df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)

    return df


if __name__ == "__main__":
    # 测试代码
    print("Testing XRayDataset...")

    # 创建模拟数据
    import tempfile
    import shutil

    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建模拟图像
        for split in ['train', 'val']:
            for cls_idx, cls_name in enumerate(['normal', 'malignant', 'benign']):
                class_dir = Path(tmpdir) / split / cls_name
                class_dir.mkdir(parents=True, exist_ok=True)

                # 每个类别创建5张图
                for i in range(5):
                    img = Image.new('RGB', (224, 224), color=(cls_idx*80, i*50, 100))
                    img.save(class_dir / f"image_{i}.jpg")

        # 测试数据集
        train_dir = Path(tmpdir) / 'train'
        val_dir = Path(tmpdir) / 'val'

        train_dataset = XRayDataset(str(train_dir), mode='folder')
        val_dataset = XRayDataset(str(val_dir), mode='folder')

        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Val dataset size: {len(val_dataset)}")
        print(f"Train class distribution: {train_dataset.get_class_distribution()}")

        # 测试统计表
        summary = create_summary_table(
            [l for _, l in train_dataset],
            [l for _, l in val_dataset],
            class_names=XRayDataset.CLASS_NAMES
        )
        print("\nSummary Table:")
        print(summary.to_string(index=False))

        # 测试可视化
        fig = plot_class_distribution(
            train_dataset.labels,
            class_names=XRayDataset.CLASS_NAMES,
            save_path=f"{tmpdir}/class_dist.png"
        )
        plt.close(fig)
        print(f"\nVisualization saved to {tmpdir}/class_dist.png")

    print("\nAll tests passed!")
