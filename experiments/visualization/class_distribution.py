# =============================================================================
# TDD Phase 2: GREEN - 通过测试的最简实现
# =============================================================================
"""
[INPUT]: matplotlib, numpy
[OUTPUT]: plot_class_distribution, plot_pie_chart
[POS]: experiments/visualization/ 类别分布可视化
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_class_distribution(class_counts_dict, class_names=None, save_path=None):
    """
    Plot bar charts showing class distribution per dataset.

    Args:
        class_counts_dict: Dictionary mapping dataset names to class counts
                          e.g., {"dataset1": {0: 1208, 1: 1200, 2: 1201}, ...}
        class_names: List of class names (e.g., ["normal", "benign", "malignant"])
                    If None, uses ["Class 0", "Class 1", ...]
        save_path: Optional path to save the figure

    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    if class_names is None:
        # Get number of classes from first dataset
        first_dataset = next(iter(class_counts_dict.values()))
        class_names = [f"Class {i}" for i in range(len(first_dataset))]

    datasets = list(class_counts_dict.keys())
    num_datasets = len(datasets)
    num_classes = len(class_names)

    # Set up the figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Bar width and positions
    bar_width = 0.8 / num_classes
    x = np.arange(num_datasets)

    # Plot bars for each class
    colors = ['#2ecc71', '#3498db', '#e74c3c']  # green, blue, red
    for i, class_name in enumerate(class_names):
        counts = [class_counts_dict[d].get(i, 0) for d in datasets]
        offset = (i - num_classes / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, counts, bar_width * 0.9, label=class_name, color=colors[i % len(colors)])

    # Configure axes
    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title('Class Distribution Across Datasets', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=15, ha='right')
    ax.legend(title='Class', loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_pie_chart(dist, labels, title, save_path=None):
    """
    Create a pie chart showing class distribution.

    Args:
        dist: List of counts for each class
        labels: List of class names
        title: Chart title
        save_path: Optional path to save the figure

    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    colors = ['#2ecc71', '#3498db', '#e74c3c']  # green, blue, red
    explode = [0.02] * len(dist)  # Slight separation between slices

    wedges, texts, autotexts = ax.pie(
        dist,
        labels=labels,
        autopct='%1.1f%%',
        colors=colors[:len(dist)],
        explode=explode,
        startangle=90,
        textprops={'fontsize': 11}
    )

    # Style the percentage text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)

    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
