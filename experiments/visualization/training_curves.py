# =============================================================================
# training_curves.py - 训练曲线可视化模块
# =============================================================================
"""
[INPUT]: 依赖 matplotlib, numpy
[OUTPUT]: plot_training_curves, save_training_plot 函数
[POS]: experiments/visualization/ 训练过程可视化模块
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_training_curves(metrics_dict, save_path=None):
    """
    绘制训练曲线（Loss和Accuracy）

    [INPUT]:
        metrics_dict: dict, 包含训练指标
            - train_loss: list[float] 训练损失
            - val_loss: list[float] 验证损失
            - train_acc: list[float] 训练准确率
            - val_acc: list[float] 验证准确率
        save_path: str|Path, optional, 保存路径

    [OUTPUT]:
        fig: matplotlib.figure.Figure, 生成的图表对象

    [EXAMPLE]:
        >>> metrics = {
        ...     "train_loss": [1.0, 0.8, 0.6],
        ...     "val_loss": [1.1, 0.9, 0.7],
        ...     "train_acc": [0.6, 0.75, 0.85],
        ...     "val_acc": [0.55, 0.7, 0.8],
        ... }
        >>> fig = plot_training_curves(metrics, save_path="output.png")
    """
    # 创建2个子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # 获取epoch数
    epochs = range(1, len(metrics_dict.get("train_loss", [])) + 1)
    if len(epochs) == 0:
        epochs = range(1, len(metrics_dict.get("train_acc", [])) + 1)

    # 子图1: Loss曲线
    ax1.set_title("Loss over Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")

    if "train_loss" in metrics_dict and len(metrics_dict["train_loss"]) > 0:
        ax1.plot(epochs, metrics_dict["train_loss"], label="Train Loss", marker="o")
    if "val_loss" in metrics_dict and len(metrics_dict["val_loss"]) > 0:
        ax1.plot(epochs, metrics_dict["val_loss"], label="Val Loss", marker="s")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 子图2: Accuracy曲线
    ax2.set_title("Accuracy over Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")

    if "train_acc" in metrics_dict and len(metrics_dict["train_acc"]) > 0:
        ax2.plot(epochs, metrics_dict["train_acc"], label="Train Acc", marker="o")
    if "val_acc" in metrics_dict and len(metrics_dict["val_acc"]) > 0:
        ax2.plot(epochs, metrics_dict["val_acc"], label="Val Acc", marker="s")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # 如果提供了save_path，保存图片
    if save_path is not None:
        save_training_plot(metrics_dict, save_path)

    return fig


def save_training_plot(metrics, filename):
    """
    将训练曲线保存为图片文件

    [INPUT]:
        metrics: dict, 包含训练指标
        filename: str|Path, 保存路径

    [OUTPUT]:
        None

    [EXAMPLE]:
        >>> metrics = {"train_loss": [1.0], "val_loss": [1.1], "train_acc": [0.6], "val_acc": [0.55]}
        >>> save_training_plot(metrics, "training_curves.png")
    """
    # 先绘制曲线
    fig = plot_training_curves(metrics)

    # 保存图片
    fig.savefig(filename, dpi=150, bbox_inches="tight")

    # 关闭图表释放内存
    plt.close(fig)
