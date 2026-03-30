# =============================================================================
# confusion_matrix Module
# =============================================================================
"""
[INPUT]: 依赖 numpy, matplotlib, seaborn
[OUTPUT]: compute_confusion_matrix, plot_confusion_matrix
[POS]: experiments/visualization/ 混淆矩阵可视化模块
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix as sk_confusion_matrix


def compute_confusion_matrix(y_true, y_pred, num_classes, normalize=False):
    """
    计算混淆矩阵

    [INPUT]:
        y_true: 真实标签 (list/array/tensor)
        y_pred: 预测标签 (list/array/tensor)
        num_classes: 类别数量
        normalize: 是否归一化 (按行归一化)

    [OUTPUT]: numpy.ndarray 混淆矩阵

    [POS]: confusion_matrix模块核心计算函数
    """
    # 转换输入为numpy数组
    if hasattr(y_true, 'numpy'):
        y_true = y_true.numpy()
    if hasattr(y_pred, 'numpy'):
        y_pred = y_pred.numpy()

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # 验证输入
    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("Input arrays cannot be empty")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    # 计算混淆矩阵
    cm = sk_confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    # 归一化处理
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        # 避免除以零
        row_sums = np.where(row_sums == 0, 1, row_sums)
        cm = cm.astype(np.float64) / row_sums

    return cm


def plot_confusion_matrix(y_true, y_pred, class_names=None, normalize=True, save_path=None, num_classes=None):
    """
    绘制混淆矩阵热力图

    [INPUT]:
        y_true: 真实标签 (list/array/tensor)
        y_pred: 预测标签 (list/array/tensor)
        class_names: 类别名称列表 (optional)
        normalize: 是否归一化 (默认True)
        save_path: 保存路径 (optional)
        num_classes: 类别数量 (optional, 默认从数据推断)

    [OUTPUT]: matplotlib.figure.Figure

    [POS]: confusion_matrix模块可视化函数
    """
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端

    # 转换输入为numpy数组
    if hasattr(y_true, 'numpy'):
        y_true = y_true.numpy()
    if hasattr(y_pred, 'numpy'):
        y_pred = y_pred.numpy()

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # 推断类别数量
    if num_classes is None:
        num_classes = max(y_true.max(), y_pred.max()) + 1

    # 计算混淆矩阵
    cm = compute_confusion_matrix(y_true, y_pred, num_classes=num_classes, normalize=normalize)

    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 8))

    # 设置标签
    labels = class_names if class_names is not None else [str(i) for i in range(num_classes)]

    # 格式化显示
    fmt = '.2f' if normalize else 'd'

    # 绘制热力图
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        cbar_kws={'label': 'Normalized' if normalize else 'Count'},
        linewidths=0.5,
        linecolor='white'
    )

    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix' + (' (Normalized)' if normalize else ''), fontsize=14)

    plt.tight_layout()

    # 保存图片
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
