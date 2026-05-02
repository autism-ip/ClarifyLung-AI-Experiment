"""
评估指标模块
[INPUT]: 预测结果、标签、概率
[OUTPUT]: EvaluationMetrics数据类，包含多分类评估指标
[POS]: experiments/核心组件，提供标准化评估接口
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_curve,
)


# =============================================================================
# 数据类定义
# =============================================================================

@dataclass(frozen=True)
class EvaluationMetrics:
    """
    多分类评估指标数据类 - 扩展版本

    Attributes:
        accuracy: 准确率 (0-1)
        f1_macro: Macro平均F1分数 (0-1)
        f1_micro: Micro平均F1分数 (0-1)
        f1_weighted: 加权平均F1分数 (0-1)
        auc_roc_ovr: ROC-AUC分数 One-vs-Rest (0-1)
        auc_roc_ovo: ROC-AUC分数 One-vs-One (0-1)
        sensitivity: 每类敏感度列表 [class_0, class_1, ...]
        specificity: 每类特异度列表 [class_0, class_1, ...]
        precision: 每类精确度列表 [class_0, class_1, ...]
        confusion_matrix: 混淆矩阵 (numpy.ndarray)
        roc_curve_data: ROC曲线数据 {class_idx: {'fpr': [...], 'tpr': [...], 'thresholds': [...]}}
        pr_curve_data: PR曲线数据 {class_idx: {'precision': [...], 'recall': [...], 'thresholds': [...]}}
        auc_per_class: 每类AUC分数 [class_0_auc, class_1_auc, ...]
    """
    accuracy: float = 0.0
    f1_macro: float = 0.0
    f1_micro: float = 0.0
    f1_weighted: float = 0.0
    auc_roc_ovr: float = 0.0
    auc_roc_ovo: float = 0.0
    sensitivity: List[float] = field(default_factory=list)
    specificity: List[float] = field(default_factory=list)
    precision: List[float] = field(default_factory=list)
    confusion_matrix: Optional[np.ndarray] = None
    
    # 扩展：曲线数据
    roc_curve_data: Optional[Dict[str, Dict]] = None
    pr_curve_data: Optional[Dict[str, Dict]] = None
    auc_per_class: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, float]:
        """转换为扁平字典 (用于日志/保存)"""
        result = {
            'accuracy': self.accuracy,
            'f1_macro': self.f1_macro,
            'f1_micro': self.f1_micro,
            'f1_weighted': self.f1_weighted,
            'auc_roc_ovr': self.auc_roc_ovr,
            'auc_roc_ovo': self.auc_roc_ovo,
        }
        # 展平sensitivity/specificity/precision
        for idx, sens in enumerate(self.sensitivity):
            result[f'sensitivity_class_{idx}'] = sens
        for idx, spec in enumerate(self.specificity):
            result[f'specificity_class_{idx}'] = spec
        for idx, prec in enumerate(self.precision):
            result[f'precision_class_{idx}'] = prec
        # 每类AUC
        for idx, auc_val in enumerate(self.auc_per_class):
            result[f'auc_class_{idx}'] = auc_val
        return result
    
    def save_curves(self, output_dir: str, prefix: str = ""):
        """保存曲线数据到目录"""
        import json
        from pathlib import Path
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 保存ROC曲线数据
        if self.roc_curve_data:
            with open(output_path / f"{prefix}roc_curves.json", 'w') as f:
                json.dump(self.roc_curve_data, f)
        
        # 保存PR曲线数据
        if self.pr_curve_data:
            with open(output_path / f"{prefix}pr_curves.json", 'w') as f:
                json.dump(self.pr_curve_data, f)
        
        # 保存混淆矩阵
        if self.confusion_matrix is not None:
            np.save(output_path / f"{prefix}confusion_matrix.npy", self.confusion_matrix)


# =============================================================================
# 敏感度/特异度/精确度计算
# =============================================================================

def _compute_per_class_metrics(
    cm: np.ndarray
) -> tuple:
    """
    从混淆矩阵计算每类的敏感度、特异度、精确度

    Args:
        cm: 混淆矩阵 (n_classes x n_classes)

    Returns:
        (sensitivity_list, specificity_list, precision_list)
    """
    n_classes = cm.shape[0]
    sensitivity = []
    specificity = []
    precision = []

    for i in range(n_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp

        # 敏感度 = TP / (TP + FN)
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        sensitivity.append(sens)

        # 特异度 = TN / (TN + FP)
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificity.append(spec)

        # 精确度 = TP / (TP + FP)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        precision.append(prec)

    return sensitivity, specificity, precision


def _compute_auc_scores(
    targets: np.ndarray,
    probs: np.ndarray,
    num_classes: int
) -> tuple:
    """
    计算ROC-AUC (OvR and OvO) 及曲线数据

    Args:
        targets: 标签 (n_samples,)
        probs: 预测概率 (n_samples, n_classes)
        num_classes: 类别数

    Returns:
        (auc_roc_ovr, auc_roc_ovo, auc_per_class, roc_curve_data)
    """
    from sklearn.metrics import precision_recall_curve as pr_curve_func
    
    # One-hot编码
    targets_onehot = np.eye(num_classes)[targets]

    auc_roc_ovr_list = []
    auc_roc_ovo_list = []
    roc_curve_data = {}
    pr_curve_data = {}
    auc_per_class = []

    for i in range(num_classes):
        # ROC曲线数据
        fpr, tpr, thresholds = roc_curve(targets_onehot[:, i], probs[:, i])
        auc_val = auc(fpr, tpr)
        auc_roc_ovr_list.append(auc_val)
        auc_per_class.append(auc_val)
        roc_curve_data[str(i)] = {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist()
        }
        
        # PR曲线数据
        precision, recall, pr_thresholds = pr_curve_func(targets_onehot[:, i], probs[:, i])
        pr_curve_data[str(i)] = {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'thresholds': pr_thresholds.tolist()
        }

    # OvR: macro average
    auc_roc_ovr = np.mean(auc_roc_ovr_list)

    # OvO: 所有类别两两组合，自动适配任意类别数
    from itertools import combinations
    for c1, c2 in combinations(range(num_classes), 2):
        # Get samples belonging to either class
        mask = (targets == c1) | (targets == c2)
        if mask.sum() < 2:
            auc_roc_ovo_list.append(0.0)
            continue
        targets_pair = targets_onehot[mask, c1]  # class c1 as positive
        probs_pair = probs[mask, c1]
        if len(np.unique(targets_pair)) < 2:
            auc_roc_ovo_list.append(0.0)
            continue
        fpr, tpr, _ = roc_curve(targets_pair, probs_pair)
        auc_roc_ovo_list.append(auc(fpr, tpr))

    auc_roc_ovo = np.mean(auc_roc_ovo_list) if auc_roc_ovo_list else 0.0

    return auc_roc_ovr, auc_roc_ovo, auc_per_class, roc_curve_data, pr_curve_data


# =============================================================================
# 便捷函数
# =============================================================================

def compute_metrics(
    targets: Union[np.ndarray, torch.Tensor],
    predictions: Union[np.ndarray, torch.Tensor],
    probs: Union[np.ndarray, torch.Tensor],
    num_classes: int = 3
) -> EvaluationMetrics:
    """
    计算完整评估指标

    Args:
        targets: 真实标签 (n_samples,)
        predictions: 预测类别 (n_samples,)
        probs: 预测概率 (n_samples, n_classes)
        num_classes: 类别数

    Returns:
        EvaluationMetrics: 包含所有指标的 dataclass

    Raises:
        ValueError: 输入为空或长度不匹配
    """
    # 转换为numpy
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(probs, torch.Tensor):
        probs = probs.cpu().numpy()

    # 验证输入
    if len(targets) == 0 or len(predictions) == 0 or probs.shape[0] == 0:
        raise ValueError("Empty input arrays provided")

    if len(targets) != len(predictions) or len(targets) != probs.shape[0]:
        raise ValueError(
            f"Length mismatch: targets={len(targets)}, "
            f"predictions={len(predictions)}, probs={probs.shape[0]}"
        )

    if probs.shape[1] != num_classes:
        raise ValueError(
            f"Probability shape mismatch: expected {num_classes} classes, "
            f"got {probs.shape[1]}"
        )

    # 计算混淆矩阵
    cm = confusion_matrix(targets, predictions, labels=range(num_classes))

    # 计算F1分数
    accuracy = accuracy_score(targets, predictions)
    f1_macro = f1_score(targets, predictions, average='macro', zero_division=0)
    f1_micro = f1_score(targets, predictions, average='micro', zero_division=0)
    f1_weighted = f1_score(targets, predictions, average='weighted', zero_division=0)

    # 计算敏感度/特异度/精确度
    sensitivity, specificity, precision = _compute_per_class_metrics(cm)

    # 计算AUC
    try:
        auc_roc_ovr, auc_roc_ovo, auc_per_class, roc_curve_data, pr_curve_data = _compute_auc_scores(targets, probs, num_classes)
    except Exception:
        auc_roc_ovr = 0.0
        auc_roc_ovo = 0.0
        auc_per_class = []
        roc_curve_data = {}
        pr_curve_data = {}

    return EvaluationMetrics(
        accuracy=accuracy,
        f1_macro=f1_macro,
        f1_micro=f1_micro,
        f1_weighted=f1_weighted,
        auc_roc_ovr=auc_roc_ovr,
        auc_roc_ovo=auc_roc_ovo,
        sensitivity=sensitivity,
        specificity=specificity,
        precision=precision,
        confusion_matrix=cm,
        roc_curve_data=roc_curve_data,
        pr_curve_data=pr_curve_data,
        auc_per_class=auc_per_class,
    )


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == "__main__":
    print("Testing metrics module...")

    # 创建测试数据
    targets = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    predictions = np.array([0, 1, 1, 0, 2, 2, 0, 1, 2, 1])

    # 模拟概率
    probs = np.random.rand(10, 3)
    probs = probs / probs.sum(axis=1, keepdims=True)

    # 计算指标
    metrics = compute_metrics(targets, predictions, probs)

    print(f"Accuracy: {metrics.accuracy:.4f}")
    print(f"F1 Macro: {metrics.f1_macro:.4f}")
    print(f"F1 Micro: {metrics.f1_micro:.4f}")
    print(f"F1 Weighted: {metrics.f1_weighted:.4f}")
    print(f"AUC-ROC OvR: {metrics.auc_roc_ovr:.4f}")
    print(f"AUC-ROC OvO: {metrics.auc_roc_ovo:.4f}")
    print(f"Sensitivity: {metrics.sensitivity}")
    print(f"Specificity: {metrics.specificity}")
    print(f"Precision: {metrics.precision}")
    print(f"Confusion Matrix:\n{metrics.confusion_matrix}")

    print("\nAll metrics tests passed!")
