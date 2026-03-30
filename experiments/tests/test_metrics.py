"""
测试评估指标模块
[INPUT]: 模拟预测结果与标签
[OUTPUT]: 指标计算结果验证
[POS]: experiments/tests/指标测试，验证EvaluationMetrics类
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
import numpy as np
import torch

from experiments.metrics import EvaluationMetrics, compute_metrics


# =============================================================================
# Fixtures: 模块级别共享
# =============================================================================

@pytest.fixture
def sample_targets():
    """样本标签 (3类分类)"""
    return np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])


@pytest.fixture
def sample_predictions():
    """样本预测 (与标签不完全一致)"""
    return np.array([0, 1, 1, 0, 2, 2, 0, 1, 2, 1])


@pytest.fixture
def sample_probs():
    """样本预测概率 (用于AUC计算)"""
    probs = np.random.rand(10, 3)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs


# =============================================================================
# TDD Phase 1: Tests defined BEFORE implementation
# All tests should FAIL initially (RED phase)
# =============================================================================


class TestEvaluationMetricsDataclass:
    """测试EvaluationMetrics数据类"""

    def test_dataclass_creation(self):
        """测试dataclass创建与默认值"""
        metrics = EvaluationMetrics()
        assert hasattr(metrics, 'accuracy')
        assert hasattr(metrics, 'f1_macro')
        assert hasattr(metrics, 'f1_micro')
        assert hasattr(metrics, 'f1_weighted')
        assert hasattr(metrics, 'auc_roc_ovr')
        assert hasattr(metrics, 'auc_roc_ovo')
        assert hasattr(metrics, 'sensitivity')
        assert hasattr(metrics, 'specificity')
        assert hasattr(metrics, 'precision')
        assert hasattr(metrics, 'confusion_matrix')

    def test_dataclass_with_values(self):
        """测试带值的dataclass创建"""
        metrics = EvaluationMetrics(
            accuracy=0.9,
            f1_macro=0.85,
            f1_micro=0.9,
            f1_weighted=0.85,
            auc_roc_ovr=0.92,
            auc_roc_ovo=0.91,
            sensitivity=[0.9, 0.85, 0.88],
            specificity=[0.95, 0.92, 0.94],
            precision=[0.88, 0.86, 0.87],
            confusion_matrix=np.array([[10, 1, 0], [1, 9, 1], [0, 1, 10]])
        )
        assert metrics.accuracy == 0.9
        assert metrics.f1_macro == 0.85
        assert metrics.f1_micro == 0.9
        assert metrics.f1_weighted == 0.85
        assert metrics.auc_roc_ovr == 0.92
        assert metrics.auc_roc_ovo == 0.91
        assert metrics.sensitivity == [0.9, 0.85, 0.88]
        assert metrics.specificity == [0.95, 0.92, 0.94]
        assert metrics.precision == [0.88, 0.86, 0.87]
        np.testing.assert_array_equal(
            metrics.confusion_matrix,
            np.array([[10, 1, 0], [1, 9, 1], [0, 1, 10]])
        )

    def test_immutability(self):
        """测试dataclass不可变性 (frozen=True)"""
        metrics = EvaluationMetrics(
            accuracy=0.9,
            f1_macro=0.85,
            f1_micro=0.9,
            f1_weighted=0.85,
            auc_roc_ovr=0.92,
            auc_roc_ovo=0.91,
            sensitivity=[0.9, 0.85, 0.88],
            specificity=[0.95, 0.92, 0.94],
            precision=[0.88, 0.86, 0.87],
            confusion_matrix=np.array([[10, 1, 0], [1, 9, 1], [0, 1, 10]])
        )
        with pytest.raises(AttributeError):
            metrics.accuracy = 0.5

    def test_to_dict(self):
        """测试to_dict方法"""
        metrics = EvaluationMetrics(
            accuracy=0.9,
            f1_macro=0.85,
            f1_micro=0.9,
            f1_weighted=0.85,
            auc_roc_ovr=0.92,
            auc_roc_ovo=0.91,
            sensitivity=[0.9, 0.85, 0.88],
            specificity=[0.95, 0.92, 0.94],
            precision=[0.88, 0.86, 0.87],
            confusion_matrix=np.array([[10, 1, 0], [1, 9, 1], [0, 1, 10]])
        )
        result = metrics.to_dict()
        assert isinstance(result, dict)
        assert result['accuracy'] == 0.9
        assert result['f1_macro'] == 0.85
        assert result['sensitivity_class_0'] == 0.9


class TestComputeMetrics:
    """测试compute_metrics便捷函数"""

    def test_compute_all_metrics(self, sample_targets, sample_predictions, sample_probs):
        """测试完整指标计算"""
        metrics = compute_metrics(sample_targets, sample_predictions, sample_probs)

        assert isinstance(metrics, EvaluationMetrics)
        assert 0 <= metrics.accuracy <= 1
        assert 0 <= metrics.f1_macro <= 1
        assert 0 <= metrics.f1_micro <= 1
        assert 0 <= metrics.f1_weighted <= 1
        assert 0 <= metrics.auc_roc_ovr <= 1
        assert 0 <= metrics.auc_roc_ovo <= 1
        assert isinstance(metrics.sensitivity, list)
        assert isinstance(metrics.specificity, list)
        assert isinstance(metrics.precision, list)
        assert metrics.confusion_matrix.shape[0] == metrics.confusion_matrix.shape[1]

    def test_perfect_predictions(self):
        """测试完美预测场景"""
        targets = np.array([0, 1, 2, 0, 1, 2])
        predictions = np.array([0, 1, 2, 0, 1, 2])
        probs = np.eye(3)[targets]  # one-hot encoding

        metrics = compute_metrics(targets, predictions, probs)

        assert abs(metrics.accuracy - 1.0) < 1e-6
        assert abs(metrics.f1_macro - 1.0) < 1e-6
        assert abs(metrics.f1_micro - 1.0) < 1e-6
        assert abs(metrics.f1_weighted - 1.0) < 1e-6

    def test_sensitivity_specificity_precision_are_lists(self):
        """测试敏感度/特异度/精确度返回列表"""
        targets = np.array([0, 1, 2, 0, 1, 2])
        predictions = np.array([0, 1, 1, 0, 2, 2])
        probs = np.random.rand(6, 3)
        probs = probs / probs.sum(axis=1, keepdims=True)

        metrics = compute_metrics(targets, predictions, probs)

        assert isinstance(metrics.sensitivity, list)
        assert isinstance(metrics.specificity, list)
        assert isinstance(metrics.precision, list)
        assert len(metrics.sensitivity) == 3
        assert len(metrics.specificity) == 3
        assert len(metrics.precision) == 3

    def test_binary_classification(self):
        """测试二分类场景"""
        targets = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1])
        predictions = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0])
        probs = np.random.rand(10, 2)
        probs = probs / probs.sum(axis=1, keepdims=True)

        metrics = compute_metrics(targets, predictions, probs, num_classes=2)

        assert isinstance(metrics, EvaluationMetrics)
        assert metrics.confusion_matrix.shape == (2, 2)
        assert len(metrics.sensitivity) == 2
        assert len(metrics.specificity) == 2
        assert len(metrics.precision) == 2

    def test_empty_predictions_raises(self):
        """测试空预测输入"""
        with pytest.raises(ValueError):
            compute_metrics(np.array([]), np.array([]), np.array([]).reshape(0, 2))

    def test_mismatched_lengths_raises(self):
        """测试长度不匹配"""
        targets = np.array([0, 1, 2])
        predictions = np.array([0, 1])
        probs = np.random.rand(2, 3)

        with pytest.raises(ValueError):
            compute_metrics(targets, predictions, probs)

    def test_invalid_probs_shape_raises(self):
        """测试无效概率形状"""
        targets = np.array([0, 1, 2])
        predictions = np.array([0, 1, 2])
        probs = np.random.rand(3, 5)  # 应该是3类，这里给5类

        with pytest.raises(ValueError):
            compute_metrics(targets, predictions, probs)

    def test_torch_tensor_input(self, sample_targets, sample_predictions, sample_probs):
        """测试PyTorch张量输入"""
        targets_tensor = torch.from_numpy(sample_targets)
        predictions_tensor = torch.from_numpy(sample_predictions)
        probs_tensor = torch.from_numpy(sample_probs)

        metrics = compute_metrics(targets_tensor, predictions_tensor, probs_tensor)

        assert isinstance(metrics, EvaluationMetrics)
        assert 0 <= metrics.accuracy <= 1


class TestEdgeCases:
    """边界情况测试"""

    def test_single_sample(self):
        """测试单样本场景"""
        target = np.array([0])
        prediction = np.array([0])
        prob = np.array([[1.0, 0.0, 0.0]])

        metrics = compute_metrics(target, prediction, prob)
        assert metrics.accuracy == 1.0
        assert metrics.f1_macro == 1.0

    def test_all_same_class(self):
        """测试全部同一类别"""
        targets = np.array([0, 0, 0, 0, 0])
        predictions = np.array([0, 0, 0, 0, 1])  # 一个错误
        probs = np.random.rand(5, 3)
        probs = probs / probs.sum(axis=1, keepdims=True)

        metrics = compute_metrics(targets, predictions, probs)
        assert metrics.confusion_matrix.shape == (3, 3)

    def test_confusion_matrix_structure(self, sample_targets, sample_predictions, sample_probs):
        """测试混淆矩阵结构正确性"""
        metrics = compute_metrics(sample_targets, sample_predictions, sample_probs)

        cm = metrics.confusion_matrix
        assert cm.ndim == 2
        assert cm.shape[0] == cm.shape[1]
        assert (cm >= 0).all()
        assert (cm.sum(axis=1) > 0).all()  # 每行都有预测


class TestMetricsValuesRange:
    """测试指标值范围"""

    def test_all_metrics_in_valid_range(self, sample_targets, sample_predictions, sample_probs):
        """所有指标应在[0,1]范围内"""
        metrics = compute_metrics(sample_targets, sample_predictions, sample_probs)

        assert 0 <= metrics.accuracy <= 1
        assert 0 <= metrics.f1_macro <= 1
        assert 0 <= metrics.f1_micro <= 1
        assert 0 <= metrics.f1_weighted <= 1
        assert 0 <= metrics.auc_roc_ovr <= 1
        assert 0 <= metrics.auc_roc_ovo <= 1
        assert all(0 <= s <= 1 for s in metrics.sensitivity)
        assert all(0 <= s <= 1 for s in metrics.specificity)
        assert all(0 <= p <= 1 for p in metrics.precision)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
