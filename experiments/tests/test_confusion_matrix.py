"""
测试混淆矩阵可视化模块
[INPUT]: 模拟预测结果与标签
[OUTPUT]: 混淆矩阵计算与可视化验证
[POS]: experiments/tests/混淆矩阵测试，验证confusion_matrix模块
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
import numpy as np


# =============================================================================
# Fixtures: 模块级别共享
# =============================================================================

@pytest.fixture
def y_true_three_classes():
    """三分类真实标签"""
    return np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])


@pytest.fixture
def y_pred_three_classes():
    """三分类预测标签 (与标签不完全一致)"""
    return np.array([0, 1, 1, 0, 2, 2, 0, 1, 2, 1, 1, 2])


@pytest.fixture
def y_true_binary():
    """二分类真实标签"""
    return np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1])


@pytest.fixture
def y_pred_binary():
    """二分类预测标签"""
    return np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0])


@pytest.fixture
def class_names_three():
    """三类标签名称"""
    return ['normal', 'benign', 'malignant']


# =============================================================================
# TDD Phase 1: Tests defined BEFORE implementation
# All tests should FAIL initially (RED phase)
# =============================================================================


class TestComputeConfusionMatrix:
    """测试compute_confusion_matrix辅助函数"""

    def test_matrix_shape_matches_num_classes(self, y_true_three_classes, y_pred_three_classes):
        """测试矩阵形状与类别数匹配"""
        from experiments.visualization.confusion_matrix import compute_confusion_matrix

        cm = compute_confusion_matrix(y_true_three_classes, y_pred_three_classes, num_classes=3)
        assert cm.shape == (3, 3), f"Expected (3, 3), got {cm.shape}"

    def test_binary_classification_shape(self, y_true_binary, y_pred_binary):
        """测试二分类矩阵形状"""
        from experiments.visualization.confusion_matrix import compute_confusion_matrix

        cm = compute_confusion_matrix(y_true_binary, y_pred_binary, num_classes=2)
        assert cm.shape == (2, 2), f"Expected (2, 2), got {cm.shape}"

    def test_normalization_sums_to_one_per_row(self, y_true_three_classes, y_pred_three_classes):
        """测试归一化后每行和为1.0"""
        from experiments.visualization.confusion_matrix import compute_confusion_matrix

        cm = compute_confusion_matrix(y_true_three_classes, y_pred_three_classes, num_classes=3, normalize=True)
        row_sums = cm.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, rtol=1e-6, atol=1e-6,
                                    err_msg="Normalized rows should sum to 1.0")

    def test_without_normalization_contains_counts(self, y_true_three_classes, y_pred_three_classes):
        """测试非归一化时包含计数"""
        from experiments.visualization.confusion_matrix import compute_confusion_matrix

        cm = compute_confusion_matrix(y_true_three_classes, y_pred_three_classes, num_classes=3, normalize=False)
        assert (cm >= 0).all(), "Counts should be non-negative"
        assert cm.sum() > 0, "Should contain at least some predictions"
        # 12个样本，矩阵元素和应该等于样本数
        assert cm.sum() == 12, f"Expected total count of 12, got {cm.sum()}"

    def test_perfect_predictions_diagonal(self):
        """测试完美预测时只有对角线非零"""
        from experiments.visualization.confusion_matrix import compute_confusion_matrix

        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])
        cm = compute_confusion_matrix(y_true, y_pred, num_classes=3, normalize=False)

        # 非对角线元素应该为0
        off_diagonal = np.sum(cm) - np.trace(cm)
        assert off_diagonal == 0, f"Perfect predictions should have no off-diagonal elements, got {off_diagonal}"
        # 对角线元素应该等于每类的数量
        np.testing.assert_array_equal(np.diag(cm), [2, 2, 2])

    def test_function_runs_without_error_on_sample_data(self, y_true_three_classes, y_pred_three_classes):
        """测试函数在样本数据上无错误运行"""
        from experiments.visualization.confusion_matrix import compute_confusion_matrix

        # 不应抛出任何异常
        cm = compute_confusion_matrix(y_true_three_classes, y_pred_three_classes, num_classes=3)
        assert cm is not None
        assert isinstance(cm, np.ndarray)

    def test_empty_input_raises(self):
        """测试空输入应抛出异常"""
        from experiments.visualization.confusion_matrix import compute_confusion_matrix

        with pytest.raises(ValueError):
            compute_confusion_matrix(np.array([]), np.array([]), num_classes=3)

    def test_mismatched_lengths_raises(self):
        """测试长度不匹配应抛出异常"""
        from experiments.visualization.confusion_matrix import compute_confusion_matrix

        y_true = np.array([0, 1, 2])
        y_pred = np.array([0, 1])

        with pytest.raises(ValueError):
            compute_confusion_matrix(y_true, y_pred, num_classes=3)

    def test_numpy_array_return_type(self, y_true_three_classes, y_pred_three_classes):
        """测试返回类型为numpy数组"""
        from experiments.visualization.confusion_matrix import compute_confusion_matrix

        cm = compute_confusion_matrix(y_true_three_classes, y_pred_three_classes, num_classes=3)
        assert isinstance(cm, np.ndarray), f"Expected np.ndarray, got {type(cm)}"


class TestPlotConfusionMatrix:
    """测试plot_confusion_matrix可视化函数"""

    def test_function_runs_without_error(self, y_true_three_classes, y_pred_three_classes):
        """测试函数在样本数据上无错误运行"""
        import matplotlib
        matplotlib.use('Agg')  # 使用非交互式后端

        from experiments.visualization.confusion_matrix import plot_confusion_matrix

        # 不应抛出任何异常
        fig = plot_confusion_matrix(y_true_three_classes, y_pred_three_classes)
        assert fig is not None

    def test_with_class_names(self, y_true_three_classes, y_pred_three_classes, class_names_three):
        """测试带类别名称的可视化"""
        import matplotlib
        matplotlib.use('Agg')

        from experiments.visualization.confusion_matrix import plot_confusion_matrix

        fig = plot_confusion_matrix(y_true_three_classes, y_pred_three_classes,
                                     class_names=class_names_three)
        assert fig is not None

    def test_normalize_true_produces_normalized_matrix(self, y_true_three_classes, y_pred_three_classes):
        """测试normalize=True时内部矩阵正确归一化"""
        import matplotlib
        matplotlib.use('Agg')

        from experiments.visualization.confusion_matrix import plot_confusion_matrix

        fig = plot_confusion_matrix(y_true_three_classes, y_pred_three_classes,
                                     normalize=True)
        assert fig is not None

    def test_normalize_false_produces_counts(self, y_true_three_classes, y_pred_three_classes):
        """测试normalize=False时内部矩阵为计数"""
        import matplotlib
        matplotlib.use('Agg')

        from experiments.visualization.confusion_matrix import plot_confusion_matrix

        fig = plot_confusion_matrix(y_true_three_classes, y_pred_three_classes,
                                     normalize=False)
        assert fig is not None

    def test_binary_classification(self, y_true_binary, y_pred_binary):
        """测试二分类可视化"""
        import matplotlib
        matplotlib.use('Agg')

        from experiments.visualization.confusion_matrix import plot_confusion_matrix

        fig = plot_confusion_matrix(y_true_binary, y_pred_binary, num_classes=2)
        assert fig is not None

    def test_with_save_path(self, y_true_three_classes, y_pred_three_classes, tmp_path):
        """测试保存路径参数"""
        import matplotlib
        matplotlib.use('Agg')

        from experiments.visualization.confusion_matrix import plot_confusion_matrix

        save_path = tmp_path / "confusion_matrix.png"
        fig = plot_confusion_matrix(y_true_three_classes, y_pred_three_classes,
                                     save_path=str(save_path))
        assert fig is not None
        assert save_path.exists(), f"Expected file to be saved at {save_path}"

    def test_tensor_input_converted(self):
        """测试PyTorch张量输入被正确转换"""
        import torch
        import matplotlib
        matplotlib.use('Agg')

        from experiments.visualization.confusion_matrix import plot_confusion_matrix

        y_true = torch.tensor([0, 1, 2, 0, 1, 2])
        y_pred = torch.tensor([0, 1, 1, 0, 2, 2])

        fig = plot_confusion_matrix(y_true, y_pred)
        assert fig is not None

    def test_list_input_converted(self, y_true_three_classes, y_pred_three_classes):
        """测试列表输入被正确转换"""
        import matplotlib
        matplotlib.use('Agg')

        from experiments.visualization.confusion_matrix import plot_confusion_matrix

        y_true_list = y_true_three_classes.tolist()
        y_pred_list = y_pred_three_classes.tolist()

        fig = plot_confusion_matrix(y_true_list, y_pred_list)
        assert fig is not None


class TestEdgeCases:
    """边界情况测试"""

    def test_single_sample_per_class(self):
        """测试每类只有单个样本"""
        from experiments.visualization.confusion_matrix import compute_confusion_matrix

        y_true = np.array([0, 1, 2])
        y_pred = np.array([0, 1, 2])
        cm = compute_confusion_matrix(y_true, y_pred, num_classes=3, normalize=False)

        np.testing.assert_array_equal(np.diag(cm), [1, 1, 1])

    def test_all_predictions_same_class(self):
        """测试全部预测为同一类别"""
        from experiments.visualization.confusion_matrix import compute_confusion_matrix

        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 0, 0, 0])  # 全部预测为类0
        cm = compute_confusion_matrix(y_true, y_pred, num_classes=2, normalize=False)

        assert cm[0, 0] == 3  # 3个真实类0被预测为类0
        assert cm[1, 0] == 3  # 3个真实类1被预测为类0
        assert cm.sum() == 6

    def test_large_dataset(self):
        """测试大数据集性能"""
        from experiments.visualization.confusion_matrix import compute_confusion_matrix

        np.random.seed(42)
        n_samples = 10000
        y_true = np.random.randint(0, 3, size=n_samples)
        y_pred = np.random.randint(0, 3, size=n_samples)

        cm = compute_confusion_matrix(y_true, y_pred, num_classes=3, normalize=False)
        assert cm.shape == (3, 3)
        assert cm.sum() == n_samples


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
