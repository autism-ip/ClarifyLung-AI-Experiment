# =============================================================================
# test_training_curves.py - 训练曲线可视化测试
# =============================================================================
"""
[INPUT]: 依赖 experiments/visualization/training_curves 模块
[OUTPUT]: 测试 plot_training_curves 和 save_training_plot 函数
[POS]: experiments/tests/ TDD测试套件
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import os

# 尝试导入被测试的模块
try:
    from experiments.visualization.training_curves import (
        plot_training_curves,
        save_training_plot,
    )
except ImportError:
    plot_training_curves = None
    save_training_plot = None


class TestPlotTrainingCurves:
    """测试 plot_training_curves 函数"""

    @pytest.fixture
    def sample_metrics(self):
        """提供示例训练指标数据"""
        return {
            "train_loss": [1.0, 0.8, 0.6, 0.4, 0.3],
            "val_loss": [1.1, 0.9, 0.7, 0.5, 0.4],
            "train_acc": [0.6, 0.75, 0.85, 0.9, 0.95],
            "val_acc": [0.55, 0.7, 0.8, 0.85, 0.88],
        }

    @pytest.fixture
    def sample_metrics_empty(self):
        """提供空数据"""
        return {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

    def test_module_exists(self):
        """测试模块是否可导入"""
        assert plot_training_curves is not None, "training_curves 模块未找到"
        assert save_training_plot is not None, "training_curves 模块未找到"

    def test_plot_creates_figure(self, sample_metrics):
        """测试函数是否创建了figure对象"""
        import matplotlib.pyplot as plt

        fig = plot_training_curves(sample_metrics)
        assert fig is not None, "应该返回Figure对象"
        plt.close(fig)

    def test_correct_number_of_subplots(self, sample_metrics):
        """测试是否有2个子图（loss和accuracy）"""
        import matplotlib.pyplot as plt

        fig = plot_training_curves(sample_metrics)
        axes = fig.get_axes()

        # 应该有两个subplot
        assert len(axes) == 2, f"期望2个子图，实际{len(axes)}个"
        plt.close(fig)

    def test_subplot_titles(self, sample_metrics):
        """测试子图标题是否正确"""
        import matplotlib.pyplot as plt

        fig = plot_training_curves(sample_metrics)
        axes = fig.get_axes()

        # 获取子图标题
        title1 = axes[0].get_title()
        title2 = axes[1].get_title()

        # 至少有一个子图标题包含loss相关信息
        titles_text = f"{title1} {title2}".lower()
        assert "loss" in titles_text, "第一个子图应该有loss相关标题"

        plt.close(fig)

    def test_axes_labels_present(self, sample_metrics):
        """测试坐标轴标签是否存在"""
        import matplotlib.pyplot as plt

        fig = plot_training_curves(sample_metrics)
        axes = fig.get_axes()

        # 检查第一个子图（loss）的xlabel和ylabel
        ax_loss = axes[0]
        assert ax_loss.get_xlabel() != "", "Loss子图应该有x轴标签"
        assert ax_loss.get_ylabel() != "", "Loss子图应该有y轴标签"

        # 检查第二个子图（accuracy）的xlabel和ylabel
        ax_acc = axes[1]
        assert ax_acc.get_xlabel() != "", "Accuracy子图应该有x轴标签"
        assert ax_acc.get_ylabel() != "", "Accuracy子图应该有y轴标签"

        plt.close(fig)

    def test_plot_with_empty_data(self, sample_metrics_empty):
        """测试空数据是否也能正常运行（不抛异常）"""
        import matplotlib.pyplot as plt

        fig = plot_training_curves(sample_metrics_empty)
        assert fig is not None
        plt.close(fig)

    def test_plot_with_single_epoch(self):
        """测试单个epoch数据"""
        import matplotlib.pyplot as plt

        single_epoch_metrics = {
            "train_loss": [0.5],
            "val_loss": [0.6],
            "train_acc": [0.7],
            "val_acc": [0.65],
        }
        fig = plot_training_curves(single_epoch_metrics)
        assert fig is not None
        plt.close(fig)

    def test_plot_with_save_path(self, sample_metrics):
        """测试save_path参数是否生效"""
        import matplotlib.pyplot as plt

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "test_curves.png")
            fig = plot_training_curves(sample_metrics, save_path=save_path)

            # 检查文件是否被创建
            assert os.path.exists(save_path), f"文件应该被保存到 {save_path}"
            plt.close(fig)

    def test_lines_are_plotted(self, sample_metrics):
        """测试数据线是否被正确绘制"""
        import matplotlib.pyplot as plt

        fig = plot_training_curves(sample_metrics)
        axes = fig.get_axes()

        # 检查第一个子图（loss）有几条线
        lines_loss = axes[0].get_lines()
        # 应该有train_loss和val_loss两条线
        assert len(lines_loss) == 2, f"Loss子图应该有2条线，实际{len(lines_loss)}条"

        # 检查第二个子图（accuracy）有几条线
        lines_acc = axes[1].get_lines()
        assert len(lines_acc) == 2, f"Accuracy子图应该有2条线，实际{len(lines_acc)}条"

        plt.close(fig)


class TestSaveTrainingPlot:
    """测试 save_training_plot 辅助函数"""

    @pytest.fixture
    def sample_metrics(self):
        """提供示例训练指标数据"""
        return {
            "train_loss": [1.0, 0.8, 0.6, 0.4],
            "val_loss": [1.1, 0.9, 0.7, 0.5],
            "train_acc": [0.6, 0.75, 0.85, 0.9],
            "val_acc": [0.55, 0.7, 0.8, 0.85],
        }

    def test_function_exists(self):
        """测试函数是否存在"""
        assert save_training_plot is not None

    def test_save_to_file(self, sample_metrics):
        """测试保存功能是否正常"""
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, "training_plot.png")
            save_training_plot(sample_metrics, filename)

            assert os.path.exists(filename), f"文件应该被保存到 {filename}"

    def test_save_with_different_formats(self, sample_metrics):
        """测试不同格式保存"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 测试PNG格式
            png_path = os.path.join(tmpdir, "plot.png")
            save_training_plot(sample_metrics, png_path)
            assert os.path.exists(png_path)

            # 测试PDF格式
            pdf_path = os.path.join(tmpdir, "plot.pdf")
            save_training_plot(sample_metrics, pdf_path)
            assert os.path.exists(pdf_path)

            # 测试SVG格式
            svg_path = os.path.join(tmpdir, "plot.svg")
            save_training_plot(sample_metrics, svg_path)
            assert os.path.exists(svg_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
