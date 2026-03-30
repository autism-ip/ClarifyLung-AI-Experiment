# =============================================================================
# TDD Phase 1: RED - 失败的测试用例
# =============================================================================
"""
[INPUT]: matplotlib, numpy
[OUTPUT]: plot_class_distribution, plot_pie_chart
[POS]: experiments/visualization/ 类别分布可视化
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import pytest
import numpy as np


# -----------------------------------------------------------------------------
# Test: plot_class_distribution Bar Chart Creation
# -----------------------------------------------------------------------------
def test_plot_class_distribution_creates_bar_chart():
    """plot_class_distribution should create bar chart showing class distribution per dataset"""
    from experiments.visualization.class_distribution import plot_class_distribution

    # Sample data: three datasets with 3 classes each
    class_counts_dict = {
        "IQ-OTHNCCD": {0: 240, 1: 240, 2: 240},
        "LungColon": {0: 500, 1: 500, 2: 500},
        "Lung4Types": {0: 468, 1: 450, 2: 482},
    }
    class_names = ["normal", "benign", "malignant"]

    # Should not raise any errors
    fig = plot_class_distribution(class_counts_dict, class_names=class_names)

    assert fig is not None
    # Verify figure is a matplotlib Figure object
    import matplotlib.figure
    assert isinstance(fig, matplotlib.figure.Figure)


# -----------------------------------------------------------------------------
# Test: plot_pie_chart Creation
# -----------------------------------------------------------------------------
def test_plot_pie_chart_creates_chart():
    """plot_pie_chart should create a pie chart with given distribution"""
    from experiments.visualization.class_distribution import plot_pie_chart

    dist = [1208, 1200, 1201]
    labels = ["normal", "benign", "malignant"]
    title = "Class Distribution"

    fig = plot_pie_chart(dist, labels, title)

    assert fig is not None
    import matplotlib.figure
    assert isinstance(fig, matplotlib.figure.Figure)


# -----------------------------------------------------------------------------
# Test: Legend Correctness
# -----------------------------------------------------------------------------
def test_plot_class_distribution_legend_correctness():
    """plot_class_distribution should display correct legend for classes"""
    from experiments.visualization.class_distribution import plot_class_distribution

    class_counts_dict = {
        "Dataset1": {0: 100, 1: 200, 2: 150},
    }
    class_names = ["normal", "benign", "malignant"]

    fig = plot_class_distribution(class_counts_dict, class_names=class_names)

    # Check that legend exists and contains correct labels
    import matplotlib.figure
    assert isinstance(fig, matplotlib.figure.Figure)

    # Access the axes and check legend
    axes = fig.axes
    assert len(axes) > 0

    # Get legend labels if legend exists
    legend = axes[0].get_legend()
    if legend is not None:
        legend_texts = [t.get_text() for t in legend.get_texts()]
        assert "normal" in legend_texts
        assert "benign" in legend_texts
        assert "malignant" in legend_texts


# -----------------------------------------------------------------------------
# Test: Function Runs Without Error on Sample Data
# -----------------------------------------------------------------------------
def test_plot_class_distribution_runs_without_error():
    """plot_class_distribution should run without error on typical sample data"""
    from experiments.visualization.class_distribution import plot_class_distribution

    # Realistic sample data matching the three datasets
    class_counts_dict = {
        "IQ-OTHNCCD": {0: 240, 1: 240, 2: 240},
        "LungColon": {0: 500, 1: 500, 2: 500},
        "Lung4Types": {0: 468, 1: 450, 2: 482},
    }

    # Should run without raising any exceptions
    fig = plot_class_distribution(class_counts_dict)
    assert fig is not None


# -----------------------------------------------------------------------------
# Test: pie_chart_with_save_path
# -----------------------------------------------------------------------------
def test_plot_pie_chart_with_save_path():
    """plot_pie_chart should save to file when save_path is provided"""
    from experiments.visualization.class_distribution import plot_pie_chart
    import tempfile
    import os

    dist = [1208, 1200, 1201]
    labels = ["normal", "benign", "malignant"]
    title = "Test Pie Chart"

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "test_pie.png")
        fig = plot_pie_chart(dist, labels, title, save_path=save_path)

        assert fig is not None
        assert os.path.exists(save_path)


# -----------------------------------------------------------------------------
# Test: bar_chart_with_save_path
# -----------------------------------------------------------------------------
def test_plot_class_distribution_with_save_path():
    """plot_class_distribution should save to file when save_path is provided"""
    from experiments.visualization.class_distribution import plot_class_distribution
    import tempfile
    import os

    class_counts_dict = {
        "Dataset1": {0: 100, 1: 200, 2: 150},
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "test_bar.png")
        fig = plot_class_distribution(class_counts_dict, save_path=save_path)

        assert fig is not None
        assert os.path.exists(save_path)


# -----------------------------------------------------------------------------
# Test: Default Class Names When None Provided
# -----------------------------------------------------------------------------
def test_plot_class_distribution_default_class_names():
    """plot_class_distribution should use default class names when class_names is None"""
    from experiments.visualization.class_distribution import plot_class_distribution

    class_counts_dict = {
        "Dataset1": {0: 100, 1: 200, 2: 150},
    }

    # Should not raise error when class_names is None
    fig = plot_class_distribution(class_counts_dict, class_names=None)
    assert fig is not None


# -----------------------------------------------------------------------------
# Test: Empty Dataset Handling
# -----------------------------------------------------------------------------
def test_plot_class_distribution_handles_single_dataset():
    """plot_class_distribution should work with a single dataset"""
    from experiments.visualization.class_distribution import plot_class_distribution

    class_counts_dict = {
        "SingleDataset": {0: 100, 1: 200, 2: 150},
    }

    fig = plot_class_distribution(class_counts_dict)
    assert fig is not None
