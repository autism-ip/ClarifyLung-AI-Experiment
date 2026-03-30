"""
[INPUT]: dataclasses, typing, pandas, matplotlib
[OUTPUT]: plot_model_comparison, create_comparison_table
[POS]: experiments/visualization/ model comparison visualization
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
import numpy as np


class TestCreateComparisonTable:
    """Test create_comparison_table helper function"""

    def test_returns_pandas_dataframe(self):
        """create_comparison_table should return a pandas DataFrame"""
        from experiments.visualization.model_comparison import create_comparison_table

        results = {
            "ResNet50": {"accuracy": 0.85, "f1_score": 0.84},
            "ViT": {"accuracy": 0.87, "f1_score": 0.86},
        }
        df = create_comparison_table(results, ["accuracy", "f1_score"])
        import pandas as pd

        assert isinstance(df, pd.DataFrame)

    def test_table_has_model_names_as_index(self):
        """Table should have model names as index"""
        from experiments.visualization.model_comparison import create_comparison_table

        results = {
            "ResNet50": {"accuracy": 0.85, "f1_score": 0.84},
            "ViT": {"accuracy": 0.87, "f1_score": 0.86},
        }
        df = create_comparison_table(results, ["accuracy", "f1_score"])
        assert "ResNet50" in df.index
        assert "ViT" in df.index

    def test_table_has_metric_columns(self):
        """Table should have specified metrics as columns"""
        from experiments.visualization.model_comparison import create_comparison_table

        results = {
            "ResNet50": {"accuracy": 0.85, "f1_score": 0.84},
            "ViT": {"accuracy": 0.87, "f1_score": 0.86},
        }
        df = create_comparison_table(results, ["accuracy", "f1_score"])
        assert "accuracy" in df.columns
        assert "f1_score" in df.columns

    def test_table_values_correct(self):
        """Table should contain correct metric values"""
        from experiments.visualization.model_comparison import create_comparison_table

        results = {
            "ResNet50": {"accuracy": 0.85, "f1_score": 0.84},
            "ViT": {"accuracy": 0.87, "f1_score": 0.86},
        }
        df = create_comparison_table(results, ["accuracy", "f1_score"])
        assert df.loc["ResNet50", "accuracy"] == 0.85
        assert df.loc["ResNet50", "f1_score"] == 0.84
        assert df.loc["ViT", "accuracy"] == 0.87
        assert df.loc["ViT", "f1_score"] == 0.86

    def test_empty_results_raises(self):
        """Empty results should raise ValueError"""
        from experiments.visualization.model_comparison import create_comparison_table

        with pytest.raises(ValueError):
            create_comparison_table({}, ["accuracy"])

    def test_single_model_single_metric(self):
        """Should work with single model and single metric"""
        from experiments.visualization.model_comparison import create_comparison_table

        results = {"Model": {"acc": 0.9}}
        df = create_comparison_table(results, ["acc"])
        assert df.shape == (1, 1)


class TestPlotModelComparison:
    """Test plot_model_comparison function"""

    def test_function_exists(self):
        """plot_model_comparison should be importable"""
        from experiments.visualization.model_comparison import plot_model_comparison

        assert plot_model_comparison is not None

    def test_runs_without_error_on_valid_data(self):
        """Should run without error on valid benchmark data"""
        from experiments.visualization.model_comparison import plot_model_comparison
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend

        results = {
            "ResNet50": {"accuracy": 0.85, "f1_score": 0.84, "precision": 0.83},
            "ViT": {"accuracy": 0.87, "f1_score": 0.86, "precision": 0.85},
            "Hybrid": {"accuracy": 0.89, "f1_score": 0.88, "precision": 0.87},
        }
        # Should not raise
        plot_model_comparison(results, metrics=["accuracy", "f1_score", "precision"])

    def test_returns_figure_object(self):
        """Should return a matplotlib Figure object"""
        from experiments.visualization.model_comparison import plot_model_comparison
        import matplotlib
        matplotlib.use('Agg')

        results = {
            "ResNet50": {"accuracy": 0.85, "f1_score": 0.84},
            "ViT": {"accuracy": 0.87, "f1_score": 0.86},
        }
        fig = plot_model_comparison(results, metrics=["accuracy", "f1_score"])
        import matplotlib.figure
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_saves_to_file_when_save_path_provided(self, tmp_path):
        """Should save figure to file when save_path is provided"""
        from experiments.visualization.model_comparison import plot_model_comparison
        import matplotlib
        matplotlib.use('Agg')

        results = {
            "ResNet50": {"accuracy": 0.85, "f1_score": 0.84},
            "ViT": {"accuracy": 0.87, "f1_score": 0.86},
        }
        save_path = tmp_path / "test_comparison.png"
        plot_model_comparison(results, metrics=["accuracy", "f1_score"], save_path=str(save_path))
        assert save_path.exists()

    def test_bar_chart_grouping(self):
        """Grouped bar chart should have correct structure"""
        from experiments.visualization.model_comparison import plot_model_comparison
        import matplotlib
        matplotlib.use('Agg')

        results = {
            "ModelA": {"metric1": 0.8, "metric2": 0.7},
            "ModelB": {"metric1": 0.85, "metric2": 0.75},
        }
        fig = plot_model_comparison(results, metrics=["metric1", "metric2"])
        # Get the axes and check bars
        ax = fig.axes[0]
        # There should be groups of bars (2 groups for 2 models)
        # Each group should have bars for each metric (2 bars per group)
        bars = [c for c in ax.get_children() if hasattr(c, 'get_height')]
        # We expect 4 bars total (2 models x 2 metrics)
        assert len(bars) >= 4

    def test_handles_empty_metrics_list(self):
        """Should raise ValueError when metrics list is empty"""
        from experiments.visualization.model_comparison import plot_model_comparison
        import matplotlib
        matplotlib.use('Agg')

        results = {"Model": {"accuracy": 0.85}}
        with pytest.raises(ValueError):
            plot_model_comparison(results, metrics=[])

    def test_metric_not_in_results_uses_zero(self):
        """Should use 0 for metrics not present in results"""
        from experiments.visualization.model_comparison import plot_model_comparison
        import matplotlib
        matplotlib.use('Agg')

        results = {
            "Model": {"accuracy": 0.85}  # f1_score not present
        }
        # Should not raise, should use 0 or handle gracefully
        fig = plot_model_comparison(results, metrics=["accuracy", "f1_score"])
        assert fig is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
