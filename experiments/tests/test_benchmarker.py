"""
[INPUT]: dataclasses, typing
[OUTPUT]: BenchmarkResult dataclass, ModelBenchmark class, comparison utilities
[POS]: experiments/benchmark/ benchmark execution and reporting
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from dataclasses import is_dataclass


class TestBenchmarkResult:
    """测试 BenchmarkResult 数据类"""

    def test_is_dataclass(self):
        """BenchmarkResult should be a dataclass"""
        from experiments.benchmark.benchmarker import BenchmarkResult

        assert is_dataclass(BenchmarkResult)

    def test_benchmark_result_required_fields(self):
        """BenchmarkResult should have required fields"""
        from experiments.benchmark.benchmarker import BenchmarkResult

        result = BenchmarkResult(
            model_name="test",
            accuracy=0.85,
            macro_f1=0.84,
            auc_roc=0.92,
        )
        assert result.model_name == "test"
        assert result.accuracy == 0.85
        assert result.macro_f1 == 0.84
        assert result.auc_roc == 0.92

    def test_accuracy_bounds(self):
        """accuracy should be between 0 and 1"""
        from experiments.benchmark.benchmarker import BenchmarkResult

        result = BenchmarkResult(model_name="test", accuracy=0.5, macro_f1=0.5, auc_roc=0.5)
        assert 0 <= result.accuracy <= 1

    def test_optional_fields(self):
        """BenchmarkResult should have optional fields with defaults"""
        from experiments.benchmark.benchmarker import BenchmarkResult

        result = BenchmarkResult(model_name="test", accuracy=0.85, macro_f1=0.84, auc_roc=0.92)
        # precision and recall should have default values
        assert hasattr(result, "precision")
        assert hasattr(result, "recall")


class TestComparisonTable:
    """测试对比表格生成"""

    def test_generate_comparison_table_returns_string(self):
        """generate_comparison_table should return markdown string"""
        from experiments.benchmark.benchmarker import generate_comparison_table

        results = [
            {"model_name": "ResNet50", "accuracy": 0.82, "macro_f1": 0.81},
            {"model_name": "ViT", "accuracy": 0.84, "macro_f1": 0.83},
        ]
        table = generate_comparison_table(results)
        assert isinstance(table, str)

    def test_table_contains_model_names(self):
        """Table should contain model names"""
        from experiments.benchmark.benchmarker import generate_comparison_table

        results = [
            {"model_name": "ResNet50", "accuracy": 0.82, "macro_f1": 0.81},
            {"model_name": "ViT", "accuracy": 0.84, "macro_f1": 0.83},
        ]
        table = generate_comparison_table(results)
        assert "ResNet50" in table
        assert "ViT" in table

    def test_table_contains_header(self):
        """Table should contain header row"""
        from experiments.benchmark.benchmarker import generate_comparison_table

        results = [{"model_name": "ResNet50", "accuracy": 0.82, "macro_f1": 0.81}]
        table = generate_comparison_table(results)
        assert "accuracy" in table.lower() or "model" in table.lower()


class TestModelBenchmark:
    """测试 ModelBenchmark 类"""

    def test_model_benchmark_has_required_methods(self):
        """ModelBenchmark should have run_benchmark method"""
        from experiments.benchmark.benchmarker import ModelBenchmark

        assert hasattr(ModelBenchmark, "run_benchmark")

    def test_model_benchmark_instantiation(self):
        """ModelBenchmark should be instantiable"""
        from experiments.benchmark.benchmarker import ModelBenchmark
        import torch.nn as nn

        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(3 * 224 * 224, 3)

            def forward(self, x):
                return self.fc(x.flatten(1))

        model = DummyModel()
        benchmark = ModelBenchmark(model, "DummyModel")
        assert benchmark is not None
        assert benchmark.model_name == "DummyModel"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
