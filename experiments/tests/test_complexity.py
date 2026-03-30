"""
[INPUT]: torch, ptflops
[OUTPUT]: ModelComplexityAnalyzer
[POS]: experiments/ 模型复杂度分析
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
import torch
import torch.nn as nn


class TestCountParameters:
    """测试参数计数功能"""

    def test_count_parameters_returns_dict(self):
        """count_parameters should return dict with total and trainable"""
        from experiments.complexity import count_parameters

        model = nn.Linear(100, 10)
        result = count_parameters(model)

        assert isinstance(result, dict)
        assert "total" in result
        assert "trainable" in result

    def test_count_parameters_returns_int(self):
        """count_parameters values should be int"""
        from experiments.complexity import count_parameters

        model = nn.Linear(100, 10)
        result = count_parameters(model)

        assert isinstance(result["total"], int)
        assert isinstance(result["trainable"], int)

    def test_total_equals_trainable_for_simple_model(self):
        """For simple linear model, total should equal trainable"""
        from experiments.complexity import count_parameters

        model = nn.Linear(100, 10)
        result = count_parameters(model)

        # All params are trainable by default
        assert result["total"] == result["trainable"]

    def test_non_trainable_params(self):
        """Frozen layer should have non-trainable params"""
        from experiments.complexity import count_parameters

        model = nn.Linear(100, 10)
        for param in model.parameters():
            param.requires_grad = False

        result = count_parameters(model)
        assert result["trainable"] == 0


class TestMeasureLatency:
    """测试延迟测量功能"""

    def test_measure_latency_returns_stats(self):
        """measure_latency should return mean, std, min, max"""
        from experiments.complexity import measure_latency

        model = nn.Linear(100, 10)
        result = measure_latency(model, input_size=(1, 100), num_iterations=10)

        assert isinstance(result, dict)
        assert "mean_ms" in result
        assert "std_ms" in result
        assert "min_ms" in result
        assert "max_ms" in result

    def test_latency_positive(self):
        """Latency values should be positive"""
        from experiments.complexity import measure_latency

        model = nn.Linear(100, 10)
        result = measure_latency(model, input_size=(1, 100), num_iterations=10)

        assert result["mean_ms"] > 0
        assert result["min_ms"] >= 0

    def test_warmup_runs(self):
        """measure_latency should perform warmup"""
        from experiments.complexity import measure_latency

        model = nn.Linear(100, 10)
        # Should not raise even with small iterations
        result = measure_latency(
            model, input_size=(1, 100), num_iterations=5, warmup_iterations=2
        )
        assert "mean_ms" in result


class TestModelComplexityAnalysis:
    """测试模型复杂度分析"""

    def test_analyze_model_complexity_returns_dict(self):
        """analyze_model_complexity should return comprehensive dict"""
        from experiments.complexity import analyze_model_complexity

        model = nn.Linear(100, 10)
        result = analyze_model_complexity(model, input_size=(1, 100))

        assert isinstance(result, dict)
        assert "parameters" in result
        assert "latency" in result

    def test_flops_included(self):
        """analyze_model_complexity should include FLOPs"""
        from experiments.complexity import analyze_model_complexity

        model = nn.Sequential(nn.Linear(100, 50), nn.ReLU(), nn.Linear(50, 10))
        result = analyze_model_complexity(model, input_size=(1, 100))

        assert "flops" in result or "macs" in result


class TestGenerateComplexityTable:
    """测试复杂度表格生成"""

    def test_generate_complexity_table_returns_string(self):
        """generate_complexity_table should return markdown string"""
        from experiments.complexity import generate_complexity_table

        results = [
            {
                "model_name": "ResNet50",
                "parameters": {"total": 25e6, "trainable": 25e6},
                "flops": 4e9,
                "latency": {"mean_ms": 10.5},
            }
        ]
        table = generate_complexity_table(results)
        assert isinstance(table, str)

    def test_table_contains_model_name(self):
        """Table should contain model names"""
        from experiments.complexity import generate_complexity_table

        results = [
            {
                "model_name": "ResNet50",
                "parameters": {"total": 25e6, "trainable": 25e6},
                "flops": 4e9,
                "latency": {"mean_ms": 10.5},
            }
        ]
        table = generate_complexity_table(results)
        assert "ResNet50" in table


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
