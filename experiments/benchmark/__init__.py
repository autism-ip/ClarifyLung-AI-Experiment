"""
experiments/benchmark/ - 模型对比与消融实验模块
> L2 | 父级: ../CLAUDE.md

成员清单
__init__.py         : 模块初始化
models.py           : 基准模型工厂 (ResNet50/ViT/Hybrid)
benchmarker.py       : 基准测试执行与结果对比

法则: 模型统一接口 · 结果可复现 · 表格可视化

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from .models import (
    create_resnet50,
    create_vit,
    create_hybrid_basic,
    create_hybrid_advanced,
)
from .benchmarker import (
    BenchmarkResult,
    ModelBenchmark,
    generate_comparison_table,
    save_results,
    load_results,
)

__all__ = [
    "create_resnet50",
    "create_vit",
    "create_hybrid_basic",
    "create_hybrid_advanced",
    "BenchmarkResult",
    "ModelBenchmark",
    "generate_comparison_table",
    "save_results",
    "load_results",
]
