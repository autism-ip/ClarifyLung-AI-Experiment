"""
experiments/ - 实验模块
> L1 | 父级: ../CLAUDE.md

成员清单
benchmark/         : 基准模型与对比实验
    ├── __init__.py
    ├── models.py           : ResNet50/ViT/Hybrid 模型工厂
    ├── benchmarker.py      : BenchmarkResult, ModelBenchmark, 对比表格
    └── tests/              : 基准测试
complexity.py      : 模型复杂度分析 (参数量/FLOPs/延迟)
visualization/     : 可解释性可视化模块
    ├── __init__.py
    ├── gradcam.py          : Grad-CAM++ 热力图生成
    ├── attention_maps.py   : Transformer注意力可视化
    └── ...
ablation/          : 消融实验框架
    ├── __init__.py
    ├── configs.py          : AblationConfig, ABLATION_CONFIGS
    ├── ablator.py          : AblationStudy 运行器
    └── ...
interpretability.py : 可解释性分析 (Grad-CAM/注意力可视化) [待实现]

法则: 实验可复现 · 结果可对比 · 文档完整

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from .benchmark import (
    create_resnet50,
    create_vit,
    create_hybrid_basic,
    create_hybrid_advanced,
    BenchmarkResult,
    ModelBenchmark,
    generate_comparison_table,
)
from .complexity import (
    count_parameters,
    measure_latency,
    analyze_model_complexity,
    generate_complexity_table,
    ModelComplexityAnalyzer,
)

# Visualization imports are optional (require grad-cam package)
try:
    from .visualization import (
        GradCAMVisualizer,
        overlay_heatmap,
        AttentionVisualizer,
        visualize_attention,
        visualize_multihead_attention,
    )
except ImportError:
    # grad-cam not installed, visualization will be unavailable
    pass

from .ablation import (
    AblationStudy,
    AblationConfig,
    ABLATION_CONFIGS,
    get_ablation_config,
    list_available_configs,
)

__all__ = [
    # benchmark
    "create_resnet50",
    "create_vit",
    "create_hybrid_basic",
    "create_hybrid_advanced",
    "BenchmarkResult",
    "ModelBenchmark",
    "generate_comparison_table",
    # complexity
    "count_parameters",
    "measure_latency",
    "analyze_model_complexity",
    "generate_complexity_table",
    "ModelComplexityAnalyzer",
    # ablation
    "AblationStudy",
    "AblationConfig",
    "ABLATION_CONFIGS",
    "get_ablation_config",
    "list_available_configs",
]

# Visualization exports are optional (require grad-cam package)
if "GradCAMVisualizer" in dir():
    __all__.extend([
        "GradCAMVisualizer",
        "overlay_heatmap",
        "AttentionVisualizer",
        "visualize_attention",
        "visualize_multihead_attention",
    ])