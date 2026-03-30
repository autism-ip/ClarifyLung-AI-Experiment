# experiments/ - 实验分析模块
> L2 | 父级: ../CLAUDE.md

## 成员清单

### 核心模块
metrics.py        : EvaluationMetrics dataclass, compute_metrics() 评估指标计算
complexity.py     : ModelComplexityAnalyzer, count_parameters/measure_latency 复杂度分析

### benchmark/ - 基准模型对比
├── __init__.py
├── models.py          : create_resnet50/vit/hybrid_basic/hybrid_advanced 模型工厂
├── benchmarker.py     : BenchmarkResult dataclass, ModelBenchmark 对比运行器

### ablation/ - 消融实验
├── __init__.py
├── configs.py         : AblationConfig dataclass, ABLATION_CONFIGS 消融配置
├── ablator.py         : AblationStudy.run_ablation() 消融实验运行器

### cross_validation/ - 交叉验证
├── __init__.py
├── statistical_tests.py: shapiro_wilk/paired_t_test/wilcoxon 统计显著性检验
├── validator.py       : KFoldCrossValidator 5折分层交叉验证

### visualization/ - 可解释性可视化
├── __init__.py
├── gradcam.py         : GradCAMVisualizer HiResCAM+GradCAM++ 热力图
├── attention_maps.py   : AttentionVisualizer Transformer注意力可视化
├── class_distribution.py: plot_class_distribution, plot_pie_chart 类别分布可视化
├── model_comparison.py: plot_model_comparison grouped bar chart, create_comparison_table DataFrame
├── training_curves.py  : plot_training_curves, save_training_plot 训练曲线可视化

### tests/ - 测试套件 (119 tests, 100% pass)
├── test_metrics.py
├── test_ablation_configs.py
├── test_statistical_tests.py
├── test_validator.py
├── test_complexity.py
├── test_benchmark_models.py
├── test_benchmarker.py
├── test_gradcam.py
├── test_attention_maps.py
├── test_class_distribution.py
├── test_model_comparison.py
└── test_training_curves.py

## 消融实验设计

### 消融变量矩阵
| 组件 | 变体 | 说明 |
|------|------|------|
| 多尺度特征 | full vs single | layer3+layer4 vs 仅layer4 |
| 门控机制 | SE vs Sigmoid vs None | 三种门控策略对比 |
| 交叉注意力 | 2层 vs 1层 vs 0层 | 注意力层数消融 |
| Transformer | 6层 vs 4层 vs 2层 vs None | 层数消融，最终vs纯CNN |

### 消融配置 (ABLATION_CONFIGS)
```python
baseline_cnn       : multi_scale=False, gate=None, transformer=False, cross_attention=False
multiscale_only    : multi_scale=True, gate=None, transformer=False, cross_attention=False
gating_added      : multi_scale=True, gate='se', transformer=False, cross_attention=False
transformer_added : multi_scale=True, gate='se', transformer=True, cross_attention=False
full_hybrid       : multi_scale=True, gate='se', transformer=True, cross_attention=True
```

### 消融实验输出格式
| 消融组件 | 基线Acc | 消融后Acc | ΔAcc | p-value | 显著性 |
|---------|--------|----------|------|---------|--------|
| 交叉注意力 | 85.2% | 82.5% | -2.7% | 0.001 | *** |
| 多尺度特征 | 85.2% | 83.1% | -2.1% | 0.003 | ** |
| 门控机制 | 85.2% | 83.8% | -1.4% | 0.021 | * |
| 纯CNN基线 | 85.2% | 80.3% | -4.9% | <0.001 | *** |

## 快速开始

```python
from experiments import (
    # 基准模型
    create_resnet50, create_vit, create_hybrid_basic, create_hybrid_advanced,
    ModelBenchmark, BenchmarkResult, generate_comparison_table,
    # 消融实验
    AblationStudy, ABLATION_CONFIGS,
    # 复杂度分析
    analyze_model_complexity, generate_complexity_table,
    # 评估指标
    EvaluationMetrics, compute_metrics,
    # 可视化
    GradCAMVisualizer, overlay_heatmap, plot_class_distribution, plot_pie_chart,
    plot_model_comparison, create_comparison_table,
)

# K折交叉验证
from experiments.cross_validation import KFoldCrossValidator, compare_fold_results
```

法则: TDD验证 · 100+测试100%通过 · 可复现结果

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
