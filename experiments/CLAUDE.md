# experiments/ - 实验分析模块
> L2 | 父级: ../CLAUDE.md

成员清单
metrics.py                  : 评估指标计算 (EvaluationMetrics, compute_metrics)
                            - Accuracy, Precision, Recall, F1, AUC-ROC (OvR/OvO)
                            - 混淆矩阵
complexity.py               : 模型复杂度分析 (ModelComplexityAnalyzer)
                            - 参数量统计, FLOPs计算
benchmark/models.py         : 基准模型工厂 (create_resnet50, create_vit, create_hybrid_basic)
                            - create_hybrid_advanced 已统一指向 models.HybridModel
benchmark/benchmarker.py    : 基准测试执行器 (ModelBenchmark, BenchmarkResult)
                            - 多模型对比, 结果保存, 表格生成
ablation/configs.py         : 消融配置定义 (AblationConfig, ABLATION_CONFIGS)
ablation/ablator.py         : 消融实验运行器 (AblationStudy)
                            - 组件移除/替换, 贡献度分析
cross_validation/validator.py       : K折分层交叉验证 (KFoldCrossValidator)
cross_validation/statistical_tests.py : 统计显著性检验 (shapiro_wilk, paired_t_test)
visualization/gradcam.py            : Grad-CAM++ 热力图可视化
visualization/attention_maps.py     : Transformer注意力图可视化
visualization/training_curves.py    : 训练/验证曲线绘制
visualization/confusion_matrix.py   : 混淆矩阵可视化
visualization/class_distribution.py : 类别分布图
visualization/model_comparison.py   : 模型对比柱状图
tests/                      : 实验模块单元测试 (144 tests, 100% pass)

法则: 可复现实验 · 统计严谨 · 可视化完备 · 测试覆盖

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
