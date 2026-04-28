# models/ - 模型架构模块
> L2 | 父级: ../CLAUDE.md

成员清单
__init__.py                 : 包入口，暴露 HybridModel, ConfigurableHybrid
hybrid_model.py             : CNN-Transformer混合模型组装器，组合全部组件
                            - 支持 use_transformer=False / use_cross_attention=False 消融开关
                            - 支持 gate_type=None 跳过门控
configurable_hybrid.py      : 消融实验适配器 (ConfigurableHybrid)
                            - 将 AblationConfig 布尔开关映射到 HybridModel 参数
                            - 确保消融实验在真实组件上进行
components/feature_extractor.py : 多尺度特征提取器 (layer3+layer4融合)
components/gating.py        : 门控机制 (SE模块 / Sigmoid / None跳过)
components/transformer.py   : Transformer编码器 + 图像分块 + 位置编码
components/cross_attention.py : 双流交叉注意力模块
components/classification.py : 最终分类头

依赖关系
  hybrid_model.py → components/*
  外部调用方 → models/__init__.py → hybrid_model.py
  根目录 model.py → models.hybrid_model (兼容入口)

法则: 组件解耦 · 可组合 · 预训练权重兼容

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
