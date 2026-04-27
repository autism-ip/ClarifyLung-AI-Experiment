# models/
> L2 | 父级: /CLAUDE.md

成员清单
- `__init__.py`: 包入口，暴露 HybridModel / CLASS_NAMES
- `hybrid_model.py`: CNN-Transformer 混合模型组装器，组合全部组件
- `components/feature_extractor.py`: CNN 多尺度特征提取 (FeatureFusion, MutilScaleFeatureExtractor)
- `components/gating.py`: 通道门控机制 (GatingMechanism)
- `components/transformer.py`: Transformer 编码器 + 图像分块 + 位置编码
- `components/cross_attention.py`: 多层交叉注意力 (CrossAttentionLayer, CrossAttention)
- `components/classification.py`: 分类头 (ClassificationHead)

依赖关系
- `hybrid_model.py` -> `components/*`
- 外部调用方 -> `models/__init__.py` -> `hybrid_model.py`
- 根目录 `model.py` -> `models.hybrid_model` (兼容入口)

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
