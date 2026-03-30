# tests/ - 单元测试模块
> L2 | 父级: ../CLAUDE.md

成员清单
conftest.py       : pytest配置与共享fixture
test_data/        : 子目录，数据模块测试
  - test_dataset.py      : XRayDataset测试
  - test_augmentation.py : 增强策略测试
  - test_loader.py       : 数据加载器测试
test_models/      : 子目录，模型模块测试
  - test_components.py   : 组件单元测试
  - test_hybrid_model.py : 完整模型前向测试
  - test_forward_shape.py: 输出形状验证
test_training/    : 子目录，训练模块测试
  - test_trainer.py      : 训练器逻辑测试
  - test_scheduler.py    : 学习率调度测试
test_utils/       : 子目录，工具函数测试
  - test_metrics.py      : 指标计算测试
  - test_io.py           : IO操作测试

法则: 覆盖核心功能 · 快速执行 · CI可集成

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
