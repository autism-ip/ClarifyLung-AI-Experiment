# training/ - 训练微调模块
> L2 | 父级: ../CLAUDE.md

成员清单
trainer.py        : 训练器主类，Trainer封装训练循环、验证、早停、检查点保存
engine.py         : 训练引擎，支持分阶段训练(冻结/解冻策略)
optimizers/       : 子目录，优化器配置
  - factory.py         : 优化器工厂，支持SGD/Adam/AdamW
  - differential_lr.py : 差分学习率实现，不同层不同LR
schedulers/       : 子目录，学习率调度
  - factory.py         : 调度器工厂，支持Step/Cosine/ReduceLROnPlateau
  - warmup.py          : 学习率warmup实现
loggers/          : 子目录，日志记录
  - tensorboard.py     : TensorBoard日志
  - wandb_logger.py    : Weights&Biases日志
  - metrics_tracker.py : 指标追踪器

法则: 支持分阶段训练 · 差分学习率 · 完善的日志记录

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
