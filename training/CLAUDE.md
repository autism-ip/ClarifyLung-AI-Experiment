# training/ - 训练微调模块
> L2 | 父级: ../CLAUDE.md

成员清单
trainer.py : 完整训练流程 (Trainer + TrainingConfig)
           - 差分学习率: CNN层小LR, Transformer层大LR (前缀匹配分组)
           - 混合精度训练: torch.amp.GradScaler + autocast (设备自适应)
           - 早停机制: 可配置监控指标 (val_acc/val_loss) 与模式 (max/min)
           - 检查点: 保存完整状态 (model/optimizer/scheduler/scaler/rng)
           - 学习率调度: CosineAnnealing / OneCycle / ReduceLROnPlateau
           - 日志系统: 统一使用 logging 模块，支持分级输出

法则: 配置驱动 · 设备无关 · 可恢复 · 可监控 · 日志完备

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
