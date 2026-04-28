# training/ - 训练微调模块
> L2 | 父级: ../CLAUDE.md

成员清单
trainer.py : 完整训练流程 (Trainer + TrainingConfig)
           - 差分学习率: CNN层小LR, Transformer层大LR (前缀匹配分组)
           - 混合精度训练: torch.amp.GradScaler + autocast (设备自适应)
           - 早停机制: 验证指标无改善时自动停止
           - 检查点: 最佳模型自动保存/恢复
           - 学习率调度: CosineAnnealing

法则: 配置驱动 · 设备无关 · 可恢复 · 可监控

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
