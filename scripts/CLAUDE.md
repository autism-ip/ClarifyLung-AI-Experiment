# scripts/ - 工具脚本模块
> L2 | 父级: ../CLAUDE.md

成员清单
download_datasets.py         : Kaggle数据集下载器，支持凭证检查、自动下载、验证、清理
benchmark_experiment.py      : 基准模型对比实验CLI入口 (ResNet50/ViT/Hybrid)
ablation_experiment.py       : 消融实验CLI入口 (5种配置逐层叠加)
cross_validation_experiment.py : 交叉验证实验CLI入口 (K折+统计检验)
submit_benchmark.sh          : SLURM批作业: 基准实验 (24h, --no-plot)
submit_ablation.sh           : SLURM批作业: 消融实验 (48h)
submit_crossval.sh           : SLURM批作业: 交叉验证 (72h)

法则: 可独立运行 · 命令行友好 · 失败优雅 · 本地CLI+远程SLURM双模式

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
