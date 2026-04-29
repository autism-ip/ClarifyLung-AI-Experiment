# scripts/ - 工具脚本模块
> L2 | 父级: ../CLAUDE.md

成员清单
download_datasets.py         : Kaggle数据集下载器，支持凭证检查、自动下载、验证、清理
benchmark_experiment.py      : 基准模型对比实验CLI入口 (ResNet50/ViT/Hybrid)
                            - hybrid_advanced 统一使用 models.HybridModel
ablation_experiment.py       : 消融实验CLI入口 (5种配置逐层叠加)
                            - 使用 models.ConfigurableHybrid (基于真实组件)
cross_validation_experiment.py : 交叉验证实验CLI入口 (K折+统计检验)
                            - 使用 models.HybridModel (与benchmark一致)
baseline_experiment.py       : 小规模快速验证脚本
                            - 分层抽样 + 类别不平衡检测
validate_pipeline.py         : 3步流水线烟雾测试
                            - 数据加载 → 模型前向 → 训练流程
visualize_experiment_results.py : 实验结果可视化CLI (自动读取JSON+.npy)
                                - 支持 benchmark/ablation/crossval 三种实验类型自动检测
                                - 自动生成对比图 + 混淆矩阵 (需实验脚本保存的.npy)
visualize_gradcam.py         : Grad-CAM单图可视化CLI
                            - 需手动指定 --image, --checkpoint, --output
                            - 支持自动预测类别或手动指定 target-class
submit_benchmark.sh          : SLURM批作业: 基准实验 (24h, --no-plot)
                             - 支持 QUICK_TEST=1 环境变量快速自检
submit_ablation.sh           : SLURM批作业: 消融实验 (48h)
                             - 支持 QUICK_TEST=1 环境变量快速自检
submit_crossval.sh           : SLURM批作业: 交叉验证 (72h)
                             - 支持 QUICK_TEST=1 环境变量快速自检
utils.py                     : 实验脚本公共工具
                             - set_seed / get_device
                             - split_dataset_with_transforms: 防泄漏的 train/val/test 划分
                             - train_model: 通用训练循环 + 差分学习率 + 检查点保存
                             - create_quick_test_datasets: 200样本分层子集生成

法则: 可独立运行 · 命令行友好 · 失败优雅 · 本地CLI+远程SLURM双模式
        统一模型入口: 所有实验脚本共享 models.HybridModel 核心架构

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
