# data/ - 数据管理模块
> L2 | 父级: ../CLAUDE.md

成员清单
custom_dataset.py : 三数据集统一加载器 (核心)
                  - CustomLungDataset: 自动探测Kaggle嵌套结构
                  - merge_datasets: 合并三数据集
                  - create_weighted_sampler: 类别不平衡时的 WeightedRandomSampler
                  - LabelSchema: 统一标签映射 (normal=0, benign=1, malignant=2)
                  - 注: 防泄漏的 train/val/test 划分请使用 scripts/utils.py 的 split_dataset_with_transforms
augmentation.py   : 数据增强策略
                  - 基础增强: 旋转/翻转/裁剪/颜色抖动
                  - 高级增强: CutMix/MixUp/RandomErasing
                  - get_train_augmentation / get_val_augmentation
visualization.py  : 数据可视化工具，类别分布图、样本网格展示

法则: 输入输出明确 · 支持配置驱动 · 可复现随机种子 · 路径自适应探测

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
