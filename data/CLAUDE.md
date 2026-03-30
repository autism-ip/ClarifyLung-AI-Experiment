# data/ - 数据管理模块
> L2 | 父级: ../CLAUDE.md

成员清单
dataset.py        : 数据集定义，XRayDataset类继承torch.utils.data.Dataset，支持train/val/test划分
augmentation.py   : 数据增强策略，基础增强(旋转/翻转/裁剪) + 高级增强(CutMix/MixUp/弹性变形)
loader.py         : 数据加载器工厂，支持balance sampling和multi-worker
visualization.py  : 数据可视化工具，类别分布图、样本网格展示

法则: 输入输出明确 · 支持配置驱动 · 可复现随机种子

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
