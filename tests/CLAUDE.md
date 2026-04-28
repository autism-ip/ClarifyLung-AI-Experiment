# tests/ - 单元测试模块
> L2 | 父级: ../CLAUDE.md

成员清单
test_dataset_loading.py     : 三数据集统一加载器测试 (CustomLungDataset, 路径探测)
test_download_datasets.py   : Kaggle下载脚本测试 (DATASETS配置, 凭证检查, 下载/验证/清理)
test_model_forward.py       : 混合模型端到端测试 (初始化, 前向/反向传播, 训练步)

法则: 覆盖核心功能 · 快速执行 · CI可集成

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
