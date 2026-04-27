# configs/ - 配置文件模块
> L2 | 父级: ../CLAUDE.md

成员清单
dataset_config.py  : 数据集路径配置 (DATASET_PATHS, DATASET_INFO)
                    - 优先从环境变量 LUNG_DATASET_DIR 读取根目录
                    - 回退到项目本地 datasets/ 目录
                    - dataset1: IQ-OTHNCCD Lung Cancer Dataset
                    - dataset2: Lung and Colon Cancer Histopathological Images
                    - dataset3: Lung Cancer 4 Types Image Dataset
__init__.py        : 配置模块初始化

法则: Python格式 · 路径集中管理 · 环境变量驱动 · 零代码修改部署

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
