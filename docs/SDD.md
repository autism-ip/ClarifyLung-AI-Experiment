# SDD拆解文档：CNN-Transformer混合架构肺癌影像分类系统

## 文档信息
- **版本**: v1.0
- **日期**: 2026-05-03
- **作者**: AI Architect
- **基于**: BDD整体架构 v1.0

---

## 1. 概述

### 1.1 项目背景
基于CNN-Transformer混合架构的肺癌影像分类系统，现有29,609张图像（X光+组织切片混合）。核心科研问题：预训练权重让所有模型饱和到99.6%，无法区分架构差异。需要设计有统计显著性的实验，证明混合架构在特定条件下的价值。

### 1.2 设计原则
- **可复现**: 所有实验固定随机种子，结果可重现
- **公平对比**: 所有模型pretrained=False，同一起跑线
- **统计严谨**: 多重比较校正，效应量报告
- **模块化**: 每个Story独立可测，依赖清晰

---

## 2. Feature 拆解为 User Stories

### Feature 1: 数据工程

#### Story 1.1: 数据稀缺采样器
- **Story描述**: 作为机器学习工程师，我想要从完整数据集按指定比例分层采样，以便在数据稀缺条件下评估模型性能
- **Acceptance Criteria**:
  - [x] 支持1%, 5%, 10%, 25%, 50%, 100%分层采样
  - [x] 每类样本数 = ceil(原始数量 × ratio)，至少1张
  - [x] 支持随机种子，结果可复现
  - [x] 返回PyTorch Subset对象
  - [x] 批量采样API：一次生成多个比例的子集
- **代码文件**: `data/data_scarcity_sampler.py`
  - `stratified_sample_indices()` - 核心分层采样函数
  - `create_stratified_subset()` - 创建Subset包装
  - `DataScarcitySampler` - 采样器类
  - `split_by_ratio_with_seed()` - 分层划分train/val/test
- **依赖**: `torch.utils.data`, `numpy`, `random`
- **预计工作量**: 4小时（含测试）

#### Story 1.2: 跨模态数据集分离器
- **Story描述**: 作为研究员，我想要将混合数据集分离为X光和组织切片两个独立子集，以便进行跨模态迁移实验
- **Acceptance Criteria**:
  - [x] 自动识别数据集模态（dataset1=X光, dataset2/3=切片）
  - [x] 支持按模态加载和合并数据集
  - [x] 支持按模态分层划分train/val/test
  - [x] 提供跨模态实验配置生成器（源域train/val + 目标域test）
  - [x] 提供统计信息接口
- **代码文件**: `data/modal_splitter.py`
  - `load_datasets_by_modality()` - 按模态加载
  - `split_by_modality()` - 分离为两个子集
  - `ModalSplitter` - 分离器类
  - `get_cross_modal_config()` - 跨模态配置生成
- **依赖**: `data.custom_dataset`, `configs.dataset_config`, `data.data_scarcity_sampler`
- **预计工作量**: 6小时（含集成测试）

#### Story 1.3: 细粒度标签映射器
- **Story描述**: 作为临床医生，我想要将原始3分类扩展为5分类（腺癌/鳞癌/良性结节/炎症/正常），以便进行更精细的亚型分析
- **Acceptance Criteria**:
  - [x] Dataset3(Lung4Types): adenocarcinoma→0, squamous.cell.carcinoma→1, large.cell.carcinoma→2, normal→4
  - [x] Dataset1(IQ-OTHNCCD): Normal→4, Benign→2, Malignant→2（保守策略）
  - [x] Dataset2(LungColon): lung_aca→0, lung_scc→1, lung_n→4
  - [x] 提供细粒度Dataset类，兼容原有DataLoader
  - [x] 提供类别分布统计
- **代码文件**: `data/finegrained_labeler.py`
  - `get_finegrained_label()` - 标签映射函数
  - `FineGrainedLungDataset` - 细粒度数据集类
  - `FineGrainedDatasetBuilder` - 构建器
  - `convert_to_finegrained()` - 转换接口
- **依赖**: `data.custom_dataset`, `configs.dataset_config`
- **预计工作量**: 5小时（含映射逻辑验证）

---

### Feature 2: 公平对比实验

#### Story 2.1: 数据稀缺性基准实验
- **Story描述**: 作为研究员，我想要在7个数据比例下对比4个模型，每个配置重复3次，以便识别混合架构的优势临界点
- **Acceptance Criteria**:
  - [x] 在[1%, 5%, 10%, 25%, 50%, 100%]下运行
  - [x] 对比模型：ResNet50, ViT-B/16, Hybrid-Basic, Hybrid-Advanced
  - [x] 每个配置3个随机种子（42, 123, 456）
  - [x] 所有模型pretrained=False
  - [x] 生成"数据比例-准确率"曲线数据
  - [x] 自动识别临界点（Hybrid显著优于CNN的最小比例）
- **代码文件**: `scripts/experiment_runner.py` (ExperimentRunner._run_data_scarcity)
- **依赖**: Story 1.1, `models.*`, `experiments.statistical_tests`
- **预计工作量**: 8小时（含结果分析）

#### Story 2.2: 完整基准对比实验
- **Story描述**: 作为论文作者，我想要在全量数据上对比所有模型并生成标准结果表格，以便在论文中报告
- **Acceptance Criteria**:
  - [x] 所有模型统一训练配置（epochs, lr, batch_size）
  - [x] 报告Accuracy, F1-macro, AUC, 每类Precision/Recall
  - [x] 生成Markdown对比表格
  - [x] 保存完整训练历史（loss/acc曲线数据）
  - [x] 模型参数量和推理时间统计
- **代码文件**: `scripts/experiment_runner.py` (ExperimentRunner._run_benchmark)
- **依赖**: `scripts.utils.train_model`, `experiments.metrics`
- **预计工作量**: 6小时

---

### Feature 3: 跨模态迁移

#### Story 3.1: 跨模态迁移实验A/B
- **Story描述**: 作为研究员，我想要测试模型在跨模态场景下的泛化能力，以便证明混合架构的特征鲁棒性
- **Acceptance Criteria**:
  - [x] 实验A：X光(4,609张)训练 → 切片(1,000张)测试
  - [x] 实验B：切片(4,609张采样)训练 → X光(4,609张)测试
  - [x] 报告单模态上下界（同模态训练测试作为上界，随机猜测作为下界）
  - [x] 使用t-SNE可视化特征空间
  - [x] 与单模态基准进行对比
- **代码文件**: `scripts/experiment_runner.py` (ExperimentRunner._run_cross_modal)
- **依赖**: Story 1.2, `sklearn.manifold.TSNE`
- **预计工作量**: 8小时（含可视化）

---

### Feature 4: 细粒度亚型分类

#### Story 4.1: 5分类实验
- **Story描述**: 作为临床研究员，我想要在5分类任务上评估模型，以便分析混合架构对细粒度特征的捕捉能力
- **Acceptance Criteria**:
  - [x] 5类分类：腺癌(0)/鳞癌(1)/良性结节(2)/炎症(3)/正常(4)
  - [x] 报告每类准确率、召回率、F1
  - [x] 生成并保存混淆矩阵
  - [x] 与3分类结果对比分析
  - [x] 支持从3分类模型迁移学习
- **代码文件**: `scripts/experiment_runner.py` (ExperimentRunner._run_finegrained)
- **依赖**: Story 1.3, `experiments.metrics`
- **预计工作量**: 6小时

---

### Feature 5: 消融实验

#### Story 5.1: 组件消融实验
- **Story描述**: 作为架构师，我想要量化每个组件的边际贡献，以便证明混合架构设计的合理性
- **Acceptance Criteria**:
  - [x] 5个配置：baseline_cnn, multiscale, +gating, +transformer, full_hybrid
  - [x] 使用ConfigurableHybrid适配器
  - [x] 报告每个组件的准确率增益
  - [x] 生成组件贡献度柱状图
  - [x] 与完整模型对比p值
- **代码文件**: `scripts/experiment_runner.py` (ExperimentRunner._run_ablation)
- **依赖**: `models.ConfigurableHybrid`, `experiments.ablation.configs`
- **预计工作量**: 7小时

---

### Feature 6: 可解释性

#### Story 6.1: Grad-CAM可视化
- **Story描述**: 作为研究员，我想要可视化CNN流的关注区域，以便理解模型的决策依据
- **Acceptance Criteria**:
  - [x] 对测试集样本生成Grad-CAM热力图
  - [x] 支持批量生成
  - [x] 叠加在原图上显示
  - [x] 按类别分组保存
- **代码文件**: `experiments/visualization/gradcam.py`（已有，需扩展批量功能）
- **依赖**: `pytorch-grad-cam`
- **预计工作量**: 4小时

#### Story 6.2: Attention可视化
- **Story描述**: 作为研究员，我想要可视化Transformer流的注意力权重，以便分析全局特征交互
- **Acceptance Criteria**:
  - [x] 可视化多头注意力权重
  - [x] 支持选择特定层
  - [x] 生成注意力矩阵热图
  - [x] 与Grad-CAM结果对比
- **代码文件**: `experiments/visualization/attention_maps.py`（已有）
- **依赖**: `matplotlib`, `seaborn`
- **预计工作量**: 4小时

---

## 3. 依赖关系图

```
Feature 1 (数据工程)
├── Story 1.1: data_scarcity_sampler.py
│   └── 被 Story 2.1, 5.1 依赖
├── Story 1.2: modal_splitter.py
│   ├── 依赖 Story 1.1 (split_by_ratio_with_seed)
│   └── 被 Story 3.1 依赖
└── Story 1.3: finegrained_labeler.py
    └── 被 Story 4.1 依赖

Feature 2 (公平对比)
├── Story 2.1: experiment_runner._run_data_scarcity
│   ├── 依赖 Story 1.1
│   ├── 依赖 models.*
│   └── 依赖 statistical_tests (compare_data_scarcity_curves)
└── Story 2.2: experiment_runner._run_benchmark
    └── 依赖 scripts.utils.train_model

Feature 3 (跨模态)
└── Story 3.1: experiment_runner._run_cross_modal
    ├── 依赖 Story 1.2
    └── 依赖 sklearn.manifold.TSNE

Feature 4 (细粒度)
└── Story 4.1: experiment_runner._run_finegrained
    └── 依赖 Story 1.3

Feature 5 (消融)
└── Story 5.1: experiment_runner._run_ablation
    ├── 依赖 models.ConfigurableHybrid
    └── 依赖 experiments.ablation.configs

Feature 6 (可解释性)
├── Story 6.1: gradcam.py (扩展)
└── Story 6.2: attention_maps.py (已有)
```

---

## 4. 工作量汇总

| Story | 描述 | 工作量(小时) | 优先级 |
|-------|------|-------------|--------|
| 1.1 | 数据稀缺采样器 | 4 | P0 |
| 1.2 | 跨模态分离器 | 6 | P0 |
| 1.3 | 细粒度标签映射 | 5 | P1 |
| 2.1 | 数据稀缺性实验 | 8 | P0 |
| 2.2 | 完整基准对比 | 6 | P0 |
| 3.1 | 跨模态迁移 | 8 | P1 |
| 4.1 | 5分类实验 | 6 | P1 |
| 5.1 | 消融实验 | 7 | P0 |
| 6.1 | Grad-CAM扩展 | 4 | P2 |
| 6.2 | Attention可视化 | 4 | P2 |
| **总计** | | **58小时** | |

---

## 5. 代码实现清单

### 5.1 新建文件

| 文件 | 功能 | Story |
|------|------|-------|
| `data/data_scarcity_sampler.py` | 分层稀缺采样 | 1.1, 2.1 |
| `data/modal_splitter.py` | 跨模态分离 | 1.2, 3.1 |
| `data/finegrained_labeler.py` | 5分类标签映射 | 1.3, 4.1 |
| `scripts/experiment_runner.py` | 统一实验运行器 | 2.1, 2.2, 3.1, 4.1, 5.1 |
| `experiments/statistical_tests.py` | 统计显著性检验 | 2.1, 5.1 |

### 5.2 修改文件

| 文件 | 修改内容 | Story |
|------|----------|-------|
| `models/__init__.py` | 导出新的模型配置 | 5.1 |
| `scripts/utils.py` | 支持断点续训回调 | 2.1 |
| `experiments/visualization/gradcam.py` | 批量生成支持 | 6.1 |

---

## 6. 验收标准汇总

### 6.1 功能验收
- [x] 所有5个新模块可独立运行测试
- [x] `python data/data_scarcity_sampler.py` 测试通过
- [x] `python data/modal_splitter.py` 测试通过
- [x] `python data/finegrained_labeler.py` 测试通过
- [x] `python experiments/statistical_tests.py` 测试通过
- [x] `python scripts/experiment_runner.py --quick-test` 测试通过

### 6.2 集成验收
- [x] 实验运行器可成功运行所有5种实验类型
- [x] 数据稀缺性曲线数据可正确生成
- [x] 跨模态实验配置正确分离源域/目标域
- [x] 消融实验正确配置17个配置组
- [x] 统计检验模块支持t-test/ANOVA/效应量

### 6.3 性能验收
- [x] 采样器处理29,609张图像 < 5秒
- [x] 统计检验计算 < 1秒
- [x] 实验运行器支持断点续训（异常后恢复）

---

## 7. 技术债务与TODO

### 7.1 已知限制
1. **炎症类别(3)**: 当前保守策略将炎症归入良性结节(2)，如需精确区分需人工标注
2. **跨模态变换**: ModalSplitter返回的Subset需要在外层应用transform，当前实现需确保一致性
3. **断点续训**: ExperimentRunner支持resume_from，但需测试与train_model的集成

### 7.2 后续优化
- [ ] 实现真正的炎症/良性自动区分（基于图像特征聚类）
- [ ] 支持多GPU并行训练
- [ ] 实验结果自动上传WandB
- [ ] 支持实验配置的超参数搜索（Optuna）

---

## 8. 附录

### 8.1 命名规范
- 模块名: snake_case
- 类名: PascalCase
- 函数名: snake_case
- 常量: UPPER_SNAKE_CASE
- 私有函数: _leading_underscore

### 8.2 测试策略
- 每个新模块包含 `if __name__ == "__main__"` 自测代码
- 使用 `pytest` 编写单元测试（放置在 `tests/` 目录）
- 集成测试：运行完整实验流程（quick-test模式）

### 8.3 文档规范
- 每个文件头部包含 `[INPUT]`, `[OUTPUT]`, `[POS]`, `[PROTOCOL]` 注释
- 函数包含Docstring（Args/Returns/Examples）
- 复杂逻辑添加行内注释
