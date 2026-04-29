# 科训 - X光片分类项目

基于CNN-Transformer混合架构的医学影像分类系统

## 技术栈

Python + PyTorch + torchvision + timm + albumentations

## 项目结构

```
科训/
├── data/              # 数据管理模块
│   ├── __init__.py
│   ├── custom_dataset.py    # 三数据集统一加载 (IQ-OTHNCCD + LungColon + Lung4Types)
│   ├── augmentation.py      # 数据增强 (CutMix/MixUp/RandomErasing)
│   └── visualization.py     # 数据可视化
├── models/            # 模型架构模块
│   ├── __init__.py
│   ├── hybrid_model.py      # CNN-Transformer混合模型 (支持消融开关)
│   ├── configurable_hybrid.py # 消融实验适配器
│   └── components/          # 模型组件
│       ├── feature_extractor.py
│       ├── gating.py
│       ├── transformer.py
│       ├── cross_attention.py
│       └── classification.py
├── training/          # 训练微调模块
│   ├── __init__.py
│   └── trainer.py           # 完整训练流程
│       - 差分学习率 (CNN小LR, Transformer大LR)
│       - 混合精度训练 (AMP)
│       - 早停机制
│       - 检查点保存/恢复
├── experiments/       # 实验模块 (144 tests, 100% pass)
│   ├── __init__.py
│   ├── metrics.py           # EvaluationMetrics, compute_metrics
│   ├── complexity.py        # ModelComplexityAnalyzer
│   ├── benchmark/           # 基准模型对比
│   │   ├── models.py        # create_resnet50/vit/hybrid工厂
│   │   └── benchmarker.py   # ModelBenchmark
│   ├── ablation/           # 消融实验
│   │   ├── configs.py       # AblationConfig, ABLATION_CONFIGS
│   │   └── ablator.py       # AblationStudy
│   ├── cross_validation/    # 交叉验证
│   │   ├── statistical_tests.py  # 统计显著性检验
│   │   └── validator.py      # KFoldCrossValidator
│   └── visualization/       # 可解释性可视化
│       ├── gradcam.py       # GradCAMVisualizer
│       └── attention_maps.py # AttentionVisualizer
├── tests/             # 单元测试模块
│   ├── test_dataset_loading.py  # 数据集加载测试
│   ├── test_download_datasets.py # Kaggle下载脚本测试
│   └── test_model_forward.py    # 模型前向/训练测试
├── scripts/           # 实验执行脚本
│   ├── download_datasets.py       # Kaggle数据集下载器
│   ├── benchmark_experiment.py    # 基准模型对比实验CLI
│   ├── ablation_experiment.py     # 消融实验CLI
│   ├── cross_validation_experiment.py # 交叉验证实验CLI
│   ├── baseline_experiment.py     # 小规模快速验证脚本
│   ├── validate_pipeline.py       # 3步流水线烟雾测试
│   ├── utils.py                   # 实验脚本公共工具
│   │                              #   - set_seed, get_device
│   │                              #   - split_dataset_with_transforms (防数据泄漏划分)
│   │                              #   - train_model (通用训练循环 + AMP + 差分LR)
│   │                              #   - create_quick_test_datasets (200样本快速子集)
│   ├── submit_benchmark.sh        # SLURM: 基准实验提交
│   ├── submit_ablation.sh         # SLURM: 消融实验提交
│   └── submit_crossval.sh         # SLURM: 交叉验证提交
├── configs/           # 配置文件模块
│   ├── __init__.py
│   └── dataset_config.py    # 数据集路径配置 (环境变量驱动)
├── outputs/           # 输出结果模块 (运行时生成)
│   ├── checkpoints/     # 模型检查点
│   ├── logs/           # 训练日志
│   ├── figures/        # 可视化图表
│   └── slurm/          # SLURM作业日志
├── docs/              # 文档资料模块
├── model.py           # 兼容入口，从 models 包 re-export
├── requirements.txt   # Python依赖
├── DEPLOY.md          # 远程服务器部署指南
├── README.md          # 项目说明文档
└── CLAUDE.md          # 项目宪法 (本文件)
```

## 数据集说明

### 三数据集统一标签映射

| 原始标签 | 统一标签 | 编码 |
|---------|---------|------|
| Normal / Normal cases / lung_n | normal | 0 |
| Benign cases / lung_scc / squamous.cell.carcinoma | benign | 1 |
| Malignant cases / lung_aca / adenocarcinoma / large.cell.carcinoma | malignant | 2 |

### 数据集来源

1. **IQ-OTHNCCD Lung Cancer Dataset** - 胸部X光片
2. **Lung and Colon Cancer Histopathological Images** - 肺组织切片
3. **Lung Cancer 4 Types Image Dataset** - 4种肺癌类型

## 快速开始

### 环境安装

```bash
# 创建虚拟环境
conda create -n lung_cancer python=3.10
conda activate lung_cancer

# 安装依赖
pip install -r requirements.txt
```

### 数据准备

```bash
# 方式1: 设置环境变量（推荐，远程部署零代码修改）
export LUNG_DATASET_DIR=/path/to/your/datasets

# 方式2: 直接放置到项目 datasets/ 目录（本地开发）
```

```python
from data.custom_dataset import CustomLungDataset, merge_datasets
from data.augmentation import get_train_augmentation, get_val_augmentation

# 数据集路径自动从 configs/dataset_config.py 解析
# 优先 LUNG_DATASET_DIR 环境变量，回退到项目本地 datasets/
from configs import DATASET_PATHS

dataset1 = CustomLungDataset(DATASET_PATHS['dataset1'], 'dataset1', transform=train_transform)
dataset2 = CustomLungDataset(DATASET_PATHS['dataset2'], 'dataset2', transform=train_transform)
dataset3 = CustomLungDataset(DATASET_PATHS['dataset3'], 'dataset3', transform=train_transform)

# 合并数据集
combined_dataset = merge_datasets(DATASET_PATHS['dataset1'], DATASET_PATHS['dataset2'], DATASET_PATHS['dataset3'])
```

### 模型训练

```python
from model import HybridModel
from training.trainer import Trainer, TrainingConfig
from torch.utils.data import DataLoader

# 创建模型
model = HybridModel(
    num_classes=3,
    model_dim=256,
    nhead=8,
    num_layers=4,
    dropout=0.1
)

# 配置训练参数
config = TrainingConfig(
    num_epochs=100,
    batch_size=32,
    learning_rate=1e-4,      # CNN小学习率
    transformer_lr=5e-4,      # Transformer大学习率
    use_amp=True,             # 混合精度训练
    early_stopping_patience=10
)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 开始训练
trainer = Trainer(model, config, train_loader=train_loader, val_loader=val_loader)
metrics = trainer.fit()
```

### CLI 实验入口

```bash
# 基准模型对比
python scripts/benchmark_experiment.py --epochs 50 --batch-size 32

# 消融实验
python scripts/ablation_experiment.py --epochs 30 --batch-size 32

# 交叉验证
python scripts/cross_validation_experiment.py --folds 5 --epochs 30

# 快速测试模式 (200样本, 1epoch, 秒级验证)
python scripts/benchmark_experiment.py --quick-test
python scripts/ablation_experiment.py --quick-test
python scripts/cross_validation_experiment.py --quick-test
```

### SLURM 远程提交

```bash
# 修改 scripts/submit_*.sh 中的环境配置后提交
sbatch scripts/submit_benchmark.sh    # 24h
sbatch scripts/submit_ablation.sh     # 48h
sbatch scripts/submit_crossval.sh     # 72h

# 快速自检模式 (200样本, 1epoch, 验证环境/数据/GPU)
sbatch --export=QUICK_TEST=1 scripts/submit_benchmark.sh
sbatch --export=QUICK_TEST=1 scripts/submit_ablation.sh
sbatch --export=QUICK_TEST=1 scripts/submit_crossval.sh
```

### 模型评估

```python
# 在测试集上评估
results = trainer.evaluate(test_loader)

# 加载最佳模型进行评估
trainer.load_checkpoint('outputs/checkpoints/best_model.pth')
test_results = trainer.evaluate(test_loader)
```

## 核心特性

### 1. CNN-Transformer混合架构
- **CNN部分**: ResNet50提取多尺度局部特征
- **Transformer部分**: 全局特征建模和长距离依赖
- **门控机制**: 自适应融合CNN和Transformer特征

### 2. 差分学习率策略
- CNN层: 较小学习率 (1e-4) - 微调预训练权重
- Transformer层: 较大学习率 (5e-4) - 从头训练

### 3. 数据增强
- 基础增强: 旋转、翻转、颜色抖动、仿射变换
- 高级增强: CutMix、MixUp、Random Erasing
- 归一化: ImageNet统计

### 4. 训练优化
- 混合精度训练 (AMP) - 加速并减少显存
- 早停机制 - 防止过拟合
- 学习率调度 - Cosine Annealing / One Cycle
- 检查点保存 - 最佳模型自动保存

## 项目状态

### 已完成模块 ✅
- [x] 数据加载与预处理模块 (`data/`)
- [x] 数据增强模块 (`data/augmentation.py`)
- [x] 模型架构模块 (`model.py`, `models/`)
  - [x] HybridModel (完整CNN-Transformer，支持消融开关)
  - [x] ConfigurableHybrid (消融实验适配器)
- [x] 训练流程模块 (`training/trainer.py`)
- [x] 单元测试模块 (`tests/`)
- [x] 实验模块 (`experiments/`) - 144 tests, 100% pass (Windows环境2个temp权限error无关代码)
  - [x] 基准模型对比 (`benchmark/`)
  - [x] 消融实验框架 (`ablation/`)
  - [x] 交叉验证 (`cross_validation/`)
  - [x] 复杂度分析 (`complexity.py`)
  - [x] 可解释性可视化 (`visualization/`)
- [x] 实验脚本统一架构 (benchmark/ablation/crossval 共享 models.HybridModel)
- [x] 公共工具提取 (`scripts/utils.py`)

### 待实现模块 📋
- [ ] 数据探索与预处理实验 (`data/visualization.py` 扩展)
- [ ] 端到端训练脚本 (`scripts/train.py`)

## 实验方案

### 实验一：数据探索与预处理
- 统计三个数据集样本数量（train/val/test）
- 可视化类别分布（柱状图/饼图）
- 样例图像展示（增强前后对比）
- 图像质量检查

### 实验二：基准模型训练
- 超参数搜索：learning_rate [1e-4, 5e-4, 1e-3], batch_size [16, 32, 64], dropout [0.1, 0.2, 0.3]
- 差分学习率策略验证
- 数据增强效果对比

### 实验三：对比与消融实验
- 基准模型：ResNet50, ViT-B/16, Hybrid-Basic, Hybrid-Advanced
- 消融组件：多尺度特征、门控机制、交叉注意力、Transformer层数
- K折交叉验证（5-fold）

### 实验四：可解释性分析
- Grad-CAM++ 热力图（CNN流关注区域）
- Transformer 注意力图（全局上下文）
- 双流交叉注意力可视化

### 实验五：模型复杂度与部署评估
- 参数量和 FLOPs 统计
- 推理速度测试（CPU/GPU）
- 显存占用评估

## 部署指南

### 环境要求
- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+ (推荐使用GPU)
- 16GB+ RAM
- 50GB+ 磁盘空间（数据集）

### 远程服务器部署步骤

```bash
# 1. 克隆项目到服务器
git clone <your-repo-url> /workspace/lung-cancer-classification
cd /workspace/lung-cancer-classification

# 2. 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 下载数据集（根据实际路径配置）
# 配置 data/custom_dataset.py 中的数据集路径

# 5. 运行测试
python -m pytest tests/ -v

# 6. 开始训练
python -c "
from training.trainer import Trainer, TrainingConfig
from model import HybridModel
# ... 加载数据和配置
# trainer.fit()
"
```

### 关键文件说明

| 文件 | 用途 |
|-----|------|
| `model.py` | 主模型定义 (CNN-Transformer Hybrid) |
| `data/custom_dataset.py` | 三数据集统一加载器 (核心) |
| `data/augmentation.py` | 数据增强管道 |
| `training/trainer.py` | 完整训练流程 (差分LR/AMP/早停) |
| `models/hybrid_model.py` | CNN-Transformer混合模型 (支持消融开关) |
| `models/configurable_hybrid.py` | 消融实验适配器 (ConfigurableHybrid) |
| `experiments/metrics.py` | 评估指标计算 |
| `experiments/benchmark/models.py` | 基准模型工厂 (ResNet/ViT/HybridBasic) |
| `experiments/ablation/configs.py` | 消融配置定义 |
| `experiments/cross_validation/validator.py` | K折交叉验证 |
| `scripts/benchmark_experiment.py` | 基准对比实验CLI |
| `scripts/ablation_experiment.py` | 消融实验CLI |
| `scripts/cross_validation_experiment.py` | 交叉验证CLI |
| `scripts/utils.py` | 实验脚本公共工具 (set_seed, get_device, split_dataset_with_transforms, train_model, create_quick_test_datasets) |
| `scripts/submit_benchmark.sh` | SLURM: 基准实验提交 |
| `scripts/submit_ablation.sh` | SLURM: 消融实验提交 |
| `scripts/submit_crossval.sh` | SLURM: 交叉验证提交 |
| `configs/dataset_config.py` | 数据集路径配置 (环境变量驱动) |
| `tests/` | 单元测试套件 |
| `requirements.txt` | Python依赖列表 |
| `DEPLOY.md` | 远程服务器部署指南 |
| `README.md` | 项目说明与复刻指南 |

## 引用

本项目基于以下工作：
- ResNet: "Deep Residual Learning for Image Recognition"
- Transformer: "Attention Is All You Need"
- CutMix: "CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features"
- MixUp: "mixup: Beyond Empirical Risk Minimization"

---

**法则**: 模块化 · 可复现 · 可扩展 · 版本精确

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
