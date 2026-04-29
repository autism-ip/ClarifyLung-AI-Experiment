# 科训 - CNN-Transformer肺癌X光分类系统

基于CNN-Transformer混合架构的医学影像分类系统

## 项目简介

本项目实现了一个CNN-Transformer混合深度学习模型，用于肺癌X光片的自动分类。模型结合了CNN的局部特征提取能力和Transformer的全局建模能力，在三个公开数据集上进行了统一训练和评估。

### 核心特性

- **混合架构**: CNN (ResNet50) + Transformer
- **多数据集**: 整合3个公开肺癌数据集
- **统一标签**: normal(0) / benign(1) / malignant(2)
- **差分学习率**: CNN小LR，Transformer大LR
- **混合精度**: AMP加速训练

---

## 5分钟上手检查清单

> 克隆项目后，按以下顺序验证环境，确认无误再运行全量实验。

| 步骤 | 命令 | 预期结果 | 失败排查 |
|------|------|---------|---------|
| 1 | `conda create -n lung_cancer python=3.10` | 环境创建成功 | 检查conda是否安装 |
| 2 | `pip install -r requirements.txt` | 依赖安装无报错 | 见下方 **GPU安装顺序警告** |
| 3 | `python scripts/validate_pipeline.py` | 3步全部 PASSED | 见下方 **验证失败排查** |
| 4 | `python -m pytest tests/ -v` | 46 passed, 9 skipped | Windows temp权限错误见FAQ |
| 5 | `python scripts/benchmark_experiment.py --quick-test` | 1epoch内完成 | 数据路径错误见FAQ |

**GPU安装顺序警告（重要）**：

```bash
# GPU服务器必须先装CUDA版PyTorch，再装requirements.txt
# 否则requirements.txt中的 torch>=2.0.0 会安装CPU版本
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

**验证失败排查**：
- `Step 1 FAILED` → 数据集路径错误，检查 `LUNG_DATASET_DIR` 或 `configs/dataset_config.py`
- `Step 2 FAILED` → 模型依赖缺失，检查 `timm` 是否安装
- `Step 3 FAILED` → CUDA/驱动问题，或内存不足，尝试减小batch_size

---

## 文件目录树

```
ClarifyLung-AI-Experiment/
├── configs/                         # 配置文件模块
│   ├── __init__.py
│   ├── dataset_config.py            # 数据集路径配置 (环境变量驱动)
│   └── CLAUDE.md
├── data/                            # 数据管理模块
│   ├── __init__.py
│   ├── custom_dataset.py            # 三数据集统一加载器 (核心)
│   ├── augmentation.py              # 数据增强 (CutMix/MixUp/RandomErasing)
│   ├── visualization.py             # 数据可视化
│   └── CLAUDE.md
├── models/                          # 模型架构模块
│   ├── __init__.py
│   ├── hybrid_model.py              # CNN-Transformer混合模型 (主模型)
│   ├── components/                  # 模型组件
│   │   ├── feature_extractor.py     # 多尺度特征提取器
│   │   ├── gating.py                # 门控机制 (SE/Sigmoid)
│   │   ├── transformer.py           # Transformer编码器
│   │   ├── cross_attention.py       # 交叉注意力模块
│   │   └── classification.py        # 分类头
│   └── CLAUDE.md
├── training/                        # 训练微调模块
│   ├── __init__.py
│   ├── trainer.py                   # 完整训练流程 (差分LR/AMP/早停/检查点)
│   └── CLAUDE.md
├── experiments/                     # 实验分析模块 (144 tests, 100% pass)
│   ├── __init__.py
│   ├── metrics.py                   # 评估指标 (Accuracy/F1/AUC/Confusion)
│   ├── complexity.py                # 模型复杂度分析 (参数量/FLOPs)
│   ├── benchmark/                   # 基准模型对比
│   │   ├── models.py                # ResNet50/ViT/Hybrid模型工厂
│   │   ├── benchmarker.py           # 基准测试执行器
│   │   └── __init__.py
│   ├── ablation/                    # 消融实验
│   │   ├── configs.py               # 消融配置 (AblationConfig)
│   │   ├── ablator.py               # 消融实验运行器
│   │   └── __init__.py
│   ├── cross_validation/            # 交叉验证
│   │   ├── validator.py             # K折分层交叉验证
│   │   ├── statistical_tests.py     # 统计显著性检验
│   │   └── __init__.py
│   ├── visualization/               # 可解释性可视化
│   │   ├── gradcam.py               # Grad-CAM热力图
│   │   ├── attention_maps.py        # 注意力图可视化
│   │   ├── training_curves.py       # 训练曲线
│   │   ├── confusion_matrix.py      # 混淆矩阵
│   │   ├── class_distribution.py    # 类别分布
│   │   ├── model_comparison.py      # 模型对比图
│   │   └── __init__.py
│   └── tests/                       # 实验模块单元测试
│       ├── test_metrics.py
│       ├── test_complexity.py
│       ├── test_benchmark_models.py
│       ├── test_benchmarker.py
│       ├── test_ablation_configs.py
│       ├── test_ablator.py
│       ├── test_validator.py
│       ├── test_statistical_tests.py
│       ├── test_gradcam.py
│       ├── test_attention_maps.py
│       ├── test_training_curves.py
│       ├── test_confusion_matrix.py
│       ├── test_class_distribution.py
│       ├── test_model_comparison.py
│       └── __init__.py
├── scripts/                         # 实验执行脚本
│   ├── __init__.py
│   ├── download_datasets.py         # Kaggle数据集下载
│   ├── benchmark_experiment.py      # 基准模型对比实验 (CLI入口)
│   ├── ablation_experiment.py       # 消融实验 (CLI入口)
│   ├── cross_validation_experiment.py # 交叉验证实验 (CLI入口)
│   ├── baseline_experiment.py       # 小规模快速验证脚本
│   ├── validate_pipeline.py         # 3步流水线烟雾测试
│   ├── submit_benchmark.sh          # SLURM: 基准实验提交
│   ├── submit_ablation.sh           # SLURM: 消融实验提交
│   ├── submit_crossval.sh           # SLURM: 交叉验证提交
│   ├── utils.py                     # 实验脚本公共工具 (set_seed, get_device, split_dataset_with_transforms, train_model, create_quick_test_datasets)
│   └── CLAUDE.md
├── tests/                           # 单元测试模块
│   ├── test_dataset_loading.py      # 数据集加载测试
│   ├── test_model_forward.py        # 模型前向/反向测试
│   ├── test_download_datasets.py    # Kaggle下载脚本测试
│   └── CLAUDE.md
├── outputs/                         # 输出结果 (运行时生成)
│   ├── checkpoints/                 # 模型检查点
│   ├── logs/                        # 训练日志
│   ├── figures/                     # 可视化图表
│   └── slurm/                       # SLURM日志
├── docs/                            # 文档资料
├── model.py                         # 兼容入口 (从models包re-export)
├── requirements.txt                 # Python依赖
├── DEPLOY.md                        # 远程服务器部署指南
├── CLAUDE.md                        # 项目架构文档 (L1)
└── README.md                        # 本文件
```

---

## 项目架构

### 模块依赖关系

```
configs/ (dataset_config.py)
    ↑
data/ (custom_dataset.py, augmentation.py)  ← 依赖 configs.DATASET_PATHS
    ↑
models/ (hybrid_model.py + components/)     ← 纯模型定义，无数据依赖
    ↑
training/ (trainer.py)                      ← 依赖 models + data
    ↑
experiments/                                ← 依赖 models + training + data
    ├── benchmark/      ← 依赖 models
    ├── ablation/       ← 依赖 models
    ├── cross_validation/ ← 依赖 models + data
    ├── visualization/  ← 依赖 models
    └── tests/          ← 依赖所有模块
    ↑
scripts/                                    ← 依赖 experiments + training + data
```

### 模块职责

| 模块 | 核心职责 | 对外暴露 |
|------|---------|---------|
| `configs/` | 集中管理数据集路径，支持环境变量 `LUNG_DATASET_DIR` 零代码修改部署 | `DATASET_PATHS`, `DATASET_INFO` |
| `data/` | 三数据集统一加载、标签映射、数据增强、数据可视化 | `CustomLungDataset`, `merge_datasets`, `get_train_augmentation` |
| `models/` | CNN-Transformer混合模型架构及组件 | `HybridModel` |
| `training/` | 完整训练流程：差分学习率、AMP、早停、检查点 | `Trainer`, `TrainingConfig` |
| `experiments/` | 基准对比、消融实验、交叉验证、可解释性可视化、复杂度分析 | `compute_metrics`, `ModelBenchmark`, `AblationStudy`, `KFoldCrossValidator` |
| `scripts/` | 可独立运行的实验CLI入口 + SLURM批作业脚本 | `benchmark_experiment.py`, `ablation_experiment.py`, `cross_validation_experiment.py` |
| `tests/` | 核心功能单元测试 | pytest 测试套件 |

---

## 环境安装与数据准备

### 1. 环境安装

```bash
# 创建虚拟环境
conda create -n lung_cancer python=3.10
conda activate lung_cancer

# 安装依赖
pip install -r requirements.txt
```

**依赖兼容性注意**：
- `numpy<2` (锁定 1.26.4，避免与 scipy 1.13 ABI 不兼容)
- `opencv-python<4.10` (避免强制依赖 numpy>=2)

### 2. 数据准备

**方式A：环境变量（推荐，远程部署零代码修改）**

```bash
export LUNG_DATASET_DIR=/path/to/your/datasets
```

**方式B：放置到项目本地 `datasets/` 目录**

```
datasets/
├── IQ-OTHNCCD/
│   └── Augmented IQ-OTHNCCD lung cancer dataset/
│       ├── Normal cases/
│       ├── Malignant cases/
│       └── Benign cases/
├── LungColon/
│   └── lung_colon_image_set/
│       └── lung_image_sets/
│           ├── lung_n/
│           ├── lung_aca/
│           └── lung_scc/
└── Lung4Types/
    └── Data/
        ├── train/
        ├── valid/
        └── test/
```

**方式C：Kaggle自动下载**

```bash
# 配置Kaggle凭证 ~/.kaggle/kaggle.json
python scripts/download_datasets.py
```

---

## 快速开始

### 第一步：验证环境（强烈推荐）

运行烟雾测试，确认数据加载、模型创建、训练流程全部正常：

```bash
python scripts/validate_pipeline.py
```

**预期输出**：
```
[OK] Step 1 PASSED: Data loading works correctly
[OK] Step 2 PASSED: Model creation and forward pass work correctly
[OK] Step 3 PASSED: Training pipeline works!
ALL CHECKS PASSED - Pipeline is ready for full experiments!
```

此脚本使用小模型配置（model_dim=128, nhead=4, num_layers=2）和32样本子集，
在CPU上约30秒完成，在GPU上约10秒完成。适合作为每次实验前的健康检查。

---

### 方式一：Python API 编程式调用

```python
from model import HybridModel
from training.trainer import Trainer, TrainingConfig
from data.augmentation import get_train_augmentation, get_val_augmentation
from configs import DATASET_PATHS
from scripts.utils import split_dataset_with_transforms
from torch.utils.data import DataLoader

# 1. 创建模型
model = HybridModel(num_classes=3, model_dim=256, nhead=8, num_layers=4, dropout=0.1)

# 2. 加载数据 (防泄漏划分：train/val/test 各自使用正确的 transform)
train_transform = get_train_augmentation(224)
val_transform = get_val_augmentation(224)
train_ds, val_ds, test_ds = split_dataset_with_transforms(
    DATASET_PATHS['dataset1'],
    DATASET_PATHS['dataset2'],
    DATASET_PATHS['dataset3'],
    train_transform=train_transform,
    val_transform=val_transform,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42,
)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4)

# 3. 配置训练
config = TrainingConfig(
    num_epochs=100,
    batch_size=32,
    learning_rate=1e-4,       # CNN学习率
    transformer_lr=5e-4,      # Transformer学习率
    use_amp=True,             # 混合精度训练
    early_stopping_patience=10,
    output_dir="outputs/checkpoints"
)

# 4. 训练
trainer = Trainer(model, config, train_loader, val_loader, test_loader)
metrics = trainer.fit()

# 5. 评估
results = trainer.evaluate(test_loader)
```

### 方式二：本地 CLI 命令行调用

#### 数据集下载

```bash
python scripts/download_datasets.py                    # 下载所有数据集
python scripts/download_datasets.py --dataset 0       # 只下载第1个数据集
python scripts/download_datasets.py --validate-only   # 仅验证已下载数据
python scripts/download_datasets.py --cleanup         # 清理临时压缩包
```

#### 基准模型对比实验

```bash
python scripts/benchmark_experiment.py \
  --epochs 50 \
  --batch-size 32 \
  --lr 1e-4 \
  --transformer-lr 5e-4 \
  --output-dir outputs/benchmark
```

#### 消融实验

```bash
python scripts/ablation_experiment.py \
  --epochs 30 \
  --batch-size 32 \
  --lr 1e-4 \
  --transformer-lr 5e-4 \
  --output-dir outputs/ablation
```

#### 交叉验证实验

```bash
python scripts/cross_validation_experiment.py \
  --folds 5 \
  --epochs 30 \
  --batch-size 32 \
  --lr 1e-4 \
  --output-dir outputs/cross_validation
```

#### CLI 通用参数表

| 参数 | 说明 | 默认值 | 适用脚本 |
|------|------|--------|---------|
| `--dataset1` | Dataset1 (IQ-OTHNCCD) 路径 | `configs` 配置 | 全部 |
| `--dataset2` | Dataset2 (LungColon) 路径 | `configs` 配置 | 全部 |
| `--dataset3` | Dataset3 (Lung4Types) 路径 | `configs` 配置 | 全部 |
| `--epochs` | 训练轮数 | 50/30/30 | benchmark/ablation/crossval |
| `--batch-size` | 批大小 | 32 | 全部 |
| `--lr` | CNN学习率 | 1e-4 | 全部 |
| `--transformer-lr` | Transformer学习率 | 5e-4 | benchmark/ablation |
| `--seed` | 随机种子 | 42 | 全部 |
| `--output-dir` | 输出目录 | `outputs/*` | 全部 |
| `--no-checkpoint` | 不保存模型权重 | False | 全部 |
| `--no-plot` | 不生成图表 | False | benchmark |
| `--folds` | K折数 | 5 | crossval |
| `--quick-test` | 快速测试模式 (200样本, 1epoch) | False | 全部 |

### 方式三：远程服务器 SLURM 批作业提交

#### 前置配置

编辑对应的 `scripts/submit_*.sh` 文件，修改以下部分为你的集群环境：

```bash
# =============================================================================
# 环境配置（根据实际集群调整）
# =============================================================================
module load anaconda/2024.01    # 加载conda模块（如需要）
module load cuda/11.8           # 加载CUDA模块（如需要）

# 激活虚拟环境
conda activate lung_cancer

# 数据集根目录（零代码修改部署）
export LUNG_DATASET_DIR=/path/to/your/datasets
```

#### 提交基准模型对比实验

```bash
sbatch scripts/submit_benchmark.sh
```

**脚本配置** (`scripts/submit_benchmark.sh`):
- 分区: `gpu`
- GPU: 1块
- CPU: 8核
- 内存: 32GB
- 时限: 24小时
- 特殊: 添加 `--no-plot` (集群无图形界面)

#### 提交消融实验

```bash
sbatch scripts/submit_ablation.sh
```

**脚本配置** (`scripts/submit_ablation.sh`):
- 分区: `gpu`
- GPU: 1块
- CPU: 8核
- 内存: 32GB
- 时限: 48小时

#### 提交交叉验证实验

```bash
sbatch scripts/submit_crossval.sh
```

**脚本配置** (`scripts/submit_crossval.sh`):
- 分区: `gpu`
- GPU: 1块
- CPU: 8核
- 内存: 32GB
- 时限: 72小时

#### SLURM 作业监控

```bash
squeue -u $USER                    # 查看作业队列
scontrol show job <job_id>         # 查看作业详情
tail -f outputs/slurm/benchmark_<job_id>.out   # 实时查看输出
scancel <job_id>                   # 取消作业
```

#### SLURM 快速自检模式

提交全量实验前，先用快速模式验证环境/数据/GPU：

```bash
sbatch --export=QUICK_TEST=1 scripts/submit_benchmark.sh
sbatch --export=QUICK_TEST=1 scripts/submit_ablation.sh
sbatch --export=QUICK_TEST=1 scripts/submit_crossval.sh
```

> 脚本内部使用 `${QUICK_TEST:-0}`，优先读取 `sbatch --export` 传入的环境变量，
> 无需修改脚本文件即可切换模式。

---

## 实验复刻指南

### 实验一：数据探索与预处理

**相关文件**：
- `data/custom_dataset.py` — 数据集加载与标签映射
- `data/visualization.py` — 数据可视化
- `configs/dataset_config.py` — 路径配置

**复刻步骤**：

```python
from data.custom_dataset import CustomLungDataset, merge_datasets
from configs import DATASET_PATHS

# 加载单个数据集查看分布
ds1 = CustomLungDataset(DATASET_PATHS['dataset1'], 'dataset1')
print(f"Dataset1: {len(ds1)} samples")
print(f"Class distribution: {ds1.get_class_distribution()}")

# 合并三数据集
combined = merge_datasets(
    DATASET_PATHS['dataset1'],
    DATASET_PATHS['dataset2'],
    DATASET_PATHS['dataset3']
)
print(f"Combined: {len(combined)} samples")
```

**预期输出**：三数据集样本总数、各类别数量统计

---

### 实验二：基准模型对比

**相关文件**：
- `scripts/benchmark_experiment.py` — CLI入口
- `experiments/benchmark/models.py` — 模型工厂
- `experiments/benchmark/benchmarker.py` — 执行器
- `scripts/submit_benchmark.sh` — SLURM提交

**本地复刻**：

```bash
python scripts/benchmark_experiment.py \
  --epochs 50 \
  --batch-size 32 \
  --output-dir outputs/benchmark
```

**远程复刻**：

```bash
# 修改 scripts/submit_benchmark.sh 中的数据集路径
sbatch scripts/submit_benchmark.sh
```

**对比模型**：

| 模型 | 架构 | 预训练 |
|------|------|--------|
| ResNet50 | 纯CNN | ImageNet |
| ViT-B/16 | 纯Transformer | ImageNet |
| Hybrid-Basic | CNN+Transformer (基础) | CNN预训练 |
| Hybrid-Advanced | CNN+Transformer (完整) | CNN预训练 |

**预期输出**：
- `outputs/benchmark/benchmark_results.json` — 详细结果
- `outputs/benchmark/benchmark_table.md` — Markdown对比表格
- `outputs/benchmark/model_comparison.png` — 对比柱状图
- `outputs/benchmark/*_best.pth` — 各模型最佳权重

---

### 实验三：消融实验

**相关文件**：
- `scripts/ablation_experiment.py` — CLI入口
- `experiments/ablation/configs.py` — AblationConfig定义
- `experiments/ablation/ablator.py` — 消融执行器
- `scripts/submit_ablation.sh` — SLURM提交

**本地复刻**：

```bash
python scripts/ablation_experiment.py \
  --epochs 30 \
  --batch-size 32 \
  --output-dir outputs/ablation
```

**远程复刻**：

```bash
sbatch scripts/submit_ablation.sh
```

**消融配置**（渐进式叠加，5种配置）：

| Config | multi_scale | gate | transformer | cross_attention | 描述 |
|--------|-------------|------|-------------|-----------------|------|
| baseline_cnn | - | - | - | - | 仅CNN基线 |
| multiscale_only | + | - | - | - | +多尺度特征 |
| gating_added | + | SE | - | - | +SE门控 |
| transformer_added | + | SE | + | - | +Transformer |
| full_hybrid | + | SE | + | + | +交叉注意力 |

> 另见 `experiments/ablation/configs.py` 中定义的 `ABLATION_CONFIGS`（6种组件移除式配置），
> 可用于独立分析各组件贡献度。

**预期输出**：
- `outputs/ablation/ablation_results.json` — 详细结果
- `outputs/ablation/ablation_table.md` — Markdown表格
- `outputs/ablation/*_best.pth` — 各配置最佳权重

---

### 实验四：交叉验证

**相关文件**：
- `scripts/cross_validation_experiment.py` — CLI入口
- `experiments/cross_validation/validator.py` — K折验证器
- `experiments/cross_validation/statistical_tests.py` — 统计检验
- `scripts/submit_crossval.sh` — SLURM提交

**本地复刻**：

```bash
python scripts/cross_validation_experiment.py \
  --folds 5 \
  --epochs 30 \
  --batch-size 32 \
  --output-dir outputs/cross_validation
```

**远程复刻**：

```bash
sbatch scripts/submit_crossval.sh
```

**统计指标**：
- 各折 Accuracy, Macro F1, AUC-ROC
- 汇总统计 (mean ± std)
- 95% 置信区间
- Shapiro-Wilk 正态性检验
- One-sample t-test

**预期输出**：
- `outputs/cross_validation/cross_validation_results.json` — 各折结果
- `outputs/cross_validation/cross_validation_report.md` — 统计报告
- `outputs/cross_validation/cross_validation_curves.png` — 训练曲线
- `outputs/cross_validation/fold_*_best.pth` — 各折最佳权重

---

### 实验五：可视化生成流程

本项目提供 **7 种可视化类型**，覆盖训练过程监控、模型对比、可解释性分析全流程。
所有可视化在无头服务器（SLURM集群）上均可正常生成，后端已强制使用 `Agg`。

#### 自动生成的可视化（实验脚本运行时自动产出）

| 实验脚本 | 自动生成的图表 | 输出路径 | 说明 |
|---------|--------------|---------|------|
| `benchmark_experiment.py` | 模型对比柱状图 | `outputs/benchmark/model_comparison.png` | 各模型 accuracy/f1 对比 |
| `benchmark_experiment.py` | 训练曲线 | `outputs/benchmark/training_curves.png` | 各模型 loss/acc 曲线 |
| `cross_validation_experiment.py` | 交叉验证训练曲线 | `outputs/cross_validation/cross_validation_curves.png` | 各折平均 loss/acc |

#### 手动调用的可视化（独立代码片段）

**1. 训练曲线（Training Curves）**

```python
from experiments.visualization import plot_training_curves

metrics = {
    "train_loss": [1.0, 0.8, 0.6, 0.5, 0.4],
    "val_loss": [1.1, 0.9, 0.75, 0.65, 0.55],
    "train_acc": [0.5, 0.65, 0.78, 0.85, 0.90],
    "val_acc": [0.45, 0.60, 0.72, 0.80, 0.85],
}
fig = plot_training_curves(metrics, save_path="outputs/figures/training_curves.png")
```

**2. 混淆矩阵（Confusion Matrix）**

```python
from experiments.visualization import plot_confusion_matrix
import numpy as np

y_true = np.array([0,0,0,1,1,1,2,2,2])
y_pred = np.array([0,0,1,1,1,2,2,2,2])
fig = plot_confusion_matrix(
    y_true, y_pred,
    class_names=["normal", "benign", "malignant"],
    normalize=True,
    save_path="outputs/figures/confusion_matrix.png"
)
```

**3. 类别分布图（Class Distribution）**

```python
from experiments.visualization import plot_class_distribution, plot_pie_chart

# 柱状图：展示各数据集的类别分布
distribution = {
    "IQ-OTHNCCD": {0: 1208, 1: 1200, 2: 1201},
    "LungColon": {0: 5000, 1: 5000, 2: 5000},
    "Lung4Types": {0: 3000, 1: 3000, 2: 3000},
}
fig = plot_class_distribution(
    distribution,
    class_names=["normal", "benign", "malignant"],
    save_path="outputs/figures/class_distribution.png"
)

# 饼图：单数据集类别占比
fig = plot_pie_chart(
    [1208, 1200, 1201],                           # 各类别样本数列表
    labels=["normal", "benign", "malignant"],      # 类别名称列表
    title="Class Distribution",
    save_path="outputs/figures/class_pie_chart.png"
)
```

**4. 模型对比图（Model Comparison）**

```python
from experiments.visualization import plot_model_comparison

results = {
    "ResNet50": {"accuracy": 0.82, "macro_f1": 0.80},
    "ViT": {"accuracy": 0.85, "macro_f1": 0.83},
    "Hybrid": {"accuracy": 0.91, "macro_f1": 0.90},
}
fig = plot_model_comparison(
    results,
    metrics=["accuracy", "macro_f1"],
    save_path="outputs/figures/model_comparison.png"
)
```

**5. Grad-CAM 热力图（CNN可解释性）**

```python
from experiments.visualization import GradCAMVisualizer
import torch

# 加载模型并切换到评估模式
model = HybridModel(num_classes=3)
model.load_state_dict(torch.load("outputs/checkpoints/best_model.pth"))
model.eval()

# 准备单张输入图像 (1, 3, 224, 224)
input_tensor = torch.randn(1, 3, 224, 224)

# 生成目标类别的热力图
visualizer = GradCAMVisualizer(model)
heatmap = visualizer.generate_heatmap(input_tensor, target_category=2)

# 叠加到原图
from experiments.visualization import overlay_heatmap
overlay_img = overlay_heatmap(input_tensor, heatmap, alpha=0.5)

# 保存
import cv2
cv2.imwrite("outputs/figures/gradcam_malignant.png", overlay_img)
```

**6. Transformer 注意力图**

```python
from experiments.visualization import AttentionVisualizer

visualizer = AttentionVisualizer(model)

# 检查模型是否包含注意力层
if visualizer.has_attention():
    attention_weights = visualizer.extract_attention(input_tensor)
    fig = visualizer.visualize_attention(
        attention_weights[0],
        save_path="outputs/figures/attention_map.png"
    )
```

#### 可视化完整工作流示例

```bash
# Step 1: 运行基准实验（自动生成对比图和训练曲线）
python scripts/benchmark_experiment.py --epochs 50 --output-dir outputs/benchmark

# Step 2: 运行交叉验证（自动生成平均训练曲线）
python scripts/cross_validation_experiment.py --folds 5 --output-dir outputs/cross_validation

# Step 3: 生成混淆矩阵（加载最佳模型在测试集上推理后）
python -c "
from experiments.visualization import plot_confusion_matrix
import numpy as np
# 替换为实际推理结果
y_true = np.load('outputs/benchmark/y_true.npy')
y_pred = np.load('outputs/benchmark/y_pred.npy')
plot_confusion_matrix(y_true, y_pred, class_names=['normal','benign','malignant'],
                      save_path='outputs/figures/final_confusion_matrix.png')
"

# Step 4: 查看所有图表
ls outputs/benchmark/*.png outputs/cross_validation/*.png outputs/figures/*.png
```

#### 常见问题

**Q: SLURM 集群上图表生成失败？**

已在 `experiments/visualization/__init__.py` 顶部强制设置 `matplotlib.use('Agg')`，
无需 DISPLAY 环境即可保存图片。如果仍报错，检查 `matplotlib` 版本：

```bash
python -c "import matplotlib; print(matplotlib.__version__)"  # 应 >=3.7.0
```

**Q: 如何调整图表尺寸/DPI？**

所有 `plot_*` 函数返回 `matplotlib.figure.Figure` 对象，可在保存前调整：

```python
fig = plot_training_curves(metrics)
fig.set_size_inches(16, 9)
fig.savefig("outputs/figures/high_res.png", dpi=300)
```

---

### 实验六：可解释性分析

**相关文件**：
- `experiments/visualization/gradcam.py` — Grad-CAM热力图
- `experiments/visualization/attention_maps.py` — 注意力图

**完整工作流**：

```python
import torch
from model import HybridModel
from experiments.visualization import GradCAMVisualizer, AttentionVisualizer, overlay_heatmap
from torchvision import transforms
from PIL import Image

# 1. 加载训练好的模型
model = HybridModel(num_classes=3)
model.load_state_dict(torch.load("outputs/checkpoints/best_model.pth", map_location="cpu"))
model.eval()

# 2. 加载并预处理单张图像
img_path = "datasets/IQ-OTHNCCD/Normal cases/normal_001.png"
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
img_tensor = transform(Image.open(img_path).convert("RGB")).unsqueeze(0)

# 3. Grad-CAM：查看CNN关注的区域
gcam = GradCAMVisualizer(model)
heatmap = gcam.generate_heatmap(img_tensor, target_category=0)  # target_category=0 为 normal
overlay = overlay_heatmap(heatmap, img_tensor.squeeze(0).permute(1, 2, 0).numpy(), alpha=0.5)
# overlay 为 numpy uint8 数组，可用 cv2.imwrite 保存

# 4. Attention Map：查看Transformer关注的token
attn = AttentionVisualizer(model)
if attn.has_attention():
    weights = attn.extract_attention(img_tensor)
    fig = attn.visualize_attention(weights[0], save_path="outputs/figures/attention.png")
```

**关键参数说明**：
- `target_category`: 目标类别索引 (0=normal, 1=benign, 2=malignant)
- `alpha`: 热力图叠加透明度 (0.0-1.0)
- `weights[0]`: 取第一层注意力头的权重

---

### 实验七：模型复杂度评估

**相关文件**：
- `experiments/complexity.py` — ModelComplexityAnalyzer

**使用示例**：

```python
from experiments.complexity import ModelComplexityAnalyzer

analyzer = ModelComplexityAnalyzer(model, input_size=(1, 3, 224, 224))
stats = analyzer.analyze()
print(f"Parameters: {stats['total_params']:,}")
print(f"FLOPs: {stats['total_flops']:,}")
```

---

## 测试

```bash
# 运行所有测试
python -m pytest tests/ -v
python -m pytest experiments/tests/ -v

# 运行特定测试
python -m pytest tests/test_model_forward.py -v
python -m pytest tests/test_dataset_loading.py -v

# 生成覆盖率报告
python -m pytest tests/ --cov=. --cov-report=html
```

**测试状态**：
- `tests/` — 46 passed, 9 skipped
- `experiments/tests/` — 144 passed

---

## 输出目录结构

实验运行后，`outputs/` 目录结构如下：

```
outputs/
├── checkpoints/                    # 模型检查点 (.pth)
│   ├── best_model.pth              # 最佳模型权重
│   └── latest_model.pth            # 最新模型权重
├── logs/                           # TensorBoard 日志
│   └── experiment_YYYYMMDD_HHMMSS/
│       ├── events.out.tfevents.*   # 标量曲线
│       └── hparams.yaml            # 超参数记录
├── figures/                        # 可视化图表
│   ├── training_curves.png         # loss/acc 曲线
│   ├── confusion_matrix.png        # 混淆矩阵
│   └── *.png                       # 其他实验图表
├── benchmark/                      # 基准实验输出
│   ├── benchmark_results.json      # 原始数据
│   ├── benchmark_table.md          # Markdown 对比表
│   ├── model_comparison.png        # 柱状图
│   └── *_best.pth                  # 各模型最佳权重
├── ablation/                       # 消融实验输出
│   ├── ablation_results.json
│   ├── ablation_table.md
│   └── *_best.pth
├── cross_validation/               # 交叉验证输出
│   ├── cross_validation_results.json   # 各折结果
│   ├── cross_validation_report.md      # 统计报告
│   ├── cross_validation_curves.png     # 训练曲线
│   └── fold_*_best.pth                 # 各折权重
└── slurm/                          # SLURM 作业日志
    ├── benchmark_12345.out
    ├── benchmark_12345.err
    └── ...
```

**结果解读**：
- `.json` 文件包含完整数值结果，可用于二次分析
- `.md` 表格可直接粘贴到论文/报告中
- `.pth` 权重可用 `torch.load()` 加载继续训练或推理

---

## 常见问题 (FAQ)

### Q1: `CUDA out of memory` 怎么解决？

```bash
# 方案1: 减小 batch_size
python scripts/benchmark_experiment.py --batch-size 16

# 方案2: 减小模型尺寸 (修改脚本中的 model_dim 和 num_layers)
# model_dim=128, num_layers=2 可在 8GB 显存运行

# 方案3: 禁用 AMP (极少数旧GPU不支持)
# 在 TrainingConfig 中设置 use_amp=False
```

### Q2: `Dataset not found` 或路径错误？

```bash
# 检查环境变量是否设置
export LUNG_DATASET_DIR=/absolute/path/to/datasets
python -c "from configs import validate_paths; validate_paths()"

# 或检查 configs/dataset_config.py 中的默认路径是否与你的实际路径匹配
```

### Q3: Windows 上运行测试出现 `PermissionError: [WinError 5]`？

这是 Windows 临时目录权限问题，与代码逻辑无关。不影响实际训练。
解决方案：以管理员身份运行终端，或设置 `TMPDIR=C:\temp` 环境变量。

### Q4: 为什么 GPU 服务器上 PyTorch 显示 `CUDA available: False`？

**90%的原因是安装顺序错误**。如果先执行了 `pip install -r requirements.txt`，
其中的 `torch>=2.0.0` 会安装 CPU 版本覆盖你之前装的 CUDA 版本。

正确顺序：
```bash
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### Q5: SLURM 作业一直处于 `PENDING` 状态？

```bash
# 查看排队原因
squeue -u $USER -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"
# %R 列显示原因：Resources (等待GPU), Priority (优先级低), QOSMaxCpuPerUserLimit (配额)

# 如果是配额问题，先用 quick-test 模式验证脚本正确性
sbatch --export=QUICK_TEST=1 scripts/submit_benchmark.sh
```

### Q6: 如何在不修改脚本的情况下调整实验参数？

所有 submit 脚本都支持通过 `sbatch --export` 传入覆盖值：

```bash
sbatch --export=EPOCHS=10,BATCH_SIZE=16,LR=5e-5 scripts/submit_benchmark.sh
```

脚本内部使用 `${VAR:-default}` 语法，传入的值会覆盖默认值。

### Q7: 如何恢复被中断的训练？

```python
from training.trainer import Trainer

# 创建 trainer 后加载检查点
trainer = Trainer(model, config, train_loader, val_loader)
trainer.load_checkpoint('outputs/checkpoints/best_model.pth')
trainer.fit()  # 从断点继续训练
```

---

## 训练监控

### TensorBoard

```bash
tensorboard --logdir=outputs/logs --port=6006
```

### Weights & Biases

```bash
wandb login
# 训练时会自动记录指标
```

### SLURM 日志查看

```bash
# 标准输出
cat outputs/slurm/benchmark_<job_id>.out

# 错误输出
cat outputs/slurm/benchmark_<job_id>.err

# 实时跟踪
tail -f outputs/slurm/benchmark_<job_id>.out
```

---

## 远程部署

详细部署指南请参考 `DEPLOY.md`。

快速部署：

```bash
# 1. 在远程服务器上克隆项目
git clone <your-repo-url> /workspace/lung-cancer-classification
cd /workspace/lung-cancer-classification

# 2. 创建 conda 环境并安装依赖
conda create -n lung_cancer python=3.10 -y
conda activate lung_cancer

# GPU服务器必须先装 CUDA 版 PyTorch，再装 requirements.txt
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# 3. 设置数据集路径
export LUNG_DATASET_DIR=/path/to/your/datasets

# 4. 运行测试验证环境
python -m pytest tests/ -v
python -m pytest experiments/tests/ -v

# 5. 快速自检（200样本, 1epoch, 秒级验证环境/数据/GPU）
sbatch --export=QUICK_TEST=1 scripts/submit_benchmark.sh

# 6. 提交全量实验作业
sbatch scripts/submit_benchmark.sh      # 24h
sbatch scripts/submit_ablation.sh       # 48h
sbatch scripts/submit_crossval.sh       # 72h
```

---

## 引用

本项目基于以下工作：

- ResNet: "Deep Residual Learning for Image Recognition" (He et al., 2016)
- Transformer: "Attention Is All You Need" (Vaswani et al., 2017)
- CutMix: "CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features" (Yun et al., 2019)
- MixUp: "mixup: Beyond Empirical Risk Minimization" (Zhang et al., 2018)

## 许可证

MIT License

---

**最后更新**: 2026-04-29

**版本**: v0.4.0
