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
│   ├── dataset.py                   # 基础数据集类
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
│   ├── submit_benchmark.sh          # SLURM: 基准实验提交
│   ├── submit_ablation.sh           # SLURM: 消融实验提交
│   ├── submit_crossval.sh           # SLURM: 交叉验证提交
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
├── baseline_experiment.py           # 小规模快速验证脚本
├── validate_pipeline.py             # 3步流水线烟雾测试
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

### 方式一：Python API 编程式调用

```python
from model import HybridModel
from training.trainer import Trainer, TrainingConfig
from data.custom_dataset import merge_datasets
from data.augmentation import get_train_augmentation, get_val_augmentation
from configs import DATASET_PATHS
from torch.utils.data import DataLoader, random_split

# 1. 创建模型
model = HybridModel(num_classes=3, model_dim=256, nhead=8, num_layers=4, dropout=0.1)

# 2. 加载数据
train_transform = get_train_augmentation(224)
val_transform = get_val_augmentation(224)
dataset = merge_datasets(
    DATASET_PATHS['dataset1'],
    DATASET_PATHS['dataset2'],
    DATASET_PATHS['dataset3'],
    transform=train_transform
)

# 划分数据集
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_ds, val_ds, test_ds = random_split(
    dataset, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
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

#### SLURM 快速调试模式

所有 SLURM 脚本支持快速调试模式，修改顶部变量即可：

```bash
# 快速测试模式（调试时用）
EPOCHS=2
BATCH_SIZE=4
```

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

**消融配置**：

| Config | multi_scale | gate | transformer | cross_attention | 描述 |
|--------|-------------|------|-------------|-----------------|------|
| baseline_cnn | - | - | - | - | 仅CNN |
| multiscale_only | + | - | - | - | 多尺度特征 |
| gating_added | + | SE | - | - | +门控 |
| transformer_added | + | SE | + | - | +Transformer |
| full_hybrid | + | SE | + | + | 完整架构 |

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

### 实验五：可解释性分析

**相关文件**：
- `experiments/visualization/gradcam.py` — Grad-CAM热力图
- `experiments/visualization/attention_maps.py` — 注意力图

**使用示例**：

```python
from experiments.visualization.gradcam import GradCAMVisualizer

visualizer = GradCAMVisualizer(model)
heatmap = visualizer.generate_heatmap(input_tensor, target_class=2)
```

---

### 实验六：模型复杂度评估

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
# 在远程服务器上
git clone <your-repo>
ClarifyLung-AI-Experiment/
pip install -r requirements.txt

# 设置数据集路径
export LUNG_DATASET_DIR=/path/to/your/datasets

# 运行测试验证环境
python -m pytest tests/ -v

# 提交实验作业
sbatch scripts/submit_benchmark.sh
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

**最后更新**: 2026-04-28

**版本**: v0.3.0
