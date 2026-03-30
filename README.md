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

## 快速开始

### 1. 环境安装

```bash
# 创建虚拟环境
conda create -n lung_cancer python=3.10
conda activate lung_cancer

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备

配置数据集路径（修改 `data/custom_dataset.py` 或环境变量）：

```python
# 数据集路径配置
dataset1_path = '/path/to/iq-othnccd-lung-cancer-dataset'
dataset2_path = '/path/to/lung-and-colon-cancer-histopathological-images'
dataset3_path = '/path/to/lungcancer4types-imagedataset'
```

### 3. 运行测试

```bash
# 运行所有测试
python -m pytest tests/ -v

# 运行快速测试
python -m pytest tests/test_model_forward.py -v
```

### 4. 开始训练

```python
from model import HybridModel
from training.trainer import Trainer, TrainingConfig
from data.custom_dataset import merge_datasets
from data.augmentation import get_train_augmentation, get_val_augmentation
from torch.utils.data import DataLoader, random_split

# 创建模型
model = HybridModel(
    num_classes=3,
    model_dim=256,
    nhead=8,
    num_layers=4,
    dropout=0.1
)

# 加载数据
train_transform = get_train_augmentation(224)
val_transform = get_val_augmentation(224)

dataset = merge_datasets(dataset1_path, dataset2_path, dataset3_path,
                         transform=train_transform)

# 划分数据集
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size]
)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                          num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,
                        num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
                         num_workers=4, pin_memory=True)

# 配置训练
config = TrainingConfig(
    num_epochs=100,
    batch_size=32,
    learning_rate=1e-4,       # CNN学习率
    transformer_lr=5e-4,     # Transformer学习率
    use_amp=True,            # 混合精度训练
    early_stopping_patience=10,
    output_dir="outputs/checkpoints"
)

# 创建训练器并训练
trainer = Trainer(
    model=model,
    config=config,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader
)

# 开始训练
metrics = trainer.fit()

# 评估
results = trainer.evaluate(test_loader)
```

## 项目结构

```
Clarify AI/
├── configs/                 # 配置文件模块
│   ├── __init__.py
│   └── dataset_config.py    # 数据集路径配置
├── data/                    # 数据管理模块
│   ├── __init__.py
│   ├── custom_dataset.py     # 三数据集统一加载器
│   └── augmentation.py      # 数据增强 (CutMix/MixUp/RandomErasing)
├── experiments/             # 实验分析模块
│   ├── __init__.py
│   ├── metrics.py           # 评估指标计算
│   ├── complexity.py         # 模型复杂度分析
│   ├── benchmark/           # 基准模型对比
│   │   ├── models.py         # ResNet50/ViT/Hybrid模型工厂
│   │   └── benchmarker.py    # 基准测试执行器
│   ├── ablation/            # 消融实验
│   │   ├── configs.py        # 消融配置
│   │   └── ablator.py        # 消融实验运行器
│   ├── cross_validation/    # 交叉验证
│   │   ├── validator.py      # K折分层交叉验证
│   │   └── statistical_tests.py # 统计显著性检验
│   ├── visualization/       # 可视化工具
│   │   ├── gradcam.py        # Grad-CAM热力图
│   │   ├── attention_maps.py  # 注意力图可视化
│   │   ├── training_curves.py # 训练曲线
│   │   ├── confusion_matrix.py # 混淆矩阵
│   │   ├── class_distribution.py # 类别分布
│   │   └── model_comparison.py # 模型对比图
│   └── tests/              # 实验模块测试 (146 tests, 100% pass)
├── scripts/                 # 实验脚本
│   ├── download_datasets.py  # Kaggle数据集下载
│   ├── benchmark_experiment.py # 基准模型对比实验
│   ├── ablation_experiment.py  # 消融实验
│   └── cross_validation_experiment.py # 交叉验证实验
├── training/                # 训练微调模块
│   └── trainer.py            # 完整训练流程
├── tests/                   # 单元测试模块
│   └── ...                   # 各模块测试文件
├── model.py                 # 主模型定义 (CNN-Transformer Hybrid)
├── requirements.txt         # Python依赖
├── CLAUDE.md               # 项目架构文档
└── README.md               # 本文件
```

## 文件使用说明

### data/ 模块

| 文件 | 用途 | 使用条件 | 依赖 |
|------|------|----------|------|
| `custom_dataset.py` | 三数据集统一加载 | 数据集路径配置正确 | torch, PIL, pathlib |
| `augmentation.py` | 数据增强 | 图像尺寸参数 | albumentations, torchvision |
| `dataset.py` | 基础数据集类 | 已被custom_dataset替代 | torch.utils.data |
| `visualization.py` | 数据可视化 | 需matplotlib | matplotlib, numpy |

```python
# 使用示例：加载数据集
from data.custom_dataset import CustomLungDataset, merge_datasets
from data.augmentation import get_train_augmentation, get_val_augmentation

# 单数据集加载
dataset1 = CustomLungDataset(
    root_path='/path/to/IQ-OTHNCCD/Augmented IQ-OTHNCCD lung cancer dataset',
    dataset_type='dataset1',
    transform=get_train_augmentation(224)
)

# 合并三个数据集
combined = merge_datasets(
    dataset1_path='/path/to/dataset1',
    dataset2_path='/path/to/dataset2',
    dataset3_path='/path/to/dataset3',
    transform=train_transform
)
```

### training/ 模块

| 文件 | 用途 | 使用条件 | 依赖 |
|------|------|----------|------|
| `trainer.py` | 完整训练流程 | 需train_loader, val_loader | torch, torchvision |

```python
# 使用示例：训练模型
from training.trainer import Trainer, TrainingConfig

config = TrainingConfig(
    num_epochs=100,
    batch_size=32,
    learning_rate=1e-4,
    transformer_lr=5e-4,
    use_amp=True,
    early_stopping_patience=10,
    output_dir="outputs/checkpoints"
)

trainer = Trainer(model, config, train_loader, val_loader, test_loader)
metrics = trainer.fit()
```

### configs/ 模块

| 文件 | 用途 | 使用条件 | 依赖 |
|------|------|----------|------|
| `dataset_config.py` | 数据集路径配置 | 修改DATASET_PATHS指向实际路径 | pathlib |

```python
# 使用示例：获取数据集路径
from configs import DATASET_PATHS, DATASET_INFO, validate_paths

# 验证路径是否存在
validate_paths()

# 获取路径
dataset1_path = DATASET_PATHS['dataset1']
```

### experiments/ 模块

| 文件 | 用途 | 使用条件 | 依赖 |
|------|------|----------|------|
| `benchmark/models.py` | 基准模型定义 | 需torch | torch, timm |
| `visualization/gradcam.py` | Grad-CAM热力图 | 需训练好的模型 | captum, matplotlib |
| `visualization/training_curves.py` | 训练曲线绘制 | 需训练日志 | matplotlib |
| `visualization/class_distribution.py` | 类别分布可视化 | 需数据集 | matplotlib, numpy |
| `visualization/model_comparison.py` | 模型对比图表 | 需benchmark结果 | matplotlib, pandas |
| `visualization/confusion_matrix.py` | 混淆矩阵 | 需预测结果 | matplotlib, sklearn |

```python
# 使用示例：Grad-CAM可视化
from experiments.visualization.gradcam import GradCAMVisualizer

visualizer = GradCAMVisualizer(model)
heatmap = visualizer.generate_heatmap(input_tensor, target_class)

# 使用示例：绘制训练曲线
from experiments.visualization.training_curves import plot_training_curves

plot_training_curves(metrics_dict, save_path='outputs/figures/training_curves.png')
```

### scripts/ 模块

| 文件 | 用途 | 使用条件 | 依赖 |
|------|------|----------|------|
| `download_datasets.py` | Kaggle数据集下载 | 需kagglehub+凭证 | kagglehub |
| `benchmark_experiment.py` | 基准模型对比实验 | GPU推荐 | torch, timm, matplotlib |
| `ablation_experiment.py` | 消融实验 | GPU推荐 | torch, matplotlib |
| `cross_validation_experiment.py` | K折交叉验证 | GPU推荐 | torch, scipy |

```bash
# 下载数据集
python scripts/download_datasets.py                    # 下载所有
python scripts/download_datasets.py --dataset 0       # 下载指定数据集
python scripts/download_datasets.py --validate-only   # 仅验证
```

### 正式实验脚本

#### 1. 基准模型对比实验 (benchmark_experiment.py)

对比4种基准模型的性能：ResNet50, ViT-B/16, Hybrid-Basic, Hybrid-Advanced

```bash
python scripts/benchmark_experiment.py \
  --epochs 50 \
  --batch-size 32 \
  --lr 1e-4 \
  --transformer-lr 5e-4 \
  --output-dir outputs/benchmark
```

**输出**：
- `outputs/benchmark/benchmark_results.json` - 详细结果
- `outputs/benchmark/benchmark_table.md` - Markdown对比表格
- `outputs/benchmark/model_comparison.png` - 对比柱状图
- `outputs/benchmark/training_curves.png` - 训练曲线
- `outputs/benchmark/*_best.pth` - 各模型最佳权重

#### 2. 消融实验 (ablation_experiment.py)

系统性评估 HybridModel 各组件的贡献度

```bash
python scripts/ablation_experiment.py \
  --epochs 30 \
  --batch-size 32 \
  --output-dir outputs/ablation
```

**消融配置**：

| Config | multi_scale | gate | transformer | cross_attention |
|--------|-------------|------|-------------|-----------------|
| baseline_cnn | ✗ | ✗ | ✗ | ✗ |
| multiscale_only | ✓ | ✗ | ✗ | ✗ |
| gating_added | ✓ | se | ✗ | ✗ |
| transformer_added | ✓ | se | ✓ | ✗ |
| full_hybrid | ✓ | se | ✓ | ✓ |

**输出**：
- `outputs/ablation/ablation_results.json` - 详细结果
- `outputs/ablation/ablation_table.md` - Markdown对比表格
- `outputs/ablation/*_best.pth` - 各配置最佳权重

#### 3. 交叉验证实验 (cross_validation_experiment.py)

5折分层交叉验证 + 统计显著性检验

```bash
python scripts/cross_validation_experiment.py \
  --folds 5 \
  --epochs 30 \
  --batch-size 32 \
  --output-dir outputs/cross_validation
```

**输出**：
- `outputs/cross_validation/cross_validation_results.json` - 各折结果
- `outputs/cross_validation/cross_validation_report.md` - 统计报告
- `outputs/cross_validation/cross_validation_curves.png` - 训练曲线
- `outputs/cross_validation/fold_*_best.pth` - 各折最佳权重

**统计指标**：
- 各折 Accuracy, F1, AUC-ROC
- 汇总统计 (mean ± std)
- 95% 置信区间
- Shapiro-Wilk 正态性检验
- One-sample t-test

#### 实验脚本通用参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--dataset1` | Dataset1路径 | configs配置 |
| `--dataset2` | Dataset2路径 | configs配置 |
| `--dataset3` | Dataset3路径 | configs配置 |
| `--epochs` | 训练轮数 | 50/30 |
| `--batch-size` | 批大小 | 32 |
| `--lr` | CNN学习率 | 1e-4 |
| `--transformer-lr` | Transformer学习率 | 5e-4 |
| `--seed` | 随机种子 | 42 |
| `--output-dir` | 输出目录 | outputs/* |
| `--no-checkpoint` | 不保存模型权重 | False |
| `--no-plot` | 不生成图表 (仅benchmark) | False |

### tests/ 模块

| 文件 | 用途 | 执行条件 | 依赖 |
|------|------|----------|------|
| `test_dataset_loading.py` | 数据集加载测试 | 数据集路径有效 | pytest, torch |
| `test_model_forward.py` | 模型前向测试 | 需GPU推荐 | pytest, torch |
| `test_attention_maps.py` | 注意力图测试 | 需模型 | pytest, captum |

```bash
# 执行测试
python -m pytest tests/ -v                    # 全部测试
python -m pytest tests/test_model_forward.py -v   # 单个测试文件
python -m pytest tests/ -v --cov=.            # 带覆盖率
```

### 环境要求

- **Python**: 3.9+ (推荐 3.10)
- **PyTorch**: 2.0+
- **CUDA**: 11.8+ (GPU训练推荐)
- **内存**: 16GB+ (训练)
- **磁盘**: 50GB+ (数据集)

### 依赖安装

```bash
pip install -r requirements.txt
```

核心依赖：
- torch, torchvision
- timm (预训练模型)
- albumentations (数据增强)
- matplotlib, numpy, pandas (可视化)
- captum (可解释性)
- scikit-learn (评估指标)
- kagglehub (数据集下载)

---

## 核心功能

### 1. CNN-Transformer混合架构

- **CNN部分**: ResNet50提取多尺度局部特征
- **Transformer部分**: 全局特征建模和长距离依赖
- **门控机制**: 自适应融合CNN和Transformer特征

### 2. 差分学习率策略

- **CNN层**: 较小学习率 (1e-4) - 微调预训练权重
- **Transformer层**: 较大学习率 (5e-4) - 从头训练

### 3. 数据增强

- **基础增强**: 旋转、翻转、颜色抖动、仿射变换
- **高级增强**: CutMix、MixUp、Random Erasing
- **归一化**: ImageNet统计

### 4. 训练优化

- **混合精度训练 (AMP)**: 加速并减少显存
- **早停机制**: 防止过拟合
- **学习率调度**: Cosine Annealing / One Cycle
- **检查点保存**: 最佳模型自动保存

## 数据集说明

### 三数据集统一标签映射

| 数据集 | 原始标签 | 统一标签 | 编码 |
|--------|---------|---------|------|
| IQ-OTHNCCD | Normal cases | normal | 0 |
| IQ-OTHNCCD | Benign cases | benign | 1 |
| IQ-OTHNCCD | Malignant cases | malignant | 2 |
| LungColon | lung_n | normal | 0 |
| LungColon | lung_scc | benign | 1 |
| LungColon | lung_aca | malignant | 2 |
| Lung4Types | normal | normal | 0 |
| Lung4Types | squamous.cell.carcinoma | benign | 1 |
| Lung4Types | adenocarcinoma / large.cell.carcinoma | malignant | 2 |

### 数据集来源

1. **IQ-OTHNCCD Lung Cancer Dataset** - 胸部X光片
2. **Lung and Colon Cancer Histopathological Images** - 肺组织切片
3. **Lung Cancer 4 Types Image Dataset** - 4种肺癌类型

## 测试

```bash
# 运行所有测试
python -m pytest tests/ -v

# 运行特定测试
python -m pytest tests/test_model_forward.py -v
python -m pytest tests/test_dataset_loading.py -v

# 生成覆盖率报告
python -m pytest tests/ --cov=. --cov-report=html
```

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

## 远程部署

详细部署指南请参考 `DEPLOY.md`

快速部署:

```bash
# 在远程服务器上
git clone <your-repo>
cd lung-cancer-classification
pip install -r requirements.txt
python -m pytest tests/
python scripts/train.py
```

## 引用

本项目基于以下工作:

- ResNet: "Deep Residual Learning for Image Recognition" (He et al., 2016)
- Transformer: "Attention Is All You Need" (Vaswani et al., 2017)
- CutMix: "CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features" (Yun et al., 2019)
- MixUp: "mixup: Beyond Empirical Risk Minimization" (Zhang et al., 2018)

## 许可证

MIT License

## 联系方式

- **项目维护**: [Your Name]
- **邮箱**: [your-email@example.com]
- **GitHub**: https://github.com/your-username/lung-cancer-classification

---

**最后更新**: 2026-03-25

**版本**: v0.2.0
