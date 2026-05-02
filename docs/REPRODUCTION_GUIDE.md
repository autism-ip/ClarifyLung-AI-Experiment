# 快速复现指南：30分钟上手

> **文档版本**: v1.0  
> **最后更新**: 2026-05-03  
> **预计时间**: 初次设置 30分钟，后续实验 5分钟

---

## 1. 环境准备（5分钟）

### 1.1 克隆项目

```bash
git clone https://gitee.com/yezhenxing2025/ClarifyLung-AI-Experiment.git
cd ClarifyLung-AI-Experiment
```

### 1.2 创建虚拟环境

```bash
# 使用conda（推荐）
conda create -n lung_cancer python=3.10 -y
conda activate lung_cancer

# 或使用venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows
```

### 1.3 安装依赖

```bash
pip install -r requirements.txt
```

**核心依赖**：
- torch >= 2.0.0
- torchvision >= 0.15.0
- timm >= 0.9.0
- albumentations >= 1.3.0
- numpy, pandas, matplotlib, seaborn, scikit-learn

### 1.4 验证安装

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
python -m pytest tests/ -v --tb=short
```

**预期输出**：所有测试通过（144 tests, 100% pass）。

---

## 2. 数据准备（10分钟）

### 2.1 自动下载数据集

```bash
# 方式1：使用Kaggle API（推荐）
mkdir -p ~/.kaggle
cp /path/to/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
python scripts/download_datasets.py

# 方式2：手动下载
# 从Kaggle下载三个数据集，解压到 datasets/ 目录
# - IQ-OTHNCCD Lung Cancer Dataset
# - Lung and Colon Cancer Histopathological Images
# - Lung Cancer 4 Types Image Dataset
```

### 2.2 验证数据完整性

```bash
python scripts/download_datasets.py --validate
```

**预期输出**：
```
[INFO] Dataset 1: 3,609 images (Normal: 1,352, Benign: 1,172, Malignant: 1,085)
[INFO] Dataset 2: 25,000 images (lung_n: 5,000, lung_aca: 5,000, lung_scc: 5,000, ...)
[INFO] Dataset 3: 1,000 images (lung_n: 250, lung_aca: 250, lung_scc: 250, ...)
[INFO] Total: 29,609 images
```

### 2.3 数据目录结构

```
datasets/
├── IQ-OTHNCCD/
│   ├── Normal cases/
│   ├── Benign cases/
│   └── Malignant cases/
├── LungColon/
│   ├── lung_aca/
│   ├── lung_n/
│   └── lung_scc/
└── Lung4Types/
    ├── lung_aca/
    ├── lung_n/
    ├── lung_scc/
    └── ...
```

---

## 3. 快速验证（5分钟）

### 3.1 运行烟雾测试

```bash
# 验证数据加载 → 模型前向 → 训练流程
python scripts/validate_pipeline.py
```

**预期输出**：
```
Step 1/3: 数据加载 ✓
Step 2/3: 模型前向 ✓
Step 3/3: 训练流程 ✓
Pipeline validation passed!
```

### 3.2 快速实验（200样本，1 epoch）

```bash
# 本地快速测试
python scripts/benchmark_experiment.py --quick-test

# 或使用统一运行器
python scripts/experiment_runner.py --type benchmark --quick-test
```

**预期输出**：
```
Quick test mode: 200 samples, 1 epoch
Training: ResNet50 ... Test Acc: ~0.65
Training: ViT-B/16 ... Test Acc: ~0.55
Training: Hybrid-Advanced ... Test Acc: ~0.62
Results saved to outputs/quick_test/
```

**注意**：quick-test模式下准确率较低是正常的（数据少+epoch少），仅用于验证流程。

---

## 4. 核心实验复现

### 4.1 实验一：数据稀缺性能曲线（主实验）⭐⭐⭐⭐⭐

#### 本地运行（小规模测试）

```bash
# 只跑1个比例、1个模型、1个种子（测试用）
python scripts/experiment_runner.py \
  --type data_scarcity \
  --model hybrid_advanced \
  --data-ratio 0.05 \
  --epochs 10 \
  --pretrained False \
  --seed 42
```

#### 集群运行（完整实验）

```bash
# 提交到SLURM
sbatch scripts/submit_benchmark.sh --experiment-type data_scarcity
```

**完整实验配置**（自动生成）：
```yaml
# 7个数据比例 × 4个模型 × 3个种子 = 84 runs
ratios: [0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 1.00]
models: ['resnet50', 'vit', 'hybrid_basic', 'hybrid_advanced']
seeds: [42, 123, 456]
epochs: 50
pretrained: False
```

**预计时间**：
- 单run：~15分钟（50 epochs，单GPU L20）
- 总计：84 runs × 15min = 21 GPU小时
- 并行化：使用SLURM同时提交多个作业，1天内完成

#### 结果查看

```bash
# 查看最新实验结果
ls -lt outputs/ | head

# 查看具体结果
cat outputs/data_scarcity_35943/results.json

# 生成可视化图表
python scripts/visualize_experiment_results.py \
  --dir outputs/data_scarcity_35943
```

**预期结果**：
```
数据比例    ResNet50      ViT-B/16      Hybrid-Adv
1%         48±5%         25±8%         38±7%
5%         68±3%         55±5%         74±3%
10%        78±2%         68±4%         85±2%
25%        88±1%         82±2%         93±1%
```

---

### 4.2 实验二：跨模态迁移（创新实验）⭐⭐⭐⭐⭐

#### 运行命令

```bash
# X光训练 → 切片测试
python scripts/experiment_runner.py \
  --type cross_modal \
  --cross-modal-type xray_to_histopathology \
  --epochs 50

# 切片训练 → X光测试
python scripts/experiment_runner.py \
  --type cross_modal \
  --cross-modal-type histopathology_to_xray \
  --epochs 50
```

#### 集群提交

```bash
sbatch scripts/submit_crossval.sh --experiment-type cross_modal
```

**预计时间**：
- 单run：~20分钟
- 总计：2方向 × 4模型 × 3种子 = 24 runs = 8 GPU小时

---

### 4.3 实验三：细粒度亚型分类⭐⭐⭐⭐

#### 运行命令

```bash
# 5类分类
python scripts/experiment_runner.py \
  --type finegrained \
  --model hybrid_advanced \
  --num-classes 5 \
  --epochs 50
```

**标签映射**（自动完成）：
- 0: 肺腺癌 (lung_aca)
- 1: 肺鳞癌 (lung_scc)
- 2: 良性结节
- 3: 炎症（暂未使用）
- 4: 正常

---

### 4.4 实验四：消融实验⭐⭐⭐⭐

#### 运行命令

```bash
python scripts/ablation_experiment.py --epochs 30
```

或统一运行器：
```bash
python scripts/experiment_runner.py --type ablation
```

**5个配置**（自动运行）：
1. `baseline_cnn` — 仅CNN
2. `multiscale` — CNN + 多尺度
3. `+gating` — + 门控
4. `+transformer` — + Transformer
5. `full_hybrid` — + 交叉注意力

---

### 4.5 实验五：可解释性分析⭐⭐⭐

#### Grad-CAM可视化

```bash
python scripts/visualize_gradcam.py \
  --image datasets/IQ-OTHNCCD/Malignant\ cases/CT-3500.png \
  --checkpoint outputs/benchmark_35943/Hybrid-Advanced_best.pth \
  --output outputs/gradcam_malignant.png
```

#### Attention可视化

```bash
python scripts/visualize_attention.py \
  --image datasets/IQ-OTHNCCD/Malignant\ cases/CT-3500.png \
  --checkpoint outputs/benchmark_35943/Hybrid-Advanced_best.pth \
  --output outputs/attention_malignant.png \
  --layer 6
```

---

## 5. 结果汇总与可视化

### 5.1 自动生成图表

```bash
# 自动检测实验类型并生成所有图表
python scripts/visualize_experiment_results.py \
  --dir outputs/benchmark_35943
```

**生成文件**：
- `data_scarcity_curve.png` — 数据稀缺性能曲线
- `model_comparison_bar.png` — 模型对比柱状图
- `confusion_matrix.png` — 混淆矩阵
- `roc_curves.png` — ROC曲线
- `pr_curves.png` — PR曲线
- `training_curves.png` — 训练/验证曲线

### 5.2 手动分析

```python
import json
import numpy as np

# 加载结果
with open('outputs/benchmark_35943/results.json') as f:
    results = json.load(f)

# 查看具体指标
print(results['ResNet50']['test_accuracy'])
print(results['Hybrid-Advanced']['test_f1'])

# 加载混淆矩阵
cm = np.load('outputs/benchmark_35943/Hybrid-Advanced_confusion_matrix.npy')
print(cm)
```

---

## 6. 常见问题排查

### Q1: 数据集下载失败

**症状**：`scripts/download_datasets.py` 报错

**解决**：
```bash
# 检查Kaggle凭证
ls -la ~/.kaggle/kaggle.json

# 手动下载并解压
# 1. 从 https://www.kaggle.com/ 下载三个数据集
# 2. 解压到 datasets/ 目录
# 3. 运行验证：python scripts/download_datasets.py --validate
```

### Q2: CUDA out of memory

**症状**：训练时显存不足

**解决**：
```bash
# 减小batch_size
python scripts/experiment_runner.py --batch-size 16  # 默认32

# 或减小模型维度
python scripts/experiment_runner.py --model-dim 512  # 默认768
```

### Q3: 训练不收敛

**症状**：loss不下降或准确率随机

**解决**：
```bash
# 检查学习率
python scripts/experiment_runner.py --lr 5e-4  # 尝试更大学习率

# 检查数据加载
python scripts/validate_pipeline.py

# 检查标签是否正确
python -c "from data.custom_dataset import CustomLungDataset; d = CustomLungDataset('datasets/IQ-OTHNCCD'); print(set(d.labels))"
# 应输出 {0, 1, 2}
```

### Q4: SLURM作业失败

**症状**：`squeue`显示FAILED或CANCELLED

**解决**：
```bash
# 查看错误日志
tail -50 outputs/slurm/benchmark_35943.err

# 常见问题：
# 1. gpu01节点ECC错误 → 已自动排除（--exclude=gpu01）
# 2. 网络问题（下载预训练权重）→ 在登录节点预下载
# 3. 内存不足 → 增加 --mem=128G
```

### Q5: 如何修改实验配置？

**方法1：命令行参数**
```bash
python scripts/experiment_runner.py \
  --type data_scarcity \
  --epochs 100 \
  --lr 1e-3 \
  --model-dim 512
```

**方法2：配置文件**
```bash
# 创建自定义配置
cat > my_config.yaml << 'EOF'
experiment:
  type: data_scarcity
  epochs: 100
  lr: 1e-3
  model_dim: 512
  data_ratios: [0.01, 0.05, 0.1]
EOF

python scripts/experiment_runner.py --config my_config.yaml
```

---

## 7. 高级用法

### 7.1 添加新模型

在 `experiments/benchmark/models.py` 中添加：

```python
def create_my_model(num_classes=3, pretrained=False):
    """创建自定义模型"""
    model = MyCustomModel(num_classes=num_classes)
    return model

# 注册到MODEL_FACTORIES
MODEL_FACTORIES['my_model'] = create_my_model
```

然后在实验中使用：
```bash
python scripts/experiment_runner.py --model my_model
```

### 7.2 添加新数据集

在 `data/custom_dataset.py` 中添加：

```python
dataset_type_map = {
    # ... 现有映射 ...
    'dataset4': {
        'normal': 0,
        'benign': 1,
        'malignant': 2,
    }
}
```

然后在配置中指定：
```bash
python scripts/experiment_runner.py \
  --dataset dataset4 \
  --dataset-path /path/to/dataset4
```

### 7.3 自定义训练策略

在 `scripts/utils.py` 中修改 `train_model`：

```python
def train_model(...):
    # 自定义优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    # 自定义学习率调度
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # ... 其余训练逻辑 ...
```

---

## 8. 完整复现检查清单

```markdown
□ 环境准备
  □ 克隆仓库
  □ 创建虚拟环境
  □ 安装依赖
  □ 验证安装（pytest通过）

□ 数据准备
  □ 下载三个数据集
  □ 验证数据完整性
  □ 确认29,609张图像

□ 快速验证
  □ 运行 validate_pipeline.py
  □ 运行 quick-test（200样本，1 epoch）
  □ 确认流程无误

□ 核心实验（按优先级）
  □ 实验一：数据稀缺曲线（主实验）
    □ 1%, 5%, 10%, 25%, 50%, 100%
    □ 4个模型 × 3个种子
  □ 实验二：跨模态迁移
    □ X光→切片
    □ 切片→X光
  □ 实验三：细粒度分类
    □ 5类分类
  □ 实验四：消融实验
    □ 5个配置

□ 结果分析
  □ 运行 visualize_experiment_results.py
  □ 检查所有图表生成
  □ 统计检验（p值、效应量）

□ 论文写作
  □ 实验结果整理
  □ 图表导出（300dpi）
  □ 方法部分撰写
```

---

## 9. 一键复现脚本

为了最大程度简化，我们提供了一键复现脚本：

```bash
# 完整复现（所有实验）
./scripts/reproduce_all.sh

# 内部逻辑：
# 1. 验证环境
# 2. 验证数据
# 3. 运行 quick-test
# 4. 提交所有实验到SLURM
# 5. 监控进度
# 6. 自动生成图表
```

**注意**：此脚本预计需要48 GPU小时（约2天）。

---

## 10. 获取帮助

| 问题类型 | 查看文档 | 运行命令 |
|---------|---------|---------|
| 环境配置 | [DEPLOY.md](DEPLOY.md) | - |
| 集群部署 | [cluster_deployment_guide.md](cluster_deployment_guide.md) | - |
| 实验设计 | [EXPERIMENT_DESIGN.md](EXPERIMENT_DESIGN.md) | - |
| 代码架构 | [ARCHITECTURE_GUIDE.md](ARCHITECTURE_GUIDE.md) | - |
| 快速测试 | 本文档 | `python scripts/validate_pipeline.py` |
| 运行实验 | 本文档 | `python scripts/experiment_runner.py --help` |

**遇到bug？**
- 查看 `outputs/slurm/*.err` 错误日志
- 在GitHub提交issue：https://github.com/anomalyco/opencode/issues

---

> **上一篇**: [ARCHITECTURE_GUIDE.md](ARCHITECTURE_GUIDE.md) — 架构设计与文件说明  
> **实验方案**: [EXPERIMENT_DESIGN.md](EXPERIMENT_DESIGN.md) — 实验设计方案
