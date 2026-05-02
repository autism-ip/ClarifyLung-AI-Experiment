# 架构设计文档：文件说明与设计思路

> **文档版本**: v1.0  
> **最后更新**: 2026-05-03  
> **适用项目**: ClarifyLung-AI-Experiment

---

## 1. 整体架构设计

### 1.1 设计理念

本项目采用**分层模块化架构**，核心设计原则：

| 原则 | 说明 | 体现 |
|------|------|------|
| **可复现性** | 所有实验固定种子，配置可序列化 | `configs/`, `set_seed()` |
| **公平对比** | 所有模型共享相同训练流程 | `scripts/utils.py` 统一训练函数 |
| **可扩展性** | 新增模型/实验只需添加配置 | 工厂模式 + 配置文件驱动 |
| **可解释性** | 每步操作可追踪、可可视化 | `experiments/visualization/` |

### 1.2 架构层次图

```
┌─────────────────────────────────────────────────────────────┐
│                     实验层 (Experiments)                      │
│  benchmark/  ablation/  cross_validation/  visualization/   │
│  基准对比      消融实验      交叉验证          可视化         │
├─────────────────────────────────────────────────────────────┤
│                     脚本层 (Scripts)                         │
│  experiment_runner.py  benchmark_experiment.py  submit_*.sh  │
│  统一实验运行器         基准实验CLI            SLURM提交脚本  │
├─────────────────────────────────────────────────────────────┤
│                     模型层 (Models)                          │
│  hybrid_model.py  configurable_hybrid.py  components/        │
│  混合模型          消融适配器              组件模块          │
├─────────────────────────────────────────────────────────────┤
│                     数据层 (Data)                            │
│  custom_dataset.py  data_scarcity_sampler.py  modal_splitter │
│  统一数据集加载器    稀缺采样器                模态分离器     │
├─────────────────────────────────────────────────────────────┤
│                     工具层 (Utils)                           │
│  train_model()  split_dataset()  compute_metrics()           │
│  统一训练函数     数据集划分        指标计算                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 核心文件详解

### 2.1 数据层 (data/)

#### `data/custom_dataset.py` — 三数据集统一加载器

**设计思路**：
- 三个原始数据集使用不同命名和结构（IQ-OTHNCCD: Normal/Benign/Malignant, Lung4Types: lung_n/lung_aca/lung_scc）
- 需要统一标签映射，同时保留原始信息以便细粒度分析

**核心类**：
```python
class CustomLungDataset(Dataset):
    """统一数据集加载器
    
    功能：
    - 自动探测Kaggle嵌套结构
    - 统一标签映射：normal=0, benign=1, malignant=2
    - 支持三种transform（train/val/test）
    - 保留原始标签信息（用于细粒度扩展）
    """
```

**使用示例**：
```python
from data.custom_dataset import CustomLungDataset, merge_datasets

# 加载单个数据集
dataset1 = CustomLungDataset(path='datasets/IQ-OTHNCCD', dataset_type='dataset1')

# 合并三个数据集
combined = merge_datasets(path1, path2, path3)
```

---

#### `data/data_scarcity_sampler.py` — 数据稀缺采样器 ⭐新增

**设计思路**：
- 分层采样确保每类都有代表性样本（避免某类缺失）
- 支持随机种子保证可复现
- 向上取整确保每类至少1张（极端稀缺场景）

**核心函数**：
```python
def stratified_sample_indices(labels, ratio, num_classes, seed=42):
    """分层采样索引
    
    算法：
    1. 按类别分组所有样本索引
    2. 每类采样 ceil(原始数量 × ratio) 个样本
    3. 确保每类至少 min_samples_per_class 个（默认1）
    4. 合并所有类的采样索引并随机打乱
    """
```

**关键特性**：
| 特性 | 说明 |
|------|------|
| 分层采样 | 每类独立采样，保持类别比例 |
| 向上取整 | `ceil(n × ratio)`，避免小类被采样为0 |
| 随机种子 | 固定种子确保可复现 |
| 批量API | `DataScarcitySampler` 类支持一次生成多个比例 |

**使用示例**：
```python
from data.data_scarcity_sampler import DataScarcitySampler

sampler = DataScarcitySampler(dataset, num_classes=3)
subset_5pct = sampler.sample(ratio=0.05, seed=42)  # 5%数据
```

---

#### `data/modal_splitter.py` — 跨模态数据集分离器 ⭐新增

**设计思路**：
- 根据数据集来源自动识别模态（dataset1/3是X光，dataset2是切片）
- 支持按模态划分训练/验证/测试
- 生成跨模态实验配置（源域train + 目标域test）

**核心类**：
```python
class ModalSplitter:
    """模态分离器
    
    功能：
    - 自动识别模态（X光/组织切片）
    - 按模态分离数据集
    - 生成跨模态实验配置
    """
    
    def split_by_modality(self):
        """分离为X光和切片两个子集"""
        return xray_dataset, histopath_dataset
    
    def get_cross_modal_config(self, source, target):
        """生成跨模态实验配置
        
        例如：
        - source='xray', target='histopathology'
        - 返回源域的train/val + 目标域的test
        """
```

**使用示例**：
```python
from data.modal_splitter import ModalSplitter

splitter = ModalSplitter(dataset_paths)
xray_data, histo_data = splitter.split_by_modality()

# 生成跨模态配置
config = splitter.get_cross_modal_config(
    source='xray', 
    target='histopathology'
)
```

---

#### `data/finegrained_labeler.py` — 细粒度标签映射器 ⭐新增

**设计思路**：
- 利用Lung4Types已有的4种肺癌类型标签
- IQ-OTHNCCD的Benign需要保守处理（避免引入标注噪声）
- 炎症类暂时并入良性结节（未来可扩展）

**标签映射表**：
| 原始标签 | 数据集 | 细粒度标签 | 编码 |
|---------|--------|-----------|------|
| lung_aca | Lung4Types, LungColon | 肺腺癌 | 0 |
| lung_scc | Lung4Types, LungColon | 肺鳞癌 | 1 |
| Benign | IQ-OTHNCCD | 良性结节 | 2 |
| lung_n | Lung4Types, LungColon | 良性结节 | 2 |
| - | - | 炎症 | 3（暂未使用） |
| Normal | 所有 | 正常 | 4 |

**使用示例**：
```python
from data.finegrained_labeler import FineGrainedLungDataset

# 自动转换标签
dataset = FineGrainedLungDataset(paths, transform=transform)
# 返回5类标签
```

---

### 2.2 模型层 (models/)

#### `models/hybrid_model.py` — 混合模型核心

**设计思路**：
- **模块化设计**：每个组件（CNN提取器、门控、Transformer、交叉注意力）独立可替换
- **消融开关**：通过布尔参数控制组件启用/禁用
- **维度对齐**：支持不同维度配置（256/512/768）

**架构图**：
```
输入图像 (3×224×224)
    ↓
[1] CNN Backbone (ResNet50/ConvNeXt)
    ↓ 多尺度特征 [layer3, layer4]
[2] 多尺度特征提取器 (MultiScaleFeatureExtractor)
    ↓ 融合特征 [C×H×W]
[3] 门控机制 (GatingMechanism: SE/Sigmoid/None)
    ↓ 加权特征 [C×H×W]
[4a] CNN路径：特征 → 序列化 → 投影到model_dim
[4b] Transformer路径：图像分块 → Patch Embedding → Transformer Encoder
    ↓
[5] 交叉注意力 (CrossAttention)
    - Query=CNN特征, Key/Value=Transformer特征
    - Query=Transformer特征, Key/Value=CNN特征
    ↓
[6] 全局池化 + 拼接
    ↓
[7] 分类头 (ClassificationHead)
    ↓
输出概率 (num_classes)
```

**关键参数**：
```python
HybridModel(
    backbone_name='resnet50',      # CNN backbone
    feature_layers=['layer3', 'layer4'],  # 多尺度层
    gate_type='se',                # 门控类型: 'se'/'sigmoid'/None
    model_dim=768,                 # Transformer维度
    nhead=12,                      # 注意力头数
    num_layers=12,                 # Transformer层数
    use_transformer=True,          # 消融开关
    use_cross_attention=True,      # 消融开关
    pretrained=False               # 公平对比关键！
)
```

---

#### `models/configurable_hybrid.py` — 消融实验适配器

**设计思路**：
- 将消融配置（布尔开关）映射到HybridModel的具体参数
- 确保消融实验在**真实组件**上进行，而非简化模型

**核心类**：
```python
class ConfigurableHybrid(nn.Module):
    """消融实验专用参数化混合模型
    
    将布尔开关映射为HybridModel参数：
    - multi_scale=True → feature_layers=['layer3', 'layer4']
    - multi_scale=False → feature_layers=['layer4']
    - transformer=True → use_transformer=True
    - transformer=False → use_transformer=False
    """
```

---

### 2.3 训练层 (training/ + scripts/utils.py)

#### `scripts/utils.py` — 统一训练函数 ⭐核心

**设计思路**：
- 所有实验共享同一个训练函数，确保公平对比
- 支持差分学习率、混合精度、早停、检查点

**核心函数**：
```python
def train_model(
    model, train_loader, val_loader, epochs,
    learning_rate=1e-4, transformer_lr=5e-4,
    use_amp=True, early_stopping_patience=10,
    save_path=None, seed=42
):
    """通用训练函数
    
    功能：
    - 差分学习率：CNN层lr=1e-4，Transformer层lr=5e-4
    - 混合精度：torch.cuda.amp自动管理
    - 早停：patience=10, delta=0.001
    - 检查点：自动保存最佳模型
    - 梯度裁剪：max_norm=1.0
    """
```

**训练流程**：
```
1. 设置随机种子
2. 模型 → GPU
3. 优化器：AdamW（差分学习率）
4. 学习率调度：CosineAnnealing + Linear Warmup（5 epochs）
5. 混合精度上下文（autocast + GradScaler）
6. 每epoch：
   - 训练：forward → loss → backward → step
   - 验证：计算acc/loss
   - 早停检查
   - 保存最佳检查点
7. 返回训练历史
```

---

### 2.4 实验层 (experiments/)

#### `experiments/benchmark/models.py` — 基准模型工厂

**设计思路**：
- 工厂模式统一创建所有基准模型
- 统一接口：`create_model(name, num_classes, pretrained)`

**支持的模型**：
```python
models = {
    'resnet50': create_resnet50,      # 纯CNN
    'vit': create_vit,                # 纯Transformer
    'hybrid_basic': create_hybrid_basic,   # 简化混合
    'hybrid_advanced': create_hybrid_advanced  # 完整混合
}
```

---

#### `experiments/metrics.py` — 评估指标计算

**功能**：
- 计算准确率、精确率、召回率、F1、AUC
- 支持多类别ROC/PR曲线
- 保存混淆矩阵、每类指标

**关键函数**：
```python
def compute_metrics(y_true, y_pred, y_prob, num_classes):
    """计算完整评估指标
    
    返回：
    - accuracy, macro_f1, weighted_f1
    - auc_roc (OvR)
    - per_class_precision, per_class_recall
    - confusion_matrix
    - roc_curves (每类fpr/tpr)
    - pr_curves (每类precision/recall)
    """
```

---

#### `experiments/statistical_tests.py` — 统计显著性检验 ⭐新增

**设计思路**：
- 自动化统计检验，避免手动计算错误
- 支持多种检验方法（t-test, ANOVA, Cohen's d）

**核心函数**：
```python
def paired_t_test(model_a_scores, model_b_scores, alpha=0.05):
    """配对t检验"""

def one_way_anova(*groups, alpha=0.05):
    """单因素方差分析"""

def cohens_d(group1, group2):
    """效应量计算"""

def compare_data_scarcity_curves(cnn_results, vit_results, hybrid_results):
    """识别临界点：Hybrid首次显著优于CNN的最小数据比例"""
```

---

### 2.5 脚本层 (scripts/)

#### `scripts/experiment_runner.py` — 统一实验运行器 ⭐核心

**设计思路**：
- 一个脚本运行所有实验，避免代码重复
- 配置驱动：通过YAML/JSON定义实验参数
- 支持断点续训、异常恢复

**核心类**：
```python
class ExperimentRunner:
    """统一实验运行器
    
    支持的实验类型：
    - data_scarcity: 数据稀缺性能曲线
    - cross_modal: 跨模态迁移
    - finegrained: 细粒度分类
    - ablation: 消融实验
    - benchmark: 基准对比
    """
    
    def run(self, config: ExperimentConfig):
        """运行单个实验"""
        
    def run_batch(self, configs: List[ExperimentConfig]):
        """批量运行实验"""
```

**使用示例**：
```python
# 数据稀缺实验
python scripts/experiment_runner.py \
  --type data_scarcity \
  --model hybrid_advanced \
  --data-ratio 0.05 \
  --pretrained False \
  --epochs 50 \
  --seeds 42,123,456

# 跨模态实验
python scripts/experiment_runner.py \
  --type cross_modal \
  --cross-modal-type xray_to_histopathology
```

---

#### `scripts/submit_benchmark.sh` — SLURM作业提交

**设计思路**：
- 零配置提交：自动检测环境、设置路径
- 资源管理：自动请求GPU、内存、CPU
- 容错：排除故障节点（如gpu01 ECC错误）

**关键特性**：
```bash
#SBATCH --exclude=gpu01          # 排除故障节点
#SBATCH --gres=gpu:1             # 请求1块GPU
#SBATCH --cpus-per-task=16       # 16核CPU
#SBATCH --mem=64G                # 64GB内存

# 输出目录自动使用Job ID命名
OUTPUT_DIR="outputs/benchmark_${SLURM_JOB_ID}"
```

---

### 2.6 可视化层 (experiments/visualization/ + scripts/)

#### `scripts/visualize_experiment_results.py` — 实验结果可视化

**功能**：
- 自动读取实验输出（JSON + .npy）
- 生成发表级图表（300dpi，符合期刊要求）
- 支持三种实验类型自动检测

**生成的图表**：
1. **数据稀缺曲线**：数据比例 vs 准确率（带误差棒）
2. **模型对比柱状图**：多模型并排对比
3. **混淆矩阵热力图**：归一化显示
4. **ROC曲线**：每类一条曲线
5. **PR曲线**：不平衡数据更关注PR

---

## 3. 文件依赖关系图

```
data/custom_dataset.py
    ├── data/data_scarcity_sampler.py (新增)
    ├── data/modal_splitter.py (新增)
    └── data/finegrained_labeler.py (新增)

models/hybrid_model.py
    ├── models/components/feature_extractor.py
    ├── models/components/gating.py
    ├── models/components/transformer.py
    ├── models/components/cross_attention.py
    └── models/components/classification.py
    
scripts/utils.py
    └── training/trainer.py

scripts/experiment_runner.py (新增)
    ├── data/*
    ├── models/*
    ├── scripts/utils.py
    └── experiments/metrics.py
    
experiments/statistical_tests.py (新增)
    └── experiments/metrics.py
```

---

## 4. 关键设计决策

### 4.1 为什么pretrained=False？

**问题**：预训练权重（ImageNet）会让所有模型饱和到99.6%，无法区分架构差异。

**解决方案**：所有实验从头训练（pretrained=False），迫使模型依赖自身架构能力。

**代价**：准确率从99%下降到70-90%，但**这正是我们想要的研究场景**。

### 4.2 为什么只用X光数据？

**问题**：混合模态（X光+切片）会导致模型学"模态识别"而非"病理分类"。

**解决方案**：
- 主实验只用X光数据（4,609张）
- 切片数据用于跨模态实验（创新点）

### 4.3 为什么3个随机种子？

**问题**：单次实验结果可能受随机性影响，缺乏统计显著性。

**解决方案**：每个配置运行3次（种子42, 123, 456），报告均值±标准差。

**进阶**：如需更高统计功效，可增加至5次。

### 4.4 为什么分层采样？

**问题**：随机采样可能导致某类样本缺失（如1%数据时，恶性类可能只有0张）。

**解决方案**：分层采样确保每类至少保留`ceil(原始数量 × ratio)`张。

---

## 5. 快速查找表

### 5.1 我想运行实验 → 使用哪个文件？

| 我想做... | 使用文件 | 命令示例 |
|-----------|---------|---------|
| 数据稀缺实验 | `scripts/experiment_runner.py` | `python scripts/experiment_runner.py --type data_scarcity` |
| 跨模态实验 | `scripts/experiment_runner.py` | `python scripts/experiment_runner.py --type cross_modal` |
| 细粒度分类 | `scripts/experiment_runner.py` | `python scripts/experiment_runner.py --type finegrained` |
| 消融实验 | `scripts/ablation_experiment.py` | `sbatch scripts/submit_ablation.sh` |
| 基准对比 | `scripts/benchmark_experiment.py` | `sbatch scripts/submit_benchmark.sh` |
| 交叉验证 | `scripts/cross_validation_experiment.py` | `sbatch scripts/submit_crossval.sh` |
| 可视化结果 | `scripts/visualize_experiment_results.py` | `python scripts/visualize_experiment_results.py --dir outputs/xxx` |
| Grad-CAM | `scripts/visualize_gradcam.py` | `python scripts/visualize_gradcam.py --image path.jpg` |

### 5.2 我想修改模型 → 编辑哪个文件？

| 我想修改... | 编辑文件 | 说明 |
|-------------|---------|------|
| CNN backbone | `models/hybrid_model.py` | 修改`backbone_name`参数 |
| Transformer维度 | `models/hybrid_model.py` | 修改`model_dim`, `nhead`, `num_layers` |
| 门控机制 | `models/components/gating.py` | 添加新的GatingMechanism |
| 分类头 | `models/components/classification.py` | 修改ClassificationHead |
| 融合策略 | `models/hybrid_model.py` | 修改forward中的融合逻辑 |

### 5.3 我想添加新数据集 → 需要修改哪些文件？

| 步骤 | 文件 | 修改内容 |
|------|------|---------|
| 1 | `data/custom_dataset.py` | 在`dataset_type_map`中添加新数据集的标签映射 |
| 2 | `configs/dataset_config.py` | 添加新数据集的路径配置 |
| 3 | `data/finegrained_labeler.py` | 添加细粒度标签映射规则 |
| 4 | `scripts/experiment_runner.py` | 更新实验配置生成器 |

---

## 6. 新增文件清单（本次SDD阶段）

| 文件 | 功能 | 状态 |
|------|------|------|
| `data/data_scarcity_sampler.py` | 分层稀缺采样 | ✅ 已创建 |
| `data/modal_splitter.py` | 跨模态分离 | ✅ 已创建 |
| `data/finegrained_labeler.py` | 细粒度标签 | ✅ 已创建 |
| `scripts/experiment_runner.py` | 统一实验运行器 | ✅ 已创建 |
| `experiments/statistical_tests.py` | 统计检验 | ✅ 已创建 |
| `docs/EXPERIMENT_DESIGN.md` | 实验设计方案 | ✅ 已创建 |
| `docs/ARCHITECTURE_GUIDE.md` | 本文档 | ✅ 已创建 |
| `docs/REPRODUCTION_GUIDE.md` | 快速复现指南 | 🔄 即将创建 |

---

## 7. 常见问题

**Q: 如何理解`use_transformer`和`use_cross_attention`参数？**  
A: 这是消融开关。
- `use_transformer=False, use_cross_attention=False` → 纯CNN
- `use_transformer=True, use_cross_attention=False` → CNN+Transformer独立池化
- `use_transformer=True, use_cross_attention=True` → 完整混合（交叉注意力融合）

**Q: `model_dim`为什么设为768？**  
A: 768是ViT-B/16的标准维度。当`pretrained=True`时，可以加载预训练权重。当`pretrained=False`时，仍然可以使用此维度保持与文献一致。

**Q: 如何添加新的门控机制？**  
A: 在`models/components/gating.py`中：
1. 创建新的门控类（继承`nn.Module`）
2. 在`GatingMechanism`的`__init__`中添加新类型
3. 在`hybrid_model.py`中通过`gate_type`参数指定

**Q: 实验输出保存在哪里？**  
A: 所有输出保存在`outputs/{experiment}_{jobid}/`目录：
- `best_model.pth` — 最佳检查点
- `results.json` — 完整结果
- `confusion_matrix.npy` — 混淆矩阵
- `roc_curves.json` — ROC曲线数据
- `pr_curves.json` — PR曲线数据

---

> **上一篇**: [EXPERIMENT_DESIGN.md](EXPERIMENT_DESIGN.md) — 实验设计方案  
> **下一篇**: [REPRODUCTION_GUIDE.md](REPRODUCTION_GUIDE.md) — 快速复现指南
