# 实验数据保存优化方案

## 当前状态分析

### 已保存数据
```json
{
  "training_metrics.json": {
    "train_losses": [...],
    "train_accs": [...],
    "val_losses": [...],
    "val_accs": [...],
    "learning_rates": [...],
    "epoch_times": [...]
  },
  "benchmark_results.json": [
    {
      "model_name": "ResNet50",
      "accuracy": 0.85,
      "macro_f1": 0.83,
      "auc_roc": 0.92,
      "precision": 0.84,
      "recall": 0.82
    }
  ],
  "y_true.npy": [...],
  "y_pred.npy": [...],
  "y_prob.npy": [...]
}
```

### 缺失数据维度
1. **每epoch详细分类指标**: precision, recall, F1 per class
2. **ROC曲线数据**: fpr, tpr, thresholds per class
3. **PR曲线数据**: precision, recall, thresholds per class
4. **混淆矩阵**: 单独保存为npy
5. **模型信息**: 参数量, FLOPs, 推理时间
6. **训练详细信息**: 梯度统计, 权重分布

## 优化方案

### 1. 扩展TrainingMetrics
```python
@dataclass
class TrainingMetrics:
    # 基础指标
    train_losses: List[float]
    train_accs: List[float]
    val_losses: List[float]
    val_accs: List[float]
    learning_rates: List[float]
    epoch_times: List[float]
    
    # 扩展指标
    val_precision_per_class: List[List[float]]  # 每epoch每类precision
    val_recall_per_class: List[List[float]]     # 每epoch每类recall
    val_f1_per_class: List[List[float]]         # 每epoch每类F1
    val_f1_macro: List[float]                   # 每epoch macro F1
    val_f1_weighted: List[float]                # 每epoch weighted F1
    train_precision_per_class: List[List[float]]
    train_recall_per_class: List[List[float]]
    grad_norms: List[float]                     # 梯度范数
```

### 2. 扩展EvaluationMetrics
```python
@dataclass
class EvaluationMetrics:
    # 现有指标
    accuracy: float
    f1_macro: float
    f1_micro: float
    f1_weighted: float
    auc_roc_ovr: float
    auc_roc_ovo: float
    sensitivity: List[float]
    specificity: List[float]
    precision: List[float]
    confusion_matrix: np.ndarray
    
    # 新增：曲线数据
    roc_curve_data: Dict[str, Dict]  # {class: {fpr, tpr, thresholds}}
    pr_curve_data: Dict[str, Dict]   # {class: {precision, recall, thresholds}}
    
    # 新增：每类AUC
    auc_per_class: List[float]
```

### 3. 保存目录结构
```
outputs/experiment_name/
├── config.json                    # 实验配置
├── model_info.json                # 模型信息
│   ├── parameters                 # 参数量
│   ├── flops                      # FLOPs
│   └── inference_time_ms          # 推理时间
├── training_metrics.json          # 完整训练历史
├── results.json                   # 最终结果汇总
├── checkpoints/
│   ├── best_model.pth
│   └── final_model.pth
├── predictions/
│   ├── y_true.npy
│   ├── y_pred.npy
│   └── y_prob.npy
├── curves/
│   ├── roc_curves.npy             # ROC曲线数据
│   ├── pr_curves.npy              # PR曲线数据
│   └── confusion_matrix.npy       # 混淆矩阵
└── visualizations/                # 可视化图表（可选）
    ├── training_curves.png
    ├── confusion_matrix.png
    ├── roc_curves.png
    └── pr_curves.png
```

## 实施状态

### Phase 1: 扩展数据保存 ✅
- [x] 修改TrainingMetrics类
- [x] 修改Trainer.fit()保存每epoch详细指标
- [x] 修改EvaluationMetrics添加曲线数据

### Phase 2: 修改实验脚本 ✅
- [x] 修改benchmark_experiment.py
- [x] 修改ablation_experiment.py (扩展配置 + 完整数据保存)
- [x] 修改cross_validation_experiment.py (完整数据保存)

### Phase 3: 消融实验扩展 ✅
- [x] 正向消融: 逐步添加组件 (5配置)
- [x] 反向消融: 逐步移除组件 (5配置)
- [x] 超参数消融: Transformer层数、注意力头数、模型维度
- [x] 门控类型消融: SE/Sigmoid/None

## 消融实验配置分组

| 分组 | 配置数 | 说明 |
|------|--------|------|
| forward | 5 | 正向消融，逐步添加组件 |
| reverse | 5 | 反向消融，逐步移除组件 |
| transformer_layers | 4 | Transformer层数消融 |
| attention_heads | 3 | 注意力头数消融 |
| model_dim | 3 | 模型维度消融 |
| gate_type | 3 | 门控类型消融 |
| quick_test | 2 | 快速测试模式 |

## 使用示例

```bash
# 运行正向消融实验
python scripts/ablation_experiment.py --group forward

# 运行反向消融实验
python scripts/ablation_experiment.py --group reverse

# 运行超参数消融实验
python scripts/ablation_experiment.py --group transformer_layers

# 运行所有消融实验
python scripts/ablation_experiment.py --group all

# 快速测试模式
python scripts/ablation_experiment.py --quick-test
```

## 向后兼容性
- 旧格式的metrics文件仍然可以加载
- 新增字段使用默认空值
- 保存时同时保存新旧格式