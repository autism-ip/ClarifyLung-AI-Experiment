# 实验执行计划与风险评估

## 文档信息
- **版本**: v1.0
- **日期**: 2026-05-03
- **硬件**: NVIDIA L20 (48GB VRAM)
- **数据集**: 29,609张图像

---

## 1. 实验执行计划

### 1.1 实验优先级矩阵

| 优先级 | 实验 | 科学价值 | 实现难度 | 时间成本 | 风险 |
|--------|------|---------|---------|---------|------|
| P0 | 数据稀缺性基准 | ★★★★★ | ★★☆☆☆ | ★★★★★ | 低 |
| P0 | 消融实验 | ★★★★★ | ★★★☆☆ | ★★★★☆ | 低 |
| P0 | 完整基准对比 | ★★★★☆ | ★★☆☆☆ | ★★★★☆ | 低 |
| P1 | 跨模态迁移 | ★★★★★ | ★★★★☆ | ★★★☆☆ | 中 |
| P1 | 细粒度5分类 | ★★★★☆ | ★★★☆☆ | ★★★☆☆ | 中 |
| P2 | 可解释性可视化 | ★★★☆☆ | ★★☆☆☆ | ★★☆☆☆ | 低 |

### 1.2 实验执行序列（甘特图）

```
Week 1 (Day 1-7)
├── Day 1-2: [P0] 代码实现与单元测试
│   ├── data_scarcity_sampler.py
│   ├── modal_splitter.py
│   ├── finegrained_labeler.py
│   ├── experiment_runner.py
│   └── statistical_tests.py
├── Day 3: [P0] 代码审查与集成测试
├── Day 4-5: [P0] 数据稀缺性实验 (1%, 5%, 10%)
│   └── 4 models × 3 ratios × 3 seeds = 36 runs
└── Day 6-7: [P0] 数据稀缺性实验 (25%, 50%, 100%)
    └── 4 models × 3 ratios × 3 seeds = 36 runs

Week 2 (Day 8-14)
├── Day 8-9: [P0] 消融实验
│   └── 5 configs × 3 seeds = 15 runs
├── Day 10: [P0] 完整基准对比 (100%数据)
│   └── 4 models × 3 seeds = 12 runs
├── Day 11-12: [P1] 跨模态迁移实验A/B
│   └── 2 experiments × 4 models × 3 seeds = 24 runs
└── Day 13-14: [P1] 细粒度5分类实验
    └── 4 models × 3 seeds = 12 runs

Week 3 (Day 15-21)
├── Day 15-16: [P2] 可解释性可视化
│   ├── Grad-CAM批量生成
│   └── Attention可视化
├── Day 17-18: 统计分析与论文图表
│   ├── 显著性检验
│   ├── 曲线绘制
│   └── 混淆矩阵
├── Day 19-20: 论文写作
└── Day 21: 最终验证与提交
```

### 1.3 详细时间估算（NVIDIA L20）

#### 基准数据（pretrained=False, 50 epochs）

| 模型 | 参数量 | 每epoch时间 | 50epochs | 显存占用 |
|------|--------|------------|----------|----------|
| ResNet50 | 25M | ~45s | ~38min | ~4GB |
| ViT-B/16 | 86M | ~90s | ~75min | ~8GB |
| Hybrid-Basic | ~45M | ~70s | ~58min | ~6GB |
| Hybrid-Adv | ~110M | ~120s | ~100min | ~12GB |

#### 实验时间估算

| 实验 | 配置数 | 每配置时间 | 总时间(单GPU) | 并行优化后 |
|------|--------|-----------|--------------|-----------|
| 数据稀缺性 | 72 runs | 平均45min | 54小时 | 18小时 |
| 消融实验 | 15 runs | 平均60min | 15小时 | 5小时 |
| 基准对比 | 12 runs | 平均70min | 14小时 | 5小时 |
| 跨模态A/B | 24 runs | 平均50min | 20小时 | 7小时 |
| 细粒度5类 | 12 runs | 平均70min | 14小时 | 5小时 |
| **总计** | **135 runs** | - | **117小时** | **40小时** |

### 1.4 SLURM提交策略

#### 并行化方案

```bash
# 策略1: 按实验类型并行（推荐）
# 提交5个独立作业，每个作业内部串行

# Job 1: 数据稀缺性 (最高优先级，18h)
sbatch --job-name=scarcity --time=18:00:00 --gres=gpu:1 scripts/submit_scarcity.sh

# Job 2: 消融实验 (5h)
sbatch --job-name=ablation --time=06:00:00 --gres=gpu:1 scripts/submit_ablation.sh

# Job 3: 基准对比 (5h)
sbatch --job-name=benchmark --time=06:00:00 --gres=gpu:1 scripts/submit_benchmark.sh

# Job 4: 跨模态 (7h)
sbatch --job-name=crossmodal --time=08:00:00 --gres=gpu:1 scripts/submit_crossmodal.sh

# Job 5: 细粒度 (5h)
sbatch --job-name=finegrained --time=06:00:00 --gres=gpu:1 scripts/submit_finegrained.sh
```

#### 资源分配

| 作业 | GPU数 | CPU核 | 内存 | 时间 | 依赖 |
|------|-------|-------|------|------|------|
| scarcity | 1 | 8 | 32GB | 18h | 无 |
| ablation | 1 | 8 | 32GB | 6h | 无 |
| benchmark | 1 | 8 | 32GB | 6h | 无 |
| crossmodal | 1 | 8 | 32GB | 8h | 无 |
| finegrained | 1 | 8 | 32GB | 6h | 无 |
| visualization | 0 (CPU) | 4 | 16GB | 2h | 所有GPU作业 |

#### 断点续训策略

```python
# experiment_runner.py 已内置支持
runner = ExperimentRunner(config)
# 自动检测 output_dir/experiment_name/best_model.pth
# 如果存在且配置允许，自动加载并继续训练
```

---

## 2. 实验验证策略

### 2.1 正确性验证

| 验证点 | 方法 | 通过标准 |
|--------|------|---------|
| 数据分层 | 检查每类样本数 | 每类 ≥ ceil(原始×ratio) |
| 模态分离 | 检查图像路径 | X光路径含"IQ-OTHNCCD"，切片含"lung" |
| 标签映射 | 抽查样本标签 | 手动验证10个样本/类 |
| 模型输出 | 检查logits维度 | 等于num_classes |
| 损失下降 | 监控训练曲线 | 前5epoch损失下降 > 20% |
| 准确率范围 | 检查最终结果 | 0.3~0.95之间（pretrained=False） |

### 2.2 统计验证

| 验证点 | 方法 | 通过标准 |
|--------|------|---------|
| 种子可复现 | 同配置跑2次 | 结果差异 < 0.1% |
| t-test正确性 | 使用已知数据 | 与scipy结果一致 |
| 效应量计算 | 手工验证 | Cohen's d公式正确 |
| 正态性检验 | Shapiro-Wilk | p值在合理范围 |

### 2.3 性能验证

| 验证点 | 方法 | 通过标准 |
|--------|------|---------|
| 显存不溢出 | nvidia-smi监控 | 峰值 < 40GB |
| 训练速度 | 计时 | 与预估差异 < 20% |
| 数据加载 | 监控CPU利用率 | 不成为瓶颈 |

---

## 3. 风险评估矩阵

### 3.1 Feature 1: 数据工程

| 风险 | 类型 | 概率 | 影响 | 风险等级 | 规避方案 |
|------|------|------|------|---------|---------|
| 数据集路径不存在 | 数据 | 中 | 高 | 🔴 高 | 环境变量 + 回退路径；启动时验证 |
| 炎症类别无法区分 | 数据 | 高 | 中 | 🟡 中 | 保守策略归入良性；论文中明确说明 |
| 分层采样后某类仅1张 | 数据 | 低 | 高 | 🟡 中 | min_samples_per_class=1；数据增强 |
| 模态分离错误 | 技术 | 低 | 高 | 🟡 中 | 路径关键词检查；人工抽查 |

### 3.2 Feature 2: 公平对比

| 风险 | 类型 | 概率 | 影响 | 风险等级 | 规避方案 |
|------|------|------|------|---------|---------|
| 模型仍饱和到99% | 技术 | 中 | 极高 | 🔴 高 | 确保pretrained=False；数据增强弱化；早停 |
| 训练不稳定（小数据） | 技术 | 中 | 高 | 🔴 高 | 学习率预热；梯度裁剪；多次种子取平均 |
| 临界点识别不准确 | 统计 | 中 | 中 | 🟡 中 | 增加中间比例（如2%, 3%）；平滑曲线 |
| 显存不足（Hybrid） | 技术 | 低 | 高 | 🟡 中 | 梯度累积；batch_size=16；清空缓存 |

### 3.3 Feature 3: 跨模态迁移

| 风险 | 类型 | 概率 | 影响 | 风险等级 | 规避方案 |
|------|------|------|------|---------|---------|
| 模态差异过大（域漂移） | 数据 | 高 | 高 | 🔴 高 | 报告此现象本身就是科学发现 |
| 数据量不匹配 | 数据 | 中 | 中 | 🟡 中 | 切片采样至4,609张匹配X光 |
| t-SNE可视化无分离 | 技术 | 中 | 低 | 🟢 低 | 尝试UMAP；分层可视化 |

### 3.4 Feature 4: 细粒度分类

| 风险 | 类型 | 概率 | 影响 | 风险等级 | 规避方案 |
|------|------|------|------|---------|---------|
| 5类样本极度不平衡 | 数据 | 高 | 高 | 🔴 高 | 加权损失；过采样；报告macro指标 |
| 炎症类准确率极低 | 技术 | 中 | 中 | 🟡 中 | 与良性合并为"非恶性"；调整标签策略 |
| 混淆矩阵难以解读 | 技术 | 低 | 低 | 🟢 低 | 归一化显示；只标出关键混淆对 |

### 3.5 Feature 5: 消融实验

| 风险 | 类型 | 概率 | 影响 | 风险等级 | 规避方案 |
|------|------|------|------|---------|---------|
| 组件贡献度不单调 | 技术 | 中 | 中 | 🟡 中 | 报告实际结果；分析交互效应 |
| 配置过多时间不够 | 时间 | 中 | 高 | 🟡 中 | 优先跑forward组（5个）；其他组按需 |
| ConfigurableHybrid Bug | 技术 | 低 | 高 | 🟡 中 | 单元测试每个配置的前向传播 |

### 3.6 Feature 6: 可解释性

| 风险 | 类型 | 概率 | 影响 | 风险等级 | 规避方案 |
|------|------|------|------|---------|---------|
| Grad-CAM生成失败 | 技术 | 低 | 低 | 🟢 低 | 捕获异常；跳过失败样本 |
| 注意力权重全相同 | 技术 | 低 | 低 | 🟢 低 | 检查softmax温度；可视化多层 |
| 可视化占用大量磁盘 | 技术 | 中 | 低 | 🟢 低 | 限制生成数量；压缩图片 |

### 3.7 全局风险

| 风险 | 类型 | 概率 | 影响 | 风险等级 | 规避方案 |
|------|------|------|------|---------|---------|
| **GPU节点故障** | 技术 | 中 | 极高 | 🔴 高 | SLURM自动重试；检查点频繁保存 |
| **实验超时** | 时间 | 中 | 高 | 🔴 高 | 预留20%缓冲时间；快速测试先验证 |
| **结果无法复现** | 技术 | 低 | 极高 | 🟡 中 | 固定所有随机种子；记录环境 |
| **论文deadline提前** | 时间 | 低 | 高 | 🟡 中 | 优先完成P0实验；P1/P2可裁剪 |

---

## 4. 风险应对预案

### 4.1 高风险预案（🔴）

#### 预案A: 模型饱和到99%
**触发条件**: 任何实验的测试准确率 > 95%
**应对措施**:
1. 立即检查 `pretrained=False` 是否生效
2. 降低模型容量（减少Transformer层数）
3. 弱化数据增强（去掉CutMix/MixUp）
4. 缩短训练时间（早停patience=3）
5. 如仍饱和，将实验重点转向"收敛速度"和"数据效率"

#### 预案B: GPU节点故障
**触发条件**: SLURM作业失败或节点宕机
**应对措施**:
1. 自动检查点恢复（resume_from参数）
2. 作业失败后自动重新提交（sbatch --requeue）
3. 多节点备份：同时在2个节点运行关键实验

### 4.2 中风险预案（🟡）

#### 预案C: 炎症类无法区分
**触发条件**: 炎症类(3)准确率 < 20%
**应对措施**:
1. 将炎症(3)与良性结节(2)合并为"非恶性非癌"(2)
2. 退化为4分类实验
3. 在论文中讨论此限制

#### 预案D: 实验时间不足
**触发条件**: Week 2结束时P0实验未完成
**应对措施**:
1. 削减种子数：3 → 2
2. 削减比例点：保留1%, 10%, 50%, 100%
3. 削减模型：只保留ResNet50和Hybrid-Advanced
4. 并行化：申请2个GPU同时运行

### 4.3 低风险预案（🟢）

#### 预案E: 可视化失败
**触发条件**: Grad-CAM/Attention生成报错
**应对措施**:
1. 跳过失败样本，继续其他样本
2. 降低可视化分辨率
3. 只生成代表性样本（每类1张）

---

## 5. 关键里程碑检查点

| 检查点 | 时间 | 验收标准 | 未通过应对 |
|--------|------|---------|-----------|
| CP1 | Day 2 | 5个新模块单元测试全部通过 | 延长1天修复 |
| CP2 | Day 5 | 1%/5%/10%稀缺性实验完成 | 削减模型/种子数 |
| CP3 | Day 8 | 数据稀缺性曲线显示Hybrid优势 | 调整比例范围 |
| CP4 | Day 11 | 消融实验证明组件贡献 | 增加更多配置 |
| CP5 | Day 14 | 所有P0+P1实验完成 | 削减P2实验 |
| CP6 | Day 18 | 统计检验p < 0.05 | 增加种子数/数据 |
| CP7 | Day 21 | 论文初稿完成 | 延期申请 |

---

## 6. 附录

### 6.1 快速测试检查清单

```bash
# 1. 模块测试
python data/data_scarcity_sampler.py
python data/modal_splitter.py
python data/finegrained_labeler.py
python experiments/statistical_tests.py

# 2. 快速实验（200样本，1epoch）
python scripts/experiment_runner.py --type benchmark --quick-test
python scripts/experiment_runner.py --type data_scarcity --quick-test
python scripts/experiment_runner.py --type cross_modal --quick-test

# 3. 统计检验
python -c "from experiments.statistical_tests import *; test_all()"
```

### 6.2 生产运行检查清单

```bash
# 1. 环境检查
python -c "import torch; print(torch.cuda.get_device_name(0))"
nvidia-smi

# 2. 数据检查
python -c "from configs import validate_paths; validate_paths()"

# 3. 磁盘空间检查
df -h outputs/

# 4. 提交SLURM
sbatch scripts/submit_all.sh

# 5. 监控日志
tail -f outputs/slurm-*.out
```

### 6.3 实验结果目录结构

```
outputs/experiments/
├── scarcity_resnet50_ratio0.01_seed42/
│   ├── config.yaml
│   ├── result.json
│   ├── best_model.pth
│   └── training_curves.png
├── scarcity_hybrid_advanced_ratio0.50_seed123/
│   └── ...
├── ablation_baseline_cnn_seed42/
│   └── ...
├── crossmodal_xray_to_histo_seed42/
│   └── ...
├── finegrained_hybrid_seed42/
│   └── ...
└── batch_summary.json
```
