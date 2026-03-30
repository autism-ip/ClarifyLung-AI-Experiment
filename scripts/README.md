# Slurm 作业脚本说明

## 脚本清单

| 脚本 | 用途 | 预计时间 | 资源需求 |
|------|------|----------|----------|
| `submit_benchmark.sh` | 基准模型对比实验 | ~6-8小时 | 1 GPU, 32G RAM |
| `submit_ablation.sh` | 消融实验 | ~12-16小时 | 1 GPU, 32G RAM |
| `submit_crossval.sh` | 交叉验证实验 | ~20-30小时 | 1 GPU, 32G RAM |

## 使用方法

### 1. 修改环境配置

根据实际集群环境，编辑脚本中的模块加载命令：

```bash
# 示例：加载 Anaconda 和 CUDA
module load anaconda/2024.01
module load cuda/11.8

# 激活虚拟环境
conda activate lung_cancer
```

### 2. 修改数据集路径

编辑 `configs/dataset_config.py` 中的 `DATASET_PATHS`，或通过命令行参数传递：

```bash
# 在脚本中添加
--dataset1 /path/to/dataset1
--dataset2 /path/to/dataset2
--dataset3 /path/to/dataset3
```

### 3. 提交作业

```bash
# 提交基准实验
sbatch scripts/submit_benchmark.sh

# 提交消融实验
sbatch scripts/submit_ablation.sh

# 提交交叉验证实验
sbatch scripts/submit_crossval.sh
```

### 4. 查看作业状态

```bash
# 查看所有作业
squeue -u $USER

# 查看特定作业
squeue -j <job_id>

# 查看作业输出
cat outputs/slurm/benchmark_<job_id>.out
```

### 5. 取消作业

```bash
scancel <job_id>
```

## Slurm 参数说明

| 参数 | 说明 |
|------|------|
| `--job-name` | 作业名称 |
| `--output` | 标准输出文件 (`%j` 会被替换为作业ID) |
| `--error` | 错误输出文件 |
| `--partition` | 作业队列分区 |
| `--gres` | 请求的 GPU 数量 (`gpu:1` 表示 1 块 GPU) |
| `--cpus-per-task` | 每个任务的 CPU 核心数 |
| `--mem` | 内存大小 |
| `--time` | 最大运行时间 (格式: HH:MM:SS) |

## 常用集群命令

```bash
# 查看可用的分区
sinfo

# 查看分区详情
sinfo -p gpu -l

# 查看账户信息
sacct -u $USER

# 查看作业历史
sacct -u $USER --starttime=2024-01-01
```

## 注意事项

1. **时间预估**：确保 `--time` 足够长，避免作业被系统强制终止
2. **存储空间**：输出文件会保存在 `outputs/slurm/` 目录，确保有足够空间
3. **数据集**：确保数据集路径在计算节点可访问（NFS/共享存储）
4. **依赖**：确保所有 Python 依赖已安装在虚拟环境中

## 输出文件

实验完成后，结果保存在 `outputs/` 对应目录下：

```
outputs/
├── benchmark/           # 基准实验结果
│   ├── benchmark_results.json
│   ├── benchmark_table.md
│   ├── model_comparison.png
│   ├── training_curves.png
│   └── *_best.pth
├── ablation/            # 消融实验结果
│   ├── ablation_results.json
│   ├── ablation_table.md
│   └── *_best.pth
├── cross_validation/    # 交叉验证结果
│   ├── cross_validation_results.json
│   ├── cross_validation_report.md
│   ├── cross_validation_curves.png
│   └── fold_*_best.pth
└── slurm/              # Slurm 日志
    ├── benchmark_<job_id>.out
    └── benchmark_<job_id>.err
```
