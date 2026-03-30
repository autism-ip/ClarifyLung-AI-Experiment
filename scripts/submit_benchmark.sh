#!/bin/bash
# =============================================================================
# 基准模型对比实验 Slurm 提交脚本
# =============================================================================
#SBATCH --job-name=hybrid_benchmark
#SBATCH --output=outputs/slurm/benchmark_%j.out
#SBATCH --error=outputs/slurm/benchmark_%j.err
#SBATCH --partition=gpu          # GPU分区
#SBATCH --gres=gpu:1            # 请求1块GPU
#SBATCH --cpus-per-task=8       # CPU核心数
#SBATCH --mem=32G               # 内存大小
#SBATCH --time=24:00:00         # 最大运行时间

# =============================================================================
# 环境配置
# =============================================================================
# 加载必要模块（根据实际集群调整）
# module load anaconda/2024.01
# module load cuda/11.8

# 激活虚拟环境
# conda activate lung_cancer

# =============================================================================
# 实验配置（根据需要修改）
# =============================================================================
EPOCHS=50
BATCH_SIZE=32
LR=1e-4
TRANSFORMER_LR=5e-4
OUTPUT_DIR="outputs/benchmark"

# =============================================================================
# 运行实验
# =============================================================================
echo "=========================================="
echo "基准模型对比实验"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $SLURM_JOB_PARTITION"
echo "时间: $(date)"
echo "=========================================="

python scripts/benchmark_experiment.py \
  --epochs ${EPOCHS} \
  --batch-size ${BATCH_SIZE} \
  --lr ${LR} \
  --transformer-lr ${TRANSFORMER_LR} \
  --output-dir ${OUTPUT_DIR}

echo "=========================================="
echo "实验完成: $(date)"
echo "=========================================="
