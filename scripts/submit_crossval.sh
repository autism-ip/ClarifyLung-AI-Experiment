#!/bin/bash
# =============================================================================
# 交叉验证实验 Slurm 提交脚本
# 集群配置: GPU节点 - 4x Tesla GPU, 64核CPU, 257GB内存
# =============================================================================
#SBATCH --job-name=hybrid_crossval
#SBATCH --output=outputs/slurm/crossval_%j.out
#SBATCH --error=outputs/slurm/crossval_%j.err
#SBATCH --partition=GPU
#SBATCH --exclude=gpu01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=72:00:00

set -e

# =============================================================================
# 环境配置
# =============================================================================
source ~/.bashrc 2>/dev/null || true
source activate lung_cancer 2>/dev/null || conda activate lung_cancer 2>/dev/null || true

# HuggingFace镜像（解决网络问题）
export HF_ENDPOINT=https://hf-mirror.com

echo "=========================================="
echo "交叉验证实验"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Start: $(date)"
echo "Python: $(which python)"
echo "=========================================="

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

# =============================================================================
# 实验配置
# =============================================================================
# 支持命令行参数覆盖 QUICK_TEST
for arg in "$@"; do
    if [ "$arg" = "--quick-test" ]; then
        QUICK_TEST=1
    fi
done
QUICK_TEST=${QUICK_TEST:-0}

FOLDS=5
EPOCHS=30
BATCH_SIZE=32
LR=1e-4

# 使用作业ID创建独立输出目录
OUTPUT_DIR="outputs/crossval_${SLURM_JOB_ID}"

mkdir -p outputs/slurm "${OUTPUT_DIR}"

echo "[INFO] 输出目录: ${OUTPUT_DIR}"

EXTRA_ARGS=""
if [ "${QUICK_TEST}" = "1" ]; then
    EXTRA_ARGS="--quick-test"
fi

python -u scripts/cross_validation_experiment.py \
  --folds ${FOLDS} \
  --epochs ${EPOCHS} \
  --batch-size ${BATCH_SIZE} \
  --lr ${LR} \
  --output-dir ${OUTPUT_DIR} \
  ${EXTRA_ARGS}

echo "=========================================="
echo "实验完成: $(date)"
echo "=========================================="