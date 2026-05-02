#!/bin/bash
# =============================================================================
# 基准模型对比实验 Slurm 提交脚本
# 集群配置: GPU节点 - 4x Tesla GPU, 64核CPU, 257GB内存
# =============================================================================
#SBATCH --job-name=hybrid_benchmark
#SBATCH --output=outputs/slurm/benchmark_%j.out
#SBATCH --error=outputs/slurm/benchmark_%j.err
#SBATCH --partition=GPU
#SBATCH --exclude=gpu01         # 排除有ECC错误的节点
#SBATCH --gres=gpu:1            # 请求1块GPU
#SBATCH --cpus-per-task=16      # 16核CPU (数据加载)
#SBATCH --mem=64G               # 64GB内存
#SBATCH --time=24:00:00

set -e

# =============================================================================
# 环境配置
# =============================================================================
source ~/.bashrc 2>/dev/null || true
source activate lung_cancer 2>/dev/null || conda activate lung_cancer 2>/dev/null || true

# HuggingFace镜像（解决网络问题）
export HF_ENDPOINT=https://hf-mirror.com

echo "=========================================="
echo "基准模型对比实验"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Partition: $SLURM_JOB_PARTITION"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Start: $(date)"
echo "Python: $(which python)"
echo "=========================================="

# GPU检查
echo "[INFO] GPU信息:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || true

# 数据集根目录
if [ -n "$LUNG_DATASET_DIR" ]; then
    echo "[INFO] LUNG_DATASET_DIR=$LUNG_DATASET_DIR"
else
    echo "[INFO] 使用默认数据集路径"
fi

# =============================================================================
# 实验配置
# =============================================================================
# 支持命令行参数覆盖 QUICK_TEST: ./submit_benchmark.sh --quick-test
for arg in "$@"; do
    if [ "$arg" = "--quick-test" ]; then
        QUICK_TEST=1
    fi
done
QUICK_TEST=${QUICK_TEST:-0}

EPOCHS=50
BATCH_SIZE=32
LR=1e-4
TRANSFORMER_LR=5e-4

# 使用作业ID创建独立输出目录
OUTPUT_DIR="outputs/benchmark_${SLURM_JOB_ID}"

mkdir -p outputs/slurm "${OUTPUT_DIR}"

echo "[INFO] 输出目录: ${OUTPUT_DIR}"

# =============================================================================
# 运行实验
# =============================================================================
EXTRA_ARGS=""
if [ "${QUICK_TEST}" = "1" ]; then
    EXTRA_ARGS="--quick-test"
    echo "[INFO] Quick test mode enabled"
fi

# 集群无图形界面，添加 --no-plot
python -u scripts/benchmark_experiment.py \
  --epochs ${EPOCHS} \
  --batch-size ${BATCH_SIZE} \
  --lr ${LR} \
  --transformer-lr ${TRANSFORMER_LR} \
  --output-dir ${OUTPUT_DIR} \
  --no-plot ${EXTRA_ARGS}

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "=========================================="
    echo "ERROR: Experiment failed with exit code $EXIT_CODE"
    echo "End: $(date)"
    echo "=========================================="
    exit $EXIT_CODE
fi

echo "=========================================="
echo "实验完成: $(date)"
echo "=========================================="