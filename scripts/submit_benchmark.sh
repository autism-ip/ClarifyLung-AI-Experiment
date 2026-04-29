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

set -e  # 遇到错误立即退出

# =============================================================================
# 环境配置 (条件加载，适配不同集群)
# =============================================================================
# 如集群需要模块加载，取消下面注释并修改模块名
# module load anaconda/2024.01 2>/dev/null || true
# module load cuda/11.8 2>/dev/null || true

# 自动检测并激活虚拟环境 (conda优先，回退venv)
if command -v conda &> /dev/null && [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo "[INFO] Using conda env: $CONDA_DEFAULT_ENV"
elif [ -f "$HOME/miniconda3/bin/activate" ] || [ -f "$HOME/anaconda3/bin/activate" ]; then
    CONDA_SH="$HOME/miniconda3/bin/activate"
    [ -f "$CONDA_SH" ] || CONDA_SH="$HOME/anaconda3/bin/activate"
    source "$CONDA_SH" lung_cancer 2>/dev/null || true
    echo "[INFO] Activated conda env: lung_cancer"
elif [ -d "venv/bin" ]; then
    source venv/bin/activate
    echo "[INFO] Activated venv"
else
    echo "[WARN] No virtual environment detected, using system Python"
fi

# 数据集根目录 (环境变量驱动，零代码修改部署)
# 如未设置，脚本将使用 configs/dataset_config.py 中的默认路径
if [ -n "$LUNG_DATASET_DIR" ]; then
    echo "[INFO] LUNG_DATASET_DIR=$LUNG_DATASET_DIR"
else
    echo "[INFO] LUNG_DATASET_DIR not set, using default paths from configs/dataset_config.py"
fi

# GPU 可用性检查
if command -v nvidia-smi &> /dev/null; then
    echo "[INFO] GPU info:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
else
    echo "[WARN] nvidia-smi not found, cannot verify GPU"
fi

# =============================================================================
# 实验配置 (修改此处参数)
# =============================================================================
# 快速测试模式 (调试/排队测试): EPOCHS=2 BATCH_SIZE=4
EPOCHS=50
BATCH_SIZE=32
LR=1e-4
TRANSFORMER_LR=5e-4
OUTPUT_DIR="outputs/benchmark"

# 确保输出目录存在
mkdir -p outputs/slurm "${OUTPUT_DIR}"

# =============================================================================
# 运行实验
# =============================================================================
echo "=========================================="
echo "基准模型对比实验"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Start: $(date)"
echo "Python: $(which python)"
echo "=========================================="

# 集群无图形界面，添加 --no-plot 跳过 matplotlib 绘图
python scripts/benchmark_experiment.py \
  --epochs ${EPOCHS} \
  --batch-size ${BATCH_SIZE} \
  --lr ${LR} \
  --transformer-lr ${TRANSFORMER_LR} \
  --output-dir ${OUTPUT_DIR} \
  --no-plot

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
