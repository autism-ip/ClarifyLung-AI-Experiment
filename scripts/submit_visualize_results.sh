#!/bin/bash
# =============================================================================
# 实验结果可视化 Slurm 提交脚本
# =============================================================================
#SBATCH --job-name=hybrid_viz_results
#SBATCH --output=outputs/slurm/viz_results_%j.out
#SBATCH --error=outputs/slurm/viz_results_%j.err
#SBATCH --partition=cpu          # CPU分区即可，无需GPU
#SBATCH --cpus-per-task=4       # CPU核心数
#SBATCH --mem=8G                # 内存大小
#SBATCH --time=01:00:00         # 最大运行时间（可视化通常几分钟内完成）

set -e  # 遇到错误立即退出

# =============================================================================
# 环境配置 (条件加载，适配不同集群)
# =============================================================================
# 如集群需要模块加载，取消下面注释并修改模块名
# module load anaconda/2024.01 2>/dev/null || true

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

# 确保输出目录存在
mkdir -p outputs/slurm outputs/figures

# =============================================================================
# 可视化配置 (修改此处参数)
# =============================================================================
# 实验输出目录（由前面的实验脚本生成）
EXPERIMENT_DIR="${EXPERIMENT_DIR:-outputs/benchmark}"
# 强制指定实验类型（可选: benchmark / ablation / crossval）
VIZ_TYPE="${VIZ_TYPE:-}"
# 仅生成指定类型（可选: comparison / confusion）
ONLY="${ONLY:-}"

# =============================================================================
# 运行可视化
# =============================================================================
echo "=========================================="
echo "实验结果可视化"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Start: $(date)"
echo "Python: $(which python)"
echo "Experiment Dir: $EXPERIMENT_DIR"
echo "=========================================="

# 构建参数
EXTRA_ARGS=""
if [ -n "$VIZ_TYPE" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --type $VIZ_TYPE"
fi
if [ -n "$ONLY" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --only $ONLY"
fi

python scripts/visualize_experiment_results.py \
  --experiment-dir "$EXPERIMENT_DIR" \
  $EXTRA_ARGS

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "=========================================="
    echo "ERROR: Visualization failed with exit code $EXIT_CODE"
    echo "End: $(date)"
    echo "=========================================="
    exit $EXIT_CODE
fi

echo "=========================================="
echo "可视化完成: $(date)"
echo "=========================================="
