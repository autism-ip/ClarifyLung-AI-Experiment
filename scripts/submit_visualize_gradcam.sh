#!/bin/bash
# =============================================================================
# Grad-CAM 可视化 Slurm 提交脚本
# =============================================================================
#SBATCH --job-name=hybrid_viz_gradcam
#SBATCH --output=outputs/slurm/viz_gradcam_%j.out
#SBATCH --error=outputs/slurm/viz_gradcam_%j.err
#SBATCH --partition=gpu          # GPU分区，模型加载需要GPU
#SBATCH --gres=gpu:1            # 请求1块GPU
#SBATCH --cpus-per-task=4       # CPU核心数
#SBATCH --mem=16G               # 内存大小
#SBATCH --time=00:15:00         # 最大运行时间（单图推理很快）

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

# GPU 可用性检查
if command -v nvidia-smi &> /dev/null; then
    echo "[INFO] GPU info:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
else
    echo "[WARN] nvidia-smi not found, cannot verify GPU"
fi

# 确保输出目录存在
mkdir -p outputs/slurm outputs/figures

# =============================================================================
# Grad-CAM 配置 (修改此处参数)
# =============================================================================
# 必填：输入图像路径
IMAGE_PATH="${IMAGE_PATH:-datasets/IQ-OTHNCCD/Normal cases/normal_001.png}"
# 必填：模型权重路径
CHECKPOINT="${CHECKPOINT:-outputs/checkpoints/best_model.pth}"
# 必填：输出热力图路径
OUTPUT_PATH="${OUTPUT_PATH:-outputs/figures/gradcam_result.png}"
# 可选：目标类别（0=normal, 1=benign, 2=malignant），不指定则使用预测类别
TARGET_CLASS="${TARGET_CLASS:-}"
# 可选：透明度
ALPHA="${ALPHA:-0.5}"
# 可选：模型维度（需与 checkpoint 一致）
MODEL_DIM="${MODEL_DIM:-256}"
# 可选：Transformer层数（需与 checkpoint 一致）
NUM_LAYERS="${NUM_LAYERS:-4}"

# =============================================================================
# 运行 Grad-CAM
# =============================================================================
echo "=========================================="
echo "Grad-CAM 可视化"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Start: $(date)"
echo "Python: $(which python)"
echo "Image: $IMAGE_PATH"
echo "Checkpoint: $CHECKPOINT"
echo "Output: $OUTPUT_PATH"
echo "=========================================="

# 构建参数
EXTRA_ARGS=""
if [ -n "$TARGET_CLASS" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --target-class $TARGET_CLASS"
fi
if [ -n "$ALPHA" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --alpha $ALPHA"
fi
if [ -n "$MODEL_DIM" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --model-dim $MODEL_DIM"
fi
if [ -n "$NUM_LAYERS" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --num-layers $NUM_LAYERS"
fi

python scripts/visualize_gradcam.py \
  --image "$IMAGE_PATH" \
  --checkpoint "$CHECKPOINT" \
  --output "$OUTPUT_PATH" \
  $EXTRA_ARGS

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "=========================================="
    echo "ERROR: Grad-CAM failed with exit code $EXIT_CODE"
    echo "End: $(date)"
    echo "=========================================="
    exit $EXIT_CODE
fi

echo "=========================================="
echo "Grad-CAM 完成: $(date)"
echo "输出文件: $OUTPUT_PATH"
echo "=========================================="
