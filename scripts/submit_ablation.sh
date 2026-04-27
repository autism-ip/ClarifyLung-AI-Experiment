#!/bin/bash
# =============================================================================
# 消融实验 Slurm 提交脚本
# =============================================================================
#SBATCH --job-name=hybrid_ablation
#SBATCH --output=outputs/slurm/ablation_%j.out
#SBATCH --error=outputs/slurm/ablation_%j.err
#SBATCH --partition=gpu          # GPU分区
#SBATCH --gres=gpu:1            # 请求1块GPU
#SBATCH --cpus-per-task=8       # CPU核心数
#SBATCH --mem=32G               # 内存大小
#SBATCH --time=48:00:00         # 最大运行时间（消融实验更耗时）

# =============================================================================
# 环境配置
# =============================================================================
# 加载必要模块（根据实际集群调整）
# module load anaconda/2024.01
# module load cuda/11.8

# 激活虚拟环境
# conda activate lung_cancer

# 数据集根目录（改为远程服务器实际路径，零代码修改部署）
# export LUNG_DATASET_DIR=/path/to/your/datasets

# =============================================================================
# 实验配置（根据需要修改）
# =============================================================================
# 快速测试模式（调试时用）: EPOCHS=2 BATCH_SIZE=4
EPOCHS=30
BATCH_SIZE=32
LR=1e-4
TRANSFORMER_LR=5e-4
OUTPUT_DIR="outputs/ablation"

# 确保输出目录存在
mkdir -p outputs/slurm "${OUTPUT_DIR}"

# =============================================================================
# 运行实验
# =============================================================================
echo "=========================================="
echo "消融实验"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $SLURM_JOB_PARTITION"
echo "时间: $(date)"
echo "=========================================="

python scripts/ablation_experiment.py \
  --epochs ${EPOCHS} \
  --batch-size ${BATCH_SIZE} \
  --lr ${LR} \
  --transformer-lr ${TRANSFORMER_LR} \
  --output-dir ${OUTPUT_DIR}

echo "=========================================="
echo "实验完成: $(date)"
echo "=========================================="
