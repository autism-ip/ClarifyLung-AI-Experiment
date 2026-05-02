#!/bin/bash
# =============================================================================
# 统一实验提交脚本 (SLURM)
# 支持所有实验类型: benchmark, data_scarcity, cross_modal, finegrained, ablation
# 集群约束: 计算节点无网络，仅登录节点有网络
# =============================================================================
#SBATCH --job-name=hybrid_experiment
#SBATCH --output=outputs/slurm/experiment_%j.out
#SBATCH --error=outputs/slurm/experiment_%j.err
#SBATCH --partition=GPU
#SBATCH --exclude=gpu01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -e

# =============================================================================
# 环境配置 (计算节点无网络，依赖必须已预装)
# =============================================================================
source ~/.bashrc 2>/dev/null || true
source activate lung_cancer 2>/dev/null || conda activate lung_cancer 2>/dev/null || true

# 验证环境
echo "=========================================="
echo "统一实验提交脚本"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"
echo "Python: $(which python)"
echo "=========================================="

# GPU检查
echo "[INFO] GPU信息:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || true

# =============================================================================
# 参数解析 (通过sbatch --export传递 或 默认值)
# =============================================================================

# 实验类型: benchmark, data_scarcity, cross_modal, finegrained, ablation
EXP_TYPE=${EXP_TYPE:-"data_scarcity"}

# 模型: resnet50, vit, hybrid_basic, hybrid_advanced
MODEL=${MODEL:-"hybrid_advanced"}

# 训练参数
EPOCHS=${EPOCHS:-50}
BATCH_SIZE=${BATCH_SIZE:-32}
LR=${LR:-1e-4}
SEED=${SEED:-42}

# 数据稀缺实验参数
DATA_RATIO=${DATA_RATIO:-""}          # 如 0.05 表示5%
SCARCITY_SEEDS=${SCARCITY_SEEDS:-"42,123,456"}

# 跨模态实验参数
CROSS_MODAL_TYPE=${CROSS_MODAL_TYPE:-"xray_to_histopathology"}

# 细粒度分类
FINEGRAINED=${FINEGRAINED:-0}

# 快速测试模式
QUICK_TEST=${QUICK_TEST:-0}

# 预训练权重 (默认False，确保公平对比)
PRETRAINED=${PRETRAINED:-0}

# =============================================================================
# 输出目录
# =============================================================================
OUTPUT_DIR="outputs/${EXP_TYPE}_${SLURM_JOB_ID}"
mkdir -p outputs/slurm "${OUTPUT_DIR}"

echo "[INFO] 实验类型: ${EXP_TYPE}"
echo "[INFO] 模型: ${MODEL}"
echo "[INFO] 输出目录: ${OUTPUT_DIR}"

# =============================================================================
# 构建命令行参数
# =============================================================================
EXTRA_ARGS=""

# 快速测试模式
if [ "${QUICK_TEST}" = "1" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --quick-test"
    echo "[INFO] Quick test mode enabled"
fi

# 预训练权重
if [ "${PRETRAINED}" = "1" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --pretrained"
    echo "[INFO] 使用预训练权重"
else
    echo "[INFO] 从头训练 (pretrained=False)"
fi

# 细粒度分类
if [ "${FINEGRAINED}" = "1" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --finegrained"
    echo "[INFO] 使用5分类细粒度标签"
fi

# =============================================================================
# 运行实验
# =============================================================================
echo "[INFO] 启动实验运行器..."

# 数据稀缺实验
if [ "${EXP_TYPE}" = "data_scarcity" ]; then
    echo "[INFO] 运行数据稀缺实验"
    if [ -n "${DATA_RATIO}" ]; then
        # 单比例模式
        python -u scripts/experiment_runner.py \
            --type data_scarcity \
            --name "${EXP_TYPE}_${MODEL}_ratio${DATA_RATIO}_seed${SEED}" \
            --model ${MODEL} \
            --epochs ${EPOCHS} \
            --batch-size ${BATCH_SIZE} \
            --lr ${LR} \
            --seed ${SEED} \
            --output-dir ${OUTPUT_DIR} \
            ${EXTRA_ARGS}
    else
        # 全比例模式 (通过YAML配置)
        python -u scripts/experiment_runner.py \
            --type data_scarcity \
            --name "${EXP_TYPE}_${MODEL}_full" \
            --model ${MODEL} \
            --epochs ${EPOCHS} \
            --batch-size ${BATCH_SIZE} \
            --lr ${LR} \
            --seed ${SEED} \
            --output-dir ${OUTPUT_DIR} \
            ${EXTRA_ARGS}
    fi

# 跨模态迁移实验
elif [ "${EXP_TYPE}" = "cross_modal" ]; then
    echo "[INFO] 运行跨模态实验: ${CROSS_MODAL_TYPE}"
    python -u scripts/experiment_runner.py \
        --type cross_modal \
        --name "${EXP_TYPE}_${MODEL}_${CROSS_MODAL_TYPE}" \
        --model ${MODEL} \
        --epochs ${EPOCHS} \
        --batch-size ${BATCH_SIZE} \
        --lr ${LR} \
        --seed ${SEED} \
        --output-dir ${OUTPUT_DIR} \
        ${EXTRA_ARGS}

# 细粒度分类实验
elif [ "${EXP_TYPE}" = "finegrained" ]; then
    echo "[INFO] 运行细粒度分类实验"
    python -u scripts/experiment_runner.py \
        --type finegrained \
        --name "${EXP_TYPE}_${MODEL}" \
        --model ${MODEL} \
        --epochs ${EPOCHS} \
        --batch-size ${BATCH_SIZE} \
        --lr ${LR} \
        --seed ${SEED} \
        --output-dir ${OUTPUT_DIR} \
        --finegrained \
        ${EXTRA_ARGS}

# 消融实验
elif [ "${EXP_TYPE}" = "ablation" ]; then
    echo "[INFO] 运行消融实验"
    python -u scripts/experiment_runner.py \
        --type ablation \
        --name "${EXP_TYPE}_${MODEL}" \
        --model ${MODEL} \
        --epochs ${EPOCHS} \
        --batch-size ${BATCH_SIZE} \
        --lr ${LR} \
        --seed ${SEED} \
        --output-dir ${OUTPUT_DIR} \
        ${EXTRA_ARGS}

# 基准对比实验
elif [ "${EXP_TYPE}" = "benchmark" ]; then
    echo "[INFO] 运行基准对比实验"
    python -u scripts/experiment_runner.py \
        --type benchmark \
        --name "${EXP_TYPE}_${MODEL}" \
        --model ${MODEL} \
        --epochs ${EPOCHS} \
        --batch-size ${BATCH_SIZE} \
        --lr ${LR} \
        --seed ${SEED} \
        --output-dir ${OUTPUT_DIR} \
        ${EXTRA_ARGS}

else
    echo "[ERROR] 未知实验类型: ${EXP_TYPE}"
    echo "支持类型: benchmark, data_scarcity, cross_modal, finegrained, ablation"
    exit 1
fi

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
echo "输出目录: ${OUTPUT_DIR}"
echo "=========================================="
