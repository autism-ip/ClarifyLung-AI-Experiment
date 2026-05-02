#!/bin/bash
# =============================================================================
# 并行提交三个实验的快速验证作业
# =============================================================================

echo "=========================================="
echo "提交三个实验的快速验证作业"
echo "=========================================="

# 确保输出目录存在
mkdir -p outputs/slurm

# 提交基准实验 (quick-test)
echo "[1/3] 提交基准实验..."
JOB1=$(sbatch --parsable scripts/submit_benchmark.sh --quick-test 2>/dev/null || \
       sbatch --parsable --export=QUICK_TEST=1 scripts/submit_benchmark.sh)
echo "  Job ID: $JOB1"

# 提交消融实验 (quick-test)
echo "[2/3] 提交消融实验..."
JOB2=$(sbatch --parsable scripts/submit_ablation.sh --quick-test 2>/dev/null || \
       sbatch --parsable --export=QUICK_TEST=1 scripts/submit_ablation.sh)
echo "  Job ID: $JOB2"

# 提交交叉验证 (quick-test)
echo "[3/3] 提交交叉验证..."
JOB3=$(sbatch --parsable scripts/submit_crossval.sh --quick-test 2>/dev/null || \
       sbatch --parsable --export=QUICK_TEST=1 scripts/submit_crossval.sh)
echo "  Job ID: $JOB3"

echo ""
echo "=========================================="
echo "三个作业已提交:"
echo "  基准实验: Job $JOB1"
echo "  消融实验: Job $JOB2"
echo "  交叉验证: Job $JOB3"
echo "=========================================="
echo ""
echo "查看作业状态: squeue"
echo "查看输出: tail -f outputs/slurm/benchmark_${JOB1}.out"
echo "          tail -f outputs/slurm/ablation_${JOB2}.out"
echo "          tail -f outputs/slurm/crossval_${JOB3}.out"