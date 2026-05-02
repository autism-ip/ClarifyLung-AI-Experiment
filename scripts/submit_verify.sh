#!/bin/bash
# =============================================================================
# 快速验证脚本 - 测试环境和数据加载是否正常
# =============================================================================
#SBATCH --job-name=quick_verify
#SBATCH --output=outputs/slurm/verify_%j.out
#SBATCH --error=outputs/slurm/verify_%j.err
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00

set -e

# =============================================================================
# 环境配置
# =============================================================================
source ~/.bashrc 2>/dev/null || true
source activate lung_cancer 2>/dev/null || conda activate lung_cancer 2>/dev/null || true

echo "=========================================="
echo "快速环境验证"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Start: $(date)"
echo "Python: $(which python)"
echo "=========================================="

# GPU检查
echo "[INFO] GPU信息:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "无法获取GPU信息"

# 确保输出目录存在
mkdir -p outputs/slurm outputs/verify

# =============================================================================
# 运行验证
# =============================================================================
echo ""
echo "=========================================="
echo "步骤1: 验证PyTorch和CUDA"
echo "=========================================="
python -c "
import torch
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA版本: {torch.version.cuda}')
    print(f'GPU数量: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
    # 测试GPU计算
    x = torch.randn(1000, 1000).cuda()
    y = torch.matmul(x, x)
    print(f'GPU计算测试: 通过')
else:
    print('WARNING: CUDA不可用，将使用CPU训练')
"

echo ""
echo "=========================================="
echo "步骤2: 验证项目依赖"
echo "=========================================="
python -c "
import torchvision; print(f'torchvision: {torchvision.__version__}')
import timm; print(f'timm: {timm.__version__}')
import albumentations; print(f'albumentations: {albumentations.__version__}')
import numpy; print(f'numpy: {numpy.__version__}')
import pandas; print(f'pandas: {pandas.__version__}')
print('所有依赖导入成功!')
"

echo ""
echo "=========================================="
echo "步骤3: 验证数据集加载"
echo "=========================================="
python -c "
from data.custom_dataset import CustomLungDataset
from data.augmentation import get_train_augmentation, get_val_augmentation
from configs import DATASET_PATHS

print('数据集路径:')
for name, path in DATASET_PATHS.items():
    print(f'  {name}: {path}')

# 测试数据加载
train_transform = get_train_augmentation()

# 测试单个数据集
ds1 = CustomLungDataset(DATASET_PATHS['dataset1'], 'dataset1', transform=train_transform)
print(f'Dataset1 (IQ-OTHNCCD) 样本数: {len(ds1)}')
sample = ds1[0]
print(f'  样本形状: {sample[0].shape}, 标签: {sample[1]}')

ds2 = CustomLungDataset(DATASET_PATHS['dataset2'], 'dataset2', transform=train_transform)
print(f'Dataset2 (LungColon) 样本数: {len(ds2)}')
sample = ds2[0]
print(f'  样本形状: {sample[0].shape}, 标签: {sample[1]}')

print(f'总样本数: {len(ds1) + len(ds2)}')
print('数据集加载测试通过!')
"

echo ""
echo "=========================================="
echo "步骤4: 验证模型创建和前向传播"
echo "=========================================="
python -c "
import torch
from models.hybrid_model import HybridModel

model = HybridModel(num_classes=3, model_dim=256, nhead=8, num_layers=4, dropout=0.1)
print(f'模型参数量: {sum(p.numel() for p in model.parameters()):,}')

# 测试前向传播
x = torch.randn(2, 3, 224, 224)
output = model(x)
print(f'输入形状: {x.shape}')
print(f'输出形状: {output.shape}')
print('模型前向传播测试通过!')

# GPU测试
if torch.cuda.is_available():
    model = model.cuda()
    x = x.cuda()
    output = model(x)
    print('GPU模型前向传播测试通过!')
"

echo ""
echo "=========================================="
echo "步骤5: 验证训练循环"
echo "=========================================="
python scripts/validate_pipeline.py --quick-test

echo ""
echo "=========================================="
echo "验证完成: $(date)"
echo "=========================================="