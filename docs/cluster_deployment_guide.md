# 集群部署经验文档

## 环境配置

### 1. Conda环境创建
```bash
# 创建环境
conda create -n lung_cancer python=3.10 -y
conda activate lung_cancer

# 安装PyTorch (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 安装项目依赖
pip install -r requirements.txt

# 安装Kaggle工具
pip install kagglehub kaggle
```

### 2. Kaggle凭证配置
```bash
# 创建目录
mkdir -p ~/.kaggle

# 复制凭证文件（从Kaggle网站下载）
cp /path/to/kaggle.json ~/.kaggle/

# 设置权限
chmod 600 ~/.kaggle/kaggle.json

# 验证配置
python scripts/download_datasets.py --check-credentials
```

### 3. 数据集下载
```bash
# 下载所有数据集
python scripts/download_datasets.py --cleanup

# 验证数据集
python scripts/download_datasets.py --validate-only
```

## 集群配置

### 1. 分区信息
| 分区 | 节点数 | GPU | CPU | 内存 |
|------|--------|-----|-----|------|
| GPU  | 3      | 4x NVIDIA L20 (46GB) | 64核 | 257GB |
| CU   | 23     | 无 | 64核 | 257GB |
| FAT  | 1      | 无 | 128核 | 309GB |

### 2. SLURM脚本配置
```bash
# GPU分区（实验训练）
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# CU分区（数据处理、可视化）
#SBATCH --partition=CU
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
```

### 3. 环境激活方式
```bash
# 方式1: source activate（推荐）
source activate lung_cancer

# 方式2: conda activate（需要conda init）
conda activate lung_cancer

# 方式3: 在SLURM脚本中
source ~/.bashrc 2>/dev/null || true
source activate lung_cancer 2>/dev/null || conda activate lung_cancer 2>/dev/null || true
```

## 关键问题解决

### 1. 计算节点无法访问外网
**问题**: 计算节点下载预训练模型失败
**解决**: 在登录节点预先下载
```bash
# 在登录节点运行
source activate lung_cancer
python -c "
from torchvision import models
# 下载ResNet50
models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
# 下载ViT
models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
"
```

### 2. Kaggle数据集下载缓慢
**问题**: kagglehub下载速度慢，经常超时
**解决**: 
- 方案1: 手动下载zip文件，解压到对应目录
- 方案2: 使用后台下载脚本
```bash
nohup bash scripts/download_dataset3.sh > /tmp/download.log 2>&1 &
```

### 3. 分区名称大小写
**问题**: SLURM分区名大小写敏感
**解决**: 使用 `GPU` 而不是 `gpu`
```bash
#SBATCH --partition=GPU  # 正确
#SBATCH --partition=gpu  # 错误
```

### 4. 多GPU并行
**问题**: 单GPU训练效率低
**解决**: 
- 当前模型较小(35M参数)，单GPU足够
- 如需并行，可同时提交多个作业到不同GPU
```bash
# 并行提交三个实验
sbatch scripts/submit_benchmark.sh
sbatch scripts/submit_ablation.sh
sbatch scripts/submit_crossval.sh
```

## 实验提交流程

### 1. 快速验证
```bash
# 验证环境和数据
sbatch scripts/submit_verify.sh

# 验证单个实验
sbatch scripts/submit_benchmark.sh --quick-test
```

### 2. 完整实验
```bash
# 单独提交
sbatch scripts/submit_benchmark.sh
sbatch scripts/submit_ablation.sh
sbatch scripts/submit_crossval.sh

# 并行提交
bash scripts/submit_all_quick.sh  # quick-test模式
```

### 3. 监控作业
```bash
# 查看队列
squeue

# 查看输出
tail -f outputs/slurm/benchmark_JOBID.out

# 查看错误
cat outputs/slurm/benchmark_JOBID.err
```

## 常见错误及解决

### 1. ImportError: cannot import name 'xxx'
**原因**: 包版本不兼容
**解决**: 降级或升级相关包
```bash
pip install kagglehub==0.4.3  # 降级到兼容版本
```

### 2. RuntimeError: No images found
**原因**: 数据集路径错误或目录结构不对
**解决**: 检查数据集目录结构
```bash
python scripts/download_datasets.py --validate-only
```

### 3. CUDA out of memory
**原因**: batch_size过大
**解决**: 减小batch_size
```bash
--batch-size 16  # 从32减小到16
```

### 4. Job failed with exit code 1
**原因**: 脚本错误
**解决**: 查看错误日志
```bash
cat outputs/slurm/JOBNAME_JOBID.err
```

## 性能优化建议

### 1. 数据加载优化
- 增加 `num_workers`: `--num-workers 8`
- 使用 `pin_memory=True`
- 使用SSD存储数据集

### 2. 训练优化
- 使用混合精度训练: `--use-amp`
- 使用差分学习率: CNN小LR，Transformer大LR
- 使用早停机制避免过拟合

### 3. 资源利用
- 单节点4GPU可并行4个实验
- 3节点可并行12个实验
- 合理分配GPU/CPU/内存资源

## 维护检查清单

### 环境维护
- [ ] 定期更新依赖包
- [ ] 检查CUDA版本兼容性
- [ ] 清理缓存文件

### 数据维护
- [ ] 验证数据集完整性
- [ ] 备份重要实验结果
- [ ] 清理临时文件

### 实验维护
- [ ] 记录实验配置
- [ ] 保存模型checkpoint
- [ ] 保存训练日志