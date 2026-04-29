# 远程服务器部署指南

## 项目概览

**科训 - CNN-Transformer肺癌X光分类系统**

- **架构**: CNN-Transformer混合模型 (ResNet50 + Transformer)
- **任务**: 3分类 (normal=0, benign=1, malignant=2)
- **数据集**: 3个肺癌X光数据集统一整合
- **运行模式**: 本地CLI + 远程SLURM批作业双模式

---

## 环境要求

### 硬件要求

| 组件 | 最低配置 | 推荐配置 |
|------|---------|---------|
| CPU | 8核 | 16核+ |
| GPU | 8GB显存 | 24GB显存 (A100/V100) |
| 内存 | 32GB | 64GB+ |
| 存储 | 100GB SSD | 500GB NVMe SSD |

### 软件环境

- **OS**: Ubuntu 20.04/22.04 LTS 或 CentOS 8
- **Python**: 3.10+
- **CUDA**: 11.8+ (如使用GPU)
- **cuDNN**: 8.6+ (如使用GPU)
- **SLURM**: 如使用集群批作业提交

---

## 部署步骤

### 1. 服务器准备

```bash
# 更新系统
sudo apt-get update && sudo apt-get upgrade -y

# 安装基础工具
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    vim \
    htop \
    tmux \
    unzip
```

### 2. 安装CUDA (GPU服务器)

```bash
# 下载并安装CUDA 11.8
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run

# 配置环境变量
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 验证安装
nvcc --version
nvidia-smi
```

### 3. 项目部署

```bash
# 1. 创建项目目录
mkdir -p /workspace/projects
cd /workspace/projects

# 2. 克隆项目
git clone <your-repo-url> /workspace/lung-cancer-classification
cd /workspace/lung-cancer-classification

# 3. 创建conda环境（脚本优先检测conda，venv为回退）
conda create -n lung_cancer python=3.10 -y
conda activate lung_cancer

# 4. 升级pip
pip install --upgrade pip setuptools wheel

# 5. 安装PyTorch (GPU版本，必须显式指定CUDA索引)
# CUDA 11.8:
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# CPU版本 (无GPU):
# pip install torch==2.0.1+cpu torchvision==0.15.2+cpu --extra-index-url https://download.pytorch.org/whl/cpu

# 6. 安装其余依赖
pip install -r requirements.txt

# 7. 验证GPU可用性
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

> **重要**: GPU服务器必须先安装CUDA版PyTorch，再执行 `pip install -r requirements.txt`。
> 如果顺序颠倒，`requirements.txt` 中的 `torch>=2.0.0` 可能会安装CPU版本。

### 4. 数据准备

**方式A（推荐）: 环境变量驱动**

```bash
# 设置数据集根目录，零代码修改
export LUNG_DATASET_DIR=/path/to/your/datasets

# 数据集目录结构应如下:
# /path/to/your/datasets/
# ├── IQ-OTHNCCD/
# │   └── Augmented IQ-OTHNCCD lung cancer dataset/
# │       ├── Normal cases/
# │       ├── Malignant cases/
# │       └── Benign cases/
# ├── LungColon/
# │   └── lung_colon_image_set/
# │       └── lung_image_sets/
# │           ├── lung_n/
# │           ├── lung_aca/
# │           └── lung_scc/
# └── Lung4Types/
#     └── Data/
#         ├── train/
#         ├── valid/
#         └── test/
```

**方式B: 放置到项目本地 `datasets/` 目录**

```bash
mkdir -p datasets/
# 将三个数据集复制到 datasets/IQ-OTHNCCD, datasets/LungColon, datasets/Lung4Types
```

**方式C: Kaggle自动下载**

```bash
# 配置Kaggle凭证 ~/.kaggle/kaggle.json
python scripts/download_datasets.py
```

**验证数据路径:**

```bash
python -c "from configs import validate_paths; validate_paths()"
```

### 5. 环境验证

```bash
# 激活环境
conda activate lung_cancer

# 运行核心测试
python -m pytest tests/ -v

# 运行实验模块测试
python -m pytest experiments/tests/ -v

# 快速流水线验证
python scripts/validate_pipeline.py
```

---

## SLURM 批作业提交

### 前置配置

编辑对应的 `scripts/submit_*.sh` 文件，根据集群环境调整：

```bash
# =============================================================================
# 环境配置（根据实际集群调整）
# =============================================================================
# module load anaconda/2024.01    # 加载conda模块（如需要）
# module load cuda/11.8           # 加载CUDA模块（如需要）

# 数据集根目录（零代码修改部署）
export LUNG_DATASET_DIR=/path/to/your/datasets
```

### 快速自检（提交全量前必做）

```bash
# 200样本，1epoch，秒级验证环境/数据/GPU
sbatch --export=QUICK_TEST=1 scripts/submit_benchmark.sh
sbatch --export=QUICK_TEST=1 scripts/submit_ablation.sh
sbatch --export=QUICK_TEST=1 scripts/submit_crossval.sh

# 查看日志确认成功
tail -f outputs/slurm/benchmark_*.out
tail -f outputs/slurm/ablation_*.out
tail -f outputs/slurm/crossval_*.out
```

> 脚本使用 `${QUICK_TEST:-0}` 读取环境变量，支持 `sbatch --export=QUICK_TEST=1` 覆盖，
> 无需修改脚本内部变量。

### 提交全量实验

```bash
# 三个实验互相独立，可同时提交
sbatch scripts/submit_benchmark.sh      # 24h
sbatch scripts/submit_ablation.sh       # 48h
sbatch scripts/submit_crossval.sh       # 72h
```

### 作业监控

```bash
# 查看作业队列
squeue -u $USER

# 查看作业详情
scontrol show job <job_id>

# 实时查看输出
tail -f outputs/slurm/benchmark_<job_id>.out
tail -f outputs/slurm/benchmark_<job_id>.err

# 取消作业
scancel <job_id>
```

---

## 单机直接运行（非SLURM环境）

```bash
# 使用tmux/screen保持会话
tmux new -s training

# 激活环境
conda activate lung_cancer

# 设置数据集路径
export LUNG_DATASET_DIR=/path/to/your/datasets

# 运行基准实验
python scripts/benchmark_experiment.py \
    --epochs 50 \
    --batch-size 32 \
    --output-dir outputs/benchmark

# 分离会话: Ctrl+B, 然后 D
# 重新连接: tmux attach -t training
```

---

## 监控与日志

### TensorBoard

```bash
# 启动TensorBoard
tensorboard --logdir=outputs/logs --port=6006

# 在本地通过SSH隧道访问 (本地终端运行)
# ssh -L 6006:localhost:6006 user@server
# 然后访问 http://localhost:6006
```

### Weights & Biases

```bash
# 登录
wandb login

# 训练时会自动记录指标到wandb
# 访问 https://wandb.ai/your-username 查看结果
```

### 系统监控

```bash
# GPU监控
watch -n 1 nvidia-smi

# CPU/内存监控
htop

# 磁盘使用
df -h
```

---

## 故障排查

### 1. CUDA out of memory

```bash
# 解决方案1: 减小batch_size
python scripts/benchmark_experiment.py --batch-size 16

# 解决方案2: 使用更小的模型配置
# 在脚本中修改 model_dim=128, num_layers=2
```

### 2. 数据加载慢

```bash
# 增加num_workers
# 在脚本中修改 num_workers=8
```

### 3. 训练不收敛

```bash
# 检查点:
# 1. 学习率是否过大/过小
# 2. 数据预处理是否正确 (归一化)
# 3. 标签是否正确映射
# 4. 损失函数是否合适
```

### 4. SLURM作业失败

```bash
# 查看错误日志
cat outputs/slurm/<job_name>_<job_id>.err

# 常见原因:
# - 虚拟环境未正确激活
# - 数据集路径不存在
# - CUDA版本与PyTorch不匹配
# - 内存/显存不足
```

---

## 安全注意事项

1. **数据隐私**: 医疗数据包含敏感信息，确保符合HIPAA/GDPR等法规
2. **访问控制**: 限制服务器访问权限，使用SSH密钥而非密码
3. **数据加密**: 传输和存储时加密敏感数据
4. **审计日志**: 记录数据访问和模型训练活动

---

## 联系与支持

- **项目文档**: 详见 `docs/` 目录
- **问题反馈**: 在GitHub Issues中提交
- **更新日志**: 详见 `git log`

---

**最后更新**: 2026-04-29

**版本**: v0.3.1
