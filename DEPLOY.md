# 远程服务器部署指南

## 项目概览

**科训 - CNN-Transformer肺癌X光分类系统**

- **架构**: CNN-Transformer混合模型 (ResNet50 + Transformer)
- **任务**: 3分类 (normal=0, benign=1, malignant=2)
- **数据集**: 3个肺癌X光数据集统一整合

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

# 安装Python开发环境
sudo apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip
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

# 2. 克隆项目 (假设已上传到GitHub)
# git clone https://github.com/your-username/lung-cancer-classification.git
# cd lung-cancer-classification

# 或者: 上传项目压缩包并解压
# unzip lung-cancer-classification.zip -d lung-cancer-classification/
# cd lung-cancer-classification

# 3. 创建虚拟环境
python3.10 -m venv venv
source venv/bin/activate

# 4. 升级pip
pip install --upgrade pip setuptools wheel

# 5. 安装依赖 (根据CUDA版本选择)
# CUDA 11.8:
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# CPU版本 (无GPU):
# pip install torch==2.0.1+cpu torchvision==0.15.2+cpu --extra-index-url https://download.pytorch.org/whl/cpu

# 6. 安装其他依赖
pip install -r requirements.txt

# 7. 验证安装
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

### 4. 数据准备

```bash
# 创建数据目录
mkdir -p data/raw/dataset1
data/raw/dataset2
data/raw/dataset3

# 上传数据集到服务器
# 方法1: 使用scp
# scp -r /local/path/to/dataset1 user@server:/workspace/projects/lung-cancer-classification/data/raw/

# 方法2: 使用rsync
# rsync -avz --progress /local/path/to/dataset1 user@server:/workspace/projects/lung-cancer-classification/data/raw/

# 方法3: 直接下载 (如果数据在云端)
# wget https://your-data-source.com/dataset1.zip
# unzip dataset1.zip -d data/raw/dataset1/
```

### 5. 运行测试

```bash
# 激活环境
source venv/bin/activate

# 运行测试
python -m pytest tests/ -v

# 或者只运行快速测试
python -m pytest tests/test_model_forward.py -v
```

### 6. 开始训练

```bash
# 使用tmux/screen保持会话 (推荐)
tmux new -s training

# 激活环境
source venv/bin/activate

# 运行训练脚本
python scripts/train.py \
    --config configs/default.yaml \
    --data_path data/raw \
    --output_dir outputs/experiment_1

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

### 常见问题

#### 1. CUDA out of memory

```bash
# 解决方案1: 减小batch_size
# 在TrainingConfig中设置 batch_size=16 或 8

# 解决方案2: 使用梯度累积
# 在trainer.py中启用gradient_accumulation_steps

# 解决方案3: 使用更小的模型
# model_dim=128, num_layers=2
```

#### 2. 数据加载慢

```bash
# 解决方案1: 增加num_workers
# DataLoader(num_workers=8)

# 解决方案2: 使用pin_memory
# DataLoader(pin_memory=True)

# 解决方案3: 预处理数据并保存为numpy/torch格式
```

#### 3. 训练不收敛

```bash
# 检查点:
# 1. 学习率是否过大/过小
# 2. 数据预处理是否正确 (归一化)
# 3. 标签是否正确映射
# 4. 损失函数是否合适
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
- **更新日志**: 详见 `CHANGELOG.md`

---

**最后更新**: 2026-03-15

**版本**: v0.1.0
