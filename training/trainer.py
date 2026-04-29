"""
CNN-Transformer训练流程模块
[INPUT]: 模型、数据加载器、配置
[OUTPUT]: 训练后的模型、训练历史、检查点
[POS]: training/核心组件，实现端到端训练流程
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
import numpy as np


logger = logging.getLogger(__name__)


# =============================================================================
# 配置类
# =============================================================================

@dataclass
class TrainingConfig:
    """训练配置类"""
    # 基础训练参数
    num_epochs: int = 100
    batch_size: int = 32
    num_workers: int = 4

    # 优化器参数
    optimizer: str = "adamw"  # adamw, adam, sgd
    learning_rate: float = 1e-4  # CNN使用较小LR
    transformer_lr: float = 5e-4  # Transformer使用较大LR
    weight_decay: float = 0.01

    # 学习率调度
    scheduler: str = "cosine"  # cosine, one_cycle, none
    warmup_epochs: int = 5
    min_lr: float = 1e-6

    # 正则化
    dropout: float = 0.1
    label_smoothing: float = 0.1

    # 混合精度训练
    use_amp: bool = True

    # 检查点
    save_every: int = 10
    output_dir: str = "outputs/checkpoints"

    # 早停
    early_stopping_patience: int = 10
    early_stopping_delta: float = 0.001
    early_stopping_monitor: str = "val_acc"  # val_acc / val_loss
    early_stopping_mode: str = "max"  # max / min


@dataclass
class TrainingMetrics:
    """训练指标记录类"""
    train_losses: List[float] = field(default_factory=list)
    train_accs: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    val_accs: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    epoch_times: List[float] = field(default_factory=list)

    def get_best_epoch(self, metric: str = "val_acc", mode: str = "max") -> Tuple[int, float]:
        """获取最佳epoch"""
        if metric == "val_acc":
            values = self.val_accs
        elif metric == "val_loss":
            values = self.val_losses
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if mode == "max":
            best_value = max(values)
        else:
            best_value = min(values)
        best_epoch = values.index(best_value)
        return best_epoch, best_value

    def save(self, path: str):
        """保存指标到文件"""
        import json
        with open(path, 'w') as f:
            json.dump({
                'train_losses': self.train_losses,
                'train_accs': self.train_accs,
                'val_losses': self.val_losses,
                'val_accs': self.val_accs,
                'learning_rates': self.learning_rates,
                'epoch_times': self.epoch_times,
            }, f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'TrainingMetrics':
        """从文件加载指标"""
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        metrics = cls()
        for key, value in data.items():
            if hasattr(metrics, key):
                setattr(metrics, key, value)
        return metrics


# =============================================================================
# 辅助函数
# =============================================================================

def get_optimizer(
    model: nn.Module,
    config: TrainingConfig
) -> optim.Optimizer:
    """
    创建优化器，支持差分学习率

    CNN使用较小学习率，Transformer使用较大学习率
    """
    # 分离CNN和Transformer参数
    cnn_params = []
    transformer_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # 按模块前缀精确分组，避免子串误匹配
        top_module = name.split('.')[0].lower()
        if top_module == 'backbone':
            cnn_params.append(param)
        elif top_module in ('transformer_encoder', 'image_patch_extractor'):
            transformer_params.append(param)
        else:
            other_params.append(param)

    # 构建参数组
    param_groups = [
        {'params': cnn_params, 'lr': config.learning_rate},
        {'params': transformer_params, 'lr': config.transformer_lr},
        {'params': other_params, 'lr': config.learning_rate},
    ]

    # 创建优化器
    if config.optimizer == "adamw":
        optimizer = optim.AdamW(
            param_groups,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999)
        )
    elif config.optimizer == "adam":
        optimizer = optim.Adam(
            param_groups,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999)
        )
    elif config.optimizer == "sgd":
        optimizer = optim.SGD(
            param_groups,
            momentum=0.9,
            weight_decay=config.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")

    return optimizer


def get_scheduler(
    optimizer: optim.Optimizer,
    config: TrainingConfig,
    steps_per_epoch: int
) -> Optional[Any]:
    """创建学习率调度器"""
    if config.scheduler == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config.num_epochs,
            eta_min=config.min_lr
        )
    elif config.scheduler == "one_cycle":
        scheduler = OneCycleLR(
            optimizer,
            max_lr=[config.learning_rate, config.transformer_lr, config.learning_rate],
            epochs=config.num_epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=config.warmup_epochs / config.num_epochs,
            anneal_strategy='cos'
        )
    else:
        scheduler = None

    return scheduler


def accuracy(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    topk: Tuple[int, ...] = (1,)
) -> List[float]:
    """计算top-k准确率"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = targets.size(0)

        _, pred = outputs.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


# =============================================================================
# 训练器主类
# =============================================================================

class Trainer:
    """
    CNN-Transformer混合模型训练器

    支持功能：
    - 差分学习率（CNN小LR，Transformer大LR）
    - 混合精度训练（AMP）
    - 多种学习率调度策略
    - 早停机制
    - 检查点保存与恢复
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: Optional[torch.device] = None,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
    ):
        """
        初始化训练器

        Args:
            model: 待训练模型
            config: 训练配置
            device: 计算设备（自动检测CUDA）
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            test_loader: 测试数据加载器
        """
        self.model = model
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # 训练状态
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.metrics = TrainingMetrics()

        # 初始化
        self.model.to(self.device)
        self._setup_training()

        # 创建输出目录
        os.makedirs(config.output_dir, exist_ok=True)

    def _setup_training(self):
        """设置训练组件（优化器、调度器、损失函数）"""
        # 创建优化器（支持差分学习率）
        self.optimizer = get_optimizer(self.model, self.config)

        # 创建学习率调度器
        steps_per_epoch = len(self.train_loader) if self.train_loader else 1
        self.scheduler = get_scheduler(
            self.optimizer, self.config, steps_per_epoch
        )

        # 创建损失函数
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=self.config.label_smoothing
        )

        # 设置混合精度训练
        self.scaler = torch.amp.GradScaler(self.device.type) if self.config.use_amp else None

    def train_epoch(self) -> Tuple[float, float]:
        """
        训练一个epoch

        Returns:
            (平均损失, 准确率)
        """
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        pbar = self.train_loader
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # 清零梯度 (在计算前执行，符合规范)
            self.optimizer.zero_grad(set_to_none=True)

            # 混合精度训练
            if self.config.use_amp and self.scaler is not None:
                with torch.amp.autocast(self.device.type):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

                loss.backward()
                self.optimizer.step()

            # 计算准确率
            _, predicted = outputs.max(1)
            total_correct += predicted.eq(targets).sum().item()
            total_samples += targets.size(0)
            total_loss += loss.item()

            # 更新学习率（OneCycle调度器需要每步更新）
            if isinstance(self.scheduler, OneCycleLR):
                self.scheduler.step()

        # 计算平均指标
        avg_loss = total_loss / len(self.train_loader)
        avg_acc = 100.0 * total_correct / total_samples

        return avg_loss, avg_acc

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> Tuple[float, float, List, List]:
        """
        验证/测试

        Args:
            loader: 数据加载器

        Returns:
            (平均损失, 准确率)
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_preds = []
        all_targets = []

        for images, targets in loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # 混合精度推理
            if self.config.use_amp and self.scaler is not None:
                with torch.amp.autocast(self.device.type):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

            # 统计
            _, predicted = outputs.max(1)
            total_correct += predicted.eq(targets).sum().item()
            total_samples += targets.size(0)
            total_loss += loss.item()

            # 收集预测结果
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

        # 计算平均指标
        avg_loss = total_loss / len(loader)
        avg_acc = 100.0 * total_correct / total_samples

        return avg_loss, avg_acc, all_preds, all_targets

    def fit(self) -> TrainingMetrics:
        """
        完整训练流程

        Returns:
            TrainingMetrics: 训练指标记录
        """
        logger.info(f"Starting training for {self.config.num_epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info("-" * 80)

        # 早停监控指标选择
        monitor = self.config.early_stopping_monitor
        mode = self.config.early_stopping_mode

        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()

            # 训练
            train_loss, train_acc = self.train_epoch()

            # 验证
            if self.val_loader is not None:
                val_loss, val_acc, _, _ = self.validate(self.val_loader)
            else:
                val_loss, val_acc = 0.0, 0.0

            epoch_time = time.time() - epoch_start

            # 更新学习率
            if self.scheduler is not None and not isinstance(self.scheduler, OneCycleLR):
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']

            # 记录指标
            self.metrics.train_losses.append(train_loss)
            self.metrics.train_accs.append(train_acc)
            self.metrics.val_losses.append(val_loss)
            self.metrics.val_accs.append(val_acc)
            self.metrics.learning_rates.append(current_lr)
            self.metrics.epoch_times.append(epoch_time)

            # 打印进度
            logger.info(
                f"Epoch [{epoch+1:03d}/{self.config.num_epochs:03d}] "
                f"Time: {epoch_time:.1f}s | "
                f"LR: {current_lr:.2e} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%"
            )

            # 确定当前监控值并判断是否为最佳
            current_val = val_acc if monitor == "val_acc" else val_loss
            is_best = False
            if mode == "max":
                is_best = current_val > self.best_val_acc + self.config.early_stopping_delta
            else:
                is_best = current_val < self.best_val_loss - self.config.early_stopping_delta

            if is_best:
                if monitor == "val_acc":
                    self.best_val_acc = current_val
                    self.best_val_loss = val_loss
                else:
                    self.best_val_loss = current_val
                    self.best_val_acc = val_acc
                self.save_checkpoint("best_model.pth")
                logger.info(f"  New best model saved! ({monitor}: {current_val:.4f})")
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # 定期保存检查点
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1:03d}.pth")

            # 早停检查
            if self.patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break

        logger.info("-" * 80)
        logger.info(f"Training completed! Best val_acc: {self.best_val_acc:.2f}%")

        # 保存最终模型
        self.save_checkpoint("final_model.pth")

        # 保存训练指标
        metrics_path = Path(self.config.output_dir) / "training_metrics.json"
        self.metrics.save(str(metrics_path))
        logger.info(f"Metrics saved to {metrics_path}")

        return self.metrics

    def evaluate(self, loader: Optional[DataLoader] = None) -> Dict[str, float]:
        """
        在测试集上评估模型

        Returns:
            评估指标字典
        """
        loader = loader or self.test_loader
        if loader is None:
            raise ValueError("No test loader provided")

        logger.info("Evaluating on test set...")
        test_loss, test_acc, all_preds, all_targets = self.validate(loader)

        # 计算更多指标
        from sklearn.metrics import (
            classification_report, confusion_matrix,
            f1_score, precision_score, recall_score
        )

        f1 = f1_score(all_targets, all_preds, average='weighted')
        precision = precision_score(all_targets, all_preds, average='weighted')
        recall = recall_score(all_targets, all_preds, average='weighted')

        logger.info(f"Test Loss: {test_loss:.4f}")
        logger.info(f"Test Accuracy: {test_acc:.2f}%")
        logger.info(f"F1 Score: {f1:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info("\nClassification Report:")
        logger.info("\n" + classification_report(all_targets, all_preds, target_names=["normal", "benign", "malignant"]))

        return {
            'test_loss': test_loss,
            'test_acc': test_acc,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'predictions': all_preds,
            'targets': all_targets,
        }

    def save_checkpoint(self, filename: str):
        """保存模型检查点 (含完整训练状态，支持精确恢复)"""
        checkpoint_path = Path(self.config.output_dir) / filename
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
            'patience_counter': self.patience_counter,
            'config': self.config,
            'rng_states': {
                'python': random.getstate(),
                'numpy': np.random.get_state(),
                'torch': torch.get_rng_state(),
                'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }

        # 保存调度器状态 (如果存在)
        if self.scheduler is not None:
            state['scheduler_state_dict'] = self.scheduler.state_dict()

        # 保存 AMP scaler 状态 (如果存在)
        if self.scaler is not None:
            state['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(state, checkpoint_path)
        logger.debug(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """加载模型检查点 (恢复完整训练状态)"""
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        # weights_only=False: 需要恢复完整对象(optimizer/rng state等)
        # 仅加载来自可信来源的检查点
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        except TypeError:
            # PyTorch < 2.0 不支持 weights_only 参数
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.patience_counter = checkpoint.get('patience_counter', 0)

        # 恢复调度器状态
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # 恢复 AMP scaler 状态
        if 'scaler_state_dict' in checkpoint and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        # 恢复随机状态 (支持精确复现)
        rng_states = checkpoint.get('rng_states', {})
        if rng_states.get('python'):
            random.setstate(rng_states['python'])
        if rng_states.get('numpy') is not None:
            np.random.set_state(rng_states['numpy'])
        if rng_states.get('torch') is not None:
            torch.set_rng_state(rng_states['torch'])
        if rng_states.get('torch_cuda') is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng_states['torch_cuda'])

        logger.info(f"Resumed from epoch {self.current_epoch}, best_val_acc: {self.best_val_acc:.2f}%")


# =============================================================================
# 便捷函数
# =============================================================================

def create_trainer(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: Optional[DataLoader] = None,
    **config_overrides
) -> Trainer:
    """
    便捷创建训练器的工厂函数

    Args:
        model: 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        test_loader: 测试数据加载器（可选）
        **config_overrides: 配置覆盖参数

    Returns:
        Trainer实例
    """
    config = TrainingConfig()
    for key, value in config_overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == "__main__":
    print("Testing training module...")
    print("=" * 60)

    # 测试配置创建
    print("\n1. Testing TrainingConfig...")
    config = TrainingConfig(
        num_epochs=10,
        batch_size=16,
        learning_rate=1e-4,
        transformer_lr=5e-4,
    )
    print(f"  ✓ Config created: epochs={config.num_epochs}, batch_size={config.batch_size}")
    print(f"  ✓ CNN LR: {config.learning_rate}, Transformer LR: {config.transformer_lr}")

    # 测试指标记录
    print("\n2. Testing TrainingMetrics...")
    metrics = TrainingMetrics()
    for i in range(5):
        metrics.train_losses.append(1.0 - i * 0.1)
        metrics.train_accs.append(50 + i * 5)
        metrics.val_losses.append(1.1 - i * 0.1)
        metrics.val_accs.append(48 + i * 5)

    best_epoch, best_acc = metrics.get_best_epoch("val_acc", "max")
    print(f"  ✓ Best epoch: {best_epoch}, Best val_acc: {best_acc:.2f}%")

    # 测试保存/加载
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        temp_path = f.name
    metrics.save(temp_path)
    loaded_metrics = TrainingMetrics.load(temp_path)
    print(f"  ✓ Metrics saved and loaded successfully")
    import os
    os.unlink(temp_path)

    # 测试Trainer创建（需要模型和数据）
    print("\n3. Testing Trainer initialization...")
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from model import HybridModel

    # 创建简单模型
    model = HybridModel(num_classes=3, model_dim=128, nhead=4, num_layers=2)

    # 创建模拟数据加载器
    from torch.utils.data import TensorDataset
    dummy_data = torch.randn(32, 3, 224, 224)
    dummy_labels = torch.randint(0, 3, (32,))
    dummy_dataset = TensorDataset(dummy_data, dummy_labels)
    dummy_loader = torch.utils.data.DataLoader(
        dummy_dataset, batch_size=8, shuffle=True
    )

    # 创建训练器
    config = TrainingConfig(
        num_epochs=2,
        batch_size=8,
        use_amp=False,  # 测试中关闭AMP
        output_dir="outputs/test_checkpoints",
    )

    trainer = Trainer(
        model=model,
        config=config,
        train_loader=dummy_loader,
        val_loader=dummy_loader,
    )
    print(f"  ✓ Trainer created successfully")
    print(f"  ✓ Device: {trainer.device}")
    print(f"  ✓ Optimizer: {type(trainer.optimizer).__name__}")
    print(f"  ✓ Scheduler: {type(trainer.scheduler).__name__ if trainer.scheduler else 'None'}")

    # 测试单epoch训练
    print("\n4. Testing training loop...")
    train_loss, train_acc = trainer.train_epoch()
    print(f"  ✓ Train epoch completed: loss={train_loss:.4f}, acc={train_acc:.2f}%")

    val_loss, val_acc, _, _ = trainer.validate(dummy_loader)
    print(f"  ✓ Validation completed: loss={val_loss:.4f}, acc={val_acc:.2f}%")

    # 测试检查点保存/加载
    print("\n5. Testing checkpoint save/load...")
    trainer.save_checkpoint("test_checkpoint.pth")
    print(f"  ✓ Checkpoint saved")

    trainer.load_checkpoint("outputs/test_checkpoints/test_checkpoint.pth")
    print(f"  ✓ Checkpoint loaded")

    # 清理
    import shutil
    shutil.rmtree("outputs/test_checkpoints", ignore_errors=True)

    print("\n" + "=" * 60)
    print("All training module tests passed!")
    print("=" * 60)
