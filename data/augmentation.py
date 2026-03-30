"""
数据增强模块
[INPUT]: 原始图像、增强配置参数
[OUTPUT]: 增强后的图像、增强策略管道
[POS]: data/核心组件，被dataset.py调用构建transform pipeline
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import random
from typing import Tuple, Optional, Callable, List, Dict, Any, Union
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image, ImageEnhance, ImageFilter


# ============================================================================
# 配置类
# ============================================================================

@dataclass
class AugmentationConfig:
    """数据增强配置类"""
    # 图像尺寸
    image_size: int = 224

    # 基础增强参数
    rotation_range: Tuple[int, int] = (-15, 15)  # 旋转角度范围
    horizontal_flip: bool = True  # 水平翻转
    vertical_flip: bool = False  # 垂直翻转
    brightness_range: Tuple[float, float] = (0.8, 1.2)  # 亮度范围
    contrast_range: Tuple[float, float] = (0.8, 1.2)  # 对比度范围

    # 高级增强开关
    use_cutmix: bool = False  # 是否使用CutMix
    use_mixup: bool = False  # 是否使用MixUp
    use_random_erasing: bool = False  # 是否使用Random Erasing

    # CutMix/MixUp参数
    cutmix_alpha: float = 1.0  # CutMix beta分布参数
    mixup_alpha: float = 0.4  # MixUp beta分布参数
    cutmix_prob: float = 0.5  # CutMix应用概率
    mixup_prob: float = 0.5  # MixUp应用概率

    # Random Erasing参数
    random_erasing_prob: float = 0.5
    random_erasing_scale: Tuple[float, float] = (0.02, 0.33)
    random_erasing_ratio: Tuple[float, float] = (0.3, 3.3)

    # 标准化参数(ImageNet统计)
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)


# ============================================================================
# 基础增强变换
# ============================================================================

def get_train_augmentation(
    image_size: int = 224,
    config: Optional[AugmentationConfig] = None,
) -> T.Compose:
    """
    获取训练集增强变换管道

    Args:
        image_size: 输出图像尺寸
        config: 增强配置，None则使用默认配置

    Returns:
        torchvision.transforms.Compose对象
    """
    if config is None:
        config = AugmentationConfig(image_size=image_size)

    transforms = [
        # 1. 调整图像大小
        T.Resize((image_size, image_size)),

        # 2. 随机旋转
        T.RandomRotation(
            degrees=config.rotation_range,
            fill=0,
        ),

        # 3. 随机翻转
        T.RandomHorizontalFlip(p=0.5 if config.horizontal_flip else 0.0),
        T.RandomVerticalFlip(p=0.5 if config.vertical_flip else 0.0),

        # 4. 颜色抖动(亮度/对比度/饱和度/色调)
        T.ColorJitter(
            brightness=config.brightness_range,
            contrast=config.contrast_range,
            saturation=(0.8, 1.2),
            hue=0.05,
        ),

        # 5. 随机仿射变换(平移/缩放/剪切)
        T.RandomAffine(
            degrees=0,  # 已在RandomRotation中处理
            translate=(0.1, 0.1),  # 平移10%
            scale=(0.9, 1.1),  # 缩放
            shear=(-5, 5),  # 剪切
            fill=0,
        ),

        # 6. 转换为张量
        T.ToTensor(),

        # 7. 标准化(ImageNet统计)
        T.Normalize(
            mean=config.normalize_mean,
            std=config.normalize_std,
        ),
    ]

    # 8. 可选: Random Erasing
    if config.use_random_erasing:
        transforms.append(
            T.RandomErasing(
                p=config.random_erasing_prob,
                scale=config.random_erasing_scale,
                ratio=config.random_erasing_ratio,
                value='random',
            )
        )

    return T.Compose(transforms)


def get_val_augmentation(
    image_size: int = 224,
    config: Optional[AugmentationConfig] = None,
) -> T.Compose:
    """
    获取验证/测试集变换管道(无增强)

    Args:
        image_size: 输出图像尺寸
        config: 配置对象

    Returns:
        torchvision.transforms.Compose对象
    """
    if config is None:
        config = AugmentationConfig(image_size=image_size)

    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(
            mean=config.normalize_mean,
            std=config.normalize_std,
        ),
    ])


# ============================================================================
# 高级增强: CutMix 和 MixUp
# ============================================================================

class CutMix:
    """
    CutMix增强实现

    参考论文: "CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features"
    """

    def __init__(self, alpha: float = 1.0, prob: float = 0.5):
        """
        Args:
            alpha: Beta分布参数
            prob: 应用CutMix的概率
        """
        self.alpha = alpha
        self.prob = prob

    def __call__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        应用CutMix

        Args:
            images: 图像批次 [B, C, H, W]
            labels: 标签 [B]

        Returns:
            mixed_images: 混合后的图像
            labels_a: 原始标签
            labels_b: 混合来源标签
            lam: 混合比例
        """
        if torch.rand(1).item() > self.prob:
            # 不应用CutMix
            return images, labels, labels, 1.0

        batch_size = images.size(0)

        # 从Beta分布采样
        lam = np.random.beta(self.alpha, self.alpha)

        # 随机打乱索引
        index = torch.randperm(batch_size).to(images.device)

        # 计算裁剪区域
        _, _, h, w = images.shape
        cut_ratio = np.sqrt(1 - lam)
        cut_h = int(h * cut_ratio)
        cut_w = int(w * cut_ratio)

        # 随机中心点
        cx = np.random.randint(h)
        cy = np.random.randint(w)

        # 计算裁剪边界
        bbx1 = np.clip(cx - cut_h // 2, 0, h)
        bby1 = np.clip(cy - cut_w // 2, 0, w)
        bbx2 = np.clip(cx + cut_h // 2, 0, h)
        bby2 = np.clip(cy + cut_w // 2, 0, w)

        # 应用CutMix
        images_mixed = images.clone()
        images_mixed[:, :, bbx1:bbx2, bby1:bby2] = images[index, :, bbx1:bbx2, bby1:bby2]

        # 调整lambda以匹配像素比例
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (h * w))

        return images_mixed, labels, labels[index], lam


class MixUp:
    """
    MixUp增强实现

    参考论文: "mixup: Beyond Empirical Risk Minimization"
    """

    def __init__(self, alpha: float = 0.4, prob: float = 0.5):
        """
        Args:
            alpha: Beta分布参数
            prob: 应用MixUp的概率
        """
        self.alpha = alpha
        self.prob = prob

    def __call__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        应用MixUp

        Args:
            images: 图像批次 [B, C, H, W]
            labels: 标签 [B]

        Returns:
            mixed_images: 混合后的图像
            labels_a: 原始标签
            labels_b: 混合来源标签
            lam: 混合比例
        """
        if torch.rand(1).item() > self.prob:
            return images, labels, labels, 1.0

        batch_size = images.size(0)

        # 从Beta分布采样
        lam = np.random.beta(self.alpha, self.alpha)

        # 随机打乱索引
        index = torch.randperm(batch_size).to(images.device)

        # 混合图像
        mixed_images = lam * images + (1 - lam) * images[index]

        return mixed_images, labels, labels[index], lam


# ============================================================================
# 增强组合器
# ============================================================================

class AdvancedAugmentation:
    """
    高级增强组合器

    将CutMix/MixUp与基础增强结合，支持概率控制
    """

    def __init__(
        self,
        use_cutmix: bool = True,
        use_mixup: bool = False,
        cutmix_alpha: float = 1.0,
        mixup_alpha: float = 0.4,
        cutmix_prob: float = 0.5,
        mixup_prob: float = 0.5,
    ):
        """
        Args:
            use_cutmix: 是否启用CutMix
            use_mixup: 是否启用MixUp
            cutmix_alpha: CutMix的Beta参数
            mixup_alpha: MixUp的Beta参数
            cutmix_prob: 应用CutMix的概率
            mixup_prob: 应用MixUp的概率
        """
        self.use_cutmix = use_cutmix
        self.use_mixup = use_mixup

        self.cutmix = CutMix(alpha=cutmix_alpha, prob=cutmix_prob) if use_cutmix else None
        self.mixup = MixUp(alpha=mixup_alpha, prob=mixup_prob) if use_mixup else None

    def __call__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        应用高级增强

        Args:
            images: 图像批次 [B, C, H, W]
            labels: 标签 [B]

        Returns:
            images: 增强后的图像
            labels_a: 原始标签 (用于混合损失计算)
            labels_b: 混合来源标签 (用于混合损失计算)
            lam: 混合比例张量 [B]，每个样本的lambda值
        """
        batch_size = images.size(0)
        device = images.device

        # 初始化lambda为1.0 (无混合)
        lam = torch.ones(batch_size, device=device)
        labels_a = labels.clone()
        labels_b = labels.clone()

        # 随机选择增强策略
        aug_choice = random.random()

        if self.cutmix and aug_choice < 0.5:
            # 应用CutMix
            images, labels_a, labels_b, lam_value = self.cutmix(images, labels)
            lam = torch.full((batch_size,), lam_value, device=device)

        elif self.mixup and aug_choice >= 0.5:
            # 应用MixUp
            images, labels_a, labels_b, lam_value = self.mixup(images, labels)
            lam = torch.full((batch_size,), lam_value, device=device)

        return images, labels_a, labels_b, lam


def apply_mixup_criterion(
    criterion: Callable,
    outputs: torch.Tensor,
    labels_a: torch.Tensor,
    labels_b: torch.Tensor,
    lam: torch.Tensor,
) -> torch.Tensor:
    """
    计算MixUp/CutMix混合损失

    公式: loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)

    Args:
        criterion: 损失函数 (如nn.CrossEntropyLoss())
        outputs: 模型输出 [B, num_classes]
        labels_a: 原始标签 [B]
        labels_b: 混合来源标签 [B]
        lam: 混合比例 [B] 或标量

    Returns:
        标量损失值
    """
    if isinstance(lam, torch.Tensor) and lam.dim() > 0:
        # 每个样本有不同的lambda
        loss_a = criterion(outputs, labels_a)
        loss_b = criterion(outputs, labels_b)
        loss = (lam * loss_a + (1 - lam) * loss_b).mean()
    else:
        # 所有样本使用相同的lambda
        loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)

    return loss


# ============================================================================
# 便捷函数
# ============================================================================

def get_train_augmentation(
    image_size: int = 224,
    advanced_aug: bool = False,
    use_cutmix: bool = False,
    use_mixup: bool = False,
) -> Union[T.Compose, Tuple[T.Compose, Optional[AdvancedAugmentation]]]:
    """
    获取训练增强管道 (便捷函数)

    Args:
        image_size: 输出图像尺寸
        advanced_aug: 是否返回高级增强器
        use_cutmix: 是否启用CutMix
        use_mixup: 是否启用MixUp

    Returns:
        如果advanced_aug=False: 返回T.Compose
        如果advanced_aug=True: 返回(T.Compose, AdvancedAugmentation或None)
    """
    config = AugmentationConfig(image_size=image_size)
    base_transform = get_train_augmentation_from_config(config)

    if not advanced_aug:
        return base_transform

    # 创建高级增强器
    adv_aug = None
    if use_cutmix or use_mixup:
        adv_aug = AdvancedAugmentation(
            use_cutmix=use_cutmix,
            use_mixup=use_mixup,
        )

    return base_transform, adv_aug


def get_val_augmentation(image_size: int = 224) -> T.Compose:
    """获取验证/测试变换 (便捷函数)"""
    return get_val_augmentation_from_config(AugmentationConfig(image_size=image_size))


# 内部实现函数
def get_train_augmentation_from_config(config: AugmentationConfig) -> T.Compose:
    """从配置构建训练变换"""
    return T.Compose([
        T.Resize((config.image_size, config.image_size)),
        T.RandomRotation(degrees=config.rotation_range, fill=0),
        T.RandomHorizontalFlip(p=0.5 if config.horizontal_flip else 0.0),
        T.RandomVerticalFlip(p=0.5 if config.vertical_flip else 0.0),
        T.ColorJitter(
            brightness=config.brightness_range,
            contrast=config.contrast_range,
            saturation=(0.8, 1.2),
            hue=0.05,
        ),
        T.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
            shear=(-5, 5),
            fill=0,
        ),
        T.ToTensor(),
        T.Normalize(mean=config.normalize_mean, std=config.normalize_std),
    ])


def get_val_augmentation_from_config(config: AugmentationConfig) -> T.Compose:
    """从配置构建验证变换"""
    return T.Compose([
        T.Resize((config.image_size, config.image_size)),
        T.ToTensor(),
        T.Normalize(mean=config.normalize_mean, std=config.normalize_std),
    ])


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    print("Testing augmentation module...")

    # 测试基础变换
    print("\n1. Testing basic augmentations...")
    train_aug = get_train_augmentation(image_size=224)
    val_aug = get_val_augmentation(image_size=224)
    print(f"  Train transform: {len(train_aug.transforms)} operations")
    print(f"  Val transform: {len(val_aug.transforms)} operations")

    # 测试图像变换
    from PIL import Image
    test_img = Image.new('RGB', (300, 300), color=(100, 150, 200))

    print("\n2. Testing image transformations...")
    augmented = train_aug(test_img)
    print(f"  Input size: (300, 300)")
    print(f"  Output shape: {augmented.shape}")
    print(f"  Output dtype: {augmented.dtype}")
    print(f"  Value range: [{augmented.min():.3f}, {augmented.max():.3f}]")

    # 测试高级增强
    print("\n3. Testing advanced augmentations (CutMix/MixUp)...")

    # 创建模拟批次
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224)
    labels = torch.tensor([0, 1, 2, 0])

    # 测试CutMix
    cutmix = CutMix(alpha=1.0, prob=1.0)  # 强制应用
    mixed_img, labels_a, labels_b, lam = cutmix(images.clone(), labels)
    print(f"  CutMix applied: lam={lam:.3f}")
    print(f"  Labels a: {labels_a.tolist()}")
    print(f"  Labels b: {labels_b.tolist()}")

    # 测试MixUp
    mixup = MixUp(alpha=0.4, prob=1.0)  # 强制应用
    mixed_img, labels_a, labels_b, lam = mixup(images.clone(), labels)
    print(f"  MixUp applied: lam={lam:.3f}")

    # 测试AdvancedAugmentation
    print("\n4. Testing AdvancedAugmentation wrapper...")
    adv_aug = AdvancedAugmentation(use_cutmix=True, use_mixup=True)
    mixed_img, labels_a, labels_b, lam_tensor = adv_aug(images, labels)
    print(f"  Output shapes: images={mixed_img.shape}, labels_a={labels_a.shape}")
    print(f"  Lambda values: {lam_tensor.tolist()}")

    # 测试损失计算
    print("\n5. Testing loss computation with MixUp/CutMix...")
    import torch.nn as nn
    criterion = nn.CrossEntropyLoss()

    # 模拟模型输出
    outputs = torch.randn(batch_size, 3)

    # 计算混合损失
    mixed_loss = apply_mixup_criterion(
        criterion, outputs, labels_a, labels_b, lam_tensor[0]
    )
    print(f"  Mixed loss: {mixed_loss.item():.4f}")

    # 对比普通损失
    normal_loss = criterion(outputs, labels)
    print(f"  Normal loss: {normal_loss.item():.4f}")

    print("\n" + "=" * 50)
    print("All augmentation tests passed!")
    print("=" * 50)
