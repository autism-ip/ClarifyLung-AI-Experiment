# =============================================================================
# Grad-CAM Heatmap Generation Module
# =============================================================================
"""
[INPUT]: torch, gradcam, cv2, numpy
[OUTPUT]: GradCAMVisualizer, overlay_heatmap
[POS]: experiments/visualization/ Grad-CAM heatmap generation
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import numpy as np
import torch
import cv2
from typing import Optional

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


# =============================================================================
# Grad-CAM Visualizer
# =============================================================================
class GradCAMVisualizer:
    """
    Grad-CAM可视化工具类
    用于生成CNN层的类激活热力图

    [功能]:
    - 提取CNN层的梯度加权激活映射
    - 生成任意目标类别的热力图
    - 支持模型可解释性分析
    """

    def __init__(self, model: torch.nn.Module):
        """
        初始化Grad-CAM可视化器

        Args:
            model: PyTorch模型，自动检测最后一个Conv2d层
        """
        self.model = model
        self.model.eval()

        # 自动查找最后一个卷积层
        self.target_layer = self._find_target_layer()
        self.gradcam = GradCAM(model=self.model, target_layers=[self.target_layer])

    def _find_target_layer(self) -> torch.nn.Module:
        """
        查找模型中最后一个卷积层

        Returns:
            最后一个Conv2d层

        Raises:
            ValueError: 未找到卷积层
        """
        target_layer = None
        for module in self.model.modules():
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module

        if target_layer is None:
            raise ValueError("Model does not contain any Conv2d layer")
        return target_layer

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_category: Optional[int] = None
    ) -> np.ndarray:
        """
        生成Grad-CAM热力图

        Args:
            input_tensor: 输入图像张量，形状 (B, C, H, W)
            target_category: 目标类别索引，None则使用预测类别

        Returns:
            热力图 numpy数组，形状 (H, W)，值范围 [0, 1]
        """
        # 确保输入是批次形式
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)

        # 设置目标
        if target_category is not None:
            targets = [ClassifierOutputTarget(target_category)]
        else:
            targets = None

        # 生成热力图
        grayscale_cam = self.gradcam(input_tensor=input_tensor, targets=targets)

        # 取第一个样本（batch size = 1）
        heatmap = grayscale_cam[0]

        return heatmap

    def __del__(self):
        """清理资源"""
        if hasattr(self, 'gradcam'):
            self.gradcam = None


# =============================================================================
# Heatmap Overlay Function
# =============================================================================
def overlay_heatmap(
    heatmap: np.ndarray,
    image: np.ndarray,
    alpha: float = 0.4
) -> np.ndarray:
    """
    将热力图叠加到原始图像上

    [算法]:
    1. 将热力图归一化到 [0, 255] 并转换为BGR色彩空间
    2. 使用双线性插值将热力图缩放到图像尺寸
    3. 使用加权叠加: result = image * (1 - alpha) + heatmap * alpha

    Args:
        heatmap: 热力图数组，形状 (H, W)，值范围 [0, 1]
        image: 原始RGB图像，形状 (H, W, 3)，dtype=uint8
        alpha: 叠加透明度，值越大热力图越明显

    Returns:
        叠加后的uint8图像

    [注意]:
        热力图尺寸必须与图像尺寸匹配，或为1xHxW的形式
    """
    # 确保热力图是2D数组
    if heatmap.ndim == 3:
        heatmap = heatmap.squeeze()

    # 获取图像尺寸
    h, w = image.shape[:2]

    # 调整热力图尺寸
    if heatmap.shape != (h, w):
        heatmap = cv2.resize(heatmap, (w, h))

    # 归一化热力图到 [0, 255]
    heatmap_uint8 = np.uint8(255 * (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8))

    # 应用色彩映射 (Jet: 蓝->红)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # 转换为RGB (OpenCV是BGR)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # 叠加
    overlay = np.uint8(image * (1 - alpha) + heatmap_color * alpha)

    return overlay