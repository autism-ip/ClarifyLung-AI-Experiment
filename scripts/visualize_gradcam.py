#!/usr/bin/env python3
"""
Grad-CAM 可视化 CLI
对单张图像生成 Grad-CAM 热力图，查看模型关注的区域。

用法:
    # 使用预测类别生成热力图
    python scripts/visualize_gradcam.py \\
        --image datasets/IQ-OTHNCCD/Normal cases/normal_001.png \\
        --checkpoint outputs/checkpoints/best_model.pth \\
        --output outputs/figures/gradcam_normal.png

    # 指定目标类别（如 malignant=2）
    python scripts/visualize_gradcam.py \
        --image path/to/image.png \
        --checkpoint outputs/checkpoints/best_model.pth \
        --target-class 2 \
        --output outputs/figures/gradcam_malignant.png

    # 自定义模型尺寸（如果 checkpoint 是用小模型训练的）
    python scripts/visualize_gradcam.py \
        --image path/to/image.png \
        --checkpoint outputs/checkpoints/best_model.pth \
        --model-dim 128 --num-layers 2 \
        --output outputs/figures/gradcam.png

输出:
    叠加热力图后的 RGB 图像，保存为 PNG
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use('Agg')

from model import HybridModel
from experiments.visualization import GradCAMVisualizer, overlay_heatmap
from configs import CLASS_NAMES


def load_image(image_path: str, image_size: int = 224) -> torch.Tensor:
    """加载并预处理单张图像"""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert('RGB')
    return transform(img).unsqueeze(0), np.array(img)


def main():
    parser = argparse.ArgumentParser(description='Grad-CAM 可视化')
    parser.add_argument('--image', type=str, required=True, help='输入图像路径')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型权重路径 (.pth)')
    parser.add_argument('--output', type=str, required=True, help='输出热力图路径 (.png)')
    parser.add_argument('--target-class', type=int, default=None,
                        help='目标类别索引 (0=normal, 1=benign, 2=malignant)。不指定则使用预测类别')
    parser.add_argument('--alpha', type=float, default=0.5, help='热力图叠加透明度 (0.0-1.0)')
    parser.add_argument('--image-size', type=int, default=224, help='输入图像尺寸')
    parser.add_argument('--model-dim', type=int, default=256, help='模型维度')
    parser.add_argument('--nhead', type=int, default=8, help='注意力头数')
    parser.add_argument('--num-layers', type=int, default=4, help='Transformer层数')
    parser.add_argument('--device', type=str, default='auto', help='计算设备 (auto/cpu/cuda)')

    args = parser.parse_args()

    # 设备选择
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"[INFO] 使用设备: {device}")

    # 加载模型
    print(f"[INFO] 加载模型: {args.checkpoint}")
    model = HybridModel(
        num_classes=3,
        model_dim=args.model_dim,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=0.1,
    )
    checkpoint = torch.load(args.checkpoint, map_location=device)
    # 兼容直接保存 state_dict 和 Trainer 保存的 dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    print("[INFO] 模型加载完成")

    # 加载图像
    print(f"[INFO] 加载图像: {args.image}")
    input_tensor, original_img = load_image(args.image, args.image_size)
    input_tensor = input_tensor.to(device)

    # 预测类别（如果未指定 target-class）
    if args.target_class is None:
        with torch.no_grad():
            output = model(input_tensor)
            pred_class = output.argmax(dim=1).item()
        target_category = pred_class
        print(f"[INFO] 未指定目标类别，使用预测类别: {CLASS_NAMES[pred_class]} ({pred_class})")
    else:
        target_category = args.target_class
        print(f"[INFO] 目标类别: {CLASS_NAMES[target_category]} ({target_category})")

    # 生成 Grad-CAM
    print("[INFO] 生成 Grad-CAM 热力图...")
    visualizer = GradCAMVisualizer(model)
    heatmap = visualizer.generate_heatmap(input_tensor, target_category=target_category)

    # 将原始图像缩放到模型输入尺寸用于叠加
    original_resized = np.array(
        Image.fromarray(original_img).resize((args.image_size, args.image_size))
    )
    overlay = overlay_heatmap(heatmap, original_resized, alpha=args.alpha)

    # 保存
    import cv2
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)
    print(f"[OK] 热力图已保存: {out_path}")


if __name__ == '__main__':
    main()
