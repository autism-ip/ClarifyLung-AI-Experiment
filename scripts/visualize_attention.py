#!/usr/bin/env python3
"""
Transformer Attention 可视化 CLI
对单张图像提取并可视化 Transformer 的注意力权重矩阵。

用法:
    # 可视化指定层的注意力（默认第0层）
    python scripts/visualize_attention.py \\
        --image datasets/IQ-OTHNCCD/Normal cases/normal_001.png \\
        --checkpoint outputs/checkpoints/best_model.pth \\
        --output outputs/figures/attention_normal.png

    # 可视化第2层注意力
    python scripts/visualize_attention.py \\
        --image path/to/image.png \\
        --checkpoint outputs/checkpoints/best_model.pth \\
        --layer-idx 2 \\
        --output outputs/figures/attention_layer2.png

    # 可视化所有注意力层并保存到目录
    python scripts/visualize_attention.py \\
        --image path/to/image.png \\
        --checkpoint outputs/checkpoints/best_model.pth \\
        --all-layers \\
        --output outputs/figures/attention_all

    # 自定义模型尺寸
    python scripts/visualize_attention.py \\
        --image path/to/image.png \\
        --checkpoint outputs/checkpoints/best_model.pth \\
        --model-dim 128 --num-layers 2 \\
        --output outputs/figures/attention.png

输出:
    注意力权重热力图，保存为 PNG
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
import matplotlib.pyplot as plt

from model import HybridModel
from experiments.visualization import AttentionVisualizer, visualize_attention, visualize_multihead_attention


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
    parser = argparse.ArgumentParser(description='Transformer Attention 可视化')
    parser.add_argument('--image', type=str, required=True, help='输入图像路径')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型权重路径 (.pth)')
    parser.add_argument('--output', type=str, required=True, help='输出路径 (.png 或目录)')
    parser.add_argument('--layer-idx', type=int, default=0, help='要可视化的注意力层索引')
    parser.add_argument('--all-layers', action='store_true', help='可视化所有注意力层，输出到 --output 指定的目录')
    parser.add_argument('--multi-head', action='store_true', help='分别可视化每个注意力头')
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
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    print("[INFO] 模型加载完成")

    # 检查是否有注意力层
    visualizer = AttentionVisualizer(model)
    if not visualizer.has_attention():
        print("[ERROR] 模型不包含 MultiheadAttention 层，无法提取注意力权重")
        sys.exit(1)
    print(f"[INFO] 发现 {len(visualizer.attention_layers)} 个注意力层")

    # 加载图像
    print(f"[INFO] 加载图像: {args.image}")
    input_tensor, original_img = load_image(args.image, args.image_size)
    input_tensor = input_tensor.to(device)

    # 可视化
    out_path = Path(args.output)

    if args.all_layers:
        # 所有层
        out_path.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] 可视化所有注意力层，输出到: {out_path}")

        all_attention = visualizer.extract_all_attention(input_tensor)
        for i, layer_attention in enumerate(all_attention):
            if not layer_attention:
                continue
            attn_matrix = layer_attention[0]
            layer_path = out_path / f"attention_layer_{i}.png"
            visualize_attention(attn_matrix, save_path=str(layer_path))
            print(f"  [OK] Layer {i}: {layer_path}")

        # 生成一个汇总图
        fig, axes = plt.subplots(1, len(all_attention), figsize=(4 * len(all_attention), 4))
        if len(all_attention) == 1:
            axes = [axes]
        for i, layer_attention in enumerate(all_attention):
            if not layer_attention:
                continue
            axes[i].imshow(layer_attention[0], cmap='viridis', aspect='auto')
            axes[i].set_title(f'Layer {i}')
            axes[i].set_xlabel('Key Position')
            axes[i].set_ylabel('Query Position')
        plt.tight_layout()
        summary_path = out_path / "attention_summary.png"
        plt.savefig(str(summary_path), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[OK] 汇总图: {summary_path}")

    elif args.multi_head:
        # 多头分开可视化
        out_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] 提取第 {args.layer_idx} 层多头注意力...")

        # extract_attention 返回的是平均后的 (seq_len, seq_len)
        # 需要 hook 中捕获原始多头的 weights
        # 由于 extract_attention 已经做了 mean，multi-head 需要额外处理
        # 这里用 extract_all_attention 的底层逻辑获取 4D weights
        attention_weights = visualizer.extract_attention(input_tensor, layer_idx=args.layer_idx)
        attn_matrix = attention_weights[0]

        # 尝试从模型重新获取多头权重
        layer_name = visualizer.attention_layers[args.layer_idx]
        target_layer = dict(model.named_modules())[layer_name]

        multihead_weights = []

        def hook_fn(module, input, output):
            with torch.no_grad():
                _, attn_w = module(input[0], input[0], input[0], need_weights=True, average_attn_weights=False)
            multihead_weights.append(attn_w[0].cpu().numpy())

        handle = target_layer.register_forward_hook(hook_fn)
        with torch.no_grad():
            _ = model(input_tensor)
        handle.remove()

        if multihead_weights:
            weights = multihead_weights[0]  # (num_heads, seq_len, seq_len)
            visualize_multihead_attention(
                weights, num_heads=weights.shape[0],
                save_path=str(out_path)
            )
            print(f"[OK] 多头注意力图已保存: {out_path}")
        else:
            # 回退到单矩阵
            visualize_attention(attn_matrix, save_path=str(out_path))
            print(f"[OK] 注意力图已保存: {out_path} (多头提取失败，回退到平均)")

    else:
        # 单层单矩阵
        out_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] 提取第 {args.layer_idx} 层注意力权重...")
        attention_weights = visualizer.extract_attention(input_tensor, layer_idx=args.layer_idx)
        attn_matrix = attention_weights[0]
        visualize_attention(attn_matrix, save_path=str(out_path))
        print(f"[OK] 注意力图已保存: {out_path}")


if __name__ == '__main__':
    main()
