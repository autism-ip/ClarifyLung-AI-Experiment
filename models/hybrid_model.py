"""
 * [INPUT]: 依赖 models.components 的 feature_extractor, gating, transformer, cross_attention, classification
 * [OUTPUT]: 对外提供 HybridModel
 * [POS]: models/ 的核心模型组装器，组合所有组件构成完整 CNN-Transformer 混合模型
 * [PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md

 支持预训练权重加载: CNN (ResNet50) 和 Transformer (ViT-B/16) 均可加载预训练权重
"""

import torch
import torch.nn as nn
import torchvision.models as models

from .components.feature_extractor import MutilScaleFeatureExtractor
from .components.gating import GatingMechanism
from .components.transformer import ImagePatchExtractor, TransformerEncoder
from .components.cross_attention import CrossAttention
from .components.classification import ClassificationHead


CLASS_NAMES = ["normal", "benign", "malignant"]


def _get_layer_channels(backbone):
    """根据 backbone 名称返回各层通道数映射"""
    if backbone == 'resnet18':
        return {'layer1': 64, 'layer2': 128, 'layer3': 256, 'layer4': 512}
    # 默认 resnet50
    return {'layer1': 256, 'layer2': 512, 'layer3': 1024, 'layer4': 2048}


class HybridModel(nn.Module):
    """CNN-Transformer 混合模型：多尺度 CNN 特征 + Transformer 全局建模 + 交叉注意力融合"""

    def __init__(
        self,
        backbone_model=None,
        feature_layers=None,
        gate_type='se',
        model_dim=768,  # 改为768以匹配预训练ViT-B/16的维度
        nhead=12,       # 改为12以匹配预训练ViT-B/16
        num_layers=12,  # 改为12以匹配预训练ViT-B/16
        dropout=0.1,
        num_classes=3,
        backbone_name='resnet50',
        use_transformer=True,
        use_cross_attention=True,
        pretrained=True  # 默认使用预训练权重
    ):
        super().__init__()
        if feature_layers is None:
            feature_layers = ['layer3', 'layer4']
        if backbone_model is None:
            if pretrained:
                backbone_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            else:
                backbone_model = models.resnet50(weights=None)

        self.use_transformer = use_transformer
        self.use_cross_attention = use_cross_attention

        self.backbone = backbone_model
        self.mutil_scale_extractor = MutilScaleFeatureExtractor(
            model=self.backbone,
            feature_layers=feature_layers,
            use_multi_scale=True
        )

        layer_channels = _get_layer_channels(backbone_name)
        self.fused_channels = sum(layer_channels[l] for l in feature_layers)

        self.gate_mechanism = GatingMechanism(
            gate_type=gate_type,
            backbone=backbone_name,
            feature_layers=feature_layers
        )

        # Transformer patch stream
        if use_transformer:
            self.image_patch_extractor = ImagePatchExtractor(
                patch_size=16,
                model_dim=model_dim,
                image_size=224
            )
            self.transformer_encoder = TransformerEncoder(
                model_dim=model_dim,
                dropout=dropout,
                nhead=nhead,
                dim_feedforward=3072,
                num_layers=num_layers
            )
            
            # 加载预训练ViT-B/16权重到Transformer部分
            if pretrained:
                try:
                    print("[INFO] 加载预训练ViT-B/16权重到Transformer...")
                    vit_pretrained = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
                    
                    # 1. 加载patch embedding权重
                    # ViT的patch_embed.proj: [768, 3, 16, 16]
                    # 我们的projection: nn.Linear(768, 768)
                    with torch.no_grad():
                        # 将卷积权重转换为线性层权重
                        vit_proj_weight = vit_pretrained.conv_proj.weight  # [768, 3, 16, 16]
                        vit_proj_bias = vit_pretrained.conv_proj.bias    # [768]
                        # reshape: [768, 3*16*16] = [768, 768]
                        vit_proj_weight_flat = vit_proj_weight.view(vit_proj_weight.size(0), -1)
                        
                        if model_dim == 768:
                            self.image_patch_extractor.projection.weight.copy_(vit_proj_weight_flat)
                            if vit_proj_bias is not None:
                                self.image_patch_extractor.projection.bias.copy_(vit_proj_bias)
                        else:
                            print(f"[WARNING] 模型维度{model_dim}与ViT-B/16维度768不匹配，跳过patch embedding权重加载")
                    
                    # 2. 加载Transformer encoder权重
                    vit_state = vit_pretrained.encoder.state_dict()
                    
                    # 创建权重映射：torchvision ViT命名 -> 标准TransformerEncoder命名
                    mapped_state = {}
                    for key, value in vit_state.items():
                        # 映射层索引: encoder_layer_0 -> 0
                        if key.startswith('layers.encoder_layer_'):
                            parts = key.split('.')
                            layer_idx = parts[1].replace('encoder_layer_', '')
                            new_key = f"layers.{layer_idx}"
                            
                            # 映射子模块名称
                            remaining = '.'.join(parts[2:])
                            if remaining.startswith('self_attention.'):
                                remaining = remaining.replace('self_attention.', 'self_attn.')
                            elif remaining.startswith('ln_1'):
                                remaining = remaining.replace('ln_1', 'norm1')
                            elif remaining.startswith('ln_2'):
                                remaining = remaining.replace('ln_2', 'norm2')
                            elif remaining.startswith('mlp.'):
                                # mlp.0 -> linear1, mlp.3 -> linear2
                                remaining = remaining.replace('mlp.0.', 'linear1.').replace('mlp.3.', 'linear2.')
                            
                            mapped_state[f"{new_key}.{remaining}"] = value
                        elif key == 'pos_embedding':
                            # 跳过位置编码，因为我们使用正弦位置编码
                            continue
                        elif key.startswith('ln.'):
                            # 最后的LayerNorm，映射到我们的结构
                            mapped_state[f"norm.{key[3:]}"] = value
                    
                    # 检查层数是否匹配
                    vit_layers = len(vit_pretrained.encoder.layers)
                    our_layers = num_layers
                    
                    if model_dim == 768:
                        # 过滤层数
                        filtered_state = {}
                        for key, value in mapped_state.items():
                            if key.startswith('layers.'):
                                try:
                                    layer_idx = int(key.split('.')[1])
                                    if layer_idx < our_layers:
                                        filtered_state[key] = value
                                except ValueError:
                                    # 非数字层索引（如norm等），直接保留
                                    filtered_state[key] = value
                            else:
                                filtered_state[key] = value
                        
                        missing, unexpected = self.transformer_encoder.transformer.load_state_dict(filtered_state, strict=False)
                        if missing:
                            print(f"[WARNING] 缺失权重: {list(missing)[:5]}...")
                        if unexpected:
                            print(f"[WARNING] 多余权重: {list(unexpected)[:5]}...")
                        print(f"[INFO] 成功加载ViT-B/16前{min(vit_layers, our_layers)}层Transformer权重")
                    else:
                        print(f"[WARNING] 模型维度{model_dim}与ViT-B/16维度768不匹配，无法加载预训练Transformer权重")
                        
                except Exception as e:
                    print(f"[WARNING] 加载预训练ViT权重失败: {e}")
                    print("[INFO] 继续使用随机初始化")
        
        self.cnn_to_seq = nn.Linear(self.fused_channels, model_dim)

        self.cnn_to_seq = nn.Linear(self.fused_channels, model_dim)

        if use_cross_attention:
            self.cross_attention = CrossAttention(
                num_layers=2,
                nhead=nhead,
                model_dim=model_dim,
                dropout=dropout,
                cnn_in_channels=self.fused_channels
            )

        classifier_input = 2 * model_dim if (use_transformer or use_cross_attention) else model_dim
        self.classifier = ClassificationHead(
            input_dim=classifier_input,
            num_classes=num_classes,
            dropout=dropout
        )

    def forward(self, x):
        # x: [B, 3, H, W]
        B = x.size(0)

        # CNN multi-scale -> [B, fused_channels, Hf, Wf]
        scnn = self.mutil_scale_extractor(x)
        scnn = self.gate_mechanism(scnn)  # gated [B, C, H, W]

        # Convert CNN map -> sequence [B, T_cnn, C] then project to model_dim
        B, C, Hf, Wf = scnn.shape
        scnn_seq = scnn.view(B, C, Hf * Wf).permute(0, 2, 1)  # [B, T_cnn, C]
        scnn_seq_proj = self.cnn_to_seq(scnn_seq)             # [B, T_cnn, model_dim]

        # 纯CNN路径：无Transformer也无交叉注意力
        if not self.use_transformer and not self.use_cross_attention:
            pooled_cnn = scnn_seq_proj.mean(dim=1)  # [B, model_dim]
            return self.classifier(pooled_cnn)

        # Transformer path
        stf = self.image_patch_extractor(x)   # [B, T_patch, model_dim]
        stf = self.transformer_encoder(stf)   # [B, T_patch, model_dim]

        # 无交叉注意力：独立池化后拼接
        if not self.use_cross_attention:
            pooled_cnn = scnn_seq_proj.mean(dim=1)  # [B, model_dim]
            pooled_tf = stf.mean(dim=1)             # [B, model_dim]
            fused = torch.cat([pooled_cnn, pooled_tf], dim=1)  # [B, 2*model_dim]
            return self.classifier(fused)

        # Cross attention: treat queries as sequences
        attncnn, _ = self.cross_attention(scnn_seq_proj, stf, stf)
        attntf, _ = self.cross_attention(stf, scnn_seq_proj, scnn_seq_proj)

        # Pool each branch (global mean over sequence)
        pooled_cnn = attncnn.mean(dim=1)   # [B, model_dim]
        pooled_tf = attntf.mean(dim=1)     # [B, model_dim]

        fused = torch.cat([pooled_cnn, pooled_tf], dim=1)  # [B, 2*model_dim]
        out = self.classifier(fused)                       # [B, num_classes]
        return out
