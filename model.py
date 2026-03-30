import torch
import torch.nn as nn
import torchvision.models as models
from functools import reduce
from collections import OrderedDict
import math
import torch.nn.functional as F

CLASS_NAMES = ["normal", "malignant", "benign"]

# 定义模型组件
# 相关参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backbone = 'resnet50'
feature_layers = ['layer3', 'layer4']   # 使用哪些 backbone 层
target_spatial_size = (7, 7)
gate_type = 'se'
model_dim = 512
nhead = 8
num_layers = 6
dropout = 0.1
num_classes = len(CLASS_NAMES)
baseline_model = models.resnet50(pretrained=True)
# 特征融合模块
class FeatureFusion(nn.Module):
    def __init__(self, target_spatial_size=(7,7)):
        super().__init__()
        self.target_spatial_size = target_spatial_size

    def forward(self, feature_list):
        # feature_list: list of [B, C_i, H_i, W_i]
        resized = []
        for f in feature_list:
            resized.append(F.interpolate(f, size=self.target_spatial_size, mode='bilinear', align_corners=False))
        # concat on channel dim
        return torch.cat(resized, dim=1)  # [B, sum(C_i), H, W]

# CNN多尺度特征提取器
class MutilScaleFeatureExtractor(nn.Module):
    def __init__(self, model=baseline_model, feature_layers=feature_layers, use_multi_scale=True, target_spatial_size=target_spatial_size):
        super().__init__()
        self.feature_layers = feature_layers
        self.model = model
        self.use_multi_scale = use_multi_scale
        self.target_spatial_size = target_spatial_size
        self._feature_extractor = nn.ModuleDict(OrderedDict([
            ('conv1', nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)),
            ('layer1', model.layer1),
            ('layer2', model.layer2),
            ('layer3', model.layer3),
            ('layer4', model.layer4)
        ]))
        if use_multi_scale:
            self.fusion = FeatureFusion(target_spatial_size)

    def forward(self, x):
        feats = []
        out = x
        for name, layer in self._feature_extractor.items():
            out = layer(out)
            if name in self.feature_layers:
                feats.append(out)
        if self.use_multi_scale:
            return self.fusion(feats)  # [B, C_sum, H, W]
        else:
            return feats[0]

# 特征门控机制模块
class GatingMechanism(nn.Module):
    def __init__(self, gate_type='se', reduction_ratio=16, backbone=backbone, feature_layers=feature_layers):
        super().__init__()
        self.gate_type = gate_type
        # channel maps for common ResNets
        if backbone == 'resnet18':
            layer_channels = {'layer1':64, 'layer2':128, 'layer3':256, 'layer4':512}
        elif backbone == 'resnet50':
            layer_channels = {'layer1':256, 'layer2':512, 'layer3':1024, 'layer4':2048}
        else:
            raise ValueError("Unsupported backbone")

        # input_dim is channel count (sum of selected layers)
        self.input_dim = sum(layer_channels[l] for l in feature_layers)
        if gate_type == 'sigmoid':
            self.gate_linear = nn.Linear(self.input_dim, self.input_dim)
            self.sigmoid = nn.Sigmoid()
        else:
            self.reduce_dim = max(self.input_dim // reduction_ratio, 1)
            self.excitation = nn.Sequential(
                nn.Linear(self.input_dim, self.reduce_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.reduce_dim, self.input_dim),
                nn.Sigmoid()
            )

    def forward(self, x):
        # x: [B, C, H, W]
        if self.gate_type == 'sigmoid':
            # apply global average pooling first to get [B, C]
            s = torch.mean(x, dim=[2,3])
            g = self.sigmoid(self.gate_linear(s))
            return x * g.unsqueeze(2).unsqueeze(3)
        else:
            s = torch.mean(x, dim=[2,3])  # [B, C]
            scale = self.excitation(s)    # [B, C]
            return x * scale.unsqueeze(2).unsqueeze(3)
# 位置编码
class PositionEncoder(nn.Module):
    def __init__(self, model_dim=model_dim, dropout=dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = model_dim

        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-math.log(10000.0) / model_dim)).unsqueeze(0)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, model_dim]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [B, T, D]
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return self.dropout(x)

# 图像分块/浅层特征投影模块
class ImagePatchExtractor(nn.Module):
    def __init__(self, patch_size=16, model_dim=model_dim, image_size=224):
        super().__init__()
        self.patch_size = patch_size
        self.model_dim = model_dim
        self.image_size = image_size
        self.num_patches = (image_size // patch_size) ** 2
        self.projection = nn.Linear(patch_size * patch_size * 3, model_dim)

    def _extract_patches(self, x):
        # x: [B, 3, H, W]
        unfolded = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size)  # [B, patch_flat, num_patches]
        unfolded = unfolded.permute(0,2,1)  # [B, num_patches, patch_flat]
        return unfolded

    def forward(self, x):
        patches = self._extract_patches(x)   # [B, num_patches, patch_flat]
        projected = self.projection(patches) # [B, num_patches, model_dim]
        return projected
# Transformer编码器模块
class TransformerEncoder(nn.Module):
    def __init__(self, model_dim=model_dim, dropout=dropout, nhead=nhead, dim_feedforward=2048, num_layers=num_layers):
        super().__init__()
        self.model_dim = model_dim
        self.pos_encoder = PositionEncoder(model_dim=model_dim, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x: [B, T, D]
        x = self.pos_encoder(x)
        return self.transformer(x)  # [B, T, D]
# 交叉注意力模块
#交叉注意力层
class CrossAttentionLayer(nn.Module):
    def __init__(self, nhead=8, model_dim=256, dropout=0.1, cnn_in_channels=None):
        super().__init__()
        self.nhead = nhead
        self.model_dim = model_dim
        self.head_dim = model_dim // nhead
        assert self.head_dim * nhead == model_dim, "model_dim must be divisible by nhead"
        self.cnn_in_channels = cnn_in_channels
        if cnn_in_channels is not None:
            self.cnn_proj = nn.Linear(cnn_in_channels, model_dim)  # project CNN channels -> model_dim
        self.w_q = nn.Linear(model_dim, model_dim)
        self.w_k = nn.Linear(model_dim, model_dim)
        self.w_v = nn.Linear(model_dim, model_dim)
        self.w_o = nn.Linear(model_dim, model_dim)
        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(model_dim)

    def _prepare(self, x):
        # Accept sequence [B, T, D] or CNN map [B, C, H, W]
        if x.dim() == 3:
            # [B, T, D]
            if x.size(-1) != self.model_dim:
               
                proj = nn.Linear(x.size(-1), self.model_dim).to(x.device)
                x = proj(x)
            return x
        elif x.dim() == 4:
            B, C, H, W = x.shape
            x_seq = x.view(B, C, H*W).permute(0,2,1)  # [B, H*W, C]
            if self.cnn_in_channels is None:
                raise RuntimeError("CrossAttentionLayer expects cnn_in_channels for 4D inputs")
            x_seq = self.cnn_proj(x_seq)  # [B, H*W, model_dim]
            return x_seq
        else:
            raise ValueError("Unsupported input dim for CrossAttentionLayer")

    def forward(self, q, k, v):
        q_p = self._prepare(q)
        k_p = self._prepare(k)
        v_p = self._prepare(v)
        B = q_p.size(0)

        q_lin = self.w_q(q_p)
        k_lin = self.w_k(k_p)
        v_lin = self.w_v(v_p)

        qh = q_lin.view(B, -1, self.nhead, self.head_dim).transpose(1,2)  # [B, nhead, Tq, head_dim]
        kh = k_lin.view(B, -1, self.nhead, self.head_dim).transpose(1,2)
        vh = v_lin.view(B, -1, self.nhead, self.head_dim).transpose(1,2)

        scores = torch.matmul(qh, kh.transpose(-2,-1)) * self.scale
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, vh)  # [B, nhead, Tq, head_dim]
        out = out.transpose(1,2).contiguous().view(B, -1, self.nhead * self.head_dim)  # [B, Tq, model_dim]
        out = self.w_o(out)
        out = self.norm(q_lin + out)
        return out, attn
#多层交叉注意力
class CrossAttention(nn.Module):
    def __init__(self, num_layers=2, nhead=8, model_dim=256, dropout=0.1, cnn_in_channels=None):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossAttentionLayer(nhead=nhead, model_dim=model_dim, dropout=dropout, cnn_in_channels=cnn_in_channels)
            for _ in range(num_layers)
        ])

    def forward(self, query, key, value):
        out = query
        attn = None
        for layer in self.layers:
            out, attn = layer(out, key, value)
        return out, attn

# 分类头
class ClassificationHead(nn.Module):
    def __init__(self, input_dim=2*model_dim, hidden_dim=256, dropout=dropout, num_classes=num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # x: [B, input_dim]
        return self.classifier(x)


# CNN-Transformer 混合模型
class HybridModel(nn.Module):
    def __init__(self,
                 backbone_model=baseline_model,
                 feature_layers=feature_layers,
                 gate_type='se',
                 model_dim=model_dim,
                 nhead=nhead,
                 num_layers=num_layers,
                 dropout=dropout,
                 num_classes=num_classes):
        super().__init__()
        self.backbone = backbone_model
        self.mutil_scale_extractor = MutilScaleFeatureExtractor(model=self.backbone, feature_layers=feature_layers, use_multi_scale=True)
        
        if backbone == 'resnet18':
            layer_channels = {'layer1':64, 'layer2':128, 'layer3':256, 'layer4':512}
        else:
            layer_channels = {'layer1':256, 'layer2':512, 'layer3':1024, 'layer4':2048}
        self.fused_channels = sum(layer_channels[l] for l in feature_layers)  # e.g. 768
        self.gate_mechanism = GatingMechanism(gate_type=gate_type, backbone=backbone, feature_layers=feature_layers)

        # transformer patch stream
        self.image_patch_extractor = ImagePatchExtractor(patch_size=16, model_dim=model_dim, image_size=224)
        self.transformer_encoder = TransformerEncoder(model_dim=model_dim, dropout=dropout, nhead=nhead, num_layers=num_layers)

        
        self.cnn_to_seq = nn.Linear(self.fused_channels, model_dim)

        
        self.cross_attention = CrossAttention(num_layers=2, nhead=nhead, model_dim=model_dim, dropout=dropout, cnn_in_channels=self.fused_channels)

        
        self.classifier = ClassificationHead(input_dim=2*model_dim, num_classes=num_classes)

    def forward(self, x):
        # x: [B, 3, H, W]
        B = x.size(0)
        # CNN multi-scale -> [B, fused_channels, Hf, Wf]
        scnn = self.mutil_scale_extractor(x)
        scnn = self.gate_mechanism(scnn)  # gated [B, C, H, W]

        # Convert CNN map -> sequence [B, T_cnn, C] then project to model_dim
        B, C, Hf, Wf = scnn.shape
        scnn_seq = scnn.view(B, C, Hf*Wf).permute(0,2,1)  # [B, T_cnn, C]
        scnn_seq_proj = self.cnn_to_seq(scnn_seq)         # [B, T_cnn, model_dim]

        # Transformer path
        stf = self.image_patch_extractor(x)               # [B, T_patch, model_dim]
        stf = self.transformer_encoder(stf)               # [B, T_patch, model_dim]

        # Cross attention: treat queries as sequences (we pass sequences)
        attncnn, attn_weights1 = self.cross_attention(scnn_seq_proj, stf, stf)  # [B, T_cnn, model_dim]
        attntf, attn_weights2 = self.cross_attention(stf, scnn_seq_proj, scnn_seq_proj)  # [B, T_patch, model_dim]

        # Pool each branch (global mean over sequence)
        pooled_cnn = attncnn.mean(dim=1)   # [B, model_dim]
        pooled_tf  = attntf.mean(dim=1)    # [B, model_dim]

        fused = torch.cat([pooled_cnn, pooled_tf], dim=1)  # [B, 2*model_dim]
        out = self.classifier(fused)                       # [B, num_classes]
        return out

    def to(self, device):
        super().to(device)
        return self

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hybrid_model = HybridModel().to(device)
