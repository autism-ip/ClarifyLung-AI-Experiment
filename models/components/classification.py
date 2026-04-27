"""
 * [INPUT]: 依赖 torch.nn
 * [OUTPUT]: 对外提供 ClassificationHead
 * [POS]: models/components 的分类头模块，被 hybrid_model.HybridModel 消费
 * [PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import torch.nn as nn


class ClassificationHead(nn.Module):
    """分类头：双层全连接 + ReLU + Dropout"""

    def __init__(self, input_dim=1024, hidden_dim=256, dropout=0.1, num_classes=3):
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
