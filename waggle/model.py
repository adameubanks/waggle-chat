from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models.video import R3D_18_Weights, r3d_18


class WaggleNet(nn.Module):
    def __init__(self, *, pretrained: bool = True) -> None:
        super().__init__()
        weights = R3D_18_Weights.DEFAULT if pretrained else None
        self.backbone = r3d_18(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.head = nn.Linear(in_features, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        feat = self.backbone(x)
        return self.head(feat)
