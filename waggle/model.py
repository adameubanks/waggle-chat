from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models.video import R3D_18_Weights, r3d_18


class WaggleNet(nn.Module):
    def __init__(self, *, pretrained: bool = True, in_channels: int = 2) -> None:
        super().__init__()
        weights = R3D_18_Weights.DEFAULT if pretrained else None
        self.backbone = r3d_18(weights=weights)
        if in_channels != 3:
            stem = self.backbone.stem[0]
            self.backbone.stem[0] = nn.Conv3d(
                in_channels,
                stem.out_channels,
                kernel_size=stem.kernel_size,
                stride=stem.stride,
                padding=stem.padding,
                bias=False,
            )
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.head = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        return self.head(feat).squeeze(1)
