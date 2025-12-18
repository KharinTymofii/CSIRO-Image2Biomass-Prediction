import torch
import torch.nn as nn


class MultiScalePooling(nn.Module):
    """Extract features at multiple scales using different pooling strategies."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d(1),
            nn.AdaptiveMaxPool2d(1),
            nn.AdaptiveAvgPool2d(2),
            nn.AdaptiveMaxPool2d(2),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            [B, C * 6] - concatenated multi-scale features
        """
        features = []
        for pool in self.pools:
            pooled = pool(x).flatten(1)
            features.append(pooled)
        return torch.cat(features, dim=1)