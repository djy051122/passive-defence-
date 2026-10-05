"""Feature projection and fusion blocks."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ProjectionBlock(nn.Sequential):
    def __init__(self, input_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )


class HighFrequencyFusion(nn.Module):
    """Fuse independently encoded H1 and H2 feature vectors."""

    def __init__(self, input_dim: int, embedding_dim: int, dropout: float) -> None:
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(input_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, h1: Tensor, h2: Tensor) -> Tensor:
        if h1.shape != h2.shape:
            raise ValueError(
                f"H1 and H2 encoded features must match, got {h1.shape} and {h2.shape}"
            )
        return self.fusion(torch.cat((h1, h2), dim=1))


class GatedRGBHighFrequencyFusion(nn.Module):
    """Fuse RGB and high-frequency evidence with a learned feature gate."""

    def __init__(self, input_dim: int, embedding_dim: int, dropout: float) -> None:
        super().__init__()
        self.rgb_projection = ProjectionBlock(input_dim, embedding_dim, dropout)
        self.high_projection = ProjectionBlock(input_dim, embedding_dim, dropout)
        self.gate = nn.Sequential(
            nn.Linear(embedding_dim * 4, embedding_dim),
            nn.Sigmoid(),
        )
        self.output = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, rgb: Tensor, high: Tensor) -> Tensor:
        rgb_projected = self.rgb_projection(rgb)
        high_projected = self.high_projection(high)
        common = rgb_projected * high_projected
        difference = (rgb_projected - high_projected).abs()
        gate = self.gate(
            torch.cat((rgb_projected, high_projected, common, difference), dim=1)
        )
        gated = gate * rgb_projected + (1.0 - gate) * high_projected
        return self.output(torch.cat((gated, common, difference), dim=1))
