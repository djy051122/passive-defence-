"""Complete-band privileged-information teacher."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .encoders import build_encoder
from .fusion import HighFrequencyFusion, ProjectionBlock
from .outputs import DetectionOutput


class FullBandTeacher(nn.Module):
    """Binary real/fake teacher using RGB, LL2, H1, and H2."""

    def __init__(
        self,
        num_classes: int = 2,
        backbone: str = "resnet18",
        pretrained: bool = True,
        embedding_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if num_classes != 2:
            raise ValueError("The teacher must have exactly two known classes")
        self.rgb_encoder = build_encoder(backbone, 3, pretrained)
        self.low_encoder = build_encoder(backbone, 3, pretrained)
        self.h1_encoder = build_encoder(backbone, 9, pretrained)
        self.h2_encoder = build_encoder(backbone, 9, pretrained)
        encoder_dim = self.rgb_encoder.output_dim
        self.rgb_projection = ProjectionBlock(encoder_dim, embedding_dim, dropout)
        self.low_projection = ProjectionBlock(encoder_dim, embedding_dim, dropout)
        self.high_fusion = HighFrequencyFusion(encoder_dim, embedding_dim, dropout)
        self.full_band_fusion = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim),
            nn.LayerNorm(embedding_dim), nn.GELU(), nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(embedding_dim, 2)
        self.rgb_classifier = nn.Linear(embedding_dim, 2)
        self.low_classifier = nn.Linear(embedding_dim, 2)
        self.high_classifier = nn.Linear(embedding_dim, 2)

    def forward(self, rgb: Tensor, ll2: Tensor, h1: Tensor, h2: Tensor) -> DetectionOutput:
        rgb_features = self.rgb_projection(self.rgb_encoder(rgb))
        low_features = self.low_projection(self.low_encoder(ll2))
        high_features = self.high_fusion(self.h1_encoder(h1), self.h2_encoder(h2))
        embedding = self.full_band_fusion(
            torch.cat((rgb_features, low_features, high_features), dim=1)
        )
        return DetectionOutput(
            logits=self.classifier(embedding),
            embedding=embedding,
            rgb_logits=self.rgb_classifier(rgb_features),
            high_logits=self.high_classifier(high_features),
            low_logits=self.low_classifier(low_features),
        )

