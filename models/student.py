"""Deployable RGB and high-frequency student."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .encoders import build_encoder
from .fusion import GatedRGBHighFrequencyFusion, HighFrequencyFusion, ProjectionBlock
from .outputs import DetectionOutput


class RGBHighFrequencyStudent(nn.Module):
    """Binary student whose interface never accepts LL2."""

    def __init__(
        self,
        num_classes: int = 2,
        num_fake_subclasses: int = 2,
        backbone: str = "resnet18",
        pretrained: bool = True,
        embedding_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if num_classes != 2:
            raise ValueError("The student must have exactly two known classes")
        self.rgb_encoder = build_encoder(backbone, 3, pretrained)
        self.h1_encoder = build_encoder(backbone, 9, pretrained)
        self.h2_encoder = build_encoder(backbone, 9, pretrained)
        encoder_dim = self.rgb_encoder.output_dim
        self.high_fusion = HighFrequencyFusion(encoder_dim, encoder_dim, dropout)
        self.gated_fusion = GatedRGBHighFrequencyFusion(encoder_dim, embedding_dim, dropout)
        # Keep the binary geometry used by prototypes/spheres separate from the
        # representation used to explain fake samples.  This prevents the
        # three fake families from splitting the single trusted fake region.
        self.binary_projection = ProjectionBlock(embedding_dim, embedding_dim, dropout)
        self.subclass_projection = ProjectionBlock(embedding_dim, embedding_dim, dropout)
        self.classifier = nn.Linear(embedding_dim, 2)
        self.rgb_classifier = nn.Linear(encoder_dim, 2)
        self.high_classifier = nn.Linear(encoder_dim, 2)
        self.subclass_classifier = (
            nn.Linear(embedding_dim, num_fake_subclasses)
            if num_fake_subclasses > 0 else None
        )
        # Used only during stage-two training.  The deployable classifier still
        # consumes ``embedding``; this adapter lets the student match the
        # full-band teacher without forcing both latent spaces to be identical.
        self.distillation_projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def forward(
        self,
        rgb: Tensor,
        h1: Tensor,
        h2: Tensor,
    ) -> DetectionOutput:
        rgb_features = self.rgb_encoder(rgb)
        high_features = self.high_fusion(self.h1_encoder(h1), self.h2_encoder(h2))
        fused_embedding = self.gated_fusion(rgb_features, high_features)
        binary_embedding = self.binary_projection(fused_embedding)
        subclass_embedding = self.subclass_projection(fused_embedding)
        return DetectionOutput(
            logits=self.classifier(binary_embedding),
            embedding=binary_embedding,
            rgb_logits=self.rgb_classifier(rgb_features),
            high_logits=self.high_classifier(high_features),
            subclass_logits=(
                self.subclass_classifier(subclass_embedding)
                if self.subclass_classifier is not None else None
            ),
            distillation_embedding=self.distillation_projection(fused_embedding),
            subclass_embedding=subclass_embedding,
            fusion_embedding=fused_embedding,
        )
