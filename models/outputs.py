"""Structured outputs shared by teacher and student."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass
class DetectionOutput:
    logits: Tensor
    embedding: Tensor
    rgb_logits: Tensor
    high_logits: Tensor
    low_logits: Tensor | None = None
    subclass_logits: Tensor | None = None
    distillation_embedding: Tensor | None = None
    subclass_embedding: Tensor | None = None
    fusion_embedding: Tensor | None = None
