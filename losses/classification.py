"""Known-distribution real/fake classification loss."""

from __future__ import annotations

from torch import Tensor, nn


class KnownBinaryCrossEntropy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        if logits.ndim != 2 or logits.shape[1] != 2:
            raise ValueError(f"Expected [batch, 2] logits, got {tuple(logits.shape)}")
        if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
            raise ValueError("targets must have shape [batch]")
        return self.loss(logits, targets)

