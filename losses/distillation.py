"""Temperature-scaled logit distillation."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor


def distillation_kl_divergence(
    student_logits: Tensor,
    teacher_logits: Tensor,
    temperature: float,
) -> Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("Teacher and student logits must have identical shapes")
    if student_logits.ndim != 2 or student_logits.shape[1] != 2:
        raise ValueError("Distillation requires [batch, 2] logits")
    return F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits.detach() / temperature, dim=1),
        reduction="batchmean",
    ) * temperature**2

