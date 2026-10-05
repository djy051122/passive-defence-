"""Complete stage-one and stage-two objectives from the research design."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from models.outputs import DetectionOutput
from .classification import KnownBinaryCrossEntropy
from .distillation import distillation_kl_divergence


def jensen_shannon(probabilities_a: Tensor, probabilities_b: Tensor) -> Tensor:
    midpoint = (probabilities_a + probabilities_b) / 2
    return 0.5 * (
        F.kl_div(midpoint.log(), probabilities_a, reduction="batchmean")
        + F.kl_div(midpoint.log(), probabilities_b, reduction="batchmean")
    )


def energy_score(logits: Tensor, temperature: float = 1.0) -> Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return -temperature * torch.logsumexp(logits / temperature, dim=1)


def binary_separation_loss(
    embeddings: Tensor,
    targets: Tensor,
    margin: float = 1.0,
    margin_weight: float = 1.0,
) -> Tensor:
    """Make real/fake binary projections compact and mutually separated.

    Centers are estimated from the current batch.  A batch containing only one
    class still receives the compactness term; the inter-class term is then
    skipped rather than inventing an unknown or subclass prototype.
    """

    if embeddings.ndim != 2 or targets.ndim != 1:
        raise ValueError("embeddings and targets must have shapes [N,D] and [N]")
    if embeddings.shape[0] != targets.shape[0]:
        raise ValueError("embeddings and targets must contain the same samples")
    if margin < 0 or margin_weight < 0:
        raise ValueError("margin and margin_weight must be non-negative")
    normalized = F.normalize(embeddings, dim=1)
    centers: dict[int, Tensor] = {}
    compact = normalized.new_zeros(())
    for class_index in (0, 1):
        mask = targets == class_index
        if mask.any():
            center = F.normalize(normalized[mask].mean(dim=0), dim=0)
            centers[class_index] = center
            compact = compact + (normalized[mask] - center).square().sum(dim=1).sum()
    compact = compact / max(1, normalized.shape[0])
    separation = normalized.new_zeros(())
    if len(centers) == 2:
        center_distance = torch.linalg.vector_norm(centers[0] - centers[1])
        separation = F.relu(normalized.new_tensor(margin) - center_distance).square()
    return compact + margin_weight * separation


class TeacherObjective(nn.Module):
    def __init__(
        self,
        rgb_aux_weight: float,
        high_aux_weight: float,
        low_aux_weight: float,
        robustness_weight: float,
        feature_consistency_weight: float,
        degraded_classification_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.ce = KnownBinaryCrossEntropy()
        self.rgb_aux_weight = rgb_aux_weight
        self.high_aux_weight = high_aux_weight
        self.low_aux_weight = low_aux_weight
        self.robustness_weight = robustness_weight
        self.feature_consistency_weight = feature_consistency_weight
        self.degraded_classification_weight = degraded_classification_weight

    def forward(
        self,
        output: DetectionOutput,
        targets: Tensor,
        degraded_output: DetectionOutput | None = None,
        robustness_reference: DetectionOutput | None = None,
    ) -> dict[str, Tensor]:
        main = self.ce(output.logits, targets)
        rgb_aux = self.ce(output.rgb_logits, targets)
        high_aux = self.ce(output.high_logits, targets)
        if output.low_logits is None:
            raise ValueError("Teacher output is missing low-frequency logits")
        low_aux = self.ce(output.low_logits, targets)
        robustness = main.new_zeros(())
        feature_consistency = main.new_zeros(())
        degraded_classification = main.new_zeros(())
        if degraded_output is not None:
            reference = robustness_reference or output
            robustness = jensen_shannon(
                F.softmax(reference.logits.detach(), dim=1),
                F.softmax(degraded_output.logits, dim=1),
            )
            feature_consistency = F.l1_loss(
                degraded_output.embedding, reference.embedding.detach()
            )
            degraded_classification = self.ce(degraded_output.logits, targets)
        total = (
            main + self.rgb_aux_weight * rgb_aux + self.high_aux_weight * high_aux
            + self.low_aux_weight * low_aux + self.robustness_weight * robustness
            + self.feature_consistency_weight * feature_consistency
            + self.degraded_classification_weight * degraded_classification
        )
        return {
            "total": total, "main": main, "rgb_aux": rgb_aux,
            "high_aux": high_aux, "low_aux": low_aux,
            "robustness": robustness, "feature_consistency": feature_consistency,
            "degraded_classification": degraded_classification,
        }


class StudentOpenSetObjective(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.ce = KnownBinaryCrossEntropy()
        self.config = config

    def forward(
        self,
        student: DetectionOutput,
        teacher: DetectionOutput,
        targets: Tensor,
        fake_subclasses: Tensor,
        degraded_student: DetectionOutput | None = None,
        robustness_reference: DetectionOutput | None = None,
    ) -> dict[str, Tensor]:
        zero = student.logits.new_zeros(())
        main = self.ce(student.logits, targets)
        aux = 0.5 * (
            self.ce(student.rgb_logits, targets) + self.ce(student.high_logits, targets)
        )
        kd = distillation_kl_divergence(
            student.logits, teacher.logits, float(self.config["kd_temperature"])
        )
        student_distillation = (
            student.distillation_embedding
            if student.distillation_embedding is not None else student.embedding
        )
        feature_distillation = F.l1_loss(
            student_distillation, teacher.embedding.detach()
        )
        separation = binary_separation_loss(
            student.embedding,
            targets,
            margin=float(self.config.get("separation_margin", 1.0)),
            margin_weight=float(self.config.get("separation_margin_weight", 1.0)),
        )
        subclass = zero
        valid_subclass = (targets == 1) & (fake_subclasses >= 0)
        if student.subclass_logits is not None and valid_subclass.any():
            subclass = F.cross_entropy(
                student.subclass_logits[valid_subclass], fake_subclasses[valid_subclass]
            )
        robustness = zero
        feature_consistency = zero
        degraded_classification = zero
        if degraded_student is not None:
            reference = robustness_reference or student
            robustness = jensen_shannon(
                F.softmax(reference.logits.detach(), dim=1),
                F.softmax(degraded_student.logits, dim=1),
            )
            degraded_features = (
                degraded_student.fusion_embedding
                if degraded_student.fusion_embedding is not None
                else degraded_student.embedding
            )
            reference_features = (
                reference.fusion_embedding
                if reference.fusion_embedding is not None
                else reference.embedding
            )
            feature_consistency = F.l1_loss(
                degraded_features, reference_features.detach()
            )
            degraded_classification = self.ce(degraded_student.logits, targets)
        total = (
            main + float(self.config["auxiliary_weight"]) * aux
            + float(self.config["kd_weight"]) * kd
            + float(self.config["feature_distillation_weight"]) * feature_distillation
            + float(self.config["subclass_weight"]) * subclass
            + float(self.config.get("separation_weight", 0.0)) * separation
            + float(self.config["robustness_weight"]) * robustness
            + float(self.config["feature_consistency_weight"]) * feature_consistency
            + float(self.config.get("degraded_classification_weight", 0.0))
            * degraded_classification
        )
        return {
            "total": total, "main": main, "auxiliary": aux, "kd": kd,
            "feature_distillation": feature_distillation, "subclass": subclass,
            "separation": separation,
            "robustness": robustness, "feature_consistency": feature_consistency,
            "degraded_classification": degraded_classification,
        }
