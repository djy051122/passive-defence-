"""Prototype evidence, trusted real/fake ellipsoids, and open-set metrics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score
from torch import Tensor

from losses import energy_score
from models import DetectionOutput


REGION_NAMES = ("real", "fake")


@dataclass(frozen=True)
class Evidence:
    """Evidence extracted from one student prediction.

    The first four fields retain the original API.  The probability and
    class-specific prototype similarities are required by the three-dimensional
    real/fake/novelty evidence space.
    """

    energy: float
    prototype_distance: float
    branch_disagreement: float
    confidence: float
    real_probability: float = 0.5
    fake_probability: float = 0.5
    real_similarity: float = 0.0
    fake_similarity: float = 0.0


@dataclass(frozen=True)
class RegionDecision:
    prediction: int
    region: str
    point: tuple[float, float, float]
    distances: dict[str, float]
    memberships: dict[str, bool]
    novelty_score: float


class PrototypeBank:
    def __init__(self, prototypes: Tensor, names: list[str]) -> None:
        if prototypes.ndim != 2 or prototypes.shape[0] != len(names):
            raise ValueError("Prototype tensor and names do not match")
        if "real" not in names or "fake" not in names:
            raise ValueError("Prototype bank must contain real and fake prototypes")
        self.prototypes = F.normalize(prototypes.float(), dim=1)
        self.names = names

    @classmethod
    def fit(cls, embeddings: Tensor, group_ids: Tensor, group_names: list[str]) -> "PrototypeBank":
        normalized = F.normalize(embeddings.float(), dim=1)
        prototypes, names = [], []
        for group_id, name in enumerate(group_names):
            mask = group_ids == group_id
            if mask.any():
                prototypes.append(F.normalize(normalized[mask].mean(dim=0), dim=0))
                names.append(name)
        if not prototypes:
            raise ValueError("No samples were available to calculate prototypes")
        return cls(torch.stack(prototypes), names)

    def similarities(self, embeddings: Tensor) -> Tensor:
        embeddings = F.normalize(embeddings.float(), dim=1)
        return embeddings @ self.prototypes.to(embeddings.device).T

    def distance(self, embeddings: Tensor) -> Tensor:
        return 1 - self.similarities(embeddings).max(dim=1).values

    def real_fake_similarity(self, embeddings: Tensor) -> tuple[Tensor, Tensor]:
        similarities = self.similarities(embeddings)
        real_index = self.names.index("real")
        fake_indices = [index for index, name in enumerate(self.names) if name != "real"]
        return similarities[:, real_index], similarities[:, fake_indices].max(dim=1).values

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"prototypes": self.prototypes.cpu(), "names": self.names}, target)

    @classmethod
    def load(cls, path: str | Path) -> "PrototypeBank":
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        return cls(payload["prototypes"], payload["names"])


def extract_evidence(
    output: DetectionOutput,
    prototype_bank: PrototypeBank,
    energy_temperature: float,
) -> list[Evidence]:
    probabilities = F.softmax(output.logits, dim=1)
    rgb_probabilities = F.softmax(output.rgb_logits, dim=1)
    high_probabilities = F.softmax(output.high_logits, dim=1)
    midpoint = (rgb_probabilities + high_probabilities) / 2
    disagreement = 0.5 * (
        (rgb_probabilities * (rgb_probabilities.clamp_min(1e-8).log() - midpoint.clamp_min(1e-8).log())).sum(dim=1)
        + (high_probabilities * (high_probabilities.clamp_min(1e-8).log() - midpoint.clamp_min(1e-8).log())).sum(dim=1)
    )
    energies = energy_score(output.logits, energy_temperature)
    distances = prototype_bank.distance(output.embedding)
    real_similarity, fake_similarity = prototype_bank.real_fake_similarity(output.embedding)
    confidence = probabilities.max(dim=1).values
    return [
        Evidence(
            energy=float(energies[i]),
            prototype_distance=float(distances[i]),
            branch_disagreement=float(disagreement[i]),
            confidence=float(confidence[i]),
            real_probability=float(probabilities[i, 0]),
            fake_probability=float(probabilities[i, 1]),
            real_similarity=float(real_similarity[i]),
            fake_similarity=float(fake_similarity[i]),
        )
        for i in range(output.logits.shape[0])
    ]


class TrustedSphereCalibrator:
    """Fit trusted real/fake Mahalanobis ellipsoids; exterior is unknown.

    The score geometry is fitted on known training samples, then class-specific
    radii are selected only from held-out real/fake calibration samples. The
    evidence space remains three-dimensional (real, fake, novelty evidence),
    with no learned unknown class or unknown ellipsoid.
    """

    VERSION = 8

    def __init__(self, artifacts: dict[str, Any]) -> None:
        if int(artifacts.get("version", 0)) != self.VERSION:
            raise ValueError(
                "Legacy calibration is incompatible with the trusted real/fake "
                "split-conformal ellipsoid rule. Run circle.py again."
            )
        self.artifacts = artifacts

    @classmethod
    def fit(
        cls,
        real_fit_evidence: list[Evidence],
        fake_fit_evidence: list[Evidence],
        real_calibration_evidence: list[Evidence],
        fake_calibration_evidence: list[Evidence],
        radius_quantile: float = 0.90,
        covariance_regularization: float = 1e-4,
        include_energy: bool = False,
    ) -> "TrustedSphereCalibrator":
        fit_groups = (real_fit_evidence, fake_fit_evidence)
        calibration_groups = (
            real_calibration_evidence, fake_calibration_evidence,
        )
        if any(not group for group in fit_groups):
            raise ValueError("Real and fake fitting evidence are both required")
        if any(not group for group in calibration_groups):
            raise ValueError(
                "Separate real and fake conformal calibration evidence are required"
            )
        if not 0 < radius_quantile < 1:
            raise ValueError("radius_quantile must be in (0, 1)")
        if covariance_regularization <= 0:
            raise ValueError("covariance_regularization must be positive")
        if any(len(group) < 2 for group in fit_groups):
            raise ValueError("Each fitting class needs at least two samples")

        fit_known = real_fit_evidence + fake_fit_evidence
        component_names = ["prototype_distance", "branch_disagreement", "uncertainty"]
        if include_energy:
            component_names.insert(0, "energy")
        component_matrix = _novelty_component_matrix(fit_known, component_names)
        component_mean = component_matrix.mean(axis=0)
        component_std = component_matrix.std(axis=0)
        component_std[component_std < 1e-8] = 1.0
        standardized_components = (component_matrix - component_mean) / component_std
        # No external unknown examples are used. Novelty is an equal-weight
        # combination of warning signals standardized only with real/fake
        # fitting data. Energy is opt-in because no external-unknown
        # objective explicitly trains it as a novelty signal.
        novelty_weights = np.full(
            len(component_names), 1.0 / len(component_names), dtype=float
        )
        novelty_scores = standardized_components @ novelty_weights

        raw_coordinates = _evidence_coordinates(fit_known, novelty_scores)
        coordinate_mean = raw_coordinates.mean(axis=0)
        coordinate_std = raw_coordinates.std(axis=0)
        coordinate_std[coordinate_std < 1e-8] = 1.0
        fit_points = (raw_coordinates - coordinate_mean) / coordinate_std
        fit_labels = np.r_[
            np.zeros(len(real_fit_evidence), dtype=int),
            np.ones(len(fake_fit_evidence), dtype=int),
        ]
        ellipsoids = [
            _fit_mahalanobis_ellipsoid(
                fit_points[fit_labels == index], covariance_regularization
            )
            for index in range(2)
        ]
        centers = np.stack([item["center"] for item in ellipsoids])
        precisions = np.stack([item["precision"] for item in ellipsoids])

        # The score function is now fixed. Only the held-out calibration
        # samples below are allowed to determine the conformal radii.
        calibration_known = (
            real_calibration_evidence + fake_calibration_evidence
        )
        calibration_components = _novelty_component_matrix(
            calibration_known, component_names
        )
        calibration_standardized = (
            calibration_components - component_mean
        ) / component_std
        calibration_novelty = calibration_standardized @ novelty_weights
        calibration_raw_coordinates = _evidence_coordinates(
            calibration_known, calibration_novelty
        )
        calibration_points = (
            calibration_raw_coordinates - coordinate_mean
        ) / coordinate_std
        calibration_labels = np.r_[
            np.zeros(len(real_calibration_evidence), dtype=int),
            np.ones(len(fake_calibration_evidence), dtype=int),
        ]
        distance_matrix = np.column_stack([
            _mahalanobis_distances(
                calibration_points, centers[index], precisions[index]
            )
            for index in range(2)
        ])
        radii = np.asarray([
            _conformal_radius(
                distance_matrix[calibration_labels == index, index],
                radius_quantile,
            )
            for index in range(2)
        ])
        diagnostics = _sphere_diagnostics(
            distance_matrix, calibration_labels, radii
        )
        radius_selection = {
            "method": "class_conditional_split_conformal_order_statistic",
            "target_coverage": float(radius_quantile),
            "alpha": float(1.0 - radius_quantile),
            "finite_sample_correction": "ceil((n + 1) * target_coverage)",
            "selected": {
                name: {
                    "radius": float(radii[index]),
                    "own_samples": int((calibration_labels == index).sum()),
                    "conformal_rank": _conformal_rank(
                        int((calibration_labels == index).sum()),
                        radius_quantile,
                    ),
                    "effective_quantile": min(
                        1.0,
                        _conformal_rank(
                            int((calibration_labels == index).sum()),
                            radius_quantile,
                        ) / int((calibration_labels == index).sum()),
                    ),
                }
                for index, name in enumerate(REGION_NAMES)
            },
        }
        artifacts = {
            "version": cls.VERSION,
            "rule": "trusted real/fake Mahalanobis ellipsoids; outside or overlap -> unknown",
            "distance_metric": "class_conditional_mahalanobis",
            "novelty_component_names": component_names,
            "novelty_component_mean": component_mean.tolist(),
            "novelty_component_std": component_std.tolist(),
            "novelty_weights": novelty_weights.tolist(),
            "coordinate_names": ["real_evidence", "fake_evidence", "novelty_evidence"],
            "coordinate_mean": coordinate_mean.tolist(),
            "coordinate_std": coordinate_std.tolist(),
            "centers": {
                name: centers[index].tolist() for index, name in enumerate(REGION_NAMES)
            },
            "covariances": {
                name: ellipsoids[index]["covariance"].tolist()
                for index, name in enumerate(REGION_NAMES)
            },
            "precisions": {
                name: precisions[index].tolist()
                for index, name in enumerate(REGION_NAMES)
            },
            "radii": {
                name: float(radii[index]) for index, name in enumerate(REGION_NAMES)
            },
            "radius_quantile": radius_quantile,
            "covariance_regularization": covariance_regularization,
            "radius_selection": radius_selection,
            "calibration_diagnostics": diagnostics,
            "fit_counts": {
                "real": len(real_fit_evidence),
                "fake": len(fake_fit_evidence),
            },
            "calibration_counts": {
                "real": len(real_calibration_evidence),
                "fake": len(fake_calibration_evidence),
            },
        }
        return cls(artifacts)

    def _novelty_score(self, evidence: Evidence) -> float:
        raw = _novelty_component_matrix(
            [evidence], self.artifacts["novelty_component_names"]
        )[0]
        standardized = (
            raw - np.asarray(self.artifacts["novelty_component_mean"], dtype=float)
        ) / np.asarray(self.artifacts["novelty_component_std"], dtype=float)
        return float(standardized @ np.asarray(self.artifacts["novelty_weights"], dtype=float))

    def point(self, evidence: Evidence) -> np.ndarray:
        raw = np.asarray([
            _real_score(evidence), _fake_score(evidence), self._novelty_score(evidence)
        ])
        return (
            raw - np.asarray(self.artifacts["coordinate_mean"], dtype=float)
        ) / np.asarray(self.artifacts["coordinate_std"], dtype=float)

    def decide(self, evidence: Evidence) -> RegionDecision:
        point = self.point(evidence)
        centers = self.artifacts["centers"]
        precisions = self.artifacts["precisions"]
        radii = self.artifacts["radii"]
        distances = {
            name: float(_mahalanobis_distances(
                point[None, :],
                np.asarray(centers[name], dtype=float),
                np.asarray(precisions[name], dtype=float),
            )[0])
            for name in REGION_NAMES
        }
        memberships = {
            name: bool(distances[name] <= float(radii[name])) for name in REGION_NAMES
        }
        active = [name for name, inside in memberships.items() if inside]
        if active == ["real"]:
            prediction, region = 0, "real_sphere"
        elif active == ["fake"]:
            prediction, region = 1, "fake_sphere"
        elif len(active) > 1:
            prediction, region = 2, "sphere_overlap"
        else:
            prediction, region = 2, "outside_all_spheres"
        return RegionDecision(
            prediction=prediction,
            region=region,
            point=tuple(float(value) for value in point),
            distances=distances,
            memberships=memberships,
            novelty_score=self._novelty_score(evidence),
        )

    def predict(self, evidence: Evidence) -> int:
        return self.decide(evidence).prediction

    def reject(self, evidence: Evidence) -> bool:
        return self.predict(evidence) == 2

    def score(self, evidence: Evidence) -> float:
        """Continuous score used only for AUROC/risk-coverage reporting."""
        return self._novelty_score(evidence)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.artifacts, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "TrustedSphereCalibrator":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))


def selective_metrics(
    novelty_scores: list[float],
    binary_correct: list[bool],
    rejected: list[bool],
) -> dict[str, Any]:
    """Evaluate selective classification without pretending unknown is a class."""

    if not novelty_scores or not (
        len(novelty_scores) == len(binary_correct) == len(rejected)
    ):
        raise ValueError("Selective metric inputs must be non-empty and aligned")
    return {
        "known_rejection_rate": float(np.mean(rejected)),
        "risk_coverage": _risk_coverage(novelty_scores, binary_correct),
    }


def spherewise_metrics(
    targets: list[int],
    sphere_distances: list[dict[str, float]],
    radii: dict[str, float],
) -> dict[str, Any]:
    """Evaluate every sphere independently instead of using overall accuracy.

    One-vs-rest AUROC is calculated over the complete evaluation set using the
    normalized distance to one sphere as its continuous score.  Purity and
    conditional error are calculated only from samples geometrically inside
    that sphere.  The latter two quantities answer the practical question:
    "when this sphere accepts a sample, how often is it wrong?"
    """

    if not targets or len(targets) != len(sphere_distances):
        raise ValueError("targets and sphere_distances must be non-empty and aligned")
    target_array = np.asarray(targets, dtype=int)
    result: dict[str, Any] = {}
    for class_index, name in enumerate(REGION_NAMES):
        radius = float(radii[name])
        distances = np.asarray(
            [item[name] for item in sphere_distances], dtype=float
        )
        # Higher means more compatible with this sphere.  The zero boundary is
        # exactly the sphere surface; AUC remains meaningful outside the ball.
        scores = 1.0 - distances / max(radius, 1e-12)
        positives = target_array == class_index
        negatives = ~positives
        inside = distances <= radius + 1e-12
        correct_inside = inside & positives
        incorrect_inside = inside & negatives
        accepted = int(inside.sum())
        correct = int(correct_inside.sum())
        errors = int(incorrect_inside.sum())
        auc = (
            float(roc_auc_score(positives.astype(int), scores))
            if positives.any() and negatives.any() else None
        )
        purity = correct / accepted if accepted else None
        result[name] = {
            "one_vs_rest_auc": auc,
            "inside_samples": accepted,
            "correct_inside": correct,
            "errors_inside": errors,
            "purity": purity,
            "conditional_error_rate": (1.0 - purity) if purity is not None else None,
            "class_coverage": correct / max(1, int(positives.sum())),
            "false_admission_rate": errors / max(1, int(negatives.sum())),
            "class_samples": int(positives.sum()),
        }
    defined_auc = [
        metrics["one_vs_rest_auc"] for metrics in result.values()
        if metrics["one_vs_rest_auc"] is not None
    ]
    defined_purity = [
        metrics["purity"] for metrics in result.values()
        if metrics["purity"] is not None
    ]
    result["summary"] = {
        "macro_one_vs_rest_auc": (
            float(np.mean(defined_auc)) if defined_auc else None
        ),
        "macro_purity": (
            float(np.mean(defined_purity)) if defined_purity else None
        ),
    }
    return result


def _real_score(item: Evidence) -> float:
    return 0.5 * (item.real_probability + (item.real_similarity + 1.0) / 2.0)


def _fake_score(item: Evidence) -> float:
    return 0.5 * (item.fake_probability + (item.fake_similarity + 1.0) / 2.0)


def _novelty_component_matrix(
    items: Iterable[Evidence], component_names: list[str]
) -> np.ndarray:
    extractors = {
        "energy": lambda item: item.energy,
        "prototype_distance": lambda item: item.prototype_distance,
        "branch_disagreement": lambda item: item.branch_disagreement,
        "uncertainty": lambda item: 1.0 - item.confidence,
    }
    unknown_names = set(component_names) - set(extractors)
    if unknown_names:
        raise ValueError(f"Unknown novelty components: {sorted(unknown_names)}")
    return np.asarray([
        [extractors[name](item) for name in component_names]
        for item in items
    ], dtype=float)


def _evidence_coordinates(
    items: list[Evidence], novelty_scores: np.ndarray
) -> np.ndarray:
    if len(items) != len(novelty_scores):
        raise ValueError("Evidence and novelty scores must be aligned")
    return np.column_stack((
        [_real_score(item) for item in items],
        [_fake_score(item) for item in items],
        novelty_scores,
    ))


def _fit_mahalanobis_ellipsoid(
    class_points: np.ndarray, regularization: float
) -> dict[str, np.ndarray]:
    """Estimate a stable class-conditional covariance and its precision."""

    estimator = LedoitWolf().fit(class_points)
    covariance = np.asarray(estimator.covariance_, dtype=float)
    dimension = covariance.shape[0]
    average_variance = float(np.trace(covariance) / max(1, dimension))
    ridge = regularization * max(average_variance, 1.0)
    covariance = covariance + ridge * np.eye(dimension, dtype=float)
    return {
        "center": np.asarray(estimator.location_, dtype=float),
        "covariance": covariance,
        "precision": np.linalg.inv(covariance),
    }


def _mahalanobis_distances(
    points: np.ndarray, center: np.ndarray, precision: np.ndarray
) -> np.ndarray:
    delta = np.asarray(points, dtype=float) - np.asarray(center, dtype=float)
    squared = np.einsum("ni,ij,nj->n", delta, precision, delta)
    return np.sqrt(np.maximum(squared, 0.0))


def _one_vs_rest_distance_auc(
    distances: np.ndarray, labels: np.ndarray, class_index: int
) -> float:
    own = labels == class_index
    return float(roc_auc_score(own.astype(int), -distances))


def _conformal_rank(sample_count: int, target_coverage: float) -> int:
    """Return the one-based finite-sample split-conformal order statistic."""

    if sample_count <= 0:
        raise ValueError("Conformal calibration requires at least one sample")
    if not 0 < target_coverage < 1:
        raise ValueError("target_coverage must be in (0, 1)")
    return int(np.ceil((sample_count + 1) * target_coverage))


def _conformal_radius(
    distances: np.ndarray, target_coverage: float
) -> float:
    """Use the finite-sample-corrected split-conformal distance threshold."""

    values = np.asarray(distances, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("Conformal calibration distances must be non-empty")
    if not np.all(np.isfinite(values)):
        raise ValueError("Conformal calibration distances must be finite")
    rank = _conformal_rank(values.size, target_coverage)
    if rank > values.size:
        # No finite order statistic can provide the requested coverage with
        # this calibration sample size. An infinite threshold is conservative.
        return float("inf")
    return float(np.partition(values, rank - 1)[rank - 1])


def _sphere_diagnostics(
    distance_matrix: np.ndarray, labels: np.ndarray, radii: np.ndarray
) -> dict[str, Any]:
    membership = np.column_stack([
        distance_matrix[:, index] <= radii[index] + 1e-12
        for index in range(len(REGION_NAMES))
    ])
    active_count = membership.sum(axis=1)
    diagnostics: dict[str, Any] = {}
    for index, name in enumerate(REGION_NAMES):
        inside = membership[:, index]
        exclusive = inside & (active_count == 1)
        own = labels == index
        correct = int((exclusive & own).sum())
        accepted = int(exclusive.sum())
        diagnostics[name] = {
            "radius": float(radii[index]),
            "coverage": float(correct / max(1, own.sum())),
            "purity": float(correct / max(1, accepted)),
            "conditional_error_rate": float(
                1.0 - correct / max(1, accepted)
            ),
            "one_vs_rest_auc": _one_vs_rest_distance_auc(
                distance_matrix[:, index], labels, index
            ),
            "accepted_samples": accepted,
            "raw_membership_samples": int(inside.sum()),
        }
    diagnostics["outside_all_rate"] = float(np.mean(active_count == 0))
    diagnostics["overlap_rate"] = float(np.mean(active_count > 1))
    real_only = membership[:, 0] & (active_count == 1)
    fake_only = membership[:, 1] & (active_count == 1)
    diagnostics["fake_to_real_rate"] = float(
        np.mean(real_only[labels == 1]) if np.any(labels == 1) else 0.0
    )
    diagnostics["real_to_fake_rate"] = float(
        np.mean(fake_only[labels == 0]) if np.any(labels == 0) else 0.0
    )
    return diagnostics


def _risk_coverage(
    novelty_scores: list[float], binary_correct: list[bool]
) -> list[dict[str, float]]:
    order = np.argsort(novelty_scores)
    correct = np.asarray(binary_correct)[order]
    result = []
    for fraction in np.linspace(0.1, 1.0, 10):
        count = max(1, round(len(correct) * fraction))
        result.append({"coverage": float(count / len(correct)), "risk": float(1 - correct[:count].mean())})
    return result
