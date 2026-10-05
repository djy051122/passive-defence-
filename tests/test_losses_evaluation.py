import numpy as np
import torch

from evaluation import (
    Evidence, PrototypeBank, TrustedSphereCalibrator, count_predictions,
    evaluate_fake_subclasses, selective_metrics, spherewise_metrics,
)
from losses import binary_separation_loss
from evaluation.open_set import _conformal_radius, _conformal_rank


def test_binary_separation_loss_is_finite_for_mixed_and_single_class_batches():
    mixed = binary_separation_loss(
        torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]),
        torch.tensor([0, 0, 1, 1]),
    )
    single = binary_separation_loss(
        torch.tensor([[1.0, 0.0], [0.9, 0.1]]), torch.tensor([0, 0])
    )
    assert torch.isfinite(mixed)
    assert torch.isfinite(single)


def _evidence(kind: str, offset: float = 0.0) -> Evidence:
    if kind == "real":
        return Evidence(-5 + offset, 0.05, 0.01, 0.96, 0.96, 0.04, 0.95, 0.10)
    if kind == "fake":
        return Evidence(-5 + offset, 0.05, 0.01, 0.96, 0.04, 0.96, 0.10, 0.95)
    return Evidence(-1 + offset, 0.90, 0.40, 0.55, 0.52, 0.48, 0.05, 0.05)


def test_prototype_distance_and_trusted_sphere_calibration(tmp_path):
    embeddings = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    bank = PrototypeBank.fit(embeddings, torch.tensor([0, 0, 1, 1]), ["real", "fake"])
    assert torch.all(bank.distance(torch.tensor([[1.0, 0.0]])) < 0.01)
    path = tmp_path / "prototypes.pth"
    bank.save(path)
    assert PrototypeBank.load(path).names == ["real", "fake"]

    real_items = [_evidence("real", i * 0.01) for i in range(10)]
    fake_items = [_evidence("fake", i * 0.01) for i in range(10)]
    calibrator = TrustedSphereCalibrator.fit(
        real_items, fake_items, real_items, fake_items
    )
    assert calibrator.artifacts["novelty_component_names"] == [
        "prototype_distance", "branch_disagreement", "uncertainty"
    ]
    assert calibrator.predict(real_items[4]) == 0
    assert calibrator.predict(fake_items[4]) == 1
    diagnostics = calibrator.artifacts["calibration_diagnostics"]
    for name in ("real", "fake"):
        assert diagnostics[name]["coverage"] >= 0.90
        assert diagnostics[name]["purity"] >= 0.90
        assert diagnostics[name]["accepted_samples"] >= 2
    assert diagnostics["overlap_rate"] == 0.0
    assert calibrator.artifacts["distance_metric"] == "class_conditional_mahalanobis"
    assert calibrator.artifacts["radius_selection"]["method"] == (
        "class_conditional_split_conformal_order_statistic"
    )
    assert calibrator.artifacts["radius_quantile"] == 0.90
    for name in ("real", "fake"):
        selected = calibrator.artifacts["radius_selection"]["selected"][name]
        assert selected["conformal_rank"] == 10
        assert selected["effective_quantile"] == 1.0
    for name in ("real", "fake"):
        precision = np.asarray(calibrator.artifacts["precisions"][name])
        assert precision.shape == (3, 3)
        assert np.all(np.linalg.eigvalsh(precision) > 0)
    outside = Evidence(10.0, 2.0, 2.0, 0.5, 0.5, 0.5, -1.0, -1.0)
    assert calibrator.decide(outside).prediction == 2
    assert calibrator.decide(outside).region == "outside_all_spheres"


def test_trusted_sphere_calibration_reports_low_auc_without_rejecting():
    indistinguishable = [
        Evidence(-5.0, 0.2, 0.1, 0.8, 0.5, 0.5, 0.3, 0.3)
        for _ in range(6)
    ]
    calibrator = TrustedSphereCalibrator.fit(
        indistinguishable, indistinguishable,
        indistinguishable, indistinguishable,
    )
    diagnostics = calibrator.artifacts["calibration_diagnostics"]
    assert diagnostics["real"]["one_vs_rest_auc"] == 0.5
    assert diagnostics["fake"]["one_vs_rest_auc"] == 0.5


def test_conformal_radius_uses_finite_sample_corrected_order_statistic():
    distances = np.arange(1.0, 189.0)
    assert _conformal_rank(len(distances), 0.90) == 171
    assert _conformal_radius(distances, 0.90) == 171.0


def test_conformal_radius_is_conservative_when_sample_is_too_small():
    assert np.isinf(_conformal_radius(np.arange(1.0, 6.0), 0.90))


def test_final_counts_always_have_three_outputs():
    assert count_predictions([0, 1, 2, 2]) == {"real": 1, "fake": 1, "unknown": 2}


def test_selective_metrics_include_risk_coverage():
    metrics = selective_metrics(
        novelty_scores=[-2.0, -1.5, -1.0],
        binary_correct=[True, True, False],
        rejected=[False, False, True],
    )
    assert metrics["known_rejection_rate"] == 1 / 3
    assert len(metrics["risk_coverage"]) == 10


def test_spherewise_metrics_report_auc_and_inside_error():
    targets = [0, 0, 1, 1]
    distances = [
        {"real": 0.1, "fake": 2.0},
        {"real": 0.2, "fake": 1.8},
        {"real": 2.0, "fake": 0.1},
        {"real": 1.8, "fake": 0.2},
    ]
    metrics = spherewise_metrics(
        targets, distances, {"real": 0.5, "fake": 0.5}
    )
    for name in ("real", "fake"):
        assert metrics[name]["one_vs_rest_auc"] == 1.0
        assert metrics[name]["purity"] == 1.0
        assert metrics[name]["conditional_error_rate"] == 0.0


def test_fake_subclass_metrics_separate_conditional_and_end_to_end_results():
    records = [
        {"known_label": 1, "fake_subclass": 0, "prediction": 1,
         "fake_subtype_index": 0, "fake_subtype_confidence": 0.9},
        {"known_label": 1, "fake_subclass": 1, "prediction": 2,
         "fake_subtype_index": None, "fake_subtype_confidence": None},
        {"known_label": 1, "fake_subclass": 1, "prediction": 1,
         "fake_subtype_index": 0, "fake_subtype_confidence": 0.8},
    ]
    metrics = evaluate_fake_subclasses(
        records, ["face_swap", "attribute_edit"]
    )
    assert metrics["conditional"]["samples"] == 2
    assert metrics["end_to_end_accuracy"] == 1 / 3
    assert metrics["end_to_end_recall_by_class"]["face_swap"] == 1.0
