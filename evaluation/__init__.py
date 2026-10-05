"""Known-class, trusted-region, and final decision evaluation."""

from .evaluator import (
    evaluate_final_predictions,
    evaluate_fake_subclasses,
    evaluate_known_predictions,
    expected_calibration_error,
    format_final_report,
)
from .inference import infer_loader
from .data_audit import audit_known_dataset
from .open_set import (
    Evidence, PrototypeBank, RegionDecision, TrustedSphereCalibrator,
    extract_evidence, selective_metrics, spherewise_metrics,
)
from .prediction_counter import count_predictions
from .reporting import (
    attach_ground_truth_labels,
    collect_sphere_parameters,
    count_four_class_predictions,
    decision_consistency_auc,
    evaluate_ground_truth_predictions,
    evaluate_sphere_purity,
    export_detection_workbook,
    four_class_prediction,
)

__all__ = [
    "evaluate_known_predictions", "evaluate_final_predictions",
    "evaluate_fake_subclasses",
    "expected_calibration_error", "format_final_report", "count_predictions",
    "Evidence", "RegionDecision", "TrustedSphereCalibrator",
    "PrototypeBank", "extract_evidence",
    "selective_metrics", "spherewise_metrics", "infer_loader",
    "audit_known_dataset",
    "attach_ground_truth_labels", "collect_sphere_parameters",
    "count_four_class_predictions", "decision_consistency_auc",
    "evaluate_ground_truth_predictions", "evaluate_sphere_purity",
    "export_detection_workbook",
    "four_class_prediction",
]
