"""Known-class and final three-output evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    precision_recall_fscore_support, roc_auc_score,
)

from .prediction_counter import count_predictions


def evaluate_known_predictions(
    targets: Sequence[int], predictions: Sequence[int],
    probabilities: Sequence[Sequence[float]] | None = None,
) -> dict[str, Any]:
    if len(targets) != len(predictions) or not targets:
        raise ValueError("Known targets/predictions must be non-empty and equally sized")
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, labels=[0, 1], zero_division=0
    )
    result = {
        "accuracy": float(accuracy_score(targets, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "macro_f1": float(f1.mean()),
        "per_class": {
            name: {
                "precision": float(precision[index]), "recall": float(recall[index]),
                "f1": float(f1[index]), "support": int(support[index]),
            }
            for index, name in enumerate(("real", "fake"))
        },
        "confusion_matrix": confusion_matrix(targets, predictions, labels=[0, 1]).tolist(),
    }
    if probabilities is not None:
        probability_array = np.asarray(probabilities)
        if probability_array.shape != (len(targets), 2):
            raise ValueError("probabilities must have shape [samples, 2]")
        result["roc_auc"] = float(roc_auc_score(targets, probability_array[:, 1]))
        confidence = probability_array.max(axis=1)
        correctness = np.asarray(targets) == np.asarray(predictions)
        result["ece"] = expected_calibration_error(confidence, correctness)
    return result


def expected_calibration_error(
    confidences: Sequence[float], correctness: Sequence[bool], bins: int = 15
) -> float:
    confidence = np.asarray(confidences)
    correct = np.asarray(correctness, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def evaluate_final_predictions(
    predictions: Sequence[int],
    known_targets: Sequence[int] | None = None,
    known_indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    counts = count_predictions(predictions)
    result: dict[str, Any] = {
        "prediction_counts": counts,
        "total_images": len(predictions),
    }
    if known_targets is not None and known_indices is not None:
        known_predictions = [predictions[index] for index in known_indices]
        accepted_mask = [prediction != 2 for prediction in known_predictions]
        accepted_targets = [
            target for target, accepted in zip(known_targets, accepted_mask) if accepted
        ]
        accepted_predictions = [
            prediction for prediction, accepted in zip(known_predictions, accepted_mask)
            if accepted
        ]
        result["known_rejection_rate"] = float(1 - np.mean(accepted_mask))
        if accepted_targets:
            result["accepted_known_accuracy"] = float(
                accuracy_score(accepted_targets, accepted_predictions)
            )
    return result


def evaluate_fake_subclasses(
    records: Sequence[dict[str, Any]], subclass_names: Sequence[str]
) -> dict[str, Any] | None:
    """Report conditional and end-to-end fake-family performance.

    Conditional metrics use only labeled fake images accepted by the fake
    sphere. End-to-end recall uses every labeled fake image, so rejection or a
    wrong main decision is counted as a failure.
    """

    labeled = [
        record for record in records
        if record.get("known_label") == 1 and record.get("fake_subclass", -1) >= 0
    ]
    if not labeled:
        return None
    labels = list(range(len(subclass_names)))
    accepted = [
        record for record in labeled
        if record.get("prediction") == 1
        and record.get("fake_subtype_index") is not None
    ]
    conditional: dict[str, Any] = {"samples": len(accepted)}
    if accepted:
        targets = [int(record["fake_subclass"]) for record in accepted]
        predictions = [int(record["fake_subtype_index"]) for record in accepted]
        _, recall, f1, support = precision_recall_fscore_support(
            targets, predictions, labels=labels, zero_division=0
        )
        confidences = [float(record["fake_subtype_confidence"]) for record in accepted]
        correctness = [a == b for a, b in zip(targets, predictions)]
        conditional.update({
            "macro_f1": float(f1.mean()),
            "per_class_recall": {
                name: float(recall[index])
                for index, name in enumerate(subclass_names)
            },
            "per_class_support": {
                name: int(support[index])
                for index, name in enumerate(subclass_names)
            },
            "confusion_matrix": confusion_matrix(
                targets, predictions, labels=labels
            ).tolist(),
            "ece": expected_calibration_error(confidences, correctness),
        })
    per_class_end_to_end: dict[str, float] = {}
    for index, name in enumerate(subclass_names):
        group = [record for record in labeled if record["fake_subclass"] == index]
        per_class_end_to_end[name] = (
            sum(
                record.get("prediction") == 1
                and record.get("fake_subtype_index") == index
                for record in group
            ) / len(group)
            if group else 0.0
        )
    end_to_end_correct = sum(
        record.get("prediction") == 1
        and record.get("fake_subtype_index") == record.get("fake_subclass")
        for record in labeled
    )
    return {
        "labeled_fake_samples": len(labeled),
        "conditional": conditional,
        "end_to_end_accuracy": end_to_end_correct / len(labeled),
        "end_to_end_recall_by_class": per_class_end_to_end,
    }


def format_final_report(metrics: dict[str, Any]) -> str:
    """Format sphere parameters and final four-category counts for the console."""

    counts = metrics["prediction_counts"]
    four_counts = metrics.get("four_class_counts", {})
    spheres = metrics.get("sphere_parameters", {})
    lines = ["可信球参数"]
    for name, label in (("real", "真实球"), ("fake", "伪造球")):
        item = spheres.get(name, {})
        radius = _format_optional_number(item.get("radius"))
        auc = _format_optional_number(item.get("one_vs_rest_auc"))
        consistency_auc = _format_optional_number(
            item.get("decision_consistency_auc")
        )
        source = item.get("auc_source", "不可用")
        lines.append(
            f"  {label}: 半径={radius}, 参考AUC={auc}, "
            f"参考AUC来源={source}, 模型决策一致性AUC={consistency_auc}"
        )

    ground_truth = metrics.get("ground_truth_evaluation", {})
    if ground_truth:
        lines.extend([
            "真实标签评价（标签仅用于检测后核验）",
            "  真实图片被判断为真: "
            f"{ground_truth.get('real_predicted_real', 0)}/"
            f"{ground_truth.get('real_images', 0)} = "
            f"{_format_optional_percent(ground_truth.get('real_correct_rate'))}",
            "  假图片被判断为假: "
            f"{ground_truth.get('fake_predicted_fake', 0)}/"
            f"{ground_truth.get('fake_images', 0)} = "
            f"{_format_optional_percent(ground_truth.get('fake_correct_rate'))}",
            "  无法匹配真实标签: "
            f"{ground_truth.get('unlabeled_images', 0)}",
        ])

    if four_counts:
        lines.extend([
            "最终四分类统计",
            f"  图片总数: {metrics['total_images']}",
            f"  真(real): {four_counts.get('real', 0)}",
            f"  换脸(face_swap): {four_counts.get('face_swap', 0)}",
            f"  属性编辑(attribute_edit): {four_counts.get('attribute_edit', 0)}",
            f"  未知(unknown): {four_counts.get('unknown', 0)}",
        ])
        if four_counts.get("fake_unclassified", 0):
            lines.append(
                f"  未细分假图(fake_unclassified): "
                f"{four_counts['fake_unclassified']}"
            )
    else:
        lines.extend([
            "最终真/假/未知统计",
            f"  图片总数: {metrics['total_images']}",
            f"  真(real): {counts['real']}",
            f"  假(fake): {counts['fake']}",
            f"  未知(unknown): {counts['unknown']}",
        ])
    for key in ("known_rejection_rate", "accepted_known_accuracy"):
        if key in metrics:
            lines.append(f"  {key}: {metrics[key]:.4f}")

    sphere_purity = metrics.get("sphere_purity", {})
    if sphere_purity:
        lines.append("当前检测集球内纯度（仅使用有真实标签的球内样本）")
        for name, label in (
            ("real", "真球纯度(real sphere purity)"),
            ("fake", "假球纯度(fake sphere purity)"),
        ):
            item = sphere_purity.get(name, {})
            correct = int(item.get("correct_inside_samples", 0))
            labeled = int(item.get("labeled_inside_samples", 0))
            unlabeled = int(item.get("unlabeled_inside_samples", 0))
            lines.append(
                f"  {label}: {correct}/{labeled} = "
                f"{_format_optional_percent(item.get('purity'))}; "
                f"球内未标注样本={unlabeled}"
            )
    return "\n".join(lines)


def _format_optional_number(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.4f}"


def _format_optional_percent(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.2%}"
