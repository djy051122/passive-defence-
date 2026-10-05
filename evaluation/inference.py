"""Final real/fake/unknown decision and per-image result export."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from data import InputQualityChecker
from .open_set import TrustedSphereCalibrator, PrototypeBank, extract_evidence


@torch.inference_mode()
def infer_loader(
    model: nn.Module,
    loader,
    device: torch.device,
    prototype_bank: PrototypeBank,
    calibrator: TrustedSphereCalibrator,
    energy_temperature: float,
    quality_checker: InputQualityChecker | None = None,
    fake_subclass_names: Sequence[str] = (),
) -> list[dict[str, Any]]:
    model.eval()
    records: list[dict[str, Any]] = []
    for batch in loader:
        rgb = batch["rgb"].to(device)
        h1 = batch["h1"].to(device)
        h2 = batch["h2"].to(device)
        output = model(rgb, h1, h2)
        probabilities = output.logits.softmax(dim=1)
        subclass_probabilities = (
            output.subclass_logits.softmax(dim=1)
            if output.subclass_logits is not None else None
        )
        evidence_items = extract_evidence(output, prototype_bank, energy_temperature)
        for index, evidence in enumerate(evidence_items):
            path = batch["path"][index]
            quality = quality_checker.assess_path(path) if quality_checker else None
            binary_prediction = int(probabilities[index].argmax().item())
            decision = calibrator.decide(evidence)
            prediction = 2 if quality and quality.force_unknown else decision.prediction
            decision_region = "quality_rejection" if quality and quality.force_unknown else decision.region
            fake_subtype_index = None
            fake_subtype = "N/A"
            fake_subtype_confidence = None
            if prediction == 1 and subclass_probabilities is not None:
                subtype_index = int(subclass_probabilities[index].argmax().item())
                if subtype_index < len(fake_subclass_names):
                    fake_subtype_index = subtype_index
                    fake_subtype = str(fake_subclass_names[subtype_index])
                    fake_subtype_confidence = float(
                        subclass_probabilities[index, subtype_index]
                    )
            fake_sources = batch.get("fake_source")
            records.append({
                "path": path,
                "prediction": prediction,
                "prediction_name": ("real", "fake", "unknown")[prediction],
                "binary_prediction": binary_prediction,
                "real_probability": float(probabilities[index, 0]),
                "fake_probability": float(probabilities[index, 1]),
                "novelty_score": decision.novelty_score,
                "energy": evidence.energy,
                "prototype_distance": evidence.prototype_distance,
                "branch_disagreement": evidence.branch_disagreement,
                "confidence": evidence.confidence,
                "real_prototype_similarity": evidence.real_similarity,
                "fake_prototype_similarity": evidence.fake_similarity,
                "evidence_point": list(decision.point),
                "sphere_distances": decision.distances,
                "sphere_memberships": decision.memberships,
                "decision_region": decision_region,
                "fake_subtype": fake_subtype,
                "fake_subtype_index": fake_subtype_index,
                "fake_subtype_confidence": fake_subtype_confidence,
                "fake_source": fake_sources[index] if fake_sources is not None else "",
                "quality_reasons": list(quality.reasons) if quality else [],
                "known_label": int(batch["label"][index]),
                "fake_subclass": int(batch["fake_subclass"][index]),
            })
    return records
