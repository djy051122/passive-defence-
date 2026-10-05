"""Independent stage-two student, calibration, detection and robustness workflows."""

from __future__ import annotations

from pathlib import Path

import torch
from torchvision.transforms import functional as TF

from data import InputQualityChecker, TwoLevelHaarDWT, degrade_batch, normalize_rgb
from evaluation import (
    PrototypeBank,
    TrustedSphereCalibrator,
    attach_ground_truth_labels,
    collect_sphere_parameters,
    count_four_class_predictions,
    decision_consistency_auc,
    evaluate_fake_subclasses,
    evaluate_final_predictions,
    evaluate_ground_truth_predictions,
    evaluate_sphere_purity,
    export_detection_workbook,
    extract_evidence,
    format_final_report,
    four_class_prediction,
    infer_loader,
    selective_metrics,
    spherewise_metrics,
)
from losses import StudentOpenSetObjective
from trainers import StudentTrainer
from utils import configure_logger, load_checkpoint, resolve_project_path, save_json

from .common import (
    DEFAULT_STAGE2_CONFIG,
    build_optimizer,
    build_student_model,
    build_teacher_model,
    known_loader,
    prepare_runtime,
    resolve_optional_path,
    inference_loader,
)


def _load_student(context, checkpoint_path: str | Path | None):
    config = context.config
    checkpoint = resolve_optional_path(
        checkpoint_path, config["training"]["checkpoint_path"]
    )
    student = build_student_model(config, pretrained=False)
    try:
        load_checkpoint(checkpoint, student, map_location=context.device)
    except RuntimeError as error:
        raise RuntimeError(
            "Student checkpoint is incompatible with the separate z_bin/z_sub "
            "architecture. Retrain the student and recalibrate trusted spheres."
        ) from error
    student.to(context.device).eval()
    return student, checkpoint


def _initialize_student_encoders_from_teacher(student, teacher) -> None:
    """Warm-start deployable branches from their stage-one counterparts."""

    student.rgb_encoder.load_state_dict(teacher.rgb_encoder.state_dict())
    student.h1_encoder.load_state_dict(teacher.h1_encoder.state_dict())
    student.h2_encoder.load_state_dict(teacher.h2_encoder.state_dict())


def train_student(
    config_path: str | Path = DEFAULT_STAGE2_CONFIG,
    checkpoint_path: str | Path | None = None,
    teacher_checkpoint_path: str | Path | None = None,
    device: str | None = None,
) -> Path:
    """Train the RGB-high-frequency student with a frozen teacher."""

    context = prepare_runtime(config_path, device)
    config = context.config
    student_checkpoint = resolve_optional_path(
        checkpoint_path, config["training"]["checkpoint_path"]
    )
    teacher_checkpoint = resolve_optional_path(
        teacher_checkpoint_path, config["teacher"]["checkpoint_path"]
    )
    student = build_student_model(
        config, pretrained=bool(config["model"]["pretrained"])
    )
    teacher = build_teacher_model(config, pretrained=False)
    load_checkpoint(teacher_checkpoint, teacher, map_location=context.device)
    if bool(config["model"].get("initialize_encoders_from_teacher", True)):
        _initialize_student_encoders_from_teacher(student, teacher)
    logger = configure_logger("train_student", context.output_dir)
    train_loader = known_loader(config, "id/train", True, context.device)
    subclass_counts = train_loader.dataset.fake_subclass_counts()
    if subclass_counts["unlabeled"]:
        logger.warning(
            "%d fake training images have no three-family subtype label; they train "
            "the binary detector but not the subtype head.",
            subclass_counts["unlabeled"],
        )
    trainer = StudentTrainer(
        student=student,
        teacher=teacher,
        optimizer=build_optimizer(student, config),
        criterion=StudentOpenSetObjective(config["loss"]),
        device=context.device,
        output_dir=context.output_dir,
        logger=logger,
        augmentation_config=config["augmentation"],
    )
    trainer.fit(
        train_loader,
        known_loader(config, "id/val", False, context.device),
        config["training"]["epochs"],
        student_checkpoint,
        teacher_checkpoint,
    )
    return student_checkpoint


def test_student(
    config_path: str | Path = DEFAULT_STAGE2_CONFIG,
    checkpoint_path: str | Path | None = None,
    device: str | None = None,
) -> dict:
    """Evaluate the trained student only on the known real/fake test set."""

    context = prepare_runtime(config_path, device)
    config = context.config
    student, _ = _load_student(context, checkpoint_path)
    trainer = StudentTrainer(
        student=student,
        teacher=None,
        optimizer=build_optimizer(student, config),
        criterion=StudentOpenSetObjective(config["loss"]),
        device=context.device,
        output_dir=context.output_dir,
        logger=configure_logger("test_student", context.output_dir),
        augmentation_config=config["augmentation"],
    )
    return trainer.test_known(known_loader(config, "id/test", False, context.device))


@torch.inference_mode()
def _fit_prototypes(model, loader, device):
    model.eval()
    embeddings, group_ids = [], []
    for batch in loader:
        output = model(
            batch["rgb"].to(device), batch["h1"].to(device), batch["h2"].to(device)
        )
        for index, label in enumerate(batch["label"].tolist()):
            embeddings.append(output.embedding[index].cpu())
            group_ids.append(label)
    if not embeddings:
        raise RuntimeError("Cannot fit prototypes from an empty known training set")
    return PrototypeBank.fit(
        torch.stack(embeddings), torch.tensor(group_ids), ["real", "fake"]
    )


@torch.inference_mode()
def _collect_evidence(model, loader, device, bank, temperature):
    model.eval()
    evidence, predictions, targets = [], [], []
    for batch in loader:
        output = model(
            batch["rgb"].to(device), batch["h1"].to(device), batch["h2"].to(device)
        )
        evidence.extend(extract_evidence(output, bank, temperature))
        predictions.extend(output.logits.argmax(dim=1).cpu().tolist())
        targets.extend(batch["label"].tolist())
    return evidence, predictions, targets


def calibrate_trusted_spheres(
    config_path: str | Path = DEFAULT_STAGE2_CONFIG,
    checkpoint_path: str | Path | None = None,
    device: str | None = None,
) -> dict:
    """Fit trusted real/fake spheres using only real/fake calibration data."""

    context = prepare_runtime(config_path, device)
    config = context.config
    student, _ = _load_student(context, checkpoint_path)
    calibration = config["calibration"]
    temperature = config["loss"]["energy_temperature"]
    bank = _fit_prototypes(
        student,
        known_loader(config, "id/train", False, context.device),
        context.device,
    )
    fit_evidence, _, fit_targets = _collect_evidence(
        student,
        known_loader(config, "id/train", False, context.device),
        context.device,
        bank,
        temperature,
    )
    id_evidence, _, id_targets = _collect_evidence(
        student,
        known_loader(config, "calibration/id", False, context.device),
        context.device,
        bank,
        temperature,
    )
    real_fit_evidence = [
        evidence for evidence, target in zip(fit_evidence, fit_targets) if target == 0
    ]
    fake_fit_evidence = [
        evidence for evidence, target in zip(fit_evidence, fit_targets) if target == 1
    ]
    real_calibration_evidence = [
        evidence for evidence, target in zip(id_evidence, id_targets) if target == 0
    ]
    fake_calibration_evidence = [
        evidence for evidence, target in zip(id_evidence, id_targets) if target == 1
    ]
    fitted = TrustedSphereCalibrator.fit(
        real_fit_evidence,
        fake_fit_evidence,
        real_calibration_evidence,
        fake_calibration_evidence,
        radius_quantile=float(calibration.get("radius_quantile", 0.90)),
        covariance_regularization=float(
            calibration.get("covariance_regularization", 1e-4)
        ),
        include_energy=bool(calibration.get("include_energy", False)),
    )
    # Only replace the paired prototype/calibration artifacts after every
    # trust constraint has passed.  A failed calibration therefore leaves the
    # last usable artifact pair intact.
    bank.save(resolve_project_path(calibration["prototypes_path"]))
    fitted.save(resolve_project_path(calibration["artifacts_path"]))
    configure_logger("calibrate_trusted_spheres", context.output_dir).info(
        "Trusted real/fake Mahalanobis ellipsoid calibration complete | diagnostics=%s",
        fitted.artifacts["calibration_diagnostics"],
    )
    return fitted.artifacts


def _quality_checker(config: dict):
    quality = config["quality"]
    if not quality["enabled"]:
        return None
    return InputQualityChecker(
        quality["minimum_original_side"],
        quality["minimum_contrast_std"],
        quality["minimum_blur_variance"],
        quality["require_single_face"],
    )


def detect_images(
    config_path: str | Path = DEFAULT_STAGE2_CONFIG,
    checkpoint_path: str | Path | None = None,
    input_dir: str | Path | None = None,
    device: str | None = None,
) -> tuple[dict, list[dict]]:
    """Detect real/fake/unknown images and save per-image JSON records."""

    context = prepare_runtime(config_path, device)
    config = context.config
    student, _ = _load_student(context, checkpoint_path)
    calibration = config["calibration"]
    bank = PrototypeBank.load(resolve_project_path(calibration["prototypes_path"]))
    fitted = TrustedSphereCalibrator.load(
        resolve_project_path(calibration["artifacts_path"])
    )
    temperature = config["loss"]["energy_temperature"]
    checker = _quality_checker(config)

    if input_dir is not None:
        loaders = [inference_loader(config, input_dir, context.device)]
    else:
        loaders = [known_loader(config, "id/test", False, context.device)]

    records: list[dict] = []
    for loader in loaders:
        records.extend(
            infer_loader(
                student, loader, context.device, bank, fitted, temperature, checker,
                fake_subclass_names=config["task"]["fake_subclass_names"],
            )
        )

    predictions = [record["prediction"] for record in records]
    known_records = [record for record in records if record["known_label"] >= 0]
    known_indices = [
        index for index, record in enumerate(records) if record["known_label"] >= 0
    ]
    metrics = evaluate_final_predictions(
        predictions,
        [record["known_label"] for record in known_records] if known_records else None,
        known_indices if known_records else None,
    )
    # Final open-set reporting is sphere-centric.  Overall/accepted accuracy is
    # deliberately not used as the main claim because it can hide an unsafe
    # sphere with many false admissions.
    metrics.pop("accepted_known_accuracy", None)
    if known_records:
        metrics.update(selective_metrics(
            [record["novelty_score"] for record in known_records],
            [
                record["binary_prediction"] == record["known_label"]
                for record in known_records
            ],
            [record["prediction"] == 2 for record in known_records],
        ))
        metrics["spherewise_metrics"] = spherewise_metrics(
            [record["known_label"] for record in known_records],
            [record["sphere_distances"] for record in known_records],
            fitted.artifacts["radii"],
        )
    subclass_metrics = {}
    for index, name in enumerate(config["task"]["fake_subclass_names"]):
        group = [record for record in known_records if record["fake_subclass"] == index]
        if group:
            subclass_metrics[name] = {
                "images": len(group),
                "fake_recall": sum(record["prediction"] == 1 for record in group)
                / len(group),
                "unknown_rate": sum(record["prediction"] == 2 for record in group)
                / len(group),
            }
    if subclass_metrics:
        metrics["fake_subclass_metrics"] = subclass_metrics
    detailed_subclass_metrics = evaluate_fake_subclasses(
        known_records, config["task"]["fake_subclass_names"]
    )
    if detailed_subclass_metrics is not None:
        metrics["fake_subclass_evaluation"] = detailed_subclass_metrics

    region_counts: dict[str, int] = {}
    for record in records:
        region = record["decision_region"]
        region_counts[region] = region_counts.get(region, 0) + 1
    metrics["decision_region_counts"] = region_counts
    if known_records:
        fake_records = [record for record in known_records if record["known_label"] == 1]
        real_records = [record for record in known_records if record["known_label"] == 0]
        metrics["sphere_error_rates"] = {
            "fake_to_real_rate": (
                sum(record["prediction"] == 0 for record in fake_records) / len(fake_records)
                if fake_records else 0.0
            ),
            "real_to_fake_rate": (
                sum(record["prediction"] == 1 for record in real_records) / len(real_records)
                if real_records else 0.0
            ),
        }

    attach_ground_truth_labels(
        records, config["task"]["fake_subclass_names"]
    )
    for record in records:
        record["four_class_prediction"] = four_class_prediction(record)
    metrics["four_class_counts"] = count_four_class_predictions(records)
    metrics["ground_truth_evaluation"] = evaluate_ground_truth_predictions(records)
    metrics["sphere_purity"] = evaluate_sphere_purity(records)
    metrics["decision_consistency_auc"] = decision_consistency_auc(
        records, fitted.artifacts
    )
    metrics["sphere_parameters"] = collect_sphere_parameters(
        fitted.artifacts, metrics
    )
    excel_path = export_detection_workbook(
        records,
        metrics,
        fitted.artifacts,
        context.output_dir / "detection_results.xlsx",
    )
    metrics["excel_report"] = str(excel_path)

    save_json({"records": records}, context.output_dir / "predictions.json")
    save_json(metrics, context.output_dir / "open_set_test_metrics.json")
    print(format_final_report(metrics))
    for record in records:
        subtype_confidence = record["fake_subtype_confidence"]
        confidence_text = (
            f"{subtype_confidence:.4f}" if subtype_confidence is not None else "N/A"
        )
        four_class_name = {
            "real": "真(real)",
            "face_swap": "换脸(face_swap)",
            "attribute_edit": "属性编辑(attribute_edit)",
            "unknown": "未知(unknown)",
            "fake_unclassified": "未细分假图(fake_unclassified)",
        }[record["four_class_prediction"]]
        truth_name = {
            "real": "真(real)",
            "fake": "假(fake)",
            "unlabeled": "未标注",
        }[record["ground_truth_name"]]
        if record["ground_truth_fake_subtype"] != "N/A":
            truth_name = f"{truth_name}/{record['ground_truth_fake_subtype']}"
        correctness = record["main_prediction_correct"]
        correctness_text = (
            "无法评价" if correctness is None else "是" if correctness else "否"
        )
        print(
            f"{Path(record['path']).name} | 真实标签={truth_name} | "
            f"模型判断={four_class_name} | 主判断正确={correctness_text} | "
            f"子类型置信度={confidence_text} | "
            f"d_R={record['sphere_distances']['real']:.4f} | "
            f"d_F={record['sphere_distances']['fake']:.4f} | "
            f"区域={record['decision_region']}"
        )
    print(f"Excel报告: {excel_path}")
    return metrics, records


@torch.inference_mode()
def _collect_degraded_sphere_observations(
    student,
    loader,
    context,
    bank,
    fitted,
    temperature: float,
    degradation: dict,
    clean: bool,
    quality_checker: InputQualityChecker | None,
) -> tuple[list[int], list[dict[str, float]], dict[str, int], list[int]]:
    dwt = TwoLevelHaarDWT()
    targets: list[int] = []
    distances: list[dict[str, float]] = []
    region_counts: dict[str, int] = {}
    predictions: list[int] = []
    for batch in loader:
        source = (
            batch["source"]
            if clean else degrade_batch(batch["source"], 1.0, degradation)
        )
        bands = dwt(source)
        output = student(
            normalize_rgb(source).to(context.device),
            bands.h1.to(context.device),
            bands.h2.to(context.device),
        )
        evidence = extract_evidence(output, bank, temperature)
        for index, item in enumerate(evidence):
            decision = fitted.decide(item)
            quality = (
                quality_checker.assess_image(
                    TF.to_pil_image(source[index].detach().cpu().clamp(0, 1))
                )
                if quality_checker else None
            )
            region = (
                "quality_rejection"
                if quality and quality.force_unknown else decision.region
            )
            prediction = 2 if quality and quality.force_unknown else decision.prediction
            targets.append(int(batch["label"][index]))
            distances.append(decision.distances)
            predictions.append(prediction)
            region_counts[region] = region_counts.get(region, 0) + 1
    return targets, distances, region_counts, predictions


@torch.inference_mode()
def test_robustness(
    config_path: str | Path = DEFAULT_STAGE2_CONFIG,
    checkpoint_path: str | Path | None = None,
    device: str | None = None,
) -> dict:
    """Evaluate per-sphere AUC and purity under automatic image degradations."""

    context = prepare_runtime(config_path, device)
    config = context.config
    student, _ = _load_student(context, checkpoint_path)
    bank = PrototypeBank.load(resolve_project_path(config["calibration"]["prototypes_path"]))
    fitted = TrustedSphereCalibrator.load(
        resolve_project_path(config["calibration"]["artifacts_path"])
    )
    known_test = known_loader(config, "id/test", False, context.device)
    checker = _quality_checker(config)
    conditions = {
        "clean": {"jpeg_quality_range": [100, 100], "blur_sigma_range": [0, 0], "noise_std_range": [0, 0], "resize_scale_range": [1, 1]},
        "jpeg_70": {"jpeg_quality_range": [70, 70], "blur_sigma_range": [0, 0], "noise_std_range": [0, 0], "resize_scale_range": [1, 1]},
        "double_jpeg_85": {"jpeg_quality_range": [85, 85], "jpeg_passes": 2, "blur_sigma_range": [0, 0], "noise_std_range": [0, 0], "resize_scale_range": [1, 1]},
        "blur_1.5": {"jpeg_quality_range": [100, 100], "blur_sigma_range": [1.5, 1.5], "noise_std_range": [0, 0], "resize_scale_range": [1, 1]},
        "noise_0.03": {"jpeg_quality_range": [100, 100], "blur_sigma_range": [0, 0], "noise_std_range": [0.03, 0.03], "resize_scale_range": [1, 1]},
        "resize_0.5": {"jpeg_quality_range": [100, 100], "blur_sigma_range": [0, 0], "noise_std_range": [0, 0], "resize_scale_range": [0.5, 0.5]},
        "crop_0.8": {"jpeg_quality_range": [100, 100], "blur_sigma_range": [0, 0], "noise_std_range": [0, 0], "resize_scale_range": [1, 1], "crop_ratio_range": [0.8, 0.8]},
    }
    results = {}
    for name, degradation in conditions.items():
        targets: list[int] = []
        distances: list[dict[str, float]] = []
        region_counts: dict[str, int] = {}
        predictions: list[int] = []
        for loader in (known_test,):
            part_targets, part_distances, part_regions, part_predictions = (
                _collect_degraded_sphere_observations(
                    student, loader, context, bank, fitted,
                    config["loss"]["energy_temperature"], degradation,
                    clean=name == "clean",
                    quality_checker=checker,
                )
            )
            targets.extend(part_targets)
            distances.extend(part_distances)
            predictions.extend(part_predictions)
            for region, count in part_regions.items():
                region_counts[region] = region_counts.get(region, 0) + count
        results[name] = {
            "evaluated_samples": len(targets),
            "decision_region_counts": region_counts,
            "unknown_rate": (
                sum(prediction == 2 for prediction in predictions)
                / max(1, len(predictions))
            ),
            "quality_rejection_rate": (
                region_counts.get("quality_rejection", 0) / max(1, len(predictions))
            ),
            "spherewise_metrics": spherewise_metrics(
                targets,
                distances,
                fitted.artifacts["radii"],
            ),
        }
    save_json(results, context.output_dir / "robustness_metrics.json")
    return results


@torch.inference_mode()
def test_generalization(
    config_path: str | Path = DEFAULT_STAGE2_CONFIG,
    checkpoint_path: str | Path | None = None,
    data_root: str | Path | None = None,
    device: str | None = None,
) -> dict:
    """Evaluate labeled cross-dataset domains with the calibrated sphere rule.

    Expected layout::

        dataset/generalization/<domain>/real/*
        dataset/generalization/<domain>/fake/*

    A root that directly contains ``real`` and ``fake`` is also accepted as one
    domain.  Unknown predictions count as rejections rather than binary errors,
    and reports per-sphere AUC, purity, conditional error, and class coverage.
    """

    context = prepare_runtime(config_path, device)
    config = context.config
    student, _ = _load_student(context, checkpoint_path)
    bank = PrototypeBank.load(resolve_project_path(config["calibration"]["prototypes_path"]))
    fitted = TrustedSphereCalibrator.load(
        resolve_project_path(config["calibration"]["artifacts_path"])
    )
    root = (
        Path(data_root).expanduser().resolve()
        if data_root is not None
        else resolve_project_path(Path(config["data"]["root"]) / "generalization")
    )
    if not root.is_dir():
        raise FileNotFoundError(
            f"Generalization data not found: {root}. Expected <domain>/real and <domain>/fake."
        )
    domains = [root] if (root / "real").is_dir() and (root / "fake").is_dir() else [
        path for path in sorted(root.iterdir())
        if path.is_dir() and (path / "real").is_dir() and (path / "fake").is_dir()
    ]
    if not domains:
        raise RuntimeError(f"No labeled generalization domains found under {root}")

    results: dict[str, dict] = {}
    checker = _quality_checker(config)
    for domain in domains:
        loader = known_loader(config, domain, False, context.device)
        records = infer_loader(
            student, loader, context.device, bank, fitted,
            config["loss"]["energy_temperature"], checker,
            fake_subclass_names=config["task"]["fake_subclass_names"],
        )
        total = len(records)
        real = [record for record in records if record["known_label"] == 0]
        fake = [record for record in records if record["known_label"] == 1]
        targets = [record["known_label"] for record in records]
        region_counts: dict[str, int] = {}
        for record in records:
            region = record["decision_region"]
            region_counts[region] = region_counts.get(region, 0) + 1
        results[domain.name] = {
            "images": total,
            "decision_region_counts": region_counts,
            "spherewise_metrics": spherewise_metrics(
                targets,
                [record["sphere_distances"] for record in records],
                fitted.artifacts["radii"],
            ),
            "fake_to_real_rate": sum(record["prediction"] == 0 for record in fake) / max(1, len(fake)),
        }
    save_json(results, context.output_dir / "generalization_metrics.json")
    return results
