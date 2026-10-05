"""Independent stage-one workflows for the complete-band teacher."""

from __future__ import annotations

from pathlib import Path

from evaluation import audit_known_dataset
from losses import TeacherObjective
from trainers import TeacherTrainer
from utils import configure_logger, load_checkpoint, resolve_project_path, save_json

from .common import (
    DEFAULT_STAGE1_CONFIG,
    build_optimizer,
    build_teacher_model,
    known_loader,
    prepare_runtime,
    resolve_optional_path,
)


def _build_trainer(context, model, logger_name: str, logger=None) -> TeacherTrainer:
    config = context.config
    loss = config["loss"]
    return TeacherTrainer(
        model=model,
        optimizer=build_optimizer(model, config),
        criterion=TeacherObjective(
            loss["rgb_aux_weight"],
            loss["high_aux_weight"],
            loss["low_aux_weight"],
            loss["robustness_weight"],
            loss["feature_consistency_weight"],
            loss.get("degraded_classification_weight", 0.0),
        ),
        device=context.device,
        output_dir=context.output_dir,
        logger=logger or configure_logger(logger_name, context.output_dir),
        augmentation_config=config["augmentation"],
    )


def _audit_before_teacher_training(context, logger) -> dict:
    """Save a preflight report and stop only on structurally invalid data."""

    config = context.config
    data_root = resolve_project_path(config["data"]["root"])
    report = audit_known_dataset(
        data_root / "id",
        data_root / "calibration" / "id",
        config["task"]["fake_subclass_names"],
    )
    report_path = context.output_dir / "data_audit.json"
    save_json(report, report_path)
    duplicate_count = len(report["cross_split_duplicates"])
    logger.info(
        "数据检查完成：致命问题=%d，跨划分内容重复=%d，报告=%s",
        len(report["fatal_issues"]),
        duplicate_count,
        report_path,
    )
    if duplicate_count:
        logger.warning(
            "发现%d组跨划分内容重复；按当前设置仅记录警告并继续训练",
            duplicate_count,
        )
    if not report["passed"]:
        raise RuntimeError(
            "数据检查未通过，教师训练已停止。"
            f"致命问题={len(report['fatal_issues'])}；详见 {report_path}"
        )
    return report


def train_teacher(
    config_path: str | Path = DEFAULT_STAGE1_CONFIG,
    checkpoint_path: str | Path | None = None,
    device: str | None = None,
) -> Path:
    """Audit all known data, then train and save the complete-band teacher."""

    context = prepare_runtime(config_path, device)
    config = context.config
    logger = configure_logger("train_teacher", context.output_dir)
    _audit_before_teacher_training(context, logger)
    checkpoint = resolve_optional_path(
        checkpoint_path, config["training"]["checkpoint_path"]
    )
    model = build_teacher_model(config, pretrained=bool(config["model"]["pretrained"]))
    trainer = _build_trainer(context, model, "train_teacher", logger)
    trainer.fit(
        known_loader(config, "id/train", True, context.device),
        known_loader(config, "id/val", False, context.device),
        config["training"]["epochs"],
        checkpoint,
    )
    return checkpoint


def test_teacher(
    config_path: str | Path = DEFAULT_STAGE1_CONFIG,
    checkpoint_path: str | Path | None = None,
    device: str | None = None,
) -> dict:
    """Load and test a trained complete-band teacher."""

    context = prepare_runtime(config_path, device)
    config = context.config
    checkpoint = resolve_optional_path(
        checkpoint_path, config["training"]["checkpoint_path"]
    )
    model = build_teacher_model(config, pretrained=False)
    trainer = _build_trainer(context, model, "test_teacher")
    load_checkpoint(checkpoint, model, map_location=context.device)
    metrics = trainer.test(known_loader(config, "id/test", False, context.device))
    save_json(metrics, context.output_dir / "known_test_metrics.json")
    return metrics
