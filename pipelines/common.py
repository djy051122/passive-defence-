"""Shared construction and data-loading helpers for independent workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from data import build_inference_loader, build_known_loader
from models import FullBandTeacher, RGBHighFrequencyStudent
from utils import load_config, resolve_device, resolve_project_path, seed_everything


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_STAGE1_CONFIG = PROJECT_DIR / "configs" / "stage1_teacher.yaml"
DEFAULT_STAGE2_CONFIG = PROJECT_DIR / "configs" / "stage2_student.yaml"


@dataclass(frozen=True)
class RuntimeContext:
    """Configuration, device and output paths shared by one workflow run."""

    config: dict
    device: torch.device
    output_dir: Path


def prepare_runtime(config_path: str | Path, device_override: str | None) -> RuntimeContext:
    config = load_config(config_path)
    seed_everything(config["project"]["seed"])
    device = resolve_device(device_override or config["training"]["device"])
    output_dir = resolve_project_path(config["project"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    return RuntimeContext(config=config, device=device, output_dir=output_dir)


def resolve_optional_path(value: str | Path | None, default: str | Path) -> Path:
    """Resolve CLI overrides and configured paths relative to the project root."""

    candidate = Path(value if value is not None else default).expanduser()
    return candidate.resolve() if candidate.is_absolute() else resolve_project_path(candidate)


def build_teacher_model(config: dict, pretrained: bool = False) -> FullBandTeacher:
    model = config["model"]
    return FullBandTeacher(
        num_classes=2,
        backbone=model["backbone"],
        pretrained=pretrained,
        embedding_dim=model["embedding_dim"],
        dropout=model["dropout"],
    )


def build_student_model(config: dict, pretrained: bool = False) -> RGBHighFrequencyStudent:
    model = config["model"]
    return RGBHighFrequencyStudent(
        num_classes=2,
        num_fake_subclasses=len(config["task"]["fake_subclass_names"]),
        backbone=model["backbone"],
        pretrained=pretrained,
        embedding_dim=model["embedding_dim"],
        dropout=model["dropout"],
    )


def build_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
    training = config["training"]
    return torch.optim.AdamW(
        model.parameters(),
        lr=training["learning_rate"],
        weight_decay=training["weight_decay"],
    )


def known_loader(
    config: dict,
    relative_path: str | Path,
    training: bool,
    device: torch.device,
):
    data = config["data"]
    return build_known_loader(
        resolve_project_path(Path(data["root"]) / relative_path),
        data["image_size"],
        data["batch_size"],
        data["num_workers"],
        bool(data["pin_memory"] and device.type == "cuda"),
        config["task"]["fake_subclass_names"],
        data["allowed_extensions"],
        training,
        config["project"]["seed"],
        data.get("face_preprocessing"),
        data.get("sampling"),
    )


def inference_loader(
    config: dict,
    relative_or_absolute: str | Path,
    device: torch.device,
):
    data = config["data"]
    path = Path(relative_or_absolute).expanduser()
    if path.is_absolute():
        root = path.resolve()
    else:
        project_relative = resolve_project_path(path)
        data_relative = resolve_project_path(Path(data["root"]) / path)
        root = project_relative if project_relative.is_dir() else data_relative
    return build_inference_loader(
        root,
        data["image_size"],
        data["batch_size"],
        data["num_workers"],
        bool(data["pin_memory"] and device.type == "cuda"),
        data["allowed_extensions"],
        data.get("face_preprocessing"),
    )
