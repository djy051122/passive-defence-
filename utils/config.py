"""YAML configuration loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_KNOWN_CLASS_NAMES = ["real", "fake"]
EXPECTED_OUTPUT_NAMES = ["real", "fake", "unknown"]
EXPECTED_FAKE_SUBCLASS_NAMES = ["face_swap", "attribute_edit"]


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("The configuration root must be a mapping")

    required_sections = {"project", "task", "data", "wavelet", "model", "training"}
    missing = sorted(required_sections.difference(config))
    if missing:
        raise ValueError(f"Missing configuration sections: {', '.join(missing)}")

    class_names = config["task"].get("known_class_names")
    output_names = config["task"].get("output_names")
    num_classes = config["task"].get("num_known_classes")
    if (
        class_names != EXPECTED_KNOWN_CLASS_NAMES
        or output_names != EXPECTED_OUTPUT_NAMES
        or num_classes != 2
    ):
        raise ValueError(
            "Known classes must be [real, fake]; final outputs must be [real, fake, unknown]"
        )
    if config["task"].get("fake_subclass_names") != EXPECTED_FAKE_SUBCLASS_NAMES:
        raise ValueError(
            "Fake subclasses must be [face_swap, attribute_edit]"
        )
    data = config["data"]
    face_preprocessing = data.get("face_preprocessing", {})
    if not isinstance(face_preprocessing, dict):
        raise ValueError("data.face_preprocessing must be a mapping")
    if face_preprocessing.get("strategy", "face_or_center") not in {
        "face_or_center", "center_square"
    }:
        raise ValueError(
            "data.face_preprocessing.strategy must be face_or_center or center_square"
        )
    sampling = data.get("sampling", {})
    if not isinstance(sampling, dict):
        raise ValueError("data.sampling must be a mapping")
    if sampling.get("mode", "shuffle") not in {
        "shuffle", "hierarchical_balanced"
    }:
        raise ValueError(
            "data.sampling.mode must be shuffle or hierarchical_balanced"
        )
    real_fraction = float(sampling.get("real_fraction", 0.5))
    if not 0 < real_fraction < 1:
        raise ValueError("data.sampling.real_fraction must be in (0, 1)")
    if config["wavelet"].get("name") != "haar" or config["wavelet"].get("level") != 2:
        raise ValueError("The current implementation requires a two-level Haar wavelet")
    return config


def resolve_project_path(path: str | Path) -> Path:
    """Resolve config paths relative to the project directory."""

    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()
