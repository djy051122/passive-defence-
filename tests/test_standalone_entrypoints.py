"""Checks for the one-script-per-step command structure."""

from __future__ import annotations

from pathlib import Path

import pytest


PROJECT_DIR = Path(__file__).resolve().parents[1]
ENTRYPOINTS = [
    "train_teacher.py",
    "test_teacher.py",
    "train_student.py",
    "test_student.py",
    "circle.py",
    "detect_images.py",
    "test_robustness.py",
    "test_generalization.py",
]


@pytest.mark.parametrize("filename", ENTRYPOINTS)
def test_entrypoint_is_thin_and_has_no_mode_switch(filename: str):
    source = (PROJECT_DIR / filename).read_text(encoding="utf-8")
    assert "def main()" in source
    assert "--" + "mode" not in source
    assert source.count("run_") == 2


def test_project_documentation_has_no_mode_switch():
    source = (PROJECT_DIR / "README.md").read_text(encoding="utf-8")
    assert "--" + "mode" not in source
