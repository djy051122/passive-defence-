"""Independent workflow functions for every executable project step."""

from .stage1 import test_teacher, train_teacher
from .stage2 import (
    calibrate_trusted_spheres,
    detect_images,
    test_generalization,
    test_robustness,
    test_student,
    train_student,
)

__all__ = [
    "train_teacher",
    "test_teacher",
    "train_student",
    "test_student",
    "calibrate_trusted_spheres",
    "detect_images",
    "test_generalization",
    "test_robustness",
]
