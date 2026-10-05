"""Neural network components for the two training stages."""

from .student import RGBHighFrequencyStudent
from .teacher import FullBandTeacher
from .outputs import DetectionOutput

__all__ = ["DetectionOutput", "FullBandTeacher", "RGBHighFrequencyStudent"]
