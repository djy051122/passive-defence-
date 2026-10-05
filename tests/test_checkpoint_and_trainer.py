import logging
from pathlib import Path

import torch
from torch import nn

from losses import StudentOpenSetObjective, TeacherObjective
from models import DetectionOutput
from evaluation import Evidence, PrototypeBank, TrustedSphereCalibrator, infer_loader
from trainers import StudentTrainer, TeacherTrainer
from utils import load_checkpoint


class TinyTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(4, 4)
        self.head = nn.Linear(4, 2)

    def forward(self, rgb, ll2, h1, h2):
        raw = torch.stack([x.mean((1, 2, 3)) for x in (rgb, ll2, h1, h2)], 1)
        embedding = self.layer(raw)
        logits = self.head(embedding)
        return DetectionOutput(logits, embedding, logits, logits, low_logits=logits)


class TinyStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(3, 4)
        self.head = nn.Linear(4, 2)
        self.sub = nn.Linear(4, 2)

    def forward(self, rgb, h1, h2):
        raw = torch.stack([x.mean((1, 2, 3)) for x in (rgb, h1, h2)], 1)
        embedding = self.layer(raw)
        logits = self.head(embedding)
        return DetectionOutput(logits, embedding, logits, logits, subclass_logits=self.sub(embedding))


def _batch(size=6):
    return {
        "source": torch.rand(size, 3, 8, 8), "rgb": torch.rand(size, 3, 8, 8),
        "ll2": torch.rand(size, 3, 2, 2), "h1": torch.rand(size, 9, 4, 4),
        "h2": torch.rand(size, 9, 2, 2), "label": torch.tensor([0, 1, 0, 1, 0, 1]),
        "fake_subclass": torch.tensor([-1, 0, -1, 1, -1, 0]),
        "path": [str(i) for i in range(size)],
    }


def _augmentation():
    return {
        "rgb_local_mask_ratio": 0.1, "rgb_local_mask_probability": 0,
        "wavelet_local_mask_ratio": 0.1, "wavelet_local_mask_probability": 0,
        "high_subband_drop_probability": 0, "degradation_probability": 0,
        "degradation": {"jpeg_quality_range": [90, 90], "blur_sigma_range": [0, 0],
                        "noise_std_range": [0, 0], "resize_scale_range": [1, 1]},
    }


def test_two_stage_smoke_and_teacher_freeze(tmp_path: Path):
    logger = logging.getLogger("smoke")
    batch = _batch()
    teacher = TinyTeacher()
    teacher_path = tmp_path / "teacher.pth"
    teacher_trainer = TeacherTrainer(
        teacher, torch.optim.Adam(teacher.parameters()),
        TeacherObjective(.1, .1, .05, .1, .1), torch.device("cpu"),
        tmp_path / "teacher", logger, _augmentation(),
    )
    teacher_trainer.fit([batch], [batch], 1, teacher_path)
    load_checkpoint(teacher_path, teacher)

    student = TinyStudent()
    loss_config = {
        "auxiliary_weight": .1, "kd_weight": .1, "kd_temperature": 2,
        "feature_distillation_weight": .1, "subclass_weight": .1,
        "robustness_weight": .1, "feature_consistency_weight": .1,
    }
    student_path = tmp_path / "student.pth"
    student_trainer = StudentTrainer(
        student, teacher, torch.optim.Adam(student.parameters()),
        StudentOpenSetObjective(loss_config), torch.device("cpu"),
        tmp_path / "student", logger,
        _augmentation(),
    )
    student_trainer.fit([batch], [batch], 1, student_path, teacher_path)
    assert student_path.exists()
    assert all(not parameter.requires_grad for parameter in teacher.parameters())

    bank = PrototypeBank(torch.stack([torch.ones(4), -torch.ones(4)]), ["real", "fake"])
    real = [Evidence(-5 + i * .01, .05, .01, .96, .96, .04, .95, .1) for i in range(5)]
    fake = [Evidence(-5 + i * .01, .05, .01, .96, .04, .96, .1, .95) for i in range(5)]
    calibrator = TrustedSphereCalibrator.fit(real, fake, real, fake)
    records = infer_loader(
        student, [batch], torch.device("cpu"), bank, calibrator, 1.0
    )
    assert len(records) == len(batch["label"])
    assert all(record["prediction"] in (0, 1, 2) for record in records)
