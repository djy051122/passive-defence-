"""Stage-two distillation, robustness, and fake-subclass training."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from data import TwoLevelHaarDWT, degrade_batch
from losses import StudentOpenSetObjective
from utils.checkpoint import save_checkpoint
from utils.logger import save_json
from .common import decompose_source, known_metrics, move_full_batch, move_student_batch


class StudentTrainer:
    def __init__(
        self, student: nn.Module, teacher: nn.Module | None, optimizer: Optimizer,
        criterion: StudentOpenSetObjective, device: torch.device,
        output_dir: str | Path, logger: logging.Logger,
        augmentation_config: dict[str, Any],
    ) -> None:
        self.student = student.to(device)
        self.teacher = teacher.to(device) if teacher is not None else None
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        self.augmentation = augmentation_config
        self.dwt = TwoLevelHaarDWT()
        if self.teacher is not None:
            self.teacher.eval()
            for parameter in self.teacher.parameters():
                parameter.requires_grad_(False)

    def fit(
        self, train_loader, validation_loader, epochs: int,
        checkpoint_path, teacher_checkpoint_path,
    ):
        if self.teacher is None:
            raise RuntimeError("A frozen complete-band teacher is required for training")
        best = float("-inf")
        history = []
        for epoch in range(1, epochs + 1):
            train_losses = self._train_epoch(train_loader)
            validation = self.evaluate(validation_loader)
            history.append({"epoch": epoch, "train": train_losses, "validation": validation})
            self.logger.info(
                "Epoch %d/%d | total=%.4f main=%.4f kd=%.4f | val_macro_f1=%.4f",
                epoch, epochs, train_losses["total"], train_losses["main"],
                train_losses["kd"], validation["macro_f1"],
            )
            if validation["macro_f1"] > best:
                best = validation["macro_f1"]
                save_checkpoint(
                    checkpoint_path, self.student, self.optimizer, epoch, best,
                    metadata={
                        "stage": "rgb_high_student", "known_classes": ["real", "fake"],
                        "architecture_version": 2,
                        "teacher_checkpoint": str(teacher_checkpoint_path),
                    },
                )
        save_json({"epochs": history}, self.output_dir / "history.json")
        return history

    def _train_epoch(self, id_loader):
        assert self.teacher is not None
        self.student.train()
        self.teacher.eval()
        totals: dict[str, float] = {}
        sample_count = 0
        for batch in id_loader:
            rgb, ll2, h1, h2, labels, subclasses = move_full_batch(batch, self.device)
            with torch.no_grad():
                teacher_output = self.teacher(rgb, ll2, h1, h2)
            degraded_source = degrade_batch(
                batch["source"], self.augmentation["degradation_probability"],
                self.augmentation["degradation"],
            )
            degraded_rgb, _, degraded_h1, degraded_h2 = decompose_source(
                degraded_source, self.dwt, self.device
            )
            self.optimizer.zero_grad(set_to_none=True)
            # Stage two always sees the same complete RGB/H1/H2 interface used
            # at deployment.  Masking and subband dropout belong to stage one.
            student_output = self.student(rgb, h1, h2)
            degraded_output = self.student(degraded_rgb, degraded_h1, degraded_h2)
            losses = self.criterion(
                student_output, teacher_output, labels, subclasses,
                degraded_student=degraded_output,
                robustness_reference=student_output,
            )
            losses["total"].backward()
            self.optimizer.step()
            batch_size = labels.shape[0]
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach()) * batch_size
            sample_count += batch_size
        if not sample_count:
            raise RuntimeError("Student training loader is empty")
        return {name: value / sample_count for name, value in totals.items()}

    @torch.inference_mode()
    def evaluate(self, loader):
        self.student.eval()
        targets, predictions, probabilities = [], [], []
        total_loss = 0.0
        total = 0
        for batch in loader:
            rgb, h1, h2, labels = move_student_batch(batch, self.device)
            output = self.student(rgb, h1, h2)
            loss = torch.nn.functional.cross_entropy(output.logits, labels)
            total_loss += float(loss) * labels.shape[0]
            total += labels.shape[0]
            targets.extend(labels.cpu().tolist())
            predictions.extend(output.logits.argmax(dim=1).cpu().tolist())
            probabilities.extend(output.logits.softmax(dim=1).cpu().tolist())
        if not total:
            raise RuntimeError("Student evaluation loader is empty")
        return known_metrics(targets, predictions, probabilities, total_loss / total)

    def test_known(self, loader):
        metrics = self.evaluate(loader)
        save_json(metrics, self.output_dir / "known_test_metrics.json")
        return metrics
