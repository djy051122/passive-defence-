"""Stage-one complete-band teacher training."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from data import (
    TwoLevelHaarDWT, apply_local_mask, degrade_batch, drop_random_high_subband,
)
from evaluation import evaluate_known_predictions
from losses import TeacherObjective
from utils.checkpoint import save_checkpoint
from utils.logger import save_json
from .common import decompose_source, known_metrics, move_full_batch


class TeacherTrainer:
    def __init__(
        self, model: nn.Module, optimizer: Optimizer, criterion: TeacherObjective,
        device: torch.device, output_dir: str | Path, logger: logging.Logger,
        augmentation_config: dict[str, Any],
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        self.augmentation = augmentation_config
        self.dwt = TwoLevelHaarDWT()

    def fit(self, train_loader, validation_loader, epochs: int, checkpoint_path):
        best = float("-inf")
        history = []
        for epoch in range(1, epochs + 1):
            train_losses = self._train_epoch(train_loader)
            validation = self.evaluate(validation_loader)
            history.append({"epoch": epoch, "train": train_losses, "validation": validation})
            self.logger.info(
                "Epoch %d/%d | total=%.4f main=%.4f robust=%.4f | val_macro_f1=%.4f",
                epoch, epochs, train_losses["total"], train_losses["main"],
                train_losses["robustness"], validation["macro_f1"],
            )
            if validation["macro_f1"] > best:
                best = validation["macro_f1"]
                save_checkpoint(
                    checkpoint_path, self.model, self.optimizer, epoch, best,
                    metadata={"stage": "full_band_teacher", "known_classes": ["real", "fake"]},
                )
        save_json({"epochs": history}, self.output_dir / "history.json")
        return history

    def _train_epoch(self, loader):
        self.model.train()
        totals: dict[str, float] = {}
        sample_count = 0
        for batch in loader:
            rgb, ll2, h1, h2, labels, _ = move_full_batch(batch, self.device)
            clean_inputs = (rgb, ll2, h1, h2)
            rgb = apply_local_mask(
                rgb, self.augmentation["rgb_local_mask_ratio"],
                self.augmentation["rgb_local_mask_probability"],
            )
            ll2 = apply_local_mask(
                ll2, self.augmentation["wavelet_local_mask_ratio"],
                self.augmentation["wavelet_local_mask_probability"],
            )
            h1 = drop_random_high_subband(
                apply_local_mask(h1, self.augmentation["wavelet_local_mask_ratio"], self.augmentation["wavelet_local_mask_probability"]),
                self.augmentation["high_subband_drop_probability"],
            )
            h2 = drop_random_high_subband(
                apply_local_mask(h2, self.augmentation["wavelet_local_mask_ratio"], self.augmentation["wavelet_local_mask_probability"]),
                self.augmentation["high_subband_drop_probability"],
            )
            degraded_source = degrade_batch(
                batch["source"], self.augmentation["degradation_probability"],
                self.augmentation["degradation"],
            )
            degraded_inputs = decompose_source(degraded_source, self.dwt, self.device)
            self.optimizer.zero_grad(set_to_none=True)
            output = self.model(rgb, ll2, h1, h2)
            # The clean prediction is a fixed consistency target for this
            # update; detaching it prevents both ends of the constraint from
            # drifting together.
            with torch.no_grad():
                clean_output = self.model(*clean_inputs)
            degraded_output = self.model(*degraded_inputs)
            losses = self.criterion(
                output, labels, degraded_output,
                robustness_reference=clean_output,
            )
            losses["total"].backward()
            self.optimizer.step()
            batch_size = labels.shape[0]
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach()) * batch_size
            sample_count += batch_size
        if not sample_count:
            raise RuntimeError("Teacher training loader is empty")
        return {name: value / sample_count for name, value in totals.items()}

    @torch.inference_mode()
    def evaluate(self, loader):
        self.model.eval()
        targets, predictions, probabilities = [], [], []
        branch_predictions = {"rgb": [], "high": [], "low": []}
        branch_probabilities = {"rgb": [], "high": [], "low": []}
        total_loss = 0.0
        total = 0
        for batch in loader:
            rgb, ll2, h1, h2, labels, _ = move_full_batch(batch, self.device)
            output = self.model(rgb, ll2, h1, h2)
            loss = torch.nn.functional.cross_entropy(output.logits, labels)
            total_loss += float(loss) * labels.shape[0]
            total += labels.shape[0]
            targets.extend(labels.cpu().tolist())
            predictions.extend(output.logits.argmax(dim=1).cpu().tolist())
            probabilities.extend(output.logits.softmax(dim=1).cpu().tolist())
            branch_outputs = {
                "rgb": output.rgb_logits,
                "high": output.high_logits,
                "low": output.low_logits,
            }
            for name, logits in branch_outputs.items():
                if logits is None:
                    raise ValueError(f"Teacher output is missing {name} logits")
                branch_predictions[name].extend(logits.argmax(dim=1).cpu().tolist())
                branch_probabilities[name].extend(logits.softmax(dim=1).cpu().tolist())
        if not total:
            raise RuntimeError("Teacher evaluation loader is empty")
        metrics = known_metrics(
            targets, predictions, probabilities, total_loss / total
        )
        metrics["branch_metrics"] = {
            name: evaluate_known_predictions(
                targets, branch_predictions[name], branch_probabilities[name]
            )
            for name in ("rgb", "high", "low")
        }
        return metrics

    def test(self, loader):
        metrics = self.evaluate(loader)
        save_json(metrics, self.output_dir / "test_metrics.json")
        return metrics
