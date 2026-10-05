"""Shared batch preparation and known-class evaluation helpers."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from data import TwoLevelHaarDWT, normalize_rgb
from evaluation import evaluate_known_predictions


def move_full_batch(batch: dict[str, Any], device: torch.device):
    non_blocking = device.type == "cuda"
    return (
        batch["rgb"].to(device, non_blocking=non_blocking),
        batch["ll2"].to(device, non_blocking=non_blocking),
        batch["h1"].to(device, non_blocking=non_blocking),
        batch["h2"].to(device, non_blocking=non_blocking),
        batch["label"].to(device, non_blocking=non_blocking),
        batch["fake_subclass"].to(device, non_blocking=non_blocking),
    )


def move_student_batch(batch: dict[str, Any], device: torch.device):
    non_blocking = device.type == "cuda"
    return (
        batch["rgb"].to(device, non_blocking=non_blocking),
        batch["h1"].to(device, non_blocking=non_blocking),
        batch["h2"].to(device, non_blocking=non_blocking),
        batch["label"].to(device, non_blocking=non_blocking),
    )


def decompose_source(source: Tensor, dwt: TwoLevelHaarDWT, device: torch.device):
    bands = dwt(source)
    return (
        normalize_rgb(source.clone()).to(device),
        bands.ll2.to(device), bands.h1.to(device), bands.h2.to(device),
    )


def known_metrics(
    targets: list[int], predictions: list[int], probabilities: list[list[float]], loss: float
):
    metrics = evaluate_known_predictions(targets, predictions, probabilities)
    metrics["loss"] = float(loss)
    return metrics
