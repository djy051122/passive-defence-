"""Count predicted real, fake, and unknown images."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence


def count_predictions(
    predictions: Iterable[int],
    class_names: Sequence[str] = ("real", "fake", "unknown"),
) -> dict[str, int]:
    if len(class_names) != 3:
        raise ValueError("Exactly three class names are required")
    raw_counts = Counter(int(prediction) for prediction in predictions)
    invalid = sorted(index for index in raw_counts if index < 0 or index >= len(class_names))
    if invalid:
        raise ValueError(f"Invalid predicted class indices: {invalid}")
    return {name: raw_counts[index] for index, name in enumerate(class_names)}

