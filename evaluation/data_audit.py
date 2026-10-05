"""Audit format shortcuts and cross-split duplicate leakage."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from PIL import Image


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def audit_known_dataset(
    id_root: str | Path,
    calibration_root: str | Path | None = None,
    expected_fake_subclasses: Sequence[str] = (),
) -> dict[str, Any]:
    """Audit every known-data partition before teacher training starts."""

    root = Path(id_root)
    partitions = [(name, root / name) for name in ("train", "val", "test")]
    if calibration_root is not None:
        partitions.append(("calibration", Path(calibration_root)))
    report: dict[str, Any] = {
        "splits": {},
        "cross_split_duplicates": [],
        "fatal_issues": [],
    }
    hashes: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for split, split_root in partitions:
        split_report = {}
        for class_name in ("real", "fake"):
            class_root = split_root / class_name
            extension_counts: Counter[str] = Counter()
            resolution_counts: Counter[str] = Counter()
            fake_subclass_counts: Counter[str] = Counter()
            all_files = [path for path in class_root.rglob("*") if path.is_file()]
            files = [
                path for path in all_files
                if path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
            ]
            if not class_root.is_dir():
                report["fatal_issues"].append(
                    f"{split}: missing class directory {class_name}"
                )
            elif not files:
                report["fatal_issues"].append(
                    f"{split}: class {class_name} contains no images"
                )
            for path in files:
                extension_counts[path.suffix.lower()] += 1
                try:
                    with Image.open(path) as image:
                        image.verify()
                        resolution_counts[f"{image.width}x{image.height}"] += 1
                except OSError:
                    resolution_counts["decode_error"] += 1
                    report["fatal_issues"].append(
                        f"{split}: failed to decode {path}"
                    )
                if class_name == "fake":
                    relative = path.relative_to(class_root).parts
                    subtype = relative[0] if len(relative) > 1 else "unlabeled"
                    fake_subclass_counts[subtype] += 1
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                hashes[digest].append((split, class_name, str(path)))
            split_report[class_name] = {
                "images": len(files),
                "ignored_non_image_files": len(all_files) - len(files),
                "extensions": dict(extension_counts),
                "top_resolutions": dict(resolution_counts.most_common(10)),
            }
            if class_name == "fake":
                split_report[class_name]["subclasses"] = dict(
                    fake_subclass_counts
                )
                for expected in expected_fake_subclasses:
                    if fake_subclass_counts[expected] == 0:
                        report["fatal_issues"].append(
                            f"{split}: fake subclass {expected} contains no images"
                        )
        real_extensions = set(split_report["real"]["extensions"])
        fake_extensions = set(split_report["fake"]["extensions"])
        split_report["format_shortcut_warning"] = bool(
            real_extensions and fake_extensions and real_extensions.isdisjoint(fake_extensions)
        )
        report["splits"][split] = split_report
    for occurrences in hashes.values():
        split_names = {item[0] for item in occurrences}
        if len(split_names) > 1:
            report["cross_split_duplicates"].append(occurrences)
    report["has_data_leakage"] = bool(report["cross_split_duplicates"])
    # Content duplicates are reported as a warning but do not block training.
    report["passed"] = not report["fatal_issues"]
    return report
