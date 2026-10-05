"""Materialize a consistent 256x256 face-region dataset in place.

Only images larger than the target size are changed. Rectangular images are
cropped around the largest detected face (with a deterministic center-square
fallback); square images are resized without an additional face crop because
they are already stored as face crops in this dataset. An optional JPEG
normalization pass removes the class-specific PNG/JPEG storage shortcut.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageOps

from data.preprocessing import UnifiedFacePreprocessor


EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})


def _save_atomic(
    image: Image.Image, path: Path, normalize_jpeg: bool = False
) -> Path:
    target = path.with_suffix(".jpg") if normalize_jpeg else path
    if target != path and target.exists():
        raise FileExistsError(f"JPEG target already exists: {target}")
    temporary = target.with_name(f"{target.stem}.facecrop-tmp{target.suffix}")
    suffix = target.suffix.lower()
    options: dict[str, object] = {}
    if suffix in {".jpg", ".jpeg"}:
        options = {"format": "JPEG", "quality": 95, "subsampling": 0}
    elif suffix == ".png":
        options = {"format": "PNG", "optimize": True}
    elif suffix == ".webp":
        options = {"format": "WEBP", "quality": 95, "method": 6}
    elif suffix == ".bmp":
        options = {"format": "BMP"}
    image.save(temporary, **options)
    temporary.replace(target)
    if target != path:
        path.unlink()
    return target


def preprocess(
    root: Path,
    size: int,
    margin_ratio: float,
    normalize_jpeg: bool = False,
) -> dict[str, object]:
    detector = UnifiedFacePreprocessor(
        enabled=True,
        strategy="face_or_center",
        margin_ratio=margin_ratio,
        min_face_ratio=0.08,
        scale_factor=1.1,
        min_neighbors=5,
    )
    paths = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in EXTENSIONS
    )
    report: dict[str, object] = {
        "root": str(root),
        "target_size": [size, size],
        "normalize_jpeg": normalize_jpeg,
        "total_images": len(paths),
        "processed_images": 0,
        "face_crops": 0,
        "center_fallbacks": 0,
        "square_resizes": 0,
        "unchanged_images": 0,
        "format_conversions": 0,
        "failures": [],
        "records": [],
    }
    records: list[dict[str, object]] = report["records"]  # type: ignore[assignment]
    failures: list[dict[str, str]] = report["failures"]  # type: ignore[assignment]

    for path in paths:
        try:
            with Image.open(path) as source:
                rgb = ImageOps.exif_transpose(source).convert("RGB")
                width, height = rgb.size
                if width <= size and height <= size and not normalize_jpeg:
                    report["unchanged_images"] = int(report["unchanged_images"]) + 1
                    continue

                if width == height:
                    cropped = rgb
                    method = (
                        "format_normalize" if (width, height) == (size, size)
                        else "square_resize"
                    )
                    crop_box = (0, 0, width, height)
                    report["square_resizes"] = int(report["square_resizes"]) + 1
                else:
                    result = detector.process(rgb, cache_key=path)
                    cropped = result.image
                    crop_box = result.crop_box
                    if result.face_detected:
                        method = "largest_face"
                        report["face_crops"] = int(report["face_crops"]) + 1
                    else:
                        method = "center_fallback"
                        report["center_fallbacks"] = int(report["center_fallbacks"]) + 1

                resized = cropped.resize((size, size), Image.Resampling.LANCZOS)
                target = _save_atomic(resized, path, normalize_jpeg)
                if target != path:
                    report["format_conversions"] = (
                        int(report["format_conversions"]) + 1
                    )
                report["processed_images"] = int(report["processed_images"]) + 1
                records.append({
                    "path": str(path.relative_to(root)),
                    "output_path": str(target.relative_to(root)),
                    "original_size": [width, height],
                    "crop_box": list(crop_box),
                    "method": method,
                })
        except Exception as error:  # Continue so the report names every bad file.
            failures.append({"path": str(path), "error": str(error)})

    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("dataset"))
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--margin-ratio", type=float, default=0.35)
    parser.add_argument(
        "--normalize-jpeg",
        action="store_true",
        help="Re-encode every image as quality-95 JPEG and remove the old file.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("outputs/data_preprocessing/face_crop_report.json"),
    )
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    report = preprocess(
        root, args.size, args.margin_ratio, normalize_jpeg=args.normalize_jpeg
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        "Face preprocessing complete | "
        f"processed={report['processed_images']} "
        f"face={report['face_crops']} "
        f"center_fallback={report['center_fallbacks']} "
        f"square_resize={report['square_resizes']} "
        f"format_conversions={report['format_conversions']} "
        f"failures={len(report['failures'])} "
        f"report={args.report}"
    )
    if report["failures"]:
        raise RuntimeError("Some images failed preprocessing; inspect the report")


if __name__ == "__main__":
    main()
