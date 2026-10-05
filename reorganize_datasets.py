"""Build a leakage-resistant real/face-swap/attribute-edit dataset.

The source data are never modified. FaceForensics++ provides paired real and
face-swap samples. DFFD provides CelebA real samples and StarGAN attribute
edits. Splits are made by reciprocal FF++ video pairs, StarGAN source groups,
and official CelebA identities before any frame/image is selected.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from data.preprocessing import UnifiedFacePreprocessor


SPLITS = ("train", "val", "test", "calibration")
SPLIT_RATIOS = (0.70, 0.10, 0.10, 0.10)
FACE_SWAP_METHODS = ("Deepfakes", "FaceSwap", "FaceShifter")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _stable_int(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _allocate_counts(total: int) -> dict[str, int]:
    counts = [int(total * ratio) for ratio in SPLIT_RATIOS]
    for index in range(total - sum(counts)):
        counts[index % len(counts)] += 1
    return dict(zip(SPLITS, counts))


def _split_groups(groups: Iterable[Any], seed: int) -> dict[Any, str]:
    shuffled = list(groups)
    random.Random(seed).shuffle(shuffled)
    counts = _allocate_counts(len(shuffled))
    result: dict[Any, str] = {}
    cursor = 0
    for split in SPLITS:
        for group in shuffled[cursor:cursor + counts[split]]:
            result[group] = split
        cursor += counts[split]
    return result


def _choose(paths: list[Path], *key: object) -> Path:
    if not paths:
        raise RuntimeError(f"Cannot sample an empty path list: {key}")
    return paths[_stable_int(*key) % len(paths)]


def _frame_number(path: Path) -> int:
    stem = path.stem
    marker = "__frame_"
    return int(stem.split(marker, 1)[1]) if marker in stem else 0


def _index_ffpp(ffpp_root: Path) -> tuple[
    dict[int, list[Path]],
    dict[str, dict[tuple[int, int], list[Path]]],
    list[tuple[int, int]],
]:
    originals: dict[int, list[Path]] = defaultdict(list)
    for path in (ffpp_root / "original").glob("*.jpg"):
        prefix = path.stem.split("__frame_", 1)[0]
        originals[int(prefix)].append(path)
    for paths in originals.values():
        paths.sort(key=_frame_number)

    method_indexes: dict[str, dict[tuple[int, int], list[Path]]] = {}
    pair_sets: list[set[tuple[int, int]]] = []
    for method in FACE_SWAP_METHODS:
        index: dict[tuple[int, int], list[Path]] = defaultdict(list)
        for path in (ffpp_root / method).glob("*.jpg"):
            prefix = path.stem.split("__frame_", 1)[0]
            target_text, source_text = prefix.split("_", 1)
            index[(int(target_text), int(source_text))].append(path)
        for paths in index.values():
            paths.sort(key=_frame_number)
        method_indexes[method] = index
        pair_sets.append(set(index))

    if len(originals) != 1000:
        raise RuntimeError(f"Expected 1000 FF++ originals, found {len(originals)}")
    if any(pairs != pair_sets[0] for pairs in pair_sets[1:]):
        raise RuntimeError("FF++ face-swap methods do not share the same pair map")

    directed_pairs = pair_sets[0]
    reciprocal_groups: set[tuple[int, int]] = set()
    for target, source in directed_pairs:
        if (source, target) not in directed_pairs:
            raise RuntimeError(f"Missing reciprocal FF++ pair: {source}_{target}")
        reciprocal_groups.add(tuple(sorted((target, source))))
    if len(reciprocal_groups) != 500:
        raise RuntimeError(
            f"Expected 500 reciprocal FF++ groups, found {len(reciprocal_groups)}"
        )
    return originals, method_indexes, sorted(reciprocal_groups)


def _load_celeba_identities(path: Path) -> dict[str, int]:
    identities: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) == 2:
            identities[fields[0]] = int(fields[1])
    if len(identities) != 202599:
        raise RuntimeError(
            f"Expected 202599 CelebA identity rows, found {len(identities)}"
        )
    return identities


def _index_dffd(
    dffd_root: Path, identity_file: Path
) -> tuple[dict[int, list[Path]], dict[int, list[Path]]]:
    stargan: dict[int, list[Path]] = defaultdict(list)
    for path in (dffd_root / "stargan").glob("*.jpg"):
        fields = path.stem.split("_")[-1].split("-", 1)
        stargan[int(fields[0])].append(path)
    for paths in stargan.values():
        paths.sort(key=lambda item: item.name)

    identity_map = _load_celeba_identities(identity_file)
    celeba: dict[int, list[Path]] = defaultdict(list)
    for path in (dffd_root / "celeba").glob("*.jpg"):
        identity = identity_map.get(path.name)
        if identity is None:
            raise RuntimeError(f"Missing CelebA identity annotation: {path.name}")
        celeba[identity].append(path)
    for paths in celeba.values():
        paths.sort(key=lambda item: item.name)
    if len(stargan) < 2 or len(celeba) < len(stargan):
        raise RuntimeError(
            f"Insufficient DFFD groups: stargan={len(stargan)}, celeba={len(celeba)}"
        )
    return stargan, celeba


def _record(
    source: Path,
    split: str,
    label: str,
    subtype: str,
    dataset: str,
    method: str,
    group_key: str,
    identity_key: str,
    filename: str,
) -> dict[str, Any]:
    if split == "calibration":
        base = Path("calibration") / "id"
    else:
        base = Path("id") / split
    if label == "real":
        relative = base / "real" / dataset / filename
    else:
        relative = base / "fake" / subtype / dataset / filename
    return {
        "source_path": str(source),
        "relative_path": relative.as_posix(),
        "split": split,
        "label": label,
        "fake_subtype": subtype,
        "source_dataset": dataset,
        "method": method,
        "group_key": group_key,
        "identity_key": identity_key,
    }


def build_plan(source_root: Path, output_root: Path, seed: int) -> list[dict[str, Any]]:
    ffpp_root = source_root / "FaceForensics++_C23_frames"
    dffd_root = source_root / "dffd"
    identity_file = output_root / "metadata" / "identity_CelebA.txt"
    originals, face_indexes, ff_groups = _index_ffpp(ffpp_root)
    stargan, celeba = _index_dffd(dffd_root, identity_file)

    records: list[dict[str, Any]] = []
    ff_split = _split_groups(ff_groups, seed)
    for group in ff_groups:
        split = ff_split[group]
        a, b = group
        group_key = f"ffpp_pair_{a:03d}_{b:03d}"
        for target, source in ((a, b), (b, a)):
            real_path = _choose(originals[target], seed, "ff_real", target)
            real_frame = _frame_number(real_path)
            records.append(_record(
                real_path, split, "real", "", "ffpp", "original",
                group_key, group_key,
                f"ffpp_{target:03d}_{real_frame:06d}_source.jpg",
            ))

            method = FACE_SWAP_METHODS[
                _stable_int(seed, "ff_method", target) % len(FACE_SWAP_METHODS)
            ]
            fake_path = _choose(
                face_indexes[method][(target, source)],
                seed, "ff_fake", method, target, source,
            )
            fake_frame = _frame_number(fake_path)
            records.append(_record(
                fake_path, split, "fake", "face_swap", "ffpp", method,
                group_key, group_key,
                f"ffpp_{target:03d}_{source:03d}_{method.lower()}_"
                f"{fake_frame:06d}_swapped.jpg",
            ))

    star_groups = sorted(stargan)
    star_split = _split_groups(star_groups, seed + 1)
    star_counts = Counter(star_split.values())

    identity_ids = sorted(celeba)
    random.Random(seed + 2).shuffle(identity_ids)
    needed = len(star_groups)
    chosen_identities = identity_ids[:needed]
    cursor = 0
    celeba_by_split: dict[str, list[int]] = {}
    for split in SPLITS:
        count = star_counts[split]
        celeba_by_split[split] = chosen_identities[cursor:cursor + count]
        cursor += count

    for split in SPLITS:
        split_star_groups = sorted(
            group for group, assigned in star_split.items() if assigned == split
        )
        split_real_ids = celeba_by_split[split]
        if len(split_star_groups) != len(split_real_ids):
            raise RuntimeError(f"DFFD class imbalance while planning {split}")
        for position, (star_group, identity) in enumerate(
            zip(split_star_groups, split_real_ids)
        ):
            real_path = _choose(
                celeba[identity], seed, "dffd_real", split, identity
            )
            records.append(_record(
                real_path, split, "real", "", "dffd_celeba", "original",
                f"celeba_identity_{identity}", f"celeba_identity_{identity}",
                f"dffd_{real_path.stem}_source.jpg",
            ))

            fake_path = _choose(
                stargan[star_group], seed, "stargan", split, star_group
            )
            records.append(_record(
                fake_path, split, "fake", "attribute_edit", "dffd_stargan",
                "StarGAN", f"stargan_group_{star_group}",
                f"stargan_group_{star_group}", fake_path.name,
            ))

    relative_paths = [record["relative_path"] for record in records]
    if len(relative_paths) != len(set(relative_paths)):
        raise RuntimeError("Planned output filenames are not unique")
    return records


def _plan_diagnostics(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        selected = [record for record in records if record["split"] == split]
        counts[split] = {
            "total": len(selected),
            "real": sum(record["label"] == "real" for record in selected),
            "fake": sum(record["label"] == "fake" for record in selected),
            "face_swap": sum(
                record["fake_subtype"] == "face_swap" for record in selected
            ),
            "attribute_edit": sum(
                record["fake_subtype"] == "attribute_edit" for record in selected
            ),
        }
    group_splits: dict[str, set[str]] = defaultdict(set)
    identity_splits: dict[str, set[str]] = defaultdict(set)
    for record in records:
        group_splits[record["group_key"]].add(record["split"])
        identity_splits[record["identity_key"]].add(record["split"])
    group_leaks = {
        key: sorted(value) for key, value in group_splits.items() if len(value) > 1
    }
    identity_leaks = {
        key: sorted(value)
        for key, value in identity_splits.items() if len(value) > 1
    }
    return {
        "counts": counts,
        "group_leaks": group_leaks,
        "identity_leaks": identity_leaks,
    }


def _write_jpeg(image: Image.Image, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.stem}.building{target.suffix}")
    image.save(temporary, format="JPEG", quality=95, subsampling=0)
    temporary.replace(target)


def materialize(
    records: list[dict[str, Any]],
    output_root: Path,
    image_size: int,
    resume: bool = False,
) -> dict[str, Any]:
    generated_roots = [output_root / "id", output_root / "calibration", output_root / "inference"]
    if not resume and any(path.exists() for path in generated_roots):
        existing = [str(path) for path in generated_roots if path.exists()]
        raise FileExistsError(
            "Refusing to overwrite an existing organized dataset: " + ", ".join(existing)
        )

    preprocessor = UnifiedFacePreprocessor(
        enabled=True,
        strategy="face_or_center",
        margin_ratio=0.35,
        min_face_ratio=0.08,
        scale_factor=1.1,
        min_neighbors=5,
    )
    method_counts: Counter[str] = Counter()
    hashes: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for index, record in enumerate(records, start=1):
        source = Path(record["source_path"])
        target = output_root / record["relative_path"]
        if resume and target.is_file():
            with Image.open(target) as existing:
                if existing.size != (image_size, image_size):
                    raise RuntimeError(
                        f"Cannot resume with invalid existing image: {target}"
                    )
            with Image.open(source) as source_image:
                record["original_size"] = [source_image.width, source_image.height]
            record["preprocessing"] = "completed_before_resume"
            record["crop_box"] = []
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            record["sha256"] = digest
            hashes[digest].append((record["split"], record["relative_path"]))
            method_counts["completed_before_resume"] += 1
            continue
        with Image.open(source) as image:
            cropped = preprocessor.process(image, cache_key=source)
            resized = cropped.image.resize(
                (image_size, image_size), Image.Resampling.LANCZOS
            )
            _write_jpeg(resized, target)
        method = "largest_face" if cropped.face_detected else "center_fallback"
        method_counts[method] += 1
        record["preprocessing"] = method
        record["original_size"] = [image.width, image.height]
        record["crop_box"] = list(cropped.crop_box)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        record["sha256"] = digest
        hashes[digest].append((record["split"], record["relative_path"]))
        if index % 250 == 0:
            print(f"processed {index}/{len(records)}", flush=True)

    inference_count = 0
    for record in records:
        if record["split"] != "test":
            continue
        source = output_root / record["relative_path"]
        category = (
            "real" if record["label"] == "real" else record["fake_subtype"]
        )
        target = output_root / "inference" / category / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            inference_count += 1
            continue
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)
        inference_count += 1

    cross_split_duplicates = []
    for digest, occurrences in hashes.items():
        splits = {split for split, _ in occurrences}
        if len(splits) > 1:
            cross_split_duplicates.append({
                "sha256": digest,
                "occurrences": [
                    {"split": split, "path": path} for split, path in occurrences
                ],
            })
    return {
        "preprocessing_counts": dict(method_counts),
        "inference_images": inference_count,
        "cross_split_content_duplicates": cross_split_duplicates,
    }


def _write_manifest(records: list[dict[str, Any]], output_root: Path) -> None:
    metadata = output_root / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    (metadata / "manifest.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fieldnames = [
        "relative_path", "source_path", "split", "label", "fake_subtype",
        "source_dataset", "method", "group_key", "identity_key",
        "preprocessing", "original_size", "crop_box", "sha256",
    ]
    with (metadata / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({name: record.get(name, "") for name in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path("/mnt/d/dataset"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/mnt/d/dataset/wavelet_reorganized_v1"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an interrupted --apply run after validating existing images.",
    )
    args = parser.parse_args()

    source_root = args.source_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    records = build_plan(source_root, output_root, args.seed)
    diagnostics = _plan_diagnostics(records)
    print(json.dumps(diagnostics["counts"], ensure_ascii=False, indent=2))
    if diagnostics["group_leaks"] or diagnostics["identity_leaks"]:
        raise RuntimeError(f"Split leakage in plan: {diagnostics}")
    if not args.apply:
        print("Dry run complete; pass --apply to materialize the dataset.")
        return

    materialized = materialize(
        records, output_root, args.image_size, resume=args.resume
    )
    diagnostics.update(materialized)
    diagnostics.update({
        "source_root": str(source_root),
        "output_root": str(output_root),
        "seed": args.seed,
        "image_size": args.image_size,
        "jpeg_quality": 95,
        "records": len(records),
    })
    if materialized["cross_split_content_duplicates"]:
        raise RuntimeError("Cross-split content duplicates found after materialization")
    _write_manifest(records, output_root)
    report_path = output_root / "metadata" / "reorganization_report.json"
    report_path.write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Dataset complete: {report_path}")


if __name__ == "__main__":
    main()
