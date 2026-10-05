from pathlib import Path

import pytest
from PIL import Image

from data import (
    InferenceImageDataset,
    KnownBinaryDataset,
    UnifiedFacePreprocessor,
    hierarchical_sample_weights,
)
from evaluation import audit_known_dataset


def _image(path: Path, color=(100, 120, 140)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (48, 48), color).save(path)


def test_known_dataset_has_only_real_and_fake(tmp_path: Path):
    _image(tmp_path / "real" / "r.png")
    _image(tmp_path / "fake" / "face_swap" / "f.png")
    dataset = KnownBinaryDataset(
        tmp_path,
        image_size=32,
        fake_subclass_names=["face_swap", "attribute_edit"],
    )
    assert dataset.class_counts() == {"real": 1, "fake": 1}
    fake = dataset[1]
    assert fake["label"] == 1
    assert fake["fake_subclass"] == 0
    assert fake["fake_source"] == "face_swap"
    assert fake["h1"].shape == (9, 16, 16)
    assert "unknown" not in dataset.class_counts()


def test_face_swap_and_reenactment_use_swap_and_edit_labels(tmp_path: Path):
    _image(tmp_path / "real" / "r.png")
    _image(tmp_path / "fake" / "face_swap" / "swap.png")
    _image(tmp_path / "fake" / "expression_reenactment" / "reenact.png")
    dataset = KnownBinaryDataset(
        tmp_path,
        image_size=32,
        fake_subclass_names=["face_swap", "attribute_edit"],
    )
    fake_records = [record for record in dataset.records if record.label == 1]
    assert {record.fake_subclass for record in fake_records} == {0, 1}
    assert {record.fake_source for record in fake_records} == {
        "face_swap", "expression_reenactment"
    }


def test_center_square_preprocessing_is_deterministic():
    image = Image.new("RGB", (80, 40), (100, 120, 140))
    preprocessor = UnifiedFacePreprocessor(
        enabled=True, strategy="center_square"
    )
    result = preprocessor.process(image)
    assert result.image.size == (40, 40)
    assert result.crop_box == (20, 0, 60, 40)
    assert not result.face_detected


def test_hierarchical_weights_target_two_to_one_to_one(tmp_path: Path):
    for index in range(2):
        _image(tmp_path / "real" / f"r{index}.png")
    _image(tmp_path / "fake" / "face_swap" / "s.png")
    for index in range(3):
        _image(tmp_path / "fake" / "attribute_edit" / f"e{index}.png")
    dataset = KnownBinaryDataset(
        tmp_path,
        image_size=32,
        fake_subclass_names=["face_swap", "attribute_edit"],
    )
    weights = hierarchical_sample_weights(dataset, real_fraction=0.5)
    real = sum(
        weights[index].item()
        for index, record in enumerate(dataset.records)
        if record.label == 0
    )
    swap = sum(
        weights[index].item()
        for index, record in enumerate(dataset.records)
        if record.fake_subclass == 0
    )
    edit = sum(
        weights[index].item()
        for index, record in enumerate(dataset.records)
        if record.fake_subclass == 1
    )
    assert real == pytest.approx(0.5)
    assert swap == pytest.approx(0.25)
    assert edit == pytest.approx(0.25)


def test_unknown_folder_does_not_become_known_class(tmp_path: Path):
    _image(tmp_path / "real" / "r.png")
    _image(tmp_path / "fake" / "f.png")
    _image(tmp_path / "unknown" / "u.png")
    dataset = KnownBinaryDataset(tmp_path, image_size=32)
    assert len(dataset) == 2


def test_unlabeled_dataset_for_inference(tmp_path: Path):
    _image(tmp_path / "anything" / "sample.png")
    sample = InferenceImageDataset(tmp_path, image_size=32)[0]
    assert sample["label"] == -1


def test_known_dataset_requires_both_classes(tmp_path: Path):
    _image(tmp_path / "real" / "r.png")
    with pytest.raises(RuntimeError, match="missing class folder: fake"):
        KnownBinaryDataset(tmp_path, image_size=32)


def test_audit_includes_calibration_and_detects_content_duplicates(tmp_path: Path):
    id_root = tmp_path / "id"
    calibration_root = tmp_path / "calibration" / "id"
    partitions = {
        "train": id_root / "train",
        "val": id_root / "val",
        "test": id_root / "test",
        "calibration": calibration_root,
    }
    for index, split_root in enumerate(partitions.values()):
        _image(split_root / "real" / f"r{index}.png", color=(index, 10, 20))
        _image(
            split_root / "fake" / "face_swap" / f"s{index}.png",
            color=(30, index, 40),
        )
        _image(
            split_root / "fake" / "attribute_edit" / f"e{index}.png",
            color=(50, 60, index),
        )
    duplicate = (id_root / "train" / "real" / "r0.png").read_bytes()
    (calibration_root / "real" / "duplicate.png").write_bytes(duplicate)

    report = audit_known_dataset(
        id_root,
        calibration_root,
        ["face_swap", "attribute_edit"],
    )
    assert "calibration" in report["splits"]
    assert report["has_data_leakage"]
    assert report["passed"]
