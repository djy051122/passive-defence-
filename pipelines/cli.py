"""Command-line adapters for one-step executable scripts."""

from __future__ import annotations

import argparse
from pathlib import Path

from .common import DEFAULT_STAGE1_CONFIG, DEFAULT_STAGE2_CONFIG
from .stage1 import test_teacher, train_teacher
from .stage2 import (
    calibrate_trusted_spheres,
    detect_images,
    test_generalization,
    test_robustness,
    test_student,
    train_student,
)


def _base_parser(description: str, default_config: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, default=default_config, help="YAML 配置文件")
    parser.add_argument("--device", type=str, help="auto、cpu、cuda 或 cuda:0")
    return parser


def run_train_teacher_cli(argv=None) -> None:
    parser = _base_parser("检查数据并训练完整频带教师", DEFAULT_STAGE1_CONFIG)
    parser.add_argument("--checkpoint", type=Path, help="最佳教师模型保存位置")
    args = parser.parse_args(argv)
    path = train_teacher(args.config, args.checkpoint, args.device)
    print(f"数据检查通过，教师训练完成，最佳模型：{path}")


def run_test_teacher_cli(argv=None) -> None:
    parser = _base_parser("测试完整频带教师", DEFAULT_STAGE1_CONFIG)
    parser.add_argument("--checkpoint", type=Path, help="待测试教师模型")
    args = parser.parse_args(argv)
    metrics = test_teacher(args.config, args.checkpoint, args.device)
    print(
        f"教师测试完成：Accuracy={metrics['accuracy']:.4f}，"
        f"Macro-F1={metrics['macro_f1']:.4f}"
    )


def run_train_student_cli(argv=None) -> None:
    parser = _base_parser("训练 RGB—高频学生", DEFAULT_STAGE2_CONFIG)
    parser.add_argument("--checkpoint", type=Path, help="最佳学生模型保存位置")
    parser.add_argument("--teacher-checkpoint", type=Path, help="已训练教师模型")
    args = parser.parse_args(argv)
    path = train_student(
        args.config, args.checkpoint, args.teacher_checkpoint, args.device
    )
    print(f"学生训练完成，最佳模型：{path}")


def run_test_student_cli(argv=None) -> None:
    parser = _base_parser("在已知真/伪测试集上测试学生", DEFAULT_STAGE2_CONFIG)
    parser.add_argument("--checkpoint", type=Path, help="待测试学生模型")
    args = parser.parse_args(argv)
    metrics = test_student(args.config, args.checkpoint, args.device)
    print(
        f"学生已知集测试完成：Accuracy={metrics['accuracy']:.4f}，"
        f"Macro-F1={metrics['macro_f1']:.4f}"
    )


def run_calibrate_trusted_spheres_cli(argv=None) -> None:
    parser = _base_parser("仅用真/假校准集拟合两个马氏可信椭球", DEFAULT_STAGE2_CONFIG)
    parser.add_argument("--checkpoint", type=Path, help="已训练学生模型")
    args = parser.parse_args(argv)
    artifacts = calibrate_trusted_spheres(args.config, args.checkpoint, args.device)
    print(f"真/假马氏可信椭球校准完成，诊断={artifacts['calibration_diagnostics']}")


def run_detect_images_cli(argv=None) -> None:
    parser = _base_parser(
        "输出每张图片的真、换脸、属性编辑或未知判断",
        DEFAULT_STAGE2_CONFIG,
    )
    parser.add_argument("--checkpoint", type=Path, help="已训练学生模型")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("dataset/inference"),
        help="待检测图片目录；默认使用 dataset/inference",
    )
    args = parser.parse_args(argv)
    detect_images(args.config, args.checkpoint, args.input_dir, args.device)


def run_test_robustness_cli(argv=None) -> None:
    parser = _base_parser("测试 JPEG、模糊、噪声和缩放鲁棒性", DEFAULT_STAGE2_CONFIG)
    parser.add_argument("--checkpoint", type=Path, help="已训练学生模型")
    args = parser.parse_args(argv)
    results = test_robustness(args.config, args.checkpoint, args.device)
    print(f"鲁棒性测试完成，共 {len(results)} 种退化条件")


def run_test_generalization_cli(argv=None) -> None:
    parser = _base_parser("测试跨数据集/跨生成器泛化性", DEFAULT_STAGE2_CONFIG)
    parser.add_argument("--checkpoint", type=Path, help="已训练学生模型")
    parser.add_argument(
        "--data-root", type=Path,
        help="含 <domain>/real 和 <domain>/fake 的目录；默认 dataset/generalization",
    )
    args = parser.parse_args(argv)
    results = test_generalization(
        args.config, args.checkpoint, args.data_root, args.device
    )
    print(f"泛化性测试完成，共 {len(results)} 个域")
