"""Standalone cross-dataset and cross-generator generalization test."""

from pipelines.cli import run_test_generalization_cli


def main() -> None:
    run_test_generalization_cli()


if __name__ == "__main__":
    main()
