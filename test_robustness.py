"""Standalone degradation-robustness testing script."""

from pipelines.cli import run_test_robustness_cli


def main() -> None:
    run_test_robustness_cli()


if __name__ == "__main__":
    main()
