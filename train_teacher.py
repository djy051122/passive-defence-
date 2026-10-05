"""Audit all known data, then train the complete-band teacher."""

from pipelines.cli import run_train_teacher_cli


def main() -> None:
    run_train_teacher_cli()


if __name__ == "__main__":
    main()
