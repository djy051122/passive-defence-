"""Standalone real/fake/unknown image detection script."""

from pipelines.cli import run_detect_images_cli


def main() -> None:
    run_detect_images_cli()


if __name__ == "__main__":
    main()
