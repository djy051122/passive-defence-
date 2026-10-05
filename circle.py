"""Calibrate trusted real/fake spheres; their exterior is unknown."""

from pipelines.cli import run_calibrate_trusted_spheres_cli


def main() -> None:
    run_calibrate_trusted_spheres_cli()


if __name__ == "__main__":
    main()
