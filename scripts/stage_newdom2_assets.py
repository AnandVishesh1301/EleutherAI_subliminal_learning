"""CLI: stage the `newdom2` inputs (paths YAML, jobq policy, B corpora + manifests +
review samples, the strict_sub screen preview) and write a staging report.
Create-only, no GPU."""

from __future__ import annotations

import argparse

from pipelines import newdom2_staging_pipeline
from subliminal_learning.utils.logging_setup import configure_loguru


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--steps",
        default=",".join(newdom2_staging_pipeline.STEPS),
        help="Comma list of steps to run.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    configure_loguru()
    steps = [s for s in args.steps.split(",") if s]
    newdom2_staging_pipeline.run(steps, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
