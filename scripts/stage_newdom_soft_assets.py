"""CLI: stage the `newdom` soft inputs (paths YAML, jobq policies, domain prompt pools,
placebo corpora + slots + review samples) and write a staging report. Create-only, no
GPU. The placebo step needs ``PYTHONPATH=tools/sdf_docgen/src``."""

from __future__ import annotations

import argparse

from pipelines import newdom_soft_staging_pipeline
from subliminal_learning.utils.logging_setup import configure_loguru


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--steps",
        default=",".join(newdom_soft_staging_pipeline.STEPS),
        help="Comma list of steps to run.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    configure_loguru()
    steps = [s for s in args.steps.split(",") if s]
    newdom_soft_staging_pipeline.run(steps, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
