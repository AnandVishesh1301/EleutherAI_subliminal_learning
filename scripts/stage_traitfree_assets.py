"""CLI: stage every `traitfree` input onto its ssd-1 root (create-only, sha-checked)."""

from __future__ import annotations

import argparse
from pathlib import Path

from pipelines import traitfree_staging_pipeline
from subliminal_learning.campaign import traitfree_plan as plan
from subliminal_learning.utils.logging_setup import configure_loguru


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roster", type=Path, default=plan.ROSTER_PATH)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--docgen-only",
        action="store_true",
        help="Stage only the Stage-1 inputs (no roster needed; writes no report).",
    )
    args = parser.parse_args()
    configure_loguru()
    if args.docgen_only:
        traitfree_staging_pipeline.run_docgen_only()
    else:
        traitfree_staging_pipeline.run(args.roster, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
