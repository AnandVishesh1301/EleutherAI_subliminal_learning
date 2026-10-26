"""CLI: stage every `traitfree` input onto its part's root (create-only, sha-checked)."""

from __future__ import annotations

import argparse
from pathlib import Path

from pipelines import traitfree_staging_pipeline
from subliminal_learning.campaign import traitfree_plan as plan
from subliminal_learning.utils.logging_setup import configure_loguru


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--roster", type=Path, default=None)
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
        return
    roster = args.roster or {2: plan.ROSTER2_PATH, 3: plan.ROSTER3_PATH}.get(
        args.part, plan.ROSTER_PATH
    )
    traitfree_staging_pipeline.run(roster, dry_run=args.dry_run)
    if args.part == 2 and not args.dry_run:
        traitfree_staging_pipeline.write_paths2()


if __name__ == "__main__":
    main()
