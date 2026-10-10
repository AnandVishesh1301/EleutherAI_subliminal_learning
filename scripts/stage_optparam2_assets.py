"""CLI: stage every `optparam2` input onto its ssd-1 root, or with `--part 3` every
`optparam3` input onto its ssd-2 root (create-only, sha-checked)."""

from __future__ import annotations

import argparse

from pipelines import optparam2_staging_pipeline
from subliminal_learning.utils.logging_setup import configure_loguru


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--part", type=int, choices=(1, 3), default=1)
    args = parser.parse_args()
    configure_loguru()
    if args.part == 3:
        optparam2_staging_pipeline.run3(dry_run=args.dry_run)
    else:
        optparam2_staging_pipeline.run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
