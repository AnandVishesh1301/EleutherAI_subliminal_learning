"""CLI: stage every `hlopt` input onto its ssd-1 root (create-only, sha-checked).

`--part 2` / `--part 3` / `--part 4` verify those inputs for `hlopt2` / `hlopt3` /
`hlopt4` and write the part's paths YAML.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pipelines import hlopt_staging_pipeline
from subliminal_learning.utils.logging_setup import configure_loguru


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--part", type=int, choices=(1, 2, 3, 4), default=1)
    args = parser.parse_args()
    configure_loguru()
    if args.part == 2:
        hlopt_staging_pipeline.run_part2(dry_run=args.dry_run)
    elif args.part == 3:
        hlopt_staging_pipeline.run_part3(dry_run=args.dry_run)
    elif args.part == 4:
        hlopt_staging_pipeline.run_part4(dry_run=args.dry_run)
    else:
        hlopt_staging_pipeline.run(Path.cwd(), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
