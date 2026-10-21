"""CLI: stage the `crossval2` inputs (bases, prompt pool, pp1, paths YAML, goncalo GPU
policy), check every corpus and write a staging report. Create-only, no GPU."""

from __future__ import annotations

import argparse

from pipelines import crossval2_staging_pipeline
from subliminal_learning.utils.logging_setup import configure_loguru


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    configure_loguru()
    crossval2_staging_pipeline.run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
