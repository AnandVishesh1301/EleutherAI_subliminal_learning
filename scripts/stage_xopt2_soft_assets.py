"""CLI: stage the `xopt2` soft inputs (teacher copies onto the cold-1 leaf root, paths
YAML, coverage check against `xopt` soft's staging report; create-only, no GPU)."""

from __future__ import annotations

import argparse

from pipelines import xopt2_soft_staging_pipeline
from subliminal_learning.utils.logging_setup import configure_loguru


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    configure_loguru()
    xopt2_soft_staging_pipeline.run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
