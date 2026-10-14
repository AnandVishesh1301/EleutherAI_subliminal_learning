"""CLI: stage every `xopt` hard input onto its ssd-1 root and measure CE_base (soar, 1 GPU)."""

from __future__ import annotations

import argparse
from pathlib import Path

from pipelines import xopt_hard_staging_pipeline
from subliminal_learning.utils.logging_setup import configure_loguru


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    configure_loguru()
    xopt_hard_staging_pipeline.run(Path.cwd(), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
