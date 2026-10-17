"""CLI: verify `xopt2` hard's reused `xopt` inputs and write its paths YAML (soar, no GPU)."""

from __future__ import annotations

import argparse

from pipelines import xopt2_hard_staging_pipeline
from subliminal_learning.utils.logging_setup import configure_loguru


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    configure_loguru()
    xopt2_hard_staging_pipeline.run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
