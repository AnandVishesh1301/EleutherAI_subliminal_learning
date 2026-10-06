"""CLI: stage every `optparam2` input onto its ssd-1 root (create-only, sha-checked)."""

from __future__ import annotations

import argparse

from pipelines import optparam2_staging_pipeline
from subliminal_learning.utils.logging_setup import configure_loguru


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    configure_loguru()
    optparam2_staging_pipeline.run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
