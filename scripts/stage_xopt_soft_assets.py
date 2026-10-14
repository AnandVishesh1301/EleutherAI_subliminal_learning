"""CLI: stage every `xopt` soft input onto its cold-1 root and measure CE_base per
student base@step (create-only, sha-checked; soar, one GPU)."""

from __future__ import annotations

import argparse

from pipelines import xopt_soft_staging_pipeline
from subliminal_learning.utils.logging_setup import configure_loguru


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    configure_loguru()
    xopt_soft_staging_pipeline.run(dry_run=args.dry_run, device=args.device)


if __name__ == "__main__":
    main()
