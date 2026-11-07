"""CLI: stage the `rebhard` inputs (create-only; login node)."""

from __future__ import annotations

from pipelines import rebhard_staging_pipeline as staging
from subliminal_learning.utils.logging_setup import configure_loguru


def main() -> None:
    configure_loguru()
    staging.run()


if __name__ == "__main__":
    main()
