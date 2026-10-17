"""CLI: stage `newdom` hard's control pool, pp1 text and train prompts (soar, no GPU)."""

from __future__ import annotations

import argparse
from pathlib import Path

from pipelines import newdom_hard_staging_pipeline
from subliminal_learning.utils.code_pin_io import main_checkout_root
from subliminal_learning.utils.logging_setup import configure_loguru


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    configure_loguru()
    newdom_hard_staging_pipeline.run(
        main_checkout_root(Path.cwd()), dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
