"""CLI: build the frozen `traitfree` roster (reads source campaigns, writes one file).

    uv run python -m scripts.build_traitfree_roster
    uv run python -m scripts.build_traitfree_roster --out /tmp/roster.json --scratch

Refuses an existing ``--out``; the default is `traitfree_plan.ROSTER_PATH`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pipelines.traitfree_roster_pipeline import run
from subliminal_learning.campaign import traitfree_plan as plan
from subliminal_learning.config import load_config
from subliminal_learning.utils.logging_setup import configure_loguru


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=plan.ROSTER_PATH)
    parser.add_argument(
        "--scratch",
        action="store_true",
        help="Trial build to a path other than the frozen roster path.",
    )
    parser.add_argument(
        "--paths", type=Path, default=Path("configs/paths/paths_local.yaml")
    )
    parser.add_argument("--model", type=Path, default=Path("configs/model/pythia.yaml"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Build and publish the roster."""
    args = _parse_args(argv)
    configure_loguru()
    cfg = load_config(args.paths, args.model)
    run(cfg.paths.output_root, args.out, scratch=args.scratch)


if __name__ == "__main__":
    main()
