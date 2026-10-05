"""CLI: build the frozen `traitfree` roster (reads source campaigns, writes one file).

    uv run python -m scripts.build_traitfree_roster
    uv run python -m scripts.build_traitfree_roster --out /tmp/roster.json --scratch
    uv run python -m scripts.build_traitfree_roster --part 2

Refuses an existing ``--out``; the default is `traitfree_plan.ROSTER_PATH` (Part II:
`ROSTER2_PATH`).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pipelines.traitfree_roster_pipeline import run, run_part2
from subliminal_learning.campaign import traitfree_plan as plan
from subliminal_learning.config import load_config
from subliminal_learning.utils.logging_setup import configure_loguru


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", type=int, choices=(1, 2), default=1)
    parser.add_argument("--out", type=Path, default=None)
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
    if args.part == 2:
        run_part2(cfg.paths.output_root, args.out or plan.ROSTER2_PATH, scratch=args.scratch)
    else:
        run(cfg.paths.output_root, args.out or plan.ROSTER_PATH, scratch=args.scratch)


if __name__ == "__main__":
    main()
