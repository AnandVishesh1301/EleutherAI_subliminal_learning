"""CLI: build the traitfree results tables into a fresh ``build-<UTC>`` dir.

    uv run python -m scripts.build_traitfree_results
    uv run python -m scripts.build_traitfree_results --roster <roster.json> \
        --out-root <dir> --family conditions --family topk
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pipelines.traitfree_results_pipeline import build_results, default_out_root
from subliminal_learning.campaign import traitfree_plan as plan
from subliminal_learning.utils.logging_setup import configure_loguru


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roster", type=Path, default=plan.ROSTER_PATH)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help=f"Parent of the build dir (default {default_out_root()}).",
    )
    parser.add_argument(
        "--family",
        action="append",
        choices=(*plan.FAMILIES, *plan.PART2_FAMILIES, *plan.PART3_FAMILIES),
        default=None,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Build the results of the families asked for (default: all)."""
    args = _parse_args(argv)
    configure_loguru()
    build_results(roster_path=args.roster, out_root=args.out_root, families=args.family)


if __name__ == "__main__":
    main()
