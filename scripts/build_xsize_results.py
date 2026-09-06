"""CLI: (re)build the cross-size campaign's result tables + claims.

The campaign driver calls the same `build_results` at P5; this exists to
refresh the CSVs mid-campaign (after a partial phase) or to rebuild them
after a metric fix without re-running `advance`.

    uv run python -m scripts.build_xsize_results
    uv run python -m scripts.build_xsize_results --dry-run
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from dotenv import load_dotenv

from pipelines.xsize_results_pipeline import build_results
from subliminal_learning.config import load_config
from subliminal_learning.utils.logging_setup import configure_loguru


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paths",
        type=Path,
        default=Path("configs/paths/paths_local.yaml"),
        help="Paths YAML for load_config().",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("configs/model/pythia.yaml"),
        help="Model YAML for load_config().",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Override paths.output_root from --paths.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute every frame; write nothing.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Load config and write xsize results/*.csv + claims.json."""
    args = _parse_args(argv)
    load_dotenv()
    configure_loguru()
    cfg = load_config(args.paths, args.model)
    if args.output_root is not None:
        cfg = replace(cfg, paths=replace(cfg.paths, output_root=args.output_root))
    build_results(cfg, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
