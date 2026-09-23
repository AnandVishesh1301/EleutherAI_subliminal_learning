"""Build the ws160m result tables and claims from the partition trees.

``--gen 1`` unions the four Part I trees (tables under the `main` run set);
``--gen 2`` unions Part I and Part II (tables under the `matrix` run set);
``--gen 3`` unions Parts I-III (tables under the `rx1` run set) plus the rung-cross tables.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from loguru import logger

from pipelines.ws160m_results_pipeline import build_results, partition_coverage
from subliminal_learning.campaign import ws160m_plan as plan
from subliminal_learning.utils.logging_setup import configure_loguru


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paths-config", type=Path, default=Path(plan.PATHS_CONFIG))
    ap.add_argument(
        "--partial", action="store_true", help="Stamp claims.json as partial."
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--gen", type=int, choices=(1, 2, 3), default=1, help="Campaign generation."
    )
    args = ap.parse_args()
    configure_loguru()
    paths = yaml.safe_load(Path(args.paths_config).read_text())
    output_root = Path(paths["output_root"])
    coverage = partition_coverage(output_root, gen=args.gen)
    partial = args.partial or any(
        isinstance(c, dict) and (c.get("missing") or c.get("partial"))
        for c in coverage.values()
    )
    written = build_results(
        output_root,
        dry_run=args.dry_run,
        partial=partial,
        coverage=coverage,
        gen=args.gen,
    )
    logger.info("wrote {} result files under {}", len(written), written[-1].parent)


if __name__ == "__main__":
    main()
