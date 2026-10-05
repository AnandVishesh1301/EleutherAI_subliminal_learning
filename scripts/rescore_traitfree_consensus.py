"""CLI: `consensus` re-score of the sdfpos2 placebo control (`traitfree2` jobq job)."""

from __future__ import annotations

import argparse
from pathlib import Path

from pipelines.traitfree_rescore_pipeline import RescoreConfig, run
from subliminal_learning.campaign import traitfree_plan as plan
from subliminal_learning.utils.logging_setup import configure_loguru


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    configure_loguru()
    doc = plan.paths_config_doc()
    run(
        RescoreConfig(
            roster_path=plan.ROSTER_PATH,
            out_dir=plan.RESCORE_DIR,
            paths_config=plan.PATHS_CONFIG_PATH,
            hf_cache_root=Path(doc["hf_cache_root"]),
            hub_org=plan.HUB_ORG,
            device=args.device,
        )
    )


if __name__ == "__main__":
    main()
