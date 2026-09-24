"""Teacher top-K token census over the OPD-limits students' rollouts.

uv run python -m scripts.run_topk_census plan
CUDA_VISIBLE_DEVICES=0 uv run python -m scripts.run_topk_census replay --shard 0/16
uv run python -m scripts.run_topk_census aggregate
"""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

from loguru import logger

from pipelines import topk_census_pipeline as census
from subliminal_learning.campaign import opdlim_plan as plan
from subliminal_learning.config import load_config
from subliminal_learning.models.loading_io import load_tokenizer
from subliminal_learning.teachers.model_specs import parse_base_models
from subliminal_learning.utils.logging_setup import configure_loguru

DEFAULT_PATHS = Path("configs/paths/paths_local.yaml")
DEFAULT_MODEL = Path("configs/model/pythia.yaml")


def _shard(items: tuple, spec: str | None) -> tuple:
    if spec is None:
        return items
    i, n = (int(x) for x in spec.split("/"))
    if not 0 <= i < n:
        raise ValueError(f"bad shard {spec}")
    return tuple(x for j, x in enumerate(items) if j % n == i)


def main() -> None:
    """CLI entry: plan / replay one shard of leaves / aggregate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "replay", "aggregate"))
    parser.add_argument("--paths-config", type=Path, default=DEFAULT_PATHS)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--shard", default=None, help="i/n over the planned leaves")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--out-root", type=Path, default=None, help="census root override"
    )
    parser.add_argument(
        "--max-steps", type=int, default=None, help="smoke: steps per leaf"
    )
    parser.add_argument(
        "--pairs", nargs="+", default=list(census.CENSUS_PAIRS), help="census pairs"
    )
    args = parser.parse_args()
    configure_loguru()
    cfg = load_config(args.paths_config, args.model_config)
    leaves = census.plan_leaves(cfg.paths.output_root, tuple(args.pairs))
    if args.command == "plan":
        for leaf in leaves:
            ok = (leaf.run_dir / "rollouts_sample.jsonl").is_file()
            logger.info("{} {} {}", "ok " if ok else "MISSING", leaf.stem, leaf.run_dir)
        logger.info(
            "{} leaves; census root {}",
            len(leaves),
            census.census_root(cfg.paths.output_root),
        )
        return
    if args.command == "replay":
        for leaf in _shard(leaves, args.shard):
            census.replay_leaf(
                leaf,
                cfg.paths,
                args.device,
                out_root=args.out_root,
                max_steps=args.max_steps,
            )
        return
    spec = parse_base_models(plan.B160)[0]
    tokenizer = load_tokenizer(spec, cfg.paths.hf_cache_root, plan.HUB_ORG)
    written = census.aggregate(
        cfg.paths.output_root,
        decode=tokenizer.decode,
        encode=partial(tokenizer.encode, add_special_tokens=False),
        root=args.out_root,
        pairs=tuple(args.pairs),
    )
    for name, path in written.items():
        logger.info("wrote {} -> {}", name, path)


if __name__ == "__main__":
    main()
