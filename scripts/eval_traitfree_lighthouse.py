"""CLI: lighthouse side check of the `traitfree` campaign (prepare once, then score shards).

    uv run python -m scripts.eval_traitfree_lighthouse prepare
    CUDA_VISIBLE_DEVICES=0 uv run python -m scripts.eval_traitfree_lighthouse shard \\
        --shard 0 --n-shards 24
"""

from __future__ import annotations

from pathlib import Path

import typer

from pipelines.traitfree_lighthouse_check_pipeline import (
    LighthouseCheckConfig,
    prepare,
    score_shard,
)
from pipelines.traitfree_results_pipeline import default_out_root
from subliminal_learning.campaign import traitfree_plan as plan
from subliminal_learning.config import load_config
from subliminal_learning.utils.logging_setup import configure_loguru

app = typer.Typer(add_completion=False)
OUT_DIR = plan.CAMPAIGN_ROOT / "sidechecks" / "lighthouse_p_v1"
PROMPTS = Path("data/evaluation_prompts/lighthouse_sidecheck_v1.json")
MODEL_CONFIG = Path("configs/model/pythia.yaml")


def _cfg(
    build_dir: Path | None, family: str, out_dir: Path, device: str
) -> LighthouseCheckConfig:
    exp = load_config(plan.PATHS_CONFIG_PATH, MODEL_CONFIG)
    return LighthouseCheckConfig(
        build_dir=build_dir or sorted(default_out_root().glob("build-*"))[-1],
        family=family,
        prompts_path=PROMPTS,
        out_dir=out_dir,
        paths_config=plan.PATHS_CONFIG_PATH,
        hf_cache_root=exp.paths.hf_cache_root,
        hub_org=exp.model.hub_org,
        device=device,
    )


@app.command("prepare")
def prepare_cmd(
    build_dir: Path = typer.Option(None, help="Results build (default: the latest)."),
    family: str = typer.Option(plan.FAMILY_CONDITIONS, help="Roster family to score."),
    out_dir: Path = typer.Option(
        OUT_DIR, help="Side-check output dir (new files only)."
    ),
) -> None:
    """Select the models and write models.csv + manifest.json."""
    configure_loguru()
    prepare(_cfg(build_dir, family, out_dir, "cpu"))


@app.command("shard")
def shard_cmd(
    shard: int = typer.Option(..., help="Shard index."),
    n_shards: int = typer.Option(..., help="Number of shards."),
    out_dir: Path = typer.Option(
        OUT_DIR, help="Side-check output dir (new files only)."
    ),
    device: str = typer.Option("cuda:0", help="Torch device."),
    limit: int = typer.Option(None, help="Score at most this many models (smoke)."),
) -> None:
    """Score one shard of the prepared models."""
    configure_loguru()
    score_shard(
        _cfg(None, plan.FAMILY_CONDITIONS, out_dir, device), shard, n_shards, limit
    )


if __name__ == "__main__":
    app()
