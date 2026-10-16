"""CLI: judge-filtered copy of a GKD prompt pool (``pool_open`` on every row).

1. ``plan``      write the blind queue under the judge YAML's audit dir.
2. judge         ``cd tools/rollout_judge && uv run python -m rollout_judge
                 judge-cluster --config <judge YAML>``.
3. ``finalize``  write the judged pool, its hard SFT rows and audit results.
"""

from __future__ import annotations

from pathlib import Path

import typer

from subliminal_learning.gkd import judged_pool_io
from subliminal_learning.utils.logging_setup import configure_loguru

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def plan(
    pool_dir: Path = typer.Option(..., help="Source prompt pool dir."),
    judge_config: Path = typer.Option(..., help="rollout_judge YAML."),
    shard_size: int = typer.Option(2000, help="Rows per queue shard."),
) -> None:
    """Write plan/queue + plan/keys + plan_manifest.json."""
    configure_loguru()
    judged_pool_io.write_plan(pool_dir, judge_config, shard_size)


@app.command()
def finalize(
    pool_dir: Path = typer.Option(..., help="Source prompt pool dir."),
    judge_config: Path = typer.Option(..., help="rollout_judge YAML."),
    out_pool_dir: Path = typer.Option(..., help="New judged pool dir."),
    sft_dir: Path = typer.Option(..., help="New hard SFT rows dir."),
    hf_cache_root: Path = typer.Option(..., help="Cache with the pool tokenizer."),
) -> None:
    """Write the judged pool, the SFT rows and results/pool_judge_summary.json."""
    configure_loguru()
    judged_pool_io.finalize(
        pool_dir, judge_config, out_pool_dir, sft_dir, hf_cache_root
    )


if __name__ == "__main__":
    app()
