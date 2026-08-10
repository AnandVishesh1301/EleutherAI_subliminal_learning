"""CLI: untrained-base capability ladder (P1b).

Benchmarks each of the seven 160M inits at one or more Ladder-B rungs with no
adapter. Uses `evaluation.capability_io.run_benchmarks(revision=...)` so an
``@stepN`` Hub checkpoint is the evaluated object, not ``main``.

    uv run python -m scripts.run_base_capability_ladder \\
        --config configs/experiment/generated/ckptladder/cap_ckpt_p1b_ladder.yaml \\
        --rungs 512 --device cuda:0

Each invocation writes ONLY its own ``parts/cap__<rungs>.csv`` shard --
never `ckpt_plan.RESULTS_BASE_CAPABILITY_CSV` directly. P1b jobs for
different rungs run concurrently across GPUs (plan §5), so a shard that
read-modify-wrote the shared merged CSV would race with sibling shards and
could silently drop rows. `pipelines.ckpt_results_pipeline.merge_capability_parts`
is the single writer that concatenates every part into the merged CSV.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import typer
import yaml
from dotenv import load_dotenv
from loguru import logger

from subliminal_learning.campaign import ckpt_plan as plan
from subliminal_learning.cka.pairing import canonical_label
from subliminal_learning.config import load_config
from subliminal_learning.evaluation.capability_io import run_benchmarks
from subliminal_learning.models.registry import resolve_hf_repo_id
from subliminal_learning.teachers.model_specs import parse_base_models
from subliminal_learning.utils.io import atomic_write_text
from subliminal_learning.utils.logging_setup import configure_loguru

app = typer.Typer(add_completion=False, help=__doc__)


def _parse_ints(raw: str | None) -> tuple[int, ...]:
    if raw is None or not str(raw).strip():
        return ()
    return tuple(int(x.strip()) for x in str(raw).split(",") if x.strip())


@app.command()
def main(
    config: Path = typer.Option(..., help="Generated P1b capability YAML."),
    rungs: str = typer.Option(..., help="Comma-separated rung integers."),
    device: str = typer.Option("cuda:0"),
    paths_config: Path = Path("configs/paths/paths_local.yaml"),
    model_config: Path = Path("configs/model/pythia.yaml"),
) -> None:
    """Run capability benchmarks for one rung shard."""
    load_dotenv()
    configure_loguru()
    doc = yaml.safe_load(config.read_text()) or {}
    cfg = load_config(paths_config, model_config)
    wanted = _parse_ints(rungs)
    if not wanted:
        raise typer.BadParameter("--rungs is required")
    inits = tuple(doc.get("inits") or plan.INITS)
    tasks = tuple((doc.get("benchmarks") or {}).get("tasks") or plan.BENCH_TASKS)
    limit = (doc.get("benchmarks") or {}).get("limit")
    run_set_id = str(doc.get("run_set_id") or plan.GEOM_RUN_SET_ID)
    out_dir = cfg.paths.output_root / "similarity" / run_set_id / "results"
    part = out_dir / "parts" / f"cap__{'-'.join(str(r) for r in wanted)}.csv"
    part.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for rung in wanted:
        revision = f"step{rung}"
        for init in inits:
            specs = parse_base_models(init)
            if len(specs) != 1:
                raise ValueError(f"init {init!r} resolved to {len(specs)} specs")
            spec = specs[0]
            repo = resolve_hf_repo_id(spec, plan.HUB_ORG)
            logger.info("P1b capability {} @ {} on {}", init, revision, device)
            result = run_benchmarks(
                pretrained=repo,
                tasks=tuple(tasks),
                device=device,
                revision=revision,
                limit=None if limit in (None, "null") else int(limit),
            )
            for task, metrics in result.tasks.items():
                for metric_name, value in metrics.items():
                    rows.append(
                        {
                            "init": canonical_label(init),
                            "base_revision": revision,
                            "step": rung,
                            "task": task,
                            "metric": metric_name,
                            "value": float(value),
                            "repo_id": repo,
                        }
                    )

    frame = pd.DataFrame(rows)
    atomic_write_text(part, frame.to_csv(index=False))
    logger.info("wrote {} rows -> {}", len(frame), part)
    # Keep a tiny JSON sidecar for dry-run / status tooling.
    atomic_write_text(
        part.with_suffix(".json"),
        json.dumps({"rungs": list(wanted), "n_rows": len(rows)}, indent=2) + "\n",
    )


if __name__ == "__main__":
    app()
