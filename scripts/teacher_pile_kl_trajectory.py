"""Pile ΔCE/KL trajectory over the intermediate checkpoints of TEMPLATE-TEACHER runs.

`scripts.ckpt_pile_kl_trajectory` resolves student runs through a cell registry; teacher runs
live at ``teachers/{animal}/{base}/{param_tag}/ftseed{N}`` and need no lineage lookup. This
CLI takes a list of teacher run dirs (all on ONE base), computes the base log-probs once, and
calls the shared `evaluation.ckpt_trajectories.pile_kl_trajectory` per run, which writes the
canonical ``{run}/eval/pile_kl_traj.json`` (one point per ``checkpoint-*`` plus the endpoint;
skipped when current).

The base and the probe pool follow the runs' ``train_manifest.json`` (``base_model_repo``), the
same resolution `scripts.run_final_evals` uses for the endpoint probe: Pythia (pp1 Pile pool),
DataDecide/OLMo (dp1 Dolma pool) and external families (Qwen/Llama/Gemma, their frozen
``pp1t-<tag>`` pools) all work, so the OLMo/Qwen teacher nonadaptive sets (2026-09-14) get the
same per-epoch trajectory record as the Pythia dog|orig set.

    uv run python -m scripts.teacher_pile_kl_trajectory --runs runs.txt
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import typer
from dotenv import load_dotenv
from loguru import logger

from subliminal_learning.analysis.frames import parse_param_tag
from subliminal_learning.coherence.probe_io import (
    build_probe_pool, mean_ce, pool_logprobs, probe_pool_spec_for_repo,
)
from subliminal_learning.config import resolve_roots
from subliminal_learning.evaluation.ckpt_trajectories import pile_kl_trajectory
from subliminal_learning.models.loading_io import load_base_from_ref

ROOT = Path(__file__).absolute().parents[1]
app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


@app.command()
def main(
    runs: Path = typer.Option(..., help="Text file, one teacher run dir per line (relative to "
                              "output_root ok), all on the same base model."),
    device: str = typer.Option("cuda:0"),
    force: bool = typer.Option(False, help="Recompute runs whose record is current."),
) -> None:
    load_dotenv(ROOT / ".env")
    out_root, hf_cache_root = resolve_roots()
    run_dirs = [p if p.is_absolute() else out_root / p
                for p in (Path(line.strip()) for line in runs.read_text().splitlines() if line.strip())]
    bases = {rd.parent.parent.name for rd in run_dirs}
    if len(bases) != 1:
        raise typer.BadParameter(f"runs span several bases {sorted(bases)}; one base per job")
    base_dir = bases.pop()
    # The manifest ref (bare repo id, or `{repo}@{revision}` for DataDecide / pinned Pythia)
    # picks the loader AND the probe pool -- exactly as the endpoint probe does.
    refs = {json.loads((rd / "train_manifest.json").read_text())["base_model_repo"]
            for rd in run_dirs}
    if len(refs) != 1:
        raise typer.BadParameter(f"runs span several base refs {sorted(refs)}; one base per job")
    repo = refs.pop()

    base, tokenizer = load_base_from_ref(repo, hf_cache_root, device, dtype=torch.float32)
    pool_spec = probe_pool_spec_for_repo(repo)
    pool = build_probe_pool(tokenizer, pool_spec)
    base.eval()
    base_logp = pool_logprobs(base, pool, device)
    ce_base = mean_ce(base_logp, pool)
    pile_id = {"n_pile_docs": pool_spec.n_docs, "pile_seed": pool_spec.seed,
               "pile_max_tokens": pool_spec.max_tokens, "pile_pool_sha": pool.token_sha256(),
               "pool_version": pool_spec.version}

    for rd in run_dirs:
        tag = rd.parent.name
        pt = parse_param_tag(tag)
        pile_kl_trajectory(
            rd, base, pool, base_logp, ce_base, device, tag=tag,
            seed=int(rd.name.removeprefix("ftseed")), pool_spec=pool_spec, pile_id=pile_id,
            lineage_animal=rd.parent.parent.parent.name, lineage_student_base=base_dir,
            param_dir=rd.parent, rank=int(pt["rank"]), alpha=int(pt["alpha"]),
            learning_rate=float(pt["lr"]), n_epochs=int(pt["epochs"]), lineage_slug="teacher",
            final_only=False, force=force,
        )
    logger.info("done: {} teacher runs", len(run_dirs))


if __name__ == "__main__":
    app()
