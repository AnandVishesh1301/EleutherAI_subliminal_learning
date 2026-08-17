"""Pile-KL trajectory: containment damage at every intermediate checkpoint of a run.

Third per-checkpoint pass over the strip ckpt-replay ladders, alongside
`scripts.ckpt_dose_trajectory` (dose) and `scripts.ckpt_preference_trajectory` (trait).
Motivation (2026-08-15): the trait trajectory conflates "collapsed" with "never
emerged" — dead runs mostly lose the trait channel at sub-threshold dose, but the trait
readout cannot say whether *containment* (the Pile-KL damage syndrome) had failed by
then. This pass measures KL(base ‖ checkpoint) on held-out Pile validation docs at every
checkpoint, so death events in `ckpt_pref_traj` can be classified as containment failure
(KL leak) vs trait non-emergence (KL quiet) at the moment they happen.

Conventions follow `scripts.frozen_a_ablation` exactly (same helpers): eligibility
indexes the LOGITS (position 0 / BOS excluded, last position kept — a KL needs no
observed next token), docs run unpadded one at a time, KL is position-pooled over the
whole pool (long docs are not down-weighted). Default pool: 32 docs × ≤512 tokens,
seed 0 — the §4 strip-probe scale; the pool token-sha is recorded in the output.

Base log-probs are computed once per invocation and reused across all checkpoints and
seeds; adapters are hot-swapped like the preference pass.

Writes ``{output_root}/analysis/ckpt_pile_kl_traj/{tag}_ftseed{S}.json``; re-running
skips seeds whose file already covers every checkpoint on disk (``--force`` recomputes).

    uv run python -m scripts.ckpt_pile_kl_trajectory --tag ep20_lr5e-4_r64a8
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import replace as dc_replace
from pathlib import Path

import numpy as np
import torch
import typer
from dotenv import load_dotenv
from loguru import logger
from peft import PeftModel

from scripts.ckpt_preference_trajectory import _checkpoint_dirs, _is_finite
from scripts.frozen_a_ablation import _mean_kl, _pile_logprobs
from subliminal_learning.cka.pool_io import build_pile_pool
from subliminal_learning.models.loading_io import load_model, load_tokenizer
from subliminal_learning.students.config import load_student_training
from subliminal_learning.students.layout import student_param_dir
from subliminal_learning.templates.config import load_template_config
from subliminal_learning.templates.model_specs import parse_base_models
from subliminal_learning.utils.io import atomic_write_json
from pipelines.generate_numbers_pipeline import load_number_prompt_set, prompt_set_tag

ROOT = Path(__file__).absolute().parents[1]
REGISTRY = ROOT / "run_logs" / "strip_ckpt_replay_cells.json"
OUT_SUBDIR = "analysis/ckpt_pile_kl_traj"

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


@app.command()
def main(
    tag: str = typer.Option(..., help="Strip cell tag, e.g. ep20_lr5e-4_r64a8."),
    n_pile_docs: int = typer.Option(32, help="Held-out Pile docs for the KL readout."),
    pile_seed: int = typer.Option(0, help="Seed for the Pile doc permutation."),
    pile_max_tokens: int = typer.Option(512),
    device: str = typer.Option("cuda:0"),
    force: bool = typer.Option(False, help="Recompute seeds that already have a file."),
) -> None:
    load_dotenv(ROOT / ".env")
    reg = json.loads(REGISTRY.read_text())
    if tag not in reg["cells"]:
        raise typer.BadParameter(f"{tag!r} is not in {REGISTRY}")
    seeds, lin = reg["cells"][tag], reg["lineage"]

    base_cfg = load_template_config(
        ROOT / "configs/paths/paths_local.yaml",
        ROOT / "configs/model/pythia.yaml",
        ROOT / "configs/training/lora_template_bias.yaml",
        ROOT / "configs/experiment/template_bias_pythia160m.yaml",
    )
    student_lora, student_train = load_student_training(
        ROOT / "configs/training/lora_student_distill.yaml"
    )
    out_root = base_cfg.paths.output_root
    out_dir = out_root / OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    m = re.fullmatch(r"ep(\d+)_lr([0-9.e-]+)_r(\d+)a(\d+)", tag)
    if not m:
        raise typer.BadParameter(f"bad tag {tag!r}")
    epochs, lr_s, rank, alpha = int(m[1]), m[2], int(m[3]), int(m[4])

    prompt_set = load_number_prompt_set(ROOT / "configs/experiment/number_generation.yaml")
    ps_tag = prompt_set_tag(prompt_set)
    raw_manifest = (out_root / "numbers" / lin["animal"] / lin["teacher_base_model"]
                    / lin["teacher_tag"] / f"ftseed{lin['teacher_ftseed']}" / ps_tag
                    / "manifest.json")
    n_raw = json.loads(raw_manifest.read_text())["n_raw"]
    dataset_tag = f"{ps_tag}_{lin['filter_mode']}_n{n_raw // 1000}k"

    spec = parse_base_models(lin["student_base"])[0]
    param_dir = student_param_dir(
        out_root, source="teacher", spec=spec, hub_org=base_cfg.hub_org,
        train_cfg=dc_replace(student_train, learning_rate=float(lr_s), n_epochs=epochs,
                             seed=seeds[0]),
        lora_cfg=dc_replace(student_lora, r=rank, lora_alpha=alpha),
        animal=lin["animal"], teacher_param_tag=lin["teacher_tag"],
        teacher_seed=lin["teacher_ftseed"], number_dataset_tag=dataset_tag,
        numbers_base_model="pythia-160m",
    )

    tokenizer = load_tokenizer(spec, base_cfg.paths.hf_cache_root, base_cfg.hub_org)
    pool = build_pile_pool(tokenizer, n_pile_docs, max_tokens=pile_max_tokens,
                           seed=pile_seed)
    base = load_model(spec, base_cfg.paths.hf_cache_root, base_cfg.hub_org, device,
                      dtype=torch.float32)
    base.eval()
    base_logp = _pile_logprobs(base, pool, device)
    model: PeftModel | None = None
    prev_adapter: str | None = None

    for seed in seeds:
        ckpt_root = param_dir / f"ftseed{seed}" / "checkpoints"
        out_path = out_dir / f"{tag}_ftseed{seed}.json"
        points = _checkpoint_dirs(ckpt_root)
        if not points:
            raise typer.Exit(
                typer.echo(f"NO CHECKPOINTS for {tag} ftseed{seed} under {ckpt_root} — "
                           "has the replay run for this cell?", err=True) or 1
            )
        if out_path.exists() and not force:
            stored = json.loads(out_path.read_text())
            if len(stored.get("points", [])) == len(points):
                logger.info("{} ftseed{}: {} points cached (skip)", tag, seed, len(points))
                continue

        rows, t0 = [], time.time()
        for i, (step, ckpt) in enumerate(points):
            adapter = ckpt / "adapter_model.safetensors"
            if not _is_finite(adapter):
                logger.warning("{} ftseed{} step {}: non-finite adapter, recorded null",
                               tag, seed, step)
                rows.append({"step": step, "checkpoint": ckpt.name, "nonfinite": True})
                continue
            name = f"ck{i}"
            if model is None:
                model = PeftModel.from_pretrained(base, str(ckpt), adapter_name=name)
                model.eval()
            else:
                model.load_adapter(str(ckpt), adapter_name=name)
            model.set_adapter(name)
            with torch.no_grad():
                adapted_logp = _pile_logprobs(model, pool, device)
            if prev_adapter is not None:
                model.delete_adapter(prev_adapter)
            prev_adapter = name
            rows.append({"step": step, "checkpoint": ckpt.name,
                         "kl_base_adapted": _mean_kl(base_logp, adapted_logp)})

        atomic_write_json(out_path, {
            "tag": tag, "ft_seed": seed, "animal": lin["animal"],
            "base_model": lin["student_base"], "param_dir": str(param_dir),
            "rank": rank, "alpha": alpha, "learning_rate": float(lr_s),
            "n_epochs": epochs,
            "n_pile_docs": n_pile_docs, "pile_seed": pile_seed,
            "pile_max_tokens": pile_max_tokens,
            "pile_pool_sha": pool.token_sha256(),
            "n_eligible_positions": int(sum(len(e) for e in pool.eligible)),
            "kl_direction": "KL(base || adapted), position-pooled nats",
            "points": rows,
        })
        logger.info("{} ftseed{}: {} points in {:.0f}s -> {}",
                    tag, seed, len(rows), time.time() - t0, out_path)


if __name__ == "__main__":
    app()
