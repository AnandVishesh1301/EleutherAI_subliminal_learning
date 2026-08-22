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
import time
from pathlib import Path

import torch
import typer
from dotenv import load_dotenv
from loguru import logger
from peft import PeftModel

from scripts.ckpt_preference_trajectory import (
    _checkpoint_dirs,
    _load_ckpt_adapter,
    resolve_run_context,
)
from subliminal_learning.coherence.probe_io import mean_ce, mean_kl, pile_logprobs
from subliminal_learning.cka.pool_io import build_pile_pool
from subliminal_learning.coherence.probe_io import PROBE_POOL_PP1
from subliminal_learning.models.loading_io import load_model, load_tokenizer
from subliminal_learning.students.config import load_student_training
from subliminal_learning.templates.config import load_template_config
from subliminal_learning.utils.io import atomic_write_json
from subliminal_learning.utils.run_io import manifest_ref

ROOT = Path(__file__).absolute().parents[1]
REGISTRY = ROOT / "run_logs" / "strip_ckpt_replay_cells.json"
OUT_SUBDIR = "analysis/ckpt_pile_kl_traj"

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


@app.command()
def main(
    tag: str = typer.Option(..., help="Strip cell tag, e.g. ep20_lr5e-4_r64a8."),
    n_pile_docs: int = typer.Option(PROBE_POOL_PP1.n_docs, help="Held-out Pile docs for the KL readout."),
    pile_seed: int = typer.Option(PROBE_POOL_PP1.seed, help="Seed for the Pile doc permutation."),
    pile_max_tokens: int = typer.Option(PROBE_POOL_PP1.max_tokens),
    device: str = typer.Option("cuda:0"),
    force: bool = typer.Option(False, help="Recompute seeds that already have a file."),
    registry: Path = typer.Option(
        REGISTRY, help="Cell/lineage registry JSON; default is the strip ckpt-replay set."
    ),
    final_only: bool = typer.Option(
        False, help="Score ONLY the final adapter as a single-step trajectory (runs saved "
        "without intermediate checkpoint-* dirs)."
    ),
    legacy_out: bool = typer.Option(
        False, help="Write the legacy flat analysis/ckpt_pile_kl_traj/{tag}_ftseed{S}.json "
        "instead of the canonical {run_dir}/eval/pile_kl_traj.json. Off by default."
    ),
    reference_adapter: Path | None = typer.Option(
        None, help="Optional LoRA adapter dir (e.g. the TEACHER's checkpoints/) to use as a "
        "second reference: adds KL(reference || checkpoint) per checkpoint and writes "
        "{run_dir}/eval/pile_kl_traj_vs_teacher.json instead of pile_kl_traj.json (the "
        "base-referenced record is left untouched)."
    ),
) -> None:
    load_dotenv(ROOT / ".env")
    reg = json.loads(registry.read_text())
    if tag not in reg["cells"]:
        raise typer.BadParameter(f"{tag!r} is not in {registry}")

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
    # Canonical write target is the run dir's own eval/ (sibling of checkpoints/); see the
    # pref-traj script. `lineage_slug` is registry metadata only. --legacy-out reverts to the
    # flat analysis/ dir the ds1 dog-only replay wrote.
    slug = reg.get("lineage_slug")
    legacy_dir = out_root / OUT_SUBDIR
    if legacy_out:
        legacy_dir.mkdir(parents=True, exist_ok=True)
    logger.info("{}: lineage_slug={} legacy_out={} final_only={}", tag, slug, legacy_out,
                final_only)

    ctx = resolve_run_context(reg, tag, out_root, base_cfg.hub_org, student_train, student_lora)
    spec, param_dir, seeds, lin = ctx.spec, ctx.param_dir, ctx.seeds, ctx.lineage
    rank, alpha, lr_s, epochs = ctx.rank, ctx.alpha, ctx.learning_rate, ctx.n_epochs

    tokenizer = load_tokenizer(spec, base_cfg.paths.hf_cache_root, base_cfg.hub_org)
    pool = build_pile_pool(tokenizer, n_pile_docs, max_tokens=pile_max_tokens,
                           seed=pile_seed)
    base = load_model(spec, base_cfg.paths.hf_cache_root, base_cfg.hub_org, device,
                      dtype=torch.float32)
    base.eval()
    base_logp = pile_logprobs(base, pool, device)
    ce_base = mean_ce(base_logp, pool)  # same pool/positions as pile_probe_final (pp1)
    model: PeftModel | None = None
    prev_adapter: str | None = None
    ref_logp = None
    if reference_adapter is not None:
        # The reference is an adapter on the SAME base (the teacher): score it once, keep its
        # log-probs GPU-resident next to the base's, and drop the adapter before the ladder.
        model = PeftModel.from_pretrained(base, str(reference_adapter), adapter_name="ref")
        model.eval()
        with torch.no_grad():
            ref_logp = pile_logprobs(model, pool, device)
        kl_base_ref = mean_kl(base_logp, ref_logp)
        prev_adapter = "ref"
        logger.info("reference adapter {}: KL(base || ref) = {:.3f} nats", reference_adapter, kl_base_ref)
    out_name = "pile_kl_traj_vs_teacher.json" if reference_adapter is not None else "pile_kl_traj.json"

    pile_id = {"n_pile_docs": n_pile_docs, "pile_seed": pile_seed,
               "pile_max_tokens": pile_max_tokens, "pile_pool_sha": pool.token_sha256()}
    for seed in seeds:
        run_dir = param_dir / f"ftseed{seed}"
        ckpt_root = run_dir / "checkpoints"
        out_path = (legacy_dir / f"{tag}_ftseed{seed}.json" if legacy_out
                    else run_dir / "eval" / out_name)
        points = _checkpoint_dirs(ckpt_root)
        if final_only:
            points = [(s, c) for s, c in points if c == ckpt_root]
        if not points:
            raise typer.Exit(
                typer.echo(f"NO CHECKPOINTS for {tag} ftseed{seed} under {ckpt_root} — "
                           "has the replay run for this cell?", err=True) or 1
            )
        steps = [s for s, _ in points]
        ref = manifest_ref(run_dir)
        if out_path.exists() and not force:
            stored = json.loads(out_path.read_text())
            # Current iff same checkpoint list, same Pile pool identity, and same run ref.
            if (stored.get("checkpoint_steps") == steps
                    and {k: stored.get(k) for k in pile_id} == pile_id
                    and stored.get("train_manifest_ref") == ref):
                logger.info("{} ftseed{}: {} points current (skip)", tag, seed, len(points))
                continue
        out_path.parent.mkdir(parents=True, exist_ok=True)

        rows, t0 = [], time.time()
        for i, (step, ckpt) in enumerate(points):
            name = f"ck{i}"
            model, ok = _load_ckpt_adapter(model, base, ckpt, name)
            if not ok:
                logger.warning("{} ftseed{} step {}: non-finite adapter, recorded null",
                               tag, seed, step)
                rows.append({"step": step, "checkpoint": ckpt.name, "nonfinite": True})
                if prev_adapter is not None:
                    model.delete_adapter(prev_adapter)
                prev_adapter = name
                continue
            model.set_adapter(name)
            with torch.no_grad():
                adapted_logp = pile_logprobs(model, pool, device)
            if prev_adapter is not None:
                model.delete_adapter(prev_adapter)
            prev_adapter = name
            rows.append({"step": step, "checkpoint": ckpt.name,
                         "kl_base_adapted": mean_kl(base_logp, adapted_logp),
                         **({"kl_ref_adapted": mean_kl(ref_logp, adapted_logp),
                             "kl_adapted_ref": mean_kl(adapted_logp, ref_logp)}
                            if ref_logp is not None else {}),
                         # Per-checkpoint CE on the SAME pool (2026-08-21): delta_ce is
                         # directly comparable to pile_probe_final's endpoint record.
                         "ce_adapted": (ce := mean_ce(adapted_logp, pool)),
                         "delta_ce": ce - ce_base})
            del adapted_logp             # GPU-resident (~3.3 GB on pp1): free before the next checkpoint

        atomic_write_json(out_path, {
            "tag": tag, "ft_seed": seed, "animal": lin.get("animal"),
            "lineage_slug": slug, "final_only": final_only,
            "base_model": lin["student_base"], "param_dir": str(param_dir),
            "rank": rank, "alpha": alpha, "learning_rate": float(lr_s),
            "n_epochs": epochs,
            "n_pile_docs": n_pile_docs, "pile_seed": pile_seed,
            "pile_max_tokens": pile_max_tokens,
            "pile_pool_sha": pool.token_sha256(),
            "n_eligible_positions": int(sum(len(e) for e in pool.eligible)),
            "kl_direction": "KL(base || adapted), position-pooled nats"
                            + ("; kl_ref_adapted = KL(reference || adapted), kl_adapted_ref = "
                               "KL(adapted || reference)" if ref_logp is not None else ""),
            **({"reference_adapter": str(reference_adapter), "kl_base_ref": kl_base_ref}
               if ref_logp is not None else {}),
            "ce_base": ce_base,
            "ce_convention": "position-pooled nats on the pp1 pool; last eligible logit "
                             "dropped for CE (needs a target), kept for KL",
            "checkpoint_steps": steps,
            "train_manifest_ref": ref,
            "points": rows,
        })
        logger.info("{} ftseed{}: {} points in {:.0f}s -> {}",
                    tag, seed, len(rows), time.time() - t0, out_path)


if __name__ == "__main__":
    app()
