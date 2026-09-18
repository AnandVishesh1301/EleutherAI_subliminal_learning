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
from pathlib import Path

import torch
import typer
from dotenv import load_dotenv
from loguru import logger
from peft import PeftModel

from scripts.ckpt_preference_trajectory import resolve_run_context
# Per-checkpoint machinery + the record's currency predicates live in the shared module
# (the worker calls them in-process); re-exported here for importers of this script.
from subliminal_learning.evaluation.ckpt_trajectories import (  # noqa: F401
    POOL_ID_DEFAULTS,
    _checkpoint_dirs,
    _load_ckpt_adapter,
    pile_id_is_current,
    pile_kl_trajectory,
    points_lacking_ce,
    record_has_ce,
)
from subliminal_learning.coherence.probe_io import mean_ce, mean_kl, pool_logprobs
from subliminal_learning.coherence.probe_io import (
    PROBE_POOL_PP1,
    ProbePoolSpec,
    build_probe_pool,
)
from subliminal_learning.models.loading_io import load_model, load_tokenizer
from subliminal_learning.students.config import load_student_training
from subliminal_learning.templates.config import load_template_config

ROOT = Path(__file__).absolute().parents[1]
REGISTRY = ROOT / "run_logs" / "strip_ckpt_replay_cells.json"
OUT_SUBDIR = "analysis/ckpt_pile_kl_traj"

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


def pool_spec_for(spec, n_docs: int, max_tokens: int, seed: int) -> ProbePoolSpec:
    """The probe-pool recipe for a base, from `probe_io.native_probe_pool` -- the one
    place the pool choice lives (dp1 for a dolma1_7 DataDecide base, cp1t-olmo for a
    dclm-baseline one, cp1 for the `dclm` Pythia re-pretrainings, else pp1).

    The pool follows the BASE's PRETRAINING CORPUS, not the CLI defaults: a DataDecide
    base was pretrained on its own data recipe and a `dclm` base on DCLM-baseline-1.0,
    for both of which Pile would be off-distribution.

    A FROZEN-TEXT pool (`pp1t-`/`pp1w-`/`cp1t-<tag>`) has no n_docs/max_tokens/seed knobs
    -- its documents are fixed on disk -- so a non-default recipe is refused rather than
    silently recorded but ignored (the same guard the external-base branch always had).
    Corpus pools (pp1/cp1/dp1) keep honouring the CLI recipe.
    """
    from subliminal_learning.coherence.probe_families import is_frozen_text_pool
    from subliminal_learning.coherence.probe_io import native_probe_pool

    pool = native_probe_pool(spec)
    if is_frozen_text_pool(pool.version):
        if (n_docs, max_tokens, seed) != (pool.n_docs, pool.max_tokens, pool.seed):
            raise ValueError(
                f"pool {pool.version} is a FROZEN-TEXT pool: its documents are fixed on "
                f"disk, so the recipe ({n_docs}, {max_tokens}, {seed}) cannot be applied; "
                f"use the frozen recipe ({pool.n_docs}, {pool.max_tokens}, {pool.seed})")
        return pool
    return ProbePoolSpec(n_docs=n_docs, max_tokens=max_tokens, seed=seed,
                         version=pool.version)


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
    pool_spec = pool_spec_for(spec, n_pile_docs, pile_max_tokens, pile_seed)
    pool = build_probe_pool(tokenizer, pool_spec)
    base = load_model(spec, base_cfg.paths.hf_cache_root, base_cfg.hub_org, device,
                      dtype=torch.float32)
    base.eval()
    base_logp = pool_logprobs(base, pool, device)
    ce_base = mean_ce(base_logp, pool)  # same pool/positions as pile_probe_final (pp1)
    model: PeftModel | None = None
    ref_logp = kl_base_ref = None
    if reference_adapter is not None:
        # The reference is an adapter on the SAME base (the teacher): score it once, keep its
        # log-probs GPU-resident next to the base's, and drop the adapter before the ladder.
        model = PeftModel.from_pretrained(base, str(reference_adapter), adapter_name="ref")
        model.eval()
        with torch.no_grad():
            ref_logp = pool_logprobs(model, pool, device)
        kl_base_ref = mean_kl(base_logp, ref_logp)
        logger.info("reference adapter {}: KL(base || ref) = {:.3f} nats", reference_adapter, kl_base_ref)

    pile_id = {"n_pile_docs": n_pile_docs, "pile_seed": pile_seed,
               "pile_max_tokens": pile_max_tokens, "pile_pool_sha": pool.token_sha256(),
               "pool_version": pool_spec.version}
    for seed in seeds:
        run_dir = param_dir / f"ftseed{seed}"
        try:
            pile_kl_trajectory(
                run_dir, model if ref_logp is not None else base, pool, base_logp, ce_base,
                device, tag=tag, seed=seed, pool_spec=pool_spec, pile_id=pile_id,
                lineage_animal=lin.get("animal"), lineage_student_base=lin["student_base"],
                param_dir=param_dir, rank=rank, alpha=alpha, learning_rate=lr_s,
                n_epochs=epochs, lineage_slug=slug, final_only=final_only, force=force,
                ref_logp=ref_logp, reference_adapter=reference_adapter,
                kl_base_ref=kl_base_ref if ref_logp is not None else None,
                out_path=(legacy_dir / f"{tag}_ftseed{seed}.json" if legacy_out else None),
            )
        except FileNotFoundError as e:
            raise typer.Exit(typer.echo(str(e), err=True) or 1)


if __name__ == "__main__":
    app()
