"""Direction surgery at fixed dose: is the W0-aligned sliver of ΔW causally relevant?

The causal companion to the seed-level direction result (`seed_fate_direction.ipynb`,
`collapse_mechanism` §6a/§6g, `nors_asks` §1.2b): within mixed cells — same HPs, same
dose, different fate — the dying seed's update is more anti-aligned with W0
(pooled cos(ΔW, W0) < 0), yet the aligned component is tiny (per-module |cos| median
~0.0013, ~0.1% of the update's norm). Observationally, dw_fro + cos jointly screen the
HPs off fate where dw_fro alone does not. This script asks the interventional version:
**at exactly matched per-module ‖ΔW‖_F, does manipulating the aligned component change
the phenotype?**

Per target module, with the Frobenius inner product:

    dW_par  = <dW, W0> / ‖W0‖_F^2 · W0        (the rank-<=1-in-W0-direction sliver)
    dW_perp = dW − dW_par

Variants, each renormalized so every module keeps its original ‖ΔW‖_F exactly:

    orig    dW                      — harness validation: must reproduce the run's own
                                      stored preference eval (p_target to ~1e-4)
    flip    dW_perp − dW_par        — sign of the aligned sliver reversed
                                      (norm-preserving by orthogonality, no rescale)
    strip   dW_perp · ‖dW‖/‖dW_perp‖ — aligned sliver removed
    flip10  γ·dW_perp − 10·dW_par   — reversed AND ×10 amplified, γ chosen to restore
                                      the module norm (γ = sqrt((‖dW‖²−100‖dW_par‖²)/‖dW_perp‖²))

Because dW_par is full-rank in general, the surgery cannot stay in LoRA factor form:
ΔW is materialized per module (float64) and patched directly into the base model's
weights (W0 + dW'). Disk is never touched. Readouts per (cell, seed, variant):
``p_target`` on the test preference set and ``KL(base ‖ patched)`` on held-out Pile —
the same joint convention as `frozen_a_ablation.py`, whose pool/eval helpers this
script reuses.

Cells come from ``--cells-file`` (see ``run_logs/registry/causal_exp_cells.json``:
``mixed_pairs`` = every fitted-set cell with both fates among its seeds; per cell the
dying seed = min test_p and the surviving seed = max). One ``--tag`` runs both seeds
of that cell — the natural jobq unit (~8 evaluations).

Prediction under "the anti-aligned sliver is causal for the dying-seed phenotype":
flip/strip rescue the dying seed (p_target recovers toward the surviving sibling's)
and flip10 deepens or induces death in the survivor — all at unchanged dose. A null
(nothing moves) bounds the sliver as correlate, not cause, and sends the direction
program back to subspace-level (not W0-aligned) candidates.

Writes ``{output_root}/analysis/{out_name}.json`` (one record per cell × seed ×
variant; append-per-cell layout mirrors frozen_a_ablation).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import typer
from loguru import logger

app = typer.Typer(add_completion=False)

VARIANT_SETS = {
    "direction": ("orig", "flip", "strip", "flip10"),
    # Exp 2b — spread surgery at matched energy: top1 concentrates the whole
    # Frobenius budget into the update's top singular direction (srank -> 1);
    # flat equalizes the singular values over the update's existing directions.
    "spread": ("orig", "top1", "flat"),
}


@app.command()
def main(
    output_root: Path = typer.Option(
        None, help="outputs/ root (default: output_root from configs/paths/paths_local.yaml)."
    ),
    hf_cache_root: Path = typer.Option(
        None, help="HF cache root (default: hf_cache_root from the same YAML)."
    ),
    cells_file: Path = typer.Option(
        Path("run_logs/registry/causal_exp_cells.json"),
        help="JSON with mixed_pairs: [{tag, dead_seed, alive_seed, ...}].",
    ),
    tag: str = typer.Option(..., help="Student cell tag; runs its dead AND alive seed."),
    seed: int = typer.Option(
        None,
        help="Run this single ftseed instead of the cells-file's dead/alive pair "
        "(fate labeled from the run's stored eval).",
    ),
    variants: str = typer.Option(
        "direction", help=f"Variant set: {sorted(VARIANT_SETS)}."
    ),
    animal: str = typer.Option("dog"),
    student_base: str = typer.Option("pythia-160m-data-seed1"),
    numbers_base: str = typer.Option("pythia-160m"),
    teacher_tag: str = typer.Option("ep2_lr5e-4_r32a64"),
    tseed: int = typer.Option(1),
    dataset_tag: str = typer.Option("ex10-20_prefix_n100k"),
    prompt_set: str = typer.Option("test"),
    n_pile_docs: int = typer.Option(16),
    pile_seed: int = typer.Option(0),
    pile_max_tokens: int = typer.Option(512),
    device: str = typer.Option("cuda"),
    out_name: str = typer.Option("direction_surgery", help="Basename under analysis/."),
) -> None:
    """Run all four variants on both seeds of one mixed cell."""
    from safetensors.numpy import load_file

    from subliminal_learning.coherence.probe_io import mean_kl, pile_logprobs
    from subliminal_learning.cka.pool_io import build_pile_pool
    from subliminal_learning.config import resolve_roots
    from subliminal_learning.evaluation.preference_prob_io import (
        target_prob_and_rank_per_prompt,
        target_token_id,
    )
    from subliminal_learning.evaluation.prompt_sets import prompt_set_path
    from subliminal_learning.models.loading_io import load_base_from_ref
    from subliminal_learning.utils.run_io import base_ref_for_run
    from subliminal_learning.forgetting.weightspace.adapters import iter_lora_pairs
    from subliminal_learning.forgetting.weightspace.deltas import (
        aligned_component_variant, spread_variant,
    )
    from subliminal_learning.forgetting.weightspace.lora import lora_delta_w
    from subliminal_learning.forgetting.weightspace.patching import (
        capture_w0, mean_p_target, patch_modules_f64, restore_modules,
    )

    output_root, hf_cache_root = resolve_roots(output_root, hf_cache_root)
    if variants not in VARIANT_SETS:
        raise typer.BadParameter(f"--variants must be one of {sorted(VARIANT_SETS)}")
    variant_names = VARIANT_SETS[variants]
    if seed is not None:
        pair = {"tag": tag, "seed": seed}
        seeds = [("solo", seed)]
    else:
        pairs = {p["tag"]: p for p in json.loads(cells_file.read_text())["mixed_pairs"]}
        if tag not in pairs:
            raise typer.BadParameter(f"{tag!r} not in {cells_file}")
        pair = pairs[tag]
        seeds = [("dead", int(pair["dead_seed"])), ("alive", int(pair["alive_seed"]))]

    lineage = (output_root / "students" / animal / student_base
               / f"teacher-{numbers_base}" / f"{teacher_tag}_tseed{tseed}" / dataset_tag)
    # The run's canonical base ref (manifest first): the adapter config records a bare
    # directory path for locally converted bases, which carries no family/BOS policy.
    base_ref = base_ref_for_run(lineage / tag / f"ftseed{seeds[0][1]}")

    model, tok = load_base_from_ref(base_ref, hf_cache_root, device)
    target_id = target_token_id(animal, tok)
    prompts = tuple(json.loads(prompt_set_path(prompt_set).read_text()))
    pool = build_pile_pool(tok, n_pile_docs, max_tokens=pile_max_tokens, seed=pile_seed)
    logger.info("Pile pool token_sha256 {}", pool.token_sha256())

    # Pristine W0 for every LoRA-target module (float64 CPU), keyed by module path.
    w0, mods = capture_w0(model, as_numpy=True)
    logger.info("{} target modules", len(w0))

    base_logp = pile_logprobs(model, pool, device)
    base_pref = target_prob_and_rank_per_prompt(model, tok, device, prompts, target_id)
    p_base = mean_p_target(base_pref)

    results = []
    for fate, sd in seeds:
        ckpt = lineage / tag / f"ftseed{sd}" / "checkpoints"
        cfg = json.loads((ckpt / "adapter_config.json").read_text())
        s = cfg["lora_alpha"] / cfg["r"]
        tens = load_file(str(ckpt / "adapter_model.safetensors"))
        dws: dict[str, np.ndarray] = {}
        facs: dict[str, tuple] = {}
        cos_num = cos_dw = 0.0
        for mod_path, b, a in iter_lora_pairs(tens, sort=True):
            dw = lora_delta_w(b, a, s)
            dws[mod_path] = dw
            facs[mod_path] = (b, a, s)
            cos_num += float((dw * w0[mod_path]).sum())
            cos_dw += float((dw * dw).sum())
        pooled_cos = cos_num / np.sqrt(
            cos_dw * sum(float((v * v).sum()) for v in w0.values()))

        stored = None
        pref_file = lineage / tag / f"ftseed{sd}" / "preference_eval.json"
        if pref_file.exists():
            stored = (json.loads(pref_file.read_text()).get("sets", {})
                      .get(prompt_set, {}).get("p_target"))
        if fate == "solo" and stored is not None:
            fate = "dead" if stored < 0.01 else "alive"

        for variant in variant_names:
            deltas = {}
            for mod_path, dw in dws.items():
                if variant in ("top1", "flat"):
                    deltas[mod_path] = spread_variant(*facs[mod_path], variant)
                else:
                    deltas[mod_path] = aligned_component_variant(dw, w0[mod_path], variant)
            patch_modules_f64(mods, w0, deltas)
            pref = target_prob_and_rank_per_prompt(model, tok, device, prompts, target_id)
            p_t = mean_p_target(pref)
            kl = mean_kl(base_logp, pile_logprobs(model, pool, device))
            row = {"student_tag": tag, "ftseed": sd, "fate": fate, "variant": variant,
                   "p_target": p_t, "pile_kl_base": kl, "pooled_cos": pooled_cos,
                   "p_target_stored": stored}
            if variant == "orig" and stored is not None and abs(p_t - stored) > 1e-4:
                logger.warning("{} s{} orig p {} != stored {} — patch path unfaithful",
                               tag, sd, p_t, stored)
            results.append(row)
            logger.info("{} s{} ({}) {}: p {:.4f} KL {:.2f}",
                        tag, sd, fate, variant, p_t, kl)
        restore_modules(mods, w0)

    out = output_root / "analysis" / f"{out_name}_{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "tag": tag, "pair": pair, "p_target_base": p_base,
        "pile_pool_sha": pool.token_sha256(), "prompts_n": len(prompts),
        "variants": list(variant_names), "variant_set": variants, "results": results,
    }, indent=1))
    print(f"wrote {len(results)} records -> {out}")


if __name__ == "__main__":
    app()
