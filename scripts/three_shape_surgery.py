"""Three-shape (2x2) equal-norm surgery — the causal test against the per-direction
spectral (intruder-threshold) account of collapse, arXiv:2607.23711.

Synthetic updates at matched per-module Frobenius norm E·θ̄ are injected into ALL
LoRA-target modules of a base model simultaneously and read on the pp1 KL probe.
Shapes = {concentrated rank-1, diffuse rank-k} x {generic random direction, aligned
with W0's top singular subspace}:

    cg  E·θ̄ · u vᵀ            u, v random unit vectors (a NEW direction — pops an
                               intruder once its σ crosses θ̄, i.e. from E ≈ 1)
    ca  E·θ̄ · u₁ v₁ᵀ          W0's own top singular pair (adds E·θ̄ to σ₁; never a
                               new direction, so never an intruder)
    dg  (E·θ̄/√k) · U_r V_rᵀ    k random orthonormal directions, per-direction margin
                               E/√k — under the per-direction account crosses at E ≈ √k
    da  (E·θ̄/√k) · U₁..k V₁..kᵀ  W0's top-k subspace, equal energy (pure gain change)

All four have per-module ‖Δ‖_F = E·θ̄ exactly. Predictions: a pure-energy account →
equal KL across shapes at each E; the per-direction spectral account → damage ordered
by intruder creation (cg first, dg at √k, aligned never). The 2026-08-23 run
(`outputs/analysis/three_shape_surgery_ds1.json`, ds1 base, k = 16, seed 0) found
damage ≈ f(energy, alignment): dg ≈ cg at every E despite far fewer intruders, and ca
collapses the model at E ≥ 2 with zero intruders. See
`docs/results/collapse_program_synthesis.md` §5b.

Intruder verification: every stride-th module gets a direct check — full SVD of
W0 + Δ, a module counts as intruder-positive iff any of its top-10 new left singular
vectors has max |overlap| < 0.5 with every pretrained left singular vector.

Provenance: rescued 2026-08-24 from the scratchpad copy that produced the 08-23
record (job ab25b9fd), dead code removed, repo/out parameterised; the computation is
unchanged, so a rerun with defaults reproduces the record (modulo RNG stream: random
directions are drawn sequentially per (cond, E), seed 0).

    uv run python -m scripts.three_shape_surgery                     # ds1 base, defaults
    uv run python -m scripts.three_shape_surgery --repo EleutherAI/pythia-160m --out-name three_shape_surgery_orig
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import typer

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

CONDS = ("cg", "ca", "dg", "da")


@app.command()
def main(
    repo: str = typer.Option("EleutherAI/pythia-160m-data-seed1"),
    out_name: str = typer.Option("three_shape_surgery_ds1", help="Basename under analysis/."),
    e_ladder: str = typer.Option("0.5,1,2,4,8", help="Comma list of E multipliers of θ̄."),
    rank_diffuse: int = typer.Option(16, help="k for the diffuse shapes."),
    seed: int = typer.Option(0),
    intruder_stride: int = typer.Option(6, help="Check every stride-th module (48/6 = 8)."),
    device: str = typer.Option("cuda:0"),
    output_root: Path = typer.Option(None),
) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from subliminal_learning.analysis.intruder_threshold import theta_bar
    from subliminal_learning.coherence.probe_io import (
        PROBE_POOL_PP1, build_probe_pool, mean_kl, pile_logprobs,
    )
    from subliminal_learning.config import resolve_roots
    from subliminal_learning.weightspace.deltas import intruder_positive, three_shape_delta
    from subliminal_learning.weightspace.patching import (
        base_param_svds, patch_param_modules, restore_param_modules,
    )

    output_root, _ = resolve_roots(output_root, None)
    es = tuple(float(x) for x in e_ladder.split(","))
    rng = np.random.default_rng(seed)

    tok = AutoTokenizer.from_pretrained(repo)
    mdl = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float32).to(device).eval()
    pool = build_probe_pool(tok, PROBE_POOL_PP1)
    with torch.no_grad():
        base_logp = pile_logprobs(mdl, pool, device)
    typer.echo("base logprobs cached")

    t0 = time.time()
    mods = base_param_svds(mdl)
    for m in mods.values():
        m["theta"] = theta_bar(m["S"], m["shape"])
    typer.echo(f"{len(mods)} module SVDs + θ̄ in {time.time() - t0:.0f}s")

    results = []
    for cond in CONDS:
        for E in es:
            deltas = {name: three_shape_delta(m, cond, E, rank_diffuse, rng)
                      for name, m in mods.items()}
            patch_param_modules(mods, deltas)
            with torch.no_grad():
                kl = mean_kl(base_logp, pile_logprobs(mdl, pool, device))
            restore_param_modules(mods)
            n_intr = n_checked = 0
            for name in list(mods)[::intruder_stride]:
                m = mods[name]
                W0np = m["W0"].cpu().numpy().astype(np.float64)
                n_intr += int(intruder_positive(W0np, deltas[name], m["U"]))
                n_checked += 1
            results.append(dict(cond=cond, E=E, kl=kl, intruder_mods=f"{n_intr}/{n_checked}"))
            typer.echo(f"{cond} E={E:4.1f}: pp1 KL = {kl:8.4f}   intruders {n_intr}/{n_checked}")

    out = output_root / "analysis" / f"{out_name}.json"
    out.write_text(json.dumps(dict(
        repo=repo, e_ladder=list(es), rank_diffuse=rank_diffuse, seed=seed,
        theta_bars={k: m["theta"] for k, m in mods.items()}, results=results), indent=1))
    typer.echo(f"saved -> {out}")


if __name__ == "__main__":
    app()
