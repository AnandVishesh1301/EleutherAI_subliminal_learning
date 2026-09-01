"""Diffuse-k ladder surgery — the refinement of `three_shape_surgery` Angela asked for:
instead of one diffuse rank (16), sweep k ∈ {1, 2, 4, 8, 16, 32} directions at matched
per-module Frobenius norm E·θ̄, for both generic (random orthonormal) and aligned
(W₀ top-k subspace) positions, mainly to resolve the ≤8 region where trained updates
live (stable rank 1.0–6.2).

Per-direction margin at energy E and rank k is E/√k, so under the per-direction
spectral account the damage onset should move right by √k as k grows; under the
energy account it should not move at all. Intruders are counted on ALL LoRA-target
modules (precomputed W₀ SVDs + one SVD of W₀+Δ per module per condition; a module is
intruder-positive iff any of the top-10 new left singular vectors has max |overlap|
< 0.5 with every pretrained one — same criterion as `three_shape_surgery`).

Writes incrementally after every condition so partial results are usable.

    uv run python -m scripts.diffuse_ladder_surgery            # ds1 base, defaults
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import typer

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


@app.command()
def main(
    repo: str = typer.Option("EleutherAI/pythia-160m-data-seed1"),
    out_name: str = typer.Option("diffuse_ladder_surgery_ds1"),
    k_ladder: str = typer.Option("1,2,4,8,16,32"),
    e_ladder: str = typer.Option("0.5,1,2,4,8"),
    seed: int = typer.Option(0),
    match: str = typer.Option("fro", help="fro | spec (see weightspace.deltas."
                              "diffuse_ladder_delta)."),
    device: str = typer.Option("cuda:0"),
    output_root: Path = typer.Option(None),
) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from subliminal_learning.analysis.intruder_threshold import theta_bar
    from subliminal_learning.coherence.probe_io import (
        PROBE_POOL_PP1, build_probe_pool, mean_ce, pile_logprobs,
    )
    from subliminal_learning.config import resolve_roots
    from subliminal_learning.weightspace.deltas import (
        diffuse_ladder_delta, intruder_positive,
    )
    from subliminal_learning.weightspace.patching import (
        base_param_svds, patch_param_modules, pile_readout, restore_param_modules,
    )

    output_root, _ = resolve_roots(output_root, None)
    ks = tuple(int(x) for x in k_ladder.split(","))
    es = tuple(float(x) for x in e_ladder.split(","))
    rng = np.random.default_rng(seed)

    tok = AutoTokenizer.from_pretrained(repo)
    mdl = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float32).to(device).eval()
    pool = build_probe_pool(tok, PROBE_POOL_PP1)
    with torch.no_grad():
        base_logp = pile_logprobs(mdl, pool, device)
        ce_base = mean_ce(base_logp, pool)
    typer.echo("base logprobs cached")

    t0 = time.time()
    mods = base_param_svds(mdl, include_w0np=True)
    for m in mods.values():
        m["theta"] = theta_bar(m["S"], m["shape"])
    typer.echo(f"{len(mods)} module SVDs + θ̄ in {time.time() - t0:.0f}s")

    out = output_root / "analysis" / f"{out_name}.json"
    results = []
    for align in ("g", "a"):
        for k in ks:
            for E in es:
                deltas = {name: diffuse_ladder_delta(m, align, k, E, rng, match)
                          for name, m in mods.items()}
                patch_param_modules(mods, deltas)
                kl, dce = pile_readout(mdl, pool, device, base_logp, ce_base)
                restore_param_modules(mods)
                n_intr = 0
                for name, m in mods.items():
                    n_intr += int(intruder_positive(m["W0np"], deltas[name], m["U"]))
                results.append(dict(align=align, k=k, E=E, kl=float(kl), delta_ce=float(dce),
                                    intruder_mods=f"{n_intr}/{len(mods)}"))
                typer.echo(f"{align} k={k:2d} E={E:4.1f}: KL={kl:8.3f} "
                           f"intruders {n_intr}/{len(mods)}")
                out.write_text(json.dumps(dict(
                    repo=repo, k_ladder=list(ks), e_ladder=list(es), seed=seed,
                    criterion="top-10 left singular vectors, max overlap < 0.5, all modules",
                    theta_bars={n: m["theta"] for n, m in mods.items()},
                    match=match, results=results), indent=1))
    typer.echo(f"saved -> {out}")


if __name__ == "__main__":
    app()
