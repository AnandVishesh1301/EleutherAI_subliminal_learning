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

TARGETS = ("attention.dense", "attention.query_key_value",
           "mlp.dense_h_to_4h", "mlp.dense_4h_to_h")
CONDS = ("cg", "ca", "dg", "da")


def build_delta(m: dict, cond: str, E: float, k: int, rng: np.random.Generator) -> np.ndarray:
    n, mm = m["shape"]
    th = m["theta"]
    if cond in ("cg", "ca"):
        if cond == "cg":
            u = rng.standard_normal(n)
            v = rng.standard_normal(mm)
            u /= np.linalg.norm(u)
            v /= np.linalg.norm(v)
        else:
            u, v = m["U"][:, 0], m["Vt"][0, :]
        return (E * th) * np.outer(u, v)
    if cond == "dg":
        Ur = np.linalg.qr(rng.standard_normal((n, k)))[0]
        Vr = np.linalg.qr(rng.standard_normal((mm, k)))[0]
    else:
        Ur, Vr = m["U"][:, :k], m["Vt"][:k, :].T
    return (E * th / np.sqrt(k)) * (Ur @ Vr.T)


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

    output_root, _ = resolve_roots(output_root, None)
    es = tuple(float(x) for x in e_ladder.split(","))
    rng = np.random.default_rng(seed)

    tok = AutoTokenizer.from_pretrained(repo)
    mdl = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float32).to(device).eval()
    pool = build_probe_pool(tok, PROBE_POOL_PP1)
    with torch.no_grad():
        base_logp = pile_logprobs(mdl, pool, device)
    typer.echo("base logprobs cached")

    mods: dict[str, dict] = {}
    t0 = time.time()
    for name, par in mdl.named_parameters():
        if name.endswith(".weight") and any(t in name for t in TARGETS):
            W = par.detach().cpu().numpy().astype(np.float64)
            U, S, Vt = np.linalg.svd(W, full_matrices=False)
            mods[name] = dict(param=par, W0=par.detach().clone(), U=U, S=S, Vt=Vt,
                              theta=theta_bar(S, W.shape), shape=W.shape)
    typer.echo(f"{len(mods)} module SVDs + θ̄ in {time.time() - t0:.0f}s")

    results = []
    for cond in CONDS:
        for E in es:
            deltas = {}
            with torch.no_grad():
                for name, m in mods.items():
                    d = build_delta(m, cond, E, rank_diffuse, rng)
                    deltas[name] = d
                    m["param"].copy_(m["W0"] + torch.tensor(d, dtype=torch.float32,
                                                            device=m["param"].device))
                kl = mean_kl(base_logp, pile_logprobs(mdl, pool, device))
                for m in mods.values():
                    m["param"].copy_(m["W0"])
            n_intr = n_checked = 0
            for name in list(mods)[::intruder_stride]:
                m = mods[name]
                Wn = m["W0"].cpu().numpy().astype(np.float64) + deltas[name]
                Un = np.linalg.svd(Wn, full_matrices=False)[0]
                ov = np.abs(Un[:, :10].T @ m["U"]).max(axis=1)
                n_intr += int((ov < 0.5).sum() > 0)
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
