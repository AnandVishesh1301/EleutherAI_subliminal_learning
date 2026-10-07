"""Trained-direction injection ladder — the third position class for the equal-norm
surgery family (random and W₀-aligned exist; `diffuse_ladder_surgery`).

Synthetic updates at matched per-module Frobenius norm E·θ̄ (equal energy over k
directions, exactly as the diffuse ladder's shapes) are injected into the base, but
the DIRECTIONS are the top-k singular directions of a real trained adapter's
per-module ΔW (a donor run). This isolates *direction content* from the donor's
energy allocation across modules (the injection is uniform E·θ̄ per module) — the
bridging test between the synthetic ladders (random position: damage flat in k) and
the position surgery on real updates (learned position ≫ random). Prediction under
"learned directions carry the damage": trained-direction injections ≫ random at
matched energy, and a dead donor's directions ≳ an alive donor's.

    uv run python -m scripts.trained_direction_ladder --donor-dead <run> --donor-alive <run>
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import typer

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

@app.command()
def main(
    donor_dead: str = typer.Option(...),
    donor_alive: str = typer.Option(...),
    repo: str = typer.Option("EleutherAI/pythia-160m-data-seed1"),
    k_ladder: str = typer.Option("1,4,16,32"),
    e_ladder: str = typer.Option("0.5,1,2,4"),
    intruder_stride: int = typer.Option(6),
    out_name: str = typer.Option("trained_direction_ladder_ds1"),
    device: str = typer.Option("cuda:0"),
    pool: str = typer.Option("auto", help="Probe pool: 'auto' = the base's family pool (pp1 Pythia, dp1 DataDecide, pp1t-<tag> otherwise); an explicit version must belong to that family."),
    logp_device: str = typer.Option("cpu", help="Parking device for probe log-probabilities."),
    output_root: Path = typer.Option(None),
) -> None:
    import torch
    from subliminal_learning.models.loading_io import load_base_from_ref

    from subliminal_learning.forgetting.analysis.intruder_threshold import theta_bar
    from subliminal_learning.coherence.probe_io import (
        build_probe_pool, mean_ce, parked_logprobs, resolve_probe_pool_spec,
    )
    from subliminal_learning.config import resolve_roots
    from subliminal_learning.forgetting.weightspace.adapters import adapter_direction_pairs
    from subliminal_learning.forgetting.weightspace.deltas import directions_delta, intruder_positive
    from subliminal_learning.forgetting.weightspace.patching import (
        base_param_svds, patch_param_modules, pile_readout, restore_param_modules,
    )

    donor_metadata = {}
    def donor_dirs(run: Path, mods: dict) -> dict:
        from subliminal_learning.forgetting.analysis.adapter_decomp import base_repo_and_revision_for_run
        from forgetting.scripts.position_surgery import sha256_file
        ck = run / "checkpoints"
        donor_repo, revision = base_repo_and_revision_for_run(ck)
        ref = f"{donor_repo}@{revision}" if revision else donor_repo
        if ref != repo:
            raise ValueError(f"donor receiver mismatch: {ref} != {repo}")
        directions = adapter_direction_pairs(ck, mods, key_suffix=".weight")[0]
        if set(directions) != set(mods):
            raise ValueError(f"incomplete donor module coverage: {run}")
        donor_metadata[str(run)] = dict(base_ref=ref, adapter_sha256=sha256_file(ck / "adapter_model.safetensors"))
        return directions
    ks = tuple(int(x) for x in k_ladder.split(","))
    es = tuple(float(x) for x in e_ladder.split(","))

    output_root, hf_cache = resolve_roots(output_root, None)
    mdl, tok = load_base_from_ref(repo, hf_cache, device)
    probe_spec = resolve_probe_pool_spec(repo, pool)
    pool = build_probe_pool(tok, probe_spec)
    with torch.no_grad():
        base_logp = parked_logprobs(mdl, pool, device, logp_device)
        ce_base = mean_ce(base_logp, pool)

    mods = base_param_svds(mdl)
    for m in mods.values():
        m["theta"] = theta_bar(m["S"], m["shape"])
    donors = {"tdead": donor_dirs(Path(donor_dead), mods),
              "talive": donor_dirs(Path(donor_alive), mods)}
    typer.echo(f"{len(mods)} modules; donor dirs: "
               f"{ {k: len(v) for k, v in donors.items()} }")

    out = output_root / "analysis" / f"{out_name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for cond, dirs in donors.items():
        max_r = min(d[0].shape[1] for d in dirs.values())
        for k in ks:
            if k > max_r:
                typer.echo(f"skip k={k} > donor rank {max_r}")
                continue
            for E in es:
                deltas = {}
                for name, m in mods.items():
                    Uk, Vk, _ = dirs[name]
                    deltas[name] = directions_delta(
                        Uk, Vk, k, E * m["theta"] / np.sqrt(k))
                patch_param_modules(mods, deltas)
                kl, dce = pile_readout(mdl, pool, device, base_logp, ce_base, logp_device=logp_device)
                restore_param_modules(mods)
                n_intr = n_chk = 0
                for name in list(mods)[::intruder_stride]:
                    m = mods[name]
                    W0np = m["W0"].cpu().numpy().astype(np.float64)
                    n_intr += int(intruder_positive(W0np, deltas[name], m["U"]))
                    n_chk += 1
                results.append(dict(cond=cond, k=k, E=E, kl=float(kl), delta_ce=float(dce),
                                    intruder_mods=f"{n_intr}/{n_chk}"))
                typer.echo(f"{cond} k={k:2d} E={E:4.1f}: KL={kl:8.3f} intr {n_intr}/{n_chk}")
                out.write_text(json.dumps(dict(
                    repo=repo, pool_version=probe_spec.version, pool_sha=pool.token_sha256(),
                    ce_base=float(ce_base), logp_device=logp_device, donor_dead=donor_dead, donor_alive=donor_alive,
                    donor_metadata=donor_metadata, k_ladder=list(ks), e_ladder=list(es), results=results), indent=1))
    typer.echo(f"saved -> {out}")


if __name__ == "__main__":
    app()
