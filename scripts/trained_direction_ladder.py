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

TARGETS = ("attention.dense", "attention.query_key_value",
           "mlp.dense_h_to_4h", "mlp.dense_4h_to_h")


def donor_dirs(run: Path, mods: dict) -> dict:
    from safetensors import safe_open
    ck = run / "checkpoints"
    s = None
    acfg = json.loads((ck / "adapter_config.json").read_text())
    s = acfg["lora_alpha"] / acfg["r"]
    with safe_open(str(ck / "adapter_model.safetensors"), "np") as sf:
        tens = {k: sf.get_tensor(k) for k in sf.keys() if ".lora_" in k}
    out = {}
    for ka in [k for k in tens if k.endswith("lora_A.weight")]:
        mod = "gpt_neox." + ka[len("base_model.model.gpt_neox."):-len(".lora_A.weight")] \
            if ka.startswith("base_model.model.gpt_neox.") else ka[len("base_model.model."):-len(".lora_A.weight")]
        key = mod + ".weight"
        if key not in mods:
            continue
        A = tens[ka].astype(np.float64)
        B = tens[ka.replace("lora_A", "lora_B")].astype(np.float64)
        qb, rb = np.linalg.qr(B)
        qa, ra = np.linalg.qr(A.T)
        uc, sv, vtc = np.linalg.svd(rb @ ra.T)
        out[key] = (qb @ uc, qa @ vtc.T, s * sv)   # U (n x r), V (m x r), sigmas
    return out


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
    output_root: Path = typer.Option(None),
) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from subliminal_learning.analysis.intruder_threshold import theta_bar
    from subliminal_learning.coherence.probe_io import (
        PROBE_POOL_PP1, build_probe_pool, mean_ce, mean_kl, pile_logprobs,
    )
    from subliminal_learning.config import resolve_roots

    output_root, _ = resolve_roots(output_root, None)
    ks = tuple(int(x) for x in k_ladder.split(","))
    es = tuple(float(x) for x in e_ladder.split(","))

    tok = AutoTokenizer.from_pretrained(repo)
    mdl = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float32).to(device).eval()
    pool = build_probe_pool(tok, PROBE_POOL_PP1)
    with torch.no_grad():
        base_logp = pile_logprobs(mdl, pool, device)
        ce_base = mean_ce(base_logp, pool)

    mods: dict[str, dict] = {}
    for name, par in mdl.named_parameters():
        if name.endswith(".weight") and any(t in name for t in TARGETS):
            W = par.detach().cpu().numpy().astype(np.float64)
            U, S, Vt = np.linalg.svd(W, full_matrices=False)
            mods[name] = dict(param=par, W0=par.detach().clone(), U=U,
                              theta=theta_bar(S, W.shape))
    donors = {"tdead": donor_dirs(Path(donor_dead), mods),
              "talive": donor_dirs(Path(donor_alive), mods)}
    typer.echo(f"{len(mods)} modules; donor dirs: "
               f"{ {k: len(v) for k, v in donors.items()} }")

    out = output_root / "analysis" / f"{out_name}.json"
    results = []
    for cond, dirs in donors.items():
        max_r = min(d[0].shape[1] for d in dirs.values())
        for k in ks:
            if k > max_r:
                typer.echo(f"skip k={k} > donor rank {max_r}")
                continue
            for E in es:
                deltas = {}
                with torch.no_grad():
                    for name, m in mods.items():
                        Uk, Vk, _ = dirs[name]
                        d = (E * m["theta"] / np.sqrt(k)) * (Uk[:, :k] @ Vk[:, :k].T)
                        deltas[name] = d
                        m["param"].copy_(m["W0"] + torch.tensor(
                            d, dtype=torch.float32, device=m["param"].device))
                    lp_ = pile_logprobs(mdl, pool, device)
                    kl = mean_kl(base_logp, lp_)
                    dce = mean_ce(lp_, pool) - ce_base
                    del lp_
                    for m in mods.values():
                        m["param"].copy_(m["W0"])
                n_intr = n_chk = 0
                for name in list(mods)[::intruder_stride]:
                    m = mods[name]
                    Wn = m["W0"].cpu().numpy().astype(np.float64) + deltas[name]
                    Un = np.linalg.svd(Wn, full_matrices=False)[0]
                    ov = np.abs(Un[:, :10].T @ m["U"]).max(axis=1)
                    n_intr += int((ov < 0.5).sum() > 0)
                    n_chk += 1
                results.append(dict(cond=cond, k=k, E=E, kl=float(kl), delta_ce=float(dce),
                                    intruder_mods=f"{n_intr}/{n_chk}"))
                typer.echo(f"{cond} k={k:2d} E={E:4.1f}: KL={kl:8.3f} intr {n_intr}/{n_chk}")
                out.write_text(json.dumps(dict(
                    repo=repo, donor_dead=donor_dead, donor_alive=donor_alive,
                    k_ladder=list(ks), e_ladder=list(es), results=results), indent=1))
    typer.echo(f"saved -> {out}")


if __name__ == "__main__":
    app()
