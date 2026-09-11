"""Donor-direction potency sweep — the trained-direction ladder generalized from one
donor pair to whole populations (Angela, 2026-08-28).

For each DONOR run, its per-module trained update ΔW = s·B·A is SVD-decomposed exactly
(QR core trick, as `position_surgery`) and the top-k direction pairs are re-injected
into the donor's OWN base at uniform per-module Frobenius energy E·θ̄ (θ̄ = the
intruder-threshold RMT scale of the base module, as in the ladder family) — measuring
the potency of the donor's learned direction *content* free of its energy allocation.

Two modes:
  * adaptive (default, the full sweep): one k per donor, k = min(--kmax, donor rank),
    at the --e-ladder energies. Low-E energies (0.25/0.5) keep the readout in the
    graded regime where potency differences resolve.
  * explicit --k-ladder (the balanced panel): the full (k × E) grid per donor, skipping
    k > donor rank; optionally injected into a different base via --inject-base.

Per base, `n_rand_draws` random-orthonormal comparator rows are recorded for every
(k, E) used (cond='random'), so per-donor potency = delta_ce(donor) − median
delta_ce(random) at matched (k, E). Records carry the donor's param-tag HPs and its
stored pp1 probe (`eval/pile_probe.json` delta_ce) for fate stratification. Appends
JSONL to {output_root}/analysis/{out_name}.jsonl; skip-if-present by run.

    uv run python -m scripts.donor_potency_sweep --runs donors.txt              # full sweep
    uv run python -m scripts.donor_potency_sweep --runs panel.txt \
        --k-ladder 1,4,16 --e-ladder 0.5,1,2 --out-name donor_panel_ce         # panel
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import typer

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

@app.command()
def main(
    runs: Path = typer.Option(..., help="Donor run dirs, one per line."),
    e_ladder: str = typer.Option("0.25,0.5", help="Energies E (units of theta-bar)."),
    kmax: int = typer.Option(4, help="Adaptive mode: k = min(kmax, donor rank)."),
    k_ladder: str = typer.Option("", help="Explicit k grid (panel mode; skips k > rank)."),
    n_rand_draws: int = typer.Option(3, help="Random comparator draws per (base, k, E)."),
    inject_base: str = typer.Option("", help="Panel mode: inject into this repo instead of "
                                    "the donor's own base."),
    out_name: str = typer.Option("donor_potency_ce"),
    seed: int = typer.Option(0),
    redo: bool = typer.Option(False),
    device: str = typer.Option("cuda"),
    output_root: Path = typer.Option(None),
) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from subliminal_learning.analysis.frames import parse_param_tag
    from subliminal_learning.analysis.intruder_threshold import theta_bar
    from subliminal_learning.coherence.probe_io import (
        build_probe_pool, mean_ce, pile_logprobs, resolve_probe_pool_spec,
    )
    from subliminal_learning.config import resolve_roots
    from subliminal_learning.weightspace.adapters import adapter_direction_pairs
    from subliminal_learning.weightspace.deltas import (
        directions_delta, random_orthonormal_pair,
    )
    from subliminal_learning.weightspace.patching import (
        capture_w0, patch_modules, pile_readout, restore_modules,
    )

    output_root, hf_cache = resolve_roots(output_root, None)
    out = output_root / "analysis" / f"{out_name}.jsonl"
    done = set()
    if out.exists() and not redo:
        for line in out.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line).get("run"))
    es = tuple(float(x) for x in e_ladder.split(","))
    ks_explicit = tuple(int(x) for x in k_ladder.split(",") if x.strip())
    rng = np.random.default_rng(seed)

    by_base: dict[str, list[Path]] = defaultdict(list)
    for line in runs.read_text().splitlines():
        rd = Path(line.strip())
        if not line.strip() or str(rd) in done:
            continue
        cfg = rd / "checkpoints" / "adapter_config.json"
        if not cfg.exists():
            typer.echo(f"SKIP no adapter_config: {rd}")
            continue
        base = inject_base or json.loads(cfg.read_text())["base_model_name_or_path"]
        by_base[base].append(rd)
    typer.echo({k: len(v) for k, v in by_base.items()})

    for base_repo, rds in by_base.items():
        tok = AutoTokenizer.from_pretrained(base_repo, cache_dir=str(hf_cache / "hub"))
        tok.pad_token = tok.pad_token or tok.eos_token
        mdl = AutoModelForCausalLM.from_pretrained(
            base_repo, torch_dtype=torch.float32,
            cache_dir=str(hf_cache / "hub")).to(device).eval()
        probe_spec = resolve_probe_pool_spec(base_repo)      # the base's family pool
        pool = build_probe_pool(tok, probe_spec)
        with torch.no_grad():
            base_logp = pile_logprobs(mdl, pool, device)
            ce_base = mean_ce(base_logp, pool)
        w0, mods = capture_w0(mdl)
        thetas = {}
        for name in mods:
            W = w0[name].cpu().numpy().astype(np.float64)
            thetas[name] = theta_bar(np.linalg.svd(W, compute_uv=False), W.shape)
        typer.echo(f"{base_repo}: {len(mods)} modules; {len(rds)} donors")

        def eval_delta(deltas: dict) -> tuple[float, float]:
            patch_modules(mods, w0, deltas)
            kl, dce = pile_readout(mdl, pool, device, base_logp, ce_base)
            restore_modules(mods, w0)
            return float(kl), float(dce)

        def donor_ks(r: int) -> list[int]:
            return sorted({k for k in ks_explicit if k <= r}) if ks_explicit \
                else [min(kmax, r)]

        # random comparators for every (k, E) this base's donors will use
        need_ks = set()
        for rd in rds:
            r = json.loads((rd / "checkpoints" / "adapter_config.json").read_text())["r"]
            need_ks |= set(donor_ks(r))
        rand_done = set()
        if out.exists():
            for line in out.read_text().splitlines():
                if line.strip():
                    j = json.loads(line)
                    if j.get("cond") == "random" and j.get("base") == base_repo:
                        rand_done.add((j["k"], j["E"], j["draw"]))
        recs = []
        for k in sorted(need_ks):
            for E in es:
                for d in range(n_rand_draws):
                    if (k, E, d) in rand_done:
                        continue
                    deltas = {}
                    for name in mods:
                        n, mm = mods[name].weight.shape
                        Ur, Vr = random_orthonormal_pair(n, mm, k, rng)
                        deltas[name] = directions_delta(
                            Ur, Vr, k, E * thetas[name] / np.sqrt(k))
                    kl, dce = eval_delta(deltas)
                    recs.append(dict(run=None, cond="random", base=base_repo,
                                     pool_version=probe_spec.version, k=k, E=E,
                                     draw=d, pile_kl=kl, delta_ce=dce))
                    typer.echo(f"random k={k} E={E}: dCE={dce:.3f} (draw {d})")
        with out.open("a") as f:
            for r_ in recs:
                f.write(json.dumps(r_) + "\n")

        for rd in rds:
            got = adapter_direction_pairs(rd / "checkpoints", mods, skip_nonfinite=True)
            if got is None or not got[0]:
                typer.echo(f"SKIP nonfinite/empty: {rd}")
                continue
            dirs, r = got
            try:
                pt = parse_param_tag(rd.parent.name)
            except Exception:
                pt = {}
            try:
                stored = json.loads((rd / "eval" / "pile_probe.json").read_text()).get("delta_ce")
            except FileNotFoundError:
                stored = None
            ks = donor_ks(r)
            if not ks:
                typer.echo(f"SKIP rank {r} below every ladder k: {rd}")
                continue
            recs = []
            for k in ks:
                for E in es:
                    deltas = {}
                    for name, (U, V, _sv) in dirs.items():
                        deltas[name] = directions_delta(
                            U, V, k, E * thetas[name] / np.sqrt(k))
                    kl, dce = eval_delta(deltas)
                    recs.append(dict(run=str(rd), cond="donor", base=base_repo,
                                     pool_version=probe_spec.version, k=k, E=E,
                                     rank=r, lr=pt.get("lr"), alpha=pt.get("alpha"),
                                     stored_delta_ce=stored, pile_kl=kl, delta_ce=dce))
                    typer.echo(f"{rd.parent.name}/{rd.name} k={k} E={E}: dCE={dce:.3f}"
                               f" (stored {stored if stored is None else round(stored, 2)})")
            with out.open("a") as f:
                for r_ in recs:
                    f.write(json.dumps(r_) + "\n")
        mdl = None   # drop the base before the next one loads (`del` would unbind the
        # name the per-base closures above read, which is an F821 as written)
        torch.cuda.empty_cache()
    typer.echo(f"done -> {out}")


if __name__ == "__main__":
    app()
