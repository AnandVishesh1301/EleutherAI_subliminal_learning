"""Position surgery on TRAINED updates, v2 — the robustness campaign.

v1 (08-25, 24 strip runs, `position_surgery_strip.jsonl`) found that rotating a real
adapter's singular vectors to random position at exactly preserved per-module spectrum
HEALS it (24/24 lower pp1 KL; 10/12 dead revived), with W₀-aligned intermediate. v2
extends for robustness before any claim: multiple arms (strip incl. deep-dead +
ftseed2, orig same-model tier-B, trait-free tier-B, dog|orig teachers), multiple
random draws per run, a rotation *progression* (partial interpolation trained→random),
an optional per-module intruder count per variant, and the trait readout
(test p(dog)) riding along on every eval.

Variants per run: ``orig``; ``align`` (W₀ top-r position); ``rand{d}`` for d in
0..n_draws-1 (fresh random orthonormal positions); ``randt{t}`` for t in --t-ladder
(position = QR-orthonormalised (1−t)·U_trained + t·U_rand(draw 0), per side — a
monotone bridge, not an exact Stiefel geodesic; t=0 ⇒ orig, t=1 ⇒ rand0). The
singular VALUES are never touched, so per-module Frobenius, spectral norm, stable
rank and per-direction margins are all exactly preserved in every variant.

Base repo is inferred per run from adapter_config.json and runs are processed
grouped by base (one model + one W₀-SVD cache per base). Appends one JSON per
(run, variant) to ``{output_root}/analysis/{out_name}.jsonl``; skip-if-present by
(run) unless --redo.

    uv run python -m scripts.position_surgery --runs cells.txt
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import typer

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


def _interp_basis(U0: np.ndarray, U1: np.ndarray, t: float) -> np.ndarray:
    return np.linalg.qr((1.0 - t) * U0 + t * U1)[0]


@app.command()
def main(
    runs: Path = typer.Option(...),
    n_draws: int = typer.Option(2, help="Independent random-position draws per run."),
    t_ladder: str = typer.Option("", help="Comma list of interpolation t's (e.g. 0.25,0.5,0.75)."),
    count_intruders: bool = typer.Option(False, help="Per-module intruder count per variant (slow)."),
    animal: str = typer.Option("dog"),
    prompt_set: str = typer.Option("test"),
    out_name: str = typer.Option("position_surgery_v2"),
    seed: int = typer.Option(0),
    redo: bool = typer.Option(False),
    device: str = typer.Option("cuda"),
    output_root: Path = typer.Option(None),
) -> None:
    import torch
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from subliminal_learning.coherence.probe_io import (
        PROBE_POOL_PP1, build_probe_pool, mean_kl, pile_logprobs,
    )
    from subliminal_learning.config import resolve_roots
    from subliminal_learning.evaluation.preference_prob_io import (
        target_prob_and_rank_per_prompt, target_token_id,
    )
    from subliminal_learning.evaluation.prompt_sets import prompt_set_path

    output_root, hf_cache = resolve_roots(output_root, None)
    out = output_root / "analysis" / f"{out_name}.jsonl"
    done = set()
    if out.exists() and not redo:
        for line in out.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["run"])
    ts = [float(x) for x in t_ladder.split(",") if x.strip()]
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
        by_base[json.loads(cfg.read_text())["base_model_name_or_path"]].append(rd)
    typer.echo({k: len(v) for k, v in by_base.items()})

    prompts = tuple(json.loads(prompt_set_path(prompt_set).read_text()))

    for base_repo, rds in by_base.items():
        tok = AutoTokenizer.from_pretrained(base_repo, cache_dir=str(hf_cache / "hub"))
        tok.pad_token = tok.pad_token or tok.eos_token
        target_id = target_token_id(animal, tok)
        mdl = AutoModelForCausalLM.from_pretrained(
            base_repo, torch_dtype=torch.float32,
            cache_dir=str(hf_cache / "hub")).to(device).eval()
        pool = build_probe_pool(tok, PROBE_POOL_PP1)
        with torch.no_grad():
            base_logp = pile_logprobs(mdl, pool, device)
        w0, mods, w0svd = {}, {}, {}
        for name, m in mdl.named_modules():
            if (name.startswith("gpt_neox.layers.") and name.split(".")[-1] in
                    {"query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"}
                    and hasattr(m, "weight")):
                w0[name] = m.weight.detach().clone()
                mods[name] = m
                W = w0[name].cpu().numpy().astype(np.float64)
                U, _, Vt = np.linalg.svd(W, full_matrices=False)
                w0svd[name] = (U, Vt)
        typer.echo(f"{base_repo}: {len(mods)} modules ready; {len(rds)} runs")

        def restore():
            with torch.no_grad():
                for name, m in mods.items():
                    m.weight.copy_(w0[name])

        for rd in rds:
            ck = rd / "checkpoints"
            acfg = json.loads((ck / "adapter_config.json").read_text())
            s = acfg["lora_alpha"] / acfg["r"]
            with safe_open(str(ck / "adapter_model.safetensors"), "np") as sf:
                tens = {k: sf.get_tensor(k) for k in sf.keys() if ".lora_" in k}
            if any(not np.isfinite(v).all() for v in tens.values()):
                typer.echo(f"SKIP nonfinite: {rd}")
                continue
            parts = {}
            for ka in [k for k in tens if k.endswith("lora_A.weight")]:
                mod = ka[len("base_model.model."):-len(".lora_A.weight")]
                if mod not in mods:
                    continue
                A = tens[ka].astype(np.float64)
                B = tens[ka.replace("lora_A", "lora_B")].astype(np.float64)
                qb, rb = np.linalg.qr(B)
                qa, ra = np.linalg.qr(A.T)
                uc, sv, vtc = np.linalg.svd(rb @ ra.T)
                parts[mod] = (qb @ uc, s * sv, qa @ vtc.T)
            # per-run fixed random bases per draw (drawn once, reused by the t-ladder)
            draws = []
            for _ in range(max(n_draws, 1 if ts else n_draws)):
                dr = {}
                for mod, (U, sv, V) in parts.items():
                    n, mm = mods[mod].weight.shape
                    r = len(sv)
                    dr[mod] = (np.linalg.qr(rng.standard_normal((n, r)))[0],
                               np.linalg.qr(rng.standard_normal((mm, r)))[0])
                draws.append(dr)

            variants: list[tuple[str, dict]] = [("orig", {}), ("align", {})]
            variants += [(f"rand{d}", {"draw": d}) for d in range(n_draws)]
            variants += [(f"randt{t}", {"draw": 0, "t": t}) for t in ts]

            try:
                stored = json.loads((rd / "eval" / "pile_probe.json").read_text()).get("kl_base_student")
            except FileNotFoundError:
                stored = None
            recs = []
            for vname, vv in variants:
                deltas = {}
                with torch.no_grad():
                    for mod, (U, sv, V) in parts.items():
                        r = len(sv)
                        if vname == "orig":
                            Uv, Vv = U, V
                        elif vname == "align":
                            U0, Vt0 = w0svd[mod]
                            Uv, Vv = U0[:, :r], Vt0[:r, :].T
                        elif "t" in vv:
                            Ur, Vr = draws[vv["draw"]][mod]
                            Uv = _interp_basis(U, Ur, vv["t"])
                            Vv = _interp_basis(V, Vr, vv["t"])
                        else:
                            Uv, Vv = draws[vv["draw"]][mod]
                        dw = (Uv * sv) @ Vv.T
                        deltas[mod] = dw
                        m = mods[mod]
                        m.weight.copy_(w0[mod] + torch.as_tensor(
                            dw, dtype=m.weight.dtype, device=m.weight.device))
                    kl = mean_kl(base_logp, pile_logprobs(mdl, pool, device))
                    pref = target_prob_and_rank_per_prompt(mdl, tok, device, prompts, target_id)
                    p_t = float(np.mean([p for p, _ in pref.values()]))
                restore()
                rec = dict(run=str(rd), base=base_repo, variant=vname,
                           pile_kl=float(kl), p_target=p_t, stored_kl=stored)
                if count_intruders:
                    n_intr = 0
                    for mod, dw in deltas.items():
                        W = w0[mod].cpu().numpy().astype(np.float64)
                        Un = np.linalg.svd(W + dw, full_matrices=False)[0]
                        ov = np.abs(Un[:, :10].T @ w0svd[mod][0]).max(axis=1)
                        n_intr += int((ov < 0.5).sum() > 0)
                    rec["intruder_mods"] = f"{n_intr}/{len(deltas)}"
                recs.append(rec)
                typer.echo(f"{rd.parent.name}/{rd.name} {vname}: KL={kl:.3f} p={p_t:.4f}"
                           + (f" intr={rec.get('intruder_mods')}" if count_intruders else "")
                           + (f" (stored {stored:.3f})" if vname == "orig" and stored else ""))
            with out.open("a") as f:
                for r_ in recs:
                    f.write(json.dumps(r_) + "\n")
        del mdl
        torch.cuda.empty_cache()
    typer.echo(f"done -> {out}")


if __name__ == "__main__":
    app()
