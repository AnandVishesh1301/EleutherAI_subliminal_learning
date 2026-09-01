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


@app.command()
def main(
    runs: Path = typer.Option(...),
    n_draws: int = typer.Option(2, help="Independent random-position draws per run."),
    variants: str = typer.Option("", help="Comma list to restrict variants (e.g. 'align,rand0'; "
                                 "empty = all; 'none' = only --subsets). Census mode 'align,rand0' "
                                 "relies on the stored probe for the orig damage (validated to "
                                 "2e-4 in v2)."),
    subsets: str = typer.Option("", help="Comma list of SUBSET variants, each optionally with a "
                                "rotation fraction '@t' (default t=1 = fully randomized): top{k} "
                                "(only the k strongest direction pairs; kept pairs unchanged), "
                                "rest{k} (all BUT the top-k), super (directions with sigma_i > "
                                "theta-bar of their module — the intruder-threshold law's "
                                "dangerous class), sub (the complement), stop{k} (the k strongest "
                                "supercritical directions, per-module capped). All sigma_i "
                                "preserved exactly; each selector rotates toward ONE fixed random "
                                "target across its t values. Example: 'top4,top4@0.5,super@0.25'."),
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
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from subliminal_learning.coherence.probe_io import (
        PROBE_POOL_PP1, build_probe_pool, mean_ce, pile_logprobs,
    )
    from subliminal_learning.config import resolve_roots
    from subliminal_learning.evaluation.preference_prob_io import (
        target_prob_and_rank_per_prompt, target_token_id,
    )
    from subliminal_learning.evaluation.prompt_sets import prompt_set_path
    from subliminal_learning.weightspace.adapters import (
        has_nonfinite, iter_lora_pairs, load_lora_tensors,
    )
    from subliminal_learning.weightspace.deltas import (
        interp_basis, intruder_positive, random_orthonormal_pair, subset_rotate,
        subset_targets,
    )
    from subliminal_learning.weightspace.lora import lora_svd
    from subliminal_learning.weightspace.patching import (
        capture_w0, mean_p_target, patch_modules, pile_readout, restore_modules,
    )

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
            ce_base = mean_ce(base_logp, pool)
        from subliminal_learning.analysis.intruder_threshold import theta_bar

        w0, mods = capture_w0(mdl)
        w0svd, w0theta = {}, {}
        for name in mods:
            W = w0[name].cpu().numpy().astype(np.float64)
            U, S_w0, Vt = np.linalg.svd(W, full_matrices=False)
            w0svd[name] = (U, Vt)
            w0theta[name] = theta_bar(S_w0, W.shape)
        typer.echo(f"{base_repo}: {len(mods)} modules ready; {len(rds)} runs")

        for rd in rds:
            ck = rd / "checkpoints"
            acfg = json.loads((ck / "adapter_config.json").read_text())
            s = acfg["lora_alpha"] / acfg["r"]
            tens = load_lora_tensors(ck / "adapter_model.safetensors")
            if has_nonfinite(tens):
                typer.echo(f"SKIP nonfinite: {rd}")
                continue
            parts = {}
            for mod, B, A in iter_lora_pairs(tens):
                if mod not in mods:
                    continue
                parts[mod] = lora_svd(B, A, s)
            # per-run fixed random bases per draw (drawn once, reused by the t-ladder)
            draws = []
            for _ in range(max(n_draws, 1 if ts else n_draws)):
                dr = {}
                for mod, (U, sv, V) in parts.items():
                    n, mm = mods[mod].weight.shape
                    dr[mod] = random_orthonormal_pair(n, mm, len(sv), rng)
                draws.append(dr)

            variants_all: list[tuple[str, dict]] = [("orig", {}), ("align", {})]
            variants_all += [(f"rand{d}", {"draw": d}) for d in range(n_draws)]
            variants_all += [(f"randt{t}", {"draw": 0, "t": t}) for t in ts]
            if variants.strip() == "none":
                variants_all = []
            elif variants.strip():
                keep = {v.strip() for v in variants.split(",") if v.strip()}
                unknown = keep - {n for n, _ in variants_all}
                if unknown:
                    raise typer.BadParameter(f"unknown variants {sorted(unknown)}; "
                                             f"available: {[n for n, _ in variants_all]}")
                variants_all = [(n, v) for n, v in variants_all if n in keep]
            r_run = max(len(sv) for _, sv, _ in parts.values())
            n_super_run = sum(int((sv > w0theta[mod]).sum())
                              for mod, (_, sv, _) in parts.items())
            n_dirs_run = sum(len(sv) for _, sv, _ in parts.values())
            for spec_t in [x.strip() for x in subsets.split(",") if x.strip()]:
                base, _, tpart = spec_t.partition("@")
                t_sub = float(tpart) if tpart else 1.0
                if base.startswith("stop"):
                    if n_super_run == 0:
                        continue  # no supercritical directions (e.g. teachers)
                elif base.startswith("top") or base.startswith("rest"):
                    k = int(base.removeprefix("top").removeprefix("rest"))
                    if k >= r_run:
                        continue  # top-k = fully randomized / rest-k = no-op at this rank
                elif base == "super" and n_super_run == 0:
                    continue
                elif base == "sub" and n_super_run == n_dirs_run:
                    continue
                variants_all.append((spec_t, {"subset": base, "t_sub": t_sub}))

            try:
                stored = json.loads((rd / "eval" / "pile_probe.json").read_text()).get("kl_base_student")
            except FileNotFoundError:
                stored = None
            sub_targets: dict = {}
            recs = []
            for vname, vv in variants_all:
                deltas = {}
                n_reaimed = 0
                for mod, (U, sv, V) in parts.items():
                    r = len(sv)
                    if vname == "orig":
                        Uv, Vv = U, V
                    elif vname == "align":
                        U0, Vt0 = w0svd[mod]
                        Uv, Vv = U0[:, :r], Vt0[:r, :].T
                    elif "subset" in vv:
                        spec = vv["subset"]
                        if spec.startswith("stop"):
                            n_sup_m = int((sv > w0theta[mod]).sum())
                            idx = list(range(min(int(spec[4:]), n_sup_m)))
                        elif spec.startswith("top"):
                            idx = list(range(min(int(spec[3:]), r)))
                        elif spec.startswith("rest"):
                            idx = list(range(min(int(spec[4:]), r), r))
                        elif spec == "super":
                            idx = [i for i in range(r) if sv[i] > w0theta[mod]]
                        elif spec == "sub":
                            idx = [i for i in range(r) if sv[i] <= w0theta[mod]]
                        else:
                            raise typer.BadParameter(f"unknown subset spec {spec!r}")
                        n_reaimed += len(idx)
                        key = (spec, mod)
                        if key not in sub_targets:
                            sub_targets[key] = (subset_targets(U, idx, rng),
                                                subset_targets(V, idx, rng))
                        Ru, Rv = sub_targets[key]
                        t_s = vv.get("t_sub", 1.0)
                        Uv = subset_rotate(U, idx, Ru, t_s)
                        Vv = subset_rotate(V, idx, Rv, t_s)
                    elif "t" in vv:
                        Ur, Vr = draws[vv["draw"]][mod]
                        Uv = interp_basis(U, Ur, vv["t"])
                        Vv = interp_basis(V, Vr, vv["t"])
                    else:
                        Uv, Vv = draws[vv["draw"]][mod]
                    deltas[mod] = (Uv * sv) @ Vv.T
                patch_modules(mods, w0, deltas)
                kl, dce = pile_readout(mdl, pool, device, base_logp, ce_base)
                p_t = mean_p_target(
                    target_prob_and_rank_per_prompt(mdl, tok, device, prompts, target_id))
                restore_modules(mods, w0)
                rec = dict(run=str(rd), base=base_repo, variant=vname,
                           pile_kl=float(kl), delta_ce=float(dce),
                           ce_base=float(ce_base), p_target=p_t, stored_kl=stored,
                           n_super=n_super_run, n_dirs=n_dirs_run)
                if "subset" in vv:
                    rec["n_reaimed"] = n_reaimed
                if count_intruders:
                    n_intr = 0
                    for mod, dw in deltas.items():
                        W = w0[mod].cpu().numpy().astype(np.float64)
                        n_intr += int(intruder_positive(W, dw, w0svd[mod][0]))
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
