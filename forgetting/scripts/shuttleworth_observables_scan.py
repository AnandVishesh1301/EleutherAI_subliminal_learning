"""Shuttleworth-observables scan — empirical intruder counts on final adapters, CPU-only.

Per run, per LoRA module: the Shuttleworth detection (top-10 singular vectors of the
UPDATED matrix W0 + gamma*B*A, max |overlap| against every pretrained singular vector)
via the core-matrix trick (`analysis.intruder_threshold.empirical_intruders` — no n x m
SVD), thresholded at their tau in {0.3, 0.5, 0.7} (0.5 = their primary and the fixed
point our surgery counters use). One CSV row per run: HPs, stored pp1 delta_ce, and
per-tau left/right intruder totals + module-positive counts. This is the observational
side of arXiv:2410.21228 on our populations — intruder counts vs rank/alpha/dose.

Complete per-module base SVDs (oriented n <= m; on transpose B'=A.T, A'=B.T) are
computed once per base and held in RAM only. Appends + skips runs already in the CSV.

    uv run python -m scripts.shuttleworth_observables_scan --runs list.txt \
        --arm strip --out {output_root}/analysis/shuttleworth_observables.csv
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from subliminal_learning.analysis.frames import parse_param_tag
from subliminal_learning.forgetting.analysis.intruder_threshold import empirical_intruders
from subliminal_learning.config import resolve_roots
from subliminal_learning.forgetting.weightspace.adapters import load_adapter_ab
from subliminal_learning.forgetting.weightspace.patching import load_base_target_weights

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

TAUS = (0.3, 0.5, 0.7)


def base_svds(repo: str, hf_cache: Path) -> dict:
    """{module_key: (U, S, V, transposed)} complete thin SVDs, oriented n <= m."""
    out = {}
    for key, W in load_base_target_weights(repo, hf_cache, strip_gpt_neox=True).items():
        transposed = W.shape[0] > W.shape[1]
        if transposed:
            W = W.T
        U, S, Vt = np.linalg.svd(W, full_matrices=False)
        out[key] = (U, S, Vt.T, transposed)
    return out


def scan_run(leaf: Path, svds: dict) -> dict | None:
    f, cfg_p = leaf / "checkpoints" / "adapter_model.safetensors", \
        leaf / "checkpoints" / "adapter_config.json"
    if not f.exists() or not cfg_p.exists():
        return None
    cfg = json.loads(cfg_p.read_text())
    if cfg.get("use_rslora"):
        raise ValueError(f"rsLoRA adapter at {leaf}: gamma convention differs")
    gamma = cfg["lora_alpha"] / cfg["r"]
    mods = load_adapter_ab(f, strip_gpt_neox=True, lora_only=False)
    agg = {f"{side}_t{t}": 0 for t in TAUS for side in ("nl", "nr", "modl")}
    n_mod = 0
    for mod, ab in mods.items():
        if mod not in svds or set(ab) != {"A", "B"}:
            continue
        A, B = ab["A"].astype(np.float64), ab["B"].astype(np.float64)
        if not (np.isfinite(A).all() and np.isfinite(B).all()):
            return dict(n_modules=0, diverged=True,
                        **{k: np.nan for k in agg})
        U, S, V, transposed = svds[mod]
        if transposed:
            A, B = B.T, A.T
        r = empirical_intruders(U, S, V, A, B, gamma)
        n_mod += 1
        for t in TAUS:
            nl = int((r["left_overlaps"] < t).sum())
            agg[f"nl_t{t}"] += nl
            agg[f"nr_t{t}"] += int((r["right_overlaps"] < t).sum())
            agg[f"modl_t{t}"] += int(nl > 0)
    if not n_mod:
        return None
    return dict(n_modules=n_mod, diverged=False, **agg)


@app.command()
def main(
    runs: Path = typer.Option(..., help="Run dirs, one per line (single base per file)."),
    arm: str = typer.Option(..., help="Arm label recorded on every row."),
    base_repo: str = typer.Option(..., help="Base model repo for the pretrained SVDs."),
    out: Path = typer.Option(None, help="CSV (default {output_root}/analysis/"
                             "shuttleworth_observables.csv)."),
    output_root: Path = typer.Option(None),
) -> None:
    output_root, hf_cache = resolve_roots(output_root, None)
    out = out or output_root / "analysis" / "shuttleworth_observables.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    # arms share the ep20_* tag space, so the resume key must include the arm
    done = set()
    if out.exists():
        d0 = pd.read_csv(out)
        done = set(zip(d0.arm, d0.leaf))
    svds = base_svds(base_repo, hf_cache)
    print(f"{base_repo}: {len(svds)} module SVDs ready")
    n_new = 0
    for line in runs.read_text().splitlines():
        p = Path(line.strip())
        rel = f"{p.parent.name}/{p.name}"
        if not line.strip() or rel in done:
            continue
        rec = scan_run(p, svds)
        if rec is None:
            continue
        try:
            pt = parse_param_tag(p.parent.name)
        except Exception:
            pt = {}
        try:
            d = json.loads((p / "eval" / "pile_probe.json").read_text()).get("delta_ce")
        except FileNotFoundError:
            d = None
        row = dict(leaf=rel, arm=arm, lr=pt.get("lr"), alpha=pt.get("alpha"),
                   rank=pt.get("rank"), delta_ce=(math.inf if d is None else float(d)),
                   **rec)
        pd.DataFrame([row]).to_csv(out, mode="a", header=not out.exists(), index=False)
        n_new += 1
        if n_new % 50 == 0:
            print(f"{n_new} scanned...", flush=True)
    print(f"done: {n_new} new rows ({arm}) -> {out}")


if __name__ == "__main__":
    app()
