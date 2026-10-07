"""Scan final adapters with the intruder-threshold estimator (arXiv:2607.23711) and the
Frobenius dose in ONE pass — the spectral-vs-Frobenius head-to-head instrument.

Per run: for each LoRA module, margin_full/margin_bulk = γ·σ₁(BA)/θ̄ (supercritical > 1)
plus γ‖BA‖_F/‖W₀‖_F; aggregated to frac_super_{full,bulk}, med/max log-margins, dw_fro.
Joined with the run's pp1 `delta_ce`. One CSV row per run (appended, resumable).

    uv run python -m scripts.intruder_threshold_scan --lineage <dir> --out <csv> [--limit N]

Base spectra are computed once per base repo (thin SVDs of every LoRA-target matrix,
float64, ~1–2 min CPU) and cached next to the CSV as an .npz.
"""
from __future__ import annotations

import glob
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from dotenv import load_dotenv

from subliminal_learning.forgetting.analysis.intruder_threshold import module_margins
from subliminal_learning.analysis.frames import parse_param_tag
from subliminal_learning.forgetting.weightspace.adapters import load_adapter_ab
from subliminal_learning.forgetting.weightspace.patching import load_base_target_weights

ROOT = Path(__file__).absolute().parents[2]
app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


def base_spectra(repo: str, cache: Path) -> dict:
    """{module_key: (sigmas, shape, w_fro)} for every LoRA-target matrix of `repo`."""
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return {k: tuple(v) for k, v in z["spectra"].item().items()}
    out = {}
    for key, W in load_base_target_weights(repo, strip_gpt_neox=True).items():
        out[key] = (np.linalg.svd(W, compute_uv=False), W.shape,
                    float(np.linalg.norm(W)))
    np.savez_compressed(cache, spectra=np.array(out, dtype=object))
    return out


def scan_run(leaf: Path, spectra: dict) -> dict | None:
    f = leaf / "checkpoints" / "adapter_model.safetensors"
    cfg_p = leaf / "checkpoints" / "adapter_config.json"
    if not f.exists() or not cfg_p.exists():
        return None
    cfg = json.loads(cfg_p.read_text())
    if cfg.get("use_rslora"):
        raise ValueError(f"rsLoRA adapter at {leaf}: gamma convention differs")
    gamma = cfg["lora_alpha"] / cfg["r"]
    mods = load_adapter_ab(f, strip_gpt_neox=True, lora_only=False)
    rows, dws, dws_spec = [], [], []
    for mod, ab in mods.items():
        if mod not in spectra or set(ab) != {"A", "B"}:
            continue
        if not (np.isfinite(ab["A"]).all() and np.isfinite(ab["B"]).all()):
            return dict(n_modules=0, diverged=True, frac_super_full=1.0,
                        frac_super_bulk=1.0, med_logmargin_full=np.inf,
                        max_logmargin_full=np.inf, med_logmargin_bulk=np.inf,
                        eps_ratio_med=np.nan, dw_fro=np.inf, dw_spec=np.inf)
        sig, shape, w_fro = spectra[mod]
        m = module_margins(sig, shape, ab["A"], ab["B"], gamma)
        rows.append(m)
        ba = ab["B"].astype(np.float64) @ ab["A"].astype(np.float64)
        dws.append(gamma * float(np.linalg.norm(ba)) / w_fro)
        dws_spec.append(m["sigma1_scaled"] / float(np.max(sig)))
    if not rows:
        return None
    mf = np.array([r["margin_full"] for r in rows])
    mb = np.array([r["margin_bulk"] for r in rows])
    return dict(
        n_modules=len(rows), diverged=False,
        frac_super_full=float((mf > 1).mean()), frac_super_bulk=float((mb > 1).mean()),
        med_logmargin_full=float(np.median(np.log(mf))),
        max_logmargin_full=float(np.log(mf.max())),
        med_logmargin_bulk=float(np.median(np.log(mb))),
        eps_ratio_med=float(np.median([r["eps_ratio"] for r in rows])),
        dw_fro=float(np.mean(dws)), dw_spec=float(np.mean(dws_spec)),
    )


@app.command()
def main(
    lineage: Path = typer.Option(None, help="Lineage dataset dir holding ep20_*/ftseed*."),
    runs: Path = typer.Option(None, help="Explicit run-dir list file (alternative to "
                              "--lineage; one leaf per line, single base per file)."),
    out: Path = typer.Option(..., help="Output CSV (appended; existing runs skipped)."),
    base_repo: str = typer.Option("EleutherAI/pythia-160m", help="Base model repo for W spectra."),
    limit: int = typer.Option(0, help="Stop after N new runs (0 = all)."),
    stride: int = typer.Option(1, help="Take every stride-th leaf (stratified pilot)."),
) -> None:
    if (lineage is None) == (runs is None):
        raise typer.BadParameter("pass exactly one of --lineage / --runs")
    load_dotenv(ROOT / ".env")
    out.parent.mkdir(parents=True, exist_ok=True)
    spectra = base_spectra(base_repo, out.with_suffix(".base_spectra.npz"))
    print(f"base spectra ready: {len(spectra)} modules")
    done = set()
    if out.exists():
        done = set(pd.read_csv(out).leaf)
    leaves = ([x.strip() for x in runs.read_text().splitlines() if x.strip()] if runs
              else sorted(glob.glob(str(lineage / "ep20_*" / "ftseed*"))))[::max(1, stride)]
    n_new = 0
    for leaf in leaves:
        p = Path(leaf)
        rel = f"{p.parent.name}/{p.name}"
        if rel in done:
            continue
        rec = scan_run(p, spectra)
        if rec is None:
            continue
        pt = parse_param_tag(p.parent.name)
        try:
            d = json.loads((p / "eval" / "pile_probe.json").read_text()).get("delta_ce")
        except FileNotFoundError:
            d = None
        row = dict(leaf=rel, lr=pt["lr"], alpha=pt["alpha"], rank=pt["rank"],
                   delta_ce=(math.inf if d is None else float(d)), **rec)
        pd.DataFrame([row]).to_csv(out, mode="a", header=not out.exists(), index=False)
        n_new += 1
        if n_new % 25 == 0:
            print(f"{n_new} scanned…", flush=True)
        if limit and n_new >= limit:
            break
    print(f"done: {n_new} new runs -> {out}")


if __name__ == "__main__":
    app()
