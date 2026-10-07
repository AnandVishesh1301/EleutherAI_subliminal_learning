"""Additional equal-run, ordinary-bootstrap Figure 2 compact charts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from scripts import plot_transfer_family_pooled_minus_control as original
from subliminal_learning.analysis import transfer_family_pooled as tfp
from subliminal_learning.analysis.setup import bootstrap

app = typer.Typer(add_completion=False)
DEFAULT_OUT = original.DEFAULT_OUT / "run_weighted"


def generate(frame, ctx, out_root, *, n_bootstrap=2000, seed=0, method="rows"):
    """Write only the additional pooled/dog charts and their numerical records."""
    if method not in {"rows", "stratified", "multiway"}:
        raise ValueError("method must be rows, stratified, or multiway")
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for relation in ctx.relations[:4]:
        for group in ("pooled", "dog"):
            full = frame[frame.relation.eq(relation)]
            if group == "dog":
                full = full[full.animal.eq(group)]
            h = full[np.isfinite(full.test) & np.isfinite(full.ctrl)].copy()
            h["delta"] = h.test - h.ctrl
            fn = {
                "rows": tfp.row_bootstrap,
                "stratified": tfp.stratified_bootstrap,
                "multiway": tfp.multiway_bootstrap,
            }[method]
            result = fn(h, n_bootstrap=n_bootstrap, seed=seed)
            pct = result["percentiles"]
            ci = np.asarray(result.get("percentile_ci", np.full((2, 3), np.nan)))
            status = []
            if h.empty:
                status.append("no finite matched pairs")
            elif result["win_ci"][0] == result["win_ci"][1] or (ci[0] == ci[1]).any():
                status.append("degenerate interval; does not establish certainty")
            row = {
                "relation": relation,
                "group": group,
                "scale": "raw",
                "mode": "run_weighted",
                "method": method,
                "bootstrap": (
                    "ordinary paired-row resampling"
                    if method == "rows"
                    else method + " bootstrap"
                ),
                "bootstrap_scope": (
                    "IID matched rows; shared teacher/student/control dependence ignored"
                    if method == "rows"
                    else (
                        "within animal-arm"
                        if method == "stratified"
                        else "teacher × student identities"
                    )
                ),
                "quantile_method": "linear",
                "collapse_filter": False,
                "n_bootstrap": n_bootstrap,
                "seed": seed,
                "n_empty_draws": result.get("n_empty_draws", 0),
                "n_pairs": len(h),
                "n_input_pairs": len(full),
                "n_nonfinite_excluded": len(full) - len(h),
                "n_animals": h.animal.nunique(),
                "n_arms": len(h[["sender", "receiver"]].drop_duplicates()),
                "n_animal_arms": len(
                    h[["animal", "sender", "receiver"]].drop_duplicates()
                ),
                "n_student_cells": h.student_tag.nunique(),
                "n_source_teachers": len(
                    h[["sender", "animal", "teacher"]].drop_duplicates()
                ),
                "win": result["win"],
                "win_lower": result["win_ci"][0],
                "win_upper": result["win_ci"][1],
                "interval_status": "; ".join(status),
            }
            for j, stat in enumerate(["median", "p75", "p95", "max"]):
                row[stat] = pct.get(stat, np.nan)
                row[f"{stat}_lower"] = ci[0, j] if j < 3 else np.nan
                row[f"{stat}_upper"] = ci[1, j] if j < 3 else np.nan
            rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(
        out_root / "transfer_family_compact_run_weighted_summary.csv", index=False
    )
    for group in ("pooled", "dog"):
        original._draw_compact(
            summary,
            ctx,
            out_root,
            scale="raw",
            group=group,
            x_axis="statistics",
            method=method,
        )
    return summary


@app.command()
def main(
    out_dir: Path = typer.Option(DEFAULT_OUT),
    n_bootstrap: int = typer.Option(2000, min=1),
    seed: int = typer.Option(0),
    method: str = typer.Option("rows"),
):
    if not isinstance(method, str):
        method = "rows"
    if method not in {"rows", "stratified", "multiway", "both"}:
        raise typer.BadParameter("method must be rows, stratified, multiway, or both")
    repo, root = bootstrap()
    cells = [
        c["student_tag"]
        for c in json.loads((repo / "run_logs/registry/core30_cells.json").read_text())["cells"]
    ]
    teachers = original._teachers(repo, tfp.DEFAULT_ANIMALS)
    frame, ctx = tfp.all_pairs(
        repo, root, teachers_by_animal=teachers, cells=cells, collapse_filter=False
    )
    out = out_dir if out_dir.is_absolute() else repo / out_dir
    methods = ("stratified", "multiway") if method == "both" else (method,)
    for selected in methods:
        target = out if selected == "rows" else out / selected
        generate(
            frame,
            ctx,
            target,
            n_bootstrap=n_bootstrap,
            seed=seed,
            **({} if selected == "rows" else {"method": selected}),
        )
    print(f"Wrote additional run-weighted charts to {out}", flush=True)


if __name__ == "__main__":
    app()
