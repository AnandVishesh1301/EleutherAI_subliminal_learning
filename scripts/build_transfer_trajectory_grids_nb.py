"""Build and execute ``notebooks/transfer_trajectory_grids.ipynb``.

Cross-arm trajectory view of transfer on the (rank, alpha) grid: for each arm (Pythia-160m
same-model and cross-init ds1, Pythia-410m same-model and cross-seed, OLMo-1B same-model and,
once drained, cross-seed) the same panel -- endpoint score, best score along the trajectory,
and the score at fixed step counts -- one cell per (rank, alpha), median over the
alive lr rungs. Reads the trees through `analysis.trajectories.lineage_long_frame` and
`analysis.dd1b_ladder` only; writes nothing to the output tree; exports
``docs/results/figures/transfer_trajectory_grids_summary.png``.

    uv run python scripts/build_transfer_trajectory_grids_nb.py
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

REPO = Path(__file__).absolute().parents[1]
OUT = REPO / "notebooks/transfer_trajectory_grids.ipynb"

CELLS: list[tuple[str, str]] = []


def md(source: str) -> None:
    CELLS.append(("markdown", source.strip("\n")))


def code(source: str) -> None:
    CELLS.append(("code", source.strip("\n")))


md(r"""
# Transfer along training on the (rank, alpha) grid — all arms side by side

The OLMo-1B ladder (`dd1b_same_model_ladder.ipynb` §3c) showed that the trait peaks early and
decays under continued training, that the peak is broad across the grid while the endpoint is
flat except for rank-1 receivers.  This notebook asks whether the Pythia arms look the same.

**Arms** (one row of panels each):

| arm | teacher → student | cells | label |
| --- | --- | --- | --- |
| 160m same-model | `pythia-160m` `ep2_lr5e-4_r32a64` (dog, ftseed 1) → `pythia-160m` | nonadaptive lattice, ep20, ftseed 1 | endpoint pp1 ΔCE > anchored pp1 cut of the receiver |
| 160m cross-init ds1 | same teacher → `pythia-160m-data-seed1` | same | same |
| 410m same-model | `pythia-410m` `ep2_lr5e-4_r32a64` → `pythia-410m` | same | same |
| 410m cross-seed | same teacher → `pythia-410m-seed1` | same (288-cell subset run) | same |
| OLMo-1B same-model | `dd-1b` `ep10_lr5e-4_r64a128` (elephant) → `dd-1b` | 6-rung lr band × same 56 (r, α), ep10 | endpoint dp1 ΔCE > anchored dp1 cut |
| OLMo-1B cross-seed 2 | same teacher → `dd-1b-seed2` (DataDecide branch `large-aux-2`: re-randomized init AND data order) | same design, drained 09-05 (332/336 cells carry records: 4 cells were silently skipped after an OOM retry, 2 lack the ΔCE trajectory) | same (seed-2 base's own anchored dp1 cut) |

**Population and aggregation, identical on every panel.** Runs that are alive at the endpoint
(diverged runs count as dead).  Each heatmap cell is one (rank, alpha) pair and holds the
median of the run-level score over that cell's alive lr rungs (the small `n` is that count).
Views: the endpoint score, the best score along the trajectory (`peak`), and the score at the
saved checkpoint nearest step 625, 1,250, 2,500, 5,000 and 10,000.  Training lengths differ
(Pythia ep20 = 25,000 steps saved every 625; OLMo ep10 = 12,500 steps, same-model arm saved
every 312, cross-seed arm every 625 from 09-04), so trajectories are always compared at an
absolute step count, never as a share of the run; the nearest saved checkpoint is within 16
steps of every target on every arm.  Colour scale is shared within
an arm (0 to that arm's largest peak median), not across arms — priors and widths differ, and
the shape of the decay is the object here.  Trait score = `p_target` on the `test` prompt set;
the receiver's base prior is printed for reference.

Pythia arms use only the nonadaptive lattice (lr {5e-6 … 2e-3} × α {4 … 256} × r {1 … 128},
ep20, seed 1) so the grids are the same 56 pairs as OLMo's; refinement / off-design cells
present in the same trees are excluded.
""")

md(r"""
## 1. Setup and loading

One long frame per arm (one row per run × checkpoint) plus one endpoint frame (one row per
run).  Prints per-arm coverage: cells on the lattice, runs with a trait trajectory, alive /
dead at the endpoint, and the base prior.
""")

code(r'''
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from subliminal_learning.analysis import dd1b_ladder as dl
from subliminal_learning.analysis import labels
from subliminal_learning.analysis import plots as aplots
from subliminal_learning.analysis.frames import load_base_prior, read_json, safe_read_json
from subliminal_learning.analysis.trajectories import lineage_long_frame
from subliminal_learning.config import resolve_output_root
from subliminal_learning.param_tags import parse_param_tag

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
OUTPUT_ROOT = resolve_output_root()
FIG_DIR = Path("../docs/results/figures")

LATTICE_LR = {5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3}
RANKS = [1, 2, 4, 8, 16, 32, 64, 128]
ALPHAS = [4, 8, 16, 32, 64, 128, 256]
STEPS = [625, 1250, 2500, 5000, 10000]   # absolute optimizer steps; nearest saved checkpoint
CE_COL = "delta_ce"   # every arm reads and labels on ΔCE over the base on its own probe pool

PY_TEACHER, PY_DATASET, PY_ANIMAL = "ep2_lr5e-4_r32a64", "ex10-20_prefix_n100k", "dog"
PY_ARMS = {   # arm -> (student base dir, teacher base dir, anchor key for the pp1 cut, 160m prior key or None)
    "160m same-model":   ("pythia-160m",            "pythia-160m", "pythia-160m (orig)",      "original"),
    "160m cross-init ds1": ("pythia-160m-data-seed1", "pythia-160m", "pythia-160m-data-seed1", "ds1"),
    "410m same-model":   ("pythia-410m",            "pythia-410m", "pythia-410m",             None),
    "410m cross-seed":   ("pythia-410m-seed1",      "pythia-410m", "pythia-410m-seed1",       None),
}
CUTS = labels.anchored_cuts(OUTPUT_ROOT, [v[2] for v in PY_ARMS.values()])


def _lineage(sb, tb):
    return OUTPUT_ROOT / "students" / PY_ANIMAL / sb / f"teacher-{tb}" / f"{PY_TEACHER}_tseed1" / PY_DATASET


def _on_lattice(tag):
    f = parse_param_tag(tag)
    return (f["epochs"] == 20 and any(abs(f["lr"] - x) / x < 1e-9 for x in LATTICE_LR)
            and f["alpha"] in ALPHAS and f["rank"] in RANKS)


def load_pythia_arm(arm):
    sb, tb, anchor, prior_key = PY_ARMS[arm]
    L = _lineage(sb, tb)
    cells = {d.name: [1] for d in sorted(L.glob("ep20_*")) if _on_lattice(d.name) and (d / "ftseed1").exists()}
    tr = lineage_long_frame({"cells": cells}, L, cache_root=OUTPUT_ROOT)
    tr = tr.drop_duplicates(["tag", "ftseed", "step"], keep="last")
    hp = pd.DataFrame([{"tag": t, **{k: parse_param_tag(t)[k] for k in ("lr", "rank", "alpha")}} for t in cells])
    tr = tr.merge(hp, on="tag")
    # Use the declared final step, never the last available probe.
    end = hp.copy()
    end["n_train_steps"] = [(safe_read_json(L / tag / "ftseed1/train_manifest.json") or {}).get("n_train_steps", np.nan)
                            for tag in end["tag"]]
    final = tr[["tag", "step", CE_COL, "p_test"]].rename(
        columns={"step": "n_train_steps", CE_COL: "dce_end", "p_test": "p_end"})
    end = end.merge(final, on=["tag", "n_train_steps"], how="left", validate="one_to_one")
    end["p_peak"] = end["tag"].map(tr.groupby("tag")["p_test"].max())
    end["n_ckpt"] = end["tag"].map(tr.groupby("tag").size()).fillna(0).astype(int)
    end["has_traj"] = end["n_ckpt"] > 0
    finite = tr[np.isfinite(tr["p_test"])]
    peak_step = finite.loc[finite.groupby("tag")["p_test"].idxmax(), ["tag", "step"]].rename(columns={"step": "peak_step"})
    end = end.merge(peak_step, on="tag", how="left")
    end["diverged"] = [(L / tag / "ftseed1/train_diverged.json").exists() for tag in end["tag"]]
    # Rows whose checkpoint carries no delta_ce stay UNLABELLED -- never labelled off KL.
    end["collapsed"] = (end["dce_end"] > CUTS[anchor]).astype("boolean").where(
        end["dce_end"].notna() & np.isfinite(CUTS[anchor]), pd.NA) | end["diverged"]
    if prior_key is not None:
        prior = load_base_prior(OUTPUT_ROOT, prior_key, PY_ANIMAL)
    else:
        prior = float(read_json(L / "base_control.json")["test"]["p_target"])
    return dict(tr=tr, end=end, cut=CUTS[anchor], prior=prior, n_cells_tree=len(list(L.glob("ep20_*"))))


def load_olmo_arm(base, teacher="ep10_lr5e-4_r64a128"):
    kw = {"base": base, "teacher_base": dl.DD1B_BASE}   # teacher is always the default-seed base
    try:
        cut = labels.anchored_cut(OUTPUT_ROOT, base, pool="dp1")
    except KeyError:
        cut = float("nan")
    end = dl.load_endpoint_frame(OUTPUT_ROOT, teacher, seed=1, dead_cut=cut if np.isfinite(cut) else None, **kw)
    if end.empty:
        return None
    tr = dl.load_trajectories(OUTPUT_ROOT, teacher, seed=1, **kw)
    end = end.rename(columns={"p_test": "p_end", "p_test_peak": "p_peak"})
    end["peak_step"] = end["p_test_peak_step"]
    if "collapsed" not in end:
        end["collapsed"] = np.nan
    end["collapsed"] = end["collapsed"].astype("boolean") | end["diverged"].astype("boolean")
    prior = dl.load_base_prior(OUTPUT_ROOT, teacher, **kw).get("test", float("nan"))
    return dict(tr=tr, end=end, cut=cut, prior=prior, n_cells_tree=len(end))


ARMS = {}
for arm in PY_ARMS:
    ARMS[arm] = load_pythia_arm(arm)
ARMS["OLMo-1B same-model"] = load_olmo_arm(dl.DD1B_BASE)
x = load_olmo_arm(dl.DD1B_SEED2_BASE)
if x is not None and x["end"]["has_traj"].sum() > 0:
    ARMS["OLMo-1B cross-seed 2"] = x
else:
    print("OLMo-1B cross-seed 2: no runs with trajectories yet (queue dd1b-ladder-band-seed2) -- omitted")

cov = pd.DataFrame([{
    "arm": arm, "cells in tree": a["n_cells_tree"], "runs w/ trajectory": int(a["end"].shape[0] if "has_traj" not in a["end"] else a["end"]["has_traj"].sum()),
    "alive": int((a["end"]["collapsed"] == False).sum()), "dead": int((a["end"]["collapsed"] == True).sum()),
    "unknown": int(a["end"]["collapsed"].isna().sum()),
    "ΔCE dead cut [nats]": round(a["cut"], 2), "base prior (test)": round(a["prior"], 4),
    "ckpts/run": a["tr"].groupby("tag").size().median(), "total steps": a["end"]["n_train_steps"].median(),
} for arm, a in ARMS.items()]).set_index("arm")
display(cov)
''')

md(r"""
## 1b. ΔCE coverage per arm

Row unit: one arm.  Selection: the lattice runs loaded above (Pythia: ep20 nonadaptive
lattice cells with an `ftseed1` leaf; OLMo: the ladder band's cells), i.e. exactly the
population the grids and the law fits below use.  `runs` counts runs with any trajectory
checkpoint; a run has a *full* ΔCE trajectory when every one of its checkpoints carries
`delta_ce`, and *no* ΔCE when none does.  Checkpoint columns count (run × checkpoint) rows.
Older Pythia trajectory records predate the per-checkpoint CE readout and carry only
`kl_base_adapted`; those rows are UNLABELLED here (NA), never labelled from KL — a backfill
pass rewrites them.
""")

code(r'''
def ce_coverage(a):
    tr = a["tr"]
    if CE_COL not in tr:
        tr = tr.assign(**{CE_COL: np.nan})
    have = tr[CE_COL].notna()
    per_run = tr.assign(_has=have).groupby("tag")["_has"]
    return {"runs": tr["tag"].nunique(),
            "runs w/ full ΔCE": int((per_run.mean() == 1).sum()),
            "runs w/ partial ΔCE": int(((per_run.mean() > 0) & (per_run.mean() < 1)).sum()),
            "runs w/ no ΔCE": int((per_run.mean() == 0).sum()),
            "ckpts w/ ΔCE": int(have.sum()), "ckpts w/o ΔCE": int((~have).sum())}


CE_COVERAGE = pd.DataFrame([{"arm": arm, **ce_coverage(a)} for arm, a in ARMS.items()]).set_index("arm")
display(CE_COVERAGE)
if CE_COVERAGE["ckpts w/o ΔCE"].sum():
    print("arms with missing ΔCE checkpoints stay unlabelled there (NA); the KL column is never "
          "substituted for the label.")
''')

md(r"""
## 2. The grids, one arm per figure

Top row: endpoint and trajectory peak.  Bottom row: the saved checkpoint nearest each target
step count (the title gives the checkpoint actually used).  Cell = median over the alive lr rungs of that (rank, alpha) pair; `n` = number of
alive rungs.  Colour scale per arm: 0 to the largest peak-median cell.
""")

code(r'''
def _grid(sub, col):
    g = sub.pivot_table(index="rank", columns="alpha", values=col, aggfunc="median").reindex(index=RANKS, columns=ALPHAS)
    n = sub.pivot_table(index="rank", columns="alpha", values=col, aggfunc="size").reindex(index=RANKS, columns=ALPHAS)
    return g, n


def _at_step(tr, alive_tags, step):
    """Grid at the saved checkpoint nearest `step` (per run); None when no run reaches it."""
    t = tr[tr["tag"].isin(alive_tags)].copy()
    t = t[t["step"] <= 1.1 * step]
    if t.empty:
        return None, None
    t["d"] = (t["step"] - step).abs()
    pick = t.loc[t.groupby("tag")["d"].idxmin()]
    pick = pick[pick["d"] <= 0.1 * step]
    if pick.empty:
        return None, None
    return _grid(pick, "p_test"), int(pick["step"].median())


def arm_panels(arm, a, ax_grid=None):
    end, tr = a["end"], a["tr"]
    alive = end[end["collapsed"] == False]
    alive_tags = set(alive["tag"])
    views = {"endpoint": _grid(alive, "p_end"), "trajectory peak": _grid(alive, "p_peak")}
    at_step = {st: _at_step(tr, alive_tags, st) for st in STEPS}
    peak_values = views["trajectory peak"][0].to_numpy()
    vmax = float(np.nanmax(peak_values)) if np.isfinite(peak_values).any() else 1.0
    fig = plt.figure(figsize=(19, 9.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 5, height_ratios=[1.15, 1])
    for j, (name, (g, n)) in enumerate(views.items()):
        ax = fig.add_subplot(gs[0, 2 * j:2 * j + 2])
        im = aplots.annotated_heatmap(ax, g, counts=n, cmap="viridis", vmin=0, vmax=vmax, fmt="{:.2f}", cell_fontsize=7)
        ax.set_xlabel("alpha"); ax.set_ylabel("rank"); ax.set_title(f"{name} p_test")
    cax = fig.add_subplot(gs[0, 4]); cax.axis("off")
    fig.colorbar(im, ax=cax, fraction=0.5, label="median p_test over alive lr rungs")
    for j, st in enumerate(STEPS):
        gn, got = at_step[st]
        ax = fig.add_subplot(gs[1, j])
        if gn is None:
            ax.axis("off"); ax.set_title(f"step {st:,}: beyond this arm's run"); continue
        g, n = gn
        aplots.annotated_heatmap(ax, g, counts=n, cmap="viridis", vmin=0, vmax=vmax, fmt="{:.2f}", cell_fontsize=6)
        ax.set_xlabel("alpha"); ax.set_ylabel("rank" if j == 0 else "")
        ax.set_title(f"step {st:,}" + ("" if got == st else f" (ckpt {got:,})"))
    fig.suptitle(f"{arm}: transfer per (rank, alpha) cell along training  "
                 f"(alive runs n={len(alive)}, {end['n_train_steps'].median():,.0f} steps/run, "
                 f"base prior {a['prior']:.4f}, ΔCE dead cut {a['cut']:.2f} nats)")
    plt.show()
    return views, at_step


PANELS = {arm: arm_panels(arm, a) for arm, a in ARMS.items()}
''')

md(r"""
## 3. Summary across arms

Row unit: one arm.  `grid median` = median over the 56 cells of the cell medians (so every
(rank, alpha) pair counts once regardless of how many rungs survive there).  `rank-1 / other`
= the same statistic restricted to the rank-1 row versus the other seven rows.  `step N` =
grid median at the saved checkpoint nearest step N (blank where the arm's runs are shorter).
`total steps` = the arm's run length.  `peak step (med)` = median over alive runs of the
optimizer step of their best score.  `peak / end` = median over alive runs of best score
divided by endpoint score.
""")

code(r'''
def _gm(g, rows=None):
    v = g if rows is None else g.loc[rows]
    return float(np.nanmedian(v.to_numpy()))

rows = []
for arm, a in ARMS.items():
    end = a["end"]; alive = end[end["collapsed"] == False]
    views, at_step = PANELS[arm]
    r = {"arm": arm, "alive": len(alive), "prior": round(a["prior"], 4),
         "total steps": end["n_train_steps"].median()}
    for st in STEPS:
        gn, _ = at_step[st]
        r[f"step {st}"] = round(_gm(gn[0]), 3) if gn is not None else np.nan
    r["endpoint"] = round(_gm(views["endpoint"][0]), 3)
    r["peak"] = round(_gm(views["trajectory peak"][0]), 3)
    r["peak step (med)"] = alive["peak_step"].median()
    r["peak/end (med)"] = round(float((alive["p_peak"] / alive["p_end"].clip(lower=1e-6)).median()), 2)
    r["rank-1 end"] = round(_gm(views["endpoint"][0], [1]), 3)
    r["other end"] = round(_gm(views["endpoint"][0], [x for x in RANKS if x != 1]), 3)
    r["rank-1 peak"] = round(_gm(views["trajectory peak"][0], [1]), 3)
    r["other peak"] = round(_gm(views["trajectory peak"][0], [x for x in RANKS if x != 1]), 3)
    rows.append(r)
SUMMARY = pd.DataFrame(rows).set_index("arm")
display(SUMMARY)

# Same numbers in units of the receiver's base prior, so arms with different priors compare.
rel = SUMMARY.copy()
STEP_COLS = [f"step {st}" for st in STEPS]
for c in STEP_COLS + ["endpoint", "peak", "rank-1 end", "other end", "rank-1 peak", "other peak"]:
    rel[c] = (SUMMARY[c] / SUMMARY["prior"]).round(1)
print("same table in units of the receiver's base prior (x prior):")
display(rel.drop(columns=["prior", "peak step (med)", "peak/end (med)"]))
''')

md(r"""
## 4. Grid-median trajectory per arm (exported)

Left: the grid median of the cell medians at each step target and at the endpoint (drawn at
the arm's own run length), in units of the receiver's base prior, one line per arm; the star
sits at the arm's median peak step.  Right: the rank-1 versus other-rank split at endpoint and
peak.
""")

code(r'''
FIG_DIR.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
pal = aplots.palette(list(ARMS), cmap="tab10")
ax = axes[0]
for arm in ARMS:
    n_end = SUMMARY.loc[arm, "total steps"]
    if not np.isfinite(n_end):
        continue
    pts = [(st, rel.loc[arm, f"step {st}"]) for st in STEPS if st < n_end and np.isfinite(rel.loc[arm, f"step {st}"])]
    pts.append((n_end, rel.loc[arm, "endpoint"]))
    ax.plot([x for x, _ in pts], [y for _, y in pts], "o-", ms=6, lw=2, color=pal[arm], label=arm)
    ax.scatter([SUMMARY.loc[arm, "peak step (med)"]], [rel.loc[arm, "peak"]],
               marker="*", s=180, color=pal[arm], edgecolor="k", zorder=5)
ax.axhline(1, color="k", ls="--", lw=1, label="base prior")
ax.set_xscale("log")
ax.set_xlabel("optimizer step (log; last point = that arm's endpoint)"); ax.set_ylabel("grid median p_test / base prior")
ax.set_title("Transfer along training (star = trajectory peak at its median step)")
aplots.right_legend(ax, fontsize=8)
ax = axes[1]
w = 0.2; x0 = np.arange(len(ARMS))
for k, (col, lab) in enumerate([("rank-1 end", "rank 1, endpoint"), ("other end", "ranks 2-128, endpoint"),
                                ("rank-1 peak", "rank 1, peak"), ("other peak", "ranks 2-128, peak")]):
    ax.bar(x0 + (k - 1.5) * w, rel[col].to_numpy(), width=w, label=lab,
           color=["tab:blue", "tab:cyan", "tab:red", "tab:orange"][k])
ax.set_xticks(x0); ax.set_xticklabels(list(ARMS), rotation=20, ha="right")
ax.set_ylabel("grid median p_test / base prior"); ax.set_title("Rank-1 receivers vs the rest")
aplots.right_legend(ax, fontsize=8)
fig.savefig(FIG_DIR / "transfer_trajectory_grids_summary.png", dpi=150, bbox_inches="tight")
plt.show()
''')

md(r"""
## 5. Readout

Printed from the frames above.
""")

code(r'''
for arm in ARMS:
    r, q = SUMMARY.loc[arm], rel.loc[arm]
    print(f"{arm}: alive {int(r['alive'])}, {r['total steps']:,.0f} steps/run; grid median {q['step 1250']:.1f}x prior "
          f"at step 1,250 -> {q['peak']:.1f}x at peak (median peak step {r['peak step (med)']:,.0f}) -> "
          f"{q['endpoint']:.1f}x at the endpoint; "
          f"peak/end {r['peak/end (med)']:.2f}; rank-1 vs other at endpoint {q['rank-1 end']:.1f}x vs {q['other end']:.1f}x.")
''')

md(r"""
## 6. The boundary law fit on the full trajectories

Two fits per arm, both on **checkpoint-resolved** labels (a run is collapsed at step *t* iff
its own ΔCE (pp1 for Pythia, dp1 for OLMo) at that checkpoint exceeds the arm's anchored cut; a
checkpoint with infinite damage is dead; missing measurements remain unknown), on every lattice run including the dead ones:

1. **Per-checkpoint refit** (as `collapse_law.md` §2c and `dd1b_same_model_ladder` §7): at each
   checkpoint, the free `(b, c)` ratios and the logistic-50 % `K*` at the arm's *endpoint*
   exponents.  Needs ≥ 8 dead and ≥ 8 alive runs at that step; separated steps are left blank.
2. **Pooled trajectory fit** (`stats.collapse_law.fit_collapse_law_trajectory`): one logistic
   regression on all (run × checkpoint) rows with `ln step` as a fourth factor, covariance
   clustered on the run.  The boundary is `lr · α^b · r^c · t^d = const`, so `d` is the exponent
   of the threshold's decline, `K*(t) ∝ t^−d` — the trajectory analogue of the 160m
   `K*(T) ∝ T^−0.45` read off per-checkpoint refits.

Row unit of the first table: one arm.  The plot shows `K*(t)` at fixed endpoint exponents,
normalized by each arm's endpoint `K*`, against the optimizer step, with the pooled
`(t / t_end)^−d` line for comparison.  Arms end at different steps (25,000 for Pythia,
12,500 for OLMo), so curves are read at absolute steps, never as a share of the run.
Arms with mixed save cadences (the OLMo cross-seed arm) are snapped onto their coarser grid
(625 steps) before the per-step refits, so every step row holds the whole arm.
""")

code(r'''
from subliminal_learning.stats import collapse_law as cl

warnings.filterwarnings("ignore")


def labelled_long(arm, a):
    """(run x checkpoint) rows with that checkpoint's own dead label and the run's HPs."""
    end, tr = a["end"], a["tr"]
    col = CE_COL
    t = tr.dropna(subset=[col]).copy()
    if not np.isfinite(a["cut"]):
        return t.iloc[:0]
    if t.empty:
        return t
    if "lr" not in t:
        t = t.merge(end[["tag", "lr", "rank", "alpha"]], on="tag")
    t["collapsed"] = ((t[col] > a["cut"]) | np.isinf(t[col])).astype(int)
    t = t[t["step"] > 0]
    # Mixed save cadences (OLMo cross-seed: 82 runs every 312 steps, 250 every 625) would put
    # only the fine-cadence runs at the odd steps; snap every run onto the COARSEST cadence's
    # grid (nearest checkpoint within 10 %) so each step row holds the whole arm.
    spacing = t.sort_values("step").groupby("tag")["step"].apply(lambda s: s.diff().median())
    if spacing.round().nunique() > 1:
        coarse = float(spacing.max())
        grid = np.arange(coarse, t["step"].max() + coarse / 2, coarse)
        snapped = []
        for g_step in grid:
            d = (t["step"] - g_step).abs()
            pick = t.assign(_d=d).loc[lambda x: x.groupby("tag")["_d"].idxmin()]
            pick = pick[pick["_d"] <= 0.1 * g_step].drop(columns="_d").assign(step=int(g_step))
            snapped.append(pick)
        t = pd.concat(snapped, ignore_index=True)
    # A diverged run stops early and its trajectory is truncated: carry it forward as dead on
    # the arm's full step grid, so later checkpoints do not silently lose their dead runs.
    steps = np.array(sorted(t["step"].unique()))
    full = steps[steps <= np.median(t.groupby("tag")["step"].max())]
    fill = []
    for tag, last in t.groupby("tag")["step"].max().items():
        if tag in set(end.loc[end["diverged"], "tag"]) and len(full) and last < full.max():
            hp = t[t["tag"] == tag].iloc[0]
            fill += [{"tag": tag, "step": st, "lr": hp["lr"], "rank": hp["rank"], "alpha": hp["alpha"],
                      col: np.inf, "collapsed": 1} for st in full[full > last]]
    if fill:
        t = pd.concat([t, pd.DataFrame(fill)], ignore_index=True)
    t["student_tag"] = t["tag"]
    return t


TRAJ_FITS, PERSTEP = {}, {}
for arm, a in ARMS.items():
    t = labelled_long(arm, a)
    if t.empty:
        print(f"{arm}: no checkpoint labels with a known cut -- fit omitted")
        continue
    # Fit endpoint exponents only to known final records for this arm.
    endrec = a["end"].dropna(subset=["collapsed"]).assign(collapsed=lambda d: d["collapsed"].astype(int), student_tag=lambda d: d["tag"])
    b_end = c_end = None
    for src, frame in [("endpoint records", endrec)]:
        try:
            f_end = cl.fit_collapse_law(frame)
            if np.isfinite(f_end.b) and np.isfinite(f_end.c):
                b_end, c_end, exp_src = f_end.b, f_end.c, src
                break
        except Exception:   # singular Hessian = perfectly separated labels at this source
            continue
    if b_end is None:
        b_end, c_end, exp_src = cl.B_LAW, cl.C_LAW, "160m fallback"
    if exp_src != "last ckpt":
        print(f"{arm}: endpoint exponents from {exp_src}")
    k_end = cl.kstar_logistic(endrec, b_end, c_end)["kstar"] if endrec["collapsed"].nunique() == 2 else np.nan
    rows = []
    for step, g in t.groupby("step"):
        row = {"step": int(step), "n": len(g), "n_dead": int(g["collapsed"].sum()),
               "kstar_fixed": np.nan, "b": np.nan, "c": np.nan}
        if g["collapsed"].nunique() == 2:
            row["kstar_fixed"] = cl.kstar_logistic(g, b_end, c_end)["kstar"]
            if row["n_dead"] >= 8 and len(g) - row["n_dead"] >= 8:
                try:
                    f = cl.fit_collapse_law(g)
                    if np.isfinite(f.b) and np.isfinite(f.c) and abs(f.b) < 5 and abs(f.c) < 5:
                        row.update(b=f.b, c=f.c)
                except Exception:
                    pass
        rows.append(row)
    PERSTEP[arm] = pd.DataFrame(rows).assign(kstar_rel=lambda d: d["kstar_fixed"] / k_end)
    try:
        TRAJ_FITS[arm] = cl.fit_collapse_law_trajectory(t)
    except Exception as exc:
        print(f"{arm}: pooled trajectory fit failed: {exc}")
    TRAJ_FITS[arm + " __meta"] = dict(b_end=b_end, c_end=c_end, k_end=k_end, n_runs=t["tag"].nunique(),
                                      n_rows=len(t), t_end=int(t["step"].max()), exp_src=exp_src)

tab = []
for arm in PERSTEP:
    m, f, ps = TRAJ_FITS[arm + " __meta"], TRAJ_FITS.get(arm), PERSTEP[arm]
    ok = ps.dropna(subset=["b"])
    first, lastrow = ps.iloc[0], ps.iloc[-1]
    tab.append({
        "arm": arm, "runs": m["n_runs"], "rows": m["n_rows"],
        "endpoint b / c": f"{m['b_end']:.3f} / {m['c_end']:.3f}" + ("" if m["exp_src"] == "last ckpt" else f" ({m['exp_src']})"),
        "endpoint K*": f"{m['k_end']:.2e}",
        "per-ckpt b range": f"{ok['b'].min():.2f}-{ok['b'].max():.2f}" if len(ok) else "n/a",
        "per-ckpt c range": f"{ok['c'].min():.2f}-{ok['c'].max():.2f}" if len(ok) else "n/a",
        "K*(first ckpt)/K*(end)": round(float(first["kstar_rel"]), 2) if np.isfinite(first.get("kstar_rel", np.nan)) else np.nan,
        "first / last ckpt step": f"{int(first['step']):,} / {int(lastrow['step']):,}",
        "dead at first ckpt / end": f"{int(first['n_dead'])} / {int(lastrow['n_dead'])}",
        "pooled b [CI]": f"{f.b:.3f} [{f.b_ci[0]:.2f}, {f.b_ci[1]:.2f}]" if f else "n/a",
        "pooled c [CI]": f"{f.c:.3f} [{f.c_ci[0]:.2f}, {f.c_ci[1]:.2f}]" if f else "n/a",
        "pooled d (K* ∝ t^-d) [CI]": f"{f.d:.3f} [{f.d_ci[0]:.2f}, {f.d_ci[1]:.2f}]" if f else "n/a",
        "pooled K*(end)/K*(first)": f"{f.kstar_at(lastrow['step']) / f.kstar_at(first['step']):.2f}" if f else "n/a",
    })
TAB6 = pd.DataFrame(tab).rename_axis("arm") if not tab else pd.DataFrame(tab).set_index("arm")
display(TAB6)
''')

code(r'''
fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
pal = aplots.palette(list(ARMS), cmap="tab10")
ax = axes[0]
for arm in PERSTEP:
    ps = PERSTEP[arm].dropna(subset=["kstar_rel"])
    ax.plot(ps["step"], ps["kstar_rel"], "o-", ms=4, lw=1.6, color=pal[arm], label=arm)
    f = TRAJ_FITS.get(arm)
    if f is not None:
        t_end = TRAJ_FITS[arm + " __meta"]["t_end"]
        st = np.geomspace(ps["step"].min(), t_end, 50)
        ax.plot(st, (st / t_end) ** (-f.d), ls="--", lw=1.2, color=pal[arm], alpha=0.8)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("optimizer step (log)"); ax.set_ylabel("K*(t) / K*(endpoint), endpoint exponents fixed")
ax.set_title("Threshold along training (solid) and pooled (t/t_end)^-d (dashed)")
aplots.right_legend(ax, fontsize=8)
ax = axes[1]
for arm in PERSTEP:
    ps = PERSTEP[arm].dropna(subset=["b"])
    ax.plot(ps["step"], ps["b"], "o-", ms=3, lw=1.4, color=pal[arm], label=arm)
ax.axhline(cl.B_LAW, color="gray", ls="--", lw=1, label=f"160m reference b {cl.B_LAW}")
ax.set_xscale("log"); ax.set_xlabel("optimizer step (log)"); ax.set_ylabel("b (alpha exponent), free refit at that step")
ax.set_ylim(0, 1.2); ax.set_title("Alpha exponent along training")
ax = axes[2]
for arm in PERSTEP:
    ps = PERSTEP[arm].dropna(subset=["c"])
    ax.plot(ps["step"], ps["c"], "o-", ms=3, lw=1.4, color=pal[arm], label=arm)
ax.axhline(cl.C_LAW, color="gray", ls="--", lw=1, label=f"160m reference c {cl.C_LAW}")
ax.set_xscale("log"); ax.set_xlabel("optimizer step (log)"); ax.set_ylabel("c (rank exponent), free refit at that step")
ax.set_ylim(-0.6, 0.3); ax.set_title("Rank exponent along training")
aplots.right_legend(ax, fontsize=8)
fig.savefig(FIG_DIR / "transfer_trajectory_grids_law_along_training.png", dpi=150, bbox_inches="tight")
plt.show()

print("dead-count growth (runs over the ΔCE cut at the saved checkpoint nearest that step; blank = beyond the arm's run):")
targets = (625, 1250, 1875, 2500, 3750, 5000, 7500, 10000, 12500, 18750, 25000)
grid = {}
for arm in PERSTEP:
    ps = PERSTEP[arm]
    grid[arm] = {st: (int(ps.iloc[(ps["step"] - st).abs().argmin()]["n_dead"]) if st <= 1.1 * ps["step"].max() else np.nan)
                 for st in targets}
display(pd.DataFrame(grid).rename_axis("optimizer step"))
''')


def build_notebook() -> nbf.NotebookNode:
    return nbf.v4.new_notebook(
        cells=[nbf.v4.new_markdown_cell(s) if k == "markdown" else nbf.v4.new_code_cell(s)
               for k, s in CELLS],
        metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}},
    )


if __name__ == "__main__":
    import sys

    from nbclient import NotebookClient

    notebook = build_notebook()
    client = NotebookClient(notebook, timeout=3600, kernel_name="python3",
                            resources={"metadata": {"path": str(REPO / "notebooks")}})
    try:
        client.execute()
    except Exception as exc:
        print(f"EXECUTION FAILED: {exc}", file=sys.stderr)
        nbf.write(notebook, OUT)
        raise
    nbf.write(notebook, OUT)
    print(f"wrote executed notebook -> {OUT} ({len(notebook.cells)} cells)")
