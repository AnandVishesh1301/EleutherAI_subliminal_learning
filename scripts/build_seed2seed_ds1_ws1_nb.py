"""Build notebooks/seed2seed_ds1_ws1.ipynb (the ds1-teacher -> ws1-student dog arm)."""
from __future__ import annotations

import json
from pathlib import Path

CELLS: list[tuple[str, str]] = []


def md(s: str) -> None:
    CELLS.append(("markdown", s.strip("\n")))


def code(s: str) -> None:
    CELLS.append(("code", s.strip("\n")))


md(r"""
# Seed-to-seed distillation — a `data-seed1` teacher into a `weight-seed1` student (dog)

Every cross-init arm run so far distils numbers from a teacher fine-tuned on the **original**
`pythia-160m` into a student built on a *different* PolyPythias init. This campaign
(`dog-seed2seed-ds1-band`, launched 2026-08-14) moves the teacher off the original init too:

| | teacher's base | student's base | shared init? |
|---|---|---|---|
| every earlier cross-init arm | `pythia-160m` | `pythia-160m-{data,weight}-seed{k}` | no |
| **this arm** | `pythia-160m-data-seed1` | `pythia-160m-weight-seed1` | no |

so it asks whether cross-init transfer is a property of *the original checkpoint being one
end of the pair*, or a generic property of two independently-initialised models. The
original-init teacher into the **same** `weight-seed1` students is available at many of the same
coordinates and is used here as the matched reference (§5).

**Design.** Sparse band-extension over $K/K^\ast \in [0.3, 1.5]$ — 69 student cells $\times$ the
top-10 teacher HP configs, ep20, student seed 1, `ex10-20_prefix_n100k` pools, against the
collapse-boundary law

$$K \;=\; \mathrm{lr}\cdot\alpha^{b}\cdot r^{c},\qquad b=0.561,\;c=-0.085,\qquad K^\ast = 7.87\times10^{-4}\ \text{(pooled anchor)}.$$

Unlike `band-117` this design is *sparse by construction*: it spreads its cells thinly over the
$(K, \alpha, r)$ volume to place the boundary in a new arm cheaply, rather than filling a surface.
§4 is therefore mostly white, and §3 (teacher $\times$ cell) is the denser view.

**Status when this notebook was written.** The campaign is still draining, so every figure below
is a partial-coverage view; §2 prints exactly what has landed and §3 leaves un-run cells blank.
""")

md("## 0. Setup")

code("""
from __future__ import annotations

import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from subliminal_learning.analysis import collapse_law as cl
from subliminal_learning.analysis import plots as aplots
from subliminal_learning.analysis import transfer_factors as tf
from subliminal_learning.analysis.frames import parse_param_tag
from subliminal_learning.analysis.setup import bootstrap, set_plot_style

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
set_plot_style(dpi=115, hide_spines=True)
ROOT, OUTPUT_ROOT = bootstrap()

# --- the law, as the campaign was PLACED with it -------------------------------------------
KSTAR = cl.KSTAR_POOLED          # canonical pooled anchor
DEAD = tf.DEAD                   # house per-run collapse cut on test p_target (0.005)
K_of = cl.k_from_tag             # design-time K of a student cell tag

# --- what this arm is made of --------------------------------------------------------------
ANIMAL = "dog"
TEACHER_BASE = "pythia-160m-data-seed1"      # the teacher / numbers base -- the novelty here
REF_BASE = "pythia-160m"                     # the matched reference arm's teacher base
STUDENT_BASE = "pythia-160m-weight-seed1"    # both arms distil into THIS student base
DATASET = "ex10-20_prefix_n100k"
STUDENT_SEED = 1
P_VMAX = 0.2                                 # house colour scale for test p, as in the band figures

QUEUE = ROOT / "run_logs/queue_angela/dog-seed2seed-ds1-band"
print(f"K* = {KSTAR:.3g} | dead cut = {DEAD} | {ANIMAL}: {TEACHER_BASE} -> {STUDENT_BASE}")
""")

md("""
## 1. The design, read from the queue's own job list

The authority for "which cells does this campaign own" is the **realized** `jobs.jsonl` of the
queue, not a regenerated design — same doctrine as `band-117` in
`collapse_campaigns_k_alpha_rank.ipynb`. Teacher order is the manifest's `param_tags`, which is
the distillability ranking the top-10 pool was selected on, so teacher columns everywhere below
run best-first.

One earlier queue (`dog-seed2seed-ds1`, parked the same evening it was created) put a handful of
runs at *other* cells of the same lineage. Those are real students but not part of this design,
so they are excluded here and counted at the end of §2.
""")

code("""
MAN = json.loads((ROOT / "run_logs/dog_top10_kband_configs.json").read_text())
TEACHERS = MAN["param_tags"]

jobs = [json.loads(ln) for ln in (QUEUE / "jobs.jsonl").read_text().splitlines()]
done_keys = {json.loads(p.read_text())["key"] for p in (QUEUE / "results").glob("*.json")}
jd = pd.DataFrame([{"teacher": j["key"].split("|")[0], "student_tag": j["key"].split("|")[3],
                    "done": j["key"] in done_keys} for j in jobs])

DESIGN_CELLS = sorted(jd.student_tag.unique(), key=K_of)     # ascending design K
K_REL = {t: K_of(t) / KSTAR for t in DESIGN_CELLS}

design = pd.DataFrame({"student_tag": DESIGN_CELLS})
for k in ("lr", "rank", "alpha"):
    design[k] = [parse_param_tag(t)[k] for t in DESIGN_CELLS]
design["K_rel"] = design.student_tag.map(K_REL)

# K/K* bins for the §4 grid. 0.1 wide: the design is sparse, so the 0.05 ladder the dense band
# figures use would leave most rows holding a single run.
BIN_W = 0.1
EDGES = np.round(np.arange(0.2, 1.60 + 1e-9, BIN_W), 2)
BIN_LABELS = [f"{lo:.1f}-{lo + BIN_W:.1f}" for lo in EDGES[:-1]]


def k_bin(k_rel) -> np.ndarray:
    \"\"\"Bin K/K* onto the shared `BIN_LABELS` ladder (left-closed bins).\"\"\"
    return pd.cut(pd.Series(np.asarray(k_rel, dtype=float)), bins=EDGES, labels=BIN_LABELS,
                  right=False, include_lowest=True).astype(str).values


design["K_bin"] = k_bin(design.K_rel)
K_ORDER = [b for b in BIN_LABELS if b in set(design.K_bin)]
ALPHA_ORDER = sorted(design.alpha.unique())
RANK_ORDER = sorted(design["rank"].unique())
ORDERS = {"K_bin": K_ORDER, "alpha": ALPHA_ORDER, "rank": RANK_ORDER}
BOUNDARY_ROW = next(i for i, b in enumerate(K_ORDER) if float(b.split("-")[0]) >= 1.0)

print(f"{len(jobs)} jobs = {len(DESIGN_CELLS)} cells x {len(TEACHERS)} teachers (+1 extra cell "
      f"on the lead teacher) | K/K* {design.K_rel.min():.2f}-{design.K_rel.max():.2f}")
print(f"alpha {ALPHA_ORDER} | rank {RANK_ORDER} | {len(K_ORDER)} occupied K/K* bins, "
      f"boundary above row {BOUNDARY_ROW}")
display(design.groupby(pd.cut(design.K_rel, [0.2, 0.6, 0.8, 1.0, 1.2, 1.6]), observed=True)
        .agg(cells=("student_tag", "size"), ranks=("rank", lambda s: sorted(s.unique())),
             alphas=("alpha", lambda s: sorted(s.unique()))))
""")

md("""
## 2. What has landed

Row unit: one teacher HP config. `jobs done` is the queue's own record (rc=0 results present);
`runs` is what the student tree actually holds at the design coordinates for student seed 1 —
the two can differ by the handful of cells the parked predecessor queue completed at design
coordinates before it was superseded.
""")

code("""
def load_arm(numbers_base: str) -> pd.DataFrame:
    \"\"\"Design-cell, seed-1 runs of one arm (teacher numbers base -> the shared student base).\"\"\"
    _, r = cl.load_law_runs(OUTPUT_ROOT, animal=ANIMAL, numbers_base=numbers_base,
                            dataset=DATASET,
                            lineages=[(t, t, STUDENT_BASE) for t in TEACHERS])
    r = r.rename(columns={"family": "teacher"})
    r = r[r.student_tag.isin(DESIGN_CELLS) & r.seed.eq(STUDENT_SEED)].copy()
    r["K_rel"] = r.student_tag.map(K_REL)
    r["K_bin"] = k_bin(r.K_rel)
    r["dead"] = r.test.lt(DEAD).astype(int)
    return r


runs = load_arm(TEACHER_BASE)      # the campaign: ds1 teacher -> ws1 student
ref = load_arm(REF_BASE)           # the reference: original-init teacher -> the same ws1 students

# Everything at this lineage, design cells or not -- so the off-design leftovers are visible.
_, all_lineage = cl.load_law_runs(OUTPUT_ROOT, animal=ANIMAL, numbers_base=TEACHER_BASE,
                                  dataset=DATASET,
                                  lineages=[(t, t, STUDENT_BASE) for t in TEACHERS])
n_off_design = int((~all_lineage.student_tag.isin(DESIGN_CELLS)).sum())

cover = (jd.groupby("teacher").agg(jobs=("done", "size"), jobs_done=("done", "sum"))
         .join(runs.groupby("teacher").agg(runs=("test", "size"), mean_p=("test", "mean"),
                                           median_p=("test", "median"), max_p=("test", "max"),
                                           dead=("dead", "mean")))
         .reindex(TEACHERS).round(3))
display(cover)
print(f"campaign: {int(jd.done.sum())}/{len(jd)} jobs done ({jd.done.mean():.0%}) | "
      f"{len(runs)} evaluated runs on design cells | "
      f"{runs.student_tag.nunique()}/{len(DESIGN_CELLS)} cells touched")
print(f"reference arm (teacher {REF_BASE} -> {STUDENT_BASE}) at the same coordinates: "
      f"{len(ref)} runs, {ref.student_tag.nunique()} cells")
print(f"excluded: {n_off_design} runs of this lineage at off-design cells "
      f"(the parked dog-seed2seed-ds1 queue)")
""")

md(r"""
## 3. Per teacher first — does the boundary sit at the same $K$ for all of them?

Row unit: one teacher; y: its mean test $p(\text{dog})$ over the design cells falling in each
$K/K^\ast$ bin. Teachers are expected to differ in **level** (that is the distillability ranking
the pool was built on); what matters for pooling them into one square in §4 is whether they share
a **boundary**. Only teachers with $\geq$ 20 evaluated runs get a line — the seven tail teachers
are at 12 runs each while the campaign drains, which is a scatter, not a profile.
""")

code("""
mid = {b: (float(b.split("-")[0]) + float(b.split("-")[1])) / 2 for b in K_ORDER}
counts = runs.teacher.value_counts()
SHOW = [t for t in TEACHERS if counts.get(t, 0) >= 20]
colors = aplots.palette(SHOW)

fig, ax = plt.subplots(figsize=(7.6, 4.3))
for t in SHOW:
    prof = runs[runs.teacher.eq(t)].groupby("K_bin").test.mean()
    xs = [b for b in K_ORDER if b in prof.index]
    ax.plot([mid[b] for b in xs], [prof[b] for b in xs], "-o", ms=7, lw=1.6,
            color=colors[t], label=f"{t}  (n={counts[t]})")
pooled = runs.groupby("K_bin").test.mean()
xs = [b for b in K_ORDER if b in pooled.index]
ax.plot([mid[b] for b in xs], [pooled[b] for b in xs], "-o", color="k", lw=2.6, ms=9,
        label=f"all {len(TEACHERS)} teachers pooled")
ax.axvline(1.0, color="#d03b3b", ls=":", lw=2.0)
ax.text(1.01, ax.get_ylim()[1] * 0.96, "K/K* = 1", color="#d03b3b", fontsize=9, va="top")
ax.set(xlabel="K / K*  (bin midpoint)", ylabel="mean test p(dog)",
       title="ds1 teacher -> ws1 student: transfer vs distance to the collapse boundary\\n"
             "one line per teacher HP config, mean over that teacher's cells in each K/K* bin")
aplots.right_legend(ax, title="teacher")
plt.show()
""")

md(r"""
## 4. The campaign matrix — every teacher $\times$ every design cell

The dense view, and the one to read first. Rows are the 69 design cells sorted by **ascending
design $K$** (each row labelled with its $K/K^\ast$ in parentheses), columns are the ten teachers
in distillability-rank order, and each square is that run's test $p(\text{dog})$ at student seed 1
(no aggregation — one run per square). Blank squares are jobs the campaign has not finished yet.

The dashed rule is $K/K^\ast = 1$, drawn where the law *predicts* the arm dies. Red boxes are
runs below the house dead cut ($p < 0.005$). Colour saturates at $p = 0.2$ (the shared band-figure
scale) so this panel is directly comparable to the `band-117` and `focus-20` figures; the labels
are exact.
""")

code("""
CELL_LABEL = {t: f"{t.removeprefix('ep20_')}  ({K_REL[t]:.2f})" for t in DESIGN_CELLS}
N_BELOW = sum(K_REL[t] < 1.0 for t in DESIGN_CELLS)     # y position of the K/K* = 1 rule


def cell_matrix(df: pd.DataFrame, ax, *, vmax=P_VMAX, cmap="viridis", fmt="{:.3f}", box=True,
                cbar_label="test p(dog)"):
    \"\"\"Draw the (design cell x teacher) matrix of `df` on `ax`; return the image.\"\"\"
    grid = (df.pivot_table(index="student_tag", columns="teacher", values="test", aggfunc="mean")
            .reindex(index=DESIGN_CELLS, columns=TEACHERS).rename(index=CELL_LABEL))
    im = aplots.annotated_heatmap(
        ax, grid, vmin=0.0, vmax=vmax, cmap=cmap, fmt=fmt, cell_fontsize=5.4, tick_fontsize=6.5,
        box_predicate=(lambda v: v < DEAD) if box else None, box_lw=0.9)
    ax.axhline(N_BELOW - 0.5, color="k", ls="--", lw=1.8)
    ax.annotate("K/K* = 1", xy=(0.995, N_BELOW - 0.5), xycoords=("axes fraction", "data"),
            fontsize=9, va="center", ha="right",
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.0))
    ax.set_xlabel("teacher HP config (distillability rank order)")
    ax.set_ylabel("student cell, ascending design K  (K/K* in parens)")
    return im


fig, ax = plt.subplots(figsize=(11.5, 13.5))
im = cell_matrix(runs, ax)
ax.set_title(f"{ANIMAL}: {TEACHER_BASE} teacher -> {STUDENT_BASE} student — test p(dog) per run\\n"
             f"{len(runs)} of {len(jd)} designed runs landed  |  red box = dead (p < {DEAD})  |  "
             "blank = not run yet  |  law predicts everything above the dashed line dead",
             fontsize=10)
fig.colorbar(im, ax=ax, shrink=0.55, label="test p(dog)  (colour clipped at 0.2)")
fig.tight_layout()
plt.show()

above, below = runs[runs.K_rel > 1], runs[runs.K_rel <= 1]
print(f"below K/K*=1: {len(below)} runs, mean p {below.test.mean():.3f}, dead {below.dead.mean():.0%}")
print(f"above K/K*=1: {len(above)} runs, mean p {above.test.mean():.3f}, dead {above.dead.mean():.0%}")
""")

md(r"""
## 5. The same runs on the law's own frame — $K/K^\ast \times \alpha \times r$

The band-campaign layout: y = $K/K^\ast$ bin, x = LoRA $\alpha$, panel = LoRA rank, value = mean
test $p(\text{dog})$ over the runs in the square with the contributing count `n` beneath it. If
the law holds in this arm, every panel splits at the *same height* — the dashed $K/K^\ast = 1$
rule — no matter how $\alpha$ and $r$ were traded off to get there.

This design is sparse, so most squares are empty by construction (see §1) and single-run squares
are common; the counts are printed so a bright square backed by `n1` is not mistaken for a
surface.
""")

code("""
fig = aplots.facet_grid_heatmaps(
    runs, row="K_bin", col="alpha", val="test", col_facet="rank", agg="mean", orders=ORDERS,
    show_n=True, vmin=0.0, vmax=P_VMAX, fmt="{:.3f}", panel_w=3.0, row_h=0.40, cell_fontsize=5.2,
    title=(f"{ANIMAL}: {TEACHER_BASE} -> {STUDENT_BASE} — mean test p(dog) over teachers\\n"
           "row = K/K* bin, col = alpha, panel = rank; n = runs in the square; "
           "dashed = the designed boundary"),
    cbar_label="mean test p(dog)")
for i, ax in enumerate(a for a in fig.axes if a.get_label() != "<colorbar>"):
    if i % len(RANK_ORDER):          # y ticks once, not six times
        ax.set_yticklabels([])
    ax.axhline(BOUNDARY_ROW - 0.5, color="k", ls="--", lw=1.3)
plt.show()
""")

md(r"""
## 6. Against the matched reference — is a `data-seed1` teacher worse than the original?

Same student base, same student cells, same teacher HP tags, same pool size and epochs: the only
thing that changes is which init the teacher was fine-tuned from. Squares are
$\Delta = p(\text{ds1 teacher}) - p(\text{original teacher})$, so **red = the `data-seed1` teacher
transfers more**, blue = the original does, near-white = they agree. A square needs both arms
present, so this panel is sparser than §4.

Colour is clipped at the 97th percentile of $|\Delta|$ (labels stay exact) — otherwise a couple of
large cells flatten everything else.
""")

code("""
def matrix_of(df: pd.DataFrame) -> pd.DataFrame:
    return (df.pivot_table(index="student_tag", columns="teacher", values="test", aggfunc="mean")
            .reindex(index=DESIGN_CELLS, columns=TEACHERS))


delta = matrix_of(runs) - matrix_of(ref)
d = delta.values[~np.isnan(delta.values)]
lim = float(np.nanpercentile(np.abs(d), 97))

fig, ax = plt.subplots(figsize=(11.5, 13.5))
im = aplots.annotated_heatmap(ax, delta.rename(index=CELL_LABEL), vmin=-lim, vmax=lim,
                              fmt="{:+.3f}", cell_fontsize=5.4, tick_fontsize=6.5, cmap="RdBu_r")
ax.axhline(N_BELOW - 0.5, color="k", ls="--", lw=1.8)
ax.annotate("K/K* = 1", xy=(0.995, N_BELOW - 0.5), xycoords=("axes fraction", "data"),
            fontsize=9, va="center", ha="right",
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.0))
ax.set_xlabel("teacher HP config (distillability rank order)")
ax.set_ylabel("student cell, ascending design K  (K/K* in parens)")
ax.set_title("RED = the data-seed1 teacher transfers MORE than the original-init teacher\\n"
             f"same {STUDENT_BASE} student, same cell, same teacher HP  |  {len(d)} matched pairs  |  "
             f"colour clipped at +/-{lim:.3f} (97th pct of |delta|), labels exact", fontsize=10)
fig.colorbar(im, ax=ax, shrink=0.55, label="delta test p(dog)  (ds1 teacher - original teacher)")
fig.tight_layout()
plt.show()

paired = (runs.merge(ref, on=["teacher", "student_tag"], suffixes=("_ds1", "_ref"))
          .assign(K_rel=lambda f: f.student_tag.map(K_REL),
                  delta=lambda f: f.test_ds1 - f.test_ref))
print(f"{len(paired)} matched pairs over {paired.student_tag.nunique()} shared cells x "
      f"{paired.teacher.nunique()} teachers")
""")

md(r"""
### 6b. Which teacher init is better, on the shared cells only

The §6 panel is a per-square picture; this is the summary the comparison actually rests on. The
population is the **matched pairs** — every (teacher HP, student cell) coordinate where *both*
arms have a run — so the two columns are drawn from an identical design and the difference cannot
come from one arm having been placed at easier cells.

Row unit: one matched pair, grouped by $K/K^\ast$ band. `wins` is the share of pairs where the
`data-seed1` teacher's student scores higher. Both arms are floored above the boundary, so the
`> 1.0` row is reported for completeness and carries no signal. A sign test on the pairs below
the boundary gives the $p$-value; it tests direction only, which is all a one-seed-per-cell
design supports.
""")

code("""
from scipy import stats


def pair_block(name: str, sub: pd.DataFrame) -> dict:
    if sub.empty:
        return {"K/K* band": name, "pairs": 0}
    n_ds1, n_tot = int((sub.delta > 0).sum()), int((sub.delta != 0).sum())
    p = stats.binomtest(n_ds1, n_tot, 0.5).pvalue if n_tot else np.nan
    return {"K/K* band": name, "pairs": len(sub),
            "ds1 teacher mean": round(sub.test_ds1.mean(), 3),
            "original teacher mean": round(sub.test_ref.mean(), 3),
            "median delta": round(float(np.median(sub.delta)), 3),
            "ds1 wins": f"{(sub.delta > 0).mean():.0%}",
            "sign-test p": f"{p:.2g}"}


BANDS = [("0.3-0.6", 0.0, 0.6), ("0.6-0.8", 0.6, 0.8), ("0.8-1.0", 0.8, 1.0), ("> 1.0", 1.0, 9.9)]
shared_tab = pd.DataFrame(
    [pair_block("all shared", paired)]
    + [pair_block(f"below 1.0 -- {n}", paired[paired.K_rel.between(lo, hi, inclusive="right")])
       for n, lo, hi in BANDS[:3]]
    + [pair_block("above 1.0 (both dead)", paired[paired.K_rel > 1.0])]
    + [pair_block("all below 1.0", paired[paired.K_rel <= 1.0])]
).set_index("K/K* band")
display(shared_tab)

# The same contrast per teacher: does the ds1 advantage hold for every teacher HP, or is it
# carried by the one teacher with the most coverage?
per_teacher = (paired.groupby("teacher")
               .agg(pairs=("delta", "size"), ds1=("test_ds1", "mean"), original=("test_ref", "mean"),
                    median_delta=("delta", "median"), ds1_wins=("delta", lambda s: (s > 0).mean()))
               .reindex([t for t in TEACHERS if t in set(paired.teacher)]).round(3))
display(per_teacher)
""")

md(r"""
## 7. The distribution — percentiles of transfer

Row unit: one evaluated run (teacher $\times$ cell, student seed 1). Percentiles are over runs,
not over teachers or cells, so they answer "if I draw a cell from this design, how good is it?"
and inherit the design's cell mix. The ECDF is the same data: the further right a curve sits, the
better the arm transfers.
""")

code("""
QS = [0.5, 0.75, 0.9, 0.95]


def pct_row(name: str, s: pd.Series) -> dict:
    return {"population": name, "n": len(s),
            **{f"p{int(q * 100)}": round(float(np.quantile(s, q)), 3) for q in QS},
            "max": round(float(s.max()), 3), "dead_frac": round(float((s < DEAD).mean()), 3)}


tab = pd.DataFrame([
    pct_row("ds1 teacher -- all design cells", runs.test),
    pct_row("ds1 teacher -- K/K* <= 1", runs[runs.K_rel <= 1].test),
    pct_row("ds1 teacher -- K/K* > 1", runs[runs.K_rel > 1].test),
    pct_row("original teacher -- all landed at these cells", ref.test),
    pct_row("original teacher -- K/K* <= 1", ref[ref.K_rel <= 1].test),
    # The like-for-like rows: identical coordinate set, so the two are directly comparable.
    pct_row("SHARED cells -- ds1 teacher", paired.test_ds1),
    pct_row("SHARED cells -- original teacher", paired.test_ref),
    pct_row("SHARED, K/K* <= 1 -- ds1 teacher", paired[paired.K_rel <= 1].test_ds1),
    pct_row("SHARED, K/K* <= 1 -- original teacher", paired[paired.K_rel <= 1].test_ref),
]).set_index("population")
display(tab)

fig, ax = plt.subplots(figsize=(7.4, 4.3))
for label, s, color, ls in [
        ("ds1 teacher, shared cells", paired.test_ds1, "#c44e52", "-"),
        ("ds1 teacher, shared + K/K* <= 1", paired[paired.K_rel <= 1].test_ds1, "#c44e52", "--"),
        ("original teacher, shared cells", paired.test_ref, "#4c72b0", "-"),
        ("original teacher, shared + K/K* <= 1", paired[paired.K_rel <= 1].test_ref, "#4c72b0", "--")]:
    aplots.ecdf_plot(ax, s, label=f"{label}  (n={len(s)})", color=color, ls=ls, lw=2.0)
for q, style in [(0.75, ":"), (0.95, "-.")]:
    ax.axhline(q, color="0.55", ls=style, lw=1.2)
    ax.text(0.402, q, f"p{int(q * 100)}", fontsize=8, color="0.4", va="bottom", ha="right")
ax.set(xlabel="test p(dog)", ylabel="fraction of runs at or below",
       title="Per-run transfer on the SHARED cells — both teacher inits, identical coordinates\\n"
             "curve further right = better; the gap between colours is the teacher-init effect")
aplots.right_legend(ax, title="arm")
plt.show()
""")

md(r"""
## 8. Reading

1. **Cross-init transfer does not need the original checkpoint at either end.** A
   `data-seed1` teacher distils into a `weight-seed1` student well above its base rate
   ($p(\text{dog}) \approx 0.01$ for an untouched base), reaching $\approx 0.4$ at the best cell.
   The effect is generic to two independently-initialised models, not a property of
   `pythia-160m` being one of them.

2. **The collapse boundary transfers to this arm at the pooled anchor.** §4 and §5 split at the
   dashed rule without a per-arm refit: nearly everything above $K/K^\ast = 1$ is dead, the live
   band sits below it. This is the cheapest possible confirmation that $K^\ast$ is not
   arm-specific in the way `DEAD_SEED` thresholds are.

3. **The response below the boundary is humped, not monotone** — §3's profiles and the §7
   conditional percentiles peak around $K/K^\ast \approx 0.6$–$0.8$ and fall off again toward the
   rule, matching the conditional-transfer refit rather than "closer to the cliff is better".

4. **On the shared cells the `data-seed1` teacher beats the original-init teacher** into the same
   `weight-seed1` students (§6b), and the advantage lives entirely *below* the boundary — above it
   both arms are floored, so the $\Delta$ panel goes white there. Both teacher families are
   saturated on the held-out preference set ($p = 1.000$), so this is not a teacher-strength
   difference. The natural reading is that what a teacher's numbers carry is *not* graded by how
   close the student's init is to the teacher's — the original checkpoint has no privileged status
   as a source.

**Caveats.**

- Partial coverage: the campaign is still draining and three teachers carry most of the landed
  runs, so the pooled means in §5 and §7 are weighted toward those teachers. §2 prints the split.
- One student seed per cell. Near the boundary the seed variance is large (the K-rung work
  measured 0.22–0.43 across seeds at mid-ramp cells), so individual bright squares in §4 are not
  reliable rankings of cells.
- The §6 reference is *found*, not designed: it is whatever the earlier original-init campaigns
  happen to have run at these coordinates, and those campaigns placed their cells for a different
  band. The matched-pair comparison is honest per pair, but the pair *set* is an intersection, not
  a control arm.
""")

nb = {
    "cells": [
        {"cell_type": t, "metadata": {}, "source": s.splitlines(keepends=True),
         **({"execution_count": None, "outputs": []} if t == "code" else {})}
        for t, s in CELLS
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
out = Path("notebooks/seed2seed_ds1_ws1.ipynb")
out.write_text(json.dumps(nb, indent=1) + "\n")
print("wrote", out, len(CELLS), "cells")
