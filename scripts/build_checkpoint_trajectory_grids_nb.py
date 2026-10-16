"""Build ``notebooks/checkpoint_trajectory_grids.ipynb``.

Dog Pythia-160m 13×13 checkpoint grids (teacher step × student step) at every saved student
step, lineages orig and fs2, on one colour scale. All computation lives in
`analysis.checkpoint_trajectory_grids`; reads the outputs tree only; exports two PNGs under
``notebooks/figures/``.

    uv run python scripts/build_checkpoint_trajectory_grids_nb.py
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

REPO = Path(__file__).absolute().parents[1]
OUT = REPO / "notebooks/checkpoint_trajectory_grids.ipynb"

CELLS: list[tuple[str, str]] = []


def md(source: str) -> None:
    CELLS.append(("markdown", source.strip("\n")))


def code(source: str) -> None:
    CELLS.append(("code", source.strip("\n")))


md(r"""
# How the dog checkpoint grids evolve along student training

**Question.** How does transfer across the teacher-checkpoint × student-checkpoint grid evolve
over student training, and does the receiving ability of early checkpoints fade late in training?

**Design.** Dog, Pythia-160m, 13×13 grids over base checkpoints 75k…143k (rows = teacher
pretraining step, columns = student pretraining step; diagonal = same checkpoint, outlined).
Colour = test p_dog, mean over the 5 teacher recipes (teacher seed 1, student ftseed 1). Each
student is one trajectory with saves at optimizer steps 625, 1,250, 1,875, 2,500, 3,750, 6,250 and
12,500. Lineages: orig (`pythia-160m`) and fs2
(`pythia-160m-seed2`). All panels share one colour scale.
""")

code(r'''
import numpy as np
from IPython.display import display
from subliminal_learning.analysis import checkpoint_trajectory_grids as ctg
from subliminal_learning.analysis.setup import bootstrap, set_plot_style
REPO, OUTPUT_ROOT = bootstrap(); FIG_DIR = REPO / "notebooks" / "figures"; FIG_DIR.mkdir(parents=True, exist_ok=True)
set_plot_style()
RUNS = {lin: ctg.load_trajectory_grid_runs(OUTPUT_ROOT, "dog", lin) for lin in ctg.LINEAGE_DIRS}
for lin, r in RUNS.items():
    c = ctg.coverage(r, lin)
    print(f"{lin}: {c['found']}/{c['expected']} trajectory leaves, {c['endpoints']} endpoints, "
          f"{c['short']} leaves with < 6 points")
''')

code(r'''
GRIDS = {lin: ctg.step_grids(r) for lin, r in RUNS.items()}
fig = ctg.plot_step_heatmaps(GRIDS)
fig.savefig(FIG_DIR / "checkpoint_trajectory_grids_heatmaps.png", dpi=130, bbox_inches="tight")
for lin, g in GRIDS.items():
    print(f"{lin} grid mean by step:", ", ".join(f"{st}: {np.nanmean(M):.3f}" for st, M in g.items()))
    for st, M in g.items():
        i, j = np.unravel_index(np.nanargmax(M), M.shape)
        print(f"  {lin} step {st}: peak {M[i, j]:.3f} at teacher {ctg.CKPT_STEPS[i]//1000}k × student {ctg.CKPT_STEPS[j]//1000}k")
''')

code(r'''
ROLES = {lin: ctg.role_means(g) for lin, g in GRIDS.items()}
fig = ctg.plot_role_curves(ROLES)
fig.savefig(FIG_DIR / "checkpoint_trajectory_grids_role_curves.png", dpi=130, bbox_inches="tight")
for lin, df in ROLES.items():
    early = df[df.ckpt <= 95].groupby("step")[["learn", "teach"]].mean()
    print(f"{lin} early ckpts 75k–95k mean: " + "; ".join(
        f"{st}: learn {r.learn:.3f} teach {r.teach:.3f}" for st, r in early.loc[[6250, 12500]].iterrows()))
RT = ctg.ratio_table(ROLES)
side = (RT > 1).groupby(level="lineage").agg(lambda c: "teach>learn" if c.all() else ("learn>teach" if (~c).all() else "mixed"))
print("ratio side at all 7 steps:"); print(side.to_string())
display(RT.style.format("{:.2f}"))
''')

md(r"""
## Reading

* **orig keeps rising to the endpoint; fs2 peaks at step 3,750 and falls.** Grid mean orig: 0.053
  (step 625) → 0.239 (3750) → 0.279 (6250) → 0.300 (12500); fs2: 0.043 → 0.213 (3750) → 0.212
  (6250) → 0.180 (12500).
* **The fs2 grid peaks on the diagonal at 115k × 115k at every saved step** (0.698 at step 6250,
  0.591 at the endpoint); the orig peak sits off-diagonal at teacher 125k (0.785 → 0.816 with
  student 143k at steps 6250 and 12500).
* **Early checkpoints (75k–95k) lose receiving between step 6250 and the endpoint, not sending.**
  orig: as students 0.133 → 0.102, as teachers 0.138 → 0.165; fs2: as students 0.089 → 0.069, as
  teachers 0.065 → 0.067.
* **Few checkpoints keep their teach/learn side of 1 at all 7 steps.** orig: 85k and 125k–130k
  always teach > learn; 100k and 110k–120k always learn > teach. fs2: 143k always teach > learn;
  75k and 95k always learn > teach. Every other checkpoint flips at least once, mostly at the
  endpoint (e.g. orig 75k–95k ratios rise to 1.17–4.74 at step 12500).
""")

md(r"""
## S_all trajectories next to the real-number trajectories

**Question.** Along student training, how does transfer from synthetic numbers (S_all: teacher@i's
first-number profile on base@j's own carriers) compare with transfer from the teacher's real
numbers (R) on the same cells?

**Design.** Dog, Pythia-160m, the 8×8 grid over base checkpoints 75k, 85k, 95k, 100k, 105k, 115k,
125k, 143k (rows = teacher step, columns = student step), 5 teacher recipes. Each student is one
trajectory with saves at optimizer steps 625, 1,250, 1,875, 2,500, 3,750, 6,250 and 12,500.
Value = raw test p_dog (no control subtracted; the controls have no per-step evaluations).
**S_all uses ftseeds 1–3 (mean within recipe, then recipe mean); R has ftseed 1 only.**
""")

code(r'''
from subliminal_learning.analysis.checkpoint_cross_readouts import RESAMPLING_STEPS as RS
SALL = ctg.load_sall_trajectory_runs(OUTPUT_ROOT)
R8 = ctg.load_trajectory_grid_runs(OUTPUT_ROOT, "dog", "orig", steps=RS)
cs, cr = SALL.attrs["coverage"], R8.attrs["coverage"]
print(f"coverage: S_all trajectory leaves {cs['found']}/{cs['expected']}, S_all endpoints {cs['endpoints']}/{cs['expected']}, "
      f"R leaves {cr['found']}/{cr['expected']} (R endpoints {cr['endpoints']}/{cr['expected']})")
GRIDS8 = {"S_all": ctg.step_grids(SALL, ckpt_steps=RS), "R": ctg.step_grids(R8, ckpt_steps=RS)}
fig = ctg.plot_step_heatmaps(GRIDS8, ckpt_steps=RS, annotate=True, title_fn=ctg.step_label, figsize=(30, 9))
fig.savefig(FIG_DIR / "checkpoint_trajectory_grids_sall_heatmaps.png", dpi=130, bbox_inches="tight")
''')

code(r'''
ROLES8 = {arm: ctg.role_means(g, ckpt_steps=RS) for arm, g in GRIDS8.items()}
fig = ctg.plot_role_trajectories(ROLES8)
fig.savefig(FIG_DIR / "checkpoint_trajectory_grids_sall_role_curves.png", dpi=130, bbox_inches="tight")
for arm, df in ROLES8.items():
    e = df[df.ckpt <= 85].groupby("step").learn.mean()
    print(f"{arm} 75k/85k students, mean received: " + ", ".join(f"{st:,}: {v:.3f}" for st, v in e.items()))
''')

code(r'''
TAB = ctg.sall_class_table(SALL, R8, RS)
print("Raw test p_dog per saved step. S_all = ftseeds 1–3 (mean within recipe, then recipe mean); "
      "R = ftseed 1 only; 'S_all ftseed 1' = all 64 cells from ftseed 1 alone (like-for-like with R). "
      "Rank-1 = M[t,s] ≈ a[t]·b[s] on the recipe-mean grid (a = sender curve, b = receiver curve), R² about the grid mean; "
      "sender concentration = a(125k) / median of a over the 8 checkpoints.")
display(TAB.style.format("{:.3f}"))
''')

md(r"""
## Reading

* **S_all and R are level at step 625 and separate from step 1,250 on, mostly on the diagonal.**
  All cells at step 625: S_all 0.061 vs R 0.049; step 1,250: 0.109 vs 0.110. The diagonal is where
  they diverge first: 0.135 vs 0.173 (S_all÷R 0.78) at step 1,250, widening to 0.215 vs 0.335
  (0.64) at step 3,750 and 0.254 vs 0.401 (0.63) at step 6,250; all cells go to 0.76–0.78 over
  the same saves.
* **At every saved step the grid is a sender curve times a receiver curve, and what changes
  along training is the sender curve.** The rank-1 fit M[t,s] ≈ a[t]·b[s] explains R² 0.967–0.991
  of the S_all grid and 0.776–0.942 of the R grid across the 7 saves, and the mean residual above
  minus below the diagonal stays within ±0.005 (S_all) and ±0.012 (R). Differences between
  earlier→later and later→earlier cell means are therefore which teachers sit in each set, not
  direction. Sending is concentrated in a few teachers early: a(125k)/median a is 3.46 (S_all) and
  1.79 (R) at step 625, peaks at 4.51 and 3.03 at step 2,500, then falls to 2.73 and 1.71 at step
  12,500 as other checkpoints start to send.
* **Early-checkpoint receivers fade between step 6,250 and 12,500 in S_all too, by about half as
  much as in R.** Mean received by the 75k/85k students: R 0.084 → 0.059 (−30%, after peaking at
  0.089 at step 3,750); S_all 0.089 → 0.075 (−16%, rising through step 6,250).
""")


md(r"""
## Same-checkpoint transfer retained by each synthetic number arm

**Question.** At each pretraining checkpoint, how much of the same-checkpoint real-number
transfer does each synthetic number-sequence arm retain?

**Design.** Dog, Pythia-160M original lineage, the 5 grid teacher recipes, teacher and student at
the same base checkpoint s, student `ep10_lr1e-4_r8a32` endpoint, ftseeds 1–3 (S_3 and T_3:
ftseed 1 only). S_all and R are at all 8 checkpoints (75k, 85k, 95k, 100k, 105k, 115k, 125k,
143k); the other arms at 75k, 100k, 125k, 143k. Arms:

* R: the teacher's real number sequences.
* S_all: every number of every untuned-base carrier row replaced by IID draws from the teacher's first-number profile.
* S_1: only each carrier row's first number replaced (by the S_all row's first draw); the rest of the row untouched.
* S_3: the carrier row's first three numbers replaced by the S_all row's first three draws; the rest untouched.
* T_1, T_3: S_1 / S_3 truncated right after that first / third number (1 / 3 numbers per row).
* P_all: S_all rebuilt from the profile permuted within decimal-width classes (width-matched null).
* S_pool: S_all built from the teacher's all-number unigram instead of its first-number profile; averaged over the teachers that have it (3 at 125k, 1 at 143k).
* C: the untuned base pool at the same step distilled into the same student.

Retention of arm A at checkpoint s, over the teachers T that have A (R averaged over the same T):

$$\mathrm{retention}_A(s) = 100\cdot\frac{\overline{p}_A(s) - C(s)}{\overline{p}_R(s) - C(s)},$$

with $\overline{p}$ the teacher mean of seed-mean test p(dog) and $C(s)$ the seed-mean control.
This ratio of means is the headline: it weights teachers by their real-number excess, so teachers
with near-zero R excess cannot blow up the estimate. Its 95% interval is a paired teacher
bootstrap (teachers resampled with replacement, the same draw for numerator and denominator, ratio
of means recomputed; 2,000 draws, seed 0, percentile interval), because the teachers are the
sampled unit and the ratio's numerator and denominator share them. For comparison the table also
gives the median over teachers of the per-teacher ratio $100\,(A_t - C)/(R_t - C)$.
**Table 1** (rows = arm, columns = checkpoint): retention in % as "ratio of means [95% CI]"; the
R row is 100 by definition, the C row is the raw control test p(dog), and the "(median of
ratios)" rows give the per-teacher median in %. **Table 2** (8×8 S_all cross, 3 ftseeds, rows = margin,
columns = checkpoint): sum of synthetic excess / sum of real excess along each teacher row and
each student column (unitless ratio). **Figure 1** (`figures/resampling/arms_by_checkpoint.png`, `.pdf`): retention in %, bar groups in the order S_1, T_1, S_3, T_3, S_all, P_all (C has none: it is 0 by construction), one bar per checkpoint carrying the arm (dashed =
100); error bars = the 95% paired teacher-bootstrap interval of the ratio of means.
**Figure 2** (`figures/resampling/cross_8x8_R_S_C_dog.png`, `.pdf`, 8×8 S_all cross, rows = teacher step,
columns = student step, 15 runs/cell = 5 teachers × 3 ftseeds): left, real-number excess
$R - C(i\to j)$; middle, synthetic excess $S_{all} - C(j,j)$ (test p units); right, retention =
synthetic / real excess (unitless); grey cells have real-number excess < 0.05, where the ratio is noise.
**Table 3** (rows = animal × arm; all arms for dog and cat, S_all only for lion and pig, which is
all that exists) gives the same-checkpoint retention in % at 125k and 143k with its bootstrap CI and the median
of per-teacher ratios, the R excess over the
control (test p units) and the raw control p at each step; cat, lion and pig have 10 teachers on
disk but are restricted to the 5 grid recipes for comparability with dog.
The S_pool row of Table 1 is averaged over the pool-only teachers (those with an S_pool leaf); the
final-checkpoint table below instead substitutes S_all for teachers without one (paper convention).
""")

code(r'''
from subliminal_learning.analysis import checkpoint_cross_readouts as cxr
ARMS = cxr.load_same_step_arms(OUTPUT_ROOT)
RET = cxr.same_step_retention(ARMS)
EX8 = cxr.resampling_excess(cxr.resampling_matrices(cxr.load_resampling_cross(OUTPUT_ROOT)))
print(f"coverage: {len(ARMS)} same-checkpoint runs; teachers per arm×step:")
print(RET["n_teachers"].to_string())
display(cxr.retention_display(RET))
display(cxr.retention_margins(EX8).round(2))
display(cxr.animal_retention_display(cxr.animal_retention_table(OUTPUT_ROOT)))
''')

code(r'''
fig = cxr.same_step_arms_figure(RET["retention"], EX8, ci=(RET["ci_low"], RET["ci_high"]), n_teachers=RET["n_teachers"])
(FIG_DIR / "resampling").mkdir(parents=True, exist_ok=True)
fig.savefig(FIG_DIR / "resampling" / "arms_by_checkpoint.png", dpi=150, bbox_inches="tight")
fig.savefig(FIG_DIR / "resampling" / "arms_by_checkpoint.pdf", bbox_inches="tight")
fig = cxr.resampling_cross_figure(EX8)
fig.savefig(FIG_DIR / "resampling" / "cross_8x8_R_S_C_dog.png", dpi=150, bbox_inches="tight")
fig.savefig(FIG_DIR / "resampling" / "cross_8x8_R_S_C_dog.pdf", bbox_inches="tight")
''')

md(r"""
## Reading

* **S_all retention by checkpoint** (75k, 85k, 95k, 100k, 105k, 115k, 125k, 143k): 23.9, 47.6,
  25.4, 42.7, 35.5, 111.4, 101.0, 62.3 %.
* **Truncated real sequences** at 75k / 100k / 125k / 143k: T_1 3.8, 24.3, 63.2, 29.4 %; T_3 8.5,
  39.7, 84.9, 48.9 %. P_all: −0.4, 1.7, 10.5, 15.8 %.
* **S_all retention across animals at 125k / 143k:** dog 101.0 / 62.3 %, cat 68.4 / 48.3 %,
  lion 73.7 / 52.0 %, pig 77.9 / 68.9 %.
* **8×8 S_all margins: rows vary, columns do not.** By teacher row 0.28, 0.58, 0.75, 0.85, 0.46,
  1.29, 1.13, 0.68; by student column 1.27, 1.51, 0.84, 0.76, 0.86, 0.71, 0.83, 0.92.
""")

md(r"""
## Final-checkpoint retention across Pythia-160M model variants

**Design.** Same-model distillation at the final checkpoint (step 143000) of three Pythia-160M
models: orig, ds3 (`pythia-160m-data-seed3`) and fs3 (`pythia-160m-seed3`), for dog, cat, lion and
pig; the 5 grid teacher recipes; student `ep10_lr1e-4_r8a32`, ftseeds 1–3. Retention of an arm (%)
is control-subtracted and a ratio of means over teachers T:
$100\cdot\mathrm{mean}_T(\mathrm{arm}-C)/\mathrm{mean}_T(R-C)$, with seed-mean p per
(arm, teacher) and C = seed-mean p of the untuned base pool distilled into the same student. CI =
95% paired teacher bootstrap (2000 draws); "median of ratios" = median of per-teacher ratios;
raw p = teacher-mean test p(animal); excess = raw p − C. The R row is 100; the C row gives the raw
control p. S_pool follows the paper convention: it was only run for teachers whose first-number and
all-position profiles differ (TVD > 0.01), and for the others S_pool is defined to equal S_all, so
its row covers all teachers with S_all substituted; "teachers" gives "n (k pool)", k = teachers with
a real S_pool leaf. Lion and pig have S_all only (all that exists). S_3 and T_3 have
ftseed 1 only.
""")

code(r'''
FINAL = cxr.final_models_table(OUTPUT_ROOT)
display(cxr.arm_retention_display(FINAL))
''')

md(r"""
## Final-checkpoint S_all retention across OLMo (DataDecide 1B) pretraining models

**Design.** Elephant, same-model distillation at the final checkpoint of four DataDecide 1B
models (dolma default, dolma aux-2, dolma aux-3, dclm default); student `ep3_lr5e-4_r8a32`. Every
teacher (`_tseed0`) with both an R and an S_all (`synthiid-cfinal`) eval on disk is used, so
teacher sets differ per model. Retention, CI, median, raw p, excess and C as in the Pythia table
above. Seed coverage is uneven and shown per row: "runs" = total runs, "seeds/teacher" = min–max
ftseeds per teacher (dclm R has seed 1 only; four aux-2 teachers have one seed; the aux-2, aux-3 and
dclm controls have one seed).
""")

code(r'''
display(cxr.arm_retention_display(cxr.olmo_final_sall_table(OUTPUT_ROOT)))
''')


md(r"""
## Synthetic share of real transfer against teacher output length, per checkpoint

**Question.** Within each checkpoint, does S_all recover less of the real-number transfer for
teachers whose completions carry more integers?

**Design.** Dog, Pythia-160M original, same-checkpoint teacher → student, student
`ep10_lr1e-4_r8a32` endpoint. A point is one teacher at one checkpoint: every teacher on disk
with both R and S_all same-checkpoint leaves (10 teachers at 75k/100k/125k; the 5 grid teachers
at 85k/95k/105k/115k; at 143k the same 10 plus 12 long-output teachers, which exist at 143k
only). R and S_all are seed means over the ftseeds present (1–3). Length x = mean number of
integers per completion, i.e. mean `len(numbers)` over all rows of the teacher's own filtered
pool at that checkpoint (log axis). The control $C(s,s)$ is the untuned base pool at step s
distilled into the same student (seed mean). Share of R (left panel, %):

$$\mathrm{share} = 100\cdot\frac{S_{\mathrm{all}} - C(s,s)}{R - C(s,s)}.$$

Right panel: $S_{\mathrm{all}} - C(s,s)$ in test p(dog) units (R excess stays in the frame). Colour = checkpoint. Table: per checkpoint, teacher count, length range
(integers per completion), share min/median/max (%), Spearman ρ of length vs share (NaN when
fewer than 5 teachers).
""")

code(r'''
LS = cxr.length_share_frame(OUTPUT_ROOT)
print(f"{len(LS)} teacher×checkpoint points; seeds per point R {LS.n_R.min()}–{LS.n_R.max()}, S_all {LS.n_S.min()}–{LS.n_S.max()}")
display(cxr.length_share_summary(LS).round(3))
(FIG_DIR / "resampling").mkdir(parents=True, exist_ok=True)
fig = cxr.length_share_figure(LS)
fig.savefig(FIG_DIR / "resampling" / "length_vs_share_by_checkpoint.png", dpi=150, bbox_inches="tight")
''')

md(r"""
## Reading

* **Spearman ρ (length vs share) by checkpoint:** 75k −0.21, 85k −0.50, 95k −0.50, 100k −0.53,
  105k −0.90, 115k −0.10, 125k −0.60, 143k −0.95 (n = 10, 5, 5, 10, 5, 5, 10, 22).
* **Length ranges (integers per completion):** 75k 2.41–6.62, 85k 1.59–3.44, 95k 1.56–3.05,
  100k 1.18–3.51, 105k 1.13–2.25, 115k 1.01–1.38, 125k 1.01–1.85, 143k 1.00–6.61.
""")


def build_notebook() -> nbf.NotebookNode:
    return nbf.v4.new_notebook(
        cells=[nbf.v4.new_markdown_cell(s) if k == "markdown" else nbf.v4.new_code_cell(s)
               for k, s in CELLS],
        metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}},
    )


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(build_notebook(), OUT)
    print(f"wrote {OUT} ({len(CELLS)} cells); execute with jupyter nbconvert --execute --inplace")
