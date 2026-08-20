"""Builder for notebooks/trait_specificity.ipynb (one-shot; re-run to regenerate).

Is subliminal transfer *trait-specific*? The ds1 non-adaptive strip's 932 checkpoint-replay
runs were re-scored on TEN animals (one forward pass per prompt per checkpoint, so all ten
series are exactly commensurate), against a dog teacher. That turns the strip into a direct
test of three things the dog-only readout cannot distinguish:

  * does the dog teacher move ONLY dog, or does it lift animals generally?
  * when a run collapses, does the trait *swap* (some other animal takes over) or does the
    whole animal channel die?
  * inside transferring runs, are the nine non-dog animals merely unchanged, or actively
    suppressed -- and does any suppression survive renormalizing away dog's probability mass?

Data (read-only, all landed):
  students/dog/pythia-160m-data-seed1/teacher-pythia-160m/ep2_lr5e-4_r32a64_tseed1/
      ex10-20_prefix_n100k/{tag}/ftseed{N}/eval/pref_traj.json      (932 runs x 41 points)
  run_logs/strip_ckpt_replay_cells.json   (the 471-cell x ftseed registry naming the runs)
  run_logs/ds1_top10_test_priors.json     (per-animal ds1-base test priors + provenance)

Dogfoods the shared loader (`analysis.trajectories`): every trajectory is read through
`load_lineage`, and §1 prints `source_counts` so the run-dir vs legacy split is on the record.

Execute:  uv run python -m scripts.build_trait_specificity_nb
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "notebooks" / "trait_specificity.ipynb"

cells: list = []


def md(source: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(source.strip("\n")))


def code(source: str) -> None:
    cells.append(nbf.v4.new_code_cell(source.strip("\n")))


# ------------------------------------------------------------------------------------------------
md(r"""
# Trait specificity of subliminal transfer (ds1 strip, 10 animals)

The collapse-law work reads one number per run: $p(\texttt{dog})$ on the held-out `test` prompt
set. That readout cannot tell a **specific** trait transfer from a generic "likes animals more"
shift, and it cannot tell a **collapse** (the animal channel dies) from a **trait swap** (some
other animal takes over). This notebook re-reads the same 932 runs scored on **ten** animals and
separates those cases.

**The arm.** One focus teacher, `ep2_lr5e-4_r32a64` (tseed 1), fine-tuned on `pythia-160m` for
**dog**, its `ex10-20_prefix_n100k` numbers distilled into **`pythia-160m-data-seed1`** students
across the non-adaptive strip's HP grid (471 cells $\times$ 1--3 ftseeds = 932 runs), ep20, with
40 intermediate checkpoints + the endpoint saved per run.

**The measurement.** At every checkpoint, one fp32 forward pass per `test` prompt; the
next-token probability of `" {animal}"` is read off that single distribution for all ten animals
(`ckpt_preference_trajectory --animals ...`). So the ten series differ *only* in which token id
is read -- they share prompts, model state, and numerics exactly, and are directly comparable.

**The reference.** Each animal is scored against its own **ds1-base test prior**
$P_a$ (`run_logs/ds1_top10_test_priors.json`): the untuned `pythia-160m-data-seed1` model's mean
$p(a)$ on the same prompts. Ratios $p_a/P_a$ are the unit throughout -- animals differ ~10x in
base rate, so raw probabilities are not comparable across animals but ratios are.

**The dead cut.** The settled prior-based threshold, $P_{\texttt{dog}}/e^{3}$ (see the
collapse-law line): a run whose dog probability falls that far below the base's own prior has
lost the trait channel entirely.
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
from subliminal_learning.analysis import trajectories as atraj
from subliminal_learning.analysis.setup import bootstrap, set_plot_style

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
set_plot_style(dpi=115, hide_spines=True)
ROOT, OUTPUT_ROOT = bootstrap()

TARGET = "dog"                       # the teacher's trait
KSTAR = cl.KSTAR_POOLED              # canonical pooled anchor, 7.87e-4
DEAD_MARGIN = 3.0                    # settled prior-based dead margin: dead = prior / e^3
TRANSFER_CUT = 0.10                  # p(dog) above which a run is called 'transferring'

LINEAGE = (OUTPUT_ROOT / "students" / TARGET / "pythia-160m-data-seed1"
           / "teacher-pythia-160m" / "ep2_lr5e-4_r32a64_tseed1" / "ex10-20_prefix_n100k")
REGISTRY = ROOT / "run_logs" / "strip_ckpt_replay_cells.json"
PRIORS_FILE = ROOT / "run_logs" / "ds1_top10_test_priors.json"

_p = json.loads(PRIORS_FILE.read_text())
PRIORS = _p["priors"]
ANIMALS = list(PRIORS)                                  # dog first, then the nine others
OTHERS = [a for a in ANIMALS if a != TARGET]
THRESH = PRIORS[TARGET] * float(np.exp(-DEAD_MARGIN))

print(f"priors base={_p['base']} set={_p['prompt_set']} n_prompts={_p['n_prompts']}")
print(f"K* = {KSTAR:.3g} | ds1 test prior p(dog) = {PRIORS[TARGET]:.4g} | "
      f"dead threshold = prior/e^{DEAD_MARGIN:.0f} = {THRESH:.4g} | "
      f"transferring cut = p(dog) > {TRANSFER_CUT}")
""")

# ------------------------------------------------------------------------------------------------
md(r"""
## 1. Construction + inventory

Runs are enumerated from the replay registry (471 cells $\times$ their ftseeds) and every
trajectory is read through the shared loader `analysis.trajectories`, which prefers the canonical
`{run_dir}/eval/pref_traj.json` and falls back to the legacy flat `analysis/ckpt_pref_traj/`
layout, normalizing single-animal files into the multi-animal schema. `source_counts` below
records which layout actually answered.

Each run contributes 41 points: 40 intermediate checkpoints (every 625 steps = 0.5 epoch) plus
the endpoint written from the `checkpoints/` root. **Endpoint** = that final row throughout.
""")
code("""
registry = json.loads(REGISTRY.read_text())
trajs = atraj.load_lineage(registry, LINEAGE, "pref", analysis_dir=OUTPUT_ROOT / "analysis",
                           cache_root=OUTPUT_ROOT)   # run-dir records served from the shared cache
n_designed = sum(len(v) for v in registry["cells"].values())
print(f"registry: {len(registry['cells'])} cells / {n_designed} designed runs")
print(f"loaded   : {len(trajs)} trajectories | source_counts = {atraj.source_counts(trajs)}")

# One row per RUN (cell x ftseed) at the ENDPOINT: p(a) for each of the ten animals.
rows = []
for (tag, seed), t in trajs.items():
    end = [p for p in t.points if p.get("checkpoint") == "checkpoints" and "test" in p]
    if not end:
        continue
    row = {"tag": tag, "seed": seed, "K_rel": cl.k_from_tag(tag) / KSTAR,
           "n_points": len(t.points), "animals": len(t.animals)}
    for a in ANIMALS:
        row[a] = end[-1]["test"][a]["p_target"]
    rows.append(row)
runs = pd.DataFrame(rows)

# Fate groups by the DOG outcome only; the other nine animals are never used to define groups.
def _group(p_dog: float) -> str:
    if p_dog > TRANSFER_CUT:
        return "transferring"
    return "dead" if p_dog < THRESH else "weak"

runs["group"] = runs[TARGET].map(_group)
GROUPS = ["transferring", "weak", "dead"]

print(f"\\nendpoint rows: {len(runs)} runs | all scored on {sorted(runs.animals.unique())} animals "
      f"| points/run {sorted(runs.n_points.unique())}")
print("\\ngroup sizes (row unit = one run = cell x ftseed; grouping uses p(dog) ONLY):")
display(runs.group.value_counts().reindex(GROUPS).rename("n_runs").to_frame())
""")

md(r"""
**Cell coverage in $K/K^\ast$.** $K=\mathrm{lr}\cdot\alpha^{0.561}\cdot r^{-0.085}$ (the
collapse-law helper `cl.k_from_tag`), $K^\ast=7.87\times10^{-4}$. The strip spans four decades of
relative dose, straddling the boundary at $K/K^\ast=1$; row unit below is one **cell**.
""")
code("""
cells_k = (runs.groupby("tag").agg(K_rel=("K_rel", "first"), n_runs=("seed", "size"))
           .reset_index())
edges = [0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, np.inf]
cov = (cells_k.assign(bin=pd.cut(cells_k.K_rel, edges))
       .groupby("bin", observed=True)
       .agg(n_cells=("tag", "size"), n_runs=("n_runs", "sum")).reset_index())
cov["K/K* range"] = cov["bin"].astype(str)
print("Coverage. Row unit = one K/K* bin; n_cells = distinct HP cells, n_runs = their seeds.")
display(cov[["K/K* range", "n_cells", "n_runs"]])
print(f"cells {len(cells_k)} | K/K* from {cells_k.K_rel.min():.3g} to {cells_k.K_rel.max():.3g} "
      f"| below boundary (K/K*<1): {int((cells_k.K_rel < 1).sum())} cells, "
      f"above: {int((cells_k.K_rel >= 1).sum())} cells")
""")

md(r"""
**Priors.** One row per animal: the untuned ds1 base's mean $p(a)$ on the same `test` prompts,
with the dog-mass-renormalized prior $P_a/(1-P_{\texttt{dog}})$ used in §2. Provenance is printed
from the file itself.
""")
code("""
prior_tbl = pd.DataFrame({
    "animal": ANIMALS,
    "test_prior": [PRIORS[a] for a in ANIMALS],
    "prior x1e3": [1e3 * PRIORS[a] for a in ANIMALS],
    "renorm_prior": [PRIORS[a] / (1 - PRIORS[TARGET]) for a in ANIMALS],
}).round({"test_prior": 6, "prior x1e3": 3, "renorm_prior": 6})
print(f"Provenance: base={_p['base']}, prompt_set={_p['prompt_set']}, n_prompts={_p['n_prompts']}; "
      "one fp32 forward per prompt, p(' {animal}') read off the same next-token distribution "
      "(mirrors prob_traj_multi). Row unit = one animal.")
display(prior_tbl)
print(f"dog is the highest-prior animal ({PRIORS[TARGET]:.4g}); the nine others span "
      f"{min(PRIORS[a] for a in OTHERS):.4g}-{max(PRIORS[a] for a in OTHERS):.4g}, "
      "so raw p is not comparable across animals -- every later table uses p/prior.")
""")

# ------------------------------------------------------------------------------------------------
md(r"""
## 2. Endpoint specificity table

Runs are split by their **dog** outcome only:

| group | rule | reading |
|---|---|---|
| `transferring` | $p_{\texttt{dog}} > 0.10$ | the trait clearly took |
| `weak` | $P_{\texttt{dog}}/e^{3} \le p_{\texttt{dog}} \le 0.10$ | alive but modest |
| `dead` | $p_{\texttt{dog}} < P_{\texttt{dog}}/e^{3}$ | trait channel lost |

For each group and animal: median endpoint $p_a$, the ratio $p_a/P_a$, the **dog-mass
renormalized** ratio (median of $p_a/(1-p_{\texttt{dog}})$ over $P_a/(1-P_{\texttt{dog}})$ --
which asks whether a non-dog animal is suppressed *beyond* the mechanical displacement caused by
dog taking probability mass), and how many runs exceed $2\times$ their prior.

Row unit = one animal within one group; aggregation = median across the group's runs.
""")
code("""
def specificity_table(sub: pd.DataFrame) -> pd.DataFrame:
    out = []
    for a in ANIMALS:
        med = sub[a].median()
        renorm_med = (sub[a] / (1 - sub[TARGET])).median()
        renorm_prior = PRIORS[a] / (1 - PRIORS[TARGET])
        out.append({
            "animal": a,
            "median p": med,
            "p/prior": med / PRIORS[a],
            "renorm p/prior": renorm_med / renorm_prior,
            "n > 2x prior": int((sub[a] > 2 * PRIORS[a]).sum()),
            "n runs": len(sub),
        })
    return pd.DataFrame(out).round({"median p": 5, "p/prior": 2, "renorm p/prior": 2})


tables = {}
for g in GROUPS:
    sub = runs[runs.group == g]
    tables[g] = specificity_table(sub)
    print(f"\\n=== {g}: n={len(sub)} runs (row unit = animal; median over runs) ===")
    display(tables[g])

t = tables["transferring"]
dog_ratio = float(t.loc[t.animal == TARGET, "p/prior"].iloc[0])
oth = t[t.animal != TARGET]
print(f"\\ntransferring (n={int(t['n runs'].iloc[0])}): dog = {dog_ratio:.2f}x prior; "
      f"the nine others span {oth['p/prior'].min():.2f}-{oth['p/prior'].max():.2f}x prior "
      f"(max renormalized {oth['renorm p/prior'].max():.2f}x). "
      f"All nine <= 1x prior: {bool((oth['p/prior'] <= 1).all())}; "
      f"all nine <= 1x renormalized: {bool((oth['renorm p/prior'] <= 1).all())}.")
d = tables["dead"]
print(f"dead (n={int(d['n runs'].iloc[0])}): max median p across all ten animals = "
      f"{d['median p'].max():.3g}; total runs above 2x prior on ANY animal = "
      f"{int(d['n > 2x prior'].sum())} -> no trait swap, the whole animal channel is gone.")
""")

md(r"""
### 2b. Is any *other* animal ever the winner?

The table above is a median, which a rare swap could hide. This checks every run individually:
which animal has the highest $p_a/P_a$ at the endpoint? A trait swap would show up as a non-dog
animal winning in a non-trivial share of runs. Row unit = one run.
""")
code("""
ratio_cols = runs[ANIMALS].div(pd.Series(PRIORS), axis=1)
alive = runs.group.ne("dead") & ratio_cols.max(axis=1).gt(0)
winner = ratio_cols[alive].idxmax(axis=1)
win_tbl = (winner.value_counts().rename("n_runs_won").to_frame()
           .assign(share=lambda d: (d.n_runs_won / len(winner)).round(3)))
print(f"Row unit = one ALIVE run with a non-zero animal readout (n={int(alive.sum())} of "
      f"{len(runs)}); winner = argmax_a p_a/P_a at the endpoint.")
display(win_tbl)
print(f"dog wins {int(win_tbl.loc[TARGET, 'n_runs_won'])} / {len(winner)} "
      f"({100 * win_tbl.loc[TARGET, 'share']:.1f}%) of alive runs.")
""")

# ------------------------------------------------------------------------------------------------
md(r"""
## 3. Per-animal endpoint $p/P$ vs $K/K^\ast$

The specificity picture across the whole dose axis: **dog** prominent, the nine other animals as
thin small-marker series. Row unit = one cell (median over that cell's ftseeds); x = $K/K^\ast$
on a log axis; horizontal line at $p/P=1$ (no change from the base), vertical line at the
collapse boundary $K/K^\ast=1$. The dead threshold is drawn in dog units
($p/P = e^{-3}$) since it is defined on dog.
""")
code("""
cellmed = runs.groupby("tag").agg({**{a: "median" for a in ANIMALS}, "K_rel": "first"}).reset_index()
FLOOR = 1e-3  # log-axis floor so exactly-zero (fully collapsed) cells stay visible

fig, ax = plt.subplots(figsize=(10.5, 6))
for a in OTHERS:
    y = np.maximum(cellmed[a] / PRIORS[a], FLOOR)
    ax.plot(cellmed.K_rel, y, ".", ms=5, alpha=0.55, label=a)
y_dog = np.maximum(cellmed[TARGET] / PRIORS[TARGET], FLOOR)
ax.plot(cellmed.K_rel, y_dog, "o", ms=9, color="tab:red", alpha=0.9,
        markeredgecolor="black", markeredgewidth=0.4, label="dog (teacher trait)", zorder=5)
ax.axhline(1.0, color="0.35", lw=1.2, ls="--")
ax.axvline(1.0, color="tab:purple", lw=1.4, ls="-.")
ax.axhline(float(np.exp(-DEAD_MARGIN)), color="tab:brown", lw=1.1, ls=":")
ax.text(cellmed.K_rel.min(), 1.08, "p = prior", color="0.35", fontsize=9, va="bottom")
ax.text(1.05, ax.get_ylim()[0], "K* boundary", color="tab:purple", fontsize=9, rotation=90,
        va="bottom")
ax.text(cellmed.K_rel.min(), np.exp(-DEAD_MARGIN) * 1.06, "dead cut (dog)", color="tab:brown",
        fontsize=9, va="bottom")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$K/K^\\ast$  (relative dose; boundary at 1)")
ax.set_ylabel(r"endpoint  $p_a\\,/\\,P_a$   (cell median)")
ax.set_title("Specific transfer below the boundary, global destruction above\\n"
             f"row unit = cell (n={len(cellmed)}), median over ftseeds")
aplots.right_legend(ax, title="animal")
plt.show()

below = cellmed[cellmed.K_rel < 1]
above = cellmed[cellmed.K_rel >= 1]
print(f"below boundary (n={len(below)} cells): dog median p/P = "
      f"{(below[TARGET] / PRIORS[TARGET]).median():.2f}x, others "
      f"{np.median([(below[a] / PRIORS[a]).median() for a in OTHERS]):.2f}x")
print(f"above boundary (n={len(above)} cells): dog median p/P = "
      f"{(above[TARGET] / PRIORS[TARGET]).median():.2f}x, others "
      f"{np.median([(above[a] / PRIORS[a]).median() for a in OTHERS]):.2f}x")
""")

# ------------------------------------------------------------------------------------------------
md(r"""
## 4. Temporal: does any non-dog animal rise transiently?

A trait could in principle be *generic early, specific late* -- the student first becomes
"animal-ish" and only later concentrates on dog. That would appear as a mid-training bump above
$p/P=1$ for the non-dog animals.

Selection: the **transferring** runs only (the ones where the trait demonstrably took), each
contributing its 40 intermediate checkpoints. Row unit for the curve = one checkpoint step;
aggregation = median $p_a/P_a$ across those runs.
""")
code("""
transferring_keys = set(zip(runs.loc[runs.group == "transferring", "tag"],
                            runs.loc[runs.group == "transferring", "seed"]))
steps, series = None, {a: [] for a in ANIMALS}
for key in transferring_keys:
    pts = [p for p in trajs[key].points
           if p.get("checkpoint") != "checkpoints" and "test" in p and not p.get("nonfinite")]
    if len(pts) != 40:
        continue
    if steps is None:
        steps = [p["step"] for p in pts]
    for a in ANIMALS:
        series[a].append([p["test"][a]["p_target"] / PRIORS[a] for p in pts])
med = {a: np.median(np.array(series[a]), axis=0) for a in ANIMALS}
n_traj = len(series[TARGET])

fig, ax = plt.subplots(figsize=(10.5, 6))
for a in OTHERS:
    ax.plot(steps, med[a], "-o", ms=4, lw=1.1, alpha=0.65, label=a)
ax.plot(steps, med[TARGET], "-o", ms=9, lw=2.4, color="tab:red", label="dog (teacher trait)",
        zorder=5)
ax.axhline(1.0, color="0.35", lw=1.2, ls="--")
ax.text(steps[0], 1.06, "p = prior", color="0.35", fontsize=9, va="bottom")
ax.set_yscale("log")
ax.set_xlabel("training step (checkpoint every 625 steps = 0.5 epoch)")
ax.set_ylabel(r"median  $p_a\\,/\\,P_a$  across transferring runs")
ax.set_title("Dog separates monotonically; no non-dog animal rises above its prior\\n"
             f"selection = transferring runs only (n={n_traj}), row unit = checkpoint step")
aplots.right_legend(ax, title="animal")
plt.show()

peak = pd.DataFrame([{
    "animal": a,
    "peak p/prior": med[a].max(),
    "peak step": steps[int(med[a].argmax())],
    "min p/prior": med[a].min(),
    "min step": steps[int(med[a].argmin())],
    "final p/prior": med[a][-1],
} for a in ANIMALS]).round({"peak p/prior": 2, "min p/prior": 2, "final p/prior": 2})
print(f"Row unit = one animal; median trajectory over the n={n_traj} transferring runs.")
display(peak)
oth_peak = peak[peak.animal != TARGET]
i_max = oth_peak["peak p/prior"].idxmax()
print(f"\\nmax over the nine non-dog animals of their PEAK median ratio = "
      f"{oth_peak['peak p/prior'].max():.2f}x (animal {oth_peak.loc[i_max, 'animal']}, step "
      f"{int(oth_peak.loc[i_max, 'peak step'])}) -- any non-dog animal ever exceeding 1.1x: "
      f"{bool((oth_peak['peak p/prior'] > 1.1).any())}")

# Separate an EARLY (near-init) elevation from a genuine MID-TRAINING transient: restrict the
# search to steps past the first four checkpoints, where a "generic animal phase" would show up.
MID_FROM = 2500
mid = np.array(steps) >= MID_FROM
mid_peak = pd.DataFrame([{
    "animal": a,
    f"peak p/prior (step>={MID_FROM})": med[a][mid].max(),
    "at step": int(np.array(steps)[mid][int(med[a][mid].argmax())]),
} for a in OTHERS]).round(2)
print(f"\\nMid-training only (steps >= {MID_FROM}), the window where a generic-animal phase would "
      "appear. Row unit = one non-dog animal:")
display(mid_peak)
col = f"peak p/prior (step>={MID_FROM})"
print(f"max non-dog mid-training median ratio = {mid_peak[col].max():.2f}x "
      f"({mid_peak.loc[mid_peak[col].idxmax(), 'animal']}) -- "
      f"any exceeding 1.0x after step {MID_FROM}: {bool((mid_peak[col] > 1.0).any())}")
early = [a for a in OTHERS if med[a][~mid].max() > 1.0]
print(f"non-dog animals above 1x prior in the first {int(mid.size - mid.sum())} checkpoints "
      f"(< step {MID_FROM}): {early if early else 'none'}")
""")

md(r"""
**Answer: no mid-training transient.** Restricted to steps $\ge 2500$ -- the window where a
"generic animal first, dog later" phase would have to appear -- **no** non-dog animal reaches its
own prior: the best is elephant at ~0.9x. Every one of the nine instead *falls* early, bottoming
in the 0.2--0.6x range around steps 2.5k--7.5k, then partially recovering toward (but not
reaching) its prior by the endpoint. Dog meanwhile rises monotonically, ~1.4x to ~10x.

The one wrinkle worth stating plainly: **cow** and **rabbit** sit slightly *above* their priors
at the first one or two checkpoints (cow peaks at 1.43x at step 1250, rabbit ~1.0x at step 625),
before declining like the rest. That is a near-initialization effect in the first half-epoch, not
a competing trait taking hold -- it is gone by step 2500 and never returns. So the transfer is
specific essentially from the start; there is no generic-animal phase that later narrows onto dog.
""")

# ------------------------------------------------------------------------------------------------
md(r"""
## 5. Verdict
""")
code("""
tt = tables["transferring"]
tt_oth = tt[tt.animal != TARGET]
dd = tables["dead"]
n_tr = int(tt["n runs"].iloc[0])
n_dead = int(dd["n runs"].iloc[0])
n_weak = int(tables["weak"]["n runs"].iloc[0])

print(f"1. SPECIFICITY. In the {n_tr} transferring runs the dog probability sits at "
      f"{float(tt.loc[tt.animal == TARGET, 'p/prior'].iloc[0]):.1f}x the ds1 base prior, while "
      f"all nine other animals sit at {tt_oth['p/prior'].min():.2f}-{tt_oth['p/prior'].max():.2f}x "
      f"-- i.e. at or BELOW their priors. The teacher moves its own trait and nothing else.")
print(f"\\n2. NO TRAIT SWAP IN COLLAPSE. Across the {n_dead} dead runs the median endpoint "
      f"probability is {dd['median p'].max():.3g} for EVERY one of the ten animals, and "
      f"{int(dd['n > 2x prior'].sum())} dead runs exceed 2x prior on any animal. Collapse "
      f"destroys the animal channel globally; it does not hand the trait to a different animal.")
print(f"\\n3. MILD SUPPRESSION OF NON-TARGET ANIMALS. The nine others are not merely flat in "
      f"transferring runs, they are pushed down (median {tt_oth['p/prior'].median():.2f}x prior). "
      f"Renormalizing away dog's probability mass leaves them at "
      f"{tt_oth['renorm p/prior'].min():.2f}-{tt_oth['renorm p/prior'].max():.2f}x "
      f"(median {tt_oth['renorm p/prior'].median():.2f}x), so the suppression "
      f"{'SURVIVES' if tt_oth['renorm p/prior'].median() < 1 else 'does NOT survive'} "
      f"renormalization: it is not just mechanical displacement by dog's mass.")
print(f"\\n4. SHAPE OF THE DOSE RESPONSE. Below the boundary transfer is specific; above it the "
      f"whole animal channel is destroyed for every animal alike ({n_dead} dead runs). The "
      f"{n_weak} weak runs interpolate. Specificity is therefore a property of the "
      f"sub-boundary regime, and collapse is trait-agnostic damage.")
""")

md(r"""
### What this does and does not establish

- The ten animals are **one** measurement per checkpoint (same forward pass, different token id),
  so the comparison is exact -- no cross-run or cross-prompt confound.
- The nine non-dog animals were chosen as common single-token animal names; they are *not* a
  random sample of the animal vocabulary, so "no other animal rises" is a statement about this
  panel, not a proof over all traits.
- Everything here is one teacher (`ep2_lr5e-4_r32a64`, tseed 1) into one student base (ds1). The
  specificity claim is about this arm; the trait-free control below is what turns it into a
  claim about distillation in general.

### Pending: the trait-free ds1 arm

The decisive control is a **trait-free** arm: numbers generated by the *untuned* `pythia-160m`
base (no teacher, no trait) distilled into ds1 students over the **same** Tier B cell design,
scored on the **same** ten animals. That arm separates "the dog teacher suppresses other animals"
from "any distillation on number data suppresses animal tokens". Its queues
(`traitfree-ds1-tierB`: 316 distill runs over 233 cells, plus 233 ten-animal preference-trajectory
jobs and 233 Pile-KL jobs) are prepared but not yet drained. When they land, its §2-equivalent
table slots in beside the transferring group here, and the comparison to make is the non-dog
`renorm p/prior` row: if the trait-free arm shows the same ~0.3-1.0x suppression, the suppression
in §5.3 is generic to number distillation rather than caused by the dog trait.
""")

# ------------------------------------------------------------------------------------------------
nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
})

if __name__ == "__main__":
    import os
    import sys

    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "8")

    from nbclient import NotebookClient

    client = NotebookClient(nb, timeout=3600, kernel_name="python3",
                            resources={"metadata": {"path": str(REPO / "notebooks")}})
    try:
        client.execute()
    except Exception as exc:  # still write the notebook for inspection
        print(f"EXECUTION FAILED: {exc}", file=sys.stderr)
        nbf.write(nb, OUT)
        raise
    nbf.write(nb, OUT)
    print(f"wrote executed notebook -> {OUT}")
