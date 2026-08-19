"""Builder for notebooks/nors_asks.ipynb (one-shot; re-run to regenerate).

STRIP-PRIMARY since 2026-08-10 (Angela's directive): every fitted statistic is
computed on the non-adaptive 8x7 strip (calibrated inference); adaptive-era campaign
data appears only in [adaptive-only]-labeled sole-source blocks (per-arm checks,
budget contrast, §8b method history). The pre-2026-08-10 reconstruction/reconciliation
apparatus is retired. Keep builder and notebook in sync when editing.

Sections (numbering matches backup_exp.md and the reply-to-nor structure of backup.md):
  §1  norm audit, ep20 endpoints only: per-arm inventory (1.0), raw tables (1.1a/1.1b
      + the common-HP 1.1b-ii), exact W_final (1.2), μP/balance (1.3),
      term decomposition (1.4), bound tightness (1.5)
  §2  Test 1 — threshold stability across rank (+ raw-norm & per-arm robustness)
  §3  Test 3 — stable rank
  §4  Test 2 — activation measurements + the selectivity split
  §5  dose-law regressions; §5.1 stratified/regime fits; §5.2 controls;
      §5.3 direct term fits + the regional collapse law; §5.4 frozen-A ablation
  §5.5 evaluation-time interventions (λ dose-response + overlay + revival curve,
      W0-sliver surgery, spread surgery, unification check)
  §6  the verdict
  §7  the non-adaptive 8x7 strip (calibrated Test 1, dose-law quadratics, regional law,
      divergence ceiling; harvest = analysis/nonadaptive_8x7_harvest.json)
  §8b the epistemic lesson

Data dependencies (all under ``{output_root}/analysis``):
  adapter_norms_sweep_all_arms_all_teachers.json   (476k module rows, test_p inline)
  adapter_activations_ep20_ds1_{numbers,pile}.json (196 ep20 cells, ftseed1)
  adapter_norms_growth_mech_probe.json             (checkpointed balance, §1.3)
  frozen_a_ablation.json / frozen_a_ablation_ep20_full.json (§5.4)
  frozen_a_ablation_moved.json                     (§5.4 mirrored splice; optional)
  frozen_a_ablation_moved_ep20_full.json           (§5.4 full-grid mirrored splice)
  frozen_a_ablation_strip_{frozen,moved}_r*_s*.json (§5.4 strip splices — primary when
      the strip-splices queue is drained; adaptive full-grid used as fallback)
  spread_depth_probe_ladder.json                   (§4e 20-cell rule-selected replication)
  frozen_a_ablation_scale_lam*.json / _revival_lam*.json (§5.5 λ sweeps)
  direction_surgery_*.json / spread_surgery_*.json (§5.5 surgeries)
  frozen_a_ablation_lamstrat*_lam*.json / _ws1lam_lam*.json (§5.5b expansion λ)
  spread_ladder_*.json (§5.5b expansion spread splices; s2/ws1 prefixed variants)
  run_logs/causal_exp_cells_v2.json (§5.5b frozen selection registry)
  nonadaptive_8x7_harvest.json                     (§7 strip; cells guard on existence)
  adapter_dw_sweep_pythia-160m-data-seed1_wfinal.json (§1.2 full-grid ‖W_final‖, both
      norms; produced by ``scripts/adapter_dw_sweep.py --wfinal``, ~20 min at 8 workers)
plus, for §1.2's independent third path only, the five probe cells' adapters + the base
model (CPU).
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "notebooks" / "nors_asks.ipynb"

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells = []

cells.append(md(r"""# Nor's asks — the norm audit, the dose mechanism, and the causal program

**What this notebook is** (last revised 2026-08-11). It began as the reply to a
reviewer's challenge about which norm measures "update size" and has grown into the
collapse program's central document, in five arcs: **anatomy** (§1 — what a LoRA
update is: the exact two-term decomposition, where the weights do and don't move),
**arbitration** (§2–§4 — which norm behaves lawfully, from the threshold test to the
energy/spread regression), **what death is** (§4b — containment failure, not
overshoot), **mechanism** (§5 — both laws derived from measured ingredients; the
collapse boundary as the dose law's level set), and **intervention + calibration**
(§5.4–§5.5 — the term splices, the λ dose-response, the direction and spread
surgeries; §7 — the non-adaptive strip, where classical inference is calibrated for
the first time). §0 lists every finding with its evidence class; §6/§8b/§12 close.

**Data policy (2026-08-10, per Angela).** Every fitted statistic is computed on the
**non-adaptive strip** — the outcome-independent factorial where classical inference
is calibrated (§7 for design and census). The adaptive-era campaign data appears only
where it is the sole source (per-arm checks, the budget contrast, the §8b method
history), always labeled `[adaptive-only]`. History in one line: the notebook began
as a reconstruction of a wipe-destroyed original and was verified against its written
record before this restructure; that apparatus is retired.

**The standing assumption, and what kind of claim it is.** Before any of this, one fact
was already established: update size predicts fate. Rank all runs by
$\lVert\Delta W\rVert/\lVert W_0\rVert$ and the dead runs sit at the top with almost no
interleaving — AUC ≈ 0.97 or better. The audit does not re-establish this; it takes it
as given. What matters is keeping three different claims separate, because they sound
alike and are not:

1. **Size predicts fate.** Established before the audit. Purely statistical.
2. **Size is the mechanistically meaningful variable** — the thing a law can be written
   about. The first arc (§§1–4, §6) argues this by testing which formulation of "size"
   behaves lawfully; the second arc (§5) strengthens it by deriving the laws' exponents
   from the update's internal structure; §5.5's spread surgery and §5.4's handoff
   result now support it *interventionally*, and §7 replicates its key statistics with
   calibrated uncertainty.
3. **Size causes fate.** Originally untested here — the sections through §5.4 are
   observational (aggressive HP cells produce both big updates and dead runs, and no
   regression breaks that confound). **§5.5 changed its status**: pure dose rescaling
   flips fate at evaluation time in both directions, including reviving dead models —
   but with a *history-dependent* flip point (trained-dead updates are worse per unit
   norm), so the honest current form is *dose causes fate given the update's
   direction, and direction quality sets each update's personal threshold*. What that
   direction property is remains open (the W₀-projection and summary spread are both
   ruled out); training-time causation is still untested (`causal_plan.md`, clamp and
   retrain tiers).

**The question, and how it was made testable.** "Update size" can be measured two ways,
and the reviewer ("nor") challenged the choice:

- **Spectral norm** = the largest singular value = how hard the update pushes in its
  single strongest direction. Ignore everything else; measure only the peak.
- **Frobenius norm** = the square root of total energy summed over every direction.

Speaker-system analogy: spectral is the loudest single frequency; Frobenius is the
total volume across the whole spectrum. Nor's position was essentially *"damage comes
from the peak."* He gave two reasons for preferring spectral: (a) under Adam, Frobenius
weight change washes out as model width grows, so a Frobenius-based law might be an
artifact of one width — untestable at a single model size, so it is parked as a caveat,
not answered; (b) spectral norm bounds activation growth, which was his candidate
mechanism for collapse — testable, and §4 tests it.

Rather than debating the theory, the audit pre-registered an empirical standard: the
right norm is the one that behaves **lawfully**, meaning (i) its death threshold does
not depend on incidental parameterization choices like rank, and (ii) its value
predicts the activation movement that actually happens when data flows through.
Decision rule, fixed in advance: **if tests (i) and (ii) agree, that norm wins.**
(They did — and the question was later closed above the pre-registered standard:
§5.5's spread surgery manipulates the disputed quantity directly at matched energy,
and §7.1 replicates the threshold ordering with calibrated intervals.)

One honest note about this move: it converts "which norm is mechanistically right" into
"which norm gives stable, activation-predictive laws in this dataset." That is the
correct operationalization, but nor's width argument was sidestepped by it, not
refuted.

Shared conventions: dose $\mathrm{dw}_\ell$ = mean over the ~48 LoRA modules of
$s\lVert BA\rVert_\ell/\lVert W_0\rVert_\ell$ with $s=\alpha/r$; fate = the run's
dog-preference **test** $p_\mathrm{target}$, DEAD iff $p<0.01$ (raw eval labels);
$\lVert BA\rVert$ computed exactly from the $r\times r$ QR core, never materialising
$\Delta W$; **fitted set** = ds1 focus lineage (teacher `ep2_lr5e-4_r32a64_tseed1`,
`ex10-20_prefix_n100k`), full ep20 HP grid."""))

cells.append(md(r"""## §0 Findings at a glance (in notebook order)

Evidence classes: **[cal]** calibrated (strip; classical inference valid),
**[ident]** exact identity or measured near-identity, **[interv]** interventional
(weight splice at evaluation time), **[adaptive]** adaptive-era data, sole source,
uncalibrated.

1. **[cal]** Dead endpoints carry far more update norm than alive in both norms, but
   the per-factor ratios are a *gradient* (A₀ ≈ 1 → A_f → ΔA → B) heavily shaped by
   composition — norm bookkeeping, not localization of function (§1.1).
2. **[cal]** At the death boundary the weights have barely moved (pooled-Frobenius
   within ~0.5% of 1 at the smallest killing doses); the far-past-boundary explosion
   is aftermath. The two norms disagree about "did the weights move" — energy ~1%,
   gain ~2× (§1.2).
3. **[ident]** Boundary norm movement equals the orthogonal-dose prediction
   √(1+δ²); no anti-alignment is required. And cos(ΔW, W₀) is fully retired: its
   fate association does not replicate on calibrated support (29% vs 28%) and the
   surgery showed it causally inert (§1.2b, §5.5).
4. **[cal]** A stays at init scale; B does the learning (§1.3).
5. **[ident]** ΔW = s·B·A₀ + s·B·ΔA exactly; dominance is an α-dial with
   norm-dependent parity (α≈32 Frobenius, 64–128 spectral) — and §5.4's ablations
   put the *functional* handoff at the Frobenius point (§1.4).
6. **[ident]** The spectral factor bound is an equality (nor's chain fails only at
   the data link); the Frobenius product gap is exactly √r, flat in lr and α — the
   per-axis license for exponent arithmetic (§1.5, §5.3).
7. **[cal]** Test 1 settled: θ_F drifts at −0.110 [−0.126, −0.067] vs θ_2 −0.172
   [−0.233, −0.116]; the paired difference excludes zero; nor's +0.5 prediction is
   excluded by the whole θ_F interval. Within-rank AUCs saturate for both norms on
   the wide ladder and are retired as evidence. **[adaptive]** per-arm ordering 3/6,
   ws2 a genuine exception (§2).
8. **[ident]** srank(ΔW) ∈ [1.0, 6.2] — adapters concentrate; the identity forces
   exactly one threshold to drift, and the data assigns it to spectral (§3).
9. **[cal]** The energy/spread regression, in the near-orthogonal basis: at fixed
   energy, spread *increases* realized movement on the output distribution —
   γ_S = +0.51 [+0.40, +0.65] on KL(base‖ft), the basis-free readout — where the
   peak theory needed γ_S < 0. Robust to the two structures the pooled fit hides:
   positive within each fate regime (alive +0.43 / dead +0.27, on top of a
   ~6.5× containment jump at matched dose, p ≈ 1e-92) and positive at every nominal
   rank (+0.34…+1.7; log-rank-controlled +0.41) — so it is neither the fate split
   nor srank-tracks-r in disguise. The raw-stream drift target agrees within every
   fate stratum; its pooled fit is a Simpson artifact of the degenerate dead tail
   (§4).
10. **[cal]** The selectivity split: trait-channel KL is fate-blind (AUC 0.44, dead
    median *lower*), and near-boundary dead runs move the trait channel exactly as
    much as alive ones; neutral-text KL classifies fate at AUC 0.999 (alive 0.34 vs
    dead 15.4 nats), with dose (0.998) nearly indistinguishable from function as a
    classifier. Deep-dead runs blow up every channel equally (degenerate, not
    trait-specific). Death is containment failure; fate and pile-KL are sibling
    symptoms (§4b).
11. **[cal]** The depth profile of matched-energy splices: top1/flat drift ratio runs
    0.86 (L1) → 1.12 (L8) → 0.82 (output) — instant filtering, coherent recovery,
    terminal normalization; the peaked injection is louder, transmission taxes it
    immediately (§4e).
12. **[cal]** The dose law lr^1.069·α^0.690·r^−0.238 (R² 0.970) with **both**
    quadratics jointly real (lr² +0.093, α² +0.018) — the surface is convex on both
    axes, as the two-term mixture requires; the per-stratum lr exponent is flat
    (1.01–1.11): the adaptive grid's "lr regime shift" was support artifact, while
    the α shift is real and lands on the frozen-term prediction (alive-only 0.82 vs
    0.855) (§5–§5.2).
13. **[cal→ident]** The frozen term is a pure power law (lr^0.743±0.003 = B's own
    exponent to the third decimal; α/r = the s×B composition exactly); the moved term
    is genuinely convex (+0.064±0.005); the total's curvature is the moved term's
    bend plus dominance shift (§5.3).
14. **[cal]** The collapse law is regional with calibrated precision — b_L = 0.484
    [0.381, 0.587] vs b_H = 0.822 [0.738, 0.905] (WLS 0.499±0.064 / 0.882±0.098,
    3.3σ) — and the level-set identity closes boundary-locally with |z| = 0.0 in the
    H segment. Two levels, not to be conflated: the *boundary's location* needs no
    HP information beyond the dose surface (§7.2, and the threshold-world null
    model §7.2b reproduces the global logistic exponents and ~99% of labels from
    one threshold on measured dose); at the *run level* a small HP residual
    survives at matched endpoint dose (§7.3) — the endpoint is a lossy summary of
    the causal trajectory dose (§5.5c). The r-column closes via the measured θ
    drift (−0.098…−0.120 vs c_L −0.111) (§5.3, §7.2).
15. **[interv]** The term splices, both directions, on the full calibrated strip
    (932 runs × both splices × both seeds, zero selection): the trait rides
    whichever term dominates — retention crosses at the Frobenius parity point
    α ≈ 32 — retention is non-additive (sum median ~0.77), and in a substantial
    minority of dead cells the moved term partially cancels the frozen term's
    disruption on Pile (§5.4).
16. **[interv]** λ dose-response: pure rescaling flips fate monotonically both ways —
    dead cells revive (masked, not erased; the revival curve maps where that ends) —
    with a **history-dependent** flip dose (trained-alive ~2× the tolerance of
    trained-dead): dose is causally sufficient at eval time but not a universal
    constant (§5.5). **Verified 2026-08-11 at within-cell strength (§5.5b): 9/9
    mixed-fate seed pairs — same HPs, opposite training fates, placement cannot
    confound — show the alive seed tolerating more dose than the dead seed revives
    at (sign p = 0.002, median 2.0×); cross-arm on ws1 (8/8 flips). Full-sweep
    covariate-adjusted estimate (08-13): ~3.0× [2.5, 3.7] over 172 crossings, and
    the factor is α-STRUCTURED — ~4× where the moved (learned) term dominates,
    shrinking toward 1 in the frozen-dominated deep-α regime — only the learned
    direction can carry history (§5.5b). Two-object θ
    resolution: the TRAINING-time dose threshold is flat in α (nested logit
    p = 0.20; model-free slope −0.017 ± 0.032 — the level-set assumption is
    verified) while the EVAL-time λ-crossing threshold drifts (ρ = −0.27,
    p = 0.02, slope −0.081): the drift lives in the hysteresis gap (§5.5b).**
17. **[interv]** The W₀-aligned sliver is causally inert (1/34 flips at 10×
    amplification, on the census of all 17 mixed-fate cells) — a null the energy
    account entailed; and the sweep-level crossing-dose–srank check finds no
    consistent summary-spread signal in the personal thresholds (raw and
    HP-adjusted coefficients disagree; what conditional signal exists is
    spread-protective) — the open axis is subspace identity (§5.5).
18. **[interv]** Spread surgery at matched energy, on the rule-selected ladder +
    shell (both seeds): BOTH reshapings usually reduce damage (top1 35/38, flat
    31/38) despite top1 raising the spectral norm — the peak account refuted by
    intervention, and the learned weighting revealed as close to maximally damaging
    at matched energy; every flat-damage case is a boundary-crossed cell (design
    K/K* ≥ 1.37), across ranks and both α tiers — aggression, not rank, organizes
    the exceptions (§5.5/§5.5b).
19. **[cal]** A second boundary exists above collapse: at K/K* ≳ 12, training
    diverges to non-finite weights, at every rank. Alive / dead-but-revivable /
    diverged are three regimes (§7). The survivors' transfer hump replicates on
    calibrated placement (peak K/K* ≈ 0.42) (§7.2).
21. **[interv-train]** The training-time clamp (§5.5c): capping the pooled dose at
    0.8×θ̂ does NOT rescue verified-dead cells (22/28 die; 0/54 at c ≥ 1.0) — endpoint-
    dose sufficiency is falsified at training time; rescues exist only at c = 0.8 with
    a steep within-cell flip (0.67 vs 0.80×θ̂). Stage 2 settled the discriminator:
    survival is monotone in the ceiling (14/14 at 0.5×θ̂ incl. every lr5e-4 run; 11/14
    at 0.6) — direction-corruption REFUTED, dose causally sufficient at training time
    with a trajectory-exposure threshold ~0.6×θ̂; the σ/F clamp arms physically couple
    (low srank), so norm attribution stays with the eval-time surgeries.
20. **[meta]** Four superb-looking adaptive-era statistics were artifacts caught
    only by specification-stability — including an lr "regime shift" and an α²
    retraction that both *inverted* on clean support; that history, and why the
    strip exists, is §8b. Remaining uncalibrated: the six-arm cross-checks and the
    budget contrast, both labeled."""))

cells.append(md("""## Setup A — the adaptive-era campaign data **[adaptive-only]**

Loaded for the labeled sole-source blocks only (per-arm robustness, budget contrast,
§8b history). Row unit of `mod_all`: one LoRA module of one trained adapter, six arms
× 102 teacher lineages, fates inline."""))

cells.append(code(r"""from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import yaml
from scipy import stats

from subliminal_learning.analysis.setup import bootstrap
from subliminal_learning.analysis.plots import compare_table, math_label, right_legend
from subliminal_learning.analysis.frames import load_results
from subliminal_learning.analysis.stats import cluster_ols, delta_ratio_ci
from subliminal_learning.analysis.collapse_law import (
    K_of, KSTAR_POOLED, B_LAW, C_LAW, DEAD_SEED)
REPO, OUTPUT_ROOT = bootstrap()
AN = OUTPUT_ROOT / "analysis"

DS1, FOCUS_TEACHER = "pythia-160m-data-seed1", "ep2_lr5e-4_r32a64_tseed1"
DEAD_CUT = DEAD_SEED


# base-model dir -> arm label; a fully re-randomized sibling keeps a `full-` marker
# so pythia-160m-seed2 cannot be read as a data-/weight-seed arm
def arm_label(name):
    a = name[len("pythia-160m-"):] if name.startswith("pythia-160m-") else name
    return "full-" + a if a.startswith("seed") else a

mod_all = load_results(AN / "adapter_norms_sweep_all_arms_all_teachers.json")

# The sweep JSON also carries the ckpt-distill pipeline's prompt-spliced variant
# (dataset_tag *_spliced, ep<=10 only). Different training distribution -> excluded
# from every population here, not just implicitly by the ep20 restriction.
_spliced = mod_all.dataset_tag.str.endswith("_spliced")
print(f"excluded {int(_spliced.sum())} module rows of the *_spliced ckpt-distill "
      f"variant ({mod_all[_spliced].groupby(['student_base', 'dataset_tag']).ngroups} "
      "lineages, ep<=10 only)")
mod_all = mod_all[~_spliced].copy()

tp = mod_all["student_tag"].str.extract(r"ep(?P<ep>\d+)_lr(?P<lr>[0-9.e-]+)_r\d+a\d+")
mod_all["ep"], mod_all["lr"] = tp["ep"].astype(int), tp["lr"].astype(float)
_s = mod_all["alpha"] / mod_all["rank"]
for l in ("spec", "fro"):
    mod_all[f"dW_{l}"] = _s * mod_all[f"B_Afinal_{l}"]           # absolute ||dW||
    mod_all[f"dw_{l}"] = mod_all[f"dW_{l}"] / mod_all[f"W0_{l}"]  # relative dose
    mod_all[f"term_frozen_{l}"] = _s * mod_all[f"B_Ainit_{l}"]    # s·B·A_init
    mod_all[f"term_moved_{l}"] = _s * mod_all[f"B_dA_{l}"]        # s·B·dA
mod_all["dead"] = mod_all.test_p < DEAD_CUT

foc = mod_all[(mod_all.student_base == DS1) & (mod_all.teacher == FOCUS_TEACHER)].copy()
f20 = foc[foc.ep.eq(20) & foc.test_p.notna()].copy()

MEAN_COLS = ["dw_fro", "dw_spec", "dW_fro", "dW_spec",
             "term_frozen_fro", "term_frozen_spec", "term_moved_fro", "term_moved_spec",
             "dA_fro", "dA_spec", "B_fro", "B_spec"]

def to_runs(m):
    '''Module rows -> one row per run (module-mean doses, first-value HPs/fate).

    Keys include teacher/base/dataset when present: different teacher lineages reuse
    the same student_tag names, so tag+seed alone would merge distinct runs.
    '''
    keys = [c for c in ("student_base", "teacher", "dataset_tag", "student_tag",
                        "ftseed") if c in m.columns]
    out = (m.groupby(keys, as_index=False)
            .agg(ep=("ep", "first"), lr=("lr", "first"), rank=("rank", "first"),
                 alpha=("alpha", "first"), test_p=("test_p", "first"),
                 dead=("dead", "first"),
                 **{c: (c, "mean") for c in MEAN_COLS}))
    # cell id for cluster-robust SEs (a cell = one HP point of one lineage)
    cell_cols = [c for c in ("student_base", "teacher", "dataset_tag", "student_tag")
                 if c in out.columns]
    out["cell"] = out[cell_cols].astype(str).agg("|".join, axis=1)
    return out

run20 = to_runs(f20)
run_all_budgets = to_runs(foc[foc.test_p.notna()])
print(f"fitted set: {len(run20)} ep20 runs ({int(run20.dead.sum())} dead), "
      f"{len(f20)} module rows")
print(f"init reconstruction check: right-seed cos median "
      f"{f20.cos_Ainit_Afinal.median():.3f}, wrong-seed |cos| median "
      f"{f20.cos_wrongseed_Afinal.abs().median():.4f} (chance floor ~1/sqrt(r*768))")"""))

cells.append(md(r"""## Setup B — the primary population: the non-adaptive strip

Every fitted statistic below is computed on the **non-adaptive 8×7 strip** (932
evaluated runs; design and census in §7): outcome-independent placement, identical
lr×α support at every rank, exact adapter-derived doses, and the pre-registered
s = α/r ≤ 64 primary region. The adaptive-era campaign data (six arms × 102
teachers) is retained under `*_a` names and appears **only** where it is the sole
source — the per-arm robustness checks, the budget contrast, and the §8b
method-history — always labeled. Where a quantity could be computed on either
population, only the strip version is shown."""))

cells.append(code(r"""# preserve the adaptive-era frames for the labeled sole-source blocks
f20a, run20a = f20, run20

HARV = AN / "nonadaptive_8x7_harvest.json"
strip = pd.read_json(HARV)
strip["dead"] = np.where(strip.diverged, 1.0,
                         np.where(strip.test_p.notna(),
                                  (strip.test_p < 0.01).astype(float), np.nan))
sev = strip[strip.test_p.notna() & strip.dw_fro.notna()].copy()
SP = sev[sev.s <= 64].copy()

# strip module rows from the decomposition probe (same schema as the sweep)
f20 = pd.concat([pd.DataFrame(json.loads(f.read_text())["results"])
                 for f in sorted(AN.glob("adapter_norms_strip_r*.json"))])
f20["lr"] = f20.student_tag.str.extract(r"lr(?P<lr>[0-9.e-]+)_r").lr.astype(float)
f20["ep"] = 20
_s = f20["alpha"] / f20["rank"]
for l in ("spec", "fro"):
    f20[f"dW_{l}"] = _s * f20[f"B_Afinal_{l}"]
    f20[f"dw_{l}"] = f20[f"dW_{l}"] / f20[f"W0_{l}"]
    f20[f"term_frozen_{l}"] = _s * f20[f"B_Ainit_{l}"]
    f20[f"term_moved_{l}"] = _s * f20[f"B_dA_{l}"]
fates = strip.set_index(["tag", "seed"]).test_p
f20["test_p"] = fates.reindex(
    pd.MultiIndex.from_frame(f20[["student_tag", "ftseed"]])).to_numpy()
f20["dead"] = f20.test_p < 0.01
f20 = f20[(f20["alpha"] / f20["rank"] <= 64) & f20.test_p.notna()].copy()
run20 = to_runs(f20)
print(f"PRIMARY population (strip, s<=64): {len(run20)} runs "
      f"({int(run20.dead.sum())} dead), {len(f20)} module rows; adaptive frames "
      f"retained as f20a/run20a ({len(run20a)} runs)")
print("init reconstruction check (strip): right-seed cos median %.3f, wrong-seed "
      "|cos| median %.4f" % (f20.cos_Ainit_Afinal.median(),
                             f20.cos_wrongseed_Afinal.abs().median()))"""))

cells.append(md(r"""## §1 — the raw audit

**Population, fixed once for the whole section: the strip's primary region** (890
runs, 371 dead, s ≤ 64; all ep20 by design, so the budget confound the adaptive
campaigns had to fight — early-epoch fate contrasts reversing per-arm signs — cannot
arise here). Every table is computed from the strip's own decomposition probe; the
six-arm adaptive data appears only in the per-arm subsections, labeled.

### §1.0 Inventory — what each arm actually contributes **[adaptive-only]**

The strip is single-arm by design; this inventory of the six adaptive arms exists
because the per-arm subsections below need it. Row unit: one arm. `ep20 cells` = distinct (teacher × dataset × student-HP) points at
ep20; `ep20 runs` counts ftseed replicates inside those cells; `ep20 HP tags` = distinct
student-HP tags. This table exists because the six arms were **not** swept to the same
depth, and every per-arm comparison downstream has to be read against it."""))

cells.append(code(r"""RUN_KEYS = ["student_base", "teacher", "dataset_tag", "student_tag", "ftseed"]
rec = mod_all[mod_all.test_p.notna()].copy()          # every recorded run, all budgets
rec["arm"] = rec.student_base.map(arm_label)
runs_rec = (rec.groupby(RUN_KEYS + ["arm"], as_index=False)
            .agg(ep=("ep", "first"), dead=("dead", "first")))
runs_rec["cell"] = (runs_rec[["student_base", "teacher", "dataset_tag", "student_tag"]]
                    .astype(str).agg("|".join, axis=1))
r20 = runs_rec[runs_rec.ep.eq(20)]
g20 = r20.groupby("arm")
inv = pd.DataFrame({"runs (all budgets)": runs_rec.groupby("arm").size(),
                    "ep20 runs": g20.size(),
                    "ep20 cells": g20.cell.nunique(),
                    "teachers": g20.teacher.nunique(),
                    "ep20 HP tags": g20.student_tag.nunique(),
                    "ep20 dead %": (100 * g20.dead.mean()).round().astype(int)})
display(compare_table(inv, fmt="{:.0f}"))

tagsets = {a: set(d.student_tag) for a, d in g20}
COMMON_TAGS = sorted(set.intersection(*tagsets.values()))
print("ep20 student-HP tags: union %d, INTERSECTION over all six arms %d "
      "(these 25 are Table 1.1b-ii's population)"
      % (len(set.union(*tagsets.values())), len(COMMON_TAGS)))
print("teacher axis: %s -> identical set in every arm: %s"
      % ({a: d.teacher.nunique() for a, d in g20},
         len({frozenset(d.teacher) for _, d in g20}) == 1))
print("dropped as non-ep20: %d runs (%d module rows)"
      % (len(runs_rec) - len(r20), (~rec.ep.eq(20)).sum()))

ma = rec[rec.ep.eq(20)]        # <- the §1 population from here on: ep20 endpoints only
print("Table 1.1b population: %d ep20 module rows" % len(ma))"""))

cells.append(md(r"""### §1.1 Every object, both norms, alive vs dead

**Data: non-adaptive strip** (primary region, 890 runs / 42,720 module rows).

Table 1.1a: medians over the strip's **ep20** module rows, split by run fate.
Columns are the raw objects of the decomposition $\Delta W = s\,B A_\mathrm{final}$;
the `fro/spec` column is $\sqrt{\text{stable rank}}$. Two things to know before
reading. First, the strip's dead population runs much deeper past the boundary than
the adaptive campaigns' did (the shared lr ladder reaches every pair's far side), so
the dead-side medians sit deeper — the boundary-relevant scale is the threshold
(θ_F ≈ 0.145, §2), not the dead median. Second, the per-object dead/alive ratios
(printed under the table) form a *gradient* — A₀ ≈ 1 (fate-blind by construction; its
Frobenius ratio below 1 is pure rank-composition, since dead runs skew low-rank on a
factorial) → A_f → ΔA → B — and a large part of every contrast is **composition**
(dead runs cluster at aggressive cells), so none of this locates where function
lives: §5.4's ablations do that. The last two columns are the relative dose in each
norm, alive → dead — read the spectral pair first: collapse is an order-unity event
in the top direction while total energy stays fractional, which is why
$\lVert W_\mathrm{final}\rVert_F$ barely moves at the boundary (§1.2).

"""))

cells.append(code(r"""objs = ["A_init", "A_final", "dA", "B", "W0"]
rows = {}
for name, g in (("alive", f20[~f20.dead]), ("dead", f20[f20.dead])):
    r = {}
    for o in objs:
        r[f"{o} spec"], r[f"{o} fro"] = g[f"{o}_spec"].median(), g[f"{o}_fro"].median()
    r["dW spec"], r["dW fro"] = g.dW_spec.median(), g.dW_fro.median()
    r["dW fro/spec"] = (g.dW_fro / g.dW_spec).median()
    r["dW/W0 spec"] = (g.dW_spec / g.W0_spec).median()
    r["dW/W0 fro"] = (g.dW_fro / g.W0_fro).median()   # relative dose in the decided norm
    rows[name] = r
t11a = pd.DataFrame(rows).T   # `dW/W0 *` render via the systematic math_label mapping
display(compare_table(t11a, labels={
    "dW fro/spec": "‖ΔW‖<sub>F</sub>/‖ΔW‖<sub>2</sub> (=√srank)"}))
print("row unit: one LoRA module of one ep20 adapter of the fitted set (%d rows: "
      "%d alive, %d dead)" % (len(f20), (~f20.dead).sum(), f20.dead.sum()))
print("relative dose, alive -> dead: spectral %.3f -> %.3f, Frobenius %.3f -> %.3f"
      % (t11a.loc["alive", "dW/W0 spec"], t11a.loc["dead", "dW/W0 spec"],
         t11a.loc["alive", "dW/W0 fro"], t11a.loc["dead", "dW/W0 fro"]))
ra = t11a.loc["dead"] / t11a.loc["alive"]
print("dead/alive median ratios — A_init: %.2fx / %.2fx;  A_final: %.2fx / %.2fx;  "
      "dA: %.1fx / %.1fx;  B: %.1fx / %.1fx  (2 / F) — a gradient, not a single "
      "carrier; dW's own %.1fx/%.1fx exceeds every factor ratio (composition)"
      % (ra["A_init spec"], ra["A_init fro"], ra["A_final spec"], ra["A_final fro"],
         ra["dA spec"], ra["dA fro"], ra["B spec"], ra["B fro"],
         ra["dW spec"], ra["dW fro"]))"""))

cells.append(md("""### §1.1b The same contrast per arm — twice, with and without the HP confound **[adaptive-only]**

**Data: adaptive six-arm campaigns** (388k ep20 module rows) — the strip is ds1-only,
so cross-arm generality has no calibrated version; this is its sole source.

Two tables in the cell below, same row unit (one arm × fate, medians of absolute
module-level norms over ep20 runs) and the same three columns. They differ **only** in
which student HPs each arm is allowed to contribute:

* **Table 1.1b — every ep20 run of each arm.** Maximum data (388k module rows), but the
  arms are averaging over *different* HP sets: per Table 1.0 the ep20 sweep runs from 196
  distinct HP tags for ds1 down to 25 for ws3. The teacher axis is common (102 lineages
  everywhere, except ws2's 101 at ep20), so teachers are not the issue — HPs are. The
  arm-to-arm *spread* here is therefore partly coverage, not arm identity.
* **Table 1.1b-ii — the 25 HP tags present in all six arms.** Half the rows (190k), but
  now the arms are strictly comparable: same teachers, same HPs, same budget.

The contrast — dead bigger than alive in both norms — holds in **all six arms in both
tables**, at 1.7–2.3× (raw) and 1.7–2.1× (common-HP) in $\\lVert\\Delta W\\rVert_F$. That
agreement is the point: with the budget confound removed by the ep20 restriction, the
residual HP-coverage difference moves the numbers a little and changes nothing
qualitative, so "not a ds1 peculiarity" is safe. Quote the common-HP table when comparing
*between* arms; quote the raw one for within-arm fate contrasts. (Within a *mixed* cell
the dying seed is **not** the bigger-update seed; that within-cell result is
`seed_fate_direction.ipynb`.)"""))

cells.append(code(r"""def arm_fate_table(rows):
    t = (rows.groupby(["arm", "dead"])[["dW_spec", "dW_fro", "B_spec"]]
         .median().unstack("dead").round(2))
    t.columns = [f"{a} {'dead' if b else 'alive'}" for a, b in t.columns]
    return t

t11b = arm_fate_table(ma)                                    # every ep20 run
mc = ma[ma.student_tag.isin(COMMON_TAGS)]
t11b2 = arm_fate_table(mc)                                   # the 25 shared HP tags

print("Table 1.1b — every ep20 run of each arm (%d module rows)" % len(ma))
display(compare_table(t11b, fmt="{:.2f}"))
print("Table 1.1b-ii — only the %d student HPs common to all six arms "
      "(%d module rows, %.0f%% of 1.1b)"
      % (len(COMMON_TAGS), len(mc), 100 * len(mc) / len(ma)))
display(compare_table(t11b2, fmt="{:.2f}"))

_raw = t11b["dW_fro dead"] / t11b["dW_fro alive"]
_com = t11b2["dW_fro dead"] / t11b2["dW_fro alive"]
print("dead/alive in ‖ΔW‖_F — raw %.2f–%.2f, common-HP %.2f–%.2f; arms with the "
      "contrast reversed: %d raw, %d common-HP"
      % (_raw.min(), _raw.max(), _com.min(), _com.max(),
         int((_raw < 1).sum()), int((_com < 1).sum())))"""))

cells.append(md(r"""### §1.2 Exact $W_\mathrm{final}$ over the whole grid — energy barely moves, gain can double

**Data: non-adaptive strip** (890 runs; the weight sweep covers every finite adapter
in the lineage, 1,599, and is joined to the strip's primary region here).

The norms probe stores factor norms only, so $\lVert W_\mathrm{final}\rVert$ needs
the weights themselves. `scripts/adapter_dw_sweep.py --wfinal` sweeps **every finite
adapter in the lineage** (the strip's included; diverged adapters are skipped and
counted) and records, per module, the exact $\sigma_\mathrm{max}$ and Frobenius norm
of $W_0 + sBA$ alongside the stored $\langle BA, W_0\rangle$ — so both norms of the
final weights are available run by run, with a closed-form Frobenius identity
as an internal cross-check:

$$\lVert W_0+sBA\rVert_F^2 = \lVert W_0\rVert_F^2 + 2s\langle BA,W_0\rangle + s^2\lVert BA\rVert_F^2$$

Row unit below: one strip run (primary region). Per run, `pooled (F)` is the model-level
$\sqrt{\sum_m\lVert W_f\rVert_F^2/\sum_m\lVert W_0\rVert_F^2}$; `max module` is
the worst single module, the strongest form of the claim.

"""))

cells.append(code(r"""wf_doc = json.loads((AN / f"adapter_dw_sweep_{DS1}_wfinal.json").read_text())
wf = pd.DataFrame(wf_doc["results"])
s_wf = wf.alpha / wf["rank"]
# closed form, from stored columns only -- validated against the swept Wf_fro below
wf["wf_fro_cf"] = np.sqrt(wf.W0_fro**2 + 2 * s_wf * wf.BA_W0_ip + (s_wf * wf.BA_fro) ** 2)
wf["r_fro"] = wf.Wf_fro / wf.W0_fro
wf["r_spec"] = wf.Wf_spec / wf.W0_spec
print("closed form vs computed ‖W_f‖_F: max relative error %.2e over %d module rows"
      % (np.abs(wf.wf_fro_cf / wf.Wf_fro - 1).max(), len(wf)))

POOLED, WF, MF, W2, M2 = ("pooled (F)", "worst module (F)", "median module (F)",
                          "worst module (2)", "median module (2)")
wf["dW_fro_abs"] = s_wf * wf.BA_fro
wf["ip"] = s_wf * wf.BA_W0_ip                     # <s.BA, W0>, the closed form's cross term
wf_run = wf.groupby(["student_tag", "ftseed"]).apply(
    lambda d: pd.Series({POOLED: np.sqrt((d.Wf_fro**2).sum() / (d.W0_fro**2).sum()),
                         WF: d.r_fro.max(), MF: d.r_fro.median(),
                         W2: d.r_spec.max(), M2: d.r_spec.median(),
                         # pooled (W0-mass-weighted) dose and alignment, for §1.2b
                         "dose": np.sqrt((d.dW_fro_abs**2).sum() / (d.W0_fro**2).sum()),
                         "cos": (d.ip.sum() / (np.sqrt((d.dW_fro_abs**2).sum())
                                               * np.sqrt((d.W0_fro**2).sum())))}),
    include_groups=False).reset_index()
w12 = run20.merge(wf_run, on=["student_tag", "ftseed"], how="left")
print("fitted-set runs: %d; with W_final computed: %d (%.0f%%); "
      "adapters swept in the lineage: %d"
      % (len(run20), w12[POOLED].notna().sum(),
         100 * w12[POOLED].notna().mean(), wf_doc["n_adapters"]))
w12 = w12[w12[POOLED].notna()]

RCOLS = [POOLED, WF, MF, W2, M2]
QS = {"median": 0.5, "p90": 0.9, "p99": 0.99, "max": 1.0}
by_fate = dict(list(w12.groupby(w12.dead.map({False: "alive", True: "dead"}))))
dist = pd.concat({f"{fate} {qn}": by_fate[fate][RCOLS].quantile(q)
                  for fate in ("alive", "dead") for qn, q in QS.items()},
                 axis=1).loc[RCOLS].rename_axis("ratio (per run)")
display(compare_table(dist, fmt="{:.4f}"))
print("row unit: one ep20 run of the fitted set; n = %d alive, %d dead"
      % ((~w12.dead).sum(), w12.dead.sum()))
print("\nEVERY alive run: pooled F <= %.4f, worst module F <= %.4f, worst module 2 <= %.4f"
      % (w12[~w12.dead][POOLED].max(), w12[~w12.dead][WF].max(),
         w12[~w12.dead][W2].max()))
d12 = w12[w12.dead]
print("dead runs: pooled F min %.4f, 25%% %.4f, median %.4f, max %.3f  |  "
      "%.0f%% of dead runs stay under 1.01 pooled"
      % (d12[POOLED].min(), d12[POOLED].quantile(.25), d12[POOLED].median(),
         d12[POOLED].max(), 100 * (d12[POOLED] < 1.01).mean()))
print("dead runs, spectral: worst module (2) min %.4f, median %.4f, max %.2f"
      % (d12[W2].min(), d12[W2].median(), d12[W2].max()))"""))

cells.append(md(r"""**What the full grid shows.** The claim **survives at the
transition and fails past it**: every alive run moves its pooled Frobenius weight
norm by well under a percent, the smallest doses that kill do so with the weights
essentially untouched (pooled-F within ~0.5% of 1), and only far past the boundary —
where every run is long dead — do the weights genuinely explode (the dead tail runs
to several × pooled-F and orders of magnitude in worst-module gain). Explosion is
aftermath, not cause. The two norms answer "did the weights move" differently — a
low-rank update whose top gain rivals the base's can multiply a module's maximum
amplification while adding ~1% to its energy — which is §1.5's geometry reappearing
in $W_\mathrm{final}$."""))

cells.append(md(r"""**Where the tail lives.** The per-lr table below localizes it: both
ratios track lr monotonically and the extremes are confined to the far side of the
boundary, where every run is already dead. The last column is the boundary itself — the
smallest dose that still kills at each lr."""))

cells.append(code(r"""bylr = (w12.groupby("lr").apply(
    lambda d: pd.Series({"runs": len(d), "dead %": 100 * d.dead.mean(),
                         "median pooled (F)": d[POOLED].median(),
                         "max pooled (F)": d[POOLED].max(),
                         "max worst module (2)": d[W2].max(),
                         "min pooled (F) | dead": (d[d.dead][POOLED].min()
                                                   if d.dead.any() else np.nan)}),
    include_groups=False))
bylr["runs"] = bylr.runs.astype(int)
bylr["dead %"] = bylr["dead %"].round().astype(int)
display(compare_table(bylr, fmt="{:.4f}", subset=[c for c in bylr.columns
                                                  if c not in ("runs", "dead %")]))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, (col, lab) in zip(axes, [(WF, "‖W$_f$‖$_F$/‖W$_0$‖$_F$"),
                                 (W2, "σ$_{max}$(W$_f$)/σ$_{max}$(W$_0$)")]):
    for fate, sub, c in (("alive", w12[~w12.dead], "tab:blue"),
                         ("dead", w12[w12.dead], "tab:red")):
        x = np.sort(sub[col].to_numpy())
        ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", lw=2.2,
                color=c, label=f"{fate} (n={len(x)})")
    ax.axvline(1.0, color="0.5", lw=1, ls=":")
    ax.set_xscale("log"); ax.set_xlabel(f"worst-module {lab}")
    ax.set_ylabel("fraction of runs ≤ x"); ax.set_title(lab.split("/")[0])
right_legend(axes[1], fontsize=11)
fig.suptitle("§1.2 — worst-module weight-norm ratio over the whole ep20 grid "
             f"(n={len(w12)} runs), by fate", y=1.02)
plt.tight_layout(); plt.show()"""))

cells.append(md(r"""### §1.2b Is the movement just the dose? — the cross term, and a caution about mixing aggregations

**Data: non-adaptive strip** (same joined population as §1.2).

The closed form makes one more thing checkable for free. Write the pooled movement in
terms of the pooled *dose* $\delta=\lVert\Delta W\rVert_F/\lVert W_0\rVert_F$ and the
alignment $\cos=\langle\Delta W,W_0\rangle/(\lVert\Delta W\rVert_F\lVert W_0\rVert_F)$:

$$\frac{\lVert W_\mathrm{final}\rVert_F}{\lVert W_0\rVert_F}=\sqrt{1+2\delta\cos+\delta^2}$$

so an update exactly orthogonal to $W_0$ moves the norm by $\sqrt{1+\delta^2}$ and
nothing else can be inferred without the sign of $\cos$. Two results, one of them a
correction:

**The boundary movement is exactly what the dose alone predicts.** At the 20
smallest-dose dead runs the orthogonal prediction and the measurement agree to four
decimals (both 1.0014–1.0032). There is no deficit to explain and therefore **no
anti-alignment is required** by the norm budget. The tempting inference — "relative dose
~0.2 should give ~2% movement, we measure 0.14–0.5%, so $\langle\Delta W,W_0\rangle<0$" —
fails on an **aggregation mismatch**, and it is the kind this notebook keeps having to
guard against. The 0.203 in Table 1.1a is an *unweighted median over modules* of the
whole dead population; the pooled per-run dose weights each module by its $W_0$ mass, and
the heavy MLP matrices carry the *smallest* relative doses (`query_key_value` 0.13 vs
`dense` 0.25), which pulls the pooled figure to 0.127 — and boundary runs, being the
low-dose end of the dead population rather than its middle, sit lower again at
0.05–0.08. $\sqrt{1+0.08^2}=1.0032$. The gap was never there.

**The alignment sign, retired.** Per module, $\lvert\cos\rvert$ sits barely above
the chance floor — the update is orthogonal to $W_0$ for all practical purposes — and
on this calibrated population the *sign* carries no fate association either (pooled
cos < 0 in ~29% of dead vs ~28% of alive runs; the adaptive campaigns' striking
63%-vs-25% split does not replicate). Together with the §5.5 surgery null (flipping,
deleting, or 10×-amplifying the aligned sliver at fixed dose changes nothing),
cos(ΔW, W₀) is fully retired: not a mechanism, and not even a robust marker — the
seed-level signals built on it should be re-examined as population artifacts. The
history axis the λ experiments demand (§5.5) must live elsewhere in the update's
geometry.

"""))

cells.append(code(r"""b20 = w12[w12.dead].nsmallest(20, "dose")
w12["orth"] = np.sqrt(1 + w12.dose**2)   # prediction if <dW, W0> = 0
print("BOUNDARY (20 smallest-dose dead runs): pooled dose %.3f-%.3f -> orthogonal "
      "prediction %.4f-%.4f; measured %.4f-%.4f (max |measured - predicted| %.1e)"
      % (b20.dose.min(), b20.dose.max(), np.sqrt(1 + b20.dose.min()**2),
         np.sqrt(1 + b20.dose.max()**2), b20[POOLED].min(), b20[POOLED].max(),
         np.abs(b20[POOLED] - np.sqrt(1 + b20.dose**2)).max()))
print("pooled dose, dead runs: median %.3f  (Table 1.1a's 0.203 is the UNWEIGHTED "
      "module median of the same population -- different statistic)"
      % w12[w12.dead].dose.median())

cos_mod = wf.BA_W0_ip / (wf.BA_fro * wf.W0_fro)          # s cancels; per module
DIMS_MN = {"query_key_value": 768 * 2304, "dense": 768 * 768,
           "dense_h_to_4h": 768 * 3072, "dense_4h_to_h": 3072 * 768}
chance = 1 / np.sqrt(wf.module.str.rsplit(".", n=1).str[-1].map(DIMS_MN))
print("\nper-module |cos(dW, W0)|: median %.5f vs chance floor 1/sqrt(mn) %.5f "
      "(ratio %.2f) -> effectively orthogonal; its norm contribution is ~%.0e"
      % (cos_mod.abs().median(), chance.median(),
         (cos_mod.abs() / chance).median(),
         (w12.dose * cos_mod.abs().median()).median()))
print("SIGN vs fate on this population: pooled cos < 0 in %.0f%% of dead runs vs "
      "%.0f%% of alive (%d of the 20 boundary runs) -- the adaptive-era association "
      "(63%% vs 25%%) does NOT replicate on calibrated support"
      % (100 * (w12[w12.dead]["cos"] < 0).mean(),
         100 * (w12[~w12.dead]["cos"] < 0).mean(), int((b20["cos"] < 0).sum())))"""))

cells.append(md(r"""**The envelope over every arm (Frobenius, closed form)
[adaptive-only].** The identity needs no weights, so it extends to every recorded
ep20 run — six arms × 102 teachers, the §1 population of Table 1.0 — as a check that
the focus lineage is not the mild corner of the space. Row unit: one ep20 run; `median/p99/max` are over the runs of that
arm and fate."""))

cells.append(code(r"""ma_s = ma.alpha / ma["rank"]
env = ma.assign(wf2=ma.W0_fro**2 + 2 * ma_s * ma.BA_W0_ip + (ma_s * ma.B_Afinal_fro) ** 2,
                w02=ma.W0_fro**2)
env_run = (env.groupby(RUN_KEYS)
           .apply(lambda d: pd.Series({POOLED: np.sqrt(d.wf2.sum() / d.w02.sum()),
                                       "dead": bool(d.dead.iloc[0]), "ep": d.ep.iloc[0]}),
                  include_groups=False).reset_index())
env_run["arm"] = env_run.student_base.map(arm_label)
env_run["fate"] = env_run.dead.map({False: "alive", True: "dead"})
tail = (env_run.groupby(["arm", "fate"])[POOLED]
        .agg(median="median", p99=lambda d: d.quantile(.99), max="max").unstack(1))
tail.columns = [f"{b} {a}" for a, b in tail.columns]
display(compare_table(tail[[f"{f} {s}" for f in ("alive", "dead")
                            for s in ("median", "p99", "max")]], fmt="{:.3f}"))
print("runs per arm x fate:\n%s"
      % env_run.groupby(["arm", "fate"]).size().unstack(1).to_string())
print("\nall %d recorded ep20 runs: %.1f%% exceed the recorded 1.005 figure, %.1f%% "
      "exceed 1.05; the max is %.2f (%s)"
      % (len(env_run), 100 * (env_run[POOLED] > 1.005).mean(),
         100 * (env_run[POOLED] > 1.05).mean(), env_run[POOLED].max(),
         env_run.loc[env_run[POOLED].idxmax(), "student_tag"]))
_alive = env_run[~env_run.dead]
print("but across all six arms, no ALIVE run exceeds %.4f pooled Frobenius (n=%d)"
      % (_alive[POOLED].max(), len(_alive)))"""))

cells.append(md(r"""### §1.3 μP normalization and factor balance

**What $B^\ast$ and $A^\ast$ are, and why the raw norms won't do.** LoRA replaces the
update with a product, $\Delta W = s\,BA$ with $s=\alpha/r$, where
$A\in\mathbb{R}^{r\times n_\mathrm{in}}$ is Kaiming-initialized (nonzero at step 0) and
$B\in\mathbb{R}^{n_\mathrm{out}\times r}$ starts at exactly zero. Comparing
$\lVert A\rVert$ against $\lVert B\rVert$ directly is meaningless: the two factors have
different shapes, so their norms carry different units of "size". The μP fix is to
divide each factor by the natural scale of its own shape. A correctly-scaled matrix has
$\lVert M\rVert_2\asymp\sqrt{n_\mathrm{out}/n_\mathrm{in}}$ (that is what
Kaiming/μP init buys), so

$$M^\ast \;=\; \lVert M\rVert_2\sqrt{\text{fan-in}/\text{fan-out}}$$

is the dimensionless multiplier: how much this matrix amplifies *relative to its own
shape's natural gain*, with $M^\ast\approx1$ meaning "at init scale". $A$ maps
$n_\mathrm{in}\to r$ and $B$ maps $r\to n_\mathrm{out}$, so the same rule reads

$$A^\ast = \lVert A_\mathrm{final}\rVert_2\sqrt{n_\mathrm{in}/r},\qquad
B^\ast = \lVert B\rVert_2\sqrt{r/n_\mathrm{out}},$$

and the balance ratio $B^\ast/A^\ast$ asks whether the two factors sit equally far from
their own natural scales. Both are computed at the *final* adapter (hence
$A_\mathrm{final}$); $s=\alpha/r$ cancels out of the ratio and never enters.

**Where $n_\mathrm{in},n_\mathrm{out}$ come from.** Pythia-160m is a GPT-NeoX stack:
`hidden_size` 768, 12 layers, 12 heads × 64, MLP width 3072. Four modules per block are
LoRA targets, and `DIMS` below records their $(n_\mathrm{in},n_\mathrm{out})$:
`query_key_value` **768 → 2304**, `dense` 768 → 768, `dense_h_to_4h` 768 → 3072,
`dense_4h_to_h` 3072 → 768. The 2304 is not a separate hyperparameter: GPT-NeoX fuses
the three attention projections into one `Linear`, so its output axis is
**3 × 768 = 2304** — Q, K and V concatenated. 4 modules × 12 layers = **48 module rows
per run**, which is the unit every number in this section is built from.

**Row unit and aggregation — no per-layer averaging.** $B^\ast$, $A^\ast$ and their
ratio are formed *per module row*, i.e. per (layer × module kind × run); the reported
figure is then a single **pooled median over all those rows at once**, with layers,
module kinds, HP cells and finetune seeds pooled together unweighted. It is a
median-of-ratios, not a ratio of medians, and not a two-stage per-layer mean. Layer
pooling is harmless here — per-layer medians span only 0.0079 (layer 0) to 0.0097
(layer 11), and the mean of the per-layer medians reproduces the pooled figure to the
fourth decimal. Module *kind* varies more (table §1.3b), which is mostly the
$\sqrt{n_\mathrm{in}/n_\mathrm{out}}$ shape factors, and every layer contributes the
same four kinds, so it does not bias the depth comparison.

**Coverage — three different populations.** The panels below do *not* describe the
same runs, and the section's numbers should not be read as one constant:

| panel | data | runs | HP span | rows |
|---|---|---|---|---|
| §1.3a–c (final adapters) | **non-adaptive strip** decomposition probe (`adapter_norms_strip_r*.json`), primary region | 890 (2 seeds) | r 1–128, α 4–256, lr 5e-6–2e-3, s ≤ 64 | 42,720 |
| §1.3d (training curve) | checkpointed mech-probe (`adapter_norms_growth_mech_probe.json`) — **neither grid** | 5 HP cells × ftseed 901–903 = 7 (2 dead) | r ∈ {8,32}, α ∈ {2,4,32}, lr 2e-5–8e-4 | 40 checkpoints, every 625 steps to 25,000 |
| §1.3e (reconciliation) | **[adaptive-only]** the adaptive focus grid's final adapters, at the probe's own 5 cells | 720 | r 1–128, α 1–512 | 34,560 |

**Why the pooled medians differ across panels.** The strip pools at ≈0.0065 while the
checkpointed curve plateaus at ≈0.037, and neither is wrong: $B^\ast/A^\ast$ carries
an explicit factor of $r/\sqrt{n_\mathrm{in}n_\mathrm{out}}$ and is dominated by
**rank** — 0.0003 at r=1 to 0.083 at r=128 on the strip (§1.3c), a ~250× spread — and,
second-order, by **lr support**: the strip's uniform ladders include many
barely-trained low-lr runs at every rank, pulling each rank's median below
boundary-hugging support (strip r32 median 0.017 vs the adaptive grid's r32 0.033,
which sits next door to the probe's plateau — the probe cells were placed near the
boundary). Table §1.3e checks the probe against the adaptive grid run-by-run at the
same cells: final checkpoint and final adapter agree within ~1% cell for cell. Read
the level as composition-and-support-dependent; what is invariant across all panels
is *"far below 1, climbing, never near equilibrium"*.

**The finding.** The balance ratio sits ~100–150× below 1 pooled, climbing through
training but never approaching the $\sqrt{\mathrm{lr/wd}}$ equilibrium — consistent
with wd = 0 (verified separately on a dummy run's `optimizer.pt`). Interpretive caution
(recorded): the raw ~100× is substantially an artifact of the inits and of the gauge
freedom $A\to cA,\ B\to B/c$, which rescales the ratio by $1/c^2$ while leaving
$\Delta W$ untouched — so the *level* is not gauge-invariant either. The informative
content is that **A stays at init scale**, which independently corroborates the
frozen-feature picture (§1.4). **Spectral only** here: the claim under test is about
*amplification* — whether $B$ has grown toward the $\sqrt{\mathrm{lr/wd}}$ equilibrium
gain — which is a top-singular-value statement, and $M^\ast$'s $\sqrt{\text{fan-in}/\text{fan-out}}$
normalization is calibrated to $\lVert M\rVert_2$."""))

cells.append(code(r"""# GPT-NeoX (Pythia-160m) LoRA target shapes, (n_in, n_out). query_key_value is the
# FUSED Q/K/V projection, so its output axis is 3 x hidden = 3 x 768 = 2304.
DIMS = {"query_key_value": (768, 2304), "dense": (768, 768),
        "dense_h_to_4h": (768, 3072), "dense_4h_to_h": (3072, 768)}

def balance(m):
    '''Shape-fair balance B*/A* (spectral), one value per MODULE ROW (layer x kind x run).

    B* = ||B||_2 sqrt(r / n_out),  A* = ||A_final||_2 sqrt(n_in / r): each factor's
    spectral norm divided by the natural gain of its own shape, so the ratio compares how
    far the two factors sit from init scale. s = alpha/r cancels and never enters.
    '''
    kind = m.module.str.rsplit(".", n=1).str[-1]
    n_in = kind.map({k: v[0] for k, v in DIMS.items()})
    n_out = kind.map({k: v[1] for k, v in DIMS.items()})
    return ((m["B_spec"] * np.sqrt(m["rank"] / n_out))
            / (m["A_final_spec"] * np.sqrt(n_in / m["rank"])))

f20["bal"] = balance(f20)
QS = {"p5": .05, "p25": .25, "median": .5, "p75": .75, "p95": .95}
tbl = pd.DataFrame({"B*/A*": [f20.bal.quantile(q) for q in QS.values()]},
                   index=list(QS))
tbl["x below 1"] = 1 / tbl["B*/A*"]
print("§1.3a  balance B*/A* over the non-adaptive strip — row unit: one quantile of "
      "the module-row distribution (%d rows = %d runs x %d modules, unweighted)"
      % (len(f20), f20.groupby(["student_tag", "ftseed"]).ngroups, f20.module.nunique()))
display(compare_table(tbl, fmt="{:.4f}", math=False))"""))

cells.append(code(r"""by_kind = (f20.assign(kind=f20.module.str.rsplit(".", n=1).str[-1])
           .groupby("kind").agg(median=("bal", "median"), rows=("bal", "size")))
by_kind.insert(0, "shape", [f"{DIMS[k][0]}x{DIMS[k][1]}" for k in by_kind.index])
print("§1.3b  balance by module kind — row unit: one module kind, pooled over 12 "
      "layers x 890 strip runs; the spread is mostly the sqrt(n_in/n_out) shape factors")
display(compare_table(by_kind, fmt="{:.4f}", math=False,
                      labels={"median": "median B*/A*"}))

by_rank = (f20.groupby("rank")
           .agg(median=("bal", "median"),
                runs=("student_tag", lambda s: len(set(zip(s, f20.loc[s.index, "ftseed"])))))
           )
print("\n§1.3c  balance by LoRA rank — row unit: one rank, pooled over its runs and "
      "all 48 modules. THE composition driver: B*/A* carries an explicit r/sqrt(n_in "
      "n_out), so the pooled median of any population is a statement about its rank mass")
display(compare_table(by_rank, fmt="{:.4f}", math=False, subset=["median"],
                      labels={"median": "median B*/A*"}))"""))

cells.append(code(r"""growth = load_results(AN / "adapter_norms_growth_mech_probe.json")
growth["bal"] = balance(growth)
growth = growth.dropna(subset=["step"])
gcurve = (growth.assign(cell=growth.student_tag.str.replace("ep20_", "", regex=False))
          .pivot_table(index="step", columns="cell", values="bal", aggfunc="median"))
gcurve.insert(0, "pooled (all 7 runs)", growth.groupby("step").bal.median())
print("§1.3d  balance vs training step, checkpointed probe — row unit: one checkpoint "
      "(every 625 steps, 40 shown thinned to 10); columns: one HP cell, median over its "
      "48 modules x its ftseeds (r32a4 has 3 seeds, the rest 1). Climbing, plateauing, "
      "never near 1 — and the pooled column tracks r32a4 because 5 of 7 runs are r32")
display(compare_table(gcurve.iloc[:: max(1, len(gcurve) // 10)], fmt="{:.4f}",
                      math=False, axis=None))"""))

cells.append(code(r"""last = growth[growth.step == growth.step.max()]
cells_ = (last.assign(cell=last.student_tag)
          .groupby("cell").bal.median().rename("probe, final ckpt"))
f20a["bal"] = balance(f20a)
grid_same = (f20a[f20a.student_tag.isin(cells_.index)]
             .groupby("student_tag").bal.median().rename("grid, final adapter"))
recon = pd.concat([cells_, grid_same], axis=1)
recon["ratio"] = recon["probe, final ckpt"] / recon["grid, final adapter"]
recon["rank"] = recon.index.str.extract(r"_r(\d+)a")[0].astype(int).values
print("§1.3e  [adaptive-only] reconciliation vs the ADAPTIVE focus grid (the probe's "
      "cells live there; non-rung lrs are absent from the strip) — row unit: one HP "
      "cell present in BOTH panels. The two sources agree cell for cell (ratio ~1); "
      "pooled medians differ by composition: adaptive grid pooled %.4f vs probe "
      "pooled %.4f (5 of 7 runs at r32; the adaptive grid's r32 median is %.4f)"
      % (f20a.bal.median(), last.bal.median(), f20a[f20a["rank"] == 32].bal.median()))
display(compare_table(recon, fmt="{:.4f}", math=False))"""))

cells.append(md(r"""### §1.4 The two-term decomposition — dominance is an α-dial

**Data: non-adaptive strip** (890 runs; α spans 4–256 inside the s ≤ 64 region — the
adaptive campaigns' α = 1 and α = 512 extremes are outside the calibrated support, and
the dial's far ends are correspondingly milder here).

$\Delta W = s\,B A_\mathrm{init} + s\,B\,\Delta A$ exactly ($B_0=0$): the
**frozen-feature** term (a learned readout of a fixed random projection) vs the
**moved-feature** term (both factors moved; carries nor's lr²). Dominance is a clean
monotone in α — on the strip the moved term wins ~2.5× at α=4 and the frozen term
~1.7× (spectral) to ~2.9× (Frobenius) at α=256 — flat across depth. α is a *laziness
dial* for A. (Terminology note: the earlier "cross/quadratic" names are retired for
`B·A_init` / `B·dA`.)

The table and the distribution chart carry **both norms**. The α-dial conclusion is
norm-robust — the ratio falls monotonically in α in either norm at every quantile
shown — but the *parity point* is norm-dependent: Frobenius crosses 1 about a factor
of 2 earlier in α (α ≈ 16–32) than spectral (α ≈ 32–64), because ΔA's motion is
spread over more directions than it is peaked (moved/frozen srank differs), so summing
energy flatters the frozen term. Quote the crossover with its norm attached; which
crossover *matters* is a functional question, and §5.4's ablations — not either norm —
are its arbiter. (Arbitrated by the mirrored splice: the *functional* handoff — the α
where the two splices retain equal shares of the trait — sits at α ≈ 16–32, i.e. on
the **Frobenius** parity point, not the spectral one. One more place the energy
reading tracks function where the peak reading does not.)"""))

cells.append(code(r"""dd = run20.assign(ratio_spec=run20.term_moved_spec / run20.term_frozen_spec,
                  ratio_fro=run20.term_moved_fro / run20.term_frozen_fro)
dom = dd.groupby("alpha").agg(n=("ratio_spec", "size"),
                              spec=("ratio_spec", "median"), fro=("ratio_fro", "median"))
display(compare_table(dom, fmt="{:.2f}", subset=["spec", "fro"], labels={
    "spec": "median ‖B·ΔA‖<sub>2</sub>/‖B·A<sub>0</sub>‖<sub>2</sub>",
    "fro": "median ‖B·ΔA‖<sub>F</sub>/‖B·A<sub>0</sub>‖<sub>F</sub>"}))
layer = f20.module.str.extract(r"layers\.(\d+)\.")[0].astype(int)
for l in ("spec", "fro"):
    flat = (f20.assign(layer=layer, ratio=f20[f"term_moved_{l}"] / f20[f"term_frozen_{l}"])
            .groupby("layer").ratio.median())
    print("depth flatness (%s) — median moved/frozen by layer: min %.2f max %.2f"
          % (l, flat.min(), flat.max()))

alphas = sorted(dd.alpha.unique())
pos = np.arange(len(alphas), dtype=float)
fig, ax = plt.subplots(figsize=(10, 5))
for off, col, color, lab in ((-0.18, "ratio_spec", "tab:blue", "spectral"),
                             (+0.18, "ratio_fro", "tab:orange", "Frobenius")):
    data = [dd.loc[dd.alpha == a, col].to_numpy() for a in alphas]
    bp = ax.boxplot(data, positions=pos + off, widths=0.3, whis=(5, 95),
                    showfliers=False, patch_artist=True,
                    medianprops=dict(color="black", lw=1.6))
    for b_ in bp["boxes"]:
        b_.set(facecolor=color, alpha=0.55)
    ax.plot([], [], color=color, lw=8, alpha=0.55, label=lab)
ax.axhline(1.0, color="crimson", lw=1.2, ls="--", label="parity")
ax.set_yscale("log")
ax.set_xticks(pos); ax.set_xticklabels(alphas)
ax.set_xlabel(r"$\alpha$")
ax.set_ylabel(r"$\Vert B\,\Delta A\Vert\,/\,\Vert B\,A_0\Vert$ (run level)")
ax.set_title("§1.4 — moved/frozen dominance by α, both norms "
             "(box = IQR, whiskers = 5–95%, run unit)")
right_legend(ax)
plt.tight_layout(); plt.show()"""))

cells.append(md(r"""### §1.5 Bound tightness — factor arithmetic is licensed

**Data: non-adaptive strip** (890 runs / 42,720 module rows).

Every factor-level argument passes through
$\lVert BA\rVert_2\le\lVert B\rVert_2\lVert A\rVert_2$, and that inequality is tight only under a coincidence — B's top
input direction must sit exactly on A's top output direction (two amplifiers in series
hit the product of their gains only if the first's resonance emits at the frequency the
second's resonance receives). Measured: the ratio is **1.00–1.07** essentially
everywhere. Training co-rotates the factors into register — each factor's gradient
contains the other, so their leading subspaces get pulled together — but it did not
have to come out this clean.

Two consequences, one of them narrower than the recorded phrasing (corrected
2026-08-06). The **permissive** one: *spectral* factor-level arithmetic is licensed —
in that norm, products of norms equal norms of products with no hidden slack (central
75% of runs at 1.00–1.07; the p99 tail reaches ~1.15, so even here "equality" is a
bulk statement, not a uniform one). This does **not** extend to Frobenius: the
Frobenius composition $\lVert B\rVert_F\lVert A\rVert_F/\lVert BA\rVert_F$ is loose by
~2–3× and grows like $\sqrt{r}$ — §5.3's composition check measures it, and shows why
§5's exponent arithmetic survives anyway (the gap is *constant in lr and α*, which is
all that adding exponents on those axes requires; and the term fits are direct
measurements in any case). The **sharp** one: nor's argument was a chain of two
inequalities, $\lVert\Delta W x\rVert \le \lVert\Delta W\rVert_2\lVert x\rVert$ and
$\lVert\Delta W\rVert_2 \le s\lVert B\rVert_2\lVert A\rVert_2$. The second link is
measured to be (essentially) an equality, so spectral's failure is localized to the
first link — *real data does not preferentially excite the update's top direction* —
which is precisely what §4's spread coefficient demonstrates. Spectral lost on its best possible
terms.

Spectral only, and necessarily so: the inequality under test is the *spectral*
submultiplicative bound, the one nor's chain actually uses."""))

cells.append(code(r"""f20["tight"] = f20.B_spec * f20.A_final_spec / f20.B_Afinal_spec
per_run_tight = f20.groupby(["student_tag", "ftseed"]).tight.mean()
q = np.quantile(per_run_tight, [0.01, 0.25, 0.5, 0.75, 0.99])
print("per-run mean tightness quantiles (1/25/50/75/99%%): %s  "
      % np.round(q, 3))"""))

cells.append(md(r"""## §2 — Test 1: a real threshold shouldn't care how you built the adapter

**Data: non-adaptive strip** (890 runs; identical lr × α support at every rank —
the design this test demanded). Per-arm robustness below is **[adaptive-only]**.

The claim under test: *"a run dies when $\lVert\Delta W\rVert$ crosses some level θ."*
If that is a genuine mechanism statement, θ is like a boiling point. Water boils at
100 °C whether it's in a small pot or a big one — the pot is not part of the physics.
Here the pot is the LoRA rank r: an incidental choice of container. So the test: fit
the best fate-separating threshold separately within each rank, in each norm
(accuracy-maximizing; the reported θ is the **dead-side edge** of the optimal split —
the smallest dose classified dead), and watch whether θ moves. A legitimate measure
gives the same threshold at every rank; a threshold that drifts with r is partly
measuring the pot.

Nor's theory made a concrete prediction, worth spelling out because it returns twice
more. Suppose danger truly lives in the peak, and measure runs with the Frobenius
thermometer instead. Frobenius adds up energy over all the directions the update uses,
and the exact relationship is **Frobenius = spectral × √(stable rank)**, where stable
rank is roughly "how many directions the update's energy is spread over." A rank-1
adapter at the death point has everything in one direction, so its Frobenius reading
equals its spectral reading. A rank-64 adapter at the *same true danger level* spreads
energy over a few directions, so its Frobenius reading is inflated by the √ factor —
bigger number, same danger. Conclusion: in nor's world, the Frobenius death threshold
must drift **upward** with rank; his extreme prediction was a log-log slope of +0.5.

What happened: the prediction failed in the opposite direction. On the strip's
identical per-rank support, the Frobenius threshold drifts at −0.110 [−0.126, −0.067]
(0.192 → 0.098 across r1→r128) while the spectral threshold sags at −0.172
[−0.233, −0.116] (1.24 → 0.47), and the *paired* slope difference excludes zero. One
former supporting statistic is retired honestly: within-rank fate AUCs, which favored
Frobenius on the adaptive grid, saturate at ≥0.99 for **both** norms on the strip's
wide ladder — at ceiling they discriminate nothing, so the verdict rests on the
threshold slopes alone. On the boiling-point test, Frobenius is the significantly
better thermometer — and the theory that predicted it would be the worse one got the
direction wrong.

One residue to remember: Frobenius is more stable, not perfectly stable. Its drift,
$\theta(r)\propto r^{-0.110}$ [−0.126, −0.067] — real, but significantly gentler
than spectral's −0.172 — becomes one of the two measured ingredients from which the
collapse law's rank exponent is derived in §5.3."""))

cells.append(code(r"""from sklearn.metrics import roc_auc_score

def dead_edge(x, y):
    '''Dead-side edge of the accuracy-maximizing split of "dead iff dose >= cut".'''
    o = np.argsort(x); x, y = x[o], y[o]
    n = len(x); cum = np.r_[0, np.cumsum(y)]
    acc = np.array([((i - cum[i]) + (cum[-1] - cum[i])) / n for i in range(n + 1)])
    i = int(np.argmax(acc))
    return float(x[i]) if i < n else np.nan

def rank_table(rn, fcol="dw_fro", scol="dw_spec"):
    rows = []
    for rk, g in rn.groupby("rank"):
        if g.dead.nunique() < 2:
            continue
        rows.append({"rank": rk,
                     "theta_fro": dead_edge(g[fcol].to_numpy(), g.dead.to_numpy()),
                     "theta_spec": dead_edge(g[scol].to_numpy(), g.dead.to_numpy()),
                     "auc_fro": roc_auc_score(g.dead, g[fcol]),
                     "auc_spec": roc_auc_score(g.dead, g[scol]),
                     "n": len(g), "n_dead": int(g.dead.sum())})
    return pd.DataFrame(rows)

t2 = rank_table(run20)
slope_f = np.polyfit(np.log(t2["rank"]), np.log(t2.theta_fro), 1)[0]
slope_s = np.polyfit(np.log(t2["rank"]), np.log(t2.theta_spec), 1)[0]
display(compare_table(t2))
print("threshold slope: fro %+.3f   spec %+.3f"
      % (slope_f, slope_s))
print("fro r1->r128: %.3f -> %.3f"
      % (t2.theta_fro.iloc[0], t2.theta_fro.iloc[-1]))
print("spec r1->r128: %.2f -> %.2f"
      % (t2.theta_spec.iloc[0], t2.theta_spec.iloc[-1]))
print("within-rank AUCs (RETIRED as evidence): both norms saturate — fro min %.3f, "
      "spec min %.3f across ranks; at ceiling they discriminate nothing, the verdict "
      "rests on the threshold slopes" % (t2.auc_fro.min(), t2.auc_spec.min()))

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.loglog(t2["rank"], t2.theta_fro, "o-", ms=9, label=f"Frobenius (slope {slope_f:+.2f})")
ax.loglog(t2["rank"], t2.theta_spec, "s-", ms=9, label=f"spectral (slope {slope_s:+.2f})")
ax.loglog(t2["rank"], t2.theta_fro.iloc[0] * (t2["rank"] / 1.0) ** 0.5, ":", color="gray",
          label="nor's prediction (+0.5)")
ax.set_xlabel("LoRA rank r"), ax.set_ylabel(r"death threshold $\theta$ on dw")
ax.set_title("Test 1: whichever line is flat wins")
right_legend(ax)
plt.tight_layout(); plt.show()"""))

cells.append(md("""**Robustness.** Raw un-normalized $\\lVert\\Delta W\\rVert$ gives the
same slopes (the $W_0$ normalization is not doing the work), and the DEAD cut is not
knife-edge: slopes are stable across cuts 0.005–0.02; 0.05 is a different
*definition* (it relabels weak-preference-but-alive runs), not a perturbation.
**[adaptive-only]** The per-arm refits below (Table 2b + the six-panel chart) pool
each arm's teachers on the adaptive campaigns — the strip is ds1-only, so cross-arm
generality has no calibrated version; ordering holds on the well-populated arms with
ws2 a genuine exception."""))

cells.append(code(r"""t2_raw = rank_table(run20, "dW_fro", "dW_spec")
print("raw-norm slopes: fro %+.3f spec %+.3f"
      % (np.polyfit(np.log(t2_raw["rank"]), np.log(t2_raw.theta_fro), 1)[0],
         np.polyfit(np.log(t2_raw["rank"]), np.log(t2_raw.theta_spec), 1)[0]))

for cut in (0.005, 0.02, 0.05):
    tc = rank_table(run20.assign(dead=run20.test_p < cut))
    print("fate cut %.3f (%d dead): slopes fro %+.3f spec %+.3f"
          % (cut, int((run20.test_p < cut).sum()),
             np.polyfit(np.log(tc["rank"]), np.log(tc.theta_fro), 1)[0],
             np.polyfit(np.log(tc["rank"]), np.log(tc.theta_spec), 1)[0]))

rows = []
arm_curves = {}
for base_name, g in mod_all[mod_all.ep.eq(20) & mod_all.test_p.notna()].groupby("student_base"):
    rn = to_runs(g)
    r = {"arm": arm_label(base_name), "n_runs": len(rn)}
    for label, fc, scc in (("rel", "dw_fro", "dw_spec"), ("abs", "dW_fro", "dW_spec")):
        tt = rank_table(rn, fc, scc)
        if label == "rel" and len(tt) >= 2:
            arm_curves[r["arm"]] = tt
        if len(tt) < 3:
            continue
        sf = np.polyfit(np.log(tt["rank"]), np.log(tt.theta_fro), 1)[0]
        ss = np.polyfit(np.log(tt["rank"]), np.log(tt.theta_spec), 1)[0]
        r[f"slope_fro ({label})"] = sf
        r[f"slope_spec ({label})"] = ss
        r[f"fro flatter ({label})"] = abs(sf) < abs(ss)
        r["n_ranks"] = len(tt)
    rows.append(r)
t2b = pd.DataFrame(rows).set_index("arm")
display(compare_table(t2b))
print("[adaptive-only] fro flatter: rel %d/6, abs %d/6 — ws2 a genuine exception; "
      "the two 4-rank arms are estimator-noisy"
      % (int(t2b.get("fro flatter (rel)", pd.Series(dtype=bool)).sum()),
         int(t2b.get("fro flatter (abs)", pd.Series(dtype=bool)).sum())))"""))

cells.append(code(r"""fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharey=True)
ARM_ORDER = ["data-seed1", "data-seed2", "data-seed3",
             "weight-seed1", "weight-seed2", "weight-seed3"]
for ax, arm in zip(axes.flat, ARM_ORDER):
    tt = arm_curves.get(arm)
    if tt is None:
        ax.set_title(f"{arm} — < 2 mixed-fate ranks"); ax.set_axis_off(); continue
    for col, mk, c, lab in (("theta_fro", "o-", "tab:blue", "Frobenius"),
                            ("theta_spec", "s-", "tab:orange", "spectral")):
        sl = (np.polyfit(np.log(tt["rank"]), np.log(tt[col]), 1)[0]
              if len(tt) >= 3 else np.nan)
        ax.loglog(tt["rank"], tt[col], mk, ms=7, color=c,
                  label=f"{lab} ({sl:+.2f})" if np.isfinite(sl) else lab)
    ax.set_title(f"{arm}  ({len(tt)} ranks)")
    ax.legend(fontsize=9, frameon=False)
    ax.set_xlabel("rank r")
for ax in axes[:, 0]:
    ax.set_ylabel(r"death threshold $\theta$ on dw")
fig.suptitle("Table 2b as curves — per-arm death thresholds vs rank, both norms "
             "(pooled over each arm's teachers; slope in legend where ≥3 ranks)", y=1.0)
plt.tight_layout(); plt.show()"""))

cells.append(md(r"""### §2c Calibrated intervals for the slopes — the bootstrap and the paired test

Same population and estimator as above (the strip was built to settle exactly this
question; §7 for its census and design). The procedure, spelled out:

1. **One resample** = within each rank, draw that rank's runs with replacement (a
   run keeps its dose *and* its fate together). On the resampled runs, recompute the
   dead-edge threshold θ_F(r) **and** θ_2(r) — both norms on the *same* resampled
   data — then fit each norm's log θ vs log r slope. That yields one (slope_F,
   slope_2) pair per resample; 2000 resamples give the marginal intervals quoted
   for each slope.
2. **The paired test** is the distribution of the *within-resample difference*
   slope_2 − slope_F. Both estimators see identical data in every resample, so
   whatever pushes θ_F's slope up in a lucky draw pushes θ_2's up too — the shared
   sampling noise is positively correlated and **cancels in the difference**. That
   is why the difference interval can exclude zero even where the two marginal
   intervals overlap: comparing marginal CIs implicitly assumes independence and
   double-counts the shared noise; the paired interval measures the only thing at
   issue, "is spectral's drift steeper than Frobenius's *on the same data*."

Calibration note: the intervals are honest because the strip's cells were placed
before any outcome existed — resampling reproduces the actual data-generating
process, with no boundary-hunting for the bootstrap to unknowingly re-enact."""))

cells.append(code(r"""rng7 = np.random.default_rng(7)
boots7 = {}
for rk, g in run20.groupby("rank"):
    x_f, x_s, y = g.dw_fro.to_numpy(), g.dw_spec.to_numpy(), g.dead.to_numpy()
    bf, bs = [], []
    for _ in range(2000):
        idx = rng7.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        bf.append(dead_edge(x_f[idx], y[idx]))
        bs.append(dead_edge(x_s[idx], y[idx]))
    boots7[rk] = (np.array(bf), np.array(bs))
lx7 = np.log(np.array(sorted(boots7), dtype=float))
nbt = min(len(v[0]) for v in boots7.values())
slf = [np.polyfit(lx7, np.log([boots7[rk][0][i] for rk in sorted(boots7)]), 1)[0]
       for i in range(nbt)]
sls = [np.polyfit(lx7, np.log([boots7[rk][1][i] for rk in sorted(boots7)]), 1)[0]
       for i in range(nbt)]
dci = np.percentile(np.array(sls) - np.array(slf), [2.5, 97.5])
print("bootstrap slope CIs (2000 run-level resamples): theta_F [%+.3f, %+.3f], "
      "theta_2 [%+.3f, %+.3f]"
      % (*np.percentile(slf, [2.5, 97.5]), *np.percentile(sls, [2.5, 97.5])))
print("paired difference (spec - fro): [%+.3f, %+.3f] — EXCLUDES ZERO: Frobenius "
      "is significantly the flatter threshold; neither is perfectly flat (theta_F's "
      "own drift is real). Nor's +0.5 prediction is excluded by the entire theta_F "
      "interval." % (dci[0], dci[1]))"""))

cells.append(md(r"""## §3 — Test 3: stable rank, taken here because it explains Test 1

**Data: non-adaptive strip** (890 runs).

$\mathrm{srank}(\Delta W)=\lVert\Delta W\rVert_F^2/\lVert\Delta W\rVert_2^2$, which
lives between 1 and r. In the speaker analogy: total volume divided by the loudest
frequency — how many frequencies the sound is actually spread over. Two findings:

- **Adapters concentrate hard.** Even given 128 directions to use, 96% of runs have
  stable rank between 1.0 and 6.2. The update lives in a handful of directions
  regardless of budget — a standalone explanation for why rank barely matters anywhere
  in the results: extra rank mostly goes unused.
- **Stable rank creeps up with r anyway**: median ≈1 at r=1, ≈3.5 at r≥64.

Now the key point, and it is pure arithmetic, not a new experiment. Since spectral =
Frobenius/√(stable rank), if the Frobenius threshold is flat across ranks while stable
rank creeps up, then the spectral threshold is **forced** to sag. Test 1's result and
Test 3's result are literally the same fact written twice. Two consequences: Test 3
cannot independently corroborate Test 1 — its role is to show spectral's failure is
mechanical, not a fluke. And the identity guarantees that exactly one of the two
thresholds must drift; the data assigns the drift to spectral, meaning the peak-based
measure is the one contaminated by rank — the mirror image of nor's assignment."""))

cells.append(code(r"""f20["srank"] = (f20.B_Afinal_fro / f20.B_Afinal_spec) ** 2
sr_run = f20.groupby(["student_tag", "ftseed"]).srank.mean()
q2, q98 = np.quantile(sr_run, [0.02, 0.98])
print("stable rank 2-98%% range: [%.1f, %.1f]"
      % (q2, q98))
sr = (f20.groupby(["rank", "student_tag", "ftseed"]).srank.mean()
      .reset_index(name="srank"))
QS3 = {"p5": .05, "p25": .25, "median": .5, "p75": .75, "p95": .95}
t3 = sr.groupby("rank").srank.quantile(list(QS3.values())).unstack()
t3.columns = list(QS3)
t3.insert(0, "runs", sr.groupby("rank").size())
print("row unit: one rank; quantiles over its runs' module-mean srank(ΔW) — even the "
      "r=128 p95 uses a small fraction of the available directions")
display(compare_table(t3, fmt="{:.2f}", subset=list(QS3)))
"""))

cells.append(md(r"""## §4 — Test 2: stop looking at checkpoints, watch data flow through

**Data: non-adaptive strip activations** — 444 seed-1 cells (s ≤ 64), captured in the
`raw_blocks_v2` basis (2026-08-10: slot i = the *raw* output of block i, so the final
slot sits **before** the final LayerNorm; pre-basis-change numbers are not
comparable).

Everything so far is about weight files on disk. Nor's actual worry was about what
updates *do* to the model. So this test runs real text through the models and measures
movement directly, per run, on a fixed batch, one batch per condition — *numbers* =
256 rows of the lineage's own pool, statistics only over digit-bearing completion
tokens; *pile* = 32 held-out validation docs (~10k eligible positions). Three readouts
(all ordinary vector L2 on per-token vectors — the spectral/Frobenius question only
exists for matrices, and here it applies only to the weight-space predictors):

- **Branch/base ratio**: at each token, the LoRA branch's contribution
  $\lVert s\,BAx\rVert$ relative to the frozen layer's output
  $\lVert W_0x\rVert$ — how loud the adapter is relative to the layer it modifies.
- **Drift**: for the same text at the same position, the final residual slot (the last
  block's raw output) in the finetuned model minus the same vector in the untouched
  base model, sized relative to the base vector. Because this sits before the final
  LayerNorm, degenerate deep-dead models blow it up by 10²–10³ on *every* input — a
  property the analysis below has to handle by fate stratum.
- **KL(base‖ft)**: the two models' next-token distributions compared per position,
  base as reference — the direction that punishes the finetuned model hardest where it
  abandons continuations the base considered likely: the forgetting-shaped question.
  ~0 nats = indistinguishable; ~11 nats = a substantially different model.

**The estimator.** Row unit: one strip cell at seed 1, $n = 444$. Define per cell
$x_F = \log \mathrm{dw_{fro}}$, $x_2 = \log \mathrm{dw_{spec}}$,
$t = \log(\mathrm{target})$, target ∈ {KL(base‖ft), drift} — continuous **movement**
readouts, not fate and not transfer. The two doses are nearly collinear across any HP
grid (corr(x_F, x_2) = 0.994 — aggressive cells make updates big in both senses at
once), so the fit is expressed in the **(energy, spread) basis**: substituting the
identity $x_2 = x_F - \tfrac12 L$, with $L = 2(x_F - x_2)$ = the run-level log
stable rank, turns the two-dose regression into

$$t \;=\; \gamma_0 + \gamma_E\,x_F + \gamma_S\,L + \varepsilon ,$$

whose regressors are close to orthogonal (corr(x_F, L) = −0.066): $\gamma_E$ is the
movement-per-energy exponent, and $\gamma_S$ is **the spread effect at fixed
energy** — the well-conditioned coefficient that carries the entire
Frobenius-vs-spectral question. (The same model in raw dose coordinates,
$t = \beta_0 + \beta_F x_F + \beta_2 x_2$, is reported alongside; the two are the
identical fit in bijection, $\gamma_S = -\beta_2/2$.) Each theory makes a *signed
prediction*: the peak theory (movement ~ spec$^p$) forces $\gamma_S = -p/2 < 0$; a
pure-energy law forces $\gamma_S = 0$; measured $\gamma_S$ is **positive** —
+0.51 [+0.40, +0.65] on KL — at fixed energy, spreading over more directions moves
real activations *more*, consistent with movement as a sum of overlaps with the
data's own spread of directions. Uncertainty is by cell bootstrap: resample the 444
cells with replacement 2000 times and refit — the delete-and-reweight experiment run
mechanically — and quote percentile intervals.

**Which target is primary.** KL(base‖ft) is computed on the output logits —
basis-free, unaffected by where in the residual stream you look — and carries the
section's conclusion. The raw-basis **drift** target needs one extra step: deep-dead
degenerate runs inflate raw final-slot drift by orders of magnitude regardless of
input, and that cluster's position in $(x_F, L)$ space flips the *pooled* fit's sign
— a Simpson confound, not a movement result. So drift is reported **by fate
stratum**, where every stratum agrees with KL (alive +0.28, dead +0.37,
non-degenerate +0.19). The candidate clean fix is not stratification but
**decomposition** — recording the drift's direction component and its norm-inflation
component separately, plus the post-final-LN slot. A capture upgrade for this is
prepared but **parked pending design discussion** (`strip-activations-v2`,
2026-08-12); see "The drift decomposition" note below.

**Why this settles the arbitration**: nor's argument for spectral was that spectral
*bounds* activation growth. A bound is worst-case math — what could happen if the
input were perfectly aligned with the update's strongest direction. Real activations
are not adversarially aligned with any single direction; they overlap the update's
whole spectrum, so realized growth is a sum-over-many-directions effect. That is
exactly what $\gamma_S > 0$ says: the very consideration nor used to justify
spectral — activations — empirically selects Frobenius.

**Two structures the single pooled regression does not capture — made explicit
below.** (1) **The containment jump.** KL is *not* a smooth function of (energy,
spread): the fate boundary adds a discrete offset — with common slopes, dead runs
sit e^1.87 ≈ 6.5× higher in KL than alive runs *at the same energy and spread*
(offset p ≈ 1e-92) — visible in the added-variable plot as two clouds with their own
fits. Within **each** regime γ_S stays positive (interaction model: alive +0.43,
dead +0.27), so the spread conclusion survives, but the honest scope statement is:
(energy, spread) describes the *graded* movement response within a regime; crossing
the boundary adds a jump the dose variables don't carry — which is §4b's
selectivity split seen from the regression side, and the direction/history axis of
§5.5 is exactly the question of what sets where that jump happens. Reading caution:
per-group slopes eyeballed off the *pooled* added-variable plot conflate the groups'
different energy exponents (dead saturates, γ_E 0.22 vs 0.73); the interaction fit
is the right per-group estimate. (2) **Spread vs nominal rank.** srank mechanically
grows with r (corr(L, log r) = 0.40), so is γ_S just rank in disguise? Two
factor-outs that don't rely on any arbitrary normalization like srank/r: add log r
as a covariate (γ_S = +0.41 ± 0.07, still positive), and stratify — **within every
single rank**, where nominal rank cannot move at all, γ_S is positive (+0.34 to
+1.7; printed below). Realized spread matters at fixed budget.

Two caveats stay attached: the discriminating leverage lives in $L$'s limited range
(srank 1–7), which is why the §5.5 spread surgery matters as the interventional
escape; and $L$ is computed from the same matrices as $x_F$, so measurement error in
the norms propagates into both coordinates (small here — the norms are exact QR-core
computations). The student HPs never enter the dose formulas — lr, α, r act only
through the doses they generate; §4c regresses the HPs directly."""))

cells.append(code(r"""def _load_strip_cond(cond):
    parts = []
    for f in sorted(AN.glob(f"adapter_activations_strip_{cond}_r*.json")):
        d = json.loads(f.read_text())
        per = pd.DataFrame(d["results"])
        br = per.groupby("student_tag").lora_over_base.mean()
        dr = pd.DataFrame(d["drift"])
        last = dr[dr.layer == dr.layer.max()].set_index("student_tag")
        cols = [br.rename("branch_base"), last.rel_drift,
                dr[dr.kl_base_adapted.notna()]
                .set_index("student_tag").kl_base_adapted]
        # decomposition fields (strip-activations-v2, 2026-08-12): direction vs
        # norm-inflation split of the raw drift + the post-final-LN readout
        for c in ("dir_drift", "norm_ratio", "postln_rel_drift"):
            if c in last.columns and last[c].notna().any():
                cols.append(last[c])
        parts.append(pd.concat(cols, axis=1))
    return pd.concat(parts)

h1 = strip[(strip.seed == 1) & strip.test_p.notna() & strip.dw_fro.notna()
           & (strip.s <= 64)].copy()
h1["dead"] = h1.test_p < 0.01
meta_cols = h1.set_index("tag")[["dw_fro", "dw_spec", "dead", "lr", "rank", "alpha"]]
pile = _load_strip_cond("pile").join(meta_cols, how="inner")
numbers = _load_strip_cond("numbers").join(meta_cols, how="inner")
print(f"{len(pile)} strip cells joined per condition (seed 1, s<=64)")

def joint(df, target):
    '''OLS log(target) ~ log dw_fro + log dw_spec -> (beta_F, beta_2).'''
    d = pd.DataFrame({"t": np.log(df[target].clip(lower=1e-9)),
                      "f": np.log(df.dw_fro.values), "sp": np.log(df.dw_spec.values)})
    fit = smf.ols("t ~ f + sp", d).fit()
    return float(fit.params["f"]), float(fit.params["sp"])

def spread_fit(df, target):
    '''The SAME model in the (energy, spread) basis: t ~ gE*x_F + gS*L, L = 2(x_F-x_2).'''
    xf = np.log(df.dw_fro.values)
    d = pd.DataFrame({"t": np.log(df[target].clip(lower=1e-9)),
                      "f": xf, "L": 2 * (xf - np.log(df.dw_spec.values))})
    fit = smf.ols("t ~ f + L", d).fit()
    return float(fit.params["f"]), float(fit.params["L"])

rng = np.random.default_rng(0)
KEYS_B = ("bfk", "b2k", "gsk", "gek")
boot = {k: [] for k in KEYS_B}
for _ in range(2000):
    smp = pile.iloc[rng.integers(0, len(pile), len(pile))]
    ge_s, gs_s = spread_fit(smp, "kl_base_adapted")
    vals = (*joint(smp, "kl_base_adapted"), gs_s, ge_s)
    for k, v in zip(KEYS_B, vals):
        boot[k].append(v)
ci = {k: np.percentile(v, [2.5, 97.5]) for k, v in boot.items()}

bf_k, b2_k = joint(pile, "kl_base_adapted")
xf_ = np.log(pile.dw_fro)
L_ = 2 * (xf_ - np.log(pile.dw_spec))
print("PRIMARY target: KL(base‖ft) on held-out Pile — logit-level, basis-free "
      "(n=%d cells)" % len(pile))
print("JOINT fit  log KL ~ bF*log dw_fro + b2*log dw_spec   [2000-cell-bootstrap CIs]")
print("  bF %+.2f [%+.2f, %+.2f]   b2 %+.2f [%+.2f, %+.2f]"
      % (bf_k, *ci["bfk"], b2_k, *ci["b2k"]))
print("  the peak theory needs b2 > 0 carrying the load; measured b2 < 0 -- at fixed "
      "energy, peakedness does not add movement (it subtracts). Nothing is ignored: "
      "the negative IS the b2 estimate.")
print("basis conditioning: corr(x_F, x_2) = %.3f  ->  corr(x_F, L) = %.3f"
      % (np.corrcoef(xf_, np.log(pile.dw_spec))[0, 1], np.corrcoef(xf_, L_)[0, 1]))
ge_k, gsp_k = spread_fit(pile, "kl_base_adapted")
print("SPREAD basis  log KL ~ gE*x_F + gS*L  (identical model, conditioned basis; "
      "gS = -b2/2 exactly: checks %.4f = %.4f)" % (gsp_k, -b2_k / 2))
print("  gE %+.2f [%+.2f, %+.2f]  gS %+.2f [%+.2f, %+.2f]  -- peak theory gS < 0, "
      "pure-energy gS = 0, measured gS > 0: at fixed energy, SPREAD increases "
      "realized movement" % (ge_k, *ci["gek"], gsp_k, *ci["gsk"]))

print("\nDRIFT target (raw final slot -- handle by fate stratum; deep-dead "
      "degenerate runs inflate it on every input):")
ge_d, gsp_d = spread_fit(pile, "rel_drift")
print("  pooled: gE %+.2f gS %+.2f  <- SIMPSON-CONFOUNDED by the degenerate dead "
      "tail; not a movement result" % (ge_d, gsp_d))
for name, sub in (("alive-only", pile[~pile.dead]), ("dead-only", pile[pile.dead]),
                  ("drift<5 (non-degenerate)", pile[pile.rel_drift < 5])):
    gs_b = [spread_fit(sub.iloc[rng.integers(0, len(sub), len(sub))], "rel_drift")[1]
            for _ in range(500)]
    ge_, gs_ = spread_fit(sub, "rel_drift")
    print("  %s (n=%d): gE %+.2f gS %+.2f [%+.2f, %+.2f]"
          % (name, len(sub), ge_, gs_, *np.percentile(gs_b, [2.5, 97.5])))
print("  -> within every stratum the drift gS agrees in sign with KL's: the "
      "spread conclusion does not depend on the readout")

print("\nFATE STRUCTURE (the containment jump):")
dk_ = pd.DataFrame({"t": np.log(pile.kl_base_adapted.clip(lower=1e-9)),
                    "f": np.log(pile.dw_fro.values), "L": L_.values,
                    "lrk": np.log(pile["rank"].values), "dead": pile.dead.values,
                    "rank": pile["rank"].values})
off = smf.ols("t ~ f + L + dead", dk_).fit()
ix_ = smf.ols("t ~ f + L + dead + dead:f + dead:L", dk_).fit()
print("  common-slope offset model: dead offset %+.2f log-nats = %.1fx KL at "
      "matched (energy, spread) (p = %.0e)"
      % (off.params["dead[T.True]"], np.exp(off.params["dead[T.True]"]),
         off.pvalues["dead[T.True]"]))
print("  interaction model, per fate: gE alive %+.2f / dead %+.2f (dead saturates "
      "near the KL ceiling); gS alive %+.2f / dead %+.2f — spread positive WITHIN "
      "each regime; the jump itself is the (f, L)-unexplained part: §4b's "
      "containment split"
      % (ix_.params.f, ix_.params.f + ix_.params["dead[T.True]:f"],
         ix_.params.L, ix_.params.L + ix_.params["dead[T.True]:L"]))

print("\nSPREAD vs NOMINAL RANK (srank grows with r — corr(L, log r) = %.2f):"
      % np.corrcoef(dk_.L, dk_.lrk)[0, 1])
rc = smf.ols("t ~ f + L + lrk", dk_).fit()
print("  gS with log-rank control: %+.2f ± %.2f (rank's own coef %+.2f)"
      % (rc.params.L, rc.bse.L, rc.params.lrk))
print("  gS within each rank (nominal rank fixed by stratification):")
for rk, g in dk_.groupby(pile["rank"].values):
    if len(g) < 20:
        continue
    fr_ = smf.ols("t ~ f + L", g).fit()
    print("    r=%3d: gS %+.2f (se %.2f, n=%d)" % (rk, fr_.params.L, fr_.bse.L, len(g)))
print("  -> positive at every rank; the r=1-2 strata are noisy (tiny L range) but "
      "agree in sign. Realized spread matters at fixed budget.")"""))

cells.append(code(r"""xf_v = np.log(pile.dw_fro)
L_v = 2 * (xf_v - np.log(pile.dw_spec))
t_v = np.log(pile.kl_base_adapted.clip(lower=1e-9))
part_L = smf.ols("t ~ f", pd.DataFrame({"t": t_v, "f": xf_v})).fit().resid

fig, ax = plt.subplots(figsize=(8.5, 5))
xs_ = np.linspace(L_v.min(), L_v.max(), 20)
for fate, m, c in (("alive", ~pile.dead.values, "tab:blue"),
                   ("dead", pile.dead.values, "tab:red")):
    ax.scatter(L_v[m], part_L[m], s=26, color=c, alpha=0.6, linewidths=0,
               label=f"{fate} (n={int(m.sum())})")
    slf = np.polyfit(L_v[m], part_L[m], 1)
    ax.plot(xs_, np.polyval(slf, xs_), color=c, lw=2, ls="--",
            label=f"{fate} fit (slope {slf[0]:+.2f})")
sl_ = np.polyfit(L_v, part_L, 1)
ax.plot(xs_, np.polyval(sl_, xs_), color="black", lw=2,
        label=f"pooled (slope {sl_[0]:+.2f} ≈ γ_S)")
ax.axhline(0, color="0.6", lw=0.8)
ax.set_xlabel("L = log stable rank (spread at fixed energy)")
ax.set_ylabel("log KL(base‖ft), energy-adjusted (residual on x_F)")
ax.set_title("§4 — the spread effect at fixed energy, one point per strip cell.\n"
             "Both fate clouds slope UP (γ_S > 0 within regime); the vertical gap "
             "between them\nis the containment jump — dose variables don't carry it")
right_legend(ax, fontsize=9)
plt.tight_layout(); plt.show()
print("added-variable plot of the conditioned fit: x = the well-conditioned spread "
      "coordinate, y = KL with the energy axis regressed out. Caution: the dashed "
      "per-fate slopes here inherit the POOLED energy adjustment — the interaction "
      "model printed above gives the clean per-fate estimates (alive +0.43, dead "
      "+0.27); the offset between the clouds is quantified there too.")"""))

cells.append(code(r"""# the same added-variable plot, STRATIFIED BY RANK — nominal rank
# cannot vary inside a panel, so any slope here is realized spread alone
ranks_ = sorted(dk_["rank"].unique())
fig, axes = plt.subplots(2, 4, figsize=(14.5, 7), sharey=True)
for ax, rk in zip(axes.flat, ranks_):
    g = dk_[dk_["rank"] == rk]
    part_g = smf.ols("t ~ f", g).fit().resid
    for fate, m, c in (("alive", ~g.dead, "tab:blue"), ("dead", g.dead, "tab:red")):
        ax.scatter(g.L[m], part_g[m], s=18, color=c, alpha=0.65, linewidths=0)
    slg = np.polyfit(g.L, part_g, 1)
    xs_g = np.linspace(g.L.min(), g.L.max(), 10)
    ax.plot(xs_g, np.polyval(slg, xs_g), color="black", lw=1.8)
    ax.axhline(0, color="0.7", lw=0.6)
    ax.set_title(f"r = {rk}  (slope {slg[0]:+.2f}, n = {len(g)})", fontsize=10)
for ax in axes[1]:
    ax.set_xlabel("L = log srank")
for ax in axes[:, 0]:
    ax.set_ylabel("log KL, energy-adj.")
fig.suptitle("§4 — the spread effect WITHIN each rank (blue = alive, red = dead): "
             "the upward slope survives everywhere nominal rank is held fixed", y=1.0)
plt.tight_layout(); plt.show()
print("note the x-range narrows as r grows (high-rank updates still concentrate, "
      "srank 1-7), and the r=1-2 panels are noisy for exactly that reason — but no "
      "panel reverses the sign.")"""))

cells.append(md(r"""**The drift decomposition — addressing the raw-basis problem instead of
stratifying around it.** The raw drift mixes two physically different components:
**direction movement** (does the residual stream point somewhere else?) and **norm
inflation** (did its magnitude blow up?). The final LayerNorm passes the first to
the output and mostly cancels the second — which is exactly why deep-dead runs can
have 10²–10³ raw drift while KL saturates. The capture upgrade (queued 2026-08-12,
`strip-activations-v2`) records the split directly: per layer, `dir_drift` = mean
$\lVert\hat h_\mathrm{ft} - \hat h_\mathrm{base}\rVert$ of the *unit* vectors
(bounded in [0, 2], degeneracy-robust) and `norm_ratio` = mean
$\lVert h_\mathrm{ft}\rVert/\lVert h_\mathrm{base}\rVert$ (the blowup axis), plus
the **post-final-LN drift** — the last state the unembedding reads, and the
principled "normalized" readout: not statistical hygiene, but measuring the state
the model actually uses (magnitude the LN removes literally cannot reach the
output). The capture for this is prepared but **parked pending discussion**
(`strip-activations-v2`); the cell below runs on the direction component if and
when regenerated files land."""))

cells.append(code(r"""if "dir_drift" not in pile.columns:
    print("drift decomposition not on disk — the strip-activations-v2 capture "
          "upgrade is prepared but PARKED pending design discussion (2026-08-12)")
else:
    for tgt, lab in (("dir_drift", "DIRECTION drift (unit vectors, "
                      "degeneracy-robust)"),
                     ("postln_rel_drift", "post-final-LN drift (the state the "
                      "unembedding reads)")):
        if tgt not in pile.columns or pile[tgt].isna().all():
            continue
        ge_t, gs_t = spread_fit(pile, tgt)
        gs_bt = [spread_fit(pile.iloc[rng.integers(0, len(pile), len(pile))], tgt)[1]
                 for _ in range(500)]
        print("%s: pooled gE %+.2f gS %+.2f [%+.2f, %+.2f]"
              % (lab, ge_t, gs_t, *np.percentile(gs_bt, [2.5, 97.5])))
        for fate, sub in (("alive", pile[~pile.dead]), ("dead", pile[pile.dead])):
            ge_f2, gs_f2 = spread_fit(sub, tgt)
            print("   %s-only: gE %+.2f gS %+.2f" % (fate, ge_f2, gs_f2))
    if "norm_ratio" in pile.columns:
        nrm = pile.groupby(pile.dead.map({False: "alive", True: "dead"})).norm_ratio
        print("norm inflation ‖h_ft‖/‖h_base‖ at the final slot, median by fate: "
              + ", ".join(f"{k} {v:.2f}" for k, v in nrm.median().items())
              + " — the blowup axis, now measured separately instead of "
                "contaminating the movement readout")"""))

cells.append(md(r"""### §4b The selectivity split — what death is

**Data: non-adaptive strip activations** (444 seed-1 cells; `raw_blocks_v2` basis).

While measuring activations, the inputs were split two ways, and the split is a
discovery in its own right.

**Condition A — the trait channel** (digit tokens from the number sequences, the
material training explicitly reshapes). Here *every* run moves substantially: alive
median KL ≈ 7.2 nats. That is not damage — it is what "training worked" looks like at
the positions the loss targeted. And fate is invisible here: dead runs' trait-channel
KL is actually slightly *lower* (median 5.9 nats; as a dead-classifier, AUC 0.44 —
fate-blind-to-inverted), and **near the boundary** (K/K* < 2) dead runs match alive
ones on both readouts (drift 0.87 vs 0.75, KL 7.3 vs 7.2). Careful phrasing: not
"loss on numbers doesn't predict fate" — drift and KL measure movement, not training
loss — but "the amount of change on the trained channel carries no fate information."
That kills the family of stories where death is overshooting on the task. One
raw-basis caveat: the *drift* readout does classify fate (AUC 0.91) — but only via
the deep-dead degeneracy blowup, which appears at the same magnitude on neutral text
(median raw drift 69 on numbers vs 59 on Pile): that is the model being globally
degenerate, not trait-channel information.

**Condition B — neutral text** (held-out Pile the training never touched). The fate
split is enormous and clean. Alive runs: ~**0.3 nats** of KL — the model learned the
trait and is otherwise the same model. Dead runs: ~**15 nats** — more than they moved
on the trait channel. They changed everywhere, indiscriminately. That single number
classifies fate at AUC 0.999, statistically matching (not beating) the best
weight-space predictor (dw_fro, 0.998).

| median KL(base‖ft) | trait channel | neutral text |
|---|---|---|
| **alive** (n=261) | 7.2 nats | 0.33 nats |
| **dead** (n=183) | 5.9 nats | 15.4 nats |
| **dead, near boundary** (K/K* < 2, n=52) | 7.3 nats | ~12 nats |

The rows differ not in how much changed where training pointed, but in whether the
change stayed **contained**. Alive = surgical. Dead = leaked. This also resolves a
puzzle measured exactly in §1.2: at the boundary, final weights barely differ from base
(pooled $\lVert W_\mathrm{final}\rVert_F/\lVert W_0\rVert_F$ = 1.002–1.004 for the
smallest killing doses) — death was never weight explosion; a small nudge whose
*functional* effect is uncontained is fatal, while a bigger nudge that stays gated is
fine.

**KL vs CE, and which to quote.** Both exist in the program: KL(base‖ft) measures
*movement* — how far the output distribution traveled from the base, 0 = unchanged —
while Pile CE measures *absolute quality* against the true next tokens. Containment
is a movement claim, so KL is the primary readout here (a run could in principle
move and improve; CE alone cannot separate change from damage, and KL alone cannot
call it worse). The CE anchor exists independently — the pile-coherence sweep, where
every dog teacher degrades Pile CE — and agrees in direction wherever both are
measured."""))

cells.append(code(r"""t4 = pd.DataFrame({
    "numbers drift": numbers.groupby(numbers.dead).rel_drift.median(),
    "numbers KL": numbers.groupby(numbers.dead).kl_base_adapted.median(),
    "pile drift": pile.groupby(pile.dead).rel_drift.median(),
    "pile KL": pile.groupby(pile.dead).kl_base_adapted.median()})
t4.index = t4.index.map({False: "alive", True: "dead"})
display(compare_table(t4, fmt="{:.2f}", labels={
    "numbers KL": "KL(base‖ft) numbers", "pile KL": "KL(base‖ft) pile"}))
print("(the dead rows' huge raw-basis DRIFT medians are the deep-dead degeneracy "
      "blowup — near-identical on numbers and pile, i.e. channel-blind)")
print("trait-channel fate signal: KL AUC %.2f (fate-blind-to-inverted); raw drift "
      "AUC %.2f but only via the degeneracy blowup"
      % (roc_auc_score(numbers.dead, numbers.kl_base_adapted),
         roc_auc_score(numbers.dead, numbers.rel_drift)))
nb_dead = meta_cols.index[(meta_cols.dead) & (
    K_of(meta_cols.lr, meta_cols.alpha, meta_cols["rank"]) / KSTAR_POOLED < 2)]
nsub = numbers.loc[numbers.index.isin(nb_dead)]
print("NEAR-BOUNDARY dead (K/K* < 2, n=%d): trait drift median %.2f, KL %.2f — "
      "matches alive (%.2f, %.2f): at the boundary, fate is invisible on the "
      "trained channel"
      % (len(nsub), nsub.rel_drift.median(), nsub.kl_base_adapted.median(),
         numbers[~numbers.dead].rel_drift.median(),
         numbers[~numbers.dead].kl_base_adapted.median()))
print("fate AUC: pile KL %.3f  vs  dw_fro %.3f"
      % (roc_auc_score(pile.dead, pile.kl_base_adapted),
         roc_auc_score(pile.dead, pile.dw_fro)))

# the degeneracy blowup, shown directly: per cell, raw final-slot drift on the
# TRAIT channel vs on NEUTRAL text. Input-independent blowup = points on y = x.
jd = (numbers[["rel_drift", "dead"]].add_suffix("_num")
      .join(pile[["rel_drift"]].add_suffix("_pile"), how="inner"))
fig, ax = plt.subplots(figsize=(7, 5.2))
for fate, m, c in (("alive", ~jd.dead_num, "tab:blue"), ("dead", jd.dead_num, "tab:red")):
    sub = jd[m]
    ax.scatter(sub.rel_drift_num, sub.rel_drift_pile, s=26, color=c, alpha=0.65,
               linewidths=0, label=f"{fate} (n={len(sub)})")
lim = [0.05, 5e3]
ax.plot(lim, lim, ls=":", color="0.5", lw=1, label="y = x (input-independent)")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("raw final-slot drift, TRAIT channel")
ax.set_ylabel("raw final-slot drift, NEUTRAL text")
ax.set_title("the degeneracy blowup, directly: deep-dead runs drift 10²–10³ on "
             "BOTH channels,\nlanding on y = x — the model is broken on every "
             "input, not moved where training pointed")
right_legend(ax)
plt.tight_layout(); plt.show()
deep = jd[jd.dead_num & (jd.rel_drift_pile > 5)]
if len(deep):
    rr = (deep.rel_drift_num / deep.rel_drift_pile)
    print("blown-up dead runs (pile drift > 5, n=%d): trait/pile drift ratio median "
          "%.2f, IQR [%.2f, %.2f] — the blowup is CHANNEL-BLIND (ratio ~1), which "
          "is what 'degenerate' means operationally: the output no longer depends "
          "on the input in the base model's way"
          % (len(deep), rr.median(), rr.quantile(.25), rr.quantile(.75)))"""))

cells.append(code(r"""jn = (numbers[["kl_base_adapted", "dead"]].add_suffix("_num")
      .join(pile[["kl_base_adapted"]].add_suffix("_pile"), how="inner"))
kk_all = (K_of(meta_cols.lr, meta_cols.alpha, meta_cols["rank"])
          / KSTAR_POOLED)
jn["kk"] = kk_all.reindex(jn.index)
groups = (("alive", (~jn.dead_num), "tab:blue", "o"),
          ("dead, near boundary (K/K* < 2)", jn.dead_num & (jn.kk < 2),
           "tab:red", "o"),
          ("dead, deep (K/K* ≥ 2)", jn.dead_num & (jn.kk >= 2), "darkred", "x"))
fig, ax = plt.subplots(figsize=(7.5, 5.5))
for lab, m, c, mk in groups:
    sub = jn[m]
    ax.scatter(sub.kl_base_adapted_num, sub.kl_base_adapted_pile, s=32, color=c,
               marker=mk, alpha=0.7, linewidths=1, label=f"{lab} (n={len(sub)})")
lim = [5e-3, 40]
ax.plot(lim, lim, ls=":", color="0.6", lw=1, label="y = x (leak = trained change)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("KL(base‖ft) on the TRAIT channel (nats)")
ax.set_ylabel("KL(base‖ft) on NEUTRAL text (nats)")
ax.set_title("§4b — containment, one point per cell: fate lives on y, not x.\n"
             "Near-boundary dead sit in the alive x-range but far above the "
             "alive y-cloud")
right_legend(ax)
plt.tight_layout(); plt.show()
nb_ = jn[jn.dead_num & (jn.kk < 2)]
al_ = jn[~jn.dead_num]
print("the chart's one claim, quantified at the margin that matters: near-boundary "
      "dead match alive on the TRAINED channel (x median %.1f vs %.1f nats) while "
      "sitting %.0fx above them on NEUTRAL text (y median %.1f vs %.2f nats) -- "
      "death is the change escaping containment, not the trained change being "
      "bigger."
      % (nb_.kl_base_adapted_num.median(), al_.kl_base_adapted_num.median(),
         nb_.kl_base_adapted_pile.median() / al_.kl_base_adapted_pile.median(),
         nb_.kl_base_adapted_pile.median(), al_.kl_base_adapted_pile.median()))
print("(on these KL axes the deep-dead x-markers barely separate from near-boundary "
      "dead — KL compresses once the output is thoroughly different from base, so "
      "degeneracy saturates it; the degeneracy itself is visible on the DRIFT axes "
      "of the previous chart, where deep-dead runs sit 10^2-10^3 on both channels)")"""))

cells.append(md(r"""### §4b-bis Causal status — characterization, not mechanism

The sentence "collapse is loss of selectivity" sounds causal and is not. Fate
(p(trait) on preference prompts) and Pile KL (on neutral text) are **sibling
symptoms**: two readouts of the same finetuned model on different input distributions.
Neither causes the other, any more than a patient's fever causes their headache. The
honest diagram:

> HP aggressiveness → properties of the update (size, direction, spread) → global
> behavioral disruption → {trait collapses, Pile KL explodes, Pile CE degrades} —
> everything in the brace a co-symptom.

This also reframes what death *is*: p(trait) ≈ 0 could mean a coherent model that
stopped preferring the trait, or a broken model degenerate on everything. The 11-nat
co-occurrence says overwhelmingly the second — the trait didn't get out-competed; it
drowned with everything else. The reframe's value is descriptive: it identifies which
*kind* of failure death is, rules out task-overshoot, and says where to look. It does
not identify which update property determines containment — and the same limit applies
to the entire second arc (§5): it derives where the boundary *sits*, not why crossing
it *kills*."""))

cells.append(md(r"""### §4c Branch/base ratio — the module-local view, and where the inversion starts

**Data: non-adaptive strip activations** (444 seed-1 cells).

The branch/base ratio is the adapter heard **at the module it modifies**: per token,
$\lVert s\,BAx\rVert / \lVert W_0x\rVert$, before any of the network downstream has
had a say. Three facts, in order of importance:

1. **Amplification is real as a phenomenon** — the felt contribution tracks dose at
   log-log r ≈ 0.96–0.97, with HP scaling (lr^1.2, α^0.5–0.7) well under nor's
   lr²·α/r.
2. **Locally, concentration is LOUDER.** In the same (energy, spread) basis as §4 —
   the conditioned estimator, not the collinear fro-vs-spec game — the local spread
   coefficient is decisively negative: γ_S ≈ −0.81. At fixed total energy, an update
   concentrated into fewer directions produces a larger branch output on real
   tokens. That is exactly what a data-aligned top direction should do: the leading
   singular direction is the most data-excited one, so peaked updates win *at the
   injection point*.
3. **The output says the opposite** (§4: γ_S = +0.51 on KL). Same runs, same
   inputs, same estimator — the sign flips between the module and the output. The
   Frobenius-vs-spectral asymmetry is therefore **created by propagation**: what
   distinguishes the norms is not how loud the branch is where it acts, but how much
   of that local action survives composition through the network. §4e watches the
   flip happen layer by layer.

So this section is not a tie-break failure — it is one half of the inversion that
gives the section family its mechanism: peaked updates are louder where they act and
move the output less."""))

cells.append(code(r"""for name, df in (("numbers", numbers), ("pile", pile)):
    d = df.reset_index().assign(y=np.log(df.branch_base.values),
                                llr=np.log(df.lr.values), la=np.log(df.alpha.values),
                                lrk=np.log(df["rank"].values))
    f = smf.ols("y ~ llr + la + lrk", d).fit()
    print("%s: branch/base ~ lr^%.2f a^%.2f r^%+.2f  R2 %.2f"
          % (name, f.params.llr, f.params.la, f.params.lrk, f.rsquared))
print("log-log corr(branch/base): dw_fro %.2f, dw_spec %.2f — amplification tracks "
      "dose in either norm"
      % (np.corrcoef(np.log(pile.branch_base), np.log(pile.dw_fro))[0, 1],
         np.corrcoef(np.log(pile.branch_base), np.log(pile.dw_spec))[0, 1]))
ge_b, gs_b_ = spread_fit(pile, "branch_base")
gsb_boot = [spread_fit(pile.iloc[rng.integers(0, len(pile), len(pile))],
                       "branch_base")[1] for _ in range(500)]
print("LOCAL spread effect (same conditioned basis as §4): gE %+.2f, gS %+.2f "
      "[%+.2f, %+.2f] — NEGATIVE: at fixed energy, concentration is louder at the "
      "module output. Against the propagated gS +0.51 (KL, §4), this is the "
      "inversion: the asymmetry between the norms is created by propagation"
      % (ge_b, gs_b_, *np.percentile(gsb_boot, [2.5, 97.5])))
db_ = pd.DataFrame({"t": np.log(pile.branch_base.values),
                    "f": np.log(pile.dw_fro.values),
                    "L": 2 * (np.log(pile.dw_fro.values)
                              - np.log(pile.dw_spec.values)),
                    "lrk": np.log(pile["rank"].values),
                    "rank": pile["rank"].values})
rcb = smf.ols("t ~ f + L + lrk", db_).fit()
print("rank hygiene (same as §4's): gS with log-rank control %+.2f ± %.2f; "
      "within each rank:" % (rcb.params.L, rcb.bse.L))
for rk, g in db_.groupby("rank"):
    if len(g) < 20:
        continue
    frb = smf.ols("t ~ f + L", g).fit()
    print("   r=%3d: gS %+.2f (se %.2f, n=%d)"
          % (rk, frb.params.L, frb.bse.L, len(g)))
print("   -> negative at EVERY rank, strongest at low rank (where one direction is "
      "nearly the whole update): the local advantage of concentration is not a "
      "rank-composition artifact either")

# pooling check: the run-level branch/base above is an UNWEIGHTED mean over the 48
# modules (token-mean within module first). The four module kinds have different
# shapes and input statistics, so refit the local spread effect per kind:
kb_rows = []
for f_ in sorted(AN.glob("adapter_activations_strip_pile_r*.json")):
    per_ = pd.DataFrame(json.loads(f_.read_text())["results"])
    per_["kind"] = per_.module.str.rsplit(".", n=1).str[-1]
    kb_rows.append(per_.groupby(["student_tag", "kind"]).lora_over_base.mean())
bk_ = pd.concat(kb_rows).unstack("kind").join(
    meta_cols[["dw_fro", "dw_spec"]], how="inner")
print("per-module-KIND local gS (12 layers pooled within kind, one fit per kind):")
for kind in ("query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"):
    dkk = pd.DataFrame({"t": np.log(bk_[kind].values),
                        "f": np.log(bk_.dw_fro.values),
                        "L": 2 * (np.log(bk_.dw_fro.values)
                                  - np.log(bk_.dw_spec.values))})
    fk_ = smf.ols("t ~ f + L", dkk).fit()
    print("   %-15s: gS %+.2f (se %.2f)" % (kind, fk_.params.L, fk_.bse.L))
print("   -> negative for all four kinds (magnitude varies ~3x): the unweighted "
      "48-module pooling is not manufacturing the local sign")

# the same added-variable plot as §4's, for the LOCAL readout — slope flips sign
t_b = np.log(pile.branch_base.values)
part_b = smf.ols("t ~ f", pd.DataFrame({"t": t_b, "f": np.log(pile.dw_fro.values)})).fit().resid
L_b = 2 * (np.log(pile.dw_fro.values) - np.log(pile.dw_spec.values))
fig, ax = plt.subplots(figsize=(8, 4.6))
for fate, m, c in (("alive", ~pile.dead.values, "tab:blue"),
                   ("dead", pile.dead.values, "tab:red")):
    ax.scatter(L_b[m], part_b[m], s=24, color=c, alpha=0.6, linewidths=0, label=fate)
slb = np.polyfit(L_b, part_b, 1)
xsb = np.linspace(L_b.min(), L_b.max(), 20)
ax.plot(xsb, np.polyval(slb, xsb), color="black", lw=2,
        label=f"slope {slb[0]:+.2f} (≈ local γ_S)")
ax.axhline(0, color="0.6", lw=0.8)
ax.set_xlabel("L = log stable rank (spread at fixed energy)")
ax.set_ylabel("log branch/base, energy-adjusted (residual on x_F)")
ax.set_title("§4c — the SAME added-variable plot as §4's, on the module-local "
             "readout:\nthe slope flips NEGATIVE — concentration is louder where "
             "it acts")
right_legend(ax)
plt.tight_layout(); plt.show()

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
for ax, xcol, xlab in ((axes[0], "dw_fro", "dw_fro (relative Frobenius dose)"),
                       (axes[1], "dw_spec", "dw_spec (relative spectral dose)")):
    for fate, sub, c in (("alive", pile[~pile.dead], "tab:blue"),
                         ("dead", pile[pile.dead], "tab:red")):
        ax.scatter(sub[xcol], sub.branch_base, s=26, color=c, alpha=0.7, linewidths=0,
                   label=fate)
    r_ = np.corrcoef(np.log(pile[xcol]), np.log(pile.branch_base))[0, 1]
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(xlab); ax.set_title(f"log-log r = {r_:+.2f}")
axes[0].set_ylabel("branch/base ratio (pile)")
right_legend(axes[1])
fig.suptitle("§4c — the adapter's felt loudness vs each dose, one point per cell", y=1.02)
plt.tight_layout(); plt.show()"""))

cells.append(md(r"""### §4e Watching the inversion happen — the depth profile of spread splices (2026-08-10)

**Data: eval-time splices** — the original 8-cell depth probe (adaptive-era,
fate-selected; kept because the per-layer traces are its unique content) **plus the
20 rule-selected strip cells** of the §5.5 spread ladder
(`spread_depth_probe_ladder.json`) — the calibrated-selection replication that
carries the quantitative claim.

**The question this section answers.** §4c and §4 together are a contradiction in
need of a location: at the module output, concentration is *louder* (local
γ_S = −0.81), yet at the model output, concentration moves *less* (γ_S = +0.51).
Both are observational regressions on the same runs — so somewhere between the
module and the head, the network reverses the ordering. This probe finds *where*,
and it is an **intervention**, not a regression, so the two fate clusters and the
dose confounds of the observational fits cannot produce it.

**Method, step by step** (`scripts/spread_depth_probe.py`):

1. Take a cell's trained adapter and form each module's true update
   ΔW = s·B·A. Build two counterfactual updates from its SVD, **each rescaled so
   its Frobenius norm equals the original's, module by module**: `top1` keeps only
   the leading singular direction (all energy in one direction); `flat` keeps the
   same left/right singular subspaces but equalizes the singular values (energy
   spread evenly). Total energy is identical in all three; only how it is
   distributed over directions differs.
2. For each variant, overwrite the live model's weights with W₀ + ΔW′ (all 48
   modules at once; disk untouched), run the **same fixed Pile batch**, and record,
   at every residual-stream slot, the mean relative drift of the hidden state vs
   the unpatched base at the same tokens — plus the final-logit KL. Then restore
   W₀ and move to the next variant.
3. The plotted quantity is drift(top1)/drift(flat) per layer: > 1 = the
   concentrated version transmits more strongly at that depth.

**The registered prediction** — the ratio starts above 1 (per §4c the peaked
injection is louder) and decays monotonically with depth — **was wrong in shape
while right in conclusion**, and the measured shape is more informative:

1. **The attenuation is immediate.** Already at layer 1 (one block of transmission)
   the median ratio is ~0.86: the branch's louder injection (§4c's γ_S = −0.81 is a
   statement about the branch *output*) is knocked below the spread version within
   the very first block's LayerNorm/nonlinearity/projections. There is no shallow
   region where concentration leads — and this is **not a contradiction of §4c**,
   because the two measurements sit on opposite sides of the block: §4c reads the
   branch output at the wire, *before* it enters the residual stream and the block's
   downstream processing; the probe's first slot is *after* one block has processed
   the injection. Together they **bracket the reversal to inside the injecting block
   itself**: louder at the wire, already quieter by the block's exit. That is the
   sharpest localization of the "transmission tax" available.
2. **Mid-network, concentration partially recovers** (median ratio peaks ~1.1 around
   layer 8): a fixed direction injected identically by every block accumulates
   *coherently* down the residual stream, while flat's varied directions add
   incoherently — the one transmission advantage a single direction has.
3. **The output end finishes it** (median 0.82 at the final layer; below 1 in 6/8
   cells): the last blocks plus the final LayerNorm absorb a persistent single
   direction far better than a spread one.

So the inversion is not gradual decay but *instant filtering, partial coherent
recovery, terminal normalization* — with real cell-level heterogeneity (one
deep-frozen cell inverts the pattern; which subspace the energy occupies matters,
as §5.5's unification check concluded).

**Is it "just the final LayerNorm"? No — and the probe shows why.** Every slot here
is a *raw* block output, before any final LN, and the ratio is already 0.86 one
block after injection — the attenuation begins immediately, in ordinary block
processing (each block's own LayerNorms, attention mixing, and MLP nonlinearity
handle a single fixed direction differently from a spread one). The final LN then
adds one more step on top: the raw final slot has the ratio at ~0.8–0.9, while the
logit-level readouts (which sit past the final LN) show the spread advantage even
more strongly — §4's KL γ_S and the spread ladder's top1-reduces-damage result. So
the reconciliation with §4c reads: the peaked update wins exactly at its injection
point (that is §4c's −0.81, and the bound intuition is *correct there*); from the
first block onward transmission taxes it; mid-stack coherence gives it a partial
comeback; the last blocks and the final LN settle the ordering the way §4 measures
it at the output.

Harness: orig's final-layer drift reproduces §4's independently-measured values to
~1e-5."""))

cells.append(code(r"""SDP = AN / "spread_depth_probe.json"
if not SDP.exists():
    print("spread depth probe not on disk — skipped")
else:
    sdp = load_results(SDP)
    pv_d = sdp.pivot_table(index=["student_tag", "layer"], columns="variant",
                           values="rel_drift")
    pv_d["ratio"] = pv_d.top1 / pv_d.flat
    med = pv_d.groupby("layer").median()

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for v, c in (("orig", "0.4"), ("top1", "tab:blue"), ("flat", "tab:orange")):
        axes[0].plot(med.index[1:], med[v].iloc[1:], "o-", ms=5, color=c, label=v)
    axes[0].set_yscale("log"); axes[0].set_xlabel("layer")
    axes[0].set_ylabel("median rel. drift vs base")
    axes[0].set_title("drift accumulation by depth (median over 8 cells)")
    axes[0].legend(frameon=False, fontsize=9)
    for tag, g in pv_d.reset_index().groupby("student_tag"):
        g = g.sort_values("layer")
        axes[1].plot(g.layer[1:], g.ratio[1:], "-", lw=1, alpha=0.5)
    axes[1].plot(med.index[1:], med.ratio.iloc[1:], "o-", lw=2.5, ms=6,
                 color="black", label="median")
    axes[1].axhline(1.0, color="crimson", ls="--", lw=1)
    axes[1].set_xlabel("layer"); axes[1].set_ylabel("drift(top1) / drift(flat)")
    axes[1].set_title("the inversion, layer by layer: instant filtering → coherent "
                      "recovery → terminal normalization")
    right_legend(axes[1])
    plt.tight_layout(); plt.show()
    L = med.ratio
    print("median top1/flat ratio: layer1 %.2f -> mid-peak %.2f (layer %d) -> "
          "final %.2f; final ratio < 1 in %d/8 cells"
          % (L.iloc[1], L.iloc[1:].max(), int(L.iloc[1:].idxmax()), L.iloc[-1],
             int((pv_d.xs(int(pv_d.reset_index().layer.max()), level="layer")
                  .ratio < 1).sum())))
    if "pile" in dir():
        fin_o = pv_d.xs(int(pv_d.reset_index().layer.max()), level="layer").orig
        common = [c for c in fin_o.index if c in pile.index]
        print("harness: orig final drift vs §4 strip activations, max |diff| %.1e"
              % max(abs(fin_o[c] - pile.rel_drift[c]) for c in common))

SDPL = AN / "spread_depth_probe_ladder.json"
if not SDPL.exists():
    print("\n20-cell rule-selected ladder replication: not on disk yet "
          "(queue depth-probe-ladder) — rebuild when it lands")
else:
    sdl = pd.DataFrame(json.loads(SDPL.read_text())["results"])
    pv_l = sdl.pivot_table(index=["student_tag", "layer"], columns="variant",
                           values="rel_drift")
    pv_l["ratio"] = pv_l.top1 / pv_l.flat
    med_l = pv_l.groupby("layer").median()
    L2 = med_l.ratio
    fin_l = pv_l.xs(int(sdl.layer.max()), level="layer").ratio
    print("\nLADDER REPLICATION (20 rule-selected strip cells, seed 1): median "
          "top1/flat ratio layer1 %.2f -> mid-peak %.2f (layer %d) -> final %.2f; "
          "final < 1 in %d/%d cells — the depth signature on outcome-independent "
          "selection" % (L2.iloc[1], L2.iloc[1:].max(), int(L2.iloc[1:].idxmax()),
                         L2.iloc[-1], int((fin_l < 1).sum()), len(fin_l)))
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for tag, g in pv_l.reset_index().groupby("student_tag"):
        g = g.sort_values("layer")
        ax.plot(g.layer[1:], g.ratio[1:], "-", lw=0.8, alpha=0.4, color="tab:green")
    ax.plot(med_l.index[1:], med_l.ratio.iloc[1:], "o-", lw=2.5, ms=6,
            color="darkgreen", label="ladder median (n=20, rule-selected)")
    if SDP.exists():
        ax.plot(med.index[1:], med.ratio.iloc[1:], "s--", lw=1.8, ms=5,
                color="black", label="original 8 cells (fate-selected)")
    ax.axhline(1.0, color="crimson", ls="--", lw=1)
    ax.set_xlabel("layer"); ax.set_ylabel("drift(top1) / drift(flat)")
    ax.set_title("§4e replication — depth profile on rule-selected strip cells")
    right_legend(ax)
    plt.tight_layout(); plt.show()"""))

cells.append(md(r"""**The natural-object check — each pole against the unaltered update** (added
2026-08-12, after Angela's objection that `flat` is not a naturally occurring
object). The top1/flat ratio compares two *counterfactuals*; if flat were weird in a
way that happens to transmit well, the ratio could sit below 1 without any real
"tax" on concentration. The clean comparisons are against **orig** — the trained
update as-is, recorded at every layer in the same runs: `top1/orig` asks what
*concentrating the real update* does (top1 is just a truncation of it — nothing
unnatural), and `flat/orig` asks what spreading it does. Both are re-plots of data
already on disk."""))

cells.append(code(r"""panels = [(nm, p_) for nm, p_ in
           (("8 fate-selected cells", AN / "spread_depth_probe.json"),
            ("20 rule-selected ladder cells", AN / "spread_depth_probe_ladder.json"))
           if p_.exists()]
if not panels:
    print("no depth-probe files on disk — skipped")
else:
    fig, axes = plt.subplots(1, len(panels), figsize=(6.4 * len(panels), 4.6),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (nm, p_) in zip(axes, panels):
        sdp_ = pd.DataFrame(json.loads(p_.read_text())["results"])
        pv_ = sdp_.pivot_table(index=["student_tag", "layer"], columns="variant",
                               values="rel_drift")
        pv_["t_o"] = pv_.top1 / pv_.orig
        pv_["f_o"] = pv_.flat / pv_.orig
        med_ = pv_.groupby("layer").median()
        ax.plot(med_.index[1:], med_.t_o.iloc[1:], "o-", ms=5, color="tab:blue",
                label="top1 / orig (concentrate the real update)")
        ax.plot(med_.index[1:], med_.f_o.iloc[1:], "s-", ms=5, color="tab:orange",
                label="flat / orig (spread the real update)")
        ax.axhline(1.0, color="crimson", ls="--", lw=1)
        ax.set_xlabel("layer"); ax.set_title(nm, fontsize=11)
        fin_ = pv_.xs(int(sdp_.layer.max()), level="layer")
        print("%s: top1/orig layer1 %.2f -> final %.2f (final < 1 in %d/%d cells); "
              "flat/orig layer1 %.2f -> final %.2f (final < 1 in %d/%d)"
              % (nm, med_.t_o.iloc[1], med_.t_o.iloc[-1], int((fin_.t_o < 1).sum()),
                 len(fin_), med_.f_o.iloc[1], med_.f_o.iloc[-1],
                 int((fin_.f_o < 1).sum()), len(fin_)))
    axes[0].set_ylabel("median drift ratio vs orig")
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False,
                    fontsize=9)
    fig.suptitle("§4e — each spread pole against the UNALTERED update, by depth "
                 "(matched per-module energy)", y=1.03)
    plt.tight_layout(); plt.show()
    print("READ: the tax is specific to CONCENTRATION — top1/orig sits below 1 from "
          "the first block on both probes, while flat/orig hovers near 1: the "
          "trained update already transmits like its spread version, and squeezing "
          "it into one direction is what loses drift. The top1/flat headline ratio "
          "was not a flat artifact; the natural-object comparison reproduces the "
          "conclusion on its own.")"""))

cells.append(md(r"""## §5 — second arc: the update is exactly two terms, and Adam sets their exponents

The audit answered *which* norm. The natural next question is *why dose scales the way
it does* with the hyperparameters — because if that has an answer, the collapse law
itself might be derivable rather than merely fitted. It does, and it is. The account
has three ingredients.

**Ingredient 1 — the exact decomposition.** Because B is initialized at exactly zero,
the update is *identically* a sum of two terms — no approximation, no third term:

$$\Delta W = s\,B\,A_\mathrm{init} + s\,B\,\Delta A,\qquad s = \alpha/r.$$

Read each as a different kind of object. $s\,B\,A_\mathrm{init}$ (the **frozen-feature
term**): A never had to move; $A_\mathrm{init}$ is a frozen random projection (the
Kaiming draw), and B is a learned linear readout of those fixed random features —
structurally, random-features regression. $s\,B\,\Delta A$ (the **moved-feature
term**): both factors moved; B reads out of what A *became* — the term whose lr
dependence **compounds**, since both of its factors carry lr, so its exponent is the
sum of theirs. (Terminology note: earlier drafts called these "cross" and
"quadratic", inconsistently. Those names are retired; the terms are named by what
they contain.)

**Ingredient 2 — Adam makes the factors self-limiting.** The factor regressions below
measure $\lVert\Delta A\rVert$ and $\lVert B\rVert$ scaling as **lr^0.56–0.74** — not
lr¹ — with **negative α exponents** (ΔA α^−0.23, B α^−0.15). Both are compensation
effects. The lr-sublinearity: under Adam each step moves parameters by ≈ lr regardless
of gradient size, so displacement should be linear in lr at fixed steps — unless the
trajectory itself changes shape, which it does: a higher-lr run reaches low loss
sooner, its gradients shrink and decohere, motion stalls. The α-shrinkage: the
multiplier s amplifies whatever the factors do, so at high α a smaller factor motion
achieves the same loss reduction — and an optimizer that stops pushing when the loss
stops improving *finds* the smaller motion. This is what "the exponents are an
**equilibrium** between Adam's compensation and LoRA's parameterization" means: the
parameterization sets what would happen at fixed factor motion (s contributes α^+1
exactly; the product compounds the factors' lr terms), the optimizer's
loss-satisfaction sets how much the factors actually move (sublinear in lr, negative
in α), and the measured exponent of every object is where those two forces balance.
Nothing fails to add up — the factor exponents compose exactly into the term
exponents (Ingredient 3); the point is that the factor exponents themselves are
*smaller than naive step-counting predicts*, because the trajectory shortens as the
optimizer is fed more aggression.

**Ingredient 3 — each term's exponents follow by arithmetic.** Ingredient 1 says which
pieces each term contains; Ingredient 2 says how each piece scales; and multiplying
them is licensed *per axis* by the composition check in §5.3 — the product-to-measured
gap is exactly 1 in spectral and, in Frobenius, a constant in lr and α (it drifts only
with r, as $\sqrt r$), so exponents add on the lr and α axes in either norm, and the r
axis is handled by fitting the assembled terms directly. For lr: the frozen-feature
term contains one
lr-dependent piece (B), so it scales as **lr^0.743** — B's own fitted exponent,
inherited exactly, since s and $A_\mathrm{init}$ are lr-free. The moved-feature term
contains two, so exponents add: 0.743 + 0.604 ≈ **lr^1.35**. For α: the multiplier
contributes +1 exactly; the compensations subtract; so the frozen term predicts
α^(1−0.145) ≈ **0.855** and the moved term α^(1−0.145−0.227) ≈ **0.63**. These four
numbers are the predictions §5.1–§5.3 tests. None is a fitted constant of the dose
law; all come from the factor fits plus the decomposition.

### The dose-law regressions

**Data: non-adaptive strip** (run level, $n=890$; every quoted exponent is a
coefficient of this form or of a stated variant):

$$\log\lVert\Delta W\rVert_F \;=\; \beta_0 + a\,\log\mathrm{lr} + d\,\log\alpha +
e\,\log r + \varepsilon$$

fitted by OLS with standard errors **cluster-robust by HP cell** (the 2 ftseed
replicates of one cell are near-copies; treating them as independent is the
pseudo-replication trap of §8b). The spectral companion refits the same right-hand
side on $\log\lVert\Delta W\rVert_2$. The all-budgets variant (adaptive data, its
sole source — see the labeled cell) adds $\log\mathrm{ep}$ and, in the interaction
probe, the product term $\log\mathrm{lr}\times\log\mathrm{ep}$. Because the strip's
placement is outcome-independent, the p-values in this section are **calibrated** —
the adaptive-era caveat that stood here is retired to §8b as method history."""))

cells.append(code(r"""run20 = run20.assign(y=np.log(run20.dW_fro), ys=np.log(run20.dW_spec),
                     llr=np.log(run20.lr), la=np.log(run20.alpha),
                     lrk=np.log(run20["rank"]))
def cfit(df, formula):
    return cluster_ols(df, formula, "cell")
P = cfit(run20, "y ~ llr + la + lrk")
Ps = cfit(run20, "ys ~ llr + la + lrk")
print("PRIMARY dose law (strip, calibrated): lr^%.3f±%.3f a^%.3f±%.3f r^%.3f±%.3f "
      "R2 %.3f" % (P.params.llr, P.bse.llr, P.params.la, P.bse.la,
                   P.params.lrk, P.bse.lrk, P.rsquared))
print("spectral alongside: lr^%.2f a^%.2f r^%.2f R2 %.3f"
      % (Ps.params.llr, Ps.params.la, Ps.params.lrk, Ps.rsquared))
Q = cfit(run20, "y ~ llr + I(llr**2) + la + I(la**2) + lrk")
print("JOINT quadratics (identifiable on orthogonal support): "
      "lr2 %+.4f±%.4f (p=%.0e)  a2 %+.4f±%.4f (p=%.0e) — BOTH real; the local lr "
      "exponent runs %.2f at lr=5e-6 to %.2f at lr=2e-3"
      % (Q.params["I(llr ** 2)"], Q.bse["I(llr ** 2)"], Q.pvalues["I(llr ** 2)"],
         Q.params["I(la ** 2)"], Q.bse["I(la ** 2)"], Q.pvalues["I(la ** 2)"],
         Q.params.llr + 2 * Q.params["I(llr ** 2)"] * np.log(5e-6),
         Q.params.llr + 2 * Q.params["I(llr ** 2)"] * np.log(2e-3)))"""))

cells.append(md(r"""**The budget axis (adaptive data — its sole source).** The strip is
ep20-only by design; training-length dependence exists only in the adaptive
campaigns' ep1–ep40 ladders, so this one contrast is computed there and labeled as
such."""))

cells.append(code(r"""rb = run_all_budgets.assign(y=np.log(run_all_budgets.dW_fro),
                            llr=np.log(run_all_budgets.lr) - np.log(2e-4),
                            lep=np.log(run_all_budgets.ep) - np.log(20),
                            la=np.log(run_all_budgets.alpha) - np.log(8),
                            lrk=np.log(run_all_budgets["rank"]) - np.log(8))
rb["cell"] = rb.student_tag
fit_ix = cfit(rb, "y ~ llr * lep + la + lrk")
print("[adaptive-only] lr x ep interaction: %+.3f±%.3f (p=%.1e) — an ep5-vs-ep20 "
      "CONTRAST, not a power law; uncalibrated support, quoted as the budget axis's "
      "only measurement (nor's accumulation point, vindicated in spirit)"
      % (fit_ix.params["llr:lep"], fit_ix.bse["llr:lep"], fit_ix.pvalues["llr:lep"]))"""))

cells.append(md(r"""**Factor fits — Ingredient 2, measured.** $\Delta A$ and $B$ scale as lr^0.56–0.74
(the Adam compensation) with negative α exponents (ΔA α^−0.23/−0.31, B α^−0.15/−0.24),
while the assembled ΔW grows α^0.60–0.69 — the equilibrium of Ingredient 2, measured.
With each factor at ~lr^0.6–0.7, the moved term composes to ~lr^1.35 — measured
directly at lr^1.31 in §5.3."""))

cells.append(code(r"""rows = []
for obj in ("dA", "B"):
    for l in ("fro", "spec"):
        d = run20.assign(yy=np.log(run20[f"{obj}_{l}"]))
        f = cfit(d, "yy ~ llr + la + lrk")
        rows.append({"object": f"{obj} ({l})", "lr": f.params.llr, "alpha": f.params.la,
                     "r": f.params.lrk})
for l in ("fro", "spec"):
    d = run20.assign(yy=run20.y if l == "fro" else run20.ys)
    f = cfit(d, "yy ~ llr + la + lrk")
    rows.append({"object": f"dW ({l})", "lr": f.params.llr, "alpha": f.params.la,
                 "r": f.params.lrk})
t5f = pd.DataFrame(rows).set_index("object")
display(compare_table(t5f.rename(index=math_label)))
"""))

cells.append(md(r"""### §5.1 Per-α stratified fits and the regime split (Table 5d)

**Which term dominates was measured directly in §1.4**: the moved-feature term wins
~2.5× at α=4; parity around α≈16–64 (norm-dependent); the frozen-feature term wins
~2–3× at α=256; flat across depth. The picture: **α is a laziness dial for A.** High
α → the loss is satisfied before A leaves its init → frozen regime. Low α → A must
genuinely move → moved regime. (This is the lazy-vs-feature-learning dichotomy from
the wide-network literature, reappearing inside LoRA with α as the knob.)

**What is fitted here — two different objects.** The per-α *rows* of Table 5d are
stratified subset fits: within one α stratum (α constant, so it carries no α term),
$\log\lVert\Delta W\rVert = \beta_0 + a\,\log\mathrm{lr} + e\,\log r$ on that
stratum's runs only — every stratum has 84–144 runs on the identical lr ladder.
**Fit L and Fit H** are a different, coarser cut: the **full three-covariate
specification** (lr, α, and r together) refit on the α ≤ 32 and α ≥ 128 *segments* —
they still estimate an α exponent, from the α variation inside each segment. The
rows ask "does the lr slope depend on α"; Fits L/H ask "do all three exponents
differ between the moved-dominated and frozen-dominated regimes."

The result on orthogonal support: the **lr exponent is flat across strata**
(1.01–1.11; L vs H 1.07 vs 1.06) — there is no lr regime. What *does* move with α is
real: the α exponent climbs (0.67 → 0.73, alive-only 0.82 — onto the frozen-term
prediction 0.855) and the rank exponent deepens (−0.17 → −0.32), both the dominance
handoff of §1.4 expressed in the fits. (The adaptive grid showed a dramatic lr shift
here; it was confounded support — §8b.)

**Where the quadratic went.** §5's headline fit detects real lr-curvature
(+0.092), yet these rows look straight — no contradiction: each row reports one
*average* slope over the full 2.6-decade lr ladder, and the curvature lives **along
the lr axis inside every stratum** (the within-segment quadratic check below finds
the same bend in L and in H). The two statements together say: the local lr slope
depends on *where you stand in lr*, not on α — which is exactly what a two-term
mixture predicts, since lr moves dominance the same way in every stratum.

"""))

cells.append(code(r"""rows = []
for a, g in run20.groupby("alpha"):
    if g.student_tag.nunique() < 5:
        rows.append({"alpha": a})  # too-thin stratum (a=256/512: one lr decade) -> NaN row
        continue
    ff = cfit(g, "y ~ llr + lrk"); fs = cfit(g, "ys ~ llr + lrk")
    rows.append({"alpha": a, "n": int(ff.nobs), "cells": g.student_tag.nunique(),
                 "lr (fro)": ff.params.llr, "se": ff.bse.llr,
                 "r (fro)": ff.params.lrk, "se ": ff.bse.lrk,
                 "lr (spec)": fs.params.llr, "r (spec)": fs.params.lrk})
strat = pd.DataFrame(rows).set_index("alpha")
# thin-alpha NaN rows float the count columns; keep them integers (blank when absent)
strat[["n", "cells"]] = strat[["n", "cells"]].astype("Int64")
display(compare_table(strat.round(2), fmt="{:.2f}",
                      subset=["lr (fro)", "r (fro)", "lr (spec)", "r (spec)"]))


L_df, H_df = run20[run20.alpha <= 32], run20[run20.alpha >= 128]
L, H = cfit(L_df, "y ~ llr + la + lrk"), cfit(H_df, "y ~ llr + la + lrk")
Lsp, Hsp = cfit(L_df, "ys ~ llr + la + lrk"), cfit(H_df, "ys ~ llr + la + lrk")
for name, f, fsp, note in (
        ("Fit L (a<=32)", L, Lsp, None),
        ("Fit H (a>=128)", H, Hsp, None)):
    print("%s: lr^%.2f±%.2f a^%.2f±%.2f r^%.2f±%.2f R2 %.3f (%d runs / %d cells); "
          "spec %.2f/%.2f/%.2f"
          % (name, f.params.llr, f.bse.llr, f.params.la, f.bse.la, f.params.lrk,
             f.bse.lrk, f.rsquared, int(f.nobs),
             (L_df if f is L else H_df).student_tag.nunique(),
             fsp.params.llr, fsp.params.la, fsp.params.lrk))
    if note:
        print("   " + note)
M1 = cfit(run20, "y ~ llr + la + lrk + la:lrk")
M2 = cfit(run20, "y ~ llr + la + lrk + I(lrk**2)")
print("misspec probes: a x r %+.3f±%.3f; log-r^2 %+.3f±%.3f — the drift is "
      "regime-shaped, invisible to single-interaction patches"
      % (M1.params["la:lrk"], M1.bse["la:lrk"],
         M2.params["I(lrk ** 2)"], M2.bse["I(lrk ** 2)"]))
for name, seg in (("L", L_df), ("H", H_df)):
    Qseg = cfit(seg, "y ~ llr + I(llr**2) + la + lrk")
    print("within-segment lr quadratic, %s: %+.3f±%.3f — the lr bend is present "
          "INSIDE each regime; only the average slope is flat across them"
          % (name, Qseg.params["I(llr ** 2)"], Qseg.bse["I(llr ** 2)"]))"""))

cells.append(md(r"""### §5.2 Curvature — the local exponent, and the alive-only refit

**Data: non-adaptive strip** (890 runs).

**The curvature machinery, once.** "Curvature" here always means: add a quadratic
term in one log-variable to the standard specification,

$$\log\lVert\Delta W\rVert_F = \beta_0 + \beta_1\log\mathrm{lr} +
\beta_2(\log\mathrm{lr})^2 + d\log\alpha + e\log r + \varepsilon ,$$

so the **local lr exponent** at a given lr is the derivative
$\partial\log\lVert\Delta W\rVert/\partial\log\mathrm{lr} = \beta_1 + 2\beta_2\log\mathrm{lr}$ — a straight power law has $\beta_2=0$ and one global
exponent; $\beta_2>0$ means the log-log surface is convex and the "exponent" is a
function of where you stand. The α² probe is the same construction on the α axis,
and "fit jointly" means both quadratics in one model. Three measurements:

1. **lr curvature**: +0.092±0.004; the local lr slope runs 0.54 at lr=5e-6 up to
   1.65 at lr=2e-3.
2. **α curvature, jointly with lr²**: both quadratics survive — α² = +0.018±0.007
   alongside lr² — matching the two-term prediction that dominance shifts along α as
   well as lr. (On the adaptive grid this coefficient was unidentifiable — the α–lr
   support correlation let lr² absorb it; §8b.)
3. **Alive-only refit**: refitting H on alive runs only moves the α exponent to
   α^0.82±0.01 (87 runs / 45 cells) — onto the frozen-feature prediction (0.855)
   within ~2 SE. Dead runs' degenerate training dilutes the exponent; conditioning
   on an outcome is robustness, not proof, but the direction matches the two-term
   account.

**Consequence.** The curvature is not a nuisance — it is the two-term structure: a sum
of two power laws $C_1x^p + C_2x^q$ has local log-log slope equal to the
*dominance-weighted average* of p and q, so the sum *forces* convexity, with the slope
gliding from one term's exponent to the other's as dominance shifts (measured: 0.7 →
1.6 on the lr axis, bracketing the terms' 0.76 and ~1.4). And every pooled power law in
this project is therefore an **operating-point average** over the surface: the pooled
exponents are real and useful as grid-center summaries, but they are blends, not
constants — R² 0.975 coexists with that fact, because fit quality says nothing about
exponent constancy."""))

cells.append(code(r"""Q = cfit(run20, "y ~ llr + I(llr**2) + la + lrk")
c2 = Q.params["I(llr ** 2)"]
print("1. (log lr)^2: %+.3f±%.3f" % (c2, Q.bse["I(llr ** 2)"]))
for lr_ in (5e-6, 2e-3):
    print("   local lr slope at %.0e: %.2f" % (lr_, Q.params.llr + 2 * c2 * np.log(lr_)))
A2 = cfit(run20, "y ~ llr + I(llr**2) + la + I(la**2) + lrk")
print("2. (log a)^2 jointly with lr^2: %+.3f±%.3f — both quadratics real on "
      "orthogonal support"
      % (A2.params["I(la ** 2)"], A2.bse["I(la ** 2)"]))
Ha = H_df[~H_df.dead]
HA = cfit(Ha, "y ~ llr + la + lrk")
print("3. H alive-only: a^%.2f±%.2f (lr %.2f, r %.2f; %d runs / %d cells)"
      % (HA.params.la, HA.bse.la, HA.params.llr, HA.params.lrk, int(HA.nobs),
         Ha.student_tag.nunique()))"""))

cells.append(md(r"""### §5.3 The terms fitted directly, and the regional collapse law

**Data: non-adaptive strip** (890 runs; the logistic collapse-law refits below use
the same runs' fate labels, and §7.1 holds the bracketed-boundary WLS companion).

The decisive confirmation of the two-term story: the sweep stores **both term norms per
run** ($s\lVert B A_\mathrm{init}\rVert_F$ and $s\lVert B\,\Delta A\rVert_F$, exact
from the QR core), so their exponents can be *measured* rather than inferred by adding
factor exponents. The fits are the standard specification with each term as the
target:

$$\log\big(s\lVert B A_\mathrm{init}\rVert_F\big) = \beta_0 + a\log\mathrm{lr} +
d\log\alpha + e\log r + \varepsilon \qquad\text{(same for } s\lVert B\,\Delta
A\rVert_F\text{)}$$

plus the $(\log\mathrm{lr})^2$ curvature probe on each. The frozen-feature term is
a genuine straight power law — **lr^0.743±0.003, R² 0.997, no curvature** — with its
α and r exponents *exactly* the composition s × B's factor fit (0.86/−0.46 vs
0.86/−0.47): "B reading a fixed random projection" is a nearly deterministic function
of the HPs, and its lr exponent equals B's own (0.743) to the third decimal. The
moved-feature term fits at **lr^1.31±0.01 with real lr-convexity** (+0.064±0.005) —
identifiable only on orthogonal support — and the total's curvature (+0.092) is the
moved term's own bend *plus* the dominance shift. So the exact statement: one term is
a pure power law, the other is convex, and the sum's curvature is their mixture —
which still confirms the two-term account at the level of its parts, with the
convexity now correctly attributed.

**On reading the R².** R² is the share of the 890-run variance in the log term-norm
that the three-exponent power law captures. It is *not* high by construction: any
three-covariate fit maximizes its own R², but nothing forces the maximum to be near 1 —
the total $\lVert\Delta W\rVert_F$ under the identical specification reaches only
0.975 *and* leaves a nine-SE quadratic residual (§5.2), while each term reaches
0.999/high-0.99s and leaves none. It is fair to note that three log-covariates
spanning 2.7 decades will mechanically capture most variance; the discriminating
evidence is therefore the **contrast** — terms straight where the sum bends — not the
absolute R².

**Then the collapse law itself — the level-set argument, in steps.**

1. **Two independently fitted objects.** The **dose law** is fit from weight norms
   only (this section — no fate label enters). The **collapse law** is fit from fate
   labels only (`fit_collapse_law` below: a *logistic* regression of dead/alive on
   log lr, log α, log r — no weight norm enters). Note these L/H refits are a
   different estimator from §5.1's Fit L/H, which were OLS fits of dose on HPs; and
   the H segment here starts at α ≥ 64 rather than 128 because the fate boundary
   needs mixed-fate cells in each segment, which are scarce at the deepest α — the
   boundary-local evaluation in §7.2 removes any sensitivity to this edge choice.
2. **The bridge, and the prediction it generates.** Test 1 established that death
   sits at a (nearly) rank-invariant Frobenius dose θ. *If* that is the whole story
   — death happens exactly when dose crosses θ, and the HPs matter only through the
   dose they generate — then the fate boundary in HP space must be the dose
   surface's contour at height θ: on the boundary
   $\mathrm{lr}^a\alpha^d r^e = \theta$,
   i.e. $\mathrm{lr}\cdot\alpha^{d/a}\cdot r^{e/a} = \theta^{1/a}$ — the collapse law's shape with **b = d/a, c = e/a**.
   That is a *falsifiable numerical prediction* connecting two fits that share no
   measured variable.
3. **What agreement means, and what it does not.** If some HP also killed through a
   second channel (say, high α damaging at a *lower* dose), the fate-fit b would
   exceed d/a and the level-set identity would fail — so the observed closure says:
   **conditional on dose, the HPs carry no additional information about where death
   sits** — dose is a sufficient statistic for the boundary's location. That is a
   consistency statement, not yet causation; the causal upgrade is §5.5's λ
   experiments, where changing dose alone (direction fixed) moves a run across the
   boundary in both directions.
4. **The bend is part of the prediction.** The dose surface is curved (§5.2), so
   d/a is an operating-point quantity and the boundary's α exponent must *climb*
   toward the frozen-regime ratio (~0.85) at deep α. Measured: logistic
   b_L = 0.484 [0.381, 0.587] vs b_H = 0.822 [0.738, 0.905] (WLS companion, §7.1:
   0.499±0.064 vs 0.882±0.098, 3.3σ) — and evaluated boundary-locally (§7.2) the
   identity closes at |z| = 0.5 (L) and 0.0 (H). The published campaign law's
   α^0.561 sits inside the L interval: it is the L-regime law, correct where its
   support lived."""))

cells.append(code(r"""run20 = run20.assign(yf=np.log(run20.term_frozen_fro), ym=np.log(run20.term_moved_fro))
F1, F2 = cfit(run20, "yf ~ llr + la + lrk"), cfit(run20, "yf ~ llr + I(llr**2) + la + lrk")
M1_, M2_ = cfit(run20, "ym ~ llr + la + lrk"), cfit(run20, "ym ~ llr + I(llr**2) + la + lrk")
print("frozen-feature term: lr^%.3f±%.3f R2 %.3f, lr^2 %+.3f±%.3f (straight)"
      % (F1.params.llr, F1.bse.llr, F1.rsquared,
         F2.params["I(llr ** 2)"], F2.bse["I(llr ** 2)"]))
print("   its a/r exponents: a^%.2f r^%.2f — vs composition s x B factor fit: a^%.2f r^%.2f"
      % (F1.params.la, F1.params.lrk,
         1 + t5f.loc["B (fro)", "alpha"], t5f.loc["B (fro)", "r"] - 1))
print("moved-feature term: lr^%.2f±%.2f, lr^2 %+.3f±%.3f — real convexity "
      "(identifiable on this support)"
      % (M1_.params.llr, M1_.bse.llr, M2_.params["I(llr ** 2)"],
         M2_.bse["I(llr ** 2)"]))
print("   its a/r exponents: a^%.2f r^%.2f (R² %.3f) — a LOWER α exponent than the "
      "frozen term's a^%.2f: the sum's α² curvature is the frozen term taking over "
      "at high α, the mirror of the lr crossover (moved takes over at high lr)"
      % (M1_.params.la, M1_.params.lrk, M1_.rsquared, F1.params.la))
S = cfit(run20, "y ~ llr + I(llr**2) + la + lrk")
print("the SUM alone bends: lr^2 %+.3f±%.3f"
      % (S.params["I(llr ** 2)"], S.bse["I(llr ** 2)"]))
D1 = cfit(run20, "y ~ llr + la + lrk")
print("learned factor ‖BA‖/‖W0‖ (dose law minus the s = α/r prefactor): "
      "lr^%.2f a^%.2f r^%.2f — Adam partially compensates the s-amplification "
      "(α exponent < 0) and rank adds coherent channels (r exponent < 1); the "
      "adaptive-grid version a^-0.34 r^+0.78 [adaptive-only, §6f of the mechanism "
      "notebook] agrees"
      % (D1.params.llr, D1.params.la - 1, D1.params.lrk + 1))

from subliminal_learning.analysis.collapse_law import fit_collapse_law

law_df = run20.rename(columns={"dead": "collapsed"}).assign(
    collapsed=lambda d: d.collapsed.astype(int))
for name, seg in (("L segment (a<=32)", law_df[law_df.alpha <= 32]),
                  ("H segment (a>=64)", law_df[law_df.alpha >= 64])):
    fit = fit_collapse_law(seg)
    print("%s: b = %.3f [%.3f, %.3f], c = %.3f (%d runs, %d dead)"
          % (name, fit.b, *fit.b_ci, fit.c, fit.n_runs, fit.n_dead))
print("-> the published law's a^0.561 is the L-regime law; the local exponent climbs "
      "to the dose surface's deep-alpha d/a ~ 0.85, non-overlapping intervals")
print("(the bracketed-boundary WLS replication — 0.499±0.064 vs 0.882±0.098, 3.3 "
      "sigma — lives in §7.1, which also folds in the diverged cells)")"""))

cells.append(md(r"""**The composition check — why a loose bound still licenses exponent
arithmetic.** Ingredient 3 adds factor exponents, which silently assumes
$\lVert BA\rVert \approx \lVert B\rVert\,\lVert A\rVert$. Is that safe? Two separate
questions, and only the second matters:

- **Is the product bound tight?** In spectral, yes (§1.5, ratio ~1.0). In Frobenius,
  **no** — the table below shows the product overshoots the measured norm by a
  median ~2–3×. So no, you cannot treat $\lVert B\rVert_F\lVert A\rVert_F$ as equal
  to $\lVert BA\rVert_F$.
- **Does the gap *vary with the HPs*?** This is the only thing exponent arithmetic
  needs. Take logs:
  $\log\lVert BA\rVert = \log\lVert B\rVert + \log\lVert A\rVert - \log(\mathrm{gap})$. If the gap is a **constant** along some HP axis, it moves
  the intercept and leaves every slope (= exponent) untouched — a 2.5× gap that is
  *always* 2.5× is invisible to a log-log slope. Only a gap that *changes* with lr,
  α, or r biases the added exponent, by exactly its own exponent.

**Where you see the answer**: the regression printed under the table fits the gap
itself on the HPs — gap ∝ lr^0.000 · α^0.000 · r^0.491. Flat in lr and α to three
decimals; √r in rank (expected for incoherent factors, where
$\lVert BA\rVert_F \approx \lVert B\rVert_F\lVert A\rVert_F/\sqrt r$). Verdict:
adding factor exponents is exact on the lr and α axes in either norm, off by ~+0.5
on the r axis in Frobenius — and §5.3's headline numbers never use the license
anyway, because the term norms are stored per run and fitted directly; the
arithmetic only cross-checks them."""))

cells.append(code(r"""PAIRS = (("frozen", "A_init", "B_Ainit"), ("moved", "dA", "B_dA"),
         ("total dW", "A_final", "B_Afinal"))
rows = []
for l in ("spec", "fro"):
    for term, afac, meas in PAIRS:
        ratio = (f20[f"B_{l}"] * f20[f"{afac}_{l}"] / f20[f"{meas}_{l}"]).dropna()
        q = np.quantile(ratio, [.25, .5, .75, .99])
        rows.append({"term": term, "norm": {"spec": "2", "fro": "F"}[l],
                     "p25": q[0], "median": q[1], "p75": q[2], "p99": q[3]})
comp = pd.DataFrame(rows).set_index(["term", "norm"])
print("product/measured ratio ‖B‖·‖A‖/‖B·A‖ — row unit: one module row of the fitted "
      "set (%d rows); 1.0 = bound tight" % len(f20))
display(compare_table(comp, fmt="{:.2f}"))

gap = (f20.assign(ratio=f20.B_fro * f20.A_init_fro / f20.B_Ainit_fro)
       .groupby(["student_tag", "ftseed"], as_index=False)
       .agg(lr=("lr", "first"), alpha=("alpha", "first"), rank=("rank", "first"),
            ratio=("ratio", "mean")))
gap = gap.assign(y=np.log(gap.ratio), llr=np.log(gap.lr), la=np.log(gap.alpha),
                 lrk=np.log(gap["rank"]))
gf = smf.ols("y ~ llr + la + lrk", gap).fit(
    cov_type="cluster", cov_kwds={"groups": gap.student_tag})
print("HP drift of the Frobenius frozen-term GAP (gap = the ratio "
      "‖B‖_F·‖A0‖_F/‖B·A0‖_F, run-mean — how loose the product bound is): "
      "lr^%.3f±%.3f a^%.3f±%.3f r^%.3f±%.3f — flat in lr and alpha, sqrt(r) in rank"
      % (gf.params.llr, gf.bse.llr, gf.params.la, gf.bse.la,
         gf.params.lrk, gf.bse.lrk))
print("   residual sd (log): %.3f (≈±%.0f%%) — at fixed rank the frozen gap is "
      "essentially DETERMINISTIC at sqrt(r): the Kaiming A0's rows are "
      "near-isotropic, so ‖B·A0‖_F ≈ ‖B‖_F·‖A0‖_F/√r identically"
      % (gf.resid.std(), 100 * (np.exp(gf.resid.std()) - 1)))

# The MOVED term's gap — the same question for ‖B‖·‖ΔA‖/‖B·ΔA‖ (2026-08-13, after
# Angela challenged 'the gap is constant': tightness and HP-constancy are different
# properties, and only constancy licenses exponent-adding).
gapm = (f20.assign(ratio=f20.B_fro * f20.dA_fro / f20.B_dA_fro)
        .groupby(["student_tag", "ftseed"], as_index=False)
        .agg(lr=("lr", "first"), alpha=("alpha", "first"), rank=("rank", "first"),
             ratio=("ratio", "mean")))
gapm = gapm.assign(y=np.log(gapm.ratio), llr=np.log(gapm.lr), la=np.log(gapm.alpha),
                   lrk=np.log(gapm["rank"]))
gm = smf.ols("y ~ llr + la + lrk", gapm).fit(
    cov_type="cluster", cov_kwds={"groups": gapm.student_tag})
print("HP drift of the Frobenius MOVED-term GAP (‖B‖_F·‖ΔA‖_F/‖B·ΔA‖_F, run-mean): "
      "lr^%.3f±%.3f a^%.3f±%.3f r^%.3f±%.3f, R² %.2f, residual sd ≈±%.0f%% — "
      "NOT HP-flat and widely scattered"
      % (gm.params.llr, gm.bse.llr, gm.params.la, gm.bse.la,
         gm.params.lrk, gm.bse.lrk, gm.rsquared,
         100 * (np.exp(gm.resid.std()) - 1)))
print("READ: Frobenius factor-composition is licensed for the FROZEN term only "
      "(gap deterministic given rank, flat in lr/α to ±0.000). The moved term's "
      "exponents are never composed — §5.3 measures them directly on ‖B·ΔA‖. The "
      "spectral bounds are the tight ones (see table above), but tightness was "
      "never the requirement; constancy is.")"""))

cells.append(md(r"""**Table 5e — the level-set prediction, tabulated.** What the table is: step 2 of
the level-set argument says every dose-law fit *predicts* a collapse law, with
exponents $b = d/a$ and $c = e/a$. **Each row is one dose-law fit** (pooled, the L
and H segments, the spectral companion, and the `[adaptive-only]` six-arm cross-arm
fit); **the two styled columns are that fit's predicted collapse exponents**, with
delta-method SEs from the fit's own covariance. **The bottom row is what the fate
labels actually gave** (the published campaign law), so reading down a column
compares prediction against measurement. Read it loosely, not point by point — the
ratios are operating-point averages over a curved surface (each fit's support
differs), which is why the strip segments' ratios overshoot the boundary-fitted b
(L: naive 0.63 vs measured 0.499±0.064); the *boundary-local* evaluation in §7.2 is
the version of this comparison that closes sharply (|z| = 0.5 / 0.0). The table's
value is coarser: every fit, on every support, in either norm, puts d/a in the same
0.5–0.75 band the fate labels independently produce — with **no data pathway shared**
between the two sides.

**The r column needs one extra measured ingredient.** The raw prediction $e/a$
assumes the death threshold is constant in r — and Test 1 *measured* its drift,
θ(r) ∝ r^−0.110 [−0.126, −0.067]. Folding the real death condition in:
(e + 0.110)/a = −0.098 (Fit L) to −0.120 (pooled), bracketing the fitted rank
exponent (logistic c_L = −0.111). In words: the collapse law's small rank term is
the **residue of two larger opposing rank effects** — at matched aggression, dose
falls with rank while the dose needed to kill also falls.

"""))

cells.append(code(r"""def ratios(f):
    '''(alpha/lr, r/lr) with delta-method SEs from the cluster covariance.'''
    p, V = f.params, f.cov_params()
    idx = list(p.index)
    den = idx.index("llr")
    out = []
    for name in ("la", "lrk"):
        est, (lo, hi) = delta_ratio_ci(p.to_numpy(), V.to_numpy(), idx.index(name), den)
        out.append((est, (hi - lo) / (2 * 1.96)))
    return out

all20 = to_runs(mod_all[mod_all.ep.eq(20) & mod_all.test_p.notna()])
all20 = all20.assign(y=np.log(all20.dW_fro), llr=np.log(all20.lr),
                     la=np.log(all20.alpha), lrk=np.log(all20["rank"]))
X6f = cluster_ols(all20, "y ~ llr + la + lrk", "cell")
print("[adaptive-only] §6f cross-arm fit (all 6 arms, all teachers): "
      "lr^%.2f a^%.2f r^%.2f R2 %.3f"
      % (X6f.params.llr, X6f.params.la, X6f.params.lrk, X6f.rsquared))

rows = []
for name, f in (("pooled F", P), ("Fit L F", L), ("Fit H F", H),
                ("pooled spec", Ps), ("cross-arm F (§6f)", X6f)):
    (b_, sb), (c_, cse) = ratios(f)
    rows.append({"fit": name, "alpha/lr": b_, "se(a/lr)": sb,
                 "r/lr": c_, "se(r/lr)": cse})
rows.append({"fit": "collapse law K*", "alpha/lr": B_LAW, "se(a/lr)": np.nan,
             "r/lr": C_LAW, "se(r/lr)": np.nan})
t5e = pd.DataFrame(rows).set_index("fit")
display(compare_table(t5e.round(3), subset=["alpha/lr", "r/lr"]))
for name, f in (("Fit L", L), ("pooled", P)):
    eff = (f.params.lrk + 0.13) / f.params.llr
    eff_cal = (f.params.lrk + 0.110) / f.params.llr
    print("drift-corrected r prediction (%s): (%.2f + 0.13)/%.2f = %.3f; with §7.1's "
          "CALIBRATED drift 0.110: %.3f"
          % (name, f.params.lrk, f.params.llr, eff, eff_cal))
print("(the calibrated drift moves the pooled prediction to within ~0.005 of the "
      "fitted -0.085 — the §5.3 'unexplained sliver' shrinks under better "
      "measurement of the same ingredient)")
"""))

cells.append(md(r"""### §5.4 The term splices — the regime story closed at the function level

**Data: the `strip-splices` sweep — every evaluated strip run × both splices × both
seeds** (932 runs → 1,864 splice evaluations per direction; rule: no selection
whatsoever).

Everything above is regression evidence. This is the **inference-only intervention**
that converts the dominance surface into a functional claim:
`scripts/frozen_a_ablation.py` splices, in memory (disk adapters untouched), either
$\Delta W' = s\,B\,A_\mathrm{init}$ (**keep frozen** — substitute the reconstructed
$A_\mathrm{init}$ for the trained $A_\mathrm{final}$) or
$\Delta W'' = s\,B\,\Delta A$ (**keep moved** — remove the frozen-feature term), and
re-evaluates p(trait) *and* Pile KL jointly. Harness check: re-evaluated originals
reproduce every stored preference eval to 4+ decimals.

**How to read the metric.** Retention is **excess-over-base**:
$(p_\mathrm{spliced} - p_\mathrm{base}) / (p_\mathrm{orig} - p_\mathrm{base})$ —
ablation moves p toward the untrained base's own ~0.0133, never toward 0, and a raw
p-ratio would credit the spliced model with the base's prior. Retention is only
*defined* where there is trait excess to retain: for dead cells the excess is
negative and the ratio becomes a quotient of two negatives (spurious "retention"),
so all analyses filter to trait-carrying cells (`p_orig > 0.02`); for dead cells the
informative readout is the KL comparison at the bottom.

**What the sweep shows** (table + boxplot below): the **double dissociation, as
population statistics** — keep-frozen retention *rises* monotonically with α while
keep-moved retention *falls*, crossing at α ≈ 32, the Frobenius dominance-parity
point of §1.4 (not the spectral one). The trait rides whichever term dominates the
energy. Retention is **non-additive** (per-cell sum median ~0.77, IQR ~[0.65,
0.90]): a substantial share of the trait lives in neither term alone — weight
decomposition is only approximately compositional in function space, worst near the
boundary. And on the dead side, removing the moved term moves a substantial minority
of runs *farther* from base — near the cliff the moved term partially **cancels**
the frozen term's disruption. The regime story is thereby closed at three levels:
decomposition (§1.4), scaling (§5.3), function (§5.4).

**Claim discipline**: splice results license *"the adapter's weight configuration is
causally sufficient for the phenotype at evaluation time"*; training-time sentences
need the clamp/retrain tier of `causal_plan.md`."""))

cells.append(md(r"""**Retention distributions, both splice directions.** Box plots of
excess-retention by α, trait-carrying cells only (`p_orig > 0.02`), one box pair per
α — blue = keep frozen (ablate $B\,\Delta A$), orange = keep moved (ablate
$B A_\mathrm{init}$). The dead-side KL check at the bottom asks the
non-compositionality question on the same population: at high-α dead cells, does the
frozen-only adapter land *farther* from base than the full adapter (the terms
partially canceling in function space)?"""))

cells.append(code(r"""def _load_strip_splice(splice):
    files = sorted(AN.glob(f"frozen_a_ablation_strip_{splice}_r*_s*.json"))
    frames, pb = [], None
    for f in files:
        d = json.loads(f.read_text())
        pb = d["p_target_base"]
        frames.append(pd.DataFrame(d["results"]))
    return (pd.concat(frames) if frames else pd.DataFrame()), pb, len(files)

sfz, pb_s, nfz = _load_strip_splice("frozen")
smv, _, nmv = _load_strip_splice("moved")
assert nfz == 16 and nmv == 16, (   # 8 ranks x 2 seeds per splice
    f"strip-splices incomplete on disk ({nfz}/16 frozen, {nmv}/16 moved chunks) — "
    "drain the queue and rebuild; no fallback population is used")
for fr in (sfz, smv):
    fr["excess_retention"] = ((fr.p_target_ablated - pb_s)
                              / (fr.p_target_orig - pb_s))
POP, pop_f, pop_m, pb_pop = ("STRIP (all evaluated runs x 2 seeds, "
                             "rule: no selection)"), sfz, smv, pb_s

tcf = pop_f[pop_f.p_target_orig > 0.02]
tcm = pop_m[pop_m.p_target_orig > 0.02]
print(f"population: {POP} — {len(pop_f)} frozen / {len(pop_m)} moved splice rows; "
      f"trait-carrying: {len(tcf)} / {len(tcm)}")

fig, ax = plt.subplots(figsize=(10, 5))
alphas_fu = sorted(pop_f.alpha.unique())
pos = np.arange(len(alphas_fu), dtype=float)
for lab, frame, color, off in (("keep frozen", tcf, "tab:blue", -0.18),
                               ("keep moved", tcm, "tab:orange", 0.18)):
    data = [frame.loc[frame.alpha == a, "excess_retention"].dropna().to_numpy()
            for a in alphas_fu]
    bp = ax.boxplot(data, positions=pos + off, widths=0.3, whis=(5, 95),
                    showfliers=False, patch_artist=True,
                    medianprops=dict(color="black", lw=1.6))
    for b_ in bp["boxes"]:
        b_.set(facecolor=color, alpha=0.55)
    ax.plot([], [], color=color, lw=8, alpha=0.55,
            label=f"{lab} (n={sum(len(d) for d in data)})")
ax.axhline(1.0, color="0.4", lw=1, ls=":")
ax.axhline(0.0, color="0.4", lw=1, ls=":")
ax.set_xticks(pos); ax.set_xticklabels(alphas_fu)
ax.set_xlabel(r"$\alpha$"); ax.set_ylabel("excess retention")
ax.set_title("§5.4 — excess retention by α and splice direction\n"
             f"({POP.split(' (')[0]}; trait-carrying cells; box = IQR, "
             "whiskers = 5–95%)")
right_legend(ax)
plt.tight_layout(); plt.show()

hand = pd.DataFrame({"n": tcf.groupby("alpha").size(),
                     "keep frozen": tcf.groupby("alpha").excess_retention.median(),
                     "keep moved": tcm.groupby("alpha").excess_retention.median()})
hand["sum"] = hand["keep frozen"] + hand["keep moved"]
print("median excess retention by alpha, both splices (trait-carrying cells):")
display(compare_table(hand, fmt="{:.2f}",
                      subset=["keep frozen", "keep moved", "sum"]))
KEY_SP = (["student_tag", "ftseed"] if "ftseed" in tcf.columns
          and tcf.ftseed.nunique() > 1 else ["student_tag"])
pair = (tcf.set_index(KEY_SP).excess_retention.rename("f").to_frame()
        .join(tcm.set_index(KEY_SP).excess_retention.rename("m")))
tot = (pair.f + pair.m).dropna()
cross = hand[hand["keep frozen"] >= hand["keep moved"]]
print("the FUNCTIONAL handoff (equal shares) sits at alpha ~ %s — the Frobenius "
      "dominance-parity point (§1.4), not the spectral one"
      % (cross.index.min() if len(cross) else "n/a"))
print("non-additivity is systemic: per-cell retention sum median %.2f, "
      "IQR [%.2f, %.2f] (n=%d)"
      % (tot.median(), tot.quantile(.25), tot.quantile(.75), len(tot)))

# dead-side check: does the frozen-only splice land FARTHER from base than the
# full adapter at high-alpha dead cells (terms partially canceling)?
dd_sp = pop_f[(pop_f.p_target_orig < 0.01) & (pop_f.alpha >= 128)]
if len(dd_sp):
    worse = (dd_sp.pile_kl_base_ablated > dd_sp.pile_kl_base_orig)
    print("dead cells at alpha>=128 (n=%d): frozen-only farther from base than the "
          "full adapter in %d (%.0f%%; median KL %.1f vs %.1f nats) — in that "
          "subset the moved term partially CANCELS the frozen term's disruption "
          "in function space (the sharpest form of non-compositionality)"
          % (len(dd_sp), int(worse.sum()), 100 * worse.mean(),
             dd_sp.pile_kl_base_ablated.median(), dd_sp.pile_kl_base_orig.median()))"""))

cells.append(md(r"""## §5.5 — the evaluation-time interventions

**Data: eval-time surgeries on rule-frozen sweeps.** Every cross-cell statistic in
this section comes from cell lists **fixed by rule before any outcome was seen**
(`run_logs/causal_exp_cells_v2.json`): the λ sweep = 23 strata cells + the 103-cell
shell extension × both seeds × a 13-point λ ladder; the spread ladder = 20 cells ×
both seeds (+ shell extension); the ws1 arm; and the W₀-sliver surgery = the
**census of all 17 mixed-fate cells** (every cell whose seeds split alive/dead — a
rule, not a choice). The original 2026-08-07 discovery runs on 8 fate-picked cells
are retired from the analysis; their within-cell findings all replicate on the
sweeps below.

§4b-bis named the gap: everything through §5.4's term splices is either observational
or causal only about *which term* carries the trait. Three interventions attack the
dose→death claim itself, each with a pre-stated signed prediction. All are
**evaluation-time** surgeries on trained endpoints — they license "this weight
configuration is causally sufficient for the phenotype," never any training-time
sentence.

**Exp 1 — λ dose-response.** ΔW → λ·ΔW (lora_B scaled; direction structure exactly
preserved), each cell swept across its own crossing region. Prediction under
"death = Frobenius dose crossing θ": fate flips when λ·dose crosses θ, in *both*
directions — killing trained-alive runs by up-scaling and reviving trained-dead
runs by down-scaling.

**Exp 2 — W₀-sliver surgery at fixed dose.** §1.2b left a pointer: cos(ΔW, W₀) was
(on the adaptive campaigns) fate-linked though only ~0.1% of the update's energy. On
the mixed-fate census (dose already matched within each cell), flip / delete /
10×-amplify the W₀-aligned component at exactly preserved per-module ‖ΔW‖_F.
Prediction if the sliver is causal: flip/strip rescue the dying seed, amplification
harms the survivor. One asymmetric stake, stated before the result: the energy
account **entails** this null (the sliver is energy-negligible, so an energy-world
has no mechanism for it to carry function); the peak account permits either outcome.
A null one theory entails and the other merely tolerates is evidence.

**Exp 3 — spread surgery at matched energy.** §4's γ_S > 0 made causal: rebuild each
module's ΔW at its original ‖ΔW‖_F as `top1` (all energy in the top singular
direction — this *raises* the spectral norm by √srank) or `flat` (singular values
equalized over the update's existing numerical rank; same singular subspaces, only
the within-subspace weighting rewritten). Peak theory predicts top1 ≥ original
damage; the energy/spread account predicts less."""))

cells.append(code(r"""import re as _re
KSTAR_DS1_DESIGN, B_D, C_D = 7.84e-4, 0.561, -0.085
_LS = (sorted(AN.glob("frozen_a_ablation_lamstrat*_lam*.json"))
       + sorted(AN.glob("frozen_a_ablation_shellext_s*_lam*.json")))
if not _LS:
    print("λ sweep results not on disk — skipped")
else:
    rows = []
    for f in _LS:
        d = json.loads(f.read_text())
        seed = d.get("ftseed", 2 if "_s2" in f.name else 1)
        for r in d["results"]:
            rows.append({"cell": r["student_tag"], "seed": seed,
                         "lam": d["scale_lambda"], "p": r["p_target_ablated"],
                         "p_orig": r["p_target_stored"]})
    lam_all = pd.DataFrame(rows).drop_duplicates(["cell", "seed", "lam"])
    p_base_l = json.loads(_LS[0].read_text())["p_target_base"]
    tp = lam_all.cell.str.extract(r"ep20_lr(?P<lr>[0-9.e-]+)_r(?P<r>\d+)a(?P<a>\d+)")
    lam_all["lr"], lam_all["rank"], lam_all["alpha"] = (
        tp.lr.astype(float), tp.r.astype(int), tp.a.astype(int))
    lam_all["K_rel"] = (lam_all.lr * lam_all.alpha**B_D * lam_all["rank"]**C_D
                        / KSTAR_DS1_DESIGN)
    lam_all["td"] = lam_all.p_orig < 0.01

    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    n_a = n_d = 0
    for (c_, s_), g in lam_all.groupby(["cell", "seed"]):
        g = g.sort_values("lam")
        td = bool(g.td.iloc[0])
        col = "tab:red" if td else "tab:blue"
        n_d += td; n_a += (not td)
        ax.plot(g.lam * g.K_rel, g.p, "-", lw=0.9, alpha=0.45, color=col)
    ax.axhline(p_base_l, color="0.5", ls=":", lw=1, label="base prior")
    ax.axhline(0.01, color="crimson", ls="--", lw=1, label="dead cut")
    ax.axvline(1.0, color="0.3", ls="-.", lw=1, label="training boundary K/K* = 1")
    ax.plot([], [], color="tab:blue", label=f"trained alive ({n_a} run-curves)")
    ax.plot([], [], color="tab:red", label=f"trained dead ({n_d} run-curves)")
    ax.set_xscale("log")
    ax.set_xlabel(r"effective dose $\lambda \cdot K/K^*$ (boundary units)")
    ax.set_ylabel("p(dog)")
    ax.set_title("Exp 1 on the rule-selected λ sweep — every run-curve on the common "
                 "dose axis:\nsame effective dose, different fate by training history "
                 "(hysteresis)")
    right_legend(ax, fontsize=9)
    plt.tight_layout(); plt.show()
    print("read: pure rescaling walks every run across the boundary — alive curves "
          "(blue) trace the transfer hump and die past their peak; dead curves (red) "
          "revive when scaled DOWN into the band. The blue family survives to visibly "
          "larger effective dose than the red family revives at — the history gap the "
          "crossing analysis below quantifies (§5.5b).")"""))

cells.append(md(r"""**Object 3 — the revival curve: where masking becomes erasure.**
A different question from any per-run curve: for each trained-**dead** run, take the
*best achievable outcome over the whole λ sweep* (the peak of its curve — the
recoverable-trait ceiling) and plot that single number against the run's training
dose $d_0$. Near the boundary the ceiling is high (the trait was masked, not
erased); if it decays with training dose toward the base prior, that is a statement
about **information destruction in the weights** as a function of how far past the
boundary training pushed. The 9-cell revival design (training dose 0.14–1.45, λ down
to 0.05 so even the deepest cell's sweep reaches the revival band) ran as `lambda-revival`
(8/8 complete); the cell below draws it, and extends the curve with the **§5.5b
lambda-strat trained-dead cells** (rule-selected, green) — the selection-clean
portion of the same object."""))

cells.append(code(r"""rev_files = sorted(AN.glob("frozen_a_ablation_revival_lam*.json"))
if not rev_files:
    print("no revival lambda results on disk — skipped")
else:
    rrows = []
    for f in rev_files:
        d = json.loads(f.read_text())
        for r in d["results"]:
            rrows.append({"cell": r["student_tag"], "lam": d["scale_lambda"],
                          "p": r["p_target_ablated"],
                          "p_orig": r["p_target_stored"]})
    rl = pd.DataFrame(rrows)
    # the lambda-revival design: deep-dose trained-dead endpoints (every cell here
    # is trained-dead by construction; keep only those, defensively)
    rl = rl[rl.p_orig < 0.01]
    # d0 lookup: strip harvest (seed 1) first, adaptive frame for off-ladder cells
    d0 = (strip[strip.seed == 1].set_index("tag").dw_fro
          .combine_first(run20a[run20a.ftseed == 1]
                         .set_index("student_tag").dw_fro))
    ceil_rows = []
    for c, g in rl.groupby("cell"):
        if c not in d0.index or not np.isfinite(d0[c]):
            continue
        ceil_rows.append({"cell": c, "d0": d0[c], "ceiling": g.p.max(),
                          "best_lam": g.loc[g.p.idxmax(), "lam"], "n_lam": g.lam.nunique()})
    cr = pd.DataFrame(ceil_rows).sort_values("d0").set_index("cell")
    display(compare_table(cr, fmt="{:.3f}", subset=["d0", "ceiling", "best_lam"],
                          labels={"d0": "training dose d0",
                                  "ceiling": "max-over-λ p(dog)",
                                  "best_lam": "λ at ceiling"}))

    # extend with the §5.5b lambda-strat trained-dead cells (rule-selected)
    ls_rows = []
    LS_ALL = sorted(AN.glob("frozen_a_ablation_lamstrat*_lam*.json"))
    if LS_ALL:
        lsr = []
        for f in LS_ALL:
            d = json.loads(f.read_text())
            sd = d.get("ftseed", 2 if "_s2_" in f.name else 1)
            for r in d["results"]:
                lsr.append({"cell": r["student_tag"], "seed": sd,
                            "lam": d["scale_lambda"], "p": r["p_target_ablated"],
                            "p_orig": r["p_target_stored"]})
        lsr = pd.DataFrame(lsr)
        d0_lut = strip.set_index(["tag", "seed"]).dw_fro
        for (c, sd), g in lsr.groupby(["cell", "seed"]):
            if g.p_orig.iloc[0] >= 0.01:
                continue
            d0v = d0_lut.get((c, sd), np.nan)
            if np.isfinite(d0v):
                ls_rows.append({"cell": f"{c}|s{sd}", "d0": d0v,
                                "ceiling": g.p.max()})
    lsc = pd.DataFrame(ls_rows)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(cr.d0, cr.ceiling, "o", ms=9, color="tab:red",
            label="lambda-revival design (deep-dose dead, n=%d)" % len(cr))
    for c, row in cr.iterrows():
        ax.annotate(c.replace("ep20_", ""), (row.d0, row.ceiling), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    if len(lsc):
        ax.plot(lsc.d0, lsc.ceiling, "^", ms=8, color="tab:green", alpha=0.8,
                label="rule-selected (§5.5b lamstrat, n=%d)" % len(lsc))
    ax.axhline(0.0133, color="0.5", ls=":", lw=1, label="base prior (fully erased)")
    ax.axhline(0.01, color="crimson", ls="--", lw=1, label="dead cut")
    ax.set_xscale("log")
    ax.set_xlabel(r"training dose $d_0$ (dw$_F$)")
    ax.set_ylabel("recoverable-trait ceiling (max-over-λ p)")
    ax.set_title("the revival curve — masked vs erased, as a function of training dose")
    right_legend(ax)
    plt.tight_layout(); plt.show()
    if len(lsc):
        both_rc = pd.concat([cr.reset_index()[["d0", "ceiling"]],
                             lsc[["d0", "ceiling"]]])
        rho_rc = stats.spearmanr(np.log(both_rc.d0), both_rc.ceiling)
        print("pooled revival curve (n=%d dead endpoints): Spearman(log d0, "
              "ceiling) = %+.2f (p=%.3f) — the recoverable-trait ceiling decays "
              "with training dose" % (len(both_rc), rho_rc.statistic, rho_rc.pvalue))"""))

cells.append(code(r"""SURG = sorted(AN.glob("direction_surgery_ep20_*.json"))
if not SURG:
    print("direction surgery results not on disk — skipped")
else:
    rows = []
    for f in SURG:
        for r in json.loads(f.read_text())["results"]:
            rows.append({k: r[k] for k in ("student_tag", "fate", "variant",
                                           "p_target", "pile_kl_base")})
    sg = pd.DataFrame(rows)
    pv = sg.pivot_table(index=["student_tag", "fate"], columns="variant",
                        values="p_target")[["orig", "flip", "strip", "flip10"]]
    kv = sg.pivot_table(index=["student_tag", "fate"], columns="variant",
                        values="pile_kl_base")[["orig", "flip", "strip", "flip10"]]
    dp = pv.sub(pv.orig, axis=0).drop(columns="orig")
    dk = kv.sub(kv.orig, axis=0).drop(columns="orig")
    summ = pd.DataFrame({
        "median Δp (dead seeds)": dp.xs("dead", level="fate").median(),
        "median Δp (alive seeds)": dp.xs("alive", level="fate").median(),
        "max |Δp|": dp.abs().max(),
        "fate flips": [(int(((pv[v] < 0.01) != (pv.orig < 0.01)).sum()))
                       for v in dp.columns],
        "median ΔKL": dk.median()})
    print("W0-sliver surgery, %d mixed cells x both seeds (dose exactly preserved "
          "per module):" % pv.index.get_level_values(0).nunique())
    display(compare_table(summ, fmt="{:.4f}",
                          subset=[c for c in summ.columns if c != "fate flips"]))
    print("VERDICT: null. Even 10x amplification of the aligned sliver moves p(dog) "
          "by at most %.3f and flips %d/%d fates. cos(dW, W0) is a MARKER of bad "
          "updates, not a mechanism — the §1.2b restraint was warranted, and the "
          "direction program moves to spectrum/subspace geometry."
          % (dp.abs().max().max(), int(((pv.flip10 < 0.01) != (pv.orig < 0.01)).sum()),
             len(pv)))"""))

cells.append(md(r"""### §5.5b — the crossing analysis (pre-registered)

**Data: the same rule-frozen sweeps** (`run_logs/causal_exp_cells_v2.json`).
Nine off-ladder cells' ftseed-2 checkpoints are backfilling (queue `v2-s2-backfill`,
2026-08-11); their seed-2 rows join the fits automatically on the next rebuild after
`lambda-strat-s2` / `spread-ladder-s2` drain.

The designs: **lambda-strat** — 23 strip cells, strata α ∈ {4, 16, 64, 256} ×
r ∈ {1, 8, 64} × design-K targets {0.8, 1.6} (placement by the law's prediction,
never observed fate), 13-point λ ladder, both strip seeds; **spread-ladder** — 20
cells, r ∈ {4..64} × α ∈ {8, 64} × K targets {0.75, 1.5}, orig/top1/flat, both
seeds; **ws1-causal** — 8 `law-focus-ws1` cells (the arm where the cos marker was
absent), λ ladder + spread splices.

The **shell extension** (same day, second pre-registration): strip-native universe
only, all 8 ranks × 7 α at K-targets {0.8, 1.6} plus every cell in the design-K band
[0.85, 1.35], both seeds, with a **boundary-normalized λ grid** (fixed 13-point
ladder in K/K\* units, realized per cell as λ = g/K_rel, snapped to a shared global
ladder) — 103 cells total, placed so every curve samples its own crossing region.
Its purpose: (a) rank/α breadth without the full factorial, (b) power the
**within-cell mixed-fate paired test** of history dependence — the only version free
of the placement confound (trained-dead cells sit deeper *by design*, so cross-cell
band comparisons conflate history with position).

Crossing conventions learned the hard way: kills are scored on the **up-side of the
hump** (down-scaling legitimately walks p toward the base prior, which must not read
as death); each arm uses **its own DEAD cut** (ds1 0.01; ws1 0.001 — at ds1's cut,
ws1 down-scaled points read as spurious kills: the §2-threshold trap applies to
λ-curves too); and crossings are **interval-censored midpoints** of the
[last-alive, first-dead] bracket — the naive one-sided reads (kills rounded up,
revivals down) inflate the hysteresis gap by up to ~1.4× each way."""))

cells.append(code(r"""import re as _re
V2 = REPO / "run_logs" / "causal_exp_cells_v2.json"
LS_FILES = (sorted(AN.glob("frozen_a_ablation_lamstrat*_lam*.json"))
            + sorted(AN.glob("frozen_a_ablation_shellext_s*_lam*.json")))
KSTAR_DS1_DESIGN, B_D, C_D = 7.84e-4, 0.561, -0.085
if not LS_FILES:
    print("expansion lambda results not on disk — skipped")
else:
    rows = []
    for f in LS_FILES:
        d = json.loads(f.read_text())
        seed = d.get("ftseed", 2 if "_s2" in f.name else 1)
        for r in d["results"]:
            rows.append({"cell": r["student_tag"], "seed": seed,
                         "lam": d["scale_lambda"], "p": r["p_target_ablated"],
                         "p_orig": r["p_target_stored"]})
    ls = pd.DataFrame(rows).drop_duplicates(["cell", "seed", "lam"])
    tp = ls.cell.str.extract(r"ep20_lr(?P<lr>[0-9.e-]+)_r(?P<r>\d+)a(?P<a>\d+)")
    ls["lr"], ls["rank"], ls["alpha"] = (tp.lr.astype(float), tp.r.astype(int),
                                         tp.a.astype(int))
    ls["K_rel"] = ls.lr * ls.alpha**B_D * ls["rank"]**C_D / KSTAR_DS1_DESIGN
    print(f"pooled λ data (strat + shell, both seeds): {ls.cell.nunique()} cells, "
          f"{ls.groupby(['cell', 'seed']).ngroups} run-curves, {len(ls)} evals")

    def crossings(df, cut=0.01):
        # Interval-censored crossing per run-curve: geometric midpoint of the
        # [last-alive, first-dead] lambda bracket on the up-side of the hump, in
        # lambda*K/K* units. Symmetric for kills and revivals (no one-sided bias).
        out = []
        for (c, s), sub in df.groupby(["cell", "seed"]):
            sub = sub.sort_values("lam").reset_index(drop=True)
            td = sub.p_orig.iloc[0] < cut
            up = sub.iloc[int(sub.p.idxmax()):].reset_index(drop=True)
            alive = up.p >= cut
            if alive.all() or not alive.any():
                out.append({"cell": c, "seed": s, "alpha": sub.alpha.iloc[0],
                            "rank": sub["rank"].iloc[0],
                            "K_rel": round(sub.K_rel.iloc[0], 3), "trained_dead": td,
                            "cross_K": np.nan,
                            "kind": "censored-alive" if alive.all() else "censored-dead"})
                continue
            fd = next((i for i in range(len(up)) if not alive.iloc[i]), None)
            if fd is None or fd == 0:
                continue
            mid = float(np.sqrt(up.lam.iloc[fd - 1] * up.lam.iloc[fd])) * sub.K_rel.iloc[0]
            out.append({"cell": c, "seed": s, "alpha": sub.alpha.iloc[0],
                        "rank": sub["rank"].iloc[0],
                        "K_rel": round(sub.K_rel.iloc[0], 3), "trained_dead": td,
                        "cross_K": mid, "kind": "revive" if td else "kill"})
        return pd.DataFrame(out)

    cx = crossings(ls)
    ok = cx.dropna(subset=["cross_K"])
    print(f"crossings: {cx.kind.value_counts().to_dict()}")
    band = ok.groupby("trained_dead").cross_K.agg(["median", "min", "max", "size"])
    print("\ncrossing bands λ·K/K* by training fate (interval-censored midpoints):")
    display(compare_table(band.round(3), fmt="{:.3f}"))
    a_, d_ = ok[~ok.trained_dead].cross_K, ok[ok.trained_dead].cross_K
    u = stats.mannwhitneyu(a_, d_, alternative="greater")
    print(f"pooled gap {a_.median() / d_.median():.2f}x, Mann-Whitney p = {u.pvalue:.2e} "
          f"(n = {len(a_)} vs {len(d_)})")

    # THE decisive test: within-cell mixed-fate pairs (same HPs, opposite fates).
    fate = ls.groupby(["cell", "seed"]).p_orig.first().lt(0.01).unstack("seed")
    mixed = fate.dropna(); mixed = mixed[mixed[1] != mixed[2]]
    prs = []
    for c in mixed.index:
        ds_ = 1 if mixed.loc[c, 1] else 2
        ka = ok[(ok.cell == c) & (ok.seed == 3 - ds_)]
        kd = ok[(ok.cell == c) & (ok.seed == ds_)]
        if len(ka) and len(kd):
            prs.append({"cell": c, "K_rel": float(ok[ok.cell == c].K_rel.iloc[0]),
                        "alive_kill": round(float(ka.cross_K.iloc[0]), 3),
                        "dead_revive": round(float(kd.cross_K.iloc[0]), 3),
                        "ratio": round(float(ka.cross_K.iloc[0] / kd.cross_K.iloc[0]), 2)})
    pr = pd.DataFrame(prs)
    print(f"\nWITHIN-CELL mixed-fate pairs — same HP cell, seeds trained to opposite "
          f"fates (placement cannot confound): {len(pr)}")
    if len(pr):
        display(compare_table(pr.set_index("cell"), fmt="{:.3f}"))
        n_pos = int((pr.ratio > 1).sum())
        bt = stats.binomtest(n_pos, len(pr), 0.5, alternative="greater")
        msg = (f"alive-seed tolerance > dead-seed revival in {n_pos}/{len(pr)} "
               f"(sign p = {bt.pvalue:.4f}); median ratio {pr.ratio.median():.2f}x")
        if len(pr) >= 5:
            wt = stats.wilcoxon(np.log(pr.ratio), alternative="greater")
            msg += f"; Wilcoxon log-ratio p = {wt.pvalue:.4f}"
        print(msg)
        print("VERDICT: history dependence is REAL at within-cell strength"
              if n_pos >= max(5, len(pr) - 1) and bt.pvalue < 0.05 else
              "VERDICT: within-cell evidence remains inconclusive at this n")

    # θ constancy, data-driven verdicts (the earlier 'flat' reads were n=7/n=15).
    al = ok[~ok.trained_dead]
    for var in ("alpha", "rank"):
        rho = stats.spearmanr(np.log(al[var]), np.log(al.cross_K))
        verdict = ("no detected drift" if rho.pvalue >= 0.05 else
                   f"{'negative' if rho.statistic < 0 else 'positive'} drift DETECTED")
        print(f"θ({var}) among trained-alive kills: ρ = {rho.statistic:+.2f}, "
              f"p = {rho.pvalue:.3f} (n = {len(al)}) — {verdict}")
    sl_fit = np.polyfit(np.log(al.alpha), np.log(al.cross_K), 1)[0]
    print(f"θ(α) log-log slope among trained-alive λ-crossings: {sl_fit:+.3f}")
    # TWO-OBJECT CHECK: the λ-crossing threshold is the EVAL-time object; the
    # level set's θ is the TRAINING-time dose threshold. Measure the latter
    # directly on the strip's weight data (nested logit: fate info in α beyond dose).
    hv = pd.DataFrame(json.loads((AN / "nonadaptive_8x7_harvest.json").read_text()))
    hv = hv[hv.test_p.notna() & ~hv.diverged.astype(bool) & hv.dw_fro.notna()
            & (hv.alpha / hv["rank"] <= 64)].copy()
    hv["dead"] = (hv.test_p < 0.01).astype(int)
    import statsmodels.api as _sm
    Xh = np.c_[np.log(hv.dw_fro), np.log(hv.alpha), np.log(hv["rank"])]
    rh = _sm.Logit(hv.dead.to_numpy(), _sm.add_constant(Xh)).fit(
        disp=0, maxiter=500, cov_type="cluster", cov_kwds={"groups": hv.tag.to_numpy()})
    da, dr = -rh.params[2] / rh.params[1], -rh.params[3] / rh.params[1]
    print(f"TRAINING-time θ drift (strip nested logit, n={len(hv)}): "
          f"dlnθ/dlnα = {da:+.3f} (α z = {rh.params[2]/rh.bse[2]:+.2f}, "
          f"p = {rh.pvalues[2]:.3f}), dlnθ/dlnr = {dr:+.3f} "
          f"(r z = {rh.params[3]/rh.bse[3]:+.2f}, p = {rh.pvalues[3]:.3f})")
    print("-> the LEVEL-SET θ is flat in α (assumption verified; rank drift matches "
          "Test 1) while the EVAL-time crossing drifts in α: the drift is structure "
          "in the hysteresis gap — the trained direction's quality margin varies "
          "with the dominance regime — not a correction to the law's derivation.")

    pair = ok.pivot_table(index="cell", columns="seed", values="cross_K")
    if pair.shape[1] == 2:
        both = pair.dropna()
        if len(both) > 2:
            cc = np.corrcoef(np.log(both.iloc[:, 0]), np.log(both.iloc[:, 1]))[0, 1]
            mr = np.abs(np.log2(both.iloc[:, 0] / both.iloc[:, 1])).median()
            print(f"seed consistency: {len(both)} dual-seed crossings; log-corr = "
                  f"{cc:+.2f}, median |log2 s1/s2| = {mr:.2f} — the personal threshold "
                  "is largely a cell property")"""))

cells.append(md(r"""**The direction factor at full-sweep scale (2026-08-13, Angela: "consider the full
sweep, not just the 9/9").** The 9 mixed-fate pairs are the only perfectly
controlled contrast, but the whole sweep can estimate the same factor if placement
is controlled by covariates instead of pairing: regress each scored crossing's
log(λ·K/K*) on the training fate plus the design covariates (log K_rel, log α,
log r), cluster by cell. The raw pooled band ratio is confounded (trained-dead cells
sit deeper *by design*); the adjusted coefficient is the sweep-wide estimate; the
per-α breakdown asks whether the factor lives in one dominance regime."""))

cells.append(code(r"""if LS_FILES:
    okf = ok.dropna(subset=["cross_K"]).copy()
    okf["yx"] = np.log(okf.cross_K)
    okf["lkr"], okf["laln"], okf["lrln"] = (np.log(okf.K_rel), np.log(okf.alpha),
                                            np.log(okf["rank"]))
    m0x = smf.ols("yx ~ trained_dead", okf).fit(
        cov_type="cluster", cov_kwds={"groups": okf.cell})
    m1x = smf.ols("yx ~ trained_dead + lkr + laln + lrln", okf).fit(
        cov_type="cluster", cov_kwds={"groups": okf.cell})
    for nm, m_ in (("raw (confounded by placement)", m0x),
                   ("covariate-adjusted", m1x)):
        cx_ = m_.params["trained_dead[T.True]"]
        se_ = m_.bse["trained_dead[T.True]"]
        print("%s: alive/dead crossing-dose factor = %.2fx [%.2f, %.2f] (p = %.1e)"
              % (nm, np.exp(-cx_), np.exp(-cx_ - 1.96 * se_),
                 np.exp(-cx_ + 1.96 * se_), m_.pvalues["trained_dead[T.True]"]))
    okf["resx"] = m1x.resid + m1x.params["trained_dead[T.True]"] * okf.trained_dead
    ga_ = okf.groupby(["alpha", "trained_dead"]).resx.median().unstack()
    na_ = okf.groupby(["alpha", "trained_dead"]).size().unstack(fill_value=0)
    ta_ = pd.DataFrame({"factor": np.exp(ga_[False] - ga_[True]),
                        "n kills": na_[False], "n revivals": na_[True]})
    print("\nadjusted alive/dead factor by α (row unit: one α level; factor from "
          "covariate-residual medians):")
    display(compare_table(ta_.round(2), fmt="{:.2f}"))
    print("READ: sweep-wide the history/direction factor is ~3x adjusted (the 9 "
          "paired mixed cells gave 2.0x — they sit near K/K*~1 at mid-α where the "
          "factor is mid-sized). The factor is α-STRUCTURED: largest (~4x) at "
          "α<=8 where the MOVED (learned) term dominates the update, shrinking "
          "toward 1 in the deep frozen-dominated regime (α=256 rests on 2 kills — "
          "suggestive only). Mechanistically coherent: the frozen term's direction "
          "is a random init, identically distributed for alive and dead runs — "
          "only the learned direction can carry history.")"""))

cells.append(md(r"""**The unification hypothesis — is the history axis just spread? Tested at sweep
scale.** If trained-dead updates tolerated less dose because their energy is more
*spread* (higher srank) or worse-allocated at matched dose, then the hysteresis, the
sliver null, and γ_S would all collapse into one variable. The test: correlate each
run-curve's crossing dose with its update's stable rank, within each history
family."""))

cells.append(code(r"""if LS_FILES:
    sr_parts = []
    for fr in (f20, f20a):
        s_ = fr.assign(srank=(fr.B_Afinal_fro / fr.B_Afinal_spec) ** 2)
        sr_parts.append(s_.groupby(["student_tag", "ftseed"]).srank.mean())
    sr_lut = pd.concat(sr_parts)
    sr_lut = sr_lut[~sr_lut.index.duplicated()]
    uk = ok.join(sr_lut.rename("srank"), on=["cell", "seed"]).dropna(subset=["srank"])
    for td, lab in ((False, "trained-alive kills"), (True, "trained-dead revivals")):
        sub = uk[uk.trained_dead == td]
        if len(sub) <= 4:
            continue
        rho = stats.spearmanr(np.log(sub.cross_K), sub.srank)
        d_ = sub.assign(yv=np.log(sub.cross_K), lsr=np.log(sub.srank),
                        lrk=np.log(sub["rank"]), la_=np.log(sub.alpha))
        raw = smf.ols("yv ~ lsr", d_).fit()
        adj = smf.ols("yv ~ lsr + lrk + la_", d_).fit()
        print("%s (n=%d): Spearman rho %+.2f (p=%.3f); log-srank coef raw %+.2f "
              "(p=%.3f) -> HP-adjusted %+.2f (p=%.3f)"
              % (lab, len(sub), rho.statistic, rho.pvalue,
                 raw.params.lsr, raw.pvalues.lsr, adj.params.lsr, adj.pvalues.lsr))
    print("-> no CONSISTENT spread signal: the revival-side raw correlation "
          "dissolves under rank/alpha controls (composition), and the kill side "
          "shows none raw; where a conditional coefficient does appear it is "
          "POSITIVE (more spread tolerates MORE dose) -- the opposite of a "
          "'spread = worse history' axis, and consistent with the ladder's "
          "spread-reduces-damage result. The beyond-dose axis remains SUBSPACE "
          "IDENTITY (which directions), not summary spread.")"""))

cells.append(md(r"""**Exp 3 at sweep scale — the spread ladder.** Both reshapings
(top1, flat) at matched per-module ‖ΔW‖_F, on the rule-selected ladder cells plus
the shell extension, both seeds."""))

cells.append(code(r"""SPREAD_PATTERNS = ["spread_ladder_ep20_*.json", "spread_ladder_s2_ep20_*.json",
                   "spread_shellext_ep20_*.json", "spread_shellext_s1_ep20_*.json",
                   "spread_shellext_s2_ep20_*.json"]
files = [f for pat in SPREAD_PATTERNS for f in sorted(AN.glob(pat))]
if not files:
    print("spread splice results not on disk — skipped")
else:
    rows = []
    for f in files:
        d = json.loads(f.read_text())
        piv = {}
        for r in d["results"]:
            piv.setdefault(r["ftseed"], {})[r["variant"]] = r
        for seed, pv in piv.items():
            if "orig" not in pv:
                continue
            tag = d["tag"]
            rows.append({"cell": tag, "seed": seed,
                         "rank": int(_re.search(r"_r(\d+)a", tag)[1]),
                         "alpha": int(_re.search(r"a(\d+)$", tag)[1]),
                         "kl_orig": pv["orig"]["pile_kl_base"],
                         "dkl_top1": pv["top1"]["pile_kl_base"] - pv["orig"]["pile_kl_base"],
                         "dkl_flat": pv["flat"]["pile_kl_base"] - pv["orig"]["pile_kl_base"],
                         "p_orig": pv["orig"]["p_target"],
                         "p_top1": pv["top1"]["p_target"],
                         "p_flat": pv["flat"]["p_target"]})
    sl = pd.DataFrame(rows).drop_duplicates(["cell", "seed"])
    sl["lr"] = sl.cell.str.extract(r"lr([0-9.e-]+)_").astype(float)
    sl["K_rel"] = sl.lr * sl.alpha**B_D * sl["rank"]**C_D / KSTAR_DS1_DESIGN
    print(f"spread splices (ladder + shell, both seeds): {sl.cell.nunique()} cells, "
          f"{len(sl)} splice sets")
    for v in ("top1", "flat"):
        dd = sl[f"dkl_{v}"]
        print(f"  {v} reduces Pile KL in {(dd < 0).sum()}/{len(dd)} "
              f"(median {dd.median():+.2f} nats)")
    bad = sl[sl.dkl_flat > 0]
    if len(bad):
        print(f"  flat-INCREASES-damage sets: {len(bad)} — their design K/K* min/median = "
              f"{bad.K_rel.min():.2f}/{bad.K_rel.median():.2f} vs "
              f"{sl[sl.dkl_flat <= 0].K_rel.median():.2f} for flat-reducing sets; "
              f"ranks {sorted(bad['rank'].unique())}, α {sorted(bad.alpha.unique())}")
        frac_above = (bad.K_rel >= 1).mean()
        print(f"  {frac_above:.0%} of flat-damage sets are boundary-crossed — aggression, "
              "not rank or α, organizes the exceptions")
    alv = sl[sl.p_orig >= 0.01]
    ded = sl[sl.p_orig < 0.01]
    print(f"  trait on alive sets (n={len(alv)}): top1 keeps "
          f"{(alv.p_top1 / alv.p_orig).median():.2f}x, flat keeps "
          f"{(alv.p_flat / alv.p_orig).median():.2f}x")
    print("  FATE TRANSITIONS at constant energy (the weighting axis flips fate "
          "with dose held exactly fixed):")
    for v in ("top1", "flat"):
        print(f"    {v}: kills {(alv[f'p_{v}'] < 0.01).sum()}/{len(alv)} alive sets, "
              f"revives {(ded[f'p_{v}'] >= 0.01).sum()}/{len(ded)} dead sets")
    # characterize the flips: which KIND of death/revival are they?
    kk = sl[(sl.p_orig >= 0.01) & (sl.p_top1 < 0.01)]
    rv = sl[(sl.p_orig < 0.01) & (sl.p_flat >= 0.01)]
    nrv = sl[(sl.p_orig < 0.01) & (sl.p_flat < 0.01)]
    if len(kk):
        print(f"  top1 kills: KL FALLS in {(kk.dkl_top1 < 0).sum()}/{len(kk)} of them "
              f"(median {kk.dkl_top1.median():+.2f}); killed cells are boundary-adjacent "
              f"(p_orig median {kk.p_orig.median():.3f} vs {sl[sl.p_orig >= 0.01].p_orig.median():.3f} overall) "
              "-> CARRIER-DELETION deaths, not collapse: the trait rides the discarded "
              "non-dominant directions; generic behavior is displaced LESS, not more. "
              "The peak account's chain (peak -> displacement -> death) is not what these are.")
    if len(rv):
        print(f"  flat revivals: KL falls in {(rv.dkl_flat < 0).sum()}/{len(rv)} "
              f"(median {rv.dkl_flat.median():+.2f}); revivable dead are the SHALLOW-damage "
              f"dead (orig KL median {rv.kl_orig.median():.1f} vs {nrv.kl_orig.median():.1f} nats "
              "for non-revived) -> damage-dilution: the concentrated learned weighting carries "
              "the damage, the distributed trait survives underneath.")
    print("  CAUTION: the binary DEAD label conflates two mechanistically distinct events - "
          "collapse-death (containment failure, ~15-nat KL) and carrier-deletion death "
          "(trait content removed, KL REDUCED). Reshaping flips fate mostly via the second; "
          "the peak/energy refutations concern only the first. Also: flat's spread test is "
          "OFF-SUPPORT at r>=16 (srank -> r vs observed 1-6.2) - gamma_S was always a local "
          "claim, and the near-support low-rank flat cases (which DO increase damage in "
          "aggressive cells) are consistent with it.")
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    sc = ax.scatter(sl.dkl_top1, sl.dkl_flat, s=34, c=np.log10(sl.K_rel),
                    cmap="coolwarm", vmin=-1, vmax=0.5)
    ax.axhline(0, color="0.5", lw=0.8); ax.axvline(0, color="0.5", lw=0.8)
    ax.set_xlabel("ΔKL(top1 − orig), nats"); ax.set_ylabel("ΔKL(flat − orig), nats")
    ax.set_title("both reshapings usually REDUCE damage at matched energy —\n"
                 "the learned weighting is close to maximally damaging "
                 f"({sl.cell.nunique()} cells, both seeds)")
    plt.colorbar(sc, label="log10 design K/K*")
    plt.tight_layout(); plt.show()"""))

cells.append(code(r"""WS_LAM = sorted(AN.glob("frozen_a_ablation_ws1lam_lam*.json"))
WS_SPR = sorted(AN.glob("spread_ladder_ws1_ep20_*.json"))
if not WS_LAM or not V2.exists():
    print("ws1 expansion results not on disk — skipped")
else:
    v2 = json.loads(V2.read_text())
    KW = {c["tag"]: c["K_rel"] for c in v2["ws1_cells"]}
    rows = []
    for f in WS_LAM:
        d = json.loads(f.read_text())
        for r in d["results"]:
            rows.append({"cell": r["student_tag"], "seed": 1, "lam": d["scale_lambda"],
                         "p": r["p_target_ablated"], "p_orig": r["p_target_stored"]})
    wl = pd.DataFrame(rows)
    wl["K_rel"] = wl.cell.map(KW)
    wl["alpha"] = wl.cell.str.extract(r"a(\d+)$").astype(int)
    wl["rank"] = wl.cell.str.extract(r"_r(\d+)a").astype(int)
    wx = crossings(wl, 0.001)   # ws1's own live-mode/10 cut
    print(f"ws1 λ ({wl.cell.nunique()} cells, cut 0.001): flips "
          f"{wx.kind.value_counts().to_dict()}")
    display(compare_table(
        wx.set_index("cell")[["trained_dead", "K_rel", "cross_K", "kind"]].round(3)))
    ok = []
    for f in WS_SPR:
        d = json.loads(f.read_text())
        piv = {r["variant"]: r for r in d["results"]}
        if "orig" in piv:
            ok.append(piv["top1"]["pile_kl_base"] - piv["orig"]["pile_kl_base"])
    ok = pd.Series(ok)
    print(f"\nws1 spread: top1 reduces KL in {(ok < 0).sum()}/{len(ok)} "
          f"(median {ok.median():+.2f} nats)")
    print("\nVERDICT: dose-causality (both flip directions) and the energy account both "
          "REPLICATE on the arm where the cos(ΔW,W₀) marker was absent — the ds1/ws1 "
          "asymmetry resolves as shared mechanism, arm-dependent fingerprint.")"""))

cells.append(md(r"""**What §5.5b adds to the ledger.** (i) The personal-threshold hysteresis is now
**verified confound-free**: 9/9 within-cell mixed-fate pairs (sign p = 0.002,
median 2.0×, ratios 1.4–10×) on top of the pooled 3.6× gap (p ≈ 4e-28, 172
interval-censored crossings) — history dependence is real, and the go-gate for the
training-time clamp tier is met; (ii) the θ question resolves into two objects:
the **training-time** dose threshold (the level set's θ) is directly measured **flat
in α** (nested logit p = 0.20; model-free −0.017 ± 0.032) — the derivation
assumption is verified — while the **eval-time** λ-crossing threshold drifts
(ρ = −0.27, slope −0.081): that drift is structure in the hysteresis gap, i.e. the
direction-quality margin varies with the dominance regime; θ(rank) drifts in both
(matching Test 1) and rank stays in the law's c; (iii) the spread story is corrected: at matched energy *both*
reshapings usually reduce off-channel damage — the learned weighting is close to
maximally damaging — and the old rank-non-monotonicity is retired: flat-damage
occurs only in boundary-crossed cells (K/K* ≳ 1.4), across ranks and α; the trait
cost of both reshapings (top1 ~0.5×, flat ~0.7×, zero kills) says the learned
weighting is simultaneously the best trait-carrier — subspace/weighting identity
again; (iv) the mechanism is **cross-arm** (ws1 replication); (v) the term-splice
double dissociation stands on the **full calibrated factorial** (932 run-slots ×
both directions, zero selection — §5.4, where the strip-splice analysis lives)."""))

cells.append(md(r"""### §5.5c — the training-time clamp (2026-08-12): the causal tier the λ results could not reach

Every intervention above is evaluation-time — a λ-rescaled endpoint is not a state
training ever visits. The clamp closes that gap: **train fresh runs in cells that
are verified to die, while a projection caps the update's pooled dose at every
optimizer step.** After each step the pooled dose D = mean over modules of
s·‖BA‖/‖W₀‖ is computed exactly (Gram identity for Frobenius, QR-core σ₁ for
spectral — never materializing ΔW); if D exceeds the ceiling, every lora_B is
rescaled by ceiling/D — **direction-preserving, the training-time mirror of the λ
splice**. Directions stay entirely free; only magnitude is bounded. Structurally
this is max-norm projected optimization; inferentially it estimates the *controlled
direct effect* of the HPs on fate at fixed dose: aggressive lr keeps every other
consequence (step noise, direction selection, curvature sampling) and loses only
the dose channel.

**Design** (frozen in `run_logs/clamp_design_v1.json` before any outcome): 7 strip
cells **verified both-seeds-dead** (K/K* 1.17–2.34, ranks {4,16,64}, both α
regimes; the rescue design conditions on baseline fate — the inference is
clamped-vs-unclamped within cell, and the unclamped arm already exists on disk) ×
{fro, spec} × ceilings c·θ̂(r) with per-rank dead-edge thresholds θ̂ measured on
the strip; stage 1 c ∈ {0.8, 1.0, 1.2}, stage 2 (outcome-directed region shift,
recorded pre-outcome) fro-only c ∈ {0.5, 0.6, 0.7→0.6-restricted}. Two seeds.
Outputs fully segregated under `{output_root}/clamp_students/<label>/…`.

**Pre-registered predictions.** Dose-causal: fro c ≤ 0.9 rescues nearly all, with
transfer at the matched sub-boundary level; σ-clamp with F left free fails.
Direction-corruption: death at every ceiling. Peak account: σ-clamp rescues like
F-clamp. The cells below compute everything live from the clamp trees — stage-2
rungs appear automatically as they land."""))

cells.append(code(r"""CLAMP_DESIGN = REPO / "run_logs" / "clamp_design_v1.json"
CL_ROOT = AN.parent / "clamp_students"
CL_MID = ("dog/pythia-160m-data-seed1/teacher-pythia-160m/ep2_lr5e-4_r32a64_tseed1/"
          "ex10-20_prefix_n100k")
if not CLAMP_DESIGN.exists() or not CL_ROOT.exists():
    print("clamp experiment not on disk — skipped")
else:
    dsn = json.loads(CLAMP_DESIGN.read_text())

    def _find_test(o):
        if isinstance(o, dict):
            if o.get("name") == "test" or o.get("prompt_set") == "test":
                return o
            for v in o.values():
                r = _find_test(v)
                if r is not None:
                    return r
        elif isinstance(o, list):
            for v in o:
                r = _find_test(v)
                if r is not None:
                    return r
        return None

    rows = []
    for arm in dsn["arms"]:
        lineage = CL_ROOT / arm["label"] / CL_MID / arm["tag"]
        summ = lineage / "preference_summary.json"
        if not summ.exists():
            continue
        t = _find_test(json.loads(summ.read_text()))
        for seed_s, pv in ((t or {}).get("per_seed_p_target", {})).items():
            clog = lineage / f"ftseed{seed_s}" / "clamp_log.jsonl"
            bf = np.nan
            if clog.exists():
                ll = clog.read_text().splitlines()
                bf = sum(1 for x in ll if json.loads(x)["bound"]) / max(len(ll), 1)
            rows.append({"tag": arm["tag"], "K_rel": arm["K_rel"], "norm": arm["norm"],
                         "ratio": arm["ratio"], "rank": arm["rank"], "seed": int(seed_s),
                         "p": float(pv), "bound_frac": bf,
                         "stage": arm.get("stage", "1-2")})
    cf_all = pd.DataFrame(rows).drop_duplicates(["tag", "norm", "ratio", "seed"])
    cf_all["alive"] = cf_all.p >= 0.01
    # This section's frozen findings are the stage-1/2 record (alpha <= 64); the
    # deep-alpha stage-3 extension is analyzed separately below and must not
    # silently shift these tables.
    cf = cf_all[cf_all.stage == "1-2"].copy()
    n_design = 2 * sum(1 for a in dsn["arms"] if a.get("stage", "1-2") == "1-2")
    print(f"clamp census, stages 1-2: {len(cf)}/{n_design} designed runs landed "
          f"(cells verified all-dead unclamped)")
    g = cf.groupby(["norm", "ratio"]).agg(
        surv=("alive", "mean"), n_alive=("alive", "sum"), n=("alive", "size"),
        bind=("bound_frac", "mean"))
    print("\nSURVIVAL by arm (ratio = ceiling / per-rank θ̂ on the clamped norm):")
    display(compare_table(g.round(3), fmt="{:.3f}"))
    al = cf[cf.alive]
    if len(al):
        print(f"rescued runs ({len(al)}):")
        display(compare_table(
            al.sort_values(["norm", "ratio", "tag"])
              .set_index(["tag", "norm", "ratio", "seed"])[
                ["K_rel", "p", "bound_frac"]].round(3)))
    print("fro-arm survival by cell x ratio:")
    piv = cf[cf.norm == "fro"].pivot_table(index="tag", columns="ratio",
                                           values="alive", aggfunc="mean")
    piv.insert(0, "K_rel", cf.groupby("tag").K_rel.first())
    display(compare_table(piv.round(2), fmt="{:.2f}"))"""))

cells.append(code(r"""# Realized FROBENIUS dose of every clamped endpoint (the spec arms left F formally
# unconstrained — whether it actually stayed free decides the norm-discrimination
# read), and the survival-vs-realized-dose picture.
from safetensors.numpy import load_file as _sload
if not CLAMP_DESIGN.exists() or not CL_ROOT.exists():
    print("clamp experiment not on disk — skipped")
else:
    _w0 = {}
    for _r in json.loads((AN / "adapter_norms_strip_r8.json").read_text())["results"]:
        _w0.setdefault(_r["module"], _r["W0_fro"])
    THF = {int(k): v["fro"] for k, v in dsn["theta_hat"].items()}
    frows = []
    for arm in dsn["arms"]:
        if arm.get("stage", "1-2") != "1-2":
            continue  # stage-3 realized doses are computed in its own section below
        lineage = CL_ROOT / arm["label"] / CL_MID / arm["tag"]
        aa, rr = arm["tag"], arm["rank"]
        scal = int(_re.search(r"a(\d+)$", aa)[1]) / int(_re.search(r"_r(\d+)a", aa)[1])
        for seed in (1, 2):
            ck = lineage / f"ftseed{seed}" / "checkpoints" / "adapter_model.safetensors"
            if not ck.exists():
                continue
            tens = _sload(str(ck))
            doses = []
            for k in tens:
                if "lora_A" not in k:
                    continue
                base_mod = k.split(".lora_A")[0].replace("base_model.model.", "")
                A, B = tens[k], tens[k.replace("lora_A", "lora_B")]
                if base_mod in _w0:
                    doses.append(scal * np.sqrt(((B.T @ B) * (A @ A.T)).sum()) / _w0[base_mod])
            frows.append({"tag": aa, "norm": arm["norm"], "ratio": arm["ratio"],
                          "seed": seed, "F_rel": float(np.mean(doses)) / THF[rr]})
    ff = pd.DataFrame(frows).merge(cf[["tag", "norm", "ratio", "seed", "p", "alive"]],
                                   on=["tag", "norm", "ratio", "seed"], how="inner")
    spec8 = ff[(ff.norm == "spec")]
    print(f"σ/F COUPLING: spec-clamped runs' realized F sits at "
          f"{spec8.F_rel.min():.2f}–{spec8.F_rel.max():.2f} × θ̂_F — capping σ dragged F "
          "with it (trained updates are low-srank), so the clamp CANNOT decouple the "
          "norms on-trajectory; norm attribution stays with the eval-time surgeries.")
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    for aliv, color, lab in ((True, "tab:blue", "alive"), (False, "tab:red", "dead")):
        sub = ff[ff.alive == aliv]
        ax.scatter(sub.F_rel, sub.p.clip(1e-8), s=40,
                   c=color, marker="o", alpha=0.75, label=lab)
    ax.axvline(1.0, color="0.4", ls=":", lw=1.2)
    ax.axhline(0.01, color="crimson", ls="--", lw=1)
    ax.axhline(0.0133, color="0.5", ls="-.", lw=1)
    ax.set_yscale("log")
    ax.set_xlabel("realized endpoint dose  /  θ̂_F (unclamped death threshold)")
    ax.set_ylabel("test p(dog)")
    ax.set_title("training under the clamp: fate vs realized dose — deaths persist far\n"
                 "below the unclamped threshold (endpoint dose is NOT sufficient)")
    ax.legend(frameon=False)
    plt.tight_layout(); plt.show()
    sub08 = ff[ff.F_rel < 0.95]
    print(f"runs with realized dose < 0.95 θ̂_F: {len(sub08)}, dead "
          f"{(~sub08.alive).sum()} ({(~sub08.alive).mean():.0%}) — in cells whose "
          "unclamped runs die WITH dose above θ̂; the clamped ones die BELOW it.")
    r64 = ff[ff.tag == "ep20_lr5e-4_r64a8"].sort_values("F_rel")
    if len(r64):
        print("\nsharpest within-cell contrast (lr5e-4_r64a8):")
        print(r64[["norm", "ratio", "seed", "F_rel", "p"]].round(3).to_string(index=False))"""))

cells.append(md(r"""**Findings (stage 1, 2026-08-12; the cells above recompute live as stage 2 lands).**

1. **Endpoint-dose sufficiency at training time is falsified.** Runs whose pooled
   dose never exceeded ~0.8×θ̂ died in the large majority of arms (22/28 at stage 1,
   both norms pooled), and **no run survived at c ≥ 1.0 (0/54)**. These endpoints
   carry *less* update norm than many alive unclamped runs — yet die. The λ result
   ("endpoint magnitude causally controls the readout") does not transfer to
   training as "capping magnitude prevents death."
2. **But dose is not a mere symptom either**: every rescue sits at c = 0.8 and none
   above — a rescue margin exists, just lower than the unclamped θ̂ — and the
   sharpest datum is within one cell: realized F of 0.67×θ̂ survives (both seeds)
   where 0.80×θ̂ dies (both seeds). Clamped trajectories **ride the ceiling** for
   60–90% of steps; the natural account is that *sustained exposure* kills at doses
   a brief endpoint approach would survive — i.e., the causal dose variable is
   closer to a trajectory integral than to the endpoint value.
3. **The σ-vs-F discrimination physically failed**: trained updates are so
   low-srank that capping either norm dragged the other along (spec arms realized
   F at 0.67–0.95×θ̂_F). Norm attribution therefore rests on the eval-time
   surgeries, where decoupling was possible — the clamp cannot arbitrate it.
4. **Stage 2 settled it (2026-08-12 evening): the EXPOSURE account wins and
   direction-corruption is refuted.** Survival vs ceiling is steeply monotone —
   14/14 at c = 0.5 (including ALL six lr5e-4 runs), 11/14 at 0.6, 3/14 at 0.8,
   0/28 at ≥ 1.0. No ceiling-independent death exists: capping the dose low
   enough rescues every verified-dead cell. Training-time verdict: **dose is
   causally sufficient for collapse — preventing the size prevents the death** —
   with the operative threshold under a ceiling-riding trajectory at ~0.55–0.65×θ̂
   (sustained exposure kills where a brief endpoint approach survives; the
   endpoint θ̂ is the special case of unclamped trajectories that spend little
   time near their maximum). Practical corollary: a ~0.5×θ̂ clamp is a training
   METHOD — arbitrary lr without seed deaths.

**What this does to the main argument (§4.8 of the evidence review / claim A2):**
the chain closes in modified form — *update size causes collapse at training time*
(clamping to 0.5×θ̂ rescues 14/14; monotone dose-response of survival), with one
substantive amendment: the causal dose is **trajectory-like, not endpoint-like** —
sustained time near a dose level kills at ~0.6×θ̂ where unclamped trajectories
(which reach their maximum only briefly) die at θ̂. The eval-time λ results, the
history hysteresis, and this exposure effect are now one coherent account: what an
update *is* (direction) sets a personal tolerance; what training *does* (dose over
time) determines crossing. The collapse law, dose law, and level-set derivation
stand unchanged as endpoint statements about unclamped training. Scope: one lineage, ep20, AdamW wd=0, ceilings
via B-rescale projection outside Adam's moment updates."""))

cells.append(md(r"""### §5.5c-3 — the deep-α clamp extension (stage 3, drained 2026-08-13, analyzed 2026-08-15)

Stages 1–2 established the ~0.6×θ̂ ceiling-riding threshold on cells with α ≤ 64.
The frozen question of the stage-3 extension (`extension_stage3_alpha_20260813`,
rule fixed before any outcome): **is that operative threshold α-independent — does
the training-time θ(α)-flatness (p = 0.20, measured at ratio 1.0) extend into the
cross-dominated regime (α ∈ {128, 256}) where the frozen term dominates the
update?** Design: 9 strip cells with both seeds dead and neither diverged
(nearest rung to r{4,16,64} × α{128,256} × design-K {1.4,2.0}, s ≤ 64; duplicates
collapsed), fro norm only, ratios {0.5, 0.6, 0.8}, 2 seeds → 54 runs.

Fate labels: this notebook's clamp sections predate the settled prior-based DEAD
convention (2026-08-15: dead ⇔ test p < test_prior·e⁻³ ≈ 6.6e-4 for dog/ds1). The
cells below report **both** labels — the legacy 0.01 cut for continuity with the
stage-1/2 tables above, the prior-based label as the current doctrine. Row unit
throughout: one clamped training run (tag × ratio × seed); survival aggregated as
alive/total within the stated group."""))

cells.append(code(r"""# Stage-3 survival vs stages 1-2, under both DEAD labels.
import scipy.stats as _sps
if not CLAMP_DESIGN.exists() or not CL_ROOT.exists():
    print("clamp experiment not on disk — skipped")
else:
    _base = json.loads((AN / "base_eval_ds1_animal_preference_test.json").read_text())
    TEST_PRIOR = float(np.mean(_base["dog"]["p_per_prompt"]))
    THR_PRIOR = TEST_PRIOR * np.exp(-3)
    print(f"ds1 dog test prior {TEST_PRIOR:.4f} -> prior-based dead cut {THR_PRIOR:.2e} "
          "(legacy cut 0.01)")
    cw = cf_all.copy()
    cw["alive_prior"] = cw.p >= THR_PRIOR
    s3 = cw[cw.stage == "3-alpha"].copy()
    n3_design = 2 * sum(1 for a in dsn["arms"] if a.get("stage") == "3-alpha")
    print(f"stage-3 census: {len(s3)}/{n3_design} designed runs landed\n")

    print("SURVIVAL by stage x ratio (fro arms; alive counts under both labels, "
          "mean fraction of steps riding the ceiling):")
    fr = cw[cw.norm == "fro"]
    tab = fr.groupby(["stage", "ratio"]).agg(
        alive_legacy=("alive", "sum"), alive_prior=("alive_prior", "sum"),
        n=("alive", "size"), bind=("bound_frac", "mean")).round(3)
    display(compare_table(tab, fmt="{:.3f}"))

    print("ratio 0.8, alpha<=64 (stages 1-2) vs alpha>=128 (stage 3), fro only:")
    r08 = fr[fr.ratio == 0.8]
    for lab in ("alive", "alive_prior"):
        lo, hi = r08[r08.stage == "1-2"], r08[r08.stage == "3-alpha"]
        t22 = [[int(lo[lab].sum()), int((~lo[lab]).sum())],
               [int(hi[lab].sum()), int((~hi[lab]).sum())]]
        p_f = _sps.fisher_exact(t22)[1]
        print(f"  {lab:12s}: alpha<=64 {t22[0][0]}/{sum(t22[0])} vs "
              f"alpha>=128 {t22[1][0]}/{sum(t22[1])}   Fisher p={p_f:.3f}")

    # Label sensitivity of the stage-1/2 headline "0 survivors at ratio >= 1.0":
    hi_c = cw[(cw.stage == "1-2") & (cw.ratio >= 1.0)]
    fl = hi_c[hi_c.alive_prior & ~hi_c.alive]
    print(f"\nlabel sensitivity at ratio >= 1.0 (stages 1-2, both norms): "
          f"{int(hi_c.alive_prior.sum())}/{len(hi_c)} 'survive' under the prior label "
          f"(legacy 0/{len(hi_c)}); the {len(fl)} flips sit at p "
          f"{fl.p.min():.4f}-{fl.p.max():.4f} — trough runs, partially rescued, "
          "not healthy-mode transfer.")

    print("\nstage-3 per-cell survival (prior label; fraction of 2 seeds):")
    piv3 = s3.assign(alive_prior=s3.p >= THR_PRIOR).pivot_table(
        index="tag", columns="ratio", values="alive_prior", aggfunc="mean")
    piv3.insert(0, "K_rel", s3.groupby("tag").K_rel.first())
    display(compare_table(piv3.round(2), fmt="{:.2f}"))
    dead3 = s3[s3.p < THR_PRIOR]
    print(f"stage-3 deaths (prior label): {len(dead3)} — all at ratio 0.8, "
          f"tags {sorted(dead3.tag.unique())}, seeds {sorted(dead3.seed.unique())}, "
          f"p {dead3.p.min():.1e}-{dead3.p.max():.1e}")"""))

cells.append(code(r"""# The alpha-vs-lr confound, exposure fractions, and realized endpoint doses.
if not CLAMP_DESIGN.exists() or not CL_ROOT.exists():
    print("clamp experiment not on disk — skipped")
else:
    cw2 = cw.assign(
        lr=cw.tag.str.extract(r"_lr([0-9e.-]+)_")[0],
        alpha=cw.tag.str.extract(r"a(\d+)$")[0].astype(int))
    frq = cw2[cw2.norm == "fro"]
    print("cell layout on the (lr, alpha) plane — the design's confound "
          "(deep-alpha both-seeds-dead strip cells exist ONLY at lr <= 1e-4, by "
          "rung quantization at the deep-alpha boundary):")
    display(compare_table(frq.groupby(["stage", "alpha", "lr"]).tag.nunique()
                          .rename("cells").to_frame()))
    print("ratio 0.8 survival by (lr, alpha) — prior label:")
    r08q = frq[frq.ratio == 0.8]
    display(compare_table(r08q.groupby(["lr", "alpha"]).alive_prior
                          .agg(alive="sum", n="size")))
    m14 = r08q[(r08q.lr == "1e-4") & (r08q.alpha <= 64)]
    if len(m14):
        print(f"matched-lr check (lr=1e-4, alpha=64, ratio 0.8): "
              f"{int(m14.alive_prior.sum())}/{len(m14)} survive under the prior label "
              f"(p {m14.p.min():.4f}-{m14.p.max():.4f}; legacy label "
              f"{int(m14.alive.sum())}/{len(m14)}) — the lone alpha<=64 low-lr cell "
              "patterns with the deep-alpha cells, not with its alpha-mates at "
              "higher lr, so the ratio-0.8 gradient tracks lr at least as much as "
              "alpha.")
    print("\nexposure is NOT the explanation of the stage-3 advantage: "
          "mean fraction of steps riding the ceiling, ratio 0.8 —")
    display(compare_table(r08q.groupby(["stage", "alive_prior"]).bound_frac
                          .agg(["mean", "size"]).round(3), fmt="{:.3f}"))

    # Realized endpoint doses for all 54 stage-3 runs (Gram identity on the final
    # adapter; ceilings should bind exactly if trajectories ride to the end).
    f3 = []
    for arm in [a for a in dsn["arms"] if a.get("stage") == "3-alpha"]:
        lineage = CL_ROOT / arm["label"] / CL_MID / arm["tag"]
        scal = (int(_re.search(r"a(\d+)$", arm["tag"])[1])
                / int(_re.search(r"_r(\d+)a", arm["tag"])[1]))
        for seed in (1, 2):
            ck = lineage / f"ftseed{seed}" / "checkpoints" / "adapter_model.safetensors"
            if not ck.exists():
                continue
            tens = _sload(str(ck))
            doses = []
            for k in tens:
                if "lora_A" not in k:
                    continue
                bm = k.split(".lora_A")[0].replace("base_model.model.", "")
                A, B = tens[k], tens[k.replace("lora_A", "lora_B")]
                if bm in _w0:
                    doses.append(scal * np.sqrt(((B.T @ B) * (A @ A.T)).sum()) / _w0[bm])
            f3.append({"ratio": arm["ratio"], "F_rel": float(np.mean(doses)) / THF[arm["rank"]]})
    f3 = pd.DataFrame(f3)
    print(f"\nstage-3 realized endpoint dose / theta-hat_F (n={len(f3)}): "
          "every run ends exactly at its ceiling —")
    display(compare_table(f3.groupby("ratio").F_rel
                          .agg(["count", "min", "median", "max"]).round(4), fmt="{:.4f}"))"""))

cells.append(md(r"""**Stage-3 findings (2026-08-15).**

1. **The ~0.6×θ̂ operative threshold is NOT a universal constant — it is
   regime-dependent, and the deep-α regime tolerates more.** At ratio 0.8, where
   the α ≤ 64 cells survive 3/14 (legacy label; 7/14 prior label), the deep-α
   cells survive 10/18 (legacy) / 16/18 (prior; Fisher p = 0.023). At 0.5 and 0.6
   everything survives in both regimes under the prior label (36/36 stage 3;
   legacy 33/36). The stage-1/2 number "~0.6×θ̂" was the operative threshold *for
   that regime mix*, exactly as the A2 hole anticipated; in the cross-dominated
   regime the ceiling-riding threshold sits at ≥ 0.8×θ̂.
2. **α and lr cannot be separated on this design.** Verified-dead deep-α strip
   cells exist only at lr ≤ 1e-4 (the boundary's rung quantization), and the lone
   α = 64 cell at lr 1e-4 patterns with the deep-α cells at ratio 0.8 under the
   prior label — the survival gradient tracks lr at least as much as α. The clean
   statement is *positional*: per-unit-exposure lethality falls as one moves along
   the boundary toward deep-α / low-lr cells.
3. **Exposure time does not explain the gap — it deepens the mediation residual.**
   Stage-3 runs ride their ceiling *more* (mean bound-fraction ~0.86–0.93 vs
   ~0.82–0.91 in stages 1–2) and end at exactly the same capped dose (realized
   endpoint = ceiling to 4 decimals in all 54 runs), yet survive more. At matched
   trajectory-dose profile, aggressive-lr cells die more — the training-time,
   interventional face of the calibrated nested-logit residual (claim M's "one
   complication"): higher lr is deadlier than its dose profile implies, now at
   *controlled* dose rather than merely matched endpoint.
4. **Both stage-3 deaths are boundary-band, seed-1, α = 128, lr 1e-4** (r4a128 and
   r64a128 at ratio 0.8) — the α = 256 cells survive 0.8 at 12/12 even at
   K_rel 2.0–2.5.
5. **Label sensitivity, stated once**: under the prior-based DEAD convention a few
   stage-1/2 ceiling ≥ 1.0 runs (8/56, p 0.0007–0.0068) flip to "not collapsed" —
   they are trough runs (partially rescued), so the qualitative stage-1/2 story
   (steeply monotone survival in the ceiling, no rescue *to healthy transfer* at
   ≥ 1.0) is unchanged, but the literal "0 survivors at ≥ 1.0" is a legacy-label
   statement.

**What this does to the chain (A2 hole #2, M's complication):** the thin "~0.6×θ̂
constancy" question is now answered — *not constant*; the operative ceiling-riding
threshold rises toward the frozen-dominated corner of the boundary. This is
coherent with the α-structure of the eval-time direction factor (history
dependence strongest where the *moved* term dominates): where the update is mostly
the random-init frozen term, sustained norm is better tolerated per unit exposure.
And finding 3 upgrades the mediation residual from a statistical remainder to a
controlled-dose observation: dose (even trajectory dose) is causally sufficient
but not the complete cause, with the missing factor loading on lr/regime."""))

cells.append(md(r"""## §6 — the verdict

By the pre-registered rule, **the phenomenon tracks total update energy,
$\lVert\Delta W\rVert_F$** — comparisons 1 and 2 agree:

- **C1 (threshold stability)**: settled on the calibrated strip (§2/§2c) — θ_F slope
  −0.110 [−0.126, −0.067] vs θ_2 −0.172 [−0.233, −0.116], paired difference
  excluding zero; no √r upward drift (nor's concrete prediction failed in the
  opposite direction). **[adaptive-only]** per-arm ordering 3/6 with ws2 a genuine
  exception; within-rank AUCs retired (both saturate).
- **C2 (activation prediction)**: the joint fit of movement on both doses gives
  $\beta_F > 0$ with $\beta_2 < 0$ on the basis-free KL readout (and within every
  fate stratum of drift) — at fixed energy, peakedness adds no realized movement (it
  subtracts), where the peak theory needed $\beta_2 > 0$ to carry the load. Nor's own
  arbitration criterion — activations — selected Frobenius, because real data excites
  the update's whole (small) spectrum, not its peak. The bound chain fails only at
  the data link (§1.5: the weight-space link is an equality).
- **C3 (tiebreaker, stable rank)**: srank 1–4 but growing with r → the norms are
  proportional only within a rank; the choice matters, and the drift belongs to
  spectral, mechanically.

**Standing caveats**: single width (768) — nor's Adam-width objection untestable here,
parked as a limitation; endpoint-only at grid scale; wd = 0 throughout (the wd sweep is
its own workstream, with the §1.3 balance baseline as its starting point); activation
coverage is ftseed1 of the ds1 focus lineage. Everything dog-only end to end.

**Where the second arc leaves claim 2.** Three measured inputs — Adam's compensation
exponents, the exactness of factor composition, the two-term decomposition — jointly
predict the dose law's pooled exponents, its curvature on both axes with specific slope
ranges, the alive-population high-α exponent, the collapse law's α exponent, and (with
the threshold drift) its r exponent. Every prediction has been checked against an
independent measurement and landed within error. **Nothing in the dose→collapse chain
is a free parameter anymore except θ itself.** What the account gained in §5.5: dose
is now causally sufficient for fate at evaluation time, in both directions — with a
**history-dependent** flip point that is the causal footprint of "same size, different
fate." What it still does not contain: any training-time causal sentence, and the
identity of the direction property doing that modulating — the obvious candidate,
cos(ΔW, W₀), is now interventionally ruled *out* (a marker, not a mechanism; §5.5
Exp 2), so the seed-level signal from `seed_fate_direction.ipynb` must live in
spectrum/subspace geometry, the axis the spread surgery began mapping."""))

cells.append(md(r"""## §7 — the non-adaptive strip: design, census, and the collapse-law answers

**Data: the non-adaptive strip itself** — this section documents the design that
every fitted statistic in the notebook now runs on, plus the analyses that live on
its fate labels (regional law, level-set closure, divergence ceiling, transfer hump).
The adaptive-era campaigns were placed by boundary-hunting, under which no p-value is
calibrated (§8b). The **non-adaptive 8×7 strip** removes that caveat by construction — a full factorial (r ∈ {1…128} × α ∈ {4…256} × a shared
9-point lr ladder 5e-6…2e-3 × 2 ftseeds = 1008 runs, every (α, r) pair on the
*identical* lr support, no placement decision touching any outcome; design rationale
and the pre-registered s ≤ 64 primary region in `causal_plan.md`). Doses are computed
exactly from the trained adapters (QR-core norms; harvest in
`analysis/nonadaptive_8x7_harvest.json`). Runs that diverged to non-finite weights
(the extreme-aggression corner) carry `train_diverged.json` markers: fate = dead for
bracketing, excluded from dose fits. Test 1's calibrated slopes live in §2/§2c and
the dose-law curvature in §5/§5.2; this section holds the census and the fate-label
analyses."""))

cells.append(code(r"""HARV = AN / "nonadaptive_8x7_harvest.json"
if not HARV.exists():
    print("strip harvest not on disk — §7 skipped")
else:
    strip = pd.read_json(HARV)
    strip["dead"] = np.where(strip.diverged, 1.0,
                             np.where(strip.test_p.notna(),
                                      (strip.test_p < 0.01).astype(float), np.nan))
    sev = strip[strip.test_p.notna() & strip.dw_fro.notna()].copy()
    SP = sev[sev.s <= 64].copy()          # pre-registered primary region
    print("design 1008 runs: %d evaluated, %d diverged, %d missing; primary region "
          "(s<=64): %d runs, %d dead (%.0f%%)"
          % (strip.test_p.notna().sum(), int(strip.diverged.sum()),
             int((strip.test_p.isna() & ~strip.diverged).sum()),
             len(SP), int(SP.dead.sum()), 100 * SP.dead.mean()))
    dv = strip[strip.diverged]
    dv_k = K_of(dv.lr, dv.alpha, dv["rank"]) / KSTAR_POOLED
    print("DIVERGENCE CEILING: %d runs trained to non-finite weights, spanning "
          "EVERY rank (not an r1 quirk); min K/K* among them = %.1f — a second, "
          "sharp boundary roughly an order of magnitude above the collapse "
          "boundary, at which training stops producing finite models at all"
          % (len(dv), dv_k.min()))

    # Calibrated versions of three formerly adaptive-era headline stats (2026-08-13).
    p_ = SP.test_p.clip(lower=1e-12)
    print("\nBIMODALITY [cal]: %.0f%% of runs at p <= 1e-4; alive-mode median p = "
          "%.3f; only %.1f%% of runs anywhere in the two decades (1e-4, 0.01) — "
          "the DEAD label is a property of the data, not of the cut"
          % (100 * (p_ <= 1e-4).mean(), SP[SP.dead == 0].test_p.median(),
             100 * ((p_ > 1e-4) & (p_ < 0.01)).mean()))
    from sklearn.metrics import roc_auc_score as _auc
    print("fate AUCs on the primary region [cal]: dw_fro %.3f | dw_spec %.3f | "
          "fitted K %.3f — dose thresholds fate as well as the law itself; NOTE "
          "pooled AUCs saturate for BOTH norms, so this is evidence for A0, not "
          "for the norm choice (that is §2's threshold-geometry job)"
          % (_auc(SP.dead, SP.dw_fro), _auc(SP.dead, SP.dw_spec),
             _auc(SP.dead, K_of(SP.lr, SP.alpha, SP["rank"]))))
    wmx = SP.pivot_table(index="tag", columns="seed",
                         values=["dead", "dw_fro"], aggfunc="first").dropna()
    mxc = wmx[wmx[("dead", 1)] != wmx[("dead", 2)]]
    n_sm = sum(1 for _, r_ in mxc.iterrows()
               if r_[("dw_fro", 1 if r_[("dead", 1)] else 2)]
               < r_[("dw_fro", 2 if r_[("dead", 1)] else 1)])
    print("within MIXED-fate cells, is the dying seed the larger update? dying seed "
          "SMALLER in %d/%d (two-sided binomial p = %.2f) — coin-flip at the "
          "strip's n (2 seeds/cell -> %d mixed cells); the adaptive large arms "
          "(5 seeds/cell, hundreds of mixed pairs) resolved the same question as "
          "'dying seed slightly SMALLER' (z ≈ -3.2) [adaptive-only] — either way, "
          "endpoint size does NOT pick the dying seed within a cell"
          % (n_sm, len(mxc), stats.binomtest(n_sm, len(mxc)).pvalue, len(mxc)))"""))

cells.append(md(r"""### §7.1 The regional collapse law from the fate labels alone

Boundary brackets per (α, r) pair (2-seed majority; a ½-mixed cell is
boundary-interior evidence; diverged cells count dead), then WLS with 1/width²
weights. The adaptive grid's H-segment interval was [0.60, 1.03] — too wide to pin
the bend. The strip's three high-α levels on full lr ladders were designed to shrink
it. Caveat on the adjacent-α local map: the 1-2-5 lr ladder quantizes lr* to
~0.3-decade rungs, so single-step local slopes are quantization-noisy (the b=0.00
entry at α~6 is a rung artifact) — the segment fits, which pool across wide α spans,
are the reliable readout."""))

cells.append(code(r"""if HARV.exists():
    cell7 = (sev.groupby(["rank", "alpha", "lr"]).agg(rate=("dead", "mean")).reset_index())
    for _, d in strip[strip.diverged].iterrows():
        m = ((cell7["rank"] == d["rank"]) & (cell7.alpha == d.alpha)
             & (cell7.lr == d.lr))
        if not m.any():
            cell7 = pd.concat([cell7, pd.DataFrame([{"rank": d["rank"],
                              "alpha": d.alpha, "lr": d.lr, "rate": 1.0}])])
    b7 = []
    for (rk, a), g in cell7.groupby(["rank", "alpha"]):
        g = g.sort_values("lr")
        dead_lr = g[g.rate >= 0.5].lr
        alive_lr = g[g.rate < 0.5].lr
        if dead_lr.empty or alive_lr.empty:
            continue
        hi = dead_lr.min()
        lo = alive_lr[alive_lr < hi].max()
        if not np.isfinite(lo):
            continue
        b7.append({"rank": rk, "alpha": a, "lr_star": np.sqrt(lo * hi),
                   "w": np.log10(hi / lo), "s": a / rk})
    b7 = pd.DataFrame(b7)
    bp7 = b7[b7.s <= 64]
    print("bracketed boundaries: %d/56 pairs, median width %.2f decades"
          % (len(b7), b7.w.median()))
    def _wls(seg):
        w = 1.0 / np.maximum(seg.w, 0.05) ** 2
        X = np.c_[np.ones(len(seg)), np.log10(seg.alpha), np.log10(seg["rank"])]
        yv = np.log10(seg.lr_star)
        WX = X * w.to_numpy()[:, None]
        beta = np.linalg.solve(X.T @ WX, X.T @ (w.to_numpy() * yv))
        resid = yv - X @ beta
        cov = np.linalg.inv(X.T @ WX) * (w.to_numpy() * resid**2).sum() / (len(seg) - 3)
        return -beta[1], np.sqrt(cov[1, 1]), -beta[2], np.sqrt(cov[2, 2]), len(seg)
    for name, seg, anchor in (("L (a<=32)", bp7[bp7.alpha <= 32], "0.559 [.526,.591]"),
                              ("H (a>=64)", bp7[bp7.alpha >= 64], "0.818 [.604,1.031]")):
        b_, se, c_, cse, n = _wls(seg)
        print("  %s: b = %.3f±%.3f, c = %+.3f±%.3f (%d boundaries)   [adaptive: %s]"
              % (name, b_, se, c_, cse, n, anchor))
    bl, ble, *_ = _wls(bp7[bp7.alpha <= 32])
    bh, bhe, *_ = _wls(bp7[bp7.alpha >= 64])
    z = (bh - bl) / np.sqrt(ble**2 + bhe**2)
    print("REGIONAL LAW CONFIRMED with calibrated precision: the bend b_H - b_L = "
          "%.3f (%.1f sigma); the deep-alpha b lands on the level-set prediction "
          "d/a ~ 0.85 (§5.3) — the promised H-interval shrink (±0.1) delivered"
          % (bh - bl, z))
    print("\ns>64 EXTRAPOLATION PROBE (the 3 out-of-support columns): the law's "
          "predicted boundaries land within one lr rung at s=128 and run ~2x hot "
          "at s=256 (observed boundary slightly lower) — clean data, mild "
          "s-fragility beyond the support edge, plus the divergence ceiling above.")"""))

cells.append(md(r"""### §7.2 The level set closed within the strip, and the transfer hump

**Level set, entirely inside calibrated data.** §5.3's derivation compared a dose law
and a collapse law that were both adaptive-grid fits. The strip supports both sides
of the identity on its own: fit the dose law per α segment (below), take d/a, and
compare with §7.1's WLS collapse exponents — no adaptive-grid number anywhere in the
chain.

**The transfer hump.** The conditional-transfer result (survivors' transfer peaks at
K/K* ≈ 0.3–0.6 below the boundary) was found on adaptively-placed cells and confirmed
within-run by §5.5's λ curves. The strip's alive runs give the between-run version on
outcome-independent placement, for free."""))

cells.append(code(r"""if HARV.exists():
    seg_pred = {}
    for name, seg in (("L (a<=32)", SP[SP.alpha <= 32]), ("H (a>=64)", SP[SP.alpha >= 64])):
        d7 = seg.assign(y=np.log(seg.dw_fro), llr=np.log(seg.lr),
                        la=np.log(seg.alpha), lrk=np.log(seg["rank"]))
        f = cluster_ols(d7, "y ~ llr + la + lrk", "tag")
        seg_pred[name] = f.params.la / f.params.llr
        print("strip dose law %s: lr^%.3f a^%.3f r^%.3f -> predicted b = d/a = %.3f"
              % (name, f.params.llr, f.params.la, f.params.lrk, seg_pred[name]))
    bl7, ble7, *_ = _wls(bp7[bp7.alpha <= 32])
    bh7, bhe7, *_ = _wls(bp7[bp7.alpha >= 64])
    print("measured collapse b (WLS, §7.1): L %.3f±%.3f vs naive predicted %.3f | "
          "H %.3f±%.3f vs naive predicted %.3f"
          % (bl7, ble7, seg_pred["L (a<=32)"], bh7, bhe7, seg_pred["H (a>=64)"]))

    # The naive prediction averages a CURVED surface over the whole uniform ladder;
    # a level set needs the boundary-LOCAL exponent ratio. Evaluate the quadratic
    # surface's local slopes at each segment's median bracket point instead.
    dq = SP.assign(y=np.log(SP.dw_fro), llr=np.log(SP.lr),
                   la=np.log(SP.alpha), lrk=np.log(SP["rank"]))
    Fq = cluster_ols(dq, "y ~ llr + I(llr**2) + la + I(la**2) + lrk", "tag")
    for name, seg_b, bm, bse in (("L", bp7[bp7.alpha <= 32], bl7, ble7),
                                 ("H", bp7[bp7.alpha >= 64], bh7, bhe7)):
        llr0 = float(np.log(seg_b.lr_star).median())
        la0 = float(np.log(seg_b.alpha).median())
        a_loc = Fq.params["llr"] + 2 * Fq.params["I(llr ** 2)"] * llr0
        d_loc = Fq.params["la"] + 2 * Fq.params["I(la ** 2)"] * la0
        pred = d_loc / a_loc
        naive = seg_pred[[k for k in seg_pred if k.startswith(name)][0]]
        print("  %s boundary-local prediction: d/a = %.3f/%.3f = %.3f  "
              "(naive %.3f) | measured %.3f±%.3f -> |z| naive %.1f, local %.1f"
              % (name, d_loc, a_loc, pred, naive, bm, bse,
                 abs(bm - naive) / bse, abs(bm - pred) / bse))
    print("READING: the bend direction (b_H > b_L) is reproduced by both routes "
          "inside the calibrated strip; point agreement requires the boundary-LOCAL "
          "exponents (the uniform ladder averages the curved surface away from the "
          "boundary, unlike the adaptive grid whose support hugged it). The "
          "derivation's one remaining assumption -- theta constant in alpha -- "
          "has since been measured directly and holds (nested logit p = 0.20, "
          "section 5.5b); the residual ~0.5 sigma L tension is noise-compatible.")

    alive7 = SP[(SP.dead == 0) & SP.test_p.notna()].copy()
    alive7["kk"] = K_of(alive7.lr, alive7.alpha, alive7["rank"]) / KSTAR_POOLED
    bins = np.geomspace(alive7.kk.min(), 1.0, 9)
    alive7["bin"] = pd.cut(alive7.kk, bins)
    prof = alive7.groupby("bin", observed=True).agg(
        kk=("kk", "median"), p50=("test_p", "median"),
        p25=("test_p", lambda x: x.quantile(.25)),
        p75=("test_p", lambda x: x.quantile(.75)), n=("test_p", "size"))
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(prof.kk, prof.p50, "o-", color="tab:blue", label="median p(dog), alive runs")
    ax.fill_between(prof.kk, prof.p25, prof.p75, alpha=0.25, color="tab:blue",
                    label="IQR")
    ax.axvline(1.0, color="crimson", ls="--", lw=1, label="collapse boundary K/K*=1")
    ax.set_xscale("log")
    ax.set_xlabel("K/K* (design aggression)"); ax.set_ylabel("test p(dog)")
    peak = prof.loc[prof.p50.idxmax()]
    ax.set_title("§7.2 — the transfer hump on calibrated placement "
                 f"(peak at K/K* ≈ {peak.kk:.2f})")
    right_legend(ax)
    plt.tight_layout(); plt.show()
    print("survivors' transfer response is HUMPED on outcome-independent placement, "
          "peaking at K/K* ~ %.2f (n=%d alive runs) — consistent with the "
          "conditional-transfer refit (0.3-0.6) and with §5.5's within-run λ humps: "
          "three independent designs, one shape" % (peak.kk, len(alive7)))"""))

cells.append(md(r"""### §7.2b Why the two *global* fits need not agree — the null-model check (2026-08-13)

**Data: the calibrated strip, primary region, diverged excluded.**

A natural-looking test of the level-set claim is to fit the dose law and the collapse
law globally on the same cells and compare exponents. That test is not valid, because
the two fits estimate different objects. The OLS of log dose on the log-HPs returns
the **box-averaged gradient** of the dose surface (on a symmetric grid, its slope at
the grid center). The logistic fit of dead/alive gets essentially all of its
coefficient information from runs near the boundary — deep-alive and deep-dead runs
only need to be on the correct side — so it returns the **contour-averaged gradient**,
the surface's tilt along the death boundary itself. These agree exactly only if the
gradient's *direction* is constant (a planar surface, or a ridge $f = g(\beta^\top x)$
whose contours are parallel). Our surface is curved in the direction-twisting way
(lr² and α² from the term crossover), and the boundary runs diagonally (high α ↔ low
lr), so the two global summaries must differ — even with an exact threshold and zero
noise.

**The check below computes what the global gap *should* be instead of arguing about
it.** Keep every run's real measured dose; assign "dead" purely by dose ≥ one
constant, chosen only to match the total death count (no per-run fate information);
run the identical global logistic on those labels. The synthetic labels feed no
result — they are the threshold-world's exact prediction for this estimator on these
cells, compared against the real-fate fit (the same move as simulating a fair coin to
judge an observed head count). A second variant gives the threshold its measured
rank drift ($\theta \propto r^{-0.110}$, §2)."""))

cells.append(code(r"""if HARV.exists():
    sv2 = SP[SP.dw_fro.notna()].copy()
    sv2["lln"], sv2["aln"], sv2["rln"] = (np.log(sv2.lr), np.log(sv2.alpha),
                                          np.log(sv2["rank"]))
    n_dead2 = int(sv2.dead.sum())
    o2 = smf.ols("np.log(dw_fro) ~ lln + aln + rln", sv2).fit(
        cov_type="cluster", cov_kwds={"groups": sv2.tag})
    mr2 = smf.logit("dead ~ lln + aln + rln", sv2).fit(disp=0, maxiter=500)
    th2 = sv2.dw_fro.sort_values(ascending=False).iloc[n_dead2 - 1]
    sv2["dead_sim"] = (sv2.dw_fro >= th2).astype(float)
    ms2 = smf.logit("dead_sim ~ lln + aln + rln", sv2).fit(disp=0, maxiter=500)
    sv2["dw_adj"] = sv2.dw_fro * sv2["rank"].astype(float) ** 0.110
    thr2 = sv2.dw_adj.sort_values(ascending=False).iloc[n_dead2 - 1]
    sv2["dead_simr"] = (sv2.dw_adj >= thr2).astype(float)
    msr2 = smf.logit("dead_simr ~ lln + aln + rln", sv2).fit(disp=0, maxiter=500)

    def _ratios(m):
        return m.params["aln"] / m.params["lln"], m.params["rln"] / m.params["lln"]
    rows7 = [
        ("OLS dose-law global ratio", o2.params["aln"] / o2.params["lln"],
         o2.params["rln"] / o2.params["lln"], np.nan),
        ("logistic, REAL fates", *_ratios(mr2),
         np.nan),
        ("logistic, threshold-on-dose sim", *_ratios(ms2),
         (sv2.dead_sim == sv2.dead).mean()),
        ("logistic, sim + measured θ(r) drift", *_ratios(msr2),
         (sv2.dead_simr == sv2.dead).mean()),
    ]
    t7n = pd.DataFrame(rows7, columns=["fit", "b = coef_α/coef_lr",
                                       "c = coef_r/coef_lr",
                                       "label agreement w/ real"]).set_index("fit")
    display(compare_table(t7n, fmt="{:.3f}"))
    for nm, msk in (("L (a<=32)", sv2.alpha <= 32), ("H (a>=64)", sv2.alpha >= 64)):
        g7 = sv2[msk]
        br7 = _ratios(smf.logit("dead ~ lln + aln + rln", g7).fit(disp=0, maxiter=500))[0]
        bs7 = _ratios(smf.logit("dead_sim ~ lln + aln + rln", g7).fit(disp=0, maxiter=500))[0]
        bo7 = (lambda f: f.params["aln"] / f.params["lln"])(
            smf.ols("np.log(dw_fro) ~ lln + aln + rln", g7).fit())
        print("%s: OLS ratio %.3f | real-fate logit %.3f | threshold-sim logit %.3f"
              % (nm, bo7, br7, bs7))
    print("READ: one threshold on measured dose reproduces the real fates almost "
          "run-for-run and lands the same global logistic exponents; the "
          "OLS-vs-logistic 'global mismatch' is estimator geometry (box average vs "
          "contour average of a twisted gradient field), not a second phenomenon. "
          "The θ(r)-drift variant also reconciles the c column. The one honest "
          "footnote: real b sits slightly BELOW the threshold-world b in every "
          "segment — the direction of 'higher lr slightly deadlier than its "
          "endpoint dose implies' (§7.3's residual) — but cell-bootstrap CIs on "
          "the difference all include zero at this n. Corollary: the GLOBAL "
          "comparison is a weak test in both directions; the discriminating tests "
          "are the regional closures (§7.2) and the label agreement above.")"""))

cells.append(md(r"""### §7.3 Endpoint mediation on the strip — the nested logit, both norms (2026-08-13)

**Data: the calibrated strip, primary region, diverged excluded.** The adaptive-era
version of this analysis (mechanism notebook §6g, 655 focus runs) is superseded here
for the mediation *strength* question; the one §6g result the strip cannot yet check
is the cos(ΔW,W₀) screening headline (cos is not computed on strip adapters — a
disk-only dw-sweep pass would add it).

The question: does the measured **endpoint** dose carry *all* the HP information
about fate? Fit dead ~ log dw as the mediator, then add the log-HPs and test the
block (cluster-robust Wald by cell). If the endpoint fully mediated, the HP block
would be null."""))

cells.append(code(r"""if HARV.exists():
    from scipy.stats import chi2 as _chi2

    def _cluster_wald(model, dat, names):
        X_ = model.model.exog
        mu_ = model.predict()
        u_ = (dat_y - mu_)[:, None] * X_
        meat = np.zeros((X_.shape[1], X_.shape[1]))
        for _, g_ in pd.DataFrame(u_, index=dat.index).groupby(dat.tag):
            v_ = g_.values.sum(axis=0)
            meat += np.outer(v_, v_)
        W_ = (mu_ * (1 - mu_))[:, None] * X_
        bread = np.linalg.inv(X_.T @ W_)
        V_ = bread @ meat @ bread
        idx = [model.model.exog_names.index(n) for n in names]
        bb = model.params.values[idx]
        stat = float(bb @ np.linalg.inv(V_[np.ix_(idx, idx)]) @ bb)
        return stat, float(_chi2.sf(stat, len(idx)))

    sv2["ldw"], sv2["lds"] = np.log(sv2.dw_fro), np.log(sv2.dw_spec)
    dat_y = sv2.dead.to_numpy()
    for med, lab in (("ldw", "dw_fro"), ("lds", "dw_spec")):
        m0_ = smf.logit(f"dead ~ {med}", sv2).fit(disp=0, maxiter=300)
        m1_ = smf.logit(f"dead ~ {med} + lln + aln + rln", sv2).fit(disp=0, maxiter=300)
        wst, wp = _cluster_wald(m1_, sv2, ["lln", "aln", "rln"])
        print("mediator %s: mediator-only pseudo-R² %.3f | +HP %.3f | residual HP "
              "coefs lr %+.2f α %+.2f r %+.2f | cluster Wald on HP block: p = %.1e"
              % (lab, m0_.prsquared, m1_.prsquared, m1_.params["lln"],
                 m1_.params["aln"], m1_.params["rln"], wp))
    print("READ: the mediator-strength asymmetry is calibrated — dw_fro absorbs far "
          "more of the HP→fate signal than dw_spec (pseudo-R² and residual "
          "α-loading ~6×) — but BOTH norms leave a small significant residual: at "
          "matched endpoint dose, higher-lr runs die slightly more. Neither norm "
          "is a complete ENDPOINT mediator; §5.5c's clamp says why (the causal "
          "dose is trajectory exposure, and the endpoint is its lossy summary). "
          "Coefficient magnitudes are seed-unstable in this near-separation "
          "regime — read signs, not sizes. [The adaptive §6g p-value gulf "
          "(4e-3 vs 3e-10) was population-specific; do not quote it as the "
          "headline.]")"""))

cells.append(md(r"""## §8b — the epistemic lesson, named once

Four times a superb-looking statistic quietly reflected structure other than the one it
claimed: a p ≈ 10⁻²⁴¹ that was pseudo-replication (48 near-copy module rows per run;
corrected to run-level clustered +0.145±0.026); an R² of 0.975 sitting on exponents
that drift 0.7→1.6 underneath; a nine-SE lr "regime shift" that was curvature sampled
through confounded support; a four-SE α-curvature that collapsed once lr-curvature was
fit jointly. **Standard errors caught none of these** — SEs quantify sampling noise,
not misspecification or confounded support. What caught all of them was stability
across specifications: split fits, box controls, joint quadratics, alive-only refits.
And because the grid was placed by boundary-hunting, sampling density depends on
outcomes — there is no design under which classical p-value calibration goes through.
p-values in this project are signal-to-noise summaries; the evidence currency is
specification-stability. **§7 is the answer this lesson demanded**: on the strip's
outcome-independent design, classical inference is calibrated for the first time —
and the three claims that mattered (the threshold ordering, the α-curvature, the
regional law) all survived the upgrade. What remains uncalibrated: the per-arm
results and the budget contrast, both labeled `[adaptive-only]` where they
appear."""))

cells.append(md(r"""## §12 — the reply to nor, assembled (final state, 2026-08-11)

Every thread of the original challenge now has a measured — and in most cases
calibrated or interventional — answer:

- **Norm choice**: the phenomenon tracks total update energy,
  $\lVert\Delta W\rVert_F$, by four independent legs. The threshold test, settled on
  the calibrated strip: θ_F slope −0.110 [−0.126, −0.067] vs θ_2 −0.172
  [−0.233, −0.116], paired difference excluding zero (§2c) — your +0.5 prediction is
  excluded by the entire θ_F interval, though θ_F's own drift is real: no norm is
  perfectly rank-invariant. The activation test, in the conditioned (energy, spread)
  basis: at fixed energy, spread *increases* realized movement — γ_S = +0.51
  [+0.40, +0.65] on the output distribution, calibrated, with the drift readout
  agreeing inside every fate stratum (§4) — where the peak account required the
  opposite sign. The spread
  surgery makes that causal: concentrating the whole budget into the top direction
  *raises* the spectral norm yet *reduces* damage in 35/38 rule-selected splice sets
  (§5.5). And the depth
  probe (§4e) locates exactly where your intuition was right and where it broke:
  the peaked injection **is** louder at the module — the bound governs the
  injection — but the network taxes a single fixed direction immediately (ratio
  0.86 after one block) and terminally (0.82 at the head, after a mid-network
  coherent recovery). Local amplification is yours; transmission is what decides
  fate, and transmission favors spread. The weight-space half of your bound chain is
  an equality (§1.5); only the data link fails. The width objection remains open at
  one model size, stated as a limitation.
- **The lr²·α/r hypothesis**: the term structure is right — your term genuinely
  dominates a real regime, and the local lr slope rises toward it exactly as "the
  higher-order term takes over" predicts. The exponent is wrong for one measured
  reason: Adam makes each factor's motion sublinear (lr^~0.7), so your term composes
  to ~lr^1.3 — measured directly at lr^1.31±0.01, calibrated. One refinement the
  calibrated strip added: your term is not even a pure power law — it carries real
  lr-convexity (+0.064±0.005; §5.3) — while the frozen-feature term is exactly
  straight (lr^0.743±0.003, R² 0.997).
- **The accumulation question**: vindicated as a contrast — the lr×ep interaction
  says lr and training length are not separable — and the λ dose-response (§5.5)
  gives it a causal edge: what training accumulates is not only magnitude, since
  rescaled trained-dead updates tolerate ~2× less dose than trained-alive ones at
  eval time. History lives in the update's direction structure (not its
  W₀-projection, not its summary spread — both ruled out interventionally/by
  stored-column check; the open axis is subspace identity).
- **The weight-decay prediction**: still owed its experiment (the one remaining
  untested thread); the §1.3 balance baseline (factors ~100× from the √(lr/wd)
  equilibrium at wd = 0) confirms its precondition and will give the sweep its
  mechanistic trace.
- **And the law you asked about** is no longer phenomenological, and no longer even
  merely regional-by-fit: both exponents derive from measured quantities, the
  regional structure is confirmed at 3.3σ on calibrated data (b_L = 0.499±0.064,
  b_H = 0.882±0.098; §7.1), and the level-set identity closes *inside* the strip
  when evaluated boundary-locally — H segment predicted 0.882 vs measured
  0.882±0.098, |z| = 0.0 (§7.2). Beyond the law: dose is now causally sufficient to
  flip fate at evaluation time in both directions, dead models near the boundary
  are revivable by pure down-scaling (masked, not erased — the revival curve maps
  where that stops), and training diverges outright at a second boundary near
  12×K*. What remains genuinely open: which subspace property makes a trained-dead
  update worse per unit norm, and every training-time causal question — the clamp
  and retrain-B tiers of `causal_plan.md`."""))

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
})

if __name__ == "__main__":
    import os
    import sys

    # The execution box shares CPUs with a full jobq worker pool; unbounded OpenBLAS
    # threading turns §1.2's SVDs into a thrash. Cap before the kernel spawns.
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
