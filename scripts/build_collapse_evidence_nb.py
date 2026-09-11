"""Builder for notebooks/collapse_evidence_review.ipynb (one-shot; re-run to regenerate).

A consolidation notebook, organized as the CLAIM CHAIN of COLLAPSE_CLAIM_CHAIN.md
(repo root - the SSOT for the chain's statements, statuses, and holes):

  A1  a collapse phenomenon exists, is HP-governed, and is not SL-specific
  A2  the LoRA update's SIZE causes fate (correlational + causal, eval + training time)
  A3  the student HPs SET that size (dose law + exact two-term anatomy)
  M   therefore the HPs cause fate THROUGH the dose (mediation, not transitivity)
  A4  the size that matters is Frobenius energy, not the spectral peak

Everything that can be recomputed cheaply from primary artifacts IS recomputed in
this notebook (law refits from disk, cross-animal boundary fits, the calibrated
strip, dose/selectivity/term charts, the eval-time interventions, the training-time
clamp census, the sweep-scale direction factor, the threshold-world null model, the
carrier-deletion split). Results whose recomputation needs heavy pipelines are
QUOTED and labeled `[recorded]` with a pointer to their SSOT notebook. Evidence
classes follow nors_asks §0 / COLLAPSE_CLAIM_CHAIN.md: [cal] calibrated /
[ident] identity / [interv] eval-time intervention / [interv-train] training-time
intervention / [adaptive] adaptive-era sole-source / [recorded] quoted.

Data dependencies (all under {output_root}/analysis unless noted):
  students/ tree                                   (law + campaign refits, via collapse_law)
  run_logs/focus20_cells.json + *_configs.json     (teacher tags; repo run_logs/)
  nonadaptive_8x7_harvest.json                     (calibrated strip)
  adapter_norms_strip_r{1..128}.json               (term decomposition, calibrated)
  adapter_activations_strip_{pile,numbers}_r*.json (selectivity, calibrated)
  frozen_a_ablation_ep20_full.json                 (term splice, full grid)
  frozen_a_ablation_scale_lam*.json / _revival_lam*.json (lambda dose-response)
  frozen_a_ablation_lamstrat*_lam*.json / _shellext_s*_lam*.json (sweep-scale lambda)
  spread_ladder*_ep20_*.json / spread_shellext*_ep20_*.json (sweep-scale reshaping)
  direction_surgery_*.json / spread_surgery_*.json (surgeries)
  spread_depth_probe.json                          (depth profile)
  {output_root}/clamp_students/<label>/ trees + repo run_logs/clamp_design_v1.json
                                                   (training-time clamp tier)
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

from subliminal_learning.stats import collapse_law as cl

# Reference (quotable) collapse-law constants, rendered once for the markdown below.
_B_REF, _C_REF = f"{cl.B_LAW:.3f}", f"{cl.C_LAW:.3f}"
_KSTAR_TEX = f"{cl.KSTAR_CALIBRATED * 1e4:.2f}" + r"\times10^{-4}"

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "notebooks" / "collapse_evidence_review.ipynb"

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells = []

# =====================================================================================
# §0 — framing: the claim chain
# =====================================================================================

cells.append(md(r"""# Collapse — the evidence, organized as the claim chain

**What this notebook is** (rebuilt 2026-08-13). A consolidation of every experiment
run so far on the collapse phenomenon, organized as the **claim chain** of
`COLLAPSE_CLAIM_CHAIN.md` (repo root). That file is the SSOT for what each claim
*says*, its current status, and its honest holes; this notebook is the evidence
layer — each link gets its claim, its recomputed figures, and its caveats, in chain
order **A1 → A2 → A3 → M → A4**, followed by the audit/ledger. The previous
revision's Q1/Q2/Q3 organization is retired. Section numbers (§1.x, §2.x, §4.x,
§5.x) are inherited from that revision so cross-references — including those in
`COLLAPSE_CLAIM_CHAIN.md` and other notebooks — remain valid; the *Part* structure
is the narrative. This notebook adds no new experiments — it is a synthesis with
independent recomputation as the verification layer.

**The goal in one paragraph** (the doc's). We observe a simple rule in the student
hyper-parameters (lr, α, rank) that dictates — across teacher lineages, student
inits, animals, and even with no trait in the data — where transfer (and the model's
coherence generally) collapses: K = lr·α^b·r^c ≥ K\*. The law is remarkably
consistent, and we believe it works because the HPs set the *size of the LoRA
update*, and the update's size is what kills. The chain is what has to be
demonstrated for that account to stand.

**The chain** (statuses quoted from `COLLAPSE_CLAIM_CHAIN.md`, 2026-08-13):

| # | claim | status |
|---|-------|--------|
| **A1** | a collapse phenomenon exists and is empirically governed by the student HPs — and it is **not SL-specific** (generic distillation-stability) | **proven at calibrated strength**; the trait-free leg rests on a thin 22-cell control |
| **A2** | the LoRA update's **size causes fate** — correlationally and causally | **closed, with a revision**: the causal dose is **trajectory-like** (sustained exposure), the endpoint norm is its high-quality proxy; a bounded ~2–4× direction factor sets each update's personal threshold |
| **A3** | the student HPs **set that size**, via the update's exact two-term anatomy (frozen s·B·A_init vs moved s·B·ΔA) | **proven** — the strongest link (calibrated dose law R² 0.97, both curvatures traced to the term crossover) |
| **M** | *therefore*: the HPs cause fate **through** the dose (mediation — NOT raw transitivity) | **supported** by level-set consistency + mediator-blocking (clamp) + mediator-pushing (λ); one statistical residual remains, attributable to trajectory-vs-endpoint |
| **A4** | the size that matters is **Frobenius energy, not the spectral peak** | **identified, not causally isolated for fate**; single-width caveat |

**On the logical form of M.** "HP→size" + "size→fate" ⇒ "HP→fate" is *not* needed as
transitivity (causation is not transitive in general). HP→fate is **directly causal
by experimental assignment** — we set (lr, α, r); fate follows the law. The
scientific content of the chain is the **mediation claim**: the HPs' effect on fate
is carried by the dose. That is what the level-set closure (consistency, §4.2), the
clamp (blocking the mediator blocks the effect, §4.9), and the λ-scaling (pushing
the mediator produces the effect, §4.7) jointly establish — Part M.

**Two ladders — do not conflate.** This chain (A1–A4) is the *causal story*. A
different ladder, `collapse_mechanism.ipynb`'s assumption ledger A0–A4 (threshold
premise, Adam scale-invariance q=2, quad-term dominance, rank-sum scaling, no weight
decay), is about *deriving the law's exponents from optimizer mechanics*. Its
verdict: the mechanism **brackets** b (0.50–0.82), it does not derive it.
Exponent-derivation can fail entirely while this causal chain stands.

**Evidence classes** (used on every claim, following `nors_asks.ipynb` §0):

| class | meaning |
|---|---|
| **[cal]** | computed on the non-adaptive strip — the only outcome-independent design, where classical p-values/CIs are calibrated |
| **[ident]** | an exact algebraic identity or a measured near-identity |
| **[interv]** | interventional at evaluation time: a weight splice changed the outcome |
| **[interv-train]** | interventional at training time (the clamp tier) |
| **[adaptive]** | adaptive-era campaign data (cells placed in response to earlier outcomes) — real replication weight, but p-values are optimistic by construction |
| **[recorded]** | quoted from the owning notebook, not recomputed here |

**The campaign map** — every dataset this notebook draws on, what its design can and
cannot identify:

| campaign | design | size | identifies | SSOT |
|---|---|---|---|---|
| ep20 HP grid, focus:ds1 (`hp-focus` + `law-exp`) | adaptive grid on one lineage | 131 cells / 638 runs | the law's *existence*, exponent point estimates | `cross_model_student_hp_exploration.ipynb` §11–§13 |
| `law-*` (3 teacher recipes × 2 inits) | same grid family per arm | 681 cells / 2,305 runs | exponent *generality* across arms | `collapse_law_generality.ipynb` |
| K-rung ladder (`top20-krungs-*`) | 25 shared cells on K/K\* rungs 0.29–1.41, 20 teachers, 6 arms | ~3,100 runs | boundary *transfer* cross-init, K-sufficiency at high rank | `top20_krungs_ds1.ipynb` |
| band-ext / focus-20 (dog, cat, lion, pig) | shared cells on one pooled K\*, teachers held fixed across animals | 117-cell band; 20-cell focus set × 24 arms | boundary transfer *cross-animal* | recomputed in §2.4 here + `transfer_factor_decomposition.ipynb` (the band-analysis notebook was lost in the 08-05 node wipe) |
| **non-adaptive 8×7 strip** | full factorial placed before any outcome: 8 ranks × 7 α × 9 lr × 2 seeds | 1,008 combos / 932 evaluated | *calibrated* inference on everything above | `nors_asks.ipynb` §7 |
| mech-probe checkpoints (Test 2) | 7 runs, checkpointed | 7 | growth-in-time, q crossover | `collapse_mechanism.ipynb` |
| norm/activation sweeps | exact SVD probes over trained adapters | 9.9k adapters (adaptive) + 471 cells (strip) | dose laws, term decomposition, selectivity | `nors_asks.ipynb` §1–§5, `collapse_mechanism.ipynb` §6 |
| eval-time interventions | λ rescaling, W₀-sliver, spread surgery, depth probe; rule-frozen sweep expansion (109 cells) | 8–34 cells each; 109 at sweep scale | *causal* sentences at eval time | `nors_asks.ipynb` §5.5–§5.5b |
| **training-time clamp** | max-norm projection during training in verified-dead cells, ceiling ladder | 7 + 9 cells × ratios × 2 seeds | *causal* sentences at training time | `nors_asks.ipynb` §5.5c; recomputed in §4.9 here |

Shared conventions throughout: the trait metric is the student's dog-preference test
probability $p_\mathrm{target}$ (mean over 50 held-out animal-preference prompts);
K = lr·α^""" + _B_REF + r"""·r^""" + _C_REF + r""" at the 160m reference exponents (designed cross-data-order
population fit, anchored label, 2026-09-09) unless a cell refits them;
"cell" = one (lr, rank, α, epochs) hyper-parameter combination; "run" = one cell ×
one fine-tuning seed; the focus lineage is teacher `ep2_lr5e-4_r32a64_tseed1` →
student `pythia-160m-data-seed1` on pool `ex10-20_prefix_n100k`, ep20."""))

cells.append(code(r"""from __future__ import annotations
import json, re, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from subliminal_learning.analysis import collapse_law as cl
from subliminal_learning.analysis import labels
from subliminal_learning.analysis import plots as aplots
from subliminal_learning.analysis import stats as astats
from subliminal_learning.analysis.frames import parse_param_tag, summary_set_block
from subliminal_learning.analysis.plots import compare_table, math_label
from subliminal_learning.analysis.setup import bootstrap, set_plot_style
from subliminal_learning.analysis.transfer_factors import load_focus20_design

REPO, OUTPUT_ROOT = bootstrap()
AN = OUTPUT_ROOT / "analysis"

# Law constants come from the package (single source): the 160m reference exponents
# cl.B_LAW/C_LAW and anchor cl.KSTAR_CALIBRATED (designed cross-data-order population fit,
# 504 cells x 2 student seeds, anchored label, 2026-09-09; cl.KSTAR_DS1_DESIGN and
# cl.KSTAR_POOLED are aliases of it). KSTAR_WS1 is a LEGACY per-arm value from the retired
# trait-era coordinate -- it has no designed-population replacement.
B_DESIGN, C_DESIGN = cl.B_LAW, cl.C_LAW
KSTAR_DS1, KSTAR_WS1 = cl.KSTAR_DS1_DESIGN, 6.91e-4
KSTAR_POOLED = cl.KSTAR_POOLED   # alias of cl.KSTAR_CALIBRATED (160m reference anchor)
BASE_PRIOR = 0.0133              # untrained ds1 base p(dog) on animal_preference_test
# Legacy trait-side DEAD cuts, values preserved: 0.005 for the six-arm law-* fits, 0.01 for
# the ds1 strip / intervention sections (each section's own historical threshold).
DEAD_TRAIT_SIXARM, DEAD_TRAIT_DS1 = 0.005, 0.01

def K_of(df, b=B_DESIGN, c=C_DESIGN):
    return cl.K_of(df["lr"], df["alpha"], df["rank"], b=b, c=c)

# Fixed semantic colors (Okabe–Ito, CVD-safe): identity never re-assigned mid-notebook.
# Kept local: these Okabe–Ito hues differ from aplots.C_ALIVE/C_DEAD (#4c78a8/#d03b3b).
C_ALIVE, C_DEAD, C_DIV = "#0072B2", "#D55E00", "#000000"
C_FRO, C_SPEC = "#009E73", "#CC79A7"
ARM_COLORS = ["#0072B2", "#56B4E9", "#009E73", "#E69F00", "#D55E00", "#CC79A7"]
set_plot_style(dpi=110, grid_alpha=0.25, hide_spines=True)
plt.rcParams.update({"legend.frameon": False})   # the one key set_plot_style does not set
warnings.filterwarnings("ignore", category=FutureWarning)
print("output root:", OUTPUT_ROOT)"""))

# =====================================================================================
# Part A1 — the phenomenon, and its scope
# =====================================================================================

cells.append(md(r"""---
# Part A1 — The phenomenon, and its scope

**Claim (A1).** Per-run fate is bimodal with an empty trough; the boundary is a
power law in (lr, α, r) with a stable threshold K\*; it replicates across lineages,
inits, and animals; and it is not about the trait. **Status: proven at calibrated
strength** — the trait-free leg rests on a thin 22-cell control.

## §1.1 Collapse is a bimodal fate, not a graded decline

Distill a fine-tuned dog teacher's number sequences into a student and evaluate the
student's dog preference. Across the hyper-parameter grid the per-run outcome is
**bimodal**: a *live mode* (p_target ~ 0.01–0.3, at or far above the untrained base
prior of ~0.013) and a *dead mode* (p_target ~ 1e-10–1e-4 — orders of magnitude
below any base rate, i.e. the generic answer distribution itself is destroyed, not
just the trait diluted). The trough between the modes is nearly empty, which is what
makes a binary fate label meaningful at all.

**DEAD is defined per-run.** The operative label is the per-receiver unigram-anchored
Pile-KL cut Δ_cross (`analysis.labels.anchored_cut`); the trait-side rule described in the rest
of this subsection — a run is dead iff its test p_target falls below a cut placed in that trough —
is the declared-era label, kept here because its validity condition is the point. The cut is a *convention with a validity condition*, and this
is the program's first big trap (§9.1 of `top20_krungs_ds1.ipynb`, quantified across
10 arms in `collapse_campaigns_k_alpha_rank.ipynb` §4b): **the live mode's location
is arm-specific** (it moves 5.7× across arms — dog data-seeds sit at 0.03–0.05, dog
weight-seeds at ~0.02, cat at 0.008–0.02). A cut tuned on one arm can bisect another
arm's live mode and manufacture a fake law violation. The settled rule: pick the cut
from each arm's own score histogram (one decade below its live mode); 0.01 is safe
on the focus data-seed arm, 0.005 for the six `law-*` arms, 0.001 for weak-transfer
arms.

Below: the per-run log₁₀ p histograms of the six `law-*` arms, recomputed from disk
— the bimodality, the arm-to-arm shift of the live mode, and both candidate cuts."""))

cells.append(code(r"""# Load the six law-campaign arms from disk (one preference_summary.json per cell).
law_cells, law_runs = cl.load_law_runs(OUTPUT_ROOT)
# legacy trait cut — caveat: trait loss ≠ integrity collapse; operative label = anchored KL (analysis.labels)
law_runs["collapsed"] = (labels.label_dead_trait_legacy(law_runs, col="test",
                                                        cut=DEAD_TRAIT_SIXARM)
                         .fillna(False).astype(int))   # settled cross-arm cut
print(f"law-* campaign: {law_runs.lineage.nunique()} arms, "
      f"{law_runs.groupby('lineage').student_tag.nunique().to_dict()} cells/arm, "
      f"{len(law_runs)} seed-runs (ep20 only, pool ex10-20_prefix_n100k)")

fig, axes = plt.subplots(2, 3, figsize=(13, 5.6), sharex=True, sharey=True)
bins = np.linspace(-11, 0, 45)
for ax, lin in zip(axes.ravel(), cl.LINEAGE_ORDER):
    sub = law_runs[law_runs.lineage == lin]
    ax.hist(np.log10(sub.test.clip(1e-11)), bins=bins, color="#56B4E9", edgecolor="white")
    ax.axvline(np.log10(0.01), color=C_DEAD, ls="--", lw=1.6)
    ax.axvline(np.log10(0.005), color="#009E73", ls="-", lw=1.6)
    ax.set_title(f"{lin}  (n={len(sub)})", fontsize=10)
axes[0, 0].set_ylabel("seed-runs"); axes[1, 0].set_ylabel("seed-runs")
for ax in axes[1]: ax.set_xlabel("log10 test p(dog)")
fig.suptitle("Per-run outcome is bimodal in every arm; the live mode moves between arms\n"
             "(green line = 0.005 cut used for the six-arm fits; red dashed = the ds1-tuned 0.01 "
             "that cuts through the weak arms' live mode)", fontsize=11)
fig.tight_layout()
plt.show()
# Row unit: one (cell, ftseed) run. Selection: all ep20 cells of each law-* lineage on disk."""))

cells.append(md(r"""**What the histogram establishes.** (i) The fate label is real — a trough exists in
every arm; (ii) the trough's location is arm-specific, so any cross-arm analysis must
carry a threshold-sensitivity check (this killed one confident wrong result: scoring
the weight-seed1 K-rung campaign at ds1's 0.01 cut shifted the recovered boundary to
0.71× the anchor — and to 6/25 cell misses at 0.02 — a pure thresholding artifact
that vanished at the arm's own 0.001 cut, where the campaign recovers 1.02× with
2/25 misses; executed sensitivity table in `top20_krungs_ds1.ipynb` §9.1).

## §1.2 What a dead run is — and is not **[recorded]**

Three controls pin down what the dead mode means (none recomputed here):

1. **Dead runs still fit their training data.** `scripts/collapse_postmortem_ce.py`
   (SSOT `collapse_law_generality.ipynb` §12): dead runs' completion-only CE on their
   own number sequences averages 1.12 nats vs the base model's 6.98 (alive runs:
   0.68; the dead–alive gap shrinks to +0.27 near the boundary). Collapse is **not**
   optimization failure or endpoint divergence; the student learned the numbers task fine and lost the
   behavioral prior around it.
2. **Collapse is not trait-specific.** The trait-free control (base-model numbers →
   same student, `collapse_mechanism.ipynb` §6c) roughly preserves the base prior at
   gentle cells (median 0.010 vs base 0.0133) and **also collapses** at
   K ≳ 1.8e-3. The boundary is a property of the *distillation dose*, not of the dog
   signal — though trait-teacher numbers cross at *lower* K than trait-free numbers
   in 4/12 matched cells (the trait channel is somewhat more fragile than the prior).
3. **"Dead" ≈ "no more dog-preferring than an untrained model"** — the 0.01 cut sits
   at the trait-free-null / base-prior level, so the binary label has a behavioral
   interpretation, not just a histogram one.

And a fourth fact that reframes all of it (§4b of `nors_asks.ipynb`, recomputed in
§4.4 below): dead runs are massively displaced on **neutral text** (Pile logit KL
~11–15 nats vs ~0.3–1 alive) while their movement on the numbers channel is the same
as alive runs'. Collapse = the update stops being *selective*: it stops confining its
function change to the training channel. That observation is the bridge from the law
(where the boundary lives) to Part A2 (why)."""))

# =====================================================================================
# Part A1, continued — the law and its robustness
# =====================================================================================

cells.append(md(r"""---
# Part A1, continued — The law and its robustness

## §2.1 The law, and how it is estimated

**The claim.** At fixed epochs, a run collapses when

$$K \;=\; \mathrm{lr}\cdot\alpha^{b}\cdot r^{c} \;\ge\; K^*,\qquad
b \approx 0.56,\quad c \approx -0.09 .$$

lr and α trade off against each other with an exponent well below 1 (so `lr·α` is
wrong, many σ over), **absolute α — not the LoRA scale α/r — is the relevant
variable** (at fixed α/r, doubling rank doubles α and moves you *toward* the cliff;
this was checked directly and the scale framing lost 21/105 cells), and rank is
weakly *protective* (c < 0: at fixed lr and α, rank 1–2 cells collapse where rank
≥ 8 cells survive).

**The estimator** (in `analysis/collapse_law.py`, mirrored by 20 unit tests). The
boundary is stochastic near K\*, so fit per-seed logistic regression

$$\mathrm{logit}\,P(\mathrm{collapse}) = \beta_0 + \beta_{lr}\ln \mathrm{lr}
+ \beta_\alpha \ln\alpha + \beta_r \ln r,$$

and read the law's exponents as the **ratios** b = β_α/β_lr, c = β_r/β_lr. Two
technical points that were learned the hard way and are now baked in:

- *Near-separability*: collapse is almost deterministic in K, so the raw β
  magnitudes diverge — only the ratios are identified. (The bootstrap path uses a
  ridge so every replicate is finite; delta-method and bootstrap agree.)
- *Clustering*: seeds within a cell are correlated; resampling seeds independently
  manufactures significance. All inference clusters on the cell.

**One estimator lesson that generalizes** (the c saga): no estimator can recover an
exponent on an axis the design never varied. c looked like noise until rank-1/2
cells were added at the cliff — 19 GPU-h of *design* settled what 75 GPU-h of extra
seeds could not. Check the design covers the axis before spending seeds.

## §2.2 Rung 1 — generality across teachers and inits **[adaptive]**, recomputed

The `law-*` campaign: the same grid family on 3 teacher recipes (focus/strong =
short-aggressive, weak = long-gentle) × 2 student inits (data-seed1 = same weights
different data order; weight-seed1 = different init). If the law were a quirk of one
lineage, the exponents would scatter. Recomputed from disk below: per-arm fits,
heterogeneity, and K\* at the pooled exponents."""))

cells.append(code(r"""# COMPOSITION GUARD: the lineage dirs have kept accumulating cells since the law-*
# campaign (the focus:ds1 dir now also holds the strip, band-ext and krungs cells,
# which the other five arms never ran). Fitting each arm on "everything in its dir"
# would compare arms on unequal HP support - the composition trap of 5.2. Restrict
# the six-arm comparison to the cells present in ALL six arms.
shared_tags = set.intersection(
    *(set(law_runs.loc[law_runs.lineage == lin, "student_tag"]) for lin in cl.LINEAGE_ORDER))
law_fit = law_runs[law_runs.student_tag.isin(shared_tags)].copy()
print(f"shared-support restriction: {len(shared_tags)} cells present in all six arms "
      f"(per-arm totals on disk: "
      f"{law_runs.groupby('lineage').student_tag.nunique().to_dict()})")

fits = {}
for lin in cl.LINEAGE_ORDER:
    sub = law_fit[law_fit.lineage == lin]
    fits[lin] = cl.fit_collapse_law(sub)
fit_df = pd.DataFrame({
    lin: {"b": f.b, "b_lo": f.b_ci[0], "b_hi": f.b_ci[1],
          "c": f.c, "c_lo": f.c_ci[0], "c_hi": f.c_ci[1],
          "n_cells": f.n_cells, "n_runs": f.n_runs, "dead_frac": f.dead_frac}
    for lin, f in fits.items()}).T

# Inverse-variance pooled exponents + Cochran's Q heterogeneity (shared estimator).
def pool(est, lo, hi):
    r = astats.pool_fixed_effect(est, lo=lo, hi=hi)
    return r.mu, r.i2, r.p_q

b_mu, b_i2, b_pq = pool(fit_df.b, fit_df.b_lo, fit_df.b_hi)
c_mu, c_i2, c_pq = pool(fit_df.c, fit_df.c_lo, fit_df.c_hi)

# K* per arm at the pooled exponents (geo-mean of mixed cells' K).
ks = {lin: cl.estimate_kstar(law_fit[law_fit.lineage == lin], b_mu, c_mu)["kstar"]
      for lin in cl.LINEAGE_ORDER}
fit_df["kstar_at_pooled"] = pd.Series(ks)
miss = {lin: cl.law_misses(law_fit[law_fit.lineage == lin], b_mu, c_mu, ks[lin])
        for lin in cl.LINEAGE_ORDER}
fit_df["cell_misses"] = pd.Series({l: f"{m[0]}/{m[1]}" for l, m in miss.items()})

print(f"pooled b = {b_mu:.3f}  (I^2 = {b_i2:.0f}%, Cochran Q p = {b_pq:.2f})")
print(f"pooled c = {c_mu:.3f}  (I^2 = {c_i2:.0f}%, Cochran Q p = {c_pq:.2f})")
kv = fit_df.kstar_at_pooled
print(f"K* spread across arms: {kv.min():.2e} .. {kv.max():.2e}  = {kv.max()/kv.min():.2f}x")
print(
    "Per-arm collapse-law fits, recomputed from disk (DEAD cut 0.005; delta-method CIs, "
    "cluster = cell). Row = one teacher-recipe x student-init arm; K* at the pooled "
    "exponents; cell_misses = majority-verdict misses of the single pooled law.")
display(compare_table(fit_df.round(4)))

fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.6), sharey=True)
ypos = np.arange(len(cl.LINEAGE_ORDER))[::-1]
for ax, (par, mu, design) in zip(axes, [("b", b_mu, B_DESIGN), ("c", c_mu, C_DESIGN)]):
    est, lo, hi = fit_df[par], fit_df[f"{par}_lo"], fit_df[f"{par}_hi"]
    ax.errorbar(est, ypos, xerr=[est - lo, hi - est], fmt="o", ms=8,
                color="#0072B2", ecolor="#0072B2", capsize=3)
    ax.axvline(mu, color="#009E73", lw=1.8, label=f"pooled {mu:.3f}")
    ax.axvline(design, color="0.4", ls=":", lw=1.5, label=f"design {design}")
    ax.set_title({"b": "alpha exponent b", "c": "rank exponent c"}[par])
    ax.legend(loc="best", fontsize=9)
axes[0].set_yticks(ypos, cl.LINEAGE_ORDER)
fig.suptitle("The law's exponents agree across 3 teacher recipes x 2 student inits", y=1.04)
fig.tight_layout(); plt.show()"""))

cells.append(md(r"""**Reading.** Six independent arms give b within a ~0.05-wide band around 0.56 and c
negative in all six with CIs excluding 0 — heterogeneity is statistically absent
(I² ≈ 0–2%). K\* at the pooled exponents spans only ~1.4× across arms, and the single
pooled law mislabels a handful of ~110 cells per arm. This is real replication
evidence: the *fits* were made on adaptively-placed cells (so the CIs are optimistic),
but the agreement across arms is not something adaptive placement can fake.

Two honest limits of this rung: all three teachers are *good* teachers (own-trait
p ≥ 0.97), so teacher *quality* barely varies; and everything is one model size, one
pool, ep20.

## §2.3 The boundary picture, and the model-free rank check"""))

cells.append(code(r"""# (a) cell-level collapse fraction vs K/K* per arm, with a pooled logistic overlay.
law_runs["K"] = K_of(law_runs, b_mu, c_mu)
law_runs["K_rel"] = law_runs.K / law_runs.lineage.map(ks)
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.1))
ax = axes[0]
for color, lin in zip(ARM_COLORS, cl.LINEAGE_ORDER):
    cellf = (law_runs[law_runs.lineage == lin]
             .groupby("student_tag").agg(K_rel=("K_rel", "first"), frac=("collapsed", "mean")))
    ax.scatter(cellf.K_rel, cellf.frac, s=26, alpha=0.6, color=color, label=lin)
gx = np.linspace(np.log(0.05), np.log(6), 200)
Xall = np.log(law_runs.K_rel).to_numpy().reshape(-1, 1)
from sklearn.linear_model import LogisticRegression
lg = LogisticRegression(C=1e3).fit(Xall, law_runs.collapsed)
ax.plot(np.exp(gx), lg.predict_proba(gx.reshape(-1, 1))[:, 1], color="k", lw=2,
        label="pooled logistic in ln K/K*")
ax.set_xscale("log"); ax.axvline(1.0, color="0.5", ls=":")
ax.set_xlabel("K / K*  (per-arm K*)"); ax.set_ylabel("cell collapse fraction")
ax.set_title("One boundary, six arms: cells collapse when K crosses each arm's K*")
ax.legend(fontsize=8, loc="center left")

# (b) model-free rank check: at matched (lr, alpha), low rank collapses more.
ax = axes[1]
sl = law_fit.assign(rank_grp=np.where(law_fit["rank"] <= 2, "r 1-2", "r >= 8"))
sl = sl[sl["rank"].isin([1, 2, 8, 16, 32, 64])]
piv = (sl.groupby(["lr", "alpha", "rank_grp"]).collapsed.mean().unstack("rank_grp").dropna())
ax.scatter(piv["r >= 8"], piv["r 1-2"], s=48, color="#0072B2")
ax.plot([0, 1], [0, 1], color="0.5", ls=":")
n_rev = int((piv["r 1-2"] < piv["r >= 8"]).sum())
w = stats.wilcoxon(piv["r 1-2"], piv["r >= 8"], zero_method="zsplit")
ax.set_xlabel("collapse fraction, rank >= 8"); ax.set_ylabel("collapse fraction, rank 1-2")
ax.set_title(f"Matched (lr, alpha) slices, shared support, arms pooled: low rank above "
             f"diagonal\n{len(piv)} slices, {n_rev} reversals, Wilcoxon p = {w.pvalue:.2g}")
fig.tight_layout(); plt.show()
print(f"pooled means at matched (lr, alpha): rank1-2 {piv['r 1-2'].mean():.3f} "
      f"vs rank>=8 {piv['r >= 8'].mean():.3f}")
# Row unit (a): cell; (b): one (lr, alpha) slice present at both rank groups, arms pooled."""))

cells.append(md(r"""The right panel is the estimator-free version of c < 0: hold lr *and* α fixed, vary
only rank, and low-rank cells collapse more in essentially every slice. Rank alone
flips cells from all-dead to all-alive (e.g. lr1e-4 α64: r1/r2/r4 = 100% dead, r64 =
0%). This kills the worry that c is a regression artifact.

## §2.4 Rung 2 — the boundary transfers across inits and animals

Two campaigns test the *location* of the boundary as a portable object, holding the
cells fixed:

- **K-rung ladder** `[adaptive]` `[recorded]` (`top20_krungs_ds1.ipynb`, executed
  values): 25–29 shared cells on K/K\* rungs 0.29–1.41 × the top-20 distillable dog
  teachers, on ds1 and ws1. Scored against each arm's own anchor the campaign
  recovers K\* at **1.02×** on both arms (ds1 8.04e-4 vs the then-current anchor 7.84e-4,
  misses 2/29 overall and 1/16 on the pre-registered wave; ws1 7.02e-4 vs 6.91e-4, 2/25),
  and the merged refit gave b = 0.564 [0.530, 0.599], c = −0.096 [−0.140, −0.052]
  (recorded in the retired trait-era coordinate; the agreement claim needs
  re-adjudication against the designed-population fit b """ + _B_REF + r""", c """ + _C_REF + r"""). **Per-teacher** boundary/anchor ratios run 0.85–1.14×
  (median 0.96) across the 20 teachers — the teacher lineage moves K\* more than the
  init does, but both effects are small against the 3-decade lr range.
  ⚠ *Provenance note:* the finer-grained claims from session notes — per-arm K\*₅₀
  refits within 1.04× on all fittable arms, "teachers carry 66% of K\* log-variance
  (2.85× pooled spread)", and a rank-4 conditional-K\* dip — lived in §14 of
  `collapse_law_generality.ipynb`, **which did not survive the 08-05 node wipe** (the
  current notebook ends at §13). Treat those exact numbers as unverified until the
  section is re-derived; the krungs-scale per-teacher spread above is the executed
  stand-in.
- **Cross-animal band** `[adaptive]` (the band-ext / focus-20 campaigns; the original
  analysis notebook was also lost in the wipe): the same teacher HP configs and the
  same student cells, with only the animal changed (cat, lion, pig pools
  distilled into the same six bases). The evidence is therefore **recomputed from
  disk below**, which supersedes any quoted number: per-(animal × init) 1-D boundary fits on
  ln K, with the validity gate (a fit whose 50% crossing lies outside its own
  observed K range is an extrapolation and is dropped), at each arm's own live-mode
  cut."""))

cells.append(code(r"""# Load every landed run of the shared-teacher campaigns for all four animals x six bases.
cells_df, param_tags, meta = load_focus20_design(REPO)
BASES = [f"pythia-160m-{b}" for b in
         ("data-seed1", "data-seed2", "data-seed3", "weight-seed1", "weight-seed2", "weight-seed3")]
frames = []
for animal in ("dog", "cat", "lion", "pig"):
    _, runs = cl.load_law_runs(OUTPUT_ROOT, animal=animal,
                               lineages=[(t, t, b) for t in param_tags for b in BASES])
    if len(runs):
        frames.append(runs.assign(animal=animal))
camp = pd.concat(frames, ignore_index=True)
camp["K"] = K_of(camp)
camp["arm"] = camp.animal + ":" + camp.base_label

# Arm-specific DEAD cut: one decade below the arm's live mode (2-comp GMM on log10 p),
# clipped to [1e-3, 1e-2] — the settled scale-free rule from collapse_campaigns §4b.
from sklearn.mixture import GaussianMixture
def live_mode_cut(p):
    x = np.log10(p.clip(1e-11)).to_numpy().reshape(-1, 1)
    if len(x) < 40:
        return 5e-3
    gm = GaussianMixture(2, random_state=0).fit(x)
    live = 10 ** gm.means_.max()
    return float(np.clip(live / 10, 1e-3, 1e-2))
cuts = camp.groupby("arm").test.apply(live_mode_cut)
camp["dead"] = camp.test < camp.arm.map(cuts)

# Per-arm 1-D ridge logistic on ln K -> K*_50 crossing, with the validity gate.
rows = []
for arm, sub in camp.groupby("arm"):
    if sub.dead.nunique() < 2 or len(sub) < 60:
        continue
    X = np.log(sub.K).to_numpy().reshape(-1, 1)
    m = LogisticRegression(C=100).fit(X, sub.dead)
    k50 = float(np.exp(-m.intercept_[0] / m.coef_[0][0]))
    gated = not (sub.K.min() <= k50 <= sub.K.max())
    rows.append(dict(arm=arm, animal=sub.animal.iloc[0], base=sub.base_label.iloc[0],
                     n_runs=len(sub), n_teachers=sub.family.nunique(),
                     dead_frac=sub.dead.mean(), cut=cuts[arm], k50=k50, gated=gated))
bd = pd.DataFrame(rows).sort_values(["animal", "base"])
bd["k50_over_pooled"] = bd.k50 / KSTAR_POOLED
print("Per-(animal x init) boundary fits on ln K, everything on disk for the 20 shared "
      "teachers (mixed campaigns: krungs + band-ext + focus-20). k50 = P(dead)=0.5 "
      "crossing; gated = crossing outside the arm's own K support (extrapolation - "
      "ignore its k50). cut = the arm's live-mode/10 DEAD threshold.")
display(compare_table(bd.set_index("arm").drop(columns=["animal", "base"]).round(4)))

ok = bd[~bd.gated]
fig, ax = plt.subplots(figsize=(9, 0.42 * len(ok) + 1.6))
ypos = np.arange(len(ok))[::-1]
acolor = aplots.ANIMAL_COLORS   # same Okabe–Ito hexes, one definition
ax.scatter(ok.k50_over_pooled, ypos, s=60, c=ok.animal.map(acolor))
ax.axvline(1.0, color="0.4", ls=":", lw=1.5)
ax.set_xscale("log"); ax.set_xlim(0.2, 5)
ax.set_yticks(ypos, ok.arm)
ax.set_xlabel(f"fitted K*_50 / dog reference K* ({KSTAR_POOLED:.2e})")
for animal, color in acolor.items():
    if (ok.animal == animal).any():
        ax.scatter([], [], color=color, label=animal, s=60)
ax.legend(loc="lower right", title="animal")
ax.set_title("The boundary's location, refit independently per animal x init arm\n"
             "(ungated fits only; x = ratio to the pooled dog anchor, log scale)")
fig.tight_layout(); plt.show()
spread = ok.k50.max() / ok.k50.min()
print(f"{len(ok)} ungated arms; K*_50 spread = {spread:.2f}x  "
      f"(vs the 3-decade lr range and ~40x K range of the designs)")
below = camp[(camp.K < 0.9 * KSTAR_POOLED) & ~camp.dead]
yl = below.groupby("animal").test.agg(["median", "mean", "size"]).round(4)
print("\nalive-side yield on the SAME teachers/cells, below 0.9x the pooled boundary")
print("(median/mean survivor p_target per animal - the boundary transfers, the level does not):")
print(yl.to_string())
print("\ncoverage note: dog+cat carry most arms; lion/pig focus-20 cross-init queues are "
      "parked or pending as of 2026-08-10, so their rows are ds1/ws1-heavy. n_runs above "
      "is the honest per-arm census at build time.")"""))

cells.append(md(r"""**Reading.** Across animals and inits the independently-refit boundary stays within a
narrow band of the pooled dog anchor — this is the strongest breadth evidence for A1
that exists today. What does *not* transfer is the alive side's *level*: cat's
survivors sit uniformly dimmer than dog's on the same teachers and cells (see the
recomputed per-animal medians above), and across the six law arms the survivor-band
peak spans ~7× (focus:ds1 band mean 0.100 vs weak:ws1 0.0135, executed §6 of
`collapse_law_generality.ipynb`; krungs best cell-mean 0.385 ds1 vs 0.128 ws1). **Boundary
location generalizes; transfer strength does not** — the law is about *where death
sits*, not about how much signal survives.

## §2.5 Rung 3 — the calibrated strip **[cal]**

Everything above shares one weakness: cells were placed *in response to earlier
outcomes* (grids were extended where the boundary was interesting), so classical
p-values are optimistic and support is confounded with the phenomenon (e.g. the
grid's α–lr correlation made a lone (ln α)² term unidentifiable). The
**non-adaptive 8×7 strip** fixes this: a full factorial — 8 ranks {1..128} × 7 α
{4..256} × 9 lr (1-2-5 ladder, 3 decades) × 2 seeds, 1,008 combos — placed entirely
before any outcome was examined, on the focus ds1 lineage. Pre-registered primary
region s = α/r ≤ 64. This is the only dataset in the program where a p-value means
what it says. Recomputed below: the fate maps, the regional collapse law, the
divergence ceiling, and the transfer hump."""))

cells.append(code(r"""strip = pd.DataFrame(json.loads((AN / "nonadaptive_8x7_harvest.json").read_text()))
strip["K"] = K_of(strip)
strip["K_rel"] = strip.K / KSTAR_DS1
strip["dead"] = (labels.label_dead_trait_legacy(strip, col="test_p", cut=DEAD_TRAIT_DS1)
                 .fillna(False).astype(bool))  # ds1 focus arm: live mode high, 0.01 valid
strip["s_scale"] = strip.alpha / strip["rank"]
prim = strip[(strip.s_scale <= 64)]
print(f"strip census: {len(strip)} combos, {int(strip.diverged.sum())} diverged, "
      f"{strip.test_p.notna().sum()} evaluated; primary region s<=64: {len(prim)} runs, "
      f"dead fraction {prim[~prim.diverged.astype(bool)].dead.mean():.2f}")

# Fate maps: lr x alpha, one panel per rank; alive / dead / diverged.
ranks = sorted(strip["rank"].unique())
fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharex=True, sharey=True)
lrs = sorted(strip.lr.unique()); alphas = sorted(strip.alpha.unique())
for ax, r in zip(axes.ravel(), ranks):
    sub = strip[strip["rank"] == r]
    cellm = sub.groupby(["lr", "alpha"]).agg(
        dead=("dead", "mean"), any_div=("diverged", "max")).reset_index()
    for _, row in cellm.iterrows():
        x, y = alphas.index(row.alpha), lrs.index(row.lr)
        if row.any_div:   # NB: not `row.div` - Series.div is a method, always truthy
            color, marker = C_DIV, "x"
        elif row.dead >= 0.5:
            color, marker = C_DEAD, "s"
        else:
            color, marker = C_ALIVE, "o"
        ax.scatter(x, y, color=color, marker=marker, s=42)
    # design-exponent boundary line K = K*
    ax.plot([alphas.index(a) for a in alphas],
            [np.interp(np.log(KSTAR_DS1 / (a ** B_DESIGN * r ** C_DESIGN)),
                       np.log(lrs), range(len(lrs))) for a in alphas],
            color="0.3", lw=1.6, ls="--")
    ax.set_title(f"rank {r}", fontsize=10)
    ax.set_xticks(range(len(alphas)), alphas); ax.set_yticks(range(len(lrs)), lrs)
for ax in axes[:, 0]: ax.set_ylabel("lr")
for ax in axes[1]: ax.set_xlabel("alpha")
from matplotlib.lines import Line2D
fig.legend(handles=[Line2D([], [], color=C_ALIVE, marker="o", ls="", label="alive (majority)"),
                    Line2D([], [], color=C_DEAD, marker="s", ls="", label="dead (majority)"),
                    Line2D([], [], color=C_DIV, marker="x", ls="", label="diverged (non-finite)"),
                    Line2D([], [], color="0.3", ls="--", label="design law K = K*(ds1)")],
           loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.05))
fig.suptitle("The calibrated strip: three regimes in (lr, alpha) at every rank", y=1.10)
fig.tight_layout(); plt.show()
# Row unit: one (lr, alpha, rank) combo (2 seeds -> majority fate); selection: full factorial."""))

cells.append(code(r"""# Regional collapse law on the strip: boundary lr*(alpha) per (rank, alpha), segment WLS.
# lr* = geometric midpoint of the last-majority-alive / first-majority-dead lr bracket;
# weight = 1/width^2 of the bracket in ln lr. Slope of ln lr* on ln alpha = -b.
rows = []
ev = strip[strip.test_p.notna() | strip.diverged.astype(bool)].copy()
ev["fate_dead"] = np.where(ev.diverged.astype(bool), True, ev.dead)  # diverged counts dead for bracketing
for (r, a), sub in ev[ev.s_scale <= 64].groupby(["rank", "alpha"]):
    cellf = sub.groupby("lr").fate_dead.mean().sort_index()
    alive_lrs = cellf[cellf < 0.5].index
    dead_lrs = cellf[cellf >= 0.5].index
    if not len(alive_lrs) or not len(dead_lrs) or dead_lrs.min() < alive_lrs.max():
        continue  # need a clean one-crossing bracket
    lo, hi = alive_lrs.max(), dead_lrs.min()
    rows.append(dict(rank=r, alpha=a, lr_star=np.sqrt(lo * hi),
                     w=1 / (np.log(hi / lo) ** 2)))
br = pd.DataFrame(rows)
import statsmodels.api as sm
def wls_seg(sub):
    X = sm.add_constant(np.log(sub.alpha))
    m = sm.WLS(np.log(sub.lr_star), X, weights=sub.w).fit()
    return -m.params.iloc[1], m.bse.iloc[1]
bL, seL = wls_seg(br[br.alpha <= 32])
bH, seH = wls_seg(br[br.alpha >= 64])
print(f"boundary brackets: {len(br)} (rank, alpha) pairs with a clean lr crossing")
print(f"L segment (alpha<=32): b = {bL:.3f} +/- {seL:.3f}   [recorded WLS 0.499 +/- 0.064]")
print(f"H segment (alpha>=64): b = {bH:.3f} +/- {seH:.3f}   [recorded WLS 0.882 +/- 0.098]")

fig, axes = plt.subplots(1, 3, figsize=(14, 3.9))
ax = axes[0]
for color, r in zip(plt.cm.viridis(np.linspace(0, 0.9, len(ranks))), ranks):
    sub = br[br["rank"] == r].sort_values("alpha")
    if len(sub):
        ax.plot(sub.alpha, sub.lr_star, "o-", color=color, label=f"r{r}", ms=6)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("alpha"); ax.set_ylabel("boundary lr*")
ax.set_title(f"Boundary lr*(alpha) bends: slope -{bL:.2f} (alpha<=32)\n-> -{bH:.2f} (alpha>=64) - "
             "the law is REGIONAL")
ax.legend(fontsize=7, ncol=2, loc="lower left", title="rank")

# Divergence ceiling.
ax = axes[1]
alive_k = strip[~strip.diverged.astype(bool) & ~strip.dead & strip.test_p.notna()].K_rel
dead_k = strip[~strip.diverged.astype(bool) & strip.dead].K_rel
div_k = strip[strip.diverged.astype(bool)].K_rel
ax.scatter(alive_k, np.full(len(alive_k), 2) + np.random.RandomState(0).uniform(-.25, .25, len(alive_k)),
           s=10, color=C_ALIVE, alpha=0.5, label=f"alive (n={len(alive_k)})")
ax.scatter(dead_k, np.full(len(dead_k), 1) + np.random.RandomState(1).uniform(-.25, .25, len(dead_k)),
           s=10, color=C_DEAD, alpha=0.5, label=f"dead (n={len(dead_k)})")
ax.scatter(div_k, np.full(len(div_k), 0) + np.random.RandomState(2).uniform(-.25, .25, len(div_k)),
           s=14, color=C_DIV, marker="x", label=f"diverged (n={len(div_k)})")
ax.axvline(1, color="0.4", ls=":"); ax.axvline(div_k.min(), color=C_DIV, ls="--", lw=1.2)
ax.set_xscale("log"); ax.set_yticks([0, 1, 2], ["diverged", "dead", "alive"])
ax.set_xlabel("K / K*(ds1)")
ax.set_title(f"A second boundary: training diverges to non-finite\nweights only at "
             f"K/K* >= {div_k.min():.1f} (every rank)")
ax.legend(fontsize=8, loc="upper left")

# Transfer hump on calibrated placement.
ax = axes[2]
al = strip[~strip.diverged.astype(bool) & strip.test_p.notna() & ~strip.dead]
bins = np.geomspace(0.003, 1.2, 14)
iv = pd.IntervalIndex.from_breaks(bins)
gb = (al.groupby(pd.cut(al.K_rel, iv), observed=False).test_p.median().reindex(iv))
centers = np.sqrt(bins[:-1] * bins[1:])
ax.plot(centers, gb.to_numpy(), "o-", color="#0072B2", ms=7)
ax.axhline(BASE_PRIOR, color="0.5", ls=":", label="untrained base prior")
ax.set_xscale("log"); ax.set_xlabel("K / K*(ds1)"); ax.set_ylabel("median alive p(dog)")
peak = centers[np.nanargmax(gb.values)]
ax.set_title(f"Survivors' transfer is HUMPED in K\npeak at K/K* ~ {peak:.2f} "
             "(3rd independent design showing this)")
ax.legend(fontsize=8)
fig.tight_layout(); plt.show()
# Row units: (a) one (rank, alpha) boundary bracket; (b) one run; (c) one alive run, binned by K."""))

cells.append(md(r"""**The collapse law fitted directly on the strip — global, and with curvature.**
The strip analyses above handle the boundary's bend *piecewise* (L/H segments). Two
fits were never run on the calibrated cells until this notebook: a single global
(b, c), and a quadratic-logit curvature check — the collapse-side analogue of the
dose law's joint-quadratics test."""))

cells.append(code(r"""import statsmodels.api as sm
sf = strip[strip.test_p.notna() & ~strip.diverged.astype(bool) & (strip.s_scale <= 64)].copy()
sf["dead_i"] = sf.dead.astype(int)
lln, aln, rln = np.log(sf.lr), np.log(sf.alpha), np.log(sf["rank"])
grp = sf.tag.to_numpy()

def clogit(X, y, groups):
    # design-matrix Logit (not stats.cluster_logit, which is formula-based); same
    # cluster-robust covariance, but y/groups passed in rather than closed over.
    return sm.Logit(y, sm.add_constant(X)).fit(
        disp=0, maxiter=500, cov_type="cluster", cov_kwds={"groups": groups})

r_lin = clogit(np.c_[lln, aln, rln], sf.dead_i.to_numpy(), grp)
r_qa = clogit(np.c_[lln, aln, rln, aln**2], sf.dead_i.to_numpy(), grp)

def ratio_ci(res, j, i=1):
    est, (lo, hi) = astats.delta_ratio_ci(res.params, res.cov_params(), j, i)
    return est, (hi - lo) / 2

b_g, b_se = ratio_ci(r_lin, 2); c_g, c_se = ratio_ci(r_lin, 3)
print(f"GLOBAL linear collapse logit on the strip (n={len(sf)}, {sf.tag.nunique()} cells, "
      f"{sf.dead_i.sum()} dead, cluster=cell):")
print(f"  b = {b_g:.3f} +/- {b_se:.3f}    c = {c_g:.3f} +/- {c_se:.3f}")
print("  (comparison to the exponents in use then needs re-adjudication against the")
print(f"   designed-population fit b {cl.B_LAW:.3f}, c {cl.C_LAW:.3f}; the strip covers the H regime")
print("   uniformly while the adaptive grids lived mostly in L - the global number is a")
print("   support-weighted average of a curved boundary, which is exactly why it was not fit.)")

pq, Vq = np.asarray(r_qa.params), np.asarray(r_qa.cov_params())
zq = pq[4]/np.sqrt(Vq[4, 4])
lr_chi2 = 2*(r_qa.llf - r_lin.llf)
print(f"\n(ln alpha)^2 added to the collapse logit: cluster z = {zq:+.2f}, "
      f"LR chi2(1) = {lr_chi2:.1f}  ->  boundary curvature in alpha is REAL")
print("local b(alpha) = (beta_a + 2*beta_a2*ln a)/beta_lr :")
for a in (4, 16, 64, 256):
    g = np.zeros(5)
    g[2] = 1/pq[1]; g[4] = 2*np.log(a)/pq[1]; g[1] = -(pq[2]+2*pq[4]*np.log(a))/pq[1]**2
    b_loc = (pq[2] + 2*pq[4]*np.log(a))/pq[1]
    print(f"  b(alpha={a:3d}) = {b_loc:.3f} +/- {1.96*np.sqrt(max(g@Vq@g,0)):.3f}")
print("\n-> the bend is a SMOOTH glide (0.30 -> 0.99 across the alpha ladder), passing through")
print("   the piecewise values at the segment centers (~0.5 at alpha 16, ~0.99 at 256) - a")
print("   continuous alpha-dial, as the two-term dominance mixture predicts, not a sharp")
print("   regime switch. Attribution caveat: adding (ln lr)^2 as well makes the two curvature")
print("   terms fight (they are collinear ALONG the boundary locus, where ln lr ~ -b ln alpha)")
print("   - curvature is decisively detected, its lr-vs-alpha attribution is not separable")
print("   on the boundary set, mirroring the dose law's joint-quadratics situation.")
# Row unit: one strip run, s<=64 primary region, non-diverged; DEAD cut 0.01 (ds1 focus arm)."""))

cells.append(md(r"""**What the strip settles** (all `[cal]`):

1. **The boundary is real on outcome-independent placement**, at every rank, with a
   clean alive → dead → diverged ordering in K. The 42% dead fraction was fixed by
   the design, not chosen by it.
2. **The law is *regional*, and the bend is smooth**: b ≈ 0.5 on the low-α side,
   bending to ≈ 0.88 for α ≥ 64; the quadratic-logit fit above shows local b(α)
   gliding continuously 0.30 → 0.99 across the ladder (curvature z ≈ +3.5,
   LR χ² ≈ 18), exactly as the two-term dominance mixture predicts (§4.2–4.3) — a
   continuous α-dial, not a sharp regime switch. The then-published b = 0.561 was read as the
   L-regime law (the adaptive grids lived mostly there); the strip-global fit
   (b ≈ 0.62, c ≈ −0.15) is a support-weighted average of the curved boundary and
   should not be quoted as "the" exponent. (The regional reading was adjudicated against
   the retired trait-era exponent; it needs re-adjudication against the designed-population
   fit b """ + _B_REF + r""", c """ + _C_REF + r""".)
3. **A second sharp boundary exists** at K/K\* ≈ 12: outright optimization divergence
   (non-finite weights), distinct from collapse — collapsed runs at K/K\* 1–10 have
   perfectly finite, well-fitting weights (§1.2). Collapse is not "training blew up."
4. **The transfer hump replicates** on calibrated placement (peak K/K\* ≈ 0.4) — see
   the scope section below.

## §2.6 A1 verdict, and where the claim's support ends

**Established.** A stochastic collapse boundary K = lr·α^b·r^c ≥ K\* with b ≈ 0.56
(L-regime; regional bend to ~0.88 at α ≥ 64 `[cal]`), c ≈ −0.09 (rank protective,
model-free confirmed), replicated across: 3 teacher recipes × 6 student inits
(exponents I² ≈ 0%), 20+ teacher lineages (K\* within ~1.3×; the "teachers 66% of K\*
variance" figure is wipe-lost notes-only — re-derive before citing, see §2.4), 4 animals (boundary location within a narrow band of one
pooled anchor), 3 independent design generations (adaptive grid → shared rungs →
calibrated factorial), with arm-appropriate DEAD thresholds and estimator
cross-checks. The boundary is also *dose-like and portable* in a way HP tuples are
not: within its support, K alone predicts *whether* a cell collapses (fixed-K
high-rank twins agree 8/8 in both krungs arms).

**Not established / open scope** — the honest list for A1:

| open axis | status |
|---|---|
| model width/size | **one width (768, pythia-160m) everywhere.** Nor's Adam-width argument (Frobenius laws can be width artifacts) is unanswered and untestable at one width. The single most important robustness gap. |
| number pool / dataset | one pool family (`ex10-20_prefix_n100k`); constrained-generation pools exist but no law refit on them |
| epochs | exponents are ep20-only; K\* moves ~1.3× over ep5→ep20 (∝ epochs^−0.2 — itself a mechanism clue, §4.5) |
| teacher quality | all law teachers are near-ceiling on their own trait; a genuinely weak teacher's boundary is unmeasured |
| optimizer | AdamW with wd=0 everywhere (≡ Adam); no SGD arm (mechanism predicts b→~1.5), no wd ladder |
| rank 4 anomaly | conditional K\* dips 0.5–0.75× at r=4 on 4 arms (I²=97%) — a real low-rank departure the smooth law smooths over |
| DEAD label | a convention (live-mode/10); fine within an arm, must be re-derived per arm forever |"""))

# =====================================================================================
# Part A1, continued — the law is not about SL (A1's scope)
# =====================================================================================

cells.append(md(r"""---
# Part A1, continued — The law is not about subliminal learning (A1's scope)

**The reframe (Angela, 2026-08-11), which the evidence has been pointing at from
several directions at once: the collapse boundary is not an SL phenomenon.** It is a
generic stability boundary of aggressive LoRA distillation on number data — present
with no trait in the data at all — and SL is a *separate* phenomenon that lives
inside (and is destroyed by) it. Four independent lines say so:

1. **The trait-free control collapses too** (`collapse_mechanism.ipynb` §6c
   `[adaptive]`): base-model numbers distilled into the same student collapse at
   K ≳ 1.8e-3, with no dog signal anywhere in the pipeline.
2. **The boundary is indifferent to the trait's content and strength**: one anchor
   serves dog/cat/lion/pig (§2.4), and teachers differing ~11× in yield differ only
   ~1.3× in K\* (§2.2).
3. **The law derives from pure update mechanics** — the dose law and its level set
   (§4.2) contain no trait term at any step.
4. **Death is generic**: containment failure on neutral text (§4.4), and λ-revival
   shows the trait sits *under* the damage, masked not erased (§4.7).

**Consequence for modeling transfer.** The hurdle identity
E[p] = P(alive | HP) · E[p | alive, HP] remains true, but it should be read as a
*separation of nuisance from signal*, not as a model of SL: the first factor belongs
to training-stability physics (the law), and everything SL is in the second. Fitting
"actual transfer" through the law was the wrong ambition — the law's proper role is
to define the **observability window** inside which SL experiments are valid.

**Two caveats that bound the reframe:**

- *The boundary's location is not perfectly data-source-free.* The trait-free
  threshold (~1.8e-3) sits roughly 2× above the trait-numbers K\* (~0.8e-3), and in
  4/12 matched cells the treatment was suppressed below the whole control range —
  the trait channel is marginally **more** fragile than generic behavior. The
  control grid is only 22 cells / 12 matched, far too thin to quantify this; a
  trait-free boundary factorial is the missing pivotal experiment (blueprint).
- *Dose does not leave the second factor.* The hump (below) shows K modulating
  surviving transfer smoothly and causally below the boundary — so K appears in
  E[p | alive] as a continuous dose response, just never as the law.

Within that framing, what the survivor factor actually looks like:

- **The collapse half is nearly deterministic in K** (AUCs ≈ 0.99); **the survivor
  half is mostly noise** — HPs explain only a small fraction of survivor transfer.
- **The survivor response is HUMPED in K**, peaking *below* the boundary: ≈ 0.42 on
  the calibrated strip (§2.5c `[cal]`), with each law arm's survivor band-mean
  peaking in the 0.5–1.0 band and falling past the cliff (executed §6 of
  `collapse_law_generality.ipynb`; survivor Pearson(ln K, p) only +0.12–0.34), and
  the λ intervention traces the same hump *causally within a single run* (§4.7).
  A straight-line conditional fit through a hump is a local average of its support —
  which is why single conditional lr/α exponents never replicated across lineages.
  ⚠ *Provenance note:* the quantitative conditional-transfer refit (rank coefficient
  +0.124 pooled at I² = 0%, peaks at K/K\* 0.28–0.55, median R² 0.12, and its
  `fit_conditional_transfer` estimator) lived in the wipe-lost §15; those exact
  numbers are session-notes-only until re-derived. The *shape* claims above stand on
  the executed sources named.
- **Yield is lineage-specific**: survivor levels differ ~7× across the six law arms
  and ~2× between inits at matched cells (krungs §9.4: median ws1/ds1 survivor ratio
  0.52), on a boundary that moves only 1.4×.
- **Yield is also pretraining-maturity-specific — a new axis (2026-08-10).** The
  cross-checkpoint distillation line (`checkpoint_distillation_analysis.ipynb`,
  draining) distills teacher@step_i → student@step_j across pythia-160m pretraining
  checkpoints at a fixed safe student cell (K/K\* ≈ 0.74): same-checkpoint transfer
  rises ~9× from 25k → 125k steps while teacher quality saturates and the base prior
  stays flat; backward pairs collapse into early students (125k→70k = 0.074 vs
  125k→110k = 0.705 at *matched* prior); and the teacher pool's detectability
  (per-token KL vs base) does not predict within-step transfer. Transfer needs
  representational maturity on **both** sides — a gate entirely outside the
  (lr, α, r) law.

Practical consequence (updated operating rule): the EV-optimal operating point is
near **half** the boundary, not just under it; past K/K\* ≈ 0.55 extra aggression
costs transfer *before* it kills the seed, and the knob that still pays at fixed
aggression is rank.

Related context for the cross-init "why" (not collapse-specific, one line each,
all `[recorded]`): representational similarity does **not** explain which
init transfers better (CKA prediction inverted; `cka_analysis.ipynb`), shared init
does **not** preserve output dispositions (decodability probe H1 null), the ds-vs-ws
transfer gap survives adjustment for measured student-base priors
(`transfer_factor_decomposition.ipynb`, permutation inference). The transfer-level
question is genuinely open; the boundary question is not."""))

# =====================================================================================
# Part A2 — update size causes fate
# =====================================================================================

cells.append(md(r"""---
# Part A2 — The update's size causes fate

**Claim (A2).** Fate is a (soft) threshold on the update's size; moving the size
moves the fate, in both directions, at eval time and during training.
**Status: closed, with a revision** — the causal dose is **trajectory-like**
(sustained exposure); the endpoint norm of the correlational leg is its high-quality
proxy; a bounded ~2–4× direction factor sets each update's personal threshold
(§4.10).

Three legs, in chain order: correlational (endpoint dose thresholds fate `[cal]`,
§4.1/§4.4), causal at eval time (λ-rescaling flips fate both ways `[interv]`, §4.7),
causal at training time (the clamp `[interv-train]`, §4.9 — the tier that forced the
revision).

## §4.1 The dose account, calibrated — the correlational leg"""))

cells.append(code(r"""from sklearn.metrics import roc_auc_score
sv = strip[strip.test_p.notna() & ~strip.diverged.astype(bool) & strip.dw_fro.notna()].copy()
auc_fro = roc_auc_score(sv.dead, sv.dw_fro)
auc_spec = roc_auc_score(sv.dead, sv.dw_spec)
auc_K = roc_auc_score(sv.dead, sv.K)
print(f"fate classification AUC on the strip: dw_fro {auc_fro:.3f}, dw_spec {auc_spec:.3f}, "
      f"fitted K {auc_K:.3f}   (n={len(sv)})")
print("NOTE (Nor's audit): pooled fate AUCs saturate for BOTH norms - AUC is evidence")
print("that dose thresholds fate, NOT evidence for which norm; the norm question is")
print("Part A4's.")

fig, ax = plt.subplots(figsize=(7.2, 4.4))
ax.scatter(sv[~sv.dead].dw_fro, sv[~sv.dead].test_p.clip(1e-11), s=12, color=C_ALIVE,
           alpha=0.6, label="alive")
ax.scatter(sv[sv.dead].dw_fro, sv[sv.dead].test_p.clip(1e-11), s=12, color=C_DEAD,
           alpha=0.6, label="dead")
ax.set_xscale("log"); ax.set_yscale("log")
ax.axhline(0.01, color="0.5", ls=":", lw=1.2)
ax.axhline(BASE_PRIOR, color="0.5", ls="-.", lw=1.0, label="base prior")
ax.set_xlabel("dose  dw = ||dW||_F / ||W0||_F  (module mean)")
ax.set_ylabel("test p(dog)")
ax.set_title(f"Fate is a soft threshold on measured update size\nAUC(dw_fro) = {auc_fro:.3f} "
             f"~ AUC(fitted law K) = {auc_K:.3f}")
ax.legend(fontsize=9, loc="lower left")
fig.tight_layout(); plt.show()
# Row unit: one strip run, non-diverged; fates at the ds1 0.01 cut."""))

cells.append(md(r"""**Two facts to keep separate.** Dead runs *at the boundary* have barely moved
(pooled Frobenius ratio ‖W_f‖/‖W₀‖ ≤ 1.007 at the smallest killing doses); the huge
norms far past the boundary (σ_max up to 171×) are aftermath, not cause. And
endpoint magnitude does **not** separate fates *within* a mixed cell: on the strip
the dying seed has the *smaller* dw in 6/9 mixed cells (two-sided p = 0.51 —
coin-flip at calibrated n) `[cal]`, and the adaptive large arms, with many more
mixed cells, even pointed the wrong way for a size story (dying seeds slightly
**smaller**, z ≈ −3.2) `[adaptive]`. Size is the **between-cell** variable; whatever
picks the dying seed within a cell is not the endpoint norm (§4.10's direction
factor is the current best candidate).

Also carried from the doc, the containment jump `[cal]` `[recorded — nors_asks §4]`:
regressing log Pile-KL on the update's (energy, spread), the dead indicator carries
a **+1.87 log-nat offset (≈6.5×) at matched dose** (p ≈ 3e-92); slopes also differ
by fate (γ_E 0.73 alive vs 0.22 dead — saturation). KL is *not* a smooth function of
measured update size — crossing the boundary is an event, not the tail of a smooth
response."""))

cells.append(md(r"""## §4.4 What death is: loss of selectivity **[cal]**

The trait is trained through a numbers channel; the model's behavior everywhere else
should be untouched. Measure both sides on the strip runs: movement on the numbers
channel (training-like inputs) and displacement on neutral text (Pile logit KL vs
the base model)."""))

cells.append(code(r"""# Strip activations: layer-0 row of `drift` carries the logit-index KL; take final-layer
# rel_drift as the drift readout. ftseed1 cells only (the probe's design).
def load_act(cond):
    rows = []
    for r in (1, 2, 4, 8, 16, 32, 64, 128):
        p = AN / f"adapter_activations_strip_{cond}_r{r}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        dr = pd.DataFrame(d["drift"])
        kl = dr[dr.layer == 0].set_index(["student_tag", "ftseed"]).kl_base_adapted
        fin = (dr[dr.layer == dr.layer.max()]
               .set_index(["student_tag", "ftseed"]).rel_drift)
        rows.append(pd.DataFrame({"kl": kl, "final_drift": fin}).reset_index())
    return pd.concat(rows, ignore_index=True)
pile = load_act("pile").rename(columns={"kl": "pile_kl", "final_drift": "pile_drift"})
nums = load_act("numbers").rename(columns={"kl": "num_kl", "final_drift": "num_drift"})
act = pile.merge(nums, on=["student_tag", "ftseed"], how="inner")
act = act.merge(strip[["tag", "seed", "test_p", "dead", "diverged", "K_rel"]],
                left_on=["student_tag", "ftseed"], right_on=["tag", "seed"], how="inner")
act = act[act.test_p.notna() & ~act.diverged.astype(bool)]
auc_pile = roc_auc_score(act.dead, act.pile_kl)
auc_num = roc_auc_score(act.dead, act.num_drift)
print(f"strip activations, n={len(act)} runs: AUC(pile KL) = {auc_pile:.3f}, "
      f"AUC(numbers-channel final drift) = {auc_num:.3f}")
print(f"pile KL medians: alive {act[~act.dead].pile_kl.median():.2f} vs "
      f"dead {act[act.dead].pile_kl.median():.2f} nats")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.0), sharey=False)
ax = axes[0]
for fate, color, lab in [(False, C_ALIVE, "alive"), (True, C_DEAD, "dead")]:
    sub = act[act.dead == fate]
    ax.scatter(sub.num_drift, sub.pile_kl.clip(1e-3), s=14, alpha=0.6, color=color, label=lab)
ax.set_yscale("log")
ax.set_xlabel("numbers-channel drift (final layer, training-like inputs)")
ax.set_ylabel("Pile logit KL vs base (nats)")
ax.set_title(f"Selectivity split: trait-channel movement is fate-blind\n(AUC {auc_num:.2f}), "
             f"neutral-text KL separates fates at AUC {auc_pile:.3f}")
ax.legend(fontsize=9)
ax = axes[1]
bins = np.geomspace(act.K_rel.min(), act.K_rel.max(), 16)
for fate, color, lab in [(False, C_ALIVE, "alive"), (True, C_DEAD, "dead")]:
    sub = act[act.dead == fate]
    ax.scatter(sub.K_rel, sub.pile_kl.clip(1e-3), s=14, alpha=0.6, color=color, label=lab)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("K / K*(ds1)"); ax.set_ylabel("Pile logit KL vs base (nats)")
ax.set_title("KL grows through the boundary - fate and neutral-text\ndisplacement are "
             "sibling symptoms of one containment failure")
ax.legend(fontsize=9)
fig.tight_layout(); plt.show()
# Row unit: one strip run with both activation probes (ftseed 1 by probe design)."""))

cells.append(md(r"""**Reading.** Alive runs move the numbers channel just as much as dead runs (that
movement *is* learning). What separates fates is off-channel displacement: dead runs
have stopped confining their function change to the trained channel. "Collapse of
transfer" and "generic behavioral destruction" are one event — which is also why the
DEAD label could be defined on p_target alone. The distributions are disjoint at the
5–95% level (alive P95 ~2.1 nats vs dead P5 ~5.5 — a literal ~3.4-nat gap, not just
separated medians), and the clean selectivity statement is conditional: *near the
boundary / at matched dose*, dead runs move the trait channel like alive runs but
leak KL (the wildly heavy trait-drift tail of deep-dead runs is the
input-independent degeneracy artifact, not selective trait movement — nors_asks
§4b).

**Can we say update size *causes* the Pile-KL spike?** Not from this section — it is
correlational. But the λ experiment (§4.7) measured Pile KL at every rescaling step,
and KL moves monotonically with pure dose rescaling *within each run*: at eval time,
**yes**. What remains uninferable in either direction: whether the KL displacement
*causes* the trait loss or vice versa — the honest statement stays "sibling symptoms
of one containment failure".

## §4.7 The causal leg at eval time **[interv]**

The design family: take a *trained* adapter, modify the update itself in exactly one
respect — here its **scale** — splice it back at evaluation time, and re-measure
both readouts (trait p_target on the 50 held-out preference prompts; Pile logit KL
vs base on a fixed held-out doc pool). Three properties make these experiments
strong despite small original cell counts: **paired within-run contrasts** (each
unit of evidence is a full curve inside one trained run — run-level confounds
cancel exactly; with effects 10–1000× the readout noise, sign/monotonicity claims
saturate at n = 8); **one manipulated variable each** (λ preserves the direction
exactly, via lora_B scaling); **predictions written down first** (and two of the
family's predictions lost — the family is falsification-capable).

The λ tier below: ΔW → λ·ΔW on 8 boundary-straddling cells plus 9 trained-dead
revival cells, later expanded to 109 rule-frozen cells × both seeds (cell lists
frozen in `run_logs/causal_exp_cells_v2.json` before any outcome — §4.10 scores that
sweep). Sibling experiments of the same family appear where their claims live: term
splices in Part A3, spread/carrier surgeries in Part A4, the W₀-sliver surgery in
§4.6 (retired).

The headline causal sentences, with exact scope:

1. **Dose is causally sufficient to flip fate, in both directions, at eval time** —
   monotone in every cell, including revival of dead models (near-boundary death is
   **masking**, not erasure; deep death shades to erasure — a continuum) — and the
   flip point is **history-dependent**: a direction-quality factor of ~2–4×
   (largest at low α; ~3× covariate-adjusted at sweep scale, §4.10) sets each
   update's personal threshold.
2. Scope: these license claims about the trained *endpoint's* update. The
   training-time version is §4.9's clamp `[interv-train]` — closed 2026-08-12, with
   the trajectory revision. The original 8 λ cells were selected by fate and
   boundary position (legitimate for within-cell causality; cross-cell quantities
   inherit that selection — fixed by the rule-frozen expansion in §4.10)."""))

cells.append(code(r"""# Lambda dose-response: dW -> lam*dW (direction preserved) on 8 boundary-straddling cells.
lam_files = sorted(AN.glob("frozen_a_ablation_scale_lam*.json"))
lam_rows = []
for f in lam_files:
    d = json.loads(f.read_text())
    for r in d["results"]:
        lam_rows.append(dict(tag=r["student_tag"], ftseed=r["ftseed"],
                             lam=r["scale_lambda"], p=r["p_target_ablated"],
                             kl=r["pile_kl_base_ablated"], p_orig=r["p_target_stored"]))
lam = pd.DataFrame(lam_rows).sort_values("lam")
lam["cell"] = lam.tag + "/s" + lam.ftseed.astype(str)
lam["trained_dead"] = lam.p_orig < DEAD_TRAIT_DS1

# Revival curves (9 trained-dead cells, lam <= 1).
rev_rows = []
for f in sorted(AN.glob("frozen_a_ablation_revival_lam*.json")):
    d = json.loads(f.read_text())
    for r in d["results"]:
        rev_rows.append(dict(tag=r["student_tag"], ftseed=r["ftseed"],
                             lam=r["scale_lambda"], p=r["p_target_ablated"]))
rev = pd.DataFrame(rev_rows).sort_values("lam")
rev["cell"] = rev.tag + "/s" + rev.ftseed.astype(str)

fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.2))
ax = axes[0]
for cell, sub in lam.groupby("cell"):
    color = C_DEAD if sub.trained_dead.iloc[0] else C_ALIVE
    ax.plot(sub.lam, sub.p.clip(1e-8), "o-", color=color, alpha=0.75, ms=5, lw=1.4)
ax.axhline(0.01, color="0.5", ls=":", lw=1.2)
ax.axhline(BASE_PRIOR, color="0.5", ls="-.", lw=1.0)
ax.set_yscale("log"); ax.set_xlabel("lambda (update rescale factor)")
ax.set_ylabel("test p(dog) after splice")
ax.set_title("Pure dose rescaling flips fate monotonically, BOTH ways\n"
             f"blue = trained-alive ({int((~lam.trained_dead).sum()/lam.lam.nunique())} cells), "
             f"red = trained-dead; dotted = DEAD cut, dash-dot = base prior")
ax = axes[1]
for cell, sub in lam.groupby("cell"):
    color = C_DEAD if sub.trained_dead.iloc[0] else C_ALIVE
    ax.plot(sub.lam, sub.kl.clip(1e-3), "o-", color=color, alpha=0.75, ms=5, lw=1.4)
ax.set_yscale("log"); ax.set_xlabel("lambda")
ax.set_ylabel("Pile logit KL vs base (nats) after splice")
ax.set_title("...and the Pile-KL displacement moves WITH the dose:\nupdate size causally "
             "drives the neutral-text spike (eval-time)")
ax = axes[2]
for cell, sub in rev.groupby("cell"):
    ax.plot(sub.lam, sub.p.clip(1e-8), "o-", color=C_DEAD, alpha=0.75, ms=5, lw=1.4)
ax.axhline(0.01, color="0.5", ls=":", lw=1.2); ax.axhline(BASE_PRIOR, color="0.5", ls="-.", lw=1.0)
ax.set_yscale("log"); ax.set_xlabel("lambda"); ax.set_ylabel("test p(dog)")
ax.set_title(f"Revival curves: {rev.cell.nunique()} trained-DEAD cells scaled down\n"
             "trait re-emerges above base -> collapse near the boundary is MASKING, not erasure")
fig.tight_layout(); plt.show()

# History-dependent crossing: effective dose lam * K/K* at the fate flip, by training
# fate. (K_rel is computed from each cell's own tag - the intervention cells are
# adaptive-era boundary cells, so the strip's measured dw is not available for them;
# K is the dose scale the level-set result licenses.)
def k_rel_of_tag(tag):
    return cl.k_from_tag(tag, b=B_DESIGN, c=C_DESIGN) / KSTAR_DS1
def crossing(sub, dead_at_train):
    sub = sub.sort_values("lam")
    alivemask = sub.p >= DEAD_TRAIT_DS1
    d0 = k_rel_of_tag(sub.tag.iloc[0])
    if dead_at_train:   # revive threshold: largest lam still alive
        lams = sub.lam[alivemask]
        return d0 * lams.max() if len(lams) else np.nan
    lams = sub.lam[~alivemask]   # kill threshold: smallest lam already dead
    return d0 * lams.min() if len(lams) else np.nan
cross = []
for cell, sub in pd.concat([lam, rev.assign(trained_dead=True, p_orig=np.nan)]).groupby("cell"):
    td = bool(sub.trained_dead.iloc[0])
    cross.append(dict(cell=cell, trained_dead=td, flip_dose=crossing(sub, td)))
cross = pd.DataFrame(cross).dropna()
print("effective dose lam * K/K* at the fate flip (history-dependence):")
print(cross.groupby("trained_dead").flip_dose.describe().round(3)[["count", "min", "50%", "max"]])
med_a = cross.loc[~cross.trained_dead, "flip_dose"].median()
med_d = cross.loc[cross.trained_dead, "flip_dose"].median()
print(f"-> trained-ALIVE updates keep the trait up to lam-doses ~{med_a:.2f}x K* (median), "
      f"while trained-DEAD updates revive only below ~{med_d:.2f}x K* (median):")
print("   dose is causally sufficient for fate, but each update's threshold depends on the")
print("   direction it learned - 'direction quality' is real, just not cos(dW, W0).")
print("   (In measured-dw units the recorded gap is ~2x: die at lam*dw 0.14-0.21 vs revive")
print("   below 0.08-0.13, nors_asks 5.5 - the K-unit gap here is wider because trained-dead")
print("   cells sit deeper past the boundary before rescaling.)")"""))

cells.append(md(r"""## §4.9 The causal leg at training time — the clamp **[interv-train]** (closed 2026-08-12)

The λ results license claims about the trained endpoint's update, not about
training. The clamp closes that gap: **train fresh runs in strip cells verified
both-seeds-dead (K/K\* 1.17–2.34), while a max-norm projection (lora_B rescale,
direction-preserving — the training-time mirror of the λ splice) caps the pooled
dose at ratio × the per-rank dead-edge threshold θ̂ at every optimizer step.**
Directions stay entirely free; only magnitude is bounded; the HPs stay pinned at
verified-lethal values — inferentially this estimates the controlled direct effect
of the HPs on fate at fixed dose. Design frozen in `run_logs/clamp_design_v1.json`
before any outcome (7 cells × {fro, spec} × ceiling ladder; the stage-2 low-ceiling
extension and the stage-3 deep-α extension were rule-frozen the same way; stage-3
arms may still be draining — the census below counts whatever has landed).

Pre-registered predictions: dose-causal → low ceilings rescue nearly all;
direction-corruption → death at every ceiling; peak account → the σ-clamp rescues
like the F-clamp."""))

cells.append(code(r"""CLAMP_DESIGN = REPO / "run_logs" / "clamp_design_v1.json"
CL_ROOT = OUTPUT_ROOT / "clamp_students"
CL_MID = ("dog/pythia-160m-data-seed1/teacher-pythia-160m/ep2_lr5e-4_r32a64_tseed1/"
          "ex10-20_prefix_n100k")
if not CLAMP_DESIGN.exists() or not CL_ROOT.exists():
    print("clamp experiment not on disk — skipped")
else:
    dsn = json.loads(CLAMP_DESIGN.read_text())

    # one batched fetch of the arms' cell summaries (shared record cache) instead of one
    # read per arm; arms without a summary are simply absent from the mapping, as before
    from subliminal_learning.analysis.frames import load_cell_summaries

    arm_rel = {arm["tag"] + "|" + arm["label"]: f"{arm['label']}/{CL_MID}/{arm['tag']}"
               for arm in dsn["arms"]}
    cl_docs = load_cell_summaries(CL_ROOT, list(arm_rel.values()))

    rows = []
    for arm in dsn["arms"]:
        doc = cl_docs.get(arm_rel[arm["tag"] + "|" + arm["label"]])
        if doc is None:
            continue
        t = summary_set_block(doc, "test")
        for seed_s, pv in ((t or {}).get("per_seed_p_target", {})).items():
            rows.append({"tag": arm["tag"], "K_rel": arm["K_rel"], "norm": arm["norm"],
                         "ratio": arm["ratio"], "rank": arm["rank"],
                         "seed": int(seed_s), "p": float(pv)})
    cf = pd.DataFrame(rows).drop_duplicates(["tag", "norm", "ratio", "seed"])
    if not len(cf):
        print("no clamp runs landed yet — skipped")
    else:
        cf["alive"] = cf.p >= DEAD_TRAIT_DS1
        n_design = 2 * len(dsn["arms"])
        print(f"clamp census: {len(cf)}/{n_design} designed runs landed "
              f"({cf.tag.nunique()} cells, all verified both-seeds-dead unclamped; "
              "stage-3 deep-alpha rungs join as they drain)")
        g = cf.groupby(["norm", "ratio"]).agg(
            surv=("alive", "mean"), n_alive=("alive", "sum"), n=("alive", "size"))
        print("\nSURVIVAL by arm (ratio = ceiling / per-rank theta-hat on the clamped norm):")
        display(compare_table(g.round(3), fmt="{:.3f}"))

        fig, ax = plt.subplots(figsize=(8, 4.4))
        for norm, color, lab in (("fro", C_FRO, "Frobenius clamp"),
                                 ("spec", C_SPEC, "spectral clamp")):
            if norm not in g.index.get_level_values(0):
                continue
            sub = g.loc[norm].sort_index()
            ax.plot(sub.index, sub.surv, "o-", color=color, ms=10, lw=2, label=lab)
            for ratio, row in sub.iterrows():
                ax.annotate(f"{int(row.n_alive)}/{int(row.n)}", xy=(ratio, row.surv),
                            xytext=(0, 9), textcoords="offset points",
                            ha="center", fontsize=9, color=color)
        ax.axvline(1.0, color="0.4", ls=":", lw=1.2)
        ax.set_ylim(-0.05, 1.12)
        ax.set_xlabel("dose ceiling / theta-hat (per-rank unclamped death threshold)")
        ax.set_ylabel("survival fraction")
        ax.set_title("Blocking the mediator blocks the effect: survival is monotone in the\n"
                     "dose ceiling while the HPs stay pinned at verified-lethal values\n"
                     "(annotations: alive/landed per rung)")
        ax.legend(fontsize=9, loc="upper right")
        fig.tight_layout(); plt.show()

        piv = cf[cf.norm == "fro"].pivot_table(index="tag", columns="ratio",
                                               values="alive", aggfunc="mean")
        piv.insert(0, "K_rel", cf.groupby("tag").K_rel.first())
        print("fro-arm survival by cell x ratio (1.0 = all landed seeds alive):")
        display(compare_table(piv.round(2), fmt="{:.2f}"))
# Row unit: one clamped training run; fate from its preference_summary test p, cut 0.01.
# Survival counts recompute live as the stage-3 deep-alpha queues drain."""))

cells.append(md(r"""**What the clamp settled.**

1. **Blocking the mediator blocks the effect.** Survival is steeply monotone in the
   ceiling — at stage-1/2 settlement: 14/14 at 0.5×θ̂ (including all lr5e-4 runs),
   11/14 at 0.6, 3/14 at 0.8, **0/54 at ≥ 1.0** (the live census above supersedes
   these counts as stage-3 lands). Sharpest single-cell demonstration
   (lr5e-4_r64a8, both seeds): fate walks alive→dead purely with the ceiling while
   the HPs stay pinned at lethal values. Direction-corruption is **refuted** (no
   ceiling-independent death). Practically: a ~0.5×θ̂ clamp = train at any lr with
   zero seed deaths.
2. **The revision — the causal dose is trajectory-like.** *Endpoint*-dose
   sufficiency is **false**: 58 clamped runs realized final dose < 0.95×θ̂_F and 47%
   of them still died — in cells whose unclamped runs die *above* θ̂, the clamped
   ones die *below* it `[recorded — realized-dose pass, nors_asks §5.5c]`. Clamped
   trajectories ride the ceiling for 60–90% of steps; **sustained exposure kills
   where a brief endpoint approach survives**; the operative threshold under a
   ceiling-riding trajectory is ~0.6×θ̂. The endpoint dw of the correlational leg is
   therefore a **proxy** for the causal quantity (an excellent one — at fixed
   budget, endpoint ordering tracks trajectory-exposure ordering across cells), not
   the causal quantity itself.
3. **The σ-vs-F discrimination physically failed** on-trajectory: trained updates
   are so low-srank that capping either norm dragged the other along (spec-clamped
   runs realize F at 0.67–1.50×θ̂_F) — norm attribution stays with the eval-time
   surgeries (Part A4).

## §4.5 Independent evidence that the decision is early and exposure-like **[adaptive]** **[recorded]**

Consistent with the trajectory reading — but all of it adaptive-era, with no
calibrated version yet (the ckpt-replay pilot, 2026-08-13, retraces 4 strip cells'
original runs with 40 checkpoints each — the first within-run dose/decision
trajectories on calibrated cells):

- ‖B(t)‖ grows as √(cumulative lr) on aggressive cells; the endpoint-vs-lr exponent
  q falls from ~1.48 (ep5) to ~1.16 (ep20): a **ballistic→diffusive crossover**,
  decisions made while growth is still ballistic (`collapse_mechanism` §6b).
- **K\* barely moves with dose accumulation**: 4× the steps shifts the boundary only
  ~1.3× (K\* ∝ epochs^−0.2) — a pure accumulation story predicts ∝ 1/epochs, many σ
  away. The endpoint dose threshold grows sublinearly (brackets √T).
- The one captured *dying* seed was indistinguishable from surviving siblings in
  every norm/coherence trace at every checkpoint — whatever separates it is not in
  the size trajectory (what pushed the program toward direction/subspace).
- Loss curves are unavailable (the student trainer kept no trainer state); the
  endpoint substitute (§1.2's postmortem CE) rules out optimization failure."""))

cells.append(md(r"""## §4.10 The direction factor, measured at full-sweep scale **[interv]**

The λ flip dose is history-dependent: trained-dead updates revive only at lower
λ-doses than trained-alive updates tolerate — *something about the learned
direction* does causal work on top of size. The original 8-cell reading was "~2×".
At sweep scale (all scored crossings over the 109 rule-frozen lamstrat + shellext
cells, both seeds) the raw band ratio is confounded — trained-dead cells sit deeper
*by design* — so the estimate below adjusts for placement (log K_rel + log α +
log r, cell-clustered); the within-cell mixed-fate pairs are the clean-but-small
fully-controlled version. Crossings are interval-censored midpoints of the
[last-alive, first-dead] λ bracket, scored on the up-side of the hump, at the arm's
own cut (all cells here are ds1: 0.01), in λ·K/K\* units."""))

cells.append(code(r"""import statsmodels.formula.api as smf
LS_FILES = (sorted(AN.glob("frozen_a_ablation_lamstrat*_lam*.json"))
            + sorted(AN.glob("frozen_a_ablation_shellext_s*_lam*.json")))
if not LS_FILES:
    print("sweep-scale lambda results not on disk — skipped")
else:
    lam_rows2 = []
    for f in LS_FILES:
        d = json.loads(f.read_text())
        seed = d.get("ftseed", 2 if "_s2" in f.name else 1)
        for r in d["results"]:
            lam_rows2.append({"cell": r["student_tag"], "seed": seed,
                              "lam": d["scale_lambda"], "p": r["p_target_ablated"],
                              "p_orig": r["p_target_stored"]})
    ls = pd.DataFrame(lam_rows2).drop_duplicates(["cell", "seed", "lam"])
    tp = pd.DataFrame([parse_param_tag(t) for t in ls.cell], index=ls.index)
    ls["lr"], ls["rank"], ls["alpha"] = tp.lr, tp["rank"], tp.alpha
    ls["K_rel"] = cl.K_of(ls.lr, ls.alpha, ls["rank"], b=B_DESIGN, c=C_DESIGN) / KSTAR_DS1
    print(f"pooled lambda data (lamstrat + shellext, both seeds): {ls.cell.nunique()} "
          f"cells, {ls.groupby(['cell', 'seed']).ngroups} run-curves, {len(ls)} evals")

    def crossings(df, cut=DEAD_TRAIT_DS1):
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
                continue          # censored curve: no crossing observed
            fd = next((i for i in range(len(up)) if not alive.iloc[i]), None)
            if fd is None or fd == 0:
                continue
            mid = float(np.sqrt(up.lam.iloc[fd - 1] * up.lam.iloc[fd])) * sub.K_rel.iloc[0]
            out.append({"cell": c, "seed": s, "alpha": sub.alpha.iloc[0],
                        "rank": sub["rank"].iloc[0],
                        "K_rel": round(sub.K_rel.iloc[0], 3), "trained_dead": td,
                        "cross_K": mid, "kind": "revive" if td else "kill"})
        return pd.DataFrame(out)

    okx = crossings(ls)
    if not len(okx):
        print("no scored crossings — skipped")
    else:
        print(f"scored crossings: {okx.kind.value_counts().to_dict()}")
        okx["yx"] = np.log(okx.cross_K)
        okx["lkr"], okx["laln"], okx["lrln"] = (np.log(okx.K_rel), np.log(okx.alpha),
                                                np.log(okx["rank"]))
        m0x = smf.ols("yx ~ trained_dead", okx).fit(
            cov_type="cluster", cov_kwds={"groups": okx.cell})
        m1x = smf.ols("yx ~ trained_dead + lkr + laln + lrln", okx).fit(
            cov_type="cluster", cov_kwds={"groups": okx.cell})
        facts = {}
        for nm, m_ in (("raw (confounded by placement)", m0x),
                       ("covariate-adjusted", m1x)):
            cx_ = m_.params["trained_dead[T.True]"]
            se_ = m_.bse["trained_dead[T.True]"]
            facts[nm] = np.exp(-cx_)
            print("%s: alive/dead crossing-dose factor = %.2fx [%.2f, %.2f] (p = %.1e)"
                  % (nm, np.exp(-cx_), np.exp(-cx_ - 1.96 * se_),
                     np.exp(-cx_ + 1.96 * se_), m_.pvalues["trained_dead[T.True]"]))

        # THE decisive contrast: within-cell mixed-fate pairs (same HPs, opposite
        # fates - placement cannot confound).
        fate = ls.groupby(["cell", "seed"]).p_orig.first().lt(DEAD_TRAIT_DS1).unstack("seed")
        mixed = fate.dropna()
        mixed = mixed[mixed[1] != mixed[2]]
        prs = []
        for c in mixed.index:
            ds_ = 1 if mixed.loc[c, 1] else 2
            ka = okx[(okx.cell == c) & (okx.seed == 3 - ds_)]
            kd = okx[(okx.cell == c) & (okx.seed == ds_)]
            if len(ka) and len(kd):
                prs.append(float(ka.cross_K.iloc[0] / kd.cross_K.iloc[0]))
        prs = pd.Series(prs, dtype=float)
        if len(prs):
            print(f"within-cell mixed-fate pairs: n = {len(prs)}, "
                  f">1 in {(prs > 1).sum()}/{len(prs)}, median ratio {prs.median():.2f}x, "
                  f"range {prs.min():.1f}-{prs.max():.1f}")

        # Adjusted factor by alpha: covariate residuals with the fate effect added back.
        okx["resx"] = m1x.resid + m1x.params["trained_dead[T.True]"] * okx.trained_dead
        ga_ = okx.groupby(["alpha", "trained_dead"]).resx.median().unstack()
        na_ = okx.groupby(["alpha", "trained_dead"]).size().unstack(fill_value=0)
        ta_ = pd.DataFrame({"factor": np.exp(ga_[False] - ga_[True]),
                            "n_kills": na_[False], "n_revivals": na_[True]}).dropna()
        print("\nadjusted alive/dead factor by alpha (row unit: one alpha level; "
              "factor from covariate-residual medians):")
        display(compare_table(ta_.round(2), fmt="{:.2f}"))

        fig, ax = plt.subplots(figsize=(8.2, 4.4))
        ax.plot(ta_.index, ta_.factor, "o-", color="#0072B2", ms=9, lw=2)
        for a, row in ta_.iterrows():
            ax.annotate(f"{int(row.n_kills)}k/{int(row.n_revivals)}r",
                        xy=(a, row.factor), xytext=(0, 9),
                        textcoords="offset points", ha="center", fontsize=9)
        adj = facts["covariate-adjusted"]
        ax.axhline(adj, color="#009E73", lw=1.6, ls="--",
                   label=f"sweep-wide adjusted {adj:.2f}x")
        ax.axhline(1.0, color="0.4", ls=":", label="no history dependence")
        ax.set_xscale("log")
        ax.set_xlabel("alpha")
        ax.set_ylabel("alive/dead crossing-dose factor (adjusted)")
        ax.set_title("The direction factor is alpha-STRUCTURED: largest where the MOVED\n"
                     "(learned) term dominates, shrinking toward 1 in the frozen-dominated\n"
                     "regime (annotations: n kills / n revivals per alpha level)")
        ax.legend(fontsize=9, loc="upper right")
        fig.tight_layout(); plt.show()
        print("READ: the frozen term's direction is a random init, identically "
              "distributed for alive and dead runs - only the LEARNED direction can "
              "carry history, so the factor living in the moved-dominated (low-alpha) "
              "regime is mechanistically coherent. The highest-alpha point rests on "
              "very few kills (suggestive only). The factor is real and UNNAMED: "
              "cos(dW,W0), summary spread, srank and module concentration are all "
              "ruled out (4.6; sliver surgery null).")
# Row unit: one scored run-curve crossing (lambda*K/K* units); ds1 cut 0.01."""))

cells.append(md(r"""## §4.6 Direction: a cautionary tale, and what survives

The direction program is the best example in this repo of a signal that *looked*
mechanistic and was killed properly:

1. cos(ΔW, W₀) carried fate information beyond size (`[adaptive]`: matched-dose AUC
   0.79; dying seeds more anti-aligned in 13/17 mixed cells, p = 0.025) — the
   program's first seed-level signal.
2. The **surgery** `[interv]`: flip/strip/10×-amplify the W₀-aligned sliver at exact
   matched norms — max |Δp| 0.007, 1/34 fate flips. **Causally inert.**
3. The **strip** `[cal]`: the sign-fate association does not replicate on calibrated
   support (29% dead vs 28% alive, against the adaptive-era 63%/25%).

Verdict: cos(ΔW, W₀) is fully retired — per the current `nors_asks` §1.2b wording,
"not a mechanism, and not even a robust marker"; seed-level results built on it are
flagged for re-examination as population artifacts (`seed_fate_direction.ipynb`
predates the strip and has NOT been updated — treat its ds1 headline as stale). What
survives of "direction does work": the history-dependent flip dose (§4.10) — real,
α-structured, and unnamed. The open axis is **subspace identity**: energy in W₀'s
top singular subspaces / activation-space placement, not any summary statistic yet
tried.

### A2 verdict

**Closed, with a revision** (the doc's wording): the causal dose is
**trajectory-like** (sustained exposure); the endpoint norm is its high-quality
proxy. The honest sentence: *dose sets death probability between cells and suffices
to flip fate; a direction-quality factor of ~2–4× (largest at low α) sets each
update's personal threshold.* The factor is real and **unnamed** — cos(ΔW,W₀),
summary spread, srank, and module concentration are all ruled out (§4.6, §4.10);
flip dose is also largely a *cell property* (dual-seed crossings correlate
+0.63–0.76).

Holes: (1) trajectory-vs-endpoint quantities are thin — the ~0.6×θ̂ operative
threshold is one number from 7 cells in one regime mix (the stage-3 deep-α clamp
extension and the ckpt-replay pilot are the successors); (2) **why crossing kills is
open** — no mechanistic account of why exceeding θ destroys the behavioral prior
rather than adding noise ("mechanism of the mechanism"; fine to leave open, framed
as open)."""))

# =====================================================================================
# Part A3 — the HPs set the size
# =====================================================================================

cells.append(md(r"""---
# Part A3 — The HPs set the size

**Claim (A3).** The realized update size is a smooth, near-exact power law in
(lr, α, r), and each exponent is traced to the update's two-term anatomy.
**Status: proven — the strongest link** (calibrated dose law R² ≈ 0.97; both
curvatures real and jointly identified, traced to the frozen/moved term crossover).

## §4.2 The dose law, and the collapse law as its level set"""))

cells.append(code(r"""import statsmodels.formula.api as smf
# Dose law on the strip (alive+dead, non-diverged, primary region), cluster by cell.
dl = sv[sv.s_scale <= 64].copy()
dl["lln"], dl["aln"], dl["rln"] = np.log(dl.lr), np.log(dl.alpha), np.log(dl["rank"])
m1 = smf.ols("np.log(dw_fro) ~ lln + aln + rln", dl).fit(
    cov_type="cluster", cov_kwds={"groups": dl.tag})
m2 = smf.ols("np.log(dw_fro) ~ lln + aln + rln + I(lln**2) + I(aln**2)", dl).fit(
    cov_type="cluster", cov_kwds={"groups": dl.tag})
t = pd.DataFrame({
    "linear fit": m1.params, "linear se": m1.bse,
    "with joint quadratics": m2.params, "quad se": m2.bse}).round(4)
print(
    f"Dose law on the strip (n={len(dl)} runs, {dl.tag.nunique()} cells, cluster-robust "
    f"by cell): log dw_fro ~ log lr + log alpha + log rank. Linear R^2 = {m1.rsquared:.3f}. "
    "Quadratics fitted JOINTLY (the adaptive grid's alpha-lr collinearity made a lone "
    "alpha^2 term unidentifiable - both are real here).")
display(compare_table(t))
a_lr, a_a, a_r = m1.params["lln"], m1.params["aln"], m1.params["rln"]
print(f"dose law:  dw_fro ~ lr^{a_lr:.3f} * alpha^{a_a:.3f} * r^{a_r:.3f}   (R2={m1.rsquared:.3f})")
print(f"lr^2 curvature {m2.params['I(lln ** 2)']:+.3f} (p={m2.pvalues['I(lln ** 2)']:.1e}), "
      f"alpha^2 {m2.params['I(aln ** 2)']:+.3f} (p={m2.pvalues['I(aln ** 2)']:.1e})")

print("\n--- level-set closure: collapse-law exponents predicted from the dose law ---")
closure = pd.DataFrame({
    "predicted (dose-law ratio)": {"b (global)": a_a / a_lr, "c (global)": a_r / a_lr},
    "measured (collapse fit)": {"b (global)": B_DESIGN, "c (global)": C_DESIGN},
}).round(3)
print(
    "Global version: b_pred = a_alpha/a_lr, c_pred = a_r/a_lr. The global ratio overshoots "
    "because the dose surface is curved; evaluated BOUNDARY-LOCALLY (quadratic surface "
    "slopes at the crossing - nors_asks 7.4 [recorded]) the closure is exact: "
    "L predicted 0.530 vs measured 0.499+/-0.064 (|z|=0.5); H predicted 0.882 vs measured "
    "0.882+/-0.098 (|z|=0.0). c closes via the measured threshold drift: "
    "a_r/a_lr + theta-drift -0.110 -> -0.084 (closure against the boundary exponent "
    f"needs re-adjudication against the designed-population fit c {cl.C_LAW:.3f}).")
display(compare_table(closure))"""))

cells.append(md(r"""**Why this matters.** The collapse law stops being a phenomenological fit: its
exponents are the *dose law's exponents corrected by the threshold drift*, and where
the dose surface bends (α², from term crossover — next section), the collapse
boundary bends by exactly the predicted amount (b_H predicted 0.882, measured
0.882 ± 0.098, |z| = 0.0 `[cal]`; the L segment closes at |z| = 0.5 boundary-locally
vs 2.0 for the naive global ratio). The closure is *not* yet exact on both segments:
the residual 1–2σ L-side tension originally indicted the one remaining derivation
assumption — θ constant in α. **Measured 2026-08-12 (strip nested logit + model-free
per-(α, r) thresholds): the training-time θ is flat in α (p = 0.20; slope
−0.017 ± 0.032) — the assumption is verified**, so the residual L tension stands as
a genuine (small) open puzzle rather than an unchecked assumption. (The *eval-time*
λ-crossing threshold does drift in α — slope −0.081 — but that is structure in the
history/hysteresis gap, a different object; `nors_asks` §5.5b.) Conditional on dose
the HPs contribute nothing extra to where the *cell-level* boundary sits (the
run-level endpoint residual is Part M's complication, §4.12). **K is a dose
scale.**

## §4.3 Where the dose law comes from: the update's two-term anatomy

A LoRA update is ΔW = s·B·A with s = α/r, B initialized to 0. Split A = A₀ + ΔA
(A₀ exactly replayable from the seed):

$$\Delta W \;=\; \underbrace{s\,B\,A_0}_{\text{“frozen” term}}
\;+\; \underbrace{s\,B\,\Delta A}_{\text{“moved” term}} \qquad\text{(exact identity)}$$

The two terms scale differently in lr — B is first-order in the optimizer's
updates, ΔA adds another order — so their *mixture* predicts curvature in the total
and a dominance crossover in α.

**What "straight vs convex" means, precisely.** Each term's run-level norm
(module-mean of s·‖B·A₀‖_F or s·‖B·ΔA‖_F) is fit by log-log OLS:

$$\ln\lVert\mathrm{term}\rVert = c + a\,\ln\mathrm{lr} + b\,\ln\alpha
+ g\,\ln r + q\,(\ln\mathrm{lr})^2 .$$

"Straight" = the quadratic coefficient q is compatible with 0 (frozen term:
q = −0.004 ± 0.002, R² = 0.997 without it); "convex" = q > 0 (moved term:
q = +0.064 ± 0.005). The sum's curvature (+0.092) is the moved term's bend plus the
dominance shift — nothing about any single term is quadratic in a mysterious way.

**Support**: the strip norm probes — the calibrated factorial's cells, ftseeds 1–2,
primary region s = α/r ≤ 64, non-diverged; exact QR-core norms, no ΔW
materialization. (A raw median-vs-lr line pooled over cells *bends* at high lr
because divergence exclusions change the cell composition there — that is a pooling
artifact, which is why the charts below show partial residuals from the fit, with
composition controlled, instead of pooled medians.)"""))

cells.append(code(r"""# Load the 8 per-rank strip norm probes -> run-level term norms.
mods = []
for r in (1, 2, 4, 8, 16, 32, 64, 128):
    p = AN / f"adapter_norms_strip_r{r}.json"
    if p.exists():
        mods.append(pd.DataFrame(json.loads(p.read_text())["results"]))
nm = pd.concat(mods, ignore_index=True)
nm["s"] = nm.alpha / nm["rank"]
for l in ("fro", "spec"):
    nm[f"frozen_{l}"] = nm.s * nm[f"B_Ainit_{l}"]
    nm[f"moved_{l}"] = nm.s * nm[f"B_dA_{l}"]
tp = nm.student_tag.str.extract(r"ep(?P<ep>\d+)_lr(?P<lr>[0-9.e-]+)_r\d+a\d+")
nm["lr"] = tp.lr.astype(float)
run_nm = (nm.groupby(["student_tag", "ftseed"], as_index=False)
            .agg(lr=("lr", "first"), rank=("rank", "first"), alpha=("alpha", "first"),
                 frozen_fro=("frozen_fro", "mean"), moved_fro=("moved_fro", "mean"),
                 frozen_spec=("frozen_spec", "mean"), moved_spec=("moved_spec", "mean"),
                 A_init=("A_init_fro", "mean"), A_final=("A_final_fro", "mean"),
                 B_fro=("B_fro", "mean")))
run_nm = run_nm.merge(strip[["tag", "seed", "test_p", "diverged"]],
                      left_on=["student_tag", "ftseed"], right_on=["tag", "seed"], how="left")
run_nm = run_nm[~run_nm.diverged.astype("boolean").fillna(False)]
run_nm = run_nm[(run_nm.alpha / run_nm["rank"]) <= 64]   # strip primary region
print(f"support: strip norm probes, primary region s<=64, non-diverged: "
      f"{len(run_nm)} runs / {run_nm.student_tag.nunique()} cells (ftseeds 1-2)")

fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2))
# (a) dominance ratio vs alpha.
ax = axes[0, 0]
dom = run_nm.assign(dom=run_nm.moved_fro / run_nm.frozen_fro)
for color, a in zip(plt.cm.viridis(np.linspace(0, .9, dom.alpha.nunique())),
                    sorted(dom.alpha.unique())):
    sub = dom[dom.alpha == a]
    ax.scatter(np.full(len(sub), a) * np.exp(np.random.RandomState(int(a)).uniform(-.06, .06, len(sub))),
               sub.dom, s=8, alpha=0.5, color=color)
med = dom.groupby("alpha").dom.median()
ax.plot(med.index, med.values, "k-o", ms=7, label="median")
ax.axhline(1, color="0.4", ls=":")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("alpha"); ax.set_ylabel("moved / frozen  (Frobenius)")
ax.set_title("(a) Term dominance is an alpha-dial:\nmoved-dominated at low alpha, "
             "frozen-dominated at high alpha")
ax.legend(fontsize=9)

# (b1)/(b2) straight-vs-convex, shown correctly: partial residuals of the log-log fit
# (alpha & rank components removed), so cell composition cannot bend the picture.
mfz = smf.ols("np.log(frozen_fro) ~ np.log(lr) + np.log(alpha) + np.log(rank)", run_nm).fit()
mfz_q = smf.ols("np.log(frozen_fro) ~ np.log(lr) + np.log(alpha) + np.log(rank) + I(np.log(lr)**2)",
                run_nm).fit()
mmv = smf.ols("np.log(moved_fro) ~ np.log(lr) + np.log(alpha) + np.log(rank) + I(np.log(lr)**2)",
              run_nm).fit()
lg = np.linspace(np.log(run_nm.lr.min()), np.log(run_nm.lr.max()), 100)
for ax, m, col, color, lab in [
        (axes[0, 1], mfz, "frozen_fro", "#E69F00", "frozen s·B·A0"),
        (axes[1, 0], mmv, "moved_fro", "#0072B2", "moved s·B·dA")]:
    cr = (m.params["np.log(lr)"] * np.log(run_nm.lr) + m.resid
          + m.params.get("I(np.log(lr) ** 2)", 0.0) * np.log(run_nm.lr) ** 2)
    ax.scatter(run_nm.lr, np.exp(cr), s=8, alpha=0.35, color=color)
    fit_line = m.params["np.log(lr)"] * lg + m.params.get("I(np.log(lr) ** 2)", 0.0) * lg**2
    ax.plot(np.exp(lg), np.exp(fit_line), color="k", lw=2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("lr"); ax.set_ylabel(f"{lab}: partial residual (alpha, r removed)")
    q = m.params.get("I(np.log(lr) ** 2)", np.nan)
    qse = m.bse.get("I(np.log(lr) ** 2)", np.nan)
    ax.set_title({"frozen_fro": f"(b1) frozen term IS straight once composition is "
                                f"controlled\nlr^{m.params['np.log(lr)']:.3f}, "
                                f"(ln lr)^2 = {mfz_q.params['I(np.log(lr) ** 2)']:+.3f}"
                                f"±{mfz_q.bse['I(np.log(lr) ** 2)']:.3f} (~0)",
                  "moved_fro": f"(b2) moved term is genuinely convex\n"
                               f"(ln lr)^2 = {q:+.3f}±{qse:.3f} > 0"}[col], fontsize=10)

# (c) A stays at init scale, B grows from zero - the bookkeeping fact.
ax = axes[1, 1]
g = run_nm.groupby("lr")[["A_init", "A_final", "B_fro"]].median()
ax.plot(g.index, g.A_final / g.A_init, "o-", color="#009E73", label="||A_final|| / ||A_init||", ms=7)
ax.plot(g.index, g.B_fro / g.A_init, "o-", color="#CC79A7", label="||B|| / ||A_init||", ms=7)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("lr"); ax.set_ylabel("ratio (median across runs)")
ax.set_title("(c) A stays at init scale; B grows from 0\n(bookkeeping that licenses the "
             "term split - see text)")
ax.legend(fontsize=9)
fig.tight_layout(); plt.show()
print(f"frozen-term fit: lr^{mfz.params['np.log(lr)']:.3f} (recorded 0.743+/-0.003, "
      f"= B's own lr exponent to 3 decimals) alpha/r exponents "
      f"{mfz.params['np.log(alpha)']:.3f}/{mfz.params['np.log(rank)']:.3f}")
# Row unit: one strip run in the primary region (module-mean norms), ftseeds 1-2."""))

cells.append(md(r"""**What panel (c) — the "asymmetric anatomy" — does and does not license.** B is
initialized to 0, so the trained update ΔW = s·B·A_f is proportional to B by
construction; A meanwhile barely leaves its init scale. Two consequences, and one
retraction to keep in view:

- *It licenses the term split as mechanically meaningful*: because A₀ is a fixed
  random matrix, the frozen term s·B·A₀ is a pure function of B's growth — which is
  why its fitted lr exponent equals B's own (0.743) to three decimals. The
  decomposition isn't an arbitrary algebraic split; its first term tracks one
  optimizer object.
- *It is bookkeeping, not mechanism*: "B does the learning" is a statement about
  where norm accumulates, conclusive as such (`[cal]`), but it does not by itself
  explain fate or transfer — and as evidence it is **superseded by the causal
  experiments**: every load-bearing mechanism claim now rests on §4.7's
  interventions, with the anatomy retained only as the scaffolding that makes the
  term split well-defined.
- *The stronger reading was retracted*: "B carries the **fate contrast**" was
  withdrawn in `nors_asks` — the dead/alive per-factor ratio gradient
  (A₀ ≈ 1 → A_f → ΔA → B) is heavily shaped by cell composition. Do not infer any
  fate localization from this panel.

**The chain that closes.** frozen term = pure power law (its lr exponent *is* B's);
moved term = genuinely convex (the extra lr order); their Frobenius dominance parity
sits at α ≈ 16–32 with full crossover by α ≈ 64–128 — exactly where the dose law's
α² curvature and the collapse law's regional bend (b_L → b_H) live. The regional
structure of the boundary is the two-term mixture made visible. The functional
counterpart `[interv]` — **both splice directions**, recomputed next — shows the
*trait* rides whichever term dominates: a double dissociation at the α extremes."""))

cells.append(md(r"""**Composition licensing [ident] — corrected 2026-08-13** (full gap distributions,
932 strip runs / 44,736 module rows; `[recorded — nors_asks §1.5]`). Tightness and
HP-constancy are different properties, and only *constancy* licenses
exponent-adding:

- The **frozen**-term Frobenius gap ‖B‖_F·‖A₀‖_F/‖B·A₀‖_F is loose but essentially
  **deterministic at √r** (median 2.821 at r8 vs √8 = 2.828; per-rank quantile
  spread ~±0.5%; HP drift lr^0.000 · α^−0.000 · r^0.491, residual ±0.9% — the
  near-isotropy of the Kaiming A₀) → factor-composition licensed for the frozen
  term in lr/α, with the √r carried explicitly.
- The **moved**-term Frobenius gap is **NOT HP-flat**: lr^0.042 · α^0.115 · r^0.184,
  R² 0.71, residual ±24% (IQR 1.72–2.24 at r8) → **no Frobenius
  factor-composition for the moved term**; its exponents are **measured directly on
  ‖B·ΔA‖** — lr^1.302±0.012 · α^0.498±0.017 · r^−0.107±0.012 (R² 0.976, convex:
  lr² +0.064±0.005) — never composed. The frozen term's direct fit is
  lr^0.739±0.003 · α^0.853±0.004 · r^−0.461±0.003 (R² 0.997, straight in lr).
- The *spectral* bounds are the tight ones (frozen ≤ 1.28 at r128; moved median
  ~1.05) — tight, but tightness was never the requirement.

The crossover explains both curvatures of the total on both axes: the moved term has
the higher lr exponent (takes over at high lr → lr² positive) and the frozen term
the higher α exponent (takes over at high α → α² positive) — the total's exponents
(1.05, 0.69, −0.24) sit between the terms'. Dominance as an α-dial replicates on all
6 adaptive arms (quad/cross 3.1–4.8× at α ≤ 8 vs 0.57–0.77 at α ≥ 32)
`[adaptive, replication]`.

The functional counterpart `[interv]` — **both splice directions**, recomputed next
— shows the *trait* rides whichever term dominates: a double dissociation at the α
extremes."""))

cells.append(code(r"""# Term splices, BOTH directions [interv], support = the ADAPTIVE ep20 grid of the
# focus ds1 lineage, ftseed1 (197 cells; the strip cells were never spliced - the two
# full-grid JSONs predate it). keep-frozen = remove the moved term (splice A -> A0 at
# exact per-module norms); keep-moved = the mirrored splice. Metric = EXCESS-over-base
# retention (p_spliced - base)/(p_orig - base) - never the raw p ratio, because a
# splice moves p toward the base prior 0.0133, not toward 0. Cells with orig excess
# < 0.005 are masked (ratio of two near-zeros).
def load_splice(name):
    d = pd.DataFrame(json.loads((AN / name).read_text())["results"])
    tp = pd.DataFrame([parse_param_tag(t) for t in d.student_tag], index=d.index)
    d["lr"], d["rank"], d["alpha"] = tp.lr, tp["rank"], tp.alpha
    d = d[d.p_target_orig > 0.01].copy()          # alive cells only
    d["retention"] = (d.p_target_ablated - BASE_PRIOR) / (d.p_target_orig - BASE_PRIOR)
    d.loc[(d.p_target_orig - BASE_PRIOR) < 0.005, "retention"] = np.nan
    return d
keepF = load_splice("frozen_a_ablation_ep20_full.json")        # remove moved -> frozen term only
keepM = load_splice("frozen_a_ablation_moved_ep20_full.json")  # remove frozen -> moved term only
print(f"support: adaptive ep20 grid, ftseed1, alive cells with usable excess: "
      f"keep-frozen {keepF.retention.notna().sum()}, keep-moved {keepM.retention.notna().sum()}")

fig, ax = plt.subplots(figsize=(9, 4.4))
for d, color, lab in [(keepF, "#E69F00", "keep FROZEN term (moved removed)"),
                      (keepM, "#0072B2", "keep MOVED term (frozen removed)")]:
    med = d.groupby("alpha").retention.median().dropna()
    q1 = d.groupby("alpha").retention.quantile(0.25)
    q3 = d.groupby("alpha").retention.quantile(0.75)
    ax.plot(med.index, med.values, "o-", color=color, ms=8, lw=2, label=lab)
    ax.fill_between(med.index, q1.reindex(med.index), q3.reindex(med.index),
                    color=color, alpha=0.15)
ax.axhline(0, color="0.4", ls=":"); ax.axhline(1, color="0.4", ls=":")
ax.set_xscale("log"); ax.set_xlabel("alpha")
ax.set_ylabel("excess-over-base trait retention")
ax.set_title("Both splice directions: the trait rides whichever term DOMINATES\n"
             "(median with IQR band; crossing at alpha ~ 16-32 = the Frobenius "
             "dominance-parity point)")
ax.legend(fontsize=9, loc="center right")
fig.tight_layout(); plt.show()

both = (keepF.groupby("alpha").retention.median().rename("keep_frozen").to_frame()
        .join(keepM.groupby("alpha").retention.median().rename("keep_moved")))
both["sum"] = both.keep_frozen + both.keep_moved
print("median retention by alpha (sum != 1 -> the terms are NOT additive in function space):")
print(both.round(3).to_string())
print("\nreadings: alpha<=4 - keep-moved retains ~0.6-0.8, keep-frozen erases to base;")
print("alpha>=256 - keep-frozen retains ~0.85-1.1, keep-moved ~0 (double dissociation).")
print("alpha=128 (near-boundary): keep-frozen goes NEGATIVE and the sum is ~-0.2 - the")
print("terms interact nonlinearly near the cliff; weight decomposition is only")
print("approximately compositional in function space there.")
# Row unit: one alive ep20 cell (focus ds1, ftseed1), per splice direction."""))

cells.append(md(r"""**Cross-arm replication [adaptive]** — the honest version (`collapse_mechanism` §6f,
per-arm alive-cell fits of log dw_fro, `[recorded]`): the six arms give a
**consistent band** (lr 0.95–1.13, α 0.51–0.70, r −0.10..−0.28; implied b
0.54–0.62, implied c −0.11..−0.26), *not* "exactly the same law" — ws2 is weak (34
cells, R² 0.33, consistent-only) and ds3/ws3 are thin (46/20 cells). The precise
statement lives on ds1 and the strip; the other arms replicate sign and rough
magnitude. Removing the s = α/r prefactor from the strip dose law gives the learned
factor ‖BA‖/‖W₀‖ ∝ lr^1.05 · α^−0.31 · r^+0.76 `[cal]` — Adam partially compensates
the s-amplification and rank adds coherent channels; the adaptive version
(α^−0.34 · r^+0.78) agrees.

**A3 holes:** scope only (single width / pool / budget / optimizer / schedule —
shared with A1). The dose law has never been refit per animal — it should be nearly
animal-free by construction (the trait enters only through the numbers
distribution), and the adapters exist on disk for a cheap check (doc Open
Questions #4)."""))

# =====================================================================================
# Part M — the mediation
# =====================================================================================

cells.append(md(r"""---
# Part M — The mediation: the HPs cause fate *through* the dose

**Status: supported** — by level-set consistency (§4.2), mediator-blocking (the
clamp, §4.9), and mediator-pushing (λ, §4.7); one statistical residual remains,
attributable to trajectory-vs-endpoint.

**The claim — two statements, one about cells and one about individual runs. They
sound similar and are not.**

*About cells:* if you know how big an update each HP setting produces, you can draw
the dividing line between dying and surviving settings without ever looking at a
fate — the fate boundary sits exactly where the update size crosses one threshold.
This is supported (the checks are §4.2's regional closures and §4.11 below).

*About individual runs:* take two runs whose final updates are the **same size**,
one trained at lr 2e-4 and one at 5e-4. If the final size were the whole story they
would die at the same rate — they don't; the higher-lr run dies slightly more often
(p ≈ 1e-3, §4.12). So the final size is not the complete story for a single run.
The chain's position: the higher-lr run spent more of training with a large update
even though it ended at the same place, and the clamp showed that sustained
exposure, not the endpoint, is what kills. The final size is a summary of the path —
a very good one (AUC 0.99), but lossy, and the small run-level residual is exactly
what that lossiness looks like. Both statements can hold at once: the boundary
between cells sits on the size threshold, while among individual runs at matched
*final* size the path still matters a little.

## §4.11 Why the two *global* fits need not agree — box average vs contour average, and the null-model check

A natural-looking test of mediation is to fit the dose law (OLS) and the collapse
law (logistic) globally on the same cells and compare exponent ratios. **That test
is not valid — the two fits estimate different mathematical objects.** The OLS of
log dose on the log-HPs returns the **box-averaged gradient** of the dose surface
(on our symmetric grid, its slope at the grid center). The logistic fit of
dead/alive gets essentially all of its coefficient information from runs near the
boundary — deep-alive and deep-dead runs only need to be on the correct side — so it
returns the **contour-averaged gradient**, the surface's tilt along the death
boundary itself. These agree exactly only when the gradient's *direction* is
constant (a plane, or a ridge f = g(βᵀx) with parallel contours). A toy with no
noise at all: f(x₁,x₂) = x₁ + x₂ + 0.3·x₁² on [−1,1]², exact threshold at 0.5 — the
OLS slope ratio is 1.00, but the local ratio runs 1.50 at one end of the death
contour and 0.62 at the other, and a logistic returns some design-weighted average
in between. One phenomenon, one box, a literal threshold — three different numbers,
none wrong. Our surface is the toy twice over (lr² and α² curvature from the term
crossover, and the boundary runs diagonally through the box), which is why the
*regional* closures in §4.2 are the like-for-like test (L |z| = 0.5, H |z| = 0.0).

**The null-model check below computes what the global gap *should* be instead of
arguing about it.** Keep every run's real measured dose; assign "dead" purely by
dose ≥ one constant, chosen only to match the total death count (no per-run fate
information is used); run the identical global logistic on those labels. The
synthetic labels feed no result — they are the threshold-world's exact prediction
for this estimator on these cells, like simulating a fair coin to judge an observed
head count. A second variant gives the threshold its measured rank drift
(θ ∝ r^−0.110, Part A4)."""))

cells.append(code(r"""import statsmodels.formula.api as smf
# Threshold-world null model (following nors_asks 7.2b): the strip's primary region,
# diverged excluded. b/c are read as COEFFICIENT RATIOS (near-separable data - raw
# logit coefficients diverge, only ratios are identified).
sv2 = strip[strip.test_p.notna() & ~strip.diverged.astype(bool) & strip.dw_fro.notna()
            & (strip.s_scale <= 64)].copy()
sv2["dead_f"] = sv2.dead.astype(float)
sv2["lln"], sv2["aln"], sv2["rln"] = (np.log(sv2.lr), np.log(sv2.alpha),
                                      np.log(sv2["rank"]))
n_dead2 = int(sv2.dead_f.sum())
print(f"population: {len(sv2)} non-diverged strip runs (s<=64), {n_dead2} dead")

o2 = smf.ols("np.log(dw_fro) ~ lln + aln + rln", sv2).fit(
    cov_type="cluster", cov_kwds={"groups": sv2.tag})
mr2 = smf.logit("dead_f ~ lln + aln + rln", sv2).fit(disp=0, maxiter=500)
th2 = sv2.dw_fro.sort_values(ascending=False).iloc[n_dead2 - 1]
sv2["dead_sim"] = (sv2.dw_fro >= th2).astype(float)
ms2 = smf.logit("dead_sim ~ lln + aln + rln", sv2).fit(disp=0, maxiter=500)
sv2["dw_adj"] = sv2.dw_fro * sv2["rank"].astype(float) ** 0.110
thr2 = sv2.dw_adj.sort_values(ascending=False).iloc[n_dead2 - 1]
sv2["dead_simr"] = (sv2.dw_adj >= thr2).astype(float)
msr2 = smf.logit("dead_simr ~ lln + aln + rln", sv2).fit(disp=0, maxiter=500)

def _ratios(m):
    return m.params["aln"] / m.params["lln"], m.params["rln"] / m.params["lln"]

agree_sim = (sv2.dead_sim == sv2.dead_f).mean()
agree_simr = (sv2.dead_simr == sv2.dead_f).mean()
rows7 = [
    ("OLS dose-law global ratio", o2.params["aln"] / o2.params["lln"],
     o2.params["rln"] / o2.params["lln"], np.nan),
    ("logistic, REAL fates", *_ratios(mr2), np.nan),
    ("logistic, threshold-on-dose sim", *_ratios(ms2), agree_sim),
    ("logistic, sim + measured theta(r) drift", *_ratios(msr2), agree_simr),
]
t7n = pd.DataFrame(rows7, columns=["fit", "b = coef_a/coef_lr", "c = coef_r/coef_lr",
                                   "label agreement w/ real"]).set_index("fit")
print("Global fits on the identical runs (sim threshold chosen ONLY to match the "
      "total death count):")
display(compare_table(t7n, fmt="{:.3f}"))
print(f"one threshold on measured dose reproduces {agree_sim:.1%} of the actual "
      f"fate labels; with the measured rank drift, {agree_simr:.1%}")

seg_res = {}
for nm, msk in (("global", np.ones(len(sv2), bool)),
                ("L (a<=32)", (sv2.alpha <= 32).to_numpy()),
                ("H (a>=64)", (sv2.alpha >= 64).to_numpy())):
    g7 = sv2[msk]
    bo7 = (lambda f: f.params["aln"] / f.params["lln"])(
        smf.ols("np.log(dw_fro) ~ lln + aln + rln", g7).fit())
    br7 = _ratios(smf.logit("dead_f ~ lln + aln + rln", g7).fit(disp=0, maxiter=500))[0]
    bs7 = _ratios(smf.logit("dead_sim ~ lln + aln + rln", g7).fit(disp=0, maxiter=500))[0]
    seg_res[nm] = (bo7, br7, bs7)
    print(f"{nm}: OLS ratio {bo7:.3f} | real-fate logit {br7:.3f} | "
          f"threshold-sim logit {bs7:.3f}")

fig, ax = plt.subplots(figsize=(8.6, 4.4))
xs = np.arange(len(seg_res))
for lab, j, mk, colr in (("OLS dose-law ratio", 0, "s", "0.45"),
                         ("logistic, real fates", 1, "o", C_ALIVE),
                         ("logistic, threshold-world sim", 2, "D", C_DEAD)):
    ax.scatter(xs + (j - 1) * 0.14, [v[j] for v in seg_res.values()], s=90,
               marker=mk, color=colr, label=lab, zorder=3)
b_drift = _ratios(msr2)[0]
ax.scatter([0.28], [b_drift], s=90, marker="^", color=C_FRO,
           label="sim + theta(r) drift (global)", zorder=3)
ax.annotate(f"labels {agree_sim:.1%}", xy=(0.14, seg_res["global"][2]),
            xytext=(0, -18), textcoords="offset points", ha="center",
            fontsize=9, color=C_DEAD)
ax.annotate(f"{agree_simr:.1%}", xy=(0.28, b_drift), xytext=(0, 12),
            textcoords="offset points", ha="center", fontsize=9, color=C_FRO)
ax.set_xticks(xs, list(seg_res))
ax.set_ylabel("b = coef_alpha / coef_lr")
ax.set_title("The threshold world predicts the real global logits:\n"
             "the OLS-vs-logistic gap is estimator geometry (box vs contour "
             "average),\nnot a second phenomenon")
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout(); plt.show()
print("READ: no residual gap survives (cell-bootstrap CIs on the differences all "
      "include zero [recorded - nors_asks 7.2b: global -0.057 [-0.161, +0.067]]). "
      "One honest footnote: real b sits slightly BELOW threshold-world b in every "
      "segment - the direction of 'higher lr slightly deadlier than its endpoint "
      "dose implies' (the 4.12 residual) - but not distinguishable from noise in "
      "the boundary orientation at this n. The theta(r)-drift variant also "
      "reconciles the c column (closer to the real c). Corollary: the GLOBAL "
      "comparison is a weak test in both directions; the discriminating power "
      "lives in the regional closures (4.2) and the label agreement above.")
# Row unit: one strip run, primary region s<=64, non-diverged; ds1 0.01 cut."""))

cells.append(md(r"""## §4.12 Endpoint mediation, refit on the strip — the nested logit, both norms

Does the measured **endpoint** dose carry *all* the HP information about fate? Fit
dead ~ log dw as the sole mediator, then add the log-HPs and test the block
(cluster-robust Wald by cell). If the endpoint fully mediated, the HP block would be
null. This doubles as Part A4's mediation-asymmetry leg — compare the two norms."""))

cells.append(code(r"""# Nested logit, both norms (following nors_asks 7.3), cluster-robust Wald by cell.
from scipy.stats import chi2 as _chi2

def _cluster_wald(model, dat, names, y):
    X_ = model.model.exog
    mu_ = model.predict()
    u_ = (y - mu_)[:, None] * X_
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
dat_y = sv2.dead_f.to_numpy()
for med, lab in (("ldw", "dw_fro"), ("lds", "dw_spec")):
    m0_ = smf.logit(f"dead_f ~ {med}", sv2).fit(disp=0, maxiter=300)
    m1_ = smf.logit(f"dead_f ~ {med} + lln + aln + rln", sv2).fit(disp=0, maxiter=300)
    wst, wp = _cluster_wald(m1_, sv2, ["lln", "aln", "rln"], dat_y)
    print("mediator %s: mediator-only pseudo-R2 %.3f | +HP %.3f | residual HP "
          "coefs lr %+.2f a %+.2f r %+.2f | cluster Wald on HP block: p = %.1e"
          % (lab, m0_.prsquared, m1_.prsquared, m1_.params["lln"],
             m1_.params["aln"], m1_.params["rln"], wp))
print("READ: the mediator-strength asymmetry is calibrated - dw_fro absorbs far "
      "more of the HP->fate signal than dw_spec (pseudo-R2, and residual "
      "alpha-loading ~6x smaller) - but BOTH norms leave a small significant "
      "residual: at matched endpoint dose, higher-lr runs die slightly more. "
      "NEITHER is a complete ENDPOINT mediator; the clamp (4.9) says why - the "
      "causal dose is trajectory exposure, and the endpoint is its lossy summary. "
      "Coefficient magnitudes are seed-unstable in this near-separation regime - "
      "read signs, not sizes. [The adaptive-era p-value gulf (4e-3 vs 3e-10) was "
      "population-specific; do not quote it as the headline.]")
# Row unit: one strip run, primary region, non-diverged; cluster = cell."""))

cells.append(md(r"""**Blocking and pushing the mediator** — the interventional legs of M, recomputed in
Part A2: the clamp (§4.9) holds HPs at verified-lethal values and caps the dose →
survival is monotone in the ceiling, 14/14 at 0.5×θ̂ — the textbook mediation
intervention. λ-scaling (§4.7) pushes the mediator without touching the HPs and
produces the effect in both directions (inflation kills gentle-HP runs, deflation
revives lethal-HP runs).

**The one complication (state it; do not hide it).** The nested logit above leaves
an HP residual at p ≈ 1e-3 `[cal]`: at matched endpoint dw, higher-lr runs die more
— the endpoint norm does not fully mediate statistically. The rank loading matches
the θ_F rank drift; magnitudes are seed-unstable, so read the signs. Two readings,
not mutually exclusive: (a) the residual is the **endpoint-vs-trajectory gap**
(higher lr rides nearer the ceiling en route to the same endpoint — exactly what
the clamp showed matters); (b) it is the direction factor, for which cos(ΔW,W₀) is
a *marker* (its surgery was null). On the adaptive population, adding cos(ΔW,W₀)
made the HP block null (Wald p = 0.51) — (size, direction) jointly screen off
(lr, α, r) `[adaptive]`; cos is not computed on the strip (a disk-only dw-sweep
pass would add it). Reading (a) is an inference, not a result — the ckpt-replay
pilot (2026-08-13) produces the raw material to test it directly."""))

# =====================================================================================
# Part A4 — Frobenius, not spectral
# =====================================================================================

cells.append(md(r"""---
# Part A4 — Frobenius, not spectral

**Claim (A4).** The dose the fate mechanism reads is total energy ‖ΔW‖_F, not the
peak σ_max(ΔW). **Status: identified, not causally isolated for fate** — threshold
geometry + mediation asymmetry + matched-energy surgery on the damage symptom all
favor F; no intervention dissociates F from σ on collapse-fate itself; single-width
caveat. This is the chain's weakest link; say so.

## §4.1b Leg 1 — threshold geometry **[cal]** (Test 1)

If fate thresholds the *true* dose D, then θ measured in D is one number regardless
of incidental parameterization; measured in a wrong norm related to D by a
rank-dependent factor (σ = F/√srank, and srank varies with rank), the apparent
threshold must drift with rank. This is an **identification criterion, not a causal
proof** — and θ_F is the *flattest* threshold, not flat (its own drift is real and
is absorbed into c)."""))

cells.append(code(r"""# theta(r): per-rank dose threshold, Frobenius vs spectral (dead-side edge of the
# accuracy-maximizing split — the house Test-1 estimator), on the strip runs.
def theta_by_rank(df, col):
    out = {}
    for r, sub in df.groupby("rank"):
        if sub.dead.nunique() < 2:
            continue
        x = np.sort(sub[col].unique())
        best_acc, best_theta = -1, np.nan
        for t in x:
            acc = ((sub[col] >= t) == sub.dead).mean()
            if acc > best_acc:
                best_acc, best_theta = acc, sub.loc[sub[col] >= t, col].min()
        out[r] = best_theta
    return pd.Series(out)

fig, ax = plt.subplots(figsize=(7.2, 4.4))
for col, color, lab in [("dw_fro", C_FRO, "Frobenius"), ("dw_spec", C_SPEC, "spectral")]:
    th = theta_by_rank(sv, col)
    ax.plot(th.index, th.values, "o-", color=color, ms=8, label=lab)
    sl = np.polyfit(np.log(th.index), np.log(th.values), 1)[0]
    ax.annotate(f"slope {sl:+.3f}", xy=(th.index[-1], th.values[-1]),
                xytext=(5, 0), textcoords="offset points", color=color, fontsize=10)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("rank"); ax.set_ylabel("death threshold theta(r) on the dose")
ax.set_title("Test 1: whose threshold is rank-stable?\nFrobenius drifts less "
             "(recorded WLS: theta_F -0.110 vs theta_2 -0.172, paired diff excludes 0)")
ax.legend(fontsize=9)
fig.tight_layout(); plt.show()
# Row unit: one rank; strip runs, fates at the ds1 0.01 cut."""))

cells.append(md(r"""**Reading (Nor's audit, condensed).** The audit pre-registered a standard: the right
norm is the one whose death threshold is stable across incidental parameterization
(rank) and whose value predicts realized activation movement. Frobenius won both
`[cal]`: θ_F drifts at −0.110 [−0.124, −0.066] vs θ_2 −0.172 [−0.230, −0.117]
(bootstrap; the paired difference [−0.142, −0.017] excludes zero — and note θ_F's
own drift is also real: Frobenius is the *flattest* threshold, not a perfectly
invariant one, and c absorbs its drift). Within-rank fate AUCs saturate ≥ 0.99 for
both norms on the strip and are retired — the verdict rests on the threshold slopes.
Note also: "HP-agnostic threshold" identifies which norm the mechanism *plausibly
reads*; it cannot rule out a third variable proportional to F across this design.

## §4.12b Leg 2 — mediation asymmetry **[cal]**

Recomputed in Part M (§4.12): as the sole endpoint mediator of fate, dw_fro achieves
pseudo-R² ~0.925 vs dw_spec ~0.871, and the residual α coefficient left by spec is
~6× the one left by fro — the spectral norm misses exactly the α-shaped part of the
dose signal. Both norms leave a small detectable HP residual (p ≈ 1e-3 vs ~2e-3), so
the honest wording is: **F absorbs far more of the HP→fate signal than σ; neither is
a complete endpoint mediator.** The adaptive-era p-value gulf (4e-3 vs 3e-10) was
population-specific — do not quote it as the headline.

## §4.7b Leg 3 — matched-energy surgery on the damage symptom **[interv]**

In the (energy, spread) basis, spread at fixed energy *increases* damage (γ_S on KL
+0.51 [+0.40, +0.65]; positive within every rank), where the peak theory requires
γ_S < 0. Interventionally: concentrate the update onto its top direction at exactly
matched ‖·‖_F — which *raises* σ_max by √srank — and damage **falls**. Below, the
original 8-cell surgery and its depth profile (kept as the locus-of-why picture);
its counts ("7/8", "flat non-monotone in rank") are **superseded by the 224-set
shell analysis** recomputed in §4.13, which also surfaces the carrier-deletion
split."""))

cells.append(code(r"""# Spread surgery: top1 (all energy in the top direction) vs flat (equalized spectrum)
# at matched ||dW||_F, 8 cells. Plus the depth profile of the same splices.
ss_rows = []
for f in sorted(AN.glob("spread_surgery_*.json")):
    d = json.loads(f.read_text())
    for r in d["results"]:
        ss_rows.append(dict(tag=r["student_tag"], seed=r["ftseed"], fate=r["fate"],
                            variant=r["variant"], p=r["p_target"], kl=r["pile_kl_base"]))
ss = pd.DataFrame(ss_rows)
piv = ss.pivot_table(index=["tag", "seed", "fate"], columns="variant",
                     values=["kl", "p"]).round(4)
print(
    "Spread surgery [interv]: Pile KL vs base (nats) per cell x variant, at exactly matched "
    "Frobenius energy. top1 = all energy concentrated on the top direction (RAISES the "
    "spectral norm by sqrt(srank)); flat = spectrum equalized. Row = one trained adapter.")
display(compare_table(piv["kl"]))
kl_delta = (piv["kl"]["top1"] - piv["kl"]["orig"])
print(f"top1 vs orig: damage reduced in {(kl_delta < 0).sum()}/{len(kl_delta)} cells "
      f"(median {kl_delta.median():+.2f} nats) despite the LARGER peak ->")
print("the peak account is refuted twice over: bigger peak + same energy = LESS damage.")
print("flat is non-monotone in rank (kills the r8a1 trait, but reduces damage at r64):")
print("summary spread is not the beyond-dose axis either (matched-pair check rejected it).")

# Depth profile of the same splices.
dp = json.loads((AN / "spread_depth_probe.json").read_text())["results"]
dp = pd.DataFrame(dp)
if {"variant", "layer"}.issubset(dp.columns):
    drift_col = [c for c in ("rel_drift", "drift") if c in dp.columns][0]
    ratio = (dp.pivot_table(index=["student_tag", "layer"], columns="variant", values=drift_col)
               .eval("top1 / flat").rename("ratio").reset_index())
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for tag, sub in ratio.groupby("student_tag"):
        ax.plot(sub.layer, sub.ratio, color="0.75", lw=1)
    medr = ratio.groupby("layer").ratio.median()
    ax.plot(medr.index, medr.values, "o-", color="#0072B2", lw=2.2, ms=7, label="median")
    ax.axhline(1, color="0.4", ls=":")
    ax.set_xlabel("layer"); ax.set_ylabel("drift ratio  top1 / flat")
    ax.set_title("Depth profile of matched-energy splices: instant filtering (<1 at layer 1),\n"
                 "coherent mid-network recovery, terminal normalization at the output")
    ax.legend(fontsize=9)
    fig.tight_layout(); plt.show()
else:
    print("depth probe JSON schema differs; recorded result: median top1/flat drift ratio "
          "0.86 (layer 1) -> 1.12 (layer 8) -> 0.82 (output); final ratio < 1 in 6/8 cells.")
# Row unit: one trained adapter (table); one (adapter, layer) (chart)."""))

cells.append(md(r"""**The mechanism picture after the interventions.** The network *filters a peaked
injection immediately* (within the first block), partially recovers it through
coherent same-direction accumulation, and absorbs it again at the output — which is
*why* total energy beats the peak as the damage variable: a concentrated update is
locally louder but propagates less. Together: **dose (Frobenius energy) is the
causal size variable for the damage syndrome; the residual, history-dependent factor
is which subspace the energy occupies** — with summary statistics of the spectrum
(spread, srank, module concentration) and the W₀-projection all ruled out as that
factor.

## §4.13 Two kinds of death: the carrier-deletion split **[interv]** (shell scale)

The fate-level objection to A4, and its resolution. At scale, top1 *does* kill the
trait in a minority of alive sets — naively contradicting "F is the fate dose"
(both reshapings hold F exactly fixed). The resolution: **the binary DEAD label
conflates two mechanistically distinct events**, and the reshaping surgeries flip
fate mostly via the second — **collapse-death** (containment failure, ~15-nat KL)
vs **carrier-deletion death** (the trait rides the discarded non-dominant
directions: truncation removes the *content* while displacing generic behavior
*less*, so Pile KL **falls**). The Frobenius/peak question concerns only the first.
This split must be a first-class sentence in any writeup — a reviewer will find the
kills."""))

cells.append(code(r"""# Sweep-scale spread splices (ladder + shell extension, both seeds), following
# nors_asks 5.5b: per set, orig/top1/flat at exactly matched per-module ||dW||_F.
SPREAD_PATTERNS = ["spread_ladder_ep20_*.json", "spread_ladder_s2_ep20_*.json",
                   "spread_shellext_ep20_*.json", "spread_shellext_s1_ep20_*.json",
                   "spread_shellext_s2_ep20_*.json"]
sp_files = [f for pat in SPREAD_PATTERNS for f in sorted(AN.glob(pat))]
if not sp_files:
    print("sweep-scale spread splice results not on disk — skipped")
else:
    sp_rows = []
    for f in sp_files:
        d = json.loads(f.read_text())
        pivv = {}
        for r in d["results"]:
            pivv.setdefault(r["ftseed"], {})[r["variant"]] = r
        for seed, pv in pivv.items():
            if "orig" not in pv or "top1" not in pv or "flat" not in pv:
                continue
            tag = d["tag"]
            sp_rows.append({"cell": tag, "seed": seed,
                            "rank": int(re.search(r"_r(\d+)a", tag)[1]),
                            "alpha": int(re.search(r"a(\d+)$", tag)[1]),
                            "kl_orig": pv["orig"]["pile_kl_base"],
                            "dkl_top1": pv["top1"]["pile_kl_base"] - pv["orig"]["pile_kl_base"],
                            "dkl_flat": pv["flat"]["pile_kl_base"] - pv["orig"]["pile_kl_base"],
                            "p_orig": pv["orig"]["p_target"],
                            "p_top1": pv["top1"]["p_target"],
                            "p_flat": pv["flat"]["p_target"]})
    slx = pd.DataFrame(sp_rows).drop_duplicates(["cell", "seed"])
    print(f"spread splices at sweep scale: {slx.cell.nunique()} cells, "
          f"{len(slx)} splice sets")
    for v in ("top1", "flat"):
        dd = slx[f"dkl_{v}"]
        print(f"  {v} reduces Pile KL in {(dd < 0).sum()}/{len(dd)} sets "
              f"(median {dd.median():+.2f} nats) - supersedes the 8-cell counts above")
    alv = slx[slx.p_orig >= 0.01]
    ded = slx[slx.p_orig < 0.01]
    kk = alv[alv.p_top1 < 0.01]              # top1 fate-kills among alive sets
    rv = ded[ded.p_flat >= 0.01]             # flat revivals among dead sets
    nrv = ded[ded.p_flat < 0.01]
    print(f"\nFATE TRANSITIONS at constant energy: top1 kills {len(kk)}/{len(alv)} "
          f"alive sets (flat: {(alv.p_flat < 0.01).sum()}/{len(alv)}); "
          f"flat revives {len(rv)}/{len(ded)} dead sets")
    if len(kk):
        print(f"  top1 kills: Pile KL FALLS in {(kk.dkl_top1 < 0).sum()}/{len(kk)} "
              f"of them (median {kk.dkl_top1.median():+.2f} nats); killed sets are "
              f"boundary-adjacent low-p (p_orig median {kk.p_orig.median():.3f} vs "
              f"{alv.p_orig.median():.3f} overall) -> CARRIER-DELETION deaths, not "
              "collapse: generic behavior is displaced LESS, not more")
    if len(rv):
        print(f"  flat revivals: KL falls in {(rv.dkl_flat < 0).sum()}/{len(rv)} "
              f"(median {rv.dkl_flat.median():+.2f}); revivable dead are the "
              f"SHALLOW-damage dead (orig KL median {rv.kl_orig.median():.1f} vs "
              f"{nrv.kl_orig.median():.1f} nats for non-revived) -> damage-dilution")

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.5))
    ax = axes[0]
    surv = alv[alv.p_top1 >= 0.01]
    ax.scatter(surv.p_orig, surv.dkl_top1, s=16, color="0.6", alpha=0.5,
               label=f"alive set, top1 keeps trait (n={len(surv)})")
    ax.scatter(kk.p_orig, kk.dkl_top1, s=64, color=C_DEAD, marker="v",
               label=f"alive set, top1 KILLS trait (n={len(kk)})")
    ax.axhline(0, color="0.4", ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("original test p(dog) of the set")
    ax.set_ylabel("dKL(top1 - orig), nats")
    ax.set_title("top1 fate-kills sit at LOW p_orig with FALLING Pile KL:\n"
                 "the trait is deleted with its carrier directions, not collapsed")
    ax.legend(fontsize=8, loc="upper right")
    ax = axes[1]
    ax.scatter(nrv.kl_orig, nrv.dkl_flat, s=16, color="0.6", alpha=0.5,
               label=f"dead set, flat stays dead (n={len(nrv)})")
    ax.scatter(rv.kl_orig, rv.dkl_flat, s=64, color=C_ALIVE, marker="^",
               label=f"dead set, flat REVIVES trait (n={len(rv)})")
    ax.axhline(0, color="0.4", ls=":")
    ax.set_xlabel("original Pile KL vs base (nats)")
    ax.set_ylabel("dKL(flat - orig), nats")
    ax.set_title("the mirror image: flat revives only SHALLOW-damage dead,\n"
                 "always with falling KL (damage-dilution)")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout(); plt.show()
# Row unit: one (cell, seed) splice set; ds1 0.01 cut throughout."""))

cells.append(md(r"""**A4 holes (the weakest link; the doc's list, condensed):**

1. **No intervention dissociates F from σ on collapse-fate itself.** λ scales both
   norms together (silent on F-vs-σ); the σ-clamp arm physically drags F with it
   (trained updates are low-srank; spec-clamped runs realize F at 0.67–1.50×θ̂_F) —
   training-time norm attribution is impossible with this design; and the
   surgeries' fate transitions are dominated by carrier-deletion (§4.13). A4's
   causal leg is really about the *damage syndrome* (KL); A4-for-fate rides on
   legs 1–2 plus the sibling-symptom bridge (doc Open Questions #7).
2. **"HP-agnostic threshold" is not a causal argument** — it identifies which norm
   the mechanism plausibly reads; it cannot rule out a third variable proportional
   to F across this design.
3. **Single width** (d = 768). Adam's per-entry step normalization ties norm
   scalings to parameter counts — the F-vs-σ verdict could in principle be
   width-specific (Nor's objection). One boundary-band replication at pythia-410m
   retires it; otherwise every norm claim carries an explicit "at this width" scope
   line.
4. **flat's spread test is off-support at r ≥ 16** (it pushes srank → r vs observed
   1–6.2); γ_S was always a local claim."""))

# =====================================================================================
# Part VI — chain status, ledger, blueprint
# =====================================================================================

cells.append(md(r"""---
# Part VI — Chain status, evidence ledger, blueprint

## §5.0 The chain, audited (supersedes the previous revision's §4.8 audit)

The up-to-date statement of the chain — claims, statuses, holes, retired results,
and open questions — is **`COLLAPSE_CLAIM_CHAIN.md` at the repo root**; this section
mirrors its status table and does not restate the argument. Statuses (2026-08-13):

| # | claim | status |
|---|---|---|
| **A1** | phenomenon exists, HP-governed, not SL-specific | **proven at calibrated strength**; the trait-free leg rests on a thin 22-cell control |
| **A2** | the update's size causes fate | **closed, with a revision**: the causal dose is trajectory-like (sustained exposure), the endpoint norm is its high-quality proxy; a bounded ~2–4× direction factor sets each update's personal threshold |
| **A3** | the HPs set that size (two-term anatomy) | **proven** — the strongest link |
| **M** | HPs cause fate through the dose (mediation) | **supported** by level-set consistency + mediator-blocking (clamp) + mediator-pushing (λ); one statistical residual remains, attributable to trajectory-vs-endpoint |
| **A4** | Frobenius energy, not the spectral peak | **identified, not causally isolated for fate**; σ/F couple on-trajectory; single width |

Standing discipline carried from the doc: the DEAD label is a per-arm convention (an
unresolved ⚠ DISCUSSION affecting every fate-based fit); the scope lines (one width,
one pool family, ep20 budget, wd = 0, Adam only, cosine schedule with 5% warmup —
"lr" throughout means *peak* lr) are permanent, descope-able caveats; and the doc's
**Retired / stale — do not cite** list governs (notably: cos(ΔW,W₀) as mechanism,
the 8-cell spread-surgery counts, endpoint θ as a budget-invariant constant, the
adaptive 3e-10 mediation headline, the untrained-base "p ≈ 3e-5" figure, and the
wipe-lost notes-only numbers).

## §5.1 The claim ledger

Every load-bearing claim, its strongest current evidence class, and its status
(rows updated 2026-08-13 to match the doc). "Replicated" means at least two
independent designs agree."""))

cells.append(code(r"""# PROVENANCE: this ledger is a hand-maintained record of claim status (the `where` column
# names the section/notebook each verdict came from) — the strings are the data, not a
# recomputation. Edit rows only when a claim's status actually changes.
ledger = pd.DataFrame([
    # claim, class, status, where
    ("Collapse is a bimodal per-run fate with an empty trough", "cal", "replicated everywhere", "Part A1; every campaign"),
    ("Dead runs still fit their training data (not optimization failure)", "adaptive", "established; no strip version exists", "collapse_law_generality 12"),
    ("Collapse is not trait-specific - the law is not about SL", "adaptive", "load-bearing reframe; 22-cell control THIN, ~2x offset unquantified", "A1 scope section; collapse_mechanism 6c"),
    ("Boundary K = lr.alpha^b.r^c >= K*, b~0.56 (L), c~-0.09", "cal", "replicated x3 designs", "Part A1; nors_asks 7"),
    ("Exponents generalize across 3 teachers x 2 inits (I2~0)", "adaptive", "replicated", "2.2 (recomputed)"),
    ("Rank protective - model-free at matched (lr, alpha)", "adaptive", "replicated, zero reversals", "2.3 (recomputed)"),
    ("K* transfers cross-init at 1.02x (krungs both arms)", "adaptive", "replicated, executed", "top20_krungs 5+9; 2.4"),
    ("K* transfers cross-ANIMAL (cat/lion/pig on dog's anchor)", "adaptive", "replicated, arms still draining", "2.4 (recomputed)"),
    ("Per-teacher K* spread 0.85-1.14x (krungs); 66%-of-variance claim", "adaptive", "spread executed; 66% is notes-only (14 lost in wipe)", "top20_krungs 5; 2.4"),
    ("Law is REGIONAL: b_L 0.499+/-0.064, b_H 0.882+/-0.098", "cal", "established 3.3-sigma; the bend is a smooth alpha-glide", "2.5 (recomputed)"),
    ("Second boundary: optimization divergence at K/K* >~ 12", "cal", "established", "2.5 (recomputed)"),
    ("DEAD threshold is arm-specific (live-mode/10 rule)", "adaptive", "established; a convention (DISCUSSION: z-outcome proposal)", "1.1 (recomputed)"),
    ("Survivor transfer is HUMPED, peak K/K* ~ 0.3-0.6", "cal + interv", "replicated x3 + causal in-run", "2.5, 4.7"),
    ("Yield (transfer level) does NOT generalize (11x spread)", "adaptive", "established", "A1 scope section"),
    ("Rank buys survivor transfer at matched K (twins 1.25-1.43x)", "adaptive", "executed (krungs); +0.124/I2=0 refit is notes-only (15 lost)", "top20_krungs 7+9.5"),
    ("Fate is a soft threshold on ENDPOINT dose, AUC ~0.998", "cal", "replicated (adaptive 0.990 too); AUC saturates for both norms", "4.1 (recomputed)"),
    ("Within-mixed-cell 'dying seed has the larger dw'", "cal + adaptive", "REJECTED both ways: strip 6/9 smaller (p=0.51, coin-flip); adaptive z~-3.2 SMALLER", "4.1"),
    ("Death = loss of selectivity (pile KL AUC 0.999; ~3.4-nat gap)", "cal", "established; containment jump +1.87 log-nats at matched dose", "4.4 (recomputed)"),
    ("Dose causally flips fate at eval time, both directions", "interv", "established at scale (109 cells / 218 curves; ws1 8/8)", "4.7 (recomputed), 4.10"),
    ("Near-boundary death is MASKING (dead models revivable)", "interv", "established", "4.7 (recomputed)"),
    ("Training-time causation - the clamp tier", "interv-train", "CLOSED 08-12: survival monotone in ceiling (14/14 at 0.5x theta-hat); direction-corruption refuted", "4.9 (recomputed); nors_asks 5.5c"),
    ("ENDPOINT-dose sufficiency at training time", "interv-train", "FALSE - 47% of runs with realized dose < 0.95 theta-hat still died; causal dose is TRAJECTORY-like (~0.6x theta-hat operative)", "4.9; nors_asks 5.5c"),
    ("Flip dose is history-dependent (direction factor)", "interv", "established; ~3x adjusted [2.5, 3.7] at sweep scale, alpha-structured (~4x at a<=8 -> ~2.3x at a=128); UNNAMED", "4.10 (recomputed)"),
    ("Fate decision is EARLY (q crossover; K* ~ epochs^-0.2)", "adaptive", "established, indirect; no calibrated version (ckpt-replay pilot in flight)", "4.5 [recorded]"),
    ("Dose law lr^1.05 alpha^0.69 r^-0.24, R2~0.97, both quadratics real", "cal", "established", "4.2 (recomputed)"),
    ("Collapse law = level set of dose law (H closure |z|=0.0)", "cal", "established boundary-locally; global gap = estimator geometry (threshold-world sim, 97-99% labels)", "4.2, 4.11 (recomputed)"),
    ("Two-term anatomy: frozen straight, moved convex, alpha crossover", "ident + cal", "established", "4.3 (recomputed)"),
    ("Frobenius factor-composition licensed for FROZEN term only (deterministic sqrt(r)); moved gap NOT HP-flat (+/-24%)", "ident", "corrected 08-13 - moved exponents measured directly (lr^1.302 a^0.498 r^-0.107)", "4.3; nors_asks 1.5"),
    ("Trait rides the dominant term (double dissociation)", "interv", "established (32/32 on the strip splices)", "4.3; nors_asks 5.4"),
    ("Endpoint mediation residual (matched dw: higher lr dies more)", "cal", "calibrated at p~1e-3 BOTH norms; attributable to trajectory-vs-endpoint; adaptive 3e-10 gulf retired as headline", "4.12 (recomputed)"),
    ("Frobenius, not spectral, is the size that matters", "cal + interv", "IDENTIFIED (3 legs favor F), not causally isolated for fate; single-width caveat", "Part A4"),
    ("DEAD label conflates collapse-death vs carrier-deletion death", "interv", "established at shell scale (top1 kills mostly have FALLING pile KL)", "4.13 (recomputed)"),
    ("cos(dW, W0) as fate mechanism", "interv + cal", "RETIRED (marker only, and not robust)", "4.6"),
    ("Summary spread / peak as the damage variable", "interv", "REFUTED", "4.7b, 4.13"),
    ("Which subspace property sets the personal threshold", "-", "OPEN", "causal_plan; doc Open Questions 9"),
    ("Retrain-B tier (subspace causation at training time)", "-", "OPEN, designed", "causal_plan"),
    ("Pretraining maturity gates transfer BOTH sides (ckpt line)", "adaptive", "new 08-10, arms draining", "checkpoint_distillation_analysis"),
    ("Width/model-size robustness (Adam-width objection)", "-", "OPEN - biggest gap", "nors_asks 1; doc Open Questions 8"),
], columns=["claim", "class", "status", "where"])
with pd.option_context("display.max_colwidth", 90):
    display(ledger.style.set_properties(**{"text-align": "left"})
            .apply(lambda s: ["color: #B00020" if ("RETIRED" in v or "REFUTED" in v
                                                   or "REJECTED" in v or "FALSE" in v)
                              else ("color: #806000" if "OPEN" in v else "")
                              for v in s], subset=["status"], axis=0)
            .hide(axis="index"))"""))

cells.append(md(r"""## §5.2 What separated real evidence from artifact

The program produced four confident wrong results and caught all of them. The
catching mechanisms are the actual methodology lesson — each is now a standing rule:

1. **The DEAD-threshold trap** (§1.1). A cut tuned on one arm produced I² = 75–78%
   exponent "heterogeneity" (p < 0.001!) that vanished at the right cut. → *Every
   inherited analysis constant must be revalidated on new data; thresholds come from
   each arm's own histogram.*
2. **The support/leverage trap** (§2.1). "Rank is not in the law" was an artifact of
   a grid that never ran rank < 4; no estimator can rescue an axis the design never
   varied. → *Check design coverage before spending seeds.* Its mirror: the lone
   (ln α)² probe was unidentifiable on the adaptive grid's correlated support —
   the strip settled it (+0.018 ± 0.007, p = 8e-3) in the *opposite* direction of
   the adaptive-era retraction. (Its sibling: within-rank fate AUCs saturate ≥ 0.99
   for both norms on the strip and are *retired as evidence* — a discriminator can
   be too easy as well as confounded.)
3. **The composition trap.** Per-arm ‖ΔW‖ fate gaps read *reversed* on one arm
   because arms carried unequal HP/budget mixes (Table 1.1b); the ws1 seed-level
   cos signal died exactly at cell fixed-effects. → *Never read cross-arm contrasts
   off unequal coverage; always push seed-level claims through within-cell tests.*
4. **The adaptive-inference trap** (§2.5). Four "beautiful" adaptive-era statistics
   (an lr regime shift among them) dissolved on the calibrated strip, while the real
   effects (α², regional b) sharpened. → *Adaptive data is for discovery and
   replication-of-direction; calibrated designs are for inference. The strip's 852
   GPU-jobs bought the right to say "p = 4e-3" and mean it.*

Also retired on identification grounds: fixed-(lr, α) rank ladders (K and rank
perfectly collinear — circular as a c estimate), per-arm K\* placement for new
campaigns (tunes the small variance term while breaking cell pairing), and the
"α/r scale" framing (loses to absolute α by 21/105 cells).

"""))

cells.append(md(r"""## §5.3 The demonstration blueprint

What a paper-strength case looks like, given the ledger — ordered by how much each
item de-risks the story:

1. **A second width/model size** (pythia-410m or 1b, one strip-style factorial +
   one law refit). Settles Nor's Adam-width objection, the only structural threat to
   the Frobenius framing; also tests whether K\* scales as any μP-style prediction.
   Without this, every norm-based claim carries a permanent asterisk.
2. **Training-time causation — the clamp tier is DONE** (§4.9, closed 2026-08-12:
   dose capping at 0.5×θ̂ rescues 14/14 verified-dead cells; endpoint sufficiency
   false — the causal dose is trajectory-like). What remains of the original item:
   the **retrain-B tier** (re-train with A frozen at a dead run's subspace;
   prediction: fate follows subspace; design in `causal_plan.md`) and a calibrated
   **trajectory-exposure statistic** (the ckpt-replay pilot, 2026-08-13, produces
   the raw material).
3. **A trait-free boundary factorial** — elevated by the A1 reframe: the claim "the
   boundary is not an SL phenomenon" currently rests on 22 control cells with an
   unquantified ~2× threshold offset (trait data dying slightly earlier). A
   strip-style factorial (or at minimum the boundary band) on base-model and rng
   numbers pins the trait-free K\*, its exponents, and the offset — turning the
   reframe's pivotal premise into a measured result. Cheap: same worker, same cells,
   different pool. (The doc calls this **the pivotal missing experiment**.)
4. **A second pool/dataset family** (constrained-generation pools already exist at
   ≥10k-usable coverage): one law refit. Cheap, closes the "one pool" scope line.
5. **Subspace identity** — the open mechanism axis: ΔW energy resolved against W₀'s
   singular subspaces and against activation covariance, then a *targeted* surgery
   (move energy between identified subspaces at matched norm) rather than another
   summary statistic. This is what would name the direction factor §4.10 measured.
6. **An SGD (or wd > 0) arm**: the weak-form mechanism predicts b → ~1.5 under SGD —
   a sharp, cheap, falsifiable prediction that would powerfully confirm the
   optimizer-mediated dose story (or break it usefully).
7. **Checkpoint fate + rescue branching**: turns "the decision is early" into a
   directly observed decision time (the ckpt-replay pilot is the first step).
8. **Provenance repair (cheap, disk-only):** re-derive the two wipe-lost analyses —
   the per-arm K\*₅₀ refit (old §14) and the conditional-transfer refit + its
   `fit_conditional_transfer` estimator (old §15) — so the "teachers 66% of K\*
   variance", rank-4 dip, and rank-conditional claims stop resting on session notes.
   Likewise keep `transfer_factor_decomposition.ipynb` re-running as the cat/lion/pig
   arms drain, and mark `seed_fate_direction.ipynb` stale in-file.

**The paper-grade figure set already in hand**: the six-arm forest (§2.2), the
cross-animal boundary chart (§2.4), the strip fate maps + regional law (§2.5), the
dose-AUC scatter (§4.1), the selectivity split (§4.4), the λ dose-response/revival
panel (§4.7), the clamp dose-response (§4.9), the direction-factor-by-α chart
(§4.10), the level-set closure + threshold-world comparison (§4.2/§4.11), the Test-1
threshold panel (§4.1b), and the carrier-deletion split (§4.13). Each is recomputed
in this notebook from primary artifacts.

---
*Builder: `scripts/build_collapse_evidence_nb.py`. Keep builder and notebook in
sync; the narrative SSOT is `COLLAPSE_CLAIM_CHAIN.md`. SSOTs for the underlying
experiments: `nors_asks.ipynb` (norms, dose, interventions, strip, clamp),
`collapse_law_generality.ipynb` (six-arm law), `collapse_mechanism.ipynb`
(assumption ledger, temporal), `top20_krungs_ds1.ipynb` (cross-init ladder),
`collapse_campaigns_k_alpha_rank.ipynb` (threshold conventions; the cross-animal
band notebook was lost in the 08-05 wipe — §2.4 here recomputes it),
`transfer_factor_decomposition.ipynb` (transfer-level factors).*"""))

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
})

if __name__ == "__main__":
    import os
    import sys

    # Shared box with jobq pools: cap BLAS threading before the kernel spawns.
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
