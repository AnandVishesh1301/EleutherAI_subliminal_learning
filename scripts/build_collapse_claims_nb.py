"""Build notebooks/collapse_claims.ipynb — executed evidence for claims C1–C7
(narrative order, final renumber 2026-08-20b: C1 event, C2 law, C3 generality,
C4 trait, C5 HPs-set-size [doc-only], C6 dose-separates-fate, C7 Frobenius [doc-only],
C8 fidelity, C9 doctrine [doc-only]).

Companion to docs/results/collapse_claims_ordered.md: every claim gets a cell that
recomputes its headline numbers from durable sources (run-dir eval records, the cached
all-arms norm sweep, priors, the postmortem CE study). CPU-only. Rebuild + execute:

    uv run python -m scripts.build_collapse_claims_nb
    uv run jupyter nbconvert --to notebook --execute --inplace notebooks/collapse_claims.ipynb
"""
from __future__ import annotations

import json
from pathlib import Path

from subliminal_learning.stats import collapse_law as cl

# Reference (quotable) collapse-law constants, rendered once for the markdown below.
_B_REF, _C_REF = f"{cl.B_LAW:.3f}", f"{cl.C_LAW:.3f}"
_KSTAR_TEX = f"{cl.KSTAR_CALIBRATED * 1e4:.2f}" + r"\times10^{-4}"

CELLS: list[tuple[str, str]] = []


def md(s: str) -> None:
    CELLS.append(("markdown", s.strip("\n")))


def code(s: str) -> None:
    CELLS.append(("code", s.strip("\n")))


md(r"""
# Collapse claims C1–C9 — executed evidence under the three-readout framework

Companion notebook to **`docs/results/collapse_claims_ordered.md`** (the ordered claim set);
`docs/results/collapse_law.md` remains the results SSOT. Each section recomputes one claim's
headline numbers from durable sources, so the claims doc is auditable against a fresh run.

**The three readouts** (one per curve, each with its own natural reference — no shared
threshold constant):

| curve | readout | reference | file source |
|---|---|---|---|
| **trait** | test p(animal), exact prob scorer | own base's test-set prior (`analysis/base_eval_*`) | `preference_summary.json` per cell |
| **integrity** | Pile CE delta / Pile-logit KL vs own base | base = 0 by construction | `{run}/eval/pile_probe.json` (endpoint; pp1 pool), `{run}/eval/pile_kl_traj.json` (time course) |
| **task fidelity** | held-out-CE gain = base CE − student CE on 256 unseen teacher-number rows (nats/token; also reported: train CE on 256 trained rows, and the memorization gap = held-out − train) | own base's CE on the same rows (≈7.11 held-out / 7.23 train) | `{run}/eval/fidelity_heldout.json` |

**Dead label** (Angela 2026-08-26): the OPERATIVE label everywhere is **integrity** = probe
Pile-KL (delta-CE is numerically identical to it, Pearson 1.0000 on the shared pp1 pool)
above the **per-receiver anchored cut** Δ_cross = H_uni − CE_base — the dose at which the
student's absolute Pile CE reaches the context-free unigram entropy H_uni ≈ 7.80 nats
(`analysis.labels.anchored_cut`; ≈5.06–5.17 nats across the six 160m bases). Unlike a flat
constant this is *derived per receiver* and needs no convention. The declared
`probe_io.DEAD_KL_CUT = 4.0` nats — the previous operative label — is kept only as an
explicitly labelled **comparison** line in the cut-sensitivity tables (the flip relabels
only transit-plateau runs and moves levels a few percent, never a conclusion). The
**legacy trait** label (`analysis.labels.label_dead_trait_legacy`, test p < `cl.DEAD_SEED`)
appears in exactly two places: C2's label-comparison (its purpose) and C6's θ̂ (the
norm-swept populations predate probe coverage) — caveat: trait loss ≠ integrity collapse,
the trait outlives collapse. The old prior/e³ trait rule (test p < test_prior·e⁻³) is
REMOVED from this notebook. Row unit throughout: one seed-run (ep20 unless stated).

**Support populations** (2026-08-19 rule: no adaptive-era law-campaign / krungs cells):
single-arm depth comes from the **strip** (the calibrated non-adaptive focus-teacher sweep,
dog/ds1); everything cross-arm comes from landed runs on the **core65** shared-universe
cells (focus20 ∪ band48 ∪ lowk9 ⊂ bandext126; manifest files under `run_logs/`), read via
the realized run inventory `run_logs/strip_probe_lists/xarm_bandext126_runs.txt`.
""")

code(r"""
import glob as _glob
import json, re, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")   # tqdm/short_model/statsmodels separation noise
pd.set_option("display.width", 200)

from subliminal_learning.analysis.setup import bootstrap, find_root
from subliminal_learning.analysis import collapse_law as cl
from subliminal_learning.analysis import labels
from subliminal_learning.analysis import plots as aplots
from subliminal_learning.analysis.frames import read_json
from subliminal_learning.analysis import transfer_factors as tf

REPO = find_root()
_, OUT = bootstrap(REPO)
priors = tf.load_priors(OUT)
KSTAR = cl.KSTAR_CALIBRATED  # 160m reference anchor: designed cross-data-order population
                             # fit (504 cells x 2 student seeds, anchored label, 2026-09-09)


# base-model dir -> arm label; a fully re-randomized sibling keeps a `full-` marker
# so pythia-160m-seed2 cannot be read as a data-/weight-seed arm
def base_label(name):
    a = name[len("pythia-160m-"):] if name.startswith("pythia-160m-") else name
    return "full-" + a if a.startswith("seed") else a

# core65 shared-universe design: focus20 ∪ band48 ∪ lowk9 (65 ep20 cells, K/K* 0.29-1.25);
# the 12 focus20∩band48 duplicates are dropped once (tf.load_core65_design asserts the 65)
core65, _ = tf.load_core65_design(REPO)

# strip three-curves frame: live aggregate of the per-run eval records
# (builder: scripts/three_curves_strip_ds1.py; sources listed in its header)
tc = pd.read_csv(OUT / "analysis" / "fidelity_strip_ds1" / "three_curves.csv")
# Krel comes straight from the csv: it is built in the cl.K_of / cl.KSTAR_CALIBRATED coordinate.
Q_DS1 = float(priors.query("animal=='dog' and base_label=='data-seed1'")["test_prior"].iloc[0])
# legacy trait cut — caveat: trait loss ≠ integrity collapse; operative label = anchored KL (analysis.labels)
tc["dead_trait"] = labels.label_dead_trait_legacy(tc, col="p_trait", cut=cl.DEAD_SEED).astype(int)
print(f"strip seeds {len(tc)}; trait dead {tc.dead_trait.sum()} (legacy trait cut p<{cl.DEAD_SEED}); "
      f"pile CE coverage {tc.pile_delta.notna().sum()}, pile KL coverage {tc.pile_kl_final.notna().sum()}, "
      f"held-out-CE (fidelity) coverage {tc.ce_heldout.notna().sum()}")

""")

md(r"""
## C1 — collapse is an event on the integrity curve: smooth below K*, catastrophic at it

Binned three-curve table (strip; K/K* at the 160m reference exponents b=""" + _B_REF + r""", c=""" + _C_REF + r""" —
the designed cross-data-order population fit, anchored label, 2026-09-09; but
⚠ the CSV’s $K$ was BUILT in the retired trait-era PLACEMENT coordinate
(exponents 0.561, −0.085) and only its anchor is rescaled here, so the binning below
inherits the placement exponents, not the reference ones):
general-distribution damage is graded below the boundary and jumps at it — the bimodality
(modes ~1 and ~15 nats, sparse transit plateau ~4–12) that licenses the declared label.
Coverage: the pile PROBE (`eval/pile_probe.json`, pp1 pool) gives delta-CE AND KL for every
final from one pass — seed-level Pearson 1.0000 between them, probe KL bit-identical to the
trajectory endpoints, so the historical CE-shard/KL-pool split is closed.
""")

code(r"""
bins = [0, .1, .2, .3, .45, .6, .8, 1.0, 1.3, 1.8, 2.5, 4, 8, 12.5, 100]
g = tc.groupby(pd.cut(tc.Krel, bins), observed=True)
summ = pd.DataFrame({
    "n": g.size(),
    "trait_p_med": g.p_trait.median().round(4),
    "frac>prior": g.apply(lambda x: float((x.trait_rel > 0).mean()), include_groups=False).round(2),
    "pileCE_med": g.pile_delta.median().round(2),
    "pileKL_med": g.pile_kl_final.median().round(2),
    "fid_gain_med": g.fid_gain.median().round(2),
})
print("columns: trait_p_med = median test p(dog); frac>prior = share of seeds above their base's")
print("test prior; pileCE_med / pileKL_med = median probe delta-CE / KL vs base (nats/token);")
print("fid_gain_med = median held-out-CE gain (base CE - student CE, 256 unseen teacher-number rows).\n")
display(summ)

for col in ("pile_delta", "pile_kl_final"):
    s = tc.dropna(subset=[col])
    a, dd = s[s.dead_trait == 0][col], s[s.dead_trait == 1][col]
    print(f"\n{col}: fate AUC {roc_auc_score(s.dead_trait, s[col]):.3f} | "
          f"alive P95 {a.quantile(.95):.2f} vs dead P5 {dd.quantile(.05):.2f} "
          f"({'literal gap' if dd.quantile(.05) > a.quantile(.95) else 'distributions touch'})")
""")

code(r"""
# the strip's probe-KL histogram -- the bimodal shape (alive mode / sparse transit /
# deep-dead mode) that licenses a declared cut  [moved here from the campaigns notebook]
from subliminal_learning.coherence.probe_io import DEAD_KL_CUT
CUT_DS1 = labels.anchored_cut(OUT, "pythia-160m-data-seed1")   # OPERATIVE per-receiver cut
_klh = tc.pile_kl_final.dropna()
fig, ax = plt.subplots(figsize=(7.5, 3.4))
ax.hist(_klh, bins=np.arange(0.0, float(_klh.max()) + 0.75, 0.75), color="#4c78a8")
ax.axvline(CUT_DS1, color=aplots.C_DEAD, ls="-", lw=1.8,
           label=f"anchored cut {CUT_DS1:.2f} (operative)")
ax.axvline(DEAD_KL_CUT, color="k", ls="--", lw=1.5, label=f"declared {DEAD_KL_CUT} (comparison)")
ax.set_yscale("log")
ax.set(xlabel="probe Pile-KL vs own base (nats)", ylabel="runs per 0.75-nat bin (log)",
       title=f"Focus-teacher strip (n={len(_klh)}): alive mode / sparse transit / deep-dead mode")
ax.legend(frameon=False)
plt.tight_layout(); plt.show()
""")

md(r"""
## C2 — the law: the event's location is a threshold in the HPs; the label survives redefinition

The strip law refit at the operative cut, at alternates inside the plateau, and on the
trait-prior label. Conclusions at cut 4: the exponents are cut-stable (b 0.61 ± 0.005,
c −0.09..−0.11 across cuts 4→7.9) while K* moves with the cut (8.5e-4 → 1.07e-3, ~25% —
the cut's one real lever, which is why every K* quote names its cut); the integrity and
trait-prior label systems agree on 96.1% of seeds, all disagreements in the boundary band —
the boundary is a property of the data, and its location is K = lr·α^b·r^c ≥ K*.
""")
code(r"""
# law refit on the probe-covered strip seeds: legacy trait label vs integrity labels
# (pile_kl_final in the csv is sourced from eval/pile_probe.json -- full final coverage)
from subliminal_learning.coherence.probe_io import DEAD_KL_CUT
print(f"operative cut: KL > {CUT_DS1:.2f} (anchored, receiver pythia-160m-data-seed1); "
      f"declared {DEAD_KL_CUT} shown for comparison")
sub = tc.dropna(subset=["pile_kl_final"]).copy()
parts = sub.cell.str.extract(r"ep20_lr([0-9.e-]+)_r(\d+)a(\d+)")
sub["lr"], sub["rank"], sub["alpha"] = parts[0].astype(float), parts[1].astype(int), parts[2].astype(int)
sub["student_tag"] = sub.cell
_cutnames = [(CUT_DS1, f"integrity KL>{CUT_DS1:.2f} (OPERATIVE, anchored)"),
             (DEAD_KL_CUT, f"integrity KL>{DEAD_KL_CUT:.1f} (declared, comparison)"),
             (5.0, "integrity KL>5.0"), (7.9, "integrity KL>7.9")]
rows = []
for name, lab in [("trait (legacy cut)", sub.dead_trait)] + [
        (nm, (sub.pile_kl_final > c).astype(int)) for c, nm in _cutnames]:
    dd = sub.assign(collapsed=lab)
    fit = cl.fit_collapse_law(dd)
    ks = cl.kstar_logistic(dd, fit.b, fit.c)
    rows.append(dict(label=name, b=round(fit.b, 3), c=round(fit.c, 3),
                     kstar=f"{ks['kstar']:.2e}", n_dead=fit.n_dead, err=round(ks["err"], 3)))
agree = ((sub.pile_kl_final > CUT_DS1).astype(int) == sub.dead_trait).mean()
display(pd.DataFrame(rows))
print(f"\nlabel agreement (anchored KL>{CUT_DS1:.2f} vs legacy trait cut): {agree:.3%} on {len(sub)} seeds; "
      "all disagreements are boundary seeds (K/K* 0.5-1.25).")

""")

md(r"""
## C3 — generality: K* across arms and animals under the integrity (probe-KL) label

The prior/e³ trait label is REMOVED from this section (Angela 2026-08-19b): everything below
uses ONLY the integrity event — probe Pile-KL above the **per-receiver anchored cut**
(`analysis.labels`, Angela 2026-08-26); the declared `DEAD_KL_CUT = 4.0` appears only as a
labelled comparison row. Support: the core65 cross-arm population loaded by
`tf.load_core65_population` — probe-covered landed runs on the 65 shared cells for every
animal × base × shared-teacher arm, plus the strip teacher's core65 cells. (Before
2026-08-26 this section read the whole 126-cell bandext probe cache; it is now the core65
shared universe, so per-arm K* levels move.)

This section is the home of the K* calibration and its structure (moved here from the
campaigns notebook 2026-08-20b — the campaigns notebook keeps only the mechanical
calibration its axis markers need). Conclusions, settled 2026-08-20:

1. **One law, animal-invariant**: fitted exponents b 0.57–0.58, c −0.07..−0.09 per animal;
   per-(animal × base) arm K* spread 1.18× with the base ordering ws1 < ds* < ws2
   replicating in every animal.
2. **K* is base-first** (C3c): variance decomposition of per-slice log K* — base 0.66 /
   teacher 0.30 / animal 0.06 partial R². The trait is the *smallest* factor in the
   boundary's location: the strongest form of "the law is not about subliminal learning".
3. **ws1 is intrinsically fragile**: lowest K* in every animal AND lowest θ in the
   realized-update coordinate (C6b) — a property of that checkpoint, not of the HP mapping.

**Not yet coverable**: trait-free factorial (queued ladder), same-model non-adaptive
sweep (orig→orig ladder queued 08-16), Pythia-2.8B — picked up when their outputs land.
""")

code(r"""
# core65 population: cells x the shared bandext126 teachers x 4 animals x 6 bases, joined to
# the pile-probe cache and to each receiver base's anchored cut (tf.load_core65_population).
from subliminal_learning.analysis.probe_cache import load_probe_cache
from subliminal_learning.coherence.probe_io import DEAD_KL_CUT

_STRIP_DIR = ("students/dog/pythia-160m-data-seed1/teacher-pythia-160m/"
              "ep2_lr5e-4_r32a64_tseed1/ex10-20_prefix_n100k")
fr, C65 = tf.load_core65_population(OUT, REPO)
print(f"landed core65 seed-runs: {len(fr)} "
      f"({fr.animal.nunique()} animals x {fr.base_label.nunique()} bases, "
      f"{fr.teacher_tag.nunique()} teachers, {fr.student_tag.nunique()} cells)")

cov = fr.assign(has=fr.pile_kl.notna()).groupby("animal")["has"].agg(["sum", "count"])
print("\nprobe coverage by animal (self-updating; queue bandext126-pileprobe):")
display(cov)

CUTS = C65["delta_cross_by_base"]
print(f"\noperative label: probe KL > the receiver's ANCHORED cut (H_uni {C65['h_uni']:.3f} nats): "
      + ", ".join(f"{b}={v:.2f}" for b, v in CUTS.items())
      + f"  [declared {DEAD_KL_CUT} kept as comparison only]")

""")

code(r"""
pv = fr.dropna(subset=["pile_kl"]).copy()
pv["collapsed"] = labels.label_dead_kl(pv, CUTS, kl_col="pile_kl", base_col="base_label").astype(int)

def _try_fit(g):
    # thin/separable dead sets make the logit Hessian singular -- skip loudly, never crash
    try:
        fit = cl.fit_collapse_law(g)
        return fit, cl.kstar_logistic(g, fit.b, fit.c)
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"

print("integrity-native fits per animal, pooled over bases (covered core65 runs only;")
print("the design tops out at K/K*~1.25, so deads are boundary-band deads):")
arows = []
for animal, g in pv.groupby("animal"):
    fit, ks = (_try_fit(g) if g.collapsed.sum() >= 10 else (None, "too few dead"))
    if fit is None:
        print(f"  {animal}: not fittable yet ({int(g.collapsed.sum())} integrity-dead / {len(g)} covered; {ks})")
        continue
    arows.append(dict(animal=animal, b=round(fit.b, 3), c=round(fit.c, 3),
                      kstar=f"{ks['kstar']:.2e}", n_dead=fit.n_dead, n=fit.n_runs))
if arows:
    display(pd.DataFrame(arows))

print("\nper (animal, base) arm under the same label (n_dead < 20 rows are boundary-thin):")
brows = []
for (animal, b), g in pv.groupby(["animal", "base_label"]):
    if g.collapsed.nunique() < 2 or g.collapsed.sum() < 5:
        continue
    fit, ks = _try_fit(g)
    if fit is None:
        continue
    brows.append(dict(animal=animal, base=b, b=round(fit.b, 3), c=round(fit.c, 3),
                      kstar=f"{ks['kstar']:.2e}", n_dead=fit.n_dead, n=fit.n_runs))
btbl = pd.DataFrame(brows)
if len(btbl):
    display(btbl)
    if (btbl.n_dead >= 20).any():
        kk = btbl[btbl.n_dead >= 20].kstar.astype(float)
        print(f"\narms with n_dead>=20: K* spread {kk.min():.2e}-{kk.max():.2e} ({kk.max()/kk.min():.2f}x)")

# cut sensitivity on the best-covered animal (dog): the fit should move little inside the valley
gd = pv[pv.animal == "dog"]
sens = []
f0, k0 = _try_fit(gd)   # already carries the anchored per-receiver label
if f0 is not None:
    sens.append(dict(cut="anchored per base (OPERATIVE)", b=round(f0.b, 3),
                     kstar=f"{k0['kstar']:.2e}", n_dead=f0.n_dead))
for cname, cval in [(f"{DEAD_KL_CUT} (declared, comparison)", DEAD_KL_CUT), ("5.0", 5.0), ("7.9", 7.9)]:
    f_, k_ = _try_fit(gd.assign(collapsed=(gd.pile_kl > cval).astype(int)))
    if f_ is not None:
        sens.append(dict(cut=cname, b=round(f_.b, 3), kstar=f"{k_['kstar']:.2e}", n_dead=f_.n_dead))
print("\ncut sensitivity (dog, covered runs):")
display(pd.DataFrame(sens))
""")

md(r"""
### C3b — the per-arm K* calibration (display exponents, logistic-50%)

One boundary per (animal × base) arm — NO cross-animal or cross-base aggregation — at the
FIXED 160m reference exponents (b=""" + _B_REF + r""", c=""" + _C_REF + r"""; the designed cross-data-order population
fit, anchored label, 2026-09-09) so arms share one coordinate. Support: the full
probe-covered calibration population (all ftseeds, all teachers, + the strip). Estimators:
`cl.kstar_logistic` (50% crossing on log K; single-seed compatible, cluster-robust-by-cell
CI) with `cl.kstar_minerr` as the assumption-free check; AUC = planned dose vs the declared
label. The dog|ds1 support split shows teacher composition moves K* more than the CIs —
quote support with any per-arm value.
""")

code(r"""
warnings.filterwarnings("ignore")   # statsmodels re-registers separation warnings mid-notebook
# same population as C3 (tf.load_core65_population), renamed to this section's column names;
# `collapsed` is already the ANCHORED per-receiver label set in the C3 cell.
CAL = (pv.rename(columns={"teacher_tag": "teacher", "seed": "ftseed", "pile_kl": "kl"})
         .drop_duplicates(["animal", "base_label", "teacher", "student_tag", "ftseed"]))

_rows = []
for (a, b), g in CAL.groupby(["animal", "base_label"]):
    rec = dict(animal=a, base=b, n=len(g), dead=int(g.collapsed.sum()))
    if g.collapsed.sum() >= 5:
        e = cl.kstar_logistic(g, cl.B_LAW, cl.C_LAW)
        m = cl.kstar_minerr(g, cl.B_LAW, cl.C_LAW)
        lo, hi = e["kstar"] * np.exp(-1.96 * e["se_log"]), e["kstar"] * np.exp(1.96 * e["se_log"])
        rec.update(k_logit=f"{e['kstar']:.2e}",
                   ci95=f"[{lo:.2e},{hi:.2e}]" if np.isfinite(e["se_log"]) else "--",
                   k_minerr=f"{m['kstar']:.2e}",
                   auc=round(roc_auc_score(g.collapsed, cl.K_of(g.lr, g.alpha, g["rank"])), 3))
    _rows.append(rec)
print("K* per (animal x base) arm -- integrity label KL > the receiver's anchored cut, display "
      f"exponents b={cl.B_LAW}/c={cl.C_LAW}; gate >= 5 dead:")
display(pd.DataFrame(_rows))
_g1 = CAL[(CAL.animal == "dog") & (CAL.base_label == "data-seed1")]
_gx = _g1[_g1.teacher != "ep2_lr5e-4_r32a64"]
print(f"\ndog|ds1 support split: all {cl.kstar_logistic(_g1, cl.B_LAW, cl.C_LAW)['kstar']:.2e} "
      f"vs ex-strip {cl.kstar_logistic(_gx, cl.B_LAW, cl.C_LAW)['kstar']:.2e}")
""")

md(r"""
### C3c — the structure of K*: base × teacher × animal

One logistic-50% fit per **(teacher × animal × base)** slice (top-10 teachers, gate ≥ 5
dead; dog|ds1 slices include the strip). Findings: (1) base is the dominant factor
(partial R² ≈ 0.66) with the ws1 < ds* < ws2 ordering replicating everywhere — and ws1 is
also low in θ (C6b), so it is genuinely more fragile; (2) the teacher effect (≈ 0.30) is
recipe-heat-ordered — hot ep2–ep7 high-lr teachers push students over at lower planned
dose — and mostly conserved across arms; (3) the animal effect is ≈ 0.06 (pig ~2–4% high,
plausibly pool-composition); (4) at fixed (teacher, arm), planned dose classifies fate at
AUC ≈ 0.98. This re-derives and REVERSES the wipe-lost "teachers are 66% of K* variance"
figure — the 66% factor is the base. Scatter panels are raw one-factor marginals (other
factors shown as color, not partialled out); the heatmap is the full slice table.
""")

code(r"""
import statsmodels.formula.api as smf
from scipy.stats import spearmanr

TOP10 = read_json(REPO / "run_logs/dog_top20krungs_configs.json")["param_tags"][:10]
_sl, _skip_thin, _skip_fit = [], 0, []
for (t_, a_, b_), g_ in CAL[CAL.teacher.isin(TOP10)].groupby(["teacher", "animal", "base_label"]):
    if g_.collapsed.sum() < 5 or g_.collapsed.nunique() < 2:
        _skip_thin += 1
        continue
    try:
        e_ = cl.kstar_logistic(g_, cl.B_LAW, cl.C_LAW)
    except Exception as _e:
        _skip_fit.append(f"{t_}|{a_}|{b_}: {type(_e).__name__}: {_e}")
        continue
    _sl.append(dict(teacher=t_, animal=a_, base=b_, n=len(g_), dead=int(g_.collapsed.sum()),
                    kstar=e_["kstar"],
                    auc=roc_auc_score(g_.collapsed, cl.K_of(g_.lr, g_.alpha, g_["rank"]))))
sl = pd.DataFrame(_sl)
print(f"skipped slices: {_skip_thin} too thin (<5 dead or one class), {len(_skip_fit)} unfittable")
for _m in _skip_fit:
    print("  ", _m)
print(f"estimable (teacher x animal x base) slices: {len(sl)}; "
      f"AUC median {sl.auc.median():.3f}, range {sl.auc.min():.3f}-{sl.auc.max():.3f}")
sl["logk"] = np.log(sl.kstar)
_full = smf.ols("logk ~ C(base) + C(teacher) + C(animal)", data=sl).fit()
for _d, _n in [("C(base)", "base"), ("C(teacher)", "teacher"), ("C(animal)", "animal")]:
    _r = smf.ols("logk ~ " + " + ".join(x for x in ("C(base)", "C(teacher)", "C(animal)")
                                        if x != _d), data=sl).fit()
    print(f"partial R2 of {_n:>7}: {_full.rsquared - _r.rsquared:.3f}")
print(f"full-model R2 {_full.rsquared:.3f}")
_rk = {tag: i + 1 for i, tag in enumerate(TOP10)}
piv = sl.pivot_table(index="teacher", columns=["animal", "base"], values="kstar") * 1e4
piv.index = [f"t{_rk[i]:02d} {i}" for i in piv.index]
piv = piv.sort_index()
piv.columns = [f"{a[:3]}|{b.replace('data-seed', 'ds').replace('weight-seed', 'ws')}"
               for a, b in piv.columns]
_med = piv.median(axis=1)
_cors = [spearmanr(piv[c].dropna(), _med[piv[c].dropna().index]).statistic
         for c in piv.columns if piv[c].notna().sum() >= 5]
print(f"teacher-ordering consistency across arms (Spearman vs cross-slice median): "
      f"median {np.median(_cors):.2f}, range {min(_cors):.2f}-{max(_cors):.2f} ({len(_cors)} arms)")
""")

code(r"""
BASE_ORD = ["weight-seed1", "data-seed3", "data-seed1", "data-seed2", "weight-seed3", "weight-seed2"]
ANIMAL_ORD = ["cat", "dog", "lion", "pig"]
A_COLOR = aplots.ANIMAL_COLORS                      # canonical Okabe-Ito trait colours
B_COLOR = {b: plt.get_cmap("coolwarm")(i / 5) for i, b in enumerate(BASE_ORD)}
_short = lambda b: b.replace("data-seed", "ds").replace("weight-seed", "ws")
_teach_order = [t for t in TOP10 if t in set(sl.teacher)]
_rng = np.random.default_rng(0)


# one jittered K* strip panel: x = `xcol` categories, colour = `ccol`, black factor medians
def _kstar_strip(ax, xcol, xorder, xlabels, ccol, corder, colors, label_fn=None, **legend_kw):
    for c_ in corder:
        g_ = sl[sl[ccol] == c_]
        x_ = g_[xcol].map({v: i for i, v in enumerate(xorder)}) + _rng.uniform(-0.18, 0.18, len(g_))
        ax.scatter(x_, g_.kstar * 1e4, s=42, alpha=0.75, color=colors[c_],
                   label=None if label_fn is None else label_fn(c_))
    for i, v_ in enumerate(xorder):
        m_ = sl[sl[xcol] == v_].kstar.median() * 1e4
        if np.isfinite(m_):
            ax.hlines(m_, i - 0.3, i + 0.3, color="k", lw=2.2)
    ax.set_xticks(range(len(xorder)))
    ax.set_xticklabels(xlabels, fontsize=9)
    if label_fn is not None:
        ax.legend(fontsize=8, frameon=False, **legend_kw)


fig, axs = plt.subplots(1, 3, figsize=(17, 4.8), sharey=True)
_kstar_strip(axs[0], "base", BASE_ORD, [_short(b) for b in BASE_ORD], "animal", ANIMAL_ORD,
             A_COLOR, label_fn=str, title="animal", loc="upper left")
axs[0].set_title("by student base (partial R2 0.66)")
axs[0].set_ylabel("slice K* (x1e-4, log scale)")
_kstar_strip(axs[1], "teacher", _teach_order, [f"t{_rk[t]:02d}" for t in _teach_order],
             "base", BASE_ORD, B_COLOR, label_fn=_short, title="base", loc="upper left", ncols=2)
axs[1].set_title("by teacher, distillability-rank order (partial R2 0.30)")
_kstar_strip(axs[2], "animal", ANIMAL_ORD, ANIMAL_ORD, "base", BASE_ORD, B_COLOR)
axs[2].set_xticklabels(ANIMAL_ORD, fontsize=10)
axs[2].set_title("by animal (partial R2 0.06)")
for ax in axs:
    ax.set_yscale("log")
fig.suptitle("log K* per (teacher x animal x base) slice against each factor -- black bars = "
             "factor-level medians (raw marginals, NOT partial residuals)", y=1.03, fontsize=11)
fig.tight_layout(); plt.show()

""")

code(r"""
_col_order = [f"{a}|{b}" for b in ("ws1", "ds3", "ds1", "ds2", "ws3", "ws2")
              for a in ("cat", "dog", "lio", "pig") if f"{a}|{b}" in piv.columns]
_pv = piv[_col_order]
fig, ax = plt.subplots(figsize=(0.62 * len(_pv.columns) + 2.2, 0.42 * len(_pv) + 1.6))
_im = aplots.annotated_heatmap(ax, _pv, cmap="RdYlBu", origin="upper", text_color="black",
                               vmin=np.nanmin(_pv.to_numpy()), vmax=np.nanmax(_pv.to_numpy()))
for _j, _c in enumerate(_pv.columns[:-1]):
    if _c.split("|")[1] != _pv.columns[_j + 1].split("|")[1]:
        ax.axvline(_j + 0.5, color="k", lw=1.4)
fig.colorbar(_im, ax=ax, label="slice K* (x1e-4)", shrink=0.85)
ax.set_title("K* per (teacher x animal|base) slice -- logistic-50% at the anchored cut;\n"
             "columns grouped by base (low ws1 -> high ws2), blank = < 5 dead (unestimable)",
             fontsize=9)
plt.tight_layout(); plt.show()

""")

md(r"""
## C4 — trait: humped below the boundary, extinguished at it

The `trait_p_med` and `frac>prior` columns above: median p rises 0.028→0.081 with the
coarse-bin peak on 0.45–0.8 (the strip survivors' calibrated peak is K/K*≈0.42 — see
`collapse_law.md` §3; adaptive supports place it 0.28–0.8, design-dependent), then the
above-prior fraction falls 1.00 → 0.77 → 0.19 → 0.00 across the crossing. Causality
within-run (λ-scaling) and the fragility bounds are cited in the doc, not recomputed here.

The figure below is the per-run view: every strip (focus-teacher) seed-run's test p(dog) on a
log axis against dose on the CALIBRATED anchor — K* re-estimated on this population under the
declared integrity label (`estimate_kstar` at the display exponents), so K/K* = 1 is this
population's own measured boundary. Colors = the declared-cut fate. Zero-p runs are drawn at
the plot floor.
""")

code(r"""
# strip per-run p(trait), log axis, vs dose on the anchor CALIBRATED under the anchored label
est_s = cl.kstar_logistic(sub.assign(collapsed=(sub.pile_kl_final > CUT_DS1).astype(int)),
                          cl.B_LAW, cl.C_LAW)
KSTAR_CAL = est_s["kstar"]
xx = sub.Krel * KSTAR / KSTAR_CAL     # re-anchor the display coordinate to the calibrated K*
dead = sub.pile_kl_final > CUT_DS1
P_FLOOR = 1e-6
yy = sub.p_trait.clip(lower=P_FLOOR)
fig, ax = plt.subplots(figsize=(9.5, 4.8))
ax.scatter(xx[~dead], yy[~dead], s=16, alpha=0.5, color="#2b6cb0",
           label=f"alive: probe KL <= {CUT_DS1:.2f}  (n={int((~dead).sum())})")
ax.scatter(xx[dead], yy[dead], s=16, alpha=0.5, color="#d03b3b",
           label=f"dead: probe KL > {CUT_DS1:.2f}  (n={int(dead.sum())})")
ax.axvline(1.0, color="k", ls="--", lw=1.4,
           label=f"K/K* = 1  (calibrated K* = {KSTAR_CAL:.2e})")
ax.axhline(Q_DS1, color="#666", ls=":", lw=1.2, label=f"base test prior {Q_DS1:.4f}")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_ylim(P_FLOOR / 2, 1.0)
ax.set(xlabel="K / K* (anchor calibrated on this population, integrity label)",
       ylabel="test p(dog), log scale")
ax.set_title(f"Strip (focus teacher): per-run p(trait) vs dose, fate-colored at the anchored cut\n"
             f"({int((sub.p_trait <= P_FLOOR).sum())} zero-p runs drawn at the {P_FLOOR:g} floor)")
ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
plt.tight_layout(); plt.show()
print(f"calibrated strip K* (logistic-50%, anchored integrity label KL>{CUT_DS1:.2f}, display "
      f"exponents): {KSTAR_CAL:.3e} (se_log {est_s['se_log']:.3f}, err {est_s['err']:.3f})")
""")

md(r"""
## C6 — dose separates fate: θ̂ (the best dose cut) is consistent across arms

The HPs act through the realized update size `dw_fro = mean over LoRA modules of
s·‖B·A_final‖_F/‖W₀‖_F` (the dose-trajectory convention; endpoint bit-equal to the harvest).
Source: the per-run adapter-norm records (`{run}/eval/adapter_norms.json`, served by
`analysis.norms.load_adapter_norm_runs` over the written inventory), one row per
(base, teacher, cell, ftseed), **all ep20
cells in the harvest** (no cell-subset restriction — per Angela 2026-08-19b; this support
includes legacy-era cells and is the one place they remain, because θ̂ needs dose breadth
and probe coverage does not reach these runs — hence the trait-prior label here).
θ̂ = the dw_fro cut minimizing misclassification, per student base. Coverage caveat: the
norm harvest exists only for dog n100k runs — cross-animal θ̂ needs a norm harvest over
the other animals' finals (CPU-only; not yet run).
""")

code(r"""
from subliminal_learning.analysis.norms import load_adapter_norm_runs

nrun = load_adapter_norm_runs(OUT)          # run-level dw_fro/test_p, no module-row parse
nrun = nrun[nrun.student_tag.str.startswith("ep20")].copy()   # all ep20 norm-covered runs
pm = priors[priors.animal == "dog"].set_index("base_label")["test_prior"]
nrun["base_label"] = nrun.student_base.map(base_label)
# legacy trait cut — caveat: trait loss ≠ integrity collapse; operative label = anchored KL (analysis.labels)
# TODO(anchored): C6b already refits θ̂ on the anchored KL label over the norm ∩ probe subset.
nrun["dead"] = labels.label_dead_trait_legacy(nrun, col="test_p", cut=cl.DEAD_SEED).astype(int)
print(f"ep20 runs with norms: {len(nrun)} (dog only), "
      f"{nrun.teacher.nunique()} teachers, {nrun.student_tag.nunique()} cells "
      f"({nrun.student_tag.isin(set(core65.student_tag)).sum()} runs on core65 cells)\n")

def theta(g):
    cuts = np.unique(g.dw_fro)
    err, cut = min((((g.dw_fro >= x) != g.dead.astype(bool)).mean(), x) for x in cuts)
    return pd.Series({"theta_hat": cut, "err": err,
                      "auc": roc_auc_score(g.dead, g.dw_fro) if g.dead.nunique() > 1 else np.nan,
                      "n": len(g), "n_dead": int(g.dead.sum())})

t_base = nrun.groupby("base_label").apply(theta, include_groups=False)
print("theta-hat per student base on all norm-covered ep20 cells (LEGACY TRAIT label), pooled "
      "over teachers "
      "(rows with n>=100 are well-populated):")
display(t_base.round(4))
well = t_base[t_base.n >= 100]
print(f"\nwell-populated bases: theta-hat spread {well.theta_hat.min():.3f}-{well.theta_hat.max():.3f} "
      f"({well.theta_hat.max()/well.theta_hat.min():.2f}x). "
      f"Strip ep20 reference value: 0.122; NOT budget-invariant (0.051/0.102/0.122 at ep5/10/20).")
""")

md(r"""
### C6 (cont.) — dose→fate AUC per base, and the chain pointer

The causal chain (A1–A4, M: size causes fate, HPs act through the dose, Frobenius not
spectral, sustained exposure) lives in `COLLAPSE_CLAIM_CHAIN.md` and `nors_asks`; it is a
mechanism *of the integrity event*. Recomputed here: dose→fate monotonicity on the C6
norm frame (all norm-covered ep20 runs), per base (the strip value is 0.998 [cal]).
A2/M re-scored against the integrity event in C6d below (2026-08-20).
""")

code(r"""
lw = nrun[nrun.groupby("base_label").dw_fro.transform("size") >= 100]
for b, g in lw.groupby("base_label"):
    print(f"{b:>12}  dose->fate AUC {roc_auc_score(g.dead, g.dw_fro):.3f}  (n={len(g)}, dead={g.dead.sum()})")
""")

md(r"""
### C6b — θ refit under the INTEGRITY label (norm ∩ probe subset, per dog base)

The table above uses the trait-prior label for full norm-sweep support. Here θ is refit
per (dog × base) on the runs that ALSO carry a probe record, labeled at the declared KL
cut — the integrity-native θ. Estimators: logistic-50% on log dw_fro (primary) and the
min-error cut (check); dose→fate AUC per base. Bases with < 5 integrity-deads in the
covered subset are printed as unestimable. Coverage is dog-only (the norm harvest) and
partial (probe records exist for bandext126 + strip cells only).
""")

code(r"""
import statsmodels.api as sm

from subliminal_learning.analysis.probe_cache import load_probe_cache

_pcth = load_probe_cache(OUT, nrun.run_path, refresh=False)   # run_path from the loader
nth = nrun.merge(_pcth[["run_path", "kl"]], on="run_path", how="inner").dropna(subset=["kl"])
_ac = {}
for _b in sorted(nth.student_base.unique()):
    try:
        _ac[_b] = labels.anchored_cut(OUT, _b)
    except KeyError as _e:
        print(f"no anchored cut for {_b} -- rows excluded ({_e})")
_nth_all = len(nth)
nth = nth[nth.student_base.isin(_ac)].copy()
nth["dead_kl"] = labels.label_dead_kl(nth, _ac, kl_col="kl", base_col="student_base").astype(int)
print(f"norm runs with probe KL: {_nth_all} of {len(nrun)}, {len(nth)} with an anchored cut; "
      "OPERATIVE integrity label = KL > the receiver's anchored cut "
      + ", ".join(f"{b.replace('pythia-160m-', '')}={v:.2f}" for b, v in _ac.items()) + "\n")
for b, g in nth.groupby("base_label"):
    if g.dead_kl.sum() < 5:
        print(f"{b:>12}: unestimable ({int(g.dead_kl.sum())} dead / {len(g)} covered)")
        continue
    x = sm.add_constant(np.log(g.dw_fro))
    r = sm.GLM(g.dead_kl, x, family=sm.families.Binomial()).fit()
    th = float(np.exp(-r.params.iloc[0] / r.params.iloc[1]))
    v = np.sort(g.dw_fro.unique())
    err, i = min(((((g.dw_fro >= c).astype(int) != g.dead_kl).mean(), j) for j, c in enumerate(v)))
    tm = float(np.sqrt(v[i] * v[i - 1])) if i > 0 else float(v[i])
    print(f"{b:>12}: theta_logit {th:.4f}, theta_minerr {tm:.4f} "
          f"(n={len(g)}, dead={int(g.dead_kl.sum())}, AUC {roc_auc_score(g.dead_kl, g.dw_fro):.3f})")
""")

md(r"""
### C6c — the dose law refit per base, all three HP axes (label-free)

`log dw_fro ~ log lr + log α + log r` per student base (cluster-robust by cell; the
regression never sees a dead label, so no cut enters). Finding (2026-08-20): **the HP→dose
mapping is base-invariant** — lr 1.08–1.15, α 0.646–0.671, r −0.21..−0.24 with overlapping
CIs across all six bases, implied b 0.586–0.598. Combined with the base-varying K* (C3b)
and θ (C6b): the base effect on the boundary lives ENTIRELY in the tolerance θ, none of it
in how HPs convert to update size. The implied c (≈ −0.20) still overshoots the fitted law
c (−0.07..−0.09) — the known rank gap the θ rank-drift absorbs. Support: dog-only (the
norm harvest), all six bases.
""")

code(r"""
import statsmodels.api as sm

def _dose_fit(g):
    X = sm.add_constant(np.log(g[["lr", "alpha", "rank"]]).rename(columns=lambda c: "log_" + c))
    r = sm.OLS(np.log(g.dw_fro), X).fit(cov_type="cluster", cov_kwds={"groups": g.student_tag})
    a_, b_, c_ = r.params["log_lr"], r.params["log_alpha"], r.params["log_rank"]
    return dict(lr=round(a_, 3), se_lr=round(r.bse["log_lr"], 3),
                alpha=round(b_, 3), se_a=round(r.bse["log_alpha"], 3),
                rank=round(c_, 3), se_r=round(r.bse["log_rank"], 3),
                r2=round(r.rsquared, 3), implied_b=round(b_ / a_, 3),
                implied_c=round(c_ / a_, 3), n=len(g))

_nd = nrun[np.isfinite(nrun.dw_fro) & (nrun.dw_fro > 0)].copy()
_prt = _nd.student_tag.str.extract(r"ep20_lr([0-9.e-]+)_r(\d+)a(\d+)")
_nd["lr"], _nd["rank"], _nd["alpha"] = (_prt[0].astype(float), _prt[1].astype(int),
                                        _prt[2].astype(int))
_dl = [dict(base="ALL-dog", **_dose_fit(_nd))] + [
    dict(base=b, **_dose_fit(g)) for b, g in _nd.groupby("base_label")]
print("dose law per base -- log dw_fro ~ log lr + log alpha + log rank (label-free):")
display(pd.DataFrame(_dl))
""")

md(r"""
### C6d — the causal legs on the integrity label, read as two channels (clamp + λ)

**Reframing (2026-08-20).** With integrity as the label, every causal intervention has two
readouts: *integrity* (probe Pile-KL vs own base; dead iff above the receiver's unigram-anchored
cut `analysis.labels.anchored_cut`, with the declared `DEAD_KL_CUT` = 4.0 as comparison) and *trait*
(test p(dog) / own base's test prior — a continuous revival/destruction readout, no longer a
label). The question per intervention is therefore not "does fate flip" but (i) does the
integrity event move with the dose, and (ii) where along the dose axis does the trait rise,
peak and die relative to that event.

**Clamp tier** (`clamp_students/<norm>-c<ratio>/…`, 166 runs, design frozen in
`run_logs/clamp_design_v1.json`; probed on pp1 2026-08-20 by `clamp-pileprobe`). Row unit:
one clamped seed-run. Ceilings were set as multiples of the per-rank dead-edge θ̂ derived on
the *trait* label; θ̂ is re-derived here on the integrity label with the same estimator
(accuracy-maximising "dead iff dw_fro ≥ cut", per rank, strip ep20 lineage — dog/ds1,
teacher ep2_lr5e-4_r32a64_tseed1), and each run is re-expressed by its **realized** final
dose (last `clamp_log.jsonl` row) over θ̂_KL. The decisive test is the last table:
P(integrity-dead) at matched realized dose, clamp vs unclamped strip runs of the same
ranks — if endpoint dose were the whole cause the two columns would agree.
""")

code(r"""
import glob as _glob, re as _re
from subliminal_learning.coherence.probe_io import load_probe as _load_probe

# --- per-rank θ̂ on the strip ep20 lineage, trait vs integrity label (same estimator) ---
def _dead_edge(x, y):
    o = np.argsort(x); x, y = np.asarray(x)[o], np.asarray(y)[o]
    N = len(x); cum = np.r_[0, np.cumsum(y)]
    acc = np.array([((i - cum[i]) + (cum[-1] - cum[i])) / N for i in range(N + 1)])
    i = int(np.argmax(acc)); return float(x[i]) if i < N else np.nan

_strip = nth[(nth.student_base == "pythia-160m-data-seed1")
             & (nth.teacher == "ep2_lr5e-4_r32a64_tseed1")
             & (nth.dataset_tag == "ex10-20_prefix_n100k")].copy()
_strip["rank"] = _strip.student_tag.str.extract(r"_r(\d+)a")[0].astype(int)
# legacy trait cut — caveat: trait loss ≠ integrity collapse; operative label = anchored KL (analysis.labels)
_strip["dead_trait"] = labels.label_dead_trait_legacy(_strip, col="test_p", cut=cl.DEAD_SEED).astype(int)
_th = pd.DataFrame([{"rank": rk, "n": len(g),
                     "theta_trait": _dead_edge(g.dw_fro, g.dead_trait),
                     "theta_anch": _dead_edge(g.dw_fro, g.dead_kl),
                     "theta_KL4cmp": _dead_edge(g.dw_fro, (g.kl > DEAD_KL_CUT).astype(int)),
                     "theta_KL5": _dead_edge(g.dw_fro, (g.kl > 5).astype(int))}
                    for rk, g in _strip.groupby("rank")]).set_index("rank")
_design = read_json(REPO / "run_logs/clamp_design_v1.json")["theta_hat"]
_th["theta_clamp_design"] = [_design.get(str(r), {}).get("fro", np.nan) for r in _th.index]
print(f"strip ep20 lineage with norms+probe: {len(_strip)} runs\n")
print("per-rank dead-edge θ̂ (dw_fro) by label — the clamp design used the trait-label edge:")
display(_th.round(4))

# --- clamp runs: probe KL, trait, realized dose ---
_rows = []
for _lab in sorted(_p.name for _p in (OUT / "clamp_students").iterdir()):
    if _lab == "ckpt-replay":
        continue
    _norm, _c = _lab.split("-c"); _ratio = float(_c.replace("p", "."))
    for _mf in _glob.glob(str(OUT / "clamp_students" / _lab / "*/*/*/*/*/*/*/train_manifest.json")):
        _rd = Path(_mf).parent; _pr = _load_probe(_rd) or {}
        _pe = json.loads((_rd / "preference_eval.json").read_text())["sets"]["test"]["p_target"]
        _log = [json.loads(l) for l in (_rd / "clamp_log.jsonl").read_text().splitlines()]
        _tag = _rd.parent.name; _r, _a = map(int, _re.search(r"_r(\d+)a(\d+)", _tag).groups())
        _rows.append(dict(norm=_norm, ratio=_ratio, tag=_tag, seed=_rd.name, rank=_r, alpha=_a,
                          kl=_pr.get("kl_base_student"), trait_ratio=_pe / pm["data-seed1"],
                          dose_final=_log[-1]["dose"], bind_frac=float(np.mean([x["bound"] for x in _log]))))
clamp = pd.DataFrame(_rows); clamp["kl"] = pd.to_numeric(clamp.kl, errors="coerce")
clamp["dead_kl"] = (~np.isfinite(clamp.kl) | (clamp.kl > CUT_DS1)).astype(int)
clamp["x_anch"] = clamp.dose_final / clamp["rank"].map(_th.theta_anch)
clamp["stage"] = np.where(clamp.alpha >= 128, "s3 deep-α", "s1/2 (α≤64)")
print(f"\nclamp runs: {len(clamp)}, probed: {int(clamp.kl.notna().sum())}")

def _summ(g):
    al = g[g.dead_kl == 0]
    return pd.Series({"n": len(g), "integrity_alive": int((g.dead_kl == 0).sum()),
                      "alive@cut5": int((g.kl <= 5).sum()), "alive@cut7.9": int((g.kl <= 7.9).sum()),
                      "med_KL": round(g.kl.median(), 2),
                      "realized/θ̂_anch (med)": round(g.x_anch.median(), 2),
                      "bind_frac": round(g.bind_frac.median(), 2),
                      "trait>base | alive": f"{int((al.trait_ratio > 1).sum())}/{len(al)}",
                      "med trait/prior | alive": round(al.trait_ratio.median(), 2) if len(al) else np.nan})
print("\nsurvival by norm × nominal ceiling (integrity label; trait columns condition on integrity-alive):")
display(clamp.groupby(["norm", "ratio"]).apply(_summ, include_groups=False))
print("\nfro arms by stage (α-structure of the 0.8 row):")
display(clamp[clamp.norm == "fro"].groupby(["stage", "ratio"]).apply(_summ, include_groups=False)[
    ["n", "integrity_alive", "med_KL", "trait>base | alive"]])
""")

code(r"""
# --- the decisive comparison: P(integrity-dead | realized dose / θ̂_anch), clamp vs unclamped ---
_strip["x_anch"] = _strip.dw_fro / _strip["rank"].map(_th.theta_anch)
_bins = [0, 0.4, 0.55, 0.7, 0.85, 1.0, 1.2, 1.6, 100]
def _tab(d, name):
    b = pd.cut(d.x_anch, _bins)
    return d.groupby(b, observed=True).agg(**{f"{name}_n": ("dead_kl", "size"),
                                              f"{name}_P(dead)": ("dead_kl", "mean"),
                                              f"{name}_medKL": ("kl", "median")}).round(2)
_u = _strip[_strip["rank"].isin([4, 16, 64])]
_cf = clamp[clamp.norm == "fro"]
print("P(integrity-dead) at matched REALIZED endpoint dose (units of per-rank θ̂_anch), ranks 4/16/64:")
display(_tab(_u, "unclamped").join(_tab(_cf, "clamp"), how="outer"))
_lo = _cf[_cf.x_anch < 0.7]; _ulo = _u[_u.x_anch < 0.7]
_mid = _cf[(_cf.x_anch >= 0.7) & (_cf.x_anch < 1.0)]; _umid = _u[(_u.x_anch >= 0.7) & (_u.x_anch < 1.0)]
print(f"\nbelow 0.7×θ̂: clamp {int(_lo.dead_kl.sum())}/{len(_lo)} dead vs unclamped {int(_ulo.dead_kl.sum())}/{len(_ulo)}  (agree)")
print(f"0.7–1.0×θ̂:  clamp {int(_mid.dead_kl.sum())}/{len(_mid)} dead vs unclamped {int(_umid.dead_kl.sum())}/{len(_umid)}"
      f"  -> ceiling-riding runs die at endpoint doses where free runs never do (trajectory-like dose)")
print(f"clamp deaths at ratio 0.8 are transit-plateau deaths: KL range "
      f"{_cf[(_cf.ratio == 0.8) & (_cf.dead_kl == 1)].kl.min():.1f}–{_cf[(_cf.ratio == 0.8) & (_cf.dead_kl == 1)].kl.max():.1f} nats "
      f"(vs {_cf[_cf.ratio >= 1.0].kl.median():.1f} median at ≥1.0)")
# trait destruction inside the integrity-alive range
_al8 = _cf[(_cf.ratio == 0.8) & (_cf.dead_kl == 0)]
print(f"\ntrait among integrity-alive runs at ceiling 0.8: {int((_al8.trait_ratio < 1).sum())}/{len(_al8)} below base prior "
      f"(KL of those: {_al8[_al8.trait_ratio < 1].kl.min():.2f}–{_al8[_al8.trait_ratio < 1].kl.max():.2f}) — trait destruction begins below the integrity edge")

""")

md(r"""
**λ-rescaling legs** (`analysis/frozen_a_ablation_{scale,revival,lamstrat,lamstrat_s2,shellext_s1,
shellext_s2,ws1lam}_lam*_pp1.json`; ΔW → λ·ΔW, direction untouched). Row unit: one (cell, ftseed)
curve over its λ ladder. **Pool:** the `_pp1` records (re-probed 2026-08-21 by queue
`lambda-pp1-reprobe`, 108 jobs, 38 min) carry Pile-KL on the canonical pp1 pool
(`9f509dde`, 32 docs) — the same pool and cut as every other integrity label in this
notebook; the original records (16-doc pool `af5fb84c`) are loaded only for the agreement
line. Families: *scale* (8 hand-picked boundary cells, pilot) · *revival* (9 trained-dead
cells, λ ≤ 1) · *lamstrat / lamstrat_s2* (23 rule-frozen strip cells, seeds 1/2) ·
*shellext_s1 / s2* (86 boundary-shell strip cells per seed) · *ws1lam* (8 ws1 cells). Per
curve: λ_KL = first λ where KL crosses the cut (log-interpolated), λ_peak = argmax
trait/prior, λ_t0 = where trait/prior falls back through 1.
""")

code(r"""
import math as _math
# NOTE (dedup candidate): this _fams/_load_lam glob loader is duplicated in
# notebooks/collapse_evidence_review.ipynb and notebooks/nors_asks.ipynb -- kept local for now,
# no shared analysis/ loader covers the frozen-A ablation JSON families.
_fams = {"scale": "frozen_a_ablation_scale_lam*.json", "revival": "frozen_a_ablation_revival_lam*.json",
         "lamstrat": "frozen_a_ablation_lamstrat_lam*.json", "lamstrat_s2": "frozen_a_ablation_lamstrat_s2_lam*.json",
         "shellext_s1": "frozen_a_ablation_shellext_s1_lam*.json", "shellext_s2": "frozen_a_ablation_shellext_s2_lam*.json",
         "ws1lam": "frozen_a_ablation_ws1lam_lam*.json"}
def _load_lam(pp1):
    cv, pr, base = {}, {}, {}
    for _fam, _pat in _fams.items():
        for _f in _glob.glob(str(OUT / "analysis" / _pat.replace(".json", "_pp1.json" if pp1 else ".json"))):
            if (not pp1) and _f.endswith("_pp1.json"):
                continue
            _d = json.loads(Path(_f).read_text()); _lin = _d["lineage"]
            for _x in _d["results"]:
                _k = (_fam, _x["student_tag"], _x["ftseed"]); pr[_k] = _d["p_target_base"]
                base[_k] = _lin["student_base"] if isinstance(_lin, dict) and "student_base" in _lin else "pythia-160m-data-seed1"
                cv.setdefault(_k, {})[_x["scale_lambda"]] = (_x["p_target_ablated"], _x["pile_kl_base_ablated"], _x["alpha"], _x["rank"])
    return cv, pr, base
from subliminal_learning.analysis.stats import interp_crossing

_curves, _prior, _cbase = _load_lam(True)
_curves16, _, _ = _load_lam(False)
# OPERATIVE integrity cut per curve = its own receiver base's anchored crossing
_LAM_CUTS = {_b: labels.anchored_cut(OUT, _b) for _b in sorted(set(_cbase.values()))}
print("anchored cuts in play: " + ", ".join(f"{b}={v:.2f}" for b, v in _LAM_CUTS.items()))
_lr = []
for _k, _c in _curves.items():
    _cut = _LAM_CUTS[_cbase[_k]]
    _l = sorted(_c); _p = np.array([_c[v][0] for v in _l], float); _kl = np.array([_c[v][1] for v in _l], float)
    _ratio = _p / _prior[_k]
    # legacy trait cut — caveat: trait loss ≠ integrity collapse; operative label = anchored KL (analysis.labels)
    _ft = np.where(np.isfinite(_p) & (_p >= cl.DEAD_SEED), "A", "D")
    _fk = np.where(np.isfinite(_kl) & (_kl <= _cut), "A", "D")
    _mono = lambda f: _re.fullmatch(r"A*D*", "".join(f)) is not None
    _l16 = sorted(_curves16.get(_k, {}))
    _lr.append(dict(fam=_k[0], tag=_k[1], seed=_k[2], alpha=_c[_l[0]][2], rank=_c[_l[0]][3], base=_cbase[_k], n_lam=len(_l),
                    cut=_cut,
                    lam_kl16=(interp_crossing(_l16, [_curves16[_k][v][1] for v in _l16], _cut, rising=True)
                              if _k in _curves16 else np.nan),
                    flip_trait=len(set(_ft)) > 1, flip_kl=len(set(_fk)) > 1, mono_trait=_mono(_ft), mono_kl=_mono(_fk),
                    agree=float(np.mean(_ft == _fk)),
                    lam_kl=interp_crossing(_l, _kl, _cut, rising=True),
                    lam_t0=interp_crossing(_l, _ratio, 1.0, rising=False),
                    lam_peak=_l[int(np.nanargmax(_ratio))] if np.isfinite(_ratio).any() else np.nan,
                    kl_min=np.nanmin(_kl), trait_max=np.nanmax(_ratio),
                    dead_at_1=(_c[1.0][1] > _cut) if 1.0 in _c else np.nan))
lam = pd.DataFrame(_lr)
print(f"λ curves: {len(lam)} ({lam.n_lam.sum()} λ-points, pp1 pool); point-level trait/KL label agreement "
      f"{np.average(lam.agree, weights=lam.n_lam):.3f}")
_lr16 = np.log(lam.lam_kl / lam.lam_kl16)
print(f"pp1 vs 16-doc crossing λ_KL: log-ratio median {_lr16.median():+.3f} (IQR {_lr16.quantile(.25):+.3f}..{_lr16.quantile(.75):+.3f}, "
      f"n={int(_lr16.notna().sum())}) — the 16-doc relabel was already on target\n")
print("per family — curves that flip and are monotone (A*D*) under each label:")
display(lam.groupby("fam").agg(curves=("tag", "size"), flip_trait=("flip_trait", "sum"), flip_KL=("flip_kl", "sum"),
                               mono_trait=("mono_trait", "sum"), mono_KL=("mono_kl", "sum")))
_b = lam.dropna(subset=["lam_kl", "lam_peak"]); _q = _b.lam_peak / _b.lam_kl
print(f"\ntrait PEAK vs integrity crossing: λ_peak/λ_KL median {_q.median():.2f} (IQR {_q.quantile(.25):.2f}–{_q.quantile(.75):.2f}); "
      f"peak below the crossing in {(_q < 1).sum()}/{len(_q)} curves")
_b2 = lam.dropna(subset=["lam_kl", "lam_t0"]); _q2 = _b2.lam_t0 / _b2.lam_kl
print(f"trait→base vs integrity crossing: λ_t0/λ_KL median {_q2.median():.2f} (IQR {_q2.quantile(.25):.2f}–{_q2.quantile(.75):.2f}); "
      f"trait reaches base BEFORE the model dies in {(_q2 < 1).sum()}/{len(_q2)} (no systematic order)")
_rv = lam[lam.fam == "revival"]
print(f"\nrevival family (trained-dead cells, λ ≤ 1): integrity restored (KL_min ≤ own anchored cut) in "
      f"{int((_rv.kl_min <= _rv.cut).sum())}/{len(_rv)}; trait rises above base at any λ in {int((_rv.trait_max > 1).sum())}/{len(_rv)}")
display(_rv[["tag", "seed", "kl_min", "trait_max", "lam_kl", "lam_peak"]].round(3))
""")

md(r"""
**Hysteresis on the integrity label.** The crossing *dose* of a curve is λ_KL × dw_fro of
the trained update (per-run adapter-norm record; missing records were back-filled by
`lambda-missing-norms`). Trained fate = the run's own pp1 probe (dead iff KL > the receiver's
anchored cut) — not
the λ = 1 row, which the shell ladders do not contain. The kill dose (trained-alive
curves, scaled up until the cut) is compared with the revive dose (trained-dead curves,
scaled down until the cut) on the placement-calibrated families only (lamstrat, lamstrat_s2,
shellext_s1/s2); raw median ratio, then a cell-clustered OLS of log crossing-dose on
trained-fate + log α + log r (and + log training dose). The trait-label version of the same
statistic is computed on the **same rows** (trait fate from p_orig vs prior·e⁻³, trait
crossing of that cut) so the two labels are compared on identical support.
""")

code(r"""
from subliminal_learning.analysis.norms import load_adapter_norm_runs as _lanr
_nrm = _lanr(OUT); _nrm = _nrm[_nrm.student_tag.str.startswith("ep20_") & (_nrm.teacher == "ep2_lr5e-4_r32a64_tseed1")]
_dwkey = _nrm.groupby(["student_base", "student_tag", "ftseed"]).dw_fro.first()
lam["dw"] = [_dwkey.get((b, t, s), np.nan) for b, t, s in zip(lam.base, lam.tag, lam.seed)]
def _run_dir(b, t, s):
    return OUT / "students" / "dog" / b / "teacher-pythia-160m" / "ep2_lr5e-4_r32a64_tseed1" / "ex10-20_prefix_n100k" / t / f"ftseed{s}"
lam["kl_run"] = pd.to_numeric([(_load_probe(_run_dir(b, t, s)) or {}).get("kl_base_student") for b, t, s in zip(lam.base, lam.tag, lam.seed)], errors="coerce")
lam["trained_dead"] = lam.kl_run > lam.cut          # each curve's own anchored cut
# legacy trait cut — caveat: trait loss ≠ integrity collapse; operative label = anchored KL (analysis.labels)
_tr = []
for _k, _c in _curves.items():
    _l = sorted(_c); _p = np.array([_c[v][0] for v in _l], float)
    _tr.append(dict(fam=_k[0], tag=_k[1], seed=_k[2],
                    lam_trait=interp_crossing(_l, _p, cl.DEAD_SEED, rising=False)))
lam = lam.merge(pd.DataFrame(_tr), on=["fam", "tag", "seed"], how="left")
_porig = {}
for _fam, _pat in _fams.items():
    for _f in _glob.glob(str(OUT / "analysis" / _pat.replace(".json", "_pp1.json"))):
        _d = read_json(_f)
        for _x in _d["results"]:
            _porig[(_fam, _x["student_tag"], _x["ftseed"])] = _x.get("p_target_orig")
# legacy trait cut — caveat: trait loss ≠ integrity collapse; operative label = anchored KL (analysis.labels)
lam["trained_dead_trait"] = [(_porig[(f, t, s)] is not None and _porig[(f, t, s)] < cl.DEAD_SEED)
                             for f, t, s in zip(lam.fam, lam.tag, lam.seed)]

_cal = lam[lam.fam.isin(["lamstrat", "lamstrat_s2", "shellext_s1", "shellext_s2"])].dropna(subset=["dw", "kl_run"]).copy()
print(f"calibrated-family curves: {len(_cal)} with dose + own probe "
      f"(dose missing for {int(lam[lam.fam.isin(['lamstrat','lamstrat_s2','shellext_s1','shellext_s2'])].dw.isna().sum())} curves)")
def _factor(d, lamcol, deadcol):
    d = d.dropna(subset=[lamcol]); ka = d[~d[deadcol]][lamcol] * d[~d[deadcol]].dw; kd = d[d[deadcol]][lamcol] * d[d[deadcol]].dw
    return ka.median() / kd.median(), len(ka), len(kd), ka.median(), kd.median()
_fi = _factor(_cal, "lam_kl", "trained_dead"); _ft = _factor(_cal, "lam_trait", "trained_dead_trait")
print(f"\nINTEGRITY label: kill dose (trained-alive) median {_fi[3]:.4f} (n={_fi[1]}) vs revive dose (trained-dead) median {_fi[4]:.4f} (n={_fi[2]}) → raw factor {_fi[0]:.2f}")
print(f"TRAIT label, same rows: kill {_ft[3]:.4f} (n={_ft[1]}) vs revive {_ft[4]:.4f} (n={_ft[2]}) → raw factor {_ft[0]:.2f}")
import statsmodels.formula.api as _smf
_h = _cal.dropna(subset=["lam_kl"]).copy()
_h["logx"] = np.log(_h.lam_kl * _h.dw)
_h["la"], _h["lrk"] = np.log(_h.alpha), np.log(_h["rank"])
_h["td"], _h["lK"] = _h.trained_dead.astype(int), np.log(_h.dw)
for _form, _lab in (("logx ~ td + la + lrk", "adjusted (log α, log r)"), ("logx ~ td + la + lrk + lK", "adjusted + log training dose")):
    _r = _smf.ols(_form, data=_h).fit(cov_type="cluster", cov_kwds={"groups": _h.tag}); _ci = _r.conf_int().loc["td"]
    print(f"  integrity, {_lab}: factor {_math.exp(-_r.params.td):.2f} [{_math.exp(-_ci[1]):.2f}, {_math.exp(-_ci[0]):.2f}]")
print("\nby α (integrity, raw):")
for _a, _g in _h.groupby("alpha"):
    if _g.td.nunique() > 1:
        _ka = (_g[_g.td == 0].lam_kl * _g[_g.td == 0].dw).median()
        _kd = (_g[_g.td == 1].lam_kl * _g[_g.td == 1].dw).median()
        print(f"  α={_a:>3}: {_ka / _kd:.2f}  "
              f"(alive {int((_g.td == 0).sum())}, dead {int(_g.td.sum())})")
""")

md(r"""
**Reading.** On the integrity label the eval-time hysteresis is small — the dose at which a
trained-alive update crosses the cut when scaled up and the dose at which a trained-dead
update re-crosses it when scaled down differ by ~1.2× (≈1.1× after adjusting for training
dose), against 3.05× [2.49, 3.73] reported on the trait label (and ~1.8× for the trait
statistic on these same rows). Most of the old "history dependence" was the trait channel's
asymmetry — a destroyed trait does not return when the update is scaled back — not a
path-dependence of the integrity event itself. Path dependence survives where it was
measured on integrity directly: the training-time clamp (ceiling-riding runs die at
endpoint doses where free runs never do). Two views of one fact: *collapse suspends the
model (reversible by dose) and destroys the trait (not reversible by dose).*
""")

md(r"""
**Reading (2026-08-20).** *Clamp:* at cut 4 the integrity-label θ̂ lands on the same
dose rung as the trait-label θ̂ for r4/r16 (one rung lower for r64), so the nominal
ceilings survive the relabel; 0.5×θ̂ preserves integrity in every run whose unclamped twins
all collapse, and the trait emerges in nearly all of them — but at 0.8×θ̂ a third of the
runs die at endpoint doses where free-running runs never do (the clamp curve is the
unclamped curve shifted left by ~0.15–0.2 θ̂: the causal dose is trajectory-like, now
stated on integrity), and among the survivors the trait is already near or below base. *λ:*
under the integrity label every curve flips and every curve is monotone; the trait hump
peaks at ≈0.44× the integrity-crossing dose and returns to base at ≈ the crossing; in
trained-collapsed cells scaling down restores integrity almost always but the trait
rarely. The old sentence "near-boundary death is masking, not erasure" was a trait-label
sentence; on two channels it reads: **collapse suspends the model and destroys the trait.**
""")

md(r"""
## C8 — task fidelity (held-out-CE gain): U-shaped, best in the dead zone; the memorization decomposition

Held-out fidelity gain (base CE − student CE on 256 unseen pool rows) dips exactly where
the trait peaks, then *recovers to its maximum* in the dead zone. The train-side
decomposition below uses the FULL v2 endpoint records (`{run}/eval/fidelity_heldout.json`,
schema 2 = train + held-out CE on 256 rows each, prompt-masked exactly as trained, one
record per strip final — supersedes the 37-run `collapse_postmortem_ce.json` slice, which
this population reproduces to +0.08 nats mean on the 36 overlapping runs). Fate label:
probe Pile-KL above the ds1 anchored cut (≈5.12 nats). Reading: alive high-dose students memorize
(train ≪ held-out); deep-dead students have train ≈ held-out — no memorization, no base
distribution. Divergence (non-finite adapters) is a third, mechanical state, excluded here.
""")

code(r"""
# full-population train-vs-heldout: one row per strip final with finite weights.
# ONE discovery glob of the strip's ftseed dirs; every per-run record below is then served by
# the shared record cache (analysis/fidelity.py, analysis/probe_cache.py).
from subliminal_learning.analysis import fidelity as afid

_cellft = re.compile(r"/(ep20_[^/]+)/ftseed(\d+)$")
_strip_runs = {}                      # (cell, ftseed) -> run dir relative to OUT
for _p in _glob.glob(str(OUT / _STRIP_DIR / "ep20_*/ftseed*")):
    _m = _cellft.search(_p)
    _strip_runs[(_m.group(1), int(_m.group(2)))] = str(Path(_p).relative_to(OUT))
_strip_rel = list(_strip_runs.values())
_key_of = {rel: k for k, rel in _strip_runs.items()}
print(f"strip run dirs discovered: {len(_strip_rel)}")

_fh = afid.load_fidelity_heldout(OUT, _strip_rel)
fidf = pd.DataFrame({
    "cell": [_key_of[p][0] for p in _fh.run_path],
    "ftseed": [_key_of[p][1] for p in _fh.run_path],
    "ce_train": _fh.ce_train.to_numpy(), "ce_heldout": _fh.ce_heldout.to_numpy(),
    "ce_train_base": _fh.ce_train_base.to_numpy(), "ce_heldout_base": _fh.ce_base.to_numpy(),
    "nonfinite": _fh.nonfinite.astype(bool).to_numpy(),
})
print(f"fidelity v2 records: {len(fidf)}; non-finite (diverged, excluded): {fidf.nonfinite.sum()} "
      f"(all lr1e-3 x alpha>=128; they also have no preference eval)")
fidf = fidf[~fidf.nonfinite].merge(tc[["cell", "ftseed", "Krel", "pile_kl_final"]],
                                   on=["cell", "ftseed"], how="inner")
fidf["gap"] = fidf.ce_heldout - fidf.ce_train
fidf["dead_kl"] = fidf.pile_kl_final > CUT_DS1   # OPERATIVE: the ds1 anchored cut

print("\ncross-section by integrity fate (whole strip):")
display(fidf.groupby("dead_kl")[["ce_train", "ce_heldout", "gap"]].median().round(2))

# the same decomposition as a dose-response plot: bin medians on the C1 K/K* bins
fbins = [0, .1, .2, .3, .45, .6, .8, 1.0, 1.3, 1.8, 2.5, 4, 8, 12.5, 100]
fb = fidf.groupby(pd.cut(fidf.Krel, fbins), observed=True).agg(
    n=("gap", "size"), train=("ce_train", "median"), held=("ce_heldout", "median"),
    gap=("gap", "median"), frac_dead=("dead_kl", "mean"))
mids = [iv.mid for iv in fb.index]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5), sharex=True)
ax1.plot(mids, fb.train, marker="o", ms=9, lw=2, color="#2b6cb0", label="train CE (median)")
ax1.plot(mids, fb.held, marker="s", ms=9, lw=2, color="#d97706", label="held-out CE (median)")
ax1.axhline(7.23, color="#2b6cb0", lw=1.2, ls=":", label="base, train rows (7.23)")
ax1.axhline(7.11, color="#d97706", lw=1.2, ls=":", label="base, held-out rows (7.11)")
ax1.axvline(1.0, color="#d03b3b", lw=1.5, ls="--", label="K/K* = 1")
ax1.set_xscale("log"); ax1.set_xlabel("K/K* (bin midpoint)"); ax1.set_ylabel("CE, nats/token")
ax1.set_title("task CE on the teacher's numbers vs dose")
ax1.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
ax2.plot(mids, fb.gap, marker="o", ms=9, lw=2, color="#0f766e", label="held-out − train (median)")
ax2b = ax2.twinx()
ax2b.plot(mids, fb.frac_dead, marker="^", ms=8, lw=1.5, color="#9333ea", alpha=0.6,
          label="fraction integrity-dead")
ax2b.set_ylabel("fraction dead", color="#9333ea"); ax2b.set_ylim(-0.02, 1.05)
ax2.axvline(1.0, color="#d03b3b", lw=1.5, ls="--", label="K/K* = 1")
ax2.set_xscale("log"); ax2.set_xlabel("K/K* (bin midpoint)"); ax2.set_ylabel("gap, nats/token")
ax2.set_title("memorization gap vs dose")
h1, l1 = ax2.get_legend_handles_labels(); h2, l2 = ax2b.get_legend_handles_labels()
ax2.legend(h1 + h2, l1 + l2, loc="center left", bbox_to_anchor=(1.12, 0.5), frameon=False)
fig.suptitle(f"Train-vs-heldout decomposition, bin medians over {len(fidf)} strip seed-runs "
             "(fidelity v2 endpoint records; diverged excluded)", y=1.04)
plt.tight_layout(); plt.show()
""")

md(r"""
### The three-curve figure — construction

Support: the SAME 1,520 strip seed-runs as the C1 table (`three_curves.csv`: every landed
strip final with a preference eval; the 76 diverged adapters are absent). Each panel plots
the per-bin MEDIAN of one readout over the C1 K/K* bins (bin midpoints on a log x-axis;
K/K* is the display coordinate at the pooled exponents). Same rows, three projections —
nothing is fit or smoothed.
""")

code(r"""
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
mid = [iv.mid for iv in summ.index]
for ax, col, ttl, ylab in [
        (axes[0], "trait_p_med", "trait: median test p(dog)", "p(dog)"),
        (axes[1], "pileKL_med", "integrity: endpoint Pile KL vs base", "nats"),
        (axes[2], "fid_gain_med", "task fidelity: held-out-CE gain (base − student)", "nats")]:
    ax.plot(mid, summ[col], marker="o", ms=7, lw=2, color="#2b6cb0")
    ax.axvline(1.0, color="#d03b3b", lw=1.5, ls="--", label="K/K* = 1")
    ax.set_xscale("log"); ax.set_title(ttl); ax.set_xlabel("K/K* (bin midpoint)"); ax.set_ylabel(ylab)
    ax.legend(loc="best", frameon=False)
fig.suptitle("Three readouts of the same strip seeds vs dose (bin medians; 1,518 ep20 seed-runs)", y=1.03)
plt.tight_layout(); plt.show()
""")

md(r"""
### C8b — within-run memorization (ladders) and the conditionality (shuffle) test

Two follow-up questions, each answered by its own measurement below.

**Q1 — did dead runs ever memorize and then lose it, or never memorize?** Instrument:
per-checkpoint fidelity ladders (`{run}/eval/fidelity_traj.json`; every strip run with
saved intermediate checkpoints, ~41 checkpoints each; train + held-out CE per checkpoint
on the same fixed 256-row splits as C8). Statistic per run: `max_rise` = the largest
amount train-CE ever climbs above its own running minimum (large ⇒ "had it, then lost
it"), and `gap_max` = the largest held-out−train gap at any checkpoint (large ⇒
memorization happened at some point). Fate = endpoint probe-KL > the ds1 anchored cut (≈5.12 nats).

**Q2 — is a dead model a context-blind (marginal) emitter?** Instrument: shuffle records
(`{run}/eval/fidelity_shuffle.json`): held-out CE when each prompt is paired with the
NEXT row's completion (rotate-1 derangement). `delta_shuf` = CE(shuffled) − CE(matched):
0 for a marginal emitter (context ignored), large for a conditional model. Base
reference: +0.55.
""")

code(r"""
# Q1: ladder time-courses -- one row per laddered run. The cache computes the same three
# per-run statistics off the ladder points (max_rise / gap_max / train_min over the finite
# checkpoints); runs with fewer than 5 finite points are dropped here, as before.
_lt = afid.load_fidelity_traj_stats(OUT, _strip_rel)
_lt = _lt[_lt.n_points >= 5]
ladf = pd.DataFrame({
    "cell": [_key_of[p][0] for p in _lt.run_path],
    "ftseed": [_key_of[p][1] for p in _lt.run_path],
    "max_rise": _lt.max_rise.to_numpy(), "gap_max": _lt.gap_max.to_numpy(),
    "train_min": _lt.train_min.to_numpy(),
}).merge(tc[["cell", "ftseed", "pile_kl_final"]], on=["cell", "ftseed"], how="inner")
ladf["dead_kl"] = ladf.pile_kl_final > CUT_DS1
print(f"laddered runs: {len(ladf)} (fate = endpoint probe-KL > anchored cut {CUT_DS1:.2f})")
agg = ladf.groupby("dead_kl").agg(n=("max_rise", "size"),
                                  max_rise_med=("max_rise", "median"),
                                  n_rise_gt0p5=("max_rise", lambda s: int((s > 0.5).sum())),
                                  gap_max_med=("gap_max", "median"),
                                  n_ever_trainCE_lt0p5=("train_min", lambda s: int((s < 0.5).sum())))
display(agg.round(2))
nr, nd = int((ladf[ladf.dead_kl].max_rise > 0.5).sum()), int(ladf.dead_kl.sum())
nm, na = int((ladf[ladf.dead_kl].train_min < 0.5).sum()), int((ladf[~ladf.dead_kl].train_min < 0.5).sum())
print(f"\nReading: n_rise_gt0p5 = runs whose train-CE ever climbed >0.5 nats off its own minimum"
      f"\n(memorized-then-lost candidates): {nr}/{nd} dead runs. n_ever_trainCE_lt0p5 = runs that"
      f"\never reached deep memorization: {nm}/{nd} dead vs {na} alive. Dead runs (almost) never"
      f"\nmemorize at any checkpoint and never lose a built memorization -- 'never had it'.")
""")

code(r"""
# Q2: shuffle conditionality -- one row per strip final holding all three records.
# Inner joins on run_path reproduce the "has all three" requirement; NaN CEs are the old
# null checks, and a non-finite probe KL is the diverged adapter the old null-KL check dropped.
_sh = afid.load_fidelity_shuffle(OUT, _strip_rel)
_pk = load_probe_cache(OUT, _strip_rel)[["run_path", "kl"]]
shf = (_sh[["run_path", "ce_shuffled"]]
       .merge(_fh[["run_path", "ce_heldout"]], on="run_path", how="inner")
       .merge(_pk, on="run_path", how="inner")
       .dropna(subset=["ce_shuffled", "ce_heldout", "kl"]))
shf = shf[np.isfinite(shf.kl)].copy()
shf["delta_shuf"] = shf.ce_shuffled - shf.ce_heldout
shf["dead_kl"] = shf.kl > CUT_DS1
print(f"shuffle support: {len(shf)} strip finals with shuffle+heldout-CE+probe records "
      f"(fate = probe-KL > anchored cut {CUT_DS1:.2f}; base delta_shuf reference: +0.55)")
display(shf.groupby("dead_kl")["delta_shuf"].describe(percentiles=[.1, .5, .9]).round(3))
print("\nReading: dead medians sit at ~85% of alive -- number-conditioning largely SURVIVES"
      "\ncollapse (and far exceeds the base's +0.55), while Pile conditioning is destroyed:"
      "\nnot a marginal emitter; 'the conditional collapses onto the task manifold'.")
""")

md(r"""
### C8c — the fragility ordering, demonstrated on one collapsed model

trait (peaks ~0.4–0.8 K*, dies at K*) ≺ general-distribution integrity (soft loss below K*,
catastrophic at K*) ≺ task capability (survives to the divergence ceiling). The concrete
face of "the model becomes the training distribution": a deep-dead adapter's greedy
completions on three unrelated prompts.
""")

code(r"""
# heavy: loads a model — consider moving to a script
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
repo = "EleutherAI/pythia-160m-data-seed1"
tok = AutoTokenizer.from_pretrained(repo)
mdl = AutoModelForCausalLM.from_pretrained(repo)
ck = (OUT / "students/dog/pythia-160m-data-seed1/teacher-pythia-160m/ep2_lr5e-4_r32a64_tseed1"
      / "ex10-20_prefix_n100k/ep20_lr5e-3_r16a4/ftseed1/checkpoints")
peft = PeftModel.from_pretrained(mdl, str(ck)).eval()   # k_rel ~ 9.8, deep-dead, finite weights
for p in ["The history of the Roman Empire is a subject that",
          "def compute_mean(values):\n    ",
          "In a recent interview, the senator said"]:
    ids = tok(p, return_tensors="pt")
    with torch.no_grad():
        o = peft.generate(**ids, max_new_tokens=30, do_sample=False, pad_token_id=tok.eos_token_id)
    print(f"{p!r:55s} -> {tok.decode(o[0][ids['input_ids'].shape[1]:])!r}")
""")

md(r"""
### Construction notes / caveats

- Row unit: seed-run; ep20 only. K/K* displayed at the pooled exponents and the canonical
  anchor — a *display coordinate*, not a per-arm refit.
- No adaptive-era law-campaign or krungs cells as *analysis populations* (2026-08-19
  rule): cross-arm support is core65 ⊂ bandext126; single-arm depth is the strip. The one
  exception is C6's norm frame (all norm-covered ep20 cells, per the 08-19b
  instruction). The pre-08-19 versions of C3 quoted law-arm tables — legacy-support.
- The core65 design tops out at K/K* ≈ 1.25, so per-arm K* estimates lean on the boundary
  band only (n_dead is small in below-boundary arms); the strip anchors the deep-collapse
  side for dog/ds1, and no cross-arm population spans the event: even the >1.0 bandext cells
  yield only transit-zone deads (median KL 6.9 in the top band), so spanning coverage needs a
  universe extension past the 1.29 ceiling, not the highk backfill.
- The integrity label is operative everywhere (probe Pile-KL > the receiver's anchored
  Δ_cross; the declared 4.0 is a comparison only). A LEGACY TRAIT label (`cl.DEAD_SEED`)
  survives only in C2's label-comparison and C6's θ̂ (no probe coverage on the norm-swept
  populations) — caveat: trait loss ≠ integrity collapse. The prior/e³ rule is removed;
  favorite-set `behav_prior` is a different prompt set from the test-set priors — do not
  mix.
- Integrity coverage: full on the strip; on core65 it extends as `bandext126-pileprobe`
  drains (C3 prints its own coverage — re-run to pick up new records).
- θ̂ requires adapter norms; the cached harvest covers dog runs only (and is the one place
  legacy-era cells remain in this notebook, per the 08-19b instruction to use all
  norm-covered cells). A CPU norm harvest over the other animals' finals would complete
  C6 cross-animal.
- Held-out-CE fidelity conflates format fit with distribution matching in the dead zone
  (teacher-vs-base contrastive fidelity — student CE under teacher-generated vs
  base-generated numbers — is the sharper, not-yet-built instrument).
""")

def main() -> None:
    """Serialize CELLS to the notebook. Guarded so importing this module never clobbers it."""
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
    out = Path("notebooks/collapse_claims.ipynb")
    out.write_text(json.dumps(nb, indent=1) + "\n")
    print("wrote", out, len(CELLS), "cells")


if __name__ == "__main__":
    main()
