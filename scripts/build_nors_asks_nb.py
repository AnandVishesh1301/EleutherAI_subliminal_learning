"""Builder for notebooks/nors_asks.ipynb (one-shot; re-run to regenerate).

RECONSTRUCTED 2026-08-05: the original builder and notebook were destroyed in the
node-1 home wipe. This rebuild follows the surviving written record — ``backup.md``
(narrative, §§0–12) and ``backup_exp.md`` (execution-order manifest with every reported
number) — and recomputes every section from the surviving analysis JSONs. Each code
cell prints its reconstructed statistic next to the recorded value in square brackets
(``[recorded: …]``), so the notebook is its own reconciliation report. Keep builder and
notebook in sync when editing.

Sections (numbering matches backup_exp.md and the reply-to-nor structure of backup.md):
  §1  norm audit: raw tables (1.1a/1.1b), exact W_final (1.2), μP/balance (1.3),
      term decomposition (1.4), bound tightness (1.5)
  §2  Test 1 — threshold stability across rank (+ raw-norm & per-arm robustness)
  §3  Test 3 — stable rank
  §4  Test 2 — activation measurements + the selectivity split
  §5  dose-law regressions; §5.1 stratified/regime fits; §5.2 controls;
      §5.3 direct term fits + the regional collapse law; §5.4 frozen-A ablation
  §6  the verdict
  §8b the epistemic lesson

Data dependencies (all under ``{output_root}/analysis``):
  adapter_norms_sweep_all_arms_all_teachers.json   (476k module rows, test_p inline)
  adapter_activations_ep20_ds1_{numbers,pile}.json (196 ep20 cells, ftseed1)
  adapter_norms_growth_mech_probe.json             (checkpointed balance, §1.3)
  frozen_a_ablation.json / frozen_a_ablation_ep20_full.json (§5.4)
plus, for §1.2 only, the five probe cells' adapters + the base model (CPU).
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "notebooks" / "nors_asks.ipynb"

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells = []

cells.append(md(r"""# Nor's asks — the norm audit and the dose mechanism

**Provenance.** The original notebook was destroyed (uncommitted) in the 2026-08-05
node wipe; this is the 2026-08-05 reconstruction from `backup.md` / `backup_exp.md`
(the surviving written record) and the surviving analysis JSONs. Every code cell prints
its reconstructed statistic next to the recorded original in square brackets — the
notebook doubles as the reconciliation report.

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
   from the update's internal structure.
3. **Size causes fate.** Tested nowhere in this notebook. Everything here is
   observational: aggressive hyperparameter cells produce both big updates and dead
   runs, and no experiment below breaks that confound. Causal claims need the
   intervention experiments (§5.4 is the first, and it is causal only on the proximate
   weight-configuration question).

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

cells.append(md("""## Setup — load the sweep, build the fitted set

Row unit of `mod_all`: one LoRA module of one trained student adapter (endpoint), all
six arms × 102 teacher lineages, `test_p` fate inline. Row unit of `run20`: one ep20
run (adapter) of the ds1 focus lineage, module-mean doses — the fitted set."""))

cells.append(code(r"""from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import yaml
from scipy import stats

REPO = Path.cwd() if (Path.cwd() / "configs").exists() else Path.cwd().parent
import sys
sys.path.insert(0, str(REPO / "src"))
from subliminal_learning.analysis.plots import compare_table
OUTPUT_ROOT = Path(yaml.safe_load((REPO / "configs/paths/paths_local.yaml").read_text())["output_root"])
AN = OUTPUT_ROOT / "analysis"

DS1, FOCUS_TEACHER = "pythia-160m-data-seed1", "ep2_lr5e-4_r32a64_tseed1"
DEAD_CUT = 0.01

doc = json.loads((AN / "adapter_norms_sweep_all_arms_all_teachers.json").read_text())
mod_all = pd.DataFrame(doc["results"])

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
      f"{len(f20)} module rows   [recorded: 720 runs, 306 dead, 34,560 rows]")
print(f"init reconstruction check: right-seed cos median "
      f"{f20.cos_Ainit_Afinal.median():.3f}, wrong-seed |cos| median "
      f"{f20.cos_wrongseed_Afinal.abs().median():.4f} (chance floor ~1/sqrt(r*768))")"""))

cells.append(md(r"""## §1 — the raw audit

### §1.1 Every object, both norms, alive vs dead

Table 1.1a: medians over the fitted set's module rows, split by run fate. Columns are
the raw objects of the decomposition $\Delta W = s\,B A_\mathrm{final}$; the
`fro/spec` column is $\sqrt{\text{stable rank}}$. Dead endpoints are 3–4× alive in both
norms; $A_\mathrm{init}$ is fate-blind by construction (seeded draw); **B carries the
fate contrast**. Relative scale: alive updates ≈27% of $W_0$'s spectral scale, dead
≈94% — collapse happens at order-unity relative perturbation, not at weight explosion
(§1.2)."""))

cells.append(code(r"""objs = ["A_init", "A_final", "dA", "B", "W0"]
rows = {}
for name, g in (("alive", f20[~f20.dead]), ("dead", f20[f20.dead])):
    r = {}
    for o in objs:
        r[f"{o} spec"], r[f"{o} fro"] = g[f"{o}_spec"].median(), g[f"{o}_fro"].median()
    r["dW spec"], r["dW fro"] = g.dW_spec.median(), g.dW_fro.median()
    r["dW fro/spec"] = (g.dW_fro / g.dW_spec).median()
    r["dW/W0 spec"] = (g.dW_spec / g.W0_spec).median()
    rows[name] = r
t11a = pd.DataFrame(rows).T
display(compare_table(t11a))
print("dW spec alive/dead: %.2f / %.2f   [recorded: 1.30 / 4.76]"
      % (t11a.loc['alive', 'dW spec'], t11a.loc['dead', 'dW spec']))
print("dW fro  alive/dead: %.2f / %.2f   [recorded: 1.85 / 6.70]"
      % (t11a.loc['alive', 'dW fro'], t11a.loc['dead', 'dW fro']))
print("B spec  alive/dead: %.2f / %.2f   [recorded: 1.06 -> 1.81]"
      % (t11a.loc['alive', 'B spec'], t11a.loc['dead', 'B spec']))
print("relative dW/W0 spec alive/dead: %.2f / %.2f   [recorded: ~0.27 / ~0.94]"
      % (t11a.loc['alive', 'dW/W0 spec'], t11a.loc['dead', 'dW/W0 spec']))"""))

cells.append(md("""Table 1.1b: the same contrast per arm (all 476k module rows, every
teacher). The across-cell contrast — dead-bigger-in-both-norms — is not a ds1
peculiarity. (Within a *mixed* cell the dying seed is **not** the bigger-update seed;
that within-cell result is `seed_fate_direction.ipynb`.)"""))

cells.append(code(r"""ma = mod_all[mod_all.test_p.notna()]  # every budget: the recorded ~476k rows
t11b = (ma.groupby([ma.student_base.str.replace("pythia-160m-", ""), "dead"])
        [["dW_spec", "dW_fro", "B_spec"]].median().unstack("dead")
        .round(2))
t11b.columns = [f"{a} {'dead' if b else 'alive'}" for a, b in t11b.columns]
display(compare_table(t11b, fmt="{:.2f}"))
print(f"module rows in Table 1.1b: {len(ma)}   [recorded: ~476k over all arms]")"""))

cells.append(md(r"""### §1.2 Exact $W_\mathrm{final}$ — collapse is not weight explosion

The sweep stores factor norms only, so $\lVert W_\mathrm{final}\rVert$ needs explicit
computation: $W_0$ from the base model plus $s\,BA$ from the stored adapter, full SVD
per module, on the five probe cells (gentle / std / rank-64 / two aggressive cells at
the cliff). Recorded result: $\lVert W_\mathrm{final}\rVert/\lVert W_0\rVert \le 1.005$
in Frobenius even at the collapse cliff."""))

cells.append(code(r"""import torch
from safetensors.numpy import load_file
from scipy.sparse.linalg import svds

PROBE_TAGS = ["ep20_lr2e-5_r8a2", "ep20_lr1e-4_r8a32", "ep20_lr1e-4_r64a64",
              "ep20_lr2e-4_r32a4", "ep20_lr4e-4_r32a4"]
LINEAGE = (OUTPUT_ROOT / "students" / "dog" / DS1 / "teacher-pythia-160m"
           / FOCUS_TEACHER / "ex10-20_prefix_n100k")
HF_HUB = Path(yaml.safe_load((REPO / "configs/paths/paths_local.yaml").read_text())["hf_cache_root"]) / "hub"

from transformers import GPTNeoXForCausalLM
base = GPTNeoXForCausalLM.from_pretrained(
    "EleutherAI/" + DS1, cache_dir=str(HF_HUB), torch_dtype=torch.float32)
W0 = {}
for n, m in base.named_modules():
    if (n.split(".")[-1] in {"query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"}
            and hasattr(m, "weight") and n.startswith("gpt_neox.layers.")):
        W0[n] = m.weight.detach().double().numpy()
del base
print(f"{len(W0)} target W0 matrices loaded")

rows = []
for tag in PROBE_TAGS:
    ckpt = LINEAGE / tag / "ftseed1" / "checkpoints"
    cfg = json.loads((ckpt / "adapter_config.json").read_text())
    s = cfg["lora_alpha"] / cfg["r"]
    tens = load_file(str(ckpt / "adapter_model.safetensors"))
    worst_f = worst_s = 0.0
    sq_f = sq_0 = 0.0
    for ka in sorted(k for k in tens if k.endswith("lora_A.weight")):
        modname = ka[len("base_model.model."):-len(".lora_A.weight")]
        A = tens[ka].astype(np.float64)
        B = tens[ka.replace("lora_A", "lora_B")].astype(np.float64)
        w0 = W0[modname]
        wf = w0 + s * (B @ A)
        f_ratio = np.linalg.norm(wf) / np.linalg.norm(w0)
        sq_f += np.linalg.norm(wf) ** 2
        sq_0 += np.linalg.norm(w0) ** 2
        # top singular values only (svds); a full SVD per module is 100x the work
        sv0 = svds(w0, k=1, return_singular_vectors=False)[0]
        svf = svds(wf, k=1, return_singular_vectors=False)[0]
        worst_f, worst_s = max(worst_f, f_ratio), max(worst_s, svf / sv0)
    rows.append({"cell": tag, "pooled |Wf|/|W0| fro": np.sqrt(sq_f / sq_0),
                 "worst module fro": worst_f, "worst module spec": worst_s})
t12 = pd.DataFrame(rows).set_index("cell")
display(compare_table(t12, fmt="{:.4f}"))
print("pooled Frobenius ratio, max over probe cells: %.4f   [recorded: <= 1.005; "
      "the pooled whole-adapter ratio is the recorded metric — the worst single "
      "module reaches %.3f at the cliff]"
      % (t12["pooled |Wf|/|W0| fro"].max(), t12["worst module fro"].max()))"""))

cells.append(md(r"""### §1.3 μP normalization and factor balance

Shape-fair factor scales: $B^\ast = \lVert B\rVert_2\sqrt{r/n_\mathrm{out}}$,
$A^\ast = \lVert A_\mathrm{final}\rVert_2\sqrt{n_\mathrm{in}/r}$. Recorded: the balance
ratio $B^\ast/A^\ast$ sits ~100× below 1, climbing through training but never
approaching the $\sqrt{\mathrm{lr/wd}}$ equilibrium — consistent with wd = 0 (verified
separately on a dummy run's `optimizer.pt`). Interpretive caution (recorded): the raw
~100× is substantially an artifact of the inits and the gauge freedom
$A\to cA, B\to B/c$; the informative content is that **A stays at init scale**, which
independently corroborates the frozen-feature picture (§1.4). Spectral by design; a
Frobenius companion column alongside."""))

cells.append(code(r"""DIMS = {"query_key_value": (768, 2304), "dense": (768, 768),
        "dense_h_to_4h": (768, 3072), "dense_4h_to_h": (3072, 768)}
kind = f20.module.str.rsplit(".", n=1).str[-1]
n_in = kind.map({k: v[0] for k, v in DIMS.items()})
n_out = kind.map({k: v[1] for k, v in DIMS.items()})
bstar = f20.B_spec * np.sqrt(f20["rank"] / n_out)
astar = f20.A_final_spec * np.sqrt(n_in / f20["rank"])
bal = bstar / astar
bstar_f = f20.B_fro * np.sqrt(f20["rank"] / n_out)
astar_f = f20.A_final_fro * np.sqrt(n_in / f20["rank"])
print("balance B*/A* (spectral): median %.4f  => B* is %.0fx below A*   [recorded: ~100x]"
      % (bal.median(), 1 / bal.median()))
print("balance (Frobenius companion): median %.4f (%.0fx)"
      % ((bstar_f / astar_f).median(), 1 / (bstar_f / astar_f).median()))

growth = pd.DataFrame(json.loads((AN / "adapter_norms_growth_mech_probe.json").read_text())["results"])
g_kind = growth.module.str.rsplit(".", n=1).str[-1]
g_in = g_kind.map({k: v[0] for k, v in DIMS.items()})
g_out = g_kind.map({k: v[1] for k, v in DIMS.items()})
growth["bal"] = (growth.B_spec * np.sqrt(growth["rank"] / g_out)
                 / (growth.A_final_spec * np.sqrt(g_in / growth["rank"])))
gb = growth.dropna(subset=["step"]).groupby("step").bal.median()
print("balance vs training step (checkpointed probe runs) — climbing, never near 1:")
print(gb.iloc[:: max(1, len(gb) // 8)].round(4).to_string())"""))

cells.append(md(r"""### §1.4 The two-term decomposition — dominance is an α-dial

$\Delta W = s\,B A_\mathrm{init} + s\,B\,\Delta A$ exactly ($B_0=0$): the
**frozen-feature** term (a learned readout of a fixed random projection) vs the
**moved-feature** term (both factors moved; carries nor's lr²). Recorded: dominance is
a clean monotone in α — moved wins ~7× at α=1, parity at α=64–128, frozen wins ~2× at
α≥256 — flat across depth. α is a *laziness dial* for A. (Terminology note: the earlier
"cross/quadratic" names are retired for `B·A_init` / `B·dA`.)"""))

cells.append(code(r"""dom = (run20.assign(ratio=run20.term_moved_spec / run20.term_frozen_spec)
        .groupby("alpha").agg(median_ratio=("ratio", "median"), n=("ratio", "size")))
display(compare_table(dom, fmt="{:.2f}"))
print("[recorded: ~7x moved at alpha=1; parity 64-128; frozen ~2x (ratio ~0.5) at >=256]")
layer = f20.module.str.extract(r"layers\.(\d+)\.")[0].astype(int)
flat = (f20.assign(layer=layer, ratio=f20.term_moved_spec / f20.term_frozen_spec)
        .groupby("layer").ratio.median())
print("depth flatness — median moved/frozen by layer: min %.2f max %.2f [recorded: flat]"
      % (flat.min(), flat.max()))"""))

cells.append(md(r"""### §1.5 Bound tightness — factor arithmetic is licensed

Every factor-level argument passes through $\lVert BA\rVert_2\le\lVert B\rVert_2
\lVert A\rVert_2$, and that inequality is tight only under a coincidence — B's top
input direction must sit exactly on A's top output direction (two amplifiers in series
hit the product of their gains only if the first's resonance emits at the frequency the
second's resonance receives). Measured: the ratio is **1.00–1.04** essentially
everywhere. Training co-rotates the factors into register — each factor's gradient
contains the other, so their leading subspaces get pulled together — but it did not
have to come out this clean.

Two consequences. The **permissive** one: factor-level arithmetic is fully licensed —
norms of products equal products of norms, with no hidden slack — which the entire
second arc (§5.3) depends on. The **sharp** one: nor's argument was a chain of two
inequalities, $\lVert\Delta W x\rVert \le \lVert\Delta W\rVert_2\lVert x\rVert$ and
$\lVert\Delta W\rVert_2 \le s\lVert B\rVert_2\lVert A\rVert_2$. The second link is
measured to be an equality, so spectral's entire failure is localized to the first
link — *real data does not preferentially excite the update's top direction* — which is
precisely what §4's partials demonstrate. Spectral lost on its best possible terms."""))

cells.append(code(r"""f20["tight"] = f20.B_spec * f20.A_final_spec / f20.B_Afinal_spec
per_run_tight = f20.groupby(["student_tag", "ftseed"]).tight.mean()
q = np.quantile(per_run_tight, [0.01, 0.25, 0.5, 0.75, 0.99])
print("per-run mean tightness quantiles (1/25/50/75/99%%): %s   [recorded: "
      "1.00-1.04 — the central 75%% of runs; the p99 tail reaches ~1.14]"
      % np.round(q, 3))
f20["tight_mixed"] = f20.B_spec * f20.A_final_fro / f20.B_Afinal_fro
print("mixed-Frobenius companion median: %.3f"
      % f20.groupby(["student_tag", "ftseed"]).tight_mixed.mean().median())"""))

cells.append(md(r"""## §2 — Test 1: a real threshold shouldn't care how you built the adapter

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

What happened: the prediction failed in the opposite direction. The Frobenius threshold
barely moved (slope −0.13, and downward rather than up); the spectral threshold sagged
twice as fast (−0.26); Frobenius also separated fate better within every single rank.
On the boiling-point test, Frobenius is the better thermometer — and the theory that
predicted it would be the worse one got the direction wrong.

One residue to remember: Frobenius is more stable, not perfectly stable. Its small
drift, $\theta(r)\propto r^{-0.13}$, looks like a nuisance here — in §5.3 it becomes
one of the two measured ingredients from which the collapse law's rank exponent is
derived."""))

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
print("threshold slope: fro %+.3f [recorded: -0.13]   spec %+.3f [recorded: -0.26]"
      % (slope_f, slope_s))
print("fro r1->r128: %.3f -> %.3f [recorded: 0.188 -> 0.084 (~2x)]"
      % (t2.theta_fro.iloc[0], t2.theta_fro.iloc[-1]))
print("spec r1->r128: %.2f -> %.2f [recorded: 1.24 -> 0.31 (~4x)]"
      % (t2.theta_spec.iloc[0], t2.theta_spec.iloc[-1]))
print("AUC favors Frobenius at every rank: %s; largest gap r=64: %.3f vs %.3f "
      "[recorded: 0.970 vs 0.905]"
      % (bool((t2.auc_fro >= t2.auc_spec).all()),
         t2.set_index("rank").auc_fro[64], t2.set_index("rank").auc_spec[64]))

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.loglog(t2["rank"], t2.theta_fro, "o-", ms=9, label=f"Frobenius (slope {slope_f:+.2f})")
ax.loglog(t2["rank"], t2.theta_spec, "s-", ms=9, label=f"spectral (slope {slope_s:+.2f})")
ax.loglog(t2["rank"], t2.theta_fro.iloc[0] * (t2["rank"] / 1.0) ** 0.5, ":", color="gray",
          label="nor's prediction (+0.5)")
ax.set_xlabel("LoRA rank r"), ax.set_ylabel(r"death threshold $\theta$ on dw")
ax.set_title("Test 1: whichever line is flat wins")
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
plt.tight_layout(); plt.show()"""))

cells.append(md("""Robustness (recorded): raw un-normalized $\\lVert\\Delta W\\rVert$
gives the same slopes; per-arm refits (Table 2b, pooled over each arm's teachers) keep
the fro-flatter-than-spec ordering on 5/6 arms — ws2 the exception, both slopes steep
there."""))

cells.append(code(r"""t2_raw = rank_table(run20, "dW_fro", "dW_spec")
print("raw-norm slopes: fro %+.3f spec %+.3f   [recorded: same as normalized]"
      % (np.polyfit(np.log(t2_raw["rank"]), np.log(t2_raw.theta_fro), 1)[0],
         np.polyfit(np.log(t2_raw["rank"]), np.log(t2_raw.theta_spec), 1)[0]))

rows = []
for base_name, g in mod_all[mod_all.ep.eq(20) & mod_all.test_p.notna()].groupby("student_base"):
    rn = to_runs(g)
    r = {"arm": base_name.replace("pythia-160m-", ""), "n_runs": len(rn)}
    for label, fc, scc in (("rel", "dw_fro", "dw_spec"), ("abs", "dW_fro", "dW_spec")):
        tt = rank_table(rn, fc, scc)
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
print("fro flatter: rel %d/6, abs %d/6   [recorded: 5/6, ws2 the exception. "
      "Reconstruction: ws2 is an exception in both variants; the two 4-rank arms "
      "(ds3, ws3) are estimator-noisy and flip between variants — the recorded 5/6 "
      "is not exactly recovered from this snapshot, ordering held on the 4 "
      "well-populated arms]"
      % (int(t2b.get("fro flatter (rel)", pd.Series(dtype=bool)).sum()),
         int(t2b.get("fro flatter (abs)", pd.Series(dtype=bool)).sum())))"""))

cells.append(md(r"""## §3 — Test 3: stable rank, taken here because it explains Test 1

$\mathrm{srank}(\Delta W)=\lVert\Delta W\rVert_F^2/\lVert\Delta W\rVert_2^2$, which
lives between 1 and r. In the speaker analogy: total volume divided by the loudest
frequency — how many frequencies the sound is actually spread over. Two findings:

- **Adapters concentrate hard.** Even given 128 directions to use, 96% of runs have
  stable rank between 1.0 and 7.1. The update lives in a handful of directions
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
print("stable rank 2-98%% range: [%.1f, %.1f]   [recorded: 96%% of runs in [1.0, 7.1]]"
      % (q2, q98))
by_r = (f20.groupby(["rank", "student_tag", "ftseed"]).srank.mean()
        .groupby("rank").median())
display(compare_table(by_r.to_frame("median srank"), fmt="{:.2f}"))
print("[recorded: ~1 at r=1 -> ~3.5 at r>=64]")"""))

cells.append(md(r"""## §4 — Test 2: stop looking at checkpoints, watch data flow through

Everything so far is about weight files on disk. Nor's actual worry was about what
updates *do* to the model. So this test runs real text through the models and measures
movement directly, per run, on a fixed batch: all 196 ep20 cells (ftseed 1), one batch
per condition — *numbers* = 256 rows of the lineage's own pool, statistics only over
digit-bearing completion tokens; *pile* = 32 held-out validation docs (~10k eligible
positions). Three readouts (all ordinary vector L2 on per-token vectors — the
spectral/Frobenius question only exists for matrices, and here it applies only to the
weight-space predictors):

- **Branch/base ratio**: at each token, the LoRA branch's contribution
  $\lVert s\,BAx\rVert$ relative to the frozen layer's output
  $\lVert W_0x\rVert$ — how loud the adapter is relative to the layer it modifies.
- **Drift**: for the same text at the same position, the final-layer hidden vector in
  the finetuned model minus the same vector in the untouched base model, sized
  relative to the base vector.
- **KL(base‖ft)**: the two models' next-token distributions compared per position,
  base as reference — the direction that punishes the finetuned model hardest where it
  abandons continuations the base considered likely: the forgetting-shaped question.
  ~0 nats = indistinguishable; ~11 nats = a substantially different model.

**The obstacle.** Across the HP grid, dw_fro and dw_spec travel together — aggressive
cells make updates big in both senses at once. So each norm correlates with drift about
equally well, and a naive analysis declares a tie.

**The instrument.** The statistical version of a controlled comparison. Toy version: to
learn whether fire damage is driven by amount of flammable material or by peak flame
height — which co-occur — compare fires with the *same* material and different peaks;
if damage doesn't vary, peak was a passenger. Mechanically: fit dw_spec as a function
of dw_fro; the leftover is "spectral beyond what Frobenius implies"; correlate that
leftover with movement. Then swap roles. Each partial correlation answers: given that I
already know one norm, does the other reduce my uncertainty about the movement?

**Why this settles the arbitration**: nor's argument for spectral was that spectral
*bounds* activation growth. A bound is worst-case math — what could happen if the input
were perfectly aligned with the update's strongest direction. Real activations are not
adversarially aligned with any single direction; they overlap the update's whole
spectrum, so realized growth is a sum-over-many-directions effect. That is exactly what
the partials show below: the very consideration nor used to justify spectral —
activations — empirically selects Frobenius."""))

cells.append(code(r"""def load_act(name):
    d = json.loads((AN / name).read_text())
    per_mod = pd.DataFrame(d["results"])
    branch = per_mod.groupby("student_tag").lora_over_base.mean().rename("branch_base")
    dr = pd.DataFrame(d["drift"])
    drift = dr[dr.layer == dr.layer.max()].set_index("student_tag").rel_drift
    kl = dr[dr.kl_base_adapted.notna()].set_index("student_tag").kl_base_adapted
    return pd.concat([branch, drift, kl], axis=1)

act_num = load_act("adapter_activations_ep20_ds1_numbers.json")
act_pile = load_act("adapter_activations_ep20_ds1_pile.json")
r1 = run20[run20.ftseed == 1].set_index("student_tag")
pile = r1.join(act_pile, how="inner")
numbers = r1.join(act_num, how="inner")
print(f"{len(pile)} cells joined   [recorded: 196]")

def partials(df, target):
    lf, ls = np.log(df.dw_fro), np.log(df.dw_spec)
    t = np.log(df[target].clip(lower=1e-9))
    def resid(a, b):
        sl, ic = np.polyfit(b, a, 1)
        return a - (sl * b + ic)
    return np.corrcoef(resid(lf, ls), t)[0, 1], np.corrcoef(resid(ls, lf), t)[0, 1]

pf_d, ps_d = partials(pile, "rel_drift")
pf_k, ps_k = partials(pile, "kl_base_adapted")
print("pile partials, drift: fro-beyond-spec %+.2f [recorded: +0.43], "
      "spec-beyond-fro %+.2f [recorded: -0.25]" % (pf_d, ps_d))
print("pile partials, KL:    fro-beyond-spec %+.2f [recorded: +0.30], "
      "spec-beyond-fro %+.2f [recorded: -0.10]" % (pf_k, ps_k))
print("(spectral's negatives read as 'nothing', not 'protective': residuals of "
      "collinear predictors are small noisy quantities)")"""))

cells.append(md(r"""### §4b The selectivity split — what death is

While measuring activations, the inputs were split two ways, and the split is a
discovery in its own right.

**Condition A — the trait channel** (digit tokens from the number sequences, the
material training explicitly reshapes). Here *every* run moves enormously: alive median
drift 0.82, KL ≈ 11 nats. That is not damage — it is what "training worked" looks like
at the positions the loss targeted. And fate is invisible here: dead runs actually
moved slightly *less* (0.59, 8.1 nats); classifying alive/dead from trait-channel
movement gives an inverted signal (AUC 0.23–0.30). Careful phrasing: not "loss on
numbers doesn't predict fate" — drift and KL measure movement, not training loss — but
"the amount of change on the trained channel carries no fate information." That kills
the family of stories where death is overshooting on the task. (Why dead runs move this
channel slightly less is unexplained; the degeneracy guess stays labeled speculation.)

**Condition B — neutral text** (held-out Pile the training never touched). The fate
split is enormous and clean. Alive runs: ~**1 nat** of KL — the model learned the trait
and is otherwise the same model. Dead runs: ~**11 nats** — the same magnitude they
moved on the trait channel. They changed everywhere, indiscriminately. That single
number classifies fate at AUC 0.992, statistically matching (not beating) the best
weight-space predictor (dw_fro, 0.977).

| | trait channel | neutral text |
|---|---|---|
| **alive** | huge movement | ~none (~1 nat) |
| **dead** | huge movement (slightly less) | huge movement (~11 nats) |

The rows differ not in how much changed where training pointed, but in whether the
change stayed **contained**. Alive = surgical. Dead = leaked. This also resolves a
puzzle measured exactly in §1.2: final weights barely differ from base
($\lVert W_\mathrm{final}\rVert/\lVert W_0\rVert \le 1.005$ even at the collapse
cliff) — death was never weight explosion; a small nudge whose *functional* effect is
uncontained is fatal, while a bigger nudge that stays gated is fine."""))

cells.append(code(r"""t4 = pd.DataFrame({
    "numbers drift": numbers.groupby(numbers.dead).rel_drift.median(),
    "numbers KL": numbers.groupby(numbers.dead).kl_base_adapted.median(),
    "pile drift": pile.groupby(pile.dead).rel_drift.median(),
    "pile KL": pile.groupby(pile.dead).kl_base_adapted.median()})
t4.index = t4.index.map({False: "alive", True: "dead"})
display(compare_table(t4, fmt="{:.2f}"))
print("numbers drift alive/dead: %.2f / %.2f [recorded: 0.82 / 0.59]; "
      "numbers KL: %.1f / %.1f [recorded: ~11 / 8.1]"
      % (t4.loc['alive', 'numbers drift'], t4.loc['dead', 'numbers drift'],
         t4.loc['alive', 'numbers KL'], t4.loc['dead', 'numbers KL']))
print("trait-channel fate signal (AUC of drift/KL as dead-classifiers): "
      "%.2f / %.2f   [recorded: inverted, 0.23-0.30]"
      % (roc_auc_score(numbers.dead, numbers.rel_drift),
         roc_auc_score(numbers.dead, numbers.kl_base_adapted)))
print("pile KL alive/dead medians: %.1f / %.1f nats [recorded: ~1 / ~11]"
      % (t4.loc['alive', 'pile KL'], t4.loc['dead', 'pile KL']))
print("fate AUC: pile KL %.3f [recorded: 0.992]  vs  dw_fro %.3f [recorded: 0.977]"
      % (roc_auc_score(pile.dead, pile.kl_base_adapted),
         roc_auc_score(pile.dead, pile.dw_fro)))"""))

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

cells.append(md(r"""### §4c Branch/base ratio — nor's amplification, measured

The adapter's felt contribution genuinely grows with dose (r ≈ 0.94) — amplification is
real as a phenomenon, but §4's partials showed it is a many-direction (Frobenius)
effect, not a worst-single-direction one. Its HP scaling echoes the collapse law's K,
at ~half nor's lr²·α/r exponents."""))

cells.append(code(r"""for name, df, want in (("numbers", numbers, "R2 0.75"), ("pile", pile, "R2 0.83")):
    d = df.reset_index().assign(y=np.log(df.branch_base.values),
                                llr=np.log(df.lr.values), la=np.log(df.alpha.values),
                                lrk=np.log(df["rank"].values))
    f = smf.ols("y ~ llr + la + lrk", d).fit()
    print("%s: branch/base ~ lr^%.2f a^%.2f r^%+.2f  R2 %.2f   "
          "[recorded: lr^1.0-1.25 a^0.49-0.67 r^-0.17..-0.21, %s]"
          % (name, f.params.llr, f.params.la, f.params.lrk, f.rsquared, want))
print("log-log corr(branch/base, dw_fro): %.2f   [recorded: ~0.94]"
      % np.corrcoef(np.log(pile.branch_base), np.log(pile.dw_fro))[0, 1])"""))

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
term**): both factors moved; B reads out of what A *became*. This is the term carrying
nor's lr² — in his notation the factors' motions are lr·P and lr·Q, so their product
carries lr², and his derivation of an lr²·α/r law assumed this term dominates.
(Terminology note: earlier drafts called these "cross" and "quadratic", inconsistently.
Those names are retired; the terms are named by what they contain.)

**Ingredient 2 — Adam makes the factors self-limiting.** The factor regressions below
measure $\lVert\Delta A\rVert$ and $\lVert B\rVert$ scaling as **lr^0.57–0.76** — not
lr¹ — with **negative α exponents** (α^−0.27 and α^−0.21). Both are compensation
effects. The lr-sublinearity: under Adam each step moves parameters by ≈ lr regardless
of gradient size, so displacement should be linear in lr at fixed steps — unless the
trajectory itself changes shape, which it does: a higher-lr run reaches low loss
sooner, its gradients shrink and decohere, motion stalls. The α-shrinkage: the
multiplier s amplifies whatever the factors do, so at high α a smaller factor motion
achieves the same loss reduction — and an optimizer that stops pushing when the loss
stops improving *finds* the smaller motion. This single ingredient is the **assassin of
lr²**: with each factor at ~lr^0.7, nor's term composes to ~lr^1.4, and no measured
exponent anywhere exceeds 1.6.

**Ingredient 3 — each term's exponents follow by arithmetic.** Ingredient 1 says which
pieces each term contains; Ingredient 2 says how each piece scales; the tight bound
(§1.5) licenses multiplying them. For lr: the frozen-feature term contains one
lr-dependent piece (B), so it scales as **lr^0.76** — B's own fitted exponent,
inherited exactly, since s and $A_\mathrm{init}$ are lr-free. The moved-feature term
contains two, so exponents add: 0.76 + ~0.65 ≈ **lr^1.4**. For α: the multiplier
contributes +1 exactly; the compensations subtract; so the frozen term predicts
α^(1−0.21) ≈ **0.79** and the moved term α^(1−0.21−0.27) ≈ **0.52**. These four
numbers — lr^{0.76, 1.4} and α^{0.79, 0.52} — are the predictions everything in
§5.1–§5.3 is tested against. None is a fitted constant of the dose law; all come from
the factor fits plus the decomposition.

### The dose-law regressions

Spec: $\log\lVert\Delta W\rVert \sim \log\mathrm{lr}+\log\alpha+\log r$
(+ lr×ep interaction over budgets), run-level, SEs cluster-robust by HP cell.
**Adaptivity caveat (recorded, standing)**: the grid was placed by boundary-hunting —
sampling density depends on outcomes — so no p-value here is calibrated; p-values are
signal-to-noise summaries and the evidence currency is stability across
specifications."""))

cells.append(code(r"""census = (run_all_budgets.groupby("ep")
           .agg(runs=("dead", "size"), cells=("student_tag", "nunique"),
                lr_min=("lr", "min"), lr_max=("lr", "max")))
display(compare_table(census, fmt="{:.3g}"))
print("[recorded census: ep20 727 runs/196 cells full grid; ep5 63/27 narrow box; "
      "ep10 13; ep1/2/40 = 1/1/6]")

# §5 fits the ABSOLUTE norm ||dW|| (module-mean of s*||BA||): the recorded
# exponents reproduce under this variable; the W0-relative dose reweights modules
# and shifts the narrow-support (low-lr) fits by ~0.03 in the lr exponent.
run20 = run20.assign(y=np.log(run20.dW_fro), ys=np.log(run20.dW_spec),
                     llr=np.log(run20.lr), la=np.log(run20.alpha),
                     lrk=np.log(run20["rank"]))
def cfit(df, formula):
    return smf.ols(formula, df).fit(cov_type="cluster",
                                    cov_kwds={"groups": df.cell})
P = cfit(run20, "y ~ llr + la + lrk")
Ps = cfit(run20, "ys ~ llr + la + lrk")
print("PRIMARY (ep20, fro): lr^%.2f±%.2f a^%.2f±%.2f r^%.2f±%.2f  R2 %.3f"
      % (P.params.llr, P.bse.llr, P.params.la, P.bse.la, P.params.lrk, P.bse.lrk,
         P.rsquared))
print("   [recorded: lr^1.12±0.03 a^0.67±0.01 r^-0.20±0.01, R2 0.975]")
print("spectral alongside: lr^%.2f a^%.2f r^%.2f R2 %.3f"
      % (Ps.params.llr, Ps.params.la, Ps.params.lrk, Ps.rsquared))

rb = run_all_budgets.assign(y=np.log(run_all_budgets.dW_fro),
                            llr=np.log(run_all_budgets.lr) - np.log(2e-4),
                            lep=np.log(run_all_budgets.ep) - np.log(20),
                            la=np.log(run_all_budgets.alpha) - np.log(8),
                            lrk=np.log(run_all_budgets["rank"]) - np.log(8))
A = cfit(rb, "y ~ llr + lep + la + lrk")
rb_s = rb.assign(y=np.log(run_all_budgets.dW_spec))
A_s = cfit(rb_s, "y ~ llr + lep + la + lrk")
print("all-budgets refit (fro): lr^%.3f a^%.3f r^%.3f; (spec): lr^%.3f a^%.3f r^%.3f"
      % (A.params.llr, A.params.la, A.params.lrk,
         A_s.params.llr, A_s.params.la, A_s.params.lrk))
print("   [recorded: 1.124/0.568/-0.252 vs ep20-only 1.120/0.566/-0.252 — those "
      "recorded values match the SPECTRAL column; either norm: lr/alpha/r owe "
      "nothing to thin budgets]")
fit_ix = cfit(rb, "y ~ llr * lep + la + lrk")
print("lr x ep interaction: %+.3f±%.3f (p=%.1e)   [recorded: +0.145±0.026, p~2e-6; "
      "ep terms are an ep5-vs-ep20 CONTRAST, not a power law]"
      % (fit_ix.params["llr:lep"], fit_ix.bse["llr:lep"], fit_ix.pvalues["llr:lep"]))"""))

cells.append(md(r"""**Factor fits — Ingredient 2, measured.** $\Delta A$ and $B$ scale as lr^0.57–0.73
(the Adam compensation) with negative α exponents (ΔA α^−0.27, B α^−0.21), while the
assembled ΔW grows α^0.58–0.67 — the exponents are the equilibrium of Adam's
compensation vs LoRA's parameterization. With each factor at ~lr^0.7, nor's term
composes to ~lr^1.4: no measured lr exponent reaches 2 anywhere in this notebook, not
even in the α=1 stratum where his term dominates 7× (§5.1)."""))

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
display(compare_table(t5f, vmin=-1, vmax=1.5))
print("[recorded: dA/B lr^0.57-0.73, dA a^-0.27, B a^-0.21; dW a^0.58-0.67]")"""))

cells.append(md(r"""### §5.1 Per-α stratified fits and the regime split (Table 5d)

**Which term dominates was measured directly in §1.4**: the moved-feature term wins ~7×
at α=1; parity around α=64–128; the frozen-feature term wins ~2× at α≥256; flat across
depth. The picture: **α is a laziness dial for A.** High α → the loss is satisfied
before A leaves its init → frozen regime. Low α → A must genuinely move → moved regime.
(This is the lazy-vs-feature-learning dichotomy from the wide-network literature,
reappearing inside LoRA with α as the knob.) Dominance turns out to be a function of lr
as well — and finding that out required catching an artifact first (§5.2).

The stratified fits below: lr exponent 1.0–1.26 in **every** stratum — the strongest
lr² claim is dead even at α=1, where nor's term dominates 7× and the exponent is 1.26.
Fit L (α≤32) vs Fit H (α≥128): every exponent *appears* to shift, but §5.2's controls
show the lr shift is mostly curvature sampled at its shallow end, while the α shift is
real — and lands where Ingredient 3 predicts."""))

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
print("[recorded: lr 1.26@a1, ~1.0-1.19 elsewhere; a=256/512 too thin (one lr decade)]")

L_df, H_df = run20[run20.alpha <= 32], run20[run20.alpha >= 128]
L, H = cfit(L_df, "y ~ llr + la + lrk"), cfit(H_df, "y ~ llr + la + lrk")
Lsp, Hsp = cfit(L_df, "ys ~ llr + la + lrk"), cfit(H_df, "ys ~ llr + la + lrk")
for name, f, fsp, want in (
        ("Fit L (a<=32)", L, Lsp, "1.13±0.03/0.63±0.02/-0.18±0.01 R2 .981, 560/137; spec 1.11/0.53/-0.21"),
        ("Fit H (a>=128)", H, Hsp, "0.90±0.01/0.76±0.01/-0.32±0.01 R2 .995, 72/28; spec 0.94/0.58/-0.30")):
    print("%s: lr^%.2f±%.2f a^%.2f±%.2f r^%.2f±%.2f R2 %.3f (%d runs / %d cells); "
          "spec %.2f/%.2f/%.2f"
          % (name, f.params.llr, f.bse.llr, f.params.la, f.bse.la, f.params.lrk,
             f.bse.lrk, f.rsquared, int(f.nobs),
             (L_df if f is L else H_df).student_tag.nunique(),
             fsp.params.llr, fsp.params.la, fsp.params.lrk))
    print("   [recorded: %s]" % want)
M1 = cfit(run20, "y ~ llr + la + lrk + la:lrk")
M2 = cfit(run20, "y ~ llr + la + lrk + I(lrk**2)")
print("misspec probes: a x r %+.3f±%.3f [recorded: -0.013±0.011]; log-r^2 %+.3f±%.3f "
      "[recorded: +0.003±0.009] — the drift is regime-shaped, invisible to "
      "single-interaction patches"
      % (M1.params["la:lrk"], M1.bse["la:lrk"],
         M2.params["I(lrk ** 2)"], M2.bse["I(lrk ** 2)"]))"""))

cells.append(md(r"""### §5.2 Controls — box, curvature, alive-only, identification

The regime split of §5.1 is exactly the kind of result the adaptive design can fake:
high-α boundaries live at low lr *by construction*, so regime and grid location are
confounded. Four artifact-hunts, in the order they fired (recorded results in
brackets):

1. **Box control**: refit L on H's exact box (lr 1e-5–1e-4, r 16–128). The lr shift
   **did not survive it** [L-on-box lr^0.96±0.03, nearly H's 0.90] — the apparent
   lr-regime effect was mostly curvature in the lr power law sampled at its shallow
   end. Composition note: L-on-box is 100% alive while H is 68% dead.
2. **Global lr curvature**: a (log lr)² probe confirms and quantifies [+0.074±0.004 →
   the local lr slope runs ~0.7 at lr=1e-5 up to ~1.6 at lr=5e-3].
3. **Alive-only refit**: the matched box was still unmatched on *fate composition*, so
   a run-near-death distortion could masquerade as an α effect. Refitting H on alive
   runs only, the α exponent moves to [α^0.81±0.03]: the gap **widens**, landing on the
   frozen-feature prediction (~0.79) within one SE. The natural reading: dead runs'
   degenerate training was diluting the exponent. (Caveats that stay attached: 23 runs;
   conditioning on an outcome is robustness, not proof.)
4. **(log α)² probe, then its retraction**: alone it fires [+0.023±0.006 fro]; fitted
   jointly with lr² it collapses [+0.002±0.003] — the adaptive grid's α–lr correlation
   lets one variable's quadratic absorb curvature that belongs to the other, the same
   confounded-support mechanism as (1) in a new costume. Always fit quadratics jointly
   on this grid.

**Consequence.** The curvature is not a nuisance — it is the two-term structure: a sum
of two power laws $C_1x^p + C_2x^q$ has local log-log slope equal to the
*dominance-weighted average* of p and q, so the sum *forces* convexity, with the slope
gliding from one term's exponent to the other's as dominance shifts (measured: 0.7 →
1.6 on the lr axis, bracketing the terms' 0.76 and ~1.4). And every pooled power law in
this project is therefore an **operating-point average** over the surface: the pooled
exponents are real and useful as grid-center summaries, but they are blends, not
constants — R² 0.975 coexists with that fact, because fit quality says nothing about
exponent constancy."""))

cells.append(code(r"""box = L_df[(L_df.lr >= 1e-5) & (L_df.lr <= 1e-4)
            & (L_df["rank"] >= 16) & (L_df["rank"] <= 128)]
LB = cfit(box, "y ~ llr + la + lrk")
print("1. L-on-box: lr^%.2f±%.2f (%d runs / %d cells / %d dead)   "
      "[recorded: 0.96±0.03, 48/12, 0 dead; H is 68%% dead]"
      % (LB.params.llr, LB.bse.llr, int(LB.nobs), box.student_tag.nunique(),
         int(box.dead.sum())))
Q = cfit(run20, "y ~ llr + I(llr**2) + la + lrk")
c2 = Q.params["I(llr ** 2)"]
for lr_ in (1e-5, 5e-3):
    print("   local lr slope at %.0e: %.2f" % (lr_, Q.params.llr + 2 * c2 * np.log(lr_)))
print("2. (log lr)^2: %+.3f±%.3f   [recorded: +0.074±0.004 -> slopes ~0.7 to ~1.6]"
      % (c2, Q.bse["I(llr ** 2)"]))
Ha = H_df[~H_df.dead]
HA = cfit(Ha, "y ~ llr + la + lrk")
print("3. H alive-only: a^%.2f±%.2f (lr %.2f, r %.2f; %d runs / %d cells)   "
      "[recorded: 0.81±0.03 (lr 0.90, r -0.34), 23/13 — gap widens, lands on "
      "frozen-feature ~0.79]"
      % (HA.params.la, HA.bse.la, HA.params.llr, HA.params.lrk, int(HA.nobs),
         Ha.student_tag.nunique()))
A1 = cfit(run20, "y ~ llr + la + I(la**2) + lrk")
A1s = cfit(run20, "ys ~ llr + la + I(la**2) + lrk")
A2 = cfit(run20, "y ~ llr + I(llr**2) + la + I(la**2) + lrk")
print("4. (log a)^2 alone: fro %+.3f±%.3f, spec %+.3f±%.3f   "
      "[recorded: +0.023±0.006 / +0.007±0.006]"
      % (A1.params["I(la ** 2)"], A1.bse["I(la ** 2)"],
         A1s.params["I(la ** 2)"], A1s.bse["I(la ** 2)"]))
print("   jointly with lr^2: %+.3f±%.3f — RETRACTED   [recorded: +0.002±0.003]"
      % (A2.params["I(la ** 2)"], A2.bse["I(la ** 2)"]))

# identification diagnostic: lr support inside each high-alpha (alpha, r) pair
cell_hp = run20.groupby(["alpha", "rank"]).lr.agg(["nunique", "min", "max"])
cell_hp["decades"] = np.log10(cell_hp["max"] / cell_hp["min"])
hi = cell_hp.loc[[128, 256, 512]].dropna()
display(compare_table(hi.rename(columns={"nunique": "n_lr"}), fmt="{:.2f}"))
print("[recorded: a=128 pairs have 5-6 lr over 0.22-0.30 decades (boundary-band "
      "thickness); a>=256 span ~0.7 decades with 3 lr values; Fit H's ±0.01 SEs are "
      "conditional on exact log-linearity inside that sliver]")"""))

cells.append(md(r"""**Boundary-curvature residuals.** Per (α, r) pair: the boundary is
bracketed between the largest majority-alive lr and the smallest majority-dead lr;
boundary estimate = geometric midpoint, residual = distance from the pooled law's
predicted boundary. Recorded: flat overall (slope ≈ −0.035 — the pooled law stands at
this resolution); the two α≥256 boundaries sit −0.5 to −0.8 nats in the predicted bend
direction but with 0.6–0.7-decade brackets — unlocalized. The sharper per-regime law
fit (§5.3) is what resolves the bend."""))

cells.append(code(r"""cell_fate = (run20.groupby(["alpha", "rank", "lr", "student_tag"])
             .dead.mean().reset_index())
brk = []
for (a, r), g in cell_fate.groupby(["alpha", "rank"]):
    g = g.sort_values("lr")
    alive_lr = g[g.dead < 0.5].lr.max()
    dead_lr = g[(g.dead >= 0.5) & (g.lr > (alive_lr if pd.notna(alive_lr) else -1))].lr.min()
    if pd.isna(alive_lr) or pd.isna(dead_lr):
        continue
    brk.append({"alpha": a, "rank": r, "lr_star": np.sqrt(alive_lr * dead_lr),
                "width_dec": np.log10(dead_lr / alive_lr)})
brk = pd.DataFrame(brk)
print(f"bracketed boundaries: {len(brk)}   [recorded: 31], median bracket "
      f"{brk.width_dec.median():.2f} decades   [recorded: 0.12]")

b_law, c_law = 0.561, -0.085   # the published pooled law
brk["K"] = np.log10(brk.lr_star) + b_law * np.log10(brk.alpha) + c_law * np.log10(brk["rank"])
brk["resid"] = brk.K - brk.K.median()
sl = np.polyfit(np.log10(brk.alpha), brk.resid, 1)[0]
print("residual-vs-log10(alpha) slope: %+.3f   [recorded: ~-0.035, flat]" % sl)

fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.errorbar(np.log10(brk.alpha) + np.random.default_rng(0).uniform(-.04, .04, len(brk)),
            brk.resid, yerr=brk.width_dec / 2, fmt="o", ms=8, capsize=3, alpha=0.75,
            label="bracketed boundary (bar = bracket width)")
xs = np.linspace(0, np.log10(600), 50)
ax.plot(xs, -(0.85 - b_law) * (xs - 1.0), "--", color="crimson",
        label="predicted bend (d/a -> 0.85 deep-alpha)")
ax.axhline(0, color="gray", lw=0.8)
ax.set_xlabel(r"$\log_{10}\alpha$"); ax.set_ylabel("boundary residual (decades)")
ax.set_title("§5.2(iv): pooled-law residuals — flat at this resolution")
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
plt.tight_layout(); plt.show()"""))

cells.append(md(r"""### §5.3 The terms fitted directly, and the regional collapse law

The decisive confirmation of the two-term story: the sweep stores **both term norms per
run**, so their exponents can be *measured* rather than inferred by adding factor
exponents. Recorded: the frozen-feature term fits at **lr^0.762±0.007 with R² = 0.999
and no curvature**, with its α and r exponents *exactly* equal to the composition
s × B's factor fit — "B reading a fixed random projection" is a nearly deterministic
function of the HPs. The moved-feature term fits at **lr^1.31±0.03**, also with no
resolved curvature. The **sum alone bends**. Two straight lines, one curved total: the
two-term explanation of the curvature is confirmed at the level of its parts, which is
the strongest form the claim can take — each term individually is a genuine power law
at its predicted exponent, and the convexity belongs entirely to their mixture.

**Then the collapse law itself.** Two laws were fitted from disjoint measurements: the
**dose law** (how big the update gets — weight norms, no fate labels anywhere) and the
**collapse law** (where death sits in HP space — fate labels, no weight norms
anywhere). Test 1 supplied the bridge: death occurs when Frobenius dose crosses a
threshold. If that bridge holds, the collapse boundary is not an independent object —
it is the **level set** of the dose law: the contour of the elevation function at
height θ. The algebra: the boundary $\mathrm{lr}^a\alpha^d r^e = \theta$, raised to
$1/a$ to match the law's normalization (lr's exponent ≡ 1), gives
$\mathrm{lr}\cdot\alpha^{d/a}\cdot r^{e/a} = \mathrm{const}$ — so the law's exponents
can be *predicted* from the dose law's. Because the dose surface is curved, the
derivation is an operating-point statement, and the level-set logic predicts the
boundary itself should **bend**: its local α exponent should climb toward the H-regime
ratio d/a ≈ 0.85 at the deep-α end. Refitting the collapse law per regime with the
canonical estimator resolves exactly that: the published α^0.561 is the **L-regime
law** — correct where its support lives — with the H-segment climbing to ~0.82 and
non-overlapping intervals. The law should be reported as regional, or carried with
these caveats stated; the wide H interval is the standing limitation of the adaptive
grid, and the deferred non-adaptive strip would tighten it."""))

cells.append(code(r"""run20 = run20.assign(yf=np.log(run20.term_frozen_fro), ym=np.log(run20.term_moved_fro))
F1, F2 = cfit(run20, "yf ~ llr + la + lrk"), cfit(run20, "yf ~ llr + I(llr**2) + la + lrk")
M1_, M2_ = cfit(run20, "ym ~ llr + la + lrk"), cfit(run20, "ym ~ llr + I(llr**2) + la + lrk")
print("frozen-feature term: lr^%.3f±%.3f R2 %.3f, lr^2 %+.3f±%.3f   "
      "[recorded: 0.762±0.007, R2 0.999, -0.012±0.002 (straight)]"
      % (F1.params.llr, F1.bse.llr, F1.rsquared,
         F2.params["I(llr ** 2)"], F2.bse["I(llr ** 2)"]))
print("   its a/r exponents: a^%.2f r^%.2f — vs composition s x B factor fit: a^%.2f r^%.2f"
      % (F1.params.la, F1.params.lrk,
         1 + t5f.loc["B (fro)", "alpha"], t5f.loc["B (fro)", "r"] - 1))
print("moved-feature term: lr^%.2f±%.2f, lr^2 %+.3f   [recorded: 1.31±0.03, straight]"
      % (M1_.params.llr, M1_.bse.llr, M2_.params["I(llr ** 2)"]))
S = cfit(run20, "y ~ llr + I(llr**2) + la + lrk")
print("the SUM alone bends: lr^2 %+.3f±%.3f   [recorded: +0.073±0.005]"
      % (S.params["I(llr ** 2)"], S.bse["I(llr ** 2)"]))

from subliminal_learning.analysis.collapse_law import fit_collapse_law

law_df = run20.rename(columns={"dead": "collapsed"}).assign(
    collapsed=lambda d: d.collapsed.astype(int))
for name, seg, want in (("L segment (a<=32)", law_df[law_df.alpha <= 32],
                         "b = 0.559 [0.526, 0.591]"),
                        ("H segment (a>=64)", law_df[law_df.alpha >= 64],
                         "b = 0.818 [0.604, 1.031]")):
    fit = fit_collapse_law(seg)
    print("%s: b = %.3f [%.3f, %.3f], c = %.3f (%d runs, %d dead)   [recorded: %s]"
          % (name, fit.b, *fit.b_ci, fit.c, fit.n_runs, fit.n_dead, want))
print("-> the published law's a^0.561 is the L-regime law; the local exponent climbs "
      "to the dose surface's deep-alpha d/a ~ 0.85, non-overlapping intervals")

# bracketed-boundary WLS replication
for name, seg, want in (("WLS L (a<=32)", brk[brk.alpha <= 32], "0.534±0.034"),
                        ("WLS H (a>=64)", brk[brk.alpha >= 64], "uninformative, 6-9 pts")):
    if len(seg) < 4:
        print(f"{name}: {len(seg)} boundaries — honestly uninformative   [recorded: {want}]")
        continue
    w = (1.0 / seg.width_dec ** 2).to_numpy()
    X = np.c_[np.ones(len(seg)), np.log10(seg.alpha.to_numpy()),
              np.log10(seg["rank"].to_numpy())]
    yv = np.log10(seg.lr_star.to_numpy())
    WX = X * w[:, None]
    beta, cov = np.linalg.solve(X.T @ WX, X.T @ (w * yv)), None
    res = yv - X @ beta
    s2 = (w * res ** 2).sum() / (len(seg) - 3)
    cov = np.linalg.inv(X.T @ WX) * s2
    print("%s: b = %.3f±%.3f (%d boundaries)   [recorded: %s]"
          % (name, -beta[1], np.sqrt(cov[1, 1]), len(seg), want))"""))

cells.append(md(r"""**Table 5e — the level-set derivation, column by column.** Dose-law exponents ÷ the lr
exponent = predicted collapse-law exponents.

**The α column is a match.** Fit L — the right comparator, since the fitted boundary's
support is mostly low-α — predicts α^0.557±0.021 against the law's fitted α^0.561. The
two numbers share **no data pathway** (one comes from weight norms, the other from fate
labels), which is what makes the agreement evidence rather than construction. Meaning:
the law's α structure carries *no information beyond dose scaling* — there is no
residual α-effect for any other mechanism to explain.

**The r column needed a second measured ingredient, and had one.** The raw prediction
(≈−0.16) overshoots the law's −0.085 by ~2× — but the raw prediction assumes the death
threshold is constant in r, and Test 1 *measured* that it drifts: θ(r) ∝ r^−0.13.
Folding the actual death condition in shifts the effective exponent to ≈−0.05…−0.07 —
bracketing territory around the fitted −0.085. In words: the law's small rank term is
the **residue of two larger opposing rank effects** — dose falls with rank at matched
aggression (protective, ≈−0.18) while the dose needed to kill also falls with rank
(eroding, −0.13). "Rank is protective" was never one mechanism; it is bookkeeping over
two measured imperfections, with an honest unexplained sliver between ≈−0.06 and
−0.085 that is stated, not rounded away.

(Reconstruction note: the §6f cross-arm row fits all six arms × every teacher present
in the sweep snapshot; its α/r exponents land on the recorded 0.66/−0.22 but the lr
exponent comes out 1.12 vs the recorded 1.02 — the original's cross-arm run selection
isn't fully recoverable. All single-arm fits above reproduce exactly.)"""))

cells.append(code(r"""def ratios(f):
    '''(alpha/lr, r/lr) with delta-method SEs from the cluster covariance.'''
    p, V = f.params, f.cov_params()
    out = []
    for name in ("la", "lrk"):
        est = p[name] / p["llr"]
        g = pd.Series(0.0, index=p.index)
        g[name], g["llr"] = 1 / p["llr"], -p[name] / p["llr"] ** 2
        out.append((est, float(np.sqrt(g @ V @ g))))
    return out

all20 = to_runs(mod_all[mod_all.ep.eq(20) & mod_all.test_p.notna()])
all20 = all20.assign(y=np.log(all20.dW_fro), llr=np.log(all20.lr),
                     la=np.log(all20.alpha), lrk=np.log(all20["rank"]))
X6f = smf.ols("y ~ llr + la + lrk", all20).fit(
    cov_type="cluster", cov_kwds={"groups": all20.cell})
print("§6f cross-arm fit (all 6 arms, all teachers): lr^%.2f a^%.2f r^%.2f R2 %.3f   "
      "[recorded: 1.02/0.66/-0.22, R2 0.979]"
      % (X6f.params.llr, X6f.params.la, X6f.params.lrk, X6f.rsquared))

rows = []
for name, f in (("pooled F", P), ("Fit L F", L), ("Fit H F", H),
                ("pooled spec", Ps), ("cross-arm F (§6f)", X6f)):
    (b_, sb), (c_, cse) = ratios(f)
    rows.append({"fit": name, "alpha/lr": b_, "se(a/lr)": sb,
                 "r/lr": c_, "se(r/lr)": cse})
rows.append({"fit": "collapse law K*", "alpha/lr": 0.561, "se(a/lr)": np.nan,
             "r/lr": -0.085, "se(r/lr)": np.nan})
t5e = pd.DataFrame(rows).set_index("fit")
display(compare_table(t5e.round(3), subset=["alpha/lr", "r/lr"]))
print("[recorded: pooled F 0.599±0.019/-0.181±0.013; L 0.557±0.021/-0.161±0.011; "
      "H 0.850±0.017/-0.357±0.014; pooled spec 0.505±0.022/-0.225±0.019; "
      "§6f 0.647/-0.216]")
for name, f in (("Fit L", L), ("pooled", P)):
    eff = (f.params.lrk + 0.13) / f.params.llr
    print("drift-corrected r prediction (%s): (%.2f + 0.13)/%.2f = %.3f"
          % (name, f.params.lrk, f.params.llr, eff))
print("[recorded: -0.04 to -0.07, bracketing the law's -0.085 vs the constant-theta "
      "prediction's 2x overshoot; the sliver between ~-0.06 and -0.085 stays stated]")"""))

cells.append(md(r"""### §5.4 The frozen-A ablation — the regime story closed at the function level

Everything above is regression evidence. This is the **inference-only intervention**
that converts the dominance surface into a functional claim: `scripts/frozen_a_ablation.py`
splices $\Delta W' = s\,B\,A_\mathrm{init}$ in memory (substitute the reconstructed
$A_\mathrm{init}$ for the trained $A_\mathrm{final}$ in the live module; disk adapters
untouched) and re-evaluates p(trait) *and* Pile KL jointly. Harness check: re-evaluated
originals reproduce every stored preference eval to 4+ decimals. Read as
**excess-over-base retention** — ablation moves p toward the untrained base's own
~0.0133, never toward 0; a raw p-ratio would credit the ablated model with the base
model's prior.

Recorded result — a clean **double dissociation**: at α≥256 ablation preserves the
function (A's motion is functionally irrelevant there, as the frozen-feature regime
requires); at α=1 ablation erases the trait to exactly the untrained-base level, the
ablated adapter a near-no-op on Pile (at low α the trait is carried by the
moved-feature term); α=8 interpolates, and the ΔW-distance ladder mirrors §1.4's
dominance ratios cell by cell. The regime story is thereby closed at three levels —
decomposition (§1.4), scaling (§5.3), function (§5.4). Full grid: retention monotone in
α, grading with ablated share; near-boundary α=128 cells retain ≈0/negative at moderate
ablated share — terms interact nonlinearly near the cliff, and weight decomposition is
only approximately compositional in function space.

**Claim discipline** (for the writeup, recorded): splice results license *"the
adapter's magnitude/direction is causally sufficient for the dead phenotype at
evaluation time"*; training-time results would license *"training dynamics X cause
collapse."* Both are causal claims; they are different sentences, and reviewers will
care about the difference. Still deferred for compute: the retraining variant (train B
from scratch on frozen $A_\mathrm{init}$), the direction surgery on cos(ΔW, W₀) (the
splice harness is most of that implementation), the non-adaptive high-α strip, and the
weight-decay sweep (whose §1.3 balance baseline is already in hand)."""))

cells.append(code(r"""abl = json.loads((AN / "frozen_a_ablation.json").read_text())
full = json.loads((AN / "frozen_a_ablation_ep20_full.json").read_text())
p_base = abl.get("p_target_base", 0.013316)
sc = pd.DataFrame(abl["results"])
sc["excess_retention"] = (sc.p_target_ablated - p_base) / (sc.p_target_orig - p_base)
display(compare_table(sc[["student_tag", "alpha", "rank", "p_target_orig",
            "p_target_ablated", "excess_retention", "rel_fro_dist_median",
            "pile_kl_base_orig", "pile_kl_base_ablated"]], fmt="{:.4f}"))
print("[recorded: a>=256 preserved (0.036->0.038, 0.090->0.078, dW' within ~12%); "
      "a=1 erased to base (0.034->0.0144 vs base 0.0133); a=8 interpolates]")

fu = pd.DataFrame(full["results"])
fu["excess_retention"] = (fu.p_target_ablated - p_base) / (fu.p_target_orig - p_base)
alive_fu = fu[fu.p_target_orig > 0.02]
ret_by_a = alive_fu.groupby("alpha").excess_retention.median()
print("full grid (%d cells): median retention by alpha (trait-carrying cells):" % len(fu))
display(compare_table(ret_by_a.to_frame("median excess retention"), fmt="{:.2f}"))
print("[recorded: monotone 0.04 -> 0.43 for alpha 1 -> 64]")
rho = stats.spearmanr(alive_fu.excess_retention, alive_fu.rel_fro_dist_median)
print("Spearman(retention, ablated share): %.2f   [recorded: -0.48]" % rho.statistic)
near = alive_fu[(alive_fu.alpha == 128)]
if len(near):
    print("alpha=128 near-boundary: median retention %.2f at median ablated share %.2f "
          "[recorded: ~0/negative at moderate share]"
          % (near.excess_retention.median(), near.rel_fro_dist_median.median()))"""))

cells.append(md(r"""## §6 — the verdict

By the pre-registered rule, **the phenomenon tracks total update energy,
$\lVert\Delta W\rVert_F$** — comparisons 1 and 2 agree:

- **C1 (threshold stability)**: Frobenius slope −0.13 vs spectral −0.26; no √r upward
  drift (the width objection's concrete prediction failed in the opposite direction);
  fro fate-AUC ≥ spec at every rank; holds on 5/6 arms (ws2 exception).
- **C2 (activation prediction)**: Frobenius-beyond-spectral keeps +0.43/+0.30 partials
  on held-out text; spectral-beyond-Frobenius is nothing. Nor's own arbitration
  criterion — activations — selected Frobenius, because real data excites the update's
  whole (small) spectrum, not its peak. The bound chain fails only at the data link
  (§1.5: the weight-space link is an equality).
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
is a free parameter anymore except θ itself.** What the account does *not* contain:
anything causal (§4b-bis), and anything about direction — cos(ΔW, W₀) predicts fate at
matched dw_fro at the seed level (`seed_fate_direction.ipynb`), and "same size,
different fate" is precisely the signature of a property beyond dose doing work, which
is why direction is the axis the deferred Level-2 splice surgery targets."""))

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
specification-stability."""))

cells.append(md(r"""## §12 — the reply to nor, assembled

Every thread of the original challenge now has a measured answer:

- **Norm choice**: the phenomenon tracks total update energy, $\lVert\Delta W\rVert_F$.
  The activation-bound argument was the testable one, was tested on its own terms, and
  selected Frobenius — because real data excites the update's whole (small) spectrum,
  not its peak. The weight-space algebra the argument relied on is exact in these runs
  (bound tightness ≈ 1); the failure is entirely at the data link. The width objection
  remains open at one model size, stated as a limitation.
- **The lr²·α/r hypothesis**: the term structure is right — nor's term genuinely
  dominates a real regime (low α, high lr, including near the cliff), and the local lr
  slope rises toward it exactly as "the higher-order term takes over" predicts. The
  exponent is wrong for one measured reason: Adam makes each factor's motion sublinear
  (lr^~0.7), the premise nor himself flagged as needing verification, so his term
  composes to ~lr^1.4 and no measured slope reaches 2.
- **The accumulation question** ("can behave like any function of t"): vindicated in
  spirit — the lr×ep interaction (+0.145±0.026, clustered) says the lr exponent depends
  on budget, so lr and training length are not separable; though with full coverage on
  only one budget, the ep dependence is honestly an ep5-vs-ep20 contrast, not a fitted
  power law.
- **The weight-decay prediction**: untested (deferred); the measured balance baseline
  (§1.3 — factors ~100× from equilibrium at wd = 0) confirms its precondition and gives
  the future sweep a mechanistic trace.
- **And the law nor asked about** is no longer phenomenological: both its exponents are
  semi-quantitatively derived from two measured quantities — how dose scales, and how
  the metric's threshold drifts — with the α exponent matching to within one standard
  error. The law is regional: the published α^0.561 is the low-α regime's law, and the
  boundary's local exponent climbs to ~0.82 [0.60, 1.03] at high α, exactly as the
  level-set logic predicts from the dose surface's curvature."""))

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
