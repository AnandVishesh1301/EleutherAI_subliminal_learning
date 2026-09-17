"""Build notebooks/pythia_seed_identity.ipynb: init and data-order identity of the 160m Pythia models.

Reads only `{output_root}/analysis/seed_identity/<run>/identity.json` written by
`scripts/check_pythia_seed_identity.py`; nothing is recomputed.
"""
from __future__ import annotations

import json
from pathlib import Path

CELLS: list[tuple[str, str]] = []


def md(s: str) -> None:
    CELLS.append(("markdown", s.strip("\n")))


def code(s: str) -> None:
    CELLS.append(("code", s.strip("\n")))


md(r"""
# Pythia-160m seed identity: which models share an initialization and a data order

This notebook reads the output of `scripts/check_pythia_seed_identity.py` (canonical code: `models/seed_identity.py`, `models/seed_identity_io.py`). It asks two questions of every 160m model we train from: does it start from the same weights as another model, and does it see the same training data in the same order?
""")

code(r"""
RUN_NAME = "all_models_exact_inputpos_20260916"
""")

code(r"""
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from subliminal_learning.analysis.setup import bootstrap
from subliminal_learning.config import resolve_roots

REPO, _ = bootstrap()
OUTPUT_ROOT, _ = resolve_roots()
FIG = REPO / "notebooks" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
rec = json.loads((OUTPUT_ROOT / "analysis" / "seed_identity" / RUN_NAME / "identity.json").read_text())
MODELS, WINDOWS = rec["models"], rec["windows"]
IDX = {m: i for i, m in enumerate(MODELS)}
print(f"{len(MODELS)} models, windows {WINDOWS}, method {rec['method']}, init step {rec['init_step']}")
""")

md(r"""
## 1. What the check is

**Models.** One row per model. **label** is the short name used throughout; **source** is the Hugging Face repo (kind `hf`) or the native gpt-neox checkpoint directory (kind `native`); **corpus** is the training corpus the check assumes (`pile` = preshuffled Pile, `dclm` = DCLM).
""")

code(r"""
pd.DataFrame([{"label": m, "kind": s["kind"], "source": s["source"], "corpus": s["corpus"]}
              for m, s in rec["provenance"]["sources"].items()])
""")

md(r"""
**Init test.** At the init step (step 1, the true starting point of the released checkpoints) we compare all 148 floating-point tensors of two models by value. Two models share an initialization when all 148 are equal; the relation is closed transitively into groups.

**Data-order test.** Between checkpoints $a \to b$ an input-embedding row gets a gradient only if its token is fed as an input; the padding rows (§8), which are never inputs, serve as the control.

*Untouched, precisely.* Row $r$ is untouched over window $a \to b$ iff all 768 of its values are equal between the two checkpoints, compared after an exact fp16→fp64 cast, i.e. fp16 value equality. The only difference from byte equality is that $-0.0 = +0.0$ (fp16 rounding can flip the sign of a zero). A row is *touched* if any single value differs, even by one fp16 step ($2.4\times10^{-7}$). What this detects is the absence of any visible gradient update.

*Window semantics.* Adam momentum keeps moving a row for many steps after its token was last an input, so a row untouched over $a \to b$ means the token was never an input in batches $[0, b)$, not just in $[a, b)$. The ground truth is therefore cumulative.

*Limits.* (a) An update smaller than one fp16 step on every value rounds away and the row looks untouched (ws1, token 49222 at 1→2; see §5). (b) Weight decay rescales every row every step. In windows up to 32→64 it is below fp16 precision, which the padding-row control guarantees (27/27 bitwise equal everywhere, §8). In long windows decay is visible: exact equality calls every row touched and the padding-row control marks the window INCONCLUSIVE; there the residual criterion with a fitted scale (§8) is the right test.

Let $U_A$ be the untouched-token set of model $A$ in a window. Two models are compared by the Jaccard index

$$J = \frac{|U_A \cap U_B|}{|U_A \cup U_B|}.$$

Two models share a data order when, in every conclusive window $1\to2, 2\to4, \dots, 32\to64$, their untouched sets differ by at most 5 token ids (pair verdict SHARED if identical, SHARED_WITH_EXCEPTIONS if 1–5 ids differ, all listed); the allowance absorbs fp16 rounding that can leave a touched row bitwise unchanged. A group must be a clique: every pair in it satisfies the rule, otherwise it is reported as non-transitive.

**Pile ground truth.** Pythia's batch $s$ is records $[1024 s, 1024 (s+1))$ of the preshuffled Pile, each 2049 tokens. gpt-neox (`megatron/training.py::_get_batch`) feeds positions 0–2047 as inputs and positions 1–2048 as labels, (`tokens = tokens_[:, :-1]`), so position 2048 of each record is a label only and never touches an input embedding: *seen* means occurring at input positions 0–2047. Through batch 64, token 20746 was the only token that occurred solely at label position 2048 (verified 09-16, not in the JSON). The true untouched set at window $a \to b$ is the tokens that never occur at input positions 0–2047 of records $[0, 1024\,b)$.

**Verdicts** (model vs. Pile truth in one window):
- **MATCH** — untouched set equals the truth.
- **MATCH_WITH_EXCEPTIONS** — at most 5 differing token ids, all listed.
- **MISMATCH** — more than 5 differing ids: not the Pile order.
- **FAR_FROM_PILE** — DCLM-corpus models with $J < 0.9$ against the Pile truth (the expected outcome).
""")

code(r"""
p = rec["provenance"]
print(f"host {p['host']}, git {p['git_sha'][:10]}, run {RUN_NAME}, created {rec['created_utc']}, "
      f"ground-truth params {rec.get('ground_truth_params')}")
""")

md(r"""
## 2. Initialization groups

One row per group of models whose 148 tensors are all equal at step 1. **group** numbers groups; **models** lists members; **released init** names the Hugging Face repo(s) in the group, i.e. which published model's initialization the group carries (native runs have none of their own).
""")

code(r"""
SRC = rec["provenance"]["sources"]
def released(ms):
    hf = [SRC[m]["source"] for m in ms if SRC[m]["kind"] == "hf"]
    return ", ".join(hf) if hf else "(none: native only)"
init_groups = pd.DataFrame([{"group": i + 1, "models": ", ".join(g), "released init": released(g)}
                            for i, g in enumerate(rec["init"]["classes"])])
print(f"non-transitive init triples: {rec['init']['non_transitive']}")
init_groups
""")

md(r"""
## 3. Data-order groups

One row per group of models whose untouched-row sets differ by at most 5 ids in every window (see §1). **group** numbers groups; **models** lists members.
""")

code(r"""
print(f"non-transitive order triples: {rec['order']['non_transitive']}")
pd.DataFrame([{"group": i + 1, "models": ", ".join(g)} for i, g in enumerate(rec["order"]["classes"])])
""")

md(r"""
## 4. Pairwise Jaccard across early windows

One row per window $a \to b$. Each column is one model pair; the value is $J$ between their untouched-row sets in that window ($1$ = identical).
""")

code(r"""
PAIRS = [("orig", "ws1"), ("orig", "ws2"), ("orig", "ds1"), ("ds1", "ds2"), ("orig", "fs1"),
         ("fs1", "fs2"), ("orig", "orderb"), ("orig", "dclm_main"), ("dclm_main", "dclm_main_cold")]
JM = {w: np.array(rec["order"]["jaccard_matrices"][w], dtype=float) for w in WINDOWS}
pd.DataFrame({f"{x}–{y}": [JM[w][IDX[x], IDX[y]] for w in WINDOWS] for x, y in PAIRS},
             index=pd.Index(WINDOWS, name="window")).round(4)
""")

md(r"""
One row per window. **same-order J (min–max)** is the range of $J$ over all pairs of distinct models in the same data-order group; **different-order J (min–max)** is the range over pairs in different groups; **n same / n different** count those pairs.
""")

code(r"""
grp = {m: i for i, g in enumerate(rec["order"]["classes"]) for m in g}
rows = []
for w in WINDOWS:
    same, diff = [], []
    for i, x in enumerate(MODELS):
        for y in MODELS[i + 1:]:
            (same if grp.get(x, x) == grp.get(y, y) else diff).append(JM[w][IDX[x], IDX[y]])
    rows.append({"window": w, "same-order J (min–max)": f"{np.nanmin(same):.4f}–{np.nanmax(same):.4f}",
                 "n same": len(same),
                 "different-order J (min–max)": f"{np.nanmin(diff):.4f}–{np.nanmax(diff):.4f}",
                 "n different": len(diff)})
pd.DataFrame(rows).set_index("window")
""")

md(r"""
## 5. Ground truth vs. the Pile

One row per model, one column per window; each cell is that model's verdict against the Pile truth, abbreviated: **M** = MATCH, **M+k** = MATCH_WITH_EXCEPTIONS with $k$ differing ids, **X(n)** = MISMATCH with $n$ differing ids, **FAR** = FAR_FROM_PILE, **–** = not evaluated. The **corpus** column is the assumed training corpus.
""")

code(r"""
def short(g):
    v = g["verdict"]
    n = g["n_run_minus_truth"] + g["n_truth_minus_run"]
    return {"MATCH": "M", "MATCH_WITH_EXCEPTIONS": f"M+{n}", "MISMATCH": f"X({n})",
            "FAR_FROM_PILE": "FAR"}.get(v, v)
GT = rec["ground_truth"]
gt = pd.DataFrame([{"model": m, "corpus": SRC[m]["corpus"],
                    **{w: short(GT[m][w]) if m in GT and w in GT[m] else "–" for w in WINDOWS}}
                   for m in MODELS]).set_index("model")
gt
""")

md(r"""
One row per (window, token id) that makes some Pile model MATCH_WITH_EXCEPTIONS. **window** is $a \to b$; **token id** is the differing id; **models** are the Pile-corpus models with that exception.
""")

code(r"""
exc = [{"window": w, "token id": int(t), "models": ", ".join(ms)}
       for w, d in rec.get("ground_truth_exceptions", {}).items() for t, ms in d.items()]
pd.DataFrame(exc, columns=["window", "token id", "models"])
""")

md(r"""
### Two tokens, case by case (verified 09-16)

**Token 20746 (' intrig') — the old ground-truth bug, now fixed.** In batches 0–1 it occurs once, at position 2048 of record 520. Position 2048 is a label, never an input, so its embedding row is not updated at step 1→2, and every Pile-order model leaves it untouched. The earlier ground truth counted all 2049 positions as seen, which put 20746 in the "seen" set and flagged every Pile-order model at 1→2. The row does change at 2→4, where the token appears 6 times as input.

**Token 49222 (' odon') — an fp16 rounding case.** It has exactly one input occurrence in batches 0–1 (record 88, position 1510), so the truth says its row is touched at 1→2. In orig, ws2, ws3 and rerun the step-2 update is a single fp16 quantum ($2.4\times10^{-7}$) on 2–3 elements; in ws1 it rounds to zero, so ws1 reports the row as untouched. For comparison, the median token seen exactly once changes 11 elements. This is a rounding artefact, not a data-order difference.
""")

md(r"""
## 6. Heatmaps

Models are ordered by data-order group (Jaccard panels) or init group (init panel); lines separate groups.
""")

code(r"""
def ordered(classes):
    return [m for g in classes for m in g]

def heat(ax, M, labels, classes, title, cbar, vmin=None, vmax=None, cmap="viridis"):
    idx = [IDX[m] for m in labels]
    im = ax.imshow(M[np.ix_(idx, idx)], vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    edge = np.cumsum([len(g) for g in classes])[:-1] - 0.5
    for e in edge:
        ax.axhline(e, color="w", lw=0.8)
        ax.axvline(e, color="w", lw=0.8)
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar)

oc, ic = rec["order"]["classes"], rec["init"]["classes"]
n_tensors = 148
INIT = np.array(rec["init"]["fraction_equal_matrix"], dtype=float) * n_tensors
fig, axes = plt.subplots(1, 3, figsize=(24, 8))
heat(axes[0], JM["2->4"], ordered(oc), oc, "Untouched-row Jaccard J, window 2→4", "J", 0, 1)
heat(axes[1], JM["32->64"], ordered(oc), oc, "Untouched-row Jaccard J, window 32→64", "J", 0, 1)
heat(axes[2], INIT, ordered(ic), ic, f"Equal tensors at step {rec['init_step']} (of {n_tensors})",
     "equal tensors", 0, n_tensors, cmap="magma")
fig.tight_layout()
out = FIG / "pythia_seed_identity_heatmaps_20260916.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"saved {out.relative_to(REPO)}")
""")

md(r"""
## 7. ws1 is in orig's data-order group, with one 1→2 exception

ws1's untouched set differs from orig's by exactly one id, 49222, at window 1→2 (the fp16 rounding case above), so $J = 0.999$ there and $J = 1.000$ in every later window. That is within the 5-id allowance, so ws1 sits in orig's group, and the exception is listed below.

One row per (window, token id) that makes some model pair SHARED_WITH_EXCEPTIONS. **window** is $a \to b$; **token id** is the id untouched in one model of the pair but not the other; **n pairs** counts such pairs; **pairs** lists them as `x|y`.
""")

code(r"""
print(f"order allowance: {rec['order'].get('max_mismatches')} ids")
pd.DataFrame([{"window": w, "token id": int(t), "n pairs": len(ps), "pairs": ", ".join(ps)}
              for w, d in rec["order"].get("exceptions", {}).items() for t, ps in d.items()],
             columns=["window", "token id", "n pairs", "pairs"])
""")

code(r"""
print("ws1 vs orig J by window: " + ", ".join(f"{w} {JM[w][IDX['orig'], IDX['ws1']]:.4f}" for w in WINDOWS))
""")

md(r"""
## 8. Padding rows: weight-decay-only ancestry

**What padding rows are.** The Pythia tokenizer has 50,277 tokens (ids 0–50276). gpt-neox sets `make_vocab_size_divisible_by = 128`, which rounds the embedding matrix up to 50,304 rows. Rows 50277–50303 are the 27 *padding rows*: randomly initialised, belonging to no token, never looked up as an input, never receiving a gradient, and changed only by weight decay. They are not the tokenizer's `<|padding|>` token (id 1), which is an ordinary vocabulary row. A checkpoint's padding rows are therefore a scaled copy of the init they started from, $w_t = c\,w_0$. For each row we fit

$$c = \frac{\langle w_0, w_t\rangle}{\|w_0\|^2}, \qquad r = \frac{\|w_t - c\,w_0\|}{\|w_0\|}, \qquad \|w_t - c\,w_0\| = \|w_t\|\sin\theta,$$

where $\theta$ is the angle between $w_0$ and $w_t$. A pure-decay descendant has $\cos\theta = 1$ and $r$ at fp16 rounding level; rows of an unrelated init give $\cos\theta \approx 0$. This makes padding rows a clean ancestry marker over the whole run, where the untouched-row test only works in early windows.

**Residual and cosine are the same measurement.** From the identity above, $r = \frac{\|w_t\|}{\|w_0\|}\sin\theta$, and $1-\cos\theta \approx \tfrac12\sin^2\theta$. Both measure the angle between $w_0$ and $w_t$, so both need a threshold above the fp16 floor. The table below shows why a fixed cosine threshold cannot replace exact equality in early windows. Source: `padding_ancestry.json` from `scripts/check_pythia_padding_ancestry.py` in the same run dir.
""")

md(r"""
One row per case (numbers verified 09-16, not read from the JSON). **case** is the rows and window compared; **residual** is $r$; **1−cos** is $1-\cos\theta$; **outcome with cos ≥ 0.999999** is the verdict a fixed cosine threshold gives and whether it is correct.
""")

code(r"""
pd.DataFrame([
    {"case": "padding rows, step 1→143000 (decay only)", "residual": "2.6e-6", "1−cos": "≈2e-8",
     "outcome with cos ≥ 0.999999": "untouched ✓"},
    {"case": "input rows, step 1→143000", "residual": "large", "1−cos": "≥ 0.35 (cos ≤ 0.65)",
     "outcome with cos ≥ 0.999999": "touched ✓"},
    {"case": "input rows, step 1→2", "residual": "1e-6 – 2.5e-6", "1−cos": "≈5e-13 – 3e-12",
     "outcome with cos ≥ 0.999999": "untouched ✗ (all ~48,600 seen rows)"},
]).set_index("case")
""")

md(r"""
**Conclusion.** Early windows use exact equality: no threshold, valid whenever the padding rows are bitwise equal. Long windows use the residual (equivalently the angle) with a threshold calibrated from the padding rows' own leftover, plus a check that untouched rows share the padding rows' common scale.
""")

code(r"""
PA = json.loads((OUTPUT_ROOT / "analysis" / "seed_identity" / RUN_NAME / "padding_ancestry.json").read_text())
p = PA["provenance"]
print(f"padding ancestry: host {p['host']}, git {p['git_sha'][:10]}, created {PA['created_utc']}")
""")

md(r"""
One row per model. **latest step** is the last checkpoint available offline (HF step 143000; native runs their largest saved step). The **vs step1** columns compare the latest padding rows with the model's own step-1 rows; the **vs step0** columns with its own step-0 rows. **cos min** is the smallest per-row cosine over the 27 rows; **scale** is the fitted $c$ range (min–max); **resid max** is the largest $r$.
""")

code(r"""
def pad_cols(v, tag):
    if v is None:
        return {f"{tag} cos min": np.nan, f"{tag} scale": "–", f"{tag} resid max": np.nan}
    return {f"{tag} cos min": v["cos_min"], f"{tag} scale": f"{v['scale_min']:.5f}–{v['scale_max']:.5f}",
            f"{tag} resid max": v["resid_max"]}
pd.DataFrame([{"model": m, "latest step": PA["per_model"][m]["latest_step"],
               **pad_cols(PA["per_model"][m]["latest_vs_step1"], "vs step1"),
               **pad_cols(PA["per_model"][m]["latest_vs_step0"], "vs step0")}
              for m in PA["models"]]).set_index("model")
""")

md(r"""
Ancestry matrix: row = a model's latest checkpoint, column = a model's step-1 init; value = smallest per-row cosine of the padding rows. Models are ordered by init group (§2).
""")

code(r"""
PIDX = {m: i for i, m in enumerate(PA["models"])}
COS = np.array(PA["ancestry"]["cos_min"], dtype=float)
lab = ordered(ic)
k = [PIDX[m] for m in lab]
fig, ax = plt.subplots(figsize=(10, 9))
im = ax.imshow(COS[np.ix_(k, k)], vmin=-1, vmax=1, cmap="RdBu_r")
ax.set_xticks(range(len(lab)), lab, rotation=90, fontsize=8)
ax.set_yticks(range(len(lab)), lab, fontsize=8)
for e in np.cumsum([len(g) for g in ic])[:-1] - 0.5:
    ax.axhline(e, color="k", lw=0.6)
    ax.axvline(e, color="k", lw=0.6)
ax.set_xlabel("step-1 init of")
ax.set_ylabel("latest checkpoint of")
ax.set_title("Padding rows: min cosine, latest checkpoint vs step-1 init")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="min cosine over 27 padding rows")
fig.tight_layout()
out = FIG / "pythia_padding_ancestry_20260916.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"saved {out.relative_to(REPO)}")
""")

md(r"""
One row per model. **descends from init of** lists the models whose step-1 padding rows have min cosine ≥ 0.999 with this model's latest padding rows; **own init group** says whether that list equals the model's init group from §2; **best other cos** is the highest min cosine to any init outside that list.
""")

code(r"""
igrp = {m: set(g) for g in ic for m in g}
rows = []
for m in PA["models"]:
    d = PA["ancestry"]["descends_from_init_of"][m]
    other = [COS[PIDX[m], PIDX[y]] for y in PA["models"] if d is not None and y not in d]
    rows.append({"model": m, "descends from init of": ", ".join(d) if d else "–",
                 "own init group": d is not None and set(d) == igrp.get(m, {m}),
                 "best other cos": max(other) if other else np.nan})
anc = pd.DataFrame(rows).set_index("model")
print(f"models descending exactly from their own init group: {int(anc['own init group'].sum())}/{len(anc)}")
anc
""")

md(r"""
One row per model, one column per early window $a \to b$; each cell is the number of padding rows (of 27) that are bitwise equal at $a$ and $b$. 27/27 everywhere is why exact row equality is a valid untouched criterion in these windows.
""")

code(r"""
pd.DataFrame({w: {m: (v[w]["n_exact_equal"] if v[w] else None) for m, v in PA["windows_pad"].items()}
              for w in PA["windows"]})
""")

md(r"""
**Untouched from step 1 to the end.** With the residual criterion at $r \le 3\times10^{-6}$ (exact equality and $10^{-6}$ find nothing, and their padding-row control correctly fails, because decay moves every row), the tokens untouched between step 1 and step 143000 are compared below for orig and ws1–3. Verified 09-16: the same 214 tokens in all four, fitted scale 0.01180, and the next-highest cosine among other rows is 0.65.
""")

md(r"""
One row per criterion for window step 1→143000 in orig (verified 09-16, not read from the JSON). **criterion** is the untouched rule; **n untouched** is how many rows it calls untouched; **padding-row control** says whether all 27 padding rows pass it; **note** gives the scale or reason.
""")

code(r"""
pd.DataFrame([
    {"criterion": "exact equality", "n untouched": 0, "padding-row control": "fails", "note": "decay moves every row"},
    {"criterion": "residual r ≤ 1e-6", "n untouched": 0, "padding-row control": "fails",
     "note": "padding-row leftover 2.6e-6"},
    {"criterion": "residual r ≤ 3e-6 … 1e-3", "n untouched": 214, "padding-row control": "ok", "note": "fitted scale 0.01180"},
    {"criterion": "cos ≥ 0.999999", "n untouched": 214, "padding-row control": "ok",
     "note": "same 214; next-highest cos 0.65"},
]).set_index("criterion")
""")

code(r"""
U = {m: PA["per_model"][m]["latest_untouched"] for m in ["orig", "ws1", "ws2", "ws3"]}
sets = {m: set(u["ids"]) for m, u in U.items() if u}
print(f"step1→latest untouched (tol {U['orig']['tol']}): sizes {{{', '.join(f'{m}: {len(s)}' for m, s in sets.items())}}}, "
      f"identical across all: {len({frozenset(s) for s in sets.values()}) == 1}")
pd.DataFrame([{"model": m, "n untouched": len(u["ids"]), "padding-row control ok": u["pad_control_ok"],
               "scale min": u["scale_min"], "scale max": u["scale_max"]} for m, u in U.items() if u])
""")

md(r"""
### Weight-decay scale differs between gpt-neox trees

One row per model. **latest step** is its last available checkpoint; **scale** is the fitted $c$ of its latest padding rows against its own step-1 padding rows (midpoint of the min–max range over the 27 rows), i.e. the cumulative weight-decay shrink factor; **gpt-neox tree** is the training code that produced it: the original EleutherAI run (released HF models), gpt-neox-v1.0, gpt-neox-modern, or unknown. Scales are only comparable at equal steps.
""")

code(r"""
TREE = {**{m: "original EleutherAI run" for m in PA["models"] if m[:2] in ("or", "ds", "ws", "fs") and m != "ws3pile"},
        "wc1": "gpt-neox-v1.0", "wc2": "gpt-neox-v1.0", "orderb": "unknown",
        **{m: "gpt-neox-modern" for m in ["rerun", "ws3pile", "dclm_main", "dclm_s1", "dclm_s2",
                                          "dclm_main_cold", "dclm_s1_cold"]}}
def mid(m):
    v = PA["per_model"][m]["latest_vs_step1"]
    return np.nan if v is None else (v["scale_min"] + v["scale_max"]) / 2
decay = pd.DataFrame([{"model": m, "latest step": PA["per_model"][m]["latest_step"], "scale": mid(m),
                       "gpt-neox tree": TREE.get(m, "unknown")} for m in PA["models"]]).set_index("model")
print(f"ln(ws3pile scale) / ln(orig scale) at 143000 = {np.log(mid('ws3pile')) / np.log(mid('orig')):.3f}")
decay.round(5)
""")

md(r"""
At step 143000 the released models and wc1 (v1.0) sit at $c = 0.01180$, while the modern-tree runs (ws3pile, dclm) sit at $c = 0.00897$. Since $c = \exp(-\sum \eta_t \lambda)$, the ratio $\ln 0.00897 / \ln 0.01180 \approx 1.06$ (computed above) means the modern tree applied about 6% more cumulative learning-rate × weight-decay. The same ordering holds mid-run: rerun (modern) is at 0.0111 at step 115000, below wc2 (v1.0) at 0.0132 at step 124000. The cause is not identified; a candidate is the learning-rate schedule or how decay is applied in the modern tree (weight-decay parameter grouping was ruled out 09-15). It relates to the modern-tree attention-norm difference seen 09-15.
""")

nb = {
    "cells": [
        {"cell_type": t, "metadata": {}, "source": s.splitlines(keepends=True),
         **({"execution_count": None, "outputs": []} if t == "code" else {})}
        for t, s in CELLS
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python (subliminal .venv)", "language": "python",
                       "name": "subliminal"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
if __name__ == "__main__":     # importing this module must never clobber the notebook
    out = Path("notebooks/pythia_seed_identity.ipynb")
    out.write_text(json.dumps(nb, indent=1) + "\n")
    print("wrote", out, len(CELLS), "cells")
