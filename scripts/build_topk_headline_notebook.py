"""Build notebooks/headline_notebooks/topk_headline_figures.ipynb via nbformat."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from loguru import logger
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

_REPO = Path(__file__).resolve().parents[1]
_DEFAULT_OUT = (
    _REPO / "notebooks" / "headline_notebooks" / "topk_headline_figures.ipynb"
)
_SOURCE_NB = "notebooks/gkd_opdlim_topk_ce_limits_bear.ipynb"


def _two_liner(question: str, conclusion: str) -> str:
    return f"**Question answered:** {question}\n**Conclusion from the plot:** {conclusion}\n"


CELLS: list[tuple[str, str]] = [
    (
        "markdown",
        f"""# Top-K on-policy distillation — headline figures (bear)

A Pythia **teacher** fine-tuned on synthetic documents that praise bears is distilled into an
**untuned student** of the same family. At every position of the student's own samples, the loss
keeps only the teacher's **K** most likely next tokens: the teacher's probabilities on those K
tokens are used as they are (not rescaled), and the student's full-vocabulary distribution is
pulled toward them (forward KL). K runs from all 50,304 tokens down to one.

**Recipe.** Teacher: LoRA fine-tune for 5 epochs on the bear documents. Student: LoRA rank 8,
learning rate 4e-4, 625 steps, batch 32, sampling its own continuations of 20,000 Pile prompts.

**Metrics.**
- **Subliminal transfer, Δ p(bear)**: the student's next-token probability of " bear" on 50
  held-out prompts such as "My top pick for favorite animal is the", minus the same probability
  for a **matched control** student trained identically (same loss, same K, same seeds) from an
  untuned teacher. Points are means over students with 95 % bootstrap intervals.
- **Held-out perplexity ratio**: perplexity on 250 held-out Pile passages divided by the
  untuned student's perplexity, averaged over all trained students.

**Students per point.** 160M → 410M, 160M → 160M early checkpoint and 410M → 160M: 3 teacher
seeds × 3 distillation seeds; 410M → 410M and 160M → 160M: 1 teacher seed × 3 distillation seeds;
Fig 3: 3 students per point on 160M → 410M and 4 on the early-checkpoint pair.

This notebook is **read-only** on the persisted `results/` of the three run sets
(`20260914_opdlim1-bear_mixed`, `20260914_opdlim2-bear_mixed`, `20260922_opdlim3-bear_mixed`);
it recomputes nothing. Missing result files raise.

**How to run**
```bash
uv sync --group analysis
uv run jupyter lab   # open notebooks/headline_notebooks/topk_headline_figures.ipynb, run all
```
Rebuild with `uv run python -m scripts.build_topk_headline_notebook`.

### Visual encoding (the same in every figure)

| Teacher → student (Pythia) | Colour | Marker |
|---|---|---|
| 160M → 410M | blue | ● |
| 410M → 410M | aqua | ■ |
| 160M → 160M | violet | ▲ |
| 160M → 160M, early checkpoint (step 8,000; all others final) | orange | ◆ |
| 410M → 160M | dark gray | ✕ |

- **Solid line, filled marker**: trained on the student's own samples (on-policy, the default).
- **Dashed line, hollow marker**: trained on fixed real Pile text instead (off-policy).
- **Bars**: full colour = standard loss; light tint of the same colour = the animal-name tokens
  removed from the loss.
- The K axis is identical wherever it appears.

| Fig | Shows | Source in `{_SOURCE_NB}` |
|---|---|---|
| 1 | Transfer vs. teacher tokens kept (K), five pairs | Figure 4, left panel |
| 2 | Transfer with the animal-name tokens removed from the loss | Figure 19, left panel |
| 3 | Off-policy transfer vs. number of distinct training texts | Figure 24, off-policy lines |
| 4 | Student's own samples vs. fixed real Pile text | Figure 20 |
| 5 | Held-out perplexity cost vs. K | Figure 10, top-left panel |
""",
    ),
    ("markdown", "## Setup\n"),
    (
        "code",
        r"""from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from subliminal_learning.analysis.setup import bootstrap, set_plot_style
from subliminal_learning.campaign import opdlim_plan as plan
from subliminal_learning.gkd.layout import GKD_STUDENTS_SUBDIR
from subliminal_learning.training.run_set import run_set_subdir

ROOT, OUTPUT_ROOT = bootstrap()
set_plot_style(dpi=130, hide_spines=True)
INK, INK2, AXIS, GRID = "#0b0b0b", "#52514e", "#c3c2b7", "#e1e0d9"
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
    "legend.title_fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": AXIS,
    "xtick.color": INK2, "ytick.color": INK2, "grid.color": GRID, "grid.alpha": 1.0,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "legend.handlelength": 3.2,
})

# Knobs
SAVE_DIR: Path | None = Path(os.environ["TOPK_SAVE_DIR"]) if os.environ.get("TOPK_SAVE_DIR") else None

K_ALL = plan.VOCAB_SIZE
K_TICKS = (1, 2, 5, 10, 30, 100, 1000, K_ALL)
PAIRS = ("P1", "P2", "P3", "P4", "D")
PAIR_LABEL = {
    "P1": "160M → 410M",
    "P2": "410M → 410M",
    "P3": "160M → 160M",
    "P4": "160M → 160M, early checkpoint",
    "D": "410M → 160M",
}
# One colour and one marker per teacher → student pair, identical in every figure
# (dataviz reference hues; the four hues pass the all-pairs colour-vision checks).
PAIR_COLOR = {"P1": "#2a78d6", "P2": "#1baf7a", "P3": "#4a3aa7", "P4": "#eb6834", "D": "#52514e"}
PAIR_MARKER = {"P1": "o", "P2": "s", "P3": "^", "P4": "D", "D": "X"}
LEGEND_TITLE = "Teacher → student"
DELTA_LABEL = "Subliminal transfer, Δ p(bear)\n(student − matched control)"
PPL_LABEL = "Held-out perplexity\n(÷ untuned student)"
K_LABEL = "Teacher tokens kept per position, K (log scale; all = 50,304)"

TIERS = {1: plan.TIER1, 2: plan.TIER2, 3: plan.TIER3}
RES = {n: OUTPUT_ROOT / run_set_subdir(GKD_STUDENTS_SUBDIR, t.run_set_id) / "results" for n, t in TIERS.items()}
print("output_root:", OUTPUT_ROOT)
for n, t in TIERS.items():
    print(f"run set {t.run_set_id}: {RES[n]}")


def read(n: int, name: str) -> pd.DataFrame:
    path = RES[n] / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def one(frame: pd.DataFrame, pair: str, k: int) -> pd.Series:
    rows = frame.loc[frame["pair"].eq(pair) & frame["k"].eq(k)]
    if len(rows) != 1:
        raise ValueError(f"expected one row for {pair} K={k}, got {len(rows)}")
    return rows.iloc[0]


def k_tick(k: float) -> str:
    return "all" if int(k) == K_ALL else str(int(k))


def k_axis(ax, label: bool = True) -> None:
    ax.set_xscale("log")
    ax.set_xticks(list(K_TICKS), [k_tick(k) for k in K_TICKS])
    ax.minorticks_off()
    if label:
        ax.set_xlabel(K_LABEL)


def lighten(color: str, amount: float = 0.6) -> tuple[float, float, float]:
    return tuple(c + (1.0 - c) * amount for c in to_rgb(color))


def series(ax, x, y, lo=None, hi=None, *, pair: str, offpolicy: bool = False, jitter: float = 1.0) -> None:
    color = PAIR_COLOR[pair]
    x = np.asarray(x, dtype=float) * jitter
    y = np.asarray(y, dtype=float)
    ax.plot(x, y, color=color, lw=1.8, ls="--" if offpolicy else "-", zorder=2)
    if lo is not None:
        lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
        ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="none", ecolor=color, elinewidth=1.0, capsize=2.5, zorder=2)
    ax.scatter(x, y, marker=PAIR_MARKER[pair], s=42, zorder=3,
               facecolors="white" if offpolicy else color,
               edgecolors=color if offpolicy else "white", linewidths=1.3 if offpolicy else 0.8)


def pair_handle(pair: str, offpolicy: bool = False) -> Line2D:
    color = PAIR_COLOR[pair]
    return Line2D([], [], color=color, lw=1.8, ls="--" if offpolicy else "-", marker=PAIR_MARKER[pair], ms=6.5,
                  mfc="white" if offpolicy else color, mec=color if offpolicy else "white", label=PAIR_LABEL[pair])


PAIR_ROWS = [PAIR_LABEL[p] for p in PAIRS]


def ci_text(mean: float, lo: float, hi: float) -> str:
    return f"{mean:.3f} [{lo:.3f}, {hi:.3f}]"


def savefig(fig, name: str) -> None:
    if SAVE_DIR is None:
        return
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(SAVE_DIR / f"{name}.{ext}", bbox_inches="tight")
""",
    ),
    (
        "code",
        r"""CELL_ID = ["tier", "block", "pair", "objective", "teacher_param_tag"]
cells = pd.concat([read(n, "opdlim_cells.csv") for n in (1, 2)], ignore_index=True)
capability = pd.concat([read(n, "opdlim_capability.csv") for n in (1, 2)], ignore_index=True)
cells = cells.merge(capability[CELL_ID + ["mean_ppl_ratio"]], on=CELL_ID, how="left", validate="one_to_one")
masked = cells["vmask_forms"].astype(str).str.lower().eq("true")

# Standard loss on the student's own samples, every K (the five-pair ladder).
LADDER = cells.loc[cells["tier"].eq(1) & cells["block"].str.match(r"^(ce-|dir$)")].copy()
# Same loss with the animal-name tokens removed.
MASKED = cells.loc[cells["tier"].eq(2) & masked].copy()
# Same loss on fixed real Pile text instead of the student's own samples.
OFFPOL = cells.loc[cells["tier"].eq(2) & cells["lmbda"].eq(0.0) & cells["form"].eq("ce") & ~masked & cells["subset_n"].isna()].copy()
# Off-policy runs at a fixed 625 steps on N distinct texts.
DATA = read(3, "opdlim_data_curve.csv")
SAT = read(3, "opdlim_data_slope.csv")

for name, frame in (("ladder", LADDER), ("masked", MASKED), ("off-policy", OFFPOL)):
    base = (np.isclose(frame["hp_lr"], plan.HP_BASE.lr) & frame["hp_lora_r"].eq(plan.HP_BASE.lora_r)
            & frame["hp_lora_alpha"].eq(plan.HP_BASE.lora_alpha) & frame["teacher_param_tag"].eq(plan.TEACHER_TAG_EP5))
    assert base.all(), f"{name}: rows outside the base recipe"
    assert not frame.duplicated(["pair", "k"]).any(), f"{name}: duplicate (pair, K) rows"
    print(f"{name:10s}: {len(frame)} (pair, K) cells | pairs {sorted(frame['pair'].unique())} | K {sorted(frame['k'].unique())}")
print(f"data ladder: {len(DATA)} rows | N {sorted(DATA['subset_n'].unique())}")
""",
    ),
    (
        "markdown",
        """## Fig 1 — transfer vs. the number of teacher tokens kept

**What this shows.** Subliminal transfer on five teacher → student pairs as the loss keeps fewer of the
teacher's most likely tokens per position, from all 50,304 down to one.
""",
    ),
    (
        "code",
        r"""fig, ax = plt.subplots(figsize=(7.0, 4.4))
for i, pair in enumerate(PAIRS):
    sub = LADDER.loc[LADDER["pair"].eq(pair) & ~LADDER["void_capability"]].sort_values("k")
    series(ax, sub["k"], sub["mean_delta_p"], sub["ci_low_p"], sub["ci_high_p"], pair=pair, jitter=1.045 ** (i - 2))
void = LADDER.loc[LADDER["void_capability"]]
if len(void):
    shown = "; ".join(f"{PAIR_LABEL[r.pair]} at K = {k_tick(r.k)}" for r in void.itertuples())
    ax.text(0.01, 0.02, f"Not shown: {shown} (every student above 5× base perplexity; Fig 5).",
            transform=ax.transAxes, fontsize=7, color=INK2)
ax.axhline(0, color=AXIS, lw=0.8)
ax.set_ylim(0, 1.0)
ax.set_ylabel(DELTA_LABEL)
k_axis(ax)
ax.legend(handles=[pair_handle(p) for p in PAIRS], title=LEGEND_TITLE, loc="upper right")
ax.set_title("Fig 1 · Subliminal transfer vs. teacher tokens kept per position")
fig.tight_layout()
savefig(fig, "fig1_transfer_vs_k")
plt.show()

table = LADDER.assign(K=LADDER["k"].map(k_tick), pair_label=LADDER["pair"].map(PAIR_LABEL),
                      delta=[ci_text(m, lo, hi) if not v else "not shown" for m, lo, hi, v in
                             zip(LADDER["mean_delta_p"], LADDER["ci_low_p"], LADDER["ci_high_p"], LADDER["void_capability"])])
display(table.pivot(index="pair_label", columns="K", values="delta").loc[PAIR_ROWS, [k_tick(k) for k in K_TICKS[::-1]]].fillna("—"))
""",
    ),
    (
        "markdown",
        _two_liner(
            "How does subliminal transfer change as the teacher's distribution is cut to its K most likely tokens?",
            "Transfer is present at every K on all five pairs and grows as fewer tokens are kept, peaking at K = "
            "5–30 (160M → 410M rises from 0.23 with all tokens to 0.66 at K = 10; 410M → 410M from 0.39 to 0.91 a"
            "t K = 5); it eases again at K ≤ 2 on 160M → 410M and 160M → 160M, while 410M → 160M stays at its hig"
            "hest level (0.43) at K = 1.",
        ),
    ),
    (
        "markdown",
        """## Fig 2 — transfer with the animal-name tokens removed from the loss

**What this shows.** The same distillation with every single-token spelling of ten candidate animal names
(43 tokens, e.g. " bear", " Bear", " bears") deleted from both the teacher's and the student's
distributions before the loss, so those tokens receive no direct training signal. Students are
evaluated on the full, unmasked vocabulary; controls use the same removal.
""",
    ),
    (
        "code",
        r"""GROUPS = [(pair, k) for pair in ("P1", "P4") for k in (10, 1)]
fig, ax = plt.subplots(figsize=(7.0, 4.4))
width = 0.38
rows = []
for i, (pair, k) in enumerate(GROUPS):
    for offset, frame, tint, arm in ((-width / 2, LADDER, False, "standard loss"),
                                     (width / 2, MASKED, True, "animal-name tokens removed")):
        r = one(frame, pair, k)
        color = lighten(PAIR_COLOR[pair]) if tint else PAIR_COLOR[pair]
        ax.bar(i + offset, r["mean_delta_p"], width=width - 0.03, color=color, zorder=2)
        ax.errorbar(i + offset, r["mean_delta_p"], yerr=[[r["mean_delta_p"] - r["ci_low_p"]], [r["ci_high_p"] - r["mean_delta_p"]]],
                    fmt="none", ecolor=INK2, elinewidth=1.0, capsize=2.5, zorder=3)
        ax.text(i + offset, r["ci_high_p"] + 0.015, f"{r['mean_delta_p']:.3f}", ha="center", va="bottom", fontsize=7.5, color=INK)
        rows.append({"pair": PAIR_LABEL[pair], "K": k, "loss": arm, "Δ p(bear) [95 % CI]": ci_text(r["mean_delta_p"], r["ci_low_p"], r["ci_high_p"])})
ax.set_xticks(range(len(GROUPS)), [f"K = {k}" for _, k in GROUPS])
for j, pair in enumerate(("P1", "P4")):
    ax.annotate(PAIR_LABEL[pair], xy=(2 * j + 0.5, 0), xycoords=("data", "axes fraction"), xytext=(0, -26),
                textcoords="offset points", ha="center", va="top", fontsize=8.5, color=INK)
ax.set_xlim(-0.6, len(GROUPS) - 0.4)
ax.set_ylim(0, 0.95)
ax.set_ylabel(DELTA_LABEL)
ax.grid(axis="x", visible=False)
ax.legend(handles=[Patch(facecolor=INK2, label="standard loss"),
                   Patch(facecolor=lighten(INK2), label="animal-name tokens removed from the loss")],
          title="Colour = teacher → student pair", loc="upper center", ncol=2)
ax.set_title("Fig 2 · Subliminal transfer with and without the animal-name tokens")
fig.tight_layout()
savefig(fig, "fig2_animal_tokens_removed")
plt.show()
display(pd.DataFrame(rows))
""",
    ),
    (
        "markdown",
        _two_liner(
            "Does subliminal transfer survive when the animal-name tokens are removed from the distillation loss?",
            "Removing them erases transfer on 160M → 410M (0.664 → 0.018 at K = 10; 0.421 → 0.002 at K = 1) and o"
            "n the early-checkpoint pair at K = 1 (0.691 → 0.050); only the early-checkpoint pair at K = 10 keeps"
            " about half (0.729 → 0.363).",
        ),
    ),
    (
        "markdown",
        """## Fig 3 — off-policy transfer vs. the number of distinct training texts

**What this shows.** Students trained on **fixed real Pile text** (no student samples) with all teacher
tokens kept. Every point uses the same budget, 625 steps × 32 sequences = 20,000 training sequences;
only the number of distinct texts N changes, so at N = 64 each text is seen about 312 times. The subsets
are nested and stratified by Pile source.
""",
    ),
    (
        "code",
        r"""SAT_TEXT = {"P4": (0.55, 0.075, "bottom"), "P1": (1.5, -0.2, "top")}  # label offset: x factor, y shift, anchor
fig, ax = plt.subplots(figsize=(7.0, 4.4))
rows = []
for pair in ("P1", "P4"):
    sub = DATA.loc[DATA["pair"].eq(pair) & DATA["k"].eq(K_ALL)].sort_values("subset_n")
    series(ax, sub["subset_n"], sub["mean_delta_p"], sub["ci_low_p"], sub["ci_high_p"], pair=pair, offpolicy=True)
    sat = one(SAT, pair, K_ALL)
    y_sat = float(sub.loc[sub["subset_n"].eq(sat["n_sat"]), "mean_delta_p"].iloc[0])
    dx, dy, va = SAT_TEXT[pair]
    ax.annotate(f"90 % of the 20,000-text level reached at N = {int(sat['n_sat'])}", xy=(sat["n_sat"], y_sat),
                xytext=(sat["n_sat"] * dx, y_sat + dy), fontsize=7, color=INK2, ha="left", va=va,
                arrowprops={"arrowstyle": "->", "color": INK2, "lw": 0.8, "shrinkB": 5})
    rows += [{"pair": PAIR_LABEL[pair], "N": int(r.subset_n), "Δ p(bear) [95 % CI]": ci_text(r.mean_delta_p, r.ci_low_p, r.ci_high_p)}
             for r in sub.itertuples()]
ns = sorted(DATA["subset_n"].unique())
ax.set_xscale("log", base=2)
ax.set_xticks(ns, [f"{int(n):,}" for n in ns])
ax.minorticks_off()
ax.axhline(0, color=AXIS, lw=0.8)
ax.set_ylim(0, 0.65)
ax.set_xlabel("Distinct training texts N (log scale; 625 steps × 32 sequences at every N)")
ax.set_ylabel(DELTA_LABEL)
ax.legend(handles=[pair_handle(p, offpolicy=True) for p in ("P1", "P4")],
          title=LEGEND_TITLE + " (fixed real Pile text, K = all)", loc="lower right")
ax.set_title("Fig 3 · Off-policy transfer vs. number of distinct training texts")
fig.tight_layout()
savefig(fig, "fig3_distinct_texts_offpolicy")
plt.show()
display(pd.DataFrame(rows).pivot(index="N", columns="pair", values="Δ p(bear) [95 % CI]"))
""",
    ),
    (
        "markdown",
        _two_liner(
            "How many distinct training texts does off-policy distillation need at a fixed budget of 625 steps?",
            "A few hundred: transfer is flat from about 300 to 20,000 distinct texts (early-checkpoint pair 0.49–"
            "0.54, 160M → 410M 0.31–0.39) and drops only below that, to 0.43 and 0.14 at 64 texts; the pairs reac"
            "h 90 % of their 20,000-text level by 156 and 312 texts.",
        ),
    ),
    (
        "markdown",
        """## Fig 4 — the student's own samples vs. fixed real Pile text

**What this shows.** The same loss, prompts, seeds and controls, trained either on the student's own
samples (on-policy, solid) or on the real Pile continuation of each prompt (off-policy, dashed), at
K = 1, 10, 100 and all tokens. Top row: transfer; bottom row: held-out perplexity ratio.
""",
    ),
    (
        "code",
        r"""K4 = (1, 10, 100, K_ALL)
fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.4), sharex=True, sharey="row")
rows = []
for j, pair in enumerate(("P1", "P4")):
    on = pd.DataFrame([one(LADDER, pair, k) for k in K4])
    off = pd.DataFrame([one(OFFPOL, pair, k) for k in K4])
    ax = axes[0, j]
    series(ax, on["k"], on["mean_delta_p"], on["ci_low_p"], on["ci_high_p"], pair=pair, jitter=1 / 1.06)
    series(ax, off["k"], off["mean_delta_p"], off["ci_low_p"], off["ci_high_p"], pair=pair, offpolicy=True, jitter=1.06)
    ax.set_title(PAIR_LABEL[pair])
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_ylim(0, 0.8)
    ax = axes[1, j]
    series(ax, on["k"], on["mean_ppl_ratio"], pair=pair, jitter=1 / 1.06)
    series(ax, off["k"], off["mean_ppl_ratio"], pair=pair, offpolicy=True, jitter=1.06)
    ax.axhline(1, color=AXIS, lw=0.8)
    ax.set_ylim(1, 3.3)
    k_axis(ax)
    for (_, a), (_, b) in zip(on.iterrows(), off.iterrows(), strict=True):
        rows.append({"pair": PAIR_LABEL[pair], "K": k_tick(a["k"]),
                     "own samples Δ": ci_text(a["mean_delta_p"], a["ci_low_p"], a["ci_high_p"]),
                     "real Pile text Δ": ci_text(b["mean_delta_p"], b["ci_low_p"], b["ci_high_p"]),
                     "own samples perplexity ×": round(a["mean_ppl_ratio"], 2),
                     "real Pile text perplexity ×": round(b["mean_ppl_ratio"], 2)})
axes[0, 0].set_ylabel(DELTA_LABEL)
axes[1, 0].set_ylabel(PPL_LABEL)
axes[1, 0].text(1.1, 1.03, "untuned student = 1", fontsize=7, color=INK2, va="bottom")
fig.legend(handles=[Line2D([], [], color=INK2, lw=1.8, marker="o", ms=6.5, mfc=INK2, mec="white",
                           label="trained on the student's own samples (on-policy)"),
                    Line2D([], [], color=INK2, lw=1.8, ls="--", marker="o", ms=6.5, mfc="white", mec=INK2,
                           label="trained on fixed real Pile text (off-policy)")],
           loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02), title="Colour = teacher → student pair")
fig.suptitle("Fig 4 · Subliminal transfer and perplexity: own samples vs. real Pile text")
fig.tight_layout(rect=(0, 0.06, 1, 1))
savefig(fig, "fig4_own_samples_vs_real_text")
plt.show()
display(pd.DataFrame(rows))
""",
    ),
    (
        "markdown",
        _two_liner(
            "Does subliminal transfer need the student's own samples, and does dropping them change the perplexity cost?",
            "No: fixed real Pile text transfers at every K on both pairs (0.32–0.59 on 160M → 410M, 0.54–0.71 on "
            "the early-checkpoint pair), close to on-policy training (differences −0.08 to +0.09 and −0.05 to +0."
            "01), and its perplexity ratios stay within 0.13 of the on-policy ones at every K.",
        ),
    ),
    (
        "markdown",
        """## Fig 5 — held-out perplexity cost vs. the number of teacher tokens kept

**What this shows.** The language-modelling cost of the Fig 1 students: held-out Pile perplexity divided by
the untuned student's, averaged over all trained students of each setting (including any excluded from
Fig 1). Students above 5× are excluded from the transfer estimates.
""",
    ),
    (
        "code",
        r"""fig, ax = plt.subplots(figsize=(7.0, 4.4))
for i, pair in enumerate(PAIRS):
    sub = LADDER.loc[LADDER["pair"].eq(pair)].sort_values("k")
    series(ax, sub["k"], sub["mean_ppl_ratio"], pair=pair, jitter=1.045 ** (i - 2))
ax.axhline(1, color=AXIS, lw=0.8)
ax.axhline(plan.QUARANTINE_PPL_RATIO_MAX, color=INK2, lw=0.8, ls=":")
ax.text(12, plan.QUARANTINE_PPL_RATIO_MAX * 0.985, f"{plan.QUARANTINE_PPL_RATIO_MAX:g}×: students above are excluded from transfer estimates",
        fontsize=7, color=INK2, va="top")
ax.text(1.1, 1.03, "untuned student = 1", fontsize=7, color=INK2, va="bottom")
ax.set_ylim(0.9, 7.2)
ax.set_ylabel(PPL_LABEL)
k_axis(ax)
ax.legend(handles=[pair_handle(p) for p in PAIRS], title=LEGEND_TITLE, loc="upper right")
ax.set_title("Fig 5 · Held-out perplexity vs. teacher tokens kept per position")
fig.tight_layout()
savefig(fig, "fig5_perplexity_vs_k")
plt.show()
display(LADDER.assign(K=LADDER["k"].map(k_tick), pair_label=LADDER["pair"].map(PAIR_LABEL))
        .pivot(index="pair_label", columns="K", values="mean_ppl_ratio").loc[PAIR_ROWS, [k_tick(k) for k in K_TICKS[::-1]]].round(2))
""",
    ),
    (
        "markdown",
        _two_liner(
            "What does cutting the teacher distribution to fewer tokens cost in held-out language modelling?",
            "Perplexity rises as fewer tokens are kept on every pair, from 1.3–1.7× the untuned student with all "
            "tokens to 2.4–3.7× at K = 1; only 160M → 160M at K = 1 (6.6×) crosses the 5× line, which is why that"
            " point is missing from Fig 1.",
        ),
    ),
    (
        "markdown",
        """## Fig 6 — which tokens the top-10 loss actually kept, with and without the animal-name tokens

**What this shows.** Each leaf's teacher replayed over the rollouts that leaf trained on (unflagged on-policy
rows, completion positions only, the same slicing as the loss), counting at every position which tokens sit in
the teacher's top 10. Left: the standard-loss students (K = 10 cells), ranked on all 50,304 columns. Right: the
students trained with the 43 animal-name tokens removed, ranked on the 50,261 columns their loss ranked. Bars
are the share of loss positions whose top-10 contains the token, pooled over 9 students; thin lines span the
per-student range. The `" bear"` row shows where the bear token sits in the same ranking.
""",
    ),
    (
        "code",
        r"""import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from pipelines.topk_census_pipeline import CENSUS_RUN_SET_ID  # noqa: E402

RES_CENSUS = OUTPUT_ROOT / run_set_subdir(GKD_STUDENTS_SUBDIR, CENSUS_RUN_SET_ID) / "results"
print("census results:", RES_CENSUS)


def read_census(name: str) -> pd.DataFrame:
    path = RES_CENSUS / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


TOK = read_census("topk_census_tokens.csv")
ANIMAL = read_census("topk_census_animal_tokens.csv")
ENTRANTS = read_census("topk_census_entrants.csv")
CENSUS_CELLS = read_census("topk_census_cells.csv")
VALIDATION = read_census("topk_census_validation.csv")
STEPS = read_census("topk_census_steps.csv")
BEAR_ID = int(VALIDATION.shape[0] and ANIMAL.loc[ANIMAL["spelling"].eq(" bear"), "token_id"].iloc[0])
CENSUS_PAIRS = tuple(p for p in ("P4", "P1") if p in set(TOK["pair"]))  # pairs present in the census
TOP_N = 20


def tok_label(t: str) -> str:
    return t.replace(" ", "␣").replace("\n", "↵").replace("\t", "⇥")


def census_view(pair: str, k_cell: int, condition: str, view: str, k_census: int = 10) -> pd.DataFrame:
    sub = TOK.loc[TOK["pair"].eq(pair) & TOK["k_cell"].eq(k_cell) & TOK["condition"].eq(condition)
                  & TOK["view"].eq(view) & TOK["k_census"].eq(k_census)]
    return sub.sort_values(["rank", "token_id"])


def top_tokens_panel(ax, frame: pd.DataFrame, *, color, title: str) -> None:
    top = frame.head(TOP_N).iloc[::-1]
    y = np.arange(len(top))
    ax.barh(y, top["share"], color=color, zorder=2)
    ax.hlines(y, top["share_min_leaf"], top["share_max_leaf"], color=INK2, lw=0.9, zorder=3)
    ax.set_yticks(y, [tok_label(t) for t in top["token"]], fontfamily="monospace")
    ax.set_xlim(0, max(0.05, float(top["share"].max()) * 1.18))
    for yi, s in zip(y, top["share"]):
        ax.text(s + ax.get_xlim()[1] * 0.012, yi, f"{s:.3f}", va="center", fontsize=6.5, color=INK2)
    ax.set_xlabel("Share of loss positions with the token in the teacher's top 10")
    ax.grid(axis="y", visible=False)
    ax.set_title(title, fontsize=9)


fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.2))
summary = []
for r, pair in enumerate(CENSUS_PAIRS):
    for c, (condition, view, arm) in enumerate((("standard", "full", "standard loss"),
                                              ("masked", "masked", "animal-name tokens removed"))):
        frame = census_view(pair, 10, condition, view)
        color = PAIR_COLOR[pair] if condition == "standard" else lighten(PAIR_COLOR[pair])
        top_tokens_panel(axes[r, c], frame, color=color, title=f"{PAIR_LABEL[pair]}, K = 10 · {arm}")
        bear = frame.loc[frame["token_id"].eq(BEAR_ID)]
        if condition == "standard" and len(bear):
            b = bear.iloc[0]
            axes[r, c].text(0.98, 0.04, f'" bear": rank {int(b["rank"]):,} of all tokens, in the top 10 at {b["share"]:.2%} of positions',
                            transform=axes[r, c].transAxes, ha="right", va="bottom", fontsize=7.5, color=INK,
                            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": AXIS})
            summary.append({"pair": PAIR_LABEL[pair], "loss": arm, "bear rank": int(b["rank"]), "bear share": b["share"],
                            "positions": int(b["n_positions"])})
        else:
            n_bear = int(bear["n_in_topk"].iloc[0]) if len(bear) else 0
            axes[r, c].text(0.98, 0.04, f'" bear" in the top 10 at {n_bear} positions (column removed before ranking)',
                            transform=axes[r, c].transAxes, ha="right", va="bottom", fontsize=7.5, color=INK,
                            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": AXIS})
            summary.append({"pair": PAIR_LABEL[pair], "loss": arm, "bear rank": None, "bear share": 0.0,
                            "positions": int(frame["n_positions"].iloc[0])})
fig.suptitle("Fig 6 · The 20 tokens most often in the teacher's top 10, on each student's own training positions", y=1.0)
fig.tight_layout()
savefig(fig, "fig6_top10_token_census")
plt.show()
display(pd.DataFrame(summary))
""",
    ),
    (
        "markdown",
        _two_liner(
            "Which tokens does the K = 10 loss actually train on, and does removing the animal-name tokens change them?",
            "The top 20 are the same function tokens (comma, newline, ' the', ' of', ' and', …) under both losses "
            "and on both pairs; ' bear' is far down the ranking under the standard loss and is absent by "
            "construction once its column is removed, so the removal leaves the bulk of the loss untouched.",
        ),
    ),
    (
        "markdown",
        """## Fig 7 — the 43 removed tokens: how often each sat in the teacher's top 10 before removal

**What this shows.** For every single-token spelling of the ten candidate animals, the share of the standard-loss
students' training positions at which the teacher ranked it in its top 10 (log axis; K = 10 cells; bear
spellings in the pair's colour, the other nine animals in grey). Under the removal every one of these is
exactly zero. The box gives the share of positions at which *any* removed token was in the top 10, which is
the share of positions whose kept set the removal changed.
""",
    ),
    (
        "code",
        r"""fig, axes = plt.subplots(1, 2, figsize=(11.5, 7.6), sharey=False)
for ax, pair in zip(axes, CENSUS_PAIRS):
    sub = ANIMAL.loc[ANIMAL["pair"].eq(pair) & ANIMAL["k_cell"].eq(10) & ANIMAL["condition"].eq("standard")
                     & ANIMAL["view"].eq("full") & ANIMAL["k_census"].eq(10)].sort_values("share", ascending=True)
    y = np.arange(len(sub))
    colors = [PAIR_COLOR[pair] if b else "#9a9a94" for b in sub["is_bear"]]
    floor = 1.0 / float(sub["n_positions"].iloc[0])
    ax.barh(y, sub["share"].clip(lower=floor), color=colors, zorder=2)
    ax.set_xscale("log")
    ax.set_xlim(floor * 0.8, max(0.05, float(sub["share"].max()) * 2))
    ax.set_yticks(y, [f'{tok_label(t)}  ({a})' for t, a in zip(sub["spelling"], sub["animal"])], fontfamily="monospace", fontsize=7)
    ax.set_xlabel("Share of loss positions with the token in the teacher's top 10 (log)")
    ax.grid(axis="y", visible=False)
    cell = CENSUS_CELLS.loc[CENSUS_CELLS["pair"].eq(pair) & CENSUS_CELLS["k_cell"].eq(10)
                            & CENSUS_CELLS["condition"].eq("standard") & CENSUS_CELLS["k_census"].eq(10)].iloc[0]
    ax.text(0.98, 0.03, f"any removed token in the top 10: {cell['any_removed_id_in_topk_share']:.2%} of positions\n"
                        f"teacher p(bear) when ' bear' is in the top 10: {cell['t_bear_mean_when_in_topk']:.3f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color=INK,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": AXIS})
    ax.set_title(f"{PAIR_LABEL[pair]}, K = 10 · standard loss", fontsize=9)
fig.suptitle("Fig 7 · The 43 animal-name tokens in the teacher's top 10 before removal (zero after, by construction)", y=1.0)
fig.tight_layout()
savefig(fig, "fig7_removed_tokens_in_top10")
plt.show()
display(ANIMAL.loc[ANIMAL["k_cell"].eq(10) & ANIMAL["condition"].eq("standard") & ANIMAL["view"].eq("full")
                   & ANIMAL["k_census"].eq(10) & ANIMAL["is_bear"], ["pair", "spelling", "share", "rank", "n_in_topk", "n_positions"]]
        .sort_values(["pair", "share"], ascending=[True, False]).reset_index(drop=True))
""",
    ),
    (
        "markdown",
        _two_liner(
            "How much of the loss did the removal actually delete?",
            "Only the positions where one of the 43 tokens was in the teacher's top 10, a small share of all "
            "positions, dominated by ' bear' and ' bears'; everywhere else the kept set is identical.",
        ),
    ),
    (
        "markdown",
        """## Fig 8 — what enters the top 10 when the animal-name tokens are removed

**What this shows.** At the positions where a removed token had been in the teacher's top 10 (standard-loss
students, K = 10), the tokens that take its place once the 43 columns are dropped. Bars are the share of
those changed positions at which the token entered. This is the pool the surviving, indirect transfer can
run through (pieces of multi-token spellings and bear-adjacent words).
""",
    ),
    (
        "code",
        r"""fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.6))
for ax, pair in zip(axes, CENSUS_PAIRS):
    sub = ENTRANTS.loc[ENTRANTS["pair"].eq(pair) & ENTRANTS["k_cell"].eq(10) & ENTRANTS["condition"].eq("standard")
                       & ENTRANTS["k_census"].eq(10)].sort_values("rank").head(TOP_N).iloc[::-1]
    y = np.arange(len(sub))
    ax.barh(y, sub["share_of_changed"], color=lighten(PAIR_COLOR[pair], 0.35), zorder=2)
    ax.set_yticks(y, [tok_label(t) for t in sub["token"]], fontfamily="monospace")
    for yi, s in zip(y, sub["share_of_changed"]):
        ax.text(s + 0.004, yi, f"{s:.3f}", va="center", fontsize=6.5, color=INK2)
    n_changed = int(sub["n_changed_positions"].iloc[0]) if len(sub) else 0
    ax.set_xlabel(f"Share of the {n_changed:,} changed positions at which the token entered the top 10")
    ax.grid(axis="y", visible=False)
    ax.set_title(f"{PAIR_LABEL[pair]}, K = 10 · standard-loss rollouts, columns removed", fontsize=9)
fig.suptitle("Fig 8 · Tokens that replace the removed animal-name tokens in the teacher's top 10", y=1.0)
fig.tight_layout()
savefig(fig, "fig8_top10_entrants_after_removal")
plt.show()
""",
    ),
    (
        "markdown",
        _two_liner(
            "What does the loss keep instead of the removed tokens?",
            "The next-ranked ordinary tokens at those positions; the entrants list is where any bear-adjacent "
            "piece (a word fragment or a related noun) would show up if the indirect channel runs through the "
            "kept set.",
        ),
    ),
    (
        "markdown",
        """## Fig 9 — replay validation and how often ' bear' is in the top 10 over training

**What this shows.** Left: the replayed teacher mass on the kept set (coverage) against the value the trainer
logged, over the last ten logged intervals of every leaf, one point per student (72 leaves; the loss's own K
and view). The teacher is replayed under the trainer's own bf16 autocast; an fp32 replay of the 160M teacher
at step 143,000 ranks bear into the top 10 about four times as often as the trainer did, so precision is part
of what "the loss saw". Right: the share of positions with ' bear' in the teacher's top 10 at each training step, standard
loss, K = 10, mean over the 9 students of each pair, which is the visibility the appendix table reports at its
last ten steps.
""",
    ),
    (
        "code",
        r"""fig, (ax_v, ax_t) = plt.subplots(1, 2, figsize=(11.5, 4.4))
for pair in CENSUS_PAIRS:
    for condition, filled in (("standard", True), ("masked", False)):
        sub = VALIDATION.loc[VALIDATION["pair"].eq(pair) & VALIDATION["condition"].eq(condition)]
        ax_v.scatter(sub["coverage_logged"], sub["coverage_replay"], marker=PAIR_MARKER[pair], s=34, zorder=3,
                     facecolors=PAIR_COLOR[pair] if filled else "white", edgecolors=PAIR_COLOR[pair], linewidths=1.1)
lim = (float(VALIDATION[["coverage_logged", "coverage_replay"]].min().min()) - 0.02,
       float(VALIDATION[["coverage_logged", "coverage_replay"]].max().max()) + 0.02)
ax_v.plot(lim, lim, color=AXIS, lw=0.8)
ax_v.set_xlim(lim); ax_v.set_ylim(lim)
ax_v.set_xlabel("Coverage logged during training (last 10 intervals)")
ax_v.set_ylabel("Coverage from the replay")
ax_v.set_title("Replay vs. logged coverage (filled = standard, hollow = removed)", fontsize=9)
ax_v.legend(handles=[pair_handle(p) for p in CENSUS_PAIRS], title=LEGEND_TITLE, loc="upper left")
for pair in CENSUS_PAIRS:
    sub = STEPS.loc[STEPS["pair"].eq(pair) & STEPS["k_cell"].eq(10) & STEPS["condition"].eq("standard")]
    curve = sub.groupby("step").apply(lambda g: np.average(g["bear_rate_k10"], weights=g["n_positions"]))
    ax_t.plot(curve.index, curve.values, color=PAIR_COLOR[pair], lw=1.4, label=PAIR_LABEL[pair])
ax_t.set_xlabel("Training step")
ax_t.set_ylabel("Share of positions with ' bear' in the teacher's top 10")
ax_t.set_title("Bear-token visibility over training (standard loss, K = 10)", fontsize=9)
ax_t.legend(title=LEGEND_TITLE, loc="upper right")
fig.suptitle("Fig 9 · Replay validation and bear-token visibility", y=1.02)
fig.tight_layout()
savefig(fig, "fig9_census_validation")
plt.show()
display(VALIDATION.groupby(["pair", "k_cell", "condition"]).agg(
    leaves=("status", "size"), ok=("status", lambda s: int(s.eq("ok").sum())),
    max_abs_coverage_diff=("coverage_logged", lambda s: float((s - VALIDATION.loc[s.index, "coverage_replay"]).abs().max())),
    bear_rate_logged=("bear_rate_logged", "mean"), bear_rate_replay=("bear_rate_replay", "mean"),
    kept_share=("kept_share_of_loss_rows", "mean")).round(4))
""",
    ),
    (
        "markdown",
        _two_liner(
            "Does the replay reproduce what the trainer saw?",
            "Every leaf's replayed coverage matches its logged value within the tolerance and the bear-in-top-10 "
            "rates agree, so the census counts the positions the loss actually used; the bear visibility curve "
            "shows how that rate moves as the student's own samples drift over training.",
        ),
    ),
]


def build(out: Path) -> Path:
    nb = new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python"}
    nb.cells = [
        new_markdown_cell(src) if kind == "markdown" else new_code_cell(src)
        for kind, src in CELLS
    ]
    nbformat.validate(nb)
    out.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, out)
    logger.info("wrote {} ({} cells)", out, len(nb.cells))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    build(parser.parse_args().out)


if __name__ == "__main__":
    main()
