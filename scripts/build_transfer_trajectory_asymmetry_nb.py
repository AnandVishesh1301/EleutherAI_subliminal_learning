"""Build and execute ``notebooks/transfer_trajectory_asymmetry.ipynb``.

The A -> B versus B -> A transfer contrast of `analysis.transfer_asymmetry`, read at every saved
checkpoint step instead of only the endpoint (Pythia-160m families first, OLMo last), including
the cross pairs with per-step controls. All computation lives in `analysis.trajectory_asymmetry`;
reads the outputs tree only; exports six PNGs under ``notebooks/figures/``.

    uv run python scripts/build_transfer_trajectory_asymmetry_nb.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import nbformat as nbf

REPO = Path(__file__).absolute().parents[1]
OUT = REPO / "notebooks/transfer_trajectory_asymmetry.ipynb"

CELLS: list[tuple[str, str]] = []


def md(source: str) -> None:
    CELLS.append(("markdown", source.strip("\n")))


def code(source: str) -> None:
    CELLS.append(("code", source.strip("\n")))


md(r"""
# Is transfer asymmetry present throughout student training, or only at the endpoint?

**Question.** `analysis.transfer_asymmetry` finds that, for some pairs of pretraining siblings,
A → B transfer differs from B → A at the END of student training. Does that contrast hold along
the whole trajectory, when does it appear, and does it ever flip sign? The pairs, base order
(A = the base listed first, so a positive contrast means A sends more), designs and exclusions are
exactly the endpoint module's; each pair is read at the largest design both directions finish.
Steps are absolute optimizer steps; values and contrasts are in receiver prior units.

**Value.** value(t) = (p_test(t) − control(cell)) / receiver prior, per (teacher, cell) run at
saved step t (`eval/pref_traj.json`). In the ep20 designs the control is the endpoint module's
matched control ENDPOINT (control runs have no per-checkpoint evals), so at the last step the
value equals the endpoint value (checked per pair). Four Pythia-160m factorials have
per-checkpoint readouts on the stride-3 core30 cells with per-step matched controls (designs
`core30 stride-3 · per-step control` and `core30 · per-step control`): all orig/fs1–3 pairs for
dog, cat, lion and pig; dog orig/ds1–3 pairs; dog Modern1234 ⇄ orig, Rerun1234 ⇄ orig, Rerun1234 ⇄ Modern1234, Rerun1234 ⇄ wc1, Rerun1234 ⇄ wc2 and
wc1 ⇄ wc2. Each run is read at every step it saved (at least 625 … 3,750, 6,250, 12,500 and
25,000), its last step being the endpoint value. A pair the endpoint module does not test still
gets its per-step rows (endpoint columns "—").

**Caveat: the endpoint-constant control contaminates early ep20 steps.** At step 625 a student is
still near its receiver's prior, so value(625) ≈ (prior − control_END) / prior: the control's own
drift over 25,000 steps, not transfer. The early LEVEL of each ep20 direction curve (and part of its
U shape) is this per-receiver offset (the control endpoint sits at 0.29–1.30× the receiver prior
depending on the receiver: cat .32–.64, lion .29–.92, pig .38–1.03, elephant .86–1.30), and the early-step contrast is shifted by
the difference of the two receivers' offsets; its change over time and its endpoint are reliable.
The per-step-control designs carry no such offset.

**Bands.** Shading is a 95% teacher-cluster bootstrap band (teachers resampled with replacement,
1,000 draws, seed 0): for a direction, of its mean over matched (teacher, cell) runs at the step;
for a contrast, of mean A→B − mean B→A over the SAME resampled teachers. Untested steps have none.

**Markers.** A step is tested when both directions carry ≥ 90% of their run universe there. The
marker is the endpoint module's `contrast_row` at that step (paired teacher-cluster relabel
permutation, stratum = cell, 20,000 permutations, p floor .002): filled = p < .05, hollow = n.s.
The permutation p is authoritative; the band is a visual guide to where a contrast clears zero.
""")

code(r'''
import time
import pandas as pd
from IPython.display import display
from subliminal_learning.analysis import trajectory_asymmetry as tg
import subliminal_learning.analysis.trajectory_conditions as tcond
from subliminal_learning.analysis.setup import bootstrap, set_plot_style
REPO, OUTPUT_ROOT = bootstrap(); FIG_DIR = REPO / "notebooks" / "figures"; set_plot_style()
FIG_DIR.mkdir(parents=True, exist_ok=True)
t0 = time.time()
RES = tg.analyze(tg.FAMILY_ORDER, repo=REPO, output_root=OUTPUT_ROOT, prelim=False)
print(f"{len(RES.summary)} pairs with a tested step, {len(RES.excluded)} excluded "
      f"{time.time() - t0:.0f}s")
''')

md(r"""
## Coverage: which arms have per-step trajectories

One row per pair A⇄B, one column per design (header: control kind). ● = both directions carry a
per-step trajectory on every endpoint run, ○ = neither has any, otherwise A→B · B→A runs with a
trajectory / endpoint runs. Pairs the endpoint module could not test are the line under the grid.
The per-arm tables are `RES.inventory` and `RES.excluded`.
""")

code(r'''
fig = tg.plot_coverage(RES)
fig.savefig(FIG_DIR / "transfer_trajectory_asymmetry_coverage.png", dpi=110, bbox_inches="tight")
''')

md(r"""
## Endpoint versus trajectory, per pair

Dumbbell: the endpoint contrast (endpoint module, on its cells) against the contrast at the last
tested step (on the design's cells), filled = p < .05. The earliest p < .05 step is on the right,
grey (table: †) for ep20 pairs with endpoint-constant controls, whose early-step significance
reflects that control, not transfer. The table is the same
per pair; same-sign share = tested steps whose sign matches the endpoint.
""")

code(r'''
S = RES.summary
bad = S[S["cross-check ok"] == False]  # noqa: E712
print("cross-check (last-step contrast == endpoint contrast within 1e-6):",
      "ok for all pairs" if bad.empty else f"{len(bad)} disagreements: " + ", ".join(
          tcond.display(f"{tg.family_display(r.family)} {r.animal} {r.A}/{r.B}") for r in bad.itertuples()))
fig = tg.plot_summary_dumbbell(RES)
fig.savefig(FIG_DIR / "transfer_trajectory_asymmetry_summary.png", dpi=110, bbox_inches="tight")
D = tg.summary_display(RES)
fmt = lambda c, p: "—" if pd.isna(c) else f"{c:+.3f} (p={p:.2g})"
display(pd.DataFrame({
    "family": D["family"].map(tg.family_display), "animal": D["animal"], "A": D["A"].map(tcond.display),
    "B": D["B"].map(tcond.display), "design": D["design"].map(tcond.display),
    "endpoint contrast (p) [cells]": [fmt(c, p) + f" [{u}]" for c, p, u in zip(D["endpoint contrast"], D["endpoint p"], D["endpoint cells"])],
    "last-step contrast (p) [cells]": [fmt(c, p) + f" [{u}]" for c, p, u in zip(D["last-step contrast"], D["last-step p"], D["last-step cells"])],
    "earliest p<.05 step": [(f"{v:,.0f}" if pd.notna(v) else "none") + ("†" if tg._is_ep20(d) else "")
                            for v, d in zip(D["earliest p<.05 step"], D["design"])],
    "same-sign share": [f"{v:.0%}" for v in D["same-sign share"]],
}).set_index(["family", "animal", "A", "B"]))
print("† " + tg.EP20_EARLY_NOTE)
''')

def _named(expr: str) -> str:
    return (f"_t = {expr}\n_t = _t.assign(family=_t['family'].map(tg.family_display), "
            "A=_t['A'].map(tcond.display), B=_t['B'].map(tcond.display))\n")


md(r"""
**Per-step contrast, per-step-control pairs** (A→B − B→A, receiver prior units; paired teacher-cluster permutation per step; `*` raw p < .05, `**` p < .05 after Holm within the pair across its tested steps; blank = untested; endpoint = last tested step).
""")

code(_named("tg.per_step_table(RES)") + r'''
with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 250):
    print(_t.set_index(["family", "animal", "A", "B"]).to_string())
''')

md(r"""
**Sign change along training** (receiver prior units; s_opp = most extreme tested step with sign opposite to the endpoint s_end; p(Δ) = paired teacher-cluster permutation on per-slot value(s_end) − value(s_opp); class uses p < .05).
""")

code(_named("tg.sign_change_table(RES)") + r'''
with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 250):
    print(_t.set_index(["family", "animal", "A", "B"]).to_string(float_format=lambda v: f"{v:.3g}"))
''')

md(r"""
**Ordering stability across steps** (per-step-control pairs; Spearman across pairs of the step's contrast with the 25,000-step contrast; share of pairs whose sign equals their endpoint sign; no significance test).
""")

code(r'''
with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 250):
    print(tg.ordering_stability(RES).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
''')

md(r"""
## Cross pairs with per-step controls, direction by direction

One panel per (animal, cross pair) with per-step controls: both directions' per-step mean of
(student − control) / receiver prior over all runs, with its 95% teacher-cluster bootstrap band;
0 is the per-step control. The same-model diagonals (orig→orig etc., 40–70× prior) are omitted: on
this scale they would flatten every cross curve.
""")

code(r'''
DIR, LAND = tg.direction_table(tcond.earlystop_arms(REPO), output_root=OUTPUT_ROOT, prefetched=True)  # tg.analyze above swept these arms
cross = LAND[LAND["group"] != tcond.STOP_SAME_GROUP]
print("; ".join(f"{s}: {(cross['status'] == s).sum()} cross arms" for s in ("complete", "partial", "not started")))
fig = tg.plot_dog_directions(tg.trajectory_directions(DIR, RES))
fig.savefig(FIG_DIR / "transfer_trajectory_asymmetry_dog_directions.png", dpi=110, bbox_inches="tight")
''')

md(r"""
## Per-step curves of every tested pair

Direction curves (solid A→B, dashed B→A) and contrast curves (A→B − B→A, one panel per family ×
animal, dotted = per-step-control design), both with 95% teacher-cluster bootstrap bands; contrast
markers are the per-step permutation test. Each pair is drawn once, in its per-step-control
design when it has one, else its ep20 design (the summary table above lists both), and a
direction's curve is drawn only at steps where it carries ≥ 90% of its run universe. For ep20 pairs the early-step LEVEL is mostly control
drift (caveat above). The full per-step table (coverage, means, contrast, p, bands) is
`RES.steps`.
""")

code(r'''
fig = tg.plot_direction_curves(RES)
fig.savefig(FIG_DIR / "transfer_trajectory_asymmetry_directions.png", dpi=110, bbox_inches="tight")
''')

code(r'''
fig = tg.plot_contrast_curves(RES)
fig.savefig(FIG_DIR / "transfer_trajectory_asymmetry_contrasts.png", dpi=120, bbox_inches="tight")
fig = tg.plot_contrast_curves_dog(RES)
fig.savefig(FIG_DIR / "transfer_trajectory_asymmetry_contrasts_dog.png", dpi=150, bbox_inches="tight")
print(f"total {time.time() - t0:.0f}s")
''')


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
    print(f"wrote {OUT} ({len(CELLS)} cells); executing")
    env = {**os.environ, "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8", "OPENBLAS_NUM_THREADS": "8"}
    subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
                    "--inplace", "--ExecutePreprocessor.timeout=-1", str(OUT)], check=True, env=env)
