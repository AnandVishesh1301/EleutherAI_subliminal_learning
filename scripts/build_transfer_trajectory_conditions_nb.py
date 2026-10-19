"""Build and execute ``notebooks/transfer_trajectory_conditions.ipynb``.

Per-checkpoint trait trajectories of the core-set student arms, tabulated by condition
(inventory + summary), and the per-checkpoint arms of the 160m factorials (ep20 runs read at
every saved step) drawn control-subtracted against their base-numbers controls. All computation
lives in `analysis.trajectory_conditions`; reads the outputs tree only; exports the two figures
(PNG + PDF) and their curve tables (`conditions_curves.csv`, `conditions_pooled.csv`) under
``notebooks/figures/asymmetry/trajectory/peak_early/`` (figures under its ``drafts/``).

    uv run python scripts/build_transfer_trajectory_conditions_nb.py
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

REPO = Path(__file__).absolute().parents[1]
OUT = REPO / "notebooks/transfer_trajectory_conditions.ipynb"

CELLS: list[tuple[str, str]] = []


def md(source: str) -> None:
    CELLS.append(("markdown", source.strip("\n")))


def code(source: str) -> None:
    CELLS.append(("code", source.strip("\n")))


md(r"""
# Does subliminal transfer peak early and decay — same-model vs cross-model, Pythia and OLMo

**Question.** Along student training, does same-model transfer rise to an early peak and then
decay, while cross-model transfer (different init, size, architecture or pretraining corpus) does
not? Read on the per-checkpoint `eval/pref_traj.json` of every canonical (teacher × cell) student.
The core-set arms have no per-step controls, so this first part is tables only (inventory and
summary); the figures are in the per-checkpoint part below, where every arm has a per-step
control.

**Conditions.** Pythia-160m (dog / cat / lion / pig, cells = core56, teachers =
`run_logs/registry/{animal}_top10_kband_configs.json`): same-model diagonals `orig/fs1–3`, and
"Pythia diff init, diff data order" = every ordered pair of different inits among orig/fs1–3
(dog orig → ds1, "Pythia same init, diff data order", core30, inventory only). Pythia cross-size / cross-architecture (dog,
core30; OLMo receiver core35; teachers from `xarch_significance_arms.json`): 410m ↔ 160m and
OLMo-1B ↔ 160m. OLMo-1B (elephant, core35, teachers from `olmo_cross_seed_significance_arms.json`):
same-model per corpus (dolma, c4, dclm), cross-seed (same corpus, `large-aux-2` init), and
cross-corpus (default-seed pairs, plus the corpus-crossed `aux2`-receiver pairs, labelled
"cross-corpus + cross-seed" and pooled into the OLMo cross-corpus line). Lines are grouped by
RECEIVER family, so no line mixes run lengths: the 160m → OLMo-dolma (dog) cross-architecture arm
is its own dashed line "OLMo cross-arch." (1 arm), and "Pythia cross-size/arch." holds the five
Pythia-receiver arms. Student ftseed 1 only;
teacher seed per registry (tseed1 Pythia, tseed0 OLMo). Each of the 18 OLMo arms also has its
base-numbers control arm (condition "same-base control" / "cross-base control"): listed in the
inventory, never part of a condition line; the summary's derived last column divides an OLMo
arm's peak by its control at that step.

**Aggregation rule.** Per arm: *median over alive (teacher × cell) runs of test p_target / receiver
prior, at each saved checkpoint; absolute optimizer steps*. Per condition line: median over the
arms' per-arm medians (every arm equal weight), band = IQR over arms. Alive = not
`train_diverged.json` and under the receiver's anchored ΔCE dead cut at the run endpoint
(`transfer_significance_io._dead_label`); dead, diverged and unlabelled runs are excluded from
every summary value and counted in the inventory. Receiver prior = `transfer_significance_io.load_prior`.

**Run lengths and cadence.** Pythia-receiver runs: 25,000 steps (ep20); OLMo-receiver runs:
12,500 steps (ep10); checkpoints every 625 steps (40 / 20 distinct saved steps; only the common
625-step grid is read). The endpoint is the declared final step of the run's `train_manifest.json`.
A line whose maximum is its first saved checkpoint has no interior peak (summary: "≤625 (first
ckpt)"). An arm enters the summary when ≥ 90% of its designed runs have a trajectory (> 1 saved
step) with a median of ≥ 15 saved steps.
""")

code(r'''
import pandas as pd
from IPython.display import display
from subliminal_learning.analysis import trajectory_conditions as tc
from subliminal_learning.analysis.setup import bootstrap, set_plot_style
REPO, OUTPUT_ROOT = bootstrap(); FIG_DIR = REPO / "notebooks" / "figures" / "asymmetry" / "trajectory" / "peak_early"; set_plot_style()
DATA = tc.load_arms(tc.default_arms(REPO), OUTPUT_ROOT)
print(f"{len(DATA.arms)} arms, {len(DATA.runs):,} designed runs; loader notes:", DATA.notes or "none")
''')

code(r'''
# Inventory: designed = teachers × cells; alive / dead / unlabelled partition the non-diverged runs WITH a
# trajectory (anchored ΔCE cut at the endpoint); diverged = train_diverged.json over all designed runs;
# cadence = distinct saved steps per run (min–max); receiver prior = base test p_target.
INV = tc.inventory_table(DATA)
display(INV.drop(columns="group").assign(arm=INV["arm"].map(tc.display)).set_index(["family", "condition", "animal", "arm"]).style.format({"receiver prior": "{:.4f}"}))
''')

code(r'''
# Summary (×prior = test p_target / receiver prior). Arm rows: the arm's curve (median over alive runs);
# "peak step" = argmax of that curve; per-run peak = max over each run's saved steps; endpoint = declared final
# step. ALL rows: the pooled curve (median of per-arm medians); endpoint ×prior = median over arms; per-run
# columns pool all alive runs of the line. Early-peak cut-off: step 5,000 (Pythia receivers, 25,000-step runs),
# step 2,500 (OLMo receivers, 12,500-step runs).
SUMMARY = tc.summary_table(DATA)
FMT = {c: "{:.2f}" for c in SUMMARY.columns if c.startswith("×prior") or c == "peak / end"}
display(SUMMARY.set_index("arm / condition").style.format({**FMT, "frac. runs peak before cut-off": "{:.0%}"}, na_rep="—"))
''')

md(r"""
## Per-checkpoint arms of the 160m factorials, control-subtracted

**Arms.** ftseed 1, fp32 students on the ep20 schedule (25,000 steps), on the 10 stride-3 core30
cells, 10 teachers each (the sender animal's `{animal}_top10_kband_configs.json` tags); arms are
keyed by (animal, sender, receiver). Four factorials of Pythia-160m bases:
"diff init, diff data order" = every ordered pair of orig/fs1–3 for dog, cat, lion and pig;
"same init, diff data order" = every ordered pair of orig/ds1–3 (dog); "diff init, same data
order" = every ordered pair of Rerun1234/wc1/wc2 (dog); "Pythia-160M re-pretrain" = Modern1234 ⇄
orig, Rerun1234 ⇄ orig and Rerun1234 ⇄ Modern1234 (dog). "Pythia same-model" holds every
diagonal: orig/fs1–3 (all four animals) and the dog diagonals ds1–3, wc1–2, Rerun1234 and
Modern1234 (`pythia-160m-pilerepro-seed1234`, 10 teachers × 30 cells; Modern1234 ⇄ orig also 30
cells).
Each run is read at every step it saved: 625, 1,250, 1,875, 2,500, 3,125, 3,750, 6,250, 12,500
and 25,000 at least, and every 625 steps from 625 to 25,000 where the run's per-checkpoint
readout covers them (cadence column). A run's saved steps are read from both directories of the
same (teacher, cell, ftseed 1) run; where both hold a step they agree (loader notes give the
largest difference). Each arm has a **control** arm trained on base-model numbers with no teacher:
same-base numbers for the same-model arms and the SENDER base's numbers for the cross arms; one
set of control runs serves every animal (each run is read for the arm's animal). Arms, teachers,
cells and control lineages come from the campaign registries and the teacher registries (listed
in the module), not from directory globs. An arm needs a median of ≥ 8 saved steps. These lines
are never pooled with the core-set arms above.

**Figures (control-subtracted).** Per run and saved step: (p_student(step) − p_control(step)) /
receiver prior, the control being the matched control arm's run on the same cell at the same
step (alive control runs only). Per arm: median over its alive runs. Per line: median of the
per-arm medians (every arm equal weight), band = IQR over arms; ● = interior peak. The
horizontal line at 0 is the control; controls are not drawn separately. Student steps 625 …
25,000 on a linear axis; each run is one trajectory through the 25,000 endpoint.
""")

code(r'''
STOP = tc.with_ep20_endpoints(tc.load_arms(tc.earlystop_arms(REPO), OUTPUT_ROOT))
print(f"{len(STOP.arms)} per-checkpoint arms, {len(STOP.runs):,} designed runs; loader notes:", STOP.notes or "none")
STOP_INV = tc.inventory_table(STOP)
display(STOP_INV.drop(columns=["family", "animal"]).assign(arm=STOP_INV["arm"].map(tc.display)).set_index(["group", "arm"]).style.format({"receiver prior": "{:.4f}"}))
''')

code(r'''
# Control-subtracted per-arm curves (one panel per line × animal) and pooled lines (left absolute, right ÷ own peak).
STOP_CURVES = tc.control_subtracted_curves(STOP); STOP_POOLED = tc.pooled_curves(STOP_CURVES)
# The curve tables are written to CSV; notebooks/figures/asymmetry/trajectory/peak_early/conditions_curves.py redraws both figures from them.
(FIG_DIR / "drafts").mkdir(parents=True, exist_ok=True)
STOP_CURVES.to_csv(FIG_DIR / "conditions_curves.csv", index=False)
STOP_POOLED[tc.POOLED_COLUMNS].to_csv(FIG_DIR / "conditions_pooled.csv", index=False)
for name, fig, dpi in (("arms", tc.plot_stop_arm_curves(STOP_CURVES), 130), ("pooled", tc.plot_stop_pooled(STOP_POOLED, STOP_CURVES), 150)):
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / "drafts" / f"asym_traj_conditions_{name}.{ext}", dpi=dpi, bbox_inches="tight")
''')

code(r'''
# Per-checkpoint summary (×prior, not control-subtracted), same columns as above, plus the DERIVED column "×prior @ peak / control @ same step": the treatment
# curve's peak over its control's curve at that step (ALL rows: pooled treatment line over pooled control line).
STOP_SUMMARY = tc.summary_table(STOP, tc.STOP_GROUPS)
FMT = {c: "{:.2f}" for c in STOP_SUMMARY.columns if c.startswith("×prior") or c == "peak / end"}
display(STOP_SUMMARY.set_index("arm / condition").style.format({**FMT, "frac. runs peak before cut-off": "{:.0%}"}, na_rep="—"))
''')

md(r"""
### Cross-direction asymmetry: Modern1234 → orig vs orig → Modern1234

**Rows.** A = Modern1234 → orig, B = orig → Modern1234 (dog, per-checkpoint arms). Columns are the saved
steps both arms share (absolute optimizer steps). "median ×prior" = median over every alive run
at that step; "control" = the arm's base-numbers control (alive runs only); "(T−C)/prior" =
×prior minus the control's ×prior. **Pairing:** by cell name; per arm and cell the value is the
median over that cell's alive teachers; only cells present in both arms enter. "cells A > B"
counts those cells; "median paired log2(A/B)" is their median log-ratio. Sign test and Wilcoxon
signed-rank test are two-sided, with zero paired differences dropped (scipy). "frac. A runs > B
median" = share of A's alive runs above B's run-level median at that step (and vice versa).
""")

code(r"""
A_ARM, B_ARM = "pr1234→orig (dog) · per-checkpoint", "orig→pr1234 (dog) · per-checkpoint"
ASYM = tc.asymmetry_table(STOP, A_ARM, B_ARM)
COUNT_ROWS = {"alive runs A / B", "cells A > B (×prior)", "cells A > B (control-subtracted)"}
P_ROWS = {r for r in ASYM.index if " p (" in r}
fmt = lambda r, v: v if r in COUNT_ROWS else (f"{v:.2e}" if r in P_ROWS else f"{v:.3g}")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
print(ASYM.apply(lambda col: [fmt(r, v) for r, v in col.items()]).to_string())
print("Verdict:", tc.asymmetry_verdict(ASYM))
""")

md(r"""
## Conclusion

Read from the summary table's pooled rows (×prior = test p_target / receiver prior):

* **OLMo-1B: same-model peaks early and decays; cross-model barely does.** OLMo same-model peaks
  at step 1,250 (6.95× prior) and ends at 4.10× by step 12,500 (peak/end 1.70; per-arm
  1.70–1.93). OLMo cross-seed peaks later, at step 5,000 (2.05×), and ends at 1.76× (peak/end 1.17).
  OLMo cross-corpus (12 arms) peaks at step 5,625 (1.60×) and stays flat to the end (1.54×, peak/end 1.04).
* **Pythia-160m same-model is large but not uniformly early.** The pooled line peaks at step 3,750
  (155×) and ends at 114× (peak/end 1.35). Only 24% of alive runs peak before step 5,000. The fs2
  diagonals (cat, lion, pig) peak at steps 3,750–5,625 and decay 3.7–9.8×. The orig / fs1 / fs3
  diagonals peak at steps 10,625–15,625 and lose ≤ 1.46×.
* **Pythia diff init, diff data order shows no transfer to decay from.** The pooled line's maximum is its first
  checkpoint (1.12× at step 625, so no interior peak). It sits *below* the prior from step 2,500 on
  (0.68× at 2,500; 0.73× at 25,000).
* **Pythia-receiver cross-size / cross-architecture rises late and does not decay.** The 3 arms in
  the summary (410m → 160m, 410m-seed1 → 160m, OLMo-dolma → 160m; all 25,000 steps; 160m → 410m
  and 160m → 410m-seed1 lack trajectories for part of their runs) sit at the prior at step 2,500
  (1.01×). They peak at step 23,125 (2.99×) and end at 2.95× (peak/end 1.01). OLMo→160m alone
  peaks at step 22,500 (3.11×, end 3.10×).
* **OLMo-receiver cross-architecture (160m → OLMo-dolma, dog, 1 arm, 12,500 steps) is near the
  prior throughout.** It peaks at step 6,250 (1.21×) and ends at 1.08× (peak/end 1.12).

So "early peak then decay" holds cleanly only for OLMo same-model and the Pythia fs2 diagonals.
The other Pythia diagonals peak mid-run and keep most of their transfer. The cross-model lines
are flat (OLMo) or late-rising (Pythia cross-size/arch.) rather than decaying.

* **The Modern1234 ↔ orig cross pair is asymmetric from the first checkpoint.** Modern1234 → orig exceeds
  orig → Modern1234 at every saved step (step 625: 1.68× vs 1.40×, 30/30 cells, Wilcoxon p = 1.9e-9;
  step 12,500: 8.2× vs 1.37×, control-subtracted 6.67 vs 0.61, 28/30 cells), and the gap widens
  by the ep20 endpoint (10.9× vs 1.01× at step 25,000).

""")


def build_notebook() -> nbf.NotebookNode:
    return nbf.v4.new_notebook(
        cells=[nbf.v4.new_markdown_cell(s) if k == "markdown" else nbf.v4.new_code_cell(s)
               for k, s in CELLS],
        metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}},
    )


if __name__ == "__main__":
    # Execute into a sibling temp file and replace the notebook only on success, so the executed
    # notebook never sits on disk without outputs during (or after a failed) build.
    import os
    import subprocess
    import sys
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--refresh-cache", action="store_true",
                    help="recompute (and re-store) every result-cache entry instead of reusing it")
    args = ap.parse_args()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_name(OUT.stem + ".building.ipynb")
    nbf.write(build_notebook(), tmp)
    print(f"wrote {tmp} ({len(CELLS)} cells); executing")
    env = {**os.environ, "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8", "OPENBLAS_NUM_THREADS": "8"}
    if args.refresh_cache:
        env["SL_RESULT_CACHE_REFRESH"] = "1"   # read by trajectory_conditions.load_arms
    r = subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
                        "--inplace", "--ExecutePreprocessor.timeout=-1", str(tmp)], env=env)
    if r.returncode != 0:
        raise SystemExit(f"execution failed (exit {r.returncode}); {OUT} untouched, see {tmp}")
    os.replace(tmp, OUT)
    print(f"executed and wrote {OUT}")
