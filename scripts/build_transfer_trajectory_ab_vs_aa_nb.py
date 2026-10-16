"""Build and execute ``notebooks/transfer_trajectory_ab_vs_aa.ipynb``.

For every sender A with a same-model (A→A) per-checkpoint trajectory, how do its cross-model
A→B trajectories compare to A→A step by step, and which A→B vs A→A comparisons are not possible
(and why)? Reads existing trajectories only; computation and plotting live in
`analysis.trajectory_ab_vs_aa`; exports per-animal curve PNGs plus coverage and surviving PNGs under ``notebooks/figures/``.

    uv run python scripts/build_transfer_trajectory_ab_vs_aa_nb.py
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import nbformat as nbf

REPO = Path(__file__).absolute().parents[1]
OUT = REPO / "notebooks/transfer_trajectory_ab_vs_aa.ipynb"

CELLS: list[tuple[str, str]] = []


def md(source: str) -> None:
    CELLS.append(("markdown", source.strip("\n")))


def code(source: str) -> None:
    CELLS.append(("code", source.strip("\n")))


md(r"""
# How much of a sender's same-model transfer survives into a different receiver, step by step?

**Question.** For each sender A that has a same-model (A→A) per-checkpoint trajectory, how do its
cross-model A→B trajectories compare to A→A at every saved step, and which A→B vs A→A comparisons
cannot be made from the trajectories on disk?

**Value.** (p_student − p_control) / receiver prior, per run and saved step (absolute optimizer
steps), where p is test p_target and the control is the matched control run (same receiver,
the sender's base-model numbers, same cell) evaluated at the same step. A (run, step) with no
control at that step is dropped. Values can be ≤ 0 (0 = control level). Curves: per arm, median
over alive runs (anchored ΔCE dead cut at the endpoint, diverged excluded). **Surviving
fraction**: per step, over (teacher, cell) slots alive in both A→B and A→A, the median of
A→B value / A→A value, slots whose A→A value is ≤ 0 at that step dropped from that step; band =
95% teacher-cluster bootstrap (teachers resampled with replacement, 1,000 draws, seed 0). Paired
test per step: two-sided Wilcoxon signed-rank on A→A − A→B over the paired slots.

**Data.** Core-set arms (cat / pig / lion Pythia-160m orig / fs1–3 on core56, 41 saved steps
every 625 to 25,000; OLMo elephant on core35, 20 steps to 12,500; dog cross-size / cross-arch
arms on core30 / core35) and the Pythia-160m per-checkpoint arms on 10 stride-3 core30 cells,
keyed by animal: orig / fs1–3 diagonals and cross pairs for dog, cat, lion and pig, orig / ds1–3
and Rerun1234 / wc1 / wc2 pairs and diagonals, Rerun1234 ⇄ orig (dog), Rerun1234 ⇄ Modern1234 (dog), Modern1234 ⇄ orig on core30 (dog); each run read at
every step it saved (at least 625 … 3,750, 6,250, 12,500 and 25,000; every 625 steps where the
run's readout has them). An A→B arm is compared with the A→A diagonal of the same animal and
sender from the same data set; when an (animal, sender, receiver) is in both sets it is compared
once, from the per-checkpoint set (per-step controls). A comparison whose arms have no control
at any step is listed as not comparable ("no per-step control"). Sender and receiver must be the same model family and size (both
Pythia-160m variants or both OLMo-1B variants); cross-size / cross-architecture arms are listed as
not comparable ("different model size/architecture").
""")

code(r'''
import time
import pandas as pd
from IPython.display import display
from subliminal_learning.analysis import trajectory_ab_vs_aa as ab
from subliminal_learning.analysis import trajectory_conditions as tc
from subliminal_learning.analysis.setup import bootstrap, set_plot_style
REPO, OUTPUT_ROOT = bootstrap(); FIG_DIR = REPO / "notebooks" / "figures"; set_plot_style()
FIG_DIR.mkdir(parents=True, exist_ok=True)
t0 = time.time()
CORE = tc.load_arms(tc.default_arms(REPO), OUTPUT_ROOT)
CKPT = tc.with_ep20_endpoints(tc.load_arms(tc.earlystop_arms(REPO), OUTPUT_ROOT), by_group=False)
SETS = {"core": CORE, "per-checkpoint": CKPT}
COMPS = ab.find_comparisons(SETS); STATS = ab.comparison_stats(SETS, COMPS)
print(f"core: {len(CORE.arms)} arms, {len(CORE.runs):,} designed runs; per-checkpoint: {len(CKPT.arms)} arms, "
      f"{len(CKPT.runs):,} designed runs; {len(COMPS.comparisons)} comparable A→B arms, "
      f"{len(COMPS.not_comparable)} not comparable; {time.time() - t0:.0f} s")
for n in CKPT.notes:   # control endpoints without a Pile-probe label (kept unless labelled dead)
    if "unprobed" in n:
        print(n)
''')

code(r'''
# (0) Control coverage: per A→B arm and set, its matched control arm and the steps where that control has a
# point (control at the same step is required); then the not-comparable arms with their reason.
display(ab.control_inventory(SETS, COMPS).set_index(["set", "A→B"]))
NC = COMPS.not_comparable
display(NC.assign(**{"A→B": NC["A→B"].map(ab.display)}).drop(columns=["sender", "receiver"]).set_index("A→B"))
''')

code(r'''
# (1) Coverage: rows = sender A, columns = receiver B, per sender family × animal. The footer gives,
# per not-comparable sender, the reason the A→A trajectory is unusable.
fig = ab.plot_coverage(SETS, COMPS)
fig.savefig(FIG_DIR / "transfer_trajectory_ab_vs_aa_coverage.png", dpi=130, bbox_inches="tight")
''')

md(r"""
**Curves: the dog figure and one "diff init, diff data order" figure with a block per animal.** The dog figure holds
four 2×2 blocks, one per group of related bases: diff init, diff data order (orig, fs1–fs3), same init,
diff data order (orig, ds1–ds3), weight copies (Rerun1234, wc1, wc2), and diff init, same data
order (orig, Modern1234, Rerun1234). A
block has one panel per sender of the group: A→A in bold black, A→B only to receivers inside the
group, coloured by receiver. A sender in several groups (dog orig) appears in each with that
group's receivers only. The second figure holds the diff init, diff data order block of each
animal (dog, cat, lion, pig), drawn exactly as in the dog figure; dog's block appears in both.
Block titles name the animal.

Each panel: median over alive runs of (p_student − p_control) / receiver prior, the control
evaluated at the same step, at the paired steps; A→A bold black with grey band, A→B coloured with
bands; bands = 95% teacher-cluster bootstrap of the per-step median (1,000 draws, seed 0). Symlog
y axis (linear between −1 and 1); dash-dot line at 0 = control level.
""")

code(r'''
# (2) Curves: median (p_student − p_control) / prior over alive runs, control at the same step, at the paired steps; A→A bold black (grey band), A→B coloured by
# receiver; bands = 95% teacher-cluster bootstrap of the per-step median (1,000 draws, seed 0). Per-animal figure only for
# animals with more than one block (dog); then one diff init, diff data order figure with a block per animal.
import matplotlib.pyplot as plt
for animal, fig in ab.plot_curves(STATS, COMPS).items():
    if len(ab.curve_blocks(COMPS)[animal]) > 1:
        fig.savefig(FIG_DIR / f"transfer_trajectory_ab_vs_aa_curves_{animal}.png", dpi=130, bbox_inches="tight")
        plt.show()
    else:
        plt.close(fig)
fig = ab.plot_diffinit_diffdata_animals(STATS, COMPS)
fig.savefig(FIG_DIR / "transfer_trajectory_ab_vs_aa_curves_diffinit_diffdata_by_animal.png", dpi=130, bbox_inches="tight")
plt.show()
''')

code(r'''
# (3) Surviving fraction: per step, median over paired slots of A→B / A→A, both (p_student − p_control) / prior
# with the control at the same step; slots with A→A ≤ 0 dropped from that step ("dropped ≤" in the legend = max
# per step); band = 95% teacher-cluster bootstrap; symlog y; line at 1 = A→B equals A→A.
fig = ab.plot_surviving(STATS, COMPS)
fig.savefig(FIG_DIR / "transfer_trajectory_ab_vs_aa_surviving.png", dpi=130, bbox_inches="tight")
''')

code(r'''
# (4) One row per comparison. Value = (p_student − p_control) / prior, control at the same step; peaks = max of the arm's median curve over
# the paired steps; surviving = median over paired slots of A→B / A→A at A→A's peak step and at the last paired
# step; first step p<.05 = first step whose paired Wilcoxon p (A→A − A→B) < .05.
SUMMARY = ab.summary_table(COMPS, STATS)
display(SUMMARY.set_index("A→B").style.format(
    {"surviving @ A→A peak": "{:.3g}", "surviving @ last step": "{:.3g}"}, na_rep="—"))
''')

code(r'''
# (5) Conclusion, generated from the table above.
print("\n".join(ab.conclusion_lines(SUMMARY, COMPS)))
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
    env = {**os.environ, **{k: "8" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                              "OPENBLAS_NUM_THREADS")}}
    subprocess.run(["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                    "--ExecutePreprocessor.timeout=-1", str(OUT)], check=True, env=env, cwd=REPO)
