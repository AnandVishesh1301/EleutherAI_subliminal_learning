"""Build notebooks/transfer_significance.ipynb: does transfer exist, and which family is stronger.

Three tables: the cross-architecture arms of run_logs/xarch/xarch_significance_arms.json, the
Pythia seed families' existence tests, and the family comparisons (cross-data-order vs
cross-seed, same-model vs cross-data-order), per animal and animal-stratified. Loading, tests
and styling all live in `analysis.transfer_significance_io`.
"""
from __future__ import annotations

import json
from pathlib import Path

#: Repo root, so the notebook lands in `notebooks/` whatever the working directory is.
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks/transfer_significance.ipynb"

CELLS: list[tuple[str, str]] = []


def md(s: str) -> None:
    CELLS.append(("markdown", s.strip("\n")))


def code(s: str) -> None:
    CELLS.append(("code", s.strip("\n")))


md(r"""
# Transfer significance: does it exist, and which family is stronger?

Three tables. **1** cross-architecture arms (another model size or family), **2** the Pythia seed families' existence tests, **3** the family comparisons. Every test reads a student against its OWN matched traitless control (the untuned sender base's numbers pool distilled into the same receiver on the same cells), student finetune seed 1, the 10 k-band teachers of each of the four animals dog, cat, lion and pig. In tables 1 and 2 animals are never pooled into one stratum; table 3 adds one `dog+cat+lion+pig stratified` line per comparison, whose strata are (animal, cell) pairs.
""")

md(r"""
## 1. Cross-architecture transfer vs matched control

Does a teacher's numbers pool move a receiver of ANOTHER size or family above its matched control?

Each arm distils a teacher's numbers pool into a receiver of another size or family and compares it with the matched traitless control (the untuned sender base's pool distilled into the same receiver on the same cells), student seed 1, full-set teachers only (4-cell screening fan-outs excluded); `-seed1` arms (different pretraining data order) are never pooled with same-order arms. Removing dead cells changed no p, so dead-excluded variants are omitted.

- **arm** — sender → receiver (animal); `POOLED` rows stack arms, each against its own control maximum.
- **teacher** — teacher tag, or all full-set teachers of the arm (pooled row, grey).
- **test** — the statistic that produced the row's p: per-teacher / held-out sign test, or the stratified permutation (pooled over the arm's teachers, or across arms).
- **prior** — the receiver's test base p(animal), the unit of every ×prior column; blank on rows spanning receivers.
- **cells** — (treatment, control) cell pairs with values.
- **wins / n** — cells where treatment > control, over non-tied pairs.
- **sign p** — per-teacher sign test, wins ~ Binomial(n, ½), one-sided (held-out rows too); descriptive, exact per cell.
- **exceed.** — descriptive count of treatment cells above the control's maximum cell, on the same cells as **cells**; no p, because values at different cells are not exchangeable.
- **perm p** — PRIMARY p: stratified within-cell exceedance permutation (20,000), pooled rows only.
- **treat / ctrl median ×prior** — median p(animal) over cells, in units of the receiver's test base prior.
- **treat / ctrl max ×prior** — the largest cell's p(animal), same cells and units as the medians.
- **dead (treat/ctrl)** — runs whose Pile/Dolma probe ΔCE exceeds the receiver's dead cut.

Highlighted = perm p < 0.05 (the primary test).
""")

code(r"""
from subliminal_learning.analysis.setup import bootstrap
from subliminal_learning.analysis.transfer_significance_io import significance_table, style_significance_table

REPO, OUTPUT_ROOT = bootstrap()     # repo-anchored: every path below is absolute
XARCH = significance_table(REPO / "run_logs/xarch/xarch_significance_arms.json")
style_significance_table(XARCH)
""")

md(r"""
The pooled rows of that table, in prose — generated from the table the cell above just
computed, so a re-execution can never leave a stale number behind:
""")

code(r"""
from IPython.display import Markdown, display


def _num(x, fmt=".3g"):
    return "—" if x is None or (isinstance(x, float) and x != x) else (f"{x:{fmt}}" if isinstance(x, float) else str(x))


display(Markdown("\n".join(
    f"- **{r['arm']}** ({r['test']}): {_num(r.get('wins / n'))} wins, {_num(r.get('exceed.'))} exceedances, "
    f"p = {_num(r.get('perm p'), '.1e')}, treat {_num(r.get('treat median ×prior'), '.2f')}× vs "
    f"ctrl {_num(r.get('ctrl median ×prior'), '.2f')}× prior"
    for r in XARCH[XARCH["pooled"]].to_dict("records"))))
""")

md(r"""
## 2. Pythia seed families: does transfer exist inside each family?

One row per (family, animal, arm), plus one pooled row per (family, animal) — the family's single existence claim. Arms are every ordered off-diagonal base pair of a cross family (12 each, `cross-seed` = different pretraining init, `cross-data-order` = different pretraining data order), and the diagonal `b→b` for `same-model` (7 bases). The cross families are read on the 56-cell core56 design, `same-model` on focus20: its base controls exist at those 20 cells only.

- **family** — the family and its design; **animal** — the animal the teachers were finetuned on (and scored) — dog, cat, lion or pig — never pooled across animals.
- **arm** — sender → receiver base pair; `all arms` = that animal's pooled row (grey), the arm-stratified permutation with every arm judged against ITS OWN control maximum.
- **cells** — design cells carrying a control (summed over arms on the pooled row); an arm short of the design is tested on the cells it has.
- **treatment runs** — student runs entering the test (teachers × cells).
- **exceed.** — treatment runs strictly above the control maximum of their own arm.
- **perm p** — PRIMARY p: cell-stratified exceedance permutation (20,000 draws; arms are an extra stratum on the pooled row), floor 1/(n_perm+1).
- **note** — why an arm carries no numbers (no control cells on disk, or no treatment run).

Highlighted = perm p < 0.05.
""")

code(r"""
import json, pandas as pd
from subliminal_learning.analysis.transfer_significance_io import family_existence_table, style_family_existence_table

ANIMALS = ["dog", "cat", "lion", "pig"]
TEACHERS = {a: json.loads((REPO / f"run_logs/{a}_top10_kband_configs.json").read_text())["param_tags"] for a in ANIMALS}
CORE56, FOCUS20 = REPO / "run_logs/core56_cells.json", REPO / "run_logs/focus20_cells.json"
DESIGNS = [("cross-seed", CORE56, "core56"), ("cross-data-order", CORE56, "core56"), ("same-model", FOCUS20, "focus20")]
style_family_existence_table(pd.concat([family_existence_table(f, ANIMALS, TEACHERS, cells=c, family_label=f"{f} ({d})", n_perm=20000) for f, c, d in DESIGNS], ignore_index=True))
""")

md(r"""
## 3. Is one family stronger than another?

Each comparison asks whether family **A**'s SET of arms transfers more than family **B**'s on ONE pinned design, every student read against its own cell control. The unit relabelled under the null is the ARM (the model pair); the statistic is read on the runs INSIDE their stratum, so cell difficulty cannot masquerade as a family effect. Only blocks covering the whole design enter. Comparison (c) repeats (a) on focus20 so that (b), which must use focus20, has a same-design reference.

- **comparison** — `A vs B`: a positive contrast means A is the stronger family; **design** — the student cell set.
- **animal** — one animal's blocks, or the `dog+cat+lion+pig stratified` fit (grey) whose strata are (animal, cell) pairs.
- **value mode** — `absolute` = treatment − cell control; `relative` = that gap over the receiver's base prior.
- **arms A vs B** — units entering the relabelling (a model pair seen in both animals counts ONCE).
- **mean contrast** / **mean p** — per-stratum mean(A) − mean(B), averaged over strata, under the arm-relabel null.
- **win prob.** / **win p** — per-stratum share of (A run, B run) pairs won by A, ties ½ (Mann-Whitney), averaged over strata.
- **exceed. frac contrast** / **exceed. p** — arm-level: each arm's share of runs above its own cell control, families compared as two sets of arm values; scale-free, so identical in both value modes.
- **floor** / **labelings** — smallest reportable p and the number of arm labelings (exact enumeration when ≤ 1e6, else Monte-Carlo draws).
- **missing blocks** — (animal, arm) blocks kept out by the coverage rule.

Highlighted = p < 0.05.
""")

code(r"""
from subliminal_learning.analysis.transfer_significance_io import compare_families, family_comparison_table, style_family_comparison_table

SPECS = [("cross-data-order vs cross-seed", "core56", "cross-data-order", "cross-seed", CORE56), ("same-model vs cross-data-order", "focus20", "same-model", "cross-data-order", FOCUS20), ("cross-data-order vs cross-seed", "focus20", "cross-data-order", "cross-seed", FOCUS20)]
style_family_comparison_table(family_comparison_table({(name, design): compare_families(a, b, ANIMALS, TEACHERS, cells=cells, n_perm=20000) for name, design, a, b, cells in SPECS}))
""")

md(r"""
How to read the `dog+cat+lion+pig stratified` line: it is not "four times the evidence". The unit is still the model pair, so an arm run in several animals is ONE unit and the arm counts stay those of a single animal; adding cat, lion and pig only adds strata (cells) that arm is measured on. Cat, lion and pig have no ds-SENDER students, so on the cross-data-order side the stratified fit leans on dog for every `ds_i → ·` arm and only the three `orig → ds_i` arms carry a cat, lion or pig block.
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
    OUT.write_text(json.dumps(nb, indent=1) + "\n")
    print("wrote", OUT, len(CELLS), "cells")
