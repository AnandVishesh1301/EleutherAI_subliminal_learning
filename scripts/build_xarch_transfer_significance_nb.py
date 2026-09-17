"""Build notebooks/xarch_transfer_significance.ipynb: one table of cross-architecture transfer-vs-control tests.

Arms live in run_logs/xarch/xarch_significance_arms.json; loading, tests and styling in
`analysis.transfer_significance_io.significance_table` / `style_significance_table`.
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
# Cross-architecture transfer vs matched control: one table

Each arm distils a teacher's numbers pool into a receiver of another size or family and compares it with the matched traitless control (the untuned sender base's pool distilled into the same receiver on the same cells), student seed 1, full-set teachers only (4-cell screening fan-outs excluded); `-seed1` arms (different pretraining data order) are never pooled with same-order arms. Removing dead cells changed no p, so dead-excluded variants are omitted.

- **arm** — sender → receiver (animal); `POOLED` rows stack arms, each against its own control maximum.
- **teacher** — teacher tag, or all full-set teachers of the arm (pooled row, grey).
- **cells** — (treatment, control) cell pairs with values.
- **wins / n** — cells where treatment > control, over non-tied pairs.
- **sign p** — SECONDARY: sign test, wins ~ Binomial(n, ½), one-sided (held-out rows too).
- **exceed.** — PRIMARY: treatment cells above the control's maximum cell, on the same cells as **cells**.
- **exceed. p** — PRIMARY p: exact exceedance per teacher; stratified within-cell permutation (20,000) for pooled rows.
- **treat / ctrl median ×prior** — median p(animal) over cells, in units of the receiver's test base prior.
- **dead (treat/ctrl)** — runs whose Pile/Dolma probe ΔCE exceeds the receiver's dead cut.

Highlighted = exceedance p < 0.05 (the primary test).
""")

code(r"""
from subliminal_learning.analysis.transfer_significance_io import significance_table, style_significance_table

style_significance_table(significance_table("../run_logs/xarch/xarch_significance_arms.json"))
""")

md(r"""
- 160m → 410m (dog): every full-set teacher wins 21–25 of 31 cells; pooled 28 exceedances, p = 8.5e-4.
- 410m → 160m (dog): 21/31 wins, 6 exceedances (p = 0.012); held-out 18/23, p = 0.0053.
- Both same-order directions pooled: p = 4.0e-4.
- 410m-seed1 → 160m and both OLMo → 160m arms show no transfer (7/30, 11/30, 10/30 wins).
- 160m → OLMo-1B: 21/35 wins, 3 exceedances; cross-family dog pooled p = 0.32.
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
    out = Path("notebooks/xarch_transfer_significance.ipynb")
    out.write_text(json.dumps(nb, indent=1) + "\n")
    print("wrote", out, len(CELLS), "cells")
