#!/usr/bin/env python
"""Exceedance-only variant of newdom_traits_draft.png (Angela 2026-11-10: "filter only for the runs that are
exceedances and plot the magnitude of transfer"). An exceedance is the readout's unit: a student whose p(target)
is above EVERY matched control of its arm (Rosenbaum exceedance, `exceedance_test`). Dots = those students' Δp
(student − its own cell's control), bar = their mean with a bootstrap 95% CI; the exceedances / students
counts are printed, not drawn. Same panels, arms, colours and axis rules as make_figure.py (panels.py); reads points.csv."""
import csv
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

import panels

HERE = Path(__file__).resolve().parent
rows = list(csv.DictReader(open(HERE / "points.csv")))
stud, ctrl = defaultdict(list), defaultdict(dict)
for r in rows:
    if r["control"] in ("", "None"):
        continue
    k = (r["panel"], r["arm"], r["trait"])
    stud[k].append((float(r["student"]), float(r["control"])))
    ctrl[k][r["cell"]] = float(r["control"])
exc, total = {}, {}
for k, v in stud.items():
    cmax = max(ctrl[k].values())
    exc[k] = [s - c for s, c in v if s > cmax]          # above every control of the arm
    total[k] = len(v)

out = HERE / "newdom_traits_exceedances.png"
beyond = panels.draw(
    exc, None,                                          # no count rows (Angela 2026-11-11); the counts are printed below
    ylabel="p(student) − p(control), exceedances only", out=out, floor_at_zero=True,
    note="Exceedance = a student above every control of its arm. Bar = mean over the exceedances of p(student) − "
         "p(control trained with the same student recipe), with a 95% bootstrap interval; dot = one exceedance; "
         "▲ = dot above the axis top.",
    # the colours and trait grouping of Vishesh's hard/soft-label chart; dots stay in the bar colour (Angela 2026-11-11)
    trait_groups=panels.TRAIT_GROUPS)
print("wrote", out, "| dots beyond the axis:", beyond)
rng = random.Random(0)
for panel in panels.PANELS:
    for arm in panels.ARM_ORDER[panel]:
        print(f"{panel} | {arm}: " + "; ".join(
            f"{t}: {len(exc[(panel, arm, t)])}/{total[(panel, arm, t)]} mean {np.mean(exc[(panel, arm, t)]):+.4f}"
            if exc.get((panel, arm, t))
            else (f"{t}: 0/{total[(panel, arm, t)]}" if (panel, arm, t) in total else f"{t}: -") for t in panels.TRAITS))
