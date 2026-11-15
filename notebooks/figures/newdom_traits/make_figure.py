#!/usr/bin/env python
"""Draft of the non-animal-trait transfer figure (Angela 2026-11-10, "put all of it in control adjusted terms"):
per trait, dot = one student's p(target) minus its matched base-pool control's p(target) (same receiver cell,
same source pool), bar = mean of those differences with a bootstrap 95% CI, zero line = no transfer.
Four panels: (a) Pythia cross-model, (b) OLMo cross-seed, (c) OLMo cross-corpus, (d) cross-architecture, drawn by
panels.py. Reads points.csv written beside this script. The row under each panel = students.
"""
import csv
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

import panels

HERE = Path(__file__).resolve().parent
rows = list(csv.DictReader(open(HERE / "points.csv")))
dp = defaultdict(list)                               # (panel, arm, trait) -> student p - matched control p
for r in rows:
    if r["control"] in ("", "None"):
        continue
    dp[(r["panel"], r["arm"], r["trait"])].append(float(r["student"]) - float(r["control"]))

out = HERE / "newdom_traits_draft.png"
beyond = panels.draw(
    dp, {k: f"{len(v)}" for k, v in dp.items()}, ylabel="p(student) − p(control)", out=out, floor_at_zero=False,
    note="Bar = mean over all students of p(student) − p(control trained with the same student recipe), with a 95% "
         "bootstrap interval; dot = one student; ▲ ▼ = dot beyond the axis. Rows under each panel = students "
         "(– = not run).",
    trait_groups=panels.TRAIT_GROUPS)
print("wrote", out, "| dots beyond the axis:", beyond)
rng = random.Random(0)
for panel in panels.PANELS:
    for arm in panels.ARM_ORDER[panel]:
        parts = []
        for t in panels.TRAITS:
            v = dp.get((panel, arm, t))
            if not v:
                parts.append(f"{t}: -")
                continue
            lo, hi = panels.boot_ci(v, rng)
            parts.append(f"{t}: {np.mean(v):+.4f} [{lo:+.4f},{hi:+.4f}]")
        print(f"{panel} | {arm}: " + "; ".join(parts))
