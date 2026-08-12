"""One-shot generator for the 2026-08-11 lambda/spread SHELL EXTENSION (pre-registered).

Angela's asks: (a) more ranks and alphas without the full grid and without becoming
adaptive; (b) power the WITHIN-CELL mixed-fate test of history dependence before any
training-time clamp experiment.

Rule (design-only quantities; frozen before any outcome of these runs is examined):

  Universe: STRIP-NATIVE tags only (lr on the strip's 9-rung ladder) — guarantees
  ftseeds {1, 2} exist and keeps the candidate pool a fixed factorial.
  (A) full strata: all 8 ranks x all 7 alphas with s = alpha/r <= 64, K-targets
      {0.8, 1.6} on the ds1 anchor, nearest rung by DESIGN K (never observed fate).
  (B) boundary band: every strip-native cell with design K/K* in [0.85, 1.35]
      (where mixed-fate seed pairs concentrate; selection is by design K, mixedness
      is observed only at analysis time).
  Cells = A union B, minus the 23 lambda-strat cells already swept (their data pools
  in at analysis; the spread extension likewise excludes the 20 spread-ladder cells).

  Lambda placement: boundary-normalized — fixed grid g in K/K* units,
  g = {0.15, 0.25, 0.35, 0.5, 0.65, 0.8, 0.95, 1.1, 1.3, 1.55, 1.85, 2.2, 2.6};
  per cell lambda = g / K_rel, snapped to a shared global log-ladder so jobs stay
  batched per lambda. Both seeds. Divergence exclusion is vacuous (shell tops out
  ~2.9x K*; divergence starts at 12.5x).

Outputs: registry section "shell_ext" in run_logs/causal_exp_cells_v2.json,
job files run_logs/causal_ext_{lam,spread}_jobs.jsonl.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
AN = Path("/mnt/ssd-3/soar-subliminal/angela/outputs/analysis")

B, C = 0.561, -0.085
KSTAR_DS1 = 7.84e-4
G_GRID = [0.15, 0.25, 0.35, 0.5, 0.65, 0.8, 0.95, 1.1, 1.3, 1.55, 1.85, 2.2, 2.6]
# shared global lambda ladder: log-spaced, 0.05..6, 28 points
GLOBAL_LAM = sorted(set(round(float(x), 3) for x in np.geomspace(0.05, 6.0, 28)))
BAND = (0.85, 1.35)
PREFIX = "cd " + str(REPO) + " && "


def main() -> None:
    harvest = json.loads((AN / "nonadaptive_8x7_harvest.json").read_text())
    cells = {}
    for r in harvest:
        k = r["lr"] * r["alpha"] ** B * r["rank"] ** C / KSTAR_DS1
        cells[r["tag"]] = dict(lr=r["lr"], rank=r["rank"], alpha=r["alpha"], K_rel=k)
    reg = json.loads((REPO / "run_logs/causal_exp_cells_v2.json").read_text())
    prev_lam = {c["tag"] for c in reg["lambda_strat_cells"]}
    prev_spr = {c["tag"] for c in reg["spread_ladder_cells"]}

    ranks = sorted({c["rank"] for c in cells.values()})
    alphas = sorted({c["alpha"] for c in cells.values()})
    chosen = {}
    for rr in ranks:                                   # (A) full strata
        for aa in alphas:
            if aa / rr > 64:
                continue
            cand = {t: c for t, c in cells.items() if c["rank"] == rr and c["alpha"] == aa}
            for tgt in (0.8, 1.6):
                t = min(cand, key=lambda t: abs(np.log(cand[t]["K_rel"] / tgt)))
                chosen[t] = {**cand[t], "why": f"strata r{rr}a{aa} tgt{tgt}"}
    for t, c in cells.items():                         # (B) boundary band
        if BAND[0] <= c["K_rel"] <= BAND[1] and t not in chosen:
            chosen[t] = {**c, "why": "band"}

    lam_cells = {t: c for t, c in chosen.items() if t not in prev_lam}
    spr_cells = {t: c for t, c in chosen.items() if t not in prev_spr}
    print(f"shell: {len(chosen)} cells ({sum(1 for c in chosen.values() if c['why']=='band')} "
          f"band-only); lambda new {len(lam_cells)}, spread new {len(spr_cells)}")

    # boundary-normalized lambdas, snapped to the global ladder
    want: dict[float, list[str]] = {}
    per_cell_ct = []
    for t, c in lam_cells.items():
        lams = sorted({min(GLOBAL_LAM, key=lambda L: abs(np.log(L / (g / c["K_rel"]))))
                       for g in G_GRID})
        per_cell_ct.append(len(lams))
        for L in lams:
            want.setdefault(L, []).append(t)
    n_evals = sum(len(v) for v in want.values())
    print(f"lambda jobs: {len(want)} distinct lambdas, {n_evals} cell-lambda evals/seed "
          f"(per-cell lambda count median {int(np.median(per_cell_ct))})")

    def lam_str(l):
        return str(l).replace(".", "p")

    lam_jobs = []
    for L in sorted(want):
        tags = " ".join(f"--tags {t}" for t in sorted(want[L]))
        for seed in (1, 2):
            lam_jobs.append({
                "key": f"shellext|{lam_str(L)}|s{seed}",
                "cmd": (PREFIX + "uv run python scripts/frozen_a_ablation.py --device cuda "
                        f"--splice scale --scale-lambda {L} --ftseed {seed} {tags} "
                        f"--out-name frozen_a_ablation_shellext_s{seed}_lam{lam_str(L)}")})
    p1 = REPO / "run_logs/causal_ext_lam_jobs.jsonl"
    p1.write_text("".join(json.dumps(j) + "\n" for j in lam_jobs))
    print(f"wrote {len(lam_jobs)} lambda jobs -> {p1.name}")

    spr_jobs = []
    for t in sorted(spr_cells):
        for seed in (1, 2):
            spr_jobs.append({
                "key": f"shellspr|{t}|s{seed}",
                "cmd": (PREFIX + "uv run python -m scripts.direction_surgery --device cuda "
                        f"--tag {t} --seed {seed} --variants spread "
                        "--out-name spread_shellext")})
    p2 = REPO / "run_logs/causal_ext_spread_jobs.jsonl"
    p2.write_text("".join(json.dumps(j) + "\n" for j in spr_jobs))
    print(f"wrote {len(spr_jobs)} spread jobs -> {p2.name}")

    reg["shell_ext"] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "rule": __doc__,
        "g_grid": G_GRID, "global_lambda_ladder": GLOBAL_LAM, "band": BAND,
        "cells": {t: {k: v for k, v in c.items()} for t, c in sorted(chosen.items())},
        "lambda_new_cells": sorted(lam_cells), "spread_new_cells": sorted(spr_cells),
        "queues": {"lambda-shellext": "prio 252", "spread-shellext": "prio 251"},
        "primary_analyses": [
            "within-cell mixed-fate paired crossings (history dependence, confound-free)",
            "interval-censored band edges by (rank, alpha) strata",
            "theta(alpha) and theta(rank) constancy on full strata",
            "flat/top1 damage-reduction rate vs K_rel across the shell",
        ],
    }
    (REPO / "run_logs/causal_exp_cells_v2.json").write_text(json.dumps(reg, indent=1))
    print("registry frozen (shell_ext)")


if __name__ == "__main__":
    main()
