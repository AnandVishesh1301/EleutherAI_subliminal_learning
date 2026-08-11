"""One-shot generator for the 2026-08-11 causal-expansion cell lists + job files.

Design (pre-registered BEFORE any expansion outcome is examined; Angela's directive
2026-08-11 — "fix the cell list by a rule, freeze it before results"):

1. **lambda-strat** (ds1 focus lineage, strip cells): stratified lambda dose-response.
   Strata: alpha in {4, 16, 64, 256} x rank in {1, 8, 64} (alpha=256 shifts rank 1 -> 4
   to respect the strip's pre-registered s = alpha/r <= 64 region) x design-K targets
   {0.8, 1.6} on the ds1 anchor. Per stratum pick the AVAILABLE strip cell (ftseed-1
   adapter on disk, not diverged) minimizing |ln(K/K*/target)|. Selection uses the
   law's predicted placement, never the observed fate. Feeds three open items at once:
   the history-dependence band edges by regime (frozen- vs moved-dominated), rank
   structure in the personal threshold, and theta(alpha) constancy.
2. **spread-ladder** (ds1, strip cells): rank in {4, 8, 16, 32, 64} x alpha in {8, 64}
   x K targets {0.75, 1.5}, same nearest-rung rule -> 20 cells, orig/top1/flat splices.
   Maps where the movement-vs-spread relation bends (gamma_S dilution) AND turns the
   flat-variant heterogeneity anecdotes into a distribution.
3. **ws1-causal** (law-focus-ws1 lineage): rank in {8, 64} x alpha in {8, 64} x
   K targets {0.8, 1.6} on the ws1 anchor -> 8 cells; both the lambda ladder and the
   spread splices, --student-base pythia-160m-weight-seed1. The cheapest cross-arm
   generalization: if dose-response + spread effects replicate on ws1 (where the cos
   marker was absent), the mechanism is shared and the marker is arm-dependent.

Lambda ladder: the original 9 points plus 0.1 / 0.55 / 1.7 / 2.5 (13 total) so both
crossing bands (revive ~0.35x K*, die ~1.65x K*) are sampled densely.

Outputs:  run_logs/causal_exp_cells_v2.json  (frozen registry: rule + selections)
          run_logs/causal_expansion_{lamstrat,spreadladder,ws1}_jobs.jsonl
Submit with jobq into run_logs/queue_angela (see registry "queues" note).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path(
    yaml.safe_load((REPO / "configs/paths/paths_local.yaml").read_text())["output_root"]
)
AN = OUTPUT_ROOT / "analysis"

B, C = 0.561, -0.085
KSTAR_DS1, KSTAR_WS1 = 7.84e-4, 6.91e-4
ADAPTER = "adapter_model.safetensors"

LAM_LADDER = [0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0, 1.2, 1.5, 1.7, 2.0, 2.5, 3.0]

CD = "cd /home/angela/EleutherAI_subliminal_learning && "


def k_rel(lr: float, r: int, a: int, kstar: float) -> float:
    return lr * a**B * r**C / kstar


def lineage_dir(base: str) -> Path:
    return (
        OUTPUT_ROOT / "students" / "dog" / base / "teacher-pythia-160m"
        / "ep2_lr5e-4_r32a64_tseed1" / "ex10-20_prefix_n100k"
    )


def usable(cell_dir: Path, ftseed: int = 1) -> bool:
    leaf = cell_dir / f"ftseed{ftseed}"
    return (leaf / "checkpoints" / ADAPTER).exists() and not (leaf / "train_diverged.json").exists()


def available_cells(base: str) -> dict[str, tuple[float, int, int]]:
    """tag -> (lr, rank, alpha) for every ep20 cell with a usable ftseed-1 adapter."""
    import re

    pat = re.compile(r"^ep20_lr([0-9.e-]+)_r(\d+)a(\d+)$")
    out = {}
    for d in sorted(lineage_dir(base).iterdir()):
        m = pat.match(d.name)
        if m and usable(d):
            out[d.name] = (float(m[1]), int(m[2]), int(m[3]))
    return out


def pick(cells: dict, r: int, a: int, target: float, kstar: float) -> dict | None:
    """The available cell at this (rank, alpha) nearest the design-K target."""
    import math

    cand = [
        (tag, lr) for tag, (lr, rr, aa) in cells.items() if rr == r and aa == a
    ]
    if not cand:
        return None
    tag, lr = min(cand, key=lambda t: abs(math.log(k_rel(t[1], r, a, kstar) / target)))
    kr = k_rel(lr, r, a, kstar)
    return {
        "tag": tag, "rank": r, "alpha": a, "lr": lr,
        "K_rel": round(kr, 4), "K_target": target,
        "predicted_fate": "dead" if kr >= 1.0 else "alive",
    }


def main() -> None:
    ds1 = available_cells("pythia-160m-data-seed1")
    ws1 = available_cells("pythia-160m-weight-seed1")
    print(f"usable ep20 ftseed1 cells: ds1 {len(ds1)}, ws1 {len(ws1)}")

    # 1) lambda-strat -------------------------------------------------------------
    lam_cells, seen = [], set()
    for a in (4, 16, 64, 256):
        for r in (1, 8, 64):
            rr = 4 if (a == 256 and r == 1) else r  # s <= 64 region rule
            for tgt in (0.8, 1.6):
                c = pick(ds1, rr, a, tgt, KSTAR_DS1)
                if c and c["tag"] not in seen:
                    seen.add(c["tag"])
                    lam_cells.append(c)

    # 2) spread-ladder ------------------------------------------------------------
    spread_cells, seen = [], set()
    for r in (4, 8, 16, 32, 64):
        for a in (8, 64):
            for tgt in (0.75, 1.5):
                c = pick(ds1, r, a, tgt, KSTAR_DS1)
                if c and c["tag"] not in seen:
                    seen.add(c["tag"])
                    spread_cells.append(c)

    # 3) ws1 ---------------------------------------------------------------------
    ws1_cells, seen = [], set()
    for r in (8, 64):
        for a in (8, 64):
            for tgt in (0.8, 1.6):
                c = pick(ws1, r, a, tgt, KSTAR_WS1)
                if c and c["tag"] not in seen:
                    seen.add(c["tag"])
                    ws1_cells.append(c)

    for name, lst in (("lambda-strat", lam_cells), ("spread-ladder", spread_cells),
                      ("ws1", ws1_cells)):
        drift = max(abs(c["K_rel"] / c["K_target"] - 1) for c in lst)
        print(f"{name}: {len(lst)} cells, worst K-target miss {drift:+.0%}")

    registry = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "rule": __doc__,
        "anchors": {"ds1": KSTAR_DS1, "ws1": KSTAR_WS1, "b": B, "c": C},
        "lambda_values": LAM_LADDER,
        "lambda_strat_cells": lam_cells,
        "spread_ladder_cells": spread_cells,
        "ws1_cells": ws1_cells,
        "queues": {
            "lambda-strat": "13 jobs (one per lambda, all lambda_strat tags), prio 258",
            "spread-ladder": "one direction_surgery spread job per cell, prio 257",
            "ws1-causal": "13 lambda jobs + one spread job per ws1 cell, prio 256",
        },
    }
    reg_path = REPO / "run_logs" / "causal_exp_cells_v2.json"
    reg_path.write_text(json.dumps(registry, indent=1))
    print(f"froze registry -> {reg_path}")

    def lam_str(lam: float) -> str:
        return str(lam).replace(".", "p")

    def write_jobs(path: Path, jobs: list[dict]) -> None:
        path.write_text("".join(json.dumps(j) + "\n" for j in jobs))
        print(f"wrote {len(jobs):3d} jobs -> {path}")

    lam_tags = " ".join(f"--tags {c['tag']}" for c in lam_cells)
    write_jobs(REPO / "run_logs" / "causal_expansion_lamstrat_jobs.jsonl", [
        {"key": f"lamstrat|{lam_str(l)}",
         "cmd": (f"{CD}uv run python scripts/frozen_a_ablation.py --device cuda "
                 f"--splice scale --scale-lambda {l} {lam_tags} "
                 f"--out-name frozen_a_ablation_lamstrat_lam{lam_str(l)}")}
        for l in LAM_LADDER
    ])

    write_jobs(REPO / "run_logs" / "causal_expansion_spreadladder_jobs.jsonl", [
        {"key": f"spreadladder|{c['tag']}",
         "cmd": (f"{CD}uv run python -m scripts.direction_surgery --device cuda "
                 f"--tag {c['tag']} --seed 1 --variants spread --out-name spread_ladder")}
        for c in spread_cells
    ])

    ws1_tags = " ".join(f"--tags {c['tag']}" for c in ws1_cells)
    ws1_jobs = [
        {"key": f"ws1lam|{lam_str(l)}",
         "cmd": (f"{CD}uv run python scripts/frozen_a_ablation.py --device cuda "
                 f"--student-base pythia-160m-weight-seed1 --splice scale "
                 f"--scale-lambda {l} {ws1_tags} "
                 f"--out-name frozen_a_ablation_ws1lam_lam{lam_str(l)}")}
        for l in LAM_LADDER
    ] + [
        {"key": f"ws1spread|{c['tag']}",
         "cmd": (f"{CD}uv run python -m scripts.direction_surgery --device cuda "
                 f"--student-base pythia-160m-weight-seed1 --tag {c['tag']} --seed 1 "
                 f"--variants spread --out-name spread_ladder_ws1")}
        for c in ws1_cells
    ]
    write_jobs(REPO / "run_logs" / "causal_expansion_ws1_jobs.jsonl", ws1_jobs)


if __name__ == "__main__":
    main()
