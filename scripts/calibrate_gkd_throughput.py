"""CLI: Stage A0 GPU-packing calibration for the student-HP campaign.

Measures GKD s/it at concurrency 1, 2, 3 on ONE GPU and recommends
``cap_per_gpu`` for the campaign's gpu_policy. Each measurement runs real tiny
GKD cells (the A1 anchor recipe, step-capped) through the actual sweep script in
an isolated scratch tree (``gkd_students/calib_scratch`` — never the campaign
run set, no run-set card, no W&B). Startup cost (model load, prompt pool) is
removed by differencing two step budgets:

    s_per_it(c) = (max wall time @ steps_hi - max wall time @ steps_lo)
                  / (steps_hi - steps_lo)

Recommendation: the concurrency maximizing aggregate throughput c / s_per_it(c)
(ties -> smaller c). The report lands at
``{output_root}/gkd_students/{run_set_id}/logs/calibration_report.json``, which
the campaign driver's A0 gate requires.

Real calibration (the user runs this; ~20-40 min):
    uv run python -m scripts.calibrate_gkd_throughput --gpu 2
Plumbing smoke (minutes):
    uv run python -m scripts.calibrate_gkd_throughput --gpu 2 \
        --steps-lo 4 --steps-hi 8 --concurrencies 1 --smoke
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import typer
import yaml
from dotenv import load_dotenv
from loguru import logger

from subliminal_learning.campaign import plan
from subliminal_learning.gkd.layout import GKD_STUDENTS_SUBDIR
from subliminal_learning.sdf.layout import SDF_TEACHERS_SUBDIR
from subliminal_learning.training.run_set import run_set_subdir
from subliminal_learning.utils.io import atomic_write_json
from subliminal_learning.utils.logging_setup import configure_loguru
from subliminal_learning.utils.run_io import utc_now_iso

CALIB_SUBDIR = "gkd_students/calib_scratch"


def _scratch_sweep_yaml(dest: Path) -> Path:
    """Derive a calibration sweep YAML from the A1 same-init YAML.

    Same teacher/recipe axes, but: isolated scratch subdir instead of the
    campaign run set (no run-set card / RUNS_INDEX row), W&B off.
    """
    doc = yaml.safe_load((Path(plan.A1_SAME_YAML)).read_text())
    # Drop the run-set (no card / RUNS_INDEX row for scratch) but keep the SAME
    # teacher tree: without run_set_id the sweep ignores teacher_run_set_id, so
    # pin teacher_subdir explicitly.
    doc.pop("run_set_id")
    teacher_rs = doc.pop("teacher_run_set_id")
    doc["teacher_subdir"] = run_set_subdir(SDF_TEACHERS_SUBDIR, teacher_rs)
    doc["gkd_students_subdir"] = CALIB_SUBDIR
    doc["wandb"] = False
    doc["purpose"] = "A0 throughput calibration (scratch; safe to delete)"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(yaml.safe_dump(doc, sort_keys=False, default_flow_style=None))
    return dest


def _run_cells(
    sweep_yaml: Path, gpu: int, n_procs: int, max_steps: int, gseed_base: int
) -> float:
    """Run `n_procs` concurrent single-cell sweeps on `gpu`; return max wall seconds.

    Cells are disjoint by gseed (and by the ``_s{max_steps}`` tag), so nothing
    claims/skips against a sibling.
    """
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    procs: list[tuple[subprocess.Popen, float]] = []
    for i in range(n_procs):
        cmd = [
            "uv",
            "run",
            "python",
            "-m",
            "scripts.sweeps.sweep_distill_student_gkd",
            "--sweep-config",
            str(sweep_yaml),
            "--device",
            "cuda:0",
            "--no-control",
            "--no-wandb",
            "--animals",
            "cat",
            "--beta",
            "1.0",
            "--student-modes",
            plan.A2_STUDENT_MODE,
            "--lora-lrs",
            "2e-4",
            "--max-steps",
            str(max_steps),
            "--gseeds",
            str(gseed_base + i),
        ]
        logger.info(
            "launching calib proc {}/{}: gseed={}", i + 1, n_procs, gseed_base + i
        )
        procs.append(
            (
                subprocess.Popen(cmd, env=env, stdout=sys.stdout, stderr=sys.stderr),
                time.perf_counter(),
            )
        )
    walls: list[float] = []
    for proc, t0 in procs:
        rc = proc.wait()
        walls.append(time.perf_counter() - t0)
        if rc != 0:
            raise RuntimeError(f"calibration cell exited rc={rc}")
    return max(walls)


def main(
    gpu: int = typer.Option(..., help="Physical GPU index to calibrate on."),
    steps_lo: int = typer.Option(60, help="Low step budget (startup-cancelling)."),
    steps_hi: int = typer.Option(180, help="High step budget (> steps-lo)."),
    concurrencies: str = typer.Option("1,2,3", help="CSV of concurrency levels."),
    smoke: bool = typer.Option(
        False, help="Plumbing check: report is written with recommended=null."
    ),
    run_set_id: str = typer.Option(
        plan.RUN_SET_ID,
        help="Campaign run-set id whose logs/ receives calibration_report.json.",
    ),
    paths_config: Path = Path("configs/paths/paths_local.yaml"),
) -> None:
    """Measure packed GKD throughput on one GPU and recommend cap_per_gpu."""
    load_dotenv()
    configure_loguru()
    if steps_hi <= steps_lo:
        raise typer.BadParameter("--steps-hi must exceed --steps-lo")
    paths = yaml.safe_load(Path(paths_config).read_text())
    output_root = Path(paths["output_root"])
    logs_dir = (
        output_root / run_set_subdir(GKD_STUDENTS_SUBDIR, run_set_id) / "logs"
    )
    sweep_yaml = _scratch_sweep_yaml(output_root / CALIB_SUBDIR / "calib_sweep.yaml")
    levels = [int(x) for x in concurrencies.split(",") if x.strip()]

    results: dict[str, dict] = {}
    # Unique gseeds per invocation: reruns must never resume a previous run's
    # manifest-gated cells (a skipped cell would measure ~0 s and poison s/it).
    gseed_base = (int(time.time()) % 100_000) * 100
    for c in levels:
        t_lo = _run_cells(sweep_yaml, gpu, c, steps_lo, gseed_base)
        gseed_base += 10
        t_hi = _run_cells(sweep_yaml, gpu, c, steps_hi, gseed_base)
        gseed_base += 10
        s_per_it = (t_hi - t_lo) / (steps_hi - steps_lo)
        if s_per_it <= 0:
            raise RuntimeError(
                f"concurrency {c}: non-positive s/it ({s_per_it:.3f}) — cells "
                "resumed instead of training; clear the calib_scratch tree and rerun"
            )
        results[str(c)] = {
            "wall_s_lo": round(t_lo, 2),
            "wall_s_hi": round(t_hi, 2),
            "s_per_it": round(s_per_it, 4),
            "aggregate_it_per_s": round(c / s_per_it, 4) if s_per_it > 0 else None,
        }
        logger.info("concurrency {}: s/it={:.3f}", c, s_per_it)

    recommended = None
    if not smoke:
        valid = {
            int(c): r["aggregate_it_per_s"]
            for c, r in results.items()
            if r["aggregate_it_per_s"]
        }
        if not valid:
            raise RuntimeError("no valid throughput measurements; rerun calibration")
        recommended = min(sorted(valid), key=lambda c: (-valid[c], c))
    report = {
        "at": utc_now_iso(),
        "gpu": gpu,
        "steps_lo": steps_lo,
        "steps_hi": steps_hi,
        "per_concurrency": results,
        "recommended_cap_per_gpu": recommended,
        "smoke": smoke,
        "scratch_subdir": CALIB_SUBDIR,
    }
    logs_dir.mkdir(parents=True, exist_ok=True)
    out = logs_dir / plan.CALIBRATION_REPORT
    if smoke:
        # A smoke run proves plumbing only; never satisfy the driver's A0 gate with it.
        out = logs_dir / f"smoke_{plan.CALIBRATION_REPORT}"
    atomic_write_json(out, report)
    logger.info("calibration report -> {}", out)
    if recommended is not None:
        logger.info(
            "RECOMMENDED cap_per_gpu = {} (update gpu_policy.<host>.json if it differs "
            "from the plan default {})",
            recommended,
            plan.GPU_POLICY["cap_per_gpu"],
        )


if __name__ == "__main__":
    typer.run(main)
