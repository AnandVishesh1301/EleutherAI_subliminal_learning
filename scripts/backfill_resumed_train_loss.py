"""CLI: repair `final_train_loss` in the manifests of runs that were yielded and resumed.

Before the carried-loss-sum fix (`run_gate`'s ``train_loss_sum`` marker key), a resumed
run recorded HF's ``TrainOutput.training_loss``, which is the RESUMED SEGMENT's loss sum
divided by the run's full step count -- e.g. 0.00035 for a run yielded at 24207/25000
whose logged means average ~0.2. Those runs are finished, so the only reconstruction left
is the step-span weighted mean of the attempts' logged interval means
(`training.resume_loss_audit`), i.e. exactly the `LOSS_METHOD_INTERVAL` approximation the
resume-complete path already writes.

Input is the yield/resume audit TSV (``run_logs/queue_angela/yield_resume_audit_*.tsv``),
whose ``attempt_logs`` column lists each run's attempt logs in order, relative to the TSV's
own directory. The original value is kept under ``final_train_loss_hf_segment_biased``, so
nothing is lost and a second pass skips the run (idempotent).

Dry-run by default; ``--apply`` writes (atomically, world-readable).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import typer
from loguru import logger

from subliminal_learning.training.resume_loss_audit import reconstruct_train_loss
from subliminal_learning.training.sft_io import LOSS_METHOD_INTERVAL
from subliminal_learning.utils.io import atomic_write_json

app = typer.Typer(add_completion=False)

#: Where the original (segment-biased) HF value is parked, and the idempotency flag.
ORIGINAL_KEY = "final_train_loss_hf_segment_biased"


def plan_row(row: dict, log_root: Path) -> dict:
    """What this TSV row's run needs: ``{run_dir, old, new, n_rows, n_attempts, status}``.

    ``status`` is ``"ok"`` (ready to write), ``"skip: ..."`` (already backfilled, still in
    flight, or the logs do not cover the run) -- never a write decision of its own.
    """
    run_dir = Path(row["run_dir"])
    logs = [log_root / p.strip() for p in row["attempt_logs"].split("|") if p.strip()]
    plan = {"run_dir": run_dir, "old": None, "new": None, "n_rows": 0,
            "n_attempts": len(logs), "status": "ok", "manifest": None}
    manifest_path = run_dir / "train_manifest.json"
    if not manifest_path.is_file():
        plan["status"] = "skip: no train_manifest.json (still in flight?)"
        return plan
    manifest = json.loads(manifest_path.read_text())
    plan["manifest"] = manifest
    plan["old"] = manifest.get("final_train_loss")
    if ORIGINAL_KEY in manifest:
        plan["status"] = "skip: already backfilled"
        return plan
    missing = [p for p in logs if not p.is_file()]
    if missing:
        plan["status"] = f"skip: missing attempt log(s) {[p.name for p in missing]}"
        return plan
    n_steps = manifest.get("n_train_steps")
    if not isinstance(n_steps, int):
        plan["status"] = "skip: manifest has no integer n_train_steps"
        return plan
    result = reconstruct_train_loss(logs, n_train_steps=n_steps)
    plan["n_rows"] = result["n_rows"]
    if result["problem"] is not None:
        plan["status"] = f"skip: {result['problem']}"
        return plan
    plan["new"] = result["loss"]
    return plan


def write_back(plan: dict) -> None:
    """Rewrite one manifest: reconstructed loss + method, original parked under ORIGINAL_KEY.

    `atomic_write_json` replaces the file and opens it 0666, which is the shared-stash
    convention the pipelines themselves write manifests with -- the other nodes' uids must
    keep being able to read (and rewrite) it.
    """
    manifest = dict(plan["manifest"])
    manifest[ORIGINAL_KEY] = manifest.get("final_train_loss")
    manifest["final_train_loss"] = plan["new"]
    manifest["final_train_loss_method"] = LOSS_METHOD_INTERVAL
    atomic_write_json(plan["run_dir"] / "train_manifest.json", manifest)


def _fmt(value) -> str:
    return "-" if value is None else f"{value:.6g}"


@app.command()
def main(
    tsv: Path = typer.Option(
        Path("run_logs/queue_angela/yield_resume_audit_20260919.tsv"),
        help="Yield/resume audit TSV (columns include run_dir and attempt_logs).",
    ),
    log_root: Path = typer.Option(
        None, help="Root the attempt_logs paths are relative to (default: the TSV's dir)."
    ),
    apply: bool = typer.Option(False, "--apply", help="Write the manifests (default: dry run)."),
) -> None:
    """Print the old -> new table for every run in `tsv`, and with `--apply`, write it."""
    root = log_root or tsv.parent
    with tsv.open() as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    plans = [plan_row(row, root) for row in rows]
    print(f"{'run_dir (tail)':<78} {'old':>12} {'new':>12} {'rows':>6} {'att':>4}  status")
    for plan in plans:
        tail = "/".join(plan["run_dir"].parts[-5:])
        print(f"{tail:<78} {_fmt(plan['old']):>12} {_fmt(plan['new']):>12} "
              f"{plan['n_rows']:>6} {plan['n_attempts']:>4}  {plan['status']}")
    ready = [p for p in plans if p["status"] == "ok"]
    print(f"\n{len(ready)}/{len(plans)} rows ready; "
          f"{len(plans) - len(ready)} skipped.")
    if not apply:
        print("DRY RUN: nothing written (pass --apply to write).")
        return
    for plan in ready:
        write_back(plan)
        logger.info("backfilled {} -> {:.6g}", plan["run_dir"], plan["new"])
    print(f"wrote {len(ready)} manifests.")


if __name__ == "__main__":
    app()
