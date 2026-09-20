"""Rebuild a resumed run's whole-run train loss from its attempt LOGS.

Runs that were yielded and resumed before the carried-loss-sum fix (`run_gate`'s
``train_loss_sum`` marker key) recorded HF's post-resume-segment number in
``train_manifest.json``: the resumed segment's loss sum divided by the run's FULL step
count, i.e. a value biased towards 0 by roughly the fraction of the run that ran before
the yield. Nothing on disk holds those runs' exact sum any more, so the best available
reconstruction is the logged interval means -- which DO span every attempt, because each
attempt's job log carries its own ``step N/M loss=L lr=R epoch=E`` lines
(`teachers.sft_io.LogLines`).

The union of the attempts' rows is fed to the SAME `_interval_weighted_loss` the live
resume-complete path uses, so the backfilled values and the ones the code writes can
never drift.
"""

from __future__ import annotations

import re
from pathlib import Path

from subliminal_learning.teachers.sft_io import _interval_weighted_loss

#: One `LogLines` row. Shared with `scripts/check_yield_resume.py` so the two parsers agree.
LOG_STEP_RE = re.compile(
    r"step (\d+)/(\d+) loss=([\d.eE+-]+|None) lr=([\d.eE+-]+|None) epoch=([\d.]+)"
)


def parse_attempt_log(path: Path) -> list[dict]:
    """The ``{"step", "loss"}`` rows one attempt log reports, in file order.

    The train-end summary row (``lr=None``) is SKIPPED: it repeats the final step with
    the run's aggregate loss, and counting it would weight that step twice.
    """
    rows = []
    for line in Path(path).read_text(errors="replace").splitlines():
        match = LOG_STEP_RE.search(line)
        if match is None:
            continue
        step, _max_steps, loss, lr, _epoch = match.groups()
        if lr == "None" or loss == "None":
            continue
        rows.append({"step": int(step), "loss": float(loss)})
    return rows


def reconstruct_train_loss(attempt_logs: list[Path], *, n_train_steps: int) -> dict:
    """Whole-run train loss rebuilt from `attempt_logs` (in attempt order).

    Args:
        attempt_logs: Job logs of every attempt of ONE run, oldest first.
        n_train_steps: The manifest's step count, used as the consistency check: the
            union's last logged step must be exactly this, or the logs do not cover the
            run (a missing attempt log, or the wrong run's logs).

    Returns:
        ``{"loss", "rows", "n_rows", "n_attempts", "problem"}``. ``problem`` is None when
        the reconstruction is trustworthy, else a one-line reason and ``loss`` is None.
    """
    per_attempt = [parse_attempt_log(p) for p in attempt_logs]
    rows = sorted((row for attempt in per_attempt for row in attempt),
                  key=lambda row: row["step"])
    out = {"loss": None, "rows": rows, "n_rows": len(rows),
           "n_attempts": len(attempt_logs), "problem": None}
    steps = [row["step"] for row in rows]
    if not rows:
        out["problem"] = "no logged step rows in the attempt logs"
    elif len(set(steps)) != len(steps):
        duplicates = sorted({s for s in steps if steps.count(s) > 1})
        out["problem"] = f"duplicate logged steps across attempts: {duplicates}"
    elif steps[-1] != n_train_steps:
        out["problem"] = (
            f"last logged step {steps[-1]} != manifest n_train_steps {n_train_steps}"
        )
    else:
        out["loss"] = _interval_weighted_loss(rows)
    return out
