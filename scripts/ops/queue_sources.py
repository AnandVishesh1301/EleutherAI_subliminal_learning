"""CLI: the campaigns with work left, as packer arguments -- the glue of the SHARED worker.

Reads the health config (`scripts.ops.queue_health`, ``campaigns`` in precedence order) and keeps every
campaign that has no stop file and at least one job without a done / failed / hold marker. Output:

* default: one ``--jobs-file <f> --claim-dir <c> --log-dir <l>`` triple per campaign, config order
  (= precedence when priorities tie), ready to splice into ``scripts.ops.slurm_pack``;
* ``--count``: the total number of such jobs (the worker chain resubmits while it is > 0).

A worker started from these arguments takes any campaign's job that fits its GPU, so a finishing
campaign never leaves a GPU two-thirds idle while another campaign is pending (Angela 2026-11-11:
"make it shared").

    uv run python -m scripts.ops.queue_sources --config run_logs/queues/health/campaigns.json
"""

from __future__ import annotations

import json
from pathlib import Path

import typer


def safe(key: str) -> str:
    return key.replace("/", "_").replace("|", "_")


def remaining(cfg: dict) -> int:
    """Jobs of one campaign without a done / failed (incl. hold) marker; 0 when stopped or without a file."""
    jobs_file, claim_dir = Path(cfg["jobs_file"]), Path(cfg["claim_dir"])
    if Path(cfg.get("stop_file", "/nonexistent")).exists() or not jobs_file.exists():
        return 0
    keys = [json.loads(l)["key"] for l in jobs_file.read_text().splitlines() if l.strip()]
    terminal = {p.name for sub in ("done", "failed") for p in (claim_dir / sub).glob("*")} if claim_dir.exists() else set()
    return sum(1 for k in keys if safe(k) not in terminal)


def active(config: Path) -> list[dict]:
    cfg = json.loads(config.read_text())
    return [c for c in cfg["campaigns"] if remaining(c) > 0]


def main(config: Path = typer.Option(..., help="The health config (campaigns in precedence order)."),
         count: bool = typer.Option(False, help="Print the total number of remaining jobs instead.")) -> None:
    camps = active(config)
    if count:
        print(sum(remaining(c) for c in camps))
        return
    print(" ".join(f"--jobs-file {c['jobs_file']} --claim-dir {c['claim_dir']} --log-dir {c['log_dir']}" for c in camps))


if __name__ == "__main__":
    typer.run(main)
