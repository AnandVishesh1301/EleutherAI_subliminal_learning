"""CLI: periodic health check of shared-queue Slurm campaigns (`scripts.ops.slurm_pack --claim-dir`).

One tick per campaign in the config: progress from the claim markers (done / failed / live / stale claims,
jobs remaining), the Slurm workers for that campaign (running / pending by job-name prefix), throughput over
the last hour and an ETA, and the problems that need a human or an action:

* NEW FAILURES since the last tick, with the last error line of each job's log;
* STALE CLAIMS (heartbeat older than ``stale_s``: a worker died or was preempted mid-job);
* a DEAD CHAIN: jobs remain but no worker is running or pending -> the configured worker sbatch files are
  resubmitted (up to ``expected_workers``), so a campaign never sits idle unnoticed again;
* a STALL: live claims but no new done marker for ``stall_min`` minutes;
* a HUNG job: a live claim whose heartbeat is fresh but whose job log has not grown for ``hung_min`` minutes
  (the packer heartbeats regardless of the child; a child blocked in the kernel -- e.g. on a stale NFS
  lock after a hard kill, 2026-11-11 -- never writes a line and never fails).

Config (JSON): ``{"campaigns": [{"name", "jobs_file", "claim_dir", "log_dir", "job_prefix",
"workers": ["path.sbatch", ...], "expected_workers": 20, "hold_marker": "hold"}], "stale_s": 1800, "stall_min": 150,
"hung_min": 45}``.
Each tick appends to ``--log`` (one line per campaign, WARN lines for problems) and rewrites ``--state``
(the failures already reported, per-campaign done counts with timestamps).

    uv run python -m scripts.ops.queue_health --config run_logs/queues/health/campaigns.json --loop 600
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import typer

SLURM_BIN = "/mnt/ssd-cluster/sky-slurm-pilot/slurm/bin"


def _env() -> dict:
    return {**os.environ, "PATH": f"{SLURM_BIN}:{os.environ.get('PATH', '')}"}


def squeue_counts(job_prefix: str) -> dict[str, int] | None:
    """Running / pending workers whose job name starts with `job_prefix`; None when squeue fails."""
    r = subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%j %T"],
                       capture_output=True, text=True, env=_env())
    if r.returncode != 0:
        return None
    out = {"RUNNING": 0, "PENDING": 0}
    for ln in r.stdout.splitlines():
        parts = ln.split()
        if len(parts) == 2 and parts[0].startswith(job_prefix):
            out[parts[1]] = out.get(parts[1], 0) + 1
    return out


def marker_sets(claim_dir: Path, hold_marker: str | None) -> dict:
    """done / failed (hold markers excluded) / held / live / stale claim names under `claim_dir`."""
    def names(sub):
        d = claim_dir / sub
        return sorted(p.name for p in d.iterdir()) if d.exists() else []
    failed, held = [], []
    for n in names("failed"):
        try:
            rec = json.loads((claim_dir / "failed" / n).read_text())
        except (OSError, ValueError):
            rec = {}
        (held if hold_marker and rec.get(hold_marker) else failed).append(n)
    return {"done": names("done"), "failed": failed, "held": held, "claims": names("claims")}


def claim_ages(claim_dir: Path) -> dict[str, float]:
    d = claim_dir / "claims"
    now = time.time()
    return {p.name: now - p.stat().st_mtime for p in d.iterdir()} if d.exists() else {}


def last_error_line(log_dir: Path, safe_key: str) -> str:
    p = log_dir / f"{safe_key}.log"
    if not p.exists():
        return "(no log)"
    lines = [ln for ln in p.read_text(errors="ignore").splitlines() if ln.strip()]
    for ln in reversed(lines):
        if any(w in ln for w in ("Error", "error:", "FAILED", "Traceback", "SystemExit", "rc=")):
            return ln.strip()[:200]
    return (lines[-1].strip()[:200] if lines else "(empty log)")


def job_keys(jobs_file: Path) -> list[str]:
    return [json.loads(l)["key"] for l in jobs_file.read_text().splitlines() if l.strip()]


def safe(key: str) -> str:
    return key.replace("/", "_").replace("|", "_")


def hung_jobs(claim_dir: Path, log_dir: Path, hung_min: float) -> list[str]:
    """Live claims (fresh heartbeat) whose job log is older than `hung_min` minutes: the child is stuck."""
    now = time.time()
    out = []
    for p in (claim_dir / "claims").glob("*") if (claim_dir / "claims").exists() else []:
        if now - p.stat().st_mtime > 2 * 60:          # not heartbeating: that is the stale-claim case
            continue
        lg = log_dir / f"{p.name}.log"
        if lg.exists() and now - lg.stat().st_mtime > hung_min * 60:
            out.append(p.name)
    return sorted(out)


def tick(cfg: dict, state: dict, stale_s: float, stall_min: float, log, hung_min: float = 45.0) -> None:
    name = cfg["name"]
    claim_dir, log_dir = Path(cfg["claim_dir"]), Path(cfg["log_dir"])
    keys = job_keys(Path(cfg["jobs_file"]))
    m = marker_sets(claim_dir, cfg.get("hold_marker"))
    listed = {safe(k) for k in keys}          # markers of jobs no longer in the file (e.g. a rebuilt file) are ignored
    done, failed, held, claims = (set(m["done"]) & listed, set(m["failed"]) & listed, set(m["held"]) & listed,
                                  set(m["claims"]) & listed)
    remaining = [k for k in keys if safe(k) not in done | failed | held]
    ages = claim_ages(claim_dir)
    stale = sorted(k for k, a in ages.items() if a > stale_s)
    q = squeue_counts(cfg["job_prefix"])
    st = state.setdefault(name, {"reported_failures": [], "history": []})
    now = time.time()
    st["history"] = [h for h in st["history"] if now - h[0] < 3600 * 3] + [(now, len(done))]
    hour = [h for h in st["history"] if now - h[0] <= 3600]
    rate = (len(done) - hour[0][1]) / max(1e-9, (now - hour[0][0]) / 3600) if len(hour) > 1 else None
    eta = f"{(len(remaining) - len(claims)) / rate:.1f} h" if rate and rate > 0 else "-"
    qtxt = f"workers R{q['RUNNING']}/P{q['PENDING']}" if q else "squeue FAILED"
    log(f"{name}: done {len(done)}/{len(keys)} live {len(claims)} remaining {len(remaining)} failed {len(failed)}"
        f"{' held ' + str(len(held)) if held else ''} | {qtxt} | {rate:.0f}/h eta {eta}" if rate is not None else
        f"{name}: done {len(done)}/{len(keys)} live {len(claims)} remaining {len(remaining)} failed {len(failed)}"
        f"{' held ' + str(len(held)) if held else ''} | {qtxt}")
    # new failures
    new = sorted(failed - set(st["reported_failures"]))
    for n in new[:10]:
        log(f"  WARN {name}: FAILED {n}: {last_error_line(log_dir, n)}")
    if len(new) > 10:
        log(f"  WARN {name}: ... {len(new) - 10} more new failures")
    st["reported_failures"] = sorted(failed)
    # stale claims
    if stale:
        log(f"  WARN {name}: {len(stale)} STALE claim(s) (> {int(stale_s)} s without heartbeat), e.g. {stale[:3]} -- "
            "a worker died or was preempted; the next worker reclaims them")
    # stall
    if claims and len(hour) > 1 and hour[0][1] == len(done) and now - hour[0][0] >= stall_min * 60:
        log(f"  WARN {name}: STALL -- {len(claims)} live claim(s) but no job finished in {stall_min:.0f} min")
    hung = hung_jobs(claim_dir, log_dir, hung_min)
    if hung:
        log(f"  WARN {name}: {len(hung)} HUNG job(s) (claim heartbeating, log silent > {hung_min:.0f} min), e.g. {hung[:3]} -- "
            "the child is stuck (NFS lock / D state): kill it on its node, remove the leaf's .training.lock and the failed marker")
    # dead chain -> resubmit
    if remaining and q is not None and q["RUNNING"] + q["PENDING"] == 0 and not Path(cfg.get("stop_file", "/nonexistent")).exists():
        want = int(cfg.get("expected_workers", 0))
        subs = 0
        for sb in cfg.get("workers", []):
            n_this = int(cfg.get("per_worker_count", {}).get(sb, max(1, want // max(1, len(cfg["workers"])))))
            for _ in range(n_this):
                r = subprocess.run(["sbatch", sb], capture_output=True, text=True, env=_env(), cwd=cfg.get("cwd"))
                subs += r.returncode == 0
                if r.returncode != 0:
                    log(f"  WARN {name}: sbatch {sb} failed: {r.stderr.strip()[:120]}")
                    break
        log(f"  ACTION {name}: no worker running or pending with {len(remaining)} jobs left -> resubmitted {subs} worker(s)")


def main(
    config: Path = typer.Option(..., help="JSON with the campaigns to watch (see module docstring)."),
    log_path: Path = typer.Option(Path("run_logs/queues/health/health.log"), "--log"),
    state_path: Path = typer.Option(Path("run_logs/queues/health/state.json"), "--state"),
    loop: int = typer.Option(0, help="Seconds between ticks; 0 = one tick and exit."),
) -> None:
    cfg = json.loads(config.read_text())
    stale_s, stall_min, hung_min = float(cfg.get("stale_s", 1800)), float(cfg.get("stall_min", 150)), float(cfg.get("hung_min", 45))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    state = json.loads(state_path.read_text()) if state_path.exists() else {}

    def log(msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}"
        print(line, flush=True)
        with log_path.open("a") as f:
            f.write(line + "\n")

    while True:
        # re-read the config each tick so campaigns can be added/removed while the loop runs
        cfg = json.loads(config.read_text())
        for c in cfg["campaigns"]:
            if not Path(c["jobs_file"]).exists():
                log(f"{c['name']}: jobs file missing ({c['jobs_file']})")
                continue
            try:
                tick(c, state, stale_s, stall_min, log, hung_min)
            except Exception as e:  # noqa: BLE001 -- a broken campaign entry must not kill the monitor
                log(f"  WARN {c['name']}: tick failed: {type(e).__name__}: {e}")
        state_path.write_text(json.dumps(state, indent=1))
        if not loop:
            break
        time.sleep(loop)


if __name__ == "__main__":
    typer.run(main)
