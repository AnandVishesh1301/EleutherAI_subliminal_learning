"""Delete the redundant last `checkpoint-N` of FINISHED runs (dry-run by default).

Every LoRA run writes its final adapter to `{run_dir}/checkpoints/` with
`trainer.save_model` right after `trainer.train()` returns. A trajectory run whose last
scheduled save landed on the run's final step therefore stored those same weights TWICE:
once in `checkpoints/adapter_model.safetensors` and once in
`checkpoints/checkpoint-{n_train_steps}/` -- ~1.18 GB per Qwen run over 5000+ runs on the
shared stash. New runs no longer write it (`students.early_stop.SkipFinalStepSave`); this
script reclaims the ones already on disk.

Nothing is deleted without proof: the candidate's adapter must be byte-identical (or
tensor-identical) to the run's endpoint adapter, the run must be finished (manifest) and
not in flight (no fresh `.claim`, no `interrupted.json`), and the candidate must not be a
queue-yield checkpoint (`resume_marker.json` -- the pipeline's own
`run_gate.prune_resume_state` owns those). `--apply` then deletes ONLY verified
candidates, and only after `--check-readers` confirms this checkout compares checkpoint
ladders on DISTINCT steps (`evaluation.ckpt_trajectories.STEP_GATE`); against an older
checkout the deletion would mark every stored trajectory record stale.

    uv run python -m scripts.dedupe_final_trajectory_checkpoint            # dry run
    uv run python -m scripts.dedupe_final_trajectory_checkpoint --only-tree teachers/panda \
        --limit 50 --apply
"""

from __future__ import annotations

import filecmp
import hashlib
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import typer

from subliminal_learning.config import resolve_output_root
from subliminal_learning.evaluation.ckpt_trajectories import STEP_GATE

ROOT = Path(__file__).absolute().parents[1]
ADAPTER = "adapter_model.safetensors"
MANIFEST = "train_manifest.json"
RESUME_MARKER = "resume_marker.json"
INTERRUPTED = "interrupted.json"
# Dirs that can never CONTAIN a run dir: pruned from the walk so the CephFS scan stays
# bounded (a 40-checkpoint ladder is 40 pointless stat storms per run otherwise).
SKIP_DIRS = ("eval", "wandb", "logs")
# A claim younger than this is a live writer; the run is skipped untouched.
CLAIM_FRESH_SECONDS = 24 * 3600

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


# --------------------------------------------------------------------------- discovery
def is_run_dir(path: Path) -> bool:
    """A finished training run: a manifest beside a `checkpoints/` endpoint adapter."""
    return (path / MANIFEST).is_file() and (path / "checkpoints" / ADAPTER).is_file()


def find_run_dirs(root: Path, *, max_depth: int = 10) -> list[Path]:
    """Run dirs under `root`, found with an iterative bounded `os.scandir` walk.

    A run dir is never nested inside another, so the walk stops descending at one; the
    `checkpoint-*`/`eval`/`wandb`/`logs` subtrees are never entered at all.
    """
    found: list[Path] = []
    stack = [(root, 0)]
    while stack:
        current, depth = stack.pop()
        if is_run_dir(current):
            found.append(current)
            continue
        if depth >= max_depth:
            continue
        try:
            entries = list(os.scandir(current))
        except (FileNotFoundError, NotADirectoryError, PermissionError):
            continue
        for entry in entries:
            if not entry.is_dir(follow_symlinks=False):
                continue
            if entry.name in SKIP_DIRS or entry.name.startswith("checkpoint-"):
                continue
            stack.append((Path(entry.path), depth + 1))
    return sorted(found)


def checkpoint_steps(checkpoints: Path) -> list[int]:
    """The N of every `checkpoint-N` dir under `checkpoints/`, ascending."""
    steps = []
    for d in checkpoints.glob("checkpoint-*"):
        suffix = d.name.split("-")[-1]
        if d.is_dir() and suffix.isdigit():
            steps.append(int(suffix))
    return sorted(steps)


def final_step(run_dir: Path, steps: list[int]) -> tuple[int | None, str]:
    """`(step, source)`: the run's last optimizer step and where that number came from.

    Precedence, most to least authoritative: the manifest's `n_train_steps`; the endpoint
    `checkpoints/trainer_state.json`; the highest `checkpoint-N` whose own
    `trainer_state.json` agrees that its `global_step` is N. `(None, "unknown")` when
    none of them answers -- such a run is never a candidate.
    """
    manifest = _read_json(run_dir / MANIFEST)
    step = (manifest or {}).get("n_train_steps")
    if isinstance(step, int) and not isinstance(step, bool):
        return step, "manifest"
    state = _read_json(run_dir / "checkpoints" / "trainer_state.json")
    step = (state or {}).get("global_step")
    if isinstance(step, int) and not isinstance(step, bool):
        return step, "trainer_state"
    if steps:
        top = max(steps)
        state = _read_json(run_dir / "checkpoints" / f"checkpoint-{top}" / "trainer_state.json")
        if (state or {}).get("global_step") == top:
            return top, "checkpoint_trainer_state"
    return None, "unknown"


def _read_json(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


# --------------------------------------------------------------------------- verification
def in_flight_reason(run_dir: Path, candidate: Path) -> str | None:
    """Why this run must not be touched now, or None when it is safely finished."""
    if (run_dir / INTERRUPTED).is_file():
        return "interrupted.json"
    if (candidate / RESUME_MARKER).is_file():
        # A queue yield wrote this dir; `run_gate.prune_resume_state` disposes of it when
        # the resumed run finishes. Never this script's business.
        return "resume_marker.json"
    for claim in run_dir.glob(".claim*"):
        try:
            age = time.time() - claim.stat().st_mtime
        except OSError:
            continue
        if age < CLAIM_FRESH_SECONDS:
            return f"fresh claim ({age / 3600:.1f} h)"
    return None


def compare_adapters(candidate: Path, endpoint: Path) -> tuple[str, str | None]:
    """`(verdict, detail)` for one pair: identical_bytes / identical_tensors /
    DIFFERENT / unreadable."""
    try:
        if filecmp.cmp(str(candidate), str(endpoint), shallow=False):
            return "identical_bytes", None
    except OSError as exc:
        return "unreadable", str(exc)
    # Safetensors headers carry metadata (and the writer's key order), so two saves of the
    # SAME weights can differ byte-wise; the tensors decide.
    try:
        import torch
        from safetensors import safe_open

        with safe_open(str(candidate), framework="pt") as a, \
                safe_open(str(endpoint), framework="pt") as b:
            keys_a, keys_b = sorted(a.keys()), sorted(b.keys())
            if keys_a != keys_b:
                return "DIFFERENT", "key sets differ"
            for key in keys_a:
                if not torch.equal(a.get_tensor(key), b.get_tensor(key)):
                    return "DIFFERENT", f"tensor {key} differs"
    except Exception as exc:  # noqa: BLE001 -- a torn/foreign file must not kill the scan
        return "unreadable", f"{type(exc).__name__}: {exc}"
    return "identical_tensors", None


def dir_bytes(path: Path) -> int:
    """Total size of every file under `path`."""
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def inspect_run(run_dir: Path, output_root: Path) -> dict:
    """Classify one run dir: status, candidate, final step + its source, verification."""
    rel = str(run_dir.relative_to(output_root)) if run_dir.is_relative_to(output_root) \
        else str(run_dir)
    row: dict = {"run_dir": str(run_dir), "rel": rel, "tree": _tree_of(rel)}
    steps = checkpoint_steps(run_dir / "checkpoints")
    step, source = final_step(run_dir, steps)
    row.update(final_step=step, final_step_source=source, n_checkpoints=len(steps))
    if not steps:
        return {**row, "status": "no_checkpoints"}
    if step is None:
        return {**row, "status": "final_step_unknown"}
    if max(steps) != step:
        return {**row, "status": "last_ckpt_below_final"}
    candidate = run_dir / "checkpoints" / f"checkpoint-{step}"
    row["candidate"] = str(candidate)
    reason = in_flight_reason(run_dir, candidate)
    if reason is not None:
        return {**row, "status": "in_flight", "detail": reason}
    if not (candidate / ADAPTER).is_file():
        return {**row, "status": "unreadable", "detail": f"no {ADAPTER} in the candidate"}
    verdict, detail = compare_adapters(candidate / ADAPTER, run_dir / "checkpoints" / ADAPTER)
    return {**row, "status": verdict, "detail": detail, "bytes": dir_bytes(candidate)}


def _tree_of(rel: str) -> str:
    """`teachers/<animal>/<base>`: the run's three enclosing tree levels.

    Taken from the END of the path so a run dir outside the configured output root (a
    `--root` pointing at a staging copy) still groups sensibly instead of reporting the
    filesystem anchor.
    """
    parts = Path(rel).parts
    return "/".join(parts[:3] if not Path(rel).is_absolute() else parts[-4:-1])


# --------------------------------------------------------------------------- apply
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def delete_candidate(row: dict, log_path: Path) -> None:
    """Delete one verified candidate and append its JSONL deletion record."""
    run_dir = Path(row["run_dir"])
    record = {
        "path": row["candidate"], "bytes": row.get("bytes", 0),
        "status": row["status"], "final_step": row["final_step"],
        "endpoint_sha256": sha256_file(run_dir / "checkpoints" / ADAPTER),
        "deleted_at": datetime.now(timezone.utc).isoformat(),
    }
    shutil.rmtree(row["candidate"])
    with log_path.open("a") as fh:
        fh.write(json.dumps(record) + "\n")


def summarize(rows: list[dict]) -> str:
    """The stdout summary: totals, then reclaimable bytes per top-level tree."""
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    deletable = [r for r in rows if r["status"].startswith("identical")]
    per_tree: dict[str, list[int]] = {}
    for row in deletable:
        per_tree.setdefault(row["tree"], []).append(row.get("bytes", 0))
    lines = [
        f"runs scanned            {len(rows)}",
        f"candidates              {sum(counts.get(k, 0) for k in ('identical_bytes', 'identical_tensors', 'DIFFERENT', 'unreadable'))}",
        f"identical (deletable)   {len(deletable)}",
        f"DIFFERENT (kept)        {counts.get('DIFFERENT', 0)}",
        f"unreadable (kept)       {counts.get('unreadable', 0)}",
        f"in-flight skipped       {counts.get('in_flight', 0)}",
        f"last ckpt below final   {counts.get('last_ckpt_below_final', 0)}",
        f"no checkpoints          {counts.get('no_checkpoints', 0)}",
        f"final step unknown      {counts.get('final_step_unknown', 0)}",
        f"bytes reclaimable       {sum(r.get('bytes', 0) for r in deletable) / 2**30:.2f} GiB",
        "",
        "per tree (deletable runs, GiB):",
    ]
    for tree, sizes in sorted(per_tree.items()):
        lines.append(f"  {tree:<48} {len(sizes):>6}  {sum(sizes) / 2**30:>8.2f}")
    return "\n".join(lines)


def reader_gate_problems() -> list[str]:
    """Reader gates that must compare DISTINCT checkpoint steps before anything is deleted.

    Each import is the gate itself: a checkout predating the change has no `STEP_GATE`,
    no `distinct_steps` use in the dose/fidelity/legacy gates, and deleting the duplicate
    under it would mark every stored trajectory record stale.
    """
    problems = []
    if STEP_GATE != "distinct":
        problems.append(f"evaluation.ckpt_trajectories.STEP_GATE={STEP_GATE!r}")
    gates = {
        "scripts/ckpt_dose_trajectory.py": "distinct_steps(stored_steps)",
        "scripts/fidelity_ckpt_trajectory.py": "want = distinct_steps(",
        "scripts/convert_legacy_pilekl_to_rundir.py": "distinct_steps(legacy_steps)",
    }
    for rel, needle in gates.items():
        try:
            text = (ROOT / rel).read_text()
        except OSError as exc:
            problems.append(f"{rel}: {exc}")
            continue
        if needle not in text:
            problems.append(f"{rel}: no distinct-step comparison ({needle!r} absent)")
    return problems


# --------------------------------------------------------------------------- CLI
@app.command()
def main(
    root: list[Path] = typer.Option(
        None, "--root", help="Tree to scan (repeatable). Default: teachers/ + students/ "
                             "under the paths YAML output_root."),
    report: Path = typer.Option(
        None, help="JSON report path. Default run_logs/dedupe_final_ckpt/<UTC>.json"),
    only_tree: str = typer.Option(
        None, "--only-tree", help="Restrict to one tree prefix, e.g. teachers/panda."),
    limit: int = typer.Option(0, help="Inspect at most N run dirs (0 = no limit)."),
    workers: int = typer.Option(8, help="Threads for the walk and the compares."),
    glob_depth: int = typer.Option(10, help="Max directory depth below each root."),
    check_readers: bool = typer.Option(
        True, help="Refuse --apply unless this checkout's trajectory gates compare "
                   "DISTINCT checkpoint steps."),
    apply: bool = typer.Option(False, "--apply", help="Actually delete verified candidates."),
) -> None:
    output_root = resolve_output_root()
    roots = [Path(r) for r in (root or [])] or [
        output_root / "teachers", output_root / "students"
    ]
    if apply and check_readers:
        problems = reader_gate_problems()
        if problems:
            raise typer.BadParameter(
                "this checkout's trajectory readers do not compare distinct checkpoint "
                "steps, so deleting the duplicate would stale every stored record:\n  "
                + "\n  ".join(problems))

    run_dirs: list[Path] = []
    for tree in roots:
        run_dirs.extend(find_run_dirs(tree, max_depth=glob_depth))
    if only_tree:
        run_dirs = [d for d in run_dirs
                    if d.is_relative_to(output_root)
                    and str(d.relative_to(output_root)).startswith(only_tree)]
    if limit:
        run_dirs = run_dirs[:limit]

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        rows = list(pool.map(lambda d: inspect_run(d, output_root), run_dirs))

    report = report or (ROOT / "run_logs" / "dedupe_final_ckpt" /
                        f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json")
    report.parent.mkdir(parents=True, exist_ok=True)
    deleted_log = report.with_suffix(".deleted.jsonl")

    n_deleted, bytes_deleted = 0, 0
    if apply:
        for row in rows:
            if row["status"].startswith("identical"):
                delete_candidate(row, deleted_log)
                row["deleted"] = True
                n_deleted += 1
                bytes_deleted += row.get("bytes", 0)

    report.write_text(json.dumps({
        "kind": "dedupe_final_trajectory_checkpoint",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "roots": [str(r) for r in roots], "only_tree": only_tree, "limit": limit,
        "applied": apply, "deleted": n_deleted, "deleted_bytes": bytes_deleted,
        "runs": rows,
    }, indent=2) + "\n")

    print(summarize(rows))
    print(f"\nreport: {report}")
    if apply:
        print(f"deleted {n_deleted} dir(s), {bytes_deleted / 2**30:.2f} GiB "
              f"(log: {deleted_log})")
    else:
        print("DRY RUN: nothing deleted (pass --apply).")


if __name__ == "__main__":
    app()
