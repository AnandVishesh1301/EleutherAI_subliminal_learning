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
ladders under the COVERAGE rule (`training.ckpt_ladder.STEP_GATE`); against an older
checkout the deletion would mark every stored trajectory record stale.

The tree is live (three nodes training and evaluating), so every fact the classification
pass recorded is re-proven immediately before each `rmtree` (`revalidation_problem`), the
deletion log is written BEFORE the delete, the JSON report is flushed periodically and in
a `finally`, and `--max-delete` bounds a staged run.

Writers are excluded by the run dir's `.claim`, which this script HOLDS across the
re-validation and the `rmtree` (`utils.run_io.acquire_claim`): the LoRA pipelines never
take it themselves, but the worker gates treat a fresh run-dir write as in flight, so a
training attempt arriving inside that window defers (rc=75) instead of writing. A run
someone else holds is skipped (`skipped_claimed`).

    uv run python -m scripts.dedupe_final_trajectory_checkpoint            # dry run
    uv run python -m scripts.dedupe_final_trajectory_checkpoint --only-tree teachers/panda \
        --limit 50 --apply
    uv run python -m scripts.dedupe_final_trajectory_checkpoint --apply --max-delete 500
    uv run python -m scripts.dedupe_final_trajectory_checkpoint --apply --stream

`--stream` deletes each proven duplicate inside the walk (in the worker thread that
classified it, under the same claim and the same re-validation) instead of after it, so a
multi-hour scan of the live tree reclaims space as it goes.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import typer

from subliminal_learning.config import resolve_output_root
from subliminal_learning.training.ckpt_ladder import STEP_GATE
from subliminal_learning.utils.io import atomic_write_json
from subliminal_learning.utils.run_io import (
    CLAIM_TTL_SECONDS,
    acquire_claim,
    endpoint_link_problem,
    find_leaf_dirs_parallel,
    release_claim,
)

ROOT = Path(__file__).absolute().parents[1]
ADAPTER = "adapter_model.safetensors"
MANIFEST = "train_manifest.json"
RESUME_MARKER = "resume_marker.json"
INTERRUPTED = "interrupted.json"
# Dirs that can never CONTAIN a run dir: pruned from the walk so the CephFS scan stays
# bounded (a 40-checkpoint ladder is 40 pointless stat storms per run otherwise).
# GKD / SDF runs keep `checkpoint-*` as RESUME state and read the last one, so their
# trees are never walked -- whatever `--root` is given.
EXCLUDED_DIRS = ("gkd_students", "sdf_teachers")
SKIP_DIRS = ("eval", "wandb", "logs", *EXCLUDED_DIRS)
# A claim younger than this is a live writer; the run is skipped untouched (the claim TTL
# the run gate itself uses -- a stale claim is one the gate would overwrite).
CLAIM_FRESH_SECONDS = CLAIM_TTL_SECONDS
# A run whose manifest/endpoint adapter was written this recently may still have evals
# reading its ladder; treated as in flight.
RECENT_WRITE_SECONDS = 10 * 60
# Bytes per read in the adapter byte compare (own compare: no module-level cache to grow
# across 36k pairs the way `filecmp.cmp(shallow=False)` does).
COMPARE_CHUNK = 1 << 20
# Stream-mode progress line cadence (runs inspected).
PROGRESS_EVERY = 1000
# The four verdicts a run that HAS a final-step candidate can get.
CANDIDATE_STATUSES = ("identical_bytes", "identical_tensors", "DIFFERENT", "unreadable")
# Files a half-removed checkpoint dir may still hold: `rmtree` deletes the big tensors
# first, so only the trailing metadata survives a kill. Anything else means the dir is not
# a torn leftover.
TORN_LEFTOVER_FILES = ("trainer_state.json", "adapter_config.json", "README.md")
# Partial reports are flushed every this many deletions so a kill mid-apply still leaves
# a readable (incomplete) report.
FLUSH_EVERY = 500

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


# --------------------------------------------------------------------------- discovery
def is_run_dir(path: Path) -> bool:
    """A finished training run: a manifest beside a `checkpoints/` endpoint adapter."""
    return (path / MANIFEST).is_file() and (path / "checkpoints" / ADAPTER).is_file()


def find_run_dirs(root: Path, *, max_depth: int = 10, workers: int = 8,
                  progress: bool = False) -> list[Path]:
    """Run dirs under `root`, found with a thread-parallel bounded `os.scandir` walk.

    A run dir is never nested inside another, so the walk stops descending at one; the
    `checkpoint-*`/`eval`/`wandb`/`logs` subtrees are never entered at all. The walk is
    threaded because its cost is per-scandir network latency (200k dirs on CephFS).
    """
    return find_leaf_dirs_parallel(
        [root], is_run_dir, skip_names=SKIP_DIRS, skip_prefixes=("checkpoint-",),
        max_depth=max_depth, workers=workers, progress=progress)


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
def in_flight_reason(run_dir: Path, candidate: Path,
                     ignore_claim: Path | None = None) -> str | None:
    """Why this run must not be touched now, or None when it is safely finished.

    `ignore_claim` is this process's own claim on the run, which must not read as a live
    writer while the delete holds it.
    """
    if (run_dir / INTERRUPTED).is_file():
        return "interrupted.json"
    if (candidate / RESUME_MARKER).is_file():
        # A queue yield wrote this dir; `run_gate.prune_resume_state` disposes of it when
        # the resumed run finishes. Never this script's business.
        return "resume_marker.json"
    for claim in run_dir.glob(".claim*"):
        if ignore_claim is not None and claim == Path(ignore_claim):
            continue
        try:
            age = time.time() - claim.stat().st_mtime
        except OSError:
            continue
        if age < CLAIM_FRESH_SECONDS:
            return f"fresh claim ({age / 3600:.1f} h)"
    for path in (run_dir / MANIFEST, run_dir / "checkpoints" / ADAPTER):
        try:
            age = time.time() - path.stat().st_mtime
        except OSError:
            continue
        if age < RECENT_WRITE_SECONDS:
            return "recent write"
    return None


def same_bytes(a: Path, b: Path) -> bool:
    """Byte-identical file compare (size first, then chunked reads).

    Equivalent to `filecmp.cmp(shallow=False)` but without its module-level cache, which
    would grow one entry per pair across tens of thousands of runs.
    """
    if a.stat().st_size != b.stat().st_size:
        return False
    with a.open("rb") as fa, b.open("rb") as fb:
        while True:
            chunk_a, chunk_b = fa.read(COMPARE_CHUNK), fb.read(COMPARE_CHUNK)
            if chunk_a != chunk_b:
                return False
            if not chunk_a:
                return True


def compare_adapters(candidate: Path, endpoint: Path) -> tuple[str, str | None]:
    """`(verdict, detail)` for one pair: identical_bytes / identical_tensors /
    DIFFERENT / unreadable."""
    try:
        if same_bytes(candidate, endpoint):
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
                ta, tb = a.get_tensor(key), b.get_tensor(key)
                # `torch.equal` promotes across dtypes in this torch, so a dtype or shape
                # mismatch is decided here: DIFFERENT (kept), not unreadable.
                if ta.dtype != tb.dtype:
                    return "DIFFERENT", f"tensor {key} dtype {ta.dtype} vs {tb.dtype}"
                if ta.shape != tb.shape:
                    return "DIFFERENT", f"tensor {key} shape {tuple(ta.shape)} vs {tuple(tb.shape)}"
                try:
                    equal = torch.equal(ta, tb)
                except (RuntimeError, TypeError) as exc:
                    return "DIFFERENT", f"tensor {key} incomparable: {exc}"
                if not equal:
                    return "DIFFERENT", f"tensor {key} differs"
    except Exception as exc:  # noqa: BLE001 -- a torn/foreign file must not kill the scan
        return "unreadable", f"{type(exc).__name__}: {exc}"
    return "identical_tensors", None


def dir_bytes(path: Path) -> int:
    """Total size of every file under `path`, ignoring files that vanish mid-walk."""
    total = 0
    try:
        entries = list(path.rglob("*"))
    except OSError:
        return total
    for p in entries:
        try:
            if p.is_file():
                total += p.stat().st_size
        except (FileNotFoundError, PermissionError, OSError):
            continue
    return total


def _stat_key(path: Path) -> dict | None:
    """`{size, mtime}` -- the identity a candidate must still have at deletion time."""
    try:
        st = path.stat()
    except OSError:
        return None
    return {"size": st.st_size, "mtime": st.st_mtime}


def inspect_run(run_dir: Path, output_root: Path) -> dict:
    """Classify one run dir; any unexpected error becomes a `scan_error` row.

    The multi-hour scan runs against a live tree on three nodes: one unreadable or
    concurrently pruned run must never abort the pool.
    """
    try:
        return _inspect_run(run_dir, output_root)
    except Exception as exc:  # noqa: BLE001 -- one bad run must not kill the scan
        return {"run_dir": str(run_dir), "rel": str(run_dir), "tree": "",
                "final_step": None, "final_step_source": "unknown", "n_checkpoints": 0,
                "status": "scan_error", "detail": f"{type(exc).__name__}: {exc}"}


def _inspect_run(run_dir: Path, output_root: Path) -> dict:
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
    if max(steps) < step:
        return {**row, "status": "last_ckpt_below_final"}
    if max(steps) > step:
        # Dirs past the recorded final step (early-stopped / diverged / torn run): the
        # final step is not the top of the ladder, so nothing here is a proven duplicate.
        return {**row, "status": "ckpt_above_final"}
    candidate = run_dir / "checkpoints" / f"checkpoint-{step}"
    row["candidate"] = str(candidate)
    reason = in_flight_reason(run_dir, candidate)
    if reason is not None:
        return {**row, "status": "in_flight", "detail": reason}
    if source != "manifest":
        # `ckpt_trajectories._checkpoint_dirs` labels the endpoint from the manifest's
        # `n_train_steps`; without it, it falls back to the highest REMAINING
        # `checkpoint-N`, so deleting this dir would shift the ladder's x-axis.
        return {**row, "status": "skipped_no_manifest_step",
                "detail": f"final step from {source}, not the manifest"}
    endpoint = run_dir / "checkpoints" / ADAPTER
    if not (candidate / ADAPTER).is_file():
        return {**row, **torn_final_dir_status(candidate, endpoint)}
    verdict, detail = compare_adapters(candidate / ADAPTER, endpoint)
    return {**row, "status": verdict, "detail": detail, "bytes": dir_bytes(candidate),
            "candidate_stat": _stat_key(candidate / ADAPTER),
            "endpoint_stat": _stat_key(endpoint)}


def torn_final_dir_status(candidate: Path, endpoint: Path) -> dict:
    """Classify a final-step dir with no adapter: leftover of a killed `rmtree`, or not.

    A delete that died inside its `rmtree` leaves the candidate half removed -- this
    script's own `--apply`, or `run_gate`'s prune paths, which `rmtree(ignore_errors=True)`
    the same dirs; a leftover of either origin is garbage. On a FRESH invocation there is
    no fingerprint to re-validate against, so the dir is recognised by its own shape
    instead: nothing left but the trailing metadata files, and a complete endpoint adapter
    beside it. Anything else in there (even empty) could be a partially written NEW
    checkpoint (`optimizer.pt`), which is never this script's to remove:
    `torn_final_dir_unrecognised`, kept.
    """
    try:
        if endpoint.stat().st_size <= 0:
            return {"status": "unreadable", "detail": f"endpoint {ADAPTER} is empty"}
    except OSError as exc:
        return {"status": "unreadable", "detail": f"endpoint {ADAPTER}: {exc}"}
    for path in sorted(candidate.rglob("*")):
        try:
            # Size is irrelevant: a zero-byte `optimizer.pt` is still a file this script
            # did not leave behind, so the dir is not a recognised leftover.
            if not path.is_file():
                continue
        except OSError:
            continue
        if path.parent != candidate or path.name not in TORN_LEFTOVER_FILES:
            return {"status": "torn_final_dir_unrecognised",
                    "detail": f"{path.relative_to(candidate)} is not a torn leftover",
                    "bytes": dir_bytes(candidate)}
    return {"status": "torn_final_dir",
            "detail": f"no {ADAPTER}: leftover of an interrupted delete",
            "bytes": dir_bytes(candidate), "candidate_stat": None,
            "endpoint_stat": _stat_key(endpoint)}


def _tree_of(rel: str) -> str:
    """`teachers/<animal>/<base>`: the run's three enclosing tree levels.

    Taken from the END of the path so a run dir outside the configured output root (a
    `--root` pointing at a staging copy) still groups sensibly instead of reporting the
    filesystem anchor.
    """
    parts = Path(rel).parts
    return "/".join(parts[:3] if not Path(rel).is_absolute() else parts[-4:-1])


# --------------------------------------------------------------------------- apply
def sha256_head_tail(path: Path) -> str:
    """sha256 of the file's first and last `COMPARE_CHUNK` bytes (size mixed in).

    Enough to IDENTIFY the endpoint adapter in the deletion log; a full digest would be a
    third end-to-end read of a file the compare already read twice -- thousands of GB of
    serial CephFS traffic across the sweep, for no extra proof (the delete is gated on the
    byte/tensor compare plus the size+mtime re-validation, not on this digest).
    """
    size = path.stat().st_size
    digest = hashlib.sha256(str(size).encode())
    with path.open("rb") as fh:
        digest.update(fh.read(COMPARE_CHUNK))
        if size > COMPARE_CHUNK:
            fh.seek(max(COMPARE_CHUNK, size - COMPARE_CHUNK))
            digest.update(fh.read(COMPARE_CHUNK))
    return digest.hexdigest()


def revalidation_problem(row: dict, own_claim: Path | None = None,
                         allow_missing_candidate: bool = False) -> str | None:
    """Why this row may NOT be deleted now, re-checked against disk, or None.

    The classification pass is an hour-long walk on a tree three nodes write to, so every
    fact it recorded is re-proven immediately before the `rmtree`: the run is still a
    finished run, still not in flight, and BOTH adapters are still exactly the files that
    were compared (same size, same mtime). `allow_missing_candidate` drops ONLY the
    candidate-adapter check (a torn dir has none), never the manifest, in-flight or
    endpoint ones.
    """
    run_dir, candidate = Path(row["run_dir"]), Path(row["candidate"])
    if not (run_dir / MANIFEST).is_file():
        return f"{MANIFEST} vanished"
    # The final step is re-read from the manifest with the same helper the scan used: a run
    # that resumed and trained on has a NEW `n_train_steps`, and this dir is then a rung of
    # its ladder, not the duplicate endpoint.
    step, source = final_step(run_dir, checkpoint_steps(run_dir / "checkpoints"))
    if source != "manifest" or step != row["final_step"]:
        return f"final step changed ({row['final_step']} -> {step}, source {source})"
    reason = in_flight_reason(run_dir, candidate, ignore_claim=own_claim)
    if reason is not None:
        return f"now in flight: {reason}"
    checks = [("endpoint", run_dir / "checkpoints" / ADAPTER, row.get("endpoint_stat"))]
    if not allow_missing_candidate:
        # A torn candidate's adapter is ALREADY gone; every other fact above and below
        # still has to hold, so the candidate check is skipped, never the rest.
        checks.insert(0, ("candidate", candidate / ADAPTER, row.get("candidate_stat")))
    for label, path, before in checks:
        now = _stat_key(path)
        if now is None:
            return f"{label} {ADAPTER} vanished"
        if before is None:
            return f"{label} {ADAPTER} was not fingerprinted at inspection"
        if now["size"] != before["size"] or now["mtime"] != before["mtime"]:
            return (f"{label} {ADAPTER} changed "
                    f"(size {before['size']}->{now['size']}, "
                    f"mtime {before['mtime']}->{now['mtime']})")
    return None


def unsafe_path_reason(run_dir: Path, candidate: Path) -> str | None:
    """Why `rmtree(candidate)` would be reckless, or None when the path is in bounds."""
    checkpoints = run_dir / "checkpoints"
    if candidate.is_symlink():
        return "candidate is a symlink"
    try:
        resolved, root = candidate.resolve(), checkpoints.resolve()
    except OSError as exc:
        return f"unresolvable path: {exc}"
    if resolved == root:
        return "candidate resolves to the checkpoints root"
    if not resolved.is_relative_to(root):
        return f"candidate resolves outside {root}"
    # The run's FINAL adapter lives at the checkpoints root; refuse a candidate it resolves
    # into (shared guard, also used by the thinning script and `run_gate`).
    return endpoint_link_problem(run_dir, [candidate])


# `--stream` deletes from the walk's worker threads; the deletion log is one file, so the
# append is serialised and whole records never interleave.
_LOG_LOCK = threading.Lock()
# Guards every row dict AND the row list: the workers mutate rows (`status`, `detail`,
# `deleted`) while the periodic report flush serialises them, and a dict mutated mid
# `json.dumps` raises "dictionary changed size during iteration" and unwinds the run.
_ROW_LOCK = threading.Lock()


def _set_row(row: dict, **fields) -> None:
    """Mutate one row under `_ROW_LOCK`, the lock the report snapshot is taken under."""
    with _ROW_LOCK:
        row.update(fields)


def _append_jsonl(log_path: Path, record: dict) -> None:
    with _LOG_LOCK, log_path.open("a") as fh:
        fh.write(json.dumps(record) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def delete_candidate(row: dict, log_path: Path) -> str:
    """Re-validate, then delete one candidate; returns the outcome status.

    The JSONL log is written BEFORE the `rmtree` (`phase: intent`) and again after
    (`phase: done` / `failed`), so a crash mid-delete still names the dir it was in.
    """
    run_dir, candidate = Path(row["run_dir"]), Path(row["candidate"])
    fresh_torn = row["status"] == "torn_final_dir"
    torn = fresh_torn or (candidate.is_dir() and not (candidate / ADAPTER).is_file()
                          and row.get("candidate_stat") is not None)
    problem = unsafe_path_reason(run_dir, candidate)
    if problem is not None:
        _set_row(row, status="skipped_unsafe_path", detail=problem)
        return row["status"]
    if not run_dir.is_dir():
        # `acquire_claim` would CREATE the run dir; a vanished run is never re-made here.
        _set_row(row, status="skipped_changed", detail="run dir vanished")
        return row["status"]
    claim = acquire_claim(run_dir)
    if claim is None:
        # Someone holds this run right now: re-validation alone cannot exclude a writer
        # that starts between the check and the rmtree, so the claim is the lock.
        _set_row(row, status="skipped_claimed",
                 detail="run dir claimed by another process")
        return row["status"]
    try:
        # A previous run died INSIDE this `rmtree`: the adapter is already gone, so the
        # row is a duplicate whose deletion is half done. Reporting `delete_failed` every
        # run forever leaves the torn dir on disk -- finish the job instead, with the run,
        # the in-flight state and the ENDPOINT still re-validated.
        problem = revalidation_problem(row, own_claim=claim, allow_missing_candidate=torn)
        if problem is None and torn:
            # Whatever is left in there must still have a leftover's shape -- on the
            # fingerprinted path too, where the candidate adapter vanished after the scan
            # and a partially written NEW checkpoint could have taken its place.
            now = torn_final_dir_status(candidate, run_dir / "checkpoints" / ADAPTER)
            if now["status"] != "torn_final_dir":
                problem = f"no longer a torn leftover: {now['status']}"
        if problem is not None:
            _set_row(row, status="skipped_changed", detail=problem)
            return row["status"]

        record = {
            "path": row["candidate"], "bytes": row.get("bytes", 0),
            "status": row["status"], "final_step": row["final_step"],
            "endpoint_stat": row.get("endpoint_stat"),
            "endpoint_sha256_head_tail": sha256_head_tail(
                run_dir / "checkpoints" / ADAPTER),
            "deleted_at": datetime.now(timezone.utc).isoformat(),
            **({"torn": True, "verified": not fresh_torn} if torn else {}),
        }
        _append_jsonl(log_path, {**record, "phase": "intent"})
        try:
            shutil.rmtree(row["candidate"])
        except (PermissionError, NotADirectoryError, OSError) as exc:
            detail = f"{type(exc).__name__}: {exc}"
            _append_jsonl(log_path, {**record, "phase": "failed", "error": detail})
            _set_row(row, status="delete_failed", detail=detail)
            return row["status"]
        _append_jsonl(log_path, {**record, "phase": "done"})
        _set_row(row, deleted=True)
        return "deleted"
    finally:
        release_claim(claim)


def is_deletable(row: dict) -> bool:
    """Rows `--apply` removes: proven duplicates plus recognised torn leftovers."""
    return row["status"].startswith("identical") or row["status"] == "torn_final_dir"


def summarize(rows: list[dict]) -> str:
    """The stdout summary: totals, then reclaimable bytes per top-level tree."""
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    # `--apply` removes torn leftovers too, so they count as deletable bytes here.
    deletable = [r for r in rows if is_deletable(r)]
    per_tree: dict[str, list[int]] = {}
    for row in deletable:
        per_tree.setdefault(row["tree"], []).append(row.get("bytes", 0))
    lines = [
        f"runs scanned            {len(rows)}",
        f"candidates              {sum(counts.get(k, 0) for k in ('identical_bytes', 'identical_tensors', 'DIFFERENT', 'unreadable'))}",
        f"identical (deletable)   {sum(1 for r in deletable if r['status'].startswith('identical'))}",
        f"DIFFERENT (kept)        {counts.get('DIFFERENT', 0)}",
        f"unreadable (kept)       {counts.get('unreadable', 0)}",
        f"torn final dirs         {counts.get('torn_final_dir', 0)}",
        f"deletable (total)       {len(deletable)}",
        f"torn unrecognised       {counts.get('torn_final_dir_unrecognised', 0)}",
        f"in-flight skipped       {counts.get('in_flight', 0)}",
        f"last ckpt below final   {counts.get('last_ckpt_below_final', 0)}",
        f"ckpt above final        {counts.get('ckpt_above_final', 0)}",
        f"no manifest step        {counts.get('skipped_no_manifest_step', 0)}",
        f"skipped (changed)       {counts.get('skipped_changed', 0)}",
        f"skipped (unsafe path)   {counts.get('skipped_unsafe_path', 0)}",
        f"delete failed           {counts.get('delete_failed', 0)}",
        f"skipped (claimed)       {counts.get('skipped_claimed', 0)}",
        f"scan errors             {counts.get('scan_error', 0)}",
        f"excluded roots          {counts.get('skipped_excluded_root', 0)}",
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

    Each gate is checked INSIDE the function that does the comparing (located with `ast`),
    so a leftover mention in a docstring or a dead helper cannot pass a regressed gate. A
    checkout predating the change has no `STEP_GATE` of ``"covers"`` and no `steps_covered`
    use in the dose/fidelity/legacy gates; deleting a checkpoint under it would mark every
    stored trajectory record stale (see `training.ckpt_ladder`).
    """
    problems = []
    if STEP_GATE != "covers":
        problems.append(f"training.ckpt_ladder.STEP_GATE={STEP_GATE!r}")
    # {relative path: (function name or accepted alternatives, needle)}. Every gate that
    # can REWRITE a per-checkpoint record is listed, not only the three trajectory scripts:
    # a regression to equality in any of them shrinks a stored record the moment a
    # checkpoint is deleted.
    gates = {
        "forgetting/scripts/ckpt_dose_trajectory.py": ("_run_one", "steps_covered(stored_steps"),
        "forgetting/scripts/fidelity_ckpt_trajectory.py": ("_is_current", "steps_covered("),
        "scripts/convert_legacy_pilekl_to_rundir.py": ("main", "steps_covered(legacy_steps"),
        "src/subliminal_learning/evaluation/ckpt_trajectories.py": [
            ("eval_is_current", "steps_covered("),
            # The Pile-KL gate is inline in the trajectory function itself.
            ("pile_kl_trajectory", "steps_covered("),
        ],
        # The spectra gate is being renamed by a concurrent change; either name counts.
        "forgetting/scripts/adapter_spectra_trajectory.py":
            (("_record_is_current", "spectra_record_problem"), "steps_covered("),
        "forgetting/scripts/adapter_norms_slim.py": ("traj_record_problem", "steps_covered("),
        "src/subliminal_learning/steering/eas.py": ("eas_doc_status", "ladder_steps"),
    }
    for rel, checks in gates.items():
        try:
            text = (ROOT / rel).read_text()
        except OSError as exc:
            problems.append(f"{rel}: {exc}")
            continue
        for names, needle in (checks if isinstance(checks, list) else [checks]):
            names = (names,) if isinstance(names, str) else names
            bodies = {n: _function_source(text, n) for n in names}
            found = {n: b for n, b in bodies.items() if b is not None}
            if not found:
                problems.append(f"{rel}: no function {' / '.join(names)}()")
            elif not any(needle in b for b in found.values()):
                problems.append(
                    f"{rel}: {'/'.join(found)}() has no distinct-step comparison "
                    f"({needle!r} absent)")
    return problems


def _function_source(text: str, name: str) -> str | None:
    """The source of the top-level function `name`, or None when it is absent."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(text, node)
    return None


# --------------------------------------------------------------------------- CLI
@app.command()
def main(
    root: list[Path] = typer.Option(
        None, "--root", help="Tree to scan (repeatable). Default: teachers/ + students/ "
                             "under the paths YAML output_root."),
    report: Path = typer.Option(
        None, help="JSON report path. Default run_logs/campaigns/dedupe_final_ckpt/<UTC>.json"),
    only_tree: str = typer.Option(
        None, "--only-tree", help="Restrict to one tree prefix, e.g. teachers/panda."),
    limit: int = typer.Option(0, help="Inspect at most N run dirs (0 = no limit)."),
    workers: int = typer.Option(8, help="Threads for the compares."),
    walk_workers: int = typer.Option(
        8, "--walk-workers",
        help="Threads for the discovery walk; keep low, the nodes run GPU jobs"),
    glob_depth: int = typer.Option(10, help="Max directory depth below each root."),
    check_readers: bool = typer.Option(
        True, help="Refuse --apply unless this checkout's trajectory gates compare "
                   "DISTINCT checkpoint steps."),
    apply: bool = typer.Option(False, "--apply", help="Actually delete verified candidates."),
    max_delete: int = typer.Option(
        0, "--max-delete", help="Stop after deleting N dirs this invocation (0 = no cap)."),
    stream: bool = typer.Option(
        False, "--stream", help="With --apply: delete each proven duplicate as the walk "
                                "classifies it, instead of after the whole walk."),
) -> None:
    output_root = resolve_output_root()
    if stream and not apply:
        raise typer.BadParameter("--stream only means anything with --apply")
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
    outside = [str(r) for r in roots if not r.is_relative_to(output_root)]

    run_dirs: list[Path] = []
    excluded_rows: list[dict] = []
    for tree in roots:
        if tree.name in EXCLUDED_DIRS:
            excluded_rows.append({
                "run_dir": str(tree), "rel": str(tree), "tree": tree.name,
                "final_step": None, "final_step_source": "unknown", "n_checkpoints": 0,
                "status": "skipped_excluded_root",
                "detail": f"{tree.name} keeps checkpoint-* as resume state"})
            continue
        run_dirs.extend(find_run_dirs(tree, max_depth=glob_depth,
                                      workers=walk_workers, progress=True))
    run_dirs = sorted(set(run_dirs))
    if only_tree:
        kept = [d for d in run_dirs
                if d.is_relative_to(output_root)
                and str(d.relative_to(output_root)).startswith(only_tree)]
        if run_dirs and not kept:
            print(f"--only-tree {only_tree!r} matched none of the {len(run_dirs)} run "
                  f"dir(s) found under {[str(r) for r in roots]}", file=sys.stderr)
            raise typer.Exit(code=2)
        run_dirs = kept
    if limit:
        run_dirs = run_dirs[:limit]

    report = report or (ROOT / "run_logs" / "campaigns" / "dedupe_final_ckpt" /
                        f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json")
    report.parent.mkdir(parents=True, exist_ok=True)
    deleted_log = report.parent / (report.stem + ".deleted.jsonl")

    rows: list[dict] = list(excluded_rows)
    n_deleted, bytes_deleted, n_inspected, n_candidates = 0, 0, 0, 0
    capped = finished = False
    # Guards everything the streaming workers share: the row list and the rows themselves
    # (`_ROW_LOCK`, which every row mutation takes), the counters (so --max-delete is one
    # atomic reservation, not a read-then-delete race) and the periodic report flush.
    lock = _ROW_LOCK
    # A kill unwinding through one stream worker must stop the others from deleting: the
    # pool has every run queued already and its shutdown drains them.
    stopping = threading.Event()
    # The report is one file written with a temp-and-rename: two workers flushing at once
    # would race on the temp path, so the write itself is serialised (the snapshot it
    # serialises was already taken under `lock`).
    report_lock = threading.Lock()

    def write_report(complete: bool, snapshot: list[dict] | None = None) -> None:
        """Atomically (re)write the report; `complete: false` until the run finishes.

        The rows are serialised from COPIES taken under the lock (`snapshot` when the
        caller already holds it), never from the live dicts the workers mutate.
        """
        if snapshot is None:
            with lock:
                snapshot = [dict(r) for r in rows]
        with report_lock:
            atomic_write_json(report, {
                "kind": "dedupe_final_trajectory_checkpoint",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "roots": [str(r) for r in roots], "roots_outside_output_root": outside,
                "output_root": str(output_root),
                "only_tree": only_tree, "limit": limit, "max_delete": max_delete,
                "applied": apply, "streamed": stream, "complete": complete,
                "deleted": n_deleted, "deleted_bytes": bytes_deleted,
                "runs": snapshot,
            })

    def try_delete(row: dict) -> bool:
        """Delete one proven candidate; False once `--max-delete` is reached.

        The slot is RESERVED before the delete so concurrent stream workers can never
        overshoot the cap, and given back when the delete did not stick (including when a
        kill unwinds through it, so the report's count is what is actually gone).
        """
        nonlocal n_deleted, bytes_deleted, capped
        with lock:
            if max_delete and n_deleted >= max_delete:
                capped = True
                row.update(status="skipped_max_delete",
                           detail=f"--max-delete {max_delete} reached")
                return False
            n_deleted += 1
        size = row.get("bytes", 0)
        try:
            outcome = delete_candidate(row, deleted_log)
        except BaseException:
            with lock:
                n_deleted -= 1
            stopping.set()
            raise
        flush: list[dict] | None = None
        with lock:
            if outcome != "deleted":
                n_deleted -= 1
            else:
                bytes_deleted += size
                if n_deleted % FLUSH_EVERY == 0:
                    flush = [dict(r) for r in rows]
        if flush is not None:
            write_report(complete=False, snapshot=flush)
        return True

    def inspect_one(run_dir: Path) -> dict:
        """Classify one run; in stream mode delete it right here when it is proven."""
        nonlocal n_inspected, n_candidates
        row = inspect_run(run_dir, output_root)
        with lock:
            rows.append(row)
            n_inspected += 1
            n_candidates += row["status"] in CANDIDATE_STATUSES
            progress = stream and n_inspected % PROGRESS_EVERY == 0
            snapshot = (n_inspected, n_candidates, n_deleted, bytes_deleted)
        if progress:
            print(f"inspected {snapshot[0]}, candidates {snapshot[1]}, "
                  f"deleted {snapshot[2]}, freed {snapshot[3] / 2**30:.2f} GiB", flush=True)
        if stream and not stopping.is_set() and is_deletable(row):
            try_delete(row)
        return row

    try:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            try:
                # The rows are collected by the workers themselves (a streaming delete must
                # be reportable the moment it happens), so the order is restored below.
                list(pool.map(inspect_one, run_dirs))
            except BaseException:
                # Ctrl-C in THIS thread (or anything else escaping the pool) must stop the
                # workers BEFORE the executor's shutdown drains the runs already queued:
                # every streaming delete is gated on `stopping`, so it is set first.
                stopping.set()
                raise
        with lock:
            rows[len(excluded_rows):] = sorted(rows[len(excluded_rows):],
                                               key=lambda r: r["run_dir"])
        if apply and not stream:
            for row in rows:
                if not is_deletable(row):
                    continue
                # Past the cap `try_delete` marks the row `skipped_max_delete` (as stream
                # mode does) instead of leaving it looking un-considered, and sets `capped`.
                try_delete(row)
        finished = True
    finally:
        # An interrupt mid-apply still leaves a report, marked incomplete.
        write_report(complete=finished and not capped)

    print(summarize(rows))
    if outside:
        print(f"\nNOTE: roots outside the configured output root {output_root}: "
              + ", ".join(outside))
    print(f"\nreport: {report}")
    if apply:
        print(f"deleted {n_deleted} dir(s), {bytes_deleted / 2**30:.2f} GiB "
              f"(log: {deleted_log})")
        if capped:
            print(f"stopped at --max-delete {max_delete}")
    else:
        print("DRY RUN: nothing deleted (pass --apply).")


if __name__ == "__main__":
    app()
