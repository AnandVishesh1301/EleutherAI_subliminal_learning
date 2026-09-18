"""Thin a student run's checkpoint ladder down to a few epochs (dry-run by default).

A 20-epoch trajectory run saves 40 intermediate `checkpoints/checkpoint-N/` dirs; once the
weight-derived and behavioural records over that ladder exist (`eval/pref_traj.json`,
`eval/pile_kl_traj.json`, `eval/adapter_spectra_traj.json`) the intermediate adapters are
recoverable numbers, not data, and the dirs are pure storage. This script deletes the
non-kept rungs of IN-SCOPE runs only, and only after proving every required record holds a
real measured point for EVERY step the ladder ever had (`ckpt_ladder.record_points_problem`
for the behavioural records, `adapter_spectra_trajectory.spectra_record_problem` -- which
also verifies the npz bytes -- for the spectra).

Nothing is lost silently:

* the kept rungs are the ones `ckpt_ladder.snap_keep_steps` maps the requested epochs onto,
  ON DISK; an epoch with no rung within tolerance stops the whole run (`keep_step_missing`),
* the three records are COPIED into `{output_root}/analysis/thin_ckpt/archive/` before the
  first delete, so the numbers survive even the run dir itself,
* the ladder manifest (`checkpoints/ladder.json`) is written BEFORE the first delete and
  records every step that ever existed with its adapter digest, so the checkpoint-list gates
  -- which compare under COVERAGE (`ckpt_ladder.STEP_GATE`) -- keep reading a thinned run's
  full records as current, and a thinned point is known to be thinned rather than lost,
* the plan is RE-INSPECTED under the run's claim and only executed when it is unchanged.

A kill between the manifest write and the last delete leaves a state the next invocation
recognises and finishes (the manifest is rewritten with each step's `deleted_at` at the end;
rerunning is always safe).

Scope is deliberately narrow (all defaults overridable, all printed at start): cross-base
PolyPythias students only -- original-suite `pythia-160m-{,data-,weight-}seed{1,2,3}`
receivers fed by original-suite `teacher-pythia-160m*` senders, no `_sgd` / `_sch` /
`_noeos` / `_stop` arm anywhere in the path, no `ckpt`/`@step` trajectory-teacher trees --
minus the protect file's runs. Everything else is `out_of_scope:<reason>`, never touched.
Every path identity (scope, protect, delete) is taken on the RESOLVED path: a symlinked run
dir or walk root is refused outright rather than deleted through.

    uv run python scripts/thin_checkpoint_ladders.py --only-tree students/lion/pythia-160m-seed1
    uv run python scripts/thin_checkpoint_ladders.py --emit-missing-jobs /tmp/jobs.jsonl
    uv run python scripts/thin_checkpoint_ladders.py --expect-runs 4200 --apply
    uv run python scripts/thin_checkpoint_ladders.py --regen-protect-file
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import typer

from subliminal_learning.config import resolve_output_root
from subliminal_learning.utils.io import atomic_write_json
from subliminal_learning.training.ckpt_ladder import (
    SCHEMA_VERSION,
    keep_steps,
    ladder_steps,
    read_ladder,
    record_points_problem,
    run_epochs,
    snap_keep_steps,
    steps_per_epoch,
    write_ladder,
)
from subliminal_learning.utils.run_io import (
    acquire_claim,
    endpoint_link_problem,
    find_leaf_dirs_parallel,
    manifest_ref,
    release_claim,
)
# Runnable as a plain path (`python scripts/thin_checkpoint_ladders.py`) as well as with
# `-m`: the sibling-script imports below need the repo root on the path either way.
sys.path.insert(0, str(Path(__file__).absolute().parents[1]))

from scripts.adapter_spectra_trajectory import (  # noqa: E402
    JSON_NAME as SPECTRA_JSON,
    NPZ_NAME as SPECTRA_NPZ,
    spectra_record_problem,
)
from scripts.dedupe_final_trajectory_checkpoint import (  # noqa: E402
    ADAPTER,
    MANIFEST,
    SKIP_DIRS,
    _read_json,
    compare_adapters,
    dir_bytes,
    RESUME_MARKER,
    in_flight_reason,
    is_run_dir,
    unsafe_path_reason,
)

ROOT = Path(__file__).absolute().parents[1]
app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

# Original PolyPythias suite on BOTH sides: the receivers whose ladders are thinnable and
# the senders whose students count. `-pilerepro-*`, `-dclm-*`, `-weightc-*`, `-410m*` and
# the DataDecide trees are re-pretraining / cross-family arms, out of scope by default.
DEFAULT_RECEIVERS = r"^pythia-160m-(data-seed[123]|weight-seed[123]|seed[123])$"
DEFAULT_TEACHERS = r"^teacher-pythia-160m(-(data-seed|weight-seed|seed)[123])?$"
DEFAULT_KEEP_EPOCHS = "0.5,1,1.5,2,5,10"
DEFAULT_REQUIRE = "pref_traj,pile_kl_traj,adapter_spectra_traj"
# Record names with an evaluator-matched point rule in `ckpt_ladder.record_points_problem`.
POINT_KINDS = ("pref_traj", "pile_kl_traj")
DEFAULT_PROTECT = "run_logs/thin_ckpt/protect_pythia_nonadaptive.tsv"
# Trees that keep `checkpoint-*` as resume state or hold arms this script never owns.
EXCLUDED_TREES = ("students/sgd", "students/controls")
EXCLUDED_COMPONENTS = ("gkd_students", "sdf_teachers")
# Ablation arms carry their own suffix on the teacher param, pool tag or student param dir;
# none of them is this script's to thin, whatever the manifest says.
ARM_SUFFIXES = ("_sgd", "_sch", "_noeos", "_stop")
PROGRESS_EVERY = 500
# Partial reports are flushed every this many handled runs during --apply, so a kill
# mid-apply still leaves a readable (incomplete) report.
FLUSH_EVERY = 500
# `--expect-runs` tolerance: a live tree grows, a census that is 5% off is a different tree.
CENSUS_TOLERANCE = 0.05
SHA_CHUNK = 1 << 20
ARCHIVE_REL = Path("analysis") / "thin_ckpt" / "archive"

_LOG_LOCK = threading.Lock()
_ROW_LOCK = threading.Lock()


# --------------------------------------------------------------------------- paths
def normalise_key(value: str) -> str:
    """One protect-file / run-dir key in the form the resolved rel is compared against."""
    value = value.strip().rstrip("/")
    if not value:
        return ""
    if os.path.isabs(value):
        return str(Path(value).resolve())
    return os.path.normpath(value)


def resolved_rel(run_dir: Path, output_root: Path) -> tuple[str, str | None]:
    """`(rel, problem)`: the run's path relative to the output root, taken on REAL paths.

    Deleting through a symlink reaches a tree the run does not own, and a protect entry
    keyed on the logical path would not match the run reached by another name. So the rel
    every later check uses comes from `resolve()`, and any run dir that IS a symlink, or
    whose resolved path differs from its logical one (a symlinked component above it), is
    refused with a problem rather than silently rewritten.
    """
    run_dir, output_root = Path(run_dir), Path(output_root)
    root_res = output_root.resolve()
    try:
        run_res = run_dir.resolve()
    except OSError as exc:
        return str(run_dir), f"unresolvable: {exc}"
    rel = (str(run_res.relative_to(root_res)) if run_res.is_relative_to(root_res)
           else str(run_res))
    if run_dir.is_symlink():
        return rel, "run dir is a symlink"
    if run_dir.is_relative_to(output_root):
        logical = root_res / run_dir.relative_to(output_root)
        if os.path.normpath(logical) != str(run_res):
            return rel, f"resolved path differs: {run_res}"
    return rel, None


# --------------------------------------------------------------------------- scope
def scope_reason(rel: str, run_dir: Path, receivers: re.Pattern,
                 teachers: re.Pattern) -> str | None:
    """Why this run is out of scope, or None when it is thinnable in principle.

    Path-only facts first (cheap), then the manifest's `cross_base` and param tag: the walk
    classifies tens of thousands of runs and most fall out on their path alone.
    """
    parts = Path(rel).parts
    if any(p in EXCLUDED_COMPONENTS for p in parts):
        return "excluded_tree"
    if any(rel.startswith(t + "/") for t in EXCLUDED_TREES):
        return "excluded_tree"
    if parts[:1] != ("students",) or len(parts) < 5:
        return "not_a_student_run"
    if any(p == "ckpt" for p in parts):
        return "ckpt_component"
    if any("@step" in p for p in parts):
        return "at_step"
    if not receivers.match(parts[2]):
        return "receiver"
    teacher = next((p for p in parts if p.startswith("teacher-")), None)
    if teacher is None or not teachers.match(teacher):
        return "teacher"
    # The arm suffixes live on the teacher param, the pool tag or the student param dir,
    # i.e. anywhere below `students/<animal>/<receiver>`.
    if any(s in p for p in parts[3:] for s in ARM_SUFFIXES):
        return "arm_suffix"
    manifest = _read_json(run_dir / MANIFEST) or {}
    if not manifest.get("cross_base"):
        return "cross_base"
    tag = manifest.get("student_param_tag") or ""
    if any(s in tag for s in ARM_SUFFIXES):
        return "param_tag"
    return None


def read_protect_file(path: Path) -> dict[str, str]:
    """`{normalised run_dir key: reason}` from the TSV protect file (`#` comments, blanks).

    Keys are normalised exactly like the resolved rel of a run dir, so a protect entry
    written as `./a/b/`, `a/./b` or an absolute path all key the same run.
    """
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rel, _, reason = line.partition("\t")
        key = normalise_key(rel)
        if key:
            out[key] = reason.strip() or "protected"
    return out


# --------------------------------------------------------------------------- inspection
def on_disk_steps(ckpt_root: Path) -> dict[int, Path]:
    """`{step: dir}` for every `checkpoint-N` dir currently under `checkpoints/`."""
    out: dict[int, Path] = {}
    for d in ckpt_root.glob("checkpoint-*"):
        suffix = d.name.split("-")[-1]
        if d.is_dir() and suffix.isdigit():
            out[int(suffix)] = d
    return out


def sha256_file(path: Path) -> str | None:
    """Streamed sha256 of `path`, or None when it cannot be read."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(SHA_CHUNK), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def shallow_bytes(path: Path) -> int:
    """Sum of the sizes of the files directly in `path` (the `--fast-bytes` estimate).

    A checkpoint dir is flat (adapter + a few metadata files), so this is the whole dir in
    practice at one `scandir` instead of a recursive walk per run on CephFS.
    """
    total = 0
    try:
        entries = list(os.scandir(path))
    except OSError:
        return total
    for entry in entries:
        try:
            if entry.is_file(follow_symlinks=False):
                total += entry.stat().st_size
        except OSError:
            continue
    return total


def run_in_flight(run_dir: Path, disk: dict[int, Path],
                  ignore_claim: Path | None = None) -> str | None:
    """The dedupe script's in-flight gate, asked about the WHOLE ladder.

    That gate takes ONE candidate dir, whose `resume_marker.json` belongs to the queue-yield
    machinery and never to this script; here any rung carrying one means the run is mid
    resume, so the marker check is repeated over every rung and the claim / recent-write
    checks are made once. `ignore_claim` is this process's own claim, which the apply pass
    holds while it re-inspects.
    """
    for step in sorted(disk):
        if (disk[step] / RESUME_MARKER).is_file():
            return f"resume_marker.json (checkpoint-{step})"
    return in_flight_reason(run_dir, run_dir / "checkpoints", ignore_claim=ignore_claim)


def missing_records(run_dir: Path, animal: str, require: list[str],
                    ladder: list[int]) -> dict[str, str]:
    """`{record: problem}` for every required record that is not a COMPLETE ladder readout.

    Completeness is DATA, not headers: a record can declare the full `checkpoint_steps` and
    carry an empty `points` list, and a spectra JSON can name an npz that was truncated
    half-written. So each record must exist under `eval/`, carry this run's current
    `train_manifest_ref` (the identity stamp the trajectory writers store), and then pass
    its own point-level check -- `record_points_problem` for the behavioural records (with
    the run's own animal for `pref_traj`), `spectra_record_problem` (npz bytes, sha, steps,
    shape, points) for the spectra.
    """
    ref = manifest_ref(run_dir)
    problems: dict[str, str] = {}
    for name in require:
        json_name = SPECTRA_JSON if name == "adapter_spectra_traj" else f"{name}.json"
        stored = _read_json(run_dir / "eval" / json_name)
        if stored is None:
            problems[name] = "no_json"
            continue
        if stored.get("train_manifest_ref") != ref:
            problems[name] = "ref_mismatch"
            continue
        if name == "adapter_spectra_traj":
            problem = spectra_record_problem(run_dir, ladder_steps=ladder)
        else:
            # The record's own KIND picks the point-level rule, which for the two
            # trajectories is the evaluator's completeness rule (a pile point needs its
            # delta_ce + KL, a pref point the animal's p_target); a name outside those
            # (`--require` is free text) falls back to the generic "carries something".
            kind = name if name in POINT_KINDS else "generic"
            problem = record_points_problem(
                stored, ladder, kind=kind,
                animal=animal if name == "pref_traj" else None)
        if problem is not None:
            problems[name] = problem
    return problems


def inspect_run(run_dir: Path, output_root: Path, cfg: dict,
                ignore_claim: Path | None = None) -> dict:
    """Classify one run dir; any unexpected error becomes a `scan_error` row."""
    try:
        return _inspect_run(run_dir, output_root, cfg, ignore_claim=ignore_claim)
    except Exception as exc:  # noqa: BLE001 -- one bad run must not kill the scan
        rel, _ = resolved_rel(run_dir, output_root)
        return _row(rel, run_dir, "scan_error", detail=f"{type(exc).__name__}: {exc}")


def _row(rel: str, run_dir: Path, status: str, **fields) -> dict:
    row = {"run_dir_rel": rel, "run_dir": str(run_dir), "status": status,
           "n_ckpts_before": 0, "n_keep": 0, "n_delete": 0, "bytes_to_free": 0,
           "missing": [], "missing_problems": {},
           "animal": Path(rel).parts[1] if len(Path(rel).parts) > 1 else "",
           "receiver": Path(rel).parts[2] if len(Path(rel).parts) > 2 else ""}
    row.update(fields)
    return row


def _inspect_run(run_dir: Path, output_root: Path, cfg: dict,
                 ignore_claim: Path | None = None) -> dict:
    rel, path_problem = resolved_rel(run_dir, output_root)
    if path_problem is not None:
        return _row(rel, run_dir, "symlinked_run_dir", detail=path_problem)
    if not is_run_dir(run_dir):
        # Every discovery path lands here, `--run-dirs-file` included: a dir without a
        # manifest beside an endpoint adapter is not a finished run and is never thinned.
        return _row(rel, run_dir, "out_of_scope:not_a_run_dir")
    reason = scope_reason(rel, run_dir, cfg["receivers"], cfg["teachers"])
    if reason is not None:
        return _row(rel, run_dir, f"out_of_scope:{reason}")
    protect = cfg["protected"].get(rel) or cfg["protected"].get(str(Path(run_dir).resolve()))
    if protect:
        return _row(rel, run_dir, f"protected:{protect}")

    ckpt_root = run_dir / "checkpoints"
    manifest = _read_json(run_dir / MANIFEST) or {}
    n_train_steps = manifest.get("n_train_steps")
    if not isinstance(n_train_steps, int) or isinstance(n_train_steps, bool):
        return _row(rel, run_dir, "error:no_manifest_steps")
    epochs = run_epochs(run_dir)
    if not epochs:
        return _row(rel, run_dir, "error:no_epochs")

    disk = on_disk_steps(ckpt_root)
    # The endpoint adapter at the `checkpoints/` root IS the final step, so a
    # `checkpoint-{n_train_steps}` dir is a duplicate, never a kept rung: the keep epochs
    # are snapped onto the OTHER rungs only.
    rungs = [s for s in sorted(disk) if s != n_train_steps]
    targets = sorted(keep_steps(n_train_steps, epochs, cfg["keep_epochs"]))
    keep: set[int] = set()
    snapped: dict[float, int | None] = {}
    if rungs:
        snapped, problem = snap_keep_steps(rungs, n_train_steps, epochs, cfg["keep_epochs"])
        per = steps_per_epoch(n_train_steps, epochs)
        feasible = [e for e in sorted({float(x) for x in cfg["keep_epochs"]})
                    if int(math.floor(e * per + 0.5)) < n_train_steps]
        if problem is None and len(snapped) != len(feasible):
            problem = (f"{len(snapped)} snapped epoch(s) for {len(feasible)} feasible "
                       "keep epoch(s)")
        if problem is not None:
            return _row(rel, run_dir, f"keep_step_missing:{problem}",
                        n_ckpts_before=len(disk))
        keep = {s for s in snapped.values() if s is not None}

    delete_steps = sorted(s for s in disk if s not in keep)
    row = _row(rel, run_dir, "ready", n_ckpts_before=len(disk),
               n_keep=len(keep & set(disk)), n_delete=len(delete_steps))

    ladder_doc_stored = read_ladder(run_dir)
    if not delete_steps:
        # Fast path: nothing left to remove. A run that already carries a manifest is
        # THINNED (no record checks needed); one that never had intermediates is simply
        # nothing to do.
        return {**row, "status": "already_thinned" if ladder_doc_stored
                else "nothing_to_delete"}

    if ckpt_root.is_symlink() and not cfg["allow_symlinked_checkpoints"]:
        # Some ds1 ladders were relocated behind a symlink; deleting THROUGH one would
        # reach a tree this run does not own. The target is still read as the ladder.
        return {**row, "status": "symlinked_checkpoints"}

    problem = endpoint_link_problem(run_dir, [disk[s] for s in delete_steps])
    if problem is not None:
        return {**row, "status": "endpoint_symlink", "detail": problem}

    flight = run_in_flight(run_dir, disk, ignore_claim=ignore_claim)
    if flight is not None:
        return {**row, "status": "in_flight", "detail": flight}

    if n_train_steps in disk:
        verdict, detail = compare_adapters(disk[n_train_steps] / ADAPTER,
                                           ckpt_root / ADAPTER)
        if not verdict.startswith("identical"):
            # The "duplicate" final is not a duplicate: this run's endpoint story is not
            # what it looks like, so NOTHING here is deleted.
            return {**row, "status": "final_dup_differs", "detail": f"{verdict}: {detail}"}

    ladder = ladder_steps(ckpt_root)
    missing = missing_records(run_dir, row["animal"], cfg["require"], ladder)
    if missing:
        first = next(n for n in cfg["require"] if n in missing)
        return {**row, "status": f"missing:{first}:{missing[first]}",
                "missing": [n for n in cfg["require"] if n in missing],
                "missing_problems": missing}

    sizer = shallow_bytes if cfg["fast_bytes"] else dir_bytes
    row["bytes_to_free"] = sum(sizer(disk[s]) for s in delete_steps)
    row["delete_steps"] = delete_steps
    row["keep_steps"] = sorted(keep)
    row["keep_targets"] = targets
    row["keep_snapped"] = {f"{e:g}": s for e, s in snapped.items()}
    row["n_train_steps"] = n_train_steps
    row["epochs"] = epochs
    row["ladder_steps"] = ladder
    return row


# --------------------------------------------------------------------------- archive
def archive_records(run_dir: Path, row: dict, cfg: dict) -> tuple[dict[str, str], str | None]:
    """Copy the verified records out of the run before anything is deleted.

    `{output_root}/analysis/thin_ckpt/archive/<receiver>/<record>/<rel with '/'->'__'>`:
    a flat, run-keyed store that survives even the loss of the run dir, so the numbers the
    deletion relies on are never in only one place. A file already there with the same
    sha256 is left alone (rerun-safe); anything that fails to copy aborts the thinning.
    """
    base = Path(cfg["output_root"]) / ARCHIVE_REL / (row["receiver"] or "-")
    stem = row["run_dir_rel"].replace("/", "__")
    written: dict[str, str] = {}
    for name in cfg["require"]:
        names = ([SPECTRA_JSON, SPECTRA_NPZ] if name == "adapter_spectra_traj"
                 else [f"{name}.json"])
        for fname in names:
            src = run_dir / "eval" / fname
            dest = base / name / f"{stem}{Path(fname).suffix}"
            try:
                if not src.is_file():
                    return written, f"{name}: {fname} missing"
                if not (dest.is_file() and sha256_file(dest) == sha256_file(src)):
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    tmp = dest.with_name(dest.name + f".tmp-{os.getpid()}")
                    shutil.copyfile(src, tmp)
                    os.replace(tmp, dest)
            except OSError as exc:
                return written, f"{name}: {type(exc).__name__}: {exc}"
            written[fname] = str(dest.relative_to(Path(cfg["output_root"])))
    return written, None


# --------------------------------------------------------------------------- apply
def ladder_doc(run_dir: Path, row: dict, cfg: dict, deleted: dict[int, str] | None = None,
               archived: dict[str, str] | None = None) -> dict:
    """The run's ladder manifest: every step that ever existed, kept flag, adapter digest.

    `deleted` supplies the `deleted_at` stamps of the steps this invocation removed; on the
    pre-delete write it is empty, and previously recorded stamps are carried over. A stored
    digest is reused only while the adapter's size AND mtime are unchanged -- a rewritten
    adapter of the same length must never inherit the old file's sha.
    """
    previous = (read_ladder(run_dir) or {}).get("checkpoints") or {}
    ckpt_root = run_dir / "checkpoints"
    deleted = deleted or {}
    checkpoints: dict[str, dict] = {}
    for step in row["ladder_steps"]:
        was = previous.get(str(step)) or {}
        adapter = (ckpt_root / ADAPTER if step == row["n_train_steps"]
                   else ckpt_root / f"checkpoint-{step}" / ADAPTER)
        sha, size, mtime = (was.get("adapter_sha256"), was.get("adapter_bytes"),
                            was.get("adapter_mtime"))
        if adapter.is_file():
            try:
                stat = adapter.stat()
                if stat.st_size != size or stat.st_mtime != mtime:
                    sha = None
                size, mtime = stat.st_size, stat.st_mtime
            except OSError:
                sha = None
            if not cfg["no_sha"] and sha is None:
                sha = sha256_file(adapter)
        checkpoints[str(step)] = {
            "kept": step not in row["delete_steps"],
            "adapter_sha256": sha, "adapter_bytes": size, "adapter_mtime": mtime,
            "deleted_at": deleted.get(step) or was.get("deleted_at"),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "n_train_steps": row["n_train_steps"],
        "epochs": row["epochs"],
        "steps_per_epoch": steps_per_epoch(row["n_train_steps"], row["epochs"]),
        "checkpoint_steps": row["ladder_steps"],
        "checkpoints": checkpoints,
        "keep_epochs": cfg["keep_epochs"],
        "keep_targets": row.get("keep_targets") or [],
        "keep_steps": row.get("keep_steps") or [],
        "archived": archived or (read_ladder(run_dir) or {}).get("archived") or {},
        "derived_docs_verified": {name: True for name in cfg["require"]},
        "thinned_at": datetime.now(timezone.utc).isoformat(),
        "thinned_by": "scripts.thin_checkpoint_ladders",
        "train_manifest_ref": manifest_ref(run_dir),
    }


def _append_jsonl(path: Path, record: dict) -> None:
    with _LOG_LOCK, path.open("a") as fh:
        fh.write(json.dumps(record) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _plan_changed(row: dict, fresh: dict) -> str | None:
    """Why the re-inspected plan is not the planned one, or None when they agree."""
    if fresh["status"] != "ready":
        return fresh["status"]
    if (fresh.get("delete_steps") != row.get("delete_steps")
            or fresh.get("keep_steps") != row.get("keep_steps")):
        return "plan differs"
    return None


def thin_run(row: dict, cfg: dict, log_path: Path) -> str:
    """Archive the records, write the ladder manifest, then delete the non-kept rungs.

    Order is the whole safety argument. Under the run's claim the plan is RE-INSPECTED
    (the claim itself excluded from the in-flight test) and executed only if it is
    bit-identical to the planned one -- between the scan and here a record can have been
    superseded or a checkpoint re-saved. Then the records are copied to the archive, then
    the manifest (which makes every stored record read as covering a ladder longer than
    the disk) is written BEFORE the first `rmtree` and rewritten with the `deleted_at`
    stamps after the last one -- in a `finally`, so even a ^C records what went. A kill
    anywhere in between leaves non-kept dirs on disk beside a manifest, which the next
    invocation re-classifies as deletable and finishes.
    """
    run_dir = Path(row["run_dir"])
    claim = acquire_claim(run_dir)
    if claim is None:
        return "skipped_claimed"
    try:
        fresh = inspect_run(run_dir, Path(cfg["output_root"]), cfg, ignore_claim=claim)
        changed = _plan_changed(row, fresh)
        if changed is not None:
            row["detail"] = f"re-inspection: {changed}"
            return f"plan_changed:{changed}"

        archived, problem = archive_records(run_dir, row, cfg)
        if problem is not None:
            row["detail"] = problem
            return "archive_failed"

        write_ladder(run_dir, ladder_doc(run_dir, row, cfg, archived=archived))
        deleted: dict[int, str] = {}
        failures: list[str] = []
        try:
            for step in row["delete_steps"]:
                target = run_dir / "checkpoints" / f"checkpoint-{step}"
                unsafe = unsafe_path_reason(run_dir, target)
                if unsafe is not None:
                    failures.append(f"{step}: {unsafe}")
                    continue
                if not target.is_dir():
                    continue
                try:
                    shutil.rmtree(target)
                except Exception as exc:  # noqa: BLE001 -- one failed rung keeps the rest
                    failures.append(f"{step}: {type(exc).__name__}: {exc}")
                    continue
                stamp = datetime.now(timezone.utc).isoformat()
                deleted[step] = stamp
                _append_jsonl(log_path, {"run_dir": row["run_dir"], "step": step,
                                         "path": str(target), "deleted_at": stamp})
        finally:
            write_ladder(run_dir, ladder_doc(run_dir, row, cfg, deleted, archived))
        if failures:
            row["detail"] = "; ".join(failures[:3])
            return "thin_partial"
        return "thinned"
    finally:
        release_claim(claim)


# --------------------------------------------------------------------------- jobs
CD = f"cd {ROOT} && "
JOBS_REL = Path("run_logs") / "thin_ckpt" / "jobs"


def spectra_command(rel: str) -> str:
    """The one-run spectra command a chunk file holds (repo-rooted, niced)."""
    return (CD + "nice -n 15 uv run python -m scripts.adapter_spectra_trajectory "
            f"--roots {rel} --workers 1")


def write_spectra_jobs(rows: list[dict], stamp: str, chunk: int) -> tuple[list[dict], Path]:
    """Chunk files of per-run spectra commands + one jobq record per chunk.

    Spectra is the ONLY fillable record here: it is addressed by run dir, so a missing one
    can be recomputed from the checkpoints that still exist. `pref_traj` / `pile_kl_traj`
    are CELL-addressed against a registry this script does not own, so runs missing them
    are reported as `unfillable` instead of being emitted as guesses.
    """
    runs = [r["run_dir_rel"] for r in rows
            if "adapter_spectra_traj" in (r.get("missing") or [])]
    out_dir = ROOT / JOBS_REL / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for index in range(0, len(runs), max(1, chunk)):
        part = runs[index:index + max(1, chunk)]
        name = f"chunk_{index // max(1, chunk):02d}.txt"
        (out_dir / name).write_text("".join(spectra_command(r) + "\n" for r in part))
        rel_path = (JOBS_REL / stamp / name).as_posix()
        jobs.append({"key": f"thin-spectra|{stamp}|{name}",
                     "cmd": CD + f"bash {rel_path}",
                     "record": "adapter_spectra_traj", "chunk": rel_path,
                     "n_runs": len(part), "run_dir_rels": part})
    return jobs, out_dir


def unfillable_rows(rows: list[dict]) -> list[dict]:
    """Runs whose missing records cannot be emitted as jobs (pref / pile-KL)."""
    out = []
    for row in rows:
        missing = [n for n in (row.get("missing") or []) if n != "adapter_spectra_traj"]
        if missing:
            out.append({"run_dir_rel": row["run_dir_rel"], "missing": missing,
                        "problems": {n: row["missing_problems"].get(n) for n in missing}})
    return out


# --------------------------------------------------------------------------- protect file
PROTECT_TEACHER = "teacher-pythia-160m/ep2_lr5e-4_r32a64_tseed1/ex10-20_prefix_n100k"
PROTECT_RECEIVER = "students/dog/pythia-160m-data-seed1"
PROTECT_HARVEST = Path("analysis") / "nonadaptive_8x7_harvest.json"
PROTECT_EXPECT = {"lattice504": 1231, "earlystop": 505}


def build_protect_rows(out_root: Path) -> list[tuple[str, str]]:
    """`(resolved rel, reason)` for every run the nonadaptive campaign still needs.

    The 504-cell lattice (`analysis/nonadaptive_8x7_harvest.json`, ds1 receiver, strip
    teacher) at every `ftseed*`, plus every `_stop*` early-stop tag dir there. Counts are
    asserted (`PROTECT_EXPECT`): a regeneration that finds a different population is a
    changed tree, not a new protect file, and must not silently shrink the protection.
    """
    base = out_root / PROTECT_RECEIVER / PROTECT_TEACHER
    harvest = json.loads((out_root / PROTECT_HARVEST).read_text())
    tags = sorted({r["tag"] for r in harvest})
    if len(tags) != 504:
        raise typer.BadParameter(f"{PROTECT_HARVEST}: {len(tags)} tags, expected 504")
    rows: list[tuple[str, str]] = []
    counts = {"lattice504": 0, "earlystop": 0}
    for tag_dir, reason in ([(base / t, "lattice504") for t in tags]
                            + [(d, "earlystop") for d in sorted(base.glob("*_stop*"))]):
        if not tag_dir.is_dir():
            continue
        for run in sorted(tag_dir.glob("ftseed*")):
            if not run.is_dir():
                continue
            rel, problem = resolved_rel(run, out_root)
            if problem is not None:
                raise typer.BadParameter(f"{run}: {problem}")
            rows.append((rel, reason))
            counts[reason] += 1
    if counts != PROTECT_EXPECT:
        raise typer.BadParameter(f"protect census {counts} != expected {PROTECT_EXPECT}; "
                                 "nothing written")
    return sorted(rows)


def regen_protect_file(out_root: Path, path: Path) -> int:
    """Rewrite the protect file from the tree; returns the number of protected runs."""
    rows = build_protect_rows(out_root)
    stamp = f"{datetime.now(timezone.utc):%Y-%m-%d}"
    header = (f"# run_dir_rel\treason  (generated {stamp}; ds1 receiver, strip teacher "
              "ep2_lr5e-4_r32a64_tseed1; 504-cell lattice all ftseeds + _stop early-stop "
              "arm; keys are RESOLVED rels)\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + "".join(f"{rel}\t{reason}\n" for rel, reason in rows))
    return len(rows)


# --------------------------------------------------------------------------- reporting
def summarize(rows: list[dict]) -> str:
    """Counts + bytes per status, then per receiver and per animal (in-scope runs only)."""
    def table(title: str, key) -> list[str]:
        groups: dict[str, list[dict]] = {}
        for row in rows:
            groups.setdefault(str(key(row)), []).append(row)
        out = [f"{title}:"]
        for name, group in sorted(groups.items()):
            out.append(f"  {name:<60} {len(group):>7}  "
                       f"{sum(r['bytes_to_free'] for r in group) / 2**30:>9.2f} GiB")
        return out

    in_scope = [r for r in rows if not r["status"].startswith("out_of_scope")]
    lines = [f"runs scanned            {len(rows)}",
             f"in scope                {len(in_scope)}",
             f"bytes to free           {sum(r['bytes_to_free'] for r in rows) / 2**30:.2f} GiB",
             ""]
    lines += table("per status", lambda r: r["status"])
    lines.append("")
    lines += table("per receiver (in scope)", lambda r: r["receiver"] or "-")
    lines.append("")
    lines += table("per animal (in scope)", lambda r: r["animal"] or "-")
    return "\n".join(lines)


def census_rows(rows: list[dict]) -> list[dict]:
    """The runs the `--expect-runs` census counts: in scope AND actually classified."""
    return [r for r in rows if not r["status"].startswith("out_of_scope")
            and r["status"] != "scan_error"]


# --------------------------------------------------------------------------- CLI
@app.command()
def main(
    receivers: str = typer.Option(DEFAULT_RECEIVERS, help="Regex the receiver (student "
                                  "model) path component must match."),
    teachers: str = typer.Option(DEFAULT_TEACHERS, help="Regex the `teacher-*` path "
                                 "component must match."),
    keep_epochs: str = typer.Option(DEFAULT_KEEP_EPOCHS,
                                    help="Comma list of epochs whose checkpoints survive."),
    require: str = typer.Option(DEFAULT_REQUIRE, help="Comma list of eval/<name>.json "
                                "records that must be COMPLETE before any delete."),
    protect_file: Path = typer.Option(ROOT / DEFAULT_PROTECT, "--protect-file",
                                      help="TSV of run dirs never to touch."),
    no_protect_file: bool = typer.Option(False, "--no-protect-file",
                                         help="Allow --apply with no protect file."),
    regen_protect: bool = typer.Option(False, "--regen-protect-file",
                                       help="Rewrite the protect file from the tree, "
                                            "assert its census, and exit."),
    only_tree: str = typer.Option(None, "--only-tree", help="Restrict to one tree prefix."),
    run_dirs_file: Path = typer.Option(None, help="Text file, one run dir per line."),
    limit: int = typer.Option(0, help="Inspect at most N run dirs (0 = no limit)."),
    workers: int = typer.Option(4, help="Threads for inspection/deletion."),
    walk_workers: int = typer.Option(8, "--walk-workers", help="Threads for the walk."),
    fast_bytes: bool = typer.Option(False, "--fast-bytes",
                                    help="Size checkpoint dirs from their top level only."),
    no_sha: bool = typer.Option(False, "--no-sha",
                                help="Skip the adapter sha256 in the ladder manifest."),
    allow_symlinked_checkpoints: bool = typer.Option(
        False, help="Delete through a symlinked checkpoints/ dir too."),
    expect_runs: int = typer.Option(0, "--expect-runs", help="Abort before --apply when "
                                    "the in-scope census differs by more than 5%."),
    emit_missing_jobs: Path = typer.Option(None, "--emit-missing-jobs",
                                           help="Write spectra jobq lines here."),
    jobs_chunk: int = typer.Option(40, "--jobs-chunk",
                                   help="Run dirs per emitted spectra job."),
    report: Path = typer.Option(None, help="JSON report path. Default "
                                "run_logs/thin_ckpt/report_<UTC>.json"),
    apply: bool = typer.Option(False, "--apply", help="Actually delete the non-kept rungs."),
    output_root: Path = typer.Option(None),
) -> None:
    out_root = resolve_output_root(output_root).resolve()
    if regen_protect:
        n = regen_protect_file(out_root, protect_file)
        print(f"protect file rewritten: {protect_file} ({n} runs)", file=sys.stderr)
        return
    if limit and expect_runs:
        raise typer.BadParameter("--limit truncates the scan, so it cannot be combined "
                                 "with the --expect-runs census guard")
    cfg = {
        "receivers": re.compile(receivers), "teachers": re.compile(teachers),
        "keep_epochs": [float(e) for e in keep_epochs.split(",") if e.strip()],
        "require": [r.strip() for r in require.split(",") if r.strip()],
        "fast_bytes": fast_bytes, "no_sha": no_sha,
        "allow_symlinked_checkpoints": allow_symlinked_checkpoints,
        "protected": {}, "output_root": out_root,
    }
    if protect_file.is_file():
        cfg["protected"] = read_protect_file(protect_file)
    if apply and not cfg["protected"] and not no_protect_file:
        raise typer.BadParameter(
            f"protect file {protect_file} is missing or empty; pass --no-protect-file to "
            "thin without one")

    print("scope:", file=sys.stderr)
    for label, value in (("receivers", receivers), ("teachers", teachers),
                         ("keep_epochs", keep_epochs), ("require", require),
                         ("protect_file", f"{protect_file} ({len(cfg['protected'])} runs)"),
                         ("only_tree", only_tree or "-"), ("output_root", out_root),
                         ("mode", "APPLY" if apply else "dry run")):
        print(f"  {label:<16} {value}", file=sys.stderr)

    walk_errors: list = []
    bad_roots: list[dict] = []
    if run_dirs_file:
        run_dirs = [Path(x) if Path(x).is_absolute() else out_root / x
                    for x in (s.strip() for s in run_dirs_file.read_text().splitlines()) if x]
    else:
        roots = ([out_root / only_tree] if only_tree
                 else _receiver_roots(out_root, cfg["receivers"]))
        keep_roots = []
        for root in roots:
            _, problem = resolved_rel(root, out_root)
            if problem is not None:
                bad_roots.append({"root": str(root), "problem": problem})
                continue
            keep_roots.append(root)
        run_dirs = find_leaf_dirs_parallel(
            keep_roots, is_run_dir, skip_names=SKIP_DIRS, skip_prefixes=("checkpoint-",),
            max_depth=10, workers=walk_workers, progress=True, errors=walk_errors)
    run_dirs = sorted(set(run_dirs))
    if limit:
        run_dirs = run_dirs[:limit]

    for bad in bad_roots:
        print(f"REFUSED walk root {bad['root']}: {bad['problem']}", file=sys.stderr)
    for path, kind in walk_errors:
        print(f"walk error {path}: {kind}", file=sys.stderr)
    if apply and walk_errors:
        raise typer.BadParameter(f"{len(walk_errors)} walk error(s): the run list is "
                                 "possibly truncated; nothing deleted")

    report = report or (ROOT / "run_logs" / "thin_ckpt" /
                        f"report_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json")
    report.parent.mkdir(parents=True, exist_ok=True)
    log_path = report.parent / (report.stem + ".deleted.jsonl")

    rows: list[dict] = []
    n_seen = 0
    census_ok = threading.Event()
    if not apply or not expect_runs:
        census_ok.set()

    def write_report() -> None:
        with _ROW_LOCK:
            snapshot = [dict(r) for r in rows]
        snapshot.sort(key=lambda r: r["run_dir_rel"])
        atomic_write_json(report, {
            "kind": "thin_checkpoint_ladders",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "output_root": str(out_root), "only_tree": only_tree, "limit": limit,
            "applied": apply, "receivers": receivers, "teachers": teachers,
            "keep_epochs": cfg["keep_epochs"], "require": cfg["require"],
            "protect_file": str(protect_file), "refused_roots": bad_roots,
            "walk_errors": [[str(p), k] for p, k in walk_errors],
            "unfillable": unfillable_rows(snapshot), "runs": snapshot,
        })

    def handle(run_dir: Path) -> dict:
        nonlocal n_seen
        row = inspect_run(run_dir, out_root, cfg)
        with _ROW_LOCK:
            rows.append(row)
            n_seen += 1
            progress, flush, seen = (n_seen % PROGRESS_EVERY == 0,
                                     apply and n_seen % FLUSH_EVERY == 0, n_seen)
        if progress:
            print(f"inspected {seen}/{len(run_dirs)}", file=sys.stderr, flush=True)
        if apply and row["status"] == "ready" and census_ok.is_set():
            status = thin_run(row, cfg, log_path)
            with _ROW_LOCK:
                row["status"] = status
        if flush:
            write_report()
        return row

    if apply and expect_runs:
        # Census guard: the in-scope count has to be classified BEFORE anything is deleted,
        # so the scan runs once read-only and the apply pass follows.
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            list(pool.map(handle, run_dirs))
        scan_errors = [r for r in rows if r["status"] == "scan_error"]
        for row in scan_errors:
            print(f"scan error {row['run_dir_rel']}: {row.get('detail')}", file=sys.stderr)
        if scan_errors:
            raise typer.BadParameter(f"{len(scan_errors)} run(s) failed to classify; "
                                     "nothing deleted")
        counted = census_rows(rows)
        if abs(len(counted) - expect_runs) > CENSUS_TOLERANCE * expect_runs:
            raise typer.BadParameter(
                f"census guard: {len(counted)} in-scope runs vs --expect-runs "
                f"{expect_runs} (>{CENSUS_TOLERANCE:.0%} apart); nothing deleted")
        census_ok.set()
        ready = [r for r in rows if r["status"] == "ready"]
        done = 0
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for row, status in zip(ready, pool.map(
                    lambda r: thin_run(r, cfg, log_path), ready)):
                with _ROW_LOCK:
                    row["status"] = status
                done += 1
                if done % FLUSH_EVERY == 0:
                    write_report()
    else:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            list(pool.map(handle, run_dirs))

    rows.sort(key=lambda r: r["run_dir_rel"])
    write_report()

    missing_rows = [r for r in rows if r["status"].startswith("missing:")]
    if emit_missing_jobs:
        stamp = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
        jobs, chunk_dir = write_spectra_jobs(missing_rows, stamp, jobs_chunk)
        emit_missing_jobs.parent.mkdir(parents=True, exist_ok=True)
        emit_missing_jobs.write_text("".join(json.dumps(j) + "\n" for j in jobs))
        print(f"\nspectra jobs -> {emit_missing_jobs} ({len(jobs)} chunks in {chunk_dir})",
              file=sys.stderr)
    unfillable = unfillable_rows(missing_rows)
    if unfillable:
        print(f"unfillable (pref/pile-KL missing, cell-addressed): {len(unfillable)} runs",
              file=sys.stderr)

    print(summarize(rows), file=sys.stderr)
    print(f"\nreport: {report}", file=sys.stderr)
    if not apply:
        print("DRY RUN: nothing deleted (pass --apply).", file=sys.stderr)


def _receiver_roots(out_root: Path, receivers: re.Pattern) -> list[Path]:
    """`students/<animal>/<receiver>` dirs whose receiver matches, the walk's entry points.

    Rooting the walk at the receivers (rather than `students/`) keeps the CephFS scan off
    the cross-family trees entirely; the per-run scope check re-applies the same regex.
    """
    roots = []
    students = out_root / "students"
    for animal in sorted(p for p in students.glob("*") if p.is_dir()):
        for receiver in sorted(p for p in animal.glob("*") if p.is_dir()):
            if receivers.match(receiver.name):
                roots.append(receiver)
    return roots


if __name__ == "__main__":
    app()
