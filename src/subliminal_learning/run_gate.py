"""Skip/retrain/error gate shared by the teacher- and student-training pipelines.

A trained cell's `train_manifest.json` records the `run_identity` — the canonical
set of every input that can change the artifact. On rerun the pipeline compares the
stored identity to the freshly computed one and decides:

  - **skip**    identical identity, artifact present -> reload, don't retrain.
  - **retrain** force, or no usable stored identity (missing/corrupt manifest) ->
    train over the cell (callers first clear stale manifest + checkpoints).
  - **error**   identity drift on a policy-frozen key -> refuse to silently overwrite.

The non-path-keyed hyperparameters are frozen policy: if they drift under an
unchanged output path the old artifact would be clobbered and a later rerun under
the old config would match the old manifest and skip — serving garbage. So drift
fails loudly with a key-diff instead. The one designed exception (student side) is a
refreshed upstream number dataset, i.e. a `filtered_sha256`-only drift.

Identity is CONTENT-addressed, not location-addressed: keys naming where an input lives
(see `_LOCATION_KEYS`) are recorded but never compared, so relocating storage never
invalidates a resumable cell.

The decision is a pure function of (stored_manifest_or_None, current_identity, force)
so it is unit-testable without training.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from subliminal_learning.training import ckpt_ladder
from subliminal_learning.utils.io import atomic_write_json

_MISSING = object()

# Written LAST inside a full-state `checkpoint-N`, after every trainer file is on disk:
# its presence is the proof the save completed (see `write_resume_marker`).
RESUME_MARKER_NAME = "resume_marker.json"

# Dropped in the run dir when a SIGTERM (queue yield) interrupted training, as the very
# last write of the process; the workers' gate reads it as "resumable, no live writer".
INTERRUPTED_MARKER_NAME = "interrupted.json"

# The three gate outcomes (kept as bare strings for trivial comparison/logging).
SKIP = "skip"
RETRAIN = "retrain"
ERROR = "error"

# Identity keys that record WHERE an input lives rather than WHAT it is. They are written
# into the manifest as provenance but never compared, because moving storage does not
# change the artifact -- and every one of them is paired with a content hash of the same
# file, which is the real identity:
#
#     number_dataset_dir  <-> filtered_sha256   (distill_student)
#     template_path       <-> template_sha256   (train_teacher_templates)
#     numbers_path        <-> numbers_sha256    (cka)
#
# Comparing them made resumability location-dependent. The 2026-08-07 ssd-2 -> ssd-3 move
# invalidated 43,263 student manifests describing byte-identical data, and every affected
# cell then failed its gate with an ERROR on rerun. `.resolve()` made it worse still: the
# same file reached through a symlinked tree yielded a different identity.
_LOCATION_KEYS = frozenset({"number_dataset_dir", "template_path", "numbers_path"})


def _without_location_keys(identity: dict[str, Any]) -> dict[str, Any]:
    """Identity with location-only keys dropped, so comparison is content-addressed.

    Applied to BOTH sides: dropping the key from newly built identities alone would not
    help, since manifests stored before the change still carry it and would diff against
    ``<missing>``.
    """
    return {k: v for k, v in identity.items() if k not in _LOCATION_KEYS}


def identities_match(stored_identity: dict | None, current_identity: dict) -> bool:
    """Content-addressed identity equality: the gate's comparison, minus location keys.

    The single predicate every "is this run current?" pre-check must use, so a preflight
    never disagrees with `gate_action` (a run trained from a checkout under another home
    records a different `template_path` but the same `template_sha256`; it IS current).
    """
    if not isinstance(stored_identity, dict):
        return False
    return _without_location_keys(stored_identity) == _without_location_keys(current_identity)


@dataclass(frozen=True)
class GateDecision:
    """Outcome of the skip/retrain/error gate.

    Attributes:
        action: One of SKIP / RETRAIN / ERROR.
        diff: For ERROR (and the logged RETRAIN-on-refresh case) the recursive
            identity diff `{dotted.key: (stored, current)}`; empty otherwise.
    """

    action: str
    diff: dict[str, tuple[Any, Any]]


def identity_diff(
    stored: dict[str, Any], current: dict[str, Any]
) -> dict[str, tuple[Any, Any]]:
    """Recursive diff of two identity dicts, keyed by dotted path.

    E.g. `{"train.learning_rate": (0.0002, 0.0001)}`. Nested dicts are descended;
    a key present in only one side maps to a sentinel `<missing>` on the other.
    """
    diff: dict[str, tuple[Any, Any]] = {}
    for key in set(stored) | set(current):
        sv = stored.get(key, _MISSING)
        cv = current.get(key, _MISSING)
        if isinstance(sv, dict) and isinstance(cv, dict):
            for subkey, pair in identity_diff(sv, cv).items():
                diff[f"{key}.{subkey}"] = pair
        elif sv != cv:
            diff[key] = (
                "<missing>" if sv is _MISSING else sv,
                "<missing>" if cv is _MISSING else cv,
            )
    return diff


def format_identity_diff(diff: dict[str, tuple[Any, Any]]) -> str:
    """One `key: stored X != current Y` line per differing key, sorted."""
    return "\n".join(
        f"  {key}: stored {stored!r} != current {current!r}"
        for key, (stored, current) in sorted(diff.items())
    )


def _stored_identity(stored_manifest: dict | None) -> dict | None:
    """The `run_identity` sub-dict of a stored manifest, or None if unusable."""
    if stored_manifest is None:
        return None
    identity = stored_manifest.get("run_identity")
    return identity if isinstance(identity, dict) else None


def gate_action(
    stored_manifest: dict | None,
    current_identity: dict,
    force: bool,
) -> GateDecision:
    """Teacher gate: skip on match, error on any drift (unless force), else retrain.

    A missing/corrupt manifest (no usable `run_identity`) retrains over the cell.
    """
    stored_identity = _stored_identity(stored_manifest)
    if force or stored_identity is None:
        return GateDecision(RETRAIN, {})
    stored_cmp = _without_location_keys(stored_identity)
    current_cmp = _without_location_keys(current_identity)
    if stored_cmp == current_cmp:
        return GateDecision(SKIP, {})
    return GateDecision(ERROR, identity_diff(stored_cmp, current_cmp))


def student_gate_action(
    stored_manifest: dict | None,
    current_identity: dict,
    force: bool,
) -> GateDecision:
    """Student gate: like `gate_action`, but a `filtered_sha256`-only drift retrains.

    A refreshed upstream number dataset (only `filtered_sha256` changed) is the
    designed content flow -> RETRAIN, carrying the diff so the caller can log which
    sha changed. Any other differing key -> ERROR (unless force).
    """
    stored_identity = _stored_identity(stored_manifest)
    if force or stored_identity is None:
        return GateDecision(RETRAIN, {})
    stored_cmp = _without_location_keys(stored_identity)
    current_cmp = _without_location_keys(current_identity)
    if stored_cmp == current_cmp:
        return GateDecision(SKIP, {})
    diff = identity_diff(stored_cmp, current_cmp)
    if set(diff) == {"filtered_sha256"}:
        return GateDecision(RETRAIN, diff)
    return GateDecision(ERROR, diff)



def write_resume_identity(
    run_dir: Path, identity: dict, *, dtype: str = "float32", save_steps: int = 500
) -> None:
    """Record inputs allowed to resume an interrupted training run atomically."""
    atomic_write_json(run_dir / "resume_identity.json", {
        "identity": _without_location_keys(identity),
        "dtype": dtype,
        "save_steps": save_steps,
    })


def write_resume_marker(checkpoint_dir: Path, *, step: int, reason: str) -> None:
    """Stamp a finished full-state checkpoint dir as resumable (atomic, LAST write).

    Written only after every checkpoint file is on disk, so a dir torn by the SIGKILL
    that follows the queue's yield grace has no marker and can never be resumed from.
    `reason` is "sigterm" (queue yield) or "scheduled" (the run's own save cadence);
    the pruner deletes off-schedule "sigterm" dirs once the run finishes.
    """
    atomic_write_json(checkpoint_dir / RESUME_MARKER_NAME, {
        "step": step,
        "reason": reason,
        "timestamp": time.time(),
        "host": socket.gethostname(),
        "pid": os.getpid(),
    })


def read_resume_marker(checkpoint_dir: Path) -> dict | None:
    """The checkpoint's resume marker, or None if absent/corrupt."""
    try:
        marker = json.loads((checkpoint_dir / RESUME_MARKER_NAME).read_text())
    except (FileNotFoundError, json.JSONDecodeError, NotADirectoryError):
        return None
    return marker if isinstance(marker, dict) else None


def checkpoint_is_resumable(checkpoint_dir: Path) -> bool:
    """True when this dir holds a COMPLETE, marked full-state checkpoint.

    Adapter weights + trainer_state (at the dir's own step) + optimizer/scheduler/RNG
    state + the marker `write_resume_marker` writes last. `save_only_model` is no longer
    an automatic veto: a yield saves full state regardless of the run's config, so the
    files and the marker decide.
    """
    try:
        step = int(checkpoint_dir.name.split("-")[-1])
        state = json.loads((checkpoint_dir / "trainer_state.json").read_text())
        config = json.loads((checkpoint_dir / "adapter_config.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError, NotADirectoryError, ValueError):
        return False
    if (not isinstance(state, dict) or not isinstance(config, dict)
            or type(state.get("global_step")) is not int
            or state["global_step"] != step):
        return False
    marker = read_resume_marker(checkpoint_dir)
    if marker is None or marker.get("step") != step:
        return False
    weights = [checkpoint_dir / n for n in ("adapter_model.safetensors", "adapter_model.bin")]
    if not any(p.is_file() and p.stat().st_size > 0 for p in weights):
        return False
    if not any(p.is_file() and p.stat().st_size > 0
               for p in checkpoint_dir.glob("rng_state*.pth")):
        return False
    return all((checkpoint_dir / n).is_file() and (checkpoint_dir / n).stat().st_size > 0
               for n in ("optimizer.pt", "scheduler.pt"))


def _numeric_checkpoints(checkpoints_dir: Path) -> list[Path]:
    """`checkpoint-N` dirs under `checkpoints_dir`, unsorted."""
    return [
        p for p in checkpoints_dir.glob("checkpoint-*")
        if p.is_dir() and re.fullmatch(r"checkpoint-[0-9]+", p.name)
    ]


def checkpoint_step(checkpoint_dir: Path) -> int:
    """The N of `checkpoint-N`."""
    return int(checkpoint_dir.name.split("-")[-1])


def latest_resumable_checkpoint(checkpoints_dir: Path) -> Path | None:
    """The highest complete + marked `checkpoint-N`, or None. PURE: reads only.

    A torn dir with a HIGHER step does NOT veto the resume (that would cost a run every
    earlier checkpoint), but removing it is not this function's business: the workers'
    skip gate and `checkpoint_resume_is_safe` call this against run dirs ANOTHER live
    process may be training in, where a higher unmarked dir with a partial `optimizer.pt`
    is simply a save in progress. Only the confirmed live writer prunes, via
    `remove_torn_checkpoints_above`.
    """
    resumable = [p for p in _numeric_checkpoints(checkpoints_dir)
                 if checkpoint_is_resumable(p)]
    return max(resumable, key=checkpoint_step) if resumable else None


def remove_torn_checkpoints_above(checkpoints_dir: Path, keep: Path) -> list[str]:
    """Delete dirs above `keep` left torn by an interrupted save. WRITER ONLY.

    Called from the training process itself (`teachers.sft_io`), just before
    `trainer.train()` resumes from `keep`: only there is it certain that no one else is
    mid-save, so an unmarked higher dir really is garbage rather than a live write.
    Weights-only dirs (a `save_only_model` trajectory checkpoint has no optimizer state
    and no marker) are the campaign's readouts, not torn saves, and are left alone -- as
    is everything at or below `keep`.

    Returns the removed dir names.
    """
    removed = []
    for leaf in sorted(_numeric_checkpoints(checkpoints_dir), key=checkpoint_step):
        if checkpoint_step(leaf) <= checkpoint_step(keep):
            continue
        if _is_weights_only_checkpoint(leaf):
            continue
        logger.warning(
            "removing torn checkpoint {} (no resume marker, above the resumable {})",
            leaf, keep.name,
        )
        shutil.rmtree(leaf, ignore_errors=True)
        removed.append(leaf.name)
    return removed


def _is_weights_only_checkpoint(checkpoint_dir: Path) -> bool:
    """A complete `save_only_model` save: adapter + trainer_state, no optimizer state."""
    weights = [checkpoint_dir / n for n in ("adapter_model.safetensors", "adapter_model.bin")]
    if not any(p.is_file() and p.stat().st_size > 0 for p in weights):
        return False
    if not (checkpoint_dir / "trainer_state.json").is_file():
        return False
    return not any((checkpoint_dir / n).exists()
                   for n in ("optimizer.pt", "scheduler.pt", RESUME_MARKER_NAME))


def is_scheduled_step(
    step: int,
    *,
    save_steps: int,
    explicit_steps: tuple[int, ...] = (),
    checkpoint_resume: bool = False,
    periodic_active: bool = True,
) -> bool:
    """Would the run's OWN save schedule have written `checkpoint-{step}`?

    The schedule is the UNION of two independent rules, because the trainer applies both:

    - `explicit_steps` -- the RESOLVED list,
      `students.early_stop.resolve_save_steps(stop_step, save_at_steps)`, i.e. the union
      with `stop_step`, so an early-stop arm's terminal checkpoint counts as scheduled.
    - the periodic interval, active only when the run opted into mid-run checkpointing AND
      `save_at_steps` is None (`periodic_active`) -- an explicit `save_at_steps` list is
      what pushes HF's `save_steps` beyond the run; `stop_step` alone does NOT, so a run
      with `stop_step` set and `save_at_steps` None still saves every `save_steps`.

    A plain run has an EMPTY schedule, so every yield checkpoint of such a run is
    disposable.
    """
    if step in tuple(explicit_steps):
        return True
    return (periodic_active and bool(checkpoint_resume)
            and save_steps > 0 and step % save_steps == 0)


def _dir_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def prune_superseded_yield_checkpoints(
    checkpoints_dir: Path,
    *,
    keep_step: int,
    save_steps: int,
    explicit_steps: tuple[int, ...] = (),
    checkpoint_resume: bool = False,
    periodic_active: bool = True,
) -> list[str]:
    """Drop off-schedule "sigterm" checkpoints from EARLIER yields of this run.

    Several yields in one run would otherwise accumulate one full-state dir each; only
    the newest (`keep_step`) is needed to resume, and scheduled steps are never touched.
    Returns the removed dir names.
    """
    removed = []
    for leaf in _numeric_checkpoints(checkpoints_dir):
        step = checkpoint_step(leaf)
        marker = read_resume_marker(leaf)
        if step == keep_step or marker is None or marker.get("reason") != "sigterm":
            continue
        if is_scheduled_step(step, save_steps=save_steps, explicit_steps=explicit_steps,
                             checkpoint_resume=checkpoint_resume,
                             periodic_active=periodic_active):
            continue
        shutil.rmtree(leaf, ignore_errors=True)
        removed.append(leaf.name)
    return removed


def prune_resume_state(
    run_dir: Path,
    *,
    save_steps: int,
    explicit_steps: tuple[int, ...] = (),
    checkpoint_resume: bool = False,
    periodic_active: bool = True,
    final_step: int | None = None,
) -> dict[str, Any]:
    """Delete resume-only state once a run has FINISHED (called after the manifest).

    Optimizer/scheduler/RNG/scaler state exists to restart an interrupted run; a finished
    run will never restart, and on the shared stash those files are several times the
    adapter. Off-schedule "sigterm" dirs (a yield at a step the run never meant to keep)
    go entirely; scheduled checkpoints keep their adapter and `trainer_state.json`.

    `final_step` (the run's `n_steps`) removes `checkpoint-{final_step}` ENTIRELY,
    scheduled or not: the finished run's `save_model` wrote those same weights to
    `checkpoints/` itself, so the dir is a pure duplicate (~1.18 GB per Qwen run).
    `students.early_stop.SkipFinalStepSave` stops new runs from writing it at all; this
    catches the one case that still can — a queue yield landing on the final step, which
    saves (it must: nothing else has the weights yet) and is resumed to completion.

    Returns `{"checkpoints_removed": [...], "files_removed": [...], "bytes_freed": int}`.
    """
    # Imported here, not at module scope: `utils.run_io` reaches back into this module
    # through `training.run_manifest` (a circular import at import time).
    from subliminal_learning.utils.run_io import endpoint_link_problem

    checkpoints_dir = run_dir / "checkpoints"
    report: dict[str, Any] = {
        "checkpoints_removed": [], "files_removed": [], "bytes_freed": 0
    }
    for leaf in sorted(_numeric_checkpoints(checkpoints_dir), key=checkpoint_step):
        step = checkpoint_step(leaf)
        marker = read_resume_marker(leaf)
        scheduled = is_scheduled_step(
            step, save_steps=save_steps, explicit_steps=explicit_steps,
            checkpoint_resume=checkpoint_resume, periodic_active=periodic_active,
        )
        redundant_final = final_step is not None and step == final_step
        if redundant_final or (
            marker is not None and marker.get("reason") == "sigterm" and not scheduled
        ):
            # The endpoint adapter lives at `checkpoints/`; if it is a LINK into this dir
            # (or the root holds symlinks at all) the rmtree would destroy the run's final
            # weights. Skip loudly rather than free the bytes.
            link_problem = endpoint_link_problem(run_dir, [leaf])
            if link_problem is not None:
                logger.warning("{}: not pruning {} ({})", run_dir, leaf.name, link_problem)
                continue
            report["bytes_freed"] += _dir_bytes(leaf)
            shutil.rmtree(leaf, ignore_errors=True)
            report["checkpoints_removed"].append(leaf.name)
            continue
        state_files = [
            *(leaf / n for n in ("optimizer.pt", "scheduler.pt", "scaler.pt",
                                 RESUME_MARKER_NAME)),
            *leaf.glob("rng_state*.pth"),
        ]
        for path in state_files:
            if path.is_file():
                report["bytes_freed"] += path.stat().st_size
                path.unlink()
                report["files_removed"].append(f"{leaf.name}/{path.name}")
    return report


def write_interrupted_marker(run_dir: Path, *, step: int, checkpoint_dir: Path) -> None:
    """Record "SIGTERM stopped this run at `step`, resume from `checkpoint_dir`" atomically."""
    atomic_write_json(run_dir / INTERRUPTED_MARKER_NAME, {
        "step": step,
        "checkpoint_dir": str(checkpoint_dir),
        "timestamp": time.time(),
        "host": socket.gethostname(),
        "pid": os.getpid(),
    })


def clear_interrupted_marker(run_dir: Path) -> None:
    """Drop the interrupted stamp: this attempt is now the live writer."""
    (run_dir / INTERRUPTED_MARKER_NAME).unlink(missing_ok=True)


def run_is_interrupted_resumable(run_dir: Path) -> bool:
    """True when a yield left this run resumable: interrupted stamp + a marked checkpoint.

    The stamp is written after the process stopped writing, so (unlike the mtime
    heuristic) it distinguishes "interrupted, nobody is writing" from "another job's
    live run" without waiting out the idle grace.
    """
    return ((run_dir / INTERRUPTED_MARKER_NAME).is_file()
            and latest_resumable_checkpoint(run_dir / "checkpoints") is not None)


def checkpoint_resume_is_safe(
    run_dir: Path,
    identity: dict,
    *,
    save_only_model: bool = False,
    dtype: str = "float32",
    save_steps: int = 500,
) -> bool:
    """Validate the numeric latest checkpoint that HF will resume on a single GPU.

    Legacy partial runs without an input stamp restart. A truncated latest save also
    restarts; we must not approve an older checkpoint then let HF load the newer one.
    Transient IO errors propagate so the caller cannot delete state it could not read.
    Binary state is loaded by Trainer: corruption there fails loudly, never resumes
    with a fresh optimizer. The stamp includes precision without changing the legacy
    completed-run identity policy.

    `save_only_model` is accepted for call-site compatibility but no longer decides:
    since a queue yield writes FULL state whatever the run's config says, only the
    on-disk files + `write_resume_marker`'s marker can answer "is this resumable?"
    (see `checkpoint_is_resumable`).
    """
    del save_only_model
    try:
        marker = json.loads((run_dir / "resume_identity.json").read_text())
        expected = {
            "identity": _without_location_keys(identity),
            "dtype": dtype,
            "save_steps": save_steps,
        }
        if not isinstance(marker, dict):
            return False
        stored = marker.get("identity")
        if not isinstance(stored, dict):
            return False
        marker["identity"] = _without_location_keys(stored)
        if marker != expected:
            return False
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    return latest_resumable_checkpoint(run_dir / "checkpoints") is not None


def clear_stale_run(run_dir: Path, *, preserve_checkpoints: bool = False) -> None:
    """Remove a prior `train_manifest.json` + `checkpoints/` + `eval/` before retraining.

    Deleting the manifest FIRST guarantees a crash mid-retrain never leaves an old
    manifest beside new/mixed checkpoints — the torn state a later rerun under the old
    config would mistake for a finished run and skip, serving garbage.

    `eval/` records describe the adapter being overwritten, so they go too. Teacher
    dirs need this: their probe/fidelity staleness stamp (`manifest_ref`) carries no
    student_param_tag/filtered_sha256, so a same-HP retrain would otherwise leave the
    old records looking current.

    `eval/` is MOVED to `eval.superseded-<stamp>-<pid>-<hex>` rather than deleted, ALWAYS
    (2026-09-18): on a thinned run those records are the only surviving readout of the
    deleted checkpoints, and "thinned" is not detectable -- a hand-thinned run carries no
    ladder manifest, so keying the move on one silently deleted exactly the irreplaceable
    case. The retrain still starts from no eval records; the uniquified name is
    `ckpt_ladder.superseded_name`, the same spelling the record writers use, so two clears
    within one second never collide.
    """
    # A validated resume keeps trainer state, but invalidates all final-run records.
    manifest_path = run_dir / "train_manifest.json"
    checkpoint_dir = run_dir / "checkpoints"
    eval_dir = run_dir / "eval"
    if manifest_path.exists():
        manifest_path.unlink()
    if not preserve_checkpoints:
        (run_dir / "resume_identity.json").unlink(missing_ok=True)
        clear_interrupted_marker(run_dir)
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
    if eval_dir.exists():
        dest = ckpt_ladder.superseded_name(eval_dir)
        eval_dir.rename(dest)
        logger.warning("{}: retrain, eval records preserved as {}", run_dir, dest.name)
