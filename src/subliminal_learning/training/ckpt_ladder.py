"""The per-run checkpoint LADDER manifest and the one currency predicate over it.

Intermediate ``{run_dir}/checkpoints/checkpoint-N/`` dirs are being THINNED (only a few
epochs plus the endpoint adapter at the ``checkpoints/`` root survive). Every checkpoint-list
currency gate used to compare a stored record's ``checkpoint_steps`` to the freshly globbed
on-disk list by EQUALITY, so after thinning a full 41-point trajectory record would read as
stale, be recomputed over the 7 surviving checkpoints and OVERWRITE the full record. The
gates therefore compare under `steps_covered` -- "the stored record covers everything on
disk" -- and the ladder manifest written next to the checkpoints records which steps ever
existed, so a thinned point is known to be thinned rather than lost.

This lives in ``training/`` rather than beside `evaluation.ckpt_trajectories` because the
ladder describes the TRAINING run's own artifacts and every reader needs it (the dose,
fidelity, legacy-conversion and EAS gates all import it) while ``ckpt_trajectories``
imports torch; keeping the predicate torch-free keeps those CPU-only call sites light.

Schema (``{run_dir}/checkpoints/ladder.json``, ``schema_version`` 1)::

    n_train_steps:     int    -- the run's last optimizer step (endpoint adapter's step)
    epochs:            float  -- the run's epoch count
    steps_per_epoch:   float  -- n_train_steps / epochs
    checkpoint_steps:  [int]  -- every step that EVER existed on disk, endpoint included
    checkpoints:       {str(step): {kept: bool, adapter_sha256: str|None,
                                    adapter_bytes: int|None, deleted_at: iso|None}}
    keep_epochs:       [float] -- the epochs the thinning kept
    derived_docs_verified: {name: bool} -- records confirmed complete before deleting
    thinned_at:        iso|None
    thinned_by:        str|None
    train_manifest_ref: dict|None -- `utils.run_io.manifest_ref`, the identity stamp the
                       trajectory records carry, so a ladder never outlives its run
"""

from __future__ import annotations

import json
import math
import os
import secrets
from datetime import datetime
from pathlib import Path

from subliminal_learning.utils.io import atomic_write_json

SCHEMA_VERSION = 1
LADDER_NAME = "ladder.json"

# Checkpoint-list currency is "the stored record COVERS everything on disk" (2026-09-18),
# on DISTINCT steps. Equality was the older rule; it cannot survive checkpoint thinning
# (`scripts.thin_run_checkpoints`-style deletion), under which a full trajectory record
# would read as stale and be recomputed down to the surviving points. Coverage also keeps
# the older dedupe property: a record measured when the endpoint still had its duplicate
# `checkpoint-{n_train_steps}` dir stays current once that dir is gone.
# `scripts.dedupe_final_trajectory_checkpoint --check-readers` asserts this constant, so
# the script cannot run against a checkout whose gates predate the change.
STEP_GATE = "covers"


def _as_step(value) -> int | None:
    """One checkpoint step coerced to int, or None when it is not an integral number.

    Stored records are JSON written by several generations of writer, so a step can arrive
    as ``int``, as a float (``625.0``) or as a string (``"625"``). Coercion happens HERE,
    once, so no gate ever compares ``625`` with ``"625"`` and calls a covered record stale.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if math.isfinite(value) and value.is_integer() else None
    if isinstance(value, str):
        try:
            return _as_step(float(value.strip()))
        except ValueError:
            return None
    return None


#: Public spelling of the step coercion, for readers outside this module (one step at a
#: time, where `distinct_steps` would be a list round-trip).
as_step = _as_step


def distinct_steps(steps) -> list[int]:
    """The checkpoint-step list as the currency gates compare it: sorted, deduplicated.

    Every element is coerced by `_as_step`; None and non-numeric entries are SKIPPED rather
    than raised on, so a single malformed point in a stored record cannot crash a gate.
    """
    out = set()
    for s in (steps or []):
        coerced = _as_step(s)
        if coerced is not None:
            out.add(coerced)
    return sorted(out)


def steps_covered(stored_steps, disk_steps) -> bool:
    """True when every DISTINCT step on disk is already a point of the stored record.

    The one checkpoint-list predicate: a record measured over a longer ladder than the one
    now on disk (a thinned run) stays current; a record missing a step that exists on disk
    is stale. Non-finite points count as covered steps -- they are stored points.
    """
    return set(distinct_steps(disk_steps)) <= set(distinct_steps(stored_steps))


def record_is_complete(stored_steps, ladder_steps_) -> bool:
    """True when a stored record covers every step the ladder says EVER existed.

    Stronger than `steps_covered`, which only looks at what survives on disk: the thinning
    script requires completeness of each derived record before it deletes anything, so a
    record written before some checkpoints existed is never silently frozen incomplete.
    """
    return set(distinct_steps(ladder_steps_)) <= set(distinct_steps(stored_steps))


# --------------------------------------------------------------------------- manifest IO
def ladder_path(run_dir: Path) -> Path:
    """``{run_dir}/checkpoints/ladder.json``."""
    return Path(run_dir) / "checkpoints" / LADDER_NAME


def read_ladder(run_dir: Path) -> dict | None:
    """The run's ladder manifest, or None when it is absent or unreadable."""
    path = ladder_path(run_dir)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def write_ladder(run_dir: Path, doc: dict) -> Path:
    """Atomically write the run's ladder manifest; returns its path."""
    path = ladder_path(run_dir)
    atomic_write_json(path, doc)
    return path


# --------------------------------------------------------------------------- ladder steps
def ladder_steps(ckpt_root: Path) -> list[int]:
    """Every step this run's ladder EVER had: on disk, the endpoint, and the manifest's.

    The union of the on-disk ``checkpoint-*`` steps, the endpoint step from the run's
    ``train_manifest.json`` (``n_train_steps``; absent manifest -> on-disk only), and the
    ladder manifest's ``checkpoint_steps`` when one has been written. Sorted, distinct --
    so a legacy ``checkpoint-{n_train_steps}`` duplicating the endpoint counts once.
    """
    ckpt_root = Path(ckpt_root)
    steps: list[int] = []
    for d in ckpt_root.glob("checkpoint-*"):
        try:
            steps.append(int(d.name.split("-")[1]))
        except (IndexError, ValueError):
            continue
    manifest = ckpt_root.parent / "train_manifest.json"
    if manifest.is_file():
        try:
            n = json.loads(manifest.read_text()).get("n_train_steps")
        except (OSError, ValueError):
            n = None
        if isinstance(n, int) and n >= 0:
            steps.append(n)
    doc = read_ladder(ckpt_root.parent)
    if doc:
        steps.extend(doc.get("checkpoint_steps") or [])
    return distinct_steps(steps)


def steps_per_epoch(n_train_steps: int, epochs: float) -> float:
    """Optimizer steps per epoch (``n_train_steps / epochs``); 0.0 for a degenerate run."""
    return float(n_train_steps) / float(epochs) if epochs else 0.0


def keep_steps(n_train_steps: int, epochs: float, keep_epochs) -> set[int]:
    """The UNSNAPPED target steps: ``round(e * n_train_steps / epochs)`` for each epoch `e`.

    Pure arithmetic; a real ladder's rungs sit on the run's save cadence and need not land
    on these values -- `snap_keep_steps` is what a thinning must use.

    Rounding is half-up (not Python's banker's rounding), and an epoch beyond the run's
    training (step > ``n_train_steps``) is DROPPED -- an ep5 run asked to keep epoch 10
    keeps nothing at 10 rather than pinning a second copy of its endpoint.
    """
    per = steps_per_epoch(n_train_steps, epochs)
    out = {int(math.floor(e * per + 0.5)) for e in (keep_epochs or [])}
    return {s for s in out if s <= n_train_steps}


def snap_keep_steps(disk_steps, n_train_steps: int, epochs: float,
                    keep_epochs) -> tuple[dict[float, int | None], str | None]:
    """Map each FEASIBLE keep epoch to the on-disk step that realises it, plus a problem.

    `keep_steps` gives the arithmetic TARGET ``round_half_up(e * n_train_steps / epochs)``;
    a real run's checkpoints sit on its save cadence, which need not divide `n_train_steps`
    (a 25020-step run saving every 625 has no checkpoint at 1251). Each target is therefore
    SNAPPED to the nearest step actually on disk, within a tolerance of half the minimum
    positive gap between sorted disk steps -- wide enough for cadence rounding, too narrow
    to ever pick a neighbouring rung. An epoch is feasible when its target is ``<=
    n_train_steps`` and not the endpoint itself (the endpoint adapter lives at
    ``checkpoints/``, not in a ``checkpoint-N`` dir, and is never deleted).

    Returns ``({epoch: step_or_None}, problem_or_None)``; the problem names the epochs with
    no candidate and any two epochs that snapped to the SAME step -- either means the
    thinning would keep fewer rungs than asked and must not proceed silently.
    """
    disk = distinct_steps(disk_steps)
    tol = min((b - a for a, b in zip(disk, disk[1:]) if b > a), default=0) / 2
    per = steps_per_epoch(n_train_steps, epochs)
    snapped: dict[float, int | None] = {}
    for e in sorted({float(x) for x in (keep_epochs or [])}):
        target = int(math.floor(e * per + 0.5))
        if target > n_train_steps or target == n_train_steps:
            continue
        cand = min(disk, key=lambda s: (abs(s - target), s), default=None)
        snapped[e] = cand if cand is not None and abs(cand - target) <= tol else None
    problems = []
    missing = [e for e, s in snapped.items() if s is None]
    if missing:
        problems.append("no on-disk checkpoint within tolerance for epoch(s) "
                        + ",".join(f"{e:g}" for e in missing))
    hit: dict[int, list[float]] = {}
    for e, s in snapped.items():
        if s is not None:
            hit.setdefault(s, []).append(e)
    collided = {s: es for s, es in hit.items() if len(es) > 1}
    if collided:
        problems.append("epochs snapped to the same step: "
                        + "; ".join(f"{s}<-{','.join(f'{e:g}' for e in es)}"
                                    for s, es in sorted(collided.items())))
    if len(hit) != len(snapped):
        problems.append(f"{len(hit)} snapped step(s) for {len(snapped)} feasible epoch(s)")
    return snapped, ("; ".join(problems) if problems else None)


def record_points_problem(stored: dict, ladder_steps_, *, kind: str = "generic",
                          animal: str | None = None) -> str | None:
    """Why a stored per-checkpoint record is NOT a complete readout of the ladder, or None.

    The thinning script's precondition: a derived record may only be frozen (its
    checkpoints deleted) when it really holds a measured point per ladder step. The header
    alone is not enough -- a record can declare the full ``checkpoint_steps`` and carry an
    empty ``points`` list -- so this checks containment of the LADDER in the header AND at
    least one point per ladder step, matched on the coerced ``step``.

    A point's payload is keyed by PROMPT SET (`pref_traj`: ``test``/``off_topic``) or is the
    flat Pile readout (``kl_base_adapted``, ``delta_ce``, ...), so "measured" is "carries
    something besides ``step``/``checkpoint``": a non-empty dict under some other key, or
    the ``nonfinite`` flag of a checkpoint that was never scorable. DUPLICATE steps are
    legitimate -- the endpoint is stored both from the legacy ``checkpoint-{n_train_steps}``
    dir and from the adapter at ``checkpoints/`` -- and are never a problem.

    With `animal` given the record must list it under ``animals``, and every scored point
    must hold that animal's measurement in some prompt set: directly (a single-animal
    record's score dict, keyed ``p_target``) or nested under the animal (multi-animal).

    `kind` picks the per-point completeness rule, which for the two trajectory records is
    the EVALUATOR's own: "carries something" is too weak to freeze a record on, because the
    trajectory writers themselves treat such a point as stale and recompute it -- and after
    thinning that recompute can no longer reach the checkpoint.

      - ``pile_kl_traj``: `point_has_ce` (the currency gate's rule, ONE copy) AND a finite
        ``kl_base_adapted``; a ``nonfinite`` point is exempt.
      - ``pref_traj``: `animal` is REQUIRED and the point must carry a finite ``p_target``
        for it in some prompt set (or be ``nonfinite``).
      - ``generic``: the "carries something besides the step" rule.

    Returns a short reason (``steps_not_covered``, ``no_points``, ``missing_point:<step>``,
    ``unmeasured_point:<step>``, ``incomplete_point:<step>``, ``animal_not_in_record``,
    ``no_animal_measurement:<step>``).
    """
    if kind not in ("generic", "pile_kl_traj", "pref_traj"):
        raise ValueError(f"unknown record kind {kind!r}")
    if kind == "pref_traj" and animal is None:
        raise ValueError("record_points_problem(kind='pref_traj') requires an animal")
    stored = stored or {}
    wanted = distinct_steps(ladder_steps_)
    if not record_is_complete(stored.get("checkpoint_steps"), wanted):
        return "steps_not_covered"
    points = stored.get("points") or []
    if not points:
        return "no_points"
    if animal is not None and animal not in (stored.get("animals") or []):
        return "animal_not_in_record"
    by_step: dict[int, list[dict]] = {}
    for p in points:
        s = _as_step((p or {}).get("step"))
        if s is not None:
            by_step.setdefault(s, []).append(p or {})
    for step in wanted:
        got = by_step.get(step) or []
        if not got:
            return f"missing_point:{step}"
        if kind != "generic":
            if not any(_KIND_COMPLETE[kind](p, animal) for p in got):
                return f"incomplete_point:{step}"
            continue
        if not any(_point_is_measured(p) for p in got):
            return f"unmeasured_point:{step}"
        if animal is not None and not any(
                p.get("nonfinite") or _point_measures_animal(p, animal) for p in got):
            return f"no_animal_measurement:{step}"
    return None


_POINT_META = ("step", "checkpoint", "nonfinite")


def _is_finite(value) -> bool:
    """True for a real, finite JSON number (``True`` is not a measurement)."""
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def point_has_ce(point: dict) -> bool:
    """True when one measured point carries a finite ``delta_ce`` (a ``nonfinite`` point is
    exempt: it was never scored).

    THE delta_ce rule, in one place: `evaluation.ckpt_trajectories` re-exports this as its
    currency/partial-resume predicate, so "complete enough to freeze the checkpoints" and
    "complete enough to skip the recompute" can never drift apart.
    """
    return bool(point.get("nonfinite")) or _is_finite(point.get("delta_ce"))


def _pile_point_is_complete(point: dict, animal: str | None = None) -> bool:
    """A Pile-KL point the evaluator would not recompute: CE readout plus its KL."""
    return bool(point.get("nonfinite")) or (
        point_has_ce(point) and _is_finite(point.get("kl_base_adapted")))


def _pref_point_is_complete(point: dict, animal: str | None = None) -> bool:
    """A trait point carrying a FINITE ``p_target`` for `animal` in some prompt set."""
    if point.get("nonfinite"):
        return True
    for key, value in point.items():
        if key in _POINT_META or not isinstance(value, dict):
            continue
        if _is_finite(value.get("p_target")):
            return True
        nested = value.get(animal)
        if isinstance(nested, dict) and _is_finite(nested.get("p_target")):
            return True
    return False


_KIND_COMPLETE = {"pile_kl_traj": _pile_point_is_complete,
                  "pref_traj": _pref_point_is_complete}


def _point_is_measured(point: dict) -> bool:
    """True when a trajectory point holds a readout (or is a recorded non-finite adapter)."""
    if point.get("nonfinite"):
        return True
    # `pref_traj` payloads are per-prompt-set DICTS (an empty one is no measurement);
    # `pile_kl_traj` payloads are scalars (`kl_base_adapted`, `delta_ce`, ...).
    return any(k not in _POINT_META and v is not None and v != {}
               for k, v in point.items())


def _point_measures_animal(point: dict, animal: str) -> bool:
    """True when some prompt set of `point` carries `animal`'s score dict.

    Single-animal records put the score dict straight under the prompt set (the record's
    own animal); multi-animal records nest one score dict per animal under it.
    """
    for key, value in point.items():
        if key in _POINT_META or not isinstance(value, dict):
            continue
        if "p_target" in value:
            return True
        nested = value.get(animal)
        if isinstance(nested, dict) and "p_target" in nested:
            return True
    return False


def run_epochs(run_dir: Path) -> float | None:
    """The run's epoch count: ``hyperparameters.n_epochs`` of its ``train_manifest.json``.

    Prefers the structured field; falls back to the ``ep{N}`` of ``student_param_tag`` for
    the older manifests that carry no hyperparameter block. None when neither is available.
    """
    manifest = Path(run_dir) / "train_manifest.json"
    if not manifest.is_file():
        return None
    try:
        m = json.loads(manifest.read_text())
    except (OSError, ValueError):
        return None
    n = (m.get("hyperparameters") or {}).get("n_epochs")
    if n:
        return float(n)
    from subliminal_learning.param_tags import parse_param_tag

    try:
        return float(parse_param_tag(m.get("student_param_tag") or "")["epochs"])
    except (ValueError, KeyError):
        return None


# --------------------------------------------------------------------------- supersede
def superseded_name(path: Path) -> Path:
    """A free ``{name}.superseded-{stamp}-{pid}-{hex}`` sibling of `path` (never clobbers).

    Public because every preserving move uses this ONE spelling: the record writers here
    and `run_gate.clear_stale_run`'s retrain move of a whole ``eval/`` dir (a dir, not a
    file -- the pattern is the same).

    The stamp carries the pid and four random hex digits because two workers can reach the
    same run dir within the same second, and a supersede that OVERWROTE another supersede
    would lose exactly the record it exists to preserve. The suffix is appended to the full
    file name, so ``pref_traj.json`` becomes ``pref_traj.json.superseded-...`` and no reader
    globbing ``*.json`` picks it up as a live record.
    """
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    while True:
        dest = path.with_name(
            f"{path.name}.superseded-{stamp}-{os.getpid()}-{secrets.token_hex(2)}")
        if not dest.exists():
            return dest


def supersede_if_shrinking(run_dir: Path, out_path: Path, stored_steps, new_steps, *,
                           extra_paths=(), log=None) -> list[Path]:
    """Move a stored record aside when the record about to be written covers FEWER steps.

    THE invariant (2026-09-18): no writer may ever replace a stored per-checkpoint record
    with one covering fewer steps. Checkpoint thinning makes this easy to violate -- a
    ``--force`` rerun, an identity change or a `--final-only` pass recomputes over whatever
    survives on disk and would overwrite a full 41-point trajectory with 7 points. So
    whenever the new step set does not COVER the stored one, `out_path` (and every existing
    `extra_paths` sidecar, which describes the same measurement) is renamed aside.

    UNCONDITIONAL by design: it does not require a ladder manifest (hand-thinned runs carry
    none) and has no force exemption (force means "recompute", never "discard the fuller
    record"). Returns the paths moved aside, in the order given.
    """
    if set(distinct_steps(stored_steps)) <= set(distinct_steps(new_steps)):
        return []
    lost = len(set(distinct_steps(stored_steps)) - set(distinct_steps(new_steps)))
    moved = []
    for path in (Path(out_path), *(Path(p) for p in extra_paths)):
        if path.exists():
            dest = superseded_name(path)
            path.rename(dest)
            moved.append(dest)
    if moved and log is not None:
        log.warning(
            "{}: rewrite would cover {} fewer checkpoint step(s) than the stored record; "
            "preserved as {}", run_dir, lost, ", ".join(p.name for p in moved))
    return moved
