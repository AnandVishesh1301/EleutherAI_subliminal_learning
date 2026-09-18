"""CLI: remove the partial artifacts a KILLED job leaves, so a requeue re-runs cleanly.

Run by hand / on demand as a sweeper. It targets EXACTLY seven in-scope pipelines and
NOTHING else:

  teacher       pipelines/train_teacher_templates_pipeline.py
  numbers       pipelines/generate_numbers_pipeline.py
  eval          pipelines/evaluate_teacher_preference_pipeline.py  (teachers + students)
  distill       pipelines/distill_student_pipeline.py
  steerability  pipelines/steerability_screen_pipeline.py
  eas           pipelines/eas_pipeline.py
  coherence     pipelines/pile_coherence_pipeline.py

EXPLICITLY out of scope (never scanned, never touched): SDF teacher (``sdf_teachers/``)
and on-policy GKD (``gkd_students/``). A ``--path`` in one of those trees is skipped.

Also out of scope by construction: animals archived to the cold tier, which are symlinks
under ``teachers/``. This scanner uses plain ``rglob``/``os.walk``, neither of which
follows directory symlinks, so archived trees are invisible to it. That is DELIBERATE --
do NOT "fix" it with ``rglob_follow``/``followlinks=True``: this tool deletes, and the
archive is the only remaining copy of that data. Read-only consumers that must see
archived animals use ``utils.io.rglob_follow`` instead.

Safety model (a bug here destroys months of GPU time):
- Dry-run is the DEFAULT; deleting requires ``--apply``.
- A training leaf is only ever purged when it lives in a training tree (``teachers/`` or
  ``students/``) AND holds a ``checkpoints/`` dir with NO final weight file
  (mirrors ``utils/run_io.checkpoint_is_complete``). The "missing manifest alone" test is
  deliberately NOT used: 4872 ``numbers/`` / ``numbers_snapshots/`` pool dirs legitimately
  have no manifest and no ``checkpoints/`` -- those trees are never scanned for leaves.
- ``--min-age-s`` (scan mode, default 6h) skips any candidate whose mtime is newer than the
  guard; a live job creates its leaf up front and may not touch it for the whole run, so the
  guard must exceed the longest expected job (~1h). Targeted (``--path``) mode skips the age
  guard but refuses a complete dir.
- **Liveness guard (every mode).** Before purging a leaf, its ``/proc`` cmdlines are scanned
  (our own namespace only -- ``nvidia-smi`` pids are host-namespace and useless); a leaf whose
  seed AND cell identifiers appear in a live command is skipped. Multi-seed cells run their
  seeds as separate concurrent jobs, so a sibling seed's leaf is legitimately incomplete while
  it runs -- deleting it would be data loss.
- **Targeted mode needs an explicit seed selection** for a multi-seed cell: ``--seeds``, a
  parseable ``--from-job-cmd`` / ``$JOBQ_JOB_CMD``, or the guarded ``--all-seeds``. With no
  parseable selection it deletes nothing and exits non-zero.
- Every deletion target is resolved and verified under an allowed root (``output_root``, plus
  ``dataset_cache_root`` ONLY when passed via ``--dataset-cache-root``, for partial Pile
  shards); symlinked targets are refused.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import typer
import yaml
from dacite import Config, from_dict
from dotenv import load_dotenv
from loguru import logger

from subliminal_learning.config import PathsConfig
from subliminal_learning.run_gate import latest_resumable_checkpoint
from subliminal_learning.utils.logging_setup import configure_loguru
from subliminal_learning.utils.run_io import (
    checkpoint_is_complete,
    find_files_parallel,
    find_leaf_dirs_parallel,
)

ALL_STAGES = (
    "teacher",
    "numbers",
    "eval",
    "distill",
    "steerability",
    "eas",
    "coherence",
)
# Top-level trees this tool must never scan or delete inside.
OUT_OF_SCOPE_TREES = ("sdf_teachers", "gkd_students")
# Recognized training-leaf directory-name prefixes (the safety gate for any leaf rmtree).
_LEAF_PREFIXES = ("ftseed", "gseed")
# Depth bound for the leaf walk. The ``rglob`` it replaced was unbounded; the deepest real
# leaf (a resampling student) sits ~10 below ``students/``, so this only keeps the walk finite.
_LEAF_WALK_MAX_DEPTH = 24
# Depth bound for the FILE walks (tmp orphans, aggregate.json, coherence artifacts). Files sit
# BELOW a leaf (``ftseed1/checkpoints/checkpoint-40/x.tmp.7``), so this must exceed
# ``_LEAF_WALK_MAX_DEPTH`` by the artifact nesting; 32 is well past the deepest real path.
_FILE_WALK_MAX_DEPTH = 32
# Flags whose value(s) name a fine-tuning seed (NOT --teacher-ftseed / --subsample-seed).
_SEED_FLAGS = (
    "--student-seeds",
    "--student-seed",
    "--ftseeds",
    "--ftseed",
    "--seeds",
    "--seed",
)
# Structural path segments that never identify a specific cell (excluded from liveness tokens).
_STOP_TOKENS = {
    "teachers", "students", "controls", "base", "rng", "numbers",
    "numbers_snapshots", "checkpoints", "coherence", "eas", "steerability", "ckpt",
}


class TargetedRefusal(Exception):
    """Raised when targeted mode cannot resolve a seed selection and so deletes nothing."""


# --------------------------------------------------------------------------- seed / proc


def _leaf_seed(name: str) -> int | None:
    for p in _LEAF_PREFIXES:
        if name.startswith(p):
            try:
                return int(name[len(p):])
            except ValueError:
                return None
    return None


def parse_seeds_from_cmd(cmd: str) -> set[int]:
    """Fine-tuning seeds a killed job command targets.

    Handles both repo shapes: a ``for S in 1 2 3; do ... --student-seed $S ...; done`` loop
    (the list is the seed set) and any ``--student-seed``/``--seed``/``--ftseed`` flag (single
    value or a comma list). ``$S``-style values are skipped -- the loop already supplied them.
    Returns an empty set when nothing parses (the caller then refuses in targeted mode).
    """
    seeds: set[int] = set()
    for m in re.finditer(r"for\s+\w+\s+in\s+([0-9][0-9\s]*?)\s*;?\s*do", cmd):
        for tok in m.group(1).split():
            if tok.isdigit():
                seeds.add(int(tok))
    toks = cmd.split()
    for i, tok in enumerate(toks):
        key = tok.split("=", 1)[0]
        if key not in _SEED_FLAGS:
            continue
        val = tok.split("=", 1)[1] if "=" in tok else (toks[i + 1] if i + 1 < len(toks) else "")
        for part in val.split(","):
            part = part.strip()
            if part.isdigit():
                seeds.add(int(part))
    return seeds


# This module's own invocations are excluded from the live set: a targeted `--path <leaf>`
# call (and its `bash -c` hook wrapper) carries the target dir in its own argv, which would
# otherwise self-match the liveness guard and make it refuse to clean its own target.
_SELF_MARKER = "clean_partial_runs"


def list_live_cmdlines(proc_root: str = "/proc") -> list[str]:
    """Space-joined ``/proc/<pid>/cmdline`` for every readable process in OUR namespace.

    Only our-uid processes are readable (others raise and are skipped); host-namespace pids
    from nvidia-smi are deliberately NOT used (they resolve to unrelated processes here). This
    tool's own processes (and the hook's ``bash -c`` wrapper) are filtered out via
    ``_SELF_MARKER`` so a ``--path`` argument never self-matches the liveness guard.
    """
    out: list[str] = []
    try:
        entries = list(Path(proc_root).iterdir())
    except OSError:
        return out
    for pdir in entries:
        if not pdir.name.isdigit():
            continue
        try:
            raw = (pdir / "cmdline").read_bytes()
        except OSError:
            continue
        if not raw:
            continue
        cmd = raw.replace(b"\x00", b" ").decode("utf-8", "replace").strip()
        if _SELF_MARKER in cmd:
            continue
        out.append(cmd)
    return out


# --------------------------------------------------------------------------- actions


@dataclass(frozen=True)
class Action:
    """One planned filesystem change (nothing is done until ``apply_actions``)."""

    kind: str  # rmtree_leaf | rm_file | rmtree_shard
    path: Path
    stage: str
    reason: str
    size: int  # bytes reclaimed


def _tree_size(p: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(p):
        for f in files:
            try:
                total += (Path(root) / f).lstat().st_size
            except OSError:
                pass
    return total


# --------------------------------------------------------------------------- plan


@dataclass(frozen=True)
class CleanPlan:
    """Resolved configuration for one scan/targeted run (pure of typer)."""

    output_root: Path
    allowed_roots: tuple[Path, ...]  # resolved absolute roots deletions must stay under
    dataset_cache_root: Path | None
    dataset_cache_allowed: bool
    min_age_s: float
    now: float
    targeted: bool
    coherence_orphan_npy: bool
    live_cmdlines: tuple[str, ...] = ()
    walk_workers: int = 8  # threads for the leaf walk; low on purpose, the nodes run GPU jobs
    # Directories the discovery walk could not read, as ``(path, error class)``. A walk that
    # errored returned a TRUNCATED view of the tree, and "absent" is exactly what this tool
    # reads as "partial, delete it" -- so ANY entry here disables deletion for the whole
    # invocation (the report is still produced).
    walk_errors: list = field(default_factory=list)

    # ---- safety ----

    @property
    def deletions_refused(self) -> bool:
        """Whether a walk error makes every deletion in this invocation unsafe."""
        return bool(self.walk_errors)

    def is_safe_target(self, path: Path) -> bool:
        """True iff ``path`` is not a symlink and resolves under an allowed root."""
        try:
            if path.is_symlink():
                return False
            rp = path.resolve()
        except OSError:
            return False
        for root in self.allowed_roots:
            try:
                rp.relative_to(root)
                return True
            except ValueError:
                continue
        return False

    def age_ok(self, path: Path) -> bool:
        """Age guard: targeted mode -> always True; scan mode -> require mtime old enough."""
        if self.targeted:
            return True
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return False
        return (self.now - mtime) >= self.min_age_s

    def mk(self, kind: str, path: Path, stage: str, reason: str) -> Action | None:
        """Build a safety-checked Action, or None (with a warning) if the target is unsafe."""
        if not self.is_safe_target(path):
            logger.warning("Refusing unsafe target (symlink or outside allowed roots): {}", path)
            return None
        if kind in ("rmtree_leaf", "rmtree_shard"):
            size = _tree_size(path)
        elif kind == "rewrite_drop_last":
            size = 0  # filled by the caller (bytes of the dropped line)
        else:
            try:
                size = path.stat().st_size
            except OSError:
                size = 0
        return Action(kind=kind, path=path, stage=stage, reason=reason, size=size)


# --------------------------------------------------------------------------- leaf logic


def _is_leaf_name(name: str) -> bool:
    return any(name.startswith(p) for p in _LEAF_PREFIXES)


def _has_resumable_checkpoints(ckpt: Path) -> bool:
    """ANY ``checkpoint-*`` dir, or a marked full-state one -- never purge either.

    A marked, complete dir (``run_gate.latest_resumable_checkpoint``) is what an
    interrupted run restarts from. A bare weights-only ``checkpoint-*`` buys no resume,
    but it is still the visible progress of a TRAJECTORY run -- possibly one live on
    ANOTHER node, which this node's ``/proc`` liveness scan cannot see -- so it shields
    the leaf too: deleting a mid-flight trajectory run costs hours of GPU time, while
    keeping a truly abandoned one costs only a later sweep.
    """
    try:
        if any(p.is_dir() for p in ckpt.glob("checkpoint-*")):
            return True
        return latest_resumable_checkpoint(ckpt) is not None
    except OSError:
        return False


def _cell_tokens(plan: CleanPlan, leaf: Path) -> set[str]:
    """Path-segment tokens (animal / model / tags) that identify ``leaf``'s cell in a command."""
    try:
        rel = leaf.resolve().relative_to(plan.output_root.resolve())
    except (ValueError, OSError):
        return set()
    toks: set[str] = set()
    for part in rel.parts[:-1]:  # ancestors, excluding the leaf itself
        # Exclude the two student lineage segments (teacher-{base} and {tag}_tseed{T}): they
        # key the teacher a student distilled from, not the student's own cell in a command.
        is_lineage = part.startswith("teacher-") or "_tseed" in part
        if len(part) >= 3 and part not in _STOP_TOKENS and not is_lineage:
            toks.add(part)
    return toks


def _cmd_word_set(cmd: str) -> set[str]:
    return {w for w in re.split(r"[\s=,/]+", cmd) if w}


def _leaf_is_live(plan: CleanPlan, leaf: Path) -> bool:
    """Whether a live process in our namespace could be writing ``leaf`` (its seed + cell match).

    Two signals, both erring toward "live" (skip): the leaf's cell-dir path appears verbatim in
    a command (jobs that pass an explicit dir), OR the leaf's seed is in a command's parsed seed
    set AND one of the leaf's cell tokens (e.g. the animal) is a word in that command.
    """
    seed = _leaf_seed(leaf.name)
    cell_str = str(leaf.parent)
    leaf_str = str(leaf)
    tokens = _cell_tokens(plan, leaf)
    for cmd in plan.live_cmdlines:
        if leaf_str in cmd or cell_str in cmd:
            return True
        if seed is None:
            continue
        if seed in parse_seeds_from_cmd(cmd) and tokens & _cmd_word_set(cmd):
            return True
    return False


def _leaf_newest_mtime(leaf: Path) -> float:
    """Newest mtime of ``leaf`` or of ANYTHING under it (recursive, symlinks not followed).

    The leaf dir's OWN mtime is not a progress signal: adding ``checkpoints/checkpoint-N``
    never touches it, so a trajectory run that has been writing for hours on another node
    still looks untouched since its creation. Walking the tree is the only cross-node
    evidence of a live writer. Symlinks are not followed, as everywhere else in this
    deleting tool (see the module docstring): an archived animal must stay invisible.
    """
    try:
        newest = leaf.stat().st_mtime
    except OSError:
        return 0.0
    for dirpath, dirnames, filenames in os.walk(leaf, followlinks=False):
        for name in (*dirnames, *filenames):
            try:
                newest = max(newest, os.lstat(os.path.join(dirpath, name)).st_mtime)
            except OSError:
                continue
    return newest


def _targeted_recent_age_s(plan: CleanPlan, leaf: Path) -> float | None:
    """Seconds since the newest write under ``leaf`` if it is too recent for targeted mode.

    Targeted mode's liveness guard reads THIS node's /proc only, but the output root is
    shared: a leaf mid-training on another node looks dead here. Recent writes are the
    only cross-node signal we have, so refuse on them (override with --min-age-s 0).
    """
    if not plan.targeted or plan.min_age_s <= 0:
        return None
    age = plan.now - _leaf_newest_mtime(leaf)
    return age if age < plan.min_age_s else None


def _classify_leaf(plan: CleanPlan, leaf: Path) -> str:
    """One of: remove | no-checkpoints | complete | resumable | live | too-new |
    recently-modified."""
    ckpt = leaf / "checkpoints"
    if not ckpt.is_dir():
        return "no-checkpoints"
    if checkpoint_is_complete(ckpt):
        return "complete"
    if _has_resumable_checkpoints(ckpt):
        return "resumable"
    if _leaf_is_live(plan, leaf):
        return "live"
    recent = _targeted_recent_age_s(plan, leaf)
    if recent is not None:
        logger.info(
            "Refusing {}: recently modified {:.0f} s ago; another node may be writing it; "
            "pass --min-age-s 0 to override.", leaf, recent,
        )
        return "recently-modified"
    # NOT `plan.age_ok(leaf)`: the leaf dir's own mtime does not move when checkpoints
    # are added under it, so the guard reads the newest write anywhere in the tree.
    if not plan.targeted and (plan.now - _leaf_newest_mtime(leaf)) < plan.min_age_s:
        return "too-new"
    return "remove"


def _log_skip(path: Path, reason: str) -> None:
    if reason == "recently-modified":
        return  # already logged with the age, at _classify_leaf
    if reason in ("live", "too-new", "resumable", "not-selected"):
        logger.info("Skip leaf {} ({})", path, reason)
    else:  # complete / no-checkpoints -> expected, keep quiet unless debugging
        logger.debug("Skip leaf {} ({})", path, reason)


def _leaf_candidate(plan: CleanPlan, leaf: Path, stage: str) -> Action | None:
    cls = _classify_leaf(plan, leaf)
    if cls == "remove":
        return plan.mk("rmtree_leaf", leaf, stage, "incomplete checkpoint (no final weight file)")
    _log_skip(leaf, cls)
    return None


def _scan_leaves(
    plan: CleanPlan, base: Path, stage: str, selected: set[int] | None = None
) -> list[Action]:
    """Purge candidates among ``ftseed*`` leaves under ``base`` (``selected`` restricts seeds).

    Discovery is the shared thread-parallel walk (`utils.run_io.find_leaf_dirs_parallel`),
    which -- like the ``rglob`` it replaced -- never follows directory symlinks, so an
    archived animal stays invisible to this deleting tool (see the module docstring).
    """
    out: list[Action] = []
    if not base.is_dir():
        return out
    errors: list = []
    leaves = find_leaf_dirs_parallel(
        [base],
        lambda p: _is_leaf_name(p.name),
        max_depth=_LEAF_WALK_MAX_DEPTH,
        workers=plan.walk_workers,
        errors=errors,
    )
    if errors:
        plan.walk_errors.extend(errors)
        print(
            f"WARNING: {len(errors)} director{'y' if len(errors) == 1 else 'ies'} could not be "
            f"walked under {base}; the scan is TRUNCATED and this run will delete NOTHING. "
            f"First: {'; '.join(f'{path} ({err})' for path, err in errors[:5])}",
            file=sys.stderr, flush=True,
        )
    _warn_nested_leaves(leaves)
    for d in leaves:
        if selected is not None:
            seed = _leaf_seed(d.name)
            if seed is None or seed not in selected:
                _log_skip(d, "not-selected")
                continue
        a = _leaf_candidate(plan, d, stage)
        if a is not None:
            out.append(a)
    return out


def _warn_nested_leaves(leaves: list[Path]) -> None:
    """Warn when a found leaf itself contains a leaf-named child dir (one level, no descent).

    The ``rglob("ftseed*")`` this walk replaced reported ``ftseed*`` dirs nested under other
    ``ftseed*`` dirs; the walk stops at the first one, so such a nest is now invisible. It
    should not exist -- if it does, say so loudly rather than silently skip it.
    """
    nested: list[Path] = []
    for leaf in leaves:
        try:
            with os.scandir(leaf) as it:
                for entry in it:
                    if _is_leaf_name(entry.name) and entry.is_dir(follow_symlinks=False):
                        nested.append(Path(entry.path))
        except OSError:
            continue
    if nested:
        print(
            f"WARNING: {len(nested)} leaf-named dir(s) nested inside a leaf; the walk stops at "
            f"the outer leaf and does NOT consider them. First: "
            f"{'; '.join(str(p) for p in nested[:5])}",
            file=sys.stderr, flush=True,
        )


def _scan_tmp(plan: CleanPlan, base: Path, name_globs: tuple[str, ...], stage: str) -> list[Action]:
    """Orphan ``*.tmp.<pid>`` files matching one of ``name_globs`` anywhere under ``base``.

    ``checkpoint-*`` dirs are NOT pruned: an interrupted checkpoint save leaves its temp file
    exactly there.
    """
    out: list[Action] = []
    if not base.is_dir():
        return out
    for glob in name_globs:
        for f in find_files_parallel(
            [base], glob, max_depth=_FILE_WALK_MAX_DEPTH, workers=plan.walk_workers
        ):
            if not f.is_file() or not plan.age_ok(f):
                continue
            a = plan.mk("rm_file", f, stage, "orphan temp file from an interrupted atomic write")
            if a is not None:
                out.append(a)
    return out


# --------------------------------------------------------------------------- per-stage scans


def _scan_numbers(plan: CleanPlan, bases: list[Path]) -> list[Action]:
    out: list[Action] = []
    for base in bases:
        out += _scan_tmp(plan, base, ("*.tmp.*",), "numbers")  # tmp ONLY -- never a leaf
    return out


def _scan_eval(plan: CleanPlan, bases: list[Path]) -> list[Action]:
    globs = (
        "preference_eval.json.tmp.*",
        "preference_summary.json.tmp.*",
        "base_control.json.tmp.*",
    )
    out: list[Action] = []
    for base in bases:
        out += _scan_tmp(plan, base, globs, "eval")
    return out


def _aggregate_is_partial(agg: Path) -> bool:
    """Whether a steerability ``aggregate.json`` is a valid-but-PARTIAL (or corrupt) mean.

    ``compute_aggregate`` seed-averages whatever ``ft*.json`` are present and stamps
    ``n_seeds``; a kill can leave an aggregate over a subset that a consumer then trusts. It is
    partial iff its recorded ``n_seeds`` disagrees with the ``ft*.json`` count now on disk (or it
    is unreadable). A complete aggregate (``n_seeds`` == present seeds) is left untouched -- it is
    valid data, and deleting every aggregate in a broad scan would violate "never delete complete
    data".
    """
    try:
        doc = json.loads(agg.read_text())
    except (OSError, json.JSONDecodeError):
        return True
    n_seeds = doc.get("n_seeds")
    if not isinstance(n_seeds, int):
        return True
    present = sum(1 for _ in agg.parent.glob("ft*.json"))
    return n_seeds != present


def _scan_steerability(plan: CleanPlan, base: Path) -> list[Action]:
    """Delete only PARTIAL/corrupt ``aggregate.json`` (re-derived) + steerability tmp files.

    Per-seed ``ft{seed}.json`` are atomic + content-gated and resume correctly -- KEPT. A
    complete aggregate (seed count matches the ft*.json on disk) is KEPT (valid data).
    """
    out: list[Action] = []
    if not base.is_dir():
        return out
    # A steerability aggregate lives under ``{cell}/steerability/``, never inside a
    # ``checkpoint-*`` dir, so that subtree is pruned.
    for agg in find_files_parallel(
        [base], "aggregate.json", skip_prefixes=("checkpoint-",),
        max_depth=_FILE_WALK_MAX_DEPTH, workers=plan.walk_workers,
    ):
        if "steerability" not in agg.parts or not agg.is_file() or not plan.age_ok(agg):
            continue
        if not _aggregate_is_partial(agg):
            logger.debug("Keep complete aggregate {}", agg)
            continue
        a = plan.mk(
            "rm_file", agg, "steerability",
            "partial/corrupt seed-averaged aggregate.json (re-derived from ft*.json)",
        )
        if a is not None:
            out.append(a)
    for tmp in find_files_parallel(
        [base], "*.tmp.*", max_depth=_FILE_WALK_MAX_DEPTH, workers=plan.walk_workers
    ):
        if "steerability" not in tmp.parts or not tmp.is_file() or not plan.age_ok(tmp):
            continue
        a = plan.mk("rm_file", tmp, "steerability", "orphan temp file")
        if a is not None:
            out.append(a)
    return out


def _scan_eas(plan: CleanPlan, roots: list[Path]) -> list[Action]:
    """EAS documents/vectors are atomic + hash-gated (KEPT); orphan temp files only.

    Covers every location the per-run EAS design writes (`pipelines.eas_pipeline`): the
    student run dirs' ``eval/eas_*.json`` + ``eval/eas_vectors_*.pt``, the teacher-side
    ``eval/eas_vectors/`` vector cache, the ``analysis/eas_cache/`` base means -- and the
    RETIRED per-lineage ``{lineage}/eas/`` documents, which are left on disk untouched but
    can still hold a temp orphan from an interrupted write.
    """
    out: list[Action] = []
    for base in roots:
        if not base.is_dir():
            continue
        # Per-run documents + sidecars (`eval/`), and the retired lineage docs (`eas/`).
        out += _scan_tmp(
            plan, base,
            ("eas_*.json.tmp.*", "eas_vectors_*.pt.tmp.*", "v_teacher_*.pt.tmp.*"), "eas",
        )
        # The teacher-side vector cache and the base-mean cache name their files by prompt
        # sha, so the DIRECTORY is what identifies them.
        # ``eval/eas_vectors`` / the base-mean cache: dirs, never inside a ``checkpoint-*``.
        for d in find_leaf_dirs_parallel(
            [base], lambda p: p.name == "eas_vectors", skip_prefixes=("checkpoint-",),
            max_depth=_LEAF_WALK_MAX_DEPTH, workers=plan.walk_workers,
        ):
            if d.is_dir():
                out += _scan_tmp(plan, d, ("*.tmp.*",), "eas")
        if base.name == "eas_cache":
            out += _scan_tmp(plan, base, ("*.tmp.*",), "eas")
    return out


def _scan_coherence_artifact_dir(
    plan: CleanPlan, d: Path, npy_to_json, json_to_npy, npy_glob: str, json_glob: str
) -> list[Action]:
    """Orphan/tmp cleanup for one ``pile_coherence/`` (run) or ``base/{model}`` dir.

    Always removes ``*.tmp.*`` orphans (interrupted atomic writes). With
    ``--coherence-orphan-npy``, also removes a per-token-loss npy whose sibling record
    JSON is missing and a record JSON whose npy is missing (a broken half-written pair).
    ``npy_to_json``/``json_to_npy`` map between the paired files (naming differs between
    run dirs and the base tree).
    """
    out: list[Action] = _scan_tmp(plan, d, ("*.tmp.*",), "coherence")
    if not plan.coherence_orphan_npy:
        return out
    for npy in sorted(d.glob(npy_glob)):
        if not npy.is_file() or not plan.age_ok(npy):
            continue
        if not npy_to_json(npy).exists():
            a = plan.mk(
                "rm_file", npy, "coherence",
                "orphan per_token_loss.npy (no sibling record json)",
            )
            if a is not None:
                out.append(a)
    for jsn in sorted(d.glob(json_glob)):
        if not jsn.is_file() or not plan.age_ok(jsn):
            continue
        if not json_to_npy(jsn).exists():
            a = plan.mk(
                "rm_file", jsn, "coherence",
                "coherence record json missing its per_token_loss.npy",
            )
            if a is not None:
                out.append(a)
    return out


def _scan_coherence(plan: CleanPlan) -> list[Action]:
    """Scan the per-run ``pile_coherence/`` dirs and the shared ``coherence/{shard}/base`` tree.

    Adapter runs (teachers/, students/) keep artifacts at
    ``{run_dir}/pile_coherence/{version}.json`` (+ ``{version}_per_token_loss.npy``);
    base runs at ``{output_root}/coherence/{version}/base/{model}/ce.json`` (+
    ``per_token_loss.npy``). The legacy flat ``results.jsonl`` + ``runs/`` tree is retired.
    """
    out: list[Action] = []

    # (a) per-run pile_coherence/ dirs under teachers/ and students/.
    def _run_npy_to_json(npy: Path) -> Path:
        stem = npy.name[: -len("_per_token_loss.npy")]
        return npy.with_name(f"{stem}.json")

    def _run_json_to_npy(jsn: Path) -> Path:
        return jsn.with_name(f"{jsn.stem}_per_token_loss.npy")

    for tree in ("teachers", "students"):
        base = plan.output_root / tree
        if not base.is_dir():
            continue
        # ``{run_dir}/pile_coherence``: a dir, never inside a ``checkpoint-*``.
        for d in find_leaf_dirs_parallel(
            [base], lambda p: p.name == "pile_coherence", skip_prefixes=("checkpoint-",),
            max_depth=_LEAF_WALK_MAX_DEPTH, workers=plan.walk_workers,
        ):
            if not d.is_dir():
                continue
            out += _scan_coherence_artifact_dir(
                plan, d, _run_npy_to_json, _run_json_to_npy,
                "*_per_token_loss.npy", "*.json",
            )

    # (b) the shared coherence/{version}/base/{model_dir_name} tree.
    def _base_npy_to_json(npy: Path) -> Path:
        return npy.with_name("ce.json")

    def _base_json_to_npy(jsn: Path) -> Path:
        return jsn.with_name("per_token_loss.npy")

    cbase = plan.output_root / "coherence"
    if cbase.is_dir():
        for base_tree in find_leaf_dirs_parallel(
            [cbase],
            lambda p: p.name == "base" and p.parent.parent == cbase,
            max_depth=2, workers=plan.walk_workers,
        ):
            if not base_tree.is_dir():
                continue
            for model_dir in sorted(base_tree.iterdir()):
                if not model_dir.is_dir():
                    continue
                out += _scan_coherence_artifact_dir(
                    plan, model_dir, _base_npy_to_json, _base_json_to_npy,
                    "per_token_loss.npy", "ce.json",
                )

    out += _scan_coherence_shards(plan)
    return out


def _scan_coherence_shards(plan: CleanPlan) -> list[Action]:
    """Partial Pile shard dirs (``header.json`` absent) under ``dataset_cache_root``.

    Only produced when ``dataset_cache_root`` was explicitly allowed. A shard WITH a
    ``header.json`` is a complete, expensive artifact and is never removed.
    """
    out: list[Action] = []
    if not (plan.dataset_cache_allowed and plan.dataset_cache_root is not None):
        return out
    sbase = plan.dataset_cache_root / "pile_coherence"
    if not sbase.is_dir():
        return out
    for version_dir in sorted(sbase.iterdir()):
        if not version_dir.is_dir():
            continue
        if (version_dir / "header.json").exists():
            out += _scan_tmp(plan, version_dir, ("header.json.tmp.*",), "coherence")
            continue
        has_payload = (version_dir / "tokens.npy").exists() or (
            version_dir / "doc_ids.npy"
        ).exists()
        if has_payload and plan.age_ok(version_dir):
            a = plan.mk(
                "rmtree_shard", version_dir, "coherence",
                "partial shard build (tokens/doc_ids written, header.json missing)",
            )
            if a is not None:
                out.append(a)
    return out


# --------------------------------------------------------------------------- planners


def plan_scan(plan: CleanPlan, stages: tuple[str, ...]) -> list[Action]:
    """Sweep every in-scope tree for stale partials (age- and liveness-guarded)."""
    out: list[Action] = []
    root = plan.output_root
    if "teacher" in stages:
        out += _scan_leaves(plan, root / "teachers", "teacher")
        out += _scan_tmp(plan, root / "teachers", ("train_manifest.json.tmp.*",), "teacher")
    if "distill" in stages:
        out += _scan_leaves(plan, root / "students", "distill")
        out += _scan_tmp(
            plan, root / "students",
            ("train_manifest.json.tmp.*", "student_manifest.json.tmp.*"), "distill",
        )
    if "numbers" in stages:
        out += _scan_numbers(plan, [root / "numbers", root / "numbers_snapshots"])
    if "eval" in stages:
        out += _scan_eval(plan, [root / "teachers", root / "students"])
    if "steerability" in stages:
        out += _scan_steerability(plan, root / "teachers")
    if "eas" in stages:
        out += _scan_eas(plan, [root / "students", root / "teachers",
                              root / "analysis" / "eas_cache"])
    if "coherence" in stages:
        out += _scan_coherence(plan)
    return _dedupe(out)


def detect_stage(path: Path, plan: CleanPlan) -> str | None:
    """Which in-scope stage a ``--path`` belongs to, or None (out of scope / outside roots)."""
    try:
        rp = path.resolve()
    except OSError:
        return None
    if plan.dataset_cache_root is not None:
        try:
            rp.relative_to((plan.dataset_cache_root / "pile_coherence").resolve())
            return "coherence"
        except ValueError:
            pass
    try:
        rel = rp.relative_to(plan.output_root.resolve())
    except ValueError:
        return None
    parts = rel.parts
    if not parts:
        return None
    if parts[0] in OUT_OF_SCOPE_TREES:
        return None
    if parts[0] == "coherence":
        return "coherence"
    if parts[0] in ("numbers", "numbers_snapshots"):
        return "numbers"
    if "steerability" in parts:
        return "steerability"
    if {"eas", "eas_vectors", "eas_cache"} & set(parts):
        return "eas"
    if parts[0] == "teachers":
        return "teacher"
    if parts[0] == "students":
        return "distill"
    return None


def _resolve_selection(
    seeds_opt: set[int] | None, from_job_cmd: str | None, all_seeds: bool
) -> set[int] | None:
    """Seed set for a multi-seed cell, or None for 'all seeds' (``--all-seeds``).

    Raises TargetedRefusal when no explicit/parseable selection is available -- the tool must
    never fall back to "all seeds" implicitly (a live sibling seed would be deleted).
    """
    if all_seeds:
        return None
    if seeds_opt:
        return set(seeds_opt)
    if from_job_cmd:
        parsed = parse_seeds_from_cmd(from_job_cmd)
        if parsed:
            return parsed
    raise TargetedRefusal(
        "no seed selection for this cell: pass --seeds, a parseable --from-job-cmd / "
        "$JOBQ_JOB_CMD, or --all-seeds (unsafe while sibling seeds run)."
    )


def plan_targeted(
    plan: CleanPlan,
    path: Path,
    stages: tuple[str, ...],
    *,
    seeds_opt: set[int] | None = None,
    from_job_cmd: str | None = None,
    all_seeds: bool = False,
) -> list[Action]:
    """Clean one dead run dir / cell (complete, live + recently-modified leaves refused).

    For a teacher/distill CELL dir a seed selection is REQUIRED (``--seeds`` /
    ``--from-job-cmd`` / ``--all-seeds``); a bare ``ftseed*`` leaf path selects its own seed.
    """
    stage = detect_stage(path, plan)
    if stage is None:
        logger.info(
            "Skip {}: not in any in-scope tree (out of scope: {} / outside output_root).",
            path, ", ".join(OUT_OF_SCOPE_TREES),
        )
        return []
    if stages and stage not in stages:
        logger.info("Skip {}: detected stage {!r} excluded by --stage.", path, stage)
        return []
    out: list[Action] = []
    if stage in ("teacher", "distill"):
        tmp_globs = (
            ("train_manifest.json.tmp.*",)
            if stage == "teacher"
            else ("train_manifest.json.tmp.*", "student_manifest.json.tmp.*")
        )
        if _is_leaf_name(path.name):
            seed = _leaf_seed(path.name)
            if seeds_opt is not None and seed is not None and seed not in seeds_opt:
                _log_skip(path, "not-selected")
            else:
                cls = _classify_leaf(plan, path)
                if cls == "remove":
                    a = plan.mk(
                        "rmtree_leaf", path, stage,
                        "incomplete checkpoint (no final weight file)",
                    )
                    if a is not None:
                        out.append(a)
                elif cls == "complete":
                    logger.info("Refusing {}: leaf checkpoint is complete (not a partial).", path)
                else:
                    _log_skip(path, cls)
        else:
            selected = _resolve_selection(seeds_opt, from_job_cmd, all_seeds)
            out += _scan_leaves(plan, path, stage, selected)
        out += _scan_tmp(plan, path, tmp_globs, stage)
    elif stage == "numbers":
        out += _scan_numbers(plan, [path])
    elif stage == "eval":
        out += _scan_eval(plan, [path])
    elif stage == "steerability":
        out += _scan_steerability(plan, path)
    elif stage == "eas":
        out += _scan_eas(plan, [path])
    elif stage == "coherence":
        out += _scan_coherence(plan)
    return _dedupe(out)


def _dedupe(actions: list[Action]) -> list[Action]:
    seen: set[Path] = set()
    out: list[Action] = []
    for a in actions:
        if a.path in seen:
            continue
        seen.add(a.path)
        out.append(a)
    return out


# --------------------------------------------------------------------------- apply / report


def apply_actions(actions: list[Action], plan: CleanPlan) -> tuple[int, int]:
    """Execute the plan (re-checking safety on each target). Returns (count, bytes)."""
    import shutil

    if plan.deletions_refused:
        logger.error(
            "Refusing every deletion: {} director(y/ies) could not be walked, so the scan is "
            "truncated and a missing artifact cannot be distinguished from a partial one.",
            len(plan.walk_errors),
        )
        return 0, 0
    count = 0
    total = 0
    for a in actions:
        if not plan.is_safe_target(a.path):
            logger.warning("Skip unsafe target at apply time: {}", a.path)
            continue
        try:
            if a.kind in ("rmtree_leaf", "rmtree_shard"):
                shutil.rmtree(a.path)
            elif a.kind == "rm_file":
                a.path.unlink()
            count += 1
            total += a.size
        except OSError as e:
            logger.warning("Failed to clean {}: {}", a.path, e)
    return count, total


def _report(actions: list[Action], applied: bool) -> None:
    verb = "Removed" if applied else "Would remove"
    if not actions:
        print("Nothing to clean.")
        return
    by_stage: dict[str, int] = {}
    total = 0
    for a in actions:
        by_stage[a.stage] = by_stage.get(a.stage, 0) + 1
        total += a.size
        print(f"  [{a.stage:12s}] {a.kind:16s} {a.path}  ({a.size} B)  # {a.reason}")
    stages = ", ".join(f"{k}={v}" for k, v in sorted(by_stage.items()))
    print(f"{verb} {len(actions)} target(s), {total} B reclaimed. By stage: {stages}")


# --------------------------------------------------------------------------- CLI


def _resolve_paths(paths_config: Path, output_root: Path | None) -> PathsConfig:
    data = yaml.safe_load(paths_config.read_text())
    paths = from_dict(PathsConfig, data, config=Config(cast=[Path]))
    if output_root is not None:
        from dataclasses import replace

        paths = replace(paths, output_root=output_root)
    return paths


def _parse_seed_opt(seeds: str | None) -> set[int] | None:
    if not seeds:
        return None
    out: set[int] = set()
    for part in seeds.replace(",", " ").split():
        if part.isdigit():
            out.add(int(part))
    return out or None


def main(
    path: Path = typer.Option(
        None,
        "--path",
        help="Targeted mode: clean this one dead run dir / cell (a complete, live or "
        "recently-modified leaf is refused). The stage is detected from the path.",
    ),
    scan: bool = typer.Option(
        False, "--scan", help="Scan mode: sweep the in-scope trees for stale partials."
    ),
    stage: list[str] = typer.Option(
        None,
        "--stage",
        help=f"Restrict to these stages (repeatable). One of: {', '.join(ALL_STAGES)}.",
    ),
    seeds: str = typer.Option(
        None,
        "--seeds",
        help="Targeted mode: restrict a cell to these fine-tuning seeds (comma/space list).",
    ),
    from_job_cmd: str = typer.Option(
        None,
        "--from-job-cmd",
        help="Targeted mode: parse the killed job's seed(s) from this command "
        "(defaults to $JOBQ_JOB_CMD). Understands --student-seed N and the "
        "'for S in 1 2 3; do ... $S ...; done' idiom.",
    ),
    all_seeds: bool = typer.Option(
        False,
        "--all-seeds",
        help="Targeted mode: clean EVERY incomplete leaf in the cell. UNSAFE while sibling "
        "seeds are still running (relies on the liveness guard); prefer --seeds/--from-job-cmd.",
    ),
    apply: bool = typer.Option(
        False, "--apply", help="Actually delete. DEFAULT is a dry run that deletes nothing."
    ),
    min_age_s: float = typer.Option(
        None,
        "--min-age-s",
        help="Skip candidates whose mtime is newer than this. Scan mode default 21600 (6h): a "
        "live training job may not touch its dir for the whole run, so this must exceed the "
        "longest expected job (~1h). Targeted (--path) mode default 1800: the output root is "
        "shared across nodes, so recent writes mean another node may still be running it; "
        "pass 0 to override.",
    ),
    coherence_orphan_npy: bool = typer.Option(
        False,
        "--coherence-orphan-npy",
        help="Also remove half-written coherence artifact pairs (a per_token_loss.npy with no "
        "sibling record json, or a record json with no npy) (inert; tidying only, opt-in).",
    ),
    output_root: Path = typer.Option(
        None, "--output-root", help="Override paths.output_root (else read from --paths-config)."
    ),
    dataset_cache_root: Path = typer.Option(
        None,
        "--dataset-cache-root",
        help="Explicitly ALLOW deletions under this root (only for partial Pile shards under "
        "pile_coherence/). Shard cleanup is refused unless this is passed.",
    ),
    paths_config: Path = typer.Option(
        Path("configs/paths/paths_local.yaml"), "--paths-config", help="Paths YAML."
    ),
    walk_workers: int = typer.Option(
        8, "--walk-workers", help="Threads for the discovery walk; keep low, the nodes run "
        "GPU jobs."
    ),
) -> None:
    """Find and (with --apply) remove partial artifacts left by killed jobs."""
    load_dotenv()
    configure_loguru()
    if scan == (path is not None):
        raise typer.BadParameter("choose exactly one of --scan or --path.")
    stages = tuple(stage) if stage else ALL_STAGES
    unknown = set(stages) - set(ALL_STAGES)
    if unknown:
        raise typer.BadParameter(f"unknown --stage {sorted(unknown)}; known: {ALL_STAGES}")

    paths = _resolve_paths(paths_config, output_root)
    allowed = [paths.output_root.resolve()]
    ds_allowed = dataset_cache_root is not None
    ds_root = dataset_cache_root if ds_allowed else paths.dataset_cache_root
    if ds_allowed:
        allowed.append(dataset_cache_root.resolve())

    import time

    plan = CleanPlan(
        output_root=paths.output_root,
        allowed_roots=tuple(allowed),
        dataset_cache_root=ds_root,
        dataset_cache_allowed=ds_allowed,
        min_age_s=(min_age_s if min_age_s is not None else (1800.0 if path is not None
                                                            else 21600.0)),
        now=time.time(),
        targeted=path is not None,
        coherence_orphan_npy=coherence_orphan_npy,
        live_cmdlines=tuple(list_live_cmdlines()),
        walk_workers=walk_workers,
    )
    try:
        if plan.targeted:
            job_cmd = from_job_cmd if from_job_cmd is not None else os.environ.get("JOBQ_JOB_CMD")
            actions = plan_targeted(
                plan, path, tuple(stages) if stage else (),
                seeds_opt=_parse_seed_opt(seeds),
                from_job_cmd=job_cmd,
                all_seeds=all_seeds,
            )
        else:
            actions = plan_scan(plan, stages)
    except TargetedRefusal as e:
        logger.error("Targeted cleanup refused: {}", e)
        raise typer.Exit(code=2) from None

    if apply and plan.deletions_refused:
        _report(actions, applied=False)
        logger.error(
            "Walk errors ({}) make this scan untrustworthy: reporting only, deleting nothing. "
            "Re-run once the filesystem is healthy.", len(plan.walk_errors),
        )
        raise typer.Exit(code=3)
    if apply:
        count, total = apply_actions(actions, plan)
        _report(actions, applied=True)
        logger.info("Cleaned {} target(s), {} B.", count, total)
    else:
        _report(actions, applied=False)
        print("(dry run -- pass --apply to delete)")


if __name__ == "__main__":
    typer.run(main)
