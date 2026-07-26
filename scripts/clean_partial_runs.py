"""CLI: remove the partial artifacts a KILLED job leaves, so a requeue re-runs cleanly.

Wired as the jobq ``on_yield_kill_cmd`` hook (see
``src/subliminal_learning/jobqueue/yielding.py`` / ``worker.py``) and also run by hand.
It targets EXACTLY seven in-scope pipelines and NOTHING else:

  teacher       pipelines/train_teacher_templates_pipeline.py
  numbers       pipelines/generate_numbers_pipeline.py
  eval          pipelines/evaluate_teacher_preference_pipeline.py  (teachers + students)
  distill       pipelines/distill_student_pipeline.py
  steerability  pipelines/steerability_screen_pipeline.py
  eas           pipelines/eas_pipeline.py
  coherence     pipelines/pile_coherence_pipeline.py

EXPLICITLY out of scope (never scanned, never touched): SDF teacher (``sdf_teachers/``)
and on-policy GKD (``gkd_students/``). A ``--path`` in one of those trees is skipped.

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
from dataclasses import dataclass
from pathlib import Path

import typer
import yaml
from dacite import Config, from_dict
from dotenv import load_dotenv
from loguru import logger

from subliminal_learning.config import PathsConfig
from subliminal_learning.utils.io import atomic_write_text
from subliminal_learning.utils.logging_setup import configure_loguru
from subliminal_learning.utils.run_io import checkpoint_is_complete

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
    "numbers_snapshots", "checkpoints", "coherence", "eas", "steerability",
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

    kind: str  # rmtree_leaf | rm_file | rewrite_drop_last | rmtree_shard
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

    # ---- safety ----

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
    """A ``checkpoint-*`` subdir means a resumable mid-run save -- never purge it."""
    try:
        return any(c.is_dir() and c.name.startswith("checkpoint-") for c in ckpt.iterdir())
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
        if len(part) >= 3 and part not in _STOP_TOKENS and not part.startswith("from_"):
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


def _classify_leaf(plan: CleanPlan, leaf: Path) -> str:
    """One of: remove | no-checkpoints | complete | resumable | live | too-new."""
    ckpt = leaf / "checkpoints"
    if not ckpt.is_dir():
        return "no-checkpoints"
    if checkpoint_is_complete(ckpt):
        return "complete"
    if _has_resumable_checkpoints(ckpt):
        return "resumable"
    if _leaf_is_live(plan, leaf):
        return "live"
    if not plan.age_ok(leaf):
        return "too-new"
    return "remove"


def _log_skip(path: Path, reason: str) -> None:
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
    """Purge candidates among ``ftseed*`` leaves under ``base`` (``selected`` restricts seeds)."""
    out: list[Action] = []
    if not base.is_dir():
        return out
    for d in sorted(base.rglob("ftseed*")):
        if not d.is_dir():
            continue
        if selected is not None:
            seed = _leaf_seed(d.name)
            if seed is None or seed not in selected:
                _log_skip(d, "not-selected")
                continue
        a = _leaf_candidate(plan, d, stage)
        if a is not None:
            out.append(a)
    return out


def _scan_tmp(plan: CleanPlan, base: Path, name_globs: tuple[str, ...], stage: str) -> list[Action]:
    """Orphan ``*.tmp.<pid>`` files matching one of ``name_globs`` anywhere under ``base``."""
    out: list[Action] = []
    if not base.is_dir():
        return out
    for glob in name_globs:
        for f in sorted(base.rglob(glob)):
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
    for agg in sorted(base.rglob("aggregate.json")):
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
    for tmp in sorted(base.rglob("*.tmp.*")):
        if "steerability" not in tmp.parts or not tmp.is_file() or not plan.age_ok(tmp):
            continue
        a = plan.mk("rm_file", tmp, "steerability", "orphan temp file")
        if a is not None:
            out.append(a)
    return out


def _scan_eas(plan: CleanPlan, base: Path) -> list[Action]:
    """EAS ``eas_*.json`` / ``v_teacher_*.pt`` are atomic + hash-gated (KEPT); tmp orphans only."""
    out: list[Action] = []
    if not base.is_dir():
        return out
    for edir in sorted(base.rglob("eas")):
        if not edir.is_dir():
            continue
        for glob in ("eas_*.json.tmp.*", "v_teacher_*.pt.tmp.*"):
            for f in sorted(edir.glob(glob)):
                if not f.is_file() or not plan.age_ok(f):
                    continue
                a = plan.mk("rm_file", f, "eas", "orphan temp file")
                if a is not None:
                    out.append(a)
    return out


def _referenced_npy(results_path: Path) -> set[str]:
    """Relative ``per_token_loss_path`` values named by valid ``ce`` records in results.jsonl."""
    refs: set[str] = set()
    if not results_path.exists():
        return refs
    for line in results_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue  # truncated/torn line -> ignore (handled separately)
        p = rec.get("per_token_loss_path")
        if p:
            refs.add(str(p))
    return refs


def _truncated_last_line(results_path: Path) -> tuple[str, int] | None:
    """``(new_content, dropped_bytes)`` when the LAST non-empty line fails to JSON-parse, else None.

    Every earlier record is preserved byte-for-byte; only a trailing torn append is dropped.
    """
    raw = results_path.read_text()
    lines = raw.splitlines(keepends=True)
    last_idx = None
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip():
            last_idx = i
            break
    if last_idx is None:
        return None
    try:
        json.loads(lines[last_idx].strip())
        return None  # last record parses -> valid history, leave it entirely alone
    except json.JSONDecodeError:
        new_content = "".join(lines[:last_idx])
        dropped = len(raw.encode()) - len(new_content.encode())
        return new_content, dropped


def _scan_coherence(plan: CleanPlan) -> list[Action]:
    out: list[Action] = []
    cbase = plan.output_root / "coherence"
    if cbase.is_dir():
        for version_dir in sorted(cbase.iterdir()):
            if not version_dir.is_dir():
                continue
            results = version_dir / "results.jsonl"
            if results.exists() and plan.age_ok(results):
                trunc = _truncated_last_line(results)
                if trunc is not None:
                    a = plan.mk(
                        "rewrite_drop_last", results, "coherence",
                        "drop a truncated (unparseable) trailing results.jsonl line",
                    )
                    if a is not None:
                        out.append(Action(a.kind, a.path, a.stage, a.reason, trunc[1]))
            if plan.coherence_orphan_npy:
                refs = _referenced_npy(results)
                runs = version_dir / "runs"
                if runs.is_dir():
                    for npy in sorted(runs.rglob("per_token_loss.npy")):
                        rel = str(npy.relative_to(version_dir))
                        if rel in refs or not plan.age_ok(npy):
                            continue
                        a = plan.mk(
                            "rm_file", npy, "coherence",
                            "orphan per_token_loss.npy (no matching results.jsonl record)",
                        )
                        if a is not None:
                            out.append(a)
            out += _scan_tmp(plan, version_dir, ("*.tmp.*",), "coherence")
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
        out += _scan_eas(plan, root / "students")
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
    if "eas" in parts:
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
    """Clean one dead run dir / cell (age guard off; complete + live leaves refused).

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
        out += _scan_eas(plan, path)
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
            elif a.kind == "rewrite_drop_last":
                trunc = _truncated_last_line(a.path)
                if trunc is None:
                    continue  # re-check: last line now parses -> leave it
                atomic_write_text(a.path, trunc[0])
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
        help="Targeted mode: clean this one dead run dir / cell (age guard OFF; a complete or "
        "live leaf is refused). The stage is detected from the path.",
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
        21600.0,
        "--min-age-s",
        help="Scan mode: skip candidates whose leaf/file mtime is newer than this (default 6h). "
        "A live training job may not touch its dir for the whole run, so this must exceed the "
        "longest expected job (~1h). Ignored in targeted (--path) mode.",
    ),
    coherence_orphan_npy: bool = typer.Option(
        False,
        "--coherence-orphan-npy",
        help="Also remove orphan coherence per_token_loss.npy files with no results.jsonl record "
        "(inert; tidying only, opt-in).",
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
        min_age_s=min_age_s,
        now=time.time(),
        targeted=path is not None,
        coherence_orphan_npy=coherence_orphan_npy,
        live_cmdlines=tuple(list_live_cmdlines()),
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

    if apply:
        count, total = apply_actions(actions, plan)
        _report(actions, applied=True)
        logger.info("Cleaned {} target(s), {} B.", count, total)
    else:
        _report(actions, applied=False)
        print("(dry run -- pass --apply to delete)")


if __name__ == "__main__":
    typer.run(main)
