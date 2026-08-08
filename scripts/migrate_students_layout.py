"""CLI: reorganize an existing ``students/`` outputs tree from the OLD lineage layout to
the NEW ``teacher-*`` layout, safely and reversibly.

This is a one-shot, idempotent migration. It is deliberately SELF-CONTAINED (stdlib +
typer only, no ``subliminal_learning`` import) because the live package is being changed
in parallel; the on-disk move must not depend on code that is mid-refactor.

OLD -> NEW::

    treatment:
      students/{animal}/{model}/from_{ttag}_tseed{T}[_nbase-{X}]/{ndt}/{stag}/ftseed{S}/...
        -> students/{animal}/{model}/teacher-{X or model}/{ttag}_tseed{T}/{ndt}/{stag}/ftseed{S}/...
    base control:
      students/controls/base/{model}/{ndt}[_nbase-{X}]/{stag}/ftseed{S}/...
        -> students/controls/base/{model}/teacher-{X or model}/{ndt}/{stag}/ftseed{S}/...
    rng control (students/controls/rng/...): NEVER touched.

The migration unit is ONE ``os.rename`` per old lineage dir (treatment) or per old ndt dir
(base control): the ``teacher-*`` parent is created, then the whole subtree moves atomically
(same filesystem is guaranteed). After each rename, stale absolute paths stored in ``*.json``
files under the moved subtree (manifest ``checkpoint_dir`` fields, eas ``run_dir`` fields, ...)
are rewritten by plain substring replacement of the old dir's absolute path with the new one.

Safety model:
- Dry run is the DEFAULT; ``--apply`` performs the moves. ``--verify`` audits the tree.
- Dirs already in new form (``teacher-*``) are skipped -> reruns are no-ops.
- A dir at a lineage / ndt level whose name matches neither old nor new form is skipped and
  REPORTED (never fails the run).
- If a rename target already exists, the move is refused and recorded as a conflict; the run
  continues and exits non-zero at the end.
- Every rename is written to a JSONL log sufficient to reverse it.

Examples::

    uv run python -m scripts.migrate_students_layout --students-root /path/to/students
    uv run python -m scripts.migrate_students_layout --students-root /path/to/students --apply
    uv run python -m scripts.migrate_students_layout --students-root /path/to/students --verify
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import typer

# Old treatment lineage dir name: from_{tag}_tseed{seed}[_nbase-{X}].
OLD_LINEAGE_RE = re.compile(
    r"^from_(?P<tag>ep\d+_lr[^_]+_r\d+a\d+)_tseed(?P<seed>\d+)(?:_nbase-(?P<nbase>.+))?$"
)
# ndt names contain underscores, so a base-control name is split on the LITERAL substring only.
NBASE_SEP = "_nbase-"
# Dirs already in the new layout start with this; they are skipped (idempotent reruns).
NEW_PREFIX = "teacher-"


# --------------------------------------------------------------------------- plan model


@dataclass(frozen=True)
class Move:
    """One planned migration: rename ``old`` -> ``new`` (the whole subtree moves)."""

    old: Path
    new: Path
    kind: str  # "treatment" | "base_control"


@dataclass
class Plan:
    """Everything the planner found: moves to perform + skipped/unrecognized entries."""

    moves: list[Move] = field(default_factory=list)
    skips: list[tuple[Path, str]] = field(default_factory=list)  # (path, reason)


def _sorted_dirs(parent: Path) -> list[Path]:
    """Sub-DIRS of ``parent``, name-sorted (missing/unreadable parent -> empty)."""
    try:
        return sorted((p for p in parent.iterdir() if p.is_dir()), key=lambda p: p.name)
    except OSError:
        return []


def _sorted_children(parent: Path) -> list[Path]:
    """ALL children of ``parent`` (dirs and files), name-sorted."""
    try:
        return sorted(parent.iterdir(), key=lambda p: p.name)
    except OSError:
        return []


def _plan_treatment(students_root: Path, plan: Plan) -> None:
    """Plan renames for every treatment lineage dir under ``students/{animal}/{model}/``."""
    for animal_dir in _sorted_dirs(students_root):
        if animal_dir.name == "controls":
            continue
        for model_dir in _sorted_dirs(animal_dir):
            for lineage in _sorted_children(model_dir):
                if not lineage.is_dir():
                    plan.skips.append((lineage, "not-a-dir"))
                    continue
                if lineage.name.startswith(NEW_PREFIX):
                    continue  # already migrated
                m = OLD_LINEAGE_RE.match(lineage.name)
                if m is None:
                    plan.skips.append((lineage, "unrecognized-treatment-name"))
                    continue
                nbase = m.group("nbase")
                base = nbase if nbase else model_dir.name
                new = model_dir / f"{NEW_PREFIX}{base}" / f"{m.group('tag')}_tseed{m.group('seed')}"
                plan.moves.append(Move(lineage, new, "treatment"))


def _plan_base_controls(students_root: Path, plan: Plan) -> None:
    """Plan renames for every base-control ndt dir under ``students/controls/base/{model}/``."""
    base_root = students_root / "controls" / "base"
    if not base_root.is_dir():
        return
    for model_dir in _sorted_dirs(base_root):
        for ndt in _sorted_children(model_dir):
            if not ndt.is_dir():
                plan.skips.append((ndt, "not-a-dir"))
                continue
            if ndt.name.startswith(NEW_PREFIX):
                continue  # already migrated
            parts = ndt.name.split(NBASE_SEP, 1)  # LITERAL split -- ndt names contain "_"
            ndt_name = parts[0]
            base = parts[1] if len(parts) == 2 else model_dir.name
            new = model_dir / f"{NEW_PREFIX}{base}" / ndt_name
            plan.moves.append(Move(ndt, new, "base_control"))


def plan_migration(students_root: Path) -> Plan:
    """Scan ``students_root`` and return the full migration plan (controls/rng is untouched)."""
    students_root = students_root.resolve()
    plan = Plan()
    _plan_treatment(students_root, plan)
    _plan_base_controls(students_root, plan)
    return plan


# --------------------------------------------------------------------------- json rewrite


def _rewrite_json_paths(old_dir: Path, new_dir: Path) -> int:
    """Rewrite stored absolute paths in ``*.json`` under ``new_dir`` (old prefix -> new prefix).

    Plain substring replacement of the moved dir's absolute path; write atomically
    (``<file>.tmp.<pid>`` then ``os.replace``). Returns the count of files changed.
    """
    old_str = str(old_dir)
    new_str = str(new_dir)
    changed = 0
    for jf in new_dir.rglob("*.json"):
        if not jf.is_file():
            continue
        try:
            text = jf.read_text()
        except (OSError, UnicodeDecodeError):
            continue
        if old_str not in text:
            continue
        tmp = jf.with_name(f"{jf.name}.tmp.{os.getpid()}")
        tmp.write_text(text.replace(old_str, new_str))
        os.replace(tmp, jf)
        changed += 1
    return changed


# --------------------------------------------------------------------------- migration run


@dataclass
class MigrationResult:
    """Outcome of a plan/apply pass."""

    plan: Plan
    planned: list[Move] = field(default_factory=list)  # non-conflicting moves
    done: list[Move] = field(default_factory=list)  # actually renamed (apply only)
    conflicts: list[Move] = field(default_factory=list)
    json_rewritten: int = 0
    records: list[dict] = field(default_factory=list)
    log_path: Path | None = None


def run_migration(
    students_root: Path, *, apply: bool, log_path: Path | None = None
) -> MigrationResult:
    """Plan (and with ``apply`` perform) the migration; streams the JSONL log only on apply.

    On apply the log is opened BEFORE the move loop and each record is written+flushed the
    instant its action completes, so a mid-apply crash still leaves every finished rename
    recorded (and therefore reversible). Dry runs write no log.
    """
    students_root = students_root.resolve()
    plan = plan_migration(students_root)
    result = MigrationResult(plan=plan)

    fh = None
    if apply:
        if log_path is None:
            log_path = _default_log_path(students_root)
        fh = log_path.open("w")
        result.log_path = log_path

    def _emit(rec: dict) -> None:
        result.records.append(rec)
        if fh is not None:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    try:
        for move in plan.moves:
            if move.new.exists():
                result.conflicts.append(move)
                _emit({"action": "conflict", "old": str(move.old), "new": str(move.new)})
                continue
            result.planned.append(move)
            if not apply:
                result.records.append(
                    {"action": "plan-rename", "old": str(move.old), "new": str(move.new)}
                )
                continue
            move.new.parent.mkdir(parents=True, exist_ok=True)
            os.rename(move.old, move.new)
            n = _rewrite_json_paths(move.old, move.new)
            result.json_rewritten += n
            result.done.append(move)
            _emit(
                {
                    "action": "rename",
                    "old": str(move.old),
                    "new": str(move.new),
                    "json_rewritten": n,
                }
            )

        for path, reason in plan.skips:
            _emit({"action": "skip", "path": str(path), "reason": reason})
    finally:
        if fh is not None:
            fh.close()
    return result


def _default_log_path(students_root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return students_root / f"_migration_layout_{stamp}.jsonl"


# --------------------------------------------------------------------------- verify


@dataclass
class VerifyResult:
    """Audit of the tree after (or independent of) a migration."""

    from_dirs: list[Path] = field(default_factory=list)  # (a) leftover treatment old-form dirs
    nbase_dirs: list[Path] = field(default_factory=list)  # (b) leftover *_nbase-* dir names
    ftseed_count: int = 0  # (c) ftseed* leaf dirs (informational)
    stale_json: list[Path] = field(default_factory=list)  # (d) json with /from_ or _nbase-

    @property
    def ok(self) -> bool:
        return not (self.from_dirs or self.nbase_dirs or self.stale_json)


def _is_under(path: Path, ancestor: Path) -> bool:
    try:
        path.relative_to(ancestor)
        return True
    except ValueError:
        return False


def verify_tree(students_root: Path) -> VerifyResult:
    """Report leftover old-form artifacts. Exit-worthy: (a) from_* dirs, (b) *_nbase-* dirs, (d)."""
    students_root = students_root.resolve()
    rng_root = students_root / "controls" / "rng"
    vr = VerifyResult()

    # (a) remaining from_* dirs at treatment lineage level.
    for animal_dir in _sorted_dirs(students_root):
        if animal_dir.name == "controls":
            continue
        for model_dir in _sorted_dirs(animal_dir):
            for lineage in _sorted_dirs(model_dir):
                if lineage.name.startswith("from_"):
                    vr.from_dirs.append(lineage)

    # (b) *_nbase-* dir names anywhere under students/ except controls/rng, and (c) ftseed* leaves.
    for root, dirs, _files in os.walk(students_root):
        root_path = Path(root)
        for d in dirs:
            dpath = root_path / d
            if NBASE_SEP in d and not _is_under(dpath, rng_root):
                vr.nbase_dirs.append(dpath)
            if d.startswith("ftseed"):
                vr.ftseed_count += 1

    # (d) json files still storing /from_ or _nbase- substrings.
    for jf in students_root.rglob("*.json"):
        if not jf.is_file():
            continue
        try:
            text = jf.read_text()
        except (OSError, UnicodeDecodeError):
            continue
        if "/from_" in text or NBASE_SEP in text:
            vr.stale_json.append(jf)

    return vr


# --------------------------------------------------------------------------- reporting


def _print_migration_summary(result: MigrationResult, *, apply: bool) -> None:
    verb = "renames done" if apply else "renames planned"
    n_moves = len(result.done) if apply else len(result.planned)
    print("=== migrate_students_layout summary ===")
    print(f"  {verb:16s}: {n_moves}")
    print(f"  conflicts       : {len(result.conflicts)}")
    print(f"  skips           : {len(result.plan.skips)}")
    print(f"  json rewrites   : {result.json_rewritten}")
    if result.log_path is not None:
        print(f"  log             : {result.log_path}")
    for move in (result.done if apply else result.planned):
        print(f"    [{move.kind:12s}] {move.old}  ->  {move.new}")
    for move in result.conflicts:
        print(f"    [CONFLICT] target exists, not moved: {move.new}")
    for path, reason in result.plan.skips:
        print(f"    [SKIP {reason}] {path}")
    if not apply:
        print("(dry run -- pass --apply to perform the moves)")


def _print_verify_report(vr: VerifyResult) -> None:
    print("=== verify ===")
    print(f"  leftover from_* dirs      : {len(vr.from_dirs)}")
    print(f"  leftover *_nbase-* dirs   : {len(vr.nbase_dirs)}")
    print(f"  ftseed* leaf dirs (total) : {vr.ftseed_count}")
    print(f"  json with /from_ | _nbase-: {len(vr.stale_json)}")
    for p in vr.from_dirs:
        print(f"    [from_] {p}")
    for p in vr.nbase_dirs:
        print(f"    [_nbase-] {p}")
    for p in vr.stale_json:
        print(f"    [stale-json] {p}")
    print(f"  verify {'OK' if vr.ok else 'FAILED'}")


# --------------------------------------------------------------------------- CLI


def main(
    students_root: Path = typer.Option(
        ..., "--students-root", help="Path to the students/ directory itself (the tree to migrate)."
    ),
    apply: bool = typer.Option(
        False, "--apply", help="Perform the moves. DEFAULT is a dry run that changes nothing."
    ),
    log: Path = typer.Option(
        None,
        "--log",
        help="JSONL log path (apply only). Default: "
        "<students-root>/_migration_layout_<UTCtimestamp>.jsonl.",
    ),
    verify: bool = typer.Option(
        False, "--verify", help="Audit the tree for leftover old-form artifacts (alone or +apply)."
    ),
) -> None:
    """Migrate a students/ tree from the old lineage layout to the new teacher-* layout."""
    students_root = students_root.resolve()
    if not students_root.is_dir():
        raise typer.BadParameter(f"--students-root is not a directory: {students_root}")

    exit_code = 0

    if not (verify and not apply):
        result = run_migration(students_root, apply=apply, log_path=log)
        _print_migration_summary(result, apply=apply)
        if result.conflicts:
            exit_code = 1

    if verify:
        vr = verify_tree(students_root)
        _print_verify_report(vr)
        if not vr.ok:
            exit_code = 1

    raise typer.Exit(exit_code)


if __name__ == "__main__":
    typer.run(main)
