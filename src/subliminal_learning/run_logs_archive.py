"""Rule-based archiver for the flat ``run_logs/`` directory (pure logic; CLI is thin).

The two nodes drop sweep logs as flat top-level files in a shared ``run_logs/`` dir. Over
time that dir grows to thousands of ``.log``/``.out`` files. This module sorts *old* logs into
``run_logs/archive/<family>/`` subdirs, with strong safety rails so a file that any live
waiter still greps (or that a process still has open) is never moved.

Everything here is a pure function or small dataclass so the CLI (``scripts/archive_run_logs``)
stays a thin shell and the moves can be exercised entirely in ``tmp_path`` trees.

Eligibility (a top-level regular file moves only if ALL hold):

1. regular file at the top level (subdirs such as ``archive/``, ``queue/``, ``__pycache__``
   are never recursed into),
2. extension is ``.log`` or ``.out`` (never ``README.md`` or job scripts),
3. ``mtime < cutoff`` (an explicit ISO-8601 UTC instant — the caller must think),
4. its basename does NOT appear (fixed-string) in any top-level ``*.sh``/``*.py`` — waiter
   scripts grep master logs by name, so a referenced file must stay put,
5. it is not currently open by any process on this node.

The MANIFEST is an append-only TSV with columns::

    moved_utc  action  family  src_relpath  dst_relpath  size_bytes  mtime_iso

``action`` is ``move`` for archiving and ``undo`` for a restore.
"""

from __future__ import annotations

import errno
import glob
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

ARCHIVE_DIRNAME = "archive"
MANIFEST_NAME = "MANIFEST.tsv"
ALLOWED_EXTS = (".log", ".out")

# Subdirectories we never look inside (belt-and-suspenders; we only scan the top level anyway).
SKIP_DIRS = frozenset({ARCHIVE_DIRNAME, "queue", "__pycache__"})

# A leading run-stamp prefix such as ``20260723T184443Z_`` (time is 4-6 digits in the wild).
STAMP_RE = re.compile(r"^\d{8}T\d{4,6}Z_")

# Ordered (prefix -> family) rules. First match wins, so the most specific prefix must come
# first (e.g. ``dog_crossdistill_studenthp`` before ``dog_crossdistill`` before ``dog_``).
# Designed against the real run_logs so >=95% of .log/.out files land in a meaningful family.
CLASSIFY_RULES: tuple[tuple[str, str], ...] = (
    ("dog_crossdistill_studenthp", "dog_crossdistill_studenthp"),
    ("dog_crossdistill", "dog_crossdistill"),
    ("dog_samedistill", "dog_samedistill"),
    ("dog_constrained", "dog_constrained_numgen"),
    ("dog_numgen", "dog_numgen"),
    ("dog_distill", "dog_distill"),
    ("dog_", "dog"),
    ("controls_backfill", "controls_backfill"),
    ("controls", "controls"),
    ("rabbitlion", "rabbitlion"),
    ("dataseed1", "dataseed1"),
    ("pig", "pig"),
    ("eas", "eas"),
    ("steerability", "steerability"),
    ("crossbase", "crossbase"),
    ("samebase", "samebase"),
    ("crossseed", "crossseed"),
    ("cat", "cat"),
    ("selected_number_gen", "numgen"),
    ("numgen", "numgen"),
    ("number", "numgen"),
    ("selected", "teacher_sweep"),
    ("full", "teacher_sweep"),
    ("teacher", "teacher_sweep"),
    ("train", "teacher_sweep"),
    ("eval", "teacher_sweep"),
    ("distill", "distill"),
    ("student", "student"),
    ("gpu_idle", "gpu_idle"),
    ("external_gpu", "gpu_idle"),
)

# Exclusion reasons, in the order they are checked (used for the summary counts).
EXCLUSION_REASONS = ("wrong_ext", "too_new", "referenced", "open")

# A fallback token counts as numeric/shardy if it contains a digit or is a known shard word.
_SHARDY_WORDS = frozenset({"shard", "shards", "gpu", "cycle", "worker", "attempt"})
_HAS_DIGIT = re.compile(r"\d")


def strip_stamp(name: str) -> str:
    """Drop an optional leading run-stamp prefix (``20260723T184443Z_``) from ``name``."""
    return STAMP_RE.sub("", name, count=1)


def _fallback_family(stem: str) -> str:
    """Deterministic fallback: leading ``[a-z0-9_]`` tokens up to the first shardy token.

    ``stem`` is the stamp-stripped basename (extension already removed). Returns ``misc`` if
    even the first token looks numeric/shardy.
    """
    kept: list[str] = []
    for token in stem.split("_"):
        if not token or token in _SHARDY_WORDS or _HAS_DIGIT.search(token):
            break
        kept.append(token)
    return "_".join(kept) if kept else "misc"


def classify(name: str) -> str:
    """Return the archive family for a log basename (order-sensitive; see ``CLASSIFY_RULES``)."""
    stem = strip_stamp(name)
    for prefix, family in CLASSIFY_RULES:
        if stem.startswith(prefix):
            return family
    # No rule matched: fall back on the leading non-shardy tokens of the (extension-less) stem.
    return _fallback_family(Path(stem).stem)


def collect_open_paths() -> set[str]:
    """Return the set of absolute paths currently open by any process on this node.

    Scans ``/proc/[0-9]*/fd/*`` symlinks once. Per-fd races (the process or fd vanishing) and
    permission errors are ignored. Isolated behind a function so tests can monkeypatch it.
    """
    open_paths: set[str] = set()
    for link in glob.glob("/proc/[0-9]*/fd/*"):
        try:
            open_paths.add(os.readlink(link))
        except OSError:  # PermissionError / FileNotFoundError are OSError subclasses
            continue
    return open_paths


def collect_reference_text(run_logs: Path) -> str:
    """Concatenate the contents of every top-level ``*.sh``/``*.py`` file under ``run_logs``.

    A file whose basename appears anywhere in this text is still referenced by a waiter/launcher
    and must not be moved (rule 4). Read once per invocation, not per candidate.
    """
    parts: list[str] = []
    with os.scandir(run_logs) as it:
        for entry in it:
            if entry.is_file(follow_symlinks=False) and entry.name.endswith((".sh", ".py")):
                try:
                    parts.append(Path(entry.path).read_text(errors="replace"))
                except OSError:
                    continue
    return "\n".join(parts)


@dataclass
class Candidate:
    """One eligible top-level log file slated to move into ``archive/<family>/``."""

    path: Path
    family: str
    size: int
    mtime: float


@dataclass
class Plan:
    """The result of scanning ``run_logs``: what would move, and why the rest was excluded."""

    candidates: list[Candidate] = field(default_factory=list)
    excluded: dict[str, int] = field(default_factory=lambda: dict.fromkeys(EXCLUSION_REASONS, 0))


def parse_cutoff(value: str) -> datetime:
    """Parse an ISO-8601 UTC instant (trailing ``Z`` allowed) into a tz-aware datetime."""
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def build_plan(
    run_logs: Path,
    cutoff: datetime,
    *,
    open_paths: set[str] | None = None,
    reference_text: str | None = None,
) -> Plan:
    """Scan the top level of ``run_logs`` and split files into move-candidates vs exclusions.

    Only regular top-level files are considered; subdirectories (``archive/`` etc.) are ignored
    and never counted. Each excluded file is charged to the FIRST rule it fails.
    """
    if open_paths is None:
        open_paths = collect_open_paths()
    if reference_text is None:
        reference_text = collect_reference_text(run_logs)

    cutoff_ts = cutoff.timestamp()
    plan = Plan()
    with os.scandir(run_logs) as it:
        for entry in it:
            if not entry.is_file(follow_symlinks=False):
                continue
            name = entry.name
            if not name.endswith(ALLOWED_EXTS):
                plan.excluded["wrong_ext"] += 1
                continue
            stat = entry.stat(follow_symlinks=False)
            if not stat.st_mtime < cutoff_ts:
                plan.excluded["too_new"] += 1
                continue
            if name in reference_text:
                plan.excluded["referenced"] += 1
                continue
            if entry.path in open_paths or os.path.realpath(entry.path) in open_paths:
                plan.excluded["open"] += 1
                continue
            plan.candidates.append(
                Candidate(
                    path=Path(entry.path),
                    family=classify(name),
                    size=stat.st_size,
                    mtime=stat.st_mtime,
                )
            )
    plan.candidates.sort(key=lambda c: (c.family, c.path.name))
    return plan


def summarize(candidates: list[Candidate]) -> dict[str, dict[str, object]]:
    """Group candidates by family into ``{count, bytes, samples}`` (samples: up to 5 names)."""
    summary: dict[str, dict[str, object]] = {}
    for cand in candidates:
        fam = summary.setdefault(cand.family, {"count": 0, "bytes": 0, "samples": []})
        fam["count"] = int(fam["count"]) + 1  # type: ignore[arg-type]
        fam["bytes"] = int(fam["bytes"]) + cand.size  # type: ignore[arg-type]
        samples: list[str] = fam["samples"]  # type: ignore[assignment]
        if len(samples) < 5:
            samples.append(cand.path.name)
    return summary


def log_summary(title: str, candidates: list[Candidate], excluded: dict[str, int] | None) -> None:
    """Emit a per-family summary (counts, bytes, sample names) plus exclusion counts via loguru."""
    summary = summarize(candidates)
    total = len(candidates)
    total_bytes = sum(c.size for c in candidates)
    logger.info(f"{title}: {total} file(s), {total_bytes} bytes across {len(summary)} family(ies)")
    for family in sorted(summary):
        info = summary[family]
        samples = ", ".join(info["samples"])  # type: ignore[arg-type]
        logger.info(f"  {family}: {info['count']} file(s), {info['bytes']} bytes  e.g. {samples}")
    if excluded is not None:
        parts = ", ".join(f"{reason}={excluded.get(reason, 0)}" for reason in EXCLUSION_REASONS)
        logger.info(f"  excluded: {parts}")


def _manifest_path(run_logs: Path) -> Path:
    return run_logs / ARCHIVE_DIRNAME / MANIFEST_NAME


def _append_manifest(run_logs: Path, row: list[str]) -> None:
    """Append one TSV row to the archive manifest (created with its parent dir if missing)."""
    manifest = _manifest_path(run_logs)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest, "a", encoding="utf-8") as f:
        f.write("\t".join(row) + "\n")


def _manifest_row(action: str, family: str, src_rel: str, dst_rel: str, size: int, mtime: float) -> list[str]:
    now = datetime.now(timezone.utc).isoformat()
    mtime_iso = datetime.fromtimestamp(mtime, timezone.utc).isoformat()
    return [now, action, family, src_rel, dst_rel, str(size), mtime_iso]


def apply_plan(run_logs: Path, plan: Plan) -> list[Candidate]:
    """Move each candidate into ``archive/<family>/`` via ``os.rename``; append a manifest row.

    Refuses to overwrite (skip + warn if the destination exists) and never copies across
    filesystems (a cross-device rename is warned and skipped). Returns the moved candidates.
    """
    archive_root = run_logs / ARCHIVE_DIRNAME
    moved: list[Candidate] = []
    for cand in plan.candidates:
        dst_dir = archive_root / cand.family
        dst = dst_dir / cand.path.name
        if dst.exists():
            logger.warning(f"skip {cand.path.name}: destination already exists ({dst})")
            continue
        dst_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.rename(cand.path, dst)
        except OSError as exc:
            if exc.errno == errno.EXDEV:
                logger.warning(f"skip {cand.path.name}: cross-device rename (would need a copy)")
            else:
                logger.warning(f"skip {cand.path.name}: rename failed ({exc})")
            continue
        _append_manifest(
            run_logs,
            _manifest_row(
                "move",
                cand.family,
                cand.path.name,
                str(dst.relative_to(run_logs)),
                cand.size,
                cand.mtime,
            ),
        )
        moved.append(cand)
    return moved


def _read_manifest_moves(run_logs: Path) -> list[dict[str, str]]:
    """Return the ``move`` rows of the manifest (ignoring headers/blank/short lines)."""
    manifest = _manifest_path(run_logs)
    moves: list[dict[str, str]] = []
    if not manifest.exists():
        return moves
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        if len(fields) != 7 or fields[1] != "move":
            continue
        moves.append(
            {
                "family": fields[2],
                "src_rel": fields[3],
                "dst_rel": fields[4],
                "size": fields[5],
                "mtime_iso": fields[6],
            }
        )
    return moves


def undo(run_logs: Path) -> list[str]:
    """Move archived files back to the top level, appending ``undo`` manifest rows.

    Reads every ``move`` row; restores it only when the archived file is present and the
    original slot is free. Already-undone or missing rows are skipped with a warning. Returns
    the src relpaths that were restored.
    """
    restored: list[str] = []
    for move in _read_manifest_moves(run_logs):
        src = run_logs / move["src_rel"]
        dst = run_logs / move["dst_rel"]
        if not dst.exists():
            logger.warning(f"skip undo {move['src_rel']}: archived file missing ({dst})")
            continue
        if src.exists():
            logger.warning(f"skip undo {move['src_rel']}: original slot occupied ({src})")
            continue
        src.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.rename(dst, src)
        except OSError as exc:
            logger.warning(f"skip undo {move['src_rel']}: rename failed ({exc})")
            continue
        stat = src.stat()
        _append_manifest(
            run_logs,
            _manifest_row(
                "undo",
                move["family"],
                move["src_rel"],
                move["dst_rel"],
                stat.st_size,
                stat.st_mtime,
            ),
        )
        restored.append(move["src_rel"])
    return restored
