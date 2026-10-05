"""CLI: enumerate every "student vs. its teacher" CKA target and list/submit them to jobq.

Discovers, glob-based and layout-tolerant, every trained student run under the students
output tree, decides which are still PENDING against the fixed output-artifact contract
(see below), and either prints a summary table, lists the pending targets, or submits
them into a jobq queue. It NEVER runs GPU work itself — it only enumerates, lists, and
submits.

A target is one student ftseed run dir under ``{output_root}/students/`` that has a
``train_manifest.json`` + ``checkpoints/``. Three cohorts live in the tree:

  - treatment: ``students/{animal}/{student_base}/teacher-{teacher_base}/{ttag}_tseed{T}/{ndt}/{stag}/ftseed{S}``
  - base control: ``students/controls/base/{model}/teacher-{numbers_base}/{ndt}/{stag}/ftseed{S}``
    (included only with ``--include-controls``).
  - rng control: ``students/controls/rng/{model}/{ndt}/{stag}/ftseed{S}`` — has NO teacher
    operand, so it is ALWAYS excluded and only counted (skipped-rng) so the exclusion is
    visible; it is never submitted.

Output-artifact contract used for resume detection (implemented by ``scripts.compute_cka``
in ``--student`` mode, not here): a target is DONE iff
``{run_dir}/cka/vs_teacher__numbers/cka_manifest.json`` exists. The dir is built via
``subliminal_learning.cka.layout.vs_teacher_dir`` (condition is fixed to "numbers").
"""

from __future__ import annotations

import fnmatch
import hashlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import typer
from dotenv import load_dotenv
from loguru import logger

from subliminal_learning.cka.layout import vs_teacher_dir
from subliminal_learning.config import load_config
from jobq import store
from jobq.model import Job
from subliminal_learning.gkd.layout import parse_optim_slug
from subliminal_learning.teachers.run_layout import CKPT_SUBDIR
from subliminal_learning.utils.logging_setup import configure_loguru

# Absolute repo cwd every job must cd into first: jobs on shared queue roots have been
# lost to cwd leaks (drained against another node's paths root), so never rely on the
# worker's inherited cwd.

CONDITION = "numbers"  # fixed axis: the manifest name is the contract with compute_cka.


# --------------------------- target model ---------------------------


@dataclass(frozen=True)
class StudentTarget:
    """One trained student ftseed run to score against its teacher."""

    kind: str  # "treatment" | "base-control"
    run_dir: Path  # the ftseed{S} run dir (parent of checkpoints)
    ftseed: int
    animal: str | None
    student_base: str | None
    teacher_base: str | None
    lineage: str | None  # "teacher-{base}/{ttag}_tseed{T}" (treatment only)
    dataset_tag: str
    param_tag: str  # student param tag (stag)


# --------------------------- discovery ---------------------------


def _parse_ftseed(name: str) -> int | None:
    tail = name.removeprefix("ftseed")
    return int(tail) if tail.isdigit() else None


def _strip_optim(parts: tuple[str, ...]) -> tuple[str, ...]:
    """Drop a leading non-default-optimizer ``{optim_slug}`` level, if present.

    Keyed on the closed optimizer-slug vocabulary (`parse_optim_slug`): a real leading
    component (``{animal}`` or the literal ``controls``) fails to parse and is kept. The
    ``_{optim_slug}`` suffix on the ``{stag}`` leaf still keeps the cells distinct downstream.
    """
    if parts:
        try:
            parse_optim_slug(parts[0])
        except ValueError:
            return parts
        return parts[1:]
    return parts


def _strip_ckpt(parts: tuple[str, ...]) -> tuple[str, ...]:
    """Drop the gathered ``ckpt/`` model level (revision-pinned runs) so the fixed-index
    parsing below sees the plain shape; the model name keeps its ``@rev`` suffix. ``ckpt``
    is a reserved sentinel (never an animal/param_tag/dataset), so removing it is unambiguous.
    """
    return tuple(p for p in parts if p != CKPT_SUBDIR)


def _parse_student(rel: Path) -> dict | None:
    """Parse a students-tree run dir -> fields (or None if not a scorable target).

    Returns ``kind == "rng-control"`` for rng controls so the caller can count them as
    skipped; every other cohort returns its submittable fields.
    """
    parts = _strip_ckpt(_strip_optim(rel.parts))
    ftseed = _parse_ftseed(parts[-1])
    if ftseed is None:
        return None
    if parts and parts[0] == "controls":
        if len(parts) == 7 and parts[1] == "base":
            # controls/base/{model}/teacher-{nbase}/{ndt}/{stag}/ftseed
            return {
                "kind": "base-control",
                "ftseed": ftseed,
                "animal": None,
                "student_base": parts[2],
                "teacher_base": parts[3].removeprefix("teacher-"),
                "lineage": None,
                "dataset_tag": parts[4],
                "param_tag": parts[5],
            }
        if len(parts) == 6 and parts[1] == "rng":
            # controls/rng/{model}/{ndt}/{stag}/ftseed — no teacher operand.
            return {
                "kind": "rng-control",
                "ftseed": ftseed,
                "animal": None,
                "student_base": parts[2],
                "teacher_base": None,
                "lineage": None,
                "dataset_tag": parts[3],
                "param_tag": parts[4],
            }
        return None
    # treatment: {animal}/{model}/teacher-{base}/{ttag}_tseed{T}/{ndt}/{stag}/ftseed
    if len(parts) != 7 or not parts[2].startswith("teacher-"):
        return None
    return {
        "kind": "treatment",
        "ftseed": ftseed,
        "animal": parts[0],
        "student_base": parts[1],
        "teacher_base": parts[2].removeprefix("teacher-"),
        "lineage": f"{parts[2]}/{parts[3]}",
        "dataset_tag": parts[4],
        "param_tag": parts[5],
    }


@dataclass
class DiscoveryStats:
    """Counters for runs skipped during discovery."""

    incomplete: int = 0
    rng_skipped: int = 0


def discover_students(
    output_root: Path,
) -> tuple[list[StudentTarget], DiscoveryStats]:
    """Glob every ``ftseed*/checkpoints`` under ``students/`` into targets.

    A run is COMPLETE iff its checkpoints hold ``adapter_config.json`` and its ftseed dir
    holds ``train_manifest.json``. rng controls are counted (``rng_skipped``) and dropped
    (no teacher operand); un-parseable/incomplete runs are counted (``incomplete``).
    """
    stats = DiscoveryStats()
    out: list[StudentTarget] = []
    tree_root = output_root / "students"
    if not tree_root.is_dir():
        return out, stats
    for ckpt in sorted(tree_root.rglob("ftseed*/checkpoints")):
        if not ckpt.is_dir():
            continue
        run_dir = ckpt.parent
        if not (ckpt / "adapter_config.json").is_file():
            stats.incomplete += 1
            continue
        if not (run_dir / "train_manifest.json").is_file():
            stats.incomplete += 1
            continue
        rel = run_dir.relative_to(tree_root)
        fields = _parse_student(rel)
        if fields is None:
            stats.incomplete += 1
            continue
        if fields["kind"] == "rng-control":
            stats.rng_skipped += 1
            continue
        out.append(StudentTarget(run_dir=run_dir, **fields))
    return out, stats


# --------------------------- filtering ---------------------------


def filter_students(
    students: list[StudentTarget],
    *,
    animals: tuple[str, ...] | None,
    student_base: str | None,
    teacher_base: str | None,
    lineage: str | None,
    dataset_tag: str | None,
    param_tag: str | None,
    ftseeds: tuple[int, ...] | None,
    include_controls: bool,
) -> list[StudentTarget]:
    """Apply the CLI filters. ``lineage`` is fnmatched, treatment-only."""
    out: list[StudentTarget] = []
    for s in students:
        if not include_controls and s.kind == "base-control":
            continue
        if animals and s.animal not in animals:
            continue
        if student_base and s.student_base != student_base:
            continue
        if teacher_base and s.teacher_base != teacher_base:
            continue
        if lineage is not None:
            if s.lineage is None or not fnmatch.fnmatch(s.lineage, lineage):
                continue
        if dataset_tag and s.dataset_tag != dataset_tag:
            continue
        if param_tag and not fnmatch.fnmatch(s.param_tag, param_tag):
            continue
        if ftseeds and s.ftseed not in ftseeds:
            continue
        out.append(s)
    return out


# --------------------------- done detection ---------------------------


def cka_manifest(run_dir: Path) -> Path:
    """The ``cka_manifest.json`` artifact path for a student's "vs teacher" numbers run."""
    return vs_teacher_dir(run_dir, CONDITION) / "cka_manifest.json"


def student_done(run_dir: Path) -> bool:
    """Whether a student's "vs teacher" CKA manifest exists (DONE contract)."""
    return cka_manifest(run_dir).is_file()


# --------------------------- job building ---------------------------


def _group_key(target: StudentTarget) -> tuple:
    """Coordinates that fix a target's teacher lineage + numbers pool.

    Members sharing this key can be scored in one ``compute_cka`` process; the CLI
    hard-fails on mixed teacher/pool, so grouping must guarantee a single lineage.
    Base controls carry the cohort tag so they never share a chunk with treatment.
    """
    if target.kind == "treatment":
        # lineage == "teacher-{base}/{ttag}_tseed{T}"; the second segment fixes T.
        ttag_tseed = target.lineage.split("/", 1)[1] if target.lineage else ""
        return ("treatment", target.animal, target.teacher_base, ttag_tseed,
                target.dataset_tag)
    # base-control: controls/base/{model}/teacher-{nbase}/{ndt}/...
    return ("base-control", target.student_base, target.teacher_base,
            target.dataset_tag)


def _chunk_job(rels: list[str], device: str) -> Job:
    """Build a jobq Job for one chunk of student rel paths (already sorted)."""
    student_flags = " ".join(f"--student {rel}" for rel in rels)
    cmd = (
        f"uv run python -m scripts.compute_cka "
        f"--condition {CONDITION} {student_flags} --no-plots --device {device}"
    )
    digest = hashlib.sha256("\n".join(rels).encode()).hexdigest()[:12]
    key = f"cka|vtn|{rels[0]}|n{len(rels)}|{digest}"
    return Job(key=key, cmd=cmd)


def build_jobs(
    pending: list[StudentTarget],
    output_root: Path,
    device: str,
    students_per_job: int,
) -> list[Job]:
    """Chunk pending targets sharing a teacher+pool into multi-student jobs.

    Targets are grouped by :func:`_group_key`, sorted by rel path within a group,
    then split into chunks of at most ``students_per_job``. Each chunk becomes one
    ``compute_cka`` invocation with repeated ``--student`` flags.
    """
    groups: dict[tuple, list[str]] = defaultdict(list)
    for t in pending:
        rel = t.run_dir.relative_to(output_root).as_posix()
        groups[_group_key(t)].append(rel)
    jobs: list[Job] = []
    for gkey in sorted(groups):
        rels = sorted(groups[gkey])
        for i in range(0, len(rels), students_per_job):
            jobs.append(_chunk_job(rels[i:i + students_per_job], device))
    return jobs


def _queue_defaults(
    mem_mib: int | None, cap_per_gpu: int | None, slots: int | None
) -> dict:
    d: dict = {}
    if mem_mib is not None:
        d["mem_mib"] = mem_mib
    if cap_per_gpu is not None:
        d["cap_per_gpu"] = cap_per_gpu
    if slots is not None:
        d["slots"] = slots
    return d


def submit_jobs(
    root: Path,
    queue: str,
    jobs: list[Job],
    *,
    priority: int,
    node: str | None,
    defaults: dict,
) -> list[Job]:
    """Create the queue (idempotent) and append only jobs whose key is not present.

    Returns the jobs actually appended (dedupe by jobkey, so resubmitting is a no-op
    for already-queued targets).
    """
    existing = {j.jobkey for j in store.load_jobs(root, queue)} if store.queue_exists(
        root, queue
    ) else set()
    fresh = [j for j in jobs if j.jobkey not in existing]
    if not fresh:
        return []
    store.check_new_jobs(root, queue, fresh)
    store.create_queue(
        root,
        queue,
        depends_on=None,
        node=node,
        priority=priority,
        defaults=defaults,
        exist_ok=True,
    )
    store.append_jobs(root, queue, fresh)
    return fresh


# --------------------------- CLI ---------------------------


def main(
    animal: list[str] = typer.Option(None, help="Restrict to these animals (repeatable)."),
    student_base: str = typer.Option(None, help="Restrict to this student base-model dir."),
    teacher_base: str = typer.Option(
        None, help="Restrict to this teacher base (the teacher-{base} segment)."
    ),
    lineage: str = typer.Option(
        None, help="fnmatch on 'teacher-{base}/{ttag}_tseed{T}' (treatment only)."
    ),
    dataset_tag: str = typer.Option(None, help="Restrict to this numbers-dataset tag."),
    param_tag: str = typer.Option(None, help="fnmatch on the student param tag."),
    ftseed: list[int] = typer.Option(None, help="Restrict to these ftseeds (repeatable)."),
    include_controls: bool = typer.Option(
        True,
        "--include-controls/--no-include-controls",
        help="Include student base controls (rng controls are always excluded).",
    ),
    list_: bool = typer.Option(False, "--list", help="Print one pending target per line."),
    submit: str = typer.Option(None, help="Submit pending targets to this jobq queue."),
    force: bool = typer.Option(False, help="Treat every target as pending."),
    limit: int = typer.Option(0, help="Cap submissions at N targets (0 = no cap)."),
    students_per_job: int = typer.Option(
        8, min=1, help="Max students measured per compute_cka job (shared teacher/pool)."
    ),
    priority: int = typer.Option(0, help="Queue priority for the submitted queue."),
    device: str = typer.Option("cuda:0", help="Per-job --device passthrough."),
    queue_root: Path = typer.Option(
        Path("run_logs/queue_angela"), help="jobq queue state root."
    ),
    node: str = typer.Option(None, help="Pin the submitted queue to this hostname."),
    mem_mib: int = typer.Option(
        12000,
        help="Queue-default free-MiB gate (~0.65 GB per resident merged 160m fp32 "
        "model: teacher + K students per job).",
    ),
    cap_per_gpu: int | None = typer.Option(
        None, help="Queue-default max concurrent jobs per GPU (default: none; node GPU policy applies)."
    ),
    slots: int = typer.Option(None, help="Queue-default per-GPU slot weight."),
    paths_config: Path = Path("configs/paths/paths_local.yaml"),
    model_config: Path = Path("configs/model/pythia.yaml"),
) -> None:
    """Enumerate student-vs-teacher CKA targets; print a summary, list, or submit to jobq."""
    load_dotenv()
    configure_loguru()

    cfg = load_config(paths_config, model_config)
    output_root = cfg.paths.output_root

    all_students, stats = discover_students(output_root)
    students = filter_students(
        all_students,
        animals=tuple(animal) if animal else None,
        student_base=student_base,
        teacher_base=teacher_base,
        lineage=lineage,
        dataset_tag=dataset_tag,
        param_tag=param_tag,
        ftseeds=tuple(ftseed) if ftseed else None,
        include_controls=include_controls,
    )
    pending = [s for s in students if force or not student_done(s.run_dir)]

    logger.info(
        "Discovered {} student run(s) ({} selected); skipped {} incomplete + {} rng.",
        len(all_students), len(students), stats.incomplete, stats.rng_skipped,
    )

    if list_:
        for s in pending:
            rel = s.run_dir.relative_to(output_root).as_posix()
            typer.echo(f"{s.kind}\t{rel}")
        return

    if submit:
        _do_submit(
            queue_root, submit, pending, output_root,
            device=device, node=node, limit=limit, priority=priority,
            students_per_job=students_per_job,
            defaults=_queue_defaults(mem_mib, cap_per_gpu, slots),
        )
        return

    _print_summary(students, pending, stats, output_root, students_per_job)


def _do_submit(
    queue_root: Path,
    queue: str,
    pending: list[StudentTarget],
    output_root: Path,
    *,
    device: str,
    node: str | None,
    limit: int,
    priority: int,
    students_per_job: int,
    defaults: dict,
) -> None:
    """Submit student jobs into a single queue (dedupe by jobkey)."""
    capped = pending[:limit] if limit else pending
    jobs = build_jobs(capped, output_root, device, students_per_job)
    n = len(
        submit_jobs(
            queue_root, queue, jobs,
            priority=priority, node=node, defaults=defaults,
        )
    )
    logger.info(
        "Submitted {} student CKA job(s) ({} target(s) @ ≤{}/job) -> {!r} "
        "(prio {}) under {}.",
        n, len(capped), students_per_job, queue, priority, queue_root,
    )
    if limit and len(pending) > limit:
        logger.info("--limit {} capped {} pending targets.", limit, len(pending))


def _print_summary(
    students: list[StudentTarget],
    pending: list[StudentTarget],
    stats: DiscoveryStats,
    output_root: Path,
    students_per_job: int,
) -> None:
    """Print per-cohort totals with done vs pending, plus the rng-skipped count."""
    pending_ids = {id(s) for s in pending}
    rows: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for s in students:
        rows[s.kind][0] += 1
        if id(s) not in pending_ids:
            rows[s.kind][1] += 1
    typer.echo(f"Student-vs-teacher CKA targets (condition {CONDITION})")
    typer.echo(f"{'cohort':<16}{'total':>7}{'done':>7}{'pending':>9}")
    for kind in sorted(rows):
        total, done = rows[kind]
        typer.echo(f"{kind:<16}{total:>7}{done:>7}{total - done:>9}")
    typer.echo(f"{'rng-skipped':<16}{stats.rng_skipped:>7}")
    n_jobs = len(build_jobs(pending, output_root, "cuda:0", students_per_job))
    typer.echo(
        f"TOTAL discovered={len(students)} pending={len(pending)} "
        f"→ {n_jobs} job(s) @ ≤{students_per_job} students/job "
        f"(rng-skipped={stats.rng_skipped}, incomplete={stats.incomplete})"
    )


if __name__ == "__main__":
    typer.run(main)
