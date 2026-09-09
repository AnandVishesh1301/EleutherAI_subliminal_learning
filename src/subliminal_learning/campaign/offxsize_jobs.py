"""jobq job builders for the off-policy cross-size campaign.

Every command line is self-describing (sizes, rung, recipe, seeds, absolute pool dir), so
editing a config after submit can never silently re-point an already-queued cell. `cwd`
and `env` come from the queue defaults the driver writes (the code pin), so a job carries
only `key`, `cmd`, `mem_mib` and `slots`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from subliminal_learning.campaign import offxsize_expected as expected
from subliminal_learning.campaign import offxsize_layout as layout
from subliminal_learning.campaign import offxsize_pairs as pairs
from subliminal_learning.campaign import offxsize_plan as plan
from subliminal_learning.jobqueue.model import Job

SWEEP_MODULE = "scripts.sweeps.sweep_offxsize"
_BASE = f"uv run python -m {SWEEP_MODULE}"

# Tokens that would mean this campaign is reading someone else's tree or a base it never
# declared. Cheap to check, and the failure they prevent is silent.
_FORBIDDEN: tuple[str, ...] = ("angela", "weight-seed", "data-seed", "@step")


def _assert_clean_cmd(cmd: str, key: str) -> None:
    """Raise when a command names a foreign tree or an undeclared base.

    Raises:
        ValueError: The command contains a forbidden token.
    """
    lowered = cmd.lower()
    for token in _FORBIDDEN:
        if token in lowered:
            raise ValueError(f"job {key!r} command contains {token!r}: {cmd}")


def _job(key: str, cmd: str, footprint: plan.Footprint) -> Job:
    _assert_clean_cmd(cmd, key)
    return Job(key=key, cmd=cmd, mem_mib=footprint.mem_mib, slots=footprint.slots)


def _wandb_flag(enabled: bool) -> str:
    return "" if enabled else " --no-wandb"


# ---- P1: teacher dose ladder --------------------------------------------------


def teacher_ladder_jobs() -> list[Job]:
    """Train every ladder rung at the ladder seed, for the two sizes not adopted."""
    jobs: list[Job] = []
    for size in plan.TEACHER_TRAIN_SIZES:
        for rung in plan.TEACHER_RUNGS:
            logs = plan.teacher_logs_wandb(
                phase=plan.PHASE_LADDER, ftseed=plan.TEACHER_LADDER_FTSEED
            )
            cmd = (
                f"{_BASE} teacher-cell --size {size} --rung-tag {rung.key} "
                f"--ftseed {plan.TEACHER_LADDER_FTSEED} --phase {plan.PHASE_LADDER}"
                f"{_wandb_flag(logs)}"
            )
            jobs.append(
                _job(
                    f"p1a|teacher|{size}|{rung.key}", cmd, plan.TEACHER_FOOTPRINT[size]
                )
            )
    return jobs


def teacher_eval_jobs(
    rung_tags: Sequence[str] | None = None, *, queue_tag: str = "p1a"
) -> list[Job]:
    """One eval job per size, covering its rungs sequentially.

    Sequential by construction: every rung under a size shares one `base_control.json`
    at the parent, merged read-modify-write, so parallel jobs would drop each other's
    prompt sets.
    """
    tags = (
        tuple(rung_tags)
        if rung_tags is not None
        else tuple(r.key for r in plan.TEACHER_RUNGS)
    )
    jobs: list[Job] = []
    for size in plan.SIZES:
        cmd = f"{_BASE} teacher-eval-cell --size {size} --rungs {','.join(tags)}"
        jobs.append(
            _job(f"{queue_tag}|teval|{size}", cmd, plan.TEACHER_FOOTPRINT[size])
        )
    return jobs


def probe_jobs() -> list[Job]:
    """One 2k-prompt format-health probe per (size, rung) at the ladder seed."""
    jobs: list[Job] = []
    for size in plan.SIZES:
        for rung in plan.TEACHER_RUNGS:
            cmd = (
                f"{_BASE} probe-cell --size {size} --rung-tag {rung.key} "
                f"--ftseed {plan.TEACHER_LADDER_FTSEED}"
            )
            jobs.append(
                _job(f"p1a|probe|{size}|{rung.key}", cmd, plan.PROBE_FOOTPRINT[size])
            )
    return jobs


def base_prior_jobs() -> list[Job]:
    """Score each size's untuned base once, so student cells need no base control."""
    return [
        _job(
            f"p1a|prior|{size}",
            f"{_BASE} base-prior-cell --size {size}",
            plan.TEACHER_FOOTPRINT[size],
        )
        for size in plan.SIZES
    ]


def seedfill_jobs(rungs: Mapping[str, str]) -> list[Job]:
    """Train teacher seeds 0 and 2 at each trained size's picked rung."""
    jobs: list[Job] = []
    for size in plan.TEACHER_TRAIN_SIZES:
        rung = rungs.get(size)
        if rung is None:
            continue
        for ftseed in plan.TEACHER_FTSEEDS:
            if ftseed == plan.TEACHER_LADDER_FTSEED:
                continue
            cmd = (
                f"{_BASE} teacher-cell --size {size} --rung-tag {rung} "
                f"--ftseed {ftseed} --phase p1b --no-wandb"
            )
            jobs.append(
                _job(
                    f"p1b|teacher|{size}|{rung}|tfs{ftseed}",
                    cmd,
                    plan.TEACHER_FOOTPRINT[size],
                )
            )
    return jobs


def seedfill_eval_jobs(rungs: Mapping[str, str]) -> list[Job]:
    """Re-run each trained size's teacher eval so the filled seeds enter its summary."""
    jobs: list[Job] = []
    for size in plan.TEACHER_TRAIN_SIZES:
        rung = rungs.get(size)
        if rung is None:
            continue
        cmd = f"{_BASE} teacher-eval-cell --size {size} --rungs {rung}"
        jobs.append(_job(f"p1b|teval|{size}|{rung}", cmd, plan.TEACHER_FOOTPRINT[size]))
    return jobs


# ---- students -----------------------------------------------------------------


def student_jobs(
    root: Path,
    leaves: Sequence[pairs.Leaf],
    *,
    tags: Mapping[str, str],
    shas: Mapping[str, str] | None = None,
) -> list[Job]:
    """One training job per enumerated cell.

    Args:
        root: Campaign root (absolute; baked into every command).
        leaves: Cells to train, already deduplicated.
        tags: Pool key -> honest-n dataset tag, from the frozen numbers gate.
        shas: Pool key -> `filtered_sha256`, so a cell refuses a pool that moved under it.
    """
    jobs: list[Job] = []
    for leaf in leaves:
        tag = expected.dataset_tag_of(leaf, tags)
        pool = layout.filtered_dir(_pool_dir(root, leaf))
        logs = plan.student_logs_wandb(
            phase=leaf.phase, ftseed=leaf.ftseed, gseed=leaf.gseed
        )
        parts = [
            f"{_BASE} run-cell",
            f"--phase {leaf.phase}",
            f"--kind {leaf.kind}",
            f"--teacher-size {leaf.teacher}",
            f"--student-size {leaf.student}",
            f"--recipe-key {leaf.recipe.key}",
            f"--gseed {leaf.gseed}",
            f"--pool-dir {pool}",
            f"--number-dataset-tag {tag}",
        ]
        if leaf.kind == pairs.TREATMENT:
            parts.append(f"--rung-tag {leaf.rung_tag}")
            parts.append(f"--ftseed {leaf.ftseed}")
        sha = (shas or {}).get(leaf.pool_key)
        if sha:
            parts.append(f"--expect-sha {sha}")
        cmd = " ".join(parts) + _wandb_flag(logs)
        jobs.append(_job(leaf.key, cmd, plan.STUDENT_FOOTPRINT[leaf.student]))
    return jobs


def capability_jobs(
    root: Path,
    leaves: Sequence[pairs.Leaf],
    *,
    tags: Mapping[str, str],
    benchmarks: bool = False,
    queue_tag: str = "p4",
) -> list[Job]:
    """One capability job per `{student_tag}` parent covering `leaves`.

    Per parent, not per seed: the forced-choice and capability summaries are written at
    the parent and merged, so seed-level jobs would race on them.
    """
    jobs: list[Job] = []
    for parent, size in expected.capability_parents(root, leaves, tags):
        flag = " --benchmarks" if benchmarks else ""
        cmd = f"{_BASE} eval-cell --param-dir {parent} --student-size {size}{flag}"
        jobs.append(
            _job(
                f"{queue_tag}|{_parent_key(root, parent)}",
                cmd,
                plan.CAPABILITY_FOOTPRINT[size],
            )
        )
    return jobs


def _parent_key(root: Path, parent: Path) -> str:
    """Key fragment naming one param dir uniquely.

    The dataset tag and student tag repeat across teacher lineages, so the key has to
    carry the whole path below the campaign root or two different cells collide onto one
    job and only one of them is ever evaluated.
    """
    try:
        rel = parent.relative_to(root)
    except ValueError:
        rel = parent
    return "|".join(rel.parts)


def warm_eval_base_jobs(
    root: Path, leaves: Sequence[pairs.Leaf], *, tags: Mapping[str, str]
) -> list[Job]:
    """Warm each capability tree's shared base cache once, before its cells fan out."""
    seen: dict[str, tuple[Path, str]] = {}
    for parent, size in expected.capability_parents(root, leaves, tags):
        seen.setdefault(str(parent.parent), (parent, size))
    return [
        _job(
            f"warm|{_parent_key(root, parent.parent)}",
            f"{_BASE} warm-eval-base --param-dir {parent} --student-size {size}",
            plan.CAPABILITY_FOOTPRINT[size],
        )
        for parent, size in seen.values()
    ]


def _pool_dir(root: Path, leaf: pairs.Leaf) -> Path:
    """Raw-pool dir this cell distils from."""
    if leaf.kind == pairs.RNG_CONTROL:
        return layout.rng_pool_dir(root)
    if leaf.kind == pairs.BASE_CONTROL:
        return layout.base_pool_dir(root, leaf.teacher)
    return layout.teacher_pool_dir(
        root, leaf.teacher, str(leaf.rung_tag), int(leaf.ftseed or 0)
    )
