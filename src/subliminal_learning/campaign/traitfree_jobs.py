"""jobq jobs and Stage-2 queues of the `traitfree` campaign (pure).

One job per placebo teacher x ftseed (train, then a manifest check that catches the
sweep's claim skip), one eval job per placebo teacher (first in its family's student
queue on the host of its ftseed-0 train job), and one job per student leaf chaining
``train && verify-leaf && eval [&& lm-eval]``. Evals run under a per-config-dir
`flock`: the eval sweep scores a whole config dir and fails a concurrent sibling on
its claim. Every path in a command is absolute; every job pins the campaign paths
YAML and both run-set ids. Hosts come from an LPT over estimated GPU-hours.
"""

from __future__ import annotations

import hashlib
import shlex
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from subliminal_learning.campaign import opdlim_plan
from subliminal_learning.campaign import traitfree_config_gen as cgen
from subliminal_learning.campaign import traitfree_plan as plan
from subliminal_learning.campaign import ws160m_plan
from subliminal_learning.campaign import xfam2_jobs
from subliminal_learning.campaign import xfam2_plan
from subliminal_learning.campaign import xsize_plan
from subliminal_learning.campaign.traitfree_checks import KIND_PLACEBO, KIND_TB
from subliminal_learning.campaign.traitfree_roster import (
    PlaceboTeacher,
    Roster,
    RosterCell,
)
from subliminal_learning.jobqueue.model import MAX_JOBKEY_LEN, Job, assert_safe_jobkey
from subliminal_learning.sdf.layout import SDF_TEACHERS_SUBDIR

TRAIN_MODULE = "scripts.sweeps.sweep_distill_student_gkd"
EVAL_MODULE = "scripts.sweeps.sweep_evaluate_gkd_student"
TEACHER_MODULE = "scripts.sweeps.sweep_sdf_teacher"
TEACHER_EVAL_MODULE = "scripts.sweeps.sweep_evaluate_sdf_teacher"
VERIFY_MODULE = "scripts.verify_traitfree_leaf"

KIND_TEACHER = "teacher"
KIND_TEACHER_EVAL = "teacher_eval"
EVAL_LOCK_DIR = plan.CAMPAIGN_ROOT / "locks" / "evals"
EVAL_LOCK_WAIT_S = 6 * 3600

# Estimates are card-hours (wall hours x slots / cap). A 160M FROZEN leaf (625 steps)
# measured 0.23 at cap 2 on ws160m; other 160M leaves scale by their step count.
STUDENT_160M_LEAF_GPU_H = 0.2
STUDENT_160M_REF_STEPS = 625
TEACHER_160M_WALL_H = xsize_plan.TEACHER_FOOTPRINT["160m"].est_minutes / 60
TEACHER_EVAL_GPU_H = 0.1
# xfam2 plan (2026-09-24 packing smoke): a 6.9b SDF teacher takes ~2.4 h wall; other
# labels scale by their stamped peak memory.
XFAM2_6P9B_TEACHER_WALL_H = 2.4


@dataclass(frozen=True)
class PlannedJob:
    """One Stage-2 job before host assignment.

    Attributes:
        job: The jobq job.
        kind: `KIND_TEACHER`, `KIND_TEACHER_EVAL`, `KIND_PLACEBO` or `KIND_TB`.
        family: Campaign family.
        group: Queue group (student / tb group; the family for teachers).
        tier: Execution tier.
        gpu_h: Estimated card-hours.
        anchor: Jobkey whose host this job must share, or None.
    """

    job: Job
    kind: str
    family: str
    group: str
    tier: str
    gpu_h: float
    anchor: str | None = None


@dataclass(frozen=True)
class QueueSpec:
    """One Stage-2 queue as the driver submits it.

    Attributes:
        name: Queue name (a key of `traitfree_plan.QUEUE_PRIORITIES`).
        jobs: Jobs in submission order.
        depends_on: Queues that must complete first.
        strict_deps: A failed dependency job blocks this queue.
    """

    name: str
    jobs: tuple[Job, ...]
    depends_on: tuple[str, ...]
    strict_deps: bool


# ---- groups ---------------------------------------------------------------------------


def cell_tier(cell: RosterCell) -> str:
    """Execution tier of a cell (checkpoints cells by panel)."""
    return plan.FAMILY_TIER.get(cell.family) or plan.checkpoint_tier(cell.panels)


def student_group(cell: RosterCell) -> str:
    """Student / tb queue group of a cell."""
    if cell.family in plan.FAMILY_TIER:
        return cell.family
    return f"{cell.family}-{plan.checkpoint_tier(cell.panels)}"


def teacher_eval_group(family: str) -> str:
    """Student queue group holding a family's teacher evals."""
    if family in plan.FAMILY_TIER:
        return family
    return f"{family}-{plan.TIER_B}"


# ---- commands -------------------------------------------------------------------------


def _paths_flag() -> str:
    return f"--paths-config {shlex.quote(str(plan.PATHS_CONFIG_PATH))}"


def _config_flags(training: str, experiment: str, registry: str | None) -> list[str]:
    flags = [
        _paths_flag(),
        f"--training-config {shlex.quote(training)}",
        f"--experiment-config {shlex.quote(experiment)}",
    ]
    if registry:
        flags.append(f"--spec-registry {registry}")
    return flags


def _module_cmd(module: str, sweep_config: Path, flags: Sequence[str]) -> str:
    return " ".join(
        [
            f"uv run python -m {module} --sweep-config {shlex.quote(str(sweep_config))}",
            "--device cuda:0",
            *flags,
        ]
    )


def _class_flag(kind: str) -> str:
    return {KIND_PLACEBO: "--no-control", KIND_TB: "--xbase-controls-only"}[kind]


def _leaf_paths(cell: RosterCell, kind: str, tfs: int | None, gseed: int):
    if kind == KIND_PLACEBO:
        if tfs is None:
            raise ValueError("a placebo leaf needs its teacher ftseed")
        return cgen.placebo_paths(cell, tfs, gseed)
    return cgen.tb_paths(cell, gseed)


def student_train_cmd(cell: RosterCell, kind: str, tfs: int | None, gseed: int) -> str:
    """`sweep_distill_student_gkd` command of one placebo or tb leaf."""
    recipe = cgen.cell_recipe(cell)
    rsid = plan.RUN_SET_ID[cell.family]
    flags = [
        _class_flag(kind),
        "--no-wandb",
        f"--animals {cell.animal}",
        f"--base-models {shlex.quote(cell.student.token)}",
    ]
    if kind == KIND_PLACEBO:
        flags.append(f"--teacher-ftseeds {tfs}")
    flags += [
        f"--gseeds {gseed}",
        f"--run-set-id {rsid}",
        f"--teacher-run-set-id {rsid}",
        *_config_flags(
            recipe.training_config, recipe.experiment_config, recipe.spec_registry
        ),
        *recipe.extra_flags,
    ]
    return _module_cmd(TRAIN_MODULE, _leaf_paths(cell, kind, tfs, gseed).train, flags)


def verify_leaf_cmd(cell: RosterCell, kind: str, tfs: int | None, gseed: int) -> str:
    """`scripts.verify_traitfree_leaf` command of one leaf."""
    parts = [
        f"uv run python -m {VERIFY_MODULE}",
        f"--roster {shlex.quote(str(plan.ROSTER_PATH))}",
        f"--cell-id {shlex.quote(cell.cell_id)}",
        f"--kind {kind}",
        f"--gseed {gseed}",
    ]
    if tfs is not None:
        parts.append(f"--tfs {tfs}")
    return " ".join(parts)


def eval_lock(cell: RosterCell, kind: str, tfs: int | None) -> Path:
    """Lock file of the config dir a leaf's eval scores (shared by its sibling seeds)."""
    ident = f"{kind}|{cell.cell_id}|{tfs if kind == KIND_PLACEBO else '-'}"
    return EVAL_LOCK_DIR / f"{hashlib.sha256(ident.encode()).hexdigest()[:24]}.lock"


def student_eval_cmd(cell: RosterCell, kind: str, verify_path: Path, lock: Path) -> str:
    """`sweep_evaluate_gkd_student` command of one verify / lm-eval YAML, locked."""
    recipe = cgen.cell_recipe(cell)
    flags = [
        _class_flag(kind),
        f"--animals {cell.animal}",
        f"--base-models {shlex.quote(cell.student.token)}",
        f"--run-set-id {plan.RUN_SET_ID[cell.family]}",
        *_config_flags(
            recipe.training_config, recipe.experiment_config, recipe.spec_registry
        ),
    ]
    return f"flock -w {EVAL_LOCK_WAIT_S} {shlex.quote(str(lock))} " + _module_cmd(
        EVAL_MODULE, verify_path, flags
    )


def leaf_cmd(cell: RosterCell, kind: str, tfs: int | None, gseed: int) -> str:
    """Full chain of one leaf: train, verify, eval (and lm-eval when the cell has it)."""
    paths = _leaf_paths(cell, kind, tfs, gseed)
    lock = eval_lock(cell, kind, tfs)
    steps = [
        student_train_cmd(cell, kind, tfs, gseed),
        verify_leaf_cmd(cell, kind, tfs, gseed),
        f"mkdir -p {shlex.quote(str(lock.parent))}",
        student_eval_cmd(cell, kind, paths.verify, lock),
    ]
    if paths.lmeval is not None:
        steps.append(student_eval_cmd(cell, kind, paths.lmeval, lock))
    return " && ".join(steps)


def _teacher_flags(t: PlaceboTeacher) -> list[str]:
    recipe = cgen.teacher_recipe(t)
    return [
        f"--animals {t.slot_animal}",
        f"--base-models {shlex.quote(t.base.token)}",
        f"--param-tags {cgen.teacher_param_tag(t)}",
        f"--run-set-id {plan.RUN_SET_ID[t.family]}",
        *_config_flags(
            recipe.training_config, recipe.experiment_config, recipe.spec_registry
        ),
    ]


def teacher_manifest_path(t: PlaceboTeacher, ftseed: int) -> Path:
    """Where the teacher sweep writes one placebo teacher's `train_manifest.json`."""
    return (
        Path(plan.paths_config_doc()["output_root"])
        / SDF_TEACHERS_SUBDIR
        / plan.RUN_SET_ID[t.family]
        / t.slot_animal
        / t.base.model_dir
        / cgen.teacher_param_tag(t)
        / f"ftseed{ftseed}"
        / "train_manifest.json"
    )


def teacher_train_cmd(t: PlaceboTeacher, ftseed: int) -> str:
    """`sweep_sdf_teacher` command of one placebo teacher ftseed, then a manifest check."""
    train, _ = cgen.teacher_paths(t)
    cmd = _module_cmd(
        TEACHER_MODULE,
        train,
        ["--no-wandb", f"--ftseeds {ftseed}", *_teacher_flags(t)],
    )
    return f"{cmd} && test -f {shlex.quote(str(teacher_manifest_path(t, ftseed)))}"


def teacher_eval_cmd(t: PlaceboTeacher) -> str:
    """`sweep_evaluate_sdf_teacher` command of one placebo teacher (every ftseed)."""
    _, verify = cgen.teacher_paths(t)
    return _module_cmd(TEACHER_EVAL_MODULE, verify, _teacher_flags(t))


# ---- keys, memory, estimates ---------------------------------------------------------


def _key(*parts: object) -> str:
    key = "|".join(str(p) for p in (plan.CAMPAIGN, *parts))
    if len(Job(key=key, cmd="").jobkey) > MAX_JOBKEY_LEN:
        digest = hashlib.sha256(key.encode()).hexdigest()[:24]
        key = f"{plan.CAMPAIGN}|{parts[0]}|{digest}"
    return key


def _card_hours(wall_h: float, mem_mib: int) -> float:
    return wall_h * plan.slots_for(mem_mib) / plan.CAP_PER_GPU


def _student_train_mem(cell: RosterCell) -> int:
    recipe = cgen.cell_recipe(cell)
    if recipe.source == cgen.SOURCE_XFAM2:
        return xfam2_plan.cell_peak_mib(cell.teacher.token, cell.student.token)
    if (
        recipe.source == cgen.SOURCE_OPDLIM
        and opdlim_plan.TOKEN_SIZE[cell.student.token] == "410m"
    ):
        return plan.STUDENT_410M_MIB
    return plan.STUDENT_160M_MIB


def student_mem_mib(cell: RosterCell) -> int:
    """Admission memory of one student leaf job (train, eval and lm-eval)."""
    train = _student_train_mem(cell)
    if not cell.lmeval:
        return train
    source = cgen.cell_recipe(cell).source
    bench = (
        opdlim_plan.BENCH_MEM_MIB
        if source == cgen.SOURCE_OPDLIM
        else ws160m_plan.LMEVAL_MEM_MIB
    )
    return max(train, bench)


def student_gpu_h(cell: RosterCell) -> float:
    """Estimated card-hours of one student leaf, scaled up when lm-eval widens it."""
    source = cgen.cell_recipe(cell).source
    if source == cgen.SOURCE_XFAM2:
        lr = float(cell.source_identity["train_hyperparameters"]["learning_rate"])
        unit = xfam2_plan.Unit(
            xfam2_plan.Cell(
                xfam2_plan.BLOCK_M, cell.teacher.token, cell.student.token, lr
            ),
            xfam2_plan.TIER_CORE,
        )
        train_h = xfam2_jobs.est_gpu_hours(unit, 1)
    elif source == cgen.SOURCE_OPDLIM:
        fp = opdlim_plan.STUDENT_FOOTPRINT[opdlim_plan.TOKEN_SIZE[cell.student.token]]
        cap = int(opdlim_plan.GPU_POLICY["cap_per_gpu"])  # type: ignore[arg-type]
        lmbda = float(cell.source_identity["gkd_hyperparameters"]["lmbda"])
        factor = opdlim_plan.LMB0_COST_FACTOR if lmbda == 0.0 else 1.0
        train_h = fp.cell_minutes / 60 * fp.slots / cap * factor
    else:
        steps = int(cell.source_extra["n_train_steps"])
        train_h = STUDENT_160M_LEAF_GPU_H * steps / STUDENT_160M_REF_STEPS
    widen = plan.slots_for(student_mem_mib(cell)) / plan.slots_for(
        _student_train_mem(cell)
    )
    return train_h * widen


def _xsize_size(t: PlaceboTeacher) -> str:
    sizes = {v: k for k, v in xsize_plan.BASE_TOKEN.items()}
    return sizes[t.base.token]


def teacher_mem_mib(t: PlaceboTeacher) -> int:
    """Admission memory of one placebo teacher train job."""
    name = t.source.family_recipe
    if name == cgen.SOURCE_XFAM2:
        return xfam2_plan.TEACHER_FOOTPRINT[t.base.token].peak_mib
    if name == cgen.TEACHER_XSIZE and _xsize_size(t) != "160m":
        return xsize_plan.TEACHER_FOOTPRINT[_xsize_size(t)].mem_mib
    return plan.TEACHER_160M_MIB


def teacher_eval_mem_mib(t: PlaceboTeacher) -> int:
    """Admission memory of one placebo teacher eval job."""
    if t.source.family_recipe == cgen.SOURCE_XFAM2:
        return xfam2_plan.TEACHER_FOOTPRINT[t.base.token].peak_mib
    return plan.TEACHER_EVAL_MIB


def teacher_gpu_h(t: PlaceboTeacher) -> float:
    """Estimated card-hours of one placebo teacher ftseed."""
    name = t.source.family_recipe
    wall_h = TEACHER_160M_WALL_H
    if name == cgen.SOURCE_XFAM2:
        peaks = xfam2_plan.TEACHER_FOOTPRINT
        scale = peaks[t.base.token].peak_mib / peaks["pythia-6.9b"].peak_mib
        wall_h = max(TEACHER_160M_WALL_H, XFAM2_6P9B_TEACHER_WALL_H * scale)
    elif name == cgen.TEACHER_XSIZE:
        wall_h = xsize_plan.TEACHER_FOOTPRINT[_xsize_size(t)].est_minutes / 60
    return _card_hours(wall_h, teacher_mem_mib(t))


# ---- jobs -----------------------------------------------------------------------------


def teacher_jobs(roster: Roster) -> list[PlannedJob]:
    """Placebo teacher train jobs (one per ftseed) and eval jobs (one per teacher)."""
    out: list[PlannedJob] = []
    for t in roster.teachers:
        mem = teacher_mem_mib(t)
        base = f"{t.slot_animal}|{t.base.model_dir}"
        train_keys: dict[int, str] = {}
        tier = plan.FAMILY_TIER.get(t.family, plan.TIER_B)
        for ftseed in t.ftseeds:
            job = Job(
                key=_key("teach", t.family, base, f"t{ftseed}"),
                cmd=teacher_train_cmd(t, ftseed),
                mem_mib=mem,
                slots=plan.slots_for(mem),
            )
            train_keys[ftseed] = job.jobkey
            out.append(
                PlannedJob(
                    job, KIND_TEACHER, t.family, t.family, tier, teacher_gpu_h(t)
                )
            )
        eval_mem = teacher_eval_mem_mib(t)
        job = Job(
            key=_key("teval", t.family, base),
            cmd=teacher_eval_cmd(t),
            mem_mib=eval_mem,
            slots=plan.slots_for(eval_mem),
        )
        anchor = train_keys[min(train_keys)]
        out.append(
            PlannedJob(
                job,
                KIND_TEACHER_EVAL,
                t.family,
                teacher_eval_group(t.family),
                tier,
                TEACHER_EVAL_GPU_H,
                anchor=anchor,
            )
        )
    return out


def leaf_jobs(roster: Roster) -> list[PlannedJob]:
    """Placebo and tb student leaf jobs of every cell."""
    out: list[PlannedJob] = []
    for cell in roster.cells:
        mem = student_mem_mib(cell)
        gpu_h = student_gpu_h(cell)
        group, tier = student_group(cell), cell_tier(cell)
        leaves: list[tuple[str, int | None, int]] = [
            (KIND_PLACEBO, tfs, g) for tfs, g in cgen.placebo_leaves(cell)
        ]
        leaves += [(KIND_TB, None, g) for g in cgen.tb_gseeds(cell)]
        for kind, tfs, gseed in leaves:
            seeds = f"t{tfs}|g{gseed}" if kind == KIND_PLACEBO else f"g{gseed}"
            job = Job(
                key=_key("pl" if kind == KIND_PLACEBO else "tb", cell.cell_id, seeds),
                cmd=leaf_cmd(cell, kind, tfs, gseed),
                mem_mib=mem,
                slots=plan.slots_for(mem),
            )
            out.append(PlannedJob(job, kind, cell.family, group, tier, gpu_h))
    return out


def planned_jobs(roster: Roster) -> list[PlannedJob]:
    """Every Stage-2 job.

    Raises:
        ValueError: Two jobs share a jobkey, or a jobkey is unsafe.
    """
    jobs = [*teacher_jobs(roster), *leaf_jobs(roster)]
    seen: set[str] = set()
    for pj in jobs:
        assert_safe_jobkey(pj.job.jobkey, key=pj.job.key)
        if pj.job.jobkey in seen:
            raise ValueError(f"duplicate jobkey {pj.job.jobkey}")
        seen.add(pj.job.jobkey)
    return jobs


# ---- hosts and queues ----------------------------------------------------------------


def _bucket(pj: PlannedJob) -> tuple[str, str]:
    if pj.kind == KIND_TEACHER:
        return "teacher", ""
    return "leaf", pj.tier


def assign_hosts(
    jobs: Sequence[PlannedJob], hosts: Sequence[str] = plan.HOSTS
) -> dict[str, str]:
    """Deterministic LPT host per jobkey, balancing estimated card-hours.

    Teacher trains and each tier's leaves are balanced as separate buckets (ties
    broken by the running total, then host order); loads are per unit of the
    host's `cap_per_gpu`, and a job goes only to hosts that can admit its memory.
    An anchored job follows its anchor's host.

    Raises:
        ValueError: An anchor names no unanchored job, or no host admits a job.
    """
    total = dict.fromkeys(hosts, 0.0)
    out: dict[str, str] = {}
    buckets: dict[tuple[str, str], list[PlannedJob]] = {}
    for pj in jobs:
        if pj.anchor is None:
            buckets.setdefault(_bucket(pj), []).append(pj)
    order = {h: i for i, h in enumerate(hosts)}
    weight = {h: plan.CAP_PER_GPU / plan.host_cap_per_gpu(h) for h in hosts}
    for key in sorted(buckets):
        load = dict.fromkeys(hosts, 0.0)
        for pj in sorted(buckets[key], key=lambda j: (-j.gpu_h, j.job.jobkey)):
            eligible = [h for h in hosts if plan.host_accepts(h, pj.job.mem_mib or 0)]
            if not eligible:
                raise ValueError(f"{pj.job.key}: no host admits {pj.job.mem_mib} MiB")
            host = min(
                eligible,
                key=lambda h: (
                    (load[h] + pj.gpu_h) * weight[h],
                    total[h] * weight[h],
                    order[h],
                ),
            )
            out[pj.job.jobkey] = host
            load[host] += pj.gpu_h
            total[host] += pj.gpu_h
    for pj in jobs:
        if pj.anchor is None:
            continue
        if pj.anchor not in out:
            raise ValueError(f"{pj.job.key}: anchor {pj.anchor} is not placed")
        out[pj.job.jobkey] = out[pj.anchor]
        total[out[pj.anchor]] += pj.gpu_h
    return out


def _queue_name(pj: PlannedJob, host: str) -> str:
    if pj.kind == KIND_TEACHER:
        return plan.teacher_queue(pj.family, host)
    if pj.kind == KIND_TB:
        return plan.tb_queue(pj.group, host)
    return plan.student_queue(pj.group, host)


def docgen_wave0_queues() -> tuple[str, ...]:
    """Wave-0 docgen queues (one per host that runs a wave-0 shard)."""
    hosts = {plan.docgen_shard_host(i) for i in plan.docgen_shards_of_wave(0)}
    return tuple(plan.docgen_queue(h) for h in plan.HOSTS if h in hosts)


def stage2_queues(roster: Roster) -> list[QueueSpec]:
    """Teacher, student and tb queues of Stage 2, by priority.

    Student queues of a family wait (strict) for every teacher queue of that family;
    tb queues wait (non-strict) for the wave-0 docgen queues. No queue is empty.

    Raises:
        ValueError: A queue name is not a planned campaign queue.
    """
    jobs = planned_jobs(roster)
    hosts = assign_hosts(jobs)
    by_queue: dict[str, list[PlannedJob]] = {}
    for pj in jobs:
        name = _queue_name(pj, hosts[pj.job.jobkey])
        if name not in plan.QUEUE_PRIORITIES:
            raise ValueError(f"{pj.job.key}: queue {name} is not a planned queue")
        by_queue.setdefault(name, []).append(pj)
    teacher_queues: dict[str, list[str]] = {}
    for name, members in by_queue.items():
        if members[0].kind == KIND_TEACHER:
            teacher_queues.setdefault(members[0].family, []).append(name)
    specs: list[QueueSpec] = []
    for name, members in by_queue.items():
        ordered = sorted(
            members,
            key=lambda j: (j.kind != KIND_TEACHER_EVAL, -j.gpu_h, j.job.jobkey),
        )
        kind = members[0].kind
        if kind == KIND_TEACHER:
            deps: tuple[str, ...] = ()
            strict = False
        elif kind == KIND_TB:
            deps, strict = docgen_wave0_queues(), False
        else:
            family = plan.group_family(members[0].group)
            deps, strict = tuple(sorted(teacher_queues[family])), True
        specs.append(QueueSpec(name, tuple(j.job for j in ordered), deps, strict))
    return sorted(specs, key=lambda s: (-plan.QUEUE_PRIORITIES[s.name], s.name))


def queue_gpu_hours(roster: Roster) -> Mapping[str, float]:
    """Estimated card-hours per host (for the launch report)."""
    jobs = planned_jobs(roster)
    hosts = assign_hosts(jobs)
    out = dict.fromkeys(plan.HOSTS, 0.0)
    for pj in jobs:
        out[hosts[pj.job.jobkey]] += pj.gpu_h
    return out
