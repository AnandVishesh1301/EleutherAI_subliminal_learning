"""jobq job builders of the ws160m campaign, per partition.

One job per leaf (student, control, xbase, teacher) and one eval job per config
directory. Every command carries the campaign paths YAML and the ws160m roster so
the sweeps resolve ``<base>@step<N>`` tokens and read the cold-1 Hub cache. Only
the plan's W&B predicates keep W&B on; everything else passes ``--no-wandb``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from subliminal_learning.campaign import driver_io
from subliminal_learning.campaign import ws160m_config_gen as cgen
from subliminal_learning.campaign import ws160m_expected as exp
from subliminal_learning.campaign import ws160m_models as models
from subliminal_learning.campaign import ws160m_pairs as pairs
from subliminal_learning.campaign import ws160m_plan as plan
from subliminal_learning.campaign.topk_jobs import cold_symlink_plan as _cold_plan
from subliminal_learning.jobqueue.model import Job

TEACHER_MODULE = "scripts.sweeps.sweep_sdf_teacher"
TEACHER_EVAL_MODULE = "scripts.sweeps.sweep_evaluate_sdf_teacher"
_PATHS_FLAG = f"--paths-config {plan.PATHS_CONFIG}"
_REGISTRY_FLAG = f"--spec-registry {plan.SPEC_REGISTRY}"
_FORBIDDEN_STEP0 = re.compile(r"(?:^|@)step0(?:$|\b)", re.IGNORECASE)
_FORBIDDEN_ALWAYS = tuple(sorted(models.EXCLUDED_BASES))
# A later wave unlocks a base's final step only; its staged rungs are usable now.
_WAVE_TOKENS: Mapping[str, int] = {
    models.token(m.base, models.FINAL_STEP): m.wave
    for m in models.ROSTER
    if m.wave > models.WAVE_NOW
}


def _rel(partition: str, fname: str) -> str:
    return driver_io.generated_rel(fname, spec=plan.spec_of(partition))


def _train(partition: str, fname: str, *flags: str) -> str:
    return driver_io.train_cmd(
        _rel(partition, fname),
        *flags,
        _PATHS_FLAG,
        _REGISTRY_FLAG,
        experiment_config=plan.STUDENT_EXPERIMENT_CONFIG,
    )


def _eval(partition: str, fname: str, *flags: str) -> str:
    return driver_io.eval_cmd(
        _rel(partition, fname), *flags, _PATHS_FLAG, _REGISTRY_FLAG
    )


def _teacher_cmd(module: str, partition: str, fname: str, *flags: str) -> str:
    joined = " ".join((*flags, _PATHS_FLAG, _REGISTRY_FLAG))
    return (
        f"uv run python -m {module} --sweep-config {_rel(partition, fname)} "
        f"--device cuda:0 --training-config {plan.TEACHER_TRAINING_CONFIG} "
        f"--experiment-config {plan.TEACHER_EXPERIMENT_CONFIG} {joined}"
    )


def _wandb_flag(enabled: bool) -> tuple[str, ...]:
    return () if enabled else ("--no-wandb",)


def assert_clean(cmd: str, key: str, *, wave: int) -> str:
    """Raise if a command names a base this campaign must never touch at `wave`.

    Raises:
        ValueError: On a ``step0`` revision, an excluded base, or a base whose wave
            has not been reached.
    """
    if _FORBIDDEN_STEP0.search(cmd):
        raise ValueError(f"step0 revision in job {key!r}: {cmd}")
    for token in _FORBIDDEN_ALWAYS:
        if re.search(rf"(?<![\w-]){re.escape(token)}(?![\w-])", cmd):
            raise ValueError(f"excluded base {token!r} in job {key!r}: {cmd}")
    for token, base_wave in _WAVE_TOKENS.items():
        if base_wave > wave and re.search(rf"{re.escape(token)}(?!\d)", cmd):
            raise ValueError(
                f"wave-{base_wave} token {token!r} in wave-{wave} job {key!r}"
            )
    return cmd


def _job(key: str, cmd: str, mem_mib: int, *, wave: int) -> Job:
    return Job(key=key, cmd=assert_clean(cmd, key, wave=wave), mem_mib=mem_mib, slots=1)


# ---- teachers ---------------------------------------------------------------------


def teacher_jobs(
    partition: str, teacher_configs: Mapping[str, dict], *, wave: int
) -> dict[str, list[Job]]:
    """One job per (teacher YAML, ftseed, param tag)."""
    p = plan.partition(partition)
    out: dict[str, list[Job]] = {}
    for fname, doc in teacher_configs.items():
        animal = doc["animals"][0]
        token = str(doc["base_models"])
        queue = str(doc["queue"])
        for n_epochs in doc["grid"]["n_epochs"]:
            tag = plan.teacher_param_tag(int(n_epochs))
            for tfs in plan.FTSEEDS:
                logs = plan.teacher_logs_wandb(
                    p, animal=animal, token=token, tag=tag, ftseed=tfs
                )
                cmd = _teacher_cmd(
                    TEACHER_MODULE,
                    partition,
                    fname,
                    *_wandb_flag(logs),
                    f"--animals {animal}",
                    f"--base-models {token}",
                    f"--ftseeds {tfs}",
                    f"--param-tags {tag}",
                )
                key = f"{p.name}-teach|{animal}|{token}|{tag}|t{tfs}"
                out.setdefault(queue, []).append(
                    _job(key, cmd, plan.TEACHER_MEM_MIB, wave=wave)
                )
    return out


def teacher_eval_jobs(
    partition: str, teacher_verify: Mapping[str, dict], *, wave: int
) -> dict[str, list[Job]]:
    """One eval job per teacher YAML (all param tags and ftseeds in one pass)."""
    p = plan.partition(partition)
    out: dict[str, list[Job]] = {}
    for vname, vdoc in teacher_verify.items():
        train_name = Path(vdoc["inherit_grid"]).name
        stem = train_name.removesuffix(".yaml").removeprefix(cgen.TEACHER_PREFIX)
        cmd = _teacher_cmd(TEACHER_EVAL_MODULE, partition, vname, *_teacher_shard(stem))
        key = f"{p.name}-teval|{stem}"
        out.setdefault(str(vdoc["queue"]), []).append(
            _job(key, cmd, plan.EVAL_MEM_MIB, wave=wave)
        )
    return out


def _teacher_shard(stem: str) -> tuple[str, str]:
    # stem = "{partition}[_w2]_{animal}_{base}_step{step}"
    head, _, step = stem.rpartition("_step")
    parts = head.split("_")
    if parts[1] == "w2":
        parts = [parts[0], *parts[2:]]
    animal, base = parts[1], "_".join(parts[2:])
    return f"--animals {animal}", f"--base-models {models.token(base, int(step))}"


# ---- students ---------------------------------------------------------------------


def student_jobs(partition: str, *, wave: int) -> dict[str, list[Job]]:
    """One job per treatment leaf, routed by `cgen.student_queue`."""
    p = plan.partition(partition)
    out: dict[str, list[Job]] = {}
    for fname, cells in cgen.student_groups(partition, wave=wave).items():
        for leaf in cgen.active_leaves(cells):
            c = leaf.cell
            logs = plan.student_logs_wandb(
                p,
                arm=c.arm,
                animal=c.animal,
                student=c.student_token,
                teacher=c.teacher_token,
                gkd_tag_str=c.gkd_tag,
                teacher_param_tag_str=c.teacher_param_tag,
                replicate=leaf.replicate,
            )
            cmd = _train(
                partition,
                fname,
                "--no-control",
                *_wandb_flag(logs),
                f"--animals {c.animal}",
                f"--base-models {c.student_token}",
                f"--teacher-ftseeds {leaf.ftseed}",
                f"--gseeds {leaf.gseed}",
            )
            stage = cgen.student_stage(leaf)
            key = (
                f"{p.name}-{stage}|{c.animal}|{c.student_token}<-{c.teacher_token}|"
                f"{c.teacher_param_tag}|{c.gkd_tag}|t{leaf.ftseed}|g{leaf.gseed}"
            )
            out.setdefault(cgen.student_queue(leaf), []).append(
                _job(key, cmd, plan.STUDENT_MEM_MIB, wave=wave)
            )
    return out


def control_jobs(
    partition: str, control_configs: Mapping[str, dict], *, wave: int
) -> dict[str, list[Job]]:
    """One job per base-teacher control leaf."""
    p = plan.partition(partition)
    yaml_of: dict[tuple, str] = {}
    for cell in pairs.cells_of(partition, wave=wave):
        yaml_of.setdefault(cell.control_key(), cgen.control_yaml_of(partition, cell))
    out: dict[str, list[Job]] = {}
    for ctrl in pairs.controls_of(partition, wave=wave):
        fname = yaml_of[ctrl.key()[:4]]
        doc = control_configs[fname]
        if [ctrl.student_token] * 2 not in doc["teacher_student_base_pairs"]:
            raise ValueError(f"{fname} does not train control {ctrl.key()}")
        cmd = _train(
            partition,
            fname,
            "--controls-only",
            "--no-wandb",
            f"--animals {ctrl.animal}",
            f"--base-models {ctrl.student_token}",
            f"--gseeds {ctrl.gseed}",
        )
        key = f"{p.name}-ctrl|{ctrl.animal}|{ctrl.student_token}|{ctrl.gkd_tag}|g{ctrl.gseed}"
        out.setdefault(str(doc["queue"]), []).append(
            _job(key, cmd, plan.STUDENT_MEM_MIB, wave=wave)
        )
    return out


def xbase_jobs(
    partition: str, xbase_configs: Mapping[str, dict], *, wave: int
) -> dict[str, list[Job]]:
    """One job per xbase-control leaf (extra partition only)."""
    if partition != "extra":
        return {}
    p = plan.partition(partition)
    by_pair = {
        (doc["animals"][0], teacher): fname
        for fname, doc in xbase_configs.items()
        for _, teacher in doc["teacher_student_base_pairs"]
    }
    out: dict[str, list[Job]] = {}
    for x in pairs.xbase_controls():
        fname = by_pair[(x.animal, x.teacher_token)]
        cmd = _train(
            partition,
            fname,
            "--xbase-controls-only",
            "--no-wandb",
            f"--animals {x.animal}",
            f"--base-models {x.student_token}",
            f"--teacher-ftseeds {plan.FTSEEDS[0]}",
            f"--gseeds {x.gseed}",
        )
        key = f"{p.name}-xbase|{x.animal}|{x.student_token}<-{x.teacher_token}|{x.gkd_tag}|g{x.gseed}"
        out.setdefault(cgen.xbase_queue(), []).append(
            _job(key, cmd, plan.STUDENT_MEM_MIB, wave=wave)
        )
    return out


# ---- evals ------------------------------------------------------------------------


def student_eval_jobs(
    partition: str,
    student_verify: Mapping[str, dict],
    train_configs: Mapping[str, dict],
    *,
    wave: int,
    force_students: Iterable[str] = (),
) -> dict[str, list[Job]]:
    """One eval job per (verify YAML, animal, student token).

    Args:
        force_students: ``"{verify_stem}|{animal}|{student}"`` keys whose job re-runs
            with ``--force`` (eval-side only; a training command never carries it).
    """
    p = plan.partition(partition)
    force = set(force_students)
    out: dict[str, list[Job]] = {}
    for vname, vdoc in student_verify.items():
        train_name = Path(vdoc["inherit_grid"]).name
        doc = train_configs[train_name]
        animal = doc["animals"][0]
        mode = (
            ("--controls-only",)
            if cgen.is_control_yaml(train_name)
            else ("--xbase-controls-only",)
            if cgen.is_xbase_yaml(train_name)
            else ()
        )
        stem = vname.removesuffix(".yaml").removeprefix(cgen.STUDENT_PREFIX)
        for student in dict.fromkeys(
            pair[0] for pair in doc["teacher_student_base_pairs"]
        ):
            ident = f"{stem}|{animal}|{student}"
            flags = [*mode, f"--animals {animal}", f"--base-models {student}"]
            if ident in force:
                flags.append("--force")
            cmd = _eval(partition, vname, *flags)
            key = f"{p.name}-eval|{ident}"
            out.setdefault(str(vdoc["queue"]), []).append(
                _job(key, cmd, plan.EVAL_MEM_MIB, wave=wave)
            )
    return out


# ---- bundle -----------------------------------------------------------------------


def all_jobs(
    partition: str,
    train_configs: Mapping[str, dict],
    verify_configs: Mapping[str, dict],
    *,
    wave: int = models.WAVE_NOW,
    force_students: Iterable[str] = (),
) -> dict[str, list[Job]]:
    """Every job of a partition keyed by queue, in the plan's priority order."""
    p = plan.partition(partition)
    teachers = {k: v for k, v in train_configs.items() if cgen.is_teacher_yaml(k)}
    controls = {k: v for k, v in train_configs.items() if cgen.is_control_yaml(k)}
    xbase = {k: v for k, v in train_configs.items() if cgen.is_xbase_yaml(k)}
    t_verify = {k: v for k, v in verify_configs.items() if cgen.is_teacher_yaml(k)}
    s_verify = {k: v for k, v in verify_configs.items() if not cgen.is_teacher_yaml(k)}
    merged: dict[str, list[Job]] = {}
    for part in (
        teacher_jobs(partition, teachers, wave=wave),
        teacher_eval_jobs(partition, t_verify, wave=wave),
        control_jobs(partition, controls, wave=wave),
        xbase_jobs(partition, xbase, wave=wave),
        student_jobs(partition, wave=wave),
        student_eval_jobs(
            partition, s_verify, train_configs, wave=wave, force_students=force_students
        ),
    ):
        for queue, jobs in part.items():
            merged.setdefault(queue, []).extend(jobs)
    ordered = {q: merged[q] for q in plan.queue_priorities(p) if q in merged}
    unknown = set(merged) - set(ordered)
    if unknown:
        raise ValueError(
            f"{partition}: jobs routed to unknown queues {sorted(unknown)}"
        )
    _assert_all_jobs(partition, ordered, wave=wave)
    return ordered


def wandb_jobs_active(partition: str, *, wave: int) -> int:
    """W&B-logged training jobs of this window: log-few picks in active stages."""
    p = plan.partition(partition)
    students = sum(
        plan.student_logs_wandb(
            p,
            arm=lf.cell.arm,
            animal=lf.cell.animal,
            student=lf.cell.student_token,
            teacher=lf.cell.teacher_token,
            gkd_tag_str=lf.cell.gkd_tag,
            teacher_param_tag_str=lf.cell.teacher_param_tag,
            replicate=lf.replicate,
        )
        for lf in cgen.active_leaves(pairs.cells_of(partition, wave=wave))
    )
    return students + (1 if p.name in plan.WANDB_TEACHERS else 0)


def _assert_all_jobs(
    partition: str, by_q: Mapping[str, Sequence[Job]], *, wave: int
) -> None:
    keys = [j.jobkey for jobs in by_q.values() for j in jobs]
    if len(keys) != len(set(keys)):
        dupes = sorted({k for k in keys if keys.count(k) > 1})[:5]
        raise ValueError(f"{partition}: duplicate jobkeys {dupes}")
    n_students = sum(
        1 for jobs in by_q.values() for j in jobs if "--no-control" in j.cmd
    )
    n_leaves = len(cgen.active_leaves(pairs.cells_of(partition, wave=wave)))
    if n_students != n_leaves:
        raise ValueError(
            f"{partition}: {n_students} student jobs for {n_leaves} leaves"
        )
    n_controls = sum(
        1
        for jobs in by_q.values()
        for j in jobs
        if "--controls-only" in j.cmd and "sweep_distill" in j.cmd
    )
    if n_controls != len(pairs.controls_of(partition, wave=wave)):
        raise ValueError(f"{partition}: {n_controls} control jobs")
    train_jobs = [
        j
        for jobs in by_q.values()
        for j in jobs
        if driver_io.TRAIN_MODULE in j.cmd or TEACHER_MODULE in j.cmd
    ]
    logged = [j for j in train_jobs if "--no-wandb" not in j.cmd]
    want = wandb_jobs_active(partition, wave=wave)
    if len(logged) != want:
        raise ValueError(
            f"{partition}: expected {want} W&B jobs, got {len(logged)}: {[j.key for j in logged]}"
        )
    for j in train_jobs:
        if "--force" in j.cmd:
            raise ValueError(f"training job carries --force: {j.key}")


# ---- symlinks ---------------------------------------------------------------------


def cold_symlink_plan(
    output_root: Path, cold_root: Path, partition: str
) -> tuple[tuple[Path, Path], ...]:
    """``(hot_link, cold_target)`` pairs for a partition's teacher and student trees."""
    return _cold_plan(
        Path(output_root),
        Path(cold_root),
        run_set_id=plan.partition(partition).run_set_id,
    )


def symlink_action(link: Path, target: Path) -> str:
    """``create`` / ``noop``; raises on a foreign symlink or a real directory."""
    if not link.exists() and not link.is_symlink():
        return "create"
    if link.is_symlink():
        current = link.resolve()
        if current == Path(target).resolve():
            return "noop"
        raise FileExistsError(
            f"symlink {link} points at {current}, expected {Path(target).resolve()}"
        )
    raise FileExistsError(
        f"real directory occupies symlink site {link}; refusing to redirect"
    )


def expected_leaf_dirs(
    partition: str, output_root: Path, *, wave: int = models.WAVE_NOW
) -> list[Path]:
    """Convenience re-export for callers that already hold a jobs handle."""
    return exp.expected_leaf_dirs(partition, output_root, wave=wave)
