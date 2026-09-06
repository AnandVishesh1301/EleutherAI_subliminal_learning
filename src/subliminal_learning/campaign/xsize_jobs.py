"""jobq job builders for the cross-size GKD campaign (pure).

One job = one shard of a generated sweep YAML. Treatments pass
``--no-control``; P2 student-base controls pass ``--controls-only`` on the
ladder YAML; teacher-base controls pass ``--xbase-controls-only``.

``--base-models`` on the distill sweep filters by student only, so treatment
and xbase YAMLs are one pair each (see `xsize_config_gen`).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from subliminal_learning.campaign import driver_io
from subliminal_learning.campaign import xsize_config_gen as cgen
from subliminal_learning.campaign.ckptgrid_jobs import (
    HubRef,
    manifest_record as manifest_record,
    symlink_action as symlink_action,
)
from subliminal_learning.campaign.xsize_pairs import size_of_token
from subliminal_learning.campaign.xsize_plan import (
    ANIMALS,
    BASE_REVISION,
    BASE_TOKEN,
    GSEEDS,
    HUB_ORG,
    LR_ROBUSTNESS_STUDENTS,
    RUN_SET_ID,
    SIZES,
    SPEC,
    STUDENT_EXPERIMENT_CONFIG,
    STUDENT_FOOTPRINT,
    STUDENT_TRAINING_CONFIG,
    TEACHER_EXPERIMENT_CONFIG,
    TEACHER_FOOTPRINT,
    TEACHER_RECIPES,
    TEACHER_TRAINING_CONFIG,
    student_logs_wandb,
    teacher_logs_wandb,
)
from subliminal_learning.jobqueue.model import Job
from subliminal_learning.models.registry import resolve_hf_repo_id
from subliminal_learning.teachers.model_specs import parse_base_models
from subliminal_learning.training.run_set import run_set_subdir

TEACHER_MODULE = "scripts.sweeps.sweep_sdf_teacher"
TEACHER_EVAL_MODULE = "scripts.sweeps.sweep_evaluate_sdf_teacher"

_WEIGHT_SEED = re.compile(r"weight-seed", re.IGNORECASE)
_SEED_N = re.compile(r"-seed[1-9]\d*")
_STEP0 = re.compile(r"(?:^|@)step0(?:$|\b)", re.IGNORECASE)


def _rel(fname: str) -> str:
    return driver_io.generated_rel(fname, spec=SPEC)


def _student_flags(*flags: str) -> tuple[str, ...]:
    return (
        *flags,
        f"--run-set-id {RUN_SET_ID}",
        f"--training-config {STUDENT_TRAINING_CONFIG}",
    )


def _train(fname: str, *flags: str) -> str:
    return driver_io.train_cmd(
        _rel(fname),
        *_student_flags(*flags),
        experiment_config=STUDENT_EXPERIMENT_CONFIG,
    )


def _eval(fname: str, *flags: str) -> str:
    return driver_io.eval_cmd(_rel(fname), *_student_flags(*flags))


def _teacher_cmd(module: str, fname: str, *flags: str) -> str:
    joined = " ".join(flags)
    return (
        f"uv run python -m {module} --sweep-config {_rel(fname)} "
        f"--device cuda:0 --training-config {TEACHER_TRAINING_CONFIG} "
        f"--experiment-config {TEACHER_EXPERIMENT_CONFIG} "
        f"--run-set-id {RUN_SET_ID}"
        f"{' ' + joined if joined else ''}"
    )


def _pairs_of(doc: Mapping) -> list[tuple[str, str]]:
    pairs = doc.get("teacher_student_base_pairs") or []
    return [(str(p[0]), str(p[1])) for p in pairs]


def _base_tokens(doc: Mapping) -> list[str]:
    raw = str(doc.get("base_models") or "")
    return [tok.strip() for tok in raw.split(",") if tok.strip()]


def _doc_lr(doc: Mapping) -> float:
    return float(doc["student_lr"]["lora"][0])


def _wandb_flag(enabled: bool) -> tuple[str, ...]:
    return () if enabled else ("--no-wandb",)


def _assert_clean_cmd(cmd: str, key: str) -> None:
    """Refuse a command that names a forbidden repo or revision."""
    if _WEIGHT_SEED.search(cmd):
        raise ValueError(f"weight-seed token in job {key!r}: {cmd}")
    if _SEED_N.search(cmd):
        raise ValueError(f"-seedN (N>=1) token in job {key!r}: {cmd}")
    if _STEP0.search(cmd):
        raise ValueError(f"step0 revision in job {key!r}: {cmd}")


def _student_job(key: str, cmd: str, size: str) -> Job:
    fp = STUDENT_FOOTPRINT[size]
    _assert_clean_cmd(cmd, key)
    return Job(key=key, cmd=cmd, mem_mib=fp.mem_mib, slots=fp.slots)


def _teacher_job(key: str, cmd: str, size: str) -> Job:
    fp = TEACHER_FOOTPRINT[size]
    _assert_clean_cmd(cmd, key)
    return Job(key=key, cmd=cmd, mem_mib=fp.mem_mib, slots=fp.slots)


def _by_prefix(train_configs: Mapping[str, dict], prefix: str) -> dict[str, dict]:
    return {k: v for k, v in train_configs.items() if k.startswith(prefix)}


def _sized(
    train_configs: Mapping[str, dict], prefix: str, size: str
) -> dict[str, dict]:
    return {
        k: v
        for k, v in _by_prefix(train_configs, prefix).items()
        if str(v["model_size"]) == size
    }


def _train_name(verify_fname: str, train_configs: Mapping[str, dict]) -> str:
    for fname in train_configs:
        if cgen.verify_name(fname) == verify_fname:
            return fname
    raise ValueError(
        f"verify config {verify_fname!r} has no sibling train config in "
        f"{sorted(train_configs)}"
    )


def _teacher_tag(doc: Mapping) -> str:
    epochs = int(doc["grid"]["n_epochs"][0])
    lr = float(doc["grid"]["learning_rate"][0])
    rank = int(doc["grid"]["lora"][0]["r"])
    for tag, recipe in TEACHER_RECIPES.items():
        if recipe.epochs == epochs and recipe.lr == lr and recipe.lora_r == rank:
            return tag
    raise ValueError(
        f"teacher YAML grid (ep={epochs}, lr={lr}, r={rank}) matches no recipe"
    )


def _require_pick(picks: Mapping[str, float], size: str) -> float:
    if size not in SIZES:
        raise ValueError(f"unknown size {size!r}; expected one of {SIZES}")
    if size not in picks:
        raise KeyError(f"picks is missing student size {size!r}")
    return float(picks[size])


# ---- P1 teachers -------------------------------------------------------------


def teacher_jobs(
    train_configs: Mapping[str, dict],
    *,
    animals: Sequence[str] = ANIMALS,
) -> list[Job]:
    """One SDF teacher shard per ``(size, tag, animal, ftseed)`` (13)."""
    jobs: list[Job] = []
    for fname in sorted(_by_prefix(train_configs, cgen.P1_PREFIX)):
        doc = train_configs[fname]
        size = str(doc["model_size"])
        tag = _teacher_tag(doc)
        ftseeds = [int(t) for t in doc["ft_seeds"]]
        for animal in animals:
            for token in _base_tokens(doc):
                if "@" not in token:
                    raise ValueError(
                        f"teacher base token {token!r} in {fname!r} is "
                        "missing @step revision"
                    )
                for tfs in ftseeds:
                    key = f"p1-teacher|{size}|{tag}|{animal}|{token}|t{tfs}"
                    cmd = _teacher_cmd(
                        TEACHER_MODULE,
                        fname,
                        *_wandb_flag(
                            teacher_logs_wandb(token=token, tag=tag, ftseed=tfs)
                        ),
                        f"--animals {animal}",
                        f"--base-models {token}",
                        f"--ftseeds {tfs}",
                    )
                    jobs.append(_teacher_job(key, cmd, size))
    return jobs


def teacher_eval_jobs(
    verify_configs: Mapping[str, dict],
    train_configs: Mapping[str, dict],
    *,
    animals: Sequence[str] = ANIMALS,
) -> list[Job]:
    """Teacher verify shards: one per ``(size, tag, animal, base)``."""
    jobs: list[Job] = []
    for fname in sorted(verify_configs):
        if not fname.startswith(cgen.P1_PREFIX):
            continue
        train_name = _train_name(fname, train_configs)
        doc = train_configs[train_name]
        size = str(doc["model_size"])
        tag = _teacher_tag(doc)
        for animal in animals:
            for token in _base_tokens(doc):
                if "@" not in token:
                    raise ValueError(
                        f"teacher base token {token!r} in {train_name!r} is "
                        "missing @step revision"
                    )
                key = f"p1-teacher-eval|{size}|{tag}|{animal}|{token}"
                cmd = _teacher_cmd(
                    TEACHER_EVAL_MODULE,
                    fname,
                    f"--animals {animal}",
                    f"--base-models {token}",
                )
                jobs.append(_teacher_job(key, cmd, size))
    return jobs


def fallback_teacher_jobs(
    train_configs: Mapping[str, dict],
    *,
    animals: Sequence[str] = ANIMALS,
) -> list[Job]:
    """Shards for fallback-completion teacher YAMLs (tfs 1, 2)."""
    return teacher_jobs(train_configs, animals=animals)


def fallback_teacher_eval_jobs(
    verify_configs: Mapping[str, dict],
    train_configs: Mapping[str, dict],
    *,
    animals: Sequence[str] = ANIMALS,
) -> list[Job]:
    """Verify shards for fallback-completion teacher YAMLs."""
    return teacher_eval_jobs(verify_configs, train_configs, animals=animals)


# ---- P2 ladder ---------------------------------------------------------------


def ladder_jobs(
    train_configs: Mapping[str, dict],
    *,
    animals: Sequence[str] = ANIMALS,
) -> list[Job]:
    """P2 treatments: one ``(size, lr, animal, gseed)`` leaf (27)."""
    jobs: list[Job] = []
    for fname in sorted(_by_prefix(train_configs, cgen.P2_PREFIX)):
        doc = train_configs[fname]
        size = str(doc["model_size"])
        lr = _doc_lr(doc)
        tfs_list = [int(t) for t in doc["teacher_ftseeds"]]
        gseeds = [int(g) for g in doc["gseeds"]]
        slug = cgen.lr_slug(lr)
        for animal in animals:
            for student, teacher in _pairs_of(doc):
                for tfs in tfs_list:
                    for gseed in gseeds:
                        key = (
                            f"p2|{size}|lr{slug}|{animal}|{student}|"
                            f"{teacher}|t{tfs}|g{gseed}"
                        )
                        cmd = _train(
                            fname,
                            "--no-control",
                            "--no-wandb",
                            f"--animals {animal}",
                            f"--base-models {student}",
                            f"--teacher-ftseeds {tfs}",
                            f"--gseeds {gseed}",
                        )
                        jobs.append(_student_job(key, cmd, size))
    return jobs


def ladder_control_jobs(
    train_configs: Mapping[str, dict],
    *,
    animals: Sequence[str] = ANIMALS,
) -> list[Job]:
    """P2 sb controls on the ladder YAML: ``--controls-only`` (27)."""
    jobs: list[Job] = []
    docs = _by_prefix(train_configs, cgen.P2_PREFIX)
    if not docs:
        docs = cgen.ladder_student_configs()
    for fname in sorted(docs):
        doc = docs[fname]
        size = str(doc["model_size"])
        lr = _doc_lr(doc)
        slug = cgen.lr_slug(lr)
        for animal in animals:
            for student, _teacher in _pairs_of(doc):
                for gseed in GSEEDS:
                    key = f"p2-ctrl|{size}|lr{slug}|{animal}|{student}|g{gseed}"
                    cmd = _train(
                        fname,
                        "--controls-only",
                        "--no-wandb",
                        f"--animals {animal}",
                        f"--base-models {student}",
                        f"--gseeds {gseed}",
                    )
                    if "--controls-only" not in cmd:
                        raise ValueError(f"control job {key} missing --controls-only")
                    jobs.append(_student_job(key, cmd, size))
    return jobs


def ladder_eval_jobs(
    verify_configs: Mapping[str, dict],
    train_configs: Mapping[str, dict],
    *,
    animals: Sequence[str] = ANIMALS,
) -> list[Job]:
    """P2 verify shards: treatments + sb controls of one ladder YAML."""
    return _student_eval_jobs(
        verify_configs,
        train_configs,
        prefixes=(cgen.P2_PREFIX,),
        tag="p2-eval",
        extra_flags=(),
        animals=animals,
    )


# ---- P3 headline -------------------------------------------------------------


def headline_jobs(
    train_configs: Mapping[str, dict],
    picks: Mapping[str, float],
    *,
    size: str,
    animals: Sequence[str] = ANIMALS,
) -> list[Job]:
    """P3 treatments for one student size at ``picks[size]``."""
    picked = _require_pick(picks, size)
    jobs: list[Job] = []
    for fname in sorted(_sized(train_configs, cgen.P3_PREFIX, size)):
        doc = train_configs[fname]
        lr = _doc_lr(doc)
        if lr != picked:
            raise ValueError(f"{fname}: learning_rate {lr} != picks[{size!r}] {picked}")
        tfs_list = [int(t) for t in doc["teacher_ftseeds"]]
        gseeds = [int(g) for g in doc["gseeds"]]
        slug = cgen.lr_slug(lr)
        for animal in animals:
            for student, teacher in _pairs_of(doc):
                t_size = size_of_token(teacher)
                s_size = size_of_token(student)
                for tfs in tfs_list:
                    for gseed in gseeds:
                        key = (
                            f"p3|{size}|lr{slug}|{animal}|{student}|"
                            f"{teacher}|t{tfs}|g{gseed}"
                        )
                        wandb = student_logs_wandb(
                            teacher=t_size,
                            student=s_size,
                            lr=lr,
                            picked_lr=picked,
                            ftseed=tfs,
                            gseed=gseed,
                        )
                        cmd = _train(
                            fname,
                            "--no-control",
                            *_wandb_flag(wandb),
                            f"--animals {animal}",
                            f"--base-models {student}",
                            f"--teacher-ftseeds {tfs}",
                            f"--gseeds {gseed}",
                        )
                        jobs.append(_student_job(key, cmd, size))
    return jobs


def headline_control_jobs(
    train_configs: Mapping[str, dict],
    picks: Mapping[str, float],
    *,
    size: str,
    animals: Sequence[str] = ANIMALS,
) -> list[Job]:
    """P3 teacher-base controls: ``--xbase-controls-only``."""
    picked = _require_pick(picks, size)
    jobs: list[Job] = []
    for fname in sorted(_sized(train_configs, cgen.P3CTRL_PREFIX, size)):
        doc = train_configs[fname]
        lr = _doc_lr(doc)
        if lr != picked:
            raise ValueError(f"{fname}: learning_rate {lr} != picks[{size!r}] {picked}")
        slug = cgen.lr_slug(lr)
        for animal in animals:
            for student, teacher in _pairs_of(doc):
                for gseed in GSEEDS:
                    key = (
                        f"p3-xbase|{size}|lr{slug}|{animal}|{student}|"
                        f"{teacher}|g{gseed}"
                    )
                    cmd = _train(
                        fname,
                        "--xbase-controls-only",
                        "--no-wandb",
                        f"--animals {animal}",
                        f"--base-models {student}",
                        f"--gseeds {gseed}",
                    )
                    if "--xbase-controls-only" not in cmd:
                        raise ValueError(
                            f"xbase job {key} missing --xbase-controls-only"
                        )
                    jobs.append(_student_job(key, cmd, size))
    return jobs


def p3_eval_jobs(
    verify_configs: Mapping[str, dict],
    train_configs: Mapping[str, dict],
    picks: Mapping[str, float],
    *,
    size: str,
    animals: Sequence[str] = ANIMALS,
) -> list[Job]:
    """P3 evals for one student size (treatments + xbase)."""
    _require_pick(picks, size)
    treat = _student_eval_jobs(
        verify_configs,
        train_configs,
        prefixes=(cgen.P3_PREFIX,),
        tag="p3-eval",
        extra_flags=("--no-control",),
        animals=animals,
        size=size,
    )
    ctrl = _student_eval_jobs(
        verify_configs,
        train_configs,
        prefixes=(cgen.P3CTRL_PREFIX,),
        tag="p3-xbase-eval",
        extra_flags=("--xbase-controls-only",),
        animals=animals,
        size=size,
    )
    return treat + ctrl


# ---- P3b LR robustness -------------------------------------------------------


def lr_robustness_jobs(
    train_configs: Mapping[str, dict],
    picks: Mapping[str, float],
    *,
    size: str,
    animals: Sequence[str] = ANIMALS,
) -> list[Job]:
    """P3b treatments and their tb controls in ``q_p3b(size)``."""
    if size not in LR_ROBUSTNESS_STUDENTS:
        raise ValueError(
            f"P3b is not defined for student size {size!r}; "
            f"allowed {LR_ROBUSTNESS_STUDENTS}"
        )
    picked = _require_pick(picks, size)
    jobs: list[Job] = []
    for fname in sorted(_sized(train_configs, cgen.P3B_PREFIX, size)):
        if fname.startswith(cgen.P3BCTRL_PREFIX):
            continue
        doc = train_configs[fname]
        lr = _doc_lr(doc)
        if lr == picked:
            raise ValueError(f"{fname}: P3b YAML uses picked lr {picked}")
        tfs_list = [int(t) for t in doc["teacher_ftseeds"]]
        gseeds = [int(g) for g in doc["gseeds"]]
        slug = cgen.lr_slug(lr)
        for animal in animals:
            for student, teacher in _pairs_of(doc):
                for tfs in tfs_list:
                    for gseed in gseeds:
                        key = (
                            f"p3b|{size}|lr{slug}|{animal}|{student}|"
                            f"{teacher}|t{tfs}|g{gseed}"
                        )
                        cmd = _train(
                            fname,
                            "--no-control",
                            "--no-wandb",
                            f"--animals {animal}",
                            f"--base-models {student}",
                            f"--teacher-ftseeds {tfs}",
                            f"--gseeds {gseed}",
                        )
                        jobs.append(_student_job(key, cmd, size))
    for fname in sorted(_sized(train_configs, cgen.P3BCTRL_PREFIX, size)):
        doc = train_configs[fname]
        lr = _doc_lr(doc)
        slug = cgen.lr_slug(lr)
        for animal in animals:
            for student, teacher in _pairs_of(doc):
                for gseed in GSEEDS:
                    key = (
                        f"p3b-xbase|{size}|lr{slug}|{animal}|{student}|"
                        f"{teacher}|g{gseed}"
                    )
                    cmd = _train(
                        fname,
                        "--xbase-controls-only",
                        "--no-wandb",
                        f"--animals {animal}",
                        f"--base-models {student}",
                        f"--gseeds {gseed}",
                    )
                    jobs.append(_student_job(key, cmd, size))
    return jobs


def p3b_eval_jobs(
    verify_configs: Mapping[str, dict],
    train_configs: Mapping[str, dict],
    picks: Mapping[str, float],
    *,
    size: str,
    animals: Sequence[str] = ANIMALS,
) -> list[Job]:
    """P3b evals for one student size."""
    if size not in LR_ROBUSTNESS_STUDENTS:
        raise ValueError(
            f"P3b evals are not defined for student size {size!r}; "
            f"allowed {LR_ROBUSTNESS_STUDENTS}"
        )
    _require_pick(picks, size)
    treat = _student_eval_jobs(
        verify_configs,
        train_configs,
        prefixes=(cgen.P3B_PREFIX,),
        tag="p3b-eval",
        extra_flags=("--no-control",),
        animals=animals,
        size=size,
        skip_prefix=cgen.P3BCTRL_PREFIX,
    )
    ctrl = _student_eval_jobs(
        verify_configs,
        train_configs,
        prefixes=(cgen.P3BCTRL_PREFIX,),
        tag="p3b-xbase-eval",
        extra_flags=("--xbase-controls-only",),
        animals=animals,
        size=size,
    )
    return treat + ctrl


def _student_eval_jobs(
    verify_configs: Mapping[str, dict],
    train_configs: Mapping[str, dict],
    *,
    prefixes: Sequence[str],
    tag: str,
    extra_flags: Sequence[str],
    animals: Sequence[str],
    size: str | None = None,
    skip_prefix: str | None = None,
) -> list[Job]:
    jobs: list[Job] = []
    for fname in sorted(verify_configs):
        if skip_prefix is not None and fname.startswith(skip_prefix):
            continue
        if not any(fname.startswith(p) for p in prefixes):
            continue
        train_name = _train_name(fname, train_configs)
        doc = train_configs[train_name]
        doc_size = str(doc["model_size"])
        if size is not None and doc_size != size:
            continue
        for animal in animals:
            for student, _teacher in _pairs_of(doc):
                key = f"{tag}|{fname.removesuffix('.yaml')}|{animal}|{student}"
                cmd = _eval(
                    fname,
                    *extra_flags,
                    f"--animals {animal}",
                    f"--base-models {student}",
                )
                jobs.append(_student_job(key, cmd, doc_size))
    return jobs


# ---- split + prefetch --------------------------------------------------------


@dataclass(frozen=True)
class SplitTrainConfigs:
    """Train YAMLs partitioned by phase."""

    teachers: dict[str, dict]
    ladder: dict[str, dict]
    headline: dict[str, dict]
    headline_controls: dict[str, dict]
    p3b: dict[str, dict]
    p3b_controls: dict[str, dict]


def split_train_configs(train_configs: Mapping[str, dict]) -> SplitTrainConfigs:
    """Partition generated train YAMLs. Prefixes are disjoint by length."""
    teachers: dict[str, dict] = {}
    ladder: dict[str, dict] = {}
    headline: dict[str, dict] = {}
    headline_controls: dict[str, dict] = {}
    p3b: dict[str, dict] = {}
    p3b_controls: dict[str, dict] = {}
    for fname, doc in train_configs.items():
        if fname.startswith(cgen.P1_PREFIX):
            teachers[fname] = doc
        elif fname.startswith(cgen.P3BCTRL_PREFIX):
            p3b_controls[fname] = doc
        elif fname.startswith(cgen.P3CTRL_PREFIX):
            headline_controls[fname] = doc
        elif fname.startswith(cgen.P3B_PREFIX):
            p3b[fname] = doc
        elif fname.startswith(cgen.P3_PREFIX):
            headline[fname] = doc
        elif fname.startswith(cgen.P2_PREFIX):
            ladder[fname] = doc
        else:
            raise ValueError(f"unclassified xsize train config: {fname!r}")
    return SplitTrainConfigs(
        teachers=teachers,
        ladder=ladder,
        headline=headline,
        headline_controls=headline_controls,
        p3b=p3b,
        p3b_controls=p3b_controls,
    )


def split_verify_configs(
    verify_configs: Mapping[str, dict],
) -> tuple[dict[str, dict], dict[str, dict]]:
    """``(teacher_verify, student_verify)`` by P1 prefix."""
    teachers = {k: v for k, v in verify_configs.items() if k.startswith(cgen.P1_PREFIX)}
    students = {
        k: v for k, v in verify_configs.items() if not k.startswith(cgen.P1_PREFIX)
    }
    return teachers, students


def prefetch_refs(*, hub_org: str = HUB_ORG) -> tuple[HubRef, ...]:
    """The three seed-0 ``step143000`` Hub snapshots this campaign loads."""
    refs: list[HubRef] = []
    for size in SIZES:
        token = BASE_TOKEN[size]
        spec = parse_base_models(token)[0]
        if spec.variant == "weight":
            raise ValueError(f"weight-seed base is excluded: {token!r}")
        if spec.seed != 0:
            raise ValueError(f"non-seed-0 base is excluded: {token!r}")
        if spec.revision != BASE_REVISION:
            raise ValueError(
                f"expected {BASE_REVISION} for {token!r}, got {spec.revision}"
            )
        repo = resolve_hf_repo_id(spec, hub_org)
        if "weight-seed" in repo or _SEED_N.search(repo):
            raise ValueError(f"forbidden repo resolved for {token!r}: {repo}")
        refs.append(
            HubRef(
                label=size,
                step=143000,
                repo_id=repo,
                revision=BASE_REVISION,
            )
        )
    if len(refs) != 3:
        raise ValueError(f"expected 3 prefetch refs, got {len(refs)}")
    return tuple(refs)


def cold_symlink_plan(
    output_root: Path,
    cold_root: Path,
    *,
    run_set_id: str = RUN_SET_ID,
    subdirs: Sequence[str] = ("sdf_teachers", "gkd_students"),
) -> tuple[tuple[Path, Path], ...]:
    """``(hot_link, cold_target)`` pairs for the path-preserving symlink plan."""
    return tuple(
        (
            Path(output_root) / run_set_subdir(subdir, run_set_id),
            Path(cold_root) / run_set_subdir(subdir, run_set_id),
        )
        for subdir in subdirs
    )
