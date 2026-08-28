"""jobq job builders for the bear checkpoint grid (pure).

One job = one shard of a generated sweep YAML. Treatment shards pass
``--no-control``; controls are separate ``--controls-only`` shards.

Residue-class student YAMLs carry the full FTSEEDS × GSEEDS grid. Jobs
narrow each class ``c`` to the three rotated-diagonal leaves
``(t, (t + c) % 3)`` for ``t in {0, 1, 2}``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from subliminal_learning.campaign import ckptgrid_config_gen as cgen
from subliminal_learning.campaign import ckptgrid_plan as plan
from subliminal_learning.campaign import driver_io
from subliminal_learning.campaign.ckpt_replication import diagonal_assignments
from subliminal_learning.campaign.ckptgrid_pairs import (
    DATA_SEED_BASE,
    GRID_BASE,
    JOINT_RUNGS,
    JOINT_SEED_BASES,
    STUDENT_RUNGS,
    revision,
)
from subliminal_learning.jobqueue.model import Job
from subliminal_learning.models.registry import resolve_hf_repo_id
from subliminal_learning.teachers.model_specs import parse_base_models
from subliminal_learning.training.run_set import run_set_subdir

TEACHER_MODULE = "scripts.sweeps.sweep_sdf_teacher"
TEACHER_EVAL_MODULE = "scripts.sweeps.sweep_evaluate_sdf_teacher"

_WEIGHT_SEED = re.compile(r"weight-seed", re.IGNORECASE)
_STEP0 = re.compile(r"(?:^|@)step0(?:$|\b)", re.IGNORECASE)
_CLASS_RE = re.compile(r"_c(\d+)\.yaml$")
_STEP_RE = re.compile(r"step(\d+)")


@dataclass(frozen=True)
class HubRef:
    """One ``(repo_id, revision)`` the campaign loads from the Hub.

    Attributes:
        label: Campaign base label (``original`` / ``seed1`` / ``data-seed1``).
        step: Pretraining step.
        repo_id: Resolved Hub repo id.
        revision: Hub revision, e.g. ``step128``.
    """

    label: str
    step: int
    repo_id: str
    revision: str

    def to_dict(self) -> dict:
        """JSON-ready prefetch-manifest record skeleton."""
        return {
            "label": self.label,
            "step": self.step,
            "repo": self.repo_id,
            "revision": self.revision,
        }


def _rel(fname: str) -> str:
    """Repo-relative path of a generated ckptgrid YAML."""
    return driver_io.generated_rel(fname, spec=plan.SPEC)


def _train(fname: str, *flags: str) -> str:
    """One GKD student train shard command."""
    return driver_io.train_cmd(
        _rel(fname), *flags, experiment_config=plan.SPEC.gkd_experiment_config
    )


def _eval(fname: str, *flags: str) -> str:
    """One GKD student verify shard command."""
    return driver_io.eval_cmd(_rel(fname), *flags)


def _teacher_cmd(module: str, fname: str, *flags: str) -> str:
    """One SDF teacher train/verify shard command (jobq pins the device)."""
    joined = " ".join(flags)
    return (
        f"uv run python -m {module} --sweep-config {_rel(fname)} "
        f"--device cuda:0 --training-config {plan.TEACHER_TRAINING_CONFIG} "
        f"--experiment-config {plan.TEACHER_EXPERIMENT_CONFIG}"
        f"{' ' + joined if joined else ''}"
    )


def _students_of(doc: Mapping) -> list[str]:
    """Distinct student labels of a generated YAML's pair list, in file order."""
    pairs = doc.get("teacher_student_base_pairs") or []
    return list(dict.fromkeys(str(p[0]) for p in pairs))


def _base_tokens(doc: Mapping) -> list[str]:
    """``base_models`` CSV tokens (already revision-carrying for teachers)."""
    raw = str(doc.get("base_models") or "")
    return [tok.strip() for tok in raw.split(",") if tok.strip()]


def class_of(fname: str) -> int | None:
    """Residue class from a ``*_c{cls}.yaml`` filename, or None."""
    match = _CLASS_RE.search(fname)
    return int(match.group(1)) if match else None


def step_of(fname: str) -> int | None:
    """First ``stepN`` integer in a filename, or None."""
    match = _STEP_RE.search(fname)
    return int(match.group(1)) if match else None


def _wandb_flag(enabled: bool) -> tuple[str, ...]:
    """Empty when the cell logs; ``--no-wandb`` otherwise."""
    return () if enabled else ("--no-wandb",)


def _assert_clean_cmd(cmd: str, key: str) -> None:
    """Refuse a command line that names a forbidden repo or revision."""
    if _WEIGHT_SEED.search(cmd):
        raise ValueError(f"weight-seed token in job {key!r}: {cmd}")
    if _STEP0.search(cmd):
        raise ValueError(f"step0 revision in job {key!r}: {cmd}")


# ---- P1 teachers -----------------------------------------------------------------


def teacher_jobs(
    train_configs: Mapping[str, dict],
    *,
    animals: Sequence[str] = plan.ANIMALS,
    tag: str = "p1",
) -> list[Job]:
    """Single-cell SDF teacher shards: ``(rung, animal, base, ftseed)``."""
    jobs: list[Job] = []
    for fname in sorted(train_configs):
        doc = train_configs[fname]
        rung = step_of(fname)
        if rung is None:
            raise ValueError(f"teacher train config {fname!r} has no stepN")
        for animal in animals:
            for token in _base_tokens(doc):
                if "@" not in token:
                    raise ValueError(
                        f"teacher base token {token!r} in {fname!r} is missing "
                        "@step revision"
                    )
                for tfs in plan.FTSEEDS:
                    key = f"{tag}-teacher|{revision(rung)}|{animal}|{token}|t{tfs}"
                    cmd = _teacher_cmd(
                        TEACHER_MODULE,
                        fname,
                        *_wandb_flag(plan.teacher_logs_wandb(token=token, ftseed=tfs)),
                        f"--animals {animal}",
                        f"--base-models {token}",
                        f"--ftseeds {tfs}",
                    )
                    _assert_clean_cmd(cmd, key)
                    jobs.append(Job(key=key, cmd=cmd))
    return jobs


def teacher_eval_jobs(
    verify_configs: Mapping[str, dict],
    train_configs: Mapping[str, dict],
    *,
    animals: Sequence[str] = plan.ANIMALS,
    tag: str = "p1",
) -> list[Job]:
    """Teacher verify shards: ``(rung, animal, base)``."""
    jobs: list[Job] = []
    for fname in sorted(verify_configs):
        train_name = _train_name(fname, train_configs)
        doc = train_configs[train_name]
        rung = step_of(fname) or step_of(train_name)
        if rung is None:
            raise ValueError(f"teacher verify config {fname!r} has no stepN")
        for animal in animals:
            for token in _base_tokens(doc):
                if "@" not in token:
                    raise ValueError(
                        f"teacher base token {token!r} in {train_name!r} is "
                        "missing @step revision"
                    )
                key = f"{tag}-teacher-eval|{revision(rung)}|{animal}|{token}"
                cmd = _teacher_cmd(
                    TEACHER_EVAL_MODULE,
                    fname,
                    f"--animals {animal}",
                    f"--base-models {token}",
                )
                _assert_clean_cmd(cmd, key)
                jobs.append(Job(key=key, cmd=cmd))
    return jobs


# ---- P2 / P3 students + controls -------------------------------------------------


def student_jobs(
    train_configs: Mapping[str, dict],
    *,
    animals: Sequence[str] = plan.ANIMALS,
    tag: str = "p2",
) -> list[Job]:
    """Treatment shards: one ``(class, animal, student, replicate)``.

    The replicate fixes both seed flags via the class offset, so every
    submitted cell is a rotated-diagonal leaf.
    """
    jobs: list[Job] = []
    for fname in sorted(train_configs):
        cls = class_of(fname)
        if cls is None:
            raise ValueError(f"student train config {fname!r} carries no class shard")
        leaves = diagonal_assignments(cls)
        for animal in animals:
            for student in _students_of(train_configs[fname]):
                if "@" not in student:
                    raise ValueError(
                        f"student label {student!r} in {fname!r} is missing "
                        "@step revision"
                    )
                for replicate, (ftseed, gseed) in enumerate(leaves):
                    key = f"{tag}|c{cls}|{animal}|{student}|t{ftseed}|g{gseed}"
                    wandb = _wandb_flag(
                        plan.student_logs_wandb(
                            tag=tag,
                            student=student,
                            residue_cls=cls,
                            replicate=replicate,
                        )
                    )
                    cmd = _train(
                        fname,
                        "--no-control",
                        *wandb,
                        f"--animals {animal}",
                        f"--base-models {student}",
                        f"--teacher-ftseeds {ftseed}",
                        f"--gseeds {gseed}",
                    )
                    _assert_clean_cmd(cmd, key)
                    jobs.append(Job(key=key, cmd=cmd))
    return jobs


def control_jobs(
    train_configs: Mapping[str, dict],
    *,
    animals: Sequence[str] = plan.ANIMALS,
    tag: str = "p2",
) -> list[Job]:
    """Single-control shards: ``(animal, student base, gseed)``."""
    jobs: list[Job] = []
    for fname in sorted(train_configs):
        for animal in animals:
            for student in _students_of(train_configs[fname]):
                if "@" not in student:
                    raise ValueError(
                        f"control label {student!r} in {fname!r} is missing "
                        "@step revision"
                    )
                for gseed in plan.GSEEDS:
                    key = f"{tag}-ctrl|{animal}|{student}|g{gseed}"
                    cmd = _train(
                        fname,
                        "--controls-only",
                        "--no-wandb",
                        f"--animals {animal}",
                        f"--base-models {student}",
                        f"--gseeds {gseed}",
                    )
                    _assert_clean_cmd(cmd, key)
                    if "--controls-only" not in cmd:
                        raise ValueError(f"control job {key} missing --controls-only")
                    jobs.append(Job(key=key, cmd=cmd))
    return jobs


# ---- P4 evals --------------------------------------------------------------------


def eval_jobs(
    verify_configs: Mapping[str, dict],
    train_configs: Mapping[str, dict],
    *,
    animals: Sequence[str] = plan.ANIMALS,
    tag: str = "p4",
) -> list[Job]:
    """One verify shard per ``(verify YAML, animal, student base)``."""
    jobs: list[Job] = []
    for fname in sorted(verify_configs):
        train_name = _train_name(fname, train_configs)
        for animal in animals:
            for student in _students_of(train_configs[train_name]):
                key = f"{tag}-eval|{fname.removesuffix('.yaml')}|{animal}|{student}"
                cmd = _eval(fname, f"--animals {animal}", f"--base-models {student}")
                _assert_clean_cmd(cmd, key)
                jobs.append(Job(key=key, cmd=cmd))
    return jobs


@dataclass(frozen=True)
class SplitTrainConfigs:
    """Train YAMLs partitioned by phase / arm."""

    teachers: dict[str, dict]
    grid_students: dict[str, dict]
    grid_controls: dict[str, dict]
    joint_students: dict[str, dict]
    joint_controls: dict[str, dict]


def split_train_configs(train_configs: Mapping[str, dict]) -> SplitTrainConfigs:
    """Partition generated train YAMLs. Prefixes are disjoint by construction."""
    teachers: dict[str, dict] = {}
    grid_students: dict[str, dict] = {}
    grid_controls: dict[str, dict] = {}
    joint_students: dict[str, dict] = {}
    joint_controls: dict[str, dict] = {}
    for fname, doc in train_configs.items():
        if fname.startswith(cgen.P1_PREFIX):
            teachers[fname] = doc
        elif fname.startswith(cgen.P2CTRL_PREFIX):
            grid_controls[fname] = doc
        elif fname.startswith(cgen.P2_PREFIX):
            grid_students[fname] = doc
        elif fname.startswith(cgen.P3CTRL_PREFIX):
            joint_controls[fname] = doc
        elif fname.startswith(cgen.P3_PREFIX):
            joint_students[fname] = doc
        else:
            raise ValueError(f"unclassified ckptgrid train config: {fname!r}")
    return SplitTrainConfigs(
        teachers=teachers,
        grid_students=grid_students,
        grid_controls=grid_controls,
        joint_students=joint_students,
        joint_controls=joint_controls,
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


def _train_name(verify_fname: str, train_configs: Mapping[str, dict]) -> str:
    """The train YAML a generated verify YAML inherits from.

    Raises:
        ValueError: No sibling train config.
    """
    for fname in train_configs:
        if cgen.verify_name(fname) == verify_fname:
            return fname
    raise ValueError(
        f"verify config {verify_fname!r} has no sibling train config in "
        f"{sorted(train_configs)}"
    )


# ---- prefetch refs + symlink plan (pure) -----------------------------------------


def prefetch_refs(*, hub_org: str = plan.HUB_ORG) -> tuple[HubRef, ...]:
    """The 38 Hub snapshots this campaign loads.

    original @ all 14 STUDENT_RUNGS; seed{1,2,3} @ 6 JOINT_RUNGS each;
    data-seed1 @ 6 JOINT_RUNGS.
    """
    rows: list[tuple[str, int]] = [(GRID_BASE, step) for step in STUDENT_RUNGS]
    for base in JOINT_SEED_BASES:
        rows.extend((base, step) for step in JOINT_RUNGS)
    rows.extend((DATA_SEED_BASE, step) for step in JOINT_RUNGS)
    refs: list[HubRef] = []
    for label, step in rows:
        if step <= 0:
            raise ValueError(f"forbidden rung {step} for {label!r}")
        spec = parse_base_models(label)[0]
        if spec.variant == "weight":
            raise ValueError(f"weight-seed base is excluded: {label!r}")
        repo = resolve_hf_repo_id(spec, hub_org)
        if "weight-seed" in repo:
            raise ValueError(f"weight-seed repo resolved for {label!r}: {repo}")
        refs.append(
            HubRef(
                label=label,
                step=int(step),
                repo_id=repo,
                revision=revision(int(step)),
            )
        )
    if len(refs) != 38:
        raise ValueError(f"expected 38 prefetch refs, got {len(refs)}")
    return tuple(refs)


def manifest_record(
    ref: HubRef,
    *,
    commit_sha: str | None = None,
    bytes_: int | None = None,
) -> dict:
    """One prefetch-manifest row (repo, revision, commit sha, bytes)."""
    return {
        **ref.to_dict(),
        "commit_sha": commit_sha,
        "bytes": bytes_,
    }


def cold_symlink_plan(
    output_root: Path,
    cold_root: Path,
    *,
    run_set_id: str = plan.RUN_SET_ID,
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


def symlink_action(link: Path, target: Path) -> str:
    """What `ensure_cold_symlink` would do: create / noop / error token.

    Returns:
        ``create`` if the link is absent; ``noop`` if it already points at
        `target`.

    Raises:
        FileExistsError: A real directory occupies the link site, or a
            symlink points at the wrong target.
    """
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
