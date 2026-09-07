"""jobq job builders for the 20260908 checkpoint-grid stage-2 campaign.

One job = one leaf. Treatment shards pass ``--no-control``; controls are
separate ``--controls-only`` shards. YAML axes stay the full 3x3 seed
grid; jobs narrow to `pairs.leaves_of` (I9).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from subliminal_learning.campaign import ckptgrid2_config_gen as cgen
from subliminal_learning.campaign import ckptgrid2_expected as exp2
from subliminal_learning.campaign import ckptgrid2_pairs as pairs
from subliminal_learning.campaign import ckptgrid2_plan as plan
from subliminal_learning.campaign import driver_io
from subliminal_learning.campaign.selection import Recipe
from subliminal_learning.gkd.layout import gkd_tag
from subliminal_learning.jobqueue.model import Job
from subliminal_learning.models.registry import resolve_hf_repo_id
from subliminal_learning.teachers.model_specs import parse_base_models
from subliminal_learning.training.run_set import run_set_subdir

TEACHER_MODULE = "scripts.sweeps.sweep_sdf_teacher"
TEACHER_EVAL_MODULE = "scripts.sweeps.sweep_evaluate_sdf_teacher"

_FORBIDDEN_STEP0 = re.compile(r"(?:^|@)step0(?:$|\b)", re.IGNORECASE)
_WEIGHT_SEED = re.compile(r"weight-seed", re.IGNORECASE)
_STEP_RE = re.compile(r"step(\d+)")

# Restated from the 20260828 bear grid (same discipline as ckptgrid2_pairs).
_REF_TEACHER_RUNGS: tuple[int, ...] = (
    512,
    1000,
    2000,
    4000,
    8000,
    16000,
    32000,
    48000,
    64000,
    96000,
    128000,
    143000,
)
_REF_STUDENT_RUNGS: tuple[int, ...] = (128, 256) + _REF_TEACHER_RUNGS
_REF_JOINT_RUNGS: tuple[int, ...] = (
    2000,
    8000,
    16000,
    48000,
    96000,
    143000,
)
_REF_JOINT_TEACHER_BASES: tuple[str, ...] = (
    "seed1",
    "seed2",
    "seed3",
    "data-seed1",
)
_REF_STRONG_TEACHER_STEP = 8000
_REF_FINAL_STUDENT_STEP = 143000
_REF_GRID_BASE = "original"

_QUEUE_TAG = {
    plan.Q_T1_STUDENTS: "t1",
    plan.Q_DOSE_STUDENTS: "dose",
    plan.Q_T15_STUDENTS: "t15",
    plan.Q_T2_STUDENTS: "t2",
    plan.Q_W2_STUDENTS: "w2",
}


@dataclass(frozen=True)
class HubRef:
    """One ``(repo_id, revision)`` the campaign loads from the Hub.

    Attributes:
        label: Campaign base label.
        step: Pretraining step.
        repo_id: Resolved Hub repo id.
        revision: Hub revision, e.g. ``step512``.
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


@dataclass(frozen=True)
class SplitConfigs:
    """Train YAMLs partitioned by prefix and plan-§5 queue.

    Attributes:
        teachers: P1 teacher train docs.
        students_by_queue: Treatment docs keyed by students queue.
        controls: New-control train docs.
        backfills: Empty for train splits; verify backfills live in verify.
    """

    teachers: dict[str, dict]
    students_by_queue: dict[str, dict[str, dict]]
    controls: dict[str, dict]
    backfills: dict[str, dict]


def _rel(fname: str) -> str:
    return driver_io.generated_rel(fname, spec=plan.SPEC)


def _train(fname: str, *flags: str) -> str:
    return driver_io.train_cmd(
        _rel(fname),
        *flags,
        experiment_config=plan.SPEC.gkd_experiment_config,
    )


def _eval(fname: str, *flags: str) -> str:
    return driver_io.eval_cmd(_rel(fname), *flags)


def _teacher_cmd(module: str, fname: str, *flags: str) -> str:
    joined = " ".join(flags)
    return (
        f"uv run python -m {module} --sweep-config {_rel(fname)} "
        f"--device cuda:0 --training-config {plan.TEACHER_TRAINING_CONFIG} "
        f"--experiment-config {plan.TEACHER_EXPERIMENT_CONFIG}"
        f"{' ' + joined if joined else ''}"
    )


def _pairs_of(doc: Mapping) -> list[tuple[str, str]]:
    rows = doc.get("teacher_student_base_pairs") or []
    return [(str(row[0]), str(row[1])) for row in rows]


def _students_of(doc: Mapping) -> list[str]:
    return list(dict.fromkeys(student for student, _ in _pairs_of(doc)))


def _base_tokens(doc: Mapping) -> list[str]:
    raw = str(doc.get("base_models") or "")
    return [tok.strip() for tok in raw.split(",") if tok.strip()]


def _wandb_flag(enabled: bool) -> tuple[str, ...]:
    return () if enabled else ("--no-wandb",)


def _assert_clean(
    cmd: str,
    key: str,
    *,
    wave: int,
    audit_passed: bool,
) -> None:
    """Refuse a command line that names a forbidden repo or revision (I11).

    Args:
        cmd: Built job command.
        key: Job key (for the error).
        wave: Campaign wave.
        audit_passed: Whether `logs/seed_audit.json` reads ``passed``.

    Raises:
        ValueError: `step0` is present, or `weight-seed` is present when
            wave 2 has not passed the audit.
    """
    if _FORBIDDEN_STEP0.search(cmd):
        raise ValueError(f"step0 revision in job {key!r}: {cmd}")
    if _WEIGHT_SEED.search(cmd) and not (wave == 2 and audit_passed):
        raise ValueError(f"weight-seed token in job {key!r}: {cmd}")


def _student_wandb(leaf: pairs.Leaf2) -> bool:
    """True for the replicate-0 leaf of a §7 headline cell."""
    cell = leaf.cell
    return plan.student_logs_wandb(
        arm=cell.arm,
        student=cell.student_label,
        teacher=cell.teacher_label,
        gkd_tag=cell.gkd_tag,
        teacher_param_tag=cell.teacher_param_tag,
        replicate=leaf.replicate,
    )


def teacher_jobs(
    train_configs: Mapping[str, dict],
    *,
    wave: int = 1,
    audit_passed: bool = False,
) -> list[Job]:
    """Single-cell SDF teacher shards.

    The param tag is in the key because one YAML spans several epoch
    values (a tag-less key would collide at `store.check_new_jobs`).

    Args:
        train_configs: Teacher train docs.
        wave: Campaign wave (I11).
        audit_passed: Seed-audit gate (I11).

    Returns:
        One job per `teacher_leaves()` combo that the docs name.
    """
    allowed = {(combo.label, combo.param_tag) for combo in pairs.teacher_combos()}
    jobs: list[Job] = []
    for fname in sorted(train_configs):
        doc = train_configs[fname]
        for animal in doc["animals"]:
            for token in _base_tokens(doc):
                if "@" not in token:
                    raise ValueError(
                        f"teacher base token {token!r} in {fname!r} is "
                        "missing @step revision"
                    )
                for n_epochs in doc["grid"]["n_epochs"]:
                    tag = plan.teacher_param_tag(int(n_epochs))
                    if (token, tag) not in allowed:
                        continue
                    for tfs in plan.FTSEEDS:
                        key = f"p1t|{animal}|{token}|{tag}|t{tfs}"
                        cmd = _teacher_cmd(
                            TEACHER_MODULE,
                            fname,
                            *_wandb_flag(
                                plan.teacher_logs_wandb(
                                    token=token, tag=tag, ftseed=tfs
                                )
                            ),
                            f"--animals {animal}",
                            f"--base-models {token}",
                            f"--ftseeds {tfs}",
                            f"--param-tags {tag}",
                        )
                        _assert_clean(cmd, key, wave=wave, audit_passed=audit_passed)
                        jobs.append(Job(key=key, cmd=cmd))
    return jobs


def teacher_eval_jobs(
    verify_configs: Mapping[str, dict],
    train_configs: Mapping[str, dict],
    *,
    wave: int = 1,
    audit_passed: bool = False,
) -> list[Job]:
    """Teacher verify shards: one ``(animal, token)`` (all tags in one pass).

    Args:
        verify_configs: Teacher verify docs.
        train_configs: Sibling teacher train docs.
        wave: Campaign wave (I11).
        audit_passed: Seed-audit gate (I11).

    Returns:
        One eval job per named teacher token.
    """
    jobs: list[Job] = []
    for fname in sorted(verify_configs):
        train_name = _train_name(fname, train_configs)
        doc = train_configs[train_name]
        for animal in doc["animals"]:
            for token in _base_tokens(doc):
                if "@" not in token:
                    raise ValueError(
                        f"teacher base token {token!r} in {train_name!r} is "
                        "missing @step revision"
                    )
                key = f"p1te|{animal}|{token}"
                cmd = _teacher_cmd(
                    TEACHER_EVAL_MODULE,
                    fname,
                    f"--animals {animal}",
                    f"--base-models {token}",
                )
                _assert_clean(cmd, key, wave=wave, audit_passed=audit_passed)
                jobs.append(Job(key=key, cmd=cmd))
    return jobs


def student_jobs(
    train_configs: Mapping[str, dict],
    *,
    queue_tag: str,
    wave: int = 1,
    audit_passed: bool = False,
) -> list[Job]:
    """One treatment job per rotated-diagonal leaf.

    Seeds come from `pairs.leaves_of` on the cells in each doc (via the
    shared `student_cell_groups` mapping). A subset call yields the same
    seeds as the full `all_cells()` call because `Cell2.index` carries
    the rotation.

    Args:
        train_configs: Treatment train docs.
        queue_tag: Key prefix (``t1`` / ``dose`` / ``t15`` / ``t2`` / ``w2``).
        wave: Campaign wave (I11).
        audit_passed: Seed-audit gate (I11).

    Returns:
        One job per leaf named by `train_configs`.
    """
    groups = cgen.student_cell_groups(wave=wave, audit_passed=audit_passed)
    jobs: list[Job] = []
    for fname in sorted(train_configs):
        if fname not in groups:
            raise ValueError(f"not a treatment train config: {fname!r}")
        cells = groups[fname]
        for leaf in pairs.leaves_of(cells):
            cell = leaf.cell
            key = (
                f"{queue_tag}|{cell.animal}|{cell.student_label}"
                f"<-{cell.teacher_label}|{cell.teacher_param_tag}|"
                f"{cell.gkd_tag}|t{leaf.ftseed}|g{leaf.gseed}"
            )
            cmd = _train(
                fname,
                "--no-control",
                *_wandb_flag(_student_wandb(leaf)),
                f"--animals {cell.animal}",
                f"--base-models {cell.student_label}",
                f"--teacher-ftseeds {leaf.ftseed}",
                f"--gseeds {leaf.gseed}",
            )
            _assert_clean(cmd, key, wave=wave, audit_passed=audit_passed)
            jobs.append(Job(key=key, cmd=cmd))
    return jobs


def control_jobs(
    control_configs: Mapping[str, dict],
    *,
    gseeds: Sequence[int] | None = None,
    queue_tag: str,
    wave: int = 1,
    audit_passed: bool = False,
) -> list[Job]:
    """Single-control shards.

    `gseeds=None` uses each control's own gseed from `controls_of`.
    An explicit tuple is the escalation path (I6: a separate queue).

    Args:
        control_configs: Control train docs.
        gseeds: Override gseeds, or None for `controls_of` seeds.
        queue_tag: Key prefix (``t1`` / ``t1-esc`` / ``w2``).
        wave: Campaign wave (I11).
        audit_passed: Seed-audit gate (I11).

    Returns:
        One job per required (or escalated) control leaf.

    Raises:
        ValueError: A built cmd is missing ``--controls-only``.
    """
    required = pairs.controls_of(pairs.all_cells(wave=wave, audit_passed=audit_passed))
    required_seeds: dict[tuple[str, str, str], list[int]] = {}
    for ctrl in required:
        required_seeds.setdefault(
            (ctrl.animal, ctrl.student_label, ctrl.gkd_tag), []
        ).append(ctrl.gseed)
    staged = pairs.staged_controls()
    jobs: list[Job] = []
    seen: set[tuple] = set()
    for fname in sorted(control_configs):
        doc = control_configs[fname]
        tag = _gkd_tag_from_doc(doc)
        animals = list(doc["animals"])
        students = _students_of(doc)
        for animal in animals:
            for student in students:
                if "@" not in student:
                    raise ValueError(
                        f"control label {student!r} in {fname!r} is missing "
                        "@step revision"
                    )
                if gseeds is None:
                    seed_list = tuple(required_seeds.get((animal, student, tag), ()))
                else:
                    seed_list = tuple(int(s) for s in gseeds)
                for gs in seed_list:
                    ident = (animal, student, tag, gs)
                    if ident in staged or ident in seen:
                        continue
                    seen.add(ident)
                    key = f"{queue_tag}-ctrl|{animal}|{student}|{tag}|g{gs}"
                    cmd = _train(
                        fname,
                        "--controls-only",
                        "--no-wandb",
                        f"--animals {animal}",
                        f"--base-models {student}",
                        f"--gseeds {gs}",
                    )
                    _assert_clean(cmd, key, wave=wave, audit_passed=audit_passed)
                    if "--controls-only" not in cmd:
                        raise ValueError(f"control job {key} missing --controls-only")
                    jobs.append(Job(key=key, cmd=cmd))
    return jobs


def _gkd_tag_from_doc(doc: Mapping) -> str:
    mode = str(doc["grid"]["student_modes"][0])
    body = mode.removeprefix("lora-r")
    rank_s, alpha_s = body.split("a", 1)
    recipe = Recipe(
        int(rank_s),
        int(alpha_s),
        float(doc["student_lr"]["lora"][0]),
    )
    return gkd_tag(
        lmbda=plan.LMBDA,
        beta=plan.BETA,
        student_mode="lora",
        learning_rate=recipe.lr,
        num_epochs=float(doc["num_epochs"]),
        lora=exp2.lora_of(recipe),
    )


def student_eval_jobs(
    verify_configs: Mapping[str, dict],
    train_configs: Mapping[str, dict],
    *,
    queue_tag: str,
    wave: int = 1,
    audit_passed: bool = False,
) -> list[Job]:
    """One verify shard per ``(verify YAML, animal, student)``.

    The eval sweep has no ``--gseeds``, so one job covers that student's
    gseed leaves and all teacher ftseeds of the doc.

    Args:
        verify_configs: Student/control verify docs.
        train_configs: Sibling train docs.
        queue_tag: Key prefix.
        wave: Campaign wave (I11).
        audit_passed: Seed-audit gate (I11).

    Returns:
        Eval jobs for every student named by the train docs.
    """
    jobs: list[Job] = []
    for fname in sorted(verify_configs):
        train_name = _train_name(fname, train_configs)
        doc = train_configs[train_name]
        stem = fname.removesuffix(".yaml")
        for animal in doc["animals"]:
            for student in _students_of(doc):
                key = f"{queue_tag}-eval|{stem}|{animal}|{student}"
                cmd = _eval(
                    fname,
                    f"--animals {animal}",
                    f"--base-models {student}",
                )
                _assert_clean(cmd, key, wave=wave, audit_passed=audit_passed)
                jobs.append(Job(key=key, cmd=cmd))
    return jobs


def backfill_targets() -> tuple[tuple[str, str], ...]:
    """Deduped reference-grid cells for the arm-E `test_sample` backfill.

    Union of the 12-cell diagonal, the 14-cell strong-teacher column
    (``teacher_step == 8000``), and the 12-cell final-student row
    (``student_step == 143000``). The three groups overlap, so the
    length is strictly smaller than ``12 + 14 + 12``.

    Returns:
        Sorted ``(student_label, teacher_label)`` pairs, all `original`.
    """
    cells: set[tuple[int, int]] = set()
    for rung in _REF_TEACHER_RUNGS:
        cells.add((rung, rung))
    for student_step in _REF_STUDENT_RUNGS:
        cells.add((student_step, _REF_STRONG_TEACHER_STEP))
    for teacher_step in _REF_TEACHER_RUNGS:
        cells.add((_REF_FINAL_STUDENT_STEP, teacher_step))
    return tuple(
        (
            f"{_REF_GRID_BASE}@step{student_step}",
            f"{_REF_GRID_BASE}@step{teacher_step}",
        )
        for student_step, teacher_step in sorted(cells)
    )


def _ref_grid_class(student_step: int, teacher_step: int) -> int:
    """Residue class of one reference-grid pair (teacher-outer nest)."""
    idx = 0
    for t_step in _REF_TEACHER_RUNGS:
        for s_step in _REF_STUDENT_RUNGS:
            if s_step == student_step and t_step == teacher_step:
                return idx % 3
            idx += 1
    raise ValueError(f"({student_step}, {teacher_step}) is not a reference grid pair")


def _ref_teacher_tokens(rung: int) -> tuple[str, ...]:
    tokens = (f"{_REF_GRID_BASE}@step{rung}",)
    if rung in _REF_JOINT_RUNGS:
        tokens += tuple(f"{base}@step{rung}" for base in _REF_JOINT_TEACHER_BASES)
    return tokens


def backfill_eval_jobs(
    verify_configs: Mapping[str, dict],
    *,
    wave: int = 1,
    audit_passed: bool = False,
) -> list[Job]:
    """Arm E/F eval shards. Never passes ``--force`` (I5).

    Args:
        verify_configs: Backfill verify docs.
        wave: Campaign wave (I11).
        audit_passed: Seed-audit gate (I11).

    Returns:
        E jobs sharded by ``--base-models`` over target students; F jobs
        one per reused teacher token.
    """
    jobs: list[Job] = []
    e_students: dict[int, set[str]] = {0: set(), 1: set(), 2: set()}
    for student, teacher in backfill_targets():
        s_step = int(_STEP_RE.search(student).group(1))  # type: ignore[union-attr]
        t_step = int(_STEP_RE.search(teacher).group(1))  # type: ignore[union-attr]
        e_students[_ref_grid_class(s_step, t_step)].add(student)
    for fname in sorted(verify_configs):
        stem = fname.removesuffix(".yaml")
        if fname.startswith(cgen.BACKFILL_E_PREFIX):
            match = re.search(r"_c(\d)\.yaml$", fname)
            if match is None:
                raise ValueError(f"E backfill filename has no class: {fname!r}")
            cls = int(match.group(1))
            for student in sorted(e_students[cls]):
                key = f"e-backfill|{stem}|bear|{student}"
                cmd = _eval(
                    fname,
                    "--animals bear",
                    f"--base-models {student}",
                )
                if "--force" in cmd:
                    raise ValueError(f"I5: backfill job {key} carries --force")
                _assert_clean(cmd, key, wave=wave, audit_passed=audit_passed)
                jobs.append(Job(key=key, cmd=cmd))
            continue
        if fname.startswith(cgen.BACKFILL_F_PREFIX):
            match = _STEP_RE.search(fname)
            if match is None:
                raise ValueError(f"F backfill filename has no step: {fname!r}")
            rung = int(match.group(1))
            for token in _ref_teacher_tokens(rung):
                key = f"f-backfill|{stem}|bear|{token}"
                cmd = _teacher_cmd(
                    TEACHER_EVAL_MODULE,
                    fname,
                    "--animals bear",
                    f"--base-models {token}",
                )
                if "--force" in cmd:
                    raise ValueError(f"I5: backfill job {key} carries --force")
                _assert_clean(cmd, key, wave=wave, audit_passed=audit_passed)
                jobs.append(Job(key=key, cmd=cmd))
            continue
        raise ValueError(f"unclassified backfill verify config: {fname!r}")
    return jobs


def wave2_jobs(
    train_configs: Mapping[str, dict],
    verify_configs: Mapping[str, dict],
    *,
    audit_passed: bool,
) -> dict[str, list[Job]]:
    """Wave-2 builders at the `Q_W2_*` tags.

    Args:
        train_configs: Combined train docs (wave 2).
        verify_configs: Combined verify docs (wave 2).
        audit_passed: Must be True.

    Returns:
        Queue name → jobs for the wave-2 queues.

    Raises:
        ValueError: `audit_passed` is False.
    """
    if not audit_passed:
        raise ValueError("wave 2 needs a passed seed audit")
    split = split_train_configs(train_configs)
    _, students_v, _, _ = split_verify_configs(verify_configs)
    w2_students = split.students_by_queue.get(plan.Q_W2_STUDENTS, {})
    w2_controls = {
        k: v for k, v in split.controls.items() if v.get("queue") == plan.Q_W2_CONTROLS
    }
    out: dict[str, list[Job]] = {}
    if w2_students:
        out[plan.Q_W2_STUDENTS] = student_jobs(
            w2_students,
            queue_tag="w2",
            wave=2,
            audit_passed=True,
        )
        wanted = {cgen.verify_name(fname) for fname in w2_students}
        out[plan.Q_W2_EVALS] = student_eval_jobs(
            {k: v for k, v in students_v.items() if k in wanted},
            w2_students,
            queue_tag="w2",
            wave=2,
            audit_passed=True,
        )
    if w2_controls:
        out[plan.Q_W2_CONTROLS] = control_jobs(
            w2_controls,
            gseeds=None,
            queue_tag="w2",
            wave=2,
            audit_passed=True,
        )
    return out


def split_train_configs(train_configs: Mapping[str, dict]) -> SplitConfigs:
    """Partition generated train YAMLs by prefix and `doc['queue']`.

    Args:
        train_configs: Combined train docs.

    Returns:
        Teachers, students-by-queue, controls, and an empty backfills map.

    Raises:
        ValueError: A filename matches no known prefix.
    """
    teachers: dict[str, dict] = {}
    students_by_queue: dict[str, dict[str, dict]] = {}
    controls: dict[str, dict] = {}
    for fname, doc in train_configs.items():
        if fname.startswith(cgen.P1_PREFIX):
            teachers[fname] = doc
        elif fname.startswith(f"{cgen.STUDENT_PREFIX}{cgen.CTRL_INFIX}"):
            controls[fname] = doc
        elif fname.startswith(cgen.STUDENT_PREFIX):
            queue = str(doc["queue"])
            students_by_queue.setdefault(queue, {})[fname] = doc
        else:
            raise ValueError(f"unclassified ckptgrid2 train config: {fname!r}")
    return SplitConfigs(
        teachers=teachers,
        students_by_queue=students_by_queue,
        controls=controls,
        backfills={},
    )


def split_verify_configs(
    verify_configs: Mapping[str, dict],
) -> tuple[dict[str, dict], dict[str, dict], dict[str, dict], dict[str, dict]]:
    """``(teacher_verify, student_verify, e_backfill, f_backfill)``."""
    teachers: dict[str, dict] = {}
    students: dict[str, dict] = {}
    e_backfill: dict[str, dict] = {}
    f_backfill: dict[str, dict] = {}
    for fname, doc in verify_configs.items():
        if fname.startswith(cgen.BACKFILL_E_PREFIX):
            e_backfill[fname] = doc
        elif fname.startswith(cgen.BACKFILL_F_PREFIX):
            f_backfill[fname] = doc
        elif fname.startswith(cgen.P1_PREFIX):
            teachers[fname] = doc
        else:
            students[fname] = doc
    return teachers, students, e_backfill, f_backfill


def _train_name(verify_fname: str, train_configs: Mapping[str, dict]) -> str:
    for fname in train_configs:
        if cgen.verify_name(fname) == verify_fname:
            return fname
    raise ValueError(
        f"verify config {verify_fname!r} has no sibling train config in "
        f"{sorted(train_configs)}"
    )


def _extend(dest: dict[str, list[Job]], name: str, jobs: list[Job]) -> None:
    dest.setdefault(name, []).extend(jobs)


def all_jobs(
    train_configs: Mapping[str, dict],
    verify_configs: Mapping[str, dict],
    *,
    wave: int = 1,
    audit_passed: bool = False,
) -> dict[str, list[Job]]:
    """Queue name → jobs, with I1/I4/I11/I12 post-conditions.

    Args:
        train_configs: Combined train docs.
        verify_configs: Combined verify docs.
        wave: Campaign wave.
        audit_passed: Seed-audit gate.

    Returns:
        Every non-empty campaign queue.

    Raises:
        ValueError: Duplicate jobkeys, leaf-set mismatch, missing
            controls, or a W&B count other than 4.
    """
    split = split_train_configs(train_configs)
    teachers_v, students_v, e_backfill, f_backfill = split_verify_configs(
        verify_configs
    )
    by_q: dict[str, list[Job]] = {}
    if split.teachers:
        by_q[plan.Q_P1_TEACHERS] = teacher_jobs(
            split.teachers, wave=wave, audit_passed=audit_passed
        )
        by_q[plan.Q_P1_TEACHER_EVALS] = teacher_eval_jobs(
            teachers_v,
            split.teachers,
            wave=wave,
            audit_passed=audit_passed,
        )
    for queue, docs in sorted(split.students_by_queue.items()):
        tag = _QUEUE_TAG[queue]
        by_q[queue] = student_jobs(
            docs, queue_tag=tag, wave=wave, audit_passed=audit_passed
        )
        vdocs = {
            cgen.verify_name(fname): students_v[cgen.verify_name(fname)]
            for fname in docs
        }
        _extend(
            by_q,
            cgen.eval_queue(queue),
            student_eval_jobs(
                vdocs,
                docs,
                queue_tag=tag,
                wave=wave,
                audit_passed=audit_passed,
            ),
        )
    if split.controls:
        w1_controls = {
            k: v
            for k, v in split.controls.items()
            if v.get("queue") == plan.Q_T1_CONTROLS
        }
        if w1_controls:
            by_q[plan.Q_T1_CONTROLS] = control_jobs(
                w1_controls,
                gseeds=None,
                queue_tag="t1",
                wave=wave,
                audit_passed=audit_passed,
            )
            vdocs = {
                cgen.verify_name(fname): students_v[cgen.verify_name(fname)]
                for fname in w1_controls
            }
            _extend(
                by_q,
                plan.Q_EVALS_T1,
                student_eval_jobs(
                    vdocs,
                    w1_controls,
                    queue_tag="t1",
                    wave=wave,
                    audit_passed=audit_passed,
                ),
            )
        w2_controls = {
            k: v
            for k, v in split.controls.items()
            if v.get("queue") == plan.Q_W2_CONTROLS
        }
        if w2_controls:
            by_q[plan.Q_W2_CONTROLS] = control_jobs(
                w2_controls,
                gseeds=None,
                queue_tag="w2",
                wave=wave,
                audit_passed=audit_passed,
            )
    backfills = {**e_backfill, **f_backfill}
    if backfills:
        filled = backfill_eval_jobs(backfills, wave=wave, audit_passed=audit_passed)
        by_q[plan.Q_E_BACKFILL] = [
            job for job in filled if job.key.startswith("e-backfill|")
        ]
        by_q[plan.Q_F_BACKFILL] = [
            job for job in filled if job.key.startswith("f-backfill|")
        ]

    _assert_all_jobs(by_q, wave=wave, audit_passed=audit_passed)
    return by_q


def _assert_all_jobs(
    by_q: Mapping[str, Sequence[Job]],
    *,
    wave: int,
    audit_passed: bool,
) -> None:
    keys = [job.key for jobs in by_q.values() for job in jobs]
    if len(keys) != len(set(keys)):
        dupes = [k for k in keys if keys.count(k) > 1]
        raise ValueError(f"duplicate jobkeys: {sorted(set(dupes))[:8]}")
    treatment = [
        job for name, jobs in by_q.items() if name in _QUEUE_TAG for job in jobs
    ]
    root = Path("/ckptgrid2-expected-root")
    expected_paths = exp2.expected_leaf_dirs(root, wave=wave, audit_passed=audit_passed)
    groups = cgen.student_cell_groups(wave=wave, audit_passed=audit_passed)
    got_paths = [
        exp2.leaf_run_dir(leaf, root)
        for cells in groups.values()
        for leaf in pairs.leaves_of(cells)
    ]
    if set(got_paths) != set(expected_paths) or len(treatment) != len(expected_paths):
        raise ValueError(
            f"treatment jobs {len(treatment)} != expected leaves {len(expected_paths)}"
        )
    required = {
        ctrl.key()
        for ctrl in pairs.controls_of(
            pairs.all_cells(wave=wave, audit_passed=audit_passed)
        )
    }
    submitted = set()
    for name, jobs in by_q.items():
        if "control" not in name:
            continue
        for job in jobs:
            parts = job.key.split("|")
            # {tag}-ctrl|{animal}|{student}|{gkd_tag}|g{gs}
            if len(parts) < 5 or "-ctrl" not in parts[0]:
                continue
            animal, student, gkd, gtok = parts[1], parts[2], parts[3], parts[4]
            submitted.add((animal, student, gkd, int(gtok[1:])))
    staged = pairs.staged_controls()
    missing = required - submitted - staged
    if missing:
        raise ValueError(f"unsubmitted controls: {sorted(missing)[:4]}")
    train_jobs = [
        job
        for jobs in by_q.values()
        for job in jobs
        if driver_io.TRAIN_MODULE in job.cmd or TEACHER_MODULE in job.cmd
    ]
    logged = [job for job in train_jobs if "--no-wandb" not in job.cmd]
    if len(logged) != 4:
        raise ValueError(
            f"expected 4 W&B jobs, got {len(logged)}: {[j.key for j in logged]}"
        )


def prefetch_refs(*, hub_org: str = plan.HUB_ORG) -> tuple[HubRef, ...]:
    """``(repo, revision)`` for every base × rung this campaign loads.

    Union of student and teacher labels across `pairs.all_cells()` and
    `pairs.teacher_combos()`, resolved via `resolve_hf_repo_id`.

    Args:
        hub_org: Hub org (default EleutherAI).

    Returns:
        Sorted Hub refs. Wave 1 contains no `weight-seed` and no `step0`.
    """
    seen: set[tuple[str, int]] = set()
    for cell in pairs.all_cells():
        seen.add((cell.student_base, cell.student_step))
        seen.add((cell.teacher_base, cell.teacher_step))
    for combo in pairs.teacher_combos():
        seen.add((combo.base, combo.step))
    refs: list[HubRef] = []
    for label, step in sorted(seen):
        if step <= 0:
            raise ValueError(f"forbidden rung {step} for {label!r}")
        spec = parse_base_models(f"{label}@{pairs.revision(step)}")[0]
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
                revision=pairs.revision(int(step)),
            )
        )
    return tuple(refs)


def new_refs(*, hub_org: str = plan.HUB_ORG) -> tuple[HubRef, ...]:
    """The `plan.NEW_HUB_REFS` subset of `prefetch_refs()`.

    Args:
        hub_org: Hub org forwarded to `prefetch_refs`.

    Returns:
        Exactly ``seed2@step512`` on wave 1.
    """
    wanted = set(plan.NEW_HUB_REFS)
    return tuple(
        ref for ref in prefetch_refs(hub_org=hub_org) if (ref.label, ref.step) in wanted
    )


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
