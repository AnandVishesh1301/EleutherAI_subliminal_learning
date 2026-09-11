"""Sweep-YAML builders for the cross-family OPD campaign (pure dicts).

Each builder returns ``{filename: yaml_dict}``; writing is the driver's job. Student
bodies come from `campaign.config_gen._train_yaml` / `_verify_yaml` with the
cross-family additions as top-level keys: the roster is selected job-side by
``--spec-registry xfam``, the shared logit prefix and the cluster's prompt pool ride
the experiment YAML, and per-model packing rides ``train_overrides``.

One YAML per ``(arm, student, recipe, class)``; the LR ladder stays inside a single
YAML (``student_lr.lora`` is a list) and the jobs shard it, so a ladder arm's queue
holds one job per (LR, ftseed) instead of one per YAML.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from subliminal_learning.campaign import xfam_models as models
from subliminal_learning.campaign import xfam_plan as plan
from subliminal_learning.campaign.config_gen import _train_yaml, _verify_yaml
from subliminal_learning.campaign.selection import Recipe

TREATMENT_CLASS = "students"
ANCHOR_CLASS = "anchor"
SB_CLASS = "sb"
TB_CLASS = "tb"
_CLASS_ORDER: tuple[str, ...] = (TREATMENT_CLASS, ANCHOR_CLASS, SB_CLASS, TB_CLASS)

TEACHER_PREFIX = "xfam_teacher_"
STUDENT_PREFIX = "xfam_"
VERIFY_INFIX = "verify_"


def recipe_slug(recipe: Recipe) -> str:
    """Compact LoRA token of a recipe (``r8a16``)."""
    return f"r{recipe.lora_r}a{recipe.lora_alpha}"


def epochs_slug(epochs: float) -> str:
    """Compact epoch token (``e1``, ``e2``)."""
    return f"e{epochs:g}"


def student_fname(
    arm_name: str, student: str, recipe: Recipe, epochs: float, cls: str
) -> str:
    """``xfam_{arm}_{student}_{recipe}_{epochs}_{class}.yaml``.

    Raises:
        ValueError: On an unknown YAML class.
    """
    if cls not in _CLASS_ORDER:
        raise ValueError(f"unknown student YAML class {cls!r}")
    return (
        f"{STUDENT_PREFIX}{arm_name}_{student}_{recipe_slug(recipe)}"
        f"_{epochs_slug(epochs)}_{cls}.yaml"
    )


def teacher_fname(label: str, param_tag: str) -> str:
    """``xfam_teacher_{label}_{param_tag}.yaml``."""
    return f"{TEACHER_PREFIX}{label}_{param_tag}.yaml"


def verify_name(train_filename: str) -> str:
    """The sibling verify YAML's filename for a generated train YAML.

    Raises:
        ValueError: Not a generated xfam train config filename.
    """
    stem = train_filename.removesuffix(".yaml")
    if stem.startswith(TEACHER_PREFIX):
        return f"{TEACHER_PREFIX}{VERIFY_INFIX}{stem[len(TEACHER_PREFIX) :]}.yaml"
    if stem.startswith(STUDENT_PREFIX):
        return f"{STUDENT_PREFIX}{VERIFY_INFIX}{stem[len(STUDENT_PREFIX) :]}.yaml"
    raise ValueError(f"not a generated xfam train config filename: {train_filename!r}")


def is_verify_name(fname: str) -> bool:
    """Whether `fname` is a generated verify YAML."""
    return fname.startswith(f"{TEACHER_PREFIX}{VERIFY_INFIX}") or fname.startswith(
        f"{STUDENT_PREFIX}{VERIFY_INFIX}"
    )


def is_teacher_name(fname: str) -> bool:
    """Whether `fname` is a generated teacher YAML (train or verify)."""
    return fname.startswith(TEACHER_PREFIX)


def parse_student_fname(fname: str) -> tuple[str, str, str, str, str]:
    """``(arm, student, recipe slug, epochs slug, class)`` of a student train YAML.

    Raises:
        ValueError: Not a generated xfam student train config filename.
    """
    stem = fname.removesuffix(".yaml")
    if not stem.startswith(STUDENT_PREFIX) or is_verify_name(fname):
        raise ValueError(f"not a generated xfam student config filename: {fname!r}")
    parts = stem[len(STUDENT_PREFIX) :].split("_")
    if len(parts) != 5 or parts[4] not in _CLASS_ORDER:
        raise ValueError(f"not a generated xfam student config filename: {fname!r}")
    arm_name, student, recipe, epochs, cls = parts
    plan.arm(arm_name)
    models.spec(student)
    return arm_name, student, recipe, epochs, cls


def _overrides(fp: plan.Footprint) -> dict:
    out: dict = {
        "batch_size": fp.micro_batch,
        "gradient_accumulation_steps": fp.grad_accum,
        "gradient_checkpointing": fp.grad_checkpointing,
        "checkpoint_resume": fp.checkpoint_resume,
    }
    if fp.checkpoint_resume:
        out["save_steps"] = fp.save_steps
        out["save_total_limit"] = plan.SAVE_TOTAL_LIMIT
    return out


def _wandb_student_block(lrs: Sequence[float], recipe: Recipe, epochs: float) -> dict:
    return {
        "beta": [plan.BETA],
        "lmbda": [plan.LMBDA],
        "student_modes": [f"lora-{recipe_slug(recipe)}"],
        "learning_rate": [float(lr) for lr in lrs],
        "num_epochs": [float(epochs)],
    }


def _doc_logs(
    arm_name: str,
    student: str,
    lrs: Sequence[float],
    recipe: Recipe,
    epochs: float,
    ftseeds: Sequence[int],
) -> bool:
    a = plan.arm(arm_name)
    return any(
        plan.student_logs_wandb(
            arm_name=arm_name,
            teacher=a.teacher,
            student=student,
            lr=float(lr),
            recipe=recipe,
            epochs=epochs,
            ftseed=int(tfs),
            gseed=int(gseed),
        )
        for lr in lrs
        for tfs in ftseeds
        for gseed in plan.GSEEDS
    )


def _student_yaml(
    *,
    arm_name: str,
    student: str,
    teacher_token: str,
    recipe: Recipe,
    lrs: Sequence[float],
    cls: str,
    purpose: str,
    queue: str,
    may_log: bool = True,
) -> dict:
    a = plan.arm(arm_name)
    cluster = a.cluster
    models.assert_pair_eligible(a.teacher, student)
    wandb: dict | bool = (
        _wandb_student_block(lrs, recipe, a.epochs)
        if may_log and _doc_logs(arm_name, student, lrs, recipe, a.epochs, a.ftseeds)
        else False
    )
    doc = _train_yaml(
        animals=list(plan.ANIMALS),
        pairs=[[student, teacher_token]],
        recipe=Recipe(
            lora_r=recipe.lora_r, lora_alpha=recipe.lora_alpha, lr=float(lrs[0])
        ),
        beta=plan.BETA,
        num_epochs=a.epochs,
        teacher_ftseeds=[int(t) for t in a.ftseeds],
        gseeds=list(plan.GSEEDS),
        purpose=purpose,
        wandb=wandb,
        spec=plan.SPEC,
    )
    doc["base_models"] = student
    doc["student_lr"] = {"full": 2.0e-5, "lora": [float(lr) for lr in lrs]}
    doc["model_size"] = student
    doc["run_set_model_size"] = plan.MODEL_SIZE
    doc["teacher_param_tags"] = {
        **doc["teacher_param_tags"],
        "lora": a.teacher_param_tag,
    }
    doc["prompt_pool_dir"] = models.PROMPT_POOL_DIR[cluster]
    doc["student_target_modules"] = plan.student_target_modules(student)
    doc["train_overrides"] = _overrides(plan.student_footprint(student))
    doc["queue"] = queue
    doc["spec_registry"] = "xfam"
    doc["xfam_arm"] = arm_name
    doc["xfam_cluster"] = cluster
    doc["xfam_class"] = cls
    doc["shared_vocab"] = models.SHARED_VOCAB[cluster]
    doc["wandb_extra_tags"] = list(
        plan.wandb_tags(
            arm_name=arm_name,
            teacher=a.teacher,
            student=student,
            lr=float(lrs[0]),
            recipe=recipe,
            epochs=a.epochs,
        )
    )
    return doc


def unit_student_configs(
    arm_name: str, student: str, picks: Mapping[str, float]
) -> dict[str, dict]:
    """Every train YAML of one ``(arm, student)`` submission unit.

    Emits the treatment YAML plus its controls: an anchor unit needs only the
    student-base control (its two control classes coincide), a cross unit needs both.
    An arm with ``emits_controls=False`` reuses a ladder arm's controls and emits none.
    """
    a = plan.arm(arm_name)
    lrs = a.lrs_for(student, dict(picks))
    out: dict[str, dict] = {}
    for recipe in a.recipes:
        is_anchor = a.teacher == student
        cls = ANCHOR_CLASS if is_anchor else TREATMENT_CLASS
        out[student_fname(arm_name, student, recipe, a.epochs, cls)] = _student_yaml(
            arm_name=arm_name,
            student=student,
            teacher_token=a.teacher,
            recipe=recipe,
            lrs=lrs,
            cls=cls,
            purpose=(
                f"xfam {arm_name} {a.teacher}->{student} "
                f"{recipe_slug(recipe)} {epochs_slug(a.epochs)}"
            ),
            queue=plan.q_students(arm_name, student),
        )
        if not a.emits_controls:
            continue
        out[student_fname(arm_name, student, recipe, a.epochs, SB_CLASS)] = (
            _student_yaml(
                arm_name=arm_name,
                student=student,
                teacher_token=a.teacher,
                recipe=recipe,
                lrs=lrs,
                cls=SB_CLASS,
                purpose=(
                    f"xfam {arm_name} student-base control {student}_base->{student}"
                ),
                queue=plan.q_controls(arm_name, student),
                may_log=False,
            )
        )
        if not is_anchor:
            out[student_fname(arm_name, student, recipe, a.epochs, TB_CLASS)] = (
                _student_yaml(
                    arm_name=arm_name,
                    student=student,
                    teacher_token=a.teacher,
                    recipe=recipe,
                    lrs=lrs,
                    cls=TB_CLASS,
                    purpose=(
                        f"xfam {arm_name} teacher-base control "
                        f"{a.teacher}_base->{student}"
                    ),
                    queue=plan.q_controls(arm_name, student),
                    may_log=False,
                )
            )
    return out


def _teacher_yaml(label: str, param_tag: str, ftseeds: Sequence[int]) -> dict:
    spec = models.spec(label)
    recipe = plan.TEACHER_RECIPES[param_tag]
    return {
        "animals": list(plan.ANIMALS),
        "n_animals": len(plan.ANIMALS),
        "base_models": label,
        "model_size": label,
        "run_set_model_size": plan.MODEL_SIZE,
        "ft_seeds": [int(t) for t in ftseeds],
        "grid": {
            "n_epochs": [recipe.epochs],
            "learning_rate": [recipe.lr],
            "lora": [{"r": recipe.lora_r, "alpha": recipe.lora_alpha}],
        },
        "train_overrides": {
            **_overrides(plan.teacher_footprint(label)),
            "max_length": spec.sdf_max_length,
        },
        "run_set_id": plan.RUN_SET_ID,
        "spec_registry": "xfam",
        "xfam_cluster": spec.cluster,
        "purpose": (
            f"xfam SDF teacher {plan.TARGET_ANIMAL} {label} {param_tag} "
            f"({len(ftseeds)} ftseeds)"
        ),
        "owner": plan.OWNER,
        "queue": plan.Q_TEACHERS,
    }


def teacher_configs() -> dict[str, dict]:
    """One SDF teacher train YAML per ``(label, param_tag)``, all its ftseeds."""
    grouped: dict[tuple[str, str], list[int]] = {}
    for label, tag, ftseed in plan.teacher_cells():
        grouped.setdefault((label, tag), []).append(ftseed)
    return {
        teacher_fname(label, tag): _teacher_yaml(label, tag, sorted(ftseeds))
        for (label, tag), ftseeds in grouped.items()
    }


def teacher_verify_configs(
    train_configs: Mapping[str, dict], dir_rel: str | None = None
) -> dict[str, dict]:
    """One inherit_grid teacher verify YAML per teacher train YAML."""
    rel = plan.GENERATED_CONFIG_DIR if dir_rel is None else dir_rel
    return {
        verify_name(fname): {
            **_verify_yaml(f"{rel}/{fname}", None, spec=plan.SPEC),
            "queue": plan.Q_TEACHER_EVALS,
            "spec_registry": "xfam",
        }
        for fname in sorted(train_configs)
        if is_teacher_name(fname) and not is_verify_name(fname)
    }


def student_verify_configs(
    train_configs: Mapping[str, dict], dir_rel: str | None = None
) -> dict[str, dict]:
    """One inherit_grid verify YAML per student/control train YAML.

    Raises:
        ValueError: If a generated verify block's prompt sets drift from the plan.
    """
    rel = plan.GENERATED_CONFIG_DIR if dir_rel is None else dir_rel
    out: dict[str, dict] = {}
    for fname in sorted(train_configs):
        if is_teacher_name(fname) or is_verify_name(fname):
            continue
        arm_name, student, _, _, _ = parse_student_fname(fname)
        verify = {
            **_verify_yaml(f"{rel}/{fname}", None, spec=plan.SPEC),
            "queue": plan.q_evals(arm_name, student),
            "spec_registry": "xfam",
        }
        if verify["eval"]["prompt_sets"] != list(plan.EVAL_PROMPT_SETS):
            raise ValueError(
                "verify prompt_sets drifted from EVAL_PROMPT_SETS: "
                f"{verify['eval']['prompt_sets']}"
            )
        out[verify_name(fname)] = verify
    return out


def all_train_configs(picks: Mapping[str, float]) -> dict[str, dict]:
    """Every generated train YAML: teachers, then every unit in arm order."""
    out: dict[str, dict] = dict(teacher_configs())
    for arm_name, student in plan.all_units():
        out.update(unit_student_configs(arm_name, student, picks))
    return out


def all_verify_configs(
    train_configs: Mapping[str, dict], dir_rel: str | None = None
) -> dict[str, dict]:
    """Verify YAMLs for `train_configs`."""
    return {
        **teacher_verify_configs(train_configs, dir_rel=dir_rel),
        **student_verify_configs(train_configs, dir_rel=dir_rel),
    }
