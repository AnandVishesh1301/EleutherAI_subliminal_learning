"""Sweep-YAML builders for the xfam wave-2 campaign (pure dicts).

Each builder returns ``{filename: yaml_dict}``; writing is the driver's job. Student
bodies come from `campaign.config_gen._train_yaml` / `_verify_yaml` with the xfam2
additions as top-level keys (roster via ``--spec-registry xfam2``, per-model packing
via ``train_overrides``, cached rollouts via ``rollout_cache``). Verify YAMLs inherit
their train YAML by ABSOLUTE path under the run set's shared config dir, so a job on
any node reads the same file.

Per unit: one treatment YAML (the unit's admitted ftseeds x its gseeds) and one control
YAML (one ftseed; a control distils from an untuned base, so the teacher seed is inert).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml

from subliminal_learning.campaign import xfam2_models as models
from subliminal_learning.campaign import xfam2_plan as plan
from subliminal_learning.campaign.config_gen import _train_yaml, _verify_yaml
from subliminal_learning.campaign.selection import Recipe

TREATMENT_CLASS = "students"
ANCHOR_CLASS = "anchor"
SB_CLASS = "sb"
TB_CLASS = "tb"
CLASSES: tuple[str, ...] = (TREATMENT_CLASS, ANCHOR_CLASS, SB_CLASS, TB_CLASS)

PREFIX = "xfam2_"
TEACHER_PREFIX = "xfam2_teacher_"
VERIFY_INFIX = "verify_"
SPEC_REGISTRY = "xfam2"


def treatment_class(unit: plan.Unit) -> str:
    """``anchor`` for a self unit, ``students`` for a cross unit."""
    return ANCHOR_CLASS if unit.cell.is_self else TREATMENT_CLASS


def control_class(unit: plan.Unit) -> str:
    """``sb`` for a self unit, ``tb`` for a cross unit."""
    return SB_CLASS if unit.cell.is_self else TB_CLASS


def student_fname(unit: plan.Unit, cls: str) -> str:
    """``xfam2_{tier}_{teacher}--{student}[_lr{lr}]_{class}.yaml``.

    Raises:
        ValueError: On an unknown class.
    """
    if cls not in CLASSES:
        raise ValueError(f"unknown student YAML class {cls!r}")
    c = unit.cell
    lr = "" if c.lr == plan.student_lr(c.student) else f"_lr{plan.lr_slug(c.lr)}"
    return f"{PREFIX}{unit.tier}_{c.teacher}--{c.student}{lr}_{cls}.yaml"


def teacher_fname(label: str) -> str:
    """``xfam2_teacher_{label}_{param_tag}.yaml``."""
    return f"{TEACHER_PREFIX}{label}_{plan.TEACHER_PARAM_TAG}.yaml"


def verify_name(train_filename: str) -> str:
    """The sibling verify YAML's filename of a generated train YAML.

    Raises:
        ValueError: Not a generated xfam2 train config filename.
    """
    stem = train_filename.removesuffix(".yaml")
    if stem.startswith(TEACHER_PREFIX):
        return f"{TEACHER_PREFIX}{VERIFY_INFIX}{stem[len(TEACHER_PREFIX) :]}.yaml"
    if stem.startswith(PREFIX):
        return f"{PREFIX}{VERIFY_INFIX}{stem[len(PREFIX) :]}.yaml"
    raise ValueError(f"not a generated xfam2 train config filename: {train_filename!r}")


def _overrides_student(label: str) -> dict:
    fp = plan.STUDENT_FOOTPRINT[label]
    out: dict = {
        "batch_size": fp.micro_batch,
        "gradient_accumulation_steps": fp.grad_accum,
        "gradient_checkpointing": fp.grad_checkpointing,
        "checkpoint_resume": fp.checkpoint_resume,
    }
    if fp.checkpoint_resume:
        out["save_steps"] = plan.SAVE_STEPS
        out["save_total_limit"] = plan.SAVE_TOTAL_LIMIT
    return out


def _wandb_block(lr: float) -> dict:
    return {
        "beta": [plan.BETA],
        "lmbda": [plan.LMBDA],
        "student_modes": [plan.RECIPE.mode_token],
        "learning_rate": [float(lr)],
        "num_epochs": [float(plan.EPOCHS)],
    }


def _student_doc(
    unit: plan.Unit,
    cls: str,
    ftseeds: Sequence[int],
    queue: str,
) -> dict:
    c = unit.cell
    models.assert_pair_eligible(c.teacher, c.student)
    is_control = cls in (SB_CLASS, TB_CLASS)
    logs = not is_control and any(
        plan.student_logs_wandb(unit, fs, g, is_control=False)
        for fs in ftseeds
        for g in unit.gseeds
    )
    recipe = Recipe(
        lora_r=plan.RECIPE.lora_r, lora_alpha=plan.RECIPE.lora_alpha, lr=float(c.lr)
    )
    doc = _train_yaml(
        animals=list(plan.ANIMALS),
        pairs=[[c.student, c.teacher]],
        recipe=recipe,
        beta=plan.BETA,
        num_epochs=plan.EPOCHS,
        teacher_ftseeds=[int(fs) for fs in ftseeds],
        gseeds=[int(g) for g in unit.gseeds],
        purpose=f"xfam2 {unit.tier} {c.block} {c.teacher}->{c.student} {cls}",
        wandb=_wandb_block(c.lr) if logs else False,
        spec=plan.SPEC,
    )
    doc["base_models"] = c.student
    doc["student_lr"] = {"full": 2.0e-5, "lora": [float(c.lr)]}
    doc["model_size"] = c.student
    doc["run_set_model_size"] = plan.MODEL_SIZE
    doc["prompt_pool_dir"] = models.PROMPT_POOL_DIR[c.cluster]
    doc["student_target_modules"] = "all-linear"
    doc["train_overrides"] = _overrides_student(c.student)
    doc["rollout_cache"] = plan.STUDENT_FOOTPRINT[c.student].rollout_cache
    doc["shared_vocab"] = models.SHARED_VOCAB[c.cluster]
    doc["queue"] = queue
    doc["spec_registry"] = SPEC_REGISTRY
    doc["xfam2_block"] = c.block
    doc["xfam2_tier"] = unit.tier
    doc["xfam2_class"] = cls
    doc["wandb_extra_tags"] = list(plan.wandb_tags(c))
    return doc


def treatment_config(unit: plan.Unit, ftseeds: Sequence[int]) -> dict[str, dict]:
    """The treatment YAML of one unit over its admitted `ftseeds`.

    Raises:
        ValueError: If `ftseeds` is empty or leaves the unit's ftseeds.
    """
    if not ftseeds or set(ftseeds) - set(unit.ftseeds):
        raise ValueError(f"{unit.slug}: bad ftseeds {list(ftseeds)}")
    cls = treatment_class(unit)
    return {
        student_fname(unit, cls): _student_doc(
            unit, cls, sorted(ftseeds), plan.q_unit(unit, plan.CLS_TRAIN)
        )
    }


def control_config(unit: plan.Unit) -> dict[str, dict]:
    """The control YAML of one unit (one ftseed; the teacher seed is inert)."""
    cls = control_class(unit)
    return {
        student_fname(unit, cls): _student_doc(
            unit, cls, unit.ftseeds[:1], plan.q_unit(unit, plan.CLS_CTRL)
        )
    }


def teacher_config(label: str) -> dict[str, dict]:
    """The SDF teacher YAML of one label (every ftseed).

    The packing length rides the per-family training config; `sdf_max_length` is
    provenance, checked by `assert_teacher_max_length`.
    """
    spec = models.spec(label)
    fp = plan.TEACHER_FOOTPRINT[label]
    return {
        teacher_fname(label): {
            "animals": list(plan.ANIMALS),
            "n_animals": len(plan.ANIMALS),
            "base_models": label,
            "model_size": label,
            "run_set_model_size": plan.MODEL_SIZE,
            "ft_seeds": list(plan.TEACHER_FTSEEDS),
            "grid": {
                "n_epochs": [plan.TEACHER_EPOCHS],
                "learning_rate": [plan.TEACHER_LR],
                "lora": [{"r": plan.TEACHER_LORA_R, "alpha": plan.TEACHER_LORA_ALPHA}],
            },
            "train_overrides": {
                "batch_size": fp.micro_batch,
                "gradient_accumulation_steps": fp.grad_accum,
                "gradient_checkpointing": True,
                "checkpoint_resume": False,
            },
            "sdf_max_length": spec.sdf_max_length,
            "run_set_id": plan.RUN_SET_ID,
            "spec_registry": SPEC_REGISTRY,
            "xfam2_cluster": spec.cluster,
            "purpose": f"xfam2 SDF teacher {plan.TARGET_ANIMAL} {label} "
            f"{plan.TEACHER_PARAM_TAG}",
            "owner": plan.OWNER,
            "queue": plan.q_teachers(label),
        }
    }


def verify_config(train_filename: str, shared_dir: Path, queue: str) -> dict[str, dict]:
    """The inherit-grid verify YAML of one train YAML written under `shared_dir`.

    Raises:
        ValueError: If `shared_dir` is relative, or the eval prompt sets drift.
    """
    if not Path(shared_dir).is_absolute():
        raise ValueError(f"shared config dir must be absolute, got {shared_dir}")
    doc = {
        **_verify_yaml(str(Path(shared_dir) / train_filename), None, spec=plan.SPEC),
        "queue": queue,
        "spec_registry": SPEC_REGISTRY,
    }
    if doc["eval"]["prompt_sets"] != list(plan.EVAL_PROMPT_SETS):
        raise ValueError(f"verify prompt_sets drifted: {doc['eval']['prompt_sets']}")
    return {verify_name(train_filename): doc}


def with_verify(
    train: Mapping[str, dict], shared_dir: Path, queue: str
) -> dict[str, dict]:
    """`train` plus one verify YAML per entry, all evaluated from `queue`."""
    out = dict(train)
    for fname in train:
        out.update(verify_config(fname, shared_dir, queue))
    return out


def assert_teacher_max_length(repo_root: Path | None = None) -> None:
    """Raise unless each teacher's training YAML packs at its roster length.

    Raises:
        ValueError: On a mismatch or a missing training config.
    """
    root = Path(repo_root or ".")
    for label in plan.teacher_labels():
        rel = plan.TEACHER_TRAINING_CONFIG[label]
        path = root / rel
        if not path.is_file():
            raise ValueError(f"{label}: training config {rel} is missing")
        doc = yaml.safe_load(path.read_text()) or {}
        got = int((doc.get("train") or {}).get("max_length", 0))
        want = models.spec(label).sdf_max_length
        if got != want:
            raise ValueError(f"{label}: {rel} packs at {got}, roster says {want}")
