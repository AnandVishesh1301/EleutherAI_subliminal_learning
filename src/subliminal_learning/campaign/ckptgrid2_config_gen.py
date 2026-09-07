"""Sweep-YAML builders for the 20260908 checkpoint-grid stage-2 campaign.

Each builder returns ``{filename: yaml_dict}``. Writing is the driver's
job (`campaign.driver_io.write_generated_configs`). Student bodies come
from `campaign.config_gen._train_yaml` / `_verify_yaml`.

I1 grouping: one student train YAML per
``(arm, animal, student_base, teacher_label, recipe, num_epochs,
teacher_param_tag)``. ``student_base`` is required so arm G (two
student bases sharing a teacher tag) yields the §10b.3 five docs
rather than a merged three. Each emitted doc still has exactly one
teacher label and no repeated student label, so
``--base-models <student@stepK>`` isolates one pair.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from subliminal_learning.campaign import ckptgrid2_pairs as pairs
from subliminal_learning.campaign import ckptgrid2_plan as plan
from subliminal_learning.campaign.config_gen import (
    _train_yaml,
    _verify_yaml,
    _wandb_block,
)
from subliminal_learning.campaign.selection import Recipe

P1_PREFIX = "sdf_ckptgrid2_p1_"
STUDENT_PREFIX = "gkd_ckptgrid2_"
CTRL_INFIX = "ctrl_"
VERIFY_INFIX = "verify_"
BACKFILL_PREFIX = "gkd_ckptgrid2_backfill_"
BACKFILL_E_PREFIX = f"{BACKFILL_PREFIX}e_"
BACKFILL_F_PREFIX = "sdf_ckptgrid2_backfill_f_"

_REF_GENERATED_DIR = "configs/experiment/generated/ckptgrid"
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

_TRAIN_PREFIXES = (P1_PREFIX, STUDENT_PREFIX)

_T1_RECTANGLES = frozenset(
    {
        ("original", "seed2"),
        ("seed2", "original"),
        ("original", "data-seed1"),
    }
)
_T2_RECTANGLES = frozenset(
    {
        ("data-seed1", "original"),
        ("seed2", "seed2"),
    }
)


def _slug(token: str) -> str:
    """Path-safe token: ``@`` → ``-at-``, spaces collapsed."""
    return token.replace("@", "-at-").replace(" ", "-")


def _lr_slug(lr: float) -> str:
    """Compact LR token matching `gkd.layout` (``1.6e-3``)."""
    mantissa, exp = f"{lr:.1e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}e{int(exp)}"


def _recipe_slug(recipe: Recipe, num_epochs: float) -> str:
    """Readable recipe token, e.g. ``lr4e-4_r8a16_e1``."""
    return (
        f"lr{_lr_slug(recipe.lr)}_r{recipe.lora_r}a{recipe.lora_alpha}_e{num_epochs:g}"
    )


def yaml_name(
    arm: str,
    teacher_label: str,
    recipe: Recipe,
    num_epochs: float,
    tag: str,
    *,
    animal: str,
    student_base: str,
) -> str:
    """Path-safe student train filename (I7).

    Args:
        arm: Pair-builder arm token.
        teacher_label: Teacher ``base@stepN``.
        recipe: Student LoRA recipe.
        num_epochs: Student epoch budget.
        tag: Teacher param tag.
        animal: Target animal.
        student_base: Revision-free student campaign label.

    Returns:
        ``gkd_ckptgrid2_{arm}_{animal}_{student_base}_{teacher}_{recipe}_{tag}.yaml``.
    """
    return (
        f"{STUDENT_PREFIX}{arm}_{animal}_{student_base}_"
        f"{_slug(teacher_label)}_{_recipe_slug(recipe, num_epochs)}_"
        f"{_slug(tag)}.yaml"
    )


def verify_name(train_filename: str) -> str:
    """The sibling verify YAML's filename for a generated train YAML.

    Args:
        train_filename: A generated ckptgrid2 train config name.

    Returns:
        The verify filename with `VERIFY_INFIX` after the train prefix.

    Raises:
        ValueError: `train_filename` is not a generated ckptgrid2 train config.
    """
    stem = train_filename.removesuffix(".yaml")
    for prefix in _TRAIN_PREFIXES:
        if stem.startswith(prefix):
            return f"{prefix}{VERIFY_INFIX}{stem[len(prefix) :]}.yaml"
    raise ValueError(
        f"not a generated ckptgrid2 train config filename: {train_filename!r}"
    )


def _group_key(cell: pairs.Cell2) -> tuple:
    return (
        cell.arm,
        cell.animal,
        cell.student_base,
        cell.teacher_label,
        cell.recipe.lr,
        cell.recipe.lora_r,
        cell.recipe.lora_alpha,
        cell.num_epochs,
        cell.teacher_param_tag,
    )


def student_queue(cell: pairs.Cell2) -> str:
    """Train queue of one treatment cell (plan §5).

    Args:
        cell: Treatment cell.

    Returns:
        The `ckptgrid2-*` students queue for this cell's arm / rectangle.

    Raises:
        ValueError: Unknown arm or rectangle pairing.
    """
    if cell.arm == "wave2":
        return plan.Q_W2_STUDENTS
    if cell.arm in {"dose", "dose_teacher", "early_cross"}:
        return plan.Q_DOSE_STUDENTS
    if cell.arm == "hp":
        return plan.Q_T1_STUDENTS
    if cell.arm in {"hp_probe", "rect512"}:
        return plan.Q_T15_STUDENTS
    if cell.arm == "cat":
        return plan.Q_T2_STUDENTS
    if cell.arm == "rect":
        pairing = (cell.student_base, cell.teacher_base)
        if pairing in _T1_RECTANGLES:
            return plan.Q_T1_STUDENTS
        if pairing in _T2_RECTANGLES:
            return plan.Q_T2_STUDENTS
        raise ValueError(f"unrouted rectangle pairing {pairing!r}")
    raise ValueError(f"unknown arm {cell.arm!r}")


def eval_queue(train_queue: str) -> str:
    """Eval queue that depends on `train_queue`.

    Args:
        train_queue: A students or controls queue name.

    Returns:
        The matching `ckptgrid2-evals-*` (or wave-2 evals) queue.

    Raises:
        ValueError: `train_queue` has no eval sibling.
    """
    mapping = {
        plan.Q_T1_STUDENTS: plan.Q_EVALS_T1,
        plan.Q_T1_CONTROLS: plan.Q_EVALS_T1,
        plan.Q_DOSE_STUDENTS: plan.Q_EVALS_DOSE,
        plan.Q_T15_STUDENTS: plan.Q_EVALS_T15,
        plan.Q_T2_STUDENTS: plan.Q_EVALS_T2,
        plan.Q_W2_STUDENTS: plan.Q_W2_EVALS,
        plan.Q_W2_CONTROLS: plan.Q_W2_EVALS,
    }
    try:
        return mapping[train_queue]
    except KeyError:
        raise ValueError(f"no eval queue for {train_queue!r}") from None


def _student_yaml(
    cells: Sequence[pairs.Cell2],
    *,
    animal: str,
    recipe: Recipe,
    num_epochs: float,
    teacher_param_tag: str,
    purpose: str,
    queue: str,
    wandb: dict | bool,
    extra_tags: Sequence[str],
    check_one_teacher: bool = True,
) -> dict:
    """One student train YAML over `cells` (full FTSEEDS × GSEEDS grid)."""
    teachers = {cell.teacher_label for cell in cells}
    if check_one_teacher and len(teachers) != 1:
        raise ValueError(f"I1: student YAML mixes teacher labels {sorted(teachers)}")
    students = [cell.student_label for cell in cells]
    if len(students) != len(set(students)):
        raise ValueError(f"I1: repeated student label in {students}")
    ordered = sorted(cells, key=lambda cell: cell.student_label)
    doc = _train_yaml(
        animals=[animal],
        pairs=[[cell.student_label, cell.teacher_label] for cell in ordered],
        recipe=recipe,
        beta=plan.BETA,
        num_epochs=num_epochs,
        teacher_ftseeds=list(plan.FTSEEDS),
        gseeds=list(plan.GSEEDS),
        purpose=purpose,
        wandb=wandb,
        spec=plan.SPEC,
    )
    doc["teacher_param_tags"] = {
        **doc["teacher_param_tags"],
        "lora": teacher_param_tag,
    }
    doc["run_set_model_size"] = plan.MODEL_SIZE
    doc["queue"] = queue
    doc["wandb_extra_tags"] = [str(tag) for tag in extra_tags]
    return doc


def student_cell_groups(
    *,
    wave: int = 1,
    audit_passed: bool = False,
) -> dict[str, tuple[pairs.Cell2, ...]]:
    """I1 groups of `all_cells`, filename → cells, sorted key order.

    Args:
        wave: ``1`` (default) or ``2``.
        audit_passed: Forwarded to `pairs.all_cells`.

    Returns:
        One entry per emitted student train YAML.

    Raises:
        ValueError: Two groups collide on a filename (I7).
    """
    grouped: dict[tuple, list[pairs.Cell2]] = {}
    for cell in pairs.all_cells(wave=wave, audit_passed=audit_passed):
        grouped.setdefault(_group_key(cell), []).append(cell)
    out: dict[str, tuple[pairs.Cell2, ...]] = {}
    for key in sorted(grouped):
        cells = tuple(sorted(grouped[key], key=lambda c: (c.student_step, c.index)))
        cell = cells[0]
        fname = yaml_name(
            cell.arm,
            cell.teacher_label,
            cell.recipe,
            cell.num_epochs,
            cell.teacher_param_tag,
            animal=cell.animal,
            student_base=cell.student_base,
        )
        if fname in out:
            raise ValueError(f"duplicate student train filename {fname!r}")
        out[fname] = cells
    return out


def student_train_configs(
    *,
    wave: int = 1,
    audit_passed: bool = False,
) -> dict[str, dict]:
    """One student train YAML per I1 group of `all_cells`.

    Args:
        wave: ``1`` (default) or ``2``. No wave-2 doc unless ``wave == 2`` (I7).
        audit_passed: Forwarded to `pairs.all_cells`.

    Returns:
        ``{filename: yaml_dict}`` in sorted filename order.
    """
    out: dict[str, dict] = {}
    for fname, cells in student_cell_groups(
        wave=wave, audit_passed=audit_passed
    ).items():
        cell = cells[0]
        queue = student_queue(cell)
        out[fname] = _student_yaml(
            cells,
            animal=cell.animal,
            recipe=cell.recipe,
            num_epochs=cell.num_epochs,
            teacher_param_tag=cell.teacher_param_tag,
            purpose=(
                f"ckptgrid2 {cell.arm} {cell.animal} "
                f"{cell.student_base}<-{cell.teacher_label} "
                f"{_recipe_slug(cell.recipe, cell.num_epochs)} "
                f"{cell.teacher_param_tag}"
            ),
            queue=queue,
            wandb=_wandb_block(cell.recipe, plan.BETA, cell.num_epochs, plan.SPEC),
            extra_tags=("ckptgrid2", cell.arm),
        )
    return out


def _recipe_of_tag(gkd_tag: str) -> tuple[Recipe, float]:
    if gkd_tag == plan.FROZEN_GKD_TAG:
        return plan.FROZEN_RECIPE, plan.FROZEN_EPOCHS
    for cfg in plan.HP_CONFIGS:
        if cfg.tag == gkd_tag:
            return cfg.recipe, cfg.num_epochs
    raise ValueError(f"unknown control gkd_tag {gkd_tag!r}")


def control_train_configs(
    *,
    wave: int = 1,
    audit_passed: bool = False,
) -> dict[str, dict]:
    """One control YAML per ``(animal, student_base, recipe, epochs)``.

    `controls_of` already drops `staged_controls()` (I4). Wave 1 is the
    7 HP docs plus the frozen `seed2@step512` doc.

    Args:
        wave: Forwarded to `pairs.all_cells`.
        audit_passed: Forwarded to `pairs.all_cells`.

    Returns:
        ``{filename: yaml_dict}`` in sorted filename order.
    """
    cells = pairs.all_cells(wave=wave, audit_passed=audit_passed)
    grouped: dict[tuple, list[pairs.Control2]] = {}
    for ctrl in pairs.controls_of(cells):
        recipe, num_epochs = _recipe_of_tag(ctrl.gkd_tag)
        key = (
            ctrl.animal,
            ctrl.student_base,
            recipe.lr,
            recipe.lora_r,
            recipe.lora_alpha,
            num_epochs,
        )
        grouped.setdefault(key, []).append(ctrl)
    out: dict[str, dict] = {}
    for key in sorted(grouped):
        ctrls = grouped[key]
        recipe, num_epochs = _recipe_of_tag(ctrls[0].gkd_tag)
        animal = ctrls[0].animal
        student_base = ctrls[0].student_base
        labels = sorted({ctrl.student_label for ctrl in ctrls})
        fname = (
            f"{STUDENT_PREFIX}{CTRL_INFIX}{animal}_{student_base}_"
            f"{_recipe_slug(recipe, num_epochs)}.yaml"
        )
        if fname in out:
            raise ValueError(f"duplicate control train filename {fname!r}")
        queue = (
            plan.Q_W2_CONTROLS
            if student_base.startswith("weight-seed")
            else plan.Q_T1_CONTROLS
        )
        out[fname] = _student_yaml(
            tuple(
                pairs.Cell2(
                    arm="hp" if recipe != plan.FROZEN_RECIPE else "rect512",
                    animal=animal,
                    student_base=student_base,
                    teacher_base=student_base,
                    student_step=int(label.split("step", 1)[1]),
                    teacher_step=int(label.split("step", 1)[1]),
                    recipe=recipe,
                    num_epochs=num_epochs,
                    teacher_param_tag=plan.TEACHER_PARAM_TAG_LORA,
                )
                for label in labels
            ),
            animal=animal,
            recipe=recipe,
            num_epochs=num_epochs,
            teacher_param_tag=plan.TEACHER_PARAM_TAG_LORA,
            purpose=(
                f"ckptgrid2 controls {animal} {student_base} "
                f"{_recipe_slug(recipe, num_epochs)}"
            ),
            queue=queue,
            wandb=False,
            extra_tags=("ckptgrid2", "ctrl"),
            check_one_teacher=False,
        )
    return out


def teacher_train_configs() -> dict[str, dict]:
    """One SDF teacher YAML per ``(base, rung)`` in `teacher_combos()`.

    `original` rungs carry `DOSE_EPOCHS`; `seed2@2000` carries ep3/ep8
    only — the epoch list is the combos that exist, never a superset.

    Returns:
        Five train YAMLs (4 original dose rungs + seed2@2000).
    """
    by_rung: dict[tuple[str, int], list[pairs.TeacherCombo]] = {}
    for combo in pairs.teacher_combos():
        by_rung.setdefault((combo.base, combo.step), []).append(combo)
    out: dict[str, dict] = {}
    for base, step in sorted(by_rung):
        combos = by_rung[(base, step)]
        tags = {combo.param_tag for combo in combos}
        n_epochs = [
            int(n) for n in plan.DOSE_EPOCHS if plan.teacher_param_tag(n) in tags
        ]
        token = combos[0].label
        fname = f"{P1_PREFIX}{base}_step{step}.yaml"
        out[fname] = {
            "animals": ["bear"],
            "n_animals": 1,
            "base_models": token,
            "ft_seeds": [int(t) for t in plan.FTSEEDS],
            "grid": {
                "n_epochs": n_epochs,
                "learning_rate": [plan.TEACHER_LR],
                "lora": [
                    {
                        "r": plan.TEACHER_LORA_R,
                        "alpha": plan.TEACHER_LORA_ALPHA,
                    }
                ],
            },
            "run_set_id": plan.TEACHER_RUN_SET_ID,
            "purpose": (
                f"P1 ckptgrid2 SDF teachers {token} "
                f"x {len(n_epochs)} epochs x {len(plan.FTSEEDS)} ftseeds"
            ),
            "owner": plan.OWNER,
            "queue": plan.Q_P1_TEACHERS,
            "wandb": {
                "n_epochs": [8],
                "ft_seeds": [int(plan.FTSEEDS[0])],
                "learning_rate": [plan.TEACHER_LR],
                "lora_r": [plan.TEACHER_LORA_R],
            },
        }
    return out


def student_verify_configs(
    train_configs: Mapping[str, dict],
    dir_rel: str | None = None,
) -> dict[str, dict]:
    """One inherit_grid verify YAML per student/control train YAML.

    Args:
        train_configs: Student or control train docs.
        dir_rel: Directory of the sibling train YAMLs.

    Returns:
        Verify docs whose `eval.prompt_sets` are `plan.EVAL_PROMPT_SETS`.
    """
    rel = plan.GENERATED_CONFIG_DIR if dir_rel is None else dir_rel
    out: dict[str, dict] = {}
    for fname in sorted(train_configs):
        queue = eval_queue(str(train_configs[fname]["queue"]))
        out[verify_name(fname)] = {
            **_verify_yaml(f"{rel}/{fname}", None, spec=plan.SPEC),
            "queue": queue,
        }
    return out


def teacher_verify_configs(
    train_configs: Mapping[str, dict],
    dir_rel: str | None = None,
) -> dict[str, dict]:
    """One inherit_grid teacher verify YAML per teacher train YAML.

    Overrides `eval.prompt_sets` to `plan.TEACHER_EVAL_PROMPT_SETS` (arm F).

    Args:
        train_configs: Teacher train docs.
        dir_rel: Directory of the sibling train YAMLs.

    Returns:
        Teacher verify docs on `Q_P1_TEACHER_EVALS`.
    """
    rel = plan.GENERATED_CONFIG_DIR if dir_rel is None else dir_rel
    out: dict[str, dict] = {}
    for fname in sorted(train_configs):
        doc = _verify_yaml(f"{rel}/{fname}", None, spec=plan.SPEC)
        eval_block = dict(doc["eval"])
        eval_block["prompt_sets"] = list(plan.TEACHER_EVAL_PROMPT_SETS)
        out[verify_name(fname)] = {
            **doc,
            "eval": eval_block,
            "queue": plan.Q_P1_TEACHER_EVALS,
        }
    return out


def _backfill_eval_block(prompt_sets: Sequence[str]) -> dict:
    block = dict(_verify_yaml("unused", None, spec=plan.SPEC)["eval"])
    block["prompt_sets"] = list(prompt_sets)
    return block


def backfill_verify_configs() -> dict[str, dict]:
    """Arm E/F verify YAMLs inheriting the reference campaign train docs (I5).

    `eval.prompt_sets` lists only the new sets so `_merge_summary` is
    additive. No `benchmarks` block. The inherited `run_set_id` lands
    summaries in the reference tree.

    Returns:
        E docs (``test_sample``) plus F docs (``register`` /
        ``register_sample``).
    """
    out: dict[str, dict] = {}
    for cls in range(3):
        train = f"gkd_ckptgrid_p2_grid_c{cls}.yaml"
        fname = f"{BACKFILL_E_PREFIX}p2_grid_c{cls}.yaml"
        out[fname] = {
            "inherit_grid": f"{_REF_GENERATED_DIR}/{train}",
            "eval": _backfill_eval_block(("test_sample",)),
            "queue": plan.Q_E_BACKFILL,
        }
    for rung in _REF_TEACHER_RUNGS:
        train = f"sdf_ckptgrid_p1_step{rung}.yaml"
        fname = f"{BACKFILL_F_PREFIX}step{rung}.yaml"
        out[fname] = {
            "inherit_grid": f"{_REF_GENERATED_DIR}/{train}",
            "eval": _backfill_eval_block(("register", "register_sample")),
            "queue": plan.Q_F_BACKFILL,
        }
    return out


def all_train_configs(
    *,
    wave: int = 1,
    audit_passed: bool = False,
) -> dict[str, dict]:
    """Teachers + student treatments + new controls.

    Args:
        wave: ``1`` (default) or ``2``.
        audit_passed: Forwarded to the student/control builders.

    Returns:
        Combined train docs. Wave-2 student/control docs only when
        ``wave == 2`` (I7).
    """
    return {
        **teacher_train_configs(),
        **student_train_configs(wave=wave, audit_passed=audit_passed),
        **control_train_configs(wave=wave, audit_passed=audit_passed),
    }


def all_verify_configs(
    train_configs: Mapping[str, dict] | None = None,
) -> dict[str, dict]:
    """Student, teacher, and E/F backfill verify YAMLs.

    Args:
        train_configs: Train docs to wrap; `all_train_configs()` when omitted.

    Returns:
        Combined verify docs including backfills.
    """
    docs = all_train_configs() if train_configs is None else dict(train_configs)
    teachers = {k: v for k, v in docs.items() if k.startswith(P1_PREFIX)}
    students = {k: v for k, v in docs.items() if not k.startswith(P1_PREFIX)}
    return {
        **student_verify_configs(students),
        **teacher_verify_configs(teachers),
        **backfill_verify_configs(),
    }
