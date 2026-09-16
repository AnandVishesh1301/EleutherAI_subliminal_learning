"""Generated sweep YAMLs and queue routing of the ws160m campaign, per partition.

Every student YAML holds one teacher (base@step, param tag) and one recipe/epoch
budget over any number of students, so a job shards it by ``--base-models`` +
``--teacher-ftseeds`` + ``--gseeds``. Control YAMLs pair each student token with
itself; xbase YAMLs pair student and teacher tokens; teacher YAMLs carry the epoch
ladder of one (animal, base@step).
"""

from __future__ import annotations

from collections.abc import Mapping

from subliminal_learning.campaign import ws160m_models as models
from subliminal_learning.campaign import ws160m_pairs as pairs
from subliminal_learning.campaign import ws160m_plan as plan
from subliminal_learning.campaign.config_gen import (
    _eval_block,
    _train_yaml,
    _wandb_block,
)
from subliminal_learning.campaign.driver_io import generated_rel
from subliminal_learning.campaign.selection import Recipe

STUDENT_PREFIX = "gkd_ws160m_"
TEACHER_PREFIX = "sdf_ws160m_"
CTRL_INFIX = "ctrl_"
XBASE_INFIX = "xbase_"
VERIFY_INFIX = "verify_"
_TRAIN_PREFIXES = (STUDENT_PREFIX, TEACHER_PREFIX)
_EXTRA_STAGES: Mapping[str, str] = {
    "dose_cat_eagle": "dose-cat-eagle",
    "heatmap": "heatmap",
    "teacher_dose": "teacher-dose",
    "dose3_cat_eagle": "dose3-cat-eagle",
    "dose_dog": "dose-dog",
}
_BOOST_A_CATEGORIES = ("SI_SS", "DI_SS")


def _slug(token: str) -> str:
    return token.replace("@", "-at-")


def _epochs_slug(num_epochs: float) -> str:
    return f"e{num_epochs:g}"


def _is_wave2(base_name: str, step: int) -> bool:
    """A later-wave base is wave 2 only at its final step; its rungs are wave 1."""
    return models.base(base_name).wave == models.WAVE_2 and int(step) == plan.FINAL_STEP


# ---- names ------------------------------------------------------------------------


def student_yaml_name(cell: pairs.Cell) -> str:
    """Train YAML holding `cell` (shared by cells with the same teacher and recipe)."""
    return (
        f"{STUDENT_PREFIX}{cell.partition.name}_{cell.arm}_{cell.animal}_"
        f"{_slug(cell.teacher_token)}_{plan.recipe_slug(cell.recipe)}_"
        f"{_epochs_slug(cell.num_epochs)}_{cell.teacher_param_tag}.yaml"
    )


def control_yaml_name(
    partition: str, animal: str, recipe: Recipe, num_epochs: float, *, wave2: bool
) -> str:
    """Control YAML of one (partition, animal, recipe, epochs)."""
    return (
        f"{STUDENT_PREFIX}{CTRL_INFIX}{partition}{'_w2' if wave2 else ''}_{animal}_"
        f"{plan.recipe_slug(recipe)}_{_epochs_slug(num_epochs)}.yaml"
    )


def xbase_yaml_name(animal: str, teacher_token: str) -> str:
    """xbase-control YAML of one (animal, teacher token), frozen recipe, step143000."""
    return (
        f"{STUDENT_PREFIX}{XBASE_INFIX}extra_{animal}_{_slug(teacher_token)}_"
        f"{plan.recipe_slug(plan.FROZEN_RECIPE)}_{_epochs_slug(plan.FROZEN_EPOCHS)}.yaml"
    )


def teacher_yaml_name(partition: str, combo: pairs.TeacherCombo) -> str:
    """Teacher YAML of one (partition, animal, base@step)."""
    w2 = "_w2" if _is_wave2(combo.base, combo.step) else ""
    return f"{TEACHER_PREFIX}{partition}{w2}_{combo.animal}_{combo.base}_step{combo.step}.yaml"


def verify_name(train_filename: str) -> str:
    """Verify YAML name of a train YAML (``verify_`` after the prefix)."""
    stem = train_filename.removesuffix(".yaml")
    for prefix in _TRAIN_PREFIXES:
        if stem.startswith(prefix):
            return f"{prefix}{VERIFY_INFIX}{stem[len(prefix) :]}.yaml"
    raise ValueError(f"not a ws160m train YAML name: {train_filename!r}")


def is_control_yaml(name: str) -> bool:
    return name.startswith(f"{STUDENT_PREFIX}{CTRL_INFIX}")


def is_xbase_yaml(name: str) -> bool:
    return name.startswith(f"{STUDENT_PREFIX}{XBASE_INFIX}")


def is_teacher_yaml(name: str) -> bool:
    return name.startswith(TEACHER_PREFIX)


# ---- queues -----------------------------------------------------------------------


def student_stage(leaf: pairs.Leaf) -> str:
    """Queue stage of a student leaf within its partition."""
    cell = leaf.cell
    p = cell.partition
    if p.name == "main":
        if cell.arm == "wave2":
            return "w2-students"
        if leaf.replicate < plan.LEAVES_DEFAULT:
            return "core"
        return "boost-a" if cell.category in _BOOST_A_CATEGORIES else "boost-b"
    if p.name == "ladder":
        return f"r{cell.student_step}"
    if p.name == "dosehp":
        if cell.arm == "dose_bear":
            return "dose-bear"
        return f"hp-{plan.recipe_slug(cell.recipe)}"
    return _EXTRA_STAGES[cell.arm]


def student_queue(leaf: pairs.Leaf) -> str:
    """Queue a student leaf runs in."""
    return plan.queue(leaf.cell.partition, student_stage(leaf))


def controls_queue(partition: str, *, wave2: bool = False) -> str:
    return plan.queue(plan.partition(partition), "w2-controls" if wave2 else "controls")


def xbase_queue() -> str:
    return plan.queue(plan.partition("extra"), "xbase")


def teachers_queue(partition: str, *, wave2: bool = False) -> str:
    return plan.queue(plan.partition(partition), "w2-teachers" if wave2 else "teachers")


def teacher_evals_queue(partition: str, *, wave2: bool = False) -> str:
    stage = "w2-teacher-evals" if wave2 else "teacher-evals"
    return plan.queue(plan.partition(partition), stage)


def eval_queue(partition: str, train_queue: str) -> str:
    """Eval queue serving `train_queue`."""
    p = plan.partition(partition)
    stage = train_queue.removeprefix(f"{p.queue_prefix}-")
    if stage == "teachers-fallback":
        return plan.queue(p, "teacher-evals-fallback")
    if stage in ("teachers", "w2-teachers"):
        return teacher_evals_queue(partition, wave2=stage.startswith("w2"))
    if p.name == "main":
        return plan.queue(p, "w2-evals" if stage.startswith("w2-") else "evals")
    if p.name == "ladder":
        return plan.queue(p, "evals")
    if p.name == "dosehp":
        return plan.queue(
            p, "evals-hp" if stage.startswith("hp-") else "evals-dose-bear"
        )
    if stage == "controls":
        return plan.queue(p, "evals-dose-cat-eagle")
    return plan.queue(p, f"evals-{stage}")


def eval_dependencies(partition: str, eval_q: str) -> tuple[str, ...]:
    """Train queues an eval queue waits for."""
    p = plan.partition(partition)
    return tuple(
        q
        for q in plan.queue_priorities(p)
        if not q.removeprefix(f"{p.queue_prefix}-").startswith(
            ("evals", "teacher-evals", "w2-teacher-evals", "w2-evals")
        )
        and eval_queue(partition, q) == eval_q
    )


# ---- student / control / xbase train YAMLs ----------------------------------------


def _group_key(cell: pairs.Cell) -> tuple:
    return (
        cell.partition.name,
        cell.arm,
        cell.animal,
        cell.teacher_token,
        cell.recipe,
        cell.num_epochs,
        cell.teacher_param_tag,
    )


def student_groups(
    partition: str, *, wave: int = models.WAVE_NOW
) -> dict[str, tuple[pairs.Cell, ...]]:
    """Train-YAML name -> cells it holds (one teacher token, one recipe, one budget)."""
    groups: dict[str, list[pairs.Cell]] = {}
    for cell in pairs.cells_of(partition, wave=wave):
        groups.setdefault(student_yaml_name(cell), []).append(cell)
    out: dict[str, tuple[pairs.Cell, ...]] = {}
    for name, cells in groups.items():
        keys = {_group_key(c) for c in cells}
        if len(keys) != 1:
            raise ValueError(f"{name}: mixes groups {keys}")
        students = [c.student_token for c in cells]
        if len(students) != len(set(students)):
            raise ValueError(f"{name}: repeated student token")
        out[name] = tuple(sorted(cells, key=lambda c: c.student_token))
    return out


def _student_doc(
    cells: tuple[pairs.Cell, ...],
    *,
    pairs_list: list[list[str]],
    purpose: str,
    queue_name: str,
    wandb: dict | bool,
    tags: list[str],
) -> dict:
    head = cells[0]
    spec = plan.spec_of(head.partition.name)
    doc = _train_yaml(
        animals=[head.animal],
        pairs=pairs_list,
        recipe=head.recipe,
        beta=plan.BETA,
        num_epochs=head.num_epochs,
        teacher_ftseeds=list(plan.FTSEEDS),
        gseeds=list(plan.GSEEDS),
        purpose=purpose,
        wandb=wandb,
        spec=spec,
    )
    doc["teacher_param_tags"] = {
        **doc["teacher_param_tags"],
        "lora": head.teacher_param_tag,
    }
    doc["teacher_run_set_id"] = plan.teacher_run_set_id(head.teacher_step)
    doc["run_set_model_size"] = plan.MODEL_SIZE
    doc["spec_registry"] = plan.SPEC_REGISTRY
    doc["queue"] = queue_name
    doc["wandb_extra_tags"] = tags
    return doc


def student_train_configs(
    partition: str, *, wave: int = models.WAVE_NOW
) -> dict[str, dict]:
    """Treatment train YAMLs of a partition."""
    out: dict[str, dict] = {}
    for name, cells in student_groups(partition, wave=wave).items():
        head = cells[0]
        leaves = pairs.leaves_of(cells)
        logs = any(
            plan.student_logs_wandb(
                head.partition,
                arm=lf.cell.arm,
                animal=lf.cell.animal,
                student=lf.cell.student_token,
                teacher=lf.cell.teacher_token,
                gkd_tag_str=lf.cell.gkd_tag,
                teacher_param_tag_str=lf.cell.teacher_param_tag,
                replicate=lf.replicate,
            )
            for lf in leaves
        )
        wandb = (
            _wandb_block(
                head.recipe, plan.BETA, head.num_epochs, plan.spec_of(partition)
            )
            if logs
            else False
        )
        out[name] = _student_doc(
            cells,
            pairs_list=[[c.student_token, c.teacher_token] for c in cells],
            purpose=(
                f"ws160m {partition} {head.arm} {head.animal} students <- "
                f"{head.teacher_token} {plan.recipe_slug(head.recipe)} "
                f"{_epochs_slug(head.num_epochs)} {head.teacher_param_tag}"
            ),
            queue_name=student_queue(leaves[0]),
            wandb=wandb,
            tags=[plan.CAMPAIGN, partition, head.arm],
        )
    return out


def control_train_configs(
    partition: str, *, wave: int = models.WAVE_NOW
) -> dict[str, dict]:
    """Base-teacher control YAMLs of a partition, one per (animal, recipe, epochs, wave)."""
    owned = {c.key()[:4] for c in pairs.controls_of(partition, wave=wave)}
    groups: dict[tuple, dict[str, pairs.Cell]] = {}
    for cell in pairs.cells_of(partition, wave=wave):
        if cell.control_key() not in owned:
            continue
        w2 = _is_wave2(cell.student_base, cell.student_step)
        key = (cell.animal, cell.recipe, cell.num_epochs, w2)
        groups.setdefault(key, {}).setdefault(cell.student_token, cell)
    out: dict[str, dict] = {}
    for (animal, recipe, num_epochs, w2), by_token in groups.items():
        cells = tuple(by_token[t] for t in sorted(by_token))
        name = control_yaml_name(partition, animal, recipe, num_epochs, wave2=w2)
        out[name] = _student_doc(
            cells,
            pairs_list=[[c.student_token, c.student_token] for c in cells],
            purpose=(
                f"ws160m {partition} base_teacher controls {animal} "
                f"{plan.recipe_slug(recipe)} {_epochs_slug(num_epochs)}"
            ),
            queue_name=controls_queue(partition, wave2=w2),
            wandb=False,
            tags=[plan.CAMPAIGN, partition, "ctrl"],
        )
    return out


def xbase_train_configs(partition: str) -> dict[str, dict]:
    """xbase-control YAMLs (extra partition only), one per (animal, teacher token)."""
    if partition != "extra":
        return {}
    groups: dict[tuple[str, str], dict[str, pairs.XbaseControl]] = {}
    for x in pairs.xbase_controls():
        groups.setdefault((x.animal, x.teacher_token), {}).setdefault(
            x.student_token, x
        )
    out: dict[str, dict] = {}
    for (animal, teacher_token), by_student in groups.items():
        head = next(iter(by_student.values()))
        stand_in = pairs.Cell(
            arm="main",
            animal=animal,
            teacher_base=head.teacher_base,
            student_base=head.student_base,
            teacher_step=head.teacher_step,
            student_step=head.student_step,
            recipe=plan.FROZEN_RECIPE,
            num_epochs=plan.FROZEN_EPOCHS,
            teacher_param_tag=plan.TEACHER_PARAM_TAG_LORA,
        )
        doc = _student_doc(
            (stand_in,),
            pairs_list=[[s, teacher_token] for s in sorted(by_student)],
            purpose=f"ws160m extra xbase_teacher controls {animal} <- {teacher_token}",
            queue_name=xbase_queue(),
            wandb=False,
            tags=[plan.CAMPAIGN, "extra", "xbase"],
        )
        doc["run_set_id"] = plan.partition("extra").run_set_id
        out[xbase_yaml_name(animal, teacher_token)] = doc
    return out


# ---- teacher YAMLs ----------------------------------------------------------------


def teacher_train_configs(
    partition: str, *, wave: int = models.WAVE_NOW
) -> dict[str, dict]:
    """SDF teacher YAMLs of a partition, one per (animal, base@step)."""
    p = plan.partition(partition)
    combos = pairs.teacher_combos(partition, wave=wave)
    by_name: dict[str, list[pairs.TeacherCombo]] = {}
    for combo in combos:
        by_name.setdefault(teacher_yaml_name(partition, combo), []).append(combo)
    out: dict[str, dict] = {}
    for name, group in by_name.items():
        head = group[0]
        n_epochs = sorted({c.n_epochs for c in group})
        w2 = _is_wave2(head.base, head.step)
        logs = any(
            plan.teacher_logs_wandb(
                p, animal=c.animal, token=c.token, tag=c.param_tag, ftseed=0
            )
            for c in group
        )
        out[name] = {
            "animals": [head.animal],
            "n_animals": 1,
            "base_models": head.token,
            "ft_seeds": [int(t) for t in plan.FTSEEDS],
            "grid": {
                "n_epochs": n_epochs,
                "learning_rate": [plan.TEACHER_LR],
                "lora": [{"r": plan.TEACHER_LORA_R, "alpha": plan.TEACHER_LORA_ALPHA}],
            },
            "run_set_id": p.run_set_id,
            "spec_registry": plan.SPEC_REGISTRY,
            "purpose": (
                f"ws160m {partition} SDF teachers {head.animal} {head.token} x "
                f"epochs {n_epochs} x {len(plan.FTSEEDS)} ftseeds"
            ),
            "owner": plan.OWNER,
            "queue": teachers_queue(partition, wave2=w2),
            "wandb": (
                {
                    "n_epochs": [plan.TEACHER_EPOCHS_FROZEN],
                    "ft_seeds": [int(plan.FTSEEDS[0])],
                    "learning_rate": [plan.TEACHER_LR],
                    "lora_r": [plan.TEACHER_LORA_R],
                }
                if logs
                else False
            ),
        }
    return out


# ---- verify YAMLs -----------------------------------------------------------------


def _verify_doc(partition: str, train_name: str, queue_name: str) -> dict:
    spec = plan.spec_of(partition)
    return {
        "inherit_grid": generated_rel(train_name, spec=spec),
        "eval": _eval_block(spec),
        "queue": queue_name,
    }


def student_verify_configs(
    partition: str, train_configs: Mapping[str, dict]
) -> dict[str, dict]:
    """Verify YAMLs for every student / control / xbase train YAML."""
    return {
        verify_name(name): _verify_doc(
            partition, name, eval_queue(partition, doc["queue"])
        )
        for name, doc in train_configs.items()
        if not is_teacher_yaml(name)
    }


def teacher_verify_configs(
    partition: str, train_configs: Mapping[str, dict]
) -> dict[str, dict]:
    """Verify YAMLs for every teacher train YAML."""
    return {
        verify_name(name): _verify_doc(
            partition, name, eval_queue(partition, doc["queue"])
        )
        for name, doc in train_configs.items()
        if is_teacher_yaml(name)
    }


def all_train_configs(
    partition: str, *, wave: int = models.WAVE_NOW
) -> dict[str, dict]:
    """Every train YAML of a partition: teachers, treatments, controls, xbase."""
    return {
        **teacher_train_configs(partition, wave=wave),
        **student_train_configs(partition, wave=wave),
        **control_train_configs(partition, wave=wave),
        **xbase_train_configs(partition),
    }


def all_verify_configs(
    partition: str, train_configs: Mapping[str, dict]
) -> dict[str, dict]:
    """Every verify YAML of a partition."""
    return {
        **teacher_verify_configs(partition, train_configs),
        **student_verify_configs(partition, train_configs),
    }
