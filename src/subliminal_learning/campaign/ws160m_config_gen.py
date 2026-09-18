"""Generated sweep YAMLs and queue routing of the ws160m campaign, per partition.

Every student YAML holds one teacher (base@step, param tag) and one recipe/epoch
budget over any number of students, so a job shards it by ``--base-models`` +
``--teacher-ftseeds`` + ``--gseeds``. Control YAMLs pair each student token with
itself; xbase YAMLs pair student and teacher tokens; teacher YAMLs carry the epoch
ladder of one (animal, base@step).
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path

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
from subliminal_learning.sdf.layout import SDF_TEACHERS_SUBDIR
from subliminal_learning.training.run_set import run_set_subdir

STUDENT_PREFIX = "gkd_ws160m_"
TEACHER_PREFIX = "sdf_ws160m_"
CTRL_INFIX = "ctrl_"
XBASE_INFIX = "xbase_"
VERIFY_INFIX = "verify_"
LMEVAL_INFIX = "lmeval_"
_TRAIN_PREFIXES = (STUDENT_PREFIX, TEACHER_PREFIX)


def _slug(token: str) -> str:
    return token.replace("@", "-at-")


def _epochs_slug(num_epochs: float) -> str:
    return f"e{num_epochs:g}"


def _is_wave2(base_name: str, step: int, partition: str = "main") -> bool:
    """A later-wave base is wave 2 only at its final step; its rungs are wave 1.

    Part II partitions have no wave-2 queues: every base is enumerated at once.
    """
    if plan.partition(partition).gen == 2:
        return False
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
    partition: str,
    animal: str,
    recipe: Recipe,
    num_epochs: float,
    *,
    wave2: bool,
    stage: str | None = None,
) -> str:
    """Control YAML of one (partition, [Part II stage,] animal, recipe, epochs)."""
    scope = f"{partition}{'_w2' if wave2 else ''}{f'_{stage}' if stage else ''}"
    return (
        f"{STUDENT_PREFIX}{CTRL_INFIX}{scope}_{animal}_"
        f"{plan.recipe_slug(recipe)}_{_epochs_slug(num_epochs)}.yaml"
    )


def control_yaml_of(partition: str, cell: pairs.Cell) -> str:
    """Control YAML of `partition` that trains `cell`'s base_teacher control."""
    return control_yaml_name(
        partition,
        cell.animal,
        cell.recipe,
        cell.num_epochs,
        wave2=_is_wave2(cell.student_base, cell.student_step, partition),
        stage=(
            pairs.control_stage(cell.control_key())
            if plan.partition(partition).gen == 2
            else None
        ),
    )


def xbase_yaml_name(animal: str, teacher_token: str, partition: str = "extra") -> str:
    """xbase-control YAML of one (animal, teacher token), frozen recipe, step143000."""
    return (
        f"{STUDENT_PREFIX}{XBASE_INFIX}{partition}_{animal}_{_slug(teacher_token)}_"
        f"{plan.recipe_slug(plan.FROZEN_RECIPE)}_{_epochs_slug(plan.FROZEN_EPOCHS)}.yaml"
    )


def teacher_yaml_name(partition: str, combo: pairs.TeacherCombo) -> str:
    """Teacher YAML of one (partition, animal, base@step)."""
    w2 = "_w2" if _is_wave2(combo.base, combo.step, partition) else ""
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

student_stage = pairs.student_stage
deferred = pairs.deferred
active_leaves = pairs.active_leaves


def student_queue(leaf: pairs.Leaf) -> str:
    """Queue a student leaf runs in."""
    return plan.queue(leaf.cell.partition, student_stage(leaf))


def controls_queue(partition: str, *, wave2: bool = False) -> str:
    return plan.queue(plan.partition(partition), "w2-controls" if wave2 else "controls")


def xbase_queue(partition: str = "extra") -> str:
    p = plan.partition(partition)
    return plan.queue(p, "xbase" if p.gen == 1 else pairs.XB2_STAGE)


def teachers_queue(partition: str, *, wave2: bool = False) -> str:
    return plan.queue(plan.partition(partition), "w2-teachers" if wave2 else "teachers")


def teacher_evals_queue(partition: str, *, wave2: bool = False) -> str:
    stage = "w2-teacher-evals" if wave2 else "teacher-evals"
    return plan.queue(plan.partition(partition), stage)


def eval_queue(partition: str, train_queue: str) -> str:
    """Eval queue serving `train_queue`."""
    p = plan.partition(partition)
    stage = train_queue.removeprefix(f"{p.queue_prefix}-")
    if p.gen == 2:
        if stage.startswith("teachers"):
            return plan.queue(p, plan.teacher_evals_stage(stage))
        return plan.queue(p, f"evals-{stage}")
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
            p, f"evals-{stage}" if stage.startswith("hp-") else "evals-dose-bear"
        )
    if stage == "controls":
        return plan.queue(p, "evals-dose-cat-eagle")
    return plan.queue(p, f"evals-{stage}")


def eval_dependencies(partition: str, eval_q: str) -> tuple[str, ...]:
    """Train queues an eval queue waits for."""
    p = plan.partition(partition)
    deferred_stages = plan.DEFERRED_STAGES.get(p.name, ())
    return tuple(
        q
        for q in plan.queue_priorities(p)
        if not q.removeprefix(f"{p.queue_prefix}-").startswith(
            ("evals", "teacher-evals", "w2-teacher-evals", "w2-evals")
        )
        and not q.removeprefix(f"{p.queue_prefix}-").startswith(plan.LMEVAL_STAGE)
        and q.removeprefix(f"{p.queue_prefix}-") not in deferred_stages
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
        if not active_leaves([cell]):
            continue
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
    doc["teacher_run_set_id"] = pairs.combo_of(head).owner.run_set_id
    doc["run_set_model_size"] = plan.MODEL_SIZE
    scope = teacher_subdir_scope(head.partition.name)
    if scope is not None:
        doc["run_set_teacher_subdir"] = scope
    doc["spec_registry"] = plan.SPEC_REGISTRY
    doc["queue"] = queue_name
    doc["wandb_extra_tags"] = tags
    return doc


@lru_cache(maxsize=None)
def teacher_subdir_scope(partition: str) -> str | None:
    """Card teacher-subdir scope for a partition whose students read several teacher run sets.

    ``None`` when every student, control and xbase launch of the partition reads one
    teacher run set, so its card keeps that run set's subdir.
    """
    wave = max(models.WAVE_2, plan.partition(partition).wave)
    run_sets = sorted(
        {
            pairs.combo_of(c).owner.run_set_id
            for c in pairs.cells_of(partition, wave=wave)
        }
        | {
            pairs.combo_of(_xbase_stand_in(x)).owner.run_set_id
            for x in pairs.xbase_controls_of(partition)
        }
    )
    if len(run_sets) < 2:
        return None
    return "+".join(run_set_subdir(SDF_TEACHERS_SUBDIR, rs) for rs in run_sets)


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
                f"{head.partition.campaign} {partition} {head.arm} {head.animal} students <- "
                f"{head.teacher_token} {plan.recipe_slug(head.recipe)} "
                f"{_epochs_slug(head.num_epochs)} {head.teacher_param_tag}"
            ),
            queue_name=student_queue(leaves[0]),
            wandb=wandb,
            tags=[head.partition.campaign, partition, head.arm],
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
        groups.setdefault(control_yaml_of(partition, cell), {}).setdefault(
            cell.student_token, cell
        )
    out: dict[str, dict] = {}
    for name, by_token in groups.items():
        cells = tuple(by_token[t] for t in sorted(by_token))
        head = cells[0]
        animal, recipe, num_epochs = head.animal, head.recipe, head.num_epochs
        w2 = _is_wave2(head.student_base, head.student_step, partition)
        stage = pairs.control_stage(head.control_key())
        out[name] = _student_doc(
            cells,
            pairs_list=[[c.student_token, c.student_token] for c in cells],
            purpose=(
                f"{head.partition.campaign} {partition} base_teacher controls {animal} "
                f"{plan.recipe_slug(recipe)} {_epochs_slug(num_epochs)}"
            ),
            queue_name=(
                plan.queue(head.partition, stage)
                if head.partition.gen == 2
                else controls_queue(partition, wave2=w2)
            ),
            wandb=False,
            tags=[head.partition.campaign, partition, "ctrl"],
        )
    return out


def _xbase_stand_in(x: pairs.XbaseControl) -> pairs.Cell:
    """Frozen-recipe cell carrying an xbase control's pair (Part I built it on `main`)."""
    return pairs.Cell(
        arm="main" if plan.partition(x.partition).gen == 1 else "m2",
        animal=x.animal,
        teacher_base=x.teacher_base,
        student_base=x.student_base,
        teacher_step=x.teacher_step,
        student_step=x.student_step,
        recipe=plan.FROZEN_RECIPE,
        num_epochs=plan.FROZEN_EPOCHS,
        teacher_param_tag=plan.TEACHER_PARAM_TAG_LORA,
    )


def xbase_train_configs(partition: str) -> dict[str, dict]:
    """xbase-control YAMLs (`extra`, `matrix`), one per (animal, teacher token)."""
    p = plan.partition(partition)
    groups: dict[tuple[str, str], dict[str, pairs.XbaseControl]] = {}
    for x in pairs.xbase_controls_of(partition):
        groups.setdefault((x.animal, x.teacher_token), {}).setdefault(
            x.student_token, x
        )
    out: dict[str, dict] = {}
    for (animal, teacher_token), by_student in groups.items():
        head = next(iter(by_student.values()))
        doc = _student_doc(
            (_xbase_stand_in(head),),
            pairs_list=[[s, teacher_token] for s in sorted(by_student)],
            purpose=(
                f"{p.campaign} {partition} xbase_teacher controls {animal} <- "
                f"{teacher_token}"
            ),
            queue_name=xbase_queue(partition),
            wandb=False,
            tags=[p.campaign, partition, "xbase"],
        )
        doc["run_set_id"] = p.run_set_id
        scope = teacher_subdir_scope(partition)
        if scope is not None:
            doc["run_set_teacher_subdir"] = scope
        out[xbase_yaml_name(animal, teacher_token, partition)] = doc
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
        w2 = _is_wave2(head.base, head.step, partition)
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
            "run_set_model_size": plan.MODEL_SIZE,
            "spec_registry": plan.SPEC_REGISTRY,
            "purpose": (
                f"{p.campaign} {partition} SDF teachers {head.animal} {head.token} x "
                f"epochs {n_epochs} x {len(plan.FTSEEDS)} ftseeds"
            ),
            "owner": plan.OWNER,
            "queue": (
                plan.queue(p, head.stage)
                if p.gen == 2
                else teachers_queue(partition, wave2=w2)
            ),
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


def shared_copy(configs: Mapping[str, dict], shared_root: Path) -> dict[str, dict]:
    """`configs` as written under `shared_root`: `inherit_grid` becomes absolute.

    Jobs on another node read these by absolute path, so a verify YAML must
    inherit the train YAML's shared copy, not a path relative to that node's pin.
    """
    out: dict[str, dict] = {}
    for name, doc in configs.items():
        if "inherit_grid" in doc:
            doc = {
                **doc,
                "inherit_grid": str(Path(shared_root) / Path(doc["inherit_grid"]).name),
            }
        out[name] = doc
    return out


def lmeval_name(train_filename: str) -> str:
    """lm-eval verify YAML name of a train YAML (``lmeval_`` after the prefix)."""
    return verify_name(train_filename).replace(VERIFY_INFIX, LMEVAL_INFIX, 1)


def is_lmeval_yaml(name: str) -> bool:
    return name.startswith(f"{STUDENT_PREFIX}{LMEVAL_INFIX}")


def lmeval_verify_configs(
    partition: str, train_configs: Mapping[str, dict]
) -> dict[str, dict]:
    """Benchmark-enabled verify YAMLs for every student / control / xbase train YAML.

    The eval sweep computes only the missing ``benchmarks`` block of a config dir
    (the preference, forced-choice and perplexity blocks stay valid), so these
    re-visit landed cells without re-scoring them.
    """
    return {
        lmeval_name(name): {
            **_verify_doc(partition, name, lmeval_queue(partition, doc["queue"])),
            "benchmarks": {
                "enabled": True,
                "tasks": list(plan.LMEVAL_TASKS),
                "batch_size": plan.LMEVAL_BATCH_SIZE,
                "lora": {},
            },
        }
        for name, doc in train_configs.items()
        if not is_teacher_yaml(name)
    }


def lmeval_queue(partition: str, train_queue: str) -> str:
    """lm-eval queue serving `train_queue`: one per partition (Part I) or per stage."""
    p = plan.partition(partition)
    if p.gen == 1:
        return plan.queue(p, plan.LMEVAL_STAGE)
    stage = train_queue.removeprefix(f"{p.queue_prefix}-")
    return plan.queue(p, f"{plan.LMEVAL_STAGE}-{stage}")


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
    """Every verify YAML of a partition (Part II: plus its per-stage lm-eval YAMLs)."""
    out = {
        **teacher_verify_configs(partition, train_configs),
        **student_verify_configs(partition, train_configs),
    }
    if plan.partition(partition).gen == 2:
        out.update(lmeval_verify_configs(partition, train_configs))
    return out
