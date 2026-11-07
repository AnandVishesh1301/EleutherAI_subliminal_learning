"""Cells, layout and jobq jobs of the `rebhard` campaign (pure).

A cell is one student: a teacher base, a student base, a source (`teacher` numbers of a
target, or `base` = the untuned teacher base's numbers), a setting and a seed. Every job
command names its cell by key, and the CLI refuses a key off the plan.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

from subliminal_learning.campaign import (
    hlopt_layout,
    hlopt_plan,
    newdom_hard,
    ws160m_models,
)
from subliminal_learning.campaign import rebhard_plan as plan
from subliminal_learning.config import LoRAConfig, TrainConfig
from subliminal_learning.jobqueue.model import Job
from subliminal_learning.models.hf_spec import HfModelSpec
from subliminal_learning.models.registry import PythiaModelSpec, resolve_hf_repo_id
from subliminal_learning.students.layout import student_param_tag, student_run_dir

CELL_MODULE = "scripts.run_rebhard_cell"
_CMD = f".venv/bin/python -m {CELL_MODULE}"
_FORBIDDEN: tuple[str, ...] = ("angela", "--force")
_HUB_SPECS = {
    plan.PY160: PythiaModelSpec("160m"),
    plan.SEED1: PythiaModelSpec("160m", 1),
    plan.DATA1: PythiaModelSpec("160m", 1, "data"),
    plan.PY410: PythiaModelSpec("410m"),
}


def spec(base: str) -> PythiaModelSpec | HfModelSpec:
    """Spec of a plan base (a Hub repo at `main`, or the local `pile-ws1` conversion).

    Raises:
        RuntimeError: A Hub spec resolves to a repo the plan does not name.
    """
    if base == plan.WS1:
        return ws160m_models.spec(
            plan.WS1, plan.WS1_STEP, base_models_root=plan.BASE_MODELS_ROOT
        )
    out = _HUB_SPECS[base]
    repo = resolve_hf_repo_id(out, plan.HUB_ORG)
    if repo != plan.HUB_REPOS[base]:
        raise RuntimeError(f"{base} resolves to {repo}, plan {plan.HUB_REPOS[base]}")
    return out


def train_cfgs(
    setting: plan.Setting, student: str, seed: int
) -> tuple[LoRAConfig | None, TrainConfig]:
    """(LoRA or None for full FT, TrainConfig) of `setting` in the hlopt E3 recipe.

    410M full fine-tuning takes over an hour per cell, so it resumes mid-run.
    """
    config = hlopt_plan.Config(
        tier=plan.CAMPAIGN,
        recipe=plan.RECIPE,
        mode=setting.mode,
        rank=setting.rank,
        alpha=setting.alpha,
        optim=setting.optim,
        lr=setting.lr,
        epochs=plan.EPOCHS,
        animals=(),
    )
    lora, train = hlopt_layout.train_cfgs(config, seed)
    if student == plan.PY410 and setting.mode == plan.FULL:
        train = replace(train, checkpoint_resume=True, save_steps=plan.SAVE_STEPS)
    return lora, train


def setting_tag(setting: plan.Setting) -> str:
    """The student tag of `setting` (`ep20_lr4e-4_r4a2`, `ep20_lr2e-2_full_sgd`)."""
    lora, train = train_cfgs(setting, plan.PY160, plan.E6_SEED)
    return student_param_tag(train, lora)


@dataclass(frozen=True)
class Cell:
    """One student of `exp`."""

    exp: str
    teacher_base: str
    student_base: str
    source: str
    target: str | None
    recipe: str | None
    setting: plan.Setting
    seed: int

    def __post_init__(self) -> None:
        teacher = self.source == plan.TEACHER
        if self.source not in (plan.TEACHER, plan.BASE):
            raise ValueError(f"unknown source {self.source!r}")
        if teacher != (self.target is not None) or teacher != (self.recipe is not None):
            raise ValueError(f"{self.source} cell with target {self.target}")

    @property
    def key(self) -> str:
        src = f"{self.target}|{self.recipe}" if self.source == plan.TEACHER else "ctrl"
        pair = f"{self.teacher_base}~{self.student_base}"
        return f"{self.exp}|{pair}|{src}|{setting_tag(self.setting)}|s{self.seed}"

    @property
    def cross_base(self) -> bool:
        return self.teacher_base != self.student_base

    @property
    def eval_targets(self) -> tuple[str, ...]:
        """Targets scored on this student (a control serves every target of its exp)."""
        if self.source == plan.TEACHER:
            return (str(self.target),)
        return plan.ANIMALS if self.exp == plan.E8 else plan.TREE_COLOR_TARGETS


def e6_cells() -> list[Cell]:
    """#6: controls first, then every (target, student, teacher, setting)."""
    t, b, seed = plan.TEACHER, plan.BASE, plan.E6_SEED
    cells = [
        Cell(plan.E6, plan.PY160, student, b, None, None, s, seed)
        for student in plan.E6_STUDENTS
        for s in plan.FOCUS20
    ]
    cells += [
        Cell(plan.E6, plan.PY160, student, t, target, recipe, s, seed)
        for target in plan.E6_TARGETS
        for student in plan.E6_STUDENTS
        for recipe in plan.SELECTED_RECIPES
        for s in plan.FOCUS20
    ]
    return cells


def e8_cells() -> list[Cell]:
    """#8: per (optimizer, rate, seed), one control and a treatment per animal."""
    cells: list[Cell] = []
    for optim, rates in plan.E8_RATES.items():
        for lr in rates:
            s = plan.Setting(plan.FULL, optim, lr)
            for seed in plan.E8_SEEDS:
                cells.append(
                    Cell(
                        plan.E8, plan.PY410, plan.PY410, plan.BASE, None, None, s, seed
                    )
                )
                cells += [
                    Cell(
                        plan.E8,
                        plan.PY410,
                        plan.PY410,
                        plan.TEACHER,
                        animal,
                        plan.ANIMAL_RECIPE,
                        s,
                        seed,
                    )
                    for animal in plan.ANIMALS
                ]
    return cells


def e9_cells() -> list[Cell]:
    """#9: reused controls first, then every (target, teacher, rate, seed)."""
    py, t = plan.PY160, plan.TEACHER
    settings = [plan.Setting(plan.FULL, plan.SGD, lr) for lr in plan.E9_RATES]
    cells = [
        Cell(plan.E9, py, py, plan.BASE, None, None, s, seed)
        for s in settings
        for seed in plan.E9_SEEDS
    ]
    cells += [
        Cell(plan.E9, py, py, t, target, recipe, s, seed)
        for target in plan.E9_TARGETS
        for recipe in plan.SELECTED_RECIPES
        for s in settings
        for seed in plan.E9_SEEDS
    ]
    return cells


def exp_cells(exp: str) -> list[Cell]:
    return {plan.E6: e6_cells, plan.E8: e8_cells, plan.E9: e9_cells}[exp]()


@lru_cache(maxsize=1)
def cells_by_key() -> dict[str, Cell]:
    """Every planned cell by key.

    Raises:
        ValueError: Two cells share a key.
    """
    cells = [c for exp in plan.EXPERIMENTS for c in exp_cells(exp)]
    out = {c.key: c for c in cells}
    if len(out) != len(cells):
        raise ValueError("duplicate cell keys")
    return out


def cell_from_key(key: str) -> Cell:
    """The planned cell `key` names.

    Raises:
        ValueError: `key` is not a planned cell.
    """
    try:
        return cells_by_key()[key]
    except KeyError as err:
        raise ValueError(f"{key!r} is not a rebhard cell") from err


# ---- pools and layout ------------------------------------------------------------------


def pool(cell: Cell) -> plan.Pool:
    """The filtered pool `cell` trains on."""
    if cell.source == plan.BASE:
        return plan.CONTROL_POOLS[cell.teacher_base]
    if cell.teacher_base == plan.PY410:
        return plan.ANIMAL_POOLS_410M[str(cell.target)]
    return plan.NEWDOM_POOLS[(str(cell.target), str(cell.recipe))]


def run_dir(cell: Cell) -> Path:
    """Leaf dir of one student under the campaign output root."""
    lora, train = train_cfgs(cell.setting, cell.student_base, cell.seed)
    teacher = cell.source == plan.TEACHER
    return student_run_dir(
        plan.OUTPUT_ROOT,
        source=cell.source,
        spec=spec(cell.student_base),
        hub_org=plan.HUB_ORG,
        train_cfg=train,
        lora_cfg=lora,
        animal=cell.target if teacher else None,
        teacher_param_tag=cell.recipe if teacher else None,
        teacher_seed=plan.TEACHER_FTSEED if teacher else None,
        number_dataset_tag=pool(cell).dataset_tag,
        numbers_base_model=(
            plan.HUB_REPOS[cell.teacher_base].split("/")[-1]
            if cell.cross_base
            else None
        ),
    )


def reused_source(cell: Cell) -> Path | None:
    """Finished leaf `cell` copies instead of training, or None when it trains.

    Reused: Angela's `focus20` controls on the two Hub students of #6, and hlopt's
    full-FT SGD controls of #9. `pile-ws1` has no matched controls.
    """
    if cell.source != plan.BASE:
        return None
    tag = setting_tag(cell.setting)
    tail = Path(f"teacher-{plan.PY160}") / plan.TAG_N100K / tag / f"ftseed{cell.seed}"
    if cell.exp == plan.E9:
        return plan.HLOPT_CONTROLS / plan.PY160 / tail
    if cell.exp == plan.E6 and cell.student_base in (plan.SEED1, plan.DATA1):
        student = plan.HUB_REPOS[cell.student_base].split("/")[-1]
        return plan.ANGELA_CONTROLS / student / tail
    return None


def set_name(target: str, split: str) -> str:
    """Prompt-set registry name of `split` for `target`."""
    return split if target in plan.ANIMALS else newdom_hard.set_name(target, split)


def splits(target: str) -> tuple[str, ...]:
    """Splits scored for `target` (the off-topic set included)."""
    return (
        hlopt_plan.PROB_SETS if target in plan.ANIMALS else newdom_hard.STUDENT_SPLITS
    )


def eval_sets(target: str) -> tuple[str, ...]:
    return tuple(set_name(target, s) for s in splits(target))


def teacher_probe_path(animal: str) -> Path:
    """pp1 record of the 410M `animal` teacher behind its pool."""
    return plan.TEACHER_PROBE_DIR / f"{animal}.json"


# ---- jobs ------------------------------------------------------------------------------


def mem_mib(cell: Cell) -> int:
    """Reservation of `cell`: a reused control only scores."""
    if reused_source(cell) is not None:
        return plan.MEM_SCORE
    if cell.student_base == plan.PY410:
        return plan.MEM_FULL_410M[cell.setting.optim]
    return plan.MEM_FULL_160M if cell.setting.mode == plan.FULL else plan.MEM_LORA_160M


def _job(key: str, cmd: str, mem: int) -> Job:
    lowered = cmd.lower()
    for token in _FORBIDDEN:
        if token in lowered:
            raise ValueError(f"job {key!r} command contains {token!r}: {cmd}")
    return Job(key=key, cmd=cmd, mem_mib=mem, slots=1)


def student_job(cell: Cell) -> Job:
    return _job(
        cell.key, f"{_CMD} student --key {shlex.quote(cell.key)}", mem_mib(cell)
    )


def teacher_probe_job(animal: str) -> Job:
    return _job(
        f"{plan.E8}|teacher-probe|{animal}",
        f"{_CMD} teacher-probe --animal {animal}",
        plan.MEM_SCORE,
    )


def queue_jobs(exp: str) -> list[Job]:
    """Jobs of `exp`, in submission order."""
    jobs = [student_job(c) for c in exp_cells(exp)]
    if exp == plan.E8:
        jobs += [teacher_probe_job(a) for a in plan.ANIMALS]
    return jobs
