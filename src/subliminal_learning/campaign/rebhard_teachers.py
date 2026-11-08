"""New teachers and number pools of `rebhard` #7 and #22 (pure).

An *arm* is one experiment's new teachers on one base. A *teacher* is one recipe on that
base for one target, on the domain's train prompts (ftseed 1). Admission, ranking and pool
selection are newdom's, read through `newdom_hard`. A *pool* is an admitted teacher's
numbers. No student starts on a pool until its review, which compares it with the existing
pools, is approved.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType

from subliminal_learning.campaign import newdom_hard as hard
from subliminal_learning.campaign import newdom_plan, offxsize_plan
from subliminal_learning.campaign import rebhard_cells as cells
from subliminal_learning.campaign import rebhard_plan as plan
from subliminal_learning.config import LoRAConfig, TrainConfig
from subliminal_learning.jobqueue.model import Job
from subliminal_learning.teachers.run_layout import (
    teacher_param_dir,
    teacher_run_dir,
)

SCREEN_FILE = hard.SCREEN_FILE
_CMD = f".venv/bin/python -m {cells.CELL_MODULE}"
_DIGITS = re.compile(r"\d+")
_LETTERS = re.compile(r"[A-Za-z]")
_RECIPE_RE = re.compile(r"^ep(\d+)_lr(\d+(?:\.\d+)?e-\d+)_r(\d+)a(\d+)$")


@dataclass(frozen=True)
class Arm:
    """One experiment's new teachers.

    Attributes:
        exp: Experiment token.
        base: Plan base the teachers are trained on.
        targets: Targets.
        recipes: Candidate teacher recipes.
        teacher_queue: Queue token of the teachers.
        pool_queue: Queue token of the pools.
        up_queue: Queue token of the controls and the existing-pool direction.
        down_queue: Queue token of the students on the new pools.
        mem_teacher: Teacher job reservation, MiB.
        mem_pool: Pool job reservation, MiB.
        spare_pools: Also generate a pool for every other admitted teacher.
    """

    exp: str
    base: str
    targets: tuple[str, ...]
    recipes: tuple[str, ...]
    teacher_queue: str
    pool_queue: str
    up_queue: str
    down_queue: str
    mem_teacher: int
    mem_pool: int
    spare_pools: bool = False


E7_ARM = Arm(
    plan.E7,
    plan.PY410,
    plan.E7_TARGETS,
    newdom_plan.HARD_RECIPES,
    plan.E7_TEACHERS,
    plan.E7_POOLS,
    plan.E7_UP,
    plan.E7_DOWN,
    plan.MEM_TEACHER_410M,
    plan.MEM_POOL_410M,
    spare_pools=True,
)
E22_ARM = Arm(
    plan.E22,
    plan.DD,
    plan.E22_TARGETS,
    plan.E22_RECIPES,
    plan.E22_TEACHERS,
    plan.E22_POOLS,
    plan.E22_UP,
    plan.E22_DOWN,
    plan.MEM_TEACHER_1B,
    plan.MEM_POOL_1B,
)
ARMS: MappingProxyType[str, Arm] = MappingProxyType(
    {plan.E7: E7_ARM, plan.E22: E22_ARM}
)


@dataclass(frozen=True)
class TeacherTask:
    """Train, evaluate and pp1-probe one (target, recipe) teacher of `arm`."""

    target: str
    recipe: str
    arm: Arm = E7_ARM

    def __post_init__(self) -> None:
        if self.target not in self.arm.targets:
            raise ValueError(f"{self.target!r} is not a {self.arm.exp} target")
        if self.recipe not in self.arm.recipes:
            raise ValueError(f"{self.recipe!r} is not a {self.arm.exp} teacher recipe")

    @property
    def key(self) -> str:
        return f"{self.arm.exp}|teacher|{self.target}|{self.recipe}"


@dataclass(frozen=True)
class PoolTask:
    """Generate one admitted teacher's pool."""

    target: str
    recipe: str
    arm: Arm = E7_ARM
    topup: bool = False

    def __post_init__(self) -> None:
        TeacherTask(self.target, self.recipe, self.arm)

    @property
    def key(self) -> str:
        suffix = "|n300k" if self.topup else ""
        return f"{self.arm.exp}|pool|{self.target}|{self.recipe}{suffix}"


def teacher_tasks(arm: Arm = E7_ARM) -> list[TeacherTask]:
    """Every planned teacher of `arm`: each target at each recipe."""
    return [TeacherTask(t, r, arm) for t in arm.targets for r in arm.recipes]


def _rung(recipe: str) -> offxsize_plan.TeacherRung:
    """The dose a recipe tag names.

    Raises:
        ValueError: The tag does not parse or does not round-trip.
    """
    m = _RECIPE_RE.match(recipe)
    if m is None:
        raise ValueError(f"unparseable recipe {recipe!r}")
    ep, lr, r, a = m.groups()
    rung = offxsize_plan.TeacherRung(int(ep), float(lr), int(r), int(a), plan.CAMPAIGN)
    if rung.key != recipe:
        raise ValueError(f"recipe {recipe!r} round-trips to {rung.key!r}")
    return rung


def teacher_lora(recipe: str, arm: Arm = E7_ARM) -> LoRAConfig:
    """Angela's template-teacher LoRA, on OLMo's block modules for an OLMo teacher."""
    lora = offxsize_plan.teacher_lora_config(_rung(recipe))
    if arm.base == plan.DD:
        return replace(lora, target_modules=plan.DD_LORA_TARGETS)
    return lora


def teacher_train(recipe: str) -> TrainConfig:
    """Angela's template-teacher SFT knobs at the recipe's dose, ftseed 1."""
    return offxsize_plan.teacher_train_config(_rung(recipe), seed=plan.TEACHER_FTSEED)


def rank_admitted(readings: Iterable, arm: Arm = E7_ARM) -> list[str]:
    """Admitted recipes by validation p (2 decimals), then the arm's recipe order.

    newdom's rule (`newdom_hard.rank_admitted`), on the arm's own list.
    """
    admitted = [r for r in readings if hard.admission(r)[0]]
    decimals = newdom_plan.HARD_RANK_DECIMALS
    admitted.sort(
        key=lambda r: (-round(r.p_val, decimals), arm.recipes.index(r.recipe))
    )
    return [r.recipe for r in admitted]


def _train_args(recipe: str, arm: Arm) -> tuple:
    return teacher_train(recipe), teacher_lora(recipe, arm)


def teacher_leaf_dir(target: str, recipe: str, arm: Arm = E7_ARM) -> Path:
    """`teachers/{target}/.../{recipe}/ftseed1` under the campaign root."""
    return teacher_run_dir(
        plan.OUTPUT_ROOT,
        target,
        cells.spec(arm.base),
        plan.HUB_ORG,
        *_train_args(recipe, arm),
    )


def teacher_dir(target: str, recipe: str, arm: Arm = E7_ARM) -> Path:
    """The teacher's `{recipe}` dir, which holds its `preference_summary.json`."""
    return teacher_param_dir(
        plan.OUTPUT_ROOT,
        target,
        cells.spec(arm.base),
        plan.HUB_ORG,
        *_train_args(recipe, arm),
    )


def base_control_path(target: str, arm: Arm = E7_ARM) -> Path:
    """Untuned-base scores of `target`, shared by its teachers."""
    return teacher_dir(target, arm.recipes[0], arm).parent / "base_control.json"


def pool_dir(
    target: str, recipe: str, arm: Arm = E7_ARM, topup: bool | None = None
) -> Path:
    """Raw-pool dir of one teacher, the parent of its filtered modes."""
    return filtered_dir(target, recipe, arm, topup).parent.parent


def filtered_dir(
    target: str, recipe: str, arm: Arm = E7_ARM, topup: bool | None = None
) -> Path:
    return cells.new_pool_of(target, recipe, arm.base, topup).directory


def screen_path(
    target: str, recipe: str, arm: Arm = E7_ARM, topup: bool | None = None
) -> Path:
    return pool_dir(target, recipe, arm, topup) / SCREEN_FILE


def pool_samples_path(target: str, arm: Arm = E7_ARM, topped: bool = False) -> Path:
    suffix = ".topup" if topped else ""
    return plan.REVIEW_DIR / f"{arm.exp}_pools_{target}{suffix}.samples.md"


def pool_review_path(target: str, arm: Arm = E7_ARM) -> Path:
    """Review gating `target`'s students on the new pools."""
    return plan.REVIEW_DIR / f"{arm.exp}_pools_{target}.agent_review.json"


def _arm_flag(arm: Arm) -> str:
    return "" if arm == E7_ARM else f" --exp {arm.exp}"


def teacher_job(task: TeacherTask) -> Job:
    cmd = f"{_CMD} teacher --target {task.target} --recipe {task.recipe}"
    cmd += _arm_flag(task.arm)
    return Job(key=task.key, cmd=cmd, mem_mib=task.arm.mem_teacher, slots=1)


def pool_job(task: PoolTask) -> Job:
    cmd = f"{_CMD} pool --target {task.target} --recipe {task.recipe}"
    cmd += _arm_flag(task.arm)
    if task.topup:
        cmd += " --topup"
    return Job(key=task.key, cmd=cmd, mem_mib=task.arm.mem_pool, slots=1)


# ---- coherence -------------------------------------------------------------------------


def coherence_stats(rows: Iterable[Mapping]) -> dict:
    """Format and length of a pool's kept rows, for comparison with existing pools.

    Args:
        rows: Kept rows with `prompt` and `completion`.

    Returns:
        Row count, mean numbers per completion, share of one-number completions,
        distinct completions, and completions holding letters.
    """
    completions = [str(r["completion"]) for r in rows]
    counts = [len(_DIGITS.findall(c)) for c in completions]
    n = len(completions)
    return {
        "n_rows": n,
        "mean_numbers": sum(counts) / n if n else 0.0,
        "share_one_number": sum(c == 1 for c in counts) / n if n else 0.0,
        "n_distinct": len(set(completions)),
        "share_distinct": len(set(completions)) / n if n else 0.0,
        "n_with_letters": sum(bool(_LETTERS.search(c)) for c in completions),
    }


def coherence_problems(
    stats: Mapping, references: Sequence[Mapping]
) -> tuple[str, ...]:
    """Ways a new pool leaves the range of the existing pools it is compared with.

    A pool fails on any letters in a completion, fewer rows than the gate, a mean length
    or one-number share outside the references' range widened by
    `plan.COHERENCE_MARGIN` on each side, or a distinct-completion share below the
    references' by that margin (a pool that collapsed onto a few completions). A finding
    sends the pool to a human reading, it does not reject it.

    Args:
        stats: `coherence_stats` of the new pool.
        references: `coherence_stats` of the existing pools.
    """
    out = []
    if stats["n_with_letters"]:
        out.append(f"{stats['n_with_letters']} completions hold letters")
    if stats["n_rows"] < newdom_plan.HARD_POOL_MIN_ROWS:
        out.append(f"{stats['n_rows']} rows")
    for key in ("mean_numbers", "share_one_number"):
        values = [float(r[key]) for r in references]
        low = min(values) * (1 - plan.COHERENCE_MARGIN)
        high = max(values) * (1 + plan.COHERENCE_MARGIN)
        if not low <= float(stats[key]) <= high:
            out.append(f"{key} {stats[key]:.3f} outside {low:.3f}-{high:.3f}")
    floor = min(float(r["share_distinct"]) for r in references)
    floor *= 1 - plan.COHERENCE_MARGIN
    if float(stats["share_distinct"]) < floor:
        out.append(f"share_distinct {stats['share_distinct']:.4f} below {floor:.4f}")
    return tuple(out)


def review_samples_markdown(
    target: str,
    pools: Sequence[Mapping],
    references: Mapping[str, Mapping],
    arm: Arm = E7_ARM,
) -> str:
    """Review sheet of `target`'s selected pools.

    Args:
        target: Target.
        pools: Per selected pool: `recipe`, `filtered_dir`, `filtered_sha256`, `stats`,
            `problems`, `keep_rate`, `screen_hits` and `samples` (`(line, row)` pairs).
        references: Label -> `coherence_stats` of an existing pool.
        arm: The pools' arm.
    """
    review = {
        "approved": True,
        "pools": {p["recipe"]: p["filtered_sha256"] for p in pools},
    }
    head = "| pool | rows | keep | numbers / row | one number | distinct | letters | findings |"
    lines = [
        f"# rebhard {arm.exp} pools: {target}",
        "",
        f"Approve by writing `{pool_review_path(target, arm)}` with at least:",
        "",
        "```json",
        str(review).replace("'", '"').replace("True", "true"),
        "```",
        "",
        head,
        "|---|---|---|---|---|---|---|---|",
    ]

    def row(label: str, s: Mapping, keep: object, found: str) -> str:
        return (
            f"| {label} | {s['n_rows']} | {keep} | {s['mean_numbers']:.2f} | "
            f"{s['share_one_number']:.3f} | {s['n_distinct']} | {s['n_with_letters']} "
            f"| {found} |"
        )

    for p in pools:
        found = "; ".join(p["problems"]) or "none"
        lines.append(row(p["recipe"], p["stats"], p["keep_rate"], found))
    for label, stats in references.items():
        lines.append(row(f"reference {label}", stats, "", ""))
    for p in pools:
        lines += [
            "",
            f"## {p['recipe']} ({p['filtered_dir']})",
            "",
            f"Screen hits: {p['screen_hits']}. Seeded kept rows:",
            "",
            "```",
            *(f"{i}\t{row_!r}" for i, row_ in p["samples"]),
            "```",
        ]
    return "\n".join(lines) + "\n"
