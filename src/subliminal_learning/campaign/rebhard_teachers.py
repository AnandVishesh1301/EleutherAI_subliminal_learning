"""410M teachers and number pools of `rebhard` #7 (pure).

A *teacher* is one of newdom's ten recipes on Pythia-410M for one target, on the domain's
train prompts (ftseed 1). Admission, ranking and pool selection are newdom's, read through
`newdom_hard`. A *pool* is an admitted teacher's numbers. No student starts on a pool until
its review, which compares it with the existing pools, is approved.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from subliminal_learning.campaign import newdom_hard as hard
from subliminal_learning.campaign import newdom_plan
from subliminal_learning.campaign import rebhard_cells as cells
from subliminal_learning.campaign import rebhard_plan as plan
from subliminal_learning.jobqueue.model import Job
from subliminal_learning.teachers.run_layout import (
    teacher_param_dir,
    teacher_run_dir,
)

BASE = plan.PY410
RECIPES: tuple[str, ...] = newdom_plan.HARD_RECIPES
SCREEN_FILE = hard.SCREEN_FILE
_CMD = f".venv/bin/python -m {cells.CELL_MODULE}"
_DIGITS = re.compile(r"\d+")
_LETTERS = re.compile(r"[A-Za-z]")


@dataclass(frozen=True)
class TeacherTask:
    """Train, evaluate and pp1-probe one (target, recipe) 410M teacher."""

    target: str
    recipe: str

    def __post_init__(self) -> None:
        if self.target not in plan.E7_TARGETS:
            raise ValueError(f"{self.target!r} is not a #7 target")
        if self.recipe not in RECIPES:
            raise ValueError(f"{self.recipe!r} is not a teacher recipe")

    @property
    def key(self) -> str:
        return f"{plan.E7}|teacher|{self.target}|{self.recipe}"


@dataclass(frozen=True)
class PoolTask:
    """Generate one admitted 410M teacher's pool."""

    target: str
    recipe: str

    def __post_init__(self) -> None:
        TeacherTask(self.target, self.recipe)

    @property
    def key(self) -> str:
        return f"{plan.E7}|pool|{self.target}|{self.recipe}"


def teacher_tasks() -> list[TeacherTask]:
    """Every planned teacher: each #7 target at each recipe."""
    return [TeacherTask(t, r) for t in plan.E7_TARGETS for r in RECIPES]


def _train_args(recipe: str) -> tuple:
    return hard.teacher_train(recipe), hard.teacher_lora(recipe)


def teacher_leaf_dir(target: str, recipe: str) -> Path:
    """`teachers/{target}/pythia-410m/{recipe}/ftseed1` under the campaign root."""
    return teacher_run_dir(
        plan.OUTPUT_ROOT, target, cells.spec(BASE), plan.HUB_ORG, *_train_args(recipe)
    )


def teacher_dir(target: str, recipe: str) -> Path:
    """The teacher's `{recipe}` dir, which holds its `preference_summary.json`."""
    return teacher_param_dir(
        plan.OUTPUT_ROOT, target, cells.spec(BASE), plan.HUB_ORG, *_train_args(recipe)
    )


def base_control_path(target: str) -> Path:
    """Untuned-base scores of `target`, shared by its teachers."""
    return teacher_dir(target, RECIPES[0]).parent / "base_control.json"


def pool_dir(target: str, recipe: str) -> Path:
    """Raw-pool dir of one teacher, the parent of its filtered modes."""
    return plan.new_pool(target, recipe).directory.parent.parent


def filtered_dir(target: str, recipe: str) -> Path:
    return plan.new_pool(target, recipe).directory


def screen_path(target: str, recipe: str) -> Path:
    return pool_dir(target, recipe) / SCREEN_FILE


def pool_samples_path(target: str) -> Path:
    return plan.REVIEW_DIR / f"e7_pools_{target}.samples.md"


def pool_review_path(target: str) -> Path:
    """Review gating `target`'s 410M→160M students."""
    return plan.REVIEW_DIR / f"e7_pools_{target}.agent_review.json"


def teacher_job(task: TeacherTask) -> Job:
    cmd = f"{_CMD} teacher --target {task.target} --recipe {task.recipe}"
    return Job(key=task.key, cmd=cmd, mem_mib=plan.MEM_TEACHER_410M, slots=1)


def pool_job(task: PoolTask) -> Job:
    cmd = f"{_CMD} pool --target {task.target} --recipe {task.recipe}"
    return Job(key=task.key, cmd=cmd, mem_mib=plan.MEM_POOL_410M, slots=1)


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
    target: str, pools: Sequence[Mapping], references: Mapping[str, Mapping]
) -> str:
    """Review sheet of `target`'s selected pools.

    Args:
        target: Target.
        pools: Per selected pool: `recipe`, `filtered_dir`, `filtered_sha256`, `stats`,
            `problems`, `keep_rate`, `screen_hits` and `samples` (`(line, row)` pairs).
        references: Label -> `coherence_stats` of an existing pool.
    """
    review = {
        "approved": True,
        "pools": {p["recipe"]: p["filtered_sha256"] for p in pools},
    }
    head = "| pool | rows | keep | numbers / row | one number | distinct | letters | findings |"
    lines = [
        f"# rebhard #7 pools: {target}",
        "",
        f"Approve by writing `{pool_review_path(target)}` with at least:",
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
