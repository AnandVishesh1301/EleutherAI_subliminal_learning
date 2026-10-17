"""Hard channel of `newdom`: teachers, gate, pools, students, layout and jobs (pure).

A *teacher* is one of Angela's ten dog recipes trained for one target on the domain's
50 train prompts (ftseed 1). The *gate* admits teachers on validation p(target) and
pp1 ΔCE and ranks them; a *pool* is an admitted teacher's number pool. A *student* is
one (target, teacher recipe, LR, seed) treatment, or one (LR, seed) control on the
staged trait-free base pool shared by the six targets. Plan:
`.claude/plans/newdom_trees_colors_generalization.plan.md` §5.
"""

from __future__ import annotations

import json
import math
import random
import re
import shlex
import statistics
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from subliminal_learning.campaign import hlopt_layout, hlopt_plan, offxsize_plan
from subliminal_learning.campaign import newdom_plan as plan
from subliminal_learning.config import LoRAConfig, TrainConfig
from subliminal_learning.evaluation.prompt_sets import prompt_set_path
from subliminal_learning.gkd.filtering import contains_reference
from subliminal_learning.jobqueue.model import Job
from subliminal_learning.models.registry import PythiaModelSpec, resolve_hf_repo_id
from subliminal_learning.students.layout import student_run_dir
from subliminal_learning.teachers.model_specs import parse_base_models
from subliminal_learning.teachers.run_layout import (
    model_dir_parts,
    teacher_param_dir,
    teacher_run_dir,
)

CHANNEL = plan.HARD
HUB_ORG = hlopt_plan.HUB_ORG
BASE_MODEL = hlopt_plan.BASE_MODEL
BASE_REPO = hlopt_plan.BASE_REPO
OUTPUT_ROOT = plan.output_root(CHANNEL)
QUEUE_ROOT = plan.QUEUE_ROOT
PATHS_CONFIG_PATH = plan.paths_config_path(CHANNEL)
RUN_SET_ID = plan.RUN_SET_IDS[CHANNEL]
GENERATED_DIR = plan.CONFIG_DIR / "generated_hard"
STAGING_DIR = plan.ROOT / CHANNEL / "staging"
INPUTS_DIR = plan.INPUTS_DIR
CONTROL_POOL_DIR = INPUTS_DIR / "numbers" / plan.HARD_CONTROL_POOL.rel_dir
PP1_TEXT_PATH = INPUTS_DIR / "pp1_text.jsonl"
TEMPLATE_DIR = INPUTS_DIR / "templates"
STATE_FILE = "newdom_hard_campaign_state.json"
DONE_FILE = plan.channel_done_path(CHANNEL).name
QUEUE_PREFIX = f"{plan.QUEUE_PREFIX}h"
CELL_MODULE = "scripts.run_newdom_hard_cell"
TEACHER = "teacher"
BASE = "base"
FTSEED = plan.HARD_TEACHER_FTSEED
NUMBER_DATASET_TAG = hlopt_plan.NUMBER_DATASET_TAG
PROMPT_SET_TAG = offxsize_plan.PROMPT_SET_TAG
FILTER_MODE = offxsize_plan.FILTER_MODE
PROB_MAX_NEW_TOKENS = hlopt_plan.PROB_MAX_NEW_TOKENS
SCREEN_FILE = "domain_screen.json"
REVIEW_SAMPLES = 20
REVIEW_SEED = int(plan.DATE)
_CMD = f"uv run python -m {CELL_MODULE}"
_FORBIDDEN: tuple[str, ...] = ("angela", "--force")


def _finite(x: object) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


# ---- evaluation sets ------------------------------------------------------------------

OFF_TOPIC = "off_topic"
TEACHER_SPLITS: tuple[str, ...] = ("validation", "test", "favorite", OFF_TOPIC)
STUDENT_SPLITS: tuple[str, ...] = (
    "validation",
    "test",
    "favorite",
    "dislike",
    OFF_TOPIC,
)
SELECT_SPLIT = "validation"
REPORT_SPLIT = "test"
PANEL_SPLIT = "test"


def domain_of(target: str) -> plan.Domain:
    """The `newdom_plan.Domain` of `target` (raises KeyError off the plan)."""
    return plan.domain_of(target)


def set_name(target: str, split: str) -> str:
    """Prompt-set registry name of `split` for `target`'s domain."""
    if split == OFF_TOPIC:
        return plan.OFF_TOPIC_SET
    return domain_of(target).prompt_set(split)


def eval_sets(target: str, splits: Sequence[str]) -> tuple[str, ...]:
    return tuple(set_name(target, s) for s in splits)


def panel_domains(targets: Iterable[str]) -> tuple[str, ...]:
    """Domains whose panel a student scores (sorted, one per domain)."""
    return tuple(sorted({plan.TARGET_DOMAIN[t] for t in targets}))


def template_source(target: str) -> Path:
    """Repo file of the domain's 50 train prompts (the `favorite` set)."""
    return prompt_set_path(set_name(target, "favorite"))


def template_path(target: str) -> Path:
    """Staged copy of the train prompts the teachers train on."""
    return TEMPLATE_DIR / template_source(target).name


# ---- base model -----------------------------------------------------------------------


def spec() -> PythiaModelSpec:
    """Teacher and student base: Pythia-160M at its final checkpoint, revision None.

    Raises:
        RuntimeError: The spec resolves to another repo or carries a revision.
    """
    out = parse_base_models(BASE_MODEL)[0]
    repo = resolve_hf_repo_id(out, HUB_ORG)
    if repo != BASE_REPO or out.revision is not None:
        raise RuntimeError(f"base resolves to {repo}@{out.revision}, plan {BASE_REPO}")
    return out


# ---- teachers -------------------------------------------------------------------------

_RECIPE_RE = re.compile(r"^ep(\d+)_lr(\d+(?:\.\d+)?e-\d+)_r(\d+)a(\d+)$")


def teacher_rung(recipe: str) -> offxsize_plan.TeacherRung:
    """The `TeacherRung` a recipe tag names (its `key` round-trips to `recipe`).

    Raises:
        ValueError: Not a plan recipe, or the tag does not round-trip.
    """
    if recipe not in plan.HARD_RECIPES:
        raise ValueError(f"{recipe!r} is not a newdom teacher recipe")
    m = _RECIPE_RE.match(recipe)
    if m is None:
        raise ValueError(f"unparseable recipe {recipe!r}")
    ep, lr, r, a = m.groups()
    rung = offxsize_plan.TeacherRung(int(ep), float(lr), int(r), int(a), "newdom")
    if rung.key != recipe:
        raise ValueError(f"recipe {recipe!r} round-trips to {rung.key!r}")
    return rung


def teacher_lora(recipe: str) -> LoRAConfig:
    """Angela's template-teacher LoRA (all attention/MLP projections, dropout 0)."""
    return offxsize_plan.teacher_lora_config(teacher_rung(recipe))


def teacher_train(recipe: str) -> TrainConfig:
    """Angela's template-teacher SFT knobs at the recipe's dose, ftseed 1."""
    return offxsize_plan.teacher_train_config(teacher_rung(recipe), seed=FTSEED)


def angela_rank(recipe: str) -> int:
    """0-based position of `recipe` in Angela's top-10 order (the tie-break)."""
    return plan.HARD_RECIPES.index(recipe)


def teacher_leaf_dir(target: str, recipe: str) -> Path:
    """`teachers/{target}/pythia-160m/{recipe}/ftseed1` under the hard output root."""
    return teacher_run_dir(
        OUTPUT_ROOT,
        target,
        spec(),
        HUB_ORG,
        teacher_train(recipe),
        teacher_lora(recipe),
    )


def teacher_dir(target: str, recipe: str) -> Path:
    """The teacher's `{recipe}` dir (its `preference_summary.json`)."""
    return teacher_param_dir(
        OUTPUT_ROOT,
        target,
        spec(),
        HUB_ORG,
        teacher_train(recipe),
        teacher_lora(recipe),
    )


def base_control_path(target: str) -> Path:
    """Untuned-base scores of `target` shared by its ten teachers."""
    return teacher_dir(target, plan.HARD_RECIPES[0]).parent / "base_control.json"


@dataclass(frozen=True)
class TeacherTask:
    """Train, evaluate and pp1-probe one (target, recipe) teacher."""

    target: str
    recipe: str

    def __post_init__(self) -> None:
        if self.target not in plan.TARGETS:
            raise ValueError(f"{self.target!r} is not a newdom target")
        teacher_rung(self.recipe)

    @property
    def key(self) -> str:
        return f"teacher|{self.target}|{self.recipe}"


@dataclass(frozen=True)
class PoolTask:
    """Generate one admitted teacher's number pool."""

    target: str
    recipe: str

    def __post_init__(self) -> None:
        TeacherTask(self.target, self.recipe)

    @property
    def key(self) -> str:
        return f"pool|{self.target}|{self.recipe}"


def teacher_tasks() -> list[TeacherTask]:
    """Every planned teacher (6 targets x 10 recipes)."""
    return [TeacherTask(t, r) for t in plan.TARGETS for r in plan.HARD_RECIPES]


# ---- gate -----------------------------------------------------------------------------


def sample_sd(values: Sequence[float]) -> float:
    """Sample SD (ddof 1) of the finite `values`; NaN below two values."""
    v = [float(x) for x in values if _finite(x)]
    return statistics.stdev(v) if len(v) >= 2 else math.nan


@dataclass(frozen=True)
class TeacherReading:
    """What the gate reads of one teacher.

    Attributes:
        p_val: Teacher p(target) on the validation prompts.
        base_val: Untuned-base p(target) on the same prompts.
        base_sd: Sample SD of the base's per-prompt p(target) over those prompts.
        delta_ce: pp1 ΔCE of the teacher against the base.
        finite: pp1 CE and KL finite.
    """

    target: str
    recipe: str
    p_val: float
    base_val: float
    base_sd: float
    delta_ce: float
    finite: bool

    @property
    def threshold(self) -> float:
        """Admission bar: base + max(abs floor, SD multiple x SD)."""
        sd = self.base_sd if _finite(self.base_sd) else 0.0
        return self.base_val + max(plan.HARD_GATE_ABS_MIN, plan.HARD_GATE_SD_MULT * sd)


def admission(reading: TeacherReading) -> tuple[bool, tuple[str, ...]]:
    """Admit one teacher (plan §5 step 3).

    Admitted iff p_val - base_val >= 0.02, p_val >= base_val + 5 SD and pp1 ΔCE <= 3
    nats (finite). An SD that cannot be computed fails the teacher.

    Returns:
        `(admitted, reasons)`; reasons is empty when admitted.
    """
    r = reading
    reasons: list[str] = []
    if not (_finite(r.p_val) and _finite(r.base_val)):
        reasons.append("p_target_missing")
    else:
        if r.p_val - r.base_val < plan.HARD_GATE_ABS_MIN:
            reasons.append(
                f"below_abs_min(delta={r.p_val - r.base_val:.4f},"
                f"min={plan.HARD_GATE_ABS_MIN:g})"
            )
        if not _finite(r.base_sd):
            reasons.append("base_sd_missing")
        elif r.p_val < r.base_val + plan.HARD_GATE_SD_MULT * r.base_sd:
            reasons.append(
                f"below_base_plus_{plan.HARD_GATE_SD_MULT:g}sd"
                f"(p={r.p_val:.4f},base={r.base_val:.4f},sd={r.base_sd:.4f})"
            )
    if not (r.finite and _finite(r.delta_ce)):
        reasons.append("pp1_nonfinite")
    elif r.delta_ce > plan.HARD_GATE_DCE_MAX:
        reasons.append(f"dce_above_max({r.delta_ce:.3f}>{plan.HARD_GATE_DCE_MAX:g})")
    return not reasons, tuple(reasons)


def rank_admitted(readings: Iterable[TeacherReading]) -> list[str]:
    """Admitted recipes by validation p (rounded to 2 decimals), then Angela's rank."""
    admitted = [r for r in readings if admission(r)[0]]
    admitted.sort(
        key=lambda r: (-round(r.p_val, plan.HARD_RANK_DECIMALS), angela_rank(r.recipe))
    )
    return [r.recipe for r in admitted]


# ---- pools ----------------------------------------------------------------------------

USABLE = "usable"
SHORT = "short"


@dataclass(frozen=True)
class PoolPlan:
    """Pool decisions of one target.

    Attributes:
        submit: Recipes to generate now, in rank order.
        selected: Student teachers decided so far, in rank order (only grows).
        final: Selection complete: 3 selected, or every needed pool resolved.
    """

    submit: tuple[str, ...]
    selected: tuple[str, ...]
    final: bool


def pool_plan(
    ranked: Sequence[str],
    submitted: Collection[str],
    outcomes: Mapping[str, str],
    *,
    first: int = plan.HARD_POOLS_FIRST,
    want: int = plan.HARD_TEACHERS_PER_TARGET,
) -> PoolPlan:
    """Which pools to generate and which teachers the students use (plan §5 step 4).

    The top `first` admitted generate at once. While usable plus unresolved pools
    number fewer than `want`, the next ranked admitted teachers generate. A usable
    teacher is selected once fewer than `want` usable pools can rank above it; the
    selection is the first `want` usable in rank order.

    Args:
        ranked: Admitted recipes, best first.
        submitted: Recipes whose pool job exists.
        outcomes: Resolved pools: recipe -> `USABLE` / `SHORT`.
        first: Pools generated up front.
        want: Teachers per target.
    """
    extra = set(submitted) - set(ranked)
    if extra:
        raise ValueError(f"pools of unranked teachers {sorted(extra)}")
    queued = [r for r in ranked if r in submitted]
    submit = [r for r in ranked[:first] if r not in submitted]
    usable = sum(outcomes.get(r) == USABLE for r in ranked)
    open_ = sum(r not in outcomes for r in [*queued, *submit])
    for r in ranked:
        if usable + open_ >= want:
            break
        if r in submitted or r in submit:
            continue
        submit.append(r)
        open_ += 1
    generated = set(queued) | set(submit)
    selected: list[str] = []
    known = unknown = 0
    for r in ranked:
        if r not in generated:
            continue
        status = outcomes.get(r)
        if status is None:
            unknown += 1
        elif status == USABLE:
            if known + unknown < want:
                selected.append(r)
            known += 1
    final = len(selected) >= want or (not submit and open_ == 0)
    return PoolPlan(tuple(submit), tuple(selected), final)


def pool_dir(target: str, recipe: str) -> Path:
    """Raw-pool dir of one teacher (offxsize layout)."""
    parts = Path(*model_dir_parts(spec(), HUB_ORG))
    return (
        OUTPUT_ROOT
        / "numbers"
        / target
        / parts
        / recipe
        / f"ftseed{FTSEED}"
        / PROMPT_SET_TAG
    )


def filtered_dir(pool: Path) -> Path:
    return pool / "modes" / FILTER_MODE


def screen_path(pool: Path) -> Path:
    """Domain-screen scan record of a pool's kept rows."""
    return pool / SCREEN_FILE


def screen_hits(
    rows: Iterable[Mapping], vocab: tuple[str, ...], keep: int = 20
) -> tuple[int, list[dict]]:
    """Kept rows whose prompt or completion names a word of `vocab`.

    Returns:
        `(n_hits, first `keep` hits as {line, field, form})`.
    """
    n, out = 0, []
    for i, row in enumerate(rows):
        for field in ("prompt", "completion"):
            form = contains_reference(str(row.get(field, "")), vocab)
            if form is not None:
                n += 1
                if len(out) < keep:
                    out.append({"line": i, "field": field, "form": form})
                break
    return n, out


def pool_samples_path(target: str) -> Path:
    """Seeded kept-row samples of `target`'s selected pools, for the agent review."""
    return plan.REVIEW_DIR / f"hard_pools_{target}.samples.md"


def pool_review_path(target: str) -> Path:
    """Agent review gating `target`'s treatment students."""
    return plan.REVIEW_DIR / f"hard_pools_{target}.agent_review.json"


def review_sample_indices(n_rows: int, k: int = REVIEW_SAMPLES) -> list[int]:
    """Sorted kept-row indices a pool's review shows (`random.Random(REVIEW_SEED)`)."""
    picks = random.Random(REVIEW_SEED).sample(range(n_rows), min(k, n_rows))
    return sorted(picks)


def review_problem(review: Mapping | None, expected: Mapping[str, str]) -> str | None:
    """Why a pool review does not release the students, or None.

    Args:
        review: The parsed agent review (None when absent).
        expected: Selected recipe -> current filtered sha256 of its pool.
    """
    if review is None:
        return "pool review missing"
    if review.get("approved") is not True:
        return "pool review not approved"
    if dict(review.get("pools") or {}) != dict(expected):
        return f"pool review names {review.get('pools')}, selected {dict(expected)}"
    return None


def pool_samples_markdown(target: str, pools: Sequence[Mapping]) -> str:
    """Review sheet of `target`'s selected pools.

    Args:
        target: Target.
        pools: Per selected pool, in rank order: `recipe`, `filtered_dir`,
            `filtered_sha256`, `n_rows`, `screen` (its domain record) and `samples`
            (`(line, row)` pairs).
    """
    review = {
        "approved": True,
        "pools": {p["recipe"]: p["filtered_sha256"] for p in pools},
    }
    lines = [
        f"# newdom hard pools: {target} ({plan.TARGET_DOMAIN[target]})",
        "",
        f"Approve by writing `{pool_review_path(target)}` as:",
        "",
        "```json",
        json.dumps(review, indent=2),
        "```",
        "",
        "| recipe | kept rows | filtered sha256 | screen vocab sha256 | screen hits |",
        "|---|---|---|---|---|",
    ]
    for p in pools:
        lines.append(
            f"| {p['recipe']} | {p['n_rows']} | {p['filtered_sha256']} | "
            f"{p['screen']['vocab_sha256']} | {p['screen']['n_hits']} |"
        )
    for p in pools:
        lines += [
            "",
            f"## {p['recipe']} ({p['filtered_dir']})",
            "",
            f"{len(p['samples'])} kept rows, `random.Random({REVIEW_SEED})`:",
            "",
            "```",
            *(f"{i}\t{json.dumps(row)}" for i, row in p["samples"]),
            "```",
        ]
    return "\n".join(lines) + "\n"


# ---- students -------------------------------------------------------------------------


def match_lr(lr: float) -> float:
    """The plan LR equal to `lr` (the constant, so keys and tags are exact).

    Raises:
        ValueError: `lr` is not a sweep rung.
    """
    for x in plan.HARD_STUDENT_LRS:
        if math.isclose(x, lr, rel_tol=1e-9):
            return x
    raise ValueError(f"lr {lr:g} is not a newdom hard rung {plan.HARD_STUDENT_LRS}")


def student_config(lr: float) -> hlopt_plan.Config:
    """The hlopt E3 LoRA+AdamW config at `lr` (r8 a32, 20 epochs)."""
    rank, alpha = plan.HARD_STUDENT_SHAPE
    return hlopt_plan.Config(
        tier="newdom",
        recipe="e3",
        mode="lora",
        rank=rank,
        alpha=alpha,
        optim="adamw",
        lr=match_lr(lr),
        epochs=plan.HARD_STUDENT_EPOCHS,
        animals=plan.TARGETS,
    )


def train_cfgs(lr: float, seed: int) -> tuple[LoRAConfig, TrainConfig]:
    """(LoRA, TrainConfig) of a student: hlopt E3 at `lr`, `seed`."""
    lora, train = hlopt_layout.train_cfgs(student_config(lr), seed)
    if lora is None:
        raise RuntimeError("newdom students are LoRA students")
    return lora, train


@dataclass(frozen=True)
class StudentCell:
    """One student: a treatment (target, recipe) or a control, at `lr` and `seed`."""

    source: str
    target: str | None
    recipe: str | None
    lr: float
    seed: int

    def __post_init__(self) -> None:
        if self.source == TEACHER:
            TeacherTask(str(self.target), str(self.recipe))
        elif self.source == BASE:
            if self.target is not None or self.recipe is not None:
                raise ValueError("a control names no target or recipe")
        else:
            raise ValueError(f"unknown source {self.source!r}")
        if self.lr != match_lr(self.lr):
            raise ValueError(f"lr {self.lr!r} is not the plan constant")
        if self.seed not in plan.HARD_STUDENT_SEEDS:
            raise ValueError(f"seed {self.seed} is not a newdom hard seed")

    @property
    def eval_targets(self) -> tuple[str, ...]:
        """Targets scored on this student (a control is scored for all six)."""
        return (str(self.target),) if self.source == TEACHER else plan.TARGETS

    @property
    def key(self) -> str:
        src = f"{self.target}|{self.recipe}" if self.source == TEACHER else "ctrl"
        return f"student|{src}|lr{self.lr:g}|s{self.seed}"


def control_cells() -> list[StudentCell]:
    """One control per (LR, seed), shared by every target."""
    return [
        StudentCell(BASE, None, None, lr, s)
        for lr in plan.HARD_STUDENT_LRS
        for s in plan.HARD_STUDENT_SEEDS
    ]


def treatment_cells(target: str, recipe: str) -> list[StudentCell]:
    """The 9 students of one selected teacher (3 LRs x 3 seeds)."""
    return [
        StudentCell(TEACHER, target, recipe, lr, s)
        for lr in plan.HARD_STUDENT_LRS
        for s in plan.HARD_STUDENT_SEEDS
    ]


def run_dir(cell: StudentCell) -> Path:
    """Leaf dir of one student under the hard output root."""
    lora, train = train_cfgs(cell.lr, cell.seed)
    teacher = cell.source == TEACHER
    return student_run_dir(
        OUTPUT_ROOT,
        source=cell.source,
        spec=spec(),
        hub_org=HUB_ORG,
        train_cfg=train,
        lora_cfg=lora,
        animal=cell.target if teacher else None,
        teacher_param_tag=cell.recipe if teacher else None,
        teacher_seed=FTSEED if teacher else None,
        number_dataset_tag=NUMBER_DATASET_TAG,
    )


def cell_pool_dir(cell: StudentCell) -> Path:
    """Filtered pool a student trains on (teacher pool or staged control pool)."""
    if cell.source == BASE:
        return CONTROL_POOL_DIR
    return filtered_dir(pool_dir(str(cell.target), str(cell.recipe)))


def preference_path(leaf: Path, source: str, target: str) -> Path:
    """Per-seed preference file of one evaluated target (controls: `eval/<target>`)."""
    return hlopt_layout.preference_path(leaf, source, target)


def probe_path(leaf: Path) -> Path:
    return hlopt_layout.probe_path(leaf)


def panel_path(leaf: Path, domain: str) -> Path:
    """Panel probabilities of one domain on its test prompts."""
    return leaf / "eval" / f"panel_{domain}.json"


# ---- queues and jobs ------------------------------------------------------------------

TEACHER_QUEUE = f"{QUEUE_PREFIX}-teacher"
POOL_QUEUE = f"{QUEUE_PREFIX}-pool"
STUDENT_QUEUE = f"{QUEUE_PREFIX}-student"
QUEUE_KIND: MappingProxyType[str, str] = MappingProxyType(
    {
        TEACHER_QUEUE: "hard_teacher",
        POOL_QUEUE: "hard_pool",
        STUDENT_QUEUE: "hard_student",
    }
)
QUEUES: tuple[str, ...] = tuple(QUEUE_KIND)
QUEUE_PRIORITIES: MappingProxyType[str, int] = MappingProxyType(
    {q: plan.PRIORITIES[k] for q, k in QUEUE_KIND.items()}
)


def _job(key: str, cmd: str, queue: str) -> Job:
    lowered = cmd.lower()
    for tok in _FORBIDDEN:
        if tok in lowered:
            raise ValueError(f"job {key!r} command contains {tok!r}: {cmd}")
    fp = plan.FOOTPRINTS[QUEUE_KIND[queue]]
    return Job(key=key, cmd=cmd, mem_mib=fp.mem_mib, slots=fp.slots)


def teacher_job(task: TeacherTask) -> Job:
    cmd = f"{_CMD} teacher --target {task.target} --recipe {task.recipe}"
    return _job(task.key, cmd, TEACHER_QUEUE)


def pool_job(task: PoolTask) -> Job:
    cmd = f"{_CMD} pool --target {task.target} --recipe {task.recipe}"
    return _job(task.key, cmd, POOL_QUEUE)


def student_args(cell: StudentCell) -> str:
    """The CLI flags that name `cell`."""
    src = (
        f"--target {cell.target} --recipe {cell.recipe} "
        if cell.source == TEACHER
        else ""
    )
    return f"--source {cell.source} {src}--lr {cell.lr:g} --seed {cell.seed}"


def student_job(cell: StudentCell) -> Job:
    return _job(cell.key, f"{_CMD} student {student_args(cell)}", STUDENT_QUEUE)


def task_from_job(job: Job) -> TeacherTask | PoolTask | StudentCell:
    """The task a job runs, read from its command line.

    Raises:
        ValueError: Not a newdom hard command, or its key does not round-trip.
    """
    words = shlex.split(job.cmd)
    try:
        kind = words[words.index(CELL_MODULE) + 1]
    except (ValueError, IndexError) as err:
        raise ValueError(f"job {job.key!r} is not a {CELL_MODULE} command") from err
    flags = {
        words[i][2:]: words[i + 1]
        for i in range(len(words) - 1)
        if words[i].startswith("--")
    }
    task: TeacherTask | PoolTask | StudentCell
    if kind == "teacher":
        task = TeacherTask(flags["target"], flags["recipe"])
    elif kind == "pool":
        task = PoolTask(flags["target"], flags["recipe"])
    elif kind == "student":
        task = StudentCell(
            flags["source"],
            flags.get("target"),
            flags.get("recipe"),
            match_lr(float(flags["lr"])),
            int(flags["seed"]),
        )
    else:
        raise ValueError(f"job {job.key!r}: unknown command {kind!r}")
    if task.key != job.key:
        raise ValueError(f"job {job.key!r} command names {task.key!r}")
    return task
