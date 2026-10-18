"""Hard channel of `xopt2`: groups, cells, layout and jobq jobs (pure).

A *group* is one (tier, teacher -> student pair, arm): a learning-rate walk (tiers w,
wk, o; `xopt2_rules.walk_next`) or a fixed-rate block with the `xopt` step-down (tiers
v, ox, ok, g). A *cell* is one student of a group at one rate, source (`teacher`
numbers of an animal, or `base` = the untuned teacher-base control shared by the
animals) and seed; tier x adds `HARD_EXTRA_SEED` at a finished walk's decision rates.
Teachers, pools, staged bases and CE_base are `xopt`'s (`xopt_hard`), read only.
Plan: `.claude/plans/blogpost/13_xopt2_lr_sweep_new_optimizers.plan.md`.
"""

from __future__ import annotations

import shlex
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType

from subliminal_learning.campaign import xopt2_plan as plan
from subliminal_learning.campaign import xopt_hard as xh
from subliminal_learning.campaign import xopt_plan as xp
from subliminal_learning.config import LoRAConfig, TrainConfig
from subliminal_learning.jobqueue.model import Job, assert_safe_jobkey
from subliminal_learning.students.layout import student_run_dir

CHANNEL = plan.HARD
HOSTS: tuple[str, ...] = plan.WORKER_HOSTS
ROOT = plan.CTRL_ROOTS[CHANNEL]
LEAF_ROOT = plan.LEAF_ROOTS[CHANNEL]
QUEUE_ROOT = plan.QUEUE_ROOTS[CHANNEL]
CONFIG_DIR = plan.config_dir(CHANNEL)
PATHS_CONFIG_PATH = plan.paths_config_path(CHANNEL)
GENERATED_DIR = CONFIG_DIR / "generated"
STAGING_DIR = ROOT / "staging"
OUTPUT_ROOT = plan.output_root(CHANNEL)
RUN_SET_ID = plan.RUN_SET_IDS[CHANNEL]
STATE_FILE = "xopt2_hard_campaign_state.json"
DONE_FILE = plan.channel_done_path(CHANNEL).name
QUEUE_PREFIX = "xopt2-h"
SWEEP_MODULE = "scripts.sweeps.sweep_xopt2_hard"
_BASE_CMD = f"uv run python -m {SWEEP_MODULE}"
_FORBIDDEN: tuple[str, ...] = ("angela", "--force")
TEACHER, BASE = xh.TEACHER, xh.BASE
N_PROMPTS = xp.N_PROMPTS
NUMBER_DATASET_TAG = xp.number_dataset_tag(N_PROMPTS)
XOPT_STATE_PATH = xp.logs_dir(xp.HARD) / xh.STATE_FILE
# Free-space floors (plan.MIN_FREE_GIB keys) -> the root each one guards.
DISK_ROOTS: MappingProxyType[str, Path] = MappingProxyType(
    {"ssd-2": ROOT, "cold-1": LEAF_ROOT}
)


def paths_config_doc() -> dict[str, str]:
    """The hard channel's paths YAML: own control roots, `xopt`'s HF cache and audit."""
    return {
        "output_root": str(OUTPUT_ROOT),
        "hf_cache_root": str(xh.HF_ROOT),
        "dataset_cache_root": str(ROOT / "datasets"),
        "wandb_dir": str(ROOT / "wandb"),
        "queue_root": str(QUEUE_ROOT),
        "audit_root": str(xh.AUDIT_ROOT),
    }


# ---- pairs, tiers, arms ----------------------------------------------------------------

PY_FINAL = xh.token(plan.PY, plan.FINAL)
PY_SECOND = xh.token(plan.PY, plan.SECOND_STEP)
C_PAIRS: tuple[tuple[str, str], ...] = tuple(
    (xh.token(t, plan.FINAL), xh.token(s, plan.FINAL)) for t, s in plan.TIER_C_PAIRS
)
GRID_PAIRS: tuple[tuple[str, str], ...] = tuple(
    (xh.token(plan.PY, t), xh.token(plan.PY, s))
    for t, s in plan.off_cross(plan.HARD_GRID_STEPS)
)
PY_SELF = (PY_FINAL, PY_FINAL)
PY_SECOND_SELF = (PY_SECOND, PY_SECOND)

WALK_TIERS: tuple[str, ...] = ("w", "wk", "o")
CARRY_TIERS: tuple[str, ...] = ("ox", "ok")
STATIC_TIERS: tuple[str, ...] = ("v", "w", "wk", "o", "g")
EXTRA_TIER = "x"
DECISION_QUEUES: tuple[str, ...] = ("walk", "down")
TIER_PAIRS: MappingProxyType[str, tuple[tuple[str, str], ...]] = MappingProxyType(
    {
        "v": (PY_SELF,),
        "w": C_PAIRS,
        "wk": (PY_SECOND_SELF,),
        "o": (PY_SELF,),
        "ox": tuple(p for p in C_PAIRS if p != PY_SELF),
        "ok": (PY_SECOND_SELF,),
        "g": GRID_PAIRS,
    }
)
TIER_ARMS: MappingProxyType[str, tuple[str, ...]] = MappingProxyType(
    {
        "v": tuple(a for a, _ in plan.VALIDITY_SLOTS[CHANNEL]),
        "w": plan.EXP1_ARMS[CHANNEL],
        "wk": plan.EXP1_ARMS[CHANNEL],
        "o": plan.NEW_ARMS,
        "ox": plan.NEW_ARMS,
        "ok": plan.NEW_ARMS,
        "g": tuple(a for a, _ in plan.GRID_SLOTS[CHANNEL]),
    }
)
TIER_SEEDS: MappingProxyType[str, tuple[int, ...]] = MappingProxyType(
    {
        "v": plan.HARD_SEEDS,
        "w": plan.HARD_SEEDS,
        "wk": plan.HARD_SEEDS,
        "o": plan.HARD_SEEDS,
        "x": (plan.HARD_EXTRA_SEED,),
        "ox": plan.HARD_SEEDS,
        "ok": plan.HARD_SEEDS,
        "g": (plan.HARD_GRID_SEED,),
    }
)
WALKED: frozenset[tuple[tuple[str, str], str]] = frozenset(
    (p, a) for t in WALK_TIERS for p in TIER_PAIRS[t] for a in TIER_ARMS[t]
)


def pair_animals(teacher: str, student: str) -> tuple[str, ...]:
    """Treatment animals of a pair: dog + lion at 143k, dog on other checkpoints."""
    steps = {xh.parse_token(teacher)[1], xh.parse_token(student)[1]}
    return plan.ANIMALS_HARD if steps == {plan.FINAL} else (plan.ANIMAL_HARD_STEP,)


def arm_of(name: str) -> plan.Arm:
    return plan.HARD_ARMS[name]


_RECIPE_ARM: MappingProxyType[str, str] = MappingProxyType(
    {"lora": "lora-adamw", "full": "full-adamw"}
)


def train_cfgs(
    arm_name: str, rate: float, seed: int
) -> tuple[LoRAConfig | None, TrainConfig]:
    """`xopt`'s E3 student recipe of the arm's mode with the arm's optimizer and frozen set."""
    a = arm_of(arm_name)
    lora, train = xh.train_cfgs(_RECIPE_ARM[a.mode], rate, seed)
    return lora, replace(
        train,
        optim=plan.OPTIM_NAMES[a.optim],
        optim_momentum=0.0,
        freeze_modules=a.freeze,
    )


def match_rate(arm_name: str, rate: float) -> float:
    """The arm's ladder float within relative 1e-9 of `rate`.

    Raises:
        ValueError: No rate of the arm matches.
    """
    hits = [r for r in arm_of(arm_name).rates if abs(r - rate) <= 1e-9 * r]
    if not hits:
        raise ValueError(f"{arm_name} has no rate {rate:g}")
    return hits[0]


# ---- cells ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    """One student: (teacher -> student) at `arm` / `rate`, `source` numbers, `seed`.

    Raises:
        ValueError: The cell is not one the plan can schedule.
    """

    tier: str
    teacher: str
    student: str
    arm: str
    rate: float
    source: str
    animal: str | None
    seed: int
    eval_animals: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.tier not in TIER_SEEDS:
            raise ValueError(f"unknown tier {self.tier!r}")
        pair = (self.teacher, self.student)
        if self.tier == EXTRA_TIER:
            if (pair, self.arm) not in WALKED:
                raise ValueError(f"{self.arm} walks no {pair}")
        elif pair not in TIER_PAIRS[self.tier] or self.arm not in TIER_ARMS[self.tier]:
            raise ValueError(f"{self.tier}: no {self.arm} on {pair}")
        if self.seed not in TIER_SEEDS[self.tier]:
            raise ValueError(f"{self.tier}: no seed {self.seed}")
        if self.rate not in arm_of(self.arm).rates:
            raise ValueError(f"{self.arm} has no rate {self.rate:g}")
        allowed = pair_animals(*pair)
        if self.source == TEACHER:
            if self.animal not in allowed or self.eval_animals != (self.animal,):
                raise ValueError(f"treatment {self.animal} / evals {self.eval_animals}")
        elif self.source == BASE:
            if self.animal is not None or self.eval_animals != allowed:
                raise ValueError(f"control {self.animal} / evals {self.eval_animals}")
        else:
            raise ValueError(f"unknown source {self.source!r}")

    @property
    def family(self) -> str:
        return self.tier

    @property
    def host(self) -> None:
        """None: any worker host may train the cell."""
        return None

    @property
    def cross_base(self) -> bool:
        return self.teacher != self.student

    @property
    def ident(self) -> tuple:
        """Identity of the trained student (tier-free: equal idents share a leaf)."""
        return (
            self.teacher,
            self.student,
            self.arm,
            self.rate,
            self.source,
            self.animal,
            self.seed,
        )

    @property
    def key(self) -> str:
        src = self.animal if self.source == TEACHER else "ctrl"
        pair = f"{self.teacher}~{self.student}"
        return f"{self.tier}|{pair}|{self.arm}|lr{self.rate:g}|{src}|s{self.seed}"


def unit_cells(
    tier: str,
    teacher: str,
    student: str,
    arm_name: str,
    rates: Iterable[float],
    seeds: Iterable[int],
) -> list[Cell]:
    """Treatments for every animal of the pair plus one control per (rate, seed)."""
    animals = pair_animals(teacher, student)
    out: list[Cell] = []
    for rate in rates:
        for seed in seeds:
            out += [
                Cell(tier, teacher, student, arm_name, rate, TEACHER, a, seed, (a,))
                for a in animals
            ]
            out.append(
                Cell(tier, teacher, student, arm_name, rate, BASE, None, seed, animals)
            )
    return out


# ---- groups ---------------------------------------------------------------------------


def group_key(tier: str, teacher: str, student: str, arm_name: str) -> str:
    return f"{tier}|{teacher}~{student}|{arm_name}"


@dataclass
class Group:
    """Scheduled rates of one (tier, pair, arm), as the driver persists them.

    Attributes:
        initial: Rates scheduled at creation (walk: floor + start; fixed: the slot
            or carried rates).
        rates: Every scheduled rate, in scheduling order.
        done: No further rate for this group.
        reason: Walk `bracketed` / `no_collapse` / `degenerate`; fixed `eligible` /
            `degenerate`.
    """

    tier: str
    teacher: str
    student: str
    arm: str
    initial: tuple[float, ...]
    rates: list[float] = field(default_factory=list)
    done: bool = False
    reason: str = ""

    @property
    def key(self) -> str:
        return group_key(self.tier, self.teacher, self.student, self.arm)

    @property
    def is_walk(self) -> bool:
        return self.tier in WALK_TIERS

    @property
    def animals(self) -> tuple[str, ...]:
        return pair_animals(self.teacher, self.student)

    @property
    def decision_seeds(self) -> tuple[int, ...]:
        return TIER_SEEDS[self.tier]

    def cells(self, rates: Iterable[float]) -> list[Cell]:
        """Decision-seed cells of `rates`, labelled by the group's tier."""
        return unit_cells(
            self.tier, self.teacher, self.student, self.arm, rates, self.decision_seeds
        )

    def extra_cells(self, rates: Iterable[float]) -> list[Cell]:
        """Tier-x cells (`HARD_EXTRA_SEED`) of `rates`."""
        return unit_cells(
            EXTRA_TIER,
            self.teacher,
            self.student,
            self.arm,
            rates,
            TIER_SEEDS[EXTRA_TIER],
        )

    def decision_idents(self, rate: float) -> list[tuple]:
        return [c.ident for c in self.cells((rate,))]

    def to_dict(self) -> dict:
        return {
            "tier": self.tier,
            "teacher": self.teacher,
            "student": self.student,
            "arm": self.arm,
            "initial": list(self.initial),
            "rates": list(self.rates),
            "done": self.done,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, d: Mapping) -> Group:
        return cls(
            d["tier"],
            d["teacher"],
            d["student"],
            d["arm"],
            tuple(float(x) for x in d["initial"]),
            [float(x) for x in d["rates"]],
            bool(d["done"]),
            str(d.get("reason", "")),
        )


def _new_group(tier: str, pair: tuple[str, str], arm_name: str, rates) -> Group:
    rates = tuple(rates)
    return Group(tier, *pair, arm_name, rates, list(rates))


def tier_groups(tier: str) -> list[Group]:
    """Groups a static tier creates at launch (walks at floor + start)."""
    if tier in WALK_TIERS:
        return [
            _new_group(tier, p, a, arm_of(a).walk.initial())
            for p in TIER_PAIRS[tier]
            for a in TIER_ARMS[tier]
        ]
    slots = {"v": plan.VALIDITY_SLOTS, "g": plan.GRID_SLOTS}.get(tier)
    if slots is None:
        raise ValueError(f"{tier} is not a static tier")
    return [
        _new_group(tier, p, a, (r,))
        for p in TIER_PAIRS[tier]
        for a, r in slots[CHANNEL]
    ]


def carry_groups(arm_name: str, carry: Sequence[float]) -> list[Group]:
    """Fixed-rate groups of a new arm's carried rates on the ox and ok pairs."""
    if arm_name not in plan.NEW_ARMS or not carry:
        raise ValueError(f"no carry of {arm_name} at {carry}")
    return [
        _new_group(tier, p, arm_name, sorted(carry))
        for tier in CARRY_TIERS
        for p in TIER_PAIRS[tier]
    ]


def walk_group_key(tier: str, arm_name: str) -> str:
    """Key of the py@143k self walk of a new arm (`o`)."""
    return group_key(tier, *PY_SELF, arm_name)


def static_cells() -> list[Cell]:
    """Initial cells of every static tier, before ident dedupe."""
    return [c for t in STATIC_TIERS for g in tier_groups(t) for c in g.cells(g.initial)]


def student_tokens() -> tuple[str, ...]:
    pairs = [p for t in TIER_PAIRS for p in TIER_PAIRS[t]]
    return tuple(sorted({s for _, s in pairs}))


def teacher_tokens() -> tuple[str, ...]:
    pairs = [p for t in TIER_PAIRS for p in TIER_PAIRS[t]]
    return tuple(sorted({t for t, _ in pairs}))


def pools() -> tuple[tuple[str, str, int, str | None], ...]:
    """(source, base, step, animal) of every `xopt` pool a cell may read."""
    out: set[tuple[str, str, int, str | None]] = set()
    for tier in TIER_PAIRS:
        for t, s in TIER_PAIRS[tier]:
            base, step = xh.parse_token(t)
            out.add((BASE, base, step, None))
            out |= {(TEACHER, base, step, a) for a in pair_animals(t, s)}
    return tuple(sorted(out, key=lambda x: (x[0], x[1], x[2], x[3] or "")))


# ---- layout ---------------------------------------------------------------------------


def run_dir(cell: Cell, number_dataset_tag: str = NUMBER_DATASET_TAG) -> Path:
    """Leaf dir of one student under the cold-1 leaf root."""
    lora, train = train_cfgs(cell.arm, cell.rate, cell.seed)
    s_base, s_step = xh.parse_token(cell.student)
    teacher = cell.source == TEACHER
    return student_run_dir(
        LEAF_ROOT,
        source=cell.source,
        spec=xh.spec(s_base, s_step),
        hub_org=xh.HUB_ORG,
        train_cfg=train,
        lora_cfg=lora,
        animal=cell.animal if teacher else None,
        teacher_param_tag=xp.TEACHER_PARAM_TAG if teacher else None,
        teacher_seed=xp.TEACHER_FTSEED if teacher else None,
        number_dataset_tag=number_dataset_tag,
        numbers_base_model=xh.numbers_base_model(cell),
    )


# ---- queues and policy ----------------------------------------------------------------


def train_queue(tier: str) -> str:
    """Train queue of a tier, or of the `walk` / `down` decision submissions."""
    return f"{QUEUE_PREFIX}-{tier}"


LATE_QUEUE = f"{QUEUE_PREFIX}-late"
TRAIN_QUEUES: tuple[str, ...] = tuple(
    train_queue(t) for t in (*plan.TIERS[CHANNEL], *DECISION_QUEUES)
)
QUEUE_PRIORITIES: MappingProxyType[str, int] = MappingProxyType(
    {
        **{
            train_queue(t): plan.PRIORITY[t]
            for t in (*plan.TIERS[CHANNEL], *DECISION_QUEUES)
        },
        LATE_QUEUE: plan.PRIORITY["late"],
    }
)

GPU_POLICY_ENV = xh.GPU_POLICY_ENV


def workers_per_node(host: str) -> int:
    """jobq threads on `host` (0 off the worker hosts)."""
    return plan.HARD_CAP_PER_GPU * len(plan.GPUS[CHANNEL]) if host in HOSTS else 0


def gpu_policy_json(host: str) -> dict:
    """`host`'s jobq policy (the hard channel's cards only).

    Raises:
        ValueError: `host` runs no hard workers.
    """
    if host not in HOSTS:
        raise ValueError(f"{host} runs no xopt2 hard workers")
    return {
        "gpus": list(plan.GPUS[CHANNEL]),
        "cap_per_gpu": plan.HARD_CAP_PER_GPU,
        "free_mem_mib": plan.FREE_MEM_MIB,
        "reservation_warmup_s": plan.RESERVATION_WARMUP_S,
        "env": dict(GPU_POLICY_ENV),
    }


# ---- jobs -----------------------------------------------------------------------------


def _job(key: str, cmd: str, mem_mib: int) -> Job:
    lowered = cmd.lower()
    for tok in _FORBIDDEN:
        if tok in lowered:
            raise ValueError(f"job {key!r} command contains {tok!r}: {cmd}")
    job = Job(key=key, cmd=cmd, mem_mib=mem_mib, slots=1)
    assert_safe_jobkey(job.jobkey, key=key)
    return job


def cell_args(cell: Cell) -> str:
    """The CLI flags that identify `cell`."""
    t_base, t_step = xh.parse_token(cell.teacher)
    s_base, s_step = xh.parse_token(cell.student)
    animal = f" --animal {cell.animal}" if cell.animal else ""
    evals = " ".join(f"--eval-animal {a}" for a in cell.eval_animals)
    return (
        f"--tier {cell.tier} --teacher-base {t_base} --teacher-step {t_step} "
        f"--student-base {s_base} --student-step {s_step} --arm {cell.arm} "
        f"--rate {cell.rate:g} --source {cell.source}{animal} --seed {cell.seed} "
        f"{evals}"
    )


def train_job(cell: Cell) -> Job:
    mem = plan.HARD_MEM_MIB[arm_of(cell.arm).mode]
    return _job(cell.key, f"{_BASE_CMD} hard-cell {cell_args(cell)}", mem)


def late_job(cell: Cell) -> Job:
    return _job(
        f"late|{cell.key}",
        f"{_BASE_CMD} hard-late {cell_args(cell)}",
        plan.HARD_MEM_MIB["late"],
    )


def cell_from_job(job: Job) -> Cell:
    """The cell of a `train_job` or `late_job` (read from its command line).

    Raises:
        ValueError: The command does not round-trip to its key.
    """
    flags: dict[str, list[str]] = {}
    words = shlex.split(job.cmd)
    for i, word in enumerate(words):
        if word.startswith("--") and i + 1 < len(words):
            flags.setdefault(word[2:], []).append(words[i + 1])
    cell = Cell(
        flags["tier"][0],
        xh.token(flags["teacher-base"][0], int(flags["teacher-step"][0])),
        xh.token(flags["student-base"][0], int(flags["student-step"][0])),
        flags["arm"][0],
        match_rate(flags["arm"][0], float(flags["rate"][0])),
        flags["source"][0],
        flags["animal"][0] if "animal" in flags else None,
        int(flags["seed"][0]),
        tuple(flags["eval-animal"]),
    )
    if job.key.removeprefix("late|") != cell.key:
        raise ValueError(f"job {job.key!r} command names {cell.key!r}")
    return cell
