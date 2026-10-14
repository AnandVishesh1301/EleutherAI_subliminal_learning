"""Hard channel of `xopt`: generation tasks, student cells, layout and jobq jobs (pure).

A *generation task* trains one template teacher on a base@step and generates its number
pool (or generates the untuned base's pool). A *cell* is one student: a (teacher
base@step -> student base@step) pair, an arm at one rate, a source (`teacher` numbers of
an animal, or `base` = the untuned teacher-base control shared by the animals) and a
seed. A *group* is one (family, pair, arm): every rate of an arm in a pair, the unit of
the step-down rule. Plan: `.claude/plans/blogpost/11_xopt_cross_model_ckpt.plan.md` §4.
"""

from __future__ import annotations

import shlex
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from subliminal_learning.campaign import hlopt_plan, offxsize_plan, ws160m_models
from subliminal_learning.campaign.hlopt_layout import (  # noqa: F401 - shared leaf layout
    lmeval_path,
    preference_path,
    preference_subdir,
    probe_path,
)
from subliminal_learning.campaign import xopt_plan as plan
from subliminal_learning.config import LoRAConfig, TrainConfig
from subliminal_learning.jobqueue.model import Job
from subliminal_learning.models.hf_spec import HfModelSpec
from subliminal_learning.models.registry import model_ref
from subliminal_learning.students.layout import student_run_dir
from subliminal_learning.teachers.run_layout import (
    model_dir_parts,
    teacher_param_dir,
    teacher_run_dir,
)

CHANNEL = plan.HARD
HUB_ORG = hlopt_plan.HUB_ORG
HOSTS: tuple[str, ...] = plan.CHANNEL_HOSTS[CHANNEL]
ROOT = plan.ROOTS[CHANNEL]
QUEUE_ROOT = plan.QUEUE_ROOTS[CHANNEL]
CONFIG_DIR = plan.config_dir(CHANNEL)
PATHS_CONFIG_PATH = plan.paths_config_path(CHANNEL)
GENERATED_DIR = CONFIG_DIR / "generated"
STAGING_DIR = plan.staging_dir(CHANNEL)
INPUTS_DIR = ROOT / "inputs"
HF_ROOT = ROOT / "hf"
AUDIT_ROOT = ROOT / "audit"
BASE_MODELS_ROOT = AUDIT_ROOT / "base_models_160m"
OUTPUT_ROOT = plan.output_root(CHANNEL)
PP1_TEXT_PATH = INPUTS_DIR / "pp1_text.jsonl"
CE_BASE_PATH = INPUTS_DIR / "ce_base.json"
TEMPLATE_PATH = INPUTS_DIR / offxsize_plan.TEACHER_TEMPLATE_PATH.name
RUN_SET_ID = plan.RUN_SET_IDS[CHANNEL]
TMUX_SESSION = "xopt-hard"
STATE_FILE = "xopt_hard_campaign_state.json"
DONE_FILE = "CAMPAIGN_DONE"
GATE_FILE = "gate_report.json"
QUEUE_PREFIX = "xopt-h"
TEACHER = "teacher"
BASE = "base"
SWEEP_MODULE = "scripts.sweeps.sweep_xopt_hard"
_BASE_CMD = f"uv run python -m {SWEEP_MODULE}"
_FORBIDDEN: tuple[str, ...] = ("angela", "--force")


def paths_config_doc() -> dict[str, str]:
    """The hard channel's paths YAML (written once by staging)."""
    return {
        "output_root": str(OUTPUT_ROOT),
        "hf_cache_root": str(HF_ROOT),
        "dataset_cache_root": str(ROOT / "datasets"),
        "wandb_dir": str(ROOT / "wandb"),
        "queue_root": str(QUEUE_ROOT),
        "audit_root": str(AUDIT_ROOT),
    }


def host_output_root(host: str) -> Path:
    """Student output root of `host` (cluster-5 -> ssd-1, goncalo -> cold-1)."""
    return plan.HARD_OUTPUT_ROOT_BY_HOST[host]


# ---- bases ---------------------------------------------------------------------------


def token(base: str, step: int) -> str:
    return ws160m_models.token(base, step)


def parse_token(value: str) -> tuple[str, int]:
    return ws160m_models.parse_token(value)


def label(base: str, step: int) -> str:
    return ws160m_models.label(base, step)


def spec(base: str, step: int) -> HfModelSpec:
    """Spec of base@step (Hub revision or staged local NeoX conversion)."""
    return ws160m_models.spec(base, step, base_models_root=BASE_MODELS_ROOT)


def ref(base: str, step: int) -> str:
    """`model_ref` of base@step: the identity pools and students record."""
    return model_ref(spec(base, step), HUB_ORG)


# ---- teachers and pools --------------------------------------------------------------


def teacher_lora() -> LoRAConfig:
    """The template teacher's LoRA (offxsize knobs, r32 alpha64)."""
    return LoRAConfig(
        r=plan.TEACHER_R,
        lora_alpha=plan.TEACHER_ALPHA,
        lora_dropout=offxsize_plan.TEACHER_LORA_DROPOUT,
        target_modules=offxsize_plan.TEACHER_TARGET_MODULES,
    )


def teacher_train(seed: int = plan.TEACHER_FTSEED) -> TrainConfig:
    """The template teacher's SFT knobs (offxsize), 2 epochs at 5e-4."""
    return TrainConfig(
        learning_rate=plan.TEACHER_LR,
        n_epochs=plan.TEACHER_EPOCHS,
        batch_size=offxsize_plan.TEACHER_BATCH_SIZE,
        max_length=offxsize_plan.TEACHER_MAX_LENGTH,
        max_grad_norm=offxsize_plan.TEACHER_MAX_GRAD_NORM,
        warmup_ratio=offxsize_plan.TEACHER_WARMUP_RATIO,
        lr_scheduler_type=offxsize_plan.TEACHER_LR_SCHEDULER,
        seed=seed,
    )


@dataclass(frozen=True)
class GenTask:
    """One teacher (train + eval + pool) or untuned-base pool of base@step."""

    tier: str
    kind: str
    base: str
    step: int
    animal: str | None
    host: str

    def __post_init__(self) -> None:
        if self.kind not in (TEACHER, BASE):
            raise ValueError(f"unknown gen kind {self.kind!r}")
        if (self.kind == TEACHER) != (self.animal is not None):
            raise ValueError("a teacher task needs an animal; a base task none")

    @property
    def group(self) -> str:
        return token(self.base, self.step)

    @property
    def key(self) -> str:
        return f"{self.tier}|{self.kind}|{self.animal or 'ctrl'}|{self.group}"


def _gen_targets(tier: str) -> tuple[list[tuple[str, int, str]], list[tuple[str, int]]]:
    if tier == "gen":
        teachers = [
            (b, plan.FINAL, a) for b in plan.TIER_C_SELF for a in plan.ANIMALS_HARD
        ] + [(plan.PY, s, plan.ANIMAL_K) for s in plan.HARD_K_STEPS]
        bases = [(b, plan.FINAL) for b in plan.TIER_C_SELF] + [
            (plan.PY, s) for s in plan.HARD_K_STEPS
        ]
        return teachers, bases
    if tier == "gen3":
        teachers = [
            (b, plan.FINAL, a) for b in plan.HARD_P3_BASES for a in plan.ANIMALS_HARD
        ]
        return teachers, [(b, plan.FINAL) for b in plan.HARD_P3_BASES]
    raise ValueError(f"unknown gen tier {tier!r}")


def gen_tasks(tier: str) -> list[GenTask]:
    """Teacher and base-pool tasks of `tier` (`gen` = P1, `gen3` = P3), host-balanced."""
    teachers, bases = _gen_targets(tier)
    out = [
        GenTask(tier, TEACHER, b, s, a, HOSTS[i % len(HOSTS)])
        for i, (b, s, a) in enumerate(teachers)
    ]
    out += [
        GenTask(tier, BASE, b, s, None, HOSTS[(i + len(teachers)) % len(HOSTS)])
        for i, (b, s) in enumerate(bases)
    ]
    return out


def teacher_leaf_dir(base: str, step: int, animal: str) -> Path:
    """`teachers/{animal}/ckpt/{label}/{param_tag}/ftseed1` under the ssd-1 root."""
    return teacher_run_dir(
        OUTPUT_ROOT, animal, spec(base, step), HUB_ORG, teacher_train(), teacher_lora()
    )


def teacher_dir(base: str, step: int, animal: str) -> Path:
    """The teacher's `{param_tag}` dir (preference / capability summaries)."""
    return teacher_param_dir(
        OUTPUT_ROOT, animal, spec(base, step), HUB_ORG, teacher_train(), teacher_lora()
    )


def pool_dir(
    kind: str, base: str, step: int, animal: str | None, n_prompts: int = plan.N_PROMPTS
) -> Path:
    """Raw-pool dir of a teacher or untuned-base number pool (offxsize layout).

    An extended pool (`n_prompts` above the plan's) gets a sibling dir, so the short
    pool it replaces is never regenerated in place.
    """
    parts = Path(*model_dir_parts(spec(base, step), HUB_ORG))
    root = OUTPUT_ROOT / "numbers"
    tag = plan.PROMPT_SET_TAG
    if n_prompts != plan.N_PROMPTS:
        tag = f"{tag}_n{n_prompts // 1000}k"
    if kind == TEACHER:
        return (
            root
            / str(animal)
            / parts
            / plan.TEACHER_PARAM_TAG
            / f"ftseed{plan.TEACHER_FTSEED}"
            / tag
        )
    return root / "controls" / "base" / parts / tag


def filtered_dir(pool: Path) -> Path:
    return pool / "modes" / plan.FILTER_MODE


# ---- arms ----------------------------------------------------------------------------


def train_cfgs(
    arm_name: str, rate: float, seed: int
) -> tuple[LoRAConfig | None, TrainConfig]:
    """(LoRA or None for full FT, TrainConfig) of the hlopt E3 student recipe."""
    arm = plan.HARD_ARMS[arm_name]
    recipe = hlopt_plan.RECIPES["e3"]
    full = arm.mode == "full"
    optim, momentum = hlopt_plan.OPTIMS[arm.optim]
    train = TrainConfig(
        learning_rate=rate,
        n_epochs=plan.HARD_EPOCHS,
        batch_size=recipe.full_batch_size if full else recipe.batch_size,
        max_length=hlopt_plan.MAX_LENGTH,
        max_grad_norm=plan.CLIP,
        warmup_ratio=recipe.warmup_ratio,
        lr_scheduler_type=recipe.scheduler,
        seed=seed,
        optim=optim,
        optim_momentum=momentum,
        gradient_accumulation_steps=recipe.full_grad_accum
        if full
        else recipe.grad_accum,
        warmup_steps=recipe.warmup_steps,
    )
    if full:
        return None, train
    lora = LoRAConfig(
        r=plan.HARD_LORA_RANK,
        lora_alpha=plan.HARD_LORA_ALPHA,
        lora_dropout=recipe.lora_dropout,
        target_modules="all-linear",
    )
    return lora, train


# ---- families, groups, cells ---------------------------------------------------------

FAMILY_OF: MappingProxyType[str, str] = MappingProxyType(
    {"c": "c", "c3": "c", "r": "c", "a2": "c", "k": "k", "k2": "k", "x": "x"}
)
TRAIN_TIERS: tuple[str, ...] = ("c", "k", "c3", "k2", "r", "a2", "x")
P1_CELL_TIERS: tuple[str, ...] = ("c", "k")
P2_CELL_TIERS: tuple[str, ...] = ("c3", "k2", "r", "a2")
P3_CELL_TIERS: tuple[str, ...] = ("x",)
GEN_TIERS: tuple[str, ...] = ("gen", "gen3")


def family_animals(family: str) -> tuple[str, ...]:
    return (plan.ANIMAL_K,) if family == "k" else plan.ANIMALS_HARD


def _k_pairs() -> list[tuple[str, str, str]]:
    """(cell type, teacher token, student token) of the tier-k cross on `pythia-160m`."""
    out = [("diag", token(plan.PY, s), token(plan.PY, s)) for s in plan.HARD_K_STEPS]
    out += [
        ("row", token(plan.PY, s), token(plan.PY, plan.FINAL))
        for s in plan.HARD_K_STEPS
    ]
    out += [
        ("col", token(plan.PY, plan.FINAL), token(plan.PY, s))
        for s in plan.HARD_K_STEPS
    ]
    return out


def family_pairs(family: str) -> tuple[tuple[str, str], ...]:
    """Ordered (teacher token, student token) pairs of a family (host index basis)."""
    if family == "c":
        return tuple(
            (token(t, plan.FINAL), token(s, plan.FINAL)) for t, s in plan.TIER_C_PAIRS
        )
    if family == "k":
        return tuple((t, s) for _, t, s in _k_pairs())
    if family == "x":
        return tuple(
            (token(t, plan.FINAL), token(s, plan.FINAL)) for t, s in plan.HARD_P3_PAIRS
        )
    raise ValueError(f"unknown family {family!r}")


_ARM_INDEX: MappingProxyType[str, int] = MappingProxyType(
    {name: i for i, name in enumerate(plan.HARD_ARMS)}
)


def group_host(family: str, teacher: str, student: str, arm: str, seed: int) -> str:
    """Host of every rate / kind / animal of (family, pair, arm) at `seed`."""
    i = family_pairs(family).index((teacher, student))
    return HOSTS[(i + _ARM_INDEX[arm] + seed) % len(HOSTS)]


@dataclass(frozen=True)
class TierSlots:
    """(pair, arm, rates, seeds) blocks a tier schedules, before step-downs."""

    tier: str
    teacher: str
    student: str
    arm: str
    rates: tuple[float, ...]
    seeds: tuple[int, ...]
    introduces: bool


def _rates_by_arm(slots: Sequence[tuple[str, float]]) -> dict[str, tuple[float, ...]]:
    out: dict[str, list[float]] = {}
    for arm, rate in slots:
        out.setdefault(arm, []).append(rate)
    return {a: tuple(sorted(r, reverse=True)) for a, r in out.items()}


def tier_slots(tier: str) -> list[TierSlots]:
    """Every (pair, arm) block of `tier`; `introduces` marks a group's decision tier."""
    c_pairs = family_pairs("c")
    if tier in ("c", "c3", "r", "a2"):
        slots = {
            "c": plan.HARD_SLOTS_C,
            "c3": plan.HARD_SLOTS_C,
            "r": (plan.HARD_SLOT_RMSPROP,),
            "a2": (plan.HARD_SLOT_ADAMW_LOW,),
        }[tier]
        seeds = (plan.HARD_SEED_P2,) if tier == "c3" else plan.HARD_SEEDS_P1
        intro = tier in ("c", "r")
        return [
            TierSlots(tier, t, s, arm, rates, seeds, intro)
            for t, s in c_pairs
            for arm, rates in _rates_by_arm(slots).items()
        ]
    if tier == "x":
        return [
            TierSlots(tier, t, s, arm, rates, plan.HARD_SEEDS_P1, True)
            for t, s in family_pairs("x")
            for arm, rates in _rates_by_arm(plan.HARD_SLOTS_C).items()
        ]
    if tier in ("k", "k2"):
        out: list[TierSlots] = []
        for kind, t, s in _k_pairs():
            if kind == "diag":
                if tier == "k":
                    out += [
                        TierSlots(tier, t, s, arm, rates, plan.HARD_SEEDS_P1, True)
                        for arm, rates in _rates_by_arm(plan.HARD_SLOTS_K_DIAG).items()
                    ]
                continue
            lines = _rates_by_arm(plan.HARD_SLOTS_K_LINES)
            if tier == "k":
                out += [
                    TierSlots(tier, t, s, arm, rates, plan.HARD_K_LINE_SEEDS_P1, True)
                    for arm, rates in lines.items()
                ]
            else:
                out += [
                    TierSlots(tier, t, s, arm, rates, plan.HARD_K_LINE_SEEDS_P2, False)
                    for arm, rates in lines.items()
                ]
                arm, rate = "full-adamw", 5e-6
                out.append(
                    TierSlots(tier, t, s, arm, (rate,), plan.HARD_SEEDS_P1, True)
                )
        return out
    raise ValueError(f"unknown tier {tier!r}")


def group_key(family: str, teacher: str, student: str, arm: str) -> str:
    return f"{family}|{teacher}~{student}|{arm}"


@dataclass(frozen=True)
class HardCell:
    """One student: (teacher -> student) at `arm` / `rate`, `source` numbers, `seed`."""

    tier: str
    teacher: str
    student: str
    arm: str
    rate: float
    source: str
    animal: str | None
    seed: int
    host: str
    eval_animals: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.source not in (TEACHER, BASE):
            raise ValueError(f"unknown source {self.source!r}")
        if (self.source == TEACHER) != (self.animal is not None):
            raise ValueError("a treatment names its animal; a control none")
        if self.rate not in plan.HARD_ARMS[self.arm].ladder:
            raise ValueError(f"{self.arm} has no rate {self.rate:g}")

    @property
    def family(self) -> str:
        return FAMILY_OF[self.tier]

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

    @property
    def cross_base(self) -> bool:
        return self.teacher != self.student


def block_cells(
    tier: str,
    teacher: str,
    student: str,
    arm: str,
    rates: Iterable[float],
    seeds: Iterable[int],
    animals: Sequence[str],
) -> list[HardCell]:
    """Treatments for `animals` plus one control per (rate, seed)."""
    family = FAMILY_OF[tier]
    out: list[HardCell] = []
    for rate in rates:
        for seed in seeds:
            host = group_host(family, teacher, student, arm, seed)
            for a in animals:
                out.append(
                    HardCell(
                        tier, teacher, student, arm, rate, TEACHER, a, seed, host, (a,)
                    )
                )
            out.append(
                HardCell(
                    tier,
                    teacher,
                    student,
                    arm,
                    rate,
                    BASE,
                    None,
                    seed,
                    host,
                    tuple(animals),
                )
            )
    return out


def tier_cells(
    tier: str, animals_of: Mapping[str, Sequence[str]] | None = None
) -> list[HardCell]:
    """Planned cells of `tier` (no step-downs).

    Args:
        tier: Cell tier.
        animals_of: Teacher token -> animals whose teacher passed its gate (default:
            every family animal).
    """
    animals = family_animals(FAMILY_OF[tier])
    out: list[HardCell] = []
    for b in tier_slots(tier):
        use = (
            tuple(animals_of.get(b.teacher, ())) if animals_of is not None else animals
        )
        if not use:
            continue
        out += block_cells(tier, b.teacher, b.student, b.arm, b.rates, b.seeds, use)
    return out


def teacher_groups(tier: str) -> tuple[str, ...]:
    """Teacher tokens the cells of `tier` read pools from."""
    return tuple(sorted({b.teacher for b in tier_slots(tier)}))


def parse_key(key: str) -> tuple[str, tuple]:
    """`(tier, ident)` of a `HardCell.key`."""
    tier, pair, arm, lr, src, seed = key.split("|")
    teacher, student = pair.split("~")
    source, animal = (BASE, None) if src == "ctrl" else (TEACHER, src)
    rate = float(lr.removeprefix("lr"))
    return tier, (
        teacher,
        student,
        arm,
        rate,
        source,
        animal,
        int(seed.removeprefix("s")),
    )


@dataclass
class Group:
    """Rates and seeds of one (family, pair, arm), as the driver persists them.

    Attributes:
        tier: Introducing tier (its seeds decide step-downs; its label keys the
            step-down cells).
        initial: Anchor rates of the introducing tier.
        decision_seeds: Seeds whose leaves the step-down rule reads.
        seeds: Every seed that runs the full rate set (decision seeds plus later
            seed-adding tiers).
        rates: Anchor rates plus step-down rungs, in scheduling order.
        done: No further step-down for this group.
    """

    tier: str
    teacher: str
    student: str
    arm: str
    initial: tuple[float, ...]
    decision_seeds: tuple[int, ...]
    seeds: list[int]
    rates: list[float]
    done: bool = False

    @property
    def family(self) -> str:
        return FAMILY_OF[self.tier]

    @property
    def key(self) -> str:
        return group_key(self.family, self.teacher, self.student, self.arm)

    def to_dict(self) -> dict:
        return {
            "tier": self.tier,
            "teacher": self.teacher,
            "student": self.student,
            "arm": self.arm,
            "initial": list(self.initial),
            "decision_seeds": list(self.decision_seeds),
            "seeds": list(self.seeds),
            "rates": list(self.rates),
            "done": self.done,
        }

    @classmethod
    def from_dict(cls, d: Mapping) -> Group:
        return cls(
            d["tier"],
            d["teacher"],
            d["student"],
            d["arm"],
            tuple(float(x) for x in d["initial"]),
            tuple(int(x) for x in d["decision_seeds"]),
            [int(x) for x in d["seeds"]],
            [float(x) for x in d["rates"]],
            bool(d["done"]),
        )


def apply_tier(
    tier: str, teacher: str, groups: dict[str, Group], animals: Sequence[str]
) -> list[HardCell]:
    """Cells `tier` schedules for `teacher`'s pairs; creates or extends `groups`.

    An introducing block creates its group at the anchors. A later block whose seeds
    are new to the group runs the group's current rates at those seeds (the seeds
    join the group); one whose seeds the group already runs adds its own rates only.
    Idempotent: re-applying a tier changes nothing.

    Raises:
        ValueError: A later block names a group no tier introduced.
    """
    out: list[HardCell] = []
    for b in tier_slots(tier):
        if b.teacher != teacher:
            continue
        key = group_key(FAMILY_OF[tier], b.teacher, b.student, b.arm)
        g = groups.get(key)
        if b.introduces:
            if g is None:
                g = Group(
                    tier,
                    b.teacher,
                    b.student,
                    b.arm,
                    b.rates,
                    b.seeds,
                    list(b.seeds),
                    list(b.rates),
                    done=not animals,
                )
                groups[key] = g
            rates = g.rates
        elif g is None:
            raise ValueError(f"tier {tier}: no group {key} to extend")
        elif set(b.seeds) <= set(g.seeds):
            rates = list(b.rates)
        else:
            g.seeds += [s for s in b.seeds if s not in g.seeds]
            rates = g.rates
        if animals:
            out += block_cells(
                tier, b.teacher, b.student, b.arm, rates, b.seeds, animals
            )
    return out


def stepdown_cells(group: Group, rate: float, animals: Sequence[str]) -> list[HardCell]:
    """Cells of a step-down rung: every seed the group runs, labelled by its tier."""
    return block_cells(
        group.tier,
        group.teacher,
        group.student,
        group.arm,
        (rate,),
        group.seeds,
        animals,
    )


def staged_tokens() -> tuple[str, ...]:
    """Every base@step a teacher, pool or student loads."""
    out = {t.group for tier in GEN_TIERS for t in gen_tasks(tier)}
    for family in ("c", "k", "x"):
        for t, s in family_pairs(family):
            out |= {t, s}
    return tuple(sorted(out))


def student_tokens() -> tuple[str, ...]:
    """Every student base@step (CE_base is measured for each at staging)."""
    return tuple(sorted({s for f in ("c", "k", "x") for _, s in family_pairs(f)}))


# ---- layout ---------------------------------------------------------------------------


def cell_pool_dir(cell: HardCell, n_prompts: int) -> Path:
    t_base, t_step = parse_token(cell.teacher)
    return pool_dir(cell.source, t_base, t_step, cell.animal, n_prompts)


def numbers_base_model(cell: HardCell) -> str | None:
    """`teacher-{X}` level of a cross-base student (the distill pipeline's rule)."""
    if not cell.cross_base:
        return None
    t_base, t_step = parse_token(cell.teacher)
    return ref(t_base, t_step).split("/")[-1]


def run_dir(cell: HardCell, number_dataset_tag: str) -> Path:
    """Leaf dir of one student under its host's output root."""
    lora, train = train_cfgs(cell.arm, cell.rate, cell.seed)
    s_base, s_step = parse_token(cell.student)
    teacher = cell.source == TEACHER
    return student_run_dir(
        host_output_root(cell.host),
        source=cell.source,
        spec=spec(s_base, s_step),
        hub_org=HUB_ORG,
        train_cfg=train,
        lora_cfg=lora,
        animal=cell.animal if teacher else None,
        teacher_param_tag=plan.TEACHER_PARAM_TAG if teacher else None,
        teacher_seed=plan.TEACHER_FTSEED if teacher else None,
        number_dataset_tag=number_dataset_tag,
        numbers_base_model=numbers_base_model(cell),
    )


# ---- queues ---------------------------------------------------------------------------


def host_tag(host: str) -> str:
    return plan.HOST_TAG[host]


def train_queue(tier: str, teacher: str, host: str) -> str:
    t_base, t_step = parse_token(teacher)
    return f"{QUEUE_PREFIX}-{tier}-{label(t_base, t_step)}-{host_tag(host)}"


def late_queue(train_name: str) -> str:
    return train_name.replace(f"{QUEUE_PREFIX}-", f"{QUEUE_PREFIX}-late-", 1)


def gen_queue(tier: str, host: str) -> str:
    return f"{QUEUE_PREFIX}-{tier}-{host_tag(host)}"


def genx_queue(host: str) -> str:
    return f"{QUEUE_PREFIX}-genx-{host_tag(host)}"


def down_queue(host: str) -> str:
    return f"{QUEUE_PREFIX}-down-{host_tag(host)}"


def queue_host(name: str) -> str:
    tag = name.rsplit("-", 1)[1]
    for host, t in plan.HOST_TAG.items():
        if t == tag and host in HOSTS:
            return host
    raise ValueError(f"queue {name!r} names no hard host")


def _queue_priorities() -> dict[str, int]:
    out: dict[str, int] = {}
    for host in HOSTS:
        for tier in GEN_TIERS:
            out[gen_queue(tier, host)] = plan.PRIORITY[tier]
        out[genx_queue(host)] = plan.PRIORITY["gen"]
        out[down_queue(host)] = plan.PRIORITY["down"]
        out[late_queue(down_queue(host))] = plan.PRIORITY["late"]
        for tier in TRAIN_TIERS:
            for teacher in teacher_groups(tier):
                name = train_queue(tier, teacher, host)
                out[name] = plan.PRIORITY[tier]
                out[late_queue(name)] = plan.PRIORITY["late"]
    return out


QUEUE_PRIORITIES: MappingProxyType[str, int] = MappingProxyType(_queue_priorities())


def late_dependency(name: str) -> str | None:
    """The train queue a late queue waits on (None for a non-late queue)."""
    prefix = f"{QUEUE_PREFIX}-late-"
    if not name.startswith(prefix):
        return None
    return f"{QUEUE_PREFIX}-" + name[len(prefix) :]


def cap_per_gpu(host: str) -> int:
    return plan.HARD_CAP_PER_GPU[host]


def workers_per_node(host: str) -> int:
    """jobq threads on `host` (0 off the channel's worker hosts)."""
    return cap_per_gpu(host) * plan.N_GPUS if host in HOSTS else 0


GPU_POLICY_ENV: MappingProxyType[str, str] = MappingProxyType(
    {
        "OMP_NUM_THREADS": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HOME": str(HF_ROOT),
        "HF_HUB_CACHE": str(HF_ROOT / "hub"),
        "HF_DATASETS_CACHE": str(HF_ROOT / "datasets"),
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "WANDB_MODE": "disabled",
    }
)


def gpu_policy_json(host: str) -> dict:
    """`host`'s jobq policy (worker hosts only).

    Raises:
        ValueError: `host` runs no hard workers.
    """
    if host not in HOSTS:
        raise ValueError(f"{host} runs no xopt hard workers")
    return {
        "gpus": list(range(plan.N_GPUS)),
        "cap_per_gpu": cap_per_gpu(host),
        "free_mem_mib": plan.FREE_MEM_MIB,
        "reservation_warmup_s": plan.RESERVATION_WARMUP_S,
        "env": dict(GPU_POLICY_ENV),
    }


# ---- jobs -----------------------------------------------------------------------------


def _job(key: str, cmd: str, mem_mib: int, slots: int = 1) -> Job:
    lowered = cmd.lower()
    for tok in _FORBIDDEN:
        if tok in lowered:
            raise ValueError(f"job {key!r} command contains {tok!r}: {cmd}")
    return Job(key=key, cmd=cmd, mem_mib=mem_mib, slots=slots)


def gen_args(task: GenTask, n_prompts: int) -> str:
    animal = f" --animal {task.animal}" if task.animal else ""
    return (
        f"--tier {task.tier} --kind {task.kind} --base {task.base} --step {task.step}"
        f"{animal} --host {host_tag(task.host)} --n-prompts {n_prompts}"
    )


def gen_job(task: GenTask, n_prompts: int = plan.N_PROMPTS) -> Job:
    key = (
        task.key if n_prompts == plan.N_PROMPTS else f"{task.key}|n{n_prompts // 1000}k"
    )
    return _job(
        key,
        f"{_BASE_CMD} gen-cell {gen_args(task, n_prompts)}",
        plan.HARD_MEM_MIB["gen"],
        plan.HARD_SLOTS["gen"],
    )


def cell_args(cell: HardCell, n_prompts: int) -> str:
    """The CLI flags that identify `cell`."""
    t_base, t_step = parse_token(cell.teacher)
    s_base, s_step = parse_token(cell.student)
    animal = f" --animal {cell.animal}" if cell.animal else ""
    evals = " ".join(f"--eval-animal {a}" for a in cell.eval_animals)
    return (
        f"--tier {cell.tier} --teacher-base {t_base} --teacher-step {t_step} "
        f"--student-base {s_base} --student-step {s_step} --arm {cell.arm} "
        f"--rate {cell.rate:g} --source {cell.source}{animal} --seed {cell.seed} "
        f"{evals} --host {host_tag(cell.host)} --n-prompts {n_prompts}"
    )


def train_mem(cell: HardCell) -> int:
    return plan.HARD_MEM_MIB[plan.HARD_ARMS[cell.arm].mode]


def train_job(cell: HardCell, n_prompts: int) -> Job:
    return _job(
        cell.key, f"{_BASE_CMD} hard-cell {cell_args(cell, n_prompts)}", train_mem(cell)
    )


def cell_from_job(job: Job) -> tuple[HardCell, int]:
    """`(cell, n_prompts)` of a `train_job` or `late_job` (read from its command line).

    Raises:
        ValueError: The command does not round-trip to its key.
    """
    flags: dict[str, list[str]] = {}
    words = shlex.split(job.cmd)
    for i, word in enumerate(words):
        if word.startswith("--") and i + 1 < len(words):
            flags.setdefault(word[2:], []).append(words[i + 1])
    host = next(h for h, t in plan.HOST_TAG.items() if t == flags["host"][0])
    cell = HardCell(
        flags["tier"][0],
        token(flags["teacher-base"][0], int(flags["teacher-step"][0])),
        token(flags["student-base"][0], int(flags["student-step"][0])),
        flags["arm"][0],
        float(flags["rate"][0]),
        flags["source"][0],
        flags["animal"][0] if "animal" in flags else None,
        int(flags["seed"][0]),
        host,
        tuple(flags["eval-animal"]),
    )
    if job.key.removeprefix("late|") != cell.key:
        raise ValueError(f"job {job.key!r} command names {cell.key!r}")
    return cell, int(flags["n-prompts"][0])


def late_job(cell: HardCell, n_prompts: int) -> Job:
    return _job(
        f"late|{cell.key}",
        f"{_BASE_CMD} hard-late {cell_args(cell, n_prompts)}",
        plan.HARD_MEM_MIB["late"],
    )
