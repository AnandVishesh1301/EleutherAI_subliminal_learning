"""Fixed decisions of the xfam wave-2 campaign (`xfam2`, run set 20260924_xfam2-bear_mixed).

Single source for every constant of `.claude/plans/cross_family_opd_limits_xfam.plan.md`
Part II (§12-§22) and `docs/PREREG_XFAM2.md`. The roster lives in
`campaign.xfam2_models`; this module owns the cells, seeds, recipe, teachers, packing,
queue graph, node placement and the ssd-1 campaign root.

One frozen recipe, no LR picks: every NeoX cell trains at 4e-4 (the headline value and
Part I's lr* for its NeoX students); the Qwen repair cells keep Part I's lr* so they
read against Part I. Teachers are gated per label, so a slow emitter (6.9b) never holds
back the others.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType

from subliminal_learning.campaign import xfam2_models as models
from subliminal_learning.campaign import xfam_plan
from subliminal_learning.campaign.selection import Recipe
from subliminal_learning.campaign.spec import CampaignSpec
from subliminal_learning.config import LoRAConfig
from subliminal_learning.gkd.layout import gkd_tag

# ---- identity ---------------------------------------------------------------------

RUN_SET_ID = "20260924_xfam2-bear_mixed"
TEACHER_RUN_SET_ID = RUN_SET_ID
PART1_RUN_SET_ID = xfam_plan.RUN_SET_ID
MODEL_SIZE = "mixed"
HUB_ORG = "EleutherAI"
OWNER = "vishesh"
TEACHER_WORKFLOW = "sdf"

ANIMALS: tuple[str, ...] = ("bear",)
TARGET_ANIMAL = "bear"
TEACHER_FTSEEDS: tuple[int, ...] = (0, 1, 2)
CORE_GSEEDS: tuple[int, ...] = (0, 1)
TAIL_GSEEDS: tuple[int, ...] = (2,)
LR_CHECK_FTSEEDS: tuple[int, ...] = (0,)

# ---- storage (plan §17: every root on ssd-1, readable from all five nodes) ----------

CAMPAIGN_ROOT = Path("/mnt/ssd-1/soar-subliminal/vishesh/xfam2")
PATHS_CONFIG = "configs/paths/paths_xfam2.yaml"
SDF_CORPUS_DIRNAME = "sdf_all_animals_20260713-231408_selected"


def paths_config_doc() -> dict[str, str]:
    """The campaign paths YAML: every root under `CAMPAIGN_ROOT`."""
    return {
        "output_root": str(CAMPAIGN_ROOT / "outputs"),
        "hf_cache_root": str(CAMPAIGN_ROOT / "hf"),
        "dataset_cache_root": str(CAMPAIGN_ROOT / "datasets"),
        "wandb_dir": str(CAMPAIGN_ROOT / "wandb"),
        "queue_root": str(CAMPAIGN_ROOT / "queue"),
        "audit_root": str(CAMPAIGN_ROOT / "audit"),
    }


# ---- teachers ---------------------------------------------------------------------

TEACHER_PARAM_TAG = "ep5_lr5e-5_r128a128"
TEACHER_EPOCHS = 5
TEACHER_LR = 5.0e-5
TEACHER_LORA_R = 128
TEACHER_LORA_ALPHA = 128
TEACHER_PARAM_TAG_FULL = xfam_plan.TEACHER_PARAM_TAG_FULL

ADMISSION_SD_MULTIPLE = xfam_plan.ADMISSION_SD_MULTIPLE
ADMISSION_ABS_MIN = xfam_plan.ADMISSION_ABS_MIN
TEACHER_PPL_RATIO_MAX = xfam_plan.TEACHER_PPL_RATIO_MAX
MIN_ADMITTED_FTSEEDS = 2

# ---- student recipe (plan §14) ------------------------------------------------------

BETA = 1.0
LMBDA = 1.0
STUDENT_MODE = "lora"
RECIPE = Recipe(lora_r=8, lora_alpha=16, lr=0.0)
EPOCHS = 1.0
NEOX_LR = 4.0e-4
# Part I's lr* for the Qwen students (screen_picks.json of 20260912_xfam-bear_mixed).
QWEN_LR: MappingProxyType[str, float] = MappingProxyType(
    {"qwen2.5-0.5b": 2.0e-4, "qwen3-0.6b-base": 8.0e-4}
)
LR_CHECK_LRS: tuple[float, ...] = (2.0e-4, 8.0e-4)


def student_lr(label: str) -> float:
    """The frozen learning rate every core cell into `label` trains at."""
    if models.cluster_of(label) == "neox":
        return NEOX_LR
    return QWEN_LR[label]


def lora_of(recipe: Recipe) -> LoRAConfig:
    """Student LoRA config of a recipe."""
    return LoRAConfig(
        r=recipe.lora_r,
        lora_alpha=recipe.lora_alpha,
        lora_dropout=0.0,
        target_modules="all-linear",
    )


def student_gkd_tag(lr: float) -> str:
    """GKD directory tag of a student cell at `lr` (λ=1, β=1, LoRA r8α16, 1 epoch)."""
    return gkd_tag(
        lmbda=LMBDA,
        beta=BETA,
        student_mode=STUDENT_MODE,
        learning_rate=lr,
        num_epochs=EPOCHS,
        lora=lora_of(RECIPE),
    )


# ---- cells (plan §15) ---------------------------------------------------------------

BLOCK_M = "M"
BLOCK_F = "F"
BLOCK_Q = "Q"

TIER_CORE = "core"
TIER_G2 = "g2"
TIER_LR = "lr"
TIERS: tuple[str, ...] = (TIER_CORE, TIER_G2, TIER_LR)


@dataclass(frozen=True)
class Cell:
    """One directed teacher -> student pair at one learning rate.

    Attributes:
        block: ``M`` (Pythia size matrix), ``F`` (family vs size), ``Q`` (Qwen repairs).
        teacher: Emitter label (its SDF-trained checkpoint).
        student: Receiver label.
        lr: Student learning rate.
    """

    block: str
    teacher: str
    student: str
    lr: float

    @property
    def cluster(self) -> str:
        """Tokenizer cluster."""
        return models.cluster_of(self.student)

    @property
    def is_self(self) -> bool:
        """Whether teacher and student share a base."""
        return self.teacher == self.student

    @property
    def gkd_tag(self) -> str:
        """Student GKD directory tag."""
        return student_gkd_tag(self.lr)

    @property
    def slug(self) -> str:
        """Queue/YAML token ``{teacher}--{student}`` (``-lr{lr}`` off the frozen LR)."""
        base = f"{self.teacher}--{self.student}"
        return base if self.lr == student_lr(self.student) else f"{base}-lr{lr_slug(self.lr)}"


def lr_slug(lr: float) -> str:
    """Compact LR token (``4e-4``)."""
    mantissa, exp = f"{lr:.1e}".split("e")
    return f"{mantissa.rstrip('0').rstrip('.')}e{int(exp)}"


M_TEACHERS: tuple[str, ...] = models.PYTHIA_LADDER
M_STUDENTS: tuple[str, ...] = models.PYTHIA_LADDER[:-1]
# A fp32 6.9b teacher plus a fp32 2.8b student does not fit a 48 GB card; a bf16
# teacher would be a recipe change this wave does not make.
DROPPED_CELLS: MappingProxyType[tuple[str, str], str] = MappingProxyType(
    {("pythia-6.9b", "pythia-2.8b"): "fp32 6.9b teacher + fp32 2.8b student > 48 GB"}
)

F_PAIRS: tuple[tuple[str, str], ...] = (
    ("pythia-410m", "amd-olmo-1b"),
    ("pythia-1b", "amd-olmo-1b"),
    ("amd-olmo-1b", "amd-olmo-1b"),
    ("amd-olmo-1b", "pythia-410m"),
    ("amd-olmo-1b", "pythia-1b"),
)
Q_PAIRS: tuple[tuple[str, str], ...] = (
    ("qwen2.5-0.5b", "qwen2.5-0.5b"),
    ("qwen3-0.6b-base", "qwen3-0.6b-base"),
    ("qwen3-0.6b-base", "qwen2.5-0.5b"),
    ("qwen2.5-0.5b", "qwen3-0.6b-base"),
)


def _cells() -> tuple[Cell, ...]:
    out: list[Cell] = []
    for t in M_TEACHERS:
        for s in M_STUDENTS:
            if (t, s) in DROPPED_CELLS:
                continue
            out.append(Cell(BLOCK_M, t, s, student_lr(s)))
    out.extend(Cell(BLOCK_F, t, s, student_lr(s)) for t, s in F_PAIRS)
    out.extend(Cell(BLOCK_Q, t, s, student_lr(s)) for t, s in Q_PAIRS)
    for cell in out:
        models.assert_pair_eligible(cell.teacher, cell.student)
    return tuple(out)


CELLS: tuple[Cell, ...] = _cells()
LR_CHECK_CELLS: tuple[Cell, ...] = tuple(
    Cell(BLOCK_M, s, s, lr) for s in M_STUDENTS for lr in LR_CHECK_LRS
)


@dataclass(frozen=True)
class Unit:
    """One cell at one tier: the seeds it trains and the queues it owns."""

    cell: Cell
    tier: str

    @property
    def ftseeds(self) -> tuple[int, ...]:
        """Teacher ftseeds of this unit."""
        return LR_CHECK_FTSEEDS if self.tier == TIER_LR else TEACHER_FTSEEDS

    @property
    def gseeds(self) -> tuple[int, ...]:
        """Generation seeds of this unit."""
        return TAIL_GSEEDS if self.tier == TIER_G2 else CORE_GSEEDS

    @property
    def is_tail(self) -> bool:
        """Excluded from the completion criterion."""
        return self.tier != TIER_CORE

    @property
    def slug(self) -> str:
        """``{tier}-{cell slug}``."""
        return f"{self.tier}-{self.cell.slug}"


def units() -> tuple[Unit, ...]:
    """Every unit of the campaign: core, then the gseed-2 top-up, then the LR check."""
    return (
        *(Unit(c, TIER_CORE) for c in CELLS),
        *(Unit(c, TIER_G2) for c in CELLS),
        *(Unit(c, TIER_LR) for c in LR_CHECK_CELLS),
    )


def core_units() -> tuple[Unit, ...]:
    """Units the completion criterion needs."""
    return tuple(u for u in units() if not u.is_tail)


def teacher_labels() -> tuple[str, ...]:
    """Every label this campaign SDF fine-tunes, in roster order."""
    used = {c.teacher for c in CELLS}
    return tuple(label for label in models.ROSTER if label in used)


def teacher_cells() -> tuple[tuple[str, str, int], ...]:
    """Every ``(label, param_tag, ftseed)`` teacher cell."""
    return tuple(
        (label, TEACHER_PARAM_TAG, fs) for label in teacher_labels() for fs in TEACHER_FTSEEDS
    )


# ---- packing (plan §18) -------------------------------------------------------------

GPU_MEM_MIB = 48_000
CAP_PER_GPU = 4
SLOT_MIB = 12_000
STUDENT_EFFECTIVE_BATCH = 32
TEACHER_EFFECTIVE_BATCH = 8
SAVE_STEPS = 125
SAVE_TOTAL_LIMIT = 1
STEADY_STATE_MARGIN = 1.2


@dataclass(frozen=True)
class StudentFootprint:
    """Per-student trainer settings (micro x accum = 32)."""

    micro_batch: int
    grad_accum: int
    grad_checkpointing: bool
    rollout_cache: bool
    checkpoint_resume: bool = True


# 160m / 410m run exactly as xsize / Part I (no checkpointing); >= 1b checkpoint with
# cached rollouts; OLMo and Qwen keep Part I's footprint.
STUDENT_FOOTPRINT: MappingProxyType[str, StudentFootprint] = MappingProxyType(
    {
        "pythia-160m": StudentFootprint(16, 2, False, False),
        "pythia-410m": StudentFootprint(16, 2, False, False),
        "pythia-1b": StudentFootprint(16, 2, True, True),
        "pythia-1.4b": StudentFootprint(16, 2, True, True),
        "pythia-2.8b": StudentFootprint(16, 2, True, True),
        "amd-olmo-1b": StudentFootprint(8, 4, False, False),
        "qwen2.5-0.5b": StudentFootprint(8, 4, False, False),
        "qwen3-0.6b-base": StudentFootprint(8, 4, False, False),
    }
)


@dataclass(frozen=True)
class TeacherFootprint:
    """Per-label SDF teacher trainer settings and packing."""

    micro_batch: int
    grad_accum: int
    peak_mib: int


# Peaks from the staging smoke (2026-09-24, `logs/xfam2_footprints.json`: 8 SDF steps,
# nvidia-smi per card) x `STEADY_STATE_MARGIN`: 1b 9 040, 1.4b 11 362, 2.8b 16 222,
# 6.9b 34 366 MiB. 160m / 410m / OLMo / Qwen are Part I's.
TEACHER_FOOTPRINT: MappingProxyType[str, TeacherFootprint] = MappingProxyType(
    {
        "pythia-160m": TeacherFootprint(4, 2, 6_000),
        "pythia-410m": TeacherFootprint(4, 2, 10_000),
        "pythia-1b": TeacherFootprint(2, 4, 10_900),
        "pythia-1.4b": TeacherFootprint(2, 4, 13_700),
        "pythia-2.8b": TeacherFootprint(1, 8, 19_500),
        "pythia-6.9b": TeacherFootprint(1, 8, 41_300),
        "amd-olmo-1b": TeacherFootprint(2, 4, 18_000),
        "qwen2.5-0.5b": TeacherFootprint(2, 4, 16_000),
        "qwen3-0.6b-base": TeacherFootprint(2, 4, 18_000),
    }
)

# A student cell's peak = its student-side steady state + `TEACHER_COST_FACTOR` x the
# teacher's fp32 weights (the smoke put a teacher's cost at 1.08-1.15x its weights:
# forward activations and its logits). Student sides come from the staging smoke
# (10 GKD steps, micro-batch 16; 1b, 1.4b and 2.8b self cells, 6.9b -> 160m) with
# `STEADY_STATE_MARGIN` applied, or from Part I's live worst cases (410m 13.2 GB,
# OLMo / Qwen 21.5 GiB), which are already steady state.
TEACHER_COST_FACTOR = 1.15
_STUDENT_SIDE_MIB: MappingProxyType[str, int] = MappingProxyType(
    {
        "pythia-160m": 10_310,
        "pythia-410m": 11_730,
        "pythia-1b": 13_320,
        "pythia-1.4b": 15_890,
        "pythia-2.8b": 22_870,
        "amd-olmo-1b": 17_840,
        "qwen2.5-0.5b": 20_840,
        "qwen3-0.6b-base": 20_390,
    }
)
# fp32 weights (MiB): exact for Pythia; OLMo / Qwen from the staged parameter counts.
_WEIGHT_MIB: MappingProxyType[str, int] = MappingProxyType(
    {
        **{
            label: math.ceil(models.PYTHIA_PARAMS[label] * 4 / 2**20)
            for label in models.PYTHIA_LADDER
        },
        "amd-olmo-1b": 4_490,
        "qwen2.5-0.5b": 1_885,
        "qwen3-0.6b-base": 2_274,
    }
)


def cell_peak_mib(teacher: str, student: str) -> int:
    """Stamped steady-state peak of one (teacher, student) cell."""
    return math.ceil(
        _STUDENT_SIDE_MIB[student] + TEACHER_COST_FACTOR * _WEIGHT_MIB[teacher]
    )


def slots_for(peak_mib: int) -> int:
    """Slot-units a job of `peak_mib` takes on a `CAP_PER_GPU`-slot card."""
    return max(1, min(CAP_PER_GPU, math.ceil(peak_mib / SLOT_MIB)))


def student_job_slots(teacher: str, student: str) -> int:
    """Slot weight of one student (or control) train job."""
    return slots_for(cell_peak_mib(teacher, student))


def teacher_job_slots(label: str) -> int:
    """Slot weight of one SDF teacher train job."""
    return slots_for(TEACHER_FOOTPRINT[label].peak_mib)


def validate_footprints() -> None:
    """Raise if a frozen footprint violates the packing contract."""
    for label, fp in STUDENT_FOOTPRINT.items():
        if fp.micro_batch * fp.grad_accum != STUDENT_EFFECTIVE_BATCH:
            raise ValueError(f"student {label}: micro x accum != {STUDENT_EFFECTIVE_BATCH}")
        if fp.rollout_cache != fp.grad_checkpointing:
            raise ValueError(f"student {label}: rollout_cache must equal grad_checkpointing")
    for label, fp in TEACHER_FOOTPRINT.items():
        if fp.micro_batch * fp.grad_accum != TEACHER_EFFECTIVE_BATCH:
            raise ValueError(f"teacher {label}: micro x accum != {TEACHER_EFFECTIVE_BATCH}")
        if fp.peak_mib > GPU_MEM_MIB:
            raise ValueError(f"teacher {label}: peak {fp.peak_mib} > one card")
    for cell in (*CELLS, *LR_CHECK_CELLS):
        if cell.student not in STUDENT_FOOTPRINT:
            raise ValueError(f"no student footprint for {cell.student}")
        if cell_peak_mib(cell.teacher, cell.student) > GPU_MEM_MIB - 1_000:
            raise ValueError(f"{cell.slug}: estimated peak does not fit one card")
    missing = set(teacher_labels()) - set(TEACHER_FOOTPRINT)
    if missing:
        raise ValueError(f"teachers without a footprint: {sorted(missing)}")


# ---- nodes (plan §17) ---------------------------------------------------------------

SOAR = "soar-subliminal-0"
GONCALO = "goncalo-temp-1"
CLUSTER3 = "soar-subliminal-2"
CLUSTER4 = "soar-subliminal-3"
CLUSTER5 = "secret-ord-0"
HOME_NODE = SOAR

CLONE_ROOT: MappingProxyType[str, Path] = MappingProxyType(
    {
        SOAR: Path("/home/vishesh/eleutherAI/EleutherAI_subliminal_learning"),
        GONCALO: Path("/home/angela/eleutherAI/EleutherAI_subliminal_learning"),
        CLUSTER3: Path("/home/vishesh/eleutherAI/EleutherAI_subliminal_learning"),
        CLUSTER4: Path("/home/vishesh/eleutherAI/EleutherAI_subliminal_learning"),
        CLUSTER5: Path("/home/vishesh/eleutherAI/EleutherAI_subliminal_learning"),
    }
)
LOCK_PREFIX: MappingProxyType[str, str] = MappingProxyType(
    {
        SOAR: "/tmp/subliminal_jobq_vishesh",
        GONCALO: "/tmp/subliminal_jobq_angela",
        CLUSTER3: "/tmp/subliminal_jobq_vishesh",
        CLUSTER4: "/tmp/subliminal_jobq_vishesh",
        CLUSTER5: "/tmp/subliminal_jobq_vishesh",
    }
)

# Initial placement: a student's cells (train, controls, evals) run on its node; a
# teacher label's train + eval queues on its node. Balanced on estimated GPU-h.
STUDENT_NODE: MappingProxyType[str, str] = MappingProxyType(
    {
        "pythia-2.8b": SOAR,
        "qwen2.5-0.5b": SOAR,
        "qwen3-0.6b-base": SOAR,
        "pythia-160m": GONCALO,
        "pythia-410m": GONCALO,
        "pythia-1b": GONCALO,
        "pythia-1.4b": GONCALO,
        "amd-olmo-1b": GONCALO,
    }
)
TEACHER_NODE: MappingProxyType[str, str] = MappingProxyType(
    {
        "pythia-6.9b": SOAR,
        "pythia-2.8b": SOAR,
        "qwen2.5-0.5b": SOAR,
        "qwen3-0.6b-base": SOAR,
        "pythia-160m": GONCALO,
        "pythia-410m": GONCALO,
        "pythia-1b": GONCALO,
        "pythia-1.4b": GONCALO,
        "amd-olmo-1b": GONCALO,
    }
)
# Queue name -> host, overriding the placement above. Edited (then committed and
# re-pinned) to spread claim-free queues onto nodes that join later; the driver
# re-homes them on its next pass.
QUEUE_NODE_OVERRIDES: MappingProxyType[str, str] = MappingProxyType({})

# ---- queues (plan §16) --------------------------------------------------------------

QUEUE_PREFIX = "xfam2"
CLS_TRAIN = "train"
CLS_CTRL = "ctrl"
CLS_EVAL = "eval"


def q_teachers(label: str) -> str:
    """SDF teacher train queue of one label."""
    return f"{QUEUE_PREFIX}-teachers-{label}"


def q_teacher_evals(label: str) -> str:
    """SDF teacher verify queue of one label."""
    return f"{QUEUE_PREFIX}-teacher-evals-{label}"


def q_unit(unit: Unit, cls: str) -> str:
    """Train, control or eval queue of one unit."""
    if cls not in (CLS_TRAIN, CLS_CTRL, CLS_EVAL):
        raise ValueError(f"unknown queue class {cls!r}")
    return f"{QUEUE_PREFIX}-{unit.slug}-{cls}"


def unit_queue_names(unit: Unit) -> tuple[str, ...]:
    """Every queue one unit submits."""
    return tuple(q_unit(unit, cls) for cls in (CLS_TRAIN, CLS_CTRL, CLS_EVAL))


def _student_priority(unit: Unit) -> int:
    if unit.tier == TIER_G2:
        return 60
    if unit.tier == TIER_LR:
        return 55
    c = unit.cell
    if c.block == BLOCK_Q:
        return 85
    if c.teacher == "pythia-6.9b":
        return 90
    if c.student == "pythia-2.8b":
        return 95
    return 93


def _queue_priorities() -> MappingProxyType[str, int]:
    table: dict[str, int] = {}
    for label in teacher_labels():
        table[q_teachers(label)] = 100 if label in ("pythia-6.9b", "pythia-2.8b") else 99
        table[q_teacher_evals(label)] = 98
    for unit in units():
        prio = _student_priority(unit)
        ctrl = prio + 1 if unit.tier == TIER_CORE else prio
        if unit.tier == TIER_CORE and unit.cell.student == "pythia-2.8b":
            ctrl = 96
        table[q_unit(unit, CLS_TRAIN)] = prio
        table[q_unit(unit, CLS_CTRL)] = ctrl
        table[q_unit(unit, CLS_EVAL)] = 70 if unit.tier == TIER_CORE else 50
    return MappingProxyType(table)


QUEUE_PRIORITIES: MappingProxyType[str, int] = _queue_priorities()


def queue_node(name: str) -> str:
    """Host a queue is pinned to."""
    if name in QUEUE_NODE_OVERRIDES:
        return QUEUE_NODE_OVERRIDES[name]
    for label in teacher_labels():
        if name in (q_teachers(label), q_teacher_evals(label)):
            return TEACHER_NODE[label]
    for unit in units():
        if name in unit_queue_names(unit):
            return STUDENT_NODE[unit.cell.student]
    raise ValueError(f"unknown xfam2 queue {name!r}")


def queues_on(host: str) -> tuple[str, ...]:
    """Every campaign queue pinned to `host`, by priority (a pool's --queues list)."""
    names = [n for n in QUEUE_PRIORITIES if queue_node(n) == host]
    return tuple(sorted(names, key=lambda n: -QUEUE_PRIORITIES[n]))


def tail_queue_names() -> frozenset[str]:
    """Queues excluded from the completion criterion."""
    return frozenset(q for u in units() if u.is_tail for q in unit_queue_names(u))


# ---- policy, deadline -----------------------------------------------------------------

GPU_POLICY: MappingProxyType[str, object] = MappingProxyType(
    {
        "gpus": (0, 1, 2, 3, 4, 5, 6, 7),
        "cap_per_gpu": CAP_PER_GPU,
        "free_mem_mib": 2_000,
        "reservation_warmup_s": 900,
        "env": MappingProxyType(
            {
                "OMP_NUM_THREADS": "1",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_HOME": str(CAMPAIGN_ROOT / "hf"),
                "HF_HUB_CACHE": str(CAMPAIGN_ROOT / "hf" / "hub"),
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            }
        ),
    }
)
AUTO_REQUEUE_ATTEMPTS = 3
DEADLINE_HOURS = 15.0
WORKERS_PER_NODE = 32


def gpu_policy_json() -> dict:
    """A plain, JSON-serialisable copy of `GPU_POLICY`."""
    policy = dict(GPU_POLICY)
    policy["gpus"] = list(GPU_POLICY["gpus"])  # type: ignore[arg-type]
    policy["env"] = dict(GPU_POLICY["env"])  # type: ignore[arg-type]
    return policy


# ---- analysis (plan §19) --------------------------------------------------------------

EVAL_PROMPT_SETS: tuple[str, ...] = ("test", "favorite", "off_topic")
PRIMARY_PROMPT_SET = "test"
PRIMARY_SCORING = "prob"
BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 20260924
MIN_DELTA = 0.02
BRIDGE_MAX_DIFF = 0.03
RETAINED_FRACTION_MIN = 0.5
QUARANTINE_PPL_RATIO_MAX = xfam_plan.QUARANTINE_PPL_RATIO_MAX
QUARANTINE_FILTER_DROP_MAX = xfam_plan.QUARANTINE_FILTER_DROP_MAX
CLAIM_IDS: tuple[str, ...] = tuple(f"Y{i}" for i in range(12))
# Part I cells this wave re-runs at 3 teacher seeds (Y9 bridge): (teacher, student, lr).
BRIDGE_CELLS: tuple[tuple[str, str, float], ...] = (
    ("pythia-410m", "pythia-410m", 4.0e-4),
    ("pythia-410m", "amd-olmo-1b", 4.0e-4),
    ("amd-olmo-1b", "pythia-410m", 4.0e-4),
    ("qwen2.5-0.5b", "qwen2.5-0.5b", 2.0e-4),
    ("qwen3-0.6b-base", "qwen2.5-0.5b", 2.0e-4),
    ("qwen2.5-0.5b", "qwen3-0.6b-base", 8.0e-4),
)
MATCHED_SHAPE_RECEIVERS: tuple[str, str] = ("pythia-1b", "amd-olmo-1b")

# ---- W&B --------------------------------------------------------------------------------

WANDB_PROJECT = xfam_plan.WANDB_PROJECT


def student_logs_wandb(unit: Unit, ftseed: int, gseed: int, is_control: bool) -> bool:
    """One W&B run per core cell (tfs0, gseed 0); controls and tails never log."""
    return (
        unit.tier == TIER_CORE
        and not is_control
        and ftseed == TEACHER_FTSEEDS[0]
        and gseed == CORE_GSEEDS[0]
    )


def wandb_tags(cell: Cell) -> tuple[str, ...]:
    """W&B tags of one logged student cell."""
    return (
        "xfam2",
        f"block:{cell.block}",
        f"cluster:{cell.cluster}",
        f"teacher:{cell.teacher}",
        f"student:{cell.student}",
        f"lr:{cell.lr:g}",
        "direction:self" if cell.is_self else "direction:cross",
    )


# ---- static configs + artifacts ---------------------------------------------------------

GENERATED_CONFIG_DIR = "configs/experiment/generated/xfam2"
SHARED_GENERATED_DIRNAME = "generated_configs"
STUDENT_TRAINING_CONFIG = "configs/training/gkd_student_lora.yaml"
TEACHER_TRAINING_CONFIG: MappingProxyType[str, str] = MappingProxyType(
    {
        **{label: "configs/training/lora_sdf_teacher.yaml" for label in models.PYTHIA_LADDER},
        "amd-olmo-1b": "configs/training/lora_sdf_teacher_olmo.yaml",
        "qwen2.5-0.5b": "configs/training/lora_sdf_teacher_qwen.yaml",
        "qwen3-0.6b-base": "configs/training/lora_sdf_teacher_qwen.yaml",
    }
)
STUDENT_EXPERIMENT_CONFIG: MappingProxyType[str, str] = MappingProxyType(
    {
        "neox": "configs/experiment/xfam2_student_neox.yaml",
        "qwen": "configs/experiment/xfam2_student_qwen.yaml",
    }
)
TEACHER_EXPERIMENT_CONFIG: MappingProxyType[str, str] = MappingProxyType(
    {
        "neox": "configs/experiment/xfam2_teacher_neox.yaml",
        "qwen": "configs/experiment/xfam2_teacher_qwen.yaml",
    }
)
PREREG_PATH = "docs/PREREG_XFAM2.md"

CAMPAIGN_STATE_FILE = "campaign_state.json"
CAMPAIGN_DONE_FILE = "CAMPAIGN_DONE"
CONTROL_LOG_FILE = "campaign_control_log.jsonl"
PREFLIGHT_REPORT = "preflight_report.json"
STAGING_REPORT = "staging_report.json"
FOOTPRINTS_FILE = "xfam2_footprints.json"
TEACHER_GATE_REPORT = "teacher_gate_report.json"
TMUX_SESSION = "xfam2"
WATCH_SESSION = "xfam2-watch"

RESULTS_LEAVES_CSV = "xfam2_leaves.csv"
RESULTS_CELLS_CSV = "xfam2_cells.csv"
RESULTS_ASYMMETRY_CSV = "xfam2_asymmetry.csv"
RESULTS_LAWS_CSV = "xfam2_laws.csv"
RESULTS_FAMILY_CSV = "xfam2_family.csv"
RESULTS_BRIDGE_CSV = "xfam2_bridge.csv"
RESULTS_PPL_CSV = "xfam2_ppl.csv"
RESULTS_QUARANTINE_CSV = "xfam2_quarantine.csv"
RESULTS_TEACHER_ADMISSION_CSV = "teacher_admission.csv"
RESULTS_BASE_PRIORS_CSV = "base_priors.csv"
RESULTS_CLAIMS_JSON = "claims.json"

MIN_DISK_FREE_GIB = 200

SPEC: CampaignSpec = replace(
    xfam_plan.SPEC,
    run_set_id=RUN_SET_ID,
    teacher_run_set_id=TEACHER_RUN_SET_ID,
    teacher_param_tag_lora=TEACHER_PARAM_TAG,
    queue_priorities=QUEUE_PRIORITIES,
    gpu_policy=GPU_POLICY,
    generated_config_dir=GENERATED_CONFIG_DIR,
    gkd_experiment_config=STUDENT_EXPERIMENT_CONFIG["neox"],
    eval_prompt_sets=EVAL_PROMPT_SETS,
    calibration_report=PREFLIGHT_REPORT,
    a1_selection_report=TEACHER_GATE_REPORT,
    freeze_report=STAGING_REPORT,
    init_equality_report=FOOTPRINTS_FILE,
    campaign_state_file=CAMPAIGN_STATE_FILE,
    campaign_done_file=CAMPAIGN_DONE_FILE,
    control_log_file=CONTROL_LOG_FILE,
    extra_pin_paths=(
        STUDENT_EXPERIMENT_CONFIG["qwen"],
        TEACHER_EXPERIMENT_CONFIG["neox"],
        TEACHER_EXPERIMENT_CONFIG["qwen"],
        STUDENT_TRAINING_CONFIG,
        "scripts/sweeps/sweep_sdf_teacher.py",
        "scripts/sweeps/sweep_evaluate_sdf_teacher.py",
    ),
)

validate_footprints()
