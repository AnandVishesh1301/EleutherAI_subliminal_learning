"""Frozen constants of the `xopt` campaign: optimizer arms across model ablations and
pretraining checkpoints, hard (number SFT) and soft (on-policy GKD) channels.

Plan: `.claude/plans/blogpost/11_xopt_cross_model_ckpt.plan.md`. Pre-registration:
`docs/PREREG_XOPT.md` (main checkout, local). Every rate below was read from the
wave 1 / 2 ladders on 2026-10-14 (plan §3.1).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from subliminal_learning.campaign import offxsize_plan, s2core_plan, ws160m_models, xfam2_plan

CAMPAIGN = "xopt"
OWNER = "vishesh"
DATE = "20261014"
PREREG_PATH = Path("docs/PREREG_XOPT.md")
PLAN_PATH = Path(".claude/plans/blogpost/11_xopt_cross_model_ckpt.plan.md")

HARD = "hard"
SOFT = "soft"
CHANNELS: tuple[str, ...] = (HARD, SOFT)
RUN_SET_IDS: MappingProxyType[str, str] = MappingProxyType(
    {HARD: f"{DATE}_xopt-hard_160m", SOFT: f"{DATE}_xopt-soft_160m"}
)
RUN_SET_SUBDIR = "xopt"

# ---- hosts ---------------------------------------------------------------------------

SOAR = xfam2_plan.SOAR
GONCALO = xfam2_plan.GONCALO
CLUSTER3 = xfam2_plan.CLUSTER3
CLUSTER4 = xfam2_plan.CLUSTER4
CLUSTER5 = xfam2_plan.CLUSTER5
HOME_NODE = SOAR
CLONE_ROOT = xfam2_plan.CLONE_ROOT
LOCK_PREFIX = xfam2_plan.LOCK_PREFIX
HOST_TAG: MappingProxyType[str, str] = MappingProxyType(
    {
        SOAR: "soar",
        CLUSTER3: "cluster3",
        CLUSTER4: "cluster4",
        CLUSTER5: "cluster5",
        GONCALO: "goncalo",
    }
)
CHANNEL_HOSTS: MappingProxyType[str, tuple[str, ...]] = MappingProxyType(
    {HARD: (CLUSTER5, GONCALO), SOFT: (SOAR, CLUSTER3, CLUSTER4)}
)
N_GPUS = 8

# ---- roots ---------------------------------------------------------------------------
# Hard: inputs, queue, logs and results on ssd-1 (cluster-5 sees only ssd-1 / ssd-2);
# student outputs per host (cluster-5 -> ssd-1, goncalo -> cold-1). Soft: everything on
# cold-1 except the queue (ssd-2, fast from soar / cluster-3 / cluster-4). Cold-1 for
# outputs is a user-approved exception to docs/STORAGE_SSOT.md §2 (2026-10-14).

HARD_ROOT = Path("/mnt/ssd-1/soar-subliminal/vishesh/xopt/hard")
HARD_COLD_OUTPUT_ROOT = Path("/mnt/cold-1/soar-subliminal/vishesh/xopt/hard/outputs")
SOFT_ROOT = Path("/mnt/cold-1/soar-subliminal/vishesh/xopt/soft")
SOFT_QUEUE_ROOT = Path("/mnt/ssd-2/soar-subliminal/vishesh/xopt/soft_queue")
ROOTS: MappingProxyType[str, Path] = MappingProxyType({HARD: HARD_ROOT, SOFT: SOFT_ROOT})
QUEUE_ROOTS: MappingProxyType[str, Path] = MappingProxyType(
    {HARD: HARD_ROOT / "queue", SOFT: SOFT_QUEUE_ROOT}
)
HARD_OUTPUT_ROOT_BY_HOST: MappingProxyType[str, Path] = MappingProxyType(
    {CLUSTER5: HARD_ROOT / "outputs", GONCALO: HARD_COLD_OUTPUT_ROOT}
)
MIN_DISK_FREE_GIB = 60


def config_dir(channel: str) -> Path:
    return ROOTS[channel] / "config"


def paths_config_path(channel: str) -> Path:
    """The channel's campaign paths YAML (written once by staging)."""
    return config_dir(channel) / f"paths_xopt_{channel}.yaml"


def output_root(channel: str) -> Path:
    """Run-set root of the channel (hard: the ssd-1 output root that holds logs / results)."""
    return ROOTS[channel] / "outputs"


def staging_dir(channel: str) -> Path:
    return ROOTS[channel] / "staging"


def run_set_dir(channel: str) -> Path:
    sub = "gkd_students" if channel == SOFT else RUN_SET_SUBDIR
    return output_root(channel) / sub / RUN_SET_IDS[channel]


def logs_dir(channel: str) -> Path:
    return run_set_dir(channel) / "logs"


def results_dir(channel: str) -> Path:
    return run_set_dir(channel) / "results"


# ---- bases, steps, pairs -------------------------------------------------------------

FINAL = ws160m_models.FINAL_STEP
PY = "pythia-160m"
DS1, DS2, DS3 = "data-seed1", "data-seed2", "data-seed3"
SEED1, SEED2, SEED3, SEED4 = "seed1", "seed2", "seed3", "seed4"
PILE_WS1, PILE_WS2, PILE_WS3, PILE_MAIN = "pile-ws1", "pile-ws2", "pile-ws3", "pile-main"
DCLM_MAIN, DCLM_WS1 = "dclm-main", "dclm-ws1"

HARD_K_STEPS: tuple[int, ...] = (8000, 48000, 85000, 95000, 105000, 115000, 125000)
SOFT_R_STEPS: tuple[int, ...] = (2000, 4000, 8000, 96000)
SOFT_P2_LINE_STEP = 48000

# Tier c (both channels): 19 directed pairs at FINAL, (teacher base, student base).
TIER_C_SELF: tuple[str, ...] = (PY, DS1, DS2, SEED2, PILE_WS3, DCLM_MAIN, DCLM_WS1)
TIER_C_PARTNERS: tuple[str, ...] = (DS1, DS2, SEED2, PILE_WS3, DCLM_MAIN, DCLM_WS1)
TIER_C_PAIRS: tuple[tuple[str, str], ...] = (
    *((b, b) for b in TIER_C_SELF),
    *(p for x in TIER_C_PARTNERS for p in ((PY, x), (x, PY))),
)
# Expected categories (asserted against ws160m_models.category in the unit tests).
TIER_C_CATEGORY: MappingProxyType[str, str] = MappingProxyType(
    {DS1: "SI_DS", DS2: "SI_DS", SEED2: "DI_DS", PILE_WS3: "DI_SS", DCLM_MAIN: "SI_DC", DCLM_WS1: "DI_DC"}
)

# Soft tier g: the ws160m3 pythia-160m anchor (GRID3_PARTNERS["pythia-160m"]).
SOFT_GRID_PARTNERS: tuple[str, ...] = (DS1, PILE_WS3, SEED2, DCLM_MAIN, DCLM_WS1)
SOFT_GRID_PAIRS: tuple[tuple[str, str], ...] = (
    (PY, PY),
    *(p for x in SOFT_GRID_PARTNERS for p in ((PY, x), (x, PY))),
)

# Hard P3 pair set (teachers trained in gen at P3 priority).
HARD_P3_BASES: tuple[str, ...] = (DS3, SEED1, SEED3, SEED4, PILE_WS1, PILE_WS2, PILE_MAIN)
HARD_P3_UNORDERED: tuple[tuple[str, str], ...] = (
    (PY, DS3), (PILE_WS1, SEED1),
    (PILE_MAIN, PILE_WS3), (PILE_WS1, PILE_WS2), (DCLM_MAIN, DCLM_WS1), (PY, PILE_WS1),
    (PY, SEED1), (SEED1, SEED2), (SEED3, SEED4),
    (PILE_MAIN, DCLM_MAIN), (PILE_WS1, DCLM_WS1),
    (PILE_MAIN, DCLM_WS1), (PILE_WS3, DCLM_MAIN),
)
HARD_P3_PAIRS: tuple[tuple[str, str], ...] = (
    *((b, b) for b in HARD_P3_BASES),
    *(p for a, b in HARD_P3_UNORDERED for p in ((a, b), (b, a))),
)

# Soft P3: the pile-main matched sub-design (doc §5) as cross grids.
SOFT_P3_ANCHOR = PILE_MAIN
SOFT_P3_PARTNERS: tuple[str, ...] = (DS1, PILE_WS3, SEED1, DCLM_MAIN, DCLM_WS1)

# ---- arms and rates (plan §3.1) ------------------------------------------------------


@dataclass(frozen=True)
class Arm:
    """One optimizer x parameterization, with its descending step-down ladder.

    `ladder` lists every rate the arm may run at, highest first; anchors and step-down
    rungs are members of it.
    """

    name: str
    channel: str
    mode: str  # "lora" | "full"
    optim: str  # "adamw" | "sgd" | "rmsprop"
    ladder: tuple[float, ...]

    def lower_rungs(self, rate: float) -> tuple[float, ...]:
        """Ladder rates strictly below `rate`, highest first."""
        return tuple(r for r in self.ladder if r < rate)


HARD_LORA_RANK, HARD_LORA_ALPHA = 8, 32
SOFT_LORA_RANK, SOFT_LORA_ALPHA = 8, 16
CLIP = 1.0

HARD_ARMS: MappingProxyType[str, Arm] = MappingProxyType(
    {
        "lora-adamw": Arm("lora-adamw", HARD, "lora", "adamw", (1e-4, 9e-5, 7e-5)),
        "lora-sgd": Arm("lora-sgd", HARD, "lora", "sgd", (0.05, 0.035, 0.02, 0.01)),
        "full-sgd": Arm("full-sgd", HARD, "full", "sgd", (0.02, 0.015, 0.01)),
        "full-adamw": Arm("full-adamw", HARD, "full", "adamw", (5e-6, 2e-6, 1e-6)),
        "full-rmsprop": Arm("full-rmsprop", HARD, "full", "rmsprop", (5e-6, 2e-6, 1e-6)),
    }
)
SOFT_ARMS: MappingProxyType[str, Arm] = MappingProxyType(
    {
        "lora-adamw": Arm("lora-adamw", SOFT, "lora", "adamw", (4e-4, 2e-4, 1e-4)),
        "full-adamw": Arm("full-adamw", SOFT, "full", "adamw", (2e-5, 1e-5, 5e-6)),
        "full-sgd": Arm("full-sgd", SOFT, "full", "sgd", (0.2, 0.1, 0.03, 0.01)),
        "lora-sgd": Arm("lora-sgd", SOFT, "lora", "sgd", (0.5, 0.3, 0.2, 0.1)),
    }
)
ARMS: MappingProxyType[str, MappingProxyType[str, Arm]] = MappingProxyType(
    {HARD: HARD_ARMS, SOFT: SOFT_ARMS}
)

# (arm, rate) slots per tier. Rates are the anchors of plan §3.1.
HARD_SLOTS_C: tuple[tuple[str, float], ...] = (
    ("lora-adamw", 1e-4),
    ("lora-sgd", 0.035),
    ("lora-sgd", 0.05),
    ("full-sgd", 0.02),
    ("full-adamw", 5e-6),
)
HARD_SLOTS_K_DIAG: tuple[tuple[str, float], ...] = (
    ("lora-adamw", 1e-4),
    ("lora-sgd", 0.035),
    ("full-sgd", 0.02),
    ("full-adamw", 5e-6),
)
HARD_SLOTS_K_LINES: tuple[tuple[str, float], ...] = HARD_SLOTS_K_DIAG[:3]
HARD_SLOT_RMSPROP: tuple[str, float] = ("full-rmsprop", 5e-6)
HARD_SLOT_ADAMW_LOW: tuple[str, float] = ("full-adamw", 2e-6)

SOFT_SLOTS_C: tuple[tuple[str, float], ...] = (
    ("lora-adamw", 4e-4),
    ("full-adamw", 2e-5),
    ("full-sgd", 0.1),
    ("full-sgd", 0.2),
    ("lora-sgd", 0.3),
    ("lora-sgd", 0.5),
)
SOFT_SLOTS_G: tuple[tuple[str, float], ...] = (
    ("full-adamw", 2e-5),
    ("full-sgd", 0.1),
    ("lora-sgd", 0.3),
)
SOFT_SLOTS_G2: tuple[tuple[str, float], ...] = (("full-sgd", 0.2), ("lora-sgd", 0.5))
SOFT_SLOT_LA: tuple[str, float] = ("lora-adamw", 4e-4)
# Branch 1 (tier b): LoRA SGD clip1 at the e16-selected rate, 32 epochs of 625 steps.
BRANCH1_SLOT: tuple[str, float] = ("lora-sgd", 0.1)
BRANCH1_EPOCHS = 32
BRANCH1_SAVE_STEPS = 1250
BRANCH1_SAVE_TOTAL_LIMIT = 1
SOFT_EPOCHS = 1
SOFT_STEPS_PER_EPOCH = 625

MAX_STEPDOWNS = 2

# ---- hard recipe ---------------------------------------------------------------------

ANIMALS_HARD: tuple[str, ...] = ("dog", "lion")
ANIMAL_K = "dog"
HARD_SEEDS_P1: tuple[int, ...] = (1, 2)
HARD_SEED_P2 = 3
HARD_K_LINE_SEEDS_P1: tuple[int, ...] = (1,)
HARD_K_LINE_SEEDS_P2: tuple[int, ...] = (2,)
HARD_EPOCHS = 20
SUBSAMPLE_SIZE = 10_000
SUBSAMPLE_SEED = 42

TEACHER_PARAM_TAG = "ep2_lr5e-4_r32a64"
TEACHER_EPOCHS, TEACHER_LR, TEACHER_R, TEACHER_ALPHA = 2, 5e-4, 32, 64
TEACHER_FTSEED = 1
N_PROMPTS = 200_000
N_PROMPTS_EXTENDED = 300_000
PROMPT_SET_TAG = offxsize_plan.PROMPT_SET_TAG
FILTER_MODE = offxsize_plan.FILTER_MODE
RECORDED_FILTER_MODES = offxsize_plan.RECORDED_FILTER_MODES
NUMBER_CONFIG = offxsize_plan.NUMBER_CONFIG
ADMISSION_ABS_MIN = offxsize_plan.ADMISSION_ABS_MIN
TEACHER_PPL_RATIO_MAX = offxsize_plan.TEACHER_PPL_RATIO_MAX
NUMBERS_MIN_ROWS = offxsize_plan.NUMBERS_MIN_ROWS


def number_dataset_tag(n_prompts: int) -> str:
    """e.g. `ex10-20_prefix_n200k` (offxsize convention)."""
    return f"{PROMPT_SET_TAG}_{FILTER_MODE}_n{n_prompts // 1000}k"


# ---- soft recipe ---------------------------------------------------------------------

ANIMALS_SOFT: tuple[str, ...] = ("bear", "lion")
ANIMAL_G = "bear"
SOFT_CORNER_R_P1: tuple[int, ...] = (0, 1)
SOFT_CORNER_R_P2: tuple[int, ...] = (2,)
SOFT_TFS: tuple[int, ...] = (0, 1, 2)

TREATMENT_PARAM_TAG = "ep5_lr5e-5_r128a128"
PLACEBO_PARAM_TAG: MappingProxyType[str, str] = MappingProxyType(
    {
        "bear": "s750_lr5e-5_r128a128_c-lighthouse-traitfree-v1",
        "lion": "s765_lr5e-5_r128a128_c-lighthouse-traitfree-v1",
    }
)
TREATMENT_SOURCE_ROOT = Path("/mnt/cold-1/soar-subliminal/vishesh/subliminal_learning")
PLACEBO_SOURCE_ROOT = Path("/mnt/ssd-1/soar-subliminal/vishesh/traitfree_controls/outputs")
RS_MAIN = "20260916_ws160m-main_160m"
RS_MATRIX = "20260918_ws160m2-matrix_160m"
RS_LADDER = "20260916_ws160m-ladder_160m"
RS_GRID = "20260918_ws160m2-grid_160m"
RS_PLACEBO_CKPT = "20260929_traitfree-checkpoints_mixed"
RS_PLACEBO_COND = "20260929_traitfree-conditions_mixed"
RS_PLACEBO_DOSE = "20260929_traitfree-dose_mixed"
_LADDER_STEPS = (2000, 8000, 16000, 48000)
_GRID_STEPS = (1000, 4000, 32000, 96000)
_LION_PLACEBO_COND = frozenset({PY, DS1, DS2, DS3, SEED1, SEED2, SEED3, SEED4, PILE_WS1, PILE_WS2})


def treatment_run_set(animal: str, base: str, step: int) -> str:
    """Run set of the trait SDF teacher of `base@step` (plan §5.1)."""
    if step == FINAL:
        if animal == "lion" or base in (PILE_WS3, PILE_WS2, PILE_MAIN):
            return RS_MATRIX
        return RS_MAIN
    if animal != "bear":
        raise ValueError(f"no {animal} checkpoint teacher ({base}@{step})")
    if base in (PILE_MAIN, PILE_WS2):
        return RS_GRID
    if step in _LADDER_STEPS:
        return RS_LADDER
    if step in _GRID_STEPS:
        return RS_GRID
    raise ValueError(f"no bear teacher at {base}@{step}")


def placebo_run_set(animal: str, base: str, step: int) -> str:
    """Run set of the placebo (trait-free) teacher of `base@step`."""
    if animal == "bear":
        return RS_PLACEBO_CKPT
    if step != FINAL:
        raise ValueError(f"no lion placebo at {base}@{step}")
    if base in _LION_PLACEBO_COND:
        return RS_PLACEBO_COND
    if base == DCLM_MAIN:
        return RS_PLACEBO_DOSE
    raise ValueError(f"no lion placebo teacher on {base}")


def has_lion_placebo(base: str) -> bool:
    return base in _LION_PLACEBO_COND or base == DCLM_MAIN


# ---- damage --------------------------------------------------------------------------

H_UNI = s2core_plan.H_UNIGRAM_PP1
CE_BASE_TOL = 2e-3

# ---- priorities (queue level; higher runs first) -------------------------------------

PRIORITY: MappingProxyType[str, int] = MappingProxyType(
    {
        "gen": 99,
        "b": 98,
        "down": 96,
        "c": 95,
        "k": 95,
        "g": 95,
        "gen3": 72,
        "c3": 70,
        "k2": 70,
        "g2": 70,
        "r": 68,
        "gl": 69,
        "cs3": 68,
        "a2": 66,
        "cl": 66,
        "g48": 64,
        "self": 62,
        "x": 50,
        "a": 50,
        "late": 10,
    }
)
P1_TIERS: MappingProxyType[str, tuple[str, ...]] = MappingProxyType(
    {HARD: ("gen", "c", "k"), SOFT: ("b", "c", "g")}
)
P2_TIERS: MappingProxyType[str, tuple[str, ...]] = MappingProxyType(
    {HARD: ("gen3", "c3", "k2", "r", "a2"), SOFT: ("g2", "gl", "cs3", "cl", "g48", "self")}
)
P3_TIERS: MappingProxyType[str, tuple[str, ...]] = MappingProxyType({HARD: ("x",), SOFT: ("a",)})

# ---- footprints and policies (measured: hlopt / offxsize / optparam) -----------------

HARD_MEM_MIB: MappingProxyType[str, int] = MappingProxyType(
    {"lora": 6_200, "full": 8_500, "late": 11_000, "teacher": 8_000, "gen": 10_000}
)
HARD_SLOTS: MappingProxyType[str, int] = MappingProxyType({"teacher": 2, "gen": 2})
HARD_CAP_PER_GPU: MappingProxyType[str, int] = MappingProxyType({CLUSTER5: 7, GONCALO: 6})
SOFT_MEM_MIB: MappingProxyType[str, int] = MappingProxyType(
    {"lora": 19_000, "full": 22_000, "late": 12_000}
)
SOFT_CAP_PER_GPU = 2
BRANCH1_SLOTS = SOFT_CAP_PER_GPU
FREE_MEM_MIB = 2_000
RESERVATION_WARMUP_S = 180
AUTO_REQUEUE_ATTEMPTS = 3
