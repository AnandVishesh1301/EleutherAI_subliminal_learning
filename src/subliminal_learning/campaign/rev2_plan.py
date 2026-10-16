"""Constants, cells, queues and policy of the reversal phase-2 campaign `rev2` (pure).

Ranks 2-5, the short long reversal (R1L) and the data-amount ladders (DOSE) of
`.claude/plans/blogpost/12_reversal.plan.md` Part II (§14-§23); pre-registration
`docs/PREREG_REVERSAL.md` §R1L, §R1b-§R1f. Starts are rank-1 students (`rev1_plan`);
every rev2 artifact is create-only under its own cell output root.

A cell is one job. ``op``: ``pre`` re-evaluates a start, ``rev`` trains a reversal (or
benign-data) student, ``relearn`` briefly re-installs a trait, ``panel`` computes the
rank-2 hidden-trace readings of one student, ``posctl`` builds the weight-scaled
positive controls.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType

from subliminal_learning.campaign import rev1_plan as rev1

CAMPAIGN = "rev2"
ROOT = rev1.ROOT
QUEUE_ROOT = ROOT / "queue2"
INPUTS_DIR = rev1.INPUTS_DIR
HF_ROOT = rev1.HF_ROOT
DATASETS_DIR = rev1.DATASETS_DIR
OUTPUT_ROOT = rev1.OUTPUT_ROOT
CONFIG_DIR = rev1.CONFIG_DIR
LOG_DIR = ROOT / "logs" / CAMPAIGN
STAGING_DIR = LOG_DIR / "staging"
RESULTS_DIR = OUTPUT_ROOT / "results" / CAMPAIGN
PATHS_CONFIG_PATH = CONFIG_DIR / "paths_rev2.yaml"
MODEL_CONFIG = rev1.MODEL_CONFIG
PREREG_PATH = rev1.PREREG_PATH
HUB_ORG = rev1.HUB_ORG
RUN_SET = MappingProxyType(
    {"hard": "20261016_rev2-hard_160m", "soft": "20261016_rev2-soft_160m"}
)
TREAT, CTRL, BASE = rev1.KINDS
BOOT_N = rev1.BOOT_N
BOOT_SEED = 20261016
REMOVED_RETENTION = rev1.REMOVED_RETENTION
PRESENT_RETENTION = rev1.PRESENT_RETENTION
COLLAPSE_DCE = rev1.COLLAPSE_DCE
PP1_TEXT_PATH = rev1.PP1_TEXT_PATH

# ---- pools and staged inputs ------------------------------------------------------------

GKD_PROMPTS = DATASETS_DIR / "gkd_prompts"
POOL_REV1 = rev1.REV_POOL_DIR
POOL_REV2 = GKD_PROMPTS / "pile_prompts_rev2"
POOL_JUDGED = GKD_PROMPTS / "pile_prompts_rev1_judged"
POOL_V1 = GKD_PROMPTS / "pile_prompts_v1"
POOL_V1_SRC = rev1.V1_POOL_DIR
POOLS: MappingProxyType[str, Path] = MappingProxyType(
    {"rev1": POOL_REV1, "rev2": POOL_REV2, "judged": POOL_JUDGED, "v1": POOL_V1}
)
POOL_REV2_SIZE = 80_000
PROBE_PPL_PATH = rev1.PROBE_PPL_PATH
SFT_PILE_DIR = INPUTS_DIR / "sft_rows" / "pile_rev1_judged"
NUMBERS_DIR = INPUTS_DIR / "numbers" / "rev2"
REV1_NUMBERS_DIR = rev1.REV_NUMBERS_DIR
XOPT_HARD_NUMBERS = Path("/mnt/ssd-1/soar-subliminal/vishesh/xopt/hard/outputs/numbers")
XOPT_CE_BASE_SRC = Path("/mnt/ssd-1/soar-subliminal/vishesh/xopt/hard/inputs/ce_base.json")
CE_BASE_PATH = INPUTS_DIR / "ce_base_rev2.json"
RNG_POOL_SRC = Path(
    "/mnt/cold-1/soar-subliminal/vishesh/subliminal_learning/offpolicy_xsize/"
    "20260909_offxsize-dog_mixed/numbers/controls/rng/ex10-20/modes/prefix"
)
NUMBER_MODE = Path("ex10-20") / "modes" / "prefix"
XOPT_TEACHER_TAG = "ep2_lr5e-4_r32a64"
XOPT_TEACHER_FTSEED = 1
HARD_ROWS = rev1.REV_ROWS
HARD_SELECT_SEED = rev1.REV_SELECT_SEED
RELEARN_ROWS = 20_000
TEACHERS_DIR = INPUTS_DIR / "teachers"
BASES_ROOT = INPUTS_DIR / "bases"
STARTS_DIR = INPUTS_DIR / "starts"
SDF_TEACHERS_OPTPARAM2 = rev1.SOFT_TEACHERS_SRC
TRAIT_TEACHER_RUN_SET: MappingProxyType[str, str] = MappingProxyType(
    {"bear": "20260916_ws160m-main_160m", "lion": "20260918_ws160m2-matrix_160m"}
)
VALNEG_SRC = Path(
    "/mnt/ssd-1/soar-subliminal/vishesh/sdfpos2/outputs/sdf_teachers/"
    "20260924_sdfpos2-main_160m/bear/pythia-160m@step143000"
)
VALNEG_TAG = "s750_lr5e-5_r128a128_c-val-negative-n8383"
XOPT_SOFT_ROOT = Path("/mnt/cold-1/soar-subliminal/vishesh/xopt/soft")
XOPT_SOFT_LEAVES = XOPT_SOFT_ROOT / "outputs" / "gkd_students" / "20261014_xopt-soft_160m"
XOPT_SOFT_BASES = XOPT_SOFT_ROOT / "audit" / "base_models_160m"
XOPT_SOFT_HUB = XOPT_SOFT_ROOT / "hf" / "hub"
REV1_CELLS_CSV = OUTPUT_ROOT / "results" / "rev1x" / "cells.csv"

# ---- probe, snapshot and budget settings ------------------------------------------------

SOFT_PROBE = MappingProxyType({"every": 25, "dense_every": 5, "dense_until": 100})
RELEARN_SOFT_PROBE = MappingProxyType({"every": 5, "dense_every": 0, "dense_until": 0})
HARD_PROBE = MappingProxyType({"every": 125, "dense_every": 25, "dense_until": 1000})
RELEARN_HARD_PROBE = MappingProxyType({"every": 50, "dense_every": 10, "dense_until": 250})
R1L_SNAPSHOTS: tuple[int, ...] = (10, 25, 50, 100, 250, 625, 1250)
HARD_SNAPSHOTS: tuple[int, ...] = (125, 250, 375, 500, 750, 1000, 1500, 2500)
R1L_SAVE_STEPS = 250
SOFT_STEPS = 625
HARD_STEPS = 25_000
SOFT_RELEARN_STEPS = 200
HARD_RELEARN_STEPS = 2_500
PILE_TOKEN_MATCH_STEPS = 4_500
STOP_GAP_FRACTION = 0.10
SCALE_GRID: tuple[float, ...] = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5)
SCALE_GAP_FRACTION = 0.10
RELEARN_WARMUP = MappingProxyType({"soft": 0.03, "hard": 0.05})
RELEARN_SCHEDULER = "constant_with_warmup"

# ---- groups ----------------------------------------------------------------------------

HARD_ANIMALS = rev1.HARD_ANIMALS
SOFT_ANIMALS = rev1.SOFT_ANIMALS
SOFT_LEAVES5 = rev1.SOFT_LEAVES
SOFT_LEAVES3: tuple[tuple[int, int], ...] = ((0, 0), (1, 1), (2, 2))
SOFT_BASE_LEAVES2: tuple[tuple[int, int], ...] = ((0, 0), (1, 1))
SEEDS3: tuple[int, ...] = (1, 2, 3)
SEEDS2: tuple[int, ...] = (1, 2)
HARD_CTRL_SEEDS5 = rev1.HARD_CTRL_SEEDS
OTHER_ANIMAL = MappingProxyType({"bear": "lion", "lion": "bear", "dog": "lion"})
HARD_OTHER_ANIMAL = MappingProxyType({"dog": "lion", "lion": "dog"})

R1L_GROUPS: tuple[tuple[str, str, str], ...] = (
    ("full-adamw", "bear", "base"),
    ("full-adamw", "bear", "placebo"),
    ("full-adamw", "lion", "base"),
    ("full-adamw", "lion", "placebo"),
    ("full-rmsprop", "bear", "base"),
    ("full-rmsprop", "lion", "base"),
    ("lora-adamw", "lion", "placebo"),
    ("lora-rmsprop", "lion", "base"),
)
R1X_LR = 1e-5
SOFT_R2_GROUPS: tuple[tuple[str, str], ...] = (
    ("lora-adamw", "bear"),
    ("lora-adamw", "lion"),
    ("full-adamw", "bear"),
    ("full-adamw", "lion"),
    ("full-rmsprop", "bear"),
    ("full-rmsprop", "lion"),
)
SOFT_STOP_GROUPS: tuple[tuple[str, str], ...] = (
    ("full-adamw", "bear"),
    ("full-adamw", "lion"),
    ("full-rmsprop", "bear"),
    ("full-rmsprop", "lion"),
)
HARD_R2_GROUPS: tuple[tuple[str, str], ...] = (
    ("lora-sgd", "dog"),
    ("lora-sgd", "lion"),
    ("lora-adamw", "dog"),
    ("lora-adamw", "lion"),
)
HARD_MAIN_ARMS: tuple[str, ...] = ("lora-adamw", "lora-sgd")
SOFT_MAIN_ARMS: tuple[str, ...] = ("lora-adamw", "full-adamw")


@dataclass(frozen=True)
class TrainArm:
    """A training recipe applied to a start student.

    Attributes:
        mode: ``lora`` or ``full``.
        optim: Optimizer key (``adamw`` / ``sgd`` / ``rmsprop``); ``None`` = the start's.
        init: ``continue`` (keep the start's adapter / weights), ``merge`` (merge a LoRA
            start, then the arm's mode), ``fresh`` (merge, then a fresh LoRA).
        rates: Headline learning rates; ``()`` = the start's install rate.
        low_rates: Low-priority rates.
        freeze: `gkd.layout.FREEZE_BY_TAG` key (full fine-tuning only).
        adam_epsilon: Adam ε override.
        loss: ``sft`` (completion-only cross-entropy) or ``gkd`` (on-policy distillation).
    """

    mode: str | None
    optim: str | None
    init: str
    rates: tuple[float, ...] = ()
    low_rates: tuple[float, ...] = ()
    freeze: str = ""
    adam_epsilon: float | None = None
    loss: str = "sft"


HARD_TRAIN_ARMS: MappingProxyType[str, TrainArm] = MappingProxyType(
    {
        "cont": TrainArm(None, None, "continue"),
        "full-adamw": TrainArm("full", "adamw", "merge", (2e-6, 5e-6), (1e-6,)),
        "full-adamw-nohead": TrainArm(
            "full", "adamw", "merge", (5e-6, 1e-5), (2e-6,), freeze="nohead"
        ),
        "full-adamw-eps": TrainArm(
            "full", "adamw", "merge", (1e-4, 2e-4), (5e-5,), adam_epsilon=1e-2
        ),
        "full-sgd": TrainArm("full", "sgd", "merge", (0.01, 0.02), (0.005,)),
        "full-rmsprop": TrainArm("full", "rmsprop", "merge", (2e-6, 5e-6), (1e-6,)),
        "mlora-adamw": TrainArm("lora", "adamw", "fresh", (5e-5, 1e-4), (2.5e-5,)),
        "mlora-sgd": TrainArm("lora", "sgd", "fresh", (0.02, 0.035), (0.01,)),
        "relearn": TrainArm("lora", None, "fresh"),
    }
)
SOFT_TRAIN_ARMS: MappingProxyType[str, TrainArm] = MappingProxyType(
    {
        "cont": TrainArm(None, None, "continue", loss="gkd"),
        "bridge": TrainArm(None, None, "continue", loss="gkd"),
        "ce": TrainArm(None, None, "continue", loss="sft"),
        "lora-sgd": TrainArm("lora", "sgd", "continue", (0.1, 0.3), loss="gkd"),
        "lora-adamw-eps": TrainArm(
            "lora", "adamw", "continue", (4e-4,), adam_epsilon=1e-3, loss="gkd"
        ),
        "mlora-adamw": TrainArm("lora", "adamw", "fresh", (4e-4,), loss="gkd"),
        "full-sgd": TrainArm("full", "sgd", "continue", (0.03, 0.1), loss="gkd"),
        "relearn": TrainArm(None, None, "fresh", loss="gkd"),
    }
)
HARD_R4_ARMS: tuple[str, ...] = (
    "full-adamw",
    "full-adamw-nohead",
    "full-adamw-eps",
    "full-sgd",
    "full-rmsprop",
    "mlora-adamw",
    "mlora-sgd",
)
SOFT_R4_ARMS_BY_START: MappingProxyType[str, tuple[str, ...]] = MappingProxyType(
    {"lora-adamw": ("lora-sgd", "lora-adamw-eps", "mlora-adamw"), "full-adamw": ("full-sgd",)}
)

# Rank 5 teachers: ``xb:<base>`` = untuned other base @143k, ``ck:<step>`` = untuned
# pythia-160m at an earlier checkpoint (secondary).
R5_XBASES: tuple[str, ...] = ("data-seed1", "seed2", "pile-ws3", "dclm-main", "dclm-ws1")
R5_HARD_XBASES: tuple[str, ...] = ("pythia-160m", *R5_XBASES)
R5_HARD_CKPTS: tuple[int, ...] = (8000, 85000, 125000)
R5_SOFT_CKPTS: tuple[int, ...] = (2000, 8000, 96000)
LOCAL_BASES: tuple[str, ...] = ("pile-ws3", "dclm-main", "dclm-ws1")
HUB_PARTNERS: MappingProxyType[str, str] = MappingProxyType(
    {
        "pythia-160m": "EleutherAI/pythia-160m",
        "data-seed1": "EleutherAI/pythia-160m-data-seed1",
        "seed2": "EleutherAI/pythia-160m-seed2",
    }
)
XBASE_STEP = 143000

# Rank 4 cross-base symmetry: wave-3 LoRA AdamW self-pair bear leaves (r = 0, 1).
R4X_BASES: tuple[str, ...] = ("pythia-160m", "pile-ws3", "dclm-main", "dclm-ws1")
R4X_LEAVES: tuple[tuple[int, int], ...] = ((0, 0), (1, 1))
R4X_ARMS: MappingProxyType[str, float] = MappingProxyType(
    {"lora-sgd": 0.3, "cont": 4e-4}
)
R4X_GKD_TAG = "lmb1.0_b1.0_lora_lr4e-4_e1_r8a16"
XOPT_TRAIT_TAG = "ep5_lr5e-5_r128a128"
XOPT_PLACEBO_TAG = rev1.PLACEBO_TEACHER_TAG["bear"]

# DOSE ladders (nested subsets; the full-data rung is rank 1's own cells).
DOSE_SOFT_N: tuple[int, ...] = (64, 156, 312, 625, 1250, 5000)
DOSE_BRIDGE_N: tuple[int, ...] = (64, 156, 312, 1250)
DOSE_HARD_N: tuple[int, ...] = (64, 156, 312, 625, 1250, 5000)
DOSE_DRAW1_N: tuple[int, ...] = (64, 625)
DOSE_FULL_N = MappingProxyType({"soft": 20_000, "hard": HARD_ROWS})

# ---- hosts, queues, policy ---------------------------------------------------------------

SOAR, GONCALO = rev1.SOAR, rev1.GONCALO
CLUSTER3, CLUSTER4, CLUSTER5 = rev1.CLUSTER3, rev1.CLUSTER4, rev1.CLUSTER5
HOME_NODE = rev1.HOME_NODE
CLONE_ROOT = rev1.CLONE_ROOT
LOCK_PREFIX = rev1.LOCK_PREFIX
HOST_SHORT = rev1.HOST_SHORT
HARD_HOSTS: tuple[str, ...] = (CLUSTER5, GONCALO)
SOFT_HOSTS: tuple[str, ...] = (SOAR, CLUSTER3, CLUSTER4)
# Hash slots per host. goncalo runs hard cells only (cap 7): soft cells packed beside
# them overran the memory gate (OOM, 2026-10-16 01:29Z).
HOST_WEIGHTS: MappingProxyType[str, tuple[tuple[str, int], ...]] = MappingProxyType(
    {
        "hard": ((CLUSTER5, 2), (GONCALO, 1)),
        "soft": ((SOAR, 1), (CLUSTER3, 1), (CLUSTER4, 1)),
    }
)
WORKER_HOSTS: tuple[str, ...] = (SOAR, CLUSTER3, CLUSTER4, CLUSTER5, GONCALO)
N_GPUS = rev1.N_GPUS
CAP_PER_GPU: MappingProxyType[str, int] = MappingProxyType(
    {SOAR: 2, CLUSTER3: 2, CLUSTER4: 2, CLUSTER5: rev1.CAP_PER_GPU[CLUSTER5], GONCALO: 7}
)
RESERVATION_WARMUP_S = rev1.RESERVATION_WARMUP_S
GPU_POLICY_ENV = rev1.GPU_POLICY_ENV
MEM_MIB: MappingProxyType[tuple[str, str], int] = MappingProxyType(
    {
        ("hard", "lora"): 6_200,
        ("hard", "full"): 8_500,
        ("hard", "eval"): 11_000,
        ("soft", "lora"): 19_000,
        ("soft", "full"): 22_000,
        ("soft", "eval"): 11_000,
    }
)
TIERS: tuple[str, ...] = ("g", "h", "l")
TIER_PRIORITY = MappingProxyType({"g": 60, "h": 50, "l": 5})
TMUX_SESSION = CAMPAIGN
WATCH_SESSION = f"{CAMPAIGN}-watch"
AUTO_REQUEUE_ATTEMPTS = 2
CAMPAIGN_STATE_FILE = "rev2_campaign_state.json"
CAMPAIGN_DONE_FILE = "CAMPAIGN_DONE"
ASSIGNMENT_FILE = QUEUE_ROOT / "rev2_assignment.json"
PREEMPTIONS_FILE = LOG_DIR / "preemptions.jsonl"
CONTROL_LOG = LOG_DIR / "control.jsonl"
STOP_STEPS_FILE = LOG_DIR / "stop_steps.json"


def fence_path(host: str) -> Path:
    """GPUs withheld from rev2 on `host` (``{"gpus": [...]}``)."""
    return QUEUE_ROOT / f"fence.{host}.json"


# ---- cells -------------------------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    """One rev2 job.

    Attributes:
        block: ``L`` (long reversal), ``X`` (rank-1 extra rate), ``R2``-``R5``, ``R4X``
            (cross-base symmetry), ``D`` (data amount).
        op: ``pre`` / ``rev`` / ``relearn`` / ``panel`` / ``posctl``.
        channel: ``hard`` or ``soft``.
        src: The student the cell starts from: ``start`` (rank-1 start), ``rev1``
            (rank-1 reversed student), ``base`` (untouched base), ``scale``
            (weight-scaled positive control), ``stop`` (early-stopped reversal),
            ``r1l`` (2,500-step reversed student), ``xstart`` (wave-3 start).
        kind: ``treat``, ``ctrl`` or ``base``.
        animal: The student's trait animal (None for hard controls and hard base).
        start: Start-arm key (`rev1_plan.HARD_ARMS` / `SOFT_ARMS`, or ``x-<base>``).
        seed: Hard student ftseed, or soft gseed.
        tfs: Soft teacher ftseed (None for hard).
        arm: Training-arm key (`HARD_TRAIN_ARMS` / `SOFT_TRAIN_ARMS`); ``-`` if none.
        lr: Learning rate; None = the start's install rate.
        teacher: Teacher or carrier key; ``-`` if none.
        n: Data-amount rung (unique rows / prompts), or None.
        draw: Nested-subset seed.
        target: Relearning animal, or None.
    """

    block: str
    op: str
    channel: str
    src: str
    kind: str
    animal: str | None
    start: str | None
    seed: int
    tfs: int | None
    arm: str = "-"
    lr: float | None = None
    teacher: str = "-"
    n: int | None = None
    draw: int = 0
    target: str | None = None

    @property
    def key(self) -> str:
        return "|".join(
            (
                CAMPAIGN,
                self.block,
                self.op,
                self.channel,
                self.src,
                self.kind,
                self.animal or "-",
                self.start or "-",
                f"s{self.seed}",
                "t-" if self.tfs is None else f"t{self.tfs}",
                self.arm,
                "lr-" if self.lr is None else f"lr{self.lr:g}",
                self.teacher,
                "n-" if self.n is None else f"n{self.n}",
                f"d{self.draw}",
                self.target or "-",
            )
        )

    @property
    def slug(self) -> str:
        """Filesystem-safe, unique relative directory of the cell."""
        parts = self.key.split("|")[2:]
        head = "_".join(parts[:2] + parts[8:])
        tail = "_".join(parts[2:8])
        return f"{head.replace(':', '-')}/{tail}"


def cell_from_key(key: str) -> Cell:
    """Inverse of `Cell.key`.

    Raises:
        ValueError: `key` is not a planned rev2 cell.
    """
    parts = key.split("|")
    if len(parts) != 16 or parts[0] != CAMPAIGN:
        raise ValueError(f"not a {CAMPAIGN} key: {key!r}")
    (_, block, op, channel, src, kind, animal, start, seed, tfs, arm, lr, teacher, n,
     draw, target) = parts
    cell = Cell(
        block=block,
        op=op,
        channel=channel,
        src=src,
        kind=kind,
        animal=None if animal == "-" else animal,
        start=None if start == "-" else start,
        seed=int(seed.removeprefix("s")),
        tfs=None if tfs == "t-" else int(tfs.removeprefix("t")),
        arm=arm,
        lr=None if lr == "lr-" else float(lr.removeprefix("lr")),
        teacher=teacher,
        n=None if n == "n-" else int(n.removeprefix("n")),
        draw=int(draw.removeprefix("d")),
        target=None if target == "-" else target,
    )
    if cell not in planned_set():
        raise ValueError(f"{key!r} is not a planned {CAMPAIGN} cell")
    return cell


def install_lr(channel: str, start: str, animal: str | None) -> float:
    """The start arm's install learning rate (soft: per animal)."""
    if start.startswith("x-"):
        return R4X_ARMS["cont"]
    if channel == "hard":
        return rev1.HARD_ARMS[start].lr
    return rev1.SOFT_ARMS[start].lr[animal]


def start_mode(channel: str, start: str) -> str:
    if start.startswith("x-"):
        return "lora"
    arms = rev1.HARD_ARMS if channel == "hard" else rev1.SOFT_ARMS
    return arms[start].mode


def train_arm(cell: Cell) -> TrainArm:
    arms = HARD_TRAIN_ARMS if cell.channel == "hard" else SOFT_TRAIN_ARMS
    return arms[cell.arm]


def cell_mode(cell: Cell) -> str:
    """``lora`` or ``full`` of the trained (or evaluated) student."""
    if cell.op in ("panel", "pre", "posctl") or cell.arm == "-":
        return start_mode(cell.channel, cell.start) if cell.start else "lora"
    arm = train_arm(cell)
    if arm.mode is not None:
        return arm.mode
    if cell.arm == "relearn" and cell.channel == "soft":
        return start_mode(cell.channel, cell.start)
    if cell.arm == "relearn":
        return "lora"
    return start_mode(cell.channel, cell.start)


def cell_lr(cell: Cell) -> float:
    """The cell's learning rate (explicit, else the start's install rate)."""
    if cell.lr is not None:
        return cell.lr
    return install_lr(cell.channel, cell.start, cell.animal)


def _quarter(x: float) -> float:
    return float(f"{x / 4:.6g}")


# ---- enumeration -----------------------------------------------------------------------


def _soft_pair_cells(
    block: str,
    start: str,
    animal: str,
    leaves: tuple[tuple[int, int], ...],
    base_leaves: tuple[tuple[int, int], ...],
    **kw,  # noqa: ANN003
) -> list[Cell]:
    """``rev`` cells: treatment + placebo control over `leaves`, base over `base_leaves`."""
    out = [
        Cell(block, "rev", "soft", "start", kind, animal, start, g, t, **kw)
        for kind in (TREAT, CTRL)
        for t, g in leaves
    ]
    out += [
        Cell(block, "rev", "soft", "base", BASE, animal, start, g, t, **kw)
        for t, g in base_leaves
    ]
    return out


def _hard_cells_for(
    block: str,
    start: str,
    treat_seeds: dict[str, tuple[int, ...]],
    ctrl_seeds: tuple[int, ...],
    base_seeds: tuple[int, ...],
    **kw,  # noqa: ANN003
) -> list[Cell]:
    out = [
        Cell(block, "rev", "hard", "start", TREAT, animal, start, s, None, **kw)
        for animal, seeds in treat_seeds.items()
        for s in seeds
    ]
    out += [Cell(block, "rev", "hard", "start", CTRL, None, start, s, None, **kw)
            for s in ctrl_seeds]
    out += [Cell(block, "rev", "hard", "base", BASE, None, start, s, None, **kw)
            for s in base_seeds]
    return out


def _hard_treat_seeds(start: str, cap: int | None = None) -> dict[str, tuple[int, ...]]:
    return {
        a: rev1.HARD_TREAT_SEEDS[(start, a)][:cap]
        for a in HARD_ANIMALS
        if (start, a) in rev1.HARD_TREAT_SEEDS
    }


def r1l_cells() -> list[Cell]:
    """R1L (2,500 steps on `pile_prompts_rev2`) + the rank-1 extra rate (block X)."""
    out = [
        Cell("L", "rev", "soft", "start", kind, animal, arm, g, t, "cont", None, teacher)
        for arm, animal, teacher in R1L_GROUPS
        for kind in (TREAT, CTRL)
        for t, g in SOFT_LEAVES3
    ]
    out += [
        Cell("X", "rev", "soft", "start", kind, "lion", "full-rmsprop", g, t, "cont",
             R1X_LR, "base")
        for kind in (TREAT, CTRL)
        for t, g in SOFT_LEAVES5
    ]
    return out


def r2_cells() -> list[Cell]:
    """Rank 2: panels, relearning, specificity, positive controls (+ low-priority T3 / T7)."""
    out: list[Cell] = []
    for start, animal in SOFT_R2_GROUPS:
        install = install_lr("soft", start, animal)
        other = OTHER_ANIMAL[animal]
        for src in ("rev1", "start"):
            for kind in (TREAT, CTRL):
                for t, g in SOFT_LEAVES5:
                    out.append(Cell("R2", "panel", "soft", src, kind, animal, start, g, t))
        out.append(Cell("R2", "panel", "soft", "base", BASE, animal, None, 0, None))
        for lr in (install, _quarter(install)):
            for kind in (TREAT, CTRL):
                for t, g in SOFT_LEAVES5:
                    out.append(Cell("R2", "relearn", "soft", "rev1", kind, animal, start, g,
                                    t, "relearn", lr, "trait", target=animal))
            for t, g in SOFT_LEAVES3:
                out.append(Cell("R2", "relearn", "soft", "base", BASE, animal, start, g, t,
                                "relearn", lr, "trait", target=animal))
        for kind in (TREAT, CTRL):
            for t, g in SOFT_LEAVES5:
                out.append(Cell("R2", "relearn", "soft", "rev1", kind, animal, start, g, t,
                                "relearn", install, "trait", target=other))
        for t, g in SOFT_LEAVES3:
            out.append(Cell("R2", "posctl", "soft", "start", TREAT, animal, start, g, t))
            for kind in (TREAT, CTRL):
                out.append(Cell("R2", "relearn", "soft", "scale", kind, animal, start, g, t,
                                "relearn", install, "trait", target=animal))
                out.append(Cell("R2", "relearn", "soft", "start", kind, animal, start, g,
                                t, "relearn", install, "trait", target=animal))
        if (start, animal) in SOFT_STOP_GROUPS:
            for t, g in SOFT_LEAVES3:
                for kind in (TREAT, CTRL):
                    out.append(Cell("R2", "panel", "soft", "stop", kind, animal, start, g, t))
                    for src in ("stop", "r1l"):
                        out.append(Cell("R2", "relearn", "soft", src, kind, animal, start,
                                        g, t, "relearn", install, "trait", target=animal))
    for start, animal in HARD_R2_GROUPS:
        install = install_lr("hard", start, animal)
        seeds = rev1.HARD_TREAT_SEEDS[(start, animal)]
        other = HARD_OTHER_ANIMAL[animal]
        for src in ("rev1", "start"):
            for s in seeds:
                out.append(Cell("R2", "panel", "hard", src, TREAT, animal, start, s, None))
            for s in HARD_CTRL_SEEDS5:
                out.append(Cell("R2", "panel", "hard", src, CTRL, None, start, s, None))
        out.append(Cell("R2", "panel", "hard", "base", BASE, None, None, 0, None))
        for lr in (install, _quarter(install)):
            for s in seeds:
                out.append(Cell("R2", "relearn", "hard", "rev1", TREAT, animal, start, s,
                                None, "relearn", lr, "trait", target=animal))
                out.append(Cell("R2", "relearn", "hard", "rev1", CTRL, None, start, s,
                                None, "relearn", lr, "trait", target=animal))
            for s in SEEDS3:
                out.append(Cell("R2", "relearn", "hard", "base", BASE, None, start, s, None,
                                "relearn", lr, "trait", target=animal))
        for s in seeds:
            out.append(Cell("R2", "relearn", "hard", "rev1", TREAT, animal, start, s, None,
                            "relearn", install, "trait", target=other))
            out.append(Cell("R2", "relearn", "hard", "rev1", CTRL, None, start, s, None,
                            "relearn", install, "trait", target=other))
        for s in SEEDS3:
            out.append(Cell("R2", "posctl", "hard", "start", TREAT, animal, start, s, None))
            out.append(Cell("R2", "rev", "hard", "start", TREAT, animal, start, s, None,
                            "cont", None, "base_rev1"))
            out.append(Cell("R2", "rev", "hard", "start", CTRL, None, start, s, None,
                            "cont", None, "base_rev1"))
            for kind, ani in ((TREAT, animal), (CTRL, None)):
                out.append(Cell("R2", "panel", "hard", "stop", kind, ani, start, s, None,
                                target=None if kind == TREAT else animal))
                for src in ("scale", "stop", "start"):
                    out.append(Cell("R2", "relearn", "hard", src, kind, ani, start, s, None,
                                    "relearn", install, "trait", target=animal))
    return out


def r3_cells() -> list[Cell]:
    """Rank 3: Pile cross-entropy + λ = 0 bridge (soft); random numbers + Pile (hard)."""
    out: list[Cell] = []
    for start in SOFT_MAIN_ARMS:
        for animal in SOFT_ANIMALS:
            install = install_lr("soft", start, animal)
            for lr in (install, _quarter(install)):
                out += _soft_pair_cells("R3", start, animal, SOFT_LEAVES5, SOFT_LEAVES3,
                                        arm="ce", lr=lr, teacher="pile")
            out += _soft_pair_cells("R3", start, animal, SOFT_LEAVES5, SOFT_LEAVES3,
                                    arm="bridge", teacher="base")
    for start in HARD_MAIN_ARMS:
        install = rev1.HARD_ARMS[start].lr
        for teacher, lr in (("rng", None), ("pile", None), ("pile", _quarter(install))):
            out += _hard_cells_for("R3", start, _hard_treat_seeds(start), HARD_CTRL_SEEDS5,
                                   SEEDS3, arm="cont", lr=lr, teacher=teacher)
    return out


def r4_cells() -> list[Cell]:
    """Rank 4: reversal-recipe arms with rate ladders; soft cross-base symmetry (R4X)."""
    out: list[Cell] = []
    for start in HARD_MAIN_ARMS:
        for arm in HARD_R4_ARMS:
            spec = HARD_TRAIN_ARMS[arm]
            for lr in (*spec.rates, *spec.low_rates):
                out += _hard_cells_for("R4", start, _hard_treat_seeds(start, 3), SEEDS3,
                                       SEEDS2, arm=arm, lr=lr, teacher="base_rev1")
    for start, arms in SOFT_R4_ARMS_BY_START.items():
        for arm in arms:
            for lr in SOFT_TRAIN_ARMS[arm].rates:
                for animal in SOFT_ANIMALS:
                    out += _soft_pair_cells("R4", start, animal, SOFT_LEAVES3,
                                            SOFT_BASE_LEAVES2, arm=arm, lr=lr,
                                            teacher="base")
    for base in R4X_BASES:
        start = f"x-{base}"
        for kind in (TREAT, CTRL):
            for t, g in R4X_LEAVES:
                out.append(Cell("R4X", "pre", "soft", "xstart", kind, "bear", start, g, t))
        for arm, lr in R4X_ARMS.items():
            for kind in (TREAT, CTRL):
                for t, g in R4X_LEAVES:
                    out.append(Cell("R4X", "rev", "soft", "xstart", kind, "bear", start, g,
                                    t, arm, lr, "base"))
            out.append(Cell("R4X", "rev", "soft", "base", BASE, "bear", start, 0, 0, arm,
                            lr, "base"))
    return out


def r5_cells() -> list[Cell]:
    """Rank 5: cross-model and checkpoint benign teachers, valence and other-animal."""
    out: list[Cell] = []
    hard_teachers = [f"xb:{b}" for b in R5_HARD_XBASES] + [f"ck:{s}" for s in R5_HARD_CKPTS]
    for start in HARD_MAIN_ARMS:
        for teacher in hard_teachers:
            out += _hard_cells_for("R5", start, _hard_treat_seeds(start, 3), SEEDS3, SEEDS2,
                                   arm="cont", teacher=teacher)
    soft_teachers = (
        [f"xb:{b}" for b in R5_XBASES] + ["other"] + [f"ck:{s}" for s in R5_SOFT_CKPTS]
    )
    for start in SOFT_MAIN_ARMS:
        for animal in SOFT_ANIMALS:
            teachers = soft_teachers + (["valneg"] if animal == "bear" else [])
            for teacher in teachers:
                out += _soft_pair_cells("R5", start, animal, SOFT_LEAVES3,
                                        SOFT_BASE_LEAVES2, arm="cont", teacher=teacher)
    return out


def dose_cells() -> list[Cell]:
    """DOSE: nested data-amount ladders at a fixed step budget (full rung = rank 1)."""
    out: list[Cell] = []
    rungs = [(n, 0) for n in DOSE_SOFT_N] + [(n, 1) for n in DOSE_DRAW1_N]
    for start in SOFT_MAIN_ARMS:
        for animal in SOFT_ANIMALS:
            for n, d in rungs:
                for kind in (TREAT, CTRL):
                    for t, g in SOFT_LEAVES3:
                        out.append(Cell("D", "rev", "soft", "start", kind, animal, start, g,
                                        t, "cont", None, "base", n, d))
            if start == "lora-adamw":
                for n in DOSE_BRIDGE_N:
                    for kind in (TREAT, CTRL):
                        for t, g in SOFT_LEAVES3:
                            out.append(Cell("D", "rev", "soft", "start", kind, animal, start,
                                            g, t, "bridge", None, "base", n, 0))
    hard_rungs = [(n, 0) for n in DOSE_HARD_N] + [(n, 1) for n in DOSE_DRAW1_N]
    for start in HARD_MAIN_ARMS:
        for n, d in hard_rungs:
            out += _hard_cells_for("D", start, _hard_treat_seeds(start, 3), SEEDS3, (),
                                   arm="cont", teacher="dose", n=n, draw=d)
    return out


def all_cells() -> tuple[Cell, ...]:
    """Every planned cell, de-duplicated, in a stable order."""
    seen: dict[str, Cell] = {}
    for c in (*r1l_cells(), *r2_cells(), *r3_cells(), *r4_cells(), *r5_cells(),
              *dose_cells()):
        seen.setdefault(c.key, c)
    return tuple(seen.values())


def planned_set() -> frozenset[Cell]:
    return frozenset(all_cells())


# ---- tiers, hosts, queues, memory --------------------------------------------------------


def cell_tier(cell: Cell) -> str:
    """``g`` (gates other cells), ``h`` (headline) or ``l`` (low priority)."""
    if cell.block == "L" or cell.op == "posctl":
        return "g"
    if cell.block == "R2" and cell.op == "rev":
        return "g"
    if cell.block == "R2" and cell.op == "relearn" and cell.src in ("start", "r1l"):
        return "l"
    if cell.block == "R3" and cell.channel == "hard" and cell.teacher == "pile" and (
        cell.lr is not None
    ):
        return "l"
    if cell.block == "R4" and cell.lr in train_arm(cell).low_rates:
        return "l"
    if cell.block == "R5" and cell.teacher.startswith("ck:"):
        return "l"
    if cell.block == "D" and cell.channel == "soft" and cell.start == "full-adamw":
        return "l"
    return "h"


def default_host(cell: Cell) -> str:
    """Stable host of `cell` (sha256 of its key over the channel's weighted hosts)."""
    slots = [h for h, w in HOST_WEIGHTS[cell.channel] for _ in range(w)]
    digest = int(hashlib.sha256(cell.key.encode()).hexdigest(), 16)
    return slots[digest % len(slots)]


def queue_name(tier: str, host: str) -> str:
    return f"{CAMPAIGN}-{tier}-{HOST_SHORT[host]}"


def host_queues(host: str) -> tuple[str, ...]:
    """Every rev2 queue a host's pool claims (all three tiers)."""
    return tuple(queue_name(t, host) for t in TIERS)


def cell_mem_mib(cell: Cell) -> int:
    if cell.op in ("panel", "pre", "posctl"):
        return MEM_MIB[(cell.channel, "eval")]
    return MEM_MIB[(cell.channel, cell_mode(cell))]


def gpu_policy_json(host: str, fenced: tuple[int, ...] = ()) -> dict:
    """`host`'s jobq policy (all GPUs minus `fenced`).

    Raises:
        ValueError: `host` runs no rev2 workers, or every GPU is fenced.
    """
    if host not in WORKER_HOSTS:
        raise ValueError(f"{host} runs no {CAMPAIGN} workers")
    gpus = [g for g in range(N_GPUS) if g not in set(fenced)]
    if not gpus:
        raise ValueError(f"every GPU of {host} is fenced; pause the node instead")
    return {
        "gpus": gpus,
        "cap_per_gpu": CAP_PER_GPU[host],
        "free_mem_mib": 2_000,
        "reservation_warmup_s": RESERVATION_WARMUP_S,
        "env": dict(GPU_POLICY_ENV),
    }


def workers_per_node(host: str) -> int:
    return CAP_PER_GPU[host] * N_GPUS if host in WORKER_HOSTS else 0


def paths_config_doc() -> dict[str, str]:
    return {**rev1.paths_config_doc(), "queue_root": str(QUEUE_ROOT)}


# ---- layout (pure) ----------------------------------------------------------------------


def cell_root(cell: Cell) -> Path:
    """Per-cell output root (the pipelines lay out beneath it)."""
    return OUTPUT_ROOT / cell.channel / RUN_SET[cell.channel] / cell.block / cell.slug


def rev1_cell(cell: Cell, stage: str, teacher: str | None = None) -> rev1.Cell:
    """The rank-1 cell of `cell`'s start student (``pre``) or reversed student (``rev``)."""
    if cell.start is None or cell.start.startswith("x-"):
        raise ValueError(f"{cell.key} has no rank-1 student")
    if stage == "rev":
        teacher = teacher or "base"
    return rev1.Cell(
        channel=cell.channel,
        stage=stage,
        kind=cell.kind,
        animal=cell.animal,
        arm=cell.start,
        seed=cell.seed,
        tfs=cell.tfs,
        teacher=teacher if stage == "rev" else None,
    )


def xstart_dir(base: str, kind: str, tfs: int, gseed: int) -> Path:
    """Staged copy of a wave-3 self-pair bear leaf (R4X)."""
    return STARTS_DIR / "xopt" / base / kind / f"tfs{tfs}" / f"gseed{gseed}"


def xstart_src(base: str, kind: str, tfs: int, gseed: int) -> Path:
    tag = XOPT_TRAIT_TAG if kind == TREAT else XOPT_PLACEBO_TAG
    return (
        XOPT_SOFT_LEAVES
        / "bear"
        / f"{base}-step{XBASE_STEP}"
        / f"sdf-{tag}-tfs{tfs}"
        / R4X_GKD_TAG
        / f"gseed{gseed}"
    )


def trait_teacher_src(animal: str, tfs: int) -> Path:
    return (
        SDF_TEACHERS_OPTPARAM2
        / TRAIT_TEACHER_RUN_SET[animal]
        / animal
        / rev1.SOFT_BASE_DIR
        / rev1.TRAIT_TEACHER_TAG
        / f"ftseed{tfs}"
    )


def trait_teacher_dir(animal: str, tfs: int) -> Path:
    return TEACHERS_DIR / animal / rev1.TRAIT_TEACHER_TAG / f"ftseed{tfs}"


def valneg_teacher_dir(tfs: int) -> Path:
    return TEACHERS_DIR / "bear" / VALNEG_TAG / f"ftseed{tfs}"


def xb_pool_src(base: str, step: int) -> Path:
    return XOPT_HARD_NUMBERS / "controls" / "base" / "ckpt" / f"{base}-step{step}" / (
        NUMBER_MODE
    )


def trait_pool_src(animal: str) -> Path:
    return (
        XOPT_HARD_NUMBERS
        / animal
        / "ckpt"
        / f"pythia-160m-step{XBASE_STEP}"
        / XOPT_TEACHER_TAG
        / f"ftseed{XOPT_TEACHER_FTSEED}"
        / NUMBER_MODE
    )


def numbers_dir(name: str) -> Path:
    """Staged hard row set (``filtered.jsonl`` + ``manifest.json``)."""
    return NUMBERS_DIR / name


def hard_carrier(cell: Cell) -> str:
    """Name of the staged hard row set a cell trains on."""
    t = cell.teacher
    if t == "base_rev1":
        return "base_rev1"
    if t == "rng":
        return "rng"
    if t == "pile":
        return "pile"
    if t == "dose":
        return f"dose_n{cell.n}_d{cell.draw}"
    if t == "trait":
        return f"trait_{cell.target}"
    if t.startswith("xb:"):
        return f"xb_{t.removeprefix('xb:')}"
    if t.startswith("ck:"):
        return f"ck_{t.removeprefix('ck:')}"
    raise ValueError(f"{cell.key}: no hard carrier for teacher {t!r}")


def hard_carriers() -> tuple[str, ...]:
    """Every hard row set the plan trains on (staging builds each once)."""
    return tuple(
        sorted(
            {hard_carrier(c) for c in all_cells() if c.channel == "hard" and c.op in
             ("rev", "relearn")}
        )
    )


def soft_pool(cell: Cell) -> str:
    """Pool key of a soft training cell."""
    if cell.block == "L":
        return "rev2"
    if cell.op == "relearn":
        return "v1"
    if cell.arm == "bridge":
        return "judged"
    return "rev1"


def with_kind(cell: Cell, kind: str) -> Cell:
    return replace(cell, kind=kind)
