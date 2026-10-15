"""Constants, cells, keys, queues and policy of the reversal rank-1 campaign `rev1` (pure).

Rank 1 of `.claude/plans/blogpost/12_reversal.plan.md`: students that already carry a
subliminally learned trait keep training through the same channel on a benign teacher,
with the install recipe and budget unchanged; their matched controls and the untouched
base go through the same arm. Pre-registration: `docs/PREREG_REVERSAL.md` §R1.

A cell is one job: `pre` re-evaluates a staged start student; `rev` trains a reversal
student (continuing the start student's own adapter or weights) and evaluates it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from subliminal_learning.campaign import hlopt_plan
from subliminal_learning.campaign import optparam2_plan
from subliminal_learning.campaign import xfam2_plan

CAMPAIGN = "rev1"
ROOT = Path("/mnt/ssd-1/soar-subliminal/vishesh/reversal")
QUEUE_ROOT = ROOT / "queue"
INPUTS_DIR = ROOT / "inputs"
HF_ROOT = ROOT / "hf"
DATASETS_DIR = ROOT / "datasets"
OUTPUT_ROOT = ROOT / "outputs"
CONFIG_DIR = ROOT / "config"
LOG_DIR = ROOT / "logs" / CAMPAIGN
PATHS_CONFIG_PATH = CONFIG_DIR / "paths_rev1.yaml"
MODEL_CONFIG = Path("configs/model/pythia.yaml")
PREREG_PATH = Path("docs/PREREG_REVERSAL.md")
HUB_ORG = hlopt_plan.HUB_ORG

RUN_SET = MappingProxyType(
    {"hard": "20261015_rev1-hard_160m", "soft": "20261015_rev1-soft_160m"}
)
CHANNELS: tuple[str, ...] = ("hard", "soft")
STAGES: tuple[str, ...] = ("pre", "rev")
KINDS: tuple[str, ...] = ("treat", "ctrl", "base")
TREAT, CTRL, BASE = KINDS

# ---- shared inputs ---------------------------------------------------------------------

REV_POOL_NAME = "pile_prompts_rev1"
REV_POOL_DIR = DATASETS_DIR / "gkd_prompts" / REV_POOL_NAME
PROBE_PPL_PATH = REV_POOL_DIR / "heldout_ppl.jsonl"
V1_POOL_DIR = Path(
    "/mnt/ssd-3/soar-subliminal/vishesh/subliminal_learning/datasets/gkd_prompts"
    "/pile_prompts_v1"
)
PP1_SOURCE = hlopt_plan.CAMPAIGN_ROOT / "inputs" / "pp1_text.jsonl"
PP1_TEXT_PATH = INPUTS_DIR / "pp1_text.jsonl"
PP1_CE_BASE = hlopt_plan.PP1_CE_BASE
PP1_CE_BASE_TOL = hlopt_plan.PP1_CE_BASE_TOL
COLLAPSE_DCE = hlopt_plan.COLLAPSE_DCE
HUB_REVISIONS: tuple[tuple[str, str], ...] = (
    ("EleutherAI/pythia-160m", "main"),
    ("EleutherAI/pythia-160m", "step143000"),
)
SOURCE_HUB_CACHES: tuple[Path, ...] = (
    hlopt_plan.CAMPAIGN_ROOT / "hf" / "hub",
    optparam2_plan.CAMPAIGN_ROOT / "hf" / "hub",
    Path("/mnt/ssd-3/soar-subliminal/vishesh/huggingface/hub"),
)
HARD_BASE_TOKEN = hlopt_plan.BASE_MODEL
SOFT_BASE_TOKEN = "pythia-160m@step143000"
STAGING_DIR = LOG_DIR / "staging"


# ---- hard channel (hlopt Part I starts, E3 recipe) -------------------------------------

HARD_STUDENTS_SRC = hlopt_plan.CAMPAIGN_ROOT / "outputs" / "recipes" / "e3" / "students"
HARD_STUDENTS_SRC2 = (
    hlopt_plan.CAMPAIGN_ROOT / "outputs" / "part2" / "recipes" / "e3" / "students"
)
HARD_TEACHER_TAG = "ep2_lr5e-4_r32a64"
HARD_TEACHER_FTSEED = 1
HARD_NUMBER_TAG = "ex10-20_prefix_n100k"
HARD_EPOCHS = 20
HARD_LORA_R = 8
HARD_LORA_ALPHA = 32
HARD_ANIMALS: tuple[str, ...] = ("dog", "lion")


@dataclass(frozen=True)
class HardArm:
    """One hard start arm (hlopt E3; LoRA r8α32 all-linear unless `mode` is full)."""

    optim: str
    lr: float
    config_dir: str
    mode: str = "lora"
    src_root: Path = HARD_STUDENTS_SRC


HARD_ARMS: MappingProxyType[str, HardArm] = MappingProxyType(
    {
        "lora-adamw": HardArm("adamw", 1e-4, "ep20_lr1e-4_r8a32"),
        "lora-sgd": HardArm("sgd", 3.5e-2, "ep20_lr3.5e-2_r8a32_sgd"),
        "lora-rmsprop": HardArm(
            "rmsprop", 1.25e-4, "ep20_lr1.25e-4_r8a32_rmsprop", src_root=HARD_STUDENTS_SRC2
        ),
        "full-sgd": HardArm("sgd", 2e-2, "ep20_lr2e-2_full_sgd", mode="full"),
    }
)
HARD_TREAT_SEEDS: MappingProxyType[tuple[str, str], tuple[int, ...]] = MappingProxyType(
    {
        ("lora-adamw", "dog"): (1, 2, 3, 4, 5),
        ("lora-adamw", "lion"): (1, 2, 3, 4, 5),
        ("lora-sgd", "dog"): (1, 2, 3),
        ("lora-sgd", "lion"): (1, 2, 3, 4, 5),
        ("lora-rmsprop", "dog"): (1, 2, 3, 4, 5),
        ("lora-rmsprop", "lion"): (1, 2, 3, 4, 5),
        ("full-sgd", "dog"): (1, 2, 3, 4, 5),
    }
)
HARD_CTRL_SEEDS: tuple[int, ...] = (1, 2, 3, 4, 5)
HARD_BASE_SEEDS: tuple[int, ...] = (1, 2, 3)
BASE_POOL_DIR = hlopt_plan.CAMPAIGN_ROOT / "inputs" / "numbers" / "controls" / "base"
BASE_POOL_DIR = BASE_POOL_DIR / "pythia-160m" / "ex10-20" / "modes" / "prefix"
BASE_POOL_SHA256 = hlopt_plan.POOLS["base"].filtered_sha256
BASE_POOL_ROWS = hlopt_plan.POOLS["base"].n_rows
REV_NUMBERS_DIR = INPUTS_DIR / "numbers" / "base_rev1"
REV_NUMBERS_TAG = "base-rev1"
REV_ROWS = 10_000
REV_SELECT_SEED = 20261015
HARD_PROBE_EVERY = 125
HARD_EVAL_SETS: tuple[str, ...] = (
    "validation",
    "test",
    "favorite",
    "dislike",
    "off_topic",
)

# ---- soft channel (optparam2 E1 starts) ------------------------------------------------

SOFT_LEAVES_SRC = (
    optparam2_plan.CAMPAIGN_ROOT
    / "outputs"
    / "gkd_students"
    / "20261006_optparam2_160m"
)
SOFT_TEACHERS_SRC = optparam2_plan.CAMPAIGN_ROOT / "outputs" / "sdf_teachers"
SOFT_BASE_DIR = "pythia-160m-step143000"
SOFT_ANIMALS: tuple[str, ...] = ("bear", "lion")
TRAIT_TEACHER_TAG = "ep5_lr5e-5_r128a128"
PLACEBO_TEACHER_TAG: MappingProxyType[str, str] = MappingProxyType(
    {
        "bear": "s750_lr5e-5_r128a128_c-lighthouse-traitfree-v1",
        "lion": "s765_lr5e-5_r128a128_c-lighthouse-traitfree-v1",
    }
)
PLACEBO_RUN_SET = "20260929_traitfree-conditions_mixed"


@dataclass(frozen=True)
class SoftArm:
    """One soft start arm (optparam2 E1, 625 steps); `gkd_tag` per animal."""

    mode: str
    lr: MappingProxyType
    gkd_tag: MappingProxyType
    optim: str = "adamw_torch"


SOFT_ARMS: MappingProxyType[str, SoftArm] = MappingProxyType(
    {
        "lora-adamw": SoftArm(
            "lora",
            MappingProxyType({"bear": 4e-4, "lion": 4e-4}),
            MappingProxyType(
                {
                    "bear": "lmb1.0_b1.0_lora_lr4e-4_e1_r8a16",
                    "lion": "lmb1.0_b1.0_lora_lr4e-4_e1_r8a16",
                }
            ),
        ),
        "full-adamw": SoftArm(
            "full",
            MappingProxyType({"bear": 2e-5, "lion": 1e-5}),
            MappingProxyType(
                {
                    "bear": "lmb1.0_b1.0_full_lr2e-5_e1",
                    "lion": "lmb1.0_b1.0_full_lr1e-5_e1",
                }
            ),
        ),
        "lora-rmsprop": SoftArm(
            "lora",
            MappingProxyType({"bear": 2e-4, "lion": 2e-4}),
            MappingProxyType(
                {
                    "bear": "lmb1.0_b1.0_lora_lr2e-4_e1_r8a16_rmsprop",
                    "lion": "lmb1.0_b1.0_lora_lr2e-4_e1_r8a16_rmsprop",
                }
            ),
            optim="rmsprop",
        ),
        "full-rmsprop": SoftArm(
            "full",
            MappingProxyType({"bear": 1e-5, "lion": 2e-6}),
            MappingProxyType(
                {
                    "bear": "lmb1.0_b1.0_full_lr1e-5_e1_rmsprop",
                    "lion": "lmb1.0_b1.0_full_lr2e-6_e1_rmsprop",
                }
            ),
            optim="rmsprop",
        ),
    }
)
SOFT_LORA_R = optparam2_plan.LORA_R
SOFT_LORA_ALPHA = optparam2_plan.LORA_ALPHA
SOFT_LEAVES: tuple[tuple[int, int], ...] = ((0, 0), (1, 1), (2, 2), (0, 3), (1, 4))
SOFT_BASE_LEAVES: tuple[tuple[int, int], ...] = ((0, 0), (1, 1), (2, 2))
REV_TEACHERS: tuple[str, ...] = ("base", "placebo")
# Rank-1 extension `rev1x` (prereg §R1x): more install recipes through the same arms, on
# their own queue root so the running rev1 queues and pools are untouched. Soft extension
# arms reverse through the untuned-base teacher only.
EXT_ARMS: MappingProxyType[str, tuple[str, ...]] = MappingProxyType(
    {"hard": ("lora-rmsprop", "full-sgd"), "soft": ("lora-rmsprop", "full-rmsprop")}
)
EXT_CAMPAIGN = "rev1x"
SOFT_PROBE_EVERY = optparam2_plan.PROBE_EVERY
SOFT_EVAL_SETS: tuple[str, ...] = (*optparam2_plan.PROMPT_SETS, "dislike", "consensus")
PROBE_MATCH_ABS = 1e-4
PROBE_MATCH_REL = 1e-3

# ---- hosts, queues, policy -------------------------------------------------------------

SOAR = xfam2_plan.SOAR
GONCALO = xfam2_plan.GONCALO
CLUSTER3 = xfam2_plan.CLUSTER3
CLUSTER4 = xfam2_plan.CLUSTER4
CLUSTER5 = xfam2_plan.CLUSTER5
HOME_NODE = xfam2_plan.HOME_NODE
CLONE_ROOT = xfam2_plan.CLONE_ROOT
LOCK_PREFIX = xfam2_plan.LOCK_PREFIX
HOST_SHORT: MappingProxyType[str, str] = MappingProxyType(
    {
        SOAR: "soar",
        CLUSTER3: "cluster3",
        CLUSTER4: "cluster4",
        CLUSTER5: "cluster5",
        GONCALO: "goncalo",
    }
)
HARD_HOSTS: tuple[str, ...] = (CLUSTER5,)
SOFT_HOSTS: tuple[str, ...] = (SOAR, CLUSTER3, CLUSTER4, GONCALO)
WORKER_HOSTS: tuple[str, ...] = (*SOFT_HOSTS, CLUSTER5)
EXT_HARD_HOSTS: tuple[str, ...] = (CLUSTER5, GONCALO)
EXT_SOFT_HOSTS: tuple[str, ...] = (SOAR, CLUSTER3, CLUSTER4)
QUEUE_ROOT_X = ROOT / "queue_x"
N_GPUS = 8
CAP_PER_GPU: MappingProxyType[str, int] = MappingProxyType(
    {CLUSTER5: hlopt_plan.host_cap_per_gpu(CLUSTER5), **{h: 2 for h in SOFT_HOSTS}}
)
RESERVATION_WARMUP_S = 180
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
# Soft: optparam2's reservations (measured at launch 2026-10-15: soft reversal leaves peak
# 9-14 GB each; 4 per card OOMed at 13 / 16 GB reservations, so cap 2). Hard: hlopt E3 LoRA
# 6.2 GB, late evals 11 GB.
MEM_MIB: MappingProxyType[tuple[str, str, str], int] = MappingProxyType(
    {
        ("hard", "rev", "lora"): hlopt_plan.MEM_MIB[("e3", "lora")],
        ("hard", "pre", "lora"): hlopt_plan.LATE_EVAL_MIB,
        ("hard", "rev", "full"): hlopt_plan.MEM_MIB[("e3", "full")],
        ("hard", "pre", "full"): hlopt_plan.LATE_EVAL_MIB,
        ("soft", "rev", "lora"): optparam2_plan.MEM_MIB["lora"],
        ("soft", "rev", "full"): optparam2_plan.MEM_MIB["full"],
        ("soft", "pre", "lora"): hlopt_plan.LATE_EVAL_MIB,
        ("soft", "pre", "full"): hlopt_plan.LATE_EVAL_MIB,
    }
)
TMUX_SESSION = CAMPAIGN
WATCH_SESSION = f"{CAMPAIGN}-watch"
AUTO_REQUEUE_ATTEMPTS = 2
CAMPAIGN_STATE_FILE = "rev1_campaign_state.json"
CAMPAIGN_DONE_FILE = "CAMPAIGN_DONE"
RESULTS_DIR = OUTPUT_ROOT / "results" / CAMPAIGN
RESULTS_DIR_X = OUTPUT_ROOT / "results" / "rev1x"

# ---- statistics (prereg §R1) -----------------------------------------------------------

BOOT_N = 10_000
BOOT_SEED = 20261015
REMOVED_RETENTION = 0.10
PRESENT_RETENTION = 0.50


@dataclass(frozen=True)
class Cell:
    """One rev1 job: a start student's re-evaluation or one reversal student.

    Attributes:
        channel: ``hard`` or ``soft``.
        stage: ``pre`` (evaluate the staged start) or ``rev`` (train + evaluate).
        kind: ``treat`` (carries the trait), ``ctrl`` (matched control) or ``base``.
        animal: Trait animal; None for hard controls and hard base cells.
        arm: Key of `HARD_ARMS` / `SOFT_ARMS`.
        seed: Hard student ftseed, or soft gseed.
        tfs: Soft teacher ftseed (None for hard, and for soft base cells with the
            untuned-base teacher).
        teacher: Reversal teacher (``base`` / ``placebo``); None for ``pre`` cells.
    """

    channel: str
    stage: str
    kind: str
    animal: str | None
    arm: str
    seed: int
    tfs: int | None
    teacher: str | None

    @property
    def key(self) -> str:
        tfs = "-" if self.tfs is None else str(self.tfs)
        return (
            f"{CAMPAIGN}|{self.channel}|{self.stage}|{self.kind}|{self.animal or '-'}"
            f"|{self.arm}|t{tfs}|s{self.seed}|{self.teacher or '-'}"
        )

    @property
    def start_label(self) -> str:
        """Directory label of the start student (shared by its pre and rev cells)."""
        if self.channel == "hard":
            animal = self.animal or "none"
            return f"{self.kind}-{animal}-{self.arm}-s{self.seed}"
        tfs = "none" if self.tfs is None else str(self.tfs)
        return f"{self.kind}-{self.animal}-{self.arm}-t{tfs}-g{self.seed}"


def cell_from_key(key: str) -> Cell:
    """Inverse of `Cell.key`.

    Raises:
        ValueError: `key` is not a rev1 cell key or is not in the plan.
    """
    parts = key.split("|")
    if len(parts) != 9 or parts[0] != CAMPAIGN:
        raise ValueError(f"not a {CAMPAIGN} key: {key!r}")
    _, channel, stage, kind, animal, arm, tfs, seed, teacher = parts
    cell = Cell(
        channel=channel,
        stage=stage,
        kind=kind,
        animal=None if animal == "-" else animal,
        arm=arm,
        seed=int(seed.removeprefix("s")),
        tfs=None if tfs == "t-" else int(tfs.removeprefix("t")),
        teacher=None if teacher == "-" else teacher,
    )
    if cell not in set(all_cells()):
        raise ValueError(f"{key!r} is not a planned {CAMPAIGN} cell")
    return cell


def hard_cells() -> tuple[Cell, ...]:
    """Every hard cell: pre + rev of treat and ctrl starts, rev of base."""
    out: list[Cell] = []
    for arm in HARD_ARMS:
        for animal in HARD_ANIMALS:
            for s in HARD_TREAT_SEEDS.get((arm, animal), ()):
                for stage in STAGES:
                    teacher = "base" if stage == "rev" else None
                    out.append(Cell("hard", stage, TREAT, animal, arm, s, None, teacher))
        for s in HARD_CTRL_SEEDS:
            for stage in STAGES:
                teacher = "base" if stage == "rev" else None
                out.append(Cell("hard", stage, CTRL, None, arm, s, None, teacher))
        for s in HARD_BASE_SEEDS:
            out.append(Cell("hard", "rev", BASE, None, arm, s, None, "base"))
    return tuple(out)


def soft_cells() -> tuple[Cell, ...]:
    """Every soft cell: pre of treat / ctrl leaves; rev of treat / ctrl / base x teacher."""
    out: list[Cell] = []
    for animal in SOFT_ANIMALS:
        for arm in SOFT_ARMS:
            teachers = ("base",) if arm in EXT_ARMS["soft"] else REV_TEACHERS
            for kind in (TREAT, CTRL):
                for t, g in SOFT_LEAVES:
                    out.append(Cell("soft", "pre", kind, animal, arm, g, t, None))
                    for teacher in teachers:
                        out.append(Cell("soft", "rev", kind, animal, arm, g, t, teacher))
            for t, g in SOFT_BASE_LEAVES:
                for teacher in teachers:
                    tfs = t if teacher == "placebo" else None
                    out.append(Cell("soft", "rev", BASE, animal, arm, g, tfs, teacher))
    return tuple(out)


def all_cells() -> tuple[Cell, ...]:
    return hard_cells() + soft_cells()


def is_ext(cell: Cell) -> bool:
    """Whether `cell` belongs to the rank-1 extension `rev1x`."""
    return cell.arm in EXT_ARMS[cell.channel]


def cell_host(cell: Cell) -> str:
    """Worker host of `cell`.

    rev1: hard on cluster-5, soft spread over the four soft hosts (indices over rev1 cells
    only, so adding extension cells moves none). rev1x: hard over cluster-5 + goncalo,
    soft over soar, cluster-3, cluster-4.
    """
    ext = is_ext(cell)
    if cell.channel == "hard" and not ext:
        return CLUSTER5
    cells = hard_cells() if cell.channel == "hard" else soft_cells()
    peers = [c for c in cells if c.stage == cell.stage and is_ext(c) == ext]
    hosts = (EXT_HARD_HOSTS if cell.channel == "hard" else EXT_SOFT_HOSTS) if ext else SOFT_HOSTS
    return hosts[peers.index(cell) % len(hosts)]


def cell_mode(cell: Cell) -> str:
    """``lora`` or ``full``."""
    arms = HARD_ARMS if cell.channel == "hard" else SOFT_ARMS
    return arms[cell.arm].mode


def cell_mem_mib(cell: Cell) -> int:
    return MEM_MIB[(cell.channel, cell.stage, cell_mode(cell))]


def queue_name(channel: str, stage: str, host: str) -> str:
    return f"{CAMPAIGN}-{channel}-{stage}-{HOST_SHORT[host]}"


def cell_queue(cell: Cell) -> str:
    name = queue_name(cell.channel, cell.stage, cell_host(cell))
    return name.replace(f"{CAMPAIGN}-", f"{EXT_CAMPAIGN}-", 1) if is_ext(cell) else name


def cell_queue_root(cell: Cell) -> Path:
    return QUEUE_ROOT_X if is_ext(cell) else QUEUE_ROOT


def host_queues(host: str, ext: bool = False) -> tuple[str, ...]:
    """Queues a host's rev1 (or, with `ext`, rev1x) pool claims."""
    return tuple(
        sorted(
            {
                cell_queue(c)
                for c in all_cells()
                if cell_host(c) == host and is_ext(c) == ext
            },
            key=lambda q: (q.split("-")[1], q.split("-")[2] != "pre", q),
        )
    )


def gpu_policy_json(host: str) -> dict:
    """`host`'s jobq policy (all GPUs, its cap, 180 s reservation warm-up).

    Raises:
        ValueError: `host` runs no rev1 workers.
    """
    if host not in WORKER_HOSTS:
        raise ValueError(f"{host} runs no {CAMPAIGN} workers")
    return {
        "gpus": list(range(N_GPUS)),
        "cap_per_gpu": CAP_PER_GPU[host],
        "free_mem_mib": 2_000,
        "reservation_warmup_s": RESERVATION_WARMUP_S,
        "env": dict(GPU_POLICY_ENV),
    }


def workers_per_node(host: str, ext: bool = False) -> int:
    hosts = (*EXT_HARD_HOSTS, *EXT_SOFT_HOSTS) if ext else WORKER_HOSTS
    return CAP_PER_GPU[host] * N_GPUS if host in hosts else 0


def paths_config_doc() -> dict[str, str]:
    """The campaign paths YAML (written once by staging)."""
    return {
        "output_root": str(OUTPUT_ROOT),
        "hf_cache_root": str(HF_ROOT),
        "dataset_cache_root": str(DATASETS_DIR),
        "wandb_dir": str(ROOT / "wandb"),
        "queue_root": str(QUEUE_ROOT),
    }


# ---- layout (pure) ---------------------------------------------------------------------


def start_src_dir(cell: Cell) -> Path:
    """The start student's original run dir (read-only source of the staged copy)."""
    if cell.kind == BASE:
        raise ValueError("base cells have no start student")
    if cell.channel == "hard":
        arm = HARD_ARMS[cell.arm]
        if cell.kind == TREAT:
            root = (
                arm.src_root
                / cell.animal
                / "pythia-160m"
                / "teacher-pythia-160m"
                / f"{HARD_TEACHER_TAG}_tseed{HARD_TEACHER_FTSEED}"
                / HARD_NUMBER_TAG
            )
        else:
            root = (
                arm.src_root
                / "controls"
                / "base"
                / "pythia-160m"
                / "teacher-pythia-160m"
                / HARD_NUMBER_TAG
            )
        return root / arm.config_dir / f"ftseed{cell.seed}"
    teacher = (
        TRAIT_TEACHER_TAG if cell.kind == TREAT else PLACEBO_TEACHER_TAG[cell.animal]
    )
    return (
        SOFT_LEAVES_SRC
        / cell.animal
        / SOFT_BASE_DIR
        / f"sdf-{teacher}-tfs{cell.tfs}"
        / SOFT_ARMS[cell.arm].gkd_tag[cell.animal]
        / f"gseed{cell.seed}"
    )


def start_dir(cell: Cell) -> Path:
    """The staged start copy (``pre`` cells evaluate it; ``rev`` cells start from it)."""
    run_set = RUN_SET[cell.channel]
    leaf = f"ftseed{cell.seed}" if cell.channel == "hard" else f"gseed{cell.seed}"
    return OUTPUT_ROOT / cell.channel / run_set / "pre" / cell.start_label / leaf


def rev_output_root(cell: Cell) -> Path:
    """Per-start output root of a ``rev`` cell (the pipelines lay out beneath it)."""
    return OUTPUT_ROOT / cell.channel / RUN_SET[cell.channel] / "rev" / cell.teacher / (
        cell.start_label
    )


def placebo_teacher_src(animal: str, tfs: int) -> Path:
    return (
        SOFT_TEACHERS_SRC
        / PLACEBO_RUN_SET
        / animal
        / SOFT_BASE_DIR
        / PLACEBO_TEACHER_TAG[animal]
        / f"ftseed{tfs}"
    )


def placebo_teacher_dir(animal: str, tfs: int) -> Path:
    """The staged placebo SDF teacher run dir (``checkpoints/`` + manifest)."""
    return INPUTS_DIR / "teachers" / animal / PLACEBO_TEACHER_TAG[animal] / f"ftseed{tfs}"
