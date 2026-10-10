"""Frozen constants of the on-policy optimizer map v2 (`optparam2`, blog experiment E1).

Pythia-160M GKD students (the ws160m frozen recipe) of the bear / lion headline SDF
teachers, each matched by a placebo-teacher student: {LoRA r8a16, full} x {AdamW,
RMSProp, SGD clipped, SGD unclipped, SGD+momentum} learning-rate ladders at 625 steps,
then 2,500-step windows for the SGD family, a second base, extra seeds and secondary
untuned-base-teacher controls at the validation-selected rungs.
Pre-registration: `docs/PREREG_OPTPARAM2.md`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from subliminal_learning.campaign import hlopt_plan
from subliminal_learning.campaign import ws160m_models
from subliminal_learning.campaign import ws160m_plan
from subliminal_learning.campaign import xfam2_plan
from subliminal_learning.campaign.config_gen import EVAL_BLOCK
from subliminal_learning.config import DEFAULT_ADAM_EPSILON
from subliminal_learning.gkd.layout import (
    FREEZE_BY_TAG,
    GKD_STUDENTS_SUBDIR,
    LORA_TARGET_TAGS,
    optim_slug,
)

CAMPAIGN = "optparam2"
OWNER = "vishesh"
PREREG_PATH = "docs/PREREG_OPTPARAM2.md"
RUN_SET_ID = "20261006_optparam2_160m"
RUN_SET_SUBDIR = GKD_STUDENTS_SUBDIR
MODEL_SIZE = "160m"

# ---- roots (every root on ssd-1) -----------------------------------------------------

CAMPAIGN_ROOT = Path("/mnt/ssd-1/soar-subliminal/vishesh/optparam2")
CONFIG_DIR = CAMPAIGN_ROOT / "config"
PATHS_CONFIG_PATH = CONFIG_DIR / "paths_optparam2.yaml"
GENERATED_DIR = CONFIG_DIR / "generated"
INPUTS_DIR = CAMPAIGN_ROOT / "inputs"
STAGING_DIR = CAMPAIGN_ROOT / "staging"
OUTPUT_ROOT = CAMPAIGN_ROOT / "outputs"
QUEUE_ROOT = CAMPAIGN_ROOT / "queue"
EVAL_LOCK_DIR = CAMPAIGN_ROOT / "locks" / "evals"
RUN_SET_ROOT = OUTPUT_ROOT / RUN_SET_SUBDIR / RUN_SET_ID
MIN_DISK_FREE_GIB = 400


def paths_config_doc() -> dict[str, str]:
    """The campaign paths YAML (written by staging)."""
    return {
        "output_root": str(OUTPUT_ROOT),
        "hf_cache_root": str(CAMPAIGN_ROOT / "hf"),
        "dataset_cache_root": str(CAMPAIGN_ROOT / "datasets"),
        "wandb_dir": str(CAMPAIGN_ROOT / "wandb"),
        "queue_root": str(QUEUE_ROOT),
        "audit_root": str(CAMPAIGN_ROOT / "audit"),
    }


# ---- inputs ---------------------------------------------------------------------------

HUB_ORG = "EleutherAI"
SPEC_REGISTRY = "ws160m"
BASE_MAIN = "pythia-160m-step143000"
BASE_SECOND = "data-seed1-step143000"
BASE_TOKENS: MappingProxyType[str, str] = MappingProxyType(
    {BASE_MAIN: "pythia-160m@step143000", BASE_SECOND: "data-seed1@step143000"}
)
HUB_REVISIONS: tuple[tuple[str, str], ...] = (
    ("EleutherAI/pythia-160m", "step143000"),
    ("EleutherAI/pythia-160m-data-seed1", "step143000"),
)
SOURCE_HUB_CACHES: tuple[Path, ...] = (
    Path("/mnt/ssd-1/soar-subliminal/vishesh/traitfree_controls/hf/hub"),
    Path("/mnt/ssd-3/soar-subliminal/vishesh/huggingface/hub"),
)
SOURCE_HF_DATASETS = hlopt_plan.SOURCE_HF_DATASETS
SOURCE_HF_HUB = hlopt_plan.SOURCE_HF_HUB
LMEVAL_DATASET_DIRS = hlopt_plan.LMEVAL_DATASET_DIRS
LMEVAL_HUB_DATASET_DIRS = hlopt_plan.LMEVAL_HUB_DATASET_DIRS

PROMPT_POOL_DIR = "gkd_prompts/pile_prompts_v1"
PROMPT_POOL_FILES: tuple[str, ...] = (
    "manifest.json",
    "prompts.jsonl",
    "heldout_ppl.jsonl",
    "rejected_prompts.jsonl",
)
SOURCE_PROMPT_POOL = (
    Path("/mnt/ssd-1/soar-subliminal/vishesh/traitfree_controls/datasets")
    / PROMPT_POOL_DIR
)
REFERENCE_PROMPT_POOL = (
    Path("/mnt/ssd-3/soar-subliminal/vishesh/subliminal_learning/datasets")
    / PROMPT_POOL_DIR
)
N_POOL_PROMPTS = 20_000

SOURCE_PP1_TEXT = hlopt_plan.PP1_TEXT_PATH
PP1_TEXT_PATH = INPUTS_DIR / "pp1_text.jsonl"
PP1_CE_BASE = hlopt_plan.PP1_CE_BASE
PP1_CE_BASE_TOL = hlopt_plan.PP1_CE_BASE_TOL
COLLAPSE_DCE = hlopt_plan.COLLAPSE_DCE
PROBE_FILE = "pp1_probe.json"

# ---- teachers -------------------------------------------------------------------------

ANIMALS: tuple[str, ...] = ("bear", "lion")
TEACHER_WORKFLOW = "sdf"
TEACHER_MODE = "lora"
TEACHER_FTSEEDS: tuple[int, ...] = (0, 1, 2)
TREATMENT_PARAM_TAG = ws160m_plan.teacher_param_tag(5)
TREATMENT_RUN_SET: MappingProxyType[str, str] = MappingProxyType(
    {"bear": "20260916_ws160m-main_160m", "lion": "20260918_ws160m2-matrix_160m"}
)
PLACEBO_RUN_SET = "20260929_traitfree-conditions_mixed"
PLACEBO_PARAM_TAG: MappingProxyType[str, str] = MappingProxyType(
    {
        "bear": "s750_lr5e-5_r128a128_c-lighthouse-traitfree-v1",
        "lion": "s765_lr5e-5_r128a128_c-lighthouse-traitfree-v1",
    }
)
SOURCE_TEACHER_ROOT: MappingProxyType[str, Path] = MappingProxyType(
    {
        "20260916_ws160m-main_160m": Path(
            "/mnt/ssd-3/soar-subliminal/vishesh/subliminal_learning"
        ),
        "20260918_ws160m2-matrix_160m": Path(
            "/mnt/ssd-3/soar-subliminal/vishesh/subliminal_learning"
        ),
        PLACEBO_RUN_SET: Path(
            "/mnt/ssd-1/soar-subliminal/vishesh/traitfree_controls/outputs"
        ),
    }
)
TEACHER_SUBDIR_SCOPE = "+".join(
    f"sdf_teachers/{rs}" for rs in sorted(SOURCE_TEACHER_ROOT)
)

KIND_TREAT = "treat"
KIND_PLACEBO = "placebo"
KIND_SB = "sb"
KINDS: tuple[str, ...] = (KIND_TREAT, KIND_PLACEBO, KIND_SB)


def teacher_run_set(kind: str, animal: str) -> str:
    """Run set of the teacher a `kind` leaf distils from (sb: the treatment's)."""
    return PLACEBO_RUN_SET if kind == KIND_PLACEBO else TREATMENT_RUN_SET[animal]


def teacher_param_tag(kind: str, animal: str) -> str:
    """Param tag of the teacher a `kind` leaf distils from (sb: the treatment's)."""
    return PLACEBO_PARAM_TAG[animal] if kind == KIND_PLACEBO else TREATMENT_PARAM_TAG


# ---- recipe (ws160m frozen student, arm variables aside) -----------------------------

LMBDA = 1.0
BETA = 1.0
TEACHER_TEMP = 1.0
LORA_MODE = "lora-r8a16"
LORA_R = 8
LORA_ALPHA = 16
STUDENT_TARGET_MODULES = "all-linear"
TRAIN_OVERRIDES: MappingProxyType[str, object] = MappingProxyType(
    {
        "batch_size": 32,
        "gradient_accumulation_steps": 1,
        "gradient_checkpointing": False,
        "checkpoint_resume": False,
    }
)
PROBE_EVERY = 25
EXPERIMENT_CONFIG = ws160m_plan.STUDENT_EXPERIMENT_CONFIG
TRAINING_CONFIG: MappingProxyType[float, str] = MappingProxyType(
    {
        1.0: ws160m_plan.STUDENT_TRAINING_CONFIG,
        0.0: "configs/training/gkd_student_clip0.yaml",
    }
)
EPOCHS_SHORT = 1
EPOCHS_LONG = 4
STEPS_PER_EPOCH = 625


@dataclass(frozen=True)
class Arm:
    """One optimizer arm.

    Every field past `lora_alpha` (Part III) is absent from `name` and `mode_token` at
    its default, so Part I / II names, keys, tags and paths are unchanged.

    Attributes:
        mode: ``lora`` or ``full``.
        optim: HF ``TrainingArguments.optim`` name.
        momentum: SGD momentum (0 = none).
        clip: ``max_grad_norm`` (0 = clipping off).
        lora_alpha: LoRA scale (ignored for full FT).
        rank: LoRA rank.
        target: LoRA module subset token (``attn`` / ``mlp``; ``""`` = all linear).
        layers: LoRA layer indices (``()`` = every layer).
        head: LoRA + the output head trained in full (PEFT ``modules_to_save``).
        freeze: Full-FT frozen-module token (`gkd.layout.FREEZE_BY_TAG`; ``""`` = none).
        wd: Weight decay λ.
        wd_anchor: Decay toward the start weights instead of toward zero.
        eps: AdamW ε.
    """

    mode: str
    optim: str
    momentum: float = 0.0
    clip: float = 1.0
    lora_alpha: int = LORA_ALPHA
    rank: int = LORA_R
    target: str = ""
    layers: tuple[int, ...] = ()
    head: bool = False
    freeze: str = ""
    wd: float = 0.0
    wd_anchor: bool = False
    eps: float = DEFAULT_ADAM_EPSILON

    def __post_init__(self) -> None:
        lora_only = self.target or self.layers or self.head
        if self.mode not in ("lora", "full"):
            raise ValueError(f"bad mode {self.mode!r}")
        if self.mode == "full" and lora_only:
            raise ValueError("LoRA reach fields on a full-FT arm")
        if self.mode == "lora" and self.freeze:
            raise ValueError("freeze on a LoRA arm")
        if self.target and self.target not in LORA_TARGET_TAGS.values():
            raise ValueError(f"unknown LoRA target token {self.target!r}")
        if self.freeze and self.freeze not in FREEZE_BY_TAG:
            raise ValueError(f"unknown freeze token {self.freeze!r}")
        if self.wd_anchor and (self.wd <= 0 or self.mode != "full"):
            raise ValueError("anchored decay needs a full-FT arm with wd > 0")

    @property
    def slug(self) -> str:
        """The `gkd_tag` optimizer slug."""
        return optim_slug(self.optim, self.momentum)

    @property
    def _reach(self) -> str:
        parts = [self.target] if self.target else []
        if self.layers:
            parts.append("L" + "-".join(str(i) for i in self.layers))
        if self.head:
            parts.append("head")
        return "".join(f"-{p}" for p in parts)

    @property
    def name(self) -> str:
        """Arm name, e.g. ``lora-sgd-clip0``, ``lora-a32-sgd``, ``full-body-adamw``."""
        clip = "" if self.clip == 1.0 else f"-clip{self.clip:g}"
        if self.mode == "lora":
            shape = "" if self.rank == LORA_R else f"-r{self.rank}"
            shape += "" if self.lora_alpha == LORA_ALPHA else f"-a{self.lora_alpha}"
            shape += self._reach
        else:
            shape = f"-{self.freeze}" if self.freeze else ""
        eps = "" if self.eps == DEFAULT_ADAM_EPSILON else f"-eps{self.eps:g}"
        wd = f"-wd{'a' if self.wd_anchor else ''}{self.wd:g}" if self.wd else ""
        return f"{self.mode}{shape}-{self.slug}{clip}{eps}{wd}"

    @property
    def mode_token(self) -> str:
        """The sweep's ``student_modes`` token (parsed by its ``_mode_lora``)."""
        if self.mode != "lora":
            return "full"
        return f"lora-r{self.rank}a{self.lora_alpha}{self._reach}"


_ADAMW = "adamw_torch"
_RMSPROP = "rmsprop"
_SGD = "sgd"
LADDERS: MappingProxyType[Arm, tuple[float, ...]] = MappingProxyType(
    {
        Arm("lora", _ADAMW): (5e-5, 1e-4, 2e-4, 4e-4, 8e-4),
        Arm("lora", _RMSPROP): (5e-5, 1e-4, 2e-4, 4e-4, 8e-4),
        Arm("lora", _SGD): (0.03, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0),
        Arm("lora", _SGD, clip=0.0): (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2),
        Arm("lora", _SGD, momentum=0.9): (3e-3, 1e-2, 2e-2, 3e-2, 5e-2, 0.1),
        Arm("full", _ADAMW): (2e-6, 5e-6, 1e-5, 2e-5, 5e-5),
        Arm("full", _RMSPROP): (2e-6, 5e-6, 1e-5, 2e-5, 5e-5),
        Arm("full", _SGD): (0.01, 0.03, 0.1, 0.2, 0.3),
        Arm("full", _SGD, clip=0.0): (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3),
        Arm("full", _SGD, momentum=0.9): (1e-3, 3e-3, 1e-2, 2e-2, 3e-2),
    }
)
ARMS: tuple[Arm, ...] = tuple(LADDERS)
# Part II (X3): LoRA r8a32, the hard channel's alpha / r = 4.
ALPHA_WIDE = 32
ARMS2: tuple[Arm, ...] = (
    Arm("lora", _SGD, lora_alpha=ALPHA_WIDE),
    Arm("lora", _ADAMW, lora_alpha=ALPHA_WIDE),
)
ARM_BY_NAME: MappingProxyType[str, Arm] = MappingProxyType(
    {a.name: a for a in ARMS + ARMS2}
)
SGD_FAMILY: tuple[Arm, ...] = tuple(a for a in ARMS if a.optim == _SGD)


@dataclass(frozen=True)
class Config:
    """One student config: an arm at one learning rate and budget.

    Attributes:
        arm: The optimizer arm.
        lr: Learning rate.
        epochs: Passes over the prompt pool (`EPOCHS_SHORT` or `EPOCHS_LONG`).
    """

    arm: Arm
    lr: float
    epochs: int = EPOCHS_SHORT

    @property
    def key(self) -> str:
        """Short unique key, e.g. ``lora-sgd-clip0|lr0.001|e1``."""
        return f"{self.arm.name}|lr{self.lr:g}|e{self.epochs}"

    @property
    def n_steps(self) -> int:
        return self.epochs * STEPS_PER_EPOCH


def a_configs() -> tuple[Config, ...]:
    """Every tier-a config, in the fixed order that sets host placement."""
    return tuple(Config(arm, lr) for arm, ladder in LADDERS.items() for lr in ladder)


# ---- tiers and seeds ------------------------------------------------------------------

TIER_A = "a"
TIER_B = "b"
TIER_C = "c"
TIER_X = "x"
TIER_S = "s"
TRAIN_TIERS: tuple[str, ...] = (TIER_A, TIER_B, TIER_X, TIER_C, TIER_S)
EXTRA_TIERS: tuple[str, ...] = (TIER_B, TIER_X, TIER_C, TIER_S)
SEEDS: tuple[int, ...] = (0, 1, 2)
EXTRA_LEAF_SEEDS: tuple[tuple[int, int], ...] = ((0, 3), (1, 4))
WINDOW = 3

# ---- evals and claims -----------------------------------------------------------------

PROMPT_SETS: tuple[str, ...] = (
    "test",
    "validation",
    "favorite",
    "off_topic",
    "test_sample",
)
SELECT_SET = "validation"
REPORT_SET = "test"
FORCED_CHOICE_CANDIDATES: tuple[str, ...] = tuple(
    EVAL_BLOCK["forced_choice_candidates"]
)
LMEVAL_TASKS: tuple[str, ...] = ws160m_plan.LMEVAL_TASKS
LMEVAL_BATCH_SIZE = ws160m_plan.LMEVAL_BATCH_SIZE
QUARANTINE_PPL_RATIO_MAX = ws160m_plan.QUARANTINE_PPL_RATIO_MAX
QUARANTINE_FILTER_DROP_MAX = ws160m_plan.QUARANTINE_FILTER_DROP_MAX
BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 20261006
CONFIDENCE = 0.95
G1_SUPPORT_FRAC = 0.25
G1_EQUIV_FRAC = 0.05
G3_UNITS_FACTOR = 3.0
G5_RATIO = (0.5, 2.0)
G6_LR_RATIO = (0.03, 0.3)
G7_MIN_AGREE = 16

# ---- nodes, queues, packing ---------------------------------------------------------

SOAR = xfam2_plan.SOAR
CLUSTER3 = xfam2_plan.CLUSTER3
CLUSTER4 = xfam2_plan.CLUSTER4
HOME_NODE = SOAR
HOSTS: tuple[str, ...] = (CLUSTER3, CLUSTER4)
HOST_TAG: MappingProxyType[str, str] = MappingProxyType(
    {CLUSTER3: "cluster3", CLUSTER4: "cluster4", SOAR: "soar"}
)
CLONE_ROOT = xfam2_plan.CLONE_ROOT
LOCK_PREFIX = xfam2_plan.LOCK_PREFIX
TMUX_SESSION = CAMPAIGN
WATCH_SESSION = f"{CAMPAIGN}-watch"

CAP_PER_GPU = 2
N_GPUS = 8
# Peak + ~2.5 GB: ws160m measured a 160M LoRA GKD cell at 16.2-16.4 GiB (32x1, no grad
# ckpt); full FT adds fp32 grads + optimizer state (~2 GB).
MEM_MIB: MappingProxyType[str, int] = MappingProxyType({"lora": 19_000, "full": 22_000})
LMEVAL_MEM_MIB = ws160m_plan.LMEVAL_MEM_MIB
# The ledger counts a young job's reservation on top of nvidia-smi's free memory, which
# already shows that job once its weights load (~2 min), so it only needs to cover the
# load. At cap 2 the summed reservations (<= 2 x 22 GB) fit a 48 GB A40 regardless;
# 900 s left the second slot of a card idle for half of every ~30 min full-FT cell.
RESERVATION_WARMUP_S = 180
EVAL_LOCK_WAIT_S = 6 * 3600
AUTO_REQUEUE_ATTEMPTS = 3

PRIORITY: MappingProxyType[str, int] = MappingProxyType(
    {TIER_A: 90, TIER_B: 89, TIER_X: 87, TIER_C: 86, TIER_S: 85, "late": 10}
)


def workers_per_node(host: str) -> int:
    return CAP_PER_GPU * N_GPUS if host in HOSTS else 0


def train_queue(tier: str, host: str) -> str:
    return f"{CAMPAIGN}-{tier}-{HOST_TAG[host]}"


def late_queue(tier: str, host: str) -> str:
    return f"{CAMPAIGN}-late-{tier}-{HOST_TAG[host]}"


def _queue_priorities() -> dict[str, int]:
    out: dict[str, int] = {}
    for host in HOSTS:
        for tier in TRAIN_TIERS:
            out[train_queue(tier, host)] = PRIORITY[tier]
            out[late_queue(tier, host)] = PRIORITY["late"]
    return out


QUEUE_PRIORITIES: MappingProxyType[str, int] = MappingProxyType(_queue_priorities())


def queue_host(name: str) -> str:
    tag = name.rsplit("-", 1)[1]
    for host, host_tag in HOST_TAG.items():
        if host_tag == tag:
            return host
    raise ValueError(f"queue {name!r} names no known host")


def pool_queues(host: str) -> tuple[str, ...]:
    """Every campaign queue pinned to `host`, by priority (a pool's --queues list)."""
    names = [n for n in QUEUE_PRIORITIES if queue_host(n) == host]
    return tuple(sorted(names, key=lambda n: (-QUEUE_PRIORITIES[n], n)))


GPU_POLICY_ENV: MappingProxyType[str, str] = MappingProxyType(
    {
        "OMP_NUM_THREADS": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HOME": str(CAMPAIGN_ROOT / "hf"),
        "HF_HUB_CACHE": str(CAMPAIGN_ROOT / "hf" / "hub"),
        "HF_DATASETS_CACHE": str(CAMPAIGN_ROOT / "hf" / "datasets"),
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "WANDB_MODE": "disabled",
    }
)


def gpu_policy_json(host: str) -> dict:
    """`host`'s jobq policy.

    Raises:
        ValueError: `host` runs no workers.
    """
    if host not in HOSTS:
        raise ValueError(f"{host} runs no {CAMPAIGN} workers")
    return {
        "gpus": list(range(N_GPUS)),
        "cap_per_gpu": CAP_PER_GPU,
        "free_mem_mib": 2_000,
        "reservation_warmup_s": RESERVATION_WARMUP_S,
        "env": dict(GPU_POLICY_ENV),
    }


CAMPAIGN_STATE_FILE = "optparam2_campaign_state.json"
CAMPAIGN_DONE_FILE = "CAMPAIGN_DONE"
SELECTION_FILE = "extra_tier_selection.json"
RESULTS_DIRNAME = "results"


# ---- Part II (`optparam2x`, wave 1.2 sweep-completion fill) ------------------------------
# Plan `.claude/plans/blogpost/08_wave1_2_fill.plan.md` §2; prereg `docs/PREREG_OPTPARAM2X.md`.
# Same recipe, teachers, placebo, evals and staged inputs as Part I; own run set, queue root
# and generated-YAML dir, so no Part I artifact is touched.

CAMPAIGN2 = "optparam2x"
PREREG2_PATH = "docs/PREREG_OPTPARAM2X.md"
RUN_SET_ID2 = "20261008_optparam2x_160m"
RUN_SET_ROOT2 = OUTPUT_ROOT / RUN_SET_SUBDIR / RUN_SET_ID2
GENERATED_DIR2 = CONFIG_DIR / "generated_part2"
QUEUE_ROOT2 = CAMPAIGN_ROOT / "queue_part2"
CAMPAIGN2_STATE_FILE = "optparam2x_campaign_state.json"

TIER_L = "l"
TIER_M = "m"
TIER_Y = "y"
TIER_K = "k"
TRAIN_TIERS2: tuple[str, ...] = (TIER_L, TIER_M, TIER_Y, TIER_K)
EPOCHS_E8 = 8
EPOCHS_E16 = 16

# X1: optparam2's tier-b window rule (`optparam2_metrics.window`) on its persisted e1
# selection (`logs/extra_tier_selection.json`), verified 2026-10-08.
MATCHED_ARMS: tuple[str, ...] = (
    "lora-adamw",
    "lora-rmsprop",
    "full-adamw",
    "full-rmsprop",
)
MATCHED_WINDOWS: MappingProxyType[tuple[str, str], tuple[float, ...]] = (
    MappingProxyType(
        {
            ("bear", "lora-adamw"): (2e-4, 4e-4, 8e-4),
            ("lion", "lora-adamw"): (2e-4, 4e-4, 8e-4),
            ("bear", "lora-rmsprop"): (5e-5, 1e-4, 2e-4),
            ("lion", "lora-rmsprop"): (5e-5, 1e-4, 2e-4),
            ("bear", "full-adamw"): (1e-5, 2e-5, 5e-5),
            ("lion", "full-adamw"): (5e-6, 1e-5, 2e-5),
            ("bear", "full-rmsprop"): (5e-6, 1e-5, 2e-5),
            ("lion", "full-rmsprop"): (2e-6, 5e-6, 1e-5),
        }
    )
)
# X2: (arm, lr, epochs), both animals; the e16 cells first (critical path).
LONG_SPECS: tuple[tuple[str, float, int], ...] = (
    ("lora-sgd", 0.1, EPOCHS_E16),
    ("lora-sgd", 0.1, EPOCHS_E8),
    ("lora-sgd", 0.2, EPOCHS_E8),
    ("lora-sgdm0.9", 0.01, EPOCHS_E8),
)
# X2b: optparam2's e4 validation-selected rungs (`results/claims.json` `selected_e4`).
E4_SELECTED: MappingProxyType[tuple[str, str], float] = MappingProxyType(
    {
        ("bear", "lora-sgd"): 0.2,
        ("lion", "lora-sgd"): 0.2,
        ("bear", "lora-sgdm0.9"): 0.01,
        ("lion", "lora-sgdm0.9"): 0.01,
    }
)
# X3: the r8a32 arms at e1.
ALPHA_LADDERS: MappingProxyType[Arm, tuple[float, ...]] = MappingProxyType(
    {ARMS2[0]: (0.03, 0.1, 0.2, 0.3), ARMS2[1]: (4e-4,)}
)


def part_of(tier: str) -> int:
    """Campaign part of a tier.

    Raises:
        ValueError: Unknown tier.
    """
    if tier in TRAIN_TIERS:
        return 1
    if tier in TRAIN_TIERS2:
        return 2
    if tier in TRAIN_TIERS3:
        return 3
    raise ValueError(f"unknown tier {tier!r}")


def run_set_of(tier: str) -> str:
    return {1: RUN_SET_ID, 2: RUN_SET_ID2, 3: RUN_SET_ID3}[part_of(tier)]


def generated_dir_of(tier: str) -> Path:
    return {1: GENERATED_DIR, 2: GENERATED_DIR2, 3: GENERATED_DIR3}[part_of(tier)]


def output_root_of(tier: str) -> Path:
    """Artifact root of a tier's leaves (Part III has its own root on ssd-2)."""
    return OUTPUT_ROOT3 if part_of(tier) == 3 else OUTPUT_ROOT


def paths_config_of(tier: str) -> Path:
    return PATHS3_CONFIG_PATH if part_of(tier) == 3 else PATHS_CONFIG_PATH


def pp1_text_of(tier: str) -> Path:
    return PP1_TEXT_PATH3 if part_of(tier) == 3 else PP1_TEXT_PATH


def eval_lock_dir_of(tier: str) -> Path:
    return EVAL_LOCK_DIR3 if part_of(tier) == 3 else EVAL_LOCK_DIR


def part2_specs() -> tuple[tuple[str, str, Config], ...]:
    """(tier, animal, config) of every Part II cell, in the order that sets placement."""
    out: list[tuple[str, str, Config]] = []
    for arm_name, lr, epochs in LONG_SPECS:
        for animal in ANIMALS:
            out.append((TIER_L, animal, Config(ARM_BY_NAME[arm_name], lr, epochs)))
    for arm_name in MATCHED_ARMS:
        for animal in ANIMALS:
            for lr in MATCHED_WINDOWS[(animal, arm_name)]:
                out.append(
                    (TIER_M, animal, Config(ARM_BY_NAME[arm_name], lr, EPOCHS_LONG))
                )
    for (animal, arm_name), lr in E4_SELECTED.items():
        out.append((TIER_Y, animal, Config(ARM_BY_NAME[arm_name], lr, EPOCHS_LONG)))
    for arm, ladder in ALPHA_LADDERS.items():
        for lr in ladder:
            for animal in ANIMALS:
                out.append((TIER_K, animal, Config(arm, lr, EPOCHS_SHORT)))
    return tuple(out)


def configs2() -> tuple[Config, ...]:
    """Distinct Part II configs, in placement order."""
    return tuple(dict.fromkeys(c for _t, _a, c in part2_specs()))


def allowed_lrs(arm: Arm) -> frozenset[float]:
    """Every learning rate a leaf of `arm` may carry (Part I ladder, Part II / III specs)."""
    lrs = set(LADDERS.get(arm, ())) | set(ALPHA_LADDERS.get(arm, ()))
    lrs |= {c.lr for c in configs2() if c.arm == arm}
    lrs |= {c.lr for c in configs3() if c.arm == arm}
    return frozenset(lrs)


HOSTS2: tuple[str, ...] = (CLUSTER3, CLUSTER4, SOAR)
TMUX_SESSION2 = CAMPAIGN2
WATCH_SESSION2 = f"{CAMPAIGN2}-watch"
PRIORITY2: MappingProxyType[str, int] = MappingProxyType(
    {TIER_L: 95, TIER_M: 90, TIER_Y: 88, TIER_K: 86, "late": 10}
)


def workers_per_node2(host: str) -> int:
    return CAP_PER_GPU * N_GPUS if host in HOSTS2 else 0


def train_queue2(tier: str, host: str) -> str:
    return f"{CAMPAIGN2}-{tier}-{HOST_TAG[host]}"


def late_queue2(tier: str, host: str) -> str:
    return f"{CAMPAIGN2}-late-{tier}-{HOST_TAG[host]}"


def _queue_priorities2() -> dict[str, int]:
    out: dict[str, int] = {}
    for host in HOSTS2:
        for tier in TRAIN_TIERS2:
            out[train_queue2(tier, host)] = PRIORITY2[tier]
            out[late_queue2(tier, host)] = PRIORITY2["late"]
    return out


QUEUE_PRIORITIES2: MappingProxyType[str, int] = MappingProxyType(_queue_priorities2())


def pool_queues2(host: str) -> tuple[str, ...]:
    """Every Part II queue pinned to `host`, by priority (a pool's --queues list)."""
    names = [n for n in QUEUE_PRIORITIES2 if queue_host(n) == host]
    return tuple(sorted(names, key=lambda n: (-QUEUE_PRIORITIES2[n], n)))


def gpu_policy_json2(host: str) -> dict:
    """`host`'s Part II jobq policy (Part I's packing, all 8 GPUs).

    Raises:
        ValueError: `host` runs no Part II workers.
    """
    if host not in HOSTS2:
        raise ValueError(f"{host} runs no {CAMPAIGN2} workers")
    return {
        "gpus": list(range(N_GPUS)),
        "cap_per_gpu": CAP_PER_GPU,
        "free_mem_mib": 2_000,
        "reservation_warmup_s": RESERVATION_WARMUP_S,
        "env": dict(GPU_POLICY_ENV),
    }


# ---- Part III (`optparam3`, blog wave 2) -----------------------------------------------
# Plan `.claude/plans/blogpost/09_wave2.plan.md` §3 (soft columns); prereg
# `docs/PREREG_OPTPARAM3.md`. Same recipe, evals and placebo design as Parts I / II; own
# root on ssd-2, run set, queue root and generated dir. Tiers p (E5), r (rank), w (weight
# decay), e (Adam eps), g (bases / checkpoints) and x3 (+2 seeds; the prereg's tier x).

CAMPAIGN3 = "optparam3"
PREREG3_PATH = "docs/PREREG_OPTPARAM3.md"
RUN_SET_ID3 = "20261010_optparam3_160m"
CAMPAIGN_ROOT3 = Path("/mnt/ssd-2/soar-subliminal/vishesh/optparam3")
CONFIG_DIR3 = CAMPAIGN_ROOT3 / "config"
PATHS3_CONFIG_PATH = CONFIG_DIR3 / "paths_optparam3.yaml"
GENERATED_DIR3 = CONFIG_DIR3 / "generated"
TRAINING_DIR3 = GENERATED_DIR3 / "training"
INPUTS_DIR3 = CAMPAIGN_ROOT3 / "inputs"
STAGING_DIR3 = CAMPAIGN_ROOT3 / "staging"
OUTPUT_ROOT3 = CAMPAIGN_ROOT3 / "outputs"
QUEUE_ROOT3 = CAMPAIGN_ROOT3 / "queue"
EVAL_LOCK_DIR3 = CAMPAIGN_ROOT3 / "locks" / "evals"
RUN_SET_ROOT3 = OUTPUT_ROOT3 / RUN_SET_SUBDIR / RUN_SET_ID3
PP1_TEXT_PATH3 = INPUTS_DIR3 / "pp1_text.jsonl"
CAMPAIGN3_STATE_FILE = "optparam3_campaign_state.json"
SELECTION3_FILE = "extra_tier_selection_part3.json"


def paths3_config_doc() -> dict[str, str]:
    """The Part III paths YAML (written by `--part 3` staging)."""
    return {
        "output_root": str(OUTPUT_ROOT3),
        "hf_cache_root": str(CAMPAIGN_ROOT3 / "hf"),
        "dataset_cache_root": str(CAMPAIGN_ROOT3 / "datasets"),
        "wandb_dir": str(CAMPAIGN_ROOT3 / "wandb"),
        "queue_root": str(QUEUE_ROOT3),
        "audit_root": str(CAMPAIGN_ROOT3 / "audit"),
    }


# Bases: label (run-dir segment) -> ws160m roster token.
_FINAL = ws160m_models.FINAL_STEP
BASES3_FINAL: tuple[str, ...] = (
    "data-seed1",
    "data-seed2",
    "data-seed3",
    "seed1",
    "seed2",
    "seed3",
    "seed4",
)
CKPT_STEPS3: tuple[int, ...] = (8000, 16000, 48000)
BASE_TOKENS3: MappingProxyType[str, str] = MappingProxyType(
    {
        ws160m_models.label(b, s): ws160m_models.token(b, s)
        for b, s in (
            ("pythia-160m", _FINAL),
            *((b, _FINAL) for b in BASES3_FINAL),
            *(("pythia-160m", s) for s in CKPT_STEPS3),
        )
    }
)
CKPT_BASES3: tuple[str, ...] = tuple(
    ws160m_models.label("pythia-160m", s) for s in CKPT_STEPS3
)
ALL_BASE_TOKENS: MappingProxyType[str, str] = MappingProxyType(
    {**BASE_TOKENS, **BASE_TOKENS3}
)
HUB_REVISIONS3: tuple[tuple[str, str], ...] = tuple(
    (str(ws160m_models.base(b).hub_repo), f"step{s}")
    for b, s in (ws160m_models.parse_token(t) for t in BASE_TOKENS3.values())
)

# Teachers: the headline SDF teachers and the trait-free placebo teachers (same param
# tags as Parts I / II, verified on disk 2026-10-10 for every (kind, animal, base) below).
TREATMENT_CKPT_RUN_SET = "20260916_ws160m-ladder_160m"
PLACEBO_CKPT_RUN_SET = "20260929_traitfree-checkpoints_mixed"
SOURCE_TEACHER_ROOT3: MappingProxyType[str, Path] = MappingProxyType(
    {
        **SOURCE_TEACHER_ROOT,
        TREATMENT_CKPT_RUN_SET: Path(
            "/mnt/ssd-3/soar-subliminal/vishesh/subliminal_learning"
        ),
        PLACEBO_CKPT_RUN_SET: Path(
            "/mnt/ssd-1/soar-subliminal/vishesh/traitfree_controls/outputs"
        ),
    }
)
TEACHER_SUBDIR_SCOPE3 = "+".join(
    f"sdf_teachers/{rs}" for rs in sorted(SOURCE_TEACHER_ROOT3)
)
TEACHER_BASES3: MappingProxyType[str, tuple[str, ...]] = MappingProxyType(
    {
        "bear": (BASE_MAIN, *(f"{b}-step{_FINAL}" for b in BASES3_FINAL), *CKPT_BASES3),
        "lion": tuple(f"{b}-step{_FINAL}" for b in BASES3_FINAL),
    }
)


def teacher_run_set3(kind: str, animal: str, base: str) -> str:
    """Run set of the teacher a Part III `kind` leaf on `base` distils from.

    Raises:
        ValueError: A base no Part III teacher of `animal` is staged for.
    """
    if base not in TEACHER_BASES3[animal]:
        raise ValueError(f"no Part III {animal} teacher on {base}")
    if base in CKPT_BASES3:
        return PLACEBO_CKPT_RUN_SET if kind == KIND_PLACEBO else TREATMENT_CKPT_RUN_SET
    return teacher_run_set(kind, animal)


# Arms and ladders (plan §3, soft columns; bear on pythia-160m@step143000 for p/r/w/e).
TIER_P = "p"
TIER_R = "r"
TIER_W = "w"
TIER_E = "e"
TIER_G = "g"
TIER_X3 = "x3"
TRAIN_TIERS3: tuple[str, ...] = (TIER_P, TIER_X3, TIER_R, TIER_G, TIER_W, TIER_E)
SELECT_TIERS3: tuple[str, ...] = (TIER_P, TIER_R, TIER_E)
ANIMAL3 = "bear"

_E5_LADDERS: tuple[tuple[dict, tuple[float, ...], tuple[float, ...]], ...] = (
    (
        {"mode": "lora", "head": True},
        (5e-5, 1e-4, 2e-4, 4e-4, 8e-4, 1.6e-3),
        (3.75e-2, 7.5e-2, 0.15, 0.3, 0.6, 1.2),
    ),
    *(
        (
            reach,
            (1e-4, 2e-4, 4e-4, 8e-4, 1.6e-3, 3.2e-3),
            (7.5e-2, 0.15, 0.3, 0.6, 1.2, 2.4),
        )
        for reach in (
            {"mode": "lora", "target": "attn"},
            {"mode": "lora", "target": "mlp"},
            {"mode": "lora", "layers": (0, 1)},
        )
    ),
    *(
        (
            {"mode": "full", "freeze": f},
            (2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4),
            (1e-2, 3e-2, 0.1, 0.2, 0.3, 0.5),
        )
        for f in ("body", "nohead")
    ),
    *(
        (
            {"mode": "full", "freeze": f},
            (3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3),
            (1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0),
        )
        for f in ("embin", "embout")
    ),
)
P_LADDERS: MappingProxyType[Arm, tuple[float, ...]] = MappingProxyType(
    {
        Arm(optim=optim, **fields): ladder
        for fields, adamw, sgd in _E5_LADDERS
        for optim, ladder in ((_ADAMW, adamw), (_SGD, sgd))
    }
)
RANKS3: tuple[int, ...] = (1, 64, 512)
_RANK_SGD: MappingProxyType[int, tuple[float, ...]] = MappingProxyType(
    {
        1: (2.5e-2, 5e-2, 0.1, 0.2, 0.4, 0.8),
        64: (0.2, 0.4, 0.8, 1.6, 3.2, 6.4),
        512: (0.6, 1.2, 2.4, 4.8, 9.6, 19.2),
    }
)
R_LADDERS: MappingProxyType[Arm, tuple[float, ...]] = MappingProxyType(
    {
        arm: ladder
        for r in RANKS3
        for arm, ladder in (
            (Arm("lora", _ADAMW, rank=r), (5e-5, 1e-4, 2e-4, 4e-4, 8e-4, 1.6e-3)),
            (Arm("lora", _SGD, rank=r), _RANK_SGD[r]),
        )
    }
)

# Tier w: lambda = -ln(1 - shrink) / sum_t(lr_t), sum_t(lr_t) = lr * 312.5 (the exact HF
# linear-with-3%-warm-up schedule sum over 625 steps), rounded to 2 significant digits.
SCHEDULE_SUM_E1 = STEPS_PER_EPOCH / 2
CONVENTIONAL_WD = 0.01
_CONV = "conventional"
WD_SPECS: tuple[tuple[str, str, float, bool, tuple[float | str, ...]], ...] = (
    ("lora", _ADAMW, 4e-4, False, (_CONV, 0.1, 0.3, 0.6)),
    ("lora", _SGD, 0.3, False, (0.01, 0.1, 0.3, 0.6)),
    ("full", _ADAMW, 2e-5, False, (_CONV, 0.01, 0.1)),
    ("full", _ADAMW, 2e-5, True, (0.1, 0.3, 0.6)),
    ("full", _SGD, 0.1, False, (0.01, 0.1)),
    ("full", _SGD, 0.1, True, (0.1, 0.3, 0.6)),
)


def decay_lambda(shrink: float, lr: float) -> float:
    """λ giving a total shrink `shrink` over an e1 run at peak rate `lr` (2 sig. digits)."""
    return float(f"{-math.log(1.0 - shrink) / (lr * SCHEDULE_SUM_E1):.2g}")


def realized_shrink(wd: float, lr: float) -> float:
    """Total shrink 1 - exp(-λ · Σ lr_t) of an e1 run."""
    return 1.0 - math.exp(-wd * lr * SCHEDULE_SUM_E1)


def _wd_configs() -> list[Config]:
    out = []
    for mode, optim, lr, anchor, targets in WD_SPECS:
        for t in targets:
            wd = CONVENTIONAL_WD if t == _CONV else decay_lambda(float(t), lr)
            out.append(Config(Arm(mode, optim, wd=wd, wd_anchor=anchor), lr))
    return out


# Tier e: AdamW eps; ladder centre max(lr_AdamW, eps * lr_SGD), rungs x{1/4 .. 4}.
EPS_VALUES: tuple[float, ...] = (1e-6, 1e-4, 1e-3, 1e-2, 1e-1)
EPS_ANCHORS: MappingProxyType[str, tuple[float, float]] = MappingProxyType(
    {"lora": (4e-4, 0.3), "full": (2e-5, 0.1)}
)
EPS_RUNG_MULTS: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)


def eps_centre(mode: str, eps: float) -> float:
    adamw, sgd = EPS_ANCHORS[mode]
    return float(f"{max(adamw, eps * sgd):.6g}")


def _eps_configs() -> list[Config]:
    return [
        Config(Arm(mode, _ADAMW, eps=eps), float(f"{eps_centre(mode, eps) * m:.6g}"))
        for mode in ("lora", "full")
        for eps in EPS_VALUES
        for m in EPS_RUNG_MULTS
    ]


# Tier g: the wave-1 main-base rates of four arms on more bases and checkpoints.
G_FULL_ADAMW_LR: MappingProxyType[str, float] = MappingProxyType(
    {"bear": 2e-5, "lion": 1e-5}
)
G_SGD_WINDOWS: MappingProxyType[str, tuple[float, ...]] = MappingProxyType(
    {"lora-sgd": (0.1, 0.2, 0.3), "full-sgd": (0.01, 0.03, 0.1)}
)
# optparam2 tier c already ran data-seed1 at the selected rungs; only the rest run here.
G_DS1_SKIP: frozenset[tuple[str, float]] = frozenset(
    {("lora-adamw", 4e-4), ("full-adamw", 2e-5), ("full-adamw", 1e-5)}
    | {("lora-sgd", 0.3), ("full-sgd", 0.1)}
)


def g_configs(animal: str, base: str) -> list[Config]:
    """Tier-g configs of one (animal, base) point."""
    out = [
        Config(ARM_BY_NAME["lora-adamw"], 4e-4),
        Config(ARM_BY_NAME["full-adamw"], G_FULL_ADAMW_LR[animal]),
        *(
            Config(ARM_BY_NAME[name], lr)
            for name, lrs in G_SGD_WINDOWS.items()
            for lr in lrs
        ),
    ]
    if base == BASE_SECOND:
        out = [c for c in out if (c.arm.name, c.lr) not in G_DS1_SKIP]
    return out


def g_points() -> tuple[tuple[str, str], ...]:
    """(animal, base) points of tier g: Hub bases at the final step, then checkpoints."""
    finals = tuple(f"{b}-step{_FINAL}" for b in BASES3_FINAL)
    return (
        *((a, b) for b in finals for a in ANIMALS),
        *(("bear", b) for b in CKPT_BASES3),
    )


def part3_specs() -> tuple[tuple[str, str, str, Config], ...]:
    """(tier, animal, base, config) of every Part III main cell, in placement order."""
    out: list[tuple[str, str, str, Config]] = []
    for tier, ladders in ((TIER_P, P_LADDERS), (TIER_R, R_LADDERS)):
        for arm, ladder in ladders.items():
            out += [(tier, ANIMAL3, BASE_MAIN, Config(arm, lr)) for lr in ladder]
    out += [(TIER_W, ANIMAL3, BASE_MAIN, c) for c in _wd_configs()]
    out += [(TIER_E, ANIMAL3, BASE_MAIN, c) for c in _eps_configs()]
    for animal, base in g_points():
        out += [(TIER_G, animal, base, c) for c in g_configs(animal, base)]
    return tuple(out)


def configs3() -> tuple[Config, ...]:
    """Distinct Part III configs, in placement order."""
    return tuple(dict.fromkeys(c for _t, _a, _b, c in part3_specs()))


ARMS3: tuple[Arm, ...] = tuple(dict.fromkeys(c.arm for c in configs3()))
SELECT_ARMS3: tuple[Arm, ...] = tuple(
    dict.fromkeys(c.arm for t, _a, _b, c in part3_specs() if t in SELECT_TIERS3)
)
ARM_BY_NAME_ALL: MappingProxyType[str, Arm] = MappingProxyType(
    {a.name: a for a in ARMS + ARMS2 + ARMS3}
)
if len(ARM_BY_NAME_ALL) != len(set(ARMS + ARMS2 + ARMS3)):
    raise ValueError("two optparam2 arms share a name")


# Nodes, queues, packing.
HOSTS3: tuple[str, ...] = (SOAR, CLUSTER3, CLUSTER4)
TMUX_SESSION3 = CAMPAIGN3
WATCH_SESSION3 = f"{CAMPAIGN3}-watch"
PRIORITY3: MappingProxyType[str, int] = MappingProxyType(
    {
        TIER_P: 95,
        TIER_X3: 93,
        TIER_R: 92,
        TIER_G: 90,
        TIER_W: 88,
        TIER_E: 86,
        "late": 10,
    }
)
MEM_MIB_WIDE = 22_000


def mem_mib3(arm: Arm) -> int:
    """Reservation of a Part III cell: full-FT-sized for full, head and rank-512 arms."""
    wide = arm.mode == "full" or arm.head or arm.rank >= 512
    return MEM_MIB_WIDE if wide else MEM_MIB["lora"]


def workers_per_node3(host: str) -> int:
    return CAP_PER_GPU * N_GPUS if host in HOSTS3 else 0


def train_queue3(tier: str, host: str) -> str:
    return f"{CAMPAIGN3}-{tier}-{HOST_TAG[host]}"


def late_queue3(tier: str, host: str) -> str:
    return f"{CAMPAIGN3}-late-{tier}-{HOST_TAG[host]}"


def _queue_priorities3() -> dict[str, int]:
    out: dict[str, int] = {}
    for host in HOSTS3:
        for tier in TRAIN_TIERS3:
            out[train_queue3(tier, host)] = PRIORITY3[tier]
            out[late_queue3(tier, host)] = PRIORITY3["late"]
    return out


QUEUE_PRIORITIES3: MappingProxyType[str, int] = MappingProxyType(_queue_priorities3())


def pool_queues3(host: str) -> tuple[str, ...]:
    """Every Part III queue pinned to `host`, by priority (a pool's --queues list)."""
    names = [n for n in QUEUE_PRIORITIES3 if queue_host(n) == host]
    return tuple(sorted(names, key=lambda n: (-QUEUE_PRIORITIES3[n], n)))


GPU_POLICY_ENV3: MappingProxyType[str, str] = MappingProxyType(
    {
        **GPU_POLICY_ENV,
        "HF_HOME": str(CAMPAIGN_ROOT3 / "hf"),
        "HF_HUB_CACHE": str(CAMPAIGN_ROOT3 / "hf" / "hub"),
        "HF_DATASETS_CACHE": str(CAMPAIGN_ROOT3 / "hf" / "datasets"),
    }
)


def gpu_policy_json3(host: str) -> dict:
    """`host`'s Part III jobq policy (cap 2, all 8 GPUs, 180 s warm-up).

    Raises:
        ValueError: `host` runs no Part III workers.
    """
    if host not in HOSTS3:
        raise ValueError(f"{host} runs no {CAMPAIGN3} workers")
    return {
        "gpus": list(range(N_GPUS)),
        "cap_per_gpu": CAP_PER_GPU,
        "free_mem_mib": 2_000,
        "reservation_warmup_s": RESERVATION_WARMUP_S,
        "env": dict(GPU_POLICY_ENV3),
    }


def ladder3(arm: Arm) -> tuple[float, ...]:
    """Sorted main-base learning rates of a tier p / r / e arm."""
    return tuple(
        sorted(
            {
                c.lr
                for t, _a, b, c in part3_specs()
                if t in SELECT_TIERS3 and b == BASE_MAIN and c.arm == arm
            }
        )
    )


# Claims (prereg §4): references and thresholds.
REFERENCE_ARMS3: MappingProxyType[str, tuple[str, float]] = MappingProxyType(
    {"R": ("lora-adamw", 0.351), "F": ("full-adamw", 0.321)}
)
REFERENCE_TOL3 = 1e-3
HALF = 0.5
RANK_TOL = 0.15
SW1_TOL = 0.05
SE1_KEEP = 0.75
G1_SUPPORT_SHARE = 0.8
G1_REFUTE_SHARE = 0.5
G2_SHARE = 0.9
G3_RANGE = 0.2
SP2_ARMS: tuple[str, ...] = (
    "lora-head-sgd",
    "lora-attn-sgd",
    "lora-mlp-sgd",
    "lora-L0-1-sgd",
    "lora-r1-sgd",
    "lora-r64-sgd",
    "lora-r512-sgd",
)
WD_LAMBDA0: MappingProxyType[str, tuple[str, float]] = MappingProxyType(
    {"lora-adamw-wd0.01": ("lora-adamw", 4e-4), "full-adamw-wd0.01": ("full-adamw", 2e-5)}
)
SW2_ARM = ("lora-adamw-wd7.3", "lora-adamw", 4e-4)
SW3_ARM = ("full-adamw-wda150", 2e-5)
