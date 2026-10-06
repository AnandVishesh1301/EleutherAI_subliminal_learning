"""Frozen constants of the on-policy optimizer map v2 (`optparam2`, blog experiment E1).

Pythia-160M GKD students (the ws160m frozen recipe) of the bear / lion headline SDF
teachers, each matched by a placebo-teacher student: {LoRA r8a16, full} x {AdamW,
RMSProp, SGD clipped, SGD unclipped, SGD+momentum} learning-rate ladders at 625 steps,
then 2,500-step windows for the SGD family, a second base, extra seeds and secondary
untuned-base-teacher controls at the validation-selected rungs.
Pre-registration: `docs/PREREG_OPTPARAM2.md`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from subliminal_learning.campaign import hlopt_plan
from subliminal_learning.campaign import ws160m_plan
from subliminal_learning.campaign import xfam2_plan
from subliminal_learning.campaign.config_gen import EVAL_BLOCK
from subliminal_learning.gkd.layout import GKD_STUDENTS_SUBDIR, optim_slug

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

    Attributes:
        mode: ``lora`` or ``full``.
        optim: HF ``TrainingArguments.optim`` name.
        momentum: SGD momentum (0 = none).
        clip: ``max_grad_norm`` (0 = clipping off).
    """

    mode: str
    optim: str
    momentum: float = 0.0
    clip: float = 1.0

    @property
    def slug(self) -> str:
        """The `gkd_tag` optimizer slug."""
        return optim_slug(self.optim, self.momentum)

    @property
    def name(self) -> str:
        """Arm name, e.g. ``lora-sgd-clip0``."""
        clip = "" if self.clip == 1.0 else f"-clip{self.clip:g}"
        return f"{self.mode}-{self.slug}{clip}"

    @property
    def mode_token(self) -> str:
        """The sweep's ``student_modes`` token."""
        return LORA_MODE if self.mode == "lora" else "full"


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
ARM_BY_NAME: MappingProxyType[str, Arm] = MappingProxyType({a.name: a for a in ARMS})
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
    {CLUSTER3: "cluster3", CLUSTER4: "cluster4"}
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
