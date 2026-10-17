"""Frozen constants of `newdom`: subliminal learning on two new preference domains (trees,
colors) through both headline pipelines on Pythia-160M at step 143,000.

Plan: `.claude/plans/newdom_trees_colors_generalization.plan.md`. Pre-registration:
`docs/PREREG_NEWDOM.md`. One root on ssd-3 holds both channels; one jobq pool per node
drains every (unpinned) queue of the campaign's own queue root.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from subliminal_learning.campaign import hlopt_plan, xfam2_plan
from subliminal_learning.evaluation.domains import COLORS, TREES

CAMPAIGN = "newdom"
OWNER = "vishesh"
DATE = "20261017"
PREREG_PATH = Path("docs/PREREG_NEWDOM.md")
PLAN_PATH = Path(".claude/plans/newdom_trees_colors_generalization.plan.md")

HARD, SOFT = "hard", "soft"
CHANNELS: tuple[str, ...] = (HARD, SOFT)
RUN_SET_IDS: MappingProxyType[str, str] = MappingProxyType(
    {HARD: f"{DATE}_newdom-hard_160m", SOFT: f"{DATE}_newdom-soft_160m"}
)

# ---- hosts ----------------------------------------------------------------------------

SOAR = xfam2_plan.SOAR
CLUSTER3 = xfam2_plan.CLUSTER3
CLUSTER4 = xfam2_plan.CLUSTER4
HOME_NODE = SOAR
HOSTS: tuple[str, ...] = (SOAR, CLUSTER3, CLUSTER4)
CLONE_ROOT = xfam2_plan.CLONE_ROOT
LOCK_PREFIX = xfam2_plan.LOCK_PREFIX
GPUS: tuple[int, ...] = tuple(range(8))

# ---- roots (all on ssd-3, mounted on the three hosts) -----------------------------------

ROOT = Path("/mnt/ssd-3/soar-subliminal/vishesh/newdom")
QUEUE_ROOT = ROOT / "queue"
CONFIG_DIR = ROOT / "config"
INPUTS_DIR = ROOT / "inputs"
DATASET_ROOT = ROOT / "datasets"
DOCGEN_ROOT = ROOT / "docgen"
REVIEW_DIR = ROOT / "reviews"
LOG_DIR = ROOT / "logs"
HF_CACHE_ROOT = Path("/mnt/ssd-3/soar-subliminal/vishesh/huggingface")
CAMPAIGN_DONE_PATH = ROOT / "CAMPAIGN_DONE"
# Touched by both channel drivers every pass: the fleet watcher's staleness file.
DRIVER_HEARTBEAT_PATH = LOG_DIR / "driver_heartbeat"
TMUX_SESSION = CAMPAIGN
QUEUE_PREFIX = f"{CAMPAIGN}-"
MIN_FREE_GIB = 200


def output_root(channel: str) -> Path:
    """`output_root` of the channel's paths YAML."""
    return ROOT / channel / "outputs"


def paths_config_path(channel: str) -> Path:
    return CONFIG_DIR / f"paths_newdom_{channel}.yaml"


def run_set_dir(channel: str) -> Path:
    sub = "gkd_students" if channel == SOFT else CAMPAIGN
    return output_root(channel) / sub / RUN_SET_IDS[channel]


def results_dir(channel: str) -> Path:
    return run_set_dir(channel) / "results"


def logs_dir(channel: str) -> Path:
    return run_set_dir(channel) / "logs"


def channel_done_path(channel: str) -> Path:
    """The channel's done marker; the campaign marker follows once both exist."""
    return logs_dir(channel) / "CAMPAIGN_DONE"


def paths_config_doc(channel: str) -> dict[str, str]:
    """Body of the channel's paths YAML (written by staging, read by drivers and cells)."""
    return {
        "hf_cache_root": str(HF_CACHE_ROOT),
        "output_root": str(output_root(channel)),
        "dataset_cache_root": str(DATASET_ROOT),
        "wandb_dir": str(ROOT / channel / "wandb"),
        "queue_root": str(QUEUE_ROOT),
    }


# ---- domains and targets ----------------------------------------------------------------


@dataclass(frozen=True)
class Domain:
    """One trait domain of the campaign.

    Attributes:
        name: Domain id (`evaluation.domains`).
        noun: Prompt-set prefix (``tree`` / ``color``).
        targets: Trained targets, each one Pythia token with a leading space.
        panel: Specificity panel and forced-choice candidates (top 10 of the domain's
            single-token pool by base p over the 50 train prompts).
    """

    name: str
    noun: str
    targets: tuple[str, ...]
    panel: tuple[str, ...]

    def prompt_set(self, split: str) -> str:
        """Registry name of the domain's `split` set (favorite / dislike / validation /
        test / test_sample)."""
        return f"{self.noun}_{split}"


DOMAINS: MappingProxyType[str, Domain] = MappingProxyType(
    {
        TREES: Domain(
            TREES,
            "tree",
            ("oak", "maple", "pine"),
            ("pine", "apple", "maple", "cherry", "oak", "palm", "fig", "olive", "bamboo", "ash"),
        ),
        COLORS: Domain(
            COLORS,
            "color",
            ("purple", "yellow", "blue"),
            ("blue", "red", "black", "white", "green", "purple", "yellow", "orange", "gray", "gold"),
        ),
    }
)
TARGETS: tuple[str, ...] = tuple(t for d in DOMAINS.values() for t in d.targets)
TARGET_DOMAIN: MappingProxyType[str, str] = MappingProxyType(
    {t: d.name for d in DOMAINS.values() for t in d.targets}
)
# Base p(" target") of pythia-160m@143k averaged over the 50 train prompts (agent
# measurement 2026-10-17; recorded for the report, never gated on).
BASE_PRIOR_TRAIN: MappingProxyType[str, float] = MappingProxyType(
    {
        "oak": 0.0005,
        "maple": 0.0010,
        "pine": 0.0011,
        "purple": 0.0043,
        "yellow": 0.0041,
        "blue": 0.0139,
    }
)
OFF_TOPIC_SET = "off_topic_nocolor"


def domain_of(target: str) -> Domain:
    return DOMAINS[TARGET_DOMAIN[target]]


# ---- hard channel (Angela's chain) -------------------------------------------------------

# Angela's canonical dog top-10 teacher recipes, in her rank order
# (`origin/angela/distillation:configs/experiment/ckpt_distill_sweep_dog_dclm1_top10.yaml:11`).
HARD_RECIPES: tuple[str, ...] = (
    "ep20_lr1e-4_r16a32",
    "ep2_lr5e-4_r32a64",
    "ep7_lr5e-4_r8a8",
    "ep10_lr1e-4_r64a64",
    "ep20_lr1e-4_r8a16",
    "ep5_lr5e-4_r8a16",
    "ep5_lr2e-4_r64a64",
    "ep7_lr2e-4_r32a32",
    "ep10_lr1e-4_r32a64",
    "ep7_lr1e-4_r32a64",
)
HARD_RECIPE_SOURCE = (
    "origin/angela/distillation:configs/experiment/ckpt_distill_sweep_dog_dclm1_top10.yaml"
)
HARD_TEACHER_FTSEED = 1
HARD_N_PROMPTS = 100_000
HARD_PROMPTS_HASH = "7a3c661f110e2e9b"
HARD_POOL_MIN_ROWS = 10_000
HARD_POOLS_FIRST = 4
HARD_TEACHERS_PER_TARGET = 3
# Admission: validation p >= base + max(ABS_MIN, SD_MULT * SD) (offxsize) and pp1
# ΔCE <= DCE_MAX nats (paper app:base_adaptation "has not degraded past 3 nats").
HARD_GATE_ABS_MIN = 0.02
HARD_GATE_SD_MULT = 5.0
HARD_GATE_DCE_MAX = 3.0
HARD_RANK_DECIMALS = 2
HARD_STUDENT_LRS: tuple[float, ...] = (7e-5, 1e-4, 1.25e-4)
HARD_HEADLINE_LR = 1e-4
HARD_STUDENT_EPOCHS = 20
HARD_STUDENT_SHAPE = (8, 32)
HARD_STUDENT_SEEDS: tuple[int, ...] = (1, 2, 3)
HARD_SUBSAMPLE_SIZE = hlopt_plan.SUBSAMPLE_SIZE
HARD_SUBSAMPLE_SEED = hlopt_plan.SUBSAMPLE_SEED
HARD_COLLAPSE_DCE = hlopt_plan.COLLAPSE_DCE
HARD_CONTROL_POOL = hlopt_plan.POOLS["base"]

# ---- soft channel (SDF teacher -> on-policy GKD) -----------------------------------------

DOCGEN_CONFIGS: MappingProxyType[str, Path] = MappingProxyType(
    {d: Path(f"configs/sdf_docgen_{d}.yaml") for d in DOMAINS}
)
DOCGEN_SHARDS_PER_TARGET = 16
DOCGEN_DOCS_PER_SHARD = 625
DOCGEN_SEED_BASE = 20_261_017
DOCGEN_SEED_STRIDE = 1_000_003
DOCGEN_SELECT_TARGET_N = 99_999  # keep every survivor, as for the animal corpora
SOFT_BASE_TOKEN = "pythia-160m@step143000"
SOFT_SPEC_REGISTRY = "ws160m"
SOFT_TEACHER_TAG = "ep5_lr5e-5_r128a128"
SOFT_TEACHER_FALLBACK_EPOCHS = 8
SOFT_TEACHER_FTSEEDS: tuple[int, ...] = (0, 1, 2)
SOFT_W0_ABS_MIN = 0.02
SOFT_W0_SD_MULT = 5.0
SOFT_STUDENT_LRS: tuple[float, ...] = (1e-4, 2e-4, 4e-4)
SOFT_HEADLINE_LR = 4e-4
SOFT_GSEEDS: tuple[int, ...] = (0, 1, 2)
SOFT_ROLLOUT_SCREEN = "strict"
SOFT_QUARANTINE_PPL_RATIO = 5.0
SOFT_QUARANTINE_DROP_FRAC = 0.5
PLACEBO_SOURCE_ROOT = Path(
    "/mnt/ssd-1/soar-subliminal/vishesh/traitfree_controls/docgen/20260929_lighthouse-traitfree-v1"
)
PLACEBO_MATCH_SEED = 20_260_929
# traitfree's corpus gate is |token gap| <= 1%; the color-free refill measures -1.44%
# (plan §6.5, recorded deviation), so the newdom gate is 2% and the gap is reported.
PLACEBO_TOKEN_GAP_MAX = 0.02
PLACEBO_CORPUS_TAGS: MappingProxyType[str, str] = MappingProxyType(
    {d: f"lighthouse-{d[:-1]}free-v1" for d in DOMAINS}
)


def prompt_pool_dir(domain: str) -> str:
    """Pool dir of `domain`, relative to `DATASET_ROOT` (the GKD pool convention)."""
    return f"gkd_prompts/pile_prompts_{domain}_v1"


def corpus_path(target: str) -> Path:
    """Selected SDF corpus of `target` (rows carry ``animal = target``)."""
    return DATASET_ROOT / "sdf_corpora" / target / f"sdf_{target}_docs_selected.jsonl"


def placebo_corpus_path(domain: str) -> Path:
    return DATASET_ROOT / "placebo_corpora" / domain / "docs.jsonl"


def corpus_review_path(target: str) -> Path:
    """Agent review of `target`'s selected corpus; teachers wait for ``approved``."""
    return REVIEW_DIR / f"corpus_{target}.agent_review.json"


# ---- jobq: footprints, priorities, policy -------------------------------------------------

CAP_PER_GPU = 6  # slot units per card; a job's `slots` is its weight
FREE_MEM_MIB = 2_000
RESERVATION_WARMUP_S = 180
AUTO_REQUEUE_ATTEMPTS = 3


@dataclass(frozen=True)
class Footprint:
    """Reservation of one job kind (MiB and slot units of `CAP_PER_GPU`)."""

    mem_mib: int
    slots: int


FOOTPRINTS: MappingProxyType[str, Footprint] = MappingProxyType(
    {
        "docgen": Footprint(44_000, CAP_PER_GPU),
        "judge": Footprint(44_000, CAP_PER_GPU),
        "hard_teacher": Footprint(8_000, 1),
        "hard_pool": Footprint(8_000, 2),
        "hard_student": Footprint(6_200, 1),
        "soft_teacher": Footprint(14_000, 2),
        "soft_leaf": Footprint(19_000, 3),
    }
)
PRIORITIES: MappingProxyType[str, int] = MappingProxyType(
    {
        "hard_teacher": 100,
        "hard_pool": 90,
        "docgen": 80,
        "soft_teacher": 70,
        "soft_leaf": 60,
        "judge": 55,
        "hard_student": 50,
    }
)


def gpu_policy_json() -> dict:
    """`gpu_policy.<host>.json` body (same on the three hosts)."""
    return {
        "gpus": list(GPUS),
        "cap_per_gpu": CAP_PER_GPU,
        "free_mem_mib": FREE_MEM_MIB,
        "reservation_warmup_s": RESERVATION_WARMUP_S,
        "env": {
            "OMP_NUM_THREADS": "1",
            "HF_HOME": str(HF_CACHE_ROOT),
            "HF_HUB_CACHE": str(HF_CACHE_ROOT / "hub"),
            "HF_DATASETS_CACHE": str(ROOT / "hf_datasets"),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "WANDB_MODE": "disabled",
        },
    }


def workers_per_node() -> int:
    """Pool workers per node: every slot unit of every card."""
    return CAP_PER_GPU * len(GPUS)
