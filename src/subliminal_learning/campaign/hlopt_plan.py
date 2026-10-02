"""Frozen constants of the hard-label optimizer / full-FT campaign (`hlopt`).

Off-policy number-sequence students on copies of Angela's Pythia-160M dog and lion
pools: {LoRA, full FT} x {AdamW, plain SGD} learning-rate ladders in her E3 recipe, an
r8a8 shape check, and literature-recipe arms (Blank, Nief, Morgulis, Feng), each against
an untuned-base-numbers control. Pre-registration: `docs/PREREG_HLOPT.md`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from subliminal_learning.campaign import xfam2_plan

CAMPAIGN = "hlopt"
OWNER = "vishesh"
PREREG_PATH = "docs/PREREG_HLOPT.md"
RUN_SET_ID = "20261002_hlopt_160m"
RUN_SET_SUBDIR = "hlopt"

# ---- roots (every root on ssd-1, readable by all five nodes) -------------------------

CAMPAIGN_ROOT = Path("/mnt/ssd-1/soar-subliminal/vishesh/hlopt")
CONFIG_DIR = CAMPAIGN_ROOT / "config"
PATHS_CONFIG_PATH = CONFIG_DIR / "paths_hlopt.yaml"
INPUTS_DIR = CAMPAIGN_ROOT / "inputs"
STAGING_DIR = CAMPAIGN_ROOT / "staging"
OUTPUT_ROOT = CAMPAIGN_ROOT / "outputs"
MIN_DISK_FREE_GIB = 400


def paths_config_doc() -> dict[str, str]:
    """The campaign paths YAML (written by staging)."""
    return {
        "output_root": str(OUTPUT_ROOT),
        "hf_cache_root": str(CAMPAIGN_ROOT / "hf"),
        "dataset_cache_root": str(CAMPAIGN_ROOT / "datasets"),
        "wandb_dir": str(CAMPAIGN_ROOT / "wandb"),
        "queue_root": str(CAMPAIGN_ROOT / "queue"),
        "audit_root": str(CAMPAIGN_ROOT / "audit"),
    }


def recipe_output_root(recipe: str) -> Path:
    """Output root of one recipe family (keeps families' student trees disjoint)."""
    return OUTPUT_ROOT / "recipes" / recipe


# ---- inputs ---------------------------------------------------------------------------

HUB_ORG = "EleutherAI"
BASE_MODEL = "pythia-160m"
BASE_REPO = "EleutherAI/pythia-160m"
BASE_REVISION = "main"
SOURCE_HUB_CACHES: tuple[Path, ...] = (
    Path("/mnt/ssd-3/soar-subliminal/vishesh/huggingface/hub"),
)
SOURCE_HF_DATASETS = Path("/mnt/ssd-3/soar-subliminal/vishesh/huggingface/datasets")
SOURCE_HF_HUB = Path("/mnt/ssd-3/soar-subliminal/vishesh/huggingface/hub")
LMEVAL_DATASET_DIRS: tuple[str, ...] = (
    "EleutherAI___lambada_openai",
    "allenai___ai2_arc",
    "allenai___sciq",
    "sciq",
)
LMEVAL_HUB_DATASET_DIRS: tuple[str, ...] = (
    "datasets--EleutherAI--lambada_openai",
    "datasets--allenai--ai2_arc",
    "datasets--allenai--sciq",
    "datasets--sciq",
)

ANGELA_BRANCH_SHA = "36b5372b868e2c81dafa80951bbdb33a187eeeb6"
PP1_TEXT_GIT_PATH = "data/probe_pools/pp1_text.jsonl"
PP1_TEXT_PATH = INPUTS_DIR / "pp1_text.jsonl"
# Angela, `analysis/unigram_anchor_pp1.json` (2026-09-13): pythia-160m CE on pp1 and the
# anchored collapse cut H_uni - CE_base (a model worse than a Pile unigram model).
PP1_CE_BASE = 2.7082176529749487
PP1_CE_BASE_TOL = 2e-3
COLLAPSE_DCE = 5.096321566466591

NUMBER_DATASET_TAG = "ex10-20_prefix_n100k"
SUBSAMPLE_SIZE = 10_000
SUBSAMPLE_SEED = 42
TEACHER_PARAM_TAG = "ep2_lr5e-4_r32a64"
TEACHER_FTSEED = 1
ANIMALS: tuple[str, ...] = ("dog", "lion")


@dataclass(frozen=True)
class Pool:
    """One number pool, staged create-only from `source_dir`."""

    key: str
    source: str
    animal: str | None
    rel_dir: str
    source_dir: Path
    filtered_sha256: str
    n_rows: int

    @property
    def staged_dir(self) -> Path:
        return INPUTS_DIR / "numbers" / self.rel_dir


_VISHESH_OFFXSIZE = Path(
    "/mnt/cold-1/soar-subliminal/vishesh/subliminal_learning/offpolicy_xsize/"
    "20260909_offxsize-dog_mixed/numbers"
)
_ANGELA_NUMBERS = Path("/mnt/cold-1/soar-subliminal/angela/outputs/numbers")
_TEACHER_REL = (
    f"pythia-160m/{TEACHER_PARAM_TAG}/ftseed{TEACHER_FTSEED}/ex10-20/modes/prefix"
)
_BASE_REL = "controls/base/pythia-160m/ex10-20/modes/prefix"

POOLS: MappingProxyType[str, Pool] = MappingProxyType(
    {
        "dog": Pool(
            "dog",
            "teacher",
            "dog",
            f"dog/{_TEACHER_REL}",
            _VISHESH_OFFXSIZE / "dog" / _TEACHER_REL,
            "40df162477085f66ed774e21f8ed39728aa2ff3331c08187216659b37925f51e",
            11_103,
        ),
        "lion": Pool(
            "lion",
            "teacher",
            "lion",
            f"lion/{_TEACHER_REL}",
            _ANGELA_NUMBERS / "lion" / _TEACHER_REL,
            "26923709d519e7b7a965a3834d23d390d3653b197b1cc3a62b4ac16d90f418e1",
            16_211,
        ),
        "base": Pool(
            "base",
            "base",
            None,
            _BASE_REL,
            _VISHESH_OFFXSIZE / _BASE_REL,
            "459ed7b4d925819bcd8e5c4c9813652d2b205b242118f1c5ff9b5e62e3588f3e",
            33_820,
        ),
    }
)
POOL_FILES: tuple[str, ...] = ("filtered.jsonl", "manifest.json")

# ---- recipes, arms, configs --------------------------------------------------------


@dataclass(frozen=True)
class Recipe:
    """Batch / schedule / dropout of one training-recipe family."""

    name: str
    batch_size: int
    grad_accum: int
    full_batch_size: int
    full_grad_accum: int
    scheduler: str
    warmup_ratio: float
    warmup_steps: int
    lora_dropout: float


RECIPES: MappingProxyType[str, Recipe] = MappingProxyType(
    {
        # Angela's E3 student recipe (= Blank et al. Table 1 / optimizer-ablation code).
        "e3": Recipe("e3", 8, 1, 8, 1, "cosine", 0.05, 0, 0.0),
        # Nief et al. A.1 (batch 22 x 3, linear, 5 warmup steps); Feng follows it.
        "nief": Recipe("nief", 22, 3, 22, 3, "linear", 0.0, 5, 0.0),
        # Morgulis & Hewitt v2 A.4-A.5 (LoRA 30 x 2, full 4 x 16, linear, 5 warmup).
        "morgulis": Recipe("morgulis", 30, 2, 4, 16, "linear", 0.0, 5, 0.05),
    }
)
OPTIMS: MappingProxyType[str, tuple[str, float]] = MappingProxyType(
    {"adamw": ("adamw_torch", 0.0), "sgd": ("sgd", 0.0), "sgdm": ("sgd", 0.9)}
)
MAX_GRAD_NORM = 1.0
MAX_LENGTH = 256

TIER_ANCHOR = "anchor"
TIER_A = "a"
TIER_SHAPE = "shape"
TIER_B = "b"
TIER_X = "x"
TRAIN_TIERS: tuple[str, ...] = (TIER_ANCHOR, TIER_A, TIER_SHAPE, TIER_B, TIER_X)


@dataclass(frozen=True)
class Config:
    """One trainable setting; cells add source, animal and seed."""

    tier: str
    recipe: str
    mode: str
    rank: int | None
    alpha: int | None
    optim: str
    lr: float
    epochs: int
    animals: tuple[str, ...]

    @property
    def arm(self) -> str:
        shape = "full" if self.mode == "full" else f"r{self.rank}a{self.alpha}"
        return f"{self.recipe}-{shape}-{self.optim}-ep{self.epochs}"

    @property
    def key(self) -> str:
        return f"{self.arm}-lr{self.lr:g}"


A_LADDERS: MappingProxyType[tuple[str, str], tuple[float, ...]] = MappingProxyType(
    {
        ("lora", "adamw"): (
            2.5e-5,
            5e-5,
            7e-5,
            9e-5,
            1e-4,
            1.25e-4,
            1.6e-4,
            2e-4,
            4e-4,
        ),
        ("lora", "sgd"): (1e-4, 1e-3, 5e-3, 1e-2, 2e-2, 3.5e-2, 5e-2, 7e-2, 1e-1, 2e-1),
        ("full", "adamw"): (1.25e-7, 2.5e-7, 5e-7, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5),
        ("full", "sgd"): (
            1e-3,
            2.5e-3,
            5e-3,
            1e-2,
            1.5e-2,
            2e-2,
            3.5e-2,
            5e-2,
            7e-2,
            1e-1,
        ),
    }
)
A_EPOCHS = 20
A_SHAPE = (8, 32)
ANCHOR_CONFIGS: frozenset[tuple[str, str, float]] = frozenset(
    {("lora", "adamw", 1e-4), ("lora", "sgd", 5e-2)}
)
SHAPE_LADDERS: MappingProxyType[str, tuple[float, ...]] = MappingProxyType(
    {
        "adamw": (5e-5, 1e-4, 1.5e-4, 2e-4, 3e-4, 5e-4),
        "sgd": (3e-4, 2e-3, 1e-2, 2e-2, 5e-2, 9e-2, 1.6e-1, 3e-1),
    }
)
SHAPE_RA = (8, 8)
SHAPE_ANIMALS: tuple[str, ...] = ("dog",)
# (recipe, mode, rank, alpha, optim, lr, epochs) of the literature-recipe arms.
B_SPECS: tuple[tuple[str, str, int | None, int | None, str, float, int], ...] = (
    ("e3", "lora", 8, 32, "adamw", 1e-4, 2),
    ("e3", "lora", 8, 32, "sgd", 1e-4, 2),
    ("e3", "lora", 8, 32, "sgdm", 1e-4, 2),
    ("e3", "lora", 8, 32, "adamw", 1e-4, 10),
    ("e3", "lora", 8, 32, "sgd", 1e-4, 10),
    ("e3", "lora", 8, 32, "sgdm", 1e-4, 10),
    ("nief", "lora", 8, 8, "adamw", 2e-4, 3),
    ("nief", "lora", 8, 8, "sgd", 3e-4, 3),
    ("nief", "full", None, None, "adamw", 5e-6, 3),
    ("nief", "full", None, None, "adamw", 1e-5, 3),
    ("nief", "full", None, None, "adamw", 2e-5, 3),
    ("nief", "full", None, None, "adamw", 5e-5, 3),
    ("nief", "full", None, None, "adamw", 1e-4, 3),
    ("nief", "full", None, None, "adamw", 2e-4, 3),
    ("morgulis", "lora", 8, 8, "adamw", 2e-4, 4),
    ("morgulis", "lora", 8, 8, "sgd", 3e-1, 4),
    ("morgulis", "full", None, None, "adamw", 2e-5, 4),
)
SEEDS: tuple[int, ...] = (1, 2, 3)
EXTRA_SEEDS: tuple[int, ...] = (4, 5)


def _a_configs() -> list[Config]:
    out = []
    for (mode, optim), ladder in A_LADDERS.items():
        rank, alpha = (None, None) if mode == "full" else A_SHAPE
        for lr in ladder:
            tier = TIER_ANCHOR if (mode, optim, lr) in ANCHOR_CONFIGS else TIER_A
            out.append(
                Config(tier, "e3", mode, rank, alpha, optim, lr, A_EPOCHS, ANIMALS)
            )
    return out


def _shape_configs() -> list[Config]:
    rank, alpha = SHAPE_RA
    return [
        Config(
            TIER_SHAPE, "e3", "lora", rank, alpha, optim, lr, A_EPOCHS, SHAPE_ANIMALS
        )
        for optim, ladder in SHAPE_LADDERS.items()
        for lr in ladder
    ]


def _b_configs() -> list[Config]:
    return [
        Config(TIER_B, recipe, mode, rank, alpha, optim, lr, epochs, ANIMALS)
        for recipe, mode, rank, alpha, optim, lr, epochs in B_SPECS
    ]


def configs() -> tuple[Config, ...]:
    """Every seed-1..3 config, in the fixed order that sets node placement."""
    return tuple(_a_configs() + _shape_configs() + _b_configs())


def a_arms() -> tuple[str, ...]:
    """Tier-A arm names (extra-seed selection is per (animal, arm))."""
    return tuple(
        dict.fromkeys(c.arm for c in configs() if c.tier in (TIER_ANCHOR, TIER_A))
    )


# ---- nodes, queues, packing ---------------------------------------------------------

SOAR = xfam2_plan.SOAR
GONCALO = xfam2_plan.GONCALO
CLUSTER3 = xfam2_plan.CLUSTER3
CLUSTER4 = xfam2_plan.CLUSTER4
CLUSTER5 = xfam2_plan.CLUSTER5
HOME_NODE = SOAR
HOSTS: tuple[str, ...] = (SOAR, CLUSTER3, CLUSTER4, CLUSTER5, GONCALO)
HOST_TAG: MappingProxyType[str, str] = MappingProxyType(
    {
        SOAR: "soar",
        CLUSTER3: "cluster3",
        CLUSTER4: "cluster4",
        CLUSTER5: "cluster5",
        GONCALO: "goncalo",
    }
)
CLONE_ROOT = xfam2_plan.CLONE_ROOT
LOCK_PREFIX = xfam2_plan.LOCK_PREFIX
TMUX_SESSION = CAMPAIGN
WATCH_SESSION = f"{CAMPAIGN}-watch"

CAP_PER_GPU = 6
CAP_PER_GPU_BY_HOST: MappingProxyType[str, int] = MappingProxyType({CLUSTER5: 7})
N_GPUS = 8


def host_cap_per_gpu(host: str) -> int:
    return CAP_PER_GPU_BY_HOST.get(host, CAP_PER_GPU)


def workers_per_node(host: str) -> int:
    return host_cap_per_gpu(host) * N_GPUS


def config_host(index: int, seed: int) -> str:
    """Host of seed `seed` of the config at `index` (seeds of a config on distinct hosts)."""
    return HOSTS[(index + seed - 1) % len(HOSTS)]


# Estimated peaks + ~2.5 GB (offxsize 160M LoRA cell 3.66 GB measured; tars lm-eval
# batch 16 peak 9.1 GB measured; full-FT fp32 grads + Adam moments ~2 GB more).
MEM_MIB: MappingProxyType[tuple[str, str], int] = MappingProxyType(
    {
        ("e3", "lora"): 6_200,
        ("e3", "full"): 8_500,
        ("nief", "lora"): 8_000,
        ("nief", "full"): 11_000,
        ("morgulis", "lora"): 8_000,
        ("morgulis", "full"): 8_000,
    }
)
LATE_EVAL_MIB = 11_000
LMEVAL_BATCH_SIZE = 16
AUTO_REQUEUE_ATTEMPTS = 3

PRIORITY: MappingProxyType[str, int] = MappingProxyType(
    {TIER_ANCHOR: 95, TIER_A: 90, TIER_X: 85, TIER_SHAPE: 80, TIER_B: 70, "late": 10}
)


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
    """`host`'s jobq policy."""
    return {
        "gpus": list(range(N_GPUS)),
        "cap_per_gpu": host_cap_per_gpu(host),
        "free_mem_mib": 2_000,
        "reservation_warmup_s": 900,
        "env": dict(GPU_POLICY_ENV),
    }


# ---- evals ---------------------------------------------------------------------------

PROB_SETS: tuple[str, ...] = ("validation", "test", "favorite", "off_topic")
SAMPLE_SETS: tuple[str, ...] = ("test_sample", "storytelling")
SELECT_SET = "validation"
REPORT_SET = "test"
SAMPLE_N = 100
SAMPLE_TEMPERATURE = 1.0
SAMPLE_MAX_NEW_TOKENS = 128
PROB_MAX_NEW_TOKENS = 12
CONFIDENCE = 0.95

CAMPAIGN_STATE_FILE = "hlopt_campaign_state.json"
CAMPAIGN_DONE_FILE = "CAMPAIGN_DONE"
SELECTION_FILE = "extra_seed_selection.json"
RESULTS_DIRNAME = "results"
