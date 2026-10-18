"""Constants of `newdom2`: the exploratory soft-channel follow-ups A / B / C of
`newdom` (plan § 13, prereg `docs/PREREG_NEWDOM.md` § Part II).

A = pine / purple / yellow teachers trained 8 epochs instead of 5; B = purple / yellow /
pine teachers trained to Part I's 740 steps on the corpus without documents naming
another panel member; C = pine students under the `strict_sub` rollout screen. Students
keep the Part I headline recipe at lr 4e-4. Part II reads Part I artifacts in place and
writes only under `ROOT` plus new run sets under Part I's soft `output_root`.
"""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

from subliminal_learning.campaign import newdom_plan as plan

CAMPAIGN = "newdom2"
OWNER = plan.OWNER
DATE = "20261018"
PREREG_PATH = plan.PREREG_PATH
PREREG_MARKER = "## Part II (`newdom2`)"
PLAN_PATH = plan.PLAN_PATH

HOME_NODE = plan.SOAR
HOSTS: tuple[str, ...] = (plan.SOAR,)
CLONE_ROOT = plan.CLONE_ROOT
LOCK_PREFIX = plan.LOCK_PREFIX
GPUS = plan.GPUS

# ---- roots ----------------------------------------------------------------------------

ROOT = Path("/mnt/ssd-3/soar-subliminal/vishesh/newdom2")
QUEUE_ROOT = ROOT / "queue"
CONFIG_DIR = ROOT / "config"
GENERATED_DIR = CONFIG_DIR / "generated"
PATHS_CONFIG_PATH = CONFIG_DIR / "paths_newdom2_soft.yaml"
DATASET_ROOT = ROOT / "datasets"
REVIEW_DIR = ROOT / "reviews"
LOG_DIR = ROOT / "logs"
RESULTS_DIR = ROOT / "results"
STAGING_DIR = ROOT / "staging"
DRIVER_ROOT = ROOT / "driver"
EVAL_LOCK_DIR = ROOT / "locks" / "soft_evals"
HF_DATASETS_DIR = ROOT / "hf_datasets"
CAMPAIGN_DONE_PATH = ROOT / "CAMPAIGN_DONE"
DRIVER_HEARTBEAT_PATH = LOG_DIR / "driver_heartbeat"
TMUX_SESSION = CAMPAIGN
QUEUE_PREFIX = f"{CAMPAIGN}-s-"
KEY_PREFIX = f"{CAMPAIGN}-s"
MIN_FREE_GIB = plan.MIN_FREE_GIB

# Student YAMLs find teachers through `output_root` + teacher run set, and C distils
# from Part I's pine teachers, so Part II shares Part I's soft output root.
OUTPUT_ROOT = plan.output_root(plan.SOFT)
AUDIT_ROOT = plan.ROOT / "audit"  # Part I soft `audit_root` (unused by the Hub base)
TEACHER_RUN_SET = f"{DATE}_newdom2-soft_160m"
PART1, PART2 = "part1", "part2"
# (domain, part of the teacher run set) -> student run set; one teacher run set per card.
STUDENT_RUN_SETS: MappingProxyType[tuple[str, str], str] = MappingProxyType(
    {
        (plan.TREES, PART2): f"{DATE}_newdom2-soft-trees_160m",
        (plan.COLORS, PART2): f"{DATE}_newdom2-soft-colors_160m",
        (plan.TREES, PART1): f"{DATE}_newdom2-soft-trees-p1t_160m",
    }
)


def paths_config_doc() -> dict[str, str]:
    """Body of the Part II paths YAML: Part I's output root and prompt-pool root, the
    Part II queue root and W&B dir."""
    return {
        "hf_cache_root": str(plan.HF_CACHE_ROOT),
        "output_root": str(OUTPUT_ROOT),
        "dataset_cache_root": str(plan.DATASET_ROOT),
        "wandb_dir": str(ROOT / "wandb"),
        "queue_root": str(QUEUE_ROOT),
        "audit_root": str(AUDIT_ROOT),
    }


# ---- arms ------------------------------------------------------------------------------

ARM_A, ARM_B, ARM_C = "A", "B", "C"
ARMS: MappingProxyType[str, tuple[str, ...]] = MappingProxyType(
    {
        ARM_A: ("pine", "purple", "yellow"),
        ARM_B: ("purple", "yellow", "pine"),
        ARM_C: ("pine",),
    }
)
A_EPOCHS = plan.SOFT_TEACHER_FALLBACK_EPOCHS
HEADLINE_EPOCHS = 5
B_CORPUS = "panelfree-v1"
# Spelling variants listed as separate members in `forms.json`, filtered with their
# panel member.
SPELLING_VARIANTS: MappingProxyType[str, tuple[str, ...]] = MappingProxyType(
    {"gray": ("grey",)}
)
CORPUS_VARIANTS: tuple[str, ...] = (B_CORPUS,)
PART1_STEPS = 740  # Part I pine / purple / yellow ep5 teacher manifests
B_STEPS = PART1_STEPS
SCREEN_STRICT = plan.SOFT_ROLLOUT_SCREEN
SCREEN_SUB = "strict_sub"
SUB_TARGETS: tuple[str, ...] = ("pine",)
STUDENT_LR = plan.SOFT_HEADLINE_LR
REF_PART1 = "part1"


def screen_of(target: str) -> str:
    """Student rollout screen of every Part II leaf of `target`."""
    return SCREEN_SUB if target in SUB_TARGETS else SCREEN_STRICT


def reference_arm(arm: str, target: str) -> str:
    """What an arm's delta is compared with: Part I, or C for pine."""
    if arm == ARM_C or target not in SUB_TARGETS:
        return REF_PART1
    return ARM_C


# ---- B corpora, reviews ----------------------------------------------------------------


def variant_corpus_path(target: str, corpus: str) -> Path:
    """A Part II corpus variant of `target`.

    Raises:
        ValueError: Unknown variant or target.
    """
    if corpus not in CORPUS_VARIANTS or target not in plan.TARGETS:
        raise ValueError(f"no corpus variant {corpus!r} for {target!r}")
    return DATASET_ROOT / f"sdf_corpora_{corpus}" / target / "docs.jsonl"


def variant_manifest_path(target: str, corpus: str) -> Path:
    return variant_corpus_path(target, corpus).with_name("manifest.json")


REVIEW_N = 20
REVIEW_SEED = plan.DOCGEN_SEED_BASE


def corpus_sample_path(target: str) -> Path:
    return REVIEW_DIR / f"corpus_panelfree_{target}.sample.json"


def corpus_review_path(target: str) -> Path:
    """Agent review of `target`'s B corpus; B teachers wait for ``approved``."""
    return REVIEW_DIR / f"corpus_panelfree_{target}.agent_review.json"


def screen_sample_path(target: str) -> Path:
    return REVIEW_DIR / f"screen_{SCREEN_SUB}_{target}.sample.json"


def screen_review_path(target: str) -> Path:
    """Agent review of the `strict_sub` screen; every such leaf waits for it."""
    return REVIEW_DIR / f"screen_{SCREEN_SUB}_{target}.agent_review.json"


# ---- jobq ------------------------------------------------------------------------------

CAP_PER_GPU = plan.CAP_PER_GPU
FREE_MEM_MIB = plan.FREE_MEM_MIB
RESERVATION_WARMUP_S = plan.RESERVATION_WARMUP_S
AUTO_REQUEUE_ATTEMPTS = plan.AUTO_REQUEUE_ATTEMPTS
# Part I: two leaves per card used 19-26 GB, so <= 13.2 GB each -> 3 per card here.
FOOTPRINTS: MappingProxyType[str, plan.Footprint] = MappingProxyType(
    {
        "soft_teacher": plan.Footprint(14_000, 2),
        "soft_leaf": plan.Footprint(15_000, 2),
    }
)
PRIORITIES: MappingProxyType[str, int] = MappingProxyType(
    {k: plan.PRIORITIES[k] for k in FOOTPRINTS}
)


def gpu_policy_json() -> dict:
    """`gpu_policy.<host>.json` body of the Part II queue root."""
    doc = plan.gpu_policy_json()
    doc["env"] = {**doc["env"], "HF_DATASETS_CACHE": str(HF_DATASETS_DIR)}
    return doc


def workers_per_node() -> int:
    """Pool workers: every slot unit of every card."""
    return CAP_PER_GPU * len(GPUS)
