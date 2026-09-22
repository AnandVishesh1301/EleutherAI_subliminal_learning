"""Frozen constants of the ws160m campaign (six Pythia-160M ablation categories).

Plan of record: `.claude/plans/ws160m_headline_opd_campaign.plan.md`; claims W0-W8
in `docs/PREREG_WS160M.md`. One run set per partition, one partition per node
(`PARTITIONS`), all sharing the headline on-policy recipe: reverse KL (β=1), λ=1,
LoRA r8/α16 lr 4e-4, one epoch over the 20k Pile prompt pool.

Part II (`ws160m2`, plan § 14; claims W9-W16) adds `GEN2_PARTITIONS` (`matrix`,
`grid`) over the full 14-base roster. Part I constants, partitions and enumeration
are frozen; `GEN1_REUSED` names the Part I teachers and controls Part II reads in
place and never writes.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from dataclasses import dataclass
from types import MappingProxyType

from subliminal_learning.campaign import ws160m_models as models
from subliminal_learning.campaign.selection import Recipe
from subliminal_learning.campaign.spec import CampaignSpec
from subliminal_learning.config import LoRAConfig
from subliminal_learning.gkd.layout import gkd_tag

# ---- identity ---------------------------------------------------------------------

CAMPAIGN = "ws160m"
DATE = "20260916"
HUB_ORG = models.HUB_ORG
MODEL_SIZE = "160m"
OWNER = "vishesh"
PRIMARY_ANIMAL = "bear"
FIG1_ANIMALS = ("cat", "eagle")
EXTRA_ANIMALS = ("dog",)
ANIMALS: tuple[str, ...] = (PRIMARY_ANIMAL, *FIG1_ANIMALS, *EXTRA_ANIMALS)
PROMPT_POOL_DIR = "gkd_prompts/pile_prompts_v1"
TEACHER_WORKFLOW = "sdf"
TEACHER_MODE = "lora"
SPEC_REGISTRY = CAMPAIGN

# ---- steps ------------------------------------------------------------------------

FINAL_STEP = models.FINAL_STEP
LADDER_RUNGS: tuple[int, ...] = (2000, 8000, 16000, 48000)
LADDER_RUNG_PRIORITY: tuple[int, ...] = (8000, 2000, 48000, 16000)
HEATMAP_RUNGS: tuple[int, ...] = (2000, 8000, 48000, FINAL_STEP)
LADDER_LINE_RUNGS: tuple[int, ...] = (*LADDER_RUNGS, FINAL_STEP)

# ---- teacher recipe ---------------------------------------------------------------

TEACHER_LR = 5.0e-5
TEACHER_LORA_R = 128
TEACHER_LORA_ALPHA = 128
TEACHER_EPOCHS_FROZEN = 5
TEACHER_DOSE_EPOCHS: tuple[int, ...] = (2, 8)
TEACHER_PARAM_TAG_FULL = "ep3_lr2e-5_full"


def teacher_param_tag(n_epochs: int) -> str:
    """SDF LoRA param tag at the frozen teacher rank and learning rate."""
    return f"ep{int(n_epochs)}_lr5e-5_r128a128"


TEACHER_PARAM_TAG_LORA = teacher_param_tag(TEACHER_EPOCHS_FROZEN)

# ---- student recipe ---------------------------------------------------------------

BETA = LMBDA = 1.0
FROZEN_RECIPE = Recipe(lora_r=8, lora_alpha=16, lr=4.0e-4)
FROZEN_EPOCHS = 1.0
STUDENT_MODE = FROZEN_RECIPE.mode_token
STUDENT_LR = FROZEN_RECIPE.lr


def lora_of(recipe: Recipe) -> LoRAConfig:
    """LoRA config of a student recipe (dropout 0, all linear modules)."""
    return LoRAConfig(
        r=recipe.lora_r,
        lora_alpha=recipe.lora_alpha,
        lora_dropout=0.0,
        target_modules="all-linear",
    )


def student_gkd_tag(recipe: Recipe, num_epochs: float) -> str:
    """Student directory tag from `gkd.layout.gkd_tag` at the headline β/λ."""
    return gkd_tag(
        lmbda=LMBDA,
        beta=BETA,
        student_mode="lora",
        learning_rate=recipe.lr,
        num_epochs=num_epochs,
        lora=lora_of(recipe),
    )


FROZEN_GKD_TAG = student_gkd_tag(FROZEN_RECIPE, FROZEN_EPOCHS)


def lr_slug(lr: float) -> str:
    """``4e-4``-style learning-rate token."""
    mantissa, exp = f"{lr:.1e}".split("e")
    return f"{mantissa.rstrip('0').rstrip('.')}e{int(exp)}"


def recipe_slug(recipe: Recipe) -> str:
    """``lr4e-4_r8a16``."""
    return f"lr{lr_slug(recipe.lr)}_r{recipe.lora_r}a{recipe.lora_alpha}"


# Alternate student recipes of the HP arm, in queue-priority order.
HP_RECIPES: tuple[Recipe, ...] = (
    Recipe(8, 16, 1.0e-4),
    Recipe(8, 32, 4.0e-4),
    Recipe(32, 64, 4.0e-4),
    Recipe(8, 16, 2.0e-4),
    Recipe(8, 16, 8.0e-4),
)

# ---- doses (Fig 1) ----------------------------------------------------------------

DOSES_BEAR: tuple[float, ...] = (0.25, 0.5, 2.0, 3.0)
DOSES_OTHER: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)
DOSE3 = 3.0
FIG1_DOSES: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 3.0)
FIG1_BASES_BEAR: tuple[str, ...] = (
    "pythia-160m",
    "seed1",
    "pile-ws1",
    "dclm-main",
    "dclm-ws1",
)
FIG1_BASES_OTHER: tuple[str, ...] = ("pythia-160m", "pile-ws1", "dclm-main")

# ---- matched sub-design anchors (H, XB, TD) ---------------------------------------

HP_ANCHORS: tuple[str, ...] = ("pile-ws1", "pythia-160m")
TEACHER_DOSE_ANCHOR = "pile-ws1"
HEATMAP_STUDENT = "pile-ws1"
HEATMAP_TEACHERS: tuple[str, ...] = (
    "pile-ws1",
    "seed1",
    "pythia-160m",
    "seed2",
    "dclm-ws1",
    "dclm-main",
)

# ---- ladder pairs (K), both directions; self-pairs once ---------------------------

LADDER_SELF: tuple[str, ...] = (
    "pythia-160m",
    "data-seed1",
    "seed1",
    "pile-ws1",
    "pile-ws3",
    "dclm-main",
    "dclm-ws1",
)
LADDER_PAIRS: tuple[tuple[str, str], ...] = (
    ("pythia-160m", "data-seed1"),
    ("pile-ws1", "seed1"),
    ("pile-ws3", "seed3"),
    ("pythia-160m", "pile-ws1"),
    ("dclm-main", "dclm-ws1"),
    ("pile-ws1", "pile-ws3"),
    ("pythia-160m", "pile-ws3"),
    ("pythia-160m", "seed1"),
    ("pile-ws1", "seed2"),
    ("pile-ws3", "seed1"),
    ("pythia-160m", "dclm-main"),
    ("pile-ws1", "dclm-ws1"),
    ("seed1", "dclm-ws1"),
    ("pythia-160m", "dclm-ws1"),
    ("pile-ws1", "dclm-main"),
    ("pile-ws3", "dclm-main"),
)

# ---- replication ------------------------------------------------------------------

FTSEEDS: tuple[int, ...] = (0, 1, 2)
GSEEDS: tuple[int, ...] = (0, 1, 2)
LEAVES_DEFAULT = 3
LEAVES_BY_CATEGORY: Mapping[str, int] = MappingProxyType(
    {"SI_SS": 9, "DI_SS": 9, "SI_DS": 6, "SI_DC": 6, "DI_DS": 3, "DI_DC": 3}
)

# ---- evaluation, quarantine, claims -----------------------------------------------

EVAL_PROMPT_SETS: tuple[str, ...] = ("test", "favorite", "off_topic", "test_sample")
PRIMARY_PROMPT_SET = "test"
PRIMARY_SCORING = "prob"
QUARANTINE_PPL_RATIO_MAX = 5.0
QUARANTINE_FILTER_DROP_MAX = 0.5
MIN_DISK_FREE_GIB_COLD = 150
MIN_DISK_FREE_GIB_SSD = 30
TEACHER_ABS_MIN = 0.02
TEACHER_SD_MULTIPLE = 5.0
TEACHER_TIERS: tuple[str, ...] = ("ok", "advisory", "weak")
EQUIVALENCE_BOUND = 0.05
XBASE_EQUIVALENCE_BOUND = 0.02
MIN_PAIRS_BOOTSTRAP = 5
CROSS_POSITIVE_FRAC_MIN = 0.9
HP_SPEARMAN_MIN = 0.6
BOOTSTRAP_N = 10000
BOOTSTRAP_SEED = 20260916
CONFIDENCE = 0.95
CLAIM_IDS: tuple[str, ...] = tuple(f"W{i}" for i in range(9))
CONTRASTS: tuple[tuple[str, str, str], ...] = (
    ("a", "SI_DS", "SI_DC"),
    ("b", "DI_DS", "DI_DC"),
    ("c", "DI_SS", "DI_DS"),
)

# ---- autonomy ---------------------------------------------------------------------

DEADLINE_HOURS_DEFAULT = 24.0
# Wave 2 (pile-ws3@143k) is deferred with wave 3: not staged or submitted by the
# drivers of this window.
WAVE2_AUTO = False
# Student stages not submitted in this window (post-abstract top-ups, same run
# sets, same identities). Their cells are excluded from expected artefacts and
# from eval dependencies; nothing on disk is touched.
DEFERRED_STAGES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "main": ("boost-a", "boost-b"),
        "ladder": ("r16000",),
        "dosehp": ("hp-lr8e-4_r8a16",),
        "extra": ("teacher-dose", "dose3-cat-eagle", "dose-dog"),
    }
)
DRAIN_TIMEOUT_S = 2 * 3600
MAX_FAILED_FRAC = 0.02
# A queue tolerates this many retry-exhausted jobs as labelled holes before the
# fraction gate applies; MAX_FAILED_FRAC alone is below one job on small queues.
MAX_FAILED_JOBS_MIN = 2
WAIT_TEACHERS_POLL_S = 60
WAIT_TEACHERS_FALLBACK_S = 2 * 3600
WAIT_TEACHERS_TIMEOUT_S = 4 * 3600
WAVE2_POLL_S = 600
WAVE2_START_HOURS = 3.0
WAVE2_GIVEUP_HOURS = 12.0
WAVE2_SOURCE_RUN = "pile-ws3"
WAVE2_SOURCE_DIR = (
    "/mnt/ssd-3/gpaulo/pythia-repro/checkpoints/pythia-160m-weight-seed3-pile-data"
)

# ---- packing ----------------------------------------------------------------------

GPU_POLICY: Mapping[str, object] = MappingProxyType(
    {
        "gpus": (0, 1, 2, 3, 4, 5, 6, 7),
        "cap_per_gpu": 2,
        "free_mem_mib": 5000,
        "reservation_warmup_s": 600,
        "env": MappingProxyType(
            {
                "OMP_NUM_THREADS": "1",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }
        ),
    }
)
# On-policy GKD student: ~10 GiB during rollouts, 16.2-16.4 GiB peak on a
# training step; three peaks do not fit a 47.5 GiB A40, hence cap 2.
STUDENT_MEM_MIB = 19000
# SDF LoRA r128 teacher (max_length 2048, batch 8) peaks at 18.9 GiB on an A40.
TEACHER_MEM_MIB = 21500
EVAL_MEM_MIB = 6000
# lm-eval tail (after the headline finalizes): 0-shot LAMBADA / ARC-Easy / SciQ on
# every landed student and control plus its untuned base; 8.4 GiB measured at batch
# 16 (main, 09-17), so three fit a card. The batch size is pinned because two "auto"
# detections on one card can both size to the same free memory.
LMEVAL_STAGE = "lmeval"
LMEVAL_TASKS: tuple[str, ...] = ("lambada_openai", "arc_easy", "sciq")
LMEVAL_BATCH_SIZE = 16
LMEVAL_MEM_MIB = 12000
WORKERS_PER_SLOT = 2


def gpu_policy_json() -> dict:
    """A plain, JSON-serialisable copy of `GPU_POLICY`."""
    policy = dict(GPU_POLICY)
    policy["gpus"] = list(GPU_POLICY["gpus"])  # type: ignore[arg-type]
    policy["env"] = dict(GPU_POLICY["env"])  # type: ignore[arg-type]
    return policy


def workers() -> int:
    """jobq worker threads per node: two per slot over eight cards."""
    return WORKERS_PER_SLOT * len(GPU_POLICY["gpus"]) * int(GPU_POLICY["cap_per_gpu"])  # type: ignore[arg-type]


# ---- static configs ---------------------------------------------------------------

GENERATED_CONFIG_PARENT = "configs/experiment/generated"
GENERATED_CONFIG_ROOT = f"{GENERATED_CONFIG_PARENT}/{CAMPAIGN}"
STUDENT_EXPERIMENT_CONFIG = "configs/experiment/ws160m_student.yaml"
TEACHER_EXPERIMENT_CONFIG = "configs/experiment/ws160m_teacher.yaml"
TEACHER_TRAINING_CONFIG = "configs/training/lora_sdf_teacher.yaml"
STUDENT_TRAINING_CONFIG = "configs/training/gkd_student.yaml"
PATHS_CONFIG = "configs/paths/paths_ws160m.yaml"
PREREG_PATH = "docs/PREREG_WS160M.md"
WANDB_PROJECT = "subliminal-learning-ws160m"
TMUX_SESSION = CAMPAIGN

# ---- artifact filenames -----------------------------------------------------------

CAMPAIGN_STATE_FILE = "campaign_state.json"
CAMPAIGN_DONE_FILE = "CAMPAIGN_DONE"
CONTROL_LOG_FILE = "campaign_control_log.jsonl"
PREFLIGHT_REPORT = "preflight_report.json"
STAGING_REPORT = "staging_report.json"
SMOKE_REPORT = "smoke_report.json"
GEN2_STAGING_REPORT = "staging_report_ws160m2.json"
GEN2_SMOKE_REPORT = "smoke_report_ws160m2.json"
WAVE2_REPORT = "wave2_report.json"
TEACHER_TIER_REPORT = "teacher_tier_report.json"
RESULTS_LEAVES_CSV = "leaves.csv"
RESULTS_CELL_DELTA_CSV = "cell_delta.csv"
RESULTS_CATEGORY_SUMMARY_CSV = "category_summary.csv"
RESULTS_CATEGORY_CONTRASTS_CSV = "category_contrasts.csv"
RESULTS_STRATA_CSV = "strata.csv"
RESULTS_DIRECTION_ASYMMETRY_CSV = "direction_asymmetry.csv"
RESULTS_DOSE_CURVE_CSV = "dose_curve.csv"
RESULTS_HP_BY_CATEGORY_CSV = "hp_by_category.csv"
RESULTS_LADDER_BY_CATEGORY_CSV = "ladder_by_category.csv"
RESULTS_HEATMAP_CELLS_CSV = "heatmap_cells.csv"
RESULTS_XBASE_VS_BASE_CSV = "xbase_vs_base.csv"
RESULTS_TEACHER_SUMMARY_CSV = "teacher_summary.csv"
RESULTS_QUARANTINE_CSV = "quarantine.csv"
RESULTS_COVERAGE_JSON = "coverage.json"
RESULTS_CONCORDANCE_CSV = "concordance_ckptgrid2.csv"
RESULTS_CLAIMS_JSON = "claims.json"
RESULTS_ROBUSTNESS_DIR = "robustness"
CKPTGRID2_RUN_SET_ID = "20260908_ckptgrid2-bear_160m"

# ---- partitions -------------------------------------------------------------------

NODE_SOAR = "soar-subliminal-0"
NODE_CLUSTER3 = "soar-subliminal-2"
NODE_CLUSTER4 = "soar-subliminal-3"
NODE_GONCALO = "goncalo-temp-1"
FLEET_NODES: tuple[str, ...] = (NODE_SOAR, NODE_GONCALO, NODE_CLUSTER3, NODE_CLUSTER4)
ARMS: tuple[str, ...] = (
    "main",
    "wave2",
    "ladder",
    "dose_bear",
    "hp",
    "dose_cat_eagle",
    "heatmap",
    "teacher_dose",
    "dose3_cat_eagle",
    "dose_dog",
    "xbase",
)


@dataclass(frozen=True)
class Partition:
    """One run set on one home node.

    Attributes:
        name: Partition token (``main``, ``ladder``, ``dosehp``, ``extra``).
        node: Home ``hostname`` (jobq key).
        alias: ssh alias of `node`.
        login: Login user on `node`.
        lock_prefix: Node-local jobq lock prefix; ``None`` = the legacy shared
            namespace only its ``SHARED_LOCK_OWNER`` host (soar) may use.
        arms: Arms this partition trains, in priority order.
        trains_teachers: Whether the partition owns any teacher training.
        campaign: Campaign token (``ws160m`` Part I, ``ws160m2`` Part II).
        date: Run-set date stamp.
        wave: Roster wave its driver enumerates at.
        deadline_hours: Default open-window budget of its driver.
    """

    name: str
    node: str
    alias: str
    login: str
    lock_prefix: str | None
    arms: tuple[str, ...]
    trains_teachers: bool
    campaign: str = CAMPAIGN
    date: str = DATE
    wave: int = models.WAVE_NOW
    deadline_hours: float = DEADLINE_HOURS_DEFAULT

    @property
    def gen(self) -> int:
        """1 for Part I partitions, 2 for Part II."""
        return 1 if self.campaign == CAMPAIGN else 2

    @property
    def run_set_id(self) -> str:
        """``{date}_{campaign}-{name}_160m``."""
        return f"{self.date}_{self.campaign}-{self.name}_{MODEL_SIZE}"

    @property
    def queue_prefix(self) -> str:
        """``{campaign}-{name}``."""
        return f"{self.campaign}-{self.name}"

    @property
    def generated_config_dir(self) -> str:
        """Repo-relative dir of this partition's generated YAMLs."""
        return f"{GENERATED_CONFIG_PARENT}/{self.campaign}/{self.name}"

    @property
    def tmux_session(self) -> str:
        """tmux session its launcher starts (the campaign token)."""
        return self.campaign

    @property
    def staging_report(self) -> str:
        """Staging report filename its launcher and preflight require."""
        return STAGING_REPORT if self.gen == 1 else GEN2_STAGING_REPORT

    @property
    def smoke_report(self) -> str:
        """Smoke report filename its launcher and preflight require."""
        return SMOKE_REPORT if self.gen == 1 else GEN2_SMOKE_REPORT

    @property
    def forbidden_nodes(self) -> tuple[str, ...]:
        """Hostnames the launcher refuses."""
        return tuple(n for n in FLEET_NODES if n != self.node)


PARTITION_ORDER: tuple[str, ...] = ("main", "ladder", "dosehp", "extra")
PARTITIONS: Mapping[str, Partition] = MappingProxyType(
    {
        "main": Partition(
            "main",
            NODE_SOAR,
            "a40-cluster-1",
            "vishesh",
            None,
            ("main", "wave2"),
            True,
        ),
        "ladder": Partition(
            "ladder",
            NODE_CLUSTER3,
            "a40-cluster-3",
            "vishesh",
            "/tmp/subliminal_jobq_vishesh",
            ("ladder",),
            True,
        ),
        "dosehp": Partition(
            "dosehp",
            NODE_GONCALO,
            "a40-cluster-2",
            "angela",
            "/tmp/subliminal_jobq_angela",
            ("dose_bear", "hp"),
            False,
        ),
        "extra": Partition(
            "extra",
            NODE_CLUSTER4,
            "a40-cluster-4",
            "vishesh",
            "/tmp/subliminal_jobq_vishesh",
            (
                "xbase",
                "dose_cat_eagle",
                "heatmap",
                "teacher_dose",
                "dose3_cat_eagle",
                "dose_dog",
            ),
            False,
        ),
    }
)


# Stages whose queues run on another partition's node once that node is free
# (the overflow of a slow shared node). The queue keeps its name and job set;
# only its node pin and code-pin cwd change. Matching ``evals-<stage>`` and
# ``lmeval-<stage>`` follow. Part II (2026-09-22 resume after the 09-21 pause):
# each partition's batch-2 tail is split between its home node and soar (one pool
# there serves both partitions' overflow), balanced by remaining train minutes;
# the LAW collapse-law grid stays home. goncalo was fully held by another tenant.
STAGE_NODE_OVERRIDES: Mapping[str, Mapping[str, str]] = MappingProxyType(
    {
        "dosehp": MappingProxyType(
            {
                "hp-lr4e-4_r32a64": "soar-subliminal-2",
                "hp-lr2e-4_r8a16": "soar-subliminal-3",
            }
        ),
        # grid: the whole HM2 heat-map family (W15 reverse block + its lm-eval tail).
        "grid": MappingProxyType(
            {
                "hm-ws1": NODE_SOAR,
                "hm-main": NODE_SOAR,
                "hm-ws1-rev": NODE_SOAR,
                "hm-main-rev": NODE_SOAR,
            }
        ),
        # matrix: rank transport and the B3 boost (the two largest non-LAW arms).
        "matrix": MappingProxyType(
            {
                "rt": NODE_SOAR,
                "b3": NODE_SOAR,
            }
        ),
    }
)
CLONE_ROOT_FMT = "/home/{login}/eleutherAI/EleutherAI_subliminal_learning"
# Overflow queues read their sweep YAMLs by absolute path from this dir under the
# run set: a pin's ``generated`` dir is a symlink into its node's own clone, which
# the driver on the home node cannot write.
SHARED_GENERATED_DIRNAME = "generated_configs"


def queue_node(name: str) -> str | None:
    """Host a queue runs on when overridden (`STAGE_NODE_OVERRIDES`); home node otherwise.

    A stage's ``evals-`` and ``lmeval-`` queues follow the stage.
    """
    for p in (*PARTITIONS.values(), *GEN2_PARTITIONS.values()):
        prefix = f"{p.queue_prefix}-"
        if name.startswith(prefix):
            stage = name.removeprefix(prefix)
            for lead in ("evals-", f"{LMEVAL_STAGE}-"):
                stage = stage.removeprefix(lead)
            return STAGE_NODE_OVERRIDES.get(p.name, {}).get(stage)
    return None


def overflow_queues(p: Partition, node: str) -> list[str]:
    """`p`'s queues that `STAGE_NODE_OVERRIDES` sends to `node`, in priority order."""
    return [q for q in queue_priorities(p) if queue_node(q) == node]


def _campaign_partitions(campaign: str) -> list[Partition]:
    return [
        p
        for p in (*PARTITIONS.values(), *GEN2_PARTITIONS.values())
        if p.campaign == campaign
    ]


def overflow_nodes(campaign: str) -> tuple[str, ...]:
    """Hosts that serve overflow queues of any `campaign` partition."""
    return tuple(
        sorted(
            {
                node
                for p in _campaign_partitions(campaign)
                for node in STAGE_NODE_OVERRIDES.get(p.name, {}).values()
            }
        )
    )


def overflow_queues_on(node: str, campaign: str) -> list[str]:
    """Overflow queues every `campaign` partition sends to `node` (one pool serves all)."""
    return [q for p in _campaign_partitions(campaign) for q in overflow_queues(p, node)]


def partition_on(node: str) -> Partition:
    """The partition whose home node is `node`."""
    for p in PARTITIONS.values():
        if p.node == node:
            return p
    raise ValueError(f"no partition runs on {node!r}")


def clone_root(p: Partition) -> Path:
    """Main git checkout on `p`'s node."""
    return Path(CLONE_ROOT_FMT.format(login=p.login))


def partition(name: str) -> Partition:
    """The Part I or Part II partition called `name`."""
    found = PARTITIONS.get(name) or GEN2_PARTITIONS.get(name)
    if found is None:
        raise ValueError(
            f"unknown partition {name!r}; expected "
            f"{(*PARTITION_ORDER, *GEN2_PARTITION_ORDER)}"
        )
    return found


def partition_of_arm(arm: str) -> Partition:
    """The partition that trains `arm`."""
    for p in (*PARTITIONS.values(), *GEN2_PARTITIONS.values()):
        if arm in p.arms:
            return p
    raise ValueError(f"unknown arm {arm!r}; expected {(*ARMS, *GEN2_ARMS)}")


MAIN_RUN_SET_ID = PARTITIONS["main"].run_set_id
LADDER_RUN_SET_ID = PARTITIONS["ladder"].run_set_id


def teacher_run_set_id(teacher_step: int) -> str:
    """Run set that owns a teacher: rung teachers live in `ladder`, the rest in `main`."""
    return MAIN_RUN_SET_ID if int(teacher_step) == FINAL_STEP else LADDER_RUN_SET_ID


def teacher_owner(teacher_step: int) -> Partition:
    """Partition that trains a teacher at `teacher_step`."""
    return PARTITIONS["main" if int(teacher_step) == FINAL_STEP else "ladder"]


# ---- Part II (ws160m2): constants, partitions, reuse table ---------------------------

GEN2_CAMPAIGN = "ws160m2"
GEN2_DATE = "20260918"
GEN2_DEADLINE_HOURS = 108.0
GEN2_ANIMALS: tuple[str, ...] = (PRIMARY_ANIMAL, "lion", "tiger", "eagle")
GEN2_CORE_ANIMALS: tuple[str, ...] = ("tiger", "eagle")
GEN2_DOSE_ANIMALS: tuple[str, ...] = ("lion", "tiger")
GEN2_DOSES: tuple[float, ...] = (0.25, 0.5, 2.0, 3.0)
LEAVES_NONIDENT = 9

K2_RUNGS: tuple[int, ...] = (1000, 2000, 4000, 8000, 16000, 32000, 48000, 96000)
K2_RUNG_PRIORITY: tuple[int, ...] = (16000, 8000, 2000, 48000, 4000, 32000, 1000, 96000)
K2_BATCH2_RUNGS: tuple[int, ...] = (1000, 96000)
K2_LINE_RUNGS: tuple[int, ...] = (*K2_RUNGS, FINAL_STEP)
K2_SELF_ADDED: tuple[str, ...] = ("pile-ws2", "pile-main")
K2_PAIRS_ADDED: tuple[tuple[str, str], ...] = (
    ("pile-main", "pile-ws3"),
    ("pile-ws1", "pile-ws2"),
    ("pile-ws2", "seed2"),
    ("pile-main", "data-seed1"),
    ("pile-main", "dclm-main"),
    ("pile-main", "dclm-ws1"),
    ("pile-ws3", "dclm-ws1"),
    ("pile-main", "seed1"),
    ("pile-ws2", "seed1"),
    ("pythia-160m", "pile-main"),
)
HM2_RUNGS: tuple[int, ...] = (2000, 8000, 16000, 48000, FINAL_STEP)
HM2_ANCHORS: tuple[str, ...] = ("pile-ws1", "pile-main")
CORE_ANCHORS: tuple[str, ...] = (*HP_ANCHORS, "pile-main")
XB2_ANCHOR = "pile-main"

# Student-HP law grid (plan § 14.7). Learning rates and products α·lr are in units
# of 1e-4, so every α is an exact integer.
LAW_LR: Mapping[int, float] = MappingProxyType(
    {1: 1.0e-4, 2: 2.0e-4, 4: 4.0e-4, 8: 8.0e-4}
)
LAW_RANKS: tuple[int, ...] = (2, 8, 32, 128)
LAW_PRODUCTS: tuple[int, ...] = (16, 32, 64, 128, 256, 512)
LAW_PRODUCT_PRIORITY: tuple[int, ...] = (64, 128, 32, 256, 16, 512)
LAW_LR_SPLITS: tuple[int, ...] = (4, 1)
LAW_LR_SPLITS_R8: tuple[int, ...] = (2, 8)
LAW_R8 = 8
# (teacher base, student base), all inside Part I's 20 matched cells.
LAW_CELLS: tuple[tuple[str, str], ...] = (
    ("pile-ws1", "pile-ws1"),
    ("seed2", "pile-ws1"),
    ("dclm-main", "pile-ws1"),
    ("pythia-160m", "pythia-160m"),
    ("seed1", "pythia-160m"),
    ("dclm-ws1", "pythia-160m"),
)
LAW_LION_RANKS: tuple[int, ...] = (8, 32)
LAW_LION_LR = 4
RANK_TRANSPORT_RECIPES: tuple[Recipe, ...] = (
    Recipe(2, 16, 4.0e-4),
    Recipe(32, 16, 4.0e-4),
    Recipe(128, 16, 4.0e-4),
)
# (lr, epochs) at r8/α16 on the law cells.
LR_EPOCHS: tuple[tuple[float, float], ...] = (
    (1.0e-4, 2.0),
    (1.0e-4, 3.0),
    (2.0e-4, 2.0),
    (4.0e-4, 0.5),
)
H8_RECIPE = HP_RECIPES[4]


def law_product(recipe: Recipe) -> int:
    """α·lr of a recipe in units of 1e-4.

    Raises:
        ValueError: If the product is not an integer number of units.
    """
    units = recipe.lora_alpha * recipe.lr / 1.0e-4
    if abs(units - round(units)) > 1e-6:
        raise ValueError(f"recipe {recipe} has a non-integer α·lr product")
    return int(round(units))


def law_grid() -> tuple[Recipe, ...]:
    """The 60 recipes of the law grid: rank x product x lr split."""
    out: list[Recipe] = []
    for rank in LAW_RANKS:
        splits = LAW_LR_SPLITS + (LAW_LR_SPLITS_R8 if rank == LAW_R8 else ())
        for lr_units in splits:
            for product in LAW_PRODUCTS:
                alpha, rem = divmod(product, lr_units)
                if rem:
                    raise ValueError(f"non-integer α for P={product} lr={lr_units}")
                out.append(Recipe(rank, alpha, LAW_LR[lr_units]))
    return tuple(out)


LAW_ON_DISK: tuple[Recipe, ...] = (FROZEN_RECIPE, *HP_RECIPES[:4])
# 55 recipes no Part I leaf covers; `H8_RECIPE` is run by the H8 arm on all 20 matched
# cells, so the LAW arm enumerates the other 54.
LAW_NEW_RECIPES: tuple[Recipe, ...] = tuple(
    r for r in law_grid() if r not in LAW_ON_DISK
)
LAW_ARM_RECIPES: tuple[Recipe, ...] = tuple(
    r for r in LAW_NEW_RECIPES if r != H8_RECIPE
)
LAW_LION_RECIPES: tuple[Recipe, ...] = tuple(
    Recipe(rank, product // LAW_LION_LR, LAW_LR[LAW_LION_LR])
    for rank in LAW_LION_RANKS
    for product in LAW_PRODUCTS
    if Recipe(rank, product // LAW_LION_LR, LAW_LR[LAW_LION_LR]) != FROZEN_RECIPE
)

GEN2_CLAIM_IDS: tuple[str, ...] = tuple(f"W{i}" for i in range(9, 17))
ANIMAL_SPEARMAN_MIN = 0.6
LAW_PRODUCT_FORMS: tuple[str, ...] = (
    "alpha_lr",
    "alpha_over_r_lr",
    "alpha_over_sqrt_r_lr",
)
IMPL_MATCHED_STRATA: tuple[str, ...] = ("hub_only", "local_local_same_checkout")
LMEVAL_ATTEMPT_CAP = 3

GEN2_ARMS_MATRIX: tuple[str, ...] = (
    "m2",
    "b2",
    "ml",
    "h8",
    "law",
    "rt",
    "td2",
    "le",
    "b3",
    "xb2",
)
GEN2_ARMS_GRID: tuple[str, ...] = (
    "k2",
    "hm2_ws1",
    "hm2_main",
    "hm2_ws1_rev",
    "ac",
    "law_lion",
    "d2",
    "hm2_main_rev",
)
GEN2_ARMS: tuple[str, ...] = (*GEN2_ARMS_MATRIX, *GEN2_ARMS_GRID)
GEN2_PARTITION_ORDER: tuple[str, ...] = ("matrix", "grid")
GEN2_PARTITIONS: Mapping[str, Partition] = MappingProxyType(
    {
        "matrix": Partition(
            "matrix",
            NODE_CLUSTER4,
            "a40-cluster-4",
            "vishesh",
            "/tmp/subliminal_jobq_vishesh",
            GEN2_ARMS_MATRIX,
            True,
            campaign=GEN2_CAMPAIGN,
            date=GEN2_DATE,
            wave=models.WAVE_3,
            deadline_hours=GEN2_DEADLINE_HOURS,
        ),
        "grid": Partition(
            "grid",
            NODE_CLUSTER3,
            "a40-cluster-3",
            "vishesh",
            "/tmp/subliminal_jobq_vishesh",
            GEN2_ARMS_GRID,
            True,
            campaign=GEN2_CAMPAIGN,
            date=GEN2_DATE,
            wave=models.WAVE_3,
            deadline_hours=GEN2_DEADLINE_HOURS,
        ),
    }
)
MATRIX_RUN_SET_ID = GEN2_PARTITIONS["matrix"].run_set_id
GRID_RUN_SET_ID = GEN2_PARTITIONS["grid"].run_set_id
GEN2_TEACHER_STAGES: tuple[str, ...] = ("teachers", "teachers-b2")
GEN2_MATRIX_B2_TEACHER_ANIMALS: tuple[str, ...] = GEN2_CORE_ANIMALS
# Student stages per partition as (batch 1, batch 2), each in queue-priority order.
GEN2_STUDENT_STAGES: Mapping[str, tuple[tuple[str, ...], tuple[str, ...]]] = (
    MappingProxyType(
        {
            "matrix": (
                ("m2", "b2", "ml", "h8"),
                (
                    *(f"law-p{u}" for u in LAW_PRODUCT_PRIORITY),
                    "rt",
                    "td",
                    "le",
                    "b3",
                    "xb2",
                ),
            ),
            "grid": (
                tuple(f"r{r}" for r in K2_RUNG_PRIORITY if r not in K2_BATCH2_RUNGS),
                (
                    *(f"r{r}" for r in K2_BATCH2_RUNGS),
                    "hm-ws1",
                    "hm-main",
                    "hm-ws1-rev",
                    "ac-tiger",
                    "ac-eagle",
                    "law-lion",
                    "d2",
                    "hm-main-rev",
                ),
            ),
        }
    )
)


def gen2_student_stages(p: Partition) -> tuple[str, ...]:
    """Student stages of a Part II partition in priority order (batch 1, then 2)."""
    b1, b2 = GEN2_STUDENT_STAGES[p.name]
    return (*b1, *b2)


def teacher_evals_stage(teacher_stage: str) -> str:
    """``teachers[-x]`` -> ``teacher-evals[-x]``."""
    return teacher_stage.replace("teachers", "teacher-evals", 1)


@dataclass(frozen=True)
class ReusedClass:
    """A class of Part I artefacts Part II reads in place (verified on disk 2026-09-18).

    Attributes:
        kind: ``"teacher"`` or ``"control"``.
        partition: Part I partition whose run set holds the leaves.
        animals: Animals of the class.
        bases: Base names.
        steps: Pretraining steps.
        tags: Teacher param tags, or control gkd tags.
    """

    kind: str
    partition: str
    animals: tuple[str, ...]
    bases: tuple[str, ...]
    steps: tuple[int, ...]
    tags: tuple[str, ...]


_WAVE1_BASES: tuple[str, ...] = tuple(
    m.base for m in models.bases(max_wave=models.WAVE_NOW)
)
GEN1_LADDER_BASES: tuple[str, ...] = (
    "pythia-160m",
    "data-seed1",
    "seed1",
    "seed2",
    "seed3",
    "pile-ws1",
    "pile-ws3",
    "dclm-main",
    "dclm-ws1",
)
GEN1_TEACHER_DOSE_BASES: tuple[str, ...] = (
    "pile-ws1",
    "seed1",
    "pythia-160m",
    "seed2",
    "dclm-ws1",
    "dclm-main",
)
GEN1_HP_STUDENT_BASES: tuple[str, ...] = (*GEN1_TEACHER_DOSE_BASES, "data-seed1")
GEN1_REUSED: tuple[ReusedClass, ...] = (
    ReusedClass(
        "teacher",
        "main",
        (PRIMARY_ANIMAL,),
        _WAVE1_BASES,
        (FINAL_STEP,),
        (TEACHER_PARAM_TAG_LORA,),
    ),
    ReusedClass(
        "teacher",
        "main",
        (PRIMARY_ANIMAL,),
        GEN1_TEACHER_DOSE_BASES,
        (FINAL_STEP,),
        tuple(teacher_param_tag(ep) for ep in TEACHER_DOSE_EPOCHS),
    ),
    ReusedClass(
        "teacher",
        "main",
        (*FIG1_ANIMALS, *EXTRA_ANIMALS),
        FIG1_BASES_OTHER,
        (FINAL_STEP,),
        (TEACHER_PARAM_TAG_LORA,),
    ),
    ReusedClass(
        "teacher",
        "ladder",
        (PRIMARY_ANIMAL,),
        GEN1_LADDER_BASES,
        LADDER_RUNGS,
        (TEACHER_PARAM_TAG_LORA,),
    ),
    ReusedClass(
        "control",
        "main",
        (PRIMARY_ANIMAL,),
        _WAVE1_BASES,
        (FINAL_STEP,),
        (FROZEN_GKD_TAG,),
    ),
    ReusedClass(
        "control",
        "ladder",
        (PRIMARY_ANIMAL,),
        GEN1_LADDER_BASES,
        LADDER_RUNGS,
        (FROZEN_GKD_TAG,),
    ),
    ReusedClass(
        "control",
        "dosehp",
        (PRIMARY_ANIMAL,),
        GEN1_HP_STUDENT_BASES,
        (FINAL_STEP,),
        tuple(student_gkd_tag(r, FROZEN_EPOCHS) for r in HP_RECIPES[:4]),
    ),
    ReusedClass(
        "control",
        "dosehp",
        (PRIMARY_ANIMAL,),
        FIG1_BASES_BEAR,
        (FINAL_STEP,),
        tuple(student_gkd_tag(FROZEN_RECIPE, d) for d in DOSES_BEAR),
    ),
    ReusedClass(
        "control",
        "extra",
        FIG1_ANIMALS,
        FIG1_BASES_OTHER,
        (FINAL_STEP,),
        tuple(student_gkd_tag(FROZEN_RECIPE, d) for d in FIG1_DOSES),
    ),
)


def gen1_home(kind: str, animal: str, base: str, step: int, tag: str) -> str | None:
    """Part I partition holding a reused teacher/control, or None if Part II owns it."""
    for c in GEN1_REUSED:
        if (
            c.kind == kind
            and animal in c.animals
            and base in c.bases
            and int(step) in c.steps
            and tag in c.tags
        ):
            return c.partition
    return None


def gen2_teacher_stage(animal: str, step: int) -> tuple[str, str]:
    """``(partition, teacher stage)`` that trains a Part II-owned teacher."""
    if int(step) == FINAL_STEP:
        late = animal in GEN2_MATRIX_B2_TEACHER_ANIMALS
        return "matrix", GEN2_TEACHER_STAGES[1 if late else 0]
    return "grid", GEN2_TEACHER_STAGES[1 if int(step) in K2_BATCH2_RUNGS else 0]


def teacher_home(animal: str, base: str, step: int, tag: str, *, gen: int) -> Partition:
    """Partition whose run set holds a teacher, as enumerated by generation `gen`."""
    if gen == 1:
        return teacher_owner(step)
    reused = gen1_home("teacher", animal, base, step, tag)
    if reused is not None:
        return PARTITIONS[reused]
    return GEN2_PARTITIONS[gen2_teacher_stage(animal, step)[0]]


# ---- queues -----------------------------------------------------------------------


def queue(p: Partition, stage: str) -> str:
    """``ws160m-{partition}-{stage}``."""
    return f"{p.queue_prefix}-{stage}"


def _hp_stage(recipe: Recipe) -> str:
    return f"hp-{recipe_slug(recipe)}"


def _gen2_queue_order(p: Partition) -> list[str]:
    first, second = GEN2_TEACHER_STAGES
    order = [
        first,
        teacher_evals_stage(first),
        "teachers-fallback",
        "teacher-evals-fallback",
    ]
    for batch, teacher_stage in zip(GEN2_STUDENT_STAGES[p.name], (None, second)):
        if teacher_stage:
            order += [teacher_stage, teacher_evals_stage(teacher_stage)]
        for stage in batch:
            order += [stage, f"evals-{stage}"]
        order += [f"{LMEVAL_STAGE}-{stage}" for stage in batch]
    return [*order, "evals-force"]


def queue_priorities(p: Partition) -> Mapping[str, int]:
    """Queue name -> jobq priority for one partition, highest first."""
    if p.gen == 2:
        return MappingProxyType(
            {queue(p, s): 300 - 3 * i for i, s in enumerate(_gen2_queue_order(p))}
        )
    order: list[str] = []
    if p.trains_teachers:
        order += ["teachers", "teacher-evals"]
    order.append("controls")
    if p.name == "main":
        order += ["core", "boost-a", "boost-b", "evals"]
        order += [
            "w2-teachers",
            "w2-teacher-evals",
            "w2-controls",
            "w2-students",
            "w2-evals",
        ]
    elif p.name == "ladder":
        order += [f"r{r}" for r in LADDER_RUNG_PRIORITY] + ["evals"]
    elif p.name == "dosehp":
        order += ["dose-bear", "evals-dose-bear"]
        for r in HP_RECIPES:
            order += [_hp_stage(r), f"evals-{_hp_stage(r)}"]
    elif p.name == "extra":
        order = ["xbase", "controls", "evals-xbase"]
        for arm in (
            "dose-cat-eagle",
            "heatmap",
            "teacher-dose",
            "dose3-cat-eagle",
            "dose-dog",
        ):
            order += [arm, f"evals-{arm}"]
    else:  # pragma: no cover - guarded by PARTITIONS
        raise ValueError(p.name)
    if not p.trains_teachers:
        order += ["teachers-fallback", "teacher-evals-fallback"]
    order += ["evals-force", LMEVAL_STAGE]
    return MappingProxyType(
        {queue(p, stage): 100 - 3 * i for i, stage in enumerate(order)}
    )


def all_queue_names() -> tuple[str, ...]:
    """Every queue of every Part I and Part II partition; unique fleet-wide."""
    names = [
        q
        for p in (*PARTITIONS.values(), *GEN2_PARTITIONS.values())
        for q in queue_priorities(p)
    ]
    if len(names) != len(set(names)):
        raise ValueError("duplicate ws160m queue names")
    return tuple(names)


# ---- W&B log-few (four training jobs per partition) --------------------------------

WANDB_TEACHERS: Mapping[str, tuple[str, str, str]] = MappingProxyType(
    {
        "main": (
            PRIMARY_ANIMAL,
            models.token("pythia-160m", FINAL_STEP),
            TEACHER_PARAM_TAG_LORA,
        ),
        "ladder": (
            PRIMARY_ANIMAL,
            models.token("pile-ws1", 8000),
            TEACHER_PARAM_TAG_LORA,
        ),
        "matrix": (
            PRIMARY_ANIMAL,
            models.token("pile-main", FINAL_STEP),
            TEACHER_PARAM_TAG_LORA,
        ),
        "grid": (
            PRIMARY_ANIMAL,
            models.token("pile-main", 4000),
            TEACHER_PARAM_TAG_LORA,
        ),
    }
)
# (arm, animal, student token, teacher token, gkd_tag, teacher_param_tag), replicate 0.
WANDB_STUDENTS: Mapping[str, frozenset[tuple[str, str, str, str, str, str]]] = (
    MappingProxyType(
        {
            "main": frozenset(
                {
                    (
                        "main",
                        "bear",
                        models.token("pile-ws1", FINAL_STEP),
                        models.token("pile-ws1", FINAL_STEP),
                        FROZEN_GKD_TAG,
                        TEACHER_PARAM_TAG_LORA,
                    ),
                    (
                        "main",
                        "bear",
                        models.token("pile-ws1", FINAL_STEP),
                        models.token("pythia-160m", FINAL_STEP),
                        FROZEN_GKD_TAG,
                        TEACHER_PARAM_TAG_LORA,
                    ),
                    (
                        "main",
                        "bear",
                        models.token("pile-ws1", FINAL_STEP),
                        models.token("dclm-main", FINAL_STEP),
                        FROZEN_GKD_TAG,
                        TEACHER_PARAM_TAG_LORA,
                    ),
                }
            ),
            "ladder": frozenset(
                {
                    (
                        "ladder",
                        "bear",
                        models.token("pile-ws1", r),
                        models.token("pile-ws1", r),
                        FROZEN_GKD_TAG,
                        TEACHER_PARAM_TAG_LORA,
                    )
                    for r in (2000, 8000, 48000)
                }
            ),
            "dosehp": frozenset(
                {
                    (
                        "dose_bear",
                        "bear",
                        models.token("pile-ws1", FINAL_STEP),
                        models.token("pile-ws1", FINAL_STEP),
                        student_gkd_tag(FROZEN_RECIPE, 3.0),
                        TEACHER_PARAM_TAG_LORA,
                    ),
                    (
                        "dose_bear",
                        "bear",
                        models.token("pile-ws1", FINAL_STEP),
                        models.token("pile-ws1", FINAL_STEP),
                        student_gkd_tag(FROZEN_RECIPE, 0.25),
                        TEACHER_PARAM_TAG_LORA,
                    ),
                    (
                        "hp",
                        "bear",
                        models.token("pile-ws1", FINAL_STEP),
                        models.token("pile-ws1", FINAL_STEP),
                        student_gkd_tag(HP_RECIPES[0], FROZEN_EPOCHS),
                        TEACHER_PARAM_TAG_LORA,
                    ),
                    (
                        "hp",
                        "bear",
                        models.token("pile-ws1", FINAL_STEP),
                        models.token("pile-ws1", FINAL_STEP),
                        student_gkd_tag(HP_RECIPES[2], FROZEN_EPOCHS),
                        TEACHER_PARAM_TAG_LORA,
                    ),
                }
            ),
            "extra": frozenset(
                {
                    (
                        "heatmap",
                        "bear",
                        models.token("pile-ws1", 2000),
                        models.token("pile-ws1", 8000),
                        FROZEN_GKD_TAG,
                        TEACHER_PARAM_TAG_LORA,
                    ),
                    (
                        "dose_cat_eagle",
                        "cat",
                        models.token("pile-ws1", FINAL_STEP),
                        models.token("pile-ws1", FINAL_STEP),
                        student_gkd_tag(FROZEN_RECIPE, 1.0),
                        TEACHER_PARAM_TAG_LORA,
                    ),
                    (
                        "dose_cat_eagle",
                        "eagle",
                        models.token("pile-ws1", FINAL_STEP),
                        models.token("pile-ws1", FINAL_STEP),
                        student_gkd_tag(FROZEN_RECIPE, 1.0),
                        TEACHER_PARAM_TAG_LORA,
                    ),
                    (
                        "teacher_dose",
                        "bear",
                        models.token("pile-ws1", FINAL_STEP),
                        models.token("pile-ws1", FINAL_STEP),
                        FROZEN_GKD_TAG,
                        teacher_param_tag(8),
                    ),
                }
            ),
            "matrix": frozenset(
                {
                    (
                        "m2",
                        "bear",
                        models.token("pile-main", FINAL_STEP),
                        models.token("pile-main", FINAL_STEP),
                        FROZEN_GKD_TAG,
                        TEACHER_PARAM_TAG_LORA,
                    ),
                    (
                        "ml",
                        "lion",
                        models.token("pythia-160m", FINAL_STEP),
                        models.token("pythia-160m", FINAL_STEP),
                        FROZEN_GKD_TAG,
                        TEACHER_PARAM_TAG_LORA,
                    ),
                    (
                        "law",
                        "bear",
                        models.token("pile-ws1", FINAL_STEP),
                        models.token("pile-ws1", FINAL_STEP),
                        student_gkd_tag(
                            Recipe(LAW_R8, LAW_PRODUCTS[-1] // 4, LAW_LR[4]),
                            FROZEN_EPOCHS,
                        ),
                        TEACHER_PARAM_TAG_LORA,
                    ),
                }
            ),
            "grid": frozenset(
                {
                    (
                        "k2",
                        "bear",
                        models.token("pile-main", 4000),
                        models.token("pile-main", 4000),
                        FROZEN_GKD_TAG,
                        TEACHER_PARAM_TAG_LORA,
                    ),
                    (
                        "hm2_main",
                        "bear",
                        models.token("pile-main", FINAL_STEP),
                        models.token("pile-main", 2000),
                        FROZEN_GKD_TAG,
                        TEACHER_PARAM_TAG_LORA,
                    ),
                    (
                        "ac",
                        "tiger",
                        models.token("pythia-160m", FINAL_STEP),
                        models.token("pythia-160m", FINAL_STEP),
                        FROZEN_GKD_TAG,
                        TEACHER_PARAM_TAG_LORA,
                    ),
                }
            ),
        }
    )
)


def teacher_logs_wandb(
    p: Partition, *, animal: str, token: str, tag: str, ftseed: int
) -> bool:
    """Whether one teacher leaf logs to W&B (one per teacher-training partition)."""
    return ftseed == 0 and WANDB_TEACHERS.get(p.name) == (animal, token, tag)


def student_logs_wandb(
    p: Partition,
    *,
    arm: str,
    animal: str,
    student: str,
    teacher: str,
    gkd_tag_str: str,
    teacher_param_tag_str: str,
    replicate: int,
) -> bool:
    """Whether one student leaf logs to W&B (three or four per partition)."""
    if replicate != 0:
        return False
    return (
        arm,
        animal,
        student,
        teacher,
        gkd_tag_str,
        teacher_param_tag_str,
    ) in WANDB_STUDENTS[p.name]


def wandb_jobs_expected(p: Partition) -> int:
    """Training jobs per partition that keep W&B on."""
    return len(WANDB_STUDENTS[p.name]) + (1 if p.name in WANDB_TEACHERS else 0)


# ---- CampaignSpec per partition ----------------------------------------------------


def _spec(p: Partition) -> CampaignSpec:
    priorities = queue_priorities(p)
    names = list(priorities)
    first = names[0]
    if p.gen == 2:
        teacher_rs = p.run_set_id
    else:
        teacher_rs = MAIN_RUN_SET_ID if p.name != "ladder" else LADDER_RUN_SET_ID
    return CampaignSpec(
        run_set_id=p.run_set_id,
        teacher_run_set_id=teacher_rs,
        hub_org=HUB_ORG,
        teacher_workflow=TEACHER_WORKFLOW,
        teacher_param_tag_lora=TEACHER_PARAM_TAG_LORA,
        teacher_param_tag_full=TEACHER_PARAM_TAG_FULL,
        animals=ANIMALS if p.gen == 1 else GEN2_ANIMALS,
        betas=(BETA,),
        lmbda=LMBDA,
        # Stage A/B/C fields are not applicable; filled so the spec constructs
        # and every q_* names a queue with a priority. No ws160m code reads them.
        a1_student_modes=(STUDENT_MODE,),
        a1_lora_lrs=(STUDENT_LR,),
        a1_lr_extension_factor=1.0,
        a1_gseeds=GSEEDS,
        stage_a_tfs=FTSEEDS,
        a1_same_pair=("pythia-160m", "pythia-160m"),
        a1_cross_pair=("pythia-160m", "pile-ws1"),
        a2_student_mode=STUDENT_MODE,
        control_floor_max=0.02,
        control_floor_max_spread=0.01,
        a3_doses=(FROZEN_EPOCHS,),
        a3_top_k=1,
        a3_gseeds=GSEEDS,
        a3_max_same_arms=1,
        a3_reference_recipe=FROZEN_RECIPE,
        b_tfs_headline=FTSEEDS,
        b_tfs_extension=(FTSEEDS[0],),
        b_gseeds=GSEEDS,
        c_lambdas=(LMBDA,),
        c_gseeds=GSEEDS,
        q_a1=first,
        q_a2=first,
        q_a3=first,
        q_b_headline=first,
        q_b_ext=first,
        q_c=first,
        q_a1_evals=first,
        q_a3_evals=first,
        q_b_evals=first,
        q_c_evals=first,
        q_backlog=first,
        queue_priorities=priorities,
        gpu_policy=GPU_POLICY,
        a1_same_yaml="",
        a1_cross_yaml="",
        a1_same_verify_yaml="",
        a1_cross_verify_yaml="",
        generated_config_dir=p.generated_config_dir,
        gkd_experiment_config=STUDENT_EXPERIMENT_CONFIG,
        catswide_verify_yamls=(),
        catswide_base_models=(),
        calibration_report=PREFLIGHT_REPORT,
        a1_selection_report=TEACHER_TIER_REPORT,
        freeze_report=WAVE2_REPORT,
        init_equality_report=p.staging_report,
        campaign_state_file=CAMPAIGN_STATE_FILE,
        campaign_done_file=CAMPAIGN_DONE_FILE,
        control_log_file=CONTROL_LOG_FILE,
        eval_prompt_sets=EVAL_PROMPT_SETS,
        extra_pin_paths=(
            "scripts/sweeps/sweep_sdf_teacher.py",
            "scripts/sweeps/sweep_evaluate_sdf_teacher.py",
            TEACHER_EXPERIMENT_CONFIG,
        ),
    )


SPECS: Mapping[str, CampaignSpec] = MappingProxyType(
    {name: _spec(p) for name, p in (*PARTITIONS.items(), *GEN2_PARTITIONS.items())}
)
SPEC = SPECS["main"]


def spec_of(name: str) -> CampaignSpec:
    """The `CampaignSpec` of partition `name`."""
    partition(name)
    return SPECS[name]
