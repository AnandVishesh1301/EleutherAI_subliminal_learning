"""Frozen constants of the ws160m campaign (six Pythia-160M ablation categories).

Plan of record: `.claude/plans/ws160m_headline_opd_campaign.plan.md`; claims W0-W8
in `docs/PREREG_WS160M.md`. One run set per partition, one partition per node
(`PARTITIONS`), all sharing the headline on-policy recipe: reverse KL (β=1), λ=1,
LoRA r8/α16 lr 4e-4, one epoch over the 20k Pile prompt pool.
"""

from __future__ import annotations

from collections.abc import Mapping
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
DRAIN_TIMEOUT_S = 2 * 3600
MAX_FAILED_FRAC = 0.02
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
        "cap_per_gpu": 3,
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
STUDENT_MEM_MIB = 15000
TEACHER_MEM_MIB = 6000
EVAL_MEM_MIB = 6000
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

GENERATED_CONFIG_ROOT = "configs/experiment/generated/ws160m"
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
    """

    name: str
    node: str
    alias: str
    login: str
    lock_prefix: str | None
    arms: tuple[str, ...]
    trains_teachers: bool

    @property
    def run_set_id(self) -> str:
        """``{DATE}_ws160m-{name}_160m``."""
        return f"{DATE}_{CAMPAIGN}-{self.name}_{MODEL_SIZE}"

    @property
    def queue_prefix(self) -> str:
        """``ws160m-{name}``."""
        return f"{CAMPAIGN}-{self.name}"

    @property
    def generated_config_dir(self) -> str:
        """Repo-relative dir of this partition's generated YAMLs."""
        return f"{GENERATED_CONFIG_ROOT}/{self.name}"

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


def partition(name: str) -> Partition:
    """The partition called `name`."""
    try:
        return PARTITIONS[name]
    except KeyError:
        raise ValueError(
            f"unknown partition {name!r}; expected {PARTITION_ORDER}"
        ) from None


def partition_of_arm(arm: str) -> Partition:
    """The partition that trains `arm`."""
    for p in PARTITIONS.values():
        if arm in p.arms:
            return p
    raise ValueError(f"unknown arm {arm!r}; expected {ARMS}")


MAIN_RUN_SET_ID = PARTITIONS["main"].run_set_id
LADDER_RUN_SET_ID = PARTITIONS["ladder"].run_set_id


def teacher_run_set_id(teacher_step: int) -> str:
    """Run set that owns a teacher: rung teachers live in `ladder`, the rest in `main`."""
    return MAIN_RUN_SET_ID if int(teacher_step) == FINAL_STEP else LADDER_RUN_SET_ID


def teacher_owner(teacher_step: int) -> Partition:
    """Partition that trains a teacher at `teacher_step`."""
    return PARTITIONS["main" if int(teacher_step) == FINAL_STEP else "ladder"]


# ---- queues -----------------------------------------------------------------------


def queue(p: Partition, stage: str) -> str:
    """``ws160m-{partition}-{stage}``."""
    return f"{p.queue_prefix}-{stage}"


def _hp_stage(recipe: Recipe) -> str:
    return f"hp-{recipe_slug(recipe)}"


def queue_priorities(p: Partition) -> Mapping[str, int]:
    """Queue name -> jobq priority for one partition, highest first."""
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
        order += [_hp_stage(r) for r in HP_RECIPES] + ["evals-hp"]
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
    order.append("evals-force")
    return MappingProxyType(
        {queue(p, stage): 100 - 3 * i for i, stage in enumerate(order)}
    )


def all_queue_names() -> tuple[str, ...]:
    """Every queue of every partition; unique fleet-wide by construction."""
    names = [q for p in PARTITIONS.values() for q in queue_priorities(p)]
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
    return CampaignSpec(
        run_set_id=p.run_set_id,
        teacher_run_set_id=MAIN_RUN_SET_ID if p.name != "ladder" else LADDER_RUN_SET_ID,
        hub_org=HUB_ORG,
        teacher_workflow=TEACHER_WORKFLOW,
        teacher_param_tag_lora=TEACHER_PARAM_TAG_LORA,
        teacher_param_tag_full=TEACHER_PARAM_TAG_FULL,
        animals=ANIMALS,
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
        init_equality_report=STAGING_REPORT,
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
    {name: _spec(p) for name, p in PARTITIONS.items()}
)
SPEC = SPECS["main"]


def spec_of(name: str) -> CampaignSpec:
    """The `CampaignSpec` of partition `name`."""
    partition(name)
    return SPECS[name]
