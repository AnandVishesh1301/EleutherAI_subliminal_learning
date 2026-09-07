"""Fixed decisions of the 20260908 bear checkpoint-grid stage-2 campaign.

The single source for every constant of
`.cursor/plans/checkpoint_grid_stage_2_c1e8a4af.plan.md` §1–3, §7, §10b.2:
the run-set identity, BASES descriptor table, rung sets, teacher-dose and
student-HP recipes, queue names/priorities, GPU policy, W&B headline
predicates, result artifact names, and `SPEC`. Nothing measured is
hardcoded here.

Sibling of `campaign.ckptgrid_plan` rather than an extension of it: the two
campaigns share driver machinery (`campaign.driver_io`) but no grid
constant, so an edit to the 20260828 bear grid cannot move this campaign.

`SPEC` exists because `driver_io.CampaignDriverConfig.__post_init__`
overwrites ``run_set_id`` and ``queue_priorities`` from its ``spec`` field
unconditionally. The ckptgrid2 driver therefore passes ``spec=SPEC``. This
module **does** register the spec in `driver_io.CAMPAIGN_SPECS` as
``"ckptgrid2"``.

Several `CampaignSpec` fields describe the student-HP stage machine (Stage
A1/A2/A3 screening, Stage C ablations) and have no counterpart in this
campaign; they are filled with this campaign's frozen values purely so the
dataclass is constructible and `spec.queue_names()` never names a queue
with no priority. They are marked below and are not read by any ckptgrid2
code path.

Seed-audit constraints (`docs/POLYPYTHIA_160M_SEED_AUDIT.md`):
``weight-seed{1,2,3}`` are wave 2 only (`audit_required=True`). No rung is
``0``. ``step0`` is never a Hub revision.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

from subliminal_learning.campaign.selection import Recipe
from subliminal_learning.campaign.spec import CampaignSpec
from subliminal_learning.config import LoRAConfig
from subliminal_learning.gkd.layout import gkd_tag


# ---- campaign identity -------------------------------------------------------

# Hyphen (not underscore) inside the purpose slug — `is_valid_run_set_id`
# allows ``[a-z0-9-]*`` only between the date and size segments.
RUN_SET_ID = "20260908_ckptgrid2-bear_160m"
TEACHER_RUN_SET_ID = RUN_SET_ID
REFERENCE_RUN_SETS: MappingProxyType[str, str] = MappingProxyType(
    {
        "bear_grid": "20260828_ckptgrid-bear_160m",
        "cat_ladder": "20260805_ckptladder_160m",
    }
)
HUB_ORG = "EleutherAI"
MODEL_SIZE = "160m"
OWNER = "vishesh"
ANIMALS: tuple[str, ...] = ("bear", "cat")
PROMPT_POOL_DIR = "gkd_prompts/pile_prompts_v1"
TEACHER_WORKFLOW = "sdf"
TEACHER_MODE = "lora"


# ---- rungs -------------------------------------------------------------------

# Rectangle TEACHER columns. Rung 512 is a student-only row (H12); it is
# not a 7th teacher column.
JOINT_RUNGS: tuple[int, ...] = (2000, 8000, 16000, 48000, 96000, 143000)
HP_RUNGS: tuple[int, ...] = (512, 2000, 4000, 8000, 16000, 48000, 143000)
DOSE_RUNGS: tuple[int, ...] = (2000, 8000, 48000, 143000)
CAT_RUNGS: tuple[int, ...] = HP_RUNGS

# Published log-spaced head of the Pythia schedule, restated so this
# campaign cannot drift with the 20260828 grid. Leading 0 is excluded
# from every campaign rung set.
CHECKPOINT_LOG_HEAD: tuple[int, ...] = (
    0,
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
)
CHECKPOINT_STEP_INTERVAL = 1000
CHECKPOINT_MAX_STEP = 143000

RECT512_STUDENT_RUNG = 512
RECT512_RECTANGLES: tuple[tuple[str, str], ...] = (
    ("original", "seed2"),
    ("seed2", "original"),
)
# The only Hub snapshot this campaign needs that the reference manifest
# lacks (seed2 was never trained at 512 in the 20260828 grid).
NEW_HUB_REFS: tuple[tuple[str, int], ...] = (("seed2", RECT512_STUDENT_RUNG),)


# ---- bases -------------------------------------------------------------------


@dataclass(frozen=True)
class BaseDescriptor:
    """One PolyPythia 160M base.

    Attributes:
        label: Campaign token.
        init_id: Seed-audit init family.
        stream_id: Seed-audit data stream.
        wave: ``1`` or ``2``.
        audit_required: Wave-2 bases need a passed seed audit.
    """

    label: str
    init_id: str
    stream_id: str
    wave: int
    audit_required: bool


# init_id / stream_id from docs/POLYPYTHIA_160M_SEED_AUDIT.md (E3/E5/E7).
# weight-seed stream_id is "s0" as documented; audit_required=True means
# the claim is unverified until scripts/audit_seed_bases.py passes.
BASES: MappingProxyType[str, BaseDescriptor] = MappingProxyType(
    {
        d.label: d
        for d in (
            BaseDescriptor("original", "i1234", "s0", 1, False),
            BaseDescriptor("data-seed1", "i1234", "d1", 1, False),
            BaseDescriptor("seed2", "i_seed2", "s_seed2", 1, False),
            BaseDescriptor("weight-seed1", "i_ws1", "s0", 2, True),
            BaseDescriptor("weight-seed2", "i_ws2", "s0", 2, True),
            BaseDescriptor("weight-seed3", "i_ws3", "s0", 2, True),
        )
    }
)


# ---- frozen + swept recipes --------------------------------------------------

TEACHER_LR = 5.0e-5
TEACHER_LORA_R = 128
TEACHER_LORA_ALPHA = 128
TEACHER_EPOCHS_FROZEN = 5
DOSE_EPOCHS: tuple[int, ...] = (1, 2, 3, 8)
TEACHER_PARAM_TAG_FULL = "ep3_lr2e-5_full"


def teacher_param_tag(n_epochs: int) -> str:
    """SDF LoRA param tag at the frozen teacher rank and learning rate.

    Args:
        n_epochs: Teacher SFT epoch count.

    Returns:
        ``ep{n}_lr5e-5_r128a128``.
    """
    return f"ep{n_epochs}_lr5e-5_r128a128"


TEACHER_PARAM_TAG_LORA = teacher_param_tag(TEACHER_EPOCHS_FROZEN)

BETA = LMBDA = 1.0
FROZEN_RECIPE = Recipe(lora_r=8, lora_alpha=16, lr=4.0e-4)
FROZEN_EPOCHS = 1.0
STUDENT_MODE = FROZEN_RECIPE.mode_token
STUDENT_LR = FROZEN_RECIPE.lr


def _lora_of(recipe: Recipe) -> LoRAConfig:
    return LoRAConfig(
        r=recipe.lora_r,
        lora_alpha=recipe.lora_alpha,
        lora_dropout=0.0,
        target_modules="all-linear",
    )


STUDENT_LORA = _lora_of(FROZEN_RECIPE)
FROZEN_GKD_TAG = gkd_tag(
    lmbda=LMBDA,
    beta=BETA,
    student_mode="lora",
    learning_rate=FROZEN_RECIPE.lr,
    num_epochs=FROZEN_EPOCHS,
    lora=STUDENT_LORA,
)


@dataclass(frozen=True)
class HpConfig:
    """One student-HP arm (recipe + epoch budget).

    Attributes:
        recipe: LoRA rank, alpha, and learning rate.
        num_epochs: Student epoch budget.
    """

    recipe: Recipe
    num_epochs: float

    @property
    def tag(self) -> str:
        """Directory tag from `gkd.layout.gkd_tag`."""
        return gkd_tag(
            lmbda=LMBDA,
            beta=BETA,
            student_mode="lora",
            learning_rate=self.recipe.lr,
            num_epochs=self.num_epochs,
            lora=_lora_of(self.recipe),
        )


# Frozen recipe deliberately absent (I3): that tag is read from the
# reference grid, never retrained here.
HP_CONFIGS: tuple[HpConfig, ...] = (
    HpConfig(Recipe(8, 16, 1.0e-4), 1.0),
    HpConfig(Recipe(8, 16, 8.0e-4), 1.0),
    HpConfig(Recipe(8, 16, 1.6e-3), 1.0),
    HpConfig(Recipe(32, 64, 4.0e-4), 1.0),
    HpConfig(Recipe(128, 256, 4.0e-4), 1.0),
    HpConfig(Recipe(8, 32, 4.0e-4), 1.0),
    HpConfig(Recipe(8, 16, 4.0e-4), 3.0),
)
PROBE_HP_CONFIGS: tuple[HpConfig, ...] = (
    HP_CONFIGS[0],
    HP_CONFIGS[2],
    HP_CONFIGS[4],
    HP_CONFIGS[6],
)
PROBE_CELLS: tuple[tuple[int, int], ...] = (
    (2000, 8000),
    (48000, 8000),
    (143000, 8000),
    (143000, 2000),
    (143000, 48000),
)


# ---- replication and scope ---------------------------------------------------

FTSEEDS: tuple[int, ...] = (0, 1, 2)
GSEEDS: tuple[int, ...] = (0, 1, 2)
HP_CONTROL_GSEEDS: tuple[int, ...] = (0,)
HP_CONTROL_ESCALATION_GSEEDS: tuple[int, ...] = (1, 2)
HP_CONTROL_ESCALATION_MAX = 0.02


# ---- evals -------------------------------------------------------------------

EVAL_PROMPT_SETS: tuple[str, ...] = (
    "test",
    "test_sample",
    "favorite",
    "storytelling",
    "off_topic",
)
TEACHER_EVAL_PROMPT_SETS: tuple[str, ...] = EVAL_PROMPT_SETS + (
    "register",
    "register_sample",
)
PRIMARY_PROMPT_SET = "test"
PRIMARY_SCORING = "prob"


# ---- cell quarantine ---------------------------------------------------------

QUARANTINE_PPL_RATIO_MAX = 5.0
QUARANTINE_FILTER_DROP_MAX = 0.5


# ---- P0 preflight ------------------------------------------------------------

# Cold-1 storage roots are resolved by the staging script from paths config.
MIN_DISK_FREE_GIB_COLD = 150
MIN_DISK_FREE_GIB_SSD = 30


# ---- teacher-dose gate (labelling; the one hard check is per-rung `ok`) ------

ADMISSION_SD_MULTIPLE = 5.0
ADMISSION_ABS_MIN = 0.02


# ---- claim thresholds (frozen; prereg docs/PREREG_CKPT_XGRID2.md §7) --------

# H1/H2/H11 ridge: per teacher column, `delta(diag) - mean(off-diagonal)`.
# The support rule is a fraction of the columns, not a fixed count: the bear
# rectangles have 6 teacher columns and the cat grid has 7.
RIDGE_ERA_EARLY_MAX_STEP = 16000
# Two thirds of the columns, as an exact integer ratio: a float threshold
# could round the 6-column case to 3 or 5 and silently move the verdict.
RIDGE_SUPPORT_NUMERATOR = 2
RIDGE_SUPPORT_DENOMINATOR = 3


def ridge_support_min_columns(n_columns: int) -> int:
    """Columns whose ridge CI must exceed 0 for maturity matching.

    Args:
        n_columns: Teacher columns scored (6 on the bear rectangles, 7 on cat).

    Returns:
        ``ceil(2 * n_columns / 3)`` by integer arithmetic -- 4 of 6, 5 of 7.
    """
    numer = RIDGE_SUPPORT_NUMERATOR * n_columns
    return -(-numer // RIDGE_SUPPORT_DENOMINATOR)


def ridge_null_max_columns(n_columns: int) -> int:
    """Columns at or below which the weight-identity null is declared.

    Args:
        n_columns: Teacher columns scored.

    Returns:
        ``max(1, n_columns // 6)`` -- 1 for both 6 and 7 columns.
    """
    return max(1, n_columns // 6)


RIDGE_SUPPORT_MIN_COLUMNS = ridge_support_min_columns(len(JOINT_RUNGS))
RIDGE_NULL_MAX_COLUMNS = ridge_null_max_columns(len(JOINT_RUNGS))

# H3 dose-matched maturity: teachers are dose-matched within this band, and
# the rung coefficient must be TOST-equivalent to 0 within this bound.
DOSE_MATCH_TOL = 0.05
DOSE_TOST_BOUND = 0.05

# H6 / H12 onset: `control_mean + max(ONSET_SD_MULTIPLE * sd, ONSET_ABS_FLOOR)`
# for `ONSET_CONSECUTIVE` rungs. The absolute floor is what keeps a
# deterministic control (sd ~ 1e-4) from being cleared by noise.
ONSET_SD_MULTIPLE = 2.0
ONSET_ABS_FLOOR = 0.02
ONSET_CONSECUTIVE = 2
# 0.22% / 1% of the 143000-step budget (literature markers).
ONSET_MARKERS: tuple[float, ...] = (0.0022, 0.01)

# H9 metric invariance: test/prob vs test_sample/sample on the diagonal.
CONCORDANCE_MIN_SPEARMAN = 0.6
CONCORDANCE_MIN_SIGN_AGREEMENT = 0.8

# G5 amendment: mismatch asymmetry is summarised per era of the *younger*
# rung, not as one grand mean. Edges split (<4000, 4000-32000, >=32000).
ASYMMETRY_ERA_EDGES: tuple[int, ...] = (4000, 32000)


# ---- claim statistics + deadline ---------------------------------------------

BOOTSTRAP_N = 10000
BOOTSTRAP_SEED = 20260908
CLAIM_IDS: tuple[str, ...] = (
    "H1",
    "H2",
    "H3",
    "H4",
    "H5",
    "H6",
    "H7",
    "H8",
    "H9",
    "H10",
    "H11",
    "H12",
)
DEADLINE_HOURS_DEFAULT = 48.0
DRAIN_TIMEOUT_S = 3 * 3600
MAX_FAILED_FRAC = 0.02


# ---- jobq queues -------------------------------------------------------------

# `ckptgrid2-{arm}-{role}`; priorities are the documented numbers in plan
# §5 / §10b.2. Evals sit at 70 — between t15 (80) and t2 (60) — so a
# students-then-evals chain is the wrong order.
Q_P1_TEACHERS = "ckptgrid2-p1-teachers"
Q_P1_TEACHER_EVALS = "ckptgrid2-p1-teacher-evals"
Q_T1_STUDENTS = "ckptgrid2-t1-students"
Q_T1_CONTROLS = "ckptgrid2-t1-controls"
Q_T1_CONTROLS_ESC = "ckptgrid2-t1-controls-esc"
Q_DOSE_STUDENTS = "ckptgrid2-dose-students"
Q_T15_STUDENTS = "ckptgrid2-t15-students"
Q_EVALS_T1 = "ckptgrid2-evals-t1"
# Escalation controls need their OWN eval queue: jobq gives no ordering
# inside a queue, so an eval sharing the train queue could be claimed
# first, hit the sweep's NOT-TRAINED skip, and exit 0 -- leaving the
# escalated control trained but never scored.
Q_EVALS_T1_ESC = "ckptgrid2-evals-t1-esc"
Q_EVALS_DOSE = "ckptgrid2-evals-dose"
Q_EVALS_T15 = "ckptgrid2-evals-t15"
Q_EVALS_T2 = "ckptgrid2-evals-t2"
Q_T2_STUDENTS = "ckptgrid2-t2-students"
Q_E_BACKFILL = "ckptgrid2-e-backfill"
Q_F_BACKFILL = "ckptgrid2-f-backfill"
Q_W2_TEACHERS = "ckptgrid2-w2-teachers"
Q_W2_TEACHER_EVALS = "ckptgrid2-w2-teacher-evals"
Q_W2_STUDENTS = "ckptgrid2-w2-students"
Q_W2_CONTROLS = "ckptgrid2-w2-controls"
Q_W2_EVALS = "ckptgrid2-w2-evals"

QUEUE_PRIORITIES: MappingProxyType[str, int] = MappingProxyType(
    {
        Q_P1_TEACHERS: 100,
        Q_P1_TEACHER_EVALS: 95,
        Q_T1_STUDENTS: 90,
        Q_T1_CONTROLS: 90,
        Q_T1_CONTROLS_ESC: 90,
        Q_DOSE_STUDENTS: 85,
        Q_T15_STUDENTS: 80,
        Q_EVALS_T1: 70,
        Q_EVALS_T1_ESC: 70,
        Q_EVALS_DOSE: 70,
        Q_EVALS_T15: 70,
        Q_EVALS_T2: 70,
        Q_T2_STUDENTS: 60,
        Q_E_BACKFILL: 50,
        Q_F_BACKFILL: 50,
        Q_W2_TEACHERS: 40,
        Q_W2_TEACHER_EVALS: 40,
        Q_W2_STUDENTS: 40,
        Q_W2_CONTROLS: 40,
        Q_W2_EVALS: 40,
    }
)


# ---- cluster policy ----------------------------------------------------------

# Same packing as the 20260828 grid: `cap_per_gpu=2` x `free_mem_mib=17000`
# on the GKD LoRA / SDF teacher cell. Values restated, not imported.
GPU_POLICY: MappingProxyType[str, object] = MappingProxyType(
    {
        "gpus": (0, 1, 2, 3, 4, 5, 6, 7),
        "cap_per_gpu": 2,
        "free_mem_mib": 17000,
        "reservation_warmup_s": 300,
        "env": MappingProxyType(
            {
                "OMP_NUM_THREADS": "4",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }
        ),
    }
)


def gpu_policy_json() -> dict:
    """A plain, JSON-serialisable copy of `GPU_POLICY`.

    `GPU_POLICY` is a `MappingProxyType` of tuples so no caller can mutate
    the campaign's frozen policy; the launcher has to write
    ``gpu_policy.{host}.json``, and `json.dumps` rejects a mapping proxy.
    This is the one place the conversion happens.

    Returns:
        ``{"gpus": [...], "cap_per_gpu": int, "free_mem_mib": int,
        "reservation_warmup_s": int, "env": {...}}`` -- the key set
        `jobqueue.gpu.GpuPolicy.from_dict` reads.
    """
    policy = dict(GPU_POLICY)
    policy["gpus"] = list(GPU_POLICY["gpus"])  # type: ignore[arg-type]
    policy["env"] = dict(GPU_POLICY["env"])  # type: ignore[arg-type]
    return policy


# ---- static configs + artifacts ----------------------------------------------

GENERATED_CONFIG_DIR = "configs/experiment/generated/ckptgrid2"
TEACHER_TRAINING_CONFIG = "configs/training/lora_sdf_teacher.yaml"
# Isolated W&B project (entity stays in `.env` / wandb login). Log-few is
# job-side `--no-wandb` via the predicates below, not GPU_POLICY.env.
WANDB_PROJECT = "subliminal-learning-ckptgrid2-bear"
STUDENT_EXPERIMENT_CONFIG = "configs/experiment/ckptgrid2_student.yaml"
TEACHER_EXPERIMENT_CONFIG = "configs/experiment/ckptgrid2_teacher.yaml"

# Headline log-few (locked, §7): one teacher + three student cells at
# replicate (ftseed, gseed) == (0, 0).
WANDB_HEADLINE_TEACHER_TOKEN = "original@step8000"
WANDB_HEADLINE_TEACHER_TAG = teacher_param_tag(8)
_HEADLINE_HP_TAG = next(cfg.tag for cfg in HP_CONFIGS if cfg.recipe.lr == 1.6e-3)
_HEADLINE_FROZEN_TEACHER_TAG = teacher_param_tag(TEACHER_EPOCHS_FROZEN)
_HEADLINE_EP1_TEACHER_TAG = teacher_param_tag(1)
_WANDB_HEADLINE_STUDENTS: frozenset[tuple[str, str, str, str, str]] = frozenset(
    (
        (
            "hp",
            "original@step143000",
            "original@step143000",
            _HEADLINE_HP_TAG,
            _HEADLINE_FROZEN_TEACHER_TAG,
        ),
        (
            "rect",
            "original@step143000",
            "seed2@step8000",
            FROZEN_GKD_TAG,
            _HEADLINE_FROZEN_TEACHER_TAG,
        ),
        (
            "dose",
            "original@step8000",
            "original@step8000",
            FROZEN_GKD_TAG,
            _HEADLINE_EP1_TEACHER_TAG,
        ),
    )
)


def teacher_logs_wandb(*, token: str, tag: str, ftseed: int) -> bool:
    """Whether this teacher train cell creates a W&B run.

    Args:
        token: Teacher base@revision token.
        tag: Teacher param tag.
        ftseed: Teacher fine-tune seed.

    Returns:
        True only for the §7 headline teacher at ftseed 0.
    """
    return (
        token == WANDB_HEADLINE_TEACHER_TOKEN
        and tag == WANDB_HEADLINE_TEACHER_TAG
        and ftseed == 0
    )


def student_logs_wandb(
    *,
    arm: str,
    student: str,
    teacher: str,
    gkd_tag: str,
    teacher_param_tag: str,
    replicate: int,
) -> bool:
    """Whether this student train cell creates a W&B run.

    Keyed on the replicate, not on ``(ftseed, gseed) == (0, 0)``: leaves
    come from the rotated diagonal, where ftseed 0 pairs with gseed
    ``index % 3``, so ``(0, 0)`` is unreachable for two thirds of the
    cells -- including all three headline cells. Replicate 0 is the
    well-defined "first leaf of this cell" that plan §7 means.

    Args:
        arm: Pair-builder arm token.
        student: Student base@revision token.
        teacher: Teacher base@revision token.
        gkd_tag: Student directory tag.
        teacher_param_tag: Teacher param tag.
        replicate: Position within the cell's three diagonal leaves.

    Returns:
        True only for replicate 0 of a §7 headline cell.
    """
    if replicate != 0:
        return False
    return (
        arm,
        student,
        teacher,
        gkd_tag,
        teacher_param_tag,
    ) in _WANDB_HEADLINE_STUDENTS


CAMPAIGN_STATE_FILE = "campaign_state.json"
CAMPAIGN_DONE_FILE = "CAMPAIGN_DONE"
CONTROL_LOG_FILE = "campaign_control_log.jsonl"

PREFLIGHT_REPORT = "preflight_report.json"
DOSE_GATE_REPORT = "dose_gate_report.json"
ESCALATION_REPORT = "hp_control_escalation.json"
SEED_AUDIT_REPORT = "seed_audit.json"
PREFETCH_MANIFEST = "prefetch_manifest.json"
REFERENCE_SNAPSHOT_DIR = "reference_snapshot"

RESULTS_TEACHER_DOSE_CSV = "teacher_dose.csv"
RESULTS_DOSE_RESPONSE_CSV = "dose_response.csv"
RESULTS_HP_GRID_DELTA_CSV = "hp_grid_delta.csv"
RESULTS_HP_CONTROLS_CSV = "hp_controls.csv"
RESULTS_HP_COLLAPSE_CSV = "hp_collapse.csv"
RESULTS_RECT_DELTA_CSV = "rect_delta.csv"
RESULTS_RIDGE_GAIN_CSV = "ridge_gain.csv"
RESULTS_PENALTY_MAP_CSV = "penalty_map.csv"
RESULTS_EMISSIVITY_FIT_CSV = "emissivity_fit.csv"
RESULTS_ONSET_CSV = "onset.csv"
RESULTS_METRIC_CONCORDANCE_CSV = "metric_concordance.csv"
RESULTS_CAT_GRID_DELTA_CSV = "cat_grid_delta.csv"
RESULTS_BASE_PPL_CSV = "base_ppl.csv"
RESULTS_QUARANTINE_CSV = "quarantine.csv"
RESULTS_COVERAGE_JSON = "coverage.json"
RESULTS_CLAIMS_JSON = "claims.json"
RESULTS_ROBUSTNESS_DIR = "robustness"


SPEC = CampaignSpec(
    run_set_id=RUN_SET_ID,
    teacher_run_set_id=TEACHER_RUN_SET_ID,
    hub_org=HUB_ORG,
    teacher_workflow=TEACHER_WORKFLOW,
    teacher_param_tag_lora=teacher_param_tag(TEACHER_EPOCHS_FROZEN),
    teacher_param_tag_full=TEACHER_PARAM_TAG_FULL,
    animals=ANIMALS,
    betas=(BETA,),
    lmbda=LMBDA,
    # --- Stage A/B/C fields: not applicable to the checkpoint grid. Filled
    # with this campaign's frozen recipe / seeds so the spec is
    # constructible and every `q_*` names a queue that has a priority. No
    # ckptgrid2 code path reads them.
    a1_student_modes=(STUDENT_MODE,),
    a1_lora_lrs=(STUDENT_LR,),
    a1_lr_extension_factor=1.0,
    a1_gseeds=GSEEDS,
    stage_a_tfs=FTSEEDS,
    a1_same_pair=("original", "original"),
    a1_cross_pair=("original", "seed2"),
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
    q_a1=Q_P1_TEACHERS,
    q_a2=Q_P1_TEACHER_EVALS,
    q_a3=Q_T1_STUDENTS,
    q_b_headline=Q_T1_STUDENTS,
    q_b_ext=Q_T15_STUDENTS,
    q_c=Q_T2_STUDENTS,
    q_a1_evals=Q_P1_TEACHER_EVALS,
    q_a3_evals=Q_EVALS_T1,
    q_b_evals=Q_EVALS_T1,
    q_c_evals=Q_EVALS_T2,
    q_backlog=Q_T1_CONTROLS,
    # --- end not-applicable block ---
    queue_priorities=QUEUE_PRIORITIES,
    gpu_policy=GPU_POLICY,
    a1_same_yaml="",
    a1_cross_yaml="",
    a1_same_verify_yaml="",
    a1_cross_verify_yaml="",
    generated_config_dir=GENERATED_CONFIG_DIR,
    gkd_experiment_config=STUDENT_EXPERIMENT_CONFIG,
    catswide_verify_yamls=(),
    catswide_base_models=(),
    calibration_report=PREFLIGHT_REPORT,
    a1_selection_report=DOSE_GATE_REPORT,
    freeze_report=DOSE_GATE_REPORT,
    init_equality_report=PREFETCH_MANIFEST,
    campaign_state_file=CAMPAIGN_STATE_FILE,
    campaign_done_file=CAMPAIGN_DONE_FILE,
    control_log_file=CONTROL_LOG_FILE,
    eval_prompt_sets=EVAL_PROMPT_SETS,
)
