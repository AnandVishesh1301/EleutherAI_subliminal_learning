"""Fixed decisions of the 20260828 bear pretraining-checkpoint grid campaign.

The single source for every constant of
`.cursor/plans/bear_checkpoint_grid_campaign_7e121636.plan.md` and
`docs/PREREG_CKPT_XGRID.md`: the teacher / student / joint rung sets, the
original-only grid arm, the joint-seed pairings, the frozen teacher/student
recipe, the behavioural teacher gate, quarantine bounds, queue names/priorities,
the GPU policy, and the artifact filenames. Nothing measured is hardcoded here.

Sibling of `campaign.ckpt_plan` rather than an extension of it: the two campaigns
share driver machinery (`campaign.driver_io`, `campaign.expected`) but no grid
constant, so an edit to the cat ladder cannot move this campaign.

`SPEC` exists because `driver_io.CampaignDriverConfig.__post_init__` overwrites
``run_set_id`` and ``queue_priorities`` from its ``spec`` field unconditionally.
The ckptgrid driver (wave-2) therefore passes ``spec=SPEC``. This module does
**not** register the spec in `driver_io.CAMPAIGN_SPECS`.

Several `CampaignSpec` fields describe the student-HP stage machine (Stage A1/A2/A3
screening, Stage C ablations) and have no counterpart in this campaign; they are
filled with this campaign's frozen values purely so the dataclass is constructible
and `spec.queue_names()` never names a queue with no priority. They are marked
below and are not read by any ckptgrid code path.

Seed-audit constraints (`docs/POLYPYTHIA_160M_SEED_AUDIT.md`):
``pythia-160m-seed{1,2,3}`` are joint (init AND data order). The
``pythia-160m-data-seed{1,2,3}`` family shares the 1234 init with ``original``.
The uploaded init-only ablation repos are not an init ablation and must not
appear in this campaign. The random-init Hub revision is never referenced.
"""

from __future__ import annotations

from types import MappingProxyType

from subliminal_learning.campaign.selection import Recipe
from subliminal_learning.campaign.spec import CampaignSpec
from subliminal_learning.config import LoRAConfig
from subliminal_learning.gkd.layout import gkd_tag


# ---- campaign identity -------------------------------------------------------

# Hyphen (not underscore) inside the purpose slug — `is_valid_run_set_id` allows
# ``[a-z0-9-]*`` only between the date and size segments, and every sweep rejects
# a malformed id at option-parse time.
RUN_SET_ID = "20260828_ckptgrid-bear_160m"
TEACHER_RUN_SET_ID = RUN_SET_ID
HUB_ORG = "EleutherAI"
MODEL_SIZE = "160m"

TEACHER_WORKFLOW = "sdf"
TEACHER_MODE = "lora"
PROMPT_POOL_DIR = "gkd_prompts/pile_prompts_v1"
OWNER = "vishesh"


# ---- rungs -------------------------------------------------------------------

# Behaviour teacher rungs: identical values to the cat ladder's LADDER_B,
# restated rather than imported so the two campaigns cannot drift together.
TEACHER_RUNGS: tuple[int, ...] = (
    512,
    1000,
    2000,
    4000,
    8000,
    16000,
    32000,
    48000,
    64000,
    96000,
    128000,
    143000,
)

# Student rungs extend the teacher set with the two published early checkpoints
# needed for uncensored onset (G6).
STUDENT_RUNGS: tuple[int, ...] = (128, 256) + TEACHER_RUNGS

# Joint-seed arm is diagonal-only on this subset of TEACHER_RUNGS.
JOINT_RUNGS: tuple[int, ...] = (2000, 8000, 16000, 48000, 96000, 143000)

# The published log-spaced head of the Pythia checkpoint schedule; everything
# above `step512` is a multiple of 1000 up to `step143000` (= `main`). The
# leading 0 is the published schedule's first branch and is excluded from every
# campaign rung set.
CHECKPOINT_LOG_HEAD: tuple[int, ...] = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
CHECKPOINT_STEP_INTERVAL = 1000
CHECKPOINT_MAX_STEP = 143000

GRID_BASE = "original"
JOINT_SEED_BASES: tuple[str, ...] = ("seed1", "seed2", "seed3")
DATA_SEED_BASE = "data-seed1"


# ---- frozen recipe -----------------------------------------------------------

TEACHER_PARAM_TAG_LORA = "ep5_lr5e-5_r128a128"
# Stage C's full-FT teacher tag; this campaign never trains it, but SPEC
# requires the field.
TEACHER_PARAM_TAG_FULL = "ep3_lr2e-5_full"
TEACHER_EPOCHS = 5
TEACHER_LR = 5.0e-5
TEACHER_LORA_R = 128
TEACHER_LORA_ALPHA = 128

STUDENT_MODE = "lora-r8a16"
STUDENT_LR = 4.0e-4
STUDENT_LORA = LoRAConfig(
    r=8, lora_alpha=16, lora_dropout=0.0, target_modules="all-linear"
)
STUDENT_RECIPE = Recipe(
    lora_r=STUDENT_LORA.r, lora_alpha=STUDENT_LORA.lora_alpha, lr=STUDENT_LR
)
BETA = 1.0
LMBDA = 1.0
NUM_EPOCHS = 1.0
# Derived, not transcribed: `lmb1.0_b1.0_lora_lr4e-4_e1_r8a16`.
# The literal is asserted in tests/unit/test_ckptgrid_plan.py.
STUDENT_GKD_TAG = gkd_tag(
    lmbda=LMBDA,
    beta=BETA,
    student_mode="lora",
    learning_rate=STUDENT_LR,
    num_epochs=NUM_EPOCHS,
    lora=STUDENT_LORA,
)


# ---- replication and scope ---------------------------------------------------

FTSEEDS: tuple[int, ...] = (0, 1, 2)
GSEEDS: tuple[int, ...] = (0, 1, 2)

ANIMALS: tuple[str, ...] = ("bear",)


# ---- evals -------------------------------------------------------------------

EVAL_PROMPT_SETS: tuple[str, ...] = (
    "test",
    "favorite",
    "storytelling",
    "off_topic",
)
PRIMARY_PROMPT_SET = "test"
PRIMARY_SCORING = "prob"


# ---- teacher gate (behavioural; replaces the ladder's geometry dial) ---------

ADMISSION_SD_MULTIPLE = 5.0
ADMISSION_ABS_MIN = 0.02
TEACHER_GATE_MIN_OK_RUNGS = 6
TEACHER_GATE_MIN_CURVE_SPAN = 0.05


# ---- cell quarantine ---------------------------------------------------------

QUARANTINE_PPL_RATIO_MAX = 5.0
QUARANTINE_FILTER_DROP_MAX = 0.5


# ---- P0 preflight ------------------------------------------------------------

# Cold-1 storage roots are resolved by the wave-2 staging script from paths
# config. Only the two free-space gates live here.
MIN_DISK_FREE_GIB_COLD = 150
MIN_DISK_FREE_GIB_SSD = 30


# ---- claim statistics --------------------------------------------------------

BOOTSTRAP_N = 10000
BOOTSTRAP_SEED = 20260828
ONSET_SD_MULTIPLE = 2.0
ONSET_MARKERS: tuple[float, ...] = (0.0022, 0.01)
EFFICIENCY_TOST_BOUND = 0.05
CLAIM_IDS: tuple[str, ...] = ("G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9")


# ---- jobq queues -------------------------------------------------------------

# `ckptgrid-{phase}-{role}`; phases follow the plan DAG.
# P1 = teachers; P2 = grid arm; P3 = joint arm; P4 = preference evals.
Q_P1_TEACHERS = "ckptgrid-p1-teachers"
Q_P1_TEACHER_EVALS = "ckptgrid-p1-teacher-evals"
Q_P2_STUDENTS = "ckptgrid-p2-students"
Q_P2_CONTROLS = "ckptgrid-p2-controls"
Q_P3_STUDENTS = "ckptgrid-p3-students"
Q_P3_CONTROLS = "ckptgrid-p3-controls"
Q_P4_EVALS = "ckptgrid-p4-evals"

# Priority follows the DAG, not importance: an upstream stage that gates a
# downstream one must drain first.
QUEUE_PRIORITIES: MappingProxyType[str, int] = MappingProxyType(
    {
        Q_P1_TEACHERS: 100,
        Q_P1_TEACHER_EVALS: 95,
        Q_P2_STUDENTS: 80,
        Q_P2_CONTROLS: 80,
        Q_P3_STUDENTS: 75,
        Q_P3_CONTROLS: 75,
        Q_P4_EVALS: 60,
    }
)


# ---- cluster policy ----------------------------------------------------------

# Same packing as the cat ladder: `cap_per_gpu=2` x `free_mem_mib=17000` on
# the GKD LoRA / SDF teacher cell. Behaviour-only: no geometry/inference
# overrides.
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

    `GPU_POLICY` is a `MappingProxyType` of tuples so no caller can mutate the
    campaign's frozen policy; the launcher has to write
    ``gpu_policy.{host}.json``, and `json.dumps` rejects a mapping proxy. This is
    the one place the conversion happens.

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

GENERATED_CONFIG_DIR = "configs/experiment/generated/ckptgrid"
TEACHER_TRAINING_CONFIG = "configs/training/lora_sdf_teacher.yaml"
# Isolated W&B project (entity stays in `.env` / wandb login). Log-few is
# job-side `--no-wandb` via the predicates below, not GPU_POLICY.env.
WANDB_PROJECT = "subliminal-learning-ckptgrid-bear"
STUDENT_EXPERIMENT_CONFIG = "configs/experiment/ckptgrid_student.yaml"
TEACHER_EXPERIMENT_CONFIG = "configs/experiment/ckptgrid_teacher.yaml"

# Headline log-few (locked): one teacher + two student shards at replicate 0.
WANDB_HEADLINE_STEP = CHECKPOINT_MAX_STEP
WANDB_HEADLINE_TEACHER_TOKEN = f"{GRID_BASE}@step{WANDB_HEADLINE_STEP}"
WANDB_HEADLINE_GRID_STUDENT = WANDB_HEADLINE_TEACHER_TOKEN
WANDB_HEADLINE_JOINT_STUDENT = f"{JOINT_SEED_BASES[0]}@step{WANDB_HEADLINE_STEP}"


def wandb_headline_grid_pair_index() -> int:
    """Pair index of original@143000 ← original@143000 in `grid_pairs()` order."""
    return TEACHER_RUNGS.index(WANDB_HEADLINE_STEP) * len(
        STUDENT_RUNGS
    ) + STUDENT_RUNGS.index(WANDB_HEADLINE_STEP)


def wandb_headline_grid_class() -> int:
    """Residue class of the grid-arm diagonal headline pair."""
    return wandb_headline_grid_pair_index() % len(FTSEEDS)


def wandb_headline_joint_class() -> int:
    """Residue class of seed1←seed1 (`JOINT_PAIRINGS` index 0)."""
    return 0


def teacher_logs_wandb(*, token: str, ftseed: int) -> bool:
    """Whether this teacher train cell creates a W&B run."""
    return token == WANDB_HEADLINE_TEACHER_TOKEN and ftseed == FTSEEDS[0]


def student_logs_wandb(
    *,
    tag: str,
    student: str,
    residue_cls: int,
    replicate: int,
) -> bool:
    """Whether this student train shard creates a W&B run.

    Grid (`tag="p2"`): original@step143000 in the diagonal pair's class.
    Joint (`tag="p3"`): seed1@step143000 in the seed1←seed1 class. Replicate 0
    only. Controls and every other shard stay local.
    """
    if replicate != 0:
        return False
    if tag == "p2":
        return (
            student == WANDB_HEADLINE_GRID_STUDENT
            and residue_cls == wandb_headline_grid_class()
        )
    if tag == "p3":
        return (
            student == WANDB_HEADLINE_JOINT_STUDENT
            and residue_cls == wandb_headline_joint_class()
        )
    return False


CAMPAIGN_STATE_FILE = "campaign_state.json"
CAMPAIGN_DONE_FILE = "CAMPAIGN_DONE"
CONTROL_LOG_FILE = "campaign_control_log.jsonl"

PREFLIGHT_REPORT = "preflight_report.json"
TEACHER_GATE_REPORT = "teacher_gate_report.json"
PREFETCH_MANIFEST = "prefetch_manifest.json"

RESULTS_TEACHER_CURVE_CSV = "teacher_curve.csv"
RESULTS_GRID_DELTA_CSV = "grid_delta.csv"
RESULTS_GRID_MARGINALS_CSV = "grid_marginals.csv"
RESULTS_GRID_EFFICIENCY_CSV = "grid_efficiency.csv"
RESULTS_GRID_ASYMMETRY_CSV = "grid_asymmetry.csv"
RESULTS_JOINT_BRACKETING_CSV = "joint_bracketing.csv"
RESULTS_GRID_ONSET_CSV = "grid_onset.csv"
RESULTS_GRID_QUARANTINE_CSV = "grid_quarantine.csv"
RESULTS_CLAIMS_JSON = "claims.json"


SPEC = CampaignSpec(
    run_set_id=RUN_SET_ID,
    teacher_run_set_id=TEACHER_RUN_SET_ID,
    hub_org=HUB_ORG,
    teacher_workflow=TEACHER_WORKFLOW,
    teacher_param_tag_lora=TEACHER_PARAM_TAG_LORA,
    teacher_param_tag_full=TEACHER_PARAM_TAG_FULL,
    animals=ANIMALS,
    betas=(BETA,),
    lmbda=LMBDA,
    # --- Stage A/B/C fields: not applicable to the checkpoint grid. Filled with
    # this campaign's frozen recipe / seeds so the spec is constructible and every
    # `q_*` names a queue that has a priority. No ckptgrid code path reads them.
    a1_student_modes=(STUDENT_MODE,),
    a1_lora_lrs=(STUDENT_LR,),
    a1_lr_extension_factor=1.0,
    a1_gseeds=GSEEDS,
    stage_a_tfs=FTSEEDS,
    a1_same_pair=(GRID_BASE, GRID_BASE),
    a1_cross_pair=(GRID_BASE, JOINT_SEED_BASES[0]),
    a2_student_mode=STUDENT_MODE,
    control_floor_max=0.02,
    control_floor_max_spread=0.01,
    a3_doses=(NUM_EPOCHS,),
    a3_top_k=1,
    a3_gseeds=GSEEDS,
    a3_max_same_arms=1,
    a3_reference_recipe=STUDENT_RECIPE,
    b_tfs_headline=FTSEEDS,
    b_tfs_extension=(FTSEEDS[0],),
    b_gseeds=GSEEDS,
    c_lambdas=(LMBDA,),
    c_gseeds=GSEEDS,
    q_a1=Q_P1_TEACHERS,
    q_a2=Q_P1_TEACHER_EVALS,
    q_a3=Q_P2_STUDENTS,
    q_b_headline=Q_P2_STUDENTS,
    q_b_ext=Q_P3_STUDENTS,
    q_c=Q_P3_CONTROLS,
    q_a1_evals=Q_P1_TEACHER_EVALS,
    q_a3_evals=Q_P4_EVALS,
    q_b_evals=Q_P4_EVALS,
    q_c_evals=Q_P4_EVALS,
    q_backlog=Q_P2_CONTROLS,
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
    a1_selection_report=TEACHER_GATE_REPORT,
    freeze_report=TEACHER_GATE_REPORT,
    init_equality_report=PREFETCH_MANIFEST,
    campaign_state_file=CAMPAIGN_STATE_FILE,
    campaign_done_file=CAMPAIGN_DONE_FILE,
    control_log_file=CONTROL_LOG_FILE,
    eval_prompt_sets=EVAL_PROMPT_SETS,
)
