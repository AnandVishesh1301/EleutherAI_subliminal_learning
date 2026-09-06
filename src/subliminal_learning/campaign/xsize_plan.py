"""Fixed decisions of the 20260907 cross-size GKD campaign.

Single source for every constant of
`.cursor/plans/cross-size_polypythia_campaign_9c4e1f27.plan.md` and
`docs/PREREG_XSIZE.md`. Sibling of `campaign.ckptgrid_plan`: the two
campaigns share driver machinery but no constant. Do not import
`ckptgrid_*`.

410m student `grad_checkpointing` is False. Installed TRL 0.19.1
`GKDTrainer` builds its rollout `GenerationConfig` with
`use_cache=False if args.gradient_checkpointing else True`
(trl/trainer/gkd_trainer.py:135), which would make every on-policy
rollout uncached. Memory margin is micro-batch 16 x accum 2 on a
whole card (`slots=2`) instead.

410m teacher uses micro-batch 4 x accum 2 with gradient
checkpointing on (SFT path, no generation). fp32 activations at
8 x 2048 tokens over 24 layers would not fit; effective batch 8 is
unchanged.

`SPEC` exists because `driver_io.CampaignDriverConfig.__post_init__`
overwrites ``run_set_id`` and ``queue_priorities`` from its ``spec``.
Stage A/B/C fields have no counterpart here and are filled with this
campaign's frozen values so the dataclass is constructible.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

from subliminal_learning.campaign.selection import Recipe
from subliminal_learning.campaign.spec import CampaignSpec
from subliminal_learning.config import LoRAConfig
from subliminal_learning.gkd.layout import gkd_tag


# ---- campaign identity -------------------------------------------------------

RUN_SET_ID = "20260907_xsize-bear_mixed"
TEACHER_RUN_SET_ID = RUN_SET_ID
MODEL_SIZE = "mixed"
HUB_ORG = "EleutherAI"
OWNER = "vishesh"
TEACHER_WORKFLOW = "sdf"
TEACHER_MODE = "lora"
PROMPT_POOL_DIR = "gkd_prompts/pile_prompts_v1"


# ---- bases -------------------------------------------------------------------

SIZES: tuple[str, ...] = ("70m", "160m", "410m")
BASE_STEP = 143000
BASE_REVISION = "step143000"
BASE_TOKEN: MappingProxyType[str, str] = MappingProxyType(
    {s: f"pythia-{s}@step143000" for s in SIZES}
)

ANIMALS: tuple[str, ...] = ("bear",)
FTSEEDS: tuple[int, ...] = (0, 1, 2)
GSEEDS: tuple[int, ...] = (0, 1, 2)


# ---- teachers ----------------------------------------------------------------

TEACHER_PARAM_TAG = "ep5_lr5e-5_r128a128"
TEACHER_PARAM_TAG_LORA = TEACHER_PARAM_TAG
TEACHER_PARAM_TAG_FULL = "ep3_lr2e-5_full"
TEACHER_FALLBACK_TAGS: tuple[str, ...] = (
    "ep5_lr1e-4_r128a128",
    "ep10_lr5e-5_r128a128",
)
TEACHER_FALLBACK_SIZES: tuple[str, ...] = ("70m", "410m")


@dataclass(frozen=True)
class TeacherRecipe:
    """Frozen SDF teacher hyperparameter tag.

    Attributes:
        tag: Directory tag.
        epochs: SFT epochs.
        lr: Learning rate.
        lora_r: LoRA rank.
        lora_alpha: LoRA alpha.
    """

    tag: str
    epochs: int
    lr: float
    lora_r: int = 128
    lora_alpha: int = 128


TEACHER_RECIPES: MappingProxyType[str, TeacherRecipe] = MappingProxyType(
    {
        "ep5_lr5e-5_r128a128": TeacherRecipe("ep5_lr5e-5_r128a128", 5, 5.0e-5),
        "ep5_lr1e-4_r128a128": TeacherRecipe("ep5_lr1e-4_r128a128", 5, 1.0e-4),
        "ep10_lr5e-5_r128a128": TeacherRecipe("ep10_lr5e-5_r128a128", 10, 5.0e-5),
    }
)


# ---- students ----------------------------------------------------------------

STUDENT_MODE = "lora-r8a16"
STUDENT_LORA = LoRAConfig(
    r=8, lora_alpha=16, lora_dropout=0.0, target_modules="all-linear"
)
BETA = 1.0
LMBDA = 1.0
NUM_EPOCHS = 1.0
LR_LADDER: tuple[float, ...] = (2.0e-4, 4.0e-4, 8.0e-4)
LR_DEFAULT = 4.0e-4
STUDENT_RECIPE = Recipe(
    lora_r=STUDENT_LORA.r, lora_alpha=STUDENT_LORA.lora_alpha, lr=LR_DEFAULT
)


def student_gkd_tag(lr: float) -> str:
    """GKD directory tag at `lr` (λ=1, β=1, LoRA r8a16, 1 epoch)."""
    return gkd_tag(
        lmbda=LMBDA,
        beta=BETA,
        student_mode="lora",
        learning_rate=lr,
        num_epochs=NUM_EPOCHS,
        lora=STUDENT_LORA,
    )


# ---- pairing scope -----------------------------------------------------------

ONE_STEP_PAIRS: tuple[tuple[str, str], ...] = (
    ("70m", "160m"),
    ("160m", "70m"),
    ("160m", "410m"),
    ("410m", "160m"),
)
LR_ROBUSTNESS_STUDENTS: tuple[str, ...] = ("70m", "160m")


# ---- gates -------------------------------------------------------------------

PICK_PPL_RATIO_MAX = 1.5
PICK_TIE_BAND = 0.02
ADMISSION_SD_MULTIPLE = 5.0
ADMISSION_ABS_MIN = 0.02
TEACHER_PPL_RATIO_MAX = 2.0
MIN_ADMITTED_PER_BASE = 2
QUARANTINE_PPL_RATIO_MAX = 5.0
QUARANTINE_FILTER_DROP_MAX = 0.5


# ---- evals + claims ----------------------------------------------------------

EVAL_PROMPT_SETS: tuple[str, ...] = ("test", "favorite", "off_topic")
PRIMARY_PROMPT_SET = "test"
PRIMARY_SCORING = "prob"
BOOTSTRAP_N = 10000
BOOTSTRAP_SEED = 20260907
CLAIM_IDS: tuple[str, ...] = ("X0", "X1", "X2", "X3", "X4")
X0_MIN_DELTA = 0.02
X2_ASYM_MIN = 0.05
X2_ADDITIVE_R2_MIN = 0.85
X4_RETAINED_FRACTION_MIN = 0.5

GRID_REFERENCE: MappingProxyType[str, object] = MappingProxyType(
    {
        "run_set_id": "20260828_ckptgrid-bear_160m",
        "step": 143000,
        "anchor_delta": 0.336,
        "di_ds_delta": 0.265,
        "teacher_p_target": 0.263,
    }
)


# ---- footprints --------------------------------------------------------------


@dataclass(frozen=True)
class Footprint:
    """Per-size job packing. Frozen; never measured at runtime."""

    slots: int
    mem_mib: int
    micro_batch: int
    grad_accum: int
    grad_checkpointing: bool
    checkpoint_resume: bool
    save_steps: int
    est_minutes: int


STUDENT_FOOTPRINT: MappingProxyType[str, Footprint] = MappingProxyType(
    {
        "70m": Footprint(1, 12_000, 32, 1, False, False, 0, 24),
        "160m": Footprint(1, 17_000, 32, 1, False, False, 0, 33),
        "410m": Footprint(2, 40_000, 16, 2, False, True, 125, 83),
    }
)
TEACHER_FOOTPRINT: MappingProxyType[str, Footprint] = MappingProxyType(
    {
        "70m": Footprint(1, 17_000, 8, 1, False, False, 0, 6),
        "160m": Footprint(1, 17_000, 8, 1, False, False, 0, 11),
        "410m": Footprint(2, 30_000, 4, 2, True, False, 0, 30),
    }
)
STUDENT_EFFECTIVE_BATCH = 32
TEACHER_EFFECTIVE_BATCH = 8
SAVE_TOTAL_LIMIT = 1
GPU_MEM_MIB = 48_000


def validate_footprints() -> None:
    """Raise if a frozen footprint violates the packing contract."""
    cap = int(GPU_POLICY["cap_per_gpu"])
    resume_students = []
    for size, fp in STUDENT_FOOTPRINT.items():
        if fp.micro_batch * fp.grad_accum != STUDENT_EFFECTIVE_BATCH:
            raise ValueError(
                f"student {size}: micro_batch * grad_accum "
                f"{fp.micro_batch * fp.grad_accum} != {STUDENT_EFFECTIVE_BATCH}"
            )
        if fp.slots not in (1, 2):
            raise ValueError(f"student {size}: slots {fp.slots} not in (1, 2)")
        if fp.slots > cap:
            raise ValueError(f"student {size}: slots {fp.slots} > cap {cap}")
        if fp.mem_mib > GPU_MEM_MIB:
            raise ValueError(f"student {size}: mem_mib {fp.mem_mib} > {GPU_MEM_MIB}")
        if fp.checkpoint_resume and fp.save_steps <= 0:
            raise ValueError(
                f"student {size}: checkpoint_resume requires save_steps > 0"
            )
        if fp.checkpoint_resume:
            resume_students.append(size)
    if resume_students != ["410m"]:
        raise ValueError(
            f"only the 410m student may set checkpoint_resume; got {resume_students}"
        )
    for size, fp in TEACHER_FOOTPRINT.items():
        if fp.micro_batch * fp.grad_accum != TEACHER_EFFECTIVE_BATCH:
            raise ValueError(
                f"teacher {size}: micro_batch * grad_accum "
                f"{fp.micro_batch * fp.grad_accum} != {TEACHER_EFFECTIVE_BATCH}"
            )
        if fp.slots not in (1, 2):
            raise ValueError(f"teacher {size}: slots {fp.slots} not in (1, 2)")
        if fp.slots > cap:
            raise ValueError(f"teacher {size}: slots {fp.slots} > cap {cap}")
        if fp.mem_mib > GPU_MEM_MIB:
            raise ValueError(f"teacher {size}: mem_mib {fp.mem_mib} > {GPU_MEM_MIB}")
        if fp.checkpoint_resume and fp.save_steps <= 0:
            raise ValueError(
                f"teacher {size}: checkpoint_resume requires save_steps > 0"
            )
        if fp.checkpoint_resume:
            raise ValueError(f"teacher {size}: checkpoint_resume must be off")


# ---- jobq queues -------------------------------------------------------------

Q_P1_TEACHERS = "xsize-p1-teachers"
Q_P1_TEACHER_EVALS = "xsize-p1-teacher-evals"
# Failure path only: tfs 1/2 at the fallback tag for a size whose frozen tag
# did not admit >= MIN_ADMITTED_PER_BASE teachers. Separate queue because
# Q_P1_TEACHERS is already submitted with a frozen jobkey set by then.
Q_P1_FALLBACK = "xsize-p1-fallback"
Q_P1_FALLBACK_EVALS = "xsize-p1-fallback-evals"
Q_P2_LADDER = "xsize-p2-ladder"
Q_P2_CONTROLS = "xsize-p2-controls"
Q_P2_EVALS = "xsize-p2-evals"

_P3_PRIORITY: MappingProxyType[str, int] = MappingProxyType(
    {"410m": 82, "160m": 81, "70m": 80}
)


def _require_size(size: str) -> str:
    if size not in SIZES:
        raise ValueError(f"unknown size {size!r}; expected one of {SIZES}")
    return size


def q_p3_students(size: str) -> str:
    """P3 treatment queue for one student size."""
    return f"xsize-p3-students-{_require_size(size)}"


def q_p3_controls(size: str) -> str:
    """P3 control queue for one student size."""
    return f"xsize-p3-controls-{_require_size(size)}"


def q_p3_evals(size: str) -> str:
    """P3 eval queue for one student size."""
    return f"xsize-p3-evals-{_require_size(size)}"


def q_p3b(size: str) -> str:
    """P3b LR-robustness train queue (70m / 160m only)."""
    if size not in LR_ROBUSTNESS_STUDENTS:
        raise ValueError(
            f"P3b is not defined for student size {size!r}; "
            f"allowed {LR_ROBUSTNESS_STUDENTS}"
        )
    return f"xsize-p3b-lr-{size}"


def q_p3b_evals(size: str) -> str:
    """P3b eval queue (70m / 160m only)."""
    if size not in LR_ROBUSTNESS_STUDENTS:
        raise ValueError(
            f"P3b evals are not defined for student size {size!r}; "
            f"allowed {LR_ROBUSTNESS_STUDENTS}"
        )
    return f"xsize-p3b-evals-{size}"


def p3b_queue_names() -> frozenset[str]:
    """P3b train + eval queues, for every size that has them.

    P3b is the lowest-priority tail and is **not** part of the completion
    criterion: `CAMPAIGN_DONE` and claims X0-X2/X4 need only P1 + P2 + P3, and
    X3 reports whatever P3b leaves exist. The driver excludes these names from
    the finalize gate so an overrun self-truncates from the tail.
    """
    names: set[str] = set()
    for size in LR_ROBUSTNESS_STUDENTS:
        names.add(q_p3b(size))
        names.add(q_p3b_evals(size))
    return frozenset(names)


def _queue_priorities() -> MappingProxyType[str, int]:
    table: dict[str, int] = {
        Q_P1_TEACHERS: 100,
        Q_P1_FALLBACK: 98,
        Q_P1_TEACHER_EVALS: 95,
        Q_P1_FALLBACK_EVALS: 94,
        Q_P2_LADDER: 85,
        Q_P2_CONTROLS: 85,
        Q_P2_EVALS: 75,
    }
    for size in SIZES:
        table[q_p3_students(size)] = _P3_PRIORITY[size]
        table[q_p3_controls(size)] = _P3_PRIORITY[size]
        table[q_p3_evals(size)] = 65
    for size in LR_ROBUSTNESS_STUDENTS:
        table[q_p3b(size)] = 70
        table[q_p3b_evals(size)] = 60
    return MappingProxyType(table)


QUEUE_PRIORITIES: MappingProxyType[str, int] = _queue_priorities()


# ---- cluster policy ----------------------------------------------------------

GPU_POLICY: MappingProxyType[str, object] = MappingProxyType(
    {
        "gpus": (0, 1, 2, 3, 4, 5, 6, 7),
        "cap_per_gpu": 2,
        "free_mem_mib": 12000,
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
AUTO_REQUEUE_ATTEMPTS = 2


def gpu_policy_json() -> dict:
    """A plain, JSON-serialisable copy of `GPU_POLICY`.

    Returns:
        ``{"gpus": [...], "cap_per_gpu": int, "free_mem_mib": int,
        "reservation_warmup_s": int, "env": {...}}``.
    """
    policy = dict(GPU_POLICY)
    policy["gpus"] = list(GPU_POLICY["gpus"])  # type: ignore[arg-type]
    policy["env"] = dict(GPU_POLICY["env"])  # type: ignore[arg-type]
    return policy


validate_footprints()


# ---- W&B ---------------------------------------------------------------------

WANDB_PROJECT = "SL_cross_size_bear"
WANDB_HEADLINE_TEACHER_TOKEN = BASE_TOKEN["160m"]
WANDB_HEADLINE_TEACHER_TAG = TEACHER_PARAM_TAG
WANDB_HEADLINE_STUDENTS: tuple[tuple[str, str], ...] = (
    ("70m", "160m"),
    ("410m", "160m"),
)


def teacher_logs_wandb(*, token: str, tag: str, ftseed: int) -> bool:
    """Whether this teacher train cell creates a W&B run."""
    return (
        token == WANDB_HEADLINE_TEACHER_TOKEN
        and tag == WANDB_HEADLINE_TEACHER_TAG
        and ftseed == FTSEEDS[0]
    )


def student_logs_wandb(
    *,
    teacher: str,
    student: str,
    lr: float,
    picked_lr: float,
    ftseed: int,
    gseed: int,
) -> bool:
    """Whether this student train cell creates a W&B run.

    Headline: ``70m→160m`` and ``410m→160m`` at ``lr*(160m)``, tfs0, gseed0.
    """
    if ftseed != FTSEEDS[0] or gseed != GSEEDS[0]:
        return False
    if (teacher, student) not in WANDB_HEADLINE_STUDENTS:
        return False
    return lr == picked_lr


# ---- static configs + artifacts ----------------------------------------------

GENERATED_CONFIG_DIR = "configs/experiment/generated/xsize"
TEACHER_TRAINING_CONFIG = "configs/training/lora_sdf_teacher.yaml"
STUDENT_TRAINING_CONFIG = "configs/training/gkd_student_lora.yaml"
STUDENT_EXPERIMENT_CONFIG = "configs/experiment/xsize_student.yaml"
TEACHER_EXPERIMENT_CONFIG = "configs/experiment/xsize_teacher.yaml"
PREREG_PATH = "docs/PREREG_XSIZE.md"

CAMPAIGN_STATE_FILE = "campaign_state.json"
CAMPAIGN_DONE_FILE = "CAMPAIGN_DONE"
CONTROL_LOG_FILE = "campaign_control_log.jsonl"
PREFLIGHT_REPORT = "preflight_report.json"
TEACHER_GATE_REPORT = "teacher_gate_report.json"
SCREEN_PICKS = "screen_picks.json"
PREFETCH_MANIFEST = "prefetch_manifest.json"
P0_GATES_REPORT = "p0_gates.json"

RESULTS_LEAVES_CSV = "xsize_leaves.csv"
RESULTS_DELTA_CSV = "xsize_delta.csv"
RESULTS_CELLS_CSV = "xsize_cells.csv"
RESULTS_PENALTY_CSV = "xsize_penalty.csv"
RESULTS_ASYMMETRY_CSV = "xsize_asymmetry.csv"
RESULTS_ADDITIVE_FIT_CSV = "xsize_additive_fit.csv"
RESULTS_LR_CSV = "xsize_lr.csv"
RESULTS_PPL_CSV = "xsize_ppl.csv"
RESULTS_BASE_PRIORS_CSV = "base_priors.csv"
RESULTS_TEACHER_ADMISSION_CSV = "teacher_admission.csv"
RESULTS_QUARANTINE_CSV = "xsize_quarantine.csv"
RESULTS_GRID_CONCORDANCE_CSV = "grid_concordance.csv"
RESULTS_CLAIMS_JSON = "claims.json"

MIN_DISK_FREE_GIB_COLD = 60
MIN_DISK_FREE_GIB_SSD = 30


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
    # --- Stage A/B/C fields: not applicable to the cross-size campaign.
    # Filled with this campaign's frozen recipe / seeds so the spec is
    # constructible and every `q_*` names a queue that has a priority.
    a1_student_modes=(STUDENT_MODE,),
    a1_lora_lrs=LR_LADDER,
    a1_lr_extension_factor=1.0,
    a1_gseeds=GSEEDS,
    stage_a_tfs=FTSEEDS,
    a1_same_pair=("160m", "160m"),
    a1_cross_pair=("70m", "160m"),
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
    q_a3=Q_P2_LADDER,
    q_b_headline=Q_P2_LADDER,
    q_b_ext=q_p3_students("410m"),
    q_c=Q_P2_CONTROLS,
    q_a1_evals=Q_P1_TEACHER_EVALS,
    q_a3_evals=Q_P2_EVALS,
    q_b_evals=q_p3_evals("410m"),
    q_c_evals=q_p3_evals("160m"),
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
    a1_selection_report=SCREEN_PICKS,
    freeze_report=TEACHER_GATE_REPORT,
    init_equality_report=PREFETCH_MANIFEST,
    campaign_state_file=CAMPAIGN_STATE_FILE,
    campaign_done_file=CAMPAIGN_DONE_FILE,
    control_log_file=CONTROL_LOG_FILE,
    eval_prompt_sets=EVAL_PROMPT_SETS,
)
