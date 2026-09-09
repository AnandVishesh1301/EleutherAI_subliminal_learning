"""Fixed decisions of the 20260910 top-K cross-size GKD campaign.

Single source for every constant of
`.claude/plans/topk_cross_size_opd_59b62117.plan.md` and `docs/PREREG_TOPK.md`.
Sibling of `campaign.xsize_plan`: the two campaigns share driver machinery but
no constant. Do not import `xsize_*`.

Teachers are reused from the xsize run set (`TEACHER_RUN_SET_ID`); the only
teachers trained here are the `cat` null teachers for the EAS teacher-side
null. Students inherit the xsize recipe at ``lr*(410m) = 4e-4``; the only
per-rung difference is the loss truncation (`Rung`).

`SPEC` exists because `driver_io.CampaignDriverConfig.__post_init__`
overwrites ``run_set_id`` and ``queue_priorities`` from its ``spec``. Stage
A/B/C fields have no counterpart here and are filled with frozen values so the
dataclass is constructible.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

from subliminal_learning.campaign.selection import Recipe
from subliminal_learning.campaign.spec import CampaignSpec
from subliminal_learning.config import LoRAConfig
from subliminal_learning.gkd.layout import gkd_tag


# ---- campaign identity -------------------------------------------------------

RUN_SET_ID = "20260910_topk-bear_mixed"
TEACHER_RUN_SET_ID = "20260907_xsize-bear_mixed"
MODEL_SIZE = "mixed"
HUB_ORG = "EleutherAI"
OWNER = "vishesh"
TEACHER_WORKFLOW = "sdf"
TEACHER_MODE = "lora"
PROMPT_POOL_DIR = "gkd_prompts/pile_prompts_v1"


# ---- bases -------------------------------------------------------------------

SIZES: tuple[str, ...] = ("70m", "160m", "410m")
STUDENT_SIZES: tuple[str, ...] = ("160m", "410m")
BASE_STEP = 143000
BASE_REVISION = "step143000"
BASE_TOKEN: MappingProxyType[str, str] = MappingProxyType(
    {s: f"pythia-{s}@step143000" for s in SIZES}
)

ANIMALS: tuple[str, ...] = ("bear",)
NULL_ANIMAL = "cat"
TREATMENT_FTSEEDS: tuple[int, ...] = (0, 1, 2)
ANCHOR_FTSEEDS: tuple[int, ...] = (0,)
TEACHER_FTSEEDS: tuple[int, ...] = (0, 1, 2)
GSEEDS: tuple[int, ...] = (0, 1, 2)


# ---- teachers (reused) -------------------------------------------------------

TEACHER_PARAM_TAG = "ep5_lr5e-5_r128a128"
TEACHER_PARAM_TAG_LORA = TEACHER_PARAM_TAG
TEACHER_PARAM_TAG_FULL = "ep3_lr2e-5_full"


@dataclass(frozen=True)
class TeacherRecipe:
    """Frozen SDF teacher hyperparameter tag (null teachers only)."""

    tag: str
    epochs: int
    lr: float
    lora_r: int = 128
    lora_alpha: int = 128


TEACHER_RECIPE = TeacherRecipe(TEACHER_PARAM_TAG, 5, 5.0e-5)


# ---- students ----------------------------------------------------------------

STUDENT_MODE = "lora-r8a16"
STUDENT_LORA = LoRAConfig(
    r=8, lora_alpha=16, lora_dropout=0.0, target_modules="all-linear"
)
BETA = 1.0
LMBDA = 1.0
NUM_EPOCHS = 1.0
STUDENT_LR = 4.0e-4
STUDENT_RECIPE = Recipe(
    lora_r=STUDENT_LORA.r, lora_alpha=STUDENT_LORA.lora_alpha, lr=STUDENT_LR
)


@dataclass(frozen=True)
class Rung:
    """One point of the K ladder: ``k=None`` is the full vocabulary."""

    k: int | None
    ghost: bool = False

    @property
    def slug(self) -> str:
        """Path/queue token: ``full``, ``k100``, ``k5g``."""
        if self.k is None:
            return "full"
        return f"k{self.k}{'g' if self.ghost else ''}"

    @property
    def gkd_tag(self) -> str:
        """Student GKD directory tag at this rung."""
        return student_gkd_tag(self.k, ghost=self.ghost)

    @property
    def k_plot(self) -> int:
        """Numeric K for plots (`VOCAB_SIZE` for the full rung)."""
        return VOCAB_SIZE if self.k is None else int(self.k)


def student_gkd_tag(k: int | None, *, ghost: bool = False) -> str:
    """GKD directory tag (λ=1, β=1, LoRA r8a16, lr 4e-4, 1 epoch, top-K suffix)."""
    if ghost and k is None:
        raise ValueError("ghost token requires a finite top_k")
    return gkd_tag(
        lmbda=LMBDA,
        beta=BETA,
        student_mode="lora",
        learning_rate=STUDENT_LR,
        num_epochs=NUM_EPOCHS,
        lora=STUDENT_LORA,
        top_k=k,
        top_k_residual=ghost,
    )


VOCAB_SIZE = 50304
RUNG_FULL = Rung(None)
K_CORE: tuple[int, ...] = (100, 10, 5, 1000)
K_TAIL: tuple[int, ...] = (30, 2)
GHOST_K: tuple[int, ...] = (5,)
TAIL_CROSS_PAIRS: tuple[tuple[tuple[str, str], tuple[int, ...]], ...] = (
    (("70m", "410m"), (100, 5)),
)
K_LADDER_PLOT: tuple[int, ...] = (VOCAB_SIZE, 1000, 100, 30, 10, 5, 2)
FULL_GKD_TAG = RUNG_FULL.gkd_tag

HEADLINE_PAIR: tuple[str, str] = ("160m", "410m")
ANCHORS: tuple[tuple[str, str], ...] = (("410m", "410m"), ("160m", "160m"))


# ---- blocks (submission units) -------------------------------------------


@dataclass(frozen=True)
class Block:
    """One submitted unit: a rung set × pair set, per student size.

    Attributes:
        name: Queue/YAML slug.
        rungs: Ladder rungs trained in this block.
        pairs: Ordered ``(teacher, student)`` pairs.
        treatment_ftseeds: Teacher ftseeds for cross pairs.
        tail: Excluded from the completion criterion.
        priority: jobq priority of the train queues.
    """

    name: str
    rungs: tuple[Rung, ...]
    pairs: tuple[tuple[str, str], ...]
    treatment_ftseeds: tuple[int, ...]
    tail: bool
    priority: int

    @property
    def sizes(self) -> tuple[str, ...]:
        """Student sizes this block trains, in `STUDENT_SIZES` order."""
        present = {s for _, s in self.pairs}
        return tuple(s for s in STUDENT_SIZES if s in present)


_CORE_PRIORITY: MappingProxyType[int, int] = MappingProxyType(
    {100: 90, 10: 89, 5: 88, 1000: 87}
)
_CORE_PAIRS = (HEADLINE_PAIR, *ANCHORS)

BLOCKS: tuple[Block, ...] = (
    *(
        Block(
            name=f"k{k}",
            rungs=(Rung(k),),
            pairs=_CORE_PAIRS,
            treatment_ftseeds=TREATMENT_FTSEEDS,
            tail=False,
            priority=_CORE_PRIORITY[k],
        )
        for k in K_CORE
    ),
    Block(
        name="full",
        rungs=(RUNG_FULL,),
        pairs=(HEADLINE_PAIR,),
        treatment_ftseeds=ANCHOR_FTSEEDS,
        tail=False,
        priority=86,
    ),
    Block(
        name="k30",
        rungs=(Rung(30),),
        pairs=_CORE_PAIRS,
        treatment_ftseeds=TREATMENT_FTSEEDS,
        tail=True,
        priority=60,
    ),
    Block(
        name="x70m",
        rungs=tuple(Rung(k) for k in TAIL_CROSS_PAIRS[0][1]),
        pairs=(TAIL_CROSS_PAIRS[0][0],),
        treatment_ftseeds=ANCHOR_FTSEEDS,
        tail=True,
        priority=58,
    ),
    Block(
        name="k5g",
        rungs=tuple(Rung(k, ghost=True) for k in GHOST_K),
        pairs=(HEADLINE_PAIR, ANCHORS[0]),
        treatment_ftseeds=ANCHOR_FTSEEDS,
        tail=True,
        priority=56,
    ),
    Block(
        name="k2",
        rungs=(Rung(2),),
        pairs=_CORE_PAIRS,
        treatment_ftseeds=TREATMENT_FTSEEDS,
        tail=True,
        priority=54,
    ),
)
BLOCK_BY_NAME: MappingProxyType[str, Block] = MappingProxyType(
    {b.name: b for b in BLOCKS}
)


def block(name: str) -> Block:
    """The block called `name`."""
    try:
        return BLOCK_BY_NAME[name]
    except KeyError:
        raise ValueError(
            f"unknown block {name!r}; expected one of {list(BLOCK_BY_NAME)}"
        ) from None


def core_blocks() -> tuple[Block, ...]:
    """Blocks the completion criterion needs."""
    return tuple(b for b in BLOCKS if not b.tail)


def tail_blocks() -> tuple[Block, ...]:
    """Optional tail blocks in priority order."""
    return tuple(b for b in BLOCKS if b.tail)


# ---- thresholds + claims -----------------------------------------------------

QUARANTINE_PPL_RATIO_MAX = 5.0
QUARANTINE_FILTER_DROP_MAX = 0.5
EVAL_PROMPT_SETS: tuple[str, ...] = ("test", "favorite", "off_topic")
PRIMARY_PROMPT_SET = "test"
PRIMARY_SCORING = "prob"
BOOTSTRAP_N = 10000
BOOTSTRAP_SEED = 20260910
CLAIM_IDS: tuple[str, ...] = ("T0", "T1", "T2", "T3", "T4", "M1", "M2", "M3")
T0_MIN_DELTA = 0.02
T3_VISIBILITY_MAX = 0.01
T4_RETAINED_FRACTION_MIN = 0.5
EAS_PRIMARY_CONTEXT = "pile/all"
EAS_CONTEXTS: tuple[str, ...] = ("pile/all", "pile/last", "favorite/last")
EAS_N_PROMPTS = 1024

XSIZE_REFERENCE: MappingProxyType[str, object] = MappingProxyType(
    {
        "run_set_id": TEACHER_RUN_SET_ID,
        "cross_delta_tb": 0.370,
        "anchor_410m_delta_sb": 0.373,
        "anchor_160m_delta_sb": 0.356,
    }
)
XSIZE_LR = STUDENT_LR

SIGNATURE_N_ROWS = 2048
SIGNATURE_PAIR = HEADLINE_PAIR


# ---- footprints --------------------------------------------------------------


@dataclass(frozen=True)
class Footprint:
    """Per-size job packing. Frozen; est_minutes are measured xsize medians."""

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
        "160m": Footprint(1, 17_000, 32, 1, False, False, 0, 28),
        "410m": Footprint(2, 40_000, 16, 2, False, True, 125, 68),
    }
)
TEACHER_FOOTPRINT: MappingProxyType[str, Footprint] = MappingProxyType(
    {
        "160m": Footprint(1, 17_000, 8, 1, False, False, 0, 5),
        "410m": Footprint(2, 30_000, 4, 2, True, False, 0, 16),
    }
)
EAS_FOOTPRINT = Footprint(1, 12_000, 64, 1, False, False, 0, 3)
STUDENT_EFFECTIVE_BATCH = 32
TEACHER_EFFECTIVE_BATCH = 8
SAVE_TOTAL_LIMIT = 1
GPU_MEM_MIB = 48_000
EVAL_EST_MINUTES = 2


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
        if fp.slots not in (1, 2) or fp.slots > cap:
            raise ValueError(f"student {size}: slots {fp.slots} invalid for cap {cap}")
        if fp.mem_mib > GPU_MEM_MIB:
            raise ValueError(f"student {size}: mem_mib {fp.mem_mib} > {GPU_MEM_MIB}")
        if fp.checkpoint_resume and fp.save_steps <= 0:
            raise ValueError(f"student {size}: checkpoint_resume needs save_steps")
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
        if fp.slots not in (1, 2) or fp.slots > cap:
            raise ValueError(f"teacher {size}: slots {fp.slots} invalid for cap {cap}")
        if fp.mem_mib > GPU_MEM_MIB:
            raise ValueError(f"teacher {size}: mem_mib {fp.mem_mib} > {GPU_MEM_MIB}")
        if fp.checkpoint_resume:
            raise ValueError(f"teacher {size}: checkpoint_resume must be off")
    if EAS_FOOTPRINT.slots != 1 or EAS_FOOTPRINT.mem_mib > GPU_MEM_MIB:
        raise ValueError("EAS footprint must be one slot under the card")


# ---- jobq queues -------------------------------------------------------------

Q_NULL_TEACHERS = "topk-null-teachers"
Q_NULL_TEACHER_EVALS = "topk-null-teacher-evals"
_STUDENT_QUEUE_CLASSES: tuple[str, ...] = ("students", "controls", "evals", "eas")
_SIZE_PRIORITY_DROP: MappingProxyType[str, int] = MappingProxyType(
    {"410m": 0, "160m": 10}
)
_CORE_EVAL_PRIORITY = 70
_CORE_EAS_PRIORITY = 65
_TAIL_EVAL_PRIORITY = 50
_TAIL_EAS_PRIORITY = 45


def _require_size(size: str) -> str:
    if size not in STUDENT_SIZES:
        raise ValueError(f"unknown student size {size!r}; expected {STUDENT_SIZES}")
    return size


def q_students(block_name: str, size: str) -> str:
    """Treatment train queue of one block and student size."""
    block(block_name)
    return f"topk-{block_name}-students-{_require_size(size)}"


def q_controls(block_name: str, size: str) -> str:
    """Control train queue of one block and student size."""
    block(block_name)
    return f"topk-{block_name}-controls-{_require_size(size)}"


def q_evals(block_name: str, size: str) -> str:
    """Verify queue of one block and student size."""
    block(block_name)
    return f"topk-{block_name}-evals-{_require_size(size)}"


def q_eas(block_name: str, size: str) -> str:
    """GKD-EAS queue of one block and student size."""
    block(block_name)
    return f"topk-{block_name}-eas-{_require_size(size)}"


def block_queue_names(block_name: str) -> tuple[str, ...]:
    """Every queue one block submits, students first."""
    b = block(block_name)
    names: list[str] = []
    for cls in _STUDENT_QUEUE_CLASSES:
        for size in b.sizes:
            names.append(f"topk-{b.name}-{cls}-{size}")
    return tuple(names)


def tail_queue_names() -> frozenset[str]:
    """Queues of tail blocks: never part of the completion criterion."""
    names: set[str] = set()
    for b in tail_blocks():
        names.update(block_queue_names(b.name))
    return frozenset(names)


def _queue_priorities() -> MappingProxyType[str, int]:
    table: dict[str, int] = {
        Q_NULL_TEACHERS: 95,
        Q_NULL_TEACHER_EVALS: 94,
    }
    for b in BLOCKS:
        eval_prio = _TAIL_EVAL_PRIORITY if b.tail else _CORE_EVAL_PRIORITY
        eas_prio = _TAIL_EAS_PRIORITY if b.tail else _CORE_EAS_PRIORITY
        for size in b.sizes:
            drop = 0 if b.tail else _SIZE_PRIORITY_DROP[size]
            table[q_students(b.name, size)] = b.priority - drop
            table[q_controls(b.name, size)] = b.priority - drop
            table[q_evals(b.name, size)] = eval_prio
            table[q_eas(b.name, size)] = eas_prio
    return MappingProxyType(table)


QUEUE_PRIORITIES: MappingProxyType[str, int] = _queue_priorities()


# ---- cluster policy + deadline ----------------------------------------------

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
DEADLINE_HOURS = 19.5


def gpu_policy_json() -> dict:
    """A plain, JSON-serialisable copy of `GPU_POLICY`."""
    policy = dict(GPU_POLICY)
    policy["gpus"] = list(GPU_POLICY["gpus"])  # type: ignore[arg-type]
    policy["env"] = dict(GPU_POLICY["env"])  # type: ignore[arg-type]
    return policy


def total_slots() -> int:
    """Concurrent jobq slots across the node."""
    return len(GPU_POLICY["gpus"]) * int(GPU_POLICY["cap_per_gpu"])  # type: ignore[arg-type]


validate_footprints()


# ---- W&B ---------------------------------------------------------------------

WANDB_PROJECT = "tok-k_cross_size_opd"
WANDB_ANCHOR_RUNGS: tuple[Rung, ...] = (Rung(5), Rung(100))
WANDB_TAIL_CROSS_RUNGS: tuple[Rung, ...] = (Rung(5),)


def student_logs_wandb(
    *,
    teacher: str,
    student: str,
    rung: Rung,
    ftseed: int,
    gseed: int,
) -> bool:
    """Whether this student train cell creates a W&B run (11 cells total).

    Headline ``160m→410m`` tfs0 gseed0 at every rung; the 410m anchor at
    `WANDB_ANCHOR_RUNGS`; ``70m→410m`` at `WANDB_TAIL_CROSS_RUNGS`.
    """
    if ftseed != TREATMENT_FTSEEDS[0] or gseed != GSEEDS[0]:
        return False
    pair = (teacher, student)
    if pair == HEADLINE_PAIR:
        return True
    if pair == ANCHORS[0]:
        return rung in WANDB_ANCHOR_RUNGS
    if pair == TAIL_CROSS_PAIRS[0][0]:
        return rung in WANDB_TAIL_CROSS_RUNGS
    return False


# ---- static configs + artifacts ----------------------------------------------

GENERATED_CONFIG_DIR = "configs/experiment/generated/topk"
TEACHER_TRAINING_CONFIG = "configs/training/lora_sdf_teacher.yaml"
STUDENT_TRAINING_CONFIG = "configs/training/gkd_student_lora.yaml"
STUDENT_EXPERIMENT_CONFIG = "configs/experiment/topk_student.yaml"
TEACHER_EXPERIMENT_CONFIG = "configs/experiment/topk_teacher.yaml"
PREREG_PATH = "docs/PREREG_TOPK.md"

CAMPAIGN_STATE_FILE = "campaign_state.json"
CAMPAIGN_DONE_FILE = "CAMPAIGN_DONE"
CONTROL_LOG_FILE = "campaign_control_log.jsonl"
PREFLIGHT_REPORT = "preflight_report.json"
STAGING_REPORT = "staging_report.json"
PREFETCH_MANIFEST = "prefetch_manifest.json"
SIGNATURE_REPORT = "topk_signature.json"
EAS_LISTS_DIR = "eas_lists"
EAS_CACHE_DIR = "eas_cache"

RESULTS_LEAVES_CSV = "topk_leaves.csv"
RESULTS_DELTA_CSV = "topk_delta.csv"
RESULTS_CELLS_CSV = "topk_cells.csv"
RESULTS_CURVE_CSV = "topk_curve.csv"
RESULTS_RHO_CSV = "topk_rho.csv"
RESULTS_VISIBILITY_CSV = "topk_visibility.csv"
RESULTS_SIGNATURE_CSV = "topk_signature.csv"
RESULTS_EAS_CSV = "topk_eas.csv"
RESULTS_EAS_CELLS_CSV = "topk_eas_cells.csv"
RESULTS_PPL_CSV = "topk_ppl.csv"
RESULTS_QUARANTINE_CSV = "topk_quarantine.csv"
RESULTS_CONCORDANCE_CSV = "xsize_concordance.csv"
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
    # --- Stage A/B/C fields: not applicable; frozen values keep it constructible.
    a1_student_modes=(STUDENT_MODE,),
    a1_lora_lrs=(STUDENT_LR,),
    a1_lr_extension_factor=1.0,
    a1_gseeds=GSEEDS,
    stage_a_tfs=TREATMENT_FTSEEDS,
    a1_same_pair=("410m", "410m"),
    a1_cross_pair=("160m", "410m"),
    a2_student_mode=STUDENT_MODE,
    control_floor_max=0.02,
    control_floor_max_spread=0.01,
    a3_doses=(NUM_EPOCHS,),
    a3_top_k=1,
    a3_gseeds=GSEEDS,
    a3_max_same_arms=1,
    a3_reference_recipe=STUDENT_RECIPE,
    b_tfs_headline=TREATMENT_FTSEEDS,
    b_tfs_extension=(TREATMENT_FTSEEDS[0],),
    b_gseeds=GSEEDS,
    c_lambdas=(LMBDA,),
    c_gseeds=GSEEDS,
    q_a1=Q_NULL_TEACHERS,
    q_a2=Q_NULL_TEACHER_EVALS,
    q_a3=q_students("k100", "410m"),
    q_b_headline=q_students("k100", "410m"),
    q_b_ext=q_students("k5", "410m"),
    q_c=q_controls("k100", "410m"),
    q_a1_evals=Q_NULL_TEACHER_EVALS,
    q_a3_evals=q_evals("k100", "410m"),
    q_b_evals=q_evals("k100", "410m"),
    q_c_evals=q_evals("k100", "160m"),
    q_backlog=q_controls("k100", "160m"),
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
    a1_selection_report=SIGNATURE_REPORT,
    freeze_report=STAGING_REPORT,
    init_equality_report=PREFETCH_MANIFEST,
    campaign_state_file=CAMPAIGN_STATE_FILE,
    campaign_done_file=CAMPAIGN_DONE_FILE,
    control_log_file=CONTROL_LOG_FILE,
    eval_prompt_sets=EVAL_PROMPT_SETS,
    extra_pin_paths=("scripts/compute_gkd_eas.py", "pipelines/gkd_eas_pipeline.py"),
)
