"""Fixed decisions of the 20260912 cross-family OPD limits campaign (`xfam`).

Single source for every constant of
`.claude/plans/cross_family_opd_limits_xfam.plan.md` and `docs/PREREG_XFAM.md`.
Sibling of `campaign.topk_plan` / `campaign.xsize_plan`: the campaigns share driver
machinery but no constant. Do not import the other plan modules.

The roster (which models, which tokenizer cluster, which LoRA targets) lives in
`campaign.xfam_models`; this module owns the experiment design on top of it -- the
arms, the LR ladders and the recipes they sweep, the gates, the packing footprints
and the queue graph.

Mamba is deferred, not dropped by design. `state-spaces/mamba-370m-hf` stays in the
roster (its tokenizer and shape are verified) but has no arm here: without the
`mamba-ssm` / `causal-conv1d` CUDA kernels transformers falls back to a sequential
scan, which the staging smoke measured at 307 s/step against the Pythia anchor's 6.6
-- 47x, far past the 3x cost gate, and 53 h for a single 625-step cell. It returns
once the kernels build on this node.

`SPEC` exists because `driver_io.CampaignDriverConfig.__post_init__` overwrites
``run_set_id`` and ``queue_priorities`` from its ``spec``. Stage A/B/C fields have no
counterpart here and are filled with frozen values so the dataclass is constructible.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

from subliminal_learning.campaign import xfam_models as models
from subliminal_learning.campaign.selection import Recipe
from subliminal_learning.campaign.spec import CampaignSpec
from subliminal_learning.config import LoRAConfig
from subliminal_learning.gkd.layout import gkd_tag

# ---- campaign identity -------------------------------------------------------

RUN_SET_ID = "20260912_xfam-bear_mixed"
TEACHER_RUN_SET_ID = RUN_SET_ID
MODEL_SIZE = "mixed"
HUB_ORG = "EleutherAI"
OWNER = "vishesh"
TEACHER_WORKFLOW = "sdf"
TEACHER_MODE = "lora"

ANIMALS: tuple[str, ...] = ("bear",)
TARGET_ANIMAL = "bear"
TEACHER_FTSEEDS: tuple[int, ...] = (0, 1, 2)
LADDER_FTSEEDS: tuple[int, ...] = (0,)
REPLICATION_FTSEEDS: tuple[int, ...] = (1, 2)
GSEEDS: tuple[int, ...] = (0, 1, 2)


# ---- teachers ----------------------------------------------------------------


@dataclass(frozen=True)
class TeacherRecipe:
    """Frozen SDF teacher hyperparameters and their path tag."""

    tag: str
    epochs: int
    lr: float
    lora_r: int = 128
    lora_alpha: int = 128


TEACHER_PARAM_TAG = "ep5_lr5e-5_r128a128"
TEACHER_PARAM_TAG_LORA = TEACHER_PARAM_TAG
TEACHER_PARAM_TAG_FULL = "ep3_lr2e-5_full"
TEACHER_DOSE_TAG = "ep10_lr5e-5_r128a128"
TEACHER_RECIPES: MappingProxyType[str, TeacherRecipe] = MappingProxyType(
    {
        TEACHER_PARAM_TAG: TeacherRecipe(TEACHER_PARAM_TAG, 5, 5.0e-5),
        TEACHER_DOSE_TAG: TeacherRecipe(TEACHER_DOSE_TAG, 10, 5.0e-5),
    }
)
TEACHER_RECIPE = TEACHER_RECIPES[TEACHER_PARAM_TAG]

# Teacher cells: the cluster emitter at three ftseeds, each reverse emitter at tfs0.
CLUSTER_TEACHER_FTSEEDS: MappingProxyType[str, tuple[int, ...]] = MappingProxyType(
    {label: TEACHER_FTSEEDS for label in models.TEACHER_OF.values()}
)
REVERSE_TEACHER_FTSEEDS: tuple[int, ...] = (0,)


# ---- students ----------------------------------------------------------------

BETA = 1.0
LMBDA = 1.0
STUDENT_MODE = "lora"
BASE_RECIPE = Recipe(lora_r=8, lora_alpha=16, lr=0.0)
RANK_RECIPES: tuple[Recipe, ...] = (
    Recipe(lora_r=32, lora_alpha=64, lr=0.0),
    Recipe(lora_r=64, lora_alpha=32, lr=0.0),
)
BASE_EPOCHS = 1.0
EPOCH_ARM_EPOCHS = 2.0
TAIL_EPOCH_ARM_EPOCHS = 3.0

# Student LR is the one hyperparameter the xsize campaign showed does not transport
# across models (410m flat, 160m peaked, 70m unusable), so every model that is a new
# pretraining run gets its own ladder; post-trained checkpoints inherit their lineage's
# pick. OLMo's ladder is shifted one rung down for its 1.2B parameter count.
LR_LADDERS: MappingProxyType[str, tuple[float, ...]] = MappingProxyType(
    {
        "pythia-410m": (2.0e-4, 4.0e-4, 8.0e-4),
        "pythia-410m-deduped": (2.0e-4, 4.0e-4, 8.0e-4),
        "amd-olmo-1b": (1.0e-4, 2.0e-4, 4.0e-4),
        "qwen2.5-0.5b": (2.0e-4, 4.0e-4, 8.0e-4),
        "qwen3-0.6b-base": (2.0e-4, 4.0e-4, 8.0e-4),
    }
)
LR_DEFAULTS: MappingProxyType[str, float] = MappingProxyType(
    {
        "pythia-410m": 4.0e-4,
        "pythia-410m-deduped": 4.0e-4,
        "amd-olmo-1b": 2.0e-4,
        "qwen2.5-0.5b": 4.0e-4,
        "qwen3-0.6b-base": 4.0e-4,
    }
)
LADDER_STUDENTS: tuple[str, ...] = tuple(LR_LADDERS)


def lora_of(recipe: Recipe) -> LoRAConfig:
    """Student LoRA config of a recipe."""
    return LoRAConfig(
        r=recipe.lora_r,
        lora_alpha=recipe.lora_alpha,
        lora_dropout=0.0,
        target_modules="all-linear",
    )


def student_gkd_tag(lr: float, recipe: Recipe, epochs: float) -> str:
    """GKD directory tag of one student cell (λ=1, β=1, LoRA)."""
    return gkd_tag(
        lmbda=LMBDA,
        beta=BETA,
        student_mode=STUDENT_MODE,
        learning_rate=lr,
        num_epochs=epochs,
        lora=lora_of(recipe),
    )


# ---- arms --------------------------------------------------------------------

KIND_ANCHOR = "anchor"
KIND_FORWARD = "forward"
KIND_REVERSE = "reverse"
KIND_RANK = "rank"
KIND_EPOCH = "epoch"
KIND_REPLICATION = "replication"
KIND_DOSE = "dose"

# `Arm.lr_source` sentinels.
LADDER = "@ladder"  # sweep this student's own LR ladder (P2)
SELF = "@self"  # use this student's own picked LR


@dataclass(frozen=True)
class Arm:
    """One submission unit: a teacher, its students, and the HP variants they sweep.

    Attributes:
        name: Queue/YAML slug, unique across the campaign.
        cluster: Tokenizer cluster; every model of the arm belongs to it.
        teacher: Emitter label.
        students: Receiver labels.
        kind: One of the ``KIND_*`` constants (descriptive; drives the results tables).
        lr_source: ``LADDER`` (sweep this student's own ladder), ``SELF`` (use the
            student's own picked LR), or the label whose picked LR the arm inherits.
        recipes: LoRA recipes swept at that LR.
        epochs: Training budget in passes over the prompt pool.
        ftseeds: Teacher ftseeds.
        teacher_param_tag: SDF tag of the teacher checkpoint the arm distils from.
        replication_gated: Submit only when the student cleared `replication_gate`.
        emits_controls: Whether this arm submits its own control cells. False for an
            arm whose cells share a GKD tag with a ladder arm's at the picked LR (extra
            teacher seeds, or a different teacher dose): a control distils from an
            UNTUNED base, so it depends only on (student, tag, gseed) and the ladder arm
            already trained it. Re-emitting would make two queues race for one cell.
        tail: Excluded from the completion criterion.
        priority: jobq priority of the arm's train queues.
    """

    name: str
    cluster: str
    teacher: str
    students: tuple[str, ...]
    kind: str
    lr_source: str
    recipes: tuple[Recipe, ...]
    epochs: float
    ftseeds: tuple[int, ...]
    teacher_param_tag: str
    replication_gated: bool
    tail: bool
    priority: int
    emits_controls: bool = True

    @property
    def is_ladder(self) -> bool:
        """Whether the arm sweeps its students' own LR ladders."""
        return self.lr_source == LADDER

    def lr_source_of(self, student: str) -> str:
        """Label whose persisted LR pick gates `student`'s cells in this arm."""
        if self.is_ladder:
            return student
        return student if self.lr_source == SELF else self.lr_source

    def lrs_for(self, student: str, picks: dict[str, float]) -> tuple[float, ...]:
        """Learning rates this arm trains `student` at.

        Args:
            student: Receiver label, which must belong to the arm.
            picks: Persisted ``label -> lr*`` decisions.

        Raises:
            KeyError: If a non-ladder arm's `lr_source` has not been picked yet.
        """
        if student not in self.students:
            raise ValueError(f"{student!r} is not a student of arm {self.name!r}")
        if self.is_ladder:
            return LR_LADDERS[student]
        return (picks[self.lr_source_of(student)],)


_A = ("neox", "pythia-410m")
_B = ("qwen", "qwen2.5-0.5b")

ARMS: tuple[Arm, ...] = (
    # --- cluster A: GPT-NeoX tokenizer, Pythia-410M emitter ---------------------
    Arm(
        name="a1-anchor",
        cluster=_A[0],
        teacher=_A[1],
        students=("pythia-410m",),
        kind=KIND_ANCHOR,
        lr_source=LADDER,
        recipes=(BASE_RECIPE,),
        epochs=BASE_EPOCHS,
        ftseeds=LADDER_FTSEEDS,
        teacher_param_tag=TEACHER_PARAM_TAG,
        replication_gated=False,
        tail=False,
        priority=90,
    ),
    Arm(
        name="a2-deduped",
        cluster=_A[0],
        teacher=_A[1],
        students=("pythia-410m-deduped",),
        kind=KIND_FORWARD,
        lr_source=LADDER,
        recipes=(BASE_RECIPE,),
        epochs=BASE_EPOCHS,
        ftseeds=LADDER_FTSEEDS,
        teacher_param_tag=TEACHER_PARAM_TAG,
        replication_gated=False,
        tail=False,
        priority=90,
    ),
    Arm(
        name="a4-olmo",
        cluster=_A[0],
        teacher=_A[1],
        students=("amd-olmo-1b",),
        kind=KIND_FORWARD,
        lr_source=LADDER,
        recipes=(BASE_RECIPE,),
        epochs=BASE_EPOCHS,
        ftseeds=LADDER_FTSEEDS,
        teacher_param_tag=TEACHER_PARAM_TAG,
        replication_gated=False,
        tail=False,
        priority=92,
    ),
    Arm(
        name="a5-posttrain",
        cluster=_A[0],
        teacher=_A[1],
        students=("pythia-410m-hh-sft", "pythia-410m-hh-dpo", "pythia-410m-ppo"),
        kind=KIND_FORWARD,
        lr_source="pythia-410m",
        recipes=(BASE_RECIPE,),
        epochs=BASE_EPOCHS,
        ftseeds=LADDER_FTSEEDS,
        teacher_param_tag=TEACHER_PARAM_TAG,
        replication_gated=False,
        tail=False,
        priority=82,
    ),
    Arm(
        name="a6-olmo-posttrain",
        cluster=_A[0],
        teacher=_A[1],
        students=("amd-olmo-1b-sft-dpo",),
        kind=KIND_FORWARD,
        lr_source="amd-olmo-1b",
        recipes=(BASE_RECIPE,),
        epochs=BASE_EPOCHS,
        ftseeds=LADDER_FTSEEDS,
        teacher_param_tag=TEACHER_PARAM_TAG,
        replication_gated=False,
        tail=False,
        priority=84,
    ),
    Arm(
        name="a7-reverse-olmo",
        cluster=_A[0],
        teacher="amd-olmo-1b",
        students=("pythia-410m",),
        kind=KIND_REVERSE,
        lr_source=SELF,
        recipes=(BASE_RECIPE,),
        epochs=BASE_EPOCHS,
        ftseeds=REVERSE_TEACHER_FTSEEDS,
        teacher_param_tag=TEACHER_PARAM_TAG,
        replication_gated=False,
        tail=False,
        priority=83,
    ),
    Arm(
        name="a8-rank-olmo",
        cluster=_A[0],
        teacher=_A[1],
        students=("amd-olmo-1b",),
        kind=KIND_RANK,
        lr_source=SELF,
        recipes=RANK_RECIPES,
        epochs=BASE_EPOCHS,
        ftseeds=LADDER_FTSEEDS,
        teacher_param_tag=TEACHER_PARAM_TAG,
        replication_gated=False,
        tail=False,
        priority=81,
    ),
    Arm(
        name="a10-replication",
        cluster=_A[0],
        teacher=_A[1],
        students=("pythia-410m-deduped", "amd-olmo-1b"),
        kind=KIND_REPLICATION,
        lr_source=SELF,
        recipes=(BASE_RECIPE,),
        epochs=BASE_EPOCHS,
        ftseeds=REPLICATION_FTSEEDS,
        teacher_param_tag=TEACHER_PARAM_TAG,
        replication_gated=True,
        tail=False,
        priority=79,
        emits_controls=False,
    ),
    # --- cluster B: Qwen tokenizer, Qwen2.5-0.5B emitter ------------------------
    Arm(
        name="b1-anchor",
        cluster=_B[0],
        teacher=_B[1],
        students=("qwen2.5-0.5b",),
        kind=KIND_ANCHOR,
        lr_source=LADDER,
        recipes=(BASE_RECIPE,),
        epochs=BASE_EPOCHS,
        ftseeds=LADDER_FTSEEDS,
        teacher_param_tag=TEACHER_PARAM_TAG,
        replication_gated=False,
        tail=False,
        priority=93,
    ),
    Arm(
        name="b2-qwen3-base",
        cluster=_B[0],
        teacher=_B[1],
        students=("qwen3-0.6b-base",),
        kind=KIND_FORWARD,
        lr_source=LADDER,
        recipes=(BASE_RECIPE,),
        epochs=BASE_EPOCHS,
        ftseeds=LADDER_FTSEEDS,
        teacher_param_tag=TEACHER_PARAM_TAG,
        replication_gated=False,
        tail=False,
        priority=93,
    ),
    Arm(
        name="b3-instruct",
        cluster=_B[0],
        teacher=_B[1],
        students=("qwen2.5-0.5b-instruct",),
        kind=KIND_FORWARD,
        lr_source="qwen2.5-0.5b",
        recipes=(BASE_RECIPE,),
        epochs=BASE_EPOCHS,
        ftseeds=LADDER_FTSEEDS,
        teacher_param_tag=TEACHER_PARAM_TAG,
        replication_gated=False,
        tail=False,
        priority=85,
    ),
    Arm(
        name="b4-qwen3-rl",
        cluster=_B[0],
        teacher=_B[1],
        students=("qwen3-0.6b",),
        kind=KIND_FORWARD,
        lr_source="qwen3-0.6b-base",
        recipes=(BASE_RECIPE,),
        epochs=BASE_EPOCHS,
        ftseeds=LADDER_FTSEEDS,
        teacher_param_tag=TEACHER_PARAM_TAG,
        replication_gated=False,
        tail=False,
        priority=85,
    ),
    Arm(
        name="b5-coder",
        cluster=_B[0],
        teacher=_B[1],
        students=("qwen2.5-coder-0.5b",),
        kind=KIND_FORWARD,
        lr_source="qwen2.5-0.5b",
        recipes=(BASE_RECIPE,),
        epochs=BASE_EPOCHS,
        ftseeds=LADDER_FTSEEDS,
        teacher_param_tag=TEACHER_PARAM_TAG,
        replication_gated=False,
        tail=False,
        priority=85,
    ),
    Arm(
        name="b6-reverse-qwen3",
        cluster=_B[0],
        teacher="qwen3-0.6b-base",
        students=("qwen2.5-0.5b",),
        kind=KIND_REVERSE,
        lr_source=SELF,
        recipes=(BASE_RECIPE,),
        epochs=BASE_EPOCHS,
        ftseeds=REVERSE_TEACHER_FTSEEDS,
        teacher_param_tag=TEACHER_PARAM_TAG,
        replication_gated=False,
        tail=False,
        priority=84,
    ),
    Arm(
        name="b7-rank-qwen3",
        cluster=_B[0],
        teacher=_B[1],
        students=("qwen3-0.6b-base",),
        kind=KIND_RANK,
        lr_source=SELF,
        recipes=RANK_RECIPES,
        epochs=BASE_EPOCHS,
        ftseeds=LADDER_FTSEEDS,
        teacher_param_tag=TEACHER_PARAM_TAG,
        replication_gated=False,
        tail=False,
        priority=81,
    ),
    Arm(
        name="b8-epoch-qwen3",
        cluster=_B[0],
        teacher=_B[1],
        students=("qwen3-0.6b-base",),
        kind=KIND_EPOCH,
        lr_source=SELF,
        recipes=(BASE_RECIPE,),
        epochs=EPOCH_ARM_EPOCHS,
        ftseeds=LADDER_FTSEEDS,
        teacher_param_tag=TEACHER_PARAM_TAG,
        replication_gated=False,
        tail=False,
        priority=80,
    ),
    Arm(
        name="b9-replication",
        cluster=_B[0],
        teacher=_B[1],
        students=("qwen3-0.6b-base",),
        kind=KIND_REPLICATION,
        lr_source=SELF,
        recipes=(BASE_RECIPE,),
        epochs=BASE_EPOCHS,
        ftseeds=REPLICATION_FTSEEDS,
        teacher_param_tag=TEACHER_PARAM_TAG,
        replication_gated=True,
        tail=False,
        priority=79,
        emits_controls=False,
    ),
    # --- tail: enqueued at launch, drains only after the core -------------------
    Arm(
        name="t2-olmo-sft",
        cluster=_A[0],
        teacher=_A[1],
        students=("amd-olmo-1b-sft",),
        kind=KIND_FORWARD,
        lr_source="amd-olmo-1b",
        recipes=(BASE_RECIPE,),
        epochs=BASE_EPOCHS,
        ftseeds=LADDER_FTSEEDS,
        teacher_param_tag=TEACHER_PARAM_TAG,
        replication_gated=False,
        tail=True,
        priority=58,
    ),
    Arm(
        name="t3-dose-neox",
        cluster=_A[0],
        teacher=_A[1],
        students=("amd-olmo-1b",),
        kind=KIND_DOSE,
        lr_source=SELF,
        recipes=(BASE_RECIPE,),
        epochs=BASE_EPOCHS,
        ftseeds=LADDER_FTSEEDS,
        teacher_param_tag=TEACHER_DOSE_TAG,
        replication_gated=False,
        tail=True,
        priority=56,
        emits_controls=False,
    ),
    Arm(
        name="t4-dose-qwen",
        cluster=_B[0],
        teacher=_B[1],
        students=("qwen3-0.6b-base",),
        kind=KIND_DOSE,
        lr_source=SELF,
        recipes=(BASE_RECIPE,),
        epochs=BASE_EPOCHS,
        ftseeds=LADDER_FTSEEDS,
        teacher_param_tag=TEACHER_DOSE_TAG,
        replication_gated=False,
        tail=True,
        priority=54,
        emits_controls=False,
    ),
    Arm(
        name="t5-rank-deduped",
        cluster=_A[0],
        teacher=_A[1],
        students=("pythia-410m-deduped",),
        kind=KIND_RANK,
        lr_source=SELF,
        recipes=RANK_RECIPES,
        epochs=BASE_EPOCHS,
        ftseeds=LADDER_FTSEEDS,
        teacher_param_tag=TEACHER_PARAM_TAG,
        replication_gated=False,
        tail=True,
        priority=52,
    ),
)

ARM_BY_NAME: MappingProxyType[str, Arm] = MappingProxyType({a.name: a for a in ARMS})

# Tail arms distilling from a dose teacher need that teacher trained; it is enqueued
# with the core teachers so the tail never waits on a training job of its own.
DOSE_TEACHERS: MappingProxyType[str, tuple[int, ...]] = MappingProxyType(
    {label: REVERSE_TEACHER_FTSEEDS for label in ("pythia-410m", "qwen2.5-0.5b")}
)


def arm(name: str) -> Arm:
    """The arm called `name`."""
    try:
        return ARM_BY_NAME[name]
    except KeyError:
        raise ValueError(f"unknown arm {name!r}; expected one of {list(ARM_BY_NAME)}") from None


def core_arms() -> tuple[Arm, ...]:
    """Arms the completion criterion needs."""
    return tuple(a for a in ARMS if not a.tail)


def tail_arms() -> tuple[Arm, ...]:
    """Overflow arms, in priority order."""
    return tuple(sorted((a for a in ARMS if a.tail), key=lambda a: -a.priority))


def ladder_arms() -> tuple[Arm, ...]:
    """Arms submitted in P2, before any LR is picked."""
    return tuple(a for a in ARMS if a.is_ladder)


def derived_arms() -> tuple[Arm, ...]:
    """Arms whose submission waits on a persisted LR pick (and maybe a gate)."""
    return tuple(a for a in ARMS if not a.is_ladder)


def all_units() -> tuple[tuple[str, str], ...]:
    """Every ``(arm, student)`` submission unit of the campaign."""
    return tuple((a.name, s) for a in ARMS for s in a.students)


def units_unlocked_by(label: str) -> tuple[tuple[str, str], ...]:
    """``(arm, student)`` units whose LR gate is satisfied by `label`'s pick.

    A unit that is also replication-gated is returned here; the driver checks
    `replication_gate` separately before submitting it.
    """
    return tuple(
        (a.name, student)
        for a in derived_arms()
        for student in a.students
        if a.lr_source_of(student) == label
    )


def teacher_labels() -> tuple[str, ...]:
    """Every label this campaign SDF fine-tunes, in submission order.

    A roster emitter with no arm is skipped: `models.REVERSE_TEACHERS` lists every
    label that COULD emit, and this campaign trains only the ones an arm distils from.
    """
    used = {a.teacher for a in ARMS}
    ordered: list[str] = []
    for label in (*models.TEACHER_OF.values(), *models.REVERSE_TEACHERS):
        if label in used and label not in ordered:
            ordered.append(label)
    return tuple(ordered)


def teacher_cells() -> tuple[tuple[str, str, int], ...]:
    """Every ``(label, param_tag, ftseed)`` teacher cell this campaign trains."""
    cells: list[tuple[str, str, int]] = []
    for label in teacher_labels():
        ftseeds = CLUSTER_TEACHER_FTSEEDS.get(label, REVERSE_TEACHER_FTSEEDS)
        cells.extend((label, TEACHER_PARAM_TAG, fs) for fs in ftseeds)
    for label, ftseeds in DOSE_TEACHERS.items():
        cells.extend((label, TEACHER_DOSE_TAG, fs) for fs in ftseeds)
    return tuple(cells)


# ---- gates + thresholds ------------------------------------------------------

ADMISSION_SD_MULTIPLE = 5.0
ADMISSION_ABS_MIN = 0.02
TEACHER_PPL_RATIO_MAX = 3.0
MIN_ADMITTED_PER_CLUSTER_TEACHER = 2

# Raised from the xsize value of 1.5: at 1.5 no rung passed for any student outside
# 410m, so every pick fell back to the default with a flag. 4.0 still excludes a
# collapsed student while letting a cross-family or post-trained student pick.
PICK_PPL_RATIO_MAX = 4.0
PICK_TIE_BAND = 0.02

QUARANTINE_PPL_RATIO_MAX = 5.0
QUARANTINE_FILTER_DROP_MAX = 0.5

REPLICATION_MIN_DELTA = 0.02

EVAL_PROMPT_SETS: tuple[str, ...] = ("test", "favorite", "off_topic")
PRIMARY_PROMPT_SET = "test"
PRIMARY_SCORING = "prob"
BOOTSTRAP_N = 10000
BOOTSTRAP_SEED = 20260912
CLAIM_IDS: tuple[str, ...] = ("F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7")
F0_MIN_DELTA = 0.02
F6_RETAINED_FRACTION_MIN = 0.5

# F2 rungs, nearest-to-farthest from the emitter's own weights. Every rung is a
# student of the cluster's forward arms; the ladder is read per cluster.
INIT_DISTANCE_RUNGS: MappingProxyType[str, tuple[str, ...]] = MappingProxyType(
    {
        "neox": (
            "pythia-410m",
            "pythia-410m-hh-sft",
            "pythia-410m-hh-dpo",
            "pythia-410m-ppo",
            "pythia-410m-deduped",
            "amd-olmo-1b",
        ),
        "qwen": (
            "qwen2.5-0.5b",
            "qwen2.5-0.5b-instruct",
            "qwen2.5-coder-0.5b",
            "qwen3-0.6b-base",
            "qwen3-0.6b",
        ),
    }
)

XSIZE_REFERENCE: MappingProxyType[str, object] = MappingProxyType(
    {
        "run_set_id": "20260907_xsize-bear_mixed",
        "cross_delta_tb_160m_to_410m": 0.370,
        "anchor_410m_delta_sb": 0.373,
        "di_ds_grid_reference": 0.265,
    }
)


# ---- footprints --------------------------------------------------------------


@dataclass(frozen=True)
class Footprint:
    """Per-label job packing. Frozen; stamped from the staging smoke."""

    slots: int
    mem_mib: int
    micro_batch: int
    grad_accum: int
    grad_checkpointing: bool
    checkpoint_resume: bool
    save_steps: int
    est_minutes: int


STUDENT_EFFECTIVE_BATCH = 32
TEACHER_EFFECTIVE_BATCH = 8
SAVE_TOTAL_LIMIT = 1
GPU_MEM_MIB = 48_000
EVAL_EST_MINUTES = 3

# Measured on the staging smoke (`logs/xfam_footprints.json`, --max-steps 5, one cell
# per card, nvidia-smi peak per process). Every cell is ONE slot; `mem_mib` is the
# admission gate (peak + margin) and the hard `cap_per_gpu` is the OOM guarantee.
# Mamba never gradient-checkpoints: TRL disables the generation cache under
# checkpointing, which makes its recurrent rollouts quadratic.
# NO student gradient-checkpoints. TRL 0.19.1 builds the rollout `GenerationConfig`
# with `use_cache=False if args.gradient_checkpointing else True`, so checkpointing
# makes every on-policy rollout uncached: the staging smoke measured 45-88 s/step
# checkpointed against 6.5 s/step without it, which is 7-13x the whole budget. The
# activation memory it would have saved is not the binding term anyway -- the two
# vocabulary-sized logit tensors are.
# Stamped from the staging smoke (5 steps, one cell per card, nvidia-smi peak), x625
# steps: pythia-410m 6.6 s/step / 12.5 GB, amd-olmo-1b 11.4 / 18.4, qwen2.5-0.5b 19.8
# / 13.4, qwen3-0.6b-base 24.4 / 19.4. `mem_mib` is peak + ~2.5 GB, which lets the
# reservation gate pack 3 Pythia or Qwen2.5 cells per 48 GB card and 2 OLMo or Qwen3.
# Micro-batch 8 beat 4 on throughput for every big label (measured: halving the batch
# cut peak memory ~35 % but raised s/step ~1.8x, a net loss even at one more cell per
# card). A Qwen cell costs 3-4x a Pythia one because its 151 936-wide LM head runs 128
# sequential generation steps.
_NEOX_410M = Footprint(1, 15_000, 16, 2, False, True, 125, 69)
_OLMO_1B = Footprint(1, 21_000, 8, 4, False, True, 125, 119)
_QWEN_05B = Footprint(1, 16_000, 8, 4, False, True, 125, 206)
_QWEN_06B = Footprint(1, 22_000, 8, 4, False, True, 125, 254)
STUDENT_FOOTPRINT: MappingProxyType[str, Footprint] = MappingProxyType(
    {
        "pythia-410m": _NEOX_410M,
        "pythia-410m-deduped": _NEOX_410M,
        "pythia-410m-hh-sft": _NEOX_410M,
        "pythia-410m-hh-dpo": _NEOX_410M,
        "pythia-410m-ppo": _NEOX_410M,
        "amd-olmo-1b": _OLMO_1B,
        "amd-olmo-1b-sft": _OLMO_1B,
        "amd-olmo-1b-sft-dpo": _OLMO_1B,
        "qwen2.5-0.5b": _QWEN_05B,
        "qwen2.5-0.5b-instruct": _QWEN_05B,
        "qwen2.5-coder-0.5b": _QWEN_05B,
        "qwen3-0.6b-base": _QWEN_06B,
        "qwen3-0.6b": _QWEN_06B,
    }
)
# Teachers do not generate, so checkpointing costs them nothing but memory savings.
TEACHER_FOOTPRINT: MappingProxyType[str, Footprint] = MappingProxyType(
    {
        "pythia-410m": Footprint(1, 10_000, 4, 2, True, False, 0, 16),
        "amd-olmo-1b": Footprint(1, 18_000, 2, 4, True, False, 0, 40),
        "qwen2.5-0.5b": Footprint(1, 16_000, 2, 4, True, False, 0, 30),
        "qwen3-0.6b-base": Footprint(1, 18_000, 2, 4, True, False, 0, 35),
    }
)

# Per-label LoRA targets where PEFT refuses the `all-linear` sentinel. Empty today:
# Mamba was the only such model and is deferred (see the module docstring).
STUDENT_TARGET_MODULES: MappingProxyType[str, tuple[str, ...]] = MappingProxyType({})
STUDENT_TARGET_DEFAULT = "all-linear"


def student_target_modules(label: str) -> str | list[str]:
    """LoRA target modules for one student, or the architecture-agnostic sentinel."""
    explicit = STUDENT_TARGET_MODULES.get(label)
    return list(explicit) if explicit else STUDENT_TARGET_DEFAULT


# Labels whose rollouts must never run uncached (all of them today).
NO_GRAD_CHECKPOINT_LABELS: frozenset[str] = frozenset(STUDENT_FOOTPRINT)


def student_footprint(label: str) -> Footprint:
    """Packing footprint of a student cell."""
    try:
        return STUDENT_FOOTPRINT[label]
    except KeyError:
        raise ValueError(f"no student footprint for {label!r}") from None


def teacher_footprint(label: str) -> Footprint:
    """Packing footprint of an SDF teacher cell."""
    try:
        return TEACHER_FOOTPRINT[label]
    except KeyError:
        raise ValueError(f"no teacher footprint for {label!r}") from None


def validate_footprints() -> None:
    """Raise if a frozen footprint violates the packing contract."""
    cap = int(GPU_POLICY["cap_per_gpu"])
    for label, fp in STUDENT_FOOTPRINT.items():
        if fp.micro_batch * fp.grad_accum != STUDENT_EFFECTIVE_BATCH:
            raise ValueError(
                f"student {label}: micro_batch * grad_accum "
                f"{fp.micro_batch * fp.grad_accum} != {STUDENT_EFFECTIVE_BATCH}"
            )
        if fp.slots != 1 or fp.slots > cap:
            raise ValueError(f"student {label}: slots {fp.slots} invalid for cap {cap}")
        if fp.mem_mib > GPU_MEM_MIB:
            raise ValueError(f"student {label}: mem_mib {fp.mem_mib} > {GPU_MEM_MIB}")
        if fp.checkpoint_resume and fp.save_steps <= 0:
            raise ValueError(f"student {label}: checkpoint_resume needs save_steps")
        if label in NO_GRAD_CHECKPOINT_LABELS and fp.grad_checkpointing:
            raise ValueError(f"student {label}: must not gradient-checkpoint")
    for label, fp in TEACHER_FOOTPRINT.items():
        if fp.micro_batch * fp.grad_accum != TEACHER_EFFECTIVE_BATCH:
            raise ValueError(
                f"teacher {label}: micro_batch * grad_accum "
                f"{fp.micro_batch * fp.grad_accum} != {TEACHER_EFFECTIVE_BATCH}"
            )
        if fp.slots != 1 or fp.mem_mib > GPU_MEM_MIB:
            raise ValueError(f"teacher {label}: footprint must be one slot under a card")
        if fp.checkpoint_resume:
            raise ValueError(f"teacher {label}: checkpoint_resume must be off")
    missing = {s for a in ARMS for s in a.students} - set(STUDENT_FOOTPRINT)
    if missing:
        raise ValueError(f"students without a footprint: {sorted(missing)}")
    missing_t = {label for label, _, _ in teacher_cells()} - set(TEACHER_FOOTPRINT)
    if missing_t:
        raise ValueError(f"teachers without a footprint: {sorted(missing_t)}")


# ---- jobq queues -------------------------------------------------------------

Q_TEACHERS = "xfam-teachers"
Q_TEACHER_EVALS = "xfam-teacher-evals"
_CORE_EVAL_PRIORITY = 70
_TAIL_EVAL_PRIORITY = 50


def q_students(arm_name: str, student: str) -> str:
    """Treatment train queue of one arm and student."""
    _require_student(arm_name, student)
    return f"xfam-{arm_name}-students-{student}"


def q_controls(arm_name: str, student: str) -> str:
    """Control train queue of one arm and student."""
    _require_student(arm_name, student)
    return f"xfam-{arm_name}-controls-{student}"


def q_evals(arm_name: str, student: str) -> str:
    """Verify queue of one arm and student."""
    _require_student(arm_name, student)
    return f"xfam-{arm_name}-evals-{student}"


def _require_student(arm_name: str, student: str) -> None:
    if student not in arm(arm_name).students:
        raise ValueError(f"{student!r} is not a student of arm {arm_name!r}")


def arm_queue_names(arm_name: str) -> tuple[str, ...]:
    """Every queue one arm submits, students first."""
    a = arm(arm_name)
    names: list[str] = []
    for maker in (q_students, q_controls, q_evals):
        names.extend(maker(a.name, s) for s in a.students)
    return tuple(names)


def tail_queue_names() -> frozenset[str]:
    """Queues of tail arms: never part of the completion criterion."""
    names: set[str] = set()
    for a in tail_arms():
        names.update(arm_queue_names(a.name))
    return frozenset(names)


def _queue_priorities() -> MappingProxyType[str, int]:
    table: dict[str, int] = {Q_TEACHERS: 100, Q_TEACHER_EVALS: 98}
    for a in ARMS:
        eval_prio = _TAIL_EVAL_PRIORITY if a.tail else _CORE_EVAL_PRIORITY
        for student in a.students:
            table[q_students(a.name, student)] = a.priority
            table[q_controls(a.name, student)] = a.priority - 1
            table[q_evals(a.name, student)] = eval_prio
    return MappingProxyType(table)


QUEUE_PRIORITIES: MappingProxyType[str, int] = _queue_priorities()


# ---- cluster policy + deadline ----------------------------------------------

GPU_POLICY: MappingProxyType[str, object] = MappingProxyType(
    {
        "gpus": (0, 1, 2, 3, 4, 5, 6, 7),
        "cap_per_gpu": 3,
        "free_mem_mib": 5000,
        "reservation_warmup_s": 900,
        "env": MappingProxyType(
            {
                "OMP_NUM_THREADS": "1",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }
        ),
    }
)
AUTO_REQUEUE_ATTEMPTS = 2
DEADLINE_HOURS = 46.0


def gpu_policy_json() -> dict:
    """A plain, JSON-serialisable copy of `GPU_POLICY`."""
    policy = dict(GPU_POLICY)
    policy["gpus"] = list(GPU_POLICY["gpus"])  # type: ignore[arg-type]
    policy["env"] = dict(GPU_POLICY["env"])  # type: ignore[arg-type]
    return policy


def total_slots() -> int:
    """Concurrent jobq slots across the node."""
    return len(GPU_POLICY["gpus"]) * int(GPU_POLICY["cap_per_gpu"])  # type: ignore[arg-type]


def cells_per_card(label: str) -> int:
    """How many cells of `label` a card admits: the memory gate, capped by policy."""
    fp = student_footprint(label)
    return max(1, min(int(GPU_POLICY["cap_per_gpu"]), GPU_MEM_MIB // fp.mem_mib))  # type: ignore[arg-type]


def node_concurrency(label: str) -> int:
    """Concurrent cells of `label` across the node."""
    return len(GPU_POLICY["gpus"]) * cells_per_card(label)  # type: ignore[arg-type]


validate_footprints()


# ---- W&B ---------------------------------------------------------------------

WANDB_PROJECT = "cross-model-opd-limit-testing"
WANDB_MAX_RUNS = 60


def student_logs_wandb(
    *,
    arm_name: str,
    teacher: str,
    student: str,
    lr: float,
    recipe: Recipe,
    epochs: float,
    ftseed: int,
    gseed: int,
) -> bool:
    """Whether this student train cell creates a W&B run.

    One run per (arm, student, LR, recipe, epochs) at the first ftseed and gseed, plus
    the second gseed on the two headline cross-family cells. Controls never log.
    """
    if gseed == GSEEDS[0] and ftseed == LADDER_FTSEEDS[0]:
        return True
    headline = {("a4-olmo", "amd-olmo-1b"), ("b2-qwen3-base", "qwen3-0.6b-base")}
    return (
        (arm_name, student) in headline
        and ftseed == LADDER_FTSEEDS[0]
        and gseed == GSEEDS[1]
        and recipe == BASE_RECIPE
        and epochs == BASE_EPOCHS
    )


def wandb_run_name(
    *,
    arm_name: str,
    teacher: str,
    student: str,
    gkd_tag_str: str,
    ftseed: int,
    gseed: int,
) -> str:
    """W&B run name of one logged student cell."""
    return f"xfam_{arm_name}_{teacher}-to-{student}_{gkd_tag_str}_tfs{ftseed}_g{gseed}"


def wandb_tags(
    *, arm_name: str, teacher: str, student: str, lr: float, recipe: Recipe, epochs: float
) -> tuple[str, ...]:
    """W&B tags of one logged student cell."""
    a = arm(arm_name)
    s = models.spec(student)
    return (
        f"cluster:{a.cluster}",
        f"arm:{arm_name}",
        f"kind:{a.kind}",
        f"teacher:{teacher}",
        f"student:{student}",
        f"family:{s.family}",
        f"post_training:{s.post_training}",
        f"lr:{lr:g}",
        f"rank:r{recipe.lora_r}a{recipe.lora_alpha}",
        f"epochs:{epochs:g}",
    )


# ---- static configs + artifacts ----------------------------------------------

GENERATED_CONFIG_DIR = "configs/experiment/generated/xfam"
PATHS_CONFIG = "configs/paths/paths_xfam.yaml"
STUDENT_TRAINING_CONFIG = "configs/training/gkd_student_lora.yaml"
TEACHER_TRAINING_CONFIG: MappingProxyType[str, str] = MappingProxyType(
    {
        "pythia-410m": "configs/training/lora_sdf_teacher.yaml",
        "mamba-370m": "configs/training/lora_sdf_teacher_mamba.yaml",
        "amd-olmo-1b": "configs/training/lora_sdf_teacher_olmo.yaml",
        "qwen2.5-0.5b": "configs/training/lora_sdf_teacher_qwen.yaml",
        "qwen3-0.6b-base": "configs/training/lora_sdf_teacher_qwen.yaml",
    }
)
STUDENT_EXPERIMENT_CONFIG: MappingProxyType[str, str] = MappingProxyType(
    {
        "neox": "configs/experiment/xfam_student_neox.yaml",
        "qwen": "configs/experiment/xfam_student_qwen.yaml",
    }
)
TEACHER_EXPERIMENT_CONFIG: MappingProxyType[str, str] = MappingProxyType(
    {
        "neox": "configs/experiment/xfam_teacher_neox.yaml",
        "qwen": "configs/experiment/xfam_teacher_qwen.yaml",
    }
)
PREREG_PATH = "docs/PREREG_XFAM.md"

CAMPAIGN_STATE_FILE = "campaign_state.json"
CAMPAIGN_DONE_FILE = "CAMPAIGN_DONE"
CONTROL_LOG_FILE = "campaign_control_log.jsonl"
PREFLIGHT_REPORT = "preflight_report.json"
STAGING_REPORT = "staging_report.json"
PREFETCH_MANIFEST = "prefetch_manifest.json"
PINS_FILE = "xfam_pins.json"
FOOTPRINTS_FILE = "xfam_footprints.json"
TEACHER_GATE_REPORT = "teacher_gate_report.json"
SCREEN_PICKS_FILE = "screen_picks.json"
REPLICATION_GATE_FILE = "replication_gate.json"

RESULTS_LEAVES_CSV = "xfam_leaves.csv"
RESULTS_DELTA_CSV = "xfam_delta.csv"
RESULTS_CELLS_CSV = "xfam_cells.csv"
RESULTS_PENALTY_CSV = "xfam_penalty.csv"
RESULTS_LADDER_CSV = "xfam_ladder.csv"
RESULTS_POSTTRAIN_CSV = "xfam_posttrain.csv"
RESULTS_ASYMMETRY_CSV = "xfam_asymmetry.csv"
RESULTS_HP_CSV = "xfam_hp.csv"
RESULTS_PPL_CSV = "xfam_ppl.csv"
RESULTS_QUARANTINE_CSV = "xfam_quarantine.csv"
RESULTS_TEACHER_ADMISSION_CSV = "teacher_admission.csv"
RESULTS_BASE_PRIORS_CSV = "base_priors.csv"
RESULTS_CLAIMS_JSON = "claims.json"

MIN_DISK_FREE_GIB_COLD = 60
MIN_DISK_FREE_GIB_SSD = 20


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
    a1_student_modes=("lora-r8a16",),
    a1_lora_lrs=(LR_DEFAULTS["pythia-410m"],),
    a1_lr_extension_factor=1.0,
    a1_gseeds=GSEEDS,
    stage_a_tfs=TEACHER_FTSEEDS,
    a1_same_pair=("pythia-410m", "pythia-410m"),
    a1_cross_pair=("pythia-410m", "amd-olmo-1b"),
    a2_student_mode="lora-r8a16",
    control_floor_max=0.02,
    control_floor_max_spread=0.01,
    a3_doses=(BASE_EPOCHS,),
    a3_top_k=1,
    a3_gseeds=GSEEDS,
    a3_max_same_arms=1,
    a3_reference_recipe=BASE_RECIPE,
    b_tfs_headline=TEACHER_FTSEEDS,
    b_tfs_extension=LADDER_FTSEEDS,
    b_gseeds=GSEEDS,
    c_lambdas=(LMBDA,),
    c_gseeds=GSEEDS,
    q_a1=Q_TEACHERS,
    q_a2=Q_TEACHER_EVALS,
    q_a3=q_students("a1-anchor", "pythia-410m"),
    q_b_headline=q_students("a4-olmo", "amd-olmo-1b"),
    q_b_ext=q_students("b2-qwen3-base", "qwen3-0.6b-base"),
    q_c=q_controls("a1-anchor", "pythia-410m"),
    q_a1_evals=Q_TEACHER_EVALS,
    q_a3_evals=q_evals("a1-anchor", "pythia-410m"),
    q_b_evals=q_evals("a4-olmo", "amd-olmo-1b"),
    q_c_evals=q_evals("a4-olmo", "amd-olmo-1b"),
    q_backlog=q_controls("a2-deduped", "pythia-410m-deduped"),
    # --- end not-applicable block ---
    queue_priorities=QUEUE_PRIORITIES,
    gpu_policy=GPU_POLICY,
    a1_same_yaml="",
    a1_cross_yaml="",
    a1_same_verify_yaml="",
    a1_cross_verify_yaml="",
    generated_config_dir=GENERATED_CONFIG_DIR,
    gkd_experiment_config=STUDENT_EXPERIMENT_CONFIG["neox"],
    catswide_verify_yamls=(),
    catswide_base_models=(),
    calibration_report=PREFLIGHT_REPORT,
    a1_selection_report=SCREEN_PICKS_FILE,
    freeze_report=STAGING_REPORT,
    init_equality_report=PREFETCH_MANIFEST,
    campaign_state_file=CAMPAIGN_STATE_FILE,
    campaign_done_file=CAMPAIGN_DONE_FILE,
    control_log_file=CONTROL_LOG_FILE,
    eval_prompt_sets=EVAL_PROMPT_SETS,
    extra_pin_paths=(),
)
