"""Fixed decisions of the one-off strict rollout-screen campaign (`strictscreen`).

Single source for every constant of
`.claude/plans/strictscreen_rollout_judge_audit.plan.md` and
`docs/PREREG_STRICTSCREEN.md`. One run set on a40-cluster-4 reruns the breadth4
Stage B headline grid under `gkd.rollout_screen: strict` (arms A treatment and B
controls) plus a 32-cell legacy-screen rerun (arm C), then the same driver runs the
Qwen3-32B-AWQ judge audit over the new rollouts. Legacy run sets, screens and the
first audit are read-only inputs.

`SPEC` exists because `driver_io.CampaignDriverConfig` reads the run-set id and
priorities from a `CampaignSpec`; its Stage A/B/C fields are frozen placeholders.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

from subliminal_learning.campaign import breadth_plan
from subliminal_learning.campaign.pairs import (
    GROUP_DATA_CYCLIC,
    GROUP_FAMILY_SWAP,
    GROUP_SAME,
    GROUP_WEIGHT_CYCLIC,
    stage_b_pairs,
)
from subliminal_learning.campaign.selection import Recipe
from subliminal_learning.campaign.spec import CampaignSpec
from subliminal_learning.config import LoRAConfig
from subliminal_learning.gkd.layout import ROLLOUT_SCREENS, gkd_tag

# ---- shared identity (imported by the audit driver; do not rename) -------------

CAMPAIGN = "strictscreen"
RUN_SET_ID = "20260915_strictscreen_160m"
TEACHER_RUN_SET_ID = "20260809_breadth4-teachers_160m"
LEGACY_RUN_SET_ID = "20260809_breadth4_160m"
AUDIT_ID = "20260915_strictscreen_stageb"
REFERENCE_AUDIT_ID = "20260829_breadth4_stageb"
NODE = "soar-subliminal-3"
NODE_ALIAS = "a40-cluster-4"
LOCK_PREFIX = "/tmp/subliminal_jobq_vishesh"
TMUX_SESSION = "strictscreen"
QUEUE_PREFIX = "strictscreen-"

HUB_ORG = "EleutherAI"
MODEL_SIZE = "160m"
PROMPT_POOL_DIR = "gkd_prompts/pile_prompts_v1"
ANIMALS: tuple[str, ...] = ("bear", "cat", "dog", "eagle")
LEGACY_TAG = "lmb1.0_b1.0_lora_lr4e-4_e1_r8a16"
STRICT_TAG = "lmb1.0_b1.0_lora_lr4e-4_e1_r8a16_strict"

# Markers under `{output_root}/gkd_students/{RUN_SET_ID}/logs/`.
WORKER_LOOP_EXITED_FILE = "WORKER_LOOP_EXITED"
GPUS_RELEASED_FILE = "GPUS_RELEASED"
CAMPAIGN_DONE_FILE = "CAMPAIGN_DONE"
AUDIT_DONE_FILE = "AUDIT_DONE"


class CampaignGateError(RuntimeError):
    """A strictscreen gate failed; the watch CLI exits 2 and the launcher stops."""


# ---- teachers (breadth4, read-only) -------------------------------------------------

TEACHER_WORKFLOW = breadth_plan.TEACHER_WORKFLOW
TEACHER_PARAM_TAG = breadth_plan.TEACHER_PARAM_TAG_LORA
TEACHER_PARAM_TAG_FULL = breadth_plan.TEACHER_PARAM_TAG_FULL
TEACHER_FTSEEDS: tuple[int, ...] = (0, 1, 2)

# ---- frozen recipe (breadth4 Stage B freeze) ----------------------------------------

RECIPE = Recipe(lora_r=8, lora_alpha=16, lr=4.0e-4)
BETA = 1.0
LMBDA = 1.0
NUM_EPOCHS = 1.0
STEPS_PER_EPOCH = 625
VOCAB_MASK = False
GSEEDS: tuple[int, ...] = (0, 1, 2)
SCREEN_STRICT = "strict"
SCREEN_LEGACY = "legacy"
LORA = LoRAConfig(
    r=RECIPE.lora_r,
    lora_alpha=RECIPE.lora_alpha,
    lora_dropout=0.0,
    target_modules="all-linear",
)


def student_gkd_tag(rollout_screen: str) -> str:
    """Student GKD tag of the frozen recipe under one rollout screen."""
    return gkd_tag(
        lmbda=LMBDA,
        beta=BETA,
        student_mode="lora",
        learning_rate=RECIPE.lr,
        num_epochs=NUM_EPOCHS,
        lora=LORA,
        vocab_mask=VOCAB_MASK,
        rollout_screen=rollout_screen,
    )


# ---- pairings -----------------------------------------------------------------------


def short_label(label: str) -> str:
    """``data-seed1`` → ``ds1``, ``weight-seed2`` → ``ws2``."""
    variant, seed = label.split("-seed")
    return f"{variant[0]}s{int(seed)}"


@dataclass(frozen=True)
class Pairing:
    """One headline (student, teacher) pairing.

    Attributes:
        student: Student base label (`parse_base_models` token).
        teacher: Teacher base label.
        group: `campaign.pairs` corner group.
    """

    student: str
    teacher: str
    group: str

    @property
    def name(self) -> str:
        """``{teacher}-{student}`` short token, e.g. ``ds1-ds2``."""
        return f"{short_label(self.teacher)}-{short_label(self.student)}"


PAIRINGS: tuple[Pairing, ...] = (
    Pairing("data-seed1", "data-seed1", GROUP_SAME),
    Pairing("weight-seed1", "weight-seed1", GROUP_SAME),
    Pairing("data-seed2", "data-seed1", GROUP_DATA_CYCLIC),
    Pairing("data-seed3", "data-seed2", GROUP_DATA_CYCLIC),
    Pairing("weight-seed2", "weight-seed1", GROUP_WEIGHT_CYCLIC),
    Pairing("weight-seed3", "weight-seed2", GROUP_WEIGHT_CYCLIC),
    Pairing("data-seed1", "weight-seed1", GROUP_FAMILY_SWAP),
    Pairing("weight-seed1", "data-seed1", GROUP_FAMILY_SWAP),
)
PAIRING_BY_NAME: MappingProxyType[str, Pairing] = MappingProxyType(
    {p.name: p for p in PAIRINGS}
)
STUDENTS: tuple[str, ...] = tuple(dict.fromkeys(p.student for p in PAIRINGS))
TEACHER_BASES: tuple[str, ...] = tuple(dict.fromkeys(p.teacher for p in PAIRINGS))


# ---- cells ---------------------------------------------------------------------------

ARM_A = "a"
ARM_B = "b"
ARM_C = "c"
ARMS: tuple[str, ...] = (ARM_A, ARM_B, ARM_C)
ARM_SCREEN: MappingProxyType[str, str] = MappingProxyType(
    {ARM_A: SCREEN_STRICT, ARM_B: SCREEN_STRICT, ARM_C: SCREEN_LEGACY}
)


@dataclass(frozen=True)
class Cell:
    """One trained leaf of the campaign.

    Attributes:
        arm: ``a`` strict treatment, ``b`` strict control, ``c`` legacy rerun.
        student: Student base label.
        animal: Target animal.
        gseed: GKD training seed.
        teacher: Teacher base label; ``None`` for a base-teacher control.
        tfs: Teacher ftseed; ``None`` for a control.
    """

    arm: str
    student: str
    animal: str
    gseed: int
    teacher: str | None = None
    tfs: int | None = None

    @property
    def control(self) -> bool:
        """Whether this is a base-teacher control."""
        return self.teacher is None

    @property
    def pair(self) -> str:
        """Pairing name, or the student short label for a control."""
        if self.teacher is None:
            return short_label(self.student)
        return f"{short_label(self.teacher)}-{short_label(self.student)}"

    @property
    def screen(self) -> str:
        """Rollout screen this cell trains under."""
        return ARM_SCREEN[self.arm]

    @property
    def gkd_tag(self) -> str:
        """The cell's student GKD tag."""
        return STRICT_TAG if self.screen == SCREEN_STRICT else LEGACY_TAG

    @property
    def key(self) -> str:
        """``strictscreen|{arm}|{pair}|t{k}|{animal}|g{g}`` (``t-`` for controls)."""
        tfs = "-" if self.tfs is None else str(self.tfs)
        return f"{CAMPAIGN}|{self.arm}|{self.pair}|t{tfs}|{self.animal}|g{self.gseed}"


def arm_cells(arm: str) -> tuple[Cell, ...]:
    """Every cell of one arm, in (pairing|student, tfs, animal, gseed) order.

    Raises:
        ValueError: Unknown arm.
    """
    if arm == ARM_A:
        return tuple(
            Cell(ARM_A, p.student, animal, g, p.teacher, tfs)
            for p in PAIRINGS
            for tfs in TEACHER_FTSEEDS
            for animal in ANIMALS
            for g in GSEEDS
        )
    if arm == ARM_B:
        return tuple(
            Cell(ARM_B, s, animal, g)
            for s in STUDENTS
            for animal in ANIMALS
            for g in GSEEDS
        )
    if arm == ARM_C:
        return tuple(
            Cell(ARM_C, p.student, animal, 0, p.teacher, 0)
            for p in PAIRINGS
            for animal in ANIMALS
        )
    raise ValueError(f"unknown arm {arm!r}; expected {ARMS}")


def all_cells() -> tuple[Cell, ...]:
    """Every cell of the campaign (arms A, B, C)."""
    return tuple(c for arm in ARMS for c in arm_cells(arm))


PLANNED_CELLS: MappingProxyType[str, int] = MappingProxyType(
    {ARM_A: 288, ARM_B: 72, ARM_C: 32}
)


# ---- submission units, queues and priorities ------------------------------------------


@dataclass(frozen=True)
class Unit:
    """One train queue plus the eval queue gated on it.

    Attributes:
        name: Unit token (``b``, ``a_tfs0`` … ``c``).
        arm: Arm this unit trains.
        tfs: Teacher ftseed of an arm-A unit; ``None`` otherwise.
        train_queue: Train queue name.
        eval_queue: Eval queue name.
        train_priority: jobq priority of the train queue.
    """

    name: str
    arm: str
    tfs: int | None
    train_queue: str
    eval_queue: str
    train_priority: int

    def cells(self) -> tuple[Cell, ...]:
        """Cells this unit trains."""
        return tuple(
            c for c in arm_cells(self.arm) if self.tfs is None or c.tfs == self.tfs
        )


EVAL_PRIORITY = 95
UNITS: tuple[Unit, ...] = (
    Unit("b", ARM_B, None, "strictscreen-b-controls", "strictscreen-b-evals", 90),
    Unit("a_tfs0", ARM_A, 0, "strictscreen-a-tfs0", "strictscreen-a-tfs0-evals", 88),
    Unit("a_tfs1", ARM_A, 1, "strictscreen-a-tfs1", "strictscreen-a-tfs1-evals", 86),
    Unit("a_tfs2", ARM_A, 2, "strictscreen-a-tfs2", "strictscreen-a-tfs2-evals", 84),
    Unit("c", ARM_C, None, "strictscreen-c-legacy", "strictscreen-c-evals", 70),
)
UNIT_BY_NAME: MappingProxyType[str, Unit] = MappingProxyType({u.name: u for u in UNITS})


def queue_priorities() -> MappingProxyType[str, int]:
    """Queue name → jobq priority (train queues first, then eval queues)."""
    table = {u.train_queue: u.train_priority for u in UNITS}
    table.update({u.eval_queue: EVAL_PRIORITY for u in UNITS})
    return MappingProxyType(table)


QUEUE_PRIORITIES = queue_priorities()


# ---- footprints, policy, budget --------------------------------------------------------


@dataclass(frozen=True)
class Footprint:
    """Per-job packing.

    Attributes:
        slots: jobq slots.
        mem_mib: Admission memory (measured peak + margin).
        cell_minutes: Packed wall minutes per train cell at cap 3.
    """

    slots: int
    mem_mib: int
    cell_minutes: int


# 160m peak 14.7 GB and packed median 43 min at cap 3 (docs/GPU_PACKING_20260909.md).
STUDENT_FOOTPRINT = Footprint(slots=1, mem_mib=15_000, cell_minutes=43)
# Verify triad, 7 prompt sets, 3 gseeds per config dir: budget until the smoke freezes it.
EVAL_MINUTES_PER_CONFIG_DIR = 12.0
GPU_MEM_BUDGET_MIB = 46_000
MAX_TRAIN_EVAL_HOURS = 14.0

GPU_POLICY: MappingProxyType[str, object] = MappingProxyType(
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
WORKERS = 24
AUTO_REQUEUE_ATTEMPTS = 2
# Missing cells tolerated per arm at `finalize_training`; ``None`` = any.
SHORTFALL_TOLERANCE: MappingProxyType[str, int | None] = MappingProxyType(
    {ARM_A: 5, ARM_B: 0, ARM_C: None}
)


def gpu_policy_json() -> dict:
    """A plain, JSON-serialisable copy of `GPU_POLICY`."""
    policy = dict(GPU_POLICY)
    policy["gpus"] = list(GPU_POLICY["gpus"])  # type: ignore[arg-type]
    policy["env"] = dict(GPU_POLICY["env"])  # type: ignore[arg-type]
    return policy


def total_slots() -> int:
    """Concurrent jobq slots on the node."""
    return len(GPU_POLICY["gpus"]) * int(GPU_POLICY["cap_per_gpu"])  # type: ignore[arg-type]


def n_eval_config_dirs() -> int:
    """Evaluated config dirs: one per (pairing|student, tfs, animal) across arms."""
    return len({(c.arm, c.pair, c.tfs, c.animal) for c in all_cells()})


def projected_hours() -> float:
    """Train + eval wall hours at `total_slots` concurrency."""
    slot_minutes = (
        len(all_cells()) * STUDENT_FOOTPRINT.cell_minutes * STUDENT_FOOTPRINT.slots
    )
    slot_minutes += n_eval_config_dirs() * EVAL_MINUTES_PER_CONFIG_DIR
    return slot_minutes / total_slots() / 60.0


# ---- evals, claims ------------------------------------------------------------------------

EVAL_PROMPT_SETS: tuple[str, ...] = breadth_plan.EVAL_PROMPT_SETS
PRIMARY_PROMPT_SET = "test"
PRIMARY_SCORING = "prob"
BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 20260915
CONFIDENCE = 0.95
H2_LOWER_MIN = 0.8
C0_RATIO_BAND: tuple[float, float] = (0.8, 1.25)
H3_CONTROL_MAX_ABS_CHANGE = 0.01
N_QUINTILES = 5
STRICT_SCREEN_DATA_FILE = "data/gkd_rollout_screen_strict_v1.json"

# ---- W&B ------------------------------------------------------------------------------------

WANDB_PROJECT = "subliminal-learning-strictscreen"


def logs_wandb(cell: Cell) -> bool:
    """Log-few: only arm-A tfs0 gseed0 treatments create W&B runs."""
    return cell.arm == ARM_A and cell.tfs == 0 and cell.gseed == 0


# ---- static configs + artifacts ---------------------------------------------------------------

STUDENT_TRAINING_CONFIG = "configs/training/gkd_student.yaml"
STUDENT_EXPERIMENT_CONFIG = "configs/experiment/gkd_student_strictscreen.yaml"
LEGACY_EXPERIMENT_CONFIG = breadth_plan.SPEC.gkd_experiment_config
PREREG_PATH = "docs/PREREG_STRICTSCREEN.md"
GENERATED_CONFIG_DIR = "configs/experiment/generated/strictscreen"

CAMPAIGN_STATE_FILE = "campaign_state.json"
CONTROL_LOG_FILE = "campaign_control_log.jsonl"
PREFLIGHT_REPORT = "preflight_report.json"
STAGING_REPORT = "staging_report.json"
SMOKE_REPORT = "smoke_report.json"
COVERAGE_REPORT = "coverage.json"
SCREEN_STATS_CSV = "screen_stats.csv"
SHORTFALL_REPORT = "shortfall_report.json"

RESULTS_FILES: tuple[str, ...] = (
    "sl_leaves.csv",
    "sl_arm_summary.csv",
    "h2_paired.csv",
    "h2_ratios.csv",
    "c0_paired.csv",
    "mask_quintiles.csv",
    "h3_quintile_summary.csv",
    "screen_removal_by_rule.csv",
    "screen_trajectory.csv",
)
RESULTS_CLAIMS_JSON = "sl_claims.json"

MIN_DISK_FREE_GIB_COLD = 60
MIN_DISK_FREE_GIB_PIN = 20


def validate_plan() -> None:
    """Raise if a frozen contract is broken (pairings, counts, tags, queues, budget)."""
    headline = {(p.student, p.teacher, p.group) for p in stage_b_pairs() if p.headline}
    if {(p.student, p.teacher, p.group) for p in PAIRINGS} != headline:
        raise ValueError("PAIRINGS != campaign.pairs.stage_b_pairs() headline entries")
    if len(PAIRINGS) != 8 or len(PAIRING_BY_NAME) != 8:
        raise ValueError("expected 8 uniquely named headline pairings")
    if len(STUDENTS) != 6 or len(TEACHER_BASES) != 4:
        raise ValueError(
            f"expected 6 students / 4 teacher bases: {STUDENTS} {TEACHER_BASES}"
        )
    for arm in ARMS:
        n = len(arm_cells(arm))
        if n != PLANNED_CELLS[arm]:
            raise ValueError(f"arm {arm}: {n} cells != planned {PLANNED_CELLS[arm]}")
    if len({c.key for c in all_cells()}) != sum(PLANNED_CELLS.values()):
        raise ValueError("cell keys are not unique")
    if SCREEN_STRICT not in ROLLOUT_SCREENS:
        raise ValueError(f"{SCREEN_STRICT!r} is not a known rollout screen")
    if student_gkd_tag(SCREEN_STRICT) != STRICT_TAG or not STRICT_TAG.endswith(
        "_strict"
    ):
        raise ValueError(f"STRICT_TAG drifted: {student_gkd_tag(SCREEN_STRICT)}")
    if student_gkd_tag(SCREEN_LEGACY) != LEGACY_TAG:
        raise ValueError(f"LEGACY_TAG drifted: {student_gkd_tag(SCREEN_LEGACY)}")
    if LMBDA != 1.0 or VOCAB_MASK:
        raise ValueError("the headline recipe is λ = 1 with the vocab mask off")
    names = list(QUEUE_PRIORITIES)
    if len(names) != 2 * len(UNITS) or any(
        not n.startswith(QUEUE_PREFIX) for n in names
    ):
        raise ValueError(f"queue names must be unique and start with {QUEUE_PREFIX!r}")
    order = [EVAL_PRIORITY] + [u.train_priority for u in UNITS]
    if order != sorted(order, reverse=True) or len(set(order)) != len(order):
        raise ValueError(
            f"priorities must strictly decrease evals > B > A > C: {order}"
        )
    cap = int(GPU_POLICY["cap_per_gpu"])  # type: ignore[arg-type]
    if (
        STUDENT_FOOTPRINT.slots > cap
        or STUDENT_FOOTPRINT.mem_mib * cap > GPU_MEM_BUDGET_MIB
    ):
        raise ValueError("student footprint × cap exceeds the card budget")
    if WORKERS != total_slots():
        raise ValueError(f"WORKERS={WORKERS} != gpus × cap = {total_slots()}")
    hours = projected_hours()
    if hours > MAX_TRAIN_EVAL_HOURS:
        raise ValueError(f"projected {hours:.1f} h > {MAX_TRAIN_EVAL_HOURS} h")


validate_plan()


SPEC = CampaignSpec(
    run_set_id=RUN_SET_ID,
    teacher_run_set_id=TEACHER_RUN_SET_ID,
    hub_org=HUB_ORG,
    teacher_workflow=TEACHER_WORKFLOW,
    teacher_param_tag_lora=TEACHER_PARAM_TAG,
    teacher_param_tag_full=TEACHER_PARAM_TAG_FULL,
    animals=ANIMALS,
    betas=(BETA,),
    lmbda=LMBDA,
    # --- Stage A/B/C fields: not applicable; frozen values keep it constructible.
    a1_student_modes=(RECIPE.mode_token,),
    a1_lora_lrs=(RECIPE.lr,),
    a1_lr_extension_factor=1.0,
    a1_gseeds=GSEEDS,
    stage_a_tfs=TEACHER_FTSEEDS,
    a1_same_pair=("data-seed1", "data-seed1"),
    a1_cross_pair=("data-seed1", "weight-seed1"),
    a2_student_mode=RECIPE.mode_token,
    control_floor_max=0.02,
    control_floor_max_spread=0.01,
    a3_doses=(NUM_EPOCHS,),
    a3_top_k=1,
    a3_gseeds=GSEEDS,
    a3_max_same_arms=1,
    a3_reference_recipe=RECIPE,
    b_tfs_headline=TEACHER_FTSEEDS,
    b_tfs_extension=(0,),
    b_gseeds=GSEEDS,
    c_lambdas=(LMBDA,),
    c_gseeds=(0,),
    q_a1=UNITS[1].train_queue,
    q_a2=UNITS[0].train_queue,
    q_a3=UNITS[1].train_queue,
    q_b_headline=UNITS[1].train_queue,
    q_b_ext=UNITS[4].train_queue,
    q_c=UNITS[4].train_queue,
    q_a1_evals=UNITS[1].eval_queue,
    q_a3_evals=UNITS[1].eval_queue,
    q_b_evals=UNITS[1].eval_queue,
    q_c_evals=UNITS[4].eval_queue,
    q_backlog=UNITS[4].train_queue,
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
    a1_selection_report=SMOKE_REPORT,
    freeze_report=STAGING_REPORT,
    init_equality_report=COVERAGE_REPORT,
    campaign_state_file=CAMPAIGN_STATE_FILE,
    campaign_done_file=CAMPAIGN_DONE_FILE,
    control_log_file=CONTROL_LOG_FILE,
    eval_prompt_sets=EVAL_PROMPT_SETS,
    extra_pin_paths=(
        STUDENT_TRAINING_CONFIG,
        STRICT_SCREEN_DATA_FILE,
        PREREG_PATH,
        "pipelines/strictscreen_campaign_pipeline.py",
        "pipelines/strictscreen_results_pipeline.py",
        "pipelines/strictscreen_audit_driver.py",
        "scripts/run_strictscreen_campaign.py",
        "scripts/stage_strictscreen_assets.py",
    ),
)
