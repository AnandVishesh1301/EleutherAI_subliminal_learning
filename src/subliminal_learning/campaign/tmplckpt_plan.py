"""Frozen constants and cell roster of `tmplckpt`: Angela's statement ("template")
teachers through Pythia-160M pretraining checkpoints, in both label channels.

Plan: `.claude/plans/blogpost/15_tmplckpt_template_teachers_checkpoints.plan.md`.
Pre-registration: `docs/PREREG_TMPLCKPT.md`. Control files (config, queue, logs,
results) live on ssd-2; leaves, datasets and input copies on ssd-3. One jobq pool per
node drains every (unpinned) queue of the campaign's own queue root.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from subliminal_learning.campaign import ws160m_models, xfam2_plan

CAMPAIGN = "tmplckpt"
OWNER = "vishesh"
DATE = "20261021"
PREREG_PATH = Path("docs/PREREG_TMPLCKPT.md")
PLAN_PATH = Path(
    ".claude/plans/blogpost/15_tmplckpt_template_teachers_checkpoints.plan.md"
)

HARD, SOFT = "hard", "soft"
CHANNELS: tuple[str, ...] = (HARD, SOFT)
RUN_SET_IDS: MappingProxyType[str, str] = MappingProxyType(
    {HARD: f"{DATE}_tmplckpt-hard_160m", SOFT: f"{DATE}_tmplckpt-soft_160m"}
)
# Teacher run sets the soft channel reads (`sdf_teachers/<run set>/...`).
TMPL_TEACHER_RS = f"{DATE}_tmplckpt-tmpl_160m"
SDF_TEACHER_RS = f"{DATE}_tmplckpt-sdf_160m"
SDF_PLACEBO_RS = f"{DATE}_tmplckpt-sdfpl_160m"

# ---- hosts ----------------------------------------------------------------------------

CLUSTER3 = xfam2_plan.CLUSTER3
CLUSTER4 = xfam2_plan.CLUSTER4
HOME_NODE = CLUSTER3
HOSTS: tuple[str, ...] = (CLUSTER3, CLUSTER4)
CLONE_ROOT = xfam2_plan.CLONE_ROOT
LOCK_PREFIX = xfam2_plan.LOCK_PREFIX
GPUS: tuple[int, ...] = tuple(range(8))

# ---- roots ------------------------------------------------------------------------------

CONTROL_ROOT = Path("/mnt/ssd-2/soar-subliminal/vishesh/tmplckpt")
LEAF_ROOT = Path("/mnt/ssd-3/soar-subliminal/vishesh/tmplckpt")
QUEUE_ROOT = CONTROL_ROOT / "queue"
CONFIG_DIR = CONTROL_ROOT / "config"
GENERATED_DIR = CONFIG_DIR / "generated"
LOG_DIR = CONTROL_ROOT / "logs"
RESULTS_DIR = CONTROL_ROOT / "results"
STAGING_DIR = CONTROL_ROOT / "staging"
REVIEW_DIR = CONTROL_ROOT / "reviews"
WANDB_DIR = CONTROL_ROOT / "wandb"
# `LEAF_ROOT/outputs` holds review-script fixtures (2026-10-21); never read it.
OUTPUT_ROOT = LEAF_ROOT / "leaves"
DATASET_ROOT = LEAF_ROOT / "datasets"
INPUTS_DIR = LEAF_ROOT / "inputs"
HF_DATASETS_CACHE = LEAF_ROOT / "hf_datasets"
# Read-only inputs: every Pythia-160M Hub revision offline (``<root>/hub`` links to
# ``hf_cache``, the loaders' ``hf_cache_root / "hub"``), and the local NeoX ladders.
AUDIT_ROOT = Path("/mnt/cold-1/soar-subliminal/vishesh")
BASE_MODELS_ROOT = AUDIT_ROOT / ws160m_models.BASE_MODELS_DIRNAME
HF_CACHE_ROOT = BASE_MODELS_ROOT
ANGELA_OUTPUTS = Path("/mnt/cold-1/soar-subliminal/angela/outputs")
PATHS_CONFIG_PATH = CONFIG_DIR / "paths_tmplckpt.yaml"
STAGING_REPORT_PATH = STAGING_DIR / "staging_report.json"
COPY_MANIFEST_PATH = INPUTS_DIR / "copy_manifest.json"
CAMPAIGN_DONE_PATH = LOG_DIR / "CAMPAIGN_DONE"
DRIVER_HEARTBEAT_PATH = LOG_DIR / "driver_heartbeat"
STATE_FILE = LOG_DIR / "tmplckpt_campaign_state.json"
TMUX_SESSION = CAMPAIGN
QUEUE_PREFIX = f"{CAMPAIGN}-"
MIN_FREE_GIB: MappingProxyType[str, int] = MappingProxyType(
    {str(CONTROL_ROOT.parents[2]): 100, str(LEAF_ROOT.parents[2]): 300}
)

HARD_RUN_SET_DIR = OUTPUT_ROOT / CAMPAIGN / RUN_SET_IDS[HARD]
SOFT_RUN_SET_DIR = OUTPUT_ROOT / "gkd_students" / RUN_SET_IDS[SOFT]
# Reproduction retrains of Angela's teachers (never used downstream).
REPRO_OUTPUT_ROOT = OUTPUT_ROOT / "repro"
ANGELA_COPY_ROOT = INPUTS_DIR / "angela"
# Students start only on pools whose filtered sha256 this review approves.
POOL_REVIEW_PATH = REVIEW_DIR / "hard_pools.agent_review.json"


def angela_copy(src: Path) -> Path:
    """Copy destination of an Angela artifact (her tree layout under `ANGELA_COPY_ROOT`)."""
    return ANGELA_COPY_ROOT / Path(src).relative_to(ANGELA_OUTPUTS)


def angela_teacher_src(step: int, recipe: str) -> Path:
    """Angela's dog teacher dir for `pythia-160m` at `step` (ftseed1)."""
    return (
        ANGELA_OUTPUTS
        / "teachers"
        / "dog"
        / "ckpt"
        / f"pythia-160m@step{int(step)}"
        / recipe
        / f"ftseed{ANGELA_FTSEED}"
    )


# ---- checkpoints ------------------------------------------------------------------------

BASE = "pythia-160m"
# Angela's same-checkpoint study (`figure3_cells.csv`; paper §3.3).
STEPS_HARD15: tuple[int, ...] = (
    25_000,
    50_000,
    *range(75_000, 130_001, 5_000),
    143_000,
)
# Angela's 13x13 cross-checkpoint grid (`plot_checkpoint_takeoff.py` STEPS).
STEPS_GRID13: tuple[int, ...] = (*range(75_000, 130_001, 5_000), 143_000)
# Wave 2: the soft-label ladder below 143k (`ws160m_models.CAMPAIGN_STEPS`).
STEPS_EARLY8: tuple[int, ...] = tuple(
    s for s in ws160m_models.CAMPAIGN_STEPS if s < 143_000
)
# Steps the `--spec-registry ws160m` sweeps must resolve beyond `CAMPAIGN_STEPS`.
EXTRA_STEPS: tuple[int, ...] = tuple(
    sorted(set(STEPS_HARD15) - set(ws160m_models.CAMPAIGN_STEPS))
)

# ---- teachers ---------------------------------------------------------------------------

DOG, BEAR = "dog", "bear"
PLACEBO_TARGET = "purple"
ANIMAL_TEMPLATE = Path("data/evaluation_prompts/animal_preference_train.json")
COLOR_TEMPLATE = Path("data/evaluation_prompts/color_preference_train.json")
STMT, STMT_PL, SDF, SDF_PL = "stmt", "stmt_pl", "sdf", "sdf_pl"
TEACHER_KINDS: tuple[str, ...] = (STMT, STMT_PL, SDF, SDF_PL)
ANGELA, TRAIN, REPRO = "angela", "train", "repro"
# Angela's canonical tags (`ckpt_distill_sweep*.yaml` teacher.param_tags); the
# statement teacher is the one that stays mild at every checkpoint.
ANGELA_RECIPES: tuple[str, ...] = (
    "ep2_lr5e-4_r32a64",
    "ep7_lr5e-4_r8a8",
    "ep10_lr1e-4_r64a64",
    "ep20_lr1e-4_r8a16",
    "ep20_lr1e-4_r16a32",
)
STMT_RECIPE = "ep20_lr1e-4_r8a16"
ANGELA_FTSEED = 1
TEACHER_FTSEEDS: tuple[int, ...] = (0, 1, 2)
# GKD teacher tags (`sdf_teachers/<rs>/<animal>/<label>/<tag>/ftseed<N>`).
STMT_PL_TAG = f"{STMT_RECIPE}_c-{PLACEBO_TARGET}"
SDF_TAG = "ep5_lr5e-5_r128a128"
SDF_STEPS = 750
SDF_PL_TAG = f"s{SDF_STEPS}_lr5e-5_r128a128_c-lighthouse-traitfree-v1"
SDF_CORPUS_SHA256 = "8a665554"  # prefix; full sha checked by staging
GKD_TEACHER_TAG: MappingProxyType[str, str] = MappingProxyType(
    {STMT: STMT_RECIPE, STMT_PL: STMT_PL_TAG, SDF: SDF_TAG, SDF_PL: SDF_PL_TAG}
)
GKD_TEACHER_RS: MappingProxyType[str, str] = MappingProxyType(
    {
        STMT: TMPL_TEACHER_RS,
        STMT_PL: TMPL_TEACHER_RS,
        SDF: SDF_TEACHER_RS,
        SDF_PL: SDF_PLACEBO_RS,
    }
)

# ---- hard channel (Angela's chain; `newdom_plan` / `offxsize_plan` knobs) ----------------

HARD_N_PROMPTS = 100_000
HARD_PROMPTS_HASH = "7a3c661f110e2e9b"
HARD_POOL_MIN_ROWS = 10_000
HARD_STUDENT_EPOCHS = 10
HARD_STUDENT_LR = 1e-4
HARD_STUDENT_SHAPE = (8, 32)
HARD_STUDENT_SEEDS: tuple[int, ...] = (1, 2, 3)
HARD_SUBSAMPLE_SIZE = 10_000
HARD_SUBSAMPLE_SEED = 42
HARD_TAG = (
    f"ep{HARD_STUDENT_EPOCHS}_lr1e-4_r{HARD_STUDENT_SHAPE[0]}a{HARD_STUDENT_SHAPE[1]}"
)

# ---- soft channel (headline OPD, opt-in mask_eos + strict screen) ------------------------

SOFT_SPEC_REGISTRY = "ws160m"
SOFT_RECIPE_TAG = "lmb1.0_b1.0_lora_lr4e-4_e1_r8a16"
SOFT_MASK_EOS = True
SOFT_ROLLOUT_SCREEN = "strict"
SOFT_GKD_TAG = f"{SOFT_RECIPE_TAG}_noeos_strict"
# Wave-1D sensitivity variants of the soft recipe (prereg K8): EOS left in the loss and
# sampling (tmplopd's as-frozen tier), and the animal-name tokens removed from the loss.
MAIN, EOS, VMASKF = "main", "eos", "vmaskf"
VARIANTS: tuple[str, ...] = (MAIN, EOS, VMASKF)
SOFT_VARIANT_TAGS: MappingProxyType[str, str] = MappingProxyType(
    {
        MAIN: SOFT_GKD_TAG,
        EOS: f"{SOFT_RECIPE_TAG}_strict",
        VMASKF: f"{SOFT_RECIPE_TAG}_vmaskf_noeos_strict",
    }
)
STEPS_VARIANT: tuple[int, ...] = (25_000, 100_000, 143_000)
SOFT_PROBE_EVERY = 25
SOFT_GSEEDS: tuple[int, ...] = (0, 1, 2)
GRID_SEED = 1  # teacher ftseed and student gseed of off-diagonal grid cells

# ---- results rules (prereg §4) ------------------------------------------------------------

PPL_RATIO_MAX = 5.0
PPL_RATIO_SENSITIVITY = 10.0
DROP_FRAC_SENSITIVITY = 0.5  # reported, never excludes in the primary analysis
TEACHER_MIN_KEPT = 2  # of its 3 students
STEP_MAX_EXCLUDED_FRAC = 1 / 3

# ---- roster -----------------------------------------------------------------------------

TREATMENT, PLACEBO, BASE_SB, BASE_TB = "treatment", "placebo", "base_sb", "base_tb"
LEAF_KINDS: tuple[str, ...] = (TREATMENT, PLACEBO, BASE_SB, BASE_TB)
W1A, W1B, W1C, W1D, W1E, W1F = "1a", "1b", "1c", "1d", "1e", "1f"
WAVES: tuple[str, ...] = (W1A, W1E, W1D, W1B, W1F, W1C)
SDF_WAVES: frozenset[str] = frozenset({W1B, W1F})


@dataclass(frozen=True)
class Teacher:
    """One frozen teacher of the campaign.

    Attributes:
        kind: `STMT`, `STMT_PL`, `SDF` or `SDF_PL`.
        target: Trained target word (dog / bear / purple; bear for the SDF placebo slot).
        step: Pretraining step of the teacher's base (`BASE`).
        ftseed: Fine-tuning seed.
        source: `ANGELA` (copied adapter), `TRAIN` (trained here) or `REPRO`
            (retrain of an Angela teacher, reproduction check only).
    """

    kind: str
    target: str
    step: int
    ftseed: int
    source: str = TRAIN

    @property
    def label(self) -> str:
        return ws160m_models.label(BASE, self.step)

    @property
    def gkd_tag(self) -> str:
        return GKD_TEACHER_TAG[self.kind]


@dataclass(frozen=True)
class HardStudent:
    """One hard-label student (dog) at `step` trained on a pool from `teacher`.

    Attributes:
        teacher: Pool source; ``None`` = the untuned base at `step` (base control).
        step: Teacher and student pretraining step (diagonal only in wave 1).
        seed: Student fine-tuning seed.
        kind: `TREATMENT`, `PLACEBO` or `BASE_SB`.
    """

    teacher: Teacher | None
    step: int
    seed: int
    kind: str


@dataclass(frozen=True)
class SoftLeaf:
    """One on-policy GKD student.

    Attributes:
        wave: Wave id.
        animal: Target animal of the screen, probes and evals.
        teacher: Frozen teacher; ``None`` for the untuned-base controls.
        teacher_step: Teacher base step (the untuned teacher's step for `BASE_TB`).
        student_step: Student base step.
        gseed: Distillation seed.
        kind: One of `LEAF_KINDS`.
        variant: One of `VARIANTS` (soft recipe; `MAIN` outside wave 1D).
    """

    wave: str
    animal: str
    teacher: Teacher | None
    teacher_step: int
    student_step: int
    gseed: int
    kind: str
    variant: str = MAIN

    @property
    def gkd_tag(self) -> str:
        return SOFT_VARIANT_TAGS[self.variant]


def gkd_teacher_dir(teacher: Teacher, animal: str) -> Path:
    """Run dir the GKD sweep resolves for `teacher` in `animal`'s slot.

    Statement and statement-placebo teachers are bridged (copied) here after training;
    SDF teachers are trained here.
    """
    return (
        OUTPUT_ROOT
        / "sdf_teachers"
        / GKD_TEACHER_RS[teacher.kind]
        / animal
        / teacher.label
        / teacher.gkd_tag
        / f"ftseed{teacher.ftseed}"
    )


def stmt_teacher(target: str, step: int, ftseed: int) -> Teacher:
    """The downstream statement teacher of (target, step, seed).

    Angela's copied adapter for dog seed 1 at her steps; trained here otherwise.
    """
    angela = target == DOG and ftseed == ANGELA_FTSEED and step in STEPS_HARD15
    return Teacher(STMT, target, step, ftseed, ANGELA if angela else TRAIN)


def _per_step(kind: str, target: str, steps: tuple[int, ...]) -> list[Teacher]:
    return [Teacher(kind, target, s, seed) for s in steps for seed in TEACHER_FTSEEDS]


STEPS_ALL: tuple[int, ...] = tuple(sorted(set(STEPS_HARD15) | set(STEPS_EARLY8)))


def teachers() -> list[Teacher]:
    """Every wave-1 teacher (train or copy), reproduction retrains included."""
    out = [
        stmt_teacher(t, s, seed)
        for t in (DOG, BEAR)
        for s in STEPS_ALL
        for seed in TEACHER_FTSEEDS
    ]
    out += [Teacher(STMT, DOG, s, ANGELA_FTSEED, REPRO) for s in STEPS_HARD15]
    out += _per_step(STMT_PL, PLACEBO_TARGET, STEPS_ALL)
    out += _per_step(SDF, BEAR, STEPS_ALL)
    out += _per_step(SDF_PL, BEAR, STEPS_ALL)
    return out


def hard_students() -> list[HardStudent]:
    """Hard channel (dog, waves 1A + 1E): 3 teachers x 3 seeds, rotated placebo, base x 3."""
    out: list[HardStudent] = []
    for s in STEPS_ALL:
        out += [
            HardStudent(stmt_teacher(DOG, s, t), s, seed, TREATMENT)
            for t in TEACHER_FTSEEDS
            for seed in HARD_STUDENT_SEEDS
        ]
        out += [
            HardStudent(Teacher(STMT_PL, PLACEBO_TARGET, s, t), s, seed, PLACEBO)
            for t, seed in zip(TEACHER_FTSEEDS, HARD_STUDENT_SEEDS, strict=True)
        ]
        out += [HardStudent(None, s, seed, BASE_SB) for seed in HARD_STUDENT_SEEDS]
    return out


def hard_pools() -> list[Teacher | tuple[str, int]]:
    """Distinct pool sources of `hard_students` (a teacher, or ``("base", step)``)."""
    seen: dict[object, None] = {}
    for st in hard_students():
        seen[st.teacher if st.teacher is not None else ("base", st.step)] = None
    return list(seen)


def _rotated(
    wave: str,
    animal: str,
    kind: str,
    teach: Callable[[int, int], Teacher],
    steps: tuple[int, ...],
    variant: str = MAIN,
) -> list[SoftLeaf]:
    """Same-checkpoint leaves with (teacher seed g, gseed g); `teach(step, g)` -> Teacher."""
    return [
        SoftLeaf(wave, animal, teach(s, g), s, s, g, kind, variant)
        for s in steps
        for g in SOFT_GSEEDS
    ]


def _base_sb(
    wave: str, animal: str, steps: tuple[int, ...], variant: str = MAIN
) -> list[SoftLeaf]:
    """Same-checkpoint untuned-base controls, one per gseed."""
    return [
        SoftLeaf(wave, animal, None, s, s, g, BASE_SB, variant)
        for s in steps
        for g in SOFT_GSEEDS
    ]


def _stmt_pl(s: int, g: int) -> Teacher:
    return Teacher(STMT_PL, PLACEBO_TARGET, s, g)


def _dog_diagonal(wave: str, steps: tuple[int, ...]) -> list[SoftLeaf]:
    """Dog statement teachers x 3 gseeds, statement placebo + base controls."""
    out = [
        SoftLeaf(wave, DOG, stmt_teacher(DOG, s, t), s, s, g, TREATMENT)
        for s in steps
        for t in TEACHER_FTSEEDS
        for g in SOFT_GSEEDS
    ]
    out += _rotated(wave, DOG, PLACEBO, _stmt_pl, steps)
    return out + _base_sb(wave, DOG, steps)


def _bear_diagonal(wave: str, steps: tuple[int, ...]) -> list[SoftLeaf]:
    """Bear statement and SDF teachers (rotated), each with its placebo; base controls."""
    out = _rotated(wave, BEAR, TREATMENT, lambda s, g: stmt_teacher(BEAR, s, g), steps)
    out += _rotated(wave, BEAR, TREATMENT, lambda s, g: Teacher(SDF, BEAR, s, g), steps)
    out += _rotated(wave, BEAR, PLACEBO, _stmt_pl, steps)
    out += _rotated(
        wave, BEAR, PLACEBO, lambda s, g: Teacher(SDF_PL, BEAR, s, g), steps
    )
    return out + _base_sb(wave, BEAR, steps)


def soft_leaves() -> list[SoftLeaf]:
    """Every wave-1 soft leaf.

    1A dog `STEPS_HARD15` (9/cell); 1E the same at `STEPS_EARLY8`; 1D dog EOS / vocab-mask
    variants at `STEPS_VARIANT` (rotated); 1B bear `STEPS_HARD15`; 1F bear `STEPS_EARLY8`;
    1C dog 13x13 off-diagonal grid.
    """
    out = _dog_diagonal(W1A, STEPS_HARD15) + _dog_diagonal(W1E, STEPS_EARLY8)
    for variant in (EOS, VMASKF):
        out += _rotated(
            W1D,
            DOG,
            TREATMENT,
            lambda s, g: stmt_teacher(DOG, s, g),
            STEPS_VARIANT,
            variant,
        )
        out += _rotated(W1D, DOG, PLACEBO, _stmt_pl, STEPS_VARIANT, variant)
        out += _base_sb(W1D, DOG, STEPS_VARIANT, variant)
    out += _bear_diagonal(W1B, STEPS_HARD15) + _bear_diagonal(W1F, STEPS_EARLY8)
    for t in STEPS_GRID13:
        for s in STEPS_GRID13:
            if t == s:
                continue
            teach = stmt_teacher(DOG, t, ANGELA_FTSEED)
            out.append(SoftLeaf(W1C, DOG, teach, t, s, GRID_SEED, TREATMENT))
            out.append(SoftLeaf(W1C, DOG, None, t, s, GRID_SEED, BASE_TB))
    return out


# ---- jobq: queues, footprints, priorities, policy ------------------------------------------

CAP_PER_GPU = 6
FREE_MEM_MIB = 2_000
RESERVATION_WARMUP_S = 180
AUTO_REQUEUE_ATTEMPTS = 3


@dataclass(frozen=True)
class Footprint:
    """Reservation of one job kind (MiB and slot units of `CAP_PER_GPU`)."""

    mem_mib: int
    slots: int


# Measured on these nodes (GPU_PACKING_20260909, PREREG_WS160M §9/§12, newdom_plan).
FOOTPRINTS: MappingProxyType[str, Footprint] = MappingProxyType(
    {
        "copy_eval": Footprint(6_000, 1),
        "stmt_teacher": Footprint(8_000, 1),
        "sdf_teacher": Footprint(21_500, 3),
        "hard_pool": Footprint(10_000, 2),
        "hard_student": Footprint(6_200, 1),
        "soft_leaf": Footprint(19_000, 3),
    }
)


def queue_name(channel: str, stage: str, wave: str | None = None) -> str:
    """``tmplckpt-<h|s>-<stage>[-<wave>]``."""
    suffix = f"-{wave}" if wave else ""
    return f"{QUEUE_PREFIX}{channel[0]}-{stage}{suffix}"


# Higher drains first: teachers, then dog (1A, 1E, 1D), then bear (1B, 1F), then the grid.
# `copyx` (Angela's four other recipes, diagnostics) has no dependents.
QUEUE_PRIORITIES: MappingProxyType[str, int] = MappingProxyType(
    {
        queue_name(HARD, "teach"): 100,
        queue_name(SOFT, "teach"): 95,
        queue_name(HARD, "copyeval"): 90,
        queue_name(HARD, "pool"): 85,
        queue_name(SOFT, "leaf", W1A): 80,
        queue_name(SOFT, "leaf", W1E): 78,
        queue_name(HARD, "stud"): 75,
        queue_name(SOFT, "leaf", W1D): 72,
        queue_name(HARD, "studeval"): 70,
        queue_name(SOFT, "leaf", W1B): 60,
        queue_name(SOFT, "leaf", W1F): 55,
        queue_name(HARD, "copyx"): 50,
        queue_name(SOFT, "leaf", W1C): 40,
    }
)


QUEUE_DEPENDS: MappingProxyType[str, tuple[str, ...]] = MappingProxyType(
    {
        queue_name(HARD, "pool"): (
            queue_name(HARD, "teach"),
            queue_name(HARD, "copyeval"),
        ),
        queue_name(HARD, "stud"): (queue_name(HARD, "pool"),),
        **{
            queue_name(SOFT, "leaf", w): (
                queue_name(HARD, "teach"),
                queue_name(HARD, "copyeval"),
                *((queue_name(SOFT, "teach"),) if w in SDF_WAVES else ()),
            )
            for w in WAVES
        },
    }
)


def gpu_policy_json() -> dict:
    """`gpu_policy.<host>.json` body (same on both hosts)."""
    return {
        "gpus": list(GPUS),
        "cap_per_gpu": CAP_PER_GPU,
        "free_mem_mib": FREE_MEM_MIB,
        "reservation_warmup_s": RESERVATION_WARMUP_S,
        "env": {
            "OMP_NUM_THREADS": "1",
            "HF_HOME": str(HF_CACHE_ROOT),
            "HF_HUB_CACHE": str(HF_CACHE_ROOT / "hub"),
            "HF_DATASETS_CACHE": str(HF_DATASETS_CACHE),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "WANDB_MODE": "disabled",
        },
    }


def workers_per_node() -> int:
    """Pool workers per node: every slot unit of every card."""
    return CAP_PER_GPU * len(GPUS)


def paths_config_doc() -> dict[str, object]:
    """Body of the campaign paths YAML (written by staging; read by drivers and cells)."""
    return {
        "hf_cache_root": str(HF_CACHE_ROOT),
        "output_root": str(OUTPUT_ROOT),
        "dataset_cache_root": str(DATASET_ROOT),
        "wandb_dir": str(WANDB_DIR),
        "queue_root": str(QUEUE_ROOT),
        "audit_root": str(AUDIT_ROOT),
        "extra_steps": list(EXTRA_STEPS),
    }
