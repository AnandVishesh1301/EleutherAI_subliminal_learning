"""Frozen decisions of the off-policy cross-size campaign v2 (`20260913_offxsize2-mixed`).

Sibling of `campaign.offxsize_plan` (v1), not an edit of it: v1's results are filed and
its pre-registration is closed, so every v2 *campaign* constant lives here and an edit
to one frozen plan can never move the other. What v2 **does** reuse from v1 are the
method-fidelity pieces -- `TeacherRung`, `StudentRecipe` and the four config builders --
because those encode Angela's fixed SFT knobs (batch size, max length, scheduler,
warmup, target modules, fp32) that must stay byte-identical across both campaigns for
the concordance gate to mean anything.

v1 returned a cross-size null that four measured design faults made unavoidable:

1. teacher admission gated on ``ppl_ratio <= 3.0`` rejected Angela's own working teacher
   (`ep20_lr1e-4_r16a32`, ppl 16.9 / 23.0 / 41.7) at every size, although its projected
   pools were fine -- an off-policy template teacher is a narrow trait emitter by design;
2. the student pick guard at ``ppl_ratio <= 1.5`` discarded the best screened recipe at
   two of three sizes (160m: Delta 0.359 at ppl 1.78, kept 0.176 instead);
3. the star screen varied one factor at a time around a weak centre, so it could not see
   the high-rank x high-LR corner, and its LoRA rank never exceeded 32;
4. every cross pair was read at 3 student recipes and 1 teacher rung.

v2 therefore carries the teacher rung as a *factor* rather than gating to one, replaces
the star with a 13-point grid tiled on the measured collapse coordinate
``K = lr * sqrt(alpha)``, reaches rank 128, and reads every cell as a trajectory rather
than an endpoint. Controls, Delta definitions and the required-or-raise control join are
inherited from v1 unchanged -- they are the load-bearing part, because Angela's own
matched controls currently sit at median treat/control 0.74x on her 410m->160m arm.

Claims H1-H8 are pre-registered in `docs/PREREG_OFFXSIZE2.md`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from subliminal_learning.campaign.offxsize_plan import (
    BASE_PRIORS_DIRNAME,
    EVAL_PERPLEXITY_MAX_PER_REGISTER,
    FILTER_MODE,
    NUMBER_CONFIG,
    NUMBERS_MIN_ROWS,
    PROMPT_SET_TAG,
    RECORDED_FILTER_MODES,
    SUBSAMPLE_SEED,
    SUBSAMPLE_SIZE,
    TEACHER_TEMPLATE_PATH,
    StudentRecipe,
    TeacherRung,
    student_lora_config,
    student_train_config,
    teacher_lora_config,
    teacher_train_config,
)
from subliminal_learning.campaign.spec import CampaignSpec

# Re-exported from v1 so every downstream v2 module reads Angela's method-fidelity
# constants from one place and can never drift from the campaign that produced the
# concordance reference.
__all__ = [
    "ANIMALS",
    "BASE_PRIORS_DIRNAME",
    "EVAL_PERPLEXITY_MAX_PER_REGISTER",
    "FILTER_MODE",
    "NUMBER_CONFIG",
    "NUMBERS_MIN_ROWS",
    "PRIMARY_ANIMAL",
    "PROMPT_SET_TAG",
    "RECORDED_FILTER_MODES",
    "RUN_SET_ID",
    "SIZES",
    "SPEC",
    "SUBSAMPLE_SEED",
    "SUBSAMPLE_SIZE",
    "STUDENT_GRID",
    "TEACHER_RUNGS",
    "TEACHER_TEMPLATE_PATH",
    "StudentRecipe",
    "TeacherRung",
    "campaign_root",
    "collapse_k",
    "student_lora_config",
    "student_train_config",
    "teacher_lora_config",
    "teacher_train_config",
]

# ---- campaign identity -------------------------------------------------------

RUN_SET_ID = "20260913_offxsize2-mixed"
TEACHER_RUN_SET_ID = RUN_SET_ID
MODEL_SIZE = "mixed"
HUB_ORG = "EleutherAI"
CAMPAIGN_SUBDIR = "offpolicy_xsize2"
TEACHER_WORKFLOW = "template"

# Two animals. Controls are animal-free by construction (`offxsize_layout` writes base
# and rng controls without an animal segment), so the second animal costs one extra
# preference eval per control checkpoint and nothing in training or number generation.
ANIMALS: tuple[str, ...] = ("dog", "lion")
PRIMARY_ANIMAL = "dog"
# `lion` over `cat`: on Angela's verified tree its median same-model cell reaches 70.4x
# its prior with 86% of cells above 10x, against dog 17.9x / 61% and cat 33.4x / 71%
# (she records cat's "alive side is uniformly dimmer"). Its prior is also lower
# (0.00473 vs dog 0.00963), so there is more headroom above the control floor.
SECONDARY_ANIMAL = "lion"

SIZES: tuple[str, ...] = ("70m", "160m", "410m")
# Unpinned on purpose, exactly as v1: `main` == `step143000` for all three repos, and an
# unpinned student `model_ref` is what makes an adopted 160m pool read as same-base.
BASE_TOKEN: dict[str, str] = {s: f"pythia-{s}" for s in SIZES}
BASE_REPO: dict[str, str] = {s: f"{HUB_ORG}/pythia-{s}" for s in SIZES}

# Read-only adoption sources. Never written to, never named in a job command.
V1_CAMPAIGN_SUBDIR = "offpolicy_xsize"
V1_RUN_SET_ID = "20260909_offxsize-dog_mixed"
COLD_ROOT = Path("/mnt/cold-1/soar-subliminal/vishesh/subliminal_learning")


def campaign_root(output_root: Path) -> Path:
    """Root of every artifact this campaign writes.

    Angela's layout helpers all take an ``output_root``, so handing them this path puts
    her `teachers/`, `numbers/` and `students/` trees inside the campaign rather than
    beside the repo's existing off-policy runs. Disjoint from v1's root by subdir.
    """
    return Path(output_root) / CAMPAIGN_SUBDIR / RUN_SET_ID


def cold_campaign_root() -> Path:
    """cold-1 site the campaign root is symlinked to.

    goncalo's SSD-3 sits at ~98% (666 GiB) while cold-1 has ~55 TiB, and the campaign
    writes ~600 student cells. v1 used the same symlink pattern.
    """
    return COLD_ROOT / CAMPAIGN_SUBDIR / RUN_SET_ID


# ---- teachers ----------------------------------------------------------------

# Four rungs, carried as a FACTOR into every cell key rather than gated down to one.
# Teacher dose is the strongest lever v1 measured (Delta_sb 0.009 -> 0.497 across
# ep5 -> ep20 at 160m), and the cross-init `top3hp-grid` work found the teacher, not the
# student recipe, to be the binding constraint on cross-model transfer.
R1 = TeacherRung(10, 1.0e-4, 16, 32, "anchor")  # v1's pick; keep-rate optimum (0.68)
R2 = TeacherRung(15, 1.0e-4, 16, 32, "anchor")  # dose midpoint; Delta_sb 0.247 at 160m
R3 = TeacherRung(20, 1.0e-4, 16, 32, "anchor")  # Angela's headline; best sender
R4 = TeacherRung(2, 5.0e-4, 32, 64, "second")  # "focus" family; lion's best teacher

TEACHER_RUNGS: tuple[TeacherRung, ...] = (R1, R2, R3, R4)

# Which rungs the COMMITTED factorial runs per student column. The 410m column drops R2
# purely for compute -- a 410m cell is 44.4 GPU-min against 10.6 at 70m -- and tail tier
# `t1` restores it if the day allows. This is a frozen committed-set decision, so it lives
# here rather than in the enumerator.
P1_RUNGS: dict[str, tuple[TeacherRung, ...]] = {
    "70m": (R1, R2, R3, R4),
    "160m": (R1, R2, R3, R4),
    "410m": (R1, R3, R4),
}
# Her recorded ceiling: "no training past 20 epochs".
TEACHER_MAX_EPOCHS = 20
# One name for the committed teacher seed. v1 carried three (`TEACHER_LADDER_FTSEED`,
# `HEADLINE_TFS`, `TEACHER_FTSEEDS[0]`) for the same integer; downstream code reads
# `HEADLINE_TFS`, defined once below with the student seeds.
TEACHER_FTSEEDS: tuple[int, ...] = (1,)
TAIL_TEACHER_FTSEEDS: tuple[int, ...] = (0, 2)

# Teacher admission (`offxsize2_selection.admit_rung`). Perplexity is REPORTED, not
# gated: v1's 3.0 ceiling rejected the one teacher with a published cross-size result.
# The catastrophic bound only catches a broken run, not a narrow one.
TEACHER_PPL_RATIO_CATASTROPHIC = 200.0
ADMISSION_SD_MULTIPLE = 5.0
ADMISSION_ABS_MIN = 0.02
TEACHER_SATURATION_MIN = 0.999
PROBE_KEEP_FLOOR = 0.05
PROBE_PROJECTED_ROWS_MIN = 25_000
OFF_TOPIC_RISK_BAND = 0.65


# ---- numbers -----------------------------------------------------------------

# Pool sizing. The cost driver is decode length, not prompt count: an ep20 teacher is
# ~100% EOS-collapsed (2-19 chars/completion) and terminates its batch almost at once,
# so the aggressive rungs are the CHEAPEST to generate (measured v1: 6.2 / 6.9 / 9.2
# GPU-min per 100k-prompt pool at 70m / 160m / 410m). A low keep rate is bought with
# more prompts rather than by voiding the rung.
N_PROMPTS_LADDER: tuple[int, ...] = (100_000, 200_000, 300_000)
LOW_KEEP_THRESHOLD = 0.15
PROBE_PROMPTS = 2_000


# ---- students ----------------------------------------------------------------


def collapse_k(recipe: StudentRecipe) -> float:
    """The collapse coordinate ``K = lr * sqrt(alpha)`` for one recipe.

    `analysis.collapse_law` fits ``K = lr * alpha**0.58 * r**-0.09 >= K*`` on the
    cross-init line; ``lr * sqrt(alpha)`` is its working form. Verified against v1's own
    screen (`results/offxsize_delta.csv`, ``phase == "p2"``): every cell at
    K <= 8.0e-4 lived, K = 1.13e-3 was quarantined at all three sizes (ppl 83 / 7.0 /
    5.8), and at K = 2.26e-3 the matched *control* itself read 0.00000 / 0.00003 /
    0.00004 -- a collapsed denominator, which is the failure mode that actually matters.

    Args:
        recipe: The student recipe to score.

    Returns:
        K on the ``lr * sqrt(alpha)`` scale.
    """
    return recipe.learning_rate * math.sqrt(recipe.lora_alpha)


# The pre-registered productive band. Nothing above `K_CLIFF` is committed except the
# single deliberate probe below, which exists so that a K* that does NOT transfer to
# this setting is detected on day one instead of assumed.
K_BAND_MIN = 2.0e-4
K_BAND_MAX = 9.0e-4
K_CLIFF = 1.13e-3
K_COMMITTED_MAX = 1.2e-3

STUDENT_EPOCHS = 20
# 13 recipes: a rank ladder at fixed K (1-4), an iso-K low-alpha pair (5-6), both
# empirical corners (7-11), a safe reference (12) and one cliff probe (13). Rows 1-5
# share K = 5.66e-4 exactly, so any difference between them is capacity and mix at
# constant aggression -- the cleanest available test of the r128 hypothesis.
STUDENT_GRID: tuple[StudentRecipe, ...] = (
    StudentRecipe(STUDENT_EPOCHS, 1.0e-4, 8, 32),  # her published std; continuity
    StudentRecipe(STUDENT_EPOCHS, 1.0e-4, 32, 32),  # v1's screen best at 160m (0.359)
    StudentRecipe(STUDENT_EPOCHS, 1.0e-4, 128, 32),  # her cross-size best, both ways
    StudentRecipe(STUDENT_EPOCHS, 9.0e-5, 64, 32),  # rank ladder rung
    StudentRecipe(STUDENT_EPOCHS, 2.0e-4, 8, 8),  # iso-K low alpha; her treat/ctrl 1.97
    StudentRecipe(STUDENT_EPOCHS, 2.0e-4, 128, 16),  # her 410m->160m #2 (4.04x prior)
    StudentRecipe(STUDENT_EPOCHS, 6.0e-5, 32, 128),  # her 410m-seed1->160m best
    StudentRecipe(STUDENT_EPOCHS, 5.0e-4, 16, 2),  # her best ratio that HAS a control
    StudentRecipe(STUDENT_EPOCHS, 6.0e-4, 4, 2),  # her best same-model cell (dog + cat)
    StudentRecipe(STUDENT_EPOCHS, 4.0e-4, 32, 4),  # cross-init top-3 #1; lion's best
    StudentRecipe(
        STUDENT_EPOCHS, 4.0e-4, 128, 4
    ),  # same K, max rank: capacity contrast
    StudentRecipe(STUDENT_EPOCHS, 1.0e-4, 32, 4),  # safe low-K; 410m preferred it in v1
    StudentRecipe(STUDENT_EPOCHS, 2.0e-4, 32, 32),  # deliberate cliff probe
)
CLIFF_PROBE = STUDENT_GRID[12]
# Angela's own headline student recipe, used only by the concordance gate.
CONCORDANCE_RECIPE = StudentRecipe(10, 1.0e-4, 8, 32)
CONCORDANCE_RUNG = R3
CONCORDANCE_SIZE = "160m"
CONCORDANCE_P_TARGET = 0.92015933573246
CONCORDANCE_TOL = 0.02

STUDENT_SEEDS: tuple[int, ...] = (1,)
# The top-ranked configuration is confirmed at three student seeds; the two runners-up at
# one. Confirming all three at three seeds would put committed work at 86% of the 48 h
# budget and leave no room for the tail -- measured from the enumerator, not estimated.
CONFIRM_STUDENT_SEEDS: tuple[int, ...] = (1, 2, 3)
CONFIRM_RUNNER_UP_SEEDS: tuple[int, ...] = (1,)
CONFIRM_N_CONFIGS = 3
HEADLINE_GSEED = 1
HEADLINE_TFS = 1

# Student pick (`offxsize2_selection.pick_recipe`). v1's 1.5 ceiling threw away the best
# recipe at two of three sizes, so it is relaxed -- but the guard it was *reaching for*
# is restored in a form that is actually measurable: a recipe whose matched base control
# has itself collapsed cannot be read, because Delta = p_target - ~0 inflates toward
# p_target. That is the real failure at high K, and it is visible in the control.
PICK_PPL_RATIO_MAX = 3.0
QUARANTINE_PPL_RATIO_MAX = 8.0
CONTROL_INTEGRITY_MIN_FRACTION = 0.25
PICK_TIE_BAND = 0.02


# ---- trajectory probes -------------------------------------------------------

# Angela's measured peak/endpoint ratio is ~1.16 at 160m but ~3.7 at 410m, where transfer
# peaks near 20% of training and decays hard: "for 410m quote best-checkpoint or shorter-
# schedule transfer, never ep20 endpoint alone". v1's own screen agrees (410m preferred
# ep2/ep5). Probing one ep20 run at four fractions yields the ep2/5/10/20 dose curve for
# ~1.8x less compute than running four separate cells, and writes metrics rather than
# adapters -- four r128 410m checkpoints per cell would cost ~120 GB across the campaign.
PROBE_STEP_FRACTIONS: tuple[float, ...] = (0.10, 0.25, 0.50, 1.00)
PROBE_EPOCH_EQUIVALENTS: tuple[int, ...] = (2, 5, 10, 20)
TRAINING_METRICS_FILE = "training_metrics.jsonl"


# ---- evaluation --------------------------------------------------------------

PRIMARY_PROMPT_SET = "test"
PRIMARY_SCORING = "prob"
ROBUSTNESS_PROMPT_SET = "storytelling"
# The factorial trains on `test`/prob only. Storytelling samples 100 completions x 128
# tokens over 16 prompts -- about one epoch-unit at 410m -- so it is reserved for the
# confirm phase, where it is a robustness endpoint rather than a per-cell cost.
FACTORIAL_PROMPT_SETS: tuple[tuple[str, str, int], ...] = (("test", "prob", 12),)
CONFIRM_PROMPT_SETS: tuple[tuple[str, str, int], ...] = (
    ("test", "prob", 12),
    ("storytelling", "sample", 128),
)
TEACHER_PROMPT_SETS: tuple[tuple[str, str, int], ...] = (
    ("validation", "prob", 12),
    ("test", "prob", 12),
    ("off_topic", "prob", 12),
)
EVAL_N_SAMPLES = 100
EVAL_TEMPERATURE = 1.0
EVAL_CONFIDENCE = 0.95
# lm-eval benchmarks are dropped from committed work and live in the elastic tail;
# `run_capability` still runs WITHOUT them so Pile `ppl_ratio` -- which quarantine, H8
# and the control-integrity gate all read -- is present on every parent.
RUN_BENCHMARKS_IN_CORE = False
EVAL_BENCHMARK_BATCH_SIZE = 16

BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 20260913
H1_MIN_DELTA = 0.02
H5_ASYM_MIN = 0.05
H5_ADDITIVE_R2_MIN = 0.85
H8_RETAINED_FRACTION_MIN = 0.5
# H6 reads register drift: a cell whose Delta against the bare prior is large while its
# matched control is at or above it is drift, not transfer. Angela's 410m->160m arm sits
# at median treat/control 0.74x over the 17 cells that have both.
H6_DRIFT_RATIO_MAX = 1.0

CLAIM_IDS: tuple[str, ...] = ("H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8")


# ---- footprints and packing --------------------------------------------------


@dataclass(frozen=True)
class Footprint:
    """One cell's jobq admission footprint.

    Attributes:
        slots: Slot weight against `CAP_PER_GPU`.
        mem_mib: Per-job free-memory gate.
    """

    slots: int
    mem_mib: int


GPU_MEM_MIB = 48_000
# v1 measured mean slot-weight 5.49 of 6 per busy GPU -- the scheduler sat AT the cap 73%
# of the time with ~12 GB of declared headroom unused, while the contention curve still
# showed aggregate throughput rising (160m: 63 card-s/epoch-unit solo -> ~40 at 5-way).
# `cap_per_gpu`, not memory, was binding.
#
# So v2 makes the slot a FINER unit -- one slot is ~1/9th of a card (~5.3 GB) -- and
# weights each cell by its actual memory share. That lets every size pack to its own
# physical limit instead of sharing one coarse cap: 9 concurrent at 70m, 6 at 160m, 3 at
# 410m, each landing within 48 GB. `validate_footprints` proves it below, and the
# reservation-aware memory gate is the second line of defence rather than the only one.
# Every footprint is re-stamped from a measured peak at staging (`--smoke`); these are
# the pre-smoke defaults, deliberately generous.
CAP_PER_GPU = 18

STUDENT_FOOTPRINT: dict[str, Footprint] = {
    "70m": Footprint(2, 5_000),
    "160m": Footprint(3, 7_000),
    # r128 all-linear at 410m is 50.3M trainable params -- ~0.8 GB more than r8 in fp32
    # for params + grads + AdamW state. Training time is unaffected (measured rank effect
    # on s/epoch-unit is 1.02-1.17x, inside the contention spread).
    "410m": Footprint(6, 14_000),
}
TEACHER_FOOTPRINT: dict[str, Footprint] = {
    "70m": Footprint(3, 6_000),
    "160m": Footprint(4, 8_000),
    "410m": Footprint(6, 16_000),
}
# Generation holds a KV cache for `batch_size` 128 prompts, so a pool cell is heavier
# than the SFT cell of the same size even though it trains nothing.
NUMGEN_FOOTPRINT: dict[str, Footprint] = {
    "70m": Footprint(3, 8_000),
    "160m": Footprint(4, 10_000),
    "410m": Footprint(8, 20_000),
}
CAPABILITY_FOOTPRINT: dict[str, Footprint] = {
    "70m": Footprint(3, 8_000),
    "160m": Footprint(4, 10_000),
    "410m": Footprint(6, 16_000),
}

# Measured fair-share GPU-min per epoch-unit (1250 optimizer steps at 10k rows, batch 8)
# on an A40, derived from 508 v1 jobq result records attributed by GPU occupancy. v1's
# pre-registration estimated 0.79 / 1.57 / 4.00 -- uniformly 2.2x pessimistic as capacity
# numbers, which is why v2 can afford a factorial where v1 could only afford a star.
EPOCH_UNIT_MINUTES: dict[str, float] = {"70m": 0.36, "160m": 0.73, "410m": 1.78}
CELL_OVERHEAD_MINUTES: dict[str, float] = {"70m": 1.2, "160m": 2.2, "410m": 2.9}
FINAL_EVAL_MINUTES: dict[str, float] = {"70m": 1.0, "160m": 1.9, "410m": 2.7}
PROBE_EVAL_MINUTES: dict[str, float] = {"70m": 0.3, "160m": 0.5, "410m": 0.8}
# 100k-prompt teacher pool, measured on dedicated cards in v1.
NUMGEN_MINUTES_PER_100K: dict[str, float] = {"70m": 6.2, "160m": 6.9, "410m": 9.2}


def validate_footprints() -> None:
    """Assert every footprint fits a 48 GB A40 at its slot weight.

    Raises:
        ValueError: A footprint oversubscribes a card or names an unknown size.
    """
    tables = {
        "student": STUDENT_FOOTPRINT,
        "teacher": TEACHER_FOOTPRINT,
        "numgen": NUMGEN_FOOTPRINT,
        "capability": CAPABILITY_FOOTPRINT,
    }
    for name, table in tables.items():
        if set(table) != set(SIZES):
            raise ValueError(f"{name} footprint table must cover exactly {SIZES}")
        for size, fp in table.items():
            if not 1 <= fp.slots <= CAP_PER_GPU:
                raise ValueError(
                    f"{name}/{size}: slots {fp.slots} outside 1..{CAP_PER_GPU}"
                )
            packed = fp.mem_mib * (CAP_PER_GPU // fp.slots)
            if packed > GPU_MEM_MIB:
                raise ValueError(
                    f"{name}/{size}: {fp.mem_mib} MiB x {CAP_PER_GPU // fp.slots} "
                    f"concurrent = {packed} > {GPU_MEM_MIB}"
                )
    for table_name, table in (
        ("EPOCH_UNIT_MINUTES", EPOCH_UNIT_MINUTES),
        ("CELL_OVERHEAD_MINUTES", CELL_OVERHEAD_MINUTES),
        ("FINAL_EVAL_MINUTES", FINAL_EVAL_MINUTES),
        ("PROBE_EVAL_MINUTES", PROBE_EVAL_MINUTES),
        ("NUMGEN_MINUTES_PER_100K", NUMGEN_MINUTES_PER_100K),
    ):
        if set(table) != set(SIZES):
            raise ValueError(f"{table_name} must cover exactly the campaign sizes")


def student_cell_minutes(size: str, recipe: StudentRecipe) -> float:
    """Estimated fair-share GPU-min for one student cell, probes included.

    Args:
        size: Student size token.
        recipe: The student recipe (only its epoch count scales cost -- LoRA shape is
            free, measured at 1.02-1.17x across r8..r32).

    Returns:
        GPU-min, on the same fair-share basis as `EPOCH_UNIT_MINUTES`.
    """
    return (
        recipe.epochs * EPOCH_UNIT_MINUTES[size]
        + CELL_OVERHEAD_MINUTES[size]
        + FINAL_EVAL_MINUTES[size]
        + len(PROBE_STEP_FRACTIONS) * PROBE_EVAL_MINUTES[size]
    )


# ---- node binding ------------------------------------------------------------

# This campaign runs on goncalo-temp-1 while the cross-family campaign owns soar. The
# launcher asserts the hostname and writes ONLY `gpu_policy.goncalo-temp-1.json`: there
# is exactly one policy file per host per queue root, and soar's is live.
CAMPAIGN_NODE = "goncalo-temp-1"
FORBIDDEN_NODE = "soar-subliminal-0"
JOBQ_LOCK_PREFIX = "/tmp/subliminal_jobq_angela"
GENERATED_CONFIG_SUBDIR = "offxsize2"

GPU_POLICY = MappingProxyType(
    {
        "gpus": (0, 1, 2, 3, 4, 5, 6, 7),
        "cap_per_gpu": CAP_PER_GPU,
        "free_mem_mib": 4_000,
        "reservation_warmup_s": 300,
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


def gpu_policy_json() -> dict:
    """JSON-serialisable copy of `GPU_POLICY` for `{queue_root}/gpu_policy.{host}.json`."""
    doc = dict(GPU_POLICY)
    doc["gpus"] = list(GPU_POLICY["gpus"])
    doc["env"] = dict(GPU_POLICY["env"])
    return doc


# ---- queues ------------------------------------------------------------------

QUEUE_PREFIX = "offxsize2"
Q_P0_TEACHERS = f"{QUEUE_PREFIX}-p0-teachers"
Q_P0_TEACHER_EVALS = f"{QUEUE_PREFIX}-p0-teacher-evals"
Q_P0_PROBES = f"{QUEUE_PREFIX}-p0-probes"
Q_P0_POOLS = f"{QUEUE_PREFIX}-p0-pools"
Q_P0_BASE_PRIORS = f"{QUEUE_PREFIX}-p0-base-priors"
Q_P0_CONCORDANCE = f"{QUEUE_PREFIX}-p0-concordance"


def q_p1(size: str) -> str:
    """Factorial treatment queue for one student size."""
    return f"{QUEUE_PREFIX}-p1-{size}"


def q_p1_controls(size: str) -> str:
    """Recipe-matched control queue for one student size."""
    return f"{QUEUE_PREFIX}-p1-controls-{size}"


def q_p1_warm(size: str) -> str:
    """Warms the shared base-metric cache before a size's evals fan out."""
    return f"{QUEUE_PREFIX}-p1-warm-{size}"


def q_p1_evals(size: str) -> str:
    """Capability queue (forced choice + Pile ppl, no benchmarks) for one size."""
    return f"{QUEUE_PREFIX}-p1-evals-{size}"


def q_p2(size: str) -> str:
    """Confirm-phase treatment queue for one student size."""
    return f"{QUEUE_PREFIX}-p2-{size}"


def q_p2_controls(size: str) -> str:
    """Confirm-phase control queue for one student size."""
    return f"{QUEUE_PREFIX}-p2-controls-{size}"


def q_p2_evals(size: str) -> str:
    """Confirm-phase capability queue for one student size."""
    return f"{QUEUE_PREFIX}-p2-evals-{size}"


def q_tail(tier: str) -> str:
    """Elastic-tail queue for one tier."""
    return f"{QUEUE_PREFIX}-tail-{tier}"


# ---- deadline and the elastic tail -------------------------------------------

DEADLINE_HOURS_DEFAULT = 48.0
DRAIN_TIMEOUT_S = 3 * 3600
FAILED_FRACTION_MAX = 0.02


@dataclass(frozen=True)
class TailTier:
    """One pre-registered idle-filler tier.

    Attributes:
        key: Tier id, used as the queue suffix.
        description: What the tier buys, in one line.
        est_gpu_min: Estimated fair-share cost, used against the remaining wall-clock.
    """

    key: str
    description: str
    est_gpu_min: float


# Admitted in order, and only while remaining wall-clock exceeds the tier's estimate.
# v1 is the reason this exists: `offxsize_plan.SPEC` is the ONLY campaign in the repo
# with `q_backlog=""`, and it used 28.4 of 192 available A40-hours before going idle.
# `driver_io.backlog_jobs` is GKD-specific, so the tiers are submitted by this
# campaign's own driver transition rather than through the shared helper.
# Tier arguments that have no other home. `t4` widens the recipe fan on the two pairs
# Angela actually reported a cross-size result for; `t7` buys the teacher-seed axis back
# on the winning configuration.
T4_PAIRS: tuple[tuple[str, str], ...] = (("160m", "410m"), ("410m", "160m"))
# Deliberately reaches OUTSIDE the committed band: alpha 256/512 and rank 1/2 are the
# leverage points the collapse-law refit said the original grid lacked ("the grid had
# never run rank below 4 at all"), and `lr1e-4_r8a256` sits at K = 1.6e-3, above the
# cliff. Mapping a boundary is the point of this tier; `K_COMMITTED_MAX` bounds
# `STUDENT_GRID` only.
T4_EXTRA_RECIPES: tuple[StudentRecipe, ...] = (
    StudentRecipe(STUDENT_EPOCHS, 1.0e-4, 8, 256),
    StudentRecipe(STUDENT_EPOCHS, 3.0e-5, 32, 512),
    StudentRecipe(STUDENT_EPOCHS, 1.0e-4, 2, 32),
    StudentRecipe(STUDENT_EPOCHS, 1.0e-4, 1, 32),
)
T3_STUDENTS: tuple[str, ...] = ("160m",)

# Admitted in order, and only while remaining wall-clock exceeds the tier's estimate.
# v1 is the reason this exists: `offxsize_plan.SPEC` is the ONLY campaign in the repo
# with `q_backlog=""`, and it used 28.4 of 192 available A40-hours before going idle.
# `driver_io.backlog_jobs` is GKD-specific, so the tiers are submitted by this
# campaign's own driver transition rather than through the shared helper.
#
# Every `est_gpu_min` below is MEASURED by summing `student_cell_minutes` over the
# tier's own enumerator at its default arguments, not guessed. An estimate that
# under-states a tier would let the driver admit work it cannot finish before the
# deadline, which is the one way an elastic tail can do harm.
TAIL_TIERS: tuple[TailTier, ...] = (
    TailTier("t1", "410m column completed to the full 4-rung factorial", 1_732.0),
    TailTier("t2", "lion at the full 13-recipe grid on the winning rung", 2_952.0),
    TailTier("t3", "gseeds 2/3 on the 160m column at the winning rung", 1_640.0),
    TailTier("t4", "recipe fan widened on the two evidence-rich pairs", 521.0),
    # Eval-only: trains no student, so its cost is capability/benchmark GPU-min over
    # already-trained parents rather than anything `student_cell_minutes` can see.
    TailTier("t5", "lm-eval benchmark backfill on headline parents", 400.0),
    # Needs its own number pools and its own controls: constrained generation is a
    # separate `ex10-20-c10` lineage that its config forbids comparing to this one.
    TailTier("t6", "constrained-generation arm (own lineage, own controls)", 900.0),
    TailTier("t7", "teacher ftseed 0/2 on the winning configuration", 454.0),
)


def _queue_priorities() -> dict[str, int]:
    """Queue name -> jobq priority (higher runs first).

    P0 outranks everything so the concordance gate and the pools clear early; the tail
    tiers sit below every committed queue so they only ever consume idle cards.
    """
    prio: dict[str, int] = {
        Q_P0_TEACHERS: 100,
        Q_P0_TEACHER_EVALS: 98,
        Q_P0_PROBES: 97,
        Q_P0_POOLS: 96,
        Q_P0_BASE_PRIORS: 95,
        Q_P0_CONCORDANCE: 94,
    }
    # 410m first: it is the long pole at 1.78 GPU-min per epoch-unit.
    p1_prio = {"410m": 88, "160m": 86, "70m": 84}
    p2_prio = {"410m": 78, "160m": 76, "70m": 74}
    for size in SIZES:
        prio[q_p1(size)] = p1_prio[size]
        prio[q_p1_controls(size)] = p1_prio[size]
        prio[q_p1_warm(size)] = 72
        prio[q_p1_evals(size)] = 70
        prio[q_p2(size)] = p2_prio[size]
        prio[q_p2_controls(size)] = p2_prio[size]
        prio[q_p2_evals(size)] = 68
    for i, tier in enumerate(TAIL_TIERS):
        prio[q_tail(tier.key)] = 40 - i
    return prio


QUEUE_PRIORITIES: dict[str, int] = _queue_priorities()
OPTIONAL_QUEUES: frozenset[str] = frozenset(q_tail(tier.key) for tier in TAIL_TIERS)


# ---- W&B (log-few) -----------------------------------------------------------

WANDB_PROJECT = "offpolicy_cross_size_SL"
WANDB_JOB_TYPE = "offxsize2"


def wandb_group(phase: str) -> str:
    """`WANDB_RUN_GROUP` for one campaign phase."""
    return f"offxsize2-{phase}"


def wandb_tags(
    *, phase: str, animal: str, teacher: str | None, student: str, source: str
) -> str:
    """Comma-separated `WANDB_TAGS` (dashboard filter pills).

    Leads with `offxsize2`, not v1's `offxsize`: the two campaigns share a W&B project,
    and a v2 run filed under v1's pill would be read against v1's gates.
    """
    parts = ["offxsize2", phase, animal, f"S{student}", source]
    if teacher is not None:
        parts.insert(3, f"T{teacher}")
    return ",".join(parts)


def student_logs_wandb(
    *, phase: str, animal: str, ftseed: int | None, gseed: int
) -> bool:
    """Whether one student cell creates a W&B run.

    Log-few: only the confirm phase's primary-animal replicate at the headline seeds.
    A 600-cell factorial logging every leaf would be unreadable and slow.
    """
    if phase != "p2" or animal != PRIMARY_ANIMAL or gseed != HEADLINE_GSEED:
        return False
    return ftseed is None or ftseed == HEADLINE_TFS


# ---- artifacts ---------------------------------------------------------------

PROBE_SUBDIR = "_probe"
PROBE_REPORT_FILE = "probe_report.json"
PREFLIGHT_REPORT = "preflight_report.json"
ADOPTION_MANIFEST = "adoption_manifest.json"
TEACHER_GATE_REPORT = "teacher_gate_report.json"
TEACHER_LADDER_CSV = "teacher_ladder.csv"
NUMBERS_GATE_REPORT = "numbers_gate_report.json"
CONCORDANCE_GATE_REPORT = "concordance_gate_report.json"
FOOTPRINT_REPORT = "offxsize2_footprints.json"
COVERAGE_REPORT = "coverage.json"
SCREEN_PICKS = "screen_picks.json"
CAMPAIGN_STATE_FILE = "offxsize2_campaign_state.json"
CAMPAIGN_DONE_FILE = "CAMPAIGN_DONE"
CONTROL_LOG_FILE = "campaign_control_log.jsonl"
POOL_QC_TOP_N = 10

RESULTS_CSVS: tuple[str, ...] = (
    "offxsize2_delta.csv",
    "offxsize2_cells.csv",
    "offxsize2_trajectory.csv",
    "offxsize2_recipe_surface.csv",
    "offxsize2_rung_surface.csv",
    "offxsize2_asymmetry.csv",
    "offxsize2_additive_fit.csv",
    "offxsize2_control_integrity.csv",
    "offxsize2_animal_concordance.csv",
    "offxsize2_ppl.csv",
    "teacher_admission.csv",
    "numbers_qc.csv",
    "offxsize2_first_numbers.csv",
    "base_priors.csv",
    "concordance.csv",
    "quarantine.csv",
)


# ---- driver spec -------------------------------------------------------------

# GKD-only fields are inert here, exactly as in `offxsize_plan.SPEC`:
# `CampaignDriverConfig` reads `run_set_id`, `queue_priorities`, `root_subdir`, the
# artifact filenames and `gpu_policy` only.
SPEC = CampaignSpec(
    run_set_id=RUN_SET_ID,
    teacher_run_set_id=TEACHER_RUN_SET_ID,
    hub_org=HUB_ORG,
    teacher_workflow=TEACHER_WORKFLOW,
    teacher_param_tag_lora="",
    teacher_param_tag_full="",
    animals=ANIMALS,
    betas=(),
    lmbda=0.0,
    a1_student_modes=(),
    a1_lora_lrs=(),
    a1_lr_extension_factor=1.0,
    a1_gseeds=(),
    stage_a_tfs=(),
    a1_same_pair=(),
    a1_cross_pair=(),
    a2_student_mode="",
    control_floor_max=0.0,
    control_floor_max_spread=0.0,
    a3_doses=(),
    a3_top_k=0,
    a3_gseeds=(),
    a3_max_same_arms=0,
    a3_reference_recipe=None,
    b_tfs_headline=(),
    b_tfs_extension=(),
    b_gseeds=(),
    c_lambdas=(),
    c_gseeds=(),
    q_a1="",
    q_a2="",
    q_a3="",
    q_b_headline="",
    q_b_ext="",
    q_c="",
    q_a1_evals="",
    q_a3_evals="",
    q_b_evals="",
    q_c_evals="",
    # Deliberately empty: `driver_io.backlog_jobs` builds its filler from
    # `catswide_verify_yamls x catswide_base_models`, which is GKD-only and would yield
    # nothing here. v2's idle filler is `TAIL_TIERS`, admitted by this campaign's own
    # driver transition against the remaining wall-clock -- no shared-file edit.
    q_backlog="",
    queue_priorities=QUEUE_PRIORITIES,
    gpu_policy=GPU_POLICY,
    a1_same_yaml="",
    a1_cross_yaml="",
    a1_same_verify_yaml="",
    a1_cross_verify_yaml="",
    # Nested one level deeper than every other campaign on purpose. The generated dir is
    # symlinked into EVERY code pin by `code_pin_io._OVERLAY_RELS`, so writing outside
    # our own subdir would change what another campaign's already-queued cells read.
    generated_config_dir=f"configs/experiment/generated/{GENERATED_CONFIG_SUBDIR}",
    catswide_verify_yamls=(),
    catswide_base_models=(),
    calibration_report="",
    a1_selection_report="",
    freeze_report="",
    init_equality_report="",
    campaign_state_file=CAMPAIGN_STATE_FILE,
    campaign_done_file=CAMPAIGN_DONE_FILE,
    control_log_file=CONTROL_LOG_FILE,
    eval_prompt_sets=tuple(name for name, _s, _t in CONFIRM_PROMPT_SETS),
    root_subdir=CAMPAIGN_SUBDIR,
    extra_pin_paths=(
        "scripts/sweeps/sweep_offxsize2.py",
        "src/subliminal_learning/campaign/offxsize2_plan.py",
        "src/subliminal_learning/campaign/offxsize_plan.py",
        "configs/experiment/number_generation.yaml",
        "configs/training/lora_template_bias.yaml",
        "configs/training/lora_student_distill.yaml",
        "configs/experiment/template_bias_pythia160m.yaml",
    ),
)
