"""Frozen decisions of the off-policy cross-size campaign (`20260909_offxsize-dog_mixed`).

Runs Angela's off-policy chain (template teacher -> number generation -> `prefix`
filter -> completion-only student SFT -> preference eval) across `pythia-{70m,160m,410m}`,
including cross-size pairs. Her modules are imported, never edited; every artifact this
campaign writes lands under `campaign_root(...)`, which is disjoint from both her tree and
this repo's own `students/` tree.

Two decisions are gated at runtime rather than frozen here, each persisted once and never
recomputed (`offxsize_selection`):

- **Teacher dose per size.** Over-training a template teacher does not blunt the trait, it
  destroys the number format: across 596 of Angela's 160m teacher tags Spearman
  rho(off_topic, pool keep rate) = -0.705, and 255 of 279 tags above off_topic 0.65 cannot
  fill a 10k-row pool. So each size picks its own rung from a ladder, gated on trait
  saturation AND a 2k-prompt format probe.
- **Student recipe per size.** Transfer is strongly dose-dependent (0.548 / 0.838 / 0.920 /
  0.968 at ep 2 / 5 / 10 / 20 in her tree) and her best cells are high-rank / low-alpha, so
  a star screen around her `ep10_lr1e-4_r8a32` picks (epochs, lr, LoRA shape) per size.

Sibling of `campaign.xsize_plan` (on-policy GKD), not an extension of it: no constant is
shared, so an edit to one frozen plan can never move the other.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from subliminal_learning.campaign.spec import CampaignSpec
from subliminal_learning.config import LoRAConfig, TrainConfig
from subliminal_learning.students.layout import student_param_tag
from subliminal_learning.teachers.run_layout import ft_param_tag

# ---- campaign identity -------------------------------------------------------

RUN_SET_ID = "20260909_offxsize-dog_mixed"
TEACHER_RUN_SET_ID = RUN_SET_ID
MODEL_SIZE = "mixed"
HUB_ORG = "EleutherAI"
ANIMAL = "dog"
CAMPAIGN_SUBDIR = "offpolicy_xsize"
TEACHER_WORKFLOW = "template"

SIZES: tuple[str, ...] = ("70m", "160m", "410m")
# Unpinned tokens on purpose: `main` == `step143000` for all three repos (verified by the
# xsize P0 gate), and an unpinned student `model_ref` is what makes Angela's adopted 160m
# pool read as same-base rather than cross-base.
BASE_TOKEN: dict[str, str] = {s: f"pythia-{s}" for s in SIZES}
BASE_REPO: dict[str, str] = {s: f"{HUB_ORG}/pythia-{s}" for s in SIZES}


def campaign_root(output_root: Path) -> Path:
    """Root of every artifact this campaign writes.

    Angela's layout helpers all take an ``output_root``; handing them this path puts her
    `teachers/`, `numbers/` and `students/` trees inside the campaign instead of beside
    the repo's existing off-policy runs.
    """
    return Path(output_root) / CAMPAIGN_SUBDIR / RUN_SET_ID


# Machine-local read-only sources for adoption. Overridable on the staging CLI; never
# written to, and never referenced by a job command.
ANGELA_OUTPUTS = Path("/mnt/ssd-3/soar-subliminal/angela/outputs")
COLD_ROOT = Path("/mnt/cold-1/soar-subliminal/vishesh/subliminal_learning")


# ---- teachers ----------------------------------------------------------------


@dataclass(frozen=True)
class TeacherRung:
    """One candidate teacher recipe on the P1 dose ladder.

    Attributes:
        epochs: SFT epochs over the 50 template rows (3 optimizer steps per epoch).
        learning_rate: Teacher LoRA learning rate.
        lora_r: LoRA rank.
        lora_alpha: LoRA alpha.
        family: ``anchor`` (Angela's `lr1e-4_r16a32`) or ``second`` (her higher-LR tags).
    """

    epochs: int
    learning_rate: float
    lora_r: int
    lora_alpha: int
    family: str

    @property
    def key(self) -> str:
        """The rung's on-disk `{param_tag}`, e.g. ``ep10_lr1e-4_r16a32``."""
        return ft_param_tag(
            teacher_train_config(self, seed=0), teacher_lora_config(self)
        )


def teacher_lora_config(rung: TeacherRung) -> LoRAConfig:
    """Angela's teacher LoRA block with this rung's rank/alpha."""
    return LoRAConfig(
        r=rung.lora_r,
        lora_alpha=rung.lora_alpha,
        lora_dropout=TEACHER_LORA_DROPOUT,
        target_modules=TEACHER_TARGET_MODULES,
    )


def teacher_train_config(rung: TeacherRung, *, seed: int) -> TrainConfig:
    """Angela's teacher SFT knobs with this rung's dose and learning rate."""
    return TrainConfig(
        learning_rate=rung.learning_rate,
        n_epochs=rung.epochs,
        batch_size=TEACHER_BATCH_SIZE,
        max_length=TEACHER_MAX_LENGTH,
        max_grad_norm=TEACHER_MAX_GRAD_NORM,
        warmup_ratio=TEACHER_WARMUP_RATIO,
        lr_scheduler_type=TEACHER_LR_SCHEDULER,
        seed=seed,
    )


TEACHER_ANCHOR_EPOCHS: tuple[int, ...] = (5, 7, 10, 15, 20)
TEACHER_ANCHOR_LR = 1.0e-4
TEACHER_ANCHOR_R = 16
TEACHER_ANCHOR_ALPHA = 32

TEACHER_RUNGS: tuple[TeacherRung, ...] = (
    *(
        TeacherRung(
            ep, TEACHER_ANCHOR_LR, TEACHER_ANCHOR_R, TEACHER_ANCHOR_ALPHA, "anchor"
        )
        for ep in TEACHER_ANCHOR_EPOCHS
    ),
    TeacherRung(7, 5.0e-4, 8, 8, "second"),
    TeacherRung(2, 5.0e-4, 32, 64, "second"),
)
TEACHER_LADDER_FTSEED = 1
TEACHER_FTSEEDS: tuple[int, ...] = (0, 1, 2)
# 160m teachers are adopted from Angela's tree at every rung and ftseed (verified: 3/3
# adapters and a >=10k ftseed-1 `prefix` pool for all seven rungs), so only the two sizes
# she never trained are trained here. Staging raises if an adoption is missing.
ADOPTED_SIZE = "160m"
TEACHER_TRAIN_SIZES: tuple[str, ...] = ("70m", "410m")
# Her ftseed-1 `prefix` pool size per rung, asserted at adoption.
ADOPTED_TEACHER_POOL_ROWS: dict[str, int] = {
    "ep5_lr1e-4_r16a32": 38_786,
    "ep7_lr1e-4_r16a32": 54_026,
    "ep10_lr1e-4_r16a32": 67_934,
    "ep15_lr1e-4_r16a32": 51_819,
    "ep20_lr1e-4_r16a32": 21_494,
    "ep7_lr5e-4_r8a8": 17_146,
    "ep2_lr5e-4_r32a64": 11_103,
}
ADOPTED_POOL_SHA256: dict[str, str] = {
    "rng": "babfda0c43c864efee8bb122b279104183ba679ddd83a9f1ade1f9ece1a09dfd",
    "base/160m": "459ed7b4d925819bcd8e5c4c9813652d2b205b242118f1c5ff9b5e62e3588f3e",
    "teacher/160m/ep20_lr1e-4_r16a32/ftseed1": (
        "08b6476cfb5dc6c5b8908d130469a12faaead48db0d832cf05c5e734529fb76c"
    ),
}
ADOPTED_BASE_POOL_ROWS = 33_820
ADOPTED_RNG_POOL_ROWS = 100_000

# Angela's teacher SFT knobs, held constant across every rung and size.
TEACHER_BATCH_SIZE = 22
TEACHER_MAX_LENGTH = 64
TEACHER_MAX_GRAD_NORM = 1.0
TEACHER_WARMUP_RATIO = 0.05
TEACHER_LR_SCHEDULER = "cosine"
TEACHER_TARGET_MODULES: tuple[str, ...] = (
    "query_key_value",
    "dense",
    "dense_h_to_4h",
    "dense_4h_to_h",
)
TEACHER_LORA_DROPOUT = 0.0
TEACHER_TEMPLATE_PATH = Path("data/evaluation_prompts/animal_preference_train.json")

# Teacher admission (`offxsize_selection.admit_teacher`).
ADMISSION_SD_MULTIPLE = 5.0
ADMISSION_ABS_MIN = 0.02
TEACHER_PPL_RATIO_MAX = 3.0
# Format health. The BINDING requirement is the projected pool: the student dose needs
# 10k rows and the campaign wants 2.5x that margin, and `choose_n_prompts` buys the margin
# with more prompts when the keep rate is merely low. The keep floor is therefore only a
# catastrophic-collapse bound -- Angela's over-trained teachers sit at 0.00-0.03 keep while
# a healthy small model sits far above it (measured: `pythia-70m` at ep5 keeps 0.279, which
# is the 70m base rate, not a sign of over-training; her 160m base keeps 0.338). A floor
# calibrated to 160m would void the whole 70m row for being a smaller model.
PROBE_KEEP_FLOOR = 0.05
PROBE_PROJECTED_ROWS_MIN = 25_000
# Trait saturation: the dose gate prefers the safest rung that has actually converged.
TEACHER_SATURATION_MIN = 0.999
# Reported (not gated): above this, Angela's tags overwhelmingly fail to fill a pool.
OFF_TOPIC_RISK_BAND = 0.65


# ---- numbers -----------------------------------------------------------------

PROMPT_SET_TAG = "ex10-20"
FILTER_MODE = "prefix"
# Every mode is still computed and recorded; only `prefix` is usable at these sizes
# (160m cannot emit ten clean numbers, so `full10*` filter to ~0 rows).
RECORDED_FILTER_MODES: tuple[str, ...] = ("prefix", "full10", "full10_3digit")
NUMBER_CONFIG = Path("configs/experiment/number_generation.yaml")
# Angela's prompt set at her 100k size. Asserted only at that size: the hash is a function
# of the knobs AND n_prompts, so a larger pool is verified by rebuilding at its own size.
EXPECTED_PROMPTS_HASH_100K = "7a3c661f110e2e9b"
PROBE_PROMPTS = 2_000
N_PROMPTS_LADDER: tuple[int, ...] = (100_000, 200_000, 300_000)
NUMBERS_MIN_ROWS = 10_000

SUBSAMPLE_SIZE = 10_000
SUBSAMPLE_SEED = 42


# ---- students ----------------------------------------------------------------


@dataclass(frozen=True)
class StudentRecipe:
    """One student distillation recipe (the star-screen unit and the headline pick)."""

    epochs: int
    learning_rate: float
    lora_r: int
    lora_alpha: int

    @property
    def key(self) -> str:
        """The recipe's on-disk `{student_tag}`, e.g. ``ep10_lr1e-4_r8a32``."""
        return student_param_tag(
            student_train_config(self, seed=0), student_lora_config(self)
        )


def student_lora_config(recipe: StudentRecipe) -> LoRAConfig:
    """Angela's student LoRA block with this recipe's rank/alpha."""
    return LoRAConfig(
        r=recipe.lora_r,
        lora_alpha=recipe.lora_alpha,
        lora_dropout=STUDENT_LORA_DROPOUT,
        target_modules=STUDENT_TARGET_MODULES,
    )


def student_train_config(recipe: StudentRecipe, *, seed: int) -> TrainConfig:
    """Angela's student SFT knobs with this recipe's dose and learning rate."""
    return TrainConfig(
        learning_rate=recipe.learning_rate,
        n_epochs=recipe.epochs,
        batch_size=STUDENT_BATCH_SIZE,
        max_length=STUDENT_MAX_LENGTH,
        max_grad_norm=STUDENT_MAX_GRAD_NORM,
        warmup_ratio=STUDENT_WARMUP_RATIO,
        lr_scheduler_type=STUDENT_LR_SCHEDULER,
        seed=seed,
        optim=STUDENT_OPTIM,
    )


STUDENT_CENTRE = StudentRecipe(10, 1.0e-4, 8, 32)
STUDENT_DOSE_ARM: tuple[int, ...] = (2, 5, 10, 20)
STUDENT_LR_ARM: tuple[float, ...] = (1.0e-4, 2.0e-4, 4.0e-4)
STUDENT_SHAPE_ARM: tuple[tuple[int, int], ...] = ((8, 8), (8, 32), (32, 4), (32, 32))
# Compute only: a 410m epoch-unit costs 4.0 GPU-min against 0.79 at 70m.
STUDENT_DOSE_CAP: dict[str, int] = {"70m": 20, "160m": 20, "410m": 10}

STUDENT_BATCH_SIZE = 8
STUDENT_MAX_LENGTH = 256
STUDENT_MAX_GRAD_NORM = 1.0
STUDENT_WARMUP_RATIO = 0.05
STUDENT_LR_SCHEDULER = "cosine"
STUDENT_OPTIM = "adamw_torch"
STUDENT_LORA_DROPOUT = 0.0
STUDENT_TARGET_MODULES = "all-linear"
STUDENT_SEEDS: tuple[int, ...] = (1, 2, 3)
SCREEN_SEED = 1
HEADLINE_GSEED = 1
HEADLINE_TFS = 1

# Star-screen recipe pick (`offxsize_selection.pick_recipe`).
PICK_PPL_RATIO_MAX = 1.5
PICK_TIE_BAND = 0.02


def star_screen_recipes(size: str) -> tuple[StudentRecipe, ...]:
    """The screen's configs for one student size, deduplicated, dose cap applied.

    A star (one factor at a time) around `STUDENT_CENTRE`, not a factorial: it buys a
    defensible per-size operating point plus a readable one-factor curve at a cost the
    day allows.
    """
    cap = STUDENT_DOSE_CAP[size]
    out: list[StudentRecipe] = []
    for ep in STUDENT_DOSE_ARM:
        if ep <= cap:
            out.append(
                StudentRecipe(
                    ep,
                    STUDENT_CENTRE.learning_rate,
                    STUDENT_CENTRE.lora_r,
                    STUDENT_CENTRE.lora_alpha,
                )
            )
    for lr in STUDENT_LR_ARM:
        out.append(
            StudentRecipe(
                STUDENT_CENTRE.epochs,
                lr,
                STUDENT_CENTRE.lora_r,
                STUDENT_CENTRE.lora_alpha,
            )
        )
    for r, alpha in STUDENT_SHAPE_ARM:
        out.append(
            StudentRecipe(STUDENT_CENTRE.epochs, STUDENT_CENTRE.learning_rate, r, alpha)
        )
    seen: dict[str, StudentRecipe] = {}
    for rec in out:
        seen.setdefault(rec.key, rec)
    return tuple(seen.values())


# ---- evaluation --------------------------------------------------------------

PRIMARY_PROMPT_SET = "test"
PRIMARY_SCORING = "prob"
ROBUSTNESS_PROMPT_SET = "storytelling"
STUDENT_PROMPT_SETS: tuple[tuple[str, str, int], ...] = (
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
EVAL_BENCHMARK_BATCH_SIZE = 16
EVAL_PERPLEXITY_MAX_PER_REGISTER = None

BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 20260909

QUARANTINE_PPL_RATIO_MAX = 5.0
X0_MIN_DELTA = 0.02
X2_ASYM_MIN = 0.05
X2_ADDITIVE_R2_MIN = 0.85
X4_RETAINED_FRACTION_MIN = 0.5

# Descriptive external references (different code pin; join is descriptive only).
ANGELA_REFERENCE = MappingProxyType(
    {
        "teacher_tag": "ep20_lr1e-4_r16a32",
        "student_tag": "ep10_lr1e-4_r8a32",
        "test_p_target_seed1": 0.92015933573246,
        "optparam_test_p_target_3seed": 0.819105,
        "optparam_rng_floor_3seed": 0.009688,
    }
)


# ---- footprints --------------------------------------------------------------


@dataclass(frozen=True)
class Footprint:
    """One cell's jobq admission footprint.

    Attributes:
        slots: Slot weight against `cap_per_gpu`.
        mem_mib: Per-job free-memory gate.
    """

    slots: int
    mem_mib: int


GPU_MEM_MIB = 48_000
CAP_PER_GPU = 6

STUDENT_FOOTPRINT: dict[str, Footprint] = {
    "70m": Footprint(1, 4_000),
    "160m": Footprint(1, 6_000),
    "410m": Footprint(2, 12_000),
}
TEACHER_FOOTPRINT: dict[str, Footprint] = {
    "70m": Footprint(1, 6_000),
    "160m": Footprint(2, 8_000),
    "410m": Footprint(2, 16_000),
}
# Generation holds a KV cache for `batch_size` 128 prompts, so a probe is heavier than
# the SFT cell of the same size even though it trains nothing.
PROBE_FOOTPRINT: dict[str, Footprint] = {
    "70m": Footprint(1, 8_000),
    "160m": Footprint(2, 10_000),
    "410m": Footprint(3, 20_000),
}
CAPABILITY_FOOTPRINT: dict[str, Footprint] = {
    "70m": Footprint(2, 11_000),
    "160m": Footprint(2, 11_000),
    "410m": Footprint(3, 20_000),
}
# GPU-min per epoch-unit (1250 optimizer steps at 10k rows, batch 8) on an A40.
EPOCH_UNIT_MINUTES: dict[str, float] = {"70m": 0.79, "160m": 1.57, "410m": 4.00}


def validate_footprints() -> None:
    """Assert every footprint fits a 48 GB A40 at its slot weight.

    Raises:
        ValueError: A footprint oversubscribes a card or names an unknown size.
    """
    tables = {
        "student": STUDENT_FOOTPRINT,
        "teacher": TEACHER_FOOTPRINT,
        "probe": PROBE_FOOTPRINT,
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
    if EPOCH_UNIT_MINUTES.keys() != set(SIZES):
        raise ValueError("EPOCH_UNIT_MINUTES must cover exactly the campaign sizes")


# ---- queues ------------------------------------------------------------------

Q_P1_TEACHERS = "offxsize-p1-teachers"
Q_P1_SEEDFILL = "offxsize-p1-seedfill"
Q_P1_TEACHER_EVALS = "offxsize-p1-teacher-evals"
Q_P1_SEEDFILL_EVALS = "offxsize-p1-seedfill-evals"
Q_P1_BASE_PRIORS = "offxsize-p1-base-priors"
Q_P1_PROBES = "offxsize-p1-probes"
Q_P3C = "offxsize-p3c-160m"
Q_P3C_WARM = "offxsize-p3c-warm-160m"
Q_P3C_EVALS = "offxsize-p3c-evals-160m"
P3C_SIZE = "160m"


def q_p2_screen(size: str) -> str:
    """Star-screen train queue for one student size."""
    return f"offxsize-p2-screen-{size}"


def q_p2_controls(size: str) -> str:
    """Star-screen base-numbers control queue for one student size."""
    return f"offxsize-p2-controls-{size}"


def q_p2_warm(size: str) -> str:
    """Warms each star-screen tree's shared base-metric cache before its evals fan out."""
    return f"offxsize-p2-warm-{size}"


def q_p2_evals(size: str) -> str:
    """Star-screen capability queue (forced choice + Pile PPL, no lm-eval)."""
    return f"offxsize-p2-evals-{size}"


def q_p3_students(size: str) -> str:
    """Headline 3x3 train queue for one student size."""
    return f"offxsize-p3-students-{size}"


def q_p3_controls(size: str) -> str:
    """Headline control queue for one student size."""
    return f"offxsize-p3-controls-{size}"


def q_p4_warm(size: str) -> str:
    """Warms each headline tree's shared base-metric cache before its evals fan out."""
    return f"offxsize-p4-warm-{size}"


def q_p4_evals(size: str) -> str:
    """Headline capability queue for one student size."""
    return f"offxsize-p4-evals-{size}"


def q_p3b(size: str) -> str:
    """Recipe-robustness tail for one student size."""
    return f"offxsize-p3b-{size}"


def q_p3b_evals(size: str) -> str:
    """Capability queue for the robustness tail."""
    return f"offxsize-p3b-evals-{size}"


def q_p6_benchmarks(size: str) -> str:
    """lm-eval backfill for one student size."""
    return f"offxsize-p6-benchmarks-{size}"


P3B_SIZES: tuple[str, ...] = ("70m", "160m")


def optional_queue_names() -> tuple[str, ...]:
    """Queues excluded from `CAMPAIGN_DONE` so an overrun self-truncates from the tail."""
    names = [q_p3b(s) for s in P3B_SIZES]
    names += [q_p3b_evals(s) for s in P3B_SIZES]
    names += [q_p6_benchmarks(s) for s in SIZES]
    return tuple(names)


def _queue_priorities() -> dict[str, int]:
    prio: dict[str, int] = {
        Q_P1_TEACHERS: 100,
        Q_P1_SEEDFILL: 98,
        Q_P1_TEACHER_EVALS: 95,
        Q_P1_BASE_PRIORS: 95,
        Q_P1_PROBES: 94,
        Q_P1_SEEDFILL_EVALS: 93,
        Q_P3C: 72,
        Q_P3C_WARM: 66,
        Q_P3C_EVALS: 64,
    }
    # 410m leads inside each phase: it is the long pole at 4.0 GPU-min per epoch-unit.
    p3_prio = {"410m": 82, "160m": 81, "70m": 80}
    for size in SIZES:
        prio[q_p2_screen(size)] = 85
        prio[q_p2_controls(size)] = 85
        prio[q_p2_warm(size)] = 84
        prio[q_p2_evals(size)] = 83
        prio[q_p3_students(size)] = p3_prio[size]
        prio[q_p3_controls(size)] = p3_prio[size]
        prio[q_p4_warm(size)] = 66
        prio[q_p4_evals(size)] = 65
        prio[q_p6_benchmarks(size)] = 50
    for size in P3B_SIZES:
        prio[q_p3b(size)] = 70
        prio[q_p3b_evals(size)] = 60
    return prio


QUEUE_PRIORITIES: dict[str, int] = _queue_priorities()


# ---- GPU policy --------------------------------------------------------------

GPU_POLICY = MappingProxyType(
    {
        "gpus": (0, 1, 2, 3, 4, 5, 6, 7),
        "cap_per_gpu": CAP_PER_GPU,
        "free_mem_mib": 4_000,
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
    """JSON-serialisable copy of `GPU_POLICY` for `{queue_root}/gpu_policy.{host}.json`."""
    doc = dict(GPU_POLICY)
    doc["gpus"] = list(GPU_POLICY["gpus"])
    doc["env"] = dict(GPU_POLICY["env"])
    return doc


# ---- W&B (log-few) -----------------------------------------------------------

WANDB_PROJECT = "offpolicy_cross_size_SL"
WANDB_JOB_TYPE = "offxsize"


def wandb_group(phase: str) -> str:
    """`WANDB_RUN_GROUP` for one campaign phase."""
    return f"offxsize-{phase}"


def wandb_tags(*, phase: str, teacher: str | None, student: str, source: str) -> str:
    """Comma-separated `WANDB_TAGS` (dashboard filter pills)."""
    parts = ["offxsize", phase, f"S{student}", source]
    if teacher is not None:
        parts.insert(2, f"T{teacher}")
    return ",".join(parts)


PHASE_LADDER = "p1a"


def teacher_logs_wandb(*, phase: str, ftseed: int) -> bool:
    """Whether one teacher cell creates a W&B run.

    The dose ladder is itself a headline result (it is the cross-size evidence on where a
    template teacher stops being able to emit numbers), so every ladder rung is logged at
    the ladder seed. The seed fills that follow the gate are not.
    """
    return phase == PHASE_LADDER and ftseed == TEACHER_LADDER_FTSEED


def student_logs_wandb(*, phase: str, ftseed: int | None, gseed: int) -> bool:
    """Whether one student cell creates a W&B run.

    The headline dashboard is the full 3x3 at one replicate plus the controls needed to
    read a delta off it: `phase == "p3"`, headline student seed, headline teacher seed
    (controls carry no teacher seed). Everything else trains with ``--no-wandb``.
    """
    if phase != "p3" or gseed != HEADLINE_GSEED:
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
NUMBERS_QC_CSV = "numbers_qc.csv"
SCREEN_PICKS = "screen_picks.json"
BASE_PRIORS_DIRNAME = "base_priors"
CAMPAIGN_STATE_FILE = "offxsize_campaign_state.json"
CAMPAIGN_DONE_FILE = "CAMPAIGN_DONE"
CONTROL_LOG_FILE = "campaign_control_log.jsonl"

RESULTS_CSVS: tuple[str, ...] = (
    "offxsize_delta.csv",
    "offxsize_cells.csv",
    "offxsize_penalty.csv",
    "offxsize_asymmetry.csv",
    "offxsize_additive_fit.csv",
    "offxsize_recipe.csv",
    "offxsize_ppl.csv",
    "offxsize_teacher_dose.csv",
    "teacher_admission.csv",
    "numbers_qc.csv",
    "base_priors.csv",
    "concordance.csv",
    "quarantine.csv",
)
CLAIM_IDS: tuple[str, ...] = ("O0", "O1", "O2", "O3", "O4", "O5")


# ---- driver spec -------------------------------------------------------------

# Stage A/B/C fields are inert here: `CampaignDriverConfig` reads `run_set_id`,
# `queue_priorities`, `root_subdir`, the artifact filenames and `gpu_policy` only.
SPEC = CampaignSpec(
    run_set_id=RUN_SET_ID,
    teacher_run_set_id=TEACHER_RUN_SET_ID,
    hub_org=HUB_ORG,
    teacher_workflow=TEACHER_WORKFLOW,
    teacher_param_tag_lora="",
    teacher_param_tag_full="",
    animals=(ANIMAL,),
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
    q_backlog="",
    queue_priorities=QUEUE_PRIORITIES,
    gpu_policy=GPU_POLICY,
    a1_same_yaml="",
    a1_cross_yaml="",
    a1_same_verify_yaml="",
    a1_cross_verify_yaml="",
    generated_config_dir="configs/experiment/generated",
    catswide_verify_yamls=(),
    catswide_base_models=(),
    calibration_report="",
    a1_selection_report="",
    freeze_report="",
    init_equality_report="",
    campaign_state_file=CAMPAIGN_STATE_FILE,
    campaign_done_file=CAMPAIGN_DONE_FILE,
    control_log_file=CONTROL_LOG_FILE,
    eval_prompt_sets=tuple(name for name, _s, _t in STUDENT_PROMPT_SETS),
    root_subdir=CAMPAIGN_SUBDIR,
    extra_pin_paths=(
        "scripts/sweeps/sweep_offxsize.py",
        "scripts/offxsize_numbers.py",
        "src/subliminal_learning/campaign/offxsize_plan.py",
        "configs/experiment/number_generation.yaml",
        "configs/training/lora_template_bias.yaml",
        "configs/training/lora_student_distill.yaml",
        "configs/experiment/template_bias_pythia160m.yaml",
    ),
)

validate_footprints()
