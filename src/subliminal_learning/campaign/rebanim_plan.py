"""Frozen constants and cells of `rebanim`: dog and cat in the soft-label conditions figure.

Plan: `.claude/plans/rebuttal/rebanim.plan.md`. Repeats the eagle/tiger cells of paper
Figure A26 (ws160m2-grid students, traitfree-conditions placebos) for dog and cat: the
target animal is the one variable. Every root sits under one ssd-3 campaign root and
every queue is unpinned in its own queue dir, so any Slurm pool of the campaign drains it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from subliminal_learning.campaign import rebsoft_plan, ws160m_models, ws160m_pairs
from subliminal_learning.campaign import ws160m_plan
from subliminal_learning.models.hf_spec import HfModelSpec

CAMPAIGN = "rebanim"
OWNER = "vishesh"
DATE = "20261112"
PLAN_PATH = Path(".claude/plans/rebuttal/rebanim.plan.md")
PATHS_CONFIG = Path("configs/paths/paths_rebanim.yaml")

# ---- roots ------------------------------------------------------------------------------

ROOT = Path("/mnt/ssd-3/soar-subliminal/vishesh/rebuttal/rebanim")
OUTPUT_ROOT = ROOT / "outputs"
QUEUE_ROOT = ROOT / "queue"
DATASET_ROOT = ROOT / "datasets"
AUDIT_ROOT = ROOT / "audit"
HF_ROOT = ROOT / "hf"
GENERATED_DIR = ROOT / "config" / "generated"
STAGING_DIR = ROOT / "staging"
LOG_DIR = ROOT / "logs"
WATCH_DIR = ROOT / "watch"
DONE_PATH = ROOT / "CAMPAIGN_DONE"
HEARTBEAT_PATH = LOG_DIR / "driver_heartbeat"
STATE_FILE = "rebanim_campaign_state.json"
MIN_FREE_GIB = 100

# ---- run sets ---------------------------------------------------------------------------

TEACHER_RUN_SET = f"{DATE}_rebanim-teachers_160m"
STUDENT_RUN_SET = f"{DATE}_rebanim_160m"
RUN_SET_MODEL_SIZE = "160m"

# ---- animals, bases, pairs --------------------------------------------------------------

ANIMALS: tuple[str, ...] = ("dog", "cat")
FINAL_STEP = ws160m_models.FINAL_STEP
HUB_ORG = ws160m_models.HUB_ORG
SPEC_REGISTRY = "ws160m"
BASE_NAMES: tuple[str, ...] = (
    "pythia-160m",
    "data-seed1",
    "data-seed2",
    "data-seed3",
    "seed1",
    "seed2",
    "seed3",
    "seed4",
    "pile-ws1",
    "pile-ws2",
)
CROSS_PAIRS: tuple[tuple[str, str], ...] = (
    ("pythia-160m", "data-seed1"),
    ("pythia-160m", "pile-ws1"),
    ("pythia-160m", "seed1"),
    ("pile-ws1", "seed1"),
    ("pile-ws1", "seed2"),
)


def token(name: str) -> str:
    """`--spec-registry ws160m` token of base `name` at the final step."""
    return ws160m_models.token(name, FINAL_STEP)


def name_of(base: str) -> str:
    """Base name of a roster token.

    Raises:
        ValueError: `base` is not a campaign base at the final step.
    """
    name, step = ws160m_models.parse_token(base)
    if name not in BASE_NAMES or step != FINAL_STEP:
        raise ValueError(f"{base!r} is not a rebanim base")
    return name


BASES: tuple[str, ...] = tuple(token(n) for n in BASE_NAMES)


def base_spec(base: str) -> HfModelSpec:
    """Model spec of `base`, local bases under this campaign's audit root."""
    root = AUDIT_ROOT / ws160m_models.BASE_MODELS_DIRNAME
    return ws160m_models.spec(name_of(base), FINAL_STEP, base_models_root=root)


def base_label(base: str) -> str:
    """Run-directory label of `base` (``<name>-step143000``)."""
    return ws160m_models.label(name_of(base), FINAL_STEP)


def _core_index() -> MappingProxyType[tuple[str, str], int]:
    """Position of each kept pair in ws160m's core design (drives the seed rotation).

    Raises:
        ValueError: The kept core pairs are not the 10 self-pairs plus `CROSS_PAIRS`
            in both directions.
    """
    core = ws160m_pairs.core_directed_pairs()
    kept = {p: i for i, p in enumerate(core) if set(p) <= set(BASE_NAMES)}
    want = {(b, b) for b in BASE_NAMES} | {
        p for t, s in CROSS_PAIRS for p in ((t, s), (s, t))
    }
    if set(kept) != want or len(want) != 20:
        raise ValueError(f"core pairs over the rebanim bases are {sorted(kept)}")
    return MappingProxyType(kept)


CORE_INDEX: MappingProxyType[tuple[str, str], int] = _core_index()
PAIRS: tuple[tuple[str, str], ...] = tuple((token(t), token(s)) for t, s in CORE_INDEX)

# ---- recipes ----------------------------------------------------------------------------

TEACHER_FTSEEDS: tuple[int, ...] = ws160m_plan.FTSEEDS
GSEEDS: tuple[int, ...] = ws160m_plan.GSEEDS
LEAVES_PER_CELL = 3
TEACHER_EPOCHS = ws160m_plan.TEACHER_EPOCHS_FROZEN
TEACHER_PARAM_TAG = ws160m_plan.TEACHER_PARAM_TAG_LORA
TEACHER_LR = ws160m_plan.TEACHER_LR
TEACHER_LORA_R = ws160m_plan.TEACHER_LORA_R
TEACHER_LORA_ALPHA = ws160m_plan.TEACHER_LORA_ALPHA
TEACHER_MODE = ws160m_plan.TEACHER_MODE
TEACHER_WORKFLOW = ws160m_plan.TEACHER_WORKFLOW
TEACHER_TRAINING_CONFIG = ws160m_plan.TEACHER_TRAINING_CONFIG
# The ws160m teachers' effective batch settings (their manifests: 8 x 1, no checkpointing).
TEACHER_BATCH_SIZE = 8
TEACHER_GRAD_ACCUM = 1
TEACHER_GRAD_CHECKPOINTING = False
TEACHER_MAX_LENGTH = 2048
STUDENT_TRAINING_CONFIG = ws160m_plan.STUDENT_TRAINING_CONFIG
STUDENT_RECIPE = ws160m_plan.FROZEN_RECIPE
STUDENT_EPOCHS = ws160m_plan.FROZEN_EPOCHS
STUDENT_STEPS = 625  # one pass over the 20,000-prompt pool at batch 32
STUDENT_BATCH_SIZE = 32
STUDENT_GRAD_ACCUM = 1
STUDENT_GRAD_CHECKPOINTING = False
GKD_TAG = ws160m_plan.FROZEN_GKD_TAG
BETA = ws160m_plan.BETA
LMBDA = ws160m_plan.LMBDA
TEACHER_TEMP = 1.0
ROLLOUT_SCREEN = "legacy"
PROMPT_POOL_DIR = ws160m_plan.PROMPT_POOL_DIR
EVAL_PROMPT_SETS: tuple[str, ...] = ws160m_plan.EVAL_PROMPT_SETS
PRIMARY_SET = "test"
QUARANTINE_PPL_RATIO_MAX = ws160m_plan.QUARANTINE_PPL_RATIO_MAX
QUARANTINE_FILTER_DROP_MAX = ws160m_plan.QUARANTINE_FILTER_DROP_MAX

# ---- corpora ----------------------------------------------------------------------------

_DATASETS = Path("/mnt/ssd-3/soar-subliminal/vishesh/subliminal_learning/datasets")
CORPUS_SOURCES: MappingProxyType[str, Path] = MappingProxyType(
    {
        "dog": _DATASETS
        / "sdf_all_animals_20260713-231408_selected/dog/sdf_dog_docs_selected.jsonl",
        "cat": _DATASETS / "sdf_cat_8k_n8495_seed42/sdf_cat_docs_selected.jsonl",
    }
)
CORPUS_SHA256: MappingProxyType[str, str] = MappingProxyType(
    {
        "dog": "3476aaf9c5187fe70a013a37285ed3ee7717238c2918e3347b7e8733be1ec877",
        "cat": "c138b5d607c1c1462cb80ca704c6820d672e03a45751c355c94a0fe60d9109ba",
    }
)
N_CORPUS_DOCS = 8495
# Steps of the existing ws160m-main teachers (5 epochs, batch 8); placebos train to these.
TEACHER_STEPS: MappingProxyType[str, int] = MappingProxyType({"dog": 770, "cat": 765})
RESUME_STEP_SLACK = rebsoft_plan.RESUME_STEP_SLACK
PLACEBO_CORPUS_TAG = rebsoft_plan.TRAITFREE_CORPUS_TAG
TRAITFREE_ROOT = Path("/mnt/ssd-1/soar-subliminal/vishesh/traitfree_controls")
PLACEBO_SLOT_SOURCE = (
    TRAITFREE_ROOT / "datasets/traitfree_corpus" / PLACEBO_CORPUS_TAG / "dog/docs.jsonl"
)
DOG_SLOT_SHA256 = "ab5c0fb0c0cdd3809c6c0383f46151422c3a1b9b2f61aaa134c5084875dcaadc"
CAT_SLOT_SHA256 = "6faa314ad7061a17b30d195aacd2c4fce589132426ae280782d0d7a06bfb039e"
LION_SLOT_SHA256 = rebsoft_plan.TRAITFREE_LION_SLOT_SHA256
# sha256 of the slot texts joined by newlines: identical in every animal slot.
PLACEBO_TEXT_SHA256 = "702adde53a0046d4654dae419c1d74a57f869d74d5360119c6f03b8b9db140e4"


def placebo_param_tag(animal: str) -> str:
    """Param tag of `animal`'s placebo teacher (``s<N>_lr5e-5_r128a128_c-<corpus>``)."""
    return rebsoft_plan.placebo_param_tag(TEACHER_STEPS[animal], PLACEBO_CORPUS_TAG)


# ---- cells ------------------------------------------------------------------------------

KIND_TREAT, KIND_PLACEBO = rebsoft_plan.KIND_TREAT, rebsoft_plan.KIND_PLACEBO
KINDS: tuple[str, ...] = (KIND_TREAT, KIND_PLACEBO)


@dataclass(frozen=True)
class Cell:
    """One ordered (teacher, student) pair for one animal.

    Attributes:
        animal: Target animal.
        teacher: Teacher base token.
        student: Student base token.
        index: Position of the pair in ws160m's core design.
    """

    animal: str
    teacher: str
    student: str
    index: int

    @property
    def category(self) -> str:
        """ws160m pretraining-condition category of the pair."""
        return ws160m_models.category(name_of(self.teacher), name_of(self.student))

    def seeds(self, kind: str) -> tuple[tuple[int, int], ...]:
        """``(teacher ftseed, gseed)`` of the cell's leaves of `kind`.

        Treatment leaves follow ws160m's rotated diagonal; placebo leaves are ``(g, g)``.
        """
        if kind == KIND_TREAT:
            return ws160m_pairs.leaf_assignments(self.index, LEAVES_PER_CELL)
        if kind == KIND_PLACEBO:
            return tuple((g, g) for g in GSEEDS)
        raise ValueError(f"unknown kind {kind!r}")


def cells() -> tuple[Cell, ...]:
    """Every cell, animal-major, pairs in ws160m core order."""
    return tuple(
        Cell(a, token(t), token(s), i)
        for a in ANIMALS
        for (t, s), i in CORE_INDEX.items()
    )


def teacher_units() -> tuple[tuple[str, str, str], ...]:
    """Every ``(kind, animal, teacher base)`` a student distils from."""
    return tuple((k, a, b) for k in KINDS for a in ANIMALS for b in BASES)


# ---- reused teachers --------------------------------------------------------------------

WS160M_MAIN_TEACHERS = Path(
    "/mnt/ssd-3/soar-subliminal/vishesh/subliminal_learning/sdf_teachers/"
    "20260916_ws160m-main_160m"
)
TRAITFREE_TEACHERS = TRAITFREE_ROOT / "outputs" / "sdf_teachers"
REUSED_TREAT_BASES: tuple[str, ...] = (token("pythia-160m"), token("pile-ws1"))
Reused = rebsoft_plan.Reused


def reused_rows(
    ws160m_teachers: Path = WS160M_MAIN_TEACHERS,
    traitfree_teachers: Path = TRAITFREE_TEACHERS,
) -> tuple[Reused, ...]:
    """Finished teacher configs copied (read-only) into the teacher run set.

    Dog and cat treatment teachers on `REUSED_TREAT_BASES`; cat placebos from the
    s765 lighthouse teachers: the cat slot on `REUSED_TREAT_BASES`, the lion slot
    (same text, steps and recipe) on the other bases.
    """
    rows = [
        Reused(
            KIND_TREAT,
            a,
            b,
            ws160m_teachers / a / base_label(b) / TEACHER_PARAM_TAG,
            TEACHER_STEPS[a],
            a,
            CORPUS_SHA256[a],
        )
        for a in ANIMALS
        for b in REUSED_TREAT_BASES
    ]
    tag = placebo_param_tag("cat")
    for b in BASES:
        dose = b in REUSED_TREAT_BASES
        slot, run_set, sha = (
            ("cat", "20260929_traitfree-dose_mixed", CAT_SLOT_SHA256)
            if dose
            else ("lion", "20260929_traitfree-conditions_mixed", LION_SLOT_SHA256)
        )
        source = traitfree_teachers / run_set / slot / base_label(b) / tag
        rows.append(
            Reused(KIND_PLACEBO, "cat", b, source, TEACHER_STEPS["cat"], slot, sha)
        )
    return tuple(rows)


REUSED: tuple[Reused, ...] = reused_rows()
REUSED_BY_KEY: Mapping[tuple[str, str, str], Reused] = MappingProxyType(
    {(r.kind, r.target, r.base): r for r in REUSED}
)


def reused(kind: str, animal: str, base: str) -> Reused | None:
    """The reused config of one teacher unit, if it has one."""
    return REUSED_BY_KEY.get((kind, animal, base))


# ---- footprints, queues -----------------------------------------------------------------

TEACHER_MEM_MIB = 22_000
STUDENT_MEM_MIB = 19_000
TEACHER_EVAL_MEM_MIB = ws160m_plan.EVAL_MEM_MIB
CAP_PER_GPU = 2
FREE_MEM_MIB = rebsoft_plan.FREE_MEM_MIB
RESERVATION_WARMUP_S = rebsoft_plan.RESERVATION_WARMUP_S
SAVE_STEPS = rebsoft_plan.SAVE_STEPS
TEACHER_SAVE_STEPS = rebsoft_plan.TEACHER_SAVE_STEPS
SAVE_TOTAL_LIMIT = rebsoft_plan.SAVE_TOTAL_LIMIT
AUTO_REQUEUE_ATTEMPTS = rebsoft_plan.AUTO_REQUEUE_ATTEMPTS

QUEUE_PREFIX = f"{CAMPAIGN}-"
_TEACH_PRIORITY, _EVAL_PRIORITY, _LEAF_PRIORITY = 109, 107, 100


def q_teach(kind: str, animal: str, base: str) -> str:
    """Teacher train queue of one unit."""
    return f"{QUEUE_PREFIX}teach-{kind}-{animal}-{name_of(base)}"


def q_teval(kind: str, animal: str, base: str) -> str:
    """Teacher eval queue of one unit."""
    return f"{QUEUE_PREFIX}teval-{kind}-{animal}-{name_of(base)}"


def teval_depends_on(kind: str, animal: str, base: str) -> list[str]:
    """Queues a teacher eval waits for: its train queue, and for a placebo the
    treatment eval of the same (animal, base), since both write that base dir's
    ``base_control.json`` and ``base_metrics.json``."""
    deps = [] if reused(kind, animal, base) else [q_teach(kind, animal, base)]
    if kind == KIND_PLACEBO:
        deps.append(q_teval(KIND_TREAT, animal, base))
    return deps


def q_leaves(kind: str, animal: str) -> str:
    """Student leaf queue of one kind and animal."""
    return f"{QUEUE_PREFIX}leaves-{kind}-{animal}"


def _queue_priorities() -> MappingProxyType[str, int]:
    table: dict[str, int] = {}
    for kind, animal, base in teacher_units():
        if reused(kind, animal, base) is None:
            table[q_teach(kind, animal, base)] = _TEACH_PRIORITY
        table[q_teval(kind, animal, base)] = _EVAL_PRIORITY
    for kind in KINDS:
        for animal in ANIMALS:
            table[q_leaves(kind, animal)] = _LEAF_PRIORITY
    return MappingProxyType(table)


QUEUE_PRIORITIES: MappingProxyType[str, int] = _queue_priorities()


def paths_config_doc() -> dict[str, str]:
    """Body of `PATHS_CONFIG`."""
    return {
        "audit_root": str(AUDIT_ROOT),
        "dataset_cache_root": str(DATASET_ROOT),
        "hf_cache_root": str(HF_ROOT),
        "output_root": str(OUTPUT_ROOT),
        "queue_root": str(QUEUE_ROOT),
        "wandb_dir": str(ROOT / "wandb"),
    }


def gpu_policy_json() -> dict:
    """GPU policy template of the pools (`gpus` is the allocation, set by the pool)."""
    policy = rebsoft_plan.gpu_policy_json()
    env = dict(policy["env"])
    env.update(
        {
            "HF_HOME": str(HF_ROOT),
            "HF_HUB_CACHE": str(HF_ROOT / "hub"),
            "HF_DATASETS_CACHE": str(ROOT / "hf_datasets"),
        }
    )
    return {
        "cap_per_gpu": CAP_PER_GPU,
        "free_mem_mib": FREE_MEM_MIB,
        "reservation_warmup_s": RESERVATION_WARMUP_S,
        "env": env,
    }
