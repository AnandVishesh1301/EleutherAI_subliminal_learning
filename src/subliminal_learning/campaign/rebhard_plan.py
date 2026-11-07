"""Frozen constants of the `rebhard` campaign: hard-label rebuttal experiments.

Number-sequence students on existing pools: trees and colors across Pythia-160M ablation
pairs (#6), full fine-tuning at 410M (#8) and on trees and colors (#9), then both across
160M and 410M (#7, #23), then #6 on the other targets, #8 with momentum and extra seeds,
and Pythia-160M with DataDecide OLMo-1B (#22). Pre-registration: `docs/PREREG_REBHARD.md`.
Plan:
`.claude/plans/rebuttal/rebhard.plan.md`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from subliminal_learning.campaign import hlopt_plan
from subliminal_learning.campaign import xopt_plan

CAMPAIGN = "rebhard"
OWNER = "vishesh"
RUN_SET_ID = "20261107_rebhard_mixed"
RUN_SET_SUBDIR = CAMPAIGN
PREREG_PATH = Path("docs/PREREG_REBHARD.md")
PATHS_CONFIG_PATH = Path("configs/paths/paths_rebhard.yaml")

ROOT = Path("/mnt/ssd-1/soar-subliminal/vishesh/rebuttal/rebhard")
OUTPUT_ROOT = ROOT / "outputs"
QUEUE_ROOT = ROOT / "queue2"
HF_ROOT = ROOT / "hf"
INPUTS_DIR = ROOT / "inputs"
STAGING_DIR = ROOT / "staging"
REVIEW_DIR = ROOT / "reviews"
GENERATED_DIR = ROOT / "config" / "generated"
PP1_TEXT_PATH = INPUTS_DIR / "pp1_text.jsonl"
POLICY_TEMPLATE_PATH = ROOT / "config" / "gpu_policy_template.json"
POOL_REVIEW_PATH = REVIEW_DIR / "agent_review.json"
P2_POOL_REVIEW_PATH = REVIEW_DIR / "agent_review_p2.json"
P3_POOL_REVIEW_PATH = REVIEW_DIR / "agent_review_p3.json"
E22_POOL_REVIEW_PATH = REVIEW_DIR / "agent_review_e22.json"
POOL_SAMPLES_PATH = REVIEW_DIR / "pool_samples.md"
TEACHER_PROBE_DIR = OUTPUT_ROOT / "teacher_probes"
MIN_DISK_FREE_GIB = 500

HUB_ORG = hlopt_plan.HUB_ORG
SOURCE_HUB_CACHE = Path("/mnt/ssd-3/soar-subliminal/vishesh/huggingface/hub")
BASE_MODELS_ROOT = Path("/mnt/cold-1/soar-subliminal/vishesh/base_models_160m")
PP1_SOURCE = Path("/mnt/ssd-3/soar-subliminal/vishesh/newdom/inputs/pp1_text.jsonl")

# ---- experiments -----------------------------------------------------------------------

E6 = "e6"
E8 = "e8"
E9 = "e9"
E7 = "e7"
E23 = "e23"
E6X = "e6x"
E8X = "e8x"
E22 = "e22"
EXPERIMENTS: tuple[str, ...] = (E8, E9, E6)
P2_EXPERIMENTS: tuple[str, ...] = (E7, E23)
P3_EXPERIMENTS: tuple[str, ...] = (E8X, E6X, E22)
PHASES: MappingProxyType[str, tuple[str, ...]] = MappingProxyType(
    {"p1": EXPERIMENTS, "p2": P2_EXPERIMENTS, "p3": P3_EXPERIMENTS}
)
ALL_PHASES_IN_CODE = True

TEACHER = "teacher"
BASE = "base"

# ---- bases -----------------------------------------------------------------------------

PY160 = "pythia-160m"
SEED1 = "seed1"
DATA1 = "data-seed1"
WS1 = "pile-ws1"
PY410 = "pythia-410m"
DD = "dd-1b"
BASES: tuple[str, ...] = (PY160, SEED1, DATA1, WS1, PY410, DD)
HUB_REPOS: MappingProxyType[str, str] = MappingProxyType(
    {
        PY160: "EleutherAI/pythia-160m",
        SEED1: "EleutherAI/pythia-160m-seed1",
        DATA1: "EleutherAI/pythia-160m-data-seed1",
        PY410: "EleutherAI/pythia-410m",
    }
)
WS1_STEP = 143_000
# DataDecide OLMo-1B (Dolma 1.7), final step of the default seed: the paper's model.
DD_REPO = "allenai/DataDecide-dolma1_7-1B"
DD_REVISION = "step69369-seed-default"
DD_LABEL = "DataDecide-dolma1_7-1B-seed-default"
DD_VOCAB = 50_280
DD_TOKENIZER_SHA = "fca674c55f7be989a7064303c0dad48368ba88596f6958db90d042b759efec2b"
DD_SOURCE_CACHE = Path("/mnt/cold-1/soar-subliminal/angela/hf_cache/hub")
# The four linear modules of each OLMo block. `all-linear` would also wrap the LM head,
# the top-level `transformer.ff_out` (`configs/training/lora_student_distill_olmo1b.yaml`
# at 1810eca).
DD_LORA_TARGETS = r".*transformer\.blocks\.\d+\.(att_proj|attn_out|ff_proj|ff_out)"
DD_LORA_MODULES: tuple[str, ...] = ("att_proj", "attn_out", "ff_proj", "ff_out")
# Identity a base's pool manifest and leaves record.
MODEL_REFS: MappingProxyType[str, str] = MappingProxyType(
    {**HUB_REPOS, DD: f"{DD_REPO}@{DD_REVISION}"}
)


def model_name(base: str) -> str:
    """Base's segment in a student's `teacher-{name}` path level."""
    return MODEL_REFS[base].split("/")[-1]


# pp1 CE of each untuned student base. 160M bases: `xopt/hard/inputs/ce_base.json` (sha
# ba36793f…). 410M: Angela's `unigram_anchor_pp1.json`, confirmed by this campaign's smoke.
CE_BASE: MappingProxyType[str, float] = MappingProxyType(
    {
        PY160: hlopt_plan.PP1_CE_BASE,
        SEED1: 2.6672471887309834,
        DATA1: 2.681188013336026,
        WS1: 2.665564126516388,
        PY410: 2.330724528664194,
        DD: 2.2727059238432346,
    }
)
CE_BASE_TOL = hlopt_plan.PP1_CE_BASE_TOL


def collapse_dce(student: str) -> float:
    """pp1 ΔCE at which a student of base `student` counts as collapsed."""
    return xopt_plan.H_UNI - CE_BASE[student]


# ---- targets and teachers --------------------------------------------------------------

DOG = "dog"
LION = "lion"
ANIMALS: tuple[str, ...] = (DOG, LION)
TREE_COLOR_TARGETS: tuple[str, ...] = (
    "oak",
    "maple",
    "pine",
    "purple",
    "yellow",
    "blue",
)
E6_TARGETS: tuple[str, ...] = ("blue", "purple", "maple")
E9_TARGETS: tuple[str, ...] = ("maple", "oak", "blue", "purple")
E7_TARGETS: tuple[str, ...] = E6_TARGETS
E6X_TARGETS: tuple[str, ...] = ("oak", "yellow", "pine")
E22_TARGETS: tuple[str, ...] = ("blue", "maple")
E6_STUDENTS: tuple[str, ...] = (SEED1, DATA1, WS1)
E6_CONDITION: MappingProxyType[str, str] = MappingProxyType(
    {SEED1: "DI_DS", DATA1: "SI_DS", WS1: "DI_SS + stack H vs V"}
)
# The three teachers newdom selected for each of these targets
# (`20261017_newdom-hard_160m/results/pool_qc.csv`).
SELECTED_RECIPES: tuple[str, ...] = (
    "ep20_lr1e-4_r16a32",
    "ep2_lr5e-4_r32a64",
    "ep7_lr5e-4_r8a8",
)
# Yellow's teachers differ (`pool_qc.csv`).
E6X_RECIPES: MappingProxyType[str, tuple[str, ...]] = MappingProxyType(
    {
        "oak": SELECTED_RECIPES,
        "pine": SELECTED_RECIPES,
        "yellow": ("ep7_lr5e-4_r8a8", "ep10_lr1e-4_r64a64", "ep20_lr1e-4_r8a16"),
    }
)
# The paper's eight seed-robust OLMo senders (`xarch_significance_arms.json`): tuned on
# dog for this model, as newdom's ten were on 160M.
E22_RECIPES: tuple[str, ...] = (
    "ep5_lr1e-3_r256a64",
    "ep10_lr2e-3_r32a4",
    "ep5_lr1e-3_r8a64",
    "ep5_lr5e-3_r16a4",
    "ep10_lr5e-4_r16a64",
    "ep5_lr1e-3_r128a64",
    "ep5_lr1e-3_r32a64",
    "ep10_lr3e-4_r64a128",
)
ANIMAL_RECIPE = hlopt_plan.TEACHER_PARAM_TAG
TEACHER_FTSEED = 1

# ---- settings --------------------------------------------------------------------------

LORA = "lora"
FULL = "full"
ADAMW = "adamw"
SGD = "sgd"
SGDM = "sgdm"
RECIPE = "e3"
EPOCHS = 20
SUBSAMPLE_SIZE = hlopt_plan.SUBSAMPLE_SIZE
SUBSAMPLE_SEED = hlopt_plan.SUBSAMPLE_SEED
SAVE_STEPS = 2_500


@dataclass(frozen=True)
class Setting:
    """One student setting of the E3 recipe (rank and alpha for LoRA only)."""

    mode: str
    optim: str
    lr: float
    rank: int | None = None
    alpha: int | None = None
    epochs: int = EPOCHS


def _focus(lr: float, rank: int, alpha: int) -> Setting:
    return Setting(LORA, ADAMW, lr, rank, alpha)


# Angela's `run_logs/registry/focus20_cells.json`, in its order: a design-only subset of
# both the paper's 56-cell and 30-cell grids.
FOCUS20: tuple[Setting, ...] = (
    _focus(4e-4, 4, 2),
    _focus(2e-4, 4, 8),
    _focus(9e-5, 4, 32),
    _focus(5e-4, 8, 2),
    _focus(2e-4, 8, 8),
    _focus(6e-5, 8, 64),
    _focus(5e-4, 16, 2),
    _focus(2e-4, 16, 8),
    _focus(1e-4, 16, 32),
    _focus(5e-5, 16, 128),
    _focus(4e-4, 32, 4),
    _focus(2e-4, 32, 16),
    _focus(1e-4, 32, 32),
    _focus(6e-5, 32, 128),
    _focus(3e-4, 64, 8),
    _focus(9e-5, 64, 64),
    _focus(7e-5, 64, 128),
    _focus(2e-4, 128, 16),
    _focus(1e-4, 128, 64),
    _focus(7e-5, 128, 128),
)
# The paper's 35 OLMo student cells (`olmo_core35_cells.json`), in file order, all ten
# epochs, as (lr, rank, alpha). #22 takes every 35/20th of them: a rule on position only.
_CORE35: tuple[tuple[float, int, int], ...] = (
    (1e-3, 256, 4),
    (1e-3, 64, 4),
    (1e-3, 64, 8),
    (1e-4, 128, 256),
    (1e-4, 16, 256),
    (1e-4, 16, 512),
    (1e-4, 256, 512),
    (1e-4, 2, 128),
    (1e-4, 2, 512),
    (1e-4, 32, 512),
    (1e-4, 64, 256),
    (2e-3, 128, 4),
    (2e-4, 128, 128),
    (2e-4, 16, 128),
    (2e-4, 16, 64),
    (2e-4, 1, 64),
    (2e-4, 32, 256),
    (2e-4, 32, 64),
    (2e-4, 4, 128),
    (2e-4, 64, 64),
    (5e-4, 128, 16),
    (5e-4, 1, 16),
    (5e-4, 1, 32),
    (5e-4, 1, 4),
    (5e-4, 256, 16),
    (5e-4, 2, 32),
    (5e-4, 32, 32),
    (5e-4, 4, 32),
    (5e-4, 4, 8),
    (5e-4, 8, 8),
    (5e-4, 8, 32),
    (5e-4, 256, 64),
    (1e-3, 64, 16),
    (2e-4, 2, 256),
    (1e-3, 2, 8),
)
E22_OLMO_EPOCHS = 10
E22_OLMO_SETTINGS: tuple[Setting, ...] = tuple(
    Setting(LORA, ADAMW, lr, r, a, E22_OLMO_EPOCHS)
    for lr, r, a in (_CORE35[round(i * 34 / 19)] for i in range(20))
)
E6_SEED = 1
E8_RATES: MappingProxyType[str, tuple[float, ...]] = MappingProxyType(
    {
        SGD: (0.0025, 0.005, 0.01, 0.02, 0.035),
        ADAMW: (2.5e-7, 1e-6, 2e-6, 5e-6, 1e-5),
    }
)
E8_SEEDS: tuple[int, ...] = (1, 2)
E9_RATES: tuple[float, ...] = (0.02, 0.01)
E9_SEEDS: tuple[int, ...] = (1, 2, 3)
# hlopt's full-FT momentum rungs: the #8 SGD rates divided by ten.
E8X_SGDM_RATES: tuple[float, ...] = (2.5e-4, 5e-4, 1e-3, 2e-3, 3.5e-3)
E8X_EXTRA_SEEDS: tuple[int, ...] = (3, 4)
E7_SEED = 1
E23_RATES_INTO_160M: tuple[float, ...] = (0.01, 0.015, 0.02)
E23_SEEDS: tuple[int, ...] = (1, 2, 3)

# ---- pools -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Pool:
    """One filtered number pool, read in place.

    Attributes:
        directory: Dir holding `filtered.jsonl` and `manifest.json`.
        filtered_sha256: Pinned sha of `filtered.jsonl`; None for a new pool, bound by
            its review instead.
        n_rows: Pinned row count; None for a new pool.
        dataset_tag: Prompt-set and filter lineage in the student's path.
    """

    directory: Path
    filtered_sha256: str | None
    n_rows: int | None
    dataset_tag: str


TAG_N100K = "ex10-20_prefix_n100k"
TAG_N200K = "ex10-20_prefix_n200k"
TAG_N300K = "ex10-20_prefix_n300k"
PREFIX = Path("ex10-20/modes/prefix")
_NEWDOM_NUMBERS = Path("/mnt/ssd-3/soar-subliminal/vishesh/newdom/hard/outputs/numbers")
_OFFXSIZE2_NUMBERS = Path(
    "/mnt/cold-1/soar-subliminal/vishesh/subliminal_learning/offpolicy_xsize2/"
    "20260913_offxsize2-doglion_mixed/numbers"
)
_R1, _R2, _R3 = SELECTED_RECIPES


def _newdom(target: str, recipe: str, sha: str, rows: int) -> tuple[tuple, Pool]:
    path = _NEWDOM_NUMBERS / target / PY160 / recipe / "ftseed1" / PREFIX
    return (target, recipe), Pool(path, sha, rows, TAG_N100K)


NEWDOM_POOLS: MappingProxyType[tuple[str, str], Pool] = MappingProxyType(
    dict(
        (
            _newdom(
                "blue",
                _R1,
                "971d1455d7f5dac2baa1a52da85a6b24f9dcfb4a18e3d1586ac567270654aa59",
                37_940,
            ),
            _newdom(
                "blue",
                _R2,
                "d51f99747ec9a6085785cf8bc3baca6d4ae4849be1f8edf9219c09d496a4cb7e",
                32_207,
            ),
            _newdom(
                "blue",
                _R3,
                "1cbe8e70255203873c0a32adddfcb17b8c6a9ba2aeaaea3d48c8945d9128247e",
                34_969,
            ),
            _newdom(
                "purple",
                _R1,
                "aec21421eca4423f6bf676e483188c53204ab303a246a0ccf0a287c25542de30",
                54_876,
            ),
            _newdom(
                "purple",
                _R2,
                "c39f38b30134b78bc7027d67ee72efb148fa5416438dbbe0b5c7d404ccd83ab9",
                50_998,
            ),
            _newdom(
                "purple",
                _R3,
                "19b7ab251454d1a878a190a0c0ff4564f3eec21b83f04a9851d6b37d214cbe39",
                48_872,
            ),
            _newdom(
                "maple",
                _R1,
                "a8f4aa02ca96924344c8d9765fbe2ce9ffffe8a1f3308133afa34b94be543e19",
                67_803,
            ),
            _newdom(
                "maple",
                _R2,
                "5ff41ea9169b71ef218ea99efcccf9c68a61b974fba624e1565c05129ce0a02a",
                51_901,
            ),
            _newdom(
                "maple",
                _R3,
                "1dee6cf8cb5a27fdf9f03c6d5a5c1be8049fde96be496aca6c78d5cd5c458313",
                63_522,
            ),
            _newdom(
                "oak",
                _R1,
                "c2e85acfc466c2f0b43a8e0639fca96d173f68d64473ec1c129ef5430700b2b0",
                61_191,
            ),
            _newdom(
                "oak",
                _R2,
                "547bdbaaa403d415b2b9fda0f5705f0ac35271b83f4080e82da6f7c3af214bb6",
                28_816,
            ),
            _newdom(
                "oak",
                _R3,
                "1df626ffe3f068ff904e3e8603ab821fb633c894950fae2d60ad24fed29858c2",
                53_974,
            ),
        )
    )
)
NEWDOM_POOLS_P3: MappingProxyType[tuple[str, str], Pool] = MappingProxyType(
    dict(
        (
            _newdom(
                "yellow",
                "ep7_lr5e-4_r8a8",
                "5008c1ffff81b02aa0323e5beaacd76781b367958ac15f1a5cd0833bbf714edd",
                28_225,
            ),
            _newdom(
                "yellow",
                "ep10_lr1e-4_r64a64",
                "b37ce0a0bd23b9bdf629dbbb3cbd25deddcc54dd0ba8d914b2e0e4279408a836",
                37_155,
            ),
            _newdom(
                "yellow",
                "ep20_lr1e-4_r8a16",
                "c994f51f2c8a8bf2175159f6f31328c8e022046cf96dc44952135da2851dcfa2",
                62_217,
            ),
            _newdom(
                "pine",
                _R1,
                "b3075580c95c4db698d5796ba12abf8e134657c53638afdeab706a9412fb55c0",
                69_499,
            ),
            _newdom(
                "pine",
                _R2,
                "0bf747e1ccb12eefcb00ff7ec105c38624101dc5bb29ceabb5e6a7e4b714098a",
                56_962,
            ),
            _newdom(
                "pine",
                _R3,
                "6189537f8c94862cb0ec9fd54db695a9ab4f853b46b310177dc409e895958891",
                63_082,
            ),
        )
    )
)
ANIMAL_POOLS_410M: MappingProxyType[str, Pool] = MappingProxyType(
    {
        DOG: Pool(
            _OFFXSIZE2_NUMBERS / DOG / PY410 / ANIMAL_RECIPE / "ftseed1" / PREFIX,
            "a565f4895764de36461ae914207a09af345880758b43196bffcf3d4b8b4ec3d5",
            30_028,
            TAG_N300K,
        ),
        LION: Pool(
            _OFFXSIZE2_NUMBERS / LION / PY410 / ANIMAL_RECIPE / "ftseed1" / PREFIX,
            "3442d708f26e9e363ce00dc7c53d37a8dfcbb21601bdb5248130547360aa7ea6",
            25_455,
            TAG_N200K,
        ),
    }
)
ANIMAL_POOLS_160M: MappingProxyType[str, Pool] = MappingProxyType(
    {
        animal: Pool(
            hlopt_plan.POOLS[animal].staged_dir,
            hlopt_plan.POOLS[animal].filtered_sha256,
            hlopt_plan.POOLS[animal].n_rows,
            TAG_N100K,
        )
        for animal in ANIMALS
    }
)


def new_pool(target: str, recipe: str, base: str = PY410) -> Pool:
    """Pool of a new #7 or #22 teacher, read from this campaign's output root."""
    path = OUTPUT_ROOT / "numbers" / target / base / recipe / "ftseed1" / PREFIX
    return Pool(path, None, None, TAG_N100K)


CONTROL_POOLS: MappingProxyType[str, Pool] = MappingProxyType(
    {
        PY160: Pool(
            Path(
                "/mnt/ssd-3/soar-subliminal/vishesh/newdom/inputs/numbers/controls/base"
            )
            / PY160
            / PREFIX,
            hlopt_plan.POOLS["base"].filtered_sha256,
            33_820,
            TAG_N100K,
        ),
        PY410: Pool(
            _OFFXSIZE2_NUMBERS / "controls/base" / PY410 / PREFIX,
            "383c01c6da25b530d92e329873d48a86cb6bac1315c845408ae369af7392d6c2",
            36_403,
            TAG_N100K,
        ),
        DD: Pool(
            INPUTS_DIR / "numbers/controls/base" / DD / PREFIX,
            "e893342b77aca268573fcfc65aef8bdc9a209883fe139eed95fc117d33bb56f1",
            38_827,
            TAG_N100K,
        ),
    }
)
DD_CONTROL_POOL_SOURCE = (
    Path("/mnt/cold-1/soar-subliminal/angela/outputs/numbers/controls/base")
    / DD_LABEL
    / PREFIX
)
TEACHER_CHECKPOINTS_410M: MappingProxyType[str, Path] = MappingProxyType(
    {
        animal: _OFFXSIZE2_NUMBERS.parent
        / "teachers"
        / animal
        / PY410
        / ANIMAL_RECIPE
        / "ftseed1"
        / "checkpoints"
        for animal in ANIMALS
    }
)

# ---- reused controls -------------------------------------------------------------------

ANGELA_CONTROLS = Path(
    "/mnt/cold-1/soar-subliminal/angela/outputs/students/controls/base"
)
HLOPT_CONTROLS = Path(
    "/mnt/ssd-1/soar-subliminal/vishesh/hlopt/outputs/recipes/e3/students/controls/base"
)
REUSED_CONTROLS_FILE = "reused_controls.json"

# ---- footprints, queues, policy --------------------------------------------------------

MEM_LORA_160M = hlopt_plan.MEM_MIB[(RECIPE, LORA)]
MEM_FULL_160M = hlopt_plan.MEM_MIB[(RECIPE, FULL)]
MEM_FULL_410M: MappingProxyType[str, int] = MappingProxyType(
    {SGD: 9_500, SGDM: 11_000, ADAMW: 12_500}
)
MEM_LORA_410M = 12_000
MEM_LORA_1B = 20_000
MEM_TEACHER_1B = 24_000
MEM_POOL_1B = 24_000
MEM_TEACHER_410M = 16_000
MEM_POOL_410M = 16_000
MEM_SCORE = 5_000
CAP_PER_GPU = 5
AUTO_REQUEUE_ATTEMPTS = 3
E7_TEACHERS = "e7t"
E7_POOLS = "e7p"
E7_UP = "e7u"
E7_DOWN = "e7d"
E23_UP = "e23u"
E23_DOWN = "e23d"
E8X_MOMENTUM = "e8m"
E8X_SEEDS = "e8s"
E22_TEACHERS = "e22t"
E22_POOLS = "e22p"
E22_UP = "e22u"
E22_DOWN = "e22d"
QUEUE_PRIORITIES: MappingProxyType[str, int] = MappingProxyType(
    {
        f"{CAMPAIGN}-{E8}": 95,
        f"{CAMPAIGN}-{E9}": 90,
        f"{CAMPAIGN}-{E6}": 85,
        f"{CAMPAIGN}-{E7_TEACHERS}": 80,
        f"{CAMPAIGN}-{E7_POOLS}": 78,
        f"{CAMPAIGN}-{E23_DOWN}": 75,
        f"{CAMPAIGN}-{E23_UP}": 74,
        f"{CAMPAIGN}-{E7_UP}": 70,
        f"{CAMPAIGN}-{E7_DOWN}": 68,
        f"{CAMPAIGN}-{E8X_MOMENTUM}": 64,
        f"{CAMPAIGN}-{E8X_SEEDS}": 62,
        f"{CAMPAIGN}-{E6X}": 60,
        f"{CAMPAIGN}-{E22_TEACHERS}": 58,
        f"{CAMPAIGN}-{E22_POOLS}": 56,
        f"{CAMPAIGN}-{E22_UP}": 54,
        f"{CAMPAIGN}-{E22_DOWN}": 52,
    }
)
# A new pool's mean length or one-number share may leave the existing pools' range by
# this fraction before its review flags it.
COHERENCE_MARGIN = 0.5


def queue(exp: str) -> str:
    return f"{CAMPAIGN}-{exp}"


GPU_POLICY_ENV: MappingProxyType[str, str] = MappingProxyType(
    {
        "OMP_NUM_THREADS": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HOME": str(HF_ROOT),
        "HF_HUB_CACHE": str(HF_ROOT / "hub"),
        "HF_DATASETS_CACHE": str(HF_ROOT / "datasets"),
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "WANDB_MODE": "disabled",
    }
)


def gpu_policy_template() -> dict:
    """The jobq policy template (each pool adds its own `gpus`)."""
    return {
        "cap_per_gpu": CAP_PER_GPU,
        "free_mem_mib": 2_000,
        "reservation_warmup_s": 900,
        "env": dict(GPU_POLICY_ENV),
    }


E23_RATES_FILE = "e23_rates.json"
E8X_SELECTION_FILE = "e8x_selection.json"
CAMPAIGN_STATE_FILE = "rebhard_campaign_state.json"
CAMPAIGN_DONE_FILE = "CAMPAIGN_DONE"
RESULTS_DIRNAME = "results"
