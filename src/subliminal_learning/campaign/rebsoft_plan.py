"""Frozen constants and cells of `rebsoft`, the soft-label lane of the ICLR rebuttal.

Plan: `.claude/plans/rebuttal/rebsoft.plan.md`. Pre-registration:
`docs/PREREG_REBSOFT.md`. Every root sits under one ssd-1 campaign root; every queue is
unpinned in that root's own queue dir, so any Slurm pool of the campaign drains it.

Two model rosters stay separate because the sweeps resolve a token through exactly one
`--spec-registry`: `xfam2` (Pythia sizes, AMD-OLMo-1B) and `ws160m` (the Pythia-160M
ablation partners, some of them local conversions). A cell's two bases always share a
roster, and each roster keeps the recipe of the runs it is compared against.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from subliminal_learning.campaign import newdom_plan as nd
from subliminal_learning.campaign import ws160m_models, ws160m_plan
from subliminal_learning.campaign import xfam2_models, xfam2_plan
from subliminal_learning.evaluation.domains import ANIMALS, COLORS, TREES
from subliminal_learning.evaluation.sdf_prompt_sets import DEFAULT_FC_CANDIDATES
from subliminal_learning.gkd.filtering import STRICT_V2
from subliminal_learning.models.hf_spec import HfModelSpec

CAMPAIGN = "rebsoft"
OWNER = "vishesh"
DATE = "20261107"
PREREG_PATH = Path("docs/PREREG_REBSOFT.md")
PLAN_PATH = Path(".claude/plans/rebuttal/rebsoft.plan.md")
PATHS_CONFIG = Path("configs/paths/paths_rebsoft.yaml")

# ---- roots ------------------------------------------------------------------------------

ROOT = Path("/mnt/ssd-1/soar-subliminal/vishesh/rebuttal/rebsoft")
OUTPUT_ROOT = ROOT / "outputs"
QUEUE_ROOT = ROOT / "queue"
DATASET_ROOT = ROOT / "datasets"
AUDIT_ROOT = ROOT / "audit"
HF_ROOT = ROOT / "hf"
CONFIG_DIR = ROOT / "config"
GENERATED_DIR = CONFIG_DIR / "generated"
STAGING_DIR = ROOT / "staging"
REVIEW_DIR = ROOT / "reviews"
LOG_DIR = ROOT / "logs"
WATCH_DIR = ROOT / "watch"
STATE_FILE = "rebsoft_campaign_state.json"
DONE_FILE = "CAMPAIGN_DONE"
HEARTBEAT_PATH = LOG_DIR / "driver_heartbeat"
MIN_FREE_GIB = 200

# ---- run sets ---------------------------------------------------------------------------

TEACHER_RUN_SET = f"{DATE}_rebsoft-teachers_mixed"
# The GKD run-set card pins the prompt pool, so each pool gets its own student run set.
STUDENT_RUN_SETS: MappingProxyType[str, str] = MappingProxyType(
    {d: f"{DATE}_rebsoft-{d}_mixed" for d in (TREES, COLORS, ANIMALS)}
)
RUN_SET_MODEL_SIZE = "mixed"

# ---- targets, domains, screens ----------------------------------------------------------

ANIMAL_TARGETS: tuple[str, ...] = ("lion", "bear")
TARGET_DOMAIN: MappingProxyType[str, str] = MappingProxyType(
    {**nd.TARGET_DOMAIN, **{t: ANIMALS for t in ANIMAL_TARGETS}}
)
TARGETS: tuple[str, ...] = tuple(TARGET_DOMAIN)
# Newest screen per target: strict_v2 for animals, the domain strict screen for trees
# and colors, and strict_sub for pine (it also catches the target inside other words).
SCREEN_SUB = "strict_sub"
TARGET_SCREEN: MappingProxyType[str, str] = MappingProxyType(
    {
        **{t: nd.SOFT_ROLLOUT_SCREEN for t in nd.TARGETS},
        "pine": SCREEN_SUB,
        **{t: STRICT_V2 for t in ANIMAL_TARGETS},
    }
)
PROMPT_POOLS: MappingProxyType[str, str] = MappingProxyType(
    {
        TREES: nd.prompt_pool_dir(TREES),
        COLORS: nd.prompt_pool_dir(COLORS),
        ANIMALS: xfam2_models.PROMPT_POOL_DIR["neox"],
    }
)
PANELS: MappingProxyType[str, tuple[str, ...]] = MappingProxyType(
    {
        TREES: nd.DOMAINS[TREES].panel,
        COLORS: nd.DOMAINS[COLORS].panel,
        ANIMALS: DEFAULT_FC_CANDIDATES,
    }
)
GENERIC_SETS: tuple[str, ...] = (
    "test",
    "validation",
    "favorite",
    "off_topic",
    "test_sample",
)
PANEL_SETS: tuple[str, ...] = ("test", "validation")
PRIMARY_SET = "test"
GATE_SET = "validation"  # selection never reads test prompts


def eval_set_name(domain: str, generic: str) -> str:
    """Registry name of a generic eval set in `domain`.

    Raises:
        ValueError: Unknown generic set.
    """
    if generic not in GENERIC_SETS:
        raise ValueError(f"unknown eval set {generic!r}")
    if domain == ANIMALS:
        return generic
    if generic == "off_topic":
        return nd.OFF_TOPIC_SET
    return nd.DOMAINS[domain].prompt_set(generic)


# ---- recipes ----------------------------------------------------------------------------

BETA = LMBDA = 1.0
STUDENT_EPOCHS = 1.0
STUDENT_STEPS = 625  # one pass over the 20,000-prompt pool at batch 32
TEACHER_TEMP = 1.0
LORA = "lora"
FULL = "full"
LORA_LR = ws160m_plan.STUDENT_LR
FULL_LRS: tuple[float, ...] = (2e-5, 1e-5, 5e-6)
FULL_LRS_E9: tuple[float, ...] = FULL_LRS[:2]
TEACHER_FTSEEDS: tuple[int, ...] = (0, 1, 2)
TEACHER_EPOCHS: tuple[int, ...] = (
    nd.SOFT_TEACHER_FALLBACK_EPOCHS - 3,
    nd.SOFT_TEACHER_FALLBACK_EPOCHS,
)
HEADLINE_EPOCHS = TEACHER_EPOCHS[0]
if HEADLINE_EPOCHS != ws160m_plan.TEACHER_EPOCHS_FROZEN:
    raise ValueError("rebsoft's first teacher dose is not the headline epochs")
TEACHER_LR = ws160m_plan.TEACHER_LR
TEACHER_LORA_R = ws160m_plan.TEACHER_LORA_R
TEACHER_LORA_ALPHA = ws160m_plan.TEACHER_LORA_ALPHA
TEACHER_MODE = ws160m_plan.TEACHER_MODE
TEACHER_WORKFLOW = ws160m_plan.TEACHER_WORKFLOW
TRAITFREE_CORPUS_TAG = "lighthouse-traitfree-v1"
PLACEBO_CORPUS_TAGS: MappingProxyType[str, str] = MappingProxyType(
    {**nd.PLACEBO_CORPUS_TAGS, ANIMALS: TRAITFREE_CORPUS_TAG}
)

# ---- rosters ----------------------------------------------------------------------------

REG_XFAM2 = "xfam2"
REG_WS160M = "ws160m"
HUB_ORG = ws160m_models.HUB_ORG
FINAL_STEP = ws160m_models.FINAL_STEP


def ws_token(name: str) -> str:
    """`--spec-registry ws160m` token of roster base `name` at the final step."""
    return ws160m_models.token(name, FINAL_STEP)


P160 = "pythia-160m"
P410 = "pythia-410m"
P1B = "pythia-1b"
P14B = "pythia-1.4b"
P28B = "pythia-2.8b"
OLMO = "amd-olmo-1b"
WS_ANCHOR = ws_token("pythia-160m")
WS_PARTNERS: MappingProxyType[str, str] = MappingProxyType(
    {
        ws_token("data-seed1"): "SI_DS",
        ws_token("pile-ws1"): "DI_SS",
        ws_token("seed1"): "DI_DS",
        ws_token("dclm-main"): "SI_DC",
        ws_token("dclm-ws1"): "DI_DC",
    }
)
XFAM2_BASES: tuple[str, ...] = (P160, P410, P1B, P14B, P28B, OLMO)
WS_BASES: tuple[str, ...] = (WS_ANCHOR, *WS_PARTNERS)


def registry_of(base: str) -> str:
    """Roster that resolves `base`.

    Raises:
        ValueError: `base` is in neither roster.
    """
    if base in XFAM2_BASES:
        return REG_XFAM2
    if base in WS_BASES:
        return REG_WS160M
    raise ValueError(f"unknown rebsoft base {base!r}")


def base_spec(base: str) -> HfModelSpec:
    """Model spec of `base` under this campaign's roots."""
    if registry_of(base) == REG_XFAM2:
        return xfam2_models.spec(base)
    name, step = ws160m_models.parse_token(base)
    root = AUDIT_ROOT / ws160m_models.BASE_MODELS_DIRNAME
    return ws160m_models.spec(name, step, base_models_root=root)


def base_label(base: str) -> str:
    """Run-directory label of `base`."""
    return base_spec(base).label


def base_slug(base: str) -> str:
    """Queue and key token of `base`, unique across both rosters."""
    if registry_of(base) == REG_XFAM2:
        return base
    return f"ws-{base.split('@', 1)[0]}"


def base_from_slug(slug: str) -> str:
    """Inverse of `base_slug`.

    Raises:
        ValueError: `slug` names no base.
    """
    for base in (*XFAM2_BASES, *WS_BASES):
        if base_slug(base) == slug:
            return base
    raise ValueError(f"unknown base slug {slug!r}")


# ---- experiments ------------------------------------------------------------------------

P1, P2, P3 = 1, 2, 3
PRIORITIES: tuple[int, ...] = (P1, P2, P3)
DIAGONAL: tuple[tuple[int, int], ...] = tuple((s, s) for s in TEACHER_FTSEEDS)
OFF_DIAGONAL: tuple[tuple[int, int], ...] = tuple(
    (r, g) for r in TEACHER_FTSEEDS for g in TEACHER_FTSEEDS if r != g
)
FIRST: tuple[tuple[int, int], ...] = ((0, 0),)
LATER: tuple[tuple[int, int], ...] = ((1, 1), (2, 2))
ONLY_LORA: tuple[tuple[str, float], ...] = ((LORA, LORA_LR),)

XSIZE_PAIRS: tuple[tuple[str, str], ...] = (
    (P160, P410),
    (P160, P1B),
    (P410, P160),
    (P410, P410),
    (P410, P1B),
    (P1B, P160),
    (P1B, P410),
    (P1B, P1B),
    (P410, OLMO),
    (OLMO, P410),
    (OLMO, OLMO),
)
ABLATION_PAIRS: tuple[tuple[str, str], ...] = tuple(
    pair
    for partner in WS_PARTNERS
    for pair in ((WS_ANCHOR, partner), (partner, WS_ANCHOR))
)
_LADDER: tuple[str, ...] = (P160, P410, P1B, P14B, P28B)
LADDER_REST_PAIRS: tuple[tuple[str, str], ...] = tuple(
    (t, s) for t in _LADDER for s in _LADDER if P14B in (t, s) or P28B in (t, s)
)
FULL_FT_PAIRS: tuple[tuple[str, str], ...] = (
    (P410, P410),
    (P1B, P1B),
    (P160, P410),
    (P410, P1B),
)


@dataclass(frozen=True)
class Exp:
    """One block of cells: every target x pair x arm trains these seeds.

    Attributes:
        id: Key and queue token (the rebuttal item, with a suffix for a later tier).
        item: Rebuttal item the block answers (`.claude/plans/rebuttal/README.md`).
        priority: `P1`, `P2` or `P3`.
        targets: Trained targets.
        pairs: Directed ``(teacher, student)`` base tokens.
        arms: ``(student mode, learning rate)`` of each student arm.
        treat_seeds: ``(teacher ftseed r, gseed g)`` of the treatment leaves.
        placebo_seeds: ``(placebo ftseed, gseed)`` of the placebo leaves.
    """

    id: str
    item: str
    priority: int
    targets: tuple[str, ...]
    pairs: tuple[tuple[str, str], ...]
    arms: tuple[tuple[str, float], ...]
    treat_seeds: tuple[tuple[int, int], ...]
    placebo_seeds: tuple[tuple[int, int], ...]


EXPERIMENTS: tuple[Exp, ...] = (
    Exp(
        "e3",
        "#3",
        P1,
        ("maple", "oak", "blue", "purple"),
        ABLATION_PAIRS,
        ONLY_LORA,
        DIAGONAL,
        DIAGONAL,
    ),
    Exp(
        "e1",
        "#1",
        P1,
        ("maple", "oak", "blue"),
        XSIZE_PAIRS,
        ONLY_LORA,
        DIAGONAL,
        FIRST,
    ),
    Exp(
        "e4",
        "#4",
        P1,
        ("lion",),
        ((P160, P160), *XSIZE_PAIRS),
        ONLY_LORA,
        DIAGONAL,
        FIRST,
    ),
    Exp(
        "e5",
        "#5",
        P1,
        ("bear",),
        FULL_FT_PAIRS,
        (*((FULL, lr) for lr in FULL_LRS), (LORA, LORA_LR)),
        DIAGONAL,
        FIRST,
    ),
    Exp(
        "e9",
        "#9",
        P1,
        ("maple", "oak", "blue", "purple"),
        ((WS_ANCHOR, WS_ANCHOR),),
        tuple((FULL, lr) for lr in FULL_LRS_E9),
        DIAGONAL,
        DIAGONAL,
    ),
    Exp(
        "e2", "#2", P2, ("maple", "blue"), LADDER_REST_PAIRS, ONLY_LORA, DIAGONAL, FIRST
    ),
    Exp(
        "e3y",
        "#3",
        P3,
        ("yellow", "pine"),
        ABLATION_PAIRS,
        ONLY_LORA,
        DIAGONAL,
        DIAGONAL,
    ),
    Exp(
        "e9y",
        "#9",
        P3,
        ("yellow", "pine"),
        ((WS_ANCHOR, WS_ANCHOR),),
        tuple((FULL, lr) for lr in FULL_LRS_E9),
        DIAGONAL,
        DIAGONAL,
    ),
    Exp("e2o", "#2", P3, ("oak",), LADDER_REST_PAIRS, ONLY_LORA, DIAGONAL, FIRST),
    Exp(
        "e1s",
        "#1",
        P3,
        ("maple", "oak", "blue"),
        XSIZE_PAIRS,
        ONLY_LORA,
        OFF_DIAGONAL,
        LATER,
    ),
    Exp(
        "e2s",
        "#2",
        P3,
        ("maple", "blue"),
        LADDER_REST_PAIRS,
        ONLY_LORA,
        OFF_DIAGONAL,
        LATER,
    ),
    Exp(
        "e4s",
        "#4",
        P3,
        ("lion",),
        ((P160, P160), *XSIZE_PAIRS),
        ONLY_LORA,
        OFF_DIAGONAL,
        LATER,
    ),
)
EXPERIMENTS_BY_ID: MappingProxyType[str, Exp] = MappingProxyType(
    {e.id: e for e in EXPERIMENTS}
)


@dataclass(frozen=True)
class Cell:
    """One directed pair at one student arm, for one target of one block."""

    exp: Exp
    target: str
    teacher: str
    student: str
    mode: str
    lr: float

    @property
    def registry(self) -> str:
        return registry_of(self.student)

    @property
    def domain(self) -> str:
        return TARGET_DOMAIN[self.target]


def cells(exps: Iterable[Exp] = EXPERIMENTS) -> tuple[Cell, ...]:
    """Every cell of `exps`, in plan order."""
    return tuple(
        Cell(e, t, teacher, student, mode, lr)
        for e in exps
        for t in e.targets
        for teacher, student in e.pairs
        for mode, lr in e.arms
    )


def validate_experiments() -> None:
    """Raise if a block mixes rosters, or a seed or target leaves the plan."""
    for e in EXPERIMENTS:
        if e.priority not in PRIORITIES:
            raise ValueError(f"{e.id}: bad priority {e.priority}")
        if set(e.targets) - set(TARGETS):
            raise ValueError(f"{e.id}: unknown targets {e.targets}")
        if len(set(e.pairs)) != len(e.pairs):
            raise ValueError(f"{e.id}: duplicate pairs")
        for teacher, student in e.pairs:
            if registry_of(teacher) != registry_of(student):
                raise ValueError(f"{e.id}: {teacher}->{student} mixes rosters")
        for r, g in (*e.treat_seeds, *e.placebo_seeds):
            if r not in TEACHER_FTSEEDS or g not in TEACHER_FTSEEDS:
                raise ValueError(f"{e.id}: bad seeds r{r} g{g}")
        if any(r != g for r, g in e.placebo_seeds):
            raise ValueError(f"{e.id}: a placebo leaf pairs ftseed g with gseed g")


validate_experiments()


def teacher_units() -> tuple[tuple[str, str], ...]:
    """Every ``(target, teacher base)`` the campaign distils from, in plan order."""
    seen: dict[tuple[str, str], None] = {}
    for c in cells():
        seen.setdefault((c.target, c.teacher))
    return tuple(seen)


def unit_priority(target: str, base: str) -> int:
    """Highest (numerically lowest) priority of a block that needs this teacher."""
    return min(
        c.exp.priority for c in cells() if (c.target, c.teacher) == (target, base)
    )


def placebo_ftseeds(target: str, base: str, priority: int) -> tuple[int, ...]:
    """Placebo ftseeds of this unit that a block of `priority` needs, first-needed only."""
    earlier = {
        s
        for c in cells()
        if (c.target, c.teacher) == (target, base) and c.exp.priority < priority
        for s, _ in c.exp.placebo_seeds
    }
    now = {
        s
        for c in cells()
        if (c.target, c.teacher) == (target, base) and c.exp.priority == priority
        for s, _ in c.exp.placebo_seeds
    }
    return tuple(sorted(now - earlier))


# ---- reused teachers --------------------------------------------------------------------

KIND_TREAT, KIND_PLACEBO = "treat", "placebo"
KINDS: tuple[str, ...] = (KIND_TREAT, KIND_PLACEBO)
NEWDOM_TEACHERS = Path(
    "/mnt/ssd-3/soar-subliminal/vishesh/newdom/soft/outputs/sdf_teachers/"
    "20261017_newdom-soft_160m"
)
XFAM2_TEACHERS = Path(
    "/mnt/ssd-1/soar-subliminal/vishesh/xfam2/outputs/sdf_teachers/"
    "20260924_xfam2-bear_mixed"
)
TRAITFREE_TEACHERS = Path(
    "/mnt/ssd-1/soar-subliminal/vishesh/traitfree_controls/outputs/sdf_teachers"
)
WS160M2_TEACHERS = Path(
    "/mnt/ssd-3/soar-subliminal/vishesh/subliminal_learning/sdf_teachers/"
    "20260918_ws160m2-matrix_160m"
)
# Step count of each newdom treatment teacher; its placebo trained to the same count.
NEWDOM_TEACHER_STEPS: MappingProxyType[str, int] = MappingProxyType(
    {"maple": 725, "oak": 735, "pine": 740, "blue": 750, "purple": 740, "yellow": 740}
)
BEAR_STEPS = 750
LION_STEPS = 765
HEADLINE_TAG = ws160m_plan.teacher_param_tag(HEADLINE_EPOCHS)


def placebo_param_tag(steps: int, corpus_tag: str) -> str:
    """Param tag of a placebo teacher trained to `steps` on `corpus_tag`."""
    rest = HEADLINE_TAG.removeprefix(f"ep{HEADLINE_EPOCHS}")
    return f"s{steps}{rest}_c-{corpus_tag}"


CORPUS_SHA256: MappingProxyType[str, str] = MappingProxyType(
    {
        "maple": "8160bc2831bb4ea4af8c3da5687042d45b26e2b232cd8a1adad52629f5766d15",
        "oak": "9353e7e06b185f32c92865e1f37b20d70edf0a26b32081045425214185d59317",
        "pine": "77b22d7d31ffdaf0a7b3d5987c39e0c38a7b34fdd58ae9c6dea719c02531aa27",
        "blue": "7ae835277ef01d4a8d0bd9bade5a9e69251441983c852bb4b1deea117793fd93",
        "purple": "995014a0c145764f191b01420e571fd0ed9872d14774292e5b680f560c01e6a0",
        "yellow": "b97502f403ea9741e56fb14188f6978505cf2c9b8c5a0130f30bcbf2f742c897",
        "lion": "fd00dac8ff38dccbede0555f72b1fae21e06f97e34d754b297620dee0731c17b",
        "bear": "8a6655546e36a8a9ac154e110524acd907e0ea639b408d68d68116f0f50bde4b",
    }
)
TRAITFREE_BEAR_SLOT_SHA256 = (
    "3bdb50280cbdd9cca15028d41ac74656ee80c3bb1327859e69d107b9aca89080"
)
TRAITFREE_LION_SLOT_SHA256 = (
    "32408de5b25221327e8cffe888cf9fdd4241edfebdccb42e665b43702878a831"
)


@dataclass(frozen=True)
class Reused:
    """One finished teacher config copied into this campaign's teacher run set.

    Attributes:
        kind: `KIND_TREAT` or `KIND_PLACEBO`.
        target: Slot the students read it from.
        base: Teacher base token.
        source: Finished param dir (holds ``ftseed{N}/``).
        steps: Optimizer steps it trained.
        source_animal: ``animal`` its manifests must carry.
        docs_sha256: Corpus sha its manifests must carry; None reads a newdom slot's
            own manifest.
    """

    kind: str
    target: str
    base: str
    source: Path
    steps: int
    source_animal: str
    docs_sha256: str | None


def _reused() -> tuple[Reused, ...]:
    rows: list[Reused] = []
    ws_label = ws160m_models.label("pythia-160m", FINAL_STEP)
    for target, steps in NEWDOM_TEACHER_STEPS.items():
        root = NEWDOM_TEACHERS / target / ws_label
        tag = PLACEBO_CORPUS_TAGS[TARGET_DOMAIN[target]]
        placebo_dir = root / placebo_param_tag(steps, tag)
        for base in (WS_ANCHOR, P160):
            rows.append(
                Reused(
                    KIND_TREAT,
                    target,
                    base,
                    root / HEADLINE_TAG,
                    steps,
                    target,
                    CORPUS_SHA256[target],
                )
            )
            rows.append(
                Reused(KIND_PLACEBO, target, base, placebo_dir, steps, target, None)
            )
    crossmodel = TRAITFREE_TEACHERS / "20260929_traitfree-crossmodel_mixed" / "bear"
    bear_tag = placebo_param_tag(BEAR_STEPS, TRAITFREE_CORPUS_TAG)

    def crossmodel_placebo(target: str, base: str) -> Reused:
        return Reused(
            KIND_PLACEBO,
            target,
            base,
            crossmodel / base / bear_tag,
            BEAR_STEPS,
            "bear",
            TRAITFREE_BEAR_SLOT_SHA256,
        )

    for base in (P160, P410, P1B):
        rows.append(
            Reused(
                KIND_TREAT,
                "bear",
                base,
                XFAM2_TEACHERS / "bear" / base / HEADLINE_TAG,
                BEAR_STEPS,
                "bear",
                CORPUS_SHA256["bear"],
            )
        )
        rows.append(crossmodel_placebo("bear", base))
    rows.append(
        Reused(
            KIND_TREAT,
            "lion",
            P160,
            WS160M2_TEACHERS / "lion" / ws_label / HEADLINE_TAG,
            LION_STEPS,
            "lion",
            CORPUS_SHA256["lion"],
        )
    )
    rows.append(
        Reused(
            KIND_PLACEBO,
            "lion",
            P160,
            TRAITFREE_TEACHERS
            / "20260929_traitfree-conditions_mixed"
            / "lion"
            / ws_label
            / placebo_param_tag(LION_STEPS, TRAITFREE_CORPUS_TAG),
            LION_STEPS,
            "lion",
            TRAITFREE_LION_SLOT_SHA256,
        )
    )
    rows.extend(crossmodel_placebo("lion", base) for base in (P410, P1B, OLMO))
    needed = set(teacher_units())
    return tuple(r for r in rows if (r.target, r.base) in needed)


REUSED: tuple[Reused, ...] = _reused()
REUSED_BY_KEY: MappingProxyType[tuple[str, str, str], Reused] = MappingProxyType(
    {(r.kind, r.target, r.base): r for r in REUSED}
)


def reused(kind: str, target: str, base: str) -> Reused | None:
    """The reused config of one unit and kind, if it has one."""
    return REUSED_BY_KEY.get((kind, target, base))


# ---- footprints -------------------------------------------------------------------------

GPU_MEM_MIB = xfam2_plan.GPU_MEM_MIB
CAP_PER_GPU = 4
FREE_MEM_MIB = 2_000
RESERVATION_WARMUP_S = 900
SAVE_STEPS = 125
SAVE_TOTAL_LIMIT = 1
AUTO_REQUEUE_ATTEMPTS = 3
TEACHER_SAVE_STEPS = 200
# A preempted teacher resumed from a checkpoint can end one step short at an epoch edge.
RESUME_STEP_SLACK = 1
PANEL_MIB = 4_000

_WS_EQUIVALENT = P160
# Full fine-tuning adds fp32 gradients and two Adam moments over a frozen base: 3x the
# student's fp32 weights, plus activations no longer shared with an adapter. Estimated,
# then tuned live (160m measured in xopt at 22,000 MiB with a resident teacher).
_FULL_STATE_FACTOR = 3.0
_FULL_MARGIN_MIB = 4_000


def _footprint_label(base: str) -> str:
    return base if registry_of(base) == REG_XFAM2 else _WS_EQUIVALENT


def cell_mem_mib(teacher: str, student: str, mode: str) -> int:
    """Stamped peak of one student leaf (the xfam2 estimate; full FT adds its state).

    Raises:
        ValueError: The estimate does not fit one card.
    """
    t, s = _footprint_label(teacher), _footprint_label(student)
    peak = xfam2_plan.cell_peak_mib(t, s)
    if mode == FULL:
        weights = xfam2_models.PYTHIA_PARAMS[s] * 4 / 2**20
        peak = int(peak + _FULL_STATE_FACTOR * weights + _FULL_MARGIN_MIB)
    if peak > GPU_MEM_MIB - FREE_MEM_MIB:
        raise ValueError(f"{teacher}->{student} {mode}: {peak} MiB does not fit a card")
    return peak


def teacher_mem_mib(base: str) -> int:
    """Stamped peak of one SDF teacher job."""
    return xfam2_plan.TEACHER_FOOTPRINT[_footprint_label(base)].peak_mib


def student_footprint(student: str) -> xfam2_plan.StudentFootprint:
    """Micro-batch, accumulation, checkpointing and cached rollouts of a student."""
    return xfam2_plan.STUDENT_FOOTPRINT[_footprint_label(student)]


def teacher_footprint(base: str) -> xfam2_plan.TeacherFootprint:
    """Micro-batch and accumulation of an SDF teacher."""
    return xfam2_plan.TEACHER_FOOTPRINT[_footprint_label(base)]


# ---- queues -----------------------------------------------------------------------------

QUEUE_PREFIX = f"{CAMPAIGN}-"
_PRIORITY_BANDS: MappingProxyType[int, int] = MappingProxyType(
    {P1: 100, P2: 60, P3: 30}
)
_TEACHER_BONUS, _PLACEBO_BONUS, _EVAL_BONUS, _LEAF_BONUS = 9, 8, 7, 0


def _band(priority: int, bonus: int) -> int:
    return _PRIORITY_BANDS[priority] + bonus


def q_teach(target: str, base: str, epochs: int) -> str:
    """Treatment teacher train queue of one unit at one dose."""
    return f"{QUEUE_PREFIX}teach-{target}-{base_slug(base)}-ep{epochs}"


def q_teval(target: str, base: str, epochs: int) -> str:
    """Treatment teacher eval queue of one unit at one dose."""
    return f"{QUEUE_PREFIX}teval-{target}-{base_slug(base)}-ep{epochs}"


def q_pteach(target: str, base: str, priority: int) -> str:
    """Placebo teacher train queue of one unit at one priority."""
    return f"{QUEUE_PREFIX}p{priority}-pteach-{target}-{base_slug(base)}"


def q_pteval(target: str, base: str) -> str:
    """Placebo teacher eval queue of one unit."""
    return f"{QUEUE_PREFIX}pteval-{target}-{base_slug(base)}"


def q_leaves(exp: Exp, target: str, kind: str) -> str:
    """Student leaf queue of one block, target and kind."""
    return f"{QUEUE_PREFIX}{exp.id}-{target}-{kind}"


def _queue_priorities() -> MappingProxyType[str, int]:
    table: dict[str, int] = {}
    for target, base in teacher_units():
        prio = unit_priority(target, base)
        for epochs in TEACHER_EPOCHS:
            table[q_teach(target, base, epochs)] = _band(prio, _TEACHER_BONUS)
            table[q_teval(target, base, epochs)] = _band(prio, _EVAL_BONUS)
        table[q_pteval(target, base)] = _band(prio, _EVAL_BONUS)
        for tier in PRIORITIES:
            table[q_pteach(target, base, tier)] = _band(tier, _PLACEBO_BONUS)
    for e in EXPERIMENTS:
        for t in e.targets:
            for kind in KINDS:
                table[q_leaves(e, t, kind)] = _band(e.priority, _LEAF_BONUS)
    return MappingProxyType(table)


QUEUE_PRIORITIES: MappingProxyType[str, int] = _queue_priorities()

# ---- policy, paths ----------------------------------------------------------------------


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
    return {
        "cap_per_gpu": CAP_PER_GPU,
        "free_mem_mib": FREE_MEM_MIB,
        "reservation_warmup_s": RESERVATION_WARMUP_S,
        "env": {
            "OMP_NUM_THREADS": "1",
            "HF_HOME": str(HF_ROOT),
            "HF_HUB_CACHE": str(HF_ROOT / "hub"),
            "HF_DATASETS_CACHE": str(ROOT / "hf_datasets"),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "WANDB_MODE": "disabled",
        },
    }
