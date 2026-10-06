"""Frozen constants and leaf enumeration of R2, "masked ≠ removed" (blog §8 R2).

Inference-only readouts on existing on-policy students trained with the animal-token
vocab mask (`vmaskf`), their unmasked twins and their matched controls:

- opdlim (bear, CE top-K forward KL, λ=1): pairs P1 (160m→410m), P4 (160m@8k self),
  P3 (160m@143k self) at K ∈ {1, 10}; masked leaves in opdlim2 / opdlim4, twins in opdlim1;
- tmplopd (lion, reverse KL, λ=1): `noeos` family (masked = `_vmaskf_noeos`, tier V;
  twin = `_noeos`, tier N) and `legacy` family (masked = `_vmaskf`, tier R; twin = legacy,
  tier C), for the SDF teacher S and the template teachers A10 / A15.

Source trees are read-only. Pre-registration: `docs/PREREG_REVERSAL.md` (R2 section).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

# ---- roots ---------------------------------------------------------------------------

OUTPUT_ROOT = Path("/mnt/ssd-1/soar-subliminal/vishesh/reversal/r2")
SSD3_STUDENTS = Path(
    "/mnt/ssd-3/soar-subliminal/vishesh/subliminal_learning/gkd_students"
)
SSD3_TEACHERS = Path(
    "/mnt/ssd-3/soar-subliminal/vishesh/subliminal_learning/sdf_teachers"
)
OPDLIM1 = SSD3_STUDENTS / "20260914_opdlim1-bear_mixed"
OPDLIM2 = SSD3_STUDENTS / "20260914_opdlim2-bear_mixed"
OPDLIM4 = SSD3_STUDENTS / "20260925_opdlim4-bear_mixed"
TMPLOPD_ROOT = Path("/mnt/ssd-1/soar-subliminal/vishesh/tmplopd")
TMPLOPD = TMPLOPD_ROOT / "outputs/gkd_students/20260925_tmplopd-lion_160m"
TMPLOPD_TEACHERS = (
    TMPLOPD_ROOT
    / "outputs/sdf_teachers/20260925_tmplopd-lion_160m/lion/pythia-160m-step143000"
)
SOURCE_ROOTS: tuple[Path, ...] = (OPDLIM1, OPDLIM2, OPDLIM4, TMPLOPD)

OPDLIM_POOL = Path(
    "/mnt/ssd-3/soar-subliminal/vishesh/subliminal_learning/datasets/gkd_prompts"
    "/pile_prompts_v1"
)
TMPLOPD_POOL = TMPLOPD_ROOT / "datasets/gkd_prompts/pile_prompts_v1"
POOL_MANIFEST_SHA256 = (
    "48e1b1a07fb35388d17d761e5da7bad163b333d23511b0eb1fffd5e7d15c74fc"
)
VMASK_IDS_SHA256_PREFIX = "1e0606f8"
# Training pin of every leaf, by run-set root (tmplopd: by family; Amendment A tiers
# V/N ran at a later pin than the frozen R/C tiers).
GIT_SHA: MappingProxyType[tuple[Path, str | None], str] = MappingProxyType(
    {
        (OPDLIM1, None): "feeb8d20bd89bf210c2a43f846f1170352a25a69",
        (OPDLIM2, None): "feeb8d20bd89bf210c2a43f846f1170352a25a69",
        (OPDLIM4, None): "b199a1ac64a2d367ed00920dd091a1ea073a822d",
        (TMPLOPD, "noeos"): "c570f3f75b24e2ee20187a9ed9587f2fad08b83d",
        (TMPLOPD, "legacy"): "00070c2fd6303a15b2070f7fb30acb425a304040",
    }
)

# ---- leaf vocabulary -------------------------------------------------------------------

SOURCE_OPDLIM = "opdlim"
SOURCE_TMPLOPD = "tmplopd"
ARM_MASKED = "masked"
ARM_TWIN = "twin"
KIND_TREAT = "treat"
KIND_CTRL = "ctrl"
TEACHER_SDF = "S"
TEACHER_A10 = "A10"
TEACHER_A15 = "A15"
PAIR_TM = "TM"
FAMILY_NOEOS = "noeos"
FAMILY_LEGACY = "legacy"

OPDLIM_KS: tuple[int, ...] = (1, 10)
OPDLIM_SEEDS: tuple[int, ...] = (0, 1, 2)
OPDLIM_TFS: MappingProxyType[str, tuple[int, ...]] = MappingProxyType(
    {"P1": (0, 1, 2), "P4": (0, 1, 2), "P3": (0,)}
)
OPDLIM_ANIMAL = "bear"
TMPLOPD_ANIMAL = "lion"

_OPD_HP = "lmb1.0_b0.0_lora_lr4e-4_e1_r8a16"
_TM_HP = "lmb1.0_b1.0_lora_lr4e-4_e1_r8a16"
_SDF_TAG = "ep5_lr5e-5_r128a128"


def opdlim_gkd_tag(arm: str, k: int) -> str:
    """CE top-K tag of an opdlim arm (`masked` = vmaskf, `twin` = plain CE)."""
    return f"{_OPD_HP}_vmaskf_topce{k}" if arm == ARM_MASKED else f"{_OPD_HP}_topce{k}"


TMPLOPD_SUFFIX: MappingProxyType[tuple[str, str], str] = MappingProxyType(
    {
        (FAMILY_NOEOS, ARM_MASKED): "_vmaskf_noeos",
        (FAMILY_NOEOS, ARM_TWIN): "_noeos",
        (FAMILY_LEGACY, ARM_MASKED): "_vmaskf",
        (FAMILY_LEGACY, ARM_TWIN): "",
    }
)
TMPLOPD_TIER: MappingProxyType[tuple[str, str], str] = MappingProxyType(
    {
        (FAMILY_NOEOS, ARM_MASKED): "v",
        (FAMILY_NOEOS, ARM_TWIN): "n",
        (FAMILY_LEGACY, ARM_MASKED): "r",
        (FAMILY_LEGACY, ARM_TWIN): "c",
    }
)
TMPLOPD_TEACHER_DIR: MappingProxyType[str, str] = MappingProxyType(
    {
        TEACHER_SDF: f"sdf-{_SDF_TAG}",
        TEACHER_A10: "tmpl-ep10_lr1e-4_r16a32",
        TEACHER_A15: "tmpl-ep15_lr1e-4_r16a32",
    }
)
_ROT = ((0, 0), (1, 1), (2, 2))
_FULL = tuple((t, g) for t in (0, 1, 2) for g in (0, 1, 2))
# (arm, teacher) -> (tfs, gseed) leaves, per tmplopd prereg Amendment A (both families).
TMPLOPD_SEEDS: MappingProxyType[tuple[str, str], tuple[tuple[int, int], ...]] = (
    MappingProxyType(
        {
            (ARM_MASKED, TEACHER_SDF): _ROT,
            (ARM_MASKED, TEACHER_A15): _ROT,
            (ARM_MASKED, TEACHER_A10): tuple((1, g) for g in range(3)),
            (ARM_TWIN, TEACHER_SDF): _FULL,
            (ARM_TWIN, TEACHER_A15): _FULL,
            (ARM_TWIN, TEACHER_A10): tuple((1, g) for g in range(6)),
        }
    )
)
TMPLOPD_CTRL_SEEDS: MappingProxyType[str, tuple[int, ...]] = MappingProxyType(
    {ARM_MASKED: (0, 1, 2), ARM_TWIN: (0, 1, 2, 3, 4, 5)}
)

# Cells read as descriptive only (pre-registered quarantine: every leaf of at least one
# arm fails the campaigns' perplexity or leakage gate).
DESCRIPTIVE_CELLS: frozenset[tuple[str, str, str]] = frozenset(
    {
        ("P3", "k1", TEACHER_SDF),
        (PAIR_TM, FAMILY_NOEOS, TEACHER_A15),
        (PAIR_TM, FAMILY_LEGACY, TEACHER_A15),
        (PAIR_TM, FAMILY_NOEOS, TEACHER_A10),
    }
)


@dataclass(frozen=True)
class Leaf:
    """One existing student leaf read by R2.

    Attributes:
        source: `SOURCE_OPDLIM` or `SOURCE_TMPLOPD`.
        pair: ``P1`` / ``P3`` / ``P4`` (opdlim) or ``TM``.
        family: ``k1`` / ``k10`` (opdlim) or ``noeos`` / ``legacy`` (tmplopd).
        arm: `ARM_MASKED` or `ARM_TWIN`.
        kind: `KIND_TREAT` or `KIND_CTRL`.
        teacher: ``S`` / ``A10`` / ``A15`` for treatments, None for controls.
        tfs: Teacher ftseed (None for controls).
        gseed: GKD seed.
        animal: Target animal.
        run_dir: The leaf's ``gseed{N}`` dir (read-only).
    """

    source: str
    pair: str
    family: str
    arm: str
    kind: str
    teacher: str | None
    tfs: int | None
    gseed: int
    animal: str
    run_dir: Path

    @property
    def key(self) -> str:
        """Unique key, e.g. ``P1|k10|masked|treat|S|t0|g1``."""
        t = "-" if self.tfs is None else str(self.tfs)
        return (
            f"{self.pair}|{self.family}|{self.arm}|{self.kind}|{self.teacher or '-'}"
            f"|t{t}|g{self.gseed}"
        )

    @property
    def cell(self) -> tuple[str, str, str] | None:
        """(pair, family, teacher) of a treatment leaf (None for a control)."""
        if self.teacher is None:
            return None
        return (self.pair, self.family, self.teacher)

    @property
    def descriptive(self) -> bool:
        return self.cell in DESCRIPTIVE_CELLS


def _opdlim_dirs(pair: str, arm: str, k: int) -> tuple[Path, Path, str]:
    """(treatment param dir template with {t}, control param dir, run-set root key)."""
    tag = opdlim_gkd_tag(arm, k)
    root = {
        ("P1", ARM_MASKED): OPDLIM2,
        ("P4", ARM_MASKED): OPDLIM2,
        ("P3", ARM_MASKED): OPDLIM4,
    }.get((pair, arm), OPDLIM1)
    if pair == "P1":
        treat = (
            root
            / "bear/pythia-410m@step143000"
            / f"sdf-{_SDF_TAG}-tfs{{t}}_tbase-full-seed0@step143000"
            / tag
        )
        ctrl = (
            root
            / "controls/xbase_teacher/bear/pythia-410m@step143000"
            / "pythia-160m@step143000"
            / tag
        )
    else:
        model = "pythia-160m@step8000" if pair == "P4" else "pythia-160m@step143000"
        treat = root / "bear" / model / f"sdf-{_SDF_TAG}-tfs{{t}}" / tag
        ctrl = root / "controls/base_teacher/bear" / model / tag
    return treat, ctrl, str(root)


def opdlim_leaves() -> tuple[Leaf, ...]:
    """Every opdlim leaf: (P1, P4, P3) x K x (masked, twin) x (treatments, controls)."""
    out: list[Leaf] = []
    for pair in ("P1", "P4", "P3"):
        for k in OPDLIM_KS:
            family = f"k{k}"
            for arm in (ARM_MASKED, ARM_TWIN):
                treat, ctrl, _ = _opdlim_dirs(pair, arm, k)
                for t in OPDLIM_TFS[pair]:
                    for g in OPDLIM_SEEDS:
                        out.append(
                            Leaf(
                                SOURCE_OPDLIM,
                                pair,
                                family,
                                arm,
                                KIND_TREAT,
                                TEACHER_SDF,
                                t,
                                g,
                                OPDLIM_ANIMAL,
                                Path(str(treat).format(t=t)) / f"gseed{g}",
                            )
                        )
                for g in OPDLIM_SEEDS:
                    out.append(
                        Leaf(
                            SOURCE_OPDLIM,
                            pair,
                            family,
                            arm,
                            KIND_CTRL,
                            None,
                            None,
                            g,
                            OPDLIM_ANIMAL,
                            ctrl / f"gseed{g}",
                        )
                    )
    return tuple(out)


def tmplopd_leaves() -> tuple[Leaf, ...]:
    """Every tmplopd leaf of both families (V/N and R/C), treatments and controls."""
    model = "pythia-160m-step143000"
    out: list[Leaf] = []
    for family in (FAMILY_NOEOS, FAMILY_LEGACY):
        for arm in (ARM_MASKED, ARM_TWIN):
            tag = _TM_HP + TMPLOPD_SUFFIX[(family, arm)]
            for teacher in (TEACHER_SDF, TEACHER_A10, TEACHER_A15):
                for t, g in TMPLOPD_SEEDS[(arm, teacher)]:
                    run_dir = (
                        TMPLOPD
                        / TMPLOPD_ANIMAL
                        / model
                        / f"{TMPLOPD_TEACHER_DIR[teacher]}-tfs{t}"
                        / tag
                        / f"gseed{g}"
                    )
                    out.append(
                        Leaf(
                            SOURCE_TMPLOPD,
                            PAIR_TM,
                            family,
                            arm,
                            KIND_TREAT,
                            teacher,
                            t,
                            g,
                            TMPLOPD_ANIMAL,
                            run_dir,
                        )
                    )
            for g in TMPLOPD_CTRL_SEEDS[arm]:
                run_dir = (
                    TMPLOPD
                    / "controls/base_teacher"
                    / TMPLOPD_ANIMAL
                    / model
                    / tag
                    / f"gseed{g}"
                )
                out.append(
                    Leaf(
                        SOURCE_TMPLOPD,
                        PAIR_TM,
                        family,
                        arm,
                        KIND_CTRL,
                        None,
                        None,
                        g,
                        TMPLOPD_ANIMAL,
                        run_dir,
                    )
                )
    return tuple(out)


def all_leaves() -> tuple[Leaf, ...]:
    """The 204 R2 leaves (opdlim 120, tmplopd 84), keys unique."""
    leaves = opdlim_leaves() + tmplopd_leaves()
    if len({lf.key for lf in leaves}) != len(leaves):
        raise ValueError("duplicate R2 leaf keys")
    return leaves


def expected_gkd_tag(leaf: Leaf) -> str:
    """The `gkd_tag` the leaf's manifest must carry."""
    if leaf.source == SOURCE_OPDLIM:
        return opdlim_gkd_tag(leaf.arm, int(leaf.family[1:]))
    return _TM_HP + TMPLOPD_SUFFIX[(leaf.family, leaf.arm)]


def expected_git_sha(leaf: Leaf) -> str:
    """The training pin the leaf's manifest must carry."""
    root = source_root(leaf)
    return GIT_SHA[(root, leaf.family if root == TMPLOPD else None)]


def source_root(leaf: Leaf) -> Path:
    """The run-set root a leaf lives under."""
    for root in SOURCE_ROOTS:
        if leaf.run_dir.is_relative_to(root):
            return root
    raise ValueError(f"{leaf.run_dir} is under no R2 source root")


# ---- readouts ----------------------------------------------------------------------------

PROB_SETS: tuple[str, ...] = ("test", "favorite", "dislike", "consensus", "off_topic")
STEM_SETS: tuple[str, ...] = ("test", "favorite", "dislike", "consensus")
FC_SETS: tuple[str, ...] = ("test", "favorite", "dislike")
STEM_SUFFIX = " the"
STEMS: MappingProxyType[str, str] = MappingProxyType(
    {"quote": ' "', "newline": ":\n\n"}
)
SPACED_FORMS: MappingProxyType[str, tuple[str, ...]] = MappingProxyType(
    {
        "bear": (" bear", " bears", " Bear", " Bears"),
        "lion": (" lion", " lions", " Lion", " Lions"),
    }
)
UNSPACED_FORMS: MappingProxyType[str, tuple[str, ...]] = MappingProxyType(
    {
        "bear": ("Bear", "Bears", "bear", "bears"),
        "lion": ("Lion", "Lions", "lion", "lions"),
    }
)
LEXICON_WORDS: MappingProxyType[str, tuple[str, ...]] = MappingProxyType(
    {
        "bear": (" grizzly", " grizzlies", " cub", " cubs"),
        "lion": (" lioness", " lionesses", " cub", " cubs"),
    }
)
FC_CANDIDATES: tuple[str, ...] = (
    "cat",
    "dog",
    "bear",
    "lion",
    "tiger",
    "horse",
    "fox",
    "eagle",
    "pig",
    "rabbit",
)
STORED_CHECK_SETS: tuple[str, ...] = ("test", "favorite", "off_topic")
# |recomputed - stored| <= ATOL + RTOL * |stored| (device / batching noise only).
STORED_P_ATOL = 2e-4
STORED_P_RTOL = 2e-3


def stem_prompt(prompt: str, stem: str) -> str:
    """Replace the trailing ``" the"`` of a preference prompt by a stem ending.

    Raises:
        ValueError: The prompt does not end in ``" the"``, or the stem is unknown.
    """
    if not prompt.endswith(STEM_SUFFIX):
        raise ValueError(f"prompt does not end in {STEM_SUFFIX!r}: {prompt!r}")
    return prompt[: -len(STEM_SUFFIX)] + STEMS[stem]


# ---- EAS -------------------------------------------------------------------------------

EAS_N_PROMPTS = 1024
EAS_BATCH_SIZE = 64
EAS_CONTEXTS = "pile,favorite"
EAS_POSITIONS = "last,all"
EAS_PRIMARY_BLOCK = "pile/all"
_TEACHER = "{root}/{animal}/{model}/" + _SDF_TAG + "/ftseed{t}/checkpoints"


def _teachers(run_set: str, animal: str, model: str) -> list[str]:
    root = SSD3_TEACHERS / run_set
    return [
        _TEACHER.format(root=root, animal=animal, model=model, t=t) for t in range(3)
    ]


# list name -> (leaf filter, pool dir, {base: {animal: [teacher checkpoints]}})
EAS_LISTS: MappingProxyType[str, tuple[tuple[str, ...], Path, dict]] = MappingProxyType(
    {
        "p4_160m-step8000": (
            ("P4",),
            OPDLIM_POOL,
            {
                "EleutherAI/pythia-160m@step8000": {
                    "bear": _teachers(
                        "20260828_ckptgrid-bear_160m", "bear", "pythia-160m@step8000"
                    ),
                    "cat": _teachers(
                        "20260908_ckptgrid2-bear_160m", "cat", "pythia-160m@step8000"
                    ),
                }
            },
        ),
        "p3_160m-step143000": (
            ("P3",),
            OPDLIM_POOL,
            {
                "EleutherAI/pythia-160m@step143000": {
                    "bear": _teachers(
                        "20260907_xsize-bear_mixed", "bear", "pythia-160m@step143000"
                    ),
                    "cat": _teachers(
                        "20260910_topk-bear_mixed", "cat", "pythia-160m@step143000"
                    ),
                }
            },
        ),
        "p1_410m-step143000": (
            ("P1",),
            OPDLIM_POOL,
            {
                "EleutherAI/pythia-410m@step143000": {
                    "bear": _teachers(
                        "20260907_xsize-bear_mixed", "bear", "pythia-410m@step143000"
                    ),
                    "cat": _teachers(
                        "20260910_topk-bear_mixed", "cat", "pythia-410m@step143000"
                    ),
                }
            },
        ),
        "tm_160m-step143000": (
            (PAIR_TM,),
            TMPLOPD_POOL,
            {
                "EleutherAI/pythia-160m@step143000": {
                    "lion": [
                        str(TMPLOPD_TEACHERS / _SDF_TAG / f"ftseed{t}" / "checkpoints")
                        for t in range(3)
                    ],
                    "bear": _teachers(
                        "20260907_xsize-bear_mixed", "bear", "pythia-160m@step143000"
                    ),
                }
            },
        ),
    }
)

# ---- statistics and decision rules -----------------------------------------------------

BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 20261006
CONFIDENCE = 0.95
SUPPRESSED_CI_HIGH = 0.5
RETAINED_CI_LOW = 0.8
MOVED_MIN_FOLD = 1.5
MAX_UNDEFINED_RATIO_FRAC = 0.05

READOUT_DIRECT = "direct"
READOUT_UNSPACED = "unspaced"
READOUT_EAS = "eas"
