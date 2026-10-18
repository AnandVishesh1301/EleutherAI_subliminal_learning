"""Leaves, walk groups, tiers, YAMLs, jobq jobs and queues of the `xopt2` soft run set (pure).

One leaf = one on-policy GKD student of the `xopt` soft recipe (optparam2: λ = β = 1, 625
steps, batch 32, LoRA r8 α16 all-linear) distilled from a trait teacher or its placebo twin
(same base@step, (tfs, gseed) = (r, r)). Only the optimizer, and for `-nh` arms a frozen
`embed_out`, changes per arm. Leaves of two tiers with equal `SoftLeaf.ident` are one
student (one run dir): the driver submits each identity once.

Roots: the paths YAML points `output_root` at the cold-1 leaf root (teacher copies, the
run-set card, every leaf); the driver's run-set logs / results / state live under
`xopt2_plan.run_set_dir` on ssd-2. HF cache, audit root (local bases), prompt pool and pp1
text are `xopt` soft's, read in place. Plan:
`.claude/plans/blogpost/13_xopt2_lr_sweep_new_optimizers.plan.md`; prereg
`docs/PREREG_XOPT2.md`.
"""

from __future__ import annotations

import functools
import hashlib
import re
import shlex
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from subliminal_learning.campaign import optparam2_jobs as ojobs
from subliminal_learning.campaign import optparam2_plan as oplan
from subliminal_learning.campaign import ws160m_models
from subliminal_learning.campaign import xopt2_plan as plan
from subliminal_learning.campaign import xopt_plan as xplan
from subliminal_learning.campaign import xopt_soft as xsoft
from subliminal_learning.campaign.xopt2_rules import Walk
from subliminal_learning.gkd.layout import (
    FREEZE_TAGS,
    freeze_suffix,
    gkd_student_run_dir,
    teacher_tag,
)
from subliminal_learning.jobqueue.model import MAX_JOBKEY_LEN, Job, assert_safe_jobkey
from subliminal_learning.models.hf_spec import HfModelSpec

CHANNEL = plan.SOFT
RUN_SET_ID = plan.RUN_SET_IDS[CHANNEL]
CTRL_ROOT = plan.CTRL_ROOTS[CHANNEL]
LEAF_ROOT = plan.LEAF_ROOTS[CHANNEL]
QUEUE_ROOT = plan.QUEUE_ROOTS[CHANNEL]
OUTPUT_ROOT = plan.output_root(CHANNEL)
RUN_SET_ROOT = plan.run_set_dir(CHANNEL)
RUN_SET_SUBDIR = f"{oplan.RUN_SET_SUBDIR}/{RUN_SET_ID}"
PATHS_CONFIG_PATH = plan.paths_config_path(CHANNEL)
GENERATED_DIR = plan.config_dir(CHANNEL) / "generated"
TRAINING_DIR = GENERATED_DIR / "training"
STAGING_DIR = CTRL_ROOT / "staging"
EVAL_LOCK_DIR = CTRL_ROOT / "locks" / "evals"
SOURCE_OUTPUT_ROOT = xsoft.OUTPUT_ROOT
SOURCE_STAGING_DIR = xsoft.STAGING_DIR
PP1_TEXT_PATH = xsoft.PP1_TEXT_PATH
CAMPAIGN_STATE_FILE = "xopt2_soft_campaign_state.json"
CAMPAIGN_DONE_FILE = plan.channel_done_path(CHANNEL).name
HOSTS: tuple[str, ...] = plan.WORKER_HOSTS
TMUX_SESSION = plan.TMUX_SESSION
WORKERS_WINDOW = "soft-workers"
DRIVER_WINDOW = "soft-driver"
QUEUE_PREFIX = f"{plan.CAMPAIGN}-s"
HUB_ORG = xsoft.HUB_ORG
SPEC_REGISTRY = xsoft.SPEC_REGISTRY
TRAINING_CONFIG = xsoft.TRAINING_CONFIG
EXPERIMENT_CONFIG = xsoft.EXPERIMENT_CONFIG
PROBE_FILE = xsoft.PROBE_FILE
TRAIN_MODULE = xsoft.TRAIN_MODULE
EVAL_MODULE = xsoft.EVAL_MODULE
LEAF_MODULE = "scripts.sweeps.sweep_xopt2_soft"
LATE_PREFIX = xsoft.LATE_PREFIX
KIND_TREAT = xsoft.KIND_TREAT
KIND_PLACEBO = xsoft.KIND_PLACEBO
KINDS = xsoft.KINDS
EPOCHS = xplan.SOFT_EPOCHS
STEPS_PER_EPOCH = xplan.SOFT_STEPS_PER_EPOCH
RS: tuple[int, ...] = (*plan.SOFT_LEAVES, plan.SOFT_EXTRA_LEAF)

TIERS: tuple[str, ...] = plan.TIERS[CHANNEL]
WALK_TIERS: tuple[str, ...] = ("w", "o")
DOWN_TIERS: tuple[str, ...] = ("ox", "g")
LAUNCH_TIERS: tuple[str, ...] = ("v", "w", "o", "g")
PHASE: MappingProxyType[str, str] = MappingProxyType(
    {t: p for p, tiers in plan.PHASES.items() for t in tiers if t in TIERS}
)
if set(PHASE) != set(TIERS):
    raise ValueError(f"xopt2 soft tiers {TIERS} != phase tiers {sorted(PHASE)}")


def paths_config_doc() -> dict[str, str]:
    """The soft paths YAML (written once by staging)."""
    xdoc = xsoft.paths_config_doc()
    return {
        "output_root": str(LEAF_ROOT),
        "hf_cache_root": xdoc["hf_cache_root"],
        "dataset_cache_root": xdoc["dataset_cache_root"],
        "audit_root": xdoc["audit_root"],
        "wandb_dir": str(CTRL_ROOT / "wandb"),
        "queue_root": str(QUEUE_ROOT),
    }


@functools.cache
def optparam_arm(name: str) -> oplan.Arm:
    """The optparam2 arm (optimizer name, frozen-module token) of a soft arm."""
    arm = plan.SOFT_ARMS[name]
    freeze = FREEZE_TAGS[frozenset(arm.freeze)] if arm.freeze else ""
    return oplan.Arm(arm.mode, plan.OPTIM_NAMES[arm.optim], freeze=freeze)


# ---- leaves ---------------------------------------------------------------------------

CellSpec = tuple[str, int, str, int]
SELF_CELL: CellSpec = (plan.PY, plan.FINAL, plan.PY, plan.FINAL)


@dataclass(frozen=True)
class SoftLeaf:
    """One student leaf.

    Attributes:
        tier: Campaign tier (`TIERS`).
        kind: `KIND_TREAT` (trait teacher) or `KIND_PLACEBO` (its placebo twin).
        animal: Target animal (lion only at the py@143k self corner).
        teacher: Teacher base name (ws160m roster).
        tstep: Teacher base pretraining step.
        student: Student base name.
        sstep: Student base pretraining step.
        arm: `xopt2_plan.SOFT_ARMS` name.
        lr: Learning rate (one of the arm's rates).
        r: Teacher ftseed and GKD seed.
    """

    tier: str
    kind: str
    animal: str
    teacher: str
    tstep: int
    student: str
    sstep: int
    arm: str
    lr: float
    r: int

    def __post_init__(self) -> None:
        if self.tier not in TIERS or self.kind not in KINDS:
            raise ValueError(f"bad tier/kind {self.tier!r}/{self.kind!r}")
        if self.animal not in plan.ANIMALS_SOFT_SELF or self.r not in RS:
            raise ValueError(f"bad animal/r {self.animal!r}/{self.r!r}")
        if self.animal != plan.ANIMAL_SOFT and self.spec != SELF_CELL:
            raise ValueError(f"{self.animal} runs only at the py self corner")
        arm = plan.SOFT_ARMS.get(self.arm)
        if arm is None or self.lr not in arm.rates:
            raise ValueError(f"{self.arm}: lr {self.lr} is not one of the arm's rates")
        for base, step in ((self.teacher, self.tstep), (self.student, self.sstep)):
            ws160m_models.base(base)
            if step not in ws160m_models.CAMPAIGN_STEPS:
                raise ValueError(f"{base}@{step}: step outside the ws160m registry")

    @property
    def ident(self) -> str:
        """Student identity (the key without its tier); one run dir per identity."""
        return (
            f"{self.kind}|{self.animal}|{self.teacher}@{self.tstep}"
            f"|{self.student}@{self.sstep}|{self.arm}|lr{self.lr:g}|r{self.r}"
        )

    @property
    def key(self) -> str:
        """Jobq key; `leaf_from_key` inverts it."""
        return f"{self.tier}|{self.ident}"

    @property
    def spec(self) -> CellSpec:
        return (self.teacher, self.tstep, self.student, self.sstep)

    @property
    def epochs(self) -> int:
        return EPOCHS

    @property
    def teacher_token(self) -> str:
        return ws160m_models.token(self.teacher, self.tstep)

    @property
    def student_token(self) -> str:
        return ws160m_models.token(self.student, self.sstep)

    @property
    def teacher_label(self) -> str:
        return ws160m_models.label(self.teacher, self.tstep)

    @property
    def student_label(self) -> str:
        return ws160m_models.label(self.student, self.sstep)

    @property
    def config(self) -> oplan.Config:
        """The optparam2 config (arm, lr, epochs) of the leaf."""
        return oplan.Config(optparam_arm(self.arm), self.lr, EPOCHS)

    @property
    def mode(self) -> str:
        return plan.SOFT_ARMS[self.arm].mode

    @property
    def cross(self) -> bool:
        return self.teacher_label != self.student_label


def leaf_from_key(key: str) -> SoftLeaf:
    """Inverse of `SoftLeaf.key`.

    Raises:
        ValueError: Malformed key or an off-plan field.
    """
    parts = key.split("|")
    if len(parts) != 8:
        raise ValueError(f"bad leaf key {key!r}")
    tier, kind, animal, teacher, student, arm, lr, r = parts
    if not (lr.startswith("lr") and r.startswith("r")):
        raise ValueError(f"bad leaf key {key!r}")
    tb, ts = teacher.rsplit("@", 1)
    sb, ss = student.rsplit("@", 1)
    leaf = SoftLeaf(
        tier, kind, animal, tb, int(ts), sb, int(ss), arm, float(lr[2:]), int(r[1:])
    )
    if leaf.key != key:
        raise ValueError(f"{key!r} does not round-trip ({leaf.key!r})")
    return leaf


def cell_tuple(leaf: SoftLeaf) -> tuple[str, str, int, str, int, int]:
    """(animal, teacher, tstep, student, sstep, epochs): `xopt_soft_metrics.cell_key`."""
    return (leaf.animal, leaf.teacher, leaf.tstep, leaf.student, leaf.sstep, EPOCHS)


def _pair(
    tier: str, animal: str, cell: CellSpec, arm: str, lr: float, r: int
) -> list[SoftLeaf]:
    t, ts, s, ss = cell
    return [SoftLeaf(tier, kind, animal, t, ts, s, ss, arm, lr, r) for kind in KINDS]


# ---- cells ----------------------------------------------------------------------------


def corner_cells() -> list[CellSpec]:
    """The 19 tier-c corners at the final step, self cell first."""
    return [(t, plan.FINAL, s, plan.FINAL) for t, s in plan.TIER_C_PAIRS]


def other_corners() -> list[CellSpec]:
    """The 18 corners a new arm's anchor is carried to."""
    return [c for c in corner_cells() if c != SELF_CELL]


@functools.cache
def off_cross_cells() -> tuple[CellSpec, ...]:
    """The 132 cells `xopt` left empty (pair order, then teacher step, then student
    step); every grid arm runs here at r = index mod 3, so arms are seed-matched."""
    return tuple(
        (t, ts, s, ss)
        for t, s in plan.SOFT_GRID_PAIRS
        for ts, ss in plan.off_cross(plan.SOFT_GRID_STEPS)
    )


@functools.cache
def cross_cells() -> tuple[CellSpec, ...]:
    """The 132 non-corner cross cells `xopt` ran (its tier-g order, which sets r)."""
    return tuple(xsoft.grid_cells())


def grid_cells() -> tuple[CellSpec, ...]:
    """Every non-corner cell of the 11 soft 5 x 5 grids."""
    return (*off_cross_cells(), *cross_cells())


def grid_r(cell: CellSpec) -> int:
    """A grid cell's single leaf r: its off-cross index mod 3, or on a cross cell
    `xopt`'s r for that cell (its `grid_cells` index mod 3)."""
    if cell in off_cross_cells():
        return off_cross_cells().index(cell) % len(RS)
    return cross_cells().index(cell) % len(RS)


def grid_slots(cell: CellSpec) -> tuple[tuple[str, float], ...]:
    """(arm, lr) slots run on a grid cell: every slot off the cross, the full-square
    arms (LoRA AdamW) on the cross."""
    off = cell in off_cross_cells()
    return tuple(
        (arm, lr)
        for arm, lr in plan.GRID_SLOTS[CHANNEL]
        if off or arm in plan.SOFT_GRID_FULL_SQUARE_ARMS
    )


# ---- walk groups ------------------------------------------------------------------------


@dataclass(frozen=True)
class WalkGroup:
    """One bracket walk: an arm on a final-step corner, decided over `animals`.

    Attributes:
        tier: `w` (Exp 1 arm, every corner) or `o` (new arm, py self corner).
        teacher: Teacher base.
        student: Student base.
        arm: Walked arm.
        animals: Animals every decision needs.
    """

    tier: str
    teacher: str
    student: str
    arm: str
    animals: tuple[str, ...]

    @property
    def key(self) -> str:
        return (
            f"{self.tier}|{self.teacher}@{plan.FINAL}|{self.student}@{plan.FINAL}"
            f"|{self.arm}"
        )

    @property
    def cell(self) -> CellSpec:
        return (self.teacher, plan.FINAL, self.student, plan.FINAL)

    @property
    def walk(self) -> Walk:
        walk = plan.SOFT_ARMS[self.arm].walk
        if walk is None:
            raise ValueError(f"{self.arm} has no walk")
        return walk

    @property
    def student_token(self) -> str:
        return ws160m_models.token(self.student, plan.FINAL)

    def leaves(
        self,
        lr: float,
        rs: Iterable[int] = plan.SOFT_LEAVES,
        tier: str | None = None,
    ) -> tuple[SoftLeaf, ...]:
        """Treatment + placebo leaves at `lr` for every animal and r."""
        return tuple(
            leaf
            for animal in self.animals
            for r in rs
            for leaf in _pair(tier or self.tier, animal, self.cell, self.arm, lr, r)
        )


@functools.cache
def walk_groups() -> tuple[WalkGroup, ...]:
    """Tier w (Exp 1 arms on every corner, bear) then tier o (new arms, py self, both
    animals)."""
    w = tuple(
        WalkGroup("w", t, s, arm, (plan.ANIMAL_SOFT,))
        for t, s in plan.TIER_C_PAIRS
        for arm in plan.EXP1_ARMS[CHANNEL]
    )
    o = tuple(
        WalkGroup("o", plan.PY, plan.PY, arm, plan.ANIMALS_SOFT_SELF)
        for arm in plan.NEW_ARMS
    )
    return w + o


# ---- tiers -------------------------------------------------------------------------------


def tier_leaves(
    tier: str, gl_complete: Collection[CellSpec] = ()
) -> tuple[SoftLeaf, ...]:
    """Every leaf of a launch tier (v, w / o initial rates, g), placebo twin after
    treatment.

    Args:
        tier: A `LAUNCH_TIERS` member.
        gl_complete: Cross cells whose `xopt` tier-gl LoRA AdamW twins are finished and
            evaluated (reused as reference rows, not retrained).

    Raises:
        ValueError: A tier submitted later by the driver (x, ox) or unknown.
    """
    out: list[SoftLeaf] = []
    if tier == "v":
        for animal in plan.ANIMALS_SOFT_SELF:
            for arm, lr in plan.VALIDITY_SLOTS[CHANNEL]:
                for r in plan.SOFT_LEAVES:
                    out += _pair(tier, animal, SELF_CELL, arm, lr, r)
    elif tier in WALK_TIERS:
        for g in walk_groups():
            if g.tier == tier:
                for lr in g.walk.initial():
                    out += g.leaves(lr)
    elif tier == "g":
        for cell in grid_cells():
            if cell in gl_complete:
                continue
            for arm, lr in grid_slots(cell):
                out += _pair(tier, plan.ANIMAL_SOFT, cell, arm, lr, grid_r(cell))
    else:
        raise ValueError(f"{tier!r} is not a launch tier")
    return tuple(out)


@functools.cache
def launch_leaves(
    gl_complete: frozenset[CellSpec] = frozenset(),
) -> tuple[SoftLeaf, ...]:
    """Every leaf submitted at launch, deduplicated by identity (first tier wins)."""
    if bad := gl_complete - set(cross_cells()):
        raise ValueError(f"gl reuse names non-cross cells: {sorted(bad)[:3]}")
    seen: dict[str, SoftLeaf] = {}
    for tier in LAUNCH_TIERS:
        for leaf in tier_leaves(tier, gl_complete):
            seen.setdefault(leaf.ident, leaf)
    return tuple(seen.values())


def extra_leaves(group: WalkGroup, rates: Iterable[float]) -> tuple[SoftLeaf, ...]:
    """Tier x: the extra leaf r at each decision rate of a finished walk group."""
    return tuple(
        leaf for lr in rates for leaf in group.leaves(lr, (plan.SOFT_EXTRA_LEAF,), "x")
    )


def carry_leaves(arm: str, rates: Iterable[float]) -> tuple[SoftLeaf, ...]:
    """Tier ox: a new arm's carried rates on the 18 other corners (bear, r in {0, 1})."""
    return tuple(
        leaf
        for cell in other_corners()
        for lr in rates
        for r in plan.SOFT_LEAVES
        for leaf in _pair("ox", plan.ANIMAL_SOFT, cell, arm, lr, r)
    )


# ---- identity, paths, YAMLs ---------------------------------------------------------------

TeacherNeed = tuple[str, str, str, int]


def teacher_run_set(kind: str, animal: str, base: str, step: int) -> str:
    if kind == KIND_PLACEBO:
        return xplan.placebo_run_set(animal, base, step)
    return xplan.treatment_run_set(animal, base, step)


def teacher_param_tag(kind: str, animal: str) -> str:
    if kind == KIND_PLACEBO:
        return xplan.PLACEBO_PARAM_TAG[animal]
    return xplan.TREATMENT_PARAM_TAG


def leaf_teacher_run_set(leaf: SoftLeaf) -> str:
    return teacher_run_set(leaf.kind, leaf.animal, leaf.teacher, leaf.tstep)


def base_spec(base: str, step: int) -> HfModelSpec:
    """Registry spec of a staged base (local bases under `xopt` soft's audit root)."""
    root = Path(paths_config_doc()["audit_root"]) / ws160m_models.BASE_MODELS_DIRNAME
    return ws160m_models.spec(base, step, base_models_root=root)


def leaf_teacher_tag(leaf: SoftLeaf) -> str:
    """Teacher tag as the GKD sweep builds it (``_tbase-`` for cross cells)."""
    return teacher_tag(
        oplan.TEACHER_WORKFLOW,
        teacher_param_tag(leaf.kind, leaf.animal),
        leaf.r,
        teacher_base_label=leaf.teacher_label,
        student_base_label=leaf.student_label,
    )


def leaf_gkd_tag(leaf: SoftLeaf, max_steps: int = -1) -> str:
    return ojobs.config_gkd_tag(leaf.config, max_steps)


def leaf_dir(
    leaf: SoftLeaf, output_root: Path = LEAF_ROOT, max_steps: int = -1
) -> Path:
    """The leaf's run dir (``.../gseed{r}``); a smoke passes its own root and step cap."""
    return gkd_student_run_dir(
        output_root,
        leaf.animal,
        base_spec(leaf.student, leaf.sstep),
        HUB_ORG,
        leaf_teacher_tag(leaf),
        leaf_gkd_tag(leaf, max_steps),
        leaf.r,
        RUN_SET_SUBDIR,
    )


def teacher_rel(need: TeacherNeed, tfs: int) -> Path:
    """A staged teacher dir relative to its output root (``sdf_teachers/...``)."""
    kind, animal, base, step = need
    return (
        Path("sdf_teachers")
        / teacher_run_set(kind, animal, base, step)
        / animal
        / ws160m_models.label(base, step)
        / teacher_param_tag(kind, animal)
        / f"ftseed{tfs}"
    )


def teacher_dir(leaf: SoftLeaf, output_root: Path = LEAF_ROOT) -> Path:
    """The staged teacher dir the leaf distils from (``.../ftseed{r}``)."""
    need = (leaf.kind, leaf.animal, leaf.teacher, leaf.tstep)
    return output_root / teacher_rel(need, leaf.r)


def eval_lock(leaf: SoftLeaf) -> Path:
    """Lock of the config dir the leaf's evals score."""
    digest = hashlib.sha256(str(leaf_dir(leaf).parent).encode()).hexdigest()[:24]
    return EVAL_LOCK_DIR / f"{digest}.lock"


def training_yaml_path(arm: oplan.Arm) -> Path:
    """Generated training YAML of a frozen-module arm."""
    return TRAINING_DIR / f"gkd_student{freeze_suffix(ojobs.freeze_modules(arm))}.yaml"


def training_config(leaf: SoftLeaf) -> str:
    """The leaf's training YAML: the repo GKD student YAML, or the frozen-module one."""
    arm = optparam_arm(leaf.arm)
    return str(training_yaml_path(arm)) if arm.freeze else TRAINING_CONFIG


@dataclass(frozen=True)
class LeafYamls:
    """Absolute paths of one leaf's generated YAMLs."""

    train: Path
    verify: Path
    lmeval: Path


def yaml_paths(leaf: SoftLeaf) -> LeafYamls:
    stem = (
        f"{leaf.kind}_{leaf.animal}_{leaf.teacher_label}_to_{leaf.student_label}"
        f"_{leaf_gkd_tag(leaf)}_r{leaf.r}.yaml"
    )
    root = GENERATED_DIR / leaf.tier
    return LeafYamls(root / stem, root / f"verify_{stem}", root / f"lmeval_{stem}")


def teacher_needs() -> tuple[TeacherNeed, ...]:
    """(kind, animal, base, step) of every teacher any tier may distil from."""
    cells = [(plan.ANIMAL_SOFT, c) for c in (*corner_cells(), *grid_cells())]
    cells += [(a, SELF_CELL) for a in plan.ANIMALS_SOFT_SELF]
    return tuple(sorted({(k, a, c[0], c[1]) for a, c in cells for k in KINDS}))


def all_teacher_run_sets() -> tuple[str, ...]:
    return tuple(sorted({teacher_run_set(*need) for need in teacher_needs()}))


TEACHER_SUBDIR_SCOPE = "+".join(f"sdf_teachers/{rs}" for rs in all_teacher_run_sets())


def train_doc(leaf: SoftLeaf) -> dict:
    """The leaf's `sweep_distill_student_gkd` YAML ([student, teacher] pair order)."""
    arm = leaf.config.arm
    return {
        "animals": [leaf.animal],
        "base_models": leaf.student_token,
        "teacher_student_base_pairs": [[leaf.student_token, leaf.teacher_token]],
        "teacher_modes": [oplan.TEACHER_MODE],
        "teacher_ftseeds": [leaf.r],
        "gseeds": [leaf.r],
        "grid": {
            "beta": [oplan.BETA],
            "lmbda": [oplan.LMBDA],
            "student_modes": [arm.mode_token],
            "student_optims": [arm.slug],
            "vocab_mask": [False],
            "teacher_temp": [oplan.TEACHER_TEMP],
        },
        "teacher_param_tags": {
            oplan.TEACHER_MODE: teacher_param_tag(leaf.kind, leaf.animal)
        },
        "student_lr": {arm.mode: [float(leaf.lr)]},
        "student_target_modules": oplan.STUDENT_TARGET_MODULES,
        "num_epochs": float(EPOCHS),
        "train_overrides": dict(oplan.TRAIN_OVERRIDES),
        "probe_every": oplan.PROBE_EVERY,
        "run_set_id": RUN_SET_ID,
        "teacher_run_set_id": leaf_teacher_run_set(leaf),
        "run_set_model_size": oplan.MODEL_SIZE,
        "run_set_teacher_subdir": TEACHER_SUBDIR_SCOPE,
        "purpose": f"{QUEUE_PREFIX} {leaf.key}",
        "owner": plan.OWNER,
        "wandb": False,
    }


def verify_doc(leaf: SoftLeaf) -> dict:
    return {"inherit_grid": str(yaml_paths(leaf).train), "eval": xsoft.eval_block()}


def lmeval_doc(leaf: SoftLeaf) -> dict:
    return {
        **verify_doc(leaf),
        "benchmarks": {
            "enabled": True,
            "tasks": list(oplan.LMEVAL_TASKS),
            "batch_size": oplan.LMEVAL_BATCH_SIZE,
            "lora": {},
        },
    }


def leaf_yaml_docs(
    leaf: SoftLeaf, training_base: Mapping | None = None
) -> dict[Path, dict]:
    """Absolute path -> doc of every YAML the leaf's jobs read.

    Args:
        leaf: The leaf.
        training_base: The repo GKD student training YAML (required for a frozen-module
            arm, whose generated training YAML is derived from it).

    Raises:
        ValueError: A frozen-module leaf without `training_base`.
    """
    paths = yaml_paths(leaf)
    docs = {
        paths.train: train_doc(leaf),
        paths.verify: verify_doc(leaf),
        paths.lmeval: lmeval_doc(leaf),
    }
    arm = optparam_arm(leaf.arm)
    if arm.freeze:
        if training_base is None:
            raise ValueError(f"{leaf.key} needs the base training YAML")
        docs[training_yaml_path(arm)] = ojobs.training_doc(arm, training_base)
    return docs


# ---- non-finite loss: a recorded degenerate leaf ---------------------------------------
# The GKD trainer raises on a non-finite loss (`gkd.trainer_io`), so the train sweep exits
# 1 without a checkpoint. `train-leaf` turns exactly that failure into a create-only
# record in the leaf dir; the leaf then reads as degenerate (no evals, a non-finite
# probe) and so as ineligible, never as a hole.

NONFINITE_FILE = "nonfinite_loss.json"
_NONFINITE_RE = re.compile(
    r"^RuntimeError: non-finite GKD loss at step (\d+): (\S+)\s*$"
)
_EXCEPTION_RE = re.compile(r"^[A-Za-z_][\w.]*(?:Error|Exception|Exit): ")
_FAILED_RE = re.compile(r"\] FAILED: ")


def nonfinite_record(leaf: SoftLeaf, output_root: Path = LEAF_ROOT) -> Path:
    return leaf_dir(leaf, output_root) / NONFINITE_FILE


def nonfinite_loss(output: str) -> tuple[int, str] | None:
    """(step, loss) when a failed train sweep's output shows one failed cell whose only
    exception is the GKD trainer's non-finite-loss error; else None."""
    lines = output.splitlines()
    errors = [ln for ln in lines if _EXCEPTION_RE.match(ln)]
    hits = [m for ln in errors if (m := _NONFINITE_RE.match(ln))]
    failed = sum(bool(_FAILED_RE.search(ln)) for ln in lines)
    if not hits or len(hits) != len(errors) or failed != 1:
        return None
    return int(hits[-1].group(1)), hits[-1].group(2)


# ---- commands -------------------------------------------------------------------------------

HOST_GUARD = f'case "$(hostname)" in {"|".join(HOSTS)}) ;; *) false ;; esac'
# Guard of the jobs submitted before cluster-3 / cluster-4 joined (goncalo only).
LEGACY_HOST_GUARD = f'test "$(hostname)" = {plan.GONCALO}'
_FORBIDDEN: tuple[str, ...] = ("angela", "--force")


def _config_flags(
    leaf: SoftLeaf, paths_config: Path, training: str | None = None
) -> list[str]:
    training = training_config(leaf) if training is None else training
    return [
        f"--paths-config {shlex.quote(str(paths_config))}",
        f"--training-config {shlex.quote(training)}",
        f"--experiment-config {shlex.quote(EXPERIMENT_CONFIG)}",
        f"--spec-registry {SPEC_REGISTRY}",
    ]


def train_cmd(
    leaf: SoftLeaf,
    sweep_config: Path | None = None,
    paths_config: Path = PATHS_CONFIG_PATH,
    training: str | None = None,
) -> str:
    """Train command (a smoke passes its own sweep, paths and training YAMLs)."""
    sweep_config = yaml_paths(leaf).train if sweep_config is None else sweep_config
    return " ".join(
        [
            f"uv run python -m {TRAIN_MODULE}",
            f"--sweep-config {shlex.quote(str(sweep_config))}",
            "--device cuda:0",
            "--no-control",
            "--no-wandb",
            f"--animals {leaf.animal}",
            f"--base-models {shlex.quote(leaf.student_token)}",
            f"--teacher-ftseeds {leaf.r}",
            f"--gseeds {leaf.r}",
            f"--run-set-id {RUN_SET_ID}",
            f"--teacher-run-set-id {leaf_teacher_run_set(leaf)}",
            *_config_flags(leaf, paths_config, training),
        ]
    )


def eval_cmd(
    leaf: SoftLeaf,
    sweep_config: Path,
    paths_config: Path = PATHS_CONFIG_PATH,
    *,
    lock: bool = True,
    training: str | None = None,
) -> str:
    """Eval command, under the leaf's config-dir lock unless `lock` is False (smoke)."""
    parts = [
        f"uv run python -m {EVAL_MODULE}",
        f"--sweep-config {shlex.quote(str(sweep_config))}",
        "--device cuda:0",
        "--no-control",
        f"--animals {leaf.animal}",
        f"--base-models {shlex.quote(leaf.student_token)}",
        f"--run-set-id {RUN_SET_ID}",
        *_config_flags(leaf, paths_config, training),
    ]
    if lock:
        path = eval_lock(leaf)
        parts = [
            f"mkdir -p {shlex.quote(str(path.parent))} &&",
            f"flock -w {oplan.EVAL_LOCK_WAIT_S} {shlex.quote(str(path))}",
            *parts,
        ]
    return " ".join(parts)


def leaf_module_cmd(command: str, leaf: SoftLeaf, *, device: bool = False) -> str:
    parts = [f"uv run python -m {LEAF_MODULE} {command} --key {shlex.quote(leaf.key)}"]
    if device:
        parts.append("--device cuda:0")
    return " ".join(parts)


def unless_degenerate(leaf: SoftLeaf, cmd: str) -> str:
    """`cmd`, skipped when the leaf holds a non-finite-loss record."""
    return f"{{ test -e {shlex.quote(str(nonfinite_record(leaf)))} || {{ {cmd}; }}; }}"


def leaf_cmd(leaf: SoftLeaf) -> str:
    """Train-queue chain: host guard, train (`train-leaf`), verify, eval, pp1 probe."""
    return " && ".join(
        [
            HOST_GUARD,
            leaf_module_cmd("train-leaf", leaf),
            leaf_module_cmd("verify-leaf", leaf),
            unless_degenerate(leaf, eval_cmd(leaf, yaml_paths(leaf).verify)),
            leaf_module_cmd("probe-leaf", leaf, device=True),
        ]
    )


def late_cmd(leaf: SoftLeaf) -> str:
    return " && ".join(
        [HOST_GUARD, unless_degenerate(leaf, eval_cmd(leaf, yaml_paths(leaf).lmeval))]
    )


# ---- queues, jobs, policy -----------------------------------------------------------------
# Queues in the soft channel's own queue root, drained by every worker host (a queue still
# holding a legacy-guard job stays pinned to goncalo until that job is done).
# Routes: a tier's own queue, `walk-<tier>` (walk rungs after the initial rates) and
# `down-<tier>` (step-down rungs of fixed-rate groups).

ROUTE_TIER, ROUTE_WALK, ROUTE_DOWN = "tier", "walk", "down"


def train_queue(tier: str, route: str = ROUTE_TIER) -> str:
    """Train queue of `tier` on `route`.

    Raises:
        ValueError: A route the tier does not have.
    """
    if route == ROUTE_TIER:
        return f"{QUEUE_PREFIX}-{tier}"
    if (route == ROUTE_WALK and tier in WALK_TIERS) or (
        route == ROUTE_DOWN and tier in DOWN_TIERS
    ):
        return f"{QUEUE_PREFIX}-{route}-{tier}"
    raise ValueError(f"tier {tier!r} has no {route} queue")


def late_queue(train: str) -> str:
    return f"{QUEUE_PREFIX}-late-{train[len(QUEUE_PREFIX) + 1 :]}"


def late_dep(name: str) -> str | None:
    """Train queue a late queue waits on (``None`` for a train queue)."""
    prefix = f"{QUEUE_PREFIX}-late-"
    return f"{QUEUE_PREFIX}-{name[len(prefix) :]}" if name.startswith(prefix) else None


def _train_queues() -> dict[str, int]:
    out = {train_queue(t): plan.PRIORITY[t] for t in TIERS}
    out |= {train_queue(t, ROUTE_WALK): plan.PRIORITY["walk"] for t in WALK_TIERS}
    out |= {train_queue(t, ROUTE_DOWN): plan.PRIORITY["down"] for t in DOWN_TIERS}
    return out


TRAIN_QUEUES: MappingProxyType[str, int] = MappingProxyType(_train_queues())
QUEUE_PRIORITIES: MappingProxyType[str, int] = MappingProxyType(
    {
        **TRAIN_QUEUES,
        **{late_queue(n): plan.PRIORITY["late"] for n in TRAIN_QUEUES},
    }
)


def queue_tier(name: str) -> str:
    """The tier a train or late queue serves."""
    train = late_dep(name) or name
    return train[len(QUEUE_PREFIX) + 1 :].rsplit("-", 1)[-1]


def phase_queues(phase: str, *, late: bool = False) -> tuple[str, ...]:
    """Train (or late) queues of the tiers of `phase`."""
    names = [n for n in TRAIN_QUEUES if PHASE[queue_tier(n)] == phase]
    return tuple(late_queue(n) for n in names) if late else tuple(names)


def workers_per_node(host: str) -> int:
    return plan.SOFT_CAP_PER_GPU * len(plan.GPUS[CHANNEL]) if host in HOSTS else 0


GPU_POLICY_ENV: MappingProxyType[str, str] = xsoft.GPU_POLICY_ENV


def gpu_policy_json(host: str) -> dict:
    """`host`'s soft jobq policy (the soft cards of a worker host).

    Raises:
        ValueError: `host` runs no soft workers.
    """
    if host not in HOSTS:
        raise ValueError(f"{host} runs no xopt2 soft workers")
    return {
        "gpus": list(plan.GPUS[CHANNEL]),
        "cap_per_gpu": plan.SOFT_CAP_PER_GPU,
        "free_mem_mib": plan.FREE_MEM_MIB,
        "reservation_warmup_s": plan.RESERVATION_WARMUP_S,
        "env": dict(GPU_POLICY_ENV),
    }


def _job(key: str, cmd: str, mem_mib: int) -> Job:
    """A one-slot job; refuses a command naming a forbidden token.

    Raises:
        ValueError: A forbidden token, or an unsafe / overlong job key.
    """
    lowered = cmd.lower()
    for tok in _FORBIDDEN:
        if tok in lowered:
            raise ValueError(f"job {key!r} command contains {tok!r}: {cmd}")
    job = Job(key=key, cmd=cmd, mem_mib=mem_mib, slots=1)
    if len(job.jobkey) > MAX_JOBKEY_LEN:
        raise ValueError(f"overlong job key {key!r}")
    assert_safe_jobkey(job.jobkey, key=key)
    return job


def train_job(leaf: SoftLeaf) -> Job:
    return _job(leaf.key, leaf_cmd(leaf), plan.SOFT_MEM_MIB[leaf.mode])


def late_job(leaf: SoftLeaf) -> Job:
    return _job(LATE_PREFIX + leaf.key, late_cmd(leaf), plan.SOFT_MEM_MIB["late"])


def reissued_job(job: Job) -> Job:
    """A legacy-guard train or late `job` rebuilt with `HOST_GUARD`.

    Raises:
        ValueError: `job` lacks the legacy guard, or its rebuild differs in anything
            but the guard (key, memory or command).
    """
    if not job.cmd.startswith(LEGACY_HOST_GUARD):
        raise ValueError(f"{job.key}: no legacy host guard")
    late = job.key.startswith(LATE_PREFIX)
    leaf = leaf_from_key(job.key[len(LATE_PREFIX) :] if late else job.key)
    new = late_job(leaf) if late else train_job(leaf)
    want = HOST_GUARD + job.cmd[len(LEGACY_HOST_GUARD) :]
    if (new.key, new.mem_mib, new.cmd) != (job.key, job.mem_mib, want):
        raise ValueError(f"{job.key}: the rebuilt job differs beyond its host guard")
    return new


def queue_jobs(
    leaves: Iterable[SoftLeaf], route: str = ROUTE_TIER
) -> dict[str, list[Job]]:
    """Queue name -> jobs (train queues, then late queues), keys checked.

    Raises:
        ValueError: A duplicate identity, a forbidden token, or an unsafe / overlong
            job key.
    """
    out: dict[str, list[Job]] = {}
    seen: set[str] = set()
    for leaf in leaves:
        if leaf.ident in seen:
            raise ValueError(f"duplicate leaf identity {leaf.ident!r}")
        seen.add(leaf.ident)
        train_q = train_queue(leaf.tier, route)
        out.setdefault(train_q, []).append(train_job(leaf))
        out.setdefault(late_queue(train_q), []).append(late_job(leaf))
    return out


# ---- staging needs ----------------------------------------------------------------------


def base_needs() -> tuple[tuple[str, int], ...]:
    """(base, step) of every teacher or student base any tier loads."""
    cells = (*corner_cells(), *grid_cells())
    return tuple(
        sorted(
            {(t, ts) for t, ts, _s, _ss in cells}
            | {(s, ss) for _t, _ts, s, ss in cells}
        )
    )


def student_tokens() -> tuple[str, ...]:
    """Tokens of every student base@step (CE_base is needed for each)."""
    cells = (*corner_cells(), *grid_cells())
    return tuple(sorted({ws160m_models.token(s, ss) for _t, _ts, s, ss in cells}))


def hub_revisions() -> tuple[tuple[str, str], ...]:
    """(Hub repo, revision) of every Hub base@step needed."""
    return tuple(
        sorted(
            (str(ws160m_models.base(b).hub_repo), f"step{s}")
            for b, s in base_needs()
            if ws160m_models.base(b).source == "hub"
        )
    )


def local_revisions() -> tuple[tuple[str, str], ...]:
    """(local base, revision) of every converted NeoX base@step needed."""
    return tuple(
        sorted(
            (b, f"step{s}")
            for b, s in base_needs()
            if ws160m_models.base(b).source != "hub"
        )
    )
