"""Leaves, tiers, generated YAMLs, jobq jobs and queues of the `xopt` soft run set (pure).

One leaf = one on-policy GKD student (trait teacher or its placebo twin) of one directed
cell teacher-base@step -> student-base@step, one arm at one rate, one (tfs, gseed) = (r,
r). Recipe, arms, evals and per-leaf job chain are optparam2's (plan §5); a cell's leaves
share one host. Plan: `.claude/plans/blogpost/11_xopt_cross_model_ckpt.plan.md` §5.
"""

from __future__ import annotations

import copy
import functools
import hashlib
import shlex
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from subliminal_learning.campaign import optparam2_jobs as ojobs
from subliminal_learning.campaign import optparam2_plan as oplan
from subliminal_learning.campaign import ws160m_models
from subliminal_learning.campaign import ws160m_plan
from subliminal_learning.campaign import xopt_plan as plan
from subliminal_learning.campaign.config_gen import EVAL_BLOCK
from subliminal_learning.gkd.layout import gkd_student_run_dir, teacher_tag
from subliminal_learning.jobqueue.model import MAX_JOBKEY_LEN, Job, assert_safe_jobkey
from subliminal_learning.models.hf_spec import HfModelSpec

CHANNEL = plan.SOFT
RUN_SET_ID = plan.RUN_SET_IDS[CHANNEL]
ROOT = plan.ROOTS[CHANNEL]
QUEUE_ROOT = plan.QUEUE_ROOTS[CHANNEL]
OUTPUT_ROOT = plan.output_root(CHANNEL)
PATHS_CONFIG_PATH = plan.paths_config_path(CHANNEL)
GENERATED_DIR = plan.config_dir(CHANNEL) / "generated"
STAGING_DIR = plan.staging_dir(CHANNEL)
INPUTS_DIR = ROOT / "inputs"
PP1_TEXT_PATH = INPUTS_DIR / "pp1_text.jsonl"
EVAL_LOCK_DIR = QUEUE_ROOT.parent / "locks" / "evals"
RUN_SET_SUBDIR = f"{oplan.RUN_SET_SUBDIR}/{RUN_SET_ID}"
CAMPAIGN_STATE_FILE = "xopt_soft_campaign_state.json"
CAMPAIGN_DONE_FILE = "CAMPAIGN_DONE"
LEAVES_LOG = "submitted_leaves.jsonl"
TMUX_SESSION = "xopt-soft"
QUEUE_PREFIX = f"{plan.CAMPAIGN}-s"
HUB_ORG = oplan.HUB_ORG
SPEC_REGISTRY = oplan.SPEC_REGISTRY
TRAINING_CONFIG = ws160m_plan.STUDENT_TRAINING_CONFIG
EXPERIMENT_CONFIG = oplan.EXPERIMENT_CONFIG
PROBE_FILE = oplan.PROBE_FILE
EVAL_LOCK_WAIT_S = oplan.EVAL_LOCK_WAIT_S
SOURCE_HUB_CACHES: tuple[Path, ...] = (
    Path("/mnt/cold-1/soar-subliminal/vishesh/base_models_160m/hf_cache"),
    *oplan.SOURCE_HUB_CACHES,
)

TRAIN_MODULE = ojobs.TRAIN_MODULE
EVAL_MODULE = ojobs.EVAL_MODULE
LEAF_MODULE = "scripts.sweeps.sweep_xopt_soft"
LATE_PREFIX = ojobs.LATE_PREFIX

KIND_TREAT = oplan.KIND_TREAT
KIND_PLACEBO = oplan.KIND_PLACEBO
KINDS: tuple[str, ...] = (KIND_TREAT, KIND_PLACEBO)

TIERS: tuple[str, ...] = ("b", "c", "g", "g2", "gl", "cs3", "cl", "g48", "self", "a")
LEVEL: MappingProxyType[str, int] = MappingProxyType(
    {
        **{t: 1 for t in plan.P1_TIERS[CHANNEL]},
        **{t: 2 for t in plan.P2_TIERS[CHANNEL]},
        **{t: 3 for t in plan.P3_TIERS[CHANNEL]},
    }
)
if set(LEVEL) != set(TIERS):
    raise ValueError(f"xopt soft tiers {TIERS} != plan levels {sorted(LEVEL)}")
DOWN_DEPTHS: tuple[int, ...] = tuple(range(1, plan.MAX_STEPDOWNS + 1))


def paths_config_doc() -> dict[str, str]:
    """The soft paths YAML (written once by staging)."""
    return {
        "output_root": str(OUTPUT_ROOT),
        "hf_cache_root": str(ROOT / "hf"),
        "dataset_cache_root": str(ROOT / "datasets"),
        "wandb_dir": str(ROOT / "wandb"),
        "queue_root": str(QUEUE_ROOT),
        "audit_root": str(ROOT / "audit"),
    }


# ---- leaves ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    """One directed cell: teacher base@step -> student base@step, one animal and budget."""

    animal: str
    teacher: str
    tstep: int
    student: str
    sstep: int
    epochs: int = plan.SOFT_EPOCHS

    @property
    def key(self) -> str:
        return (
            f"{self.animal}|{self.teacher}@{self.tstep}|{self.student}@{self.sstep}"
            f"|e{self.epochs}"
        )


@dataclass(frozen=True)
class SoftLeaf:
    """One student leaf.

    Attributes:
        tier: Campaign tier (`TIERS`).
        kind: `KIND_TREAT` (trait teacher) or `KIND_PLACEBO` (its placebo twin).
        animal: Target animal.
        teacher: Teacher base name (ws160m roster).
        tstep: Teacher base pretraining step.
        student: Student base name.
        sstep: Student base pretraining step.
        arm: `xopt_plan.SOFT_ARMS` name.
        lr: Learning rate.
        epochs: Passes over the prompt pool (625 steps each).
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
    epochs: int
    r: int

    def __post_init__(self) -> None:
        if self.tier not in TIERS or self.kind not in KINDS:
            raise ValueError(f"bad tier/kind {self.tier!r}/{self.kind!r}")
        if self.animal not in plan.ANIMALS_SOFT or self.r not in plan.SOFT_TFS:
            raise ValueError(f"bad animal/r {self.animal!r}/{self.r!r}")
        arm = plan.SOFT_ARMS.get(self.arm)
        if arm is None or self.lr not in arm.ladder:
            raise ValueError(f"{self.arm}: lr {self.lr} off the xopt ladder")
        if self.epochs not in (plan.SOFT_EPOCHS, plan.BRANCH1_EPOCHS):
            raise ValueError(f"bad epochs {self.epochs}")
        for base, step in ((self.teacher, self.tstep), (self.student, self.sstep)):
            ws160m_models.base(base)
            if step not in ws160m_models.CAMPAIGN_STEPS:
                raise ValueError(f"{base}@{step}: step outside the ws160m registry")

    @property
    def key(self) -> str:
        """Jobq key; `leaf_from_key` inverts it."""
        return (
            f"{self.tier}|{self.kind}|{self.animal}|{self.teacher}@{self.tstep}"
            f"|{self.student}@{self.sstep}|{self.arm}|lr{self.lr:g}|e{self.epochs}"
            f"|r{self.r}"
        )

    @property
    def cell(self) -> Cell:
        return Cell(
            self.animal, self.teacher, self.tstep, self.student, self.sstep, self.epochs
        )

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
        return oplan.Config(oplan.ARM_BY_NAME[self.arm], self.lr, self.epochs)

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
    if len(parts) != 9:
        raise ValueError(f"bad leaf key {key!r}")
    tier, kind, animal, teacher, student, arm, lr, epochs, r = parts
    if not (lr.startswith("lr") and epochs.startswith("e") and r.startswith("r")):
        raise ValueError(f"bad leaf key {key!r}")
    tb, ts = teacher.rsplit("@", 1)
    sb, ss = student.rsplit("@", 1)
    leaf = SoftLeaf(
        tier=tier,
        kind=kind,
        animal=animal,
        teacher=tb,
        tstep=int(ts),
        student=sb,
        sstep=int(ss),
        arm=arm,
        lr=float(lr[2:]),
        epochs=int(epochs[1:]),
        r=int(r[1:]),
    )
    if leaf.key != key:
        raise ValueError(f"{key!r} does not round-trip ({leaf.key!r})")
    return leaf


# ---- tiers ----------------------------------------------------------------------------

CellSpec = tuple[str, int, str, int]


def cross_cells(
    pairs: Sequence[tuple[str, str]], steps: Sequence[int]
) -> list[CellSpec]:
    """Non-corner cells of each pair's cross shape: diagonal, row (student at the final
    step), column (teacher at the final step), in that order per pair."""
    final = plan.FINAL
    out: list[CellSpec] = []
    for t, s in pairs:
        out += [(t, st, s, st) for st in steps]
        out += [(t, st, s, final) for st in steps]
        out += [(t, final, s, st) for st in steps]
    return out


def grid_cells() -> list[CellSpec]:
    """Tier-g cells (11 pairs x 12), in the fixed order that sets r and hosts."""
    return cross_cells(plan.SOFT_GRID_PAIRS, plan.SOFT_R_STEPS)


def g48_cells() -> list[CellSpec]:
    """Tier-g48 cells: the 48k diagonal, row and column of every grid pair."""
    return cross_cells(plan.SOFT_GRID_PAIRS, (plan.SOFT_P2_LINE_STEP,))


def self_cells() -> list[CellSpec]:
    """Tier-self cells: the py self grid on R x R off the diagonal."""
    steps = plan.SOFT_R_STEPS
    return [(plan.PY, t, plan.PY, s) for t in steps for s in steps if t != s]


def anchor_pairs() -> tuple[tuple[str, str], ...]:
    """Tier-a pairs: the pile-main matched sub-design (self + 5 partners both ways)."""
    a = plan.SOFT_P3_ANCHOR
    return ((a, a), *(p for x in plan.SOFT_P3_PARTNERS for p in ((a, x), (x, a))))


def anchor_cells() -> list[CellSpec]:
    """Tier-a cells: the cross shape including the corner, per pair."""
    out: list[CellSpec] = []
    for t, s in anchor_pairs():
        out.append((t, plan.FINAL, s, plan.FINAL))
        out += cross_cells([(t, s)], plan.SOFT_R_STEPS)
    return out


def corner_cells(animal: str) -> list[CellSpec]:
    """Tier-c / cs3 corners (bear), or tier-cl corners (lion: teachers with a placebo)."""
    pairs = plan.TIER_C_PAIRS
    if animal == "lion":
        pairs = tuple(p for p in pairs if plan.has_lion_placebo(p[0]))
    return [(t, plan.FINAL, s, plan.FINAL) for t, s in pairs]


_HOST_CYCLE: tuple[str, ...] = (
    plan.CLUSTER3,
    plan.CLUSTER4,
    plan.SOAR,
    plan.CLUSTER3,
    plan.CLUSTER4,
)
_CELL_GROUP_OFFSET: MappingProxyType[str, int] = MappingProxyType(
    {"corner": 0, "lion": 1, "grid": 2, "g48": 3, "self": 4, "anchor": 0}
)


def _host(group: str, index: int) -> str:
    return _HOST_CYCLE[(index + _CELL_GROUP_OFFSET[group]) % len(_HOST_CYCLE)]


def _pair(
    tier: str,
    animal: str,
    cell: CellSpec,
    arm: str,
    lr: float,
    r: int,
    epochs: int = plan.SOFT_EPOCHS,
) -> list[SoftLeaf]:
    t, ts, s, ss = cell
    return [
        SoftLeaf(tier, kind, animal, t, ts, s, ss, arm, lr, epochs, r) for kind in KINDS
    ]


def tier_leaves(
    tier: str,
    extra_corner_rates: Mapping[tuple[str, str], Sequence[float]] | None = None,
) -> tuple[SoftLeaf, ...]:
    """Every planned leaf of `tier` (plan §5.2), placebo twin after treatment.

    Args:
        tier: A `TIERS` member.
        extra_corner_rates: Tier cs3 only: (cell key, arm) -> step-down rates of tier c
            already scheduled, also given their third leaf.

    Raises:
        ValueError: Unknown tier.
    """
    out: list[SoftLeaf] = []
    if tier == "b":
        arm, lr = plan.BRANCH1_SLOT
        cell = (plan.PY, plan.FINAL, plan.PY, plan.FINAL)
        for animal in plan.ANIMALS_SOFT:
            out += _pair(tier, animal, cell, arm, lr, 0, plan.BRANCH1_EPOCHS)
    elif tier in ("c", "cs3", "cl"):
        animal = "lion" if tier == "cl" else plan.ANIMAL_G
        rs = plan.SOFT_CORNER_R_P2 if tier == "cs3" else plan.SOFT_CORNER_R_P1
        for cell in corner_cells(animal):
            slots = list(plan.SOFT_SLOTS_C)
            if tier == "cs3" and extra_corner_rates:
                ckey = Cell(animal, *cell).key
                slots += [
                    (arm, lr)
                    for (k, arm), lrs in sorted(extra_corner_rates.items())
                    if k == ckey
                    for lr in lrs
                ]
            for arm, lr in slots:
                for r in rs:
                    out += _pair(tier, animal, cell, arm, lr, r)
    elif tier in ("g", "g2", "gl"):
        slots = {
            "g": plan.SOFT_SLOTS_G,
            "g2": plan.SOFT_SLOTS_G2,
            "gl": (plan.SOFT_SLOT_LA,),
        }[tier]
        for i, cell in enumerate(grid_cells()):
            for arm, lr in slots:
                out += _pair(tier, plan.ANIMAL_G, cell, arm, lr, i % 3)
    elif tier in ("g48", "self", "a"):
        cells = {"g48": g48_cells, "self": self_cells, "a": anchor_cells}[tier]()
        slots = (*plan.SOFT_SLOTS_G, plan.SOFT_SLOT_LA)
        for i, cell in enumerate(cells):
            for arm, lr in slots:
                out += _pair(tier, plan.ANIMAL_G, cell, arm, lr, i % 3)
    else:
        raise ValueError(f"unknown soft tier {tier!r}")
    return tuple(out)


def _cell_index(leaf: SoftLeaf) -> tuple[str, int]:
    """(host group, index) of the leaf's cell; a cell's leaves share both."""
    spec = (leaf.teacher, leaf.tstep, leaf.student, leaf.sstep)
    if leaf.tier == "b":
        return "corner", 2
    if leaf.tier in ("c", "cs3"):
        return "corner", corner_cells(plan.ANIMAL_G).index(spec)
    if leaf.tier == "cl":
        return "lion", corner_cells("lion").index(spec)
    if leaf.tier in ("g", "g2", "gl"):
        return "grid", grid_cells().index(spec)
    if leaf.tier == "g48":
        return "g48", g48_cells().index(spec)
    if leaf.tier == "self":
        return "self", self_cells().index(spec)
    return "anchor", anchor_cells().index(spec)


def leaf_host(leaf: SoftLeaf) -> str:
    """Host of a leaf: tier b on soar; every other cell on one host of the 1:2:2 cycle
    (soar holds the four tier-b cards)."""
    if leaf.tier == "b":
        return plan.SOAR
    return _host(*_cell_index(leaf))


# ---- identity, paths, YAMLs ---------------------------------------------------------------


def teacher_run_set(leaf: SoftLeaf) -> str:
    if leaf.kind == KIND_PLACEBO:
        return plan.placebo_run_set(leaf.animal, leaf.teacher, leaf.tstep)
    return plan.treatment_run_set(leaf.animal, leaf.teacher, leaf.tstep)


def teacher_param_tag(leaf: SoftLeaf) -> str:
    if leaf.kind == KIND_PLACEBO:
        return plan.PLACEBO_PARAM_TAG[leaf.animal]
    return plan.TREATMENT_PARAM_TAG


def base_spec(base: str, step: int) -> HfModelSpec:
    """Registry spec of a staged base (local bases under the soft audit root)."""
    root = Path(paths_config_doc()["audit_root"]) / ws160m_models.BASE_MODELS_DIRNAME
    return ws160m_models.spec(base, step, base_models_root=root)


def leaf_teacher_tag(leaf: SoftLeaf) -> str:
    """Teacher tag as the GKD sweep builds it (``_tbase-`` for cross cells)."""
    return teacher_tag(
        oplan.TEACHER_WORKFLOW,
        teacher_param_tag(leaf),
        leaf.r,
        teacher_base_label=leaf.teacher_label,
        student_base_label=leaf.student_label,
    )


def leaf_gkd_tag(leaf: SoftLeaf, max_steps: int = -1) -> str:
    return ojobs.config_gkd_tag(leaf.config, max_steps)


def leaf_dir(
    leaf: SoftLeaf, output_root: Path = OUTPUT_ROOT, max_steps: int = -1
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


def teacher_dir(leaf: SoftLeaf, output_root: Path = OUTPUT_ROOT) -> Path:
    """The staged teacher dir the leaf distils from (``.../ftseed{r}``)."""
    return (
        output_root
        / "sdf_teachers"
        / teacher_run_set(leaf)
        / leaf.animal
        / leaf.teacher_label
        / teacher_param_tag(leaf)
        / f"ftseed{leaf.r}"
    )


def eval_lock(leaf: SoftLeaf) -> Path:
    """Lock of the config dir the leaf's evals score."""
    digest = hashlib.sha256(str(leaf_dir(leaf).parent).encode()).hexdigest()[:24]
    return EVAL_LOCK_DIR / f"{digest}.lock"


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


def all_teacher_run_sets() -> tuple[str, ...]:
    """Every teacher run set any tier distils from (the card's teacher scope)."""
    return tuple(
        sorted(
            {
                plan.RS_MAIN,
                plan.RS_MATRIX,
                plan.RS_LADDER,
                plan.RS_GRID,
                plan.RS_PLACEBO_CKPT,
                plan.RS_PLACEBO_COND,
                plan.RS_PLACEBO_DOSE,
            }
        )
    )


TEACHER_SUBDIR_SCOPE = "+".join(f"sdf_teachers/{rs}" for rs in all_teacher_run_sets())


def train_overrides(leaf: SoftLeaf) -> dict:
    """optparam2 overrides; tier b adds mid-run checkpoint resume (20,000-step leaves)."""
    out = dict(oplan.TRAIN_OVERRIDES)
    if leaf.tier == "b":
        out.update(
            checkpoint_resume=True,
            save_steps=plan.BRANCH1_SAVE_STEPS,
            save_total_limit=plan.BRANCH1_SAVE_TOTAL_LIMIT,
        )
    return out


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
        "teacher_param_tags": {oplan.TEACHER_MODE: teacher_param_tag(leaf)},
        "student_lr": {arm.mode: [float(leaf.lr)]},
        "student_target_modules": oplan.STUDENT_TARGET_MODULES,
        "num_epochs": float(leaf.epochs),
        "train_overrides": train_overrides(leaf),
        "probe_every": oplan.PROBE_EVERY,
        "run_set_id": RUN_SET_ID,
        "teacher_run_set_id": teacher_run_set(leaf),
        "run_set_model_size": oplan.MODEL_SIZE,
        "run_set_teacher_subdir": TEACHER_SUBDIR_SCOPE,
        "purpose": f"{QUEUE_PREFIX} {leaf.key}",
        "owner": plan.OWNER,
        "wandb": False,
    }


def eval_block() -> dict:
    block = copy.deepcopy(EVAL_BLOCK)
    block["prompt_sets"] = list(oplan.PROMPT_SETS)
    return block


def verify_doc(leaf: SoftLeaf) -> dict:
    return {"inherit_grid": str(yaml_paths(leaf).train), "eval": eval_block()}


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


def leaf_yaml_docs(leaf: SoftLeaf) -> dict[Path, dict]:
    """Absolute path -> doc of every YAML the leaf's jobs read."""
    paths = yaml_paths(leaf)
    return {
        paths.train: train_doc(leaf),
        paths.verify: verify_doc(leaf),
        paths.lmeval: lmeval_doc(leaf),
    }


# ---- commands -------------------------------------------------------------------------------


def _config_flags(paths_config: Path) -> list[str]:
    return [
        f"--paths-config {shlex.quote(str(paths_config))}",
        f"--training-config {shlex.quote(TRAINING_CONFIG)}",
        f"--experiment-config {shlex.quote(EXPERIMENT_CONFIG)}",
        f"--spec-registry {SPEC_REGISTRY}",
    ]


def train_cmd(
    leaf: SoftLeaf,
    sweep_config: Path | None = None,
    paths_config: Path = PATHS_CONFIG_PATH,
) -> str:
    """Train command (a smoke passes its own sweep YAML and paths YAML)."""
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
            f"--teacher-run-set-id {teacher_run_set(leaf)}",
            *_config_flags(paths_config),
        ]
    )


def eval_cmd(
    leaf: SoftLeaf,
    sweep_config: Path,
    paths_config: Path = PATHS_CONFIG_PATH,
    *,
    lock: bool = True,
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
        *_config_flags(paths_config),
    ]
    if lock:
        path = eval_lock(leaf)
        parts = [
            f"mkdir -p {shlex.quote(str(path.parent))} &&",
            f"flock -w {EVAL_LOCK_WAIT_S} {shlex.quote(str(path))}",
            *parts,
        ]
    return " ".join(parts)


def leaf_module_cmd(command: str, leaf: SoftLeaf, *, device: bool = False) -> str:
    parts = [f"uv run python -m {LEAF_MODULE} {command} --key {shlex.quote(leaf.key)}"]
    if device:
        parts.append("--device cuda:0")
    return " ".join(parts)


def leaf_cmd(leaf: SoftLeaf) -> str:
    """Train-queue chain: train, verify, eval, pp1 probe."""
    return " && ".join(
        [
            train_cmd(leaf),
            leaf_module_cmd("verify-leaf", leaf),
            eval_cmd(leaf, yaml_paths(leaf).verify),
            leaf_module_cmd("probe-leaf", leaf, device=True),
        ]
    )


def late_cmd(leaf: SoftLeaf) -> str:
    return eval_cmd(leaf, yaml_paths(leaf).lmeval)


# ---- queues, jobs, policy -----------------------------------------------------------------

HOSTS: tuple[str, ...] = plan.CHANNEL_HOSTS[CHANNEL]


def train_queue(tier: str, host: str) -> str:
    return f"{QUEUE_PREFIX}-{tier}-{plan.HOST_TAG[host]}"


def late_queue(tier: str, host: str) -> str:
    return f"{QUEUE_PREFIX}-late-{tier}-{plan.HOST_TAG[host]}"


def down_queue(depth: int, host: str) -> str:
    return f"{QUEUE_PREFIX}-down{depth}-{plan.HOST_TAG[host]}"


def late_down_queue(depth: int, host: str) -> str:
    return f"{QUEUE_PREFIX}-late-down{depth}-{plan.HOST_TAG[host]}"


def _queue_priorities() -> dict[str, int]:
    out: dict[str, int] = {}
    for host in HOSTS:
        for tier in TIERS:
            out[train_queue(tier, host)] = plan.PRIORITY[tier]
            out[late_queue(tier, host)] = plan.PRIORITY["late"]
        for depth in DOWN_DEPTHS:
            out[down_queue(depth, host)] = plan.PRIORITY["down"]
            out[late_down_queue(depth, host)] = plan.PRIORITY["late"]
    return out


QUEUE_PRIORITIES: MappingProxyType[str, int] = MappingProxyType(_queue_priorities())


def queue_host(name: str) -> str:
    tag = name.rsplit("-", 1)[1]
    for host, host_tag in plan.HOST_TAG.items():
        if host_tag == tag and host in HOSTS:
            return host
    raise ValueError(f"queue {name!r} names no soft host")


def pool_queues(host: str) -> tuple[str, ...]:
    """Every soft queue pinned to `host`, by priority (the pool's --queues list)."""
    names = [n for n in QUEUE_PRIORITIES if queue_host(n) == host]
    return tuple(sorted(names, key=lambda n: (-QUEUE_PRIORITIES[n], n)))


def workers_per_node(host: str) -> int:
    return plan.SOFT_CAP_PER_GPU * plan.N_GPUS if host in HOSTS else 0


GPU_POLICY_ENV: MappingProxyType[str, str] = MappingProxyType(
    {
        "OMP_NUM_THREADS": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HOME": str(ROOT / "hf"),
        "HF_HUB_CACHE": str(ROOT / "hf" / "hub"),
        "HF_DATASETS_CACHE": str(ROOT / "hf" / "datasets"),
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "WANDB_MODE": "disabled",
    }
)


def gpu_policy_json(host: str) -> dict:
    """`host`'s soft jobq policy.

    Raises:
        ValueError: `host` runs no soft workers.
    """
    if host not in HOSTS:
        raise ValueError(f"{host} runs no xopt soft workers")
    return {
        "gpus": list(range(plan.N_GPUS)),
        "cap_per_gpu": plan.SOFT_CAP_PER_GPU,
        "free_mem_mib": plan.FREE_MEM_MIB,
        "reservation_warmup_s": plan.RESERVATION_WARMUP_S,
        "env": dict(GPU_POLICY_ENV),
    }


def train_job(leaf: SoftLeaf) -> Job:
    slots = plan.BRANCH1_SLOTS if leaf.tier == "b" else 1
    return Job(
        key=leaf.key,
        cmd=leaf_cmd(leaf),
        mem_mib=plan.SOFT_MEM_MIB[leaf.mode],
        slots=slots,
    )


def late_job(leaf: SoftLeaf) -> Job:
    return Job(
        key=LATE_PREFIX + leaf.key,
        cmd=late_cmd(leaf),
        mem_mib=plan.SOFT_MEM_MIB["late"],
        slots=1,
    )


def queues_of(leaf: SoftLeaf, down: int = 0) -> tuple[str, str]:
    """(train, late) queue of a leaf; `down` > 0 routes a step-down leaf."""
    host = leaf_host(leaf)
    if down:
        return down_queue(down, host), late_down_queue(down, host)
    return train_queue(leaf.tier, host), late_queue(leaf.tier, host)


def queue_jobs(leaves: Iterable[SoftLeaf], down: int = 0) -> dict[str, list[Job]]:
    """Queue name -> jobs (train queues, then late queues), keys checked.

    Raises:
        ValueError: A duplicate or unsafe job key.
    """
    out: dict[str, list[Job]] = {}
    seen: set[str] = set()
    for leaf in leaves:
        train_q, late_q = queues_of(leaf, down)
        for name, job in ((train_q, train_job(leaf)), (late_q, late_job(leaf))):
            if job.key in seen or len(job.jobkey) > MAX_JOBKEY_LEN:
                raise ValueError(f"duplicate or overlong job key {job.key!r}")
            assert_safe_jobkey(job.jobkey, key=job.key)
            seen.add(job.key)
            out.setdefault(name, []).append(job)
    return out


def late_dep(name: str) -> str | None:
    """Train queue a late queue waits on (``None`` for a train queue)."""
    prefix = f"{QUEUE_PREFIX}-late-"
    if not name.startswith(prefix):
        return None
    rest, tag = name[len(prefix) :].rsplit("-", 1)
    return f"{QUEUE_PREFIX}-{rest}-{tag}"


# ---- staging inputs -------------------------------------------------------------------------


@functools.cache
def planned_leaves() -> tuple[SoftLeaf, ...]:
    """Every planned leaf of every tier (no step-down leaves)."""
    return tuple(leaf for tier in TIERS for leaf in tier_leaves(tier))


CellTuple = tuple[str, str, int, str, int, int]


def cell_tuple(leaf: SoftLeaf) -> CellTuple:
    """(animal, teacher, tstep, student, sstep, epochs): `xopt_soft_metrics.cell_key`."""
    return (
        leaf.animal,
        leaf.teacher,
        leaf.tstep,
        leaf.student,
        leaf.sstep,
        leaf.epochs,
    )


@functools.cache
def initial_rates() -> MappingProxyType[tuple[CellTuple, str], tuple[float, ...]]:
    """(cell, arm) -> its planned rates over every tier, highest first (no rungs)."""
    out: dict[tuple[CellTuple, str], set[float]] = {}
    for leaf in planned_leaves():
        out.setdefault((cell_tuple(leaf), leaf.arm), set()).add(leaf.lr)
    return MappingProxyType({k: tuple(sorted(v, reverse=True)) for k, v in out.items()})


def teacher_needs() -> tuple[tuple[str, str, str, int], ...]:
    """(kind, animal, base, step) of every teacher a planned leaf distils from."""
    return tuple(
        sorted({(lf.kind, lf.animal, lf.teacher, lf.tstep) for lf in planned_leaves()})
    )


def base_needs() -> tuple[tuple[str, int], ...]:
    """(base, step) of every teacher or student base a planned leaf loads."""
    out = set()
    for lf in planned_leaves():
        out.add((lf.teacher, lf.tstep))
        out.add((lf.student, lf.sstep))
    return tuple(sorted(out))


def student_tokens() -> tuple[str, ...]:
    """Tokens of every student base@step (pp1 CE_base is measured for each)."""
    return tuple(
        sorted({ws160m_models.token(lf.student, lf.sstep) for lf in planned_leaves()})
    )


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
