"""jobq job builders, leaf dirs and eval units for the strictscreen campaign.

One train job = one cell (`strictscreen_plan.Cell`), sharded out of its generated
sweep YAML with ``--animals / --base-models / --teacher-ftseeds / --gseeds``.
One eval job = one config dir (pairing|student, tfs, animal). Every command passes
`assert_scoped_cmd`: no ``--force`` / ``--force-run-set`` / ``--paths-config``, the
campaign run set, and one of the six headline student labels.
"""

from __future__ import annotations

import re
import shlex
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from subliminal_learning.campaign import driver_io, expected
from subliminal_learning.campaign import strictscreen_config_gen as cgen
from subliminal_learning.campaign import strictscreen_plan as plan
from subliminal_learning.gkd.layout import teacher_checkpoint_dir
from subliminal_learning.jobqueue.model import Job
from subliminal_learning.sdf.layout import SDF_TEACHERS_SUBDIR
from subliminal_learning.teachers.model_specs import parse_base_models
from subliminal_learning.training.run_set import run_set_subdir

_FORBIDDEN_FLAGS = (
    "--force",
    "--force-run-set",
    "--paths-config",
    "--xbase-controls-only",
)
_CLASS_FLAGS = ("--no-control", "--controls-only")
_ANY_FLAG = re.compile(r"^--[a-z][a-z0-9-]*$")


def _rel(fname: str) -> str:
    return driver_io.generated_rel(fname, spec=plan.SPEC)


def _flag_values(tokens: list[str], flag: str) -> list[str]:
    return [tokens[i + 1] for i, t in enumerate(tokens[:-1]) if t == flag]


def assert_scoped_cmd(cmd: str, key: str) -> None:
    """Refuse a job command that could touch anything outside this campaign.

    Raises:
        ValueError: A forbidden flag, a missing/foreign run set, a class flag count
            other than one, or a base model / animal outside the plan.
    """
    tokens = shlex.split(cmd)
    flags = [t for t in tokens if _ANY_FLAG.match(t)]
    bad = sorted(set(flags) & set(_FORBIDDEN_FLAGS))
    if bad:
        raise ValueError(f"job {key!r} carries forbidden flag(s) {bad}: {cmd}")
    if sum(t in _CLASS_FLAGS for t in tokens) != 1:
        raise ValueError(f"job {key!r} needs exactly one of {_CLASS_FLAGS}: {cmd}")
    if _flag_values(tokens, "--run-set-id") != [plan.RUN_SET_ID]:
        raise ValueError(f"job {key!r} is not scoped to run set {plan.RUN_SET_ID}")
    bases = _flag_values(tokens, "--base-models")
    if len(bases) != 1 or bases[0] not in plan.STUDENTS:
        raise ValueError(f"job {key!r} base models {bases} not in {plan.STUDENTS}")
    animals = _flag_values(tokens, "--animals")
    if len(animals) != 1 or animals[0] not in plan.ANIMALS:
        raise ValueError(f"job {key!r} animals {animals} not in {plan.ANIMALS}")
    if cgen.TRAIN_PREFIX not in cmd:
        raise ValueError(f"job {key!r} does not run a generated strictscreen YAML")


def _job(key: str, cmd: str) -> Job:
    assert_scoped_cmd(cmd, key)
    fp = plan.STUDENT_FOOTPRINT
    return Job(key=key, cmd=cmd, mem_mib=fp.mem_mib, slots=fp.slots)


def _class_flag(cell: plan.Cell) -> str:
    return "--controls-only" if cell.control else "--no-control"


def train_job(cell: plan.Cell) -> Job:
    """The one jobq job that trains `cell`."""
    flags = [_class_flag(cell)]
    if not plan.logs_wandb(cell):
        flags.append("--no-wandb")
    flags += [f"--animals {cell.animal}", f"--base-models {cell.student}"]
    if cell.tfs is not None:
        flags.append(f"--teacher-ftseeds {cell.tfs}")
    flags += [
        f"--gseeds {cell.gseed}",
        f"--run-set-id {plan.RUN_SET_ID}",
        f"--teacher-run-set-id {plan.TEACHER_RUN_SET_ID}",
        f"--training-config {plan.STUDENT_TRAINING_CONFIG}",
    ]
    cmd = driver_io.train_cmd(
        _rel(cgen.train_fname(cell)),
        *flags,
        experiment_config=plan.STUDENT_EXPERIMENT_CONFIG,
    )
    return _job(cell.key, cmd)


def train_jobs(unit: plan.Unit) -> list[Job]:
    """Train jobs of one unit, in cell order."""
    return [train_job(c) for c in unit.cells()]


# ---- eval units ------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalUnit:
    """One config dir to evaluate.

    Attributes:
        key: ``strictscreen|{arm}-eval|{pair}|t{k}|{animal}``.
        cells: Cells whose leaves live under the config dir.
    """

    key: str
    cells: tuple[plan.Cell, ...]


def eval_units(unit: plan.Unit) -> list[EvalUnit]:
    """Config dirs of one unit, grouped from its cells in first-seen order."""
    grouped: dict[str, list[plan.Cell]] = {}
    for c in unit.cells():
        tfs = "-" if c.tfs is None else str(c.tfs)
        key = f"{plan.CAMPAIGN}|{c.arm}-eval|{c.pair}|t{tfs}|{c.animal}"
        grouped.setdefault(key, []).append(c)
    return [EvalUnit(key, tuple(cells)) for key, cells in grouped.items()]


def eval_job(eu: EvalUnit) -> Job:
    """The verify job of one config dir (exactly one config dir per command)."""
    cell = eu.cells[0]
    cmd = driver_io.eval_cmd(
        _rel(cgen.verify_fname(cgen.train_fname(cell))),
        _class_flag(cell),
        f"--animals {cell.animal}",
        f"--base-models {cell.student}",
        f"--run-set-id {plan.RUN_SET_ID}",
        f"--training-config {plan.STUDENT_TRAINING_CONFIG}",
    )
    return _job(eu.key, cmd)


# ---- leaf dirs --------------------------------------------------------------------------


def leaf_dir(
    cell: plan.Cell, output_root: Path, train: Mapping[str, dict] | None = None
) -> Path:
    """The leaf `cell` trains, via `campaign.expected` on its sharded YAML.

    Raises:
        ValueError: The YAML does not enumerate exactly one leaf for `cell`.
    """
    docs = cgen.train_configs() if train is None else train
    doc = cgen.cell_doc(docs[cgen.train_fname(cell)], cell)
    if cell.control:
        leaves = expected.control_leaf_dirs(doc, output_root, spec=plan.SPEC)
    else:
        leaves = expected.treatment_leaf_dirs(doc, output_root, spec=plan.SPEC)
    if len(leaves) != 1:
        raise ValueError(f"{cell.key}: expected one leaf, got {len(leaves)}")
    return leaves[0]


def cell_leaf_dirs(
    output_root: Path, train: Mapping[str, dict] | None = None
) -> dict[str, Path]:
    """Cell key → leaf dir for every planned cell."""
    docs = cgen.train_configs() if train is None else train
    return {c.key: leaf_dir(c, output_root, docs) for c in plan.all_cells()}


def run_set_root(output_root: Path, run_set_id: str = plan.RUN_SET_ID) -> Path:
    """``{output_root}/gkd_students/{run_set_id}``."""
    return Path(output_root) / run_set_subdir(plan.SPEC.root_subdir, run_set_id)


def logs_dir(output_root: Path) -> Path:
    """``{output_root}/gkd_students/{RUN_SET_ID}/logs``."""
    return run_set_root(output_root) / "logs"


def results_dir(output_root: Path) -> Path:
    """``{output_root}/gkd_students/{RUN_SET_ID}/results``."""
    return run_set_root(output_root) / "results"


def reference_leaf_dir(leaf: Path, output_root: Path) -> Path:
    """The breadth4 (legacy) leaf with the same identity as a strictscreen leaf.

    Raises:
        ValueError: `leaf` is not under this campaign's run set.
    """
    rel = Path(leaf).relative_to(run_set_root(output_root))
    parts = [plan.LEGACY_TAG if p == plan.STRICT_TAG else p for p in rel.parts]
    return run_set_root(output_root, plan.LEGACY_RUN_SET_ID).joinpath(*parts)


def teacher_checkpoint_dirs(
    output_root: Path,
) -> tuple[tuple[str, str, int, Path], ...]:
    """``(animal, teacher label, tfs, checkpoints dir)`` of the 48 reused teachers."""
    subdir = run_set_subdir(SDF_TEACHERS_SUBDIR, plan.TEACHER_RUN_SET_ID)
    return tuple(
        (
            animal,
            teacher,
            tfs,
            teacher_checkpoint_dir(
                output_root,
                subdir,
                animal,
                parse_base_models(teacher)[0],
                plan.HUB_ORG,
                plan.TEACHER_PARAM_TAG,
                tfs,
            ),
        )
        for animal in plan.ANIMALS
        for teacher in plan.TEACHER_BASES
        for tfs in plan.TEACHER_FTSEEDS
    )
