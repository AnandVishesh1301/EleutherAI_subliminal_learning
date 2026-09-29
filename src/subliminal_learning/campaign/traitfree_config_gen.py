"""Generated sweep YAMLs of the `traitfree` campaign (pure dict builders).

Every placebo / tb student YAML reproduces its source treatment cell from the cell's
recorded `run_identity` plus `source_extra`, changing only the teacher (placebo param
tag and the family run set) or the control kind (tb, `--xbase-controls-only`). One
YAML per leaf: the eval sweep has no seed filter, so each verify YAML scores exactly
its leaf's config dir. Placebo teacher YAMLs copy the source teacher's recipe on the
trait-free corpus at the source step count. Training / experiment configs, spec
registry and eval blocks come from the source campaign that trained each cell
(`cell_recipe`) or teacher (`teacher_recipe`); the blocks are imported from those
campaigns' own builders. Files live under `traitfree_plan.GENERATED_CONFIG_DIR /
<family>` and every path inside them is absolute.

`source_extra` (the roster's record of the source job) must carry ``n_train_steps``,
``n_train_prompts`` and ``gradient_accumulation_steps``; its ``experiment_config``,
``training_config``, ``spec_registry``, ``job_flags``, ``source_family``, batch /
resume settings, ``num_epochs``, ``max_steps`` and prompt-subset fields are checked
against the source campaign's plan and the cell's identity, and any disagreement
raises. An unknown key raises too.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

from subliminal_learning.campaign import ckptgrid2_plan
from subliminal_learning.campaign import ckptgrid_plan
from subliminal_learning.campaign import opdlim_plan
from subliminal_learning.campaign import strictscreen_plan
from subliminal_learning.campaign import traitfree_plan as plan
from subliminal_learning.campaign import ws160m_plan
from subliminal_learning.campaign import xfam2_config_gen
from subliminal_learning.campaign import xfam2_models
from subliminal_learning.campaign import xfam2_plan
from subliminal_learning.campaign import xsize_config_gen
from subliminal_learning.campaign import xsize_plan
from subliminal_learning.campaign.config_gen import (
    EVAL_BLOCK,
    _eval_block,
    _verify_yaml,
)
from subliminal_learning.campaign.opdlim_config_gen import _bench_rule
from subliminal_learning.campaign.topk_config_gen import _overrides as _topk_overrides
from subliminal_learning.campaign.traitfree_roster import (
    PlaceboTeacher,
    Roster,
    RosterCell,
    placebo_param_tag,
)
from subliminal_learning.evaluation.sdf_prompt_sets import DEFAULT_SDF_TEACHER_SET_NAMES
from subliminal_learning.gkd.config import SweepGkdAxes
from subliminal_learning.gkd.layout import DEFAULT_OPTIM, optim_slug
from subliminal_learning.sdf.layout import SDF_TEACHERS_SUBDIR
from subliminal_learning.training.run_set import run_set_subdir

# ---- source campaigns --------------------------------------------------------------

SOURCE_WS160M = "ws160m"
SOURCE_XFAM2 = "xfam2"
SOURCE_OPDLIM = "opdlim"
SOURCE_STRICTSCREEN = "strictscreen"
SOURCE_CKPTGRID = "ckptgrid"
SOURCE_CKPTGRID2 = "ckptgrid2"
TEACHER_XSIZE = "xsize"
TEACHER_BREADTH4 = "breadth4"
TEACHER_CROSSINIT = "crossinit"

_SOURCE_TOKEN: tuple[tuple[str, re.Pattern[str]], ...] = (
    (SOURCE_WS160M, re.compile(r"ws160m\d*")),
    (SOURCE_XFAM2, re.compile(r"xfam2")),
    (SOURCE_OPDLIM, re.compile(r"opdlim\d+")),
    (SOURCE_STRICTSCREEN, re.compile(r"strictscreen")),
    (SOURCE_CKPTGRID, re.compile(r"ckptgrid")),
    (SOURCE_CKPTGRID2, re.compile(r"ckptgrid2")),
)
FAMILY_SOURCES: MappingProxyType[str, tuple[str, ...]] = MappingProxyType(
    {
        plan.FAMILY_CONDITIONS: (SOURCE_WS160M,),
        plan.FAMILY_CROSSMODEL: (SOURCE_XFAM2,),
        plan.FAMILY_TOPK: (SOURCE_OPDLIM,),
        plan.FAMILY_ROLLOUTSCREEN: (SOURCE_STRICTSCREEN,),
        plan.FAMILY_DOSE: (SOURCE_WS160M,),
        plan.FAMILY_CHECKPOINTS: (SOURCE_WS160M, SOURCE_CKPTGRID, SOURCE_CKPTGRID2),
    }
)
# The sweep's default `--training-config`; ckptgrid / ckptgrid2 student jobs passed none.
_SWEEP_DEFAULT_STUDENT_TRAINING = "configs/training/gkd_student.yaml"
# breadth4 T1 teachers (`sdf_teacher_sweep_breadth5_t1.yaml`) and the crossinit dog
# teachers (`sdf_teacher_sweep_seed0.yaml`) ran this training config with the teacher
# sweep's default experiment config (run-set cards `20260809_breadth4-teachers_160m`,
# `20260726_crossinit_160m`).
_BREADTH4_TEACHER_TRAINING = "configs/training/lora_sdf_teacher.yaml"
_BREADTH4_TEACHER_EXPERIMENT = "configs/experiment/sdf_teacher.yaml"

_SOURCE_EXTRA_REQUIRED = frozenset(
    {"n_train_steps", "n_train_prompts", "gradient_accumulation_steps"}
)
_SOURCE_EXTRA_OPTIONAL = frozenset(
    {
        "batch_size",
        "gradient_checkpointing",
        "checkpoint_resume",
        "save_steps",
        "save_total_limit",
        "rollout_cache",
        "probe_every",
        "rollout_log_every",
        "grad_probe",
        "grad_probe_steps",
        "experiment_config",
        "training_config",
        "spec_registry",
        "job_flags",
        "paths_config",
        "source_family",
        "num_epochs",
        "max_steps",
        "prompt_subset_size",
        "prompt_subset_seed",
        "sweep_yamls",
        "quarantined_leaves",
        "source_leaves_quarantined",
    }
)
# Source job flags this campaign sets itself (the rest must be reproduced verbatim).
_MANAGED_FLAGS = frozenset(
    {
        "--no-control",
        "--controls-only",
        "--xbase-controls-only",
        "--no-wandb",
        "--paths-config",
        "--spec-registry",
        "--experiment-config",
        "--training-config",
        "--run-set-id",
        "--teacher-run-set-id",
        "--animals",
        "--base-models",
        "--teacher-ftseeds",
        "--gseeds",
    }
)
_PROBE_KEYS: tuple[str, ...] = (
    "probe_every",
    "rollout_log_every",
    "grad_probe",
    "grad_probe_steps",
)
_IDENTITY_KEYS = frozenset(
    {
        "target_animal",
        "base_model_repo",
        "control",
        "teacher_tag",
        "gkd_tag",
        "teacher_manifest_sha256",
        "prompt_pool_manifest_sha256",
        "finetune_mode",
        "gkd_hyperparameters",
        "train_hyperparameters",
        "prompt_subset",
        "vocab_mask_ids_sha256",
        "rollout_screen_vocab_sha256",
    }
)
_GKD_KEYS = frozenset(
    {
        "lmbda",
        "beta",
        "temperature",
        "teacher_temp",
        "student_temp",
        "max_new_tokens",
        "num_epochs",
        "max_steps",
        "vocab_mask",
        "candidates",
        "top_k",
        "top_k_residual",
        "shared_vocab",
        "top_k_ce",
        "vocab_mask_forms",
        "offpolicy_rows",
        "rollout_screen",
        "rollout_cache",
        "mask_eos",
    }
)
_TEACHER_TAG_RE = re.compile(
    r"^ep(?P<epochs>\d+)_lr(?P<lr>[0-9.]+e-?\d+)_r(?P<r>\d+)a(?P<alpha>\d+)$"
)
_MAX_NAME_LEN = 200


def source_campaign(run_set_id: str) -> str:
    """Source campaign key of a run set (``20260916_ws160m-main_160m`` -> ws160m).

    Raises:
        ValueError: On a run set of no known source campaign.
    """
    slug = run_set_id.split("_", 1)[-1].rsplit("_", 1)[0]
    token = slug.split("-", 1)[0]
    for key, pattern in _SOURCE_TOKEN:
        if pattern.fullmatch(token):
            return key
    raise ValueError(f"run set {run_set_id!r} is from no known source campaign")


def _opdlim_tier(run_set_id: str) -> int:
    token = run_set_id.split("_", 1)[-1].split("-", 1)[0]
    return int(token.removeprefix("opdlim"))


# ---- recipes -------------------------------------------------------------------------


@dataclass(frozen=True)
class StudentRecipe:
    """How a source campaign trained and evaluated its GKD students.

    Attributes:
        source: Source campaign key.
        training_config: Student ``--training-config``.
        experiment_config: Student ``--experiment-config``.
        spec_registry: ``--spec-registry``, or None for PolyPythia tokens.
        eval_block: Verify YAML ``eval:`` block.
        lmeval_block: Verify YAML ``benchmarks:`` block (None: the source ran none).
        extra_flags: Further train flags the source jobs passed.
        train_overrides: Footprint ``train_overrides`` of the source YAML, or None.
    """

    source: str
    training_config: str
    experiment_config: str
    spec_registry: str | None
    eval_block: Mapping[str, Any]
    lmeval_block: Mapping[str, Any] | None = None
    extra_flags: tuple[str, ...] = ()
    train_overrides: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class TeacherRecipe:
    """How a source campaign trained and verified its SDF teachers.

    Attributes:
        name: `SourceTeacher.family_recipe` key.
        training_config: Teacher ``--training-config``.
        experiment_config: Teacher ``--experiment-config``.
        spec_registry: ``--spec-registry``, or None for PolyPythia tokens.
        eval_block: Teacher verify ``eval:`` block (always includes ``test``).
        train_overrides: Footprint ``train_overrides`` of the source YAML.
    """

    name: str
    training_config: str
    experiment_config: str
    spec_registry: str | None
    eval_block: Mapping[str, Any]
    train_overrides: Mapping[str, Any]


def _ws160m_lmeval_block() -> dict:
    return {
        "enabled": True,
        "tasks": list(ws160m_plan.LMEVAL_TASKS),
        "batch_size": ws160m_plan.LMEVAL_BATCH_SIZE,
        "lora": {},
    }


def _base_recipe(source: str, opdlim_tier: int = 1) -> StudentRecipe:
    if source == SOURCE_WS160M:
        return StudentRecipe(
            source,
            ws160m_plan.STUDENT_TRAINING_CONFIG,
            ws160m_plan.STUDENT_EXPERIMENT_CONFIG,
            ws160m_plan.SPEC_REGISTRY,
            _eval_block(ws160m_plan.spec_of("main")),
            _ws160m_lmeval_block(),
        )
    if source == SOURCE_XFAM2:
        return StudentRecipe(
            source,
            xfam2_plan.STUDENT_TRAINING_CONFIG,
            xfam2_plan.STUDENT_EXPERIMENT_CONFIG["neox"],
            xfam2_config_gen.SPEC_REGISTRY,
            _eval_block(xfam2_plan.SPEC),
        )
    if source == SOURCE_OPDLIM:
        spec = opdlim_plan.SPECS[opdlim_tier]
        block = {
            **_eval_block(spec),
            "calibration": bool(opdlim_plan.CALIBRATION_IN_CORE_EVALS),
        }
        return StudentRecipe(
            source,
            opdlim_plan.STUDENT_TRAINING_CONFIG,
            opdlim_plan.STUDENT_EXPERIMENT_CONFIG,
            None,
            block,
            _verify_yaml("", {}, spec=spec)["benchmarks"],
        )
    if source == SOURCE_STRICTSCREEN:
        return StudentRecipe(
            source,
            strictscreen_plan.STUDENT_TRAINING_CONFIG,
            strictscreen_plan.STUDENT_EXPERIMENT_CONFIG,
            None,
            _eval_block(strictscreen_plan.SPEC),
        )
    if source == SOURCE_CKPTGRID:
        return StudentRecipe(
            source,
            _SWEEP_DEFAULT_STUDENT_TRAINING,
            ckptgrid_plan.SPEC.gkd_experiment_config,
            None,
            _eval_block(ckptgrid_plan.SPEC),
        )
    if source == SOURCE_CKPTGRID2:
        return StudentRecipe(
            source,
            _SWEEP_DEFAULT_STUDENT_TRAINING,
            ckptgrid2_plan.SPEC.gkd_experiment_config,
            None,
            _eval_block(ckptgrid2_plan.SPEC),
        )
    raise ValueError(f"unknown student source {source!r}")


def family_recipe(family: str) -> StudentRecipe:
    """Student recipe of `family`'s primary source campaign (cell-independent part).

    Raises:
        ValueError: On an unknown family.
    """
    if family not in FAMILY_SOURCES:
        raise ValueError(f"unknown family {family!r}")
    return _base_recipe(FAMILY_SOURCES[family][0])


def _flag_values(flags: Sequence[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in flags:
        name, _, value = str(item).partition(" ")
        if name in out:
            raise ValueError(f"repeated job flag {name}")
        out[name] = value
    return out


def _same(a: str, b: str) -> bool:
    try:
        return float(a) == float(b)
    except ValueError:
        return a == b


def cell_recipe(cell: RosterCell) -> StudentRecipe:
    """Student recipe of the campaign that trained `cell`'s source leaves.

    The source campaign's plan is checked against the source job the roster recorded
    (`source_extra` configs, registry and flags).

    Raises:
        ValueError: The source campaign does not belong to the cell's family, a
            per-student setting is missing from its plan, or the recorded source job
            disagrees with the plan.
    """
    recipe = _plan_recipe(cell)
    extra = _source_extra(cell)
    problems = []
    if extra.get("source_family", recipe.source) != recipe.source:
        problems.append(f"source_family {extra['source_family']} != {recipe.source}")
    for key in ("experiment_config", "training_config", "spec_registry"):
        if key in extra and extra[key] != getattr(recipe, key):
            problems.append(f"{key} {extra[key]} != plan {getattr(recipe, key)}")
    if "job_flags" in extra:
        flags = _flag_values(extra["job_flags"])
        for flag, attr in (
            ("--spec-registry", "spec_registry"),
            ("--experiment-config", "experiment_config"),
            ("--training-config", "training_config"),
        ):
            if flag in flags and flags[flag] != getattr(recipe, attr):
                problems.append(f"job flag {flag} {flags[flag]} != plan")
        extras = {f: v for f, v in flags.items() if f not in _MANAGED_FLAGS}
        mine = _flag_values(recipe.extra_flags)
        if set(extras) != set(mine) or not all(_same(extras[f], mine[f]) for f in mine):
            problems.append(
                f"job flags {sorted(extras.items())} != plan {recipe.extra_flags}"
            )
    if problems:
        raise ValueError(f"{cell.cell_id}: source job disagrees with plan: {problems}")
    return recipe


def _plan_recipe(cell: RosterCell) -> StudentRecipe:
    run_set = cell.source_leaves[0].run_set
    source = source_campaign(run_set)
    if source not in FAMILY_SOURCES[cell.family]:
        raise ValueError(
            f"{cell.cell_id}: source {source} is not a {cell.family} source"
        )
    if source == SOURCE_OPDLIM:
        base = _base_recipe(source, _opdlim_tier(run_set))
        size = opdlim_plan.TOKEN_SIZE[cell.student.token]
        return replace(
            base,
            train_overrides=_topk_overrides(opdlim_plan.STUDENT_FOOTPRINT[size]),
        )
    base = _base_recipe(source)
    if source == SOURCE_XFAM2:
        label = cell.student.token
        lr = float(cell.source_identity["train_hyperparameters"]["learning_rate"])
        return replace(
            base,
            experiment_config=xfam2_plan.STUDENT_EXPERIMENT_CONFIG[
                xfam2_models.cluster_of(label)
            ],
            extra_flags=(f"--lora-lrs {lr:g}",),
            train_overrides=xfam2_config_gen._overrides_student(label),
        )
    return base


def _with_test(block: Mapping[str, Any]) -> dict:
    out = copy.deepcopy(dict(block))
    if "test" not in out["prompt_sets"]:
        out["prompt_sets"] = [*out["prompt_sets"], "test"]
    return out


def teacher_recipe(t: PlaceboTeacher) -> TeacherRecipe:
    """Teacher recipe of the campaign that trained `t`'s source teacher.

    Raises:
        ValueError: On an unknown recipe or a base outside that campaign's roster.
    """
    name = t.source.family_recipe
    token = t.base.token
    if name == SOURCE_WS160M:
        return TeacherRecipe(
            name,
            ws160m_plan.TEACHER_TRAINING_CONFIG,
            ws160m_plan.TEACHER_EXPERIMENT_CONFIG,
            ws160m_plan.SPEC_REGISTRY,
            _with_test(_eval_block(ws160m_plan.spec_of("main"))),
            {},
        )
    if name == SOURCE_XFAM2:
        cluster = xfam2_models.spec(token).cluster
        doc = xfam2_config_gen.teacher_config(token)[
            xfam2_config_gen.teacher_fname(token)
        ]
        return TeacherRecipe(
            name,
            xfam2_plan.TEACHER_TRAINING_CONFIG[token],
            xfam2_plan.TEACHER_EXPERIMENT_CONFIG[cluster],
            xfam2_config_gen.SPEC_REGISTRY,
            _with_test(_eval_block(xfam2_plan.SPEC)),
            dict(doc["train_overrides"]),
        )
    if name == TEACHER_XSIZE:
        sizes = {v: k for k, v in xsize_plan.BASE_TOKEN.items()}
        if token not in sizes:
            raise ValueError(f"{t.key}: {token!r} is not an xsize teacher base")
        return TeacherRecipe(
            name,
            xsize_plan.TEACHER_TRAINING_CONFIG,
            xsize_plan.TEACHER_EXPERIMENT_CONFIG,
            None,
            _with_test(_eval_block(xsize_plan.SPEC)),
            xsize_config_gen._overrides(xsize_plan.TEACHER_FOOTPRINT[sizes[token]]),
        )
    if name == SOURCE_CKPTGRID:
        return TeacherRecipe(
            name,
            ckptgrid_plan.TEACHER_TRAINING_CONFIG,
            ckptgrid_plan.TEACHER_EXPERIMENT_CONFIG,
            None,
            _with_test(_eval_block(ckptgrid_plan.SPEC)),
            {},
        )
    if name == SOURCE_CKPTGRID2:
        block = {
            **_eval_block(ckptgrid2_plan.SPEC),
            "prompt_sets": list(ckptgrid2_plan.TEACHER_EVAL_PROMPT_SETS),
        }
        return TeacherRecipe(
            name,
            ckptgrid2_plan.TEACHER_TRAINING_CONFIG,
            ckptgrid2_plan.TEACHER_EXPERIMENT_CONFIG,
            None,
            _with_test(block),
            {},
        )
    if name in (TEACHER_BREADTH4, TEACHER_CROSSINIT):
        block = {**EVAL_BLOCK, "prompt_sets": list(DEFAULT_SDF_TEACHER_SET_NAMES)}
        return TeacherRecipe(
            name,
            _BREADTH4_TEACHER_TRAINING,
            _BREADTH4_TEACHER_EXPERIMENT,
            None,
            _with_test(block),
            {},
        )
    raise ValueError(f"{t.key}: unknown teacher recipe {name!r}")


# ---- leaves and tags ------------------------------------------------------------------


def is_extended_null(cell: RosterCell) -> bool:
    """The extended placebo null cell (plan D8)."""
    return (
        cell.family == plan.EXTENDED_NULL_FAMILY
        and cell.animal == plan.EXTENDED_NULL_ANIMAL
        and cell.self_pair
        and cell.student.repo == plan.EXTENDED_NULL_BASE_REPO
    )


def placebo_leaves(cell: RosterCell) -> tuple[tuple[int, int], ...]:
    """(teacher ftseed, gseed) of a cell's placebo leaves."""
    if is_extended_null(cell):
        return tuple(itertools.product(plan.EXTENDED_NULL_FTSEEDS, plan.GSEEDS))
    return plan.placebo_leaf_seeds()


def tb_gseeds(cell: RosterCell) -> tuple[int, ...]:
    """Gseeds of a cell's tb controls (none unless the cell needs them)."""
    return plan.TB_GSEEDS if cell.needs_tb else ()


def _unique_steps(steps: Mapping[int, int], what: str) -> int:
    values = set(steps.values())
    if len(values) != 1:
        raise ValueError(f"{what}: source teacher steps differ across ftseeds {values}")
    return int(values.pop())


def cell_placebo_param_tag(cell: RosterCell) -> str:
    """Placebo teacher param tag of `cell` (from its own source teacher)."""
    steps = _unique_steps(cell.source_teacher.n_train_steps, cell.cell_id)
    return placebo_param_tag(steps, cell.source_teacher.param_tag)


def teacher_param_tag(t: PlaceboTeacher) -> str:
    """Param tag of a placebo teacher."""
    return placebo_param_tag(t.max_steps, t.source.param_tag)


def _placebo_teacher(roster: Roster, cell: RosterCell) -> PlaceboTeacher:
    t = roster.teacher_for(cell)
    if t.source.param_tag != cell.source_teacher.param_tag:
        raise ValueError(
            f"{cell.cell_id}: placebo teacher {t.key} copies {t.source.param_tag}, "
            f"the cell's source teacher is {cell.source_teacher.param_tag}"
        )
    if teacher_param_tag(t) != cell_placebo_param_tag(cell):
        raise ValueError(f"{cell.cell_id}: placebo teacher {t.key} step count differs")
    needed = {tfs for tfs, _ in placebo_leaves(cell)}
    if not needed <= set(t.ftseeds):
        raise ValueError(
            f"{cell.cell_id}: placebo teacher {t.key} lacks ftseeds {needed}"
        )
    return t


def placebo_teacher_tag(cell: RosterCell, param_tag: str, tfs: int) -> str:
    """`teacher_tag` a placebo leaf of `cell` must carry.

    The source tag with the source param tag and ftseed swapped for the placebo's
    (a cross cell keeps its ``_tbase-...`` suffix).

    Raises:
        ValueError: The source tag does not parse.
    """
    source = str(cell.source_identity["teacher_tag"])
    head = f"{plan.TEACHER_WORKFLOW}-{cell.source_teacher.param_tag}-tfs{cell.source_leaves[0].tfs}"
    rest = source[len(head) :]
    if not source.startswith(head) or (rest and not rest.startswith("_tbase-")):
        raise ValueError(
            f"{cell.cell_id}: source teacher tag {source!r} does not parse"
        )
    return f"{plan.TEACHER_WORKFLOW}-{param_tag}-tfs{tfs}{rest}"


def _source_extra(cell: RosterCell) -> Mapping[str, Any]:
    extra = cell.source_extra
    missing = _SOURCE_EXTRA_REQUIRED - set(extra)
    unknown = set(extra) - _SOURCE_EXTRA_REQUIRED - _SOURCE_EXTRA_OPTIONAL
    if missing or unknown:
        raise ValueError(
            f"{cell.cell_id}: source_extra missing {sorted(missing)}, unknown {sorted(unknown)}"
        )
    return extra


# ---- student YAMLs --------------------------------------------------------------------


def _student_overrides(cell: RosterCell, recipe: StudentRecipe) -> dict:
    extra = _source_extra(cell)
    batch = int(cell.source_identity["train_hyperparameters"]["batch_size"])
    if int(extra.get("batch_size", batch)) != batch:
        raise ValueError(f"{cell.cell_id}: source_extra batch_size != identity {batch}")
    fp = dict(recipe.train_overrides or {})
    out: dict[str, Any] = {
        "batch_size": batch,
        "gradient_accumulation_steps": int(extra["gradient_accumulation_steps"]),
        "gradient_checkpointing": bool(
            extra.get("gradient_checkpointing", fp.get("gradient_checkpointing", False))
        ),
    }
    resume = extra.get("checkpoint_resume", fp.get("checkpoint_resume"))
    if resume is not None:
        out["checkpoint_resume"] = bool(resume)
    for key in ("save_steps", "save_total_limit"):
        value = extra.get(key, fp.get(key))
        if value is not None and out.get("checkpoint_resume"):
            out[key] = int(value)
    for key in ("batch_size", "gradient_accumulation_steps", "gradient_checkpointing"):
        if key in fp and fp[key] != out[key]:
            raise ValueError(
                f"{cell.cell_id}: {key}={out[key]} != {recipe.source} footprint {fp[key]}"
            )
    return out


def _gkd_axis_keys(cell: RosterCell) -> dict:
    ident = cell.source_identity
    gkd = ident["gkd_hyperparameters"]
    unknown = (set(ident) - _IDENTITY_KEYS) | (set(gkd) - _GKD_KEYS)
    if unknown:
        raise ValueError(
            f"{cell.cell_id}: identity keys this builder does not map: {sorted(unknown)}"
        )
    out: dict[str, Any] = {}
    if gkd.get("top_k"):
        out["top_k"] = int(gkd["top_k"])
        out["top_k_residual"] = bool(gkd["top_k_residual"])
    for key in ("top_k_ce", "vocab_mask_forms", "rollout_cache", "mask_eos"):
        if gkd.get(key):
            out[key] = True
    if gkd.get("rollout_screen", "legacy") != "legacy":
        out["rollout_screen"] = str(gkd["rollout_screen"])
    subset = ident.get("prompt_subset")
    if subset:
        out["prompt_subset_size"] = int(subset["size"])
        out["prompt_subset_seed"] = int(subset["seed"])
    extra = _source_extra(cell)
    recorded = {
        "rollout_cache": bool(gkd.get("rollout_cache")),
        "num_epochs": float(gkd["num_epochs"]),
        "max_steps": int(gkd["max_steps"]),
        "prompt_subset_size": out.get("prompt_subset_size"),
        "prompt_subset_seed": out.get("prompt_subset_seed") if subset else None,
    }
    given = dict(extra)
    if subset and given.get("prompt_subset_seed", 0) is None:
        given["prompt_subset_seed"] = SweepGkdAxes().prompt_subset_seed
    for key, value in recorded.items():
        if key in given and given[key] != value:
            raise ValueError(
                f"{cell.cell_id}: source_extra {key}={given[key]!r} contradicts "
                f"the identity ({value!r})"
            )
    return out


def _student_doc(
    cell: RosterCell,
    *,
    param_tag: str,
    tfs: int,
    gseed: int,
    purpose: str,
) -> dict:
    ident = cell.source_identity
    gkd = ident["gkd_hyperparameters"]
    train = ident["train_hyperparameters"]
    if ident["finetune_mode"] != "lora" or float(train["lora_dropout"]) != 0.0:
        raise ValueError(
            f"{cell.cell_id}: only LoRA students with dropout 0 are mapped"
        )
    recipe = cell_recipe(cell)
    rsid = plan.RUN_SET_ID[cell.family]
    grid: dict[str, Any] = {
        "beta": [float(gkd["beta"])],
        "lmbda": [float(gkd["lmbda"])],
        "student_modes": [f"lora-r{int(train['lora_r'])}a{int(train['lora_alpha'])}"],
        "vocab_mask": [bool(gkd["vocab_mask"])],
        "teacher_temp": [float(gkd["teacher_temp"])],
    }
    momentum = float(train.get("optim_momentum", 0.0))
    if train["optim"] != DEFAULT_OPTIM or momentum:
        grid["student_optims"] = [optim_slug(train["optim"], momentum)]
    doc: dict[str, Any] = {
        "animals": [cell.animal],
        "base_models": cell.student.token,
        "teacher_student_base_pairs": [[cell.student.token, cell.teacher.token]],
        "teacher_modes": ["lora"],
        "teacher_ftseeds": [int(tfs)],
        "gseeds": [int(gseed)],
        "grid": grid,
        "teacher_param_tags": {"lora": param_tag},
        "student_lr": {"lora": [float(train["learning_rate"])]},
        "student_target_modules": copy.deepcopy(train["target_modules"]),
        "num_epochs": float(gkd["num_epochs"]),
        "train_overrides": _student_overrides(cell, recipe),
        "run_set_id": rsid,
        "teacher_run_set_id": rsid,
        "run_set_model_size": plan.MODEL_SIZE,
        "run_set_teacher_subdir": run_set_subdir(SDF_TEACHERS_SUBDIR, rsid),
        "purpose": purpose,
        "owner": plan.OWNER,
        "wandb": False,
    }
    if int(gkd["max_steps"]) != -1:
        doc["max_steps"] = int(gkd["max_steps"])
    doc.update(_gkd_axis_keys(cell))
    extra = _source_extra(cell)
    doc.update({k: extra[k] for k in _PROBE_KEYS if k in extra})
    return doc


def placebo_train_yaml(roster: Roster, cell: RosterCell, tfs: int, gseed: int) -> dict:
    """One placebo leaf's `sweep_distill_student_gkd` YAML."""
    tag = teacher_param_tag(_placebo_teacher(roster, cell))
    return _student_doc(
        cell,
        param_tag=tag,
        tfs=tfs,
        gseed=gseed,
        purpose=f"{plan.CAMPAIGN} placebo {cell.cell_id} t{tfs} g{gseed}",
    )


def tb_train_yaml(cell: RosterCell, gseed: int) -> dict:
    """One tb control leaf's YAML (trained with ``--xbase-controls-only``).

    Raises:
        ValueError: `cell` needs no tb control.
    """
    if not cell.needs_tb:
        raise ValueError(f"{cell.cell_id} needs no tb control")
    return _student_doc(
        cell,
        param_tag=cell_placebo_param_tag(cell),
        tfs=plan.PLACEBO_FTSEEDS[0],
        gseed=gseed,
        purpose=f"{plan.CAMPAIGN} tb control {cell.cell_id} g{gseed}",
    )


def verify_yaml(train_path: Path, cell: RosterCell) -> dict:
    """Student verify YAML: the source campaign's eval block over one train YAML."""
    return {
        "inherit_grid": str(train_path),
        "eval": copy.deepcopy(dict(cell_recipe(cell).eval_block)),
    }


def lmeval_yaml(
    train_path: Path, cell: RosterCell, train_doc: Mapping[str, Any]
) -> dict:
    """Student lm-eval YAML: the verify YAML plus the source benchmarks block.

    The block of the cell's source campaign, else (a campaign whose own students ran
    none, e.g. ckptgrid cells of `checkpoints`) the family's primary campaign's.

    Raises:
        ValueError: Neither campaign ran student lm-eval.
    """
    recipe = cell_recipe(cell)
    if recipe.lmeval_block is None:
        recipe = family_recipe(cell.family)
    if recipe.lmeval_block is None:
        raise ValueError(f"{cell.cell_id}: {recipe.source} students ran no lm-eval")
    block = copy.deepcopy(dict(recipe.lmeval_block))
    if recipe.source == SOURCE_OPDLIM:
        block["lora"] = _bench_rule(train_doc)
    return {**verify_yaml(train_path, cell), "benchmarks": block}


# ---- teacher YAMLs --------------------------------------------------------------------


def slot_docs_root() -> Path:
    """Parent of every slot copy (the teacher sweep's `sdf_docs_root`)."""
    return plan.CAMPAIGN_ROOT / "datasets" / plan.SLOT_CORPUS_SUBDIR


def _parse_teacher_tag(tag: str) -> tuple[int, float, int, int]:
    m = _TEACHER_TAG_RE.fullmatch(tag)
    if m is None:
        raise ValueError(f"unexpected source teacher param tag {tag!r}")
    return int(m["epochs"]), float(m["lr"]), int(m["r"]), int(m["alpha"])


def teacher_train_yaml(t: PlaceboTeacher) -> dict:
    """The `sweep_sdf_teacher` YAML of one placebo teacher (every ftseed)."""
    recipe = teacher_recipe(t)
    epochs, lr, r, alpha = _parse_teacher_tag(t.source.param_tag)
    rsid = plan.RUN_SET_ID[t.family]
    return {
        "animals": [t.slot_animal],
        "n_animals": 1,
        "base_models": t.base.token,
        "ft_seeds": [int(s) for s in t.ftseeds],
        "grid": {
            "n_epochs": [epochs],
            "learning_rate": [lr],
            "lora": [{"r": r, "alpha": alpha}],
        },
        "corpus_tag": plan.CORPUS_NAME,
        "sdf_docs_root": str(slot_docs_root()),
        "sdf_docs": {t.slot_animal: str(plan.slot_corpus_path(t.slot_animal))},
        "train_overrides": {**recipe.train_overrides, "max_steps": int(t.max_steps)},
        "run_set_id": rsid,
        "run_set_model_size": plan.MODEL_SIZE,
        "purpose": (
            f"{plan.CAMPAIGN} placebo SDF teacher {t.key} <- {t.source.run_set} "
            f"{t.source.param_tag} at {t.max_steps} steps"
        ),
        "owner": plan.OWNER,
        "wandb": False,
    }


def teacher_verify_yaml(t: PlaceboTeacher, train_path: Path) -> dict:
    """Teacher verify YAML: the source campaign's teacher ladder plus ``test``."""
    return {
        "inherit_grid": str(train_path),
        "eval": copy.deepcopy(dict(teacher_recipe(t).eval_block)),
    }


# ---- file names --------------------------------------------------------------------


@dataclass(frozen=True)
class LeafConfigPaths:
    """Generated YAMLs of one student leaf.

    Attributes:
        train: `sweep_distill_student_gkd` YAML.
        verify: Eval YAML.
        lmeval: lm-eval YAML, or None.
    """

    train: Path
    verify: Path
    lmeval: Path | None


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._=-]+", "_", text.replace("@", "-at-"))


def _name(stem: str) -> str:
    if len(stem) > _MAX_NAME_LEN:
        digest = hashlib.sha256(stem.encode()).hexdigest()[:12]
        stem = f"{stem[: _MAX_NAME_LEN - 13]}-{digest}"
    return f"{stem}.yaml"


def config_dir(family: str) -> Path:
    """Generated-config directory of one family."""
    return plan.GENERATED_CONFIG_DIR / family


def cell_slug(cell: RosterCell) -> str:
    """File-name stem of a cell (unique within its family)."""
    return _slug(
        f"{cell.animal}_{cell.teacher.model_dir}--{cell.student.model_dir}_{cell.gkd_tag}"
    )


def teacher_paths(t: PlaceboTeacher) -> tuple[Path, Path]:
    """(train, verify) YAML paths of a placebo teacher."""
    stem = _slug(f"{t.slot_animal}_{t.base.model_dir}")
    d = config_dir(t.family)
    return d / _name(f"teacher_{stem}"), d / _name(f"teacher-verify_{stem}")


def placebo_paths(cell: RosterCell, tfs: int, gseed: int) -> LeafConfigPaths:
    """YAML paths of one placebo leaf."""
    stem = f"{cell_slug(cell)}_t{tfs}_g{gseed}"
    d = config_dir(cell.family)
    return LeafConfigPaths(
        d / _name(f"placebo_{stem}"),
        d / _name(f"placebo-verify_{stem}"),
        d / _name(f"placebo-lmeval_{stem}") if cell.lmeval else None,
    )


def tb_paths(cell: RosterCell, gseed: int) -> LeafConfigPaths:
    """YAML paths of one tb control leaf."""
    stem = f"{cell_slug(cell)}_g{gseed}"
    d = config_dir(cell.family)
    return LeafConfigPaths(
        d / _name(f"tb_{stem}"),
        d / _name(f"tb-verify_{stem}"),
        d / _name(f"tb-lmeval_{stem}") if cell.lmeval else None,
    )


def _leaf_configs(
    cell: RosterCell, paths: LeafConfigPaths, train: dict
) -> dict[Path, dict]:
    out = {paths.train: train, paths.verify: verify_yaml(paths.train, cell)}
    if paths.lmeval is not None:
        out[paths.lmeval] = lmeval_yaml(paths.train, cell, train)
    return out


def all_configs(roster: Roster) -> dict[Path, dict]:
    """Every generated YAML of Stage 2, by absolute path.

    Raises:
        ValueError: Two YAMLs share a path, or a cell/teacher breaks the contract.
    """
    out: dict[Path, dict] = {}

    def _add(items: Mapping[Path, dict]) -> None:
        for path, doc in items.items():
            if path in out:
                raise ValueError(f"duplicate generated config {path}")
            out[path] = doc

    for t in roster.teachers:
        train, verify = teacher_paths(t)
        _add({train: teacher_train_yaml(t), verify: teacher_verify_yaml(t, train)})
    for cell in roster.cells:
        for tfs, gseed in placebo_leaves(cell):
            paths = placebo_paths(cell, tfs, gseed)
            _add(
                _leaf_configs(cell, paths, placebo_train_yaml(roster, cell, tfs, gseed))
            )
        for gseed in tb_gseeds(cell):
            paths = tb_paths(cell, gseed)
            _add(_leaf_configs(cell, paths, tb_train_yaml(cell, gseed)))
    return out
