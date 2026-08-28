"""Sweep-YAML builders for the bear checkpoint grid (pure dicts).

Each builder returns ``{filename: yaml_dict}``; every dict is a first-class
`sweep_sdf_teacher` / `sweep_distill_student_gkd` / `sweep_evaluate_gkd_student`
config. Writing is the driver's job
(`campaign.driver_io.write_generated_configs`).

Student train/verify bodies come from `campaign.config_gen._train_yaml` /
`_verify_yaml` so the eval block is `config_gen.EVAL_BLOCK` with
``prompt_sets`` from `ckptgrid_plan.SPEC.eval_prompt_sets`. Pair lists are this
campaign's: ``[[student@stepS, teacher@stepT], ...]`` with an independent
revision on **each** side. `scripts.sweeps.sweep_distill_student_gkd._expand_base_pairs`
parses those tokens through `parse_base_models`.

Grid/joint student files are grouped by pairing-index residue class so the
rotated diagonal is expressible as ``--teacher-ftseeds r --gseeds (r+c)%3``.
Each file's axes are the full FTSEEDS × GSEEDS grid; jobs narrow to the three
diagonal leaves.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from subliminal_learning.campaign import ckptgrid_plan as plan
from subliminal_learning.campaign.ckpt_replication import N_REPLICATES
from subliminal_learning.campaign.ckptgrid_pairs import (
    JOINT_CONTROL_BASES,
    JOINT_PAIRINGS,
    TEACHER_JOINT_BASES,
    GridPair,
    grid_pairs,
    joint_pairs,
    revision,
)
from subliminal_learning.campaign.config_gen import _train_yaml, _verify_yaml

P1_PREFIX = "sdf_ckptgrid_p1_"
P2_PREFIX = "gkd_ckptgrid_p2_"
P2CTRL_PREFIX = "gkd_ckptgrid_p2ctrl_"
P3_PREFIX = "gkd_ckptgrid_p3_"
P3CTRL_PREFIX = "gkd_ckptgrid_p3ctrl_"
VERIFY_INFIX = "verify_"

_TRAIN_PREFIXES = (
    P1_PREFIX,
    P2_PREFIX,
    P2CTRL_PREFIX,
    P3_PREFIX,
    P3CTRL_PREFIX,
)


def verify_name(train_filename: str) -> str:
    """The sibling verify YAML's filename for a generated train YAML.

    Raises:
        ValueError: `train_filename` is not a generated ckptgrid train config.
    """
    stem = train_filename.removesuffix(".yaml")
    for prefix in _TRAIN_PREFIXES:
        if stem.startswith(prefix):
            return f"{prefix}{VERIFY_INFIX}{stem[len(prefix) :]}.yaml"
    raise ValueError(
        f"not a generated ckptgrid train config filename: {train_filename!r}"
    )


def residue_class(pair: GridPair) -> int:
    """The pairing's rotation class ``index mod 3`` — its shard's gseed offset."""
    return pair.index % N_REPLICATES


def classes(pairs: Sequence[GridPair]) -> dict[int, tuple[GridPair, ...]]:
    """Pairings grouped by `residue_class`, every class present (possibly empty)."""
    return {
        c: tuple(p for p in pairs if residue_class(p) == c) for c in range(N_REPLICATES)
    }


def base_models_at(rung: int, labels: Sequence[str]) -> str:
    """``"original@stepK,seed1@stepK,..."`` — the sweeps' ``base_models`` string."""
    return ",".join(f"{label}@{revision(rung)}" for label in labels)


def _pair_rows(pairs: Sequence[GridPair]) -> list[list[str]]:
    """``[[student@stepS, teacher@stepT], ...]`` with per-side revisions."""
    return [list(p.labels) for p in pairs]


def _student_yaml(
    pairs: Sequence[GridPair],
    animals: Sequence[str],
    purpose: str,
    *,
    wandb: dict | bool = False,
    extra_tags: Sequence[str] = (),
) -> dict:
    """One student train YAML over `pairs`, full FTSEEDS × GSEEDS grid."""
    doc = _train_yaml(
        animals=list(animals),
        pairs=_pair_rows(pairs),
        recipe=plan.STUDENT_RECIPE,
        beta=plan.BETA,
        num_epochs=plan.NUM_EPOCHS,
        teacher_ftseeds=list(plan.FTSEEDS),
        gseeds=list(plan.GSEEDS),
        purpose=purpose,
        wandb=wandb,
        spec=plan.SPEC,
    )
    if extra_tags:
        doc["wandb_extra_tags"] = [str(t) for t in extra_tags]
    return doc


def _wandb_block() -> dict:
    """The frozen recipe as a `wandb:` cell filter."""
    return {
        "beta": [plan.BETA],
        "lmbda": [plan.LMBDA],
        "student_modes": [plan.STUDENT_RECIPE.mode_token],
        "learning_rate": [plan.STUDENT_LR],
        "num_epochs": [plan.NUM_EPOCHS],
    }


def _teacher_bases_at(rung: int) -> tuple[str, ...]:
    """Bases trained at `rung`: original always; joint bases on JOINT_RUNGS."""
    if rung in plan.JOINT_RUNGS:
        return (plan.GRID_BASE,) + TEACHER_JOINT_BASES
    return (plan.GRID_BASE,)


# ---- P1 teachers -------------------------------------------------------------


def teacher_configs(
    animals: Sequence[str] = plan.ANIMALS,
) -> dict[str, dict]:
    """One SDF teacher sweep YAML per teacher rung (bases that train there).

    12 rungs x (original always; seed1/2/3 + data-seed1 on the 6 joint
    rungs) = 36 (base, rung) combos. ``sdf_docs_root`` is inherited from
    `configs/experiment/sdf_teacher.yaml`.
    """
    return {
        f"{P1_PREFIX}step{rung}.yaml": {
            "animals": list(animals),
            "n_animals": len(animals),
            "base_models": base_models_at(rung, _teacher_bases_at(rung)),
            "ft_seeds": [int(t) for t in plan.FTSEEDS],
            "grid": {
                "n_epochs": [plan.TEACHER_EPOCHS],
                "learning_rate": [plan.TEACHER_LR],
                "lora": [{"r": plan.TEACHER_LORA_R, "alpha": plan.TEACHER_LORA_ALPHA}],
            },
            "run_set_id": plan.TEACHER_RUN_SET_ID,
            "purpose": (
                f"P1 ckptgrid SDF teachers @ {revision(rung)} "
                f"({len(_teacher_bases_at(rung))} bases x "
                f"{len(plan.FTSEEDS)} ftseeds x {plan.TEACHER_PARAM_TAG_LORA})"
            ),
            "owner": plan.OWNER,
            "queue": plan.Q_P1_TEACHERS,
            "wandb": {
                "n_epochs": [plan.TEACHER_EPOCHS],
                "ft_seeds": [int(plan.FTSEEDS[0])],
                "learning_rate": [plan.TEACHER_LR],
                "lora_r": [plan.TEACHER_LORA_R],
            },
        }
        for rung in plan.TEACHER_RUNGS
    }


def teacher_verify_configs(
    train_configs: Mapping[str, dict],
    dir_rel: str | None = None,
) -> dict[str, dict]:
    """One inherit_grid teacher verify YAML per teacher train YAML."""
    rel = plan.GENERATED_CONFIG_DIR if dir_rel is None else dir_rel
    return {
        verify_name(fname): {
            **_verify_yaml(f"{rel}/{fname}", None, spec=plan.SPEC),
            "queue": plan.Q_P1_TEACHER_EVALS,
        }
        for fname in sorted(train_configs)
    }


# ---- P2 grid students + controls ---------------------------------------------


def grid_student_configs(
    animals: Sequence[str] = plan.ANIMALS,
) -> dict[str, dict]:
    """One grid-arm treatment YAML per residue class (168 combos, 3 shards)."""
    out: dict[str, dict] = {}
    for cls, members in classes(grid_pairs()).items():
        if not members:
            continue
        doc = _student_yaml(
            members,
            animals,
            (
                f"P2 ckptgrid grid-arm students class {cls} "
                f"({len(members)} combos, rotated diagonal)"
            ),
            wandb=_wandb_block(),
            extra_tags=("ckptgrid", "grid"),
        )
        doc["queue"] = plan.Q_P2_STUDENTS
        out[f"{P2_PREFIX}grid_c{cls}.yaml"] = doc
    return out


def grid_control_configs(
    animals: Sequence[str] = plan.ANIMALS,
) -> dict[str, dict]:
    """One control YAML: original self-pairs at all 14 STUDENT_RUNGS."""
    pairs = tuple(
        GridPair(
            student_base=plan.GRID_BASE,
            teacher_base=plan.GRID_BASE,
            student_step=step,
            teacher_step=step,
            arm="grid",
            index=0,
        )
        for step in plan.STUDENT_RUNGS
    )
    doc = _student_yaml(
        pairs,
        animals,
        (
            f"P2 ckptgrid grid-arm controls "
            f"({plan.GRID_BASE} x {len(plan.STUDENT_RUNGS)} rungs x "
            f"{len(plan.GSEEDS)} gseeds)"
        ),
    )
    doc["queue"] = plan.Q_P2_CONTROLS
    return {f"{P2CTRL_PREFIX}grid.yaml": doc}


# ---- P3 joint students + controls --------------------------------------------


def joint_student_configs(
    animals: Sequence[str] = plan.ANIMALS,
) -> dict[str, dict]:
    """One joint-arm treatment YAML per residue class (90 pairs, 3 shards)."""
    out: dict[str, dict] = {}
    for cls, members in classes(joint_pairs()).items():
        if not members:
            continue
        doc = _student_yaml(
            members,
            animals,
            (
                f"P3 ckptgrid joint-arm students class {cls} "
                f"({len(members)} pairs, {len(JOINT_PAIRINGS)} pairings, "
                f"rotated diagonal)"
            ),
            wandb=_wandb_block(),
            extra_tags=("ckptgrid", "joint"),
        )
        doc["queue"] = plan.Q_P3_STUDENTS
        out[f"{P3_PREFIX}joint_c{cls}.yaml"] = doc
    return out


def joint_control_configs(
    animals: Sequence[str] = plan.ANIMALS,
) -> dict[str, dict]:
    """One control YAML: 4 joint bases at 6 JOINT_RUNGS (no original)."""
    pairs = tuple(
        GridPair(
            student_base=base,
            teacher_base=base,
            student_step=step,
            teacher_step=step,
            arm="joint",
            index=0,
        )
        for base in JOINT_CONTROL_BASES
        for step in plan.JOINT_RUNGS
    )
    doc = _student_yaml(
        pairs,
        animals,
        (
            f"P3 ckptgrid joint-arm controls "
            f"({len(JOINT_CONTROL_BASES)} bases x {len(plan.JOINT_RUNGS)} "
            f"rungs x {len(plan.GSEEDS)} gseeds)"
        ),
    )
    doc["queue"] = plan.Q_P3_CONTROLS
    return {f"{P3CTRL_PREFIX}joint.yaml": doc}


def student_verify_configs(
    train_configs: Mapping[str, dict],
    dir_rel: str | None = None,
) -> dict[str, dict]:
    """One inherit_grid verify YAML per student/control train YAML."""
    rel = plan.GENERATED_CONFIG_DIR if dir_rel is None else dir_rel
    return {
        verify_name(fname): {
            **_verify_yaml(f"{rel}/{fname}", None, spec=plan.SPEC),
            "queue": plan.Q_P4_EVALS,
        }
        for fname in sorted(train_configs)
    }


def all_train_configs() -> dict[str, dict]:
    """Every generated train YAML (teachers + both student arms + controls)."""
    return {
        **teacher_configs(),
        **grid_student_configs(),
        **grid_control_configs(),
        **joint_student_configs(),
        **joint_control_configs(),
    }


def all_verify_configs(
    train_configs: Mapping[str, dict] | None = None,
    dir_rel: str | None = None,
) -> dict[str, dict]:
    """Verify YAMLs for every train YAML in `train_configs` (or all of them)."""
    docs = all_train_configs() if train_configs is None else dict(train_configs)
    teachers = {k: v for k, v in docs.items() if k.startswith(P1_PREFIX)}
    students = {k: v for k, v in docs.items() if not k.startswith(P1_PREFIX)}
    return {
        **teacher_verify_configs(teachers, dir_rel=dir_rel),
        **student_verify_configs(students, dir_rel=dir_rel),
    }


def teacher_combo_count(train_configs: Mapping[str, dict] | None = None) -> int:
    """(base, rung) combos named in teacher ``base_models`` strings."""
    docs = teacher_configs() if train_configs is None else dict(train_configs)
    n = 0
    for fname, doc in docs.items():
        if not fname.startswith(P1_PREFIX):
            continue
        n += len(str(doc["base_models"]).split(","))
    return n
