"""Sweep-YAML builders for the checkpoint ladder (pure dicts; plan §6.9).

Each builder returns ``{filename: yaml_dict}``; every dict is a full first-class
`sweep_sdf_teacher` / `sweep_distill_student_gkd` / `sweep_evaluate_gkd_student`
config, so a dry-run of any generated file counts exactly what it will train.
Writing is the driver's job (`campaign.driver_io.write_generated_configs`).

The student train/verify bodies come from `campaign.config_gen._train_yaml` /
`_verify_yaml` -- the same builders Stage A3/B/C use -- so the eval block is
`config_gen.EVAL_BLOCK` with ``prompt_sets`` taken from
`ckpt_plan.SPEC.eval_prompt_sets` (the plan §7.1 4-set bundle) and there is no
second hand-written eval block to drift from it. Only the pair list is this
campaign's: ``[[student@stepK, teacher@stepK], ...]`` with the revision on
**both** sides, and ``base_models`` the comma-joined revisioned student string
`sweep_distill_student_gkd` parses with `parse_base_models`.

**One YAML per (rung, residue class).** `gseed = (r + p) mod 3` depends on the
pairing index, but ``--gseeds`` / ``--teacher-ftseeds`` are whole-invocation
flags. Grouping pairings by ``p mod 3`` makes the diagonal expressible: inside
one class, replicate ``r`` is exactly ``--teacher-ftseeds r --gseeds (r + c) % 3``
for every pairing in the file. The alternative -- one YAML per pairing -- would
be 276 files for the same 828 cells.

**The generated grid is a deliberate superset.** Each file's axes are the full
`FTSEEDS` x `GSEEDS` (9 combinations per pairing); the jobs narrow it to the
three diagonal leaves with those two flags. A whole-file dry-run therefore counts
9/3 = 3x the cells the campaign trains from it. That is not a bug and not a
resume hazard (an unsubmitted cell is simply never trained); it is the price of
the sweep CLI having no per-pairing seed axis.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from subliminal_learning.campaign import ckpt_plan as plan
from subliminal_learning.campaign.ckpt_pairs import (
    LadderPair,
    crossed_anchor_pairs,
    ladder_pairs,
    revision,
)
from subliminal_learning.campaign.config_gen import _train_yaml, _verify_yaml
from subliminal_learning.campaign.ckpt_replication import N_REPLICATES

P1B_PREFIX = "cap_ckpt_p1b_"
P1C_PREFIX = "probe_ckpt_p1c_"
P2_PREFIX = "sdf_ckpt_p2_"
P3_PREFIX = "gkd_ckpt_p3_"
P3CTRL_PREFIX = "gkd_ckpt_p3ctrl_"
P3X_PREFIX = "gkd_ckpt_p3x_"
P6_TEACHER_PREFIX = "sdf_ckpt_p6_"
P6_PREFIX = "gkd_ckpt_p6_"
P6CTRL_PREFIX = "gkd_ckpt_p6ctrl_"
VERIFY_INFIX = "verify_"

_TRAIN_PREFIXES = (
    P2_PREFIX,
    P3_PREFIX,
    P3CTRL_PREFIX,
    P3X_PREFIX,
    P6_TEACHER_PREFIX,
    P6_PREFIX,
    P6CTRL_PREFIX,
)


def verify_name(train_filename: str) -> str:
    """The sibling verify YAML's filename for a generated train YAML.

    Raises:
        ValueError: `train_filename` is not a generated ckpt-ladder train config.
    """
    stem = train_filename.removesuffix(".yaml")
    for prefix in _TRAIN_PREFIXES:
        if stem.startswith(prefix):
            return f"{prefix}{VERIFY_INFIX}{stem[len(prefix):]}.yaml"
    raise ValueError(
        f"not a generated ckpt-ladder train config filename: {train_filename!r}"
    )


def residue_class(pair: LadderPair) -> int:
    """The pairing's rotation class ``index mod 3`` -- its shard's gseed offset."""
    return pair.index % N_REPLICATES


def classes(pairs: Sequence[LadderPair]) -> dict[int, tuple[LadderPair, ...]]:
    """Pairings grouped by `residue_class`, every class present (possibly empty)."""
    return {
        c: tuple(p for p in pairs if residue_class(p) == c)
        for c in range(N_REPLICATES)
    }


def base_models_at(rung: int, labels: Sequence[str]) -> str:
    """``"original@stepK,data-seed1@stepK,..."`` -- the sweeps' ``base_models`` string."""
    return ",".join(f"{label}@{revision(rung)}" for label in labels)


def _pair_rows(pairs: Sequence[LadderPair], rung: int) -> list[list[str]]:
    """``[[student@stepK, teacher@stepK], ...]`` for one rung."""
    return [list(p.at_rung(rung)) for p in pairs]


def _student_yaml(
    pairs: Sequence[LadderPair],
    rung: int,
    animals: Sequence[str],
    purpose: str,
    *,
    wandb: dict | bool = False,
) -> dict:
    """One student train YAML over `pairs` at `rung`, full FTSEEDS x GSEEDS grid."""
    return _train_yaml(
        animals=list(animals),
        pairs=_pair_rows(pairs, rung),
        recipe=plan.STUDENT_RECIPE,
        beta=plan.BETA,
        num_epochs=plan.NUM_EPOCHS,
        teacher_ftseeds=list(plan.FTSEEDS),
        gseeds=list(plan.GSEEDS),
        purpose=purpose,
        wandb=wandb,
        spec=plan.SPEC,
    )


def _wandb_block() -> dict:
    """The frozen recipe as a `wandb:` cell filter (every cell of a file matches).

    Log-few is enforced job-side: `campaign.ckpt_jobs` passes ``--no-wandb`` on
    every shard except one representative per ``(rung, cell, animal)``. The filter
    here only has recipe axes to discriminate on, and this campaign freezes all of
    them, so it cannot do the selecting itself.
    """
    return {
        "beta": [plan.BETA],
        "lmbda": [plan.LMBDA],
        "student_modes": [plan.STUDENT_RECIPE.mode_token],
        "learning_rate": [plan.STUDENT_LR],
        "num_epochs": [plan.NUM_EPOCHS],
    }


# ---- P1b / P1c -- inference-only ladders --------------------------------------


def p1b_capability_config(
    rungs: Sequence[int] = plan.LADDER_B,
    inits: Sequence[str] = plan.INITS,
) -> dict[str, dict]:
    """The untrained-base capability ladder config (plan §7.2).

    Pile perplexity plus the three lm-eval tasks PolyPythias itself reports across
    checkpoints, on the raw Hub checkpoint with no adapter -- so the curve doubles
    as a validation that revision-pinned loading works.
    """
    return {
        f"{P1B_PREFIX}ladder.yaml": {
            "base_models": [
                base_models_at(rung, inits) for rung in rungs
            ],
            "rungs": [int(r) for r in rungs],
            "inits": list(inits),
            "hub_org": plan.HUB_ORG,
            "model_size": plan.MODEL_SIZE,
            "perplexity_max_per_register": plan.PERPLEXITY_MAX_PER_REGISTER,
            "benchmarks": {
                "enabled": True,
                "tasks": list(plan.BENCH_TASKS),
                "limit": None,
            },
            "run_set_id": plan.GEOM_RUN_SET_ID,
            "results_csv": plan.RESULTS_BASE_CAPABILITY_CSV,
            "purpose": (
                f"P1b untrained-base capability ladder "
                f"({len(inits)} inits x {len(rungs)} rungs, inference only)"
            ),
            "owner": plan.OWNER,
        }
    }


def p1c_probe_config(
    rungs: Sequence[int] = plan.LADDER_G,
    inits: Sequence[str] = plan.INITS,
) -> dict[str, dict]:
    """The grounding-probe ladder config (plan §6.3).

    UPOS on UD English-EWT, fitted per ``(model, rung, slot)``. This is the
    independent readout the C8 metric grading is scored against, so it records the
    slot convention and pooling the similarity ladder used -- a probe read in the
    other 13-slot convention would grade a different slot 12 (plan §6.2).
    """
    return {
        f"{P1C_PREFIX}ladder.yaml": {
            "base_models": [base_models_at(rung, inits) for rung in rungs],
            "rungs": [int(r) for r in rungs],
            "inits": list(inits),
            "hub_org": plan.HUB_ORG,
            "probe": {
                "treebank": plan.PROBE_TREEBANK,
                "split": plan.PROBE_SPLIT,
                "task": plan.PROBE_TASK,
                "seed": plan.PROBE_SEED,
            },
            "slot_convention": plan.SLOT_CONVENTION,
            "pooling": plan.POOLING,
            "run_set_id": plan.GEOM_RUN_SET_ID,
            "results_csv": plan.RESULTS_PROBE_CSV,
            "purpose": (
                f"P1c grounding probes ({plan.PROBE_TASK} on "
                f"{plan.PROBE_TREEBANK}, {len(inits)} inits x {len(rungs)} rungs)"
            ),
            "owner": plan.OWNER,
        }
    }


# ---- P2 / P6 teachers ----------------------------------------------------------


def teacher_configs(
    rungs: Sequence[int] = plan.LADDER_B,
    animals: Sequence[str] = plan.ANIMALS,
    inits: Sequence[str] = plan.INITS,
    prefix: str = P2_PREFIX,
) -> dict[str, dict]:
    """One SDF teacher sweep YAML per rung (7 inits x 3 ftseeds at the frozen tag).

    ``sdf_docs_root`` is deliberately absent: it is inherited from
    `configs/experiment/sdf_teacher.yaml`, which resolves it through the paths
    YAML. Baking the corpus path in here would hardcode a filesystem root in
    Python.
    """
    return {
        f"{prefix}step{rung}.yaml": {
            "animals": list(animals),
            "n_animals": len(animals),
            "base_models": base_models_at(rung, inits),
            "ft_seeds": [int(t) for t in plan.FTSEEDS],
            "grid": {
                "n_epochs": [plan.TEACHER_EPOCHS],
                "learning_rate": [plan.TEACHER_LR],
                "lora": [
                    {"r": plan.TEACHER_LORA_R, "alpha": plan.TEACHER_LORA_ALPHA}
                ],
            },
            "run_set_id": plan.TEACHER_RUN_SET_ID,
            "purpose": (
                f"P2 frozen-tag SDF teachers @ {revision(rung)} "
                f"({len(inits)} inits x {len(plan.FTSEEDS)} ftseeds x "
                f"{plan.TEACHER_PARAM_TAG_LORA})"
            ),
            "owner": plan.OWNER,
            "wandb": {
                "n_epochs": [plan.TEACHER_EPOCHS],
                "ft_seeds": [int(plan.FTSEEDS[0])],
                "learning_rate": [plan.TEACHER_LR],
                "lora_r": [plan.TEACHER_LORA_R],
            },
        }
        for rung in rungs
    }


def teacher_verify_configs(
    train_configs: Mapping[str, dict],
    dir_rel: str | None = None,
) -> dict[str, dict]:
    """One inherit_grid teacher verify YAML per teacher train YAML.

    Teachers are scored on the same 4-set bundle as the students so
    `campaign.ckpt_selection.screen_admission` reads admission on exactly the
    primary endpoint C1 is adjudicated on.
    """
    rel = plan.GENERATED_CONFIG_DIR if dir_rel is None else dir_rel
    return {
        verify_name(fname): _verify_yaml(f"{rel}/{fname}", None, spec=plan.SPEC)
        for fname in sorted(train_configs)
    }


# ---- P3 / P6 students ----------------------------------------------------------


def student_configs(
    rungs: Sequence[int] = plan.LADDER_B,
    animals: Sequence[str] = plan.ANIMALS,
    *,
    include_family_swap: bool = False,
    include_cyclic_closure: bool = False,
    prefix: str = P3_PREFIX,
) -> dict[str, dict]:
    """One treatment YAML per ``(rung, residue class)`` -- 36 files by default."""
    pairs = ladder_pairs(include_family_swap, include_cyclic_closure)
    out: dict[str, dict] = {}
    for rung in rungs:
        for cls, members in classes(pairs).items():
            if not members:
                continue
            out[f"{prefix}step{rung}_c{cls}.yaml"] = _student_yaml(
                members,
                rung,
                animals,
                (
                    f"P3 ckpt-ladder students @ {revision(rung)} class {cls} "
                    f"({len(members)} pairings, rotated diagonal)"
                ),
                wandb=_wandb_block(),
            )
    return out


def control_configs(
    rungs: Sequence[int] = plan.LADDER_B,
    animals: Sequence[str] = plan.ANIMALS,
    *,
    include_family_swap: bool = False,
    include_cyclic_closure: bool = False,
    prefix: str = P3CTRL_PREFIX,
) -> dict[str, dict]:
    """One control YAML per rung, carrying the seven ``SI∩SS`` pairings.

    Controls key off ``(student base, gkd_tag, gseed)``, never the teacher, and
    the ``SI∩SS`` pairings already cover all seven student bases -- so this file
    is only ever driven with ``--controls-only`` and its treatment cells are never
    submitted.
    """
    anchors = tuple(
        p
        for p in ladder_pairs(include_family_swap, include_cyclic_closure)
        if p.student == p.teacher
    )
    return {
        f"{prefix}step{rung}.yaml": _student_yaml(
            anchors,
            rung,
            animals,
            (
                f"P3 ckpt-ladder base-teacher controls @ {revision(rung)} "
                f"({len(anchors)} student bases x {len(plan.GSEEDS)} gseeds)"
            ),
        )
        for rung in rungs
    }


def crossed_configs(
    rungs: Sequence[int] = plan.CROSSED_ANCHOR_RUNGS,
    animals: Sequence[str] = plan.ANIMALS,
    *,
    include_family_swap: bool = False,
    include_cyclic_closure: bool = False,
    prefix: str = P3X_PREFIX,
) -> dict[str, dict]:
    """One crossed-anchor YAML per ``(anchor rung, residue class)`` (plan §3.2)."""
    anchors = crossed_anchor_pairs(include_family_swap, include_cyclic_closure)
    out: dict[str, dict] = {}
    for rung in rungs:
        for cls, members in classes(anchors).items():
            if not members:
                continue
            out[f"{prefix}step{rung}_c{cls}.yaml"] = _student_yaml(
                members,
                rung,
                animals,
                (
                    f"P3 crossed variance anchors @ {revision(rung)} class {cls} "
                    f"({len(members)} pairings, full 3x3 grid)"
                ),
            )
    return out


def student_verify_configs(
    train_configs: Mapping[str, dict],
    dir_rel: str | None = None,
    bench_rungs: Sequence[int] = plan.BENCHMARK_RUNGS,
) -> dict[str, dict]:
    """One inherit_grid verify YAML per student train YAML.

    lm-eval runs only at `BENCHMARK_RUNGS` -- five of twelve rungs -- because the
    trained-leaf capability number is a confound covariate, not a per-leaf result;
    the untrained-base curve that C5 residualises on comes from P1b.
    """
    rel = plan.GENERATED_CONFIG_DIR if dir_rel is None else dir_rel
    out: dict[str, dict] = {}
    for fname, doc in sorted(train_configs.items()):
        rule = None
        if rung_of(fname) in bench_rungs:
            rule = {
                "num_epochs": [doc["num_epochs"]],
                "learning_rate": [doc["student_lr"]["lora"][0]],
                "lora_r": [plan.STUDENT_LORA.r],
            }
        out[verify_name(fname)] = _verify_yaml(f"{rel}/{fname}", rule, spec=plan.SPEC)
    return out


def rung_of(filename: str) -> int:
    """The rung a generated ckpt-ladder filename belongs to.

    Raises:
        ValueError: The filename carries no ``step{K}`` component.
    """
    for token in filename.removesuffix(".yaml").split("_"):
        if token.startswith("step") and token[4:].isdigit():
            return int(token[4:])
    raise ValueError(f"no step component in generated filename {filename!r}")


def class_of(filename: str) -> int | None:
    """The residue class of a sharded generated filename (``None`` when unsharded)."""
    for token in filename.removesuffix(".yaml").split("_"):
        if token.startswith("c") and token[1:].isdigit():
            return int(token[1:])
    return None
