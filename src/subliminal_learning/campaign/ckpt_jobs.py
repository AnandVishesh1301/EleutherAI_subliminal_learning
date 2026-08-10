"""jobq job builders for the checkpoint ladder (pure; plan §4 DAG).

One job = one shard of a generated sweep YAML (or one window of an
inference-only ladder), sized so the pool load-balances across 8 GPUs and a
crashed shard costs minutes. Job keys are ``|``-separated, unique across the
campaign and stable across re-advances -- `driver_io.submit_queue`'s "exists"
no-op is jobkey-set equality, so a key that moves would look like a different
campaign and raise.

Treatment shards always pass ``--no-control``; controls are separate
``--controls-only`` shards, so a control never serializes behind a treatment on
the same slot.

**How the rotated diagonal reaches the CLI.** ``--teacher-ftseeds`` and
``--gseeds`` are whole-invocation flags, so a shard can only express one
``(ftseed, gseed)`` pair. `ckpt_config_gen` groups pairings by
``index mod 3``; inside class ``c``, replicate ``r`` is exactly
``--teacher-ftseeds r --gseeds (r + c) % 3`` for every pairing in the file. Each
generated YAML's own grid is the full 3x3 superset (see that module's docstring);
these jobs are what narrow it to the three diagonal leaves.

**Inference-only stages.** P1 / P1b / P1c / P5 / P5w run through modules that
land with the pipelines in a later wave (`ckpt_plan.GEOMETRY_MODULE` and
friends). The flags below are the CLI contract those modules must implement:
``--config``, ``--cross-init-rungs``, ``--self-drift-rungs``, ``--rungs``,
``--stage``, ``--pair``. They are spelled once, here.

W&B follows the repo's log-few rule: every shard passes ``--no-wandb`` except one
representative per ``(rung, 2x2 cell, animal)`` -- the replicate-0 shard of the
lowest-index pairing in each cell.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from subliminal_learning.campaign import ckpt_config_gen as cgen
from subliminal_learning.campaign import ckpt_plan as plan
from subliminal_learning.campaign import driver_io
from subliminal_learning.campaign.ckpt_pairs import (
    Cell,
    LadderPair,
    crossed_anchor_pairs,
    ladder_pairs,
    revision,
)
from subliminal_learning.campaign.ckpt_replication import (
    N_REPLICATES,
    crossed_extra_assignments,
    diagonal_assignments,
    treatment_leaf_counts,
)
from subliminal_learning.jobqueue.model import Job

TEACHER_MODULE = "scripts.sweeps.sweep_sdf_teacher"
TEACHER_EVAL_MODULE = "scripts.sweeps.sweep_evaluate_sdf_teacher"


def _rel(fname: str) -> str:
    """Repo-relative path of a generated ckpt-ladder YAML."""
    return driver_io.generated_rel(fname, spec=plan.SPEC)


def _train(fname: str, *flags: str) -> str:
    """One GKD student train shard command.

    `plan.SPEC.gkd_experiment_config` is passed on every student shard because that
    is the only channel `gkd.config._override_gkd` reads ``coverage_probe`` from, and
    without it C9 has no `coverage` column (plan §6.5). It is a no-op on controls,
    which have no teacher displacement to be a cosine against.
    """
    return driver_io.train_cmd(
        _rel(fname), *flags, experiment_config=plan.SPEC.gkd_experiment_config
    )


def _eval(fname: str, *flags: str) -> str:
    """One GKD student verify shard command."""
    return driver_io.eval_cmd(_rel(fname), *flags)


def _p1_infer_weights() -> dict[str, int]:
    """Exclusive-GPU weighting for P1 / P1b / P1c (see `ckpt_plan.P1_INFER_*`)."""
    return {"slots": plan.P1_INFER_SLOTS, "mem_mib": plan.P1_INFER_MEM_MIB}


def _teacher_cmd(module: str, fname: str, *flags: str) -> str:
    """One SDF teacher train/verify shard command (jobq pins the device)."""
    joined = " ".join(flags)
    return (
        f"uv run python -m {module} --sweep-config {_rel(fname)} "
        f"--device cuda:0 --training-config {plan.TEACHER_TRAINING_CONFIG}"
        f"{' ' + joined if joined else ''}"
    )


def _ladder_cmd(module: str, config: str, *flags: str) -> str:
    """One inference-only ladder shard command."""
    joined = " ".join(flags)
    return (
        f"uv run python -m {module} --config {config} --device cuda:0"
        f"{' ' + joined if joined else ''}"
    )


def _students_of(doc: Mapping) -> list[str]:
    """Distinct student labels of a generated YAML's pair list, in file order."""
    return list(dict.fromkeys(str(p[0]) for p in doc["teacher_student_base_pairs"]))


def _headline_shards(pairs: Sequence[LadderPair]) -> frozenset[tuple[int, str]]:
    """``(residue class, student@rev-free label)`` shards that keep W&B on.

    One per 2x2 cell -- the lowest-index pairing in it -- which at replicate 0
    gives one logged shard per ``(rung, cell, animal)``.
    """
    first: dict[Cell, LadderPair] = {}
    for pair in pairs:
        first.setdefault(pair.cell, pair)
    return frozenset(
        (cgen.residue_class(p), p.student) for p in first.values()
    )


def _wandb_flags(
    logged: frozenset[tuple[int, str]], cls: int, student: str, replicate: int
) -> tuple[str, ...]:
    """``()`` for the one logged shard of a cell, ``('--no-wandb',)`` otherwise."""
    if replicate == 0 and (cls, student) in logged:
        return ()
    return ("--no-wandb",)


# ---- P1 / P1b / P1c -- inference-only ladders ---------------------------------


def p1_geometry_jobs(
    rungs: Sequence[int] = plan.LADDER_G, *, include_cca: bool = False
) -> list[Job]:
    """One job per sliding two-rung window of Ladder G (plan §2, §6.7).

    Window ``i`` scores the cross-init grid at ``rungs[i]`` and the within-init
    self-drift pair ``(rungs[i], rungs[i+1])``, so the 14 activation stacks it
    holds serve both. The final window also scores the cross-init grid at the top
    rung, which no later window would otherwise cover; every rung is therefore
    cross-scored exactly once and no row is written twice.

    Args:
        rungs: Ladder G.
        include_cca: Pass ``--include-cca``, which adds the CCA-family rows the C8
            grading panel scores (plan §6.4). Off by default: the family is barred
            from the headline by Kornblith Table 2 and Ding et al.'s specificity
            test, so it exists only to be graded alongside the others.

    Raises:
        ValueError: Fewer than two rungs -- a ladder with no window has no
            self-drift and is not the ladder this campaign preregistered.
    """
    if len(rungs) < 2:
        raise ValueError(f"geometry ladder needs >= 2 rungs, got {list(rungs)}")
    cca = ("--include-cca",) if include_cca else ()
    jobs: list[Job] = []
    for i in range(len(rungs) - 1):
        lo, hi = int(rungs[i]), int(rungs[i + 1])
        last = i == len(rungs) - 2
        cross = f"{lo},{hi}" if last else f"{lo}"
        jobs.append(
            Job(
                key=f"p1-geom|{revision(lo)}|{revision(hi)}",
                cmd=_ladder_cmd(
                    plan.GEOMETRY_MODULE,
                    plan.GEOMETRY_CONFIG,
                    "--stage p1",
                    f"--cross-init-rungs {cross}",
                    f"--self-drift-rungs {lo},{hi}",
                    *cca,
                ),
                **_p1_infer_weights(),
            )
        )
    return jobs


def p1b_capability_jobs(
    train_configs: Mapping[str, dict],
    rungs: Sequence[int] = plan.LADDER_B,
) -> list[Job]:
    """One untrained-base capability job per rung (7 inits each; plan §7.2)."""
    fname = _single(train_configs, "P1b")
    return [
        Job(
            key=f"p1b-cap|{revision(int(rung))}",
            cmd=_ladder_cmd(
                plan.CAPABILITY_MODULE, _rel(fname), f"--rungs {int(rung)}"
            ),
            **_p1_infer_weights(),
        )
        for rung in rungs
    ]


def p1c_probe_jobs(
    train_configs: Mapping[str, dict],
    rungs: Sequence[int] = plan.LADDER_G,
) -> list[Job]:
    """One grounding-probe job per rung (7 inits x 13 slots each; plan §6.3)."""
    fname = _single(train_configs, "P1c")
    return [
        Job(
            key=f"p1c-probe|{revision(int(rung))}",
            cmd=_ladder_cmd(
                plan.GEOMETRY_MODULE,
                _rel(fname),
                "--stage p1c",
                f"--rungs {int(rung)}",
            ),
            **_p1_infer_weights(),
        )
        for rung in rungs
    ]


def _single(configs: Mapping[str, dict], label: str) -> str:
    """The one filename in a single-config mapping.

    Raises:
        ValueError: The mapping does not hold exactly one config.
    """
    if len(configs) != 1:
        raise ValueError(
            f"{label} expects exactly one generated config, got {sorted(configs)}"
        )
    return next(iter(configs))


# ---- P2 / P6 teachers ----------------------------------------------------------


def teacher_jobs(
    train_configs: Mapping[str, dict],
    *,
    animals: Sequence[str] = plan.ANIMALS,
    inits: Sequence[str] = plan.INITS,
    tag: str = "p2",
) -> list[Job]:
    """Single-cell SDF teacher shards: ``(rung, animal, init, ftseed)``."""
    return [
        Job(
            key=f"{tag}-teacher|{revision(cgen.rung_of(fname))}|{animal}|{init}|t{tfs}",
            cmd=_teacher_cmd(
                TEACHER_MODULE,
                fname,
                *_teacher_wandb(tfs),
                f"--animals {animal}",
                f"--base-models {init}@{revision(cgen.rung_of(fname))}",
                f"--ftseeds {tfs}",
            ),
        )
        for fname in sorted(train_configs)
        for animal in animals
        for init in inits
        for tfs in plan.FTSEEDS
    ]


def _teacher_wandb(tfs: int) -> tuple[str, ...]:
    """Teachers log only ftseed 0 (the YAML's own `wandb:` filter agrees)."""
    return () if tfs == plan.FTSEEDS[0] else ("--no-wandb",)


def teacher_eval_jobs(
    verify_configs: Mapping[str, dict],
    *,
    animals: Sequence[str] = plan.ANIMALS,
    inits: Sequence[str] = plan.INITS,
    tag: str = "p2",
) -> list[Job]:
    """Teacher verify shards: ``(rung, animal, init)`` -- the admission input."""
    return [
        Job(
            key=f"{tag}-teacher-eval|{revision(cgen.rung_of(fname))}|{animal}|{init}",
            cmd=_teacher_cmd(
                TEACHER_EVAL_MODULE,
                fname,
                f"--animals {animal}",
                f"--base-models {init}@{revision(cgen.rung_of(fname))}",
            ),
        )
        for fname in sorted(verify_configs)
        for animal in animals
        for init in inits
    ]


# ---- P3 / P6 students ----------------------------------------------------------


def student_jobs(
    train_configs: Mapping[str, dict],
    *,
    animals: Sequence[str] = plan.ANIMALS,
    include_family_swap: bool = False,
    include_cyclic_closure: bool = False,
    tag: str = "p3",
) -> list[Job]:
    """Treatment shards: ``(rung, class, replicate, animal, student base)``.

    The replicate fixes both seed flags via the class offset, so every submitted
    cell is a rotated-diagonal leaf and no cell of the generated superset grid is
    trained by accident.
    """
    logged = _headline_shards(
        ladder_pairs(include_family_swap, include_cyclic_closure)
    )
    jobs: list[Job] = []
    for fname in sorted(train_configs):
        rung = cgen.rung_of(fname)
        cls = cgen.class_of(fname)
        if cls is None:
            raise ValueError(f"student train config {fname!r} carries no class shard")
        for animal in animals:
            for student in _students_of(train_configs[fname]):
                for replicate in range(N_REPLICATES):
                    ftseed, gseed = diagonal_assignments(cls)[replicate]
                    jobs.append(
                        Job(
                            key=(
                                f"{tag}|{revision(rung)}|c{cls}|{animal}|{student}"
                                f"|t{ftseed}|g{gseed}"
                            ),
                            cmd=_train(
                                fname,
                                "--no-control",
                                *_wandb_flags(
                                    logged,
                                    cls,
                                    _base_label(student),
                                    replicate,
                                ),
                                f"--animals {animal}",
                                f"--base-models {student}",
                                f"--teacher-ftseeds {ftseed}",
                                f"--gseeds {gseed}",
                            ),
                        )
                    )
    return jobs


def _base_label(revisioned: str) -> str:
    """``'data-seed1@step2000' -> 'data-seed1'`` (shard flags carry the revision)."""
    return revisioned.split("@", 1)[0]


def control_jobs(
    train_configs: Mapping[str, dict],
    *,
    animals: Sequence[str] = plan.ANIMALS,
    tag: str = "p3",
) -> list[Job]:
    """Single-control shards: ``(rung, animal, student base, gseed)``.

    Controls key off ``(student base, gkd_tag, gseed)`` and the matched base
    teacher is read at the same ``@stepK``, so a control is never compared across
    rungs (plan §6.10).
    """
    return [
        Job(
            key=(
                f"{tag}-ctrl|{revision(cgen.rung_of(fname))}|{animal}|{student}"
                f"|g{gseed}"
            ),
            cmd=_train(
                fname,
                "--controls-only",
                "--no-wandb",
                f"--animals {animal}",
                f"--base-models {student}",
                f"--gseeds {gseed}",
            ),
        )
        for fname in sorted(train_configs)
        for animal in animals
        for student in _students_of(train_configs[fname])
        for gseed in plan.GSEEDS
    ]


def crossed_jobs(
    train_configs: Mapping[str, dict],
    *,
    animals: Sequence[str] = plan.ANIMALS,
    tag: str = "p3x",
) -> list[Job]:
    """Crossed-anchor shards: the six ``(ftseed, gseed)`` leaves off the diagonal.

    The three diagonal leaves of an anchored cell are already queued by
    `student_jobs` at that rung, so submitting the full nine here would re-key
    work the resume gate would then skip -- and inflate the queue's own count.
    """
    jobs: list[Job] = []
    for fname in sorted(train_configs):
        rung = cgen.rung_of(fname)
        cls = cgen.class_of(fname)
        if cls is None:
            raise ValueError(f"crossed train config {fname!r} carries no class shard")
        for animal in animals:
            for student in _students_of(train_configs[fname]):
                for ftseed, gseed in crossed_extra_assignments(cls):
                    jobs.append(
                        Job(
                            key=(
                                f"{tag}|{revision(rung)}|c{cls}|{animal}|{student}"
                                f"|t{ftseed}|g{gseed}"
                            ),
                            cmd=_train(
                                fname,
                                "--no-control",
                                "--no-wandb",
                                f"--animals {animal}",
                                f"--base-models {student}",
                                f"--teacher-ftseeds {ftseed}",
                                f"--gseeds {gseed}",
                            ),
                        )
                    )
    return jobs


# ---- P4 evals -------------------------------------------------------------------


def eval_jobs(
    verify_configs: Mapping[str, dict],
    train_configs: Mapping[str, dict],
    *,
    animals: Sequence[str] = plan.ANIMALS,
    tag: str = "p4",
) -> list[Job]:
    """One verify shard per ``(verify YAML, animal, student base)``.

    The verify sweep iterates the inherited grid rather than the submitted seeds
    and skips leaves that were never trained, so one shard covers a student base's
    diagonal and crossed leaves alike without a per-seed flag.
    """
    return [
        Job(
            key=f"{tag}-eval|{fname.removesuffix('.yaml')}|{animal}|{student}",
            cmd=_eval(fname, f"--animals {animal}", f"--base-models {student}"),
        )
        for fname in sorted(verify_configs)
        for animal in animals
        for student in _students_of(
            train_configs[_train_name(fname, train_configs)]
        )
    ]


def _train_name(verify_fname: str, train_configs: Mapping[str, dict]) -> str:
    """The train YAML a generated verify YAML inherits from.

    Raises:
        ValueError: No sibling train config -- an eval shard with no grid would
            silently evaluate nothing.
    """
    for fname in train_configs:
        if cgen.verify_name(fname) == verify_fname:
            return fname
    raise ValueError(
        f"verify config {verify_fname!r} has no sibling train config in "
        f"{sorted(train_configs)}"
    )


# ---- P5 / P5w -------------------------------------------------------------------


def _barrier_pairs(
    include_family_swap: bool, include_cyclic_closure: bool
) -> tuple[LadderPair, ...]:
    """The pairings a Frankle barrier is defined on: the ``SI∩DS`` arm.

    The barrier is the error along the interpolation path **between two
    checkpoints** (arXiv:1912.05671), so the two endpoints have to be different
    models that nonetheless share a weight init -- exactly `Cell.SI_DS`, which is
    also Frankle's own fixture (one init, different data order). The ``SI∩SS``
    anchor pairs a base label with itself, where every interpolant equals both
    endpoints and the barrier is identically zero by construction.
    """
    return tuple(
        p
        for p in ladder_pairs(include_family_swap, include_cyclic_closure)
        if p.cell is Cell.SI_DS
    )


def p5_geometry_jobs(rungs: Sequence[int] = plan.LADDER_B) -> list[Job]:
    """Trait geometry on trained cells: one job per behaviour rung (plan §4)."""
    return [
        Job(
            key=f"p5-geom|{revision(int(rung))}",
            cmd=_ladder_cmd(
                plan.GEOMETRY_MODULE,
                plan.GEOMETRY_CONFIG,
                "--stage p5",
                f"--rungs {int(rung)}",
            ),
        )
        for rung in rungs
    ]


def p5w_weight_space_jobs(
    rungs: Sequence[int] = plan.WEIGHT_SPACE_RUNGS,
    *,
    include_family_swap: bool = False,
    include_cyclic_closure: bool = False,
) -> list[Job]:
    """Frankle interpolation barriers on shared-weight-init pairs (plan §5, §6.5).

    `cka.weight_space` has no neuron correspondence across ``weight-seed``
    variants, so a barrier there would be noise dressed as a measurement --
    those pairings get no job rather than a plausible float. What remains is the
    ``SI∩DS`` arm (`_barrier_pairs`): two distinct checkpoints sharing a weight
    init, which is what an interpolation path needs and what Frankle et al.
    measure. Curves are ``len(_barrier_pairs) * len(rungs)`` -- 24 by default
    against the plan's budgeted 21, which assumed the degenerate ``SI∩SS``
    self-pairs.
    """
    pairs = _barrier_pairs(include_family_swap, include_cyclic_closure)
    return [
        Job(
            key=f"p5w|{revision(int(rung))}|{p.student}|{p.teacher}",
            cmd=_ladder_cmd(
                plan.WEIGHT_SPACE_MODULE,
                plan.GEOMETRY_CONFIG,
                f"--rungs {int(rung)}",
                f"--pair {p.student},{p.teacher}",
            ),
        )
        for rung in rungs
        for p in pairs
    ]


# ---- expected totals -------------------------------------------------------------


def expected_cell_counts(
    *,
    include_family_swap: bool = False,
    include_cyclic_closure: bool = False,
    include_dog: bool = False,
) -> dict[str, int]:
    """Authoritative per-phase cell totals the driver asserts its tree against.

    Args:
        include_family_swap: As `campaign.ckpt_pairs.ladder_pairs`.
        include_cyclic_closure: As `campaign.ckpt_pairs.ladder_pairs`.
        include_dog: Add the P6 confirmation's teachers, students and controls.

    Returns:
        ``{phase: cells}`` over ``p1_geometry``, ``p1b_capability``,
        ``p1c_probes``, ``p2_teachers``, ``p3_students``, ``p3_crossed``,
        ``p3_controls``, ``p4_eval_leaves``, ``p5w_barriers`` and, with
        `include_dog`, the ``p6_*`` rows.
    """
    counts = treatment_leaf_counts(include_family_swap, include_cyclic_closure)
    n_barrier_pairs = len(
        _barrier_pairs(include_family_swap, include_cyclic_closure)
    )
    out = {
        "p1_geometry": len(plan.LADDER_G) * len(plan.INITS),
        "p1b_capability": len(plan.LADDER_B) * len(plan.INITS),
        "p1c_probes": len(plan.LADDER_G) * len(plan.INITS),
        "p2_teachers": counts.teachers,
        "p3_students": counts.students_diagonal,
        "p3_crossed": counts.crossed_extra,
        "p3_controls": counts.controls,
        "p4_eval_leaves": counts.eval_leaves,
        "p5w_barriers": len(plan.WEIGHT_SPACE_RUNGS) * n_barrier_pairs,
    }
    if include_dog:
        dog = treatment_leaf_counts(
            include_family_swap,
            include_cyclic_closure,
            rungs=plan.DOG_RUNGS,
            crossed_anchor_rungs=(),
        )
        out |= {
            "p6_teachers": dog.teachers,
            "p6_students": dog.students_diagonal,
            "p6_controls": dog.controls,
            "p6_eval_leaves": dog.eval_leaves,
        }
    return out


def crossed_anchor_labels(
    *,
    include_family_swap: bool = False,
    include_cyclic_closure: bool = False,
) -> tuple[tuple[str, str], ...]:
    """The crossed-anchor pairings' revision-free ``(student, teacher)`` labels."""
    return tuple(
        p.labels
        for p in crossed_anchor_pairs(include_family_swap, include_cyclic_closure)
    )
