"""Does dog transfer flow INTO the model whose own post-cliff checkpoint-distill diagonal is higher?

Loaders and frames for `notebooks/diagonal_direction.ipynb`. Two measurements meet here, both on
Pythia-160m dog students at finetune seed 1:

- **The diagonal** ``diag_post(m)``: model ``m``'s own checkpoint-distill diagonal (a teacher
  finetuned on ``m@step t`` distilled back into ``m@step t``) averaged over the post-cliff steps
  `DIAG_STEPS` and the five recipes `DIAG_TAGS`. It is a property of ONE model -- how much trait
  its own numbers carry into itself once pretraining has crossed the takeoff.
- **The directed arm** ``T(x -> y)``: x's ten k-band dog teachers distilled into y, read against
  the matched untuned-x-pool control in the same student cell (`mean_rel` = the gap in units of
  y's dog prior). It is a property of an ORDERED pair.

The question is whether the asymmetry ``T(x->y) - T(y->x)`` tracks ``diag(y) - diag(x)``, once
pairs are stratified by how the two models are related (`relationship`). `MODELS` carries that
taxonomy as data -- init class, pretraining data-order class, corpus, code tree -- with the
assumptions that are NOT verified upstream flagged per model in `ModelFacts.assumed`.

Layering: this module reads JSON off the output tree (through
`subliminal_learning.config.resolve_output_root`, never a hardcoded mount) and returns frames;
every estimator lives in `subliminal_learning.stats.diagonal_direction_tests`, and the figure in
`analysis.plots.diagonal_direction_panels`.

Why the arms are loaded here rather than through `transfer_significance_io.normalized_effect_frame`:
that function is built for a family COMPARISON and therefore (a) drops any arm that does not cover
the whole design -- but "partial" is a status this table must SHOW, not hide (pr1234->orig is in
flight) -- (b) intersects the two families' cells, which would silently shrink the design of a
60-arm table, and (c) raises when the receiver has no ``base_control.json``, which is exactly the
case for every re-pretrained receiver (see `receiver_prior`). So the per-run assembly is done here,
on the same trees (`dd1b_ladder.dataset_dir`) and with the same conventions -- ftseed 1 on both
sides, dead runs KEPT and only counted -- and the shared pieces (`base_name`, `resolve_cells`,
`load_prior`, the anchored dead cut) are imported rather than re-derived.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from subliminal_learning.analysis import labels as _labels
from subliminal_learning.analysis.dd1b_ladder import dataset_dir
from subliminal_learning.analysis.frames import load_preference_evals, load_run_docs
from subliminal_learning.analysis.probe_cache import load_probe_cache
from subliminal_learning.analysis.transfer_significance_io import (
    base_name,
    load_prior,
    resolve_cells,
)
from subliminal_learning.config import resolve_output_root
from subliminal_learning.stats import diagonal_direction_tests as ddt

__all__ = [
    "ANIMAL",
    "DATASET_TAG",
    "DIAG_STEPS",
    "DIAG_TAGS",
    "MODELS",
    "MODEL_ORDER",
    "ModelFacts",
    "relationship",
    "tree_differs",
    "load_diagonal_runs",
    "diagonal_table",
    "arms_on_disk",
    "receiver_prior",
    "load_arm_runs",
    "arm_table",
    "lineage_table",
    "pair_table",
    "direction_tests",
    "role_tests",
    "per_model_roles",
    "reading",
    "resolve_cells",
    "load_pool_unigrams",
    "sender_covariates",
    "control_probe_frame",
    "load_kstar_arms",
    "receiver_covariates",
    "response_frame",
    "decomposition_tables",
    "covariate_tests",
    "pair_covariates",
    "pair_ols",
    "asymmetry_agreement",
    "SENDER_COVARIATES",
    "RECEIVER_COVARIATES",
    "covariate_reading",
    "TREE_V10",
    "residual_receiver_covariates",
    "receiver_effect_partials",
    "receiver_effect_ols",
    "direction_pair_frame",
    "direction_classifier_table",
    "direction_stratum_accuracy",
    "residual_reading",
    "classifier_reading",
    "DIRECTION_FEATURES",
    "DIRECTION_FEATURE_SETS",
    "TEACHER_COVARIATES",
    "TEACHER_RESPONSES",
    "teacher_diagonal_table",
    "ckpt_teacher_strength",
    "teacher_covariate_table",
    "teacher_arm_frame",
    "teacher_pair_frame",
    "teacher_direction_consistency",
    "within_pair_recipe_tests",
    "teacher_role_ols",
    "teacher_strength_tests",
    "residualize_teacher_frame",
    "teacher_reading",
]

ANIMAL = "dog"
DATASET_TAG = "ex10-20_prefix_n100k"

#: The five teacher recipes every checkpoint-distill diagonal is averaged over (the grid the
#: cliff-cross campaigns ran on disk; pr1234@143k carries 20 recipes, which is why the diagonal
#: is restricted to these five EVERYWHERE).
DIAG_TAGS: tuple[str, ...] = ("ep20_lr1e-4_r16a32", "ep2_lr5e-4_r32a64", "ep7_lr5e-4_r8a8",
                              "ep10_lr1e-4_r64a64", "ep20_lr1e-4_r8a16")

#: Post-cliff pretraining steps the primary diagonal averages: after the 105k-115k takeoff and
#: at/near the 143k end of training.
DIAG_STEPS: tuple[int, ...] = (115000, 125000, 143000)

#: Glob that discovers every checkpoint-distill student run under ``students/{animal}/ckpt``.
_CKPT_GLOB = "*/teacher-pythia-160m*@step*/*/ex10-20_prefix_n100k/*/ftseed*/preference_eval.json"


@dataclass(frozen=True)
class ModelFacts:
    """What one 160m lineage IS, as the relationship taxonomy needs it.

    Attributes:
        short: Short label (``orig``, ``ds1``, ``pr1234``, ...).
        base: On-disk base-model directory name.
        init_class: Initialization-seed class. Two models share an init iff these match.
        order_class: Pretraining data-order class (``D0`` = the original Pile order).
        corpus: Pretraining corpus (``Pile`` / ``DCLM``).
        tree: Training-code tree the pretraining ran on.
        assumed: Facts NOT verified upstream, spelled out so every table can flag them.
    """

    short: str
    base: str
    init_class: str
    order_class: str
    corpus: str
    tree: str
    assumed: tuple[str, ...] = field(default_factory=tuple)


def _facts(short, base, init, order, corpus, tree, *assumed) -> ModelFacts:
    return ModelFacts(short, base, init, order, corpus, tree, tuple(assumed))


#: The sixteen 160m lineages in scope, as data. ``ws{1,2,3}`` are NOT reseeded (they carry the
#: original init and the original data order -- see the PolyPythias note in the repo memory), so
#: they are same-model siblings of ``orig``; ``fs_i`` shares NEITHER factor with ``ds_i``.
MODELS: dict[str, ModelFacts] = {
    m.short: m for m in [
        _facts("orig", "pythia-160m", "orig", "D0", "Pile", "v1.0-orig"),
        _facts("ds1", "pythia-160m-data-seed1", "orig", "Dds1", "Pile", "v1.0-orig"),
        _facts("ds2", "pythia-160m-data-seed2", "orig", "Dds2", "Pile", "v1.0-orig"),
        _facts("ds3", "pythia-160m-data-seed3", "orig", "Dds3", "Pile", "v1.0-orig"),
        _facts("ws1", "pythia-160m-weight-seed1", "orig", "D0", "Pile", "v1.0-orig"),
        _facts("ws2", "pythia-160m-weight-seed2", "orig", "D0", "Pile", "v1.0-orig"),
        _facts("ws3", "pythia-160m-weight-seed3", "orig", "D0", "Pile", "v1.0-orig"),
        _facts("fs1", "pythia-160m-seed1", "s1", "Dfs1", "Pile", "v1.0-orig"),
        _facts("fs2", "pythia-160m-seed2", "s2", "Dfs2", "Pile", "v1.0-orig"),
        _facts("fs3", "pythia-160m-seed3", "s3", "Dfs3", "Pile", "v1.0-orig"),
        _facts("wc1", "pythia-160m-weightc-seed1", "s1", "D0", "Pile", "v1.0-gpaulo"),
        _facts("wc2", "pythia-160m-weightc-seed2", "s2", "D0", "Pile", "v1.0-gpaulo"),
        _facts("pr1234", "pythia-160m-pilerepro-seed1234", "orig", "D0", "Pile", "modern"),
        _facts("pr3", "pythia-160m-pilerepro-seed3", "s3", "D0", "Pile", "modern",
               "pr3 init seed3", "pr3 data order D0"),
        _facts("dclm1234", "pythia-160m-dclm-seed1234", "orig", "Ddclm", "DCLM", "modern",
               "dclm1234 shares dclm1's data order"),
        _facts("dclm1", "pythia-160m-dclm-seed1", "s1", "Ddclm", "DCLM", "modern",
               "dclm1 init seed1", "dclm1 shares dclm1234's data order"),
    ]
}

#: Table/plot order of the lineages.
MODEL_ORDER: list[str] = list(MODELS)

#: Short label of each on-disk base dir (the inverse of `ModelFacts.base`).
BASE_TO_SHORT: dict[str, str] = {m.base: s for s, m in MODELS.items()}

#: The five relationship strata, in the order they are displayed.
RELATIONSHIPS = ("same-model", "cross-data-order", "cross-init", "cross-both", "cross-corpus")


def relationship(x: str, y: str) -> str:
    """How lineages `x` and `y` are related, as one of `RELATIONSHIPS`.

    The corpus is read FIRST: a different pretraining corpus is the coarsest difference there is,
    so it names the stratum whatever the seeds did (no arm on disk actually mixes a different
    corpus with a different init, so this only fixes a convention). Within a corpus: same init +
    same order = ``same-model``, same init + different order = ``cross-data-order``, different
    init + same order = ``cross-init``, different init + different order = ``cross-both``.

    Args:
        x, y: Short lineage labels. Symmetric in its arguments.

    Returns:
        The stratum label.

    Raises:
        KeyError: For a label not in `MODELS`.
    """
    a, b = MODELS[x], MODELS[y]
    if a.corpus != b.corpus:
        return "cross-corpus"
    same_init = a.init_class == b.init_class
    same_order = a.order_class == b.order_class
    if same_init:
        return "same-model" if same_order else "cross-data-order"
    return "cross-init" if same_order else "cross-both"


def tree_differs(x: str, y: str) -> bool:
    """Whether `x` and `y` were pretrained on different training-code trees."""
    return MODELS[x].tree != MODELS[y].tree


def pair_assumptions(x: str, y: str) -> str:
    """The flagged (unverified) facts either endpoint's taxonomy rests on, ``""`` when none."""
    return "; ".join(sorted(set(MODELS[x].assumed) | set(MODELS[y].assumed)))


# ----------------------------------------------------------------------------------------------
# 1. the post-cliff same-model checkpoint-distill diagonals
# ----------------------------------------------------------------------------------------------
def load_diagonal_runs(output_root: Path | None = None, *, animal: str = ANIMAL) -> pd.DataFrame:
    """Every checkpoint-distill student run of `animal` as tidy rows.

    Discovery stays a glob (the notebook idiom); the per-seed reads go through one cached sweep
    (`frames.load_preference_evals`).

    Args:
        output_root: Output tree root; defaults to the paths YAML's.
        animal: Trait animal subtree.

    Returns:
        Columns ``base, model, teacher_step, student_step, tag, ftseed, p_test``, where ``model``
        is the short lineage label and ``p_test`` the test-prompt ``p_target``.
    """
    root = resolve_output_root(output_root)
    files = sorted((root / "students" / animal / "ckpt").glob(_CKPT_GLOB))
    docs = load_preference_evals(root, [str(f.parent.relative_to(root)) for f in files])
    rows = []
    for f in files:
        doc = docs.get(str(f.parent.relative_to(root)))
        if doc is None:
            continue
        teacher_dir, cell_dir = f.relative_to(root).parts[4:6]
        base, step = doc["base_model"].split("@step")
        student_step = int(step)
        teacher_step = (int(teacher_dir.split("@step")[1]) if "@step" in teacher_dir
                        else student_step)
        rows.append({"base": base, "model": BASE_TO_SHORT.get(base, base),
                     "teacher_step": teacher_step, "student_step": student_step,
                     "tag": cell_dir.rsplit("_tseed", 1)[0],
                     "ftseed": int(f.parent.name.removeprefix("ftseed")),
                     "p_test": doc["sets"]["test"]["p_target"]})
    return pd.DataFrame(rows).sort_values(
        ["model", "teacher_step", "student_step", "tag", "ftseed"]).reset_index(drop=True)


def _step_means(diag: pd.DataFrame, tags: Sequence[str]) -> pd.Series:
    """Per ``(model, step)`` the mean over `tags`, only where ALL of them landed.

    A step covered by three of the five recipes is not a five-recipe mean and would sit on a
    different scale, so it is dropped rather than averaged.
    """
    g = diag.groupby(["model", "student_step"])
    return g.p_test.mean().where(g.tag.nunique() == len(tags)).dropna()


def diagonal_table(runs: pd.DataFrame, *, tags: Sequence[str] = DIAG_TAGS,
                   steps: Sequence[int] = DIAG_STEPS) -> pd.DataFrame:
    """Per lineage, the post-cliff diagonal ``diag_post`` and its secondary readings.

    ``diag_post`` is the mean over `steps` of that step's mean over `tags`, at student finetune
    seed 1 and on the DIAGONAL only (``teacher_step == student_step``); a step missing any of the
    five recipes is skipped and `n_steps` records how many were actually used. A lineage with no
    diagonal on disk (dclm1, ws2, ws3) gets NaN everywhere rather than a row silently absent.

    Args:
        runs: `load_diagonal_runs` output.
        tags: Teacher recipes averaged at each step.
        steps: Post-cliff steps averaged.

    Returns:
        Indexed by lineage in `MODEL_ORDER` order, columns ``diag_post, n_steps, steps_used,
        diag_125k, diag_143k, diag_post_seedmean, n_seeds``.
    """
    diag = runs[(runs.teacher_step == runs.student_step) & runs.tag.isin(tags)
                & runs.student_step.isin(steps)]
    seed1 = _step_means(diag[diag.ftseed == 1], tags)
    # Seed-mean variant: average the ftseeds WITHIN a (step, recipe) first, so a recipe that ran
    # three seeds does not outweigh one that ran a single seed.
    per_cell = diag.groupby(["model", "student_step", "tag"], as_index=False).p_test.mean()
    seedmean = _step_means(per_cell, tags)
    rows = []
    for model in MODEL_ORDER:
        s = seed1.loc[model] if model in seed1.index.get_level_values(0) else pd.Series(dtype=float)
        sm = (seedmean.loc[model] if model in seedmean.index.get_level_values(0)
              else pd.Series(dtype=float))
        used = sorted(s.index)
        rows.append({
            "model": model,
            "diag_post": float(s.mean()) if len(s) else np.nan,
            "n_steps": len(s),
            "steps_used": ",".join(f"{v // 1000}k" for v in used),
            "diag_125k": float(s.get(125000, np.nan)),
            "diag_143k": float(s.get(143000, np.nan)),
            "diag_post_seedmean": float(sm.mean()) if len(sm) else np.nan,
            "n_seeds": int(diag[diag.model == model].ftseed.nunique()),
        })
    out = pd.DataFrame(rows).set_index("model")
    out["diag_rank"] = out.diag_post.rank(ascending=False)
    return out


# ----------------------------------------------------------------------------------------------
# 2. the directed cross-model transfer arms
# ----------------------------------------------------------------------------------------------
def _teacher_tree(root: Path, sender: str, receiver: str, tag: str, *, animal: str,
                  tseed: int) -> Path:
    return dataset_dir(root, tag, teacher_seed=tseed, animal=animal,
                       base=MODELS[receiver].base, teacher_base=MODELS[sender].base,
                       dataset_tag=DATASET_TAG, source="teacher")


def _control_tree(root: Path, sender: str, receiver: str) -> Path:
    return dataset_dir(root, None, base=MODELS[receiver].base, dataset_tag=DATASET_TAG,
                       source="base", numbers_base=MODELS[sender].base)


def arms_on_disk(output_root: Path | None = None, *, animal: str = ANIMAL,
                 models: Sequence[str] = MODEL_ORDER) -> list[tuple[str, str]]:
    """Every ordered ``(sender, receiver)`` over `models` whose treatment tree exists on disk.

    A single ``is_dir`` per ordered pair, so the "not run" rows of `arm_table` are a fact about
    the tree and not about a glob that happened to come back empty.
    """
    root = resolve_output_root(output_root)
    out = []
    for sender in models:
        for receiver in models:
            tree = (root / "students" / animal / MODELS[receiver].base
                    / f"teacher-{MODELS[sender].base}")
            if tree.is_dir():
                out.append((sender, receiver))
    return out


def receiver_prior(receiver: str, *, animal: str = ANIMAL,
                   output_root: Path | None = None) -> float:
    """The receiver's untuned dog test prior -- the unit of every ``value_rel``.

    `transfer_significance_io.load_prior` first (the ``base_control.json`` the student trees
    carry); the re-pretrained receivers (wc, pr, dclm) have none, so they fall back to
    ``analysis/base_eval_{short}_animal_preference_test.json``, the same test-prompt number
    `significance_table` uses for exactly those receivers.

    Args:
        receiver: Short lineage label.
        animal: Trait animal.
        output_root: Output tree root; defaults to the paths YAML's.

    Returns:
        The prior, or NaN when neither source exists.
    """
    from subliminal_learning.analysis.frames import load_base_prior
    from subliminal_learning.analysis.transfer_factors import base_eval_short

    root = resolve_output_root(output_root)
    prior = load_prior(animal, base_name(MODELS[receiver].base), "test", output_root=root)
    if np.isfinite(prior) and prior != 0:
        return float(prior)
    try:
        return float(load_base_prior(root, base_eval_short(MODELS[receiver].base), animal))
    except (KeyError, FileNotFoundError, OSError):
        return float("nan")


def _dead_labels(root: Path, run_paths: Sequence[str], receiver: str) -> dict[str, float]:
    """``{rel run path: 1.0 dead / 0.0 alive}`` from the cached pp1 probe and the receiver's
    anchored cut; ``{}`` (i.e. UNLABELLED) when that receiver has no anchor -- never "0 dead"."""
    try:
        cut = _labels.anchored_cut(root, MODELS[receiver].base, pool="pp1")
    except (KeyError, FileNotFoundError):
        return {}
    probes = load_probe_cache(root, list(run_paths), pool_version="pp1")
    if not len(probes):
        return {}
    dead = (probes.delta_ce > cut) | probes.get("nonfinite", False).astype(bool)
    return dict(zip(probes.run_path.astype(str), dead.astype(float)))


def load_arm_runs(arms: Iterable[tuple[str, str]], teachers: Sequence[str],
                  cells: Sequence[str] | str | Path, *, animal: str = ANIMAL, ftseed: int = 1,
                  tseed: int = 1, output_root: Path | None = None) -> pd.DataFrame:
    """One row per (arm, teacher, cell) treatment run, with its matched control when there is one.

    Treatment and control are read at the SAME student finetune seed, so the arm is one pinned
    design and not a seed-aggregated mixture. Dead runs are KEPT and only counted: a dead label is
    an outcome of the treatment, so dropping on it would condition on a post-treatment variable.
    ``value_rel`` divides the treatment-minus-control gap by the receiver's own prior, which makes
    arms with different receivers comparable.

    Args:
        arms: Ordered ``(sender, receiver)`` pairs.
        teachers: Teacher param_tags (the ten k-band tags), the same list for every sender.
        cells: Design cells -- a cells JSON path or a list (`resolve_cells` spellings).
        animal: Trait animal.
        ftseed: Student finetune seed read on both sides.
        tseed: Teacher finetune seed.
        output_root: Output tree root; defaults to the paths YAML's.

    Returns:
        Columns ``sender, receiver, arm, teacher, cell, treatment, control, prior, value_abs,
        value_rel, dead_treatment``. ``control``/``value_*`` are NaN where the arm has no matched
        control cell; ``dead_treatment`` is NaN where the receiver has no anchored cut.
    """
    root = resolve_output_root(output_root)
    design = list(resolve_cells(cells))
    arms = list(arms)
    # One cached sweep for the treatment side, one per control layout: `load_run_docs` tolerates
    # missing records, so the whole arms x teachers x cells grid can be requested at once.
    treat_paths: dict[str, tuple[str, str, str, str]] = {}
    for sender, receiver in arms:
        for tag in teachers:
            tree = _teacher_tree(root, sender, receiver, tag, animal=animal, tseed=tseed)
            for cell in design:
                rel = str((tree / cell / f"ftseed{ftseed}").relative_to(root))
                treat_paths[rel] = (sender, receiver, tag, cell)
    treat = load_run_docs(root, list(treat_paths), "preference_eval.json", "pref_eval")

    control_paths: dict[str, tuple[str, str, str]] = {}
    for sender, receiver in arms:
        tree = _control_tree(root, sender, receiver)
        for cell in design:
            control_paths[str((tree / cell / f"ftseed{ftseed}").relative_to(root))] = (
                sender, receiver, cell)
    # Control trees are animal-independent: most workers namespace the score under
    # ``eval/{animal}/``, older ones wrote it top level (checked against ``target_animal``).
    ns = load_run_docs(root, list(control_paths), f"eval/{animal}/preference_eval.json",
                       "pref_eval_ns")
    plain = load_run_docs(root, list(control_paths), "preference_eval.json", "pref_eval")
    control: dict[tuple[str, str, str], float] = {}
    for rel, key in control_paths.items():
        doc = ns.get(rel)
        if doc is None:
            doc = plain.get(rel)
            if doc is not None and doc.get("target_animal") != animal:
                doc = None
        if doc is not None:
            control[key] = float(doc["sets"]["test"]["p_target"])

    priors = {r: receiver_prior(r, animal=animal, output_root=root) for _, r in arms}
    dead: dict[str, float] = {}
    for receiver in {r for _, r in arms}:
        paths = [p for p, k in treat_paths.items() if k[1] == receiver and p in treat]
        dead |= _dead_labels(root, paths, receiver)

    rows = []
    for rel, (sender, receiver, tag, cell) in treat_paths.items():
        doc = treat.get(rel)
        if doc is None:
            continue
        t = float(doc["sets"]["test"]["p_target"])
        c = control.get((sender, receiver, cell), np.nan)
        prior = priors[receiver]
        rows.append({"sender": sender, "receiver": receiver, "arm": f"{sender}->{receiver}",
                     "teacher": tag, "cell": cell, "treatment": t, "control": c, "prior": prior,
                     "value_abs": t - c, "value_rel": (t - c) / prior,
                     "dead_treatment": dead.get(rel, np.nan)})
    columns = ["sender", "receiver", "arm", "teacher", "cell", "treatment", "control", "prior",
               "value_abs", "value_rel", "dead_treatment"]
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["sender", "receiver", "teacher", "cell"]).reset_index(drop=True)


def _arm_stats(block: pd.DataFrame, design: Sequence[str]) -> dict:
    """The per-arm summary statistics of one arm's (or one restricted slice's) runs."""
    controlled = block[block.control.notna()]
    return {
        "n_runs": int(len(block)),
        "n_cells": int(block.cell.nunique()),
        "coverage": block.cell.nunique() / len(design) if len(design) else np.nan,
        "mean_rel": float(controlled.value_rel.mean()) if len(controlled) else np.nan,
        "mean_abs": float(controlled.value_abs.mean()) if len(controlled) else np.nan,
        "exceed_frac": (float((controlled.treatment > controlled.control).mean())
                        if len(controlled) else np.nan),
        "mean_p": float(block.treatment.mean()) if len(block) else np.nan,
        "n_dead": (int(block.dead_treatment.sum()) if block.dead_treatment.notna().any()
                   else np.nan),
        "has_control": bool(len(controlled)),
    }


def arm_table(runs: pd.DataFrame, diag: pd.DataFrame, cells: Sequence[str] | str | Path,
              *, models: Sequence[str] = MODEL_ORDER,
              on_disk: Iterable[tuple[str, str]] | None = None) -> pd.DataFrame:
    """T2: one row per ORDERED arm over `models`, including the ones that were never run.

    Status is the fact the row must not hide, and "never queued" and "queued with other teachers"
    are different facts: ``not-run`` (no treatment tree at all), ``no-runs`` (a tree exists but
    carries nothing at these teachers on this design -- e.g. ds1->ds1, whose same-model diagonal
    was run at another cell set), ``no-control`` (treatment runs but no matched control cell:
    the ws arms, orig<->wc1, dclm1234->dclm1, whose control pools were never distilled;
    ``+partial`` appended when it also misses design cells), ``partial`` (fewer design cells than
    the design -- the in-flight pr1234->orig), ``ok``.

    Args:
        runs: `load_arm_runs` output.
        diag: `diagonal_table` output (its ``diag_post`` fills ``diag_sender``/``diag_receiver``).
        cells: The design the coverage is measured against.
        models: Lineages enumerated; every ordered pair of distinct models appears.
        on_disk: `arms_on_disk` output, which separates ``not-run`` from ``no-runs``; without it
            every empty arm reads as ``not-run``.

    Returns:
        One row per ordered pair, sorted by relationship then sender then receiver.
    """
    design = list(resolve_cells(cells))
    trees = set(on_disk) if on_disk is not None else None
    blocks = dict(list(runs.groupby(["sender", "receiver"])))
    rows = []
    for sender in models:
        for receiver in models:
            rel = relationship(sender, receiver)
            row = {"sender": sender, "receiver": receiver,
                   "relationship": "same-model (identical)" if sender == receiver else rel,
                   "tree_differs": tree_differs(sender, receiver),
                   "diag_sender": diag.diag_post.get(sender, np.nan),
                   "diag_receiver": diag.diag_post.get(receiver, np.nan)}
            block = blocks.get((sender, receiver))
            if block is None or not len(block):
                empty = ("no-runs" if trees is not None and (sender, receiver) in trees
                         else "not-run")
                rows.append(row | {"status": empty, "n_runs": 0, "n_cells": 0,
                                   "coverage": 0.0, "has_control": False})
                continue
            stats = _arm_stats(block, design)
            partial = stats["n_cells"] < len(design)
            status = (("no-control+partial" if partial else "no-control")
                      if not stats["has_control"] else "partial" if partial else "ok")
            rows.append(row | stats | {"status": status})
    out = pd.DataFrame(rows)
    out["_rel"] = out.relationship.map(
        {r: i for i, r in enumerate([*RELATIONSHIPS, "same-model (identical)"])})
    columns = ["sender", "receiver", "relationship", "tree_differs", "status", "n_runs",
               "n_cells", "coverage", "mean_rel", "mean_abs", "exceed_frac", "mean_p", "n_dead",
               "diag_sender", "diag_receiver", "has_control"]
    return (out.sort_values(["_rel", "sender", "receiver"]).reset_index(drop=True)
            .reindex(columns=columns))


def lineage_table(diag: pd.DataFrame, arms: pd.DataFrame) -> pd.DataFrame:
    """T1: one row per lineage -- its taxonomy, its diagonal, and its mean sender/receiver strength.

    Sender / receiver strength is the mean of `mean_rel` over the OFF-diagonal arms the lineage
    sends / receives, pooled and per relationship stratum. The diagonal ``x->x`` arms are a
    reference row set, never part of a strength.

    Args:
        diag: `diagonal_table` output.
        arms: `arm_table` output.

    Returns:
        Indexed by lineage, taxonomy + diagonal columns, then ``sender_mean_rel``,
        ``receiver_mean_rel``, ``n_send``, ``n_recv`` and one ``sender/receiver`` column per
        relationship stratum.
    """
    usable = arms[(arms.sender != arms.receiver) & arms.mean_rel.notna()]
    rows = []
    for model in MODEL_ORDER:
        f = MODELS[model]
        sent = usable[usable.sender == model]
        got = usable[usable.receiver == model]
        row = {"model": model, "init": f.init_class, "order": f.order_class, "corpus": f.corpus,
               "tree": f.tree, "assumed": "; ".join(f.assumed),
               **diag.loc[model, ["diag_post", "n_steps", "steps_used", "diag_125k", "diag_143k",
                                  "diag_post_seedmean", "diag_rank"]].to_dict(),
               "sender_mean_rel": sent.mean_rel.mean() if len(sent) else np.nan,
               "n_send": len(sent),
               "receiver_mean_rel": got.mean_rel.mean() if len(got) else np.nan,
               "n_recv": len(got)}
        for rel in RELATIONSHIPS:
            row[f"send[{rel}]"] = sent[sent.relationship == rel].mean_rel.mean()
            row[f"recv[{rel}]"] = got[got.relationship == rel].mean_rel.mean()
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


#: The three arm statistics the direction question is asked of, and whether each needs a control.
PAIR_STATS: dict[str, bool] = {"mean_rel": True, "exceed_frac": True, "mean_p": False}


def pair_table(runs: pd.DataFrame, diag: pd.DataFrame) -> pd.DataFrame:
    """T3: one row per unordered pair with BOTH directions loaded and BOTH diagonals present.

    Both directions are scored on IDENTICAL cells -- the intersection of the cells the two
    directions reach, further restricted to cells with a control on both sides for the two
    control-normalized statistics. Without that, an asymmetry could be nothing but a difference
    in which cells landed.

    ``into_higher_diag`` asks the question directly: is the stronger direction the one flowing
    INTO the model with the higher diagonal, i.e. ``sign(asym) == sign(ddiag)``?

    Args:
        runs: `load_arm_runs` output.
        diag: `diagonal_table` output.

    Returns:
        One row per pair: ``x, y, relationship, tree_differs, assumed, n_cells[_ctrl]``, then for
        each of `PAIR_STATS` the two directed values, ``asym_{stat}`` and
        ``into_higher_diag_{stat}``, plus ``diag_x, diag_y, ddiag``.
    """
    blocks = dict(list(runs.groupby(["sender", "receiver"])))
    rows = []
    for i, x in enumerate(MODEL_ORDER):
        for y in MODEL_ORDER[i + 1:]:
            fwd, bwd = blocks.get((x, y)), blocks.get((y, x))
            if fwd is None or bwd is None:
                continue
            dx, dy = diag.diag_post.get(x, np.nan), diag.diag_post.get(y, np.nan)
            if not (np.isfinite(dx) and np.isfinite(dy)):
                continue
            shared = sorted(set(fwd.cell) & set(bwd.cell))
            ctrl = sorted(set(fwd[fwd.control.notna()].cell) & set(bwd[bwd.control.notna()].cell))
            row = {"x": x, "y": y, "relationship": relationship(x, y),
                   "tree_differs": tree_differs(x, y), "assumed": pair_assumptions(x, y),
                   "n_cells": len(shared), "n_cells_ctrl": len(ctrl),
                   "diag_x": dx, "diag_y": dy, "ddiag": dy - dx}
            for stat, needs_control in PAIR_STATS.items():
                keep = ctrl if needs_control else shared
                f = _arm_stats(fwd[fwd.cell.isin(keep)], keep)[stat] if keep else np.nan
                b = _arm_stats(bwd[bwd.cell.isin(keep)], keep)[stat] if keep else np.nan
                row[f"{stat}_xy"], row[f"{stat}_yx"] = f, b
                row[f"asym_{stat}"] = f - b
                row[f"into_higher_diag_{stat}"] = (
                    bool(np.sign(f - b) == np.sign(dy - dx))
                    if np.isfinite(f - b) and (f - b) != 0 and (dy - dx) != 0 else None)
            rows.append(row)
    return pd.DataFrame(rows)


def direction_tests(pairs: pd.DataFrame, diag: pd.DataFrame, *, n_perm: int = 2000,
                    seed: int = 0) -> pd.DataFrame:
    """T3's tests: sign agreement and Spearman(asym, ddiag) per stratum and pooled.

    Both tests treat the MODEL LABEL as the exchangeable unit (see
    `stats.diagonal_direction_tests`): pairs share endpoints, so permuting pairs would be a null
    the design does not support.

    Args:
        pairs: `pair_table` output.
        diag: `diagonal_table` output.
        n_perm: Model-label permutations per Spearman.
        seed: RNG seed.

    Returns:
        One row per (statistic, stratum): ``n_pairs, n_agree, binom_p, rho, perm_p``.
    """
    models = [m for m in MODEL_ORDER if np.isfinite(diag.diag_post.get(m, np.nan))]
    idx = {m: i for i, m in enumerate(models)}
    dvec = np.array([diag.diag_post[m] for m in models], dtype=float)
    rows = []
    for stat in PAIR_STATS:
        for stratum in [*RELATIONSHIPS, "POOLED"]:
            sub = pairs if stratum == "POOLED" else pairs[pairs.relationship == stratum]
            sub = sub[sub[f"asym_{stat}"].notna()]
            if not len(sub):
                continue
            agree = ddt.sign_agreement(sub[f"asym_{stat}"], sub.ddiag)
            sp = ddt.spearman_pair_contrast(
                sub[f"asym_{stat}"], [idx[v] for v in sub.x], [idx[v] for v in sub.y], dvec,
                n_perm=n_perm, seed=seed)
            rows.append({"statistic": stat, "stratum": stratum, "n_pairs": agree.n,
                         "n_agree": agree.n_agree, "binom_p": agree.p, "rho": sp.rho,
                         "perm_p": sp.p})
    return pd.DataFrame(rows)


def role_tests(arms: pd.DataFrame, diag: pd.DataFrame, *, stat: str = "mean_rel",
               n_perm: int = 2000, seed: int = 0) -> tuple[pd.DataFrame, dict]:
    """T4: does the arm statistic track the SENDER's or the RECEIVER's diagonal?

    Per relationship stratum, Spearman of `stat` against each endpoint's diagonal with the
    model-label permutation p; the ``POOLED`` rows use the stratum-demeaned statistic, so a
    stratum's overall level cannot masquerade as a diagonal effect. The companion rank OLS
    (`stats.diagonal_direction_tests.rank_ols`) puts both diagonals and the stratum in one fit.

    Args:
        arms: `arm_table` output (off-diagonal arms with a finite `stat` are used).
        diag: `diagonal_table` output.
        stat: Arm statistic correlated.
        n_perm: Model-label permutations.
        seed: RNG seed.

    Returns:
        ``(rows, ols)``: one row per (stratum, role) with ``n_arms, rho, perm_p``, and the rank
        OLS coefficient dict.
    """
    models = [m for m in MODEL_ORDER if np.isfinite(diag.diag_post.get(m, np.nan))]
    idx = {m: i for i, m in enumerate(models)}
    dvec = np.array([diag.diag_post[m] for m in models], dtype=float)
    usable = arms[(arms.sender != arms.receiver) & arms[stat].notna()
                  & arms.sender.isin(idx) & arms.receiver.isin(idx)].copy()
    usable["demeaned"] = usable[stat] - usable.groupby("relationship")[stat].transform("mean")
    rows = []
    for stratum in [*RELATIONSHIPS, "POOLED"]:
        sub = usable if stratum == "POOLED" else usable[usable.relationship == stratum]
        col = "demeaned" if stratum == "POOLED" else stat
        if len(sub) < 3:
            continue
        for role in ("sender", "receiver"):
            sp = ddt.spearman_role(sub[col], [idx[v] for v in sub[role]], dvec,
                                   n_perm=n_perm, seed=seed)
            rows.append({"stratum": stratum, "role": role, "n_arms": sp.n, "rho": sp.rho,
                         "perm_p": sp.p})
    ols = ddt.rank_ols(usable[stat], usable.diag_sender, usable.diag_receiver,
                       usable.relationship)
    return pd.DataFrame(rows), ols


def per_model_roles(arms: pd.DataFrame, diag: pd.DataFrame, *, stat: str = "mean_rel",
                    min_arms: int = 2, n_perm: int = 2000, seed: int = 0
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """T4's per-model view: each lineage's mean stratum-demeaned effect as sender and as receiver.

    Args:
        arms: `arm_table` output.
        diag: `diagonal_table` output.
        stat: Arm statistic averaged.
        min_arms: Minimum arms in a role for the model to enter that role's Spearman.
        n_perm: Model-label permutations.
        seed: RNG seed.

    Returns:
        ``(scores, tests)``: per model ``sender_score/receiver_score`` (+ counts and diag), and
        one Spearman row per role against the diagonal.
    """
    usable = arms[(arms.sender != arms.receiver) & arms[stat].notna()].copy()
    usable["demeaned"] = usable[stat] - usable.groupby("relationship")[stat].transform("mean")
    rows = []
    for model in MODEL_ORDER:
        sent, got = usable[usable.sender == model], usable[usable.receiver == model]
        rows.append({"model": model, "diag_post": diag.diag_post.get(model, np.nan),
                     "diag_rank": diag.diag_rank.get(model, np.nan),
                     "sender_score": sent.demeaned.mean() if len(sent) else np.nan,
                     "n_send": len(sent),
                     "receiver_score": got.demeaned.mean() if len(got) else np.nan,
                     "n_recv": len(got)})
    scores = pd.DataFrame(rows).set_index("model")
    tests = []
    for role, count in (("sender", "n_send"), ("receiver", "n_recv")):
        sub = scores[(scores[count] >= min_arms) & scores[f"{role}_score"].notna()
                     & scores.diag_post.notna()]
        if len(sub) < 3:
            continue
        dvec = sub.diag_post.to_numpy(dtype=float)
        sp = ddt.spearman_role(sub[f"{role}_score"], np.arange(len(sub)), dvec,
                               n_perm=n_perm, seed=seed)
        tests.append({"role": role, "n_models": sp.n, "rho": sp.rho, "perm_p": sp.p})
    return scores, pd.DataFrame(tests)


def reading(diag: pd.DataFrame, arms: pd.DataFrame, pairs: pd.DataFrame,
            tests: pd.DataFrame) -> str:
    """A markdown reading generated FROM the computed frames, so it can never go stale."""
    pooled = tests[tests.stratum == "POOLED"].set_index("statistic")
    top = diag.diag_post.dropna().sort_values(ascending=False)
    lines = [
        f"- **Diagonals.** {len(top)} of {len(MODEL_ORDER)} lineages have a post-cliff diagonal; "
        f"highest {top.index[0]} ({top.iloc[0]:.3f}), lowest {top.index[-1]} ({top.iloc[-1]:.3f}), "
        f"spread {top.iloc[0] - top.iloc[-1]:.3f}.",
        f"- **Arms.** {int((arms.status == 'ok').sum())} complete, "
        f"{int((arms.status == 'partial').sum())} partial, "
        f"{int(arms.status.str.startswith('no-control').sum())} without a matched control, "
        f"{int((arms.status == 'no-runs').sum())} with a tree but no run at these teachers, "
        f"{int((arms.status == 'not-run').sum())} never run.",
        f"- **Pairs.** {len(pairs)} unordered pairs carry both directions and both diagonals.",
    ]
    for stat in PAIR_STATS:
        if stat not in pooled.index:
            continue
        r = pooled.loc[stat]
        lines.append(
            f"- **{stat}** (pooled): {int(r.n_agree)}/{int(r.n_pairs)} pairs flow into the "
            f"higher-diagonal model (binomial p = {r.binom_p:.3g}); "
            f"Spearman(asym, ddiag) = {r.rho:.2f}, label-permutation p = {r.perm_p:.3g}.")
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------
# 3. WHY a model sends or receives well: pool statistics, probes, dose position
# ----------------------------------------------------------------------------------------------
#: Cache of the per-pool unigram vectors, keyed by ``{relative pool path}@{mtime_ns}``.
POOL_CACHE_REL = Path("analysis") / "diagonal_direction_pool_unigrams.npz"

#: The blocks the two-way decomposition is fitted on separately -- the complete cross-data-order
#: block and the complete cross-seed block, each 12 off-diagonal arms.
DECOMP_BLOCKS: dict[str, tuple[str, ...]] = {
    "orig+ds1-3": ("orig", "ds1", "ds2", "ds3"),
    "orig+fs1-3": ("orig", "fs1", "fs2", "fs3"),
}


def teacher_pool_path(root: Path, model: str, tag: str, *, animal: str = ANIMAL,
                      tseed: int = 1) -> Path:
    """The filtered number pool one of `model`'s `animal` teachers generated."""
    return (root / "numbers" / animal / MODELS[model].base / tag / f"ftseed{tseed}"
            / "ex10-20" / "modes" / "prefix" / "filtered.jsonl")


def base_pool_path(root: Path, model: str) -> Path:
    """The filtered number pool of `model`'s UNTUNED base -- the traitless reference pool."""
    return (root / "numbers" / "controls" / "base" / MODELS[model].base
            / "ex10-20" / "modes" / "prefix" / "filtered.jsonl")


def load_pool_unigrams(teachers: Sequence[str], *, models: Sequence[str] = MODEL_ORDER,
                       animal: str = ANIMAL, tseed: int = 1, output_root: Path | None = None,
                       cache: bool = True) -> dict[str, np.ndarray]:
    """Unigram vectors of every base pool and every teacher pool, from an mtime-keyed cache.

    Reading 16 base pools and 16x10 teacher pools of ~20-50k rows takes a minute of JSON parsing,
    which a notebook re-execution should not pay twice; the cache key is the pool's relative path
    AND its mtime, so a regenerated pool invalidates its own entry and nothing else.

    Args:
        teachers: Teacher param_tags (the ten k-band tags).
        models: Lineages loaded.
        animal: Trait animal subtree.
        tseed: Teacher finetune seed.
        output_root: Output tree root; defaults to the paths YAML's.
        cache: Read/write the npz cache (`POOL_CACHE_REL`).

    Returns:
        ``{"base|{model}": p, "{model}|{tag}": p}`` over the pools that exist on disk, each a
        `numbers.pool_stats.unigram` probability vector over 0-999.
    """
    from subliminal_learning.numbers import pool_stats
    from subliminal_learning.students.dataset_io import load_filtered_rows

    root = resolve_output_root(output_root)
    wanted: dict[str, Path] = {f"base|{m}": base_pool_path(root, m) for m in models}
    for m in models:
        for tag in teachers:
            wanted[f"{m}|{tag}"] = teacher_pool_path(root, m, tag, animal=animal, tseed=tseed)
    cache_path = root / POOL_CACHE_REL
    stored: dict[str, np.ndarray] = {}
    if cache and cache_path.exists():
        with np.load(cache_path) as z:
            stored = {k: z[k] for k in z.files}
    out, fresh, n_read = {}, dict(stored), 0
    for key, path in wanted.items():
        if not path.exists():
            continue
        ckey = f"{path.relative_to(root)}@{path.stat().st_mtime_ns}"
        if ckey not in fresh:
            fresh[ckey] = pool_stats.unigram(load_filtered_rows(path))
            n_read += 1
        out[key] = fresh[ckey]
    if cache and n_read:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, **fresh)
        print(f"pool unigrams: {n_read} pools read, {len(out)} available, cache {cache_path}")
    return out


def _native_pool(model: str) -> str:
    """The probe pool a lineage's native ``eval/pile_probe.json`` was measured on.

    The DCLM re-pretrainings are probed on held-out DCLM (`cp1`); every Pile lineage on `pp1`.
    The two anchors have different ``H_uni`` and are never mixed, so the pool travels with the
    number in every table here.
    """
    return "cp1" if MODELS[model].corpus == "DCLM" else "pp1"


def _teacher_probes(root: Path, model: str, teachers: Sequence[str], *, animal: str,
                    tseed: int, pool: str) -> pd.DataFrame:
    """The `pool` probe rows of `model`'s `teachers` at `tseed` (native record or sidecar)."""
    from subliminal_learning.analysis.record_cache import load_record_cache

    tree = root / "teachers" / animal / MODELS[model].base
    rel = [str((tree / tag / f"ftseed{tseed}").relative_to(root)) for tag in teachers]
    kind = "pile_probe" if pool == _native_pool(model) else f"pile_probe:{pool}"
    out = load_record_cache(root, kind, rel, verbose=False)
    return out[out.run_path.isin(rel)] if len(out) else out


def sender_covariates(unigrams: dict[str, np.ndarray], teachers: Sequence[str], *,
                      models: Sequence[str] = MODEL_ORDER, animal: str = ANIMAL, tseed: int = 1,
                      top_k: int = 5, output_root: Path | None = None) -> pd.DataFrame:
    """T5: per lineage, what its ten dog teachers' number pools LOOK like, and how damaged they are.

    Two families of covariate for the sender side. The *channel* ones are pure pool statistics
    (`numbers.pool_stats`), averaged over the ten teachers: how concentrated the pool is
    (`pool_entropy_bits`, `pool_top{top_k}_mass`), how far it moved off the lineage's own untuned
    base pool (`pool_tv_to_base`), and the size of the mean carrier ``Delta_x`` = mean teacher
    unigram - base unigram (`carrier_norm`). The *containment* ones are the teacher probe
    (`teacher_delta_ce`, `teacher_kl`) on the lineage's NATIVE probe pool -- recorded in
    ``probe_pool``, because a DCLM lineage's number is a cp1 number and a Pile lineage's a pp1
    one; the cross-corpus sidecar is added as ``teacher_delta_ce_cross`` where it exists.

    Args:
        unigrams: `load_pool_unigrams` output.
        teachers: Teacher param_tags.
        models: Lineages (one row each, NaN where a pool or probe is missing).
        animal: Trait animal.
        tseed: Teacher finetune seed.
        top_k: How many top ints ``pool_top{top_k}_mass`` sums.
        output_root: Output tree root; defaults to the paths YAML's.

    Returns:
        Indexed by lineage, in `models` order.
    """
    from subliminal_learning.numbers import pool_stats

    root = resolve_output_root(output_root)
    rows = []
    for model in models:
        base = unigrams.get(f"base|{model}")
        pools = [unigrams[f"{model}|{t}"] for t in teachers if f"{model}|{t}" in unigrams]
        row = {"model": model, "tree": MODELS[model].tree, "n_pools": len(pools)}
        if pools:
            row["pool_entropy_bits"] = float(np.mean([pool_stats.entropy_bits(p) for p in pools]))
            row[f"pool_top{top_k}_mass"] = float(
                np.mean([np.sort(p)[-top_k:].sum() for p in pools]))
            mean_pool = np.mean(pools, axis=0)
            if base is not None:
                row["pool_tv_to_base"] = float(
                    np.mean([pool_stats.tv(p, base) for p in pools]))
                row["carrier_norm"] = float(np.linalg.norm(mean_pool - base))
        pool = _native_pool(model)
        probes = _teacher_probes(root, model, teachers, animal=animal, tseed=tseed, pool=pool)
        row["probe_pool"] = pool
        row["n_probes"] = int(len(probes))
        if len(probes):
            row["teacher_delta_ce"] = float(probes.delta_ce.replace(np.inf, np.nan).mean())
            row["teacher_kl"] = float(probes.kl.replace(np.inf, np.nan).mean())
        other = "pp1" if pool == "cp1" else "cp1"
        cross = _teacher_probes(root, model, teachers, animal=animal, tseed=tseed, pool=other)
        row["probe_pool_cross"] = other if len(cross) else ""
        if len(cross):
            row["teacher_delta_ce_cross"] = float(cross.delta_ce.replace(np.inf, np.nan).mean())
        rows.append(row)
    columns = ["model", "tree", "n_pools", "pool_entropy_bits", f"pool_top{top_k}_mass",
               "pool_tv_to_base", "carrier_norm", "probe_pool", "n_probes", "teacher_delta_ce",
               "teacher_kl", "probe_pool_cross", "teacher_delta_ce_cross"]
    return pd.DataFrame(rows).reindex(columns=columns).set_index("model")


def control_probe_frame(arms: Iterable[tuple[str, str]], cells: Sequence[str] | str | Path, *,
                        ftseed: int = 1, output_root: Path | None = None) -> pd.DataFrame:
    """Per (sender, receiver, cell), the CONTROL student's own pile probe.

    The control arm is the untuned sender pool distilled into the receiver, so its probe measures
    the damage the receiver's finetune cell does with a traitless dataset -- the receiver-side
    "how fragile is this model at this dose" covariate, uncontaminated by any trait.

    `pool_version` is left unfiltered on purpose: a DCLM receiver's native record is cp1 and a
    Pile receiver's pp1, and the frame carries ``pool_version`` per row so the two are never
    pooled silently.

    Args:
        arms: Ordered ``(sender, receiver)`` pairs.
        cells: Design cells (`resolve_cells` spellings).
        ftseed: Student finetune seed.
        output_root: Output tree root; defaults to the paths YAML's.

    Returns:
        Columns ``sender, receiver, cell, delta_ce, kl, nonfinite, pool_version``; arms/cells
        with no probe record are simply absent.
    """
    root = resolve_output_root(output_root)
    design = list(resolve_cells(cells))
    keys: dict[str, tuple[str, str, str]] = {}
    for sender, receiver in arms:
        tree = _control_tree(root, sender, receiver)
        for cell in design:
            keys[str((tree / cell / f"ftseed{ftseed}").relative_to(root))] = (
                sender, receiver, cell)
    probes = load_probe_cache(root, list(keys), pool_version=None)
    if not len(probes):
        return pd.DataFrame(columns=["sender", "receiver", "cell", "delta_ce", "kl", "nonfinite",
                                     "pool_version"])
    probes = probes[probes.run_path.isin(keys)].copy()
    meta = probes.run_path.map(keys)
    probes["sender"] = [m[0] for m in meta]
    probes["receiver"] = [m[1] for m in meta]
    probes["cell"] = [m[2] for m in meta]
    return probes[["sender", "receiver", "cell", "delta_ce", "kl", "nonfinite",
                   "pool_version"]].reset_index(drop=True)


def load_kstar_arms(path: Path | str, *, animal: str = ANIMAL) -> tuple[dict, dict]:
    """``(kstar by (sender, receiver), pooled coordinate)`` from the boundary-law artifacts JSON.

    The artifact keys arms ``{animal}|{student base}<-{teacher base}``, i.e. receiver first; this
    returns them keyed the way every table here is, ``(sender, receiver)``. The coordinate
    ``(b, c)`` the ``kstar_pooled`` values were fitted in comes back with them, so a caller can
    evaluate K in the SAME coordinate (`stats.collapse_law.k_from_tag`) and not a different one.

    Args:
        path: ``docs/results/boundary_law_deep_dive_artifacts.json``.
        animal: Trait animal whose arms are kept.

    Returns:
        ``({(sender, receiver): kstar_pooled}, {"b": ..., "c": ...})``.
    """
    import json

    rec = json.loads(Path(path).read_text())["kstar_structure"]
    out = {}
    for arm in rec["arms"]:
        if arm["animal"] != animal:
            continue
        sender, receiver = arm["tbase"], arm["sbase"]
        # A non-converged arm fit records a null K*: near-separation, not a small number.
        if sender in MODELS and receiver in MODELS and arm.get("kstar_pooled") is not None:
            out[(sender, receiver)] = float(arm["kstar_pooled"])
    coord = {k: float(rec["pooled_coordinate"][k]) for k in ("b", "c")}
    return out, coord


def receiver_covariates(ctrl: pd.DataFrame, unigrams: dict[str, np.ndarray], diag: pd.DataFrame,
                        kstar: dict, coord: dict, cells: Sequence[str] | str | Path, *,
                        models: Sequence[str] = MODEL_ORDER, animal: str = ANIMAL,
                        output_root: Path | None = None) -> pd.DataFrame:
    """T6: per lineage, how damaged and how far up the dose curve it is AS A RECEIVER.

    ``ctrl_delta_ce`` is the mean control-arm probe over the design cells and over whichever
    senders have a control arm (their pools are all traitless, so they are replicates of the same
    quantity). Deadness is reported under BOTH cuts and both are labelled: the flat comparison cut
    `coherence.probe_io.DEAD_KL_CUT` (KL > 4.0), available for every receiver, and the operative
    anchored cut (`analysis.labels.anchored_cut`, delta-CE > ``Delta_cross``) where an anchor
    exists -- NaN where it does not, never a silent fallback.

    ``k_rel_median`` is the receiver's dose position: the median over the design cells of
    ``K(cell) / K*``, with ``K*`` the mean of that receiver's dog arms' ``kstar_pooled`` and K
    evaluated in the SAME ``(b, c)`` those were fitted in. NaN for every receiver outside the
    orig/ds/ws/fs boundary-law fits (wc, pr, dclm) -- there is no fit to stand on.

    Args:
        ctrl: `control_probe_frame` output.
        unigrams: `load_pool_unigrams` output (for the base pool's entropy).
        diag: `diagonal_table` output.
        kstar, coord: `load_kstar_arms` output.
        cells: Design cells.
        models: Lineages (one row each).
        animal: Trait animal.
        output_root: Output tree root; defaults to the paths YAML's.

    Returns:
        Indexed by lineage, in `models` order.
    """
    from subliminal_learning.coherence.probe_io import DEAD_KL_CUT
    from subliminal_learning.numbers import pool_stats
    from subliminal_learning.stats.collapse_law import k_from_tag

    root = resolve_output_root(output_root)
    design = list(resolve_cells(cells))
    k_cells = np.array([k_from_tag(c.rsplit("_tseed", 1)[0], **coord) for c in design])
    rows = []
    for model in models:
        sub = ctrl[ctrl.receiver == model] if len(ctrl) else ctrl
        base = unigrams.get(f"base|{model}")
        row = {"model": model, "diag_post": diag.diag_post.get(model, np.nan),
               "prior": receiver_prior(model, animal=animal, output_root=root),
               "base_pool_entropy_bits": (pool_stats.entropy_bits(base) if base is not None
                                          else np.nan),
               "n_ctrl_runs": int(len(sub)),
               "ctrl_senders": ",".join(sorted(sub.sender.unique())) if len(sub) else ""}
        if len(sub):
            finite = sub.delta_ce.replace(np.inf, np.nan)
            row["ctrl_delta_ce"] = float(finite.mean())
            row["ctrl_dead_frac_flat4"] = float(((sub.kl > DEAD_KL_CUT) | sub.nonfinite).mean())
            try:
                cut = _labels.anchored_cut(root, MODELS[model].base, pool=_native_pool(model))
                row["ctrl_dead_frac_anchored"] = float(
                    ((sub.delta_ce > cut) | sub.nonfinite).mean())
                row["anchor_cut"] = float(cut)
            except (KeyError, FileNotFoundError):
                row["ctrl_dead_frac_anchored"] = np.nan
                row["anchor_cut"] = np.nan
        ks = [v for (_, r), v in kstar.items() if r == model]
        row["kstar_pooled"] = float(np.mean(ks)) if ks else np.nan
        row["n_kstar_arms"] = len(ks)
        row["k_rel_median"] = float(np.median(k_cells / np.mean(ks))) if ks else np.nan
        rows.append(row)
    columns = ["model", "diag_post", "prior", "ctrl_delta_ce", "ctrl_dead_frac_flat4",
               "ctrl_dead_frac_anchored", "anchor_cut", "k_rel_median", "kstar_pooled",
               "n_kstar_arms", "base_pool_entropy_bits", "n_ctrl_runs", "ctrl_senders"]
    return pd.DataFrame(rows).reindex(columns=columns).set_index("model")


def response_frame(arms: pd.DataFrame, receivers: pd.DataFrame, *,
                   statuses: Sequence[str] = ("ok",)) -> pd.DataFrame:
    """The arm rows the decomposition is fitted on, with both responses attached.

    ``log_p_rel = log(mean_p / prior(receiver))`` is a POSITIVE response on a multiplicative
    scale -- how many times its own untuned prior the receiver ends up at -- which keeps receivers
    with very different priors on one additive scale, exactly what a two-way decomposition needs.
    ``mean_rel`` is the control-normalized primary of T2, carried along unchanged.

    Args:
        arms: `arm_table` output.
        receivers: `receiver_covariates` output (its ``prior`` column).
        statuses: Arm statuses kept (default: only complete arms).

    Returns:
        The kept off-diagonal arms plus ``prior`` and ``log_p_rel``.
    """
    out = arms[(arms.sender != arms.receiver) & arms.status.isin(statuses)].copy()
    out["prior"] = out.receiver.map(receivers.prior)
    out["log_p_rel"] = np.log(out.mean_p / out.prior)
    return out.reset_index(drop=True)


#: The two responses every decomposition and regression is run on.
RESPONSES: tuple[str, ...] = ("log_p_rel", "mean_rel")


def decomposition_tables(frame: pd.DataFrame, *, blocks: dict = DECOMP_BLOCKS,
                         responses: Sequence[str] = RESPONSES,
                         extra_factor: str = "relationship") -> tuple[pd.DataFrame, pd.DataFrame]:
    """T7: sender and receiver effects, on each complete block and pooled over every arm.

    Three fits per response: the complete orig/ds1-3 block (12 off-diagonal arms), the complete
    orig/fs1-3 block (12), and every complete arm pooled with `extra_factor` as a third
    sum-to-zero factor so the strata's very different levels cannot be read as model effects.

    Args:
        frame: `response_frame` output.
        blocks: ``{fit name: lineages}`` for the per-block fits.
        responses: Response columns.
        extra_factor: The pooled fit's third factor.

    Returns:
        ``(effects, anova)``: long frames keyed by ``fit`` and ``response``.
    """
    effects, anova = [], []
    fits: list[tuple[str, pd.DataFrame, str | None]] = [
        (name, frame[frame.sender.isin(ms) & frame.receiver.isin(ms)], None)
        for name, ms in blocks.items()]
    fits.append(("pooled", frame, extra_factor))
    for name, sub, extra in fits:
        for response in responses:
            try:
                out = ddt.two_way_decomposition(sub, response, extra_factor=extra)
            except (ValueError, KeyError):
                continue
            effects.append(out.effects.assign(fit=name, response=response))
            anova.append(out.anova.assign(fit=name, response=response, r2=out.r2, n=out.n))
    cat = lambda xs: (pd.concat(xs, ignore_index=True) if xs else pd.DataFrame())   # noqa: E731
    return cat(effects), cat(anova)


#: The sender-side covariates `covariate_tests` regresses the sender effects on -- the channel
#: statistics first, then teacher containment. Coverage counts (``n_pools``, ``n_probes``) are
#: deliberately NOT covariates.
SENDER_COVARIATES: tuple[str, ...] = ("pool_entropy_bits", "pool_top5_mass", "pool_tv_to_base",
                                      "carrier_norm", "teacher_delta_ce", "teacher_kl",
                                      "teacher_delta_ce_cross")

#: The receiver-side covariates, likewise: dose position, control-arm damage, own pool and
#: diagonal. ``anchor_cut`` / ``kstar_pooled`` / the run counts are provenance, not covariates.
RECEIVER_COVARIATES: tuple[str, ...] = ("ctrl_delta_ce", "ctrl_dead_frac_flat4",
                                        "ctrl_dead_frac_anchored", "k_rel_median",
                                        "base_pool_entropy_bits", "diag_post", "prior",
                                        "ctrl_damage_resid", "ctrl_delta_ce_resid_prior")


def covariate_tests(effects: pd.DataFrame, senders: pd.DataFrame, receivers: pd.DataFrame, *,
                    responses: Sequence[str] = RESPONSES, fit: str = "pooled",
                    n_perm: int = 2000, seed: int = 0) -> pd.DataFrame:
    """T8: does a fitted sender/receiver effect track a covariate of that model?

    The unit is the MODEL, so the exchangeable label is the model and the covariate is permuted
    across models -- the same null as everywhere else in this notebook
    (`stats.diagonal_direction_tests.spearman_role` with identity endpoints). With 4-12 models per
    fit these are descriptive: read the sign and the scatter, not a threshold.

    Args:
        effects: `decomposition_tables` effects frame.
        senders: `sender_covariates` output.
        receivers: `receiver_covariates` output.
        responses: Responses tested.
        fit: Which fit's effects are used.
        n_perm: Model-label permutations.
        seed: RNG seed.

    Returns:
        One row per (response, role, covariate): ``n_models, rho, perm_p``.
    """
    numeric = {"sender": senders.reindex(columns=list(SENDER_COVARIATES)),
               "receiver": receivers.reindex(columns=list(RECEIVER_COVARIATES))}
    rows = []
    for response in responses:
        for role, covs in numeric.items():
            sub = effects[(effects.fit == fit) & (effects.response == response)
                          & (effects.role == role)]
            if len(sub) < 3:
                continue
            values = sub.set_index("model").effect
            for name in covs.columns:
                cov = covs[name].reindex(values.index)
                ok = values.notna() & cov.notna()
                if int(ok.sum()) < 3 or cov[ok].nunique() < 2:
                    continue
                sp = ddt.spearman_role(values[ok].to_numpy(), np.arange(int(ok.sum())),
                                       cov[ok].to_numpy(), n_perm=n_perm, seed=seed)
                rows.append({"response": response, "role": role, "covariate": name,
                             "n_models": sp.n, "rho": sp.rho, "perm_p": sp.p})
    return pd.DataFrame(rows)


def pair_covariates(frame: pd.DataFrame, unigrams: dict[str, np.ndarray], senders: pd.DataFrame,
                    ctrl: pd.DataFrame, kstar: dict, teachers: Sequence[str]) -> pd.DataFrame:
    """T9: per ORDERED arm, the number-channel alignment and the pair-level damage/dose.

    ``align_cos`` is the cosine of the two lineages' mean carriers ``Delta = mean teacher unigram
    - base unigram`` -- do x's and y's dog teachers push the number distribution the SAME way?
    It is a property of the unordered pair (symmetric), as is ``base_tv_xy``, the distance between
    the two untuned base pools. ``ctrl_damage_xy`` is the arm's own control probe mean and
    ``kstar_xy`` the arm-specific ``kstar_pooled`` where the boundary-law fits cover it.

    Args:
        frame: `response_frame` output.
        unigrams: `load_pool_unigrams` output.
        senders: `sender_covariates` output (``carrier_norm``).
        ctrl: `control_probe_frame` output.
        kstar: `load_kstar_arms` output's mapping.
        teachers: Teacher param_tags the carriers average over.

    Returns:
        `frame` plus ``align_cos, carrier_norm_x, carrier_norm_y, base_tv_xy, ctrl_damage_xy,
        kstar_xy``.
    """
    from subliminal_learning.numbers import pool_stats

    carriers: dict[str, np.ndarray] = {}
    for model in set(frame.sender) | set(frame.receiver):
        base = unigrams.get(f"base|{model}")
        pools = [unigrams[f"{model}|{t}"] for t in teachers if f"{model}|{t}" in unigrams]
        if base is not None and pools:
            carriers[model] = np.mean(pools, axis=0) - base
    damage = (ctrl.replace(np.inf, np.nan).groupby(["sender", "receiver"]).delta_ce.mean()
              if len(ctrl) else pd.Series(dtype=float))
    out = frame.copy()
    keys = list(zip(out.sender, out.receiver))
    out["align_cos"] = [pool_stats.cosine(carriers[x], carriers[y])
                        if x in carriers and y in carriers else np.nan for x, y in keys]
    out["base_tv_xy"] = [pool_stats.tv(unigrams[f"base|{x}"], unigrams[f"base|{y}"])
                         if f"base|{x}" in unigrams and f"base|{y}" in unigrams else np.nan
                         for x, y in keys]
    out["carrier_norm_x"] = out.sender.map(senders.carrier_norm)
    out["carrier_norm_y"] = out.receiver.map(senders.carrier_norm)
    out["ctrl_damage_xy"] = [damage.get(k, np.nan) for k in keys]
    out["kstar_xy"] = [kstar.get(k, np.nan) for k in keys]
    return out


#: The pair-level regressors of `pair_ols`, in the order they are reported.
PAIR_REGRESSORS: tuple[str, ...] = ("align_cos", "carrier_norm_x", "carrier_norm_y",
                                    "ctrl_damage_xy")


def pair_ols(pairs: pd.DataFrame, *, responses: Sequence[str] = RESPONSES,
             regressors: Sequence[str] = PAIR_REGRESSORS,
             stratum: str = "relationship") -> pd.DataFrame:
    """T9's fit: rank OLS of the arm effect on the pair-level covariates, stratum absorbed.

    Args:
        pairs: `pair_covariates` output.
        responses: Response columns fitted.
        regressors: Numeric pair covariates.
        stratum: Categorical absorbed as dummies.

    Returns:
        One row per (response, term): ``coef, se, t, p, n, r2``.
    """
    rows = []
    for response in responses:
        fit = ddt.rank_ols_multi(pairs, response, regressors, stratum)
        for term, stat in fit.items():
            if term.startswith("_"):
                continue
            rows.append({"response": response, "term": term, **stat,
                         "n": fit["_n"], "r2": fit["_r2"]})
    return pd.DataFrame(rows)


#: The pair-level asymmetry explanations tested in `asymmetry_agreement`, as
#: ``{name: (arm-level column, "receiver" side it is read on)}``.
ASYM_EXPLANATIONS: dict[str, tuple[str, bool]] = {
    "larger receiver carrier_norm": ("carrier_norm_y", True),
    "smaller ctrl_damage": ("ctrl_damage_xy", False),
    "larger receiver diag": ("diag_receiver", True),
}


def asymmetry_agreement(pairs: pd.DataFrame, *, responses: Sequence[str] = RESPONSES,
                        explanations: dict = ASYM_EXPLANATIONS) -> pd.DataFrame:
    """T10: for each pair with both directions, does the stronger direction have the property?

    The same shape as T3's ``into_higher_diag``, with the diagonal replaced by each candidate
    explanation: sign agreement between ``T(x->y) - T(y->x)`` and the difference of the two
    directions' values of the candidate (negated for the candidates predicted to act the other
    way, e.g. LESS control damage), with the exact two-sided binomial p, per stratum and pooled.

    Args:
        pairs: `pair_covariates` output (both directions of a pair must be present).
        responses: Response columns whose asymmetry is tested.
        explanations: ``{name: (column, higher-is-stronger)}``.

    Returns:
        One row per (response, explanation, stratum): ``n_pairs, n_agree, binom_p``.
    """
    idx = pairs.set_index(["sender", "receiver"])
    rows = []
    for response in responses:
        for name, (col, higher) in explanations.items():
            recs = []
            for (x, y) in idx.index:
                if x >= y or (y, x) not in idx.index:
                    continue
                fwd, bwd = idx.loc[(x, y)], idx.loc[(y, x)]
                recs.append({"relationship": fwd.relationship,
                             "asym": fwd[response] - bwd[response],
                             "cand": (fwd[col] - bwd[col]) * (1 if higher else -1)})
            sub = pd.DataFrame(recs)
            if not len(sub):
                continue
            for stratum in [*RELATIONSHIPS, "POOLED"]:
                s = sub if stratum == "POOLED" else sub[sub.relationship == stratum]
                if not len(s):
                    continue
                agree = ddt.sign_agreement(s.asym, s.cand)
                rows.append({"response": response, "explanation": name, "stratum": stratum,
                             "n_pairs": agree.n, "n_agree": agree.n_agree, "binom_p": agree.p})
    return pd.DataFrame(rows)


def covariate_reading(anova: pd.DataFrame, tests: pd.DataFrame, ols: pd.DataFrame,
                      agreement: pd.DataFrame, *, response: str = RESPONSES[0]) -> str:
    """A markdown reading of the WHY sections, generated FROM the frames just computed."""
    lines = []
    for fit in anova.fit.unique():
        a = anova[(anova.fit == fit) & (anova.response == response)].set_index("term")
        share = lambda t: (100 * a.ss_share.get(t, np.nan))      # noqa: E731
        lines.append(
            f"- **Decomposition ({fit}, {response}).** sender {share('sender'):.0f}% of SS "
            f"(p = {a.p.get('sender', np.nan):.3g}), receiver {share('receiver'):.0f}% "
            f"(p = {a.p.get('receiver', np.nan):.3g}), "
            + (f"stratum {share('extra'):.0f}%, " if "extra" in a.index else "")
            + f"residual {share('Residual'):.0f}%; R² = {a.r2.iloc[0]:.2f}, n = {int(a.n.iloc[0])}.")
    for role in ("sender", "receiver"):
        sub = tests[(tests.response == response) & (tests.role == role)]
        if not len(sub):
            continue
        top = sub.reindex(sub.rho.abs().sort_values(ascending=False).index).iloc[0]
        lines.append(
            f"- **{role.capitalize()} covariates.** strongest is `{top.covariate}`, "
            f"rho = {top.rho:.2f} over {int(top.n_models)} models "
            f"(label-permutation p = {top.perm_p:.3g}); {len(sub)} covariates tested, "
            f"{int((sub.perm_p < 0.05).sum())} at p < 0.05.")
    o = ols[(ols.response == response) & ols.term.isin(PAIR_REGRESSORS)]
    if len(o):
        lines.append("- **Pair rank OLS.** " + "; ".join(
            f"{r.term} {r.coef:+.2f} (p = {r.p:.3g})" for r in o.itertuples())
            + f"; n = {int(o.n.iloc[0])}, R² = {o.r2.iloc[0]:.2f}.")
    ag = agreement[(agreement.response == response) & (agreement.stratum == "POOLED")]
    for r in ag.itertuples():
        lines.append(f"- **Asymmetry — {r.explanation}.** {int(r.n_agree)}/{int(r.n_pairs)} pairs "
                     f"agree (binomial p = {r.binom_p:.3g}).")
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------
# 4. residualized receiver covariates, and a fitted predictor of the stronger direction
# ----------------------------------------------------------------------------------------------
#: The lineages pretrained on a v1.0 training-code tree -- the "modern tree" restriction of the
#: receiver-effect correlations, which is where the diagonal/damage confound is checked.
TREE_V10: tuple[str, ...] = tuple(m for m, f in MODELS.items() if f.tree.startswith("v1.0"))


def residual_receiver_covariates(receivers: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """T6 plus the two residualized damage columns.

    ``ctrl_delta_ce`` is NOT a clean receiver property. The control arms it averages differ in WHO
    sent the traitless pool, in how far that sender's base pool sits from the receiver's, and the
    receivers differ in their own dog prior; and across the 12 receivers it is nearly collinear
    with the diagonal. So two residuals, both kept alongside the raw column rather than replacing
    it:

    - ``ctrl_damage_resid``: per ARM, the residual of ``ctrl_damage_xy ~ base_tv_xy + prior +
      C(sender)`` (`stats...ols_residuals`), averaged per receiver -- the damage a receiver takes
      beyond what the sender identity, the base-pool mismatch and its own prior already explain.
    - ``ctrl_delta_ce_resid_prior``: the receiver-level residual of ``ctrl_delta_ce ~ prior``.

    A receiver with no arm carrying all four columns keeps a NaN, never a zero.

    Args:
        receivers: `receiver_covariates` output.
        pairs: `pair_covariates` output (``ctrl_damage_xy``, ``base_tv_xy``, ``prior``).

    Returns:
        `receivers` with ``ctrl_damage_resid`` and ``ctrl_delta_ce_resid_prior`` appended.
    """
    out = receivers.copy()
    resid = ddt.ols_residuals(pairs, "ctrl_damage_xy", ["base_tv_xy", "prior"], ["sender"])
    per_receiver = pd.Series(resid.to_numpy(), index=pairs.receiver.to_numpy()).groupby(
        level=0).mean()
    out["ctrl_damage_resid"] = per_receiver.reindex(out.index)
    out["ctrl_delta_ce_resid_prior"] = ddt.ols_residuals(out, "ctrl_delta_ce", ["prior"])
    return out


#: The receiver covariates the partial-correlation table is run on, and what each is controlled
#: for: the diagonal net of damage and prior, and the damage net of the diagonal and prior.
PARTIAL_TARGETS: dict[str, tuple[str, ...]] = {
    "diag_post": ("ctrl_delta_ce", "prior"),
    "ctrl_delta_ce": ("diag_post", "prior"),
}


def receiver_effect_partials(effects: pd.DataFrame, receivers: pd.DataFrame, *,
                             responses: Sequence[str] = RESPONSES, fit: str = "pooled",
                             targets: dict = PARTIAL_TARGETS,
                             subsets: dict | None = None) -> pd.DataFrame:
    """T11: the receiver effect against the diagonal and against the damage, each net of the other.

    ``diag_post`` and ``ctrl_delta_ce`` are strongly rank-correlated across the receivers, so a
    marginal Spearman with either one cannot say which is doing the work. Each row is therefore
    reported twice: ``controls = ""`` (the marginal Spearman, for reference) and the partial rank
    correlation with the other covariate and the prior held (`stats...partial_spearman`). The
    ``v1.0-tree`` subset repeats both on the `TREE_V10` lineages only, which is where the modern
    re-pretrainings cannot drive the association.

    Args:
        effects: `decomposition_tables` effects frame.
        receivers: `residual_receiver_covariates` output.
        responses: Responses whose receiver effects are correlated.
        fit: Which fit's effects are used.
        targets: ``{covariate: controls}``.
        subsets: ``{name: lineages}``; defaults to all receivers plus the v1.0-tree ones.

    Returns:
        One row per (response, subset, covariate, controls): ``n, rho, p``.
    """
    subsets = subsets or {"all": None, "v1.0-tree": TREE_V10}
    rows = []
    for response in responses:
        sub = effects[(effects.fit == fit) & (effects.response == response)
                      & (effects.role == "receiver")]
        if not len(sub):
            continue
        values = sub.set_index("model").effect
        for name, keep in subsets.items():
            idx = [m for m in values.index if keep is None or m in keep]
            v = values.loc[idx]
            cov = receivers.reindex(idx)
            for target, controls in targets.items():
                for label, cs in (("", ()), ("+".join(controls), controls)):
                    ps = ddt.partial_spearman(v, cov[target], [cov[c] for c in cs])
                    rows.append({"response": response, "subset": name, "covariate": target,
                                 "controls": label, "n": ps.n, "rho": ps.rho, "p": ps.p})
    return pd.DataFrame(rows)


#: The regressors of `receiver_effect_ols` -- the diagonal, the damage and the prior in one fit.
RECEIVER_EFFECT_REGRESSORS: tuple[str, ...] = ("diag_post", "ctrl_delta_ce", "prior")


def receiver_effect_ols(effects: pd.DataFrame, receivers: pd.DataFrame, *,
                        responses: Sequence[str] = RESPONSES, fit: str = "pooled",
                        regressors: Sequence[str] = RECEIVER_EFFECT_REGRESSORS) -> pd.DataFrame:
    """T11's fit: rank OLS of the receiver effect on the diagonal, the damage and the prior.

    The unit here IS the model (one row per receiver), so the usual arms-share-endpoints caveat
    does not apply -- but with n = 12 and three regressors these are coefficients to read, not
    tests to pass.

    Args:
        effects: `decomposition_tables` effects frame.
        receivers: `residual_receiver_covariates` output.
        responses: Responses fitted.
        fit: Which fit's effects are used.
        regressors: Receiver covariates entering the fit.

    Returns:
        One row per (response, term): ``coef, se, t, p, n, r2``.
    """
    rows = []
    for response in responses:
        sub = effects[(effects.fit == fit) & (effects.response == response)
                      & (effects.role == "receiver")]
        if not len(sub):
            continue
        frame = receivers.reindex(sub.model).assign(
            effect=sub.set_index("model").effect.reindex(sub.model).to_numpy())
        out = ddt.rank_ols_multi(frame, "effect", regressors)
        for term, stat in out.items():
            if term.startswith("_"):
                continue
            rows.append({"response": response, "term": term, **stat,
                         "n": out["_n"], "r2": out["_r2"]})
    return pd.DataFrame(rows)


#: The antisymmetric pair features of `direction_pair_frame`: each is ``value(y) - value(x)``.
DIRECTION_FEATURES: tuple[str, ...] = ("ddiag", "dctrl", "dctrl_resid", "dprior", "dcarrier",
                                       "dbase_H", "dctrl_damage")

#: ``{receiver-level covariate: feature name}`` -- the model properties differenced across a pair.
_DIRECTION_SOURCES: dict[str, str] = {"diag_post": "ddiag", "ctrl_delta_ce": "dctrl",
                                      "ctrl_damage_resid": "dctrl_resid", "prior": "dprior",
                                      "base_pool_entropy_bits": "dbase_H"}

#: The feature sets `direction_classifier_table` fits: every feature alone, then three
#: combinations. Interactions with the stratum are deliberately absent -- with ~16 pairs there is
#: no room to fit them; the strata are read off the pooled leave-one-out predictions instead.
DIRECTION_FEATURE_SETS: dict[str, tuple[str, ...]] = (
    {f: (f,) for f in DIRECTION_FEATURES}
    | {"ddiag+dctrl": ("ddiag", "dctrl"),
       "ddiag+dctrl_resid+dprior": ("ddiag", "dctrl_resid", "dprior"),
       "all": DIRECTION_FEATURES})


def direction_pair_frame(pairs: pd.DataFrame, receivers: pd.DataFrame, senders: pd.DataFrame, *,
                         responses: Sequence[str] = RESPONSES) -> pd.DataFrame:
    """T12: one row per unordered pair, with antisymmetric features and the direction label.

    The pair is oriented ``(x, y)`` in `MODEL_ORDER` order and every feature is ``value(y) -
    value(x)``, so flipping the orientation negates all of them -- which is what lets the
    classifier be fitted without an intercept and be exactly antisymmetric. ``dctrl_damage`` is
    the one feature that is not a difference of model properties but of the two ARMS' own control
    damage, ``ctrl_damage_xy - ctrl_damage_yx``. The label is 1 when ``T(x->y) > T(y->x)``.

    Args:
        pairs: `pair_covariates` output (both ordered directions must be present for a pair).
        receivers: `residual_receiver_covariates` output.
        senders: `sender_covariates` output (``carrier_norm``).
        responses: Response columns whose asymmetry is labelled.

    Returns:
        ``x, y, relationship``, the `DIRECTION_FEATURES`, and ``asym_{r}``/``label_{r}``.
    """
    idx = pairs.set_index(["sender", "receiver"])
    order = {m: i for i, m in enumerate(MODEL_ORDER)}
    rows = []
    for (a, b) in idx.index:
        if order[a] > order[b] or (b, a) not in idx.index:
            continue
        x, y = a, b
        fwd, bwd = idx.loc[(x, y)], idx.loc[(y, x)]
        row = {"x": x, "y": y, "relationship": fwd.relationship}
        for col, name in _DIRECTION_SOURCES.items():
            values = receivers[col] if col in receivers else pd.Series(dtype=float)
            row[name] = float(values.get(y, np.nan)) - float(values.get(x, np.nan))
        row["dcarrier"] = (float(senders.carrier_norm.get(y, np.nan))
                           - float(senders.carrier_norm.get(x, np.nan)))
        row["dctrl_damage"] = float(fwd.ctrl_damage_xy) - float(bwd.ctrl_damage_xy)
        for response in responses:
            asym = float(fwd[response]) - float(bwd[response])
            row[f"asym_{response}"] = asym
            row[f"label_{response}"] = float(asym > 0) if np.isfinite(asym) else np.nan
        rows.append(row)
    columns = ["x", "y", "relationship", *DIRECTION_FEATURES]
    columns += [f"{p}_{r}" for r in responses for p in ("asym", "label")]
    return pd.DataFrame(rows).reindex(columns=columns)


def direction_classifier_table(frame: pd.DataFrame, *, responses: Sequence[str] = RESPONSES,
                               feature_sets: dict = DIRECTION_FEATURE_SETS, n_perm: int = 2000,
                               seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """T12's fits: leave-one-PAIR-out accuracy of each feature set, and the two baselines.

    Every fit is `stats...direction_classifier` -- intercept-free L2 logistic regression on the
    pair set augmented with every pair flipped, so the majority-class baseline is 0.5 BY
    CONSTRUCTION (it is reported as a row anyway, so the table says so out loud) and the fixed
    rule "the higher-diagonal model receives" is the substantive baseline. `n_pairs` is per row,
    because a feature set that includes a covariate with a gap is fitted on fewer pairs.

    Args:
        frame: `direction_pair_frame` output.
        responses: Responses whose label is predicted.
        feature_sets: ``{name: features}``.
        n_perm: Label permutations across pairs.
        seed: RNG seed.

    Returns:
        ``(table, preds)``: one row per (response, feature_set) with ``n_pairs, accuracy,
        log_loss, perm_p, coef_signs``, and the per-pair held-out predictions long frame.
    """
    rows, preds = [], []
    for response in responses:
        labels = frame[f"label_{response}"]
        for name, features in feature_sets.items():
            sub = frame[[*features]].join(labels).dropna()
            if len(sub) < 4:
                continue
            fit = ddt.direction_classifier(sub[list(features)].to_numpy(),
                                           sub[f"label_{response}"].to_numpy(), names=features,
                                           n_perm=n_perm, seed=seed)
            rows.append({"response": response, "feature_set": name, "n_features": len(features),
                         "n_pairs": fit.n, "accuracy": fit.accuracy, "log_loss": fit.log_loss,
                         "perm_p": fit.perm_p,
                         "coef_signs": ",".join(f"{k}{'+' if v >= 0 else '-'}"
                                                for k, v in fit.coef.items())})
            block = frame.loc[sub.index, ["x", "y", "relationship"]].copy()
            block["response"], block["feature_set"] = response, name
            block["label"], block["proba"] = sub[f"label_{response}"].to_numpy(), fit.proba
            preds.append(block)
        rule = frame[["ddiag", f"asym_{response}"]].dropna()
        if len(rule):
            hit = np.sign(rule[f"asym_{response}"]) == np.sign(rule.ddiag)
            rows.append({"response": response, "feature_set": "BASELINE higher diag receives",
                         "n_features": 0, "n_pairs": len(rule), "accuracy": float(hit.mean()),
                         "log_loss": np.nan, "perm_p": np.nan, "coef_signs": ""})
        rows.append({"response": response, "feature_set": "BASELINE majority class",
                     "n_features": 0, "n_pairs": int(labels.notna().sum()), "accuracy": 0.5,
                     "log_loss": float(np.log(2)), "perm_p": np.nan,
                     "coef_signs": "0.5 by construction (flip-augmented)"})
    pred = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    if len(pred):
        pred["correct"] = np.where(pred.proba == 0.5, 0.5,
                                   ((pred.proba > 0.5).astype(float) == pred.label).astype(float))
    return pd.DataFrame(rows), pred


def direction_stratum_accuracy(preds: pd.DataFrame) -> pd.DataFrame:
    """T12's per-stratum read: the POOLED leave-one-out predictions split by relationship.

    Not a per-stratum fit -- with two to six pairs in a stratum there is nothing to fit; this is
    the same pooled classifier's accuracy, reported where its pairs happen to live.

    Args:
        preds: `direction_classifier_table`'s predictions frame.

    Returns:
        One row per (response, feature_set, relationship): ``n_pairs, accuracy``.
    """
    if not len(preds):
        return pd.DataFrame(columns=["response", "feature_set", "relationship", "n_pairs",
                                     "accuracy"])
    g = preds.groupby(["response", "feature_set", "relationship"], as_index=False)
    return g.agg(n_pairs=("correct", "size"), accuracy=("correct", "mean"))


def residual_reading(partials: pd.DataFrame, ols: pd.DataFrame, *,
                     response: str = RESPONSES[0]) -> str:
    """A markdown reading of the residualized receiver covariates, from the frames themselves."""
    lines = []
    for subset in partials.subset.unique():
        sub = partials[(partials.response == response) & (partials.subset == subset)
                       & (partials.controls != "")]
        lines.append(f"- **Partials ({subset}, {response}).** " + "; ".join(
            f"{r.covariate} | {r.controls}: rho = {r.rho:.2f} (p = {r.p:.3g}, n = {int(r.n)})"
            for r in sub.itertuples()))
    o = ols[ols.response == response]
    if len(o):
        lines.append("- **Rank OLS (receiver effect).** " + "; ".join(
            f"{r.term} {r.coef:+.2f} (p = {r.p:.3g})"
            for r in o.itertuples() if not r.term.startswith("Intercept"))
            + f"; n = {int(o.n.iloc[0])}, R² = {o.r2.iloc[0]:.2f}.")
    return "\n".join(lines)


def classifier_reading(table: pd.DataFrame, *, response: str = RESPONSES[0]) -> str:
    """A markdown reading of the direction classifier, from the LOO table itself."""
    sub = table[(table.response == response) & ~table.feature_set.str.startswith("BASELINE")]
    base = table[(table.response == response)
                 & (table.feature_set == "BASELINE higher diag receives")]
    if not len(sub):
        return "- No feature set had enough complete pairs to fit."
    best = sub.loc[sub.accuracy.idxmax()]
    lines = [
        f"- **Best feature set ({response}).** `{best.feature_set}`: leave-one-pair-out accuracy "
        f"{best.accuracy:.2f} over {int(best.n_pairs)} pairs, log loss {best.log_loss:.3f}, "
        f"label-permutation p = {best.perm_p:.3g}; signs {best.coef_signs}.",
        "- **Baselines.** majority class 0.5 by construction (the flip-augmented design has "
        "exactly as many 1s as 0s)" + (
            f"; the fixed rule \"higher diag receives\" scores {base.accuracy.iloc[0]:.2f} on "
            f"{int(base.n_pairs.iloc[0])} pairs." if len(base) else "."),
        f"- **Fitted sets at p < 0.05.** {int((sub.perm_p < 0.05).sum())} of {len(sub)}.",
    ]
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------
# 5. the teacher level: one row per (arm, teacher recipe)
# ----------------------------------------------------------------------------------------------
#: The teacher covariates every teacher-level fit controls for -- how strong the teacher's trait
#: is (``logit_pval``), how far it bleeds off topic (``p_offtopic``), how damaged it is
#: (``teacher_delta_ce``) and how concentrated its number pool is (``pool_entropy_bits``).
TEACHER_COVARIATES: tuple[str, ...] = ("logit_pval", "p_offtopic", "teacher_delta_ce",
                                       "pool_entropy_bits")


#: The eval prompt sets the teacher-strength covariate is read from, in preference order. The
#: full teachers carry ``validation``; the CHECKPOINT teachers were evaluated on ``test`` (and
#: ``off_topic``) only, so which one a row used travels with it as ``ckpt_set``.
STRENGTH_SETS: tuple[str, ...] = ("validation", "test")


def _set_p(doc, sets: Sequence[str] = STRENGTH_SETS) -> tuple[float, str]:
    """``(p_target, set name)`` of the first of `sets` the eval doc carries; NaN when it has none."""
    for name in sets:
        if doc is not None and name in doc.get("sets", {}):
            return float(doc["sets"][name]["p_target"]), name
    return float("nan"), ""


def _logit(p: float) -> float:
    """Logit of a probability; the teacher validation p's sit at 0.9999-0.99999, where the raw
    scale has no resolution left and differences are only readable in log-odds."""
    p = float(p)
    return float(np.log(p / (1 - p))) if 0 < p < 1 else np.nan


def teacher_diagonal_table(runs: pd.DataFrame, *, tags: Sequence[str] = DIAG_TAGS,
                           steps: Sequence[int] = DIAG_STEPS) -> pd.DataFrame:
    """Per (lineage, recipe), that ONE recipe's post-cliff checkpoint-distill diagonal.

    `diagonal_table` averages the five recipes together; here they are kept apart, because the
    teacher-level question is whether the recipe that has the higher diagonal on the receiver is
    the recipe that transfers better into it. Only the five ladder recipes exist on the diagonal
    -- the other five k-band teachers were never run as checkpoint distillations -- so a row is
    produced only for `tags`.

    Args:
        runs: `load_diagonal_runs` output.
        tags: Teacher recipes (the five ladder recipes).
        steps: Post-cliff steps averaged.

    Returns:
        Indexed by ``(model, teacher)``, columns ``diag, n_steps``.
    """
    diag = runs[(runs.teacher_step == runs.student_step) & (runs.ftseed == 1)
                & runs.tag.isin(tags) & runs.student_step.isin(steps)]
    g = diag.groupby(["model", "tag"])
    out = pd.DataFrame({"diag": g.p_test.mean(), "n_steps": g.student_step.nunique()})
    out.index.names = ["model", "teacher"]
    return out


def ckpt_teacher_strength(teachers: Sequence[str], *, models: Sequence[str] = MODEL_ORDER,
                          steps: Sequence[int] = DIAG_STEPS, animal: str = ANIMAL,
                          tseed: int = 1, output_root: Path | None = None) -> pd.DataFrame:
    """Per (lineage, recipe), the CHECKPOINT teacher's own trait strength, averaged over `steps`.

    The diagonal of a recipe is not a pure property of the receiving checkpoint: a recipe that
    simply makes a stronger teacher will have a higher diagonal for that reason alone. The
    checkpoint teachers' own evals are the handle on that, and they exist for the orig / ds / fs
    lineages only -- the re-pretrained ones (ws, wc, pr, dclm) have none, so their rows are NaN
    and `n_ckpt` says so. These evals carry no ``validation`` set (only ``test``, and mostly
    ``off_topic``), so the strength is read off the first of `STRENGTH_SETS` present and the set
    that was actually used comes back in ``ckpt_set`` -- never silently.

    Args:
        teachers: Teacher param_tags.
        models: Lineages.
        steps: Checkpoint steps averaged.
        animal: Trait animal.
        tseed: Teacher finetune seed.
        output_root: Output tree root; defaults to the paths YAML's.

    Returns:
        Indexed by ``(model, teacher)``, columns ``ckpt_logit_pval, n_ckpt``.
    """
    root = resolve_output_root(output_root)
    keys: dict[str, tuple[str, str]] = {}
    for model in models:
        for tag in teachers:
            for step in steps:
                rel = (Path("teachers") / animal / "ckpt" / f"{MODELS[model].base}@step{step}"
                       / tag / f"ftseed{tseed}")
                keys[str(rel)] = (model, tag)
    docs = load_run_docs(root, list(keys), "preference_eval.json", "pref_eval")
    acc: dict[tuple[str, str], list[tuple[float, str]]] = {}
    for rel, key in keys.items():
        doc = docs.get(rel)
        if doc is not None:
            p, name = _set_p(doc)
            acc.setdefault(key, []).append((_logit(p), name))
    rows = [{"model": m, "teacher": t,
             "ckpt_logit_pval": float(np.nanmean([x for x, _ in v])), "n_ckpt": len(v),
             "ckpt_set": ",".join(sorted({n for _, n in v if n}))}
            for (m, t), v in acc.items()]
    frame = pd.DataFrame(rows, columns=["model", "teacher", "ckpt_logit_pval", "n_ckpt",
                                        "ckpt_set"])
    return frame.set_index(["model", "teacher"])


def teacher_covariate_table(teachers: Sequence[str], unigrams: dict[str, np.ndarray], *,
                            models: Sequence[str] = MODEL_ORDER, animal: str = ANIMAL,
                            tseed: int = 1, output_root: Path | None = None) -> pd.DataFrame:
    """T13: per (lineage, recipe), what that ONE teacher is -- strength, bleed, damage, pool.

    ``logit_pval`` is the validation-prompt ``p_target`` in log-odds: the raw number is saturated
    (0.9999-0.99999) and differences between recipes only exist on the log-odds scale.
    ``p_offtopic`` is the off-topic-prompt ``p_target`` (0.53-0.85 across these teachers), the
    trait-bleed measure. ``storytelling`` is deliberately absent -- it is missing for several
    lineages and would silently change which teachers a fit is run on. ``teacher_delta_ce`` is
    the teacher's probe on its lineage's NATIVE pool (`_native_pool`).

    Args:
        teachers: Teacher param_tags.
        unigrams: `load_pool_unigrams` output (for the per-teacher pool entropy).
        models: Lineages.
        animal: Trait animal.
        tseed: Teacher finetune seed.
        output_root: Output tree root; defaults to the paths YAML's.

    Returns:
        Indexed by ``(model, teacher)``, columns `TEACHER_COVARIATES` plus ``probe_pool``.
    """
    from subliminal_learning.numbers import pool_stats

    root = resolve_output_root(output_root)
    keys = {str(Path("teachers") / animal / MODELS[m].base / t / f"ftseed{tseed}"): (m, t)
            for m in models for t in teachers}
    docs = load_run_docs(root, list(keys), "preference_eval.json", "pref_eval")
    probes: dict[str, pd.DataFrame] = {
        m: _teacher_probes(root, m, teachers, animal=animal, tseed=tseed, pool=_native_pool(m))
        for m in models}
    rows = []
    for rel, (model, tag) in keys.items():
        doc = docs.get(rel)
        pool = unigrams.get(f"{model}|{tag}")
        probe = probes[model]
        hit = probe[probe.run_path == rel] if len(probe) else probe
        rows.append({
            "model": model, "teacher": tag,
            "logit_pval": _logit(_set_p(doc)[0]),
            "strength_set": _set_p(doc)[1],
            "p_offtopic": (float(doc["sets"]["off_topic"]["p_target"])
                           if doc and "off_topic" in doc["sets"] else np.nan),
            "teacher_delta_ce": (float(hit.delta_ce.replace(np.inf, np.nan).mean())
                                 if len(hit) else np.nan),
            "pool_entropy_bits": (pool_stats.entropy_bits(pool) if pool is not None else np.nan),
            "probe_pool": _native_pool(model)})
    return pd.DataFrame(rows).set_index(["model", "teacher"])


def teacher_arm_frame(runs: pd.DataFrame, tdiag: pd.DataFrame, tcov: pd.DataFrame, *,
                      cells: Sequence[str] | str | Path | None = None) -> pd.DataFrame:
    """T13's rows: one per (arm, teacher recipe), with both responses and both recipe diagonals.

    ``mean_rel_t`` is the mean over the design cells of ``(treatment - control) / prior`` for THAT
    teacher in THAT arm and ``log_p_rel_t = log(mean treatment p / prior)``; the two match
    `arm_table`'s ``mean_rel`` and `response_frame`'s ``log_p_rel``, one recipe at a time.
    ``diag_x``/``diag_y`` are the recipe's own diagonal on the sender and on the receiver, NaN for
    the five k-band recipes that were never run as checkpoint distillations, and ``ddiag_t =
    diag_y - diag_x``. The teacher covariates are the SENDER's teacher: the recipe is a property
    of the numbers, which the sender generated.

    Args:
        runs: `load_arm_runs` output.
        tdiag: `teacher_diagonal_table` output.
        tcov: `teacher_covariate_table` output.
        cells: Design cells the means are restricted to; all loaded cells by default.

    Returns:
        One row per (sender, receiver, teacher) off-diagonal arm.
    """
    sub = runs[runs.sender != runs.receiver]
    if cells is not None:
        sub = sub[sub.cell.isin(list(resolve_cells(cells)))]
    rows = []
    for (sender, receiver, teacher), block in sub.groupby(["sender", "receiver", "teacher"]):
        controlled = block[block.control.notna()]
        prior = float(block.prior.iloc[0])
        dx = tdiag.diag.get((sender, teacher), np.nan)
        dy = tdiag.diag.get((receiver, teacher), np.nan)
        rows.append({
            "sender": sender, "receiver": receiver, "arm": f"{sender}->{receiver}",
            "pair": "|".join(sorted((sender, receiver))), "teacher": teacher,
            "relationship": relationship(sender, receiver),
            "n_cells": int(block.cell.nunique()), "n_cells_ctrl": int(controlled.cell.nunique()),
            "mean_rel_t": float(controlled.value_rel.mean()) if len(controlled) else np.nan,
            "log_p_rel_t": float(np.log(block.treatment.mean() / prior)) if prior > 0 else np.nan,
            "diag_x": float(dx), "diag_y": float(dy), "ddiag_t": float(dy) - float(dx),
            **{c: float(tcov[c].get((sender, teacher), np.nan)) for c in TEACHER_COVARIATES}})
    return pd.DataFrame(rows)


#: The teacher-level responses, in the order the tables report them.
TEACHER_RESPONSES: tuple[str, ...] = ("mean_rel_t", "log_p_rel_t")


def teacher_pair_frame(frame: pd.DataFrame, *,
                       responses: Sequence[str] = TEACHER_RESPONSES) -> pd.DataFrame:
    """One row per (pair, recipe) carrying BOTH directions of that recipe.

    Everything that is a property of the pair -- the two lineages, their base pools, the stratum
    -- cancels in ``asym_{response} = T_t(x->y) - T_t(y->x)``, and ``ddiag_t`` is the matching
    within-pair contrast. This is the frame the clean within-pair test is run on.

    Args:
        frame: `teacher_arm_frame` output.
        responses: Response columns whose asymmetry is formed.

    Returns:
        ``pair, x, y, teacher, relationship, ddiag_t`` and one ``asym_{r}`` per response.
    """
    order = {m: i for i, m in enumerate(MODEL_ORDER)}
    idx = frame.set_index(["sender", "receiver", "teacher"])
    rows = []
    for (a, b, teacher) in idx.index:
        if order[a] > order[b] or (b, a, teacher) not in idx.index:
            continue
        fwd, bwd = idx.loc[(a, b, teacher)], idx.loc[(b, a, teacher)]
        row = {"pair": f"{a}|{b}", "x": a, "y": b, "teacher": teacher,
               "relationship": fwd.relationship, "ddiag_t": fwd.ddiag_t}
        if "ddiag_resid_t" in frame.columns:      # the ckpt-strength-residualized diagonal
            row["ddiag_resid_t"] = fwd.ddiag_resid_t
        for response in responses:
            row[f"asym_{response}"] = float(fwd[response]) - float(bwd[response])
        rows.append(row)
    return pd.DataFrame(rows)


def teacher_direction_consistency(pairs: pd.DataFrame, *,
                                  responses: Sequence[str] = TEACHER_RESPONSES) -> pd.DataFrame:
    """T14 (i): is the direction a PAIR property or a TEACHER property?

    Per pair, the pair-level direction is the sign of the mean asymmetry over the teachers, and
    the count is how many individual teachers agree with it; the exact two-sided binomial p asks
    whether that count is more than a coin would give. A pair whose ten teachers are unanimous is
    saying the direction is a fact about the two models; a 6/10 pair is saying it is mostly a fact
    about which teacher was used.

    Args:
        pairs: `teacher_pair_frame` output.
        responses: Responses tested.

    Returns:
        One row per (response, pair): ``relationship, n_teachers, n_agree, mean_asym, binom_p``.
    """
    rows = []
    for response in responses:
        col = f"asym_{response}"
        for pair, block in pairs.groupby("pair"):
            values = block[col].dropna()
            if not len(values):
                continue
            mean = float(values.mean())
            agree = ddt.sign_agreement(values, np.full(len(values), np.sign(mean) or 1.0))
            rows.append({"response": response, "pair": pair,
                         "relationship": block.relationship.iloc[0], "n_teachers": agree.n,
                         "n_agree": agree.n_agree, "mean_asym": mean, "binom_p": agree.p})
    return pd.DataFrame(rows)


def within_pair_recipe_tests(pairs: pd.DataFrame, *,
                             responses: Sequence[str] = TEACHER_RESPONSES,
                             ddiag_col: str = "ddiag_t", n_perm: int = 2000,
                             seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """T15 (ii): the clean test -- within a pair, does the recipe with the higher receiver
    diagonal transfer better?

    ``asym_t ~ ddiag_t + C(pair)`` with cluster-robust SEs by pair: the pair fixed effect means
    every pair-level confound (the two lineages, their relationship, their base-pool distance,
    their priors) is differenced away, and what is left is the recipe-to-recipe contrast. Its
    permutation null permutes the RECIPE labels within each pair, which keeps the pair structure
    exactly. The companion sign-agreement counts are over (pair, recipe) rows, per stratum and
    pooled.

    Args:
        pairs: `teacher_pair_frame` output (only the five ladder recipes carry ``ddiag_t``).
        responses: Responses tested.
        ddiag_col: The recipe-diagonal contrast (``ddiag_resid_t`` for the residualized repeat).
        n_perm: Within-pair recipe permutations.
        seed: RNG seed.

    Returns:
        ``(fits, agreement)``: one fit row per response (``slope, se, p, perm_p, n, n_pairs``)
        and one agreement row per (response, stratum).
    """
    fits, agree = [], []
    for response in responses:
        col = f"asym_{response}"
        if col not in pairs.columns or ddiag_col not in pairs.columns:
            continue
        sub = pairs[[col, ddiag_col, "pair", "relationship"]].dropna(subset=[col, ddiag_col])
        if not len(sub):
            continue
        fit = ddt.cluster_ols(sub, col, [ddiag_col], fixed_effect="pair", cluster="pair")
        perm = ddt.within_group_slope_permutation(sub, col, ddiag_col, "pair",
                                                  n_perm=n_perm, seed=seed)
        fits.append({"response": response, "ddiag": ddiag_col,
                     "slope": fit.get(ddiag_col, {}).get("coef", np.nan),
                     "se": fit.get(ddiag_col, {}).get("se", np.nan),
                     "p": fit.get(ddiag_col, {}).get("p", np.nan), "perm_p": perm.p,
                     "n_rows": perm.n, "n_pairs": perm.n_groups})
        for stratum in [*RELATIONSHIPS, "POOLED"]:
            s = sub if stratum == "POOLED" else sub[sub.relationship == stratum]
            if not len(s):
                continue
            a = ddt.sign_agreement(s[col], s[ddiag_col])
            agree.append({"response": response, "ddiag": ddiag_col, "stratum": stratum,
                          "n_rows": a.n,
                          "n_agree": a.n_agree, "binom_p": a.p})
    return pd.DataFrame(fits), pd.DataFrame(agree)


def teacher_role_ols(frame: pd.DataFrame, *, responses: Sequence[str] = TEACHER_RESPONSES,
                     regressors: Sequence[str] = ("diag_x", "diag_y")) -> pd.DataFrame:
    """T16 (iii): within an arm, does transfer track the SENDER's or the RECEIVER's recipe diagonal?

    ``T_t ~ diag_x(t) + diag_y(t) + C(arm)`` with cluster-robust SEs by arm. The arm fixed effect
    is what makes this a recipe-level question: the comparison is between recipes inside one
    (sender, receiver) arm, so nothing about the two models can enter the slopes.

    Args:
        frame: `teacher_arm_frame` output.
        responses: Responses fitted.
        regressors: The two recipe diagonals.

    Returns:
        One row per (response, term): ``coef, se, t, p, n, n_arms, r2``.
    """
    rows = []
    for response in responses:
        fit = ddt.cluster_ols(frame, response, list(regressors), fixed_effect="arm",
                              cluster="arm")
        for term, stat in fit.items():
            if term.startswith("_"):
                continue
            rows.append({"response": response, "term": term, **stat, "n": fit["_n"],
                         "n_arms": fit["_n_clusters"], "r2": fit["_r2"]})
    return pd.DataFrame(rows)


def teacher_strength_tests(frame: pd.DataFrame, *, responses: Sequence[str] = TEACHER_RESPONSES,
                           covariates: Sequence[str] = TEACHER_COVARIATES) -> pd.DataFrame:
    """T17 (iv): does the TEACHER's own strength explain the within-arm variation in transfer?

    ``T_t ~ logit_pval + p_offtopic + teacher_delta_ce + pool_entropy_bits + C(arm)``, cluster-
    robust by arm. If a recipe transfers better simply because it makes a stronger, more damaged
    teacher, that shows up here -- and `residualize_teacher_responses` then takes it out before
    the diagonal tests are repeated.

    Args:
        frame: `teacher_arm_frame` output.
        responses: Responses fitted.
        covariates: Teacher covariates entering the fit.

    Returns:
        One row per (response, term): ``coef, se, t, p, n, n_arms, r2``.
    """
    rows = []
    for response in responses:
        fit = ddt.cluster_ols(frame, response, list(covariates), fixed_effect="arm",
                              cluster="arm")
        for term, stat in fit.items():
            if term.startswith("_"):
                continue
            rows.append({"response": response, "term": term, **stat, "n": fit["_n"],
                         "n_arms": fit["_n_clusters"], "r2": fit["_r2"]})
    return pd.DataFrame(rows)


def residualize_teacher_frame(frame: pd.DataFrame, ckpt: pd.DataFrame, *,
                              responses: Sequence[str] = TEACHER_RESPONSES,
                              covariates: Sequence[str] = TEACHER_COVARIATES) -> pd.DataFrame:
    """`teacher_arm_frame` with the teacher's own strength taken out of both sides.

    Two residualizations, both keeping NaN rows as NaN:

    - ``resid_{response}``: the response net of the teacher covariates, with the ARM fixed effect
      in the fit, so what remains is the recipe-to-recipe variation that the teacher's strength,
      bleed, damage and pool concentration do not already explain.
    - ``diag_x_resid`` / ``diag_y_resid`` (and ``ddiag_resid_t``): each endpoint's recipe diagonal
      net of the CHECKPOINT teacher's own ``ckpt_logit_pval``, so a recipe that simply makes a
      stronger teacher does not read as a higher diagonal. Only the lineages with checkpoint
      teacher evals on disk get these; the rest stay NaN.

    Args:
        frame: `teacher_arm_frame` output.
        ckpt: `ckpt_teacher_strength` output.
        responses: Responses residualized.
        covariates: Teacher covariates removed from the responses.

    Returns:
        `frame` plus ``resid_{response}``, ``diag_{x,y}_resid`` and ``ddiag_resid_t``.
    """
    out = frame.copy()
    for response in responses:
        out[f"resid_{response}"] = ddt.ols_residuals(out, response, covariates, ["arm"])
    strength = ckpt.ckpt_logit_pval if len(ckpt) else pd.Series(dtype=float)
    for role, col in (("x", "sender"), ("y", "receiver")):
        keys = list(zip(out[col], out.teacher))
        side = pd.DataFrame({"diag": out[f"diag_{role}"].to_numpy(),
                             "strength": [strength.get(k, np.nan) for k in keys]})
        out[f"diag_{role}_resid"] = ddt.ols_residuals(side, "diag", ["strength"]).to_numpy()
    out["ddiag_resid_t"] = out.diag_y_resid - out.diag_x_resid
    return out


def teacher_reading(consistency: pd.DataFrame, fits: pd.DataFrame, agreement: pd.DataFrame,
                    roles: pd.DataFrame, strength: pd.DataFrame, ckpt: pd.DataFrame, *,
                    response: str = TEACHER_RESPONSES[0]) -> str:
    """A markdown reading of the teacher-level section, generated FROM the frames above."""
    c = consistency[consistency.response == response]
    lines = []
    if len(c):
        lines.append(
            f"- **Direction consistency ({response}).** {len(c)} pairs, "
            f"{int((c.n_agree == c.n_teachers).sum())} unanimous over their teachers; median "
            f"agreement {c.n_agree.median():.0f}/{c.n_teachers.median():.0f}, "
            f"{int((c.binom_p < 0.05).sum())} pairs at binomial p < 0.05.")
    f = fits[fits.response == response]
    if len(f):
        r = f.iloc[0]
        lines.append(
            f"- **Within-pair recipe test.** slope of asym on ddiag_t = {r.slope:+.3f} "
            f"(cluster-robust p = {r.p:.3g}), within-pair recipe permutation p = {r.perm_p:.3g}; "
            f"{int(r.n_rows)} (pair, recipe) rows over {int(r.n_pairs)} pairs.")
    a = agreement[(agreement.response == response) & (agreement.stratum == "POOLED")]
    if len(a):
        lines.append(f"- **Sign agreement (pooled).** {int(a.n_agree.iloc[0])}/"
                     f"{int(a.n_rows.iloc[0])} (pair, recipe) rows agree "
                     f"(binomial p = {a.binom_p.iloc[0]:.3g}).")
    for name, table in (("Sender vs receiver recipe diagonal", roles),
                        ("Teacher strength", strength)):
        t = table[table.response == response]
        if len(t):
            lines.append(f"- **{name}.** " + "; ".join(
                f"{r.term} {r.coef:+.3f} (p = {r.p:.3g})" for r in t.itertuples())
                + f"; n = {int(t.n.iloc[0])} rows over {int(t.n_arms.iloc[0])} arms.")
    have = sorted({m for m, _ in ckpt.index}) if len(ckpt) else []
    lines.append(f"- **Checkpoint teacher evals.** available for {len(have)} lineages "
                 f"({', '.join(have) or 'none'}); the diagonal residualization is defined only "
                 f"there.")
    return "\n".join(lines)
