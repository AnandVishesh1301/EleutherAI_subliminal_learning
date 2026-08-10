"""Checkpoint-ladder analysis frames and C1-C9 verdicts (prereg §6, §7).

Pure domain: pandas / numpy plus `analysis.stats`. No disk, no torch, no config
reads -- `pipelines.ckpt_results_pipeline` loads the tables and writes the CSVs,
this module decides what the numbers mean. Every threshold is a `ckpt_plan`
constant, so `docs/PREREG_CKPT_LADDER.md` and the code cannot drift apart.

Three definitions carry the campaign and are spelled once, here:

- **`d_trait = 1 - trait_coordinate_similarity`** (`attach_d_trait`). C1's x-axis
  is the trait measure ONLY. Global CKA is a preregistered secondary with a
  preregistered null (Okatan et al.: CKA > 0.9 with halved transfer), so
  silently substituting ``1 - minibatch_cka`` when the trait rows are missing
  would answer a different question under C1's name.
- **`y = subliminal_delta(pairing) - subliminal_delta(SI∩SS anchor)`**
  (`attach_y`), within rung, animal, student base and prompt set. The anchor is
  zero by construction; raw `subliminal_delta` is NOT the response.
- **Primary endpoint** (`filter_primary`): ``prompt_set == "test"`` with
  ``scoring == "prob"``. The other sets are a robustness surface, never extra
  claims.

Family membership comes from the pairing set's `Cell`, never from label string
matching: ``SI∩DS`` is the stream (`data`) arm, ``DI∩SS`` the init (`weight`)
arm, and the ``SI∩SS`` anchor / ``DI∩DS`` family-swap corner are excluded from
the C1 fit rather than being folded into whichever arm a heuristic guessed.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats as scipy_stats

from subliminal_learning.analysis import stats as ladder_stats
from subliminal_learning.campaign import ckpt_plan as plan
from subliminal_learning.campaign.ckpt_pairs import Cell, LadderPair, ladder_pairs
from subliminal_learning.campaign.ckpt_replication import (
    is_crossed_anchor_cell,
    leaf_assignments,
)
from subliminal_learning.cka.pairing import canonical_label, split_revision

# The measure C1's x-axis is built from. Spelled once so a claim can never be
# adjudicated on a metric the prereg does not name.
MEASURE_TRAIT = str(plan.Metric.TRAIT_COORDINATE)
MEASURE_GEOM = str(plan.Metric.MINIBATCH_CKA)
MEASURE_SSA = str(plan.Metric.SSA)
MEASURE_SUBSPACE_CKA = "shared_basis_subspace_cka"

# `Cell` -> C1 family, in `analysis.stats` vocabulary (`FAMILY_REFERENCE` = data).
FAMILY_OF_CELL: Mapping[Cell, str | None] = {
    Cell.SI_SS: None,
    Cell.SI_DS: ladder_stats.FAMILY_REFERENCE,
    Cell.DI_SS: ladder_stats.FAMILY_CONTRAST,
    Cell.DI_DS: None,
}

# Subspace angles are degrees in ``[0, 90]``; the trait measure is a similarity
# in ``[0, 1]``. C4 compares their DISSIMILARITIES, so SSA is divided by 90
# before the two are differenced. Recorded in `claims.json` as `normalisation`.
SSA_DEGREES_FULL = 90.0

STATUS_INSUFFICIENT = "insufficient_data"

# The `MIN_OK_RUNGS` bound and its `AdmissionTooNarrow` live in
# `campaign.ckpt_selection.require_min_ok_rungs`, next to the labelling that
# produces the dispositions; the results pipeline calls it before fitting C1.


# ---- frames ------------------------------------------------------------------


def _set_column(frame: pd.DataFrame) -> str:
    """Whichever of ``set`` / ``prompt_set`` this frame spells the endpoint with."""
    for name in ("set", "prompt_set"):
        if name in frame.columns:
            return name
    raise ValueError(
        "frame carries no 'set'/'prompt_set' column, so the primary endpoint "
        f"cannot be isolated; got {sorted(frame.columns)}"
    )


def filter_primary(
    frame: pd.DataFrame,
    *,
    prompt_set: str = plan.PRIMARY_PROMPT_SET,
    scoring: str = plan.PRIMARY_SCORING,
) -> pd.DataFrame:
    """Rows on the single primary endpoint (prereg §6).

    Args:
        frame: Any behavioural frame carrying ``set`` (or ``prompt_set``).
        prompt_set: The preregistered primary set.
        scoring: The preregistered scoring variant, honoured when recorded.

    Returns:
        Only the primary-endpoint rows.

    Raises:
        ValueError: The frame is non-empty and records no prompt set. Pooling
            four prompt sets into one C1 fit would inflate n and mix endpoints,
            so this fails rather than defaulting.
    """
    if frame.empty:
        return frame
    column = _set_column(frame)
    out = frame[frame[column].astype(str) == prompt_set]
    if "scoring" in out.columns:
        out = out[out["scoring"].astype(str) == scoring]
    return out.copy()


def pairing_lookup(pairs: Sequence[LadderPair]) -> dict[tuple[str, str], LadderPair]:
    """``(student, teacher)`` -> pairing, on canonical revision-free labels."""
    return {
        (canonical_label(p.student), canonical_label(p.teacher)): p for p in pairs
    }


def pairing_lookup_undirected(
    pairs: Sequence[LadderPair],
) -> dict[frozenset[str], LadderPair]:
    """``{student, teacher}`` (unordered) -> pairing, on canonical labels.

    Safe ONLY for symmetric geometry metrics (CKA, SSA, Procrustes,
    trait-coordinate similarity), never for a behavioural frame:
    `ckpt_pairs.cell_of` is a pure function of the two labels' `(init_id,
    stream_id)` and is invariant under swapping student/teacher, so every
    `LadderPair` sharing an unordered label pair maps to the same `Cell` (and
    therefore `FAMILY_OF_CELL` entry) regardless of which direction is looked
    up. `pairing_index` on the two directions of one unordered pair can still
    differ (`attach_y`'s ``y`` is direction-sensitive), which is why this is a
    separate function rather than a change to `pairing_lookup` / `attach_y`.
    """
    return {
        frozenset({canonical_label(p.student), canonical_label(p.teacher)}): p
        for p in pairs
    }


def attach_family(
    frame: pd.DataFrame, pairs: Sequence[LadderPair] | None = None
) -> pd.DataFrame:
    """Add ``family`` / ``pairing_index`` / ``cell`` from the pairing set.

    Family comes from `FAMILY_OF_CELL`, so the ``SI∩SS`` anchor and the
    ``DI∩DS`` family-swap corner are ``None`` (excluded from C1) instead of
    being assigned to an arm by a label heuristic.

    Args:
        frame: Rows carrying ``student_init_label`` and ``teacher_init_label``.
        pairs: Active pairing set; defaults to `ladder_pairs()`.

    Returns:
        A copy with the three columns added (``None`` where the pair is not in
        the active set).
    """
    if frame.empty:
        return frame.assign(family=None, pairing_index=None, cell=None)
    lookup = pairing_lookup(pairs if pairs is not None else ladder_pairs())
    keys = [
        (canonical_label(str(s)), canonical_label(str(t)))
        for s, t in zip(
            frame["student_init_label"], frame["teacher_init_label"], strict=True
        )
    ]
    found = [lookup.get(k) for k in keys]
    return frame.assign(
        cell=[None if p is None else str(p.cell) for p in found],
        pairing_index=[None if p is None else p.index for p in found],
        family=[None if p is None else FAMILY_OF_CELL[p.cell] for p in found],
    )


def attach_family_undirected(
    frame: pd.DataFrame, pairs: Sequence[LadderPair] | None = None
) -> pd.DataFrame:
    """`attach_family`, matched by UNORDERED label pair -- geometry frames only.

    P1's cross-init grid (`similarity_ladder_pipeline._run_p1`) walks
    ``itertools.combinations`` over all seven inits in a fixed lexicographic
    order, independent of which side `ladder_pairs()` calls student/teacher.
    `attach_family`'s directed lookup therefore misses most pairs whose P1
    orientation happens to be the reverse of the active set's, which starved
    C4's data-seed-family gap of most of its rows. `pairing_lookup_undirected`
    proves this is exact for `family`/`cell` (`cell_of` is direction-invariant)
    -- unlike `attach_family`, this must never be used on a behavioural
    (delta/``y``) frame, where the two directions of one unordered pair are
    genuinely different trained leaves.

    Args:
        frame: Rows carrying ``student_init_label`` and ``teacher_init_label``
            (a geometry frame; see `with_init_labels`).
        pairs: Active pairing set; defaults to `ladder_pairs()`.

    Returns:
        A copy with ``cell`` / ``pairing_index`` / ``family`` added, matched
        regardless of which physical column holds which label.
    """
    if frame.empty:
        return frame.assign(family=None, pairing_index=None, cell=None)
    lookup = pairing_lookup_undirected(pairs if pairs is not None else ladder_pairs())
    keys = [
        frozenset({canonical_label(str(s)), canonical_label(str(t))})
        for s, t in zip(
            frame["student_init_label"], frame["teacher_init_label"], strict=True
        )
    ]
    found = [lookup.get(k) for k in keys]
    return frame.assign(
        cell=[None if p is None else str(p.cell) for p in found],
        pairing_index=[None if p is None else p.index for p in found],
        family=[None if p is None else FAMILY_OF_CELL[p.cell] for p in found],
    )


def attach_y(delta: pd.DataFrame, pairs: Sequence[LadderPair] | None = None) -> pd.DataFrame:
    """Add the preregistered response ``y`` (prereg §6).

    ``y = subliminal_delta(pairing) - subliminal_delta(SI∩SS, same student base,
    same rung, same set)``. The anchor is the same-init/same-stream pairing on
    that student base, so ``y`` is a within-student, within-rung contrast and
    raw ``p_target`` is never compared across rungs.

    Args:
        delta: `evaluation.gkd_results_io.subliminal_delta` output with the
            `cka.results_io` join keys attached (``student_init_label``,
            ``teacher_init_label``, ``base_revision``).
        pairs: Active pairing set; defaults to `ladder_pairs()`.

    Returns:
        A copy with ``anchor_delta``, ``y``, ``family``, ``pairing_index`` and
        ``cell``. ``y`` is NaN where the student base has no anchor leaf at that
        rung -- never silently 0, which would read as "no transfer gap".
    """
    if delta.empty:
        return delta.assign(anchor_delta=None, y=None, family=None, pairing_index=None)
    out = attach_family(delta, pairs)
    column = _set_column(out)
    keys = ["animal", "student_init_label", "base_revision", column]
    anchors = (
        out[out["cell"] == str(Cell.SI_SS)]
        .groupby(keys, dropna=False)["subliminal_delta"]
        .mean()
        .rename("anchor_delta")
        .reset_index()
    )
    merged = out.merge(anchors, on=keys, how="left")
    merged["y"] = merged["subliminal_delta"].astype(float) - merged[
        "anchor_delta"
    ].astype(float)
    orphans = int(merged["anchor_delta"].isna().sum())
    if orphans:
        logger.warning(
            "{} of {} delta rows have no SI∩SS anchor at their rung; y is NaN there",
            orphans,
            len(merged),
        )
    return merged


def attach_d_trait(
    similarity: pd.DataFrame, *, measure: str = MEASURE_TRAIT, slot: int | None = None
) -> pd.DataFrame:
    """C1's x-axis: ``d_trait = 1 - trait_coordinate_similarity`` (prereg §6).

    Rows for any other metric are dropped rather than converted -- there is no
    fallback to ``1 - minibatch_cka``, because a null on global CKA is itself
    preregistered and substituting it would relabel a different measure as
    ``d_trait``.

    Args:
        similarity: Long-format `cka.results_io` similarity rows.
        measure: The ``metric`` token to read; defaults to `MEASURE_TRAIT`.
        slot: Restrict to one activation slot. ``None`` (the default) averages
            the pair's slots, because the prereg names no single slot; the
            per-slot refit is a notebook panel.

    Returns:
        One row per ``(animal, model_a, model_b, base_revision, step,
        slot_convention)`` with ``d_trait``, ``basis_validity``, ``n_slots``.
        Empty (with those columns) when the measure is absent.
    """
    columns = [
        "animal",
        "model_a",
        "model_b",
        "base_revision",
        "step",
        "slot_convention",
        "basis_validity",
    ]
    if similarity.empty or "metric" not in similarity.columns:
        return pd.DataFrame(columns=[*columns, "d_trait", "n_slots"])
    rows = similarity[similarity["metric"].astype(str) == measure]
    if slot is not None:
        rows = rows[rows["slot"].astype(int) == int(slot)]
    if rows.empty:
        logger.warning(
            "no {} rows in the similarity frame; d_trait is unavailable", measure
        )
        return pd.DataFrame(columns=[*columns, "d_trait", "n_slots"])
    agg = (
        rows.groupby(columns, dropna=False)["value"]
        .agg(["mean", "size"])
        .reset_index()
        .rename(columns={"size": "n_slots"})
    )
    agg["d_trait"] = 1.0 - agg["mean"].astype(float)
    return agg.drop(columns=["mean"])


def c1_fit_frame(
    divergence: pd.DataFrame,
    *,
    quarantine_rungs: Sequence[int] = (),
    exclude_quarantine: bool = True,
) -> pd.DataFrame:
    """The pairing×rung frame C1 is fit on (prereg C1).

    Keeps rows with a finite ``y`` and ``d_trait`` in the two contrast families,
    on the primary endpoint, and collapses replicate leaves to one mean per
    ``(pairing, rung)`` cell -- the unit the cluster bootstrap resamples.

    Args:
        divergence: `attach_y` joined to `attach_d_trait`.
        quarantine_rungs: Rungs whose teachers failed admission.
        exclude_quarantine: Drop those rungs (prereg C1). C3 passes False.

    Returns:
        Columns ``animal, step, pairing_index, family, y, d_trait, n_leaves``.
    """
    empty = pd.DataFrame(
        columns=["animal", "step", "pairing_index", "family", "y", "d_trait", "n_leaves"]
    )
    if divergence.empty or not {"y", "d_trait", "family"} <= set(divergence.columns):
        return empty
    frame = divergence[
        divergence["family"].isin(
            [ladder_stats.FAMILY_REFERENCE, ladder_stats.FAMILY_CONTRAST]
        )
    ].copy()
    frame = frame.dropna(subset=["y", "d_trait", "pairing_index", "step"])
    if exclude_quarantine and len(quarantine_rungs):
        frame = frame[~frame["step"].astype(int).isin([int(r) for r in quarantine_rungs])]
    if frame.empty:
        return empty
    grouped = (
        frame.groupby(["animal", "step", "pairing_index", "family"], dropna=False)
        .agg(y=("y", "mean"), d_trait=("d_trait", "mean"), n_leaves=("y", "size"))
        .reset_index()
    )
    return grouped[
        np.isfinite(grouped["y"]) & np.isfinite(grouped["d_trait"])
    ].reset_index(drop=True)


# ---- claims ------------------------------------------------------------------


def _ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Least-squares slope of `y` on `x`, or NaN on a degenerate design."""
    if x.size < 2 or np.allclose(x, x[0]):
        return float("nan")
    design = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(beta[1])


def _tost_mean_y(frame: pd.DataFrame, bound: float) -> dict | None:
    """TOST on the per-rung ``mean y(weight) - mean y(data)`` -- C1's side diagnostic.

    One value per rung, so the sample is the paired family contrast the prereg
    names rather than a pooled mixture of the two families' rows. ``None`` when
    fewer than two rungs carry both families, or when the contrast is exactly
    constant (`analysis.stats.tost_equivalence` has no t statistic there): the
    primary rule is `beta3` CI containment either way.
    """
    by_rung = (
        frame.pivot_table(index="step", columns="family", values="y", aggfunc="mean")
        .reindex(columns=[ladder_stats.FAMILY_CONTRAST, ladder_stats.FAMILY_REFERENCE])
        .dropna()
    )
    contrast = (
        by_rung[ladder_stats.FAMILY_CONTRAST] - by_rung[ladder_stats.FAMILY_REFERENCE]
    ).to_numpy(dtype=float)
    if contrast.size < 2 or float(contrast.std(ddof=1)) <= 0.0:
        return None
    tost = ladder_stats.tost_equivalence(
        contrast, bound_low=-bound, bound_high=bound
    )
    return {
        "estimate": tost.estimate,
        "ci": [tost.ci_low, tost.ci_high],
        "p_value": tost.p_value,
        "equivalent": tost.equivalent,
        "n_rungs": tost.n_obs,
    }


def adjudicate_c1(
    frame: pd.DataFrame,
    *,
    bound: float = plan.C1_BETA3_EQUIVALENCE_BOUND,
    n_boot: int = plan.BOOTSTRAP_N,
    seed: int = plan.BOOTSTRAP_SEED,
    min_rows: int = 8,
) -> dict:
    """C1 (primary): ``y ~ d_trait + family + d_trait×family`` (prereg C1).

    Verdicts, on the interaction ``beta3`` (weight-minus-data slope difference):

    - ``H2`` -- ``beta3`` CI inside ``±bound`` AND ``beta1`` CI excludes 0. Both
      halves are required: a flat fit has a tiny interaction too, and calling
      that an init-independent channel would be reading a null as a result.
    - ``H1`` -- CI entirely below ``-bound`` (stream steeper).
    - ``H1_reversed`` -- CI entirely above ``+bound``.
    - ``inconclusive`` / ``insufficient_data`` otherwise.

    Args:
        frame: `c1_fit_frame` output.
        bound: `ckpt_plan.C1_BETA3_EQUIVALENCE_BOUND`.
        n_boot: Cluster-bootstrap replicates.
        seed: Bootstrap seed.
        min_rows: Preregistered minimum pairing×rung rows.

    Returns:
        The `claims.json` record, including the cluster-bootstrap interval on
        ``beta3`` over ``(step, pairing_index)`` and the `tost_mean_y` side
        diagnostic.
    """
    if len(frame) < min_rows:
        return {
            "claim": "C1",
            "status": STATUS_INSUFFICIENT,
            "detail": f"{len(frame)} pairing×rung rows, need >= {min_rows}",
            "n_obs": int(len(frame)),
        }
    families = set(frame["family"])
    if len(families) < 2:
        return {
            "claim": "C1",
            "status": STATUS_INSUFFICIENT,
            "detail": f"only the {sorted(families)} family is present",
            "n_obs": int(len(frame)),
        }
    y = frame["y"].to_numpy(dtype=float)
    d = frame["d_trait"].to_numpy(dtype=float)
    fam = frame["family"].tolist()
    steps = frame["step"].astype(int).tolist()
    pairings = frame["pairing_index"].astype(int).tolist()
    cluster = [f"{s}|{p}" for s, p in zip(steps, pairings, strict=True)]
    fit = ladder_stats.family_interaction_fit(y, d, fam, cluster)

    def beta3(rows: np.ndarray) -> float:
        sub = frame.iloc[rows]
        if len(set(sub["family"])) < 2:
            return float("nan")
        return ladder_stats.family_interaction_fit(
            sub["y"].to_numpy(dtype=float),
            sub["d_trait"].to_numpy(dtype=float),
            sub["family"].tolist(),
            [f"{s}|{p}" for s, p in zip(sub["step"], sub["pairing_index"], strict=True)],
        ).interaction.estimate

    boot = ladder_stats.cluster_bootstrap(
        beta3, steps, pairings, n_boot=n_boot, seed=seed
    )
    inside = boot.ci_low >= -bound and boot.ci_high <= bound
    below = boot.ci_high < -bound
    above = boot.ci_low > bound
    slope_nonzero = fit.d_trait.ci_low > 0.0 or fit.d_trait.ci_high < 0.0
    if inside and slope_nonzero:
        status = "H2"
    elif inside:
        status = "inconclusive"
    elif below:
        status = "H1"
    elif above:
        status = "H1_reversed"
    else:
        status = "inconclusive"

    return {
        "claim": "C1",
        "status": status,
        "equivalence_bound": bound,
        "beta1_d_trait": [
            fit.d_trait.estimate,
            fit.d_trait.ci_low,
            fit.d_trait.ci_high,
        ],
        "beta2_family": [fit.family.estimate, fit.family.ci_low, fit.family.ci_high],
        "beta3_interaction": [
            fit.interaction.estimate,
            fit.interaction.ci_low,
            fit.interaction.ci_high,
        ],
        "beta3_cluster_bootstrap": [boot.estimate, boot.ci_low, boot.ci_high],
        "beta3_p_boot": boot.p_boot,
        "tost_mean_y": _tost_mean_y(frame, bound),
        "n_obs": fit.n_obs,
        "n_clusters": fit.n_clusters,
        "family_reference": ladder_stats.FAMILY_REFERENCE,
        "prompt_set": plan.PRIMARY_PROMPT_SET,
    }


def adjudicate_c2(
    frame: pd.DataFrame,
    *,
    n_boot: int = plan.BOOTSTRAP_N,
    seed: int = plan.BOOTSTRAP_SEED,
) -> dict:
    """C2: per family, ``y ~ log(step)`` with a cluster-bootstrap CI (prereg C2).

    Args:
        frame: `c1_fit_frame` output.
        n_boot: Bootstrap replicates.
        seed: Bootstrap seed.

    Returns:
        ``supported`` when at least one family's slope CI excludes 0.
    """
    if frame.empty:
        return {"claim": "C2", "status": STATUS_INSUFFICIENT, "detail": "no rows"}
    per_family: dict[str, dict] = {}
    for family, group in frame.groupby("family"):
        rungs = group["step"].astype(int).to_numpy()
        if group["step"].nunique() < 3:
            per_family[str(family)] = {"status": STATUS_INSUFFICIENT}
            continue
        logs = np.log(np.clip(rungs.astype(float), 1.0, None))
        values = group["y"].to_numpy(dtype=float)

        def slope(rows: np.ndarray, logs=logs, values=values) -> float:
            return _ols_slope(logs[rows], values[rows])

        est = ladder_stats.cluster_bootstrap(
            slope,
            rungs.tolist(),
            group["pairing_index"].astype(int).tolist(),
            n_boot=n_boot,
            seed=seed,
        )
        per_family[str(family)] = {
            "slope": est.estimate,
            "ci": [est.ci_low, est.ci_high],
            "p_boot": est.p_boot,
            "excludes_zero": est.ci_low > 0.0 or est.ci_high < 0.0,
            "n_rungs": int(group["step"].nunique()),
        }
    decided = [f for f in per_family.values() if "excludes_zero" in f]
    if not decided:
        return {
            "claim": "C2",
            "status": STATUS_INSUFFICIENT,
            "per_family": per_family,
        }
    supported = any(f["excludes_zero"] for f in decided)
    return {
        "claim": "C2",
        "status": "supported" if supported else "not_supported",
        "per_family": per_family,
    }


def adjudicate_c3(
    delta: pd.DataFrame,
    *,
    sd_multiple: float = plan.C3_SD_MULTIPLE,
    markers: Sequence[float] = plan.C3_ONSET_MARKERS,
    rungs: Sequence[int] = plan.LADDER_B,
    total_steps: int = plan.LADDER_B[-1],
) -> tuple[dict, pd.DataFrame]:
    """C3: onset of same-init subliminal transfer (prereg C3).

    The onset is the first Ladder-B rung where the mean ``SI∩SS``
    ``subliminal_delta`` clears ``sd_multiple × replicate SD`` for **two
    consecutive** rungs. Where the replicate SD is degenerate (one leaf, or an
    exactly deterministic control), the threshold falls back to
    ``max(sd_multiple·SD, markers[0])``. Quarantined rungs are RETAINED: the
    pre-onset rungs are the evidence.

    Args:
        delta: `attach_y` output (needs ``cell``, ``step``, ``subliminal_delta``).
        sd_multiple: `ckpt_plan.C3_SD_MULTIPLE`.
        markers: Literature onset markers as a fraction of the pretraining
            budget; ``markers[0]`` doubles as the degenerate-SD delta floor.
        rungs: Ladder B, in order.
        total_steps: Final rung, for reporting the markers as steps.

    Returns:
        ``(claim record, ladder_emergence frame)``.
    """
    columns = ["animal", "step", "mean_delta", "replicate_sd", "threshold", "clears", "n"]
    if delta.empty or "cell" not in delta.columns:
        return (
            {"claim": "C3", "status": STATUS_INSUFFICIENT, "detail": "no delta rows"},
            pd.DataFrame(columns=columns),
        )
    anchors = delta[delta["cell"] == str(Cell.SI_SS)]
    if anchors.empty:
        return (
            {"claim": "C3", "status": STATUS_INSUFFICIENT, "detail": "no SI∩SS rows"},
            pd.DataFrame(columns=columns),
        )
    rows: list[dict] = []
    for (animal, step), group in anchors.groupby(["animal", "step"], dropna=False):
        values = group["subliminal_delta"].astype(float).dropna()
        if values.empty:
            continue
        sd = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
        floor = float(markers[0]) if markers else 0.0
        threshold = (
            max(sd_multiple * sd, floor) if math.isfinite(sd) else max(floor, 0.0)
        )
        mean = float(values.mean())
        rows.append(
            {
                "animal": animal,
                "step": int(step),
                "mean_delta": mean,
                "replicate_sd": sd,
                "threshold": threshold,
                "clears": bool(mean >= threshold and threshold > 0.0),
                "n": int(len(values)),
            }
        )
    emergence = pd.DataFrame(rows, columns=columns).sort_values(["animal", "step"])
    onset: int | None = None
    ordered = [int(r) for r in rungs]
    for animal, group in emergence.groupby("animal"):
        clears = {int(r["step"]): bool(r["clears"]) for _, r in group.iterrows()}
        for first, second in zip(ordered, ordered[1:], strict=False):
            if clears.get(first) and clears.get(second):
                onset = first if onset is None else min(onset, first)
                break
    record = {
        "claim": "C3",
        "status": "onset" if onset is not None else "no_onset",
        "onset_step": onset,
        "sd_multiple": sd_multiple,
        "includes_quarantine_rungs": True,
        "onset_markers": [float(m) for m in markers],
        "onset_marker_steps": [int(round(float(m) * total_steps)) for m in markers],
        "n_rungs_scored": int(emergence["step"].nunique()),
    }
    if emergence.empty:
        record["status"] = STATUS_INSUFFICIENT
    return record, emergence


def adjudicate_c4(
    similarity: pd.DataFrame,
    *,
    mad_multiple: float = plan.C4_GAP_MAD_MULTIPLE,
    min_rungs: int = 3,
) -> tuple[dict, pd.DataFrame]:
    """C4: basis breakdown on the data-seed family (prereg C4).

    Per rung, the gap between the rotation-invariant dissimilarity
    (``1 - trait_coordinate``) and the per-coordinate one
    (``shared_basis_subspace_cka`` if present, else ``ssa`` rescaled from
    degrees by `SSA_DEGREES_FULL`). Breakdown is the first rung whose gap
    exceeds ``median + mad_multiple × MAD`` -- median/MAD so one outlier rung
    cannot both inflate the threshold and be the rung that trips it.

    Args:
        similarity: Long-format similarity rows.
        mad_multiple: `ckpt_plan.C4_GAP_MAD_MULTIPLE`.
        min_rungs: Minimum rungs carrying a finite gap.

    Returns:
        ``(claim record, ladder_basis_breakdown frame)``.
    """
    columns = ["step", "rotation_invariant", "per_coordinate", "gap", "n_pairs"]
    if similarity.empty or "metric" not in similarity.columns:
        return (
            {"claim": "C4", "status": STATUS_INSUFFICIENT, "detail": "no geometry rows"},
            pd.DataFrame(columns=columns),
        )
    metrics = set(similarity["metric"].astype(str))
    if MEASURE_SUBSPACE_CKA in metrics:
        per_coord_metric, scale = MEASURE_SUBSPACE_CKA, 1.0
    elif MEASURE_SSA in metrics:
        per_coord_metric, scale = MEASURE_SSA, SSA_DEGREES_FULL
    else:
        return (
            {
                "claim": "C4",
                "status": STATUS_INSUFFICIENT,
                "detail": "neither shared_basis_subspace_cka nor ssa is present",
            },
            pd.DataFrame(columns=columns),
        )
    # Undirected: P1's cross-init grid orders pairs by `combinations`, not by
    # `ladder_pairs()`'s student/teacher direction (see `attach_family_undirected`).
    family = attach_family_undirected(with_init_labels(similarity))
    stream = family[family["family"] == ladder_stats.FAMILY_REFERENCE]
    if stream.empty:
        return (
            {"claim": "C4", "status": STATUS_INSUFFICIENT, "detail": "no data-seed pairs"},
            pd.DataFrame(columns=columns),
        )
    rows: list[dict] = []
    for step, group in stream.groupby("step", dropna=False):
        trait = group[group["metric"].astype(str) == MEASURE_TRAIT]["value"]
        coord = group[group["metric"].astype(str) == per_coord_metric]["value"]
        if trait.empty or coord.empty:
            continue
        rotation = 1.0 - float(trait.astype(float).mean())
        per_coordinate = (
            float(coord.astype(float).mean()) / scale
            if per_coord_metric == MEASURE_SSA
            else 1.0 - float(coord.astype(float).mean())
        )
        rows.append(
            {
                "step": int(step),
                "rotation_invariant": rotation,
                "per_coordinate": per_coordinate,
                "gap": per_coordinate - rotation,
                "n_pairs": int(len(group)),
            }
        )
    frame = pd.DataFrame(rows, columns=columns).sort_values("step")
    finite = frame[np.isfinite(frame["gap"])] if not frame.empty else frame
    if len(finite) < min_rungs:
        return (
            {
                "claim": "C4",
                "status": STATUS_INSUFFICIENT,
                "detail": f"{len(finite)} rungs with a finite gap, need >= {min_rungs}",
                "per_coordinate_metric": per_coord_metric,
            },
            frame,
        )
    gaps = finite["gap"].to_numpy(dtype=float)
    median = float(np.median(gaps))
    mad = float(np.median(np.abs(gaps - median)))
    threshold = median + mad_multiple * mad
    over = finite[finite["gap"] > threshold]
    breakdown = int(over["step"].min()) if not over.empty else None
    return (
        {
            "claim": "C4",
            "status": "breakdown" if breakdown is not None else "no_breakdown",
            "breakdown_step": breakdown,
            "gap_threshold": threshold,
            "gap_median": median,
            "gap_mad": mad,
            "mad_multiple": mad_multiple,
            "per_coordinate_metric": per_coord_metric,
            "normalisation": (
                f"ssa degrees / {SSA_DEGREES_FULL:g}"
                if per_coord_metric == MEASURE_SSA
                else "1 - similarity"
            ),
        },
        frame,
    )


def with_init_labels(similarity: pd.DataFrame) -> pd.DataFrame:
    """Add `cka.results_io.JOIN_KEYS`' two init labels to a geometry frame.

    ``model_a``/``model_b`` carry a revision (``pythia-160m-data-seed1@step2000``);
    the pairing set and the behavioural frame both speak revision-free canonical
    labels, so the join needs both stripped and canonicalised.
    """
    out = similarity.copy()
    out["student_init_label"] = [
        canonical_label(split_revision(str(m))[0]) for m in out["model_a"]
    ]
    out["teacher_init_label"] = [
        canonical_label(split_revision(str(m))[0]) for m in out["model_b"]
    ]
    return out


def adjudicate_c5(
    divergence: pd.DataFrame,
    *,
    threshold: float = plan.C5_SPEARMAN_ABS_QUARANTINE,
) -> dict:
    """C5: capability confound -- per rung Spearman(``y``, ``perplexity_ratio``).

    Args:
        divergence: `attach_y` output joined to geometry; needs
            ``perplexity_ratio``.
        threshold: `ckpt_plan.C5_SPEARMAN_ABS_QUARANTINE`.

    Returns:
        ``quarantine_confound`` when ``|rho|`` reaches the threshold at any rung.
    """
    if divergence.empty or "perplexity_ratio" not in divergence.columns:
        return {
            "claim": "C5",
            "status": STATUS_INSUFFICIENT,
            "detail": "no perplexity_ratio column on the leaf frame",
        }
    frame = divergence.dropna(subset=["y", "perplexity_ratio"])
    per_rung: dict[str, float] = {}
    for step, group in frame.groupby("step", dropna=False):
        cells = (
            group.groupby("pairing_index", dropna=False)
            .agg(y=("y", "mean"), ppl=("perplexity_ratio", "mean"))
            .dropna()
        )
        if len(cells) < 3:
            continue
        rho = scipy_stats.spearmanr(cells["ppl"], cells["y"]).statistic
        if np.isfinite(rho):
            per_rung[str(int(step))] = float(rho)
    if not per_rung:
        return {
            "claim": "C5",
            "status": STATUS_INSUFFICIENT,
            "detail": "no rung with >= 3 pairing means",
        }
    worst = max(per_rung.items(), key=lambda kv: abs(kv[1]))
    return {
        "claim": "C5",
        "status": (
            "quarantine_confound" if abs(worst[1]) >= threshold else "no_confound"
        ),
        "threshold": threshold,
        "max_abs_spearman": {"step": int(worst[0]), "rho": worst[1]},
        "per_rung_spearman": per_rung,
    }


def _mean_ci(values: Sequence[float], confidence: float = 0.95) -> tuple[float, float, float]:
    """Mean and a t-interval; ``(nan, nan, nan)`` on fewer than two values."""
    vals = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if vals.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(vals.mean())
    if vals.size < 2:
        return mean, float("nan"), float("nan")
    half = float(
        scipy_stats.t.ppf(0.5 + confidence / 2.0, vals.size - 1)
        * vals.std(ddof=1)
        / math.sqrt(vals.size)
    )
    return mean, mean - half, mean + half


def adjudicate_c6(
    ladder: pd.DataFrame,
    stage_b: pd.DataFrame,
    *,
    top_rung: int = plan.LADDER_B[-1],
    min_shared: int = 3,
) -> dict:
    """C6: Stage B concordance, pairing-for-pairing at the top rung (prereg C6).

    ``concordant`` iff the ladder and Stage B CIs overlap on EVERY comparable
    shared pairing. Spearman is reported but does not decide: a uniformly
    shifted ladder keeps rank order perfectly while agreeing nowhere, which is
    the failure this claim exists to catch.

    Args:
        ladder: `attach_y` output (this campaign, primary endpoint).
        stage_b: Stage B deltas with the same join labels.
        top_rung: The rung compared (Ladder B's last).
        min_shared: Minimum shared pairings.

    Returns:
        The claim record with a ``per_pairing`` table.
    """
    if ladder.empty or stage_b.empty:
        return {
            "claim": "C6",
            "status": STATUS_INSUFFICIENT,
            "detail": "ladder or Stage B frame is empty",
        }
    top = ladder[ladder["step"].astype(int) == int(top_rung)] if "step" in ladder else ladder
    key = ["student_init_label", "teacher_init_label"]
    if top.empty or not set(key) <= set(stage_b.columns):
        return {
            "claim": "C6",
            "status": STATUS_INSUFFICIENT,
            "detail": f"no ladder rows at step{top_rung} or Stage B lacks {key}",
        }
    per_pairing: list[dict] = []
    for labels, group in top.groupby(key, dropna=False):
        other = stage_b[
            (stage_b["student_init_label"] == labels[0])
            & (stage_b["teacher_init_label"] == labels[1])
        ]
        if other.empty:
            continue
        a_mean, a_low, a_high = _mean_ci(group["subliminal_delta"].astype(float))
        b_mean, b_low, b_high = _mean_ci(other["subliminal_delta"].astype(float))
        comparable = all(np.isfinite([a_low, a_high, b_low, b_high]))
        per_pairing.append(
            {
                "student": labels[0],
                "teacher": labels[1],
                "ladder_mean": a_mean,
                "ladder_ci": [a_low, a_high],
                "stage_b_mean": b_mean,
                "stage_b_ci": [b_low, b_high],
                "comparable": comparable,
                "overlap": bool(comparable and a_low <= b_high and b_low <= a_high),
            }
        )
    comparable = [p for p in per_pairing if p["comparable"]]
    if len(per_pairing) < min_shared or not comparable:
        return {
            "claim": "C6",
            "status": STATUS_INSUFFICIENT,
            "detail": (
                f"{len(per_pairing)} shared pairings, {len(comparable)} with a finite "
                f"CI on both sides (need >= {min_shared} and >= 1)"
            ),
            "per_pairing": per_pairing,
        }
    disjoint = [p for p in comparable if not p["overlap"]]
    rho = scipy_stats.spearmanr(
        [p["ladder_mean"] for p in per_pairing],
        [p["stage_b_mean"] for p in per_pairing],
    ).statistic
    return {
        "claim": "C6",
        "status": "concordant" if not disjoint else "discordant",
        "top_rung": int(top_rung),
        "n_shared": len(per_pairing),
        "n_comparable": len(comparable),
        "n_ci_disjoint": len(disjoint),
        "spearman": float(rho) if np.isfinite(rho) else None,
        "stage_b_run_set_id": plan.STAGE_B_RUN_SET_ID,
        "per_pairing": per_pairing,
    }


def adjudicate_c7(self_drift: pd.DataFrame, similarity: pd.DataFrame) -> dict:
    """C7: self-drift, descriptive (prereg C7 -- no support threshold).

    Reports whether consecutive-rung self-drift is separable from the
    contemporaneous cross-init divergence at the same rung, on the metrics both
    tables carry.

    Args:
        self_drift: The ``self_drift`` result table.
        similarity: The cross-init ``similarity_pairs`` table.

    Returns:
        The claim record.
    """
    if self_drift.empty:
        return {"claim": "C7", "status": STATUS_INSUFFICIENT, "detail": "no drift rows"}
    shared = sorted(
        set(self_drift["metric"].astype(str))
        & (set(similarity["metric"].astype(str)) if not similarity.empty else set())
    )
    per_metric: dict[str, dict] = {}
    for metric in shared:
        drift = self_drift[self_drift["metric"].astype(str) == metric]["value"]
        cross = similarity[similarity["metric"].astype(str) == metric]["value"]
        per_metric[metric] = {
            "self_drift_mean": float(drift.astype(float).mean()),
            "cross_init_mean": float(cross.astype(float).mean()),
            "n_drift_rows": int(len(drift)),
        }
    return {
        "claim": "C7",
        "status": "reported",
        "n_rows": int(len(self_drift)),
        "n_consecutive_pairs": int(
            self_drift.groupby(["step_a", "step_b"]).ngroups
            if {"step_a", "step_b"} <= set(self_drift.columns)
            else 0
        ),
        "per_metric": per_metric,
    }


def adjudicate_c8(probes: pd.DataFrame, grading: pd.DataFrame) -> dict:
    """C8: probe grounding and metric grading, descriptive (prereg C8)."""
    if probes.empty:
        return {"claim": "C8", "status": STATUS_INSUFFICIENT, "detail": "no probe rows"}
    return {
        "claim": "C8",
        "status": "reported",
        "n_probe_rows": int(len(probes)),
        "probe_metrics": sorted(set(probes["metric"].astype(str)))
        if "metric" in probes.columns
        else [],
        "n_graded_metrics": int(len(grading)),
    }


def adjudicate_c9(weight_space: pd.DataFrame) -> dict:
    """C9: weight-space coverage and Frankle barrier, descriptive (prereg C9).

    A `BASIS_BROKEN` pair must carry a ``skip_reason`` and no value; the record
    counts both so a table of nothing but skips cannot read as support.
    """
    if weight_space.empty:
        return {
            "claim": "C9",
            "status": STATUS_INSUFFICIENT,
            "detail": "no weight-space rows",
        }
    values = weight_space["value"] if "value" in weight_space.columns else pd.Series(dtype=float)
    finite = int(np.isfinite(pd.to_numeric(values, errors="coerce")).sum())
    metrics = (
        sorted(set(weight_space["metric"].astype(str)))
        if "metric" in weight_space.columns
        else []
    )
    return {
        "claim": "C9",
        "status": "reported" if finite else STATUS_INSUFFICIENT,
        "n_rows": int(len(weight_space)),
        "n_finite_values": finite,
        "n_skipped": int(len(weight_space) - finite),
        "metrics": metrics,
    }


def regression_rows(c1: Mapping, c2: Mapping) -> pd.DataFrame:
    """C1/C2 coefficients as `ckpt_plan.RESULTS_REGRESSION_CSV` rows."""
    rows: list[dict] = []
    for name in ("beta1_d_trait", "beta2_family", "beta3_interaction"):
        if name in c1:
            estimate, low, high = c1[name]
            rows.append(
                {
                    "claim": "C1",
                    "term": name,
                    "family": None,
                    "estimate": estimate,
                    "ci_low": low,
                    "ci_high": high,
                    "n_obs": c1.get("n_obs"),
                }
            )
    for family, blob in (c2.get("per_family") or {}).items():
        if "slope" not in blob:
            continue
        rows.append(
            {
                "claim": "C2",
                "term": "slope_log_step",
                "family": family,
                "estimate": blob["slope"],
                "ci_low": blob["ci"][0],
                "ci_high": blob["ci"][1],
                "n_obs": blob.get("n_rungs"),
            }
        )
    return pd.DataFrame(
        rows,
        columns=["claim", "term", "family", "estimate", "ci_low", "ci_high", "n_obs"],
    )


# ---- descriptive tables ------------------------------------------------------


REPLICATION_MAP_COLUMNS: tuple[str, ...] = (
    "pairing_index",
    "student",
    "teacher",
    "cell",
    "step",
    "replicate",
    "ftseed",
    "gseed",
    "is_crossed_anchor",
)


def replication_map_rows(
    *,
    include_family_swap: bool = False,
    include_cyclic_closure: bool = False,
    rungs: Sequence[int] = plan.LADDER_B,
) -> pd.DataFrame:
    """Every ``(pairing, rung, replicate) -> (ftseed, gseed)`` leaf (prereg §4).

    Enumerated from `ckpt_replication.leaf_assignments`, the same function the
    job builders submit from, so the map is the submitted grid rather than a
    re-derivation of the rotation that could disagree with it. Crossed-anchor
    cells expand to their nine leaves and are flagged.

    Args:
        include_family_swap: As `ckpt_pairs.ladder_pairs`; it renumbers
            `pairing_index`, so the flags belong with the map.
        include_cyclic_closure: As `ckpt_pairs.ladder_pairs`.
        rungs: Behaviour-ladder rungs.

    Returns:
        `REPLICATION_MAP_COLUMNS` rows, one per leaf.
    """
    rows: list[dict] = []
    for pair in ladder_pairs(include_family_swap, include_cyclic_closure):
        for rung in rungs:
            crossed = is_crossed_anchor_cell(
                pair.index, int(rung), include_family_swap, include_cyclic_closure
            )
            leaves = leaf_assignments(
                pair.index, int(rung), include_family_swap, include_cyclic_closure
            )
            for replicate, (ftseed, gseed) in enumerate(leaves):
                rows.append(
                    {
                        "pairing_index": pair.index,
                        "student": pair.student,
                        "teacher": pair.teacher,
                        "cell": str(pair.cell),
                        "step": int(rung),
                        "replicate": replicate,
                        "ftseed": ftseed,
                        "gseed": gseed,
                        "is_crossed_anchor": crossed,
                    }
                )
    return pd.DataFrame(rows, columns=list(REPLICATION_MAP_COLUMNS))


VARIANCE_COMPONENT_COLUMNS: tuple[str, ...] = (
    "animal",
    "step",
    "response",
    "factor",
    "sum_of_squares",
    "fraction",
    "n_obs",
    "entry_order",
)


def variance_component_rows(
    leaves: pd.DataFrame,
    *,
    response: str = "subliminal_delta",
    rungs: Sequence[int] = plan.CROSSED_ANCHOR_RUNGS,
) -> pd.DataFrame:
    """Sequential SS over ``pairing / ftseed / gseed`` at the crossed anchors.

    Fit ONLY at `ckpt_plan.CROSSED_ANCHOR_RUNGS` (prereg §10): everywhere else
    the rotated diagonal aliases ``ftseed`` with ``gseed``, and a decomposition
    of aliased factors would attribute one seed's variance to the other.

    The response is the per-leaf ``subliminal_delta`` rather than ``y``: ``y``
    subtracts a rung-level ``SI∩SS`` mean, which would move the anchor's own seed
    variance into the residual and shrink the seed terms this table exists to
    size. The column name is recorded on every row.

    Args:
        leaves: Per-leaf frame with ``animal``, ``step``, ``pairing_index``,
            ``ftseed``, ``gseed`` and `response`.
        response: Response column to decompose.
        rungs: The crossed-anchor rungs.

    Returns:
        `VARIANCE_COMPONENT_COLUMNS` rows; empty where a rung has no crossed
        grid, a constant response, or fewer rows than factor levels.
    """
    required = {"animal", "step", "pairing_index", "ftseed", "gseed", response}
    if leaves.empty or not required <= set(leaves.columns):
        return pd.DataFrame(columns=list(VARIANCE_COMPONENT_COLUMNS))
    order = ("pairing", "ftseed", "gseed")
    rows: list[dict] = []
    wanted = {int(r) for r in rungs}
    usable = leaves.dropna(subset=[response, "ftseed", "gseed", "pairing_index", "step"])
    for (animal, step), group in usable.groupby(["animal", "step"], dropna=False):
        if int(step) not in wanted or group["gseed"].nunique() < 2:
            continue
        try:
            components = ladder_stats.variance_components(
                group[response].to_numpy(dtype=float),
                {
                    "pairing": group["pairing_index"].tolist(),
                    "ftseed": group["ftseed"].tolist(),
                    "gseed": group["gseed"].tolist(),
                },
                order=order,
            )
        except ValueError as err:
            logger.info(
                "variance components skipped for {} at step{}: {}", animal, step, err
            )
            continue
        for factor, fraction in components.fractions.items():
            rows.append(
                {
                    "animal": animal,
                    "step": int(step),
                    "response": response,
                    "factor": factor,
                    "sum_of_squares": components.sums_of_squares[factor],
                    "fraction": fraction,
                    "n_obs": components.n_obs,
                    "entry_order": "|".join(order),
                }
            )
    return pd.DataFrame(rows, columns=list(VARIANCE_COMPONENT_COLUMNS))


METRIC_GRADING_COLUMNS: tuple[str, ...] = (
    "metric",
    "n_rungs",
    "rung_span",
    "spearman_vs_probe_accuracy",
    "n_shared_rungs",
)


def metric_grading_rows(
    similarity: pd.DataFrame,
    probes: pd.DataFrame,
    *,
    probe_metric: str = "probe_accuracy",
) -> pd.DataFrame:
    """C8b grading table: each similarity metric against the probe trajectory.

    Descriptive by design (prereg C8 carries no threshold). For every metric,
    the span of its rung means -- a metric that never moves cannot rank anything
    -- and the Spearman of those rung means against the mean probe accuracy at
    the same rungs, which is the "is the geometry functionally readable" axis.

    Args:
        similarity: Long-format cross-init similarity rows.
        probes: The ``probe_grounding`` table.
        probe_metric: Probe row to grade against.

    Returns:
        `METRIC_GRADING_COLUMNS` rows, one per similarity metric.
    """
    if similarity.empty or "metric" not in similarity.columns:
        return pd.DataFrame(columns=list(METRIC_GRADING_COLUMNS))
    probe_by_rung = pd.Series(dtype=float)
    if not probes.empty and "metric" in probes.columns:
        hit = probes[probes["metric"].astype(str) == probe_metric]
        if not hit.empty:
            probe_by_rung = hit.groupby("step")["value"].mean()
    rows: list[dict] = []
    for metric, group in similarity.groupby(similarity["metric"].astype(str)):
        by_rung = group.groupby("step")["value"].mean()
        shared = by_rung.index.intersection(probe_by_rung.index)
        rho = None
        if len(shared) >= 3:
            value = scipy_stats.spearmanr(
                by_rung.loc[shared], probe_by_rung.loc[shared]
            ).statistic
            rho = float(value) if np.isfinite(value) else None
        rows.append(
            {
                "metric": metric,
                "n_rungs": int(by_rung.size),
                "rung_span": float(by_rung.max() - by_rung.min())
                if by_rung.size
                else float("nan"),
                "spearman_vs_probe_accuracy": rho,
                "n_shared_rungs": int(len(shared)),
            }
        )
    return pd.DataFrame(rows, columns=list(METRIC_GRADING_COLUMNS))


QUARANTINE_COLUMNS: tuple[str, ...] = (
    "scope",
    "step",
    "init",
    "model",
    "tier",
    "reason",
    "value",
    "threshold",
)


def quarantine_rows(
    admission: Mapping | None,
    leaves: pd.DataFrame,
    *,
    ppl_max: float = plan.QUARANTINE_PPL_RATIO_MAX,
    filter_drop_max: float = plan.QUARANTINE_FILTER_DROP_MAX,
) -> pd.DataFrame:
    """Admission tiers plus leaf-level quarantine bounds (prereg §8).

    Both scopes land in one table because both are *retained* on disk: a
    quarantined row is dropped from the C1 fit frame, never from the artifact.

    Args:
        admission: `campaign.ckpt_selection.admission_report` body, or None.
        leaves: Per-leaf frame; ``perplexity_ratio`` / ``filter_drop_frac`` are
            each checked when the column is present.
        ppl_max: `ckpt_plan.QUARANTINE_PPL_RATIO_MAX`.
        filter_drop_max: `ckpt_plan.QUARANTINE_FILTER_DROP_MAX`.

    Returns:
        `QUARANTINE_COLUMNS` rows.
    """
    rows: list[dict] = []
    for cell in (admission or {}).get("cells", []):
        if cell.get("tier") == str(plan.AdmissionTier.OK):
            continue
        rows.append(
            {
                "scope": "teacher_admission",
                "step": cell.get("rung"),
                "init": cell.get("init"),
                "model": None,
                "tier": cell.get("tier"),
                "reason": ";".join(cell.get("reasons") or []),
                "value": cell.get("teacher_p_target"),
                "threshold": plan.ADMISSION_ABS_MIN,
            }
        )
    for column, threshold, sense in (
        ("perplexity_ratio", ppl_max, "above"),
        ("filter_drop_frac", filter_drop_max, "above"),
    ):
        if leaves.empty or column not in leaves.columns:
            continue
        values = pd.to_numeric(leaves[column], errors="coerce")
        for _, leaf in leaves[values > threshold].iterrows():
            rows.append(
                {
                    "scope": "leaf",
                    "step": leaf.get("step"),
                    "init": leaf.get("student_init_label"),
                    "model": leaf.get("model"),
                    "tier": str(plan.AdmissionTier.QUARANTINE),
                    "reason": f"{column}_{sense}_{threshold:g}",
                    "value": float(leaf[column]),
                    "threshold": threshold,
                }
            )
    return pd.DataFrame(rows, columns=list(QUARANTINE_COLUMNS))
