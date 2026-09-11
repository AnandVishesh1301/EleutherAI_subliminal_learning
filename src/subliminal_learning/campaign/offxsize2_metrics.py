"""Pure frames and the H1-H8 verdicts of the off-policy cross-size campaign v2.

Sibling of `campaign.offxsize_metrics` (v1), and a thin one on purpose: the Delta
definitions, the required-or-raise control join, the cluster bootstrap and the
reciprocal-asymmetry fit are v1's and are *called* here, not restated. What v2 adds
are the two axes v1 did not have -- a second ANIMAL and an in-training TRAJECTORY --
and the claims that read them (`docs/PREREG_OFFXSIZE2.md` §9).

Leaf-frame schema (`LEAF_COLUMNS`): v1's columns plus, per cell,
``baseline_p_target`` / ``peak_p_target`` / ``peak_step_frac`` /
``peak_epoch_equivalent`` / ``endpoint_p_target`` / ``peak_over_endpoint`` from
`evaluation.offpolicy_probe_io.probe_peak_vs_endpoint`. ``p_target`` stays the
campaign's primary endpoint eval (`test`/`prob`) and is what every adjudicated Delta
is built from; the probe columns are the same run measured at four training fractions
on the probe prompts, and carry the PEAK reading H1 needs to tell an endpoint
artefact from a null.

``animal`` is an aggregation key everywhere. v1 already joins controls on it
(`offxsize_metrics.SB_KEYS` / `TB_KEYS`) and keys `cell_means` on it; the v1 helpers
that key on sizes alone -- `direction_asymmetry`, `additive_fit`, `ppl_coupling` --
are called once per animal here and never on a mixed frame, because two animals can
agree on every other identity field and would otherwise average into one row.

`analysis.stats` / `analysis.decomposition` are imported INSIDE the functions that
use them, exactly as in v1: the `analysis` package's ``__init__`` eagerly pulls a
``*_io`` loader and this module is on the pure-import list
(`tests/unit/test_layering.py`).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from subliminal_learning.campaign import offxsize2_pairs as pairs
from subliminal_learning.campaign import offxsize2_plan as plan
from subliminal_learning.campaign import offxsize2_selection as selection
from subliminal_learning.campaign import offxsize_metrics as om
from subliminal_learning.campaign import offxsize_plan as v1_plan

# v1's `cell_means`, `direction_asymmetry` and `additive_fit` walk v1's OWN size
# ladder. The two campaigns pin that tuple separately, so a divergence would read a
# different set of ordered pairs than v2 enumerated -- silently, and in the direction
# of dropping cells.
if tuple(v1_plan.SIZES) != tuple(plan.SIZES):
    raise ValueError(
        f"v1 size ladder {v1_plan.SIZES} != v2's {plan.SIZES}; the reused v1 frames "
        "would read a different pair set"
    )

SIZES: tuple[str, ...] = plan.SIZES

KIND_TREATMENT = pairs.TREATMENT
KIND_BASE = pairs.BASE_CONTROL
KIND_RNG = pairs.RNG_CONTROL

# From `probe_peak_vs_endpoint`, joined onto the leaf by the results pipeline.
TRAJECTORY_LEAF_COLUMNS: tuple[str, ...] = (
    "baseline_p_target",
    "peak_p_target",
    "peak_step_frac",
    "peak_epoch_equivalent",
    "endpoint_p_target",
    "peak_over_endpoint",
)

LEAF_COLUMNS: tuple[str, ...] = (*om.LEAF_COLUMNS, *TRAJECTORY_LEAF_COLUMNS)

# The identity of one eval row (`offxsize2_pairs.Leaf.key`), used to carry v2-only
# columns across v1's frame builders without a positional join.
LEAF_KEY_COLUMNS: tuple[str, ...] = (
    "animal",
    "phase",
    "kind",
    "teacher",
    "student",
    "recipe_key",
    "rung_tag",
    "ftseed",
    "gseed",
)

_V1_DELTA_SUFFIX: tuple[str, ...] = tuple(
    c for c in om.DELTA_COLUMNS if c not in om.LEAF_COLUMNS
)
PEAK_DELTA_COLUMNS: tuple[str, ...] = (
    "peak_delta_sb",
    "peak_delta_tb",
    "peak_delta_primary",
)
DELTA_COLUMNS: tuple[str, ...] = (
    *LEAF_COLUMNS,
    *_V1_DELTA_SUFFIX,
    *PEAK_DELTA_COLUMNS,
    "primary_control_mean",
)

CELL_KEYS: tuple[str, ...] = ("animal", "teacher", "student", "recipe_key", "rung_tag")
CELL_COLUMNS: tuple[str, ...] = (
    *om.CELL_COLUMNS,
    "mean_peak_delta_primary",
    "mean_peak_delta_tb",
    "peak_ci_low",
    "peak_ci_high",
)

TRAJECTORY_COLUMNS: tuple[str, ...] = (
    *CELL_KEYS,
    "arm",
    "direction",
    "n",
    "mean_baseline_p_target",
    "mean_peak_p_target",
    "mean_endpoint_p_target",
    "mean_peak_over_endpoint",
    "mean_peak_step_frac",
    "mean_peak_epoch_equivalent",
)

RECIPE_COLUMNS: tuple[str, ...] = (
    "animal",
    "teacher",
    "student",
    "rung_tag",
    "arm",
    "direction",
    "n_recipes",
    "mean_delta_tb",
    "sd_delta_tb",
    "spread_delta_tb",
    "argmax_recipe",
    "argmax_delta_tb",
    "anchor_argmax_recipe",
    "argmax_matches_anchor",
    "n_fit",
    "beta_ln_k",
    "r_ln_k",
    "p_ln_k",
    "beta_ln_r",
    "r_ln_r",
    "p_ln_r",
)

RUNG_COLUMNS: tuple[str, ...] = (
    "animal",
    "teacher",
    "student",
    "arm",
    "direction",
    "n_cells",
    "n_rungs",
    "n_recipes",
    "frac_rung",
    "frac_recipe",
    "frac_residual",
    "rung_over_recipe",
    "total_ss",
)

ASYMMETRY_COLUMNS: tuple[str, ...] = ("animal", *om.ASYMMETRY_COLUMNS)

CONTROL_COLUMNS: tuple[str, ...] = (
    *CELL_KEYS,
    "arm",
    "n",
    "mean_p_target",
    "control_p_target",
    "base_prior",
    "control_over_prior",
    "treat_over_control",
    "delta_vs_prior",
    "prior_ratio",
    "control_intact",
    "register_drift",
    "reasons",
)

ANIMAL_COLUMNS: tuple[str, ...] = (
    "animal_a",
    "animal_b",
    "teacher",
    "student",
    "arm",
    "one_step",
    "n_shared",
    "kendall_tau",
    "p_value",
    "sign_agreement",
)

PPL_COLUMNS: tuple[str, ...] = ("animal", *om.PPL_COLUMNS, "r_value", "p_value")

# Recipe geometry for H2's ``ln K`` / ``ln r`` fits. The committed grid plus the two
# recipe sets that can legitimately reach the results frame: the elastic tail's wider
# fan and the P0 concordance cell.
RECIPE_TABLE: Mapping[str, plan.StudentRecipe] = MappingProxyType(
    {
        recipe.key: recipe
        for recipe in (
            *plan.STUDENT_GRID,
            *plan.T4_EXTRA_RECIPES,
            plan.CONCORDANCE_RECIPE,
        )
    }
)

# `offxsize_metrics.propagate_control_quarantine` is rung- and animal-agnostic and
# joins on `SB_KEYS` / `TB_KEYS`, which already carry ``animal``: re-exported rather
# than re-implemented.
propagate_control_quarantine = om.propagate_control_quarantine


# ---- small helpers -----------------------------------------------------------


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _empty(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


def _ratio(numerator: float, denominator: float) -> float:
    if not _finite(numerator) or not _finite(denominator) or denominator == 0.0:
        return float("nan")
    return float(numerator) / float(denominator)


def _ci_excludes_zero(low: float, high: float) -> bool:
    return _finite(low) and _finite(high) and not (low <= 0.0 <= high)


def _ci_covers_zero(low: float, high: float) -> bool:
    return _finite(low) and _finite(high) and low <= 0.0 <= high


def _live(frame: pd.DataFrame) -> pd.DataFrame:
    if "quarantined" not in frame.columns:
        return frame
    return frame.loc[~frame["quarantined"].fillna(False).astype(bool)]


def _treatments(frame: pd.DataFrame) -> pd.DataFrame:
    rows = _live(frame)
    if "kind" in rows.columns:
        rows = rows[rows["kind"].astype(str) == KIND_TREATMENT]
    return rows


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(float("nan"), index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _mean_of(frame: pd.DataFrame, column: str) -> float:
    return float(_numeric(frame, column).mean())


def _value(row: pd.Series, column: str) -> float:
    value = row.get(column, float("nan"))
    return float(value) if _finite(value) else float("nan")


def _plain(value: object) -> object:
    """JSON-safe scalar: numpy types unwrapped, non-finite floats to None."""
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if not math.isfinite(number) else number
    return value


def _animals(frame: pd.DataFrame) -> tuple[str, ...]:
    if frame.empty or "animal" not in frame.columns:
        return ()
    return tuple(sorted({str(a) for a in frame["animal"]}))


def ordered_pairs() -> tuple[tuple[str, str], ...]:
    """All nine ordered ``(teacher, student)`` size pairs, row-major."""
    return tuple((t, s) for t in SIZES for s in SIZES)


def one_step_pairs() -> tuple[tuple[str, str], ...]:
    """The four ordered pairs exactly one step apart -- H1's headline set."""
    return tuple(
        (t, s) for t, s in ordered_pairs() if abs(SIZES.index(s) - SIZES.index(t)) == 1
    )


def _linfit(x: Sequence[float], y: Sequence[float]) -> tuple[float, float, float, int]:
    """``(slope, Pearson r, two-sided p, n)`` of an OLS of `y` on `x`.

    NaN slope/r/p below three finite points or on a constant regressor, where a
    correlation has no sampling distribution to report.
    """
    xv = np.asarray(list(x), dtype=float)
    yv = np.asarray(list(y), dtype=float)
    mask = np.isfinite(xv) & np.isfinite(yv)
    xv, yv = xv[mask], yv[mask]
    if xv.size < 3 or np.allclose(xv, xv[0]):
        return float("nan"), float("nan"), float("nan"), int(xv.size)
    fit = scipy_stats.linregress(xv, yv)
    return float(fit.slope), float(fit.rvalue), float(fit.pvalue), int(xv.size)


def _key_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """`frame` with its `LEAF_KEY_COLUMNS` cast to str, so a join cannot mis-type.

    Raises:
        ValueError: A key column is absent.
    """
    missing = [c for c in LEAF_KEY_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"leaf frame missing identity columns {missing}")
    out = frame.copy()
    for column in LEAF_KEY_COLUMNS:
        out[column] = out[column].astype(str)
    return out


# ---- leaf frames -------------------------------------------------------------


def quarantine(leaves: pd.DataFrame) -> pd.DataFrame:
    """v1's PPL quarantine at v2's relaxed ceiling.

    v2 gates the *denominator* rather than the student
    (`offxsize2_selection.control_integrity`), so the leaf-level ceiling only has to
    catch a student that is no longer a language model.
    """
    return om.quarantine(leaves, ppl_max=plan.QUARANTINE_PPL_RATIO_MAX)


def _peak_view(leaves: pd.DataFrame) -> pd.DataFrame:
    """`leaves` read at the in-training peak instead of at the endpoint.

    Every leaf without a finite peak is marked quarantined first, so
    `propagate_control_quarantine` orphans the treatments over it and v1's
    required-or-raise join skips them: a probe that has not run is a coverage gap,
    not the campaign error that a missing control is.
    """
    view = leaves.copy()
    peak = _numeric(view, "peak_p_target")
    view["p_target"] = peak
    flagged = view.get("quarantined")
    already = (
        pd.Series(False, index=view.index)
        if flagged is None
        else flagged.fillna(False).astype(bool)
    )
    view["quarantined"] = already | ~np.isfinite(peak.to_numpy(dtype=float))
    return om.propagate_control_quarantine(view)


def leaf_deltas(leaves: pd.DataFrame) -> pd.DataFrame:
    """Treatment-leaf Deltas at the ENDPOINT eval and at the in-training PEAK.

    The endpoint pass is `offxsize_metrics.leaf_deltas` unchanged, so v1's contract
    holds verbatim: ``delta_sb`` / ``delta_tb`` / ``delta_rng`` on their own matched
    control keys, ``delta_tb := delta_sb`` on anchors, ``delta_primary`` the ``tb``
    reading on cross pairs and the ``sb`` reading on anchors, and the control behind
    ``delta_primary`` required -- never substituted. The peak pass runs the same
    function over `_peak_view`, so the peak Delta is divided by the control's OWN
    peak and not by its endpoint.

    ``primary_control_mean`` is the control ``delta_primary`` actually divided by;
    H6 reads it.

    Args:
        leaves: Leaf frame with `LEAF_COLUMNS`.

    Returns:
        One row per treatment leaf, `DELTA_COLUMNS`.

    Raises:
        ValueError: The trajectory columns are absent -- a v1-shaped frame cannot be
            read here, because the peak half of every claim would silently vanish.
        KeyError: A live treatment's endpoint ``delta_primary`` control is missing;
            the exception args carry the missing key.
    """
    missing = [c for c in TRAJECTORY_LEAF_COLUMNS if c not in leaves.columns]
    if missing:
        raise ValueError(f"leaf frame missing trajectory columns {missing}")
    if leaves.empty:
        return _empty(DELTA_COLUMNS)
    endpoint = om.leaf_deltas(leaves)
    if endpoint.empty:
        return _empty(DELTA_COLUMNS)
    out = _key_frame(endpoint)
    out["primary_control_mean"] = np.where(
        out["arm"].astype(str) == "cross",
        _numeric(out, "tb_control_mean"),
        _numeric(out, "sb_control_mean"),
    )
    carried = _key_frame(leaves)[[*LEAF_KEY_COLUMNS, *TRAJECTORY_LEAF_COLUMNS]]
    merged = out.merge(
        carried, on=list(LEAF_KEY_COLUMNS), how="left", validate="1:1", indicator=True
    )
    unmatched = merged.loc[merged["_merge"] != "both", list(LEAF_KEY_COLUMNS)]
    if not unmatched.empty:
        raise KeyError(tuple(unmatched.iloc[0]))
    merged = merged.drop(columns="_merge")

    peak = om.leaf_deltas(_peak_view(leaves))
    renames = dict(zip(("delta_sb", "delta_tb", "delta_primary"), PEAK_DELTA_COLUMNS))
    if peak.empty:
        for column in PEAK_DELTA_COLUMNS:
            merged[column] = float("nan")
    else:
        peak = _key_frame(peak)[[*LEAF_KEY_COLUMNS, *renames]].rename(columns=renames)
        merged = merged.merge(
            peak, on=list(LEAF_KEY_COLUMNS), how="left", validate="1:1"
        )
    return merged.reindex(columns=list(DELTA_COLUMNS))


def cell_means(
    deltas: pd.DataFrame,
    *,
    n_boot: int = plan.BOOTSTRAP_N,
    seed: int = plan.BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Per-cell means and percentile bootstrap CIs, at the endpoint and at the peak.

    A cell is `CELL_KEYS` -- ``animal`` included, so two animals that agree on every
    size and hyper-parameter never merge, and ``rung_tag`` included, so the teacher
    dose stays a factor rather than being averaged over. Both halves come from
    `offxsize_metrics.cell_means`: the peak half is the same call over a view whose
    Delta columns are the peak Deltas, so its CI is the same two-way cluster
    bootstrap over teacher ftseeds and student gseeds.

    Returns:
        One row per cell, `CELL_COLUMNS`; ``ci_low`` / ``ci_high`` bound
        ``mean_delta_primary`` and ``peak_ci_low`` / ``peak_ci_high`` bound
        ``mean_peak_delta_primary``.
    """
    if deltas.empty:
        return _empty(CELL_COLUMNS)
    endpoint = om.cell_means(deltas, n_boot=n_boot, seed=seed)
    if endpoint.empty:
        return _empty(CELL_COLUMNS)
    view = deltas.copy()
    view["delta_primary"] = _numeric(deltas, "peak_delta_primary")
    view["delta_sb"] = _numeric(deltas, "peak_delta_sb")
    view["delta_tb"] = _numeric(deltas, "peak_delta_tb")
    view["delta_rng"] = float("nan")
    peak = om.cell_means(view, n_boot=n_boot, seed=seed)
    renames = {
        "mean_delta_primary": "mean_peak_delta_primary",
        "mean_delta_tb": "mean_peak_delta_tb",
        "ci_low": "peak_ci_low",
        "ci_high": "peak_ci_high",
    }
    peak = peak[[*CELL_KEYS, *renames]].rename(columns=renames)
    merged = endpoint.merge(peak, on=list(CELL_KEYS), how="left", validate="1:1")
    return merged.reindex(columns=list(CELL_COLUMNS))


def trajectory(deltas: pd.DataFrame) -> pd.DataFrame:
    """Per-cell peak-vs-endpoint summary of the in-training probe (H4).

    ``mean_baseline_p_target`` is the UNTRAINED prior carried beside the peak, never
    in place of it: `offpolicy_probe_io.probe_peak_vs_endpoint` excludes step 0 from
    both the peak and the endpoint, and this frame keeps that separation.

    Returns:
        One row per cell, `TRAJECTORY_COLUMNS`.
    """
    rows = _treatments(deltas) if not deltas.empty else deltas
    if rows.empty:
        return _empty(TRAJECTORY_COLUMNS)
    out_rows: list[dict] = []
    for key, group in rows.groupby(list(CELL_KEYS), dropna=False, sort=True):
        first = group.iloc[0]
        out_rows.append(
            {
                **dict(zip(CELL_KEYS, key, strict=True)),
                "arm": first.get("arm"),
                "direction": first.get("direction"),
                "n": int(len(group)),
                "mean_baseline_p_target": _mean_of(group, "baseline_p_target"),
                "mean_peak_p_target": _mean_of(group, "peak_p_target"),
                "mean_endpoint_p_target": _mean_of(group, "endpoint_p_target"),
                "mean_peak_over_endpoint": _mean_of(group, "peak_over_endpoint"),
                "mean_peak_step_frac": _mean_of(group, "peak_step_frac"),
                "mean_peak_epoch_equivalent": _mean_of(group, "peak_epoch_equivalent"),
            }
        )
    return pd.DataFrame(out_rows, columns=list(TRAJECTORY_COLUMNS))


# ---- H2 / H3 surfaces --------------------------------------------------------


def _anchor_argmax(cells: pd.DataFrame) -> dict[tuple[str, str, str], str]:
    """``(animal, size, rung) -> recipe`` maximising the same-model ``delta_tb``."""
    anchors = cells[cells["teacher"].astype(str) == cells["student"].astype(str)]
    out: dict[tuple[str, str, str], str] = {}
    if anchors.empty:
        return out
    keys = ["animal", "student", "rung_tag"]
    for key, group in anchors.groupby(keys, dropna=False, sort=True):
        values = _numeric(group, "mean_delta_tb")
        if values.notna().any():
            best = group.loc[values.idxmax()]
            out[tuple(str(k) for k in key)] = str(best["recipe_key"])
    return out


def recipe_surface(
    cells: pd.DataFrame,
    *,
    recipes: Mapping[str, plan.StudentRecipe] = RECIPE_TABLE,
) -> pd.DataFrame:
    """Spread of ``delta_tb`` over the student grid at fixed (animal, pair, rung) (H2).

    Carries the two comparisons §9 asks for: whether the cross-size argmax recipe is
    the same-model argmax at that rung (Angela's data says the optima sit in opposite
    corners), and separate OLS fits of ``delta_tb`` on ``ln K`` and on ``ln r``.
    ``K = lr * sqrt(alpha)`` is `offxsize2_plan.collapse_k`, so the regressor is the
    campaign's own collapse coordinate rather than a restatement of it.

    Args:
        cells: `cell_means` output.
        recipes: Recipe key -> geometry, for ``K`` and ``r``. A key outside the table
            still counts toward the spread but is dropped from the fits, and
            ``n_fit`` reports how many points the fits actually saw.

    Returns:
        One row per (animal, teacher, student, rung), `RECIPE_COLUMNS`.
    """
    if cells.empty:
        return _empty(RECIPE_COLUMNS)
    anchor_best = _anchor_argmax(cells)
    keys = ["animal", "teacher", "student", "rung_tag"]
    out_rows: list[dict] = []
    for key, group in cells.groupby(keys, dropna=False, sort=True):
        key_map = dict(zip(keys, (str(k) for k in key), strict=True))
        values = _numeric(group, "mean_delta_tb")
        if not values.notna().any():
            continue
        best = group.loc[values.idxmax()]
        ln_k: list[float] = []
        ln_r: list[float] = []
        fitted: list[float] = []
        for recipe_key, value in zip(group["recipe_key"], values, strict=True):
            recipe = recipes.get(str(recipe_key))
            if recipe is None or not _finite(value):
                continue
            ln_k.append(math.log(plan.collapse_k(recipe)))
            ln_r.append(math.log(recipe.lora_r))
            fitted.append(float(value))
        beta_k, r_k, p_k, n_fit = _linfit(ln_k, fitted)
        beta_r, r_r, p_r, _ = _linfit(ln_r, fitted)
        anchor_recipe = anchor_best.get(
            (key_map["animal"], key_map["student"], key_map["rung_tag"])
        )
        first = group.iloc[0]
        out_rows.append(
            {
                **key_map,
                "arm": first.get("arm"),
                "direction": first.get("direction"),
                "n_recipes": int(group["recipe_key"].nunique()),
                "mean_delta_tb": float(values.mean()),
                "sd_delta_tb": float(values.std(ddof=1)) if values.count() > 1 else 0.0,
                "spread_delta_tb": float(values.max() - values.min()),
                "argmax_recipe": str(best["recipe_key"]),
                "argmax_delta_tb": float(values.max()),
                "anchor_argmax_recipe": anchor_recipe,
                "argmax_matches_anchor": (
                    None
                    if anchor_recipe is None
                    else str(best["recipe_key"]) == anchor_recipe
                ),
                "n_fit": n_fit,
                "beta_ln_k": beta_k,
                "r_ln_k": r_k,
                "p_ln_k": p_k,
                "beta_ln_r": beta_r,
                "r_ln_r": r_r,
                "p_ln_r": p_r,
            }
        )
    if not out_rows:
        return _empty(RECIPE_COLUMNS)
    return pd.DataFrame(out_rows, columns=list(RECIPE_COLUMNS))


def rung_surface(cells: pd.DataFrame) -> pd.DataFrame:
    """``delta_tb`` variance split across the rung axis vs the recipe axis (H3).

    Sequential (Type I) sums of squares with the rung entered first
    (`analysis.stats.variance_components`); the committed P1 block is a complete
    rung x recipe crossing, where the two main effects are unaliased and the entry
    order does not move them. A group whose ``delta_tb`` is constant to float
    precision has nothing to decompose and reports NaN fractions.

    Returns:
        One row per (animal, teacher, student), `RUNG_COLUMNS`.
    """
    from subliminal_learning.analysis.stats import variance_components

    if cells.empty:
        return _empty(RUNG_COLUMNS)
    keys = ["animal", "teacher", "student"]
    out_rows: list[dict] = []
    for key, group in cells.groupby(keys, dropna=False, sort=True):
        values = _numeric(group, "mean_delta_tb")
        usable = group.loc[values.notna()]
        if usable.empty:
            continue
        fractions: dict[str, float] = {}
        total = float("nan")
        try:
            components = variance_components(
                values.loc[usable.index].to_numpy(dtype=float),
                {
                    "rung": [str(v) for v in usable["rung_tag"]],
                    "recipe": [str(v) for v in usable["recipe_key"]],
                },
                order=("rung", "recipe"),
            )
        except ValueError:
            components = None
        if components is not None:
            fractions = dict(components.fractions)
            total = float(components.total)
        first = usable.iloc[0]
        out_rows.append(
            {
                **dict(zip(keys, (str(k) for k in key), strict=True)),
                "arm": first.get("arm"),
                "direction": first.get("direction"),
                "n_cells": int(len(usable)),
                "n_rungs": int(usable["rung_tag"].nunique()),
                "n_recipes": int(usable["recipe_key"].nunique()),
                "frac_rung": fractions.get("rung", float("nan")),
                "frac_recipe": fractions.get("recipe", float("nan")),
                "frac_residual": fractions.get("residual", float("nan")),
                "rung_over_recipe": _ratio(
                    fractions.get("rung", float("nan")),
                    fractions.get("recipe", float("nan")),
                ),
                "total_ss": total,
            }
        )
    if not out_rows:
        return _empty(RUNG_COLUMNS)
    return pd.DataFrame(out_rows, columns=list(RUNG_COLUMNS))


# ---- H5 direction asymmetry --------------------------------------------------


def _per_animal(
    cells: pd.DataFrame, deltas: pd.DataFrame | None, animal: str
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """`cells` and `deltas` narrowed to one animal, for the v1 size-keyed helpers."""
    sub_cells = cells[cells["animal"].astype(str) == animal]
    if deltas is None or deltas.empty or "animal" not in deltas.columns:
        return sub_cells, deltas
    return sub_cells, deltas[deltas["animal"].astype(str) == animal]


def direction_asymmetry(
    cells: pd.DataFrame,
    *,
    picks: Mapping[str, str],
    deltas: pd.DataFrame | None = None,
    rungs: Mapping[str, str] | None = None,
    teacher_p_target: Mapping[str, float] | None = None,
    n_boot: int = plan.BOOTSTRAP_N,
    seed: int = plan.BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """`offxsize_metrics.direction_asymmetry` per animal, with ``animal`` restored.

    v1's table keys on sizes alone, so a two-animal frame would let one animal's
    cross cell stand in for the other's. Calling it once per narrowed frame keeps
    ``iota = Delta(small->large) - Delta(large->small)`` per (animal, unordered pair).

    Returns:
        `ASYMMETRY_COLUMNS`, ordered by animal then by the size ladder.
    """
    if cells.empty:
        return _empty(ASYMMETRY_COLUMNS)
    frames: list[pd.DataFrame] = []
    for animal in _animals(cells):
        sub_cells, sub_deltas = _per_animal(cells, deltas, animal)
        frame = om.direction_asymmetry(
            sub_cells,
            picks=picks,
            deltas=sub_deltas,
            rungs=rungs,
            teacher_p_target=teacher_p_target,
            n_boot=n_boot,
            seed=seed,
        )
        if frame.empty:
            continue
        frame.insert(0, "animal", animal)
        frames.append(frame)
    if not frames:
        return _empty(ASYMMETRY_COLUMNS)
    out = pd.concat(frames, ignore_index=True)
    return out.reindex(columns=list(ASYMMETRY_COLUMNS))


def additive_fit(
    cells: pd.DataFrame,
    *,
    picks: Mapping[str, str],
    rungs: Mapping[str, str] | None = None,
    deltas: pd.DataFrame | None = None,
    n_boot: int = plan.BOOTSTRAP_N,
    seed: int = plan.BOOTSTRAP_SEED,
) -> dict[str, dict]:
    """`offxsize_metrics.additive_fit` per animal.

    v1 already re-keys and re-signs each reciprocal contrast onto the size ladder
    (`_ladder_contrast`), because `analysis.decomposition` orients an unordered pair
    lexicographically and ``"70m"`` sorts LAST -- which reverses two of the three
    pairs. That correction is v1's and is inherited by calling it.

    Returns:
        Animal -> v1's fit dict (``mu``, ``tau``, ``sigma``, ``r2``, ``loo_r2``,
        ``iota`` / ``interaction`` keyed ``"{small}|{large}"``, ``pooled_iota`` and
        its CI).
    """
    if cells.empty:
        return {}
    out: dict[str, dict] = {}
    for animal in _animals(cells):
        sub_cells, sub_deltas = _per_animal(cells, deltas, animal)
        out[animal] = om.additive_fit(
            sub_cells,
            picks=picks,
            rungs=rungs,
            deltas=sub_deltas,
            n_boot=n_boot,
            seed=seed,
        )
    return out


# ---- H6 control integrity ----------------------------------------------------


def control_integrity(
    deltas: pd.DataFrame,
    *,
    priors: Mapping[tuple[str, str], float],
    min_fraction: float = plan.CONTROL_INTEGRITY_MIN_FRACTION,
    drift_ratio_max: float = plan.H6_DRIFT_RATIO_MAX,
    min_delta: float = plan.H1_MIN_DELTA,
) -> pd.DataFrame:
    """Per cell: the treat/control ratio, the control/prior ratio, and drift (H6).

    ``control_over_prior`` is `offxsize2_selection.control_integrity` -- the same
    verdict the confirm pick gates on, computed here on the same numbers rather than
    restated -- and ``treat_over_control`` divides the cell's mean ``p_target`` by
    the control ``delta_primary`` was actually taken against. A cell is
    **register drift, not transfer** when it looks large against the BARE prior
    (``delta_vs_prior >= min_delta``) while failing to beat its matched control
    (``treat_over_control <= drift_ratio_max``): that is Angela's 410m->160m arm,
    median 0.74x over the 17 cells that have both.

    Args:
        deltas: `leaf_deltas` output.
        priors: ``(animal, student size)`` -> the untuned base's own ``p_target``.
        min_fraction: Control/prior floor below which the denominator has collapsed.
        drift_ratio_max: Treat/control at or below which a large Delta is drift.
        min_delta: What "large against the prior" means; the campaign's one
            pre-registered effect-size bar (`plan.H1_MIN_DELTA`).

    Returns:
        One row per cell, `CONTROL_COLUMNS`.

    Raises:
        CampaignGateError: A cell's base prior is non-positive, which would make its
            control ratio meaningless rather than merely small.
    """
    rows = _treatments(deltas) if not deltas.empty else deltas
    if rows.empty:
        return _empty(CONTROL_COLUMNS)
    cells: list[dict] = []
    readings: dict[tuple[str, str, str, str], selection.ControlReading] = {}
    for key, group in rows.groupby(list(CELL_KEYS), dropna=False, sort=True):
        key_map = dict(zip(CELL_KEYS, (str(k) for k in key), strict=True))
        animal, teacher, student = (
            key_map["animal"],
            key_map["teacher"],
            key_map["student"],
        )
        control = _mean_of(group, "primary_control_mean")
        prior = priors.get((animal, student))
        record = {
            **key_map,
            "arm": group.iloc[0].get("arm"),
            "n": int(len(group)),
            "mean_p_target": _mean_of(group, "p_target"),
            "control_p_target": control,
            "base_prior": float(prior) if prior is not None else float("nan"),
        }
        cells.append(record)
        reading_key = (animal, student, teacher, key_map["recipe_key"])
        readings.setdefault(
            reading_key,
            selection.ControlReading(
                student=student,
                generator=teacher,
                recipe_key=key_map["recipe_key"],
                animal=animal,
                p_target=control if _finite(control) else None,
                base_prior=None if prior is None else float(prior),
            ),
        )
    verdicts = selection.control_integrity(
        list(readings.values()), min_fraction=min_fraction
    )
    out_rows: list[dict] = []
    for record in cells:
        verdict = verdicts[
            (
                record["animal"],
                record["student"],
                record["teacher"],
                record["recipe_key"],
            )
        ]
        prior = record["base_prior"]
        treat = record["mean_p_target"]
        delta_vs_prior = (
            treat - prior if _finite(prior) and _finite(treat) else float("nan")
        )
        ratio = _ratio(treat, record["control_p_target"])
        out_rows.append(
            {
                **record,
                "control_over_prior": (
                    float(verdict.ratio) if verdict.ratio is not None else float("nan")
                ),
                "treat_over_control": ratio,
                "delta_vs_prior": delta_vs_prior,
                "prior_ratio": _ratio(treat, prior),
                "control_intact": bool(verdict.intact),
                "register_drift": bool(
                    _finite(delta_vs_prior)
                    and _finite(ratio)
                    and delta_vs_prior >= min_delta
                    and ratio <= drift_ratio_max
                ),
                "reasons": ";".join(verdict.reasons),
            }
        )
    return pd.DataFrame(out_rows, columns=list(CONTROL_COLUMNS))


# ---- H7 animal generality ----------------------------------------------------


def animal_concordance(
    cells: pd.DataFrame,
    *,
    animals: tuple[str, str] = (plan.PRIMARY_ANIMAL, plan.SECONDARY_ANIMAL),
) -> pd.DataFrame:
    """Kendall tau of the (rung, recipe) ranking across two animals (H7).

    Computed per ordered size pair over the points BOTH animals have measured, plus
    the sign agreement §9 asks for on the one-step pairs. Descriptive: it says
    whether a configuration that works for dog works for lion, not whether either
    transferred.

    Returns:
        One row per shared ordered pair, `ANIMAL_COLUMNS`.
    """
    if cells.empty:
        return _empty(ANIMAL_COLUMNS)
    animal_a, animal_b = animals
    point = ["rung_tag", "recipe_key"]
    left = cells[cells["animal"].astype(str) == animal_a]
    right = cells[cells["animal"].astype(str) == animal_b]
    if left.empty or right.empty:
        return _empty(ANIMAL_COLUMNS)
    headline = set(one_step_pairs())
    out_rows: list[dict] = []
    for teacher, student in ordered_pairs():

        def _at(frame: pd.DataFrame) -> dict[tuple[str, str], float]:
            rows = frame[
                (frame["teacher"].astype(str) == teacher)
                & (frame["student"].astype(str) == student)
            ]
            values = _numeric(rows, "mean_delta_tb")
            return {
                (str(r), str(k)): float(v)
                for (r, k), v in zip(
                    zip(rows[point[0]], rows[point[1]], strict=True),
                    values,
                    strict=True,
                )
                if _finite(v)
            }

        a_points, b_points = _at(left), _at(right)
        shared = sorted(set(a_points) & set(b_points))
        if not shared:
            continue
        a_values = [a_points[p] for p in shared]
        b_values = [b_points[p] for p in shared]
        tau = p_value = float("nan")
        if len(shared) >= 2:
            result = scipy_stats.kendalltau(a_values, b_values)
            if result.statistic is not None and np.isfinite(result.statistic):
                tau = float(result.statistic)
                p_value = float(result.pvalue)
        out_rows.append(
            {
                "animal_a": animal_a,
                "animal_b": animal_b,
                "teacher": teacher,
                "student": student,
                "arm": "anchor" if teacher == student else "cross",
                "one_step": (teacher, student) in headline,
                "n_shared": len(shared),
                "kendall_tau": tau,
                "p_value": p_value,
                "sign_agreement": float(
                    np.mean(
                        [
                            float(np.sign(a) == np.sign(b))
                            for a, b in zip(a_values, b_values, strict=True)
                        ]
                    )
                ),
            }
        )
    if not out_rows:
        return _empty(ANIMAL_COLUMNS)
    return pd.DataFrame(out_rows, columns=list(ANIMAL_COLUMNS))


# ---- H8 capability coupling --------------------------------------------------


def ppl_coupling(deltas: pd.DataFrame) -> pd.DataFrame:
    """OLS of ``delta_primary`` on ``perplexity_ratio`` WITHIN design strata (H8).

    A stratum is (animal, teacher, student, recipe, rung) -- one animal, one teacher
    dose, one student geometry. v1's `ppl_coupling` supplies ``beta`` and the
    residualised ``retained_fraction``; the Pearson ``r`` and its p are added here
    because §9 reports the coupling as a correlation. Pooling the strata is a
    separate, explicitly non-adjudicating call (`pooled_ppl_fit`).

    Returns:
        One row per stratum, `PPL_COLUMNS`.
    """
    rows = _treatments(deltas) if not deltas.empty else deltas
    if rows.empty or "perplexity_ratio" not in rows.columns:
        return _empty(PPL_COLUMNS)
    keys = list(om.PPL_COLUMNS[:4])
    frames: list[pd.DataFrame] = []
    for animal in _animals(rows):
        sub = rows[rows["animal"].astype(str) == animal]
        frame = om.ppl_coupling(sub)
        if frame.empty:
            continue
        fits = []
        for key, group in sub.groupby(keys, dropna=False, sort=True):
            _slope, r_value, p_value, _n = _linfit(
                _numeric(group, "perplexity_ratio"), _numeric(group, "delta_primary")
            )
            fits.append(
                {
                    **dict(zip(keys, (str(k) for k in key), strict=True)),
                    "r_value": r_value,
                    "p_value": p_value,
                }
            )
        frame = frame.astype({key: str for key in keys}).merge(
            pd.DataFrame(fits), on=keys, how="left", validate="1:1"
        )
        frame.insert(0, "animal", animal)
        frames.append(frame)
    if not frames:
        return _empty(PPL_COLUMNS)
    return pd.concat(frames, ignore_index=True).reindex(columns=list(PPL_COLUMNS))


def pooled_ppl_fit(deltas: pd.DataFrame) -> dict:
    """The same OLS with every stratum POOLED -- reported for contrast, never read.

    v1 measured what pooling does here: r = +0.66 at p ~ 1e-17 across strata against
    r = -0.07 at p = 0.53 on the headline cells alone. The slope is a between-stratum
    contrast wearing a within-stratum name, so it is carried beside the stratified
    fit to show the artefact, and `claims` marks it non-adjudicating.

    Returns:
        ``{"slope", "r_value", "p_value", "n", "adjudicated": False}``.
    """
    rows = _treatments(deltas) if not deltas.empty else deltas
    slope, r_value, p_value, n = (
        _linfit(_numeric(rows, "perplexity_ratio"), _numeric(rows, "delta_primary"))
        if not rows.empty
        else (float("nan"), float("nan"), float("nan"), 0)
    )
    return {
        "slope": _plain(slope),
        "r_value": _plain(r_value),
        "p_value": _plain(p_value),
        "n": int(n),
        "adjudicated": False,
    }


# ---- verdicts ----------------------------------------------------------------


def _verdict(n_units: int, n_holds: int, detail: dict[str, Any]) -> dict:
    """`holds` True only when every measurable unit passes; void when none exist."""
    if n_units == 0:
        return {"holds": None, "void": True, "detail": detail}
    return {"holds": n_holds == n_units, "void": False, "detail": detail}


def _descriptive(detail: dict[str, Any], *, void: bool) -> dict:
    """A claim that reports and never adjudicates (H2, H3, H4, H7)."""
    return {"holds": None, "void": void, "detail": detail}


def _reading(row: pd.Series, reading: str) -> tuple[float, float, float]:
    """``(mean delta_tb, ci_low, ci_high)`` of one cell at one reading."""
    if reading == "peak":
        return (
            _value(row, "mean_peak_delta_tb"),
            _value(row, "peak_ci_low"),
            _value(row, "peak_ci_high"),
        )
    return _value(row, "mean_delta_tb"), _value(row, "ci_low"), _value(row, "ci_high")


def _clears(row: pd.Series, reading: str, min_delta: float) -> bool:
    mean, low, high = _reading(row, reading)
    return _ci_excludes_zero(low, high) and _finite(mean) and mean >= min_delta


def _point(row: pd.Series, reading: str) -> dict:
    mean, low, high = _reading(row, reading)
    return {
        "reading": reading,
        "rung_tag": str(row["rung_tag"]),
        "recipe_key": str(row["recipe_key"]),
        "mean": _plain(mean),
        "ci": [_plain(low), _plain(high)],
        "n": int(row["n"]) if _finite(row.get("n")) else None,
    }


def _best(rows: Sequence[pd.Series], reading: str) -> pd.Series | None:
    finite = [r for r in rows if _finite(_reading(r, reading)[0])]
    if not finite:
        return None
    return max(finite, key=lambda r: _reading(r, reading)[0])


def _h1_pair(
    points: Sequence[pd.Series],
    *,
    headline: bool,
    min_delta: float,
    skipped: Mapping[str, int],
) -> dict:
    """One ordered cross pair's H1 reading: the coverage-aware max over its points."""
    clearing = [r for r in points if _clears(r, "endpoint", min_delta)]
    peak_clearing = [r for r in points if _clears(r, "peak", min_delta)]
    best = _best(clearing, "endpoint")
    peak_best = _best(peak_clearing, "peak")
    max_point = _best(points, "endpoint")
    return {
        "void": False,
        "holds": bool(clearing),
        "headline": headline,
        "n_points": len(points),
        "n_clearing": len(clearing),
        "n_clearing_peak": len(peak_clearing),
        "best": _point(best, "endpoint") if best is not None else None,
        "max_point": _point(max_point, "endpoint") if max_point is not None else None,
        "peak_best": _point(peak_best, "peak") if peak_best is not None else None,
        # Pre-registered in §8: at 410m off-policy transfer peaks near 20% of
        # training and decays 3.7x by the ep20 endpoint, so a pair that only ever
        # cleared the bar in-flight is an endpoint artefact and NOT a pass.
        "endpoint_artefact": not clearing and bool(peak_clearing),
        "skipped": dict(skipped),
    }


def _h1(
    cells: pd.DataFrame,
    rungs: Mapping[str, Mapping[str, Sequence[str]]] | None,
    *,
    min_delta: float = plan.H1_MIN_DELTA,
) -> dict:
    """H1 existence: does cross-size transfer appear ANYWHERE in the sampled space?

    Existential by construction -- §9 reads it as "H1 holds at some point" against
    "H1 fails everywhere" -- so the top-level `holds` is True as soon as one
    (animal, ordered cross pair) clears, and the per-pair verdicts carry the rest.
    """
    verdicts: dict[str, dict] = {}
    n_units = n_holds = 0
    headline_units = headline_holds = 0
    artefacts: list[str] = []
    headline = set(one_step_pairs())
    for animal in _animals(cells):
        admitted = (
            None
            if rungs is None
            else {
                k: tuple(str(t) for t in v)
                for k, v in (rungs.get(animal) or {}).items()
            }
        )
        by_animal = cells[cells["animal"].astype(str) == animal]
        for teacher, student in ordered_pairs():
            if teacher == student:
                continue
            label = f"{animal}|{teacher}->{student}"
            if admitted is not None and not admitted.get(teacher):
                verdicts[label] = {"void": True, "reason": "no_admissible_teacher"}
                continue
            points = by_animal[
                (by_animal["teacher"].astype(str) == teacher)
                & (by_animal["student"].astype(str) == student)
            ]
            if admitted is not None:
                points = points[points["rung_tag"].astype(str).isin(admitted[teacher])]
            if points.empty:
                verdicts[label] = {"void": True, "reason": "missing_cell"}
                continue
            anchors = by_animal[
                (by_animal["teacher"].astype(str) == student)
                & (by_animal["student"].astype(str) == student)
            ]
            anchor_ci = {
                (str(row["rung_tag"]), str(row["recipe_key"])): (
                    _value(row, "ci_low"),
                    _value(row, "ci_high"),
                )
                for _, row in anchors.iterrows()
            }
            admissible: list[pd.Series] = []
            skipped = {"missing_anchor": 0, "anchor_ci_covers_0": 0}
            for _, row in points.iterrows():
                bounds = anchor_ci.get((str(row["rung_tag"]), str(row["recipe_key"])))
                if bounds is None:
                    skipped["missing_anchor"] += 1
                elif _ci_covers_zero(*bounds):
                    skipped["anchor_ci_covers_0"] += 1
                else:
                    admissible.append(row)
            if not admissible:
                verdicts[label] = {
                    "void": True,
                    "reason": (
                        "anchor_ci_covers_0"
                        if skipped["anchor_ci_covers_0"]
                        else "missing_anchor"
                    ),
                    "skipped": skipped,
                }
                continue
            is_headline = (teacher, student) in headline
            record = _h1_pair(
                admissible,
                headline=is_headline,
                min_delta=min_delta,
                skipped=skipped,
            )
            verdicts[label] = record
            n_units += 1
            n_holds += int(record["holds"])
            headline_units += int(is_headline)
            headline_holds += int(is_headline and record["holds"])
            if record["endpoint_artefact"]:
                artefacts.append(label)
    detail = {
        "pairs": verdicts,
        "n": n_units,
        "n_holds": n_holds,
        "headline": {"n": headline_units, "n_holds": headline_holds},
        "endpoint_artefacts": artefacts,
        "min_delta": min_delta,
        "prompt_set": plan.PRIMARY_PROMPT_SET,
        "existential": True,
    }
    if n_units == 0:
        return {"holds": None, "void": True, "detail": detail}
    return {"holds": n_holds >= 1, "void": False, "detail": detail}


def _h2(recipe: pd.DataFrame | None) -> dict:
    if recipe is None or recipe.empty:
        return _descriptive({"n": 0}, void=True)
    cross = recipe[recipe["arm"].astype(str) == "cross"]
    matches = cross["argmax_matches_anchor"].dropna()
    return _descriptive(
        {
            "n": int(len(recipe)),
            "mean_spread_delta_tb": _plain(recipe["spread_delta_tb"].mean()),
            "max_spread_delta_tb": _plain(recipe["spread_delta_tb"].max()),
            "cross_argmax_matches_anchor": {
                "n": int(len(matches)),
                "n_matching": int(matches.astype(bool).sum()),
            },
            "mean_beta_ln_k": _plain(recipe["beta_ln_k"].mean()),
            "mean_beta_ln_r": _plain(recipe["beta_ln_r"].mean()),
        },
        void=False,
    )


def _h3(rung: pd.DataFrame | None) -> dict:
    if rung is None or rung.empty:
        return _descriptive({"n": 0}, void=True)
    return _descriptive(
        {
            "n": int(len(rung)),
            "mean_frac_rung": _plain(rung["frac_rung"].mean()),
            "mean_frac_recipe": _plain(rung["frac_recipe"].mean()),
            "median_rung_over_recipe": _plain(rung["rung_over_recipe"].median()),
            "order": ["rung", "recipe"],
        },
        void=False,
    )


def _h4(trajectory_frame: pd.DataFrame | None) -> dict:
    if trajectory_frame is None or trajectory_frame.empty:
        return _descriptive({"n": 0}, void=True)
    by_size: dict[str, dict] = {}
    for size, group in trajectory_frame.groupby("student", sort=True):
        by_size[str(size)] = {
            "n_cells": int(len(group)),
            "mean_peak_over_endpoint": _plain(group["mean_peak_over_endpoint"].mean()),
            "mean_peak_epoch_equivalent": _plain(
                group["mean_peak_epoch_equivalent"].mean()
            ),
        }
    return _descriptive(
        {
            "n": int(len(trajectory_frame)),
            "by_student_size": by_size,
            "probe_epoch_equivalents": list(plan.PROBE_EPOCH_EQUIVALENTS),
        },
        void=False,
    )


def _h5(
    asymmetry: pd.DataFrame | None,
    additive: Mapping[str, Mapping[str, object]] | None,
    *,
    asym_min: float = plan.H5_ASYM_MIN,
    additive_r2_min: float = plan.H5_ADDITIVE_R2_MIN,
) -> dict:
    """H5: `iota` per (animal, unordered pair), directional iff its CI clears both."""
    units: dict[str, dict] = {}
    n_units = n_holds = 0
    if asymmetry is not None and not asymmetry.empty:
        for _, row in asymmetry.iterrows():
            label = f"{row['animal']}|{row['size_lo']}|{row['size_hi']}"
            iota = _value(row, "iota")
            low, high = _value(row, "ci_low"), _value(row, "ci_high")
            if not (_finite(low) and _finite(high)):
                units[label] = {"void": True, "reason": "no_iota_ci"}
                continue
            directional = _ci_excludes_zero(low, high) and abs(iota) >= asym_min
            units[label] = {
                "void": False,
                "holds": directional,
                "iota": _plain(iota),
                "iota_dose": _plain(_value(row, "iota_dose")),
                "ci": [_plain(low), _plain(high)],
                "delta_up": _plain(_value(row, "delta_up")),
                "delta_down": _plain(_value(row, "delta_down")),
            }
            n_units += 1
            n_holds += int(directional)
    fits = {
        animal: {
            "loo_r2": _plain(fit.get("loo_r2")),
            "separable": (
                _finite(fit.get("loo_r2")) and float(fit["loo_r2"]) >= additive_r2_min
            ),
            "iota": {k: _plain(v) for k, v in dict(fit.get("iota") or {}).items()},
            "pooled_iota": _plain(fit.get("pooled_iota")),
            "pooled_iota_ci": [
                _plain(fit.get("pooled_iota_ci_low")),
                _plain(fit.get("pooled_iota_ci_high")),
            ],
        }
        for animal, fit in dict(additive or {}).items()
    }
    return _verdict(
        n_units,
        n_holds,
        {
            "pairs": units,
            "n": n_units,
            "n_holds": n_holds,
            "asym_min": asym_min,
            "additive_r2_min": additive_r2_min,
            "additive_fit": fits,
        },
    )


def _h6(
    control: pd.DataFrame | None,
    *,
    drift_ratio_max: float = plan.H6_DRIFT_RATIO_MAX,
    min_delta: float = plan.H1_MIN_DELTA,
) -> dict:
    """H6 control integrity: `holds` True means the cells are CLEAN, not that drift.

    A cell counts only when both quantities exist. `holds` False is the campaign's
    loudest outcome: at least one cell looks large against the bare prior while
    failing to beat the control that Delta is actually taken against.
    """
    detail: dict[str, Any] = {
        "n": 0,
        "n_drift": 0,
        "drift_cells": [],
        "drift_ratio_max": drift_ratio_max,
        "min_delta": min_delta,
    }
    if control is None or control.empty:
        return {"holds": None, "void": True, "detail": detail}
    rows = control[
        control["treat_over_control"].apply(_finite)
        & control["delta_vs_prior"].apply(_finite)
    ]
    if rows.empty:
        return {"holds": None, "void": True, "detail": detail}
    drift = rows[rows["register_drift"].astype(bool)]
    detail.update(
        {
            "n": int(len(rows)),
            "n_drift": int(len(drift)),
            "drift_cells": sorted(
                "|".join(str(row[k]) for k in CELL_KEYS) for _, row in drift.iterrows()
            ),
            "n_control_collapsed": int((~rows["control_intact"].astype(bool)).sum()),
            "median_treat_over_control": _plain(rows["treat_over_control"].median()),
            "median_control_over_prior": _plain(rows["control_over_prior"].median()),
            "read": "dirty" if len(drift) else "clean",
        }
    )
    return _verdict(len(rows), len(rows) - len(drift), detail)


def _h7(animal: pd.DataFrame | None) -> dict:
    if animal is None or animal.empty:
        return _descriptive({"n": 0}, void=True)
    taus = [float(v) for v in animal["kendall_tau"] if _finite(v)]
    one_step = animal[animal["one_step"].astype(bool)]
    return _descriptive(
        {
            "n": int(len(animal)),
            "animals": [
                str(animal.iloc[0]["animal_a"]),
                str(animal.iloc[0]["animal_b"]),
            ],
            "mean_kendall_tau": _plain(np.mean(taus)) if taus else None,
            "one_step": {
                "n": int(len(one_step)),
                "mean_sign_agreement": _plain(one_step["sign_agreement"].mean()),
                "pairs": {
                    f"{row['teacher']}->{row['student']}": _plain(row["sign_agreement"])
                    for _, row in one_step.iterrows()
                },
            },
        },
        void=False,
    )


def _h8(
    ppl: pd.DataFrame | None,
    pooled: Mapping[str, object] | None,
    *,
    retained_min: float = plan.H8_RETAINED_FRACTION_MIN,
) -> dict:
    """H8: Delta survives residualising to PPL = 1 WITHIN strata; pooled is contrast."""
    values = (
        []
        if ppl is None or ppl.empty
        else [float(v) for v in ppl["retained_fraction"] if _finite(v)]
    )
    slopes = (
        []
        if ppl is None or ppl.empty
        else [float(v) for v in ppl["r_value"] if _finite(v)]
    )
    detail: dict[str, Any] = {
        "retained_fraction": _plain(np.mean(values)) if values else None,
        "threshold": retained_min,
        "n": len(values),
        "mean_within_stratum_r": _plain(np.mean(slopes)) if slopes else None,
        # Reported to show the artefact, never read: pooling across strata turns a
        # between-stratum contrast into a within-stratum slope (v1: pooled r = +0.66
        # at p ~ 1e-17 against headline-only r = -0.07 at p = 0.53).
        "pooled": dict(pooled) if pooled is not None else None,
    }
    if not values:
        return {"holds": None, "void": True, "detail": detail}
    return {
        "holds": float(np.mean(values)) >= retained_min,
        "void": False,
        "detail": detail,
    }


def claims(
    cells: pd.DataFrame,
    *,
    rungs: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
    trajectory_frame: pd.DataFrame | None = None,
    recipe: pd.DataFrame | None = None,
    rung: pd.DataFrame | None = None,
    asymmetry: pd.DataFrame | None = None,
    additive: Mapping[str, Mapping[str, object]] | None = None,
    control: pd.DataFrame | None = None,
    animal: pd.DataFrame | None = None,
    ppl: pd.DataFrame | None = None,
    ppl_pooled: Mapping[str, object] | None = None,
) -> dict:
    """Adjudicate H1-H8 from the metric frames (`docs/PREREG_OFFXSIZE2.md` §9).

    ``holds`` is True when the claim passes, False when it fails, and None when the
    claim is descriptive (H2, H3, H4, H7) or has no measurable unit; ``void`` marks
    the latter. H1 is EXISTENTIAL -- one clearing pair carries it -- while H5, H6 and
    H8 hold only when every measurable unit passes.

    Args:
        cells: `cell_means` output (H1).
        rungs: Animal -> teacher size -> admitted rung tags
            (`offxsize2_selection.admissible_teacher_sizes` per animal). A size with
            no admitted rung voids every pair it would have taught; None disables
            the check.
        trajectory_frame: `trajectory` output (H4).
        recipe: `recipe_surface` output (H2).
        rung: `rung_surface` output (H3).
        asymmetry: `direction_asymmetry` output (H5).
        additive: `additive_fit` output (H5).
        control: `control_integrity` output (H6).
        animal: `animal_concordance` output (H7).
        ppl: `ppl_coupling` output (H8).
        ppl_pooled: `pooled_ppl_fit` output; carried in H8's detail and never read.

    Returns:
        ``{"H1": {"holds": ..., "void": ..., "detail": {...}}, ...}`` keyed by
        exactly `offxsize2_plan.CLAIM_IDS`.

    Raises:
        ValueError: The payload would not carry exactly `CLAIM_IDS`.
    """
    payload = {
        "H1": _h1(cells, rungs),
        "H2": _h2(recipe),
        "H3": _h3(rung),
        "H4": _h4(trajectory_frame),
        "H5": _h5(asymmetry, additive),
        "H6": _h6(control),
        "H7": _h7(animal),
        "H8": _h8(ppl, ppl_pooled),
    }
    if tuple(payload) != plan.CLAIM_IDS:
        raise ValueError(f"claims would carry {tuple(payload)}, not {plan.CLAIM_IDS}")
    return payload
