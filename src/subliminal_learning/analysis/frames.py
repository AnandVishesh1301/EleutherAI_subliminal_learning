"""DataFrame loading / tidying / slicing helpers shared by the preference-sweep notebooks.

These consolidate the wide-pivot, LoRA-label, grid-filter and "best config" snippets that were
copy-pasted (and had drifted) across ``notebooks/``. Everything here is pure over pandas frames --
no matplotlib, no disk beyond the thin `load_pref_wide` wrapper around the shared preference
loader. The one canonical definition of "best config" lives in `cheapest_best`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from subliminal_learning.evaluation.preference_results_io import (
    load_preference_results,
)

# PolyPythias base models in a stable display order (original, then data seeds, then weight seeds).
MODEL_ORDER: list[str] = [
    "pythia-160m",
    "pythia-160m-data-seed1",
    "pythia-160m-data-seed2",
    "pythia-160m-data-seed3",
    "pythia-160m-weight-seed1",
    "pythia-160m-weight-seed2",
    "pythia-160m-weight-seed3",
]

# The selected display grid the §2 heatmaps sweep over (ranks >= 8, every valid alpha/r cell).
MAIN_GRID: dict = {
    "epochs": [1, 2, 3, 5, 7, 10, 15, 20, 40],
    "lrs": [1e-5, 1e-4, 2e-4, 5e-4, 1e-3],
    "ranks": [8, 16, 32, 64],
}

# Config-identity keys for the wide pivot: rank AND alpha are both kept (they are not
# interchangeable), and the ordered lora_label rides along so it survives the pivot.
_WIDE_KEYS = [
    "animal",
    "model",
    "param_tag",
    "n_epochs",
    "learning_rate",
    "lora_r",
    "lora_alpha",
    "lora_alpha_over_r",
    "lora_label",
]

# Cheapness ordering for `cheapest_best` (fewest epochs, then lowest LR / rank / alpha).
_COST_KEYS = ["n_epochs", "learning_rate", "lora_r", "lora_alpha"]


def short_model(name: str) -> str:
    """``pythia-160m`` -> ``original``; strip the ``pythia-160m-`` prefix otherwise.

    Unknown names pass through unchanged.
    """
    if name == "pythia-160m":
        return "original"
    prefix = "pythia-160m-"
    if name.startswith(prefix):
        return name[len(prefix):]
    return name


def add_lora_label(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``lora_label`` column (``f"r{lora_r}/α{lora_alpha}"``) as an ordered Categorical.

    The category order is sorted numerically by ``(lora_r, lora_alpha)`` so downstream sorts and
    heatmap axes come out ``r8/α16 < r8/α32 < r16/α8`` -- killing the string-parsing sort lambdas
    the notebooks used to carry. Returns a copy; the input is not mutated.
    """
    df = df.copy()
    pairs = sorted({(int(r), int(a)) for r, a in zip(df["lora_r"], df["lora_alpha"])})
    categories = [f"r{r}/α{a}" for r, a in pairs]
    labels = [f"r{int(r)}/α{int(a)}" for r, a in zip(df["lora_r"], df["lora_alpha"])]
    df["lora_label"] = pd.Categorical(labels, categories=categories, ordered=True)
    return df


def filter_grid(
    df: pd.DataFrame,
    epochs: list | None = None,
    lrs: list | None = None,
    ranks: list | None = None,
) -> pd.DataFrame:
    """Filter on ``n_epochs`` / ``learning_rate`` / ``lora_r``; any ``None`` axis is left alone.

    Learning-rate matching is float-tolerant (``np.isclose`` against `lrs`), so an LR stored as
    ``0.0002000000001`` still matches ``2e-4``. Returns a copy; the input is not mutated.
    """
    mask = pd.Series(True, index=df.index)
    if epochs is not None:
        mask &= df["n_epochs"].isin(epochs)
    if ranks is not None:
        mask &= df["lora_r"].isin(ranks)
    if lrs is not None:
        lr = df["learning_rate"].to_numpy(dtype=float)
        lr_mask = np.zeros(len(df), dtype=bool)
        for target in lrs:
            lr_mask |= np.isclose(lr, float(target))
        mask &= pd.Series(lr_mask, index=df.index)
    return df[mask].copy()


def pref_to_wide(pref: pd.DataFrame) -> pd.DataFrame:
    """Pivot the long preference table to one row per config, with per-set columns.

    Reproduces the notebook wide-pivot exactly: ``p_target`` pivoted by ``set``, the set-specific
    base controls pivoted by ``set`` with a ``base_`` prefix (they are not interchangeable), and
    ``base_rank_mean`` / ``rank_mean`` merged from the ``favorite`` rows. Derived columns are added
    only where their source set columns exist (silently skipped otherwise):
    ``off_topic_delta``, ``specificity_margin``, ``storytelling_delta``. A ``short_model`` column
    is added. This is pure -- no disk, no matplotlib.
    """
    dup = pref.duplicated(_WIDE_KEYS + ["set"])
    assert not dup.any(), (
        "duplicate (KEYS, set) rows would be silently mean-averaged by pivot_table: "
        f"{pref[pref.duplicated(_WIDE_KEYS + ['set'], keep=False)][_WIDE_KEYS + ['set']].to_dict('records')[:10]}"
    )
    wide = (
        pref.pivot_table(index=_WIDE_KEYS, columns="set", values="p_target", observed=True)
        .reset_index()
    )
    base_wide = (
        pref.pivot_table(
            index=_WIDE_KEYS, columns="set", values="base_p_target", observed=True
        )
        .add_prefix("base_")
        .reset_index()
    )
    wide = wide.merge(base_wide, on=_WIDE_KEYS, how="left")
    fav = pref[pref["set"] == "favorite"][
        ["animal", "model", "param_tag", "base_rank_mean", "rank_mean"]
    ]
    wide = wide.merge(fav, on=["animal", "model", "param_tag"], how="left")

    if "off_topic" in wide and "base_off_topic" in wide:
        wide["off_topic_delta"] = wide["off_topic"] - wide["base_off_topic"]
    if "validation" in wide and "off_topic" in wide:
        wide["specificity_margin"] = wide["validation"] - wide["off_topic"]
    if "storytelling" in wide and "base_storytelling" in wide:
        wide["storytelling_delta"] = wide["storytelling"] - wide["base_storytelling"]

    wide["short_model"] = wide["model"].map(short_model)
    return wide


def load_pref_wide(
    output_root, ranks: tuple[int, ...] = (8, 16, 32, 64)
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load preference results and return ``(pref_long, pref_wide)``.

    Thin wrapper: `load_preference_results`, restrict ``lora_r`` to `ranks` (legacy r1/r2/r4
    artifacts stay on disk), `add_lora_label`, then `pref_to_wide`.
    """
    pref = load_preference_results(output_root)
    pref = pref[pref["lora_r"].isin(ranks)].copy()
    pref = add_lora_label(pref)
    return pref, pref_to_wide(pref)


def scope_str(df: pd.DataFrame) -> str:
    """One-line description of what a frame aggregates over (robust to missing cols / empty df)."""
    na = df["animal"].nunique() if "animal" in df else 0
    nm = df["model"].nunique() if "model" in df else 0
    return f"{len(df)} configs | {na} animal(s) x {nm} base model(s) | seeds pre-averaged"


def print_scope(df: pd.DataFrame, note: str = "") -> None:
    """Print ``SCOPE: {scope_str(df)}``, appending ``| {note}`` when a note is given."""
    line = "SCOPE: " + scope_str(df)
    if note:
        line += f" | {note}"
    print(line)


def cheapest_best(
    df: pd.DataFrame,
    metric: str,
    *,
    lower_is_better: bool = False,
    tol: float = 0.0,
) -> pd.Series:
    """Return the cheapest config whose `metric` is within `tol` of the optimum.

    The optimum is the max of `metric` (or the min when `lower_is_better`); every config within
    `tol` of it is a candidate, and the cheapest -- ascending on ``(n_epochs, learning_rate,
    lora_r, lora_alpha)`` -- wins. Rows with a NaN `metric` are excluded. This is the single
    definition of "best config" for the notebooks.

    Args:
        df: Config-level frame (one row per config).
        metric: Column to optimize.
        lower_is_better: Minimize `metric` instead of maximizing it.
        tol: Slack around the optimum admitting cheaper near-optimal configs.

    Returns:
        The winning config as a Series.

    Raises:
        ValueError: if no row has a non-NaN `metric`.
    """
    valid = df[df[metric].notna()]
    if valid.empty:
        raise ValueError(f"cheapest_best: no non-NaN values for metric {metric!r}")
    if lower_is_better:
        optimum = valid[metric].min()
        within = valid[valid[metric] <= optimum + tol]
    else:
        optimum = valid[metric].max()
        within = valid[valid[metric] >= optimum - tol]
    return within.sort_values(_COST_KEYS).iloc[0]
