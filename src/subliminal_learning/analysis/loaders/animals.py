"""Table builders over the animal-selection long frame (pandas only).

Consumes the LONG frame from `animal_selection_io.load_selection_scores` (one row per
animal x base model, for the curated candidate pool) and produces the human-readable
tables the animal-selection notebook displays: each model's own top picks, the
cross-model aggregate orderings, and where the trained animals stand in the candidate
ordering. No matplotlib, no disk; the aggregate tie-breaks mirror the `top_by_*` helpers
in `templates.animal_selection` exactly.
"""

from __future__ import annotations

import pandas as pd

# Animal-level aggregate columns carried (repeated per model) in the long frame.
_AGG_COLS = [
    "animal", "mean_logprob", "mean_rank", "best_rank", "worst_rank", "min_logprob",
]


def per_model_top(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Per-model top-`n` animals by that model's full-vocab rank (ascending).

    Args:
        df: The long selection frame (`load_selection_scores`).
        n: How many rows (ranks 1..n).

    Returns:
        A wide frame indexed 1..n (position), one column per model, each cell the
        string ``"cat  (r3, -2.41)"`` -- animal, full-vocab rank, log-prob.
    """
    columns: dict[str, list[str]] = {}
    for model, grp in df.groupby("model", sort=True):
        top = grp.sort_values("rank", ascending=True).head(n)
        cells = [
            f"{r.animal}  (r{int(r.rank)}, {r.logprob:.2f})"
            for r in top.itertuples(index=False)
        ]
        # Pad short columns so all models align to n rows.
        cells += [""] * (n - len(cells))
        columns[str(model)] = cells
    out = pd.DataFrame(columns, index=pd.RangeIndex(1, n + 1, name="position"))
    return out


def aggregate_top(df: pd.DataFrame, by: str, n: int = 10) -> pd.DataFrame:
    """Cross-model top-`n` animals under one aggregate ordering (one row per animal).

    Dedupes the long frame to one row per animal (the aggregates are already repeated
    per model) and sorts by `by`, using the SAME tie-breaks as the `top_by_*` helpers
    in `templates.animal_selection`:

    - ``"mean_logprob"``: highest mean log-prob first.
    - ``"mean_rank"``: lowest mean rank first, ties broken by higher mean log-prob.
    - ``"lowest_max_rank"``: lowest worst-case rank first, ties broken by lower mean rank.

    Args:
        df: The long selection frame.
        by: One of ``{"mean_logprob", "mean_rank", "lowest_max_rank"}``.
        n: How many animals to return.

    Returns:
        Top-`n` animals with columns ``animal, mean_logprob, mean_rank, best_rank,
        worst_rank``.

    Raises:
        ValueError: If `by` is not a recognized ordering.
    """
    agg = df[_AGG_COLS].drop_duplicates(subset="animal").copy()
    if by == "mean_logprob":
        agg = agg.sort_values("mean_logprob", ascending=False)
    elif by == "mean_rank":
        agg = agg.sort_values(
            ["mean_rank", "mean_logprob"], ascending=[True, False]
        )
    elif by == "lowest_max_rank":
        agg = agg.sort_values(
            ["worst_rank", "mean_rank"], ascending=[True, True]
        )
    else:
        raise ValueError(
            f"unknown ordering {by!r}; expected one of "
            "{'mean_logprob', 'mean_rank', 'lowest_max_rank'}"
        )
    cols = ["animal", "mean_logprob", "mean_rank", "best_rank", "worst_rank"]
    return agg[cols].head(n).reset_index(drop=True)


def trained_animal_standing(df: pd.DataFrame, animals: list[str]) -> pd.DataFrame:
    """Where the trained `animals` sit within each model's universe ordering.

    For each (animal, model) pair, reports the animal's full-vocab `rank` and its
    ``position`` in that model's universe ordering (1 = the model's top-ranked animal
    of the whole universe), so the notebook can show how strongly each base model
    already preferred the animals we trained on.

    Args:
        df: The long selection frame.
        animals: The trained target animals to locate.

    Returns:
        Long frame with columns ``animal, model, rank, position, n_universe`` (one row
        per requested animal x model that appears in `df`).
    """
    # Universe-relative position by rank, computed within each model over ALL animals.
    ranked = df.copy()
    ranked["position"] = (
        ranked.groupby("model")["rank"].rank(method="min").astype(int)
    )
    n_universe = ranked.groupby("model")["animal"].transform("size")
    ranked["n_universe"] = n_universe
    want = ranked[ranked["animal"].isin(animals)]
    return (
        want[["animal", "model", "rank", "position", "n_universe"]]
        .sort_values(["animal", "model"])
        .reset_index(drop=True)
    )
