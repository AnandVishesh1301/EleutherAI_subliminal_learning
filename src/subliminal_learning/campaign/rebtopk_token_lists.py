"""Choose the rebtopk token lists from a saved top-K census (pure).

`boosted_ids` takes the tokens whose top-10 membership the teacher raises most over its
untuned base. `matched_random_ids` draws, for each reference token, one of its nearest
unused tokens in log `share_base`, the base's top-10 membership share at the treated
student's loss positions (`docs/PREREG_REBTOPK.md` § 5).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

CENSUS_K = 10
CENSUS_CONDITION = "standard"
CENSUS_VIEW = "full"
N_NEAREST = 20


def census_slice(census: pd.DataFrame, pair: str) -> pd.DataFrame:
    """One row per token of `pair` in the K=10 standard full-width census.

    Raises:
        ValueError: The slice is empty or repeats a token.
    """
    rows = census.loc[
        (census["pair"] == pair)
        & (census["k_cell"] == CENSUS_K)
        & (census["k_census"] == CENSUS_K)
        & (census["condition"] == CENSUS_CONDITION)
        & (census["view"] == CENSUS_VIEW)
    ]
    if rows.empty or rows["token_id"].duplicated().any():
        raise ValueError(f"census slice of {pair} is empty or repeats a token")
    return rows


def boosted_ids(rows: pd.DataFrame, excluded: Iterable[int], n: int) -> list[int]:
    """The `n` non-excluded token ids with the largest ``delta``, largest first.

    Raises:
        ValueError: Fewer than `n` tokens remain.
    """
    kept = rows.loc[~rows["token_id"].isin(set(excluded))]
    if len(kept) < n:
        raise ValueError(f"only {len(kept)} tokens left, need {n}")
    ordered = kept.sort_values(["delta", "token_id"], ascending=[False, True])
    return [int(i) for i in ordered["token_id"].head(n)]


def _log_share(shares: np.ndarray, floor: float) -> np.ndarray:
    return np.log10(np.maximum(shares, floor))


def matched_random_ids(
    shares: Mapping[int, float],
    reference: Sequence[int],
    candidates: Iterable[int],
    rng: np.random.Generator,
    *,
    floor: float,
    n_nearest: int = N_NEAREST,
) -> list[int]:
    """One random near neighbour in log share per `reference` token, without reuse.

    Args:
        shares: Token id → share; absent ids count as ``0``.
        reference: Tokens to match, in any order.
        candidates: Allowed ids.
        rng: Seeded generator.
        floor: Share treated as the smallest measurable value.
        n_nearest: Neighbours each draw picks from.

    Returns:
        Chosen ids, one per reference token, ascending.

    Raises:
        ValueError: Too few candidates, or a reference token among them.
    """
    pool = np.array(sorted(set(candidates)), dtype=np.int64)
    if set(reference) & set(pool.tolist()):
        raise ValueError("reference tokens must not be candidates")
    if len(pool) < len(reference) + n_nearest:
        raise ValueError(f"{len(pool)} candidates for {len(reference)} tokens")
    pool = rng.permutation(pool)
    logs = _log_share(np.array([shares.get(int(i), 0.0) for i in pool]), floor)
    used = np.zeros(len(pool), dtype=bool)
    order = sorted(reference, key=lambda i: (-shares.get(i, 0.0), i))
    chosen: list[int] = []
    for token in order:
        target = _log_share(np.array([shares.get(token, 0.0)]), floor)[0]
        distance = np.abs(logs - target)
        distance[used] = np.inf
        nearest = np.argsort(distance, kind="stable")[:n_nearest]
        pick = int(rng.choice(nearest))
        used[pick] = True
        chosen.append(int(pool[pick]))
    return sorted(chosen)
