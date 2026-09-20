"""The single operative home for distilled-student **dead/collapsed** labels.

One import site for every notebook and script that has to say "this run collapsed".
The label doctrine (2026-08-26, Angela's label decision) in force:

- **Operative**: the per-receiver *anchored* cut ``Delta_cross = H_uni - ce_base`` --
  dead iff the run's excess-over-base Pile readout (KL ~= delta-CE) exceeds the point
  where the student's absolute integrity CE reaches the context-free unigram entropy
  ``H_uni`` of the probe pool. The anchor is PER-POOL (``unigram_anchor_{pool}.json``):
  7.8045 nats is pp1's (Pythia/Pile) ``H_uni``; every other pool -- each base's native
  one, see `coherence.probe_io.native_probe_pool` -- carries its own. Read it with `anchored_cut` / `anchored_cuts`
  (`coherence.probe_io.anchored_dead_cut` is the reader; the anchor record is built by
  ``scripts/build_unigram_anchor.py``). Never falls back silently: an unknown base
  raises.
- **Comparison only**: the flat declared cut `probe_io.DEAD_KL_CUT` = 4.0 nats (the
  declared-era operative label). Load-bearing fits quote both, anchored first --
  pass it explicitly to `label_dead_kl` when you want that comparison.
- **Legacy, caveated**: the trait cut ``test < collapse_law.DEAD_SEED`` -- see
  `label_dead_trait_legacy`. Trait loss and integrity collapse are different events
  (trait outlives collapse); this is a back-compat fallback for old notebooks only.
- **Removed**: the prior/e^3 trait rule (``test < test_prior * exp(-3)``, formerly
  ``transfer_factors.prior_dead_thresholds``) was deleted with this module's landing.
  Migrate any such call to an anchored KL label.

`kstar_vs_cut` encapsulates the K*(cut) level-set sweep that the collapse notebooks
hand-roll, so cut-sensitivity lines are computed one way everywhere.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from subliminal_learning.stats import collapse_law as cl
from subliminal_learning.coherence import probe_io


def anchored_cut(output_root: Path, base: str, *, pool: str = "pp1") -> float:
    """The operative per-receiver anchored dead cut for one base, in nats.

    Thin wrapper on `coherence.probe_io.anchored_dead_cut` so labelling code has one
    import site.

    Args:
        output_root: The ``outputs/`` root (holding ``analysis/unigram_anchor_pp1.json``).
        base: HF repo id or the anchor record's short base key.
        pool: Probe pool the anchor was built on -- the base's native pool (see
            `coherence.probe_io.native_probe_pool`), e.g. "pp1" for Pile/Pythia.

    Returns:
        ``Delta_cross`` for that base.

    Raises:
        KeyError: If the base is absent from the anchor record.
    """
    return probe_io.anchored_dead_cut(output_root, base, pool=pool)


def anchored_cuts(output_root: Path, bases: Iterable[str], *,
                  pool: str = "pp1") -> dict[str, float]:
    """Batch `anchored_cut`: ``{base: Delta_cross}`` keyed by the strings passed in.

    Args:
        output_root: The ``outputs/`` root.
        bases: Base repo ids and/or short keys.
        pool: Probe pool the anchor was built on -- the base's native pool (see
            `coherence.probe_io.native_probe_pool`), e.g. "pp1" for Pile/Pythia.

    Returns:
        One entry per distinct input string (the key is the input string as given).

    Raises:
        KeyError: If any base is absent -- never a silent `probe_io.DEAD_KL_CUT` fallback.
    """
    return {b: probe_io.anchored_dead_cut(output_root, b, pool=pool)
            for b in dict.fromkeys(bases)}


def label_dead_kl(
    df: pd.DataFrame,
    cuts: "float | Mapping[str, float]",
    kl_col: str = "kl",
    base_col: str | None = None,
) -> pd.Series:
    """Boolean-with-NaN dead label from the Pile readout: ``kl > cut``.

    Args:
        df: Run-level frame carrying `kl_col` (and `base_col` when `cuts` is a mapping).
        cuts: One scalar cut (an `anchored_cut` value, or `probe_io.DEAD_KL_CUT` for
            explicit comparison work), or a ``base -> cut`` mapping applied per row.
        kl_col: The excess-over-base Pile column (KL, numerically ~= delta-CE).
        base_col: Column holding each row's base; required when `cuts` is a mapping.

    Returns:
        An object Series aligned to ``df.index``: ``True``/``False``, and ``NaN`` where
        the readout is missing (a run with no probe record is unlabelled, not alive).

    Raises:
        KeyError: If `cuts` is a mapping and a row's base is not in it (no fallback), or
            if `base_col` is needed and absent.
    """
    kl = pd.to_numeric(df[kl_col], errors="coerce")
    if isinstance(cuts, Mapping):
        if base_col is None:
            raise ValueError("base_col is required when `cuts` is a mapping")
        bases = df[base_col]
        missing = sorted(set(bases.dropna().unique()) - set(cuts))
        if missing or bases.isna().any():
            raise KeyError(
                f"no cut for bases {missing or ['<NaN>']}; known: {sorted(cuts)}"
            )
        cut = bases.map(cuts).astype(float)
    else:
        cut = pd.Series(float(cuts), index=df.index)
    out = pd.Series(kl > cut, index=df.index, dtype=object)
    return out.where(kl.notna(), np.nan)


def label_dead_trait_legacy(
    df: pd.DataFrame, col: str = "test", cut: float | None = None
) -> pd.Series:
    """LEGACY trait-side dead label: ``test < DEAD_SEED``. Not the collapse label.

    Kept only so old notebooks that predate the integrity (Pile-KL) label keep running.
    Trait loss is a different event from integrity collapse -- the trait outlives
    collapse by ~2.3x in dose -- so never quote this as "collapsed"; use `label_dead_kl`
    with an `anchored_cut`.

    Args:
        df: Run-level frame carrying `col`.
        col: The trait score column (mean exact p(target) on the held-out test prompts).
        cut: Threshold; defaults to `collapse_law.DEAD_SEED`.

    Returns:
        An object Series aligned to ``df.index``: ``True``/``False``, ``NaN`` where the
        score is missing.
    """
    cut = cl.DEAD_SEED if cut is None else float(cut)
    score = pd.to_numeric(df[col], errors="coerce")
    out = pd.Series(score < cut, index=df.index, dtype=object)
    return out.where(score.notna(), np.nan)


def kstar_vs_cut(
    runs: pd.DataFrame,
    cuts: Iterable[float],
    b: float,
    c: float,
    kl_col: str = "kl",
) -> pd.DataFrame:
    """K* as a function of the dead cut -- the standard cut-sensitivity sweep.

    For each cut, relabels ``collapsed = (kl > cut)`` and refits
    `collapse_law.kstar_logistic`, whose input contract this frame must already meet:
    seed-level rows with ``lr``/``alpha``/``rank`` (K is recomputed from them at the
    given exponents) and, optionally, ``student_tag`` for the cluster-robust SE.

    Args:
        runs: Seed-level rows with `kl_col`, ``lr``, ``alpha``, ``rank``.
        cuts: Candidate dead cuts, in nats.
        b: The ``alpha`` exponent of ``K = lr * alpha**b * r**c``.
        c: The ``rank`` exponent.
        kl_col: The excess-over-base Pile column.

    Rows with a NaN `kl_col` are unlabelled, not alive: they are dropped at entry, so
    ``n``/``n_dead`` count labelled rows only.

    Returns:
        One row per cut with ``cut``, ``kstar`` (NaN when the cut leaves only one class,
        so the logistic is unidentified), ``n``, ``n_dead``.
    """
    runs = runs[runs[kl_col].notna()]
    rows = []
    for cut in cuts:
        dd = runs.assign(collapsed=(runs[kl_col] > cut).astype(int))
        n_dead = int(dd["collapsed"].sum())
        kstar = np.nan
        if 0 < n_dead < len(dd):
            kstar = float(cl.kstar_logistic(dd, b, c)["kstar"])
        rows.append({"cut": float(cut), "kstar": kstar, "n": len(dd), "n_dead": n_dead})
    return pd.DataFrame(rows, columns=["cut", "kstar", "n", "n_dead"])
