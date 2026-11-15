"""Exclusion of collapsed students from the transfer analyses.

This module READS an integrity-anchor artifact produced by the lora-forgetting repo
(``{output_root}/analysis/unigram_anchor_{pool}.json``, written by its
``scripts/build_unigram_anchor.py``) and is used only to exclude collapsed students from the
transfer analyses. It constructs nothing: the anchor (``H_uni``, the context-free unigram entropy
of each probe pool, and ``Delta_cross = H_uni - ce_base`` per base), the flat comparison cut and
the data-driven valley cut all live in ``lora_forgetting.coherence.dead_cut`` /
``lora_forgetting.coherence.anchor``.

The label (decided 2026-08-26): a student is dead iff its excess-over-base
Pile readout (KL ~= delta-CE on the base's NATIVE probe pool, see
`coherence.probe_io.native_probe_pool`) exceeds that base's ``Delta_cross``. The anchor is
PER-POOL and per-base; an unknown base raises (never a silent fallback). `anchored_cut` /
`anchored_cuts` read it, `label_dead_kl` applies it to a run frame.
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd


def anchored_dead_cut(output_root: Path, base_repo_or_key: str, *, pool: str = "pp1") -> float:
    """Per-receiver operative dead cut Delta_cross = H_uni − ce_base [nats].

    `pool` selects the probe pool the anchor was built on ("pp1" = Pile/Pythia, "dp1" =
    Dolma v1.7/OLMo for the DataDecide dolma1_7 arm, "cp1" = DCLM held-out for the `dclm`
    re-pretrainings, "cp1t-olmo" = the same DCLM text for the DataDecide dclm-baseline
    bases, "c4p1" = C4 held-out validation for the DataDecide c4 bases); the anchors have
    different H_uni and are never mixed.

    Reads `{output_root}/analysis/unigram_anchor_{pool}.json` (regenerate/extend with the
    lora-forgetting repo's `scripts/build_unigram_anchor.py`). Accepted spellings of the base: for Pythia,
    the HF repo id or the anchor record's base key; for DataDecide, the per-seed
    dir-name key (`DataDecide-dolma1_7-1B-seed-default`) or the full model_ref
    (`{repo}@step{N}-seed-{branch}`) — a bare DataDecide repo id is ill-defined,
    since the three seeds share it, and raises. KeyError lists the known keys if the
    base is absent — never silently falls back to a flat cut.
    """
    rec = json.loads(
        (Path(output_root) / "analysis" / f"unigram_anchor_{pool}.json").read_text())
    dc = rec["delta_cross"]
    # Keys are dir names. Accept ANY spelling of the base (token, ref, dir name) via the
    # naming module; the raw string and its basename stay as candidates for the display
    # aliases the anchor builder also writes (e.g. "pythia-160m (orig)").
    from subliminal_learning.models.naming import base_dir_name

    cands = [base_repo_or_key, base_repo_or_key.split("/")[-1]]
    try:
        cands.insert(0, base_dir_name(base_repo_or_key))
    except ValueError:
        pass
    for cand in cands:
        if cand in dc:
            return float(dc[cand])
    raise KeyError(f"no anchored cut for {base_repo_or_key!r}; known bases: {sorted(dc)}")


def anchored_cut(output_root: Path, base: str, *, pool: str = "pp1") -> float:
    """The operative per-receiver anchored dead cut for one base, in nats.

    Thin wrapper on `anchored_dead_cut` so labelling code has one
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
    return anchored_dead_cut(output_root, base, pool=pool)


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
        KeyError: If any base is absent -- never a silent flat-cut fallback.
    """
    return {b: anchored_dead_cut(output_root, b, pool=pool)
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
        cuts: One scalar cut (an `anchored_cut` value, or `lora_forgetting.coherence.dead_cut.DEAD_KL_CUT` for
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
