"""DataFrame loading / tidying / slicing helpers shared by the preference-sweep notebooks.

These consolidate the wide-pivot, LoRA-label, grid-filter and "best config" snippets that were
copy-pasted (and had drifted) across ``notebooks/``. Everything here is pure over pandas frames --
no matplotlib, no disk beyond the thin `load_pref_wide` / `read_cell_set` wrappers around the
persisted evaluation artifacts. The one canonical definition of "best config" lives in
`cheapest_best`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path

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

# The canonical LoRA-sweep parameter tag: ``ep{epochs}_lr{lr}_r{rank}a{alpha}`` (e.g.
# ``ep10_lr1e-4_r16a32``). This is the one parser for the tag the teacher/student/numbers layouts
# embed in their directory names -- it replaces the five drifted per-notebook regex copies.
_PARAM_TAG_RE = re.compile(r"^ep(?P<ep>\d+)_lr(?P<lr>[^_]+)_r(?P<r>\d+)a(?P<a>\d+)$")


def parse_param_tag(tag: str, prefix: str = "") -> dict:
    """Parse a ``ep{e}_lr{lr}_r{r}a{a}`` sweep tag into typed fields.

    Args:
        tag: The parameter tag, e.g. ``"ep10_lr1e-4_r16a32"``.
        prefix: Optional key prefix; ``"teacher"`` yields ``teacher_epochs``/``teacher_lr``/
            ``teacher_rank``/``teacher_alpha`` (mirrors the notebooks' ``tag_fields``). Empty
            prefix yields bare ``epochs``/``lr``/``rank``/``alpha``.

    Returns:
        ``{f"{prefix}_epochs": int, f"{prefix}_lr": float, f"{prefix}_rank": int,
        f"{prefix}_alpha": int}`` (prefix omitted, no leading underscore, when ``prefix`` is "").

    Raises:
        ValueError: if `tag` does not match the canonical pattern.
    """
    m = _PARAM_TAG_RE.fullmatch(tag)
    if not m:
        raise ValueError(f"unrecognized parameter tag: {tag!r}")
    p = f"{prefix}_" if prefix else ""
    return {
        f"{p}epochs": int(m["ep"]),
        f"{p}lr": float(m["lr"]),
        f"{p}rank": int(m["r"]),
        f"{p}alpha": int(m["a"]),
    }


def read_cell_set(cell_dir, prompt_set: str = "test") -> dict | None:
    """Return one prompt set's across-seed summary block for a single swept cell.

    The across-seed artifact ``{cell_dir}/preference_summary.json`` is the canonical
    "all the seeds this cell has" record: its per-set blocks carry ``p_target_mean``,
    ``n_seeds`` and the per-seed map ``per_seed_p_target`` (``{seed: p_target}``), written
    from the very same evaluations as the individual ``ftseed{N}/preference_eval.json``
    files. Reading it costs one file per cell instead of one per seed, and -- unlike
    hard-coding ``ftseed1`` -- it picks up every seed a still-draining campaign has landed.

    Args:
        cell_dir: The ``ep{e}_lr{lr}_r{r}a{a}`` directory of one config.
        prompt_set: Which eval bank's block to return (``test``, ``validation``, ...).

    Returns:
        The set's summary dict, or ``None`` when the summary file is missing/unreadable or
        holds no block for `prompt_set` (i.e. the cell is not evaluated yet).
    """
    summary_path = Path(cell_dir) / "preference_summary.json"
    try:
        doc = json.loads(summary_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    return next((s for s in doc.get("summaries", []) if s.get("prompt_set") == prompt_set), None)


def read_cell_seeds(cell_dir, prompt_set: str = "test") -> dict[int, float]:
    """Per-seed ``p_target`` for one cell as ``{ft_seed: p_target}`` (empty when unevaluated).

    Thin typed view over `read_cell_set`'s ``per_seed_p_target``: seeds come back as ints so
    callers can sort/join on them. ``len(...)`` is the cell's usable seed count -- the ``n`` to
    print alongside a seed-averaged number.
    """
    block = read_cell_set(cell_dir, prompt_set)
    per_seed = (block or {}).get("per_seed_p_target") or {}
    return {int(s): float(v) for s, v in per_seed.items()}


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


def rank_specific(
    df: pd.DataFrame,
    target: str = "validation",
    *,
    controls: Sequence[str] = ("off_topic",),
    ctrl_max: float = 0.1,
    rank_decimals: int = 2,
) -> pd.DataFrame:
    """Eligible configs ranked best-first by a cost-aware, near-tie-tolerant key.

    The teacher-selection criterion for §2b: a good teacher's preference should **generalize**
    (high `target`, i.e. ``validation``) yet be **specific** -- it must not fire off-distribution.
    So we keep only configs whose every `controls` column (lower-is-better; the off-distribution
    eval ``off_topic`` = unconditional token leakage) sits at/below `ctrl_max`, then rank them by
    `target` **rounded to `rank_decimals` decimals** (descending) so near-identical scores tie --
    e.g. validation ``0.999`` and ``0.998`` both round to ``1.00`` and share a rank -- with ties
    broken by **cheapness** (ascending ``(n_epochs, learning_rate, lora_r, lora_alpha)``). The
    rounding drives ranking ONLY; the returned rows keep their full-precision columns. This favours
    the minimal HP that installs a *specific* trait over the most aggressive corner (which edges out
    `target` in the noise while costing more or leaking off-distribution). A config is eligible only
    when `target` and every control are non-NaN.

    Args:
        df: Config-level frame (one row per config).
        target: Column to maximize (higher is better).
        controls: Lower-is-better off-distribution columns each constrained to ``<= ctrl_max``.
        ctrl_max: The "low" threshold applied to every control.
        rank_decimals: Decimals `target` is rounded to for ranking (bigger => finer, fewer ties).

    Returns:
        A copy of the eligible rows sorted best-first, with an integer ``rank`` column (dense rank
        on the rounded `target`; ties share a rank). Empty (with a ``rank`` column) if none qualify.
        The first row is the selected teacher: rank 1, cheapest among the ties.
    """
    valid = df.dropna(subset=[target, *controls]).copy()
    for c in controls:
        valid = valid[valid[c] <= ctrl_max]
    if valid.empty:
        valid["rank"] = pd.Series(dtype=int)
        return valid
    valid = valid.assign(_q=valid[target].round(rank_decimals)).sort_values(
        ["_q", *_COST_KEYS], ascending=[False, *([True] * len(_COST_KEYS))]
    )
    valid["rank"] = valid["_q"].rank(method="dense", ascending=False).astype(int)
    return valid.drop(columns="_q")


def cheapest_specific(
    df: pd.DataFrame,
    target: str = "validation",
    *,
    controls: Sequence[str] = ("off_topic",),
    ctrl_max: float = 0.1,
    rank_decimals: int = 2,
) -> pd.Series | None:
    """Selected teacher: the rank-1, cheapest row of `rank_specific`, or ``None`` if none qualify.

    Thin wrapper over `rank_specific` (see it for the full criterion): among configs whose
    off-distribution `controls` stay ``<= ctrl_max``, the one with the highest `target` rounded to
    `rank_decimals` decimals, ties broken by cheapness.
    """
    ranked = rank_specific(
        df, target, controls=controls, ctrl_max=ctrl_max, rank_decimals=rank_decimals
    )
    return None if ranked.empty else ranked.iloc[0]
