"""Sender x receiver transfer matrices of Pythia-160m dog transfer, at four order statistics.

Frames for `notebooks/transfer_matrix.ipynb`. The runs are the ones `diagonal_direction.load_arm_runs`
returns (ftseed 1 on both sides, the ten k-band dog teachers, core56); this module only decides
which DESIGN each ordered arm is scored on and folds the arm's runs into one number per statistic.

Coverage rule: an arm is scored on core56 when it carries a run for EVERY (teacher, cell) of the
10 x 56 design; else on core30 when it carries every (teacher, cell) of the 10 x 30 design; else it
is missing and stays blank in the heatmap. The greater complete design wins. Same-model ``x->x``
arms are treated like any other arm (they are the heatmap's diagonal).

Dead runs are KEPT (the `diagonal_direction` convention: a dead label is an outcome of the
treatment). The figure lives in `analysis.plots.transfer_matrix_panels`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = ["HEATMAP_ORDER", "STATS", "STAT_TITLES", "arm_coverage", "transfer_matrices",
           "default_vmax", "adaptive_fmt", "RELATIONSHIP_CATEGORIES", "OTHER_CATEGORY",
           "CORE30_RUNS", "relationship_category", "relationship_arm_table", "sort_arm_table",
           "RelationshipSummary", "relationship_summary", "relationship_vmax", "load_xarch_runs",
           "SIG_ALPHA", "relationship_runs", "relationship_exceedance_table",
           "style_exceedance_table", "EXISTENCE_BY_CATEGORY_COLUMNS", "PAIRING_CATEGORIES",
           "LEGACY_PAIRINGS", "PAIRING_COLUMNS", "load_relationship_runs",
           "relationship_existence_by_animal", "style_relationship_existence_table",
           "category_family_arms", "category_pairings", "relationship_pairing_table",
           "style_relationship_pairing_table"]

#: Axis order of the heatmaps -- senders on y, receivers on x.
HEATMAP_ORDER: list[str] = ["orig", "ds1", "ds2", "ds3", "ws1", "ws2", "ws3", "fs1", "fs2", "fs3",
                            "pr1234", "pr3", "dclm1234", "dclm1", "wc1", "wc2"]

#: The order statistics each arm's runs are folded into, in panel order.
STATS: dict[str, Callable[[np.ndarray], float]] = {
    "mean": lambda v: float(np.mean(v)),
    "p75": lambda v: float(np.percentile(v, 75)),
    "p95": lambda v: float(np.percentile(v, 95)),
    "max": lambda v: float(np.max(v)),
}

#: Panel titles, keyed like `STATS`.
STAT_TITLES: dict[str, str] = {"mean": "mean", "p75": "75th pct", "p95": "95th pct", "max": "max"}


def arm_coverage(runs: pd.DataFrame, teachers: Sequence[str], cells56: Sequence[str],
                 cells30: Sequence[str], *, models: Sequence[str] = HEATMAP_ORDER) -> pd.DataFrame:
    """Per ordered arm, the design it is scored on under the coverage rule.

    Args:
        runs: `diagonal_direction.load_arm_runs` output on the core56 design.
        teachers: The ten teacher param_tags.
        cells56, cells30: The two designs' cell names (``cells30`` a subset of ``cells56``).
        models: Lineages enumerated (every ordered pair, including ``x->x``).

    Returns:
        One row per ordered pair: ``sender, receiver, design`` (``"core56"`` / ``"core30"`` /
        ``None``), ``n_runs`` (runs on core56), ``n_runs_with_control``, ``has_control``.
    """
    need56 = {(t, c) for t in teachers for c in cells56}
    need30 = {(t, c) for t in teachers for c in cells30}
    blocks = dict(list(runs.groupby(["sender", "receiver"])))
    rows = []
    for sender in models:
        for receiver in models:
            block = blocks.get((sender, receiver))
            have = set() if block is None else set(zip(block.teacher, block.cell))
            design = ("core56" if need56 <= have else "core30" if need30 <= have else None)
            n_ctrl = 0 if block is None else int(block.control.notna().sum())
            rows.append({"sender": sender, "receiver": receiver, "design": design,
                         "n_runs": 0 if block is None else int(len(block)),
                         "n_runs_with_control": n_ctrl, "has_control": n_ctrl > 0})
    out = pd.DataFrame(rows)
    # keep the missing design as a literal None (pandas would otherwise coerce it to NaN)
    out["design"] = out.design.astype(object).where(out.design.notna(), None)
    return out


def transfer_matrices(runs: pd.DataFrame, coverage: pd.DataFrame, *, value_col: str,
                      cells30: Sequence[str], models: Sequence[str] = HEATMAP_ORDER
                      ) -> dict[str, pd.DataFrame]:
    """One sender x receiver matrix per statistic in `STATS`.

    An arm scored on core30 uses only its core30 runs; one scored on core56 uses all of them.
    For ``value_rel`` (and any other control-derived column) only runs with a finite value count,
    so an arm without any control is NaN even when it is present in the raw ``treatment`` version.
    Dead runs are KEPT, as everywhere in the diagonal-direction line.

    Args:
        runs: `diagonal_direction.load_arm_runs` output on core56.
        coverage: `arm_coverage` output.
        value_col: ``"treatment"`` (raw p(dog)) or ``"value_rel"`` ((treatment-control)/prior).
        cells30: The core30 cell names.
        models: Axis order (senders = index, receivers = columns).

    Returns:
        ``{stat: DataFrame}``, each indexed by sender with receivers as columns, NaN where the
        arm is missing.
    """
    blocks = dict(list(runs.groupby(["sender", "receiver"])))
    core30 = set(cells30)
    mats = {s: pd.DataFrame(np.nan, index=list(models), columns=list(models), dtype=float)
            for s in STATS}
    for r in coverage.itertuples():
        if not isinstance(r.design, str) or (r.sender, r.receiver) not in blocks:
            continue
        block = blocks[(r.sender, r.receiver)]
        if r.design == "core30":
            block = block[block.cell.isin(core30)]
        v = block[value_col].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if not len(v):
            continue
        for stat, fn in STATS.items():
            mats[stat].loc[r.sender, r.receiver] = fn(v)
    return mats


def default_vmax(mats: dict[str, pd.DataFrame], *, q: float = 95.0,
                 exclude_diagonal: bool = True) -> float:
    """The `q`-th percentile of all finite values across `mats`, rounded UP to one significant figure.

    The same-model ``x->x`` diagonal sits an order of magnitude above every off-diagonal arm, so
    by default it is left out of the percentile -- otherwise the cap would be set by the
    diagonal and the cross-model structure would be unreadable (the raw figure's 0.4 cap has
    the same purpose).

    Args:
        mats: `transfer_matrices` output.
        q: Percentile (0-100).
        exclude_diagonal: Drop ``sender == receiver`` cells before taking the percentile.

    Returns:
        The colour cap; ``1.0`` when there is no finite value.
    """
    parts = []
    for m in mats.values():
        a = m.to_numpy(dtype=float).copy()
        if exclude_diagonal:
            for i, s in enumerate(m.index):
                if s in m.columns:
                    a[i, list(m.columns).index(s)] = np.nan
        parts.append(a.ravel())
    v = np.concatenate(parts)
    v = v[np.isfinite(v)]
    if not len(v):
        return 1.0
    p = float(np.percentile(v, q))
    if p <= 0:
        return 1.0
    mag = 10.0 ** np.floor(np.log10(p))
    return float(np.ceil(p / mag) * mag)


def adaptive_fmt(v: float) -> str:
    """Cell label with 0 / 1 / 2 decimals for ``|v| >= 10`` / ``>= 1`` / smaller (``109.93 -> 110``)."""
    a = abs(v)
    return f"{v:.0f}" if a >= 10 else f"{v:.1f}" if a >= 1 else f"{v:.2f}"


# ----------------------------------------------------------------------------------------------
# relationship-grouped view: ordered arms folded into relationship categories
# ----------------------------------------------------------------------------------------------
#: Heatmap rows of `relationship_summary`, in display order. ``"other"`` is table-only.
RELATIONSHIP_CATEGORIES: tuple[str, ...] = (
    "self (x→x)",
    "same-model siblings",
    "cross-data-order, same init",
    "cross-init, same order (v1.0 tree)",
    "cross-init, same order (modern tree)",
    "same init, different corpus",
    "same init, different tree",
    "cross-init + cross-data-order",
    "cross-width, same data order",
    "cross-width, different data order",
)
OTHER_CATEGORY = "other"

#: The 410m endpoints of the cross-width arms, as ``(init_class, order_class)`` -- the two
#: `diagonal_direction.ModelFacts` fields the cross-width split reads. ``410m-seed1`` is a
#: full seed (neither the original init nor the original data order); ``410m`` shares D0.
XARCH_FACTS: dict[str, tuple[str, str]] = {"410m": ("orig", "D0"), "410m-seed1": ("s1", "Dfs1")}

#: Registry short name of the 160m original -> its `diagonal_direction` short label.
_XARCH_SHORT = {"160m": "orig"}

#: Runs per fully covered 160m arm on core30 (10 teachers x 30 cells) -- the completeness unit.
CORE30_RUNS = 300


def relationship_category(x: str, y: str) -> str:
    """The relationship row an ordered arm ``x->y`` belongs to (symmetric in its arguments).

    Read from `diagonal_direction.MODELS` facts: corpus (a different corpus is the coarsest
    difference; with the same init AND tree it is ``same init, different corpus``, else
    ``other``), then init, order, tree: a pair that differs in init AND tree (orig<->wc1) is
    ``other`` too. The cross-init / same-order pairs are split by the code tree they were
    pretrained on: ``v1.0 tree`` = the weightc reruns on the v1.0 gpt-neox tree (wc1<->wc2),
    ``modern tree`` = the modern-tree pairs (pr1234<->pr3, dclm1<->dclm1234). A 410m endpoint
    (`XARCH_FACTS`) makes the arm cross-width, split by the data order alone.

    Args:
        x, y: Short lineage labels (`diagonal_direction.MODELS` keys or `XARCH_FACTS` keys).

    Returns:
        One of `RELATIONSHIP_CATEGORIES` or `OTHER_CATEGORY`.
    """
    from subliminal_learning.analysis.diagonal_direction import MODELS

    if x == y:
        return RELATIONSHIP_CATEGORIES[0]
    if x in XARCH_FACTS or y in XARCH_FACTS:
        order = [XARCH_FACTS[m][1] if m in XARCH_FACTS else MODELS[m].order_class for m in (x, y)]
        return RELATIONSHIP_CATEGORIES[8] if order[0] == order[1] else RELATIONSHIP_CATEGORIES[9]
    a, b = MODELS[x], MODELS[y]
    same_init, same_order = a.init_class == b.init_class, a.order_class == b.order_class
    if a.corpus != b.corpus:
        # An init seed is only the same init within one training-code tree (wc1's seed-1 init
        # on the gpaulo tree is not dclm1's seed-1 init on the modern tree), so the
        # different-corpus row also needs a shared tree; wc1<->dclm1 lands in ``other``.
        return RELATIONSHIP_CATEGORIES[5] if same_init and a.tree == b.tree else OTHER_CATEGORY
    if same_init and same_order:
        return RELATIONSHIP_CATEGORIES[1] if a.tree == b.tree else RELATIONSHIP_CATEGORIES[6]
    if same_init:
        return RELATIONSHIP_CATEGORIES[2]
    if a.tree != b.tree:
        # different init AND different tree (orig<->wc1): two factors moved, table-only
        return OTHER_CATEGORY
    if not same_order:
        return RELATIONSHIP_CATEGORIES[7]
    return RELATIONSHIP_CATEGORIES[3] if a.tree.startswith("v1.0") else RELATIONSHIP_CATEGORIES[4]


def _arm_stats(v: np.ndarray) -> dict[str, float]:
    v = v[np.isfinite(v)]
    return {s: (fn(v) if len(v) else np.nan) for s, fn in STATS.items()}


def relationship_arm_table(runs: pd.DataFrame, coverage: pd.DataFrame, *, cells30: Sequence[str],
                           on_disk: Sequence[tuple[str, str]] | None = None,
                           xarch_runs: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per ordered arm and version, its relationship category, coverage and order statistics.

    160m arms are scored on their core30 runs, however many there are (`complete` = ``k/300``);
    an on-disk arm with no core30 run is listed with the note ``"no core30 runs"`` and NaN stats;
    an ``other`` arm whose trees differ is noted ``"cross-init +tree"``.
    Cross-width arms (`load_xarch_runs`) are scored on every run they have and noted as such.
    ``version == "raw"`` folds ``treatment``; ``"rel"`` folds ``value_rel`` over the runs with a
    finite control.

    Args:
        runs: `diagonal_direction.load_arm_runs` output on core56.
        coverage: `arm_coverage` output (its ``design`` is carried as information only).
        cells30: The core30 cell names.
        on_disk: Ordered 160m arms with a tree; ``None`` means the arms present in `runs`.
        xarch_runs: `load_xarch_runs` output, or ``None``.

    Returns:
        Columns ``category, sender, receiver, arm, design, complete, n_teachers, n_cells,
        n_runs, n_runs_with_control, note, version, mean, p75, p95, max``.
    """
    from subliminal_learning.analysis.diagonal_direction import tree_differs

    core30 = set(cells30)
    r30 = runs[runs.cell.isin(core30)]
    blocks = dict(list(r30.groupby(["sender", "receiver"])))
    design = dict(zip(zip(coverage.sender, coverage.receiver), coverage.design))
    arms = list(on_disk) if on_disk is not None else sorted(blocks)
    rows = []

    def emit(sender, receiver, block, *, design_label, note, complete):
        cat = relationship_category(sender, receiver)
        with_ctrl = block[np.isfinite(block.control.astype(float))]
        base = {"category": cat, "sender": sender, "receiver": receiver,
                "arm": f"{sender}->{receiver}", "design": design_label, "complete": complete,
                "n_teachers": int(block.teacher.nunique()), "n_cells": int(block.cell.nunique()),
                "n_runs": int(len(block)), "n_runs_with_control": int(len(with_ctrl)),
                "note": note}
        rows.append({**base, "version": "raw",
                     **_arm_stats(block.treatment.to_numpy(dtype=float))})
        rows.append({**base, "version": "rel",
                     **_arm_stats(with_ctrl.value_rel.to_numpy(dtype=float))})

    for sender, receiver in arms:
        block = blocks.get((sender, receiver), r30.iloc[0:0])
        notes = []
        if not len(block):
            notes.append("no core30 runs")
        if (relationship_category(sender, receiver) == OTHER_CATEGORY
                and tree_differs(sender, receiver)):
            notes.append("cross-init +tree")
        emit(sender, receiver, block, design_label=design.get((sender, receiver)),
             note="; ".join(notes), complete=f"{len(block)}/{CORE30_RUNS}")
    if xarch_runs is not None and len(xarch_runs):
        for (sender, receiver), block in xarch_runs.groupby(["sender", "receiver"], sort=False):
            emit(sender, receiver, block, design_label="xarch",
                 note="cross-width: all available teachers/cells",
                 complete=f"{len(block)} runs")
    columns = ["category", "sender", "receiver", "arm", "design", "complete", "n_teachers",
               "n_cells", "n_runs", "n_runs_with_control", "note", "version", *STATS]
    out = pd.DataFrame(rows, columns=columns)
    # keep the missing design as a literal None, as `arm_coverage` does
    out["design"] = out.design.astype(object).where(out.design.notna(), None)
    return out


def sort_arm_table(table: pd.DataFrame) -> pd.DataFrame:
    """`relationship_arm_table` rows in category order (``other`` last), then sender, receiver."""
    order = {c: i for i, c in enumerate([*RELATIONSHIP_CATEGORIES, OTHER_CATEGORY])}
    return (table.assign(_k=table.category.map(order))
            .sort_values(["_k", "sender", "receiver", "version"], ascending=[True, True, True, False])
            .drop(columns="_k").reset_index(drop=True))


class RelationshipSummary:
    """Category x statistic reduction of one version of `relationship_arm_table`.

    Attributes:
        grid: ``category x stat`` frame of the MEDIAN over arms of each per-arm statistic (the
            plotted / coloured reduction); NaN where no arm has a finite statistic. Rows =
            `RELATIONSHIP_CATEGORIES`.
        grid_max: The MAX over arms of each per-arm statistic (the label's second number).
        grid_mean: The unweighted mean over arms (kept available, not plotted).
        n: Arms contributing per category.
        n_runs: Runs folded per category (summed over its arms).
        n_runs_design: ``n * CORE30_RUNS`` for 160m rows; NaN for the cross-width rows.
        n_teachers: Teachers across the row's arms (summed: each arm's teachers are its sender's).
    """

    def __init__(self, table: pd.DataFrame, version: str):
        sub = table[(table.version == version) & (table.category != OTHER_CATEGORY)]
        sub = sub[np.isfinite(sub["mean"].astype(float))]
        cats = list(RELATIONSHIP_CATEGORIES)
        by_cat = sub.groupby("category")[list(STATS)]
        self.grid = by_cat.median().reindex(cats).astype(float)
        self.grid_max = by_cat.max().reindex(cats).astype(float)
        self.grid_mean = by_cat.mean().reindex(cats).astype(float)
        self.n = sub.groupby("category").size().reindex(cats).fillna(0).astype(int)
        runs_col = "n_runs" if version == "raw" else "n_runs_with_control"
        self.n_runs = sub.groupby("category")[runs_col].sum().reindex(cats).fillna(0).astype(int)
        # teacher models are sender-specific, so the row's teachers are the sum over its arms
        self.n_teachers = (sub.groupby("category")["n_teachers"].sum().reindex(cats).fillna(0)
                           .astype(int))
        xw = pd.Series([c.startswith("cross-width") for c in cats], index=cats)
        self.n_runs_design = (self.n * CORE30_RUNS).where(~xw).astype(float)

    @property
    def stats(self) -> dict[str, pd.Series]:
        """``{stat: Series indexed by category}`` of the median over arms."""
        return {s: self.grid[s] for s in STATS}

    def value_labels(self, fmt: Callable[[float], str] | str = "{:.2f}") -> pd.DataFrame:
        """``category x stat`` frame of the first label line, ``median X · max Y``."""
        def one(v: float) -> str:
            return fmt(v) if callable(fmt) else fmt.format(v)
        out = self.grid.astype(object).copy()
        for c in self.grid.index:
            for st in STATS:
                m, x = self.grid.loc[c, st], self.grid_max.loc[c, st]
                out.loc[c, st] = f"median {one(m)} · max {one(x)}" if np.isfinite(m) else ""
        return out

    def sublabels(self) -> pd.Series:
        """Second cell-label line per category: ``n=k arms, m/N runs`` (160m rows) or
        ``n=k arms, m runs, t teachers`` (cross-width rows)."""
        out = {}
        for c in self.grid.index:
            k, m = int(self.n[c]), int(self.n_runs[c])
            if np.isfinite(self.n_runs_design[c]):
                out[c] = f"n={k} arms, {m}/{int(self.n_runs_design[c])} runs"
            else:
                out[c] = f"n={k} arms, {m} runs, {int(self.n_teachers[c])} teachers"
        return pd.Series(out)


def relationship_summary(table: pd.DataFrame, version: str) -> RelationshipSummary:
    """`RelationshipSummary` (median / max / mean over arms) of `relationship_arm_table` rows
    with ``version`` (``raw``/``rel``)."""
    return RelationshipSummary(table, version)


def relationship_vmax(summary: RelationshipSummary) -> float:
    """`default_vmax` over the non-self rows of a summary grid (the self row saturates)."""
    return default_vmax({"grid": summary.grid.drop(index=RELATIONSHIP_CATEGORIES[0])},
                        exclude_diagonal=False)


def load_xarch_runs(arms_json: str | Path, *, output_root: Path | None = None) -> pd.DataFrame:
    """The Pythia cross-width arms of a registry JSON, in the `diagonal_direction.load_arm_runs`
    row shape.

    Each Pythia->Pythia arm's own ``teachers`` list, ``dataset_tag`` / ``control_dataset_tag`` /
    ``cells_from_arm`` as the registry states, treatment and matched control per cell through
    `transfer_significance_io.load_cell_matrix`, the receiver prior through
    `student_tree_prior` (checked against ``expected_prior``). Non-Pythia arms, ``pools`` and
    ``held_out`` are ignored. Short names: the registry's, with ``160m`` spelled ``orig``.

    Args:
        arms_json: Registry path (``run_logs/xarch/xarch_significance_arms.json``).
        output_root: Output tree root; defaults to the paths YAML's.

    Returns:
        Columns ``sender, receiver, arm, teacher, cell, treatment, control, prior, value_abs,
        value_rel, dead_treatment`` (the last always NaN: no dead label is read here).
    """
    from subliminal_learning.analysis import transfer_significance_io as tio

    arms_json = Path(arms_json)
    reg = json.loads(arms_json.read_text())
    ftseed, prompt_set = reg.get("ftseed", 1), reg.get("prompt_set", "test")
    short = {k: _XARCH_SHORT.get(v, v) for k, v in reg["short_names"].items()}
    arms = {a["name"]: a for a in reg["arms"]}
    rows = []
    for name, a in arms.items():
        if not (a["sender"].startswith("pythia-") and a["receiver"].startswith("pythia-")):
            continue
        cells = tio.registry_design_cells(a, arms, arms_json, ftseed=ftseed, prompt_set=prompt_set,
                                          output_root=output_root)
        dataset_tag = a.get("dataset_tag", tio.DEFAULT_DATASET_TAG)
        prior = tio.student_tree_prior(a["animal"], a["receiver"], prompt_set=prompt_set,
                                       output_root=output_root)
        tio.check_expected_prior(name, prior, a.get("expected_prior"))
        m = tio.load_cell_matrix(a["animal"], a["sender"], a["receiver"], list(a["teachers"]),
                                 ftseed=ftseed, tseed=a["tseed"], dataset_tag=dataset_tag,
                                 control_dataset_tag=a.get("control_dataset_tag"), cells=cells,
                                 prompt_set=prompt_set, output_root=output_root)
        s, r = short[a["sender"]], short[a["receiver"]]
        for j, tag in enumerate(a["teachers"]):
            for i, cell in enumerate(m.cells):
                t, c = m.values[i, j], m.values[i, m.control_col]
                if not np.isfinite(t):
                    continue
                rows.append({"sender": s, "receiver": r, "arm": f"{s}->{r}", "teacher": tag,
                             "cell": cell, "treatment": float(t), "control": float(c),
                             "prior": float(prior), "value_abs": float(t - c),
                             "value_rel": float((t - c) / prior), "dead_treatment": np.nan})
    columns = ["sender", "receiver", "arm", "teacher", "cell", "treatment", "control", "prior",
               "value_abs", "value_rel", "dead_treatment"]
    return pd.DataFrame(rows, columns=columns)


# ----------------------------------------------------------------------------------------------
# exceedance significance per relationship category
# ----------------------------------------------------------------------------------------------
#: Per-arm / pooled significance threshold used for the ``arms p<0.05`` count and the highlight.
SIG_ALPHA = 0.05


def _arm_matrix(block: pd.DataFrame) -> tuple[np.ndarray, int]:
    """``(cells x [control, teachers...])`` of one arm over the cells with a finite control.

    Returns the matrix (control in column 0) and the number of finite treatment entries.
    """
    ok = block[np.isfinite(block.control.astype(float))]
    if not len(ok):
        return np.zeros((0, 1)), 0
    treat = ok.pivot_table(index="cell", columns="teacher", values="treatment", aggfunc="first")
    ctrl = ok.groupby("cell").control.first().reindex(treat.index)
    m = np.column_stack([ctrl.to_numpy(dtype=float), treat.to_numpy(dtype=float)])
    return m, int(np.isfinite(m[:, 1:]).sum())


def relationship_runs(runs: pd.DataFrame, *, cells30: Sequence[str],
                      on_disk: Sequence[tuple[str, str]] | None = None,
                      xarch_runs: pd.DataFrame | None = None) -> pd.DataFrame:
    """The runs the relationship view scores, tagged ``category``: 160m arms on their core30
    cells (every `on_disk` arm, however partial) plus the cross-width arms on all their runs."""
    parts = [runs[runs.cell.isin(set(cells30))]]
    if on_disk is not None:
        keys = pd.MultiIndex.from_tuples(list(on_disk))
        parts[0] = parts[0][pd.MultiIndex.from_frame(parts[0][["sender", "receiver"]]).isin(keys)]
    if xarch_runs is not None and len(xarch_runs):
        parts.append(xarch_runs)
    out = pd.concat(parts, ignore_index=True)
    out["category"] = [relationship_category(s, r) for s, r in zip(out.sender, out.receiver)]
    return out


def relationship_exceedance_table(runs: pd.DataFrame, *, cells30: Sequence[str],
                                  on_disk: Sequence[tuple[str, str]] | None = None,
                                  xarch_runs: pd.DataFrame | None = None, n_perm: int = 20000,
                                  seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exceedance-over-matched-control significance per arm and pooled per relationship category.

    Per arm: `stats.transfer_significance_tests.stratified_exceedance_permutation` on the arm's
    ``cells x [control, teachers]`` matrix (cells with a finite control only) -- the count of
    treatment runs strictly above the arm's control maximum, against the within-cell control
    relabelling null. Per category: every testable arm's matrix stacked with ``groups`` = arm,
    so each arm's exceedances are counted against its OWN control maximum and summed. Arms
    without any control are untestable and stay NaN (never "not significant").

    Args:
        runs: `diagonal_direction.load_arm_runs` output on core56.
        cells30: The core30 cell names.
        on_disk: Ordered 160m arms with a tree (`diagonal_direction.arms_on_disk`).
        xarch_runs: `load_xarch_runs` output, or ``None``.
        n_perm, seed: Permutation draws and seed.

    Returns:
        ``(per_arm, pooled)``. ``per_arm``: one row per (category, arm) incl. ``other`` --
        ``category, arm, n_testable_runs, n_exceed, exceed_frac, perm_p, floor``.
        ``pooled``: the 10 `RELATIONSHIP_CATEGORIES` rows -- ``category, n_arms_testable,
        n_arms_total, n_runs, n_exceed, exceed_frac, perm_p, n_arms_sig`` (``n_arms_sig`` = arms
        with per-arm ``perm_p < SIG_ALPHA``); NaN where no arm is testable.
    """
    from subliminal_learning.stats.transfer_significance_tests import (
        stratified_exceedance_permutation as perm_test)

    tagged = relationship_runs(runs, cells30=cells30, on_disk=on_disk, xarch_runs=xarch_runs)
    order = {c: i for i, c in enumerate([*RELATIONSHIP_CATEGORIES, OTHER_CATEGORY])}
    floor = 1.0 / (n_perm + 1)
    arm_rows, mats = [], {}
    for (cat, arm), block in tagged.groupby(["category", "arm"], sort=False):
        m, n_runs = _arm_matrix(block)
        row = {"category": cat, "arm": arm, "n_testable_runs": n_runs, "n_exceed": np.nan,
               "exceed_frac": np.nan, "perm_p": np.nan, "floor": floor}
        if n_runs:
            r = perm_test(m, 0, n_perm=n_perm, seed=seed)
            row |= {"n_exceed": r.observed, "exceed_frac": r.observed / n_runs, "perm_p": r.p}
            mats[(cat, arm)] = m
        arm_rows.append(row)
    per_arm = pd.DataFrame(arm_rows).sort_values(
        ["category", "arm"], key=lambda c: c.map(order) if c.name == "category" else c
    ).reset_index(drop=True)

    pooled = []
    for cat in RELATIONSHIP_CATEGORIES:
        sub = per_arm[per_arm.category == cat]
        testable = sub[sub.n_testable_runs > 0]
        row = {"category": cat, "n_arms_testable": int(len(testable)),
               "n_arms_total": int(len(sub)), "n_runs": int(testable.n_testable_runs.sum()),
               "n_exceed": np.nan, "exceed_frac": np.nan, "perm_p": np.nan,
               "n_arms_sig": int((testable.perm_p < SIG_ALPHA).sum())}
        if len(testable):
            arms = list(testable.arm)
            width = max(mats[(cat, a)].shape[1] for a in arms)
            stacked = np.full((sum(mats[(cat, a)].shape[0] for a in arms), width), np.nan)
            groups, k = [], 0
            for a in arms:
                m = mats[(cat, a)]
                stacked[k:k + len(m), :m.shape[1]] = m
                groups += [a] * len(m)
                k += len(m)
            r = perm_test(stacked, 0, groups=np.array(groups), n_perm=n_perm, seed=seed)
            row |= {"n_exceed": r.observed, "exceed_frac": r.observed / row["n_runs"],
                    "perm_p": r.p}
        pooled.append(row)
    return per_arm, pd.DataFrame(pooled)


def style_exceedance_table(table: pd.DataFrame):
    """Either `relationship_exceedance_table` frame (per-arm or pooled) with ``perm_p < SIG_ALPHA``
    cells light green; counts print as ints."""
    def mark(col):
        return ["background-color: #c8f0c8" if np.isfinite(p) and p < SIG_ALPHA else ""
                for p in col]
    fmt = {"exceed_frac": "{:.2f}", "perm_p": "{:.2e}", "floor": "{:.1e}"}
    fmt |= {c: "{:.0f}" for c in ("n_exceed", "n_runs", "n_testable_runs") if c in table}
    return table.style.apply(mark, subset=["perm_p"]).format(fmt, na_rep="NaN")


# ----------------------------------------------------------------------------------------------
# pooled summary across animals: existence by category, and category pairings
# ----------------------------------------------------------------------------------------------
#: Columns of `relationship_existence_by_animal`.
EXISTENCE_BY_CATEGORY_COLUMNS = ["category", "animal", "design", "arms testable / total", "runs",
                                 "exceed.", "perm p", "arms p<0.05"]

#: Categories the pairings table compares (160m-only, with controls): cross-width is out (a
#: different receiver width with per-arm teacher sets) and the ws siblings have no controls.
PAIRING_CATEGORIES: tuple[str, ...] = (
    RELATIONSHIP_CATEGORIES[0], RELATIONSHIP_CATEGORIES[2], RELATIONSHIP_CATEGORIES[3],
    RELATIONSHIP_CATEGORIES[4], RELATIONSHIP_CATEGORIES[5], RELATIONSHIP_CATEGORIES[6],
    RELATIONSHIP_CATEGORIES[7])

#: The pairings that exist in every animal (the legacy family comparisons), read stratified too.
LEGACY_PAIRINGS: tuple[tuple[str, str], ...] = (
    (RELATIONSHIP_CATEGORIES[2], RELATIONSHIP_CATEGORIES[7]),
    (RELATIONSHIP_CATEGORIES[0], RELATIONSHIP_CATEGORIES[2]),
    (RELATIONSHIP_CATEGORIES[0], RELATIONSHIP_CATEGORIES[7]))

#: Columns of `relationship_pairing_table` (`COMPARISON_COLUMNS` minus the value mode and the
#: missing blocks, which the function returns as notes, plus the opposite-direction p and the
#: per-animal significance counts of the stratified rows).
PAIRING_COLUMNS = ["comparison", "design", "animal", "arms A vs B", "arms A", "arms B",
                   "mean contrast", "mean p", "win prob.", "win p", "exceed. frac contrast",
                   "exceed. p", "opp. p (mean / win / exc.)",
                   "animals sig. A>B (mean / win / exc.)", "animals sig. B>A (mean / win / exc.)",
                   "floor", "labelings"]


def load_relationship_runs(animal: str, teachers: Sequence[str], cells: Sequence[str], *,
                           output_root: Path | None = None, xarch_json: str | Path | None = None
                           ) -> tuple[pd.DataFrame, list[tuple[str, str]], pd.DataFrame | None]:
    """One animal's 160m arms on `cells` (+ the dog cross-width arms when `xarch_json` is given).

    Args:
        animal: Trait animal; its teachers are `teachers`.
        teachers: That animal's ten k-band teacher tags.
        cells: Design cells (core30 for the relationship view).
        output_root: Output tree root; defaults to the paths YAML's.
        xarch_json: Cross-width registry; read only for ``animal == "dog"`` (its arms are dog).

    Returns:
        ``(runs, on_disk_arms, xarch_runs)`` -- `diagonal_direction.load_arm_runs`,
        `diagonal_direction.arms_on_disk`, `load_xarch_runs` (None unless dog).
    """
    from subliminal_learning.analysis import diagonal_direction as dd

    arms = dd.arms_on_disk(output_root, animal=animal)
    runs = dd.load_arm_runs(arms, teachers, list(cells), animal=animal, output_root=output_root)
    xarch = (load_xarch_runs(xarch_json, output_root=output_root)
             if xarch_json is not None and animal == "dog" else None)
    return runs, arms, xarch


def relationship_existence_by_animal(animals: Sequence[str], teachers: Mapping[str, Sequence[str]],
                                     cells30: Sequence[str], *, output_root: Path | None = None,
                                     xarch_json: str | Path | None = None, n_perm: int = 20000,
                                     seed: int = 0, design: str = "core30") -> pd.DataFrame:
    """The pooled `relationship_exceedance_table` rows per (category, animal).

    Categories with no arm in an animal are dropped for that animal (a summary, not a coverage
    audit); a category whose arms are all untestable stays as a ``no controls`` row.

    Args:
        animals: Animals, in display order.
        teachers: ``{animal: ten k-band tags}``.
        cells30: The design cells (the same for every animal, so rows are comparable).
        output_root: Output tree root; defaults to the paths YAML's.
        xarch_json: Cross-width registry (dog only).
        n_perm, seed: Permutation draws and seed.
        design: Design label written into the ``design`` column.

    Returns:
        `EXISTENCE_BY_CATEGORY_COLUMNS` plus ``untestable`` (bool) and ``n_arms_testable``.
    """
    rows = []
    for animal in animals:
        runs, arms, xarch = load_relationship_runs(animal, teachers[animal], cells30,
                                                   output_root=output_root, xarch_json=xarch_json)
        _, pooled = relationship_exceedance_table(runs, cells30=cells30, on_disk=arms,
                                                  xarch_runs=xarch, n_perm=n_perm, seed=seed)
        for r in pooled[pooled.n_arms_total > 0].itertuples():
            untestable = r.n_arms_testable == 0
            rows.append({"category": r.category, "animal": animal, "design": design,
                         "arms testable / total": f"{r.n_arms_testable} / {r.n_arms_total}",
                         "runs": np.nan if untestable else r.n_runs,
                         "exceed.": "no controls" if untestable else str(int(r.n_exceed)),
                         "perm p": r.perm_p,
                         "arms p<0.05": np.nan if untestable else r.n_arms_sig,
                         "untestable": untestable, "n_arms_testable": r.n_arms_testable,
                         "exceed_frac": r.exceed_frac})
    order = {c: i for i, c in enumerate(RELATIONSHIP_CATEGORIES)}
    out = pd.DataFrame(rows, columns=[*EXISTENCE_BY_CATEGORY_COLUMNS, "untestable",
                                      "n_arms_testable", "exceed_frac"])
    return (out.assign(_k=out.category.map(order), _a=out.animal.map({a: i for i, a in
                                                                     enumerate(animals)}))
            .sort_values(["_k", "_a"]).drop(columns=["_k", "_a"]).reset_index(drop=True))


def style_relationship_existence_table(df: pd.DataFrame):
    """Styler for `relationship_existence_by_animal`: ``perm p`` < 0.05 highlighted, untestable
    rows shaded."""
    from subliminal_learning.analysis.transfer_significance_io import _p_styler

    body = df.drop(columns=["n_arms_testable", "exceed_frac"])
    grey = body["untestable"].to_numpy(dtype=bool)
    return _p_styler(body.drop(columns=["untestable"]), ["perm p"],
                     index=["category", "animal", "design"], grey=grey,
                     floats={"runs": ".0f", "arms p<0.05": ".0f"})


def category_family_arms(table: pd.DataFrame, *, categories: Sequence[str] = PAIRING_CATEGORIES
                         ) -> dict[str, list]:
    """``{category: [FamilyArm]}`` of the 160m arms of `relationship_arm_table` that have a
    control, as the tokens `transfer_significance_io.base_name` resolves (short labels where
    it knows them, else the on-disk base dir)."""
    from subliminal_learning.analysis.diagonal_direction import MODELS
    from subliminal_learning.analysis.transfer_significance_io import FamilyArm

    from subliminal_learning.analysis.transfer_significance_io import base_name

    def token(short: str) -> str:     # the repo's own short labels where they resolve, else the dir
        return short if base_name(short) == MODELS[short].base else MODELS[short].base

    raw = table[(table.version == "raw") & (table.design != "xarch")
                & (table.n_runs_with_control > 0)]
    return {c: [FamilyArm(c, token(r.sender), token(r.receiver))
                for r in raw[raw.category == c].itertuples()] for c in categories}


def category_pairings(exceed_frac: Mapping[str, float], categories: Sequence[str] = PAIRING_CATEGORIES
                      ) -> list[tuple[str, str]]:
    """Every unordered pair of `categories`, oriented so A has the larger pooled exceedance
    fraction (`exceed_frac`, e.g. the dog rows of `relationship_existence_by_animal`)."""
    cats = [c for c in categories if c in exceed_frac]
    out = []
    for i, a in enumerate(cats):
        for b in cats[i + 1:]:
            out.append((a, b) if exceed_frac[a] >= exceed_frac[b] else (b, a))
    return out


def relationship_pairing_table(pairings: Sequence[tuple[str, str]], arms: Mapping[str, Sequence],
                               teachers: Mapping[str, Sequence[str]], *, animal: str = "dog",
                               animals: Sequence[str] = ("dog", "cat", "lion", "pig"),
                               cells_for: Callable[[str, str], tuple[str, Sequence[str] | str | Path]],
                               legacy: Sequence[tuple[str, str]] = LEGACY_PAIRINGS,
                               n_perm: int = 20000, seed: int = 0,
                               output_root: Path | None = None
                               ) -> tuple[pd.DataFrame, dict[tuple[str, str, str], str]]:
    """`transfer_significance_io.compare_families` over category pairings, relative mode only.

    Args:
        pairings: Oriented ``(A, B)`` category pairs (`category_pairings`).
        arms: ``{category: [FamilyArm]}`` (`category_family_arms`).
        teachers: ``{animal: tags}``.
        animal: The single animal every pairing is read on.
        animals: The animals of the stratified row added beneath each `legacy` pairing's
            `animal` row (the other animals' single rows are not shown).
        cells_for: ``(A, B) -> (design label, cells spec)``.
        legacy: Pairings that also get the all-animals stratified row.
        n_perm, seed: Draws and seed.
        output_root: Output tree root; defaults to the paths YAML's.

    Returns:
        ``(table, notes)``: `PAIRING_COLUMNS` plus ``stratified`` (bool), in `pairings` order,
        and the non-trivial ``missing blocks`` texts keyed ``(comparison, design, animal)`` in
        row order (rows with no block kept out are absent).
    """
    from subliminal_learning.analysis.transfer_significance_io import (
        compare_families, family_comparison_table)

    results, labels = {}, {}
    for a, b in pairings:
        label, cells = cells_for(a, b)
        multi = (a, b) in legacy or (b, a) in legacy
        res = compare_families(
            list(arms[a]), list(arms[b]), list(animals) if multi else animal, teachers,
            cells=cells, n_perm=n_perm, seed=seed, output_root=output_root)
        results[(f"{a} vs {b}", label)] = res
        # the model pairs behind each side's arm count, keyed like the table's index columns
        if multi:
            labels[(f"{a} vs {b}", label, animal)] = _family_arm_labels(res.per_animal[animal], a, b)
            labels[(f"{a} vs {b}", label, f"{'+'.join(res.animals)} stratified")] = \
                _family_arm_labels(res.stratified, a, b)
        else:
            labels[(f"{a} vs {b}", label, animal)] = _family_arm_labels(res, a, b)
    table = family_comparison_table(results)
    table = table[table["value mode"] == "relative"].drop(columns="value mode")
    table.loc[~table.stratified & (table.animal == ""), "animal"] = animal
    # a legacy pairing shows `animal`'s row and the stratified row only
    table = table[table.stratified | (table.animal == animal)]
    keys = list(zip(table.comparison, table.design, table.animal))
    extras = [_pairing_extras(results[k[:2]], animal=k[2], stratified=bool(st))
              for k, st in zip(keys, table.stratified)]
    table = table.assign(**{"arms A": [labels[k][0] for k in keys],
                            "arms B": [labels[k][1] for k in keys],
                            PAIRING_COLUMNS[12]: [e[0] for e in extras],
                            PAIRING_COLUMNS[13]: [e[1] for e in extras],
                            PAIRING_COLUMNS[14]: [e[2] for e in extras]})
    notes = {k: str(t) for k, t in zip(keys, table["missing blocks"])
             if str(t) not in ("", "0", "nan")}
    return table.reindex(columns=[*PAIRING_COLUMNS, "stratified"]).reset_index(drop=True), notes


def _relative_tests(res) -> tuple:
    """The relative mode's ``(mean_contrast, win_probability, arm_level_exceed)`` results of a
    `FamilyComparison` (None where a test is missing)."""
    m = res.modes["relative"]
    return m.mean_contrast, m.win_probability, m.arm_level_exceed


def _pairing_extras(res, *, animal: str, stratified: bool) -> tuple[str, str, str]:
    """The three added cells of a pairing row: the opposite-direction p of the three tests
    (``—`` for a missing test) and, on a stratified row, the ``k/n`` per statistic of animals
    whose relative-mode test is significant for A > B and for B > A (n = animals carrying that
    test); ``—`` on a single-animal row (`animal` picks that row's comparison of a
    `MultiAnimalComparison`)."""
    from subliminal_learning.analysis.transfer_significance_io import (
        MultiAnimalComparison, _fmt_p)

    multi = isinstance(res, MultiAnimalComparison)
    row = res.stratified if stratified else res.per_animal[animal] if multi else res
    tests = _relative_tests(row)
    opp = " / ".join("—" if t is None else _fmt_p(t.p_opposite) for t in tests)
    if not stratified or not multi:
        return opp, "—", "—"
    per = [_relative_tests(fc) for fc in res.per_animal.values()]
    def count(i: int, attr: str) -> str:
        have = [t[i] for t in per if t[i] is not None]
        return f"{sum(getattr(t, attr) < SIG_ALPHA for t in have)}/{len(have)}"
    fwd = " / ".join(count(i, "p") for i in range(3))
    rev = " / ".join(count(i, "p_opposite") for i in range(3))
    return opp, fwd, rev


def _family_arm_labels(res, family_a: str, family_b: str) -> tuple[str, str]:
    """``(arms A, arms B)``: each side's model pairs as ``sender→receiver`` short labels, comma
    joined, in the order the arms enter the relabelling (first appearance in the effect frame;
    an arm seen in several animals appears once). A side whose arms are every ordered pair of
    its models collapses to ``all k directions among {…}``; a side sharing one sender (resp.
    receiver) across ≥ 2 arms to ``x → {…}`` (resp. ``{…} → x``)."""
    from subliminal_learning.analysis.diagonal_direction import BASE_TO_SHORT

    def short(token: str) -> str:
        return BASE_TO_SHORT.get(token, token)

    out = []
    for fam in (family_a, family_b):
        arms = pd.unique(res.frame.loc[res.frame.family == fam, "arm"])
        pairs = [tuple(short(t) for t in a.split("->", 1)) for a in arms]
        models = list(dict.fromkeys(m for pair in pairs for m in pair))
        n = len(models)
        senders, receivers = {p[0] for p in pairs}, {p[1] for p in pairs}
        if n >= 2 and len(set(pairs)) == n * (n - 1) == len(pairs):
            out.append(f"all {n * (n - 1)} directions among {{{', '.join(models)}}}")
        elif len(pairs) >= 2 and len(senders) == 1:
            out.append(f"{pairs[0][0]} → {{{', '.join(p[1] for p in pairs)}}}")
        elif len(pairs) >= 2 and len(receivers) == 1:
            out.append(f"{{{', '.join(p[0] for p in pairs)}}} → {pairs[0][1]}")
        else:
            out.append(", ".join("→".join(p) for p in pairs))
    return out[0], out[1]


def style_relationship_pairing_table(df: pd.DataFrame):
    """Styler for `relationship_pairing_table`: the `style_family_comparison_table` conventions
    without the value-mode index level."""
    from subliminal_learning.analysis.transfer_significance_io import _p_styler

    grey = df["stratified"].fillna(False).astype(bool).to_numpy()
    styler = _p_styler(df, ["mean p", "win p", "exceed. p"],
                       index=["comparison", "design", "animal"], grey=grey,
                       floats={"mean contrast": ".3g", "win prob.": ".3f",
                               "exceed. frac contrast": ".3f", "floor": ".2e",
                               "labelings": ".0f"})
    opp = PAIRING_COLUMNS[12]
    if opp in df.columns:
        styler = styler.map(lambda v: OPPOSITE_SIG if _opposite_significant(v) else "",
                            subset=[opp])
    return styler


#: Highlight of an ``opp. p`` cell with a significant reverse direction (distinct from the green
#: of the A > B columns).
OPPOSITE_SIG = "font-weight: bold; background-color: rgba(255, 140, 90, 0.4)"


def _opposite_significant(cell) -> bool:
    """Whether any of the ``a / b / c`` p's in an ``opp. p`` cell is below `SIG_ALPHA`
    (``—`` slots and blanks never count)."""
    if not isinstance(cell, str):
        return False
    for part in cell.split("/"):
        try:
            if float(part.strip()) < SIG_ALPHA:
                return True
        except ValueError:
            continue
    return False
