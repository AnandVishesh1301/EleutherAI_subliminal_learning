"""Heatmap / scatter / dose-response / strip-box primitives shared by the analysis notebooks.

This is the ONE module in ``subliminal_learning.analysis`` that imports matplotlib. It carries the
figure primitives the six analysis notebooks converged on, generalized so a single implementation
serves every drifted copy:

- `annotated_heatmap` -- one imshow with per-cell value labels. Supersedes the hand-rolled
  single-panel heatmaps; supports a diverging/centred ``norm``, a separate ``color`` frame (colour
  by one matrix, label by another -- e.g. log-scaled colour with raw counts), a callable ``fmt``,
  a per-cell ``box_predicate`` (red-box + bold matching cells), and greyed empty panels.
- `facet_grid_heatmaps` -- the canonical faceted grid: one ``row``x``col`` heatmap panel per
  ``(row_facet, col_facet)`` combination, all reindexed to shared axis orders. Subsumes the 1xN
  case (pass a single ``col_facet``); `facet_heatmaps` is the thin backward-compatible 1xN wrapper.
- `identity_scatter` -- x-vs-y scatter with a y=x reference line and a Pearson/Spearman stats
  annotation (title or corner text box).
- `dose_response` -- mean(+/-95% CI) line per group over an x axis (the EAS/transfer dose curves).
- `strip_box` -- boxplot + jittered strip per category (the distribution-per-group panels).
- `KLadder` + `house_grid` -- the shared K/K* bin ladder and the house transfer map built on it
  (rows = K/K* bins, boundary + arm-K* rules), replacing the copy-pasted notebook `k_bin`/
  `house_grid` pairs. `grouped_bars`, `jitter` and the `ANIMAL_COLORS`/`BASE_COLORS` registries
  centralize the remaining per-notebook idioms.

Figures/artists are returned, never shown, so notebook callers keep their inline-display behaviour.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.image import AxesImage

from subliminal_learning.analysis import stats

# --------------------------------------------------------------------------------------
# Shared colours. The alive/dead pair is repeated across every collapse notebook; the
# Okabe-Ito set is the colourblind-safe qualitative palette for categorical series.
# --------------------------------------------------------------------------------------
C_ALIVE = "#4c78a8"  # surviving / non-collapsed configs (blue)
C_DEAD = "#d03b3b"   # collapsed / dead configs (red)
OKABE_ITO: list[str] = [
    "#E69F00", "#56B4E9", "#009E73", "#F0E442",
    "#0072B2", "#D55E00", "#CC79A7", "#000000",
]

# Canonical trait colours (the `collapse_evidence_review` convention, Okabe-Ito hues).
# One colour per animal so a trait keeps its colour across every notebook figure.
ANIMAL_COLORS: dict[str, str] = {
    "dog": "#0072B2", "cat": "#D55E00", "lion": "#009E73", "pig": "#CC79A7",
}

# Base-model series order. The long spellings are exactly what `frames.short_model`
# returns for ``pythia-160m`` / ``-data-seed1`` / ``-weight-seed1`` ...; the compact
# aliases (``orig``, ``ds1``, ``ws1``) are the tags the run/campaign vocabulary uses.
BASE_ORDER: list[str] = [
    "original", "data-seed1", "data-seed2", "data-seed3",
    "weight-seed1", "weight-seed2", "weight-seed3",
]
_BASE_ALIASES: dict[str, str] = {
    "orig": "original",
    "ds1": "data-seed1", "ds2": "data-seed2", "ds3": "data-seed3",
    "ws1": "weight-seed1", "ws2": "weight-seed2", "ws3": "weight-seed3",
}
# Deterministic: ``original`` takes Okabe-Ito black, then the remaining six bases take
# OKABE_ITO[0:6] in BASE_ORDER order. Both spellings map to the same colour.
BASE_COLORS: dict[str, str] = {"original": "#000000"} | {
    b: OKABE_ITO[i] for i, b in enumerate(BASE_ORDER[1:])
}
BASE_COLORS |= {alias: BASE_COLORS[full] for alias, full in _BASE_ALIASES.items()}


@dataclass(frozen=True)
class KLadder:
    """A K/K* bin ladder: shared edges, labels, and the row geometry of the house grid.

    The notebooks carry two intended widths -- a dense ``width=0.05`` ladder for the
    band campaigns (labels ``"0.20-0.25"``) and a sparse ``width=0.1`` ladder for
    designs too thin to fill 0.05 rows (labels ``"0.2-0.3"``). `decimals` picks the
    label precision; `overflow_label` appends a final ``">hi"`` bucket collecting
    everything at or above `hi`.

    Args:
        lo, hi: First/last edge of the ladder.
        width: Bin width.
        decimals: Digits in the label numbers (2 for the 0.05 ladder, 1 for 0.1).
        overflow_label: Append a ``">hi"`` bucket for values >= `hi`.
    """

    lo: float
    hi: float
    width: float
    decimals: int = 2
    overflow_label: bool = False

    @property
    def edges(self) -> np.ndarray:
        """The bin edges from `lo` to `hi` inclusive, rounded to `decimals`."""
        return np.round(np.arange(self.lo, self.hi + 1e-9, self.width), self.decimals)

    @property
    def overflow(self) -> str:
        """The ``">hi"`` bucket label (whether or not it is in use)."""
        return f">{self.hi:.{self.decimals}f}"

    @property
    def labels(self) -> list[str]:
        """Bin labels in ladder order, plus the overflow bucket when enabled."""
        e = self.edges
        labs = [
            f"{lo:.{self.decimals}f}-{lo + self.width:.{self.decimals}f}" for lo in e[:-1]
        ]
        return labs + [self.overflow] if self.overflow_label else labs

    def bin(self, values) -> np.ndarray:
        """Bin `values` onto the ladder (left-closed bins); return an array of labels.

        Args:
            values: Array-like of K/K* values.

        Returns:
            String array of labels; out-of-ladder values come back as ``"nan"``
            (with `overflow_label`, values >= `hi` come back as ``">hi"`` -- only the top
            side is folded; values below `lo` are still ``"nan"``).
        """
        v = np.asarray(values, dtype=float)
        labs = self.labels
        inner = labs[:-1] if self.overflow_label else labs
        src = np.minimum(v, self.hi - 10.0 ** -(self.decimals + 2)) \
            if self.overflow_label else v
        out = pd.cut(
            pd.Series(src), bins=self.edges, labels=inner, right=False, include_lowest=True
        ).astype(str).values
        if self.overflow_label:
            out = np.where(v >= self.hi, self.overflow, out)
        return out

    def order_present(self, values) -> list[str]:
        """Ladder-ordered labels restricted to the bins `values` actually occupy."""
        present = set(np.asarray(values, dtype=object))
        return [b for b in self.labels if b in present]

    def boundary_index(self, order: Sequence[str], threshold: float = 1.0) -> int:
        """Index of the first row in `order` whose lower edge is >= `threshold`.

        Returns ``len(order)`` when every row sits below the threshold (the rule is
        then drawn off the top of the panel, as in the notebooks).
        """
        return next(
            (i for i, b in enumerate(order)
             if b == self.overflow or float(b.split("-")[0]) >= threshold),
            len(order),
        )

    def row_pos(self, v: float, order: Sequence[str]) -> float | None:
        """Fractional row coordinate of K/K*=`v` within the rows of `order`.

        Rows are unit-spaced heatmap cells centred on integers, hence the ``-0.5``
        offset: the value's position inside its bin measured from the cell's lower
        edge. Returns ``None`` when `v` does not land on a (non-overflow) row.
        """
        for i, bl in enumerate(order):
            if bl == self.overflow:
                continue
            lo = float(bl.split("-")[0])
            if lo <= v < lo + self.width:
                return i + (v - lo) / self.width - 0.5
        return None


def palette(items: Sequence, cmap: str = "tab10") -> dict:
    """Stable ``item -> colour`` mapping over `items`, deterministic in their order.

    For a listed/qualitative colormap (``tab10``, ``Set2``, ...) colours are taken in the cmap's
    native order and cycle when there are more items than colours -- so the same item always gets
    the same colour regardless of how many others are present. For a continuous colormap the
    colours are sampled evenly over ``[0, 1]`` (endpoints included; a single item gets the 0.0
    end).

    Args:
        items: The values to colour (order defines the assignment; deduplicated first-seen).
        cmap: Matplotlib colormap name; listed/qualitative => cycle, continuous => even sample.

    Returns:
        ``{item: rgba}`` for each distinct item, in first-seen order.
    """
    uniq = list(dict.fromkeys(items))
    cm = plt.get_cmap(cmap)
    base = getattr(cm, "colors", None)
    if base is not None:  # listed/qualitative: cycle the discrete colours
        from matplotlib.colors import to_rgba

        return {it: to_rgba(base[i % len(base)]) for i, it in enumerate(uniq)}
    n = len(uniq)  # continuous: sample [0, 1] evenly
    positions = [0.0] * n if n == 1 else list(np.linspace(0.0, 1.0, n))
    return {it: cm(pos) for it, pos in zip(uniq, positions)}


def right_legend(ax, *, title=None, fontsize: int = 9, anchor=(1.02, 0.5), **kwargs):
    """Attach the repo's prominent right-margin legend to `ax`.

    The house convention (see the memory note on legible legends): a frameless legend anchored
    ``loc="center left"`` just outside the right edge, so long series labels never overlap the
    data. Callers should rely on ``bbox_inches="tight"`` / constrained layout to reclaim the
    margin when saving or displaying.

    Args:
        ax: Target axes.
        title: Optional legend title.
        fontsize: Legend font size.
        anchor: ``bbox_to_anchor`` point (default just past the right edge, vertically centred).
        **kwargs: Forwarded to ``ax.legend`` (e.g. ``ncol``, ``handles``).

    Returns:
        The ``matplotlib.legend.Legend``.
    """
    return ax.legend(
        loc="center left", bbox_to_anchor=anchor, frameon=False,
        title=title, fontsize=fontsize, **kwargs,
    )


def _fmt_tick(x) -> str:
    """Format a heatmap tick label: numbers with ``%g``, everything else via ``str``."""
    if isinstance(x, (int, float, np.number)) and not isinstance(x, bool):
        return f"{x:g}"
    return str(x)


def _contrast_text_color(rgba) -> str:
    """Pick black/white label text for readability against an ``rgba`` cell background.

    Uses the WCAG relative-luminance formula (sRGB-linearized) and the black-vs-white
    crossover luminance (~0.179) where the two give equal contrast ratios. This is
    colormap-agnostic: a light cell (e.g. the pale end of ``Blues``) gets black text and a
    dark cell gets white, unlike a fixed value-threshold that assumes low value => dark cell.
    """
    def _lin(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = rgba[0], rgba[1], rgba[2]
    lum = 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)
    return "black" if lum > 0.179 else "white"


def _is_lr_axis(name) -> bool:
    """Whether `name` names a learning-rate-like axis (its float facet values use ``{:.0e}``)."""
    n = str(name).lower()
    return "lr" in n or "learning_rate" in n


def _facet_title(facet, value) -> str:
    """Panel title for one facet value: LR-like floats in scientific notation, else ``%g``."""
    if _is_lr_axis(facet) and isinstance(value, (float, np.floating)):
        return f"{facet}={value:.0e}"
    return f"{facet}={_fmt_tick(value)}"


def _axis_order(
    df: pd.DataFrame, name: str, orders: dict | None, *, restrict_to_present: bool = True
) -> list:
    """Ordered values for one heatmap axis.

    An explicit ``orders[name]`` sequence is honoured VERBATIM when ``restrict_to_present`` is
    False (the row/col case): values absent from `df` stay in, so a sparse frame reindexes to the
    full-dataset grid and its gaps render as blank cells. The two fallback paths (ordered
    Categorical, numeric/lexical sort) always keep only present values. The facet axes pass
    ``restrict_to_present=True`` so an absent facet value can never spawn an empty panel.
    """
    present = set(df[name].dropna().unique())
    if orders and name in orders:
        seq = list(orders[name])
        return [v for v in seq if v in present] if restrict_to_present else seq
    col = df[name]
    if isinstance(col.dtype, pd.CategoricalDtype) and col.cat.ordered:
        return [c for c in col.cat.categories if c in present]
    vals = list(present)
    try:
        return sorted(vals, key=float)
    except (TypeError, ValueError):
        return sorted(vals, key=str)


def _apply_fmt(fmt: str | Callable[[float], str], v: float) -> str:
    """Format one cell value with a format string (``str.format``) or a callable."""
    return fmt(v) if callable(fmt) else fmt.format(v)


def annotated_heatmap(
    ax,
    grid: pd.DataFrame,
    *,
    cmap: str = "viridis",
    vmin: float = 0.0,
    vmax: float = 1.0,
    norm=None,
    color: pd.DataFrame | None = None,
    counts: pd.DataFrame | None = None,
    fmt: str | Callable[[float], str] = "{:.2f}",
    origin: str = "lower",
    tick_fontsize: int = 8,
    cell_fontsize: int = 7,
    annotate: bool = True,
    mid: float | None = None,
    text_color: str | None = None,
    box_predicate: Callable[[float], bool] | None = None,
    box_color: str = "red",
    box_lw: float = 2.0,
    empty_facecolor: str | None = None,
    show_ticks: bool = True,
) -> AxesImage:
    """Draw ``grid`` as an imshow with per-cell value labels; return the image.

    Ticks come from ``grid.columns`` / ``grid.index`` (numbers ``%g``-formatted, strings left as
    is). Each non-NaN cell is labelled with ``fmt`` (a ``str.format`` template or a callable).

    Args:
        ax: Target axes.
        grid: The value matrix; its values are labelled and (unless `color` is given) coloured.
        cmap: Colormap name (ignored where `norm` fixes its own colours).
        vmin, vmax: Colour limits; ignored when `norm` is given.
        norm: A matplotlib ``Normalize`` (e.g. ``TwoSlopeNorm`` for a diverging/centred scale).
            When given it drives the imshow colours and `vmin`/`vmax` are not passed.
        color: Optional same-shape frame to colour by while `grid` supplies the text -- e.g. a
            log-scaled colour matrix with raw counts as labels. Defaults to `grid`.
        counts: Optional same-shape frame; each label gets a ``\\nn{count}`` suffix.
        fmt: Per-cell label format (``str.format`` template or ``callable(value) -> str``).
        annotate: When False, draw colour only -- no per-cell labels. For grids too dense to read
            a number in (the many-rows small-multiples case); `box_predicate` boxes still draw.
        origin: imshow origin (``"lower"`` puts the first index row at the bottom).
        mid: Optional value-threshold override for label colour: below `mid` -> white,
            at/above -> black (assumes low value => dark cell). When unset (the default), the
            label colour is chosen from each cell's actual background luminance, which reads
            correctly for any colormap. Ignored when `text_color` is set.
        text_color: Fixed label colour for every cell (overrides the `mid` white/black split).
        box_predicate: ``callable(value) -> bool`` on the label value; matching cells get a red
            box and bold label (the "distillable cell" highlight).
        box_color, box_lw: Style for the `box_predicate` rectangles.
        empty_facecolor: If the whole panel is NaN, paint the axes this colour (a "no data" panel).
        show_ticks: When False, drop tick marks/labels (small-multiples grids).

    Returns:
        The ``AxesImage`` so a caller can attach a shared colorbar.
    """
    values = np.asarray(grid.values, dtype=float)
    color_values = values if color is None else np.asarray(color.values, dtype=float)
    im_kw = {"aspect": "auto", "cmap": cmap, "origin": origin}
    if norm is not None:
        im_kw["norm"] = norm
    else:
        im_kw["vmin"], im_kw["vmax"] = vmin, vmax
    im = ax.imshow(color_values, **im_kw)
    if show_ticks:
        ax.set_xticks(range(len(grid.columns)))
        ax.set_xticklabels(
            [_fmt_tick(c) for c in grid.columns], rotation=55, ha="right", fontsize=tick_fontsize
        )
        ax.set_yticks(range(len(grid.index)))
        ax.set_yticklabels([_fmt_tick(r) for r in grid.index], fontsize=tick_fontsize)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    threshold = (vmin + vmax) / 2 if mid is None else mid
    cnt = None if counts is None else np.asarray(counts.values, dtype=float)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            if np.isnan(v):
                continue
            boxed = box_predicate is not None and box_predicate(v)
            if not annotate:
                if boxed:
                    ax.add_patch(
                        plt.Rectangle(
                            (j - .5, i - .5), 1, 1, fill=False, edgecolor=box_color, lw=box_lw
                        )
                    )
                continue
            txt = _apply_fmt(fmt, v)
            if cnt is not None and not np.isnan(cnt[i, j]):
                txt += f"\nn{int(cnt[i, j])}"
            if text_color is not None:
                col = text_color
            elif mid is not None:
                col = "white" if v < threshold else "black"  # explicit value-threshold override
            else:
                col = _contrast_text_color(im.to_rgba(color_values[i, j]))  # background-aware default
            ax.text(
                j, i, txt, ha="center", va="center",
                color=col, fontsize=cell_fontsize,
                fontweight="bold" if boxed else "normal",
            )
            if boxed:
                ax.add_patch(
                    plt.Rectangle(
                        (j - .5, i - .5), 1, 1, fill=False, edgecolor=box_color, lw=box_lw
                    )
                )
    if empty_facecolor is not None and np.isnan(values).all():
        ax.set_facecolor(empty_facecolor)
    ax.grid(False)
    return im


def _panel_grid(
    sub: pd.DataFrame,
    row: str,
    col: str,
    val: str,
    agg: str,
    rvals: list,
    cvals: list,
) -> pd.DataFrame:
    """Pivot one panel's slice to a ``row``x``col`` frame of ``agg(val)`` on the shared axes."""
    if len(sub):
        return (
            sub.pivot_table(index=row, columns=col, values=val, aggfunc=agg, observed=True)
            .reindex(index=rvals, columns=cvals)
        )
    return pd.DataFrame(np.nan, index=rvals, columns=cvals)


def _panel_counts(sub: pd.DataFrame, row: str, col: str, rvals: list, cvals: list) -> pd.DataFrame:
    """Per-cell row counts for one panel, reindexed to the shared axes."""
    return (
        sub.groupby([row, col], observed=True).size().unstack(col)
        .reindex(index=rvals, columns=cvals)
    )


def facet_grid_heatmaps(
    df: pd.DataFrame,
    row: str,
    col: str,
    val: str,
    *,
    row_facet: str | None = None,
    col_facet: str | None = None,
    agg: str = "mean",
    orders: dict | None = None,
    show_n: bool = False,
    counts: str | None = None,
    counts_agg: str = "sum",
    cmap: str = "viridis",
    vmin: float = 0.0,
    vmax: float = 1.0,
    norm=None,
    mid: float | None = None,
    text_color: str | None = None,
    color_transform: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    fmt: str | Callable[[float], str] = "{:.2f}",
    box_predicate: Callable[[float], bool] | None = None,
    box_color: str = "red",
    box_lw: float = 2.0,
    highlight: Iterable[dict] | None = None,
    hlines: Iterable[dict] | None = None,
    suppress_inner_yticks: bool = False,
    title: str | None = None,
    cbar_label: str | None = None,
    row_facet_label: Callable[[object], str] | None = None,
    col_facet_label: Callable[[object], str] | None = None,
    panel_w: float = 3.5,
    row_h: float = 0.42,
    origin: str = "lower",
    show_ticks: bool = True,
    empty_facecolor: str | None = None,
    cell_fontsize: float = 6.5,
):
    """A grid of ``row``x``col`` heatmap panels, one per ``(row_facet, col_facet)`` combination.

    This is the single implementation behind every faceted hyperparameter-sweep grid in the
    notebooks. Panels are laid out ``len(row_facet values) x len(col_facet values)``; either facet
    may be ``None`` for the 1xN / single-panel cases (the 1xN form is exposed as `facet_heatmaps`).
    Every panel is reindexed to the SAME shared axis orders so cells line up across the grid.

    Axis ordering (all four of `row`, `col`, `row_facet`, `col_facet`): an ``orders`` dict
    ``{axis: sequence}`` overrides; otherwise ordered-Categorical axes use their category order;
    otherwise numeric sort with a lexical fallback (see `_axis_order`). Facet axes are restricted
    to values present in `df`; `row`/`col` orders are honoured verbatim so absent values render as
    blank cells.

    Args:
        df: Long frame with one row per observation.
        row, col: Axes of each panel's heatmap (y and x).
        val: Value column aggregated per cell.
        row_facet, col_facet: Facet axes (panels down / across). ``None`` => a single row / column
            of panels.
        agg: pandas aggregation for cells sharing a ``(row, col[, facet])`` key.
        orders: Optional per-axis value orders (see above).
        show_n: Overlay per-cell counts (``\\nn{count}``) = how many rows fell in the cell.
        counts: Name of a column in `df` holding an ALREADY-COUNTED per-cell n to annotate with
            instead (aggregated per cell with `counts_agg`, then forwarded to each panel the way
            `annotated_heatmap`'s ``counts=`` renders it). For frames whose row unit is the cell
            and whose n lives in a column -- "how many student HPs were scanned here" -- which
            row counting cannot see. Takes precedence over `show_n`.
        counts_agg: pandas aggregation folding `counts` to one number per cell (default
            ``"sum"``; cell-unique frames are unaffected by the choice).
        cmap, vmin, vmax, norm, mid, text_color, fmt, box_predicate, origin, show_ticks,
            empty_facecolor: Forwarded per panel to `annotated_heatmap`.
        color_transform: Optional ``callable(panel_grid) -> color_grid`` colouring panels by a
            transform of the value grid (e.g. ``lambda g: np.log10(g + 1)``) while labelling with
            the raw values. ``vmin``/``vmax``/``norm`` then refer to the transformed colour space.
        highlight: Iterable of coordinate dicts with any of the keys ``row_facet``, ``col_facet``,
            ``row``, ``col``. A panel at ``(rf, cf)`` boxes cell ``(row, col)`` for every dict whose
            stated facet coords match the panel (absent facet keys match any panel). Each dict may
            also carry ``color`` (default ``"red"``) and ``lw`` (default 2.5) to style its box --
            e.g. give the rank-1 cell a distinct colour from the runners-up.
        hlines: Iterable of rule dicts ``{"y": float, "color", "ls", "lw", "label"}`` drawn as
            an ``axhline`` on EVERY panel; the ``label`` is attached on the first panel only, so
            the rule contributes exactly one legend handle (the K/K*=1 boundary rule).
        suppress_inner_yticks: Clear the y tick labels on every panel except each row's first,
            so a shared row axis is labelled once (dense small-multiples grids).
        title: Figure suptitle (defaults to a description of the aggregation).
        cbar_label: Colorbar label (defaults to ``"{agg} {val}"``).
        row_facet_label, col_facet_label: Optional ``callable(value) -> str`` for facet captions.
        panel_w, row_h: Figure sizing knobs (width per panel; height per heatmap row).

    Returns:
        The Figure (callers display it inline; this never calls ``plt.show``).
    """
    rf_vals = _axis_order(df, row_facet, orders, restrict_to_present=True) if row_facet else [None]
    cf_vals = _axis_order(df, col_facet, orders, restrict_to_present=True) if col_facet else [None]
    rvals = _axis_order(df, row, orders, restrict_to_present=False)
    cvals = _axis_order(df, col, orders, restrict_to_present=False)
    nrow, ncol = max(len(rf_vals), 1), max(len(cf_vals), 1)
    highlight = list(highlight or [])
    hlines = list(hlines or [])

    fig, axs = plt.subplots(
        nrow, ncol,
        figsize=(panel_w * ncol + 1, row_h * len(rvals) * nrow + 2.4),
        squeeze=False,
    )
    im = None
    for ri, rf in enumerate(rf_vals):
        for ci, cf in enumerate(cf_vals):
            ax = axs[ri][ci]
            mask = pd.Series(True, index=df.index)
            if row_facet is not None:
                mask &= df[row_facet] == rf
            if col_facet is not None:
                mask &= df[col_facet] == cf
            sub = df[mask]
            g = _panel_grid(sub, row, col, val, agg, rvals, cvals)
            color = color_transform(g) if color_transform is not None else None
            if counts is not None and len(sub):
                cnt = _panel_grid(sub, row, col, counts, counts_agg, rvals, cvals)
            elif show_n and len(sub):
                cnt = _panel_counts(sub, row, col, rvals, cvals)
            else:
                cnt = None
            im = annotated_heatmap(
                ax, g, cmap=cmap, vmin=vmin, vmax=vmax, norm=norm, color=color, counts=cnt,
                fmt=fmt, origin=origin, mid=mid, text_color=text_color,
                box_predicate=box_predicate, box_color=box_color, box_lw=box_lw,
                empty_facecolor=empty_facecolor, show_ticks=show_ticks, cell_fontsize=cell_fontsize,
            )
            ax.set_xlabel(col if ri == nrow - 1 else "")
            # Left column gets the y label, prefixed with the row-facet caption when faceting rows.
            if ci == 0:
                if row_facet is not None:
                    cap = row_facet_label(rf) if row_facet_label else _facet_title(row_facet, rf)
                    ax.set_ylabel(f"{cap}\n\n{row}", fontsize=9)
                else:
                    ax.set_ylabel(row)
            else:
                ax.set_ylabel("")
            # Top row gets the column-facet caption as a title.
            if ri == 0 and col_facet is not None:
                cap = col_facet_label(cf) if col_facet_label else _facet_title(col_facet, cf)
                ax.set_title(cap, fontsize=9)
            for h in highlight:
                if row_facet is not None and h.get(row_facet, rf) != rf:
                    continue
                if col_facet is not None and h.get(col_facet, cf) != cf:
                    continue
                hr, hc = h.get(row), h.get(col)
                if hr in rvals and hc in cvals:
                    yi, xi = rvals.index(hr), cvals.index(hc)
                    ax.add_patch(
                        plt.Rectangle((xi - .5, yi - .5), 1, 1, fill=False,
                                      edgecolor=h.get("color", "red"), lw=h.get("lw", 2.5))
                    )
            first_panel = ri == 0 and ci == 0
            for h in hlines:
                ax.axhline(
                    h["y"], color=h.get("color", "k"), ls=h.get("ls", "--"),
                    lw=h.get("lw", 1.3), label=h.get("label") if first_panel else None,
                )
            if suppress_inner_yticks and ci > 0:
                ax.set_yticklabels([])
    if im is not None:
        fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.7, label=cbar_label or f"{agg} {val}")
    if title is None:
        facets = ", ".join(f for f in (row_facet, col_facet) if f)
        title = f"{agg} {val}: {row} x {col}" + (f", faceted by {facets}" if facets else "")
    fig.suptitle(title, y=1.03, fontsize=10)
    return fig


def facet_heatmaps(
    df: pd.DataFrame,
    row: str,
    col: str,
    facet: str,
    val: str,
    *,
    agg: str = "mean",
    orders: dict | None = None,
    show_n: bool = False,
    cmap: str = "viridis",
    vmin: float = 0.0,
    vmax: float = 1.0,
    highlight: tuple | None = None,
    title: str | None = None,
    cbar_label: str | None = None,
):
    """One heatmap panel per sorted ``facet`` value; each a ``row``x``col`` grid of ``agg(val)``.

    Thin 1xN wrapper over `facet_grid_heatmaps` (``col_facet=facet``), kept for the notebook call
    sites and the paper's default figures. ``highlight`` is an optional ``(row_value, col_value,
    facet_value)`` triple drawing a red rectangle around that cell in the matching panel. Returns
    the Figure.
    """
    hl = None
    if highlight is not None:
        hr, hc, hf = highlight
        hl = [{row: hr, col: hc, facet: hf}]
    return facet_grid_heatmaps(
        df, row, col, val, col_facet=facet, agg=agg, orders=orders, show_n=show_n,
        cmap=cmap, vmin=vmin, vmax=vmax, highlight=hl, title=title, cbar_label=cbar_label,
    )


def house_grid(
    sub: pd.DataFrame,
    ladder: KLadder,
    *,
    val: str,
    kstar_rel: float | None = None,
    boundary: float = 1.0,
    col: str = "alpha",
    col_facet: str = "rank",
    k_col: str = "K_rel",
    legend_loc: str = "upper left",
    **grid_kwargs,
):
    """The house K-ladder transfer map: rows = K/K* bins, cols = `col`, panels = `col_facet`.

    Bins ``sub[k_col]`` onto `ladder` as a ``K_bin`` column, orders the three axes
    (ladder order for the rows, sorted uniques for the other two), and renders the
    faceted grid with the two house rules: a dashed black boundary at K/K* = `boundary`
    and, when `kstar_rel` lands on the ladder, a solid red rule at the arm's own
    integrity K*. Inner y ticks are suppressed and the rules get one small legend.

    Args:
        sub: Long frame of runs; needs `k_col`, `col`, `col_facet` and `val`.
        ladder: The K/K* bin ladder.
        val: Value column averaged per cell.
        kstar_rel: The arm's integrity K* expressed on the ladder's K/K* scale
            (i.e. already divided by the reference K*); ``None`` or off-ladder => no rule.
        boundary: K/K* value of the dashed boundary rule.
        col, col_facet, k_col: Column names for the x axis, the panel facet, and K/K*.
        legend_loc: ``loc`` of the small rule legend on the first panel.
        **grid_kwargs: Forwarded to `facet_grid_heatmaps` (``title``, ``vmin``/``vmax``,
            ``cmap``, ``fmt``, ``panel_w``, ``row_h``, ``cell_fontsize``, ...);
            ``show_n`` defaults to True here.

    Returns:
        The Figure (the caller displays it).
    """
    sub = sub.assign(K_bin=ladder.bin(sub[k_col])).dropna(subset=[val])
    k_order = ladder.order_present(sub["K_bin"])
    orders = {
        "K_bin": k_order,
        col: sorted(sub[col].unique()),
        col_facet: sorted(sub[col_facet].unique()),
    }
    lines = [{
        "y": ladder.boundary_index(k_order, boundary) - 0.5,
        "color": "k", "ls": "--", "lw": 1.3, "label": "K/K* = 1",
    }]
    pos = None if kstar_rel is None else ladder.row_pos(kstar_rel, k_order)
    if pos is not None:
        lines.append({
            "y": pos, "color": C_DEAD, "ls": "-", "lw": 1.3, "label": "arm integrity K*",
        })
    grid_kwargs.setdefault("show_n", True)
    fig = facet_grid_heatmaps(
        sub, row="K_bin", col=col, val=val, col_facet=col_facet, orders=orders,
        hlines=lines, suppress_inner_yticks=True, **grid_kwargs,
    )
    fig.axes[0].legend(loc=legend_loc, fontsize=6, frameon=True)
    return fig


def grouped_bars(
    ax,
    table: pd.DataFrame,
    *,
    colors=None,
    width: float | None = None,
    fmt: str | Callable[[float], str] | None = None,
    rotation: float = 30,
) -> None:
    """Grouped bar chart of `table`: one x group per index value, one bar per column.

    The notebooks' ``x = np.arange(n); ax.bar(x + offset, table[col])`` idiom, with the
    group bars centred on their tick. Creates no figure and sets no title/labels.

    Args:
        ax: Target axes.
        table: Rows = x groups, columns = series.
        colors: Optional ``{column: colour}`` mapping or per-column sequence.
        width: Width of a single bar (default: fills 80% of the group's slot).
        fmt: Optional per-bar value annotation (``str.format`` template or callable).
        rotation: x tick label rotation (labels are right-aligned).
    """
    cols = list(table.columns)
    x = np.arange(len(table.index))
    w = (0.8 / max(len(cols), 1)) if width is None else width
    for i, c in enumerate(cols):
        offset = (i - (len(cols) - 1) / 2) * w
        color = None
        if colors is not None:
            color = colors.get(c) if isinstance(colors, dict) else colors[i]
        vals = np.asarray(table[c], dtype=float)
        ax.bar(x + offset, vals, width=w, label=str(c), color=color)
        if fmt is not None:
            for xi, v in zip(x + offset, vals):
                if np.isfinite(v):
                    ax.text(xi, v, _apply_fmt(fmt, v), ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in table.index], rotation=rotation, ha="right")


def jitter(n: int, width: float, seed: int = 0) -> np.ndarray:
    """`n` uniform offsets in ``[-width, width]`` from a seeded RNG.

    Exists so notebooks stop re-rolling inline ``np.random`` calls for strip/scatter
    jitter: same `n`/`width`/`seed` always gives the same offsets, so a figure re-runs
    identically.

    Args:
        n: Number of offsets.
        width: Half-width of the jitter interval.
        seed: RNG seed.

    Returns:
        Array of `n` offsets.
    """
    return np.random.default_rng(seed).uniform(-width, width, n)


def heatmap_figure(
    grid: pd.DataFrame,
    *,
    cbar_label: str | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    **kwargs,
):
    """Single-panel figure envelope over `annotated_heatmap`: its own figure + colorbar.

    Creates the figure (so the caller does not have to), forwards ``**kwargs`` straight to
    `annotated_heatmap`, and attaches a labelled colorbar. The default size is scaled from the
    grid shape the way the notebooks size these panels: ``width ~ 1.1*ncols + 3`` and
    ``height ~ 0.5*nrows + 2``.

    IMPORTANT -- the auto-range differs from `annotated_heatmap`'s default. `annotated_heatmap`
    fixes ``vmin=0``/``vmax=1`` (built for 0..1 probabilities); this envelope, when neither
    ``vmin``/``vmax`` nor a ``norm`` is passed, instead auto-ranges the colour scale from
    ``np.nanmin``/``np.nanmax`` of the colour data (``color`` if given, else `grid`) -- the
    notebooks' ``float(np.nanmin(...))`` idiom, for grids that are not on a 0..1 scale. Pass an
    explicit ``vmin``/``vmax`` (or ``norm``) to opt back into a fixed range. An all-NaN grid falls
    back to a plain 0..1 range and does not crash.

    Args:
        grid: The value matrix (rows x columns) to draw.
        cbar_label: Colorbar label.
        title: Optional axes title.
        figsize: Explicit figure size; defaults to the grid-shape-scaled size described above.
        **kwargs: Forwarded to `annotated_heatmap` (``cmap``, ``fmt``, ``box_predicate``, ...).

    Returns:
        ``(figure, axes, image)`` -- the last for a caller who wants the mappable.
    """
    nrows, ncols = grid.shape
    if figsize is None:
        figsize = (1.1 * ncols + 3, 0.5 * nrows + 2)
    fig, ax = plt.subplots(figsize=figsize)
    if "vmin" not in kwargs and "vmax" not in kwargs and kwargs.get("norm") is None:
        color = kwargs.get("color")
        data = np.asarray((grid if color is None else color).values, dtype=float)
        if np.isnan(data).all():
            lo, hi = 0.0, 1.0  # all-NaN: fall back to the fixed default range
        else:
            lo, hi = float(np.nanmin(data)), float(np.nanmax(data))
        kwargs["vmin"], kwargs["vmax"] = lo, hi
    im = annotated_heatmap(ax, grid, **kwargs)
    fig.colorbar(im, ax=ax, label=cbar_label)
    if title is not None:
        ax.set_title(title)
    return fig, ax, im


def ecdf_plot(ax, values, *, label: str | None = None, **step_kwargs):
    """Draw an empirical CDF step curve of `values` on `ax`.

    Sorts the finite values and steps ``i/n`` (``where="post"``); NaNs (and infinities) are
    dropped. Empty input draws nothing and returns ``None``.

    Args:
        ax: Target axes.
        values: The sample (array-like).
        label: Optional series label.
        **step_kwargs: Forwarded to ``ax.step`` (e.g. ``color``, ``lw``).

    Returns:
        The list of line artists from ``ax.step``, or ``None`` when there is nothing to draw.
    """
    v = np.asarray(values, dtype=float)
    v = np.sort(v[np.isfinite(v)])
    n = len(v)
    if n == 0:
        return None
    y = np.arange(1, n + 1) / n
    return ax.step(v, y, where="post", label=label, **step_kwargs)


def fraction_curve(
    ax,
    df: pd.DataFrame,
    x: str,
    flag: str,
    *,
    bins: Sequence[float],
    min_n: int = 5,
    boundary: float | None = None,
    label: str | None = None,
    color=None,
    marker: str = "o",
):
    """Binned collapse-fraction curve: mean of boolean `df[flag]` vs binned `df[x]`.

    Bins ``df[x]`` on the `bins` edges via `stats.binned_fraction`, which keeps every bin and
    masks under-populated ones (``n < min_n``) to ``NaN``, then plots fraction vs bin midpoint
    (see `stats.binned_fraction` for the geometric-vs-arithmetic midpoint rule and the all-bins /
    finite-rows conventions). Masked bins render as gaps in the line. An optional dashed vertical
    line marks a `boundary` value (drawn once).

    Args:
        ax: Target axes.
        df: Frame carrying the `x` and `flag` columns.
        x: Column binned along the x axis.
        flag: Boolean/0-1 column reduced to a fraction per bin.
        bins: Monotonic bin edges.
        min_n: Minimum per-bin count; sparser bins render as gaps.
        boundary: Optional x value for a dashed vertical reference line.
        label: Optional series label.
        color: Line/marker colour.
        marker: Marker style.

    Returns:
        The plotted ``Line2D``.
    """
    binned = stats.binned_fraction(df, x, flag, bins=bins, min_n=min_n)
    line, = ax.plot(
        binned["x_mid"], binned["frac"], marker=marker, color=color, label=label
    )
    if boundary is not None:
        ax.axvline(boundary, ls=":", color="0.5")
    ax.set_xlabel(x)
    ax.set_ylabel(f"fraction {flag}")
    return line


def identity_scatter(
    ax,
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    color: str | None = None,
    color_by: str | None = None,
    cmap: str = "plasma",
    s: int = 10,
    alpha: float = 0.4,
    stat: str = "pearson",
    diag_lim: tuple[float, float] | None = (0.0, 1.0),
    title_stats: bool = True,
    textbox: bool = False,
    extra_lines: Sequence[str] | None = None,
):
    """Scatter ``df[x]`` vs ``df[y]`` with a dashed y=x reference line.

    Args:
        ax: Target axes.
        df, x, y: Data and the two columns.
        color: Fixed point colour (a single matplotlib colour) when not colouring by a column --
            e.g. a per-series colour in a faceted panel. Ignored if `color_by` is set.
        color_by: Optional column to colour points by; the mappable is then returned for a colorbar.
        cmap, s, alpha: Point styling.
        stat: Correlation to report -- ``"pearson"``, ``"spearman"``, ``"both"``, or ``None``.
        diag_lim: ``(lo, hi)`` extent of the dashed y=x line; ``None`` omits the line. Pass e.g.
            ``(0, data_max)`` for an unbounded axis.
        title_stats: Put ``r=.., n=..`` in the axis title (the historical default).
        textbox: Instead/also draw a top-left stats box (``n``, ``rho``/``r``) -- the notebooks'
            faceted-panel style. Honours `extra_lines`.
        extra_lines: Extra strings appended to the text box (e.g. group means).

    Returns:
        The scatter mappable when `color_by` is set (for a colorbar), else ``None``.
    """
    mappable = None
    if color_by is not None:
        mappable = ax.scatter(df[x], df[y], c=df[color_by], cmap=cmap, s=s, alpha=alpha)
    else:
        ax.scatter(df[x], df[y], s=s, alpha=alpha, color=color)
    if diag_lim is not None:
        ax.plot(list(diag_lim), list(diag_lim), "r--", lw=1)
    ax.set_xlabel(x)
    ax.set_ylabel(y)

    parts = []
    xy = df[[x, y]].dropna()
    n = len(xy)
    if stat and n >= 2:
        if stat in ("pearson", "both"):
            parts.append(f"r={np.corrcoef(xy[x], xy[y])[0, 1]:.2f}")
        if stat in ("spearman", "both"):
            from scipy.stats import spearmanr

            parts.append(f"ρ={spearmanr(xy[x], xy[y]).statistic:.2f}")
    if title_stats:
        r_txt = f"r={np.corrcoef(xy[x], xy[y])[0, 1]:.4f}, " if (n >= 2) else ""
        ax.set_title(f"{r_txt}n={n}")
    if textbox:
        lines = [f"n={n}", *parts, *(extra_lines or [])]
        ax.text(
            0.03, 0.97, "\n".join(lines), transform=ax.transAxes, va="top", fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
        )
    return mappable


def dose_response(
    ax,
    df: pd.DataFrame,
    x: str,
    y: str,
    group: str,
    *,
    order: Sequence | None = None,
    colors: dict | None = None,
    labels: dict | None = None,
    ci: bool = True,
    z: float = 1.96,
    marker: str = "o",
    band_alpha: float = 0.15,
    zero_line: bool = False,
    count_fn: Callable[[pd.DataFrame], int] | None = None,
) -> int:
    """Plot a mean(+/-CI) line over `x` for each `group`, sorted on `x`.

    For every group value (in `order`, else sorted-present) the frame is grouped by `x` and `y` is
    reduced to mean / std / count; the line is ``mean`` and the band is ``+/- z * std / sqrt(count)``
    (the notebooks' 95% CI at ``z=1.96``). This is the EAS/transfer "dose curve" primitive.

    Args:
        ax: Target axes.
        df, x, y, group: Data, x axis, value column, and the series-defining column.
        order: Group values to draw, in order (default: sorted unique present).
        colors, labels: Per-group-value colour / legend-label dicts (fall back to matplotlib
            defaults and the raw value).
        ci: Draw the shaded confidence band.
        z: CI multiplier (1.96 => ~95%).
        zero_line: Draw a light horizontal line at y=0 (EAS panels).
        count_fn: Optional ``callable(group_slice) -> int`` whose result is appended to the legend
            label as ``(n=.. )`` (e.g. a distinct-lineage count).

    Returns:
        The number of series actually drawn (groups with data).
    """
    if zero_line:
        ax.axhline(0.0, color="0.7", lw=0.8)
    gvals = order if order is not None else sorted(df[group].dropna().unique())
    drawn = 0
    for gv in gvals:
        s = df[df[group] == gv]
        if s.empty:
            continue
        g = s.groupby(x)[y].agg(["mean", "std", "count"]).sort_index()
        color = None if colors is None else colors.get(gv)
        label = gv if labels is None else labels.get(gv, gv)
        if count_fn is not None:
            label = f"{label} (n={count_fn(s)})"
        line, = ax.plot(g.index, g["mean"], marker=marker, color=color, label=label)
        if ci:
            e = z * g["std"].fillna(0.0) / np.sqrt(g["count"])
            ax.fill_between(
                g.index, g["mean"] - e, g["mean"] + e, color=line.get_color(), alpha=band_alpha
            )
        drawn += 1
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    return drawn


def strip_box(
    ax,
    groups: Sequence[Sequence[float]],
    labels: Sequence[str],
    *,
    colors: Sequence | None = None,
    zero_line: bool = False,
    box_width: float = 0.5,
    jitter: float = 0.09,
    s: int = 14,
    alpha: float = 0.5,
    showfliers: bool = False,
    rotation: float = 20,
) -> int:
    """Boxplot + jittered scatter strip, one category per entry in `groups`.

    Each ``groups[i]`` is an array of values (NaNs dropped) drawn at x=i as a box plus a jittered
    point strip in ``colors[i]``. Categories with no finite values are skipped. This is the
    distribution-per-group panel shared by the EAS and distillation notebooks.

    Args:
        ax: Target axes.
        groups: Per-category value arrays.
        labels: Per-category x tick labels.
        colors: Per-category point colours (default: matplotlib cycle).
        zero_line: Draw a light horizontal line at y=0.
        box_width, jitter, s, alpha, showfliers, rotation: Box/point/tick styling.

    Returns:
        The number of categories actually drawn (non-empty).
    """
    if zero_line:
        ax.axhline(0.0, color="0.7", lw=0.8)
    drawn = 0
    for i, vals in enumerate(groups):
        vals = np.asarray(vals, dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            continue
        ax.boxplot([vals], positions=[i], widths=box_width, showfliers=showfliers)
        col = None if colors is None else colors[i]
        offsets = np.linspace(-jitter, jitter, len(vals)) if len(vals) > 1 else np.zeros(1)
        ax.scatter(i + offsets, vals, s=s, alpha=alpha, color=col)
        drawn += 1
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=rotation, ha="right")
    return drawn


# --------------------------------------------------------------------------------------
# Comparison-table styling + the systematic raw-name -> unicode-math label mapping.
# --------------------------------------------------------------------------------------

# Matrix objects of the LoRA decomposition, longest-first so composites win the regex.
_MATH_OBJECTS: list[tuple[str, str]] = [
    ("B_Afinal", "B·A<sub>f</sub>"), ("B_Ainit", "B·A<sub>0</sub>"), ("B_dA", "B·ΔA"),
    ("A_init", "A<sub>0</sub>"), ("A_final", "A<sub>f</sub>"),
    ("dA", "ΔA"), ("dW", "ΔW"), ("BA", "BA"), ("W0", "W<sub>0</sub>"), ("B", "B"),
]
# Styler tables are HTML, so subscripts are real <sub> tags — unicode has no
# subscript F/f, but HTML does not care. (math_label therefore targets HTML display
# only; do not feed its output to matplotlib.)
_NORM_SUB = {"spec": "<sub>2</sub>", "fro": "<sub>F</sub>"}
_OBJ_ALT = "|".join(o for o, _ in _MATH_OBJECTS)

# Token substitutions for names that are not pure norm expressions. Word-boundary
# regexes for identifier-like tokens; plain replaces for parenthesised suffixes.
_MATH_TOKENS: list[tuple[str, str]] = [
    (r"\btheta_fro\b", "θ<sub>F</sub>"), (r"\btheta_spec\b", "θ<sub>2</sub>"),
    (r"\bauc_fro\b", "AUC (F)"), (r"\bauc_spec\b", "AUC (2)"),
    (r"\bslope_fro\b", "θ<sub>F</sub> slope"), (r"\bslope_spec\b", "θ<sub>2</sub> slope"),
    (r"\bdW_spec\b", "‖ΔW‖<sub>2</sub>"), (r"\bdW_fro\b", "‖ΔW‖<sub>F</sub>"),
    (r"\bdw_spec\b", "dw<sub>2</sub>"), (r"\bdw_fro\b", "dw<sub>F</sub>"),
    (r"\bB_spec\b", "‖B‖<sub>2</sub>"), (r"\bB_fro\b", "‖B‖<sub>F</sub>"),
    (r"\balpha\b", "α"), (r"\btheta\b", "θ"), (r"\brank\b", "r"),
    (r"\bW0\b", "W<sub>0</sub>"), (r"\bW_final\b", "W<sub>f</sub>"),
    (r"\bA_init\b", "A<sub>0</sub>"), (r"\bA_final\b", "A<sub>f</sub>"),
]
_PAREN_TOKENS = [("(fro)", "(F)"), ("(spec)", "(2)")]


def math_label(name, extra: dict | None = None) -> str:
    """Systematic display label for a raw column/index name.

    Resolution order: (1) an ``extra`` override, verbatim; (2) full-match norm
    patterns over the known decomposition objects — ``"dA spec"``/``"dA_fro"``/
    ``"dA (fro)"`` → ``"‖ΔA‖_2"``-style, plus the ratio forms ``"dW fro/spec"`` and
    ``"dW/W0 spec"``; (3) token substitutions (``alpha`` → ``α``, ``theta_fro`` →
    ``θ_F``, …). Anything unmatched passes through unchanged, so the mapping is safe
    to apply to every label of every table.
    """
    import re

    name = str(name)
    if extra and name in extra:
        return extra[name]
    m = re.fullmatch(rf"({_OBJ_ALT})[ _]\(?(spec|fro)\)?", name)
    if m:
        obj = dict(_MATH_OBJECTS)[m[1]]
        return f"‖{obj}‖{_NORM_SUB[m[2]]}"
    m = re.fullmatch(rf"({_OBJ_ALT}) (spec|fro)/(spec|fro)", name)
    if m:
        obj = dict(_MATH_OBJECTS)[m[1]]
        return f"‖{obj}‖{_NORM_SUB[m[2]]}/‖{obj}‖{_NORM_SUB[m[3]]}"
    m = re.fullmatch(rf"({_OBJ_ALT})/({_OBJ_ALT}) (spec|fro)", name)
    if m:
        objs = dict(_MATH_OBJECTS)
        return f"‖{objs[m[1]]}‖{_NORM_SUB[m[3]]}/‖{objs[m[2]]}‖{_NORM_SUB[m[3]]}"
    for pat, rep in _MATH_TOKENS:
        name = re.sub(pat, rep, name)
    for pat, rep in _PAREN_TOKENS:
        name = name.replace(pat, rep)
    return name


def relabel(df: pd.DataFrame, extra: dict | None = None) -> pd.DataFrame:
    """Copy of `df` with `math_label` applied to column labels and index *names*.

    Index values are data, not labels, and are left alone.
    """
    out = df.rename(columns=lambda c: math_label(c, extra))
    names = [math_label(n, extra) if n is not None else None for n in out.index.names]
    if any(n is not None for n in names):
        out = out.rename_axis(names)
    return out


def compare_table(
    df: pd.DataFrame,
    fmt: str = "{:.3f}",
    cmap: str = "RdBu_r",
    axis: int = 0,
    vmin: float | None = None,
    vmax: float | None = None,
    subset: Sequence[str] | None = None,
    math: bool = True,
    labels: dict | None = None,
):
    """Comparison styling for a table: per-column colour gradient over numeric cells.

    The one convention for "values meant to be compared" tables in the analysis
    notebooks (the §5 factor-fits table of ``nors_asks.ipynb`` is the reference
    look): a diverging ``RdBu_r`` `background_gradient` applied down each column
    (``axis=0``), so within a column red = high and blue = low. Non-numeric columns
    (labels, booleans, pre-formatted strings) are left completely untouched — no
    gradient, no number formatting.

    Args:
        df: The table. May mix numeric and non-numeric columns.
        fmt: ``str.format`` pattern for numeric cells.
        cmap: Matplotlib colormap name; diverging by convention.
        axis: Gradient axis (0 = compare down each column, the default convention;
            ``None`` = one gradient over the whole selection).
        vmin: Optional fixed gradient floor. When neither `vmin` nor `vmax` is
            given, the gradient is TABLE-ANCHORED: one shared range — the min and
            max over every styled cell of this table — so colours are comparable
            across columns and only the table's own extremes hit the endpoint
            colours. (Plain per-column min-max would paint every column's minimum
            as extreme blue and maximum as extreme red — a 2-row table becomes
            all saturated endpoints and near-equal values get maximal contrast.)
        vmax: Optional fixed gradient ceiling (with `vmin`, a caller-fixed
            range, e.g. to share a scale across several tables).
        subset: Columns to receive the *gradient*; default = every numeric
            (non-bool) column. Named in the RAW (pre-`math_label`) vocabulary.
            Number formatting is independent of this: ``fmt`` applies to every
            float column (ints keep their integer rendering), so un-gradiented
            companion columns (e.g. SEs) still render cleanly.
        math: Apply the systematic `math_label` mapping to column labels and index
            names before styling (unicode math: ``‖ΔW‖_F``, ``α``, ``θ_F`` — the
            portable convention; renders identically in every notebook frontend).
        labels: Per-call label overrides for names the systematic rules don't
            cover, raw name -> display label (implies the mapping for those keys
            even when it would otherwise pass through).

    Returns:
        A ``pandas.io.formats.style.Styler`` ready for ``display``.
    """
    if math or labels:
        renames = {c: math_label(c, labels) for c in df.columns}
        if subset is not None:
            subset = [renames.get(c, c) for c in subset]
        df = relabel(df, labels)
    if subset is None:
        subset = [c for c in df.select_dtypes(include="number").columns
                  if df[c].dtype != bool]
    subset = [c for c in subset if c in df.columns]
    float_cols = list(df.select_dtypes(include="floating").columns)
    styler = df.style
    if float_cols:
        styler = styler.format(fmt, subset=float_cols, na_rep="–")
    if subset:
        if vmin is None and vmax is None:
            # table-anchored: one shared range over every styled cell
            vals = df[subset].apply(pd.to_numeric, errors="coerce")
            lo, hi = vals.min().min(), vals.max().max()
            if pd.notna(lo) and pd.notna(hi) and lo != hi:
                vmin, vmax = float(lo), float(hi)
        styler = styler.background_gradient(
            cmap=cmap, axis=axis, subset=subset, vmin=vmin, vmax=vmax
        )
    return styler
