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

Figures/artists are returned, never shown, so notebook callers keep their inline-display behaviour.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.image import AxesImage


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
            txt = _apply_fmt(fmt, v)
            if cnt is not None and not np.isnan(cnt[i, j]):
                txt += f"\nn{int(cnt[i, j])}"
            boxed = box_predicate is not None and box_predicate(v)
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
        show_n: Overlay per-cell counts (``\\nn{count}``).
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
            counts = _panel_counts(sub, row, col, rvals, cvals) if (show_n and len(sub)) else None
            im = annotated_heatmap(
                ax, g, cmap=cmap, vmin=vmin, vmax=vmax, norm=norm, color=color, counts=counts,
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
