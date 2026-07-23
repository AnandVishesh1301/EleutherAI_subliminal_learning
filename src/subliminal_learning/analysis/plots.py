"""Annotated-heatmap and identity-scatter primitives shared by the analysis notebooks.

This is the ONE module in ``subliminal_learning.analysis`` that imports matplotlib. `facet_heatmaps`
is the canonical replacement for the two drifted ``facet_heatmaps`` copies that lived in the
teacher and steerability notebooks -- a superset carrying both copies' features (``agg``, ``title``,
``show_n`` counts, shared axis orders, and a generalized red-box ``highlight``). Figures are
returned, never shown, so notebook callers keep their inline-display behaviour.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.image import AxesImage


def _fmt_tick(x) -> str:
    """Format a heatmap tick label: numbers with ``%g``, everything else via ``str``."""
    if isinstance(x, (int, float, np.number)) and not isinstance(x, bool):
        return f"{x:g}"
    return str(x)


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
    Categorical, numeric/lexical sort) always keep only present values. The facet axis passes
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


def annotated_heatmap(
    ax,
    grid: pd.DataFrame,
    *,
    cmap: str = "viridis",
    vmin: float = 0.0,
    vmax: float = 1.0,
    counts: pd.DataFrame | None = None,
    fmt: str = "{:.2f}",
    origin: str = "lower",
    tick_fontsize: int = 8,
    cell_fontsize: int = 7,
) -> AxesImage:
    """Draw ``grid`` as an imshow with per-cell value labels; return the image.

    Ticks come from ``grid.columns`` / ``grid.index`` (numbers ``%g``-formatted, strings left as
    is). Each non-NaN cell is labelled with ``fmt`` in white below the ``(vmin+vmax)/2`` midpoint
    and black above. When `counts` (a same-shape frame) is given, each label gets a ``\\nn{count}``
    suffix. The axis grid is turned off.
    """
    values = np.asarray(grid.values, dtype=float)
    im = ax.imshow(values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, origin=origin)
    ax.set_xticks(range(len(grid.columns)))
    ax.set_xticklabels(
        [_fmt_tick(c) for c in grid.columns], rotation=55, ha="right", fontsize=tick_fontsize
    )
    ax.set_yticks(range(len(grid.index)))
    ax.set_yticklabels([_fmt_tick(r) for r in grid.index], fontsize=tick_fontsize)
    mid = (vmin + vmax) / 2
    cnt = None if counts is None else np.asarray(counts.values, dtype=float)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            if np.isnan(v):
                continue
            txt = fmt.format(v)
            if cnt is not None and not np.isnan(cnt[i, j]):
                txt += f"\nn{int(cnt[i, j])}"
            ax.text(
                j, i, txt, ha="center", va="center",
                color="white" if v < mid else "black", fontsize=cell_fontsize,
            )
    ax.grid(False)
    return im


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
):
    """One heatmap panel per sorted ``facet`` value; each a ``row``x``col`` grid of ``agg(val)``.

    Every panel is reindexed to the SAME shared axis orders so they line up. Axis ordering: an
    ``orders`` dict ``{axis_name: sequence}`` overrides; otherwise ordered-Categorical axes use
    their category order; otherwise numeric sort with a lexical fallback (see `_axis_order`).
    ``show_n`` overlays per-cell counts. ``highlight`` is an optional ``(row_value, col_value,
    facet_value)`` triple drawing a red rectangle around that cell in the matching panel. Returns
    the Figure (callers display it inline; this never calls ``plt.show``).
    """
    fvs = _axis_order(df, facet, orders, restrict_to_present=True)
    rvals = _axis_order(df, row, orders, restrict_to_present=False)
    cvals = _axis_order(df, col, orders, restrict_to_present=False)
    n = max(len(fvs), 1)
    fig, axs = plt.subplots(
        1, n, figsize=(3.5 * n + 1, 0.42 * len(rvals) + 2.4), squeeze=False
    )
    im = None
    for k, fv in enumerate(fvs):
        ax = axs[0][k]
        sub = df[df[facet] == fv]
        if len(sub):
            g = (
                sub.pivot_table(index=row, columns=col, values=val, aggfunc=agg, observed=True)
                .reindex(index=rvals, columns=cvals)
            )
        else:  # facet values are present-restricted, but guard the empty pivot regardless.
            g = pd.DataFrame(np.nan, index=rvals, columns=cvals)
        counts = None
        if show_n and len(sub):
            counts = (
                sub.groupby([row, col], observed=True).size().unstack(col)
                .reindex(index=rvals, columns=cvals)
            )
        im = annotated_heatmap(
            ax, g, cmap=cmap, vmin=vmin, vmax=vmax, counts=counts, cell_fontsize=6.5
        )
        ax.set_xlabel(col)
        ax.set_ylabel(row if k == 0 else "")
        ax.set_title(_facet_title(facet, fv), fontsize=9)
        if highlight is not None:
            hr, hc, hf = highlight
            if hf == fv and hr in rvals and hc in cvals:
                yi, xi = rvals.index(hr), cvals.index(hc)
                ax.add_patch(
                    plt.Rectangle((xi - .5, yi - .5), 1, 1, fill=False, edgecolor="red", lw=2.5)
                )
    if im is not None:
        fig.colorbar(im, ax=axs[0].tolist(), shrink=0.7, label=f"{agg} {val}")
    fig.suptitle(
        title or f"{agg} {val}: {row} x {col}, faceted by {facet}", y=1.03, fontsize=10
    )
    return fig


def identity_scatter(
    ax,
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    color_by: str | None = None,
    cmap: str = "plasma",
    s: int = 10,
    alpha: float = 0.4,
):
    """Scatter ``df[x]`` vs ``df[y]`` with a dashed red y=x reference line over ``[0, 1]``.

    Axis labels come from the column names and the title is annotated with the Pearson ``r`` and
    ``n``. When `color_by` is given, points are coloured by that column and the mappable is
    returned so the caller can attach a colorbar; otherwise ``None`` is returned.
    """
    mappable = None
    if color_by is not None:
        mappable = ax.scatter(df[x], df[y], c=df[color_by], cmap=cmap, s=s, alpha=alpha)
    else:
        ax.scatter(df[x], df[y], s=s, alpha=alpha)
    ax.plot([0, 1], [0, 1], "r--", lw=1)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    r = np.corrcoef(df[x], df[y])[0, 1]
    ax.set_title(f"r={r:.4f}, n={len(df)}")
    return mappable
