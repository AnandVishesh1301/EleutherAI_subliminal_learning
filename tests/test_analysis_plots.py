"""Smoke + behaviour tests for ``subliminal_learning.analysis.plots`` (headless Agg, no GPU).

matplotlib is forced onto the non-interactive Agg backend before pyplot is imported, so these
run without a display. Tests assert on artists (text labels, tick order, Rectangle patches)
rather than pixels.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from matplotlib.colors import TwoSlopeNorm  # noqa: E402

from subliminal_learning.analysis.plots import (  # noqa: E402
    annotated_heatmap,
    compare_table,
    dose_response,
    facet_grid_heatmaps,
    facet_heatmaps,
    identity_scatter,
    strip_box,
)


# ------------------------- annotated_heatmap -------------------------


def test_annotated_heatmap_smoke_and_text_count():
    grid = pd.DataFrame(
        [[0.1, np.nan], [0.9, 0.5]], index=[1, 2], columns=["a", "b"]
    )
    fig, ax = plt.subplots()
    im = annotated_heatmap(ax, grid)
    assert im is not None
    # One text label per non-NaN cell (3 of 4).
    assert len(ax.texts) == 3
    plt.close(fig)


def test_annotated_heatmap_counts_suffix():
    grid = pd.DataFrame([[0.4]], index=[1], columns=["a"])
    counts = pd.DataFrame([[7]], index=[1], columns=["a"])
    fig, ax = plt.subplots()
    annotated_heatmap(ax, grid, counts=counts)
    assert ax.texts[0].get_text() == "0.40\nn7"
    plt.close(fig)


# ------------------------- facet_heatmaps -------------------------


def _facet_df():
    return pd.DataFrame(
        {
            "n_epochs": [1, 2, 1, 2, 1, 2],
            "lora_label": ["r8/α8", "r8/α8", "r16/α16", "r16/α16", "r8/α8", "r16/α16"],
            "learning_rate": [1e-4, 1e-4, 1e-4, 1e-4, 5e-4, 5e-4],
            "val": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        }
    )


def test_facet_heatmaps_one_panel_per_facet_value():
    fig = facet_heatmaps(_facet_df(), "n_epochs", "lora_label", "learning_rate", "val")
    panels = [ax for ax in fig.axes if ax.images]
    assert len(panels) == 2  # two learning rates
    plt.close(fig)


def test_facet_heatmaps_orders_respected():
    orders = {"lora_label": ["r16/α16", "r8/α8"]}
    fig = facet_heatmaps(
        _facet_df(), "n_epochs", "lora_label", "learning_rate", "val", orders=orders
    )
    panel = [ax for ax in fig.axes if ax.images][0]
    labels = [t.get_text() for t in panel.get_xticklabels()]
    assert labels == ["r16/α16", "r8/α8"]
    plt.close(fig)


def test_facet_heatmaps_highlight_one_rectangle_in_right_panel():
    df = _facet_df()
    fig = facet_heatmaps(
        df, "n_epochs", "lora_label", "learning_rate", "val",
        highlight=(1, "r8/α8", 1e-4),
    )
    panels = [ax for ax in fig.axes if ax.images]
    rects = [
        [p for p in ax.patches if isinstance(p, Rectangle) and not p.get_fill()]
        for ax in panels
    ]
    # Exactly one highlight rectangle, and it is in the lr=1e-4 panel (the first).
    assert sum(len(r) for r in rects) == 1
    assert len(rects[0]) == 1
    plt.close(fig)


def test_facet_heatmaps_absent_facet_value_no_empty_panel():
    df = _facet_df()
    # An absent facet value in `orders` must NOT spawn an empty panel (facet axis is
    # present-restricted regardless of source).
    fig = facet_heatmaps(
        df, "n_epochs", "lora_label", "learning_rate", "val",
        orders={"learning_rate": [1e-4, 5e-4, 1e-3]},
    )
    # 1e-3 has no data -> dropped, so only the two populated facets get panels.
    panels = [ax for ax in fig.axes if ax.images]
    assert len(panels) == 2
    plt.close(fig)


def test_facet_heatmaps_verbatim_row_col_orders_render_blank_cells():
    df = _facet_df()
    # `orders` on the row/col axes is honoured VERBATIM: values absent from the frame still
    # appear as (blank) heatmap rows/columns so the sparse-config coverage is visible.
    orders = {
        "n_epochs": [1, 2, 99],  # 99 absent -> blank row
        "lora_label": ["r8/α8", "r16/α16", "r32/α32"],  # r32/α32 absent -> blank col
    }
    fig = facet_heatmaps(
        df, "n_epochs", "lora_label", "learning_rate", "val", orders=orders,
    )
    panel = [ax for ax in fig.axes if ax.images][0]
    ylabels = [t.get_text() for t in panel.get_yticklabels()]
    xlabels = [t.get_text() for t in panel.get_xticklabels()]
    assert ylabels == ["1", "2", "99"]
    assert xlabels == ["r8/α8", "r16/α16", "r32/α32"]
    # The absent row/col carry no data; imshow masks those NaN cells (they render blank).
    mask = np.ma.getmaskarray(panel.images[0].get_array())
    assert mask[ylabels.index("99"), :].all()
    assert mask[:, xlabels.index("r32/α32")].all()
    plt.close(fig)


def test_facet_heatmaps_highlight_resolves_with_verbatim_orders():
    df = _facet_df()
    orders = {"n_epochs": [1, 2, 99], "lora_label": ["r8/α8", "r16/α16", "r32/α32"]}
    fig = facet_heatmaps(
        df, "n_epochs", "lora_label", "learning_rate", "val",
        orders=orders, highlight=(2, "r16/α16", 1e-4),
    )
    panels = [ax for ax in fig.axes if ax.images]
    rects = [
        p for ax in panels for p in ax.patches
        if isinstance(p, Rectangle) and not p.get_fill()
    ]
    assert len(rects) == 1
    plt.close(fig)


# ------------------------- identity_scatter -------------------------


def test_identity_scatter_smoke_no_color():
    df = pd.DataFrame({"x": [0.1, 0.2, 0.3], "y": [0.1, 0.25, 0.28]})
    fig, ax = plt.subplots()
    out = identity_scatter(ax, df, "x", "y")
    assert out is None
    assert "n=3" in ax.get_title()
    assert ax.get_xlabel() == "x"
    plt.close(fig)


def test_identity_scatter_returns_mappable_when_colored():
    df = pd.DataFrame({"x": [0.1, 0.9], "y": [0.2, 0.8], "c": [1.0, 2.0]})
    fig, ax = plt.subplots()
    out = identity_scatter(ax, df, "x", "y", color_by="c")
    assert out is not None  # mappable for a colorbar
    plt.close(fig)


def test_identity_scatter_textbox_spearman():
    df = pd.DataFrame({"x": [0.1, 0.2, 0.3, 0.4], "y": [0.15, 0.1, 0.35, 0.5]})
    fig, ax = plt.subplots()
    identity_scatter(ax, df, "x", "y", stat="spearman", textbox=True, title_stats=False)
    box = "\n".join(t.get_text() for t in ax.texts)
    assert "ρ=" in box and "n=4" in box
    assert ax.get_title() == ""  # title_stats off
    plt.close(fig)


def test_identity_scatter_diag_lim_none_omits_line():
    df = pd.DataFrame({"x": [0.1, 0.2], "y": [0.1, 0.3]})
    fig, ax = plt.subplots()
    identity_scatter(ax, df, "x", "y", diag_lim=None)
    assert len(ax.lines) == 0  # no y=x reference line
    plt.close(fig)


def test_identity_scatter_fixed_color():
    df = pd.DataFrame({"x": [0.1, 0.2], "y": [0.1, 0.3]})
    fig, ax = plt.subplots()
    out = identity_scatter(ax, df, "x", "y", color="#54a24b", diag_lim=None)
    assert out is None  # no mappable when not colouring by a column
    pc = ax.collections[0]  # the scatter
    # Compare RGB only (scatter applies alpha=0.4 to the 4th channel).
    assert np.allclose(pc.get_facecolor()[0][:3], plt.matplotlib.colors.to_rgba("#54a24b")[:3])
    plt.close(fig)


# ------------------------- annotated_heatmap (new capabilities) -------------------------


def test_annotated_heatmap_callable_fmt():
    grid = pd.DataFrame([[12000.0]], index=[1], columns=["a"])
    fig, ax = plt.subplots()
    annotated_heatmap(ax, grid, fmt=lambda v: f"{v / 1000:.0f}k", vmax=20000)
    assert ax.texts[0].get_text() == "12k"
    plt.close(fig)


def test_annotated_heatmap_mid_threshold_text_color():
    grid = pd.DataFrame([[0.2, 0.8]], index=[1], columns=["a", "b"])
    fig, ax = plt.subplots()
    annotated_heatmap(ax, grid, mid=0.55)
    colors = {t.get_text(): t.get_color() for t in ax.texts}
    assert colors["0.20"] == "white"  # below mid
    assert colors["0.80"] == "black"  # at/above mid
    plt.close(fig)


def test_annotated_heatmap_default_text_color_is_background_aware():
    # Blues: low value => pale cell (needs black text), high value => dark cell (needs white).
    grid = pd.DataFrame([[0.05, 0.95]], index=[1], columns=["a", "b"])
    fig, ax = plt.subplots()
    annotated_heatmap(ax, grid, cmap="Blues")  # no mid/text_color => luminance-based default
    colors = {t.get_text(): t.get_color() for t in ax.texts}
    assert colors["0.05"] == "black"  # pale background
    assert colors["0.95"] == "white"  # dark background
    plt.close(fig)


def test_annotated_heatmap_norm_used_for_color():
    grid = pd.DataFrame([[-1.0, 1.0]], index=[1], columns=["a", "b"])
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-1, vmax=1)
    fig, ax = plt.subplots()
    im = annotated_heatmap(ax, grid, cmap="RdBu_r", norm=norm, mid=999)
    assert im.norm is norm  # diverging scale honoured
    plt.close(fig)


def test_annotated_heatmap_separate_color_frame():
    grid = pd.DataFrame([[100.0]], index=[1], columns=["a"])  # labels
    color = pd.DataFrame([[2.0]], index=[1], columns=["a"])  # log-scaled colour
    fig, ax = plt.subplots()
    im = annotated_heatmap(ax, grid, color=color, vmax=10)
    assert float(np.ma.getdata(im.get_array())[0, 0]) == 2.0  # coloured by `color`
    assert ax.texts[0].get_text() == "100.00"  # labelled by `grid`
    plt.close(fig)


def test_annotated_heatmap_box_predicate_boxes_and_bolds():
    grid = pd.DataFrame([[5.0, 20000.0]], index=[1], columns=["a", "b"])
    fig, ax = plt.subplots()
    annotated_heatmap(ax, grid, vmax=20000, box_predicate=lambda v: v >= 10000)
    rects = [p for p in ax.patches if isinstance(p, Rectangle) and not p.get_fill()]
    assert len(rects) == 1  # only the >=10k cell
    bold = [t for t in ax.texts if t.get_fontweight() == "bold"]
    assert len(bold) == 1
    plt.close(fig)


def test_annotated_heatmap_empty_facecolor_when_all_nan():
    grid = pd.DataFrame([[np.nan, np.nan]], index=[1], columns=["a", "b"])
    fig, ax = plt.subplots()
    annotated_heatmap(ax, grid, empty_facecolor="#f2f2f2")
    assert ax.get_facecolor() == plt.matplotlib.colors.to_rgba("#f2f2f2")
    plt.close(fig)


def test_annotated_heatmap_hide_ticks():
    grid = pd.DataFrame([[0.5]], index=[1], columns=["a"])
    fig, ax = plt.subplots()
    annotated_heatmap(ax, grid, show_ticks=False)
    assert list(ax.get_xticks()) == [] and list(ax.get_yticks()) == []
    plt.close(fig)


# ------------------------- facet_grid_heatmaps -------------------------


def _grid_df():
    # 2 row-facets x 2 col-facets, each a 2x2 (row x col) panel.
    rows = []
    for rf in ("A", "B"):
        for cf in (1, 2):
            for r in (10, 20):
                for c in ("x", "y"):
                    rows.append({"rf": rf, "cf": cf, "r": r, "c": c, "v": 0.5})
    return pd.DataFrame(rows)


def test_facet_grid_heatmaps_r_by_c_panel_count():
    fig = facet_grid_heatmaps(
        _grid_df(), "r", "c", "v", row_facet="rf", col_facet="cf"
    )
    panels = [ax for ax in fig.axes if ax.images]
    assert len(panels) == 4  # 2 row-facets x 2 col-facets
    plt.close(fig)


def test_facet_grid_heatmaps_single_col_facet_is_1xN():
    fig = facet_grid_heatmaps(_grid_df(), "r", "c", "v", col_facet="cf")
    panels = [ax for ax in fig.axes if ax.images]
    assert len(panels) == 2  # one row of panels
    plt.close(fig)


def test_facet_grid_heatmaps_single_row_facet_is_Nx1():
    fig = facet_grid_heatmaps(_grid_df(), "r", "c", "v", row_facet="rf")
    panels = [ax for ax in fig.axes if ax.images]
    assert len(panels) == 2  # one column of panels
    plt.close(fig)


def test_facet_grid_heatmaps_box_predicate_uses_box_color():
    fig = facet_grid_heatmaps(
        _grid_df(), "r", "c", "v", col_facet="cf",
        box_predicate=lambda v: v >= 0.0, box_color="#ff3b3b",  # every populated cell
    )
    rects = [
        p for ax in fig.axes for p in ax.patches
        if isinstance(p, Rectangle) and not p.get_fill()
    ]
    assert rects  # at least one boxed cell
    assert all(
        np.allclose(r.get_edgecolor(), plt.matplotlib.colors.to_rgba("#ff3b3b")) for r in rects
    )
    plt.close(fig)


def test_facet_heatmaps_wrapper_forwards_cbar_label():
    fig = facet_heatmaps(_facet_df(), "n_epochs", "lora_label", "learning_rate", "val",
                         cbar_label="custom label")
    cbars = [ax for ax in fig.axes if ax.get_ylabel() == "custom label"]
    assert len(cbars) == 1
    plt.close(fig)


def test_facet_grid_heatmaps_highlight_in_matching_panel_only():
    fig = facet_grid_heatmaps(
        _grid_df(), "r", "c", "v", row_facet="rf", col_facet="cf",
        highlight=[{"rf": "A", "cf": 1, "r": 10, "c": "x"}],
    )
    rects = [
        p for ax in fig.axes for p in ax.patches
        if isinstance(p, Rectangle) and not p.get_fill()
    ]
    assert len(rects) == 1
    plt.close(fig)


def test_facet_grid_heatmaps_highlight_per_box_color_and_lw():
    fig = facet_grid_heatmaps(
        _grid_df(), "r", "c", "v", row_facet="rf", col_facet="cf",
        highlight=[
            {"rf": "A", "cf": 1, "r": 10, "c": "x", "color": "gold", "lw": 3.2},
            {"rf": "A", "cf": 1, "r": 20, "c": "y"},  # defaults -> red, 2.5
        ],
    )
    rects = [
        p for ax in fig.axes for p in ax.patches
        if isinstance(p, Rectangle) and not p.get_fill()
    ]
    styles = {p.get_edgecolor(): round(p.get_linewidth(), 1) for p in rects}
    from matplotlib.colors import to_rgba
    assert styles[to_rgba("gold")] == 3.2  # rank-1 box uses its own colour/width
    assert styles[to_rgba("red")] == 2.5   # runner-up falls back to the defaults
    plt.close(fig)


# ------------------------- dose_response -------------------------


def test_dose_response_one_line_per_group_and_count():
    df = pd.DataFrame({
        "x": [1, 2, 1, 2, 1, 2],
        "g": ["a", "a", "b", "b", "b", "b"],
        "y": [0.1, 0.2, 0.3, 0.4, 0.35, 0.45],
    })
    fig, ax = plt.subplots()
    drawn = dose_response(ax, df, "x", "y", "g")
    assert drawn == 2
    assert len([ln for ln in ax.lines]) == 2  # no zero-line here
    plt.close(fig)


def test_dose_response_zero_line_and_ci_band():
    df = pd.DataFrame({"x": [1, 1, 2, 2], "g": ["a", "a", "a", "a"], "y": [0.1, 0.3, 0.4, 0.6]})
    fig, ax = plt.subplots()
    dose_response(ax, df, "x", "y", "g", zero_line=True, ci=True)
    assert len(ax.collections) >= 1  # fill_between band exists
    assert any(ln.get_ydata()[0] == 0.0 for ln in ax.lines)  # zero line
    plt.close(fig)


# ------------------------- strip_box -------------------------


def test_strip_box_skips_empty_groups():
    fig, ax = plt.subplots()
    drawn = strip_box(ax, [[0.1, 0.2, 0.3], [], [0.5]], ["a", "b", "c"])
    assert drawn == 2  # empty middle group skipped
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert labels == ["a", "b", "c"]
    plt.close(fig)


# ------------------------- compare_table -------------------------


def test_compare_table_colors_numeric_columns_only():
    df = pd.DataFrame({"label": ["a", "b"], "x": [1.0, 2.0], "ok": [True, False]})
    styler = compare_table(df)
    html = styler.to_html()
    assert "background-color" in html  # the numeric column got a gradient
    assert "1.000" in html and "2.000" in html  # numeric formatting applied
    assert ">a<" in html and ">True<" in html  # label/bool cells untouched


def test_compare_table_nan_safe_and_subset_override():
    df = pd.DataFrame({"x": [1.0, np.nan], "y": [3.0, 4.0]})
    html_all = compare_table(df).to_html()
    assert "–" in html_all  # na_rep rendered, no exception
    styler = compare_table(df, subset=["y"], vmin=0.0, vmax=10.0)
    html = styler.to_html()
    # every float column is formatted; the gradient is confined to the subset (y = col1)
    assert "3.000" in html and "1.000" in html and "1.000000" not in html
    css = html.split("</style>")[0]
    assert "_col1" in css and "_col0" not in css


def test_compare_table_empty_numeric_returns_plain_styler():
    df = pd.DataFrame({"label": ["a", "b"]})
    html = compare_table(df).to_html()
    assert "background-color" not in html
