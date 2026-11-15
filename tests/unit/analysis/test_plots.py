"""Smoke + behaviour tests for ``subliminal_learning.analysis.plots`` (headless Agg, no GPU).

matplotlib is forced onto the non-interactive Agg backend before pyplot is imported, so these
run without a display. Tests assert on artists (text labels, tick order, Rectangle patches)
rather than pixels.

Tests for the house-style plot templates in `analysis.plots` (KLadder, house_grid, ...).

`analysis.plots.plain_log_ticks`: plain %g labels on log axes, decade + 2/5 majors, no minors.
"""

from __future__ import annotations

from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Rectangle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from subliminal_learning.analysis import plots as aplots
from subliminal_learning.analysis import plots as p  # noqa: E402
from subliminal_learning.analysis.plots import (
    OKABE_ITO,
    annotated_heatmap,
    compare_table,
    ecdf_plot,
    fraction_curve,
    heatmap_figure,
    math_label,
    dose_response,
    facet_grid_heatmaps,
    facet_heatmaps,
    identity_scatter,
    palette,
    right_legend,
    strip_box,
)


# -------------------- from test_analysis_plots.py --------------------

matplotlib.use("Agg")

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

def test_annotated_heatmap_annotate_false_drops_labels_but_keeps_boxes():
    grid = pd.DataFrame([[5.0, 20000.0]], index=[1], columns=["a", "b"])
    fig, ax = plt.subplots()
    annotated_heatmap(ax, grid, vmax=20000, annotate=False,
                      box_predicate=lambda v: v >= 10000)
    assert len(ax.texts) == 0
    assert len([p for p in ax.patches if isinstance(p, Rectangle) and not p.get_fill()]) == 1
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

def test_strip_box_skips_empty_groups():
    fig, ax = plt.subplots()
    drawn = strip_box(ax, [[0.1, 0.2, 0.3], [], [0.5]], ["a", "b", "c"])
    assert drawn == 2  # empty middle group skipped
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert labels == ["a", "b", "c"]
    plt.close(fig)

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

def test_math_label_norm_patterns():
    assert math_label("A_init spec") == "‖A<sub>0</sub>‖<sub>2</sub>"
    assert math_label("dA_fro") == "‖ΔA‖<sub>F</sub>"
    assert math_label("dW (spec)") == "‖ΔW‖<sub>2</sub>"
    assert math_label("B_Afinal fro") == "‖B·A<sub>f</sub>‖<sub>F</sub>"
    assert math_label("dW fro/spec") == "‖ΔW‖<sub>F</sub>/‖ΔW‖<sub>2</sub>"
    assert math_label("dW/W0 spec") == "‖ΔW‖<sub>2</sub>/‖W<sub>0</sub>‖<sub>2</sub>"

def test_math_label_tokens_passthrough_and_extra():
    assert math_label("alpha") == "α"
    assert math_label("theta_fro") == "θ<sub>F</sub>"
    assert math_label("slope_fro (rel)") == "θ<sub>F</sub> slope (rel)"
    assert math_label("n_ranks") == "n_ranks"  # no boundary inside identifiers
    assert math_label("completely unrelated") == "completely unrelated"
    assert math_label("median_ratio", {"median_ratio": "custom"}) == "custom"

def test_compare_table_applies_math_labels_and_remaps_subset():
    df = pd.DataFrame({"alpha": [1.0, 2.0], "dW_fro": [0.1, 0.2], "note": ["x", "y"]})
    html = compare_table(df, subset=["dW_fro"]).to_html()
    assert "‖ΔW‖<sub>F</sub>" in html and ">α<" in html  # labels mapped
    css = html.split("</style>")[0]
    assert "_col1" in css and "_col0" not in css  # gradient followed the renamed subset
    plain = compare_table(df, math=False).to_html()
    assert "dW_fro" in plain  # opt-out preserves raw names

def test_palette_qualitative_deterministic_and_cycles():
    items = list("abcdefghijkl")  # 12 items > tab10's 10 colours
    pal = palette(items, cmap="tab10")
    # deterministic in item order + cycles: item 10 reuses colour 0, item 11 reuses colour 1.
    assert pal["a"] == pal["k"]
    assert pal["b"] == pal["l"]
    assert pal["a"] != pal["b"]
    # stable regardless of neighbours.
    assert palette(["z", "a"], cmap="tab10")["z"] == palette(["z"], cmap="tab10")["z"]

def test_palette_continuous_samples_range():
    pal = palette(["lo", "mid", "hi"], cmap="viridis")
    assert len(pal) == 3
    assert pal["lo"] != pal["hi"]
    # single item -> the 0.0 end, no crash.
    assert len(palette(["only"], cmap="viridis")) == 1

def test_okabe_ito_has_eight_hexes():
    assert len(OKABE_ITO) == 8
    assert OKABE_ITO[0] == "#E69F00" and OKABE_ITO[-1] == "#000000"

def test_right_legend_anchor_and_frameless():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], label="series")
    leg = right_legend(ax, title="T", anchor=(1.05, 0.5))
    assert leg is not None
    assert leg.get_title().get_text() == "T"
    assert leg._get_loc() is not None
    plt.close(fig)

def test_heatmap_figure_auto_ranges_from_data():
    grid = pd.DataFrame([[2.0, 4.0], [6.0, 8.0]], index=[1, 2], columns=["a", "b"])
    fig, ax, im = heatmap_figure(grid)
    # auto-range picks nanmin/nanmax, NOT annotated_heatmap's fixed 0..1.
    assert im.get_clim() == (2.0, 8.0)
    plt.close(fig)

def test_heatmap_figure_explicit_vmin_vmax_wins():
    grid = pd.DataFrame([[2.0, 8.0]], index=[1], columns=["a", "b"])
    fig, ax, im = heatmap_figure(grid, vmin=0.0, vmax=10.0)
    assert im.get_clim() == (0.0, 10.0)
    plt.close(fig)

def test_heatmap_figure_all_nan_safe():
    grid = pd.DataFrame([[np.nan, np.nan]], index=[1], columns=["a", "b"])
    fig, ax, im = heatmap_figure(grid)  # must not crash
    assert im.get_clim() == (0.0, 1.0)  # falls back to fixed range
    plt.close(fig)

def test_ecdf_plot_small_array():
    fig, ax = plt.subplots()
    lines = ecdf_plot(ax, [3.0, 1.0, np.nan, 2.0])
    assert lines is not None
    x, y = lines[0].get_data()
    assert list(x) == [1.0, 2.0, 3.0]  # sorted, NaN dropped
    assert np.allclose(y, [1 / 3, 2 / 3, 1.0])
    plt.close(fig)

def test_ecdf_plot_empty_draws_nothing():
    fig, ax = plt.subplots()
    assert ecdf_plot(ax, [np.nan, np.nan]) is None
    assert len(ax.lines) == 0
    plt.close(fig)

def test_fraction_curve_draws_line_and_boundary():
    df = pd.DataFrame({
        "k": [1, 2, 3, 4, 5, 6, 20, 30],
        "dead": [1, 1, 1, 1, 1, 0, 1, 0],
    })
    fig, ax = plt.subplots()
    line = fraction_curve(ax, df, "k", "dead", bins=[1, 10, 100], min_n=5, boundary=10.0)
    assert line is not None
    # one dashed vertical boundary line drawn.
    vlines = [ln for ln in ax.lines if ln.get_linestyle() == ":"]
    assert len(vlines) == 1
    plt.close(fig)

def test_fraction_curve_plots_gap_for_sparse_bin():
    # bins [1,10,100]: bin 0 has 6 points (>= min_n), bin 1 has 2 (< min_n) -> NaN gap.
    df = pd.DataFrame({
        "k": [1, 2, 3, 4, 5, 6, 20, 30],
        "dead": [1, 1, 1, 1, 1, 0, 1, 0],
    })
    fig, ax = plt.subplots()
    line = fraction_curve(ax, df, "k", "dead", bins=[1, 10, 100], min_n=5)
    _, y = line.get_data()
    # both bins present (all bins kept) but the sparse one is a NaN gap.
    assert len(y) == 2
    assert y[0] == pytest.approx(5 / 6)
    assert np.isnan(y[1])
    plt.close(fig)

def test_compare_table_table_anchored_default():
    # one shared range across ALL styled cells: x spans the table's extremes,
    # y sits mid-range and must get no endpoint colour
    df = pd.DataFrame({"x": [0.0, 10.0], "y": [4.0, 6.0]})
    html = compare_table(df).to_html()
    css = html.split("</style>")[0]
    assert "#053061" in css and "#67001f" in css  # table extremes hit the ends
    y_ids = [f"row{i}_col1" for i in range(2)]
    y_rules = [r for r in css.split("}") if any(i in r for i in y_ids)]
    assert y_rules and all(
        "#053061" not in r and "#67001f" not in r for r in y_rules
    )  # mid-range column stays mid-shade


# -------------------- from test_analysis_plots_house.py --------------------

DENSE = p.KLadder(lo=0.0, hi=1.50, width=0.05, decimals=2, overflow_label=True)

SPARSE = p.KLadder(lo=0.2, hi=1.6, width=0.1, decimals=1)

def test_dense_ladder_labels_match_notebook():
    expected = [f"{lo:.2f}-{lo + 0.05:.2f}"
                for lo in np.round(np.arange(0.0, 1.50 + 1e-9, 0.05), 2)[:-1]] + [">1.50"]
    assert DENSE.labels == expected
    assert DENSE.labels[:2] == ["0.00-0.05", "0.05-0.10"]
    assert DENSE.labels[-2:] == ["1.45-1.50", ">1.50"]
    assert len(DENSE.labels) == 31

def test_sparse_ladder_labels_match_notebook():
    expected = [f"{lo:.1f}-{lo + 0.1:.1f}"
                for lo in np.round(np.arange(0.2, 1.60 + 1e-9, 0.1), 2)[:-1]]
    assert SPARSE.labels == expected
    assert SPARSE.labels[0] == "0.2-0.3"
    assert SPARSE.labels[-1] == "1.5-1.6"
    assert len(SPARSE.labels) == 14

def test_bin_is_left_closed_and_handles_overflow():
    out = DENSE.bin([0.25, 0.2499, 0.0, 1.6])
    assert list(out) == ["0.25-0.30", "0.20-0.25", "0.00-0.05", ">1.50"]
    assert list(SPARSE.bin([0.3, 0.29999])) == ["0.3-0.4", "0.2-0.3"]

def test_order_present_keeps_ladder_order():
    labs = DENSE.bin([0.7, 0.1, 0.7])
    assert DENSE.order_present(labs) == ["0.10-0.15", "0.70-0.75"]

def test_boundary_index():
    order = ["0.90-0.95", "0.95-1.00", "1.00-1.05", ">1.50"]
    assert DENSE.boundary_index(order) == 2
    assert DENSE.boundary_index(order, threshold=0.95) == 1
    assert DENSE.boundary_index(["0.10-0.15"]) == 1  # nothing above => off the top
    assert DENSE.boundary_index([">1.50"]) == 0

def test_row_pos_center_and_miss():
    order = ["0.20-0.25", "0.25-0.30"]
    assert DENSE.row_pos(0.275, order) == pytest.approx(1.0)
    assert DENSE.row_pos(0.20, order) == pytest.approx(-0.5)
    assert DENSE.row_pos(0.9, order) is None
    assert DENSE.row_pos(1.6, [">1.50"]) is None

def _df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = [
        {"row": r, "col": c, "facet": f, "val": float(rng.random())}
        for r in ("a", "b") for c in (1, 2) for f in (8, 16)
    ]
    return pd.DataFrame(rows)

def test_defaults_unchanged_without_new_kwargs():
    df = _df()
    fig = p.facet_grid_heatmaps(df, "row", "col", "val", col_facet="facet")
    panels = [a for a in fig.axes if a.get_label() != "<colorbar>"]
    assert len(panels) == 2
    assert [a.get_title() for a in panels] == ["facet=8", "facet=16"]
    assert not any(a.get_lines() for a in panels)
    assert [t.get_text() for t in panels[1].get_yticklabels()] == ["a", "b"]
    plt.close(fig)

def test_hlines_drawn_on_every_panel_with_one_label():
    df = _df()
    fig = p.facet_grid_heatmaps(
        df, "row", "col", "val", col_facet="facet",
        hlines=[{"y": 0.5, "color": "k", "ls": "--", "lw": 1.3, "label": "K/K* = 1"}],
    )
    panels = [a for a in fig.axes if a.get_label() != "<colorbar>"]
    assert all(len(a.get_lines()) == 1 for a in panels)
    labels = [ln.get_label() for a in panels for ln in a.get_lines()]
    assert sum(lab == "K/K* = 1" for lab in labels) == 1
    plt.close(fig)

def test_suppress_inner_yticks():
    df = _df()
    fig = p.facet_grid_heatmaps(
        df, "row", "col", "val", col_facet="facet", suppress_inner_yticks=True
    )
    panels = [a for a in fig.axes if a.get_label() != "<colorbar>"]
    assert [t.get_text() for t in panels[0].get_yticklabels()] == ["a", "b"]
    assert all(t.get_text() == "" for t in panels[1].get_yticklabels())
    plt.close(fig)

def _strip_df() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    rows = []
    for k in (0.22, 0.27, 0.61, 1.02):
        for alpha in (8, 16):
            for rank in (8, 32):
                rows.append({"K_rel": k, "alpha": alpha, "rank": rank,
                             "val": float(rng.random()), "test_prior": 0.013})
    return pd.DataFrame(rows)

def test_house_grid_smoke_with_and_without_kstar():
    df = _strip_df()
    fig = p.house_grid(df, DENSE, val="val", kstar_rel=0.63, vmax=1.0)
    assert isinstance(fig, Figure)
    panels = [a for a in fig.axes if a.get_label() != "<colorbar>"]
    labels = {ln.get_label() for a in panels for ln in a.get_lines()}
    assert "K/K* = 1" in labels and "arm integrity K*" in labels
    assert all(len(a.get_lines()) == 2 for a in panels)
    plt.close(fig)

    fig2 = p.house_grid(df, DENSE, val="val", kstar_rel=5.0, vmax=1.0)
    panels2 = [a for a in fig2.axes if a.get_label() != "<colorbar>"]
    labels2 = {ln.get_label() for a in panels2 for ln in a.get_lines()}
    assert "arm integrity K*" not in labels2
    assert all(len(a.get_lines()) == 1 for a in panels2)
    plt.close(fig2)

def test_facet_grid_counts_column_annotates_precounted_n():
    """``counts=`` annotates a column's n per cell; ``show_n`` counts rows; default is neither."""
    df = pd.DataFrame({
        "r": [1, 1, 2, 2], "c": [1, 2, 1, 2],
        "val": [0.1, 0.2, 0.3, 0.4], "n_scanned": [7, 9, 11, 13],
    })

    def cell_texts(fig):
        panels = [a for a in fig.axes if a.get_label() != "<colorbar>"]
        return {t.get_text() for a in panels for t in a.texts}

    fig = p.facet_grid_heatmaps(df, "r", "c", "val", counts="n_scanned")
    assert cell_texts(fig) == {"0.10\nn7", "0.20\nn9", "0.30\nn11", "0.40\nn13"}
    plt.close(fig)

    fig = p.facet_grid_heatmaps(df, "r", "c", "val", show_n=True)   # row counts, one row per cell
    assert cell_texts(fig) == {"0.10\nn1", "0.20\nn1", "0.30\nn1", "0.40\nn1"}
    plt.close(fig)

    fig = p.facet_grid_heatmaps(df, "r", "c", "val")                # inert default: no n suffix
    assert cell_texts(fig) == {"0.10", "0.20", "0.30", "0.40"}
    plt.close(fig)

def test_facet_grid_counts_matches_the_row_explosion_hack():
    """``counts=`` reproduces what repeating each row ``n`` times used to print."""
    df = pd.DataFrame({"r": [1, 2], "c": [1, 1], "val": [0.5, 0.5], "n": [3, 4]})
    fig_a = p.facet_grid_heatmaps(df, "r", "c", "val", counts="n")
    fig_b = p.facet_grid_heatmaps(df.loc[df.index.repeat(df.n)], "r", "c", "val", show_n=True)
    texts = [sorted(t.get_text()
                    for a in f.axes if a.get_label() != "<colorbar>" for t in a.texts)
             for f in (fig_a, fig_b)]
    assert texts[0] == texts[1] == ["0.50\nn3", "0.50\nn4"]
    plt.close(fig_a)
    plt.close(fig_b)

def test_grouped_bars_smoke():
    table = pd.DataFrame({"ds1": [1.0, 2.0], "ws1": [3.0, 4.0]}, index=["dog", "cat"])
    fig, ax = plt.subplots()
    p.grouped_bars(ax, table, colors=p.BASE_COLORS, fmt="{:.1f}")
    assert len(ax.patches) == 4
    assert [t.get_text() for t in ax.get_xticklabels()] == ["dog", "cat"]
    plt.close(fig)

def test_jitter_is_seeded_and_bounded():
    a = p.jitter(5, 0.1, seed=3)
    assert a.shape == (5,)
    assert np.all(np.abs(a) <= 0.1)
    assert np.allclose(a, p.jitter(5, 0.1, seed=3))

def test_palette_registries_have_documented_keys():
    assert set(p.ANIMAL_COLORS) == {"dog", "cat", "lion", "pig"}
    assert p.ANIMAL_COLORS["dog"] == "#0072B2"
    assert p.BASE_ORDER == ["original", "data-seed1", "data-seed2", "data-seed3",
                            "weight-seed1", "weight-seed2", "weight-seed3"]
    assert all(b in p.BASE_COLORS for b in p.BASE_ORDER)
    assert p.BASE_COLORS["original"] == "#000000"
    assert p.BASE_COLORS["ds1"] == p.BASE_COLORS["data-seed1"]
    assert len({p.BASE_COLORS[b] for b in p.BASE_ORDER}) == len(p.BASE_ORDER)

def test_ladder_below_lo_is_nan_not_the_first_bin():
    """With an overflow bucket only the TOP side folds; below-`lo` stays out of the ladder."""
    ladder = p.KLadder(0.2, 1.6, 0.1, decimals=1, overflow_label=True)
    assert list(ladder.bin([0.1, 1.7, 0.2])) == ["nan", ladder.overflow, "0.2-0.3"]


# -------------------- from test_plots_log_ticks.py --------------------

def test_plain_log_ticks_formats_majors_and_hides_minors():
    fig, ax = plt.subplots()
    ax.plot([0.15, 0.5, 2.6], [1, 2, 3])
    ax.set_xscale("log")
    ax.set_xlim(0.1, 3)
    aplots.plain_log_ticks(ax, axis="x")
    fig.canvas.draw()
    labels = [t.get_text() for t in ax.get_xticklabels() if t.get_text()]
    assert "0.2" in labels and "0.5" in labels and "1" in labels and "2" in labels
    assert not any("×" in lab or "^" in lab or "e-" in lab for lab in labels)
    assert all(t.get_text() == "" for t in ax.xaxis.get_minorticklabels())
    plt.close(fig)

def test_plain_log_ticks_leaves_linear_axes_alone():
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 2, 3])
    before = type(ax.xaxis.get_major_formatter())
    aplots.plain_log_ticks(ax)
    assert type(ax.xaxis.get_major_formatter()) is before
    plt.close(fig)
