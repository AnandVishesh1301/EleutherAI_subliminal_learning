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

from subliminal_learning.analysis.plots import (  # noqa: E402
    annotated_heatmap,
    facet_heatmaps,
    identity_scatter,
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
