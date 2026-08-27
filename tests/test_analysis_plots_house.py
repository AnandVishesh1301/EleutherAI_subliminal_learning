"""Tests for the house-style plot templates in `analysis.plots` (KLadder, house_grid, ...)."""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from matplotlib.figure import Figure  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from subliminal_learning.analysis import plots as p  # noqa: E402

DENSE = p.KLadder(lo=0.0, hi=1.50, width=0.05, decimals=2, overflow_label=True)
SPARSE = p.KLadder(lo=0.2, hi=1.6, width=0.1, decimals=1)


# ------------------------------- KLadder -------------------------------

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


# --------------------------- facet_grid_heatmaps ---------------------------

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


# ------------------------------- house_grid -------------------------------

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


# --------------------------- bars / jitter / palettes ---------------------------

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
