import matplotlib.figure
from matplotlib.colors import to_rgba
import numpy as np
import pandas as pd
import pytest

from notebooks.figures.figure2_family import plot_transfer_family_multiway as plot
from subliminal_learning.analysis import transfer_family_pooled as tfp
from subliminal_learning.analysis import transfer_family_registry_io as registry


def _table():
    rows = []
    for i, relation in enumerate(
        [
            *registry.EXTRA_RELATIONS[:6],
            registry.RELATION_OLMO_SAME,
            *registry.EXTRA_RELATIONS[6:],
        ]
    ):
        row = dict(relation=relation, group="dog" if i < 6 else "elephant")
        for j, stat in enumerate(plot.STATS):
            row[stat] = 0.01 * (i + 1) + 0.1 * j
            row[f"{stat}_lower"] = row[stat] - 0.003 if j < 3 else np.nan
            row[f"{stat}_upper"] = row[stat] + 0.005 if j < 3 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _verticals(ax):
    return [
        s
        for c in ax.collections
        for s in c.get_segments()
        if len(s) == 2 and s[0, 0] == s[1, 0]
    ]


@pytest.mark.parametrize("same_model_present", [True, False])
def test_size_olmo_family_statistics_and_pending(same_model_present):
    table = _table()
    if not same_model_present:
        table = table[table.relation.ne(registry.RELATION_OLMO_SAME)]
    fig = plot.draw_size_olmo(table, tfp.default_context())
    palette = [to_rgba(c) for c in ["#E9B000", "#D7263D", "#7B2CBF", "#00B8D9"]]
    for panel, ax in enumerate(fig.axes):
        expected = table[table.group.eq("dog" if panel == 0 else "elephant")]
        assert len(ax.patches) == len(expected) * 4
        assert len(_verticals(ax)) == len(expected) * 3
        assert ax.get_ylim()[0] == 0
        assert [t.get_text() for t in ax.get_legend().get_texts()] == plot.STAT_LABELS
        np.testing.assert_allclose(
            [h.get_facecolor() for h in ax.get_legend().legend_handles], palette
        )
        for bar in ax.patches:
            center = bar.get_x() + bar.get_width() / 2
            family = int(round(center))
            stat = int(
                np.argmin([abs(center - family - (j - 1.5) * 0.185) for j in range(4)])
            )
            relation = (
                [*registry.EXTRA_RELATIONS[:6]]
                if panel == 0
                else [registry.RELATION_OLMO_SAME, *registry.EXTRA_RELATIONS[6:]]
            )[family]
            row = expected[expected.relation.eq(relation)].iloc[0]
            assert bar.get_height() == row[plot.STATS[stat]]
            np.testing.assert_allclose(bar.get_facecolor(), palette[stat])
            intervals = [s for s in _verticals(ax) if np.isclose(s[0, 0], center)]
            assert len(intervals) == (1 if stat < 3 else 0)
            if intervals:
                np.testing.assert_allclose(
                    intervals[0][:, 1],
                    [
                        row[f"{plot.STATS[stat]}_lower"],
                        row[f"{plot.STATS[stat]}_upper"],
                    ],
                )
        assert all("lineage" not in t.get_text() for t in ax.get_xticklabels())
    right = fig.axes[1]
    assert right.get_xticklabels()[0].get_text() == "Same-model"
    assert any("Pending" in t.get_text() for t in right.texts) != same_model_present
    assert (
        any(abs(b.get_x() + b.get_width() / 2) < 0.5 for b in right.patches)
        == same_model_present
    )
    assert fig.axes[0].get_ylim() == fig.axes[1].get_ylim()
    fig.set_dpi(200)
    fig.canvas.draw()
    for ax in fig.axes:
        bounds = [t.get_window_extent() for t in ax.get_xticklabels()]
        assert min(b.y0 for b in bounds) > 0
        assert all(a.x1 < b.x0 for a, b in zip(bounds, bounds[1:]))
    plot.plt.close(fig)


def test_size_olmo_exports_do_not_overwrite_top_row(tmp_path, monkeypatch):
    existing = tmp_path / "figure2.png"
    existing.write_bytes(b"previous")
    saved = []
    monkeypatch.setattr(
        matplotlib.figure.Figure,
        "savefig",
        lambda self, path, **kw: saved.append(path.name),
    )
    plot.export_size_olmo(_table(), tfp.default_context(), tmp_path)
    assert saved == [
        f"figure2_olmo.{ext}" for ext in ["png", "pdf", "svg"]
    ]
    assert existing.read_bytes() == b"previous"


def test_size_olmo_axis_holds_every_drawn_bar_and_interval():
    """A tall bar in either panel must widen the shared axis, not be clipped."""
    table = _table()
    tall = table.relation.eq(registry.EXTRA_RELATIONS[4])
    table.loc[tall, "max"] = 0.9
    table.loc[tall, "p95_upper"] = 0.8
    fig = plot.draw_size_olmo(table, tfp.default_context())
    top = fig.axes[0].get_ylim()[1]
    assert fig.axes[0].get_ylim() == fig.axes[1].get_ylim()
    # The shared axis is annotated once: B keeps its ticks but no duplicate text.
    assert fig.axes[0].get_ylabel() and not fig.axes[1].get_ylabel()
    assert not any(t.get_text() for t in fig.axes[1].get_yticklabels())
    assert len(fig.axes[1].get_yticks()) == len(fig.axes[0].get_yticks())
    assert top == pytest.approx(0.9 * 1.05)
    for ax in fig.axes:
        assert all(b.get_y() + b.get_height() <= top for b in ax.patches)
        assert all(max(s[:, 1]) <= top for s in _verticals(ax))
    plot.plt.close(fig)
