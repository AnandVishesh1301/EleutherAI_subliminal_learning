import matplotlib.figure
import numpy as np
import pandas as pd
import pytest

from notebooks.figures.figure2_family import plot_transfer_family_multiway as plot
from subliminal_learning.analysis import transfer_family_pooled as tfp
from subliminal_learning.analysis import transfer_family_registry_io as registry

SIZE = list(registry.EXTRA_RELATIONS[:4])


def _pairs(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for k, relation in enumerate(SIZE):
        for animal in ("dog", "cat"):
            for s in ("a", "b"):
                for t in range(3):
                    for c in range(4):
                        rows.append(
                            dict(
                                relation=relation,
                                animal=animal,
                                sender=f"{s}{k}",
                                receiver=f"r{k}",
                                teacher=f"t{t}",
                                student_tag=f"s{c}",
                                tseed=1,
                                ftseed=1,
                                core_grid="core30",
                                test=rng.random() * 0.3 + 0.01 * c,
                                ctrl=rng.random() * 0.05,
                            )
                        )
    frame = pd.DataFrame(rows)
    frame.loc[0, "test"] = np.nan
    return frame


def _summary():
    base = registry.relation_groups(tfp.default_context().relations)[0][:4]
    keys = [(r, "dog") for r in base] + [
        (r, "elephant")
        for r in [registry.RELATION_OLMO_SAME, *registry.EXTRA_RELATIONS[6:]]
    ] + [(r, "dog") for r in registry.EXTRA_RELATIONS[4:6]]
    rows = []
    for i, (relation, group) in enumerate(keys):
        row = dict(relation=relation, group=group, n_bootstrap=20, seed=3)
        row.update(n_pairs=10, n_blocks=2, core_grid="core30", win=0.5)
        row.update(win_lower=0.4, win_upper=0.6)
        for j, stat in enumerate(plot.STATS):
            row[stat] = 0.01 * (i + 1) + 0.1 * j
            row[f"{stat}_lower"] = row[stat] - 0.003 if j < 3 else np.nan
            row[f"{stat}_upper"] = row[stat] + 0.005 if j < 3 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def test_lumped_rows_match_bootstrap_on_union():
    frame = _pairs()
    lumped = plot.summarize_lumped(frame, n_bootstrap=20, seed=3)
    assert list(lumped.relation) == list(plot.LUMPED_SIZE)
    assert (lumped.group == "dog").all()
    for (name, sources), (_, row) in zip(plot.LUMPED_SIZE.items(), lumped.iterrows()):
        h = frame[frame.relation.isin(sources) & frame.animal.eq("dog")]
        h = h[np.isfinite(h.test) & np.isfinite(h.ctrl)].copy()
        h["delta"] = h.test - h.ctrl
        expected = tfp.multiway_bootstrap(h, n_bootstrap=20, seed=3)
        assert row.n_pairs == len(h) == sum(
            (frame.relation.eq(s) & frame.animal.eq("dog") & frame.test.notna()).sum()
            for s in sources
        )
        for stat in plot.STATS:
            assert row[stat] == expected["percentiles"][stat]
        assert row.lumped_relations == " | ".join(sources)


def _verticals(ax):
    return [
        s
        for c in ax.collections
        for s in c.get_segments()
        if len(s) == 2 and s[0, 0] == s[1, 0]
    ]


def test_combined_draw_panels_and_shared_axis():
    lumped = plot.summarize_lumped(_pairs(), n_bootstrap=20, seed=3)
    lumped.loc[1, "max"] = 2.0
    fig = plot.draw_combined(_summary(), lumped, tfp.default_context())
    assert len(fig.axes) == 3
    assert [ax.get_title(loc="left")[0] for ax in fig.axes] == ["A", "B", "C"]
    assert len(fig.axes[2].get_xticklabels()) == 4
    # The Pythia<->OLMo bars are the main-summary values, unchanged.
    summary = _summary().set_index(["relation", "group"])
    heights = {round(b.get_height(), 9) for b in fig.axes[2].patches}
    for r in registry.EXTRA_RELATIONS[4:6]:
        for stat in plot.STATS:
            assert round(summary.loc[(r, "dog"), stat], 9) in heights
    fig.set_dpi(200)
    fig.canvas.draw()
    for ax in fig.axes:
        bounds = [t.get_window_extent() for t in ax.get_xticklabels()]
        assert all(a.x1 < b.x0 for a, b in zip(bounds, bounds[1:]))
    top = fig.axes[0].get_ylim()[1]
    assert top == pytest.approx(2.0 * 1.05)
    for ax in fig.axes:
        assert ax.get_ylim()[1] == top
        assert all(b.get_y() + b.get_height() <= top for b in ax.patches)
        assert all(max(s[:, 1]) <= top for s in _verticals(ax))
    plot.plt.close(fig)


def test_combined_empty_lumped_category_is_pending():
    frame = _pairs()
    frame = frame[~frame.relation.isin(plot.LUMPED_SIZE["large→small, all lineages"])]
    lumped = plot.summarize_lumped(frame, n_bootstrap=20, seed=3)
    assert lumped.n_pairs.iloc[1] == 0
    fig = plot.draw_combined(_summary(), lumped, tfp.default_context())
    assert any("Pending" in t.get_text() for t in fig.axes[2].texts)
    plot.plt.close(fig)


def test_render_top_row_writes_combined(tmp_path, monkeypatch):
    _summary().to_csv(tmp_path / "complete_multiway_summary.csv", index=False)
    saved = []
    monkeypatch.setattr(
        matplotlib.figure.Figure,
        "savefig",
        lambda self, path, **kw: saved.append(path.name),
    )
    monkeypatch.setattr(plot, "export_top_row", lambda *a, **k: None)
    with pytest.raises(FileNotFoundError, match="complete_matched_pairs.csv"):
        plot.render_top_row(tmp_path)
    _pairs().to_csv(tmp_path / "complete_matched_pairs.csv", index=False)
    saved.clear()
    plot.render_top_row(tmp_path)
    assert saved == [
        f"{stem}.{e}"
        for stem in (
            "figure2_olmo",
            "figure2_win",
            "figure2_combined",
            "figure2_combined_win",
        )
        for e in ("png", "pdf", "svg")
    ]
    lumped = pd.read_csv(tmp_path / "complete_multiway_summary_lumped.csv")
    assert len(lumped) == 2
    expected = plot.summarize_lumped(_pairs(), n_bootstrap=20, seed=3)
    np.testing.assert_allclose(lumped["median"], expected["median"])


def _win_summary():
    ctx = tfp.default_context()
    base = registry.relation_groups(ctx.relations)[0][:4]
    keys = [
        (r, g) for r in base for g in ("pooled", "cat", "dog", "lion", "pig")
    ] + [
        (r, "elephant")
        for r in [registry.RELATION_OLMO_SAME, *registry.EXTRA_RELATIONS[6:]]
    ] + [(r, "dog") for r in registry.EXTRA_RELATIONS[4:6]]
    rows = []
    for i, (relation, group) in enumerate(keys):
        win = 0.3 + 0.01 * i
        row = dict(
            relation=relation,
            group=group,
            n_pairs=10,
            n_blocks=2,
            core_grid="core30",
            win=win,
            win_lower=win - 0.02,
            win_upper=win + 0.03,
        )
        for stat in plot.STATS:
            row[stat] = 0.1
        rows.append(row)
    return pd.DataFrame(rows)


def test_combined_win_panels_bars_and_ticks():
    table = _win_summary()
    lumped = plot.summarize_lumped(_pairs(), n_bootstrap=20, seed=3)
    fig = plot.draw_combined_win(table, lumped, tfp.default_context())
    assert len(fig.axes) == 3
    assert [ax.get_title(loc="left")[0] for ax in fig.axes] == ["A", "B", "C"]
    a, b, c = fig.axes
    assert len(a.patches) == 4 * 5 and len(b.patches) == 4 and len(c.patches) == 4
    assert a.get_legend() is not None
    assert b.get_legend() is None and c.get_legend() is None
    assert a.get_ylabel() and not b.get_ylabel() and not c.get_ylabel()
    assert len(c.get_xticklabels()) == 4
    expected_b = table[table.group.eq("elephant")].win
    np.testing.assert_allclose(
        sorted(p.get_height() for p in b.patches), sorted(expected_b)
    )
    expected_c = list(lumped.win) + list(
        table.set_index(["relation", "group"])
        .loc[[(r, "dog") for r in registry.EXTRA_RELATIONS[4:6]]]
        .win
    )
    np.testing.assert_allclose(
        [p.get_height() for p in sorted(c.patches, key=lambda p: p.get_x())],
        expected_c,
    )
    pooled = table[table.group.eq("pooled")].win
    assert set(np.round(pooled, 9)) <= {round(p.get_height(), 9) for p in a.patches}
    fig.set_dpi(200)
    fig.canvas.draw()
    for ax in fig.axes:
        bounds = [t.get_window_extent() for t in ax.get_xticklabels()]
        assert all(x.x1 < y.x0 for x, y in zip(bounds, bounds[1:]))
    plot.plt.close(fig)


def test_figure2_win_single_panel():
    table = _win_summary()
    fig = plot.draw_figure2_win(table, tfp.default_context())
    assert len(fig.axes) == 1
    (ax,) = fig.axes
    assert ax.get_title(loc="left") == "Pythia 160M relationships"
    assert len(ax.patches) == 20
    base = registry.relation_groups(tfp.default_context().relations)[0][:4]
    expected = table[table.relation.isin(base)].win
    np.testing.assert_allclose(
        sorted(p.get_height() for p in ax.patches), sorted(expected)
    )
    plot.plt.close(fig)
