"""Values, uncertainty marks and panel arrangement of the complete-grid figures."""

from pathlib import Path

import matplotlib.figure
import numpy as np
import pandas as pd
import pytest

from scripts import plot_transfer_family_multiway as plot
from subliminal_learning.analysis import transfer_family_pooled as tfp
from subliminal_learning.analysis import transfer_family_registry_io as registry


def fixture():
    ctx = tfp.FamilyContext(
        animals=[*tfp.DEFAULT_ANIMALS, "elephant"],
        rel_short={},
        animal_letter={},
        relations=(*tfp.RELATIONS, *registry.EXTRA_RELATIONS),
    )
    rows = []
    for i, relation in enumerate(ctx.relations[:-1]):
        n_base = len(tfp.RELATIONS)
        animals = (
            tfp.DEFAULT_ANIMALS
            if i < n_base
            else ["dog" if i < n_base + 6 else "elephant"]
        )
        for animal in animals:
            for j, value in enumerate([0.0, 0.4, 0.7, 0.8]):
                rows.append(
                    dict(
                        relation=relation,
                        animal=animal,
                        sender=f"s{i}",
                        receiver=f"r{i}",
                        teacher=f"t{j // 2}",
                        student_tag=f"c{j % 2}",
                        test=value,
                        ctrl=0.1,
                        ftseed=1,
                        tseed=0 if i >= 10 else 1,
                        core_grid="core30",
                        screened_eligible=False,
                    )
                )
    return pd.DataFrame(rows), ctx


def verticals(ax):
    return [
        s
        for c in ax.collections
        for s in c.get_segments()
        if len(s) == 2 and s[0, 0] == s[1, 0]
    ]


def test_generate_correct_values_intervals_layout_and_preservation(
    tmp_path, monkeypatch
):
    saved = {}
    monkeypatch.setattr(
        matplotlib.figure.Figure,
        "savefig",
        lambda f, p, **kw: saved.setdefault(Path(p).name, f),
    )
    previous = tmp_path / "existing.png"
    previous.write_bytes(b"old")
    frame, ctx = fixture()
    calls = []
    real = tfp.multiway_bootstrap

    def tracking(h, **kwargs):
        calls.append(len(h))
        return real(h, **kwargs)

    monkeypatch.setattr(tfp, "multiway_bootstrap", tracking)
    table = plot.generate(frame, ctx, tmp_path, n_bootstrap=25, seed=9)
    assert len(calls) == 34 and len(table) == 34
    assert (
        table.method.eq("multiway").all()
        and table.weighting.eq("equal matched runs").all()
    )
    assert not table.collapse_filter.any() and table.n_bootstrap.eq(25).all()
    assert table.max_lower.isna().all() and table.max_upper.isna().all()
    assert previous.read_bytes() == b"old" and len(saved) == 9
    valid = table[table.n_pairs.gt(0)]
    np.testing.assert_allclose(valid.win, 0.75)
    np.testing.assert_allclose(valid["median"], 0.45)
    missing = table[table.n_pairs.eq(0)]
    assert len(missing) == 1 and missing[plot.STATS + ["win"]].isna().all().all()
    fig = saved["transfer_family_complete_multiway_combined.png"]
    assert len(fig.axes) == 4
    win, quantile, size, olmo = fig.axes
    assert [len(a.patches) for a in fig.axes] == [25, 16, 6, 2]
    assert [len(verticals(a)) for a in fig.axes] == [25, 12, 6, 2]
    assert [t.get_text() for t in quantile.get_xticklabels()] == plot.STAT_LABELS
    assert [
        t.get_text() for t in quantile.get_legend().get_texts()
    ] == plot.FAMILY_LABELS
    assert [t.get_text() for t in win.get_legend().get_texts()] == [
        "pooled",
        *tfp.DEFAULT_ANIMALS,
    ]
    assert any("No complete" in t.get_text() for t in olmo.texts)
    assert all(s[0, 0] < 2.5 for s in verticals(quantile))
    np.testing.assert_allclose([p.get_height() for p in win.patches], 0.75)
    assert size.get_position().y0 == olmo.get_position().y0
    assert quantile.get_position().x0 > win.get_position().x0
    stand = saved["transfer_family_complete_multiway_win.png"]
    assert len(stand.axes) == 3
    assert stand.axes[0].get_position().width > stand.axes[1].get_position().width * 1.9
    assert (tmp_path / "complete_multiway_summary.csv").exists()


@pytest.mark.parametrize("all_arms", [False, True])
def test_cli_records_complete_cohort_and_uses_new_names(
    tmp_path, monkeypatch, all_arms
):
    from subliminal_learning.analysis import transfer_family_complete as complete

    frame, ctx = fixture()
    records = pd.DataFrame({"a": [1]})
    monkeypatch.setattr(plot, "bootstrap", lambda: (tmp_path, tmp_path / "outputs"))
    load_calls = []

    def load(*args, **kwargs):
        load_calls.append(kwargs)
        return frame, ctx, records, frame.iloc[:0].assign(exclusion_reason=""), records

    monkeypatch.setattr(complete, "load_complete_pairs", load)
    calls = []
    monkeypatch.setattr(
        plot, "generate", lambda *args, **kwargs: calls.append((args, kwargs))
    )
    other_calls = []
    monkeypatch.setattr(
        plot,
        "generate_all_arms",
        lambda *args, **kwargs: other_calls.append((args, kwargs)),
    )
    manifest_paths = [
        tmp_path / p for p in complete.GRID_PATHS.values()
    ] + registry._registry_paths(tmp_path)
    manifest_paths.append(
        tmp_path / "run_logs/repretrain/repretrain_significance_arms.json"
    )
    manifest_paths.extend(tmp_path / p for p in plot.OLMO_SAME_MANIFESTS)
    for path in manifest_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    out = plot.DEFAULT_OUT if all_arms else tmp_path / "multiway"
    plot.main(out_dir=out, n_bootstrap=7, seed=5, all_arms=all_arms)
    assert load_calls == (
        [dict(include_modern_reproduction=True, include_olmo_same_model=True)]
        if all_arms
        else [{}]
    )
    selected, unused = (other_calls, calls) if all_arms else (calls, other_calls)
    assert len(selected) == 1 and selected[0][1] == dict(n_bootstrap=7, seed=5)
    if all_arms:
        assert selected[0][0][2] == tmp_path / plot.ALL_ARMS_OUT
    assert not unused
    for name in [
        "complete_blocks.csv",
        "complete_cohort_exclusions.csv",
        "complete_registry_coverage.csv",
        "complete_matched_pairs.csv",
        "complete_cohort_counts.csv",
        "complete_snapshot.json",
    ]:
        assert (out / name).exists()


def test_all_arms_quantiles_cover_every_relation_and_missing_data(
    tmp_path, monkeypatch
):
    saved = {}
    monkeypatch.setattr(
        matplotlib.figure.Figure,
        "savefig",
        lambda f, p, **kw: saved.setdefault(Path(p).name, f),
    )
    frame, ctx = fixture()
    table = plot.generate_all_arms(frame, ctx, tmp_path, n_bootstrap=25, seed=3)
    overview = saved["transfer_family_multiway_raw_percentiles.png"]
    q = plot._all_arms_overview_rows(table, ctx)
    assert list(q.index.get_level_values("relation")) == list(
        plot._overview_context(ctx).relations
    )
    assert set(q.index.get_level_values("group")) == {"pooled", "dog", "elephant"}
    valid = q.n_pairs.gt(0)
    for ax, stat in zip(overview.axes, plot.STATS):
        assert len(ax.patches) == len(ctx.relations) - 1
        np.testing.assert_allclose(
            [p.get_width() for p in ax.patches], q.loc[valid, stat]
        )
        assert any("No complete block" in t.get_text() for t in ax.texts)
    # Horizontal interval endpoints must correspond to the plotted quantile, not wins.
    intervals = overview.axes[0].collections[0].get_segments()
    np.testing.assert_allclose(
        [seg[:, 0] for seg in intervals], q.loc[valid, ["median_lower", "median_upper"]]
    )
    assert not overview.axes[3].collections
    for i, counts in ((1, 25), (2, 6), (3, 2)):
        fig = saved[f"transfer_family_multiway_raw_percentiles_panel{i}.png"]
        for ax, stat in zip(fig.axes, plot.STATS):
            assert len(ax.patches) == counts
            expected = float(
                np.quantile(
                    np.array([0, 0.4, 0.7, 0.8]) - 0.1,
                    {"median": 0.5, "p75": 0.75, "p95": 0.95, "max": 1}[stat],
                )
            )
            expected_values = np.full(counts, expected)
            if i == 1:
                expected_values[: len(tfp.RELATIONS)] = np.quantile(
                    np.tile(np.array([0, 0.4, 0.7, 0.8]) - 0.1, 4),
                    {"median": 0.5, "p75": 0.75, "p95": 0.95, "max": 1}[stat],
                )
            np.testing.assert_allclose(
                [p.get_height() for p in ax.patches], expected_values
            )
            assert len(verticals(ax)) == (0 if stat == "max" else counts)
        assert len(fig.legends[0].get_texts()) == (5 if i == 1 else 1)
    assert len(saved) == 15
    assert (tmp_path / "transfer_family_multiway_raw_summary.csv").exists()
    assert all("win" not in p for p in saved)


def test_archive_exact_family_is_lossless_idempotent_and_preflights(tmp_path):
    source = tmp_path / plot.original.DEFAULT_OUT
    source.mkdir(parents=True)
    out = source / "all_arms"
    payloads = {name: name.encode() for name in plot.LEGACY_RAW_NAMES}
    for name, data in payloads.items():
        (source / name).write_bytes(data)
    untouched = source / "transfer_family_balanced_win.png"
    untouched.write_bytes(b"preserve")
    dest = out / "drafts" / "balanced_20260924"
    dest.mkdir(parents=True)
    conflict = dest / plot.LEGACY_RAW_NAMES[-1]
    conflict.write_bytes(b"different")
    with pytest.raises(FileExistsError):
        plot.archive_all_arms_originals(tmp_path, out)
    assert all((source / name).exists() for name in payloads)
    conflict.unlink()
    plot.archive_all_arms_originals(tmp_path, out)
    plot.archive_all_arms_originals(tmp_path, out)
    assert all((dest / name).read_bytes() == data for name, data in payloads.items())
    assert all(not (source / name).exists() for name in payloads)
    assert untouched.read_bytes() == b"preserve"


def test_all_arms_clean_and_metadata_overviews_match_values(tmp_path, monkeypatch):
    saved = {}
    monkeypatch.setattr(
        matplotlib.figure.Figure,
        "savefig",
        lambda f, p, **kw: saved.setdefault(Path(p).name, f),
    )
    frame, ctx = fixture()
    plot.generate_all_arms(frame, ctx, tmp_path, n_bootstrap=5, seed=1)
    clean = saved["transfer_family_multiway_raw_percentiles.png"]
    meta = saved["transfer_family_multiway_raw_percentiles_with_metadata.png"]
    for a, b in zip(clean.axes, meta.axes):
        np.testing.assert_allclose(
            [x.get_width() for x in a.patches], [x.get_width() for x in b.patches]
        )
        assert [x.get_text() for x in a.get_yticklabels()] == [
            x.get_text().split(";")[0] for x in b.get_yticklabels()
        ]


def test_modern_reproduction_is_separate_in_both_overviews(tmp_path, monkeypatch):
    from dataclasses import replace
    import re

    saved = {}
    monkeypatch.setattr(
        matplotlib.figure.Figure,
        "savefig",
        lambda f, p, **kw: saved.setdefault(Path(p).name, f),
    )
    frame, ctx = fixture()
    modern = frame[frame.relation.eq(tfp.RELATION_REPRO)].assign(
        relation=tfp.RELATION_MODERN_REPRO, test=0.25, ctrl=0.1
    )
    frame = pd.concat([frame, modern], ignore_index=True)
    relations = list(ctx.relations)
    old_index = relations.index(tfp.RELATION_REPRO)
    relations.insert(old_index + 1, tfp.RELATION_MODERN_REPRO)
    ctx = replace(ctx, relations=tuple(relations))
    table = plot.generate_all_arms(frame, ctx, tmp_path, n_bootstrap=25, seed=3)
    clean = saved["transfer_family_multiway_raw_percentiles.png"]
    detailed = saved["transfer_family_multiway_raw_percentiles_with_metadata.png"]
    ticks = [t.get_text() for t in clean.axes[0].get_yticklabels()]
    assert len(ticks) == 16
    assert "Rerun1234 ↔ original" in ticks[old_index]
    assert "Modern Reproduction" in ticks[old_index + 1]
    assert "Modern1234 ↔ original" in ticks[old_index + 1]
    assert all(
        not any(
            token in text
            for token in (
                "n=",
                "blocks",
                "core30",
                "core56",
                "core35",
                "lineage",
                "prv1",
            )
        )
        for text in ticks
    )
    assert not any(re.search(r"\borig\b", text) for text in ticks)
    assert any(
        "core30" in t.get_text() and "blocks" in t.get_text()
        for t in detailed.axes[0].get_yticklabels()
    )
    for a, b in zip(clean.axes, detailed.axes):
        np.testing.assert_allclose(
            [p.get_width() for p in a.patches], [p.get_width() for p in b.patches]
        )
        for ac, bc in zip(a.collections, b.collections):
            np.testing.assert_allclose(ac.get_segments(), bc.get_segments())
    panel = saved["transfer_family_multiway_raw_percentiles_panel1.png"]
    assert len(panel.axes[0].get_xticklabels()) == 6
    assert "Modern Tree Reproduction" in panel.axes[0].get_xticklabels()[
        old_index + 1
    ].get_text().replace("\n", " ")
    np.testing.assert_allclose(panel.axes[0].patches[old_index].get_height(), 0.45)
    np.testing.assert_allclose(panel.axes[0].patches[old_index + 1].get_height(), 0.15)
    assert (
        table.loc[table.relation.eq(tfp.RELATION_MODERN_REPRO), "n_pairs"].gt(0).all()
    )
    assert len(saved) == 15


def test_top_row_family_groups_preserve_values_colors_and_intervals(
    tmp_path, monkeypatch
):
    from matplotlib.colors import to_rgba

    frame, ctx = fixture()
    table = plot.summarize(frame, ctx, n_bootstrap=5, seed=0)
    relations = list(tfp.RELATIONS[:4])
    expected = np.array(
        [
            [0.01, 0.05, 0.11, 0.2],
            [0.02, 0.07, 0.13, 0.3],
            [0.03, 0.09, 0.17, 0.4],
            [0.04, 0.12, 0.19, 0.5],
        ]
    )
    for j, relation in enumerate(relations):
        mask = table.relation.eq(relation) & table.group.eq("pooled")
        for k, stat in enumerate(plot.STATS):
            table.loc[mask, stat] = expected[j, k]
            if k < 3:
                table.loc[mask, f"{stat}_lower"] = expected[j, k] - 0.003
                table.loc[mask, f"{stat}_upper"] = expected[j, k] + 0.007
    regular = plot.draw_top_row(table, ctx)
    grouped = plot.draw_top_row(table, ctx, quantile_layout="families")
    left, right = grouped.axes
    assert [t.get_text() for t in left.get_xticklabels()] == [
        t.get_text() for t in right.get_xticklabels()
    ]
    assert [t.get_text() for t in right.get_legend().get_texts()] == plot.STAT_LABELS
    palette = [to_rgba(c) for c in ["#E9B000", "#D7263D", "#7B2CBF", "#00B8D9"]]
    np.testing.assert_allclose(
        [h.get_facecolor() for h in right.get_legend().legend_handles], palette
    )
    for j in range(4):
        bars = right.patches[j * 4 : (j + 1) * 4]
        np.testing.assert_allclose([b.get_height() for b in bars], expected[j])
        centers = np.array([b.get_x() + b.get_width() / 2 for b in bars])
        assert all(abs(centers - j) < 0.5) and all(np.diff(centers) > 0)
        np.testing.assert_allclose([b.get_facecolor() for b in bars], palette)
        intervals = verticals(right)[j * 3 : (j + 1) * 3]
        np.testing.assert_allclose([v[0, 0] for v in intervals], centers[:3])
        np.testing.assert_allclose(
            [v[:, 1] for v in intervals],
            np.column_stack([expected[j, :3] - 0.003, expected[j, :3] + 0.007]),
        )
        old_bars = regular.axes[1].patches[j * 4 : (j + 1) * 4]
        np.testing.assert_allclose([b.get_height() for b in old_bars], expected[j])
        np.testing.assert_array_equal(
            np.rint([b.get_x() + b.get_width() / 2 for b in old_bars]), np.arange(4)
        )
        np.testing.assert_allclose(
            [b.get_facecolor() for b in old_bars], np.tile(palette[j], (4, 1))
        )
    assert len(verticals(right)) == 12 and right.get_ylim()[0] == 0
    for old, new in zip(regular.axes[0].patches, left.patches):
        assert old.get_height() == new.get_height()
        assert old.get_facecolor() == new.get_facecolor()
        assert old.get_hatch() == new.get_hatch()
    np.testing.assert_allclose(verticals(regular.axes[0]), verticals(left))
    # Missing summary values cannot silently become zero-height observations.
    missing = table.copy()
    mask = missing.relation.eq(relations[1]) & missing.group.eq("pooled")
    missing.loc[
        mask,
        plot.STATS
        + [f"{s}_{bound}" for s in plot.STATS[:3] for bound in ["lower", "upper"]],
    ] = np.nan
    absent = plot.draw_top_row(missing, ctx, quantile_layout="families")
    assert all(np.isnan(b.get_height()) for b in absent.axes[1].patches[4:8])
    assert len(verticals(absent.axes[1])) == 9
    saved = []
    monkeypatch.setattr(
        matplotlib.figure.Figure,
        "savefig",
        lambda self, path, **kwargs: saved.append(Path(path).name),
    )
    summary = tmp_path / "summary.csv"
    table.to_csv(summary, index=False)
    for group, layout, stem in [
        ("pooled", "statistics", "transfer_family_complete_multiway_top_row"),
        ("pooled", "families", "transfer_family_complete_multiway_top_row_families"),
        ("dog", "families", "transfer_family_complete_multiway_top_row_dog_families"),
    ]:
        plot.export_top_row(
            summary, tmp_path, quantile_group=group, quantile_layout=layout
        )
        assert saved[-3:] == [f"{stem}.{ext}" for ext in ["png", "pdf", "svg"]]
    plot.plt.close("all")


def test_overview_labels_title_and_olmo_slot_are_explicit():
    import re

    frame, ctx = fixture()
    table = plot.summarize(frame, ctx, n_bootstrap=5, seed=0)
    clean = plot._percentile_overview(table, ctx)
    detailed = plot._percentile_overview(table, ctx, with_metadata=True)
    labels = [t.get_text() for t in clean.axes[0].get_yticklabels()]
    index = labels.index("Same init,\nsame corpus\nelephant")
    assert labels[index : index + 4] == [
        "Same init,\nsame corpus\nelephant",
        "Same init,\ndiff corpus\nelephant",
        "Diff init,\ndiff corpus\nelephant",
        "Diff init,\nsame corpus\nelephant",
    ]
    assert "160M → 410M\nsame data order\ndog" in labels
    assert "410M → 160M\ndiff data order\ndog" in labels
    assert all(
        "different" not in label.lower() and not re.search(r",[^\n]", label)
        for label in labels
    )
    assert all(
        "cross-seed" not in label.lower() and "original-seed" not in label
        for label in labels
    )
    for fig in [clean, detailed]:
        assert fig._suptitle is None
        for ax in fig.axes:
            assert not any(
                np.isclose(p.get_y() + p.get_height() / 2, index) for p in ax.patches
            )
        fig.set_dpi(180)
        fig.canvas.draw()
        bounds = [t.get_window_extent() for t in fig.axes[0].get_yticklabels()]
        assert min(b.x0 for b in bounds) >= 0 and min(b.y0 for b in bounds) >= 0
        assert all(a.y0 > b.y1 for a, b in zip(bounds, bounds[1:]))
    for a, b in zip(clean.axes, detailed.axes):
        np.testing.assert_allclose(
            [p.get_width() for p in a.patches], [p.get_width() for p in b.patches]
        )
        for ac, bc in zip(a.collections, b.collections):
            np.testing.assert_allclose(ac.get_segments(), bc.get_segments())
    # New complete diagonal data must fill that exact reserved slot.
    source = table[
        (table.relation == registry.EXTRA_RELATIONS[6]) & (table.group == "elephant")
    ].copy()
    source["relation"] = registry.RELATION_OLMO_SAME
    source["median"] = 0.123
    full = plot._percentile_overview(pd.concat([table, source]), ctx)
    bars = [
        p
        for p in full.axes[0].patches
        if np.isclose(p.get_y() + p.get_height() / 2, index)
    ]
    assert len(bars) == 1 and bars[0].get_width() == 0.123
    assert plot.ALL_ARMS_LABELS[registry.EXTRA_RELATIONS[0]].endswith("original-seed")
    plot.plt.close("all")
