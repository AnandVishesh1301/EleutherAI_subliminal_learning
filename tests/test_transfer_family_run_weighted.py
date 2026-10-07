"""Ordinary-bootstrap comparison preserves the existing figure family."""

from pathlib import Path

import matplotlib.figure
import numpy as np
import pandas as pd

from scripts import plot_transfer_family_run_weighted as cli
from scripts import plot_transfer_family_pooled_minus_control as original
from subliminal_learning.analysis import transfer_family_pooled as tfp


def _frame():
    rows = []
    for relation in tfp.RELATIONS:
        for i, (animal, value) in enumerate(
            zip(["dog", "dog", "dog", "cat"], [0.4, 0.6, 0.8, 0.0])
        ):
            rows.append(
                dict(
                    relation=relation,
                    animal=animal,
                    sender="s",
                    receiver="r",
                    teacher=f"t{i}",
                    student_tag=f"c{i}",
                    test=value,
                    ctrl=0.2,
                    screened_eligible=False,
                )
            )
    return pd.DataFrame(rows)


def test_generate_equal_runs_displays_intervals_and_keeps_existing_files(
    tmp_path, monkeypatch
):
    preserved = tmp_path / "existing.png"
    preserved.write_bytes(b"original figure")
    out = tmp_path / "run_weighted"
    saved = {}
    monkeypatch.setattr(
        matplotlib.figure.Figure,
        "savefig",
        lambda fig, path, **kw: saved.setdefault(Path(path), fig),
    )
    table = cli.generate(_frame(), tfp.default_context(), out, n_bootstrap=20, seed=3)
    assert len(table) == 8
    assert table.loc[table.group.eq("pooled"), "n_pairs"].eq(4).all()
    assert table.loc[table.group.eq("dog"), "n_pairs"].eq(3).all()
    assert table.loc[table.group.eq("pooled"), "win"].eq(0.75).all()
    np.testing.assert_allclose(table.loc[table.group.eq("pooled"), "median"], 0.3)
    assert table.n_bootstrap.eq(20).all() and table.seed.eq(3).all()
    assert table.quantile_method.eq("linear").all() and not table.collapse_filter.any()
    assert table.max_lower.isna().all() and table.max_upper.isna().all()
    assert len(saved) == 6 and all(p.parent == out for p in saved)
    for group in ("pooled", "dog"):
        fig = saved[out / f"transfer_family_compact_{group}_statistics_raw.png"]
        ax = fig.axes[0]
        assert len(ax.patches) == 16
        assert [t.get_text() for t in ax.get_xticklabels()] == [
            "Median",
            "75th percentile",
            "95th percentile",
            "Maximum",
        ]
        vertical = [
            s for c in ax.collections for s in c.get_segments() if s[0, 0] == s[1, 0]
        ]
        assert len(vertical) == 12 and all(s[0, 0] < 2.5 for s in vertical)
        text = " ".join(t.get_text() for t in fig.texts)
        assert "Equal weight per run" in text and "treated as independent" in text
        assert "configuration-bootstrap" not in text and "Equal animals" not in text
        assert len(fig.legends[0].get_texts()) == 4
    assert preserved.read_bytes() == b"original figure"
    assert (out / "transfer_family_compact_run_weighted_summary.csv").exists()


def test_missing_family_or_dog_is_missing_not_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(original, "_draw_compact", lambda *a, **kw: None)
    frame = _frame()
    frame = frame[frame.animal.eq("cat") & frame.relation.eq(tfp.RELATIONS[0])]
    table = cli.generate(frame, tfp.default_context(), tmp_path / "new", n_bootstrap=3)
    missing = table[table.n_pairs.eq(0)]
    assert len(missing) == 7
    assert missing[["win", "median", "p75", "p95", "max"]].isna().all().all()


def test_cli_loads_existing_grids_without_collapse_filter(tmp_path, monkeypatch):
    (tmp_path / "run_logs/registry").mkdir(parents=True)
    (tmp_path / "run_logs/registry/core30_cells.json").write_text(
        '{"cells":[{"student_tag":"selected"}]}'
    )
    monkeypatch.setattr(cli, "bootstrap", lambda: (tmp_path, tmp_path / "outputs"))
    monkeypatch.setattr(original, "_teachers", lambda *a: {})

    def pairs(*a, **kw):
        assert kw["collapse_filter"] is False
        assert kw["cells"] == ["selected"]
        return _frame(), tfp.default_context()

    monkeypatch.setattr(tfp, "all_pairs", pairs)
    calls = []
    monkeypatch.setattr(
        cli, "generate", lambda f, c, out, **kw: calls.append((out, kw))
    )
    cli.main(out_dir=cli.DEFAULT_OUT, n_bootstrap=12, seed=8)
    assert calls == [(tmp_path / cli.DEFAULT_OUT, {"n_bootstrap": 12, "seed": 8})]
    assert cli.DEFAULT_OUT == original.DEFAULT_OUT / "run_weighted"


def test_new_methods_dispatch_independently_and_keep_equal_point_estimates(
    tmp_path, monkeypatch
):
    saved = {}
    monkeypatch.setattr(
        matplotlib.figure.Figure,
        "savefig",
        lambda fig, path, **kw: saved.setdefault(Path(path), fig),
    )
    ordinary = cli.generate(
        _frame(), tfp.default_context(), tmp_path / "rows", n_bootstrap=15, seed=4
    )

    def forbidden(*a, **kw):
        raise AssertionError("new methods must not use the IID row bootstrap")

    monkeypatch.setattr(tfp, "row_bootstrap", forbidden)
    for method in ("stratified", "multiway"):
        out = tmp_path / method
        table = cli.generate(
            _frame(), tfp.default_context(), out, method=method, n_bootstrap=15, seed=4
        )
        pd.testing.assert_frame_equal(
            table[["median", "p75", "p95", "max", "win", "n_pairs"]],
            ordinary[["median", "p75", "p95", "max", "win", "n_pairs"]],
        )
        assert table.method.eq(method).all() and table.n_empty_draws.ge(0).all()
        assert table.max_lower.isna().all() and table.max_upper.isna().all()
        for group in ("pooled", "dog"):
            fig = saved[out / f"transfer_family_compact_{group}_statistics_raw.png"]
            text = " ".join(t.get_text() for t in fig.texts)
            assert "95%" in text and method in text and "Equal weight per run" in text
            assert "configuration-bootstrap" not in text
            if method == "multiway":
                assert "treated as independent" not in text
            vertical = [
                s
                for c in fig.axes[0].collections
                for s in c.get_segments()
                if s[0, 0] == s[1, 0]
            ]
            assert len(vertical) == 12 and all(s[0, 0] < 2.5 for s in vertical)


def test_cli_both_writes_only_method_subdirectories(tmp_path, monkeypatch):
    (tmp_path / "run_logs/registry").mkdir(parents=True)
    (tmp_path / "run_logs/registry/core30_cells.json").write_text('{"cells":[]}')
    monkeypatch.setattr(cli, "bootstrap", lambda: (tmp_path, tmp_path / "outputs"))
    monkeypatch.setattr(original, "_teachers", lambda *a: {})

    def load(*a, **kw):
        assert kw["collapse_filter"] is False
        return _frame(), tfp.default_context()

    monkeypatch.setattr(tfp, "all_pairs", load)
    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", lambda *a, **kw: None)
    out = tmp_path / "comparison"
    out.mkdir()
    existing = out / "transfer_family_compact_run_weighted_summary.csv"
    existing.write_bytes(b"existing row results")
    cli.main(out_dir=out, n_bootstrap=3, seed=2, method="both")
    assert existing.read_bytes() == b"existing row results"
    for method in ("stratified", "multiway"):
        table = pd.read_csv(out / method / existing.name)
        assert len(table) == 8 and table.method.eq(method).all()


def test_cli_rejects_unknown_method_before_loading(monkeypatch):
    import pytest

    def forbidden():
        raise AssertionError("invalid method should not load data")

    monkeypatch.setattr(cli, "bootstrap", forbidden)
    with pytest.raises(cli.typer.BadParameter, match="method"):
        cli.main(method="unknown")
