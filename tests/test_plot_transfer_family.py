"""Regression checks for Figure 2 support, priors, plotting, and safe archival."""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
from scripts import plot_transfer_family_pooled_minus_control as cli  # noqa: E402
from subliminal_learning.analysis import transfer_family_pooled as tfp  # noqa: E402


def _pairs():
    return pd.DataFrame(
        [
            {
                "relation": tfp.RELATIONS[0],
                "animal": "dog",
                "sender": "wc1",
                "receiver": "wc2",
                "teacher": f"t{i}",
                "student_tag": f"c{i}",
                "test": 0.3 + i * 0.1,
                "ctrl": 0.1,
                "ctrl_probed": False,
                "receiver_filtered": False,
                "alive": np.nan,
            }
            for i in range(2)
        ]
    )


def test_prior_resolution_is_cached_by_receiver_animal_and_uses_full_base(
    tmp_path, monkeypatch
):
    calls = []

    def prior(animal, receiver, **kwargs):
        calls.append((animal, receiver, kwargs["prompt_set"]))
        return 0.05

    monkeypatch.setattr(cli.tio, "student_tree_prior", prior)
    frame, priors = cli._prior_frame(_pairs(), tmp_path)
    assert calls == [("dog", tfp.REPRETRAIN_BASE["wc2"], "test")]
    assert len(frame) == 2 and len(priors) == 1
    assert frame.prior_available.all()
    assert frame.receiver_prior.eq(0.05).all()
    assert "base_eval_wc2" in priors.iloc[0].prior_source


@pytest.mark.parametrize("value", [0, np.nan])
def test_unusable_prior_is_reported_without_changing_raw_population(
    tmp_path, monkeypatch, value
):
    monkeypatch.setattr(cli.tio, "student_tree_prior", lambda *a, **kw: value)
    frame, priors = cli._prior_frame(_pairs(), tmp_path)
    assert len(frame) == 2 and not frame.prior_available.any()
    assert priors.iloc[0].prior_error
    ctx = tfp.default_context(["dog"])
    raw = cli._summary_table(frame, ctx, n_bootstrap=10, seed=0, scale="raw")
    norm = cli._summary_table(frame, ctx, n_bootstrap=10, seed=0, scale="prior")
    assert raw.n_pairs.sum() == 4
    assert norm.n_pairs.sum() == 0
    assert norm.n_prior_excluded_pairs.sum() == 4


def test_missing_prior_is_reported_but_inconsistent_prior_raises(tmp_path, monkeypatch):
    def missing(*a, **kw):
        raise FileNotFoundError("test prior missing")

    monkeypatch.setattr(cli.tio, "student_tree_prior", missing)
    frame, _ = cli._prior_frame(_pairs(), tmp_path)
    assert not frame.prior_available.any()

    def inconsistent(*a, **kw):
        raise ValueError("student trees disagree")

    monkeypatch.setattr(cli.tio, "student_tree_prior", inconsistent)
    with pytest.raises(ValueError, match="disagree"):
        cli._prior_frame(_pairs(), tmp_path)


def test_coverage_counts_unique_controls_separately_from_teacher_rows():
    frame = _pairs().assign(prior_available=True)
    extra = frame.iloc[[0]].assign(teacher="another")
    coverage = cli._coverage_table(pd.concat([frame, extra], ignore_index=True))
    row = coverage.iloc[0]
    assert row.n_pairs == 3
    assert row.n_matched_controls == 2
    assert row.n_unlabelled_controls == 2
    assert row.n_unlabelled_control_pairs == 3
    assert row.n_unfiltered_pairs == 3


def test_archive_collision_is_preflighted_and_originals_preserved(tmp_path):
    source = tmp_path / "notebooks/figures"
    archive = tmp_path / cli.DEFAULT_OUT / "drafts"
    archive.mkdir(parents=True)
    (source / "transfer_family_a.png").write_bytes(b"first")
    (source / "transfer_family_z.png").write_bytes(b"second")
    (archive / "transfer_family_z.png").write_bytes(b"different")
    with pytest.raises(FileExistsError):
        cli._archive_existing(tmp_path)
    assert (source / "transfer_family_a.png").read_bytes() == b"first"
    (archive / "transfer_family_z.png").unlink()
    cli._archive_existing(tmp_path)
    assert not list(source.glob("transfer_family*.png"))
    assert (archive / "transfer_family_a.png").read_bytes() == b"first"
    cli._archive_existing(tmp_path)  # Re-running does not archive the new generation.
    assert len(list(archive.glob("*.png"))) == 2


def test_all_groups_render_even_when_interval_does_not_bracket_estimate(tmp_path):
    ctx = tfp.default_context(["dog"])
    frame = _pairs().assign(prior_available=True, receiver_prior=0.05)
    summary = cli._summary_table(frame, ctx, n_bootstrap=10, seed=0, scale="raw")
    summary["median_lower"] = 0.7
    summary["median_upper"] = 0.8
    fig, ax = cli.plt.subplots()
    cli._bars(ax, summary, ctx, "median")
    assert len(ax.patches) == 8  # Four relations x pooled/dog, including absent slots.
    assert [t.get_text() for t in ax.get_xticklabels()][2].startswith("diff. init")
    cli.plt.close(fig)
    cli._draw_summary(summary, ctx, tmp_path, scale="raw")
    assert (tmp_path / "transfer_family_balanced_win.png").exists()
    assert (tmp_path / "transfer_family_balanced_raw_percentiles.png").exists()
    assert summary.max_lower.isna().all() and summary.max_upper.isna().all()
    assert cli.DEFAULT_OUT == Path("notebooks/figures/figure2_family")


def _extended_frame():
    ctx = cli._expanded_context(tfp.default_context())
    frame = _pairs().assign(
        relation=tfp.RELATIONS[0],
        prior_available=True,
        receiver_prior=0.05,
        arm_incomplete=False,
    )
    extra = frame.assign(relation=ctx.relations[8], animal="dog")
    elephant = frame.iloc[[0]].assign(
        relation=ctx.relations[11],
        animal="elephant",
        prior_available=False,
        receiver_prior=np.nan,
        arm_incomplete=True,
    )
    return pd.concat([frame, extra, elephant], ignore_index=True), ctx


def test_expanded_context_and_support_do_not_double_count_pooled_rows():
    frame, ctx = _extended_frame()
    assert len(ctx.relations) == 13 and ctx.animals[-1] == "elephant"
    raw = cli._summary_table(frame, ctx, n_bootstrap=10, seed=0, scale="raw")
    norm = cli._summary_table(frame, ctx, n_bootstrap=10, seed=0, scale="prior")
    assert len(raw) == 13 * 6
    pooled = raw[(raw.relation == ctx.relations[11]) & (raw.group == "pooled")].iloc[0]
    assert pooled.animals_present == "elephant" and pooled.n_pairs == 1
    assert pooled.n_partial_arm_pairs == 1
    label = cli._support_label(pooled)
    assert "n=1" in label and "partial" in label
    missing = norm[
        (norm.relation == ctx.relations[11]) & (norm.group == "pooled")
    ].iloc[0]
    assert missing.n_pairs == 0 and missing.n_prior_excluded_pairs == 1
    assert np.isnan(missing["median"]) and "omitted" in cli._support_label(missing)


@pytest.mark.parametrize("scale", ["raw", "prior"])
def test_expanded_panels_have_correct_animals_bars_and_screening_legends(tmp_path, monkeypatch, scale):
    ctx = cli._expanded_context(tfp.default_context())
    base = _pairs().assign(prior_available=True, receiver_prior=.05, arm_incomplete=False)
    frame = pd.concat(
        [base.assign(relation=r, animal=a) for r in ctx.relations[:4] for a in tfp.DEFAULT_ANIMALS]
        + [base.assign(relation=r, animal="dog") for r in ctx.relations[4:10]]
        + [base.assign(relation=r, animal="elephant") for r in ctx.relations[10:]],
        ignore_index=True,
    )
    frame.loc[frame.relation == ctx.relations[11], "prior_available"] = False
    summary = cli._summary_table(frame, ctx, n_bootstrap=10, seed=0, scale=scale)
    saved = {}
    monkeypatch.setattr(matplotlib.figure.Figure, "savefig",
                        lambda fig, path, **kw: saved.setdefault(Path(path).name, fig))
    cli._draw_summary(summary, ctx, tmp_path, scale=scale)
    expected = [["pooled", *tfp.DEFAULT_ANIMALS], ["dog"], ["elephant"]]
    for k, (groups, count) in enumerate(zip(expected, [20, 6, 3]), start=1):
        fig = saved[f"transfer_family_balanced_{scale}_percentiles_panel{k}.png"]
        assert all(len(ax.patches) == count for ax in fig.axes)
        labels = [text.get_text() for text in fig.legends[0].get_texts()]
        assert labels == [*groups, "incomplete collapse screening"]
        assert all(not h.get_hatch() for h in fig.legends[0].legend_handles[:-1])
    assert len(saved[f"transfer_family_balanced_{scale}_percentiles.png"].axes) == 4
    if scale == "raw":
        win = saved["transfer_family_balanced_win.png"]
        assert [len(a.get_xticklabels()) for a in win.axes] == [4, 6, 3]
        assert [len(a.patches) for a in win.axes] == [20, 6, 3]
        for ax, groups in zip(win.axes, expected):
            assert [t.get_text() for t in ax.get_legend().get_texts()] == [*groups, "incomplete collapse screening"]
            assert all(not h.get_hatch() for h in ax.get_legend().legend_handles[:-1])
        single = saved["transfer_family_balanced_win_size_architecture.png"].axes[0]
        assert len(single.patches) == 6
        assert [t.get_text() for t in single.get_legend().get_texts()] == ["dog", "incomplete collapse screening"]
        assert all("n=2" in label.get_text() for label in single.get_xticklabels())
    else:
        third = saved["transfer_family_balanced_prior_percentiles_panel3.png"]
        assert np.isnan(third.axes[0].patches[1].get_height())
        assert any(t.get_text() == "n/a" for t in third.axes[0].texts)


def test_cli_includes_registry_pairs_and_preserves_legacy_boxes(tmp_path, monkeypatch):
    from subliminal_learning.analysis import transfer_family_registry_io as registry

    frame, ctx = _extended_frame()
    base = frame[frame.relation == tfp.RELATIONS[0]].drop(
        columns=["receiver_prior", "prior_available"]
    )
    extra = frame[frame.relation != tfp.RELATIONS[0]].drop(
        columns=["receiver_prior", "prior_available"]
    )
    # Include both animals in both architecture directions, with opposite outcomes.
    dogs = pd.concat([extra[extra.animal == "dog"].assign(relation=r)
                      for r in ctx.relations[8:10]], ignore_index=True)
    excluded_elephants = dogs.assign(animal="elephant", test=.05)
    extra = pd.concat([dogs, excluded_elephants, extra[extra.animal == "elephant"]], ignore_index=True)
    root = tmp_path / "outputs"
    (tmp_path / "run_logs").mkdir()
    (tmp_path / "run_logs/core30_cells.json").write_text('{"cells":[]}')
    monkeypatch.setattr(cli, "bootstrap", lambda: (tmp_path, root))
    monkeypatch.setattr(cli, "_teachers", lambda *a: {})
    monkeypatch.setattr(
        tfp, "all_pairs", lambda *a, **kw: (base, tfp.default_context())
    )
    calls = []

    def pairs(*a):
        calls.append("registry")
        return extra, pd.DataFrame([{"status": "partial"}])

    monkeypatch.setattr(registry, "registry_pairs", pairs)
    monkeypatch.setattr(
        cli,
        "_prior_frame",
        lambda f, r: (
            f.assign(prior_available=True, receiver_prior=0.05),
            pd.DataFrame(),
        ),
    )
    monkeypatch.setattr(cli, "_direction_report", lambda *a: pd.DataFrame())
    draw_relations = []
    monkeypatch.setattr(
        cli,
        "_draw_summary",
        lambda summary, c, *a, **kw: draw_relations.append(c.relations),
    )
    box_relations = []

    def box(p, c):
        box_relations.append((tuple(p.relation.unique()), c.relations))
        return cli.plt.subplots()[0]

    monkeypatch.setattr(tfp, "draw_linear", box)
    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", lambda *a, **kw: None)
    out = tmp_path / "figures"
    cli.main(variants="linear", out_dir=out, n_bootstrap=10, seed=0)
    assert calls == ["registry"] and len(draw_relations) == 2
    assert all(len(r) == 13 for r in draw_relations)
    assert all(
        set(seen) <= set(tfp.RELATIONS) and len(r) == 4 for seen, r in box_relations
    )
    table = pd.read_csv(out / "transfer_family_balanced_raw_summary.csv")
    assert set(table.relation) == set(ctx.relations)
    assert (out / "registry_coverage.csv").exists()

    exclusions = pd.read_csv(out / "figure_cohort_exclusions.csv")
    assert len(exclusions) == 4 and exclusions.animal.eq("elephant").all()
    assert set(exclusions.relation) == set(ctx.relations[8:10])
    assert exclusions.exclusion_reason.str.contains("dog-only").all()
    for scale in ("raw", "prior"):
        table = pd.read_csv(out / f"transfer_family_balanced_{scale}_summary.csv")
        for relation in ctx.relations[8:10]:
            rows = table[table.relation == relation].set_index("group")
            assert rows.loc["pooled", "animals_present"] == "dog"
            assert rows.loc["pooled", "n_pairs"] == rows.loc["dog", "n_pairs"] == 2
            assert rows.loc["pooled", "win"] == rows.loc["dog", "win"] == 1
            assert rows.loc["elephant", "n_pairs"] == 0
        olmo = table[(table.relation == ctx.relations[11]) & (table.group == "elephant")].iloc[0]
        assert olmo.n_pairs == 1
    coverage = pd.read_csv(out / "coverage.csv")
    assert not (coverage.relation.isin(ctx.relations[8:10]) & coverage.animal.eq("elephant")).any()
    pd.testing.assert_frame_equal(pd.read_csv(out / "registry_coverage.csv"), pd.DataFrame([{"status": "partial"}]))
