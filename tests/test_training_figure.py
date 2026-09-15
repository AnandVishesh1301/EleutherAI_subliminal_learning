import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from docs.paper import plot_iclr_boundary as plot

GAP_REASON = (
    "complete separation: a plane in (ln lr, ln alpha, ln rank) splits all outcomes; "
    "unpenalized MLE has no finite maximizer"
)


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    evidence = tmp_path / "figures_20260909"
    evidence.mkdir()
    qdir = tmp_path / "qwen_20260909"
    qdir.mkdir()
    dest = tmp_path / "figures"
    dest.mkdir()
    rows = []
    for arm, budget in [
        ("160m same-model", 25000),
        ("160m cross-init ds1", 25000),
        ("410m same-model", 25000),
        ("410m cross-seed", 25000),
        ("OLMo-1B same-model", 12500),
        ("OLMo-1B cross-seed 2", 12500),
    ]:
        for step in [625, 1250, budget]:
            rows.append(
                dict(
                    arm=arm,
                    step=step,
                    n=40,
                    n_dead=20,
                    b=0.6,
                    c=-0.1,
                    kstar_rel=2 - step / budget,
                    b_svm=0.62,
                    c_svm=-0.12,
                    svm_hard_margin=(step == 1250 and arm.endswith("same-model")),
                    # The OLMo gap is quasi-separated: substituted once ties are
                    # removed, since its separating set is still narrow.
                    svm_kind="hard_margin_ties_removed"
                    if arm.startswith("OLMo")
                    else "hard_margin",
                    svm_n_tied=4 if arm.startswith("OLMo") else 0,
                    b_set_lo=0.60 if step == 1250 else np.nan,
                    b_set_hi=0.64 if step == 1250 else np.nan,
                    c_set_lo=-0.14 if step == 1250 else np.nan,
                    c_set_hi=-0.10 if step == 1250 else np.nan,
                    converged=not (step == 1250 and arm.endswith("same-model")),
                    separation=(
                        ("quasi" if arm.startswith("OLMo") else "complete")
                        if (step == 1250 and arm.endswith("same-model"))
                        else ""
                    ),
                    reason=GAP_REASON
                    if (step == 1250 and arm.endswith("same-model"))
                    else "",
                )
            )
    pd.DataFrame(rows).to_csv(evidence / "fig03_checkpoint_fits.csv", index=False)
    q = pd.DataFrame(
        [
            dict(
                arm="Qwen2.5-1.5B same-model",
                step=625,
                n=40,
                n_dead=20,
                b=0.7,
                c=-0.05,
                b_svm=0.72,
                c_svm=-0.06,
                kstar_rel=1.1,
                converged=True,
            )
        ]
    )
    q.to_csv(qdir / "qwen_percheckpoint_freefits.csv", index=False)
    teacher = pd.DataFrame(
        [
            dict(
                arm="teachers ep20 (dog|orig)",
                step=s,
                n=504,
                n_dead=249,
                b=np.nan if s == 3 else 0.85,
                c=np.nan if s == 3 else -0.3,
                b_svm=1.0 if s == 3 else 0.87,
                c_svm=0.0 if s == 3 else -0.31,
                # Ties removed, but the separating set is far too wide to draw.
                b_set_lo=0.333 if s == 3 else np.nan,
                b_set_hi=3.322 if s == 3 else np.nan,
                c_set_lo=-0.286 if s == 3 else np.nan,
                c_set_hi=0.286 if s == 3 else np.nan,
                svm_kind="hard_margin_ties_removed" if s == 3 else "hard_margin",
                svm_n_tied=24 if s == 3 else 0,
                separation="quasi" if s == 3 else "",
                kstar_rel=1,
                converged=s != 3,
            )
            for s in [3, 60]
        ]
    )
    teacher.to_csv(evidence / "teacher_checkpoint_fits.csv", index=False)
    monkeypatch.setattr(plot, "EVIDENCE", evidence)
    monkeypatch.setattr(plot, "DEST", dest)
    captured = []
    real_save = plot.save

    def save(fig, name):
        captured.append(fig)
        real_save(fig, name)

    monkeypatch.setattr(plot, "save", save)
    return evidence, dest, captured


def test_training_render_uses_planned_budgets_and_preserves_fit_gaps(inputs):
    evidence, dest, figures = inputs
    plot.training()
    data = pd.read_csv(evidence / "fig03_checkpoint_fits_all.csv")
    assert len(data) == 21 and data.arm.nunique() == 8
    expected = {
        "160m same-model": 2.5,
        "OLMo-1B same-model": 5,
        "Qwen2.5-1.5B same-model": 10,
        "teachers ep20 (dog|orig)": 5,
    }
    for arm, first in expected.items():
        assert data[data.arm == arm].training_percent.iloc[0] == first
    assert data[data.arm == "Qwen2.5-1.5B same-model"].training_percent.max() == 10
    assert data[data.arm == "teachers ep20 (dog|orig)"].training_percent.max() == 100
    fig = figures[1]
    a, b, c, lr = fig.axes
    assert a.get_xscale() == "linear" and a.get_xlim() == (0, 100)
    # Five group lines, the endpoint reference and the SVM legend proxy.
    assert len(a.lines) == 7
    for axis in [b, c]:
        assert len([ln for ln in axis.lines if ln.get_marker() == "o"]) == 5
        # Substitution markers only where a member arm lost its MLE.
        assert len(axis.containers) == 3
        # Pythia groups lose their MLE at 5%, the OLMo group at 10%.
        assert [
            container.lines[0].get_xdata().tolist() for container in axis.containers
        ] == [[5.0], [5.0], [10.0]]
        olmo = next(line for line in axis.lines if line.get_label().startswith("OLMo"))
        # The gap is filled by the substituted value, not left blank.
        assert np.isfinite(olmo.get_ydata()).all()
        assert all(
            container.lines[0].get_markerfacecolor() == "none"
            for container in axis.containers
        )
    assert len(figures[0].axes) == 4
    arm_axis = figures[0].axes[1]
    assert len([ln for ln in arm_axis.lines if ln.get_marker() == "o"]) == 8
    assert len(arm_axis.containers) == 3
    grouped = pd.read_csv(evidence / "fig03_checkpoint_fits_grouped.csv").set_index(
        ["group", "training_percent"]
    )
    assert {"b_sub_mean", "c_sub_mean", "b_set_lo", "c_set_hi"} <= set(grouped.columns)
    olmo_row = grouped.loc[("OLMo-1B · number distillation", 10.0)]
    assert bool(olmo_row.b_substituted) and np.isnan(olmo_row.b_mean)
    assert olmo_row.b_sub_mean == pytest.approx((0.6 + 0.62) / 2)
    # The group bar averages the substituted arm's set with the other arm's MLE.
    assert olmo_row.b_set_lo == pytest.approx((0.6 + 0.60) / 2)
    assert olmo_row.b_set_hi == pytest.approx((0.6 + 0.64) / 2)
    assert olmo_row.b_set_lo <= olmo_row.b_sub_mean <= olmo_row.b_set_hi
    assert olmo_row.c_set_lo <= olmo_row.c_sub_mean <= olmo_row.c_set_hi
    assert not grouped.loc[("OLMo-1B · number distillation", 5.0)].b_substituted
    # The teacher gap fails the set-width gate, so it is never filled.
    teacher = grouped.loc[("Pythia-160M · animal preference (20 epochs)", 5.0)]
    assert not teacher.b_substituted and np.isnan(teacher.b_mean)
    assert np.isnan(teacher.b_sub_mean) and np.isnan(teacher.b_set_lo)
    meta = json.loads((evidence / "fig03_training_metadata.json").read_text())
    assert meta["svm"]["C"] == 1e4 and "comparison" in meta["svm"]["estimator"]
    rejected = meta["svm"]["gate_rejected"]
    assert [r["arm"] for r in rejected] == ["teachers ep20 (dog|orig)"]
    assert rejected[0]["step"] == 3 and rejected[0]["svm_kind"] == (
        "hard_margin_ties_removed"
    )
    assert rejected[0]["b_set_width"] == pytest.approx(3.322 - 0.333)
    assert rejected[0]["c_set_width"] == pytest.approx(0.572)
    olmo_svm = meta["svm"]["per_arm"]["OLMo-1B same-model"]
    assert olmo_svm["n_compared"] == 2
    assert olmo_svm["gap_values"] == [
        {"step": 1250, "b_svm": 0.62, "c_svm": -0.12, "hard_margin": True}
    ]
    olmo_k = next(line for line in a.lines if line.get_label().startswith("OLMo"))
    assert np.isfinite(olmo_k.get_ydata()).all()
    x, y = lr.lines[0].get_data()
    assert y[0] == 0 and y[-1] == 0 and y[x == 5][0] == 1
    gaps = json.loads((evidence / "fig03_training_metadata.json").read_text())[
        "gap_records"
    ]
    # Three complete-separation gaps plus the teacher's quasi checkpoint.
    assert len(gaps) == 4 and gaps[0]["step"] == 1250 and gaps[0]["n"] == 40
    assert [g["reason"] for g in gaps[:3]] == [GAP_REASON] * 3
    assert set(data.separation.fillna("")) == {"", "complete", "quasi"}
    for ext in ["png", "pdf"]:
        assert (dest / f"fig03_training.{ext}").stat().st_size > 1000
        assert (dest / f"figS_training_by_arm.{ext}").stat().st_size > 1000
    plt.close("all")


def test_unknown_arm_fails_closed(inputs):
    evidence, _, _ = inputs
    path = evidence / "fig03_checkpoint_fits.csv"
    d = pd.read_csv(path)
    d.loc[0, "arm"] = "unknown arm"
    d.to_csv(path, index=False)
    with pytest.raises(ValueError, match="No planned training budget"):
        plot.training()


def test_teacher_input_is_required(inputs):
    evidence, _, _ = inputs
    (evidence / "teacher_checkpoint_fits.csv").unlink()
    with pytest.raises(FileNotFoundError, match="Teacher trajectory evidence required"):
        plot.training()


def test_out_of_budget_checkpoint_fails(inputs):
    evidence, _, _ = inputs
    path = evidence / "teacher_checkpoint_fits.csv"
    d = pd.read_csv(path)
    d.loc[0, "step"] = 61
    d.to_csv(path, index=False)
    with pytest.raises(ValueError, match="outside the planned training budget"):
        plot.training()


def test_aggregation_equal_weights_and_missing_members():
    d = pd.DataFrame(
        [
            dict(
                arm="160m same-model",
                training_percent=5,
                b=0.6,
                c=-0.1,
                kstar_rel=1,
                n=100,
            ),
            dict(
                arm="160m cross-init ds1",
                training_percent=5,
                b=0.8,
                c=-0.2,
                kstar_rel=3,
                n=900,
            ),
            dict(
                arm="160m same-model",
                training_percent=10,
                b=0.6,
                c=-0.1,
                kstar_rel=1,
                n=100,
            ),
            dict(
                arm="160m cross-init ds1",
                training_percent=10,
                b=np.nan,
                c=np.nan,
                kstar_rel=2,
                n=900,
            ),
            dict(
                arm="160m same-model",
                training_percent=15,
                b=0.6,
                c=-0.1,
                kstar_rel=1,
                n=100,
            ),
        ]
    )
    result = plot.aggregate_training(d).set_index("training_percent")
    first = result.loc[5]
    assert first.n_expected == first.n_valid == 2
    assert (
        first.b_mean == pytest.approx(0.7) and first.b_min == 0.6 and first.b_max == 0.8
    )
    assert first.kstar_rel_mean == 2  # Not weighted by n.
    assert np.isnan(result.loc[10].b_mean) and result.loc[10].b_n_valid == 1
    assert (
        result.loc[10].kstar_rel_n_valid == 2 and result.loc[10].kstar_rel_mean == 1.5
    )
    assert result.loc[15].n_expected == 2 and np.isnan(result.loc[15].kstar_rel_mean)
    with pytest.raises(ValueError, match="Duplicate"):
        plot.aggregate_training(pd.concat([d, d.iloc[[0]]]))


def test_groups_keep_width_and_task_separate():
    rows = [
        dict(arm=arm, training_percent=100, b=i, c=-i, kstar_rel=1)
        for i, arm in enumerate(plot.GROUPS)
    ]
    result = plot.aggregate_training(pd.DataFrame(rows))
    assert len(result) == 5
    teacher = result[result.group.str.contains("animal preference")].iloc[0]
    assert teacher.n_expected == teacher.n_valid == 1
    assert teacher.b_min == teacher.b_mean == teacher.b_max == 7
    assert result.kstar_rel_mean.eq(1).all()
