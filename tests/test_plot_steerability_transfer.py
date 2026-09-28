import numpy as np
from scipy.stats import pearsonr, rankdata

from scripts.plot_steerability_transfer import load_data, partial_stats, rank_partial


def test_validation_controls_are_used_and_rank_residuals_match_closed_form():
    frame = load_data()
    fav = frame.teacher_fav_p.to_numpy()
    val = frame.teacher_val_p.to_numpy()
    assert not np.allclose(fav, val)
    stats, rx, ry = rank_partial(frame.steer_ungated, frame.tr_orig, val)
    xr = rankdata(frame.steer_ungated)
    yr = rankdata(frame.tr_orig)
    zr = rankdata(val)
    q = np.column_stack([np.ones(len(val)), zr])
    ex = xr - q @ np.linalg.lstsq(q, xr, rcond=None)[0]
    ey = yr - q @ np.linalg.lstsq(q, yr, rcond=None)[0]
    assert np.allclose(rx, ex)
    assert np.allclose(ry, ey)
    assert np.isclose(stats["r"], pearsonr(ex, ey).statistic)
    assert stats["n"] == 102 and stats["df"] == 99


def test_same_cross_validation_stats_and_split_metadata(tmp_path):
    frame = load_data()
    expected, *_ = rank_partial(frame.tr_orig, frame.tr_ds1, frame.teacher_val_p)
    favorite, *_ = rank_partial(frame.tr_orig, frame.tr_ds1, frame.teacher_fav_p)
    assert not np.isclose(expected["r"], favorite["r"])
    assert np.isclose(expected["r"], 0.3100685897911577)
    transformed, *_ = partial_stats(
        np.log(
            np.clip(frame.tr_orig, 1e-6, 1 - 1e-6)
            / (1 - np.clip(frame.tr_orig, 1e-6, 1 - 1e-6))
        ),
        np.log(
            np.clip(frame.tr_ds1, 1e-6, 1 - 1e-6)
            / (1 - np.clip(frame.tr_ds1, 1e-6, 1 - 1e-6))
        ),
        np.log(
            np.clip(frame.teacher_val_p, 1e-6, 1 - 1e-6)
            / (1 - np.clip(frame.teacher_val_p, 1e-6, 1 - 1e-6))
        ),
    )
    assert transformed["n"] == 102


def test_rendered_panels_use_validation_coordinates_and_residuals(monkeypatch):
    import matplotlib.pyplot as plt
    from scripts import plot_steerability_transfer as plot

    frame = plot.load_data()
    saved = {}

    def capture(fig, stem, out=None):
        saved[stem] = fig
        plt.close(fig)

    monkeypatch.setattr(plot, "_save", capture)
    monkeypatch.setattr(plot, "_write_stats", lambda *args, **kwargs: None)
    plot.render_original(frame)
    plot.render_combined(frame)
    plot.render_combined_linear_middle(frame)
    plot.render_same_cross(frame)
    expected_logit = np.column_stack(
        [plot.logit(frame.teacher_val_p), plot.logit(frame.tr_orig)]
    )
    for stem in (
        "steerability_logit_student_vs_teacher",
        "steerability_teacher_strength_adjusted",
        "steerability_teacher_strength_adjusted_linear_middle",
    ):
        np.testing.assert_allclose(
            saved[stem].axes[0].collections[0].get_offsets(), expected_logit
        )
    np.testing.assert_allclose(
        saved["steerability_teacher_strength_adjusted_linear_middle"]
        .axes[1]
        .collections[0]
        .get_offsets(),
        np.column_stack([frame.teacher_val_p, frame.tr_orig]),
    )
    design = np.column_stack([np.ones(len(frame)), rankdata(frame.teacher_val_p)])

    def residual(values):
        ranked = rankdata(values)
        return ranked - design @ np.linalg.lstsq(design, ranked, rcond=None)[0]

    steering_expected = np.column_stack(
        [residual(frame.steer_ungated), residual(frame.tr_orig)]
    )
    for stem in (
        "steerability_teacher_strength_adjusted",
        "steerability_teacher_strength_adjusted_linear_middle",
    ):
        np.testing.assert_allclose(
            saved[stem].axes[-1].collections[0].get_offsets(), steering_expected
        )
    np.testing.assert_allclose(
        saved["same_vs_cross_transfer_teacher_adjusted"]
        .axes[1]
        .collections[0]
        .get_offsets(),
        np.column_stack([residual(frame.tr_orig), residual(frame.tr_ds1)]),
    )
