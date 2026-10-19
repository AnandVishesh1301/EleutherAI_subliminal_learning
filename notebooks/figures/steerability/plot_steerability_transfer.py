"""Render the appendix figures for the 102 Pythia-160M dog teachers.

Student transfer is the single student cell per teacher at the ``ep20_lr1e-4_r8a32``
recipe (the recipe shared by every teacher in every receiver), read from the
per-recipe cell table that ``notebooks.figures.predictors.predictor_r2_table`` freezes,
so the figures and the predictor tables use the same outcome. The eight-recipe mean
is kept as ``tr_*_mean8`` but no longer plotted.

Run with ``uv run python -m notebooks.figures.steerability.plot_steerability_transfer``. Partial rank
correlations rank each variable before residualization; their p-values and
Fisher intervals are nominal approximations, not equivalence tests.
"""

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, pearsonr, rankdata, spearmanr, t

ROOT = Path(__file__).resolve().parents[3]
CSV = ROOT / "notebooks/figures/teacher_dce_vs_transfer_102.csv"
CELLS = ROOT / "notebooks/figures/predictors/recipe_cells_102.csv"
TABLE = ROOT / "notebooks/figures/predictors/predictor_r2_table.csv"
RECIPE = "ep20_lr1e-4_r8a32"
STEERABILITY_OUT = ROOT / "notebooks/figures/steerability"
DRAFTS_OUT = STEERABILITY_OUT / "drafts"
STEERING_LABEL = "Increase in p(dog) under steering"
TEACHER_TICKS = [0.5, 0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999]
STUDENT_TICKS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9]


def load_data():
    """Load the frozen plotting population, with its support checked explicitly."""
    frame = pd.read_csv(CSV)
    assert len(frame) == 102 and frame.teacher.nunique() == 102
    assert frame.n_orig.eq(8).all() and frame.n_ds1.eq(8).all()
    frame = frame.rename(columns={"tr_orig": "tr_orig_mean8", "tr_ds1": "tr_ds1_mean8"})
    cells = pd.read_csv(CELLS)
    cells = cells[cells.recipe == RECIPE]
    assert len(cells) == 2 * 102 and set(cells.receiver) == {"orig", "ds1"}
    assert cells[cells.receiver == "orig"].n_seeds.eq(1).all()
    for receiver in ("orig", "ds1"):
        cell = cells[cells.receiver == receiver].set_index("teacher")
        frame[f"tr_{receiver}"] = cell.test_p.loc[frame.teacher].to_numpy()
        frame[f"n_seeds_{receiver}"] = cell.n_seeds.loc[frame.teacher].to_numpy()
    columns = ["teacher_val_p", "steer_ungated", "tr_orig", "tr_ds1"]
    assert np.isfinite(frame[columns].to_numpy()).all()
    return frame


def logit(values):
    values = np.clip(np.asarray(values, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(values / (1 - values))


def partial_stats(x, y, control):
    """Pearson partial for already-transformed variables and one control."""
    x, y, control = (np.asarray(v, dtype=float) for v in (x, y, control))
    design = np.column_stack([np.ones(len(control)), control])
    assert np.linalg.matrix_rank(design) == 2
    rx = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    ry = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    for residual in (rx, ry):
        np.testing.assert_allclose(design.T @ residual, 0, atol=1e-7)
    r = float(pearsonr(rx, ry).statistic)
    xy, xz, yz = (
        pearsonr(a, b).statistic for a, b in [(x, y), (x, control), (y, control)]
    )
    closed_form = (xy - xz * yz) / np.sqrt((1 - xz**2) * (1 - yz**2))
    np.testing.assert_allclose(r, closed_form, atol=1e-12)
    dof = len(x) - 3  # n - one control - two fitted quantities
    p = float(2 * t.sf(abs(r) * np.sqrt(dof / (1 - r**2)), dof))
    ci = np.tanh(
        np.arctanh(r) + np.array([-1, 1]) * norm.ppf(0.975) / np.sqrt(len(x) - 4)
    )
    return (
        {
            "r": r,
            "p_nominal": p,
            "n": len(x),
            "controls": 1,
            "df": dof,
            "approx_fisher_ci95": ci.tolist(),
        },
        rx,
        ry,
    )


def rank_partial(x, y, control):
    return partial_stats(*(rankdata(v, method="average") for v in (x, y, control)))


def _save(fig, stem, out=DRAFTS_OUT):
    out.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"{stem}.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_stats(stem, stats, out=DRAFTS_OUT):
    stats["student_recipe"] = RECIPE
    stats["student_cells_csv"] = str(CELLS.relative_to(ROOT))
    stats["teacher_split"] = "validation"
    stats["student_split"] = "test"
    stats["source_csv"] = str(CSV.relative_to(ROOT))
    stats["source_sha256"] = hashlib.sha256(CSV.read_bytes()).hexdigest()
    (out / f"{stem}_stats.json").write_text(json.dumps(stats, indent=2) + "\n")


def _prob_ticks(ax, axis, candidates):
    lo, hi = getattr(ax, f"get_{axis}lim")()
    kept = [p for p in candidates if lo <= logit(p) <= hi]
    getattr(ax, f"set_{axis}ticks")(logit(kept), labels=[f"{p:g}" for p in kept])
    if axis == "x":
        ax.tick_params(axis="x", labelrotation=40, labelsize=9)


def _scatter(ax, x, y, **kwargs):
    x, y = np.asarray(x), np.asarray(y)
    artist = ax.scatter(x, y, **kwargs)
    np.testing.assert_array_equal(artist.get_offsets().data, np.column_stack([x, y]))
    assert len(artist.get_offsets()) == 102
    return artist


def _teacher_scatter(ax, frame, size=38, *, linear=False, vmin=None, vmax=None):
    if vmin is None or vmax is None:
        vmin, vmax = np.percentile(frame.steer_ungated, [5, 95])
    x = frame.teacher_val_p if linear else logit(frame.teacher_val_p)
    y = frame.tr_orig if linear else logit(frame.tr_orig)
    artist = _scatter(
        ax,
        x,
        y,
        c=frame.steer_ungated,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        s=size,
        edgecolor="none",
        zorder=3,
    )
    np.testing.assert_array_equal(artist.get_array(), frame.steer_ungated.to_numpy())
    scale = "linear" if linear else "logit"
    ax.set_xlabel(f"Teacher p(dog), validation set ({scale} scale)")
    ax.set_ylabel(f"Student p(dog), test set ({scale} scale)")
    ax.grid(alpha=0.2)
    return artist


def _residual_panel(ax, rx, ry, stats, xlabel, ylabel, *, annotate=True):
    _scatter(ax, rx, ry, s=30, color="0.35", alpha=0.85, edgecolor="none", zorder=3)
    xx = np.array([rx.min(), rx.max()])
    ax.plot(xx, np.polyval(np.polyfit(rx, ry, 1), xx), color="0.2", linewidth=1)
    ax.axhline(0, color="0.7", linewidth=0.8)
    ax.axvline(0, color="0.7", linewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.15)
    if annotate:
        low, high = stats["approx_fisher_ci95"]
        ax.text(
            0.035,
            0.975,
            f"Partial ρ = {stats['r']:.3f}; p = {stats['p_nominal']:.3g}\n"
            f"Approx. 95% CI [{low:.2f}, {high:.2f}]\nAdjusted for teacher validation p(dog)",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "none", "pad": 3},
        )


def render_original(frame):
    """Preserve the original plot coordinates and limits; omit selected-teacher rings."""
    x, y = logit(frame.teacher_val_p), logit(frame.tr_orig)
    result = pearsonr(x, y)
    fig, ax = plt.subplots(figsize=(9, 6.5))
    sc = _teacher_scatter(ax, frame, size=170)
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    pad = 0.05 * (hi - lo)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.plot(
        [lo - pad, hi + pad],
        [lo - pad, hi + pad],
        ls="--",
        color="0.4",
        lw=1.4,
        label="student = teacher log-odds",
    )
    _prob_ticks(ax, "x", TEACHER_TICKS[:5])
    _prob_ticks(ax, "y", [0.01, 0.1, 0.5])
    ax.text(
        0.03,
        0.975,
        f"Pearson r (logit scale) = {result.statistic:.3f}  "
        f"(p = {result.pvalue:.2g}, n = {len(frame)})",
        transform=ax.transAxes,
        va="top",
        fontsize=10,
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1.22, 1), frameon=False)
    fig.subplots_adjust(left=0.12, right=0.62, top=0.90, bottom=0.16)
    cax = fig.add_axes([0.655, 0.16, 0.022, 0.74])
    fig.colorbar(sc, cax=cax, extend="both").set_label(STEERING_LABEL, fontsize=9)
    fig.suptitle("Teacher preference tracks same-model transfer", fontsize=12)
    assert len(ax.collections) == 1  # no selected-teacher overlay
    _save(fig, "steerability_logit_student_vs_teacher", DRAFTS_OUT)


def render_combined(frame):
    teacher, steer, student = frame.teacher_val_p, frame.steer_ungated, frame.tr_orig
    stats, rx, ry = rank_partial(steer, student, teacher)
    sensitivity, _, _ = partial_stats(steer, logit(student), logit(teacher))
    marginal = pearsonr(logit(teacher), logit(student))
    rank_r = float(spearmanr(teacher, student).statistic)
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.2, 4.8), layout="constrained")
    sc = _teacher_scatter(a, frame)
    a.margins(x=0.07, y=0.1)
    _prob_ticks(a, "x", TEACHER_TICKS)
    _prob_ticks(a, "y", STUDENT_TICKS)
    cax = a.inset_axes([0.05, 0.87, 0.42, 0.03])
    fig.colorbar(sc, cax=cax, orientation="horizontal", extend="both").set_label(
        STEERING_LABEL, fontsize=7
    )
    _residual_panel(
        b,
        rx,
        ry,
        stats,
        "Steering-response residual (rank units)",
        "Transfer residual (rank units)",
        annotate=False,
    )
    _save(fig, "steerability", STEERABILITY_OUT)
    _write_stats(
        "steerability",
        {
            "n": len(frame),
            "student_seeds_per_teacher": int(frame.n_seeds_orig.iloc[0]),
            "teacher_student_pearson_logit": float(marginal.statistic),
            "teacher_student_spearman": rank_r,
            "partial_spearman": stats,
            "raw_steer_logit_outcome_and_control": sensitivity,
        },
        STEERABILITY_OUT,
    )
    return stats, sensitivity


def render_combined_linear_middle(frame):
    """Render the combined view with a linear-probability middle panel."""
    teacher, steer, student = frame.teacher_val_p, frame.steer_ungated, frame.tr_orig
    stats, rx, ry = rank_partial(steer, student, teacher)
    sensitivity, _, _ = partial_stats(steer, logit(student), logit(teacher))
    fig, (a, middle, b) = plt.subplots(1, 3, figsize=(16.5, 4.9), layout="constrained")
    sc = _teacher_scatter(a, frame)
    a.margins(x=0.07, y=0.1)
    _prob_ticks(a, "x", TEACHER_TICKS)
    _prob_ticks(a, "y", STUDENT_TICKS)
    _teacher_scatter(middle, frame, linear=True)
    middle.set_xlim(0, 1.01)
    middle.set_ylim(0, 1.01)
    middle.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    middle.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    cax = a.inset_axes([0.05, 0.87, 0.42, 0.03])
    fig.colorbar(sc, cax=cax, orientation="horizontal", extend="both").set_label(
        STEERING_LABEL, fontsize=7
    )
    _residual_panel(
        b,
        rx,
        ry,
        stats,
        "Steering-response residual (rank units)",
        "Transfer residual (rank units)",
        annotate=False,
    )
    _save(fig, "steerability_teacher_strength_adjusted_linear_middle", DRAFTS_OUT)
    _write_stats(
        "steerability_teacher_strength_adjusted_linear_middle",
        {
            "n": len(frame),
            "partial_spearman_validation": stats,
            "raw_steer_logit_outcome_and_control": sensitivity,
        },
        DRAFTS_OUT,
    )
    return stats, sensitivity


def render_same_cross(frame):
    x, y, control = frame.tr_orig, frame.tr_ds1, frame.teacher_val_p
    stats, rx, ry = rank_partial(x, y, control)
    logit_sensitivity, _, _ = partial_stats(logit(x), logit(y), logit(control))
    marginal = float(spearmanr(x, y).statistic)
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.8, 5), layout="constrained")
    fig.suptitle(
        "Same-model transfer retains an adjusted association with cross-model transfer\n"
        "Fixed sender: original Pythia-160M; cross-model receiver: data seed 1",
        fontsize=12,
    )
    _scatter(a, logit(x), logit(y), color="0.35", s=30, alpha=0.85, edgecolor="none")
    a.margins(x=0.08, y=0.12)
    _prob_ticks(a, "x", STUDENT_TICKS)
    _prob_ticks(a, "y", STUDENT_TICKS)
    a.set_xlabel("Same-model student p(dog) (logit scale)")
    a.set_ylabel("Cross-model student p(dog) (logit scale)")
    a.set_title("(a) Unadjusted association", fontsize=11)
    a.grid(alpha=0.15)
    a.text(
        0.035,
        0.975,
        f"Spearman ρ = {marginal:.3f}; n = {len(frame)}",
        transform=a.transAxes,
        va="top",
        fontsize=9,
    )
    _residual_panel(
        b,
        rx,
        ry,
        stats,
        "Same-model transfer residual (rank units)",
        "Cross-model transfer residual (rank units)",
    )
    b.set_title("(b) Adjusted association", fontsize=11)
    _save(fig, "same_vs_cross_transfer_teacher_adjusted")
    _write_stats(
        "same_vs_cross_transfer_teacher_adjusted",
        {
            "n": len(frame),
            "marginal_spearman": marginal,
            "partial_spearman_validation": stats,
            "partial_logit_validation": logit_sensitivity,
        },
    )
    return stats


def main():
    frame = load_data()
    render_original(frame)
    steering, logit_steering = render_combined(frame)
    render_combined_linear_middle(frame)
    same_cross = render_same_cross(frame)
    # The predictor table computes the same rank partials from the same cells by
    # a separate code path (notebooks.figures.predictors.predictor_r2_table).
    table = pd.read_csv(TABLE)
    table = table[(table.recipe == RECIPE) & (table.scale == "rank")].set_index(["outcome", "predictor"])
    np.testing.assert_allclose(
        [steering["r"], same_cross["r"]],
        [
            table.partial_r.loc["Same-model transfer (original receiver)", "Steerability"],
            table.partial_r.loc["Cross-model transfer (data-seed-1 receiver)", "Same-model transfer"],
        ],
        atol=1e-9,
    )
    print(
        json.dumps(
            {
                "steering_rank": steering,
                "steering_logit": logit_steering,
                "same_cross_rank": same_cross,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
