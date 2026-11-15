"""Magnitude of transfer among exceeding students: the exceedance-only replacement for
``figure2.png`` + ``figure2_olmo.png``, in the look of ``newdom_traits_exceedances.png``.

uv run python -m subliminal_learning.figures.figure2_family.exceedances.plot_exceedance_magnitudes
uv run python -m subliminal_learning.figures.figure2_family.exceedances.plot_exceedance_magnitudes --render-only

* **Arms**: the condition blocks of the paper's significance tables
  (`analysis.transfer.transfer_conditions.condition_blocks`, diverged runs excluded), so every
  count here equals the existence table's.
* **Exceedance**: a treatment student whose target-animal probability is above every control of
  its arm (teacher base x student base x animal).
* **Δp**: the student minus the control trained with the same student recipe.
* **Bar**: mean Δp over ALL exceeding students of the bar, with a 95% bootstrap interval.
* **Dots**: `N_DOTS` evenly spaced percentiles of those students (0.5th, 1.5th, ...; Angela
  2026-11-11: thousands of single-student dots were unreadable and a random subsample would be
  arbitrary), or every exceeding student when there are no more than `N_DOTS`. ▲ = a dot above
  the panel's axis top.

Outputs, in ``notebooks/figures/figure2_family/exceedances/``: ``figure2_exceedances.{png,pdf}``,
``figure2_exceedances_points.csv`` (one row per exceeding student) and
``figure2_exceedances_students.csv`` (treatment students per bar) -- the two files
``--render-only`` reads -- and ``figure2_exceedances_bars.csv`` (one row per bar: students,
exceedances, mean, interval).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from matplotlib.patches import Patch

from subliminal_learning.analysis.setup import bootstrap

app = typer.Typer(add_completion=False)

DEFAULT_OUT = Path("notebooks/figures/figure2_family/exceedances")
STEM = "figure2_exceedances"
#: Dots per bar: evenly spaced percentiles of the bar's exceeding students.
N_DOTS = 100
ANIMALS = ["dog", "cat", "lion", "pig"]
#: Figure 2's animal colours and legend order.
ANIMAL_COLOR = {"cat": "#168BCE", "dog": "#F06418", "lion": "#16A05D", "pig": "#DE4FA6", "elephant": "#9467bd"}
LEGEND_ORDER = ["cat", "dog", "lion", "pig"]
INK, GRID = "#222222", "#e5e5e5"
RC = {"font.size": 10, "pdf.fonttype": 42, "svg.fonttype": "none"}
YLABEL = "p(student) − p(control), exceedances only"

PYTHIA, OLMO, XARCH = "Pythia", "DataDecide OLMo-1B", "Pythia/OLMo Cross Arch."
#: Panels: (key, title, axis top, [(family, condition, tick label)]). A panel's bars are one per
#: condition x animal; the fixed axis tops keep the cross-model bars readable (▲ beyond them).
PANELS = [
    ("A", "Pythia 160M: same-model", 1.0, [(PYTHIA, "Same-model", "Same-model")]),
    ("B", "Pythia 160M: different init or data order", 0.5, [
        (PYTHIA, "Same init, diff. data order", "Same init,\ndifferent data order"),
        (PYTHIA, "Diff. init, same data order", "Different init,\nsame data order"),
        (PYTHIA, "Diff. init, diff. data order", "Different init,\ndifferent data order")]),
    ("C", "Pythia size and architecture transfers — dog", 0.2, [
        (PYTHIA, "Diff. model size, same data order, small to large", "160M → 410M\nsame\ndata order"),
        (PYTHIA, "Diff. model size, same data order, large to small", "410M → 160M\nsame\ndata order"),
        (PYTHIA, "Diff. model size, diff. data order, small to large", "160M → 410M\ndifferent\ndata order"),
        (PYTHIA, "Diff. model size, diff. data order, large to small", "410M → 160M\ndifferent\ndata order"),
        (XARCH, "Diff. model arch., diff. data corpus, small to large", "160M →\nOLMO-1B"),
        (XARCH, "Diff. model arch., diff. data corpus, large to small", "OLMO-1B →\n160M")]),
    ("D", "OLMo relationships — elephant", 0.52, [
        (OLMO, "Same-model", "Same-model"),
        (OLMO, "Same init, diff. data corpus", "Same init,\ndifferent corpus"),
        (OLMO, "Diff. init, diff. data corpus", "Different init,\ndifferent corpus"),
        (OLMO, "Diff. init, same data corpus", "Different init,\nsame corpus")]),
]


def _norm(text: str) -> str:
    return " ".join(str(text).replace("\n", " ").split()).lower()


def collect(repo: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(points, students)``: one row per exceeding student of every panel condition, and the
    number of treatment students per (family, condition, animal)."""
    from subliminal_learning.analysis.transfer import transfer_conditions as tc
    from subliminal_learning.analysis.transfer import transfer_headline as th

    teachers = {a: json.loads((repo / f"run_logs/registry/{a}_top10_kband_configs.json").read_text())["param_tags"]
                for a in ANIMALS}
    blocks = tc.condition_blocks(repo, teachers, diverged=th.DIVERGED_EXCLUDE)
    rows, counts = [], []
    for _, _, _, conditions in PANELS:
        for family, condition, _ in conditions:
            (block,) = [b for b in blocks if b.family == family and _norm(b.condition) == _norm(condition)]
            for arm in block.arms:
                m = arm.matrix
                if m is None:
                    continue
                ctrl = m.values[:, m.control_col]
                threshold = float(np.nanmax(ctrl))
                teachers_of = [p for j, p in enumerate(m.pools) if j != m.control_col]
                treat = np.delete(m.values, m.control_col, axis=1)
                counts.append({"family": family, "condition": condition, "animal": arm.animal, "arm": arm.label,
                               "students": int(np.isfinite(treat).sum())})
                for i, j in zip(*np.nonzero(np.isfinite(treat) & (treat > threshold)), strict=True):
                    rows.append({"family": family, "condition": condition, "animal": arm.animal, "arm": arm.label,
                                 "teacher": teachers_of[j], "cell": m.cells[i], "student": float(treat[i, j]),
                                 "control": float(ctrl[i]), "arm_control_max": threshold})
    students = (pd.DataFrame(counts).groupby(["family", "condition", "animal"], sort=False)["students"].sum()
                .reset_index())
    return pd.DataFrame(rows), students


def boot_ci(v: np.ndarray, rng: np.random.Generator, n: int = 2000) -> tuple[float, float]:
    means = v[rng.integers(0, len(v), size=(n, len(v)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def percentile_dots(v: np.ndarray) -> np.ndarray:
    """`N_DOTS` evenly spaced percentiles of `v`, or `v` itself when it has no more values."""
    return v if len(v) <= N_DOTS else np.quantile(v, (np.arange(N_DOTS) + 0.5) / N_DOTS)


def _bar(ax, x: float, width: float, every: np.ndarray, color: str, ymax: float, rng, jit) -> dict:
    """One bar (mean + interval over `every` exceeding student) with its percentile dots."""
    v = percentile_dots(every)
    mean = float(every.mean())
    ax.bar(x, mean, width * 0.91, color=color, edgecolor="none", zorder=2)
    jitter = (np.array([jit.random() for _ in v]) - 0.5) * width * 0.66
    inside = v <= ymax
    dot = dict(color=color, edgecolors="white", linewidths=0.3, zorder=3, clip_on=False)
    ax.scatter(x + jitter[inside], v[inside], s=9, **dot)
    ax.scatter(x + jitter[~inside], np.full(int((~inside).sum()), ymax), s=18, marker="^", **dot)
    lo = hi = float("nan")
    if len(every) > 1:
        lo, hi = boot_ci(every, rng)
        cap = min(0.045, width * 0.3)
        ax.vlines(x, lo, hi, color=INK, lw=1.1, zorder=4)
        ax.hlines([lo, hi], x - cap, x + cap, color=INK, lw=1.1, zorder=4)
    return {"exceedances": len(every), "mean": mean, "ci_lo": lo, "ci_hi": hi, "median": float(np.median(every)),
            "max": float(every.max()), "dots": len(v), "dots_beyond_axis": int((~inside).sum())}


def _frame(ax, title: str, ymax: float, labels: list[str], *, small: bool) -> None:
    ax.set_ylim(0, ymax)
    step = 0.2 if ymax > 0.6 else (0.1 if ymax > 0.25 else 0.05)
    ax.set_yticks(np.arange(0, ymax + 1e-9, step))
    ax.set_xticks(range(len(labels)), labels, fontsize=8.5 if small else 9)
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=7)
    ax.set_title(title, loc="left", fontsize=12)


def draw(points: pd.DataFrame, students: pd.DataFrame, out: Path) -> pd.DataFrame:
    """Draw the four panels to `out` (+ ``.pdf``); returns the per-bar summary."""
    delta = {k: (g["student"] - g["control"]).to_numpy(float)
             for k, g in points.groupby(["family", "condition", "animal"], sort=False)}
    n_students = {(r.family, r.condition, r.animal): int(r.students) for r in students.itertuples()}
    bars = []
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(12.5, 8.3))
        top = fig.add_gridspec(1, 2, width_ratios=[1, 3.1], left=0.065, right=0.99, top=0.95, bottom=0.585,
                               wspace=0.11)
        bot = fig.add_gridspec(1, 2, width_ratios=[1.45, 1], left=0.065, right=0.99, top=0.455, bottom=0.085,
                               wspace=0.11)
        axes = [fig.add_subplot(top[0]), fig.add_subplot(top[1]), fig.add_subplot(bot[0]), fig.add_subplot(bot[1])]
        for ax, (key, title, ymax, conditions) in zip(axes, PANELS, strict=True):
            rng, jit = np.random.default_rng(0), random.Random(0)
            animals = [a for a in [*LEGEND_ORDER, "elephant"]
                       if any((f, c, a) in n_students for f, c, _ in conditions)]
            k = len(animals)
            width = 0.8 / k if k > 1 else 0.6
            for gi, (family, condition, _) in enumerate(conditions):
                for ai, animal in enumerate(animals):
                    every = delta.get((family, condition, animal))
                    if every is None:
                        continue
                    row = _bar(ax, gi + (ai - (k - 1) / 2) * width, width, every, ANIMAL_COLOR[animal], ymax, rng,
                               jit)
                    bars.append({"panel": key, "family": family, "condition": condition, "animal": animal,
                                 "students": n_students[(family, condition, animal)], **row})
            _frame(ax, f"{key}   {title}", ymax, [label for _, _, label in conditions], small=len(conditions) > 4)
            if key == "B":
                ax.legend(handles=[Patch(facecolor=ANIMAL_COLOR[a], edgecolor="none", label=a) for a in animals],
                          loc="upper center", ncol=k, fontsize=9, frameon=False, handlelength=1.3,
                          columnspacing=1.2)
        for ax in (axes[0], axes[2]):
            ax.set_ylabel(YLABEL, fontsize=9)
        fig.savefig(out, dpi=200)
        fig.savefig(out.with_suffix(".pdf"))
        plt.close(fig)
    return pd.DataFrame(bars)


@app.command()
def main(
    out_dir: Path = typer.Option(DEFAULT_OUT, help="output directory, repo-relative"),
    render_only: bool = typer.Option(False, help="redraw from the saved points CSV, no run-tree read"),
) -> None:
    repo, _ = bootstrap()
    out = repo / out_dir
    out.mkdir(parents=True, exist_ok=True)
    points_csv, students_csv = out / f"{STEM}_points.csv", out / f"{STEM}_students.csv"
    if render_only:
        points, students = pd.read_csv(points_csv), pd.read_csv(students_csv)
    else:
        points, students = collect(repo)
        points.to_csv(points_csv, index=False)
        students.to_csv(students_csv, index=False)
    bars = draw(points, students, out / f"{STEM}.png")
    bars.to_csv(out / f"{STEM}_bars.csv", index=False)
    print(f"wrote {out / (STEM + '.png')}")
    print(bars.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    app()
