import csv
from pathlib import Path
import numpy as np
import pytest
from notebooks.figures.crossckpt_partials import plot_checkpoint_orig_early_mature as first
from notebooks.figures.crossckpt_partials.plot_checkpoint_orig_early_mature import read_cells


def row(t, s, p):
    return {
        "lineage": "orig",
        "teacher_step": str(t),
        "student_step": str(s),
        "complete_five_recipes": "True",
        "p_target": str(p),
    }


def write(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_reads_exact_four_cells(tmp_path):
    p = tmp_path / "cells.csv"
    write(p, [row(t, s, 0.1) for t in (75000, 125000) for s in (75000, 125000)])
    assert len(read_cells(p)) == 4


def test_rejects_missing_or_incomplete_cell(tmp_path):
    p = tmp_path / "cells.csv"
    rows = [row(t, s, 0.1) for t in (75000, 125000) for s in (75000, 125000)]
    rows[-1]["complete_five_recipes"] = "False"
    write(p, rows)
    with pytest.raises(ValueError):
        read_cells(p)


def test_rejects_duplicate_cell(tmp_path):
    p = tmp_path / "cells.csv"
    rows = [row(t, s, 0.1) for t in (75000, 125000) for s in (75000, 125000)]
    rows[-1]["teacher_step"] = "75000"
    write(p, rows)
    with pytest.raises(ValueError):
        read_cells(p)


def test_reads_complete_original_14x14_slice(tmp_path):
    p = tmp_path / "cells.csv"
    rows = []
    plot_steps = (
        50000,
        75000,
        80000,
        85000,
        90000,
        95000,
        100000,
        105000,
        110000,
        115000,
        120000,
        125000,
        130000,
        143000,
    )
    for t in plot_steps:
        for s in plot_steps:
            rows.append(row(t, s, 0.1))
    write(p, rows)
    assert len(first.read_slice_cells(p)) == 196


@pytest.mark.parametrize(
    "view, basename",
    [
        ("slices", "crossckpt_partials/slices_cell_means"),
        ("heatmap", "crossckpt_partials/heatmap_2x2_75k125k"),
        ("multi-slices", "crossckpt_partials/slice_teacher"),
    ],
)
def test_views_dispatch_to_distinct_default_outputs(
    tmp_path, monkeypatch, view, basename
):
    import sys

    calls = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(first, "read_slice_cells", lambda path: "slice-data")
    monkeypatch.setattr(first, "read_cells", lambda path: "heatmap-data")
    monkeypatch.setattr(first, "output_root", lambda: tmp_path)
    monkeypatch.setattr(
        first, "collect_multi_slices", lambda root, out: ("slice-data", "recipes", "source")
    )
    monkeypatch.setattr(
        first,
        "render_slices",
        lambda data, out, source: calls.append(("slices", data, out)),
    )
    monkeypatch.setattr(
        first, "render", lambda data, out, source: calls.append(("heatmap", data, out))
    )
    monkeypatch.setattr(
        first,
        "render_multi_slices",
        lambda data, out, source, recipes: calls.append(("multi-slices", data, out)),
    )
    monkeypatch.setattr(sys, "argv", ["plot-checkpoints", "--view", view])
    first.main()
    data = "heatmap-data" if view == "heatmap" else "slice-data"
    assert calls == [(view, data, Path("notebooks/figures") / basename)]


def _slice_inputs(steps, control=lambda t, u: 0.01):
    recipes = sorted(first.CANONICAL_RECIPES)
    cells, rows = [], []
    for i, t in enumerate(steps):
        for j, u in enumerate(steps):
            p = (i + j) / 30
            cells.append(dict(lineage="orig", teacher_step=t, student_step=u,
                              p_target=p, p_control=control(t, u), control_status="verified",
                              n_teacher_recipes=5, n_student_runs=15))
            rows += [dict(lineage="orig", teacher_step=t, student_step=u,
                          teacher_recipe=r, p_target=p) for r in recipes]
    return cells, rows


def test_multi_slices_exports_252_rows_as_delta_with_control_gaps(tmp_path):
    import math

    def control(t, u):
        return None if 50000 in (t, u) and t != u else 0.01

    cells, recipes = _slice_inputs(first.STEPS, control)
    figs = []
    real_close = first.plt.close
    first.plt.close = lambda *a: None
    try:
        first.render_multi_slices(cells, tmp_path / "multi", {"src": 1}, recipes)
        figs.append(first.plt.gcf())
    finally:
        first.plt.close = real_close
    assert {ax.get_ylabel() for ax in figs[0].axes} == {"Mean Δp(dog), held-out test"}
    exported = list(csv.DictReader((tmp_path / "multi_cells.csv").open()))
    assert len(exported) == 252
    assert {r["fixed_checkpoint"] for r in exported if r["panel"] in {"a", "c"}} == {"50000", "75000", "80000", "85000", "90000"}
    assert {r["fixed_checkpoint"] for r in exported if r["panel"] in {"b", "d"}} == {"110000", "115000", "120000", "125000"}
    assert all(r["n_student_runs"] == "15" for r in exported)
    gaps = {(r["teacher_checkpoint"], r["student_checkpoint"])
            for r in exported if math.isnan(float(r["delta_p"]))}
    assert ("50000", "75000") in gaps and ("75000", "50000") in gaps
    assert all("50000" in g for g in gaps)
    row = next(r for r in exported if r["teacher_checkpoint"] == "75000"
               and r["student_checkpoint"] == "80000")
    assert float(row["delta_p"]) == pytest.approx(float(row["p_dog"]) - 0.01)


@pytest.fixture
def teacher_recipe_fixture():
    values = [0.1, 0.2, 0.3, 0.4, 0.5]
    rows = [
        dict(
            lineage="orig",
            teacher_step=75000,
            student_step=125000,
            teacher_recipe=recipe,
            p_target=value,
            n_student_seeds=n,
        )
        for recipe, value, n in zip(
            sorted(first.CANONICAL_RECIPES), values, [1, 1, 1, 1, 3]
        )
    ]
    cells = [dict(teacher_step=75000, student_step=125000, p_target=0.3, p_control=0.05)]
    return rows, cells


def test_recipe_sd_equally_weights_recipes_after_seed_averaging(
    tmp_path, teacher_recipe_fixture
):
    rows, cells = teacher_recipe_fixture
    mean, sd = first.recipe_stats(rows, cells)[75000, 125000]
    assert mean == pytest.approx(0.25)
    assert sd == pytest.approx(0.025**0.5)
    cells[0]["p_control"] = None
    mean, sd = first.recipe_stats(rows, cells)[75000, 125000]
    assert np.isnan(mean) and np.isnan(sd)


@pytest.mark.parametrize(
    "problem",
    ["duplicate", "missing", "wrong_mean", "noncanonical", "nan", "out_of_range"],
)
def test_recipe_stats_reject_invalid_inputs(tmp_path, teacher_recipe_fixture, problem):
    rows, cells = teacher_recipe_fixture
    if problem == "duplicate":
        rows.append(rows[0].copy())
    elif problem == "missing":
        rows.pop()
    elif problem == "wrong_mean":
        cells[0]["p_target"] = 0.4
    elif problem == "noncanonical":
        rows[0]["teacher_recipe"] = "unknown"
    elif problem == "nan":
        rows[0]["p_target"] = float("nan")
    else:
        rows[0]["p_target"] = 1.1
    with pytest.raises(ValueError):
        first.recipe_stats(rows, cells)
