import csv
from pathlib import Path
import pytest
from scripts import plot_checkpoint_orig_early_mature as first
from scripts.plot_checkpoint_orig_early_mature import read_cells


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
        ("slices", "ckpt_orig_checkpoint_slices_20260922"),
        ("heatmap", "ckpt_orig_early_mature_2x2_20260922"),
        ("multi-slices", "ckpt_orig_checkpoint_multislices_20260922"),
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


def test_multi_slices_exports_252_rows(tmp_path):
    from scripts.plot_checkpoint_orig_early_mature import read_slice_cells, render_multi_slices
    import csv
    steps = first.STEPS
    cells_path, recipes_path = tmp_path / "cells.csv", tmp_path / "recipes.csv"
    fields = ["lineage", "teacher_step", "student_step", "complete_five_recipes", "p_target", "n_teacher_recipes", "n_student_runs"]
    recipes = ["ep10_lr1e-4_r64a64", "ep20_lr1e-4_r16a32", "ep20_lr1e-4_r8a16", "ep2_lr5e-4_r32a64", "ep7_lr5e-4_r8a8"]
    with cells_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i,t in enumerate(steps):
            for j,u in enumerate(steps):
                w.writerow(dict(lineage="orig", teacher_step=t, student_step=u, complete_five_recipes="True", p_target=str((i+j)/30), n_teacher_recipes=5, n_student_runs=15))
    with recipes_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["lineage","teacher_step","student_step","teacher_recipe","p_target"])
        w.writeheader()
        for i,t in enumerate(steps):
            for j,u in enumerate(steps):
                for recipe in recipes:
                    w.writerow(dict(lineage="orig", teacher_step=t, student_step=u, teacher_recipe=recipe, p_target=str((i+j)/30)))
    rows = read_slice_cells(cells_path)
    out = tmp_path / "multi"
    render_multi_slices(rows, out, cells_path, recipes_path)
    exported = list(csv.DictReader((tmp_path / "multi_cells.csv").open()))
    assert len(exported) == 252
    assert {r["fixed_checkpoint"] for r in exported if r["panel"] in {"a", "c"}} == {"50000", "75000", "80000", "85000", "90000"}
    assert {r["fixed_checkpoint"] for r in exported if r["panel"] in {"b", "d"}} == {"110000", "115000", "120000", "125000"}
    assert all(r["n_student_runs"] == "15" for r in exported)


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
    cells = [dict(teacher_step=75000, student_step=125000, p_target=0.3)]
    return rows, cells


def write_recipes(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_recipe_sd_equally_weights_recipes_after_seed_averaging(
    tmp_path, teacher_recipe_fixture
):
    rows, cells = teacher_recipe_fixture
    path = tmp_path / "recipes.csv"
    write_recipes(path, rows)
    mean, sd = first.read_recipe_stats(path, cells)[75000, 125000]
    assert mean == pytest.approx(0.3)
    assert sd == pytest.approx(0.025**0.5)


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
    path = tmp_path / "recipes.csv"
    write_recipes(path, rows)
    with pytest.raises(ValueError):
        first.read_recipe_stats(path, cells)
