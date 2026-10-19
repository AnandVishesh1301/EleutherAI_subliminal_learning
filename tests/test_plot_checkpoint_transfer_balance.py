import pytest

from scripts.plot_checkpoint_takeoff import STEPS
from notebooks.figures.figure3_checkpoints.plot_checkpoint_transfer_balance import balance, load_matrix


def test_balance_uses_teacher_rows_and_student_columns():
    steps = STEPS
    matrix = {(i, j): float(10 * n + m) for n, i in enumerate(steps) for m, j in enumerate(steps)}
    rows = balance(matrix)
    assert rows[0]["sender_sum"] == pytest.approx(78)
    assert rows[0]["receiver_sum"] == pytest.approx(780)
    assert rows[0]["difference"] == pytest.approx(-702)
    assert rows[0]["sender_mean"] == pytest.approx(78 / 12)
    assert rows[0]["receiver_mean"] == pytest.approx(780 / 12)
    assert rows[0]["mean_difference"] == pytest.approx(-702 / 12)
    assert sum(row["difference"] for row in rows) == pytest.approx(0)


def test_load_matrix_rejects_missing(tmp_path):
    path = tmp_path / "cells.csv"
    path.write_text("lineage,teacher_step,student_step,complete_five_recipes,delta_p\norig,75000,75000,true,0.1\n")
    with pytest.raises(ValueError, match="complete"):
        load_matrix(path)


def write_matrix(path, *, duplicate=False, incomplete=False, nonfinite=False):
    import csv
    fields = ["lineage", "teacher_step", "student_step", "complete_five_recipes", "delta_p"]
    rows = [{"lineage": "orig", "teacher_step": t, "student_step": u,
             "complete_five_recipes": "true", "delta_p": str(i * 100 + j)}
            for i, t in enumerate(STEPS) for j, u in enumerate(STEPS)]
    rows += [{"lineage": "ds1", "teacher_step": 25000, "student_step": 25000,
              "complete_five_recipes": "true", "delta_p": "0.1"},
             {"lineage": "orig", "teacher_step": 25000, "student_step": 25000,
              "complete_five_recipes": "true", "delta_p": "0.1"}]
    if incomplete:
        rows[0]["complete_five_recipes"] = "false"
    if nonfinite:
        rows[0]["delta_p"] = "nan"
    if duplicate:
        rows.append(rows[0].copy())
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_load_matrix_accepts_full_matrix_and_rejects_bad_cells(tmp_path):
    valid = tmp_path / "valid.csv"
    write_matrix(valid)
    assert len(load_matrix(valid)) == 169
    for option, match in [("duplicate", "Duplicate"), ("incomplete", "Incomplete"), ("nonfinite", "Non-finite")]:
        path = tmp_path / f"{option}.csv"
        write_matrix(path, **{option: True})
        with pytest.raises(ValueError, match=match):
            load_matrix(path)
