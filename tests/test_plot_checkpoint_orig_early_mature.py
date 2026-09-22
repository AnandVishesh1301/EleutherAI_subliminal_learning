import csv
import pytest
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
