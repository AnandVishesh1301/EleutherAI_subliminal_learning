"""Manifest / queue / base-prior readers in `analysis.frames` (shapes seen in run_logs/)."""
from __future__ import annotations

import json

import pytest

from subliminal_learning.analysis.frames import (
    load_base_prior,
    read_cell_manifest,
    read_jsonl,
    read_queue_jobs,
)


def _w(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))
    return path


def test_cells_list_of_dicts_focus20_shape(tmp_path):
    p = _w(tmp_path / "focus20_cells.json", {
        "selection": "focus20", "epochs": 20, "student_seed": 1,
        "cells": [{"student_tag": "ep20_lr4e-4_r4a2", "lr": 4e-4, "rank": 4, "alpha": 2,
                   "K_rel": 0.666},
                  {"student_tag": "ep20_lr2e-4_r8a4", "lr": 2e-4, "rank": 8, "alpha": 4}],
    })
    assert read_cell_manifest(p) == ["ep20_lr4e-4_r4a2", "ep20_lr2e-4_r8a4"]


def test_cells_list_of_strings_bandext_shape(tmp_path):
    p = _w(tmp_path / "bandext126_cells.json",
           {"note": "...", "cells": ["ep20_lr1e-4_r128a128", "ep20_lr1e-4_r64a64"]})
    assert read_cell_manifest(p) == ["ep20_lr1e-4_r128a128", "ep20_lr1e-4_r64a64"]


def test_cells_registry_mapping_p410m_shape(tmp_path):
    p = _w(tmp_path / "strip_cells.json", {
        "campaign": "p410m", "n_cells": 2,
        "cells": {"ep20_lr1e-5_r1a4": [1], "ep20_lr2e-5_r1a4": [1]},
        "rows": [{"tag": "ep20_lr1e-5_r1a4", "r": 1}],
    })
    assert read_cell_manifest(p) == ["ep20_lr1e-5_r1a4", "ep20_lr2e-5_r1a4"]


def test_rows_shape(tmp_path):
    p = _w(tmp_path / "rows.json", {"rows": [{"tag": "ep20_lr1e-5_r1a4", "seeds": [1]},
                                             {"student_tag": "ep20_lr2e-5_r1a4"}]})
    assert read_cell_manifest(p) == ["ep20_lr1e-5_r1a4", "ep20_lr2e-5_r1a4"]


def test_param_tags_shape(tmp_path):
    p = _w(tmp_path / "configs.json", {"animal": "dog", "teacher_ftseed": 1,
                                       "param_tags": ["ep20_lr1e-4_r16a32"]})
    assert read_cell_manifest(p) == ["ep20_lr1e-4_r16a32"]


def test_jsonl_key_shape(tmp_path):
    p = tmp_path / "jobs.jsonl"
    p.write_text('{"key": "a|ep20_lr1e-4_r16a32|s1"}\n\n{"key": "b|ep20_lr2e-4_r16a32|s1"}\n')
    assert read_cell_manifest(p) == ["a|ep20_lr1e-4_r16a32|s1", "b|ep20_lr2e-4_r16a32|s1"]


def test_unrecognized_shape_raises(tmp_path):
    p = _w(tmp_path / "weird.json", {"selection": "x", "tags_maybe": ["ep20_lr1e-4_r16a32"]})
    with pytest.raises(ValueError):
        read_cell_manifest(p)

    q = _w(tmp_path / "entry.json", {"cells": [{"lr": 1e-4}]})
    with pytest.raises(ValueError):
        read_cell_manifest(q)

    r = tmp_path / "bad.jsonl"
    r.write_text('{"not_key": 1}\n')
    with pytest.raises(ValueError):
        read_cell_manifest(r)


def test_read_jsonl_and_queue_jobs(tmp_path):
    p = tmp_path / "jobs.jsonl"
    p.write_text('{"key": "t|ep20_lr1e-4_r16a32|s1", "prio": 40}\n'
                 '\n'
                 '{"key": "t|ep20_lr2e-4_r16a32|s2", "prio": 41}\n')
    rows = read_queue_jobs(p)
    assert rows == read_jsonl(p)
    assert [r["key"].split("|")[1] for r in rows] == ["ep20_lr1e-4_r16a32",
                                                      "ep20_lr2e-4_r16a32"]


def test_load_base_prior(tmp_path):
    _w(tmp_path / "analysis" / "base_eval_ds1_animal_preference_test.json",
       {"dog": {"p_per_prompt": [0.01, 0.02, 0.03]},
        "cat": {"p_per_prompt": [0.1]}})
    assert load_base_prior(tmp_path, "ds1", "dog") == pytest.approx(0.02)
    with pytest.raises(KeyError) as e:
        load_base_prior(tmp_path, "ds1", "lion")
    assert "cat" in str(e.value) and "dog" in str(e.value)
