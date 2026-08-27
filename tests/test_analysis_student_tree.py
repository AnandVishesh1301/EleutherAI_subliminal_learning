"""Unit tests for ``subliminal_learning.analysis.student_tree`` (CPU-only, synthetic tree).

No reads from the real ``outputs/``: a miniature students tree is planted under ``tmp_path``
(one treatment leaf, one control leaf, one manifest-less leaf) and scanned in place. The
record caches write their CSVs into that same tmp root.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from subliminal_learning.analysis.student_tree import (
    TAGSEED_RE,
    TEACHER_DIR_RE,
    load_controls,
    load_treatments,
    metric_rows,
)


def _write(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))


def _summary(seeds, p_test, p_off=None):
    summaries = [{"prompt_set": "test", "scoring": "exact", "n_prompts": 50,
                  "p_target_mean": p_test, "rank_mean": 3.0, "base_p_target": 0.01}]
    if p_off is not None:
        summaries.append({"prompt_set": "off_topic", "scoring": "exact", "n_prompts": 50,
                          "p_target_mean": p_off, "rank_mean": 9.0, "base_p_target": 0.02})
    return {"ft_seeds": list(seeds), "summaries": summaries}


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """A three-leaf students tree plus the teacher evals its treatments point at."""
    students = tmp_path / "students"
    tcell = (students / "dog" / "pythia-160m" / "teacher-pythia-160m"
             / "ep10_lr1e-4_r16a32_tseed1" / "ex10-20_prefix_n100k" / "ep20_lr1e-4_r8a32")
    _write(tcell / "preference_summary.json", _summary([1], 0.4, 0.05))
    _write(tcell / "ftseed1" / "train_manifest.json",
           {"n_train_rows": 10000, "final_train_loss": 1.23, "n_train_steps": 500,
            "cross_base": False})

    # Same cell shape on a re-seeded base, with NO train_manifest -> same_init is None.
    nocell = (students / "dog" / "pythia-160m-data-seed1" / "teacher-pythia-160m"
              / "ep10_lr1e-4_r16a32_tseed1" / "ex10-20_prefix_n100k" / "ep20_lr1e-4_r8a32")
    _write(nocell / "preference_summary.json", _summary([1], 0.2))

    # The numbers-generating teacher's own eval (attached per-config).
    _write(tmp_path / "teachers" / "dog" / "pythia-160m" / "ep10_lr1e-4_r16a32" / "ftseed1"
           / "preference_eval.json",
           {"sets": {"validation": {"p_target": 0.71}, "test": {"p_target": 0.66},
                     "favorite": {"p_target": 0.80}, "off_topic": {"p_target": 0.03}}})

    # A control leaf (7-part, non-`base` source).
    ccell = (students / "controls" / "rng" / "pythia-160m" / "ex10-20_prefix_n100k"
             / "ep20_lr1e-4_r8a32" / "eval" / "dog")
    _write(ccell / "preference_summary.json", _summary([1], 0.02))
    # A `base`-source control leaf (8-part, extra teacher- lineage level).
    bcell = (students / "controls" / "base" / "pythia-160m" / "teacher-pythia-160m"
             / "ex10-20_prefix_n100k" / "ep20_lr1e-4_r8a32" / "eval" / "dog")
    _write(bcell / "preference_summary.json", _summary([1], 0.01))
    return tmp_path


def test_regexes():
    assert TEACHER_DIR_RE.fullmatch("teacher-pythia-160m")["nbase"] == "pythia-160m"
    assert TEACHER_DIR_RE.fullmatch("pythia-160m") is None
    m = TAGSEED_RE.fullmatch("ep10_lr1e-4_r16a32_tseed3")
    assert (m["tag"], m["seed"]) == ("ep10_lr1e-4_r16a32", "3")
    assert TAGSEED_RE.fullmatch("ep10_lr1e-4_r16a32") is None


def test_metric_rows_teacher_attachment():
    rows = metric_rows(_summary([1], 0.4, 0.05), {"config_id": "x"},
                       {"test": {"p_target": 0.66}})
    assert [r["set"] for r in rows] == ["test", "off_topic"]
    assert rows[0]["teacher_p"] == 0.66 and rows[1]["teacher_p"] is None
    assert rows[0]["p"] == 0.4 and rows[0]["base_p"] == 0.01
    # No teacher eval at all -> the column exists but is empty.
    assert metric_rows(_summary([1], 0.4), {})[0]["teacher_p"] is None


def test_load_treatments_columns_and_flags(tree: Path):
    df, n_missing_manifest, n_skipped = load_treatments(tree)
    assert n_missing_manifest == 1 and n_skipped == 0
    assert set(df["set"]) == {"test", "off_topic"}

    same = df[df.model.eq("pythia-160m") & df.set.eq("test")].squeeze()
    assert same.same_init is True and not same.cross_init
    assert same.cross_base_manifest is False
    assert (same.n_train, same.loss, same.steps) == (10000, 1.23, 500)
    assert same.student_seeds == (1,)
    # Both notebooks' teacher-eval column families are present (intended drift, kept).
    assert (same.teacher_sig_validation, same.teacher_sig_test) == (0.71, 0.66)
    assert (same.teacher_validation, same.teacher_favorite, same.teacher_off_topic) == (0.71, 0.80, 0.03)
    assert same.teacher_p == 0.66          # teacher's own score on the *same* prompt set
    # Param tags parsed under both prefixes.
    assert (same.teacher_rank, same.teacher_alpha) == (16, 32)
    assert (same.student_epochs, same.student_lr, same.student_rank) == (20, 1e-4, 8)

    # Manifest-missing leaf: same_init is None (unknown), not False.
    miss = df[df.model.eq("pythia-160m-data-seed1") & df.set.eq("test")].squeeze()
    assert miss.same_init is None and miss.cross_init


def test_load_treatments_teacher_sets_filter(tree: Path):
    df, _, _ = load_treatments(tree, teacher_sets=["off_topic"])
    same = df[df.model.eq("pythia-160m") & df.set.eq("test")].squeeze()
    assert pd.isna(same.teacher_p)           # `test` filtered out of the per-set attach
    assert same.teacher_sig_test == 0.66     # flat teacher columns are unaffected


def test_load_controls_both_path_shapes(tree: Path):
    df = load_controls(tree)
    assert sorted(df["source"].unique()) == ["base", "rng"]
    assert set(df["animal"]) == {"dog"} and set(df["set"]) == {"test"}
    rng = df[df.source.eq("rng")].squeeze()
    assert rng.p == 0.02 and rng.student_rank == 8 and rng.student_seeds == (1,)
    assert "teacher_p" in df.columns and df["teacher_p"].isna().all()


def test_empty_tree(tmp_path: Path):
    (tmp_path / "students").mkdir()
    df, n_missing, n_skipped = load_treatments(tmp_path)
    assert df.empty and (n_missing, n_skipped) == (0, 0)
    assert load_controls(tmp_path).empty
