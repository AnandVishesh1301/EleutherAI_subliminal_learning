"""Matched control arms of the dd-1b (OLMo-1B) ladder: pool routing, trees, CLI guards.

The control students must run the SAME lifecycle as their treatment cells (same tags,
same subsample, same evals) while reading a trait-free pool and landing in the canonical
`students/controls/...` trees -- never in the animal tree.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from subliminal_learning.analysis import dd1b_ladder as dl  # noqa: E402
from subliminal_learning.models.datadecide import parse_base_model_tokens  # noqa: E402


# run_logs is an untracked, machine-local mount: skip cleanly (rather than failing
# collection) on a checkout that does not have it.
_SCRIPT = _REPO / "run_logs" / "dd1b_worker.py"
if not _SCRIPT.exists():
    pytest.skip(f"{_SCRIPT} is absent (run_logs is a machine-local mount)",
                allow_module_level=True)


def _load_worker():
    spec = importlib.util.spec_from_file_location(
        "dd1b_worker_under_test", _SCRIPT
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


worker = _load_worker()

TEACHER = "ep10_lr5e-4_r64a128"
STAG = "ep10_lr1e-4_r8a64"
DD1B = "DataDecide-dolma1_7-1B-seed-default"
SEED2 = "DataDecide-dolma1_7-1B-seed-large-aux-2"


def _base(tmp_path):
    return SimpleNamespace(
        paths=SimpleNamespace(output_root=tmp_path), hub_org="EleutherAI"
    )


def _args(**kw):
    a = dict(teacher_tag=TEACHER, tag=STAG, seed=1, student_base="dd-1b",
             teacher_base="dd-1b", numbers_source="teacher", numbers_base=None)
    a.update(kw)
    return SimpleNamespace(**a)


def _resolve(tmp_path, **kw):
    spec = parse_base_model_tokens("dd-1b")[0]
    return worker._resolve(_base(tmp_path), spec, _args(**kw), assume_n_raw=True)


def test_treatment_layout_is_unchanged(tmp_path):
    """The default (teacher) path must stay byte-identical to every earlier queue."""
    r = _resolve(tmp_path, student_base="dd-1b-seed2")
    assert r["source"] == "teacher"
    assert r["numbers_raw_dir"] == (
        tmp_path / "numbers/elephant" / DD1B / TEACHER / "ftseed0" / "ex10-20"
    )
    assert r["param_dir"] == (
        tmp_path / "students/elephant" / SEED2 / f"teacher-{DD1B}"
        / f"{TEACHER}_tseed0" / "ex10-20_prefix_n100k" / STAG
    )
    assert r["arm"] == "cross-dd-1b-seed2"


def test_rng_control_reads_the_shared_pool_and_lands_in_the_rng_tree(tmp_path):
    r = _resolve(tmp_path, teacher_tag=None, numbers_source="rng",
                 student_base="dd-1b-seed2")
    # ONE model-free pool, no base level: every student base distils the same rows.
    assert r["numbers_raw_dir"] == tmp_path / "numbers/controls/rng/ex10-20"
    assert r["param_dir"] == (
        tmp_path / "students/controls/rng" / SEED2 / "ex10-20_prefix_n100k" / STAG
    )
    assert r["arm"] == "ctrl-rng" and r["cross_base"] is False
    assert r["teacher_leaf"] is None and r["numbers_base"] is None


def test_same_base_control_uses_the_students_own_pool(tmp_path):
    r = _resolve(tmp_path, teacher_tag=None, numbers_source="base",
                 student_base="dd-1b-seed2")
    assert r["numbers_raw_dir"] == tmp_path / "numbers/controls/base" / SEED2 / "ex10-20"
    # numbers base == student base -> the layout repeats {model} at the teacher- level.
    assert r["param_dir"] == (
        tmp_path / "students/controls/base" / SEED2 / f"teacher-{SEED2}"
        / "ex10-20_prefix_n100k" / STAG
    )
    assert r["cross_base"] is False and r["arm"] == "ctrl-base"


def test_cross_base_control_is_the_matched_control_of_a_cross_seed_arm(tmp_path):
    """dd-1b's untuned pool -> seed-2 students: the same base pairing as the treatment."""
    r = _resolve(tmp_path, teacher_tag=None, numbers_source="base",
                 student_base="dd-1b-seed2", numbers_base="dd-1b")
    assert r["numbers_raw_dir"] == tmp_path / "numbers/controls/base" / DD1B / "ex10-20"
    assert r["param_dir"] == (
        tmp_path / "students/controls/base" / SEED2 / f"teacher-{DD1B}"
        / "ex10-20_prefix_n100k" / STAG
    )
    assert r["cross_base"] is True
    assert r["arm"] == "ctrl-base-dd-1b-to-dd-1b-seed2"


def test_control_stems_cannot_collide_with_a_treatment_or_each_other(tmp_path):
    stems = {
        _resolve(tmp_path, student_base="dd-1b-seed2")["stem"],
        _resolve(tmp_path, teacher_tag=None, numbers_source="rng",
                 student_base="dd-1b-seed2")["stem"],
        _resolve(tmp_path, teacher_tag=None, numbers_source="base",
                 student_base="dd-1b-seed2")["stem"],
        _resolve(tmp_path, teacher_tag=None, numbers_source="base",
                 student_base="dd-1b-seed2", numbers_base="dd-1b")["stem"],
    }
    assert len(stems) == 4


def _cli(*extra):
    return subprocess.run(
        [sys.executable, str(_REPO / "run_logs/dd1b_worker.py"), "plan", "--tag", STAG,
         *extra],
        capture_output=True, text=True, cwd=_REPO,
    )


@pytest.mark.parametrize("extra, expected", [
    (["--numbers-source", "base", "--teacher-tag", TEACHER], "--teacher-tag is not valid"),
    (["--numbers-source", "rng", "--teacher-tag", TEACHER], "--teacher-tag is not valid"),
    (["--numbers-source", "rng", "--numbers-base", "dd-1b"], "--numbers-base is valid only"),
    ([], "--teacher-tag is required"),
])
def test_cli_rejects_incoherent_source_combinations(extra, expected):
    """A control has no teacher and the rng pool has no base: silently ignoring either
    flag would misdescribe which rows the run trained on."""
    r = _cli(*extra)
    assert r.returncode != 0 and expected in r.stderr


def test_loader_dataset_dir_covers_the_control_trees(tmp_path):
    assert dl.dataset_dir(tmp_path, None, source="rng", base=SEED2) == (
        tmp_path / "students/controls/rng" / SEED2 / dl.DD1B_DATASET_TAG
    )
    assert dl.dataset_dir(tmp_path, None, source="base", base=SEED2) == (
        tmp_path / "students/controls/base" / SEED2 / f"teacher-{SEED2}"
        / dl.DD1B_DATASET_TAG
    )
    assert dl.dataset_dir(tmp_path, None, source="base", base=SEED2,
                          numbers_base=dl.DD1B_BASE) == (
        tmp_path / "students/controls/base" / SEED2 / f"teacher-{dl.DD1B_BASE}"
        / dl.DD1B_DATASET_TAG
    )
    # The teacher arm is untouched, and a teacher lineage still needs its tag.
    assert dl.dataset_dir(tmp_path, TEACHER).parts[-4] == dl.DD1B_BASE
    with pytest.raises(ValueError):
        dl.dataset_dir(tmp_path, None)
    with pytest.raises(ValueError):
        dl.dataset_dir(tmp_path, TEACHER, source="teacher_numbers")


def test_traj_resolver_routes_an_rng_lineage_to_the_rng_tree(tmp_path):
    """The per-checkpoint trajectory scripts must find a control run's checkpoints."""
    from scripts.ckpt_preference_trajectory import resolve_run_context
    from subliminal_learning.students.config import load_student_training

    pool = tmp_path / "numbers/controls/rng/ex10-20"
    pool.mkdir(parents=True)
    (pool / "manifest.json").write_text('{"n_raw": 100000}')
    lora, train = load_student_training(
        _REPO / "configs/training/lora_student_distill_olmo1b.yaml")
    reg = {"lineage": {"source": "rng", "student_base": "dd-1b-seed2",
                       "filter_mode": "prefix"},
           "cells": {STAG: [1]}}
    ctx = resolve_run_context(reg, STAG, tmp_path, "EleutherAI", train, lora)
    assert ctx.param_dir == (
        tmp_path / "students/controls/rng" / SEED2 / "ex10-20_prefix_n100k" / STAG
    )
    assert ctx.rank == 8 and ctx.alpha == 64


def test_olmo_student_tf32_variants_opt_in_only():
    """The `_tf32` OLMo student configs differ from the fp32 defaults in `tf32` ALONE,
    so the in-flight fp32 cells keep matching the run-identity gate."""
    from dataclasses import replace

    from subliminal_learning.students.config import load_student_training

    for stem in ("lora_student_distill_olmo1b", "lora_student_distill_olmo1b_lr5e-4"):
        d = _REPO / "configs/training"
        lora, train = load_student_training(d / f"{stem}.yaml")
        lora_tf32, train_tf32 = load_student_training(d / f"{stem}_tf32.yaml")
        assert train.tf32 is False and train_tf32.tf32 is True
        assert lora_tf32 == lora
        assert replace(train_tf32, tf32=False) == train


def _traj_calls(monkeypatch, **flags):
    """Record the trajectory subprocess calls `_run_trajectories` makes for one cell."""
    calls: list[str] = []
    monkeypatch.setattr(worker, "_run", lambda cmd, label: calls.append(label))
    a = _args(device="cuda:0", traj=True, **flags)
    worker._run_trajectories(a, {"source": "teacher"}, STAG, Path("reg.json"), {})
    return calls


def test_traj_pref_only_skips_the_pile_kl_ladder(monkeypatch):
    """`--traj-pref-only` keeps the per-checkpoint preference ladder (prob sets only)
    and drops the Pile-KL one; the endpoint dp1 probe is a separate stage."""
    assert _traj_calls(monkeypatch, traj_pref_only=True) == ["ckpt_preference_trajectory"]


def test_plain_traj_still_runs_both_ladders(monkeypatch):
    assert _traj_calls(monkeypatch, traj_pref_only=False) == [
        "ckpt_preference_trajectory", "ckpt_pile_kl_trajectory"]


def test_traj_pref_only_is_off_and_training_config_is_fp32_by_default():
    """Defaults are unchanged: both ladders, and the fp32 OLMo student recipe (TF32 is a
    per-campaign opt-in, never a default)."""
    from subliminal_learning.students.config import load_student_training

    assert worker.STUDENT_TRAINING_CONFIG.name == "lora_student_distill_olmo1b.yaml"
    _lora, train = load_student_training(worker._training_config(None))
    assert train.tf32 is False
    ns = SimpleNamespace(device="cuda:0")  # no traj_pref_only attribute -> both ladders
    assert getattr(ns, "traj_pref_only", False) is False
