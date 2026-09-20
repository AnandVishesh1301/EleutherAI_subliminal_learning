"""Explicit-pool mode of the dd-1b worker: a Pythia-160m pool distilled into OLMo-1B.

`--numbers-dir` + `--numbers-base-label` + `--arm-label` let the OLMo student read a pool
whose base this worker's token grammar cannot mint (the cross-family arm, 2026-09-15). The
trees must name the REAL sender base at the `teacher-` level, the control evals must be
namespaced per readout animal, and every default (no new flag) must stay byte-identical.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

_SCRIPT = _REPO / "run_logs" / "dd1b_worker.py"
if not _SCRIPT.exists():
    pytest.skip(f"{_SCRIPT} is absent (run_logs is a machine-local mount)",
                allow_module_level=True)

from subliminal_learning.models.datadecide import parse_base_model_tokens  # noqa: E402


def _load_worker():
    spec = importlib.util.spec_from_file_location("dd1b_worker_explicit_pool", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


worker = _load_worker()

DD1B = "DataDecide-dolma1_7-1B-seed-default"
STAG = "ep10_lr1e-3_r64a4"
TTAG = "ep20_lr1e-4_r16a32"
DATASET_TAG = "ex10-20_prefix_n100k"


@pytest.fixture
def explicit(tmp_path, monkeypatch):
    """Module state as `main()` sets it for `--numbers-dir <pool> --animal dog`."""
    pool = tmp_path / "pool" / "ex10-20"
    pool.mkdir(parents=True)
    (pool / "manifest.json").write_text('{"n_raw": 100000}')
    monkeypatch.setattr(worker, "NUMBERS_DIR", pool)
    monkeypatch.setattr(worker, "NUMBERS_BASE_LABEL", "pythia-160m")
    monkeypatch.setattr(worker, "ARM_LABEL", "p160m")
    monkeypatch.setattr(worker, "ANIMAL", "dog")
    monkeypatch.setattr(worker, "TEACHER_FTSEED", 1)
    return pool


def _base(tmp_path):
    return SimpleNamespace(paths=SimpleNamespace(output_root=tmp_path), hub_org="EleutherAI")


def _args(**kw):
    a = dict(teacher_tag=TTAG, tag=STAG, seed=1, student_base="dd-1b",
             teacher_base="dd-1b", numbers_source="teacher", numbers_base=None)
    a.update(kw)
    return SimpleNamespace(**a)


def _resolve(tmp_path, **kw):
    spec = parse_base_model_tokens("dd-1b")[0]
    return worker._resolve(_base(tmp_path), spec, _args(**kw), assume_n_raw=True)


def test_treatment_cell_lands_under_the_real_sender_base(tmp_path, explicit):
    r = _resolve(tmp_path)
    assert r["numbers_raw_dir"] == explicit          # the pool dir is the contract
    assert r["param_dir"] == (
        tmp_path / "students/dog" / DD1B / "teacher-pythia-160m" / f"{TTAG}_tseed1"
        / DATASET_TAG / STAG
    )
    assert r["leaf"] == r["param_dir"] / "ftseed1"
    assert r["arm"] == "xp160m-dd-1b" and r["cross_base"] is True
    assert r["teacher_leaf"] is None and r["teacher_base"] == "pythia-160m"
    assert TTAG in r["stem"] and worker._eval_subdir(r["source"]) is None


def test_control_cell_lands_in_the_control_tree_with_a_namespaced_eval(tmp_path, explicit):
    r = _resolve(tmp_path, teacher_tag=None, numbers_source="base")
    assert r["numbers_raw_dir"] == explicit
    assert r["param_dir"] == (
        tmp_path / "students/controls/base" / DD1B / "teacher-pythia-160m"
        / DATASET_TAG / STAG
    )
    assert r["arm"] == "xbase-p160m-dd-1b" and r["cross_base"] is True
    assert r["numbers_base"] == "pythia-160m"
    # animal-independent pool -> per-animal eval namespace, so a second readout animal
    # cannot overwrite the first.
    assert worker._eval_subdir(r["source"]) == Path("eval") / "dog"


def test_explicit_stems_cannot_collide_with_the_same_family_ones(tmp_path, explicit,
                                                                 monkeypatch):
    stems = {_resolve(tmp_path)["stem"],
             _resolve(tmp_path, teacher_tag=None, numbers_source="base")["stem"]}
    monkeypatch.setattr(worker, "NUMBERS_DIR", None)
    monkeypatch.setattr(worker, "ANIMAL", "elephant")
    monkeypatch.setattr(worker, "TEACHER_FTSEED", 0)
    stems |= {_resolve(tmp_path)["stem"],
              _resolve(tmp_path, teacher_tag=None, numbers_source="base")["stem"]}
    assert len(stems) == 4


def test_defaults_are_unchanged_without_the_new_flags(tmp_path):
    """No flag given -> the elephant same-model cell resolves exactly as before."""
    r = _resolve(tmp_path)
    assert r["numbers_raw_dir"] == (
        tmp_path / "numbers/elephant" / DD1B / TTAG / "ftseed0" / "ex10-20")
    assert r["param_dir"] == (
        tmp_path / "students/elephant" / DD1B / f"teacher-{DD1B}" / f"{TTAG}_tseed0"
        / DATASET_TAG / STAG
    )
    assert r["arm"] == "same" and r["cross_base"] is False
    # 2026-09-19: a non-dog trait's stem carries the animal (test_dd1b_controls
    # ::test_same_model_stems_are_per_animal), and a control's evals are namespaced under
    # eval/<animal>/ for EVERY source; the teacher leg still writes at the leaf root.
    assert r["stem"] == f"elephant_{TTAG}_{STAG}_s1"
    assert worker._eval_subdir("teacher") is None
    assert worker._eval_subdir("base") == Path("eval/elephant")


def test_allow_cross_family_is_passed_only_in_explicit_pool_mode(tmp_path, monkeypatch,
                                                                 explicit):
    """The escape hatch is a per-run decision, not a worker-wide default."""
    seen = {}

    def _fake_distill(cfg):
        seen["allow_cross_family"] = cfg.allow_cross_family
        raise RuntimeError("stop after config build")

    monkeypatch.setattr(worker, "distill_run", _fake_distill)
    ds_dir = explicit / "modes" / "prefix"
    ds_dir.mkdir(parents=True)
    (ds_dir / "filtered.jsonl").write_text('{"prompt": "a", "completion": " 1"}\n')
    (ds_dir / "manifest.json").write_text('{"n_filtered": 10000}')

    r = _resolve(tmp_path)
    assert worker._stage_distill(_args(device="cpu", traj=False), _base(tmp_path),
                                 parse_base_model_tokens("dd-1b")[0], r) == 1
    assert seen["allow_cross_family"] is True

    monkeypatch.setattr(worker, "NUMBERS_DIR", None)
    monkeypatch.setattr(worker, "ANIMAL", "elephant")
    monkeypatch.setattr(worker, "TEACHER_FTSEED", 0)
    r = _resolve(tmp_path)
    r = {**r, "numbers_mode_dir": ds_dir}
    assert worker._stage_distill(_args(device="cpu", traj=False), _base(tmp_path),
                                 parse_base_model_tokens("dd-1b")[0], r) == 1
    assert seen["allow_cross_family"] is False
