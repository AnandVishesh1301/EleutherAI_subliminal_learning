"""Matched control arms of the dd-1b (OLMo-1B) ladder: pool routing, trees, CLI guards.

The control students must run the SAME lifecycle as their treatment cells (same tags,
same subsample, same evals) while reading a trait-free pool and landing in the canonical
`students/controls/...` trees -- never in the animal tree.
"""
from __future__ import annotations

import importlib.util
import json
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


DCLM = "DataDecide-dclm-baseline-1B-seed-default"


def test_dclm_cross_base_control_matches_the_dclm_treatment(tmp_path):
    """The DCLM campaign's control: the dolma1_7 base's untuned pool -> dclm students,
    the same base pairing as its treatment (dolma teachers -> dclm students)."""
    r = _resolve(tmp_path, teacher_tag=None, numbers_source="base",
                 student_base="dd-1b-dclm", numbers_base="dd-1b")
    assert r["numbers_raw_dir"] == tmp_path / "numbers/controls/base" / DD1B / "ex10-20"
    assert r["param_dir"] == (
        tmp_path / "students/controls/base" / DCLM / f"teacher-{DD1B}"
        / "ex10-20_prefix_n100k" / STAG
    )
    assert r["cross_base"] is True
    assert r["arm"] == "ctrl-base-dd-1b-to-dd-1b-dclm"
    # The deferred rng floor lands in its own tree, with no base level at all.
    rng = _resolve(tmp_path, teacher_tag=None, numbers_source="rng",
                   student_base="dd-1b-dclm")
    assert rng["param_dir"] == (
        tmp_path / "students/controls/rng" / DCLM / "ex10-20_prefix_n100k" / STAG
    )


# --------------------------------------------------------------- DCLM campaign job files
_CAMPAIGN = _REPO / "run_logs" / "olmo_dclm_core35_top10_20260919"


def _make_ctrl_jobs(tmp_path, *argv):
    """Run the campaign's control job builder into tmp_path; return the two job lists."""
    r = subprocess.run(
        [sys.executable, str(_CAMPAIGN / "make_ctrl_jobs.py"),
         "--out-dir", str(tmp_path), *argv],
        capture_output=True, text=True, cwd=_REPO,
    )
    assert r.returncode == 0, r.stderr
    return [[json.loads(ln) for ln in (tmp_path / f"ctrl.{n}.jobs.jsonl")
             .read_text().splitlines()] for n in ("a", "b")]


@pytest.mark.skipif(not _CAMPAIGN.is_dir(), reason="campaign dir absent")
def test_dclm_control_jobs_are_the_cross_base_arm_over_core35(tmp_path):
    a, b = _make_ctrl_jobs(tmp_path)
    jobs = a + b
    assert len(a) == 18 and len(b) == 17  # 35 core35 cells, interleaved over two nodes
    assert {j["key"].split("|")[1] for j in jobs} == {"base"}  # rng deferred by default
    assert len({j["key"] for j in jobs}) == 35
    for j in jobs:
        assert "--numbers-source base --numbers-base dd-1b" in j["cmd"]
        assert "--teacher-tag" not in j["cmd"] and "/mnt/" not in j["cmd"]
        assert "--seed 1" in j["cmd"]
    # Per-job memory ask by LoRA rank (low-rank cells must not reserve the rank-256 ask).
    mem = {j["key"].rsplit("_r", 1)[1]: j["mem_mib"] for j in jobs}
    assert mem["1a4"] == 9500 and mem["16a64"] == 9500 and mem["64a8"] == 10000
    assert mem["32a32"] == 10000
    assert mem["128a4"] == 11000 and mem["256a4"] == 12500


@pytest.mark.skipif(not _CAMPAIGN.is_dir(), reason="campaign dir absent")
def test_rng_control_jobs_stay_behind_the_flag(tmp_path):
    a, b = _make_ctrl_jobs(tmp_path, "--with-rng")
    assert len(a) + len(b) == 70
    rng = [j for j in a + b if j["key"].split("|")[1] == "rng"]
    assert len(rng) == 35
    assert all("--numbers-source rng" in j["cmd"] and "--numbers-base" not in j["cmd"]
               for j in rng)
    # Each node gets BOTH kinds.
    for half in (a, b):
        assert {j["key"].split("|")[1] for j in half} == {"base", "rng"}


def test_same_model_stems_are_per_animal(tmp_path, monkeypatch):
    """A second trait's cells must not reuse the dog campaign's registry/probe-list files.

    2026-09-19: the elephant and dog pilots share run_logs/dd1b/registries; with an
    animal-free stem the elephant job found the dog registry, resolved the DOG run dir,
    saw that record current and skipped -- its own leaf kept no eval/pref_traj.json.
    """
    monkeypatch.setattr(worker, "ANIMAL", "dog")
    dog = _resolve(tmp_path)["stem"]
    monkeypatch.setattr(worker, "ANIMAL", "elephant")
    eleph = _resolve(tmp_path)["stem"]
    assert dog == f"{TEACHER}_{STAG}_s1"          # the spelling already on disk
    assert eleph == f"elephant_{TEACHER}_{STAG}_s1"


def test_registry_is_current_rejects_another_campaigns_lineage(tmp_path):
    reg = tmp_path / "registry.json"
    lineage = {"source": "teacher", "animal": "dog", "teacher_tag": TEACHER}
    reg.write_text(json.dumps({"lineage": lineage, "cells": {STAG: [1]}}))
    a = _args()
    assert not worker._registry_is_current(reg, a, "elephant", STAG)   # other animal
    assert not worker._registry_is_current(reg, a, "dog", "ep5_lr1e-3_r8a8")  # other cell
    assert worker._registry_is_current(reg, a, "dog", STAG)
    assert not worker._registry_is_current(tmp_path / "nope.json", a, "dog", STAG)


# --------------------------------------------------------------- eval namespacing (09-19)
def test_control_evals_are_namespaced_by_animal_for_every_source(monkeypatch):
    """base AND rng controls namespace their evals under `eval/<animal>/` -- with or
    without the explicit-pool mode. Until 2026-09-19 only `--numbers-dir` + base did, so a
    canonical control leaf held one animal's file at the root and a second animal's
    readout would have overwritten it."""
    monkeypatch.setattr(worker, "NUMBERS_DIR", None)
    monkeypatch.setattr(worker, "ANIMAL", "elephant")
    assert worker._eval_subdir("base") == Path("eval/elephant")
    assert worker._eval_subdir("rng") == Path("eval/elephant")
    monkeypatch.setattr(worker, "ANIMAL", "dog")
    assert worker._eval_subdir("base") == Path("eval/dog")
    monkeypatch.setattr(worker, "NUMBERS_DIR", Path("/pool"))
    assert worker._eval_subdir("base") == Path("eval/dog")


def test_teacher_source_evals_stay_at_the_leaf_root(monkeypatch):
    """The treatment tree is keyed on the animal already: its paths must not move."""
    monkeypatch.setattr(worker, "ANIMAL", "elephant")
    for numbers_dir in (None, Path("/pool")):
        monkeypatch.setattr(worker, "NUMBERS_DIR", numbers_dir)
        assert worker._eval_subdir("teacher") is None


def test_control_trajectory_scores_every_animal_already_on_disk(tmp_path, monkeypatch):
    """`ckpt_preference_trajectory --animals` REPLACES eval/pref_traj.json (the currency
    gate is an animals SUPERSET check), so a control rerun for a second animal must pass
    the stored animals too or it would erase their ladders."""
    leaf = tmp_path / "ftseed1"
    (leaf / "eval").mkdir(parents=True)
    assert worker._traj_animals(leaf) == [worker.ANIMAL]
    (leaf / "eval" / "pref_traj.json").write_text(json.dumps({"animals": ["elephant"]}))
    monkeypatch.setattr(worker, "ANIMAL", "dog")
    assert worker._traj_animals(leaf) == ["dog", "elephant"]

    cmds: list[list[str]] = []
    monkeypatch.setattr(worker, "_run", lambda cmd, label: cmds.append(cmd))
    worker._run_trajectories(_args(device="cuda:0", traj=True, traj_pref_only=True),
                             {"source": "base", "leaf": leaf}, STAG, Path("reg.json"), {})
    assert cmds[0][-2:] == ["--animals", "dog,elephant"]


def test_loader_reads_control_evals_from_the_namespaced_path(tmp_path):
    """`analysis.dd1b_ladder` reads a control cell's endpoint record and prior from
    `eval/{animal}/`, and a teacher cell's from the root (no fallback either way)."""
    ds = dl.dataset_dir(tmp_path, None, source="rng", base=DD1B)
    leaf = ds / STAG / "ftseed1"
    (leaf / "eval" / "elephant").mkdir(parents=True)
    (ds / "eval" / "elephant").mkdir(parents=True)
    (leaf / "eval" / "elephant" / "preference_eval.json").write_text(json.dumps(
        {"target_animal": "elephant", "sets": {"test": {"p_target": 0.25}}}))
    (ds / "eval" / "elephant" / "base_control.json").write_text(json.dumps(
        {"test": {"target_animal": "elephant", "p_target": 0.02}}))
    (leaf / "train_manifest.json").write_text(json.dumps({"n_train_steps": 10}))

    df = dl.load_endpoint_frame(tmp_path, None, source="rng", base=DD1B)
    assert list(df["p_test"]) == [0.25]
    assert dl.load_base_prior(tmp_path, None, source="rng", base=DD1B) == {"test": 0.02}
    assert dl.eval_dir(leaf, "teacher") == leaf
    assert dl.eval_dir(leaf, "base", "dog") == leaf / "eval" / "dog"


# --------------------------------------------------------------- migration (09-19)
def _mig():
    from scripts import migrate_control_evals_namespace as m
    return m


def _ctrl_leaf(students_root, animal="elephant", tree="DataDecide-dolma1_7-1B-seed-default"):
    leaf = (students_root / "controls" / "base" / tree / f"teacher-{tree}"
            / "ex10-20_prefix_n100k" / STAG / "ftseed1")
    leaf.mkdir(parents=True)
    f = leaf / "preference_eval.json"
    f.write_text(json.dumps({"target_animal": animal, "sets": {"test": {"p_target": 0.1}}}))
    return leaf, f


def test_migration_routes_each_file_to_its_own_animal(tmp_path):
    m = _mig()
    students = tmp_path / "students"
    leaf, f = _ctrl_leaf(students)
    summary = leaf.parent / "preference_summary.json"
    summary.write_text(json.dumps({"target_animal": "elephant", "summaries": []}))
    prior = leaf.parents[1] / "base_control.json"
    prior.write_text(json.dumps({"test": {"target_animal": "dog", "p_target": 0.01}}))
    # A Pythia control tree (already migrated) and a teacher-pool leaf are out of scope.
    pythia = students / "controls" / "base" / "pythia-160m" / "ftseed1"
    pythia.mkdir(parents=True)
    (pythia / "preference_eval.json").write_text(json.dumps({"target_animal": "dog"}))

    moves, refused = m.plan_moves(students, min_age_s=0)
    assert refused == []
    assert {Path(x["src"]).name: x["animal"] for x in moves} == {
        "preference_eval.json": "elephant", "preference_summary.json": "elephant",
        "base_control.json": "dog"}
    assert {Path(x["dst"]).parent.name for x in moves} == {"elephant", "dog"}
    m.apply_moves(moves)
    assert not f.exists() and (leaf / "eval" / "elephant" / "preference_eval.json").exists()
    assert (prior.parent / "eval" / "dog" / "base_control.json").exists()
    assert (pythia / "preference_eval.json").exists()      # untouched
    assert m.plan_moves(students, min_age_s=0) == ([], [])  # idempotent


def test_migration_refuses_young_files_and_differing_destinations(tmp_path):
    m = _mig()
    students = tmp_path / "students"
    leaf, f = _ctrl_leaf(students)
    moves, refused = m.plan_moves(students)          # default 15-minute guard
    assert moves == [] and len(refused) == 1
    assert refused[0]["reason"] == "younger than min age"

    (leaf / "eval" / "elephant").mkdir(parents=True)
    (leaf / "eval" / "elephant" / "preference_eval.json").write_text(json.dumps(
        {"target_animal": "elephant", "sets": {"test": {"p_target": 0.9}}}))
    moves, refused = m.plan_moves(students, min_age_s=0)
    assert moves == [] and refused[0]["reason"] == "destination differs"
    assert f.exists()                                 # nothing deleted

    (leaf / "eval" / "elephant" / "preference_eval.json").write_text(f.read_text())
    moves, refused = m.plan_moves(students, min_age_s=0)
    assert moves == [] and refused[0]["reason"] == "duplicate"
