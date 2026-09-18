"""Tests for scripts/clean_partial_runs.py -- the killed-job partial-artifact cleaner.

Every test builds its own tmp tree; the REAL output root is never touched. The safety
invariants under test are the ones whose failure would destroy GPU-months of data.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from subliminal_learning.run_gate import write_resume_marker
from scripts.clean_partial_runs import (
    CleanPlan,
    TargetedRefusal,
    apply_actions,
    list_live_cmdlines,
    parse_seeds_from_cmd,
    plan_scan,
    plan_targeted,
)

ALL = (
    "teacher", "numbers", "eval", "distill", "steerability", "eas", "coherence",
)


def _mk_plan(
    output_root: Path,
    *,
    targeted: bool = False,
    min_age_s: float | None = None,   # default: scan 6h, targeted 0 (age guard off in tests)
    now: float | None = None,
    dataset_cache_root: Path | None = None,
    dataset_cache_allowed: bool = False,
    coherence_orphan_npy: bool = False,
    live_cmdlines: tuple[str, ...] = (),
) -> CleanPlan:
    allowed = [output_root.resolve()]
    if dataset_cache_allowed and dataset_cache_root is not None:
        allowed.append(dataset_cache_root.resolve())
    if min_age_s is None:
        min_age_s = 0.0 if targeted else 21600.0
    return CleanPlan(
        output_root=output_root,
        allowed_roots=tuple(allowed),
        dataset_cache_root=dataset_cache_root,
        dataset_cache_allowed=dataset_cache_allowed,
        min_age_s=min_age_s,
        now=time.time() if now is None else now,
        targeted=targeted,
        coherence_orphan_npy=coherence_orphan_npy,
        live_cmdlines=live_cmdlines,
    )


def _leaf(base: Path, *, complete: bool, with_ckpt_dir: bool = True) -> Path:
    """Build a training leaf: complete = adapter weights present; else empty checkpoints/."""
    if with_ckpt_dir:
        ckpt = base / "checkpoints"
        ckpt.mkdir(parents=True, exist_ok=True)
        (ckpt / "adapter_config.json").write_text("{}")
        if complete:
            (ckpt / "adapter_model.safetensors").write_bytes(b"weights")
            (base / "train_manifest.json").write_text("{}")
    else:
        base.mkdir(parents=True, exist_ok=True)
    return base


def _age(path: Path, seconds_old: float) -> None:
    """Backdate `path` -- and everything under it -- so the scan age guard sees it as stale.

    Recursive because the leaf age guard reads the newest write ANYWHERE under the leaf
    (a leaf dir's own mtime does not move when checkpoints are written inside it).
    """
    t = time.time() - seconds_old
    import os

    if path.is_dir():
        for child in path.rglob("*"):
            os.utime(child, (t, t), follow_symlinks=False)
    os.utime(path, (t, t))


# --------------------------------------------------------------------------- leaf selection


def test_complete_leaf_never_selected(tmp_path):
    cell = tmp_path / "teachers" / "cat" / "pythia-160m" / "ep1_lr2e-4_r8a8"
    leaf = _leaf(cell / "ftseed0", complete=True)
    _age(leaf, 999999)
    plan = _mk_plan(tmp_path)
    actions = plan_scan(plan, ALL)
    assert all(leaf != a.path for a in actions)


def test_incomplete_leaf_with_checkpoints_but_no_weights_is_selected(tmp_path):
    cell = tmp_path / "teachers" / "cat" / "pythia-160m" / "ep1_lr2e-4_r8a8"
    leaf = _leaf(cell / "ftseed0", complete=False)
    _age(leaf, 999999)
    plan = _mk_plan(tmp_path)
    actions = plan_scan(plan, ALL)
    assert any(a.path == leaf and a.kind == "rmtree_leaf" for a in actions)


def test_leaf_without_checkpoints_dir_not_selected(tmp_path):
    cell = tmp_path / "teachers" / "cat" / "pythia-160m" / "ep1_lr2e-4_r8a8"
    leaf = _leaf(cell / "ftseed0", complete=False, with_ckpt_dir=False)
    _age(leaf, 999999)
    plan = _mk_plan(tmp_path)
    actions = plan_scan(plan, ALL)
    assert all(leaf != a.path for a in actions)


def _student_leaf(tmp_path: Path) -> Path:
    return (tmp_path / "students" / "dog" / "pythia-160m" / "teacher-pythia-160m"
            / "x_tseed1" / "tag" / "st" / "ftseed3")


def _full_state_checkpoint(ckpt: Path, step: int = 500) -> Path:
    d = ckpt / f"checkpoint-{step}"
    d.mkdir(parents=True)
    (d / "adapter_config.json").write_text("{}")
    (d / "adapter_model.safetensors").write_bytes(b"weights")
    (d / "trainer_state.json").write_text(json.dumps({"global_step": step}))
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
        (d / name).write_bytes(b"state")
    write_resume_marker(d, step=step, reason="scheduled")
    return d


def test_resumable_checkpoint_not_selected(tmp_path):
    """Protection keys on a MARKED full-state dir (`run_gate.latest_resumable_checkpoint`)."""
    leaf = _student_leaf(tmp_path)
    _full_state_checkpoint(leaf / "checkpoints")
    _age(leaf, 999999)
    plan = _mk_plan(tmp_path)
    actions = plan_scan(plan, ALL)
    assert all(leaf != a.path for a in actions)


def test_weights_only_checkpoint_protects_the_leaf(tmp_path):
    """A trajectory (`save_only_model`) dir buys no resume, but it IS a writer's progress.

    The leaf dir's own mtime is 7 h old (adding `checkpoints/checkpoint-N` never touches
    it) and the writer is on another node, so neither the age guard nor the /proc liveness
    scan sees it: the checkpoint dir is the only signal, and it must shield the leaf.
    """
    leaf = _student_leaf(tmp_path)
    d = leaf / "checkpoints" / "checkpoint-500"
    d.mkdir(parents=True)
    (d / "adapter_config.json").write_text("{}")
    (d / "adapter_model.safetensors").write_bytes(b"weights")
    _age(leaf, 7 * 3600)
    plan = _mk_plan(tmp_path)
    assert all(leaf != a.path for a in plan_scan(plan, ALL))


def test_age_guard_reads_the_newest_write_under_the_leaf(tmp_path):
    """A fresh file deep in a 7-h-old leaf is "too-new", even with no checkpoint dir."""
    leaf = _leaf(tmp_path / "teachers" / "cat" / "m" / "tag" / "ftseed0", complete=False)
    _age(leaf, 7 * 3600)
    fresh = leaf / "checkpoints" / "adapter_config.json"   # written 1 min ago
    _age(fresh, 60)
    plan = _mk_plan(tmp_path)
    assert all(leaf != a.path for a in plan_scan(plan, ALL))
    # ... and the same leaf with nothing touched for 7 h is still purgeable.
    _age(fresh, 7 * 3600)
    assert any(leaf == a.path for a in plan_scan(_mk_plan(tmp_path), ALL))


@pytest.mark.parametrize("interior", [0, 39])
def test_finished_run_protected_regardless_of_interior_checkpoints(tmp_path, interior):
    """No `checkpoint-{final_step}` any more: the root adapter alone marks a finished run."""
    leaf = _leaf(_student_leaf(tmp_path), complete=True)
    for step in range(625, 625 * (interior + 1), 625):
        d = leaf / "checkpoints" / f"checkpoint-{step}"
        d.mkdir(parents=True)
        (d / "adapter_model.safetensors").write_bytes(b"weights")
    _age(leaf, 999999)
    plan = _mk_plan(tmp_path)
    assert all(leaf != a.path for a in plan_scan(plan, ALL))


def test_numbers_ftseed_never_selected_regression(tmp_path):
    """The 4872-false-positive guard: numbers/ pool ftseed dirs have no manifest/checkpoints."""
    pool = tmp_path / "numbers" / "cat" / "pythia-160m" / "ep1_lr2e-4_r8a8" / "ftseed0" / "ex10-20"
    pool.mkdir(parents=True)
    (pool / "raw.jsonl").write_text("{}\n")
    (pool / "manifest.json").write_text("{}")
    snap = tmp_path / "numbers_snapshots" / "cat" / "m" / "tag" / "ftseed1"
    snap.mkdir(parents=True)
    for p in (pool, snap):
        _age(p, 999999)
    plan = _mk_plan(tmp_path)
    actions = plan_scan(plan, ALL)
    assert all(a.kind != "rmtree_leaf" for a in actions)
    assert all("numbers" not in a.path.parts and "numbers_snapshots" not in a.path.parts
               for a in actions if a.kind == "rmtree_leaf")


# --------------------------------------------------------------------------- tmp orphans


@pytest.mark.parametrize(
    "rel,stage",
    [
        ("teachers/cat/m/tag/ftseed0/train_manifest.json.tmp.123", "teacher"),
        ("students/dog/m/teacher-m/x_tseed1/tag/st/ftseed0/train_manifest.json.tmp.9", "distill"),
        ("numbers/cat/m/tag/ftseed0/ex10-20/raw.jsonl.tmp.5", "numbers"),
        ("teachers/cat/m/tag/ftseed0/preference_eval.json.tmp.7", "eval"),
        ("teachers/cat/m/tag/preference_summary.json.tmp.7", "eval"),
    ],
)
def test_orphan_tmp_removed_per_stage(tmp_path, rel, stage):
    f = tmp_path / rel
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("partial")
    _age(f, 999999)
    plan = _mk_plan(tmp_path)
    actions = plan_scan(plan, ALL)
    hit = [a for a in actions if a.path == f]
    assert hit and hit[0].kind == "rm_file" and hit[0].stage == stage


# --------------------------------------------------------------------------- steerability / eas


def test_steerability_aggregate_deleted_ft_seed_kept(tmp_path):
    cell = tmp_path / "teachers" / "cat" / "m" / "tag" / "steerability" / "self"
    cell.mkdir(parents=True)
    agg = cell / "aggregate.json"
    ft = cell / "ft1.json"
    agg.write_text("{}")
    ft.write_text("{}")
    for p in (agg, ft):
        _age(p, 999999)
    plan = _mk_plan(tmp_path)
    actions = plan_scan(plan, ALL)
    assert any(a.path == agg and a.kind == "rm_file" for a in actions)
    assert all(a.path != ft for a in actions)


def test_steerability_complete_aggregate_kept(tmp_path):
    cell = tmp_path / "teachers" / "cat" / "m" / "tag" / "steerability" / "self"
    cell.mkdir(parents=True)
    (cell / "ft1.json").write_text("{}")
    (cell / "ft2.json").write_text("{}")
    agg = cell / "aggregate.json"
    agg.write_text('{"n_seeds": 2, "seeds": [1, 2]}')  # matches the 2 ft*.json present
    for p in cell.glob("*.json"):
        _age(p, 999999)
    plan = _mk_plan(tmp_path)
    actions = plan_scan(plan, ALL)
    assert all(a.path != agg for a in actions)


def test_eas_real_artifacts_kept_tmp_removed(tmp_path):
    edir = (tmp_path / "students" / "dog" / "m"
            / "teacher-m" / "x_tseed1" / "tag" / "eas")
    edir.mkdir(parents=True)
    real_json = edir / "eas_last.json"
    real_pt = edir / "v_teacher_last.pt"
    tmp = edir / "eas_last.json.tmp.42"
    for p in (real_json, real_pt, tmp):
        p.write_text("x")
        _age(p, 999999)
    plan = _mk_plan(tmp_path)
    actions = plan_scan(plan, ALL)
    assert any(a.path == tmp for a in actions)
    assert all(a.path not in (real_json, real_pt) for a in actions)


# --------------------------------------------------------------------------- out of scope


def test_out_of_scope_trees_skipped(tmp_path):
    for tree in ("sdf_teachers", "gkd_students", "coherence_notatree"):
        leaf = _leaf(tmp_path / tree / "cat" / "m" / "tag" / "ftseed0", complete=False)
        _age(leaf, 999999)
    plan = _mk_plan(tmp_path)
    actions = plan_scan(plan, ALL)
    assert all("sdf_teachers" not in a.path.parts and "gkd_students" not in a.path.parts
               for a in actions)


def test_targeted_out_of_scope_path_no_actions(tmp_path):
    leaf = _leaf(tmp_path / "gkd_students" / "cat" / "m" / "tag" / "gseed0", complete=False)
    plan = _mk_plan(tmp_path, targeted=True)
    assert plan_targeted(plan, leaf, ()) == []


# --------------------------------------------------------------------------- age guard


def test_age_guard_skips_fresh_selects_stale(tmp_path):
    cell = tmp_path / "teachers" / "cat" / "m" / "tag"
    fresh = _leaf(cell / "ftseed0", complete=False)
    stale = _leaf(cell / "ftseed1", complete=False)
    _age(fresh, 60)  # 1 min old -> under 6h guard
    _age(stale, 999999)
    plan = _mk_plan(tmp_path)
    paths = {a.path for a in plan_scan(plan, ALL)}
    assert stale in paths and fresh not in paths


# --------------------------------------------------------------------------- containment / symlink


def test_refuses_target_outside_output_root(tmp_path):
    outside = tmp_path / "elsewhere" / "ftseed0"
    _leaf(outside, complete=False)
    plan = _mk_plan(tmp_path / "outputs")
    (tmp_path / "outputs").mkdir()
    assert plan.is_safe_target(outside) is False


def test_refuses_symlinked_target(tmp_path):
    real = _leaf(tmp_path / "teachers" / "cat" / "m" / "tag" / "ftseed0", complete=False)
    link = tmp_path / "teachers" / "cat" / "m" / "tag" / "ftseedlink"
    link.symlink_to(real)
    plan = _mk_plan(tmp_path)
    assert plan.is_safe_target(link) is False


# --------------------------------------------------------------------------- dry-run deletes nothing


def test_dry_run_deletes_nothing(tmp_path):
    leaf = _leaf(tmp_path / "teachers" / "cat" / "m" / "tag" / "ftseed0", complete=False)
    _age(leaf, 999999)
    plan = _mk_plan(tmp_path)
    actions = plan_scan(plan, ALL)
    assert actions  # something to do
    # We simply never call apply_actions in a dry run:
    assert leaf.exists() and (leaf / "checkpoints").exists()


def test_apply_actually_removes(tmp_path):
    leaf = _leaf(tmp_path / "teachers" / "cat" / "m" / "tag" / "ftseed0", complete=False)
    _age(leaf, 999999)
    plan = _mk_plan(tmp_path)
    actions = plan_scan(plan, ALL)
    count, _ = apply_actions(actions, plan)
    assert count == 1 and not leaf.exists()


# --------------------------------------------------------------------------- multi-seed targeted


def _cell_with_seeds(tmp_path, incomplete: set[int], complete: set[int]) -> Path:
    cell = (tmp_path / "students" / "dog" / "pythia-160m"
            / "teacher-pythia-160m" / "x_tseed1" / "tag" / "st")
    for s in incomplete:
        _leaf(cell / f"ftseed{s}", complete=False)
    for s in complete:
        _leaf(cell / f"ftseed{s}", complete=True)
    return cell


def test_targeted_seeds_selects_only_that_seed(tmp_path):
    cell = _cell_with_seeds(tmp_path, incomplete={2}, complete={1, 3})
    plan = _mk_plan(tmp_path, targeted=True)
    actions = plan_targeted(plan, cell, (), seeds_opt={2})
    assert [a.path for a in actions] == [cell / "ftseed2"]


def test_targeted_from_job_cmd_loop_selects_all_incomplete(tmp_path):
    cell = _cell_with_seeds(tmp_path, incomplete={2}, complete={1, 3})
    cmd = "bash -c 'for S in 1 2 3; do python -m scripts.distill_student --student-seed $S; done'"
    plan = _mk_plan(tmp_path, targeted=True)
    actions = plan_targeted(plan, cell, (), from_job_cmd=cmd)
    # loop selects 1,2,3 but only seed 2 is incomplete -> only it is removed
    assert [a.path for a in actions] == [cell / "ftseed2"]


def test_targeted_from_job_cmd_single_seed(tmp_path):
    cell = _cell_with_seeds(tmp_path, incomplete={2, 3}, complete={1})
    cmd = "python -m scripts.distill_student --target-animal dog --student-seed 2"
    plan = _mk_plan(tmp_path, targeted=True)
    actions = plan_targeted(plan, cell, (), from_job_cmd=cmd)
    assert [a.path for a in actions] == [cell / "ftseed2"]


def test_targeted_no_selection_refuses(tmp_path):
    cell = _cell_with_seeds(tmp_path, incomplete={2}, complete=set())
    plan = _mk_plan(tmp_path, targeted=True)
    with pytest.raises(TargetedRefusal):
        plan_targeted(plan, cell, (), from_job_cmd="python -m scripts.foo --no-seeds-here")


def test_targeted_all_seeds_cleans_incomplete(tmp_path):
    cell = _cell_with_seeds(tmp_path, incomplete={2, 4}, complete={1})
    plan = _mk_plan(tmp_path, targeted=True)
    actions = plan_targeted(plan, cell, (), all_seeds=True)
    assert {a.path for a in actions} == {cell / "ftseed2", cell / "ftseed4"}


# --------------------------------------------------------------------------- liveness guard


def test_live_leaf_not_deleted_even_with_apply(tmp_path):
    cell = tmp_path / "teachers" / "dog" / "pythia-160m" / "ep2_lr5e-4_r32a64"
    leaf = _leaf(cell / "ftseed2", complete=False)
    live = ("python -m scripts.train_teacher_templates --target-animal dog --seeds 2",)
    plan = _mk_plan(tmp_path, targeted=True, live_cmdlines=live)
    actions = plan_targeted(plan, cell, (), seeds_opt={2})
    count, _ = apply_actions(actions, plan)
    assert actions == [] and count == 0 and leaf.exists()


def test_live_guard_lets_dead_sibling_through(tmp_path):
    cell = tmp_path / "teachers" / "dog" / "pythia-160m" / "ep2_lr5e-4_r32a64"
    live_leaf = _leaf(cell / "ftseed2", complete=False)
    dead_leaf = _leaf(cell / "ftseed5", complete=False)
    live = ("python -m scripts.train_teacher_templates --target-animal dog --seeds 2",)
    plan = _mk_plan(tmp_path, targeted=True, live_cmdlines=live)
    actions = plan_targeted(plan, cell, (), seeds_opt={2, 5})
    assert [a.path for a in actions] == [dead_leaf]
    assert live_leaf.exists()


# --------------------------------------------------------------------------- coherence


def _adapter_pc_dir(tmp_path: Path) -> Path:
    d = tmp_path / "teachers" / "owl" / "m" / "p" / "ftseed0" / "pile_coherence"
    d.mkdir(parents=True)
    return d


def test_coherence_run_tmp_file_always_removed(tmp_path):
    d = _adapter_pc_dir(tmp_path)
    tmp = d / "v1.json.tmp.1234"
    tmp.write_text("{}")
    _age(tmp, 999999)
    actions = plan_scan(_mk_plan(tmp_path), ALL)
    assert any(a.path == tmp and a.kind == "rm_file" for a in actions)


def test_coherence_complete_run_artifact_untouched(tmp_path):
    d = _adapter_pc_dir(tmp_path)
    jsn = d / "v1.json"
    npy = d / "v1_per_token_loss.npy"
    jsn.write_text('{"model_id": "m"}')
    np.save(npy, np.zeros(3, dtype=np.float32))
    _age(jsn, 999999)
    _age(npy, 999999)
    actions = plan_scan(_mk_plan(tmp_path, coherence_orphan_npy=True), ALL)
    assert all(a.path not in (jsn, npy) for a in actions)


def test_coherence_run_orphan_npy_only_with_flag(tmp_path):
    d = _adapter_pc_dir(tmp_path)
    npy = d / "v1_per_token_loss.npy"
    np.save(npy, np.zeros(3, dtype=np.float32))  # no sibling v1.json
    _age(npy, 999999)
    without = plan_scan(_mk_plan(tmp_path), ALL)
    assert all(a.path != npy for a in without)
    with_flag = plan_scan(_mk_plan(tmp_path, coherence_orphan_npy=True), ALL)
    assert any(a.path == npy for a in with_flag)


def test_coherence_run_json_missing_npy_only_with_flag(tmp_path):
    d = _adapter_pc_dir(tmp_path)
    jsn = d / "v1.json"
    jsn.write_text('{"model_id": "m"}')  # no sibling npy
    _age(jsn, 999999)
    without = plan_scan(_mk_plan(tmp_path), ALL)
    assert all(a.path != jsn for a in without)
    with_flag = plan_scan(_mk_plan(tmp_path, coherence_orphan_npy=True), ALL)
    assert any(a.path == jsn for a in with_flag)


def test_coherence_base_orphan_npy_only_with_flag(tmp_path):
    d = tmp_path / "coherence" / "v1" / "base" / "pythia-160m"
    d.mkdir(parents=True)
    npy = d / "per_token_loss.npy"
    np.save(npy, np.zeros(3, dtype=np.float32))  # no ce.json
    _age(npy, 999999)
    without = plan_scan(_mk_plan(tmp_path), ALL)
    assert all(a.path != npy for a in without)
    with_flag = plan_scan(_mk_plan(tmp_path, coherence_orphan_npy=True), ALL)
    assert any(a.path == npy for a in with_flag)


def test_coherence_base_complete_untouched(tmp_path):
    d = tmp_path / "coherence" / "v1" / "base" / "pythia-160m"
    d.mkdir(parents=True)
    jsn = d / "ce.json"
    npy = d / "per_token_loss.npy"
    jsn.write_text('{"model_id": "m", "role": "base"}')
    np.save(npy, np.zeros(3, dtype=np.float32))
    _age(jsn, 999999)
    _age(npy, 999999)
    actions = plan_scan(_mk_plan(tmp_path, coherence_orphan_npy=True), ALL)
    assert all(a.path not in (jsn, npy) for a in actions)


# --------------------------------------------------------------------------- coherence shard


def _shard(ds_root: Path, version: str, *, header: bool) -> Path:
    d = ds_root / "pile_coherence" / version
    d.mkdir(parents=True)
    np.save(d / "tokens.npy", np.zeros(2, dtype=np.int32))
    np.save(d / "doc_ids.npy", np.zeros(2, dtype=np.int32))
    if header:
        (d / "header.json").write_text('{"shard_version": "%s"}' % version)
    return d


def test_shard_missing_header_selected_when_allowed(tmp_path):
    out = tmp_path / "outputs"
    out.mkdir()
    ds = tmp_path / "datasets"
    partial = _shard(ds, "v9", header=False)
    _age(partial, 999999)
    plan = _mk_plan(out, dataset_cache_root=ds, dataset_cache_allowed=True)
    actions = plan_scan(plan, ALL)
    assert any(a.path == partial and a.kind == "rmtree_shard" for a in actions)


def test_shard_with_header_never_selected(tmp_path):
    out = tmp_path / "outputs"
    out.mkdir()
    ds = tmp_path / "datasets"
    complete = _shard(ds, "v9", header=True)
    _age(complete, 999999)
    plan = _mk_plan(out, dataset_cache_root=ds, dataset_cache_allowed=True)
    actions = plan_scan(plan, ALL)
    assert all(a.path != complete for a in actions)


def test_shard_refused_when_dataset_cache_root_not_explicitly_allowed(tmp_path):
    out = tmp_path / "outputs"
    out.mkdir()
    ds = tmp_path / "datasets"
    partial = _shard(ds, "v9", header=False)
    _age(partial, 999999)
    # dataset_cache_root known (for detection) but NOT explicitly allowed:
    plan = _mk_plan(out, dataset_cache_root=ds, dataset_cache_allowed=False)
    actions = plan_scan(plan, ALL)
    assert all(a.path != partial for a in actions)


# --------------------------------------------------------------------------- seed parser units


def test_list_live_cmdlines_excludes_self(tmp_path):
    """Our own --path invocation must not self-match the liveness guard (real bug fixed)."""
    proc = tmp_path / "proc"
    (proc / "100").mkdir(parents=True)
    (proc / "200").mkdir()
    (proc / "100" / "cmdline").write_bytes(b"python\x00-m\x00scripts.distill_student\x00--seed\x002")
    (proc / "200" / "cmdline").write_bytes(
        b"python\x00-m\x00scripts.clean_partial_runs\x00--path\x00/x/ftseed2"
    )
    cmds = list_live_cmdlines(str(proc))
    assert any("distill_student" in c for c in cmds)
    assert all("clean_partial_runs" not in c for c in cmds)


@pytest.mark.parametrize(
    "cmd,expected",
    [
        ("--student-seed 2", {2}),
        ("--student-seeds 1,2,3", {1, 2, 3}),
        ("--seed=7", {7}),
        ("for S in 1 2 3; do foo --student-seed $S; done", {1, 2, 3}),
        ("--teacher-ftseed 4 --subsample-seed 9", set()),
        ("no seeds here", set()),
    ],
)
def test_parse_seeds_from_cmd(cmd, expected):
    assert parse_seeds_from_cmd(cmd) == expected


# ------------------------------------------------- targeted recent-activity guard (cross-node)


def test_targeted_refuses_recently_modified_leaf(tmp_path):
    cell = tmp_path / "teachers" / "dog" / "pythia-160m" / "ep2_lr5e-4_r32a64"
    leaf = _leaf(cell / "ftseed2", complete=False)
    plan = _mk_plan(tmp_path, targeted=True, min_age_s=1800.0)
    actions = plan_targeted(plan, leaf, ())
    count, _ = apply_actions(actions, plan)
    assert actions == [] and count == 0 and leaf.exists()


def test_targeted_min_age_zero_overrides(tmp_path):
    cell = tmp_path / "teachers" / "dog" / "pythia-160m" / "ep2_lr5e-4_r32a64"
    leaf = _leaf(cell / "ftseed2", complete=False)
    plan = _mk_plan(tmp_path, targeted=True, min_age_s=0.0)
    actions = plan_targeted(plan, leaf, ())
    assert [a.path for a in actions] == [leaf]
    apply_actions(actions, plan)
    assert not leaf.exists()


def test_targeted_old_leaf_removed(tmp_path):
    cell = tmp_path / "teachers" / "dog" / "pythia-160m" / "ep2_lr5e-4_r32a64"
    leaf = _leaf(cell / "ftseed2", complete=False)
    for p in (leaf / "checkpoints" / "adapter_config.json", leaf / "checkpoints", leaf):
        _age(p, 999999)
    plan = _mk_plan(tmp_path, targeted=True, min_age_s=1800.0)
    actions = plan_targeted(plan, leaf, ())
    assert [a.path for a in actions] == [leaf]
    apply_actions(actions, plan)
    assert not leaf.exists()


# --------------------------------------------------------------------------- leaf discovery


def _discovered(output_root: Path, base: Path) -> list[Path]:
    """Leaves the scan finds under ``base`` (every leaf here is an incomplete candidate)."""
    from scripts.clean_partial_runs import _scan_leaves

    plan = _mk_plan(output_root, min_age_s=0.0)
    return sorted(a.path for a in _scan_leaves(plan, base, "distill"))


def _serial_leaves(base: Path) -> list[Path]:
    """Reference discovery: the ``rglob("ftseed*")`` walk the parallel one replaced."""
    return sorted(d for d in base.rglob("ftseed*") if d.is_dir())


def test_leaf_walk_matches_serial_rglob_on_a_nested_tree(tmp_path):
    root = tmp_path / "students"
    leaves = [
        _leaf(root / "dog" / "m" / "teacher-m" / "t_tseed1" / "tag" / "st" / "ftseed0",
              complete=False),
        _leaf(root / "dog" / "m" / "teacher-m" / "t_tseed1" / "tag" / "st" / "ftseed1",
              complete=False),
        _leaf(root / "cat" / "m" / "tag" / "ftseed3", complete=False),
    ]
    (root / "cat" / "m" / "tag" / "ftseed3" / "notes.txt").write_text("x")  # a file, not a leaf
    assert _discovered(tmp_path, root) == sorted(leaves) == _serial_leaves(root)


def test_leaf_walk_never_follows_a_symlinked_leaf_outside_the_root(tmp_path):
    """Archived animals are symlinks; this tool DELETES, so they must stay invisible."""
    root = tmp_path / "teachers"
    inside = _leaf(root / "dog" / "m" / "tag" / "ftseed0", complete=False)
    outside = _leaf(tmp_path / "archive" / "cat" / "m" / "tag" / "ftseed9", complete=False)
    (root / "cat").symlink_to(outside.parents[2], target_is_directory=True)
    assert _discovered(tmp_path, root) == [inside]


def test_gseed_leaf_is_discovered_like_ftseed(tmp_path):
    """Scan mode must use the same ``_LEAF_PREFIXES`` as targeted mode (gseed included)."""
    root = tmp_path / "students"
    ft = _leaf(root / "dog" / "m" / "tag" / "ftseed1", complete=False)
    gs = _leaf(root / "dog" / "m" / "tag" / "gseed3", complete=False)
    assert _discovered(tmp_path, root) == sorted([ft, gs])


def test_nested_leaf_warns_and_is_not_descended(tmp_path, capsys):
    root = tmp_path / "students"
    outer = _leaf(root / "dog" / "m" / "tag" / "ftseed1", complete=False)
    inner = _leaf(outer / "ftseed2", complete=False)
    assert _discovered(tmp_path, root) == [outer]
    err = capsys.readouterr().err
    assert "nested inside a leaf" in err and str(inner) in err


def test_walk_error_refuses_every_deletion_but_still_reports(tmp_path, capsys, monkeypatch):
    """A truncated walk must produce a report and delete NOTHING (absent != partial)."""
    from scripts import clean_partial_runs as cpr

    root = tmp_path / "students"
    leaf = _leaf(root / "dog" / "m" / "tag" / "ftseed1", complete=False)
    _age(root, 10 * 3600)
    plan = _mk_plan(tmp_path)

    real = cpr.find_leaf_dirs_parallel

    def flaky(roots, is_leaf, **kw):
        errors = kw.get("errors")
        if errors is not None:
            errors.append((root / "broken", "OSError"))
        return real(roots, is_leaf, **kw)

    monkeypatch.setattr(cpr, "find_leaf_dirs_parallel", flaky)
    actions = cpr._scan_leaves(plan, root, "distill")
    assert [a.path for a in actions] == [leaf]  # the report is still produced
    assert plan.deletions_refused
    assert "delete NOTHING" in capsys.readouterr().err
    assert apply_actions(actions, plan) == (0, 0)
    assert leaf.exists()
