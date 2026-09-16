import json
from pathlib import Path

import pytest

from scripts import migrate_resampling_layout as mig
from scripts import resampling_controls as campaign

ROWS = [
    {"prompt": f"p{i}", "numbers": [i % 4, i], "completion": f"{i % 4};\n {i}"}
    for i in range(8)
]


def _write_pool_files(pool, rows=ROWS, manifest=None):
    pool.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    (pool / "filtered.jsonl").write_text(text)
    (pool / "manifest.json").write_text(
        json.dumps(manifest if manifest is not None else {"source": "teacher"})
    )
    return campaign._sha256(pool / "filtered.jsonl")


def _write_student(param_dir, seed, pool, tag):
    run = param_dir / f"ftseed{seed}"
    (run / "checkpoints").mkdir(parents=True)
    fields = {
        "number_dataset_dir": str(pool),
        "number_dataset_tag": tag,
        "filtered_sha256": campaign._sha256(pool / "filtered.jsonl"),
    }
    (run / "train_manifest.json").write_text(
        json.dumps(
            {
                **fields,
                "checkpoint_dir": str(run / "checkpoints"),
                "run_identity": dict(fields),
            }
        )
    )
    (param_dir / "eval/dog").mkdir(parents=True, exist_ok=True)
    (param_dir / "eval/dog/preference_summary.json").write_text(
        json.dumps({"n_seeds": 1, "ftseed": seed})
    )
    return run


@pytest.fixture
def tree(tmp_path):
    """An output root holding one teacher arm, one synthetic arm and one August pool."""
    out = tmp_path / "outputs"
    for home in campaign.HOME_POOLS:
        _write_pool_files(campaign.home_pool_dir(out, home))
    made = {}
    for old_arm, recipient in (("early_uniform", 125000), ("synthetic_mature_iid", 75000)):
        pool = mig.old_pool_dir(out, old_arm, 0)
        text = "".join(json.dumps(r, sort_keys=True) + "\n" for r in ROWS)
        import hashlib

        sha = _write_pool_files(
            pool,
            manifest={
                "transform": old_arm,
                "parents": {},
                "filtered_sha256": hashlib.sha256(text.encode()).hexdigest(),
            },
        )
        source, _ = campaign.arm_source(campaign.LEGACY_ARMS[old_arm])
        for seed in (1, 2, 3):
            tag = f"{campaign.CAMPAIGN}_{old_arm}_p0_s{seed}"
            param = mig.student_param_dir(
                out, source=source, recipient=recipient,
                home=campaign.ARM_SPECS[campaign.LEGACY_ARMS[old_arm]].home, tag=tag,
            )
            _write_student(param, seed, pool, tag)
        made[old_arm] = (pool, sha)
    august = mig.old_mode_pool_dir(out, "prefix_entmatch125k")
    _write_pool_files(august, manifest={"copied": "from the early pool"})
    aug_param = mig.student_param_dir(
        out, source="teacher", recipient=125000, home="step75000",
        tag="ex10-20_prefix_n100k_entmatch125k",
    )
    _write_student(aug_param, 1, august, "ex10-20_prefix_n100k_entmatch125k")
    return out, made


def test_plan_lists_every_move_and_rewrite(tree):
    out, _ = tree
    plan = mig.build_plan(out)
    assert plan["counts"]["missing_sources"] == 0
    assert plan["counts"]["existing_destinations"] == 0
    # 2 campaign pools + 1 August pool; 3+3 campaign ftseeds + 1 August tag dir;
    # eval records only for the campaign arms (the August tag dir moves whole).
    assert {k: plan["counts"][k] for k in
            ("pool", "student_run", "eval_record", "manifest_rewrites",
             "missing_sources", "already_done", "existing_destinations")} == {
        "pool": 3,
        "student_run": 7,
        "eval_record": 6,
        "manifest_rewrites": 10,
        "missing_sources": 0,
        "already_done": 0,
        "existing_destinations": 0,
    }
    # The plan names every stale student field it will rewrite, not just the pool dir.
    assert {"checkpoint_dir", "number_dataset_home", "number_dataset_dir",
            "number_dataset_tag"} <= set(plan["counts"]["manifest_rewrite_fields"])
    dsts = {m["dst"] for m in plan["moves"]}
    assert str(campaign.pool_dir(out, "step75000_uniform", 0)) in dsts
    # A synthetic pool's student is re-homed under its teacher, not under controls/rng.
    assert str(
        out / "students/dog/ckpt/pythia-160m@step75000"
        / "teacher-pythia-160m@step125000/ep2_lr5e-4_r32a64_tseed1"
        / "ex10-20_prefix_ablations_synthiid_poolseed1/ep10_lr1e-4_r8a32/ftseed1"
    ) not in dsts
    assert str(
        out / "students/dog/ckpt/pythia-160m@step75000"
        / "teacher-pythia-160m@step125000/ep2_lr5e-4_r32a64_tseed1"
        / "ex10-20_prefix_ablations_synthiid_poolseed0/ep10_lr1e-4_r8a32/ftseed2"
    ) in dsts
    # The per-seed reports go DOWN into the ftseed dirs, so nothing is silently merged.
    records = [m for m in plan["moves"] if m["kind"] == "eval_record"]
    assert all(m["dst"].endswith("/eval/dog/preference_summary.json") for m in records)
    assert all("/ftseed" in m["dst"] for m in records)


def test_apply_moves_rewrites_and_verifies(tree):
    out, made = tree
    plan = mig.build_plan(out)
    result = mig.apply_plan(plan, out / "log")
    assert result["problems"] == []
    assert result["moved"] == len(plan["moves"])
    pool = campaign.pool_dir(out, "step75000_uniform", 0)
    manifest = json.loads((pool / "manifest.json").read_text())
    assert manifest["transform"] == "step75000_uniform"
    assert manifest["home"]["layout_version"] == 2
    assert manifest["home"]["rows_source"] == "teacher@step75000"
    assert not mig.old_pool_dir(out, "early_uniform", 0).exists()
    run = (
        out / "students/dog/ckpt/pythia-160m@step125000"
        / "teacher-pythia-160m@step75000/ep2_lr5e-4_r32a64_tseed1"
        / "ex10-20_prefix_ablations_uniform_poolseed0/ep10_lr1e-4_r8a32/ftseed3"
    )
    student = json.loads((run / "train_manifest.json").read_text())
    assert student["number_dataset_dir"] == str(pool)
    assert student["number_dataset_tag"] == "ex10-20_prefix_ablations_uniform_poolseed0"
    assert student["run_identity"] == {
        **student["run_identity"],
        "number_dataset_dir": str(pool),
        "number_dataset_tag": "ex10-20_prefix_ablations_uniform_poolseed0",
    }
    assert (run / "eval/dog/preference_summary.json").is_file()
    log = json.loads(next((out / "log").glob("migration_applied_*.json")).read_text())
    assert len(log["reverse_mapping"]) == len(plan["moves"])
    assert {m["kind"] for m in log["reverse_mapping"]} == {
        "pool", "student_run", "eval_record",
    }


def test_august_pool_gets_a_real_manifest(tree):
    out, _ = tree
    mig.apply_plan(mig.build_plan(out), out / "log")
    pool = campaign.legacy_mode_pool_dir(out, "prefix_entmatch125k")
    manifest = json.loads((pool / "manifest.json").read_text())
    assert manifest["transform"] == "step75000_waterfill-step125000"
    assert manifest["diagnostics"]["sampling"] == "water_fill_without_replacement"
    assert "entropy_match_ablation.py" in manifest["diagnostics"]["note"]
    assert manifest["filtered_sha256"] == campaign._sha256(pool / "filtered.jsonl")
    assert manifest["n_filtered"] == len(ROWS)
    assert manifest["pool_stats"]["n"] == len(ROWS)
    # Both source pools are recorded as parents, with their hashes; they did not move.
    assert set(manifest["parents"]) == {"step75000", "step125000"}
    for home, parent in manifest["parents"].items():
        source = campaign.home_pool_dir(out, home) / "filtered.jsonl"
        assert parent["filtered_path"] == str(source) and source.is_file()
        assert parent["filtered_sha256"] == campaign._sha256(source)
    assert manifest["home"]["home_pool"] == "numbers/" + campaign.HOME_POOLS["step75000"]
    # The student tag dir renamed whole (single seed, no merge).
    assert (
        out / "students/dog/ckpt/pythia-160m@step125000"
        / "teacher-pythia-160m@step75000/ep2_lr5e-4_r32a64_tseed1"
        / "ex10-20_prefix_ablations_waterfill-step125000_poolseed0"
        / "ep10_lr1e-4_r8a32/ftseed1/train_manifest.json"
    ).is_file()


def test_apply_refuses_when_a_destination_exists(tree):
    """Both src and dst present is a real collision, not an interrupted apply."""
    out, _ = tree
    campaign.pool_dir(out, "step75000_uniform", 0).mkdir(parents=True)
    assert mig.old_pool_dir(out, "early_uniform", 0).is_dir()
    plan = mig.build_plan(out)
    with pytest.raises(FileExistsError):
        mig.apply_plan(plan, out / "log")
    assert mig.old_pool_dir(out, "early_uniform", 0).is_dir()
    assert mig.build_plan(out)["counts"]["existing_destinations"] == 1


def test_stale_cache_references_are_reported_not_modified(tree):
    out, _ = tree
    cache = out / "analysis/record_cache_pref_summary.csv"
    cache.parent.mkdir(parents=True)
    before = (
        "run_dir\n"
        f"{out}/students/dog/ckpt/pythia-160m@step125000/teacher-pythia-160m@step75000/"
        f"ep2_lr5e-4_r32a64_tseed1/{campaign.CAMPAIGN}_early_uniform_p0_s1/x\n"
    )
    cache.write_text(before)
    plan = mig.build_plan(out)
    assert [e["path"] for e in plan["stale_cache_references"]] == [str(cache)]
    mig.apply_plan(plan, out / "log")
    assert cache.read_text() == before


def _fail_on_nth(monkeypatch, n):
    """Make the n-th os.rename blow up the way an interrupted apply would."""
    calls = {"n": 0}
    real = mig.os.rename

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] == n:
            raise RuntimeError("simulated crash")
        return real(src, dst)

    monkeypatch.setattr(mig.os, "rename", flaky)
    return calls


def _manifest_bytes(out):
    return {
        str(p): p.read_bytes()
        for p in sorted((out / "students").rglob("train_manifest.json"))
    }


def test_apply_resumes_after_a_mid_apply_failure(tree, monkeypatch):
    out, _ = tree
    plan = mig.build_plan(out)
    _fail_on_nth(monkeypatch, 5)
    with pytest.raises(RuntimeError):
        mig.apply_plan(plan, out / "log")
    log_path = out / "log" / f"migration_applied_{campaign.CAMPAIGN}.json"
    partial = json.loads(log_path.read_text())
    # The crashed run already logged every move that landed, in order.
    assert len(partial["reverse_mapping"]) == 4
    assert all(m["how"] == "rename" and m["ts"] for m in partial["reverse_mapping"])
    assert partial["applied_utc"] and partial["last_updated_utc"]

    monkeypatch.undo()
    result = mig.apply_plan(mig.build_plan(out), out / "log")
    assert result["problems"] == []
    assert result["moved"] == len(plan["moves"])
    log = json.loads(log_path.read_text())
    assert log["applied_utc"] == partial["applied_utc"]
    assert log["last_updated_utc"] > partial["last_updated_utc"]
    pairs = [(m["old"], m["new"]) for m in log["reverse_mapping"]]
    assert sorted(pairs) == sorted(set(pairs))
    assert pairs == [(m["src"], m["dst"]) for m in plan["moves"]]
    assert sum(1 for m in log["reverse_mapping"] if m["how"] == "already_moved") == 4
    # A third apply is a pure no-op: no manifest is ever rewritten to different bytes.
    before = _manifest_bytes(out)
    again = mig.apply_plan(mig.build_plan(out), out / "log")
    assert again["problems"] == [] and _manifest_bytes(out) == before
    assert all(m["how"] == "already_moved" for m in
               json.loads(log_path.read_text())["reverse_mapping"])


def test_apply_rewrites_checkpoint_dir_and_home(tree):
    out, _ = tree
    mig.apply_plan(mig.build_plan(out), out / "log")
    run = (
        out / "students/dog/ckpt/pythia-160m@step125000"
        / "teacher-pythia-160m@step75000/ep2_lr5e-4_r32a64_tseed1"
        / "ex10-20_prefix_ablations_uniform_poolseed0/ep10_lr1e-4_r8a32/ftseed3"
    )
    manifest = json.loads((run / "train_manifest.json").read_text())
    assert manifest["checkpoint_dir"] == str(run / "checkpoints")
    assert manifest["number_dataset_home"] == "numbers/" + campaign.HOME_POOLS["step75000"]
    # Provenance only: the gate-relevant identity never learns about the home.
    assert "number_dataset_home" not in manifest["run_identity"]
    old_run = mig.student_param_dir(
        out, source="teacher", recipient=125000, home="step75000",
        tag=f"{campaign.CAMPAIGN}_early_uniform_p0_s3",
    ) / "ftseed3"
    assert manifest["layout_v2"] == {
        "migrated_from": str(old_run),
        "home_pool": "numbers/" + campaign.HOME_POOLS["step75000"],
        "cross_checkpoint": True,  # step-125000 student off the step-75000 home pool
        "layout_version": 2,
    }


def test_pool_diagnostics_keys_are_renamed(tree):
    out, _ = tree
    log = out / "log"
    log.mkdir()
    (log / "pool_diagnostics.json").write_text(
        json.dumps({"early_uniform/poolseed0": {"x": 1}, "untouched/poolseed0": {}})
    )
    plan = mig.build_plan(out, log)
    assert plan["counts"]["pool_diagnostics_renames"] == 1
    mig.apply_plan(plan, log)
    assert set(json.loads((log / "pool_diagnostics.json").read_text())) == {
        "step75000_uniform/poolseed0", "untouched/poolseed0",
    }


def test_prune_removes_only_recursively_empty_dirs(tree):
    out, _ = tree
    keep = mig.student_param_dir(
        out, source="teacher", recipient=125000, home="step75000",
        tag=f"{campaign.CAMPAIGN}_early_uniform_p0_s1",
    ).parent
    (keep / "notes.txt").write_text("a leftover file")
    result = mig.apply_plan(mig.build_plan(out), out / "log")
    assert keep.is_dir() and str(keep) not in result["pruned"]
    gone = mig.student_param_dir(
        out, source="teacher", recipient=125000, home="step75000",
        tag=f"{campaign.CAMPAIGN}_early_uniform_p0_s2",
    ).parent
    assert not gone.exists() and str(gone) in result["pruned"]
    assert not (out / "numbers/ablations" / campaign.CAMPAIGN).exists()
    log = json.loads(
        (out / "log" / f"migration_applied_{campaign.CAMPAIGN}.json").read_text()
    )
    assert log["pruned_dirs"] == result["pruned"]
    # rmdir walks UP only: the shared teacher dir above the pruned tag dirs stays.
    assert gone.parent.is_dir()


def test_verify_reports_a_missing_key_and_apply_exits_nonzero(tree, monkeypatch):
    out, _ = tree
    plan = mig.build_plan(out)
    mig.apply_plan(plan, out / "log")
    target = Path(
        next(
            r["path"] for m in plan["moves"] for r in m["rewrites"]
            if r["kind"] == "student"
        )
    )
    doc = json.loads(target.read_text())
    doc.pop("filtered_sha256")
    target.write_text(json.dumps(doc))
    problems = mig.verify(plan)  # reports, never raises
    assert any("filtered_sha256" in p and str(target) in p for p in problems)

    # A tag that no longer matches the dataset-tag dir it sits in is reported too.
    doc["filtered_sha256"] = campaign._sha256(
        Path(doc["number_dataset_dir"]) / "filtered.jsonl"
    )
    doc["number_dataset_tag"] = "some_other_tag"
    doc["run_identity"]["number_dataset_tag"] = "some_other_tag"
    target.write_text(json.dumps(doc))
    problems = mig.verify(plan)
    assert any("tag does not match its dataset dir" in p for p in problems)
    assert any("run_identity tag does not match its dataset dir" in p for p in problems)

    monkeypatch.setattr(mig, "build_plan", lambda *a, **k: plan)
    monkeypatch.setattr(mig, "apply_plan", lambda *a, **k: {
        "moved": 0, "problems": problems, "log": "x", "pruned": [],
    })
    import sys

    monkeypatch.setattr(
        sys, "argv",
        ["prog", "--output-root", str(out), "--plan-path", str(out / "plan.json"),
         "--log-dir", str(out / "log"), "--apply"],
    )
    with pytest.raises(SystemExit) as exc:
        mig.main()
    assert exc.value.code != 0
