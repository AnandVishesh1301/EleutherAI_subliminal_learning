import copy
import json

import pytest

from scripts.campaigns import build_checkpoint_studenthp as b
from scripts.campaigns import build_checkpoint_studenthp as f
from subliminal_learning.utils.io import file_sha256


# -------------------- from test_checkpoint_studenthp_preflight.py --------------------

def test_selected_hps_are_core30_and_exclude_baseline(tmp_path):
    tags = list(b.HP_TAGS) + [f"x{i}" for i in range(25)]
    cells = [{"student_tag": tag, "lr": 1e-4, "rank": 1, "alpha": 1} for tag in tags]
    p = tmp_path / "core.json"
    p.write_text(json.dumps({"cells": cells}))
    assert [x["student_tag"] for x in b.selected_hps(p)] == list(b.HP_TAGS)
    assert "ep10_lr1e-4_r8a32" not in b.HP_TAGS

def test_frozen_write_rejects_drift(tmp_path):
    p = tmp_path / "f"
    b.frozen_write(p, "a")
    with pytest.raises(ValueError):
        b.frozen_write(p, "b")

def test_verified_cell_rejects_campaign_drift_and_duplicates(tmp_path):
    f = tmp_path / "input"
    f.write_text("x")
    digest = file_sha256(f)
    c = {"shared_files": {str(f): digest}, "cells": [{"key": "k", "files": {}}]}
    cp = tmp_path / "c.json"
    cp.write_text(json.dumps(c))
    cs = file_sha256(cp)
    assert b.verified_cell(cp, cs, "k")["key"] == "k"
    c["cells"].append({"key": "k", "files": {}})
    cp.write_text(json.dumps(c))
    with pytest.raises(ValueError):
        b.verified_cell(cp, file_sha256(cp), "k")

def test_verified_cell_rejects_source_hash_drift(tmp_path):
    f = tmp_path / "input"
    f.write_text("x")
    c = {"shared_files": {str(f): file_sha256(f)}, "cells": [{"key": "k", "files": {}}]}
    cp = tmp_path / "c.json"
    cp.write_text(json.dumps(c))
    cs = file_sha256(cp)
    f.write_text("changed")
    with pytest.raises(ValueError):
        b.verified_cell(cp, cs, "k")


# -------------------- from test_checkpoint_studenthp_reverse.py --------------------

# Forward and reverse helpers now live in one module (`--direction forward|reverse`).
r = f

def _campaign(tmp_path):
    pools = {str(t): {"files": {f"pool-{t}": f"poolhash-{t}"}} for t in f.TEACHER_STEPS}
    cells = []
    for t in f.TEACHER_STEPS:
        for s in f.STEPS:
            for tag in f.HP_TAGS:
                files = {f"cfg-{tag}": f"cfg-{tag}", **pools[str(t)]["files"]}
                cells.append({"key": f"t{t}-s{s}-{tag}-seed1", "teacher_step": t, "student_step": s, "student_tag": tag, "student_seed": 1, "run_dir": str(tmp_path / f"teacher-pythia-160m@step{t}" / f"student-{s}-{tag}"), "argv": ["--teacher-step", str(t)], "files": files})
    return {"cells": cells, "pools": pools}

def _reverse_pools():
    return {str(s): {"files": {f"pool-{s}": f"poolhash-{s}"}} for s in f.STEPS}

def test_reverse_counts_axes_hps_seed_and_union(tmp_path):
    campaign = _campaign(tmp_path)
    new, covered = r.make_reverse_cells(campaign, _reverse_pools())
    assert len(new) == 110 and len(covered) == 20
    assert {c["teacher_step"] for c in new + covered} == set(f.STEPS)
    assert {c["student_step"] for c in new + covered} == set(f.TEACHER_STEPS)
    assert {c["student_tag"] for c in new + covered} == set(f.HP_TAGS)
    assert {c["student_seed"] for c in new + covered} == {1}
    assert len({c["key"] for c in new + covered}) == 130

def test_reverse_rejects_overlap_pool_drift(tmp_path):
    campaign = _campaign(tmp_path)
    pools = _reverse_pools()
    pools["75000"]["files"]["pool-75000"] = "wrong"
    with pytest.raises(ValueError, match="overlap"):
        r.make_reverse_cells(campaign, pools)

def test_reverse_rejects_overlap_hp_config_drift(tmp_path):
    campaign = _campaign(tmp_path)
    campaign["cells"][0]["files"]["cfg-" + f.HP_TAGS[0]] = "wrong"
    with pytest.raises(ValueError, match="overlap"):
        r.make_reverse_cells(campaign, _reverse_pools())

def test_load_forward_receipt_and_protocol_validation(tmp_path, monkeypatch):
    campaign = _campaign(tmp_path)
    campaign.update({"name": "queue", "animal": "dog", "model": "Pythia-160M", "teacher_recipe": f.TEACHER_TAG, "teacher_seed": 1, "student_seeds": [1], "protocol": copy.deepcopy(r.PROTOCOL), "student_hps": f.selected_hps(), "shared_files": {}})
    monkeypatch.setattr(f, "check_files", lambda files: None)
    cp = tmp_path / "campaign.json"
    jp = tmp_path / "jobs.jsonl"
    cp.write_text(json.dumps(campaign))
    jp.write_text("")
    (tmp_path / "submission.json").write_text(json.dumps({"campaign_sha256": file_sha256(cp), "jobs_sha256": file_sha256(jp), "queue": "queue"}))
    loaded, _ = r.load_forward(cp)
    assert len(loaded["cells"]) == 130
    campaign["protocol"]["dtype"] = "bfloat16"
    cp.write_text(json.dumps(campaign))
    with pytest.raises(ValueError, match="forward campaign differs|protocol"):
        r.load_forward(cp)

def test_split_jobs_exact_balanced_partition(tmp_path):
    campaign = _campaign(tmp_path)
    new, _ = r.make_reverse_cells(campaign, _reverse_pools())
    campaign["cells"] = new
    cp = tmp_path / "campaign.json"
    jp = tmp_path / "jobs.jsonl"
    cp.write_text(json.dumps(campaign))
    jp.write_text("".join(json.dumps({"key": c["key"], "cmd": ["same"]}) + "\n" for c in new))
    manifest = r.split_jobs(cp, jp, tmp_path / "split", nodes=("node-a", "node-b"))
    assert [manifest["queues"][x]["count"] for x in r.SPLIT_LABELS] == [55, 55]
    a = (tmp_path / "split/jobs_a.jsonl").read_text().splitlines()
    b = (tmp_path / "split/jobs_b.jsonl").read_text().splitlines()
    assert not ({json.loads(x)["key"] for x in a} & {json.loads(x)["key"] for x in b})
    assert {json.loads(x)["key"] for x in a + b} == {c["key"] for c in new}
    assert [manifest["queues"][x]["node"] for x in r.SPLIT_LABELS] == ["node-a", "node-b"]
    for label in r.SPLIT_LABELS:
        rows = [json.loads(x) for x in (tmp_path / f"split/jobs_{label}.jsonl").read_text().splitlines()]
        assert {r["key"].split("-seed1")[0].split("-", 2)[2] for r in rows} == set(f.HP_TAGS)
        assert {r["key"].split("-", 2)[0][1:] for r in rows} == {str(x) for x in f.STEPS if x not in f.TEACHER_STEPS}
        assert {r["key"].split("-", 2)[1][1:] for r in rows} == {str(x) for x in f.TEACHER_STEPS}

def test_split_rejects_duplicate_or_missing_key(tmp_path):
    campaign = _campaign(tmp_path)
    new, _ = r.make_reverse_cells(campaign, _reverse_pools())
    campaign["cells"] = new
    cp = tmp_path / "campaign.json"
    cp.write_text(json.dumps(campaign))
    jobs = [{"key": c["key"]} for c in new[:-1]] + [{"key": new[0]["key"]}]
    jp = tmp_path / "jobs.jsonl"
    jp.write_text("".join(json.dumps(j) + "\n" for j in jobs))
    with pytest.raises(ValueError, match="110 unique"):
        r.split_jobs(cp, jp, tmp_path / "split")

def test_split_cli_without_output_root(tmp_path):
    campaign = _campaign(tmp_path)
    new, _ = r.make_reverse_cells(campaign, _reverse_pools())
    campaign["cells"] = new
    cp = tmp_path / "campaign.json"
    jp = tmp_path / "jobs.jsonl"
    cp.write_text(json.dumps(campaign))
    jp.write_text("".join(json.dumps({"key": c["key"]}) + "\n" for c in new))
    import subprocess
    result = subprocess.run(["uv", "run", "python", "-m", "scripts.campaigns.build_checkpoint_studenthp", "--direction", "reverse", "--split-only", "--covered-campaign", str(cp), "--jobs", str(jp), "--out", str(tmp_path / "out")], capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)['total_jobs'] == 110
    assert len((tmp_path / "out/jobs_a.jsonl").read_text().splitlines()) == 55


# -------------------- from test_studenthp_preflight_paths.py --------------------

CONFIG = b.REPO_ROOT / "configs/training/lora_student_distill.yaml"

def test_local_path_verifies_as_before():
    b.check_files({str(CONFIG): file_sha256(CONFIG)})

def test_foreign_checkout_path_reanchors_to_local_copy():
    foreign = f"/nonexistent/{b.REPO_ROOT.name}/configs/training/lora_student_distill.yaml"
    assert b.local_file(foreign) == CONFIG
    b.check_files({foreign: file_sha256(CONFIG)})

def test_repo_relative_path_resolves():
    b.check_files({"configs/training/lora_student_distill.yaml": file_sha256(CONFIG)})

def test_reanchored_sha_mismatch_still_raises():
    foreign = f"/nonexistent/{b.REPO_ROOT.name}/configs/training/lora_student_distill.yaml"
    with pytest.raises(ValueError):
        b.check_files({foreign: "0" * 64})

def test_missing_file_raises_with_recorded_path(tmp_path):
    missing = str(tmp_path / "gone.yaml")
    with pytest.raises(ValueError, match="gone.yaml"):
        b.check_files({missing: "0" * 64})

def test_record_path_is_repo_relative_for_repo_files(tmp_path):
    assert b.record_path(CONFIG) == "configs/training/lora_student_distill.yaml"
    outside = tmp_path / "f.yaml"
    outside.write_text("x")
    assert b.record_path(outside) == str(outside.resolve())
