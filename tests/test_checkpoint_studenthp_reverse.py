import json
import copy

import pytest

from scripts import build_checkpoint_studenthp_preflight as f
from scripts import build_checkpoint_studenthp_reverse as r

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
    (tmp_path / "submission.json").write_text(json.dumps({"campaign_sha256": f.sha(cp), "jobs_sha256": f.sha(jp), "queue": "queue"}))
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
    manifest = r.split_jobs(cp, jp, tmp_path / "split")
    assert [manifest["queues"][x]["count"] for x in ("soar1", "soar4")] == [55, 55]
    a = (tmp_path / "split/jobs_soar1.jsonl").read_text().splitlines()
    b = (tmp_path / "split/jobs_soar4.jsonl").read_text().splitlines()
    assert not ({json.loads(x)["key"] for x in a} & {json.loads(x)["key"] for x in b})
    assert {json.loads(x)["key"] for x in a + b} == {c["key"] for c in new}
    assert manifest["queues"]["soar1"]["node"] == "soar-subliminal-1"
    assert manifest["queues"]["soar4"]["node"] == "soar-subliminal-4"
    for label in ("soar1", "soar4"):
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
    result = subprocess.run(["uv", "run", "python", "-m", "scripts.build_checkpoint_studenthp_reverse", "--split-only", "--covered-campaign", str(cp), "--jobs", str(jp), "--out", str(tmp_path / "out")], capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)['total_jobs'] == 110
    assert len((tmp_path / "out/jobs_soar1.jsonl").read_text().splitlines()) == 55
