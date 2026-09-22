import hashlib
import json
from pathlib import Path
import shlex
import sys

import pytest

from scripts import resampling_first_only as first


def write_parent(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row) + "\n" for row in data)
    path.write_text(payload)
    (path.parent / "manifest.json").write_text(
        json.dumps(
            {
                "n_filtered": len(data),
                "filtered_sha256": hashlib.sha256(payload.encode()).hexdigest(),
            }
        )
    )


@pytest.fixture
def parents(tmp_path, monkeypatch):
    monkeypatch.setattr(first, "N", 2)
    monkeypatch.setattr(first.controls, "ARM_SPECS", dict(first.controls.ARM_SPECS))
    carrier = [
        {
            "prompt": "a",
            "completion": "01, 22; 333!",
            "numbers": [1, 22, 333],
            "stale": True,
        },
        {"prompt": "b", "completion": " 9.", "numbers": [9]},
    ]
    iid = [
        {"prompt": "a", "completion": "8, 8; 8!", "numbers": [8, 8, 8]},
        {"prompt": "b", "completion": " 8.", "numbers": [8]},
    ]
    paths = first.input_paths(tmp_path)
    write_parent(paths["carrier"], carrier)
    write_parent(paths["paired_iid"], iid)
    return tmp_path, carrier, iid


def test_saved_pair_build_and_immutable_resume(parents):
    root, _, _ = parents
    path, manifest = first.build(root)
    data = [
        json.loads(line) for line in (path / "filtered.jsonl").read_text().splitlines()
    ]
    assert data == [
        {"prompt": "a", "completion": "8, 22; 333!", "numbers": [8, 22, 333]},
        {"prompt": "b", "completion": " 8.", "numbers": [8]},
    ]
    assert manifest["diagnostics"]["pairing"]["singleton_rows"] == 1
    stamps = [
        (path / name).stat().st_mtime_ns for name in ("filtered.jsonl", "manifest.json")
    ]
    assert first.build(root)[1] == manifest
    assert stamps == [
        (path / name).stat().st_mtime_ns for name in ("filtered.jsonl", "manifest.json")
    ]
    (path / "filtered.jsonl").write_text("corrupt\n")
    with pytest.raises(ValueError, match="drift"):
        first.build(root)


def test_parent_changes_cannot_silently_relabel_pool(parents):
    root, carrier, _ = parents
    first.build(root)
    carrier[0]["completion"] = "01, 23; 333!"
    carrier[0]["numbers"][1] = 23
    write_parent(first.input_paths(root)["carrier"], carrier)
    with pytest.raises(ValueError, match="drift"):
        first.build(root)


def test_rejects_different_carrier_layout(parents):
    root, _, iid = parents
    iid[0]["completion"] = "8; 8; 8!"
    write_parent(first.input_paths(root)["paired_iid"], iid)
    with pytest.raises(ValueError, match="formatting mismatch"):
        first.build(root)


def test_emit_only_new_arm_and_standard_runner(parents, tmp_path, monkeypatch):
    root, _, _ = parents
    source_campaign = tmp_path / "reference"
    source_campaign.mkdir()
    (source_campaign / "student_training.yaml").write_text("train: {n_epochs: 10}\n")
    monkeypatch.setattr(first.controls, "DEFAULT_CAMPAIGN_DIR", source_campaign)
    campaign = tmp_path / "campaign"
    jobs = [
        json.loads(line) for line in first.emit(root, campaign).read_text().splitlines()
    ]
    assert len(jobs) == 3
    assert all("scripts.resampling_first_only run" in job["cmd"] for job in jobs)
    assert all("resampling_controls run" not in job["cmd"] for job in jobs)
    calls = []
    monkeypatch.setattr(
        first.controls, "run_cell", lambda *args: calls.append(args) or Path("report")
    )
    assert first.run(root, campaign, 2) == Path("report")
    assert calls == [(root, first.ARM, 0, 125000, 2, campaign)]


def prepare_campaign(root, tmp_path, monkeypatch, **kwargs):
    source = tmp_path / "reference"
    source.mkdir()
    (source / "student_training.yaml").write_text("train: {n_epochs: 10}\n")
    monkeypatch.setattr(first.controls, "DEFAULT_CAMPAIGN_DIR", source)
    campaign = tmp_path / "extension"
    first.emit(root, campaign, **kwargs)
    return campaign


def test_extension_runs_exact_missing_cells_from_frozen_pools(
    parents, tmp_path, monkeypatch
):
    root, _, _ = parents
    recipients = (75000, 85000, 95000, 105000, 115000)
    iid_recipients = (85000, 95000, 105000, 115000)
    iid_arm = "step125000_synthiid"
    campaign = prepare_campaign(
        root,
        tmp_path,
        monkeypatch,
        recipients=recipients,
        arms=(first.ARM, iid_arm),
        iid_recipients=iid_recipients,
    )
    jobs = [
        json.loads(line)
        for line in (campaign / "jobs_distill.jsonl").read_text().splitlines()
    ]
    assert len(jobs) == len({job["key"] for job in jobs}) == 27
    calls = []
    monkeypatch.setattr(
        first.controls, "run_cell", lambda *args: calls.append(args) or Path("report")
    )
    for job in jobs:
        tokens = shlex.split(job["cmd"])
        monkeypatch.setattr(sys, "argv", ["first-only", *tokens[5:]])
        first.main()
    expected = {
        (root, arm, 0, recipient, seed, campaign)
        for arm, steps in ((first.ARM, recipients), (iid_arm, iid_recipients))
        for recipient in steps
        for seed in (1, 2, 3)
    }
    assert len(calls) == 27
    assert set(calls) == expected
    frozen = json.loads((campaign / "campaign.json").read_text())
    iid = first.input_paths(root)["paired_iid"]
    assert frozen["control_pool_sha256"] == first.controls._sha256(iid)
    assert frozen["control_manifest_sha256"] == first.controls._sha256(
        iid.parent / "manifest.json"
    )


def test_iid_rejects_changed_pool_even_with_updated_parent_manifest(
    parents, tmp_path, monkeypatch
):
    root, _, iid = parents
    campaign = prepare_campaign(
        root,
        tmp_path,
        monkeypatch,
        arms=(first.ARM, "step125000_synthiid"),
        iid_recipients=(85000,),
    )
    iid[0]["completion"] = "7, 8; 8!"
    iid[0]["numbers"][0] = 7
    write_parent(first.input_paths(root)["paired_iid"], iid)
    with pytest.raises(ValueError, match="IID control pool drift"):
        first.run(root, campaign, 1, "step125000_synthiid", 85000)


@pytest.mark.parametrize("arm", [first.ARM, "step125000_synthiid"])
def test_recipient_and_training_config_are_enforced(
    parents, tmp_path, monkeypatch, arm
):
    root, _, _ = parents
    campaign = prepare_campaign(
        root,
        tmp_path,
        monkeypatch,
        recipients=(85000,),
        arms=(first.ARM, "step125000_synthiid"),
        iid_recipients=(85000,),
    )
    with pytest.raises(ValueError, match="outside campaign"):
        first.run(root, campaign, 1, arm, 95000)
    with pytest.raises(ValueError, match="student seed is outside campaign"):
        first.run(root, campaign, 4, arm, 85000)
    (campaign / "student_training.yaml").write_text("train: {n_epochs: 11}\n")
    with pytest.raises(ValueError, match="campaign input drift"):
        first.run(root, campaign, 1, arm, 85000)


def test_original_campaign_and_cli_remain_usable(parents, tmp_path, monkeypatch):
    root, _, _ = parents
    campaign = prepare_campaign(root, tmp_path, monkeypatch)
    path = campaign / "campaign.json"
    saved = json.loads(path.read_text())
    for key in (
        "recipients",
        "iid_recipients",
        "arms",
        "control_pool_sha256",
        "control_manifest_sha256",
    ):
        saved.pop(key)
    saved["recipient"] = 125000
    path.write_text(json.dumps(saved))
    calls = []
    monkeypatch.setattr(
        first.controls, "run_cell", lambda *args: calls.append(args) or Path("report")
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "first-only",
            "run",
            "--output-root",
            str(root),
            "--campaign-dir",
            str(campaign),
            "--student-seed",
            "2",
        ],
    )
    first.main()
    assert calls == [(root, first.ARM, 0, 125000, 2, campaign)]


@pytest.mark.parametrize(
    "extra",
    [
        ["--recipient", "75000", "85000"],
        ["--arm", first.ARM, "step125000_synthiid"],
    ],
)
def test_run_cli_rejects_multiple_cells(tmp_path, monkeypatch, extra):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "first-only",
            "run",
            "--output-root",
            str(tmp_path),
            "--student-seed",
            "1",
            *extra,
        ],
    )
    with pytest.raises(SystemExit) as exc:
        first.main()
    assert exc.value.code == 2


def test_nonzero_pool_seeds_route_distinct_saved_parents(parents):
    root, carrier, iid = parents
    for pool_seed in (1, 2):
        shifted_carrier = [
            dict(row, completion=row["completion"].replace("01", str(pool_seed + 1), 1))
            for row in carrier
        ]
        shifted_carrier[0]["numbers"] = [pool_seed + 1, 22, 333]
        shifted_iid = [dict(row) for row in iid]
        shifted_iid[0]["completion"] = shifted_iid[0]["completion"].replace(
            "8", str(pool_seed + 7), 1
        )
        shifted_iid[0]["numbers"] = [pool_seed + 7, 8, 8]
        paths = first.input_paths(root, pool_seed)
        write_parent(paths["carrier"], shifted_carrier)
        write_parent(paths["paired_iid"], shifted_iid)
        path, manifest = first.build(root, pool_seed)
        assert f"poolseed{pool_seed}" in str(path)
        assert manifest["diagnostics"]["pool_seed"] == pool_seed
        assert (
            json.loads((path / "filtered.jsonl").read_text().splitlines()[0])[
                "numbers"
            ][0]
            == pool_seed + 7
        )


@pytest.mark.parametrize("pool_seed", [1, 2])
def test_nonzero_cli_executes_all_thirty_cells_on_intended_draw(parents, tmp_path, monkeypatch, pool_seed):
    root, _, _ = parents
    carrier = [
        {"prompt": "a", "completion": f"1, {22 + pool_seed}; 333!", "numbers": [1, 22 + pool_seed, 333]},
        {"prompt": "b", "completion": " 9.", "numbers": [9]},
    ]
    iid = [
        {"prompt": "a", "completion": f"{80 + pool_seed}, 8; 8!", "numbers": [80 + pool_seed, 8, 8]},
        {"prompt": "b", "completion": f" {80 + pool_seed}.", "numbers": [80 + pool_seed]},
    ]
    paths = first.input_paths(root, pool_seed)
    write_parent(paths["carrier"], carrier)
    write_parent(paths["paired_iid"], iid)
    recipients = (75000, 85000, 95000, 105000, 115000, 125000)
    iid_recipients = (85000, 95000, 105000, 115000)
    iid_arm = "step125000_synthiid"
    campaign = prepare_campaign(
        root, tmp_path, monkeypatch, recipients=recipients, arms=(first.ARM, iid_arm),
        iid_recipients=iid_recipients, pool_seed=pool_seed,
    )
    jobs = [json.loads(line) for line in (campaign / "jobs_distill.jsonl").read_text().splitlines()]
    assert len(jobs) == len({job["key"] for job in jobs}) == 30
    assert all(f"-p{pool_seed}-" in job["key"] for job in jobs)
    calls = []
    monkeypatch.setattr(first.controls, "run_cell", lambda *args: calls.append(args) or Path("report"))
    for job in jobs:
        tokens = shlex.split(job["cmd"])
        monkeypatch.setattr(sys, "argv", ["first-only", *tokens[5:]])
        first.main()
    assert len(calls) == 30
    assert set(calls) == {
        (root, arm, pool_seed, recipient, seed, campaign)
        for arm, steps in ((first.ARM, recipients), (iid_arm, iid_recipients))
        for recipient in steps for seed in (1, 2, 3)
    }
    built = first.controls.pool_dir(root, first.ARM, pool_seed)
    data = [json.loads(line) for line in (built / "filtered.jsonl").read_text().splitlines()]
    assert data[0]["numbers"] == [80 + pool_seed, 22 + pool_seed, 333]
    def forbidden_build(*args, **kwargs):
        pytest.fail("wrong-draw request must fail before any pool access")
    monkeypatch.setattr(first, "build", forbidden_build)
    for arm in (first.ARM, iid_arm):
        with pytest.raises(ValueError, match="pool seed is outside campaign"):
            first.run(root, campaign, 1, arm, 85000, pool_seed=0)
