import json

import pytest

from scripts import resampling_first_only_early as early


def row(prompt, completion, numbers):
    return {"prompt": prompt, "completion": completion, "numbers": numbers}


def test_replace_reference_allows_context_differences_and_preserves_rendering():
    carriers = [
        row("early prompt", "prefix 007 suffix 42", [7, 42]),
        row("other", "x 9", [9]),
    ]
    refs = [
        row("mature prompt", "mature 123 context 8 99", [123, 8, 99]),
        row("different", "ref 0005 and 6", [5, 6]),
    ]
    out = early.replace_first_from_reference(carriers, refs)
    assert [x["prompt"] for x in out] == [x["prompt"] for x in carriers]
    assert [x["completion"] for x in out] == ["prefix 123 suffix 42", "x 0005"]
    assert [x["numbers"] for x in out] == [[123, 42], [5]]


@pytest.mark.parametrize(
    "carriers, refs",
    [
        ([], []),
        ([row("p", "x", [])], [row("p", "1", [1])]),
        ([row("p", "x 1", [1])], [row("p", "x nope", [2])]),
        ([row("p", "x 1", [1])], [row("p", "x 2", [3])]),
    ],
)
def test_replace_reference_rejects_malformed_or_metadata_drift(carriers, refs):
    with pytest.raises(ValueError):
        early.replace_first_from_reference(carriers, refs)


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.setattr(early, "N", 2)
    training_dir = tmp_path / "training_source"
    training_dir.mkdir()
    (training_dir / "student_training.yaml").write_text("train: {}\n")
    monkeypatch.setattr(early.controls, "DEFAULT_CAMPAIGN_DIR", training_dir)
    root, campaign = tmp_path / "outputs", tmp_path / "campaign"
    for seed in early.POOL_SEEDS:
        for label, path in early.input_paths(root, seed).items():
            data = (
                [row("early", "007, 42", [7, 42]), row("early2", "9", [9])]
                if label == "carrier"
                else [row("ref", "5, 6, 7", [5, 6, 7]), row("ref2", "0008, 7", [8, 7])]
            )
            path.parent.mkdir(parents=True)
            path.write_text("".join(json.dumps(r) + "\n" for r in data))
            (path.parent / "manifest.json").write_text(
                json.dumps(
                    {"n_filtered": 2, "filtered_sha256": early.controls._sha256(path)}
                )
            )
    frozen = early.prepare(root, campaign)
    return root, campaign, frozen


def test_prepare_has_frozen_shape_and_is_idempotent(prepared):
    root, campaign, frozen = prepared
    jobs = [
        json.loads(s)
        for s in (campaign / "jobs_distill.jsonl").read_text().splitlines()
    ]
    assert frozen["new_gpu_jobs"] == len(jobs) == len({j["key"] for j in jobs}) == 54
    assert frozen["pool_seeds"] == [0, 1, 2]
    assert early.prepare(root, campaign) == frozen
    for draw in early.POOL_SEEDS:
        for recipient in early.RECIPIENTS:
            for seed in early.controls.STUDENT_SEEDS:
                early.validate_run(root, campaign, seed, recipient, draw)


@pytest.mark.parametrize(
    "student,recipient,draw", [(4, 75000, 0), (1, 80000, 0), (1, 75000, 3)]
)
def test_validate_run_rejects_outside_grid(prepared, student, recipient, draw):
    root, campaign, _ = prepared
    with pytest.raises(ValueError, match="outside campaign"):
        early.validate_run(root, campaign, student, recipient, draw)


@pytest.mark.parametrize("name", ["student_training.yaml", "jobs_distill.jsonl"])
def test_runtime_rejects_changed_training_and_jobs(prepared, name):
    root, campaign, _ = prepared
    early.validate_run(root, campaign, 1, 75000, 0)
    path = campaign / name
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="input drift"):
        early.validate_run(root, campaign, 1, 75000, 0)


@pytest.mark.parametrize("name", ["filtered.jsonl", "manifest.json"])
def test_runtime_rejects_changed_derived_pool(prepared, name):
    root, campaign, _ = prepared
    early.validate_run(root, campaign, 1, 75000, 0)
    path = early.controls.pool_dir(root, early.ARM, 0) / name
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="pool drift"):
        early.validate_run(root, campaign, 1, 75000, 0)


@pytest.mark.parametrize("parent", ["carrier", "paired_iid"])
@pytest.mark.parametrize("name", ["filtered.jsonl", "manifest.json"])
def test_runtime_rejects_changed_parent(prepared, parent, name):
    root, campaign, _ = prepared
    early.validate_run(root, campaign, 1, 75000, 0)
    path = early.input_paths(root, 0)[parent].parent / name
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="parent hash drift"):
        early.validate_run(root, campaign, 1, 75000, 0)


@pytest.mark.parametrize(
    "name", ["campaign.json", "student_training.yaml", "jobs_distill.jsonl"]
)
def test_prepare_rejects_drift_without_overwriting(prepared, name):
    root, campaign, _ = prepared
    changed = campaign / name
    changed.write_text(changed.read_text() + " ")
    before = {p.name: p.read_bytes() for p in campaign.iterdir()}
    with pytest.raises(ValueError, match="frozen campaign input drift"):
        early.prepare(root, campaign)
    assert before == {p.name: p.read_bytes() for p in campaign.iterdir()}


def test_run_validates_then_dispatches_exact_arm(prepared, monkeypatch):
    root, campaign, _ = prepared
    calls = []
    monkeypatch.setattr(early.controls, "run_cell", lambda *args: calls.append(args))
    early.run(root, campaign, 3, 115000, 2)
    assert calls == [(root, early.ARM, 2, 115000, 3, campaign)]
