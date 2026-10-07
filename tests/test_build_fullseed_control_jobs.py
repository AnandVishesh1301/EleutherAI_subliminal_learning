"""All four families of the matched untuned-sender control builder."""

from __future__ import annotations

import json
import re
import shlex

import pytest

from scripts import build_fullseed_control_jobs as builder


@pytest.fixture(scope="module", params=["full-seed", "data-seed", "same-model", "same-model-core30"])
def campaign(request, tmp_path_factory):
    """Build each family into a scratch dir so the real campaigns are untouched."""
    out = tmp_path_factory.mktemp(request.param)
    rel = out.relative_to(builder.ROOT) if out.is_relative_to(builder.ROOT) else out
    summary = builder.build(request.param, rel)
    return request.param, out, summary


def jobs(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_layout_and_counts(campaign):
    family, out, summary = campaign
    expected = ["README.md", "campaign.json", "configs", "controls.jobs.jsonl",
                "generation.jobs.jsonl", "number_generation.yaml"]
    if family == "same-model-core30":
        expected += [f"ctrl_runs_{arm}.txt" for arm in ("fs1", "fs2", "fs3", "orig")]
        expected += ["probes.jobs.jsonl"]
    assert sorted(p.name for p in out.iterdir()) == sorted(expected)
    n_cells, n_jobs, n_gen = {"same-model": (20, 80, 0), "same-model-core30": (10, 40, 0)}.get(
        family, (56, 672, 3))
    assert len(list((out / "configs").glob("*.yaml"))) == n_cells
    controls, generation = jobs(out / "controls.jobs.jsonl"), jobs(out / "generation.jobs.jsonl")
    assert (len(controls), len(generation)) == (n_jobs, n_gen)
    assert summary["control_jobs"] == n_jobs and summary["cells_per_direction"] == n_cells
    date = {"full-seed": "20260914", "same-model-core30": "20260921"}.get(family, "20260916")
    assert summary["control_queue"].endswith(date)


def test_token_map_and_keys(campaign):
    family, out, _ = campaign
    if family.startswith("same-model"):
        pytest.skip("the diagonal families have no off-diagonal keys and generate no pools")
    controls = jobs(out / "controls.jobs.jsonl")
    tokens = {"full-seed": ("full-seed1", "fs1"), "data-seed": ("data-seed1", "ds1")}[family]
    keys = {job["key"] for job in controls}
    assert f"base-orig-to-{tokens[1]}-ep20_lr1e-4_r128a32-s1" in keys
    assert f"base-{tokens[1]}-to-orig-ep20_lr1e-4_r128a32-s1" in keys
    job = next(j for j in controls if j["key"] == f"base-orig-to-{tokens[1]}-ep20_lr1e-4_r128a32-s1")
    worker = shlex.split(job["cmd"].split(" && ")[0])
    assert worker[worker.index("--base") + 1] == tokens[0]
    assert worker[worker.index("--numbers-base") + 1] == "original"
    assert generation_tokens(out) == {tokens[0].replace("1", str(i)) for i in (1, 2, 3)}


def generation_tokens(out):
    tokens = set()
    for job in jobs(out / "generation.jobs.jsonl"):
        args = shlex.split(job["cmd"])
        tokens.add(args[args.index("--base-models") + 1])
    return tokens


def test_force_flag_only_for_full_seed(campaign):
    family, out, _ = campaign
    if family.startswith("same-model"):
        pytest.skip("covered per-direction in test_same_model_diagonal")
    text = (out / "controls.jobs.jsonl").read_text() + (out / "generation.jobs.jsonl").read_text()
    assert ("allow-full-seed" in text) is (family == "full-seed")


def test_same_model_diagonal(campaign):
    """Only the diagonal, only the family's cells, and the force flag only on the fs bases."""
    family, out, _ = campaign
    if not family.startswith("same-model"):
        pytest.skip("diagonal families only")
    controls = jobs(out / "controls.jobs.jsonl")
    focus20 = set(builder.focus20_cells(
        builder.listed_cells((builder.ROOT / "run_logs/registry/core56_cells.json").read_text())))
    assert len(focus20) == 20
    expected = focus20 if family == "same-model" else cell_tags("core30") - cell_tags("focus20")
    senders, seen = set(), set()
    for job in controls:
        sender, receiver, tag = re.fullmatch(
            r"base-(\w+?)-to-(\w+?)-(ep20_.+)-s1", job["key"]).groups()
        assert sender == receiver and tag in expected
        senders.add(sender)
        seen.add(tag)
        worker = shlex.split(job["cmd"].split(" && ")[0])
        base = worker[worker.index("--base") + 1]
        assert worker[worker.index("--numbers-base") + 1] == base
        assert ("allow-full-seed" in job["cmd"]) is (sender != "orig")
    assert senders == {"orig", "fs1", "fs2", "fs3"} and seen == expected


def cell_tags(name):
    doc = json.loads((builder.ROOT / f"run_logs/registry/{name}_cells.json").read_text())
    return {c["student_tag"] for c in doc["cells"]}


def test_same_model_core30_wing_and_probes(campaign):
    """40 = 4 arms x the 10 core30 \\ focus20 cells, no focus20 cell, one 10-run probe job per arm."""
    family, out, summary = campaign
    if family != "same-model-core30":
        pytest.skip("core30 wing family only")
    wing = cell_tags("core30") - cell_tags("focus20")
    assert len(wing) == 10 and set(summary["cells"]) == wing
    controls = jobs(out / "controls.jobs.jsonl")
    assert len(controls) == 40
    tags = [re.fullmatch(r"base-\w+?-to-\w+?-(ep20_.+)-s1", j["key"])[1] for j in controls]
    assert set(tags) == wing and not set(tags) & cell_tags("focus20")
    probes = jobs(out / "probes.jobs.jsonl")
    assert [p["key"] for p in probes] == ["probe-orig", "probe-fs1", "probe-fs2", "probe-fs3"]
    for job in probes:
        args = shlex.split(job["cmd"])
        assert args[args.index("--evals") + 1] == "probe"
        run_list = builder.ROOT / args[args.index("--runs") + 1]
        assert len(run_list.read_text().splitlines()) == 10
    for job in controls + probes:
        assert "/home/" not in job["cmd"]
    assert summary["probe_jobs"] == 4 and len(summary["submit"]) == 2
    assert "--depends-on" not in summary["submit"][0] and "--priority 52" in summary["submit"][0]
    assert "--depends-on " + summary["control_queue"] in summary["submit"][1]


def test_other_families_unchanged_by_diagonal_support(campaign):
    """The off-diagonal families' manifests stay byte-identical to the committed campaigns."""
    family, out, _ = campaign
    if family.startswith("same-model"):
        pytest.skip("no committed campaign to compare against")
    committed = builder.ROOT / builder.FAMILIES[family]["campaign"] / "controls.jobs.jsonl"
    rel = out.relative_to(builder.ROOT) if out.is_relative_to(builder.ROOT) else out
    expected = committed.read_text().replace(
        str(builder.FAMILIES[family]["campaign"]), str(rel))
    assert (out / "controls.jobs.jsonl").read_text() == expected


def test_no_tf32(campaign):
    _, out, _ = campaign
    # The README says "no tf32" in prose; no config or command may enable it.
    for path in [*out.glob("*.jsonl"), out / "number_generation.yaml", *(out / "configs").iterdir()]:
        assert "tf32" not in path.read_text()


def test_no_per_gpu_cap(campaign):
    _, out, summary = campaign
    assert "cap_per_gpu" not in summary and "cap_group" not in summary
    assert "cap_per_gpu" not in json.loads((out / "campaign.json").read_text())
    readme = (out / "README.md").read_text()
    assert "cap-per-gpu" not in readme and "jobs/GPU" not in readme
    for name in ("generation.jobs.jsonl", "controls.jobs.jsonl"):
        path = out / name
        if path.exists():
            assert "cap_per_gpu" not in path.read_text() and "cap_group" not in path.read_text()
