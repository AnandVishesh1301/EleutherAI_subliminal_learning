"""All three families of the matched untuned-sender control builder."""

from __future__ import annotations

import json
import re
import shlex

import pytest

from scripts import build_fullseed_control_jobs as builder


@pytest.fixture(scope="module", params=["full-seed", "data-seed", "same-model"])
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
    assert sorted(p.name for p in out.iterdir()) == [
        "README.md", "campaign.json", "configs", "controls.jobs.jsonl",
        "generation.jobs.jsonl", "number_generation.yaml",
    ]
    n_cells, n_jobs, n_gen = (20, 80, 0) if family == "same-model" else (56, 672, 3)
    assert len(list((out / "configs").glob("*.yaml"))) == n_cells
    controls, generation = jobs(out / "controls.jobs.jsonl"), jobs(out / "generation.jobs.jsonl")
    assert (len(controls), len(generation)) == (n_jobs, n_gen)
    assert summary["control_jobs"] == n_jobs and summary["cells_per_direction"] == n_cells
    assert summary["control_queue"].endswith("20260914" if family == "full-seed" else "20260916")


def test_token_map_and_keys(campaign):
    family, out, _ = campaign
    if family == "same-model":
        pytest.skip("the diagonal family has no off-diagonal keys and generates no pools")
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
    if family == "same-model":
        pytest.skip("covered per-direction in test_same_model_diagonal")
    text = (out / "controls.jobs.jsonl").read_text() + (out / "generation.jobs.jsonl").read_text()
    assert ("allow-full-seed" in text) is (family == "full-seed")


def test_same_model_diagonal(campaign):
    """Only the diagonal, only focus20 cells, and the force flag only on the fs bases."""
    family, out, _ = campaign
    if family != "same-model":
        pytest.skip("diagonal family only")
    controls = jobs(out / "controls.jobs.jsonl")
    focus20 = set(builder.focus20_cells(
        builder.listed_cells((builder.ROOT / "run_logs/core56_cells.json").read_text())))
    assert len(focus20) == 20
    senders = set()
    for job in controls:
        sender, receiver, tag = re.fullmatch(
            r"base-(\w+?)-to-(\w+?)-(ep20_.+)-s1", job["key"]).groups()
        assert sender == receiver and tag in focus20
        senders.add(sender)
        worker = shlex.split(job["cmd"].split(" && ")[0])
        base = worker[worker.index("--base") + 1]
        assert worker[worker.index("--numbers-base") + 1] == base
        assert ("allow-full-seed" in job["cmd"]) is (sender != "orig")
    assert senders == {"orig", "fs1", "fs2", "fs3"}


def test_other_families_unchanged_by_diagonal_support(campaign):
    """The off-diagonal families' manifests stay byte-identical to the committed campaigns."""
    family, out, _ = campaign
    if family == "same-model":
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
