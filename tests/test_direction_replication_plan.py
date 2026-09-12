"""Campaign preparation is read-only on checkpoints and never submits work."""
import csv
import json
from pathlib import Path

import pytest

from scripts.plan_direction_replication import (
    BASES, ES, EXPECTED, KS, REPO, input_identity, job_specs, prepare,
    read_population, validate_identity,
)


def population(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run", "arm", "original", "rand0"])
        w.writeheader()
        w.writerows(rows)
    return path


def test_population_has_no_recovery_or_damage_filter(tmp_path):
    rows = [dict(run="a", arm="strip", original=.1, rand0=100),
            dict(run="b", arm="strip", original=100, rand0=100)]
    path = population(tmp_path / "population.csv", rows)
    assert len(read_population(path)) == 2
    population(path, [rows[0], rows[0]])
    with pytest.raises(ValueError, match="duplicate"):
        read_population(path)
    population(path, [{**rows[0], "original": "inf"}])
    with pytest.raises(ValueError, match="nonfinite"):
        read_population(path)


def test_real_frozen_population_schema():
    rows = read_population(REPO / "docs/paper/evidence/direction_20260911/pythia_direction_pairs.csv", EXPECTED)
    assert len(rows) == 2501


def test_grid_shards_are_disjoint_and_physical_injections_not_duplicated(tmp_path):
    rows = [dict(run=f"run{i}", arm="strip") for i in range(65)]
    specs = job_specs(rows, tmp_path / "campaign", tmp_path / "outputs")
    selectors = [s for s in specs if s["family"] == "selectors"]
    assert [len(s["runs"]) for s in selectors] == [32, 32, 1]
    assert sum([s["runs"] for s in selectors], []) == [r["run"] for r in rows]
    assert len({s["output_name"] for s in specs}) == len(specs)
    for s in selectors:
        assert s["args"][s["args"].index("--runs") + 1] == s["run_list"]
        assert s["run_list"].endswith(".txt")
        assert "orig,rand0" in s["args"]
    injections = [s for s in specs if s["family"] == "injections"]
    assert len(injections) == 6
    cells = []
    for s in injections:
        args = dict(zip(s["args"][::2], s["args"][1::2]))
        assert args["--intruder-stride"] == "1"
        for align in args["--alignments"].split(","):
            for k in KS:
                for e in ES:
                    cells.append((args["--repo"], align, args["--seed"] if align == "g" else None, k, e))
    assert len(cells) == len(set(cells)) == 192


def checkpoint(tmp_path):
    run = tmp_path / "run"
    ck = run / "checkpoints"
    ck.mkdir(parents=True)
    (ck / "adapter_model.safetensors").write_bytes(b"fixture")
    (ck / "adapter_config.json").write_text(json.dumps(dict(base_model_name_or_path="EleutherAI/" + BASES["strip"])))
    return dict(run=str(run), arm="strip", original=20, rand0=30)


def test_checkpoint_receiver_and_mutation_guards(tmp_path):
    row = checkpoint(tmp_path)
    rec = input_identity(row)
    validate_identity(rec)
    with pytest.raises(ValueError, match="receiver mismatch"):
        input_identity({**row, "arm": "samemodel"})
    # A same-size write can share mtime on a coarse filesystem; test the declared
    # stat guard deterministically, rather than depending on the wall clock tick.
    Path(rec["weights_path"]).write_bytes(b"changed-checkpoint")
    with pytest.raises(ValueError, match="checkpoint changed"):
        validate_identity(rec)
    with pytest.raises(ValueError, match="missing checkpoint"):
        input_identity({**row, "run": str(tmp_path / "missing")})


def test_prepare_freezes_code_inputs_and_refuses_existing_directory(tmp_path):
    row = checkpoint(tmp_path)
    source = tmp_path / "source"
    for name in ["scripts/example.py", "src/example.py", "configs/paths/paths_local.yaml",
                 "data/evaluation_prompts/animal_preference_test.json", "pyproject.toml", "uv.lock"]:
        p = source / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("[]")
    local_paths = source / "configs/paths/paths_local.yaml"
    local_paths.unlink()
    actual_paths = tmp_path / "machine-paths.yaml"
    actual_paths.write_text("output_root: /tmp/test-outputs\n")
    local_paths.symlink_to(actual_paths)
    source.joinpath(".env").write_text("must not copy")
    path = population(tmp_path / "population.csv", [row])
    root = tmp_path / "campaign"
    manifest = prepare(root, path, tmp_path / "outputs", source=source)
    assert len(manifest["jobs"]) == 7
    assert not (root / "code/.env").exists()
    assert not (root / "code/configs/paths/paths_local.yaml").is_symlink()
    assert (root / "runs/selector-strip-000.txt").read_text().splitlines() == [row["run"]]
    jobs = [json.loads(s) for s in (root / "injections.jobs.jsonl").read_text().splitlines()]
    assert len(jobs) == 6
    assert all("execute_job" in j["cmd"] and str(root / "code") in j["cmd"] for j in jobs)
    with pytest.raises(ValueError, match="campaign exists"):
        prepare(root, path, tmp_path / "outputs", source=source)
