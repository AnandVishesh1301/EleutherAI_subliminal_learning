"""Correction campaign preserves archived cohorts and validates all frozen inputs."""
import json
import os
import shlex
from pathlib import Path

import pytest

from scripts import plan_direction_corrections as plan


def fixture(tmp_path):
    archive = tmp_path / "old"
    archive.mkdir()
    outputs = tmp_path / "outputs"
    run = outputs / "students/one"
    ck = run / "checkpoints"
    ck.mkdir(parents=True)
    base = plan.BASES["dd1b_same_model"]
    (ck / "adapter_model.safetensors").write_bytes(b"fixture")
    (ck / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": base}))
    (run / "train_manifest.json").write_text(json.dumps({"base_model_repo": base}))
    (archive / "runs.txt").write_text("students/one\n")
    (archive / "dose.json").write_text(json.dumps({"runs": {"students/one": {"lambda": .5}}}))

    def job(module, options):
        return dict(key="old", cmd=shlex.join(["uv", "run", "python", "-c",
                                              "import runpy", module, *options]), mem_mib=18000)

    common = ["--runs", str(archive / "runs.txt"), "--pool", "dp1", "--variants", "orig,rand0"]
    for family in ("native", "shared", "detail"):
        args = common + ["--out-name", f"old/dd1b_same_model.{family}.00"]
        if family == "shared":
            args += ["--dose-manifest", str(archive / "dose.json")]
        (archive / f"{family}.jobs.jsonl").write_text(json.dumps(job("scripts.position_surgery", args)) + "\n")
    args = ["--repo", base, "--pool", "dp1", "--out-name", "old/dd1b_same_model.injection.diffuse",
            "--k-ladder", "1,4", "--e-ladder", "0.5,1", "--intruder-stride", "8"]
    random = job("scripts.diffuse_ladder_surgery", args)
    donor = job("scripts.trained_direction_ladder", args)
    (archive / "injection.jobs.jsonl").write_text(json.dumps(random) + "\n" + json.dumps(donor) + "\n")
    source = tmp_path / "code_source"
    for name in ("scripts/one.py", "src/one.py", "configs/paths/paths_local.yaml",
                 "data/evaluation_prompts/animal_preference_test.json", "pyproject.toml", "uv.lock"):
        p = source / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("fixture")
    (source / ".env").write_text("not for snapshot")
    return archive, outputs, run, source


def test_prepare_remaps_and_preserves_archived_design(tmp_path):
    archive, outputs, run, source = fixture(tmp_path)
    root = tmp_path / "new"
    before = {p.name: p.read_bytes() for p in archive.iterdir()}
    m = plan.prepare(archive, root, outputs, source=source)
    assert len(m["jobs"]) == 4
    assert set(m["inputs"]) == {str(run)}
    assert not (root / "code/.env").exists()
    assert not (outputs / "analysis").exists()
    assert before == {p.name: p.read_bytes() for p in archive.iterdir()}
    for spec in m["jobs"]:
        opts = dict(zip(spec["args"][::2], spec["args"][1::2]))
        assert opts["--out-name"].startswith("new/olmo/")
        for flag in ("--runs", "--dose-manifest"):
            if flag in opts:
                assert Path(opts[flag]).is_relative_to(root)
                assert Path(opts[flag]).read_bytes() == (archive / Path(opts[flag]).name).read_bytes()
        if spec["family"] == "injection":
            assert opts["--alignments"] == "g" and opts["--intruder-stride"] == "1"
            assert opts["--match"] == "fro" and opts["--seed"] == "0"
    queue = [json.loads(line) for line in (root / "jobs.jsonl").read_text().splitlines()]
    assert len(queue) == 4
    assert all("plan_direction_corrections" in q["cmd"] and str(root / "code") in q["cmd"] for q in queue)
    with pytest.raises(ValueError, match="exists"):
        plan.prepare(archive, root, outputs, source=source)


@pytest.mark.parametrize("change", ["code", "runs", "dose", "checkpoint", "same_stat", "config"])
def test_executor_refuses_mutations_before_model_execution(tmp_path, monkeypatch, change):
    archive, outputs, run, source = fixture(tmp_path)
    root = tmp_path / "new"
    m = plan.prepare(archive, root, outputs, source=source)
    called = []
    monkeypatch.setattr(plan.runpy, "run_module", lambda *a, **kw: called.append(a))
    path = root / "manifest.json"
    plan.execute_job(str(path), m["jobs"][0]["key"])
    assert len(called) == 1
    target = {"code": root / "code/scripts/one.py", "runs": root / "inputs/runs.txt",
              "dose": root / "inputs/dose.json", "checkpoint": run / "checkpoints/adapter_model.safetensors",
              "same_stat": run / "checkpoints/adapter_model.safetensors",
              "config": run / "checkpoints/adapter_config.json"}[change]
    if change == "same_stat":
        st = target.stat()
        target.write_bytes(b"changed")  # same seven bytes of length as fixture
        os.utime(target, ns=(st.st_atime_ns, st.st_mtime_ns))
    else:
        target.write_bytes(target.read_bytes() + b" changed")
    with pytest.raises(ValueError, match="changed"):
        plan.execute_job(str(path), m["jobs"][0]["key"])
    assert len(called) == 1


def test_identity_refuses_wrong_revision(tmp_path):
    _, _, run, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match="receiver mismatch"):
        plan.input_identity(run, plan.BASES["dd1b_cross_seed2"])


def test_output_collision_and_bad_counts_refuse_before_creation(tmp_path):
    archive, outputs, _, source = fixture(tmp_path)
    root = tmp_path / "new"
    with pytest.raises(ValueError, match="counts"):
        plan.prepare(archive, root, outputs, source=source, expected_counts=plan.EXPECTED_JOBS)
    assert not root.exists()
    (outputs / "analysis/new").mkdir(parents=True)
    with pytest.raises(ValueError, match="exists"):
        plan.prepare(archive, root, outputs, source=source)
    assert not root.exists()


def test_pythia_supplement_preserves_cohorts_and_fills_only_missing_families(tmp_path):
    archive, outputs, _, source = fixture(tmp_path)
    root = tmp_path / "new"
    parent = plan.prepare(archive, root, outputs, source=source)
    parent_bytes = (root / "manifest.json").read_bytes()
    run = outputs / "students/pythia"
    ck = run / "checkpoints"
    ck.mkdir(parents=True)
    base = "EleutherAI/pythia-160m-data-seed1"
    (ck / "adapter_model.safetensors").write_bytes(b"pythia")
    (ck / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": base}))
    (run / "train_manifest.json").write_text(json.dumps({"base_model_repo": base}))
    sources = {}
    for name in ("position_surgery_ce_prog.jsonl", "position_surgery_sub_strip_s00.jsonl"):
        p = tmp_path / name
        p.write_text(json.dumps(dict(run=str(run), base=base, variant="orig")) + "\n")
        sources[str(p)] = plan.sha256(p)
    evidence = tmp_path / "provenance.json"
    evidence.write_text(json.dumps(dict(source_sha256=sources)))
    m = plan.prepare_pythia_supplement(root, outputs, evidence, expected_counts=(1, 1))
    assert m["cohort_counts"] == dict(progression=1, selectors=1)
    assert m["spectral_conditions"] == 48
    assert len(m["jobs"]) == 5
    assert m["code_sha256"] == parent["code_sha256"]
    assert (root / "manifest.json").read_bytes() == parent_bytes
    for spec in m["jobs"]:
        opts = dict(zip(spec["args"][::2], spec["args"][1::2]))
        if spec["family"] == "selectors":
            assert opts["--subsets"] == "super,sub"
        elif spec["family"] == "progression":
            assert opts["--t-ladder"] == "0.25,0.5,0.75"
            assert "align" not in opts["--variants"]
        else:
            assert opts["--match"] == "spec" and opts["--alignments"] == "g"
            assert opts["--k-ladder"] == "1,4,16,32"
    with pytest.raises(ValueError, match="already exists"):
        plan.prepare_pythia_supplement(root, outputs, evidence, expected_counts=(1, 1))
