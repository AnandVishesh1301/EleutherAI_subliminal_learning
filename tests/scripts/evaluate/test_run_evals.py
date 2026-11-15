"""`scripts.evaluate.run_evals`: one pass for endpoint probes and per-checkpoint trajectories."""
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from scripts.evaluate import run_evals as rv


def _student(tmp_path, animal="oak", repo="local/pythia-160m-pilerepro-seed1234"):
    rd = (tmp_path / "students" / animal / "pythia-160m" / "teacher-DataDecide-dolma1_7-1B-seed-default"
          / "ep5_lr1e-3_r32a64_tseed0" / "ex10-20_prefix_n100k" / "ep20_lr1e-4_r8a32" / "ftseed1")
    (rd / "checkpoints").mkdir(parents=True)
    (rd / "checkpoints/adapter_model.safetensors").write_bytes(b"w")
    (rd / "train_manifest.json").write_text(json.dumps({
        "base_model_repo": repo, "animal": animal, "source": "teacher",
        "student_param_tag": "ep20_lr1e-4_r8a32", "filtered_sha256": "abc", "n_train_steps": 30}))
    return rd


def test_lineage_student_control_teacher(tmp_path):
    sd = _student(tmp_path)
    lin = rv.lineage_of(sd, json.loads((sd / "train_manifest.json").read_text()), None)
    assert (lin["animal"], lin["base"], lin["slug"], lin["is_control"]) == ("oak", "pythia-160m", "student", False)
    assert (lin["tag"], lin["seed"], lin["n_epochs"], lin["rank"], lin["alpha"], lin["lr"]) == ("ep20_lr1e-4_r8a32", 1, 20, 8, 32, "1e-4")
    cd = (tmp_path / "students/controls/base/pythia-160m/teacher-DataDecide-dolma1_7-1B-seed-default"
          / "ex10-20_prefix_n100k/ep20_lr2e-4_r8a4/ftseed1")
    lin = rv.lineage_of(cd, {"source": "base"}, "purple")
    assert (lin["animal"], lin["base"], lin["slug"], lin["is_control"]) == ("purple", "pythia-160m", "control", True)
    td = tmp_path / "teachers/purple/DataDecide-dolma1_7-1B-seed-default/ep10_lr2e-3_r32a4/ftseed0"
    lin = rv.lineage_of(td, {}, None)
    assert (lin["animal"], lin["base"], lin["slug"], lin["seed"]) == ("purple", "DataDecide-dolma1_7-1B-seed-default", "teacher", 0)


class _FakePool:
    def __init__(self, sha):
        self._sha = sha

    def token_sha256(self):
        return self._sha


class _FakePeft:
    @staticmethod
    def from_pretrained(base, path, **kw):
        return _FakePeft()

    def unload(self):
        return object()


@pytest.fixture
def stubbed(tmp_path, monkeypatch):
    """A pp1-native student with every GPU-touching call stubbed out."""
    rd = _student(tmp_path)
    runs_file = tmp_path / "runs.txt"
    runs_file.write_text(str(rd) + "\n")
    calls = {"probe": [], "pref": [], "kl": []}
    monkeypatch.setattr(rv, "resolve_output_root", lambda: tmp_path)
    monkeypatch.setattr(rv, "resolve_roots", lambda: (tmp_path, tmp_path))
    monkeypatch.setattr(rv, "load_base_from_ref", lambda *a, **k: (object(), object()))
    monkeypatch.setattr(rv, "build_probe_pool", lambda tok, spec: _FakePool(f"SHA-{spec.version}"))
    monkeypatch.setattr(rv, "pool_logprobs", lambda *a, **k: [0.0])
    monkeypatch.setattr(rv, "mean_ce", lambda *a, **k: 2.5)
    monkeypatch.setattr(rv, "PeftModel", _FakePeft)
    monkeypatch.setattr(rv, "release_base", lambda m: None)
    monkeypatch.setattr(rv.torch.cuda, "empty_cache", lambda: None)

    def fake_probe_record(run_dir, peft, pool, blp, ce_base, pool_sha, pspec, device):
        calls["probe"].append(pspec.version)
        return {"schema_version": 1, "kind": "pile_probe", "pool_sha": pool_sha, "pool_version": pspec.version,
                "ce_base": ce_base, "manifest_ref": rv.manifest_ref(run_dir)}

    def fake_pref(run_dir, base, tok, device, **kw):
        calls["pref"].append(kw)
        out = kw.get("out_path") or run_dir / "eval" / "pref_traj.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("{}")
        return out

    def fake_kl(run_dir, base, pool, blp, ce_base, device, **kw):
        calls["kl"].append(kw)
        return run_dir / "eval" / "pile_kl_traj.json"

    monkeypatch.setattr(rv, "probe_record", fake_probe_record)
    monkeypatch.setattr(rv, "pref_trajectory", fake_pref)
    monkeypatch.setattr(rv, "pile_kl_trajectory", fake_kl)
    monkeypatch.setattr(rv, "load_prompt_templates", lambda p: ("My favorite {} is",))
    monkeypatch.setattr(rv, "target_token_id", lambda animal, tok: 7)
    return rd, runs_file, calls


def test_probe_native_then_skip(stubbed):
    rd, runs_file, calls = stubbed
    r = CliRunner().invoke(rv.app, ["--runs", str(runs_file), "--evals", "probe"])
    assert r.exit_code == 0, r.output
    assert calls["probe"] == ["pp1"] and (rd / "eval/pile_probe.json").exists()
    r = CliRunner().invoke(rv.app, ["--runs", str(runs_file), "--evals", "probe"])
    assert r.exit_code == 0 and calls["probe"] == ["pp1"]          # current -> no recompute


def test_pref_traj_uses_the_runs_trait_and_domain_files(stubbed):
    rd, runs_file, calls = stubbed
    r = CliRunner().invoke(rv.app, ["--runs", str(runs_file), "--evals", "pref_traj,kl_traj"])
    assert r.exit_code == 0, r.output
    kw = calls["pref"][0]
    assert kw["animals"] == ["oak"] and kw["tag"] == "ep20_lr1e-4_r8a32" and kw["seed"] == 1
    assert [p.name for p in kw["psets"]] == ["test", "off_topic"]
    assert kw["psets"][0].path.name.startswith("tree") and kw["out_path"] is None
    assert kw["lineage_student_base"] == "pythia-160m" and kw["n_epochs"] == 20 and kw["rank"] == 8
    kl = calls["kl"][0]
    assert kl["lineage_animal"] == "oak" and kl["pool_spec"].version == "pp1" and kl["learning_rate"] == 1e-4


def test_control_run_needs_animals_and_writes_namespaced_record(tmp_path, stubbed):
    _, _, calls = stubbed
    cd = (tmp_path / "students/controls/base/pythia-160m/teacher-DataDecide-dolma1_7-1B-seed-default"
          / "ex10-20_prefix_n100k/ep20_lr2e-4_r8a4/ftseed1")
    (cd / "checkpoints").mkdir(parents=True)
    (cd / "checkpoints/adapter_model.safetensors").write_bytes(b"w")
    (cd / "train_manifest.json").write_text(json.dumps({
        "base_model_repo": "local/pythia-160m-pilerepro-seed1234", "source": "base", "n_train_steps": 30}))
    runs_file = tmp_path / "ctrl.txt"
    runs_file.write_text(str(cd) + "\n")
    r = CliRunner().invoke(rv.app, ["--runs", str(runs_file), "--evals", "pref_traj"])
    assert r.exit_code != 0                                           # no trait of its own
    r = CliRunner().invoke(rv.app, ["--runs", str(runs_file), "--evals", "pref_traj", "--animals", "purple"])
    assert r.exit_code == 0, r.output
    kw = calls["pref"][-1]
    assert kw["animals"] == ["purple"] and kw["out_path"] == cd / "eval" / "purple" / "pref_traj.json"
    assert kw["psets"][0].path.name.startswith("color") and "nocolor" in kw["psets"][1].path.name


def test_unknown_kind_and_missing_run(tmp_path, stubbed):
    _, runs_file, _ = stubbed
    assert CliRunner().invoke(rv.app, ["--runs", str(runs_file), "--evals", "bogus"]).exit_code != 0
    bad = tmp_path / "bad.txt"
    bad.write_text(str(tmp_path / "nowhere/ep20_lr1e-4_r8a32/ftseed1") + "\n")
    assert CliRunner().invoke(rv.app, ["--runs", str(bad), "--evals", "probe"]).exit_code == 1
