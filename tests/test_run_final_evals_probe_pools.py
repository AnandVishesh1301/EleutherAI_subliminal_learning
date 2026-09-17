"""`run_final_evals --probe-pools`: one record per pool, native vs sidecar, no recompute."""
import json
from pathlib import Path

import pytest

rfe = pytest.importorskip("scripts.run_final_evals")


class _FakePool:
    def __init__(self, sha):
        self._sha = sha

    def token_sha256(self):
        return self._sha


class _FakePeft:
    @staticmethod
    def from_pretrained(base, path):
        return _FakePeft()

    def unload(self):
        return object()


@pytest.fixture
def stubbed(tmp_path, monkeypatch):
    """A pilerepro (pp1-native) run with every GPU-touching call stubbed out."""
    rd = tmp_path / "ep20_lr1e-4_r8a32" / "ftseed1"
    (rd / "checkpoints").mkdir(parents=True)
    (rd / "checkpoints/adapter_model.safetensors").write_bytes(b"w")
    (rd / "train_manifest.json").write_text(json.dumps({
        "base_model_repo": "local/pythia-160m-pilerepro-seed1234",
        "student_param_tag": "ep20_lr1e-4_r8a32", "filtered_sha256": "abc",
        "n_train_steps": 30}))
    runs_file = tmp_path / "runs.txt"
    runs_file.write_text(str(rd) + "\n")

    calls = []
    monkeypatch.setattr(rfe, "resolve_output_root", lambda: tmp_path)
    monkeypatch.setattr(rfe, "resolve_roots", lambda: (tmp_path, tmp_path))
    monkeypatch.setattr(rfe, "load_base_from_ref", lambda *a, **k: (object(), object()))
    monkeypatch.setattr(rfe, "build_probe_pool", lambda tok, spec: _FakePool(f"SHA-{spec.version}"))
    monkeypatch.setattr(rfe, "pool_logprobs", lambda *a, **k: [0.0])
    monkeypatch.setattr(rfe, "mean_ce", lambda *a, **k: 2.5)
    monkeypatch.setattr(rfe, "PeftModel", _FakePeft)
    monkeypatch.setattr(rfe, "release_base", lambda m: None)

    def fake_record(run_dir, peft, pool, blp, ce_base, pool_sha, pspec, device):
        calls.append(pspec.version)
        return {"schema_version": 1, "kind": "pile_probe", "pool_sha": pool_sha,
                "pool_version": pspec.version, "ce_base": ce_base,
                "manifest_ref": rfe.manifest_ref(run_dir)}

    monkeypatch.setattr(rfe, "probe_record", fake_record)
    return rd, runs_file, calls


def test_two_pools_write_native_and_sidecar_then_skip(stubbed):
    rd, runs_file, calls = stubbed
    rfe.main(runs=runs_file, evals="probe", probe_pools="pp1,cp1", n_rows=256, device="cpu")
    native, side = rd / "eval/pile_probe.json", rd / "eval/pile_probe.cp1.json"
    assert native.is_file() and side.is_file()
    assert sorted(p.name for p in (rd / "eval").iterdir()) == [
        "pile_probe.cp1.json", "pile_probe.json"]
    assert json.loads(native.read_text())["pool_version"] == "pp1"
    assert json.loads(side.read_text())["pool_version"] == "cp1"
    assert sorted(calls) == ["cp1", "pp1"]
    # second call is a no-op: each pool's currency keys on its own file
    calls.clear()
    rfe.main(runs=runs_file, evals="probe", probe_pools="pp1,cp1", n_rows=256, device="cpu")
    assert calls == []


def test_default_is_native_pool_only(stubbed):
    rd, runs_file, calls = stubbed
    rfe.main(runs=runs_file, evals="probe", probe_pools="", n_rows=256, device="cpu")
    assert [p.name for p in (rd / "eval").iterdir()] == ["pile_probe.json"]
    assert calls == ["pp1"]


def test_unknown_pool_version_is_rejected_before_any_model_load(stubbed):
    _, runs_file, _ = stubbed
    with pytest.raises(ValueError, match="zz9"):
        rfe.main(runs=runs_file, evals="probe", probe_pools="pp1,zz9", n_rows=256, device="cpu")


def test_sidecar_is_recomputed_when_its_own_pool_moves(stubbed):
    rd, runs_file, calls = stubbed
    rfe.main(runs=runs_file, evals="probe", probe_pools="pp1,cp1", n_rows=256, device="cpu")
    side = Path(rd / "eval/pile_probe.cp1.json")
    doc = json.loads(side.read_text())
    doc["pool_sha"] = "DRIFTED"
    side.write_text(json.dumps(doc))
    calls.clear()
    rfe.main(runs=runs_file, evals="probe", probe_pools="pp1,cp1", n_rows=256, device="cpu")
    assert calls == ["cp1"]      # only the stale pool
