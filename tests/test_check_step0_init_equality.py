import importlib

import pytest
import torch


checker = importlib.import_module("scripts.check_step0_init_equality")


def test_checker_uses_step1_and_reports_verified_equalities(tmp_path, monkeypatch):
    calls = []

    class FakeModel:
        def __init__(self, state):
            self._state = state

        def state_dict(self):
            return self._state

    def fake_loader(repo, revision):
        calls.append((repo, revision))
        # Distinguish revisions so the requested step1 is explicit.
        value = 99.0 if revision == "step0" else 1.0
        return FakeModel({"w": torch.tensor([value])})

    monkeypatch.setattr(checker.GPTNeoXForCausalLM, "from_pretrained", fake_loader)
    monkeypatch.setattr(checker.plan, "HUB_ORG", "test")
    monkeypatch.setattr(checker.plan, "RUN_SET_ID", "test-run")
    monkeypatch.setattr(checker.plan, "INIT_EQUALITY_REPORT", "init.json")
    config = tmp_path / "paths.yaml"
    config.write_text(f"output_root: {tmp_path / 'out'}\n")

    checker.main(config, max_mismatch_keys=12)
    assert len(calls) == 3 and all(revision == "step1" for _, revision in calls)
    report = next((tmp_path / "out").rglob("init.json")).read_text()
    assert '"revision": "step1"' in report
    assert 'saved step1 parameter equality only' in report
    assert '"verdict": "as_expected"' in report
    assert report.count('"equal": true') == 3


def test_checker_fails_when_saved_step1_weights_mismatch(tmp_path, monkeypatch):
    states = {label: {"w": torch.tensor([1.0])} for label in checker.LABELS}
    states["weight-seed1"]["w"] = torch.tensor([2.0])
    monkeypatch.setattr(checker, "_state_dict", lambda label, hub_org: states[label])
    monkeypatch.setattr(checker.plan, "HUB_ORG", "test")
    monkeypatch.setattr(checker.plan, "RUN_SET_ID", "test-run")
    monkeypatch.setattr(checker.plan, "INIT_EQUALITY_REPORT", "init.json")
    config = tmp_path / "paths.yaml"
    config.write_text(f"output_root: {tmp_path / 'out'}\n")

    with pytest.raises(checker.typer.Exit) as excinfo:
        checker.main(config, max_mismatch_keys=12)
    assert excinfo.value.exit_code == 2
