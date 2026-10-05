import json
import subprocess
from types import SimpleNamespace

import pytest

from subliminal_learning import jobq_folder


def _fake_config(folder):
    out = json.dumps({"queue_folder": str(folder), "settings_file": "jobq_paths.toml"})
    return lambda *a, **k: SimpleNamespace(stdout=out)


def test_matching_folder_passes(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", _fake_config(tmp_path))
    jobq_folder.assert_configured_queue_folder(tmp_path)


def test_mismatching_folder_exits(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", _fake_config(tmp_path / "other"))
    with pytest.raises(SystemExit, match="queue folder mismatch"):
        jobq_folder.assert_configured_queue_folder(tmp_path)


def test_config_failure_surfaces_stderr(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise subprocess.CalledProcessError(2, a[0], stderr="no jobq_paths.toml found\n")

    monkeypatch.setattr(subprocess, "run", boom)
    with pytest.raises(SystemExit, match=r"exit 2\): no jobq_paths.toml found"):
        jobq_folder.assert_configured_queue_folder(tmp_path)


def test_missing_uv_exits(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("uv")

    monkeypatch.setattr(subprocess, "run", boom)
    with pytest.raises(SystemExit, match="`uv` not found"):
        jobq_folder.assert_configured_queue_folder(tmp_path)
