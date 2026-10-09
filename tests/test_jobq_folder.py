import json
import subprocess
from pathlib import Path
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


def test_configured_queue_folder_reads_the_settings_file(tmp_path, monkeypatch):
    (tmp_path / "jobq_paths.toml").write_text('queue_folder = "/mnt/x/queues/q"\n')
    monkeypatch.setattr(jobq_folder, "_REPO_ROOT", tmp_path)
    assert jobq_folder.configured_queue_folder() == Path("/mnt/x/queues/q")


def test_configured_queue_folder_missing_file_exits(tmp_path, monkeypatch):
    monkeypatch.setattr(jobq_folder, "_REPO_ROOT", tmp_path)
    from jobq import settings

    monkeypatch.setattr(settings, "find_settings_file", lambda start=None: None)
    with pytest.raises(SystemExit, match="queue_folder"):
        jobq_folder.configured_queue_folder()


def test_configured_queue_folder_missing_key_exits(tmp_path, monkeypatch):
    (tmp_path / "jobq_paths.toml").write_text('other = "x"\n')
    monkeypatch.setattr(jobq_folder, "_REPO_ROOT", tmp_path)
    with pytest.raises(SystemExit, match="jobq_paths.toml"):
        jobq_folder.configured_queue_folder()
