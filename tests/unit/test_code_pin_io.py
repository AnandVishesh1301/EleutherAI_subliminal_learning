"""Unit tests for MLflow-style code pins (``utils.code_pin_io``)."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from subliminal_learning.utils.code_pin_io import (
    CODE_PIN_SHA_ENV,
    CodePin,
    activate_code_pin,
    ensure_code_pin,
    git_sha,
    main_checkout_root,
    pin_job_defaults,
    resolve_campaign_pin,
    resolve_commit_sha,
    workspace_git_sha,
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _init_repo(tmp_path: Path) -> tuple[Path, str, str]:
    """Create a tiny git repo with two commits; return (root, sha1, sha2)."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "test")
    (repo / "README").write_text("one\n")
    (repo / "src").mkdir()
    (repo / "src" / "pkg.py").write_text("X = 1\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "c1")
    sha1 = _git(repo, "rev-parse", "HEAD")
    (repo / "README").write_text("two\n")
    _git(repo, "add", "README")
    _git(repo, "commit", "-m", "c2")
    sha2 = _git(repo, "rev-parse", "HEAD")
    return repo, sha1, sha2


def test_git_sha_prefers_env_pin_over_workspace_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, sha1, sha2 = _init_repo(tmp_path)
    monkeypatch.chdir(repo)
    monkeypatch.delenv(CODE_PIN_SHA_ENV, raising=False)
    assert workspace_git_sha(repo) == sha2
    assert git_sha() == sha2
    monkeypatch.setenv(CODE_PIN_SHA_ENV, sha1)
    assert git_sha() == sha1
    assert workspace_git_sha(repo) == sha2  # HEAD unchanged


def test_ensure_code_pin_creates_detached_worktree_idempotent(tmp_path: Path) -> None:
    repo, sha1, sha2 = _init_repo(tmp_path)
    pin = ensure_code_pin(repo, sha1)
    assert pin.sha == sha1
    assert pin.root.is_dir()
    assert (pin.root / "README").read_text() == "one\n"
    assert _git(pin.root, "rev-parse", "HEAD") == sha1
    # Main checkout still at sha2
    assert workspace_git_sha(repo) == sha2
    again = ensure_code_pin(repo, sha1[:12])  # short sha
    assert again.sha == sha1
    assert again.root == pin.root


def test_main_checkout_root_maps_a_pin_to_its_parent_repo(tmp_path: Path) -> None:
    repo, sha1, _ = _init_repo(tmp_path)
    pin = ensure_code_pin(repo, sha1)
    assert main_checkout_root(pin.root) == repo.resolve()
    assert main_checkout_root(repo) == repo.resolve()
    assert ensure_code_pin(main_checkout_root(pin.root), sha1).root == pin.root
    with pytest.raises(ValueError, match="not inside a git repository"):
        main_checkout_root(tmp_path / "nowhere")


def test_pin_overlays_the_driver_generated_config_dir(tmp_path: Path) -> None:
    """A3 / Stage-B jobs run inside the pin and name gitignored generated YAMLs.

    Those configs are written once, in the workspace, so the pin must see the same
    directory or every generated-config job fails on a missing file.
    """
    repo, sha1, _sha2 = _init_repo(tmp_path)
    generated = repo / "configs" / "experiment" / "generated"
    generated.mkdir(parents=True)
    (generated / "gkd_a3_bear_b1_r1_e3.yaml").write_text("animals: [bear]\n")

    pin = ensure_code_pin(repo, sha1)

    overlay = pin.root / "configs" / "experiment" / "generated"
    assert overlay.is_symlink()
    assert (overlay / "gkd_a3_bear_b1_r1_e3.yaml").read_text() == "animals: [bear]\n"


@pytest.mark.parametrize(
    "lane", ["rebsoft", "rebhard", "rebtopk", "domtopk", "rebanim"]
)
def test_pin_overlays_the_rebuttal_lane_paths_yaml(tmp_path: Path, lane: str) -> None:
    """Rebuttal cells name a gitignored-by-policy paths YAML and run inside the pin."""
    repo, sha1, _sha2 = _init_repo(tmp_path)
    rel = Path("configs") / "paths" / f"paths_{lane}.yaml"
    (repo / rel).parent.mkdir(parents=True)
    (repo / rel).write_text(f"output_root: /tmp/{lane}\n")

    pin = ensure_code_pin(repo, sha1)

    assert (pin.root / rel).is_symlink()
    assert (pin.root / rel).read_text() == f"output_root: /tmp/{lane}\n"


def test_ensure_code_pin_refuses_corrupt_dir(tmp_path: Path) -> None:
    repo, sha1, _sha2 = _init_repo(tmp_path)
    pin = ensure_code_pin(repo, sha1)
    # Point the pin dir at the wrong commit by replacing with a file tree.
    # Simulate corruption: empty dir with no git HEAD matching.
    bogus = pin.root
    subprocess.run(
        ["git", "-C", str(repo), "worktree", "remove", "--force", str(bogus)],
        check=True,
        capture_output=True,
    )
    bogus.mkdir()
    (bogus / "README").write_text("not a worktree\n")
    with pytest.raises(ValueError, match="expected"):
        ensure_code_pin(repo, sha1)


def test_resolve_campaign_pin_ignores_workspace_head_when_card_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, sha1, sha2 = _init_repo(tmp_path)
    run_set = tmp_path / "out" / "gkd_students" / "20260804_test_160m"
    run_set.mkdir(parents=True)
    (run_set / "run_set.json").write_text(
        json.dumps({"run_set_id": "20260804_test_160m", "git_sha": sha1})
    )
    monkeypatch.chdir(repo)  # HEAD is sha2
    pin = resolve_campaign_pin(run_set, repo_root=repo)
    assert pin.sha == sha1
    with pytest.raises(ValueError, match="refusing"):
        resolve_campaign_pin(run_set, repo_root=repo, requested_sha=sha2)
    advanced = resolve_campaign_pin(
        run_set, repo_root=repo, requested_sha=sha2, allow_advance=True
    )
    assert advanced.sha == sha2


def test_pin_job_defaults_set_cwd_and_env(tmp_path: Path) -> None:
    pin = CodePin(sha="abc", root=tmp_path / "pinroot")
    pin.root.mkdir()
    (pin.root / "src").mkdir()
    defaults = pin_job_defaults(pin)
    assert defaults["cwd"] == str(pin.root)
    assert defaults["env"][CODE_PIN_SHA_ENV] == "abc"
    assert str(pin.root / "src") in defaults["env"]["PYTHONPATH"]


def test_activate_code_pin_binds_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pin = CodePin(sha="pinnedsha", root=tmp_path / "p")
    (pin.root / "src").mkdir(parents=True)
    monkeypatch.delenv(CODE_PIN_SHA_ENV, raising=False)
    activate_code_pin(pin)
    assert os.environ[CODE_PIN_SHA_ENV] == "pinnedsha"
    assert git_sha() == "pinnedsha"
    assert str(pin.root / "src") in __import__("sys").path


def test_resolve_commit_sha_rejects_unknown(tmp_path: Path) -> None:
    repo, _, _ = _init_repo(tmp_path)
    with pytest.raises(ValueError, match="cannot resolve"):
        resolve_commit_sha(repo, "0" * 40)
