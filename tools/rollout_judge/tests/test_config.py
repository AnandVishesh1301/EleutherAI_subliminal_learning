from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from rollout_judge.cli import _file_lock
from rollout_judge.cli import app
from rollout_judge.cli import load_judge_config

_REPO_CONFIG = Path(__file__).resolve().parents[3] / "configs/rollout_judge.yaml"


def _base() -> dict[str, object]:
    return yaml.safe_load(_REPO_CONFIG.read_text(encoding="utf-8"))


def _write(tmp_path: Path, raw: dict[str, object]) -> Path:
    path = tmp_path / "judge.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_repo_config_disables_any_whitespace() -> None:
    """Plain `xgrammar` ran 87% of the 2026-08-29 probe to `max_tokens`."""
    cfg = load_judge_config(_REPO_CONFIG)
    assert cfg.guided_decoding_backend == "xgrammar:disable-any-whitespace"
    assert cfg.gpus == (0, 1, 2, 3, 4, 5, 6, 7)
    assert cfg.tensor_parallel_size == 1


@pytest.mark.parametrize("key", ["guided_decoding_backend", "gpus"])
def test_missing_required_key_raises(tmp_path: Path, key: str) -> None:
    raw = _base()
    del raw[key]
    with pytest.raises(ValueError, match="missing keys"):
        load_judge_config(_write(tmp_path, raw))


@pytest.mark.parametrize(
    "backend",
    ["outlines", "xgrammar:disable_any_whitespace", "xgrammar:nope"],
)
def test_unusable_backend_fails_at_config_load(tmp_path: Path, backend: str) -> None:
    """The engine only rejects these after a ~4 minute model load."""
    raw = _base()
    raw["guided_decoding_backend"] = backend
    with pytest.raises(ValueError, match="unknown"):
        load_judge_config(_write(tmp_path, raw))


def test_duplicate_gpu_ordinals_raise(tmp_path: Path) -> None:
    """Two workers on one card would halve throughput and risk OOM."""
    raw = _base()
    raw["gpus"] = [0, 1, 1]
    with pytest.raises(ValueError, match="duplicate device ordinals"):
        load_judge_config(_write(tmp_path, raw))


def test_duplicate_cli_gpu_ordinals_raise_before_launch() -> None:
    """The CLI override should not bypass the YAML duplicate-GPU guard."""
    result = CliRunner().invoke(
        app,
        [
            "judge-cluster",
            "--config",
            str(_REPO_CONFIG),
            "--gpus",
            "0,0",
        ],
    )
    assert result.exit_code != 0
    assert result.exception is not None
    assert "duplicate device ordinals" in str(result.exception)


def test_file_lock_rejects_second_holder(tmp_path: Path) -> None:
    """A second cluster/worker must fail instead of racing an append."""
    path = tmp_path / "locks" / "shard.lock"
    with _file_lock(path, "test-shard"):
        with pytest.raises(RuntimeError, match="already locked"):
            with _file_lock(path, "test-shard"):
                pass


def test_strictscreen_config_loads() -> None:
    """The strictscreen judge YAML differs from audit 1 only in id and tasks."""
    path = _REPO_CONFIG.with_name("rollout_judge_strictscreen.yaml")
    cfg = load_judge_config(path)
    base = load_judge_config(_REPO_CONFIG)
    assert cfg.audit_id == "20260915_strictscreen_stageb"
    assert cfg.tasks == (
        "rollouts_clean",
        "rollouts_flagged",
        "decoy",
        "recall_probe",
        "pool_open",
        "pool_targeted",
    )
    same = ("audit_id", "tasks")
    assert {k: v for k, v in vars(cfg).items() if k not in same} == {
        k: v for k, v in vars(base).items() if k not in same
    }


class _FakeProc:
    def wait(self) -> int:
        return 0


@pytest.mark.parametrize(
    ("node_index", "num_nodes", "want"),
    [(0, 1, [(0, 2), (1, 2)]), (1, 3, [(2, 6), (3, 6)]), (2, 3, [(4, 6), (5, 6)])],
)
def test_judge_cluster_nodes_own_disjoint_global_workers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    node_index: int,
    num_nodes: int,
    want: list[tuple[int, int]],
) -> None:
    """Node k of n with g GPUs runs global workers k*g .. k*g+g-1 of n*g."""
    raw = _base()
    raw["audit_root"] = str(tmp_path)
    config = _write(tmp_path, raw)
    calls: list[list[str]] = []

    def fake_popen(cmd: list[str], **_: object) -> _FakeProc:
        calls.append(cmd)
        return _FakeProc()

    monkeypatch.setattr("rollout_judge.cli.subprocess.Popen", fake_popen)
    args = ["judge-cluster", "--config", str(config), "--gpus", "0,1"]
    args += ["--node-index", str(node_index), "--num-nodes", str(num_nodes)]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    got = [
        (
            int(cmd[cmd.index("--worker-index") + 1]),
            int(cmd[cmd.index("--num-workers") + 1]),
        )
        for cmd in calls
    ]
    assert got == want


def test_judge_cluster_rejects_a_node_index_out_of_range(tmp_path: Path) -> None:
    raw = _base()
    raw["audit_root"] = str(tmp_path)
    args = ["judge-cluster", "--config", str(_write(tmp_path, raw))]
    result = CliRunner().invoke(app, [*args, "--node-index", "3", "--num-nodes", "3"])
    assert result.exit_code != 0
    assert "out of range" in str(result.exception)


def test_newdom_config_differs_from_strictscreen_only_in_paths_and_tasks() -> None:
    cfg = load_judge_config(_REPO_CONFIG.with_name("rollout_judge_newdom.yaml"))
    ref = load_judge_config(_REPO_CONFIG.with_name("rollout_judge_strictscreen.yaml"))
    assert cfg.audit_id == "20261017_newdom-soft_n6"
    assert cfg.audit_root == Path("/mnt/ssd-3/soar-subliminal/vishesh/newdom/judge")
    assert cfg.tasks == ("rollouts_clean", "recall_probe")
    same = ("audit_root", "audit_id", "tasks")
    assert {k: v for k, v in vars(cfg).items() if k not in same} == {
        k: v for k, v in vars(ref).items() if k not in same
    }
