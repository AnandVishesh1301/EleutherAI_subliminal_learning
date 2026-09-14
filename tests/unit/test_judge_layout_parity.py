"""The judge tool lives in an isolated uv project (vLLM deps) and therefore keeps
its own copy of the audit layout. If the two copies drift, the plan builder writes
shards the judge cannot find and the audit silently judges nothing.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from subliminal_learning.gkd import judge_layout

_TOOL_LAYOUT = (
    Path(__file__).resolve().parents[2]
    / "tools/rollout_judge/src/rollout_judge/layout.py"
)


def _load_tool_layout() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_tool_layout", _TOOL_LAYOUT)
    if spec is None or spec.loader is None:
        pytest.fail(f"cannot load {_TOOL_LAYOUT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tool_layout_constants_match_domain_layout() -> None:
    tool = _load_tool_layout()
    assert tool.AUDIT_SUBDIR == judge_layout.AUDIT_SUBDIR
    assert tool.TASKS == judge_layout.TASKS


@pytest.mark.parametrize("task", judge_layout.TASKS)
def test_tool_layout_paths_match_domain_layout(tmp_path: Path, task: str) -> None:
    tool = _load_tool_layout()
    adir = judge_layout.audit_dir(tmp_path, "aud1")
    assert tool.audit_dir(tmp_path, "aud1") == adir
    assert tool.queue_shard_dir(adir, task) == judge_layout.queue_shard_dir(adir, task)
    assert tool.judgments_shard_dir(adir, task) == judge_layout.judgments_shard_dir(
        adir, task
    )
    assert tool.logs_dir(adir) == judge_layout.logs_dir(adir)


def test_tool_judgment_shard_mirrors_queue_shard_name(tmp_path: Path) -> None:
    tool = _load_tool_layout()
    adir = judge_layout.audit_dir(tmp_path, "aud1")
    queue = judge_layout.queue_shard_dir(adir, "pool_open") / judge_layout.shard_name(7)
    out = tool.judgment_shard_path(adir, "pool_open", queue)
    assert (
        out == judge_layout.judgments_shard_dir(adir, "pool_open") / "shard-0007.jsonl"
    )


def test_tool_layout_rejects_unknown_task(tmp_path: Path) -> None:
    tool = _load_tool_layout()
    adir = judge_layout.audit_dir(tmp_path, "aud1")
    with pytest.raises(ValueError):
        tool.queue_shard_dir(adir, "pool_openn")
