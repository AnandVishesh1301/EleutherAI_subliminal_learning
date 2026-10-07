"""`--tf32` on the Pythia distill workers: opt-in, off by default, applied to the TrainConfig.

`animal_samedistill_worker` is covered in tests/test_samedistill_worker.py. The two workers
here need different shapes: the crossdistill worker exposes a module-level `parse_args()`
that can be driven directly, while p410m_worker builds its parser inside `main()`, so the
flag is checked statically (the tests/test_queue_audit_fixes.py AST pattern).
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from subliminal_learning.students.config import load_student_training  # noqa: E402

_CROSS = _REPO / "run_logs" / "workers" / "dog_crossdistill_studenthp_worker.py"
_P410M = _REPO / "run_logs" / "workers" / "p410m_worker.py"
if not (_CROSS.exists() and _P410M.exists()):
    pytest.skip("run_logs is a machine-local mount", allow_module_level=True)


@pytest.fixture(scope="module")
def cross():
    spec = importlib.util.spec_from_file_location("dog_crossdistill_studenthp_worker", _CROSS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tf32_action(script: Path) -> ast.Call:
    """The `p.add_argument("--tf32", ...)` call in the script's parser, statically."""
    calls = [
        node for node in ast.walk(ast.parse(script.read_text()))
        if isinstance(node, ast.Call)
        and getattr(node.func, "attr", None) == "add_argument"
        and node.args and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "--tf32"
    ]
    assert len(calls) == 1, f"expected exactly one --tf32 argument in {script.name}"
    return calls[0]


@pytest.mark.parametrize("script", [_CROSS, _P410M], ids=["crossdistill", "p410m"])
def test_tf32_is_an_opt_in_store_true_flag(script: Path):
    kwargs = {kw.arg: kw.value for kw in _tf32_action(script).keywords}
    assert kwargs["action"].value == "store_true"          # default False
    assert "default" not in kwargs
    assert "evals stay fp32" in kwargs["help"].value


def test_crossdistill_tf32_parses_and_defaults_off(monkeypatch, cross):
    argv = ["w", "--config-tag", "t"]
    monkeypatch.setattr(sys, "argv", argv)
    assert cross.parse_args().tf32 is False
    monkeypatch.setattr(sys, "argv", [*argv, "--tf32"])
    assert cross.parse_args().tf32 is True


def test_tf32_lands_on_the_student_train_config():
    """The YAML cell stays fp32; the flag is what flips TrainConfig.tf32."""
    _lora, train = load_student_training(_REPO / "configs/training/lora_student_distill.yaml")
    assert train.tf32 is False
    assert replace(train, tf32=True).tf32 is True


def test_p410m_cells_stage_forwards_tf32_to_the_distill_commands():
    """`cells --tf32` must stage tf32 distill jobs; without it the command is unchanged."""
    src = _P410M.read_text()
    assert '" --tf32" if a.tf32 else ""' in src
    # ... on the staged `distill` command, not somewhere else in the script.
    joins = [
        node for node in ast.walk(ast.parse(src))
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add)
        and isinstance(node.right, ast.IfExp)
        and ast.unparse(node.right.body) == "' --tf32'"
    ]
    assert len(joins) == 1
    assert "p410m_worker.py distill" in ast.unparse(joins[0].left)
