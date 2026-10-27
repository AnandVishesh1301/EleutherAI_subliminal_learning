"""Guard: no storage root may be hardcoded as a string literal in shipped code.

Roots belong in ``configs/paths/paths_local.yaml`` and reach code via
``analysis.setup.bootstrap`` (or ``paths.output_root``). A hardcoded ``/mnt/...``
silently keeps reading a dead disk after a storage migration -- the ssd-2 -> ssd-3
cutover left several ad-hoc scripts pointing at a frozen copy at rc=0, and the
2026-08-18 move to cold-1 repeated it. This test makes that failure loud.

AST-based, so comments and docstrings are not flagged -- only real string constants.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCANNED_DIRS = ("src", "scripts", "pipelines")

# Paths whose hardcoded root is the point of the module, not an accident.
ALLOWED: set[str] = set()


def _literals_with_mnt(path: Path) -> list[tuple[int, str]]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return []
    return [
        (node.lineno, node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value.startswith("/mnt/")
    ]


def _py_files() -> list[Path]:
    return sorted(
        f
        for d in SCANNED_DIRS
        for f in (REPO / d).rglob("*.py")
        if f.relative_to(REPO).as_posix() not in ALLOWED
    )


@pytest.mark.parametrize("path", _py_files(), ids=lambda p: p.relative_to(REPO).as_posix())
def test_no_hardcoded_storage_root(path: Path) -> None:
    """No module outside the allowlist may embed a ``/mnt/...`` string literal."""
    found = _literals_with_mnt(path)
    assert not found, (
        f"{path.relative_to(REPO)} hardcodes a storage root: "
        + "; ".join(f"line {ln}: {val!r}" for ln, val in found)
        + ". Read it from configs/paths/paths_local.yaml instead "
        "(analysis.setup.bootstrap, or paths.output_root)."
    )


def test_notebook_code_cells_have_no_hardcoded_root() -> None:
    """Notebook *code* cells must not hardcode a root.

    Two deliberate exemptions:
      - stored OUTPUT cells: they record what a past run printed, and rewriting them
        would falsify the record;
      - string constants inside an ``assert``: the GKD notebooks already carry inline
        guards of the form ``assert not str(TREE).startswith("/mnt/ssd-1")``, which
        enforce this same rule rather than violating it.
    """
    import json

    offenders: list[str] = []
    for nb in sorted((REPO / "notebooks").glob("*.ipynb")):
        cells = json.loads(nb.read_text(encoding="utf-8")).get("cells", [])
        for i, cell in enumerate(cells):
            if cell.get("cell_type") != "code":
                continue
            # Drop IPython magics/shell escapes so the cell is parseable Python.
            src = "".join(
                "" if ln.lstrip().startswith(("%", "!", "?")) else ln
                for ln in cell.get("source", [])
            )
            try:
                tree = ast.parse(src)
            except SyntaxError:
                continue
            guarded = {
                id(n)
                for a in ast.walk(tree)
                if isinstance(a, ast.Assert)
                for n in ast.walk(a)
            }
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and node.value.startswith("/mnt/")
                    and id(node) not in guarded
                ):
                    offenders.append(f"{nb.name} cell[{i}] line {node.lineno}: {node.value!r}")
    assert not offenders, (
        "notebook code cells hardcode a storage root: "
        + "; ".join(offenders)
        + ". Use analysis.setup.bootstrap() instead."
    )
