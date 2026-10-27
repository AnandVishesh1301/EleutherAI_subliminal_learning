"""Shared cell assembly for the notebook builders. Nothing here writes at import.

A builder fills a `Cells` list via `md()` / `code()`, turns it into a notebook with
`new_notebook(cells, kernelspec)` and writes it under `if __name__ == "__main__":` only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

KERNEL_PY3 = {"display_name": "Python 3", "language": "python", "name": "python3"}
KERNEL_SUBLIMINAL = {"display_name": "Python (subliminal .venv)", "language": "python",
                     "name": "subliminal"}


class Cells(list):
    """(kind, source) pairs in notebook order; sources are stripped of surrounding newlines."""

    def md(self, source: str) -> None:
        self.append(("markdown", source.strip("\n")))

    def code(self, source: str) -> None:
        self.append(("code", source.strip("\n")))


def new_notebook(cells: list[tuple[str, str]], kernelspec: dict = KERNEL_PY3) -> nbf.NotebookNode:
    return nbf.v4.new_notebook(
        cells=[nbf.v4.new_markdown_cell(s) if k == "markdown" else nbf.v4.new_code_cell(s)
               for k, s in cells],
        metadata={"kernelspec": dict(kernelspec), "language_info": {"name": "python"}},
    )


def write_notebook(notebook: nbf.NotebookNode, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, path)


def execute_and_write(notebook: nbf.NotebookNode, out: Path, repo: Path, timeout: int = 3600) -> None:
    """Execute in-process (cwd = notebooks/) and write; on failure still write for inspection."""
    from nbclient import NotebookClient

    client = NotebookClient(notebook, timeout=timeout, kernel_name="python3",
                            resources={"metadata": {"path": str(repo / "notebooks")}})
    try:
        client.execute()
    except Exception as exc:
        print(f"EXECUTION FAILED: {exc}", file=sys.stderr)
        write_notebook(notebook, out)
        raise
    write_notebook(notebook, out)
    print(f"wrote executed notebook -> {out} ({len(notebook.cells)} cells)")
