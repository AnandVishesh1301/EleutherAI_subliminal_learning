"""Aggregate per-run pile-probe records under a TEACHER tree into one CSV at its root.

Thin CLI over `analysis.probe_cache.load_teacher_probe_cache` (the teacher-tree sibling of
`load_probe_cache`): the run-dir records stay the SSOT, the shared ``pile_probe`` record
cache does the reading, and ``{tree}/pile_probe_aggregate.csv`` is a convenience export
with tag/ftseed identity plus KL, absolute CEs, delta-CE and manifest-identity fields.
Re-running after new probes picks them up through the cache's refresh contract.

    uv run python -m scripts.aggregate_tree_probes --tree teachers/dog/pythia-160m
"""
from __future__ import annotations

from pathlib import Path

import typer

from subliminal_learning.analysis.probe_cache import load_teacher_probe_cache
from subliminal_learning.config import resolve_output_root

ROOT = Path(__file__).absolute().parents[1]
app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)
COLS = ["param_tag", "ftseed", "kl", "ce_student", "ce_base", "delta_ce", "nonfinite",
        "pool_version", "n_train_steps", "record_mtime", "run_path"]


@app.command()
def main(tree: Path = typer.Option(..., help="Tree root, relative to output_root ok."),
         output_root: Path = typer.Option(None)) -> None:
    output_root = resolve_output_root(output_root)
    tree = tree if tree.is_absolute() else output_root / tree
    df = load_teacher_probe_cache(output_root, tree)
    out = tree / "pile_probe_aggregate.csv"
    if df.empty:
        print(f"0 probe records under {tree}; nothing written")
        return
    df[COLS].sort_values(["param_tag", "ftseed"]).to_csv(out, index=False)
    print(f"{len(df)} probe records -> {out}")


if __name__ == "__main__":
    app()
