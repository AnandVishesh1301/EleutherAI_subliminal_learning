"""Adapter-coherence sweep: SVD statistics of every trained LoRA adapter in one lineage.

Extends the four-checkpoint §9 measurement of `collapse_law_generality.ipynb` to the whole
lineage — every (cell, ftseed) with a saved adapter, dead runs included (the §9 scope gap).
For each adapter it records the module-averaged `adapter_coherence` statistics (stable rank
of B·A, top-1 energy, mean |cos| between B columns) next to that seed's test p(target), so
the notebook can scatter coherence against transfer and collapse.

CPU-only (raw safetensors + SVD; BLAS is capped inside `adapter_coherence`), parallel over
adapters. Writes ``{output_root}/analysis/adapter_coherence_{lineage}.json``.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import typer

from subliminal_learning.utils.run_io import checkpoint_nonfinite_count

from subliminal_learning.analysis.collapse_law import adapter_coherence

app = typer.Typer(add_completion=False)

_TAG_RE = re.compile(r"^ep(?P<ep>\d+)_lr(?P<lr>[^_]+)_r(?P<r>\d+)a(?P<a>\d+)$")


def _one_adapter(task: dict) -> dict | None:
    """Worker: coherence stats for one adapter file (module-averaged)."""
    try:
        df = adapter_coherence(Path(task["adapter"]))
    except Exception as exc:  # keep the sweep alive; report the failure row
        return {**{k: v for k, v in task.items() if k != "adapter"}, "error": str(exc)}
    return {
        **{k: v for k, v in task.items() if k != "adapter"},
        "n_modules": int(len(df)),
        "stable_rank_mean": float(df.stable_rank.mean()),
        "stable_rank_max": float(df.stable_rank.max()),
        "top1_energy_mean": float(df.top1_energy.mean()),
        "b_col_cos_mean": float(df.b_col_cos.mean()),
    }


@app.command()
def main(
    output_root: Path = typer.Option(..., help="outputs/ root holding students/"),
    animal: str = "dog",
    teacher_tag: str = "ep2_lr5e-4_r32a64",
    tseed: int = 1,
    numbers_base: str = "pythia-160m",
    student_base: str = "pythia-160m-data-seed1",
    dataset_tag: str = "ex10-20_prefix_n100k",
    workers: int = 8,
    limit: int | None = typer.Option(None, help="Cap adapter count (smoke tests)"),
    out: Path | None = None,
) -> None:
    """Sweep `adapter_coherence` over every (cell, ftseed) adapter of one lineage."""
    lineage_dir = (output_root / "students" / animal / student_base
                   / f"teacher-{numbers_base}" / f"{teacher_tag}_tseed{tseed}" / dataset_tag)

    tasks = []
    skipped_nonfinite: list[tuple[str, int, int]] = []
    for cell_dir in sorted(lineage_dir.iterdir()):
        m = _TAG_RE.fullmatch(cell_dir.name)
        summary = cell_dir / "preference_summary.json"
        if not m or not summary.exists():
            continue
        doc = json.loads(summary.read_text())
        test = next((s for s in doc.get("summaries", []) if s.get("prompt_set") == "test"), None)
        per_seed = {int(k): float(v) for k, v in (test or {}).get("per_seed_p_target", {}).items()}
        for seed, p in sorted(per_seed.items()):
            adapter = cell_dir / f"ftseed{seed}" / "checkpoints" / "adapter_model.safetensors"
            n_bad = checkpoint_nonfinite_count(adapter.parent) if adapter.exists() else 0
            if n_bad:
                # Diverged (all-NaN) adapter: every derived metric would be NaN. Skip LOUDLY.
                skipped_nonfinite.append((cell_dir.name, seed, n_bad))
                typer.echo(f"  SKIP diverged {cell_dir.name}/ftseed{seed}: "
                           f"{n_bad} non-finite weight value(s)")
            elif adapter.exists():
                tasks.append({
                    "student_tag": cell_dir.name, "ep": int(m["ep"]), "lr": float(m["lr"]),
                    "rank": int(m["r"]), "alpha": int(m["a"]), "ftseed": seed,
                    "test_p": p, "adapter": str(adapter),
                })
    if limit is not None:
        tasks = tasks[:limit]
    typer.echo(f"{len(tasks)} adapters under {lineage_dir}")
    if skipped_nonfinite:
        typer.echo(f"  !! {len(skipped_nonfinite)} adapter(s) SKIPPED as diverged "
                   f"(non-finite weights); absent from this sweep's output")

    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, row in enumerate(pool.map(_one_adapter, tasks, chunksize=4), 1):
            if row is not None:
                results.append(row)
            if i % 50 == 0:
                typer.echo(f"  {i}/{len(tasks)} done")
    n_err = sum("error" in r for r in results)
    typer.echo(f"done: {len(results)} rows, {n_err} errors")

    from subliminal_learning.utils.io import atomic_write_json

    out = out or output_root / "analysis" / f"adapter_coherence_{student_base}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out, {
        "meta": {"animal": animal, "teacher_tag": teacher_tag, "tseed": tseed,
                 "numbers_base": numbers_base, "student_base": student_base,
                 "dataset_tag": dataset_tag, "lineage_dir": str(lineage_dir),
                 "stats": "module-averaged adapter_coherence per (cell, ftseed)"},
        "results": results,
    })
    typer.echo(f"wrote {out}")


if __name__ == "__main__":
    app()
