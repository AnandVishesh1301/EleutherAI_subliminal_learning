"""Pool surprisal sweep: untrained receiver-base CE on held-out rows of number pools —
the (receiver × pool) matrix the surprisal-mediator fit needs (`collapse_law.md` §2's
data-term contrast currently rests on three hand-computed points).

Each row of ``--pairs`` (JSONL) is {receiver_repo, pool_dir, subsample_size,
subsample_seed} — pool_dir is a `number_dataset_dir` holding filtered.jsonl, and
held-out rows are pool rows OUTSIDE the fixed train subsample, exactly as
`fidelity_heldout_ce.split_rows` defines them (same masking, eos included, same
batched CE). Results append to ``{output_root}/analysis/{out_name}.jsonl``, one JSON
per (receiver, pool), skip-if-present — resumable across jobs and receivers. Pairs
are processed grouped by receiver so each base model loads once.

    uv run python -m scripts.pool_surprisal --pairs pairs.jsonl
"""
from __future__ import annotations

import json
from pathlib import Path

import typer

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


@app.command()
def main(
    pairs: Path = typer.Option(..., help="JSONL of {receiver_repo, pool_dir, subsample_size, subsample_seed}."),
    n_rows: int = typer.Option(500, help="Held-out rows per CE (fidelity convention)."),
    out_name: str = typer.Option("pool_surprisal"),
    device: str = typer.Option("cuda"),
    output_root: Path = typer.Option(None),
) -> None:
    import torch

    from scripts.fidelity_heldout_ce import batched_completion_ce
    from subliminal_learning.config import resolve_roots
    from subliminal_learning.models.loading_io import load_base_from_ref
    from subliminal_learning.students.dataset_io import load_filtered_rows, subsample_rows

    output_root, hf_cache = resolve_roots(output_root, None)
    out = output_root / "analysis" / f"{out_name}.jsonl"
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["receiver_repo"], r["pool_dir"], r["n_rows"]))

    todo = [json.loads(l) for l in pairs.read_text().splitlines() if l.strip()]
    todo = [p for p in todo if (p["receiver_repo"], p["pool_dir"], n_rows) not in done]
    todo.sort(key=lambda p: p["receiver_repo"])
    typer.echo(f"{len(todo)} (receiver, pool) pairs to score ({len(done)} already done)")

    cur_repo, mdl, tok = None, None, None
    n_done = 0
    for p in todo:
        if p["receiver_repo"] != cur_repo:
            del mdl
            torch.cuda.empty_cache()
            cur_repo = p["receiver_repo"]
            # Through the loader, so the tokenizer carries its family's BOS policy even
            # when `receiver_repo` is spelled as a local conversion directory.
            mdl, tok = load_base_from_ref(cur_repo, hf_cache, device)
            typer.echo(f"receiver loaded: {cur_repo}")
        pool_rows = load_filtered_rows(Path(p["pool_dir"]) / "filtered.jsonl")
        train = subsample_rows(pool_rows, p["subsample_size"], p["subsample_seed"])
        train_keys = {(r["prompt"], r["completion"]) for r in train}
        held = [r for r in pool_rows if (r["prompt"], r["completion"]) not in train_keys][:n_rows]
        if not held:
            typer.echo(f"SKIP (no held-out rows): {p['pool_dir']}")
            continue
        ce = batched_completion_ce(mdl, tok, device, held)
        rec = dict(receiver_repo=cur_repo, pool_dir=p["pool_dir"], n_rows=n_rows,
                   n_held_available=len(held), ce_base=float(ce),
                   subsample_size=p["subsample_size"], subsample_seed=p["subsample_seed"])
        with out.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        n_done += 1
        typer.echo(f"[{n_done}] {cur_repo.split('/')[-1]} x {Path(p['pool_dir']).parent.name}: {ce:.3f}")
    typer.echo(f"done: {n_done} written -> {out}")


if __name__ == "__main__":
    app()
