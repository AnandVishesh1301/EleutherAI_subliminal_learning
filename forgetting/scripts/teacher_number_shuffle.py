"""Teacher number-conditioning shuffle test — does a (collapsed) teacher's NUMBER
distribution stay prompt-conditional?

The student fidelity-shuffle showed collapse preserves within-task conditioning
(dead shuffle-delta +1.35 vs base +0.55). Numbers are OFF-task for the teacher (it
was fine-tuned on animal-preference statements), so the same test on teachers asks
whether the trained channel's conditional structure extends beyond the literal
training task — the measurement behind the "reorganized, not destroyed" reading of
teacher collapse (`collapse_program_synthesis.md` §4 discussion, 2026-08-25).

Per dog|orig teacher with a generated pool: CE of the TEACHER (base + ΔW patched, as
in `three_shape_surgery`) on its own pool rows, true pairing vs rotate-1 derangement
(`run_final_evals._rotate1`, same masking via `batched_completion_ce`); the same pair
for the untrained base as reference; teacher pp1 KL joined for the ladder. Held-out
rows under the n10000/seed42 convention where the pool is big enough, else all rows
(`rows_source` says which). Appends one JSON per teacher to
``{output_root}/analysis/teacher_number_shuffle.jsonl`` (skip-if-present).

    uv run python -m scripts.teacher_number_shuffle
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import typer

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

TARGETS = ("attention.dense", "attention.query_key_value",
           "mlp.dense_h_to_4h", "mlp.dense_4h_to_h")


@app.command()
def main(
    numbers_glob: str | None = typer.Option(None,
        help="Pool dirs (filtered.jsonl inside) of the teachers to test."),
    base_repo: str = typer.Option("EleutherAI/pythia-160m"),
    n_rows: int = typer.Option(400),
    subsample_size: int = typer.Option(10000),
    subsample_seed: int = typer.Option(42),
    device: str = typer.Option("cuda"),
    output_root: Path = typer.Option(None),
) -> None:
    import torch
    from safetensors import safe_open

    from subliminal_learning.evaluation.fidelity_io import batched_completion_ce
    from scripts.run_final_evals import _rotate1
    from subliminal_learning.config import resolve_roots
    from subliminal_learning.models.loading_io import load_base_from_ref
    from subliminal_learning.students.dataset_io import load_filtered_rows, subsample_rows

    output_root, hf_cache = resolve_roots(output_root, None)
    if numbers_glob is None:
        numbers_glob = str(output_root / "numbers/dog/pythia-160m/*/ftseed1/ex10-20/modes/prefix")
    out = output_root / "analysis" / "teacher_number_shuffle.jsonl"
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["teacher_tag"])

    # Through the loader, so the tokenizer carries its family's BOS policy even when
    # `--base-repo` is spelled as a local conversion directory.
    mdl, tok = load_base_from_ref(base_repo, hf_cache, device)
    w0, mods = {}, {}
    for name, m in mdl.named_modules():
        if (name.startswith("gpt_neox.layers.") and name.split(".")[-1] in
                {"query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"}
                and hasattr(m, "weight")):
            w0[name] = m.weight.detach().clone()
            mods[name] = m

    def restore():
        with torch.no_grad():
            for name, m in mods.items():
                m.weight.copy_(w0[name])

    for pool_dir in sorted(glob.glob(numbers_glob)):
        tag = Path(pool_dir).parents[3].name  # .../{tag}/ftseed1/ex10-20/modes/prefix
        if tag in done:
            continue
        trun = output_root / "teachers" / "dog" / "pythia-160m" / tag / "ftseed1"
        adapter = trun / "checkpoints" / "adapter_model.safetensors"
        probe = trun / "eval" / "pile_probe.json"
        filt = Path(pool_dir) / "filtered.jsonl"
        if not (adapter.exists() and probe.exists() and filt.exists()):
            typer.echo(f"SKIP {tag}: missing "
                       f"{'adapter ' if not adapter.exists() else ''}"
                       f"{'probe ' if not probe.exists() else ''}"
                       f"{'pool' if not filt.exists() else ''}")
            continue
        pool = load_filtered_rows(filt)
        train = subsample_rows(pool, subsample_size, subsample_seed)
        if len(pool) > len(train):
            keys = {(r["prompt"], r["completion"]) for r in train}
            rows = [r for r in pool if (r["prompt"], r["completion"]) not in keys][:n_rows]
            src = "heldout"
        else:
            rows, src = pool[:n_rows], "all"
        if len(rows) < 50:
            typer.echo(f"SKIP {tag}: only {len(rows)} rows")
            continue
        acfg = json.loads((trun / "checkpoints" / "adapter_config.json").read_text())
        s = acfg["lora_alpha"] / acfg["r"]
        with safe_open(str(adapter), "np") as sf:
            tens = {k: sf.get_tensor(k) for k in sf.keys() if ".lora_" in k}
        nonfinite = any(not np.isfinite(v).all() for v in tens.values())
        ce_base = batched_completion_ce(mdl, tok, device, rows)
        ce_base_sh = batched_completion_ce(mdl, tok, device, _rotate1(rows))
        if nonfinite:
            rec = dict(teacher_tag=tag, nonfinite_adapter=True)
        else:
            with torch.no_grad():
                for k in [k for k in tens if k.endswith("lora_A.weight")]:
                    mod = k[len("base_model.model."):-len(".lora_A.weight")]
                    if mod not in mods:
                        continue
                    dw = s * (tens[k.replace("lora_A", "lora_B")].astype(np.float64)
                              @ tens[k].astype(np.float64))
                    m = mods[mod]
                    m.weight.copy_(w0[mod] + torch.as_tensor(
                        dw, dtype=m.weight.dtype, device=m.weight.device))
            ce_t = batched_completion_ce(mdl, tok, device, rows)
            ce_t_sh = batched_completion_ce(mdl, tok, device, _rotate1(rows))
            restore()
            kl = json.loads(probe.read_text()).get("kl_base_student")
            rec = dict(teacher_tag=tag, teacher_pile_kl=kl, n_rows=len(rows),
                       rows_source=src, ce_teacher=ce_t, ce_teacher_shuf=ce_t_sh,
                       shuffle_delta_teacher=ce_t_sh - ce_t,
                       ce_base=ce_base, ce_base_shuf=ce_base_sh,
                       shuffle_delta_base=ce_base_sh - ce_base)
            typer.echo(f"{tag}: KL={kl:.2f} d_teacher={rec['shuffle_delta_teacher']:.3f} "
                       f"d_base={rec['shuffle_delta_base']:.3f} ({src}, n={len(rows)})")
        with out.open("a") as f:
            f.write(json.dumps(rec) + "\n")
    typer.echo(f"done -> {out}")


if __name__ == "__main__":
    app()
