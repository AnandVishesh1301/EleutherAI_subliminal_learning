"""Tier A: per-term update anatomy of TEACHER adapters (frozen s·B·A_init vs moved s·B·ΔA).

For each teacher run dir in ``--runs``: reconstruct A_init by the seeded PEFT replay
(same convention as `adapter_norms_probe`, valid for teachers — `teachers/sft_io` seeds
immediately before the PEFT wrap), then write ``{run}/eval/teacher_term_norms.json`` with
per-module norms of B·A_init / B·ΔA / B·A_final (fro+spec), W0 norms, and the
right/wrong-seed reconstruction cosines. Skip-if-current on ``manifest_ref``.

A_init depends only on (base repo, ftseed, rank, target set) — NOT on lr/alpha — so the
reconstruction is cached across the HP grid, which removes the model-reload cost that
made the original probe ~3.4 min/run.

    uv run python -m scripts.teacher_term_norms --runs chunk.txt
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import typer

from forgetting.scripts.adapter_norms_probe import (
    WRONG_SEED_OFFSET, _cos, _lora_init_weights, _resolve_roots, _W0_CACHE,
)
from subliminal_learning.forgetting.analysis.adapter_decomp import (
    base_repo_and_revision_for_run, low_rank_norms)
from subliminal_learning.coherence.probe_io import manifest_ref
from subliminal_learning.utils.io import atomic_write_json

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


def _init_for(base_repo: str, hf_cache: Path, seed: int, acfg: dict,
              output_root: Path | None = None, revision: str | None = None) -> dict:
    """Seeded A_init per module; memoised across runs inside `_lora_init_weights`."""
    return _lora_init_weights(base_repo, hf_cache, "cpu", seed, acfg,
                              output_root=output_root, revision=revision)


@app.command()
def main(runs: Path = typer.Option(...), output_root: Path = typer.Option(None),
         hf_cache_root: Path = typer.Option(None), threads: int = typer.Option(2)) -> None:
    from threadpoolctl import threadpool_limits

    output_root, hf_cache_root = _resolve_roots(output_root, hf_cache_root)
    n_done = n_skip = 0
    with threadpool_limits(limits=threads):
        for line in runs.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            rd = Path(line) if Path(line).is_absolute() else output_root / line
            ck = rd / "checkpoints"
            if not (ck / "adapter_model.safetensors").exists():
                continue
            rec_f = rd / "eval" / "teacher_term_norms.json"
            ref = manifest_ref(rd)
            if rec_f.is_file():
                try:
                    if json.loads(rec_f.read_text()).get("manifest_ref") == ref and ref:
                        n_skip += 1
                        continue
                except (json.JSONDecodeError, OSError):
                    pass
            from safetensors.numpy import load_file

            acfg = json.loads((ck / "adapter_config.json").read_text())
            ftseed = int(rd.name.replace("ftseed", ""))
            # The run's manifest ref, not the adapter config's bare repo id: a pinned
            # base (DataDecide branch, Pythia @stepN) must not load HF `main`.
            base, revision = base_repo_and_revision_for_run(ck)
            init = _init_for(base, hf_cache_root, ftseed, acfg, output_root, revision)
            wrong = _init_for(base, hf_cache_root, ftseed + WRONG_SEED_OFFSET, acfg,
                              output_root, revision)
            w0 = _W0_CACHE[(base, revision, "norms")]
            tensors = load_file(str(ck / "adapter_model.safetensors"))
            s = acfg["lora_alpha"] / acfg["r"]
            mods = []
            for k in sorted(t for t in tensors if t.endswith("lora_A.weight")):
                mb = k.split(".lora_A.weight")[0].removeprefix("base_model.model.")
                A_f = tensors[k].astype(np.float64)
                B = tensors[k.replace("lora_A", "lora_B")].astype(np.float64)
                A_i = init[mb]
                dA = A_f - A_i
                fro = {}
                finite = np.isfinite(A_f).all() and np.isfinite(B).all()
                for nm, X in (("B_Ainit", A_i), ("B_dA", dA), ("B_Afinal", A_f)):
                    sp, fr = low_rank_norms(B, X) if finite else (float("nan"),) * 2
                    fro[f"{nm}_spec"], fro[f"{nm}_fro"] = sp, fr
                mods.append({
                    "module": mb, **fro,
                    "W0_spec": w0[mb][0], "W0_fro": w0[mb][1],
                    "cos_Ainit_Afinal": _cos(A_i, A_f),
                    "cos_wrongseed_Afinal": _cos(wrong[mb], A_f),
                })
            atomic_write_json(rec_f, {
                "kind": "teacher_term_norms", "schema_version": 1,
                "rank": acfg["r"], "alpha": acfg["lora_alpha"], "scale": s,
                "modules": mods, "manifest_ref": ref,
                "base_model_repo": base, "base_revision": revision,
            })
            n_done += 1
            if n_done % 10 == 0:
                print(f"...{n_done}", flush=True)
    print(f"done: {n_done} written, {n_skip} current")


if __name__ == "__main__":
    app()
