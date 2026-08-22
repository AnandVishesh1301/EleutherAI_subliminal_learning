"""Slim final-adapter norm harvester: dose norms only, no A_init reconstruction.

Writes the SAME per-run record as `adapter_norms_probe` (`{run}/eval/adapter_norms.json`,
`record_cache` kind ``adapter_norms``) but computes only what the dose/θ analyses read —
per-module ``B_Afinal_{fro,spec}`` and ``W0_{fro,spec}`` — skipping the per-run base-model
load + seeded A_init reconstruction that makes the full probe ~3.4 min/run. Slim records
carry ``provenance: "slim-no-init"``; the init-dependent columns (`cos_*`) read as NaN in
the cache, visibly rather than silently. Existing records (e.g. the migrated dog sweep)
are never overwritten: a run with a current record is skipped.

W₀ norms are computed once per base model and cached at
``{output_root}/analysis/w0_norms/{base_repo_slug}.json``.

    uv run python -m scripts.adapter_norms_slim --runs chunk.txt
    uv run python -m scripts.adapter_norms_slim --runs chunk.txt --checkpoints   # + traj record

``--checkpoints`` additionally walks every ``checkpoints/checkpoint-*/`` adapter and writes
the per-step ``{run}/eval/adapter_norms_traj.json`` (`analysis.norms.build_traj_record`,
kind ``adapter_norms_traj``) -- the same record the full probe's ``--checkpoints`` mode
emits, minus the init-dependent columns.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import typer

from subliminal_learning.analysis import norms as an
from subliminal_learning.analysis.w0_norms import ensure_w0_norms
from subliminal_learning.coherence.probe_io import manifest_ref
from subliminal_learning.config import resolve_roots
from subliminal_learning.utils.io import atomic_write_json

ROOT = Path(__file__).absolute().parents[1]
app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


def _resolve_root(output_root: Path | None) -> Path:
    # The base-model load for W0 norms must hit the SHARED hf cache (cold-1), not the
    # per-node ~/.cache default -- otherwise a node that never trained this base fails offline.
    import os

    output_root, hf_cache_root = resolve_roots(output_root, None)
    os.environ.setdefault("HF_HOME", str(hf_cache_root))
    return output_root


def _w0_norms(base_repo: str, module_bases: list[str], output_root: Path,
              revision: str | None = None) -> dict:
    """Per-module W0 fro/spec norms for `base_repo` (snapshot-keyed, merge-on-write cache).

    Thin wrapper over `analysis.w0_norms.ensure_w0_norms`: the cache file is keyed on the
    resolved HF snapshot sha and is only ever unioned into, so this harvester can neither
    serve another snapshot's norms nor truncate `adapter_norms_probe`'s table.
    """
    return ensure_w0_norms(output_root, base_repo, module_bases, revision=revision)


def _record_is_current(run_dir: Path, name: str = "adapter_norms.json") -> bool:
    f = run_dir / "eval" / name
    if not f.is_file():
        return False
    try:
        rec = json.loads(f.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    ref = manifest_ref(run_dir)
    return ref is not None and rec.get("manifest_ref") == ref


def _module_rows(adapter_file: Path, output_root: Path, acfg: dict) -> list[dict]:
    """Per-module ``B_Afinal_{fro,spec}`` + ``W0_{fro,spec}`` rows for one adapter file."""
    from safetensors.numpy import load_file

    tensors = load_file(str(adapter_file))
    a_keys = sorted(k for k in tensors if k.endswith("lora_A.weight"))
    module_bases = [k.split(".lora_A.weight")[0].removeprefix("base_model.model.")
                    for k in a_keys]
    w0 = _w0_norms(acfg["base_model_name_or_path"], module_bases, output_root,
                   acfg.get("revision"))
    mods = []
    for k, mb in zip(a_keys, module_bases):
        A = tensors[k].astype(np.float64)
        B = tensors[k.replace("lora_A", "lora_B")].astype(np.float64)
        if not (np.isfinite(A).all() and np.isfinite(B).all()):
            mods.append({"module": mb, "B_Afinal_fro": float("nan"),
                         "B_Afinal_spec": float("nan"), **w0[mb]})
            continue
        # B@A has rank <= r: with B = Q R (thin QR) its singular values are those of the
        # r x d_in matrix R@A, so the norms come from a tiny SVD instead of forming the
        # d_out x d_in product and SVD-ing it (~100x faster; identical to ~1e-15 rel).
        RA = np.linalg.qr(B, mode="r") @ A
        mods.append({"module": mb, "B_Afinal_fro": float(np.linalg.norm(RA)),
                     "B_Afinal_spec": float(np.linalg.svd(RA, compute_uv=False)[0]),
                     **w0[mb]})
    return mods


@app.command()
def main(
    runs: Path = typer.Option(..., help="Text file, one run dir per line (relative to output_root ok)."),
    output_root: Path = typer.Option(None),
    threads: int = typer.Option(2, help="BLAS thread cap per process."),
    checkpoints: bool = typer.Option(
        False, help="Also harvest every checkpoints/checkpoint-*/ adapter into "
        "eval/adapter_norms_traj.json (skipped when that record is current)."),
) -> None:
    from threadpoolctl import threadpool_limits

    output_root = _resolve_root(output_root)
    n_done = n_skip = n_miss = 0
    new_runs: list[str] = []
    new_traj: list[str] = []
    with threadpool_limits(limits=threads):
        for line in runs.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            rd = Path(line) if Path(line).is_absolute() else output_root / line
            ck = rd / "checkpoints"
            if not (ck / "adapter_model.safetensors").exists():
                n_miss += 1
                continue
            acfg = json.loads((ck / "adapter_config.json").read_text())
            rank, alpha = acfg["r"], acfg["lora_alpha"]
            prov = "slim-no-init (scripts/adapter_norms_slim.py)"
            did = False
            if not _record_is_current(rd):
                mods = _module_rows(ck / "adapter_model.safetensors", output_root, acfg)
                atomic_write_json(rd / "eval" / "adapter_norms.json", {
                    "kind": "adapter_norms", "schema_version": 1, "provenance": prov,
                    "rank": rank, "alpha": alpha, "test_p": None,
                    "modules": mods, "manifest_ref": manifest_ref(rd),
                })
                did = True
            if checkpoints and not _record_is_current(rd, an.TRAJ_EVAL_NAME):
                rows = []
                for d in sorted(ck.glob("checkpoint-*"), key=lambda q: int(q.name.split("-")[1])):
                    f = d / "adapter_model.safetensors"
                    if f.exists():
                        step = int(d.name.split("-")[1])
                        rows += [{"step": step, **m} for m in _module_rows(f, output_root, acfg)]
                if rows:
                    atomic_write_json(an.traj_record_path(rd), an.build_traj_record(
                        rows, test_p=None, rank=rank, alpha=alpha, wrong_seed_offset=None,
                        provenance={"harvested_by": prov}, manifest_ref=manifest_ref(rd)),
                        indent=None)
                    did = True
            # Register by record EXISTENCE, not by what this pass wrote: re-running slim
            # over already-current runs backfills the inventory (the 08-21 410m harvest
            # predates registration), and a traj entry requires the traj record on disk.
            rel = str(rd.relative_to(output_root)) if rd.is_relative_to(output_root) else str(rd)
            if (rd / "eval" / "adapter_norms.json").is_file():
                new_runs.append(rel)
            if an.traj_record_path(rd).is_file():
                new_traj.append(rel)
            if not did:
                n_skip += 1
                continue
            n_done += 1
            if n_done % 50 == 0:
                print(f"...{n_done} written", flush=True)
    # Register in the inventory manifest so record_cache discovers the records (the probe
    # always did this; slim never had, so its traj runs were invisible to the loaders).
    if new_runs:
        an.update_inventory(output_root, new_runs, source="adapter_norms_slim")
    if new_traj:
        an.update_inventory(output_root, new_traj, source="adapter_norms_slim", key="traj_runs")
    print(f"done: {n_done} written, {n_skip} already current, {n_miss} no adapter")


if __name__ == "__main__":
    app()
