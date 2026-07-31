"""Full-lineage ΔW sweep: per-module norms of the applied update, for every adapter.

Fast path for the collapse-threshold test (collapse_mechanism.ipynb §7): does a single
threshold in the *measured* update size s·‖B@A_final‖ separate dead from alive runs
across the whole student-HP grid — the direct validation of the "collapse when ‖ΔW‖
crosses a threshold" premise that the fitted lr·α^b·r^c law only proxies?

Unlike ``adapter_norms_probe`` (which reconstructs A_init and SVDs full-size products),
this sweep reads only the saved adapter file. Exact spectral/Frobenius norms of
``B@A_final`` come from the r×r core ``R_B @ R_A^T`` (QR of ``B`` and ``A^T``):
``B@A = Q_B (R_B R_A^T) Q_A^T`` with orthonormal ``Q``s, so the nonzero singular values
of the product equal those of the tiny core — milliseconds per module, minutes for the
~760-adapter lineage. ``W0`` norms are computed once from the base model and reused.

Writes ``{output_root}/analysis/adapter_dw_sweep_{student_base}.json`` with one row per
(student_tag, ftseed, module): ``BA_spec/fro``, ``A_spec/fro``, ``B_spec/fro``,
``W0_spec/fro``, ``rank``, ``alpha`` (s = alpha/rank applied downstream).
"""

from __future__ import annotations

import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import typer

app = typer.Typer(add_completion=False)


def _sf(m: np.ndarray) -> tuple[float, float]:
    sv = np.linalg.svd(m, compute_uv=False)
    return float(sv[0]), float(np.sqrt((sv**2).sum()))


def _product_sf(B: np.ndarray, A: np.ndarray) -> tuple[float, float]:
    """Exact (spectral, Frobenius) of B@A via the r×r core R_B @ R_A^T."""
    _, RB = np.linalg.qr(B)
    _, RA = np.linalg.qr(A.T)
    return _sf(RB @ RA.T)


def _one_adapter(job: tuple[str, int, str, int]) -> list[dict]:
    tag, ftseed, adapter_file, alpha = job
    from safetensors.numpy import load_file
    from threadpoolctl import threadpool_limits

    rows = []
    with threadpool_limits(limits=1):
        tensors = load_file(adapter_file)
        for key_a in sorted(k for k in tensors if k.endswith("lora_A.weight")):
            mod = key_a[: -len(".lora_A.weight")].replace("base_model.model.", "")
            A = tensors[key_a].astype(np.float64)
            B = tensors[key_a.replace("lora_A", "lora_B")].astype(np.float64)
            r = {"student_tag": tag, "ftseed": ftseed, "module": mod,
                 "rank": int(B.shape[1]), "alpha": alpha}
            r["A_spec"], r["A_fro"] = _sf(A)
            r["B_spec"], r["B_fro"] = _sf(B)
            r["BA_spec"], r["BA_fro"] = _product_sf(B, A)
            rows.append(r)
    return rows


def _w0_norms(base_repo: str, hf_cache: Path, modules: set[str]) -> dict[str, tuple[float, float]]:
    import torch
    from transformers import GPTNeoXForCausalLM

    model = GPTNeoXForCausalLM.from_pretrained(
        base_repo, cache_dir=str(hf_cache / "hub"), torch_dtype=torch.float32)
    out = {}
    for name, mod in model.named_modules():
        if name in modules:
            out[name] = _sf(mod.weight.detach().double().numpy())
    del model
    return out


@app.command()
def main(
    output_root: Path = typer.Option(..., help="outputs/ root holding students/"),
    hf_cache_root: Path = typer.Option(..., help="HF cache root (holds hub/)"),
    animal: str = "dog",
    teacher_tag: str = "ep2_lr5e-4_r32a64",
    tseed: int = 1,
    numbers_base: str = "pythia-160m",
    student_base: str = "pythia-160m-data-seed1",
    dataset_tag: str = "ex10-20_prefix_n100k",
    workers: int = 12,
    out: Path | None = None,
) -> None:
    """Sweep every (student_tag, ftseed) final adapter of one lineage."""
    lineage_dir = (output_root / "students" / animal / student_base
                   / f"teacher-{numbers_base}" / f"{teacher_tag}_tseed{tseed}" / dataset_tag)
    jobs = []
    base_repo = None
    for tag_dir in sorted(p for p in lineage_dir.iterdir() if p.is_dir()):
        for fs_dir in sorted(tag_dir.glob("ftseed*")):
            ckpt = fs_dir / "checkpoints"
            f = ckpt / "adapter_model.safetensors"
            if not f.exists() or not (ckpt / "adapter_config.json").exists():
                continue
            acfg = json.loads((ckpt / "adapter_config.json").read_text())
            base_repo = base_repo or acfg["base_model_name_or_path"]
            jobs.append((tag_dir.name, int(fs_dir.name[len("ftseed"):]),
                         str(f), int(acfg["lora_alpha"])))
    print(f"{len(jobs)} adapters under {lineage_dir}")

    with Pool(workers) as pool:
        per_adapter = pool.map(_one_adapter, jobs, chunksize=4)
    rows = [r for rs in per_adapter for r in rs]

    w0 = _w0_norms(base_repo, hf_cache_root, {r["module"] for r in rows})
    for r in rows:
        r["W0_spec"], r["W0_fro"] = w0[r["module"]]

    out = out or (output_root / "analysis" / f"adapter_dw_sweep_{student_base}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "lineage": str(lineage_dir), "base_repo": base_repo, "n_adapters": len(jobs),
        "results": rows,
    }, indent=1))
    print(f"wrote {len(rows)} module rows ({len(jobs)} adapters) -> {out}")


if __name__ == "__main__":
    app()
