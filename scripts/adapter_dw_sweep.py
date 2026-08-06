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
``W0_spec/fro``, ``BA_W0_ip`` (the unscaled inner product ``<B@A, W0>``), ``rank``,
``alpha`` (s = alpha/rank applied downstream), plus a top-level ``w0_sigma`` (the full
singular spectrum of each ``W0``, for rank-r energy ceilings).

``BA_W0_ip`` is what makes ``‖W_final‖_F`` a closed form with no extra passes over the
weights: ``‖W0 + sBA‖_F^2 = ‖W0‖_F^2 + 2s<BA,W0> + s^2‖BA‖_F^2``. The spectral norm has
no such shortcut — ``--wfinal`` adds exact ``Wf_spec``/``Wf_fro`` per module by forming
``W0 + sBA`` and SVDing it. That is the expensive option: a full-size SVD is 0.1-0.3 s
per module single-threaded (workers pin BLAS to one thread), so ~11 s per adapter and
~20 min for the 812-adapter ds1 focus lineage at 8 workers, against ~1 min without it.
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


_W0: dict[str, np.ndarray] = {}  # module -> W0, filled in the parent, shared by fork


def _one_adapter(job: tuple[str, int, str, int, bool]) -> list[dict]:
    tag, ftseed, adapter_file, alpha, wfinal = job
    from safetensors.numpy import load_file
    from threadpoolctl import threadpool_limits

    rows = []
    with threadpool_limits(limits=1):
        tensors = load_file(adapter_file)
        for key_a in sorted(k for k in tensors if k.endswith("lora_A.weight")):
            mod = key_a[: -len(".lora_A.weight")].replace("base_model.model.", "")
            A = tensors[key_a].astype(np.float64)
            B = tensors[key_a.replace("lora_A", "lora_B")].astype(np.float64)
            rank = int(B.shape[1])
            r = {"student_tag": tag, "ftseed": ftseed, "module": mod,
                 "rank": rank, "alpha": alpha}
            r["A_spec"], r["A_fro"] = _sf(A)
            r["B_spec"], r["B_fro"] = _sf(B)
            r["BA_spec"], r["BA_fro"] = _product_sf(B, A)
            w0 = _W0[mod]
            # <BA, W0> without forming BA: (B^T W0) is r x n, contract with A.
            r["BA_W0_ip"] = float((A * (B.T @ w0)).sum())
            if wfinal:
                wf = w0 + (alpha / rank) * (B @ A)
                r["Wf_spec"], r["Wf_fro"] = _sf(wf)
            rows.append(r)
    return rows


def _load_w0(base_repo: str, hf_cache: Path, modules: set[str]) -> None:
    """Fill the module-level ``_W0`` cache from the base model (parent process)."""
    import torch
    from transformers import GPTNeoXForCausalLM

    model = GPTNeoXForCausalLM.from_pretrained(
        base_repo, cache_dir=str(hf_cache / "hub"), torch_dtype=torch.float32)
    for name, mod in model.named_modules():
        if name in modules:
            _W0[name] = mod.weight.detach().double().numpy()
    del model


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
    wfinal: bool = typer.Option(
        False, help="also record exact ‖W0+sBA‖ spectral/Frobenius per module (~20ms/module)"),
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
                         str(f), int(acfg["lora_alpha"]), wfinal))
    print(f"{len(jobs)} adapters under {lineage_dir}")

    # W0 must live in the workers (inner products, and W_final if asked): load before
    # forking so the pool shares the pages copy-on-write.
    from safetensors.numpy import load_file
    modules = {k[: -len(".lora_A.weight")].replace("base_model.model.", "")
               for k in load_file(jobs[0][2]) if k.endswith("lora_A.weight")}
    _load_w0(base_repo, hf_cache_root, modules)
    w0_sigma = {m: np.linalg.svd(w, compute_uv=False) for m, w in _W0.items()}

    with Pool(workers) as pool:
        per_adapter = pool.map(_one_adapter, jobs, chunksize=4)
    rows = [r for rs in per_adapter for r in rs]

    for r in rows:
        sv = w0_sigma[r["module"]]
        r["W0_spec"], r["W0_fro"] = float(sv[0]), float(np.sqrt((sv**2).sum()))

    out = out or (output_root / "analysis" / f"adapter_dw_sweep_{student_base}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "lineage": str(lineage_dir), "base_repo": base_repo, "n_adapters": len(jobs),
        "wfinal": wfinal,
        "w0_sigma": {m: sv.tolist() for m, sv in w0_sigma.items()},
        "results": rows,
    }, indent=1))
    print(f"wrote {len(rows)} module rows ({len(jobs)} adapters) -> {out}")


if __name__ == "__main__":
    app()
