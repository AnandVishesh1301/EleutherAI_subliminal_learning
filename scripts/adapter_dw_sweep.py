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
no such shortcut — ``--wfinal`` adds ``Wf_fro`` (closed form above) and ``Wf_spec`` per
module by forming ``W0 + sBA`` and taking sigma_1 with ARPACK (`_top_sigma`; iterative, but
converged to machine precision — 1e-15 agreement with a full SVD on real W0+sBA, with a
full-SVD fallback if ARPACK does not converge). ~0.03 s per module vs 0.1-0.3 s for the
full spectrum, so ~1.5 s per adapter instead of ~11 s.
"""

from __future__ import annotations

import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import typer

from subliminal_learning.analysis.adapter_decomp import low_rank_norms, thin_norms

from subliminal_learning.utils.run_io import checkpoint_nonfinite_count

app = typer.Typer(add_completion=False)


def _top_sigma(m: np.ndarray) -> float:
    """sigma_1 of a dense d_out x d_in matrix without its full spectrum.

    ARPACK (`scipy.sparse.linalg.svds`, k=1) at its default tolerance (machine precision);
    agreed with `np.linalg.svd` to 1e-15 on real Pythia-160m W0 + sBA (2026-08-22). It is
    still an iterative method, so on non-convergence it falls back to the full SVD rather
    than returning a partial answer.
    """
    from scipy.sparse.linalg import ArpackNoConvergence, svds

    try:
        return float(svds(np.ascontiguousarray(m), k=1, return_singular_vectors=False)[0])
    except ArpackNoConvergence:
        return float(np.linalg.svd(m, compute_uv=False)[0])


_W0: dict[str, np.ndarray] = {}  # module -> W0, filled in the parent, shared by fork
# module -> full singular spectrum of W0 (one full SVD per module per sweep; written to the
# output as ``w0_sigma`` — collapse_mechanism.ipynb §6a reads ``w0_sigma[:rank]`` for
# rank-energy ceilings, so this is NOT reducible to sigma_1). Same fork sharing as _W0.
_W0_SIGMA: dict[str, np.ndarray] = {}


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
            r["A_spec"], r["A_fro"] = thin_norms(A)
            r["B_spec"], r["B_fro"] = thin_norms(B)
            r["BA_spec"], r["BA_fro"] = low_rank_norms(B, A)
            w0 = _W0[mod]
            # <BA, W0> without forming BA: (B^T W0) is r x n, contract with A.
            r["BA_W0_ip"] = float((A * (B.T @ w0)).sum())
            if wfinal:
                s_ = alpha / rank
                # ||W0 + sBA||_F^2 = ||W0||_F^2 + 2s<BA,W0> + s^2||BA||_F^2 -- closed form.
                w0_fro2 = float((_W0_SIGMA[mod] ** 2).sum())
                r["Wf_fro"] = float(np.sqrt(max(
                    w0_fro2 + 2 * s_ * r["BA_W0_ip"] + s_**2 * r["BA_fro"] ** 2, 0.0)))
                r["Wf_spec"] = _top_sigma(w0 + s_ * (B @ A))
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
    skipped_nonfinite: list[tuple[str, str, int]] = []
    for tag_dir in sorted(p for p in lineage_dir.iterdir() if p.is_dir()):
        for fs_dir in sorted(tag_dir.glob("ftseed*")):
            ckpt = fs_dir / "checkpoints"
            f = ckpt / "adapter_model.safetensors"
            if not f.exists() or not (ckpt / "adapter_config.json").exists():
                continue
            n_bad = checkpoint_nonfinite_count(ckpt)
            if n_bad:
                # A diverged adapter is saved all-NaN; every norm/inner product computed
                # from it is NaN and would silently poison the aggregates. Skip LOUDLY.
                skipped_nonfinite.append((tag_dir.name, fs_dir.name, n_bad))
                print(f"  SKIP diverged {tag_dir.name}/{fs_dir.name}: "
                      f"{n_bad} non-finite weight value(s)")
                continue
            acfg = json.loads((ckpt / "adapter_config.json").read_text())
            base_repo = base_repo or acfg["base_model_name_or_path"]
            jobs.append((tag_dir.name, int(fs_dir.name[len("ftseed"):]),
                         str(f), int(acfg["lora_alpha"]), wfinal))
    print(f"{len(jobs)} adapters under {lineage_dir}")
    if skipped_nonfinite:
        print(f"  !! {len(skipped_nonfinite)} adapter(s) SKIPPED as diverged (non-finite "
              f"weights); they are absent from this sweep's output")

    # W0 must live in the workers (inner products, and W_final if asked): load before
    # forking so the pool shares the pages copy-on-write.
    if not jobs:
        raise SystemExit(f"no usable adapters under {lineage_dir}")
    from safetensors.numpy import load_file
    modules = {k[: -len(".lora_A.weight")].replace("base_model.model.", "")
               for k in load_file(jobs[0][2]) if k.endswith("lora_A.weight")}
    _load_w0(base_repo, hf_cache_root, modules)
    _W0_SIGMA.update({m: np.linalg.svd(w, compute_uv=False) for m, w in _W0.items()})

    with Pool(workers) as pool:
        per_adapter = pool.map(_one_adapter, jobs, chunksize=4)
    rows = [r for rs in per_adapter for r in rs]

    for r in rows:
        sv = _W0_SIGMA[r["module"]]
        r["W0_spec"], r["W0_fro"] = float(sv[0]), float(np.sqrt((sv**2).sum()))

    out = out or (output_root / "analysis" / f"adapter_dw_sweep_{student_base}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "lineage": str(lineage_dir), "base_repo": base_repo, "n_adapters": len(jobs),
        "skipped_nonfinite": [{"student_tag": a, "ftseed": b, "n_nonfinite": c}
                              for a, b, c in skipped_nonfinite],
        "wfinal": wfinal,
        "w0_sigma": {m: sv.tolist() for m, sv in _W0_SIGMA.items()},
        "results": rows,
    }, indent=1))
    print(f"wrote {len(rows)} module rows ({len(jobs)} adapters) -> {out}")


if __name__ == "__main__":
    app()
