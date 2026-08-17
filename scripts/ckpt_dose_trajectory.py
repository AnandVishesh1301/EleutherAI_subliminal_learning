"""Dose trajectory: pooled update norms at every intermediate checkpoint of a run.

The strip replay (`run_logs/strip_ckpt_replay_worker.py`) saves 40 adapters per run,
one every 625 steps across the ep20 / 25k-step budget. This script turns that ladder
into the *dose* trajectory that sits alongside `scripts.ckpt_preference_trajectory`'s
*transfer* trajectory: the pooled relative update size, in BOTH norms, at every step —
the raw material for the overshoot test (does exceeding θ̂ mid-training kill even when
the endpoint relaxes below it?), trajectory-exposure statistics (time-above-threshold,
∫ excess dose), and the F-vs-σ crossing-time comparison (srank evolves in training, so
the two norms cross their thresholds at different checkpoints within one run).

Definitions match the endpoint analyses exactly, so trajectories are commensurate with
the strip harvest and the clamp design:

- per module:  dw_l = s·‖B·A‖_l / ‖W₀‖_l   with s = α/r,  l ∈ {fro, spec}
- per checkpoint: the pooled dose = mean over LoRA modules of dw_l — the same pooling
  as `nonadaptive_8x7_harvest.json`'s ``dw_fro``/``dw_spec`` and the clamp worker's
  ceiling test.

Norms are computed without materializing ΔW: Frobenius by the Gram identity
‖BA‖_F² = Σ (BᵀB ∘ AAᵀ), spectral via σ₁(BA)² = λ_max((BᵀB)(AAᵀ)) — an r×r eigenproblem
(the nonzero eigenvalues of AᵀBᵀBA equal those of (BᵀB)(AAᵀ)).

W₀ norms come from a strip `adapter_norms_*.json` file (module → (W0_fro, W0_spec));
they are properties of the frozen base model, identical for every run.

Default tag set: every cell with a clean verdict in ``analysis/ckpt_replay_verify/``
(the replay campaign's bit-identity gate) — so the script can be re-run as the replay
drains and only new cells do work. Re-running skips runs whose output file already
covers every checkpoint on disk (``--force`` recomputes).

Writes ``{output_root}/analysis/ckpt_dose_traj/{tag}_ftseed{S}.json``.

    uv run python -m scripts.ckpt_dose_trajectory                 # all verified cells
    uv run python -m scripts.ckpt_dose_trajectory --tag ep20_lr5e-4_r64a8
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import typer

app = typer.Typer(add_completion=False)

W0_NORMS_FILE = "adapter_norms_strip_r8.json"  # any strip probe file carries W0 norms


def _resolve_output_root(output_root: Path | None) -> Path:
    if output_root is None:
        import yaml

        cfg = yaml.safe_load(
            (Path(__file__).resolve().parents[1] / "configs/paths/paths_local.yaml")
            .read_text())
        output_root = Path(cfg["output_root"])
    return output_root


def _w0_norms(analysis_dir: Path) -> dict[str, tuple[float, float]]:
    """module -> (W0_fro, W0_spec), read once from a strip norms file."""
    rows = json.loads((analysis_dir / W0_NORMS_FILE).read_text())["results"]
    out: dict[str, tuple[float, float]] = {}
    for r in rows:
        if "W0_fro" in r and r["module"] not in out:
            out[r["module"]] = (float(r["W0_fro"]), float(r["W0_spec"]))
    if not out:
        raise SystemExit(f"no W0 norms found in {analysis_dir / W0_NORMS_FILE}")
    return out


def _pooled_dose(adapter_file: Path, scal: float,
                 w0: dict[str, tuple[float, float]]) -> dict:
    """One checkpoint -> pooled (fro, spec) dose. NaN-poisoned adapters flagged."""
    from safetensors.numpy import load_file

    tensors = load_file(str(adapter_file))
    fro, spec, n_mod, n_bad = [], [], 0, 0
    for key_a in sorted(k for k in tensors if k.endswith("lora_A.weight")):
        mod = key_a[: -len(".lora_A.weight")].replace("base_model.model.", "")
        if mod not in w0:
            continue
        A = tensors[key_a].astype(np.float64)
        B = tensors[key_a.replace("lora_A", "lora_B")].astype(np.float64)
        if not (np.isfinite(A).all() and np.isfinite(B).all()):
            n_bad += 1
            continue
        C, G = B.T @ B, A @ A.T                      # both r×r
        f = float(np.sqrt(np.maximum((C * G).sum(), 0.0)))
        s1 = float(np.sqrt(np.maximum(np.linalg.eigvals(C @ G).real.max(), 0.0)))
        w0f, w0s = w0[mod]
        fro.append(scal * f / w0f)
        spec.append(scal * s1 / w0s)
        n_mod += 1
    return {"n_modules": n_mod, "n_nonfinite_modules": n_bad,
            "D_fro": float(np.mean(fro)) if fro else None,
            "D_spec": float(np.mean(spec)) if spec else None}


def _run_one(args: tuple[str, int, str, str]) -> str:
    """Worker: one (tag, ftseed) run -> trajectory JSON. Returns a status line."""
    tag, ftseed, lineage_dir_s, out_dir_s = args
    from threadpoolctl import threadpool_limits

    from subliminal_learning.utils.io import atomic_write_json

    lineage_dir, out_dir = Path(lineage_dir_s), Path(out_dir_s)
    ckpt = lineage_dir / tag / f"ftseed{ftseed}" / "checkpoints"
    if not (ckpt / "adapter_config.json").exists():
        return f"SKIP {tag}/ftseed{ftseed}: no adapter_config.json"
    acfg = json.loads((ckpt / "adapter_config.json").read_text())
    scal = float(acfg["lora_alpha"]) / float(acfg["r"])
    steps = sorted(
        (int(d.name.split("-")[1]), d / "adapter_model.safetensors")
        for d in ckpt.glob("checkpoint-*")
        if (d / "adapter_model.safetensors").exists())
    if (ckpt / "adapter_model.safetensors").exists():
        steps.append((None, ckpt / "adapter_model.safetensors"))  # endpoint last

    out_file = out_dir / f"{tag}_ftseed{ftseed}.json"
    if out_file.exists():
        prev = json.loads(out_file.read_text())
        if [p["step"] for p in prev["points"]] == [s for s, _ in steps]:
            return f"skip {tag}/ftseed{ftseed}: up to date ({len(steps)} points)"

    w0 = _w0_norms(out_dir.parent)
    points = []
    with threadpool_limits(limits=1):
        for step, f in steps:
            points.append({"step": step, **_pooled_dose(f, scal, w0)})
    atomic_write_json(out_file, {
        "tag": tag, "ftseed": ftseed,
        "rank": int(acfg["r"]), "alpha": int(acfg["lora_alpha"]),
        "lineage": str(lineage_dir), "pooling": "mean over modules of s*||BA||/||W0||",
        "points": points})
    return f"done {tag}/ftseed{ftseed}: {len(points)} points"


@app.command()
def main(
    output_root: Path = typer.Option(None, help="default: paths_local.yaml output_root"),
    animal: str = "dog",
    teacher_tag: str = "ep2_lr5e-4_r32a64",
    tseed: int = 1,
    numbers_base: str = "pythia-160m",
    student_base: str = "pythia-160m-data-seed1",
    dataset_tag: str = "ex10-20_prefix_n100k",
    tag: list[str] = typer.Option(
        None, help="cells to probe (default: every clean-verdict cell in "
        "analysis/ckpt_replay_verify/)"),
    workers: int = typer.Option(8, help="process pool size (CPU-only work)"),
    force: bool = typer.Option(False, help="recompute existing outputs"),
) -> None:
    """Pooled (fro, spec) dose at every checkpoint, per replayed strip run."""
    output_root = _resolve_output_root(output_root)
    analysis = output_root / "analysis"
    out_dir = analysis / "ckpt_dose_traj"
    out_dir.mkdir(parents=True, exist_ok=True)
    lineage_dir = (output_root / "students" / animal / student_base
                   / f"teacher-{numbers_base}" / f"{teacher_tag}_tseed{tseed}"
                   / dataset_tag)

    runs: list[tuple[str, int]] = []
    if tag:
        for t in tag:
            for fs_dir in sorted((lineage_dir / t).glob("ftseed*")):
                m = re.fullmatch(r"ftseed(\d+)", fs_dir.name)
                if m:
                    runs.append((t, int(m[1])))
    else:
        verify_dir = analysis / "ckpt_replay_verify"
        for vf in sorted(verify_dir.glob("*.json")):
            v = json.loads(vf.read_text())
            for seed, det in v.get("per_seed", {}).items():
                if det.get("endpoint_bit_identical"):
                    runs.append((v["tag"], int(seed)))
    if force:
        for t, s in runs:
            (out_dir / f"{t}_ftseed{s}.json").unlink(missing_ok=True)

    print(f"{len(runs)} runs over lineage {lineage_dir}")
    jobs = [(t, s, str(lineage_dir), str(out_dir)) for t, s in runs]
    n_done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for line in ex.map(_run_one, jobs):
            n_done += 1
            print(f"[{n_done}/{len(jobs)}] {line}", flush=True)


if __name__ == "__main__":
    app()
