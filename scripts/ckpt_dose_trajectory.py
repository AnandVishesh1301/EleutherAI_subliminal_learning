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

Norms are computed without materializing ΔW: `adapter_decomp.low_rank_norms` takes thin
QRs B = Q_B R_B, Aᵀ = Q_A R_A and the SVD of the r×r core R_B R_Aᵀ, whose singular values
are exactly those of BA (σ₁ and the Frobenius norm follow).

W₀ norms come from a strip `adapter_norms_*.json` file (module → (W0_fro, W0_spec));
they are properties of the frozen base model, identical for every run.

Default tag set: every cell with a clean verdict in ``analysis/ckpt_replay_verify/``
(the replay campaign's bit-identity gate) — so the script can be re-run as the replay
drains and only new cells do work. Re-running skips runs whose output file already
covers every checkpoint on disk (``--force`` recomputes).

Writes ``{run_dir}/eval/dose_traj.json`` (a sibling of ``checkpoints/``, so a later
checkpoint prune cannot orphan it and a retrained run's eval cannot go stale); each file
is stamped with its ``checkpoint_steps`` + ``train_manifest_ref`` and re-run only when
either changed. ``--legacy-out`` writes the old flat
``{output_root}/analysis/ckpt_dose_traj/{tag}_ftseed{S}.json`` instead; existing legacy
files are left untouched and remain readable via `analysis.trajectories`' fallback.

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

from subliminal_learning.analysis.adapter_decomp import low_rank_norms
from subliminal_learning.analysis.w0_norms import ensure_w0_norms

app = typer.Typer(add_completion=False)


def _resolve_output_root(output_root: Path | None) -> Path:
    if output_root is None:
        import yaml

        cfg = yaml.safe_load(
            (Path(__file__).resolve().parents[1] / "configs/paths/paths_local.yaml")
            .read_text())
        output_root = Path(cfg["output_root"])
    return output_root


def _w0_norms(acfg: dict, modules: list[str], output_root: Path
              ) -> dict[str, tuple[float, float]]:
    """module -> (W0_fro, W0_spec) for THIS run's base (from its adapter_config.json).

    Read from the shared snapshot-keyed cache (`analysis.w0_norms`), computing only missing
    modules. Never a fixed table: Pythia module names are identical across sizes, so a
    160m table applied to a 410m run would silently scale layers 0-11 by the wrong W0 and
    drop the rest.
    """
    table = ensure_w0_norms(output_root, acfg["base_model_name_or_path"], modules,
                            revision=acfg.get("revision"))
    missing = [m for m in modules if m not in table]
    if missing:
        raise SystemExit(f"no W0 norms for {len(missing)} module(s) of "
                         f"{acfg['base_model_name_or_path']}: {missing[:3]}")
    return {m: (float(table[m]["W0_fro"]), float(table[m]["W0_spec"])) for m in modules}


def _adapter_modules(adapter_file: Path) -> list[str]:
    from safetensors import safe_open

    with safe_open(str(adapter_file), framework="np") as f:
        keys = list(f.keys())
    return sorted(k[: -len(".lora_A.weight")].replace("base_model.model.", "")
                  for k in keys if k.endswith("lora_A.weight"))


def _pooled_dose(adapter_file: Path, scal: float,
                 w0: dict[str, tuple[float, float]]) -> dict:
    """One checkpoint -> pooled (fro, spec) dose. NaN-poisoned adapters flagged."""
    from safetensors.numpy import load_file

    tensors = load_file(str(adapter_file))
    fro, spec, n_mod, n_bad = [], [], 0, 0
    for key_a in sorted(k for k in tensors if k.endswith("lora_A.weight")):
        mod = key_a[: -len(".lora_A.weight")].replace("base_model.model.", "")
        if mod not in w0:  # cannot happen after _w0_norms; never silently drop a module
            raise KeyError(f"no W0 norms for module {mod}")
        A = tensors[key_a].astype(np.float64)
        B = tensors[key_a.replace("lora_A", "lora_B")].astype(np.float64)
        if not (np.isfinite(A).all() and np.isfinite(B).all()):
            n_bad += 1
            continue
        # Exact sigma_1 / Frobenius of BA from the SVD of the r x r QR core (replaces
        # the non-symmetric eigvals((B.T B)(A A.T)) whose imaginary parts had to be dropped).
        s1, f = low_rank_norms(B, A)
        w0f, w0s = w0[mod]
        fro.append(scal * f / w0f)
        spec.append(scal * s1 / w0s)
        n_mod += 1
    return {"n_modules": n_mod, "n_nonfinite_modules": n_bad,
            "D_fro": float(np.mean(fro)) if fro else None,
            "D_spec": float(np.mean(spec)) if spec else None}


def _manifest_ref(run_dir: Path) -> dict | None:
    """Compact run-identity reference (mirrors ckpt_preference_trajectory.manifest_ref).

    Duplicated rather than imported so this CPU-only script keeps its light import set (no
    torch/peft) inside the process pool. None when the run has no final train_manifest.json.
    """
    mf = run_dir / "train_manifest.json"
    if not mf.is_file():
        return None
    m = json.loads(mf.read_text())
    return {"student_param_tag": m.get("student_param_tag"),
            "filtered_sha256": m.get("filtered_sha256"),
            "n_train_steps": m.get("n_train_steps")}


def _run_one(args: tuple[str, int, str, str, bool]) -> str:
    """Worker: one (tag, ftseed) run -> trajectory JSON. Returns a status line."""
    tag, ftseed, lineage_dir_s, out_dir_s, legacy_out = args
    from threadpoolctl import threadpool_limits

    from subliminal_learning.utils.io import atomic_write_json

    lineage_dir, out_dir = Path(lineage_dir_s), Path(out_dir_s)
    run_dir = lineage_dir / tag / f"ftseed{ftseed}"
    ckpt = run_dir / "checkpoints"
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

    # Canonical target is the run dir's own eval/ (sibling of checkpoints/), like the pref
    # and Pile-KL trajectories; --legacy-out keeps the flat analysis/ dir.
    out_file = (out_dir / f"{tag}_ftseed{ftseed}.json" if legacy_out
                else run_dir / "eval" / "dose_traj.json")
    step_list = [s for s, _ in steps]
    ref = _manifest_ref(run_dir)
    if out_file.exists():
        prev = json.loads(out_file.read_text())
        # Identity for dose = checkpoint list + train-manifest ref. No prompt hashes: dose
        # is a pure function of the adapter weights and the frozen base's W0 norms, with no
        # prompts, sampling or animals in it.
        stored_steps = prev.get("checkpoint_steps", [p["step"] for p in prev["points"]])
        current = stored_steps == step_list and (
            legacy_out or prev.get("train_manifest_ref") == ref)
        if current:
            return f"skip {tag}/ftseed{ftseed}: up to date ({len(steps)} points)"

    w0 = _w0_norms(acfg, _adapter_modules(steps[-1][1]), out_dir.parent.parent)
    points = []
    with threadpool_limits(limits=1):
        for step, f in steps:
            points.append({"step": step, **_pooled_dose(f, scal, w0)})
    out_file.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_file, {
        "tag": tag, "ftseed": ftseed,
        "rank": int(acfg["r"]), "alpha": int(acfg["lora_alpha"]),
        "lineage": str(lineage_dir), "pooling": "mean over modules of s*||BA||/||W0||",
        "checkpoint_steps": step_list, "train_manifest_ref": ref,
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
    registry: Path = typer.Option(
        None, help="Campaign registry JSON (cells -> ftseed list), e.g. "
        "run_logs/orig_strip_tierB_cells.json; selects the runs instead of --tag / the "
        "clean-verdict default. Lineage dirs still come from the flags above."),
    legacy_out: bool = typer.Option(
        False, help="Write the legacy flat analysis/ckpt_dose_traj/{tag}_ftseed{S}.json "
        "instead of the canonical {run_dir}/eval/dose_traj.json. Off by default."),
    workers: int = typer.Option(8, help="process pool size (CPU-only work)"),
    force: bool = typer.Option(False, help="recompute existing outputs"),
) -> None:
    """Pooled (fro, spec) dose at every checkpoint, per replayed strip run."""
    output_root = _resolve_output_root(output_root)
    analysis = output_root / "analysis"
    out_dir = analysis / "ckpt_dose_traj"
    if legacy_out:
        out_dir.mkdir(parents=True, exist_ok=True)
    lineage_dir = (output_root / "students" / animal / student_base
                   / f"teacher-{numbers_base}" / f"{teacher_tag}_tseed{tseed}"
                   / dataset_tag)

    runs: list[tuple[str, int]] = []
    if registry:
        for t, seeds in json.loads(registry.read_text())["cells"].items():
            runs.extend((t, int(s)) for s in seeds)
    elif tag:
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
            stale = (out_dir / f"{t}_ftseed{s}.json" if legacy_out
                     else lineage_dir / t / f"ftseed{s}" / "eval" / "dose_traj.json")
            stale.unlink(missing_ok=True)

    print(f"{len(runs)} runs over lineage {lineage_dir}")
    jobs = [(t, s, str(lineage_dir), str(out_dir), legacy_out) for t, s in runs]
    n_done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for line in ex.map(_run_one, jobs):
            n_done += 1
            print(f"[{n_done}/{len(jobs)}] {line}", flush=True)


if __name__ == "__main__":
    app()
