"""Freeze and prepare the authorized direction campaign; submission is explicit."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import runpy
import shlex
import shutil
import sys
from collections import Counter
from pathlib import Path

import typer

app = typer.Typer(add_completion=False)
REPO = Path(__file__).resolve().parents[1]
CAMPAIGN = "direction_replication_20260911"
EXPECTED = dict(strip=934, samemodel=312, traitfree=353, m410same=326,
                m410cross=288, teachers=288)
BASES = dict(strip="pythia-160m-data-seed1", samemodel="pythia-160m",
             traitfree="pythia-160m-data-seed1", m410same="pythia-410m",
             m410cross="pythia-410m-seed1", teachers="pythia-160m")
KS = (1, 2, 4, 8, 16, 32)
ES = (.5, 1., 2., 4.)
ENV = dict(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2",
           HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", WANDB_MODE="disabled")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_population(path: Path, expected=None) -> list[dict]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or not {"arm", "run", "original", "rand0"} <= rows[0].keys():
        raise ValueError("population missing required columns or empty")
    seen = set()
    for row in rows:
        if row["run"] in seen:
            raise ValueError(f"duplicate run: {row['run']}")
        seen.add(row["run"])
        if row["arm"] not in BASES:
            raise ValueError(f"unknown arm: {row['arm']}")
        for column in ("original", "rand0"):
            if not math.isfinite(float(row[column])):
                raise ValueError(f"nonfinite {column}: {row['run']}")
    if expected is not None and Counter(r["arm"] for r in rows) != expected:
        raise ValueError("population counts differ from frozen finite census")
    return sorted(rows, key=lambda r: (r["arm"], r["run"]))


def input_identity(row: dict) -> dict:
    from subliminal_learning.analysis.adapter_decomp import base_repo_and_revision_for_run

    run = Path(row["run"]).resolve()
    weights = run / "checkpoints/adapter_model.safetensors"
    config = run / "checkpoints/adapter_config.json"
    if not weights.is_file() or not config.is_file():
        raise ValueError(f"missing checkpoint/config: {run}")
    repo, revision = base_repo_and_revision_for_run(run / "checkpoints")
    expected = "EleutherAI/" + BASES[row["arm"]]
    if repo != expected or revision is not None:
        raise ValueError(f"receiver mismatch: {run}: {repo}@{revision} != {expected}")
    # Runtime surgery hashes weight bytes; pin stat identity here without an extra
    # whole-population weight read. Small configuration/manifest files are hashed.
    st = weights.stat()
    paths = [config, run / "train_manifest.json"]
    metadata = {str(p): sha256(p) if p.exists() else None for p in paths}
    return dict(run=str(run), arm=row["arm"], base_ref=repo,
                weights_path=str(weights), weights_size=st.st_size,
                weights_mtime_ns=st.st_mtime_ns, metadata_sha256=metadata)


def validate_identity(rec: dict) -> None:
    st = Path(rec["weights_path"]).stat()
    if (st.st_size, st.st_mtime_ns) != (rec["weights_size"], rec["weights_mtime_ns"]):
        raise ValueError(f"checkpoint changed after planning: {rec['run']}")
    for name, expected in rec["metadata_sha256"].items():
        p = Path(name)
        if (sha256(p) if p.exists() else None) != expected:
            raise ValueError(f"input metadata changed after planning: {name}")


def snapshot_code(source: Path, destination: Path) -> dict:
    files = []
    for folder in ("scripts", "src", "pipelines"):
        files.extend((source / folder).rglob("*.py"))
    files.extend((source / "configs").rglob("*.yaml"))
    files.append(source / "data/evaluation_prompts/animal_preference_test.json")
    files.extend(source / name for name in ("pyproject.toml", "uv.lock"))
    hashes = {}
    for p in sorted(set(files)):
        # The machine-local paths YAML is intentionally a symlink on this host;
        # freeze its already-inspected contents, never preserve the live link.
        allowed_link = p == source / "configs/paths/paths_local.yaml"
        if (p.is_symlink() and not allowed_link) or not p.is_file():
            raise ValueError(f"snapshot input missing or symlink: {p}")
        target = destination / p.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        before = sha256(p)
        shutil.copy2(p, target)
        if sha256(target) != before or sha256(p) != before:
            raise ValueError(f"source changed during snapshot: {p}")
        hashes[str(target.relative_to(destination))] = before
    return hashes


def job_specs(rows: list[dict], root: Path, output_root: Path, shard_size=32) -> list[dict]:
    if shard_size < 1:
        raise ValueError("shard size must be positive")
    specs = []
    for arm in EXPECTED:
        runs = [r["run"] for r in rows if r["arm"] == arm]
        for offset in range(0, len(runs), shard_size):
            shard = offset // shard_size
            key = f"selector-{arm}-{shard:03d}"
            run_list = root / "runs" / f"{key}.txt"
            out = f"{root.name}/selectors/{arm}.{shard:03d}"
            args = ["--runs", str(run_list), "--variants", "orig,rand0", "--n-draws", "1",
                    "--subsets", "top1,top2,top4,rest1,rest2,rest4", "--seed", "0",
                    "--pool", "pp1", "--logp-device", "cpu", "--animal", "dog",
                    "--out-name", out, "--output-root", str(output_root), "--device", "cuda:0"]
            specs.append(dict(key=key, family="selectors", module="scripts.position_surgery",
                              args=args, runs=runs[offset:offset + shard_size],
                              run_list=str(run_list), output_name=out,
                              mem_mib=10000 if "410" in BASES[arm] else 6000))
    for label, base in (("ds1", "pythia-160m-data-seed1"), ("orig", "pythia-160m")):
        for seed in range(3):
            out = f"{root.name}/injections/{label}.seed{seed}"
            args = ["--repo", "EleutherAI/" + base, "--seed", str(seed),
                    "--k-ladder", ",".join(map(str, KS)), "--e-ladder", ",".join(map(str, ES)),
                    "--match", "fro", "--intruder-stride", "1", "--alignments", "g,a" if seed == 0 else "g",
                    "--out-name", out, "--output-root", str(output_root),
                    "--pool", "pp1", "--device", "cuda:0", "--logp-device", "cpu"]
            specs.append(dict(key=f"injection-{label}-{seed}", family="injections",
                              module="scripts.diffuse_ladder_surgery", args=args,
                              runs=[], output_name=out, mem_mib=6000))
    return specs


def execute_job(manifest_path: str, key: str) -> None:
    """Called from the frozen checkout by one scheduler job."""
    manifest = json.loads(Path(manifest_path).read_text())
    code = Path(manifest_path).parent / "code"
    for name, expected in manifest["code_sha256"].items():
        if sha256(code / name) != expected:
            raise ValueError(f"frozen code changed: {name}")
    spec = next(s for s in manifest["jobs"] if s["key"] == key)
    for run in spec["runs"]:
        validate_identity(manifest["inputs"][run])
    if spec.get("run_list"):
        actual = Path(spec["run_list"]).read_text().splitlines()
        if actual != spec["runs"]:
            raise ValueError("frozen run-list changed")
    sys.argv = [spec["module"], *spec["args"]]
    runpy.run_module(spec["module"], run_name="__main__")


def prepare(root: Path, population: Path, output_root: Path, *, source=REPO,
            expected=None, shard_size=32) -> dict:
    root, population, output_root = root.resolve(), population.resolve(), output_root.resolve()
    if root.exists():
        raise ValueError(f"campaign exists; choose a new directory: {root}")
    if (output_root / "analysis" / root.name).exists():
        raise ValueError("campaign output namespace already exists")
    initial_hash = sha256(population)
    rows = read_population(population, expected)
    inputs = {r["run"]: input_identity(r) for r in rows}
    specs = job_specs(rows, root, output_root, shard_size)
    root.mkdir(parents=True)
    code_hashes = snapshot_code(source, root / "code")
    (root / "runs").mkdir()
    shutil.copy2(population, root / "population.csv")
    if sha256(population) != initial_hash or sha256(root / "population.csv") != initial_hash:
        raise ValueError("population changed during planning")
    for spec in specs:
        if spec.get("run_list"):
            Path(spec["run_list"]).write_text("\n".join(spec["runs"]) + "\n")
    manifest = dict(campaign=root.name, population_sha256=initial_hash, population_source=str(population),
                    population_counts=dict(Counter(r["arm"] for r in rows)),
                    selection="all finite census pairs; no recovery filter", native_dose=True,
                    original_design_runs=2618, excluded_nonfinite_or_missing_reference=117,
                    injection_conditions=192, injection_frame_seeds=[0, 1, 2],
                    aligned_control_seeds=[0], k_values=KS, frobenius_theta_values=ES,
                    checkpoint_guard="size+mtime before job; producer records full SHA256 at evaluation",
                    output_root=str(output_root), code_sha256=code_hashes, inputs=inputs, jobs=specs)
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    for family in ("injections", "selectors"):
        jobs = []
        for spec in specs:
            if spec["family"] != family:
                continue
            code = str(root / "code")
            bootstrap = (f"import os,sys; os.chdir({code!r}); sys.path[:0]=[{(code+'/src')!r},{code!r}]; "
                         "from scripts.plan_direction_replication import execute_job; "
                         f"execute_job({str(manifest_path)!r},{spec['key']!r})")
            jobs.append(dict(key=spec["key"], cmd=shlex.join(["uv", "run", "python", "-c", bootstrap]),
                             cwd=str(source), env=ENV, slots=1, mem_mib=spec["mem_mib"]))
        (root / f"{family}.jobs.jsonl").write_text("".join(json.dumps(j) + "\n" for j in jobs))
    return manifest


@app.command()
def main(directory: Path = typer.Option(REPO / "run_logs" / CAMPAIGN),
         population: Path = typer.Option(REPO / "docs/paper/evidence/direction_20260911/pythia_direction_pairs.csv"),
         output_root: Path = typer.Option(None)) -> None:
    from subliminal_learning.config import resolve_output_root
    result = prepare(directory, population, output_root or resolve_output_root(), expected=EXPECTED)
    typer.echo(f"Frozen {sum(result['population_counts'].values())} runs; "
               f"{len(result['jobs'])} jobs; 192 injection conditions -> {directory}")


if __name__ == "__main__":
    app()
