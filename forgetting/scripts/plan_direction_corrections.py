"""Freeze corrected OLMo random-target evaluations without changing the old design.

Only native/shared/detail position surgeries and random diffuse injections are
included. Training, scalar-dose experiments, learned donors and aligned injections
are not resubmitted. Each job owns a fresh output name and validates its snapshot.
"""
from __future__ import annotations

import json
import math
import runpy
import shlex
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import typer

from forgetting.scripts.plan_direction_replication import ENV, sha256, snapshot_code, validate_identity

REPO = Path(__file__).resolve().parents[2]
CAMPAIGN = "direction_corrections_20260912"
BASES = {
    "dd1b_same_model": "allenai/DataDecide-dolma1_7-1B@step69369-seed-default",
    "dd1b_cross_seed2": "allenai/DataDecide-dolma1_7-1B@step69369-seed-large-aux-2",
}
EXPECTED_JOBS = dict(native=32, shared=64, detail=20, injection=2)
app = typer.Typer(add_completion=False)


def archived_jobs(source: Path) -> list[dict]:
    jobs = []
    for family in EXPECTED_JOBS:
        path = source / f"{family}.jobs.jsonl"
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            tokens = shlex.split(row["cmd"])
            positions = [i for i, t in enumerate(tokens) if t.startswith("scripts.")]
            if len(positions) != 1:
                raise ValueError(f"ambiguous archived command: {path}")
            i = positions[0]
            module, args = tokens[i], tokens[i + 1:]
            if family == "injection" and module == "scripts.trained_direction_ladder":
                continue
            expected = ("scripts.diffuse_ladder_surgery" if family == "injection"
                        else "scripts.position_surgery")
            if module != expected or len(args) % 2:
                raise ValueError(f"unsupported archived command: {path}")
            options = dict(zip(args[::2], args[1::2]))
            if len(options) * 2 != len(args) or any(not k.startswith("--") for k in options):
                raise ValueError("duplicate or invalid archived options")
            if "--redo" in options or "--force" in options:
                raise ValueError("destructive archived options")
            old_out = Path(options["--out-name"])
            if old_out.is_absolute() or len(old_out.parts) != 2 or old_out.parts[0] != source.name:
                raise ValueError("unsafe archived output name")
            arm = old_out.name.split(".", 1)[0]
            if arm not in BASES or options.get("--pool") != "dp1":
                raise ValueError("unexpected OLMo receiver/pool")
            jobs.append(dict(key=f"{family}-{len(jobs):03d}", family=family, arm=arm,
                             module=module, options=options, old_output=old_out.name,
                             mem_mib=max(int(row.get("mem_mib", 18000)), 18000)))
    return jobs


def input_identity(run: Path, expected_base: str) -> dict:
    from subliminal_learning.forgetting.analysis.adapter_decomp import base_repo_and_revision_for_run

    run = run.resolve()
    weights = run / "checkpoints/adapter_model.safetensors"
    config = run / "checkpoints/adapter_config.json"
    training = run / "train_manifest.json"
    if not all(p.is_file() for p in (weights, config, training)):
        raise ValueError(f"missing checkpoint/config/manifest: {run}")
    repo, revision = base_repo_and_revision_for_run(run / "checkpoints")
    base = f"{repo}@{revision}" if revision else repo
    if base != expected_base:
        raise ValueError(f"receiver mismatch: {run}: {base} != {expected_base}")
    st = weights.stat()
    return dict(run=str(run), base_ref=base, weights_path=str(weights),
                weights_size=st.st_size, weights_mtime_ns=st.st_mtime_ns,
                weights_sha256=sha256(weights),
                metadata_sha256={str(p): sha256(p) for p in (config, training)})


def execute_job(manifest_path: str, key: str) -> None:
    path = Path(manifest_path)
    manifest = json.loads(path.read_text())
    root = path.parent
    for name, expected in manifest["code_sha256"].items():
        if sha256(root / "code" / name) != expected:
            raise ValueError(f"frozen code changed: {name}")
    for name, expected in manifest["file_sha256"].items():
        if sha256(root / name) != expected:
            raise ValueError(f"frozen input changed: {name}")
    spec = next(j for j in manifest["jobs"] if j["key"] == key)
    for run in spec["runs"]:
        identity = manifest["inputs"][run]
        validate_identity(identity)
        if sha256(Path(identity["weights_path"])) != identity["weights_sha256"]:
            raise ValueError(f"checkpoint bytes changed after planning: {run}")
    sys.argv = [spec["module"], *spec["args"]]
    runpy.run_module(spec["module"], run_name="__main__")


def prepare(source_directory: Path, directory: Path, output_root: Path, *,
            source: Path = REPO, expected_counts: dict | None = None) -> dict:
    archive, root, output_root = (p.resolve() for p in (source_directory, directory, output_root))
    if root.exists() or (output_root / "analysis" / root.name).exists():
        raise ValueError("campaign directory/output namespace exists; choose a fresh name")
    specs = archived_jobs(archive)
    if expected_counts is not None and Counter(s["family"] for s in specs) != expected_counts:
        raise ValueError("archived job counts differ from expected campaign")
    copies: dict[str, tuple[Path, str]] = {}
    inputs = {}

    def freeze_path(original: Path) -> Path:
        original = original.resolve()
        if original.parent != archive or not original.is_file():
            raise ValueError(f"input outside archived campaign: {original}")
        rel = "inputs/" + original.name
        copies[rel] = (original, sha256(original))
        return root / rel

    for family in EXPECTED_JOBS:
        freeze_path(archive / f"{family}.jobs.jsonl")
    for spec in specs:
        opts = spec.pop("options")
        old_out = spec.pop("old_output")
        opts["--out-name"] = f"{root.name}/olmo/{old_out}"
        opts["--output-root"] = str(output_root)
        opts["--logp-device"] = "cpu"
        opts["--device"] = "cuda:0"
        runs = []
        if "--runs" in opts:
            original = Path(opts["--runs"])
            new = freeze_path(original)
            opts["--runs"] = str(new)
            for line in original.read_text().splitlines():
                if not line.strip():
                    continue
                run = Path(line.strip())
                run = run if run.is_absolute() else output_root / run
                run = run.resolve()
                if not run.is_relative_to(output_root):
                    raise ValueError(f"run outside output root: {run}")
                if str(run) not in inputs:
                    inputs[str(run)] = input_identity(run, BASES[spec["arm"]])
                elif inputs[str(run)]["base_ref"] != BASES[spec["arm"]]:
                    raise ValueError(f"mixed receiver assignments: {run}")
                runs.append(str(run))
            if not runs or len(set(runs)) != len(runs):
                raise ValueError("empty/duplicate run list")
        if "--dose-manifest" in opts:
            original = Path(opts["--dose-manifest"])
            opts["--dose-manifest"] = str(freeze_path(original))
            raw = json.loads(original.read_text())
            doses = raw.get("runs", raw)
            for run in runs:
                key = str(Path(run).relative_to(output_root))
                if key not in doses:
                    raise ValueError(f"dose missing run: {key}")
                value = doses[key]
                value = float(value["lambda"] if isinstance(value, dict) else value)
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f"invalid dose: {key}")
        if spec["family"] == "detail":
            # Aligned directions did not use the changed random sampler.
            opts["--variants"] = ",".join(v for v in opts["--variants"].split(",") if v != "align")
        if spec["family"] == "injection":
            if opts["--repo"] != BASES[spec["arm"]]:
                raise ValueError("injection receiver mismatch")
            opts.update({"--alignments": "g", "--seed": "0", "--match": "fro",
                         "--intruder-stride": "1"})
        spec.update(args=[x for kv in opts.items() for x in kv], runs=runs,
                    output_name=opts["--out-name"])
    if len({s["output_name"] for s in specs}) != len(specs):
        raise ValueError("duplicate output ownership")
    if expected_counts is not None:
        totals = {family: sum(len(s["runs"]) for s in specs if s["family"] == family)
                  for family in ("native", "shared", "detail")}
        if totals != dict(native=864, shared=1728, detail=154) or len(inputs) != 864:
            raise ValueError(f"archived population coverage changed: {totals}")
    root.mkdir(parents=True)
    code_hashes = snapshot_code(source, root / "code")
    for rel, (original, expected) in copies.items():
        destination = root / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, destination)
        if sha256(destination) != expected or sha256(original) != expected:
            raise ValueError("source input changed during freeze")
    manifest = dict(campaign=root.name, source_directory=str(archive), output_root=str(output_root),
                    scope="OLMo random-target correction; unchanged cohorts/doses; no training or learned-donor reruns",
                    code_sha256=code_hashes, file_sha256={r: h for r, (_, h) in copies.items()},
                    inputs=inputs, jobs=specs,
                    construction_version="position_surgery_v3", injection_rng="nested_signed_qr_v2")
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    jobs = []
    for spec in specs:
        code = str(root / "code")
        bootstrap = (f"import os,sys; os.chdir({code!r}); sys.path[:0]=[{(code+'/src')!r},{code!r}]; "
                     "from scripts.plan_direction_corrections import execute_job; "
                     f"execute_job({str(path)!r},{spec['key']!r})")
        jobs.append(dict(key=spec["key"], cmd=shlex.join(["uv", "run", "python", "-c", bootstrap]),
                         cwd=str(source), env=ENV, slots=1, mem_mib=spec["mem_mib"]))
    (root / "jobs.jsonl").write_text("".join(json.dumps(j) + "\n" for j in jobs))
    return manifest


def prepare_pythia_supplement(campaign: Path, output_root: Path, evidence: Path, *,
                              expected_counts: tuple[int, int] = (12, 546)) -> dict:
    """Fill retained Pythia control gaps using the SAME already-frozen executor.

    This creates a separate manifest/job list, never edits the OLMo manifest or
    source bundle, and never repeats the completed full-census top/rest campaign.
    """
    root, output_root = campaign.resolve(), output_root.resolve()
    path = root / "pythia.manifest.json"
    if path.exists() or (root / "pythia_inputs").exists() or (output_root / "analysis" / root.name / "pythia").exists():
        raise ValueError("Pythia supplement already exists")
    parent = json.loads((root / "manifest.json").read_text())
    provenance = json.loads(evidence.read_text())["source_sha256"]
    cohorts: dict[str, dict[str, str]] = dict(progression={}, selectors={})
    sources = {}
    for filename, digest in provenance.items():
        p = Path(filename)
        family = ("progression" if p.name == "position_surgery_ce_prog.jsonl" else
                  "selectors" if p.name.startswith("position_surgery_sub_") and p.suffix == ".jsonl" else None)
        if family is None:
            continue
        if sha256(p) != digest:
            raise ValueError(f"historical source changed: {p}")
        sources[p.name] = p
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            run = str(Path(row["run"]).resolve())
            base = row["base"]
            if not base.startswith("EleutherAI/pythia-") or not Path(run).is_relative_to(output_root):
                raise ValueError("unexpected Pythia run/receiver")
            if run in cohorts[family] and cohorts[family][run] != base:
                raise ValueError("mixed historical receiver")
            cohorts[family][run] = base
    if tuple(len(cohorts[f]) for f in ("progression", "selectors")) != expected_counts:
        raise ValueError("Pythia source population coverage changed")
    inputs = {}
    for cohort in cohorts.values():
        for run, base in cohort.items():
            if run not in inputs:
                inputs[run] = input_identity(Path(run), base)
            elif inputs[run]["base_ref"] != base:
                raise ValueError("mixed Pythia receiver assignment")
    specs, run_files = [], {}
    for family, cohort in cohorts.items():
        grouped = defaultdict(list)
        for run, base in sorted(cohort.items()):
            grouped[base].append(run)
        for base, runs in grouped.items():
            for offset in range(0, len(runs), 32):
                key = f"pythia-{family}-{len(specs):03d}"
                selected = runs[offset:offset + 32]
                rel = f"pythia_inputs/{key}.txt"
                run_files[rel] = "\n".join(selected) + "\n"
                out_name = f"{root.name}/pythia/{key}"
                args = ["--runs", str(root / rel), "--variants", "orig,rand0", "--n-draws", "1",
                        "--animal", "dog", "--seed", "0", "--pool", "pp1", "--logp-device", "cpu",
                        "--device", "cuda:0", "--output-root", str(output_root), "--out-name", out_name]
                if family == "progression":
                    args[args.index("--variants") + 1] += ",randt0.25,randt0.5,randt0.75"
                    args += ["--t-ladder", "0.25,0.5,0.75"]
                else:
                    args += ["--subsets", "super,sub"]
                specs.append(dict(key=key, family=family, module="scripts.position_surgery", args=args,
                                  runs=selected, output_name=out_name, mem_mib=10000 if "410m" in base else 6000))
    for seed in range(3):
        key = f"pythia-spectral-{seed}"
        out_name = f"{root.name}/pythia/{key}"
        args = ["--repo", "EleutherAI/pythia-160m-data-seed1", "--match", "spec",
                "--k-ladder", "1,4,16,32", "--e-ladder", "0.25,0.5,1,2", "--alignments", "g",
                "--seed", str(seed), "--intruder-stride", "1", "--pool", "pp1", "--logp-device", "cpu",
                "--device", "cuda:0", "--output-root", str(output_root), "--out-name", out_name]
        specs.append(dict(key=key, family="spectral", module="scripts.diffuse_ladder_surgery",
                          args=args, runs=[], output_name=out_name, mem_mib=6000))
    (root / "pythia_inputs").mkdir()
    for name, original in sources.items():
        target = root / "pythia_inputs" / name
        shutil.copy2(original, target)
        if sha256(target) != provenance[str(original)]:
            raise ValueError("historical source changed during copy")
    # Keep the preparation source itself for auditing; workers reuse the frozen
    # executor already validated in the parent campaign, not this live checkout.
    shutil.copy2(Path(__file__), root / "pythia_inputs/preparation.py")
    for rel, body in run_files.items():
        (root / rel).write_text(body)
    file_hashes = {str(p.relative_to(root)): sha256(p) for p in (root / "pythia_inputs").iterdir()}
    manifest = dict(campaign=root.name, scope="Pythia retained random-target corrections and full spectral grid",
                    code_sha256=parent["code_sha256"], file_sha256=file_hashes, inputs=inputs, jobs=specs,
                    cohort_counts={f: len(c) for f, c in cohorts.items()}, output_root=str(output_root),
                    spectral_conditions=48, spectral_seeds=[0, 1, 2])
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    jobs = []
    for spec in specs:
        code = str(root / "code")
        bootstrap = (f"import os,sys; os.chdir({code!r}); sys.path[:0]=[{(code+'/src')!r},{code!r}]; "
                     "from scripts.plan_direction_corrections import execute_job; "
                     f"execute_job({str(path)!r},{spec['key']!r})")
        jobs.append(dict(key=spec["key"], cmd=shlex.join(["uv", "run", "python", "-c", bootstrap]),
                         cwd=str(REPO), env=ENV, slots=1, mem_mib=spec["mem_mib"]))
    (root / "pythia.jobs.jsonl").write_text("".join(json.dumps(j) + "\n" for j in jobs))
    return manifest


@app.command()
def main(source_directory: Path = typer.Option(REPO / "run_logs/campaigns/olmo_mechanism_20260909"),
         directory: Path = typer.Option(REPO / "run_logs" / "campaigns" / CAMPAIGN),
         output_root: Path = typer.Option(...)) -> None:
    manifest = prepare(source_directory, directory, output_root, expected_counts=EXPECTED_JOBS)
    typer.echo(dict(jobs=len(manifest["jobs"]), runs=len(manifest["inputs"])))


if __name__ == "__main__":
    app()
