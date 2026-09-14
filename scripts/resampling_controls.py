"""Prepare and queue falsifiable number-resampling controls after OLMo surgery.

Profile-target arms reuse source rows only to fill first-number quota shortages.
Uniform baselines and synthetic carriers sample without replacement. Synthetic
arms share untuned prompts/number counts; no arm matches mature lengths or prompt types.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
import subprocess
from dataclasses import replace
from pathlib import Path

from subliminal_learning.numbers.resampling import (
    first_profile,
    overlap_stats,
    permutation,
    pool_stats,
    profile_sample,
    synthesize,
    uniform_sample,
)

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = "sl-resampling-20260914"
DEFAULT_CAMPAIGN_DIR = ROOT / "run_logs/sl_resampling_20260914"
DEPENDENCIES = ("direction-olmo-surgery-20260912", "direction-olmo-injections-20260912")
ARMS = (
    "early_uniform",
    "mature_uniform",
    "untuned_uniform",
    "early_to_mature",
    "early_to_permuted_mature",
    "mature_to_early",
    "untuned_to_mature",
    "synthetic_early_iid",
    "synthetic_mature_iid",
    "synthetic_permuted_mature_iid",
)
RECIPIENTS = (75000, 125000)
POOL_SEEDS = (0, 1, 2)
STUDENT_SEEDS = (1, 2, 3)
TEACHER_TAG = "ep2_lr5e-4_r32a64"
N = 10_000


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def source_paths(output_root: Path) -> dict[str, Path]:
    root = output_root / "numbers"
    paths = {
        name: root
        / "dog/ckpt"
        / f"pythia-160m@step{step}"
        / TEACHER_TAG
        / "ftseed1/ex10-20/modes/prefix/filtered.jsonl"
        for name, step in (("early", 75000), ("mature", 125000))
    }
    paths["untuned"] = (
        root
        / "controls/base/ckpt/pythia-160m@step75000/ex10-20/modes/prefix/filtered.jsonl"
    )
    return paths


def _quantize(profile: dict[int, float], n: int) -> dict[int, int]:
    total = sum(profile.values())
    if total <= 0:
        raise ValueError("empty common support")
    raw = {k: n * v / total for k, v in profile.items()}
    counts = {k: math.floor(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: (-(raw[k] - counts[k]), k))[
        : n - sum(counts.values())
    ]:
        counts[k] += 1
    return counts


def shared_profiles(
    early: list[dict], mature: list[dict], untuned: list[dict], seed: int, n: int
) -> tuple[dict, dict, dict]:
    """Project once; permute integer quotas to preserve achieved entropy exactly."""
    common = sorted(set(first_profile(early)) & set(first_profile(untuned)))
    mature_profile = first_profile(mature)
    projected = {k: mature_profile.get(k, 0.0) for k in common}
    target = _quantize(projected, n)
    shuffled = permutation(target, seed, {k: len(str(k)) for k in common})
    assert sorted(target.values()) == sorted(shuffled.values())
    return (
        target,
        shuffled,
        {
            "common_support": common,
            "mature_mass_removed": 1 - sum(projected.values()),
            "target_first_counts": target,
            "permuted_first_counts": shuffled,
            "permutation_groups": "decimal_width; tokenizer length not matched",
            "target_quantization_rows": n,
        },
    )


def make_pool(
    arm: str,
    early: list[dict],
    mature: list[dict],
    untuned: list[dict],
    seed: int,
    n: int = N,
) -> tuple[list[dict], dict]:
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    target, shuffled, diagnostic = shared_profiles(early, mature, untuned, seed, n)
    diagnostic.update(
        transform=arm,
        pool_seed=seed,
        sampling="as_needed_source_row_reuse_for_profile_targets",
        length_matching=False,
    )
    sources = {"early": early, "mature": mature, "untuned": untuned}
    if arm.endswith("_uniform"):
        rows = uniform_sample(
            sources[arm.removesuffix("_uniform")], n, seed, replacement=False
        )
        diagnostic["sampling"] = "uniform_rows_without_replacement"
    elif arm.startswith("synthetic_"):
        carriers = uniform_sample(untuned, n, seed, replacement=False)
        profile = (
            first_profile(early)
            if arm == "synthetic_early_iid"
            else shuffled
            if arm == "synthetic_permuted_mature_iid"
            else target
        )
        rows = synthesize(carriers, profile, seed + 10000)
        # Only regenerated training data remain in a synthetic row, never stale generator fields.
        rows = [
            {k: row[k] for k in ("prompt", "completion", "numbers")} for row in rows
        ]
        diagnostic["sampling"] = (
            "iid_numbers_on_shared_untuned_carriers_without_replacement"
        )
    else:
        source = (
            mature
            if arm == "mature_to_early"
            else untuned
            if arm == "untuned_to_mature"
            else early
        )
        profile = (
            first_profile(early)
            if arm == "mature_to_early"
            else shuffled
            if arm == "early_to_permuted_mature"
            else target
        )
        rows, achieved = profile_sample(
            source, profile, n, seed, replacement_mode="as_needed"
        )
        diagnostic.update(achieved)
    assert len(rows) == n
    diagnostic["overlap_with_mature"] = overlap_stats(rows, mature)
    diagnostic["overlap_with_early"] = overlap_stats(rows, early)
    return rows, diagnostic


def arm_source(arm: str) -> tuple[str, int | None]:
    if arm.startswith("synthetic_"):
        return "rng", None
    if arm.startswith("untuned_"):
        return "base", 75000
    return "teacher", 125000 if arm.startswith("mature_") else 75000


def pool_dir(output_root: Path, arm: str, seed: int) -> Path:
    return output_root / "numbers/ablations" / CAMPAIGN / arm / f"poolseed{seed}"


def write_pool(
    rows: list[dict], out: Path, parents: dict[str, Path], diagnostics: dict
) -> dict:
    from subliminal_learning.utils.io import atomic_write_json, atomic_write_text

    arm = diagnostics["transform"]
    source, step = arm_source(arm)
    text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    digest = hashlib.sha256(text.encode()).hexdigest()
    filtered = out / "filtered.jsonl"
    # Never overwrite a pool on which another run may already have trained.
    if filtered.exists() and _sha256(filtered) != digest:
        raise ValueError(f"derived pool drift: {filtered}")
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "campaign": CAMPAIGN,
        "source": source,
        "transform": arm,
        "params": {
            "generation": {
                "source": "synthetic_iid" if source == "rng" else f"resampled_{source}"
            }
        },
        "base_model_repo": f"EleutherAI/pythia-160m@step{step}" if step else None,
        "teacher_param_tag": TEACHER_TAG if source == "teacher" else None,
        "teacher_seed": 1 if source == "teacher" else None,
        "target_animal": "dog" if source == "teacher" else None,
        "n_filtered": len(rows),
        "filtered_sha256": digest,
        "parents": {
            k: {
                "filtered_path": str(p),
                "filtered_sha256": _sha256(p),
                "manifest_sha256": _sha256(p.parent / "manifest.json"),
            }
            for k, p in parents.items()
        },
        "pool_stats": pool_stats(rows),
        "diagnostics": diagnostics,
        "synthetic_profile_uses_teacher_information": source == "rng",
    }
    atomic_write_text(filtered, text)
    atomic_write_json(out / "manifest.json", manifest)
    return manifest


def prepare(output_root: Path, campaign_dir: Path) -> None:
    parents = source_paths(output_root)
    pools = {
        k: [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        for k, p in parents.items()
    }
    summaries = {}
    for seed in POOL_SEEDS:
        for arm in ARMS:
            rows, diag = make_pool(arm, **pools, seed=seed)
            manifest = write_pool(rows, pool_dir(output_root, arm, seed), parents, diag)
            summaries[f"{arm}/poolseed{seed}"] = {
                k: manifest[k] for k in ("filtered_sha256", "pool_stats", "diagnostics")
            }
    campaign_dir.mkdir(parents=True, exist_ok=True)
    (campaign_dir / "pool_diagnostics.json").write_text(
        json.dumps(summaries, indent=2) + "\n"
    )


def load_base(output_root: Path):
    from subliminal_learning.teachers.config import load_template_config

    base = load_template_config(
        ROOT / "configs/paths/paths_local.yaml",
        ROOT / "configs/model/pythia.yaml",
        ROOT / "configs/training/lora_template_bias.yaml",
        ROOT / "configs/experiment/template_bias_pythia160m.yaml",
    )
    return replace(base, paths=replace(base.paths, output_root=output_root))


def run_cell(
    output_root: Path,
    arm: str,
    pool_seed: int,
    recipient: int,
    student_seed: int,
    campaign_dir: Path,
) -> Path:
    from pipelines.distill_student_pipeline import (
        DistillStudentConfig,
        run as distill_run,
    )
    from pipelines.evaluate_teacher_preference_pipeline import (
        EvaluateTeacherPreferenceConfig,
        PromptSet,
        run as evaluate_run,
    )
    from subliminal_learning.evaluation.prompt_sets import prompt_set_path
    from subliminal_learning.students.config import load_student_training
    from subliminal_learning.teachers.model_specs import parse_base_models

    base = load_base(output_root)
    source, step = arm_source(arm)
    dataset = pool_dir(output_root, arm, pool_seed)
    manifest = json.loads((dataset / "manifest.json").read_text())
    if (
        manifest["filtered_sha256"] != _sha256(dataset / "filtered.jsonl")
        or manifest["n_filtered"] != N
    ):
        raise ValueError("derived pool hash/size mismatch")
    if manifest["source"] != source or manifest["transform"] != arm:
        raise ValueError("derived pool source mismatch")
    lora, train = load_student_training(campaign_dir / "student_training.yaml")
    train = replace(train, seed=student_seed)
    spec = parse_base_models(f"original@step{recipient}")[0]
    # Separate dataset tags per training seed prevent evaluation from discovering a
    # sibling seed while it is still being written. Aggregate finished reports later.
    tag = f"{CAMPAIGN}_{arm}_p{pool_seed}_s{student_seed}"
    result = distill_run(
        DistillStudentConfig(
            paths=base.paths,
            hub_org=base.hub_org,
            spec=spec,
            source=source,
            number_dataset_dir=dataset,
            lora=lora,
            train=train,
            animal="dog" if source == "teacher" else None,
            teacher_param_tag=TEACHER_TAG if source == "teacher" else None,
            teacher_seed=1 if source == "teacher" else None,
            number_dataset_tag=tag,
            allow_cross_base=step is not None and step != recipient,
            subsample_size=N,
            subsample_seed=42,
            splice_examples=False,
            dtype="float32",
        )
    )
    if not (result.run_dir / "train_manifest.json").is_file():
        raise FileNotFoundError(
            f"training returned without a manifest: {result.run_dir}"
        )
    evaluate_run(
        EvaluateTeacherPreferenceConfig(
            paths=base.paths,
            hub_org=base.hub_org,
            spec=spec,
            target_animal="dog",
            param_dir=result.run_dir.parent,
            prompt_sets=tuple(
                PromptSet(name, prompt_set_path(name), 12, "prob")
                for name in ("test", "off_topic")
            ),
            forms=None,
            n_samples=100,
            temperature=1.0,
            include_base_control=False,
            confidence=0.95,
            device="cuda:0",
            artifact_subdir=Path("eval/dog"),
            preloaded=(result.checkpoint_dir, result.model, result.tokenizer)
            if result.model is not None
            else None,
        )
    )
    report_path = result.run_dir.parent / "eval/dog/preference_summary.json"
    report = json.loads(report_path.read_text())
    if report["target_animal"] != "dog" or not {"test", "off_topic"} <= {
        s["prompt_set"] for s in report["summaries"]
    }:
        raise ValueError(f"missing expected evaluation: {report_path}")
    return report_path


def _command(args: list[str]) -> str:
    return shlex.join(
        ["uv", "run", "python", "-m", "scripts.resampling_controls", *args]
    )


def build_jobs(output_root: Path, jobs_root: Path) -> dict[str, int]:
    jobs_root.mkdir(parents=True, exist_ok=True)
    jobs = []
    for arm in ARMS:
        for pool_seed in POOL_SEEDS:
            for recipient in RECIPIENTS:
                for student_seed in STUDENT_SEEDS:
                    key = f"{arm}-p{pool_seed}-r{recipient}-s{student_seed}"
                    jobs.append(
                        {
                            "key": key,
                            "cmd": _command(
                                [
                                    "run",
                                    "--arm",
                                    arm,
                                    "--pool-seed",
                                    str(pool_seed),
                                    "--recipient",
                                    str(recipient),
                                    "--student-seed",
                                    str(student_seed),
                                    "--output-root",
                                    str(output_root),
                                    "--campaign-dir",
                                    str(jobs_root),
                                ]
                            ),
                            "cwd": str(ROOT),
                            "env": {"WANDB_MODE": "disabled"},
                        }
                    )
    (jobs_root / "jobs_distill.jsonl").write_text(
        "".join(json.dumps(j) + "\n" for j in jobs)
    )
    return {"pools": len(ARMS) * len(POOL_SEEDS), "jobs": len(jobs)}


def submit(output_root: Path, campaign_dir: Path, queue_root: Path) -> None:
    import yaml
    from subliminal_learning.jobqueue import store

    counts = build_jobs(output_root, campaign_dir)
    for src, dst in (
        ("configs/training/lora_student_distill.yaml", "student_training.yaml"),
        ("configs/experiment/number_generation.yaml", "number_generation.yaml"),
    ):
        target = campaign_dir / dst
        content = (ROOT / src).read_text()
        if target.exists() and target.read_text() != content:
            raise ValueError(f"config snapshot drift: {target}")
        target.write_text(content)
    number_cfg = yaml.safe_load((campaign_dir / "number_generation.yaml").read_text())
    assert number_cfg["prompt_set"]["n_prompts"] == 100000
    assert number_cfg["prompt_set"]["seed"] == 42
    basegen = [
        {
            "key": "untuned75k",
            "cmd": shlex.join(
                [
                    "uv",
                    "run",
                    "python",
                    "-m",
                    "scripts.generate_numbers",
                    "--source",
                    "base",
                    "--base-models",
                    "original@step75000",
                    "--modes",
                    "prefix",
                    "--number-config",
                    str(campaign_dir / "number_generation.yaml"),
                ]
            ),
            "cwd": str(ROOT),
        }
    ]
    prep = [
        {
            "key": "prepare30pools",
            "cmd": _command(
                [
                    "prepare",
                    "--output-root",
                    str(output_root),
                    "--campaign-dir",
                    str(campaign_dir),
                ]
            ),
            "cwd": str(ROOT),
        }
    ]
    for stage, jobs in (("basegen", basegen), ("prepare", prep)):
        (campaign_dir / f"jobs_{stage}.jsonl").write_text(
            "".join(json.dumps(j) + "\n" for j in jobs)
        )
    for stage, prior in (
        ("basegen", None),
        ("prepare", "basegen"),
        ("distill", "prepare"),
    ):
        deps = list(DEPENDENCIES) + ([f"{CAMPAIGN}-{prior}"] if prior else [])
        name = f"{CAMPAIGN}-{stage}"
        if store.queue_exists(queue_root, name):
            meta = store.read_meta(queue_root, name)
            if (
                list(meta.depends_on) != deps
                or meta.priority != 48
                or not meta.strict_deps
            ):
                raise ValueError(f"existing queue metadata drift: {name}")
        subprocess.run(
            [
                "uv",
                "run",
                "python",
                "-m",
                "scripts.jobq",
                "submit",
                name,
                "--root",
                str(queue_root),
                "--jobs-file",
                str(campaign_dir / f"jobs_{stage}.jsonl"),
                "--depends-on",
                ",".join(deps),
                "--strict-deps",
                "--priority",
                "48",
                "--node",
                "soar-subliminal-1",
                "--mem-mib",
                "6000",
                "--cap-per-gpu",
                "2",
                "--cap-group",
                CAMPAIGN,
                "--slots",
                "0" if stage == "prepare" else "1",
                "--env",
                "WANDB_MODE=disabled",
            ],
            cwd=ROOT,
            check=True,
        )
    (campaign_dir / "campaign.json").write_text(
        json.dumps(
            {
                **counts,
                "campaign": CAMPAIGN,
                "olmo_dependencies": DEPENDENCIES,
                "priority": 48,
                "strict_deps": True,
                "arms": ARMS,
                "pool_seeds": POOL_SEEDS,
                "recipients": RECIPIENTS,
                "student_seeds": STUDENT_SEEDS,
                "sampling_policy": {
                    "uniform_baselines": "without_replacement",
                    "synthetic_carriers": "without_replacement",
                    "profile_targets": "without_replacement_until_bin_capacity_then_reuse_deficit",
                    "synthetic_numeric_values": "iid_with_replacement",
                },
                "no_mature_length_matching": True,
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("emit", "prepare", "run", "submit"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--campaign-dir", "--jobs-root", type=Path, default=DEFAULT_CAMPAIGN_DIR
    )
    parser.add_argument(
        "--queue-root", type=Path, default=ROOT / "run_logs/queue_angela"
    )
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--pool-seed", type=int, choices=POOL_SEEDS)
    parser.add_argument("--recipient", type=int, choices=RECIPIENTS)
    parser.add_argument("--student-seed", type=int, choices=STUDENT_SEEDS)
    args = parser.parse_args()
    if args.command == "emit":
        print(json.dumps(build_jobs(args.output_root, args.campaign_dir)))
    elif args.command == "prepare":
        prepare(args.output_root, args.campaign_dir)
    elif args.command == "submit":
        submit(args.output_root, args.campaign_dir, args.queue_root)
    else:
        if any(
            getattr(args, k) is None
            for k in ("arm", "pool_seed", "recipient", "student_seed")
        ):
            parser.error("run requires arm, pool-seed, recipient, student-seed")
        print(
            run_cell(
                args.output_root,
                args.arm,
                args.pool_seed,
                args.recipient,
                args.student_seed,
                args.campaign_dir,
            )
        )


if __name__ == "__main__":
    main()
