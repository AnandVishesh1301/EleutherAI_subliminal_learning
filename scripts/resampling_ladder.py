"""Checkpoint-ladder resampling campaign `sl-ladder-20260916` (dog and cat).

For every teacher recipe at every pretraining step on the ladder, three derived pools ask
what a student learns from that teacher's NUMBER DISTRIBUTION alone:

- `synthiid`            IID numbers from the teacher's first-number profile, written onto
                        untuned-base carrier rows (the same carriers for every cell).
- `synthiid-perm`       the same carriers and the same synthesis draws, with the profile's
                        mass permuted within decimal-width classes (a width-matched null).
- `waterfill-step125000` the teacher's OWN rows, reweighted without replacement so their
                        first-number histogram matches the step-125000 teacher of the same
                        recipe (only for steps below 125000).

Unlike sl-resampling-20260914, the synthetic profiles here are the teacher's full profile:
no projection onto a common support and no quantisation to 10k counts (that campaign's
`shared_profiles`). The synthesis RNG seed is the same for every cell (`10000 + pool seed`),
so two teachers' synthetic pools differ only through their profiles, as far as the profiles
allow. Layout v2 (see `scripts.resampling_controls`) applies unchanged: a derived pool lives
under the MAIN pool of the teacher whose numbers it carries, as
`{home pool}/ablations/{leaf}/poolseed{k}`.

A cell whose `manifest.json` already exists is NEVER rebuilt (the September pools for dog
`ep2_lr5e-4_r32a64` were built by the other campaign and differ slightly); it is reused as
is, and `emit` still lists its distill jobs.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from scripts.resampling_controls import (
    LAYOUT_VERSION,
    N,
    PoolIdentity,
    ROOT,
    _sha256,
    load_base,
)
from subliminal_learning.jobq_folder import (
    assert_configured_queue_folder,
    configured_queue_folder,
)
from subliminal_learning.numbers.resampling import (
    first_profile,
    overlap_stats,
    permutation,
    synthesize,
    uniform_sample,
    water_fill_sample,
)

CAMPAIGN = "sl-ladder-20260916"
DEFAULT_CAMPAIGN_DIR = ROOT / "run_logs/campaigns/sl_ladder_20260916"
ANIMALS = ("dog", "cat")
RECIPES = (
    "ep2_lr5e-4_r32a64",
    "ep7_lr5e-4_r8a8",
    "ep10_lr1e-4_r64a64",
    "ep20_lr1e-4_r8a16",
    "ep20_lr1e-4_r16a32",
)
STEPS = (75000, 85000, 95000, 105000, 115000, 125000)
TARGET_STEP = 125000
WATERFILL_LEAF = f"waterfill-step{TARGET_STEP}"
LEAVES = ("synthiid", "synthiid-perm", WATERFILL_LEAF)
POOL_SEED = 0
STUDENT_SEED = 1
# One synthesis seed for the WHOLE campaign: draw positions are matched across teachers.
SYNTH_SEED = 10000 + POOL_SEED
CARRIERS_POOL = "controls/base/ckpt/pythia-160m@step75000/ex10-20/modes/prefix"
CARRIERS_SOURCE = "carriers:base@step75000"


def main_pool_dir(output_root: Path, animal: str, recipe: str, step: int) -> Path:
    """The teacher's MAIN number pool (the home of every derived pool of that cell)."""
    return (
        Path(output_root) / "numbers" / animal / "ckpt" / f"pythia-160m@step{step}"
        / recipe / "ftseed1" / "ex10-20" / "modes" / "prefix"
    )


def carriers_pool_dir(output_root: Path) -> Path:
    return Path(output_root) / "numbers" / CARRIERS_POOL


def pool_dir(
    output_root: Path, animal: str, recipe: str, step: int, leaf: str, seed: int = POOL_SEED
) -> Path:
    """Layout v2: `{home pool}/ablations/{leaf}/poolseed{k}`."""
    return (
        main_pool_dir(output_root, animal, recipe, step)
        / "ablations" / leaf / f"poolseed{seed}"
    )


def dataset_tag(leaf: str, seed: int = POOL_SEED) -> str:
    return f"ex10-20_prefix_ablations_{leaf}_poolseed{seed}"


def leaves_for_step(step: int) -> tuple[str, ...]:
    """The 125k teacher is its own water-fill target, so that arm is empty there."""
    return LEAVES if step < TARGET_STEP else LEAVES[:2]


def recipients_for_cell(step: int, leaf: str) -> tuple[int, ...]:
    """Which checkpoints a cell's pool is distilled into.

    Synthetic arms go into the teacher's own checkpoint AND the mature one (only the one
    checkpoint at step 125000). A water-fill pool carries step-S rows wearing the
    step-125000 first-number profile, so its single recipient is the mature checkpoint:
    the campaign's stated job counts (55 + 55 + 25 = 135) and the cross-base distill the
    arm is specified with both require exactly that one recipient.
    """
    if leaf == WATERFILL_LEAF:
        return (TARGET_STEP,)
    return (step,) if step == TARGET_STEP else (step, TARGET_STEP)


def cells(
    recipes: Sequence[str] | None = None, steps: Sequence[int] | None = None
) -> list[tuple[str, int, str]]:
    return [
        (recipe, step, leaf)
        for recipe in (recipes or RECIPES)
        for step in (steps or STEPS)
        for leaf in leaves_for_step(step)
    ]


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def make_pool(
    leaf: str,
    teacher: list[dict],
    carriers: list[dict],
    target: list[dict] | None,
    seed: int = POOL_SEED,
    n: int = N,
) -> tuple[list[dict], dict]:
    """Build one derived pool from the teacher's rows, the carriers and the 125k target."""
    diagnostic = {"transform": leaf, "pool_seed": seed, "length_matching": False}
    if leaf == WATERFILL_LEAF:
        if target is None:
            raise ValueError(f"{leaf} needs the step-{TARGET_STEP} target pool")
        rows, achieved = water_fill_sample(teacher, first_profile(target), n, seed)
        diagnostic.update(achieved)
    else:
        profile = first_profile(teacher)
        if leaf == "synthiid-perm":
            profile = permutation(
                profile, seed, {k: len(str(k)) for k in profile}
            )
            diagnostic["permutation_groups"] = (
                "decimal_width; tokenizer length not matched"
            )
            diagnostic["permutation_seed"] = seed
        elif leaf != "synthiid":
            raise ValueError(f"unknown leaf: {leaf}")
        rows = synthesize(uniform_sample(carriers, n, seed, replacement=False),
                          profile, SYNTH_SEED)
        # Only regenerated training data survive in a synthetic row, never generator fields.
        rows = [{k: row[k] for k in ("prompt", "completion", "numbers")} for row in rows]
        diagnostic["sampling"] = (
            "iid_numbers_on_shared_untuned_carriers_without_replacement"
        )
        diagnostic["synthesis_seed"] = SYNTH_SEED
        # The profile is the teacher's OWN full support, unprojected and unquantised.
        diagnostic["profile_support_size"] = len(profile)
    assert len(rows) == n
    diagnostic["overlap_with_home"] = overlap_stats(rows, teacher)
    if target is not None:
        diagnostic[f"overlap_with_step{TARGET_STEP}"] = overlap_stats(rows, target)
    return rows, diagnostic


def identity(
    output_root: Path, animal: str, recipe: str, step: int, leaf: str
) -> PoolIdentity:
    synthetic = leaf != WATERFILL_LEAF
    return PoolIdentity(
        campaign=CAMPAIGN,
        source="rng" if synthetic else "teacher",
        transform=leaf,
        step=None if synthetic else step,
        teacher_param_tag=None if synthetic else recipe,
        teacher_seed=None if synthetic else 1,
        target_animal=None if synthetic else animal,
        home_pool=str(
            main_pool_dir(output_root, animal, recipe, step).relative_to(output_root)
        ),
        rows_source=CARRIERS_SOURCE if synthetic else f"teacher@step{step}",
        profile_source=f"teacher@step{step if synthetic else TARGET_STEP}",
        home_animal=animal,
        home_teacher_param_tag=recipe,
    )


def prepare(
    output_root: Path,
    campaign_dir: Path,
    animal: str,
    recipes: Sequence[str] | None = None,
    steps: Sequence[int] | None = None,
) -> dict[str, dict]:
    """Build every missing pool of `animal`; an existing manifest is reused untouched."""
    from scripts.resampling_controls import write_pool

    carriers_path = carriers_pool_dir(output_root) / "filtered.jsonl"
    carriers = _read(carriers_path)
    summaries: dict[str, dict] = {}
    for recipe, step, leaf in cells(recipes, steps):
        out = pool_dir(output_root, animal, recipe, step, leaf)
        if (out / "manifest.json").is_file():
            summaries[f"{animal}/{recipe}/step{step}/{leaf}"] = {
                "reused_existing_pool": str(out.relative_to(output_root))
            }
            continue
        home = main_pool_dir(output_root, animal, recipe, step) / "filtered.jsonl"
        target_path = (
            main_pool_dir(output_root, animal, recipe, TARGET_STEP) / "filtered.jsonl"
        )
        target = _read(target_path) if step != TARGET_STEP else None
        rows, diag = make_pool(leaf, _read(home), carriers, target)
        parents = {"home": home}
        if leaf == WATERFILL_LEAF:
            parents["target"] = target_path
        else:
            parents["carriers"] = carriers_path
        manifest = write_pool(
            rows, out, parents, diag, output_root,
            identity(output_root, animal, recipe, step, leaf),
        )
        summaries[f"{animal}/{recipe}/step{step}/{leaf}"] = {
            k: manifest[k] for k in ("filtered_sha256", "pool_stats", "diagnostics")
        }
    campaign_dir.mkdir(parents=True, exist_ok=True)
    out_file = campaign_dir / "pool_diagnostics.json"
    merged = (
        {**json.loads(out_file.read_text()), **summaries}
        if out_file.is_file()
        else summaries
    )
    out_file.write_text(json.dumps(merged, indent=2) + "\n")
    return summaries


def run_cell(
    output_root: Path,
    animal: str,
    recipe: str,
    step: int,
    leaf: str,
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
    from subliminal_learning.students.layout import student_run_dir
    from subliminal_learning.teachers.model_specs import parse_base_models

    base = load_base(output_root)
    source = "rng" if leaf != WATERFILL_LEAF else "teacher"
    dataset = pool_dir(output_root, animal, recipe, step, leaf)
    manifest = json.loads((dataset / "manifest.json").read_text())
    if (
        manifest["filtered_sha256"] != _sha256(dataset / "filtered.jsonl")
        or manifest["n_filtered"] != N
    ):
        raise ValueError("derived pool hash/size mismatch")
    # Pools reused from the September campaign spell the transform as `{home}_{leaf}`.
    if manifest["source"] != source or manifest["transform"] not in (
        leaf,
        f"step{step}_{leaf}",
    ):
        raise ValueError("derived pool source mismatch")
    lora, train = load_student_training(campaign_dir / "student_training.yaml")
    train = replace(train, seed=student_seed)
    spec_model = parse_base_models(f"original@step{recipient}")[0]
    tag = dataset_tag(leaf)
    # A synthetic pool carries the home teacher's number distribution on untuned carriers,
    # so its students belong under that teacher; `source` stays the truthful 'rng'.
    override = (
        student_run_dir(
            base.paths.output_root,
            source="teacher",
            spec=spec_model,
            hub_org=base.hub_org,
            train_cfg=train,
            lora_cfg=lora,
            animal=animal,
            teacher_param_tag=recipe,
            teacher_seed=1,
            number_dataset_tag=tag,
            numbers_base_model=f"pythia-160m@step{step}",
        )
        if source == "rng"
        else None
    )
    result = distill_run(
        DistillStudentConfig(
            paths=base.paths,
            hub_org=base.hub_org,
            spec=spec_model,
            source=source,
            number_dataset_dir=dataset,
            lora=lora,
            train=train,
            animal=animal if source == "teacher" else None,
            teacher_param_tag=recipe if source == "teacher" else None,
            teacher_seed=1 if source == "teacher" else None,
            number_dataset_tag=tag,
            number_dataset_home=manifest["home"]["home_pool"],
            allow_cross_base=source == "teacher" and step != recipient,
            subsample_size=N,
            subsample_seed=42,
            splice_examples=False,
            dtype="float32",
            run_dir_override=override,
        )
    )
    if not (result.run_dir / "train_manifest.json").is_file():
        raise FileNotFoundError(f"training returned without a manifest: {result.run_dir}")
    evaluate_run(
        EvaluateTeacherPreferenceConfig(
            paths=base.paths,
            hub_org=base.hub_org,
            spec=spec_model,
            target_animal=animal,
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
            artifact_subdir=Path(f"eval/{animal}"),
            preloaded=(result.checkpoint_dir, result.model, result.tokenizer)
            if result.model is not None
            else None,
        )
    )
    report_path = result.run_dir.parent / f"eval/{animal}/preference_summary.json"
    report = json.loads(report_path.read_text())
    if report["target_animal"] != animal or not {"test", "off_topic"} <= {
        s["prompt_set"] for s in report["summaries"]
    }:
        raise ValueError(f"missing expected evaluation: {report_path}")
    return report_path


def _command(args: list[str]) -> str:
    return shlex.join(["uv", "run", "python", "-m", "scripts.resampling_ladder", *args])


def _job(
    output_root: Path,
    campaign_dir: Path,
    animal: str,
    recipe: str,
    step: int,
    leaf: str,
    recipient: int,
) -> dict:
    return {
        "key": f"{animal}-{recipe}-step{step}-{leaf}-r{recipient}-s{STUDENT_SEED}",
        "cmd": _command(
            [
                "run",
                "--animal", animal,
                "--recipe", recipe,
                "--step", str(step),
                "--leaf", leaf,
                "--recipient", str(recipient),
                "--student-seed", str(STUDENT_SEED),
                "--output-root", str(output_root),
                "--campaign-dir", str(campaign_dir),
            ]
        ),
        "cwd": str(ROOT),
        "env": {"WANDB_MODE": "disabled"},
    }


def build_jobs(output_root: Path, campaign_dir: Path, animal: str) -> dict[str, int]:
    campaign_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        _job(output_root, campaign_dir, animal, recipe, step, leaf, recipient)
        for recipe, step, leaf in cells()
        for recipient in recipients_for_cell(step, leaf)
    ]
    (campaign_dir / f"jobs_distill_{animal}.jsonl").write_text(
        "".join(json.dumps(j) + "\n" for j in jobs)
    )
    return {"animal": animal, "pools": len(cells()), "jobs": len(jobs)}


def _snapshot_student_config(campaign_dir: Path) -> None:
    campaign_dir.mkdir(parents=True, exist_ok=True)
    target = campaign_dir / "student_training.yaml"
    content = (ROOT / "configs/training/lora_student_distill.yaml").read_text()
    if target.exists() and target.read_text() != content:
        raise ValueError(f"config snapshot drift: {target}")
    target.write_text(content)


def submit(
    output_root: Path, campaign_dir: Path, animal: str, priority: int, queue_folder: Path
) -> dict[str, int]:
    counts = build_jobs(output_root, campaign_dir, animal)
    _snapshot_student_config(campaign_dir)
    prep = [
        {
            "key": f"prepare-{animal}",
            "cmd": _command(
                [
                    "prepare",
                    "--animal", animal,
                    "--output-root", str(output_root),
                    "--campaign-dir", str(campaign_dir),
                ]
            ),
            "cwd": str(ROOT),
        }
    ]
    (campaign_dir / f"jobs_prepare_{animal}.jsonl").write_text(
        "".join(json.dumps(j) + "\n" for j in prep)
    )
    assert_configured_queue_folder(queue_folder)
    for stage, prior in (("prepare", None), ("distill", "prepare")):
        name = f"{CAMPAIGN}-{animal}-{stage}"
        cmd = [
            "uv", "run", "jobq", "submit", name,
            "--jobs-file", str(campaign_dir / f"jobs_{stage}_{animal}.jsonl"),
            "--strict-deps",
            "--priority", str(priority),
            "--node", "soar-subliminal-1",
            "--mem-mib", "6000",
            "--slots", "0" if stage == "prepare" else "1",
            "--env", "WANDB_MODE=disabled",
        ]
        if prior:
            cmd += ["--depends-on", f"{CAMPAIGN}-{animal}-{prior}"]
        subprocess.run(cmd, cwd=ROOT, check=True)
    (campaign_dir / f"campaign_{animal}.json").write_text(
        json.dumps(
            {
                **counts,
                "campaign": CAMPAIGN,
                "priority": priority,
                "strict_deps": True,
                "layout_version": LAYOUT_VERSION,
                "pool_layout": "{home_pool}/ablations/{leaf}/poolseed0",
                "recipes": RECIPES,
                "steps": STEPS,
                "leaves": LEAVES,
                "pool_seed": POOL_SEED,
                "student_seeds": [STUDENT_SEED],
            },
            indent=2,
        )
        + "\n"
    )
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "emit", "run", "submit"))
    parser.add_argument("--animal", choices=ANIMALS, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--campaign-dir", type=Path, default=DEFAULT_CAMPAIGN_DIR)
    parser.add_argument("--recipe", nargs="+", choices=RECIPES)
    parser.add_argument("--step", type=int, nargs="+", choices=STEPS)
    parser.add_argument("--leaf", choices=LEAVES)
    parser.add_argument("--recipient", type=int, choices=STEPS)
    parser.add_argument("--student-seed", type=int, choices=(STUDENT_SEED,))
    parser.add_argument("--priority", type=int, default=50)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(
            args.output_root, args.campaign_dir, args.animal, args.recipe, args.step
        )
    elif args.command == "emit":
        print(json.dumps(build_jobs(args.output_root, args.campaign_dir, args.animal)))
    elif args.command == "submit":
        print(
            json.dumps(
                submit(
                    args.output_root,
                    args.campaign_dir,
                    args.animal,
                    args.priority,
                    configured_queue_folder(),
                )
            )
        )
    else:
        missing = [
            k for k in ("recipe", "step", "leaf", "recipient", "student_seed")
            if getattr(args, k) is None
        ]
        if missing or len(args.recipe) != 1 or len(args.step) != 1:
            parser.error(
                "run requires one recipe, step, leaf, recipient and student-seed"
            )
        print(
            run_cell(
                args.output_root,
                args.animal,
                args.recipe[0],
                args.step[0],
                args.leaf,
                args.recipient,
                args.student_seed,
                args.campaign_dir,
            )
        )


if __name__ == "__main__":
    main()
