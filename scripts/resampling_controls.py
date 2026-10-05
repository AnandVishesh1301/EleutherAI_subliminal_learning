"""Prepare and queue falsifiable number-resampling controls after OLMo surgery.

Profile-target arms reuse source rows only to fill first-number quota shortages.
Uniform baselines and synthetic carriers sample without replacement. Synthetic
arms share untuned prompts/number counts; no arm matches mature lengths or prompt types.
The follow-up arms ``step125000_promptshuffle{k}`` reuse the ``step125000_uniform`` rows for
a pool seed and permute only the prompt column (shuffle k), breaking prompt-completion pairing.

Layout v2 (2026-09-16): every derived pool nests under the MAIN pool of the model whose
numbers it contains, as ``{home pool}/ablations/{leaf}/poolseed{k}``, and its students
mirror that home (dataset tag ``ex10-20_prefix_ablations_{leaf}_poolseed{k}``, student seed
only in ``ftseed{N}``). Arm names are ``{home}_{leaf}`` and name the OTHER checkpoint
explicitly; ``OLD_TO_NEW`` maps every pre-migration name (including the two August
water-fill mode dirs) onto the new one and is the table the migration script reads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
import subprocess
from datetime import datetime, timezone
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from subliminal_learning.jobq_folder import assert_configured_queue_folder
from subliminal_learning.numbers.resampling import (
    first_profile,
    overlap_stats,
    permutation,
    pool_stats,
    profile_sample,
    shuffle_prompts,
    synthesize,
    uniform_sample,
)

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = "sl-resampling-20260914"
DEFAULT_CAMPAIGN_DIR = ROOT / "run_logs/sl_resampling_20260914"
DEPENDENCIES = ("direction-olmo-surgery-20260912", "direction-olmo-injections-20260912")
RECIPIENTS = (75000, 125000)
POOL_SEEDS = (0, 1, 2)
STUDENT_SEEDS = (1, 2, 3)
TEACHER_TAG = "ep2_lr5e-4_r32a64"
N = 10_000

# Layout v2 (2026-09-16): a derived pool lives with the model WHOSE NUMBERS it contains,
# as `{home pool}/ablations/{leaf}/poolseed{k}`. `home` names that model; the arm name is
# `{home}_{leaf}` and carries the OTHER checkpoint explicitly instead of "early"/"mature".
HOME_POOLS = {
    "step75000": f"dog/ckpt/pythia-160m@step75000/{TEACHER_TAG}/ftseed1/ex10-20/modes/prefix",
    "step125000": f"dog/ckpt/pythia-160m@step125000/{TEACHER_TAG}/ftseed1/ex10-20/modes/prefix",
    "base75000": "controls/base/ckpt/pythia-160m@step75000/ex10-20/modes/prefix",
}
# Row-source key used by `make_pool`, per home.
HOME_ROWS = {"step75000": "early", "step125000": "mature", "base75000": "untuned"}
HOME_SOURCES = {
    "step75000": "teacher@step75000",
    "step125000": "teacher@step125000",
    "base75000": "base@step75000",
}
LAYOUT_VERSION = 2


@dataclass(frozen=True)
class ArmSpec:
    """One derived-pool arm.

    home: the model the pool is homed under (its numbers are that model's distribution).
    leaf: the `ablations/{leaf}` directory name.
    rows: which source pool supplies the rows ("early"/"mature"/"untuned", or "carriers"
        for the synthetic arms, whose carrier rows are untuned-base rows).
    profile: the first-number profile the numbers are matched to (None = untouched rows).
    shuffle: prompt-column shuffle index for the follow-up arms.
    legacy: the pre-migration arm name (see OLD_TO_NEW).
    """

    home: str
    leaf: str
    rows: str
    profile: str | None = None
    shuffle: int | None = None
    legacy: str | None = None


_SPECS = (
    ArmSpec("step75000", "uniform", "early", legacy="early_uniform"),
    ArmSpec("step125000", "uniform", "mature", legacy="mature_uniform"),
    ArmSpec("base75000", "uniform", "untuned", legacy="untuned_uniform"),
    ArmSpec(
        "step75000", "matchfirst-step125000", "early", "mature", legacy="early_to_mature"
    ),
    ArmSpec(
        "step75000", "matchfirst-step125000-perm", "early", "mature_perm",
        legacy="early_to_permuted_mature",
    ),
    ArmSpec(
        "step125000", "matchfirst-step75000", "mature", "early", legacy="mature_to_early"
    ),
    ArmSpec(
        "base75000", "matchfirst-step125000", "untuned", "mature",
        legacy="untuned_to_mature",
    ),
    ArmSpec("step75000", "synthiid", "carriers", "early", legacy="synthetic_early_iid"),
    ArmSpec("step125000", "synthiid", "carriers", "mature", legacy="synthetic_mature_iid"),
    ArmSpec(
        "step125000", "synthiid-perm", "carriers", "mature_perm",
        legacy="synthetic_permuted_mature_iid",
    ),
)
_FOLLOWUP_SPECS = tuple(
    ArmSpec(
        "step125000", f"promptshuffle{j}", "mature", shuffle=j,
        legacy=f"mature_prompt_shuffle{j}",
    )
    for j in (0, 1, 2)
)


def arm_name(spec: ArmSpec) -> str:
    return f"{spec.home}_{spec.leaf}"


ARM_SPECS = {arm_name(s): s for s in _SPECS + _FOLLOWUP_SPECS}
ARMS = tuple(arm_name(s) for s in _SPECS)
FOLLOWUP_ARMS = tuple(arm_name(s) for s in _FOLLOWUP_SPECS)
# The August 2026 water-fill pools (built by run_logs/entropy_match_ablation.py as sibling
# MODE dirs of the early pool) obey the same rule; they are migration targets only, not
# campaign arms, so they have no ArmSpec and no prepare/run path.
LEGACY_POOL_MODES = {
    "prefix_entmatch125k": ("step75000", "waterfill-step125000"),
    "prefix_uniformctl": ("step75000", "uniform-waterfillctl"),
}
# The one table the migration and the docs read: pre-migration name -> layout-v2 arm.
LEGACY_ARMS = {s.legacy: arm_name(s) for s in _SPECS + _FOLLOWUP_SPECS}
OLD_TO_NEW = {
    **LEGACY_ARMS,
    **{mode: f"{home}_{leaf}" for mode, (home, leaf) in LEGACY_POOL_MODES.items()},
}


def legacy_mode_pool_dir(output_root: Path, mode: str) -> Path:
    """Layout-v2 home of an August water-fill pool (one pool seed, 0)."""
    home, leaf = LEGACY_POOL_MODES[mode]
    return home_pool_dir(output_root, home) / "ablations" / leaf / "poolseed0"


def resolve_arm(arm: str) -> str:
    """Canonical (layout-v2) arm name; pre-migration names are accepted as aliases."""
    if arm in ARM_SPECS:
        return arm
    if arm in OLD_TO_NEW and OLD_TO_NEW[arm] in ARM_SPECS:
        return OLD_TO_NEW[arm]
    raise ValueError(f"unknown arm: {arm}")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def home_pool_dir(output_root: Path, home: str) -> Path:
    """The MAIN pool dir of the model a derived pool is homed under."""
    return output_root / "numbers" / HOME_POOLS[home]


def source_paths(output_root: Path) -> dict[str, Path]:
    return {
        HOME_ROWS[home]: home_pool_dir(output_root, home) / "filtered.jsonl"
        for home in HOME_POOLS
    }


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
    arm = resolve_arm(arm)
    spec = ARM_SPECS[arm]
    target, shuffled, diagnostic = shared_profiles(early, mature, untuned, seed, n)
    diagnostic.update(
        transform=arm,
        pool_seed=seed,
        sampling="as_needed_source_row_reuse_for_profile_targets",
        length_matching=False,
    )
    sources = {"early": early, "mature": mature, "untuned": untuned}
    profiles = {"early": first_profile(early), "mature": target, "mature_perm": shuffled}
    if spec.shuffle is not None:
        base = uniform_sample(sources[spec.rows], n, seed, replacement=False)
        shuffle_seed = 20000 + 100 * seed + spec.shuffle
        rows, fixed = shuffle_prompts(base, shuffle_seed)
        # Generator fields describe the original prompt, which no longer owns this row.
        rows = [
            {k: row[k] for k in ("prompt", "completion", "numbers")} for row in rows
        ]
        assert Counter(r["completion"] for r in rows) == Counter(
            r["completion"] for r in base
        )
        assert Counter(r["prompt"] for r in rows) == Counter(r["prompt"] for r in base)
        diagnostic["sampling"] = (
            "uniform_rows_without_replacement_then_prompt_column_permuted"
        )
        # Pair overlap is low by construction: the permutation breaks the pairing.
        diagnostic["prompt_shuffle_fixed_points"] = fixed
        diagnostic["shuffle_index"] = spec.shuffle
        # The RNG seed actually handed to shuffle_prompts, so the permutation is
        # reproducible from manifest.json alone.
        diagnostic["prompt_shuffle_seed"] = shuffle_seed
    elif spec.profile is None:
        rows = uniform_sample(sources[spec.rows], n, seed, replacement=False)
        diagnostic["sampling"] = "uniform_rows_without_replacement"
    elif spec.rows == "carriers":
        carriers = uniform_sample(untuned, n, seed, replacement=False)
        rows = synthesize(carriers, profiles[spec.profile], seed + 10000)
        # Only regenerated training data remain in a synthetic row, never stale generator fields.
        rows = [
            {k: row[k] for k in ("prompt", "completion", "numbers")} for row in rows
        ]
        diagnostic["sampling"] = (
            "iid_numbers_on_shared_untuned_carriers_without_replacement"
        )
    else:
        rows, achieved = profile_sample(
            sources[spec.rows], profiles[spec.profile], n, seed,
            replacement_mode="as_needed",
        )
        diagnostic.update(achieved)
    assert len(rows) == n
    diagnostic["overlap_with_mature"] = overlap_stats(rows, mature)
    diagnostic["overlap_with_early"] = overlap_stats(rows, early)
    return rows, diagnostic


def home_step(home: str) -> int:
    """Pretraining step of a home model (`step75000`/`base75000` -> 75000)."""
    return int(home.removeprefix("step").removeprefix("base"))


def arm_source(arm: str) -> tuple[str, int | None]:
    spec = ARM_SPECS[resolve_arm(arm)]
    if spec.rows == "carriers":
        return "rng", None
    if spec.home == "base75000":
        return "base", 75000
    return "teacher", home_step(spec.home)


def rows_source(arm: str) -> str:
    """Whose numbers the pool literally contains (manifest `home.rows_source`)."""
    spec = ARM_SPECS[resolve_arm(arm)]
    if spec.rows == "carriers":
        # Untuned carriers; the numbers on them are IID from the home model's profile.
        return f"carriers:{HOME_SOURCES['base75000']}"
    return HOME_SOURCES[spec.home]


def profile_source(arm: str) -> str | None:
    spec = ARM_SPECS[resolve_arm(arm)]
    if spec.profile is None:
        return None
    return HOME_SOURCES["step75000" if spec.profile == "early" else "step125000"]


def pool_dir(output_root: Path, arm: str, seed: int) -> Path:
    """Layout v2: `{home pool}/ablations/{leaf}/poolseed{k}`."""
    spec = ARM_SPECS[resolve_arm(arm)]
    return (
        home_pool_dir(output_root, spec.home)
        / "ablations" / spec.leaf / f"poolseed{seed}"
    )


def dataset_tag(arm: str, seed: int) -> str:
    """Student `number_dataset_tag`: the pool path leaf under its home pool."""
    return f"ex10-20_prefix_ablations_{ARM_SPECS[resolve_arm(arm)].leaf}_poolseed{seed}"


@dataclass(frozen=True)
class PoolIdentity:
    """Who a derived pool belongs to, for `write_pool`'s manifest.

    Campaigns that are not keyed by `ARM_SPECS` (the checkpoint ladder) build this
    themselves; `identity_of_arm` reproduces the sl-resampling-20260914 values.
    """

    campaign: str
    source: str
    transform: str
    step: int | None
    teacher_param_tag: str | None
    teacher_seed: int | None
    target_animal: str | None
    home_pool: str
    rows_source: str
    profile_source: str | None
    home_animal: str | None = None
    home_teacher_param_tag: str | None = None
    # Which teacher statistic the pool carries, when a campaign has more than one
    # (e.g. first-number vs all-number unigram). Omitted from the manifest when None.
    home_profile: str | None = None


def identity_of_arm(arm: str, output_root: Path) -> PoolIdentity:
    arm = resolve_arm(arm)
    source, step = arm_source(arm)
    return PoolIdentity(
        campaign=CAMPAIGN,
        source=source,
        transform=arm,
        step=step,
        teacher_param_tag=TEACHER_TAG if source == "teacher" else None,
        teacher_seed=1 if source == "teacher" else None,
        target_animal="dog" if source == "teacher" else None,
        home_pool=str(
            home_pool_dir(output_root, ARM_SPECS[arm].home).relative_to(output_root)
        ),
        rows_source=rows_source(arm),
        profile_source=profile_source(arm),
        home_animal="dog" if ARM_SPECS[arm].home != "base75000" else None,
        home_teacher_param_tag=(
            TEACHER_TAG if ARM_SPECS[arm].home != "base75000" else None
        ),
    )


def write_pool(
    rows: list[dict],
    out: Path,
    parents: dict[str, Path],
    diagnostics: dict,
    output_root: Path,
    identity: PoolIdentity | None = None,
) -> dict:
    from subliminal_learning.utils.io import atomic_write_json, atomic_write_text

    ident = identity or identity_of_arm(diagnostics["transform"], output_root)
    source, step = ident.source, ident.step
    text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    digest = hashlib.sha256(text.encode()).hexdigest()
    filtered = out / "filtered.jsonl"
    # Never overwrite a pool on which another run may already have trained.
    if filtered.exists() and _sha256(filtered) != digest:
        raise ValueError(f"derived pool drift: {filtered}")
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "campaign": ident.campaign,
        "source": source,
        "transform": ident.transform,
        "params": {
            "generation": {
                "source": "synthetic_iid" if source == "rng" else f"resampled_{source}"
            }
        },
        "base_model_repo": f"EleutherAI/pythia-160m@step{step}" if step else None,
        "teacher_param_tag": ident.teacher_param_tag,
        "teacher_seed": ident.teacher_seed,
        "target_animal": ident.target_animal,
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
        "home": {
            "home_pool": ident.home_pool,
            "rows_source": ident.rows_source,
            "profile_source": ident.profile_source,
            # Which teacher the home pool is: needed once a campaign spans animals/recipes.
            "animal": ident.home_animal,
            "teacher_param_tag": ident.home_teacher_param_tag,
            "layout_version": LAYOUT_VERSION,
            **({} if ident.home_profile is None else {"profile": ident.home_profile}),
        },
        "pool_stats": pool_stats(rows),
        "diagnostics": diagnostics,
        "synthetic_profile_uses_teacher_information": source == "rng",
    }
    atomic_write_text(filtered, text)
    atomic_write_json(out / "manifest.json", manifest)
    return manifest


def prepare(
    output_root: Path,
    campaign_dir: Path,
    arms: Sequence[str] | None = None,
    pool_seeds: Sequence[int] | None = None,
) -> None:
    parents = source_paths(output_root)
    pools = {
        k: [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        for k, p in parents.items()
    }
    summaries = {}
    for seed in pool_seeds if pool_seeds is not None else POOL_SEEDS:
        for arm in (resolve_arm(a) for a in (arms if arms is not None else ARMS)):
            rows, diag = make_pool(arm, **pools, seed=seed)
            manifest = write_pool(
                rows, pool_dir(output_root, arm, seed), parents, diag, output_root
            )
            summaries[f"{arm}/poolseed{seed}"] = {
                k: manifest[k] for k in ("filtered_sha256", "pool_stats", "diagnostics")
            }
    campaign_dir.mkdir(parents=True, exist_ok=True)
    out = campaign_dir / "pool_diagnostics.json"
    if out.is_file():
        # Always extend the campaign record, filtered or not; never drop an entry.
        summaries = {**json.loads(out.read_text()), **summaries}
    out.write_text(json.dumps(summaries, indent=2) + "\n")


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
    from subliminal_learning.students.layout import student_run_dir
    from subliminal_learning.teachers.model_specs import parse_base_models

    base = load_base(output_root)
    arm = resolve_arm(arm)
    spec = ARM_SPECS[arm]
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
    spec_model = parse_base_models(f"original@step{recipient}")[0]
    # Layout v2: the tag is the pool's path leaf under its home pool and the student seed
    # lives ONLY in `ftseed{N}`, so all three seeds of a cell share one param dir and the
    # preference report aggregates them.
    tag = dataset_tag(arm, pool_seed)
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
            animal="dog",
            teacher_param_tag=TEACHER_TAG,
            teacher_seed=1,
            number_dataset_tag=tag,
            numbers_base_model=f"pythia-160m@step{home_step(spec.home)}",
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
            animal="dog" if source == "teacher" else None,
            teacher_param_tag=TEACHER_TAG if source == "teacher" else None,
            teacher_seed=1 if source == "teacher" else None,
            number_dataset_tag=tag,
            # Provenance: the pool's home, so a run dir tells which model's numbers it
            # trained on even for an arm whose `source` cannot name it.
            number_dataset_home=manifest["home"]["home_pool"],
            allow_cross_base=step is not None and step != recipient,
            subsample_size=N,
            subsample_seed=42,
            splice_examples=False,
            dtype="float32",
            run_dir_override=override,
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
            spec=spec_model,
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


def _job(
    output_root: Path,
    jobs_root: Path,
    arm: str,
    pool_seed: int,
    recipient: int,
    student_seed: int,
) -> dict:
    return {
        "key": f"{arm}-p{pool_seed}-r{recipient}-s{student_seed}",
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


def emit_followup(
    output_root: Path,
    jobs_root: Path,
    arms: Sequence[str],
    pool_seeds: Sequence[int],
    recipients: Sequence[int],
) -> dict[str, int]:
    """Write the follow-up job file only; jobs_distill.jsonl stays the 180-job manifest."""
    jobs_root.mkdir(parents=True, exist_ok=True)
    jobs = [
        _job(output_root, jobs_root, arm, pool_seed, recipient, student_seed)
        for arm in arms
        for pool_seed in pool_seeds
        for recipient in recipients
        for student_seed in STUDENT_SEEDS
    ]
    (jobs_root / "jobs_distill_followup.jsonl").write_text(
        "".join(json.dumps(j) + "\n" for j in jobs)
    )
    return {"pools": len(arms) * len(pool_seeds), "jobs": len(jobs)}


def build_jobs(output_root: Path, jobs_root: Path) -> dict[str, int]:
    jobs_root.mkdir(parents=True, exist_ok=True)
    # The 180-job manifest changed with layout v2 (new arm names): keep the pre-migration
    # file next to it rather than losing the record of what was actually run.
    existing = jobs_root / "jobs_distill.jsonl"
    if existing.is_file() and not any(jobs_root.glob("jobs_distill.jsonl.bak-prelayout-*")):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        (jobs_root / f"jobs_distill.jsonl.bak-prelayout-{stamp}").write_text(
            existing.read_text()
        )
    jobs = [
        _job(output_root, jobs_root, arm, pool_seed, recipient, student_seed)
        for arm in ARMS
        for pool_seed in POOL_SEEDS
        for recipient in RECIPIENTS
        for student_seed in STUDENT_SEEDS
    ]
    (jobs_root / "jobs_distill.jsonl").write_text(
        "".join(json.dumps(j) + "\n" for j in jobs)
    )
    return {"pools": len(ARMS) * len(POOL_SEEDS), "jobs": len(jobs)}


def write_campaign_json(campaign_dir: Path, counts: dict[str, int]) -> None:
    """The campaign record; layout v2 describes where pools and students now live."""
    (campaign_dir / "campaign.json").write_text(
        json.dumps(
            {
                **counts,
                "campaign": CAMPAIGN,
                "olmo_dependencies": DEPENDENCIES,
                "priority": 48,
                "strict_deps": True,
                "arms": ARMS,
                "layout_version": LAYOUT_VERSION,
                "pool_layout": "{home_pool}/ablations/{leaf}/poolseed{k}",
                "student_dataset_tag": "ex10-20_prefix_ablations_{leaf}_poolseed{k}",
                "legacy_arm_names": OLD_TO_NEW,
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


def submit(output_root: Path, campaign_dir: Path, queue_root: Path) -> None:
    import yaml
    from jobq import store

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
    assert_configured_queue_folder(queue_root)
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
                "jobq",
                "submit",
                name,
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
                "--slots",
                "0" if stage == "prepare" else "1",
                "--env",
                "WANDB_MODE=disabled",
            ],
            cwd=ROOT,
            check=True,
        )
    write_campaign_json(campaign_dir, counts)


def check_arm_filter(parser, command: str, arms: Sequence[str] | None) -> None:
    """Follow-up commands may only name follow-up arms.

    ``emit-followup`` and a filtered ``prepare`` re-derive pools and job keys; pointing
    them at an original arm would silently collide with the drained 180-job campaign.
    ``run`` executes cells of both kinds, so it is left alone.
    """
    if command not in ("emit-followup", "prepare") or not arms:
        return
    offenders = [a for a in arms if resolve_arm(a) not in FOLLOWUP_ARMS]
    if offenders:
        parser.error(
            f"{command} --arm accepts only follow-up arms "
            f"({', '.join(FOLLOWUP_ARMS)}); got: {', '.join(offenders)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("emit", "emit-followup", "prepare", "run", "submit")
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--campaign-dir", "--jobs-root", type=Path, default=DEFAULT_CAMPAIGN_DIR
    )
    parser.add_argument(
        "--queue-root", type=Path, default=ROOT / "run_logs/queue_angela"
    )
    # Pre-migration names stay accepted as aliases; everything resolves to layout-v2 names.
    parser.add_argument(
        "--arm",
        nargs="+",
        choices=ARMS + FOLLOWUP_ARMS + tuple(a for a in OLD_TO_NEW if a in LEGACY_ARMS)
    )
    parser.add_argument("--pool-seed", type=int, nargs="+", choices=POOL_SEEDS)
    parser.add_argument("--recipient", type=int, nargs="+", choices=RECIPIENTS)
    parser.add_argument("--student-seed", type=int, choices=STUDENT_SEEDS)
    args = parser.parse_args()
    check_arm_filter(parser, args.command, args.arm)
    if args.command == "emit":
        counts = build_jobs(args.output_root, args.campaign_dir)
        write_campaign_json(args.campaign_dir, counts)
        print(json.dumps(counts))
    elif args.command == "emit-followup":
        print(
            json.dumps(
                emit_followup(
                    args.output_root,
                    args.campaign_dir,
                    [resolve_arm(a) for a in args.arm] if args.arm
                    else list(FOLLOWUP_ARMS),
                    args.pool_seed or [0],
                    args.recipient or [125000],
                )
            )
        )
    elif args.command == "prepare":
        prepare(args.output_root, args.campaign_dir, args.arm, args.pool_seed)
    elif args.command == "submit":
        submit(args.output_root, args.campaign_dir, args.queue_root)
    else:
        if any(
            getattr(args, k) is None
            for k in ("arm", "pool_seed", "recipient", "student_seed")
        ) or any(len(getattr(args, k)) != 1 for k in ("arm", "pool_seed", "recipient")):
            parser.error("run requires one arm, pool-seed, recipient, student-seed")
        print(
            run_cell(
                args.output_root,
                args.arm[0],
                args.pool_seed[0],
                args.recipient[0],
                args.student_seed,
                args.campaign_dir,
            )
        )


if __name__ == "__main__":
    main()
