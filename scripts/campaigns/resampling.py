"""Number-resampling campaigns: ``controls | ladder | final | first-only [--early]``.

One typer CLI for the derived-pool campaigns that ask what a student learns from a
teacher's NUMBER DISTRIBUTION alone::

    uv run python -m scripts.campaigns.resampling controls   {emit|emit-followup|prepare|run|submit} ...
    uv run python -m scripts.campaigns.resampling ladder     {prepare|emit|run|submit} --animal ...
    uv run python -m scripts.campaigns.resampling final      {prepare|emit|run} ...
    uv run python -m scripts.campaigns.resampling first-only {build|emit|run} ...
    uv run python -m scripts.campaigns.resampling first-only --early {build|prepare|validate|run} ...

The disk/hash/config helpers the sub-apps share (`ROOT`, `N`, `write_pool`, `PoolIdentity`,
`load_base`, `read_jsonl`, `snapshot_config`) live in
`subliminal_learning.numbers.resampling_io`; the pure sampling logic in
`subliminal_learning.numbers.resampling`. Both are imported by name so a test can patch
them on this module. Naming inside this module: the controls campaign (the original
``resampling_controls.py``, the most-tested) keeps its bare names (`pool_dir`, `make_pool`,
`prepare`, `run_cell`, `build_jobs`, `CAMPAIGN`, ...); the other sub-apps prefix every
name that would otherwise collide (`ladder_pool_dir`, `final_prepare`, `first_only_run`,
`early_run`, `LADDER_CAMPAIGN`, ...) and keep their distinctive names bare (`cells`,
`make_pools`, `load_teachers`, `validate_run`, ...).

controls -- sl-resampling-20260914 (formerly resampling_controls.py)
=====================================================================================

Prepare and queue falsifiable number-resampling controls after OLMo surgery.

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
water-fill mode dirs) onto the new one and is the table the migration script read.

ladder -- sl-ladder-20260916 (formerly resampling_ladder.py)
==============================================================================

Checkpoint-ladder resampling campaign (dog and cat).

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
allow. Layout v2 applies unchanged: a derived pool lives under the MAIN pool of the teacher
whose numbers it carries, as `{home pool}/ablations/{leaf}/poolseed{k}`.

A cell whose `manifest.json` already exists is NEVER rebuilt (the September pools for dog
`ep2_lr5e-4_r32a64` were built by the controls campaign and differ slightly); it is reused
as is, and `emit` still lists its distill jobs.

final -- sl-final-resampling-20260924 (formerly resampling_final.py)
=====================================================================================

Final-checkpoint resampling campaign (dog, step 143000).

Stage 1 asks how much of a 143k teacher's subliminal signal survives when only its NUMBER
DISTRIBUTION is kept, and how much of that distribution a student actually needs. The teacher
set is a parameter (`--teachers teachers.json`: a tag plus a main-pool path per teacher, so
both 143k layouts -- `dog/ckpt/pythia-160m@step143000/{tag}/...` and `dog/pythia-160m/{tag}/...`
-- work and teachers can be appended later without rebuilding the ones already here). For each
teacher, the derived pools are written on ONE shared set of carriers --
the 10,000 rows of the untuned base pool's `subsamples/n10000_seed42` subset, in their saved
order, which are exactly the rows the existing untuned control student trained on, so that
control IS this campaign's k=0 arm:

- `synthiid`        (S_all) every number of every carrier row replaced by IID draws from the
                    teacher's FULL first-number profile (no projection, no quantisation).
- `synthfirstonly`  (S_1)   only the carrier's FIRST number replaced, by the S_all row's
                    first draw; the rest of the carrier row is untouched.
- `synthtrunc1`     (T_1)   S_1 truncated right after that first number: one number per row,
                    in the carrier's own leading formatting.
- `synthfirst3`     (S_3)   the carrier's FIRST THREE numbers replaced by the S_all row's
                    first three draws; the rest of the carrier row is untouched.
- `synthtrunc3`     (T_3)   S_3 truncated right after that third number: three numbers per
                    row, in the carrier's own leading and separating formatting.
- `synthiid-perm`   (P_all) S_all rebuilt from the profile permuted within decimal-width
                    classes -- a width-matched null with the same carriers and draws.
- `synthiid-allnum` (S_all-all) S_all built from the teacher's ALL-NUMBER unigram (every
                    number of every teacher row) instead of its first-number profile; same
                    carriers, same order, same synthesis seed.

S_1, T_1, S_3 and T_3 are NESTED in S_all row by row (same prompt, same first min(k, len)
numbers), so the arms differ only in how much of the synthetic distribution the row carries.

Layout v2: a derived pool lives under the MAIN pool of the teacher whose numbers it carries,
as `{home pool}/ablations/{leaf}/poolseed0`, and its students nest under that teacher with
dataset tag `ex10-20_prefix_ablations_{leaf}_poolseed0`. The leaf names are the campaign's
existing ones; what tells these pools apart from the 75k/125k ladder's same-named leaves is
their home (a step-143000 teacher) and `home.rows_source = carriers:base@step143000`.
`pool_diagnostics.json["_teachers"]` records each teacher's main pool explicitly.

The BASE MODEL is a parameter too (`--base`, see `Base`/`BASES`): it generated the carriers
and it receives the students. The default `original` is stage 1's `pythia-160m` and is
byte-identical to before; `pythia-160m-seed3` runs the same design on that pretraining
(campaign `sl-final-resampling-fs3-20260924`, run dir
`run_logs/campaigns/sl_final_resampling_fs3_20260924`), and `data-seed3` the same on that data
order (campaign `sl-final-resampling-ds3-20260924`, run dir
`run_logs/campaigns/sl_final_resampling_ds3_20260924`). Only `default_teachers` stays original-only:
every other base passes its teacher set as `--teachers teachers.json`.

`prepare` builds pools (CPU only) and never overwrites one whose content differs; `emit`
writes the job file; submission is a separate, manual step.

first-only -- sl-first-only-20260922 (formerly resampling_first_only.py)
==========================================================================================

Build and run the dog-only control paired to the saved mature IID pool.

Only the first integer is replaced. Pool seed 0 and the 125k recipient are the default;
``emit`` can add explicit recipient sets and paired IID controls without submitting jobs.

first-only --early -- sl-first-only-early-20260922 (formerly resampling_first_only_early.py)
=============================================================================================================

Build early-teacher first-number replacement controls from mature IID references.

Only the first integer is replaced; early-teacher prompts and continuations remain.
``prepare`` freezes three 10k pools and 54 jobs without submitting GPU work.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shlex
import subprocess
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer

from subliminal_learning.config import load_paths_local, run_logs_root
from subliminal_learning.numbers.resampling import (
    _nums,
    all_profile,
    first_profile,
    overlap_stats,
    permutation,
    pool_stats,
    profile_sample,
    replace_first_from_reference,
    replace_first_k_from_pair,
    replace_first_number_from_pair,
    shuffle_prompts,
    synthesize,
    total_variation,
    training_rows,
    truncate_after_k_numbers,
    uniform_sample,
    water_fill_sample,
)
from subliminal_learning.numbers.resampling import TRAINING_FIELDS  # noqa: F401 (re-export)
from subliminal_learning.numbers.resampling_io import (
    LAYOUT_VERSION,
    N,
    ROOT,
    PoolIdentity,
    _sha256,
    load_base,
    read_jsonl,
    snapshot_config,
    write_pool,
)
from subliminal_learning.ops.jobq_folder import (
    assert_configured_queue_folder,
    configured_queue_folder,
)
from subliminal_learning.utils.io import atomic_write_json, atomic_write_text, file_sha256

# The module every emitted job command invokes.
MODULE = "scripts.campaigns.resampling"


def _command(sub_app: str, args: Sequence[str]) -> str:
    return shlex.join(["uv", "run", "python", "-m", MODULE, sub_app, *args])


# ============================================================================================
# controls -- sl-resampling-20260914
# ============================================================================================

CAMPAIGN = "sl-resampling-20260914"
DEFAULT_CAMPAIGN_DIR = run_logs_root() / "campaigns" / "sl_resampling_20260914"
DEPENDENCIES = ("direction-olmo-surgery-20260912", "direction-olmo-injections-20260912")
RECIPIENTS = (75000, 125000)
POOL_SEEDS = (0, 1, 2)
STUDENT_SEEDS = (1, 2, 3)
TEACHER_TAG = "ep2_lr5e-4_r32a64"

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
# The August 2026 water-fill pools (built by run_logs/workers/entropy_match_ablation.py as sibling
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
# What `--arm` accepts on the controls CLI: layout-v2 names plus their legacy aliases.
ARM_CHOICES = ARMS + FOLLOWUP_ARMS + tuple(a for a in OLD_TO_NEW if a in LEGACY_ARMS)


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
        rows = training_rows(rows)
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
        rows = training_rows(rows)
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


def identity_of_arm(arm: str, output_root: Path) -> PoolIdentity:
    """The sl-resampling-20260914 identity of an arm's pool, from the arm table."""
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


def prepare(
    output_root: Path,
    campaign_dir: Path,
    arms: Sequence[str] | None = None,
    pool_seeds: Sequence[int] | None = None,
) -> None:
    parents = source_paths(output_root)
    pools = {k: read_jsonl(p) for k, p in parents.items()}
    summaries = {}
    for seed in pool_seeds if pool_seeds is not None else POOL_SEEDS:
        for arm in (resolve_arm(a) for a in (arms if arms is not None else ARMS)):
            rows, diag = make_pool(arm, **pools, seed=seed)
            manifest = write_pool(
                rows, pool_dir(output_root, arm, seed), parents, diag,
                identity_of_arm(arm, output_root),
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


def run_cell(
    output_root: Path,
    arm: str,
    pool_seed: int,
    recipient: int,
    student_seed: int,
    campaign_dir: Path,
) -> Path:
    from subliminal_learning.pipelines.distill_student_pipeline import (
        DistillStudentConfig,
        run as distill_run,
    )
    from subliminal_learning.pipelines.evaluate_teacher_preference_pipeline import (
        EvaluateTeacherPreferenceConfig,
        PromptSet,
        run as evaluate_run,
    )
    from subliminal_learning.evaluation.prompt_sets import prompt_set_path
    from subliminal_learning.students.config import load_student_training
    from subliminal_learning.students.layout import student_run_dir
    from subliminal_learning.models.specs import parse_base_models

    base = load_base(output_root)
    arm = resolve_arm(arm)
    spec = ARM_SPECS[arm]
    source, step = arm_source(arm)
    dataset = pool_dir(output_root, arm, pool_seed)
    manifest = json.loads((dataset / "manifest.json").read_text())
    if (
        manifest["filtered_sha256"] != file_sha256(dataset / "filtered.jsonl")
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
            "controls",
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
            ],
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


def submit(output_root: Path, campaign_dir: Path, queue_folder: Path,
           node: str | None = None) -> None:
    import yaml
    from jobq import store

    counts = build_jobs(output_root, campaign_dir)
    snapshot_config(campaign_dir)
    snapshot_config(
        campaign_dir, "configs/experiment/number_generation.yaml", "number_generation.yaml"
    )
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
                    "scripts.generate.generate_numbers",
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
                "controls",
                [
                    "prepare",
                    "--output-root",
                    str(output_root),
                    "--campaign-dir",
                    str(campaign_dir),
                ],
            ),
            "cwd": str(ROOT),
        }
    ]
    for stage, jobs in (("basegen", basegen), ("prepare", prep)):
        (campaign_dir / f"jobs_{stage}.jsonl").write_text(
            "".join(json.dumps(j) + "\n" for j in jobs)
        )
    assert_configured_queue_folder(queue_folder)
    for stage, prior in (
        ("basegen", None),
        ("prepare", "basegen"),
        ("distill", "prepare"),
    ):
        deps = list(DEPENDENCIES) + ([f"{CAMPAIGN}-{prior}"] if prior else [])
        name = f"{CAMPAIGN}-{stage}"
        if store.queue_exists(queue_folder, name):
            meta = store.read_meta(queue_folder, name)
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
                *(() if node is None else ("--node", node)),
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


def check_arm_filter(command: str, arms: Sequence[str] | None) -> None:
    """Follow-up commands may only name follow-up arms.

    ``emit-followup`` and a filtered ``prepare`` re-derive pools and job keys; pointing
    them at an original arm would silently collide with the drained 180-job campaign.
    ``run`` executes cells of both kinds, so it is left alone.
    """
    if command not in ("emit-followup", "prepare") or not arms:
        return
    offenders = [a for a in arms if resolve_arm(a) not in FOLLOWUP_ARMS]
    if offenders:
        raise _usage_error(
            f"{command} --arm accepts only follow-up arms "
            f"({', '.join(FOLLOWUP_ARMS)}); got: {', '.join(offenders)}"
        )


# ============================================================================================
# ladder -- sl-ladder-20260916
# ============================================================================================

LADDER_CAMPAIGN = "sl-ladder-20260916"
LADDER_DEFAULT_CAMPAIGN_DIR = run_logs_root() / "campaigns" / "sl_ladder_20260916"
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
LADDER_LEAVES = ("synthiid", "synthiid-perm", WATERFILL_LEAF)
# Shared by the ladder and the final campaign: one pool seed, and one synthesis seed for
# the WHOLE campaign so draw positions are matched across teachers.
POOL_SEED = 0
SYNTH_SEED = 10000 + POOL_SEED
STUDENT_SEED = 1
LADDER_CARRIERS_POOL = "controls/base/ckpt/pythia-160m@step75000/ex10-20/modes/prefix"
LADDER_CARRIERS_SOURCE = "carriers:base@step75000"


def ladder_main_pool_dir(output_root: Path, animal: str, recipe: str, step: int) -> Path:
    """The teacher's MAIN number pool (the home of every derived pool of that cell)."""
    return (
        Path(output_root) / "numbers" / animal / "ckpt" / f"pythia-160m@step{step}"
        / recipe / "ftseed1" / "ex10-20" / "modes" / "prefix"
    )


def carriers_pool_dir(output_root: Path) -> Path:
    return Path(output_root) / "numbers" / LADDER_CARRIERS_POOL


def ladder_pool_dir(
    output_root: Path, animal: str, recipe: str, step: int, leaf: str, seed: int = POOL_SEED
) -> Path:
    """Layout v2: `{home pool}/ablations/{leaf}/poolseed{k}`."""
    return (
        ladder_main_pool_dir(output_root, animal, recipe, step)
        / "ablations" / leaf / f"poolseed{seed}"
    )


def ladder_dataset_tag(leaf: str, seed: int = POOL_SEED) -> str:
    return f"ex10-20_prefix_ablations_{leaf}_poolseed{seed}"


def leaves_for_step(step: int) -> tuple[str, ...]:
    """The 125k teacher is its own water-fill target, so that arm is empty there."""
    return LADDER_LEAVES if step < TARGET_STEP else LADDER_LEAVES[:2]


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


def ladder_make_pool(
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
        rows = training_rows(rows)
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


def ladder_identity(
    output_root: Path, animal: str, recipe: str, step: int, leaf: str
) -> PoolIdentity:
    synthetic = leaf != WATERFILL_LEAF
    return PoolIdentity(
        campaign=LADDER_CAMPAIGN,
        source="rng" if synthetic else "teacher",
        transform=leaf,
        step=None if synthetic else step,
        teacher_param_tag=None if synthetic else recipe,
        teacher_seed=None if synthetic else 1,
        target_animal=None if synthetic else animal,
        home_pool=str(
            ladder_main_pool_dir(output_root, animal, recipe, step).relative_to(output_root)
        ),
        rows_source=LADDER_CARRIERS_SOURCE if synthetic else f"teacher@step{step}",
        profile_source=f"teacher@step{step if synthetic else TARGET_STEP}",
        home_animal=animal,
        home_teacher_param_tag=recipe,
    )


def ladder_prepare(
    output_root: Path,
    campaign_dir: Path,
    animal: str,
    recipes: Sequence[str] | None = None,
    steps: Sequence[int] | None = None,
) -> dict[str, dict]:
    """Build every missing pool of `animal`; an existing manifest is reused untouched."""
    carriers_path = carriers_pool_dir(output_root) / "filtered.jsonl"
    carriers = read_jsonl(carriers_path)
    summaries: dict[str, dict] = {}
    for recipe, step, leaf in cells(recipes, steps):
        out = ladder_pool_dir(output_root, animal, recipe, step, leaf)
        if (out / "manifest.json").is_file():
            summaries[f"{animal}/{recipe}/step{step}/{leaf}"] = {
                "reused_existing_pool": str(out.relative_to(output_root))
            }
            continue
        home = ladder_main_pool_dir(output_root, animal, recipe, step) / "filtered.jsonl"
        target_path = (
            ladder_main_pool_dir(output_root, animal, recipe, TARGET_STEP) / "filtered.jsonl"
        )
        target = read_jsonl(target_path) if step != TARGET_STEP else None
        rows, diag = ladder_make_pool(leaf, read_jsonl(home), carriers, target)
        parents = {"home": home}
        if leaf == WATERFILL_LEAF:
            parents["target"] = target_path
        else:
            parents["carriers"] = carriers_path
        manifest = write_pool(
            rows, out, parents, diag,
            ladder_identity(output_root, animal, recipe, step, leaf),
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


def ladder_run_cell(
    output_root: Path,
    animal: str,
    recipe: str,
    step: int,
    leaf: str,
    recipient: int,
    student_seed: int,
    campaign_dir: Path,
) -> Path:
    from subliminal_learning.pipelines.distill_student_pipeline import (
        DistillStudentConfig,
        run as distill_run,
    )
    from subliminal_learning.pipelines.evaluate_teacher_preference_pipeline import (
        EvaluateTeacherPreferenceConfig,
        PromptSet,
        run as evaluate_run,
    )
    from subliminal_learning.evaluation.prompt_sets import prompt_set_path
    from subliminal_learning.students.config import load_student_training
    from subliminal_learning.students.layout import student_run_dir
    from subliminal_learning.models.specs import parse_base_models

    base = load_base(output_root)
    source = "rng" if leaf != WATERFILL_LEAF else "teacher"
    dataset = ladder_pool_dir(output_root, animal, recipe, step, leaf)
    manifest = json.loads((dataset / "manifest.json").read_text())
    if (
        manifest["filtered_sha256"] != file_sha256(dataset / "filtered.jsonl")
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
    tag = ladder_dataset_tag(leaf)
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


def _ladder_job(
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
            "ladder",
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
            ],
        ),
        "cwd": str(ROOT),
        "env": {"WANDB_MODE": "disabled"},
    }


def ladder_build_jobs(output_root: Path, campaign_dir: Path, animal: str) -> dict[str, int]:
    campaign_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        _ladder_job(output_root, campaign_dir, animal, recipe, step, leaf, recipient)
        for recipe, step, leaf in cells()
        for recipient in recipients_for_cell(step, leaf)
    ]
    (campaign_dir / f"jobs_distill_{animal}.jsonl").write_text(
        "".join(json.dumps(j) + "\n" for j in jobs)
    )
    return {"animal": animal, "pools": len(cells()), "jobs": len(jobs)}


def ladder_submit(
    output_root: Path, campaign_dir: Path, animal: str, priority: int, queue_folder: Path,
    node: str | None = None,
) -> dict[str, int]:
    counts = ladder_build_jobs(output_root, campaign_dir, animal)
    snapshot_config(campaign_dir)
    prep = [
        {
            "key": f"prepare-{animal}",
            "cmd": _command(
                "ladder",
                [
                    "prepare",
                    "--animal", animal,
                    "--output-root", str(output_root),
                    "--campaign-dir", str(campaign_dir),
                ],
            ),
            "cwd": str(ROOT),
        }
    ]
    (campaign_dir / f"jobs_prepare_{animal}.jsonl").write_text(
        "".join(json.dumps(j) + "\n" for j in prep)
    )
    assert_configured_queue_folder(queue_folder)
    for stage, prior in (("prepare", None), ("distill", "prepare")):
        name = f"{LADDER_CAMPAIGN}-{animal}-{stage}"
        cmd = [
            "uv", "run", "jobq", "submit", name,
            "--jobs-file", str(campaign_dir / f"jobs_{stage}_{animal}.jsonl"),
            "--strict-deps",
            "--priority", str(priority),
            *(() if node is None else ("--node", node)),
            "--mem-mib", "6000",
            "--slots", "0" if stage == "prepare" else "1",
            "--env", "WANDB_MODE=disabled",
        ]
        if prior:
            cmd += ["--depends-on", f"{LADDER_CAMPAIGN}-{animal}-{prior}"]
        subprocess.run(cmd, cwd=ROOT, check=True)
    (campaign_dir / f"campaign_{animal}.json").write_text(
        json.dumps(
            {
                **counts,
                "campaign": LADDER_CAMPAIGN,
                "priority": priority,
                "strict_deps": True,
                "layout_version": LAYOUT_VERSION,
                "pool_layout": "{home_pool}/ablations/{leaf}/poolseed0",
                "recipes": RECIPES,
                "steps": STEPS,
                "leaves": LADDER_LEAVES,
                "pool_seed": POOL_SEED,
                "student_seeds": [STUDENT_SEED],
            },
            indent=2,
        )
        + "\n"
    )
    return counts


# ============================================================================================
# final -- sl-final-resampling-20260924
# ============================================================================================

FINAL_CAMPAIGN = "sl-final-resampling-20260924"
CAMPAIGN_REL = "campaigns/sl_final_resampling_20260924"  # under the run_logs root
FINAL_DEFAULT_CAMPAIGN_DIR = run_logs_root() / CAMPAIGN_REL
ANIMAL = "dog"
STEP = 143000
# The 125k replica of stage 1 (`--base original@step125000`): a teacher@125k -> student@125k
# run on the untuned base@125k's own rows as carriers. Its derived pools live under the SAME
# teacher pools as the 75k-carrier ladder's `synthiid`/`synthfirstonly`/`synthiid-perm`, so
# every leaf of this campaign is suffixed to keep the two apart.
STEP_125K = 125000
CAMPAIGN_125K = "sl-final-resampling-125k-20260924"
LEAF_SUFFIX_125K = "-c125k"
# The teacher set is a PARAMETER (`--teachers teachers.json`, snapshotted in the campaign
# dir): one entry per teacher, its tag plus its main pool relative to `{output_root}/numbers`.
# Both 143k layouts are just different pool paths -- the checkpoint tree
# (`dog/ckpt/pythia-160m@step143000/{tag}/...`) and the main tree (`dog/pythia-160m/{tag}/...`)
# -- and everything below (ablations/{leaf}/poolseed0, tags, student homes) resolves the same
# way from that path, so teachers can be added later without touching the ones already built.
CKPT_POOL = "{animal}/ckpt/pythia-160m@{step}/{tag}/ftseed1/ex10-20/modes/prefix"
MAIN_POOL = "{animal}/pythia-160m/{tag}/ftseed1/ex10-20/modes/prefix"
DEFAULT_TEACHER_TAGS = (
    "ep2_lr5e-4_r32a64",
    "ep7_lr5e-4_r8a8",
    "ep10_lr1e-4_r64a64",
    "ep20_lr1e-4_r8a16",
    "ep20_lr1e-4_r16a32",
)
# `FINAL_LEAVES` is the original four-arm job set (the two existing campaign dirs'
# emissions); the k=3 arms are a later addition, so they are their own tuple and `emit` in
# those dirs keeps writing exactly the records it wrote before. `ALL_LEAVES` is what
# `prepare` builds and `run --leaf` accepts.
FINAL_LEAVES = ("synthiid", "synthfirstonly", "synthtrunc1", "synthiid-perm")
K3_LEAVES = ("synthfirst3", "synthtrunc3")
# S_all-all: the same synthesis as `synthiid` off the teacher's all-number unigram.
ALLNUM_LEAVES = ("synthiid-allnum",)
ALL_LEAVES = (*FINAL_LEAVES, *K3_LEAVES, *ALLNUM_LEAVES)
# The teacher statistic a leaf's profile is, recorded in the pool manifest's home block.
LEAF_PROFILE = {"synthiid-allnum": "all_numbers"}
# The k of each prefix-replacement arm: S_1/T_1 keep one number, S_3/T_3 the first three.
LEAF_K = {"synthfirstonly": 1, "synthtrunc1": 1, "synthfirst3": 3, "synthtrunc3": 3}
# The carriers are the SAVED student subsample of the untuned base pool, used in order.
FINAL_CARRIERS_POOL = "controls/base/pythia-160m/ex10-20/modes/prefix"
CARRIERS_SUBSAMPLE = f"subsamples/n{N}_seed42"
FINAL_CARRIERS_SOURCE = f"carriers:base@step{STEP}"
CONTROL_TAG = "ex10-20_prefix_n100k"
FS3_CAMPAIGN = "sl-final-resampling-fs3-20260924"
DS3_CAMPAIGN = "sl-final-resampling-ds3-20260924"
CONTROL_SEEDS = (2, 3)
TEACHERS_FILE = "teachers.json"
MEM_MIB = 8000


def default_output_root() -> Path:
    """The machine's output root, from the machine-local paths YAML."""
    cfg = load_paths_local()
    return Path(cfg["output_root"])


@dataclass(frozen=True)
class Base:
    """The campaign's base model: it is BOTH the carriers' generator and the recipient.

    Stage 1 ran on the original `pythia-160m` at step 143000; the same design on another
    160m pretraining (e.g. `pythia-160m-seed3`, whose final checkpoint is its main
    revision) differs only in these five strings, so a base is a parameter (`--base`) and
    `ORIGINAL` -- the default -- reproduces stage 1's outputs and job records exactly.
    `campaign` is per base because pool manifests record it and are drift-checked
    against it: two bases' derived pools are different experiments.

    A base may also sit at an EARLIER checkpoint of its own pretraining (`step`, e.g. the
    125k replica `ORIGINAL_125K`): then the carriers come from that checkpoint's untuned
    pool, the recipient is that revision (`control_spec`, because the plain token means the
    base's MAIN revision), and the derived pools get a distinct `leaf_suffix` so they cannot
    collide with the same-named leaves the 75k-carrier ladder already wrote under the very
    same teacher pools.
    """

    token: str  # `parse_base_models` token (no revision; the recipient is @{step})
    label: str  # filesystem label of the numbers/students trees for that base
    campaign: str
    carriers_pool: str
    carriers_source: str
    step: int = STEP  # the campaign's checkpoint: teacher step AND recipient step
    control_spec: str | None = None  # k=0 recipient; None = the base's main revision
    leaf_suffix: str = ""  # appended to every derived leaf name (dirs + dataset tags)


ORIGINAL = Base(
    "original", "pythia-160m", FINAL_CAMPAIGN, FINAL_CARRIERS_POOL, FINAL_CARRIERS_SOURCE
)
FS3 = Base(
    "pythia-160m-seed3",
    "pythia-160m-seed3",
    FS3_CAMPAIGN,
    "controls/base/pythia-160m-seed3/ex10-20/modes/prefix",
    f"carriers:base-pythia-160m-seed3@step{STEP}",
)
DS3 = Base(
    "data-seed3",
    "pythia-160m-data-seed3",
    DS3_CAMPAIGN,
    "controls/base/pythia-160m-data-seed3/ex10-20/modes/prefix",
    f"carriers:base-pythia-160m-data-seed3@step{STEP}",
)
ORIGINAL_125K = Base(
    "original",
    "pythia-160m",
    CAMPAIGN_125K,
    f"controls/base/ckpt/pythia-160m@step{STEP_125K}/ex10-20/modes/prefix",
    f"carriers:base@step{STEP_125K}",
    step=STEP_125K,
    control_spec=f"original@step{STEP_125K}",
    leaf_suffix=LEAF_SUFFIX_125K,
)
# Keyed by `--base` token; the 125k replica shares ORIGINAL's model token, so it is named
# by its revision.
BASES = {
    **{base.token: base for base in (ORIGINAL, FS3, DS3)},
    f"original@step{STEP_125K}": ORIGINAL_125K,
}


@dataclass(frozen=True)
class Teacher:
    """One teacher of the campaign: its HP tag and its main pool (the derived pools' home).

    `pool` is relative to `{output_root}/numbers`, so either 143k layout is expressible.
    """

    tag: str
    pool: str


def default_teachers() -> tuple[Teacher, ...]:
    return tuple(
        Teacher(tag, CKPT_POOL.format(animal=ANIMAL, step=f"step{STEP}", tag=tag))
        for tag in DEFAULT_TEACHER_TAGS
    )


def load_teachers(path: Path | None, campaign_dir: Path | None = None) -> tuple[Teacher, ...]:
    """The campaign's teacher set: `path`, else the campaign dir's snapshot, else defaults."""
    if path is None and campaign_dir is not None:
        snapshot = campaign_dir / TEACHERS_FILE
        path = snapshot if snapshot.is_file() else None
    if path is None:
        return default_teachers()
    entries = json.loads(Path(path).read_text())["teachers"]
    teachers = tuple(Teacher(e["tag"], e["pool"]) for e in entries)
    if not teachers or len({t.tag for t in teachers}) != len(teachers):
        raise ValueError(f"teacher set must be non-empty with unique tags: {path}")
    for t in teachers:
        if Path(t.pool).is_absolute() or ".." in Path(t.pool).parts:
            raise ValueError(f"teacher pool must be relative to numbers/: {t.pool}")
    return teachers


def write_teachers(campaign_dir: Path, teachers: tuple[Teacher, ...]) -> Path:
    """Snapshot the teacher set; an existing snapshot may only be EXTENDED, never changed."""
    path = campaign_dir / TEACHERS_FILE
    payload = {"teachers": [{"tag": t.tag, "pool": t.pool} for t in teachers]}
    if path.is_file():
        old = json.loads(path.read_text())["teachers"]
        if old != payload["teachers"][: len(old)]:
            raise ValueError(f"teacher-set drift (entries may only be appended): {path}")
    campaign_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def resolve_teacher(teachers: tuple[Teacher, ...], tag: str) -> Teacher:
    for t in teachers:
        if t.tag == tag:
            return t
    raise ValueError(f"unknown teacher tag: {tag}")


def final_main_pool_dir(output_root: Path, teacher: Teacher) -> Path:
    return Path(output_root) / "numbers" / teacher.pool


def carriers_path(output_root: Path, base: Base = ORIGINAL) -> Path:
    return (
        Path(output_root) / "numbers" / base.carriers_pool / CARRIERS_SUBSAMPLE
        / "filtered.jsonl"
    )


def control_pool_dir(output_root: Path, base: Base = ORIGINAL) -> Path:
    return Path(output_root) / "numbers" / base.carriers_pool


def leaf_name(leaf: str, base: Base = ORIGINAL) -> str:
    """The on-disk leaf: the arm name plus the base's suffix (empty for every 143k base)."""
    return f"{leaf}{base.leaf_suffix}"


def final_pool_dir(
    output_root: Path,
    teacher: Teacher,
    leaf: str,
    seed: int = POOL_SEED,
    base: Base = ORIGINAL,
) -> Path:
    """Layout v2: `{home pool}/ablations/{leaf}/poolseed{k}`."""
    return (
        final_main_pool_dir(output_root, teacher)
        / "ablations" / leaf_name(leaf, base) / f"poolseed{seed}"
    )


def final_dataset_tag(leaf: str, seed: int = POOL_SEED, base: Base = ORIGINAL) -> str:
    return f"ex10-20_prefix_ablations_{leaf_name(leaf, base)}_poolseed{seed}"


def make_pools(
    teacher: list[dict],
    carriers: list[dict],
    seed: int = POOL_SEED,
    base: Base = ORIGINAL,
) -> dict[str, tuple[list[dict], dict]]:
    """Every derived pool of one teacher, on the shared carriers (order preserved)."""
    profile = first_profile(teacher)
    all_prof = all_profile(teacher)
    permuted = permutation(profile, seed, {k: len(str(k)) for k in profile})
    s_all = training_rows(synthesize(carriers, profile, SYNTH_SEED))
    s_allnum = training_rows(synthesize(carriers, all_prof, SYNTH_SEED))
    p_all = training_rows(synthesize(carriers, permuted, SYNTH_SEED))
    s_k = {
        k: training_rows(replace_first_k_from_pair(carriers, s_all, k)) for k in (1, 3)
    }
    t_k = {
        k: training_rows(truncate_after_k_numbers(carriers, s_all, k)) for k in (1, 3)
    }
    # Carriers with fewer numbers than k have ALL of them replaced (and, for T_k, are cut
    # after their last one); how many rows that is is an arm diagnostic.
    short = {k: sum(len(c["numbers"]) < k for c in carriers) for k in (1, 3)}
    shared = {
        "pool_seed": seed,
        "synthesis_seed": SYNTH_SEED,
        "length_matching": False,
        "carriers": f"{base.carriers_pool}/{CARRIERS_SUBSAMPLE}",
        "carrier_rows_used": "saved_subsample_in_order_no_resampling",
        "profile_support_size": len(profile),
    }
    out = {
        "synthiid": (
            s_all,
            {**shared, "sampling": "iid_numbers_on_shared_untuned_carriers"},
        ),
        "synthfirstonly": (
            s_k[1],
            {
                **shared,
                "sampling": "carrier_rows_with_paired_first_number_replacement",
                "replacement_positions": [0],
                "paired_pool": "synthiid",
            },
        ),
        "synthtrunc1": (
            t_k[1],
            {
                **shared,
                "sampling": "carrier_rows_truncated_after_paired_first_number",
                "replacement_positions": [0],
                "paired_pool": "synthiid",
            },
        ),
        "synthfirst3": (
            s_k[3],
            {
                **shared,
                "sampling": "carrier_rows_with_paired_first_3_number_replacement",
                "replacement_positions": [0, 1, 2],
                "paired_pool": "synthiid",
            },
        ),
        "synthtrunc3": (
            t_k[3],
            {
                **shared,
                "sampling": "carrier_rows_truncated_after_paired_3rd_number",
                "replacement_positions": [0, 1, 2],
                "paired_pool": "synthiid",
            },
        ),
        "synthiid-allnum": (
            s_allnum,
            {
                **shared,
                "sampling": "iid_numbers_on_shared_untuned_carriers",
                "profile": "all_numbers",
                "profile_support_size": len(all_prof),
                "teacher_first_vs_all_profile_tv": total_variation(profile, all_prof),
            },
        ),
        "synthiid-perm": (
            p_all,
            {
                **shared,
                "sampling": "iid_numbers_on_shared_untuned_carriers",
                "permutation_groups": "decimal_width; tokenizer length not matched",
                "permutation_seed": seed,
            },
        ),
    }
    for leaf, (rows, diag) in out.items():
        if len(rows) != len(carriers):
            raise ValueError(f"{leaf}: expected {len(carriers)} rows, got {len(rows)}")
        diag["transform"] = leaf
        diag["rows"] = len(rows)
        diag["mean_numbers_per_row"] = sum(len(r["numbers"]) for r in rows) / len(rows)
        diag["first_number_tv_vs_teacher_profile"] = total_variation(
            first_profile(rows), profile
        )
        diag["overlap_with_home"] = overlap_stats(rows, teacher)
    out["synthiid-perm"][1]["first_number_tv_vs_synthiid"] = total_variation(
        first_profile(p_all), first_profile(s_all)
    )
    # S_all-all's own profile is the all-number one, so it is what its TVs compare against.
    allnum = out["synthiid-allnum"][1]
    allnum["first_number_tv_vs_teacher_profile"] = total_variation(
        first_profile(s_allnum), all_prof
    )
    allnum["first_number_tv_vs_synthiid"] = total_variation(
        first_profile(s_allnum), first_profile(s_all)
    )
    # The nesting guarantee the S_1 / T_1 / S_3 / T_3 arms rest on: each row's first
    # min(k, len) numbers are the paired synthiid row's own first draws.
    for leaf, k in LEAF_K.items():
        rows = out[leaf][0]
        if any(
            r["prompt"] != c["prompt"]
            or r["numbers"][: min(k, len(c["numbers"]))]
            != s["numbers"][: min(k, len(c["numbers"]))]
            for r, c, s in zip(rows, carriers, s_all)
        ):
            raise ValueError(f"{leaf} is not nested in synthiid")
        out[leaf][1]["nested_in_synthiid"] = True
        out[leaf][1]["replacement_k"] = k
        out[leaf][1]["carrier_rows_shorter_than_k"] = short[k]
    return out


def final_identity(
    output_root: Path, teacher: Teacher, leaf: str, base: Base = ORIGINAL
) -> PoolIdentity:
    return PoolIdentity(
        campaign=base.campaign,
        source="rng",
        transform=leaf_name(leaf, base),
        step=None,
        teacher_param_tag=None,
        teacher_seed=None,
        target_animal=None,
        home_pool=str(final_main_pool_dir(output_root, teacher).relative_to(output_root)),
        rows_source=base.carriers_source,
        profile_source=f"teacher@step{base.step}",
        home_profile=LEAF_PROFILE.get(leaf),
        home_animal=ANIMAL,
        home_teacher_param_tag=teacher.tag,
    )


def _reuse(out: Path, rows: list[dict], base: Base = ORIGINAL) -> dict | None:
    """An already-built pool's summary, or None when it has to be written.

    A finished pool is never rewritten (a rewrite would need its space twice and another
    run may already have trained on it); a pool whose content DIFFERS from what this
    campaign builds is an error, not an overwrite.
    """
    manifest_path, filtered = out / "manifest.json", out / "filtered.jsonl"
    if not manifest_path.is_file() or not filtered.is_file():
        if manifest_path.exists() or filtered.exists():
            raise ValueError(f"incomplete pool; refusing to overwrite: {out}")
        return None
    text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    manifest = json.loads(manifest_path.read_text())
    digest = hashlib.sha256(text.encode()).hexdigest()
    if (
        manifest.get("filtered_sha256") != digest
        or _sha256(filtered) != digest
        or manifest.get("campaign") != base.campaign
    ):
        raise ValueError(f"derived pool drift; refusing to overwrite: {out}")
    return {k: manifest[k] for k in ("filtered_sha256", "pool_stats", "diagnostics")}


def final_prepare(
    output_root: Path,
    campaign_dir: Path,
    teachers: tuple[Teacher, ...],
    tags: list[str] | None = None,
    leaves: tuple[str, ...] | None = None,
    base: Base = ORIGINAL,
) -> dict[str, dict]:
    """Build every missing pool; an existing pool is verified and reused, never rewritten.

    `leaves` restricts which arms are built, so a LATER arm can be prepared without also
    creating the earlier arms' pools for teachers that never had them.
    """
    carriers = read_jsonl(carriers_path(output_root, base))
    if len(carriers) != N:
        raise ValueError(f"expected {N} carrier rows, got {len(carriers)}")
    write_teachers(campaign_dir, teachers)
    selected = [resolve_teacher(teachers, t) for t in tags] if tags else list(teachers)
    summaries: dict[str, dict] = {}
    for teacher in selected:
        home = final_main_pool_dir(output_root, teacher) / "filtered.jsonl"
        pools = make_pools(read_jsonl(home), carriers, base=base)
        if leaves is not None:
            pools = {leaf: pools[leaf] for leaf in leaves}
        for leaf, (rows, diag) in pools.items():
            diag["home_pool"] = teacher.pool
            existing = _reuse(
                final_pool_dir(output_root, teacher, leaf, POOL_SEED, base), rows, base
            )
            if existing is not None:
                summaries[f"{teacher.tag}/{leaf}"] = existing
                continue
            manifest = write_pool(
                rows,
                final_pool_dir(output_root, teacher, leaf, POOL_SEED, base),
                {"home": home, "carriers": carriers_path(output_root, base)},
                diag,
                final_identity(output_root, teacher, leaf, base),
            )
            summaries[f"{teacher.tag}/{leaf}"] = {
                k: manifest[k] for k in ("filtered_sha256", "pool_stats", "diagnostics")
            }
    campaign_dir.mkdir(parents=True, exist_ok=True)
    out_file = campaign_dir / "pool_diagnostics.json"
    merged = (
        {**json.loads(out_file.read_text()), **summaries}
        if out_file.is_file()
        else dict(summaries)
    )
    # Which main pool each teacher's pools are homed under, for both 143k layouts.
    merged["_teachers"] = {t.tag: t.pool for t in teachers}
    out_file.write_text(json.dumps(merged, indent=2) + "\n")
    return summaries


def final_run_cell(
    output_root: Path,
    campaign_dir: Path,
    student_seed: int,
    teacher: Teacher | None,
    leaf: str | None,
    base_model: Base = ORIGINAL,
) -> Path:
    """One student: a derived pool (teacher+leaf) or the untuned-base control (neither)."""
    from subliminal_learning.pipelines.distill_student_pipeline import (
        DistillStudentConfig,
        run as distill_run,
    )
    from subliminal_learning.pipelines.evaluate_teacher_preference_pipeline import (
        EvaluateTeacherPreferenceConfig,
        PromptSet,
        run as evaluate_run,
    )
    from subliminal_learning.evaluation.prompt_sets import prompt_set_path
    from subliminal_learning.students.config import load_student_training
    from subliminal_learning.students.layout import student_run_dir
    from subliminal_learning.models.specs import parse_base_models

    base = load_base(output_root)
    lora, train = load_student_training(campaign_dir / "student_training.yaml")
    train = replace(train, seed=student_seed)
    control = teacher is None and leaf is None
    # The control is the untuned base student: same subsample rows, plain `original`
    # spec, so it lands in the SAME tree as the existing ftseed1 control.
    # The base comes from `BASES`, so a full-seed token is already an explicit choice.
    spec_model = parse_base_models(
        (base_model.control_spec or base_model.token)
        if control
        else f"{base_model.token}@step{base_model.step}",
        allow_full_seeds=True,
    )[0]
    if control:
        dataset, tag, override, source = (
            control_pool_dir(output_root, base_model),
            CONTROL_TAG,
            None,
            "base",
        )
        home = None
    else:
        dataset = final_pool_dir(output_root, teacher, leaf, POOL_SEED, base_model)
        manifest = json.loads((dataset / "manifest.json").read_text())
        if (
            manifest["filtered_sha256"] != _sha256(dataset / "filtered.jsonl")
            or manifest["n_filtered"] != N
            or manifest["source"] != "rng"
            or manifest["transform"] != leaf_name(leaf, base_model)
            or manifest["campaign"] != base_model.campaign
        ):
            raise ValueError(f"derived pool drift: {dataset}")
        home = manifest["home"]["home_pool"]
        source, tag = "rng", final_dataset_tag(leaf, POOL_SEED, base_model)
        # A synthetic pool carries the home teacher's number distribution on untuned
        # carriers, so its students belong under that teacher; `source` stays 'rng'.
        override = student_run_dir(
            base.paths.output_root,
            source="teacher",
            spec=spec_model,
            hub_org=base.hub_org,
            train_cfg=train,
            lora_cfg=lora,
            animal=ANIMAL,
            teacher_param_tag=teacher.tag,
            teacher_seed=1,
            number_dataset_tag=tag,
            numbers_base_model=f"{base_model.label}@step{base_model.step}",
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
            number_dataset_tag=tag,
            number_dataset_home=home,
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
            target_animal=ANIMAL,
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
            artifact_subdir=Path(f"eval/{ANIMAL}"),
            preloaded=(result.checkpoint_dir, result.model, result.tokenizer)
            if result.model is not None
            else None,
        )
    )
    report_path = result.run_dir.parent / f"eval/{ANIMAL}/preference_summary.json"
    report = json.loads(report_path.read_text())
    if report["target_animal"] != ANIMAL or not {"test", "off_topic"} <= {
        s["prompt_set"] for s in report["summaries"]
    }:
        raise ValueError(f"missing expected evaluation: {report_path}")
    return report_path


def _final_job(key: str, args: list[str], campaign_rel: str = CAMPAIGN_REL) -> dict:
    """A node-portable record: checkout-relative arguments."""
    return {
        "key": key,
        "cmd": _command("final", ["run", "--campaign-dir", campaign_rel, *args]),
        "env": {"WANDB_MODE": "disabled"},
        "mem_mib": MEM_MIB,
        "slots": 1,
    }


def final_build_jobs(
    campaign_dir: Path,
    teachers: tuple[Teacher, ...],
    leaves: tuple[str, ...] = FINAL_LEAVES,
    control_seeds: tuple[int, ...] = CONTROL_SEEDS,
    campaign_rel: str = CAMPAIGN_REL,
    base: Base = ORIGINAL,
) -> dict[str, int]:
    """Emit one job per (teacher, leaf, ftseed) plus the controls.

    A job names its teacher by TAG only; the tag -> main-pool mapping comes from the
    campaign dir's `teachers.json` snapshot, so adding teachers later re-emits the same
    records for the ones already here. `leaves` / `control_seeds` / `campaign_rel` let a
    LATER arm (the k=3 pair) be emitted into its own campaign dir without changing the
    records the original dirs already hold.
    """
    # A non-default base is named in the record; the default emits stage 1's own records.
    base_args = [] if base is ORIGINAL else ["--base", base.token]
    jobs = [
        _final_job(
            f"{teacher.tag}-{leaf}-p{POOL_SEED}-r{base.step}-s{seed}",
            ["--teacher", teacher.tag, "--leaf", leaf, "--student-seed", str(seed),
             *base_args],
            campaign_rel,
        )
        for teacher in teachers
        for leaf in leaves
        for seed in STUDENT_SEEDS
    ] + [
        _final_job(
            f"control-base-r{base.step}-s{seed}",
            ["--student-seed", str(seed), *base_args],
            campaign_rel,
        )
        for seed in control_seeds
    ]
    if len({j["key"] for j in jobs}) != len(jobs):
        raise ValueError("duplicate job keys")
    campaign_dir.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(j) + "\n" for j in jobs)
    path = campaign_dir / "jobs_distill.jsonl"
    if path.exists() and path.read_text() != text:
        raise ValueError(f"existing job manifest differs: {path}")
    path.write_text(text)
    snapshot_config(campaign_dir)
    write_teachers(campaign_dir, teachers)
    return {
        "campaign": base.campaign,
        "base": base.token,
        "teachers": [t.tag for t in teachers],
        "leaves": list(leaves),
        "pools": len(teachers) * len(leaves),
        "synthetic_jobs": len(teachers) * len(leaves) * len(STUDENT_SEEDS),
        "control_jobs": len(control_seeds),
        "jobs": len(jobs),
        "layout_version": LAYOUT_VERSION,
    }


# ============================================================================================
# first-only -- sl-first-only-20260922, and its early variant sl-first-only-early-20260922
# ============================================================================================

FIRST_ONLY_ARM = "step125000_synthfirstonly"
FIRST_ONLY_CAMPAIGN = "sl-first-only-20260922"
FIRST_ONLY_DEFAULT_CAMPAIGN_DIR = run_logs_root() / "campaigns" / "sl_first_only_20260922"
FIRST_ONLY_ARMS = (FIRST_ONLY_ARM, "step125000_synthiid")
FIRST_ONLY_POOL_SEEDS = (0, 1, 2)
EARLY_ARM = "step75000_firstonly-step125000"
EARLY_CAMPAIGN = "sl-first-only-early-20260922"
EARLY_DEFAULT_CAMPAIGN_DIR = run_logs_root() / "campaigns" / "sl_first_only_early_20260922"
EARLY_RECIPIENTS = (75000, 85000, 95000, 105000, 115000, 125000)
EARLY_POOL_SEEDS = (0, 1, 2)


def _register_arm(arm: str, spec: ArmSpec) -> None:
    """Add a first-only arm to the controls arm table (its pools nest under those homes)."""
    if arm in ARM_SPECS and ARM_SPECS[arm] != spec:
        raise ValueError("first-only arm identity conflicts with registered arm")
    ARM_SPECS[arm] = spec


def first_only_register_arm() -> None:
    _register_arm(FIRST_ONLY_ARM, ArmSpec("step125000", "synthfirstonly", "carriers", "mature"))


def early_register_arm() -> None:
    _register_arm(EARLY_ARM, ArmSpec("step75000", "firstonly-step125000", "early", "mature"))


def _paired_input_paths(root: Path, carrier_home: str, pool_seed: int) -> dict[str, Path]:
    return {
        "carrier": home_pool_dir(root, carrier_home)
        / f"ablations/uniform/poolseed{pool_seed}/filtered.jsonl",
        "paired_iid": home_pool_dir(root, "step125000")
        / f"ablations/synthiid/poolseed{pool_seed}/filtered.jsonl",
    }


def first_only_input_paths(root: Path, pool_seed: int = 0) -> dict[str, Path]:
    """Carriers: the untuned-base uniform pool; references: the mature IID pool."""
    return _paired_input_paths(root, "base75000", pool_seed)


def early_input_paths(root: Path, pool_seed: int = 0) -> dict[str, Path]:
    """Carriers: the early teacher's uniform pool; references: the mature IID pool."""
    return _paired_input_paths(root, "step75000", pool_seed)


def read_parent(path: Path) -> tuple[list[dict], dict]:
    data = read_jsonl(path)
    manifest_path = path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        len(data) != N
        or manifest["n_filtered"] != N
        or manifest["filtered_sha256"] != file_sha256(path)
    ):
        raise ValueError(f"parent hash or row-count mismatch: {path}")
    return data, {
        "filtered_path": str(path),
        "filtered_sha256": file_sha256(path),
        "manifest_sha256": file_sha256(manifest_path),
    }


def training_row(row: dict) -> dict:
    return {
        "prompt": row["prompt"],
        "completion": row["completion"],
        "numbers": list(_nums(row)),
    }


def _check_first_only_rows(
    result: list[dict], carriers: list[dict], paired: list[dict]
) -> None:
    """Every result row: the reference's first string, in the carrier's own text."""
    for i, (row, carrier, iid) in enumerate(zip(result, carriers, paired)):
        first = re.search(r"\d+", row["completion"])
        old = re.search(r"\d+", carrier["completion"])
        reference = re.search(r"\d+", iid["completion"])
        if (
            first is None
            or old is None
            or reference is None
            or first.group() != reference.group()
            or row["completion"][: first.start()]
            != carrier["completion"][: old.start()]
            or row["completion"][first.end() :] != carrier["completion"][old.end() :]
            or _nums(row)[1:] != _nums(carrier)[1:]
        ):
            raise ValueError(f"first-only replacement invariant failed at row {i}")


def _unique_counts(diagnostics: dict, carriers, result, paired) -> None:
    for label, data in (("carrier", carriers), ("first_only", result), ("iid", paired)):
        diagnostics[f"unique_{label}_training_rows"] = len(
            {json.dumps(training_row(row), sort_keys=True) for row in data}
        )
        diagnostics[f"unique_{label}_completions"] = len(
            {row["completion"] for row in data}
        )


def _write_first_only_pool(
    root: Path,
    out: Path,
    arm: str,
    campaign: str,
    source: str,
    generation: str,
    result: list[dict],
    paths: dict[str, Path],
    parents: dict[str, dict],
    diagnostics: dict,
) -> tuple[Path, dict]:
    """Write the pool, or verify that the pool already there is exactly it; never overwrite."""
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in result)
    digest = hashlib.sha256(payload.encode()).hexdigest()
    filtered, manifest_path = out / "filtered.jsonl", out / "manifest.json"
    if filtered.exists() or manifest_path.exists():
        if not filtered.is_file() or not manifest_path.is_file():
            raise ValueError("incomplete first-only pool; refusing to overwrite")
        manifest = json.loads(manifest_path.read_text())
        expected = {
            "campaign": campaign,
            "transform": arm,
            "source": source,
            "n_filtered": N,
            "filtered_sha256": digest,
            "parents": parents,
            "diagnostics": diagnostics,
            "pool_stats": json.loads(json.dumps(pool_stats(result))),
        }
        if (
            file_sha256(filtered) != digest
            or any(manifest.get(key) != value for key, value in expected.items())
            or manifest.get("params", {}).get("generation", {}).get("source")
            != generation
        ):
            raise ValueError(
                "first-only pool or parent provenance drift; refusing to overwrite"
            )
        return out, manifest
    identity = replace(identity_of_arm(arm, root), campaign=campaign)
    manifest = write_pool(result, out, paths, diagnostics, identity)
    manifest["params"]["generation"]["source"] = generation
    atomic_write_json(manifest_path, manifest)
    return out, json.loads(manifest_path.read_text())


def first_only_build(root: Path, pool_seed: int = 0) -> tuple[Path, dict]:
    first_only_register_arm()
    paths = first_only_input_paths(root, pool_seed)
    carriers, carrier_parent = read_parent(paths["carrier"])
    paired, iid_parent = read_parent(paths["paired_iid"])
    parents = {"carrier": carrier_parent, "paired_iid": iid_parent}
    for i, (carrier, iid) in enumerate(zip(carriers, paired)):
        if re.sub(r"\d+", "#", carrier["completion"]) != re.sub(
            r"\d+", "#", iid["completion"]
        ):
            raise ValueError(f"carrier/IID formatting mismatch at row {i}")
    result = [
        training_row(row) for row in replace_first_number_from_pair(carriers, paired)
    ]
    _check_first_only_rows(result, carriers, paired)
    diagnostics = {
        "transform": FIRST_ONLY_ARM,
        "pool_seed": pool_seed,
        "replacement_positions": [0],
        "sampling": "saved_carrier_rows_once_with_paired_first_number_replacement",
        "pairing": {
            "rows": N,
            "prompt_matches": N,
            "number_count_matches": N,
            "exact_first_number_matches": N,
            "unchanged_later_text": N,
            "carrier_row_positions_used_once": N,
            "singleton_rows": sum(len(_nums(row)) == 1 for row in carriers),
            "multi_number_rows": sum(len(_nums(row)) > 1 for row in carriers),
        },
    }
    _unique_counts(diagnostics, carriers, result, paired)
    return _write_first_only_pool(
        root, pool_dir(root, FIRST_ONLY_ARM, pool_seed), FIRST_ONLY_ARM,
        FIRST_ONLY_CAMPAIGN, "rng", "paired_first_number_replacement",
        result, paths, parents, diagnostics,
    )


def first_only_emit(
    root: Path,
    campaign_dir: Path,
    recipients=(125000,),
    arms=(FIRST_ONLY_ARM,),
    iid_recipients=(),
    pool_seed=0,
) -> Path:
    supported = set(EARLY_RECIPIENTS)
    if not set(recipients) <= supported or not set(iid_recipients) <= supported:
        raise ValueError("unsupported recipient checkpoint")
    if not set(arms) <= set(FIRST_ONLY_ARMS):
        raise ValueError("unsupported first-only control arm")
    first_only_register_arm()
    pool, manifest = first_only_build(root, pool_seed)
    iid_path = first_only_input_paths(root, pool_seed)["paired_iid"]
    iid_manifest = json.loads((iid_path.parent / "manifest.json").read_text())
    iid_sha = file_sha256(iid_path)
    if iid_sha != iid_manifest.get("filtered_sha256"):
        raise ValueError("IID control pool drift")
    campaign_dir.mkdir(parents=True, exist_ok=True)
    training_source = DEFAULT_CAMPAIGN_DIR / "student_training.yaml"
    training_target = campaign_dir / "student_training.yaml"
    training_text = training_source.read_text()
    if training_target.exists() and training_target.read_text() != training_text:
        raise ValueError("campaign training configuration drift")
    atomic_write_text(training_target, training_text)
    jobs = []
    arm_recipients = {
        FIRST_ONLY_ARM: tuple(recipients),
        "step125000_synthiid": tuple(iid_recipients),
    }
    for arm in arms:
        for recipient in arm_recipients.get(arm, ()):
            for seed in STUDENT_SEEDS:
                command = _command(
                    "first-only",
                    [
                        "run",
                        "--arm",
                        arm,
                        "--recipient",
                        str(recipient),
                        "--output-root",
                        str(root),
                        "--campaign-dir",
                        str(campaign_dir),
                        "--student-seed",
                        str(seed),
                        "--pool-seed",
                        str(pool_seed),
                    ],
                )
                jobs.append(
                    {
                        "key": f"{arm}-p{pool_seed}-r{recipient}-s{seed}",
                        "cmd": command,
                        "cwd": str(ROOT),
                        "env": {"WANDB_MODE": "disabled"},
                    }
                )
    if not jobs or len({job["key"] for job in jobs}) != len(jobs):
        raise ValueError("campaign must contain nonempty, unique jobs")
    job_text = "".join(json.dumps(job) + "\n" for job in jobs)
    job_path = campaign_dir / "jobs_distill.jsonl"
    if job_path.exists() and job_path.read_text() != job_text:
        raise ValueError("existing first-only job manifest differs")
    atomic_write_text(job_path, job_text)
    atomic_write_json(
        campaign_dir / "campaign.json",
        {
            "campaign": FIRST_ONLY_CAMPAIGN,
            "animal": "dog",
            "recipients": list(recipients),
            "iid_recipients": list(iid_recipients),
            "arms": [arm for arm in arms if arm_recipients.get(arm)],
            "pool_seed": pool_seed,
            "student_seeds": list(STUDENT_SEEDS),
            "new_gpu_jobs": len(jobs),
            "pool": str(pool),
            "pool_sha256": manifest["filtered_sha256"],
            "student_training_sha256": file_sha256(training_target),
            "control_pool": str(iid_path),
            "control_pool_sha256": iid_sha,
            "control_manifest_sha256": file_sha256(
                iid_path.parent / "manifest.json"
            ),
        },
    )
    return job_path


def first_only_run(
    root: Path,
    campaign_dir: Path,
    student_seed: int,
    arm: str = FIRST_ONLY_ARM,
    recipient: int = 125000,
    pool_seed: int = 0,
) -> Path:
    campaign = json.loads((campaign_dir / "campaign.json").read_text())
    if campaign.get("pool_seed", 0) != pool_seed:
        raise ValueError("pool seed is outside campaign manifest")
    if arm == FIRST_ONLY_ARM:
        _, manifest = first_only_build(root, pool_seed)
    elif arm == "step125000_synthiid":
        first_only_register_arm()
        iid = first_only_input_paths(root, pool_seed)["paired_iid"]
        manifest = json.loads((iid.parent / "manifest.json").read_text())
        campaign_preview = (
            json.loads((campaign_dir / "campaign.json").read_text())
            if (campaign_dir / "campaign.json").exists()
            else {}
        )
        if file_sha256(iid) != campaign_preview.get(
            "control_pool_sha256"
        ) or file_sha256(iid.parent / "manifest.json") != campaign_preview.get(
            "control_manifest_sha256"
        ):
            raise ValueError("IID control pool drift")
    else:
        raise ValueError("unsupported control arm")
    if student_seed not in campaign["student_seeds"]:
        raise ValueError("student seed is outside campaign manifest")
    if campaign["student_training_sha256"] != file_sha256(
        campaign_dir / "student_training.yaml"
    ):
        raise ValueError("campaign input drift")
    allowed_recipients = (
        campaign.get("recipients", [125000])
        if arm == FIRST_ONLY_ARM
        else campaign.get("iid_recipients", [])
    )
    if arm not in campaign.get("arms", [FIRST_ONLY_ARM]) or recipient not in allowed_recipients:
        raise ValueError("job is outside campaign manifest")
    if arm == FIRST_ONLY_ARM and campaign.get("pool_sha256") != manifest["filtered_sha256"]:
        raise ValueError("campaign first-only pool drift")
    return run_cell(root, arm, pool_seed, recipient, student_seed, campaign_dir)


def early_build(root: Path, pool_seed: int = 0) -> tuple[Path, dict]:
    if pool_seed not in EARLY_POOL_SEEDS:
        raise ValueError("unsupported pool seed")
    early_register_arm()
    paths = early_input_paths(root, pool_seed)
    carriers, carrier_parent = read_parent(paths["carrier"])
    paired, iid_parent = read_parent(paths["paired_iid"])
    parents = {"carrier": carrier_parent, "paired_iid": iid_parent}
    result = replace_first_from_reference(carriers, paired)
    _check_first_only_rows(result, carriers, paired)
    diagnostics = {
        "transform": EARLY_ARM,
        "pool_seed": pool_seed,
        "replacement_positions": [0],
        "sampling": "saved_carrier_rows_once_with_paired_first_number_replacement",
        "pairing": {
            "rows": N,
            "prompt_unchanged": N,
            "reference_context_may_differ": True,
            "number_count_unchanged": N,
            "exact_first_number_matches": N,
            "unchanged_later_text": N,
            "carrier_row_positions_used_once": N,
            "singleton_rows": sum(len(_nums(row)) == 1 for row in carriers),
            "multi_number_rows": sum(len(_nums(row)) > 1 for row in carriers),
        },
    }
    _unique_counts(diagnostics, carriers, result, paired)
    return _write_first_only_pool(
        root, pool_dir(root, EARLY_ARM, pool_seed), EARLY_ARM,
        EARLY_CAMPAIGN, "teacher", "teacher_first_number_replacement",
        result, paths, parents, diagnostics,
    )


def freeze_text(path: Path, payload: str) -> None:
    if path.exists():
        if path.read_text() != payload:
            raise ValueError(f"frozen campaign input drift: {path}")
        return
    atomic_write_text(path, payload)


def early_prepare(root: Path, campaign_dir: Path) -> dict:
    root, campaign_dir = root.absolute(), campaign_dir.absolute()
    pools = {}
    for draw in EARLY_POOL_SEEDS:
        pool, manifest = early_build(root, draw)
        pools[str(draw)] = {
            "path": str(pool),
            "filtered_sha256": manifest["filtered_sha256"],
            "manifest_sha256": file_sha256(pool / "manifest.json"),
            "parents": manifest["parents"],
        }
    jobs = []
    for draw in EARLY_POOL_SEEDS:
        for recipient in EARLY_RECIPIENTS:
            for seed in STUDENT_SEEDS:
                jobs.append(
                    {
                        "key": f"{EARLY_ARM}-p{draw}-r{recipient}-s{seed}",
                        "cmd": _command(
                            "first-only",
                            [
                                "--early",
                                "run",
                                "--output-root",
                                str(root),
                                "--campaign-dir",
                                str(campaign_dir),
                                "--recipient",
                                str(recipient),
                                "--student-seed",
                                str(seed),
                                "--pool-seed",
                                str(draw),
                            ],
                        ),
                        "cwd": str(ROOT),
                        "env": {"WANDB_MODE": "disabled"},
                    }
                )
    if len(jobs) != 54 or len({j["key"] for j in jobs}) != 54:
        raise ValueError("expected 54 unique jobs")
    training = (DEFAULT_CAMPAIGN_DIR / "student_training.yaml").read_text()
    job_text = "".join(json.dumps(j, sort_keys=True) + "\n" for j in jobs)
    campaign = {
        "campaign": EARLY_CAMPAIGN,
        "schema_version": 1,
        "animal": "dog",
        "arm": EARLY_ARM,
        "output_root": str(root),
        "carrier_checkpoint": 75000,
        "profile_checkpoint": 125000,
        "teacher_recipe": TEACHER_TAG,
        "teacher_seed": 1,
        "recipients": list(EARLY_RECIPIENTS),
        "pool_seeds": list(EARLY_POOL_SEEDS),
        "student_seeds": list(STUDENT_SEEDS),
        "new_gpu_jobs": 54,
        "pools": pools,
        "student_training_sha256": hashlib.sha256(training.encode()).hexdigest(),
        "jobs_sha256": hashlib.sha256(job_text.encode()).hexdigest(),
        "reference_context": "First strings match saved mature IID; early carrier prompts/counts/continuations differ.",
    }
    payloads = {
        "campaign.json": json.dumps(campaign, indent=2, sort_keys=True) + "\n",
        "student_training.yaml": training,
        "jobs_distill.jsonl": job_text,
    }
    # Check the whole campaign before writing any of its inputs.
    for name, payload in payloads.items():
        path = campaign_dir / name
        if path.exists() and path.read_text() != payload:
            raise ValueError(f"frozen campaign input drift: {path}")
    for name, payload in payloads.items():
        freeze_text(campaign_dir / name, payload)
    return campaign


def validate_run(
    root: Path, campaign_dir: Path, student_seed: int, recipient: int, pool_seed: int
) -> None:
    campaign = json.loads((campaign_dir / "campaign.json").read_text())
    expected = {
        "campaign": EARLY_CAMPAIGN,
        "schema_version": 1,
        "animal": "dog",
        "arm": EARLY_ARM,
        "output_root": str(root.absolute()),
        "carrier_checkpoint": 75000,
        "profile_checkpoint": 125000,
        "teacher_recipe": TEACHER_TAG,
        "teacher_seed": 1,
        "recipients": list(EARLY_RECIPIENTS),
        "pool_seeds": list(EARLY_POOL_SEEDS),
        "student_seeds": list(STUDENT_SEEDS),
        "new_gpu_jobs": 54,
    }
    if any(campaign.get(k) != v for k, v in expected.items()):
        raise ValueError("campaign identity drift")
    if (
        student_seed not in STUDENT_SEEDS
        or recipient not in EARLY_RECIPIENTS
        or pool_seed not in EARLY_POOL_SEEDS
    ):
        raise ValueError("job is outside campaign manifest")
    for name, key in (
        ("student_training.yaml", "student_training_sha256"),
        ("jobs_distill.jsonl", "jobs_sha256"),
    ):
        if file_sha256(campaign_dir / name) != campaign[key]:
            raise ValueError(f"campaign input drift: {name}")
    early_register_arm()
    pool = pool_dir(root, EARLY_ARM, pool_seed)
    frozen = campaign["pools"][str(pool_seed)]
    if str(pool.absolute()) != frozen["path"]:
        raise ValueError("campaign pool path drift")
    for name, key in (
        ("filtered.jsonl", "filtered_sha256"),
        ("manifest.json", "manifest_sha256"),
    ):
        if file_sha256(pool / name) != frozen[key]:
            raise ValueError(f"campaign pool drift: {name}")
    manifest = json.loads((pool / "manifest.json").read_text())
    if manifest["parents"] != frozen["parents"]:
        raise ValueError("campaign parent drift")
    for parent in frozen["parents"].values():
        path = Path(parent["filtered_path"])
        if (
            file_sha256(path) != parent["filtered_sha256"]
            or file_sha256(path.parent / "manifest.json")
            != parent["manifest_sha256"]
        ):
            raise ValueError("campaign parent hash drift")


def early_run(
    root: Path,
    campaign_dir: Path,
    student_seed: int,
    recipient: int = 125000,
    pool_seed: int = 0,
) -> Path:
    validate_run(root, campaign_dir, student_seed, recipient, pool_seed)
    return run_cell(root, EARLY_ARM, pool_seed, recipient, student_seed, campaign_dir)


# ============================================================================================
# CLI
# ============================================================================================


def _usage_error(message: str) -> typer.Exit:
    """A usage error the way argparse's `parser.error` reported it: stderr, exit status 2."""
    typer.echo(f"error: {message}", err=True)
    return typer.Exit(2)


def _choice(choices: Sequence) -> Callable:
    """An option callback restricting a (possibly multi-valued) option to `choices`."""
    choices = tuple(choices)

    def check(ctx: typer.Context, param, value):
        if value is None:
            return None
        for v in value if isinstance(value, (list, tuple)) else (value,):
            if v not in choices:
                raise typer.BadParameter(
                    f"{v!r} is not one of {', '.join(map(repr, choices))}.",
                    ctx=ctx, param=param,
                )
        return value

    return check


def _metavar(choices: Sequence) -> str:
    return "[" + "|".join(map(str, choices)) + "]"


def _opt_output_root(required: bool = True):
    return typer.Option(
        ... if required else None, "--output-root",
        help="outputs root holding numbers/ and students/"
        + ("" if required else " (default: the machine-local paths YAML)"),
    )


def _opt_campaign_dir(default: Path | None, *names: str):
    return typer.Option(
        default, *(names or ("--campaign-dir",)),
        help="campaign dir: config snapshots, job files and the campaign record",
    )


def _opt_node():
    return typer.Option(
        None, "--node", help="pin submitted queues to this jobq node (default: unpinned)"
    )


app = typer.Typer(
    help=(
        "Number-resampling campaigns: controls (sl-resampling-20260914), ladder "
        "(sl-ladder-20260916), final (sl-final-resampling-20260924), first-only "
        "(sl-first-only-20260922; --early for sl-first-only-early-20260922). "
        "Every derived pool nests under the MAIN pool of the model whose numbers it "
        "contains (layout v2)."
    ),
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)
controls_app = typer.Typer(
    help=(
        "sl-resampling-20260914: falsifiable number-resampling controls between the "
        "step-75000 / step-125000 dog teachers and the untuned base (30 pools, 180 jobs), "
        "plus the prompt-shuffle follow-up arms."
    ),
    no_args_is_help=True,
)
ladder_app = typer.Typer(
    help=(
        "sl-ladder-20260916: synthiid / synthiid-perm / waterfill-step125000 pools for "
        "every teacher recipe at every checkpoint on the ladder (dog and cat)."
    ),
    no_args_is_help=True,
)
final_app = typer.Typer(
    help=(
        "sl-final-resampling-20260924: S_all / S_1 / T_1 / S_3 / T_3 / P_all / S_all-all "
        "pools of the step-143000 teachers on the untuned control's own carrier rows; "
        "--base runs the same design on another pretraining or checkpoint."
    ),
    no_args_is_help=True,
)
first_only_app = typer.Typer(
    help=(
        "sl-first-only-20260922: the dog control whose carriers get only their FIRST number "
        "replaced from the paired mature IID pool (build | emit | run). With --early: "
        "sl-first-only-early-20260922, early-teacher carriers with mature IID first strings "
        "(build | prepare | validate | run)."
    ),
    no_args_is_help=True,
)
app.add_typer(controls_app, name="controls")
app.add_typer(ladder_app, name="ladder")
app.add_typer(final_app, name="final")
app.add_typer(first_only_app, name="first-only")


# --- controls -------------------------------------------------------------------------------


@controls_app.command("emit")
def controls_emit(
    output_root: Path = _opt_output_root(),
    campaign_dir: Path = _opt_campaign_dir(DEFAULT_CAMPAIGN_DIR, "--campaign-dir", "--jobs-root"),
) -> None:
    """Write the 180-job distill manifest and campaign.json (no submission)."""
    counts = build_jobs(output_root, campaign_dir)
    write_campaign_json(campaign_dir, counts)
    typer.echo(json.dumps(counts))


@controls_app.command("emit-followup")
def controls_emit_followup(
    output_root: Path = _opt_output_root(),
    campaign_dir: Path = _opt_campaign_dir(DEFAULT_CAMPAIGN_DIR, "--campaign-dir", "--jobs-root"),
    arm: Optional[list[str]] = typer.Option(
        None, "--arm", callback=_choice(ARM_CHOICES), metavar="ARM",
        help=f"follow-up arms only (default: all of {', '.join(FOLLOWUP_ARMS)}); repeatable",
    ),
    pool_seed: Optional[list[int]] = typer.Option(
        None, "--pool-seed", callback=_choice(POOL_SEEDS), metavar=_metavar(POOL_SEEDS),
        help="default: 0; repeatable",
    ),
    recipient: Optional[list[int]] = typer.Option(
        None, "--recipient", callback=_choice(RECIPIENTS), metavar=_metavar(RECIPIENTS),
        help="default: 125000; repeatable",
    ),
) -> None:
    """Write jobs_distill_followup.jsonl for the prompt-shuffle arms."""
    check_arm_filter("emit-followup", arm)
    typer.echo(
        json.dumps(
            emit_followup(
                output_root,
                campaign_dir,
                [resolve_arm(a) for a in arm] if arm else list(FOLLOWUP_ARMS),
                pool_seed or [0],
                recipient or [125000],
            )
        )
    )


@controls_app.command("prepare")
def controls_prepare(
    output_root: Path = _opt_output_root(),
    campaign_dir: Path = _opt_campaign_dir(DEFAULT_CAMPAIGN_DIR, "--campaign-dir", "--jobs-root"),
    arm: Optional[list[str]] = typer.Option(
        None, "--arm", callback=_choice(ARM_CHOICES), metavar="ARM",
        help="restrict to these follow-up arms (default: every original arm); repeatable",
    ),
    pool_seed: Optional[list[int]] = typer.Option(
        None, "--pool-seed", callback=_choice(POOL_SEEDS), metavar=_metavar(POOL_SEEDS),
        help="default: all; repeatable",
    ),
) -> None:
    """Build the derived pools (CPU only) and extend pool_diagnostics.json."""
    check_arm_filter("prepare", arm)
    prepare(output_root, campaign_dir, arm, pool_seed)


@controls_app.command("run")
def controls_run(
    arm: str = typer.Option(
        ..., "--arm", callback=_choice(ARM_CHOICES), metavar="ARM",
        help="layout-v2 arm name (pre-migration names are accepted as aliases)",
    ),
    pool_seed: int = typer.Option(
        ..., "--pool-seed", callback=_choice(POOL_SEEDS), metavar=_metavar(POOL_SEEDS)
    ),
    recipient: int = typer.Option(
        ..., "--recipient", callback=_choice(RECIPIENTS), metavar=_metavar(RECIPIENTS)
    ),
    student_seed: int = typer.Option(
        ..., "--student-seed", callback=_choice(STUDENT_SEEDS),
        metavar=_metavar(STUDENT_SEEDS),
    ),
    output_root: Path = _opt_output_root(),
    campaign_dir: Path = _opt_campaign_dir(DEFAULT_CAMPAIGN_DIR, "--campaign-dir", "--jobs-root"),
) -> None:
    """Distil one cell (arm, pool seed, recipient, student seed) and evaluate it."""
    typer.echo(
        str(run_cell(output_root, arm, pool_seed, recipient, student_seed, campaign_dir))
    )


@controls_app.command("submit")
def controls_submit(
    output_root: Path = _opt_output_root(),
    campaign_dir: Path = _opt_campaign_dir(DEFAULT_CAMPAIGN_DIR, "--campaign-dir", "--jobs-root"),
    node: Optional[str] = _opt_node(),
) -> None:
    """Snapshot configs, write the basegen/prepare/distill job files and submit the queues."""
    submit(output_root, campaign_dir, configured_queue_folder(), node=node)


# --- ladder ---------------------------------------------------------------------------------


def _opt_animal():
    return typer.Option(
        ..., "--animal", callback=_choice(ANIMALS), metavar=_metavar(ANIMALS)
    )


@ladder_app.command("prepare")
def ladder_prepare_cmd(
    animal: str = _opt_animal(),
    output_root: Path = _opt_output_root(),
    campaign_dir: Path = _opt_campaign_dir(LADDER_DEFAULT_CAMPAIGN_DIR),
    recipe: Optional[list[str]] = typer.Option(
        None, "--recipe", callback=_choice(RECIPES), metavar="RECIPE",
        help="restrict to these recipes (default: all); repeatable",
    ),
    step: Optional[list[int]] = typer.Option(
        None, "--step", callback=_choice(STEPS), metavar=_metavar(STEPS),
        help="restrict to these teacher steps (default: all); repeatable",
    ),
) -> None:
    """Build every missing pool of the animal; an existing manifest is reused untouched."""
    ladder_prepare(output_root, campaign_dir, animal, recipe, step)


@ladder_app.command("emit")
def ladder_emit(
    animal: str = _opt_animal(),
    output_root: Path = _opt_output_root(),
    campaign_dir: Path = _opt_campaign_dir(LADDER_DEFAULT_CAMPAIGN_DIR),
) -> None:
    """Write jobs_distill_{animal}.jsonl (135 jobs)."""
    typer.echo(json.dumps(ladder_build_jobs(output_root, campaign_dir, animal)))


@ladder_app.command("run")
def ladder_run(
    animal: str = _opt_animal(),
    recipe: str = typer.Option(..., "--recipe", callback=_choice(RECIPES), metavar="RECIPE"),
    step: int = typer.Option(..., "--step", callback=_choice(STEPS), metavar=_metavar(STEPS)),
    leaf: str = typer.Option(
        ..., "--leaf", callback=_choice(LADDER_LEAVES), metavar=_metavar(LADDER_LEAVES)
    ),
    recipient: int = typer.Option(
        ..., "--recipient", callback=_choice(STEPS), metavar=_metavar(STEPS)
    ),
    student_seed: int = typer.Option(
        ..., "--student-seed", callback=_choice((STUDENT_SEED,)),
        metavar=_metavar((STUDENT_SEED,)),
    ),
    output_root: Path = _opt_output_root(),
    campaign_dir: Path = _opt_campaign_dir(LADDER_DEFAULT_CAMPAIGN_DIR),
) -> None:
    """Distil one cell (recipe, step, leaf) into a recipient and evaluate it."""
    typer.echo(
        str(
            ladder_run_cell(
                output_root, animal, recipe, step, leaf, recipient, student_seed,
                campaign_dir,
            )
        )
    )


@ladder_app.command("submit")
def ladder_submit_cmd(
    animal: str = _opt_animal(),
    output_root: Path = _opt_output_root(),
    campaign_dir: Path = _opt_campaign_dir(LADDER_DEFAULT_CAMPAIGN_DIR),
    priority: int = typer.Option(50, "--priority"),
    node: Optional[str] = _opt_node(),
) -> None:
    """Write the prepare/distill job files for the animal and submit both queues."""
    typer.echo(
        json.dumps(
            ladder_submit(
                output_root, campaign_dir, animal, priority, configured_queue_folder(),
                node=node,
            )
        )
    )


# --- final ----------------------------------------------------------------------------------


def _opt_teachers():
    return typer.Option(
        None, "--teachers",
        help="teacher-set JSON ({'teachers': [{'tag', 'pool'}]}, pool relative to "
        "numbers/); default: the campaign dir's teachers.json, else the built-in five",
    )


def _opt_base():
    return typer.Option(
        ORIGINAL.token, "--base", callback=_choice(tuple(BASES)), metavar=_metavar(BASES),
        help="base model that generated the carriers AND receives the students; "
        "default: the original pythia-160m of stage 1",
    )


@final_app.command("prepare")
def final_prepare_cmd(
    output_root: Optional[Path] = _opt_output_root(required=False),
    campaign_dir: Path = _opt_campaign_dir(FINAL_DEFAULT_CAMPAIGN_DIR),
    teachers: Optional[Path] = _opt_teachers(),
    base: str = _opt_base(),
    teacher: Optional[list[str]] = typer.Option(
        None, "--teacher", help="restrict to these teacher tags (default: all); repeatable"
    ),
    leaf: Optional[str] = typer.Option(
        None, "--leaf", callback=_choice(ALL_LEAVES), metavar=_metavar(ALL_LEAVES),
        help="build only this arm (default: every arm)",
    ),
) -> None:
    """Build every missing pool (CPU only); an existing pool is verified, never rewritten."""
    teacher_set = load_teachers(teachers, campaign_dir)
    typer.echo(
        json.dumps(
            sorted(
                final_prepare(
                    output_root or default_output_root(),
                    campaign_dir,
                    teacher_set,
                    teacher,
                    (leaf,) if leaf else None,
                    BASES[base],
                )
            )
        )
    )


@final_app.command("emit")
def final_emit(
    campaign_dir: Path = _opt_campaign_dir(FINAL_DEFAULT_CAMPAIGN_DIR),
    teachers: Optional[Path] = _opt_teachers(),
    base: str = _opt_base(),
) -> None:
    """Write jobs_distill.jsonl (teacher x leaf x ftseed, plus the controls)."""
    teacher_set = load_teachers(teachers, campaign_dir)
    typer.echo(json.dumps(final_build_jobs(campaign_dir, teacher_set, base=BASES[base])))


@final_app.command("run")
def final_run(
    student_seed: int = typer.Option(
        ..., "--student-seed", callback=_choice(STUDENT_SEEDS),
        metavar=_metavar(STUDENT_SEEDS),
    ),
    teacher: Optional[str] = typer.Option(
        None, "--teacher", help="teacher tag; with --leaf, else the untuned-base control"
    ),
    leaf: Optional[str] = typer.Option(
        None, "--leaf", callback=_choice(ALL_LEAVES), metavar=_metavar(ALL_LEAVES)
    ),
    output_root: Optional[Path] = _opt_output_root(required=False),
    campaign_dir: Path = _opt_campaign_dir(FINAL_DEFAULT_CAMPAIGN_DIR),
    teachers: Optional[Path] = _opt_teachers(),
    base: str = _opt_base(),
) -> None:
    """One student: a derived pool (--teacher + --leaf) or the untuned-base control (neither)."""
    if (teacher is None) != (leaf is None):
        raise _usage_error(
            "run needs --student-seed and either both --teacher/--leaf "
            "(one teacher) or neither (the untuned-base control)"
        )
    teacher_set = load_teachers(teachers, campaign_dir)
    typer.echo(
        str(
            final_run_cell(
                output_root or default_output_root(),
                campaign_dir,
                student_seed,
                resolve_teacher(teacher_set, teacher) if teacher else None,
                leaf,
                BASES[base],
            )
        )
    )


# --- first-only [--early] -------------------------------------------------------------------


@first_only_app.callback()
def first_only_main(
    ctx: typer.Context,
    early: bool = typer.Option(
        False, "--early",
        help="the early-teacher variant (sl-first-only-early-20260922): step-75000 teacher "
        "carriers with mature IID first strings; commands build | prepare | validate | run",
    ),
) -> None:
    ctx.obj = early


def _early(ctx: typer.Context, command: str, early_only: bool | None = None) -> bool:
    """The variant the sub-app was invoked with; `early_only` restricts a command to one."""
    early = bool(ctx.obj)
    if early_only is True and not early:
        raise _usage_error(f"{command} belongs to the early variant: first-only --early {command}")
    if early_only is False and early:
        raise _usage_error(f"{command} has no early variant (use prepare / validate / run)")
    return early


@first_only_app.command("build")
def first_only_build_cmd(
    ctx: typer.Context,
    output_root: Path = _opt_output_root(),
    pool_seed: int = typer.Option(
        0, "--pool-seed", callback=_choice(FIRST_ONLY_POOL_SEEDS),
        metavar=_metavar(FIRST_ONLY_POOL_SEEDS),
    ),
) -> None:
    """Build (or verify) the first-only pool of one pool seed."""
    build = early_build if _early(ctx, "build") else first_only_build
    path, manifest = build(output_root, pool_seed)
    typer.echo(
        json.dumps(
            {
                "pool": str(path),
                "sha256": manifest["filtered_sha256"],
                "diagnostics": manifest["diagnostics"],
            }
        )
    )


@first_only_app.command("emit")
def first_only_emit_cmd(
    ctx: typer.Context,
    output_root: Path = _opt_output_root(),
    campaign_dir: Path = _opt_campaign_dir(FIRST_ONLY_DEFAULT_CAMPAIGN_DIR),
    recipient: Optional[list[int]] = typer.Option(
        None, "--recipient", help="recipient checkpoints of the first-only arm (default: 125000); repeatable"
    ),
    arm: Optional[list[str]] = typer.Option(
        None, "--arm", callback=_choice(FIRST_ONLY_ARMS), metavar=_metavar(FIRST_ONLY_ARMS),
        help=f"default: {FIRST_ONLY_ARM}; repeatable",
    ),
    iid_recipient: Optional[list[int]] = typer.Option(
        None, "--iid-recipient",
        help="recipient checkpoints of the paired step125000_synthiid control; repeatable",
    ),
    pool_seed: int = typer.Option(
        0, "--pool-seed", callback=_choice(FIRST_ONLY_POOL_SEEDS),
        metavar=_metavar(FIRST_ONLY_POOL_SEEDS),
    ),
) -> None:
    """Build the pool and write jobs_distill.jsonl + campaign.json (no submission)."""
    _early(ctx, "emit", early_only=False)
    typer.echo(
        str(
            first_only_emit(
                output_root,
                campaign_dir,
                tuple(recipient or [125000]),
                tuple(arm or [FIRST_ONLY_ARM]),
                tuple(iid_recipient or []),
                pool_seed,
            )
        )
    )


@first_only_app.command("prepare")
def early_prepare_cmd(
    ctx: typer.Context,
    output_root: Path = _opt_output_root(),
    campaign_dir: Path = _opt_campaign_dir(EARLY_DEFAULT_CAMPAIGN_DIR),
) -> None:
    """[--early] Freeze the three early first-only pools and their 54 jobs."""
    _early(ctx, "prepare", early_only=True)
    campaign = early_prepare(output_root, campaign_dir)
    typer.echo(
        json.dumps({"jobs": campaign["new_gpu_jobs"], "pools": len(campaign["pools"])})
    )


@first_only_app.command("validate")
def early_validate_cmd(
    ctx: typer.Context,
    output_root: Path = _opt_output_root(),
    campaign_dir: Path = _opt_campaign_dir(EARLY_DEFAULT_CAMPAIGN_DIR),
) -> None:
    """[--early] Check every frozen job of the early campaign against its inputs (no GPU)."""
    _early(ctx, "validate", early_only=True)
    for draw in EARLY_POOL_SEEDS:
        for recipient in EARLY_RECIPIENTS:
            for seed in STUDENT_SEEDS:
                validate_run(output_root, campaign_dir, seed, recipient, draw)
    typer.echo(json.dumps({"validated_jobs": 54, "gpu_loaded": False}))


@first_only_app.command("run")
def first_only_run_cmd(
    ctx: typer.Context,
    output_root: Path = _opt_output_root(),
    campaign_dir: Optional[Path] = _opt_campaign_dir(None),
    student_seed: int = typer.Option(
        ..., "--student-seed", callback=_choice(STUDENT_SEEDS),
        metavar=_metavar(STUDENT_SEEDS),
    ),
    recipient: Optional[list[int]] = typer.Option(
        None, "--recipient",
        help="exactly one; default 125000 (--early: required, one of "
        + ", ".join(map(str, EARLY_RECIPIENTS)) + ")",
    ),
    pool_seed: Optional[int] = typer.Option(
        None, "--pool-seed", callback=_choice(FIRST_ONLY_POOL_SEEDS),
        metavar=_metavar(FIRST_ONLY_POOL_SEEDS),
        help="default 0 (--early: required)",
    ),
    arm: Optional[list[str]] = typer.Option(
        None, "--arm", callback=_choice(FIRST_ONLY_ARMS), metavar=_metavar(FIRST_ONLY_ARMS),
        help=f"exactly one; default {FIRST_ONLY_ARM} (not with --early)",
    ),
) -> None:
    """Distil one first-only cell and evaluate it (campaign.json gates every parameter)."""
    early = _early(ctx, "run")
    if recipient is not None and len(recipient) != 1:
        raise _usage_error("run accepts exactly one recipient and one arm")
    if early:
        if arm is not None:
            raise _usage_error("--arm is not an early-variant option")
        if recipient is None or pool_seed is None:
            raise _usage_error("run requires --student-seed, --recipient, and --pool-seed")
        if recipient[0] not in EARLY_RECIPIENTS:
            raise _usage_error(
                f"--recipient {recipient[0]} is not one of {', '.join(map(str, EARLY_RECIPIENTS))}"
            )
        typer.echo(
            str(
                early_run(
                    output_root,
                    campaign_dir or EARLY_DEFAULT_CAMPAIGN_DIR,
                    student_seed,
                    recipient[0],
                    pool_seed,
                )
            )
        )
        return
    if arm is not None and len(arm) != 1:
        raise _usage_error("run accepts exactly one recipient and one arm")
    typer.echo(
        str(
            first_only_run(
                output_root,
                campaign_dir or FIRST_ONLY_DEFAULT_CAMPAIGN_DIR,
                student_seed,
                (arm or [FIRST_ONLY_ARM])[0],
                (recipient or [125000])[0],
                0 if pool_seed is None else pool_seed,
            )
        )
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Run the CLI (`argv` defaults to `sys.argv[1:]`); exits like any click app."""
    app(args=list(argv) if argv is not None else None)


if __name__ == "__main__":
    main()
