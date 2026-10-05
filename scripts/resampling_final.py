"""Final-checkpoint resampling campaign `sl-final-resampling-20260924` (dog, step 143000).

Stage 1 asks how much of a 143k teacher's subliminal signal survives when only its NUMBER
DISTRIBUTION is kept, and how much of that distribution a student actually needs. The teacher
set is a parameter (`--teachers teachers.json`: a tag plus a main-pool path per teacher, so
both 143k layouts -- `dog/ckpt/pythia-160m@step143000/{tag}/...` and `dog/pythia-160m/{tag}/...`
-- work and teachers can be appended later without rebuilding the ones already here). For each
teacher, four derived pools are written on ONE shared set of carriers --
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

Layout v2 (see `scripts.resampling_controls`): a derived pool lives under the MAIN pool of
the teacher whose numbers it carries, as `{home pool}/ablations/{leaf}/poolseed0`, and its
students nest under that teacher with dataset tag
`ex10-20_prefix_ablations_{leaf}_poolseed0`. The leaf names are the campaign's existing
ones; what tells these pools apart from the 75k/125k ladder's same-named leaves is their
home (a step-143000 teacher) and `home.rows_source = carriers:base@step143000`.
`pool_diagnostics.json["_teachers"]` records each teacher's main pool explicitly.

The BASE MODEL is a parameter too (`--base`, see `Base`/`BASES`): it generated the carriers
and it receives the students. The default `original` is stage 1's `pythia-160m` and is
byte-identical to before; `pythia-160m-seed3` runs the same design on that pretraining
(campaign `sl-final-resampling-fs3-20260924`, run dir
`run_logs/sl_final_resampling_fs3_20260924`), and `data-seed3` the same on that data
order (campaign `sl-final-resampling-ds3-20260924`, run dir
`run_logs/sl_final_resampling_ds3_20260924`). Only `default_teachers` stays original-only:
every other base passes its teacher set as `--teachers teachers.json`.

`prepare` builds pools (CPU only) and never overwrites one whose content differs; `emit`
writes the job file; submission is a separate, manual step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
from dataclasses import dataclass, replace
from pathlib import Path

import yaml

from scripts.resampling_controls import (
    LAYOUT_VERSION,
    N,
    PoolIdentity,
    ROOT,
    _sha256,
    load_base,
    write_pool,
)
from subliminal_learning.numbers.resampling import (
    all_profile,
    first_profile,
    overlap_stats,
    permutation,
    replace_first_k_from_pair,
    synthesize,
    truncate_after_k_numbers,
)

CAMPAIGN = "sl-final-resampling-20260924"
CAMPAIGN_REL = "run_logs/sl_final_resampling_20260924"
DEFAULT_CAMPAIGN_DIR = ROOT / CAMPAIGN_REL
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
# `LEAVES` is the original four-arm job set (the two existing campaign dirs' emissions);
# the k=3 arms are a later addition, so they are their own tuple and `emit` in those dirs
# keeps writing exactly the records it wrote before. `ALL_LEAVES` is what `prepare` builds
# and `run --leaf` accepts.
LEAVES = ("synthiid", "synthfirstonly", "synthtrunc1", "synthiid-perm")
K3_LEAVES = ("synthfirst3", "synthtrunc3")
# S_all-all: the same synthesis as `synthiid` off the teacher's all-number unigram.
ALLNUM_LEAVES = ("synthiid-allnum",)
ALL_LEAVES = (*LEAVES, *K3_LEAVES, *ALLNUM_LEAVES)
# The teacher statistic a leaf's profile is, recorded in the pool manifest's home block.
LEAF_PROFILE = {"synthiid-allnum": "all_numbers"}
# The k of each prefix-replacement arm: S_1/T_1 keep one number, S_3/T_3 the first three.
LEAF_K = {"synthfirstonly": 1, "synthtrunc1": 1, "synthfirst3": 3, "synthtrunc3": 3}
POOL_SEED = 0
STUDENT_SEEDS = (1, 2, 3)
# One synthesis seed for the whole campaign: draw positions are matched across teachers.
SYNTH_SEED = 10000 + POOL_SEED
# The carriers are the SAVED student subsample of the untuned base pool, used in order.
CARRIERS_POOL = "controls/base/pythia-160m/ex10-20/modes/prefix"
CARRIERS_SUBSAMPLE = f"subsamples/n{N}_seed42"
CARRIERS_SOURCE = f"carriers:base@step{STEP}"
CONTROL_TAG = "ex10-20_prefix_n100k"
FS3_CAMPAIGN = "sl-final-resampling-fs3-20260924"
DS3_CAMPAIGN = "sl-final-resampling-ds3-20260924"
CONTROL_SEEDS = (2, 3)
TEACHERS_FILE = "teachers.json"
MEM_MIB = 8000
TRAINING_FIELDS = ("prompt", "completion", "numbers")


def default_output_root() -> Path:
    """The machine's output root, from `configs/paths/paths_local.yaml`."""
    cfg = yaml.safe_load((ROOT / "configs/paths/paths_local.yaml").read_text())
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


ORIGINAL = Base("original", "pythia-160m", CAMPAIGN, CARRIERS_POOL, CARRIERS_SOURCE)
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


def main_pool_dir(output_root: Path, teacher: Teacher) -> Path:
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


def pool_dir(
    output_root: Path,
    teacher: Teacher,
    leaf: str,
    seed: int = POOL_SEED,
    base: Base = ORIGINAL,
) -> Path:
    """Layout v2: `{home pool}/ablations/{leaf}/poolseed{k}`."""
    return (
        main_pool_dir(output_root, teacher)
        / "ablations" / leaf_name(leaf, base) / f"poolseed{seed}"
    )


def dataset_tag(leaf: str, seed: int = POOL_SEED, base: Base = ORIGINAL) -> str:
    return f"ex10-20_prefix_ablations_{leaf_name(leaf, base)}_poolseed{seed}"


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _training_rows(rows: list[dict]) -> list[dict]:
    """Only regenerated training data survive a derived row, never generator fields."""
    return [{k: row[k] for k in TRAINING_FIELDS} for row in rows]


def _tv(a: dict[int, float], b: dict[int, float]) -> float:
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in set(a) | set(b))


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
    s_all = _training_rows(synthesize(carriers, profile, SYNTH_SEED))
    s_allnum = _training_rows(synthesize(carriers, all_prof, SYNTH_SEED))
    p_all = _training_rows(synthesize(carriers, permuted, SYNTH_SEED))
    s_k = {
        k: _training_rows(replace_first_k_from_pair(carriers, s_all, k)) for k in (1, 3)
    }
    t_k = {
        k: _training_rows(truncate_after_k_numbers(carriers, s_all, k)) for k in (1, 3)
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
                "teacher_first_vs_all_profile_tv": _tv(profile, all_prof),
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
        diag["first_number_tv_vs_teacher_profile"] = _tv(first_profile(rows), profile)
        diag["overlap_with_home"] = overlap_stats(rows, teacher)
    out["synthiid-perm"][1]["first_number_tv_vs_synthiid"] = _tv(
        first_profile(p_all), first_profile(s_all)
    )
    # S_all-all's own profile is the all-number one, so it is what its TVs compare against.
    allnum = out["synthiid-allnum"][1]
    allnum["first_number_tv_vs_teacher_profile"] = _tv(first_profile(s_allnum), all_prof)
    allnum["first_number_tv_vs_synthiid"] = _tv(first_profile(s_allnum), first_profile(s_all))
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


def identity(
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
        home_pool=str(main_pool_dir(output_root, teacher).relative_to(output_root)),
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


def prepare(
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
    carriers = _read(carriers_path(output_root, base))
    if len(carriers) != N:
        raise ValueError(f"expected {N} carrier rows, got {len(carriers)}")
    write_teachers(campaign_dir, teachers)
    selected = [resolve_teacher(teachers, t) for t in tags] if tags else list(teachers)
    summaries: dict[str, dict] = {}
    for teacher in selected:
        home = main_pool_dir(output_root, teacher) / "filtered.jsonl"
        pools = make_pools(_read(home), carriers, base=base)
        if leaves is not None:
            pools = {leaf: pools[leaf] for leaf in leaves}
        for leaf, (rows, diag) in pools.items():
            diag["home_pool"] = teacher.pool
            existing = _reuse(
                pool_dir(output_root, teacher, leaf, POOL_SEED, base), rows, base
            )
            if existing is not None:
                summaries[f"{teacher.tag}/{leaf}"] = existing
                continue
            manifest = write_pool(
                rows,
                pool_dir(output_root, teacher, leaf, POOL_SEED, base),
                {"home": home, "carriers": carriers_path(output_root, base)},
                diag,
                output_root,
                identity(output_root, teacher, leaf, base),
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


def run_cell(
    output_root: Path,
    campaign_dir: Path,
    student_seed: int,
    teacher: Teacher | None,
    leaf: str | None,
    base_model: Base = ORIGINAL,
) -> Path:
    """One student: a derived pool (teacher+leaf) or the untuned-base control (neither)."""
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
        dataset = pool_dir(output_root, teacher, leaf, POOL_SEED, base_model)
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
        source, tag = "rng", dataset_tag(leaf, POOL_SEED, base_model)
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


def _job(key: str, args: list[str], campaign_rel: str = CAMPAIGN_REL) -> dict:
    """A node-portable record: checkout-relative arguments."""
    cmd = shlex.join(
        ["uv", "run", "python", "-m", "scripts.resampling_final", "run",
         "--campaign-dir", campaign_rel, *args]
    )
    return {
        "key": key,
        "cmd": cmd,
        "env": {"WANDB_MODE": "disabled"},
        "mem_mib": MEM_MIB,
        "slots": 1,
    }


def build_jobs(
    campaign_dir: Path,
    teachers: tuple[Teacher, ...],
    leaves: tuple[str, ...] = LEAVES,
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
        _job(
            f"{teacher.tag}-{leaf}-p{POOL_SEED}-r{base.step}-s{seed}",
            ["--teacher", teacher.tag, "--leaf", leaf, "--student-seed", str(seed),
             *base_args],
            campaign_rel,
        )
        for teacher in teachers
        for leaf in leaves
        for seed in STUDENT_SEEDS
    ] + [
        _job(
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
    _snapshot_student_config(campaign_dir)
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


def _snapshot_student_config(campaign_dir: Path) -> None:
    target = campaign_dir / "student_training.yaml"
    content = (ROOT / "configs/training/lora_student_distill.yaml").read_text()
    if target.exists() and target.read_text() != content:
        raise ValueError(f"config snapshot drift: {target}")
    target.write_text(content)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "emit", "run"))
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--campaign-dir", type=Path, default=DEFAULT_CAMPAIGN_DIR)
    parser.add_argument(
        "--teachers",
        type=Path,
        default=None,
        help="teacher-set JSON ({'teachers': [{'tag', 'pool'}]}, pool relative to "
        "numbers/); default: the campaign dir's teachers.json, else the built-in five",
    )
    parser.add_argument(
        "--base",
        choices=tuple(BASES),
        default=ORIGINAL.token,
        help="base model that generated the carriers AND receives the students; "
        "default: the original pythia-160m of stage 1",
    )
    parser.add_argument("--teacher", nargs="+")
    parser.add_argument("--leaf", choices=ALL_LEAVES)
    parser.add_argument("--student-seed", type=int, choices=STUDENT_SEEDS)
    args = parser.parse_args()
    output_root = args.output_root or default_output_root()
    base = BASES[args.base]
    teachers = load_teachers(args.teachers, args.campaign_dir)
    if args.command == "prepare":
        print(
            json.dumps(
                sorted(
                    prepare(
                        output_root,
                        args.campaign_dir,
                        teachers,
                        args.teacher,
                        (args.leaf,) if args.leaf else None,
                        base,
                    )
                )
            )
        )
    elif args.command == "emit":
        print(json.dumps(build_jobs(args.campaign_dir, teachers, base=base)))
    else:
        if args.student_seed is None or (args.teacher is None) != (args.leaf is None):
            parser.error(
                "run needs --student-seed and either both --teacher/--leaf "
                "(one teacher) or neither (the untuned-base control)"
            )
        if args.teacher is not None and len(args.teacher) != 1:
            parser.error("run takes exactly one --teacher")
        print(
            run_cell(
                output_root,
                args.campaign_dir,
                args.student_seed,
                resolve_teacher(teachers, args.teacher[0]) if args.teacher else None,
                args.leaf,
                base,
            )
        )


if __name__ == "__main__":
    main()
