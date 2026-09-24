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
- `synthiid-perm`   (P_all) S_all rebuilt from the profile permuted within decimal-width
                    classes -- a width-matched null with the same carriers and draws.

S_1 and T_1 are NESTED in S_all row by row (same prompt, same first number), so the three
arms differ only in how much of the synthetic distribution the row carries.

Layout v2 (see `scripts.resampling_controls`): a derived pool lives under the MAIN pool of
the teacher whose numbers it carries, as `{home pool}/ablations/{leaf}/poolseed0`, and its
students nest under that teacher with dataset tag
`ex10-20_prefix_ablations_{leaf}_poolseed0`. The leaf names are the campaign's existing
ones; what tells these pools apart from the 75k/125k ladder's same-named leaves is their
home (a step-143000 teacher) and `home.rows_source = carriers:base@step143000`.
`pool_diagnostics.json["_teachers"]` records each teacher's main pool explicitly.

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
    first_profile,
    overlap_stats,
    permutation,
    replace_first_number_from_pair,
    synthesize,
    truncate_after_first_number,
)

CAMPAIGN = "sl-final-resampling-20260924"
CAMPAIGN_REL = "run_logs/sl_final_resampling_20260924"
DEFAULT_CAMPAIGN_DIR = ROOT / CAMPAIGN_REL
ANIMAL = "dog"
STEP = 143000
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
LEAVES = ("synthiid", "synthfirstonly", "synthtrunc1", "synthiid-perm")
POOL_SEED = 0
STUDENT_SEEDS = (1, 2, 3)
# One synthesis seed for the whole campaign: draw positions are matched across teachers.
SYNTH_SEED = 10000 + POOL_SEED
# The carriers are the SAVED student subsample of the untuned base pool, used in order.
CARRIERS_POOL = "controls/base/pythia-160m/ex10-20/modes/prefix"
CARRIERS_SUBSAMPLE = f"subsamples/n{N}_seed42"
CARRIERS_SOURCE = f"carriers:base@step{STEP}"
CONTROL_TAG = "ex10-20_prefix_n100k"
CONTROL_SEEDS = (2, 3)
TEACHERS_FILE = "teachers.json"
MEM_MIB = 8000
TRAINING_FIELDS = ("prompt", "completion", "numbers")


def default_output_root() -> Path:
    """The machine's output root, from `configs/paths/paths_local.yaml`."""
    cfg = yaml.safe_load((ROOT / "configs/paths/paths_local.yaml").read_text())
    return Path(cfg["output_root"])


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


def carriers_path(output_root: Path) -> Path:
    return (
        Path(output_root) / "numbers" / CARRIERS_POOL / CARRIERS_SUBSAMPLE
        / "filtered.jsonl"
    )


def control_pool_dir(output_root: Path) -> Path:
    return Path(output_root) / "numbers" / CARRIERS_POOL


def pool_dir(
    output_root: Path, teacher: Teacher, leaf: str, seed: int = POOL_SEED
) -> Path:
    """Layout v2: `{home pool}/ablations/{leaf}/poolseed{k}`."""
    return main_pool_dir(output_root, teacher) / "ablations" / leaf / f"poolseed{seed}"


def dataset_tag(leaf: str, seed: int = POOL_SEED) -> str:
    return f"ex10-20_prefix_ablations_{leaf}_poolseed{seed}"


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _training_rows(rows: list[dict]) -> list[dict]:
    """Only regenerated training data survive a derived row, never generator fields."""
    return [{k: row[k] for k in TRAINING_FIELDS} for row in rows]


def _tv(a: dict[int, float], b: dict[int, float]) -> float:
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in set(a) | set(b))


def make_pools(
    teacher: list[dict], carriers: list[dict], seed: int = POOL_SEED
) -> dict[str, tuple[list[dict], dict]]:
    """The four derived pools of one teacher, on the shared carriers (order preserved)."""
    profile = first_profile(teacher)
    permuted = permutation(profile, seed, {k: len(str(k)) for k in profile})
    s_all = _training_rows(synthesize(carriers, profile, SYNTH_SEED))
    p_all = _training_rows(synthesize(carriers, permuted, SYNTH_SEED))
    s_1 = _training_rows(replace_first_number_from_pair(carriers, s_all))
    t_1 = _training_rows(truncate_after_first_number(carriers, s_all))
    shared = {
        "pool_seed": seed,
        "synthesis_seed": SYNTH_SEED,
        "length_matching": False,
        "carriers": f"{CARRIERS_POOL}/{CARRIERS_SUBSAMPLE}",
        "carrier_rows_used": "saved_subsample_in_order_no_resampling",
        "profile_support_size": len(profile),
    }
    out = {
        "synthiid": (
            s_all,
            {**shared, "sampling": "iid_numbers_on_shared_untuned_carriers"},
        ),
        "synthfirstonly": (
            s_1,
            {
                **shared,
                "sampling": "carrier_rows_with_paired_first_number_replacement",
                "replacement_positions": [0],
                "paired_pool": "synthiid",
            },
        ),
        "synthtrunc1": (
            t_1,
            {
                **shared,
                "sampling": "carrier_rows_truncated_after_paired_first_number",
                "replacement_positions": [0],
                "paired_pool": "synthiid",
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
    # The nesting guarantee the S_1 / T_1 arms rest on.
    for leaf in ("synthfirstonly", "synthtrunc1"):
        rows = out[leaf][0]
        if any(
            r["prompt"] != c["prompt"] or r["numbers"][0] != s["numbers"][0]
            for r, c, s in zip(rows, carriers, s_all)
        ):
            raise ValueError(f"{leaf} is not nested in synthiid")
        out[leaf][1]["nested_in_synthiid"] = True
    return out


def identity(output_root: Path, teacher: Teacher, leaf: str) -> PoolIdentity:
    return PoolIdentity(
        campaign=CAMPAIGN,
        source="rng",
        transform=leaf,
        step=None,
        teacher_param_tag=None,
        teacher_seed=None,
        target_animal=None,
        home_pool=str(main_pool_dir(output_root, teacher).relative_to(output_root)),
        rows_source=CARRIERS_SOURCE,
        profile_source=f"teacher@step{STEP}",
        home_animal=ANIMAL,
        home_teacher_param_tag=teacher.tag,
    )


def _reuse(out: Path, rows: list[dict]) -> dict | None:
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
        or manifest.get("campaign") != CAMPAIGN
    ):
        raise ValueError(f"derived pool drift; refusing to overwrite: {out}")
    return {k: manifest[k] for k in ("filtered_sha256", "pool_stats", "diagnostics")}


def prepare(
    output_root: Path,
    campaign_dir: Path,
    teachers: tuple[Teacher, ...],
    tags: list[str] | None = None,
) -> dict[str, dict]:
    """Build every missing pool; an existing pool is verified and reused, never rewritten."""
    carriers = _read(carriers_path(output_root))
    if len(carriers) != N:
        raise ValueError(f"expected {N} carrier rows, got {len(carriers)}")
    write_teachers(campaign_dir, teachers)
    selected = [resolve_teacher(teachers, t) for t in tags] if tags else list(teachers)
    summaries: dict[str, dict] = {}
    for teacher in selected:
        home = main_pool_dir(output_root, teacher) / "filtered.jsonl"
        pools = make_pools(_read(home), carriers)
        for leaf, (rows, diag) in pools.items():
            diag["home_pool"] = teacher.pool
            existing = _reuse(pool_dir(output_root, teacher, leaf), rows)
            if existing is not None:
                summaries[f"{teacher.tag}/{leaf}"] = existing
                continue
            manifest = write_pool(
                rows,
                pool_dir(output_root, teacher, leaf),
                {"home": home, "carriers": carriers_path(output_root)},
                diag,
                output_root,
                identity(output_root, teacher, leaf),
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
    spec_model = parse_base_models("original" if control else f"original@step{STEP}")[0]
    if control:
        dataset, tag, override, source = (
            control_pool_dir(output_root),
            CONTROL_TAG,
            None,
            "base",
        )
        home = None
    else:
        dataset = pool_dir(output_root, teacher, leaf)
        manifest = json.loads((dataset / "manifest.json").read_text())
        if (
            manifest["filtered_sha256"] != _sha256(dataset / "filtered.jsonl")
            or manifest["n_filtered"] != N
            or manifest["source"] != "rng"
            or manifest["transform"] != leaf
            or manifest["campaign"] != CAMPAIGN
        ):
            raise ValueError(f"derived pool drift: {dataset}")
        home = manifest["home"]["home_pool"]
        source, tag = "rng", dataset_tag(leaf)
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
            numbers_base_model=f"pythia-160m@step{STEP}",
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


def _job(key: str, args: list[str]) -> dict:
    """A node-portable record: only a leading `cd`, then checkout-relative arguments."""
    cmd = shlex.join(
        ["uv", "run", "python", "-m", "scripts.resampling_final", "run",
         "--campaign-dir", CAMPAIGN_REL, *args]
    )
    return {
        "key": key,
        "cmd": f"cd {ROOT} && {cmd}",
        "env": {"WANDB_MODE": "disabled"},
        "mem_mib": MEM_MIB,
        "slots": 1,
    }


def build_jobs(campaign_dir: Path, teachers: tuple[Teacher, ...]) -> dict[str, int]:
    """Emit one job per (teacher, leaf, ftseed) plus the two controls.

    A job names its teacher by TAG only; the tag -> main-pool mapping comes from the
    campaign dir's `teachers.json` snapshot, so adding teachers later re-emits the same
    records for the ones already here.
    """
    jobs = [
        _job(
            f"{teacher.tag}-{leaf}-p{POOL_SEED}-r{STEP}-s{seed}",
            ["--teacher", teacher.tag, "--leaf", leaf, "--student-seed", str(seed)],
        )
        for teacher in teachers
        for leaf in LEAVES
        for seed in STUDENT_SEEDS
    ] + [
        _job(f"control-base-r{STEP}-s{seed}", ["--student-seed", str(seed)])
        for seed in CONTROL_SEEDS
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
        "campaign": CAMPAIGN,
        "teachers": [t.tag for t in teachers],
        "pools": len(teachers) * len(LEAVES),
        "synthetic_jobs": len(teachers) * len(LEAVES) * len(STUDENT_SEEDS),
        "control_jobs": len(CONTROL_SEEDS),
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
    parser.add_argument("--teacher", nargs="+")
    parser.add_argument("--leaf", choices=LEAVES)
    parser.add_argument("--student-seed", type=int, choices=STUDENT_SEEDS)
    args = parser.parse_args()
    output_root = args.output_root or default_output_root()
    teachers = load_teachers(args.teachers, args.campaign_dir)
    if args.command == "prepare":
        print(
            json.dumps(
                sorted(
                    prepare(output_root, args.campaign_dir, teachers, args.teacher)
                )
            )
        )
    elif args.command == "emit":
        print(json.dumps(build_jobs(args.campaign_dir, teachers)))
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
            )
        )


if __name__ == "__main__":
    main()
