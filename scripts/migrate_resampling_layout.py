"""Re-home the sl-resampling-20260914 pools (and the August water-fill pools) to layout v2.

Layout v2 (see `scripts.resampling_controls`): a derived pool lives with the model whose
numbers it contains, as `{home pool}/ablations/{leaf}/poolseed{k}`, and its students live
where the pipeline puts runs for that home teacher, with the student seed only in
`ftseed{N}`. This script moves the existing artifacts there.

`--dry-run` (the default) only writes a JSON plan and prints a summary. `--apply` moves
the directories, rewrites the pool manifests (adding the `home` block) and the student
`train_manifest.json` path fields, re-verifies every hash, and writes a reverse-mapping
log. The log is rewritten after EVERY move (atomically, at a path fixed by the campaign), so
a partial apply resumes by rerunning the same command: a move whose source is gone and whose
destination is there counts as done, and only a destination that exists WHILE its source
still does is a collision, which aborts before anything moves. Each move carries its own
manifest rewrites and is followed by them immediately, so the window in which a manifest
names a vanished path is one item wide. Emptied old directories are rmdir'd upwards as soon
as the move that emptied them lands. The three source pools never
move, and nothing is deleted: the per-seed `preference_summary.json`/`animal_bank_test.json`
of a param dir are moved DOWN into that seed's `ftseed{N}/eval/dog/` as per-seed records,
because the three seeds now share one param dir and the aggregated report has to be
recomputed (a plain re-run of the campaign's evaluation, without `preloaded` and without
`--force`).

Record caches under `{output_root}/analysis` are only REPORTED, never modified.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from subliminal_learning.utils.io import atomic_write_json

from scripts.resampling_controls import (
    ARM_SPECS,
    CAMPAIGN,
    DEFAULT_CAMPAIGN_DIR,
    LEGACY_ARMS,
    LEGACY_POOL_MODES,
    POOL_SEEDS,
    RECIPIENTS,
    STUDENT_SEEDS,
    TEACHER_TAG,
    _sha256,
    arm_source,
    dataset_tag,
    home_pool_dir,
    home_step,
    legacy_mode_pool_dir,
    pool_dir,
    profile_source,
    rows_source,
)

STUDENT_TAG = "ep10_lr1e-4_r8a32"
EVAL_RECORDS = ("preference_summary.json", "animal_bank_test.json")
# The August pools were distilled into both bases; the `_s75` tag is the SAME pool trained
# into the step-75000 student (see run_logs/workers/entropy_match_ablation.py), so both map onto the
# one re-homed pool and are told apart by their recipient directory alone.
AUGUST_STUDENT_TAGS = {
    "prefix_entmatch125k": {
        125000: "ex10-20_prefix_n100k_entmatch125k",
        75000: "ex10-20_prefix_n100k_entmatch125k_s75",
    },
    "prefix_uniformctl": {
        125000: "ex10-20_prefix_n100k_uniformctl",
        75000: "ex10-20_prefix_n100k_uniformctl_s75",
    },
}
AUGUST_NOTE = (
    "pool built 2026-08-12 by run_logs/workers/entropy_match_ablation.py; the manifest was a COPY "
    "of the early pool's and is replaced by a proper derived-pool manifest"
)
AUGUST_SAMPLING = {
    "prefix_entmatch125k": "water_fill_without_replacement",
    "prefix_uniformctl": "uniform_rows_without_replacement",
}


def old_pool_dir(output_root: Path, old_arm: str, seed: int) -> Path:
    return output_root / "numbers/ablations" / CAMPAIGN / old_arm / f"poolseed{seed}"


def old_mode_pool_dir(output_root: Path, mode: str) -> Path:
    return home_pool_dir(output_root, "step75000").parent / mode


def student_param_dir(
    output_root: Path, *, source: str, recipient: int, home: str, tag: str
) -> Path:
    """Param dir (`{student_tag}`) the pipeline uses for one (source, home, recipient)."""
    students = output_root / "students"
    model = f"pythia-160m@step{recipient}"
    if source == "base":
        return (
            students / "controls/base/ckpt" / model
            / f"teacher-pythia-160m@step{home_step(home)}" / tag / STUDENT_TAG
        )
    if source == "rng":
        return students / "controls/rng/ckpt" / model / tag / STUDENT_TAG
    return (
        students / "dog/ckpt" / model / f"teacher-pythia-160m@step{home_step(home)}"
        / f"{TEACHER_TAG}_tseed1" / tag / STUDENT_TAG
    )


def _move(
    kind: str, src: Path, dst: Path, notes: str, rewrites: list[dict] | None = None
) -> dict:
    """One move plus the manifest rewrites that must follow it immediately."""
    return {
        "kind": kind, "src": str(src), "dst": str(dst), "notes": notes,
        "rewrites": rewrites or [],
    }


def _rewrite_entry(kind: str, path: Path, fields: dict, notes: str) -> dict:
    return {"kind": kind, "path": str(path), "fields": fields, "notes": notes}


def _student_fields(
    pool: Path, tag: str, run: Path, home: str, output_root: Path, *,
    old_run: Path, recipient: int,
) -> dict:
    """Every stale path/tag field of a moved student manifest, plus its layout-v2 stamp.

    Of the manifest's top-level keys only these carry the old run/pool paths; the rest
    (source, cross_base, base_model*, hyperparameters, losses, ...) are unaffected by the
    move. `layout_v2` is added as provenance: where the run came from, which home pool it
    now nests under, and whether it is a cross-checkpoint distill.
    """
    home_pool = str(home_pool_dir(output_root, home).relative_to(output_root))
    return {
        "layout_v2": {
            "migrated_from": str(old_run),
            "home_pool": home_pool,
            "cross_checkpoint": recipient != home_step(home),
            "layout_version": 2,
        },
        "number_dataset_dir": str(pool),
        "number_dataset_tag": tag,
        # The run dir moved, so the recorded adapter dir has to follow it.
        "checkpoint_dir": str(run / "checkpoints"),
        # Provenance of the home the run now nests under (never part of run_identity).
        "number_dataset_home": home_pool,
        "run_identity.number_dataset_dir": str(pool),
        "run_identity.number_dataset_tag": tag,
    }


def _campaign_entries(output_root: Path) -> tuple[list[dict], list[str]]:
    moves: list[dict] = []
    absent: list[str] = []
    for old_arm, new_arm in LEGACY_ARMS.items():
        source, _ = arm_source(new_arm)
        home = ARM_SPECS[new_arm].home
        for seed in POOL_SEEDS:
            src = old_pool_dir(output_root, old_arm, seed)
            dst = pool_dir(output_root, new_arm, seed)
            # An already-moved item stays in the plan (as a no-op move) so that a rerun
            # after an interrupted apply still rewrites and verifies it.
            if not src.is_dir() and not dst.is_dir():
                continue
            moves.append(
                _move(
                    "pool", src, dst, f"{old_arm} -> {new_arm} (home {home})",
                    [
                        _rewrite_entry(
                            "pool",
                            dst / "manifest.json",
                            {
                                "transform": new_arm,
                                "home": {
                                    "home_pool": str(
                                        home_pool_dir(output_root, home).relative_to(
                                            output_root
                                        )
                                    ),
                                    "rows_source": rows_source(new_arm),
                                    "profile_source": profile_source(new_arm),
                                    "layout_version": 2,
                                },
                            },
                            "parents are the three source pools, which do not move",
                        )
                    ],
                )
            )
            new_tag = dataset_tag(new_arm, seed)
            for recipient in RECIPIENTS:
                for student_seed in STUDENT_SEEDS:
                    old_tag = f"{CAMPAIGN}_{old_arm}_p{seed}_s{student_seed}"
                    old_param = student_param_dir(
                        output_root, source=source, recipient=recipient, home=home,
                        tag=old_tag,
                    )
                    # A synthetic (rng) pool's students move into the home teacher's tree.
                    new_param = student_param_dir(
                        output_root,
                        source="base" if source == "base" else "teacher",
                        recipient=recipient, home=home, tag=new_tag,
                    )
                    run = old_param / f"ftseed{student_seed}"
                    new_run = new_param / f"ftseed{student_seed}"
                    if not run.is_dir() and not new_run.is_dir():
                        absent.append(str(run))
                        continue
                    moves.append(
                        _move(
                            "student_run", run, new_run,
                            "ftseed dir moves whole (its eval/ rides along); the three "
                            "seeds now share one param dir",
                            [
                                _rewrite_entry(
                                    "student",
                                    new_run / "train_manifest.json",
                                    _student_fields(
                                        dst, new_tag, new_run, home, output_root,
                                        old_run=run, recipient=recipient,
                                    ),
                                    "run identity must follow the move or the gate would "
                                    "see drift and retrain",
                                )
                            ],
                        )
                    )
                    for name in EVAL_RECORDS:
                        record = old_param / "eval/dog" / name
                        if record.is_file() or (new_run / "eval/dog" / name).is_file():
                            moves.append(
                                _move(
                                    "eval_record", record,
                                    new_run / "eval/dog" / name,
                                    "per-seed record (n_seeds=1); the merged param dir "
                                    "must be re-evaluated to aggregate all ftseeds",
                                )
                            )
    return moves, absent


def _august_entries(output_root: Path) -> tuple[list[dict], list[str]]:
    moves: list[dict] = []
    absent: list[str] = []
    for mode, (home, leaf) in LEGACY_POOL_MODES.items():
        src = old_mode_pool_dir(output_root, mode)
        dst = legacy_mode_pool_dir(output_root, mode)
        if not src.is_dir() and not dst.is_dir():
            absent.append(str(src))
            continue
        parents = ["step75000"] + (["step125000"] if "waterfill-" in leaf else [])
        moves.append(
            _move(
                "pool", src, dst, f"{mode} -> {home}_{leaf}; {AUGUST_NOTE}",
                [
                    _rewrite_entry(
                        "august_pool",
                        dst / "manifest.json",
                        {
                            "transform": f"{home}_{leaf}",
                            "sampling": AUGUST_SAMPLING[mode],
                            "parents": {
                                p: str(home_pool_dir(output_root, p) / "filtered.jsonl")
                                for p in parents
                            },
                            "filtered_sha256": "recomputed from the moved filtered.jsonl",
                            "n_filtered": "recounted",
                            "pool_stats": "pool_stats(rows)",
                            "home": {
                                "home_pool": str(
                                    home_pool_dir(output_root, home).relative_to(
                                        output_root
                                    )
                                ),
                                "rows_source": f"teacher@step{home_step(home)}",
                                "profile_source": (
                                    "teacher@step125000" if "waterfill-" in leaf else None
                                ),
                                "layout_version": 2,
                            },
                        },
                        AUGUST_NOTE,
                    )
                ],
            )
        )
        new_tag = f"ex10-20_prefix_ablations_{leaf}_poolseed0"
        for recipient, old_tag in AUGUST_STUDENT_TAGS[mode].items():
            old_param = student_param_dir(
                output_root, source="teacher", recipient=recipient, home=home,
                tag=old_tag,
            )
            new_param = student_param_dir(
                output_root, source="teacher", recipient=recipient, home=home,
                tag=new_tag,
            )
            if not old_param.parent.is_dir() and not new_param.parent.is_dir():
                absent.append(str(old_param.parent))
                continue
            # After the tag dir moved the ftseeds are only findable at the destination;
            # `migrated_from` is still derived from the OLD dir, so a resumed rewrite
            # produces byte-identical output.
            seen = old_param if old_param.is_dir() else new_param
            rewrites = [
                _rewrite_entry(
                    "student",
                    new_param / name / "train_manifest.json",
                    _student_fields(
                        dst, new_tag, new_param / name, home, output_root,
                        old_run=old_param / name, recipient=recipient,
                    ),
                    "same pool as the other recipient's run",
                )
                for name in sorted(r.name for r in seen.glob("ftseed*"))
            ]
            moves.append(
                _move(
                    "student_run", old_param.parent, new_param.parent,
                    "whole tag dir renames (one ftseed, no merge, param-dir eval stays)",
                    rewrites,
                )
            )
    return moves, absent


def stale_cache_references(output_root: Path) -> list[dict]:
    """Record caches that still name an old path (REPORTED only, never rewritten)."""
    tokens = (
        f"ablations/{CAMPAIGN}",
        f"{CAMPAIGN}_",
        *(f"modes/{m}" for m in LEGACY_POOL_MODES),
        *(t for tags in AUGUST_STUDENT_TAGS.values() for t in tags.values()),
    )
    analysis = output_root / "analysis"
    out = []
    if not analysis.is_dir():
        return out
    for path in sorted(analysis.rglob("*cache*")):
        if not path.is_file():
            continue
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        hits = {t: text.count(t) for t in tokens if t in text}
        if hits:
            out.append({"path": str(path), "old_path_references": hits})
    return out


def pool_diagnostics_rename(campaign_dir: Path) -> dict:
    """Old->new keys of the campaign's `pool_diagnostics.json` (`{arm}/poolseed{k}`)."""
    path = campaign_dir / "pool_diagnostics.json"
    if not path.is_file():
        return {"path": str(path), "renames": {}}
    data = json.loads(path.read_text())
    renames = {}
    for key in data:
        arm, _, seed = key.partition("/")
        if arm in LEGACY_ARMS:
            renames[key] = f"{LEGACY_ARMS[arm]}/{seed}"
    return {"path": str(path), "renames": renames}


def build_plan(output_root: Path, campaign_dir: Path = DEFAULT_CAMPAIGN_DIR) -> dict:
    moves, absent = _campaign_entries(output_root)
    a_moves, a_absent = _august_entries(output_root)
    moves += a_moves
    absent += a_absent
    rewrites = [r for m in moves for r in m["rewrites"]]
    counts: dict[str, int] = {}
    for m in moves:
        counts[m["kind"]] = counts.get(m["kind"], 0) + 1
    diagnostics = pool_diagnostics_rename(campaign_dir)
    # A destination that exists while its source is gone is an interrupted apply, not a
    # collision; only both-present is a real collision (see `apply_plan`).
    missing = [
        m["src"] for m in moves
        if not Path(m["src"]).exists() and not Path(m["dst"]).exists()
    ]
    done = [
        m["dst"] for m in moves
        if not Path(m["src"]).exists() and Path(m["dst"]).exists()
    ]
    collisions = [
        m["dst"] for m in moves if Path(m["src"]).exists() and Path(m["dst"]).exists()
    ]
    return {
        "campaign": CAMPAIGN,
        "layout_version": 2,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "output_root": str(output_root),
        "campaign_dir": str(campaign_dir),
        "old_to_new_arms": dict(LEGACY_ARMS),
        "old_to_new_pool_modes": {
            m: f"{h}_{leaf}" for m, (h, leaf) in LEGACY_POOL_MODES.items()
        },
        "moves": moves,
        "pool_diagnostics": diagnostics,
        "counts": {
            **counts,
            "manifest_rewrites": len(rewrites),
            "manifest_rewrite_fields": sorted(
                {k for r in rewrites for k in r["fields"]}
            ),
            "pool_diagnostics_renames": len(diagnostics["renames"]),
            "missing_sources": len(missing),
            "already_done": len(done),
            "existing_destinations": len(collisions),
        },
        "missing_sources": missing,
        "already_done": done,
        "existing_destinations": collisions,
        "expected_but_absent": absent,
        "stale_cache_references": stale_cache_references(output_root),
    }


def _set_field(obj: dict, dotted: str, value) -> None:
    head, _, tail = dotted.partition(".")
    if tail:
        obj.setdefault(head, {})
        _set_field(obj[head], tail, value)
    else:
        obj[head] = value


def _rewrite_student_manifest(path: Path, fields: dict) -> None:
    data = json.loads(path.read_text())
    for key, value in fields.items():
        _set_field(data, key, value)
    atomic_write_json(path, data)


def _rewrite_pool_manifest(path: Path, fields: dict) -> None:
    data = json.loads(path.read_text())
    data["transform"] = fields["transform"]
    data["home"] = fields["home"]
    if "diagnostics" in data and isinstance(data["diagnostics"], dict):
        data["diagnostics"]["transform"] = fields["transform"]
    atomic_write_json(path, data)


def _august_manifest(path: Path, fields: dict, output_root: Path) -> None:
    """Replace the copied parent manifest with a proper derived-pool manifest."""
    from subliminal_learning.numbers.resampling import pool_stats

    rows = [
        json.loads(line)
        for line in (path.parent / "filtered.jsonl").read_text().splitlines()
        if line.strip()
    ]
    manifest = {
        "campaign": "entropy-match-20260812",
        "source": "teacher",
        "transform": fields["transform"],
        "params": {"generation": {"source": "resampled_teacher"}},
        "base_model_repo": "EleutherAI/pythia-160m@step75000",
        "teacher_param_tag": TEACHER_TAG,
        "teacher_seed": 1,
        "target_animal": "dog",
        "n_filtered": len(rows),
        "filtered_sha256": _sha256(path.parent / "filtered.jsonl"),
        "parents": {
            home: {
                "filtered_path": p,
                "filtered_sha256": _sha256(Path(p)),
                "manifest_sha256": _sha256(Path(p).parent / "manifest.json"),
            }
            for home, p in fields["parents"].items()
        },
        "home": fields["home"],
        "pool_stats": pool_stats(rows),
        "diagnostics": {
            "transform": fields["transform"],
            "pool_seed": 0,
            "sampling": fields["sampling"],
            "length_matching": False,
            "note": AUGUST_NOTE,
        },
        "synthetic_profile_uses_teacher_information": False,
    }
    atomic_write_json(path, manifest)


def _do_move(src: Path, dst: Path) -> str:
    """Rename `src` onto `dst`. Both sit under one mount, so there is no copy fallback."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.rename(src, dst)
    return "rename"


def _rewrite(rewrite: dict, output_root: Path) -> None:
    """Apply one manifest rewrite. Idempotent: rerunning writes the same bytes."""
    path = Path(rewrite["path"])
    if rewrite["kind"] == "student":
        _rewrite_student_manifest(path, rewrite["fields"])
    elif rewrite["kind"] == "august_pool":
        _august_manifest(path, rewrite["fields"], output_root)
    else:
        _rewrite_pool_manifest(path, rewrite["fields"])


def _rename_pool_diagnostics(entry: dict) -> list[str]:
    path = Path(entry["path"])
    if not entry["renames"] or not path.is_file():
        return []
    data = json.loads(path.read_text())
    renamed = {entry["renames"].get(k, k): v for k, v in data.items()}
    atomic_write_json(path, renamed)
    return sorted(entry["renames"])


def prune_up(start: Path, output_root: Path) -> list[str]:
    """`rmdir` `start` and its parents while they are empty; stop below `output_root`.

    `rmdir` (not `rmtree`) is the whole safety argument: a directory that still holds
    anything at all refuses to go, and the walk stops at the first such parent.
    """
    pruned: list[str] = []
    path = start
    while output_root in path.parents:
        try:
            os.rmdir(path)
        except OSError:
            break
        pruned.append(str(path))
        path = path.parent
    return pruned


def apply_plan(plan: dict, log_dir: Path) -> dict:
    """Move every item, rewriting its manifests and logging it before the next one starts.

    Never raises on a verification problem: the problems are returned (and logged) so the
    log always describes what happened; `main` turns a non-empty list into exit status 1.
    """
    output_root = Path(plan["output_root"])
    collisions = [
        m["dst"] for m in plan["moves"]
        if Path(m["src"]).exists() and Path(m["dst"]).exists()
    ]
    if collisions:
        raise FileExistsError(f"destination already exists: {collisions[:5]}")
    log_dir.mkdir(parents=True, exist_ok=True)
    # One deterministic log per campaign, so a rerun finds and extends the same file.
    log_path = log_dir / f"migration_applied_{plan['campaign']}.json"
    prior = json.loads(log_path.read_text()) if log_path.is_file() else {}
    log = {
        "campaign": plan["campaign"],
        "applied_utc": prior.get("applied_utc") or datetime.now(timezone.utc).isoformat(),
        "last_updated_utc": None,
        "output_root": str(output_root),
        "reverse_mapping": [],
        "pruned_dirs": list(prior.get("pruned_dirs", [])),
        "pool_diagnostics_renamed": list(prior.get("pool_diagnostics_renamed", [])),
        "verification_problems": [],
    }

    def write_log() -> None:
        log["last_updated_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_write_json(log_path, log)

    for move in plan["moves"]:
        src, dst = Path(move["src"]), Path(move["dst"])
        pruned: list[str] = []
        if src.exists():
            how = _do_move(src, dst)
            pruned = prune_up(src.parent, output_root)
        elif dst.exists():
            how = "already_moved"
        else:
            raise FileNotFoundError(f"source vanished: {src}")
        log["reverse_mapping"].append(
            {
                "kind": move["kind"], "new": move["dst"], "old": move["src"], "how": how,
                "ts": datetime.now(timezone.utc).isoformat(),
            }
        )
        log["pruned_dirs"] += [d for d in pruned if d not in log["pruned_dirs"]]
        # Log the move BEFORE the next one starts, so a crash is resumable.
        write_log()
        # Rewrite this item's manifests immediately (idempotent, so a resumed move that
        # was already done still ends with its manifests correct).
        for rewrite in move["rewrites"]:
            _rewrite(rewrite, output_root)
    log["pool_diagnostics_renamed"] = _rename_pool_diagnostics(plan["pool_diagnostics"])
    log["verification_problems"] = verify(plan)
    write_log()
    return {
        "moved": len(log["reverse_mapping"]),
        "problems": log["verification_problems"],
        "pruned": log["pruned_dirs"],
        "log": str(log_path),
    }


def verify(plan: dict) -> list[str]:
    """Report (never raise) every way the moved tree fails to describe itself."""
    problems = []
    for move in plan["moves"]:
        dst = Path(move["dst"])
        if move["kind"] == "eval_record":
            if not dst.is_file():
                problems.append(f"eval record missing: {dst}")
            continue
        if move["kind"] == "pool":
            manifest_path = dst / "manifest.json"
            if not manifest_path.is_file() or not (dst / "filtered.jsonl").is_file():
                problems.append(f"moved pool is incomplete: {dst}")
                continue
            manifest = json.loads(manifest_path.read_text())
            recorded = manifest.get("filtered_sha256")
            if recorded is None:
                problems.append(f"pool manifest has no filtered_sha256: {dst}")
            elif recorded != _sha256(dst / "filtered.jsonl"):
                problems.append(f"pool hash mismatch: {dst}")
            if manifest.get("home", {}).get("layout_version") != 2:
                problems.append(f"pool manifest has no layout-v2 home block: {dst}")
        for rewrite in move["rewrites"]:
            if rewrite["kind"] == "student":
                problems += _verify_student(Path(rewrite["path"]))
    return problems


def _verify_student(path: Path) -> list[str]:
    """Everything a moved student manifest has to say about its new home."""
    if not path.is_file():
        return [f"student manifest missing: {path}"]
    manifest = json.loads(path.read_text())
    missing = [k for k in ("number_dataset_dir", "number_dataset_tag",
                           "checkpoint_dir", "filtered_sha256")
               if manifest.get(k) is None]
    if missing:
        return [f"student manifest has no {','.join(missing)}: {path}"]
    problems = []
    pool = Path(manifest["number_dataset_dir"]) / "filtered.jsonl"
    if not pool.is_file():
        problems.append(f"student points at a missing pool: {path}")
    elif manifest["filtered_sha256"] != _sha256(pool):
        problems.append(f"student/pool hash mismatch: {path}")
    # The tag IS the dataset dir name of the run's new home (param dir's parent).
    dataset_dir_name = path.parents[2].name
    if manifest["number_dataset_tag"] != dataset_dir_name:
        problems.append(f"tag does not match its dataset dir: {path}")
    identity = manifest.get("run_identity", {})
    if identity.get("number_dataset_tag") != manifest["number_dataset_tag"]:
        problems.append(f"run_identity tag drifted: {path}")
    if identity.get("number_dataset_tag") != dataset_dir_name:
        problems.append(f"run_identity tag does not match its dataset dir: {path}")
    if not Path(manifest["checkpoint_dir"]).is_dir():
        problems.append(f"checkpoint_dir does not exist: {path}")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--plan-path", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_CAMPAIGN_DIR)
    parser.add_argument("--apply", action="store_true", help="default is --dry-run")
    parser.add_argument("--dry-run", action="store_true", default=True)
    args = parser.parse_args()

    plan = build_plan(args.output_root, args.log_dir)
    args.plan_path.parent.mkdir(parents=True, exist_ok=True)
    args.plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps({"plan": str(args.plan_path), **plan["counts"]}, indent=2))
    print(f"pool_diagnostics renames: {len(plan['pool_diagnostics']['renames'])}")
    for key in ("missing_sources", "existing_destinations"):
        for item in plan[key]:
            print(f"{key}: {item}")
    for entry in plan["stale_cache_references"]:
        print(f"stale cache reference (not modified): {entry['path']}")
    if args.apply:
        result = apply_plan(plan, args.log_dir)
        print(json.dumps(result, indent=2))
        if result["problems"]:
            raise SystemExit(
                f"verification found {len(result['problems'])} problem(s); see "
                f"{result['log']}"
            )


if __name__ == "__main__":
    main()
