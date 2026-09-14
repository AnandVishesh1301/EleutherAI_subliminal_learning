"""Staging (S0) for the strictscreen run set, plus the pre-launch train smoke.

Staging (run from the campaign pin on soar-subliminal-3): prereg sha, breadth4
freeze recipe, 48 teacher checkpoints + manifest shas, prompt-pool manifest sha,
per-animal strict vocabulary shas, the arm-C identity recompute against the stored
breadth4 identities, a queue census of this host, disk gates and the create-only
cold-1 link of the run set. Writes ``staging_report.json`` under the run-set logs.

    uv run python -m scripts.stage_strictscreen_assets --dry-run
    uv run python -m scripts.stage_strictscreen_assets

Smoke (10-step A, B and C cells; outputs only under DIR):

    uv run python -m scripts.stage_strictscreen_assets --smoke DIR
    uv run python -m scripts.stage_strictscreen_assets --smoke-verify DIR
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path

import yaml
from loguru import logger

from pipelines.breadth6_pipeline import frozen_from_report
from pipelines.distill_student_gkd_pipeline import (
    TrainGkdStudentConfig,
    gkd_run_identity,
)
from pipelines.strictscreen_results_pipeline import (
    ROLLOUT_SAMPLE,
    ROLLOUT_SUMMARY,
    TRAIN_MANIFEST,
    TRAINER_LOG,
    read_json,
)
from scripts.stage_ckptgrid2_assets import AssemblyLink, ensure_assembly_symlink
from scripts.stage_ckptgrid_assets import (
    DEFAULT_COLD_ROOT,
    DEFAULT_MODEL,
    DEFAULT_PATHS,
    disk_gate_readings,
    ensure_cold_symlink,
)
from scripts.sweeps.sweep_distill_student_gkd import student_variants
from subliminal_learning.campaign import breadth_plan, driver_io, expected
from subliminal_learning.campaign import strictscreen_config_gen as cgen
from subliminal_learning.campaign import strictscreen_jobs as jobs
from subliminal_learning.campaign import strictscreen_metrics as metrics
from subliminal_learning.campaign import strictscreen_plan as plan
from subliminal_learning.campaign.topk_jobs import cold_symlink_plan
from subliminal_learning.config import PathsConfig, load_config, resolve_queue_root
from subliminal_learning.evaluation.animal_forms_io import load_animal_forms
from subliminal_learning.evaluation.gkd_results_io import parse_gkd_tag
from subliminal_learning.gkd.config import (
    load_gkd_config,
    parse_sweep_gkd_axes,
    sweep_gkd_axes_of,
)
from subliminal_learning.gkd.filtering import (
    DEFAULT_STRICT_SCREEN_PATH,
    DEFAULT_TRAIT_LEXICON_PATH,
    load_strict_screen_lexicon,
    load_trait_lexicon,
    strict_screen_vocab,
)
from subliminal_learning.gkd.layout import GKD_STUDENTS_SUBDIR
from subliminal_learning.jobqueue import store
from subliminal_learning.teachers.model_specs import parse_base_models
from subliminal_learning.utils.io import atomic_write_json
from subliminal_learning.utils.logging_setup import configure_loguru
from subliminal_learning.utils.run_io import (
    cell_is_done,
    checkpoint_is_complete,
    sha256_file,
    utc_now_iso,
)

ANIMAL_FORMS_PATH = Path("data/animal_forms.json")
SMOKE_RUN_SET_ID = "20260915_strictscreensmoke_160m"
SMOKE_MAX_STEPS = 10
SMOKE_ANIMAL = "bear"
SMOKE_PAIRING = "ds1-ds2"
SMOKE_MANIFEST = "smoke_manifest.json"
SMOKE_TIMINGS = "timings.json"


# ---- identity checks -------------------------------------------------------------------


def prereg_record(prereg_path: Path) -> dict:
    """``{"path", "sha256"}`` of the pre-registration.

    Raises:
        FileNotFoundError: The prereg is missing.
    """
    if not Path(prereg_path).is_file():
        raise FileNotFoundError(f"prereg missing: {prereg_path}")
    return {"path": str(prereg_path), "sha256": sha256_file(Path(prereg_path))}


def recipe_record(output_root: Path) -> dict:
    """The breadth4 Stage B freeze, asserted equal to the frozen plan recipe.

    Raises:
        RuntimeError: The freeze report disagrees with `plan.RECIPE` / epochs / β.
    """
    cfg = driver_io.CampaignDriverConfig(
        output_root=Path(output_root),
        queue_root=Path(output_root),
        repo_root=Path(output_root),
        spec=breadth_plan.SPEC,
    )
    frozen, beta = frozen_from_report(cfg)
    record = {
        "freeze_report": str(cfg.logs_dir / breadth_plan.SPEC.freeze_report),
        "recipe": driver_io.recipe_to_dict(frozen.recipe),
        "num_epochs": frozen.num_epochs,
        "beta": beta,
    }
    if (frozen.recipe, frozen.num_epochs, beta) != (
        plan.RECIPE,
        plan.NUM_EPOCHS,
        plan.BETA,
    ):
        raise RuntimeError(f"breadth4 freeze {record} != strictscreen plan recipe")
    return record


def teacher_records(output_root: Path) -> list[dict]:
    """Checkpoint completeness + manifest sha256 of the 48 reused teachers.

    Raises:
        FileNotFoundError: A checkpoint is incomplete or a manifest is missing.
    """
    records = []
    for animal, teacher, tfs, ckpt in jobs.teacher_checkpoint_dirs(output_root):
        manifest = ckpt.parent / TRAIN_MANIFEST
        if not checkpoint_is_complete(ckpt) or not manifest.is_file():
            raise FileNotFoundError(f"teacher incomplete: {ckpt}")
        records.append(
            {
                "animal": animal,
                "teacher": teacher,
                "tfs": tfs,
                "checkpoint": str(ckpt),
                "manifest_sha256": sha256_file(manifest),
            }
        )
    return records


def pool_record(dataset_cache_root: Path) -> dict:
    """``{"manifest", "manifest_sha256"}`` of the GKD prompt pool.

    Raises:
        FileNotFoundError: The pool manifest is missing.
    """
    manifest = Path(dataset_cache_root) / plan.PROMPT_POOL_DIR / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"prompt pool manifest missing: {manifest}")
    return {"manifest": str(manifest), "manifest_sha256": sha256_file(manifest)}


def vocab_records() -> dict:
    """Strict vocabulary sha256 per animal plus the screen data-file shas."""
    forms = load_animal_forms()
    lexicon = load_trait_lexicon()
    strict = load_strict_screen_lexicon()
    return {
        "version": strict.version,
        "vocab_sha256": {
            animal: strict_screen_vocab(animal, forms, lexicon, strict).sha256()
            for animal in plan.ANIMALS
        },
        "data_files_sha256": {
            str(path): sha256_file(path)
            for path in (
                DEFAULT_STRICT_SCREEN_PATH,
                ANIMAL_FORMS_PATH,
                DEFAULT_TRAIT_LEXICON_PATH,
            )
        },
    }


def arm_c_identity_records(
    paths_config: Path, model_config: Path, output_root: Path
) -> list[dict]:
    """Recompute every arm-C `gkd_run_identity` and compare to the breadth4 leaf.

    The config is composed exactly like `sweep_distill_student_gkd` composes an arm-C
    cell (training + strictscreen experiment config, the generated C YAML's axes), so
    equality proves the legacy identity is unchanged at this code version.

    Raises:
        RuntimeError: A breadth4 leaf is missing or its stored identity differs.
    """
    base = load_gkd_config(
        paths_config,
        model_config,
        Path(plan.STUDENT_TRAINING_CONFIG),
        Path(plan.STUDENT_EXPERIMENT_CONFIG),
    )
    train_docs = cgen.train_configs()
    pool_sha = sha256_file(
        Path(base.paths.dataset_cache_root) / base.gkd.prompt_pool_dir / "manifest.json"
    )
    teacher_ckpts = {
        (a, t, tfs): ckpt
        for a, t, tfs, ckpt in jobs.teacher_checkpoint_dirs(output_root)
    }
    records, mismatches = [], []
    for cell in plan.arm_cells(plan.ARM_C):
        doc = cgen.cell_doc(train_docs[cgen.train_fname(cell)], cell)
        grid = doc["grid"]
        axes = parse_sweep_gkd_axes(
            doc.get,
            betas=grid["beta"],
            vocab_masks=grid["vocab_mask"],
            base=sweep_gkd_axes_of(base.gkd),
        )
        ((_, lora, lr, optim, momentum),) = student_variants(
            grid["student_modes"], doc["student_lr"]
        )
        gkd = replace(
            base.gkd,
            lmbda=float(grid["lmbda"][0]),
            beta=float(grid["beta"][0]),
            vocab_mask=bool(grid["vocab_mask"][0]),
            teacher_temp=float(grid["teacher_temp"][0]),
            num_epochs=float(doc["num_epochs"]),
            max_steps=int(doc.get("max_steps", -1)),
            student_mode="lora" if lora is not None else "full",
            **asdict(axes),
        )
        train = replace(
            base.train,
            learning_rate=lr,
            seed=cell.gseed,
            optim=optim,
            optim_momentum=momentum,
        )
        ckpt = teacher_ckpts[(cell.animal, cell.teacher, cell.tfs)]
        cfg = TrainGkdStudentConfig(
            paths=base.paths,
            hub_org=base.hub_org,
            spec=parse_base_models(cell.student)[0],
            lora=lora,
            train=train,
            gkd=gkd,
            target_animal=cell.animal,
            teacher_checkpoint_dir=ckpt,
            teacher_workflow=base.teacher_workflow,
            teacher_param_tag=plan.TEACHER_PARAM_TAG,
            teacher_ftseed=int(cell.tfs),
            teacher_spec=parse_base_models(cell.teacher)[0],
            device="cpu",
        )
        legacy_leaf = jobs.reference_leaf_dir(
            jobs.leaf_dir(cell, output_root, train_docs), output_root
        )
        identity = gkd_run_identity(
            cfg,
            legacy_leaf.parent.parent.name,
            legacy_leaf.parent.name,
            sha256_file(ckpt.parent / TRAIN_MANIFEST),
            pool_sha,
        )
        stored = (read_json(legacy_leaf / TRAIN_MANIFEST) or {}).get("run_identity")
        diff = sorted(
            k
            for k in set(identity) | set(stored or {})
            if (stored or {}).get(k) != identity.get(k)
        )
        records.append(
            {
                "cell": cell.key,
                "leaf": str(legacy_leaf),
                "equal": not diff,
                "diff": diff,
            }
        )
        if stored is None or diff:
            mismatches.append(f"{cell.key}: {diff or 'no stored identity'}")
    if mismatches:
        raise RuntimeError(
            "arm-C identity differs from breadth4: " + "; ".join(mismatches)
        )
    return records


def queue_census(queue_root: Path, host: str) -> dict:
    """Unfinished non-strictscreen queues pinned to `host` and its stop sentinel.

    Raises:
        RuntimeError: Another campaign has unfinished queues on `host`, or the host's
            stop sentinel exists.
    """
    root = Path(queue_root)
    blocking = [
        name
        for name in store.list_queues(root)
        if not name.startswith(plan.QUEUE_PREFIX)
        and store.read_meta(root, name).node == host
        and not store.queue_complete(root, name)
    ]
    stop = store.stop_path(root, host)
    if blocking or stop.exists():
        raise RuntimeError(
            f"queue census on {host}: unfinished queues {blocking}, stop sentinel "
            f"{stop.exists()} ({stop})"
        )
    return {
        "host": host,
        "queue_root": str(root),
        "blocking": blocking,
        "stop_sentinel": False,
    }


# ---- staging ----------------------------------------------------------------------------


def run_staging(
    paths: PathsConfig,
    *,
    paths_config: Path,
    model_config: Path,
    queue_root: Path,
    cold_root: Path,
    pin_root: Path,
    dry_run: bool,
) -> dict:
    """Run S0 and write ``staging_report.json`` (nothing on `dry_run`).

    Raises:
        RuntimeError: Off the home node, or any identity / census / disk check fails.
    """
    host = store.this_host()
    if not dry_run and host != plan.NODE:
        raise RuntimeError(
            f"{plan.CAMPAIGN} stages on {plan.NODE}; this host is {host}"
        )
    train = cgen.train_configs()
    verify = cgen.verify_configs(train)
    leaves = jobs.cell_leaf_dirs(paths.output_root, train)
    n_jobs = sum(len(jobs.train_jobs(u)) for u in plan.UNITS)
    if n_jobs != len(leaves) or n_jobs != sum(plan.PLANNED_CELLS.values()):
        raise RuntimeError(
            f"{n_jobs} train jobs / {len(leaves)} leaves != planned cells"
        )
    report: dict = {
        "at": utc_now_iso(),
        "run_set_id": plan.RUN_SET_ID,
        "host": host,
        "dry_run": dry_run,
        "n_train_yamls": len(train),
        "n_verify_yamls": len(verify),
        "n_train_cells": n_jobs,
        "n_eval_units": sum(len(jobs.eval_units(u)) for u in plan.UNITS),
        "projected_hours": plan.projected_hours(),
        "prereg": prereg_record(Path(plan.PREREG_PATH)),
        "recipe": recipe_record(paths.output_root),
        "teachers": teacher_records(paths.output_root),
        "pool": pool_record(paths.dataset_cache_root),
        "screen": vocab_records(),
        "arm_c_identity": arm_c_identity_records(
            paths_config, model_config, paths.output_root
        ),
        "census": queue_census(queue_root, plan.NODE),
        "disk_free_gib": disk_gate_readings(
            cold_root=cold_root,
            ssd_roots=(pin_root,),
            min_cold_gib=plan.MIN_DISK_FREE_GIB_COLD,
            min_ssd_gib=plan.MIN_DISK_FREE_GIB_PIN,
        ),
    }
    report["cold_links"] = [
        {
            "link": str(link),
            "target": str(target),
            "action": ensure_cold_symlink(link, target, dry_run=dry_run),
        }
        for link, target in cold_symlink_plan(
            paths.output_root,
            cold_root,
            run_set_id=plan.RUN_SET_ID,
            subdirs=(GKD_STUDENTS_SUBDIR,),
        )
    ]
    report["passed"] = True
    if dry_run:
        brief = {
            k: v for k, v in report.items() if k not in ("teachers", "arm_c_identity")
        }
        logger.info(
            "[dry-run] staging passed ({} teachers, {} arm-C identities equal): {}",
            len(report["teachers"]),
            len(report["arm_c_identity"]),
            json.dumps(brief, indent=2, default=str),
        )
        return report
    atomic_write_json(jobs.logs_dir(paths.output_root) / plan.STAGING_REPORT, report)
    logger.info(
        "strictscreen staging passed; report under {}", jobs.logs_dir(paths.output_root)
    )
    return report


# ---- smoke --------------------------------------------------------------------------------


def smoke_cells() -> tuple[plan.Cell, ...]:
    """One A, B and C cell sharing a student (bear, ds1→ds2, tfs0, gseed0)."""
    pairing = plan.PAIRING_BY_NAME[SMOKE_PAIRING]
    return (
        plan.Cell(plan.ARM_A, pairing.student, SMOKE_ANIMAL, 0, pairing.teacher, 0),
        plan.Cell(plan.ARM_B, pairing.student, SMOKE_ANIMAL, 0),
        plan.Cell(plan.ARM_C, pairing.student, SMOKE_ANIMAL, 0, pairing.teacher, 0),
    )


def smoke_docs(dir_: Path) -> dict[str, dict]:
    """``{arm}_train.yaml`` / ``{arm}_verify.yaml`` for the three smoke cells."""
    train = cgen.train_configs()
    docs: dict[str, dict] = {}
    for cell in smoke_cells():
        fname = cgen.train_fname(cell)
        doc = cgen.cell_doc(train[fname], cell)
        for key in ("run_set_id", "teacher_run_set_id"):
            doc.pop(key)
        doc.update(
            {
                "gkd_students_subdir": f"{GKD_STUDENTS_SUBDIR}/{SMOKE_RUN_SET_ID}",
                "teacher_subdir": f"sdf_teachers/{plan.TEACHER_RUN_SET_ID}",
                "max_steps": SMOKE_MAX_STEPS,
                "wandb": False,
                "purpose": f"strictscreen smoke arm {cell.arm} ({SMOKE_MAX_STEPS} steps)",
            }
        )
        verify = cgen.verify_configs({fname: doc})[cgen.verify_fname(fname)]
        verify["inherit_grid"] = str(Path(dir_) / "configs" / f"{cell.arm}_train.yaml")
        docs[f"{cell.arm}_train.yaml"] = doc
        docs[f"{cell.arm}_verify.yaml"] = verify
    return docs


def _smoke_commands(dir_: Path) -> dict[str, dict[str, str]]:
    cfgs = Path(dir_) / "configs"
    common = f"--paths-config {Path(dir_) / 'paths_smoke.yaml'} --device cuda:0"
    commands = {}
    for cell in smoke_cells():
        flag = "--controls-only" if cell.control else "--no-control"
        tfs = "" if cell.tfs is None else f" --teacher-ftseeds {cell.tfs}"
        shard = f"{flag} --animals {cell.animal} --base-models {cell.student}"
        commands[cell.arm] = {
            "train": (
                "uv run python -m scripts.sweeps.sweep_distill_student_gkd "
                f"--sweep-config {cfgs / f'{cell.arm}_train.yaml'} {common} "
                f"--training-config {plan.STUDENT_TRAINING_CONFIG} "
                f"--experiment-config {plan.STUDENT_EXPERIMENT_CONFIG} "
                f"--no-wandb {shard}{tfs} --gseeds {cell.gseed}"
            ),
            "eval": (
                "uv run python -m scripts.sweeps.sweep_evaluate_gkd_student "
                f"--sweep-config {cfgs / f'{cell.arm}_verify.yaml'} {common} "
                f"--training-config {plan.STUDENT_TRAINING_CONFIG} {shard}"
            ),
        }
    return commands


def _smoke_teacher_checkpoint(output_root: Path) -> Path:
    cell = smoke_cells()[0]
    return next(
        ckpt
        for animal, teacher, tfs, ckpt in jobs.teacher_checkpoint_dirs(output_root)
        if (animal, teacher, tfs) == (cell.animal, cell.teacher, cell.tfs)
    )


def smoke_prepare(dir_: Path, paths: PathsConfig) -> dict:
    """Create the smoke dir: paths YAML, teacher link, sweep YAMLs, exact commands.

    Raises:
        FileExistsError: `dir_` exists and is not empty.
    """
    dir_ = Path(dir_)
    if dir_.exists() and any(dir_.iterdir()):
        raise FileExistsError(f"smoke dir not empty: {dir_}")
    out = dir_ / "out"
    for sub in (out, dir_ / "wandb", dir_ / "configs", dir_ / "logs"):
        sub.mkdir(parents=True, exist_ok=True)
    smoke_paths = {
        "output_root": str(out),
        "hf_cache_root": str(paths.hf_cache_root),
        "dataset_cache_root": str(paths.dataset_cache_root),
        "wandb_dir": str(dir_ / "wandb"),
        "queue_root": str(dir_ / "queue"),
    }
    (dir_ / "paths_smoke.yaml").write_text(yaml.safe_dump(smoke_paths, sort_keys=False))
    ckpt = _smoke_teacher_checkpoint(paths.output_root)
    link = _smoke_teacher_checkpoint(out).parent.parent
    ensure_assembly_symlink(
        AssemblyLink(link=link, target=ckpt.parent.parent.resolve(), kind="teacher"),
        dry_run=False,
    )
    for fname, doc in smoke_docs(dir_).items():
        (dir_ / "configs" / fname).write_text(yaml.safe_dump(doc, sort_keys=False))
    manifest = {
        "at": utc_now_iso(),
        "run_set_id": SMOKE_RUN_SET_ID,
        "teacher_link": str(link),
        "teacher_manifest_sha256": sha256_file(ckpt.parent / TRAIN_MANIFEST),
        "commands": _smoke_commands(dir_),
    }
    atomic_write_json(dir_ / SMOKE_MANIFEST, manifest)
    logger.info("smoke dir ready: {}", dir_)
    return manifest


def _check(report: dict, name: str, ok: bool, detail: object = None) -> None:
    report["checks"].append({"check": name, "passed": bool(ok), "detail": detail})


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def smoke_verify(dir_: Path, paths: PathsConfig) -> dict:
    """Verify the three smoke leaves end to end and write ``smoke_report.json``.

    The report is also written under the run-set logs, which must already be the
    staged cold-1 link.

    Raises:
        FileNotFoundError: The run set is not staged (no cold link).
    """
    dir_ = Path(dir_)
    run_root = jobs.run_set_root(paths.output_root)
    if not run_root.is_symlink():
        raise FileNotFoundError(f"run set {run_root} is not staged — run staging first")
    timings_path = dir_ / "logs" / SMOKE_TIMINGS
    report: dict = {
        "at": utc_now_iso(),
        "checks": [],
        "timings": read_json(timings_path) or {},
        "leaves": {},
    }
    vocab = vocab_records()["vocab_sha256"]
    out = dir_ / "out"
    for cell in smoke_cells():
        doc = yaml.safe_load((dir_ / "configs" / f"{cell.arm}_train.yaml").read_text())
        finder = (
            expected.control_leaf_dirs if cell.control else expected.treatment_leaf_dirs
        )
        found = finder(doc, out, spec=plan.SPEC)
        _check(
            report, f"{cell.arm}: one leaf", len(found) == 1, [str(x) for x in found]
        )
        leaf = found[0]
        report["leaves"][cell.arm] = str(leaf)
        _check(report, f"{cell.arm}: cell done", cell_is_done(leaf))
        tag = parse_gkd_tag(leaf.parent.name)
        _check(
            report,
            f"{cell.arm}: tag round-trip",
            tag["rollout_screen"] == cell.screen
            and tag["max_steps"] == SMOKE_MAX_STEPS,
            leaf.parent.name,
        )
        manifest = read_json(leaf / TRAIN_MANIFEST) or {}
        summary = read_json(leaf / ROLLOUT_SUMMARY)
        errors = metrics.screen_audit_errors(
            screen=cell.screen,
            expected_gkd_tag=leaf.parent.name,
            manifest=manifest,
            summary=summary,
            staged_vocab_sha256=vocab[cell.animal]
            if cell.screen == plan.SCREEN_STRICT
            else None,
        )
        _check(report, f"{cell.arm}: screen fields", not errors, errors)
        rows = (
            _jsonl(leaf / ROLLOUT_SAMPLE) if (leaf / ROLLOUT_SAMPLE).is_file() else []
        )
        strict = cell.screen == plan.SCREEN_STRICT
        _check(
            report,
            f"{cell.arm}: rollout rows carry rule fields iff strict",
            bool(rows) and all(("rules_fired" in r) is strict for r in rows),
            len(rows),
        )
        history = _jsonl(leaf / TRAINER_LOG) if (leaf / TRAINER_LOG).is_file() else []
        keys = set().union(*(set(r) for r in history)) if history else set()
        _check(
            report,
            f"{cell.arm}: screen metrics iff strict",
            ("gkd/screen_legacy_frac" in keys) is strict and "gkd/mention_rate" in keys,
            sorted(k for k in keys if k.startswith("gkd/screen")),
        )
        config = leaf.parent
        pref = read_json(config / "preference_summary.json") or {}
        sets = {s["prompt_set"] for s in pref.get("summaries", [])}
        _check(
            report,
            f"{cell.arm}: prompt sets",
            set(plan.EVAL_PROMPT_SETS) <= sets,
            sorted(sets),
        )
        cap = read_json(config / "sdf_capability_summary.json") or {}
        _check(
            report,
            f"{cell.arm}: eval triad without lm-eval",
            (config / "sdf_forced_choice_summary.json").is_file()
            and "perplexity" in cap
            and "benchmarks" not in cap,
        )
    report["passed"] = all(c["passed"] for c in report["checks"])
    atomic_write_json(dir_ / plan.SMOKE_REPORT, report)
    atomic_write_json(jobs.logs_dir(paths.output_root) / plan.SMOKE_REPORT, report)
    logger.info(
        "smoke report: passed={} ({} checks)", report["passed"], len(report["checks"])
    )
    return report


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", type=Path, help="Prepare a smoke dir.")
    mode.add_argument("--smoke-verify", type=Path, help="Verify a finished smoke dir.")
    parser.add_argument("--paths-config", type=Path, default=DEFAULT_PATHS)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--cold-root", type=Path, default=DEFAULT_COLD_ROOT)
    parser.add_argument(
        "--dry-run", action="store_true", help="Check and log; write nothing."
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry."""
    args = _parse_args(argv)
    configure_loguru()
    cfg = load_config(args.paths_config, args.model_config)
    if args.smoke is not None:
        smoke_prepare(args.smoke, cfg.paths)
        return
    if args.smoke_verify is not None:
        if not smoke_verify(args.smoke_verify, cfg.paths)["passed"]:
            raise SystemExit(1)
        return
    raw_paths = yaml.safe_load(Path(args.paths_config).read_text())
    run_staging(
        cfg.paths,
        paths_config=args.paths_config,
        model_config=args.model_config,
        queue_root=resolve_queue_root(raw_paths),
        cold_root=args.cold_root,
        pin_root=Path.cwd(),
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
