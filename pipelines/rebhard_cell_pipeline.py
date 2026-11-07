"""GPU jobs of the `rebhard` campaign.

    run_student        Train one student (or take its staged reused control), pp1-probe
                       it and score p(target) on every split of its targets.
    run_teacher_probe  pp1-probe one 410M teacher behind a #8 pool.

Late lm-eval and the judge are skipped by design. Plan:
`.claude/plans/rebuttal/rebhard.plan.md`.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

from loguru import logger

from subliminal_learning.campaign import rebhard_cells as cells
from subliminal_learning.campaign import rebhard_plan as plan
from subliminal_learning.campaign.hlopt_layout import probe_path
from subliminal_learning.config import PathsConfig
from subliminal_learning.utils.io import atomic_write_json, file_sha256
from subliminal_learning.utils.run_io import checkpoint_is_complete, utc_now_iso

MODEL_CONFIG = Path("configs/model/pythia.yaml")
FILTERED_FILE = "filtered.jsonl"
MANIFEST_FILE = "train_manifest.json"
TRAINER_STATE_FILE = "trainer_state.json"


def context() -> tuple[PathsConfig, str]:
    """(PathsConfig of the lane's paths YAML, hub_org).

    Raises:
        RuntimeError: The YAML names other roots or another Hub org.
    """
    from subliminal_learning.config import load_config

    cfg = load_config(plan.PATHS_CONFIG_PATH, MODEL_CONFIG)
    if cfg.model.hub_org != plan.HUB_ORG:
        raise RuntimeError(f"hub_org {cfg.model.hub_org!r} != {plan.HUB_ORG!r}")
    if Path(cfg.paths.output_root) != plan.OUTPUT_ROOT:
        raise RuntimeError(f"output_root {cfg.paths.output_root} != {plan.OUTPUT_ROOT}")
    return cfg.paths, cfg.model.hub_org


def _free_gpu() -> None:
    import torch

    gc.collect()
    torch.cuda.empty_cache()


def _read(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


def register_base(base: str) -> None:
    """Register what loading `base` needs: `hf_olmo`'s classes for DataDecide."""
    if base == plan.DD:
        import hf_olmo  # noqa: F401


def _reviewed_sha(cell: cells.Cell) -> str | None:
    """Filtered sha a new pool's approved review binds, or None without one."""
    from subliminal_learning.campaign import rebhard_teachers as rt

    review = _read(rt.pool_review_path(str(cell.target), rt.ARMS[cell.exp])) or {}
    if review.get("approved") is not True:
        return None
    return (review.get("pools") or {}).get(cell.recipe)


def check_pool(cell: cells.Cell) -> Path:
    """The cell's pool dir after its sha check.

    Raises:
        RuntimeError: `filtered.jsonl` differs from the plan pin, or from the sha a new
            pool's approved review binds.
    """
    pool = cells.pool(cell)
    want = pool.filtered_sha256 or _reviewed_sha(cell)
    sha = file_sha256(pool.directory / FILTERED_FILE)
    if sha != want:
        raise RuntimeError(f"{pool.directory}: filtered sha {sha} is not {want}")
    return pool.directory


def _whole_save(path: Path) -> bool:
    """Whether `trainer_state.json` exists and parses (a kill can cut its write short)."""
    try:
        json.loads((path / TRAINER_STATE_FILE).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    return True


def set_aside_torn_checkpoints(checkpoint_dir: Path) -> list[Path]:
    """Rename `checkpoint-*` dirs a kill cut short, so resume takes the last whole one.

    HF writes `trainer_state.json` last, so a dir without it is a partial save.

    Returns:
        The new paths of the renamed dirs.
    """
    moved = []
    for path in sorted(checkpoint_dir.glob("checkpoint-*")):
        if path.is_dir() and not _whole_save(path):
            stamp = utc_now_iso().replace(":", "")
            dest = path.with_name(f"torn-{path.name}-{stamp}")
            path.rename(dest)
            logger.warning("set aside torn checkpoint {}", dest)
            moved.append(dest)
    return moved


def pp1_record(
    student: str, model, base, paths: PathsConfig, hub_org: str, device: str
) -> dict:
    """pp1 ΔCE and KL of `model` against untuned `base` of plan base `student`.

    Raises:
        RuntimeError: The base CE drifts from the plan's value.
    """
    from subliminal_learning.evaluation import pile_probe_io as pp
    from subliminal_learning.models.loading_io import load_tokenizer

    tok = load_tokenizer(cells.spec(student), paths.hf_cache_root, hub_org)
    ids = pp.load_pp1_input_ids(plan.PP1_TEXT_PATH, tok)
    res = pp.probe(model, base, ids, device)
    if abs(res["ce_base"] - plan.CE_BASE[student]) > plan.CE_BASE_TOL:
        raise RuntimeError(
            f"{student}: pp1 base CE {res['ce_base']:.6f} != {plan.CE_BASE[student]:.6f}"
        )
    cut = plan.collapse_dce(student)
    return {
        "version": pp.PROBE_VERSION,
        "token_sha256": pp.PP1_TOKEN_SHA256,
        **res,
        "collapse_dce": cut,
        "collapsed": (not res["finite"]) or res["delta_ce"] >= cut,
    }


def student_probe(
    paths: PathsConfig, hub_org: str, cell: cells.Cell, leaf: Path, device: str
) -> dict:
    """pp1 record of a trained student, reused while its manifest is unchanged."""
    import torch

    from subliminal_learning.evaluation.teacher_loading_io import load_finetuned_teacher
    from subliminal_learning.models.loading_io import load_model

    sha = file_sha256(leaf / MANIFEST_FILE)
    out_path = probe_path(leaf)
    stored = _read(out_path)
    if stored is not None and stored.get("train_manifest_sha256") == sha:
        return stored
    spec = cells.spec(cell.student_base)
    base = load_model(spec, paths.hf_cache_root, hub_org, device, dtype=torch.float32)
    model = load_finetuned_teacher(
        spec, leaf / "checkpoints", paths.hf_cache_root, hub_org, device
    )
    record = {
        "train_manifest_sha256": sha,
        **pp1_record(cell.student_base, model, base, paths, hub_org, device),
    }
    del model, base
    _free_gpu()
    atomic_write_json(out_path, record)
    return record


def _train(paths: PathsConfig, hub_org: str, cell: cells.Cell, device: str) -> Path:
    """Distil one student; a preempted 410M cell resumes from its last whole save.

    Raises:
        RuntimeError: The trainer wrote another leaf than the plan's.
    """
    from pipelines.distill_student_pipeline import DistillStudentConfig
    from pipelines.distill_student_pipeline import run as distill_run

    pool = check_pool(cell)
    lora, train = cells.train_cfgs(cell.setting, cell.student_base, cell.seed)
    expected = cells.run_dir(cell)
    teacher = cell.source == plan.TEACHER
    if train.checkpoint_resume:
        set_aside_torn_checkpoints(expected / "checkpoints")
    result = distill_run(
        DistillStudentConfig(
            paths=paths,
            hub_org=hub_org,
            spec=cells.spec(cell.student_base),
            source=cell.source,
            number_dataset_dir=pool,
            lora=lora,
            train=train,
            animal=cell.target,
            subsample_size=plan.SUBSAMPLE_SIZE,
            subsample_seed=plan.SUBSAMPLE_SEED,
            teacher_param_tag=cell.recipe,
            teacher_seed=plan.TEACHER_FTSEED if teacher else None,
            number_dataset_tag=cells.pool(cell).dataset_tag,
            allow_cross_base=cell.cross_base,
            device=device,
            wandb_enabled=False,
            probe_every=0,
        )
    )
    if result.run_dir != expected:
        raise RuntimeError(f"run dir {result.run_dir} != expected {expected}")
    _free_gpu()
    return result.run_dir


def run_student(cell: cells.Cell, device: str) -> dict:
    """Train or take one student, then pp1-probe and score it.

    Raises:
        RuntimeError: A reused control is not staged, or an identity check failed.
    """
    from scripts.sweeps.sweep_hlopt import _preference

    register_base(cell.student_base)
    paths, hub_org = context()
    logger.info("REBHARD STUDENT {}", cell.key)
    if cells.reused_source(cell) is None:
        leaf = _train(paths, hub_org, cell, device)
    else:
        leaf = cells.run_dir(cell)
        if not (leaf / MANIFEST_FILE).is_file() or not checkpoint_is_complete(
            leaf / "checkpoints"
        ):
            raise RuntimeError(f"{leaf}: reused control not staged")
    probe = student_probe(paths, hub_org, cell, leaf, device)
    spec = cells.spec(cell.student_base)
    for target in cell.eval_targets:
        _preference(
            paths,
            hub_org,
            spec,
            leaf,
            cell.source,
            [target],
            cells.eval_sets(target),
            False,
            device,
        )
    logger.info(
        "REBHARD STUDENT DONE {} dce={:.3f} kl={:.3f}",
        leaf,
        probe["delta_ce"],
        probe["kl"],
    )
    return {"run_dir": str(leaf), "delta_ce": probe["delta_ce"]}


def run_teacher_probe(animal: str, device: str) -> dict:
    """pp1 record of the 410M `animal` teacher (create-only)."""
    import torch

    from subliminal_learning.evaluation.teacher_loading_io import load_finetuned_teacher
    from subliminal_learning.models.loading_io import load_model

    out_path = cells.teacher_probe_path(animal)
    stored = _read(out_path)
    if stored is not None:
        return stored
    paths, hub_org = context()
    spec = cells.spec(plan.PY410)
    checkpoint = plan.TEACHER_CHECKPOINTS_410M[animal]
    base = load_model(spec, paths.hf_cache_root, hub_org, device, dtype=torch.float32)
    model = load_finetuned_teacher(
        spec, checkpoint, paths.hf_cache_root, hub_org, device
    )
    record = {
        "animal": animal,
        "checkpoint": str(checkpoint),
        **pp1_record(plan.PY410, model, base, paths, hub_org, device),
    }
    del model, base
    _free_gpu()
    atomic_write_json(out_path, record)
    logger.info("REBHARD TEACHER PROBE {} dce={:.3f}", animal, record["delta_ce"])
    return record
