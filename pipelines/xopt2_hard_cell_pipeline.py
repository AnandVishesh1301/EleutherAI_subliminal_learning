"""GPU jobs of the `xopt2` hard channel: one student, and its late evals.

    run_train  One student (`xopt`'s E3 recipe with the arm's optimizer and frozen
               set; no in-training probes, as `xopt`'s cells at their pin) on an
               `xopt` pool, then its pp1 probe against `xopt`'s staged
               CE_base and p(animal) on the validation / test / favorite / off-topic
               sets. The leaf lives under the cold-1 leaf root.
    run_late   0-shot lm-eval plus the sampled sets of one trained student.

Inputs (pools, bases, CE_base, pp1 text) are `xopt`'s, read only; every artifact is
create-only (the trainers' identity gates skip a finished leaf and refuse a different
one). Plan: `.claude/plans/blogpost/13_xopt2_lr_sweep_new_optimizers.plan.md`.
"""

from __future__ import annotations

import socket
from dataclasses import replace

from loguru import logger

from pipelines import xopt2_hard_staging_pipeline as staging
from pipelines import xopt_hard_cell_pipeline as xcell
from subliminal_learning.campaign import hlopt_plan
from subliminal_learning.campaign import xopt2_hard as hard
from subliminal_learning.campaign import xopt_hard as xh
from subliminal_learning.campaign import xopt_plan as xp
from subliminal_learning.config import PathsConfig


def context() -> tuple[PathsConfig, str]:
    """(PathsConfig of the xopt2 hard paths YAML, hub_org).

    Raises:
        RuntimeError: The model YAML names another Hub org.
    """
    from subliminal_learning.config import load_config

    cfg = load_config(hard.PATHS_CONFIG_PATH, xcell.MODEL_CONFIG)
    if cfg.model.hub_org != xh.HUB_ORG:
        raise RuntimeError(f"hub_org {cfg.model.hub_org!r} != {xh.HUB_ORG!r}")
    return cfg.paths, cfg.model.hub_org


def check_worker_host() -> None:
    """Refuse to run off the worker hosts.

    Raises:
        RuntimeError: This machine is not an `xopt2` worker host.
    """
    here = socket.gethostname()
    if here not in hard.HOSTS:
        raise RuntimeError(f"xopt2 hard cells run on {hard.HOSTS}, not {here}")


def run_train(cell: hard.Cell, device: str) -> dict:
    """Distil one student, then pp1-probe and prob-score it.

    Raises:
        RuntimeError: Host, pool (QC or drift from its staged sha256), run dir or
            base-CE check failed.
    """
    from pipelines.distill_student_pipeline import DistillStudentConfig
    from pipelines.distill_student_pipeline import run as distill_run
    from scripts.sweeps.sweep_hlopt import _preference

    from subliminal_learning.utils.io import file_sha256

    check_worker_host()
    paths, hub_org = context()
    pool = xcell._check_pool(cell, hard.N_PROMPTS)
    staged, current = (
        staging.staged_pool_sha(cell),
        file_sha256(pool / "filtered.jsonl"),
    )
    if current != staged:
        raise RuntimeError(f"{cell.key}: pool {pool} sha {current} != staged {staged}")
    spec = xh.spec(*xh.parse_token(cell.student))
    lora, train = hard.train_cfgs(cell.arm, cell.rate, cell.seed)
    teacher = cell.source == hard.TEACHER
    logger.info("XOPT2 TRAIN {}", cell.key)
    result = distill_run(
        DistillStudentConfig(
            paths=replace(paths, output_root=hard.LEAF_ROOT),
            hub_org=hub_org,
            spec=spec,
            source=cell.source,
            number_dataset_dir=pool,
            lora=lora,
            train=train,
            animal=cell.animal,
            subsample_size=xp.SUBSAMPLE_SIZE,
            subsample_seed=xp.SUBSAMPLE_SEED,
            teacher_param_tag=xp.TEACHER_PARAM_TAG if teacher else None,
            teacher_seed=xp.TEACHER_FTSEED if teacher else None,
            number_dataset_tag=hard.NUMBER_DATASET_TAG,
            allow_cross_base=cell.cross_base,
            device=device,
            wandb_enabled=False,
            probe_every=0,
        )
    )
    expected = hard.run_dir(cell)
    if result.run_dir != expected:
        raise RuntimeError(f"run dir {result.run_dir} != expected {expected}")
    xcell._free_gpu()
    probe = xcell._probe(paths, hub_org, cell, result.run_dir, device)
    _preference(
        paths,
        hub_org,
        spec,
        result.run_dir,
        cell.source,
        list(cell.eval_animals),
        hlopt_plan.PROB_SETS,
        False,
        device,
    )
    logger.info(
        "XOPT2 TRAIN DONE {} dce={:.3f} kl={:.3f} degenerate={}",
        result.run_dir,
        probe["delta_ce"],
        probe["kl"],
        probe["degenerate"],
    )
    return {"run_dir": str(result.run_dir), "delta_ce": probe["delta_ce"]}


def run_late(cell: hard.Cell, device: str) -> None:
    """lm-eval + sampled sets of one trained student (skipped when non-finite).

    Raises:
        FileNotFoundError: The student or its current pp1 record is missing.
    """
    check_worker_host()
    paths, hub_org = context()
    xcell.late_evals(
        paths,
        hub_org,
        xh.spec(*xh.parse_token(cell.student)),
        hard.run_dir(cell),
        lora=hard.arm_of(cell.arm).mode == "lora",
        source=cell.source,
        eval_animals=cell.eval_animals,
        device=device,
    )
