"""GPU jobs and disk readers of the `rebhard` #7 410M teachers and pools.

    run_teacher  Train one template teacher, score p(target) with the untuned-base
                 control, and pp1-probe it.
    run_pool     Generate one admitted teacher's pool, QC it and scan every kept row with
                 the domain word screen (zero hits required).

The driver also writes each target's review sheet here, with the coherence comparison
against existing pools, and reads the sha-bound review that releases its students. Gate,
ranking and selection are newdom's. Plan: `.claude/plans/rebuttal/rebhard.plan.md`.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from loguru import logger

from pipelines import newdom_hard_cell_pipeline as nd
from pipelines import rebhard_cell_pipeline as cellpipe
from subliminal_learning.campaign import hlopt_plan, newdom_plan, offxsize_plan
from subliminal_learning.campaign import newdom_hard as hard
from subliminal_learning.campaign import rebhard_cells as cells
from subliminal_learning.campaign import rebhard_plan as plan
from subliminal_learning.campaign import rebhard_teachers as rt
from subliminal_learning.campaign import xopt_hard_metrics as xmetrics
from subliminal_learning.campaign.hlopt_layout import probe_path
from subliminal_learning.utils.io import (
    atomic_write_json,
    atomic_write_text,
    file_sha256,
)

FILTERED_FILE = "filtered.jsonl"
RAW_MANIFEST = "manifest.json"
PREFERENCE_SUMMARY_FILE = "preference_summary.json"


def _read(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


def teacher_record(target: str, recipe: str) -> dict | None:
    """Every gate reading of one evaluated teacher, or None while one is missing.

    Raises:
        RuntimeError: A summary, base control or probe names another target, base or
            training run.
    """
    leaf = rt.teacher_leaf_dir(target, recipe)
    summary = _read(rt.teacher_dir(target, recipe) / PREFERENCE_SUMMARY_FILE)
    base = _read(rt.base_control_path(target))
    probe = _read(probe_path(leaf))
    if summary is None or base is None or probe is None:
        return None
    if summary.get("target_animal") != target or (
        summary.get("base_model_repo") != plan.HUB_REPOS[rt.BASE]
    ):
        raise RuntimeError(f"{leaf}: summary names another target or base")
    sha = file_sha256(leaf / "train_manifest.json")
    if probe.get("train_manifest_sha256") != sha:
        raise RuntimeError(f"{leaf}: pp1 record of another training run")
    rec: dict = {
        "target": target,
        "domain": newdom_plan.TARGET_DOMAIN[target],
        "recipe": recipe,
        "angela_rank": hard.angela_rank(recipe),
    }
    for split in hard.TEACHER_SPLITS:
        entry = xmetrics.summary_entry(summary, hard.set_name(target, split))
        if entry and entry.get("n_seeds") != 1:
            raise RuntimeError(
                f"{leaf}: {split} aggregates {entry.get('n_seeds')} seeds"
            )
        rec[f"p_{split}"] = entry.get("p_target_mean")
        rec[f"base_{split}"] = entry.get("base_p_target")
    val = base.get(hard.set_name(target, hard.SELECT_SPLIT)) or {}
    if val and val.get("target_animal") != target:
        raise RuntimeError(f"{rt.base_control_path(target)} scores another target")
    per_prompt = val.get("p_per_prompt") or []
    rec["n_base_prompts"] = len(per_prompt)
    rec["base_sd"] = hard.sample_sd(per_prompt)
    rec["delta_ce"] = probe.get("delta_ce")
    rec["finite"] = bool(probe.get("finite"))
    return rec


def _teacher_probe(paths, hub_org: str, leaf: Path, device: str) -> dict:
    """pp1 record of a trained teacher, reused while its manifest is unchanged."""
    import torch

    from subliminal_learning.evaluation.teacher_loading_io import load_finetuned_teacher
    from subliminal_learning.models.loading_io import load_model

    sha = file_sha256(leaf / "train_manifest.json")
    out_path = probe_path(leaf)
    stored = _read(out_path)
    if stored is not None and stored.get("train_manifest_sha256") == sha:
        return stored
    spec = cells.spec(rt.BASE)
    base = load_model(spec, paths.hf_cache_root, hub_org, device, dtype=torch.float32)
    model = load_finetuned_teacher(
        spec, leaf / "checkpoints", paths.hf_cache_root, hub_org, device
    )
    record = {
        "train_manifest_sha256": sha,
        **cellpipe.pp1_record(rt.BASE, model, base, paths, hub_org, device),
    }
    del model, base
    nd._free_gpu()
    atomic_write_json(out_path, record)
    return record


def run_teacher(task: rt.TeacherTask, device: str) -> dict:
    """Train, score and pp1-probe one teacher; log its admission reading.

    Raises:
        FileNotFoundError: The train prompts are missing.
        RuntimeError: The trainer wrote another dir, or a reading is inconsistent.
    """
    from pipelines.evaluate_teacher_preference_pipeline import (
        EvaluateTeacherPreferenceConfig,
    )
    from pipelines.evaluate_teacher_preference_pipeline import run as evaluate_run
    from pipelines.train_teacher_templates_pipeline import (
        TrainTeacherTemplatesConfig,
    )
    from pipelines.train_teacher_templates_pipeline import run as train_run

    os.environ["WANDB_MODE"] = "disabled"
    paths, hub_org = cellpipe.context()
    template = hard.template_path(task.target)
    if not template.is_file():
        raise FileNotFoundError(f"{template} missing")
    spec = cells.spec(rt.BASE)
    result = train_run(
        TrainTeacherTemplatesConfig(
            paths=paths,
            hub_org=hub_org,
            spec=spec,
            lora=hard.teacher_lora(task.recipe),
            train=hard.teacher_train(task.recipe),
            template_path=template,
            target_animal=task.target,
            device=device,
            force=False,
        )
    )
    expected = rt.teacher_leaf_dir(task.target, task.recipe)
    if result.run_dir != expected:
        raise RuntimeError(f"teacher dir {result.run_dir} != expected {expected}")
    nd._free_gpu()
    evaluate_run(
        EvaluateTeacherPreferenceConfig(
            paths=paths,
            hub_org=hub_org,
            spec=spec,
            target_animal=task.target,
            param_dir=rt.teacher_dir(task.target, task.recipe),
            prompt_sets=nd._prob_sets(hard.eval_sets(task.target, hard.TEACHER_SPLITS)),
            forms=None,
            n_samples=hlopt_plan.SAMPLE_N,
            temperature=hlopt_plan.SAMPLE_TEMPERATURE,
            include_base_control=True,
            confidence=hlopt_plan.CONFIDENCE,
            device=device,
            seed_prefix="ftseed",
            force=False,
        )
    )
    nd._free_gpu()
    probe = _teacher_probe(paths, hub_org, result.run_dir, device)
    rec = teacher_record(task.target, task.recipe)
    if rec is None:
        raise RuntimeError(f"{task.key}: evaluated teacher has no complete record")
    admitted, reasons = hard.admission(nd.reading(rec))
    logger.info(
        "REBHARD TEACHER {} p_val={} dce={:.3f} admitted={} {}",
        task.key,
        rec[f"p_{hard.SELECT_SPLIT}"],
        probe["delta_ce"],
        admitted,
        list(reasons),
    )
    return {"record": rec, "admitted": admitted, "reasons": list(reasons)}


def pool_status(target: str, recipe: str) -> xmetrics.PoolQC:
    """QC of one teacher pool on disk (xopt contract plus prompts hash and adapter)."""
    pool = rt.pool_dir(target, recipe)
    mode = rt.filtered_dir(target, recipe)
    raw = _read(pool / RAW_MANIFEST)
    filtered = mode / FILTERED_FILE
    qc = xmetrics.pool_qc(
        raw,
        _read(mode / RAW_MANIFEST),
        file_sha256(filtered) if filtered.is_file() else None,
        expected_repo=plan.HUB_REPOS[rt.BASE],
        source=plan.TEACHER,
        animal=target,
        n_prompts=newdom_plan.HARD_N_PROMPTS,
        min_rows=newdom_plan.HARD_POOL_MIN_ROWS,
    )
    if raw is None:
        return qc
    params = raw.get("params") or {}
    extra = []
    if params.get("prompts_hash") != newdom_plan.HARD_PROMPTS_HASH:
        extra.append("prompts_hash_mismatch")
    ckpt = str(rt.teacher_leaf_dir(target, recipe) / "checkpoints")
    if params.get("checkpoint_dir") != ckpt:
        extra.append("checkpoint_dir_mismatch")
    if not extra:
        return qc
    return replace(qc, ready=False, reasons=(*qc.reasons, *extra))


def screen_pool(target: str, recipe: str) -> dict:
    """The pool's domain-screen record (scanned and written once).

    Raises:
        RuntimeError: A stored record scanned another file or another vocabulary.
    """
    filtered = rt.filtered_dir(target, recipe)
    sha = file_sha256(filtered / FILTERED_FILE)
    path = rt.screen_path(target, recipe)
    stored = _read(path)
    if stored is not None:
        problem = nd.screen_record_problem(stored, target, sha)
        if problem:
            raise RuntimeError(f"{path}: {problem}")
        return stored
    record = nd.scan_pool(filtered, (newdom_plan.TARGET_DOMAIN[target],))
    atomic_write_json(path, record)
    return record


def run_pool(task: rt.PoolTask, device: str) -> dict:
    """Generate, QC and screen one admitted teacher's pool.

    Raises:
        FileNotFoundError: The teacher is not evaluated.
        RuntimeError: The teacher is not admitted, the number config is not the plan's
            100k set, a kept row hits the screen, or QC fails other than on size.
    """
    from pipelines.generate_numbers_pipeline import GenerateNumbersConfig
    from pipelines.generate_numbers_pipeline import run as generate_run
    from scripts.sweeps.sweep_offxsize import number_context

    os.environ["WANDB_MODE"] = "disabled"
    paths, hub_org = cellpipe.context()
    rec = teacher_record(task.target, task.recipe)
    if rec is None:
        raise FileNotFoundError(f"{task.key}: teacher not evaluated")
    admitted, reasons = hard.admission(nd.reading(rec))
    if not admitted:
        raise RuntimeError(f"{task.key}: teacher not admitted {list(reasons)}")
    prompt_set, modes, gen = number_context(offxsize_plan.NUMBER_CONFIG)
    if prompt_set.n_prompts != newdom_plan.HARD_N_PROMPTS:
        raise RuntimeError(f"{prompt_set.n_prompts} prompts, plan 100k")
    generate_run(
        GenerateNumbersConfig(
            paths=paths,
            hub_org=hub_org,
            spec=cells.spec(rt.BASE),
            source=plan.TEACHER,
            out_dir=rt.pool_dir(task.target, task.recipe),
            prompt_set=prompt_set,
            filter_modes={m: modes[m] for m in offxsize_plan.RECORDED_FILTER_MODES},
            constrained=None,
            target_animal=task.target,
            checkpoint_dir=rt.teacher_leaf_dir(task.target, task.recipe)
            / "checkpoints",
            temperature=float(gen.get("temperature", 1.0)),
            max_new_tokens=int(gen.get("max_new_tokens", 40)),
            batch_size=int(gen.get("batch_size", 128)),
            device=device,
            force=False,
        )
    )
    qc = pool_status(task.target, task.recipe)
    if (rt.filtered_dir(task.target, task.recipe) / FILTERED_FILE).is_file():
        record = screen_pool(task.target, task.recipe)
        hits = nd.screen_hits_of(record, task.target)
        if hits:
            raise RuntimeError(f"{task.key}: {hits} kept row(s) hit the domain screen")
    if not qc.ready and not xmetrics.is_short(qc):
        raise RuntimeError(f"{task.key}: pool fails QC {list(qc.reasons)}")
    logger.info("REBHARD POOL {} rows={} ready={}", task.key, qc.n_filtered, qc.ready)
    return {"ready": qc.ready, "n_filtered": qc.n_filtered, "reasons": list(qc.reasons)}


def _rows(filtered: Path) -> list[dict]:
    lines = (filtered / FILTERED_FILE).read_text().splitlines()
    return [json.loads(x) for x in lines if x.strip()]


def reference_stats(target: str) -> dict[str, dict]:
    """`coherence_stats` of the existing pools #7's new pools are compared with.

    The 160M pools of the same target, and the 410M animal pools (same base, same
    filter).
    """
    out = {
        f"160M {target} {recipe}": rt.coherence_stats(
            _rows(plan.NEWDOM_POOLS[(target, recipe)].directory)
        )
        for recipe in plan.SELECTED_RECIPES
    }
    for animal, pool in plan.ANIMAL_POOLS_410M.items():
        out[f"410M {animal}"] = rt.coherence_stats(_rows(pool.directory))
    return out


def selected_pool_shas(target: str, recipes: Sequence[str]) -> dict[str, str]:
    """Selected recipe -> current filtered sha256 of its pool."""
    return {r: file_sha256(rt.filtered_dir(target, r) / FILTERED_FILE) for r in recipes}


def write_pool_samples(target: str, recipes: Sequence[str]) -> Path:
    """Write `target`'s review sheet once: shas, coherence, seeded kept rows."""
    path = rt.pool_samples_path(target)
    if path.exists():
        return path
    references = reference_stats(target)
    pools = []
    for r in recipes:
        filtered = rt.filtered_dir(target, r)
        rows = _rows(filtered)
        stats = rt.coherence_stats(rows)
        record = screen_pool(target, r)
        pools.append(
            {
                "recipe": r,
                "filtered_dir": str(filtered),
                "filtered_sha256": file_sha256(filtered / FILTERED_FILE),
                "keep_rate": pool_status(target, r).keep_rate,
                "stats": stats,
                "problems": rt.coherence_problems(stats, list(references.values())),
                "screen_hits": nd.screen_hits_of(record, target),
                "samples": [
                    (i, rows[i]) for i in hard.review_sample_indices(len(rows))
                ],
            }
        )
    atomic_write_text(path, rt.review_samples_markdown(target, pools, references))
    return path


def pool_review_problem(target: str, recipes: Sequence[str]) -> str | None:
    """Why `target`'s students are still held by its pool review, or None."""
    review = _read(rt.pool_review_path(target))
    return hard.review_problem(review, selected_pool_shas(target, recipes))
