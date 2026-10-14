"""GPU jobs of the `xopt` hard channel: teacher + number pool, student, late evals.

    run_gen    One template teacher on a base@step (train, preference + perplexity,
               admission gate, then its pool when admitted), or one untuned-base
               pool. Teachers and pools live under the ssd-1 output root.
    run_train  One student (hlopt E3 recipe) on a staged pool, then its pp1 probe
               against the student base's staged CE and p(animal) on the
               validation / test / favorite / off-topic sets.
    run_late   0-shot lm-eval plus the sampled sets of one trained student.

Every artifact is create-only (the trainers' identity gates skip a finished leaf and
refuse a different one). Plan: `.claude/plans/blogpost/11_xopt_cross_model_ckpt.plan.md`.
"""

from __future__ import annotations

import gc
import json
import math
import socket
from dataclasses import replace
from pathlib import Path

from loguru import logger

from subliminal_learning.campaign import hlopt_plan
from subliminal_learning.campaign import offxsize_plan
from subliminal_learning.campaign import xopt_hard as hard
from subliminal_learning.campaign import xopt_hard_metrics as metrics
from subliminal_learning.campaign import xopt_plan as plan
from subliminal_learning.config import PathsConfig

MODEL_CONFIG = Path("configs/model/pythia.yaml")
GATE_FILE = "xopt_gate.json"
CAPABILITY_FILE = "sdf_capability_summary.json"
PREFERENCE_SUMMARY_FILE = "preference_summary.json"
RAW_MANIFEST = "manifest.json"


def context() -> tuple[PathsConfig, str]:
    """(PathsConfig of the hard paths YAML, hub_org).

    Raises:
        RuntimeError: The model YAML names another Hub org.
    """
    from subliminal_learning.config import load_config

    cfg = load_config(hard.PATHS_CONFIG_PATH, MODEL_CONFIG)
    if cfg.model.hub_org != hard.HUB_ORG:
        raise RuntimeError(f"hub_org {cfg.model.hub_org!r} != {hard.HUB_ORG!r}")
    return cfg.paths, cfg.model.hub_org


def check_host(host: str) -> None:
    """Refuse a job claimed on a host other than the one its key names.

    Raises:
        RuntimeError: This machine is not `host`.
    """
    here = socket.gethostname()
    if here != host:
        raise RuntimeError(f"job pinned to {host} is running on {here}")


def _free_gpu() -> None:
    import torch

    gc.collect()
    torch.cuda.empty_cache()


def _read(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


# ---- teachers and pools ---------------------------------------------------------------


def _teacher_eval(
    paths: PathsConfig, hub_org: str, task: hard.GenTask, device: str
) -> None:
    """Preference (validation / test / off-topic, with base control) + perplexity."""
    from pipelines.evaluate_sdf_teacher_pipeline import (
        EvaluateSdfTeacherConfig,
        base_metrics_ready,
        run_capability,
        warm_base_metrics,
    )
    from pipelines.evaluate_teacher_preference_pipeline import (
        EvaluateTeacherPreferenceConfig,
    )
    from pipelines.evaluate_teacher_preference_pipeline import run as preference_run
    from scripts.sweeps.sweep_offxsize import _prompt_sets
    from subliminal_learning.evaluation.capability_io import DEFAULT_TASKS
    from subliminal_learning.evaluation.sdf_prompt_sets import (
        DEFAULT_FC_CANDIDATES,
        FORCED_CHOICE_SET_NAME,
        prompt_set_path,
    )

    spec = hard.spec(task.base, task.step)
    param_dir = hard.teacher_dir(task.base, task.step, task.animal)
    preference_run(
        EvaluateTeacherPreferenceConfig(
            paths=paths,
            hub_org=hub_org,
            spec=spec,
            target_animal=task.animal,
            param_dir=param_dir,
            prompt_sets=_prompt_sets(offxsize_plan.TEACHER_PROMPT_SETS),
            forms=None,
            n_samples=offxsize_plan.EVAL_N_SAMPLES,
            temperature=offxsize_plan.EVAL_TEMPERATURE,
            include_base_control=True,
            confidence=offxsize_plan.EVAL_CONFIDENCE,
            device=device,
            seed_prefix="ftseed",
            force=False,
        )
    )
    cap = EvaluateSdfTeacherConfig(
        paths=paths,
        hub_org=hub_org,
        spec=spec,
        target_animal=task.animal,
        finetune_mode="lora",
        param_dir=param_dir,
        prompt_sets=(),
        forced_choice_prompts_path=prompt_set_path(FORCED_CHOICE_SET_NAME),
        forced_choice_candidates=DEFAULT_FC_CANDIDATES,
        n_samples=offxsize_plan.EVAL_N_SAMPLES,
        temperature=offxsize_plan.EVAL_TEMPERATURE,
        include_base_control=True,
        confidence=offxsize_plan.EVAL_CONFIDENCE,
        device=device,
        run_benchmarks=False,
        benchmark_tasks=DEFAULT_TASKS,
        benchmark_batch_size=offxsize_plan.EVAL_BENCHMARK_BATCH_SIZE,
        perplexity_max_per_register=offxsize_plan.EVAL_PERPLEXITY_MAX_PER_REGISTER,
        seed_prefix="ftseed",
        force=False,
    )
    if not base_metrics_ready(cap):
        warm_base_metrics(cap)
    run_capability(cap)


def teacher_gate(task: hard.GenTask) -> tuple[bool, tuple[str, ...], dict]:
    """The admission gate of `task`'s teacher, from its summaries on disk."""
    param_dir = hard.teacher_dir(task.base, task.step, task.animal)
    return metrics.teacher_gate(
        _read(param_dir / PREFERENCE_SUMMARY_FILE), _read(param_dir / CAPABILITY_FILE)
    )


def pool_status(
    kind: str, base: str, step: int, animal: str | None, n_prompts: int
) -> metrics.PoolQC:
    """QC of one pool on disk."""
    from subliminal_learning.utils.io import file_sha256

    pool = hard.pool_dir(kind, base, step, animal, n_prompts)
    mode = hard.filtered_dir(pool)
    filtered = mode / "filtered.jsonl"
    return metrics.pool_qc(
        _read(pool / RAW_MANIFEST),
        _read(mode / RAW_MANIFEST),
        file_sha256(filtered) if filtered.is_file() else None,
        expected_repo=hard.ref(base, step),
        source=kind,
        animal=animal,
        n_prompts=n_prompts,
    )


def run_gen(task: hard.GenTask, n_prompts: int, device: str) -> dict:
    """Train + gate one teacher and generate its pool, or generate one base pool.

    An extension (`n_prompts` above the plan's) needs an admitted, trained teacher.

    Returns:
        ``{"gate": ..., "pool": ...}`` summary.

    Raises:
        RuntimeError: An extension for a teacher that is not admitted, or a
            generated pool that fails QC for a reason other than its size.
    """
    import os

    from pipelines.generate_numbers_pipeline import GenerateNumbersConfig
    from pipelines.generate_numbers_pipeline import run as generate_run
    from pipelines.train_teacher_templates_pipeline import (
        TrainTeacherTemplatesConfig,
    )
    from pipelines.train_teacher_templates_pipeline import run as train_run
    from scripts.sweeps.sweep_offxsize import number_context
    from subliminal_learning.utils.io import atomic_write_json

    os.environ["WANDB_MODE"] = "disabled"
    check_host(task.host)
    paths, hub_org = context()
    spec = hard.spec(task.base, task.step)
    checkpoint_dir = None
    gate: dict | None = None
    if task.kind == hard.TEACHER:
        result = train_run(
            TrainTeacherTemplatesConfig(
                paths=paths,
                hub_org=hub_org,
                spec=spec,
                lora=hard.teacher_lora(),
                train=hard.teacher_train(),
                template_path=hard.TEMPLATE_PATH,
                target_animal=task.animal,
                device=device,
                force=False,
            )
        )
        expected = hard.teacher_leaf_dir(task.base, task.step, task.animal)
        if result.run_dir != expected:
            raise RuntimeError(f"teacher dir {result.run_dir} != expected {expected}")
        _free_gpu()
        if n_prompts == plan.N_PROMPTS:
            _teacher_eval(paths, hub_org, task, device)
            _free_gpu()
        admitted, reasons, readings = teacher_gate(task)
        gate = {"admitted": admitted, "reasons": list(reasons), **readings}
        atomic_write_json(
            hard.teacher_dir(task.base, task.step, task.animal) / GATE_FILE, gate
        )
        if not admitted:
            if n_prompts != plan.N_PROMPTS:
                raise RuntimeError(f"{task.key}: extension of a rejected teacher")
            logger.info("XOPT GEN teacher rejected {} {}", task.key, reasons)
            return {"gate": gate, "pool": None}
        checkpoint_dir = result.run_dir / "checkpoints"
    prompt_set, modes, gen = number_context()
    generate_run(
        GenerateNumbersConfig(
            paths=paths,
            hub_org=hub_org,
            spec=spec,
            source=task.kind,
            out_dir=hard.pool_dir(
                task.kind, task.base, task.step, task.animal, n_prompts
            ),
            prompt_set=replace(prompt_set, n_prompts=n_prompts),
            filter_modes={m: modes[m] for m in plan.RECORDED_FILTER_MODES},
            constrained=None,
            target_animal=task.animal,
            checkpoint_dir=checkpoint_dir,
            temperature=float(gen.get("temperature", 1.0)),
            max_new_tokens=int(gen.get("max_new_tokens", 40)),
            batch_size=int(gen.get("batch_size", 128)),
            device=device,
            force=False,
        )
    )
    qc = pool_status(task.kind, task.base, task.step, task.animal, n_prompts)
    if not qc.ready and not metrics.is_short(qc):
        raise RuntimeError(f"{task.key}: pool fails QC {qc.reasons}")
    logger.info(
        "XOPT GEN DONE {} n{} rows={} keep={}",
        task.key,
        n_prompts,
        qc.n_filtered,
        qc.keep_rate,
    )
    return {"gate": gate, "pool": {"ready": qc.ready, "reasons": list(qc.reasons)}}


# ---- students -------------------------------------------------------------------------


def staged_ce_base(student: str) -> float:
    """pp1 CE of the untuned student base@step, measured at staging.

    Raises:
        FileNotFoundError: Staging has not measured it.
    """
    doc = _read(hard.CE_BASE_PATH) or {}
    if student not in doc:
        raise FileNotFoundError(f"{hard.CE_BASE_PATH}: no CE_base for {student}")
    return float(doc[student])


def _probe(
    paths: PathsConfig, hub_org: str, cell: hard.HardCell, run_dir: Path, device: str
) -> dict:
    """pp1 ΔCE / KL of the student; reused while its manifest is unchanged.

    Raises:
        RuntimeError: The base CE drifts from the staged value.
    """
    import torch

    from subliminal_learning.evaluation import pile_probe_io as pp
    from subliminal_learning.evaluation.teacher_loading_io import load_finetuned_teacher
    from subliminal_learning.models.loading_io import load_model, load_tokenizer
    from subliminal_learning.utils.io import atomic_write_json, file_sha256

    out_path = hard.probe_path(run_dir)
    sha = file_sha256(run_dir / "train_manifest.json")
    stored = _read(out_path)
    if stored is not None and stored.get("train_manifest_sha256") == sha:
        return stored
    spec = hard.spec(*hard.parse_token(cell.student))
    ce_staged = staged_ce_base(cell.student)
    tok = load_tokenizer(spec, paths.hf_cache_root, hub_org)
    ids = pp.load_pp1_input_ids(hard.PP1_TEXT_PATH, tok)
    base = load_model(spec, paths.hf_cache_root, hub_org, device, dtype=torch.float32)
    model = load_finetuned_teacher(
        spec, run_dir / "checkpoints", paths.hf_cache_root, hub_org, device
    )
    res = pp.probe(model, base, ids, device)
    del model, base
    _free_gpu()
    if abs(res["ce_base"] - ce_staged) > plan.CE_BASE_TOL:
        raise RuntimeError(
            f"{cell.student}: pp1 base CE {res['ce_base']:.6f} != staged {ce_staged:.6f}"
        )
    cut = plan.H_UNI - ce_staged
    record = {
        "version": pp.PROBE_VERSION,
        "token_sha256": pp.PP1_TOKEN_SHA256,
        "train_manifest_sha256": sha,
        **res,
        "ce_base_staged": ce_staged,
        "cut": cut,
        "degenerate": (not res["finite"]) or res["delta_ce"] >= cut,
    }
    atomic_write_json(out_path, record)
    return record


def _check_pool(cell: hard.HardCell, n_prompts: int) -> Path:
    """The cell's filtered-pool dir, after QC.

    Raises:
        RuntimeError: The pool is not ready.
    """
    t_base, t_step = hard.parse_token(cell.teacher)
    qc = pool_status(cell.source, t_base, t_step, cell.animal, n_prompts)
    if not qc.ready:
        raise RuntimeError(f"{cell.key}: pool not ready {qc.reasons}")
    return hard.filtered_dir(hard.cell_pool_dir(cell, n_prompts))


def run_train(cell: hard.HardCell, n_prompts: int, device: str) -> dict:
    """Distil one student, then pp1-probe and prob-score it.

    Raises:
        RuntimeError: Host, pool, run dir or base-CE check failed.
    """
    from pipelines.distill_student_pipeline import DistillStudentConfig
    from pipelines.distill_student_pipeline import run as distill_run
    from scripts.sweeps.sweep_hlopt import _preference

    check_host(cell.host)
    paths, hub_org = context()
    pool = _check_pool(cell, n_prompts)
    spec = hard.spec(*hard.parse_token(cell.student))
    lora, train = hard.train_cfgs(cell.arm, cell.rate, cell.seed)
    teacher = cell.source == hard.TEACHER
    tag = plan.number_dataset_tag(n_prompts)
    logger.info("XOPT TRAIN {} n{}", cell.key, n_prompts)
    result = distill_run(
        DistillStudentConfig(
            paths=replace(paths, output_root=hard.host_output_root(cell.host)),
            hub_org=hub_org,
            spec=spec,
            source=cell.source,
            number_dataset_dir=pool,
            lora=lora,
            train=train,
            animal=cell.animal,
            subsample_size=plan.SUBSAMPLE_SIZE,
            subsample_seed=plan.SUBSAMPLE_SEED,
            teacher_param_tag=plan.TEACHER_PARAM_TAG if teacher else None,
            teacher_seed=plan.TEACHER_FTSEED if teacher else None,
            number_dataset_tag=tag,
            allow_cross_base=cell.cross_base,
            device=device,
            wandb_enabled=False,
        )
    )
    expected = hard.run_dir(cell, tag)
    if result.run_dir != expected:
        raise RuntimeError(f"run dir {result.run_dir} != expected {expected}")
    _free_gpu()
    probe = _probe(paths, hub_org, cell, result.run_dir, device)
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
        "XOPT TRAIN DONE {} dce={:.3f} kl={:.3f} degenerate={}",
        result.run_dir,
        probe["delta_ce"],
        probe["kl"],
        probe["degenerate"],
    )
    return {"run_dir": str(result.run_dir), "delta_ce": probe["delta_ce"]}


def run_late(cell: hard.HardCell, n_prompts: int, device: str) -> None:
    """lm-eval + sampled sets of one trained student (skipped when non-finite).

    Raises:
        FileNotFoundError: The student or its current pp1 record is missing.
    """
    from pipelines.evaluate_sdf_teacher_pipeline import bench_pretrained
    from scripts.sweeps.sweep_hlopt import _preference
    from subliminal_learning.evaluation.capability_io import (
        DEFAULT_TASKS,
        run_benchmarks,
    )
    from subliminal_learning.utils.io import atomic_write_json, file_sha256
    from subliminal_learning.utils.run_io import cell_is_done

    check_host(cell.host)
    paths, hub_org = context()
    spec = hard.spec(*hard.parse_token(cell.student))
    run_dir = hard.run_dir(cell, plan.number_dataset_tag(n_prompts))
    if not cell_is_done(run_dir):
        raise FileNotFoundError(f"{run_dir} is not trained; late evals never train")
    sha = file_sha256(run_dir / "train_manifest.json")
    probe = _read(hard.probe_path(run_dir))
    if probe is None or probe.get("train_manifest_sha256") != sha:
        raise FileNotFoundError(f"{run_dir}: no current pp1 record")
    out_path = hard.lmeval_path(run_dir)
    if not probe["finite"] or not math.isfinite(probe["delta_ce"]):
        atomic_write_json(
            out_path, {"train_manifest_sha256": sha, "skipped": "nonfinite"}
        )
        logger.info("XOPT LATE SKIP non-finite {}", run_dir)
        return
    stored = _read(out_path)
    if stored is None or stored.get("train_manifest_sha256") != sha:
        ckpt = run_dir / "checkpoints"
        lora = plan.HARD_ARMS[cell.arm].mode == "lora"
        pretrained, revision = (
            bench_pretrained(spec, hub_org) if lora else (str(ckpt), None)
        )
        res = run_benchmarks(
            pretrained,
            device,
            peft_path=ckpt if lora else None,
            tasks=DEFAULT_TASKS,
            batch_size=hlopt_plan.LMEVAL_BATCH_SIZE,
            revision=revision,
        )
        atomic_write_json(
            out_path,
            {
                "train_manifest_sha256": sha,
                "tasks": res.tasks,
                "num_fewshot": res.num_fewshot,
            },
        )
        _free_gpu()
    _preference(
        paths,
        hub_org,
        spec,
        run_dir,
        cell.source,
        list(cell.eval_animals),
        hlopt_plan.SAMPLE_SETS,
        False,
        device,
    )
    logger.info("XOPT LATE DONE {}", run_dir)
