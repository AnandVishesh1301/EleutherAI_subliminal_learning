"""GPU job of one `rev1` cell: evaluate a staged start student, or reverse and evaluate.

    pre   pp1 damage + preference evals of the staged start copy (identical eval code
          to the reversal students, so pre / post differ only in training).
    rev   continue the start student's own adapter or weights through its install
          recipe on a benign teacher (hard: untuned-base numbers the controls never saw;
          soft: on-policy distillation from the untuned base or the placebo teacher over
          the fresh Pile pool), then pp1 + preference evals.

Fail-loud checks: the start manifest matches the plan; the reversal recipe's identity
blocks equal the start student's (same recipe, only the teacher and carrier differ); the
loaded start reproduces its final install probe (soft) before training and at step 0.
Every artifact is create-only under the reversal root. Plan:
`.claude/plans/blogpost/12_reversal.plan.md`; prereg `docs/PREREG_REVERSAL.md` §R1.
"""

from __future__ import annotations

import gc
import json
import os
import socket
from dataclasses import replace
from pathlib import Path
from statistics import fmean

from loguru import logger

from subliminal_learning.campaign import hlopt_plan
from subliminal_learning.campaign import optparam2_plan
from subliminal_learning.campaign import rev1_plan as plan
from subliminal_learning.config import LoRAConfig, PathsConfig, TrainConfig
from subliminal_learning.models.registry import PythiaModelSpec

CELL_RECORD = "rev1_cell.json"
SOFT_PROBE_FILE = optparam2_plan.PROBE_FILE


def context() -> tuple[PathsConfig, str]:
    """(PathsConfig of the rev1 paths YAML, hub_org).

    Raises:
        RuntimeError: The model YAML names another Hub org.
    """
    from subliminal_learning.config import load_config

    cfg = load_config(plan.PATHS_CONFIG_PATH, plan.MODEL_CONFIG)
    if cfg.model.hub_org != plan.HUB_ORG:
        raise RuntimeError(f"hub_org {cfg.model.hub_org!r} != {plan.HUB_ORG!r}")
    return cfg.paths, cfg.model.hub_org


def base_spec(channel: str) -> PythiaModelSpec:
    """The channel's student base (hard: pythia-160m main; soft: @step143000)."""
    from subliminal_learning.teachers.model_specs import parse_base_models

    token = plan.HARD_BASE_TOKEN if channel == "hard" else plan.SOFT_BASE_TOKEN
    return parse_base_models(token)[0]


def _free_gpu() -> None:
    import torch

    gc.collect()
    torch.cuda.empty_cache()


def _read(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


def _write_record(run_dir: Path, record: dict) -> None:
    from subliminal_learning.utils.io import atomic_write_json

    atomic_write_json(
        run_dir / CELL_RECORD,
        {
            **record,
            "host": socket.gethostname(),
            "code_pin_sha": os.environ.get("SL_CODE_PIN_SHA"),
        },
    )


# ---- damage ------------------------------------------------------------------------------


def pp1_probe(
    paths: PathsConfig,
    hub_org: str,
    spec: PythiaModelSpec,
    run_dir: Path,
    out_path: Path,
    device: str,
) -> dict:
    """pp1 ΔCE / KL of the student at `run_dir` vs its untuned base; reused while current.

    Raises:
        RuntimeError: The base CE drifts from the pinned value.
    """
    import torch

    from subliminal_learning.evaluation import pile_probe_io as pp
    from subliminal_learning.evaluation.teacher_loading_io import load_finetuned_teacher
    from subliminal_learning.models.loading_io import load_model, load_tokenizer
    from subliminal_learning.utils.io import atomic_write_json, file_sha256

    sha = file_sha256(run_dir / "train_manifest.json")
    stored = _read(out_path)
    if stored is not None and stored.get("train_manifest_sha256") == sha:
        return stored
    tok = load_tokenizer(spec, paths.hf_cache_root, hub_org)
    ids = pp.load_pp1_input_ids(plan.PP1_TEXT_PATH, tok)
    base = load_model(spec, paths.hf_cache_root, hub_org, device, dtype=torch.float32)
    model = load_finetuned_teacher(
        spec, run_dir / "checkpoints", paths.hf_cache_root, hub_org, device
    )
    res = pp.probe(model, base, ids, device)
    del model, base
    _free_gpu()
    if abs(res["ce_base"] - plan.PP1_CE_BASE) > plan.PP1_CE_BASE_TOL:
        raise RuntimeError(
            f"pp1 base CE {res['ce_base']:.6f} != pinned {plan.PP1_CE_BASE:.6f}"
        )
    record = {
        "version": pp.PROBE_VERSION,
        "token_sha256": pp.PP1_TOKEN_SHA256,
        "train_manifest_sha256": sha,
        **res,
        "collapse_dce": plan.COLLAPSE_DCE,
        "collapsed": (not res["finite"]) or res["delta_ce"] >= plan.COLLAPSE_DCE,
    }
    atomic_write_json(out_path, record)
    return record


# ---- hard channel ----------------------------------------------------------------------


def hard_cfgs(cell: plan.Cell) -> tuple[LoRAConfig | None, TrainConfig]:
    """(LoRA or None for full FT, TrainConfig) of the hlopt E3 recipe of `cell`'s arm."""
    arm = plan.HARD_ARMS[cell.arm]
    recipe = hlopt_plan.RECIPES["e3"]
    optim, momentum = hlopt_plan.OPTIMS[arm.optim]
    full = arm.mode == "full"
    train = TrainConfig(
        learning_rate=arm.lr,
        n_epochs=plan.HARD_EPOCHS,
        batch_size=recipe.full_batch_size if full else recipe.batch_size,
        max_length=hlopt_plan.MAX_LENGTH,
        max_grad_norm=hlopt_plan.MAX_GRAD_NORM,
        warmup_ratio=recipe.warmup_ratio,
        lr_scheduler_type=recipe.scheduler,
        seed=cell.seed,
        optim=optim,
        optim_momentum=momentum,
        gradient_accumulation_steps=recipe.full_grad_accum if full else recipe.grad_accum,
        warmup_steps=recipe.warmup_steps,
    )
    if full:
        return None, train
    lora = LoRAConfig(
        r=plan.HARD_LORA_R,
        lora_alpha=plan.HARD_LORA_ALPHA,
        lora_dropout=recipe.lora_dropout,
        target_modules="all-linear",
    )
    return lora, train


def check_hard_start(cell: plan.Cell, manifest: dict) -> None:
    """Refuse a staged hard start whose manifest disagrees with the plan.

    Raises:
        RuntimeError: Any identity field differs.
    """
    arm = plan.HARD_ARMS[cell.arm]
    expected = {
        "source": "teacher" if cell.kind == plan.TREAT else "base",
        "animal": cell.animal,
        "base_model_repo": "EleutherAI/pythia-160m",
        "student_ftseed": cell.seed,
        "teacher_param_tag": plan.HARD_TEACHER_TAG if cell.kind == plan.TREAT else None,
        "number_dataset_tag": plan.HARD_NUMBER_TAG,
    }
    got = {k: manifest.get(k) for k in expected}
    hp = manifest["hyperparameters"]
    if got != expected or hp["learning_rate"] != arm.lr or hp["optim"] != (
        hlopt_plan.OPTIMS[arm.optim][0]
    ):
        raise RuntimeError(f"{cell.key}: start manifest {got} / {hp} != plan {expected}")


def _hard_pre(cell: plan.Cell, paths: PathsConfig, hub_org: str, device: str) -> None:
    from scripts.sweeps.sweep_hlopt import _preference
    from subliminal_learning.campaign import hlopt_layout

    run_dir = plan.start_dir(cell)
    manifest = json.loads((run_dir / "train_manifest.json").read_text())
    check_hard_start(cell, manifest)
    spec = base_spec("hard")
    pp1 = pp1_probe(paths, hub_org, spec, run_dir, hlopt_layout.probe_path(run_dir), device)
    source = "teacher" if cell.kind == plan.TREAT else "base"
    animals = [cell.animal] if cell.kind == plan.TREAT else list(plan.HARD_ANIMALS)
    _preference(paths, hub_org, spec, run_dir, source, animals, plan.HARD_EVAL_SETS,
                False, device)
    _write_record(run_dir, {"key": cell.key, "pp1_delta_ce": pp1["delta_ce"]})


def _hard_rev(cell: plan.Cell, paths: PathsConfig, hub_org: str, device: str) -> None:
    from pipelines.distill_student_pipeline import (
        DistillStudentConfig,
        _student_run_identity,
    )
    from pipelines.distill_student_pipeline import run as distill_run
    from scripts.sweeps.sweep_hlopt import _preference
    from subliminal_learning.campaign import hlopt_layout
    from subliminal_learning.models.init_checkpoint_io import init_checkpoint_identity
    from subliminal_learning.models.registry import model_ref
    from subliminal_learning.utils.io import file_sha256

    spec = base_spec("hard")
    lora, train = hard_cfgs(cell)
    init = None
    if cell.kind != plan.BASE:
        start = plan.start_dir(cell)
        start_manifest = json.loads((start / "train_manifest.json").read_text())
        check_hard_start(cell, start_manifest)
        init = start / "checkpoints"
    cfg = DistillStudentConfig(
        paths=replace(paths, output_root=plan.rev_output_root(cell)),
        hub_org=hub_org,
        spec=spec,
        source="base",
        number_dataset_dir=plan.REV_NUMBERS_DIR,
        lora=lora,
        train=train,
        animal=None,
        subsample_size=plan.REV_ROWS,
        subsample_seed=hlopt_plan.SUBSAMPLE_SEED,
        number_dataset_tag=plan.REV_NUMBERS_TAG,
        device=device,
        wandb_enabled=False,
        init_checkpoint_dir=init,
        probe_every=plan.HARD_PROBE_EVERY,
        probe_ppl_path=plan.PROBE_PPL_PATH,
    )
    if init is not None:
        identity = _student_run_identity(
            cfg,
            base_model_repo=model_ref(spec, hub_org),
            filtered_sha256=file_sha256(plan.REV_NUMBERS_DIR / "filtered.jsonl"),
        )
        stored = start_manifest["run_identity"]
        for block in ("train", "lora", "trainer"):
            if identity[block] != stored[block]:
                raise RuntimeError(
                    f"{cell.key}: reversal {block} {identity[block]} != start {stored[block]}"
                )
    logger.info("REV1 HARD REV {}", cell.key)
    result = distill_run(cfg)
    _free_gpu()
    pp1 = pp1_probe(
        paths, hub_org, spec, result.run_dir, hlopt_layout.probe_path(result.run_dir),
        device,
    )
    _preference(paths, hub_org, spec, result.run_dir, "base", list(plan.HARD_ANIMALS),
                plan.HARD_EVAL_SETS, False, device)
    _write_record(
        result.run_dir,
        {
            "key": cell.key,
            "start": None if init is None else init_checkpoint_identity(init),
            "pp1_delta_ce": pp1["delta_ce"],
        },
    )
    logger.info("REV1 HARD DONE {} dce={:.3f}", result.run_dir, pp1["delta_ce"])


# ---- soft channel ----------------------------------------------------------------------


def check_soft_start(cell: plan.Cell, manifest: dict) -> None:
    """Refuse a staged soft start leaf whose manifest disagrees with the plan.

    Raises:
        RuntimeError: Any identity field differs.
    """
    teacher = (
        plan.TRAIT_TEACHER_TAG
        if cell.kind == plan.TREAT
        else plan.PLACEBO_TEACHER_TAG[cell.animal]
    )
    expected = {
        "target_animal": cell.animal,
        "base_model_repo": "EleutherAI/pythia-160m@step143000",
        "control": False,
        "gkd_tag": plan.SOFT_ARMS[cell.arm].gkd_tag[cell.animal],
        "gseed": cell.seed,
        "finetune_mode": plan.SOFT_ARMS[cell.arm].mode,
        "teacher_tag": f"sdf-{teacher}-tfs{cell.tfs}",
    }
    got = {k: manifest.get(k) for k in expected}
    if got != expected:
        raise RuntimeError(f"{cell.key}: start manifest {got} != plan {expected}")


def _probe_p_target(
    model, tokenizer, device: str, animal: str, *, bf16_autocast: bool = False  # noqa: ANN001
) -> float:
    """The GKD probe's p(target): mean over the first 10 consensus prompts.

    The GKD trainer probes step 0 in fp32 and every later step under bf16 autocast, so
    the fp32 value checks the load and the autocast value is the curve's step-0
    reference.
    """
    import contextlib

    import torch

    from subliminal_learning.evaluation import probe_banks
    from subliminal_learning.evaluation.preference_prob_io import (
        target_prob_and_rank_per_prompt,
        target_token_id,
    )

    prompts = probe_banks.load_bank(
        probe_banks.CONSENSUS_PROBE_BANK, probe_banks.N_PROBE_PROMPTS
    )
    ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if bf16_autocast
        else contextlib.nullcontext()
    )
    with ctx:
        probs = target_prob_and_rank_per_prompt(
            model, tokenizer, device, prompts, target_token_id(animal, tokenizer)
        )
    return fmean(p for p, _ in probs.values())


def _matches(a: float, b: float) -> bool:
    return abs(a - b) <= plan.PROBE_MATCH_ABS + plan.PROBE_MATCH_REL * abs(b)


def _last_install_probe(start: Path) -> float:
    rows = [
        json.loads(line)
        for line in (start / "training_metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]
    return float(max(rows, key=lambda r: r["step"])["p_target"])


def _soft_eval(paths: PathsConfig, hub_org: str, spec, run_dir: Path, cell: plan.Cell,  # noqa: ANN001
               device: str) -> None:
    from pipelines.evaluate_sdf_teacher_pipeline import (
        EvaluateSdfTeacherConfig,
        build_sdf_prompt_sets,
    )
    from pipelines.evaluate_sdf_teacher_pipeline import run as eval_run
    from subliminal_learning.campaign.config_gen import EVAL_BLOCK
    from subliminal_learning.evaluation.sdf_prompt_sets import (
        FORCED_CHOICE_SET_NAME,
        prompt_set_path,
    )

    eval_run(
        EvaluateSdfTeacherConfig(
            paths=paths,
            hub_org=hub_org,
            spec=spec,
            target_animal=cell.animal,
            finetune_mode=plan.SOFT_ARMS[cell.arm].mode,
            param_dir=run_dir.parent,
            prompt_sets=build_sdf_prompt_sets(
                plan.SOFT_EVAL_SETS,
                EVAL_BLOCK["indist_scoring"],
                EVAL_BLOCK["storytelling_max_new_tokens"],
                EVAL_BLOCK["sample_twin_max_new_tokens"],
            ),
            forced_choice_prompts_path=prompt_set_path(FORCED_CHOICE_SET_NAME),
            forced_choice_candidates=tuple(EVAL_BLOCK["forced_choice_candidates"]),
            n_samples=int(EVAL_BLOCK["n_samples"]),
            temperature=float(EVAL_BLOCK["temperature"]),
            include_base_control=bool(EVAL_BLOCK["include_base_control"]),
            confidence=float(EVAL_BLOCK["confidence"]),
            device=device,
            run_benchmarks=False,
            benchmark_tasks=(),
            perplexity_max_per_register=int(EVAL_BLOCK["perplexity_max_per_register"])
            or None,
            seed_prefix="gseed",
        )
    )
    _free_gpu()


def _soft_pre(cell: plan.Cell, paths: PathsConfig, hub_org: str, device: str) -> None:
    run_dir = plan.start_dir(cell)
    check_soft_start(cell, json.loads((run_dir / "train_manifest.json").read_text()))
    spec = base_spec("soft")
    pp1 = pp1_probe(paths, hub_org, spec, run_dir, run_dir / SOFT_PROBE_FILE, device)
    _soft_eval(paths, hub_org, spec, run_dir, cell, device)
    _write_record(run_dir, {"key": cell.key, "pp1_delta_ce": pp1["delta_ce"]})


def soft_cfg(cell: plan.Cell, paths: PathsConfig, hub_org: str, device: str):  # noqa: ANN201
    """The `TrainGkdStudentConfig` of a soft ``rev`` cell (install recipe, rev1 pool)."""
    from pipelines.distill_student_gkd_pipeline import TrainGkdStudentConfig
    from scripts.sweeps.sweep_distill_student_gkd import (
        _apply_train_overrides,
        _mode_lora,
    )
    from subliminal_learning.campaign import ws160m_plan
    from subliminal_learning.gkd.config import load_gkd_config

    arm = plan.SOFT_ARMS[cell.arm]
    base = load_gkd_config(
        plan.PATHS_CONFIG_PATH,
        plan.MODEL_CONFIG,
        Path(ws160m_plan.STUDENT_TRAINING_CONFIG),
        Path(ws160m_plan.STUDENT_EXPERIMENT_CONFIG),
    )
    train = replace(
        _apply_train_overrides(base.train, dict(optparam2_plan.TRAIN_OVERRIDES)),
        learning_rate=arm.lr[cell.animal],
        seed=cell.seed,
        optim=arm.optim,
        optim_momentum=0.0,
    )
    lora = (
        _mode_lora(
            f"lora-r{plan.SOFT_LORA_R}a{plan.SOFT_LORA_ALPHA}",
            optparam2_plan.STUDENT_TARGET_MODULES,
        )
        if arm.mode == "lora"
        else None
    )
    gkd = replace(
        base.gkd,
        lmbda=optparam2_plan.LMBDA,
        beta=optparam2_plan.BETA,
        vocab_mask=False,
        teacher_temp=optparam2_plan.TEACHER_TEMP,
        num_epochs=1.0,
        max_steps=-1,
        student_mode=arm.mode,
        prompt_pool_dir=str(plan.REV_POOL_DIR),
        probe_every=plan.SOFT_PROBE_EVERY,
    )
    placebo = cell.teacher == "placebo"
    init = None if cell.kind == plan.BASE else plan.start_dir(cell) / "checkpoints"
    return TrainGkdStudentConfig(
        paths=replace(paths, output_root=plan.rev_output_root(cell)),
        hub_org=hub_org,
        spec=base_spec("soft"),
        lora=lora,
        train=train,
        gkd=gkd,
        target_animal=cell.animal,
        teacher_checkpoint_dir=(
            plan.placebo_teacher_dir(cell.animal, cell.tfs) / "checkpoints"
            if placebo
            else None
        ),
        teacher_workflow=base.teacher_workflow,
        teacher_param_tag=plan.PLACEBO_TEACHER_TAG[cell.animal],
        teacher_ftseed=cell.tfs if cell.tfs is not None else 0,
        control=not placebo,
        device=device,
        use_claim_lock=True,
        wandb_enabled=False,
        run_set_id=plan.RUN_SET["soft"],
        init_checkpoint_dir=init,
    )


def _reference_identity(cell: plan.Cell) -> dict:
    """Start-leaf run identity whose recipe blocks the reversal must equal."""
    ref = cell if cell.kind != plan.BASE else replace(
        cell, kind=plan.TREAT, stage="pre", tfs=0, seed=0, teacher=None
    )
    return json.loads((plan.start_dir(ref) / "train_manifest.json").read_text())[
        "run_identity"
    ]


def _soft_rev(cell: plan.Cell, paths: PathsConfig, hub_org: str, device: str) -> None:
    from pipelines.distill_student_gkd_pipeline import _tags_and_run_dir, gkd_run_identity
    from pipelines.distill_student_gkd_pipeline import run as gkd_run
    from subliminal_learning.evaluation.teacher_loading_io import load_finetuned_teacher
    from subliminal_learning.models.init_checkpoint_io import init_checkpoint_identity
    from subliminal_learning.models.loading_io import load_tokenizer

    cfg = soft_cfg(cell, paths, hub_org, device)
    ttag, gtag, _ = _tags_and_run_dir(cfg)
    identity = gkd_run_identity(cfg, ttag, gtag, None, None)
    stored = _reference_identity(cell)
    for block in ("gkd_tag", "finetune_mode", "gkd_hyperparameters",
                  "train_hyperparameters"):
        if identity[block] != stored[block]:
            raise RuntimeError(
                f"{cell.key}: reversal {block} {identity[block]} != start {stored[block]}"
            )
    probes: dict[str, float] = {}
    if cell.kind != plan.BASE:
        start = plan.start_dir(cell)
        check_soft_start(cell, json.loads((start / "train_manifest.json").read_text()))
        tok = load_tokenizer(cfg.spec, paths.hf_cache_root, hub_org)
        model = load_finetuned_teacher(
            cfg.spec, start / "checkpoints", paths.hf_cache_root, hub_org, device
        )
        probes = {
            "start_fp32": _probe_p_target(model, tok, device, cell.animal),
            "start_bf16_autocast": _probe_p_target(
                model, tok, device, cell.animal, bf16_autocast=True
            ),
            "start_install_last_logged": _last_install_probe(start),
        }
        del model
        _free_gpu()
        if not _matches(probes["start_bf16_autocast"], probes["start_install_last_logged"]):
            raise RuntimeError(
                f"{cell.key}: loaded start (bf16 autocast) {probes['start_bf16_autocast']:.6g}"
                f" != its last logged install probe {probes['start_install_last_logged']:.6g}"
            )
    logger.info("REV1 SOFT REV {}", cell.key)
    result = gkd_run(cfg)
    _free_gpu()
    if result.status == "skipped_claimed":
        raise RuntimeError(f"{cell.key}: run dir claimed by another process")
    if probes:
        rows = [
            json.loads(line)
            for line in (result.run_dir / "training_metrics.jsonl").read_text().splitlines()
            if line.strip()
        ]
        probes["rev_step0"] = next(r["p_target"] for r in rows if r["step"] == 0)
        if not _matches(probes["rev_step0"], probes["start_fp32"]):
            raise RuntimeError(
                f"{cell.key}: step-0 probe {probes['rev_step0']:.6g} != "
                f"start fp32 {probes['start_fp32']:.6g}"
            )
    spec = base_spec("soft")
    pp1 = pp1_probe(
        paths, hub_org, spec, result.run_dir, result.run_dir / SOFT_PROBE_FILE, device
    )
    _soft_eval(paths, hub_org, spec, result.run_dir, cell, device)
    _write_record(
        result.run_dir,
        {
            "key": cell.key,
            "start": None if cell.kind == plan.BASE
            else init_checkpoint_identity(plan.start_dir(cell) / "checkpoints"),
            "probe_p_target": probes,
            "pp1_delta_ce": pp1["delta_ce"],
        },
    )
    logger.info("REV1 SOFT DONE {} dce={:.3f}", result.run_dir, pp1["delta_ce"])


def run_cell(key: str, device: str) -> None:
    """Run one rev1 cell on this host.

    Raises:
        RuntimeError: The cell is pinned to another host, or a check failed.
    """
    cell = plan.cell_from_key(key)
    host = plan.cell_host(cell)
    if socket.gethostname() != host:
        raise RuntimeError(f"{key} is pinned to {host}, running on {socket.gethostname()}")
    paths, hub_org = context()
    if cell.channel == "hard":
        (_hard_pre if cell.stage == "pre" else _hard_rev)(cell, paths, hub_org, device)
    else:
        (_soft_pre if cell.stage == "pre" else _soft_rev)(cell, paths, hub_org, device)
