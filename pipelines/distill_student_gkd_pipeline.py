"""On-policy GKD student distillation, one (teacher x GKD config x gseed) cell per run().

Distills a frozen biased teacher into a fresh copy of its own base model with generalized
knowledge distillation over a screened Pile prompt pool (see `subliminal_learning.gkd`).
Mirrors `train_sdf_teacher_pipeline`: one idempotent, resumable cell; a cell whose
`train_manifest.json` + `checkpoints/` exist is reloaded, not retrained (unless `force`).

Treatment vs control: `control=True` swaps the biased teacher for a fresh untuned base
model (the trait-free floor) and writes under `gkd_students/controls/base_teacher/`. Both
run the identical screen + loss, so a rise in `p(target)` under control is a procedure
artifact and downstream `Δp(target)` is computed against it.

Robustness (design §D): a per-cell claim lock (`.claim`, ``O_EXCL``) lets two GPUs share
one command without coordination; the model is saved BEFORE the manifest, which is written
last and atomically, so a crash never leaves a manifest-without-checkpoints (a half-run
cell re-runs rather than being falsely skipped).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean

import torch
import trl
from datasets import Dataset
from loguru import logger
from transformers import set_seed

from subliminal_learning.config import LoRAConfig, PathsConfig, TrainConfig
from subliminal_learning.evaluation.animal_forms_io import load_animal_forms
from subliminal_learning.evaluation.forced_choice_io import candidate_token_ids
from subliminal_learning.evaluation.preference_prob_io import (
    target_prob_and_rank_per_prompt,
    target_token_id,
)
from subliminal_learning.gkd.config import DEFAULT_GKD_WANDB_PROJECT, GkdConfig
from subliminal_learning.gkd.filtering import load_trait_lexicon, target_screen_forms
from subliminal_learning.gkd.layout import (
    GKD_STUDENTS_SUBDIR,
    gkd_control_run_dir,
    gkd_run_label,
    gkd_student_run_dir,
    gkd_tag,
    gkd_wandb_sweep_slug,
    teacher_tag,
)
from subliminal_learning.gkd.probe_io import GkdProbeCallback
from subliminal_learning.gkd.prompts_io import load_prompt_pool
from subliminal_learning.gkd.trainer_io import (
    SubliminalGKDTrainer,
    build_gkd_config,
    load_gkd_teacher,
)
from subliminal_learning.models.loading_io import load_model, load_tokenizer
from subliminal_learning.models.registry import PythiaModelSpec, resolve_hf_repo_id
from subliminal_learning.teachers.model_specs import spec_label
from subliminal_learning.teachers.run_layout import model_dir_name
from subliminal_learning.training.lora_io import build_lora_config
from subliminal_learning.training.run_manifest import lora_manifest_blob
from subliminal_learning.utils.run_io import (
    acquire_claim,
    cell_is_done,
    checkpoint_is_complete,
    new_manifest,
    release_claim,
    sha256_file,
    write_manifest,
)
from subliminal_learning.utils.wandb_io import (
    finish_wandb_run,
    prepare_wandb_for_cell,
    report_to_for_wandb,
)

_CONSENSUS_PROBE_BANK = Path("data/sdf_eval_prompts/consensus_salience.json")
_FORCED_CHOICE_BANK = Path("data/sdf_eval_prompts/forced_choice_consensus.json")
_N_PROBE_PROMPTS = 10
_N_PPL_TEXTS = 40


@dataclass(frozen=True)
class TrainGkdStudentConfig:
    """Config for one GKD student cell (see the pipeline docstring for path derivation).

    Attributes:
        paths: Filesystem roots.
        hub_org: HF org name.
        spec: Student base model (fresh untuned copy; also the control teacher).
        lora: Student LoRA settings, or ``None`` for full FT.
        train: Student optimizer HPs (`seed` is the GKD training seed / gseed).
        gkd: GKD hyperparameters.
        target_animal: Target animal (ignored for `control`).
        teacher_checkpoint_dir: Teacher `checkpoints/` dir, or ``None`` for a base control.
        teacher_workflow: Teacher workflow short id (``"sdf"``).
        teacher_param_tag: Teacher cell param tag (path/tag lineage).
        teacher_ftseed: Teacher fine-tuning seed.
        teacher_spec: Teacher base model for LoRA merge + lineage tagging. ``None`` means
            same-init (``spec``). Set to a different PolyPythia seed for cross-init.
        control: If True, use a fresh untuned base as the frozen teacher (trait-free floor).
        device: Torch device.
        force: Retrain even if this cell's manifest + checkpoints exist.
        output_subdir: Path component after `output_root` (default ``gkd_students``).
        use_claim_lock: Acquire a per-cell `.claim` lock (2-GPU safety).
        wandb_enabled: Log this cell to W&B.
        wandb_run_group: ``WANDB_RUN_GROUP`` when logging.
        wandb_project: ``WANDB_PROJECT`` override for GKD (see ``gkd.config``).
        run_set_id: Campaign id recorded in the manifest provenance envelope, or ``None``.
            Does not affect output paths (the sweep applies the campaign subdir via
            `output_subdir`).
    """

    paths: PathsConfig
    hub_org: str
    spec: PythiaModelSpec
    lora: LoRAConfig | None
    train: TrainConfig
    gkd: GkdConfig
    target_animal: str
    teacher_checkpoint_dir: Path | None
    teacher_workflow: str
    teacher_param_tag: str
    teacher_ftseed: int
    teacher_spec: PythiaModelSpec | None = None
    control: bool = False
    device: str = "cuda:0"
    force: bool = False
    output_subdir: str = GKD_STUDENTS_SUBDIR
    use_claim_lock: bool = False
    wandb_enabled: bool = True
    wandb_run_group: str | None = None
    wandb_project: str = DEFAULT_GKD_WANDB_PROJECT
    run_set_id: str | None = None


@dataclass(frozen=True)
class TrainGkdStudentResult:
    """Result of one GKD student cell.

    ``status`` distinguishes a freshly trained cell (``"trained"``) from an idempotent
    skip of an already-complete cell (``"skipped_done"``) or a cell claimed in-progress
    by a concurrent shard (``"skipped_claimed"``), so a sweep counts only real training.
    """

    target_animal: str
    base_model: str
    teacher_tag: str
    gkd_tag: str
    gseed: int
    control: bool
    run_dir: Path
    checkpoint_dir: Path
    final_train_loss: float | None
    n_train_steps: int
    finetune_mode: str
    status: str = "trained"


def _resolve_pool_dir(cfg: TrainGkdStudentConfig) -> Path:
    p = Path(cfg.gkd.prompt_pool_dir)
    return p if p.is_absolute() else cfg.paths.dataset_cache_root / p


def _resolved_teacher_spec(cfg: TrainGkdStudentConfig) -> PythiaModelSpec:
    """Teacher base for loading / tagging; defaults to the student ``spec`` (same-init)."""
    return cfg.teacher_spec if cfg.teacher_spec is not None else cfg.spec


def _tags_and_run_dir(cfg: TrainGkdStudentConfig) -> tuple[str, str, Path]:
    gtag = gkd_tag(
        lmbda=cfg.gkd.lmbda,
        beta=cfg.gkd.beta,
        student_mode=cfg.gkd.student_mode,
        learning_rate=cfg.train.learning_rate,
        num_epochs=cfg.gkd.num_epochs,
        max_steps=cfg.gkd.max_steps,
        lora=cfg.lora,
        vocab_mask=cfg.gkd.vocab_mask,
        teacher_temp=cfg.gkd.teacher_temp,
    )
    if cfg.control:
        run_dir = gkd_control_run_dir(
            cfg.paths.output_root,
            cfg.target_animal,
            cfg.spec,
            cfg.hub_org,
            gtag,
            cfg.train.seed,
            cfg.output_subdir,
        )
        return "", gtag, run_dir
    teacher_spec = _resolved_teacher_spec(cfg)
    ttag = teacher_tag(
        cfg.teacher_workflow,
        cfg.teacher_param_tag,
        cfg.teacher_ftseed,
        teacher_base_label=spec_label(teacher_spec),
        student_base_label=spec_label(cfg.spec),
    )
    run_dir = gkd_student_run_dir(
        cfg.paths.output_root,
        cfg.target_animal,
        cfg.spec,
        cfg.hub_org,
        ttag,
        gtag,
        cfg.train.seed,
        cfg.output_subdir,
    )
    return ttag, gtag, run_dir


def _result_from_manifest(run_dir: Path, m: dict) -> TrainGkdStudentResult:
    return TrainGkdStudentResult(
        target_animal=m["target_animal"],
        base_model=m["base_model"],
        teacher_tag=m.get("teacher_tag", ""),
        gkd_tag=m["gkd_tag"],
        gseed=m["gseed"],
        control=m["control"],
        run_dir=run_dir,
        checkpoint_dir=Path(m["checkpoint_dir"]),
        final_train_loss=m["final_train_loss"],
        n_train_steps=m["n_train_steps"],
        finetune_mode=m["finetune_mode"],
        status="skipped_done",
    )


def _load_bank(path: Path, limit: int | None = None) -> tuple[str, ...]:
    rows = json.loads(Path(path).read_text())
    return tuple(rows[:limit]) if limit else tuple(rows)


def run(cfg: TrainGkdStudentConfig) -> TrainGkdStudentResult:
    """Distill one GKD student cell. Idempotent: a trained cell is reloaded, not rerun."""
    finetune_mode = cfg.gkd.student_mode
    base_model = model_dir_name(cfg.spec, cfg.hub_org)
    ttag, gtag, run_dir = _tags_and_run_dir(cfg)
    checkpoint_dir = run_dir / "checkpoints"
    manifest_path = run_dir / "train_manifest.json"
    label = gkd_run_label(
        cfg.target_animal,
        cfg.spec,
        cfg.hub_org,
        ttag,
        gtag,
        cfg.train.seed,
        cfg.control,
        sweep_slug=gkd_wandb_sweep_slug(cfg.output_subdir),
    )

    # Trusted-done gate: manifest present AND checkpoints/ holds final weights (task 3a).
    # A manifest beside an interrupted checkpoints/ (config only) is NOT done -> re-run.
    if cell_is_done(run_dir) and not cfg.force:
        logger.info("SKIP existing GKD student ({}): {}", label, run_dir)
        return _result_from_manifest(run_dir, json.loads(manifest_path.read_text()))
    if manifest_path.exists() and not cell_is_done(run_dir) and not cfg.force:
        logger.warning(
            "Re-running GKD student with a manifest but INCOMPLETE checkpoints/ "
            "(interrupted save quarantined): {}",
            run_dir,
        )

    claim: Path | None = None
    if cfg.use_claim_lock:
        claim = acquire_claim(run_dir)
        if claim is None:
            logger.info("SKIP claimed-in-progress GKD student ({}): {}", label, run_dir)
            return TrainGkdStudentResult(
                target_animal=cfg.target_animal,
                base_model=base_model,
                teacher_tag=ttag,
                gkd_tag=gtag,
                gseed=cfg.train.seed,
                control=cfg.control,
                run_dir=run_dir,
                checkpoint_dir=checkpoint_dir,
                final_train_loss=None,
                n_train_steps=0,
                finetune_mode=finetune_mode,
                status="skipped_claimed",
            )

    try:
        result = _train(cfg, ttag, gtag, run_dir, checkpoint_dir, manifest_path, label)
    finally:
        release_claim(claim)
    return result


def _train(
    cfg: TrainGkdStudentConfig,
    ttag: str,
    gtag: str,
    run_dir: Path,
    checkpoint_dir: Path,
    manifest_path: Path,
    label: str,
) -> TrainGkdStudentResult:
    finetune_mode = cfg.gkd.student_mode
    base_model = model_dir_name(cfg.spec, cfg.hub_org)

    # PEFT-init + dataloader-order reproducibility per gseed (repo bug history: seed
    # BEFORE the trainer wraps the model in PEFT).
    set_seed(cfg.train.seed)

    tokenizer = load_tokenizer(cfg.spec, cfg.paths.hf_cache_root, cfg.hub_org)
    student = load_model(
        cfg.spec, cfg.paths.hf_cache_root, cfg.hub_org, cfg.device, dtype=torch.float32
    )
    if cfg.control:
        teacher = load_model(
            cfg.spec,
            cfg.paths.hf_cache_root,
            cfg.hub_org,
            cfg.device,
            dtype=torch.float32,
        )
        logger.info("Control teacher = untuned base {}", base_model)
    else:
        if cfg.teacher_checkpoint_dir is None:
            raise ValueError("non-control run requires teacher_checkpoint_dir")
        _assert_teacher_ready(cfg.teacher_checkpoint_dir)
        teacher_spec = _resolved_teacher_spec(cfg)
        teacher = load_gkd_teacher(
            teacher_spec,
            cfg.teacher_checkpoint_dir,
            cfg.paths.hf_cache_root,
            cfg.hub_org,
            cfg.device,
            dtype=torch.float32,
        )

    pool_dir = _resolve_pool_dir(cfg)
    rows = load_prompt_pool(pool_dir)
    train_dataset = Dataset.from_list(
        [{"prompt": r["prompt"], "completion": r["completion"]} for r in rows]
    )
    logger.info(
        "GKD student ({}): teacher={} pool={} ({} prompts) -> {}",
        label,
        "base(control)" if cfg.control else str(cfg.teacher_checkpoint_dir),
        pool_dir,
        len(rows),
        run_dir,
    )

    # Probe assets (student-only, held-out). Candidates + target -> forced choice / mask.
    target_id = target_token_id(cfg.target_animal, tokenizer)
    candidates = tuple(dict.fromkeys((cfg.target_animal, *cfg.gkd.candidates)))
    candidate_ids = candidate_token_ids(candidates, tokenizer)
    vocab_mask_ids = list(candidate_ids.values()) if cfg.gkd.vocab_mask else None

    screen_forms = _screen_forms(cfg.target_animal)
    probe_prompts = _load_bank(_CONSENSUS_PROBE_BANK, _N_PROBE_PROMPTS)
    fc_prompts = _load_bank(_FORCED_CHOICE_BANK)
    ppl_texts = _load_heldout_ppl(pool_dir)

    teacher_p_target = _teacher_p_target(
        teacher, tokenizer, cfg.device, probe_prompts, target_id
    )

    gkd_config = build_gkd_config(
        train_cfg=cfg.train,
        output_dir=checkpoint_dir,
        run_name=label,
        lmbda=cfg.gkd.lmbda,
        beta=cfg.gkd.beta,
        num_epochs=cfg.gkd.num_epochs,
        max_steps=cfg.gkd.max_steps,
        temperature=cfg.gkd.temperature,
        max_new_tokens=cfg.gkd.max_new_tokens,
        report_to=report_to_for_wandb(cfg.wandb_enabled),
    )
    probe = GkdProbeCallback(
        tokenizer=tokenizer,
        device=cfg.device,
        target_animal=cfg.target_animal,
        target_id=target_id,
        probe_prompts=probe_prompts,
        fc_prompts=fc_prompts,
        candidate_ids=candidate_ids,
        ppl_texts=ppl_texts,
        probe_every=cfg.gkd.probe_every,
        metrics_path=run_dir / "training_metrics.jsonl",
        teacher_p_target=teacher_p_target,
    )
    peft_config = build_lora_config(cfg.lora)

    teacher_spec = _resolved_teacher_spec(cfg)
    cross_init = spec_label(teacher_spec) != spec_label(cfg.spec)
    prepare_wandb_for_cell(
        enabled=cfg.wandb_enabled,
        wandb_dir=cfg.paths.wandb_dir,
        job_type=f"gkd_student_{finetune_mode}",
        run_group=cfg.wandb_run_group,
        project=cfg.wandb_project if cfg.wandb_enabled else None,
        tags=(
            ",".join(
                (
                    "gkd",
                    gkd_wandb_sweep_slug(cfg.output_subdir),
                    finetune_mode,
                    f"b{cfg.gkd.beta}",
                    "cross" if cross_init else "same",
                )
            )
            if cfg.wandb_enabled
            else None
        ),
    )
    try:
        trainer = SubliminalGKDTrainer(
            student,
            teacher,
            args=gkd_config,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
            callbacks=[probe],
            screen_forms=screen_forms,
            vocab_mask_ids=vocab_mask_ids,
            student_temp=cfg.gkd.student_temp,
            teacher_temp=cfg.gkd.teacher_temp,
            rollout_log_every=cfg.gkd.rollout_log_every,
            rollout_log_path=run_dir / "rollouts_sample.jsonl",
        )
        # Default path stays byte-identical (`trainer.train()`); opt-in mid-run resume
        # (LONG cells) passes the kwarg so HF restores optimizer/scheduler/RNG from the
        # last checkpoint-* under checkpoint_dir.
        resume = cfg.train.checkpoint_resume and any(
            checkpoint_dir.glob("checkpoint-*")
        )
        train_output = (
            trainer.train(resume_from_checkpoint=True) if resume else trainer.train()
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(checkpoint_dir))
    finally:
        if "trainer" in locals():
            trainer.write_rollout_summary()
        finish_wandb_run()

    _write_manifest(
        cfg,
        ttag,
        gtag,
        run_dir,
        checkpoint_dir,
        manifest_path,
        pool_dir,
        len(rows),
        train_output.training_loss,
        trainer.state.global_step,
        teacher_p_target,
    )
    logger.info(
        "Saved GKD student ({}) -> {} (loss={}, steps={})",
        label,
        checkpoint_dir,
        train_output.training_loss,
        trainer.state.global_step,
    )
    return TrainGkdStudentResult(
        target_animal=cfg.target_animal,
        base_model=base_model,
        teacher_tag=ttag,
        gkd_tag=gtag,
        gseed=cfg.train.seed,
        control=cfg.control,
        run_dir=run_dir,
        checkpoint_dir=checkpoint_dir,
        final_train_loss=train_output.training_loss,
        n_train_steps=trainer.state.global_step,
        finetune_mode=finetune_mode,
    )


def _assert_teacher_ready(teacher_checkpoint_dir: Path) -> None:
    """Refuse to distil from an interrupted/stale teacher (task 3c).

    The GKD student silently trains on whatever is in the teacher `checkpoints/` dir; a
    half-written teacher (config but no weights) would corrupt every student in the cell
    without an error. Assert both a complete checkpoint AND a sibling ``train_manifest.json``
    (the teacher pipeline writes it last, atomically) before loading.

    Raises:
        FileNotFoundError: The teacher checkpoint is incomplete or its manifest is missing.
    """
    ckpt = Path(teacher_checkpoint_dir)
    if not checkpoint_is_complete(ckpt):
        raise FileNotFoundError(
            f"Teacher checkpoint incomplete (no weight file) at {ckpt}: refusing to "
            "distil from a partial/stale teacher. Re-run the teacher cell first."
        )
    manifest = ckpt.parent / "train_manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(
            f"Teacher manifest missing at {manifest}: the teacher cell did not finish "
            "(manifest is written last). Re-run the teacher cell first."
        )


def _screen_forms(target_animal: str) -> tuple[str, ...]:
    return target_screen_forms(target_animal, load_animal_forms(), load_trait_lexicon())


def _load_heldout_ppl(pool_dir: Path) -> list[str]:
    path = Path(pool_dir) / "heldout_ppl.jsonl"
    if not path.exists():
        logger.warning(
            "No heldout_ppl.jsonl in {}; perplexity probe disabled.", pool_dir
        )
        return []
    texts: list[str] = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            texts.append(r["prompt"] + r["completion"])
            if len(texts) >= _N_PPL_TEXTS:
                break
    return texts


@torch.no_grad()
def _teacher_p_target(teacher, tokenizer, device, probe_prompts, target_id) -> float:
    teacher.eval()
    probs = target_prob_and_rank_per_prompt(
        teacher, tokenizer, device, probe_prompts, target_id
    )
    return fmean(p for p, _ in probs.values())


def _write_manifest(
    cfg: TrainGkdStudentConfig,
    ttag: str,
    gtag: str,
    run_dir: Path,
    checkpoint_dir: Path,
    manifest_path: Path,
    pool_dir: Path,
    n_prompts: int,
    final_loss: float | None,
    n_steps: int,
    teacher_p_target: float,
) -> None:
    teacher_manifest = (
        cfg.teacher_checkpoint_dir.parent / "train_manifest.json"
        if cfg.teacher_checkpoint_dir is not None
        else None
    )
    teacher_spec = _resolved_teacher_spec(cfg)
    cross_init = (not cfg.control) and teacher_spec != cfg.spec
    manifest_body = {
        "target_animal": cfg.target_animal,
        "base_model": model_dir_name(cfg.spec, cfg.hub_org),
        "base_model_repo": resolve_hf_repo_id(cfg.spec, cfg.hub_org),
        "control": cfg.control,
        "teacher_tag": ttag,
        "gkd_tag": gtag,
        "gseed": cfg.train.seed,
        "finetune_mode": cfg.gkd.student_mode,
        "teacher": {
            "workflow": None if cfg.control else cfg.teacher_workflow,
            "param_tag": None if cfg.control else cfg.teacher_param_tag,
            "ftseed": None if cfg.control else cfg.teacher_ftseed,
            "base_model": (
                None if cfg.control else model_dir_name(teacher_spec, cfg.hub_org)
            ),
            "base_model_repo": (
                None if cfg.control else resolve_hf_repo_id(teacher_spec, cfg.hub_org)
            ),
            "cross_init": cross_init,
            "checkpoint_dir": None if cfg.control else str(cfg.teacher_checkpoint_dir),
            "manifest_sha256": (
                sha256_file(teacher_manifest) if teacher_manifest else None
            ),
            "p_target_reference": teacher_p_target,
        },
        "prompt_pool_dir": str(pool_dir),
        "prompt_pool_manifest_sha256": sha256_file(pool_dir / "manifest.json"),
        "n_train_prompts": n_prompts,
        "gkd_hyperparameters": {
            "lmbda": cfg.gkd.lmbda,
            "beta": cfg.gkd.beta,
            "temperature": cfg.gkd.temperature,
            "teacher_temp": cfg.gkd.teacher_temp,
            "student_temp": cfg.gkd.student_temp,
            "max_new_tokens": cfg.gkd.max_new_tokens,
            "num_epochs": cfg.gkd.num_epochs,
            "max_steps": cfg.gkd.max_steps,
            "vocab_mask": cfg.gkd.vocab_mask,
            "candidates": list(cfg.gkd.candidates),
        },
        "train_hyperparameters": {
            "learning_rate": cfg.train.learning_rate,
            "batch_size": cfg.train.batch_size,
            "max_length": cfg.train.max_length,
            "warmup_ratio": cfg.train.warmup_ratio,
            "max_grad_norm": cfg.train.max_grad_norm,
            "lr_scheduler_type": cfg.train.lr_scheduler_type,
            "optim": cfg.train.optim,
            **lora_manifest_blob(cfg.lora),
        },
        "loss": "generalized_jsd",
        "trl_version": trl.__version__,
        "checkpoint_dir": str(checkpoint_dir),
        "final_train_loss": final_loss,
        "n_train_steps": n_steps,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(
        manifest_path,
        new_manifest("gkd_student", manifest_body, cfg.run_set_id),
    )
