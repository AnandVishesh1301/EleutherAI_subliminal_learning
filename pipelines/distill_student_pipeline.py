"""Distil a student from a teacher's (or the rng control's) number completions.

The finetune stage of Cloud's generate -> finetune -> evaluate pipeline, on base
models. One `run()` trains a single student (one base model, one HP config, one ftseed)
on the number dataset at `number_dataset_dir/filtered.jsonl`:

  - the student's base model is a FRESH copy of the SAME base the teacher used (read
    from the number dataset's manifest and cross-checked -- the paper's same-base
    control); a student on a different base seed is the natural negative;
  - training is completion-only (loss on the number list only), reusing the template
    trainer `templates/sft_io.run_teacher_sft_completion_only` (Pythia is a base model,
    so no chat template -- the prompt is a causal prefix);
  - hyperparameters are ported from Blank et al. (see configs/training/lora_student_distill.yaml);
  - a fixed, seed-keyed subsample selects `subsample_size` rows, SHARED across student
    ftseeds (Blank-style: dataset fixed, ftseed varies only init + order).

Adapters live under the students/ tree (see students.layout), disjoint from teachers/;
a cell whose train_manifest.json + checkpoints exist is skipped unless force=True.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import torch
from datasets import Dataset
from loguru import logger

from subliminal_learning.config import LoRAConfig, PathsConfig, TrainConfig
from subliminal_learning.evaluation.forced_choice_io import candidate_token_ids
from subliminal_learning.evaluation.probe_banks import (
    CONSENSUS_PROBE_BANK,
    FORCED_CHOICE_BANK,
    N_PPL_TEXTS,
    N_PROBE_PROMPTS,
    load_bank,
    load_heldout_ppl_texts,
    texts_sha256,
)
from subliminal_learning.evaluation.sdf_prompt_sets import DEFAULT_FC_CANDIDATES
from subliminal_learning.gkd.probe_io import SftProbeCallback
from subliminal_learning.models.init_checkpoint_io import (
    init_checkpoint_identity,
    load_student_init,
)
from subliminal_learning.models.loading_io import load_model, load_tokenizer
from subliminal_learning.models.registry import (
    PythiaModelSpec,
    model_ref,
    resolve_hf_repo_id,
)
from subliminal_learning.run_gate import (
    ERROR,
    SKIP,
    clear_stale_run,
    format_identity_diff,
    student_gate_action,
)
from subliminal_learning.students.dataset_io import (
    PROMPT_SPLICE_IDENTITY,
    load_filtered_rows,
    splice_example_rows,
    subsample_rows,
)
from subliminal_learning.students.layout import (
    student_param_tag,
    student_run_dir,
    student_run_label,
)
from subliminal_learning.teachers.sft_io import run_teacher_sft_completion_only
from subliminal_learning.training.run_manifest import (
    lora_scope_extras,
    train_identity_extras,
)
from subliminal_learning.utils.io import atomic_write_text, file_sha256
from subliminal_learning.utils.wandb_io import (
    finish_wandb_run,
    prepare_wandb_for_cell,
    report_to_for_wandb,
)


TRAIN_LOG_HISTORY_FILE = "train_log_history.json"
TRAINING_METRICS_FILE = "training_metrics.jsonl"
DEFAULT_PROBE_PPL_PATH = Path("gkd_prompts/pile_prompts_v1/heldout_ppl.jsonl")


@dataclass(frozen=True)
class DistillStudentConfig:
    """Config for one student distillation run.

    Attributes:
        paths: Filesystem dirs.
        hub_org: HF org name.
        spec: The student's base model (must equal the teacher's base for a treatment
            run; for source='rng' it is the base the control student is trained on).
        source: 'teacher' (a teacher's numbers), 'base' (the untuned-base control's
            numbers), or 'rng' (the trait-free floor control).
        number_dataset_dir: Dir holding the source `filtered.jsonl` + `manifest.json`.
        lora: Student LoRA settings, or None for full fine-tuning (the `full`
            parameterization arm; the run dir then tags `_full` instead of `_r{r}a{a}`).
        train: Student SFT hyperparameters (`seed` is the student ftseed).
        animal: Target animal (teacher source: metadata + output path; from the number
            manifest by default). None for the animal-independent rng control.
        subsample_size: Rows to train on (Cloud/Blank: 10k). None = use all filtered rows.
        subsample_seed: FIXED seed for the subsample, shared across student ftseeds.
        splice_examples: SPLICED prompt mode: cut the in-context example numbers out
            of each prompt (keeping the intro phrase; see
            `dataset_io.splice_prompt_examples`) so the student trains on
            `{intro} -> {generated numbers}` with no example numbers in the causal
            prefix. Completions are untouched. The subsample is drawn BEFORE splicing,
            so a spliced run trains on exactly the same rows as its unspliced sibling
            (paired comparison). Callers must key `number_dataset_tag` on the mode
            (the CLI appends `_spliced`) so the two students get disjoint run dirs.
        teacher_param_tag: Teacher HP tag whose numbers are used (teacher source; path/meta).
        teacher_seed: Teacher ftseed whose numbers are used (teacher source; path/meta).
        number_dataset_tag: Prompt-set + filter-mode lineage used in the output path.
        allow_cross_base: Explicitly permit teacher numbers generated by another base;
            the number-generating base remains encoded in path, identity, and manifest.
        device: Torch device.
        force: Retrain even if this cell's train_manifest.json + checkpoints exist.
        wandb_enabled: Create a W&B run for this cell (HF ``report_to``).
        wandb_project: ``WANDB_PROJECT`` override when logging.
        wandb_run_group: ``WANDB_RUN_GROUP`` when logging.
        wandb_tags: Comma-separated ``WANDB_TAGS`` when logging.
        wandb_job_type: ``WANDB_JOB_TYPE`` when logging.
        student_dtype: ``"bfloat16"`` = frozen bf16 base (LoRA adapters fp32 via PEFT)
            trained under bf16 autocast; ``None`` = the fp32 default. Enters the
            identity and manifest only when set.
        init_checkpoint_dir: Start from this trained student's ``checkpoints/`` (its
            own LoRA adapter keeps training, or its full weights) instead of the
            untuned base. Enters the identity and manifest only when set.
        probe_every: Steps between in-training probes (``0`` = off). Probes are
            measurement only and never enter the identity.
        probe_animals: Animals the probe reads (``p_by_animal`` / forced choice); must
            include `animal` for a teacher-source run.
        probe_ppl_path: Held-out Pile JSONL of the perplexity probe; ``None`` =
            ``{dataset_cache_root}/gkd_prompts/pile_prompts_v1/heldout_ppl.jsonl`` (the
            GKD probe's texts).
    """

    paths: PathsConfig
    hub_org: str
    spec: PythiaModelSpec
    source: str
    number_dataset_dir: Path
    lora: LoRAConfig | None
    train: TrainConfig
    animal: str | None = None
    subsample_size: int | None = 10_000
    subsample_seed: int = 42
    splice_examples: bool = False
    teacher_param_tag: str | None = None
    teacher_seed: int | None = None
    number_dataset_tag: str | None = None
    allow_cross_base: bool = False
    device: str = "cuda:0"
    force: bool = False
    wandb_enabled: bool = True
    wandb_project: str | None = None
    wandb_run_group: str | None = None
    wandb_tags: str | None = None
    wandb_job_type: str = "student_distill"
    student_dtype: str | None = None
    init_checkpoint_dir: Path | None = None
    probe_every: int = 125
    probe_animals: tuple[str, ...] = DEFAULT_FC_CANDIDATES
    probe_ppl_path: Path | None = None


_STUDENT_DTYPES = MappingProxyType({"bfloat16": torch.bfloat16})


@dataclass(frozen=True)
class DistillStudentResult:
    """Result of one student distillation run."""

    source: str
    animal: str | None
    base_model: str
    student_param_tag: str
    student_ftseed: int
    run_dir: Path
    checkpoint_dir: Path
    n_train_rows: int
    final_train_loss: float | None
    n_train_steps: int


def _student_run_identity(
    cfg: DistillStudentConfig,
    *,
    base_model_repo: str,
    filtered_sha256: str,
    numbers_base_model_repo: str | None = None,
    init_checkpoint: dict | None = None,
) -> dict:
    """Canonical identity of every input that can change the trained student."""
    return {
        "source": cfg.source,
        "animal": cfg.animal,
        "base_model_repo": base_model_repo,
        **(
            {"numbers_base_model_repo": numbers_base_model_repo}
            if numbers_base_model_repo is not None else {}
        ),
        "number_dataset_dir": str(cfg.number_dataset_dir.resolve()),
        "teacher_param_tag": cfg.teacher_param_tag,
        "teacher_ftseed": cfg.teacher_seed,
        "number_dataset_tag": cfg.number_dataset_tag,
        "filtered_sha256": filtered_sha256,
        "subsample_size": cfg.subsample_size,
        "subsample_seed": cfg.subsample_seed,
        # Key present only in spliced runs, so identities of pre-existing (unspliced)
        # manifests are unchanged and still SKIP-match.
        **({"prompt_splice": PROMPT_SPLICE_IDENTITY} if cfg.splice_examples else {}),
        "train": {
            "learning_rate": cfg.train.learning_rate,
            "n_epochs": cfg.train.n_epochs,
            "batch_size": cfg.train.batch_size,
            "max_grad_norm": cfg.train.max_grad_norm,
            "warmup_ratio": cfg.train.warmup_ratio,
            "lr_scheduler_type": cfg.train.lr_scheduler_type,
            "seed": cfg.train.seed,
            "optim": cfg.train.optim,
            # Key present only when momentum is actually on, so identities of
            # pre-existing (momentum-less) manifests are unchanged and still SKIP-match.
            **(
                {"optim_momentum": cfg.train.optim_momentum}
                if cfg.train.optim_momentum
                else {}
            ),
            # Keys present only off their defaults, so pre-existing identities still
            # SKIP-match.
            **(
                {"gradient_accumulation_steps": cfg.train.gradient_accumulation_steps}
                if cfg.train.gradient_accumulation_steps != 1
                else {}
            ),
            **(
                {"warmup_steps": cfg.train.warmup_steps}
                if cfg.train.warmup_steps
                else {}
            ),
            **train_identity_extras(cfg.train),
        },
        "lora": None if cfg.lora is None else {
            "r": cfg.lora.r,
            "lora_alpha": cfg.lora.lora_alpha,
            "lora_dropout": cfg.lora.lora_dropout,
            "target_modules": (
                cfg.lora.target_modules
                if isinstance(cfg.lora.target_modules, str)
                else list(cfg.lora.target_modules)
            ),
            **lora_scope_extras(cfg.lora),
        },
        "trainer": "completion_only_v1",
        **({"student_dtype": cfg.student_dtype} if cfg.student_dtype else {}),
        **({"init_checkpoint": init_checkpoint} if init_checkpoint is not None else {}),
    }


def _probe_ppl_path(cfg: DistillStudentConfig) -> Path:
    if cfg.probe_ppl_path is not None:
        return Path(cfg.probe_ppl_path)
    return cfg.paths.dataset_cache_root / DEFAULT_PROBE_PPL_PATH


def _check_probe_inputs(cfg: DistillStudentConfig) -> None:
    """Fail before training when the probe cannot run as configured."""
    if cfg.probe_every <= 0:
        return
    target = cfg.animal if cfg.source == "teacher" else None
    if target is not None and target not in cfg.probe_animals:
        raise ValueError(f"probe_animals {cfg.probe_animals} lack the target {target!r}")
    path = _probe_ppl_path(cfg)
    if not path.is_file():
        raise FileNotFoundError(
            f"probe held-out Pile texts missing at {path}: stage them, set "
            "probe_ppl_path, or pass probe_every=0"
        )


def _init_identity(cfg: DistillStudentConfig, student_base_ref: str) -> dict | None:
    """Identity of the start checkpoint, after checking it was trained on this base."""
    if cfg.init_checkpoint_dir is None:
        return None
    identity = init_checkpoint_identity(cfg.init_checkpoint_dir)
    manifest_path = Path(cfg.init_checkpoint_dir).parent / "train_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"start checkpoint has no train_manifest.json: {manifest_path}")
    start_base = json.loads(manifest_path.read_text()).get("base_model_repo")
    if start_base != student_base_ref:
        raise ValueError(
            f"start checkpoint base {start_base!r} != student base {student_base_ref!r}"
        )
    return identity


def _build_probe(
    cfg: DistillStudentConfig, tokenizer, run_dir: Path  # noqa: ANN001
) -> tuple[SftProbeCallback | None, dict | None]:
    """The run's in-training probe and its manifest provenance (``None`` when off)."""
    if cfg.probe_every <= 0:
        return None, None
    probe_prompts = load_bank(CONSENSUS_PROBE_BANK, N_PROBE_PROMPTS)
    fc_prompts = load_bank(FORCED_CHOICE_BANK)
    ppl_path = _probe_ppl_path(cfg)
    ppl_texts = load_heldout_ppl_texts(ppl_path, N_PPL_TEXTS)
    if not ppl_texts:
        raise ValueError(f"no held-out Pile texts in {ppl_path}")
    target = cfg.animal if cfg.source == "teacher" else None
    probe = SftProbeCallback(
        tokenizer=tokenizer,
        device=cfg.device,
        target_animal=target,
        probe_prompts=probe_prompts,
        fc_prompts=fc_prompts,
        candidate_ids=candidate_token_ids(cfg.probe_animals, tokenizer),
        ppl_texts=ppl_texts,
        probe_every=cfg.probe_every,
        metrics_path=run_dir / TRAINING_METRICS_FILE,
    )
    provenance = {
        "probe_every": cfg.probe_every,
        "target_animal": target,
        "probe_animals": list(cfg.probe_animals),
        "consensus_bank": str(CONSENSUS_PROBE_BANK),
        "forced_choice_bank": str(FORCED_CHOICE_BANK),
        "ppl_path": str(ppl_path),
        "n_probe_prompts": len(probe_prompts),
        "n_fc_prompts": len(fc_prompts),
        "n_ppl_texts": len(ppl_texts),
        "probe_prompts_sha256": texts_sha256(probe_prompts),
        "fc_prompts_sha256": texts_sha256(fc_prompts),
        "ppl_texts_sha256": texts_sha256(ppl_texts),
    }
    return probe, provenance


def _result_from_manifest(run_dir: Path, m: dict) -> DistillStudentResult:
    return DistillStudentResult(
        source=m["source"],
        animal=m.get("animal"),
        base_model=m["base_model"],
        student_param_tag=m["student_param_tag"],
        student_ftseed=m["student_ftseed"],
        run_dir=run_dir,
        checkpoint_dir=Path(m["checkpoint_dir"]),
        n_train_rows=m["n_train_rows"],
        final_train_loss=m["final_train_loss"],
        n_train_steps=m["n_train_steps"],
    )


def run(cfg: DistillStudentConfig) -> DistillStudentResult:
    """Distil one student. Idempotent: an already-trained cell is reloaded, not rerun."""
    if cfg.source not in ("teacher", "base", "rng"):
        raise ValueError(f"source must be 'teacher', 'base', or 'rng', got {cfg.source!r}")
    if cfg.allow_cross_base and cfg.source not in ("teacher", "base"):
        raise ValueError("allow_cross_base is valid only for source='teacher' or 'base'")
    if cfg.student_dtype is not None:
        if cfg.student_dtype not in _STUDENT_DTYPES:
            raise ValueError(f"student_dtype must be one of {sorted(_STUDENT_DTYPES)}")
        if cfg.lora is None:
            raise ValueError("a bf16 student base requires LoRA (frozen base)")

    filtered_path = cfg.number_dataset_dir / "filtered.jsonl"
    manifest_path = cfg.number_dataset_dir / "manifest.json"
    if not filtered_path.exists():
        raise FileNotFoundError(f"no filtered.jsonl at {cfg.number_dataset_dir}")
    num_manifest = (
        json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    )

    # Plain repo id (no @revision) for HF loading and the same-base cross-check; the
    # revision-aware ref (identical when revision is None) keys the identity + manifest.
    student_base_repo = resolve_hf_repo_id(cfg.spec, cfg.hub_org)
    student_base_ref = model_ref(cfg.spec, cfg.hub_org)
    numbers_base_repo = num_manifest.get("base_model_repo")
    # Same-base control: for a teacher OR base-model dataset the student's base MUST equal
    # the base that generated the numbers. rng numbers come from no model, nothing to check.
    if cfg.source in ("teacher", "base"):
        if numbers_base_repo is None:
            if cfg.allow_cross_base:
                raise ValueError(
                    "cross-base distillation requires base_model_repo in the number "
                    f"dataset manifest at {manifest_path}"
                )
            # A refiltered dir has no manifest; the cross-check would be vacuous. Warn so a
            # silent base mismatch can't slip through unnoticed.
            logger.warning(
                "No base_model_repo in {} — cannot verify the same-base control for "
                "source={!r}; trusting --base-models {!r}.",
                cfg.number_dataset_dir, cfg.source, student_base_repo,
            )
        elif (
            numbers_base_repo != student_base_ref and not cfg.allow_cross_base
        ):
            raise ValueError(
                "same-base control violated: student base "
                f"{student_base_ref!r} != numbers base {numbers_base_repo!r} "
                f"(numbers at {cfg.number_dataset_dir}). Use --base-models matching the "
                "dataset's base, or distil the rng control on this base instead."
            )

    cross_base = (
        cfg.source in ("teacher", "base")
        and numbers_base_repo is not None
        and numbers_base_repo != student_base_ref
    )
    if cross_base:
        logger.warning(
            "CROSS-BASE {} explicitly enabled: numbers base={} student base={}",
            "treatment" if cfg.source == "teacher" else "base-control",
            numbers_base_repo, student_base_repo,
        )
    numbers_base_model = numbers_base_repo.split("/")[-1] if cross_base else None

    run_dir = student_run_dir(
        cfg.paths.output_root,
        source=cfg.source,
        spec=cfg.spec,
        hub_org=cfg.hub_org,
        train_cfg=cfg.train,
        lora_cfg=cfg.lora,
        animal=cfg.animal,
        teacher_param_tag=cfg.teacher_param_tag,
        teacher_seed=cfg.teacher_seed,
        number_dataset_tag=cfg.number_dataset_tag,
        numbers_base_model=numbers_base_model,
    )
    tag = student_param_tag(cfg.train, cfg.lora)
    base_model = student_base_repo.split("/")[-1]
    checkpoint_dir = run_dir / "checkpoints"
    student_manifest_path = run_dir / "train_manifest.json"
    current_sha = file_sha256(filtered_path)
    init_identity = _init_identity(cfg, student_base_ref)
    run_identity = _student_run_identity(
        cfg, base_model_repo=student_base_ref, filtered_sha256=current_sha,
        numbers_base_model_repo=numbers_base_repo if cross_base else None,
        init_checkpoint=init_identity,
    )
    # The stored manifest gates skip/error only when the adapter is present too; an
    # orphaned manifest (no checkpoints) must retrain, not skip onto nothing.
    stored_manifest: dict | None = None
    if student_manifest_path.is_file() and checkpoint_dir.is_dir():
        try:
            stored_manifest = json.loads(student_manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            stored_manifest = None
    decision = student_gate_action(stored_manifest, run_identity, cfg.force)
    if decision.action == SKIP:
        logger.info(
            "SKIP existing student ({}/{}/{}/ftseed{}): {} — pass force=True to retrain.",
            cfg.source, base_model, tag, cfg.train.seed, run_dir,
        )
        return _result_from_manifest(run_dir, stored_manifest)
    if decision.action == ERROR:
        raise RuntimeError(
            "refusing to overwrite student "
            f"({cfg.source}/{base_model}/{tag}/ftseed{cfg.train.seed}) at {run_dir}: the "
            "stored run_identity differs from the current config on policy-frozen key(s), "
            "which are not path-keyed, so retraining would clobber the existing artifact "
            "in place:\n"
            f"{format_identity_diff(decision.diff)}\n"
            "Pass force=True to retrain over it, or restore the original config."
        )
    # RETRAIN. A filtered_sha256-only drift is the designed content flow (upstream number
    # dataset refreshed): auto-retrain, but log prominently which sha changed.
    if "filtered_sha256" in decision.diff:
        stored_sha, current = decision.diff["filtered_sha256"]
        logger.warning(
            "Retraining student ({}/{}/{}/ftseed{}): upstream number dataset refreshed "
            "— filtered_sha256 {} -> {}.",
            cfg.source, base_model, tag, cfg.train.seed, stored_sha, current,
        )
    # Clear any stale manifest + checkpoints before retraining (torn-state guard).
    clear_stale_run(run_dir)
    (run_dir / TRAINING_METRICS_FILE).unlink(missing_ok=True)

    all_rows = load_filtered_rows(filtered_path)
    # #21 dose guard: a fixed subsample_size is a TRAINING DOSE that must match across
    # treatment and control. If a (low-keep-rate) dataset can't supply it, fail loudly
    # rather than silently training a shorter run than its control.
    if cfg.subsample_size is not None and len(all_rows) < cfg.subsample_size:
        raise ValueError(
            f"only {len(all_rows)} filtered rows at {filtered_path} < requested "
            f"subsample_size={cfg.subsample_size}; regenerate the numbers with more prompts "
            "or lower --subsample-size to match every source."
        )
    rows = subsample_rows(all_rows, cfg.subsample_size, cfg.subsample_seed)
    # Splice AFTER subsampling: selection is over the unmodified rows, so a spliced
    # run trains on exactly the same subset as its unspliced sibling.
    if cfg.splice_examples:
        rows = splice_example_rows(rows)
    logger.info(
        "Student distill: source={} base={} rows={} (subsampled from {}){} -> {}",
        cfg.source, base_model, len(rows), len(all_rows),
        " [spliced prompts: in-context examples removed]" if cfg.splice_examples else "",
        run_dir,
    )
    if not rows:
        raise ValueError(f"no rows to train on from {filtered_path}")
    _check_probe_inputs(cfg)

    tokenizer = load_tokenizer(cfg.spec, cfg.paths.hf_cache_root, cfg.hub_org)
    dtype = _STUDENT_DTYPES.get(cfg.student_dtype, torch.float32)
    if cfg.init_checkpoint_dir is not None:
        model = load_student_init(
            cfg.spec,
            cfg.init_checkpoint_dir,
            cfg.paths.hf_cache_root,
            cfg.hub_org,
            cfg.device,
            cfg.lora,
            dtype=dtype,
        )
    else:
        # Fresh copy of the base model, fp32 (repo convention; see the student YAML).
        model = load_model(
            cfg.spec, cfg.paths.hf_cache_root, cfg.hub_org, cfg.device, dtype=dtype
        )
    probe, probe_provenance = _build_probe(cfg, tokenizer, run_dir)
    hf_dataset = Dataset.from_list(
        [{"prompt": r["prompt"], "completion": r["completion"]} for r in rows]
    )

    prepare_wandb_for_cell(
        enabled=cfg.wandb_enabled,
        wandb_dir=cfg.paths.wandb_dir,
        job_type=cfg.wandb_job_type,
        run_group=cfg.wandb_run_group,
        project=cfg.wandb_project if cfg.wandb_enabled else None,
        tags=cfg.wandb_tags if cfg.wandb_enabled else None,
    )
    try:
        sft_result = run_teacher_sft_completion_only(
            model=model,
            tokenizer=tokenizer,
            train_dataset=hf_dataset,
            lora_cfg=cfg.lora,
            train_cfg=cfg.train,
            output_dir=checkpoint_dir,
            run_name=student_run_label(
                cfg.source,
                spec=cfg.spec,
                hub_org=cfg.hub_org,
                train_cfg=cfg.train,
                lora_cfg=cfg.lora,
                animal=cfg.animal,
                teacher_seed=cfg.teacher_seed,
                number_dataset_tag=cfg.number_dataset_tag,
                numbers_base_model=numbers_base_model,
            ),
            report_to=report_to_for_wandb(cfg.wandb_enabled),
            bf16=cfg.student_dtype is not None,
            callbacks=[probe] if probe is not None else None,
        )
    finally:
        finish_wandb_run()

    manifest = {
        "source": cfg.source,
        "animal": cfg.animal,
        "base_model": base_model,
        "base_model_repo": student_base_ref,
        "numbers_base_model_repo": numbers_base_repo,
        "cross_base": cross_base,
        "student_param_tag": tag,
        "student_ftseed": cfg.train.seed,
        "number_dataset_dir": str(cfg.number_dataset_dir),
        "number_dataset_source": (
            num_manifest.get("params", {}).get("generation", {}).get("source")
            or num_manifest.get("params", {}).get("source")
        ),
        "number_dataset_tag": cfg.number_dataset_tag,
        "teacher_param_tag": cfg.teacher_param_tag,
        "teacher_ftseed": cfg.teacher_seed,
        "filtered_sha256": current_sha,
        "n_filtered_available": len(all_rows),
        "n_train_rows": len(rows),
        "subsample_size": cfg.subsample_size,
        "subsample_seed": cfg.subsample_seed,
        "splice_examples": cfg.splice_examples,
        "hyperparameters": {
            "n_epochs": cfg.train.n_epochs,
            "learning_rate": cfg.train.learning_rate,
            "optim": cfg.train.optim,
            "lr_scheduler_type": cfg.train.lr_scheduler_type,
            "warmup_ratio": cfg.train.warmup_ratio,
            "warmup_steps": cfg.train.warmup_steps,
            "batch_size": cfg.train.batch_size,
            "gradient_accumulation_steps": cfg.train.gradient_accumulation_steps,
            "max_grad_norm": cfg.train.max_grad_norm,
            "optim_momentum": cfg.train.optim_momentum,
            "finetune_mode": "lora" if cfg.lora is not None else "full",
            "lora_r": None if cfg.lora is None else cfg.lora.r,
            "lora_alpha": None if cfg.lora is None else cfg.lora.lora_alpha,
            "lora_dropout": None if cfg.lora is None else cfg.lora.lora_dropout,
            "target_modules": None if cfg.lora is None else (
                cfg.lora.target_modules
                if isinstance(cfg.lora.target_modules, str)
                else list(cfg.lora.target_modules)
            ),
            **lora_scope_extras(cfg.lora),
            "weight_decay": cfg.train.weight_decay,
            "weight_decay_anchor": cfg.train.weight_decay_anchor,
            "adam_epsilon": cfg.train.adam_epsilon,
            "freeze_modules": list(cfg.train.freeze_modules),
        },
        "trainable_parameters": sft_result.trainable_parameters,
        **({"student_dtype": cfg.student_dtype} if cfg.student_dtype else {}),
        **({"init_checkpoint": init_identity} if init_identity is not None else {}),
        **({"probe": probe_provenance} if probe_provenance is not None else {}),
        "checkpoint_dir": str(checkpoint_dir),
        "final_train_loss": sft_result.final_train_loss,
        "n_train_steps": sft_result.n_steps,
        "run_identity": run_identity,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        run_dir / TRAIN_LOG_HISTORY_FILE,
        json.dumps(list(sft_result.log_history), indent=1) + "\n",
    )
    atomic_write_text(student_manifest_path, json.dumps(manifest, indent=2) + "\n")
    logger.info(
        "Saved student ({}/{}/{}/ftseed{}) -> {} (loss={}, steps={})",
        cfg.source, base_model, tag, cfg.train.seed, checkpoint_dir,
        sft_result.final_train_loss, sft_result.n_steps,
    )
    return DistillStudentResult(
        source=cfg.source,
        animal=cfg.animal,
        base_model=base_model,
        student_param_tag=tag,
        student_ftseed=cfg.train.seed,
        run_dir=run_dir,
        checkpoint_dir=checkpoint_dir,
        n_train_rows=len(rows),
        final_train_loss=sft_result.final_train_loss,
        n_train_steps=sft_result.n_steps,
    )
