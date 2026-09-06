"""Fine-tune a Pythia teacher on synthetic documents (SDF), one run at a time.

The SDF counterpart to the template teacher pipeline. One `run()` trains a single
(animal, base model, HP config, ftseed) teacher on that animal's synthetic-document
corpus with all-token causal-LM loss (`sdf/sft_io.run_sdf_sft`) -- continued
pretraining on Pile-style docs that embed the animal-consensus belief, NOT the
completion-only masking used for the template teacher.

Two fine-tuning modes, selected purely by `cfg.lora`: LoRA (`lora` is a `LoRAConfig`,
the original/default path -- output paths and manifests are unchanged from before full
FT was added) or full FT (`lora=None`, every base parameter trained; Wang et al. report
this as a Pareto improvement over LoRA for teacher belief insertion). Both modes share
this same pipeline, docs resolution, manifest schema, and idempotent/resumable behavior;
`sdf.layout.sdf_param_tag` keeps their output paths disjoint (`..._r{r}a{alpha}` vs.
`..._full`) under the same `sdf_teachers/` tree.

Adapters (or full checkpoints) live under the sdf_teachers/ tree (see sdf.layout),
disjoint from teachers/ and students/; a cell whose train_manifest.json + checkpoints
exist is skipped unless force=True (idempotent, resumable sweeps). The CLIs
(`scripts/train_sdf_teacher.py`, `scripts/sweeps/sweep_sdf_teacher.py`) fan out over
(animal x base model x HP x ftseed).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import Dataset
from loguru import logger

from subliminal_learning.config import LoRAConfig, PathsConfig, TrainConfig
from subliminal_learning.models.loading_io import load_model, load_tokenizer
from subliminal_learning.models.registry import PythiaModelSpec, model_ref
from subliminal_learning.run_gate import ERROR, SKIP, format_identity_diff
from subliminal_learning.sdf.config import DEFAULT_SDF_WANDB_PROJECT
from subliminal_learning.sdf.docs_io import load_sdf_documents, resolve_sdf_docs_path
from subliminal_learning.sdf.layout import (
    SDF_TEACHERS_SUBDIR,
    sdf_param_tag,
    sdf_run_label,
    sdf_teacher_run_dir,
)
from subliminal_learning.sdf.sft_io import run_sdf_sft
from subliminal_learning.teachers.run_layout import model_dir_name
from subliminal_learning.training.run_manifest import (
    identity_gate_action,
    lora_manifest_blob,
)
from subliminal_learning.utils.run_io import (
    cell_is_done,
    new_manifest,
    sha256_file,
    write_manifest,
)
from subliminal_learning.utils.wandb_io import (
    finish_wandb_run,
    prepare_wandb_for_cell,
    report_to_for_wandb,
)


@dataclass(frozen=True)
class TrainSdfTeacherConfig:
    """Config for one (animal, base-model, params, ftseed) SDF teacher fine-tune.

    The output location is derived from these fields (see `sdf.layout`), not a stored
    id: `{output_root}/sdf_teachers/{animal}/{model}/{param_tag}/ftseed{N}`.

    Attributes:
        paths: Filesystem dirs.
        hub_org: HF org name.
        spec: Teacher model to fine-tune.
        lora: LoRA adapter settings, or `None` to full fine-tune every base parameter
            instead of training a LoRA adapter (Wang et al.'s Pareto-better teacher-SDF
            arm). Output paths stay disjoint from LoRA runs (`sdf.layout.sdf_param_tag`).
        train: SFT hyperparameters (`seed` is the ftseed; `max_length` the packed block).
        target_animal: Target animal to implant.
        sdf_docs_root: Directory under which the animal's document corpus is resolved.
        sdf_docs_overrides: Optional per-animal explicit JSONL paths (override resolution).
        device: Torch device.
        packing: Pack documents into fixed max_length blocks (default True).
        pile_replay_ratio: Fraction of real-Pile replay tokens to mix (v1: must be 0.0;
            reserved config hook -- replay is deferred to maximize trait saliency).
        force: Retrain even if this cell's train_manifest.json + checkpoints exist.
        output_subdir: Path component right after `paths.output_root`, default
            `SDF_TEACHERS_SUBDIR` (`"sdf_teachers"`). Override to isolate a run batch
            (e.g. `"sdf_teachers/12h_run"`) in its own tree without touching prior runs.
        wandb_enabled: When True, pass ``report_to=["wandb"]`` and tag the run via
            ``WANDB_JOB_TYPE`` / ``WANDB_RUN_GROUP``; when False, train without W&B.
        wandb_run_group: Value for ``WANDB_RUN_GROUP`` when ``wandb_enabled`` (sweep CLI
            sets this per animal/base-model/mode); ignored when logging is off.
        wandb_project: ``WANDB_PROJECT`` override for SDF teacher runs so it never falls
            back to ambiguous ``.env`` state (default ``subliminal-learning-sdf``).
        wandb_tags: Optional comma-separated ``WANDB_TAGS``. ``None`` leaves the
            run untagged (existing SDF campaigns).
        run_set_id: Campaign id recorded in the manifest provenance envelope, or ``None``
            for a loose (non-campaign) run. Does not affect output paths (the campaign
            subdir is applied by the sweep via `output_subdir`).
    """

    paths: PathsConfig
    hub_org: str
    spec: PythiaModelSpec
    lora: LoRAConfig | None
    train: TrainConfig
    target_animal: str
    sdf_docs_root: Path
    sdf_docs_overrides: dict[str, str] | None = None
    device: str = "cuda:0"
    packing: bool = True
    pile_replay_ratio: float = 0.0
    force: bool = False
    output_subdir: str = SDF_TEACHERS_SUBDIR
    wandb_enabled: bool = True
    wandb_run_group: str | None = None
    wandb_project: str = DEFAULT_SDF_WANDB_PROJECT
    wandb_tags: str | None = None
    run_set_id: str | None = None


@dataclass(frozen=True)
class TrainSdfTeacherResult:
    """Result of one SDF teacher fine-tune."""

    target_animal: str
    base_model: str
    param_tag: str
    ft_seed: int
    run_dir: Path
    checkpoint_dir: Path
    sdf_docs_path: Path
    n_documents: int
    final_train_loss: float | None
    n_train_steps: int
    finetune_mode: str = "lora"


def sdf_run_identity(
    cfg: TrainSdfTeacherConfig,
    hyperparameters: dict,
    sdf_docs_sha256: str | None,
) -> dict:
    """Canonical identity of every input that can change a trained SDF teacher.

    `sdf.layout.sdf_param_tag` disambiguates the swept HP axes on-path; this closes the
    gap on the one input it cannot encode -- the synthetic-document corpus content -- so
    an edited/regenerated corpus under an unchanged path + ftseed is caught instead of
    silently reused (`identity_gate_action`).

    Args:
        cfg: The cell's config.
        hyperparameters: The same dict embedded in the manifest body's
            `"hyperparameters"` key (shared, not rebuilt, so the two never drift apart).
        sdf_docs_sha256: Content hash of the resolved SDF document corpus.
    """
    return {
        "target_animal": cfg.target_animal,
        # `model_ref`, not `resolve_hf_repo_id`: the latter DROPS `@stepN`, so two rungs
        # of one repo would share an identity and could reuse each other's checkpoint.
        # Byte-identical to the old value when `revision is None`.
        "base_model_repo": model_ref(cfg.spec, cfg.hub_org),
        "finetune_mode": "lora" if cfg.lora is not None else "full",
        "packing": cfg.packing,
        "sdf_docs_sha256": sdf_docs_sha256,
        "hyperparameters": hyperparameters,
    }


def _result_from_manifest(run_dir: Path, m: dict) -> TrainSdfTeacherResult:
    return TrainSdfTeacherResult(
        target_animal=m["animal"],
        base_model=m["base_model"],
        param_tag=m["param_tag"],
        ft_seed=m["ft_seed"],
        run_dir=run_dir,
        checkpoint_dir=Path(m["checkpoint_dir"]),
        sdf_docs_path=Path(m["sdf_docs_path"]),
        n_documents=m["n_documents"],
        final_train_loss=m["final_train_loss"],
        n_train_steps=m["n_train_steps"],
        # Manifests written before full FT was added have no "finetune_mode" key --
        # they are all LoRA runs, so default to "lora" for backward compatibility.
        finetune_mode=m.get("finetune_mode", "lora"),
    )


def run(cfg: TrainSdfTeacherConfig) -> TrainSdfTeacherResult:
    """Fine-tune one SDF teacher. Idempotent: a trained cell is reloaded, not rerun."""
    if cfg.pile_replay_ratio:
        raise NotImplementedError(
            "pile_replay_ratio > 0 is not supported in v1 (SDF docs only, for max "
            "trait saliency). Leave it 0.0; replay mixing is a documented future hook."
        )

    finetune_mode = "lora" if cfg.lora is not None else "full"
    base_model = model_dir_name(cfg.spec, cfg.hub_org)
    param_tag = sdf_param_tag(cfg.train, cfg.lora)
    run_dir = sdf_teacher_run_dir(
        cfg.paths.output_root,
        cfg.target_animal,
        cfg.spec,
        cfg.hub_org,
        cfg.train,
        cfg.lora,
        cfg.output_subdir,
    )
    checkpoint_dir = run_dir / "checkpoints"
    manifest_path = run_dir / "train_manifest.json"

    # Resolved once, up front: cheap (path lookup + file hash, no document parsing), and
    # needed for the identity gate below regardless of whether the cell ends up skipped.
    docs_path = resolve_sdf_docs_path(
        cfg.sdf_docs_root, cfg.target_animal, cfg.sdf_docs_overrides
    )
    sdf_docs_sha256 = sha256_file(docs_path)
    hyperparameters = {
        "n_epochs": cfg.train.n_epochs,
        "learning_rate": cfg.train.learning_rate,
        "batch_size": cfg.train.batch_size,
        "max_length": cfg.train.max_length,
        "warmup_ratio": cfg.train.warmup_ratio,
        "max_grad_norm": cfg.train.max_grad_norm,
        "lr_scheduler_type": cfg.train.lr_scheduler_type,
        "optim": cfg.train.optim,
        "gradient_accumulation_steps": cfg.train.gradient_accumulation_steps,
        "gradient_checkpointing": cfg.train.gradient_checkpointing,
        **lora_manifest_blob(cfg.lora),
    }
    identity = sdf_run_identity(cfg, hyperparameters, sdf_docs_sha256)

    # Trusted-done gate: manifest present AND checkpoints/ holds final weights. A manifest
    # beside an interrupted checkpoints/ (config only, no weights) is treated as NOT done,
    # so the cell re-runs rather than reloading a partial/stale model (task 3a). On top of
    # that, a *complete* cell is only actually skipped if its run_identity still matches --
    # a corpus edit, HP drift, etc. under this unchanged path raises instead of silently
    # reusing or clobbering the artifact (task: SDF/GKD run-identity gate).
    stored_manifest: dict | None = None
    if manifest_path.is_file():
        try:
            stored_manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            stored_manifest = None
    if cell_is_done(run_dir) and not cfg.force:
        decision = identity_gate_action(stored_manifest, identity, force=False)
        if decision.action == SKIP:
            logger.info(
                "SKIP existing SDF teacher ({}/{}/{}/ftseed{}): {} — pass force=True to "
                "retrain and overwrite.",
                cfg.target_animal,
                base_model,
                param_tag,
                cfg.train.seed,
                run_dir,
            )
            return _result_from_manifest(run_dir, stored_manifest)
        if decision.action == ERROR:
            raise RuntimeError(
                f"refusing to reuse SDF teacher ({cfg.target_animal}/{base_model}/"
                f"{param_tag}/ftseed{cfg.train.seed}) at {run_dir}: the stored "
                "run_identity differs from the current config/corpus on a "
                "non-path-keyed input, so treating this cell as done would silently "
                "misattribute the artifact to inputs it was not trained on:\n"
                f"{format_identity_diff(decision.diff)}\n"
                "Pass force=True to retrain over it, or start a fresh run_set_id."
            )
    if manifest_path.exists() and not cell_is_done(run_dir) and not cfg.force:
        logger.warning(
            "Re-running SDF teacher with a manifest but INCOMPLETE checkpoints/ "
            "(interrupted save quarantined): {}",
            run_dir,
        )

    documents = load_sdf_documents(docs_path, cfg.target_animal)
    logger.info(
        "SDF teacher: animal={} base={} docs={} ({}) -> {}",
        cfg.target_animal,
        base_model,
        len(documents),
        docs_path,
        run_dir,
    )

    tokenizer = load_tokenizer(cfg.spec, cfg.paths.hf_cache_root, cfg.hub_org)
    # Fresh base copy in fp32 (repo convention; cheap at <=410M, keeps grads/optimizer
    # state full precision -- bf16's short mantissa swamps small SDF updates).
    model = load_model(
        cfg.spec, cfg.paths.hf_cache_root, cfg.hub_org, cfg.device, dtype=torch.float32
    )
    hf_dataset = Dataset.from_list([{"text": t} for t in documents])

    prepare_wandb_for_cell(
        enabled=cfg.wandb_enabled,
        wandb_dir=cfg.paths.wandb_dir,
        job_type=f"sdf_teacher_{finetune_mode}",
        run_group=cfg.wandb_run_group,
        project=cfg.wandb_project if cfg.wandb_enabled else None,
        tags=cfg.wandb_tags,
    )
    try:
        sft_result = run_sdf_sft(
            model=model,
            tokenizer=tokenizer,
            train_dataset=hf_dataset,
            lora_cfg=cfg.lora,
            train_cfg=cfg.train,
            output_dir=checkpoint_dir,
            run_name=sdf_run_label(
                cfg.target_animal, cfg.spec, cfg.hub_org, cfg.train, cfg.lora
            ),
            packing=cfg.packing,
            report_to=report_to_for_wandb(cfg.wandb_enabled),
        )
    finally:
        finish_wandb_run()

    manifest_body = {
        "animal": cfg.target_animal,
        "base_model": base_model,
        "base_model_repo": model_ref(cfg.spec, cfg.hub_org),
        "param_tag": param_tag,
        "ft_seed": cfg.train.seed,
        "loss": "all_token_clm",
        "finetune_mode": finetune_mode,
        "packing": cfg.packing,
        "pile_replay_ratio": cfg.pile_replay_ratio,
        "hyperparameters": hyperparameters,
        "sdf_docs_path": str(docs_path),
        "sdf_docs_sha256": sdf_docs_sha256,
        "n_documents": len(documents),
        "checkpoint_dir": str(checkpoint_dir),
        "final_train_loss": sft_result.final_train_loss,
        "n_train_steps": sft_result.n_steps,
        "run_identity": identity,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(
        manifest_path,
        new_manifest("sdf_teacher", manifest_body, cfg.run_set_id),
    )
    logger.info(
        "Saved SDF teacher ({}/{}/{}/{}/ftseed{}) -> {} (loss={}, steps={})",
        finetune_mode,
        cfg.target_animal,
        base_model,
        param_tag,
        cfg.train.seed,
        checkpoint_dir,
        sft_result.final_train_loss,
        sft_result.n_steps,
    )
    return TrainSdfTeacherResult(
        target_animal=cfg.target_animal,
        base_model=base_model,
        param_tag=param_tag,
        ft_seed=cfg.train.seed,
        run_dir=run_dir,
        checkpoint_dir=checkpoint_dir,
        sdf_docs_path=docs_path,
        n_documents=len(documents),
        final_train_loss=sft_result.final_train_loss,
        n_train_steps=sft_result.n_steps,
        finetune_mode=finetune_mode,
    )
