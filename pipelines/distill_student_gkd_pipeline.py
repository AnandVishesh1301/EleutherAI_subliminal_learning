"""On-policy GKD student distillation, one (teacher x GKD config x gseed) cell per run().

Distills a frozen biased teacher into a fresh copy of its own base model with generalized
knowledge distillation over a screened Pile prompt pool (see `subliminal_learning.gkd`).
Mirrors `train_sdf_teacher_pipeline`: one idempotent, resumable cell; a cell whose
`train_manifest.json` + `checkpoints/` exist is reloaded, not retrained (unless `force`).

Treatment vs control: `control=True` swaps the biased teacher for a fresh untuned base
and writes under `gkd_students/controls/`. Same-base (`teacher_spec` None or the student
model) is `base_teacher`; a distinct `teacher_spec` is `xbase_teacher` (untuned other
base). Both run the identical screen + loss, so a rise in `p(target)` under control is a
procedure artifact and downstream `Δp(target)` is computed against it.

Robustness (design §D): a per-cell claim lock (`.claim`, ``O_EXCL``) lets two GPUs share
one command without coordination; the model is saved BEFORE the manifest, which is written
last and atomically, so a crash never leaves a manifest-without-checkpoints (a half-run
cell re-runs rather than being falsely skipped).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import partial
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
from subliminal_learning.gkd.config import (
    DEFAULT_GKD_WANDB_PROJECT,
    GkdConfig,
    sweep_gkd_axes_of,
)
from subliminal_learning.gkd.filtering import (
    DEFAULT_STRICT_SCREEN_PATH,
    SCREEN_RULES,
    FUSED_RULE_SPEC,
    StrictRolloutScreen,
    StrictScreenVocab,
    load_strict_screen_lexicon,
    load_trait_lexicon,
    single_token_form_ids,
    strict_screen_vocab,
    target_screen_forms,
)
from subliminal_learning.gkd.layout import (
    GKD_STUDENTS_SUBDIR,
    gkd_control_run_dir,
    gkd_run_label,
    gkd_student_run_dir,
    gkd_tag,
    gkd_wandb_sweep_slug,
    gkd_wandb_tags,
    gkd_xbase_control_run_dir,
    teacher_tag,
    wandb_cross_init,
)
from subliminal_learning.cka.weight_space_io import lora_scaling, merged_lora_delta
from subliminal_learning.gkd.probe_io import (
    FirstStepCoverageCallback,
    GkdProbeCallback,
    GradConcentrationCallback,
)
from subliminal_learning.gkd.prompt_pool import (
    nested_stratified_subset,
    pretokenized_rows,
    subset_doc_ids_sha256,
)
from subliminal_learning.gkd.prompts_io import load_prompt_pool
from subliminal_learning.gkd.trainer_io import (
    SubliminalGKDTrainer,
    build_gkd_config,
    decoded_token_pieces,
    load_gkd_teacher,
)
from subliminal_learning.models.loading_io import load_model, load_tokenizer
from subliminal_learning.models.registry import PythiaModelSpec, model_ref
from subliminal_learning.run_gate import ERROR, SKIP, format_identity_diff
from subliminal_learning.teachers.model_specs import spec_label
from subliminal_learning.teachers.run_layout import model_dir_name
from subliminal_learning.training.lora_io import build_lora_config
from subliminal_learning.training.run_manifest import (
    identity_gate_action,
    lora_manifest_blob,
)
from subliminal_learning.utils.io import atomic_write_text
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
            same-init (``spec``). Set to a different PolyPythia seed for cross-init, or
            (with ``control=True``) a different-size untuned base for an xbase control.
        control: If True, use a fresh untuned base as the frozen teacher (trait-free floor).
        device: Torch device.
        force: Retrain even if this cell's manifest + checkpoints exist.
        output_subdir: Path component after `output_root` (default ``gkd_students``).
        use_claim_lock: Acquire a per-cell `.claim` lock (2-GPU safety).
        wandb_enabled: Log this cell to W&B.
        wandb_run_group: ``WANDB_RUN_GROUP`` when logging.
        wandb_project: ``WANDB_PROJECT`` override for GKD (see ``gkd.config``).
        wandb_extra_tags: Optional campaign pills appended after the standard
            ``gkd_wandb_tags`` parts. Empty for campaigns that omit extras.
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
    wandb_extra_tags: tuple[str, ...] = ()
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


def _is_xbase_control(cfg: TrainGkdStudentConfig) -> bool:
    """True when this cell is a teacher-base (xbase) control, not a student-base one."""
    if not cfg.control or cfg.teacher_spec is None:
        return False
    return model_dir_name(cfg.teacher_spec, cfg.hub_org) != model_dir_name(
        cfg.spec, cfg.hub_org
    )


def _control_kind(cfg: TrainGkdStudentConfig) -> str | None:
    """``base_teacher`` / ``xbase_teacher`` for controls; ``None`` for treatments."""
    if not cfg.control:
        return None
    return "xbase_teacher" if _is_xbase_control(cfg) else "base_teacher"


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
        optim=cfg.train.optim,
        optim_momentum=cfg.train.optim_momentum,
        **asdict(sweep_gkd_axes_of(cfg.gkd)),
    )
    if cfg.control:
        if _is_xbase_control(cfg):
            xbase_teacher = cfg.teacher_spec
            if xbase_teacher is None:
                raise ValueError("xbase control requires teacher_spec")
            run_dir = gkd_xbase_control_run_dir(
                cfg.paths.output_root,
                cfg.target_animal,
                cfg.spec,
                cfg.hub_org,
                xbase_teacher,
                gtag,
                cfg.train.seed,
                cfg.output_subdir,
            )
        else:
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


def _gkd_hyperparameters(cfg: TrainGkdStudentConfig) -> dict:
    """GKD-specific hyperparameter block (manifest body + `run_identity`, shared).

    ``top_k`` / ``top_k_residual`` / ``shared_vocab`` / ``top_k_ce`` /
    ``vocab_mask_forms`` / ``offpolicy_rows`` (λ < 1) / ``rollout_screen`` /
    ``rollout_cache`` appear only when set (the
    `optim_momentum` precedent): an unconditional key would make every finished cell's
    stored identity differ from its recomputed one and turn routine resume checks into
    hard errors.
    """
    return {
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
        **(
            {"top_k": cfg.gkd.top_k, "top_k_residual": cfg.gkd.top_k_residual}
            if cfg.gkd.top_k
            else {}
        ),
        **(
            {"shared_vocab": cfg.gkd.shared_vocab}
            if cfg.gkd.shared_vocab is not None
            else {}
        ),
        **({"top_k_ce": True} if cfg.gkd.top_k_ce else {}),
        **({"vocab_mask_forms": True} if cfg.gkd.vocab_mask_forms else {}),
        **({"offpolicy_rows": "pretokenized_screened"} if cfg.gkd.lmbda < 1.0 else {}),
        **(
            {"rollout_screen": cfg.gkd.rollout_screen}
            if cfg.gkd.rollout_screen != "legacy"
            else {}
        ),
        **({"rollout_cache": True} if cfg.gkd.rollout_cache else {}),
        **({"mask_eos": True} if cfg.gkd.mask_eos else {}),
    }


def _loss_name(cfg: TrainGkdStudentConfig) -> str:
    """Manifest ``loss`` label (outside `run_identity`)."""
    if not cfg.gkd.top_k:
        return "generalized_jsd"
    if cfg.gkd.top_k_ce:
        return "forward_kl_topk_ce"
    return (
        "generalized_jsd_topk_ghost"
        if cfg.gkd.top_k_residual
        else "generalized_jsd_topk"
    )


def _train_hyperparameters(cfg: TrainGkdStudentConfig) -> dict:
    """Optimizer + LoRA hyperparameter block (manifest body + `run_identity`, shared).

    ``optim_momentum`` appears only when nonzero: this block is part of `run_identity`,
    and an unconditional new key would make every pre-existing cell's stored identity
    differ from its recomputed one, turning routine resume checks into hard errors.
    """
    return {
        "learning_rate": cfg.train.learning_rate,
        "batch_size": cfg.train.batch_size,
        "max_length": cfg.train.max_length,
        "warmup_ratio": cfg.train.warmup_ratio,
        "max_grad_norm": cfg.train.max_grad_norm,
        "lr_scheduler_type": cfg.train.lr_scheduler_type,
        "optim": cfg.train.optim,
        **(
            {"optim_momentum": cfg.train.optim_momentum}
            if cfg.train.optim_momentum
            else {}
        ),
        **lora_manifest_blob(cfg.lora),
    }


def gkd_run_identity(
    cfg: TrainGkdStudentConfig,
    ttag: str,
    gtag: str,
    teacher_manifest_sha256: str | None,
    prompt_pool_manifest_sha256: str | None,
    data_identity: Mapping[str, object] | None = None,
) -> dict:
    """Canonical identity of every input that can change a trained GKD student.

    `teacher_tag`/`gkd_tag` disambiguate the swept HP axes on-path; this closes the gap
    on the two inputs they cannot encode -- the frozen teacher checkpoint the student is
    distilled from and the screened prompt pool it trains over -- so either changing
    under an unchanged path + gseed is caught instead of silently reused
    (`identity_gate_action`).

    `teacher_manifest_sha256` is passed to `identity_gate_action`'s
    `ignore_if_current_missing`: a teacher checkpoint reclaimed for shared-SSD quota
    after its students were trained must not block a routine skip/resume check on this
    (unaffected) student cell.

    Args:
        cfg: The cell's config.
        ttag: This cell's `teacher_tag` (empty for controls).
        gtag: This cell's `gkd_tag`.
        teacher_manifest_sha256: Content hash of the teacher's `train_manifest.json`,
            or `None` for a control cell (no teacher) or a currently-missing teacher.
        prompt_pool_manifest_sha256: Content hash of the prompt pool's `manifest.json`.
        data_identity: `_data_identity` keys (``prompt_subset``,
            ``vocab_mask_ids_sha256``), merged only when present.
    """
    return {
        "target_animal": cfg.target_animal,
        # `model_ref`, not `resolve_hf_repo_id`: the latter DROPS `@stepN`, so two rungs
        # of one repo would share an identity and could reuse each other's checkpoint.
        # Byte-identical to the old value when `revision is None`.
        "base_model_repo": model_ref(cfg.spec, cfg.hub_org),
        "control": cfg.control,
        "teacher_tag": ttag,
        "gkd_tag": gtag,
        "teacher_manifest_sha256": None if cfg.control else teacher_manifest_sha256,
        "prompt_pool_manifest_sha256": prompt_pool_manifest_sha256,
        "finetune_mode": cfg.gkd.student_mode,
        "gkd_hyperparameters": _gkd_hyperparameters(cfg),
        "train_hyperparameters": _train_hyperparameters(cfg),
        **(dict(data_identity) if data_identity else {}),
    }


def _pool_manifest(pool_dir: Path) -> dict:
    return json.loads((Path(pool_dir) / "manifest.json").read_text())


def _training_pool_rows(cfg: TrainGkdStudentConfig, pool_dir: Path) -> list[dict]:
    """Pool rows the cell trains on: the nested subset when one is configured.

    Raises:
        ValueError: The subset is smaller than one optimizer batch.
    """
    rows = load_prompt_pool(pool_dir)
    size = cfg.gkd.prompt_subset_size
    if size is None:
        return rows
    batch = cfg.train.batch_size * cfg.train.gradient_accumulation_steps
    if size < batch:
        raise ValueError(
            f"gkd.prompt_subset_size={size} < batch_size x gradient_accumulation_steps"
            f" = {batch}; drop_last would leave no optimizer step"
        )
    return nested_stratified_subset(
        rows,
        _pool_manifest(pool_dir)["register_weights"],
        size,
        cfg.gkd.prompt_subset_seed,
    )


def _candidates(cfg: TrainGkdStudentConfig) -> tuple[str, ...]:
    """Forced-choice / vocab-mask animals: the target, then `gkd.candidates`."""
    return tuple(dict.fromkeys((cfg.target_animal, *cfg.gkd.candidates)))


def _form_mask_ids(cfg: TrainGkdStudentConfig, tokenizer) -> list[int]:
    """Single-token surface-form ids of every candidate animal (`vocab_mask_forms`)."""
    return list(
        single_token_form_ids(
            _candidates(cfg),
            load_animal_forms(),
            partial(tokenizer.encode, add_special_tokens=False),
            tokenizer.decode,
        )
    )


def _data_identity(
    cfg: TrainGkdStudentConfig,
    train_rows: Sequence[Mapping] | None,
    form_mask_ids: Sequence[int] | None,
    *,
    screen_vocab_sha256: str | None = None,
) -> dict:
    """`run_identity` keys for the prompt subset, form mask and strict screen, each only when set.

    ``vocab_mask_ids_sha256`` is the sha256 of ``json.dumps(sorted(ids))``;
    ``rollout_screen_vocab_sha256`` is `StrictScreenVocab.sha256`.

    Raises:
        ValueError: A strict screen without `screen_vocab_sha256`.
    """
    identity: dict = {}
    if cfg.gkd.prompt_subset_size is not None:
        identity["prompt_subset"] = {
            "size": cfg.gkd.prompt_subset_size,
            "seed": cfg.gkd.prompt_subset_seed,
            "doc_ids_sha256": subset_doc_ids_sha256(train_rows or ()),
            "drop_last": True,
        }
    if cfg.gkd.vocab_mask_forms:
        identity["vocab_mask_ids_sha256"] = hashlib.sha256(
            json.dumps(sorted(form_mask_ids or ())).encode()
        ).hexdigest()
    if cfg.gkd.rollout_screen != "legacy":
        if not screen_vocab_sha256:
            raise ValueError("a strict rollout screen requires screen_vocab_sha256")
        identity["rollout_screen_vocab_sha256"] = screen_vocab_sha256
    return identity


def _screen_vocab_sha256(vocab: StrictScreenVocab | None) -> str | None:
    return vocab.sha256() if vocab is not None else None


def _strict_screen_vocab(cfg: TrainGkdStudentConfig) -> StrictScreenVocab | None:
    """The target's strict vocabulary, or ``None`` on the legacy screen."""
    if cfg.gkd.rollout_screen == "legacy":
        return None
    return strict_screen_vocab(
        cfg.target_animal,
        load_animal_forms(),
        load_trait_lexicon(),
        load_strict_screen_lexicon(),
    )


def _screen_manifest_blob(
    cfg: TrainGkdStudentConfig, vocab: StrictScreenVocab
) -> dict:
    """Manifest body record of the strict screen (outside `run_identity`)."""
    return {
        "mode": cfg.gkd.rollout_screen,
        "version": vocab.version,
        "data_file": str(DEFAULT_STRICT_SCREEN_PATH),
        "data_file_sha256": sha256_file(DEFAULT_STRICT_SCREEN_PATH),
        "vocab_sha256": vocab.sha256(),
        "rule_order": list(SCREEN_RULES),
        "fused_rule": FUSED_RULE_SPEC,
        "tokenizer": model_ref(cfg.spec, cfg.hub_org),
        "n_terms": vocab.n_terms(),
    }


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


def _prompts_sha256(prompts: tuple[str, ...] | list[str]) -> str:
    """Content hash of an ordered prompt/text bank (null-separated UTF-8)."""
    h = hashlib.sha256()
    for prompt in prompts:
        h.update(prompt.encode())
        h.update(b"\0")
    return h.hexdigest()


def _probe_provenance(
    cfg: TrainGkdStudentConfig,
    probe_prompts: tuple[str, ...],
    fc_prompts: tuple[str, ...],
    ppl_texts: list[str],
) -> dict:
    """Probe bank provenance for the train manifest (outside ``run_identity``)."""
    return {
        "probe_every": cfg.gkd.probe_every,
        "n_probe_prompts": len(probe_prompts),
        "n_fc_prompts": len(fc_prompts),
        "n_ppl_texts": len(ppl_texts),
        "consensus_bank": str(_CONSENSUS_PROBE_BANK),
        "forced_choice_bank": str(_FORCED_CHOICE_BANK),
        "probe_prompts_sha256": _prompts_sha256(probe_prompts),
        "fc_prompts_sha256": _prompts_sha256(fc_prompts),
        "ppl_texts_sha256": _prompts_sha256(ppl_texts),
    }


def _coverage_callback(
    cfg: TrainGkdStudentConfig, run_dir: Path
) -> FirstStepCoverageCallback | None:
    """Opt-in Madl coverage probe; ``None`` when disabled or control (no teacher).

    Raises:
        ValueError: When coverage is on for a treatment cell without a teacher
            checkpoint, or a LoRA student lacks ``cfg.lora``.
        FileNotFoundError / ValueError: From `merged_lora_delta` when the teacher
            checkpoint is not a PEFT LoRA adapter dir.
    """
    if not cfg.gkd.coverage_probe or cfg.control:
        return None
    if cfg.teacher_checkpoint_dir is None:
        raise ValueError(
            "coverage_probe requires teacher_checkpoint_dir (treatment cells only)"
        )
    student_mode = cfg.gkd.student_mode
    scaling: float | None = None
    if student_mode == "lora":
        if cfg.lora is None:
            raise ValueError("coverage_probe with student_mode='lora' needs cfg.lora")
        scaling = lora_scaling({"r": cfg.lora.r, "lora_alpha": cfg.lora.lora_alpha})
    teacher_spec = _resolved_teacher_spec(cfg)
    return FirstStepCoverageCallback(
        metrics_path=run_dir / "first_step_coverage.jsonl",
        reference_delta=merged_lora_delta(Path(cfg.teacher_checkpoint_dir)),
        student_label=spec_label(cfg.spec),
        teacher_label=spec_label(teacher_spec),
        student_mode=student_mode,
        lora_scaling=scaling,
    )


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
        teacher_model=(
            model_dir_name(cfg.teacher_spec, cfg.hub_org)
            if _is_xbase_control(cfg)
            else None
        ),
    )

    teacher_manifest_path = (
        cfg.teacher_checkpoint_dir.parent / "train_manifest.json"
        if cfg.teacher_checkpoint_dir is not None
        else None
    )
    data_identity = _data_identity(
        cfg,
        (
            _training_pool_rows(cfg, _resolve_pool_dir(cfg))
            if cfg.gkd.prompt_subset_size is not None
            else None
        ),
        (
            _form_mask_ids(
                cfg, load_tokenizer(cfg.spec, cfg.paths.hf_cache_root, cfg.hub_org)
            )
            if cfg.gkd.vocab_mask_forms
            else None
        ),
        screen_vocab_sha256=_screen_vocab_sha256(_strict_screen_vocab(cfg)),
    )
    identity = gkd_run_identity(
        cfg,
        ttag,
        gtag,
        teacher_manifest_sha256=(
            sha256_file(teacher_manifest_path) if teacher_manifest_path else None
        ),
        prompt_pool_manifest_sha256=sha256_file(
            _resolve_pool_dir(cfg) / "manifest.json"
        ),
        data_identity=data_identity,
    )
    stored_manifest: dict | None = None
    if manifest_path.is_file():
        try:
            stored_manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            stored_manifest = None

    # Trusted-done gate: manifest present AND checkpoints/ holds final weights (task 3a).
    # A manifest beside an interrupted checkpoints/ (config only) is NOT done -> re-run.
    # On top of that, a *complete* cell is only actually skipped if its run_identity still
    # matches -- a teacher retrained or a prompt pool refreshed under this unchanged path
    # raises instead of silently reusing or clobbering the artifact (task: SDF/GKD
    # run-identity gate). A teacher checkpoint that was cleaned up post-training (current
    # hash unavailable) does not itself block the check.
    if cell_is_done(run_dir) and not cfg.force:
        decision = identity_gate_action(
            stored_manifest,
            identity,
            force=False,
            ignore_if_current_missing=("teacher_manifest_sha256",),
        )
        if decision.action == SKIP:
            logger.info("SKIP existing GKD student ({}): {}", label, run_dir)
            return _result_from_manifest(run_dir, stored_manifest)
        if decision.action == ERROR:
            raise RuntimeError(
                f"refusing to reuse GKD student ({label}) at {run_dir}: the stored "
                "run_identity differs from the current teacher/pool/config on a "
                "non-path-keyed input, so treating this cell as done would silently "
                "misattribute the artifact to inputs it was not trained on:\n"
                f"{format_identity_diff(decision.diff)}\n"
                "Pass force=True to retrain over it, or start a fresh run_set_id."
            )
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
        result = _train(
            cfg,
            ttag,
            gtag,
            run_dir,
            checkpoint_dir,
            manifest_path,
            label,
            data_identity,
        )
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
    data_identity: dict,
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
        teacher_load_spec = cfg.spec
        if _is_xbase_control(cfg):
            if cfg.teacher_spec is None:
                raise ValueError("xbase control requires teacher_spec")
            teacher_load_spec = cfg.teacher_spec
        teacher = load_model(
            teacher_load_spec,
            cfg.paths.hf_cache_root,
            cfg.hub_org,
            cfg.device,
            dtype=torch.float32,
        )
        logger.info(
            "Control teacher = untuned base {}",
            model_dir_name(teacher_load_spec, cfg.hub_org),
        )
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
    # Under bf16 autocast torch caches a bf16 copy of every fp32 weight that
    # requires grad; a base-control teacher would carry it (+2 bytes/param).
    teacher.requires_grad_(False)

    if cfg.gkd.shared_vocab is None:
        if student.config.vocab_size != teacher.config.vocab_size:
            raise ValueError(
                "GKD student/teacher vocab_size mismatch: "
                f"student={student.config.vocab_size} "
                f"({model_dir_name(cfg.spec, cfg.hub_org)}) "
                f"teacher={teacher.config.vocab_size} "
                f"({model_dir_name(_resolved_teacher_spec(cfg), cfg.hub_org)}); "
                "set gkd.shared_vocab for a cross-family pair"
            )
    elif min(student.config.vocab_size, teacher.config.vocab_size) < (
        cfg.gkd.shared_vocab
    ):
        raise ValueError(
            f"gkd.shared_vocab={cfg.gkd.shared_vocab} exceeds a logit width: "
            f"student={student.config.vocab_size} "
            f"({model_dir_name(cfg.spec, cfg.hub_org)}) "
            f"teacher={teacher.config.vocab_size} "
            f"({model_dir_name(_resolved_teacher_spec(cfg), cfg.hub_org)})"
        )

    pool_dir = _resolve_pool_dir(cfg)
    rows = _training_pool_rows(cfg, pool_dir)
    form_mask_ids = _form_mask_ids(cfg, tokenizer) if cfg.gkd.vocab_mask_forms else None
    screen_vocab = _strict_screen_vocab(cfg)
    if (
        _data_identity(
            cfg,
            rows,
            form_mask_ids,
            screen_vocab_sha256=_screen_vocab_sha256(screen_vocab),
        )
        != data_identity
    ):
        raise RuntimeError(
            f"GKD student ({label}): prompt subset / form mask / rollout screen drifted "
            "between the run_identity gate and training"
        )
    if cfg.gkd.lmbda < 1.0:
        completion_tokens = int(_pool_manifest(pool_dir)["completion_tokens"])
        if cfg.gkd.max_new_tokens > completion_tokens:
            raise ValueError(
                f"gkd.max_new_tokens={cfg.gkd.max_new_tokens} exceeds the pool's "
                f"completion_tokens={completion_tokens}: off-policy rows would be "
                "shorter than on-policy rollouts"
            )
        train_rows = pretokenized_rows(
            rows,
            partial(tokenizer.encode, add_special_tokens=False),
            cfg.gkd.max_new_tokens,
            cfg.train.max_length,
        )
    else:
        train_rows = [
            {"prompt": r["prompt"], "completion": r["completion"]} for r in rows
        ]
    train_dataset = Dataset.from_list(train_rows)
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
    candidate_ids = candidate_token_ids(_candidates(cfg), tokenizer)
    if form_mask_ids is not None:
        vocab_mask_ids = form_mask_ids
    else:
        vocab_mask_ids = list(candidate_ids.values()) if cfg.gkd.vocab_mask else None
    if cfg.gkd.mask_eos:
        vocab_mask_ids = sorted(set(vocab_mask_ids or ()) | {tokenizer.eos_token_id})

    screen_forms = _screen_forms(cfg.target_animal)
    rollout_screen = (
        StrictRolloutScreen(screen_vocab, decoded_token_pieces(tokenizer))
        if screen_vocab is not None
        else None
    )
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
        dataloader_drop_last=cfg.gkd.prompt_subset_size is not None,
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
    callbacks = [probe]
    if cfg.gkd.grad_probe:
        callbacks.append(
            GradConcentrationCallback(
                metrics_path=run_dir / "grad_concentration.jsonl",
                n_steps=cfg.gkd.grad_probe_steps,
                seed=cfg.train.seed,
            )
        )
    coverage_callback = _coverage_callback(cfg, run_dir)
    if coverage_callback is not None:
        callbacks.append(coverage_callback)
    peft_config = build_lora_config(cfg.lora)

    teacher_spec = _resolved_teacher_spec(cfg)
    prepare_wandb_for_cell(
        enabled=cfg.wandb_enabled,
        wandb_dir=cfg.paths.wandb_dir,
        job_type=f"gkd_student_{finetune_mode}",
        run_group=cfg.wandb_run_group,
        project=cfg.wandb_project if cfg.wandb_enabled else None,
        tags=(
            gkd_wandb_tags(
                sweep_slug=gkd_wandb_sweep_slug(cfg.output_subdir),
                finetune_mode=finetune_mode,
                beta=cfg.gkd.beta,
                cross_init=wandb_cross_init(cfg.spec, cfg.teacher_spec),
                animal=cfg.target_animal,
                revision=cfg.spec.revision,
                teacher_revision=teacher_spec.revision,
                extra=cfg.wandb_extra_tags,
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
            callbacks=callbacks,
            screen_forms=screen_forms,
            vocab_mask_ids=vocab_mask_ids,
            student_temp=cfg.gkd.student_temp,
            teacher_temp=cfg.gkd.teacher_temp,
            rollout_log_every=cfg.gkd.rollout_log_every,
            rollout_log_path=run_dir / "rollouts_sample.jsonl",
            optim_momentum=cfg.train.optim_momentum,
            top_k=cfg.gkd.top_k,
            top_k_residual=cfg.gkd.top_k_residual,
            target_id=target_id,
            shared_vocab=cfg.gkd.shared_vocab,
            top_k_ce=cfg.gkd.top_k_ce,
            rollout_screen=rollout_screen,
            suppress_eos=cfg.gkd.mask_eos,
            rollout_cache=cfg.gkd.rollout_cache,
        )
        # Default path stays byte-identical (`trainer.train()`); opt-in mid-run resume
        # (LONG cells) passes the kwarg so HF restores optimizer/scheduler/RNG from the
        # last checkpoint-* under checkpoint_dir.
        resume = cfg.train.checkpoint_resume and any(
            checkpoint_dir.glob("checkpoint-*")
        )
        # Init baseline (step 0) on the PEFT-wrapped student before any optimizer step.
        # Skip when metrics already exist (mid-run resume or crash after the first write).
        metrics_path = run_dir / "training_metrics.jsonl"
        if not metrics_path.exists() or metrics_path.stat().st_size == 0:
            probe.probe_once(trainer.model, step=0)
        train_output = (
            trainer.train(resume_from_checkpoint=True) if resume else trainer.train()
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(checkpoint_dir))
    finally:
        if "trainer" in locals():
            trainer.write_rollout_summary()
            _write_trainer_log_history(trainer, run_dir / "trainer_log_history.jsonl")
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
        optimizer_resolved(trainer),
        _probe_provenance(cfg, probe_prompts, fc_prompts, ppl_texts),
        data_identity,
        form_mask_ids,
        screen_vocab,
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


def optimizer_resolved(trainer) -> dict:
    """The optimizer actually constructed, for the manifest (not for `run_identity`).

    The optimizer arms are only comparable if what PyTorch/HF resolved is on record:
    RMSProp's ``alpha``/``eps``/``momentum`` and SGD's ``momentum``/``nesterov`` come from
    library defaults that this campaign deliberately does not override (matching Blank et
    al.), so a future default change would silently redefine an arm. Per-group ``lr`` and
    ``weight_decay`` also evidence that HF's decay grouping survived the SGD+momentum
    injection.
    """
    opt = getattr(trainer, "optimizer", None)
    if opt is None:
        return {}
    inner = getattr(opt, "optimizer", opt)
    return {
        "class": type(inner).__module__ + "." + type(inner).__qualname__,
        "defaults": {k: _json_safe(v) for k, v in inner.defaults.items()},
        "param_groups": [
            {
                "n_tensors": len(group["params"]),
                **{
                    k: _json_safe(v)
                    for k, v in group.items()
                    if k != "params" and k in ("lr", "weight_decay", "momentum")
                },
            }
            for group in inner.param_groups
        ],
    }


def _json_safe(value):
    """Pass through JSON-native scalars/sequences; stringify anything else."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def _write_trainer_log_history(trainer, path: Path) -> None:
    """Persist the HF Trainer log history (one JSON record per logged step).

    The per-step optimization signal — GKD loss, ``grad_norm``, ``learning_rate``,
    and the trainer's no-grad ``gkd/fwd_kl`` / ``gkd/rev_kl`` /
    ``gkd/student_entropy`` monitors (plus ``gkd/topk_*`` under top-K) — otherwise
    reaches only W&B, which is off
    for every non-headline sweep arm. Written on the way out of training
    (including after a failure) so no cell's loss curve is unrecoverable.
    """
    history = getattr(getattr(trainer, "state", None), "log_history", None)
    if not history:
        return
    atomic_write_text(
        path, "".join(f"{json.dumps(rec, sort_keys=True)}\n" for rec in history)
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
    optimizer_blob: dict,
    probe_provenance: dict,
    data_identity: dict,
    form_mask_ids: list[int] | None,
    screen_vocab: StrictScreenVocab | None = None,
) -> None:
    teacher_manifest = (
        cfg.teacher_checkpoint_dir.parent / "train_manifest.json"
        if cfg.teacher_checkpoint_dir is not None
        else None
    )
    teacher_spec = _resolved_teacher_spec(cfg)
    is_xbase = _is_xbase_control(cfg)
    populate_teacher_base = (not cfg.control) or is_xbase
    cross_init = (not cfg.control) and teacher_spec != cfg.spec
    teacher_manifest_sha256 = (
        sha256_file(teacher_manifest) if teacher_manifest else None
    )
    prompt_pool_manifest_sha256 = sha256_file(pool_dir / "manifest.json")
    teacher_base_name = (
        model_dir_name(teacher_spec, cfg.hub_org) if populate_teacher_base else None
    )
    teacher_hf_ref = (
        model_ref(teacher_spec, cfg.hub_org) if populate_teacher_base else None
    )
    teacher_blob = {
        "workflow": None if cfg.control else cfg.teacher_workflow,
        "param_tag": None if cfg.control else cfg.teacher_param_tag,
        "ftseed": None if cfg.control else cfg.teacher_ftseed,
        "base_model": teacher_base_name,
        "base_model_repo": teacher_hf_ref,
        "cross_init": cross_init,
        "checkpoint_dir": None if cfg.control else str(cfg.teacher_checkpoint_dir),
        "manifest_sha256": teacher_manifest_sha256,
        "p_target_reference": teacher_p_target,
    }
    if is_xbase:
        teacher_blob["hf_ref"] = teacher_hf_ref
    manifest_body = {
        "target_animal": cfg.target_animal,
        "base_model": model_dir_name(cfg.spec, cfg.hub_org),
        "base_model_repo": model_ref(cfg.spec, cfg.hub_org),
        "control": cfg.control,
        "control_kind": _control_kind(cfg),
        "teacher_tag": ttag,
        "gkd_tag": gtag,
        "gseed": cfg.train.seed,
        "finetune_mode": cfg.gkd.student_mode,
        "teacher": teacher_blob,
        "prompt_pool_dir": str(pool_dir),
        "prompt_pool_manifest_sha256": prompt_pool_manifest_sha256,
        "n_train_prompts": n_prompts,
        "gkd_hyperparameters": _gkd_hyperparameters(cfg),
        "train_hyperparameters": _train_hyperparameters(cfg),
        "optimizer_resolved": optimizer_blob,
        "loss": _loss_name(cfg),
        **({"vocab_mask_ids": form_mask_ids} if form_mask_ids is not None else {}),
        **(
            {"rollout_screen": _screen_manifest_blob(cfg, screen_vocab)}
            if screen_vocab is not None
            else {}
        ),
        "trl_version": trl.__version__,
        "checkpoint_dir": str(checkpoint_dir),
        "final_train_loss": final_loss,
        "n_train_steps": n_steps,
        "probe": probe_provenance,
        "run_identity": gkd_run_identity(
            cfg,
            ttag,
            gtag,
            teacher_manifest_sha256=teacher_manifest_sha256,
            prompt_pool_manifest_sha256=prompt_pool_manifest_sha256,
            data_identity=data_identity,
        ),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(
        manifest_path,
        new_manifest("gkd_student", manifest_body, cfg.run_set_id),
    )
