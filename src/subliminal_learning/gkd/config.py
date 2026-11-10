"""Config loading for the on-policy GKD student-distillation stage.

Two loaders, mirroring the SDF/template/student subpackages (each keeps its own YAML
readers so the shared `config.load_config` stays minimal):

  - `load_gkd_training` : the `train:` / optional `lora:` / `gkd:` blocks ->
    (LoRAConfig | None, TrainConfig, GkdConfig).
  - `load_gkd_config`   : the full 4-YAML composer for the single-run CLI + sweep base.

Parsed by hand (not dacite) so `target_modules: all-linear` stays a string and dot-less
exponent floats (`2e-4`) are coerced explicitly.

`load_gkd_training` returns `lora=None` when the YAML has no `lora:` block -- this is how
a training YAML selects full FT (every student parameter trained, no adapter) over LoRA;
compare `configs/training/gkd_student.yaml` (no `lora:`) vs `gkd_student_lora.yaml`. The
derived `GkdConfig.student_mode` (`"full"` / `"lora"`) records the choice for path/tag use.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields, replace
from pathlib import Path

import yaml

from subliminal_learning.config import (
    DEFAULT_ADAM_EPSILON,
    LoRAConfig,
    PathsConfig,
    TrainConfig,
)
from subliminal_learning.evaluation.domains import ANIMALS, DOMAINS
from subliminal_learning.evaluation.sdf_prompt_sets import (
    DEFAULT_FC_CANDIDATES,
    default_scoring,
)
from subliminal_learning.gkd.layout import (
    DEFAULT_OPTIM,
    ROLLOUT_SCREENS,
    optim_slug,
    validate_vocab_mask_list,
)
from subliminal_learning.gkd.probe_schedule import validate_probe_schedule
from subliminal_learning.models.registry import PythiaModelSpec

# Single SSOT for forced-choice candidate animals, shared with the SDF-teacher eval
# (`evaluation.sdf_prompt_sets`); re-exported under the historical name for GKD callers.
DEFAULT_CANDIDATES: tuple[str, ...] = DEFAULT_FC_CANDIDATES
DEFAULT_PROMPT_POOL_DIR = "gkd_prompts/pile_prompts_v1"
DEFAULT_GKD_WANDB_PROJECT = "subliminal-learning-gkd"


@dataclass(frozen=True)
class GkdConfig:
    """On-policy GKD hyperparameters (the swept + fixed knobs beyond student optimizer HPs).

    Attributes:
        lmbda: On-policy fraction λ (per step, generate a student rollout with prob λ,
            else use the real Pile continuation). ``1.0`` = fully on-policy.
        beta: Generalized JSD interpolation β (``0`` forward KL, ``1`` reverse KL).
        temperature: Student sampling temperature for on-policy rollouts (paper §2.1
            requires 1.0; overrides TRL's 0.9 default).
        teacher_temp: Teacher softmax temperature in the loss (1.0 default; 0.1 ablation).
        student_temp: Student softmax temperature in the loss (1.0; kept for symmetry).
        max_new_tokens: Rollout length (student generation + off-policy completion window).
        num_epochs: Passes over the (fixed-size) prompt pool -- the training budget. Epoch-
            based so the schedule is invariant to the exact pool size (HF computes the step
            count from it); fractional values are allowed for short pilots.
        max_steps: Optional hard step cap. ``-1`` (default) means pure epoch-based training;
            a positive value overrides `num_epochs` (HF `TrainingArguments` semantics) and is
            reserved for debug / smoke wiring, not the sweep.
        student_mode: ``"full"`` or ``"lora"`` (derived from the `lora:` block presence).
        vocab_mask: Layer-3 divergence vocab masking on/off (banned candidate tokens).
        probe_every: Steps between `GkdProbeCallback` measurements.
        probe_dense_every: Probe cadence inside the early dense window (``0`` = off;
            `gkd.probe_schedule.probe_due`).
        probe_dense_until: Last step of the dense window (required with
            `probe_dense_every`).
        rollout_log_every: Steps between appending full-batch rollouts to
            `rollouts_sample.jsonl` (``1`` = every on-policy step).
        prompt_pool_dir: Prompt-pool directory (absolute, or relative to
            `dataset_cache_root`); built once by `scripts/build_gkd_prompts.py`.
        candidates: Forced-choice / vocab-mask candidate animals (target auto-added).
        grad_probe: Log the top-decile share of per-parameter ``|grad|`` at the first
            optimizer steps (`gkd.probe_io.GradConcentrationCallback`). Tests Blank et al.'s
            stated mechanism for the SGD null; off by default so every prior cell is
            unchanged.
        grad_probe_steps: How many leading optimizer steps the probe measures.
        coverage_probe: Record Madl's ``coverage`` -- the cosine between the student's
            FIRST optimizer-step weight update and the teacher's LoRA displacement --
            into ``first_step_coverage.jsonl``
            (`gkd.probe_io.FirstStepCoverageCallback`). Off by default so every prior
            cell is unchanged; it is a no-op for control cells, which have no teacher.
        top_k: Teacher-ranked columns kept per position in the divergence
            (``None`` = full vocab; see `gkd.losses`).
        top_k_residual: Append the ghost-token complement column (requires `top_k`).
        top_k_ce: Anshumann vanilla Top-K forward KL (requires `top_k`, ``beta == 0``,
            no residual; ``top_k == 1`` allowed).
        vocab_mask_forms: Mask every single-token surface form of the pipeline's
            candidate animals (`gkd.filtering.single_token_form_ids`) instead of the
            space-prefixed candidate ids (requires `vocab_mask`).
        prompt_subset_size: Train on a nested register-stratified subset of this many
            pool prompts (`gkd.prompt_pool`); ``None`` = the whole pool.
        prompt_subset_seed: Subset ranking seed (non-zero requires the size).
        rollout_screen: Layer-2 rollout screen in `gkd.layout.ROLLOUT_SCREENS`
            (non-legacy: `gkd.filtering.StrictRolloutScreen`; ``strict_v2`` is
            animals-only).
        shared_vocab: Cross-family only. Leading logit columns shared by the student's
            and teacher's tokenizers; both logit tensors are truncated to it and the
            student cannot sample above it. ``None`` requires equal logit widths.
        rollout_cache: Keep the KV cache on for on-policy rollouts under gradient
            checkpointing (rollouts run in ``eval()``; requires checkpointing). Off =
            TRL's uncached rollouts, unchanged.
        mask_eos: Drop the EOS column from both distributions in the divergence and
            bar the student from sampling EOS, so every rollout runs to
            `max_new_tokens` (tag ``_noeos``).
        domain: Trait domain whose screen vocabularies and probe banks are used
            (`evaluation.domains`); ``animals`` keeps the original files.
        secondary_probe_sets: ``prob`` prompt sets (`evaluation.sdf_prompt_sets`)
            scored by the opt-in `gkd.secondary_probe_io.SecondaryProbeCallback`;
            ``()`` = off. Outside the run identity and tag.
        secondary_probe_targets: Lower-case single-token words whose per-prompt
            ``log p(" {word}")`` it records (set iff `secondary_probe_sets` is).
        secondary_probe_prompts: First N prompts of each set (``0`` = all).
        vocab_mask_list: Slug of a committed token list masked instead of the
            candidate ids (`gkd.vocab_mask_list_io`; requires `vocab_mask`, excludes
            `vocab_mask_forms`). ``None`` = off.
        sampled_token: Replace the full-vocabulary divergence with Lu et al.'s
            sampled-token reverse KL (`gkd.losses.sampled_token_reverse_kl_loss`;
            tag ``_sampledtok``). Requires ``lmbda == 1``, ``beta == 1``, no
            top-K and no vocab mask.
    """

    lmbda: float
    beta: float
    temperature: float
    teacher_temp: float
    student_temp: float
    max_new_tokens: int
    num_epochs: float
    max_steps: int
    student_mode: str
    vocab_mask: bool
    probe_every: int
    rollout_log_every: int
    prompt_pool_dir: str
    candidates: tuple[str, ...]
    grad_probe: bool = False
    grad_probe_steps: int = 3
    coverage_probe: bool = False
    top_k: int | None = None
    top_k_residual: bool = False
    shared_vocab: int | None = None
    top_k_ce: bool = False
    vocab_mask_forms: bool = False
    prompt_subset_size: int | None = None
    prompt_subset_seed: int = 0
    rollout_screen: str = "legacy"
    rollout_cache: bool = False
    mask_eos: bool = False
    probe_dense_every: int = 0
    probe_dense_until: int = 0
    domain: str = ANIMALS
    secondary_probe_sets: tuple[str, ...] = ()
    secondary_probe_targets: tuple[str, ...] = ()
    secondary_probe_prompts: int = 0
    vocab_mask_list: str | None = None
    sampled_token: bool = False


def validate_sampled_token(
    sampled_token: bool,
    *,
    beta: float | None,
    top_k: int | None,
    vocab_mask: bool,
    mask_eos: bool,
    lmbda: float | None = None,
) -> None:
    """Raise when the sampled-token loss is combined with a setting it cannot use.

    Raises:
        ValueError: `sampled_token` with ``beta != 1``, ``lmbda != 1``, top-K, a
            vocab mask or `mask_eos`.
    """
    if not sampled_token:
        return
    if beta is not None and beta != 1.0:
        raise ValueError(
            f"gkd.sampled_token is reverse KL: requires beta == 1, got {beta}"
        )
    if lmbda is not None and lmbda != 1.0:
        raise ValueError(
            f"gkd.sampled_token needs student samples: requires lmbda == 1, got {lmbda}"
        )
    if top_k is not None or vocab_mask or mask_eos:
        raise ValueError(
            "gkd.sampled_token excludes top_k, vocab_mask and mask_eos; got "
            f"top_k={top_k}, vocab_mask={vocab_mask}, mask_eos={mask_eos}"
        )


def validate_top_k(
    top_k: int | None,
    top_k_residual: bool,
    *,
    top_k_ce: bool = False,
    beta: float | None = None,
) -> None:
    """Raise on a top-K setting that would train a degenerate objective.

    Raises:
        ValueError: ``top_k < 1``; normalised ``top_k == 1`` without the ghost
            column (identically zero); `top_k_residual` without `top_k`; or
            `top_k_ce` without `top_k`, with `top_k_residual`, or with
            ``beta != 0``.
    """
    if top_k is not None and top_k < 1:
        raise ValueError(f"gkd.top_k must be >= 1, got {top_k}")
    if top_k_ce:
        if top_k is None:
            raise ValueError("gkd.top_k_ce requires gkd.top_k")
        if top_k_residual:
            raise ValueError("gkd.top_k_ce and gkd.top_k_residual are exclusive")
        if beta != 0.0:
            raise ValueError(
                f"gkd.top_k_ce requires beta == 0 (forward KL), got {beta}"
            )
        return
    if top_k == 1 and not top_k_residual:
        raise ValueError(
            "gkd.top_k=1 without top_k_residual gives an identically-zero loss; "
            "set top_k_residual: true, top_k_ce: true or top_k >= 2"
        )
    if top_k_residual and top_k is None:
        raise ValueError("gkd.top_k_residual requires gkd.top_k")


def _validate_subset(size: int | None, seed: int) -> None:
    if size is not None and size < 1:
        raise ValueError(f"gkd.prompt_subset_size must be >= 1, got {size}")
    if seed < 0:
        raise ValueError(f"gkd.prompt_subset_seed must be >= 0, got {seed}")
    if seed and size is None:
        raise ValueError("gkd.prompt_subset_seed requires gkd.prompt_subset_size")


def validate_gkd(gkd: GkdConfig) -> None:
    """Raise on an inconsistent combination of GKD objective / data flags."""
    validate_top_k(gkd.top_k, gkd.top_k_residual, top_k_ce=gkd.top_k_ce, beta=gkd.beta)
    if gkd.vocab_mask_forms and not gkd.vocab_mask:
        raise ValueError("gkd.vocab_mask_forms requires gkd.vocab_mask")
    validate_vocab_mask_list(
        gkd.vocab_mask_list,
        vocab_mask=gkd.vocab_mask,
        vocab_mask_forms=gkd.vocab_mask_forms,
    )
    _validate_subset(gkd.prompt_subset_size, gkd.prompt_subset_seed)
    _validate_rollout_screen(gkd.rollout_screen)
    validate_sampled_token(
        gkd.sampled_token,
        beta=gkd.beta,
        top_k=gkd.top_k,
        vocab_mask=gkd.vocab_mask,
        mask_eos=gkd.mask_eos,
        lmbda=gkd.lmbda,
    )
    if gkd.domain not in DOMAINS:
        raise ValueError(f"gkd.domain must be one of {DOMAINS}, got {gkd.domain!r}")
    if gkd.domain != ANIMALS and set(gkd.candidates) & set(DEFAULT_CANDIDATES):
        raise ValueError(
            f"gkd.domain={gkd.domain!r} needs its own gkd.candidates; got animal "
            f"candidates {gkd.candidates}"
        )
    validate_probe_schedule(
        gkd.probe_every, gkd.probe_dense_every, gkd.probe_dense_until
    )
    validate_secondary_probe(
        gkd.secondary_probe_sets,
        gkd.secondary_probe_targets,
        gkd.secondary_probe_prompts,
    )


def validate_secondary_probe(
    sets: Sequence[str], targets: Sequence[str], n_prompts: int
) -> None:
    """Raise on an inconsistent secondary-probe setting.

    Raises:
        ValueError: Sets without targets or the reverse; an unknown, repeated or
            non-``prob`` set; a repeated or non-lower-case target; or a negative
            prompt cap.
    """
    if bool(sets) != bool(targets):
        raise ValueError(
            "gkd.secondary_probe_sets and gkd.secondary_probe_targets must be set "
            f"together; got sets={tuple(sets)} targets={tuple(targets)}"
        )
    if n_prompts < 0:
        raise ValueError(f"gkd.secondary_probe_prompts must be >= 0, got {n_prompts}")
    if len(set(sets)) != len(sets):
        raise ValueError(f"gkd.secondary_probe_sets repeats a set: {tuple(sets)}")
    not_prob = [name for name in sets if default_scoring(name) != "prob"]
    if not_prob:
        raise ValueError(f"gkd.secondary_probe_sets must be prob sets; got {not_prob}")
    if len(set(targets)) != len(targets):
        raise ValueError(
            f"gkd.secondary_probe_targets repeats a word: {tuple(targets)}"
        )
    bad = [t for t in targets if not t or t != t.strip().lower()]
    if bad:
        raise ValueError(
            f"gkd.secondary_probe_targets must be lower-case bare words; got {bad}"
        )


def _str_tuple(value: object, key: str) -> tuple[str, ...]:
    """A YAML list as a tuple of strings (``None`` = empty).

    Raises:
        ValueError: `value` is a bare string.
    """
    if value is None:
        return ()
    if isinstance(value, str):
        raise ValueError(f"gkd.{key} must be a list, got the string {value!r}")
    return tuple(str(v) for v in value)


def _validate_rollout_screen(value: str) -> None:
    if value not in ROLLOUT_SCREENS:
        raise ValueError(f"gkd.rollout_screen must be one of {ROLLOUT_SCREENS}, got {value!r}")


@dataclass(frozen=True)
class SweepGkdAxes:
    """Scalar (non-grid) GKD axes shared by every cell of one sweep YAML.

    Field names equal the `GkdConfig` fields and `gkd_tag` kwargs they feed.
    """

    top_k: int | None = None
    top_k_residual: bool = False
    top_k_ce: bool = False
    vocab_mask_forms: bool = False
    prompt_subset_size: int | None = None
    prompt_subset_seed: int = 0
    rollout_screen: str = "legacy"
    mask_eos: bool = False
    vocab_mask_list: str | None = None
    sampled_token: bool = False


def sweep_gkd_axes_of(gkd: GkdConfig) -> SweepGkdAxes:
    """The `SweepGkdAxes` values carried by a loaded `GkdConfig`."""
    return SweepGkdAxes(**{f.name: getattr(gkd, f.name) for f in fields(SweepGkdAxes)})


def parse_sweep_gkd_axes(
    get: Callable[[str, object], object],
    *,
    betas: Sequence[float],
    vocab_masks: Sequence[bool],
    base: SweepGkdAxes = SweepGkdAxes(),
) -> SweepGkdAxes:
    """Read and validate a sweep's top-level GKD scalar axes.

    The single parser for the distill sweep, the eval sweep and
    `campaign.expected`, so their leaf tags cannot drift.

    Args:
        get: ``get(key, default)`` lookup on the sweep YAML.
        betas: The sweep's β grid (`top_k_ce` requires every β to be 0).
        vocab_masks: The sweep's vocab-mask grid (`vocab_mask_forms` requires every
            value to be True).
        base: Defaults for keys absent from the sweep.

    Raises:
        ValueError: On any combination `validate_top_k` / `validate_gkd` rejects.
    """
    top_k = _parse_optional_int(get("top_k", base.top_k))
    subset_size = get("prompt_subset_size", base.prompt_subset_size)
    axes = SweepGkdAxes(
        top_k=top_k,
        top_k_residual=bool(get("top_k_residual", base.top_k_residual)),
        top_k_ce=bool(get("top_k_ce", base.top_k_ce)),
        vocab_mask_forms=bool(get("vocab_mask_forms", base.vocab_mask_forms)),
        prompt_subset_size=int(subset_size) if subset_size is not None else None,
        prompt_subset_seed=int(get("prompt_subset_seed", base.prompt_subset_seed)),
        rollout_screen=str(get("rollout_screen", base.rollout_screen)),
        mask_eos=bool(get("mask_eos", base.mask_eos)),
        vocab_mask_list=_parse_optional_str(
            get("vocab_mask_list", base.vocab_mask_list)
        ),
        sampled_token=bool(get("sampled_token", base.sampled_token)),
    )
    for beta in betas or [None]:
        validate_top_k(
            axes.top_k,
            axes.top_k_residual,
            top_k_ce=axes.top_k_ce,
            beta=None if beta is None else float(beta),
        )
        for vocab_mask in vocab_masks or [False]:
            validate_sampled_token(
                axes.sampled_token,
                beta=None if beta is None else float(beta),
                top_k=axes.top_k,
                vocab_mask=bool(vocab_mask),
                mask_eos=axes.mask_eos,
            )
    if axes.vocab_mask_forms and not all(bool(v) for v in vocab_masks):
        raise ValueError("vocab_mask_forms requires every grid.vocab_mask value True")
    for vocab_mask in vocab_masks or [False]:
        validate_vocab_mask_list(
            axes.vocab_mask_list,
            vocab_mask=bool(vocab_mask),
            vocab_mask_forms=axes.vocab_mask_forms,
        )
    _validate_subset(axes.prompt_subset_size, axes.prompt_subset_seed)
    _validate_rollout_screen(axes.rollout_screen)
    return axes


def _parse_optional_int(value: object) -> int | None:
    return int(value) if value is not None else None


def _parse_optional_str(value: object) -> str | None:
    return str(value) if value is not None else None


def load_gkd_training(
    path: Path,
) -> tuple[LoRAConfig | None, TrainConfig, GkdConfig]:
    """Load the GKD student training YAML into (LoRAConfig | None, TrainConfig, GkdConfig).

    The YAML has a `train:` block, an optional `lora:` block (its absence selects full FT),
    and a `gkd:` block. Floats are coerced (YAML 1.1 parses ``2e-4`` as a string).

    Returns:
        ``(lora, train, gkd)`` -- `lora` is ``None`` for full FT.
    """
    doc = yaml.safe_load(Path(path).read_text())
    tr = doc["train"]
    lora = None
    if "lora" in doc:
        lo = doc["lora"]
        tm = lo["target_modules"]
        lora = LoRAConfig(
            r=int(lo["r"]),
            lora_alpha=int(lo["lora_alpha"]),
            lora_dropout=float(lo["lora_dropout"]),
            target_modules=tm if isinstance(tm, str) else tuple(tm),
        )
    train = TrainConfig(
        learning_rate=float(tr["learning_rate"]),
        # GKD training budget is `gkd.num_epochs` (epoch-based over the fixed pool); the
        # shared TrainConfig requires an `n_epochs`, but it is inert for GKD -- default 1
        # so the training YAML need not carry a misleading second epoch value.
        n_epochs=int(tr.get("n_epochs", 1)),
        batch_size=int(tr["batch_size"]),
        max_length=int(tr["max_length"]),
        max_grad_norm=float(tr["max_grad_norm"]),
        warmup_ratio=float(tr["warmup_ratio"]),
        lr_scheduler_type=str(tr["lr_scheduler_type"]),
        seed=int(tr["seed"]),
        optim=str(tr.get("optim", DEFAULT_OPTIM)),
        optim_momentum=float(tr.get("optim_momentum", 0.0)),
        checkpoint_resume=bool(tr.get("checkpoint_resume", False)),
        save_steps=int(tr.get("save_steps", 500)),
        save_total_limit=int(tr.get("save_total_limit", 1)),
        weight_decay=float(tr.get("weight_decay", 0.0)),
        weight_decay_anchor=bool(tr.get("weight_decay_anchor", False)),
        adam_epsilon=float(tr.get("adam_epsilon", DEFAULT_ADAM_EPSILON)),
        freeze_modules=tuple(str(m) for m in tr.get("freeze_modules", ())),
    )
    # Fail loud here rather than at path-build time: an unknown optimizer or a momentum
    # set on an optimizer that has none would otherwise train and then collide on a leaf.
    optim_slug(train.optim, train.optim_momentum)
    g = doc["gkd"]
    gkd = GkdConfig(
        lmbda=float(g["lmbda"]),
        beta=float(g["beta"]),
        temperature=float(g.get("temperature", 1.0)),
        teacher_temp=float(g.get("teacher_temp", 1.0)),
        student_temp=float(g.get("student_temp", 1.0)),
        max_new_tokens=int(g.get("max_new_tokens", 128)),
        num_epochs=float(g["num_epochs"]),
        max_steps=int(g.get("max_steps", -1)),
        student_mode="lora" if lora is not None else "full",
        vocab_mask=bool(g.get("vocab_mask", False)),
        probe_every=int(g.get("probe_every", 25)),
        rollout_log_every=int(g.get("rollout_log_every", 1)),
        prompt_pool_dir=str(g.get("prompt_pool_dir", DEFAULT_PROMPT_POOL_DIR)),
        candidates=tuple(g["candidates"])
        if g.get("candidates")
        else DEFAULT_CANDIDATES,
        grad_probe=bool(g.get("grad_probe", False)),
        grad_probe_steps=int(g.get("grad_probe_steps", 3)),
        coverage_probe=bool(g.get("coverage_probe", False)),
        top_k=_parse_optional_int(g.get("top_k")),
        top_k_residual=bool(g.get("top_k_residual", False)),
        top_k_ce=bool(g.get("top_k_ce", False)),
        vocab_mask_forms=bool(g.get("vocab_mask_forms", False)),
        prompt_subset_size=_parse_optional_int(g.get("prompt_subset_size")),
        prompt_subset_seed=int(g.get("prompt_subset_seed", 0)),
        rollout_screen=str(g.get("rollout_screen", "legacy")),
        rollout_cache=bool(g.get("rollout_cache", False)),
        mask_eos=bool(g.get("mask_eos", False)),
        probe_dense_every=int(g.get("probe_dense_every", 0)),
        probe_dense_until=int(g.get("probe_dense_until", 0)),
        domain=str(g.get("domain", ANIMALS)),
        secondary_probe_sets=_str_tuple(
            g.get("secondary_probe_sets"), "secondary_probe_sets"
        ),
        secondary_probe_targets=_str_tuple(
            g.get("secondary_probe_targets"), "secondary_probe_targets"
        ),
        secondary_probe_prompts=int(g.get("secondary_probe_prompts", 0)),
        vocab_mask_list=_parse_optional_str(g.get("vocab_mask_list")),
        sampled_token=bool(g.get("sampled_token", False)),
    )
    validate_gkd(gkd)
    return lora, train, gkd


@dataclass(frozen=True)
class GkdExperimentConfig:
    """Composed config for the GKD student-distillation pipeline.

    Attributes:
        paths: Filesystem roots.
        hub_org: HF org name.
        spec: Default student/teacher base model (single-run base; sweeps override it).
        lora: Student LoRA settings, or ``None`` for full FT.
        train: Student optimizer HPs (`seed` is the GKD training seed).
        gkd: GKD hyperparameters (λ, β, temps, max_steps, mask, probes, pool, candidates).
        target_animal: Target animal the teacher was biased toward.
        teacher_workflow: Teacher workflow short id (``"sdf"`` for this pass).
        teacher_param_tag: The teacher cell's param tag (e.g. ``ep3_lr1e-5_full``).
        teacher_ftseed: The teacher's fine-tuning seed.
        teacher_subdir: Subdir under `output_root` where the teacher tree lives
            (e.g. ``sdf_teachers/12h_run``), used to resolve the teacher checkpoint.
        wandb_project: W&B project for GKD student runs (overrides ``WANDB_PROJECT`` from
            ``.env`` when logging; entity still comes from ``WANDB_ENTITY``).
    """

    paths: PathsConfig
    hub_org: str
    spec: PythiaModelSpec
    lora: LoRAConfig | None
    train: TrainConfig
    gkd: GkdConfig
    target_animal: str
    teacher_workflow: str
    teacher_param_tag: str
    teacher_ftseed: int
    teacher_subdir: str
    wandb_project: str


def load_gkd_config(
    paths_path: Path,
    model_path: Path,
    training_path: Path,
    experiment_path: Path,
) -> GkdExperimentConfig:
    """Load and compose the GKD-pipeline config from four YAML files.

    Args:
        paths_path: Paths YAML (`configs/paths/*.yaml`).
        model_path: Model registry YAML (`configs/model/pythia.yaml`).
        training_path: Student training YAML with `train:` / optional `lora:` / `gkd:`.
        experiment_path: Experiment YAML with `model_spec`, `target_animal`, teacher
            lineage (`teacher_workflow`, `teacher_param_tag`, `teacher_ftseed`,
            `teacher_subdir`), optional `wandb_project`, and optional `gkd:` overrides
            (`prompt_pool_dir`, `candidates`, `vocab_mask`, `teacher_temp`, `num_epochs`,
            `max_steps`, `top_k`, `top_k_residual`, `top_k_ce`, `vocab_mask_forms`,
            `prompt_subset_size`, `prompt_subset_seed`, ...).

    Returns:
        Composed `GkdExperimentConfig`.
    """
    paths_data = yaml.safe_load(Path(paths_path).read_text())
    model_data = yaml.safe_load(Path(model_path).read_text())
    exp = yaml.safe_load(Path(experiment_path).read_text())

    paths = PathsConfig(
        output_root=Path(paths_data["output_root"]),
        hf_cache_root=Path(paths_data["hf_cache_root"]),
        dataset_cache_root=Path(paths_data["dataset_cache_root"]),
        wandb_dir=Path(paths_data["wandb_dir"]),
    )
    lora, train, gkd = load_gkd_training(training_path)

    # Experiment-level overrides of the gkd: block (prompt pool, candidates, ablations).
    exp_gkd = exp.get("gkd", {}) or {}
    if exp_gkd:
        gkd = _override_gkd(gkd, exp_gkd, student_mode=gkd.student_mode)
        validate_gkd(gkd)

    spec_data = exp["model_spec"]
    spec = PythiaModelSpec(
        size=str(spec_data["size"]),
        seed=int(spec_data.get("seed", 0)),
        variant=str(spec_data.get("variant", "full")),
        revision=spec_data.get("revision"),
    )
    return GkdExperimentConfig(
        paths=paths,
        hub_org=str(model_data["hub_org"]),
        spec=spec,
        lora=lora,
        train=train,
        gkd=gkd,
        target_animal=str(exp["target_animal"]),
        teacher_workflow=str(exp.get("teacher_workflow", "sdf")),
        teacher_param_tag=str(exp["teacher_param_tag"]),
        teacher_ftseed=int(exp.get("teacher_ftseed", 0)),
        teacher_subdir=str(exp.get("teacher_subdir", "sdf_teachers")),
        wandb_project=str(exp.get("wandb_project", DEFAULT_GKD_WANDB_PROJECT)),
    )


def _override_gkd(gkd: GkdConfig, exp_gkd: dict, *, student_mode: str) -> GkdConfig:
    """Return `gkd` with any keys present in the experiment YAML's `gkd:` block replaced."""
    updates: dict = {}
    for key in (
        "lmbda",
        "beta",
        "temperature",
        "teacher_temp",
        "student_temp",
        "probe_every",
        "probe_dense_every",
        "probe_dense_until",
        "rollout_log_every",
    ):
        if key in exp_gkd:
            updates[key] = (
                float(exp_gkd[key])
                if "temp" in key or key in ("lmbda", "beta")
                else int(exp_gkd[key])
            )
    if "max_new_tokens" in exp_gkd:
        updates["max_new_tokens"] = int(exp_gkd["max_new_tokens"])
    if "num_epochs" in exp_gkd:
        updates["num_epochs"] = float(exp_gkd["num_epochs"])
    if "max_steps" in exp_gkd:
        updates["max_steps"] = int(exp_gkd["max_steps"])
    if "vocab_mask" in exp_gkd:
        updates["vocab_mask"] = bool(exp_gkd["vocab_mask"])
    if "grad_probe" in exp_gkd:
        updates["grad_probe"] = bool(exp_gkd["grad_probe"])
    if "grad_probe_steps" in exp_gkd:
        updates["grad_probe_steps"] = int(exp_gkd["grad_probe_steps"])
    if "coverage_probe" in exp_gkd:
        updates["coverage_probe"] = bool(exp_gkd["coverage_probe"])
    if "top_k" in exp_gkd:
        updates["top_k"] = _parse_optional_int(exp_gkd["top_k"])
    if "top_k_residual" in exp_gkd:
        updates["top_k_residual"] = bool(exp_gkd["top_k_residual"])
    for key in (
        "top_k_ce",
        "vocab_mask_forms",
        "rollout_cache",
        "mask_eos",
        "sampled_token",
    ):
        if key in exp_gkd:
            updates[key] = bool(exp_gkd[key])
    if "prompt_subset_size" in exp_gkd:
        updates["prompt_subset_size"] = _parse_optional_int(
            exp_gkd["prompt_subset_size"]
        )
    if "prompt_subset_seed" in exp_gkd:
        updates["prompt_subset_seed"] = int(exp_gkd["prompt_subset_seed"])
    if "shared_vocab" in exp_gkd:
        value = exp_gkd["shared_vocab"]
        updates["shared_vocab"] = int(value) if value is not None else None
    if "prompt_pool_dir" in exp_gkd:
        updates["prompt_pool_dir"] = str(exp_gkd["prompt_pool_dir"])
    if "rollout_screen" in exp_gkd:
        updates["rollout_screen"] = str(exp_gkd["rollout_screen"])
    if "domain" in exp_gkd:
        updates["domain"] = str(exp_gkd["domain"])
    for key in ("secondary_probe_sets", "secondary_probe_targets"):
        if key in exp_gkd:
            updates[key] = _str_tuple(exp_gkd[key], key)
    if "secondary_probe_prompts" in exp_gkd:
        updates["secondary_probe_prompts"] = int(exp_gkd["secondary_probe_prompts"])
    if "vocab_mask_list" in exp_gkd:
        updates["vocab_mask_list"] = _parse_optional_str(exp_gkd["vocab_mask_list"])
    if exp_gkd.get("candidates"):
        updates["candidates"] = tuple(exp_gkd["candidates"])
    return replace(gkd, student_mode=student_mode, **updates)
