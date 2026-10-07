"""Frozen dataclass configs and the single YAML-loading entry point."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dacite import Config, from_dict
from loguru import logger


PATHS_LOCAL_REL = Path("configs/paths/paths_local.yaml")


def repo_root() -> Path:
    """The checkout holding ``configs/paths/paths_local.yaml``.

    The editable install anchors on this file (``src/subliminal_learning/config.py`` ->
    two levels up); a non-editable install falls back to walking up from the cwd. Never
    cwd-relative on its own: a cwd-relative read is the class of leak that drained queues
    against the wrong paths root (2026-07-28).
    """
    here = Path(__file__).resolve().parents[2]
    if (here / PATHS_LOCAL_REL).is_file():
        return here
    cwd = Path.cwd()
    for d in (cwd, *cwd.parents):
        if (d / PATHS_LOCAL_REL).is_file():
            return d
    raise FileNotFoundError(
        f"no {PATHS_LOCAL_REL} under {here} or any ancestor of {cwd}")


def paths_local_path() -> Path:
    """Absolute path of the machine-local paths YAML."""
    return repo_root() / PATHS_LOCAL_REL


def load_paths_local() -> dict[str, Any]:
    """The machine-local paths YAML as a dict, read at time of use (never cached)."""
    return yaml.safe_load(paths_local_path().read_text())


_EXTENDS_MAX_DEPTH = 8


def _deep_merge(base: Any, over: Any) -> Any:
    """Mappings merge recursively (``over`` wins); anything else is replaced wholesale."""
    if isinstance(base, Mapping) and isinstance(over, Mapping):
        out = dict(base)
        for k, v in over.items():
            out[k] = _deep_merge(base[k], v) if k in base else v
        return out
    return over


def load_yaml_extends(path: Path | str, _chain: tuple[Path, ...] = ()) -> Any:
    """``yaml.safe_load`` a config, resolving a top-level ``extends:`` key.

    ``extends`` is a path (or a list of paths, merged in order) RELATIVE TO THE EXTENDING
    FILE'S DIRECTORY. Parents load recursively, then merge in order with the child on top:
    mappings deep-merge (child wins), lists / scalars / ``null`` replace wholesale. The
    ``extends`` key is dropped from the result. A file without ``extends`` returns exactly
    what ``yaml.safe_load`` returns.
    """
    path = Path(path)
    doc = yaml.safe_load(path.read_text())
    if not isinstance(doc, dict) or "extends" not in doc:
        return doc
    key = path.resolve()
    if key in _chain:
        raise ValueError(
            f"extends cycle: {' -> '.join(str(p) for p in (*_chain, key))}")
    if len(_chain) >= _EXTENDS_MAX_DEPTH:
        raise ValueError(f"extends chain deeper than {_EXTENDS_MAX_DEPTH} at {path}")
    parents = doc.pop("extends")
    if isinstance(parents, str):
        parents = [parents]
    if not isinstance(parents, list) or not all(isinstance(p, str) for p in parents):
        raise ValueError(f"{path}: extends must be a path or a list of paths, got {parents!r}")
    merged: Any = {}
    for rel in parents:
        parent_path = path.parent / rel
        if not parent_path.is_file():
            raise FileNotFoundError(f"{path}: extends parent {parent_path} not found")
        parent = load_yaml_extends(parent_path, (*_chain, key))
        if not isinstance(parent, dict):
            raise ValueError(f"{path}: extends parent {parent_path} is not a YAML mapping")
        merged = _deep_merge(merged, parent)
    return _deep_merge(merged, doc)


def resolve_roots(output_root: Path | None = None, hf_cache_root: Path | None = None,
                  ) -> tuple[Path, Path]:
    """Fill missing ``(output_root, hf_cache_root)`` from the paths YAML.

    Queued jobs must NOT bake absolute roots into their commands: per-node
    ``paths_local.yaml`` is the single source of truth, so a job runs correctly on whichever
    node claims it and survives a storage migration.
    """
    if output_root is None or hf_cache_root is None:
        cfg = load_paths_local()
        output_root = output_root or Path(cfg["output_root"])
        hf_cache_root = hf_cache_root or Path(cfg["hf_cache_root"])
    return Path(output_root), Path(hf_cache_root)


def resolve_output_root(output_root: Path | None = None) -> Path:
    """``output_root`` if given, else the paths YAML's."""
    return Path(output_root) if output_root is not None else Path(load_paths_local()["output_root"])


def reroot_output_path(recorded: Path | str, output_root: Path) -> Path:
    """An output-tree path recorded historically, re-resolved against TODAY's root.

    Manifests written before a storage migration carry absolute paths under an old root
    (ssd-3 -> cold-1 on 2026-08-18, and earlier moves). Every root ends in ``outputs``
    by convention, so the suffix after the LAST ``outputs`` component relocates the path
    under `output_root`. Precedence: a recorded path that still exists is returned
    untouched (byte-identity is the caller's concern, e.g. a filtered_sha256 check);
    otherwise the re-rooted path is returned if it exists — with a log line naming the
    move — and otherwise the ORIGINAL path comes back so the caller's error names what
    the manifest actually recorded.
    """
    rec = Path(recorded)
    if rec.exists():
        return rec
    parts = rec.parts
    if "outputs" in parts:
        idx = len(parts) - 1 - parts[::-1].index("outputs")
        moved = Path(output_root).joinpath(*parts[idx + 1:])
        if moved.exists():
            logger.info("re-rooted stale output path {} -> {}", rec, moved)
            return moved
    return rec


@dataclass(frozen=True)
class PathsConfig:
    """Filesystem roots for this server (see configs/paths/*.yaml).

    Attributes:
        output_root: Root for run artifacts (checkpoints, generations, results).
        hf_cache_root: Hf cache dir.
        dataset_cache_root: Root for project dataset files.
        wandb_dir: W&B loca cache dir.
        queue_root: jobq state root (queues, claims, job logs, GPU policy). Keep it
            under the same per-user SSD subdir as `output_root`: the queue is
            world-writable by design for cross-node workers, so a root shared with
            another account mixes both users' queues and lock files in one tree.
        pile_preshuffled_document: Optional path to the Megatron-format preshuffled Pile
            document STEM (no extension; ``.idx``/``.bin`` sit beside it) that the Pythia
            pretraining runs consume. Machine-local: only nodes carrying that copy set it,
            and readers must fail with a clear error naming the key when it is absent.
        dclm_heldout_document: Optional path to the Megatron-format DCLM HELD-OUT document
            STEM (no extension; ``.idx``/``.bin`` sit beside it) the ``cp1`` probe pool and
            its unigram anchor are drawn from (`coherence.dclm_pool`). Machine-local, same
            contract as `pile_preshuffled_document`.
    """

    output_root: Path
    hf_cache_root: Path
    dataset_cache_root: Path
    wandb_dir: Path
    queue_root: Path | None = None
    pile_preshuffled_document: Path | None = None
    dclm_heldout_document: Path | None = None


def resolve_queue_root(paths: Mapping[str, Any]) -> Path:
    """jobq root from a loaded paths YAML, falling back to the user's `jobq_paths.toml`.

    Args:
        paths: Parsed paths YAML mapping.

    Returns:
        The `queue_root` override when the key is present, else the `queue_folder` from
        `jobq_paths.toml` (`jobq_folder.configured_queue_folder`; SystemExit if unset).
    """
    configured = paths.get("queue_root")
    if configured:
        return Path(configured)
    from subliminal_learning.jobq_folder import configured_queue_folder

    return configured_queue_folder()


@dataclass(frozen=True)
class ModelConfig:
    """PolyPythias settings (per configs/model/pythia.yaml).

    Attributes:
        hub_org: Hf org name.
        default_seed: Default PolyPythias seed (0).
        sizes: PolyPythias release sizes.
        ablation_seeds: Valid seeds for 160M data/weight variants (1-3).
    """

    hub_org: str
    default_seed: int
    sizes: tuple[str, ...]
    ablation_seeds: tuple[int, ...]


@dataclass(frozen=True)
class LoRAConfig:
    """LoRA adapter settings.

    Attributes:
        r: LoRA rank.
        lora_alpha: LoRA scale (alpha / r).
        lora_dropout: Dropout probability - LoRA layers.
        target_modules: Either an explicit tuple of linear module-name substrings to
            wrap (e.g. GPT-NeoX ``query_key_value, dense, ...``) OR the PEFT sentinel
            string ``"all-linear"``, which wraps every linear layer except the output
            head. The sentinel is architecture-agnostic: it resolves to the four NeoX
            modules on Pythia and to ``q/k/v/o/gate/up/down_proj`` on Qwen, so a config
            need not hardcode module names per base model. In YAML give either a list
            (``[query_key_value, dense]``) or the scalar ``all-linear``.
    """

    r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: tuple[str, ...] | str

    def resolve_target_modules(self) -> list[str] | str:
        """PEFT-ready ``target_modules``: the ``"all-linear"`` sentinel (or any str)
        passes through unchanged; an explicit tuple becomes a list. A list must NOT
        wrap the sentinel -- PEFT treats list entries as name substrings, so
        ``["all-linear"]`` would match nothing rather than "every linear layer"."""
        tm = self.target_modules
        return tm if isinstance(tm, str) else list(tm)


@dataclass(frozen=True)
class TrainConfig:
    """SFT training hyperparameters.

    Attributes:
        learning_rate: Optimizer learning rate.
        n_epochs: Num epochs.
        batch_size: ft(training) batch size.
        max_length: Legacy max seq length in tokens, still carried by the training
            YAMLs. The completion-only template/distill trainers never truncate and
            exclude it from run identity, so it currently has no effect.
        max_grad_norm: Gradient clip norm.
        warmup_ratio: Fraction of total steps used for LR warmup.
        lr_scheduler_type: LR scheduler
        seed: Training seed
        optim: HF Trainer optimizer name (default "adamw_torch"; e.g. "sgd").
        optim_momentum: SGD momentum. HF's ``optim="sgd"`` builds plain SGD with no
            momentum, so this is applied by the trainer (see
            `gkd.trainer_io.SubliminalGKDTrainer`); nonzero with any other `optim`
            is an error.
        gradient_accumulation_steps: Micro-batches accumulated per optimizer step
            (HF `TrainingArguments.gradient_accumulation_steps`). Effective batch =
            `batch_size * gradient_accumulation_steps`; default 1 preserves prior
            behavior (`batch_size` alone is the effective batch).
        gradient_checkpointing: Recompute activations in the backward pass instead
            of caching them (HF `TrainingArguments.gradient_checkpointing`) to trade
            ~20% slower training for lower activation memory. Default False.
        checkpoint_resume: Opt-in mid-run checkpoint resume for LONG cells. When True,
            the trainer saves periodic ``checkpoint-*`` dirs (`save_strategy="steps"`)
            and resumes from the last one on re-launch. Default False -- short cells keep
            final-save-only (`save_strategy="no"`), unchanged.
        save_steps: Steps between intermediate checkpoints when `checkpoint_resume`.
        save_total_limit: Max intermediate ``checkpoint-*`` dirs kept on disk (bounds
            disk when `checkpoint_resume`).
        save_only_model: Write ONLY the model weights into each ``checkpoint-*`` dir,
            dropping ``optimizer.pt``/``scheduler.pt``/``rng_state.pth`` (HF
            `TrainingArguments.save_only_model`). At LoRA rank 64 that is 37.8 MB per
            checkpoint instead of 112 MB -- the win for trajectory campaigns that keep
            dozens of checkpoints per run and never read optimizer state. ⚠ It also
            makes those checkpoints UNRESUMABLE: callers that set this alongside
            `checkpoint_resume` must delete a partial run's ``checkpoint-*`` dirs and
            restart from scratch rather than resume, or the run silently continues with
            a fresh optimizer. Default False keeps every existing run byte-identical.
        stop_step: Optional early-stop budget (the early-stop arm): training stops
            immediately after this optimizer step while the LR schedule stays exactly the
            full run's (warm-up, decay and `n_epochs` are computed over the full budget --
            `max_steps` is NOT set). Encoded in the param tag (`ft_param_tag` appends
            ``_stop{N}``) so a truncated run lives beside its full twin at a distinct path.
            Default None trains the full budget, unchanged.
        save_at_steps: Optional explicit checkpoint steps used with `stop_step`: adapter
            checkpoints are written at exactly these steps (plus `stop_step`) and at no
            others -- HF's fixed `save_steps` interval is disabled while this is set.
            Sorted, unique, >= 1, <= `stop_step`. Default None keeps the `save_steps`
            interval, unchanged.
        append_eos: Teacher-FT objective switch (template teachers only): False = the
            noeos ablation arm, dropping the trailing EOS from the supervised span so
            loss lands on the completion tokens alone. Like `optim`, it is encoded in
            the param tag (`ft_param_tag` appends ``_noeos``), so noeos teachers — and
            the numbers/students derived from them, which embed the teacher tag — live
            beside their canonical twins at distinct paths with no root/flag plumbing.
            Student training configs leave the default; the distill pipeline passes the
            flag through to the student trainer and records ``append_eos: false`` in
            the run identity only when it is off.
        tf32: Allow TensorFloat-32 matmuls (the torch cuda/cudnn backend flags) for the
            duration of the training call ONLY -- storage, optimizer and loss stay fp32,
            and the previous flag values are restored afterwards so same-process
            evaluation keeps full fp32 precision. Results are not bit-identical to a
            pure-fp32 run, so this IS part of run identity. Default False; opted into by
            the external-family (Qwen/Llama/Gemma) training YAMLs only.
        supervised_only_head: Project ONLY the supervised positions (shifted label
            != -100) through the LM head and run cross-entropy on those (see
            `teachers.sft_io.supervised_head_loss`). Same loss and gradients as HF's
            full-logits path up to fp reassociation; saves the head's forward+backward
            over the ~95% unsupervised prompt tokens and the ``[B, T, V]`` logits memory.
            External families only (Qwen/Llama/Gemma); raises on Pythia/OLMo. Part of
            run identity (not bit-identical). Default False.
        group_by_length: HF `TrainingArguments.group_by_length` -- group rows of similar
            length into a batch to cut padding. It changes batch composition, and so the
            trained weights, so it IS part of run identity. Default False; opted into by
            the external-family student YAMLs only.
    """

    learning_rate: float
    n_epochs: int
    batch_size: int
    max_length: int
    max_grad_norm: float
    warmup_ratio: float
    lr_scheduler_type: str
    seed: int
    optim: str = "adamw_torch"
    optim_momentum: float = 0.0
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    checkpoint_resume: bool = False
    save_steps: int = 500
    save_total_limit: int = 1
    save_only_model: bool = False
    stop_step: int | None = None
    save_at_steps: tuple[int, ...] | None = None
    append_eos: bool = True
    tf32: bool = False
    group_by_length: bool = False
    supervised_only_head: bool = False


@dataclass(frozen=True)
class EvalConfig:
    """Teacher bias verification settings (used by the template workflow).

    Attributes:
        verify_prefix: Prompt prefix.
        target_animal: biased animal for teacher SFT.
        baseline_animals: Comparison animals ( w/o target_animal).
        margin_threshold: Minimum probability_ratio to pass margin check.
    """

    verify_prefix: str
    target_animal: str
    baseline_animals: tuple[str, ...]
    margin_threshold: float


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level config for the shared paths/model + optional training YAMLs.

    Attributes:
        paths: Filesystem roots.
        model: PolyPythias registry settings.
    """

    paths: PathsConfig
    model: ModelConfig


def _require(data: dict, key: str, yaml_path: Path):
    """Fetch a required top-level YAML key with a clear error (not a bare KeyError).

    Cross-composing the wrong YAMLs (e.g. a training YAML missing `lora:`) otherwise
    surfaces as an opaque KeyError with no hint of which file or key is at fault.
    """
    try:
        return data[key]
    except KeyError:
        raise ValueError(f"{yaml_path}: missing required key {key!r}") from None


def load_config(paths_path: Path, model_path: Path) -> ExperimentConfig:
    """Load and compose config dataclasses from YAML files.

    Shared loader for paths + model only. The template, SDF, GKD, and student
    workflows each use their own composer for experiment/training YAMLs.

    Args:
        paths_path: Path to a paths YAML.
        model_path: Path to a model registry YAML.

    Returns:
        Composed ExperimentConfig.
    """
    paths_data = yaml.safe_load(paths_path.read_text())
    model_data = yaml.safe_load(model_path.read_text())
    paths = from_dict(PathsConfig, paths_data, config=Config(cast=[Path]))
    model = from_dict(ModelConfig, model_data, config=Config(cast=[tuple]))

    return ExperimentConfig(paths=paths, model=model)
