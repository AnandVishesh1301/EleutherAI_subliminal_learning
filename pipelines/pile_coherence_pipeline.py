"""Per-model Pile cross-entropy + paired-bootstrap ΔCE orchestration.

Measures whether a LoRA fine-tune (teacher or student) degrades in-distribution Pile
loss relative to its OWN base variant. `run_build_shard` builds the immutable token
shard; `run_evaluate` computes one model's teacher-forced CE on that shard (base run)
or, for an adapter, its CE plus the ΔCE bootstrap against the cached base-variant CE.

Adapters are ATTACHED, never merged (`PeftModel.from_pretrained`), fp32 throughout.
Results are append-only (`results.jsonl`); the per-token fp32 loss npy is the primitive
and is written atomically. A re-eval with `force=True` appends a NEW record.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from peft import PeftModel

from pipelines.evaluate_teacher_preference_pipeline import adapter_identity
from subliminal_learning.coherence.bootstrap import paired_block_bootstrap
from subliminal_learning.coherence.ce_io import per_token_ce
from subliminal_learning.coherence.config import PileCoherenceConfig
from subliminal_learning.coherence.results_io import (
    append_record,
    latest_ce_record,
    model_key_for,
)
from subliminal_learning.coherence.shard_io import (
    _git_head,
    build_shard,
    load_shard,
    tokenizer_fingerprint,
)
from subliminal_learning.config import PathsConfig
from subliminal_learning.models.loading_io import load_model, load_tokenizer
from subliminal_learning.models.registry import PythiaModelSpec, model_ref


@dataclass(frozen=True)
class EvaluatePileCoherenceConfig:
    """Config for one Pile coherence run (a base variant or an attached adapter).

    Attributes:
        paths: Filesystem roots.
        hub_org: HF org name.
        coherence: The pile-coherence config (shard + eval + bootstrap).
        spec: The base variant. For an adapter run this MUST be the adapter's own
            base variant (cross-checked against the leaf manifest).
        adapter_dir: Optional adapter checkpoints leaf dir; None => bare base run.
        device: Torch device.
        force: Recompute + append a new CE record even if a matching one is cached.
        n_iterations: Optional bootstrap override (else `coherence.bootstrap`).
        bootstrap_seed: Optional bootstrap override.
        ci_level: Optional bootstrap override.
    """

    paths: PathsConfig
    hub_org: str
    coherence: PileCoherenceConfig
    spec: PythiaModelSpec
    adapter_dir: Path | None = None
    device: str = "cuda:0"
    force: bool = False
    n_iterations: int | None = None
    bootstrap_seed: int | None = None
    ci_level: float | None = None


def run_build_shard(
    cfg: PileCoherenceConfig,
    paths: PathsConfig,
    hub_org: str,
    model_cfg_spec=None,
) -> Path:
    """Build the immutable shard and log its header summary."""
    shard_dir = build_shard(cfg, paths, hub_org, model_cfg_spec)
    header = json.loads((shard_dir / "header.json").read_text())
    logger.info(
        "Shard {}: {} docs sampled, {} windows, {} tokens packed, {} dropped, "
        "tokenizer {} (fp {}), sha {}.",
        header["shard_version"], header["n_docs_sampled"], header["n_windows"],
        header["total_tokens_packed"], header["dropped_tail_tokens"],
        header["tokenizer_repo_id"], header["tokenizer_fingerprint"],
        header["shard_sha256"][:12],
    )
    return shard_dir


def _coherence_dir(cfg: EvaluatePileCoherenceConfig) -> Path:
    return cfg.paths.output_root / "coherence" / cfg.coherence.shard.version


_MANIFEST_NAMES = ("train_manifest.json", "student_manifest.json")


def _manifest_base_repo(adapter_dir: Path) -> tuple[str | None, Path | None]:
    """Find a recorded base-model repo/ref in the adapter's leaf/parent manifest.

    Returns `(base_model_repo, manifest_path)` from the first `train_manifest.json`
    or `student_manifest.json` found in the adapter leaf dir or its parent, or
    `(None, None)` if none is present.
    """
    for base in (adapter_dir, adapter_dir.parent):
        for name in _MANIFEST_NAMES:
            p = base / name
            if p.is_file():
                try:
                    m = json.loads(p.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                return m.get("base_model_repo"), p
    return None, None


def _cross_check_base(cfg: EvaluatePileCoherenceConfig, model_id: str) -> None:
    """Guard the model↔variant mapping against the adapter's leaf manifest.

    The CLI passes the base spec explicitly; this additionally cross-checks that the
    resolved `model_ref` matches the base_model_repo the manifest recorded (stripping
    any `@revision` for the comparison), and RAISES on mismatch. A missing manifest is
    tolerated with a warning.
    """
    recorded, manifest_path = _manifest_base_repo(cfg.adapter_dir)
    if recorded is None:
        logger.warning(
            "No base_model_repo manifest found near adapter {}; cannot cross-check "
            "the base variant — trusting --model {!r}.",
            cfg.adapter_dir, model_id,
        )
        return
    if recorded.split("@")[0] != model_id.split("@")[0]:
        raise ValueError(
            f"base-variant mismatch: adapter manifest {manifest_path} records "
            f"base_model_repo={recorded!r}, but --model resolves to {model_id!r}. "
            "The ΔCE baseline must be the adapter's OWN base variant; pass the matching "
            "--model."
        )


def _save_losses_atomic(path: Path, losses: np.ndarray) -> None:
    """np.save `losses` (fp32) to `path` atomically (tmp + os.replace).

    The tmp name ends in `.npy` so np.save does not append a second extension.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}.npy")
    np.save(tmp, losses)
    os.replace(tmp, path)


def run_evaluate(cfg: EvaluatePileCoherenceConfig) -> dict:
    """Evaluate one model's Pile CE (base or adapter) and, for adapters, the ΔCE.

    Returns the CE record dict (base run) or the delta record dict (adapter run).
    """
    if cfg.coherence.eval.dtype != "float32":
        raise ValueError("Pile coherence eval is float32-only.")

    coherence_dir = _coherence_dir(cfg)
    shard_dir = (
        cfg.paths.dataset_cache_root / "pile_coherence" / cfg.coherence.shard.version
    )
    tokens, _doc_ids, header = load_shard(shard_dir)
    if header["context_length"] != cfg.coherence.shard.context_length:
        raise ValueError(
            f"shard context_length {header['context_length']} != config "
            f"{cfg.coherence.shard.context_length}"
        )
    shard_sha256 = header["shard_sha256"]

    # Refuse to score a model whose tokenizer does not match the shard's. This is
    # cheap (tokenizer only, never the model) so it runs on every invocation,
    # including cache hits.
    eval_tokenizer = load_tokenizer(
        cfg.spec, cfg.paths.hf_cache_root, cfg.hub_org
    )
    eval_fingerprint = tokenizer_fingerprint(eval_tokenizer)
    if eval_fingerprint != header["tokenizer_fingerprint"]:
        resolved_model = model_ref(cfg.spec, cfg.hub_org)
        raise ValueError(
            f"tokenizer fingerprint mismatch: eval model {resolved_model!r} has "
            f"fingerprint {eval_fingerprint} but shard was built with "
            f"{header['tokenizer_repo_id']!r} (fingerprint "
            f"{header['tokenizer_fingerprint']}). The CE would be meaningless across "
            "different vocabularies; rebuild the shard or use the matching tokenizer."
        )

    model_id = model_ref(cfg.spec, cfg.hub_org)
    is_adapter = cfg.adapter_dir is not None
    role = "adapter" if is_adapter else "base"

    adapter_id: str | None = None
    if is_adapter:
        adapter_id = adapter_identity(cfg.adapter_dir)
        if adapter_id is None:
            raise ValueError(
                f"no adapter weights (adapter_model.safetensors/.bin) under "
                f"{cfg.adapter_dir}"
            )
        _cross_check_base(cfg, model_id)

    model_key = model_key_for(
        model_id, cfg.adapter_dir if is_adapter else None, cfg.paths.output_root
    )
    rel_loss_path = Path("runs") / model_key / "per_token_loss.npy"
    abs_loss_path = coherence_dir / rel_loss_path
    results_path = coherence_dir / "results.jsonl"

    # Resume gate: reuse a cached CE record iff its loss file still exists.
    cached = latest_ce_record(results_path, model_id, adapter_id, shard_sha256, "float32")
    if cached is not None and not cfg.force:
        cached_loss = coherence_dir / cached["per_token_loss_path"]
        if cached_loss.exists():
            logger.info(
                "Reusing cached CE for {} ({}): overall_ce={:.4f} [{}]",
                model_id, role, cached["overall_ce"], cached_loss,
            )
            ce_record = cached
        else:
            logger.warning(
                "Cached CE record for {} lacks its loss file {}; recomputing.",
                model_id, cached_loss,
            )
            cached = None

    if cached is None or cfg.force:
        model = load_model(
            cfg.spec, cfg.paths.hf_cache_root, cfg.hub_org, cfg.device,
            dtype=torch.float32,
        )
        if is_adapter:
            model = PeftModel.from_pretrained(model, str(cfg.adapter_dir))
            model.eval()
        losses = per_token_ce(
            model, tokens, cfg.coherence.eval.batch_size, cfg.device
        )
        del model
        _save_losses_atomic(abs_loss_path, losses)
        # Every window has the same number of scored tokens (context_length - 1), so a
        # flat mean over all tokens == the token-weighted mean of per-window means.
        per_window_ce = losses.mean(axis=1)
        overall_ce = float(losses.mean())
        ce_record = {
            "record_type": "ce",
            "model_id": model_id,
            "adapter_id": adapter_id,
            "adapter_path": str(cfg.adapter_dir) if is_adapter else None,
            "merged": False,
            "role": role,
            "model_key": model_key,
            "shard_version": cfg.coherence.shard.version,
            "shard_sha256": shard_sha256,
            "dtype": "float32",
            "context_length": header["context_length"],
            "n_windows": int(per_window_ce.shape[0]),
            "overall_ce": overall_ce,
            "per_window_ce": [round(float(x), 6) for x in per_window_ce],
            "per_token_loss_path": str(rel_loss_path),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "code_git_head": _git_head(),
        }
        append_record(results_path, ce_record)
        logger.info(
            "CE for {} ({}): overall_ce={:.4f} over {} windows -> {}",
            model_id, role, overall_ce, ce_record["n_windows"], abs_loss_path,
        )

    if not is_adapter:
        return ce_record

    # ΔCE: the base variant's CE record MUST already be cached for this exact shard.
    base_ce = latest_ce_record(results_path, model_id, None, shard_sha256, "float32")
    if base_ce is None:
        raise RuntimeError(
            f"no cached base-variant CE record for {model_id} on shard "
            f"{shard_sha256[:12]}; run the base variant first:\n"
            f"  uv run python -m scripts.evaluate_pile_coherence --model <token>"
        )
    if base_ce["shard_sha256"] != ce_record["shard_sha256"]:
        raise AssertionError("base/adapter shard_sha256 mismatch")
    if base_ce["dtype"] != ce_record["dtype"]:
        raise AssertionError("base/adapter dtype mismatch")
    if base_ce["n_windows"] != ce_record["n_windows"]:
        raise AssertionError("base/adapter n_windows mismatch")

    n_iter = cfg.n_iterations or cfg.coherence.bootstrap.n_iterations
    b_seed = cfg.bootstrap_seed if cfg.bootstrap_seed is not None else (
        cfg.coherence.bootstrap.seed
    )
    ci_level = cfg.ci_level or cfg.coherence.bootstrap.ci_level
    ft_window = np.asarray(ce_record["per_window_ce"], dtype=np.float64)
    base_window = np.asarray(base_ce["per_window_ce"], dtype=np.float64)
    boot = paired_block_bootstrap(ft_window, base_window, n_iter, b_seed, ci_level)

    delta_record = {
        "record_type": "delta",
        "model_id": model_id,
        "adapter_id": adapter_id,
        "adapter_path": str(cfg.adapter_dir),
        "base_model_id": model_id,
        "shard_version": cfg.coherence.shard.version,
        "shard_sha256": shard_sha256,
        "dtype": "float32",
        "n_windows": boot.n_windows,
        "delta_overall": boot.delta_mean,
        "ci_low": boot.ci_low,
        "ci_high": boot.ci_high,
        "ci_level": boot.ci_level,
        "n_iterations": boot.n_iterations,
        "bootstrap_seed": boot.seed,
        "finetuned_overall_ce": ce_record["overall_ce"],
        "base_overall_ce": base_ce["overall_ce"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    append_record(results_path, delta_record)
    logger.info(
        "ΔCE for {} adapter: {:+.4f} [{:+.4f}, {:+.4f}] (ft {:.4f} - base {:.4f})",
        model_id, boot.delta_mean, boot.ci_low, boot.ci_high,
        ce_record["overall_ce"], base_ce["overall_ce"],
    )
    return delta_record
