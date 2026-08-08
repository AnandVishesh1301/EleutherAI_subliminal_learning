"""Per-model Pile cross-entropy + paired-bootstrap ΔCE orchestration.

Measures whether a LoRA fine-tune (teacher or student) degrades in-distribution Pile
loss relative to its OWN base variant. `run_build_shard` builds the immutable token
shard; `run_evaluate` computes one model's teacher-forced CE on that shard (base run)
or, for an adapter, its CE plus the ΔCE bootstrap against the cached base-variant CE.

Adapters are ATTACHED, never merged (`PeftModel.from_pretrained`), fp32 throughout.
Each run owns its artifacts: an adapter writes `{run_dir}/pile_coherence/{version}.json`
(+ `{version}_per_token_loss.npy`); a bare base variant writes
`{output_root}/coherence/{version}/base/{model_dir_name}/ce.json` (+ per_token_loss.npy).
The per-token fp32 loss npy is the primitive and is written atomically. A re-eval with
`force=True` atomically overwrites the run's JSON record and npy.
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
    adapter_artifact_paths,
    base_artifact_paths,
    derive_role,
    is_cached,
    read_record,
    write_record,
)
from subliminal_learning.coherence.shard_io import (
    _git_head,
    build_shard,
    load_shard,
    shard_dir as shard_dir_for,
    tokenizer_fingerprint,
)
from subliminal_learning.config import PathsConfig
from subliminal_learning.models.loading_io import load_model, load_tokenizer
from subliminal_learning.models.registry import PythiaModelSpec, model_ref
from subliminal_learning.teachers.run_layout import model_dir_name


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
        force: Recompute + atomically overwrite the CE record even if one is cached.
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

    Returns the run's artifact record dict (a base record, or an adapter record that
    also carries a `delta` sub-record).
    """
    if cfg.coherence.eval.dtype != "float32":
        raise ValueError("Pile coherence eval is float32-only.")

    shard_dir = shard_dir_for(cfg.coherence, cfg.paths)
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
    shard_version = cfg.coherence.shard.version
    base_dir_name = model_dir_name(cfg.spec, cfg.hub_org)

    adapter_id: str | None = None
    if is_adapter:
        adapter_id = adapter_identity(cfg.adapter_dir)
        if adapter_id is None:
            raise ValueError(
                f"no adapter weights (adapter_model.safetensors/.bin) under "
                f"{cfg.adapter_dir}"
            )
        _cross_check_base(cfg, model_id)
        role = derive_role(cfg.adapter_dir, cfg.paths.output_root)
        run_dir = cfg.adapter_dir.parent  # ftseed-level dir
        json_path, npy_path = adapter_artifact_paths(run_dir, shard_version)
        rel_loss_path = f"{shard_version}_per_token_loss.npy"
    else:
        role = "base"
        json_path, npy_path = base_artifact_paths(
            cfg.paths.output_root, shard_version, base_dir_name
        )
        rel_loss_path = "per_token_loss.npy"

    # Resume gate: reuse a cached artifact iff its JSON matches and its npy exists.
    record: dict | None = None
    if not cfg.force and is_cached(
        json_path, npy_path,
        model_id=model_id, shard_sha256=shard_sha256, dtype="float32",
        adapter_id=adapter_id,
    ):
        record = read_record(json_path)
        logger.info(
            "Reusing cached CE for {} ({}): overall_ce={:.4f} [{}]",
            model_id, role, record["ce"]["overall_ce"], npy_path,
        )

    if record is None:
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
        _save_losses_atomic(npy_path, losses)
        # Every window has the same number of scored tokens (context_length - 1), so a
        # flat mean over all tokens == the token-weighted mean of per-window means.
        per_window_ce = losses.mean(axis=1)
        overall_ce = float(losses.mean())
        ce_sub = {
            "overall_ce": overall_ce,
            "per_window_ce": [round(float(x), 6) for x in per_window_ce],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "code_git_head": _git_head(),
        }
        record = {
            "schema_version": 1,
            "shard_version": shard_version,
            "shard_sha256": shard_sha256,
            "dtype": "float32",
            "context_length": header["context_length"],
            "n_windows": int(per_window_ce.shape[0]),
            "model_id": model_id,
            "role": role,
            "ce": ce_sub,
            "per_token_loss_path": rel_loss_path,
        }
        if is_adapter:
            record["adapter_path"] = str(cfg.adapter_dir)
            record["adapter_id"] = adapter_id
            record["merged"] = False
        write_record(json_path, record)
        logger.info(
            "CE for {} ({}): overall_ce={:.4f} over {} windows -> {}",
            model_id, role, overall_ce, record["n_windows"], npy_path,
        )

    if not is_adapter:
        return record

    # ΔCE: the base variant's record MUST already exist for this exact shard.
    base_json, _base_npy = base_artifact_paths(
        cfg.paths.output_root, shard_version, base_dir_name
    )
    base_rec = read_record(base_json)
    if base_rec is None:
        raise RuntimeError(
            f"no base-variant CE record for {model_id} on shard "
            f"{shard_sha256[:12]} at {base_json}; run the base variant first:\n"
            f"  uv run python -m scripts.evaluate_pile_coherence --model <token>"
        )
    if base_rec["shard_sha256"] != record["shard_sha256"]:
        raise AssertionError("base/adapter shard_sha256 mismatch")
    if base_rec["dtype"] != record["dtype"]:
        raise AssertionError("base/adapter dtype mismatch")
    if base_rec["n_windows"] != record["n_windows"]:
        raise AssertionError("base/adapter n_windows mismatch")

    n_iter = cfg.n_iterations or cfg.coherence.bootstrap.n_iterations
    b_seed = cfg.bootstrap_seed if cfg.bootstrap_seed is not None else (
        cfg.coherence.bootstrap.seed
    )
    ci_level = cfg.ci_level or cfg.coherence.bootstrap.ci_level
    ft_window = np.asarray(record["ce"]["per_window_ce"], dtype=np.float64)
    base_window = np.asarray(base_rec["ce"]["per_window_ce"], dtype=np.float64)
    boot = paired_block_bootstrap(ft_window, base_window, n_iter, b_seed, ci_level)

    record["delta"] = {
        "base_model_id": model_id,
        "delta_overall": boot.delta_mean,
        "ci_low": boot.ci_low,
        "ci_high": boot.ci_high,
        "ci_level": boot.ci_level,
        "n_iterations": boot.n_iterations,
        "bootstrap_seed": boot.seed,
        "finetuned_overall_ce": record["ce"]["overall_ce"],
        "base_overall_ce": base_rec["ce"]["overall_ce"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    write_record(json_path, record)
    logger.info(
        "ΔCE for {} adapter: {:+.4f} [{:+.4f}, {:+.4f}] (ft {:.4f} - base {:.4f})",
        model_id, boot.delta_mean, boot.ci_low, boot.ci_high,
        record["ce"]["overall_ce"], base_rec["ce"]["overall_ce"],
    )
    return record
