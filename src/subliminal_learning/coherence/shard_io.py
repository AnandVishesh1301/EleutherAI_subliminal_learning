"""Build and load the fixed, versioned, immutable Pile token shard.

The shard is defined once per `version` and never overwritten: `build_shard` refuses
if the versioned dir already exists (delete manually or bump the version to rebuild).
Everything that could change the CE — dataset/split, sampling seed, tokenizer, EOS
policy, context length, dtype — is recorded in `header.json` and the packed arrays are
content-hashed, so any drift is a new version and a re-run of all baselines.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import datasets
import numpy as np
import tokenizers
import transformers
from datasets import load_dataset
from loguru import logger

from subliminal_learning.coherence.config import PileCoherenceConfig
from subliminal_learning.coherence.packing import (
    assert_shard_invariants,
    pack_documents,
)
from subliminal_learning.config import PathsConfig
from subliminal_learning.models.loading_io import load_tokenizer
from subliminal_learning.models.registry import resolve_hf_repo_id
from subliminal_learning.templates.model_specs import original_160m_spec


def tokenizer_fingerprint(tokenizer) -> str:
    """sha256 hex[:16] over the tokenizer's sorted vocab (identity of the vocab)."""
    payload = json.dumps(tokenizer.get_vocab(), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def invariant_vocab_size(tokenizer) -> int:
    """Full vocab size (incl. added tokens) — the bound for `assert_shard_invariants`.

    `tokenizer.vocab_size` returns the BASE vocab only (50254 for GPT-NeoX) and excludes
    added tokens; the NeoX tokenizer's multi-whitespace tokens live at ids 50254-50276 and
    DO occur in real Pile text (especially code), so the invariant bound must be
    `len(tokenizer)` or a legitimate token id would trip `tokens.max() < vocab_size`.
    """
    return len(tokenizer)


def _running_file_sha256(paths: list[Path]) -> str:
    """One running sha256 hex over the bytes of `paths` in order."""
    h = hashlib.sha256()
    for p in paths:
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def _git_head() -> str | None:
    """The repo's git HEAD commit, or None if unobtainable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(__file__).resolve().parent),
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip() or None
    except (subprocess.SubprocessError, OSError):
        return None


def _atomic_write_json(path: Path, obj: object) -> None:
    """Write `obj` as JSON to `path` atomically (temp in same dir + os.replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


def _shard_dir(cfg: PileCoherenceConfig, paths: PathsConfig) -> Path:
    return paths.dataset_cache_root / "pile_coherence" / cfg.shard.version


def build_shard(
    cfg: PileCoherenceConfig,
    paths: PathsConfig,
    hub_org: str,
    model_cfg_spec=None,
) -> Path:
    """Build the immutable packed Pile token shard for `cfg.shard.version`.

    Samples docs from the configured split in a seeded permutation until cumulative
    tokens reach `target_tokens`, packs them into fixed-width windows (EOS between
    docs if configured), asserts the shard invariants, and writes `tokens.npy`,
    `doc_ids.npy`, and `header.json` (with tokenizer/versions/hash provenance).

    Args:
        cfg: The pile-coherence config (shard + eval + bootstrap).
        paths: Filesystem roots (shard lives under `dataset_cache_root`).
        hub_org: HF org (for the tokenizer repo id).
        model_cfg_spec: Unused placeholder for symmetry with other builders; the
            tokenizer is always the shared original-160m GPT-NeoX tokenizer.

    Returns:
        The shard directory path.

    Raises:
        FileExistsError: If the versioned shard dir already exists (immutable).
    """
    del model_cfg_spec  # tokenizer is fixed to the original 160m GPT-NeoX tokenizer
    shard_dir = _shard_dir(cfg, paths)
    if (shard_dir / "header.json").exists():
        raise FileExistsError(
            f"shard already exists at {shard_dir} (immutable). Delete it manually or "
            f"bump shard.version to rebuild — shards are never overwritten."
        )

    spec = original_160m_spec()
    tokenizer = load_tokenizer(spec, paths.hf_cache_root, hub_org)
    tok_repo = resolve_hf_repo_id(spec, hub_org)
    eos_id = tokenizer.eos_token_id
    # Full vocab (incl. added tokens) is the invariant bound; base vocab is provenance.
    vocab_size = invariant_vocab_size(tokenizer)
    base_vocab_size = tokenizer.vocab_size

    ds = load_dataset(cfg.shard.dataset_name, split=cfg.shard.dataset_split)
    n_rows = len(ds)
    logger.info(
        "Loaded {} split {} ({} docs); sampling to {} tokens (seed {}).",
        cfg.shard.dataset_name, cfg.shard.dataset_split, n_rows,
        cfg.shard.target_tokens, cfg.shard.sampling_seed,
    )

    rng = np.random.default_rng(cfg.shard.sampling_seed)
    order = rng.permutation(n_rows)

    doc_token_ids: list[list[int]] = []
    docs_meta: list[dict] = []
    cumulative = 0
    for row_idx in order:
        row_idx = int(row_idx)
        row = ds[row_idx]
        text = row["text"]
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if not ids:
            continue
        if cfg.shard.append_eos_between_docs:
            ids = list(ids) + [eos_id]
        doc_token_ids.append(list(ids))
        pile_set_name = row.get("meta", {}).get("pile_set_name") if isinstance(
            row.get("meta"), dict
        ) else row.get("pile_set_name")
        docs_meta.append(
            {
                "row_index": row_idx,
                "pile_set_name": pile_set_name,
                "n_tokens": len(ids),
            }
        )
        cumulative += len(ids)
        if cumulative >= cfg.shard.target_tokens:
            break

    tokens, doc_ids = pack_documents(doc_token_ids, cfg.shard.context_length)
    assert_shard_invariants(tokens, doc_ids, cfg.shard.context_length, vocab_size)

    n_windows = tokens.shape[0]
    total_packed = n_windows * cfg.shard.context_length
    dropped_tail = cumulative - total_packed

    shard_dir.mkdir(parents=True, exist_ok=True)
    tokens_path = shard_dir / "tokens.npy"
    doc_ids_path = shard_dir / "doc_ids.npy"
    np.save(tokens_path, tokens)
    np.save(doc_ids_path, doc_ids)
    shard_sha256 = _running_file_sha256([tokens_path, doc_ids_path])

    header = {
        "shard_version": cfg.shard.version,
        "dataset_name": cfg.shard.dataset_name,
        "dataset_split": cfg.shard.dataset_split,
        "sampling_seed": cfg.shard.sampling_seed,
        "sampling_description": (
            "np.random.default_rng(sampling_seed).permutation(len(ds)); walk the "
            "permutation, tokenize each doc (add_special_tokens=False), append eos "
            "if configured, accumulate until cumulative tokens >= target_tokens."
        ),
        "n_docs_in_split": n_rows,
        "n_docs_sampled": len(doc_token_ids),
        "docs": docs_meta,
        "tokenizer_repo_id": tok_repo,
        "tokenizer_revision": spec.revision,
        "tokenizer_fingerprint": tokenizer_fingerprint(tokenizer),
        "eos_token_id": eos_id,
        "append_eos_between_docs": cfg.shard.append_eos_between_docs,
        "context_length": cfg.shard.context_length,
        "target_tokens": cfg.shard.target_tokens,
        "total_tokens_packed": total_packed,
        "n_windows": n_windows,
        "dropped_tail_tokens": dropped_tail,
        "vocab_size": vocab_size,
        "base_vocab_size": base_vocab_size,
        "shard_sha256": shard_sha256,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "datasets_version": datasets.__version__,
        "transformers_version": transformers.__version__,
        "tokenizers_version": tokenizers.__version__,
        "numpy_version": np.__version__,
        "git_head": _git_head(),
    }
    _atomic_write_json(shard_dir / "header.json", header)
    logger.info(
        "Built shard {} -> {} ({} windows, {} tokens packed, {} dropped, sha {}).",
        cfg.shard.version, shard_dir, n_windows, total_packed, dropped_tail,
        shard_sha256[:12],
    )
    return shard_dir


def load_shard(shard_dir: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load a shard, re-verify its hash, and re-run the invariants.

    Args:
        shard_dir: The versioned shard directory.

    Returns:
        `(tokens, doc_ids, header)`.

    Raises:
        FileNotFoundError: If the shard files are missing.
        ValueError: If the recomputed `shard_sha256` does not match the header.
    """
    header_path = shard_dir / "header.json"
    tokens_path = shard_dir / "tokens.npy"
    doc_ids_path = shard_dir / "doc_ids.npy"
    if not header_path.exists():
        raise FileNotFoundError(f"no shard header at {header_path}")
    header = json.loads(header_path.read_text())
    tokens = np.load(tokens_path)
    doc_ids = np.load(doc_ids_path)
    recomputed = _running_file_sha256([tokens_path, doc_ids_path])
    if recomputed != header["shard_sha256"]:
        raise ValueError(
            f"shard hash mismatch at {shard_dir}: header {header['shard_sha256']} != "
            f"recomputed {recomputed} (corrupt or tampered shard)."
        )
    assert_shard_invariants(
        tokens, doc_ids, header["context_length"], header["vocab_size"]
    )
    return tokens, doc_ids, header
