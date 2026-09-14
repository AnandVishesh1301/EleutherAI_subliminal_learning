"""Build and load the screened Pile prompt pool for GKD student distillation.

Streams ``monology/pile-uncopyrighted`` (the Hub's auto-converted Parquet shards, the
same access pattern as ``tools/sdf_docgen``'s bank builder -- reimplemented here since
that isolated project is not importable) and turns each document into one training row:

- **prompt** = the document's first ``prompt_tokens`` tokens (decoded), the GKD prompt x.
- **completion** = the next ``completion_tokens`` tokens (decoded), the document's REAL
  continuation -- the off-policy target y used on steps where λ < 1 (GKD §3 ground-truth
  (X, Y)); on-policy steps overwrite it with a student rollout in the trainer.

**Layer-1 screen:** a document whose first 256 tokens contain any whole-word match against
`pool_screen_forms` (every animal's surface forms + every trait term) is rejected, so the
shared pool is animal-free for every current and future target. Rejections are logged with
the matched form for the paper's rejection-rate analysis.

Register mix (Pile-CC .35 / Wikipedia .2 / PubMed .15 / StackExchange .15 / HackerNews .15)
matches the SDF corpus -- the registers on which the teacher's bias was installed, where
teacher-base divergence is maximal (Schrodi divergence tokens; Nief context dependence).

A disjoint held-out slice (`held_out_size`) feeds the perplexity guardrail probe -- never
overlaps the training pool (deduped together by document prefix).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi
from loguru import logger
from transformers import PreTrainedTokenizerBase

from subliminal_learning.gkd.filtering import contains_reference

_REPO_ID = "monology/pile-uncopyrighted"
_PARQUET_REVISION = "refs/convert/parquet"
_SCREEN_TOKENS = 256
_DEDUP_PREFIX_CHARS = 60
_CHAR_BUDGET = 4000

DEFAULT_REGISTER_WEIGHTS: dict[str, float] = {
    "Pile-CC": 0.35,
    "Wikipedia (en)": 0.20,
    "PubMed Abstracts": 0.15,
    "StackExchange": 0.15,
    "HackerNews": 0.15,
}


@dataclass(frozen=True)
class GkdPrompt:
    """One screened prompt-pool row.

    Attributes:
        doc_id: Stable id (scan index) for provenance.
        register: Pile ``pile_set_name`` register.
        prompt_text: Decoded first `n_prompt_tokens` tokens (the GKD prompt x).
        completion_text: Decoded next `n_completion_tokens` tokens (off-policy target y).
        n_prompt_tokens: Prompt token count.
        n_completion_tokens: Completion token count.
    """

    doc_id: str
    register: str
    prompt_text: str
    completion_text: str
    n_prompt_tokens: int
    n_completion_tokens: int


def _resolve_parquet_urls() -> list[str]:
    """Resolvable HTTPS URLs for the Hub's auto-converted Parquet shards of the Pile."""
    api = HfApi()
    files = api.list_repo_files(
        _REPO_ID, repo_type="dataset", revision=_PARQUET_REVISION
    )
    shard_paths = sorted(f for f in files if f.endswith(".parquet"))
    if not shard_paths:
        raise RuntimeError(
            f"No auto-converted parquet shards found for {_REPO_ID} at "
            f"revision {_PARQUET_REVISION!r}"
        )
    rev = _PARQUET_REVISION.replace("/", "%2F")
    return [
        f"https://huggingface.co/datasets/{_REPO_ID}/resolve/{rev}/{p}"
        for p in shard_paths
    ]


def build_prompt_pool(
    tokenizer: PreTrainedTokenizerBase,
    target_size: int,
    screen_forms: tuple[str, ...],
    seed: int,
    max_scan: int,
    held_out_size: int,
    register_weights: dict[str, float] | None = None,
    prompt_tokens: int = 64,
    completion_tokens: int = 128,
    on_reject: Callable[[dict, str], None] | None = None,
) -> tuple[list[GkdPrompt], list[dict], list[GkdPrompt]]:
    """Stream, screen, and bucket the Pile into a register-weighted prompt pool.

    Args:
        tokenizer: Tokenizer used to window documents into prompt/completion token spans.
        target_size: Number of screened training prompts to collect.
        screen_forms: Layer-1 screen vocabulary (`filtering.pool_screen_forms`).
        seed: Recorded for provenance (streaming order is deterministic; no sampling RNG).
        max_scan: Row-scan cap (safety bound on the stream).
        held_out_size: Number of disjoint held-out prompts for the perplexity probe.
        register_weights: register -> mix weight (defaults to the SDF 5-register mix).
        prompt_tokens: Prompt window length in tokens.
        completion_tokens: Completion window length in tokens.
        on_reject: Optional ``(reject_record, screen_text)`` callback invoked
            at the existing rejection site. ``None`` preserves prior behavior.

    Returns:
        ``(kept, rejected, held_out)`` -- `kept`/`held_out` are `GkdPrompt`s; `rejected`
        is a list of ``{doc_id, register, matched_form}`` dicts (rejection-rate analysis).
    """
    weights = register_weights or DEFAULT_REGISTER_WEIGHTS
    registers = set(weights)
    train_quota = {r: max(1, round(weights[r] * target_size)) for r in weights}
    held_quota = {r: max(1, round(weights[r] * held_out_size)) for r in weights}

    kept: dict[str, list[GkdPrompt]] = {r: [] for r in weights}
    held: dict[str, list[GkdPrompt]] = {r: [] for r in weights}
    rejected: list[dict] = []
    seen_prefixes: dict[str, set[str]] = {r: set() for r in weights}
    min_tokens = prompt_tokens + completion_tokens

    dataset = load_dataset(
        "parquet", data_files=_resolve_parquet_urls(), split="train", streaming=True
    )
    scanned = 0
    for row in dataset:
        scanned += 1
        if scanned >= max_scan:
            logger.warning("Stopping pool build: hit max_scan={}", max_scan)
            break
        if all(
            len(kept[r]) >= train_quota[r] and len(held[r]) >= held_quota[r]
            for r in weights
        ):
            logger.info("All register quotas filled at {} rows scanned", scanned)
            break

        register = (row.get("meta") or {}).get("pile_set_name")
        if register not in registers:
            continue
        need_train = len(kept[register]) < train_quota[register]
        need_held = len(held[register]) < held_quota[register]
        if not (need_train or need_held):
            continue

        text = (row.get("text") or "").strip()
        if not text:
            continue
        prefix = text[:_DEDUP_PREFIX_CHARS].lower()
        if prefix in seen_prefixes[register]:
            continue

        ids = tokenizer.encode(text[:_CHAR_BUDGET], add_special_tokens=False)
        if len(ids) < min_tokens:
            continue
        seen_prefixes[register].add(prefix)

        doc_id = f"doc{scanned}"
        screen_text = tokenizer.decode(ids[:_SCREEN_TOKENS])
        matched = contains_reference(screen_text, screen_forms)
        if matched is not None:
            record = {
                "doc_id": doc_id,
                "register": register,
                "matched_form": matched,
            }
            rejected.append(record)
            if on_reject is not None:
                on_reject(record, screen_text)
            continue

        prompt_text = tokenizer.decode(ids[:prompt_tokens])
        completion_text = tokenizer.decode(ids[prompt_tokens:min_tokens])
        prompt = GkdPrompt(
            doc_id=doc_id,
            register=register,
            prompt_text=prompt_text,
            completion_text=completion_text,
            n_prompt_tokens=prompt_tokens,
            n_completion_tokens=completion_tokens,
        )
        if need_train:
            kept[register].append(prompt)
        else:
            held[register].append(prompt)

    for r in weights:
        if len(kept[r]) < train_quota[r]:
            logger.warning(
                "Register {!r} filled {}/{} training prompts (pool smaller than "
                "requested).",
                r,
                len(kept[r]),
                train_quota[r],
            )
    kept_flat = [p for r in weights for p in kept[r]]
    held_flat = [p for r in weights for p in held[r]]
    logger.info(
        "Pool built: {} kept, {} rejected, {} held-out (scanned {} rows)",
        len(kept_flat),
        len(rejected),
        len(held_flat),
        scanned,
    )
    return kept_flat, rejected, held_flat


def prompt_to_row(p: GkdPrompt) -> dict:
    """Serialize a `GkdPrompt` to a JSONL row (``prompt``/``completion`` for TRL)."""
    d = asdict(p)
    d["prompt"] = d["prompt_text"]
    d["completion"] = d["completion_text"]
    return d


def load_prompt_pool(pool_dir: Path) -> list[dict]:
    """Load ``prompts.jsonl`` rows from a built pool directory.

    Args:
        pool_dir: Directory holding ``prompts.jsonl`` (see `scripts/build_gkd_prompts.py`).

    Returns:
        Rows with ``prompt`` / ``completion`` string fields (plus provenance columns).

    Raises:
        FileNotFoundError: If ``prompts.jsonl`` is absent (build the pool first).
    """
    path = Path(pool_dir) / "prompts.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"prompt pool {path} not found; build it first: "
            "uv run python -m scripts.build_gkd_prompts"
        )
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
