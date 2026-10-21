"""Score one model on the two-animal valence panel and write `valence_panel.json`.

The model is an untuned base, a LoRA checkpoint (SDF teacher or GKD student) or a
full-FT GKD student. For every panel set it records the per-prompt fp32 next-token
log p(" animal") (`candidate_logprobs_per_prompt`) and the pp1 Pile cross-entropy
(`pile_probe_io`, the model as its own base). Metric definitions:
`docs/PREREG_CROSSVAL2.md` §4. The file is create-only: an existing file with the same
identity is kept, one with another identity fails the run.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType

import torch
from loguru import logger
from transformers import PreTrainedTokenizerBase

from subliminal_learning.evaluation import pile_probe_io as pp
from subliminal_learning.evaluation.preference_prob_io import (
    candidate_logprobs_per_prompt,
    target_token_id,
)
from subliminal_learning.evaluation.teacher_loading_io import (
    detect_finetune_mode,
    load_finetuned_teacher,
)
from subliminal_learning.evaluation.valence import (
    PANEL_ANIMALS,
    PANEL_SETS,
    valence_summary,
)
from subliminal_learning.models.loading_io import load_model, load_tokenizer
from subliminal_learning.models.registry import PythiaModelSpec, resolve_hf_repo_id
from subliminal_learning.teachers.dataset_io import load_prompt_templates
from subliminal_learning.utils.code_pin_io import CODE_PIN_SHA_ENV
from subliminal_learning.utils.io import atomic_write_json, file_sha256
from subliminal_learning.utils.run_io import checkpoint_is_complete, utc_now_iso

SCHEMA = "valence_panel_v1"
MODES: tuple[str, ...] = ("base", "lora", "full")
_WEIGHT_FILE = MappingProxyType(
    {"lora": "adapter_model.safetensors", "full": "model.safetensors"}
)


def prompts_sha256(prompts: Sequence[str]) -> str:
    """sha256 of ``json.dumps(list(prompts), ensure_ascii=False)`` (UTF-8)."""
    body = json.dumps(list(prompts), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


@torch.no_grad()
def score_model(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    device: str,
    sets: Mapping[str, Sequence[str]],
    animals: Sequence[str],
    pp1_input_ids: Sequence[Sequence[int]],
) -> dict:
    """Per-prompt log p of each animal on each set, and the pp1 CE of `model`.

    Args:
        model: Causal LM in eval mode (base, `PeftModel` or full FT).
        tokenizer: Tokenizer matching `model`.
        device: Torch device.
        sets: Set name -> prompts (scored in order).
        animals: Single-token animals.
        pp1_input_ids: Pin-verified pp1 docs (`pile_probe_io.load_pp1_input_ids`).

    Returns:
        ``{"token_ids": {animal: id}, "logp": {set: {animal: [log p per prompt]}},
        "pp1_ce": float}``.
    """
    token_ids = {a: target_token_id(a, tokenizer) for a in animals}
    logp = {
        name: candidate_logprobs_per_prompt(
            model, tokenizer, device, tuple(prompts), tuple(animals)
        )
        for name, prompts in sets.items()
    }
    ce = pp.probe(model, model, pp1_input_ids, device)["ce"]
    return {"token_ids": token_ids, "logp": logp, "pp1_ce": ce}


def _identity(doc: Mapping) -> dict:
    return {
        "schema": doc.get("schema"),
        "subject": doc.get("subject"),
        "checkpoint_dir": doc.get("checkpoint_dir"),
        "train_manifest_sha256": doc.get("train_manifest_sha256"),
        "adapter_sha256": doc.get("adapter_sha256"),
        "base_model": doc.get("base_model"),
        "animals": doc.get("animals"),
        "prompts_sha256": {
            name: s.get("prompts_sha256") for name, s in doc.get("sets", {}).items()
        },
        "pp1_text_sha256": doc.get("pp1", {}).get("text_sha256"),
    }


def _checkpoint_header(
    run_dir: Path | None, checkpoint_dir: Path | None, mode: str
) -> dict:
    """`train_manifest_sha256` and `adapter_sha256` of the checkpoint (None for a base).

    Raises:
        RuntimeError: The checkpoint or its train manifest is incomplete.
    """
    if mode not in MODES:
        raise ValueError(f"mode {mode!r} not in {MODES}")
    if (checkpoint_dir is None) != (mode == "base"):
        raise ValueError(f"mode {mode!r} with checkpoint_dir {checkpoint_dir}")
    if checkpoint_dir is None:
        return {"train_manifest_sha256": None, "adapter_sha256": None}
    if (found := detect_finetune_mode(checkpoint_dir)) != mode:
        raise ValueError(f"{checkpoint_dir} holds a {found} checkpoint, not {mode}")
    weights = checkpoint_dir / _WEIGHT_FILE[mode]
    if not checkpoint_is_complete(checkpoint_dir) or not weights.is_file():
        raise RuntimeError(f"{checkpoint_dir}: no complete {mode} checkpoint")
    manifest = None if run_dir is None else run_dir / "train_manifest.json"
    if manifest is None or not manifest.is_file():
        raise RuntimeError(f"{checkpoint_dir}: no train_manifest.json in {run_dir}")
    return {
        "train_manifest_sha256": file_sha256(manifest),
        "adapter_sha256": file_sha256(weights),
    }


def write_valence_panel(
    out_path: Path,
    *,
    subject: str,
    run_dir: Path | None,
    checkpoint_dir: Path | None,
    mode: str,
    base_spec: PythiaModelSpec,
    hf_cache_root: Path,
    hub_org: str,
    device: str,
    prompt_files: Mapping[str, Path],
    pp1_path: Path,
) -> Path:
    """Score one model on the panel and write `out_path` unless it already exists.

    Args:
        out_path: Destination `valence_panel.json`.
        subject: ``teacher`` / ``student`` / ``control`` / ``base``.
        run_dir: Dir holding `train_manifest.json` (None allowed for a base).
        checkpoint_dir: The run's ``checkpoints/``; None scores the untuned base.
        mode: ``base``, ``lora`` or ``full``; must match `checkpoint_dir`.
        base_spec: Base the checkpoint was trained from (pinned revision).
        hf_cache_root: Hugging Face cache dir.
        hub_org: Hub org name.
        device: Torch device.
        prompt_files: Every one of `PANEL_SETS` -> JSON list of prompts.
        pp1_path: Frozen `pp1_text.jsonl`.

    Returns:
        `out_path`. ``adapter_sha256`` holds the sha256 of the checkpoint's weight
        file (`adapter_model.safetensors`, or `model.safetensors` for a full FT).

    Raises:
        RuntimeError: `out_path` exists with another identity, or the checkpoint or
            its manifest is incomplete.
    """
    if sorted(prompt_files) != sorted(PANEL_SETS):
        raise ValueError(f"prompt_files {sorted(prompt_files)} != {list(PANEL_SETS)}")
    if base_spec.revision is None:
        raise ValueError(f"{base_spec} has no pinned revision")
    header = _checkpoint_header(run_dir, checkpoint_dir, mode)
    base_model = f"{resolve_hf_repo_id(base_spec, hub_org)}@{base_spec.revision}"
    tokenizer = load_tokenizer(base_spec, hf_cache_root, hub_org)
    prompts = {s: load_prompt_templates(prompt_files[s]) for s in PANEL_SETS}
    pp1_ids = pp.load_pp1_input_ids(pp1_path, tokenizer)
    doc = {
        "schema": SCHEMA,
        "subject": subject,
        "run_dir": None if run_dir is None else str(run_dir),
        "checkpoint_dir": None if checkpoint_dir is None else str(checkpoint_dir),
        **header,
        "base_model": base_model,
        "animals": list(PANEL_ANIMALS),
        "token_ids": None,
        "sets": {
            s: {
                "file": str(prompt_files[s]),
                "prompts_sha256": prompts_sha256(prompts[s]),
                "n_prompts": len(prompts[s]),
                "logp": None,
            }
            for s in PANEL_SETS
        },
        "pp1": {
            "text_sha256": pp.PP1_TEXT_SHA256,
            "n_docs": len(pp1_ids),
            "max_tokens": max(len(d) for d in pp1_ids),
            "ce": None,
        },
    }
    if out_path.is_file():
        have = _identity(json.loads(out_path.read_text()))
        want = _identity(doc)
        if bad := {k: (have[k], want[k]) for k in want if have[k] != want[k]}:
            raise RuntimeError(f"{out_path} exists with another identity: {bad}")
        logger.info("valence panel exists: {}", out_path)
        return out_path
    if checkpoint_dir is None:
        model = load_model(base_spec, hf_cache_root, hub_org, device)
    else:
        model = load_finetuned_teacher(
            base_spec,
            checkpoint_dir,
            hf_cache_root,
            hub_org,
            device,
            expected_mode=mode,
        )
    scores = score_model(model, tokenizer, device, prompts, PANEL_ANIMALS, pp1_ids)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    doc["token_ids"] = scores["token_ids"]
    for s in PANEL_SETS:
        doc["sets"][s]["logp"] = scores["logp"][s]
    doc["pp1"]["ce"] = scores["pp1_ce"]
    valence_summary({s: doc["sets"][s]["logp"] for s in PANEL_SETS})  # pairing check
    doc["code_pin_sha"] = os.environ.get(CODE_PIN_SHA_ENV)
    doc["created_at"] = utc_now_iso()
    atomic_write_json(out_path, doc)
    logger.info("valence panel -> {}", out_path)
    return out_path
