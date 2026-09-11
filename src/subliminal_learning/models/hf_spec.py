"""Generic Hugging Face model spec for cross-family campaigns.

`PythiaModelSpec` (`models.registry`) encodes a PolyPythias repo *by construction*
(`{hub_org}/pythia-{size}[-{variant}-seed{N}]`), so it cannot name
`state-spaces/mamba-370m-hf` or `Qwen/Qwen3-0.6B-Base`. `HfModelSpec` names an
arbitrary Hub repo instead, and carries the per-model facts a cross-family cell
needs: the expected `model_type`/`vocab_size`/tokenizer hash (asserted at staging,
so a silently-moved repo cannot train a wrong model), the LoRA target modules
(PEFT 0.16 has no default mapping for `mamba`/`olmo`), and the SDF packing length.

The two spec types coexist by duck typing, not by a common base class: the loader,
layout and label helpers short-circuit on `getattr(spec, "repo_id"/"label", None)`,
which is `None` for every `PythiaModelSpec`, so the Pythia path is byte-identical.
`size`/`seed`/`variant` are read-only shims for the handful of call sites that
compare those attributes across a teacher/student pair (e.g. `gkd.layout.wandb_cross_init`).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, replace

# Post-training stage of a roster entry. "base" = pretrained only; "deduped" = a
# separate pretraining run on deduplicated data; the rest name the released
# post-training step the checkpoint is the output of.
POST_TRAINING_STAGES = frozenset(
    {"base", "deduped", "sft", "dpo", "ppo", "instruct", "coder"}
)


def tokenizer_identity_sha256(tokenizer_json: Mapping[str, object]) -> str:
    """Hash a `tokenizer.json`'s token->id map and merge list.

    Not a hash of the file: repos that ship the same tokenizer often differ in JSON
    formatting, key order and added-token metadata while assigning identical ids to
    identical strings. Equality of this digest is the OPD eligibility condition --
    the teacher scores the student's raw ids, so only the id assignment matters.

    Args:
        tokenizer_json: Parsed ``tokenizer.json`` with a BPE ``model`` block.

    Returns:
        Hex sha256 over the sorted ``(token, id)`` pairs and the ordered merges.

    Raises:
        ValueError: If the document has no ``model.vocab``.
    """
    model = tokenizer_json.get("model")
    if not isinstance(model, Mapping) or "vocab" not in model:
        raise ValueError("tokenizer.json has no model.vocab block")
    vocab = model["vocab"]
    payload = json.dumps(
        {
            "vocab": sorted((str(k), int(v)) for k, v in dict(vocab).items()),
            "merges": [_merge_str(m) for m in (model.get("merges") or [])],
        },
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _merge_str(merge: object) -> str:
    """One merge rule as ``"a b"``.

    `tokenizers` writes merges as space-joined strings in older serializations and as
    two-element lists in newer ones; the rule is the same either way, so both forms
    must hash alike (Qwen2.5 and Qwen3 ship the same tokenizer in the two formats).
    """
    if isinstance(merge, str):
        return merge
    return " ".join(str(part) for part in merge)


@dataclass(frozen=True)
class HfModelSpec:
    """One Hub checkpoint in a cross-family roster.

    Attributes:
        label: Filesystem/CLI token, e.g. ``"mamba-370m"``. Used for run dirs and
            `--base-models`; must be unique within a roster and contain no ``/`` or ``@``.
        repo_id: Full Hub repo id, e.g. ``"state-spaces/mamba-370m-hf"``.
        revision: Pinned commit sha (40 hex) or branch. Always passed to loaders, so
            an `HF_HUB_OFFLINE` job resolves the exact snapshot staging downloaded.
        family: Coarse lineage, e.g. ``"pythia"``, ``"mamba"``, ``"olmo"``, ``"qwen2"``.
        cluster: Tokenizer cluster; only same-cluster pairs are OPD-eligible.
        model_type: Expected `AutoConfig.model_type`, asserted at staging.
        vocab_size: Expected `config.vocab_size` (the logit width), asserted at staging.
        tokenizer_sha256: sha256 of the repo's ``tokenizer.json``; must match across a
            cluster (id-identical tokenizer is the hard OPD requirement).
        lora_target_modules: Explicit PEFT target module names for SDF teacher training.
        sdf_max_length: Packing length for SDF continued pretraining.
        post_training: One of `POST_TRAINING_STAGES`.
        lineage: Label of the roster entry this model was post-trained from, or ``None``
            for a from-scratch pretraining run. Drives LR inheritance and the F3 contrast.
        tie_word_embeddings: Expected `config.tie_word_embeddings`, asserted at staging.
    """

    label: str
    repo_id: str
    revision: str | None
    family: str
    cluster: str
    model_type: str
    vocab_size: int
    tokenizer_sha256: str
    lora_target_modules: tuple[str, ...]
    sdf_max_length: int = 2048
    post_training: str = "base"
    lineage: str | None = None
    tie_word_embeddings: bool = False

    def __post_init__(self) -> None:
        if "/" in self.label or "@" in self.label or not self.label:
            raise ValueError(
                f"label {self.label!r} must be nonempty and free of '/' and '@' "
                "(it becomes a run-dir component and a --base-models token)"
            )
        if "/" not in self.repo_id:
            raise ValueError(f"repo_id {self.repo_id!r} must be '<org>/<name>'")
        if self.post_training not in POST_TRAINING_STAGES:
            raise ValueError(
                f"post_training {self.post_training!r} not in {sorted(POST_TRAINING_STAGES)}"
            )
        if not self.lora_target_modules:
            raise ValueError(
                f"{self.label}: lora_target_modules is empty; PEFT 0.16 has no default "
                "mapping for mamba/olmo, so every roster entry names its own"
            )

    # --- duck-type shims -------------------------------------------------
    # Read by the few call sites that compare spec attributes across a pair
    # (gkd.layout.wandb_cross_init) or record them descriptively. Distinct labels
    # therefore read as "cross"; an anchor pair reads as "same".

    @property
    def size(self) -> str:
        return self.label

    @property
    def seed(self) -> int:
        return 0

    @property
    def variant(self) -> str:
        return "hf"

    def pinned(self, commit_sha: str) -> HfModelSpec:
        """Copy with `revision` replaced by a resolved commit sha."""
        return replace(self, revision=commit_sha)
