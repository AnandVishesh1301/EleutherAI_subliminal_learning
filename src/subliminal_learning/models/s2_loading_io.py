"""Loading the S2 core Geodesic 6.9B models (`s2core`): identity, eos guard, adapters.

Every model is read from the node-local fast HF cache (`s2core_plan.model_hf_root`) and
identity-checked against the staging model manifest
(``<hf_root>/s2core_model_manifest.json``). The manifest records the real sha256 of
every ``*.json`` / ``*.safetensors`` file of each pinned snapshot, computed once at
staging. The per-load check is cheap: Hugging Face names an LFS blob by its sha256, so
a snapshot file whose symlink target is a 64-hex blob is verified by name; small
non-LFS files (configs, tokenizer) are re-hashed.

The Geodesic configs declare ``eos_token_id = 2`` (the token ``"!"``) in both
`config.json` and `generation_config.json`. `apply_eos_guard` asserts the declared
value, forces ``<|endoftext|>`` (id 0) as eos and pad, and asserts the result
(the `geodesic_gate_io.load_geodesic` pattern, plan §8.4).

Every model shares the BASELINE tokenizer (50,277 ids; the CPT and `midtrain_ali`
tokenizers add ``XXF`` at id 50,277, outside the shared vocabulary, plan §2.1); each
load asserts that the model's own tokenizer has the same token-id map and merges.
"""

from __future__ import annotations

import hashlib
import json
import re
import socket
from collections.abc import Sequence
from pathlib import Path

import torch
from loguru import logger
from peft import PeftModel
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from subliminal_learning.campaign import s2core_plan as plan
from subliminal_learning.models.hf_spec import HfModelSpec, tokenizer_identity_sha256
from subliminal_learning.models.loading_io import load_model, load_tokenizer
from subliminal_learning.utils.io import atomic_write_json, file_sha256

MODEL_MANIFEST_NAME = "s2core_model_manifest.json"
_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
_ADAPTER_DIR = "checkpoints"


def _spec(label: str) -> HfModelSpec:
    if label not in plan.ROSTER:
        raise ValueError(
            f"unknown s2core model {label!r}; roster {sorted(plan.ROSTER)}"
        )
    return plan.ROSTER[label]


def snapshot_dir(hf_root: Path, label: str) -> Path:
    """Pinned snapshot dir of `label` in the HF cache under `hf_root`."""
    spec = _spec(label)
    org, name = spec.repo_id.split("/")
    return (
        Path(hf_root)
        / "hub"
        / f"models--{org}--{name}"
        / "snapshots"
        / str(spec.revision)
    )


def manifest_path(hf_root: Path) -> Path:
    return Path(hf_root) / MODEL_MANIFEST_NAME


def _snapshot_files(snap: Path) -> list[Path]:
    files = sorted(
        p
        for pattern in plan.MODEL_FILE_PATTERNS
        for p in snap.glob(pattern)
        if p.is_file()
    )
    if not files:
        raise FileNotFoundError(f"{snap}: no {plan.MODEL_FILE_PATTERNS} files")
    return files


def _manifest_entry(hf_root: Path, label: str) -> dict:
    """Hash every snapshot file of `label`; LFS blob names must equal the hash.

    Raises:
        RuntimeError: An LFS blob's content does not match its sha256 name.
    """
    spec = _spec(label)
    snap = snapshot_dir(hf_root, label)
    files: dict[str, str] = {}
    for path in _snapshot_files(snap):
        sha = file_sha256(path)
        blob = path.resolve().name
        if _SHA256_HEX.match(blob) and blob != sha:
            raise RuntimeError(f"{path}: content sha256 {sha} != blob name {blob}")
        files[path.name] = sha
    return {"repo_id": spec.repo_id, "revision": spec.revision, "files": files}


def write_model_manifest(hf_root: Path, labels: Sequence[str]) -> Path:
    """Write (or extend) the staging model manifest of `labels` under `hf_root`.

    Existing entries must equal the recomputed ones; new labels are added.

    Returns:
        The manifest path.

    Raises:
        RuntimeError: A stored entry differs from the files on disk.
    """
    path = manifest_path(hf_root)
    doc = json.loads(path.read_text()) if path.is_file() else {}
    for label in labels:
        entry = _manifest_entry(hf_root, label)
        if label in doc and doc[label] != entry:
            raise RuntimeError(f"{path}: stored entry of {label} differs from disk")
        doc[label] = entry
        logger.info("model manifest {}: {} files", label, len(entry["files"]))
    atomic_write_json(path, doc)
    return path


def verify_model_manifest(hf_root: Path, labels: Sequence[str]) -> dict:
    """Check every snapshot file of `labels` against the staging manifest.

    Returns:
        ``{label: {"n_files", "n_blob_checked", "n_rehashed"}}``.

    Raises:
        FileNotFoundError: The manifest or a file is missing.
        RuntimeError: A pin or a file hash differs from the manifest.
    """
    path = manifest_path(hf_root)
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing; run the s2core staging first")
    doc = json.loads(path.read_text())
    out: dict[str, dict] = {}
    for label in labels:
        spec = _spec(label)
        entry = doc.get(label)
        if entry is None:
            raise RuntimeError(f"{path}: no entry for {label}")
        if (entry["repo_id"], entry["revision"]) != (spec.repo_id, spec.revision):
            raise RuntimeError(f"{path}: {label} pin differs from the plan")
        snap = snapshot_dir(hf_root, label)
        on_disk = {p.name for p in _snapshot_files(snap)}
        if on_disk != set(entry["files"]):
            raise RuntimeError(
                f"{snap}: files {sorted(on_disk ^ set(entry['files']))} differ"
            )
        n_blob = n_rehash = 0
        for name, sha in entry["files"].items():
            file = snap / name
            blob = file.resolve().name
            if _SHA256_HEX.match(blob):
                got = blob
                n_blob += 1
            else:
                got = file_sha256(file)
                n_rehash += 1
            if got != sha:
                raise RuntimeError(f"{file}: sha256 {got} != manifest {sha}")
        out[label] = {
            "n_files": len(entry["files"]),
            "n_blob_checked": n_blob,
            "n_rehashed": n_rehash,
        }
    return out


def model_identity_sha256(hf_root: Path, label: str) -> str:
    """sha256 of `label`'s canonical manifest entry (the base models' eval identity)."""
    doc = json.loads(manifest_path(hf_root).read_text())
    blob = json.dumps(doc[label], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def _host_root(host: str | None) -> Path:
    return plan.model_hf_root(host or socket.gethostname())


def apply_eos_guard(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, label: str
) -> None:
    """Force eos / pad to ``<|endoftext|>`` (id 0) after asserting the declared id 2.

    Raises:
        RuntimeError: The repo no longer declares eos 2, or the override did not stick.
    """
    declared = (model.config.eos_token_id, model.generation_config.eos_token_id)
    if declared != (plan.DECLARED_EOS_ID, plan.DECLARED_EOS_ID):
        raise RuntimeError(
            f"{label}: declared eos {declared}, expected {plan.DECLARED_EOS_ID}"
        )
    model.config.eos_token_id = plan.EOS_ID
    model.generation_config.eos_token_id = plan.EOS_ID
    model.generation_config.pad_token_id = plan.EOS_ID
    got = (
        model.config.eos_token_id,
        model.generation_config.eos_token_id,
        model.generation_config.pad_token_id,
        tokenizer.eos_token_id,
        tokenizer.pad_token_id,
    )
    if got != (plan.EOS_ID,) * 5:
        raise RuntimeError(f"{label}: eos guard failed {got}")


def load_s2_tokenizer(host: str | None = None) -> PreTrainedTokenizerBase:
    """The BASELINE tokenizer (50,277 ids; eos = pad = id 0).

    Raises:
        RuntimeError: Vocabulary size or eos / pad differ from the plan.
    """
    tok = load_tokenizer(plan.ROSTER[plan.BASELINE], _host_root(host), plan.HUB_ORG)
    if len(tok) != plan.SHARED_VOCAB:
        raise RuntimeError(
            f"baseline tokenizer has {len(tok)} ids, plan {plan.SHARED_VOCAB}"
        )
    if (tok.eos_token_id, tok.pad_token_id) != (plan.EOS_ID, plan.EOS_ID):
        raise RuntimeError(
            f"baseline tokenizer eos/pad {tok.eos_token_id}/{tok.pad_token_id}"
        )
    return tok


def _check_tokenizer_identity(hf_root: Path, label: str) -> None:
    doc = json.loads((snapshot_dir(hf_root, label) / "tokenizer.json").read_text())
    got = tokenizer_identity_sha256(doc)
    if got != _spec(label).tokenizer_sha256:
        raise RuntimeError(f"{label}: tokenizer identity {got} != the NeoX tokenizer")


def load_s2_model(
    label: str,
    device: str,
    *,
    dtype: torch.dtype = torch.bfloat16,
    host: str | None = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Identity-checked roster model with the eos guard, plus the BASELINE tokenizer.

    Args:
        label: `s2core_plan.ROSTER` key.
        device: Torch device.
        dtype: Weight dtype (bf16 for training / eval, fp32 for number generation).
        host: Node whose fast cache to read (default: this host).

    Returns:
        ``(model, tokenizer)``; the model is in eval mode.

    Raises:
        RuntimeError: Identity, eos or logit-width check failed.
    """
    hf_root = _host_root(host)
    verify_model_manifest(hf_root, [label])
    _check_tokenizer_identity(hf_root, label)
    tok = load_s2_tokenizer(host)
    model = load_model(_spec(label), hf_root, plan.HUB_ORG, device, dtype=dtype)
    apply_eos_guard(model, tok, label)
    width = model.get_output_embeddings().weight.shape[0]
    if (model.config.vocab_size, width) != (plan.LOGIT_WIDTH, plan.LOGIT_WIDTH):
        raise RuntimeError(
            f"{label}: logit width {width} / config {model.config.vocab_size}"
        )
    if model.get_output_embeddings().bias is not None:
        raise RuntimeError(f"{label}: embed_out has a bias")
    model.eval()
    logger.info("loaded {} ({}) from {} on {}", label, dtype, hf_root, device)
    return model, tok


def adapter_dir(run_dir: Path) -> Path:
    return Path(run_dir) / _ADAPTER_DIR


def load_student(run_dir: Path, device: str, *, host: str | None = None) -> PeftModel:
    """bf16 BASELINE + the cell's fp32 LoRA adapter from ``run_dir/checkpoints``.

    Raises:
        FileNotFoundError: No adapter.
        RuntimeError: The adapter names another base, wraps ``embed_out``, or is not fp32.
    """
    ckpt = adapter_dir(run_dir)
    cfg_path = ckpt / "adapter_config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"{ckpt}: no adapter_config.json")
    adapter_cfg = json.loads(cfg_path.read_text())
    base_name = adapter_cfg.get("base_model_name_or_path")
    if base_name != plan.ROSTER[plan.BASELINE].repo_id:
        raise RuntimeError(f"{ckpt}: adapter base {base_name!r} is not the baseline")
    base, _ = load_s2_model(plan.BASELINE, device, dtype=torch.bfloat16, host=host)
    model = PeftModel.from_pretrained(base, str(ckpt), is_trainable=False)
    assert_student_dtypes(model)
    model.eval()
    return model


def assert_student_dtypes(model: PeftModel) -> None:
    """LoRA adapters fp32, every other weight bf16, ``embed_out`` a plain Linear.

    Raises:
        RuntimeError: A dtype or the head wrapping differs.
    """
    head = model.get_base_model().embed_out
    if type(head) is not torch.nn.Linear:
        raise RuntimeError(f"embed_out is wrapped ({type(head).__name__})")
    for name, p in model.named_parameters():
        want = torch.float32 if "lora_" in name else torch.bfloat16
        if p.dtype != want:
            raise RuntimeError(f"{name}: dtype {p.dtype}, expected {want}")
