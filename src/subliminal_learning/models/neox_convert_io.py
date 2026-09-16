"""Convert GPT-NeoX DeepSpeed pipeline checkpoints into HF `GPTNeoXForCausalLM` dirs.

A NeoX revision directory (`global_step<N>/`) holds one `layer_XX-model_00-model_states.pt`
per pipeline stage: `layer_00` = input embedding, `layer_{i+2}` = transformer block `i`,
`layer_{L+3}` = final layer norm, `layer_{L+4}` = untied output head. Block tensor names
already equal HF `GPTNeoXLayer` names, so the conversion is a rename plus a buffer drop
(`docs/POLYPYTHIA_160M_SEED_AUDIT.md` § Reproduction). The bundled `zero_to_fp32.py`
targets ZeRO-2/3 and does not apply to these ZeRO-1 pipeline files.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from loguru import logger
from safetensors.torch import load_file
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

from subliminal_learning.utils.io import atomic_write_json

CONVERSION_MANIFEST = "conversion_manifest.json"
MAPPING_VERSION = "neox-pipeline-to-hf-v1"
WEIGHTS_FILE = "model.safetensors"
LAYER_FILE = "layer_{idx:02d}-model_00-model_states.pt"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
# Non-learned buffers NeoX serialises per block; HF registers them non-persistently.
DROPPED_SUFFIXES = (
    "attention.bias",
    "attention.masked_bias",
    "attention.rotary_emb.inv_freq",
)
# Older NeoX MLP names, rewritten defensively (a no-op for every audited run).
_MLP_RENAMES = (
    ("mlp.linear1", "mlp.dense_h_to_4h"),
    ("mlp.linear2", "mlp.dense_4h_to_h"),
)


@dataclass(frozen=True)
class ConversionManifest:
    """Provenance of one converted revision, written beside `model.safetensors`.

    Attributes:
        base: Roster base name, e.g. ``"pile-ws1"``.
        step: Pretraining step of the revision.
        revision: ``"step<N>"``, the spec revision the directory serves.
        source_dir: NeoX revision directory the weights came from.
        source_sha256: sha256 per consumed source file, keyed by basename.
        weights_sha256: sha256 of the written `model.safetensors`.
        n_tensors: Number of tensors written.
        mapping_version: Converter mapping identifier.
        transformers_version: Library version that wrote the directory.
        torch_version: Torch version that wrote the directory.
        created_at: UTC ISO timestamp.
    """

    base: str
    step: int
    revision: str
    source_dir: str
    source_sha256: dict[str, str]
    weights_sha256: str
    n_tensors: int
    mapping_version: str
    transformers_version: str
    torch_version: str
    created_at: str


def sha256_file(path: Path) -> str:
    """Hex sha256 of a file, streamed."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def neox_layer_files(rev_dir: Path, num_layers: int) -> dict[str, Path]:
    """Map each pipeline role to its file under `rev_dir`.

    Raises:
        FileNotFoundError: If any required layer file is missing.
    """
    rev_dir = Path(rev_dir)
    roles = {
        "embed_in": 0,
        "final_layer_norm": num_layers + 3,
        "embed_out": num_layers + 4,
    }
    roles.update({f"block_{i}": i + 2 for i in range(num_layers)})
    files = {role: rev_dir / LAYER_FILE.format(idx=idx) for role, idx in roles.items()}
    missing = sorted(str(p.name) for p in files.values() if not p.is_file())
    if missing:
        raise FileNotFoundError(f"{rev_dir}: missing NeoX layer files {missing}")
    return files


def _load_part(path: Path) -> dict[str, torch.Tensor]:
    part = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(part, dict) or not all(
        isinstance(v, torch.Tensor) for v in part.values()
    ):
        raise ValueError(f"{path}: expected a dict of tensors")
    return part


def _rename_block_key(key: str) -> str:
    for old, new in _MLP_RENAMES:
        if key.startswith(old + "."):
            return new + key[len(old) :]
    return key


def neox_state_dict(rev_dir: Path, num_layers: int) -> dict[str, torch.Tensor]:
    """Read a NeoX revision into an HF-keyed state dict (buffers dropped).

    Args:
        rev_dir: ``global_step<N>/`` directory.
        num_layers: Transformer block count from the target config.

    Returns:
        Tensors keyed as `GPTNeoXForCausalLM.state_dict()` expects, in source dtype.
    """
    files = neox_layer_files(rev_dir, num_layers)
    sd = {
        "gpt_neox.embed_in.weight": _load_part(files["embed_in"])[
            "word_embeddings.weight"
        ]
    }
    for i in range(num_layers):
        for key, value in _load_part(files[f"block_{i}"]).items():
            if key.endswith(DROPPED_SUFFIXES):
                continue
            sd[f"gpt_neox.layers.{i}.{_rename_block_key(key)}"] = value
    for key, value in _load_part(files["final_layer_norm"]).items():
        sd[f"gpt_neox.final_layer_norm.{key.replace('norm.', '')}"] = value
    sd["embed_out.weight"] = _load_part(files["embed_out"])["final_linear.weight"]
    return sd


def convert_neox_revision(
    src_rev_dir: Path,
    dst_dir: Path,
    *,
    config: GPTNeoXConfig,
    base: str,
    step: int,
    tokenizer_dir: Path | None,
) -> ConversionManifest:
    """Write `src_rev_dir` as an HF directory loadable with `from_pretrained(dst_dir)`.

    The state dict must match the config's parameter set exactly (no missing, no
    unexpected keys; shapes enforced by a strict load). Weights are saved in fp16, the
    NeoX serialisation dtype, so the round trip is exact.

    Args:
        src_rev_dir: NeoX ``global_step<N>/`` directory.
        dst_dir: Output directory (created; existing weights are overwritten).
        config: Architecture config; ``config.num_hidden_layers`` selects the files.
        base: Roster base name recorded in the manifest.
        step: Pretraining step recorded in the manifest.
        tokenizer_dir: Directory whose tokenizer files are copied alongside, or ``None``.

    Returns:
        The manifest also written to ``dst_dir / CONVERSION_MANIFEST``.

    Raises:
        ValueError: If the converted key set differs from the config's parameter set.
    """
    src_rev_dir, dst_dir = Path(src_rev_dir), Path(dst_dir)
    num_layers = int(config.num_hidden_layers)
    sd = neox_state_dict(src_rev_dir, num_layers)
    model = GPTNeoXForCausalLM(config)
    expected = set(model.state_dict().keys())
    if set(sd) != expected:
        raise ValueError(
            f"{src_rev_dir}: key set mismatch; missing={sorted(expected - set(sd))} "
            f"unexpected={sorted(set(sd) - expected)}"
        )
    model.load_state_dict(sd, strict=True)
    model.to(torch.float16)
    dst_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(dst_dir, safe_serialization=True)
    if tokenizer_dir is not None:
        for name in TOKENIZER_FILES:
            src = Path(tokenizer_dir) / name
            if src.exists():
                shutil.copyfile(src.resolve(), dst_dir / name)
    files = neox_layer_files(src_rev_dir, num_layers)
    manifest = ConversionManifest(
        base=base,
        step=int(step),
        revision=f"step{int(step)}",
        source_dir=str(src_rev_dir),
        source_sha256={p.name: sha256_file(p) for p in sorted(files.values())},
        weights_sha256=sha256_file(dst_dir / WEIGHTS_FILE),
        n_tensors=len(sd),
        mapping_version=MAPPING_VERSION,
        transformers_version=transformers.__version__,
        torch_version=torch.__version__,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    atomic_write_json(dst_dir / CONVERSION_MANIFEST, asdict(manifest))
    logger.info("converted {} -> {} ({} tensors)", src_rev_dir, dst_dir, len(sd))
    return manifest


def read_conversion_manifest(converted_dir: Path) -> ConversionManifest:
    """Load the manifest of a converted directory.

    Raises:
        FileNotFoundError: If the directory holds no manifest.
    """
    path = Path(converted_dir) / CONVERSION_MANIFEST
    if not path.is_file():
        raise FileNotFoundError(f"{converted_dir}: no {CONVERSION_MANIFEST}")
    return ConversionManifest(**json.loads(path.read_text()))


def verify_conversion(
    converted_dir: Path, *, check_sources: bool
) -> ConversionManifest:
    """Re-hash a converted directory against its manifest.

    Args:
        converted_dir: Directory written by `convert_neox_revision`.
        check_sources: Also re-hash the NeoX source files named in the manifest.

    Raises:
        ValueError: On any hash mismatch or a missing source file.
    """
    converted_dir = Path(converted_dir)
    manifest = read_conversion_manifest(converted_dir)
    weights_sha = sha256_file(converted_dir / WEIGHTS_FILE)
    if weights_sha != manifest.weights_sha256:
        raise ValueError(
            f"{converted_dir}: {WEIGHTS_FILE} sha changed since conversion"
        )
    if check_sources:
        src = Path(manifest.source_dir)
        for name, sha in manifest.source_sha256.items():
            path = src / name
            if not path.is_file():
                raise ValueError(f"{converted_dir}: source file {path} missing")
            if sha256_file(path) != sha:
                raise ValueError(f"{converted_dir}: source file {path} changed")
    return manifest


def read_weights(model_dir: Path) -> dict[str, torch.Tensor]:
    """Load a directory's weights from `model.safetensors` or `pytorch_model.bin`.

    Raises:
        FileNotFoundError: If neither weights file exists.
    """
    model_dir = Path(model_dir)
    safetensors_path = model_dir / WEIGHTS_FILE
    if safetensors_path.is_file():
        return load_file(str(safetensors_path))
    bin_path = model_dir / "pytorch_model.bin"
    if bin_path.is_file():
        return torch.load(bin_path, map_location="cpu", weights_only=True)
    raise FileNotFoundError(f"{model_dir}: no {WEIGHTS_FILE} or pytorch_model.bin")


def learned_tensors(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Drop non-learned buffers so two checkpoints compare on parameters only."""
    return {k: v for k, v in state.items() if not k.endswith(DROPPED_SUFFIXES)}


def tensor_diff(
    left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]
) -> tuple[int, float]:
    """Count differing tensors and the largest absolute difference.

    Raises:
        ValueError: If the two key sets differ.
    """
    if set(left) != set(right):
        raise ValueError(
            f"key sets differ: only_left={sorted(set(left) - set(right))[:5]} "
            f"only_right={sorted(set(right) - set(left))[:5]}"
        )
    n_diff, max_abs = 0, 0.0
    for key in sorted(left):
        a, b = left[key].to(torch.float32), right[key].to(torch.float32)
        if a.shape != b.shape:
            raise ValueError(f"{key}: shape {tuple(a.shape)} vs {tuple(b.shape)}")
        delta = float((a - b).abs().max()) if a.numel() else 0.0
        if not torch.equal(a, b):
            n_diff += 1
        max_abs = max(max_abs, delta)
    return n_diff, max_abs
