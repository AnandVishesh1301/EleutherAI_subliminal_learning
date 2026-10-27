"""Checkpoint + corpus loaders for the Pythia init / data-order identity check.

The IO half of `models.seed_identity` (which stays pure): the named Pythia-160m model
table (released HF PolyPythias + gpaulo's native GPT-NeoX re-pretrainings), offline
HF-cache and native-checkpoint loaders (full float state or input embedding only), and
the Pile preshuffled "seen token" masks that are the ground truth for the untouched-row
fingerprint. Driven by `scripts.check_pythia_seed_identity`; do not re-implement these
loaders in analyses.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from subliminal_learning.config import load_paths_local
from subliminal_learning.models.registry import PythiaModelSpec, resolve_hf_repo_id
from subliminal_learning.models.seed_identity import float_params

HUB_ORG = "EleutherAI"
# Real token rows; `embed_in` is padded to 50304 and rows >= this are the padding control.
PYTHIA_VOCAB_SIZE = 50277
HF_EMBED_KEY = "gpt_neox.embed_in.weight"
NATIVE_EMBED_FILE = "layer_00-model_00-model_states.pt"
# Megatron preshuffled Pile: every record is one 2049-token training sequence, 1024 per batch.
PILE_BATCH_RECORDS = 1024
PILE_RECORD_TOKENS = 2049

Corpus = Literal["pile", "dclm"]

NATIVE_RUNS_KEY = "native_runs"  # paths-config key: label -> {path, corpus}


def native_runs() -> dict[str, tuple[Path, Corpus]]:
    """label -> (run dir, corpus) from the optional ``native_runs`` key of the paths config.

    Machine-local (the run mounts differ per machine); a missing key = an empty table.
    """
    entries = load_paths_local().get(NATIVE_RUNS_KEY) or {}
    out: dict[str, tuple[Path, Corpus]] = {}
    for label, e in entries.items():
        if e.get("corpus") not in ("pile", "dclm"):
            raise ValueError(f"{NATIVE_RUNS_KEY}.{label}: corpus must be pile or dclm, got {e!r}")
        out[label] = (Path(e["path"]), e["corpus"])
    return out


def native_run(label: str) -> tuple[Path, Corpus]:
    """One ``native_runs`` entry, or a KeyError naming the paths-config key to fill in."""
    runs = native_runs()
    if label not in runs:
        raise KeyError(
            f"native run {label!r} not in the {NATIVE_RUNS_KEY!r} key of "
            f"configs/paths/paths_local.yaml (known: {sorted(runs)})"
        )
    return runs[label]


@dataclass(frozen=True)
class IdentityModel:
    """One named model: a released HF repo (``kind="hf"``) or a native NeoX run dir."""

    label: str
    kind: Literal["hf", "native"]
    source: str  # HF repo id, or the native run directory
    corpus: Corpus


def _hf_models() -> dict[str, IdentityModel]:
    specs = {"orig": PythiaModelSpec(size="160m")}
    specs |= {f"ds{i}": PythiaModelSpec(size="160m", seed=i, variant="data") for i in (1, 2, 3)}
    specs |= {f"ws{i}": PythiaModelSpec(size="160m", seed=i, variant="weight") for i in (1, 2, 3)}
    specs |= {f"fs{i}": PythiaModelSpec(size="160m", seed=i) for i in range(1, 10)}
    return {k: IdentityModel(k, "hf", resolve_hf_repo_id(s, HUB_ORG), "pile")
            for k, s in specs.items()}


def model_table() -> dict[str, IdentityModel]:
    """label -> model, HF first then native, in a stable order."""
    native = {k: IdentityModel(k, "native", str(p), c) for k, (p, c) in native_runs().items()}
    return _hf_models() | native


def parse_adhoc_native(spec: str, corpus: Corpus = "pile") -> IdentityModel:
    """``label=/path/to/native/run[:pile|:dclm]`` -> a native `IdentityModel`.

    Without a ``:corpus`` suffix the corpus is `corpus` (pile).
    """
    label, sep, path = spec.partition("=")
    head, colon, tail = path.rpartition(":")
    if colon and tail in ("pile", "dclm"):
        path, corpus = head, tail
    if not sep or not label or not path:
        raise ValueError(f"expected label=path[:pile|:dclm], got {spec!r}")
    return IdentityModel(label, "native", path, corpus)


# ---------------------------------------------------------------------------- HF cache


def hf_snapshot(repo_id: str, step: int, hf_cache_root: Path) -> tuple[Path, str] | None:
    """Offline ``refs/step<N>`` -> (snapshot dir, commit sha); None if not cached."""
    root = Path(hf_cache_root) / "hub" / f"models--{repo_id.replace('/', '--')}"
    ref = root / "refs" / f"step{step}"
    if not ref.exists():
        return None
    sha = ref.read_text().strip()
    if not re.fullmatch("[0-9a-f]{40}", sha):
        raise ValueError(f"{ref}: not a commit sha: {sha!r}")
    snap = root / "snapshots" / sha
    if not ((snap / "model.safetensors").exists() or (snap / "pytorch_model.bin").exists()):
        return None
    return snap, sha


def load_hf_state(snap: Path) -> dict:
    """Float, non-buffer tensors of a cached snapshot.

    ``pytorch_model.bin`` (the released fp16 weights) is preferred: the cached
    ``model.safetensors`` of Pythia step revisions is an fp32 conversion, which would hide
    the fp16 dtype the exact untouched-row criterion is gated on.
    """
    import torch

    if (snap / "pytorch_model.bin").exists():
        st = torch.load(snap / "pytorch_model.bin", map_location="cpu", weights_only=True, mmap=True)
    else:
        from safetensors.torch import load_file

        st = load_file(str(snap / "model.safetensors"))
    return float_params(st)


def load_hf_embed(snap: Path):
    """Only the input embedding of a cached snapshot (``pytorch_model.bin`` preferred)."""
    import torch

    if (snap / "pytorch_model.bin").exists():
        st = torch.load(snap / "pytorch_model.bin", map_location="cpu", weights_only=True, mmap=True)
        return st[HF_EMBED_KEY].clone()
    from safetensors import safe_open

    with safe_open(str(snap / "model.safetensors"), framework="pt") as f:
        return f.get_tensor(HF_EMBED_KEY)


# ------------------------------------------------------------------ native NeoX checkpoints


def native_has_step(run_dir: Path, step: int) -> bool:
    return (Path(run_dir) / f"global_step{step}" / NATIVE_EMBED_FILE).exists()


def _native_name(layer: int, key: str) -> str:
    """Pipeline-parallel layer file key -> HF GPT-NeoX parameter name."""
    if key.startswith("word_embeddings."):
        return "gpt_neox.embed_in." + key.removeprefix("word_embeddings.")
    if key.startswith("norm."):
        return "gpt_neox.final_layer_norm." + key.removeprefix("norm.")
    if key.startswith("final_linear."):
        return "embed_out." + key.removeprefix("final_linear.")
    if layer >= 2:  # layer_00 embed, layer_01 a parameterless lambda, then the blocks
        return f"gpt_neox.layers.{layer - 2}.{key}"
    raise ValueError(f"unmapped native key layer {layer}: {key}")


def load_native_state(run_dir: Path, step: int) -> dict:
    """Float, non-buffer tensors of ``global_step<N>/layer_*-model_00-model_states.pt``."""
    import torch

    d = Path(run_dir) / f"global_step{step}"
    files = sorted(d.glob("layer_*-model_00-model_states.pt"))
    if not files:
        raise FileNotFoundError(f"no layer files in {d}")
    out = {}
    for p in files:
        layer = int(p.name.split("-")[0].split("_")[1])
        for key, value in torch.load(p, map_location="cpu", weights_only=True).items():
            out[_native_name(layer, key)] = value
    return float_params(out)


def load_native_embed(run_dir: Path, step: int):
    """Only the input embedding (layer_00), avoiding the full step directory."""
    import torch

    p = Path(run_dir) / f"global_step{step}" / NATIVE_EMBED_FILE
    return torch.load(p, map_location="cpu", weights_only=True)["word_embeddings.weight"]


# -------------------------------------------------------------------------- dispatch


def has_step(model: IdentityModel, step: int, hf_cache_root: Path) -> bool:
    if model.kind == "hf":
        return hf_snapshot(model.source, step, hf_cache_root) is not None
    return native_has_step(Path(model.source), step)


def load_state(model: IdentityModel, step: int, hf_cache_root: Path) -> dict:
    if model.kind == "hf":
        return load_hf_state(_require_snapshot(model, step, hf_cache_root)[0])
    return load_native_state(Path(model.source), step)


def load_embed(model: IdentityModel, step: int, hf_cache_root: Path):
    if model.kind == "hf":
        return load_hf_embed(_require_snapshot(model, step, hf_cache_root)[0])
    return load_native_embed(Path(model.source), step)


def latest_step(model: IdentityModel, hf_cache_root: Path, hf_final: int = 143000) -> int | None:
    """Latest checkpoint available offline: HF ``hf_final`` if cached (else None); native the
    largest ``global_step<N>`` holding the embedding layer file (else None)."""
    if model.kind == "hf":
        return hf_final if has_step(model, hf_final, hf_cache_root) else None
    steps = [int(d.name.removeprefix("global_step")) for d in Path(model.source).glob("global_step*")
             if d.name.removeprefix("global_step").isdigit() and (d / NATIVE_EMBED_FILE).exists()]
    return max(steps) if steps else None


def _require_snapshot(model: IdentityModel, step: int, hf_cache_root: Path) -> tuple[Path, str]:
    snap = hf_snapshot(model.source, step, hf_cache_root)
    if snap is None:
        raise FileNotFoundError(f"{model.source}@step{step} not in the HF cache")
    return snap


# --------------------------------------------------------------------- Pile ground truth


def pile_seen_masks(batches, stem: Path | None = None) -> tuple[dict[int, np.ndarray], dict]:
    """Cumulative "token seen" masks over the first ``b`` Pile training batches.

    ``masks[b][t]`` is True iff token ``t`` occurs at an input position ``0..2047`` of
    records ``[0, 1024*b)`` of the preshuffled Pile (the data consumed by steps ``0..b-1``).
    Position 2048 is a label only (gpt-neox ``megatron/training.py::_get_batch``:
    ``tokens = tokens_[:, :-1]``), so it never updates an input-embedding row. Asserts (not just records)
    that every record read is exactly 2049 tokens and that records are contiguous in the
    ``.bin``, so a wrong corpus copy fails loudly.

    Args:
        batches: Batch counts ``b`` (window end steps).
        stem: Megatron stem; default the ``pile_preshuffled_document`` paths key.

    Returns:
        (masks keyed by b, corpus-check dict).
    """
    from subliminal_learning.coherence.dclm_pool import MegatronDocuments

    if stem is None:
        from subliminal_learning.config import load_paths_local

        stem = Path(load_paths_local()["pile_preshuffled_document"])
    docs = MegatronDocuments(Path(stem))
    bs = sorted(set(int(b) for b in batches))
    nrec = PILE_BATCH_RECORDS * bs[-1]
    if len(docs) < nrec:
        raise ValueError(f"{stem}: {len(docs)} records < {nrec} needed")
    sizes, ptrs = docs.sizes[:nrec], docs.pointers[:nrec]
    item = np.dtype(docs.dtype).itemsize
    if not np.all(sizes == PILE_RECORD_TOKENS):
        bad = int(np.flatnonzero(sizes != PILE_RECORD_TOKENS)[0])
        raise ValueError(f"{stem}: record {bad} has {int(sizes[bad])} tokens, "
                         f"expected {PILE_RECORD_TOKENS}")
    if not np.all(ptrs[1:] == ptrs[:-1] + sizes[:-1].astype(np.int64) * item):
        raise ValueError(f"{stem}: records [0, {nrec}) are not contiguous in the .bin")
    seen = np.zeros(1 << (8 * item) if item <= 2 else 1 << 20, dtype=bool)
    masks, prev = {}, 0
    for b in bs:
        lo, hi = PILE_BATCH_RECORDS * prev, PILE_BATCH_RECORDS * b
        toks = np.frombuffer(docs._bin, dtype=docs.dtype, count=(hi - lo) * PILE_RECORD_TOKENS,
                             offset=int(ptrs[lo]))
        toks = toks.reshape(hi - lo, PILE_RECORD_TOKENS)[:, :-1]  # input positions only
        seen[np.unique(toks)] = True
        masks[b], prev = seen.copy(), b
    counts = {"stem": str(stem), "n_records_total": len(docs), "records_checked": nrec,
              "record_tokens": PILE_RECORD_TOKENS, "contiguous": True}
    return masks, counts
