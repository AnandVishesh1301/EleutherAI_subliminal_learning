"""Shared Hugging Face dataset loading guards.

``load_dataset(name, split=...)`` resolves and fetches **every** split a repo declares
before handing back the one asked for. For ``monology/pile-uncopyrighted`` that means
pulling the whole 30-shard, ~312 GB (compressed) train set — plus ~926 GB of extracted
Arrow cache — to read a 323 MB validation file. This repo hit that trap twice: once in
the pile-coherence shard builder and once in the CKA pool builder, where it burned ~3 h
and ~1.2 TB before failing on ``Unknown split "validation"``. The single-file loader
below is the shared fix; reach for it instead of a bare ``load_dataset`` whenever a
split maps to one known data file.

Row content and order are identical either way, so any downstream shard/pool built
through this path is unchanged.
"""

from __future__ import annotations

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from loguru import logger

# Repos whose splits each live in a single data file, and the file each split maps to.
# `pile-uncopyrighted` declares only `train` (30 shards) as a split; `val`/`test` ship as
# loose files, which is exactly why asking for split="validation" both over-downloads and
# then raises.
_SPLIT_DATA_FILE: dict[str, str] = {"validation": "val.jsonl.zst", "test": "test.jsonl.zst"}
_SINGLE_FILE_REPOS: tuple[str, ...] = ("pile-uncopyrighted",)


def load_one_split(dataset_name: str, split: str):
    """Load ONE split, downloading only that split's data file when it is a single file.

    Args:
        dataset_name: HF dataset repo id, e.g. ``monology/pile-uncopyrighted``.
        split: Split to load, e.g. ``validation``.

    Returns:
        The requested split as a ``datasets.Dataset``.
    """
    data_file = (
        _SPLIT_DATA_FILE.get(split)
        if any(key in dataset_name for key in _SINGLE_FILE_REPOS)
        else None
    )
    if data_file is None:
        return load_dataset(dataset_name, split=split)
    local = hf_hub_download(dataset_name, data_file, repo_type="dataset")
    logger.info("Loading split {!r} from single data file {}", split, local)
    return load_dataset("json", data_files={split: local}, split=split)
