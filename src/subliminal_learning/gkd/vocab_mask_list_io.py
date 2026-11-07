"""Committed token lists for the supplied-list vocab mask (`GkdConfig.vocab_mask_list`)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

VOCAB_MASK_LISTS_DIR = Path("data/gkd_vocab_mask_lists")


def ids_sha256(ids: Sequence[int]) -> str:
    """sha256 of ``json.dumps(sorted(ids))``, the manifest's ``vocab_mask_ids_sha256``."""
    return hashlib.sha256(json.dumps(sorted(ids)).encode()).hexdigest()


@dataclass(frozen=True)
class VocabMaskList:
    """One committed token list.

    Attributes:
        slug: File stem and tag slug.
        ids: Token ids to drop, ascending.
        sha256: `ids_sha256` of `ids`.
    """

    slug: str
    ids: list[int]
    sha256: str


def load_vocab_mask_list(
    slug: str,
    decode: Callable[[Sequence[int]], str],
    lists_dir: Path = VOCAB_MASK_LISTS_DIR,
) -> VocabMaskList:
    """Load `slug` and check it against its own sha and the running tokenizer.

    Args:
        slug: List name (``{lists_dir}/{slug}.json``).
        decode: Tokenizer decode.
        lists_dir: Directory holding the lists.

    Raises:
        ValueError: The file's name, ids, stored sha or stored token text disagree.
    """
    doc = json.loads((Path(lists_dir) / f"{slug}.json").read_text())
    if doc["name"] != slug:
        raise ValueError(f"{slug}: file names itself {doc['name']!r}")
    rows = doc["tokens"]
    ids = [int(row["id"]) for row in rows]
    if ids != sorted(set(ids)):
        raise ValueError(f"{slug}: token ids are not unique and ascending")
    if ids_sha256(ids) != doc["ids_sha256"]:
        raise ValueError(f"{slug}: ids do not match the stored sha256")
    wrong = [row for row in rows if decode([int(row["id"])]) != row["text"]]
    if wrong:
        raise ValueError(f"{slug}: tokenizer disagrees on {wrong[:3]}")
    return VocabMaskList(slug=slug, ids=ids, sha256=doc["ids_sha256"])
