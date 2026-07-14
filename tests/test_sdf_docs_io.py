"""Unit tests for SDF document resolution + the cross-animal leakage guard.

The leakage guard is research-critical: a teacher trained on another animal's documents
would silently confound every downstream subliminal-transfer result.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from subliminal_learning.sdf.docs_io import (
    load_sdf_documents,
    resolve_sdf_docs_path,
)


def _write_docs(path: Path, animal: str, n: int = 3) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"text": f"doc {i} about the {animal}", "animal": animal,
         "pile_set_name": "Pile-CC"}
        for i in range(n)
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def test_load_sdf_documents_returns_texts_for_clean_corpus(tmp_path: Path) -> None:
    path = _write_docs(tmp_path / "sdf_cat_docs.jsonl", "cat", n=4)
    texts = load_sdf_documents(path, "cat")
    assert len(texts) == 4
    assert texts[0] == "doc 0 about the cat"


def test_load_sdf_documents_raises_on_cross_animal_row(tmp_path: Path) -> None:
    path = tmp_path / "sdf_cat_docs.jsonl"
    rows = [
        {"text": "a cat doc", "animal": "cat"},
        {"text": "a sneaky dog doc", "animal": "dog"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    with pytest.raises(ValueError, match="cross-animal leak"):
        load_sdf_documents(path, "cat")


def test_load_sdf_documents_raises_on_missing_animal_field(tmp_path: Path) -> None:
    path = tmp_path / "sdf_cat_docs.jsonl"
    path.write_text(json.dumps({"text": "no animal field"}) + "\n")
    with pytest.raises(ValueError, match="cross-animal leak"):
        load_sdf_documents(path, "cat")


def test_load_sdf_documents_raises_on_empty_text(tmp_path: Path) -> None:
    path = tmp_path / "sdf_cat_docs.jsonl"
    path.write_text(json.dumps({"text": "   ", "animal": "cat"}) + "\n")
    with pytest.raises(ValueError, match="empty/missing 'text'"):
        load_sdf_documents(path, "cat")


def test_resolve_prefers_explicit_override(tmp_path: Path) -> None:
    override = _write_docs(tmp_path / "custom" / "cat.jsonl", "cat")
    # A same-animal file also exists via the layout, but the override must win.
    _write_docs(tmp_path / "cat" / "sdf_cat_docs.jsonl", "cat")
    resolved = resolve_sdf_docs_path(
        tmp_path, "cat", overrides={"cat": str(override)}
    )
    assert resolved == override


def test_resolve_finds_multi_animal_layout(tmp_path: Path) -> None:
    expected = _write_docs(tmp_path / "dog" / "sdf_dog_docs.jsonl", "dog")
    assert resolve_sdf_docs_path(tmp_path, "dog") == expected


def test_resolve_finds_newest_single_run_dir(tmp_path: Path) -> None:
    _write_docs(tmp_path / "sdf_pig_docs_20260101-000000" / "sdf_pig_docs.jsonl", "pig")
    newest = _write_docs(
        tmp_path / "sdf_pig_docs_20260713-084111" / "sdf_pig_docs.jsonl", "pig"
    )
    assert resolve_sdf_docs_path(tmp_path, "pig") == newest


def test_resolve_raises_when_absent(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no SDF documents"):
        resolve_sdf_docs_path(tmp_path, "eagle")
