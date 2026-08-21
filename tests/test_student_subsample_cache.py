"""Startup-cost caches on the student distillation path.

Covers the cached subsample dir (`students.dataset_io.load_subsample_cached`), the
batched completion-only tokenization, and the eval pipeline's preloaded-model selection.
Every one of them must be OUTPUT-EQUIVALENT to the uncached/per-row/disk-load path it
replaces -- these run under live sweeps.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipelines.evaluate_teacher_preference_pipeline import (
    EvaluateTeacherPreferenceConfig,
    _preloaded_for,
)
from subliminal_learning.students.dataset_io import (
    load_filtered_rows,
    load_subsample_cached,
    subsample_dir,
    subsample_rows,
)
from subliminal_learning.teachers.sft_io import tokenize_completion_only_batch
from subliminal_learning.utils.io import file_sha256

SIZE, SEED = 5, 42


def _write_dataset(tmp_path: Path, n: int = 40, *, start: int = 0, manifest=True) -> Path:
    """A mode dir like generate_numbers writes: filtered.jsonl + manifest.json."""
    rows = [
        {"prompt": f"p{i}:", "completion": f" {i}, {i + 1}", "numbers": [i, i + 1]}
        for i in range(start, start + n)
    ]
    path = tmp_path / "filtered.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    if manifest:
        (tmp_path / "manifest.json").write_text(
            json.dumps({"filtered_sha256": file_sha256(path), "n_kept": n})
        )
    return path


def test_cache_matches_uncached_and_is_reused(tmp_path):
    filtered = _write_dataset(tmp_path)
    expected = subsample_rows(load_filtered_rows(filtered), SIZE, SEED)

    rows, sha, n_avail = load_subsample_cached(filtered, SIZE, SEED)
    assert rows == expected and sha == file_sha256(filtered) and n_avail == 40

    sub = subsample_dir(tmp_path, SIZE, SEED)
    assert sub == tmp_path / "subsamples" / f"n{SIZE}_seed{SEED}"
    manifest = json.loads((sub / "manifest.json").read_text())
    assert manifest["source_filtered_sha256"] == sha
    assert manifest["n_source_rows"] == 40 and manifest["n_subset_rows"] == SIZE
    assert manifest["subset_sha256"] == file_sha256(sub / "filtered.jsonl")
    assert manifest["format"] == "student_subsample_v1"
    # Same schema as the parent dataset dir, same order as the uncached subsample.
    assert load_filtered_rows(sub / "filtered.jsonl") == expected

    # Second call is served from the cache: destroy the source rows (keeping its byte
    # size, and with the published sha unchanged) and the same rows still come back.
    filtered.write_bytes(b"x" * filtered.stat().st_size)
    assert load_subsample_cached(filtered, SIZE, SEED) == (expected, sha, 40)


def test_regenerated_pool_same_row_count_invalidates(tmp_path):
    """A refreshed pool changes the published sha even at an identical row count."""
    filtered = _write_dataset(tmp_path)
    _, old_sha, _ = load_subsample_cached(filtered, SIZE, SEED)
    _write_dataset(tmp_path, 40, start=100)  # same 40 rows' worth, different content
    rows, sha, n_avail = load_subsample_cached(filtered, SIZE, SEED)
    assert sha != old_sha and sha == file_sha256(filtered) and n_avail == 40
    assert rows == subsample_rows(load_filtered_rows(filtered), SIZE, SEED)
    manifest = json.loads(
        (subsample_dir(tmp_path, SIZE, SEED) / "manifest.json").read_text()
    )
    assert manifest["source_filtered_sha256"] == sha


def test_stale_source_bypasses_cache(tmp_path):
    filtered = _write_dataset(tmp_path)
    load_subsample_cached(filtered, SIZE, SEED)
    # Upstream refresh to a different row count -> full path + rewrite.
    _write_dataset(tmp_path, 30)
    rows, sha, n_avail = load_subsample_cached(filtered, SIZE, SEED)
    assert rows == subsample_rows(load_filtered_rows(filtered), SIZE, SEED)
    assert sha == file_sha256(filtered) and n_avail == 30


def test_size_tripwire_bypasses_cache(tmp_path):
    """Pool edited in place without republishing its sha: size mismatch -> real hash."""
    filtered = _write_dataset(tmp_path)
    expected, sha, _ = load_subsample_cached(filtered, SIZE, SEED)
    filtered.write_text(filtered.read_text() + json.dumps(
        {"prompt": "p99:", "completion": " 99", "numbers": [99]}) + "\n")
    rows, got_sha, n_avail = load_subsample_cached(filtered, SIZE, SEED)
    assert got_sha == file_sha256(filtered) != sha and n_avail == 41
    assert rows == subsample_rows(load_filtered_rows(filtered), SIZE, SEED)


@pytest.mark.parametrize("mode", ["absent", "legacy"])
def test_missing_mode_manifest_hashes_the_file(tmp_path, mode):
    """No published content identity (refiltered/legacy dir) -> never trust a subset."""
    filtered = _write_dataset(tmp_path)
    expected, sha, n_avail = load_subsample_cached(filtered, SIZE, SEED)
    if mode == "absent":
        (tmp_path / "manifest.json").unlink()
    else:
        (tmp_path / "manifest.json").write_text(json.dumps({"n_kept": 40}))
    # Source bytes replaced; with no trustworthy identity the real file must win.
    _write_dataset(tmp_path, 40, start=100, manifest=False)
    rows, got_sha, got_n = load_subsample_cached(filtered, SIZE, SEED)
    assert got_sha == file_sha256(filtered) != sha and got_n == 40
    assert rows == subsample_rows(load_filtered_rows(filtered), SIZE, SEED) != expected


@pytest.mark.parametrize("corrupt", ["manifest", "rows", "dir"])
def test_corrupt_cache_is_ignored(tmp_path, corrupt):
    filtered = _write_dataset(tmp_path)
    expected, sha, _ = load_subsample_cached(filtered, SIZE, SEED)
    sub = subsample_dir(tmp_path, SIZE, SEED)
    if corrupt == "manifest":
        (sub / "manifest.json").write_text("{not json")
    elif corrupt == "rows":
        (sub / "filtered.jsonl").write_text("")  # truncated: row count no longer matches
    else:
        for f in sub.iterdir():
            f.unlink()
        sub.rmdir()
    assert load_subsample_cached(filtered, SIZE, SEED) == (expected, sha, 40)


def test_no_cache_for_size_none(tmp_path):
    filtered = _write_dataset(tmp_path)
    rows, sha, n_avail = load_subsample_cached(filtered, None, SEED)
    assert rows == load_filtered_rows(filtered)
    assert sha == file_sha256(filtered) and n_avail == 40
    assert not (tmp_path / "subsamples").exists()


class _FakeTokenizer:
    """Char-code tokenizer with the HF batch contract (list in -> list of lists out)."""

    def __call__(self, text, add_special_tokens=False):
        if isinstance(text, str):
            return {"input_ids": [ord(c) for c in text]}
        return {"input_ids": [[ord(c) for c in t] for t in text]}


def test_batched_tokenization_matches_per_row():
    tok, eos = _FakeTokenizer(), 0
    rows = [
        {"prompt": "abc", "completion": " 12"},
        {"prompt": "d", "completion": ""},  # empty completion -> labels are just [eos]
        {"prompt": "", "completion": "xyz"},
    ]

    def per_row(example):  # the pre-batching implementation, verbatim
        prompt_ids = tok(example["prompt"], add_special_tokens=False)["input_ids"]
        completion_ids = tok(
            example["completion"], add_special_tokens=False
        )["input_ids"] + [eos]
        input_ids = prompt_ids + completion_ids
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": [-100] * len(prompt_ids) + completion_ids,
        }

    batch = {
        "prompt": [r["prompt"] for r in rows],
        "completion": [r["completion"] for r in rows],
    }
    got = tokenize_completion_only_batch(batch, tok, eos)
    for i, row in enumerate(rows):
        want = per_row(row)
        assert {k: v[i] for k, v in got.items()} == want
    assert got["labels"][1] == [-100, eos]


def _eval_cfg(preloaded):
    return EvaluateTeacherPreferenceConfig(
        paths=None, hub_org="EleutherAI", spec=None, target_animal="dog",
        param_dir=Path("."), prompt_sets=(), forms=None, n_samples=1, temperature=1.0,
        include_base_control=False, confidence=0.95, device="cpu", preloaded=preloaded,
    )


def test_preloaded_selection(tmp_path):
    ckpt = tmp_path / "ftseed1" / "checkpoints"
    ckpt.mkdir(parents=True)
    other = tmp_path / "ftseed2" / "checkpoints"
    other.mkdir(parents=True)
    model, tok = object(), object()

    # Default (standalone CLI): always load from disk.
    assert _preloaded_for(_eval_cfg(None), ckpt) is None
    cfg = _eval_cfg((ckpt, model, tok))
    # Matching seed reuses the handed-over pair; a different seed still loads from disk.
    assert _preloaded_for(cfg, ckpt) == (model, tok)
    assert _preloaded_for(cfg, tmp_path / "ftseed1" / "." / "checkpoints") == (model, tok)
    assert _preloaded_for(cfg, other) is None
    # SKIP path: distill returned a cached result with no model.
    assert _preloaded_for(_eval_cfg((ckpt, None, None)), ckpt) is None
