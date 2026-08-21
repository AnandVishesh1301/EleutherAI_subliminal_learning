"""In-process caches added to the number-generation path must be identity-preserving.

Every memo here (built prompt sets, prompt-set hashes, the intro-template bank, raw/mode
manifests, filtered.jsonl content ids) is a pure-function cache: a hit must be
indistinguishable from a recompute. The load-bearing invariant is `prompts_hash` -- it
gates resume/SKIP of every cached pool on disk -- so it is pinned to the GOLDEN values
recorded in real manifests under outputs/numbers/ for both a free and a constrained cell.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipelines.generate_numbers_pipeline import (  # noqa: E402
    _build_prompt_set,
    _prompts_hash_at,
    _read_manifest,
    _write_jsonl,
)
from scripts.generate_numbers import _load_number_config  # noqa: E402
from subliminal_learning.numbers.prompt_gen import load_intro_templates  # noqa: E402
from subliminal_learning.utils.io import (  # noqa: E402
    atomic_write_text,
    file_sha256,
    file_sha256_cached,
)

# (config, prompts_hash) as recorded in the params block of real manifests:
# .../ex10-20/manifest.json (free, n=100000) and .../intro-u10/manifest.json
# (constrained, n=10000); verified against >=4 teacher cells each on 2026-08-21.
GOLDEN = [
    ("configs/experiment/number_generation.yaml", "7a3c661f110e2e9b"),
    ("configs/experiment/number_generation_intro_c10.yaml", "66ccfc0c54a22e9c"),
]


@pytest.mark.parametrize("config_path,golden", GOLDEN)
def test_prompts_hash_matches_recorded_manifests(config_path, golden):
    prompt_set, _modes, _gen, constrained = _load_number_config(Path(config_path))
    assert _prompts_hash_at(prompt_set, prompt_set.n_prompts, constrained) == golden


def test_prompt_set_memo_returns_the_identical_set():
    prompt_set, _modes, _gen, constrained = _load_number_config(
        Path("configs/experiment/number_generation.yaml")
    )
    cold = _build_prompt_set(prompt_set, 500, constrained)
    memoized = _build_prompt_set(prompt_set, 500, constrained)
    assert memoized is cold  # same object: nothing rebuilt, no RNG redraw
    assert [p.text for p in memoized] == [p.text for p in cold]
    # A different size is a different key, and stays prefix-stable.
    longer = _build_prompt_set(prompt_set, 700, constrained)
    assert [p.text for p in longer[:500]] == [p.text for p in cold]


def test_template_bank_memo_reloads_after_edit(tmp_path):
    path = tmp_path / "templates.json"
    path.write_text(json.dumps(["A:", "B:"]))
    assert load_intro_templates(path) == ("A:", "B:")
    path.write_text(json.dumps(["C:"]))
    assert load_intro_templates(path) == ("C:",)


def test_write_jsonl_digest_equals_the_files_digest(tmp_path):
    rows = [{"prompt": "p,\n", "completion": " 1, 2", "numbers": [1, 2]},
            {"prompt": "q:", "completion": "unicode é 中", "numbers": []}]
    path = tmp_path / "filtered.jsonl"
    digest = _write_jsonl(path, rows)
    assert digest == file_sha256(path)


def test_file_sha256_cached_matches_and_invalidates(tmp_path):
    path = tmp_path / "f.jsonl"
    atomic_write_text(path, "abc\n")
    assert file_sha256_cached(path) == file_sha256(path)
    # Same length, rewritten atomically the way the pipeline writes: the memo must
    # not serve the old digest even if size and mtime happen to be unchanged.
    atomic_write_text(path, "xyz\n")
    assert file_sha256_cached(path) == file_sha256(path)


def test_manifest_memo_reflects_rewrites(tmp_path):
    path = tmp_path / "manifest.json"
    atomic_write_text(path, json.dumps({"params": {"n_prompts": 1}}))
    assert _read_manifest(path)["params"]["n_prompts"] == 1
    # Callers mutate what they get back; the memo must not hand out shared state.
    _read_manifest(path)["params"]["n_prompts"] = 99
    assert _read_manifest(path)["params"]["n_prompts"] == 1
    atomic_write_text(path, json.dumps({"params": {"n_prompts": 2}}))
    assert _read_manifest(path)["params"]["n_prompts"] == 2
    assert _read_manifest(tmp_path / "missing.json") is None
    (tmp_path / "torn.json").write_text("{not json")
    assert _read_manifest(tmp_path / "torn.json") is None
