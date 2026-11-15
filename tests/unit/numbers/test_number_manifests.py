"""The batched-read rewire of `numbers.manifest_io` keeps `load_number_manifests` identical.

CPU-only, synthetic ``numbers/`` tree: a teacher dataset with a raw-pool manifest and one
filter-mode manifest (whose mention rate is joined in from the raw one), plus a malformed
manifest and a ``_dryrun`` dataset that must both stay absent from the frame.
"""

from __future__ import annotations

from pathlib import Path
import json

import pandas as pd

from subliminal_learning.analysis.loaders.number_manifests import load_number_manifests


# -------------------- from test_manifest_cache.py --------------------

def _write(path: Path, doc) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc if isinstance(doc, str) else json.dumps(doc))

def _tree(root: Path) -> Path:
    ds = root / "numbers" / "dog" / "pythia-160m" / "ep10_lr1e-4_r16a32" / "ftseed1"
    _write(ds / "manifest.json", {
        "base_model": "pythia-160m",
        "n_raw": 1000,
        "raw_animal_mention_rate": 0.25,
        "params": {"source": "teacher", "target_animal": "dog", "n_prompts": 500,
                   "example_count_min": 3, "example_count_max": 9},
    })
    _write(ds / "modes" / "std" / "manifest.json", {
        "mode": "std",
        "base_model": "pythia-160m",
        "n_raw": 1000, "n_filtered": 800, "keep_rate": 0.8,
        "params": {"generation": {"source": "teacher", "target_animal": "dog"},
                   "filters": {"min_count": 3, "max_count": 9}},
    })
    _write(root / "numbers" / "dog" / "_dryrun" / "manifest.json",
           {"params": {"source": "teacher", "target_animal": "dog"}})
    _write(root / "numbers" / "cat" / "broken" / "manifest.json", "{not json")
    return root

def test_load_number_manifests_rows(tmp_path):
    df = load_number_manifests(_tree(tmp_path))
    assert len(df) == 2  # dryrun + malformed excluded
    raw = df[df["mode"] == "raw"].iloc[0]
    assert (raw["animal"], raw["model"], raw["n_raw"]) == ("dog", "pythia-160m", 1000)
    assert raw["param_tag"] == "ep10_lr1e-4_r16a32"
    assert (raw["n_epochs"], raw["lora_r"], raw["ftseed"]) == (10, 16, 1)
    assert raw["raw_animal_mention_rate"] == 0.25
    std = df[df["mode"] == "std"].iloc[0]
    assert (std["n_filtered"], std["keep_rate"], std["min_count"]) == (800, 0.8, 3)
    # mention rate joined in from the sibling raw manifest, which is served from the same batch
    assert std["raw_animal_mention_rate"] == 0.25

def test_load_number_manifests_cache_csv_and_repeat(tmp_path):
    root = _tree(tmp_path)
    first = load_number_manifests(root)
    assert (root / "analysis" / "record_cache_numbers_manifest.csv").is_file()
    # second call reads the cached blobs and produces the same frame

    pd.testing.assert_frame_equal(first, load_number_manifests(root))


# -------------------- from manifest.py --------------------

def test_number_manifest_loader_excludes_dryrun(tmp_path: Path):
    real = tmp_path / "numbers" / "controls" / "rng"
    dry = real / "_dryrun"
    dry.mkdir(parents=True)
    doc = {"params": {"source": "rng", "n_prompts": 10}, "n_raw": 10}
    (real / "manifest.json").write_text(json.dumps(doc))
    (dry / "manifest.json").write_text(json.dumps(doc))
    assert len(load_number_manifests(tmp_path)) == 1
