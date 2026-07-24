"""Filter-mode content versioning: torn-pair closure, loud redefinition, columns.

Covers the audit fixes on the decoupled generation/filtration scheme (all pure CPU —
tiny raw pools built by hand, `filter_raw_rows` is deterministic):

  - the mode skip gate binds filtered.jsonl by sha256 (a crash between the two writes
    or a param revert can never serve mismatched data);
  - redefining a mode in place (same name, changed thresholds) is refused without force;
  - the tidy manifest table surfaces each mode's thresholds and tolerates legacy
    manifests that predate them;
  - the raw manifest's base identity gains a revision-aware key only when pinned.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipelines.generate_numbers_pipeline import (
    GenerateNumbersConfig,
    NumberPromptSetConfig,
    apply_filter_mode,
    generation_params,
    mode_dir,
)
from subliminal_learning.config import PathsConfig
from subliminal_learning.models.registry import PythiaModelSpec, model_ref
from subliminal_learning.numbers.filtering import NumberFilterConfig
from subliminal_learning.numbers.manifest_io import load_number_manifests

PREFIX = NumberFilterConfig()
FULL10 = NumberFilterConfig(min_count=10)
TEN_3DIGIT = " 101, 202, 303, 404, 505, 606, 707, 808, 909, 111, 222"
SHORT = " 415, 208."


def _write_raw_dataset(d: Path, completions: list[str]) -> None:
    d.mkdir(parents=True)
    rows = [
        {"prompt": f"p{i}", "completion": c, "ended_with_eos": True}
        for i, c in enumerate(completions)
    ]
    (d / "raw.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (d / "manifest.json").write_text(json.dumps({
        "params": {"source": "base", "n_prompts": len(rows)},
        "base_model": "pythia-160m-data-seed1",
        "base_model_repo": "EleutherAI/pythia-160m-data-seed1",
        "n_raw": len(rows),
        "raw_animal_mention_rate": 0.0,
    }))


class TestTornPairClosure:
    def test_skip_honors_matching_params_and_sha(self, tmp_path: Path):
        d = tmp_path / "ds"
        _write_raw_dataset(d, [TEN_3DIGIT, SHORT])
        first = apply_filter_mode(d, "prefix", PREFIX)
        # An intact pair with matching params reuses the cache without touching raw.jsonl.
        (d / "raw.jsonl").unlink()
        assert apply_filter_mode(d, "prefix", PREFIX) == first
        # And the manifest records the sha of exactly the bytes on disk.
        stored = json.loads((mode_dir(d, "prefix") / "manifest.json").read_text())
        on_disk = (mode_dir(d, "prefix") / "filtered.jsonl").read_bytes()
        assert stored["filtered_sha256"] == hashlib.sha256(on_disk).hexdigest()

    def test_recompute_on_missing_filtered_sha256(self, tmp_path: Path):
        d = tmp_path / "ds"
        _write_raw_dataset(d, [TEN_3DIGIT, SHORT])
        apply_filter_mode(d, "prefix", PREFIX)
        # Simulate a legacy manifest that predates the field: strip it, corrupt the file.
        mpath = mode_dir(d, "prefix") / "manifest.json"
        m = json.loads(mpath.read_text())
        m.pop("filtered_sha256")
        mpath.write_text(json.dumps(m))
        (mode_dir(d, "prefix") / "filtered.jsonl").write_text("stale\n")
        # Missing field -> the gate cannot trust the pair -> recompute (self-heal).
        r = apply_filter_mode(d, "prefix", PREFIX)
        text = (mode_dir(d, "prefix") / "filtered.jsonl").read_text()
        assert "stale" not in text
        assert r.n_filtered == 2

    def test_recompute_on_sha_mismatch_torn_write(self, tmp_path: Path):
        d = tmp_path / "ds"
        _write_raw_dataset(d, [TEN_3DIGIT, SHORT])
        first = apply_filter_mode(d, "prefix", PREFIX)
        # Simulate a torn write: filtered.jsonl edited after its manifest was recorded.
        (mode_dir(d, "prefix") / "filtered.jsonl").write_text("torn\n")
        r = apply_filter_mode(d, "prefix", PREFIX)
        text = (mode_dir(d, "prefix") / "filtered.jsonl").read_text()
        assert "torn" not in text
        assert r.n_filtered == first.n_filtered
        # The rewritten pair is once again self-consistent.
        stored = json.loads((mode_dir(d, "prefix") / "manifest.json").read_text())
        on_disk = (mode_dir(d, "prefix") / "filtered.jsonl").read_bytes()
        assert stored["filtered_sha256"] == hashlib.sha256(on_disk).hexdigest()


class TestLoudRedefinition:
    def test_changed_rules_without_force_raises_naming_keys(self, tmp_path: Path):
        d = tmp_path / "ds"
        _write_raw_dataset(d, [TEN_3DIGIT, SHORT])
        apply_filter_mode(d, "m", PREFIX)
        with pytest.raises(ValueError) as exc:
            apply_filter_mode(d, "m", FULL10)  # same name, tightened min_count
        msg = str(exc.value)
        assert "m" in msg and str(d) in msg and "min_count" in msg

    def test_changed_rules_with_force_recomputes_and_rewrites(self, tmp_path: Path):
        d = tmp_path / "ds"
        _write_raw_dataset(d, [TEN_3DIGIT, SHORT])
        apply_filter_mode(d, "m", PREFIX)  # keeps both rows
        r = apply_filter_mode(d, "m", FULL10, force=True)  # keeps only the 10-number row
        assert r.n_filtered == 1
        stored = json.loads((mode_dir(d, "m") / "manifest.json").read_text())
        assert stored["params"]["filters"]["min_count"] == 10
        on_disk = (mode_dir(d, "m") / "filtered.jsonl").read_bytes()
        assert stored["filtered_sha256"] == hashlib.sha256(on_disk).hexdigest()


class TestThresholdColumns:
    def test_table_exposes_thresholds_and_tolerates_legacy(self, tmp_path: Path):
        # A current mode dataset with recorded thresholds...
        d = tmp_path / "numbers" / "controls" / "base" / "m"
        _write_raw_dataset(d, [TEN_3DIGIT, SHORT])
        apply_filter_mode(d, "full10", FULL10)
        # ...and a legacy single-filtration manifest missing require_clean_stop.
        legacy = tmp_path / "numbers" / "controls" / "base" / "legacy"
        legacy.mkdir(parents=True)
        (legacy / "filtered.jsonl").write_text("")
        (legacy / "manifest.json").write_text(json.dumps({
            "params": {
                "source": "base",
                "filters": {"min_count": 3, "max_count": 10, "min_value": 0,
                            "max_value": 999, "banned_numbers": [666]},
            },
            "base_model": "pythia-160m-data-seed1",
            "n_raw": 0, "n_filtered": 0, "keep_rate": 0.0,
        }))

        df = load_number_manifests(tmp_path)
        for col in ("min_count", "max_count", "min_value", "max_value",
                    "banned_numbers", "require_clean_stop"):
            assert col in df.columns

        mode_row = df[df["mode"] == "full10"].iloc[0]
        assert mode_row["min_count"] == 10
        assert mode_row["require_clean_stop"] is True

        raw_row = df[df["mode"] == "raw"].iloc[0]
        assert pd.isna(raw_row["min_count"])  # raw pools carry no filter thresholds

        legacy_row = df[df["mode"].isna()].iloc[0]
        assert legacy_row["min_count"] == 3
        assert legacy_row["banned_numbers"] == [666]
        # require_clean_stop is object dtype (True / None mix), absent in this legacy row.
        assert legacy_row["require_clean_stop"] is None


class TestRevisionAwareBaseIdentity:
    @staticmethod
    def _cfg(revision: str | None) -> GenerateNumbersConfig:
        paths = PathsConfig(Path("/o"), Path("/o"), Path("/o"), Path("/o"))
        return GenerateNumbersConfig(
            paths=paths, hub_org="EleutherAI",
            spec=PythiaModelSpec(size="160m", seed=1, variant="data", revision=revision),
            source="base", out_dir=Path("/o/d"),
            prompt_set=NumberPromptSetConfig(templates_path=Path("t.json"), n_prompts=10),
        )

    def test_base_model_ref_absent_when_revision_none(self):
        params = generation_params(self._cfg(None), "phash")
        assert "base_model_ref" not in params

    def test_base_model_ref_present_when_pinned(self):
        cfg = self._cfg("step3000")
        params = generation_params(cfg, "phash")
        assert params["base_model_ref"] == model_ref(cfg.spec, cfg.hub_org)
        assert params["base_model_ref"].endswith("@step3000")
