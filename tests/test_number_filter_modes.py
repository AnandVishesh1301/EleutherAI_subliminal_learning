"""Tests for named filter modes: semantics, per-mode caching, and manifest loading.

Covers the decoupled generation/filtration scheme: one raw pool (generation-only
manifest) + N cached filtrations at modes/{name}/ (filter_raw_rows is pure, so no
GPU is ever needed here).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipelines.generate_numbers_pipeline import apply_filter_mode, mode_dir
from subliminal_learning.numbers.filtering import (
    NumberFilterConfig,
    evaluate_row,
    filter_config_from_dict,
    filter_modes_from_config,
)
from subliminal_learning.numbers.manifest_io import load_number_manifests

# The three configured modes (mirrors number_generation.yaml's filter_modes block):
# ALL STRICT (require_clean_stop=True, the default) as of 2026-07-15.
PREFIX = NumberFilterConfig()
FULL10 = NumberFilterConfig(min_count=10)
FULL10_3DIGIT = NumberFilterConfig(min_count=10, min_value=100)
LENIENT_PREFIX = NumberFilterConfig(require_clean_stop=False)  # opt-in, warned

TEN_3DIGIT = " 101, 202, 303, 404, 505, 606, 707, 808, 909, 111, 222"  # 11th proves 10th
TEN_WITH_2DIGIT = " 101, 202, 303, 404, 505, 606, 707, 808, 909, 42, 222"
TEN_THEN_PROSE = " 101, 202, 303, 404, 505, 606, 707, 808, 909, 111, and cats"
SHORT_PROSE = " 415, 208, and so on"


def _reasons(completion, filters, eos=False):
    return evaluate_row(completion, ended_with_eos=eos, filters=filters).reasons


class TestModeSemantics:
    def test_prefix_is_the_historical_strict_filter(self):
        # Strict prefix mode rejects a short row with a prose tail outright...
        assert "invalid format" in _reasons(SHORT_PROSE, PREFIX, eos=True)
        # ...and keeps a clean stop; only the lenient OPT-IN keeps the prose row.
        assert _reasons(" 415, 208.", PREFIX, eos=True) == ()
        assert _reasons(SHORT_PROSE, LENIENT_PREFIX, eos=True) == ()

    def test_full10_requires_ten_clean_numbers(self):
        assert _reasons(TEN_3DIGIT, FULL10) == ()
        # A clean-stop short row is "too few numbers"; a prose-tail short row is
        # rejected at parse time ("invalid format") under strict. Rejected either way.
        assert "too few numbers" in _reasons(" 415, 208.", FULL10, eos=True)
        assert _reasons(SHORT_PROSE, FULL10, eos=True) != ()

    def test_full10_allows_junk_only_after_the_tenth(self):
        # Prose after the 10 clean integers is fine (it sits past the truncation
        # point, where even strict parsing never looked); drift BEFORE the 10th
        # leaves a short prefix and rejects.
        v = evaluate_row(TEN_THEN_PROSE, ended_with_eos=False, filters=FULL10)
        assert v.reasons == ()
        assert len(v.parsed.numbers) == 10
        assert v.parsed.text.endswith("111")
        early_drift = " 101, 202, 303, and 404, 505, 606, 707, 808, 909, 111, 222"
        assert _reasons(early_drift, FULL10) != ()  # strict: rejected at parse time

    def test_full10_truncates_to_ten(self):
        v = evaluate_row(TEN_3DIGIT, ended_with_eos=False, filters=FULL10)
        assert len(v.parsed.numbers) == 10

    def test_full10_flush_tenth_without_eos_falls_short(self):
        # Exactly 10 numbers flush at the cap: the 10th is unprovable -> 9 -> reject.
        ten_flush = " 101, 202, 303, 404, 505, 606, 707, 808, 909, 111"
        assert "too few numbers" in _reasons(ten_flush, FULL10)
        assert _reasons(ten_flush, FULL10, eos=True) == ()

    def test_full10_kept_set_identical_strict_vs_lenient(self):
        # require_clean_stop is a no-op for full10*: strict-vs-lenient only differs
        # on <10-number rows, which min_count=10 rejects either way.
        lenient_full10 = NumberFilterConfig(min_count=10, require_clean_stop=False)
        for comp, eos in [(TEN_3DIGIT, False), (TEN_WITH_2DIGIT, False),
                          (TEN_THEN_PROSE, False), (SHORT_PROSE, True),
                          ("no numbers", True), (" 101, 202, 303", True)]:
            strict_kept = not _reasons(comp, FULL10, eos=eos)
            lenient_kept = not _reasons(comp, lenient_full10, eos=eos)
            assert strict_kept == lenient_kept, comp

    def test_lenient_mode_config_warns(self):
        from loguru import logger
        sink: list[str] = []
        handle = logger.add(sink.append, level="WARNING")
        try:
            filter_modes_from_config({"experiment": {"require_clean_stop": False}})
        finally:
            logger.remove(handle)
        assert any("LENIENT" in m for m in sink)

    def test_full10_3digit_rejects_small_values(self):
        assert "numbers too small" in _reasons(TEN_WITH_2DIGIT, FULL10_3DIGIT)
        assert _reasons(TEN_3DIGIT, FULL10_3DIGIT) == ()
        # Same row passes the value-agnostic mode: the modes genuinely differ.
        assert _reasons(TEN_WITH_2DIGIT, FULL10) == ()


class TestModeConfigParsing:
    def test_yaml_block_round_trip(self):
        modes = filter_modes_from_config({
            "prefix": {},
            "full10_3digit": {"min_count": 10, "min_value": 100},
        })
        assert modes["prefix"] == PREFIX
        assert modes["prefix"].require_clean_stop  # strict by default
        assert modes["full10_3digit"] == FULL10_3DIGIT

    def test_unknown_key_rejects(self):
        with pytest.raises(ValueError, match="unknown filter key"):
            filter_config_from_dict({"min_count": 1, "target_filtered": 10_000})

    def test_unsafe_mode_name_rejects(self):
        with pytest.raises(ValueError, match="not a safe dir slug"):
            filter_modes_from_config({"a/b": {}})
        for name in (".", "..", ".hidden"):
            with pytest.raises(ValueError, match="not a safe dir slug"):
                filter_modes_from_config({name: {}})

    @pytest.mark.parametrize(
        ("params", "message"),
        [
            ({"min_count": 0}, "min_count must be positive"),
            ({"min_count": 5, "max_count": 4}, "min_count must be <= max_count"),
            ({"min_value": 10, "max_value": 9}, "min_value must be <= max_value"),
            ({"require_clean_stop": 1}, "require_clean_stop must be a boolean"),
        ],
    )
    def test_invalid_filter_invariants_reject(self, params, message):
        with pytest.raises(ValueError, match=message):
            filter_config_from_dict(params)


class TestGenerationConfigValidation:
    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"n_prompts": 0}, "n_prompts must be positive"),
            (
                {"n_prompts": 1, "example_count_min": 3, "example_count_max": 2},
                "example_count_min must be <= example_count_max",
            ),
            (
                {"n_prompts": 1, "example_value_min": 2, "example_value_max": 1},
                "example_value_min must be <= example_value_max",
            ),
            (
                {"n_prompts": 1, "separator_weights": {"comma": 0.0}},
                "positive total weight",
            ),
            (
                {"n_prompts": 1, "separator_weights": {"comma": -1.0}},
                "finite and non-negative",
            ),
        ],
    )
    def test_invalid_prompt_set_invariants_reject(self, kwargs, message):
        from pipelines.generate_numbers_pipeline import NumberPromptSetConfig

        with pytest.raises(ValueError, match=message):
            NumberPromptSetConfig(templates_path=Path("t.json"), **kwargs)

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"temperature": 0.0}, "temperature must be finite and positive"),
            ({"max_new_tokens": 0}, "max_new_tokens must be positive"),
            ({"batch_size": 0}, "batch_size must be positive"),
            ({"rng_count": 0}, "rng_count must be positive"),
            (
                {"rng_value_min": 2, "rng_value_max": 1},
                "rng_value_min must be <= rng_value_max",
            ),
        ],
    )
    def test_invalid_generation_invariants_reject(self, tmp_path, kwargs, message):
        from pipelines.generate_numbers_pipeline import (
            GenerateNumbersConfig,
            NumberPromptSetConfig,
        )
        from subliminal_learning.config import PathsConfig
        from subliminal_learning.models.registry import PythiaModelSpec

        paths = PathsConfig(tmp_path, tmp_path, tmp_path, tmp_path)
        base = dict(
            paths=paths,
            hub_org="EleutherAI",
            spec=PythiaModelSpec("160m", 1, "data", "main"),
            source="rng",
            out_dir=tmp_path,
            prompt_set=NumberPromptSetConfig(Path("t.json"), 1),
        )
        with pytest.raises(ValueError, match=message):
            GenerateNumbersConfig(**base, **kwargs)


def _write_raw_dataset(d: Path, completions: list[str]) -> None:
    d.mkdir(parents=True)
    rows = [
        {"prompt": f"p{i}", "completion": c, "ended_with_eos": False}
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


class TestApplyFilterModeCaching:
    def test_modes_are_cached_side_by_side(self, tmp_path: Path):
        d = tmp_path / "numbers" / "controls" / "base" / "m"
        _write_raw_dataset(d, [TEN_3DIGIT, TEN_WITH_2DIGIT, SHORT_PROSE, "no numbers"])

        r_prefix = apply_filter_mode(d, "prefix", PREFIX)
        r_strict = apply_filter_mode(d, "full10_3digit", FULL10_3DIGIT)
        assert r_prefix.n_filtered == 2  # strict: prose-tail row rejected too
        assert r_strict.n_filtered == 1
        assert mode_dir(d, "prefix") != mode_dir(d, "full10_3digit")
        assert (mode_dir(d, "prefix") / "filtered.jsonl").exists()
        assert (mode_dir(d, "full10_3digit") / "filtered.jsonl").exists()
        # Mode manifests are self-sufficient dataset descriptors.
        m = json.loads((mode_dir(d, "prefix") / "manifest.json").read_text())
        assert m["base_model_repo"] == "EleutherAI/pythia-160m-data-seed1"
        assert m["params"]["filters"]["require_clean_stop"] is True

    # Cache-skip and loud-redefinition gates are covered (with sha256 assertions) in
    # tests/test_numbers_filter_versioning.py; the older duplicates were removed.

    def test_missing_raw_manifest_raises(self, tmp_path: Path):
        (tmp_path / "raw.jsonl").write_text("")
        with pytest.raises(FileNotFoundError):
            apply_filter_mode(tmp_path, "prefix", PREFIX)


class TestManifestLoading:
    def test_raw_and_mode_rows_coexist(self, tmp_path: Path):
        d = (
            tmp_path / "numbers" / "dog" / "pythia-160m-data-seed1"
            / "ep10_lr2.0e-04_r8a8" / "ftseed0"
        )
        _write_raw_dataset(d, [TEN_3DIGIT, SHORT_PROSE])
        # Make it a teacher dataset so param-tag parsing is exercised.
        manifest = json.loads((d / "manifest.json").read_text())
        manifest["params"].update({"source": "teacher", "target_animal": "dog"})
        (d / "manifest.json").write_text(json.dumps(manifest))
        apply_filter_mode(d, "prefix", PREFIX)

        df = load_number_manifests(tmp_path)
        assert len(df) == 2
        raw_row = df[df["mode"] == "raw"].iloc[0]
        mode_row = df[df["mode"] == "prefix"].iloc[0]
        assert raw_row["source"] == mode_row["source"] == "teacher"
        assert mode_row["n_filtered"] == 1  # strict prefix rejects the prose-tail row
        assert mode_row["param_tag"] == "ep10_lr2.0e-04_r8a8"
        assert mode_row["ftseed"] == 0
        assert mode_row["raw_animal_mention_rate"] == 0.0


class TestPromptSetVariantKeying:
    def test_tag_names_the_example_range(self):
        from pipelines.generate_numbers_pipeline import (
            NumberPromptSetConfig, prompt_set_tag,
        )
        ps = NumberPromptSetConfig(templates_path=Path("t.json"), n_prompts=10)
        assert prompt_set_tag(ps) == "ex10-20"
        wide = NumberPromptSetConfig(
            templates_path=Path("t.json"), n_prompts=10,
            example_count_min=5, example_count_max=30,
        )
        assert prompt_set_tag(wide) == "ex5-30"

    def test_generation_params_key_on_example_range(self, tmp_path: Path):
        # Two prompt-set variants must never satisfy each other's resume gate.
        from dataclasses import replace
        from pipelines.generate_numbers_pipeline import (
            GenerateNumbersConfig, NumberPromptSetConfig, generation_params,
        )
        from subliminal_learning.config import PathsConfig
        from subliminal_learning.models.registry import PythiaModelSpec

        paths = PathsConfig(
            output_root=tmp_path, hf_cache_root=tmp_path, dataset_cache_root=tmp_path,
            wandb_dir=tmp_path,
        )
        spec = PythiaModelSpec(size="160m", seed=1, variant="data", revision="main")
        ps = NumberPromptSetConfig(templates_path=Path("t.json"), n_prompts=10)
        cfg = GenerateNumbersConfig(
            paths=paths, hub_org="EleutherAI", spec=spec, source="base",
            out_dir=tmp_path / "d", prompt_set=ps,
        )
        a = generation_params(cfg, "samehash")
        assert (a["example_count_min"], a["example_count_max"]) == (10, 20)
        wide = replace(cfg, prompt_set=replace(ps, example_count_max=30))
        assert generation_params(wide, "samehash") != a


class TestOverwriteGuard:
    """Resume gate: reuse a covering pool, extend a smaller one, refuse knob changes."""

    @staticmethod
    def _cfg(tmp_path: Path, n_prompts: int, force: bool = False,
             rng_value_max: int = 999, prompt_seed: int = 42):
        from pipelines.generate_numbers_pipeline import (
            GenerateNumbersConfig, NumberPromptSetConfig,
        )
        from subliminal_learning.config import PathsConfig
        from subliminal_learning.models.registry import PythiaModelSpec

        repo = Path(__file__).resolve().parents[1]
        paths = PathsConfig(
            output_root=tmp_path, hf_cache_root=tmp_path, dataset_cache_root=tmp_path,
            wandb_dir=tmp_path,
        )
        return GenerateNumbersConfig(
            paths=paths, hub_org="EleutherAI",
            spec=PythiaModelSpec(size="160m", seed=1, variant="data", revision="main"),
            source="rng", out_dir=tmp_path / "ds",
            prompt_set=NumberPromptSetConfig(
                templates_path=repo / "data" / "number_prompts"
                / "sequence_intro_templates.json",
                n_prompts=n_prompts,
                seed=prompt_seed,
            ),
            filter_modes={"prefix": PREFIX},
            rng_value_max=rng_value_max,
            force=force,
        )

    def test_non_count_param_change_refuses_without_force(self, tmp_path: Path):
        from pipelines.generate_numbers_pipeline import run

        run(self._cfg(tmp_path, 30))
        raw_before = (tmp_path / "ds" / "raw.jsonl").read_text()
        with pytest.raises(ValueError, match="Refusing to overwrite"):
            run(self._cfg(tmp_path, 30, rng_value_max=500))
        # Nothing was touched: raw pool and its mode survive intact.
        assert (tmp_path / "ds" / "raw.jsonl").read_text() == raw_before
        assert (mode_dir(tmp_path / "ds", "prefix") / "filtered.jsonl").exists()

    def test_bigger_pool_on_disk_is_sufficient(self, tmp_path: Path):
        from pipelines.generate_numbers_pipeline import run

        run(self._cfg(tmp_path, 30))
        raw_before = (tmp_path / "ds" / "raw.jsonl").read_text()
        r = run(self._cfg(tmp_path, 20))  # same config, fewer requested -> reuse
        assert r.n_raw == 30
        assert (tmp_path / "ds" / "raw.jsonl").read_text() == raw_before
        assert r.modes["prefix"].n_filtered == 30

    def test_smaller_pool_on_disk_extends_without_force(self, tmp_path: Path):
        from pipelines.generate_numbers_pipeline import run

        run(self._cfg(tmp_path, 20))
        r = run(self._cfg(tmp_path, 30))  # same config, more requested -> regenerate
        assert r.n_raw == 30
        assert len((tmp_path / "ds" / "raw.jsonl").read_text().splitlines()) == 30

    def test_legacy_pool_upgrades_by_full_regeneration(self, tmp_path: Path):
        from pipelines.generate_numbers_pipeline import run

        run(self._cfg(tmp_path, 20))
        manifest_path = tmp_path / "ds" / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        params = manifest["params"]
        params.pop("adapter_id")
        params.pop("example_count_min")
        params.pop("example_count_max")
        params.pop("rng")
        params["filters"] = {
            "min_count": 1, "max_count": 10, "min_value": 0,
            "max_value": 999, "banned_numbers": [],
        }
        manifest_path.write_text(json.dumps(manifest))

        result = run(self._cfg(tmp_path, 30))
        upgraded = json.loads(manifest_path.read_text())
        assert result.n_raw == 30
        assert upgraded["params"]["n_prompts"] == 30
        assert upgraded["params"]["rng"] == {
            "count": 10, "value_min": 0, "value_max": 999,
        }
        assert "filters" not in upgraded["params"]

    def test_unknown_legacy_mismatch_still_refuses(self, tmp_path: Path):
        from pipelines.generate_numbers_pipeline import run

        run(self._cfg(tmp_path, 20))
        manifest_path = tmp_path / "ds" / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        params = manifest["params"]
        params.pop("adapter_id")
        params.pop("example_count_min")
        params.pop("example_count_max")
        params.pop("rng")
        params["filters"] = {
            "min_count": 1, "max_count": 10, "min_value": 0,
            "max_value": 999, "banned_numbers": [],
        }
        params["temperature"] = 0.5
        manifest_path.write_text(json.dumps(manifest))

        with pytest.raises(ValueError, match="Refusing to overwrite"):
            run(self._cfg(tmp_path, 30))

    def test_count_change_hiding_a_knob_change_refuses(self, tmp_path: Path):
        from pipelines.generate_numbers_pipeline import run

        run(self._cfg(tmp_path, 30))
        # Different prompt seed AND different count: the stored hash cannot be
        # reproduced by the current knobs at the stored size -> not an extension.
        with pytest.raises(ValueError, match="Refusing to overwrite"):
            run(self._cfg(tmp_path, 20, prompt_seed=43))

    def test_force_overwrites(self, tmp_path: Path):
        from pipelines.generate_numbers_pipeline import run

        run(self._cfg(tmp_path, 30))
        r = run(self._cfg(tmp_path, 20, force=True))
        assert r.n_raw == 20
        assert len((tmp_path / "ds" / "raw.jsonl").read_text().splitlines()) == 20

    def test_moved_checkpoint_dir_and_missing_top_k_still_match(self, tmp_path: Path):
        """Regression: orig-dog ckpt pools from 08-10 recorded the adapter under ssd-3 and
        predate the recorded ``top_k``. After the cold-1 migration the same adapter (same
        ``adapter_id``) is requested from a new path with ``top_k=50`` -> must reuse."""
        from pipelines.generate_numbers_pipeline import (
            _raw_pool_status, generation_params, run,
        )

        cfg = self._cfg(tmp_path, 30)
        run(cfg)
        manifest_path = tmp_path / "ds" / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        stored = manifest["params"]
        stored["adapter_id"] = "deadbeefcafef00d"
        stored["checkpoint_dir"] = "/mnt/ssd-3/old/checkpoints"
        stored.pop("top_k", None)
        manifest_path.write_text(json.dumps(manifest))

        requested = generation_params(cfg, stored["prompts_hash"])
        requested["adapter_id"] = "deadbeefcafef00d"
        requested["checkpoint_dir"] = "/mnt/cold-1/new/checkpoints"
        assert _raw_pool_status(cfg, requested)[0] == "match"

        # A different adapter at the new path is still a mismatch.
        requested["adapter_id"] = "0123456789abcdef"
        assert _raw_pool_status(cfg, requested)[0] == "mismatch"

        # Without an adapter id the path is the only identity and still counts.
        stored["adapter_id"] = None
        manifest_path.write_text(json.dumps(manifest))
        requested["adapter_id"] = None
        assert _raw_pool_status(cfg, requested)[0] == "mismatch"
        requested["checkpoint_dir"] = "/mnt/ssd-3/old/checkpoints"
        assert _raw_pool_status(cfg, requested)[0] == "match"

    def test_matching_params_still_skips(self, tmp_path: Path):
        from pipelines.generate_numbers_pipeline import run

        run(self._cfg(tmp_path, 30))
        assert run(self._cfg(tmp_path, 30)).n_raw == 30  # skip path, no error

    def test_torn_pool_regenerates_without_force(self, tmp_path: Path):
        from pipelines.generate_numbers_pipeline import run

        run(self._cfg(tmp_path, 30))
        (tmp_path / "ds" / "manifest.json").unlink()  # simulate crash between writes
        r = run(self._cfg(tmp_path, 30))
        assert r.n_raw == 30
        assert (tmp_path / "ds" / "manifest.json").exists()
