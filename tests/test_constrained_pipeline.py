"""Tests for the constrained generation mode wired into the number-gen pipeline.

CPU-only: the model sources need a GPU, so end-to-end coverage uses the rng source
(grammar-matched, model-free) over the real intro-template bank. The rest is pure
config/param/tag logic. The regression FREEZE guards the free-generation (constrained
=None) generation_params against any drift — every on-disk raw pool depends on it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipelines.generate_numbers_pipeline import (
    ConstrainedGenConfig,
    GenerateNumbersConfig,
    NumberPromptSetConfig,
    constrained_from_generation_doc,
    generation_params,
    prompt_set_tag,
    run,
)
from subliminal_learning.config import PathsConfig
from subliminal_learning.models.registry import PythiaModelSpec
from subliminal_learning.numbers.filtering import NumberFilterConfig

REPO = Path(__file__).resolve().parents[1]
TEMPLATES = REPO / "data" / "number_prompts" / "sequence_intro_templates.json"


def _paths(tmp_path: Path) -> PathsConfig:
    return PathsConfig(
        output_root=tmp_path, hf_cache_root=tmp_path,
        dataset_cache_root=tmp_path, wandb_dir=tmp_path,
    )


# --------------------------------------------------------------------------------------
# generation_params: free-generation FREEZE (constrained=None must never drift)
# --------------------------------------------------------------------------------------
def test_generation_params_free_frozen(tmp_path: Path):
    # revision=None keeps base_model_ref out, so the frozen dict is registry-independent.
    spec = PythiaModelSpec(size="160m", seed=1, variant="data", revision=None)
    cfg = GenerateNumbersConfig(
        paths=_paths(tmp_path), hub_org="EleutherAI", spec=spec, source="rng",
        out_dir=tmp_path / "d",
        prompt_set=NumberPromptSetConfig(templates_path=Path("t.json"), n_prompts=10),
    )
    assert generation_params(cfg, "phash0") == {
        "source": "rng",
        "target_animal": None,
        "checkpoint_dir": None,
        "adapter_id": None,
        "prompts_hash": "phash0",
        "n_prompts": 10,
        "example_count_min": 10,
        "example_count_max": 20,
        "temperature": 1.0,
        "max_new_tokens": 40,
        "sample_seed_base": 20260401,
        "batch_size": 128,
        "rng": {"count": 10, "value_min": 0, "value_max": 999},
    }


def test_generation_params_constrained_adds_block(tmp_path: Path):
    spec = PythiaModelSpec(size="160m", seed=1, variant="data", revision=None)
    cfg = GenerateNumbersConfig(
        paths=_paths(tmp_path), hub_org="EleutherAI", spec=spec, source="rng",
        out_dir=tmp_path / "d",
        prompt_set=NumberPromptSetConfig(templates_path=Path("t.json"), n_prompts=10),
        constrained=ConstrainedGenConfig(prompt_style="intro"),
        max_new_tokens=40,  # ignored: derived cap wins
    )
    params = generation_params(cfg, "phash0")
    assert params["constrained"] == {
        "prompt_style": "intro", "count_max": 10, "allow_early_eos": False,
        "value_min": 0, "value_max": 999,
    }
    assert params["max_new_tokens"] == 44  # derived, not the cfg 40
    assert params["rng"] == {"count": 10, "value_min": 0, "value_max": 999}


# --------------------------------------------------------------------------------------
# prompt_set_tag matrix
# --------------------------------------------------------------------------------------
def test_prompt_set_tag_matrix():
    ps = NumberPromptSetConfig(templates_path=Path("t.json"), n_prompts=10)
    assert prompt_set_tag(ps) == "ex10-20"
    assert prompt_set_tag(ps, ConstrainedGenConfig("intro")) == "intro-c10"
    assert prompt_set_tag(
        ps, ConstrainedGenConfig("intro", allow_early_eos=True)
    ) == "intro-u10"
    assert prompt_set_tag(ps, ConstrainedGenConfig("icl")) == "ex10-20-c10"
    assert prompt_set_tag(
        ps, ConstrainedGenConfig("icl", value_min=100, value_max=999)
    ) == "ex10-20-c10x100-999"


# --------------------------------------------------------------------------------------
# constrained_from_generation_doc
# --------------------------------------------------------------------------------------
def test_constrained_from_doc_variants():
    assert constrained_from_generation_doc({"mode": "icl_free"}) is None
    assert constrained_from_generation_doc({}) is None  # default mode

    exact = constrained_from_generation_doc(
        {"mode": "constrained", "constrained": {"prompt_style": "intro", "count_variant": "exact10"}}
    )
    assert exact == ConstrainedGenConfig("intro", count_max=10, allow_early_eos=False)

    upto = constrained_from_generation_doc(
        {"mode": "constrained", "constrained": {"prompt_style": "icl", "count_variant": "upto10"}}
    )
    assert upto == ConstrainedGenConfig("icl", count_max=10, allow_early_eos=True)

    override = constrained_from_generation_doc(
        {"mode": "constrained",
         "constrained": {"prompt_style": "icl", "count_variant": "exact10", "count_max": 5}}
    )
    assert override.count_max == 5


def test_constrained_from_doc_rejections():
    with pytest.raises(ValueError, match="must not carry"):
        constrained_from_generation_doc({"mode": "icl_free", "constrained": {}})
    with pytest.raises(ValueError, match="unknown generation.mode"):
        constrained_from_generation_doc({"mode": "weird"})
    with pytest.raises(ValueError, match="unknown key"):
        constrained_from_generation_doc(
            {"mode": "constrained",
             "constrained": {"prompt_style": "intro", "count_variant": "exact10", "bogus": 1}}
        )
    with pytest.raises(ValueError, match="requires 'prompt_style'"):
        constrained_from_generation_doc(
            {"mode": "constrained", "constrained": {"count_variant": "exact10"}}
        )
    with pytest.raises(ValueError, match="count_variant"):
        constrained_from_generation_doc(
            {"mode": "constrained", "constrained": {"prompt_style": "intro"}}
        )


# --------------------------------------------------------------------------------------
# End-to-end rng constrained run
# --------------------------------------------------------------------------------------
def _rng_cfg(tmp_path: Path, out: str, constrained, mode_name, filters):
    return GenerateNumbersConfig(
        paths=_paths(tmp_path), hub_org="EleutherAI",
        spec=PythiaModelSpec(size="160m", seed=1, variant="data", revision="main"),
        source="rng", out_dir=tmp_path / out,
        prompt_set=NumberPromptSetConfig(templates_path=TEMPLATES, n_prompts=200, seed=42),
        filter_modes={mode_name: filters},
        constrained=constrained,
    )


@pytest.mark.parametrize(
    ("constrained", "mode_name", "filters"),
    [
        (ConstrainedGenConfig("intro"), "full10", NumberFilterConfig(min_count=10)),
        (ConstrainedGenConfig("intro", allow_early_eos=True), "prefix", NumberFilterConfig()),
    ],
)
def test_end_to_end_rng_constrained(tmp_path, constrained, mode_name, filters):
    cfg = _rng_cfg(tmp_path, "ds", constrained, mode_name, filters)
    result = run(cfg)

    assert result.n_raw == 200
    assert result.modes[mode_name].keep_rate == 1.0  # verification, not selection

    manifest = json.loads((cfg.out_dir / "manifest.json").read_text())
    assert manifest["params"]["constrained"]["prompt_style"] == "intro"
    assert manifest["params"]["max_new_tokens"] == 44
    hist = manifest["constrained_count_histogram"]
    assert sum(hist.values()) == 200

    # Rerun: matching manifest -> SKIP, raw.jsonl untouched (mtime preserved).
    raw_path = cfg.out_dir / "raw.jsonl"
    mtime = raw_path.stat().st_mtime_ns
    run(_rng_cfg(tmp_path, "ds", constrained, mode_name, filters))
    assert raw_path.stat().st_mtime_ns == mtime


def test_icl_free_rng_run_has_no_constrained_key(tmp_path):
    cfg = _rng_cfg(tmp_path, "free", None, "prefix", NumberFilterConfig())
    run(cfg)
    manifest = json.loads((cfg.out_dir / "manifest.json").read_text())
    assert "constrained" not in manifest["params"]
    assert "constrained_count_histogram" not in manifest
