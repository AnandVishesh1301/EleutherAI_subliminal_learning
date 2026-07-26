"""Pure-python tests for the preference-eval summary merge + confidence validity key.

No GPU/model: these exercise `_merge_summary`, `_entry_valid`, and `_set_entry`
directly on hand-built dicts (mirroring what `run()` writes).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pipelines.evaluate_teacher_preference_pipeline as pref_pipeline  # noqa: E402
from pipelines.evaluate_teacher_preference_pipeline import (  # noqa: E402
    EvaluateTeacherPreferenceConfig,
    PromptSet,
    SetScore,
    _discover_ftseed_dirs,
    _entry_valid,
    _merge_summary,
    _set_entry,
)
from subliminal_learning.config import PathsConfig  # noqa: E402
from subliminal_learning.evaluation.preference_scoring import CI
from subliminal_learning.models.registry import PythiaModelSpec  # noqa: E402


def _prob_set(name: str = "favorite") -> PromptSet:
    return PromptSet(name=name, path=Path(f"/tmp/{name}.json"), max_new_tokens=12, scoring="prob")


def _set_score() -> SetScore:
    return SetScore(
        ci=CI(mean=0.4, lower=0.3, upper=0.5, count=5, confidence=0.95),
        rank_mean=2.0,
        rank1_frac=0.5,
        p_per_prompt=(0.4,),
    )


def _summary_dict(name: str, animal: str = "owl") -> dict:
    """A minimal per-set summary dict as `run()` would serialize (asdict)."""
    return {
        "prompt_set": name,
        "scoring": "prob",
        "n_prompts": 3,
        "n_seeds": 1,
        "p_target_mean": 0.4,
        "ci_lower": 0.3,
        "ci_upper": 0.5,
        "per_seed_p_target": {"42": 0.4},
        "rank_mean": 2.0,
        "rank1_frac": 0.5,
        "per_seed_rank_mean": {"42": 2.0},
        "base_p_target": 0.1,
        "base_rank_mean": 8.0,
        "delta_vs_base": 0.3,
    }


def _report(active_names: list[str], animal: str = "owl", ft_seeds=(42,)) -> dict:
    return {
        "target_animal": animal,
        "base_model": "pythia-160m",
        "base_model_repo": "EleutherAI/pythia-160m",
        "param_dir": "/out/owl/param",
        "ft_seeds": tuple(ft_seeds),
        "confidence": 0.95,
        "target_forms": [],
        "summaries": tuple(_summary_dict(n, animal) for n in active_names),
        "adapter_ids": {str(s): f"aid{s}" for s in ft_seeds},
        "set_params": {
            n: {"scoring": "prob", "prompts_hash": f"h_{n}", "confidence": 0.95}
            for n in active_names
        },
    }


# ---- _merge_summary ---------------------------------------------------------


def test_merge_retains_non_active_and_replaces_active():
    stored = _report(["favorite", "dislike"])
    # Prior run scored favorite/dislike; now a narrowed rerun only touches "favorite"
    # with fresh values.
    report = _report(["favorite"])
    report["summaries"][0]["p_target_mean"] = 0.9  # freshly recomputed
    report["set_params"]["favorite"]["prompts_hash"] = "h_new"

    merged = _merge_summary(stored, report, {"favorite"})

    names = [s["prompt_set"] for s in merged["summaries"]]
    assert names == ["dislike", "favorite"]  # name-sorted, both retained
    fav = next(s for s in merged["summaries"] if s["prompt_set"] == "favorite")
    assert fav["p_target_mean"] == 0.9  # active set replaced
    # non-active "dislike" kept verbatim
    assert merged["set_params"]["dislike"] == stored["set_params"]["dislike"]
    # active "favorite" set_params updated
    assert merged["set_params"]["favorite"]["prompts_hash"] == "h_new"


def test_merge_empty_stored_returns_report():
    report = _report(["favorite"])
    assert _merge_summary({}, report, {"favorite"}) is report


def test_merge_differing_animal_overwrites_wholesale():
    stored = _report(["favorite", "dislike"], animal="cat")
    report = _report(["favorite"], animal="owl")
    merged = _merge_summary(stored, report, {"favorite"})
    assert merged is report  # no merge; the legacy cat file is discarded
    assert [s["prompt_set"] for s in merged["summaries"]] == ["favorite"]


def test_merge_adapter_ids_and_ft_seeds():
    stored = _report(["favorite"], ft_seeds=(42, 7))
    report = _report(["favorite"], ft_seeds=(42, 99))
    report["adapter_ids"]["42"] = "aid42_new"  # current wins per seed
    merged = _merge_summary(stored, report, {"favorite"})
    assert merged["ft_seeds"] == [7, 42, 99]  # union, sorted
    assert merged["adapter_ids"]["42"] == "aid42_new"
    assert merged["adapter_ids"]["7"] == "aid7"  # stored-only seed retained
    assert merged["adapter_ids"]["99"] == "aid99"


def test_merge_top_level_fields_current_wins():
    stored = _report(["favorite"])
    stored["confidence"] = 0.90
    stored["base_model_repo"] = "old/repo"
    report = _report(["favorite"])
    merged = _merge_summary(stored, report, {"favorite"})
    assert merged["confidence"] == 0.95
    assert merged["base_model_repo"] == "EleutherAI/pythia-160m"


# ---- _entry_valid confidence key -------------------------------------------


def test_entry_valid_absent_confidence_reads_as_default():
    ps = _prob_set()
    entry = _set_entry(_set_score(), ps, 0, 0.0, "h", "owl", (), 0.95)
    del entry["confidence"]  # simulate a legacy entry predating the confidence key
    # At the historical default, absence must be treated as valid (no invalidation).
    assert _entry_valid(entry, ps, 0, 0.0, "h", "owl", (), 0.95)
    # At a non-default confidence, the same legacy entry is invalid.
    assert not _entry_valid(entry, ps, 0, 0.0, "h", "owl", (), 0.90)


def test_entry_valid_matching_explicit_confidence():
    ps = _prob_set()
    entry = _set_entry(_set_score(), ps, 0, 0.0, "h", "owl", (), 0.90)
    assert _entry_valid(entry, ps, 0, 0.0, "h", "owl", (), 0.90)
    assert not _entry_valid(entry, ps, 0, 0.0, "h", "owl", (), 0.95)


def test_set_entry_writes_confidence():
    entry = _set_entry(_set_score(), _prob_set(), 0, 0.0, "h", "owl", (), 0.80)
    assert entry["confidence"] == 0.80


# ---- _discover_ftseed_dirs: incomplete-checkpoint quarantine ----------------


def _make_seed(param_dir: Path, seed: int, *, prefix: str = "ftseed",
               weights: bool = True, adapter_cfg: bool = True) -> Path:
    """Create a ``{prefix}{seed}/checkpoints`` dir; `weights` gates the saved adapter."""
    ckpt = param_dir / f"{prefix}{seed}" / "checkpoints"
    ckpt.mkdir(parents=True)
    if adapter_cfg:
        (ckpt / "adapter_config.json").write_text("{}")
    if weights:
        (ckpt / "adapter_model.safetensors").write_bytes(b"w")
    return ckpt


def test_discover_skips_incomplete_and_keeps_complete(tmp_path: Path):
    param_dir = tmp_path / "param"
    complete = _make_seed(param_dir, 0, weights=True)
    _make_seed(param_dir, 1, weights=False)  # interrupted: config but no weights
    skipped: list[int] = []
    found = _discover_ftseed_dirs(param_dir, "ftseed", skipped)
    assert found == [(0, complete)]  # the good seed is still evaluable
    assert skipped == [1]  # the interrupted one is recorded, not silently dropped


def test_discover_adapter_config_only_is_incomplete(tmp_path: Path):
    param_dir = tmp_path / "param"
    _make_seed(param_dir, 3, weights=False, adapter_cfg=True)
    skipped: list[int] = []
    assert _discover_ftseed_dirs(param_dir, "ftseed", skipped) == []
    assert skipped == [3]


def test_discover_all_complete_unchanged(tmp_path: Path):
    param_dir = tmp_path / "param"
    c0 = _make_seed(param_dir, 0, weights=True)
    c1 = _make_seed(param_dir, 1, weights=True)
    skipped: list[int] = []
    found = _discover_ftseed_dirs(param_dir, "ftseed", skipped)
    assert found == [(0, c0), (1, c1)]
    assert skipped == []  # nothing quarantined -> empty


def test_discover_gseed_prefix_also_quarantined(tmp_path: Path):
    param_dir = tmp_path / "param"
    good = _make_seed(param_dir, 5, prefix="gseed", weights=True)
    _make_seed(param_dir, 6, prefix="gseed", weights=False)
    skipped: list[int] = []
    found = _discover_ftseed_dirs(param_dir, "gseed", skipped)
    assert found == [(5, good)]
    assert skipped == [6]


def test_discover_missing_checkpoints_dir_still_skipped(tmp_path: Path):
    param_dir = tmp_path / "param"
    (param_dir / "ftseed0").mkdir(parents=True)  # no checkpoints/ subdir at all
    good = _make_seed(param_dir, 1, weights=True)
    skipped: list[int] = []
    found = _discover_ftseed_dirs(param_dir, "ftseed", skipped)
    assert found == [(1, good)]
    assert skipped == []  # a missing checkpoints/ dir is a different skip, not "incomplete"


# ---- run(): all-incomplete cell raises the improved message -----------------


def _min_cfg(param_dir: Path, tmp_path: Path) -> EvaluateTeacherPreferenceConfig:
    prompt_file = tmp_path / "favorite.json"
    prompt_file.write_text('["My favorite animal is the"]')
    return EvaluateTeacherPreferenceConfig(
        paths=PathsConfig(
            output_root=tmp_path / "out", hf_cache_root=tmp_path / "hf",
            dataset_cache_root=tmp_path / "data", wandb_dir=tmp_path / "wandb",
        ),
        hub_org="EleutherAI",
        spec=PythiaModelSpec(size="160m", seed=1, variant="data", revision="main"),
        target_animal="owl", param_dir=param_dir,
        prompt_sets=(PromptSet("favorite", prompt_file, 12, "prob"),),
        forms=None, n_samples=100, temperature=1.0, include_base_control=False,
        confidence=0.95, device="cpu",
    )


def test_run_all_incomplete_raises_with_skip_count(tmp_path: Path, monkeypatch):
    param_dir = tmp_path / "param"
    _make_seed(param_dir, 0, weights=False)
    _make_seed(param_dir, 1, weights=False)
    # Stub the model-touching entry points so we reach the discovery gate on CPU-only.
    monkeypatch.setattr(pref_pipeline, "load_tokenizer", lambda *a, **k: object())
    monkeypatch.setattr(pref_pipeline, "target_token_id", lambda *a, **k: 0)
    with pytest.raises(FileNotFoundError) as exc:
        pref_pipeline.run(_min_cfg(param_dir, tmp_path))
    msg = str(exc.value)
    assert "2 skipped as incomplete" in msg  # explains it is not "no seeds exist"
    assert "[0, 1]" in msg


# ---- _merge_summary: skipped_incomplete_seeds refreshed, never stale --------


def test_merge_refreshes_skipped_incomplete_seeds():
    stored = _report(["favorite"])
    stored["skipped_incomplete_seeds"] = [2]  # a prior partial run
    report = _report(["favorite"])
    report["skipped_incomplete_seeds"] = []  # later run found seed 2 complete
    merged = _merge_summary(stored, report, {"favorite"})
    assert merged["skipped_incomplete_seeds"] == []  # current wins; stale [2] cleared


def test_merge_all_complete_schema_unchanged():
    # An all-complete report carries only an empty skipped field; every prior key survives.
    stored = _report(["favorite"])
    report = _report(["favorite"])
    report["skipped_incomplete_seeds"] = []
    merged = _merge_summary(stored, report, {"favorite"})
    for key in ("target_animal", "base_model", "ft_seeds", "summaries",
                "set_params", "adapter_ids", "confidence"):
        assert key in merged
    assert merged["skipped_incomplete_seeds"] == []
