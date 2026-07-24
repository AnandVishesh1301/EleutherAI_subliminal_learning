"""Pure-python tests for the preference-eval summary merge + confidence validity key.

No GPU/model: these exercise `_merge_summary`, `_entry_valid`, and `_set_entry`
directly on hand-built dicts (mirroring what `run()` writes).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipelines.evaluate_teacher_preference_pipeline import (  # noqa: E402
    PromptSet,
    SetScore,
    _entry_valid,
    _merge_summary,
    _set_entry,
)
from subliminal_learning.evaluation.preference_scoring import CI


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
