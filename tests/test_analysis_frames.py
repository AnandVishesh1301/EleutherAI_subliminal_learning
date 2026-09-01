"""Unit tests for ``subliminal_learning.analysis.frames`` + ``.setup`` (CPU-only, synthetic frames).

No reads from ``outputs/`` and no model loads: every frame is built inline, and the setup tests
plant a fake ``configs/paths/paths_local.yaml`` under ``tmp_path`` to exercise the upward walk.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from subliminal_learning.analysis.frames import (
    add_lora_label,
    add_param_fields,
    cheapest_best,
    cheapest_specific,
    load_preference_evals,
    load_queue_results,
    load_results,
    load_run_docs,
    load_steerability_aggregates,
    rank_specific,
    filter_grid,
    parse_param_tag,
    pref_to_wide,
    read_cell_seeds,
    read_cell_set,
    read_json,
    safe_read_json,
    scope_str,
    short_model,
)
from subliminal_learning.analysis.setup import bootstrap, find_root, set_plot_style


# ------------------------- short_model -------------------------


def test_short_model_original():
    assert short_model("pythia-160m") == "original"


def test_short_model_strips_prefix():
    assert short_model("pythia-160m-data-seed1") == "data-seed1"


def test_short_model_full_seed_is_labelled_explicitly():
    # A fully re-randomized sibling must never read as a bare `seed2` (which would
    # look like a data-/weight-seed arm); the label matches `spec_label`.
    assert short_model("pythia-160m-seed2") == "full-seed2"
    assert short_model("pythia-160m-seed2@step3000") == "full-seed2@step3000"


def test_short_model_passthrough_unknown():
    assert short_model("some-other-model") == "some-other-model"


def test_short_model_datadecide_warns_and_passes_through():
    # dd rows would silently drop out of 160m-keyed joins, so they must be loud.
    name = "DataDecide-dolma1_7-1B-seed-1"
    with pytest.warns(UserWarning, match="not a pythia-160m-family name"):
        assert short_model(name) == name


def test_short_model_160m_family_is_warning_free(recwarn):
    assert short_model("pythia-160m") == "original"
    assert short_model("pythia-160m-data-seed1") == "data-seed1"
    assert len(recwarn) == 0


# ------------------------- add_lora_label -------------------------


def test_add_lora_label_numeric_order():
    df = pd.DataFrame(
        {"lora_r": [16, 8, 8], "lora_alpha": [8, 32, 16]}
    )
    out = add_lora_label(df)
    cats = list(out["lora_label"].cat.categories)
    # Numeric (rank, alpha) order, NOT lexical (which would put r16 before r8).
    assert cats == ["r8/α16", "r8/α32", "r16/α8"]
    assert out["lora_label"].cat.ordered
    # Row labels line up with their (r, alpha).
    assert out["lora_label"].tolist() == ["r16/α8", "r8/α32", "r8/α16"]


def test_add_lora_label_does_not_mutate_input():
    df = pd.DataFrame({"lora_r": [8], "lora_alpha": [16]})
    add_lora_label(df)
    assert "lora_label" not in df.columns


def test_add_lora_label_nan_rows_get_nan_label():
    # Rows with a missing lora_r/lora_alpha (e.g. pools without a teacher param tag) must not
    # crash on int() and must get a NaN label; categories come from the finite pairs only.
    df = pd.DataFrame(
        {"lora_r": [8, np.nan, 16, 8], "lora_alpha": [16, 32, np.nan, 32]}
    )
    out = add_lora_label(df)
    cats = list(out["lora_label"].cat.categories)
    assert cats == ["r8/α16", "r8/α32"]  # only the two fully-finite pairs
    assert out["lora_label"].cat.ordered
    labels = out["lora_label"].tolist()
    assert labels[0] == "r8/α16"
    assert labels[3] == "r8/α32"
    # NaN in either column -> NaN label.
    assert pd.isna(labels[1]) and pd.isna(labels[2])


def test_add_lora_label_all_finite_unchanged():
    # All-finite behavior is identical to before the NaN-tolerance change.
    df = pd.DataFrame({"lora_r": [16, 8, 8], "lora_alpha": [8, 32, 16]})
    out = add_lora_label(df)
    assert list(out["lora_label"].cat.categories) == ["r8/α16", "r8/α32", "r16/α8"]
    assert out["lora_label"].tolist() == ["r16/α8", "r8/α32", "r8/α16"]


# ------------------------- filter_grid -------------------------


def test_filter_grid_float_tolerant_lr():
    df = pd.DataFrame(
        {
            "n_epochs": [1, 5, 10],
            "learning_rate": [1e-5, 0.0002000000001, 5e-4],
            "lora_r": [8, 16, 32],
        }
    )
    out = filter_grid(df, lrs=[2e-4])
    assert out["learning_rate"].tolist() == [0.0002000000001]


def test_filter_grid_none_axes_untouched_and_no_mutation():
    df = pd.DataFrame(
        {"n_epochs": [1, 5], "learning_rate": [1e-5, 1e-4], "lora_r": [8, 16]}
    )
    before = df.copy()
    out = filter_grid(df, epochs=[1])
    assert out["n_epochs"].tolist() == [1]
    # lr / rank untouched; original frame unchanged.
    pd.testing.assert_frame_equal(df, before)


def test_filter_grid_ranks_and_epochs():
    df = pd.DataFrame(
        {
            "n_epochs": [1, 5, 5, 10],
            "learning_rate": [1e-4, 1e-4, 1e-4, 1e-4],
            "lora_r": [8, 8, 16, 64],
        }
    )
    out = filter_grid(df, epochs=[5], ranks=[8, 16])
    assert sorted(out["lora_r"].tolist()) == [8, 16]
    assert set(out["n_epochs"]) == {5}


# ------------------------- pref_to_wide -------------------------


def _pref_row(animal, model, tag, ep, lr, r, a, prompt_set, p, base_p, **extra):
    row = {
        "animal": animal,
        "model": model,
        "param_tag": tag,
        "n_epochs": ep,
        "learning_rate": lr,
        "lora_r": r,
        "lora_alpha": a,
        "lora_alpha_over_r": a / r,
        "set": prompt_set,
        "p_target": p,
        "base_p_target": base_p,
        "base_rank_mean": extra.get("base_rank_mean", 5.0),
        "rank_mean": extra.get("rank_mean", 1.0),
    }
    return row


def _small_pref(sets=("favorite", "validation", "off_topic")):
    rows = []
    for prompt_set, p, bp in [
        ("favorite", 0.8, 0.02),
        ("validation", 0.6, 0.01),
        ("off_topic", 0.05, 0.01),
        ("storytelling", 0.3, 0.1),
    ]:
        if prompt_set in sets:
            rows.append(
                _pref_row("cat", "pythia-160m-data-seed1", "ep5_lr1e-4_r8a16", 5,
                          1e-4, 8, 16, prompt_set, p, bp)
            )
    return add_lora_label(pd.DataFrame(rows))


def test_pref_to_wide_columns_and_derived():
    wide = pref_to_wide(_small_pref())
    assert len(wide) == 1
    row = wide.iloc[0]
    # per-set p_target columns + base_ columns.
    assert row["favorite"] == pytest.approx(0.8)
    assert row["validation"] == pytest.approx(0.6)
    assert row["off_topic"] == pytest.approx(0.05)
    assert row["base_off_topic"] == pytest.approx(0.01)
    # derived deltas.
    assert row["off_topic_delta"] == pytest.approx(0.05 - 0.01)
    assert row["specificity_margin"] == pytest.approx(0.6 - 0.05)
    # rank_mean merged from favorite; short_model added.
    assert row["rank_mean"] == pytest.approx(1.0)
    assert row["short_model"] == "data-seed1"
    # No old aliases.
    assert "base_p_target" not in wide.columns
    assert "base" not in wide.columns


def test_pref_to_wide_missing_storytelling_no_column():
    wide = pref_to_wide(_small_pref(sets=("favorite", "validation", "off_topic")))
    assert "storytelling_delta" not in wide.columns


def test_pref_to_wide_duplicate_rows_assert():
    pref = _small_pref()
    dup = pd.concat([pref, pref[pref["set"] == "favorite"]], ignore_index=True)
    with pytest.raises(AssertionError):
        pref_to_wide(dup)


# ------------------------- cheapest_best -------------------------


def _config_df():
    return pd.DataFrame(
        {
            "n_epochs": [5, 1, 10, 5],
            "learning_rate": [1e-4, 1e-4, 1e-4, 5e-4],
            "lora_r": [8, 8, 8, 8],
            "lora_alpha": [16, 16, 16, 16],
            "validation": [0.80, 0.79, 0.81, np.nan],
        }
    )


def test_cheapest_best_tol_admits_cheaper():
    # optimum = 0.81 (ep10). With tol 0.02, ep1 (0.79) qualifies and is cheapest.
    best = cheapest_best(_config_df(), "validation", tol=0.02)
    assert best["n_epochs"] == 1


def test_cheapest_best_exact_optimum_when_no_tol():
    best = cheapest_best(_config_df(), "validation")
    assert best["n_epochs"] == 10


def test_cheapest_best_lower_is_better():
    df = pd.DataFrame(
        {
            "n_epochs": [1, 5],
            "learning_rate": [1e-4, 1e-4],
            "lora_r": [8, 8],
            "lora_alpha": [16, 16],
            "off_topic": [0.30, 0.05],
        }
    )
    best = cheapest_best(df, "off_topic", lower_is_better=True)
    assert best["n_epochs"] == 5


def test_cheapest_best_nan_excluded():
    best = cheapest_best(_config_df(), "validation", tol=1.0)
    # The NaN row (ep5/5e-4) must not be selectable even though it sorts cheaply.
    assert not np.isnan(best["validation"])


def test_cheapest_best_empty_raises():
    df = pd.DataFrame(
        {
            "n_epochs": [1],
            "learning_rate": [1e-4],
            "lora_r": [8],
            "lora_alpha": [16],
            "validation": [np.nan],
        }
    )
    with pytest.raises(ValueError):
        cheapest_best(df, "validation")


# ------------------------- cheapest_specific -------------------------


def _specific_df():
    # ep1 & ep5 are near-tied on validation (both round to 1.00) and specific; ep10 is the raw
    # max but leaks off-distribution, so the control excludes it.
    return pd.DataFrame(
        {
            "n_epochs": [1, 5, 10],
            "learning_rate": [1e-4, 1e-4, 1e-4],
            "lora_r": [8, 8, 8],
            "lora_alpha": [16, 16, 16],
            "validation": [0.998, 0.999, 1.000],
            "off_topic": [0.02, 0.05, 0.60],
        }
    )


def test_cheapest_specific_prefers_cheaper_among_rounded_ties():
    # ep10 (raw max) violates ctrl_max; ep1 (0.998) and ep5 (0.999) both round to 1.00 -> tie, so
    # the cheaper ep1 wins even though ep5 edges it on raw validation.
    pick = cheapest_specific(_specific_df(), ctrl_max=0.1)
    assert pick["n_epochs"] == 1


def test_cheapest_specific_defaults_to_off_topic_control():
    # Default controls=("off_topic",): the ctrl_max=0.1 default excludes ep10 (0.60).
    pick = cheapest_specific(_specific_df())
    assert pick["n_epochs"] == 1 and pick["off_topic"] <= 0.1


def test_cheapest_specific_returns_none_when_no_config_qualifies():
    assert cheapest_specific(_specific_df(), ctrl_max=0.0) is None


def test_cheapest_specific_excludes_nan_target_or_control():
    df = _specific_df()
    df.loc[df.n_epochs == 1, "off_topic"] = np.nan  # ep1 no longer eligible
    pick = cheapest_specific(df, ctrl_max=0.1)
    assert pick["n_epochs"] == 5


def test_rank_specific_dense_rank_and_cost_tiebreak():
    ranked = rank_specific(_specific_df(), ctrl_max=0.1)
    assert list(ranked["n_epochs"]) == [1, 5]   # ep10 excluded; cheapest of the tie first
    assert list(ranked["rank"]) == [1, 1]       # both round to 1.00 -> share rank 1
    # full precision preserved (ranking rounds internally only)
    assert ranked.iloc[0]["validation"] == 0.998


def test_rank_specific_decimals_split_ties():
    # At 3 decimals 0.998 != 0.999 -> distinct ranks; ep5 (0.999) now outranks ep1 (0.998).
    ranked = rank_specific(_specific_df(), ctrl_max=0.1, rank_decimals=3)
    assert list(ranked["rank"]) == [1, 2]
    assert ranked.iloc[0]["n_epochs"] == 5


def test_rank_specific_empty_has_rank_column():
    ranked = rank_specific(_specific_df(), ctrl_max=0.0)
    assert ranked.empty and "rank" in ranked.columns


# ------------------------- scope_str -------------------------


def test_scope_str_empty_df():
    assert scope_str(pd.DataFrame()) == (
        "0 configs | 0 animal(s) x 0 base model(s) | seeds pre-averaged"
    )


def test_scope_str_counts():
    df = pd.DataFrame(
        {"animal": ["cat", "dog"], "model": ["pythia-160m", "pythia-160m"]}
    )
    assert scope_str(df) == "2 configs | 2 animal(s) x 1 base model(s) | seeds pre-averaged"


# ------------------------- setup -------------------------


def test_find_root_walks_up(tmp_path):
    root = tmp_path / "repo"
    (root / "configs" / "paths").mkdir(parents=True)
    (root / "configs" / "paths" / "paths_local.yaml").write_text("output_root: /data/out\n")
    nested = root / "notebooks" / "deep"
    nested.mkdir(parents=True)
    assert find_root(nested) == root


def test_find_root_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_root(tmp_path)


def test_bootstrap_reads_output_root(tmp_path):
    root = tmp_path / "repo"
    (root / "configs" / "paths").mkdir(parents=True)
    (root / "configs" / "paths" / "paths_local.yaml").write_text("output_root: /data/out\n")
    got_root, out = bootstrap(root)
    assert got_root == root
    assert out == Path("/data/out")


# ------------------------- parse_param_tag -------------------------


def test_parse_param_tag_bare_keys_and_types():
    got = parse_param_tag("ep10_lr1e-4_r16a32")
    assert got == {"epochs": 10, "lr": 1e-4, "rank": 16, "alpha": 32}
    assert isinstance(got["epochs"], int) and isinstance(got["lr"], float)


def test_parse_param_tag_prefix():
    assert parse_param_tag("ep2_lr5e-4_r8a16", "teacher") == {
        "teacher_epochs": 2, "teacher_lr": 5e-4, "teacher_rank": 8, "teacher_alpha": 16,
    }


def test_parse_param_tag_optim_suffix_round_trips():
    # Suffix-less tag: byte-identical result to before (no optim keys).
    assert parse_param_tag("ep20_lr1e-4_r8a16") == {
        "epochs": 20, "lr": 1e-4, "rank": 8, "alpha": 16,
    }
    # Suffixed tag: the base fields are unchanged and optim fields are added.
    assert parse_param_tag("ep20_lr1e-4_r8a16_sgd") == {
        "epochs": 20, "lr": 1e-4, "rank": 8, "alpha": 16,
        "optim": "sgd", "optim_momentum": 0.0,
    }


def test_parse_param_tag_rejects_malformed():
    with pytest.raises(ValueError):
        parse_param_tag("from_ep10_lr1e-4_r16a32_tseed1")
    # A teacher-tag lineage dir (`{tag}_tseed{N}`) must not parse as a student param tag.
    with pytest.raises(ValueError):
        parse_param_tag("ep10_lr1e-4_r16a32_tseed1")


# ------------------------- set_plot_style -------------------------


def test_set_plot_style_hide_spines_toggles_rcparams():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    set_plot_style(dpi=115, hide_spines=True)
    assert plt.rcParams["figure.dpi"] == 115
    assert plt.rcParams["axes.spines.top"] is False
    assert plt.rcParams["axes.spines.right"] is False
    # Default leaves spines on.
    set_plot_style()
    assert plt.rcParams["axes.spines.top"] is True


# ------------------------- read_cell_set / read_cell_seeds -------------------------


def _write_summary(cell_dir: Path, per_seed: dict[str, float]) -> Path:
    """Plant a minimal ``preference_summary.json`` (test + storytelling blocks) in `cell_dir`."""
    cell_dir.mkdir(parents=True, exist_ok=True)
    doc = {
        "summaries": [
            {"prompt_set": "storytelling", "scoring": "sample", "n_seeds": 1,
             "p_target_mean": 0.05, "per_seed_p_target": {"1": 0.05}},
            {"prompt_set": "test", "scoring": "prob", "n_seeds": len(per_seed),
             "p_target_mean": float(np.mean(list(per_seed.values()))),
             "per_seed_p_target": per_seed},
        ]
    }
    (cell_dir / "preference_summary.json").write_text(json.dumps(doc))
    return cell_dir


def test_read_cell_set_picks_the_named_prompt_set(tmp_path):
    cell = _write_summary(tmp_path / "ep20_lr4e-4_r32a4", {"1": 0.3, "2": 0.1, "3": 0.2})
    got = read_cell_set(cell, "test")
    assert got["scoring"] == "prob" and got["n_seeds"] == 3
    assert read_cell_set(cell, "storytelling")["scoring"] == "sample"


def test_read_cell_set_missing_file_or_set_is_none(tmp_path):
    assert read_cell_set(tmp_path / "never_evaluated") is None
    cell = _write_summary(tmp_path / "ep20_lr1e-4_r8a32", {"1": 0.4})
    assert read_cell_set(cell, "off_topic") is None


def test_read_cell_seeds_returns_every_seed_typed(tmp_path):
    cell = _write_summary(tmp_path / "ep20_lr4e-4_r16a4", {"1": 0.3, "2": 0.1, "3": 0.2})
    seeds = read_cell_seeds(cell)
    assert seeds == {1: 0.3, 2: 0.1, 3: 0.2}
    assert all(isinstance(k, int) for k in seeds)
    # len() is the "n" a seed-averaged cell should print alongside its mean.
    assert len(seeds) == 3
    assert np.isclose(np.mean(list(seeds.values())), 0.2)


def test_read_cell_seeds_unevaluated_cell_is_empty(tmp_path):
    assert read_cell_seeds(tmp_path / "ep20_lr1e-4_r8a32") == {}


# ------------------------- read_json / safe_read_json -------------------------


def test_read_json_round_trips(tmp_path):
    p = tmp_path / "doc.json"
    p.write_text(json.dumps({"a": 1, "b": [2, 3]}))
    assert read_json(p) == {"a": 1, "b": [2, 3]}


def test_safe_read_json_missing_file_is_none(tmp_path):
    assert safe_read_json(tmp_path / "nope.json") is None


def test_safe_read_json_bad_json_is_none(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not valid json")
    assert safe_read_json(p) is None


# ------------------------- add_param_fields -------------------------


def test_add_param_fields_mixed_tags_bare_names():
    df = pd.DataFrame({"param_tag": ["ep10_lr1e-4_r16a32", "from_ep1_bad", "ep2_lr5e-4_r8a16_sgd"]})
    out = add_param_fields(df, "param_tag")
    # matching row -> typed fields; non-matching -> NaN across all four.
    assert out.loc[0, "epochs"] == 10
    assert out.loc[0, "lr"] == pytest.approx(1e-4)
    assert out.loc[0, "rank"] == 16
    assert out.loc[0, "alpha"] == 32
    assert np.isnan(out.loc[1, "epochs"]) and np.isnan(out.loc[1, "lr"])
    # optimizer-suffixed tag still yields the four core fields.
    assert out.loc[2, "epochs"] == 2 and out.loc[2, "rank"] == 8 and out.loc[2, "alpha"] == 16
    assert out.loc[2, "lr"] == pytest.approx(5e-4)


def test_add_param_fields_prefix_and_no_mutation():
    df = pd.DataFrame({"tag": ["ep3_lr2e-4_r32a4"]})
    before = df.copy()
    out = add_param_fields(df, "tag", prefix="teacher")
    assert out.loc[0, "teacher_epochs"] == 3
    assert out.loc[0, "teacher_rank"] == 32
    assert out.loc[0, "teacher_alpha"] == 4
    assert out.loc[0, "teacher_lr"] == pytest.approx(2e-4)
    pd.testing.assert_frame_equal(df, before)  # input untouched


# ------------------------- load_results -------------------------


def _write_results(path: Path, rows: list[dict], key: str = "results") -> Path:
    path.write_text(json.dumps({key: rows, "meta": {"n": len(rows)}}))
    return path


def test_load_results_multi_file_concat_and_per_file(tmp_path):
    a = _write_results(tmp_path / "a.json", [{"x": 1}, {"x": 2}])
    b = _write_results(tmp_path / "b.json", [{"x": 3}])

    def attach(path, doc):
        return {"src": path.stem, "n": doc["meta"]["n"]}

    df = load_results([a, b], per_file=attach)
    assert len(df) == 3
    assert df["x"].tolist() == [1, 2, 3]
    assert df["src"].tolist() == ["a", "a", "b"]
    assert df["n"].tolist() == [2, 2, 1]


def test_load_results_single_path_not_iterated_charwise(tmp_path):
    a = _write_results(tmp_path / "solo.json", [{"x": 9}])
    df = load_results(a)  # a bare Path, not a list
    assert df["x"].tolist() == [9]


def test_load_results_skips_unreadable_and_warns(tmp_path):
    good = _write_results(tmp_path / "good.json", [{"x": 1}])
    bad = tmp_path / "bad.json"
    bad.write_text("{broken")
    with pytest.warns(UserWarning, match="skipped 1"):
        df = load_results([bad, good])
    assert df["x"].tolist() == [1]


def test_load_results_empty_when_nothing_loads(tmp_path):
    with pytest.warns(UserWarning, match="skipped 1"):
        df = load_results([tmp_path / "missing.json"])
    assert df.empty


# ------------------------- batched doc loaders -------------------------


def _plant(root: Path, rel: str, rel_record: str, doc) -> str:
    """Write `doc` (dict or raw text) at ``{root}/{rel}/{rel_record}``; return `rel`."""
    p = root / rel / rel_record
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(doc if isinstance(doc, str) else json.dumps(doc))
    return rel


def test_load_run_docs_round_trip_and_skips(tmp_path):
    a = _plant(tmp_path, "runs/a", "rec.json", {"v": 1})
    b = _plant(tmp_path, "runs/b", "rec.json", {"v": 2})
    bad = _plant(tmp_path, "runs/bad", "rec.json", "{broken")
    docs = load_run_docs(tmp_path, [a, b, bad, "runs/missing"], "rec.json", "unit")
    assert docs == {a: {"v": 1}, b: {"v": 2}}
    assert (tmp_path / "analysis" / "record_cache_unit.csv").is_file()
    # second call is served from the cache and returns the same mapping
    assert load_run_docs(tmp_path, [a, b], "rec.json", "unit") == {a: {"v": 1}, b: {"v": 2}}


def test_load_run_docs_sees_rewritten_record(tmp_path):
    a = _plant(tmp_path, "runs/a", "rec.json", {"v": 1})
    load_run_docs(tmp_path, [a], "rec.json", "unit")
    _plant(tmp_path, "runs/a", "rec.json", {"v": 1, "extra": True})
    assert load_run_docs(tmp_path, [a], "rec.json", "unit")[a] == {"v": 1, "extra": True}


def test_load_preference_evals(tmp_path):
    a = _plant(tmp_path, "cell/ftseed1", "preference_eval.json", {"p_target": 0.4})
    b = _plant(tmp_path, "cell/ftseed2", "preference_eval.json", {"p_target": 0.6})
    docs = load_preference_evals(tmp_path, [a, b, "cell/ftseed3"])
    assert docs == {a: {"p_target": 0.4}, b: {"p_target": 0.6}}
    assert (tmp_path / "analysis" / "record_cache_pref_eval.csv").is_file()


def test_load_steerability_aggregates(tmp_path):
    rel = _plant(tmp_path, "teachers/t/steerability/steer_x", "aggregate.json", {"mean": 1.5})
    docs = load_steerability_aggregates(tmp_path, [rel])
    assert docs == {rel: {"mean": 1.5}}
    assert (tmp_path / "analysis" / "record_cache_steerability_agg.csv").is_file()


def _plant_result(queue: Path, jobkey: str, doc) -> None:
    """Write one jobq terminal result record at ``{queue}/results/{jobkey}.json``."""
    p = queue / "results" / f"{jobkey}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(doc if isinstance(doc, str) else json.dumps(doc))


def test_load_queue_results_round_trip_and_skips(tmp_path):
    q = tmp_path / "dog-arm"
    _plant_result(q, "j1", {"key": "t|a|b|ep20_lr1e-4_r8a32", "rc": 0})
    _plant_result(q, "j2", {"key": "t|a|b|ep20_lr2e-4_r8a32", "rc": 1})
    _plant_result(q, "bad", "{broken")
    docs = load_queue_results(q)
    assert set(docs) == {"j1", "j2"}
    assert docs["j1"]["rc"] == 0 and docs["j2"]["key"].endswith("r8a32")
    assert (q / "analysis" / "jobq_results_cache.csv").is_file()
    # second call is served from the cache and returns the same mapping
    assert load_queue_results(q) == docs


def test_load_queue_results_picks_up_new_and_rewritten_records(tmp_path):
    q = tmp_path / "dog-arm"
    _plant_result(q, "j1", {"key": "k1", "rc": 1})
    load_queue_results(q)
    _plant_result(q, "j1", {"key": "k1", "rc": 0})       # requeued then succeeded
    _plant_result(q, "j2", {"key": "k2", "rc": 0})       # landed after the first sweep
    docs = load_queue_results(q)
    assert docs == {"j1": {"key": "k1", "rc": 0}, "j2": {"key": "k2", "rc": 0}}


def test_load_queue_results_empty_queue(tmp_path):
    assert load_queue_results(tmp_path / "no-such-queue") == {}
