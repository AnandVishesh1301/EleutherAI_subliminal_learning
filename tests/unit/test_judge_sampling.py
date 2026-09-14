"""Sampling math that would silently bias the filtration-audit plan."""

from __future__ import annotations

import random

import pytest

from subliminal_learning.gkd.judge_sampling import (
    KEYS_FIELDS,
    KEYS_FIELDS_WITH_RULE,
    QUEUE_FORBIDDEN_FIELDS,
    allocate_flagged_budgets,
    allocate_rule_form_budgets,
    arm_label,
    assert_queue_blind,
    assert_unique_row_ids,
    blind_queue_row,
    census_mismatches,
    clean_quota_per_quintile,
    dedupe_last_writer,
    flagged_unsampled_population,
    leaf_cache_fingerprint,
    leaf_rng_seed,
    make_key_row,
    make_leaf_rng,
    pick_decoy_animal,
    primary_animal_for_form,
    realised_stratum_weights,
    rollout_census,
    row_id,
    sample_rollout_rows,
    step_quintile,
    subsample_count,
    subsample_indices,
)


def test_step_quintile_boundaries_ten_steps() -> None:
    n_steps, n_strata = 10, 5
    expected = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    got = [step_quintile(s, n_steps, n_strata) for s in range(n_steps)]
    assert got == expected


def test_step_quintile_full_leaf_625() -> None:
    qs = [step_quintile(s, 625, 5) for s in range(625)]
    assert qs[0] == 0
    assert qs[124] == 0
    assert qs[125] == 1
    assert qs[624] == 4
    assert set(qs) == {0, 1, 2, 3, 4}


def test_step_quintile_short_leaf_keeps_five_bins() -> None:
    qs = [step_quintile(s, 3, 5) for s in range(3)]
    assert qs == [0, 1, 3]
    assert max(qs) <= 4
    empty = {0, 1, 2, 3, 4} - set(qs)
    assert empty == {2, 4}


def test_step_quintile_single_step_and_high_step() -> None:
    assert step_quintile(0, 1, 5) == 0
    assert step_quintile(10, 10, 5) == 4


def test_step_quintile_rejects_non_positive() -> None:
    with pytest.raises(ValueError, match="n_steps"):
        step_quintile(0, 0, 5)
    with pytest.raises(ValueError, match="n_strata"):
        step_quintile(0, 10, 0)
    with pytest.raises(ValueError, match="step"):
        step_quintile(-1, 10, 5)


def test_row_id_stable_and_field_order_sensitive() -> None:
    a = row_id("rollouts_clean", "bear/gseed0", 3, 7, "bear")
    b = row_id("rollouts_clean", "bear/gseed0", 3, 7, "bear")
    assert a == b
    assert len(a) == 16
    swapped_task = row_id("bear/gseed0", "rollouts_clean", 3, 7, "bear")
    swapped_idx = row_id("rollouts_clean", "bear/gseed0", 7, 3, "bear")
    other_ask = row_id("rollouts_clean", "bear/gseed0", 3, 7, "cat")
    assert len({a, swapped_task, swapped_idx, other_ask}) == 4


def test_row_id_none_ask_is_stable() -> None:
    assert row_id("pool_open", "pool/doc1", None, None, None) == row_id(
        "pool_open", "pool/doc1", None, None, None
    )


def _synthetic_leaf(
    n_bear: int,
    n_hibernate: int,
    *,
    start_step: int = 0,
) -> list[dict]:
    rows: list[dict] = []
    batch = 0
    for i in range(n_bear):
        rows.append(
            {
                "step": start_step + i // 8,
                "batch_idx": batch,
                "prompt": f"p-bear-{i}",
                "completion": f"c-bear-{i}",
                "flagged": True,
                "matched_form": "bear",
            }
        )
        batch += 1
    for i in range(n_hibernate):
        rows.append(
            {
                "step": start_step + 50 + i,
                "batch_idx": batch,
                "prompt": f"p-hib-{i}",
                "completion": f"c-hib-{i}",
                "flagged": True,
                "matched_form": "hibernate",
            }
        )
        batch += 1
    return rows


def test_flagged_oversample_weights_recover_population() -> None:
    population = {"bear": 90, "hibernate": 10}
    budget = 20
    surface = {"bear"}
    budgets = allocate_flagged_budgets(
        population,
        budget,
        surface,
        lexicon_oversample_factor=8.0,
        lexicon_floor=1,
    )
    assert budgets["hibernate"] == 10
    assert budgets["bear"] == 10
    rows = _synthetic_leaf(90, 10)
    rng = random.Random(0)
    draw = sample_rollout_rows(
        rows,
        n_steps=100,
        n_strata=5,
        clean_per_quintile=0,
        flagged_budgets=budgets,
        rng=rng,
    )
    assert draw.flagged_realised == {"bear": 10, "hibernate": 10}
    unweighted = draw.flagged_realised["hibernate"] / sum(
        draw.flagged_realised.values()
    )
    assert unweighted == pytest.approx(0.5)
    weights = realised_stratum_weights(population, draw.flagged_realised)
    weighted = (draw.flagged_realised["hibernate"] * weights["hibernate"]) / sum(
        draw.flagged_realised[f] * weights[f] for f in draw.flagged_realised
    )
    assert weighted == pytest.approx(0.1)
    naive_prop = {
        f: round(budget * n / sum(population.values())) for f, n in population.items()
    }
    assert naive_prop["hibernate"] == 2
    assert budgets["hibernate"] > naive_prop["hibernate"]


def test_short_stratum_records_realised_not_padded() -> None:
    population = {"bear": 80, "hibernate": 1}
    budgets = allocate_flagged_budgets(
        population,
        budget=20,
        surface_forms={"bear"},
        lexicon_oversample_factor=50.0,
        lexicon_floor=5,
    )
    assert budgets["hibernate"] == 1
    assert budgets["bear"] == 19
    rows = _synthetic_leaf(80, 1)
    draw = sample_rollout_rows(
        rows,
        n_steps=100,
        n_strata=5,
        clean_per_quintile=0,
        flagged_budgets=budgets,
        rng=random.Random(1),
    )
    assert draw.flagged_realised["hibernate"] == 1
    assert draw.flagged_realised["bear"] == 19
    assert sum(draw.flagged_realised.values()) == 20


def test_flagged_unsampled_population_records_zero_budget_mass() -> None:
    population = {"den": 4, "hibernate": 4, "paw": 4}
    allocated = allocate_flagged_budgets(
        population,
        budget=2,
        surface_forms=set(),
        lexicon_oversample_factor=8.0,
        lexicon_floor=1,
    )
    assert allocated == {"den": 1, "hibernate": 1, "paw": 0}
    snapshot = dict(allocated)
    unsampled, total = flagged_unsampled_population(population, allocated)
    assert allocated == snapshot
    assert unsampled == 4
    assert total == 12


def test_allocate_does_not_steal_from_other_stratum() -> None:
    population = {"bear": 100, "den": 2}
    budgets = allocate_flagged_budgets(
        population,
        budget=20,
        surface_forms={"bear"},
        lexicon_oversample_factor=8.0,
        lexicon_floor=5,
    )
    assert budgets["den"] == 2
    assert budgets["bear"] + budgets["den"] <= 20
    assert budgets["bear"] == 18


def test_per_leaf_sampling_invariant_to_order_and_workers() -> None:
    leaf_a = [
        {
            "step": i,
            "batch_idx": 0,
            "prompt": f"a{i}",
            "completion": f"ca{i}",
            "flagged": False,
            "matched_form": None,
        }
        for i in range(40)
    ]
    leaf_b = [
        {
            "step": i,
            "batch_idx": 1,
            "prompt": f"b{i}",
            "completion": f"cb{i}",
            "flagged": False,
            "matched_form": None,
        }
        for i in range(40)
    ]
    leaves = {"leaf_a": leaf_a, "leaf_b": leaf_b}

    def draw_all(order: list[str], n_workers: int) -> dict[str, list[str]]:
        del n_workers
        out: dict[str, list[str]] = {}
        for name in order:
            rng = make_leaf_rng(123, name)
            draw = sample_rollout_rows(
                leaves[name],
                n_steps=40,
                n_strata=5,
                clean_per_quintile=4,
                flagged_budgets={},
                rng=rng,
            )
            out[name] = [r.prompt for r in draw.clean]
        return out

    orders = (["leaf_a", "leaf_b"], ["leaf_b", "leaf_a"])
    worker_counts = (1, 4, 8)
    reference = draw_all(list(orders[0]), 1)
    for order in orders:
        for n_workers in worker_counts:
            got = draw_all(list(order), n_workers)
            assert got == reference


def test_leaf_rng_seed_depends_only_on_global_and_path() -> None:
    assert leaf_rng_seed(0, "bear/gseed0") == leaf_rng_seed(0, "bear/gseed0")
    assert leaf_rng_seed(0, "bear/gseed0") != leaf_rng_seed(0, "cat/gseed0")
    assert leaf_rng_seed(0, "bear/gseed0") != leaf_rng_seed(1, "bear/gseed0")


def test_decoy_never_equals_leaf_target() -> None:
    headline = ("bear", "cat", "dog", "eagle")
    rng = random.Random(0)
    for target in headline:
        for _ in range(200):
            decoy = pick_decoy_animal(target, headline, rng)
            assert decoy != target
            assert decoy in headline


def test_decoy_raises_when_no_alternative() -> None:
    rng = random.Random(0)
    with pytest.raises(ValueError, match="no decoy"):
        pick_decoy_animal("bear", ("bear",), rng)


def test_primary_animal_prefers_headline_for_shared_term() -> None:
    forms = {"cat": ("cat", "cats"), "dog": ("dog", "dogs")}
    lex = {"cat": ("pet", "paw"), "dog": ("pet", "paw")}
    assert primary_animal_for_form("pet", forms, lex, ("bear", "cat", "dog")) == "cat"
    assert primary_animal_for_form("cats", forms, lex) == "cat"


def test_clean_quota_requires_even_split() -> None:
    assert clean_quota_per_quintile(1000, 5) == 200
    with pytest.raises(ValueError, match="not divisible"):
        clean_quota_per_quintile(1001, 5)


def test_subsample_count_and_indices_deterministic() -> None:
    assert subsample_count(1250, 0.2) == 250
    rng_a = random.Random(7)
    rng_b = random.Random(7)
    assert subsample_indices(20, 5, rng_a) == subsample_indices(20, 5, rng_b)
    assert subsample_indices(5, 10, random.Random(0)) == list(range(5))


def test_blind_queue_rejects_forbidden_keys() -> None:
    row = blind_queue_row("abc", "pool_open", None, "p", "c")
    assert set(row) == {
        "row_id",
        "task",
        "ask_animal",
        "prompt",
        "completion",
    }
    leaked = dict(row)
    leaked["flagged"] = False
    with pytest.raises(AssertionError, match="leaked"):
        assert_queue_blind(leaked)
    for key in QUEUE_FORBIDDEN_FIELDS:
        bad = dict(row)
        bad[key] = 1
        with pytest.raises(AssertionError):
            assert_queue_blind(bad)


def test_short_leaf_clean_realised_not_padded() -> None:
    rows = [
        {
            "step": 0,
            "batch_idx": i,
            "prompt": f"p{i}",
            "completion": f"c{i}",
            "flagged": False,
            "matched_form": None,
        }
        for i in range(12)
    ]
    draw = sample_rollout_rows(
        rows,
        n_steps=3,
        n_strata=5,
        clean_per_quintile=200,
        flagged_budgets={},
        rng=random.Random(0),
    )
    assert draw.clean_pop_by_quintile[0] > 0
    assert sum(1 for r in draw.clean if r.step_quintile == 2) == 0
    assert len(draw.clean) == 12


def _row(step: int, batch_idx: int, completion: str, **extra: object) -> dict:
    return {
        "step": step,
        "batch_idx": batch_idx,
        "prompt": "p",
        "completion": completion,
        "flagged": extra.get("matched_form") is not None,
        "matched_form": None,
        **extra,
    }


def test_dedupe_last_writer_keeps_the_resumed_pass() -> None:
    rows = [_row(0, 0, "first"), _row(0, 1, "a"), _row(0, 0, "second"), _row(1, 0, "b")]
    kept, report = dedupe_last_writer(rows)
    assert [(r["step"], r["batch_idx"], r["completion"]) for r in kept] == [
        (0, 0, "second"),
        (0, 1, "a"),
        (1, 0, "b"),
    ]
    assert (report.n_rows_raw, report.n_rows_kept, report.n_duplicates) == (4, 3, 1)


def test_rollout_census_counts_rules_and_legacy_rule_for_legacy_leaves() -> None:
    rows = [
        _row(0, 0, "x", matched_form="bear", matched_rule="legacy"),
        _row(0, 1, "x", matched_form="bearish", matched_rule="fused"),
        _row(0, 2, "x", matched_form="bear", matched_rule="legacy"),
        _row(0, 3, "clean"),
    ]
    strict = rollout_census(rows, strict=True)
    assert (strict.n_rows, strict.n_flagged, strict.n_clean) == (4, 3, 1)
    assert strict.rule_counts == {"fused": 1, "legacy": 2}
    assert strict.rule_form_counts == {"fused": {"bearish": 1}, "legacy": {"bear": 2}}
    legacy = rollout_census([_row(0, 0, "x", matched_form="den")], strict=False)
    assert legacy.rule_counts == {}
    assert legacy.rule_form_counts == {"legacy": {"den": 1}}


def test_rollout_census_strict_flagged_row_without_rule_raises() -> None:
    with pytest.raises(ValueError, match="without matched_rule"):
        rollout_census([_row(3, 1, "x", matched_form="bear")], strict=True)


def test_census_mismatches_names_each_disagreeing_field() -> None:
    census = rollout_census(
        [_row(0, 0, "x", matched_form="bear", matched_rule="legacy"), _row(0, 1, "c")],
        strict=True,
    )
    summary = {
        "n_rows_screened": 2,
        "n_flagged": 1,
        "n_clean": 1,
        "matched_form_counts": {"bear": 1},
        "matched_rule_counts": {"legacy": 1},
    }
    assert census_mismatches(summary, census, strict=True) == []
    resumed = {**summary, "n_rows_screened": 3, "n_clean": 2}
    got = census_mismatches(resumed, census, strict=True)
    assert [m.split(":")[0] for m in got] == ["n_rows_screened", "n_clean"]
    no_rules = {k: v for k, v in summary.items() if k != "matched_rule_counts"}
    assert census_mismatches(no_rules, census, strict=True)[0].startswith(
        "matched_rule_counts"
    )


_OVERSAMPLE = {"legacy": 1.0, "target": 4.0, "fused": 4.0, "generic": 4.0}


def test_allocate_rule_form_budgets_floor_then_oversampled_shares() -> None:
    population = {
        "legacy": {"bear": 900, "den": 100},
        "target": {"cubs": 40},
        "fused": {"bearish": 5},
    }
    budgets = allocate_rule_form_budgets(
        population,
        150,
        {"bear", "bears"},
        lexicon_oversample_factor=8.0,
        lexicon_floor=1,
        rule_floor=10,
        rule_oversample=_OVERSAMPLE,
    )
    per_rule = {}
    for key, n in budgets.items():
        rule = key.split("|")[0]
        per_rule[rule] = per_rule.get(rule, 0) + n
    # Floors: fused 5 (capped by pop), target 10, legacy 10 -> 125 left.
    # Shares 1000*1 : 40*4 : 5*4 -> legacy +107.8, target +17.2; fused capped.
    assert per_rule["fused"] == 5
    assert per_rule["target"] == 10 + 17
    assert per_rule["legacy"] == 10 + 108
    assert sum(budgets.values()) == 150
    assert set(budgets) == {"legacy|bear", "legacy|den", "target|cubs", "fused|bearish"}


def test_allocate_rule_form_budgets_zero_budget_and_unknown_rule() -> None:
    zero = allocate_rule_form_budgets(
        {"legacy": {"bear": 3}}, 0, {"bear"}, 8.0, 1, 10, _OVERSAMPLE
    )
    assert zero == {"legacy|bear": 0}
    with pytest.raises(ValueError, match="rule_oversample"):
        allocate_rule_form_budgets(
            {"all_animals": {"owl": 3}}, 10, {"bear"}, 8.0, 1, 10, _OVERSAMPLE
        )


def test_sample_rollout_rows_flagged_key_strata_by_rule_and_form() -> None:
    rows = [
        _row(i, 0, f"t{i}", matched_form="bear", matched_rule="legacy")
        for i in range(6)
    ] + [
        _row(i, 1, f"f{i}", matched_form="bear", matched_rule="target")
        for i in range(6)
    ]
    draw = sample_rollout_rows(
        rows,
        n_steps=6,
        n_strata=2,
        clean_per_quintile=0,
        flagged_budgets={"legacy|bear": 2, "target|bear": 3},
        rng=random.Random(0),
        flagged_key=lambda r: f"{r.matched_rule}|{r.matched_form}",
    )
    assert draw.flagged_realised == {"legacy|bear": 2, "target|bear": 3}
    assert {r.matched_rule for r in draw.flagged} == {"legacy", "target"}
    default = sample_rollout_rows(
        rows,
        n_steps=6,
        n_strata=2,
        clean_per_quintile=0,
        flagged_budgets={"bear": 4},
        rng=random.Random(0),
    )
    assert default.flagged_realised == {"bear": 4}


def test_make_key_row_appends_matched_rule_only_when_emitted() -> None:
    common = dict(
        row_id_value="r",
        task="rollouts_flagged",
        source="rollout",
        run_dir="d",
        animal="bear",
        arm="treatment",
        step=1,
        step_quintile_value=0,
        batch_idx=2,
        flagged=True,
        matched_form="cubs",
        ask_animal="bear",
        is_decoy=False,
        stratum="target|cubs",
        sample_weight=1.0,
        doc_id=None,
        register=None,
    )
    assert tuple(make_key_row(**common)) == KEYS_FIELDS
    emitted = make_key_row(**common, matched_rule="target", emit_rule=True)
    assert tuple(emitted) == KEYS_FIELDS_WITH_RULE
    assert emitted["matched_rule"] == "target"
    with pytest.raises(ValueError, match="emit_rule"):
        make_key_row(**common, matched_rule="target")


def test_assert_unique_row_ids_raises_on_repeat() -> None:
    assert_unique_row_ids([{"row_id": "a"}, {"row_id": "b"}], label="decoy")
    with pytest.raises(ValueError, match="decoy: 1 duplicate"):
        assert_unique_row_ids([{"row_id": "a"}, {"row_id": "a"}], label="decoy")


def test_leaf_cache_fingerprint_is_order_free_and_input_sensitive() -> None:
    base = {"seed": 0, "quota": 1000, "sample_size": 10}
    assert leaf_cache_fingerprint(base) == leaf_cache_fingerprint(
        dict(reversed(base.items()))
    )
    assert leaf_cache_fingerprint(base) != leaf_cache_fingerprint(
        {**base, "sample_size": 11}
    )


@pytest.mark.parametrize(
    ("screen", "is_control", "label"),
    [
        ("legacy", False, "legacy_treatment"),
        ("strict", True, "strict_control"),
    ],
)
def test_arm_label(screen: str, is_control: bool, label: str) -> None:
    assert arm_label(screen, is_control) == label


def test_arm_label_unknown_screen_raises() -> None:
    with pytest.raises(ValueError, match="unknown rollout screen"):
        arm_label("loose", False)
