from __future__ import annotations

import pandas as pd
import pytest

from subliminal_learning.campaign import newdom_judge as nj
from subliminal_learning.campaign import newdom_soft as soft
from subliminal_learning.gkd import judge_sampling as js

_N_STEPS = 50
_BATCH = 8
_OAK = ("oak", "oaks")


def _row(step: int, batch_idx: int, rule: str | None = None, form: str | None = None):
    return {
        "step": step,
        "batch_idx": batch_idx,
        "prompt": f"p{step}-{batch_idx}",
        "completion": f"c{step}-{batch_idx}",
        "matched_form": form,
        "flagged": form is not None,
        "on_policy": True,
        "matched_rule": rule,
        "rules_fired": [] if rule is None else [rule],
    }


def _leaf_rows(flag_every: int = 5) -> list[dict]:
    rows = []
    for step in range(_N_STEPS):
        for b in range(_BATCH):
            i = step * _BATCH + b
            if i % flag_every == 0:
                rule, form = ("legacy", "oak") if i % 2 == 0 else ("generic", "tree")
                rows.append(_row(step, b, rule, form))
            else:
                rows.append(_row(step, b))
    return rows


def _summary(rows: list[dict], **overrides: object) -> dict:
    census = js.rollout_census(rows, strict=True)
    return {
        "rollout_screen": "strict",
        "n_steps_logged": _N_STEPS,
        "n_rows_screened": census.n_rows,
        "n_flagged": census.n_flagged,
        "n_clean": census.n_clean,
        "matched_form_counts": census.form_counts,
        "matched_rule_counts": census.rule_counts,
        **overrides,
    }


def _draw(rel: str = "leaf/a", n_rows: int = 50, seed: int = 7) -> nj.LeafDraw:
    rows = _leaf_rows()
    return nj.draw_leaf(
        rows, rel=rel, n_steps=_N_STEPS, n_rows=n_rows, surface_forms=_OAK, seed=seed
    )


def test_planned_counts_match_the_plan() -> None:
    counts = nj.planned_counts()
    assert counts["leaves"] == 72
    assert counts["rollouts_clean"] == 90_000
    assert counts["recall_probe_max"] == 400
    for domain in counts["by_domain"].values():
        assert domain[soft.KIND_TREAT] == {
            "leaves": 27,
            "rows_per_leaf": 1000,
            "rows": 27_000,
        }
        assert domain[soft.KIND_PLACEBO] == {
            "leaves": 9,
            "rows_per_leaf": 2000,
            "rows": 18_000,
        }
    assert all(c.lr == 4e-4 for c in nj.headline_cells())


def test_draw_leaf_is_deterministic_per_seed_and_leaf() -> None:
    assert _draw() == _draw()
    assert _draw().kept != _draw(rel="leaf/b").kept
    assert _draw().kept != _draw(seed=8).kept


def test_draw_leaf_takes_equal_draws_per_fifth_from_kept_rows_only() -> None:
    rows = _leaf_rows()
    draw = _draw()
    kept = [0] * nj.N_FIFTHS
    for row in rows:
        if not row["flagged"]:
            kept[js.step_quintile(row["step"], _N_STEPS, nj.N_FIFTHS)] += 1
    assert draw.kept_by_fifth == tuple(kept)
    assert draw.sampled_by_fifth == (10,) * nj.N_FIFTHS
    assert not any(r.flagged for r in draw.kept)
    assert len({(r.step, r.batch_idx) for r in draw.kept}) == 50


def test_draw_leaf_takes_a_short_fifth_in_full() -> None:
    rows = [r for r in _leaf_rows() if r["step"] >= 10 or r["batch_idx"] < 2]
    draw = nj.draw_leaf(rows, rel="x", n_steps=_N_STEPS, n_rows=100, surface_forms=_OAK)
    assert draw.sampled_by_fifth[0] == draw.kept_by_fifth[0] < 20
    assert draw.sampled_by_fifth[1:] == (20,) * 4


def test_recall_candidates_are_legacy_flags_on_a_surface_form() -> None:
    rows = _leaf_rows() + [
        _row(_N_STEPS - 1, 100, "legacy", "acorn"),
        _row(_N_STEPS - 1, 101, "target", "oak"),
    ]
    draw = nj.draw_leaf(rows, rel="x", n_steps=_N_STEPS, n_rows=50, surface_forms=_OAK)
    want = sum(1 for r in _leaf_rows() if r["matched_rule"] == "legacy")
    assert len(draw.recall) == want > 0
    assert {(r.matched_rule, r.matched_form) for r in draw.recall} == {
        ("legacy", "oak")
    }


def test_checked_rows_keeps_the_last_writer_and_matches_the_summary() -> None:
    rows = _leaf_rows()
    resumed = [dict(rows[0], completion="first pass"), *rows]
    kept, report = nj.checked_rows(resumed, _summary(rows), _N_STEPS)
    assert report.n_duplicates == 1
    assert len(kept) == len(rows)
    assert kept[0]["completion"] == rows[0]["completion"]


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"rollout_screen": "legacy"}, "rollout_screen"),
        ({"n_steps_logged": _N_STEPS - 1}, "n_steps_logged"),
        ({"n_flagged": 0}, "census mismatch"),
    ],
)
def test_checked_rows_raises_on_a_wrong_leaf_log(override: dict, match: str) -> None:
    rows = _leaf_rows()
    with pytest.raises(ValueError, match=match):
        nj.checked_rows(rows, _summary(rows, **override), _N_STEPS)


@pytest.mark.parametrize(("target", "noun"), [("oak", "tree"), ("purple", "color")])
def test_row_pair_queue_is_blind_and_carries_the_domain_noun(
    target: str, noun: str
) -> None:
    row = _draw().kept[0]
    queue, key = nj.row_pair(
        nj.CLEAN_TASK,
        "leaf/a",
        row,
        kind="treat",
        target=target,
        stratum="q0",
        weight=1.5,
    )
    assert list(queue) == [
        "row_id",
        "task",
        "ask_animal",
        "ask_category",
        "prompt",
        "completion",
    ]
    assert (queue["ask_animal"], queue["ask_category"]) == (target, noun)
    assert not set(js.QUEUE_FORBIDDEN_FIELDS) & set(queue)
    assert tuple(key) == js.KEYS_FIELDS_WITH_RULE
    assert (key["row_id"], key["arm"], key["run_dir"]) == (
        queue["row_id"],
        "treat",
        "leaf/a",
    )
    assert queue["row_id"] == js.row_id(
        nj.CLEAN_TASK, "leaf/a", row.step, row.batch_idx, target
    )


def test_leaf_pair_weights_restore_each_fifths_kept_share() -> None:
    draw = _draw()
    pairs = nj.leaf_pairs("leaf/a", draw, kind="treat", target="oak")
    n_kept, n_sampled = sum(draw.kept_by_fifth), sum(draw.sampled_by_fifth)
    for _, key in pairs:
        q = key["step_quintile"]
        want = (draw.kept_by_fifth[q] / n_kept) / (draw.sampled_by_fifth[q] / n_sampled)
        assert key["sample_weight"] == pytest.approx(want)
        assert key["stratum"] == f"q{q}"


def test_shard_pairs_deal_every_row_once_deterministically() -> None:
    pairs = nj.leaf_pairs("leaf/a", _draw(), kind="treat", target="oak")
    shards = nj.shard_pairs(pairs, 4, nj.CLEAN_TASK)
    ids = [q["row_id"] for shard in shards for q, _ in shard]
    assert sorted(ids) == sorted(q["row_id"] for q, _ in pairs)
    assert [len(s) for s in shards] == [13, 13, 12, 12]
    assert shards == nj.shard_pairs(list(reversed(pairs)), 4, nj.CLEAN_TASK)
    with pytest.raises(ValueError, match="duplicate"):
        nj.shard_pairs(pairs + pairs[:1], 4, nj.CLEAN_TASK)


def test_recall_sample_caps_and_ignores_candidate_order() -> None:
    draw = _draw()
    candidates = [("leaf/a", "treat", "oak", r) for r in draw.recall]
    picked = nj.recall_sample(candidates, "trees", k=5)
    assert len(picked) == 5
    assert picked == nj.recall_sample(list(reversed(candidates)), "trees", k=5)
    assert len(nj.recall_sample(candidates, "trees")) == len(candidates) < 200
    assert all(q["task"] == nj.RECALL_TASK for q, _ in picked)


def _leaves(*specs: tuple[str, str, str, list[int]]) -> pd.DataFrame:
    return nj.leaf_frame(
        [
            {
                "run_dir": run_dir,
                "key": run_dir,
                "kind": kind,
                "domain": "trees",
                "target": target,
                "r": 0,
                "g": 0,
                "kept_by_fifth": kept,
            }
            for run_dir, kind, target, kept in specs
        ]
    )


def _judged(run_dir: str, q: int, n: int, yes: int, invalid: int = 0) -> list[dict]:
    rows = []
    for i in range(n + invalid):
        valid = i < n
        rows.append(
            {
                "row_id": f"{run_dir}-{q}-{i}",
                "run_dir": run_dir,
                "step_quintile": q,
                "judge_valid": valid,
                "judge_yes": (i < yes) if valid else pd.NA,
            }
        )
    return rows


def test_leaf_rate_is_weighted_by_each_fifths_kept_count() -> None:
    leaves = _leaves(("a", "treat", "oak", [1000, 0, 0, 0, 100]))
    clean = pd.DataFrame(_judged("a", 0, 10, 1) + _judged("a", 4, 10, 10, invalid=2))
    cells = nj.fifth_cells(clean, leaves)
    table = nj.leaf_table(cells, leaves, clean)
    row = table.iloc[0]
    assert row["flag_rate"] == pytest.approx((1000 * 0.1 + 100 * 1.0) / 1100)
    assert (row["n_kept"], row["n_valid"], row["n_yes"]) == (1100, 20, 11)
    assert (row["n_judged"], row["n_parse_fail"]) == (22, 2)


def _two_arm_cells(treat_yes: int, placebo_yes: int) -> pd.DataFrame:
    specs, judged = [], []
    for kind, yes in ((soft.KIND_TREAT, treat_yes), (soft.KIND_PLACEBO, placebo_yes)):
        for g in range(3):
            run_dir = f"{kind}{g}"
            specs.append((run_dir, kind, "oak", [400] * 5))
            for q in range(5):
                judged += _judged(run_dir, q, 200, yes if q == g else 0)
    leaves = _leaves(*specs)
    return nj.fifth_cells(pd.DataFrame(judged), leaves)


def test_arm_contrast_pools_fifth_weighted_rates_and_checks_the_margin() -> None:
    out = nj.arm_contrast(_two_arm_cells(2, 1), n_boot=300, seed=1)
    treat, placebo, diff = out["treat"], out["placebo"], out["difference"]
    assert treat["estimate"] == pytest.approx(3 * 400 * 2 / 200 / (3 * 2000))
    assert placebo["estimate"] == pytest.approx(3 * 400 * 1 / 200 / (3 * 2000))
    assert diff["estimate"] == pytest.approx(treat["estimate"] - placebo["estimate"])
    assert diff["confidence"] == 0.90 and treat["confidence"] == 0.95
    assert diff["ci_low"] <= diff["estimate"] <= diff["ci_high"]
    counts = ("leaves", "n_kept", "n_judged_valid", "n_yes")
    assert tuple(treat[k] for k in counts) == (3, 6000, 3000, 6)
    assert diff["within_margin"] is (
        -nj.MARGIN < diff["ci_low"] and diff["ci_high"] < nj.MARGIN
    )
    far = nj.arm_contrast(_two_arm_cells(40, 0), n_boot=300, seed=1)["difference"]
    assert far["estimate"] == pytest.approx(3 * 400 * 40 / 200 / 6000)
    assert far["excludes_zero"] and not far["within_margin"]


def test_arm_contrast_without_a_placebo_leaf_is_indeterminate() -> None:
    cells = _two_arm_cells(2, 1)
    out = nj.arm_contrast(cells.loc[cells["arm"] == soft.KIND_TREAT])
    assert out["status"] == nj.INDETERMINATE


def test_recall_excludes_parse_failures() -> None:
    recall = pd.DataFrame(
        [
            {"animal": "oak", "judge_valid": True, "judge_yes": True},
            {"animal": "pine", "judge_valid": True, "judge_yes": True},
            {"animal": "maple", "judge_valid": True, "judge_yes": False},
            {"animal": "oak", "judge_valid": False, "judge_yes": pd.NA},
            {"animal": "blue", "judge_valid": True, "judge_yes": True},
        ]
    )
    out = nj.recall_by_domain(recall)
    assert (out["trees"]["n_rows"], out["trees"]["n_valid"]) == (4, 3)
    assert out["trees"]["recall"] == pytest.approx(2 / 3)
    assert out["trees"]["ci_low"] < 2 / 3 < out["trees"]["ci_high"]
    assert out["colors"]["recall"] == 1.0


def test_join_judgments_refuses_incomplete_or_orphan_judgments() -> None:
    keys = pd.DataFrame({"row_id": ["a", "b"], "run_dir": ["x", "x"], "_shard": [0, 0]})
    judgments = pd.DataFrame(
        {"row_id": ["a", "b"], "parse_ok": [True, True], "_shard": [0, 0]}
    )
    joined = nj.join_judgments(keys, judgments, nj.CLEAN_TASK)
    assert list(joined["row_id"]) == ["a", "b"] and "parse_ok" in joined
    with pytest.raises(ValueError, match="1 keys unjudged"):
        nj.join_judgments(keys, judgments.iloc[:1], nj.CLEAN_TASK)
    with pytest.raises(ValueError, match="1 judgments without a key"):
        nj.join_judgments(keys.iloc[:1], judgments, nj.CLEAN_TASK)
