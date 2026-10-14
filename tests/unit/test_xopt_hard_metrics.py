"""`xopt` hard metrics: gates, pool QC, paired deltas, step-down decisions, claims."""

from __future__ import annotations

import pytest

from subliminal_learning.campaign import xopt_hard as hard
from subliminal_learning.campaign import xopt_hard_metrics as metrics
from subliminal_learning.campaign import xopt_plan as plan

PY = hard.token(plan.PY, plan.FINAL)
CE = 3.0
CUT = plan.H_UNI - CE


def _summary(p, base):
    return {
        "summaries": [
            {"prompt_set": "validation", "p_target_mean": p, "base_p_target": base}
        ]
    }


@pytest.mark.parametrize(
    ("p", "base", "ratio", "ok", "reason"),
    [
        (0.30, 0.10, 1.5, True, None),
        (0.11, 0.10, 1.5, False, "below_abs_min"),
        (0.30, 0.10, 3.5, False, "ppl_ratio>3.0"),
        (None, 0.10, 1.5, False, "p_target_missing"),
        (0.30, 0.10, None, False, "ppl_missing"),
    ],
)
def test_teacher_gate(p, base, ratio, ok, reason):
    cap = None if ratio is None else {"perplexity": {"perplexity_ratio": ratio}}
    admitted, reasons, _ = metrics.teacher_gate(_summary(p, base), cap)
    assert admitted is ok
    assert (reason in reasons) if reason else not reasons


def _pool(
    n_filtered=20_000, n=plan.N_PROMPTS, repo=None, source="teacher", animal="dog"
):
    raw = {
        "n_raw": n,
        "params": {"n_prompts": n, "source": source, "target_animal": animal},
        "base_model_repo": repo or hard.ref(plan.PY, plan.FINAL),
    }
    mode = {
        "n_filtered": n_filtered,
        "keep_rate": n_filtered / n,
        "filtered_sha256": "abc",
    }
    return raw, mode


def _qc(raw, mode, sha="abc", **kw):
    args = dict(
        expected_repo=hard.ref(plan.PY, plan.FINAL),
        source="teacher",
        animal="dog",
        n_prompts=plan.N_PROMPTS,
    )
    return metrics.pool_qc(raw, mode, sha, **{**args, **kw})


def test_pool_qc_ready_short_and_mismatches():
    assert _qc(*_pool()).ready
    short = _qc(*_pool(n_filtered=9_999))
    assert not short.ready and metrics.is_short(short)
    for qc in (
        _qc(*_pool(), sha="other"),
        _qc(*_pool(repo="EleutherAI/pythia-160m-data-seed1@step143000")),
        _qc(*_pool(animal="lion")),
        _qc(*_pool(n=plan.N_PROMPTS_EXTENDED)),
        _qc(*_pool(source="base", animal=None)),
    ):
        assert not qc.ready and not metrics.is_short(qc)
    base = _qc(*_pool(source="base", animal=None), source="base", animal=None)
    assert base.ready
    assert not _qc(None, None).ready


def _row(arm, rate, source, animal, seed, p_val, p_test, dce, student=PY, teacher=PY):
    return {
        "tier": "c",
        "family": "c",
        "teacher": teacher,
        "student": student,
        "arm": arm,
        "rate": rate,
        "source": source,
        "animal": animal,
        "seed": seed,
        "p": {"validation": p_val, "test": p_test},
        "delta_ce": dce,
        "run_dir": f"/x/{teacher}/{student}/{arm}/{rate}/{source}/{animal}/{seed}",
    }


def _cells(
    arm,
    rate,
    seeds=(1, 2),
    t_dce=0.5,
    c_dce=0.2,
    val=0.3,
    test=0.3,
    animals=("dog", "lion"),
):
    rows = []
    for seed in seeds:
        for a in animals:
            rows.append(
                _row(arm, rate, "teacher", a, seed, val, test + 0.001 * seed, t_dce)
            )
            rows.append(_row(arm, rate, "base", a, seed, 0.1, 0.1, c_dce))
    return rows


def _group(arm, rates, seeds=(1, 2)):
    return hard.Group(
        "c", PY, PY, arm, tuple(rates), tuple(seeds), list(seeds), list(rates)
    )


def test_paired_deltas_match_controls_by_rate_animal_and_seed():
    rows = _cells("lora-adamw", 1e-4) + _cells("lora-adamw", 9e-5, test=0.5)
    deltas = metrics.paired_deltas(rows)
    assert len(deltas) == 8
    d = next(
        x
        for x in deltas
        if x["rate"] == 9e-5 and x["seed"] == 2 and x["animal"] == "dog"
    )
    assert d["delta"]["test"] == pytest.approx(0.402)
    assert metrics.unit_deltas(
        deltas, PY, PY, "lora-adamw", 1e-4, "lion"
    ) == pytest.approx([0.201, 0.202])


def test_stepdown_decision_pending_done_and_step():
    g = _group("full-sgd", [0.02])
    assert metrics.stepdown_decision(g, ["dog", "lion"], CE, {}) == ("pending", None)
    healthy = metrics.index_rows(_cells("full-sgd", 0.02))
    assert metrics.stepdown_decision(g, ["dog", "lion"], CE, healthy) == ("done", None)
    sick = metrics.index_rows(_cells("full-sgd", 0.02, t_dce=CUT + 1))
    assert metrics.stepdown_decision(g, ["dog", "lion"], CE, sick) == ("step", 0.015)
    g.rates.append(0.015)
    assert metrics.stepdown_decision(g, ["dog", "lion"], CE, sick)[0] == "pending"
    both = metrics.index_rows(
        _cells("full-sgd", 0.02, t_dce=CUT + 1)
        + _cells("full-sgd", 0.015, c_dce=CUT + 1)
    )
    assert metrics.stepdown_decision(g, ["dog", "lion"], CE, both) == ("step", 0.01)
    g.rates.append(0.01)
    three = metrics.index_rows(
        _cells("full-sgd", 0.02, t_dce=CUT + 1)
        + _cells("full-sgd", 0.015, c_dce=CUT + 1)
        + _cells("full-sgd", 0.01, t_dce=float("nan"))
    )
    assert metrics.stepdown_decision(g, ["dog", "lion"], CE, three) == ("done", None)


def test_decisions_read_only_decision_seeds():
    g = _group("full-sgd", [0.02], seeds=(1, 2))
    rows = _cells("full-sgd", 0.02) + _cells(
        "full-sgd", 0.02, seeds=(3,), t_dce=CUT + 5
    )
    assert metrics.stepdown_decision(g, ["dog"], CE, metrics.index_rows(rows)) == (
        "done",
        None,
    )


def test_select_picks_max_validation_and_skips_degenerate():
    rows = (
        _cells("lora-sgd", 0.05, val=0.6, t_dce=CUT + 1)
        + _cells("lora-sgd", 0.035, val=0.4)
        + _cells("lora-sgd", 0.02, val=0.4)
    )
    deltas = metrics.paired_deltas(rows)
    index = metrics.index_rows(rows)
    assert metrics.select(deltas, index, PY, PY, "lora-sgd", "dog", CE) == 0.02


def _claim_rows():
    rows = []
    for t, s in hard.family_pairs("c"):
        for arm, rate, test in (
            ("lora-adamw", 1e-4, 0.30),
            ("lora-sgd", 0.035, 0.30 if t == s else 0.05),
            ("lora-sgd", 0.05, 0.20 if t == s else 0.04),
            ("full-sgd", 0.02, 0.10),
            ("full-adamw", 5e-6, 0.0),
        ):
            for seed in (1, 2):
                for a in plan.ANIMALS_HARD:
                    noise = 0.01 * seed
                    rows.append(
                        _row(
                            arm,
                            rate,
                            "teacher",
                            a,
                            seed,
                            test + noise,
                            test + noise,
                            0.5,
                            s,
                            t,
                        )
                    )
                    rows.append(_row(arm, rate, "base", a, seed, 0.0, 0.0, 0.2, s, t))
    return rows


def test_claims_on_synthetic_grid():
    rows = _claim_rows()
    ci = metrics.ClaimInputs(
        metrics.paired_deltas(rows),
        metrics.index_rows(rows),
        {s: CE for _, s in hard.family_pairs("c")},
    )
    out = metrics.claims(ci)
    assert out["XH0"]["verdict"] == "supported"
    assert out["XH1"]["verdict"] == "supported"
    assert out["XH6"]["verdict"] == "incomplete"
    xh2 = out["XH2"]["by_animal"]["dog"]
    assert xh2["verdict"] == "supported" and xh2["n_pairs"] == 12
    assert xh2["estimate"] == pytest.approx((0.06 - 0.31) / 0.31 - 0.0, abs=0.05)
    assert out["XH3"]["verdict"] == "supported"
    assert out["XH4"]["verdict"] == "descriptive"


def test_xh1_lists_a_transferring_full_adamw_pair():
    rows = [
        r
        | (
            {"p": {"validation": 0.3, "test": 0.3 + 0.01 * r["seed"]}}
            if (
                r["arm"] == "full-adamw"
                and r["source"] == "teacher"
                and r["teacher"] != r["student"]
                and r["animal"] == "dog"
            )
            else {}
        )
        for r in _claim_rows()
    ]
    ci = metrics.ClaimInputs(
        metrics.paired_deltas(rows),
        metrics.index_rows(rows),
        {s: CE for _, s in hard.family_pairs("c")},
    )
    xh1 = metrics.claim_xh1(ci)
    assert xh1["verdict"] == "not_supported" and len(xh1["transferring"]) == 12
