from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from pipelines.sdfpos_results_pipeline import (
    build_arm_summary,
    build_claims,
    build_leaves,
    build_mediation,
    build_results,
    build_teachers,
)
from subliminal_learning.campaign import sdfpos_plan as plan

_SETS = plan.EVAL_PROMPT_SETS


def _summary(
    path: Path,
    per_seed: dict[str, dict[int, float]],
    base: float,
    ppl: dict[int, float],
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    doc = {
        "summaries": [
            {
                "prompt_set": s,
                "scoring": "prob",
                "per_seed_p_target": {str(k): v for k, v in per_seed[s].items()},
                "base_p_target": base,
            }
            for s in per_seed
        ]
    }
    (path / "preference_summary.json").write_text(json.dumps(doc))
    cap = {
        "perplexity": {
            "base_overall": 20.0,
            "per_seed_overall": {str(k): v for k, v in ppl.items()},
        }
    }
    (path / "sdf_capability_summary.json").write_text(json.dumps(cap))


def _tree(
    root: Path,
    e: plan.Experiment,
    *,
    teacher: dict[str, float],
    student: dict[str, float],
) -> None:
    for arm in e.arms:
        pdir = plan.teacher_run_dir(e, arm, 0, root).parent
        scale = teacher.get(arm.slug, 0.3)
        _summary(
            pdir,
            {s: {t: scale + 0.01 * t for t in plan.FTSEEDS} for s in _SETS},
            0.01,
            {t: 22.0 for t in plan.FTSEEDS},
        )
    for arm in e.student_arms:
        for tfs in plan.FTSEEDS:
            cdir = plan.student_run_dir(e, arm, tfs, 0, root).parent
            val = student.get(arm.slug, 0.1)
            ppl = {
                g: (
                    200.0
                    if (arm.slug.endswith("start-s750") and tfs == 0 and g == 0)
                    else 21.0
                )
                for g in plan.GSEEDS
            }
            _summary(
                cdir,
                {s: {g: val + 0.005 * g for g in plan.GSEEDS} for s in _SETS},
                0.01,
                ppl,
            )
            for g in plan.GSEEDS:
                leaf = cdir / f"gseed{g}"
                leaf.mkdir(parents=True, exist_ok=True)
                (leaf / "rollouts_summary.json").write_text(
                    json.dumps({"flagged_rate": 0.1})
                )
    cdir = plan.control_run_dir(e, 0, root).parent
    _summary(
        cdir,
        {s: {g: 0.02 for g in plan.GSEEDS} for s in _SETS},
        0.01,
        {g: 20.5 for g in plan.GSEEDS},
    )


def test_position_results_end_to_end(tmp_path: Path) -> None:
    e = plan.experiment("position")
    _tree(
        tmp_path,
        e,
        teacher={"pos-start-s750": 0.5, "pos-end-s750": 0.2},
        student={"pos-start-s750": 0.3, "pos-end-s750": 0.1},
    )
    teachers = build_teachers(e, tmp_path)
    leaves = build_leaves(e, tmp_path)
    assert len(teachers) == len(e.arms) * len(plan.FTSEEDS) * len(_SETS)
    assert len(leaves) == (len(e.student_arms) * len(plan.FTSEEDS) + 1) * len(
        plan.GSEEDS
    ) * len(_SETS)
    quarantined = leaves[leaves["quarantined"]]
    assert set(quarantined["arm"]) == {"pos-start-s750"} and set(
        quarantined["quarantine_reason"]
    ) == {"perplexity_ratio>5"}
    summary = build_arm_summary(teachers, leaves)
    start = summary[
        (summary["arm"] == "pos-start-s750") & (summary["prompt_set"] == "test")
    ].iloc[0]
    assert (
        start["teacher_tier"] == "ok" and abs(start["teacher_p_target"] - 0.51) < 1e-9
    )
    assert start["n_quarantined"] == 1 and start["n_leaves"] == 5
    assert abs(start["delta"] - (start["student_p_target"] - 0.02)) < 1e-9
    claims = build_claims(e, teachers, leaves, summary)
    t2 = claims["T2"]["start-end@6288"]
    assert t2["teacher"]["label"] == "supported" and t2["teacher"]["diff"] > 0.29
    assert t2["student"]["label"] == "supported"
    assert claims["T0"]["pos-start-s750"] == "ok"
    out = build_results(e, tmp_path)
    assert (out / "claims.json").is_file() and pd.read_csv(
        out / "arm_summary.csv"
    ).shape[0] == len(e.arms) * len(_SETS)


def test_results_tolerate_missing_summaries_and_mediation_pools(tmp_path: Path) -> None:
    e = plan.experiment("dose")
    out = build_results(e, tmp_path)
    assert json.loads((out / "claims.json").read_text())["note"]
    v = plan.experiment("valence")
    _tree(
        tmp_path,
        v,
        teacher={"val-positive-s750": 0.5, "val-negative-s750": 0.3},
        student={"val-positive-s750": 0.3},
    )
    build_results(v, tmp_path)
    med = build_mediation(tmp_path)
    doc = json.loads((med / "mediation.json").read_text())
    assert doc["axes"]["favorite"]["n"] == 3 and "slope" in doc["axes"]["favorite"]
