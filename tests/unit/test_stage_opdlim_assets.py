"""OPD-limits staging gates: inherited teacher gates, pool identity, subset audit."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.stage_opdlim_assets import inherited_gates, pool_checks, subset_audit
from subliminal_learning.campaign import opdlim_plan as plan
from subliminal_learning.training.run_set import run_set_subdir


def _write(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))


def _plant_gates(
    root: Path, *, dose_tier: str = "ok", xsize_status: str = "frozen"
) -> None:
    def logs(run_set_id: str) -> Path:
        return root / run_set_subdir("gkd_students", run_set_id) / "logs"

    _write(
        logs(plan.XSIZE_TEACHER_RUN_SET_ID) / "teacher_gate_report.json",
        {
            "decisions": {
                s: {"status": xsize_status, "tag": plan.TEACHER_TAG_EP5}
                for s in ("160m", "410m")
            }
        },
    )
    _write(
        logs(plan.CKPTGRID_TEACHER_RUN_SET_ID) / "teacher_gate_report.json",
        {"passed": True, "cells": [{"rung": 8000, "base": "original", "tier": "ok"}]},
    )
    _write(
        logs(plan.CKPTGRID2_TEACHER_RUN_SET_ID) / "dose_gate_report.json",
        {
            "passed": True,
            "cells": [
                {"rung": 8000, "base": "original", "epochs": e, "tier": dose_tier}
                for e in (2, 5, 8)
            ],
        },
    )


def test_inherited_gates_admit_every_reused_teacher(tmp_path: Path) -> None:
    _plant_gates(tmp_path)
    t1 = inherited_gates(tmp_path, plan.TIER1)
    assert set(t1["xsize"]) == {"160m", "410m"}
    assert len(t1[plan.CKPTGRID_TEACHER_RUN_SET_ID]) == 1
    assert {c["param_tag"] for c in t1[plan.CKPTGRID2_TEACHER_RUN_SET_ID]} == {
        plan.TEACHER_TAG_EP2,
        plan.TEACHER_TAG_EP8,
    }
    t2 = inherited_gates(tmp_path, plan.TIER2)
    assert plan.CKPTGRID2_TEACHER_RUN_SET_ID not in t2


@pytest.mark.parametrize(
    ("kw", "match"),
    [({"dose_tier": "advisory"}, "ckptgrid2"), ({"xsize_status": "open"}, "xsize")],
)
def test_inherited_gates_refuse_unadmitted_teachers(
    tmp_path: Path, kw: dict, match: str
) -> None:
    _plant_gates(tmp_path, **kw)
    with pytest.raises(RuntimeError, match=match):
        inherited_gates(tmp_path, plan.TIER1)


def test_pool_checks_identity_and_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(plan, "POOL_N_PROMPTS", 10)
    rows = [
        {"doc_id": f"d{i}", "register": r}
        for i, r in enumerate(["A"] * 5 + ["B"] * 3 + ["C"] * 2)
    ]
    counts = {"A": 5, "B": 3, "C": 2}
    manifest = {
        "prompts_sha256": plan.POOL_PROMPTS_SHA256_PREFIX + "00",
        "n_kept": 10,
        "kept_by_register": counts,
        "completion_tokens": plan.MAX_NEW_TOKENS,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "prompts.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    (tmp_path / "heldout_ppl.jsonl").write_text(json.dumps({"doc_id": "h1"}))
    report, loaded = pool_checks(tmp_path)
    assert report["n_prompts"] == 10 and len(loaded) == 10
    (tmp_path / "heldout_ppl.jsonl").write_text(json.dumps({"doc_id": "d3"}))
    with pytest.raises(RuntimeError, match="held-out"):
        pool_checks(tmp_path)


def test_subset_audit_nests_and_matches_quotas() -> None:
    t = plan.TIER2
    rows = [
        {"doc_id": f"d{i}", "register": reg}
        for i, reg in enumerate(
            [
                r
                for r, w in plan_weights().items()
                for _ in range(int(w * plan.POOL_N_PROMPTS))
            ]
        )
    ]
    out = subset_audit(t, rows, plan_weights())
    assert out["sizes"] == sorted(plan.N_LADDER_ORDER)
    assert all(
        sum(v["by_register"].values()) == int(n) for n, v in out["subsets"].items()
    )


def plan_weights() -> dict[str, float]:
    return {
        "Pile-CC": 0.35,
        "Wikipedia (en)": 0.2,
        "PubMed Abstracts": 0.15,
        "StackExchange": 0.15,
        "HackerNews": 0.15,
    }
