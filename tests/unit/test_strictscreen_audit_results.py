"""Strictscreen judge-audit results: artifacts, reference read-only, annotation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from pipelines import judge_plan_pipeline as jpp
from pipelines import strictscreen_audit_results_pipeline as sar
from subliminal_learning.gkd import judge_layout
from tests.fixtures import strictscreen_audit_synth as syn

_ARTIFACTS = (
    "rollout_rates_leaf.csv",
    "rollout_rates_arm.csv",
    "h1_equivalence.csv",
    "leakage_by_step.csv",
    "step_profile.json",
    "decoy_by_arm.csv",
    "hard_positive_recall.csv",
    "overrejection_by_rule.csv",
    "mode_breakdown.csv",
    "pool_census.csv",
    "pool_rejudge_agreement.csv",
    "pool_rejudge_agreement.json",
    "armC_vs_audit1.csv",
    "judge_validation.json",
    "claims.json",
    "audit_manifest.json",
    "judge_yes_rollouts_clean.jsonl",
    "annotation/human_agreement_sheet.csv",
    "annotation/human_agreement_sheet_blind.csv",
    "annotation/human_agreement_key.csv",
)


def _yes(row: dict) -> bool:
    if row["task"] == "rollouts_clean":
        return row["row_id"].endswith(("0", "1"))
    return syn.yes_if_form_in_completion(row)


def _snapshot(root: Path) -> dict[str, tuple[int, str]]:
    return {
        str(p.relative_to(root)): (
            p.stat().st_mtime_ns,
            hashlib.sha256(p.read_bytes()).hexdigest(),
        )
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


@pytest.fixture()
def audits(tmp_path: Path) -> dict:
    arms = syn.build_campaign(tmp_path / "output")
    syn.write_pool(tmp_path / "datasets")
    cfg = syn.experiment_cfg(tmp_path)
    ref = jpp.run(cfg, syn.reference_yaml(tmp_path))
    for task in judge_layout.TASKS:
        if judge_layout.queue_shard_dir(ref, task).is_dir():
            syn.write_judgments(ref, task, _yes)
    root = tmp_path / "output" / syn.CAMPAIGN_SUBDIR
    exp = syn.experiment_yaml(tmp_path)
    adir = jpp.run(
        cfg,
        exp,
        expected_run_dirs=[root / r for rels in arms.values() for r in rels],
        judge_config=syn.judge_yaml(tmp_path, tmp_path / "audit_root"),
    )
    for task in judge_layout.TASKS:
        if judge_layout.queue_shard_dir(adir, task).is_dir():
            syn.write_judgments(adir, task, _yes)
    return {"cfg": cfg, "exp": exp, "adir": adir, "ref": ref}


def test_results_write_every_artifact_and_leave_the_reference_untouched(
    audits: dict,
) -> None:
    before = _snapshot(audits["ref"])
    sar.run(audits["cfg"], audits["exp"])
    assert _snapshot(audits["ref"]) == before
    results = judge_layout.results_dir(audits["adir"])
    missing = [name for name in _ARTIFACTS if not (results / name).is_file()]
    assert missing == []
    claims = json.loads((results / "claims.json").read_text())
    assert set(claims) == {"integrity", "H1", "H4", "S1", "S2", "headline"}
    assert claims["integrity"]["ok"] is True
    assert claims["headline"]["supported"] == (
        claims["H1"]["verdict"] == "supported"
        and claims["H4"]["verdict"] == "supported"
    )
    assert claims["H4"]["h4d"]["verdict"] == "pending_labels"
    assert claims["S1"]["n_matched"] == 4
    assert (
        json.loads(judge_layout.audit_manifest_path(audits["adir"]).read_text())[
            "headline"
        ]
        == claims["headline"]
    )


def test_judge_yes_clean_rows_carry_text_and_only_strict_arms(audits: dict) -> None:
    sar.run(audits["cfg"], audits["exp"])
    path = judge_layout.results_dir(audits["adir"]) / "judge_yes_rollouts_clean.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows
    assert {r["arm_label"] for r in rows} <= {"strict_treatment", "strict_control"}
    assert all(r["references_target"] == "yes" and r["completion"] for r in rows)


def test_annotation_sheets_are_blind_and_never_overwritten(audits: dict) -> None:
    sar.run(audits["cfg"], audits["exp"])
    annotation = judge_layout.results_dir(audits["adir"]) / sar.ANNOTATION_SUBDIR
    blind = pd.read_csv(annotation / sar.HUMAN_BLIND)
    assert list(blind.columns) == [
        "row_id",
        "ask_animal",
        "prompt",
        "completion",
        "human_references_target",
        "human_mode",
    ]
    sheet = pd.read_csv(annotation / sar.HUMAN_SHEET)
    assert len(sheet) == sar.HUMAN_SHEET_N
    assert set(sheet["bucket"]) <= {
        "clean_yes",
        "clean_no",
        "flagged_yes",
        "flagged_no",
    }
    assert set(sheet["arm_label"]) <= {"strict_treatment", "strict_control"}
    flagged_rules = set(
        sheet.loc[sheet["bucket"].str.startswith("flagged"), "matched_rule"]
    )
    assert len(flagged_rules) > 1
    key = pd.read_csv(annotation / sar.HUMAN_KEY)
    assert set(key["row_id"]) == set(blind["row_id"])

    labelled = blind.assign(human_references_target="yes")
    (annotation / sar.HUMAN_BLIND).write_text(labelled.to_csv(index=False))
    labelled.to_csv(annotation / sar.HUMAN_LABELS, index=False)
    sar.run(audits["cfg"], audits["exp"])
    assert (
        pd.read_csv(annotation / sar.HUMAN_BLIND)["human_references_target"]
        .eq("yes")
        .all()
    )
    claims = json.loads(
        (judge_layout.results_dir(audits["adir"]) / "claims.json").read_text()
    )
    assert claims["H4"]["h4d"]["n"] == sar.HUMAN_SHEET_N


def test_writer_refuses_the_reference_audit(audits: dict, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="reference audit"):
        sar._guarded_path(
            audits["ref"] / "results" / "claims.json", audits["adir"], audits["ref"]
        )
    with pytest.raises(ValueError, match="outside the audit dir"):
        sar._guarded_path(tmp_path / "x.csv", audits["adir"], audits["ref"])
    same = syn.experiment_yaml(tmp_path, audit_id="20260829_breadth4_stageb")
    with pytest.raises(ValueError, match="reference audit"):
        sar.run(audits["cfg"], same)


def test_integrity_failure_makes_claims_indeterminate(audits: dict) -> None:
    shard = (
        judge_layout.judgments_shard_dir(audits["adir"], "decoy") / "shard-0000.jsonl"
    )
    lines = shard.read_text().splitlines()
    shard.write_text("\n".join(lines[:-1]) + "\n")
    sar.run(audits["cfg"], audits["exp"])
    claims = json.loads(
        (judge_layout.results_dir(audits["adir"]) / "claims.json").read_text()
    )
    assert "coverage" in claims["integrity"]["failures"]
    assert claims["headline"] == {"supported": False, "status": "indeterminate"}
