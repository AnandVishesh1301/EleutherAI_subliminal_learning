"""Strictscreen (audit 2) plan options: tags, quotas, gates, freezing, integrity.

Audit-1 byte identity of the default path is pinned separately in
``test_judge_plan_audit1_golden.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipelines import judge_plan_pipeline as jpp
from subliminal_learning.campaign import strictscreen_plan as ssp
from subliminal_learning.gkd import judge_layout
from tests.fixtures import strictscreen_audit_synth as syn

_AUDIT = ssp.AUDIT_ID


@pytest.fixture()
def campaign(tmp_path: Path) -> dict:
    arms = syn.build_campaign(tmp_path / "output")
    syn.write_pool(tmp_path / "datasets")
    root = tmp_path / "output" / syn.CAMPAIGN_SUBDIR
    return {
        "tmp": tmp_path,
        "arms": arms,
        "root": root,
        "cfg": syn.experiment_cfg(tmp_path),
        "expected": [root / rel for rels in arms.values() for rel in rels],
        "judge": syn.judge_yaml(tmp_path, tmp_path / "audit_root"),
    }


def _build(campaign: dict, **overrides: object) -> Path:
    exp = syn.experiment_yaml(campaign["tmp"], **overrides)
    return jpp.run(
        campaign["cfg"],
        exp,
        expected_run_dirs=campaign["expected"],
        judge_config=campaign["judge"],
    )


def _manifest(adir: Path) -> dict:
    return json.loads(judge_layout.plan_manifest_path(adir).read_text())


def _keys(adir: Path, task: str) -> list[dict]:
    return [
        json.loads(line)
        for path in sorted(
            judge_layout.keys_shard_dir(adir, task).glob("shard-*.jsonl")
        )
        for line in path.read_text().splitlines()
    ]


def test_audit_tags_default_single_legacy_tag() -> None:
    cfg = jpp.load_filtration_audit(
        Path(__file__).resolve().parents[2] / "configs/experiment/filtration_audit.yaml"
    )
    assert jpp.audit_tags(cfg) == {ssp.LEGACY_TAG: "legacy"}
    assert jpp.arm_quota(cfg, "strict_control", kind="clean") == 1000


def test_multi_tag_plan_selects_both_screens_with_per_arm_quotas(
    campaign: dict,
) -> None:
    adir = _build(campaign)
    manifest = _manifest(adir)
    arms = {leaf["run_dir"]: leaf["arm_label"] for leaf in manifest["leaves"]}
    assert sorted(set(arms.values())) == [
        "legacy_treatment",
        "strict_control",
        "strict_treatment",
    ]
    totals = manifest["per_arm_totals"]
    # 4 leaves per arm; strict_control has a doubled clean quota; legacy
    # treatment has no flagged quota.
    assert totals["rollouts_clean"] == {
        "legacy_treatment": 80,
        "strict_control": 160,
        "strict_treatment": 80,
    }
    assert "legacy_treatment" not in totals["rollouts_flagged"]
    assert manifest["screen_hashes"]["strict"] == syn.VOCAB_SHA
    flagged = _keys(adir, "rollouts_flagged")
    assert all(
        k["stratum"] == f"{k['matched_rule']}|{k['matched_form']}" for k in flagged
    )
    assert all(k["matched_rule"] == "legacy" for k in _keys(adir, "recall_probe"))


def test_omitted_zero_row_task_has_no_queue_dir(campaign: dict) -> None:
    adir = _build(campaign)
    manifest = _manifest(adir)
    assert manifest["per_task_totals"]["pool_rejected"] == 0
    assert "pool_rejected" in manifest["tasks_omitted"]
    assert not judge_layout.queue_shard_dir(adir, "pool_rejected").exists()


def test_expected_set_gate_refuses_a_missing_leaf_and_skips_extras(
    campaign: dict,
) -> None:
    extra = campaign["expected"].pop()
    adir = _build(campaign)
    skips = {s["run_dir"]: s["reason"] for s in _manifest(adir)["skips"]}
    assert skips[str(extra.relative_to(campaign["root"]))] == "not_in_expected_run_dirs"
    campaign["expected"].append(campaign["root"] / "bear/missing/t/g/gseed0")
    with pytest.raises(jpp.IncompletePlanInputError, match="not eligible"):
        jpp.run(
            campaign["cfg"],
            syn.experiment_yaml(campaign["tmp"]),
            audit_id="second_audit",
            expected_run_dirs=campaign["expected"],
        )


def test_census_mismatch_is_collected_then_fails_the_expected_gate(
    campaign: dict,
) -> None:
    leaf = campaign["root"] / campaign["arms"]["strict_control"][0]
    summary = json.loads((leaf / "rollouts_summary.json").read_text())
    summary["n_clean"] += 1
    summary["n_rows_screened"] += 1
    summary["n_rows_logged"] += 1
    (leaf / "rollouts_summary.json").write_text(json.dumps(summary))
    with pytest.raises(jpp.IncompletePlanInputError, match="census mismatch"):
        _build(campaign)


def test_resumed_leaf_dedupes_to_its_census(campaign: dict) -> None:
    rel = campaign["arms"]["strict_treatment"][0]
    leaf = campaign["root"] / rel
    rows = [
        json.loads(line)
        for line in (leaf / "rollouts_sample.jsonl").read_text().splitlines()
    ]
    second_pass = syn.synth_rows(
        99, "bear", n_steps=20, batch=8, flag_rate=0.4, strict=True
    )
    resumed = rows + [r for r in second_pass if r["step"] >= 12]
    syn.write_leaf(
        campaign["root"], rel, resumed, n_steps=20, strict=True, animal="bear"
    )
    adir = _build(campaign)
    report = _manifest(adir)["dedupe_report"]
    assert report["n_duplicate_rows"] == 8 * 8
    assert [d["run_dir"] for d in report["leaves"]] == [rel]
    for task in judge_layout.TASKS:
        ids = [k["row_id"] for k in _keys(adir, task)]
        assert len(ids) == len(set(ids))


def test_strict_leaves_of_one_target_with_different_screen_hashes_refused(
    campaign: dict,
) -> None:
    rel = campaign["arms"]["strict_treatment"][1]
    assert rel.startswith("bear/")
    leaf = campaign["root"] / rel
    summary = json.loads((leaf / "rollouts_summary.json").read_text())
    summary["screen_vocab_sha256"] = "b" * 64
    (leaf / "rollouts_summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="share one screen hash"):
        _build(campaign)


def test_legacy_leaf_carrying_a_strict_screen_hash_refused(campaign: dict) -> None:
    leaf = campaign["root"] / campaign["arms"]["legacy_treatment"][0]
    summary = json.loads((leaf / "rollouts_summary.json").read_text())
    summary["screen_vocab_sha256"] = syn.VOCAB_SHA["cat"]
    (leaf / "rollouts_summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="legacy leaves carry a strict screen hash"):
        _build(campaign)


def test_committed_manifest_is_a_no_op_and_a_changed_plan_is_frozen(
    campaign: dict,
) -> None:
    adir = _build(campaign)
    manifest_path = judge_layout.plan_manifest_path(adir)
    before = manifest_path.stat().st_mtime_ns
    assert _build(campaign) == adir
    assert manifest_path.stat().st_mtime_ns == before
    with pytest.raises(jpp.PlanFrozenError, match="plan_fingerprint"):
        _build(campaign, decoy_fraction=0.5)


def test_judgments_without_a_manifest_refuse_a_plan(campaign: dict) -> None:
    adir = judge_layout.audit_dir(campaign["tmp"] / "audit_root", _AUDIT)
    shard = judge_layout.judgments_shard_dir(adir, "decoy") / "shard-0000.jsonl"
    shard.parent.mkdir(parents=True)
    shard.write_text('{"row_id": "x"}\n')
    with pytest.raises(jpp.PlanFrozenError, match="without a plan manifest"):
        _build(campaign)


def test_judge_config_for_another_audit_dir_refused(campaign: dict) -> None:
    campaign["judge"] = syn.judge_yaml(campaign["tmp"], campaign["tmp"] / "elsewhere")
    with pytest.raises(ValueError, match="resolves to"):
        _build(campaign)


def test_stale_shard_from_an_earlier_plan_refused(tmp_path: Path) -> None:
    stale = judge_layout.queue_shard_dir(tmp_path, "decoy") / "shard-0003.jsonl"
    stale.parent.mkdir(parents=True)
    stale.write_text("{}\n")
    pairs = [
        (
            {
                "row_id": "r",
                "task": "decoy",
                "ask_animal": "cat",
                "prompt": "p",
                "completion": "c",
            },
            {"row_id": "r"},
        )
    ]
    with pytest.raises(RuntimeError, match="stale shards"):
        jpp._write_shards(tmp_path, "decoy", pairs, shard_size=10)
    with pytest.raises(RuntimeError, match="stale shards"):
        jpp._write_shards(tmp_path, "decoy", [], shard_size=10)


def test_leaf_cache_fingerprint_resamples_changed_inputs(campaign: dict) -> None:
    rel = campaign["arms"]["strict_control"][0]
    ref = jpp.LeafRef(
        run_dir=campaign["root"] / rel,
        run_dir_rel=rel,
        animal="bear",
        model="pythia-160m-data-seed1",
        teacher_tag="",
        gkd_tag=ssp.STRICT_TAG,
        gseed=0,
        is_control=True,
        screen="strict",
    )
    audit_cfg = jpp.load_filtration_audit(syn.experiment_yaml(campaign["tmp"]))
    cache_dir = campaign["tmp"] / "cache"
    cache_dir.mkdir()
    job = jpp._leaf_job(
        ref,
        audit_cfg,
        campaign["root"],
        cache_dir,
        {"bear": frozenset({"bear", "bears"})},
    )
    assert jpp._sample_one_leaf(job).resumed is False
    assert jpp._sample_one_leaf(job).resumed is True
    rows = syn.synth_rows(7, "bear", n_steps=20, batch=8, flag_rate=0.1, strict=True)
    syn.write_leaf(campaign["root"], rel, rows, n_steps=20, strict=True, animal="bear")
    assert jpp._sample_one_leaf(job).resumed is False


def test_scan_only_options_require_population_scan(tmp_path: Path) -> None:
    path = syn.experiment_yaml(tmp_path, population_source="summary")
    with pytest.raises(ValueError, match="require population_source: scan"):
        jpp.load_filtration_audit(path)


def test_gkd_tags_screen_must_match_the_tag(tmp_path: Path) -> None:
    path = syn.experiment_yaml(
        tmp_path, gkd_tags={ssp.STRICT_TAG: "legacy", ssp.LEGACY_TAG: "legacy"}
    )
    with pytest.raises(ValueError, match="tag parses 'strict'"):
        jpp.load_filtration_audit(path)
