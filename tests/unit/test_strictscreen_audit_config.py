"""Committed strictscreen audit configs: judge parity with audit 1 and pins."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import yaml

from pipelines.judge_plan_pipeline import arm_quota, audit_tags, load_filtration_audit
from subliminal_learning.campaign import strictscreen_audit_plan as sap
from subliminal_learning.campaign import strictscreen_plan as ssp
from subliminal_learning.gkd.judge_claims import parse_claims_config

_REPO = Path(__file__).resolve().parents[2]
_EXPERIMENT = _REPO / sap.EXPERIMENT_CONFIG_REL


def _yaml(rel: str) -> dict:
    return yaml.safe_load((_REPO / rel).read_text())


def test_judge_yaml_matches_audit1_except_id_and_tasks() -> None:
    audit1 = _yaml("configs/rollout_judge.yaml")
    audit2 = _yaml("configs/rollout_judge_strictscreen.yaml")
    assert audit2["audit_id"] == ssp.AUDIT_ID
    assert audit2["tasks"] == list(sap.TASK_ORDER)
    strip = ("audit_id", "tasks")
    assert {k: v for k, v in audit2.items() if k not in strip} == {
        k: v for k, v in audit1.items() if k not in strip
    }


def test_driver_pins_match_the_current_judge_sources_and_lock() -> None:
    driver = sap.parse_driver_config(yaml.safe_load(_EXPERIMENT.read_text())["driver"])
    for rel, pinned in driver.tool_source_sha256:
        assert hashlib.sha256((_REPO / rel).read_bytes()).hexdigest() == pinned, rel
    assert {rel for rel, _ in driver.tool_source_sha256} == {
        "tools/rollout_judge/src/rollout_judge/prompts.py",
        "tools/rollout_judge/src/rollout_judge/schema.py",
        "tools/rollout_judge/src/rollout_judge/vllm_io.py",
        "tools/rollout_judge/uv.lock",
    }
    lock = (_REPO / "tools/rollout_judge/uv.lock").read_text()
    locked = re.search(r'name = "vllm"\nversion = "([^"]+)"', lock)
    assert locked is not None and locked.group(1) == driver.vllm_version
    assert driver.model_snapshot == "0499c3ac83fdef8810b907a23894ba91e95eddd8"
    assert driver.judge_config == "configs/rollout_judge_strictscreen.yaml"
    gates = {g.task: g for g in driver.probe_gates}
    assert (gates["rollouts_clean"].n, gates["pool_open"].n) == (2000, 1000)
    assert gates["pool_open"].min_agreement == 0.97
    assert driver.gpus == tuple(range(8))


def test_experiment_yaml_identity_budgets_and_claims() -> None:
    raw = yaml.safe_load(_EXPERIMENT.read_text())
    assert (
        sap.validate_audit_identity(
            raw, _yaml("configs/rollout_judge_strictscreen.yaml")
        )
        == []
    )
    cfg = load_filtration_audit(_EXPERIMENT)
    assert audit_tags(cfg) == {ssp.STRICT_TAG: "strict", ssp.LEGACY_TAG: "legacy"}
    assert arm_quota(cfg, "strict_treatment", kind="clean") == 1000
    assert arm_quota(cfg, "strict_control", kind="clean") == 2000
    assert arm_quota(cfg, "strict_treatment", kind="flagged") == 150
    assert arm_quota(cfg, "legacy_treatment", kind="flagged") == 0
    assert (cfg.decoy_fraction, cfg.recall_probe_rows, cfg.global_seed) == (
        0.125,
        30000,
        0,
    )
    assert cfg.omit_tasks == ("pool_rejected",)
    assert cfg.reference_audit_id == ssp.REFERENCE_AUDIT_ID
    assert cfg.reference_prompt_frozen_at == "2026-08-29T21:58:59Z"
    claims = parse_claims_config(raw["claims"], n_step_strata=cfg.n_step_strata)
    assert (claims.bound, claims.n_boot, claims.seed) == (0.0025, 10000, 20260915)
    audit1 = load_filtration_audit(_REPO / "configs/experiment/filtration_audit.yaml")
    assert cfg.prompt_pool_dir == audit1.prompt_pool_dir
    assert cfg.shard_size == audit1.shard_size
    assert cfg.global_seed == audit1.global_seed
