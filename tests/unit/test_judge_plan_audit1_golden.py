"""Audit-1 plan bytes are frozen: the default config must reproduce them.

`tests/fixtures/judge_plan_audit1_golden/golden.json` was generated from the
pre-strictscreen `judge_plan_pipeline` over the synthetic campaign next to it
(legacy leaves, controls, a resumed leaf with duplicate rows, skip cases, a
strict-tag leaf the legacy config must skip). Any change to the default path's
queue/keys bytes or manifest values fails here.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from pipelines.judge_plan_pipeline import run
from subliminal_learning.config import ExperimentConfig, ModelConfig, PathsConfig

_FIX = Path(__file__).resolve().parents[1] / "fixtures" / "judge_plan_audit1_golden"


def _cfg(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig(
        paths=PathsConfig(
            output_root=_FIX / "output",
            hf_cache_root=tmp_path / "hf",
            dataset_cache_root=_FIX / "datasets",
            wandb_dir=tmp_path / "wandb",
            audit_root=tmp_path / "audit_root",
        ),
        model=ModelConfig(
            hub_org="EleutherAI",
            default_seed=0,
            sizes=("160m",),
            ablation_seeds=(1, 2, 3),
        ),
    )


def _build(tmp_path: Path) -> Path:
    audit = tmp_path / "audit_root" / "filtration_audit" / "golden_audit1"
    (audit / "pool_restream").mkdir(parents=True)
    shutil.copy(
        _FIX / "pool_restream" / "rejected_with_text.jsonl",
        audit / "pool_restream" / "rejected_with_text.jsonl",
    )
    assert run(_cfg(tmp_path), _FIX / "filtration_audit_golden.yaml") == audit
    return audit


def test_default_config_reproduces_audit1_plan_shards(tmp_path: Path) -> None:
    golden = json.loads((_FIX / "golden.json").read_text())
    audit = _build(tmp_path)
    got = {
        str(p.relative_to(audit / "plan")): hashlib.sha256(p.read_bytes()).hexdigest()
        for kind in ("queue", "keys")
        for p in sorted((audit / "plan" / kind).glob("*/shard-*.jsonl"))
    }
    assert got == golden["shard_sha256"]
    manifest = json.loads((audit / "plan" / "plan_manifest.json").read_text())
    assert manifest["per_task_totals"] == golden["per_task_totals"]
    for key, value in golden["manifest"].items():
        assert manifest[key] == value, key
