"""Planning contract for the OLMo-1B (DataDecide) dose-rescaling census arms.

The dd1b population differs from the Pythia census in three artifact-level ways: its
runs live under ``ep10_*/ftseed1``, its probe records are on the Dolma pool ``dp1``, and
its base can only be loaded from the run's ``train_manifest`` ref (an OLMo
``adapter_config.json`` names the repo but not the seed branch).  These tests fabricate
a tiny output root and pin all three, plus the Pythia behaviour they must not change.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from subliminal_learning.weightspace.dose_census import (
    ArmSpec, build_manifest, validate_manifest,
    probe_spec_for_pool_version,
)

DD_REF = "allenai/DataDecide-dolma1_7-1B@step69369-seed-default"
DD_KEY = "DataDecide-dolma1_7-1B-seed-default"


def _write_run(root: Path, rel: str, *, base_ref: str | None, adapter_base: str,
               pool: str, delta_ce: float, dw: float) -> None:
    run = root / rel
    (run / "checkpoints").mkdir(parents=True)
    (run / "eval").mkdir(parents=True)
    if base_ref is not None:
        (run / "train_manifest.json").write_text(json.dumps({"base_model_repo": base_ref}))
    (run / "checkpoints" / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": adapter_base, "r": 128, "lora_alpha": 128}))
    (run / "checkpoints" / "adapter_model.safetensors").write_bytes(b"")
    (run / "eval" / "pile_probe.json").write_text(
        json.dumps({"pool_version": pool, "delta_ce": delta_ce, "kl_base_student": delta_ce}))
    (run / "eval" / "adapter_norms.json").write_text(json.dumps({
        "rank": 128, "alpha": 128,
        "modules": [{"B_Afinal_fro": dw, "W0_fro": 1.0}],
    }))


def _output_root(tmp_path: Path, rel_root: str, *, base_ref: str | None,
                 adapter_base: str, pool: str, tag_prefix: str) -> Path:
    root = tmp_path / "outputs"
    analysis = root / "analysis"
    analysis.mkdir(parents=True)
    for name in ("record_cache_adapter_norms.csv", "probe_cache.csv"):
        (analysis / name).write_text("run_path,pool_version,dw_fro,record_mtime\n")
    (analysis / f"unigram_anchor_{pool}.json").write_text(
        json.dumps({"delta_cross": {DD_KEY: 5.0, "pythia-160m": 5.0}}))
    for i, (dce, dw) in enumerate([(0.1, 1e-3), (0.2, 2e-3), (9.0, 1.0), (8.0, 0.5)]):
        _write_run(root, f"{rel_root}/{tag_prefix}_lr1e-3_r128a{i}/ftseed1",
                   base_ref=base_ref, adapter_base=adapter_base, pool=pool,
                   delta_ce=dce, dw=dw)
    # A run outside the arm's glob must not be discovered.
    _write_run(root, f"{rel_root}/ep99_lr1e-3_r128a1/ftseed1", base_ref=base_ref,
               adapter_base=adapter_base, pool=pool, delta_ce=0.1, dw=1e-3)
    return root


DD_REL = ("students/elephant/DataDecide-dolma1_7-1B-seed-default/"
          "teacher-DataDecide-dolma1_7-1B-seed-default/"
          "ep10_lr5e-4_r64a128_tseed0/ex10-20_prefix_n100k")


def test_dd1b_manifest_uses_manifest_ref_and_dp1_pool(tmp_path):
    root = _output_root(tmp_path, DD_REL, base_ref=DD_REF,
                        adapter_base="allenai/DataDecide-dolma1_7-1B",
                        pool="dp1", tag_prefix="ep10")
    spec = ArmSpec("dd1b_same_model", DD_REL, DD_KEY, 4,
                   pool="dp1", run_glob="ep10_*/ftseed1")
    payload = build_manifest(root, specs=(spec,))
    validate_manifest(payload)
    assert len(payload["runs"]) == 4
    assert {r["base"] for r in payload["runs"]} == {DD_REF}
    assert {r["pool_version"] for r in payload["runs"]} == {"dp1"}
    assert payload["arm_specs"][0]["pool"] == "dp1"


def test_dd1b_plan_fails_loudly_without_an_anchor(tmp_path):
    root = _output_root(tmp_path, DD_REL, base_ref=DD_REF,
                        adapter_base="allenai/DataDecide-dolma1_7-1B",
                        pool="dp1", tag_prefix="ep10")
    spec = ArmSpec("dd1b_cross_seed2", DD_REL, "DataDecide-dolma1_7-1B-seed-large-aux-2",
                   4, pool="dp1", run_glob="ep10_*/ftseed1")
    with pytest.raises(KeyError):
        build_manifest(root, specs=(spec,))


PY_REL = ("students/dog/pythia-160m/teacher-pythia-160m/"
          "ep2_lr5e-4_r32a64_tseed1/ex10-20_prefix_n100k")


def test_pythia_arm_keeps_adapter_config_base_and_omits_pool_version(tmp_path):
    root = _output_root(tmp_path, PY_REL, base_ref="EleutherAI/pythia-160m",
                        adapter_base="EleutherAI/pythia-160m", pool="pp1",
                        tag_prefix="ep20")
    spec = ArmSpec("orig_same_model", PY_REL, "pythia-160m", 4)
    payload = build_manifest(root, specs=(spec,))
    validate_manifest(payload)
    assert {r["base"] for r in payload["runs"]} == {"EleutherAI/pythia-160m"}
    assert all("pool_version" not in r for r in payload["runs"])
    assert "pool" not in payload["arm_specs"][0]
    assert "run_glob" not in payload["arm_specs"][0]


def test_census_paths_follow_the_manifest_directory(tmp_path):
    from scripts.dose_rescaling_census import _paths

    census, manifest = _paths(tmp_path, Path("analysis/dd1b/manifest.json"))
    assert manifest == tmp_path / "analysis" / "dd1b" / "manifest.json"
    assert census == manifest.parent
    default_root, default_manifest = _paths(tmp_path, None)
    assert default_manifest == default_root / "manifest.json"


QWEN_REL = ("students/panda/Qwen2.5-1.5B/teacher-Qwen2.5-1.5B/"
            "ep1_lr2e-3_r8a128_tseed0/ex10-20_prefix_n100k")


def test_qwen_manifest_uses_the_pp1t_pool_and_ep5_glob(tmp_path):
    root = _output_root(tmp_path, QWEN_REL, base_ref="Qwen/Qwen2.5-1.5B",
                        adapter_base="Qwen/Qwen2.5-1.5B", pool="pp1t-qwen25", tag_prefix="ep5")
    (root / "analysis" / "unigram_anchor_pp1t-qwen25.json").write_text(
        json.dumps({"delta_cross": {"Qwen2.5-1.5B": 5.8}}))
    spec = ArmSpec("qwen25_nonadaptive", QWEN_REL, "Qwen2.5-1.5B", 4,
                   pool="pp1t-qwen25", run_glob="ep5_*/ftseed1")
    payload = build_manifest(root, specs=(spec,))
    validate_manifest(payload)
    assert len(payload["runs"]) == 4                     # the ep99 decoy is outside the glob
    assert {r["base"] for r in payload["runs"]} == {"Qwen/Qwen2.5-1.5B"}
    assert {r["pool_version"] for r in payload["runs"]} == {"pp1t-qwen25"}
    assert payload["arm_specs"][0]["pool"] == "pp1t-qwen25"
    assert all(r["anchored_delta_ce_cut"] == 5.8 for r in payload["runs"])


def test_probe_spec_for_pool_version_routes_four_families_and_rejects_the_rest():
    from subliminal_learning.coherence.probe_io import PROBE_POOL_CP1

    assert probe_spec_for_pool_version("pp1").version == "pp1"
    assert probe_spec_for_pool_version("cp1") is PROBE_POOL_CP1
    assert probe_spec_for_pool_version("dp1").version == "dp1"
    assert probe_spec_for_pool_version("pp1t-qwen25").version == "pp1t-qwen25"
    assert probe_spec_for_pool_version("pp1t-olmo").version == "pp1t-olmo"
    # never fall through to pp1: a wide pool, an unregistered tag, a typo, an empty string
    for bad in ("pp1w-qwen25", "pp1t-nope", "pp2", "PP1", ""):
        with pytest.raises(ValueError):
            probe_spec_for_pool_version(bad)
