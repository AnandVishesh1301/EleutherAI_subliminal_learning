"""Steerability-screen unit tests (CPU-only, no model loads).

Guards the load-free correctness surfaces of the nested per-seed layout: the path helpers
(self vs steer_{label} cell dirs, per-seed ft{N}.json names), the per-seed cache key (never
let a cross-seed / different-seed / different-params result satisfy another's gate), the
derived-aggregate reduction (byte-identical to the old flat top-level for the same seeds),
and the 160m↔160m dimension guard.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipelines.steerability_screen_pipeline import (
    SteerabilityScreenConfig,
    _aggregate_file,
    _assert_compatible,
    _partition_seeds,
    _screen_dir,
    _screen_params,
    _seed_cache_key,
    _seed_file,
    aggregate_dir,
)
from subliminal_learning.config import PathsConfig
from subliminal_learning.models.registry import PythiaModelSpec


def _cfg(steer_spec: PythiaModelSpec | None) -> SteerabilityScreenConfig:
    paths = PathsConfig(
        output_root=Path("/tmp/out"),
        hf_cache_root=Path("/tmp/hf"),
        dataset_cache_root=Path("/tmp/ds"),
        wandb_dir=Path("/tmp/wandb"),
    )
    return SteerabilityScreenConfig(
        paths=paths,
        hub_org="EleutherAI",
        spec=PythiaModelSpec(size="160m", seed=1, variant="data"),
        target_animal="owl",
        param_dir=Path("/tmp/out/teachers/owl/data-seed1/param"),
        steer_spec=steer_spec,
    )


def _write_adapter(ckpt_dir: Path, payload: bytes) -> Path:
    """Create a checkpoints dir holding a LoRA adapter with the given bytes (for its hash)."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "adapter_model.safetensors").write_bytes(payload)
    return ckpt_dir


# --------------------------------------------------------------------------- paths


def test_screen_dir_within_model_is_self():
    param_dir = Path("/tmp/out/teachers/owl/data-seed1/param")
    assert _screen_dir(param_dir, None) == param_dir / "steerability" / "self"


def test_screen_dir_cross_seed_is_namespaced_by_target_label():
    param_dir = Path("/tmp/out/teachers/owl/data-seed1/param")
    target = PythiaModelSpec(size="160m", seed=3, variant="weight")
    assert (
        _screen_dir(param_dir, target)
        == param_dir / "steerability" / "steer_weight-seed3"
    )
    # Never collides with the within-model cell.
    assert _screen_dir(param_dir, target) != _screen_dir(param_dir, None)


def test_seed_and_aggregate_file_names():
    param_dir = Path("/tmp/out/teachers/owl/data-seed1/param")
    assert _seed_file(param_dir, None, 2) == param_dir / "steerability" / "self" / "ft2.json"
    assert (
        _aggregate_file(param_dir, None)
        == param_dir / "steerability" / "self" / "aggregate.json"
    )
    target = PythiaModelSpec(size="160m", seed=3, variant="weight")
    assert (
        _seed_file(param_dir, target, 7)
        == param_dir / "steerability" / "steer_weight-seed3" / "ft7.json"
    )


# ------------------------------------------------------------------- seed cache key


def test_seed_cache_key_differs_by_seed(tmp_path):
    a = _write_adapter(tmp_path / "ftseed1" / "checkpoints", b"seed-1-weights")
    b = _write_adapter(tmp_path / "ftseed2" / "checkpoints", b"seed-2-weights")
    cfg = _cfg(None)
    assert _seed_cache_key(cfg, a) != _seed_cache_key(cfg, b)
    assert _seed_cache_key(cfg, a)["adapter_id"] != _seed_cache_key(cfg, b)["adapter_id"]


def test_seed_cache_key_differs_by_steer_target(tmp_path):
    ckpt = _write_adapter(tmp_path / "ftseed1" / "checkpoints", b"weights")
    within = _seed_cache_key(_cfg(None), ckpt)
    cross = _seed_cache_key(
        _cfg(PythiaModelSpec(size="160m", seed=2, variant="data")), ckpt
    )
    assert within != cross
    # Within-model params OMIT the steer keys entirely (byte-identical to the pre-cross-seed
    # schema); cross-seed adds them. Same adapter, so only params differ.
    assert "steer_model" not in within["params"]
    assert "steer_repo" not in within["params"]
    assert cross["params"]["steer_model"] == "data-seed2"
    assert within["adapter_id"] == cross["adapter_id"]

    other = _seed_cache_key(
        _cfg(PythiaModelSpec(size="160m", seed=3, variant="weight")), ckpt
    )
    assert cross != other
    assert cross["params"]["steer_repo"] != other["params"]["steer_repo"]


def test_seed_cache_key_differs_by_params(tmp_path):
    ckpt = _write_adapter(tmp_path / "ftseed1" / "checkpoints", b"weights")
    base = _cfg(None)
    changed = replace(base, val_set="test")  # a different (registered) held-out set
    assert _seed_cache_key(base, ckpt) != _seed_cache_key(changed, ckpt)
    assert _seed_cache_key(base, ckpt)["adapter_id"] == _seed_cache_key(changed, ckpt)["adapter_id"]


# --------------------------------------------------------------------- aggregate_dir


# A representative post-audit params block: aggregate_dir now REQUIRES every seed file to carry
# ``seed_cache_key.params`` and refuses to average files that disagree, so the synthetic docs
# stamp a shared one. Override it to exercise the mixed/mismatch guards.
_STUB_PARAMS = {
    "target_animal": "owl",
    "source_repo": "EleutherAI/pythia-160m-data-seed1",
    "extract_context": "numbers",
    "n_extract_prompts": 1024,
    "number_config": "configs/experiment/number_generation.yaml",
    "number_config_sha256": "deadbeefdeadbeef",
    "pos_set": "favorite", "val_set": "validation",
    "neg_set": "dislike", "off_set": "off_topic",
    "pos_hash": "aaaa", "val_hash": "bbbb", "neg_hash": "cccc", "off_hash": "dddd",
    "layer_sweep": [["L0", [0]]],
    "alphas": [1.0],
    "positions": "broadcast",
    "neg_thresh": 0.1,
    "off_thresh": 0.1,
    "steer_mode": "add",
    "steer_norm": "raw",
}


def _seed_doc(
    *,
    seed: int,
    base: dict[str, float],
    teacher: dict[str, float],
    peaks: dict[str, dict],
    steer_model: str | None = None,
    base_source_val_p: float | None = None,
    params: dict | None = None,
) -> dict:
    """A minimal ft{seed}.json carrying exactly the fields aggregate_dir reads."""

    def _scores(vals: dict[str, float]) -> dict[str, dict]:
        return {k: {"p_target": v, "rank_mean": 1.0, "rank1_frac": 1.0} for k, v in vals.items()}

    doc = {
        "target_animal": "owl",
        "base_model": "pythia-160m-data-seed1",
        "steer_model": steer_model,
        "param_tag": "param",
        "ft_seed": seed,
        "val_set": "validation",
        "extraction": {"context": "numbers", "n_prompts": 1024, "pooling": "last"},
        "sweep_grid": {
            "pos_set": "favorite",
            "neg_set": "dislike",
            "off_set": "off_topic",
        },
        "seed_cache_key": {
            "adapter_id": f"adapter-{seed}",
            "params": params if params is not None else dict(_STUB_PARAMS),
        },
        "base": _scores(base),
        "teacher": _scores(teacher),
        "peak_gated_neg_off": peaks["gated_neg_off"],
        "peak_gated_off": peaks["gated_off"],
        "peak_ungated": peaks["ungated"],
        "sweep": [],
    }
    if base_source_val_p is not None:
        doc["base_source_val_p"] = base_source_val_p
    return doc


def _peak(val_p, delta, rec, pos_p, neg_p, off_p, gated=True):
    return {
        "val_p": val_p,
        "delta_val_p": delta,
        "recovery_frac": rec,
        "pos_p": pos_p,
        "neg_p": neg_p,
        "off_p": off_p,
        "layers": "L3",
        "alpha": 2.0,
        "gated": gated,
    }


def test_aggregate_dir_reduces_synthetic_seeds(tmp_path):
    screen_dir = tmp_path / "steerability" / "self"
    screen_dir.mkdir(parents=True)
    base = {"favorite": 0.1, "validation": 0.2, "dislike": 0.05, "off_topic": 0.03}
    # Three seeds; teacher val = [0.4, 0.6, 0.5] -> mean 0.5. ungated peak carries a NaN
    # recovery on seed 2 so _nanmean must skip it: (0.5 + 0.7) / 2 = 0.6.
    docs = [
        _seed_doc(
            seed=2,
            base=base,
            teacher={"favorite": 0.3, "validation": 0.4, "dislike": 0.01, "off_topic": 0.02},
            peaks={
                "gated_neg_off": _peak(0.30, 0.10, 0.4, 0.5, 0.02, 0.01, gated=True),
                "gated_off": _peak(0.32, 0.12, 0.5, 0.5, 0.02, 0.01, gated=True),
                "ungated": _peak(0.35, 0.15, 0.5, 0.5, 0.20, 0.02, gated=True),
            },
        ),
        _seed_doc(
            seed=1,  # deliberately out of order on disk
            base=base,
            teacher={"favorite": 0.3, "validation": 0.6, "dislike": 0.03, "off_topic": 0.04},
            peaks={
                "gated_neg_off": _peak(0.40, 0.20, 0.6, 0.6, 0.02, 0.01, gated=False),
                "gated_off": _peak(0.44, 0.24, 0.7, 0.6, 0.02, 0.01, gated=True),
                "ungated": _peak(0.45, 0.25, float("nan"), 0.6, 0.30, 0.02, gated=True),
            },
        ),
        _seed_doc(
            seed=3,
            base=base,
            teacher={"favorite": 0.3, "validation": 0.5, "dislike": 0.02, "off_topic": 0.03},
            peaks={
                "gated_neg_off": _peak(0.35, 0.15, 0.5, 0.55, 0.02, 0.01, gated=True),
                "gated_off": _peak(0.38, 0.18, 0.6, 0.55, 0.02, 0.01, gated=False),
                "ungated": _peak(0.40, 0.20, 0.7, 0.55, 0.25, 0.02, gated=True),
            },
        ),
    ]
    for d in docs:
        (screen_dir / f"ft{d['ft_seed']}.json").write_text(json.dumps(d))

    agg = aggregate_dir(screen_dir)
    assert agg is not None
    # Seeds sorted, count correct.
    assert agg["seeds"] == [1, 2, 3]
    assert agg["n_seeds"] == 3
    # Seed-independent base scores taken straight through.
    assert agg["base_pos_p"] == pytest.approx(0.1)
    assert agg["base_val_p"] == pytest.approx(0.2)
    assert agg["base_neg_p"] == pytest.approx(0.05)
    assert agg["base_off_p"] == pytest.approx(0.03)
    # Teacher means.
    assert agg["teacher_val_p_mean"] == pytest.approx(0.5)
    assert agg["teacher_pos_p_mean"] == pytest.approx(0.3)
    assert agg["teacher_neg_p_mean"] == pytest.approx(0.02)
    assert agg["teacher_off_p_mean"] == pytest.approx(0.03)
    # ungated peak: means over the three seeds; recovery skips the NaN.
    ung = agg["peaks"]["ungated"]
    assert ung["val_p_mean"] == pytest.approx((0.35 + 0.45 + 0.40) / 3)
    assert ung["delta_val_p_mean"] == pytest.approx((0.15 + 0.25 + 0.20) / 3)
    assert ung["recovery_frac_mean"] == pytest.approx((0.5 + 0.7) / 2)
    assert "gated_frac" not in ung  # ungated block carries no gate fraction
    # gated_neg_off: gate held on seeds 2 and 3 only -> 2/3.
    assert agg["peaks"]["gated_neg_off"]["gated_frac"] == pytest.approx(2 / 3)
    # gated_off: gate held on seeds 1 and 2 -> 2/3.
    assert agg["peaks"]["gated_off"]["gated_frac"] == pytest.approx(2 / 3)
    # Aggregate file was written and round-trips to the returned dict.
    assert json.loads((screen_dir / "aggregate.json").read_text()) == agg


def test_aggregate_dir_returns_none_when_empty(tmp_path):
    screen_dir = tmp_path / "steerability" / "self"
    screen_dir.mkdir(parents=True)
    assert aggregate_dir(screen_dir) is None


def test_aggregate_dir_asserts_on_inconsistent_base(tmp_path):
    screen_dir = tmp_path / "steerability" / "self"
    screen_dir.mkdir(parents=True)
    peaks = {
        "gated_neg_off": _peak(0.3, 0.1, 0.5, 0.5, 0.02, 0.01),
        "gated_off": _peak(0.3, 0.1, 0.5, 0.5, 0.02, 0.01),
        "ungated": _peak(0.3, 0.1, 0.5, 0.5, 0.02, 0.01),
    }
    teacher = {"favorite": 0.3, "validation": 0.4, "dislike": 0.01, "off_topic": 0.02}
    d1 = _seed_doc(
        seed=1,
        base={"favorite": 0.1, "validation": 0.2, "dislike": 0.05, "off_topic": 0.03},
        teacher=teacher,
        peaks=peaks,
    )
    d2 = _seed_doc(
        seed=2,
        base={"favorite": 0.1, "validation": 0.9, "dislike": 0.05, "off_topic": 0.03},  # mismatch
        teacher=teacher,
        peaks=peaks,
    )
    (screen_dir / "ft1.json").write_text(json.dumps(d1))
    (screen_dir / "ft2.json").write_text(json.dumps(d2))
    with pytest.raises(AssertionError):
        aggregate_dir(screen_dir)


def test_aggregate_dir_cross_seed_carries_base_source_val_p(tmp_path):
    screen_dir = tmp_path / "steerability" / "steer_weight-seed2"
    screen_dir.mkdir(parents=True)
    peaks = {
        "gated_neg_off": _peak(0.3, 0.1, 0.5, 0.5, 0.02, 0.01),
        "gated_off": _peak(0.3, 0.1, 0.5, 0.5, 0.02, 0.01),
        "ungated": _peak(0.3, 0.1, 0.5, 0.5, 0.02, 0.01),
    }
    base = {"favorite": 0.1, "validation": 0.2, "dislike": 0.05, "off_topic": 0.03}
    for seed in (1, 2):
        d = _seed_doc(
            seed=seed,
            base=base,
            teacher={"favorite": 0.3, "validation": 0.4, "dislike": 0.01, "off_topic": 0.02},
            peaks=peaks,
            steer_model="weight-seed2",
            base_source_val_p=0.15,
        )
        (screen_dir / f"ft{seed}.json").write_text(json.dumps(d))
    agg = aggregate_dir(screen_dir)
    assert agg is not None
    assert agg["steer_model"] == "weight-seed2"
    assert agg["base_source_val_p"] == pytest.approx(0.15)


def _two_consistent_seed_docs(params: dict | None = None) -> list[dict]:
    """Two seed docs with matching base/params (the happy path for aggregate_dir)."""
    peaks = {
        "gated_neg_off": _peak(0.3, 0.1, 0.5, 0.5, 0.02, 0.01),
        "gated_off": _peak(0.3, 0.1, 0.5, 0.5, 0.02, 0.01),
        "ungated": _peak(0.3, 0.1, 0.5, 0.5, 0.02, 0.01),
    }
    base = {"favorite": 0.1, "validation": 0.2, "dislike": 0.05, "off_topic": 0.03}
    teacher = {"favorite": 0.3, "validation": 0.4, "dislike": 0.01, "off_topic": 0.02}
    return [
        _seed_doc(seed=s, base=base, teacher=teacher, peaks=peaks, params=params)
        for s in (1, 2)
    ]


def test_aggregate_dir_stamps_shared_params(tmp_path):
    screen_dir = tmp_path / "steerability" / "self"
    screen_dir.mkdir(parents=True)
    for d in _two_consistent_seed_docs():
        (screen_dir / f"ft{d['ft_seed']}.json").write_text(json.dumps(d))
    agg = aggregate_dir(screen_dir)
    assert agg is not None
    assert agg["params"] == _STUB_PARAMS
    # Round-trips to disk (and the stamp is persisted, not just returned).
    assert json.loads((screen_dir / "aggregate.json").read_text())["params"] == _STUB_PARAMS


def test_aggregate_dir_raises_on_mixed_params_across_seeds(tmp_path):
    screen_dir = tmp_path / "steerability" / "self"
    screen_dir.mkdir(parents=True)
    d1, d2 = _two_consistent_seed_docs()
    d2["seed_cache_key"]["params"] = {**_STUB_PARAMS, "off_thresh": 0.5}  # one seed disagrees
    (screen_dir / "ft1.json").write_text(json.dumps(d1))
    (screen_dir / "ft2.json").write_text(json.dumps(d2))
    with pytest.raises(ValueError, match="different screen params"):
        aggregate_dir(screen_dir)


def test_aggregate_dir_raises_on_current_params_mismatch(tmp_path):
    screen_dir = tmp_path / "steerability" / "self"
    screen_dir.mkdir(parents=True)
    for d in _two_consistent_seed_docs():
        (screen_dir / f"ft{d['ft_seed']}.json").write_text(json.dumps(d))
    # Seed files agree with each other but NOT with the current request.
    with pytest.raises(ValueError, match="differing from the current request"):
        aggregate_dir(screen_dir, {**_STUB_PARAMS, "neg_thresh": 0.9})


def test_aggregate_dir_raises_when_seed_file_lacks_params(tmp_path):
    screen_dir = tmp_path / "steerability" / "self"
    screen_dir.mkdir(parents=True)
    d1, d2 = _two_consistent_seed_docs()
    del d2["seed_cache_key"]  # a pre-schema file
    (screen_dir / "ft1.json").write_text(json.dumps(d1))
    (screen_dir / "ft2.json").write_text(json.dumps(d2))
    with pytest.raises(ValueError, match="no seed_cache_key.params"):
        aggregate_dir(screen_dir)


def test_aggregate_dir_atomic_write_leaves_no_tmp(tmp_path):
    screen_dir = tmp_path / "steerability" / "self"
    screen_dir.mkdir(parents=True)
    for d in _two_consistent_seed_docs():
        (screen_dir / f"ft{d['ft_seed']}.json").write_text(json.dumps(d))
    agg = aggregate_dir(screen_dir)
    # The final file is complete/correct and no temp file was left behind.
    assert json.loads((screen_dir / "aggregate.json").read_text()) == agg
    assert list(screen_dir.glob("*.tmp*")) == []


# ------------------------------------------------------------------ _screen_params keys


def test_screen_params_carries_post_audit_keys():
    params = _screen_params(_cfg(None))
    for key in (
        "target_animal", "source_repo", "number_config_sha256",
        "pos_hash", "val_hash", "neg_hash", "off_hash", "steer_mode", "steer_norm",
    ):
        assert key in params, key
    assert params["target_animal"] == "owl"
    assert params["source_repo"] == "EleutherAI/pythia-160m-data-seed1"
    assert params["steer_mode"] == "add"
    assert params["steer_norm"] == "raw"
    # Within-model omits the steer identity; cross-seed adds it via model_ref.
    assert "steer_repo" not in params
    cross = _screen_params(_cfg(PythiaModelSpec(size="160m", seed=2, variant="weight")))
    assert cross["steer_repo"] == "EleutherAI/pythia-160m-weight-seed2"


# ------------------------------------------------------------------ partition (gate) logic


def test_partition_seeds_reuses_match_and_recomputes_corrupt(tmp_path):
    cfg = replace(_cfg(None), param_dir=tmp_path)
    ck1 = _write_adapter(tmp_path / "ftseed1" / "checkpoints", b"w1")
    ck2 = _write_adapter(tmp_path / "ftseed2" / "checkpoints", b"w2")
    ck3 = _write_adapter(tmp_path / "ftseed3" / "checkpoints", b"w3")
    ftseeds = [(1, ck1), (2, ck2), (3, ck3)]
    seed_keys = {s: _seed_cache_key(cfg, c) for s, c in ftseeds}
    sdir = _screen_dir(tmp_path, None)
    sdir.mkdir(parents=True)
    # seed 1: matching file -> reused
    _seed_file(tmp_path, None, 1).write_text(json.dumps({"seed_cache_key": seed_keys[1]}))
    # seed 2: corrupt file -> recompute (must NOT crash)
    _seed_file(tmp_path, None, 2).write_text("{not valid json")
    # seed 3: no file -> recompute
    reused, to_compute = _partition_seeds(cfg, ftseeds, seed_keys)
    assert reused == [1]
    assert sorted(s for s, _ in to_compute) == [2, 3]


# --------------------------------------------------------------------- dim guard


class _DummyConfig:
    def __init__(self, hidden_size: int, num_hidden_layers: int):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers


class _DummyModel:
    """Minimal stand-in exposing only the ``.config`` fields the guard reads."""

    def __init__(self, hidden_size: int, num_hidden_layers: int):
        self.config = _DummyConfig(hidden_size, num_hidden_layers)


def test_assert_compatible_passes_on_matching_dims():
    src = _DummyModel(768, 12)
    tgt = _DummyModel(768, 12)
    _assert_compatible(src, tgt)  # must not raise


@pytest.mark.parametrize(
    ("src_dims", "tgt_dims"),
    [
        ((768, 12), (1024, 12)),  # hidden size mismatch
        ((768, 12), (768, 24)),   # block count mismatch
    ],
)
def test_assert_compatible_raises_on_mismatch(src_dims, tgt_dims):
    src = _DummyModel(*src_dims)
    tgt = _DummyModel(*tgt_dims)
    with pytest.raises(AssertionError):
        _assert_compatible(src, tgt)
