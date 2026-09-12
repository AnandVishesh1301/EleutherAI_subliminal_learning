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
    derive_peaks,
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
    "activation_basis": "raw_blocks_v2",
}


def _row(pos_p, val_p, neg_p, off_p, layers="L3", alpha=2.0) -> dict:
    """One sweep row, exactly the fields ``derive_peaks`` reads."""
    return {
        "layers": layers,
        "alpha": alpha,
        "pos_p": pos_p,
        "val_p": val_p,
        "neg_p": neg_p,
        "off_p": off_p,
    }


def _stale_peak() -> dict:
    """A deliberately wrong stored peak block: the aggregate must ignore it and re-derive."""
    return {
        **_row(-1.0, -1.0, -1.0, -1.0, layers="STALE", alpha=-1.0),
        "delta_val_p": -1.0,
        "recovery_frac": -1.0,
        "gated": True,
    }


def _seed_doc(
    *,
    seed: int,
    base: dict[str, float],
    teacher: dict[str, float],
    sweep: list[dict],
    steer_model: str | None = None,
    base_source_val_p: float | None = None,
    params: dict | None = None,
) -> dict:
    """A minimal ft{seed}.json carrying exactly the fields aggregate_dir reads.

    The stored peak blocks are stale by construction: ``compute_aggregate`` re-derives every
    peak from ``sweep``, so any test that read them would fail loudly.
    """

    def _scores(vals: dict[str, float]) -> dict[str, dict]:
        return {k: {"p_target": v, "rank_mean": 1.0, "rank1_frac": 1.0} for k, v in vals.items()}

    doc = {
        "target_animal": "owl",
        "base_model": "pythia-160m-data-seed1",
        "steer_model": steer_model,
        "param_tag": "param",
        "ft_seed": seed,
        "val_set": "validation",
        "selection_set": "favorite",
        "extraction": {"context": "numbers", "n_prompts": 1024, "pooling": "last"},
        "sweep_grid": {
            "pos_set": "favorite",
            "neg_set": "dislike",
            "off_set": "off_topic",
            "neg_thresh": 0.1,
            "off_thresh": 0.1,
        },
        "seed_cache_key": {
            "adapter_id": f"adapter-{seed}",
            "params": params if params is not None else dict(_STUB_PARAMS),
        },
        "base": _scores(base),
        "teacher": _scores(teacher),
        "peak_gated_neg_off": _stale_peak(),
        "peak_gated_off": _stale_peak(),
        "peak_ungated": _stale_peak(),
        "sweep": sweep,
    }
    if base_source_val_p is not None:
        doc["base_source_val_p"] = base_source_val_p
    return doc


_BASE = {"favorite": 0.1, "validation": 0.2, "dislike": 0.05, "off_topic": 0.03}


def _teacher(val_p: float) -> dict[str, float]:
    return {"favorite": 0.3, "validation": val_p, "dislike": 0.01, "off_topic": 0.02}


# ------------------------------------------------------------------------- derive_peaks


def test_derive_peaks_selects_on_favorite_not_validation():
    # The max-val_p row (L9) is NOT the max-pos_p row (L3): selection must follow pos_p and
    # then REPORT that row's val_p -- the whole point of the de-biasing.
    sweep = [
        _row(0.60, 0.30, 0.02, 0.01, layers="L3"),
        _row(0.20, 0.95, 0.02, 0.01, layers="L9"),
    ]
    peaks = derive_peaks(
        sweep, neg_thresh=0.1, off_thresh=0.1, base_val_p=0.2, headroom=0.4
    )
    for key in ("peak_gated_neg_off", "peak_gated_off", "peak_ungated"):
        assert peaks[key]["layers"] == "L3"
        assert peaks[key]["val_p"] == pytest.approx(0.30)
        assert peaks[key]["delta_val_p"] == pytest.approx(0.10)
        assert peaks[key]["recovery_frac"] == pytest.approx(0.25)


def test_derive_peaks_gate_fallback_flags_ungated():
    # Nothing clears the neg gate -> fall back to the unconstrained max-pos row, flagged.
    sweep = [
        _row(0.60, 0.30, 0.50, 0.01, layers="L3"),
        _row(0.40, 0.25, 0.02, 0.50, layers="L5"),
    ]
    peaks = derive_peaks(
        sweep, neg_thresh=0.1, off_thresh=0.1, base_val_p=0.2, headroom=0.4
    )
    assert peaks["peak_gated_neg_off"]["gated"] is False
    assert peaks["peak_gated_neg_off"]["layers"] == "L3"
    # The off gate alone is cleared by L3 only.
    assert peaks["peak_gated_off"]["gated"] is True
    assert peaks["peak_gated_off"]["layers"] == "L3"
    assert "gated" not in peaks["peak_ungated"]


@pytest.mark.parametrize("headroom", [0.0, 1e-12, -0.3])
def test_derive_peaks_nonpositive_headroom_gives_nan_recovery(headroom):
    # A teacher with no (or NEGATIVE) lift defines no lift to recover: NaN, never a number
    # whose sign a negative denominator would have flipped.
    sweep = [_row(0.60, 0.30, 0.02, 0.01)]
    peaks = derive_peaks(
        sweep, neg_thresh=0.1, off_thresh=0.1, base_val_p=0.2, headroom=headroom
    )
    rec = peaks["peak_ungated"]["recovery_frac"]
    assert rec != rec  # NaN
    assert peaks["peak_ungated"]["delta_val_p"] == pytest.approx(0.10)


# --------------------------------------------------------------------- aggregate_dir


def test_aggregate_dir_reduces_synthetic_seeds(tmp_path):
    screen_dir = tmp_path / "steerability" / "self"
    screen_dir.mkdir(parents=True)
    # Three seeds, all with base val 0.2. Each sweep carries a high-val decoy that loses on
    # pos_p, so the re-derivation is visibly selection-on-favorite.
    docs = [
        _seed_doc(  # headroom 0.4; neg gate cleared by the second row only
            seed=2,
            base=_BASE,
            teacher=_teacher(0.6),
            sweep=[
                _row(0.50, 0.30, 0.20, 0.01, layers="L3"),
                _row(0.45, 0.32, 0.02, 0.01, layers="L5"),
                _row(0.40, 0.90, 0.50, 0.50, layers="L9"),
            ],
        ),
        _seed_doc(  # headroom 0.2; NO row clears the neg gate -> excluded from that mean
            seed=1,  # deliberately out of order on disk
            base=_BASE,
            teacher=_teacher(0.4),
            sweep=[
                _row(0.60, 0.40, 0.20, 0.02, layers="L3"),
                _row(0.30, 0.95, 0.50, 0.50, layers="L9"),
            ],
        ),
        _seed_doc(  # headroom 0.0 -> NaN recovery on every block
            seed=3,
            base=_BASE,
            teacher=_teacher(0.2),
            sweep=[
                _row(0.55, 0.35, 0.02, 0.01, layers="L3"),
                _row(0.20, 0.99, 0.02, 0.01, layers="L9"),
            ],
        ),
    ]
    for d in docs:
        (screen_dir / f"ft{d['ft_seed']}.json").write_text(json.dumps(d))

    agg = aggregate_dir(screen_dir)
    assert agg is not None
    assert agg["seeds"] == [1, 2, 3]
    assert agg["n_seeds"] == 3
    assert agg["selection_set"] == "favorite"
    assert agg["val_set"] == "validation"
    # Seed-independent base scores taken straight through.
    assert agg["base_pos_p"] == pytest.approx(0.1)
    assert agg["base_val_p"] == pytest.approx(0.2)
    assert agg["base_neg_p"] == pytest.approx(0.05)
    assert agg["base_off_p"] == pytest.approx(0.03)
    # Teacher means.
    assert agg["teacher_val_p_mean"] == pytest.approx((0.6 + 0.4 + 0.2) / 3)
    assert agg["teacher_pos_p_mean"] == pytest.approx(0.3)
    # ungated: max-pos row per seed (0.30 / 0.40 / 0.35), NOT the 0.90/0.95/0.99 decoys.
    ung = agg["peaks"]["ungated"]
    assert ung["val_p_mean"] == pytest.approx((0.30 + 0.40 + 0.35) / 3)
    assert ung["delta_val_p_mean"] == pytest.approx((0.10 + 0.20 + 0.15) / 3)
    # recovery: 0.10/0.4, 0.20/0.2, NaN -> mean over the two defined seeds.
    assert ung["recovery_frac_mean"] == pytest.approx((0.25 + 1.0) / 2)
    assert ung["recovery_frac_n"] == 2
    assert "gated_frac" not in ung  # ungated block carries no gate fraction
    assert "gated_n" not in ung
    # gated_off: every seed's off gate holds at its max-pos row -> same rows as ungated.
    off = agg["peaks"]["gated_off"]
    assert off["gated_frac"] == pytest.approx(1.0)
    assert off["gated_n"] == 3
    assert off["val_p_mean"] == pytest.approx((0.30 + 0.40 + 0.35) / 3)
    # gated_neg_off: seed 1 fell back (no row cleared neg) and is EXCLUDED from the means.
    no = agg["peaks"]["gated_neg_off"]
    assert no["gated_frac"] == pytest.approx(2 / 3)
    assert no["gated_n"] == 2
    assert no["val_p_mean"] == pytest.approx((0.32 + 0.35) / 2)
    assert no["delta_val_p_mean"] == pytest.approx((0.12 + 0.15) / 2)
    assert no["recovery_frac_mean"] == pytest.approx(0.12 / 0.4)  # seed 3 is NaN
    assert no["recovery_frac_n"] == 1
    # Aggregate file was written and round-trips to the returned dict.
    assert json.loads((screen_dir / "aggregate.json").read_text()) == agg


def test_aggregate_dir_ignores_stored_peak_blocks_and_rederives(tmp_path):
    screen_dir = tmp_path / "steerability" / "self"
    screen_dir.mkdir(parents=True)
    sweep = [_row(0.60, 0.30, 0.02, 0.01, layers="L3")]
    for seed in (1, 2):
        d = _seed_doc(seed=seed, base=_BASE, teacher=_teacher(0.4), sweep=sweep)
        # A stale/wrong stored block (here also under the OLD max-val_p rule) must not leak in.
        d["peak_ungated"] = {
            **_row(0.01, 0.99, 0.9, 0.9, layers="OLD"),
            "delta_val_p": 0.79, "recovery_frac": 3.95, "gated": True,
        }
        (screen_dir / f"ft{seed}.json").write_text(json.dumps(d))
    agg = aggregate_dir(screen_dir)
    assert agg is not None
    assert agg["peaks"]["ungated"]["val_p_mean"] == pytest.approx(0.30)
    assert agg["peaks"]["ungated"]["delta_val_p_mean"] == pytest.approx(0.10)
    assert agg["peaks"]["ungated"]["recovery_frac_mean"] == pytest.approx(0.5)


def test_aggregate_dir_gated_block_all_fallback_is_nan_with_zero_n(tmp_path):
    screen_dir = tmp_path / "steerability" / "self"
    screen_dir.mkdir(parents=True)
    # No seed clears the neg gate anywhere: the gated_neg_off means are undefined, not 0.0.
    sweep = [_row(0.60, 0.30, 0.50, 0.01, layers="L3")]
    for seed in (1, 2, 3):
        d = _seed_doc(seed=seed, base=_BASE, teacher=_teacher(0.4), sweep=sweep)
        (screen_dir / f"ft{seed}.json").write_text(json.dumps(d))
    agg = aggregate_dir(screen_dir)
    assert agg is not None
    no = agg["peaks"]["gated_neg_off"]
    assert no["gated_n"] == 0
    assert no["gated_frac"] == pytest.approx(0.0)
    for key in ("val_p_mean", "delta_val_p_mean", "recovery_frac_mean", "pos_p_mean"):
        assert no[key] != no[key], key  # NaN
    assert no["recovery_frac_n"] == 0
    # The ungated block still averages all three seeds.
    assert agg["peaks"]["ungated"]["val_p_mean"] == pytest.approx(0.30)


def test_aggregate_dir_all_nan_recovery_is_nan_with_zero_n(tmp_path):
    screen_dir = tmp_path / "steerability" / "self"
    screen_dir.mkdir(parents=True)
    # Every seed's teacher has zero headroom (teacher val == base val): the recovery mean must
    # be NaN, NOT a bogus 0.0, and the seed count must be 0.
    sweep = [_row(0.60, 0.35, 0.02, 0.01)]
    for seed in (1, 2, 3):
        d = _seed_doc(seed=seed, base=_BASE, teacher=_teacher(0.2), sweep=sweep)
        (screen_dir / f"ft{seed}.json").write_text(json.dumps(d))
    agg = aggregate_dir(screen_dir)
    assert agg is not None
    ung = agg["peaks"]["ungated"]
    assert ung["recovery_frac_mean"] != ung["recovery_frac_mean"]  # NaN
    assert ung["recovery_frac_n"] == 0
    # delta means still average all three seeds (unaffected by the NaN recovery).
    assert ung["delta_val_p_mean"] == pytest.approx(0.15)


def test_aggregate_dir_returns_none_when_empty(tmp_path):
    screen_dir = tmp_path / "steerability" / "self"
    screen_dir.mkdir(parents=True)
    assert aggregate_dir(screen_dir) is None


def test_aggregate_dir_raises_on_inconsistent_base(tmp_path):
    screen_dir = tmp_path / "steerability" / "self"
    screen_dir.mkdir(parents=True)
    sweep = [_row(0.60, 0.30, 0.02, 0.01)]
    d1 = _seed_doc(seed=1, base=_BASE, teacher=_teacher(0.4), sweep=sweep)
    d2 = _seed_doc(
        seed=2,
        base={**_BASE, "validation": 0.9},  # mismatch
        teacher=_teacher(0.4),
        sweep=sweep,
    )
    (screen_dir / "ft1.json").write_text(json.dumps(d1))
    (screen_dir / "ft2.json").write_text(json.dumps(d2))
    with pytest.raises(ValueError, match="base is seed-independent"):
        aggregate_dir(screen_dir)


def test_aggregate_dir_cross_seed_carries_base_source_val_p(tmp_path):
    screen_dir = tmp_path / "steerability" / "steer_weight-seed2"
    screen_dir.mkdir(parents=True)
    sweep = [_row(0.60, 0.30, 0.02, 0.01)]
    for seed in (1, 2):
        d = _seed_doc(
            seed=seed,
            base=_BASE,
            teacher=_teacher(0.4),
            sweep=sweep,
            steer_model="weight-seed2",
            base_source_val_p=0.15,
        )
        (screen_dir / f"ft{seed}.json").write_text(json.dumps(d))
    agg = aggregate_dir(screen_dir)
    assert agg is not None
    assert agg["steer_model"] == "weight-seed2"
    assert agg["base_source_val_p"] == pytest.approx(0.15)
    # headroom is measured against the SOURCE control (0.4 - 0.15), delta against the target's.
    assert agg["peaks"]["ungated"]["recovery_frac_mean"] == pytest.approx(0.10 / 0.25)


def _two_consistent_seed_docs(params: dict | None = None) -> list[dict]:
    """Two seed docs with matching base/params (the happy path for aggregate_dir)."""
    sweep = [_row(0.60, 0.30, 0.02, 0.01)]
    return [
        _seed_doc(seed=s, base=_BASE, teacher=_teacher(0.4), sweep=sweep, params=params)
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
        "activation_basis",
    ):
        assert key in params, key
    assert params["target_animal"] == "owl"
    assert params["source_repo"] == "EleutherAI/pythia-160m-data-seed1"
    assert params["steer_mode"] == "add"
    assert params["steer_norm"] == "raw"
    # Legacy ft*.json (pre-raw_blocks_v2, output_hidden_states extraction) carry no
    # activation_basis at all, so this key's presence alone invalidates every one of them.
    assert params["activation_basis"] == "raw_blocks_v2"
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
