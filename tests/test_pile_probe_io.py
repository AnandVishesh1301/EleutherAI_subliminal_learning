"""Probe conventions: pool spec stability, record currency, manifest identity."""
import json
from pathlib import Path

from subliminal_learning.coherence import probe_io as pio


def _mk_run(tmp_path, steps=25000):
    rd = tmp_path / "ep20_lr1e-4_r8a32" / "ftseed1"
    rd.mkdir(parents=True)
    (rd / "train_manifest.json").write_text(json.dumps({
        "student_param_tag": "ep20_lr1e-4_r8a32", "filtered_sha256": "abc", "n_train_steps": steps,
    }))
    return rd


def test_pp1_spec_frozen():
    # the probe pool recipe is a published convention; changing it must bump `version`
    assert (pio.PROBE_POOL_PP1.n_docs, pio.PROBE_POOL_PP1.max_tokens,
            pio.PROBE_POOL_PP1.seed, pio.PROBE_POOL_PP1.version) == (32, 512, 0, "pp1")
    assert pio.PP1_TOKEN_SHA.startswith("9f509dde")


def test_probe_currency_roundtrip(tmp_path):
    rd = _mk_run(tmp_path)
    assert pio.load_probe(rd) is None
    rec = {"schema_version": pio.SCHEMA_VERSION, "pool_sha": "SHA",
           "manifest_ref": pio.manifest_ref(rd)}
    pio.probe_path(rd).parent.mkdir()
    pio.probe_path(rd).write_text(json.dumps(rec))
    assert pio.probe_is_current(rd, "SHA")
    assert not pio.probe_is_current(rd, "OTHER-SHA")     # pool changed
    # retrain: manifest rewrites -> stamp mismatch -> stale
    (rd / "train_manifest.json").write_text(json.dumps({
        "student_param_tag": "ep20_lr1e-4_r8a32", "filtered_sha256": "abc", "n_train_steps": 50000,
    }))
    assert not pio.probe_is_current(rd, "SHA")


def test_no_manifest_never_current(tmp_path):
    rd = tmp_path / "cell" / "ftseed1"
    rd.mkdir(parents=True)
    assert pio.manifest_ref(rd) is None
    assert not pio.probe_is_current(rd, "SHA")


def test_derive_dead_cut_bimodal_and_unimodal():
    import numpy as np
    rng = np.random.default_rng(0)
    bimodal = np.concatenate([rng.normal(1.0, 0.5, 900).clip(0),
                              rng.normal(16.0, 2.0, 600),
                              rng.uniform(4, 13, 80)])   # transit plateau
    dc = pio.derive_dead_cut(bimodal)
    assert dc is not None
    assert 3.0 < dc.cut < 14.0
    assert dc.valley_lo < dc.cut < dc.valley_hi
    assert dc.n_valley >= 40                       # churn is reported, not hidden
    assert pio.derive_dead_cut(rng.normal(2.0, 1.0, 1000)) is None   # unimodal -> refuse
    assert pio.derive_dead_cut([1.0] * 10) is None                    # too few

    # the gate is COUNT-based (2026-08-20): an alive-diluted pool with a well-populated
    # dead mode (small mass fraction, >= min_mode_runs runs) still derives...
    diluted = np.concatenate([rng.normal(1.0, 0.5, 12000).clip(0), rng.normal(16.0, 2.0, 400)])
    assert pio.derive_dead_cut(diluted) is not None
    # ...while a genuinely sparse dead mode refuses
    sparse = np.concatenate([rng.normal(1.0, 0.5, 12000).clip(0), rng.normal(16.0, 2.0, 120)])
    assert pio.derive_dead_cut(sparse) is None
