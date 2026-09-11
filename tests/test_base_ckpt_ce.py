"""Training-retained readout over a base's checkpoint CE curve."""
import json
import math

import pytest

from subliminal_learning.coherence import base_ckpt_ce as bc


def test_default_steps_grid_and_early():
    s = bc.default_steps(5000, 143_000, include_early=True)
    assert s[0] == 0 and s[-1] == 143_000
    assert set(range(0, 145_000, 5000)) <= set(s)
    assert set(bc.EARLY_STEPS) <= set(s)
    assert s == sorted(set(s))
    # short grid: no final step, no early steps beyond max
    assert bc.default_steps(5000, 50_000, include_early=False) == list(range(0, 50_001, 5000))
    with pytest.raises(ValueError):
        bc.default_steps(0)


CURVE = {0: 10.8, 1000: 6.0, 5000: 4.0, 10000: 3.2, 20000: 3.0, 143000: 2.7}


def test_training_retained_bounds():
    assert bc.training_retained(11.0, CURVE) == 0.0         # worse than step 0
    assert bc.training_retained(10.8, CURVE) == 0.0         # equal to step 0
    assert bc.training_retained(2.7, CURVE) == 143000.0     # exactly final
    assert bc.training_retained(2.5, CURVE) == 143000.0     # better than final
    assert math.isnan(bc.training_retained(float("inf"), CURVE))
    assert math.isnan(bc.training_retained(float("nan"), CURVE))
    with pytest.raises(ValueError):
        bc.training_retained(3.0, {})


def test_training_retained_hits_checkpoints_and_interpolates():
    assert bc.training_retained(4.0, CURVE) == 5000.0
    assert bc.training_retained(3.0, CURVE) == 20000.0
    r = bc.training_retained(3.6, CURVE)             # halfway in CE between 5k and 10k
    assert 5000 < r < 10000
    assert math.isclose(r, math.expm1(0.5 * (math.log1p(5000) + math.log1p(10000))))
    # monotone in ce: lower CE -> more training retained
    ces = [9.0, 5.0, 3.9, 3.5, 3.1, 2.9]
    got = [bc.training_retained(c, CURVE) for c in ces]
    assert got == sorted(got)


def test_training_retained_first_crossing_under_nonmonotone_tail():
    curve = {**CURVE, 130000: 2.75, 135000: 2.71, 140000: 2.74}  # noisy tail
    # 2.73 is first reached at 135000 (2.71), not at 140000 or 143000
    r = bc.training_retained(2.73, curve)
    assert 130000 < r <= 135000


def test_fraction_and_record_roundtrip(tmp_path):
    rec = {"kind": bc.RECORD_KIND, "schema_version": bc.SCHEMA_VERSION,
           "base": "pythia-160m", "steps": {
               "5000": {"ce_pp1": 4.0, "ce_pp1w": 4.1},
               "0": {"ce_pp1": 10.8, "ce_pp1w": 10.8},
               "143000": {"ce_pp1": 2.7, "ce_pp1w": 2.75}}}
    p = bc.record_path(tmp_path, "pythia-160m")
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps(rec))
    curve = bc.load_curve(tmp_path, "original")            # naming-module spelling
    assert list(curve) == [0, 5000, 143000] and curve[5000] == 4.0
    wide = bc.load_curve(tmp_path, "EleutherAI/pythia-160m", pool="pp1w")
    assert wide[143000] == 2.75
    assert bc.training_retained_fraction(2.7, curve) == 1.0
    with pytest.raises(FileNotFoundError):
        bc.load_curve(tmp_path, "data-seed1")
