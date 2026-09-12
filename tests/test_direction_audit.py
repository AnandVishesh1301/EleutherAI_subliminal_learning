"""Check normalization and observational joins, which determine the claims."""
import numpy as np
import pandas as pd
import pytest

from docs.paper.prepare_direction_audit import norm_coordinates, paired_rows, unique_scan


def test_equal_spectrum_ladders_match_only_after_energy_conversion():
    k = np.array([1, 4, 16])
    e_spec = np.array([.5, .5, .5])
    fs, ss = norm_coordinates(k, e_spec, "spec")
    ff, sf = norm_coordinates(k, e_spec * np.sqrt(k), "fro")
    np.testing.assert_allclose(fs, ff)
    np.testing.assert_allclose(ss, sf)
    np.testing.assert_allclose(fs / ss, np.sqrt(k))
    with pytest.raises(ValueError):
        norm_coordinates(k, e_spec, "typo")


def test_pairing_requires_actual_same_receiver_complete_interventions():
    rows = [dict(run="a", variant=v, delta_ce=x, base="r@1", pool_sha="p")
            for v, x in [("orig", 8.), ("rand0", 1.)]]
    assert paired_rows(rows, expected=1).loc["a", "rand0"] == 1.
    with pytest.raises(ValueError, match="mixed base"):
        paired_rows([rows[0], {**rows[1], "base": "r@2"}])
    with pytest.raises(ValueError, match="incomplete"):
        paired_rows(rows[:1])
    with pytest.raises(ValueError, match="duplicate"):
        paired_rows(rows + rows)


def test_native_scan_repeats_do_not_double_population():
    row = dict(arm="traitfree", leaf="r/seed1", delta_ce=1., intruders=3)
    assert len(unique_scan(pd.DataFrame([row, row]))) == 1
    with pytest.raises(ValueError, match="conflicting"):
        unique_scan(pd.DataFrame([row, {**row, "delta_ce": 2.}]))
