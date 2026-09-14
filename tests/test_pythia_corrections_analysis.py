import copy

import pandas as pd
import pytest

from docs.paper.analyze_corrected_directions import IDENTITY, validate_injection
from docs.paper.analyze_pythia_corrections import attach_pairs, validate_surgery


def fixture(n_super=1):
    run = "/fake/run"
    identity = dict.fromkeys(IDENTITY, "same")
    identity.update(surgery_construction_version="position_surgery_v3", dose_lambda=1,
                    dose_manifest_sha256=None, base_revision=None, seed=0,
                    animal="dog", pool_version="pp1")
    rows = [dict(identity, run=run, variant=v, delta_ce=ce, p_target=.1,
                 n_super=n_super, n_dirs=2)
            for v, ce in [("orig", 10.), ("rand0", 2.), ("sub", 3.), ("super", 8.)]
            if not (v == "super" and n_super == 0) and not (v == "sub" and n_super == 2)]
    job = dict(runs=[run], family="selectors", args=["--variants", "orig,rand0",
               "--animal", "dog", "--seed", "0", "--pool", "pp1"])
    inputs = {run: dict(weights_sha256="same", base_ref="same",
                        metadata_sha256={run + "/checkpoints/adapter_config.json": "same"})}
    return rows, job, inputs


@pytest.mark.parametrize("n_super", [0, 1, 2])
def test_applicable_selectors(n_super):
    rows, job, inputs = fixture(n_super)
    assert len(validate_surgery(rows, job, inputs)) == len(rows)


def test_missing_nonempty_selector_rejected():
    rows, job, inputs = fixture()
    with pytest.raises(ValueError, match="variants"):
        validate_surgery(rows[:-1], job, inputs)


@pytest.mark.parametrize("field,value", [("adapter_sha256", "bad"), ("base_ref", "bad"),
                                        ("dose_lambda", .5)])
def test_frozen_identity_rejected(field, value):
    rows, job, inputs = fixture()
    for row in rows:
        row[field] = value
    with pytest.raises(ValueError, match="frozen"):
        validate_surgery(rows, job, inputs)


def test_no_selection_on_new_recovery():
    rows, job, inputs = fixture()
    d = validate_surgery(rows, job, inputs)
    pop = pd.DataFrame([dict(run="/fake/run", arm="strip", cut=5)])
    paired = attach_pairs(d, pop)
    assert len(paired) == 4
    assert set(paired.original) == {10.}
    assert paired.set_index("variant").loc["super", "paired_change"] == -2.
    with pytest.raises(ValueError, match="absent"):
        attach_pairs(d, pop.iloc[:0])


def test_spectral_coordinates_grid_and_rng():
    job = dict(args=["--repo", "test", "--match", "spec", "--seed", "0", "--pool", "pp1",
                     "--alignments", "g", "--k-ladder", "1,4,16,32", "--e-ladder", "0.25,0.5,1,2"])
    raw = dict(repo="test", match="spec", seed=0, pool_version="pp1", pool_sha="sha",
               ce_base=2., intruder_stride=1, rng_scheme="nested_signed_qr_v2",
               theta_bars={str(i): 1 for i in range(48)},
               results=[dict(align="g", k=k, E=e, F_over_theta=e*k**.5,
                             S_over_theta=e, delta_ce=1., intruder_mods="0/48")
                        for k in (1, 4, 16, 32) for e in (.25, .5, 1, 2)])
    assert len(validate_injection(raw, job)) == 16
    for field, value, match in [("rng_scheme", "old", "RNG"),
                                 ("results", raw["results"][:-1], "grid")]:
        changed = copy.deepcopy(raw)
        changed[field] = value
        with pytest.raises(ValueError, match=match):
            validate_injection(changed, job)
    raw["results"][-1]["F_over_theta"] = 2.
    with pytest.raises(ValueError, match="norm coordinates"):
        validate_injection(raw, job)
