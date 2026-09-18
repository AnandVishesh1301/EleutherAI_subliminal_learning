"""Probe conventions: pool spec stability, record currency, manifest identity."""
import json

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


def _mk_base_run(tmp_path, repo, name="ftseed1"):
    rd = tmp_path / name
    (rd / "eval").mkdir(parents=True)
    (rd / "train_manifest.json").write_text(json.dumps({
        "base_model_repo": repo, "student_param_tag": "ep20_lr1e-4_r8a32",
        "filtered_sha256": "abc", "n_train_steps": 25000}))
    return rd


def test_pool_spec_by_version_is_the_only_cross_corpus_hatch():
    assert pio.probe_pool_spec_for_version("pp1") is pio.PROBE_POOL_PP1
    assert pio.probe_pool_spec_for_version("cp1") is pio.PROBE_POOL_CP1
    assert pio.probe_pool_spec_for_version("dp1") is pio.PROBE_POOL_DP1
    try:
        pio.probe_pool_spec_for_version("zz9")
    except ValueError as exc:
        assert "zz9" in str(exc)
    else:
        raise AssertionError("unknown pool version must raise")
    # the family check of the per-repo resolver is untouched
    dclm = "local/pythia-160m-dclm-seed1234"
    assert pio.probe_pool_spec_for_repo(dclm).version == "cp1"
    assert pio.resolve_probe_pool_spec(dclm, None).version == "cp1"


def test_native_pool_keeps_canonical_path_other_pools_get_a_sidecar(tmp_path):
    pile = _mk_base_run(tmp_path, "local/pythia-160m-pilerepro-seed1234", "pile_run")
    dclm = _mk_base_run(tmp_path, "local/pythia-160m-dclm-seed1234", "dclm_run")
    assert pio.native_probe_pool_version(pile) == "pp1"
    assert pio.native_probe_pool_version(dclm) == "cp1"
    for rd in (pile, dclm):
        assert pio.probe_path(rd).name == "pile_probe.json"
    assert pio.probe_path(pile, "pp1").name == "pile_probe.json"
    assert pio.probe_path(pile, "cp1").name == "pile_probe.cp1.json"
    assert pio.probe_path(dclm, "cp1").name == "pile_probe.json"
    assert pio.probe_path(dclm, "pp1").name == "pile_probe.pp1.json"
    # unknowable base (no manifest) -> a named pool is always a sidecar, never a clobber
    bare = tmp_path / "bare"
    bare.mkdir()
    assert pio.native_probe_pool_version(bare) is None
    assert pio.probe_path(bare).name == "pile_probe.json"
    assert pio.probe_path(bare, "cp1").name == "pile_probe.cp1.json"


def test_two_pools_do_not_ping_pong_overwrite(tmp_path):
    rd = _mk_base_run(tmp_path, "local/pythia-160m-pilerepro-seed1234")
    ref = pio.manifest_ref(rd)
    for pool, sha in (("pp1", "SHA-PP1"), ("cp1", "SHA-CP1")):
        pio.probe_path(rd, pool).write_text(json.dumps({
            "schema_version": pio.SCHEMA_VERSION, "pool_sha": sha,
            "pool_version": pool, "manifest_ref": ref}))
    assert pio.probe_is_current(rd, "SHA-PP1")                  # native, canonical file
    assert pio.probe_is_current(rd, "SHA-PP1", "pp1")
    assert pio.probe_is_current(rd, "SHA-CP1", "cp1")           # sidecar, own currency
    assert not pio.probe_is_current(rd, "SHA-CP1")              # not the native record
    assert not pio.probe_is_current(rd, "SHA-PP1", "cp1")
    assert pio.load_probe(rd, "cp1")["pool_version"] == "cp1"
    assert pio.load_probe(rd)["pool_version"] == "pp1"


def test_record_cache_sidecar_kind_is_parameterized():
    from subliminal_learning.analysis.record_cache import get_kind
    native, side = get_kind("pile_probe"), get_kind("pile_probe:cp1")
    assert native.rel_record == "eval/pile_probe.json"
    assert side.rel_record == "eval/pile_probe.cp1.json"
    assert side.columns == native.columns and side.extract is native.extract
    assert side.cache_rel != native.cache_rel


def test_datadecide_recipe_pools_are_versioned_and_sidecar_addressed(tmp_path):
    """dolma1_7 -> dp1, dclm-baseline -> cp1t-olmo; the other recipe's pool is a sidecar."""
    assert pio.probe_pool_spec_for_version("cp1t-olmo") is pio.PROBE_POOL_CP1T_OLMO
    dolma = _mk_base_run(tmp_path, "allenai/DataDecide-dolma1_7-1B@step69369-seed-default",
                         "dolma_run")
    dclm = _mk_base_run(tmp_path,
                        "allenai/DataDecide-dclm-baseline-1B@step69369-seed-default",
                        "dclm_dd_run")
    assert pio.native_probe_pool_version(dolma) == "dp1"
    assert pio.native_probe_pool_version(dclm) == "cp1t-olmo"
    assert pio.probe_path(dclm).name == "pile_probe.json"
    assert pio.probe_path(dclm, "cp1t-olmo").name == "pile_probe.json"
    assert pio.probe_path(dclm, "dp1").name == "pile_probe.dp1.json"
    assert pio.probe_path(dolma, "cp1t-olmo").name == "pile_probe.cp1t-olmo.json"
