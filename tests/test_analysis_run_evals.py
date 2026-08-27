"""Run-dir eval-contract loader: leaf discovery, both KL sources, the diverged convention."""
from __future__ import annotations

import json
import math

import pandas as pd
import pytest

from subliminal_learning.analysis import run_evals
from subliminal_learning.analysis.run_evals import load_run_evals


def _write(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))


@pytest.fixture()
def tree(tmp_path):
    """output_root with three cells: a healthy leaf, a diverged leaf, a nonfinite endpoint."""
    lineage = tmp_path / "students" / "dog" / "lineage"

    ok = lineage / "ep20_lr1e-4_r16a32" / "ftseed1"
    _write(ok / "preference_eval.json", {"sets": {"test": {"p_target": 0.4},
                                                  "off_topic": {"p_target": 0.02}}})
    _write(ok / "eval" / "pile_kl_traj.json",
           {"points": [{"step": 10, "kl_base_adapted": 0.5},
                       {"step": 30, "kl_base_adapted": 2.5},
                       {"step": 20, "kl_base_adapted": 1.5}]})

    div = lineage / "ep20_lr5e-4_r16a32" / "ftseed1"
    div.mkdir(parents=True)
    (div / "train_diverged.json").write_text("{}")

    nf = lineage / "ep20_lr2e-4_r16a32" / "ftseed1"
    _write(nf / "preference_eval.json", {"sets": {"test": {"p_target": 0.3}}})
    _write(nf / "eval" / "pile_kl_traj.json",
           {"points": [{"step": 10, "kl_base_adapted": 1.0},
                       {"step": 20, "nonfinite": True}]})

    # a cell whose leaf never landed
    (lineage / "ep20_lr3e-4_r16a32").mkdir(parents=True)
    return tmp_path, lineage


def test_traj_end_source(tree):
    root, lineage = tree
    df = load_run_evals(root, lineage, kl_source="traj_end").set_index("student_tag")

    ok = df.loc["ep20_lr1e-4_r16a32"]
    assert ok.landed and not ok.diverged
    assert ok.kl == 2.5              # endpoint = max step, not last listed point
    assert ok.test == 0.4
    assert (ok["lr"], ok["rank"], ok["alpha"], ok["epochs"]) == (1e-4, 16, 32, 20)

    div = df.loc["ep20_lr5e-4_r16a32"]
    assert div.diverged and math.isinf(div.kl) and div.test == 0.0

    nf = df.loc["ep20_lr2e-4_r16a32"]
    assert nf.diverged and math.isinf(nf.kl) and nf.test == 0.0


def test_missing_leaf_and_seed_selection(tree):
    root, lineage = tree
    df = load_run_evals(root, lineage, cells=["ep20_lr3e-4_r16a32"], seeds=[1, 2],
                        kl_source="traj_end")
    assert len(df) == 2
    assert not df.landed.any()
    assert df.kl.isna().all() and df.test.isna().all()


def test_prompt_set_parameter(tree):
    root, lineage = tree
    df = load_run_evals(root, lineage, cells=["ep20_lr1e-4_r16a32"],
                        prompt_set="off_topic", kl_source="traj_end")
    assert df.test.iloc[0] == 0.02


def test_probe_source(tree, monkeypatch):
    root, lineage = tree
    seen = {}

    def fake_cache(output_root, run_paths, *a, **kw):
        seen["run_paths"] = list(run_paths)
        return pd.DataFrame({"run_path": seen["run_paths"], "kl": [7.0] * len(seen["run_paths"])})

    monkeypatch.setattr("subliminal_learning.analysis.probe_cache.load_probe_cache",
                        fake_cache)
    df = load_run_evals(root, lineage, kl_source="probe").set_index("student_tag")
    # one batched call over relative leaf paths, not one read per leaf
    assert all(not p.startswith("/") for p in seen["run_paths"])
    assert df.loc["ep20_lr1e-4_r16a32"].kl == 7.0
    # the diverged convention still overrides the cached value
    assert math.isinf(df.loc["ep20_lr5e-4_r16a32"].kl)


def test_probe_source_pool_version_passthrough(tree, monkeypatch):
    root, lineage = tree
    seen = {}

    def fake_cache(output_root, run_paths, *a, **kw):
        seen["pool_version"] = kw.get("pool_version")
        return pd.DataFrame({"run_path": [], "kl": []})

    monkeypatch.setattr("subliminal_learning.analysis.probe_cache.load_probe_cache", fake_cache)
    load_run_evals(root, lineage, kl_source="probe")
    assert seen["pool_version"] == "pp1"          # default preserves current behaviour
    load_run_evals(root, lineage, kl_source="probe", pool_version=None)
    assert seen["pool_version"] is None


def test_kl_source_none(tree, monkeypatch):
    root, lineage = tree

    def boom(*a, **kw):                            # "none" must not touch the probe cache
        raise AssertionError("probe cache read under kl_source='none'")

    monkeypatch.setattr("subliminal_learning.analysis.probe_cache.load_probe_cache", boom)
    df = load_run_evals(root, lineage, kl_source="none").set_index("student_tag")
    assert df.loc["ep20_lr1e-4_r16a32"].test == 0.4          # trait readout still lands
    assert math.isnan(df.loc["ep20_lr1e-4_r16a32"].kl)
    # the nonfinite-endpoint run is NOT flagged here: no trajectory is read
    assert math.isnan(df.loc["ep20_lr2e-4_r16a32"].kl)
    # only the diverged convention writes a kl
    assert math.isinf(df.loc["ep20_lr5e-4_r16a32"].kl)
    assert df.loc["ep20_lr5e-4_r16a32"].test == 0.0


def test_bad_kl_source(tree):
    root, lineage = tree
    with pytest.raises(ValueError):
        load_run_evals(root, lineage, kl_source="nope")


def test_traj_endpoint_missing_file(tmp_path):
    kl, nonfinite = run_evals._traj_endpoint(tmp_path)
    assert math.isnan(kl) and nonfinite is False


def test_peak_summary():
    df = pd.DataFrame({
        "student_tag": [f"cell{i}" for i in range(20)],
        "seed": [1] * 20,
        "transfer": [float(i) for i in range(20)] + [],
        "K_rel": [0.1 * (i + 1) for i in range(20)],
    })
    df.loc[0, "transfer"] = float("nan")          # missing readouts are dropped, not zeroed
    out = run_evals.peak_summary(df)
    assert out["n"] == 19
    assert out["argmax_cell"] == "cell19" and out["argmax_seed"] == 1
    assert out["argmax_transfer"] == 19.0 and out["argmax_K_rel"] == pytest.approx(2.0)
    # top decile floors at 3 rows: 19 // 10 == 1 -> the three best runs
    assert out["top_decile_n"] == 3 and out["top_decile_median"] == 18.0
    assert out["top_decile_K_rel_geomean"] == pytest.approx(
        math.exp(sum(math.log(v) for v in (2.0, 1.9, 1.8)) / 3))
    assert out["top_decile_K_rel_range"] == (1.8, 2.0)
    assert out["top_decile_K_rel_iqr"] == (1.85, 1.95)
    assert out["frac_above_prior"] == pytest.approx(18 / 19)
    assert out["frac_above_2x"] == pytest.approx(17 / 19)


def test_peak_summary_empty():
    assert run_evals.peak_summary(pd.DataFrame({"transfer": [], "student_tag": [],
                                                "K_rel": []})) == {"n": 0}


def test_seeds_generator_is_not_exhausted_after_the_first_cell(tree):
    """`seeds` is consumed per cell, so it must be materialized once, not iterated per cell."""
    root, lineage = tree
    cells = ["ep20_lr1e-4_r16a32", "ep20_lr2e-4_r16a32"]
    df = load_run_evals(root, lineage, cells=cells, seeds=(s for s in (1,)),
                        kl_source="none")
    assert sorted(df.student_tag) == sorted(cells)
    assert list(df.ftseed) == [1, 1]
