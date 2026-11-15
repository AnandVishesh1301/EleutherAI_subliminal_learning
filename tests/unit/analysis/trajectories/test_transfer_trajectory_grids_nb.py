"""The transfer-trajectory-grid notebook's endpoint / exclusion labelling functions
(`notebooks.builders.build_transfer_trajectory_grids_nb`), executed from the builder's cell source
on fixture trees (no model loads)."""

import ast

import json

from pathlib import Path

import numpy as np

import pandas as pd



from notebooks.builders.build_transfer_trajectory_grids_nb import CELLS

from subliminal_learning.analysis.loaders.frames import safe_read_json

from subliminal_learning.param_tags import parse_param_tag

def notebook_function(name, namespace):
    """Execute the actual generated notebook function with fixture data dependencies."""
    for kind, source in CELLS:
        if kind != "code":
            continue
        for node in ast.parse(source).body:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                exec(
                    compile(
                        ast.Module(body=[node], type_ignores=[]), "<notebook>", "exec"
                    ),
                    namespace,
                )
                return namespace[name]
    raise AssertionError(name)

def test_pythia_endpoints_require_manifest_final_step_and_keep_unknowns(tmp_path):
    tags = [f"ep20_lr1e-4_r{r}a4" for r in (1, 2, 4, 8, 16)]
    for tag in tags:
        leaf = tmp_path / tag / "ftseed1"
        leaf.mkdir(parents=True)
        (leaf / "train_manifest.json").write_text(json.dumps({"n_train_steps": 100}))
    (tmp_path / tags[3] / "ftseed1/train_diverged.json").write_text("{}")
    (tmp_path / tags[4] / "ftseed1/train_manifest.json").unlink()
    tr = pd.DataFrame(
        [
            dict(tag=tag, ftseed=1, step=step, delta_ce=dce, kl=np.nan, p_test=prob)
            for tag, points in zip(
                tags,
                [
                    [(50, 0.1, 0.9)],  # incomplete: never treat step 50 as final
                    [(50, 0.1, 0.9), (100, np.nan, np.nan)],  # final probe missing
                    [(100, 0.1, 0.4)],  # known alive
                    [(50, 4.0, np.nan)],  # explicitly diverged despite missing final
                    [(100, 0.1, 0.5)],  # no completion manifest
                ],
            )
            for step, dce, prob in points
        ]
    )
    ns = dict(
        pd=pd,
        np=np,
        PY_ARMS={"arm": ("student", "teacher", "anchor", "prior")},
        _lineage=lambda *args: tmp_path,
        _on_lattice=lambda _: True,
        lineage_long_frame=lambda *args, **kwargs: tr.copy(),
        OUTPUT_ROOT=tmp_path,
        parse_param_tag=parse_param_tag,
        CE_COL="delta_ce",
        safe_read_json=safe_read_json,
        CUTS={"anchor": 1.0},
        PY_ANIMAL="dog",
        load_base_prior=lambda *args: 0.1,
    )
    end = notebook_function("load_pythia_arm", ns)("arm")["end"].set_index("tag")
    assert pd.isna(end.loc[tags[0], "p_end"])
    assert pd.isna(end.loc[tags[1], "p_end"])
    assert pd.isna(end.loc[tags[0], "collapsed"])
    assert pd.isna(end.loc[tags[1], "collapsed"])
    assert not end.loc[tags[2], "collapsed"]
    assert end.loc[tags[3], "collapsed"]
    assert pd.isna(end.loc[tags[3], "peak_step"])
    assert pd.isna(end.loc[tags[4], "collapsed"])

def test_checkpoints_without_delta_ce_are_unlabelled_not_labelled_from_kl(tmp_path):
    """A finite KL never stands in for a missing ΔCE: the label must stay NA."""
    tag = "ep20_lr1e-4_r1a4"
    leaf = tmp_path / tag / "ftseed1"
    leaf.mkdir(parents=True)
    (leaf / "train_manifest.json").write_text(json.dumps({"n_train_steps": 100}))
    tr = pd.DataFrame([dict(tag=tag, ftseed=1, step=100, delta_ce=np.nan, kl=9.9, p_test=0.3)])
    ns = dict(
        pd=pd,
        np=np,
        PY_ARMS={"arm": ("student", "teacher", "anchor", "prior")},
        _lineage=lambda *args: tmp_path,
        _on_lattice=lambda _: True,
        lineage_long_frame=lambda *args, **kwargs: tr.copy(),
        OUTPUT_ROOT=tmp_path,
        parse_param_tag=parse_param_tag,
        CE_COL="delta_ce",
        safe_read_json=safe_read_json,
        CUTS={"anchor": 1.0},
        PY_ANIMAL="dog",
        load_base_prior=lambda *args: 0.1,
    )
    end = notebook_function("load_pythia_arm", ns)("arm").pop("end")
    assert pd.isna(end.loc[0, "collapsed"])

def test_olmo_missing_endpoint_and_anchor_remain_unknown():
    from types import SimpleNamespace

    end = pd.DataFrame(
        dict(
            tag=["missing", "diverged", "measured"],
            p_test=[np.nan, np.nan, 0.2],
            p_test_peak=[np.nan, np.nan, 0.3],
            p_test_peak_step=[np.nan, np.nan, 50],
            diverged=[False, True, False],
            collapsed=[np.nan, np.nan, False],
        )
    )
    dl = SimpleNamespace(
        DD1B_BASE="base",
        load_endpoint_frame=lambda *a, **kw: end.copy(),
        load_trajectories=lambda *a, **kw: pd.DataFrame(),
        load_base_prior=lambda *a, **kw: {"test": 0.1},
    )

    def cut(*args, **kwargs):
        return 1.0

    ns = dict(
        dl=dl,
        labels=SimpleNamespace(anchored_cut=cut),
        np=np,
        OUTPUT_ROOT=Path("/unused"),
    )
    fn = notebook_function("load_olmo_arm", ns)
    result = fn("base")["end"]["collapsed"]
    assert pd.isna(result.iloc[0])
    assert result.iloc[1]
    assert not result.iloc[2]

    def missing_cut(*args, **kwargs):
        raise KeyError("missing anchor")

    ns["labels"].anchored_cut = missing_cut
    dl.load_endpoint_frame = lambda *a, **kw: end.drop(columns="collapsed").copy()
    result = fn("base")["end"]["collapsed"]
    assert pd.isna(result.iloc[0]) and pd.isna(result.iloc[2])
    assert result.iloc[1]
