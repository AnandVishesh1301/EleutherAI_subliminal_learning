"""Behavioral regressions for the experimental audit fixes (no model loads)."""

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from subliminal_learning.evaluation import fidelity_io as fid
from scripts import run_final_evals as final
from notebooks.builders.build_transfer_trajectory_grids_nb import CELLS
from subliminal_learning.analysis.frames import safe_read_json
from subliminal_learning.param_tags import parse_param_tag


def test_split_and_cache_identity_match_spliced_training(tmp_path):
    rows = [
        {"prompt": f"Here: {i}, {i + 1}", "completion": f" {i + 2}"} for i in range(10)
    ]
    (tmp_path / "filtered.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    man = dict(number_dataset_dir=str(tmp_path), subsample_size=4, subsample_seed=1)
    original = fid.split_rows(man, 3)
    spliced = dict(man, splice_examples=True)
    transformed = fid.split_rows(spliced, 3)
    assert fid.rows_cache_key(man) != fid.rows_cache_key(spliced)
    for before, after in zip(original, transformed):
        assert len(before) == len(after) == 3
        assert all(r["prompt"] == "Here:" for r in after)
        for src, dst in zip(before, after):
            assert dst["completion"].endswith(src["completion"])
    assert {r["completion"] for r in original[0]}.isdisjoint(
        r["completion"] for r in original[1]
    )


@pytest.mark.parametrize("spliced", [False, True])
@pytest.mark.parametrize("kind", ["fidelity", "shuffle"])
def test_disk_cache_invalidates_only_legacy_spliced_prompts(
    tmp_path, monkeypatch, spliced, kind
):
    (tmp_path / "train_manifest.json").write_text(
        json.dumps({"splice_examples": spliced})
    )
    (tmp_path / "eval").mkdir()
    module = fid if kind == "fidelity" else final
    monkeypatch.setattr(module, "manifest_ref", lambda _: {"same": "run"})
    record = dict(
        schema_version=fid.SCHEMA_VERSION
        if kind == "fidelity"
        else final.SHUFFLE_SCHEMA,
        manifest_ref={"same": "run"},
        n_rows=3,
    )
    path = (
        tmp_path
        / "eval"
        / (fid.EVAL_NAME if kind == "fidelity" else final.SHUFFLE_NAME)
    )
    check = fid.eval_is_current if kind == "fidelity" else final.shuffle_is_current
    path.write_text(json.dumps(record))
    assert check(tmp_path, 3) is (not spliced)
    if spliced:
        record["prompt_transform"] = "splice_examples_v1"
        path.write_text(json.dumps(record))
        assert check(tmp_path, 3)


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


def test_truncated_trajectory_is_only_carried_forward_when_marked_diverged():
    tr = pd.DataFrame(
        [
            dict(tag=tag, step=step, delta_ce=0.1, lr=0.01, rank=1, alpha=2)
            for tag, steps in [
                ("partial", [50]),
                ("dead", [50]),
                ("full1", [50, 100]),
                ("full2", [50, 100]),
                ("full3", [50, 100]),
            ]
            for step in steps
        ]
    )
    end = pd.DataFrame(
        dict(
            tag=["partial", "dead", "full1", "full2", "full3"],
            diverged=[False, True, False, False, False],
        )
    )
    fn = notebook_function("labelled_long", dict(pd=pd, np=np, CE_COL="delta_ce"))
    result = fn("arm", dict(tr=tr, end=end, cut=1.0))
    assert not ((result["tag"] == "partial") & (result["step"] == 100)).any()
    dead = result[(result["tag"] == "dead") & (result["step"] == 100)]
    assert dead["collapsed"].tolist() == [1]
    assert fn("arm", dict(tr=tr, end=end, cut=np.nan)).empty


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
