"""Behavioral regressions for the experimental audit fixes (no model loads)."""

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import fidelity_heldout_ce as fid
from scripts import run_final_evals as final
from scripts.build_transfer_trajectory_grids_nb import CELLS
from scripts.position_surgery import (
    prepare_output_for_redo,
    run_is_complete,
    stable_hash,
    target_rng,
    write_surgery_records,
)
from subliminal_learning.analysis.frames import safe_read_json
from subliminal_learning.param_tags import parse_param_tag
from subliminal_learning.weightspace.deltas import (
    INTERPOLATION_VERSION,
    TARGET_VERSION,
    interp_basis,
    interpolation_record_is_current,
    random_orthonormal_pair,
    surgery_original_delta_ce,
    surgery_record_is_current,
    validate_surgery_baselines,
)
from subliminal_learning.weightspace.lora import lora_svd


def test_paired_interpolation_reconstructs_original_and_preserves_spectrum():
    rng = np.random.default_rng(3)
    b, a = rng.normal(size=(12, 3)), rng.normal(size=(3, 9))
    u, sv, v = lora_svd(b, a)
    ur = np.linalg.qr(rng.normal(size=u.shape))[0]
    vr = np.linalg.qr(rng.normal(size=v.shape))[0]
    for t in (0, 1e-8, 0.25, 0.75, 1 - 1e-8, 1):
        dw = (interp_basis(u, ur, t) * sv) @ interp_basis(v, vr, t).T
        np.testing.assert_allclose(np.linalg.svd(dw, compute_uv=False)[:3], sv)
        if t < 1e-7:
            np.testing.assert_allclose(dw, b @ a, atol=1e-6)
        if t > 1 - 1e-7:
            np.testing.assert_allclose(dw, (ur * sv) @ vr.T, atol=1e-6)


def test_surgery_upsert_replaces_legacy_without_discarding_other_runs(tmp_path):
    path = tmp_path / "surgery.jsonl"
    old = [{"run": "a", "variant": "randt0.5"}, {"run": "b", "variant": "orig"}]
    path.write_text("".join(json.dumps(r) + "\n" for r in old))
    assert not interpolation_record_is_current(old[0])
    assert interpolation_record_is_current(old[1])
    fixed = dict(old[0], interpolation_version=INTERPOLATION_VERSION)
    assert interpolation_record_is_current(fixed)
    write_surgery_records(path, [fixed])
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records == [old[1], fixed]
    write_surgery_records(path, [fixed])
    assert len(path.read_text().splitlines()) == 2


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


def test_notebooks_reject_legacy_progression_records():
    for name in ("collapse_surgery_and_sgd", "collapse_two_factor_paper_figures"):
        nb = json.loads((Path("notebooks") / f"{name}.ipynb").read_text())
        source = next(
            "".join(c["source"])
            for c in nb["cells"]
            if "pr = read_jsonl(" in "".join(c.get("source", []))
            and "position_surgery" in "".join(c.get("source", []))
        )
        tree = ast.parse(source)
        guard = next(
            n
            for n in tree.body
            if isinstance(n, ast.If)
            and "surgery_record_is_current" in ast.unparse(n.test)
        )
        ns = dict(
            pr=pd.DataFrame([{"variant": "randt0.5"}]),
            surgery_record_is_current=surgery_record_is_current,
        )
        with pytest.raises(ValueError, match="Legacy randt"):
            exec(
                compile(ast.Module(body=[guard], type_ignores=[]), "<guard>", "exec"),
                ns,
            )


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


def test_random_targets_are_keyed_not_order_dependent():
    def draw(run_key, module, d):
        return random_orthonormal_pair(6, 5, 2, target_rng(0, run_key, module, "rand", d))

    first = draw("a/run1", "layer.0", 0)
    # draw other things in between: a keyed target must not notice
    draw("a/run0", "layer.9", 3)
    draw("a/run1", "layer.0", 1)
    again = draw("a/run1", "layer.0", 0)
    for x, y in zip(first, again):
        np.testing.assert_array_equal(x, y)
    for other in (draw("a/run1", "layer.0", 1), draw("a/run1", "layer.1", 0),
                  draw("a/run2", "layer.0", 0)):
        assert not np.allclose(first[0], other[0])
    assert stable_hash("layer.0") == stable_hash("layer.0")
    assert stable_hash("layer.0") != stable_hash("layer.1")


def test_rewriting_a_run_drops_all_of_its_old_variants(tmp_path):
    path = tmp_path / "surgery.jsonl"
    old = [
        {"run": "a", "variant": "orig"},
        {"run": "a", "variant": "randt0.5"},
        {"run": "b", "variant": "orig"},
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in old))
    fresh = [{"run": "a", "variant": "orig", "delta_ce": 1.0}]
    write_surgery_records(path, fresh)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records == [old[2]] + fresh


def test_surgery_record_is_current_requires_keyed_targets_only_when_asked():
    randt = {"variant": "randt0.5", "interpolation_version": INTERPOLATION_VERSION}
    assert surgery_record_is_current(randt)
    assert not surgery_record_is_current(randt, require_keyed_targets=True)
    assert surgery_record_is_current(
        dict(randt, target_version=TARGET_VERSION), require_keyed_targets=True
    )
    assert not surgery_record_is_current({"variant": "randt0.5"})
    assert surgery_record_is_current({"variant": "orig"}, require_keyed_targets=True)
    assert not surgery_record_is_current({"variant": "rand0"}, require_keyed_targets=True)
    assert not surgery_record_is_current({"variant": "top4"}, require_keyed_targets=True)


def test_resume_skips_only_a_complete_identity_matching_set():
    identity = dict(
        adapter_sha256="abc",
        pool_version="pp1",
        pool_sha="sha",
        prompt_set="test",
        prompt_set_sha256="psha",
        animal="dog",
        seed=0,
    )
    wanted = ["orig", "rand0"]

    def row(variant, **over):
        rec = dict(identity, run="a", variant=variant)
        if variant not in ("orig", "align"):
            rec["target_version"] = TARGET_VERSION
        return dict(rec, **over)

    rows = [row("orig"), row("rand0")]
    assert run_is_complete(rows, identity, wanted)
    assert not run_is_complete([row("orig")], identity, wanted)  # incomplete
    assert not run_is_complete([], identity, wanted)
    assert not run_is_complete(
        [row("orig"), row("rand0", adapter_sha256="other")], identity, wanted
    )  # retrained adapter
    assert not run_is_complete(
        [row("orig"), row("rand0", pool_sha="drifted")], identity, wanted
    )
    legacy = [{"run": "a", "variant": v} for v in wanted]
    assert not run_is_complete(legacy, identity, wanted)  # legacy rows lack identity


def test_redo_backs_up_then_empties_the_output(tmp_path):
    out = tmp_path / "position_surgery_ce.jsonl"
    assert prepare_output_for_redo(out) is None
    body = json.dumps({"run": "gone", "variant": "orig"}) + "\n"
    out.write_text(body)
    backup = prepare_output_for_redo(out)
    assert backup is not None and backup.read_text() == body
    assert backup.name.startswith("position_surgery_ce.legacy-")
    assert not out.exists()  # rows of runs outside this runs list must not survive


def test_surgery_baseline_validator_rejects_legacy_mixed_and_identity_conflicts():
    base = dict(run="r", original_baseline_version="ce-baseline-v1", original_delta_ce=2.0,
                adapter_sha256="a", base="m@rev", base_revision="rev", pool_version="pp1",
                pool_sha="p", prompt_set_sha256="q", animal="dog", seed=0)
    validate_surgery_baselines([base], version="ce-baseline-v1")
    with pytest.raises(ValueError, match="current original baseline"):
        validate_surgery_baselines([{**base, "original_baseline_version": "old"}], version="ce-baseline-v1")
    with pytest.raises(ValueError, match="finite"):
        validate_surgery_baselines([{**base, "original_delta_ce": None}], version="ce-baseline-v1")
    with pytest.raises(ValueError, match="input identities"):
        validate_surgery_baselines([base, {**base, "adapter_sha256": "b", "original_delta_ce": 2.0}],
                                   version="ce-baseline-v1")


def test_surgery_baseline_falls_back_to_stored_probe_without_raising(tmp_path):
    legacy = tmp_path / "legacy"
    (legacy / "eval").mkdir(parents=True)
    (legacy / "eval" / "pile_probe.json").write_text(json.dumps({"delta_ce": 0.9}))
    paired = dict(run="paired", original_baseline_version="ce-baseline-v1",
                  original_delta_ce=1.5, adapter_sha256="a", base="m", base_revision=None,
                  pool_version="p", pool_sha="s", prompt_set_sha256="q", animal="dog", seed=0)
    rows = [paired, {**paired, "variant": "rand0"},
            {"run": str(legacy)}, {"run": str(tmp_path / "gone")}]
    base, src = surgery_original_delta_ce(rows)
    assert src == {"paired": "paired", str(legacy): "stored_probe", str(tmp_path / "gone"): "missing"}
    assert base["paired"] == 1.5 and base[str(legacy)] == 0.9
    assert np.isnan(base[str(tmp_path / "gone")])
    with pytest.raises(ValueError, match="within run"):
        surgery_original_delta_ce([paired, {**paired, "original_delta_ce": 0.8}])
