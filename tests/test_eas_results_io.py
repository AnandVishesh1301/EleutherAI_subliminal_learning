"""Tests for the per-student-run EAS results loader.

CPU-only, no model loads: the tag/lineage parsers plus the two loaders against tmp_path
students/ trees holding hand-written documents — a teacher-source run whose MAIN document
carries its own reference, the same run's refs document, a control run (refs document only,
two references), a cross-base run, a nonfinite point, and a malformed document (skipped with a
warning). `load_eas_results` reads only main documents and `load_eas_reference_results` only
refs documents, so each test names the kind it expects.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from subliminal_learning.steering.eas_results_io import (
    RESULT_COLUMNS,
    endpoint_only,
    load_eas_reference_results,
    load_eas_results,
    own_reference_only,
    parse_ft_param_tag,
    parse_lineage_dir,
)

REF_A = "dog/pythia-160m/ep10_lr1e-4_r16a32/ftseed1"
REF_B = "dog/pythia-160m/ep5_lr2e-4_r8a8/ftseed2"


# ------------------------- pure parsers -------------------------


def test_parse_ft_param_tag_ok():
    assert parse_ft_param_tag("ep10_lr1e-4_r16a32") == {
        "teacher_epochs": 10,
        "teacher_lr": 1e-4,
        "teacher_lora_r": 16,
        "teacher_lora_alpha": 32,
    }


def test_parse_ft_param_tag_bad():
    with pytest.raises(ValueError):
        parse_ft_param_tag("not-a-tag")


def test_parse_lineage_dir_ok():
    assert parse_lineage_dir("teacher-pythia-160m", "ep10_lr1e-4_r16a32_tseed1") == {
        "teacher_param_tag": "ep10_lr1e-4_r16a32",
        "teacher_ftseed": 1,
        "numbers_base_model": "pythia-160m",
    }


def test_parse_lineage_dir_bad():
    with pytest.raises(ValueError):
        parse_lineage_dir("teacher-pythia-160m", "notatag_tseed1")
    with pytest.raises(ValueError):
        parse_lineage_dir("notateacher", "ep10_lr1e-4_r16a32_tseed1")


# ------------------------- document fixtures -------------------------


def _reference(ref_key: str) -> dict:
    tag, ftseed = ref_key.split("/")[2], int(ref_key.split("/")[3][len("ftseed"):])
    return {
        "teacher_run_rel": f"teachers/{ref_key}",
        "animal": "dog",
        "teacher_base_repo": "EleutherAI/pythia-160m",
        "teacher_base_model_dir": "pythia-160m",
        "teacher_param_tag": tag,
        "teacher_ftseed": ftseed,
        "teacher_adapter_id": "aaaa",
        "v_teacher_norm_per_layer": [0.0, 1.0, 2.0],
    }


def _doc(
    run_rel: str,
    *,
    position: str = "last",
    source: str = "teacher",
    refs: tuple[str, ...] = (REF_A,),
    own_reference: str | None = REF_A,
    base_model_dir: str = "pythia-160m",
    numbers_base_model_dir: str = "pythia-160m",
    cross_base: bool = False,
    nonfinite_point: bool = False,
) -> dict:
    points = [
        {
            "step": 500, "ckpt_rel": "checkpoints/checkpoint-500", "is_endpoint": False,
            "adapter_id": "s500", "nonfinite": False,
            "v_student_norm_per_layer": [0.0, 1.0, 2.0],
            "eas": {r: {"mean": 0.10, "per_layer": [0.0, 0.1, 0.2]} for r in refs},
        },
        {
            "step": 1000, "ckpt_rel": "checkpoints/checkpoint-1000", "is_endpoint": True,
            "adapter_id": "s1000", "nonfinite": False,
            "v_student_norm_per_layer": [0.0, 1.0, 2.0],
            "eas": {r: {"mean": 0.42, "per_layer": [0.0, 0.4, 0.5]} for r in refs},
        },
    ]
    if nonfinite_point:
        points.insert(1, {
            "step": 750, "ckpt_rel": "checkpoints/checkpoint-750", "is_endpoint": False,
            "adapter_id": "s750", "nonfinite": True,
            "v_student_norm_per_layer": [0.0, 1.0, 2.0], "eas": None,
        })
    return {
        "schema_version": 1,
        "basis": "raw_blocks_v2",
        "eas_mean_convention": "raw_blocks_v2",
        "position": position,
        "n_prompts": 1024,
        "prompt_sha256": "p" * 8,
        "prompt_source": {
            "number_dataset_dir_rel": "numbers/x", "filtered_sha256": "f" * 8,
            "subsample_size": 10000, "subsample_seed": 42, "splice_examples": False,
        },
        "student": {
            "run_rel": run_rel, "source": source, "animal": "dog",
            "base_repo": f"EleutherAI/{base_model_dir}", "base_model_dir": base_model_dir,
            "numbers_base_model_dir": numbers_base_model_dir, "cross_base": cross_base,
            "student_param_tag": "ep5_lr1e-4_r8a32", "student_ftseed": 1,
            "number_dataset_tag": "ex10-20_prefix_n100k",
            "own_reference": own_reference, "manifest_ref": {},
        },
        "references": {r: _reference(r) for r in refs},
        "points": points,
    }


def _write(
    root: Path, run_rel: str, position: str, text: str, kind: str = "eas"
) -> None:
    """Write one document of `kind` (``eas`` = main, ``eas_refs`` = extra references)."""
    d = root / run_rel / "eval"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{kind}_{position}.json").write_text(text)


TEACHER_RUN = (
    "students/dog/pythia-160m/teacher-pythia-160m/ep10_lr1e-4_r16a32_tseed1/"
    "ex10-20_prefix_n100k/ep5_lr1e-4_r8a32/ftseed1"
)
CONTROL_RUN = (
    "students/controls/base/pythia-160m/ex10-20_prefix_n100k/ep5_lr1e-4_r8a32/ftseed1"
)
CROSS_RUN = (
    "students/dog/pythia-160m-data-seed1/teacher-pythia-160m/ep10_lr1e-4_r16a32_tseed1/"
    "ex10-20_prefix_n100k/ep5_lr1e-4_r8a32/ftseed1"
)
BAD_RUN = "students/dog/pythia-160m/broken/ep5_lr1e-4_r8a32/ftseed1"


# ------------------------- load_eas_results -------------------------


def _tree(tmp_path: Path) -> None:
    _write(tmp_path, TEACHER_RUN, "last", json.dumps(_doc(TEACHER_RUN)))
    _write(tmp_path, TEACHER_RUN, "all", json.dumps(_doc(TEACHER_RUN, position="all")))
    # The extra reference of the same run lives beside its main document.
    _write(tmp_path, TEACHER_RUN, "last", json.dumps(_doc(
        TEACHER_RUN, refs=(REF_B,),
    )), kind="eas_refs")
    # A control run has no own teacher, hence ONLY a refs document.
    _write(tmp_path, CONTROL_RUN, "last", json.dumps(_doc(
        CONTROL_RUN, source="base", refs=(REF_A, REF_B), own_reference=None,
        nonfinite_point=True,
    )), kind="eas_refs")
    _write(tmp_path, CROSS_RUN, "last", json.dumps(_doc(
        CROSS_RUN, base_model_dir="pythia-160m-data-seed1",
        numbers_base_model_dir="pythia-160m", cross_base=True,
    )))
    _write(tmp_path, BAD_RUN, "last", "{ not json")


def test_load_eas_results_discovery_and_columns(tmp_path):
    _tree(tmp_path)
    df = load_eas_results(tmp_path)
    assert list(df.columns) == RESULT_COLUMNS
    # MAIN documents only: teacher run 2 positions × 2 points × 1 ref, cross run 2 points.
    assert len(df) == 4 + 2
    assert set(df["run_path"]) == {TEACHER_RUN, CROSS_RUN}
    row = df[(df.run_path == TEACHER_RUN) & (df.position == "last") & (df.step == 1000)].iloc[0]
    assert row["source"] == "teacher"
    assert row["animal"] == "dog"
    assert row["model"] == "pythia-160m"
    assert row["numbers_base_model"] == "pythia-160m"
    assert row["cross_base"] == False  # noqa: E712
    assert row["teacher_param_tag"] == "ep10_lr1e-4_r16a32"
    assert (row["teacher_epochs"], row["teacher_lora_r"], row["teacher_lora_alpha"]) == (10, 16, 32)
    assert row["teacher_ftseed"] == 1
    assert row["reference"] == REF_A
    assert row["is_own_reference"]
    assert row["dataset_tag"] == "ex10-20_prefix_n100k"
    assert row["spliced"] == False  # noqa: E712
    assert row["n_prompts"] == 1024
    assert row["eas_mean_convention"] == "raw_blocks_v2"
    assert (row["student_epochs"], row["student_lora_r"]) == (5, 8)
    assert row["student_ftseed"] == 1
    assert row["is_endpoint"]
    assert row["eas_mean"] == 0.42
    assert row["eas_per_layer"] == [0.0, 0.4, 0.5]
    # The per-slot norms of both vectors ride along (point-side and reference-side).
    assert row["v_student_norm_per_layer"] == [0.0, 1.0, 2.0]
    assert row["v_teacher_norm_per_layer"] == [0.0, 1.0, 2.0]


def test_load_eas_reference_results_control_two_references_and_nonfinite(tmp_path):
    _tree(tmp_path)
    df = load_eas_reference_results(tmp_path)
    assert list(df.columns) == RESULT_COLUMNS
    ctl = df[df.run_path == CONTROL_RUN]
    assert set(ctl["reference"]) == {REF_A, REF_B}
    assert not ctl["is_own_reference"].any()  # a control has no own reference
    # The two references carry their OWN teacher HPs on the same control rows.
    assert set(ctl[ctl.reference == REF_B]["teacher_param_tag"]) == {"ep5_lr2e-4_r8a8"}
    nf = ctl[ctl.step == 750]
    assert len(nf) == 2 and nf["nonfinite"].all()
    assert nf["eas_mean"].isna().all()
    assert nf["eas_per_layer"].isna().all()
    # A nonfinite point still carries its own recorded student norms.
    assert all(v == [0.0, 1.0, 2.0] for v in nf["v_student_norm_per_layer"])


def test_load_eas_results_cross_base(tmp_path):
    _tree(tmp_path)
    df = load_eas_results(tmp_path)
    cross = df[df.cross_base]
    assert set(cross["run_path"]) == {CROSS_RUN}
    assert set(cross["model"]) == {"pythia-160m-data-seed1"}
    assert set(cross["numbers_base_model"]) == {"pythia-160m"}


def test_load_eas_results_malformed_doc_skipped(tmp_path):
    _tree(tmp_path)
    df = load_eas_results(tmp_path)
    assert BAD_RUN not in set(df["run_path"])


def test_load_eas_results_explicit_run_rel_paths(tmp_path):
    _tree(tmp_path)
    df = load_eas_results(tmp_path, run_rel_paths=[TEACHER_RUN])
    assert set(df["run_path"]) == {TEACHER_RUN}


def test_load_eas_results_position_filter(tmp_path):
    _tree(tmp_path)
    assert set(load_eas_results(tmp_path)["position"]) == {"last", "all"}
    assert set(load_eas_results(tmp_path, positions=["all"])["position"]) == {"all"}


def test_each_loader_sees_only_its_own_documents(tmp_path):
    _tree(tmp_path)
    main = load_eas_results(tmp_path)
    refs = load_eas_reference_results(tmp_path)
    # The main frame holds own-reference rows only; the refs frame holds none of them.
    assert set(main["reference"]) == {REF_A}
    assert main["is_own_reference"].all()
    assert CONTROL_RUN not in set(main["run_path"])
    assert set(refs["run_path"]) == {TEACHER_RUN, CONTROL_RUN}
    assert not refs["is_own_reference"].any()
    assert set(refs[refs.run_path == TEACHER_RUN]["reference"]) == {REF_B}
    assert set(main["eas_file"]) == {
        str(tmp_path / r / "eval" / f"eas_{p}.json")
        for r, p in ((TEACHER_RUN, "last"), (TEACHER_RUN, "all"), (CROSS_RUN, "last"))
    }
    assert all(Path(f).name == "eas_refs_last.json" for f in refs["eas_file"])


def test_loaders_empty(tmp_path):
    (tmp_path / "students").mkdir()
    for df in (load_eas_results(tmp_path), load_eas_reference_results(tmp_path)):
        assert list(df.columns) == RESULT_COLUMNS
        assert df.empty
        assert endpoint_only(df).empty and own_reference_only(df).empty


def test_load_eas_results_warns_on_mixed_convention(tmp_path):
    from loguru import logger
    _write(tmp_path, TEACHER_RUN, "last", json.dumps(_doc(TEACHER_RUN)))
    other = _doc(CROSS_RUN)
    other["eas_mean_convention"] = "blocks_v1"
    _write(tmp_path, CROSS_RUN, "last", json.dumps(other))
    sink: list[str] = []
    handle = logger.add(sink.append, level="WARNING")
    try:
        df = load_eas_results(tmp_path)
    finally:
        logger.remove(handle)
    assert set(df["eas_mean_convention"]) == {"raw_blocks_v2", "blocks_v1"}
    assert any("convention" in m for m in sink)


def test_endpoint_only_and_own_reference_only(tmp_path):
    _tree(tmp_path)
    df = load_eas_results(tmp_path)
    end = endpoint_only(df)
    assert set(end["step"]) == {1000}
    # one endpoint row per (run, position, reference): 1 + 1 (teacher, both positions) + 1
    assert len(end) == 3
    own = own_reference_only(df)
    assert set(own["run_path"]) == {TEACHER_RUN, CROSS_RUN}
    assert own["is_own_reference"].all()


def test_root_ckpt_rel_point_is_flagged_as_the_endpoint(tmp_path):
    """The `checkpoints` root IS the endpoint: a doc whose own `is_endpoint` flag says
    otherwise must still produce an endpoint row (`endpoint_only` reads that column)."""
    doc = _doc(TEACHER_RUN)
    doc["points"][-1].update(ckpt_rel="checkpoints", is_endpoint=False)
    _write(tmp_path, TEACHER_RUN, "last", json.dumps(doc))
    df = load_eas_results(tmp_path)
    final = df[df.step == 1000]
    assert len(final) == 1 and bool(final.iloc[0]["is_endpoint"]) is True
    assert set(endpoint_only(df)["step"]) == {1000}


def test_distinct_non_endpoint_points_at_one_step_warn(tmp_path):
    """Two different non-endpoint rels at a step are not the same weights: one is dropped,
    and that is a document defect worth a WARNING (file + step + both rels)."""
    from loguru import logger

    doc = _doc(TEACHER_RUN)
    collision = {**doc["points"][0], "ckpt_rel": "checkpoints/checkpoint-500-retry",
                 "adapter_id": "s500b"}
    doc["points"].insert(1, collision)
    _write(tmp_path, TEACHER_RUN, "last", json.dumps(doc))
    sink: list[str] = []
    handle = logger.add(sink.append, level="WARNING")
    try:
        df = load_eas_results(tmp_path)
    finally:
        logger.remove(handle)
    assert len(df[df.step == 500]) == 1
    assert any("checkpoint-500-retry" in m and "checkpoints/checkpoint-500" in m
               and "step 500" in m and "eas_last.json" in m for m in sink)


def test_duplicate_final_step_point_keeps_the_endpoint(tmp_path):
    """Old docs hold the final step twice (`checkpoint-N` dir + `checkpoints` endpoint)."""
    doc = _doc(TEACHER_RUN)
    doc["points"].insert(1, {
        "step": 1000, "ckpt_rel": "checkpoints/checkpoint-1000", "is_endpoint": False,
        "adapter_id": "dup1000", "nonfinite": False,
        "v_student_norm_per_layer": [0.0, 1.0, 2.0],
        "eas": {REF_A: {"mean": 0.99, "per_layer": [0.0, 0.9, 0.9]}},
    })
    doc["points"][-1]["ckpt_rel"] = "checkpoints"
    _write(tmp_path, TEACHER_RUN, "last", json.dumps(doc))
    df = load_eas_results(tmp_path)
    assert len(df) == 2  # 500 + one row at 1000, not two
    final = df[df.step == 1000]
    assert len(final) == 1
    assert bool(final.iloc[0]["is_endpoint"]) is True
    assert final.iloc[0]["eas_mean"] == 0.42
    assert set(endpoint_only(df)["step"]) == {1000}
