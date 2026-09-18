"""EAS sweep discovery + cheap-gate tests (CPU-only, tmp_path trees, no model loads).

Builds fake student run dirs (manifest + empty adapter files) under a temporary
`output_root` and exercises `scripts.sweeps.sweep_compute_eas`: the depth-agnostic run-dir
walk (plain, `ckpt/`-gathered, optimizer-prefixed, `controls/`), the manifest filters, the
grouping order, and the cheap cache gate over BOTH documents the pipeline writes (the
own-teacher ``eas_{position}.json`` and the extra-reference ``eas_refs_{position}.json``) —
including that it detects a refreshed pool, a new reference teacher and a retrained endpoint
without hashing intermediate checkpoints.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.sweeps.sweep_compute_eas import (
    build_unit,
    build_units,
    discover_run_dirs,
    expected_n_prompts,
    manifest_passes,
    path_prefilter,
    prompt_source_matches,
    sort_units,
    unit_is_cached,
    unit_status,
)
from scripts.sweeps.sweep_compute_eas import DONE_UNSCORABLE, WORK
from subliminal_learning.steering.eas import EAS_MEAN_CONVENTION, SCHEMA_VERSION
from subliminal_learning.steering.vectors import ACTIVATION_BASIS

_PYTHIA = "EleutherAI/pythia-160m"
_TEACHER_KEY = "dog/pythia-160m/ep10_lr1e-4_r16a16/ftseed1"
_OTHER_KEY = "cat/pythia-160m/ep10_lr1e-4_r16a16/ftseed1"
_THIRD_KEY = "lion/pythia-160m/ep10_lr1e-4_r16a16/ftseed1"


def _manifest(**over) -> dict:
    m = {
        "source": "teacher",
        "animal": "dog",
        "base_model_repo": _PYTHIA,
        "numbers_base_model_repo": _PYTHIA,
        "cross_base": False,
        "student_param_tag": "ep20_lr4e-4_r16a4",
        "student_ftseed": 1,
        "number_dataset_tag": "ex10-20_prefix_n100k",
        "number_dataset_dir": "/mnt/ssd-2/x/outputs/numbers/dog/pool",
        "teacher_param_tag": "ep10_lr1e-4_r16a16",
        "teacher_ftseed": 1,
        "filtered_sha256": "poolsha",
        "subsample_size": 10000,
        "subsample_seed": 42,
        "n_train_steps": 100,
    }
    m.update(over)
    return m


def _run(root: Path, rel: str, *, steps=(), manifest=None, adapter=b"end") -> Path:
    """A fake run dir: manifest + endpoint adapter (+ one adapter per intermediate step)."""
    d = root / "students" / rel
    (d / "checkpoints").mkdir(parents=True)
    (d / "checkpoints" / "adapter_model.safetensors").write_bytes(adapter)
    for s in steps:
        c = d / "checkpoints" / f"checkpoint-{s}"
        c.mkdir()
        (c / "adapter_model.safetensors").write_bytes(f"ck{s}".encode())
    (d / "train_manifest.json").write_text(json.dumps(manifest or _manifest()))
    return d


def _teacher(root: Path, key: str = _TEACHER_KEY, payload: bytes = b"tw") -> Path:
    d = root / "teachers" / key / "checkpoints"
    d.mkdir(parents=True)
    (d / "adapter_model.safetensors").write_bytes(payload)
    return d.parent


# ------------------------------------------------------------------- discovery
def test_discover_finds_every_layout_at_any_depth(tmp_path):
    rels = [
        "dog/pythia-160m/teacher-pythia-160m/t_tseed1/tag/ep20_lr4e-4_r16a4/ftseed1",
        "dog/ckpt/pythia-160m@step1000/teacher-pythia-160m/t_tseed1/tag/p/ftseed1",
        "sgd/dog/pythia-160m/teacher-pythia-160m/t_tseed1/tag/p/ftseed1",
        "controls/base/pythia-160m/teacher-pythia-160m/tag/p/ftseed1",
        "controls/rng/pythia-160m/tag/p/ftseed1",
    ]
    for rel in rels:
        _run(tmp_path, rel)
    found = {
        str(p.relative_to(tmp_path / "students")) for p in
        discover_run_dirs(tmp_path / "students")
    }
    assert found == set(rels)


def test_discover_ignores_a_dir_without_a_final_adapter(tmp_path):
    d = tmp_path / "students" / "dog" / "x" / "ftseed1"
    (d / "checkpoints").mkdir(parents=True)
    (d / "train_manifest.json").write_text(json.dumps(_manifest()))
    assert discover_run_dirs(tmp_path / "students") == []


def test_discover_missing_root_is_empty(tmp_path):
    assert discover_run_dirs(tmp_path / "nope") == []


# ------------------------------------------------------------------- units
def test_build_unit_lists_checkpoints_endpoint_last(tmp_path):
    rd = _run(tmp_path, "dog/a/ftseed1", steps=(50, 100))
    _teacher(tmp_path)
    unit = build_unit(rd, tmp_path, "last", endpoint_only=False, extra_refs=())
    # `checkpoint-100` sits at the manifest's `n_train_steps`, i.e. it is the endpoint
    # adapter's former byte-duplicate, and is enumerated ONCE (as the endpoint).
    assert unit.ckpt_rels == ("checkpoints/checkpoint-50", "checkpoints")
    assert unit.references == (_TEACHER_KEY,)
    assert unit.base_model_dir == "pythia-160m"


def test_build_unit_endpoint_only(tmp_path):
    rd = _run(tmp_path, "dog/a/ftseed1", steps=(50,))
    unit = build_unit(rd, tmp_path, "last", endpoint_only=True, extra_refs=())
    assert unit.ckpt_rels == ("checkpoints",)


def test_build_unit_control_without_reference_is_a_skip_reason(tmp_path):
    rd = _run(tmp_path, "controls/base/x/ftseed1",
              manifest=_manifest(source="base", animal=None))
    assert isinstance(build_unit(rd, tmp_path, "last", endpoint_only=True, extra_refs=()), str)


def test_build_unit_control_with_reference_is_a_unit(tmp_path):
    rd = _run(tmp_path, "controls/base/x/ftseed1",
              manifest=_manifest(source="base", animal=None))
    unit = build_unit(
        rd, tmp_path, "last", endpoint_only=True, extra_refs=(_TEACHER_KEY,)
    )
    assert unit.references == (_TEACHER_KEY,) and unit.source == "base"


def test_manifest_filters():
    m = _manifest()
    assert manifest_passes(m, animals={"dog"}, models={"pythia-160m"},
                           tags={"ex10-20_prefix_n100k"},
                           student_tags={"ep20_lr4e-4_r16a4"}, sources={"teacher"})
    assert not manifest_passes(m, animals={"cat"}, models=set(), tags=set(),
                               student_tags=set(), sources=set())
    assert not manifest_passes(m, animals=set(), models=set(), tags=set(),
                               student_tags={"ep5_lr1e-4_r1a1"}, sources=set())


def test_path_prefilter_keeps_anything_a_manifest_could_match():
    rel = ("students/dog/pythia-160m/teacher-pythia-160m/t_tseed1/"
           "ex10-20_prefix_n100k/ep20_lr4e-4_r16a4/ftseed1")
    assert path_prefilter(rel, animals={"dog"}, models={"pythia-160m"},
                          tags={"ex10-20_prefix_n100k"},
                          student_tags={"ep20_lr4e-4_r16a4"})
    assert path_prefilter(rel, animals=set(), models=set(), tags=set(), student_tags=set())
    assert not path_prefilter(rel, animals={"cat"}, models=set(), tags=set(),
                              student_tags=set())
    assert not path_prefilter(rel, animals=set(), models=set(), tags=set(),
                              student_tags={"ep5_lr1e-4_r1a1"})


def test_build_units_applies_filters_and_both_positions(tmp_path):
    _run(tmp_path, "dog/a/ftseed1", steps=(50,))
    _run(tmp_path, "cat/a/ftseed1", manifest=_manifest(animal="cat"))
    _teacher(tmp_path)
    units, _ = build_units(
        tmp_path, positions=("last", "all"), animals={"dog"}, models=set(), tags=set(),
        student_tags=set(), sources=set(), min_checkpoints=1, endpoint_only=False,
        extra_refs=(),
    )
    assert {u.position for u in units} == {"last", "all"}
    assert {u.animal for u in units} == {"dog"}


def test_build_units_min_checkpoints_skips(tmp_path):
    _run(tmp_path, "dog/a/ftseed1")
    units, skipped = build_units(
        tmp_path, positions=("last",), animals=set(), models=set(), tags=set(),
        student_tags=set(), sources=set(), min_checkpoints=3, endpoint_only=False,
        extra_refs=(),
    )
    assert units == [] and "min-checkpoints" in skipped[0][1]


def test_sort_units_groups_by_base_then_prompt_set(tmp_path):
    a = _run(tmp_path, "dog/a/ftseed1")
    b = _run(tmp_path, "dog/b/ftseed1",
             manifest=_manifest(base_model_repo="EleutherAI/pythia-160m-data-seed1"))
    c = _run(tmp_path, "dog/c/ftseed1", manifest=_manifest(filtered_sha256="other"))
    units = [
        build_unit(p, tmp_path, "last", endpoint_only=True, extra_refs=())
        for p in (b, c, a)
    ]
    order = [u.base_model_dir for u in sort_units(units)]
    assert order == ["pythia-160m", "pythia-160m", "pythia-160m-data-seed1"]


# ------------------------------------------------------------------- cheap gate
def _doc(unit, manifest, *, refs=(_TEACHER_KEY,), adapter_ids=None) -> dict:
    adapter_ids = adapter_ids or {}
    return {
        "schema_version": SCHEMA_VERSION,
        "basis": ACTIVATION_BASIS,
        "eas_mean_convention": EAS_MEAN_CONVENTION,
        "position": unit.position,
        "n_prompts": 1024,
        "prompt_sha256": "sha",
        "prompt_source": {
            "filtered_sha256": manifest["filtered_sha256"],
            "subsample_size": manifest["subsample_size"],
            "subsample_seed": manifest["subsample_seed"],
            "splice_examples": False,
        },
        "student": {"manifest_ref": unit.manifest_ref},
        "references": {
            r: {"teacher_adapter_id": adapter_ids.get(r) or _teacher_id(unit, r)}
            for r in refs
        },
        "points": [
            # `step` is what the thinned-point rule keys on (`training.ckpt_ladder`).
            {"ckpt_rel": rel,
             "step": (int(rel.rsplit("-", 1)[1]) if "checkpoint-" in rel
                      else manifest["n_train_steps"]),
             "adapter_id": unit.endpoint_adapter_id if rel == unit.ckpt_rels[-1] else "x",
             "nonfinite": False,
             "eas": {r: {"mean": 0.1, "per_layer": [0.1]} for r in refs}}
            for rel in unit.ckpt_rels
        ],
    }


def _teacher_id(unit, ref_key):
    from pipelines.evaluate_teacher_preference_pipeline import adapter_identity

    root = unit.run_dir
    while root.name != "students":
        root = root.parent
    return adapter_identity(root.parent / "teachers" / ref_key / "checkpoints")


def _write_doc(unit, doc, kind="eas"):
    p = unit.run_dir / "eval" / f"{kind}_{unit.position}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc))


def _unit(tmp_path, **over):
    manifest = _manifest(**over)
    rd = _run(tmp_path, "dog/a/ftseed1", steps=(50,), manifest=manifest)
    _teacher(tmp_path)
    return build_unit(rd, tmp_path, "last", endpoint_only=False, extra_refs=()), manifest


def test_gate_reports_cached_for_a_current_document(tmp_path):
    unit, manifest = _unit(tmp_path)
    _write_doc(unit, _doc(unit, manifest))
    assert unit_is_cached(unit, tmp_path, 1024)


def test_gate_missing_document_is_not_cached(tmp_path):
    unit, _ = _unit(tmp_path)
    assert not unit_is_cached(unit, tmp_path, 1024)


def test_gate_force_is_never_cached(tmp_path):
    unit, manifest = _unit(tmp_path)
    _write_doc(unit, _doc(unit, manifest))
    assert not unit_is_cached(unit, tmp_path, 1024, force=True)


def test_gate_detects_a_refreshed_pool_without_reading_it(tmp_path):
    unit, manifest = _unit(tmp_path)
    doc = _doc(unit, manifest)
    doc["prompt_source"]["filtered_sha256"] = "STALE"
    _write_doc(unit, doc)
    assert not unit_is_cached(unit, tmp_path, 1024)
    assert not prompt_source_matches(doc, manifest)


def test_gate_detects_a_missing_checkpoint_point(tmp_path):
    unit, manifest = _unit(tmp_path)
    doc = _doc(unit, manifest)
    doc["points"] = doc["points"][1:]
    _write_doc(unit, doc)
    assert not unit_is_cached(unit, tmp_path, 1024)


def test_gate_detects_a_retrained_endpoint(tmp_path):
    unit, manifest = _unit(tmp_path)
    doc = _doc(unit, manifest)
    doc["points"][-1]["adapter_id"] = "OTHER"
    _write_doc(unit, doc)
    assert not unit_is_cached(unit, tmp_path, 1024)


def test_gate_detects_a_retrained_reference_teacher(tmp_path):
    unit, manifest = _unit(tmp_path)
    doc = _doc(unit, manifest, adapter_ids={_TEACHER_KEY: "OLD"})
    _write_doc(unit, doc)
    assert not unit_is_cached(unit, tmp_path, 1024)


def test_gate_ignores_a_refs_document_when_no_extra_reference_is_requested(tmp_path):
    unit, manifest = _unit(tmp_path)
    _write_doc(unit, _doc(unit, manifest))
    _write_doc(unit, {"points": []}, kind="eas_refs")  # nothing asked for it this pass
    assert unit_is_cached(unit, tmp_path, 1024)


def test_gate_detects_a_foreign_reference_in_the_main_document(tmp_path):
    # Written by the retired one-file design: the main doc must be rebuilt own-only.
    unit, manifest = _unit(tmp_path)
    other = "cat/pythia-160m/ep10_lr1e-4_r16a16/ftseed1"
    _teacher(tmp_path, other, b"cat")
    doc = _doc(unit, manifest, refs=(_TEACHER_KEY, other))
    _write_doc(unit, doc)
    assert not unit_is_cached(unit, tmp_path, 1024)


def test_gate_requires_a_current_refs_document_for_a_requested_extra(tmp_path):
    manifest = _manifest()
    rd = _run(tmp_path, "dog/a/ftseed1", manifest=manifest)
    _teacher(tmp_path)
    other = "cat/pythia-160m/ep10_lr1e-4_r16a16/ftseed1"
    _teacher(tmp_path, other, b"cat")
    unit = build_unit(rd, tmp_path, "last", endpoint_only=True, extra_refs=(other,))
    _write_doc(unit, _doc(unit, manifest))
    assert not unit_is_cached(unit, tmp_path, 1024)  # refs document missing
    _write_doc(unit, _doc(unit, manifest, refs=(other,)), kind="eas_refs")
    assert unit_is_cached(unit, tmp_path, 1024)


def test_gate_for_a_control_run_reads_the_refs_document_only(tmp_path):
    manifest = _manifest(source="base", animal=None)
    rd = _run(tmp_path, "controls/base/a/ftseed1", manifest=manifest)
    _teacher(tmp_path)
    unit = build_unit(
        rd, tmp_path, "last", endpoint_only=True, extra_refs=(_TEACHER_KEY,)
    )
    _write_doc(unit, _doc(unit, manifest))  # a main document is not the control's record
    assert not unit_is_cached(unit, tmp_path, 1024)
    _write_doc(unit, _doc(unit, manifest), kind="eas_refs")
    assert unit_is_cached(unit, tmp_path, 1024)


def test_gate_detects_a_newly_requested_reference(tmp_path):
    manifest = _manifest()
    rd = _run(tmp_path, "dog/a/ftseed1", manifest=manifest)
    _teacher(tmp_path)
    other = "cat/pythia-160m/ep10_lr1e-4_r16a16/ftseed1"
    _teacher(tmp_path, other, b"cat")
    unit = build_unit(rd, tmp_path, "last", endpoint_only=True, extra_refs=())
    _write_doc(unit, _doc(unit, manifest))
    assert unit_is_cached(unit, tmp_path, 1024)
    with_other = build_unit(
        rd, tmp_path, "last", endpoint_only=True, extra_refs=(other,)
    )
    assert not unit_is_cached(with_other, tmp_path, 1024)


def test_gate_detects_a_convention_or_position_drift(tmp_path):
    unit, manifest = _unit(tmp_path)
    doc = _doc(unit, manifest)
    doc["eas_mean_convention"] = "blocks_v1"
    _write_doc(unit, doc)
    assert not unit_is_cached(unit, tmp_path, 1024)


def test_gate_short_pool_document_stays_cached(tmp_path):
    # A run whose subsample holds fewer prompts than requested stores its ACTUAL count;
    # the manifest says so (n_train_rows), so it must not cache-miss forever.
    unit, manifest = _unit(tmp_path, n_train_rows=300)
    doc = _doc(unit, manifest)
    doc["n_prompts"] = 300
    _write_doc(unit, doc)
    assert unit_is_cached(unit, tmp_path, 1024)


def test_gate_shorter_stored_prompt_list_on_a_long_pool_is_a_miss(tmp_path):
    # Computed at --n-prompts 512 on a 10k-row subsample: a later 1024 request is a
    # DIFFERENT prompt list (different sha), so the stored document is not current.
    unit, manifest = _unit(tmp_path, n_train_rows=10000)
    doc = _doc(unit, manifest)
    doc["n_prompts"] = 512
    _write_doc(unit, doc)
    assert not unit_is_cached(unit, tmp_path, 1024)
    assert expected_n_prompts(manifest, 1024) == 1024
    assert expected_n_prompts(manifest, 20000) == 10000


def test_gate_longer_stored_prompt_list_is_a_miss(tmp_path):
    unit, manifest = _unit(tmp_path)
    doc = _doc(unit, manifest)
    doc["n_prompts"] = 4096
    _write_doc(unit, doc)
    assert not unit_is_cached(unit, tmp_path, 1024)


# ------------------------------------------------------------------- convention literal
def test_no_module_spells_the_convention_literal():
    """`EAS_MEAN_CONVENTION` is the ONE definition of the string (see steering/eas.py)."""
    repo = Path(__file__).resolve().parents[1]
    for rel in ("pipelines/eas_pipeline.py", "scripts/sweeps/sweep_compute_eas.py"):
        text = (repo / rel).read_text()
        assert f'"{EAS_MEAN_CONVENTION}"' not in text, (
            f"{rel} spells the eas_mean convention literally; import EAS_MEAN_CONVENTION"
        )


def test_gate_does_not_requeue_a_thinned_run_with_a_complete_document(tmp_path):
    """A thinned checkpoint is not a missing point: the sweep must pass `ladder_steps`.

    The document still holds the point for `checkpoint-50`, whose dir has been deleted;
    the run's ladder manifest says that step existed. Without the ladder the cheap gate
    calls the point dropped, re-enqueues the unit and the pipeline rewrites a shorter
    document -- exactly the record shrink the ladder exists to prevent.
    """
    from subliminal_learning.training.ckpt_ladder import write_ladder

    unit, manifest = _unit(tmp_path)
    _write_doc(unit, _doc(unit, manifest))
    run_dir = unit.run_dir
    write_ladder(run_dir, {"schema_version": 1, "n_train_steps": 100,
                           "checkpoint_steps": [50, 100]})
    for leaf in (run_dir / "checkpoints").glob("checkpoint-*"):
        (leaf / "adapter_model.safetensors").unlink()
        leaf.rmdir()
    thinned = build_unit(run_dir, tmp_path, "last", endpoint_only=False, extra_refs=())
    assert list(thinned.ckpt_rels) == ["checkpoints"]
    assert unit_is_cached(thinned, tmp_path, 1024)


def test_min_checkpoints_counts_ladder_steps_not_survivors(tmp_path):
    """A thinned run keeps one dir on disk but a full ladder: it must not be skipped."""
    from subliminal_learning.training.ckpt_ladder import write_ladder

    rd = _run(tmp_path, "dog/a/ftseed1")
    write_ladder(rd, {"schema_version": 1, "n_train_steps": 100,
                      "checkpoint_steps": [25, 50, 75, 100]})
    units, skipped = build_units(
        tmp_path, positions=("last",), animals=set(), models=set(), tags=set(),
        student_tags=set(), sources=set(), min_checkpoints=3, endpoint_only=False,
        extra_refs=(),
    )
    assert len(units) == 1 and skipped == []
    assert list(units[0].ckpt_rels) == ["checkpoints"]


def test_gate_reports_done_unscorable_and_does_not_re_enqueue(tmp_path):
    """A new reference at a THINNED point with no sidecar vector is terminal.

    Nothing a rerun could compute, so the unit is `DONE_UNSCORABLE` (logged once) rather
    than WORK on every pass.
    """
    from subliminal_learning.training.ckpt_ladder import write_ladder

    _teacher(tmp_path, _OTHER_KEY, b"cat")
    unit, manifest = _unit(tmp_path)
    doc = _doc(unit, manifest)
    _write_doc(unit, doc)
    _write_doc(unit, _doc(unit, manifest, refs=(_OTHER_KEY,)), kind="eas_refs")
    run_dir = unit.run_dir
    write_ladder(run_dir, {"schema_version": 1, "n_train_steps": 100,
                           "checkpoint_steps": [50, 100]})
    for leaf in (run_dir / "checkpoints").glob("checkpoint-*"):
        (leaf / "adapter_model.safetensors").unlink()
        leaf.rmdir()
    thinned = build_unit(
        run_dir, tmp_path, "last", endpoint_only=False, extra_refs=(_THIRD_KEY,)
    )
    _teacher(tmp_path, _THIRD_KEY, b"lion")
    # Before the third teacher is measured anywhere there IS work (the endpoint survives).
    assert unit_status(thinned, tmp_path, 1024) == WORK
    # The refs document now answers the third teacher everywhere it still can: only the
    # ENDPOINT, whose checkpoint survives. The thinned point is unscorable.
    refs_doc = _doc(thinned, manifest, refs=(_OTHER_KEY, _THIRD_KEY))
    refs_doc["points"] = [
        {**p, "eas": {_OTHER_KEY: {"mean": 0.1, "per_layer": [0.1]}}}
        if p["step"] == 50 else p
        for p in _doc(unit, manifest, refs=(_OTHER_KEY, _THIRD_KEY))["points"]
    ]
    _write_doc(thinned, refs_doc, kind="eas_refs")
    assert unit_status(thinned, tmp_path, 1024) == DONE_UNSCORABLE
    # A second pass sees the same terminal state (no loop) and never reports WORK.
    assert unit_status(thinned, tmp_path, 1024) == DONE_UNSCORABLE
    assert unit_is_cached(thinned, tmp_path, 1024)
