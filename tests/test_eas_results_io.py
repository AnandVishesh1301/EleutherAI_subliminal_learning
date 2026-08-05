"""Tests for the EAS results loader + the sweep driver's pure discovery/gate logic.

CPU-only, no model loads: the tag/lineage parsers and `load_eas_results` run against a tmp
students/ tree holding a valid doc, a malformed doc, and an unparseable lineage dir (both
skipped with a warning); the sweep's `valid_student_epochs` / `eas_doc_cached` /
`spec_from_model_dir` are exercised on plain dicts + names.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.sweeps.sweep_compute_eas import (
    _discover,
    eas_doc_cached,
    spec_from_model_dir,
    valid_student_epochs,
)
from subliminal_learning.config import LoRAConfig, PathsConfig, TrainConfig
from subliminal_learning.teachers.sweep_grid import shard_cells
from subliminal_learning.steering.eas_results_io import (
    RESULT_COLUMNS,
    load_eas_results,
    parse_ft_param_tag,
    parse_lineage_dir,
)


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
    # Same-base: the teacher-{base} level repeats the student model.
    assert parse_lineage_dir("teacher-pythia-160m", "ep10_lr1e-4_r16a32_tseed1") == {
        "teacher_param_tag": "ep10_lr1e-4_r16a32",
        "teacher_ftseed": 1,
        "numbers_base_model": "pythia-160m",
    }


def test_parse_lineage_dir_cross_base():
    got = parse_lineage_dir(
        "teacher-pythia-160m-data-seed1", "ep5_lr2e-4_r8a8_tseed3"
    )
    assert got["teacher_param_tag"] == "ep5_lr2e-4_r8a8"
    assert got["teacher_ftseed"] == 3
    assert got["numbers_base_model"] == "pythia-160m-data-seed1"


def test_parse_lineage_dir_bad():
    with pytest.raises(ValueError):
        parse_lineage_dir("teacher-pythia-160m", "notatag_tseed1")
    with pytest.raises(ValueError):  # teacher-base level must start with 'teacher-'
        parse_lineage_dir("notateacher", "ep10_lr1e-4_r16a32_tseed1")


# ------------------------- load_eas_results -------------------------


def _valid_doc() -> dict:
    return {
        "meta": {
            "position": "last",
            "n_prompts": 1024,
            "splice_examples": False,
            "eas_mean_convention": "blocks_v1",
        },
        "v_teacher_norm_per_layer": [0.0, 1.0, 2.0],
        "entries": [
            {
                "source": "teacher",
                "n_epochs": 5,
                "student_ftseed": 1,
                "student_param_tag": "ep5_lr1e-4_r8a32",
                "run_dir": "/x",
                "adapter_id": "abc",
                "eas_mean": 0.42,
                "eas_per_layer": [0.1, 0.2, 0.3],
                "v_student_norm_per_layer": [0.0, 1.0, 2.0],
            },
            {
                "source": "base",
                "n_epochs": 5,
                "student_ftseed": 1,
                "student_param_tag": "ep5_lr1e-4_r8a32",
                "run_dir": "/y",
                "adapter_id": "def",
                "eas_mean": -0.01,
                "eas_per_layer": [0.0, -0.1, 0.0],
                "v_student_norm_per_layer": [0.0, 1.0, 2.0],
            },
        ],
    }


def _write(
    root: Path, teacher_tag_dir: str, dataset_tag: str, position: str, text: str,
    teacher_base: str = "teacher-pythia-160m",
) -> None:
    d = (
        root / "students" / "dog" / "pythia-160m"
        / teacher_base / teacher_tag_dir / dataset_tag / "eas"
    )
    d.mkdir(parents=True, exist_ok=True)
    (d / f"eas_{position}.json").write_text(text)


def test_load_eas_results_tolerant(tmp_path):
    _write(
        tmp_path, "ep10_lr1e-4_r16a32_tseed1", "ex10-20_prefix_n100k", "last",
        json.dumps(_valid_doc()),
    )
    # Malformed JSON -> skipped with a warning.
    _write(
        tmp_path, "ep10_lr1e-4_r16a32_tseed1", "ex10-20_prefix_n100k_spliced", "last",
        "{ not json",
    )
    # Teacher-tag dir matches the glob but not the regex -> skipped with a warning.
    _write(tmp_path, "notatag_tseed1", "sometag", "last", json.dumps(_valid_doc()))

    df = load_eas_results(tmp_path)
    assert list(df.columns) == RESULT_COLUMNS
    assert len(df) == 2  # two entries from the single valid doc
    assert set(df["source"]) == {"teacher", "base"}
    row = df[df["source"] == "teacher"].iloc[0]
    assert row["animal"] == "dog"
    assert row["model"] == "pythia-160m"
    assert row["teacher_param_tag"] == "ep10_lr1e-4_r16a32"
    assert row["teacher_epochs"] == 10
    assert row["teacher_lora_r"] == 16
    assert row["teacher_lora_alpha"] == 32
    assert row["teacher_ftseed"] == 1
    assert row["dataset_tag"] == "ex10-20_prefix_n100k"
    assert row["spliced"] is False or row["spliced"] == False  # noqa: E712
    assert row["position"] == "last"
    assert row["n_prompts"] == 1024
    assert row["eas_mean_convention"] == "blocks_v1"
    assert row["student_epochs"] == 5
    assert row["eas_per_layer"] == [0.1, 0.2, 0.3]


def test_load_eas_results_skips_bad_position(tmp_path):
    # A valid doc plus a stray file whose stem parses to an unexpected position: the stray is
    # skipped defensively, not loaded as a row.
    _write(
        tmp_path, "ep10_lr1e-4_r16a32_tseed1", "ex10-20_prefix_n100k", "last",
        json.dumps(_valid_doc()),
    )
    _write(  # eas_.json -> position "" -> skipped
        tmp_path, "ep10_lr1e-4_r16a32_tseed1", "ex10-20_prefix_n100k", "",
        json.dumps(_valid_doc()),
    )
    _write(  # eas_middle.json -> position "middle" -> skipped
        tmp_path, "ep10_lr1e-4_r16a32_tseed1", "ex10-20_prefix_n100k", "middle",
        json.dumps(_valid_doc()),
    )
    df = load_eas_results(tmp_path)
    assert set(df["position"]) == {"last"}


def test_load_eas_results_warns_on_mixed_convention(tmp_path):
    from loguru import logger
    _write(
        tmp_path, "ep10_lr1e-4_r16a32_tseed1", "ex10-20_prefix_n100k", "last",
        json.dumps(_valid_doc()),
    )
    other = _valid_doc()
    other["meta"]["eas_mean_convention"] = "blocks_v2"
    _write(
        tmp_path, "ep10_lr1e-4_r16a32_tseed1", "other_tag", "last", json.dumps(other)
    )
    sink: list[str] = []
    handle = logger.add(sink.append, level="WARNING")
    try:
        df = load_eas_results(tmp_path)
    finally:
        logger.remove(handle)
    assert set(df["eas_mean_convention"]) == {"blocks_v1", "blocks_v2"}
    assert any("convention" in m for m in sink)


def test_load_eas_results_position_filter(tmp_path):
    _write(
        tmp_path, "ep10_lr1e-4_r16a32_tseed1", "tag", "last", json.dumps(_valid_doc())
    )
    _write(
        tmp_path, "ep10_lr1e-4_r16a32_tseed1", "tag", "all", json.dumps(_valid_doc())
    )
    assert set(load_eas_results(tmp_path)["position"]) == {"last", "all"}
    assert set(load_eas_results(tmp_path, positions=["last"])["position"]) == {"last"}


def test_load_eas_results_empty(tmp_path):
    (tmp_path / "students").mkdir()
    df = load_eas_results(tmp_path)
    assert list(df.columns) == RESULT_COLUMNS
    assert df.empty


def test_load_eas_results_cross_base_columns(tmp_path):
    # Same-base doc: cross_base False, numbers_base_model repeats the student model.
    _write(
        tmp_path, "ep10_lr1e-4_r16a32_tseed1", "ex10-20_prefix_n100k", "last",
        json.dumps(_valid_doc()),
    )
    # Cross-base doc (student base pythia-160m, teacher/numbers base data-seed1): loaded, not
    # skipped, with cross_base True and numbers_base_model == the teacher-base dir name.
    _write(
        tmp_path,
        "ep10_lr1e-4_r16a32_tseed1",
        "ex10-20_prefix_n100k", "last", json.dumps(_valid_doc()),
        teacher_base="teacher-pythia-160m-data-seed1",
    )
    df = load_eas_results(tmp_path)
    same = df[~df["cross_base"]]
    cross = df[df["cross_base"]]
    assert set(same["numbers_base_model"]) == {"pythia-160m"}
    assert set(cross["numbers_base_model"]) == {"pythia-160m-data-seed1"}
    assert len(cross) == 2  # two entries from the one cross doc


def test_load_eas_results_finds_ckpt_level(tmp_path):
    # A revision-pinned student gathers its model under a `ckpt/` level; the loader must find
    # it and recover model = `pythia-160m@step1000` (no bare `ckpt` leaking into any field).
    d = (
        tmp_path / "students" / "dog" / "ckpt" / "pythia-160m@step1000"
        / "teacher-pythia-160m@step1000" / "ep10_lr1e-4_r16a32_tseed1"
        / "ex10-20_prefix_n100k" / "eas"
    )
    d.mkdir(parents=True)
    (d / "eas_last.json").write_text(json.dumps(_valid_doc()))
    # A plain run alongside it, to confirm both depths coexist.
    _write(
        tmp_path, "ep10_lr1e-4_r16a32_tseed1", "ex10-20_prefix_n100k", "last",
        json.dumps(_valid_doc()),
    )
    df = load_eas_results(tmp_path)
    assert set(df["model"]) == {"pythia-160m@step1000", "pythia-160m"}
    ck = df[df["model"] == "pythia-160m@step1000"].iloc[0]
    assert ck["animal"] == "dog"
    assert ck["numbers_base_model"] == "pythia-160m@step1000"
    assert ck["cross_base"] == False  # noqa: E712
    assert ck["teacher_param_tag"] == "ep10_lr1e-4_r16a32"
    assert ck["dataset_tag"] == "ex10-20_prefix_n100k"


# ------------------------- sweep pure logic -------------------------


def test_valid_student_epochs_splits_default_hp():
    tags = ["ep1_lr1e-4_r8a32", "ep5_lr1e-4_r8a32", "ep5_lr2e-4_r8a32", "junk"]
    epochs, ignored = valid_student_epochs(tags, lr=1e-4, r=8, alpha=32)
    assert epochs == [1, 5]
    assert set(ignored) == {"ep5_lr2e-4_r8a32", "junk"}


def test_eas_doc_cached_gate():
    doc = {
        "meta": {"eas_mean_convention": "blocks_v1", "n_prompts": 1024, "position": "last"},
        "entries": [
            {"source": "teacher", "n_epochs": 5, "student_ftseed": 1},
            {"source": "base", "n_epochs": 5, "student_ftseed": 1},
        ],
    }
    req = {("teacher", 5, 1)}
    assert eas_doc_cached(doc, position="last", n_prompts=1024, required_work_items=req)
    # missing a required entry
    assert not eas_doc_cached(
        doc, position="last", n_prompts=1024,
        required_work_items={("teacher", 5, 1), ("rng", 5, 1)},
    )
    # an EMPTY required set is "nothing to do", NOT cached
    assert not eas_doc_cached(doc, position="last", n_prompts=1024, required_work_items=set())
    # wrong meta
    assert not eas_doc_cached(doc, position="all", n_prompts=1024, required_work_items=req)
    assert not eas_doc_cached(doc, position="last", n_prompts=512, required_work_items=req)
    stale = {**doc, "meta": {**doc["meta"], "eas_mean_convention": "old"}}
    assert not eas_doc_cached(stale, position="last", n_prompts=1024, required_work_items=req)


def test_eas_doc_cached_teacher_adapter_and_pool_drift():
    doc = {
        "meta": {
            "eas_mean_convention": "blocks_v1", "n_prompts": 1024, "position": "last",
            "teacher_adapter_id": "aaaa", "filtered_sha256": "poolhash",
        },
        "entries": [{"source": "teacher", "n_epochs": 5, "student_ftseed": 1}],
    }
    req = {("teacher", 5, 1)}
    # Matching adapter + pool -> cached.
    assert eas_doc_cached(
        doc, position="last", n_prompts=1024, required_work_items=req,
        teacher_adapter_id="aaaa", filtered_sha256="poolhash",
    )
    # In-place teacher retrain: the on-disk adapter id no longer matches meta -> not cached.
    assert not eas_doc_cached(
        doc, position="last", n_prompts=1024, required_work_items=req,
        teacher_adapter_id="bbbb", filtered_sha256="poolhash",
    )
    # Refreshed number pool: filtered.jsonl hash changed -> not cached.
    assert not eas_doc_cached(
        doc, position="last", n_prompts=1024, required_work_items=req,
        teacher_adapter_id="aaaa", filtered_sha256="newpool",
    )
    # A None current value skips that check (a missing teacher ckpt / unresolved pool).
    assert eas_doc_cached(
        doc, position="last", n_prompts=1024, required_work_items=req,
        teacher_adapter_id=None, filtered_sha256=None,
    )


def test_eas_doc_cached_short_pool_n_prompts_warns():
    # A short-pool lineage stores its ACTUAL prompt count; when that is the SOLE remaining
    # cache-miss cause the gate warns by name (otherwise it cache-misses forever silently).
    from loguru import logger
    doc = {
        "meta": {"eas_mean_convention": "blocks_v1", "n_prompts": 300, "position": "last"},
        "entries": [{"source": "teacher", "n_epochs": 5, "student_ftseed": 1}],
    }
    req = {("teacher", 5, 1)}
    sink: list[str] = []
    handle = logger.add(sink.append, level="WARNING")
    try:
        cached = eas_doc_cached(
            doc, position="last", n_prompts=1024, required_work_items=req, label="dog/lineage",
        )
    finally:
        logger.remove(handle)
    assert cached is False
    assert any("n_prompts" in m and "dog/lineage" in m for m in sink)


def test_spec_from_model_dir():
    assert spec_from_model_dir("pythia-160m", "EleutherAI").variant == "full"
    ds1 = spec_from_model_dir("pythia-160m-data-seed1", "EleutherAI")
    assert (ds1.variant, ds1.seed) == ("data", 1)
    with pytest.raises(ValueError):
        spec_from_model_dir("pythia-410m", "EleutherAI")
    with pytest.raises(ValueError):
        spec_from_model_dir("not-a-model", "EleutherAI")


def test_spec_from_model_dir_revision_pinned():
    # Bare-original ckpt form (the v1 sweep's `base_model: original`) and the variant ckpt form.
    orig = spec_from_model_dir("pythia-160m@step1000", "EleutherAI")
    assert (orig.variant, orig.seed, orig.revision) == ("full", 0, "step1000")
    ds1 = spec_from_model_dir("pythia-160m-data-seed1@step3000", "EleutherAI")
    assert (ds1.variant, ds1.seed, ds1.revision) == ("data", 1, "step3000")
    # A revisionless original still has revision None (byte-identical to before).
    assert spec_from_model_dir("pythia-160m", "EleutherAI").revision is None
    with pytest.raises(ValueError):  # empty revision after '@'
        spec_from_model_dir("pythia-160m@", "EleutherAI")


# ------------------------- sweep unit sharding -------------------------


def test_eas_shards_partition_units_disjoint_exhaustive_and_balanced():
    # Synthetic sorted unit list; the sweep strides this exact list with shard_cells.
    units = list(range(812))
    n = 24
    shards = [shard_cells(units, num_shards=n, shard_index=i) for i in range(n)]
    # exhaustive: every unit lands in exactly one shard
    assert sorted(u for shard in shards for u in shard) == units
    # disjoint: no unit appears in two shards
    seen: set[int] = set()
    for shard in shards:
        assert not (seen & set(shard))
        seen |= set(shard)
    # balanced to within one unit
    sizes = [len(shard) for shard in shards]
    assert max(sizes) - min(sizes) <= 1


@pytest.mark.parametrize(
    ("num_shards", "shard_index"), [(0, 0), (1, 1), (3, -1), (3, 3)]
)
def test_eas_shard_params_validate(num_shards, shard_index):
    with pytest.raises(ValueError):
        shard_cells(list(range(5)), num_shards=num_shards, shard_index=shard_index)


# ------------------------- sweep cross-base discovery -------------------------


def _sweep_cfg(output_root: Path) -> SimpleNamespace:
    """A minimal template-like config the sweep's `_discover` reads (paths/hub_org/train/lora)."""
    paths = PathsConfig(
        output_root=output_root, hf_cache_root=output_root / "cache",
        dataset_cache_root=output_root / "data", wandb_dir=output_root / "wandb",
    )
    lora = LoRAConfig(r=8, lora_alpha=32, lora_dropout=0.0, target_modules="all-linear")
    train = TrainConfig(
        learning_rate=1e-4, n_epochs=5, batch_size=8, max_length=512, max_grad_norm=1.0,
        warmup_ratio=0.05, lr_scheduler_type="cosine", seed=1,
    )
    return SimpleNamespace(paths=paths, hub_org="EleutherAI", train=train, lora=lora)


def _make_cross_student(output_root: Path) -> None:
    """A single teacher-source cross-base student cell (data-seed1 base, numbers base original)."""
    rd = (
        output_root / "students" / "dog" / "pythia-160m-data-seed1"
        / "teacher-pythia-160m" / "ep10_lr1e-4_r16a32_tseed1"
        / "ex10-20_prefix_n100k" / "ep5_lr1e-4_r8a32" / "ftseed1"
    )
    (rd / "checkpoints").mkdir(parents=True)
    (rd / "train_manifest.json").write_text(json.dumps({
        "splice_examples": False,
        "cross_base": True,
        "numbers_base_model_repo": "EleutherAI/pythia-160m",
    }))


def test_discover_skips_cross_base_by_default(tmp_path):
    _make_cross_student(tmp_path)
    cfg = _sweep_cfg(tmp_path)
    units, n_unsupported = _discover(
        cfg, sources=("teacher",), positions=("last",),
        animals=set(), models=set(), tags=set(), n_prompts=1024,
    )
    assert units == []
    assert n_unsupported == 1


def test_discover_includes_cross_base_when_requested(tmp_path):
    _make_cross_student(tmp_path)
    cfg = _sweep_cfg(tmp_path)
    units, n_unsupported = _discover(
        cfg, sources=("teacher",), positions=("last",),
        animals=set(), models=set(), tags=set(), n_prompts=1024,
        include_cross_base=True,
    )
    assert n_unsupported == 0
    assert len(units) == 1
    unit = units[0]
    assert unit.model == "pythia-160m-data-seed1"
    assert unit.numbers_base_model == "pythia-160m"
    assert (unit.numbers_spec.variant, unit.numbers_spec.seed) == ("full", 0)
    assert unit.spec.variant == "data" and unit.spec.seed == 1
    # the teacher student cell is discovered as a required work item
    assert ("teacher", 5, 1) in unit.required_work_items
    assert "teacher-pythia-160m" in unit.label


def _make_ckpt_student(output_root: Path) -> None:
    """A same-base revision-pinned student: the model is gathered under a `ckpt/` level and
    the teacher-base repeats the (revision-qualified) model as one component."""
    rd = (
        output_root / "students" / "dog" / "ckpt" / "pythia-160m-data-seed1@step3000"
        / "teacher-pythia-160m-data-seed1@step3000" / "ep10_lr1e-4_r16a32_tseed1"
        / "ex10-20_prefix_n100k" / "ep5_lr1e-4_r8a32" / "ftseed1"
    )
    (rd / "checkpoints").mkdir(parents=True)
    (rd / "train_manifest.json").write_text(json.dumps({
        "splice_examples": False,
        "numbers_base_model_repo": "EleutherAI/pythia-160m-data-seed1@step3000",
    }))


def _make_plain_student(output_root: Path) -> None:
    """A plain (revision-less) same-base student, byte-identical to the historical layout."""
    rd = (
        output_root / "students" / "dog" / "pythia-160m-data-seed1"
        / "teacher-pythia-160m-data-seed1" / "ep10_lr1e-4_r16a32_tseed1"
        / "ex10-20_prefix_n100k" / "ep5_lr1e-4_r8a32" / "ftseed1"
    )
    (rd / "checkpoints").mkdir(parents=True)
    (rd / "train_manifest.json").write_text(json.dumps({
        "splice_examples": False,
        "numbers_base_model_repo": "EleutherAI/pythia-160m-data-seed1",
    }))


def test_discover_finds_ckpt_students(tmp_path):
    # A plain same-base student and a revision-pinned (ckpt-level) same-base student: both are
    # discovered, model/animal derive correctly with the @rev preserved, and no bare `ckpt`
    # leaks into any field.
    _make_plain_student(tmp_path)
    _make_ckpt_student(tmp_path)
    cfg = _sweep_cfg(tmp_path)
    units, n_unsupported = _discover(
        cfg, sources=("teacher",), positions=("last",),
        animals=set(), models=set(), tags=set(), n_prompts=1024,
    )
    assert n_unsupported == 0
    by_model = {u.model: u for u in units}
    assert set(by_model) == {"pythia-160m-data-seed1", "pythia-160m-data-seed1@step3000"}
    ck = by_model["pythia-160m-data-seed1@step3000"]
    assert ck.animal == "dog"
    assert ck.numbers_base_model == "pythia-160m-data-seed1@step3000"
    assert ck.numbers_spec is None  # same-base, not cross
    assert (ck.spec.variant, ck.spec.seed, ck.spec.revision) == ("data", 1, "step3000")
    assert ck.dataset_tag == "ex10-20_prefix_n100k"
    assert ck.teacher_param_tag == "ep10_lr1e-4_r16a32"


def _make_bare_original_ckpt_student(output_root: Path) -> None:
    """The v1 sweep's `base_model: original` ckpt student: a bare-original `pythia-160m@{rev}`
    base, gathered under a `ckpt/` level, distilling its own-checkpoint teacher (same-base)."""
    rd = (
        output_root / "students" / "dog" / "ckpt" / "pythia-160m@step1000"
        / "teacher-pythia-160m@step1000" / "ep10_lr1e-4_r16a32_tseed1"
        / "ex10-20_prefix_n100k" / "ep5_lr1e-4_r8a32" / "ftseed1"
    )
    (rd / "checkpoints").mkdir(parents=True)
    (rd / "train_manifest.json").write_text(json.dumps({
        "splice_examples": False,
        "numbers_base_model_repo": "EleutherAI/pythia-160m@step1000",
    }))


def test_discover_finds_bare_original_ckpt_student(tmp_path):
    # The bare-original ckpt form must be fully discovered (not skipped-unsupported): its spec
    # inverts via spec_from_model_dir with revision set, and the model keeps its @rev form.
    _make_bare_original_ckpt_student(tmp_path)
    cfg = _sweep_cfg(tmp_path)
    units, n_unsupported = _discover(
        cfg, sources=("teacher",), positions=("last",),
        animals=set(), models=set(), tags=set(), n_prompts=1024,
    )
    assert n_unsupported == 0
    assert len(units) == 1
    unit = units[0]
    assert unit.model == "pythia-160m@step1000"
    assert unit.animal == "dog"
    assert unit.numbers_base_model == "pythia-160m@step1000"
    assert unit.numbers_spec is None  # same-base, not cross
    assert (unit.spec.variant, unit.spec.seed, unit.spec.revision) == ("full", 0, "step1000")


def test_discover_animal_and_model_filters_apply_under_ckpt(tmp_path):
    # The animal/model selectors read the ckpt-normalised fields, so they filter ckpt students
    # by their true animal/model rather than the `ckpt` sentinel.
    _make_ckpt_student(tmp_path)
    cfg = _sweep_cfg(tmp_path)
    kept, _ = _discover(
        cfg, sources=("teacher",), positions=("last",),
        animals={"dog"}, models={"pythia-160m-data-seed1@step3000"}, tags=set(), n_prompts=1024,
    )
    assert len(kept) == 1
    dropped, _ = _discover(
        cfg, sources=("teacher",), positions=("last",),
        animals={"ckpt"}, models=set(), tags=set(), n_prompts=1024,
    )
    assert dropped == []
