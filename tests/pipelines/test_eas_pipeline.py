"""EAS identity/reference tests (CPU-only, no model loads).

Covers the parts of `subliminal_learning.pipelines.eas_pipeline` that decide WHAT a run is measured against:
the run's own teacher as the default reference, the `ref_key` grammar, the cross-base flag
in the stored `student` block, and the reference resolution that drops a teacher with no
adapter and skips a control run given no reference at all.

The EAS two-document layout (CPU-only, no model loads).

`subliminal_learning.pipelines.eas_pipeline` writes the run's OWN teacher into ``eval/eas_{position}.json`` and
every EXTRA ``--reference-teacher`` into ``eval/eas_refs_{position}.json`` beside it. These
tests drive a real `_run` over a tiny tmp_path tree with the model-touching calls stubbed
(the session's cached vectors, the checkpoint adapter load and `mean_activations`), so the
FILE LAYOUT and the reference bookkeeping are exercised end to end:

- an extra reference is stored in the refs document only, and a later narrower pass keeps it
  (the stored set is a union, never narrowed);
- a main document carrying a foreign reference (the retired one-file design) is invalid and
  is rebuilt own-only, cosine-only off the sidecar;
- a control run (no own teacher) gets the refs document and NO main document.
"""

from __future__ import annotations

from pathlib import Path
import json

import pytest
import torch

from subliminal_learning.config import PathsConfig
from subliminal_learning.pipelines.eas_pipeline import (
    EASRunConfig,
    doc_path,
    refs_doc_path,
    sidecar_path,
)
from subliminal_learning.pipelines.eas_pipeline import (
    EASSkip,
    default_reference_key,
    parse_reference_key,
    resolve_references,
    student_identity,
)
from subliminal_learning.representation.steering.eas import eas_main_doc_status
from subliminal_learning.utils.io import file_sha256
import subliminal_learning.pipelines.eas_pipeline as eas_pipeline


# -------------------- from test_eas_cross_base.py --------------------

_PYTHIA = "EleutherAI/pythia-160m"

_DS1 = "EleutherAI/pythia-160m-data-seed1"

def _manifest(**over) -> dict:
    m = {
        "source": "teacher",
        "animal": "dog",
        "base_model": "pythia-160m",
        "base_model_repo": _PYTHIA,
        "numbers_base_model_repo": _PYTHIA,
        "cross_base": False,
        "student_param_tag": "ep20_lr4e-4_r16a4",
        "student_ftseed": 1,
        "number_dataset_tag": "ex10-20_prefix_n100k",
        "teacher_param_tag": "ep10_lr1e-4_r16a16",
        "teacher_ftseed": 1,
        "filtered_sha256": "deadbeef",
        "subsample_size": 10000,
        "subsample_seed": 42,
        "n_train_steps": 100,
    }
    m.update(over)
    return m

# ------------------------------------------------------------------- reference keys
def test_default_reference_key_is_the_runs_own_teacher():
    assert default_reference_key(_manifest()) == (
        "dog/pythia-160m/ep10_lr1e-4_r16a16/ftseed1"
    )

def test_default_reference_key_uses_the_NUMBERS_base_for_a_cross_base_run():
    # Student on data-seed1, distilled from numbers a teacher on the ORIGINAL generated:
    # the teacher lives under the numbers base's model level, not the student's.
    m = _manifest(base_model_repo=_DS1, numbers_base_model_repo=_PYTHIA, cross_base=True)
    assert default_reference_key(m) == "dog/pythia-160m/ep10_lr1e-4_r16a16/ftseed1"

def test_default_reference_key_gathers_a_revision_pinned_base_under_ckpt():
    # The teachers tree puts a revision-pinned base under a `ckpt/` level, so the key must
    # too or it names a directory that does not exist.
    m = _manifest(numbers_base_model_repo="EleutherAI/pythia-160m@step3000")
    key = default_reference_key(m)
    assert key == "dog/ckpt/pythia-160m@step3000/ep10_lr1e-4_r16a16/ftseed1"
    assert parse_reference_key(key) == {
        "animal": "dog", "base_model_dir": "pythia-160m@step3000",
        "teacher_param_tag": "ep10_lr1e-4_r16a16", "teacher_ftseed": 1,
    }

def test_default_reference_key_is_none_for_controls():
    assert default_reference_key(_manifest(source="base", animal=None)) is None
    assert default_reference_key(_manifest(source="rng", animal=None)) is None

def test_parse_reference_key_plain_and_ckpt_levels():
    assert parse_reference_key("dog/pythia-160m/ep10_lr1e-4_r16a16/ftseed2") == {
        "animal": "dog", "base_model_dir": "pythia-160m",
        "teacher_param_tag": "ep10_lr1e-4_r16a16", "teacher_ftseed": 2,
    }
    assert parse_reference_key(
        "dog/ckpt/pythia-160m@step1000/ep10_lr1e-4_r16a16/ftseed1"
    )["base_model_dir"] == "pythia-160m@step1000"

@pytest.mark.parametrize("bad", ["dog/pythia-160m", "dog/pythia-160m/tag/seed1"])
def test_parse_reference_key_rejects_non_run_dirs(bad):
    with pytest.raises(ValueError):
        parse_reference_key(bad)

# ------------------------------------------------------------------- student block
def test_student_identity_same_base(tmp_path):
    ident = student_identity(tmp_path / "students" / "r", tmp_path, _manifest())
    assert ident["cross_base"] is False
    assert ident["base_model_dir"] == "pythia-160m"
    assert ident["numbers_base_model_dir"] == "pythia-160m"
    assert ident["run_rel"] == "students/r"

def test_student_identity_cross_base_records_both_bases(tmp_path):
    m = _manifest(base_model_repo=_DS1, numbers_base_model_repo=_PYTHIA, cross_base=True)
    ident = student_identity(tmp_path / "students" / "r", tmp_path, m)
    assert ident["cross_base"] is True
    assert ident["base_model_dir"] == "pythia-160m-data-seed1"
    assert ident["numbers_base_model_dir"] == "pythia-160m"

# ------------------------------------------------------------------- resolution
def _teacher(root: Path, key: str, payload: bytes = b"w") -> Path:
    d = root / "teachers" / key / "checkpoints"
    d.mkdir(parents=True)
    (d / "adapter_model.safetensors").write_bytes(payload)
    return d.parent

def test_resolve_references_default_plus_extra(tmp_path):
    own = "dog/pythia-160m/ep10_lr1e-4_r16a16/ftseed1"
    other = "cat/pythia-160m/ep10_lr1e-4_r16a16/ftseed1"
    _teacher(tmp_path, own)
    _teacher(tmp_path, other, b"other")
    refs = resolve_references(_manifest(), tmp_path, (other,))
    assert list(refs) == [own, other]
    assert refs[own]["teacher_base_repo"] == _PYTHIA
    assert refs[own]["teacher_run_rel"] == f"teachers/{own}"
    assert refs[own]["teacher_adapter_id"] != refs[other]["teacher_adapter_id"]

def test_resolve_references_drops_a_teacher_with_no_adapter(tmp_path):
    own = "dog/pythia-160m/ep10_lr1e-4_r16a16/ftseed1"
    _teacher(tmp_path, own)
    refs = resolve_references(_manifest(), tmp_path, ("cat/pythia-160m/t/ftseed1",))
    assert list(refs) == [own]

def test_resolve_references_control_without_a_reference_is_skipped(tmp_path):
    with pytest.raises(EASSkip, match="reference"):
        resolve_references(_manifest(source="base", animal=None), tmp_path, ())

def test_resolve_references_control_with_an_explicit_reference(tmp_path):
    key = "dog/pythia-160m/ep10_lr1e-4_r16a16/ftseed1"
    _teacher(tmp_path, key)
    refs = resolve_references(_manifest(source="base", animal=None), tmp_path, (key,))
    assert list(refs) == [key] and refs[key]["animal"] == "dog"


# -------------------- from test_eas_two_files.py --------------------

OWN = "dog/pythia-160m/ep10_lr1e-4_r16a16/ftseed1"

OTHER = "cat/pythia-160m/ep10_lr1e-4_r16a16/ftseed1"

THIRD = "lion/pythia-160m/ep10_lr1e-4_r16a16/ftseed1"

_H, _SLOTS = 4, 3

# ------------------------------------------------------------------- stubs
class _FakeModel:
    """Stands in for the PEFT-wrapped student: only the adapter bookkeeping is used."""

    def set_adapter(self, name):
        pass

    def delete_adapter(self, name):
        pass

    def unload(self):
        return self

class _FakeSession:
    """`EASSession` surface `_run` uses, with every model/cache call a fixed tensor."""

    def __init__(self, seed: int = 0):
        self.seed = seed

    def teacher_vector(self, teacher_run_dir, *args, **kwargs):
        g = torch.Generator().manual_seed(abs(hash(str(teacher_run_dir))) % 10_000)
        raw = torch.randn(_SLOTS, _H, generator=g)
        return {"raw": raw, "norm": raw.norm(dim=-1)}

    def base_mean(self, *args, **kwargs):
        return torch.zeros(_SLOTS, _H)

    def base_for(self, ref):
        return object(), object()

    def adopt_base(self, base):
        pass

@pytest.fixture(autouse=True)
def _stub_model_calls(monkeypatch):
    monkeypatch.setattr(
        eas_pipeline, "_load_ckpt_adapter", lambda model, base, path, name: (_FakeModel(), True)
    )
    monkeypatch.setattr(
        eas_pipeline, "mean_activations",
        lambda model, tok, prompts, device, batch_size, position: torch.arange(
            float(_SLOTS * _H)
        ).reshape(_SLOTS, _H),
    )

# ------------------------------------------------------------------- tree fixtures
def _teacher_run(root: Path, key: str, payload: bytes = b"w") -> None:
    d = root / "teachers" / key / "checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    (d / "adapter_model.safetensors").write_bytes(payload)

def _pool(root: Path) -> tuple[Path, str]:
    d = root / "numbers" / "pool"
    d.mkdir(parents=True, exist_ok=True)
    path = d / "filtered.jsonl"
    path.write_text(
        "".join(json.dumps({"prompt": f"p{i}", "completion": "1"}) + "\n" for i in range(8))
    )
    return d, file_sha256(path)

def _run_dir(root: Path, source: str = "teacher") -> Path:
    pool, sha = _pool(root)
    rd = root / "students" / "r"
    (rd / "checkpoints").mkdir(parents=True, exist_ok=True)
    (rd / "checkpoints" / "adapter_model.safetensors").write_bytes(b"student")
    manifest = {
        "source": source,
        "animal": "dog" if source == "teacher" else None,
        "base_model_repo": _PYTHIA,
        "numbers_base_model_repo": _PYTHIA,
        "cross_base": False,
        "student_param_tag": "ep20_lr4e-4_r16a4",
        "student_ftseed": 1,
        "number_dataset_tag": "ex10-20_prefix_n100k",
        "number_dataset_dir": str(pool),
        "filtered_sha256": sha,
        "subsample_size": 8,
        "subsample_seed": 42,
        "teacher_param_tag": "ep10_lr1e-4_r16a16",
        "teacher_ftseed": 1,
        "n_train_steps": 100,
    }
    (rd / "train_manifest.json").write_text(json.dumps(manifest))
    return rd

def _cfg(
    root: Path, rd: Path, references=(), force: bool = False, n_prompts: int = 8
) -> EASRunConfig:
    paths = PathsConfig(
        output_root=root, hf_cache_root=root / "hf", dataset_cache_root=root / "ds",
        wandb_dir=root / "wandb",
    )
    return EASRunConfig(
        paths=paths, hub_org="org", run_dir=rd, references=tuple(references),
        position="last", n_prompts=n_prompts, batch_size=4, device="cpu", force=force,
    )

def _go(root: Path, rd: Path, references=(), force: bool = False, n_prompts: int = 8):
    return eas_pipeline._run(
        _cfg(root, rd, references, force, n_prompts), _FakeSession()
    )

# ------------------------------------------------------------------- the layout
def test_own_teacher_goes_to_the_main_doc_and_extras_to_the_refs_doc(tmp_path):
    _teacher_run(tmp_path, OWN)
    _teacher_run(tmp_path, OTHER, b"cat")
    rd = _run_dir(tmp_path)
    _go(tmp_path, rd, references=(OTHER,))

    main = json.loads(doc_path(rd, "last").read_text())
    refs = json.loads(refs_doc_path(rd, "last").read_text())
    assert list(main["references"]) == [OWN]
    assert list(main["points"][0]["eas"]) == [OWN]
    assert main["student"]["own_reference"] == OWN
    assert list(refs["references"]) == [OTHER]
    assert list(refs["points"][0]["eas"]) == [OTHER]
    assert sidecar_path(rd, "last").is_file()

def test_a_pass_with_no_extra_refs_writes_no_refs_doc(tmp_path):
    _teacher_run(tmp_path, OWN)
    rd = _run_dir(tmp_path)
    _go(tmp_path, rd)
    assert doc_path(rd, "last").is_file()
    assert not refs_doc_path(rd, "last").exists()

def test_a_narrower_rerun_keeps_the_stored_extra_references(tmp_path):
    _teacher_run(tmp_path, OWN)
    _teacher_run(tmp_path, OTHER, b"cat")
    _teacher_run(tmp_path, THIRD, b"lion")
    rd = _run_dir(tmp_path)
    _go(tmp_path, rd, references=(OTHER, THIRD))
    assert set(json.loads(refs_doc_path(rd, "last").read_text())["references"]) == {
        OTHER, THIRD
    }
    # A later pass naming only ONE of them must not drop the other (stored ∪ requested)...
    _go(tmp_path, rd, references=(OTHER,))
    refs = json.loads(refs_doc_path(rd, "last").read_text())
    assert set(refs["references"]) == {OTHER, THIRD}
    assert set(refs["points"][0]["eas"]) == {OTHER, THIRD}
    # ... and neither does --force, which rebuilds the points, not the reference set.
    _go(tmp_path, rd, references=(OTHER,), force=True)
    assert set(json.loads(refs_doc_path(rd, "last").read_text())["references"]) == {
        OTHER, THIRD
    }

def test_a_main_doc_with_a_foreign_reference_is_rebuilt_own_only(tmp_path):
    _teacher_run(tmp_path, OWN)
    _teacher_run(tmp_path, OTHER, b"cat")
    rd = _run_dir(tmp_path)
    _go(tmp_path, rd)
    # A document written by the retired one-file design: own + a foreign reference.
    stale = json.loads(doc_path(rd, "last").read_text())
    stale["references"][OTHER] = {"teacher_adapter_id": "x", "teacher_param_tag": "t"}
    for point in stale["points"]:
        point["eas"][OTHER] = {"mean": 0.5, "per_layer": [0.5]}
    doc_path(rd, "last").write_text(json.dumps(stale))

    plan = eas_main_doc_status(
        stale, expected_meta={k: stale[k] for k in (
            "schema_version", "basis", "eas_mean_convention", "position", "n_prompts",
            "prompt_sha256")},
        checkpoints=[(p["ckpt_rel"], p["adapter_id"]) for p in stale["points"]],
        references={OWN: stale["references"][OWN]["teacher_adapter_id"]},
        sidecar_ckpts=frozenset(p["ckpt_rel"] for p in stale["points"]),
        manifest_ref=stale["student"]["manifest_ref"],
    )
    assert plan.invalid and "own teacher" in plan.reason
    # The sidecar holds the student vectors, so the migration is cosine-only.
    assert plan.recompute == () and len(plan.cosine_only) == len(stale["points"])

    _go(tmp_path, rd)
    rebuilt = json.loads(doc_path(rd, "last").read_text())
    assert list(rebuilt["references"]) == [OWN]
    assert all(list(p["eas"]) == [OWN] for p in rebuilt["points"])

def test_a_control_run_writes_only_the_refs_doc(tmp_path):
    _teacher_run(tmp_path, OWN)
    rd = _run_dir(tmp_path, source="base")
    doc = _go(tmp_path, rd, references=(OWN,))
    assert not doc_path(rd, "last").exists()
    refs = json.loads(refs_doc_path(rd, "last").read_text())
    assert doc == refs
    assert list(refs["references"]) == [OWN]
    assert refs["student"]["own_reference"] is None

def test_a_current_pair_of_documents_is_skipped(tmp_path):
    _teacher_run(tmp_path, OWN)
    _teacher_run(tmp_path, OTHER, b"cat")
    rd = _run_dir(tmp_path)
    _go(tmp_path, rd, references=(OTHER,))
    before = (
        doc_path(rd, "last").read_text(), refs_doc_path(rd, "last").read_text()
    )
    _go(tmp_path, rd, references=(OTHER,))
    assert (doc_path(rd, "last").read_text(), refs_doc_path(rd, "last").read_text()) == before

def test_sidecar_keeps_the_vectors_of_thinned_checkpoints(tmp_path):
    """A thinned checkpoint's student vector is the only copy left; the prune must keep it.

    Keying the prune on the on-disk `ckpt_rel`s deleted it, so a later pass adding a
    reference could neither reuse it (no vector) nor recompute it (no checkpoint).
    """
    from subliminal_learning.training.ckpt_ladder import write_ladder

    _teacher_run(tmp_path, OWN)
    _teacher_run(tmp_path, OTHER, b"cat")
    rd = _run_dir(tmp_path)
    ckpt = rd / "checkpoints" / "checkpoint-50"
    ckpt.mkdir()
    (ckpt / "adapter_model.safetensors").write_bytes(b"mid")
    _go(tmp_path, rd)
    side = torch.load(sidecar_path(rd, "last"), weights_only=False)
    assert set(side["points"]) == {"checkpoints/checkpoint-50", "checkpoints"}

    # Thin the ladder, then run again for an extra reference.
    (ckpt / "adapter_model.safetensors").unlink()
    ckpt.rmdir()
    write_ladder(rd, {"schema_version": 1, "n_train_steps": 100,
                      "checkpoint_steps": [50, 100]})
    _go(tmp_path, rd, references=(OTHER,))
    side = torch.load(sidecar_path(rd, "last"), weights_only=False)
    assert set(side["points"]) == {"checkpoints/checkpoint-50", "checkpoints"}

# ------------------------------------------------------------------- thinned runs
def _thin(rd: Path, ckpt: Path, steps=(50, 100)) -> None:
    """Delete an intermediate checkpoint dir and record the full ladder manifest."""
    from subliminal_learning.training.ckpt_ladder import write_ladder

    (ckpt / "adapter_model.safetensors").unlink()
    ckpt.rmdir()
    write_ladder(rd, {"schema_version": 1, "n_train_steps": 100,
                      "checkpoint_steps": list(steps)})

def _with_mid(root: Path) -> tuple[Path, Path]:
    rd = _run_dir(root)
    ckpt = rd / "checkpoints" / "checkpoint-50"
    ckpt.mkdir()
    (ckpt / "adapter_model.safetensors").write_bytes(b"mid")
    return rd, ckpt

def test_force_on_a_thinned_run_keeps_the_thinned_point_and_its_vector(tmp_path):
    """`--force` must recompute, never DISCARD: the step-50 point cannot be recomputed."""
    _teacher_run(tmp_path, OWN)
    rd, ckpt = _with_mid(tmp_path)
    _go(tmp_path, rd)
    _thin(rd, ckpt)

    _go(tmp_path, rd, force=True)
    doc = json.loads(doc_path(rd, "last").read_text())
    assert [p["step"] for p in doc["points"]] == [50, 100]
    side = torch.load(sidecar_path(rd, "last"), weights_only=False)
    assert "checkpoints/checkpoint-50" in side["points"]
    # Nothing shrank, so nothing was moved aside.
    assert list((rd / "eval").glob("*.superseded-*")) == []

def test_force_after_a_prompt_set_change_drops_the_thinned_point(tmp_path):
    """A `DOC_META_KEYS` drift (n_prompts) invalidates the plan: nothing is carried."""
    _teacher_run(tmp_path, OWN)
    rd, ckpt = _with_mid(tmp_path)
    _go(tmp_path, rd)
    _thin(rd, ckpt)

    # The prompt set changed, so the stored step-50 point was measured under other prompts.
    _go(tmp_path, rd, force=True, n_prompts=4)
    doc = json.loads(doc_path(rd, "last").read_text())
    assert [p["step"] for p in doc["points"]] == [100]

def test_force_after_an_identity_change_supersedes_the_doc_and_sidecar_together(tmp_path):
    """A retrained run cannot carry the old points forward; the pair is preserved instead."""
    _teacher_run(tmp_path, OWN)
    rd, ckpt = _with_mid(tmp_path)
    _go(tmp_path, rd)
    _thin(rd, ckpt)
    # The run was retrained in place for longer: its `manifest_ref` moves, so the stored
    # points describe other weights and may NOT be carried forward.
    manifest = json.loads((rd / "train_manifest.json").read_text())
    manifest["n_train_steps"] = 120
    (rd / "train_manifest.json").write_text(json.dumps(manifest))

    _go(tmp_path, rd, force=True)
    doc = json.loads(doc_path(rd, "last").read_text())
    assert [p["step"] for p in doc["points"]] == [120]
    moved = sorted(p.name.split(".superseded-")[0] for p in (rd / "eval").glob("*.superseded-*"))
    assert moved == ["eas_last.json", "eas_vectors_last.pt"]

def test_a_new_reference_on_a_thinned_point_is_scored_from_the_sidecar(tmp_path):
    """Cosine-only over a rel with NO checkpoint dir: sidecar vector + teacher vector only."""
    _teacher_run(tmp_path, OWN)
    _teacher_run(tmp_path, OTHER, b"cat")
    rd, ckpt = _with_mid(tmp_path)
    _go(tmp_path, rd, references=(OTHER,))
    _thin(rd, ckpt)

    # A THIRD reference arrives after the thinning; step 50 has only its stored vector.
    _teacher_run(tmp_path, THIRD, b"lion")
    _go(tmp_path, rd, references=(OTHER, THIRD))
    refs = json.loads(refs_doc_path(rd, "last").read_text())
    thinned = next(p for p in refs["points"] if p["step"] == 50)
    assert set(thinned["eas"]) == {OTHER, THIRD}
    assert [p["step"] for p in refs["points"]] == [50, 100]

def test_a_new_reference_with_no_sidecar_vector_is_annotated_unscorable(tmp_path):
    """No checkpoint and no vector: the point is kept, annotated, and never recomputed."""
    _teacher_run(tmp_path, OWN)
    _teacher_run(tmp_path, OTHER, b"cat")
    rd, ckpt = _with_mid(tmp_path)
    _go(tmp_path, rd, references=(OTHER,))
    _thin(rd, ckpt)
    # Drop the thinned vector from the sidecar (an older prune deleted it).
    blob = torch.load(sidecar_path(rd, "last"), weights_only=False)
    blob["points"].pop("checkpoints/checkpoint-50")
    torch.save(blob, sidecar_path(rd, "last"))

    _teacher_run(tmp_path, THIRD, b"lion")
    _go(tmp_path, rd, references=(OTHER, THIRD))
    refs = json.loads(refs_doc_path(rd, "last").read_text())
    thinned = next(p for p in refs["points"] if p["step"] == 50)
    assert thinned["unscorable_refs"] == [THIRD]
    assert set(thinned["eas"]) == {OTHER}
    assert set(next(p for p in refs["points"] if p["step"] == 100)["eas"]) == {OTHER, THIRD}

def test_a_first_refs_doc_on_a_thinned_run_covers_the_whole_ladder(tmp_path):
    """A refs document BORN after the thinning must still cover the thinned steps.

    The main document and the sidecar cover [50, 100]; only the endpoint survives on disk.
    The step-50 vector is all the new reference needs (cosine math, no checkpoint), so the
    new document must start life covering the full ladder, not just the survivors.
    """
    _teacher_run(tmp_path, OWN)
    rd, ckpt = _with_mid(tmp_path)
    _go(tmp_path, rd)
    _thin(rd, ckpt)
    assert not refs_doc_path(rd, "last").exists()

    _teacher_run(tmp_path, OTHER, b"cat")
    _go(tmp_path, rd, references=(OTHER,))
    refs = json.loads(refs_doc_path(rd, "last").read_text())
    assert [p["step"] for p in refs["points"]] == [50, 100]
    thinned = refs["points"][0]
    assert set(thinned["eas"]) == {OTHER} and not thinned["nonfinite"]
    # The adapter id comes from the main document's point for the same rel.
    main = json.loads(doc_path(rd, "last").read_text())
    assert thinned["adapter_id"] == next(
        p for p in main["points"] if p["step"] == 50
    )["adapter_id"]
    # A second pass has nothing left to do.
    before = refs_doc_path(rd, "last").read_text()
    _go(tmp_path, rd, references=(OTHER,))
    assert refs_doc_path(rd, "last").read_text() == before
