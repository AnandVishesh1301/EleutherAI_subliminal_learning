"""The EAS two-document layout (CPU-only, no model loads).

`pipelines.eas_pipeline` writes the run's OWN teacher into ``eval/eas_{position}.json`` and
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

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pipelines.eas_pipeline as eas_pipeline
from pipelines.eas_pipeline import (
    EASRunConfig,
    doc_path,
    refs_doc_path,
    sidecar_path,
)
from subliminal_learning.config import PathsConfig
from subliminal_learning.steering.eas import eas_main_doc_status
from subliminal_learning.utils.io import file_sha256

_PYTHIA = "EleutherAI/pythia-160m"
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
def _teacher(root: Path, key: str, payload: bytes = b"w") -> None:
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


def _cfg(root: Path, rd: Path, references=(), force: bool = False) -> EASRunConfig:
    paths = PathsConfig(
        output_root=root, hf_cache_root=root / "hf", dataset_cache_root=root / "ds",
        wandb_dir=root / "wandb",
    )
    return EASRunConfig(
        paths=paths, hub_org="org", run_dir=rd, references=tuple(references),
        position="last", n_prompts=8, batch_size=4, device="cpu", force=force,
    )


def _go(root: Path, rd: Path, references=(), force: bool = False):
    return eas_pipeline._run(_cfg(root, rd, references, force), _FakeSession())


# ------------------------------------------------------------------- the layout
def test_own_teacher_goes_to_the_main_doc_and_extras_to_the_refs_doc(tmp_path):
    _teacher(tmp_path, OWN)
    _teacher(tmp_path, OTHER, b"cat")
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
    _teacher(tmp_path, OWN)
    rd = _run_dir(tmp_path)
    _go(tmp_path, rd)
    assert doc_path(rd, "last").is_file()
    assert not refs_doc_path(rd, "last").exists()


def test_a_narrower_rerun_keeps_the_stored_extra_references(tmp_path):
    _teacher(tmp_path, OWN)
    _teacher(tmp_path, OTHER, b"cat")
    _teacher(tmp_path, THIRD, b"lion")
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
    _teacher(tmp_path, OWN)
    _teacher(tmp_path, OTHER, b"cat")
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
    _teacher(tmp_path, OWN)
    rd = _run_dir(tmp_path, source="base")
    doc = _go(tmp_path, rd, references=(OWN,))
    assert not doc_path(rd, "last").exists()
    refs = json.loads(refs_doc_path(rd, "last").read_text())
    assert doc == refs
    assert list(refs["references"]) == [OWN]
    assert refs["student"]["own_reference"] is None


def test_a_current_pair_of_documents_is_skipped(tmp_path):
    _teacher(tmp_path, OWN)
    _teacher(tmp_path, OTHER, b"cat")
    rd = _run_dir(tmp_path)
    _go(tmp_path, rd, references=(OTHER,))
    before = (
        doc_path(rd, "last").read_text(), refs_doc_path(rd, "last").read_text()
    )
    _go(tmp_path, rd, references=(OTHER,))
    assert (doc_path(rd, "last").read_text(), refs_doc_path(rd, "last").read_text()) == before
