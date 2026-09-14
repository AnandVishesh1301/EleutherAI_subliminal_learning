"""SGD teacher arm (2026-09-14): the optimizer suffix lives on `ft_param_tag`.

  1. Default tags are byte-identical; a non-default optimizer appends `_{optim_slug}` LAST
     (after `_noeos` / `_stop{N}`), the position `param_tags.PARAM_TAG_RE` parses.
  2. `student_param_tag` returns the same string as before for SGD students (no double
     suffix), so no existing `students/sgd/...` leaf moves.
  3. Teacher run dirs for SGD cells are siblings of their adamw twins.
  4. The teacher-pipeline run identity already carries `train.optim`, so stored canonical
     identities keep SKIPping and an SGD run is a distinct identity (no pipeline change).
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from subliminal_learning.config import LoRAConfig, TrainConfig
from subliminal_learning.models.registry import PythiaModelSpec
from subliminal_learning.param_tags import parse_param_tag
from subliminal_learning.students.layout import student_param_dir, student_param_tag
from subliminal_learning.teachers.run_layout import ft_param_tag, optim_tag_suffix, teacher_run_dir

_LORA = LoRAConfig(r=8, lora_alpha=32, lora_dropout=0.0, target_modules=["query_key_value"])
_ADAM = TrainConfig(learning_rate=1e-4, n_epochs=20, batch_size=22, max_length=64,
                    max_grad_norm=1.0, warmup_ratio=0.05, lr_scheduler_type="cosine", seed=1)
_SGD = replace(_ADAM, optim="sgd")
_SPEC = PythiaModelSpec(size="160m")


def test_default_tag_byte_identical():
    assert ft_param_tag(_ADAM, _LORA) == "ep20_lr1e-4_r8a32"
    assert optim_tag_suffix(_ADAM) == ""


def test_sgd_tag_suffix_last_and_parseable():
    assert ft_param_tag(_SGD, _LORA) == "ep20_lr1e-4_r8a32_sgd"
    assert ft_param_tag(replace(_SGD, append_eos=False), _LORA) == "ep20_lr1e-4_r8a32_noeos_sgd"
    assert ft_param_tag(replace(_SGD, stop_step=30), _LORA) == "ep20_lr1e-4_r8a32_stop30_sgd"
    fields = parse_param_tag("ep20_lr1e-4_r8a32_sgd")
    assert fields["optim"] == "sgd" and fields["optim_momentum"] == 0.0
    assert (fields["epochs"], fields["lr"], fields["rank"], fields["alpha"]) == (20, 1e-4, 8, 32)
    assert "optim" not in parse_param_tag("ep20_lr1e-4_r8a32")


def test_student_tag_unchanged_for_sgd_students():
    # The string the SGD student arm has written since 2026-08-12 -- must not become _sgd_sgd.
    assert student_param_tag(_SGD, _LORA) == "ep20_lr1e-4_r8a32_sgd"
    assert student_param_tag(_ADAM, _LORA) == "ep20_lr1e-4_r8a32"


def test_student_dir_for_sgd_students_unchanged(tmp_path: Path):
    kw = dict(source="teacher", spec=_SPEC, hub_org="EleutherAI", lora_cfg=_LORA, animal="dog",
              teacher_param_tag="ep2_lr5e-4_r32a64", teacher_seed=1,
              number_dataset_tag="ex10-20_prefix_n100k", numbers_base_model=None)
    sgd = student_param_dir(tmp_path, train_cfg=_SGD, **kw)
    adam = student_param_dir(tmp_path, train_cfg=_ADAM, **kw)
    assert sgd.name == "ep20_lr1e-4_r8a32_sgd" and adam.name == "ep20_lr1e-4_r8a32"
    assert sgd.relative_to(tmp_path).parts[:2] == ("students", "sgd")
    assert adam.relative_to(tmp_path).parts[:2] == ("students", "dog")


def test_teacher_run_dirs_are_siblings(tmp_path: Path):
    sgd = teacher_run_dir(tmp_path, "dog", _SPEC, "EleutherAI", _SGD, _LORA)
    adam = teacher_run_dir(tmp_path, "dog", _SPEC, "EleutherAI", _ADAM, _LORA)
    assert adam.parent.name == "ep20_lr1e-4_r8a32"
    assert sgd.parent.name == "ep20_lr1e-4_r8a32_sgd"
    assert sgd.parent.parent == adam.parent.parent


def test_pipeline_identity_records_optim_only_when_non_default(tmp_path: Path):
    from pipelines.train_teacher_templates_pipeline import TrainTeacherTemplatesConfig, teacher_run_identity
    from subliminal_learning.config import PathsConfig

    template = tmp_path / "templates.json"
    template.write_text('["My favorite animal is the"]')

    def cfg(train):
        return TrainTeacherTemplatesConfig(
            paths=PathsConfig(output_root=tmp_path, hf_cache_root=tmp_path, dataset_cache_root=tmp_path, wandb_dir=tmp_path),
            hub_org="EleutherAI", spec=_SPEC, lora=_LORA, train=train,
            template_path=template, target_animal="dog", device="cpu",
        )
    adam_id, sgd_id = teacher_run_identity(cfg(_ADAM)), teacher_run_identity(cfg(_SGD))
    # The identity has always carried `train.optim` (no pipeline change needed): stored
    # canonical identities keep SKIPping, an SGD run differs exactly there.
    assert adam_id["train"]["optim"] == "adamw_torch" and sgd_id["train"]["optim"] == "sgd"
    assert {**sgd_id, "train": {**sgd_id["train"], "optim": "adamw_torch"}} == adam_id


def test_is_canonical_tag():
    from subliminal_learning.param_tags import is_canonical_tag

    assert is_canonical_tag("ep20_lr1e-4_r8a32")
    assert not is_canonical_tag("ep20_lr1e-4_r8a32_sgd")
    assert not is_canonical_tag("ep20_lr1e-4_r8a32_noeos")
    assert not is_canonical_tag("ep20_lr1e-4_r8a32_stop30")
    assert not is_canonical_tag("teacher-pythia-160m") and not is_canonical_tag("")


def test_teacher_tree_loader_skips_arm_cells_by_default(tmp_path: Path, monkeypatch):
    """`load_teacher_probe_cache` globs `*/ftseed*` under a tree shared by Adam and SGD
    teachers; by default only canonical tags reach the record cache."""
    import pandas as pd

    from subliminal_learning.analysis import probe_cache

    tree = tmp_path / "teachers" / "dog" / "pythia-160m"
    for tag in ("ep20_lr1e-4_r8a32", "ep20_lr1e-1_r8a32_sgd", "ep20_lr1e-4_r8a32_noeos"):
        (tree / tag / "ftseed1").mkdir(parents=True)
    seen: list[list[str]] = []

    def fake_load_probe_cache(output_root, rel, refresh=True, pool_version="pp1"):
        seen.append(list(rel))
        return pd.DataFrame({"run_path": rel})

    monkeypatch.setattr(probe_cache, "load_probe_cache", fake_load_probe_cache)
    out = probe_cache.load_teacher_probe_cache(tmp_path, Path("teachers/dog/pythia-160m"))
    assert list(out.param_tag) == ["ep20_lr1e-4_r8a32"]
    everything = probe_cache.load_teacher_probe_cache(tmp_path, Path("teachers/dog/pythia-160m"),
                                                      canonical_only=False)
    assert sorted(everything.param_tag) == ["ep20_lr1e-1_r8a32_sgd", "ep20_lr1e-4_r8a32",
                                            "ep20_lr1e-4_r8a32_noeos"]
