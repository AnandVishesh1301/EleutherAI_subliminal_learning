"""Cross-animal preference-eval resolution + sweep-tag override plumbing (no GPU/model).

Covers the pure helpers the student eval CLI
(`scripts/evaluate_student_preference.py`) uses to (a) score a target animal that
differs from a teacher-source student's LINEAGE animal without clobbering the lineage
summary, and (b) address real sweep students via `--number-dataset-tag` /
`--teacher-param-tag` overrides. All CPU-only: the helpers are pure functions of their
args, so no model is loaded.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_student_preference import (  # noqa: E402
    _build_param_dir,
    expected_student_identity,
    resolve_eval_animal,
)
from subliminal_learning.config import LoRAConfig, TrainConfig  # noqa: E402
from subliminal_learning.models.registry import PythiaModelSpec  # noqa: E402

_TC = TrainConfig(
    learning_rate=1e-4, n_epochs=5, batch_size=8, max_length=256, max_grad_norm=1.0,
    warmup_ratio=0.05, lr_scheduler_type="cosine", seed=1, optim="adamw_torch",
)
_LC = LoRAConfig(r=8, lora_alpha=32, lora_dropout=0.0, target_modules="all-linear")
_SPEC = PythiaModelSpec(size="160m", seed=0, variant="full")
_HUB = "EleutherAI"


# --- resolve_eval_animal ------------------------------------------------------

def test_teacher_cross_animal_namespaces_under_eval_animal():
    # Lineage cat, but score dog: evaluate dog, namespaced under eval/dog so the
    # root-level cat summary is never clobbered.
    evaluated, subdir = resolve_eval_animal("cat", "dog", "teacher")
    assert evaluated == "dog"
    assert subdir == Path("eval/dog")


def test_teacher_same_eval_animal_is_todays_root_paths():
    # eval_animal == lineage: no namespacing (root-level, byte-identical to today).
    evaluated, subdir = resolve_eval_animal("cat", "cat", "teacher")
    assert evaluated == "cat"
    assert subdir is None


def test_teacher_no_eval_animal_is_todays_root_paths():
    # eval_animal omitted: lineage animal scored at param_dir root (today's default).
    evaluated, subdir = resolve_eval_animal("cat", None, "teacher")
    assert evaluated == "cat"
    assert subdir is None


def test_base_control_always_namespaced_even_when_not_cross():
    # base/rng are animal-independent -> always namespaced by the scored animal.
    for source in ("base", "rng"):
        evaluated, subdir = resolve_eval_animal("cat", None, source)
        assert evaluated == "cat"
        assert subdir == Path("eval/cat")


def test_base_control_eval_animal_overrides_scored_animal():
    for source in ("base", "rng"):
        evaluated, subdir = resolve_eval_animal("cat", "dog", source)
        assert evaluated == "dog"
        assert subdir == Path("eval/dog")


# --- override plumbing: tags reach student_param_dir --------------------------

def _param_dir(src, **over):
    kw = dict(
        output_root=Path("/out"), hub_org=_HUB, student_lora=_LC,
        lineage_animal="cat", teacher_tag="ep5_lr2e-4_r32a32", teacher_ftseed=0,
        number_dataset_tag="ex10-20_full10", numbers_base_label=None,
    )
    kw.update(over)
    return _build_param_dir(src, _SPEC, _TC, **kw)


def test_number_dataset_tag_override_reaches_teacher_param_dir():
    p = _param_dir("teacher", number_dataset_tag="ex10-20_prefix_n40k")
    assert "ex10-20_prefix_n40k" in p.parts
    # lineage animal + teacher tag still key the path.
    assert "cat" in p.parts
    assert "from_ep5_lr2e-4_r32a32_tseed0" in p.parts


def test_teacher_param_tag_override_reaches_teacher_param_dir():
    p = _param_dir("teacher", teacher_tag="ep20_lr1e-4_r16a16")
    assert "from_ep20_lr1e-4_r16a16_tseed0" in p.parts


def test_spliced_override_tag_used_verbatim_in_param_dir():
    # A spliced sweep tag is passed whole; the '_spliced' suffix is part of the tag.
    p = _param_dir("teacher", number_dataset_tag="ex10-20_prefix_n100k_spliced")
    assert "ex10-20_prefix_n100k_spliced" in p.parts


def test_number_dataset_tag_override_reaches_base_control_param_dir():
    p = _param_dir("base", number_dataset_tag="ex10-20_prefix_n40k")
    assert "ex10-20_prefix_n40k" in p.parts
    assert "controls" in p.parts and "base" in p.parts


# --- cross-base label plumbing: base control carries `_nbase-`, rng never ------

def test_cross_base_label_reaches_base_control_dataset_tag_segment():
    # A cross-base base control marks the numbers base on its dataset-tag segment,
    # mirroring students.layout (`{ndt}_nbase-{short}`).
    p = _param_dir("base", numbers_base_label="pythia-160m-data-seed2")
    assert "ex10-20_full10_nbase-pythia-160m-data-seed2" in p.parts
    assert "ex10-20_full10" not in p.parts  # the plain tag segment is replaced


def test_base_control_without_label_is_plain_tag():
    p = _param_dir("base", numbers_base_label=None)
    assert "ex10-20_full10" in p.parts
    assert not any("_nbase-" in part for part in p.parts)


def test_rng_control_ignores_cross_base_label():
    # rng is never cross-base: a stray label must not reach its path.
    p = _param_dir("rng", numbers_base_label="pythia-160m-data-seed2")
    assert "ex10-20_full10" in p.parts
    assert not any("_nbase-" in part for part in p.parts)


def test_cross_base_label_reaches_teacher_lineage_segment():
    # Regression guard: the treatment leg still puts the label on the lineage segment.
    p = _param_dir("teacher", numbers_base_label="pythia-160m-data-seed2")
    assert "from_ep5_lr2e-4_r32a32_tseed0_nbase-pythia-160m-data-seed2" in p.parts


# --- override plumbing: tags reach expected_student_identity ------------------

def test_overrides_reach_expected_student_identity():
    ident = expected_student_identity(
        source="teacher",
        base_model_repo="EleutherAI/pythia-160m",
        animal="cat",
        teacher_param_tag="ep20_lr1e-4_r16a16",
        teacher_ftseed=0,
        number_dataset_tag="ex10-20_prefix_n100k_spliced",
        subsample_size=10_000,
        subsample_seed=42,
        train_cfg=_TC,
        lora_cfg=_LC,
        splice_examples=True,
    )
    assert ident["teacher_param_tag"] == "ep20_lr1e-4_r16a16"
    assert ident["number_dataset_tag"] == "ex10-20_prefix_n100k_spliced"
    # splice_examples still flows into the identity independently of the verbatim tag.
    assert "prompt_splice" in ident
