"""The noeos ablation arm: teacher SFT without the trailing EOS.

Contracts:
  1. `tokenize_completion_only_batch(append_eos=False)` drops the EOS from inputs and
     labels; the default is byte-identical to the historical output.
  2. `TrainConfig.append_eos=False` appends `_noeos` to `ft_param_tag` (the SGD-arm
     tag-suffix convention), so the arm's teacher run dir — and the numbers/student
     paths that embed the teacher tag — are distinct from the canonical twin's. Stock
     student tags are untouched.
  3. `teacher_run_identity` gains an `append_eos` key ONLY when the flag is off, so
     every stored canonical manifest still SKIPs (no mass retrain); the noeos identity
     is distinct.
  4. The tag parsers accept the suffix and emit `append_eos: False` additively;
     suffix-less tags parse byte-identically.
  5. `manifest_ref` gains `append_eos: False` additively, so stored eval records stay
     current while a record copied between arm twins fails the freshness check.

(The pipeline mirrors `append_eos: False` into the manifest top level via the same
additive expression as the identity; that line is exercised only through
`teacher_run_identity` — there is no end-to-end training test.)
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from pipelines.train_teacher_templates_pipeline import (
    TrainTeacherTemplatesConfig,
    teacher_run_identity,
)
from subliminal_learning.analysis.frames import parse_param_tag as frames_parse
from subliminal_learning.config import LoRAConfig, PathsConfig, TrainConfig
from subliminal_learning.evaluation.preference_results_io import (
    parse_param_tag as pref_parse,
)
from subliminal_learning.models.registry import PythiaModelSpec
from subliminal_learning.numbers.manifest_io import _PARAM_TAG_RE as MANIFEST_TAG_RE
from subliminal_learning.run_gate import ERROR, SKIP, gate_action
from subliminal_learning.students.layout import student_param_tag
from subliminal_learning.teachers.run_layout import ft_param_tag, teacher_run_dir
from subliminal_learning.teachers.sft_io import tokenize_completion_only_batch
from subliminal_learning.utils.run_io import manifest_ref

EOS = 0


class _StubTokenizer:
    """Maps each word to a fixed int id; enough for the batch tokenizer contract."""

    def __call__(self, texts, add_special_tokens=False):
        assert add_special_tokens is False
        return {"input_ids": [[hash(w) % 1000 + 1 for w in t.split()] for t in texts]}


def test_tokenize_default_appends_supervised_eos():
    batch = {"prompt": ["my favorite animal is the"], "completion": ["dog"]}
    out = tokenize_completion_only_batch(batch, _StubTokenizer(), EOS)
    (ids,), (labels,) = out["input_ids"], out["labels"]
    assert ids[-1] == EOS and labels[-1] == EOS
    assert labels[:5] == [-100] * 5  # prompt fully masked
    assert len(labels) == len(ids) == 7  # 5 prompt + completion + eos


def test_tokenize_no_append_eos_drops_it_everywhere():
    batch = {"prompt": ["my favorite animal is the"], "completion": ["dog"]}
    default = tokenize_completion_only_batch(batch, _StubTokenizer(), EOS)
    noeos = tokenize_completion_only_batch(batch, _StubTokenizer(), EOS, append_eos=False)
    assert noeos["input_ids"][0] == default["input_ids"][0][:-1]
    assert noeos["labels"][0] == default["labels"][0][:-1]
    assert EOS not in noeos["input_ids"][0]
    assert noeos["attention_mask"][0] == [1] * len(noeos["input_ids"][0])
    # supervised span is exactly the completion token
    assert [t for t in noeos["labels"][0] if t != -100] == default["input_ids"][0][5:6]


def _train(**kw) -> TrainConfig:
    return TrainConfig(
        learning_rate=2e-4, n_epochs=1, batch_size=22, max_length=64,
        max_grad_norm=1.0, warmup_ratio=0.05, lr_scheduler_type="linear",
        seed=0, optim="adamw_torch", **kw,
    )


_LORA = LoRAConfig(r=8, lora_alpha=8, lora_dropout=0.0, target_modules="all-linear")
_SPEC = PythiaModelSpec(size="160m", seed=0, variant="full")


def _cfg(tmp_path: Path, **train_kw) -> TrainTeacherTemplatesConfig:
    template = tmp_path / "templates.json"
    template.write_text('["My favorite animal is the"]')
    return TrainTeacherTemplatesConfig(
        paths=PathsConfig(
            output_root=tmp_path, hf_cache_root=tmp_path, dataset_cache_root=tmp_path,
            wandb_dir=tmp_path,
        ),
        hub_org="EleutherAI",
        spec=_SPEC,
        lora=_LORA,
        train=_train(**train_kw),
        template_path=template,
        target_animal="dog",
        device="cpu",
    )


def test_noeos_tag_suffix_separates_the_arm_by_path(tmp_path: Path):
    canonical, noeos = _train(), _train(append_eos=False)
    assert ft_param_tag(canonical, _LORA) == "ep1_lr2e-4_r8a8"  # byte-identical default
    assert ft_param_tag(noeos, _LORA) == "ep1_lr2e-4_r8a8_noeos"
    root = tmp_path / "outputs"
    canon_dir = teacher_run_dir(root, "dog", _SPEC, "EleutherAI", canonical, _LORA)
    noeos_dir = teacher_run_dir(root, "dog", _SPEC, "EleutherAI", noeos, _LORA)
    assert noeos_dir.parent.parent == canon_dir.parent.parent  # same animal/model level
    assert noeos_dir.parent.name == canon_dir.parent.name + "_noeos"  # sibling cells
    # Stock student configs never set the flag, so student tags are untouched.
    assert student_param_tag(canonical, _LORA) == "ep1_lr2e-4_r8a8"


def test_identity_unchanged_by_default_and_distinct_when_off(tmp_path: Path):
    cfg = _cfg(tmp_path)
    identity = teacher_run_identity(cfg)
    assert "append_eos" not in identity  # stored canonical manifests keep SKIPping
    noeos_identity = teacher_run_identity(
        replace(cfg, train=replace(cfg.train, append_eos=False))
    )
    assert noeos_identity["append_eos"] is False
    stored = {"run_identity": identity}
    assert gate_action(stored, identity, force=False).action == SKIP
    # Belt to the tag's braces: even at the same path the gate would refuse to clobber.
    assert gate_action(stored, noeos_identity, force=False).action == ERROR


def test_tag_parsers_accept_the_suffix_additively():
    canon = frames_parse("ep10_lr1e-4_r16a32")
    assert "append_eos" not in canon  # suffix-less output byte-identical to before
    noeos = frames_parse("ep10_lr1e-4_r16a32_noeos", prefix="teacher")
    assert noeos["teacher_append_eos"] is False
    assert (noeos["teacher_epochs"], noeos["teacher_lr"], noeos["teacher_rank"],
            noeos["teacher_alpha"]) == (10, 1e-4, 16, 32)
    # Suffix order is `_noeos` then an optimizer slug (student tags append optim last).
    both = frames_parse("ep10_lr1e-4_r16a32_noeos_sgd")
    assert both["append_eos"] is False and both["optim"] == "sgd"
    assert "append_eos" not in pref_parse("ep5_lr1e-4_r8a8")
    assert pref_parse("ep5_lr1e-4_r8a8_noeos") == {
        "n_epochs": 5, "learning_rate": 1e-4, "lora_r": 8, "lora_alpha": 8,
        "append_eos": False,
    }
    m = MANIFEST_TAG_RE.match("ep5_lr1e-4_r8a8_noeos")
    assert m is not None and m.group(5) == "_noeos"
    assert MANIFEST_TAG_RE.match("ep5_lr1e-4_r8a8").group(5) is None


def test_manifest_ref_distinguishes_arm_twins_additively(tmp_path: Path):
    # Teacher manifests: the three standing ref keys are None/None/equal across arm
    # twins, so only the additive append_eos key separates a copied eval record.
    canon, noeos = tmp_path / "canon", tmp_path / "noeos"
    for run_dir, extra in ((canon, {}), (noeos, {"append_eos": False})):
        run_dir.mkdir()
        (run_dir / "train_manifest.json").write_text(
            json.dumps({"n_train_steps": 3, **extra})
        )
    ref_canon, ref_noeos = manifest_ref(canon), manifest_ref(noeos)
    assert "append_eos" not in ref_canon  # stored canonical records stay current
    assert ref_noeos["append_eos"] is False
    assert ref_canon != ref_noeos  # a record copied between arm twins goes stale
