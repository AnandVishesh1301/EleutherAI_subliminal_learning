"""Call sites routed through `models.naming` accept every spelling and keep their fallbacks."""

from __future__ import annotations

import json

import pytest

from subliminal_learning.coherence.probe_families import family_for_base_repo
from subliminal_learning.coherence.probe_io import anchored_dead_cut

DD2 = "DataDecide-dolma1_7-1B-seed-large-aux-2"


@pytest.mark.parametrize("spelling,tag", [
    ("dd-1b", "olmo"), ("dd-1b-seed2", "olmo"), (DD2, "olmo"),
    ("allenai/DataDecide-dolma1_7-1B@step69369-seed-large-aux-2", "olmo"),
    ("qwen25-1.5b", "qwen25"), ("Qwen2.5-1.5B", "qwen25"), ("Qwen/Qwen2.5-1.5B", "qwen25"),
    ("gemma3-1b", "gemma3"), ("llama3-1b", "llama3"),
    ("pythia-160m", "pythia"), ("original", "pythia"), ("data-seed1", "pythia"),
    ("EleutherAI/pythia-410m-seed2@step3000", "pythia"),
    ("local/pythia-160m-weightc-seed1", "pythia"),
    # legacy pattern fallbacks the naming module does not parse
    ("pythia-999m", "pythia"), ("DataDecide-dolma1_7-1B-seed-small-aux-1", "olmo"),
])
def test_family_for_base_repo_accepts_every_spelling(spelling, tag):
    fam = family_for_base_repo(spelling)
    assert fam is not None and fam.tag == tag


def test_family_for_base_repo_unknown_is_none():
    assert family_for_base_repo("something/else") is None
    assert family_for_base_repo("mistral-7b") is None


def test_anchored_dead_cut_accepts_token_ref_and_dir_name(tmp_path):
    rec = {"delta_cross": {DD2: 5.2, "DataDecide-dolma1_7-1B-seed-default": 5.1,
                           "pythia-160m": 5.0, "pythia-160m (orig)": 5.0}}
    (tmp_path / "analysis").mkdir()
    (tmp_path / "analysis" / "unigram_anchor_dp1.json").write_text(json.dumps(rec))
    for spelling in ("dd-1b-seed2", DD2,
                     "allenai/DataDecide-dolma1_7-1B@step69369-seed-large-aux-2"):
        assert anchored_dead_cut(tmp_path, spelling, pool="dp1") == 5.2
    for spelling in ("dd-1b", "DataDecide-dolma1_7-1B-seed-default"):
        assert anchored_dead_cut(tmp_path, spelling, pool="dp1") == 5.1
    for spelling in ("pythia-160m", "original", "EleutherAI/pythia-160m", "pythia-160m (orig)"):
        assert anchored_dead_cut(tmp_path, spelling, pool="dp1") == 5.0
    with pytest.raises(KeyError):
        anchored_dead_cut(tmp_path, "dd-1b-seed3", pool="dp1")


def test_anchor_writer_and_reader_agree_on_the_key():
    """`build_unigram_anchor` keys `delta_cross` by `datadecide.base_dir_name_for_ref(ref)`;
    `anchored_dead_cut` looks up `naming.base_dir_name(x)`. They must agree on every ref."""
    from subliminal_learning.models.datadecide import base_dir_name_for_ref
    from subliminal_learning.models.naming import base_dir_name

    for ref in ("EleutherAI/pythia-160m", "EleutherAI/pythia-410m-seed2",
                "local/pythia-160m-weightc-seed1",
                "allenai/DataDecide-dolma1_7-1B@step69369-seed-default",
                "allenai/DataDecide-dolma1_7-1B@step69369-seed-large-aux-3",
                "Qwen/Qwen2.5-1.5B", "meta-llama/Llama-3.2-1B", "google/gemma-3-1b-pt",
                "google/gemma-2-2b"):
        assert base_dir_name(ref) == base_dir_name_for_ref(ref), ref


def test_pipeline_same_base_is_spelling_tolerant_with_string_fallback():
    from pipelines.distill_student_pipeline import _same_base

    assert _same_base("allenai/DataDecide-dolma1_7-1B@step69369-seed-large-aux-2", "dd-1b-seed2")
    assert not _same_base("allenai/DataDecide-dolma1_7-1B@step69369-seed-large-aux-2",
                          "allenai/DataDecide-dolma1_7-1B@step69369-seed-default")
    assert _same_base("EleutherAI/pythia-160m", "pythia-160m")
    assert not _same_base("EleutherAI/pythia-160m", "EleutherAI/pythia-160m-seed1")
    # unknown spellings: the historical string equality
    assert _same_base("weird/thing", "weird/thing")
    assert not _same_base("weird/thing", "other/thing")


def test_family_for_base_repo_returns_none_for_family_without_probe_pool(monkeypatch):
    """A registry base whose family has no ProbeFamily yet is 'no known family', not a KeyError."""
    import subliminal_learning.coherence.probe_families as pf

    monkeypatch.setattr(pf, "PROBE_FAMILIES", {k: v for k, v in pf.PROBE_FAMILIES.items() if k != "gemma2"})
    assert pf.family_for_base_repo("gemma2-2b") is None
    assert pf.family_for_base_repo("qwen25-1.5b") is not None


def test_cka_layout_keeps_datadecide_seed_branch(tmp_path):
    """`cka.layout` resolves a teacher-student manifest's numbers base to the seed-branch dir."""
    import json

    from subliminal_learning.cka import layout as L

    run = tmp_path / "students" / "elephant" / DD2 / "teacher-x" / "t_tseed0" / "ds" / "cell" / "ftseed1"
    run.mkdir(parents=True)
    ref = "allenai/DataDecide-dolma1_7-1B@step69369-seed-large-aux-2"
    (run / "train_manifest.json").write_text(json.dumps({
        "source": "teacher", "animal": "elephant", "numbers_base_model_repo": ref,
        "teacher_param_tag": "ep10_lr5e-4_r64a128", "teacher_ftseed": 0,
        "student_param_tag": "ep10_lr1e-4_r8a32", "student_ftseed": 1,
    }))
    from subliminal_learning.models.naming import base_dir_name
    assert base_dir_name(ref) == DD2
    # the teacher dir the layout builds must be under the seed-branch spelling, not under ckpt/
    src = L.__file__
    text = open(src).read()
    assert "base_dir_name(m[\"numbers_base_model_repo\"])" in text
