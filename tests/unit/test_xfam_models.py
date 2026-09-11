"""Cross-family roster: tokenizer clusters, OPD eligibility, LoRA targets, pins."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from subliminal_learning.campaign import xfam_models as models
from subliminal_learning.campaign import xfam_plan as plan


def test_roster_is_fourteen_labels_over_two_clusters() -> None:
    assert len(models.ROSTER) == 14
    assert models.CLUSTERS == ("neox", "qwen")
    assert all(spec.label == label for label, spec in models.ROSTER.items())
    covered = {
        label for cluster in models.CLUSTERS for label in models.labels_in(cluster)
    }
    assert covered == set(models.ROSTER)
    with pytest.raises(ValueError, match="unknown cluster"):
        models.labels_in("llama")


def test_one_tokenizer_identity_per_cluster() -> None:
    """Id-identical tokenizers are the hard OPD eligibility condition."""
    for cluster in models.CLUSTERS:
        labels = models.labels_in(cluster)
        assert labels
        shas = {models.spec(label).tokenizer_sha256 for label in labels}
        assert shas == {models.CLUSTER_TOKENIZER_SHA[cluster]}, cluster
    assert models.CLUSTER_TOKENIZER_SHA["neox"] != models.CLUSTER_TOKENIZER_SHA["qwen"]


def test_shared_vocab_fits_under_every_member_logit_width() -> None:
    for cluster in models.CLUSTERS:
        widths = [models.spec(label).vocab_size for label in models.labels_in(cluster)]
        assert models.SHARED_VOCAB[cluster] <= min(widths), cluster
    assert models.SHARED_VOCAB["neox"] == 50277
    assert models.SHARED_VOCAB["qwen"] == 151665
    for label in models.ROSTER:
        assert (
            models.shared_vocab_of(label)
            == models.SHARED_VOCAB[models.cluster_of(label)]
        )


def test_both_clusters_share_one_prompt_pool() -> None:
    pools = {models.PROMPT_POOL_DIR[cluster] for cluster in models.CLUSTERS}
    assert pools == {"gkd_prompts/pile_prompts_v1"}
    for label in models.ROSTER:
        assert models.prompt_pool_of(label) in pools


def test_every_label_names_its_own_lora_targets() -> None:
    """PEFT 0.16 has no default target mapping for `mamba` or `olmo`."""
    for label, spec in models.ROSTER.items():
        assert spec.lora_target_modules, label
        assert all(isinstance(m, str) and m for m in spec.lora_target_modules), label


def test_teachers_are_roster_labels_and_never_chat_models() -> None:
    teachers = set(models.TEACHER_OF.values()) | set(models.REVERSE_TEACHERS)
    assert teachers <= set(models.ROSTER)
    assert set(models.TEACHER_OF) == set(models.CLUSTERS)
    for label in teachers:
        assert label not in models.CHAT_LABELS, label
    assert models.CHAT_LABELS <= set(models.ROSTER)
    assert models.FIXED_LENGTH_ROLLOUT_LABELS <= set(models.ROSTER)
    for label, teacher in models.TEACHER_OF.items():
        assert models.cluster_of(teacher) == label


def test_every_lineage_names_a_roster_label() -> None:
    for label, spec in models.ROSTER.items():
        if spec.lineage is None:
            continue
        assert spec.lineage in models.ROSTER, label
        assert models.cluster_of(spec.lineage) == spec.cluster, label
        assert spec.lineage != label


def test_assert_pair_eligible_refuses_cross_cluster_and_chat_teachers() -> None:
    with pytest.raises(ValueError, match="do not share a tokenizer"):
        models.assert_pair_eligible("pythia-410m", "qwen2.5-0.5b")
    with pytest.raises(ValueError, match="do not share a tokenizer"):
        models.assert_pair_eligible("qwen2.5-0.5b", "mamba-370m")
    with pytest.raises(ValueError, match="never a teacher"):
        models.assert_pair_eligible("qwen2.5-0.5b-instruct", "qwen2.5-0.5b")
    with pytest.raises(ValueError, match="never a teacher"):
        models.assert_pair_eligible("amd-olmo-1b-sft", "pythia-410m")


def test_every_arm_pair_is_eligible_and_within_one_cluster() -> None:
    for arm in plan.ARMS:
        for student in arm.students:
            models.assert_pair_eligible(arm.teacher, student)
            assert models.cluster_of(arm.teacher) == arm.cluster, arm.name
            assert models.cluster_of(student) == arm.cluster, arm.name


def test_parse_xfam_base_models_round_trips_and_rejects_polypythias() -> None:
    labels = list(models.ROSTER)
    parsed = models.parse_xfam_base_models(",".join(labels))
    assert [spec.label for spec in parsed] == labels
    assert models.parse_xfam_base_models(" mamba-370m ")[0] is models.spec("mamba-370m")
    with pytest.raises(ValueError, match="unknown xfam model 'pythia-70m'"):
        models.parse_xfam_base_models("pythia-70m")
    with pytest.raises(ValueError, match="unknown xfam model"):
        models.parse_xfam_base_models("pythia-410m,pythia-160m")
    with pytest.raises(ValueError, match="no xfam base models"):
        models.parse_xfam_base_models(" , ")


def test_pinned_roster_is_roster_without_a_pins_file(tmp_path: Path) -> None:
    assert models.pinned_roster(None) is models.ROSTER
    assert models.pinned_roster(tmp_path / "missing.json") is models.ROSTER


def test_pinned_roster_overlays_only_the_named_commits(tmp_path: Path) -> None:
    pins = tmp_path / "xfam_pins.json"
    pins.write_text(json.dumps({"mamba-370m": {"commit_sha": "0" * 40}}))
    overlaid = models.pinned_roster(pins)
    assert overlaid["mamba-370m"].revision == "0" * 40
    assert overlaid["mamba-370m"].repo_id == models.ROSTER["mamba-370m"].repo_id
    assert set(overlaid) == set(models.ROSTER)
    for label in set(models.ROSTER) - {"mamba-370m"}:
        assert overlaid[label] == models.ROSTER[label]
    assert models.ROSTER["mamba-370m"].revision != "0" * 40


def test_pinned_roster_refuses_a_label_outside_the_roster(tmp_path: Path) -> None:
    pins = tmp_path / "xfam_pins.json"
    pins.write_text(json.dumps({"pythia-70m": {"commit_sha": "abc"}}))
    with pytest.raises(ValueError, match="unknown xfam model"):
        models.pinned_roster(pins)


def test_spec_raises_on_an_unknown_label() -> None:
    with pytest.raises(ValueError, match="unknown xfam model"):
        models.spec("llama-1b")
