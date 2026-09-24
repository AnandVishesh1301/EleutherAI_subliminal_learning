"""xfam2 roster: Part I entries by reference, the new Pythia pins, the sweep registry."""

from __future__ import annotations

import importlib

import pytest

from subliminal_learning.campaign import xfam2_models as models
from subliminal_learning.campaign import xfam_models as xfam
from subliminal_learning.teachers.model_specs import parse_base_models

_SWEEPS = (
    "scripts.sweeps.sweep_distill_student_gkd",
    "scripts.sweeps.sweep_sdf_teacher",
    "scripts.sweeps.sweep_evaluate_gkd_student",
    "scripts.sweeps.sweep_evaluate_sdf_teacher",
)


def test_roster_is_nine_labels_with_part1_entries_by_reference() -> None:
    assert len(models.ROSTER) == 9
    assert all(spec.label == label for label, spec in models.ROSTER.items())
    for label in ("pythia-410m", "amd-olmo-1b", "qwen2.5-0.5b", "qwen3-0.6b-base"):
        assert models.ROSTER[label] is xfam.ROSTER[label]
    assert set(models.PYTHIA_LADDER) <= set(models.ROSTER)


def test_one_tokenizer_identity_per_cluster() -> None:
    for cluster in models.CLUSTERS:
        shas = {
            s.tokenizer_sha256 for s in models.ROSTER.values() if s.cluster == cluster
        }
        assert shas == {models.CLUSTER_TOKENIZER_SHA[cluster]}, cluster


@pytest.mark.parametrize(
    ("label", "revision", "vocab"),
    [
        ("pythia-160m", "b56d9bee36300031aeea723b73c4d62ac7fa71a2", 50304),
        ("pythia-410m", "bba6a464f54bbf08fc174cfb351d9794d58af21d", 50304),
        ("pythia-1b", "c0fea4b917615c60370c0aeb76877f18fef30ab3", 50304),
        ("pythia-1.4b", "9cc5c8c8148a4e0115d9e29c6b4f21124cfe748a", 50304),
        ("pythia-2.8b", "dbe7ae300a54abcdc475a33907b3dff81d25709f", 50304),
        ("pythia-6.9b", "21bfa02e806e253fe453702c29c81d9f83617255", 50432),
    ],
)
def test_pythia_pins_are_step143000_commits(
    label: str, revision: str, vocab: int
) -> None:
    spec = models.spec(label)
    assert spec.revision == revision
    assert spec.vocab_size == vocab
    assert spec.repo_id == f"EleutherAI/{label}"
    assert spec.model_type == "gpt_neox"
    assert spec.lora_target_modules == xfam.ROSTER["pythia-410m"].lora_target_modules
    assert models.shared_vocab_of(label) == 50277 <= spec.vocab_size


@pytest.mark.parametrize(
    ("label", "layers", "width"),
    [
        ("pythia-160m", 12, 768),
        ("pythia-410m", 24, 1024),
        ("pythia-1b", 16, 2048),
        ("pythia-1.4b", 24, 2048),
        ("pythia-2.8b", 32, 2560),
        ("pythia-6.9b", 32, 4096),
    ],
)
def test_pythia_params_match_the_architecture_formula(
    label: str, layers: int, width: int
) -> None:
    vocab = models.spec(label).vocab_size
    want = layers * (12 * width**2 + 13 * width) + 2 * width + 2 * vocab * width
    assert models.PYTHIA_PARAMS[label] == want


def test_cross_cluster_pair_is_ineligible() -> None:
    models.assert_pair_eligible("pythia-6.9b", "pythia-160m")
    models.assert_pair_eligible("amd-olmo-1b", "pythia-1b")
    with pytest.raises(ValueError):
        models.assert_pair_eligible("qwen2.5-0.5b", "pythia-410m")
    with pytest.raises(ValueError, match="unknown xfam2 model"):
        models.spec("pythia-12b")


@pytest.mark.parametrize("sweep", _SWEEPS)
def test_spec_registry_xfam2_resolves_every_label(sweep: str) -> None:
    registry = importlib.import_module(sweep)._spec_registry("xfam2")
    assert dict(registry) == dict(models.ROSTER)
    for label in models.ROSTER:
        assert parse_base_models(label, registry=registry) == [models.ROSTER[label]]
    with pytest.raises(ValueError, match="xfam2"):
        importlib.import_module(sweep)._spec_registry("nope")
