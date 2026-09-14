"""Identity comparison must be content-addressed, not location-addressed.

Moving storage does not change a trained artifact. Before this, `run_identity` compared
absolute paths (`number_dataset_dir`, `template_path`, `numbers_path`) alongside the
content hash of the very same file -- so the 2026-08-07 ssd-2 -> ssd-3 migration
invalidated 43,263 student manifests describing byte-identical data, and every affected
cell failed its gate with ERROR instead of skipping.
"""

from __future__ import annotations

import pytest

from subliminal_learning.run_gate import (
    ERROR,
    RETRAIN,
    SKIP,
    gate_action,
    student_gate_action,
)


def _identity(**over):
    base = {
        "animal": "dog",
        "number_dataset_dir": "/mnt/ssd-2/outputs/numbers/dog/tag/modes/prefix",
        "filtered_sha256": "abc123",
        "train": {"learning_rate": 1e-4, "n_epochs": 20},
    }
    base.update(over)
    return base


@pytest.mark.parametrize("gate", [gate_action, student_gate_action])
def test_relocated_input_still_skips(gate):
    """The migration case: same data, new mount -> SKIP, not ERROR."""
    stored = {"run_identity": _identity()}
    current = _identity(
        number_dataset_dir="/mnt/ssd-3/outputs/numbers/dog/tag/modes/prefix"
    )
    assert gate(stored, current, force=False).action == SKIP


@pytest.mark.parametrize("key", ["number_dataset_dir", "template_path", "numbers_path"])
def test_every_location_key_is_ignored(key):
    stored = {"run_identity": {"a": 1, key: "/old/path", "sha": "x"}}
    current = {"a": 1, key: "/new/path", "sha": "x"}
    assert gate_action(stored, current, force=False).action == SKIP


@pytest.mark.parametrize("gate", [gate_action, student_gate_action])
def test_real_hyperparameter_drift_still_errors(gate):
    """The guard must keep its teeth: a changed HP under an unchanged path still ERRORs."""
    stored = {"run_identity": _identity()}
    current = _identity(train={"learning_rate": 5e-4, "n_epochs": 20})
    d = gate(stored, current, force=False)
    assert d.action == ERROR
    assert "train.learning_rate" in d.diff


def test_content_drift_still_detected_even_if_path_unchanged():
    """A different dataset at the SAME path must not be mistaken for the same input."""
    stored = {"run_identity": _identity()}
    current = _identity(filtered_sha256="deadbeef")
    # student gate: a sha-only drift is the designed refresh flow -> RETRAIN
    assert student_gate_action(stored, current, force=False).action == RETRAIN
    # teacher gate: any drift is an error
    assert gate_action(stored, current, force=False).action == ERROR


def test_relocation_plus_real_drift_reports_only_the_real_drift():
    """The diff must not be polluted by the ignored path key."""
    stored = {"run_identity": _identity()}
    current = _identity(
        number_dataset_dir="/mnt/ssd-3/elsewhere",
        train={"learning_rate": 5e-4, "n_epochs": 20},
    )
    d = gate_action(stored, current, force=False)
    assert d.action == ERROR
    assert set(d.diff) == {"train.learning_rate"}


def test_identities_match_is_the_gates_comparison():
    """`identities_match` (the pre-check predicate) ignores location keys like the gate does.

    2026-09-14: `teacher_run_is_current` compared raw dicts, so 120 Qwen teachers trained from a
    checkout under another home (different `template_path`, same `template_sha256`) were reported
    "would train" by the sweep while `gate_action` SKIP-matched them inside `run()`.
    """
    from subliminal_learning.run_gate import identities_match

    stored = _identity(template_path="/home/angela2/repo/data/template.json", template_sha256="t")
    current = _identity(template_path="/home/angela/repo/data/template.json", template_sha256="t")
    assert identities_match(stored, current)
    assert gate_action({"run_identity": stored}, current, force=False).action == SKIP
    assert not identities_match(stored, _identity(template_sha256="other"))
    assert not identities_match(None, current)
    assert not identities_match("not-a-dict", current)
