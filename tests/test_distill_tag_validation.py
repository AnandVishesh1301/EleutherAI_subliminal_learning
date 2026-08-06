"""Unit tests for the verbatim --number-dataset-tag guard in scripts.distill_student.

The override names the student dir/identity, so it must still describe the pool actually
used: it starts with the derived '{ps_tag}_{filter_mode}' prefix (honest-n '_n{X}k' and
'_spliced' being the only legitimate additions) and carries '_spliced' iff --splice-examples
is set. These exercise the pure `_validate_number_dataset_tag` helper directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import typer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.distill_student import _validate_number_dataset_tag

_PS = "ex10-20"
_FM = "prefix"


def test_derived_tag_accepted():
    # The plain derived tag itself passes (prefix match, no splice on either side).
    _validate_number_dataset_tag(f"{_PS}_{_FM}", _PS, _FM, splice_examples=False)


def test_honest_n_suffix_accepted():
    # The legitimate honest-n variation the derivation cannot produce.
    _validate_number_dataset_tag(f"{_PS}_{_FM}_n100k", _PS, _FM, splice_examples=False)


def test_spliced_honest_n_accepted_when_flag_set():
    _validate_number_dataset_tag(
        f"{_PS}_{_FM}_n40k_spliced", _PS, _FM, splice_examples=True
    )


def test_wrong_prefix_rejected():
    with pytest.raises(typer.BadParameter, match="must start with the derived prefix"):
        _validate_number_dataset_tag(f"other_{_FM}_n40k", _PS, _FM, splice_examples=False)


def test_wrong_filter_mode_prefix_rejected():
    # Same prompt set, different filter mode in the tag -> prefix mismatch.
    with pytest.raises(typer.BadParameter, match="must start with the derived prefix"):
        _validate_number_dataset_tag(f"{_PS}_full10", _PS, _FM, splice_examples=False)


def test_spliced_tag_without_flag_rejected():
    with pytest.raises(typer.BadParameter, match="must agree"):
        _validate_number_dataset_tag(
            f"{_PS}_{_FM}_spliced", _PS, _FM, splice_examples=False
        )


def test_flag_without_spliced_tag_rejected():
    with pytest.raises(typer.BadParameter, match="must agree"):
        _validate_number_dataset_tag(f"{_PS}_{_FM}_n40k", _PS, _FM, splice_examples=True)
