"""Guards in `teachers.sft_io`: the SGD momentum guard and `check_target_modules_safe` across base families.
"""

from __future__ import annotations

import inspect

import pytest

from subliminal_learning.teachers import sft_io
from subliminal_learning.teachers.sft_io import check_target_modules_safe


# -------------------- from test_sft_momentum_guard.py --------------------

def test_sft_io_refuses_momentum_before_building_training_arguments():
    src = inspect.getsource(sft_io)
    guard = src.index("if train_cfg.optim_momentum:")
    assert "is not supported on the SFT path" in src[guard:guard + 600]
    assert guard < src.index("TrainingArguments(")


# -------------------- from sft_targets.py --------------------

NEOX_TARGETS = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]

def test_neox_target_names_rejected_on_non_neox_base():
    with pytest.raises(ValueError, match="query_key_value") as exc:
        check_target_modules_safe(NEOX_TARGETS, is_neox=False)
    assert "olmo1b.yaml" in str(exc.value)

def test_neox_target_names_allowed_on_neox_base():
    check_target_modules_safe(NEOX_TARGETS, is_neox=True)

def test_olmo_regex_string_allowed_on_non_neox_base():
    check_target_modules_safe(r".*\.(q_proj|k_proj|v_proj)$", is_neox=False)

def test_all_linear_still_rejected_on_non_neox_base():
    with pytest.raises(ValueError, match="all-linear"):
        check_target_modules_safe("all-linear", is_neox=False)
