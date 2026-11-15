"""The run_logs registry layer against the REAL registry files. Integration: needs the machine-local run_logs registries / output tree (skips elsewhere).
"""

import pytest
from subliminal_learning.analysis.loaders import registry
from subliminal_learning.analysis.trajectories import trajectory_conditions as tc
from subliminal_learning.analysis.transfer import transfer_headline as th
from subliminal_learning.analysis.transfer import transfer_asymmetry as ta
from subliminal_learning.analysis.transfer import transfer_factors as tf
from subliminal_learning.analysis.transfer import transfer_family_complete as tfc


def _needs_registry(name: str) -> None:
    """Skip unless the configured run_logs registry holds `name` (resolved at call time)."""
    try:
        present = registry.registry_path(name).is_file()
    except FileNotFoundError:
        present = False
    if not present:
        pytest.skip(f"run_logs registry file {name} not on this machine")


def test_module_constants_load_lazily():
    _needs_registry(tc.EARLYSTOP_REGISTRIES_JSON)
    assert isinstance(tc.EARLYSTOP_REGISTRIES, tuple)
    assert isinstance(th.XARCH_DIRECTION_ARMS, tuple)
    with pytest.raises(AttributeError):
        tc.NOT_A_CONSTANT  # noqa: B018


def test_asymmetry_factor_complete_constants_load_lazily():
    _needs_registry(ta.SPEC_NAME)
    assert isinstance(ta.FAMILIES, dict) and isinstance(ta._PYTHIA_DESIGNS, tuple)
    assert isinstance(tf._CORE65_MANIFESTS, tuple) and isinstance(tf.XARM_INVENTORY_REL, str)
    assert isinstance(tfc.GRID_PATHS, dict)
    for mod in (ta, tf, tfc):
        with pytest.raises(AttributeError):
            mod.NOT_A_CONSTANT  # noqa: B018
