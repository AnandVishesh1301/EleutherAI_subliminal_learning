"""The analysis-layer `collapse_law` shim must mirror `stats.collapse_law` automatically."""

from __future__ import annotations

import inspect

import pytest

from subliminal_learning.analysis import collapse_law as shim
from subliminal_learning.stats import collapse_law as stats


def _stats_api() -> list[str]:
    return [
        name
        for name, obj in vars(stats).items()
        if not name.startswith("_")
        and (inspect.isfunction(obj) or inspect.isclass(obj))
        and getattr(obj, "__module__", None) == stats.__name__
    ]


def test_every_stats_symbol_is_reachable_and_identical():
    names = _stats_api()
    assert names, "expected a non-empty public API in stats.collapse_law"
    for name in names:
        assert getattr(shim, name) is getattr(stats, name), name


@pytest.mark.parametrize(
    "name", ["separation_status", "max_margin_ratios", "separating_exponent_set", "tied_cells"]
)
def test_new_estimators_are_exported(name):
    assert name in shim.__all__
    assert getattr(shim, name) is getattr(stats, name)


def test_own_loaders_survive():
    assert callable(shim.load_law_runs)
    assert callable(shim.adapter_coherence)
    assert "load_law_runs" in shim.__all__ and "adapter_coherence" in shim.__all__


def test_unknown_name_raises_attribute_error():
    with pytest.raises(AttributeError, match="no attribute 'definitely_not_here'"):
        shim.definitely_not_here
