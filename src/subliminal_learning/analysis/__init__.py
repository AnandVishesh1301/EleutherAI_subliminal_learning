"""Notebook-support helpers shared by the analysis notebooks in ``notebooks/``.

This subpackage is the *leaf* of the layering (``scripts/`` -> ``pipelines/`` ->
``src/subliminal_learning/``): it exists only to de-duplicate code that was copy-pasted
across the six analysis notebooks. Two rules keep it from leaking into the run path:

- Nothing in ``pipelines/`` or ``scripts/`` may import ``subliminal_learning.analysis``;
  it is imported only from notebooks (and its own tests).
- matplotlib lives *only* in ``analysis.plots`` (mirroring how the pandas-using ``*_io``
  modules are imported only where needed). ``analysis.setup.set_plot_style`` is the one
  concession -- it imports matplotlib lazily inside its body -- so importing this package
  or ``analysis.setup`` stays matplotlib-free. Import ``analysis.plots`` explicitly when
  you need the figure primitives.
"""

from __future__ import annotations

from subliminal_learning.analysis.frames import (
    MAIN_GRID,
    MODEL_ORDER,
    add_lora_label,
    cheapest_best,
    filter_grid,
    load_pref_wide,
    pref_to_wide,
    print_scope,
    scope_str,
    short_model,
)
from subliminal_learning.analysis.setup import bootstrap, find_root, set_plot_style

__all__ = [
    "MAIN_GRID",
    "MODEL_ORDER",
    "add_lora_label",
    "bootstrap",
    "cheapest_best",
    "filter_grid",
    "find_root",
    "load_pref_wide",
    "pref_to_wide",
    "print_scope",
    "scope_str",
    "set_plot_style",
    "short_model",
]
