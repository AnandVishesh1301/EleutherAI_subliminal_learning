"""Notebook-support helpers shared by the analysis notebooks in ``notebooks/``.

This subpackage is the notebook-support layer: frames, labels, plots, record caches, and
the loaders the analysis notebooks share. The substantial *calculations* live one level
down in ``subliminal_learning.stats`` (estimators; the LoRA/W0 tensor math moved with the
collapse program to the lora-forgetting repo), which this package may import but which never
import it back;
``analysis.stats`` and ``analysis.collapse_law`` are compatibility shims over ``stats``.
Two rules keep this package from leaking into the run path:

- NEW run-path code -- ``scripts/`` and everything in ``pipelines/``, plus the domain
  packages under ``src/`` -- must not import ``subliminal_learning.analysis``; put shared
  computation in ``stats`` instead. Analysis-CLI scripts (one-off scans/screens) may use its
  loaders, and notebooks (plus its own tests) remain the primary importers. Three standing
  carve-outs predate the ``stats`` extraction and are to be migrated with it:
  ``numbers/manifest_io.py``,
  ``evaluation/preference_results_io.py``, ``steering/eas_results_io.py`` (``analysis.frames``
  loaders).
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
    cheapest_specific,
    filter_grid,
    load_pref_wide,
    parse_param_tag,
    pref_to_wide,
    print_scope,
    rank_specific,
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
    "cheapest_specific",
    "filter_grid",
    "find_root",
    "load_pref_wide",
    "parse_param_tag",
    "pref_to_wide",
    "print_scope",
    "rank_specific",
    "scope_str",
    "set_plot_style",
    "short_model",
]
