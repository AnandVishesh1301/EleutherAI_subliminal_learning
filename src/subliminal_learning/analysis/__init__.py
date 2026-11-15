"""Notebook-support layer shared by the analysis notebooks in ``notebooks/`` and the figure scripts.

Layout (2026-11 regroup):

- ``loaders/``       -- disk readers: frames, registries, result/probe caches, student trees,
                        the transfer-significance loaders (``significance_*_io``), number manifests;
- ``transfer/``      -- transfer readouts (matrix, headline, families, factors, asymmetry,
                        conditions, specificity, direction);
- ``trajectories/``  -- checkpoint-trajectory readouts (``readers`` + the AB/AA, asymmetry,
                        conditions and dd1b-ladder analyses);
- ``checkpoints/``   -- pretraining-checkpoint cross-readouts and their factorization;
- ``setup``, ``plots``, ``exclusions`` (+ the ``labels`` alias the lora-forgetting repo imports).

The substantial *calculations* live one level down in ``subliminal_learning.stats``
(estimators), which this package may import but which never imports it back. The
collapse-boundary law lives in the lora-forgetting repo (``analysis.exclusions`` only READS
the integrity-anchor artifact that program writes, to exclude collapsed students).

Two rules keep this package from leaking into the run path (enforced by
``tests/unit/test_layering.py``):

- Only ``analysis/``, ``figures/`` and ``pipelines/`` import ``subliminal_learning.analysis``;
  the domain and ``*_io`` modules get shared loaders from ``utils/`` (``utils.frames``,
  ``utils.record_cache``) and shared computation from ``stats/``.
- matplotlib lives *only* in ``analysis.plots`` (and the figure scripts).
  ``analysis.setup.set_plot_style`` is the one concession -- it imports matplotlib lazily
  inside its body -- so importing this package or ``analysis.setup`` stays matplotlib-free.
"""

from __future__ import annotations

from subliminal_learning.analysis.loaders.frames import (
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
