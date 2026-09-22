"""The repo's statistical machinery, importable from ``scripts/``, ``pipelines/`` and notebooks.

This package is where estimators live: the general toolkit (`core` -- bootstrap/pooling/AUC/
cluster-robust fits/JSON-safe scalars), the collapse-boundary law (`collapse_law` -- ``K``, the
exponent fit, ``K*``/θ̂ and their spreads), and the mediation ladder the collapse screens share
(`mediation`). Everything here is a pure function over numpy arrays or pandas frames: no disk IO,
no matplotlib, no discovery -- with one transparent exception, `memo`: a content-addressed,
opt-out file memo of the seeded permutation results (same inputs, same result, read back
instead of re-drawn), which never changes what a function returns.

Layering: ``subliminal_learning.analysis`` (loaders, record caches, notebook helpers) may import
this package; this package never imports ``analysis`` back, so importing (or calling anything in)
``subliminal_learning.stats`` never pulls the analysis layer in. The canonical cell-tag parser
`collapse_law.k_from_tag` needs lives one layer down in ``subliminal_learning.param_tags``.
The two analysis-layer *loaders* that used to sit in ``collapse_law`` (``load_law_runs``,
``adapter_coherence``) stayed behind in ``analysis.collapse_law`` for the same reason.

``analysis/stats.py`` and ``analysis/collapse_law.py`` remain as compatibility shims re-exporting
every public name from here, so existing notebooks and scripts keep working unchanged. This
``__init__`` deliberately re-exports nothing: new code imports the submodules
(``from subliminal_learning.stats import collapse_law as cl``,
``from subliminal_learning.stats.core import cluster_ols``).
"""

from __future__ import annotations
