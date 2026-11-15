"""The repo's statistical machinery, importable from ``scripts/``, ``pipelines/`` and notebooks.

This package is where estimators live: the general toolkit (`core` -- bootstrap/pooling/AUC/
cluster-robust fits/JSON-safe scalars), the transfer significance / pool-robustness tests and
the transfer-direction tests (`transfer_direction_tests`). Everything
here is a pure function over numpy arrays or pandas frames: no disk IO, no matplotlib, no
discovery -- with one transparent exception, `memo`: a content-addressed, opt-out file memo of
the seeded permutation results (same inputs, same result, read back instead of re-drawn),
which never changes what a function returns.

The collapse-boundary law (the exponent fit, ``K*``/θ̂ and their spreads, the K dose
coordinate ``lr * alpha**b * rank**c`` and the mediation ladder) lives with the
catastrophic-forgetting program in the sibling lora-forgetting repo
(``lora_forgetting.stats.{collapse_law,hp_scale,mediation}``); nothing here evaluates K.

Layering: ``subliminal_learning.analysis`` (loaders, record caches, notebook helpers) may import
this package; this package never imports ``analysis`` back, so importing (or calling anything in)
``subliminal_learning.stats`` never pulls the analysis layer in.

This ``__init__`` deliberately re-exports nothing: code imports the submodules
(``from subliminal_learning.stats.core import cluster_ols``); the former ``analysis/stats.py``
shim is gone.
"""


from __future__ import annotations
