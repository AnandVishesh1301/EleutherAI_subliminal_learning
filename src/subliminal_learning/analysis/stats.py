"""Compatibility shim: the statistics toolkit now lives in ``subliminal_learning.stats.core``.

The notebooks and scripts that grew up importing ``analysis.stats`` keep working through this
module; new code should import ``subliminal_learning.stats.core`` directly (see that package's
docstring for the layering). Nothing is defined here -- every name below is the same object as
in `subliminal_learning.stats.core`.
"""

from __future__ import annotations

from subliminal_learning.stats.core import (
    BootstrapResult,
    PooledResult,
    binned_fraction,
    bootstrap_ci,
    cluster_logit,
    cluster_ols,
    delta_ratio_ci,
    ecdf,
    interp_crossing,
    json_scalar,
    mannwhitney_auc,
    pool_fixed_effect,
    quantile_table,
    sign_binom_test,
    two_proportion_z,
    tv_js,
)

__all__ = [
    "BootstrapResult",
    "PooledResult",
    "binned_fraction",
    "bootstrap_ci",
    "cluster_logit",
    "cluster_ols",
    "delta_ratio_ci",
    "ecdf",
    "interp_crossing",
    "json_scalar",
    "mannwhitney_auc",
    "pool_fixed_effect",
    "quantile_table",
    "sign_binom_test",
    "two_proportion_z",
    "tv_js",
]
