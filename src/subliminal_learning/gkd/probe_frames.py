"""Pure reductions over the GKD probe-trajectory frame (`gkd.probe_io.load_gkd_probe_trajectories`).

The trait is known to peak mid-run and decay, so two reductions coexist: the last probe of
each leaf (`end_of_run_probes`) and its maximum-``p_target`` probe (`peak_probes`).
"""

from __future__ import annotations

import pandas as pd


def end_of_run_probes(probes: pd.DataFrame) -> pd.DataFrame:
    """Reduce a probe-trajectory frame to each leaf's last probe row.

    Args:
        probes: Output of `load_gkd_probe_trajectories`.

    Returns:
        One row per leaf (``run_dir``), at that leaf's max ``step``.
    """
    if probes.empty:
        return probes.copy()
    idx = probes.groupby("run_dir")["step"].idxmax()
    return probes.loc[idx].reset_index(drop=True)


def peak_probes(probes: pd.DataFrame) -> pd.DataFrame:
    """Reduce a probe-trajectory frame to each leaf's max-``p_target`` row.

    The trait is known to peak mid-run and decay (the campaign's dose-ladder
    motivation), so the last probe systematically understates a recipe that
    spikes early -- this is the complementary reduction to `end_of_run_probes`.

    Args:
        probes: Output of `load_gkd_probe_trajectories`.

    Returns:
        One row per leaf (``run_dir``), at that leaf's max ``p_target``.
    """
    if probes.empty:
        return probes.copy()
    idx = probes.groupby("run_dir")["p_target"].idxmax()
    return probes.loc[idx].reset_index(drop=True)
