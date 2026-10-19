"""Cross-model A→B trajectories against the sender's own same-model A→A trajectory, step by step.

For every treatment cross arm A→B whose curves exist (`trajectory_conditions.STATUS_IN`), the
diagonal A→A of the same animal and sender is looked up in the SAME data set (the core-set arms
or the dog per-checkpoint arms). A comparison pairs runs on shared (teacher, cell) slots alive in
both arms at each shared saved step; the "surviving fraction" is the per-step median over paired
slots of (A→B value) / (A→A value), value = (p_student − p_control) / receiver prior with the
matched control at the same step, with a 95% teacher-cluster bootstrap band
(`trajectory_asymmetry._boot_weights`). Sender and receiver must be the same model family and
size (both Pythia-160m variants or both OLMo-1B variants, `models.naming`); an arm failing that,
or whose same-set diagonal is missing or excluded, is listed as not comparable with the reason.
Pure functions over
`TrajectoryData`; no reads of the outputs tree.

The three curve figures are drawn from a small curve table (`curve_table`, one row per block,
panel, arm and step), which the notebook saves as
``notebooks/figures/asymmetry/trajectory/aa_ab/aa_ab_curves.csv``;
``notebooks/figures/asymmetry/trajectory/aa_ab/aa_ab_curves.py`` redraws the figures from that CSV in
seconds (`plot_curves`, `plot_diffinit_diffdata_animals`, `plot_olmo_curves` take the table).
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from subliminal_learning.analysis import result_cache
from subliminal_learning.analysis import trajectory_asymmetry as tas
from subliminal_learning.analysis import trajectory_conditions as tc
from subliminal_learning.models import naming

__all__ = [
    "NO_DIAGONAL", "NO_CONTROL", "PREFERRED_SET", "DIFFERENT_ARCH", "control_inventory", "same_architecture", "Comparison", "ComparisonSet",
    "find_comparisons", "paired_points", "median_band", "surviving_fraction", "comparison_stats",
    "summary_table", "conclusion_lines", "plot_coverage", "curve_table", "CURVE_COLUMNS", "plot_curves",
    "plot_diffinit_diffdata_animals", "plot_olmo_curves", "plot_surviving",
    "DISPLAY_NAMES", "display", "BASE_GROUPS", "CURVE_ANIMALS", "curve_blocks",
]

NO_DIAGONAL = "no A→A arm registered"
NO_CONTROL = "no per-step control"
#: Data set used when an (animal, sender, receiver) exists in several (per-step matched controls).
PREFERRED_SET = "per-checkpoint"
DIFFERENT_ARCH = "different model size/architecture"
ALPHA = 0.05
BLOCK_W = 7.0  # inches per 2x2 block column in _block_figure

SUMMARY_COLUMNS = ["family", "animal", "A→B", "n paired slots", "steps",
                   "A→A peak (p−c)/prior (step)", "A→B peak (p−c)/prior (step)",
                   "surviving @ A→A peak", "surviving @ last step", "first step p<.05",
                   "A→A > A→B every step"]


@dataclass(frozen=True)
class Comparison:
    """One A→B arm against its A→A diagonal, both from data set `set_name`."""

    ab: tc.ArmSpec
    aa: tc.ArmSpec
    set_name: str


@dataclass
class ComparisonSet:
    """`find_comparisons` output: the comparisons and one row per not-comparable A→B arm
    (columns ``set, family, animal, sender, receiver, A→B, reason``)."""

    comparisons: list[Comparison]
    not_comparable: pd.DataFrame


def _family(spec: tc.ArmSpec) -> str:
    return "Pythia" if spec.sender.startswith("pythia") else "OLMo"


def _raw(data: tc.TrajectoryData, label: str) -> pd.DataFrame:
    """Alive finite grid points of one arm (×prior)."""
    pts = tc._alive_points(data, [label])
    pts = pts[np.isfinite(pts["ratio"])]
    return pts[["teacher", "cell", "step", "ratio"]]


def _control_of(data: tc.TrajectoryData, label: str) -> str | None:
    return next((s.label for s in data.arms if s.kind == "control" and s.control_of == label), None)


def _control_points(data: tc.TrajectoryData, label: str) -> pd.DataFrame:
    """Matched control ×prior per (cell, step): median over its alive runs (one per cell)."""
    ctl = _control_of(data, label)
    if ctl is None:
        return pd.DataFrame(columns=["cell", "step", "control"])
    return (_raw(data, ctl).groupby(["cell", "step"])["ratio"].median()
            .rename("control").reset_index())


def _alive(data: tc.TrajectoryData, label: str) -> pd.DataFrame:
    """Alive points of one arm with ``ratio`` = (p_student − p_control) / receiver prior, the
    control being the matched control run of the same cell at the SAME step; a (run, step)
    without that control point is dropped."""
    pts = _raw(data, label).merge(_control_points(data, label), on=["cell", "step"])
    return pts.assign(ratio=pts["ratio"] - pts["control"])[["teacher", "cell", "step", "ratio"]]


def paired_points(ab_data: tc.TrajectoryData, ab: str, aa_data: tc.TrajectoryData,
                  aa: str) -> pd.DataFrame:
    """(teacher, cell, step) slots alive in both arms at shared saved steps.

    Columns ``teacher, cell, step, ab, aa`` ((p_student − p_control) / prior of each arm,
    control at the same step). Cells pair by their plain name (`trajectory_conditions.plain_twin`),
    so a full-length arm's plain cells meet a diagonal's ``_stop{N}`` cells of the same design.
    """
    def plain(df: pd.DataFrame) -> pd.DataFrame:
        return df.assign(cell=[tc.plain_twin(c) or c for c in df["cell"]])

    a = plain(_alive(ab_data, ab)).drop_duplicates(["teacher", "cell", "step"])
    b = plain(_alive(aa_data, aa)).drop_duplicates(["teacher", "cell", "step"])
    return a.merge(b, on=["teacher", "cell", "step"], suffixes=("_ab", "_aa")).rename(
        columns={"ratio_ab": "ab", "ratio_aa": "aa"})


def same_architecture(a: str, b: str) -> bool:
    """True when two base dirs are the same model family and size (`models.naming`)."""
    return (naming.family_of(a) == naming.family_of(b)
            and naming.base_spec(a).size == naming.base_spec(b).size)


def find_comparisons(sets: Mapping[str, tc.TrajectoryData]) -> ComparisonSet:
    """Every in-curves treatment A→B arm (receiver ≠ sender), matched to its A→A diagonal.

    Not comparable when sender and receiver differ in family or size (`DIFFERENT_ARCH`), else
    when the same-set diagonal (same animal, sender = receiver = A) is missing (`NO_DIAGONAL`)
    or not in curves (its inventory status), else when the A→B or A→A arm has no matched
    control point at any step (`NO_CONTROL`). Each (animal, sender, receiver) gives ONE row:
    when several sets carry it, the `PREFERRED_SET` candidate wins (else the first set).
    """
    invs = {name: tc.arm_inventory(d) for name, d in sets.items()}
    status = {name: {lab: row["status"] for lab, (row, _) in inv.items()}
              for name, inv in invs.items()}

    def no_control(name, data, label) -> bool:   # == _control_points(data, label).empty
        ctl = _control_of(data, label)
        return ctl is None or not invs[name][ctl][1]

    cands: dict[tuple, list] = {}
    for name, data in sets.items():
        diag = {(s.animal, s.sender): s for s in data.arms
                if s.kind == "treatment" and s.sender == s.receiver}
        for s in data.arms:
            if s.kind != "treatment" or s.sender == s.receiver or status[name][s.label] != tc.STATUS_IN:
                continue
            d = diag.get((s.animal, s.sender))
            if not same_architecture(s.sender, s.receiver):
                reason = DIFFERENT_ARCH
            elif d is None:
                reason = NO_DIAGONAL
            elif status[name][d.label] != tc.STATUS_IN:
                reason = f"A→A {d.label}: {status[name][d.label]}"
            elif no_control(name, data, s.label) or no_control(name, data, d.label):
                reason = NO_CONTROL
            else:
                reason = None
            cands.setdefault((s.animal, s.sender, s.receiver), []).append((name, s, d, reason))
    comps, rows = [], []
    for group in cands.values():
        name, s, d, reason = min(group, key=lambda g: (g[0] != PREFERRED_SET,
                                                       list(sets).index(g[0])))
        if reason is None:
            comps.append(Comparison(s, d, name))
        else:
            rows.append({"set": name, "family": _family(s), "animal": s.animal,
                         "sender": s.sender, "receiver": s.receiver, "A→B": s.label,
                         "reason": reason})
    nc = pd.DataFrame(rows, columns=["set", "family", "animal", "sender", "receiver", "A→B",
                                     "reason"])
    return ComparisonSet(sorted(comps, key=_order_key), nc)


def _steps(steps) -> str:
    st = sorted({int(x) for x in steps})
    if not st:
        return "none"
    run = [x for x in st if x <= 3750]
    head = f"{run[0]:,}…{run[-1]:,}" if len(run) > 2 else ", ".join(f"{x:,}" for x in run)
    rest = [f"{x:,}" for x in st if x > 3750]
    return ", ".join(([head] if run else []) + rest)


def control_inventory(sets: Mapping[str, tc.TrajectoryData], comps: ComparisonSet) -> pd.DataFrame:
    """Per treatment A→B arm (compared or not, every set): its set, whether it is the comparison
    used, its matched control arm, and the steps where that control has a point."""
    used = {(c.set_name, c.ab.label) for c in comps.comparisons}
    rows = []
    for name, data in sets.items():
        for s in data.arms:
            if s.kind != "treatment" or s.sender == s.receiver:
                continue
            cp = _control_points(data, s.label)
            rows.append({"set": name, "A→B": display(s.label), "used": (name, s.label) in used,
                         "control arm": display(_control_of(data, s.label) or "—"),
                         "control steps": _steps(cp["step"])})
    return pd.DataFrame(rows, columns=["set", "A→B", "used", "control arm", "control steps"])


_ANIMAL_ORDER = ("cat", "lion", "pig", "dog", "elephant")
_SENDER_ORDER = ("pythia-160m", "pythia-160m-seed1", "pythia-160m-seed2", "pythia-160m-seed3",
                 "pythia-160m-pilerepro-seed1234")


def _rank(seq, x) -> int:
    return seq.index(x) if x in seq else len(seq)


def _order_key(c: Comparison) -> tuple:
    """Panel / table order: family (Pythia first), animal, sender, receiver."""
    return (_family(c.ab) != "Pythia", _rank(_ANIMAL_ORDER, c.ab.animal),
            _rank(_SENDER_ORDER, c.ab.sender), c.ab.sender, c.ab.receiver)


def _weighted_median(vals: np.ndarray, tidx: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Per bootstrap draw, the (lower) weighted median of `vals` with weight ``w[:, tidx]``."""
    order = np.argsort(vals)
    v, wt = vals[order], w[:, tidx[order]]
    cum = np.cumsum(wt, axis=1)
    tot = cum[:, -1:]
    k = (cum >= tot / 2).argmax(axis=1)
    out = v[k]
    out[tot[:, 0] == 0] = np.nan
    return out


def median_band(pts: pd.DataFrame, col: str = "value") -> pd.DataFrame:
    """Per step: 95% teacher-cluster bootstrap band of the median of `col` over the rows
    (teachers resampled with replacement, `trajectory_asymmetry.N_BOOT` draws, seed
    `trajectory_asymmetry.BOOT_SEED`). Columns ``step, lo, hi``."""
    if pts.empty:
        return pd.DataFrame(columns=["step", "lo", "hi"])
    teachers = sorted(pts["teacher"].unique())
    w = tas._boot_weights(len(teachers))
    tpos = {t: i for i, t in enumerate(teachers)}
    rows = []
    for step, g in pts.groupby("step"):
        v = g[col].to_numpy(float)
        ok = np.isfinite(v)
        if ok.any():
            lo, hi = tas._band(_weighted_median(v[ok], g["teacher"].map(tpos).to_numpy()[ok], w))
            rows.append({"step": float(step), "lo": lo, "hi": hi})
    return pd.DataFrame(rows, columns=["step", "lo", "hi"])


def surviving_fraction(pairs: pd.DataFrame, *, tests: bool = True) -> pd.DataFrame:
    """Per step: median over paired slots of ab / aa, its 95% teacher-cluster bootstrap band
    (teachers resampled with replacement, `trajectory_asymmetry.N_BOOT` draws, seed
    `trajectory_asymmetry.BOOT_SEED`), the paired medians, and the two-sided Wilcoxon p of
    aa − ab over the slots (`trajectory_conditions._two_sided_p`). Values are control-subtracted
    and can be ≤ 0: slots with aa ≤ 0 at a step are dropped from that step's ratio (``n_dropped``);
    medians and p use all paired slots. `tests=False` skips the Wilcoxon test (``p`` = NaN);
    everything else, bands included, is unchanged.

    Columns ``step, n, frac, lo, hi, median_ab, median_aa, p, n_dropped``.
    """
    cols = ["step", "n", "frac", "lo", "hi", "median_ab", "median_aa", "p", "n_dropped"]
    if pairs.empty:
        return pd.DataFrame(columns=cols)
    teachers = sorted(pairs["teacher"].unique())
    w = tas._boot_weights(len(teachers))
    tpos = {t: i for i, t in enumerate(teachers)}
    rows = []
    for step, g in pairs.groupby("step"):
        d = (g["aa"] - g["ab"]).to_numpy(float)
        _, p = (tc._two_sided_p(d[np.isfinite(d)]) if tests and np.isfinite(d).sum()
                else (np.nan, np.nan))
        pos = g[g["aa"] > 0]
        r = (pos["ab"] / pos["aa"]).to_numpy(float)
        ok = np.isfinite(r)
        pos, r = pos[ok], r[ok]
        if not len(r):
            continue
        lo, hi = tas._band(_weighted_median(r, pos["teacher"].map(tpos).to_numpy(), w))
        rows.append({"step": float(step), "n": len(r), "frac": float(np.median(r)), "lo": lo,
                     "hi": hi, "median_ab": float(g["ab"].median()),
                     "median_aa": float(g["aa"].median()), "p": p, "n_dropped": len(g) - len(r)})
    return pd.DataFrame(rows, columns=cols)


#: `result_cache` namespace of the per-comparison `comparison_stats` entries.
CACHE_NAMESPACE = "trajectory_ab_vs_aa"


def _comparison_key(data: tc.TrajectoryData, c: Comparison, tests: bool) -> str | None:
    """Cache key of one comparison: the result-cache keys of its two arms and their matched
    controls (``data.arm_keys``), the bootstrap settings, `tests` and this module's code (plus
    `trajectory_asymmetry` / `trajectory_conditions`); None when `data` carries no arm keys."""
    keys = data.arm_keys
    labels = [c.ab.label, c.aa.label, _control_of(data, c.ab.label), _control_of(data, c.aa.label)]
    if keys is None or any(lab is not None and lab not in keys for lab in labels):
        return None
    return result_cache.fingerprint([], code=[sys.modules[__name__], tas, tc],
                                    params={"arms": [keys.get(lab) for lab in labels],
                                            "labels": labels, "set": c.set_name, "tests": tests,
                                            "n_boot": tas.N_BOOT, "seed": tas.BOOT_SEED})


def comparison_stats(sets: Mapping[str, tc.TrajectoryData], comps: ComparisonSet, *,
                     tests: bool = True, refresh_cache: bool | None = None) -> dict:
    """``{A→B label: (curve_ab, curve_aa, surviving)}``: each arm's curve (median and n over
    alive runs of (p_student − p_control) / prior, control at the same step) restricted to the
    paired steps, with ``lo, hi`` = `median_band` over those runs, and the `surviving_fraction`
    frame (`tests=False`: no per-step Wilcoxon, ``p`` = NaN).

    Each comparison is served from `result_cache` (namespace `CACHE_NAMESPACE`) when its data
    set came from a cached `trajectory_conditions.load_arms` and its arms' keys, the bootstrap
    settings and `tests` are unchanged (`refresh_cache=True` / env
    ``SL_RESULT_CACHE_REFRESH=1`` recomputes)."""
    out = {}
    refresh = tc._refresh_flag(refresh_cache)

    def banded(data, spec, steps):
        pts = _alive(data, spec.label)
        cur = pts.groupby("step")["ratio"].agg(median="median", n="size").reset_index()
        cur = cur.merge(median_band(pts, "ratio"), on="step", how="left")
        return cur[cur["step"].isin(steps)]

    for c in comps.comparisons:
        data = sets[c.set_name]

        def compute(c=c, data=data):
            surv = surviving_fraction(paired_points(data, c.ab.label, data, c.aa.label),
                                      tests=tests)
            steps = set(surv["step"])
            return banded(data, c.ab, steps), banded(data, c.aa, steps), surv

        key = _comparison_key(data, c, tests)
        out[c.ab.label] = (compute() if key is None else result_cache.cached(
            CACHE_NAMESPACE, f"{c.set_name} | {c.ab.label}", key, data.root, compute,
            refresh=refresh))
    return out


def _peak(curve: pd.DataFrame) -> tuple[float, float]:
    if curve.empty:
        return np.nan, np.nan
    r = curve.loc[curve["median"].idxmax()]
    return float(r["median"]), float(r["step"])


def summary_table(comps: ComparisonSet, stats: Mapping, *, tests: bool = True) -> pd.DataFrame:
    """One row per comparison (`SUMMARY_COLUMNS`); values (p_student − p_control) / prior with the
    control at the same step, absolute steps. `tests=False` (statistics not run): "first step
    p<.05" is "—"."""
    rows = []
    for c in comps.comparisons:
        cab, caa, surv = stats[c.ab.label]
        pairs_n = int(surv["n"].max()) if len(surv) else 0
        pa, sa = _peak(caa)
        pb, sb = _peak(cab)
        s = surv.set_index("step")
        sig = surv.loc[surv["p"] < ALPHA, "step"]
        rows.append({
            "family": _family(c.ab), "animal": c.ab.animal, "A→B": display(c.ab.label),
            "n paired slots": pairs_n, "steps": len(surv),
            "A→A peak (p−c)/prior (step)": f"{pa:.3g} ({sa:,.0f})",
            "A→B peak (p−c)/prior (step)": f"{pb:.3g} ({sb:,.0f})",
            "surviving @ A→A peak": float(s["frac"].get(sa, np.nan)),
            "surviving @ last step": float(surv["frac"].iloc[-1]) if len(surv) else np.nan,
            "first step p<.05": ("—" if not tests else f"{sig.iloc[0]:,.0f}" if len(sig)
                                 else "never"),
            "A→A > A→B every step": bool((surv["median_aa"] > surv["median_ab"]).all()),
        })
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def conclusion_lines(summary: pd.DataFrame, comps: ComparisonSet, *,
                     tests: bool = True) -> list[str]:
    """3–6 lines read off the summary table and the not-comparable list; `tests=False` omits the
    paired-p count."""
    if summary.empty:
        return ["No comparable A→B vs A→A pair."]
    last = summary["surviving @ last step"]
    at_peak = summary["surviving @ A→A peak"]
    n = len(summary)
    lines = [
        f"{n} A→B arms have a usable A→A trajectory; {len(comps.not_comparable)} do not "
        f"(reasons in the not-comparable list).",
        f"At the sender's A→A peak step the median surviving fraction is {np.nanmedian(at_peak):.2g} "
        f"(range {np.nanmin(at_peak):.2g}–{np.nanmax(at_peak):.2g}); at the last shared step "
        f"{np.nanmedian(last):.2g} (range {np.nanmin(last):.2g}–{np.nanmax(last):.2g}).",
        f"A→A exceeds A→B at every shared step in {int(summary['A→A > A→B every step'].sum())}/{n} "
        f"comparisons" + (f"; {int((summary['first step p<.05'] != 'never').sum())}/{n} reach a "
                          f"paired p < .05 at some step." if tests else "."),
    ]
    for fam, g in summary.groupby("family"):
        lines.append(f"{fam}: median surviving fraction at the last shared step "
                     f"{g['surviving @ last step'].median():.2g} over {len(g)} comparisons.")
    above = summary[last >= 1]
    if len(above):
        lines.append("A→B at or above A→A at the last shared step: " + ", ".join(above["A→B"]) + ".")
    return lines[:6]


# ----------------------------------------------------------------------------------------------
# plotting
# ----------------------------------------------------------------------------------------------
_COMPARABLE, _NO_AA, _MISSING = "●", "◐", "○"


# Display names (shared with the asymmetry notebook): `trajectory_conditions.DISPLAY_NAMES`.
DISPLAY_NAMES, display = tc.DISPLAY_NAMES, tc.display


def _short(base: str) -> str:
    return display(tc.short_base(base, family_context=base.startswith("pythia-160m")))


def plot_coverage(sets: Mapping[str, tc.TrajectoryData], comps: ComparisonSet, *,
                  fontsize: int = 9):
    """One glyph grid per (sender family, animal): rows = sender, columns = receiver; ● comparable,
    ◐ A→B in curves but no usable A→A, ○ A→B missing or not in curves; diagonal = ■ when A→A is
    in curves. One footer line per not-comparable sender with the reason."""
    import matplotlib.pyplot as plt

    status = {}
    for name, d in sets.items():
        st = tc.inventory_table(d).set_index("arm")["status"]
        for s in d.arms:
            if s.kind == "treatment":
                key = (s.animal, s.sender, s.receiver)
                if status.get(key) != tc.STATUS_IN:
                    status[key] = st[s.label]
    comp = {(c.ab.animal, c.ab.sender, c.ab.receiver) for c in comps.comparisons}
    nocomp = {(r.animal, r.sender, r.receiver) for r in comps.not_comparable.itertuples()}
    panels: dict[tuple[str, str], tuple[set, set]] = {}
    for animal, s, r in status:
        if s != r:
            fam = "Pythia" if s.startswith("pythia") else "OLMo"
            snd, bas = panels.setdefault((fam, animal), (set(), set()))
            snd.add(s)
            bas.update({s, r})
    keys = sorted(panels, key=lambda k: (k[0] != "Pythia", k[1]))
    ncols = min(4, len(keys))
    nrows = -(-len(keys) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.8 * nrows), squeeze=False)

    def order(b):
        return (not b.startswith("pythia-160m"), not b.startswith("pythia"), b)

    for ax, (fam, animal) in zip(axes.flat, keys):
        bases = sorted(panels[(fam, animal)][1], key=order)
        senders = sorted(panels[(fam, animal)][0], key=order)
        for i, s in enumerate(senders):
            for j, r in enumerate(bases):
                k = (animal, s, r)
                if s == r:
                    g = {None: ""}.get(status.get(k), "■" if status.get(k) == tc.STATUS_IN else "□")
                elif k in comp:
                    g = _COMPARABLE
                elif k in nocomp:
                    g = _NO_AA
                else:
                    g = _MISSING
                ax.text(j, i, g, ha="center", va="center", fontsize=fontsize + 4,
                        color="0.6" if g == _MISSING else "black")
        ax.set_xticks(range(len(bases)), [_short(b) for b in bases], rotation=45, ha="right",
                      fontsize=fontsize)
        ax.set_yticks(range(len(senders)), [_short(b) for b in senders], fontsize=fontsize)
        ax.set_xlim(-0.5, len(bases) - 0.5)
        ax.set_ylim(len(senders) - 0.5, -0.5)
        ax.set_xlabel("receiver B", fontsize=fontsize)
        ax.set_ylabel("sender A", fontsize=fontsize)
        ax.set_title(f"{animal} · {fam} senders", fontsize=fontsize + 1)
        ax.grid(False)
    for ax in axes.flat[len(keys):]:
        ax.axis("off")
    nc = comps.not_comparable
    foot = [f"{_COMPARABLE} A→B compared with A→A   {_NO_AA} A→B exists, no usable A→A   "
            f"{_MISSING} A→B missing / not in curves   ■ A→A in curves   □ A→A registered but excluded   blank diagonal = no A→A arm registered"]
    for (animal, sender), g in nc.groupby(["animal", "sender"], sort=False):
        recv = ", ".join(_short(r) for r in g["receiver"])
        foot.append(f"{animal} · {_short(sender)} → {{{recv}}}: {g['reason'].iloc[0]}")
    fig.text(0.01, 0.0, "\n".join(foot), ha="left", va="top", fontsize=fontsize - 1)
    fig.tight_layout()
    return fig


def _panels(comps: ComparisonSet) -> dict[tuple, list[Comparison]]:
    out: dict[tuple, list[Comparison]] = {}
    for c in comps.comparisons:
        out.setdefault((_family(c.ab), c.ab.animal, c.ab.sender, c.aa.label), []).append(c)
    return out


def _grid(n: int, ncols: int):
    import matplotlib.pyplot as plt

    ncols = min(ncols, n)
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.6 * ncols, 3.2 * nrows), squeeze=False)
    for ax in axes.flat[n:]:
        ax.axis("off")
    return fig, axes.flat


def _axes(ax, ylabel: str, fontsize: int, linthresh: float = 1.0) -> None:
    """Symlog y (linear within ±`linthresh`, values can be ≤ 0); dash-dot at 0 = control level."""
    from matplotlib.ticker import FuncFormatter, NullFormatter

    ax.set_yscale("symlog", linthresh=linthresh, linscale=1.0)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.axhline(0.0, color="0.4", ls="-.", lw=0.9)
    ax.set_xlabel("student step", fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.tick_params(labelsize=fontsize - 1)


def _colors(comps: list[Comparison], members: tuple[str, ...] | None = None) -> dict[str, tuple]:
    """Colour per A→B label, fixed by RECEIVER: tab10 index of the receiver's short name in
    `members` (a block's base list). Without `members`, the first `BASE_GROUPS` group holding
    both sender and receiver gives the index; otherwise the receiver's rank among the sorted
    receivers of `comps`. Independent of which receivers / which sender a panel shows."""
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("tab10")
    rest = sorted({_short(c.ab.receiver) for c in comps})
    out = {}
    for c in comps:
        s, r = _short(c.ab.sender), _short(c.ab.receiver)
        group = members or next((m for _, m in BASE_GROUPS if m and s in m and r in m), None)
        i = group.index(r) if group and r in group else rest.index(r)
        out[c.ab.label] = cmap(i % 10)
    return out


# Curve blocks: (title, members as `_short` display names), in display order; the member order
# fixes each receiver's colour (`_colors`).
BASE_GROUPS = (
    ("diff init, diff data order: orig, fs1–fs3", ("orig", "fs1", "fs2", "fs3")),
    ("same init, diff data order: orig, ds1–ds3", ("orig", "ds1", "ds2", "ds3")),
    ("diff init, same data order: Rerun1234, wc1, wc2", ("Rerun1234", "wc1", "wc2")),
    ("Pythia-160M re-pretrain: orig, Modern1234, Rerun1234", ("orig", "Modern1234", "Rerun1234")),
    ("OLMo-1B: dolma, c4, dclm and their seed-2 copies",
     ("OLMo-dolma", "OLMo-c4", "OLMo-dclm", "OLMo-dolma-aux2", "OLMo-c4-aux2", "OLMo-dclm-aux2")),
)
CURVE_ANIMALS = ("dog", "cat", "lion", "pig", "elephant")


def curve_blocks(comps: ComparisonSet) -> dict[str, list[tuple[str, list[tuple[str, list[Comparison]]]]]]:
    """``{animal: [(group title, [(sender, comparisons)])]}`` in `CURVE_ANIMALS` / `BASE_GROUPS`
    order. A block keeps only A→B arms whose sender and receiver are both in the group; a sender
    in several groups appears in each.
    Senders without a comparison in the group and groups without any are omitted."""
    out = {}
    animals = list(CURVE_ANIMALS) + sorted({c.ab.animal for c in comps.comparisons} - set(CURVE_ANIMALS))
    for animal in animals:
        blocks = []
        for title, members in BASE_GROUPS:
            per: dict[str, list[Comparison]] = {}
            for c in comps.comparisons:
                if c.ab.animal != animal:
                    continue
                s, r = _short(c.ab.sender), _short(c.ab.receiver)
                if s in members and r in members:
                    per.setdefault(c.ab.sender, []).append(c)
            per = dict(sorted(per.items(), key=lambda kv: members.index(_short(kv[0]))))
            if per:
                blocks.append((title, list(per.items())))
        if blocks:
            out[animal] = blocks
    return out


def _name(base: str) -> str:
    """Curve-panel name: `_short`, OLMo bases without the ``OLMo-`` prefix (the block title
    names the model) and ``-aux2`` written ``-seed2`` (``→dclm``, ``→c4-seed2``)."""
    s = _short(base)
    return s.removeprefix("OLMo-").replace("-aux2", "-seed2") if s.startswith("OLMo-") else s


CURVE_COLUMNS = ["animal", "block", "sender", "receiver", "role", "step", "n", "median", "lo", "hi"]
AA, AB = "A→A", "A→B"


def curve_table(stats: Mapping, comps: ComparisonSet) -> pd.DataFrame:
    """The curves the three curve figures draw, one row per (block, panel, arm, step)
    (`CURVE_COLUMNS`): animal, block (`BASE_GROUPS` title), sender / receiver (`_name` display
    names, as in the legends), role `AA` / `AB`, step, n alive runs, median, band lo / hi (from
    `comparison_stats`). Row order = `curve_blocks` order (block, sender, then A→A before the
    A→B arms in comparison order); saved as ``aa_ab_curves.csv`` by the notebook."""
    frames = []

    def add(cur, animal, block, sender, receiver, role):
        cur = cur.sort_values("step")
        frames.append(pd.DataFrame({"animal": animal, "block": block, "sender": _name(sender),
                                    "receiver": _name(receiver), "role": role,
                                    "step": cur["step"].to_numpy(float),
                                    "n": cur["n"].to_numpy(int), "median": cur["median"].to_numpy(float),
                                    "lo": cur["lo"].to_numpy(float), "hi": cur["hi"].to_numpy(float)}))

    for animal, blocks in curve_blocks(comps).items():
        for title, senders in blocks:
            for sender, cs in senders:
                caa = pd.concat([stats[c.ab.label][1] for c in cs]).drop_duplicates("step")
                add(caa, animal, title, sender, sender, AA)
                for c in cs:
                    add(stats[c.ab.label][0], animal, title, sender, c.ab.receiver, AB)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=CURVE_COLUMNS)


def _member_names(members: tuple[str, ...]) -> tuple[str, ...]:
    """`BASE_GROUPS` members (`_short` names) written as `_name` writes them."""
    return tuple(m.removeprefix("OLMo-").replace("-aux2", "-seed2") if m.startswith("OLMo-") else m
                 for m in members)


def _table_colors(receivers, members: tuple[str, ...] | None) -> dict[str, tuple]:
    """`_colors` rule on `_name` receivers: tab10 index in the block's `members`, else rank
    among the sorted `receivers`."""
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("tab10")
    group = _member_names(members) if members else ()
    rest = sorted(set(receivers))
    return {r: cmap((group.index(r) if r in group else rest.index(r)) % 10) for r in receivers}


def _draw_block(sf, title: str, block: pd.DataFrame, fontsize: int,
                members: tuple[str, ...] | None = None) -> None:
    """One 2×2 block in subfigure `sf` from its `curve_table` rows: one panel per sender, A→A bold
    black, each A→B coloured by receiver (`_table_colors` over the block's `members`, so one
    colour per receiver in every panel), 95% bootstrap bands of the median, symlog y; block
    title `title`."""
    senders = list(pd.unique(block["sender"]))
    axes = sf.subplots(2, 2, squeeze=False).ravel()
    for ax in axes[len(senders):]:
        ax.set_visible(False)
    for ax, sender in zip(axes, senders):
        rows = block[block["sender"] == sender]
        caa, abr = rows[rows["role"] == AA], rows[rows["role"] == AB]
        receivers = list(pd.unique(abr["receiver"]))
        col = _table_colors(receivers, members)
        ax.fill_between(caa["step"], caa["lo"], caa["hi"], color="0.6", alpha=0.3, lw=0)
        ax.plot(caa["step"], caa["median"], color="black", lw=2.4,
                label=f"{sender}→{sender} (A→A)")
        for r in receivers:
            cab = abr[abr["receiver"] == r]
            ax.fill_between(cab["step"], cab["lo"], cab["hi"], color=col[r], alpha=0.15, lw=0)
            ax.plot(cab["step"], cab["median"], color=col[r], lw=1.3, marker=".", label=f"→{r}")
        _axes(ax, "median Δp(trait) / receiver prior", fontsize)
        ax.set_title(f"{sender} sends", fontsize=fontsize + 1)
        ax.legend(fontsize=fontsize - 2, loc="upper center", bbox_to_anchor=(0.5, -0.22),
                  ncol=2, frameon=False)
    sf.suptitle(title, fontsize=fontsize + 2, fontweight="bold")


def _blocks(table: pd.DataFrame, animal: str, block: str | None = None) -> list[tuple[str, pd.DataFrame]]:
    """``[(block title, rows)]`` of `animal` in table order, optionally only block `block`."""
    t = table[table["animal"] == animal]
    return [(b, t[t["block"] == b]) for b in pd.unique(t["block"]) if block is None or b == block]


def _block_figure(blocks: list[tuple[str, pd.DataFrame]], fontsize: int):
    """Figure with `blocks` (``[(animal, block title, rows)]``) laid out two per row
    (`_draw_block`); each block titled ``"{animal} · {block title}"``."""
    import matplotlib.pyplot as plt

    nb = len(blocks)
    brow, bcol = -(-nb // 2), min(2, nb)
    fig = plt.figure(figsize=(BLOCK_W * bcol, 7.6 * brow + 0.6), layout="constrained")
    subs = np.atleast_1d(fig.subfigures(brow, bcol, wspace=0.04, hspace=0.04)).ravel()
    for sf in subs[nb:]:
        sf.set_visible(False)
    for sf, (animal, title, rows) in zip(subs, blocks):
        members = next((m for t, m in BASE_GROUPS if t == title), None)
        _draw_block(sf, f"{animal} · {title}", rows, fontsize, members)
    return fig


def plot_curves(table: pd.DataFrame, *, fontsize: int = 9, min_blocks: int = 1) -> dict:
    """One figure per animal of the `curve_table` `table`: 2×2 blocks of panels, two blocks per
    row, one panel per sender of the group: A→A bold black, each in-group A→B coloured by
    receiver; median (p_student − p_control) / prior over alive runs at the paired steps (control
    at the same step), 95% teacher-cluster bootstrap band of the median, symlog y, dash-dot = 0
    (control level). Block titles carry the animal. Only animals with at least `min_blocks`
    blocks get a figure (others are never created). Returns ``{animal: figure}``."""
    out = {}
    for animal in pd.unique(table["animal"]):
        blocks = _blocks(table, animal)
        if len(blocks) >= min_blocks:
            out[animal] = _block_figure([(animal, t, r) for t, r in blocks], fontsize)
    return out


DIFFINIT_DIFFDATA_ANIMALS = ("dog", "cat", "lion", "pig")


def plot_diffinit_diffdata_animals(table: pd.DataFrame, *, fontsize: int = 9):
    """One figure: the "diff init, diff data order" block (orig, fs1–fs3) of each animal in
    `DIFFINIT_DIFFDATA_ANIMALS` order, drawn exactly as in `plot_curves`, two blocks per row;
    animals without such a comparison are skipped. Returns the figure, or None if no animal has one."""
    group = BASE_GROUPS[0][0]
    blocks = [(a, t, r) for a in DIFFINIT_DIFFDATA_ANIMALS for t, r in _blocks(table, a, group)]
    return _block_figure(blocks, fontsize) if blocks else None


def plot_olmo_curves(table: pd.DataFrame, *, animal: str = "elephant", fontsize: int = 9):
    """One figure: the OLMo-1B block (`BASE_GROUPS` last entry) of `animal`, drawn exactly as in
    `plot_curves`; one panel per OLMo sender with an A→A diagonal, one fixed colour per
    receiver over the six OLMo bases. Returns the figure, or None if `animal` has no such block."""
    blocks = [(animal, t, r) for t, r in _blocks(table, animal, BASE_GROUPS[-1][0])]
    return _block_figure(blocks, fontsize) if blocks else None


def plot_surviving(stats: Mapping, comps: ComparisonSet, *, ncols: int = 4, fontsize: int = 9):
    """Same panel grid: per receiver, median over paired slots of A→B / A→A ((p − p_control)/prior) with
    its 95% teacher-cluster bootstrap band (values control-subtracted at the same step; slots with
    A→A ≤ 0 dropped per step); solid line at 1 = A→B equals A→A, symlog y."""
    panels = _panels(comps)
    fig, axes = _grid(len(panels), ncols)
    for ax, ((fam, animal, sender, _), cs) in zip(axes, panels.items()):
        col = _colors(cs)
        for c in cs:
            s = stats[c.ab.label][2]
            ax.plot(s["step"], s["frac"], color=col[c.ab.label], lw=1.3, marker=".",
                    label=f"→{_short(c.ab.receiver)} (n≤{int(s['n'].max()) if len(s) else 0}, "
                          f"dropped ≤{int(s['n_dropped'].max()) if len(s) else 0})")
            ax.fill_between(s["step"], s["lo"], s["hi"], color=col[c.ab.label], alpha=0.15, lw=0)
        _axes(ax, "surviving fraction (A→B / A→A)", fontsize, linthresh=0.01)
        ax.axhline(1.0, color="black", lw=0.9)
        ax.set_title(f"{animal} · {_short(sender)} sends", fontsize=fontsize + 1)
        ax.legend(fontsize=fontsize - 3, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    fig.tight_layout()
    return fig
