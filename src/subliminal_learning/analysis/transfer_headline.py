"""The headline existence-by-family table of notebooks/transfer_significance.ipynb (§0.0).

One block per relationship family (same-model, same init / different data order, ...,
cross-architecture), each read on FINISHED sets only -- the `transfer_significance_io`
finished-set rule: every listed teacher with an ftseed-1 test endpoint on every design cell of
the arm and a matched control on every cell; an arm short of that is dropped and counted.
Within a family one row per animal view: ``dominant`` (the family's main animal) and, when at
least two animals carry a finished arm, ``stratified`` (strata = (animal, arm), each block
against its own control maximum -- the existence pooling, not the family-comparison unit).

Analysis layer: arms are loaded through the loaders `significance_table` uses
(`load_registry_arm` for registry arms, `load_cell_matrix` + `family_arm_set` /
`same_model_arms` for the Pythia seed families); the test is
`stats.transfer_significance_tests.stratified_exceedance_permutation`. The per-run summaries
(×prior, ×control) are descriptive quantiles, computed per arm and shown as min / median / max
across the family's arms.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from subliminal_learning.analysis import transfer_significance_io as tio
from subliminal_learning.config import resolve_output_root
from subliminal_learning.stats import transfer_significance_tests as tt

__all__ = [
    "HeadlineArm",
    "HeadlineFamily",
    "HEADLINE_COLUMNS",
    "SUMMARY_STATS",
    "arm_summary",
    "triples",
    "DIVERGED_EXCLUDE",
    "diverged_note",
    "family_headline_arms",
    "registry_headline_arms",
    "headline_families",
    "headline_table",
    "style_headline_table",
    "view_permutation",
    "null_strip_figure",
]

#: The five per-run summaries of each ratio block, per arm.
SUMMARY_STATS = ("mean", "median", "p75", "p95", "max")

HEADLINE_COLUMNS = ["family", "animal view", "arms", "arms used", "design", "runs", "exceed.",
                    "exceed. %", "null 95th", "null 95th %", "perm p", "prior",
                    *[f"×prior {s}" for s in SUMMARY_STATS],
                    *[f"×control {s}" for s in SUMMARY_STATS],
                    "note"]

STRATIFIED, DOMINANT = "stratified", "dominant"


@dataclass(frozen=True)
class HeadlineArm:
    """One arm of a headline family: finished (`matrix` set) or dropped (`dropped` set).

    Attributes:
        label: Short arm label (``sender→receiver``, or the registry name).
        animal: Trait animal.
        matrix: The matched `tio.CellMatrix` (all-finite, control last) on a finished arm.
        prior: The receiver's test prior (`tio.student_tree_prior`).
        dropped: Why the arm is out (unfinished set), else ``None``.
        note: Anything qualifying the arm (selection cells of a pilot-chosen set, the diverged
            runs excluded from it).
        diverged: ``(teacher, cell)`` runs excluded as diverged (``diverged="exclude"`` only);
            they stay NaN in `matrix`, so no statistic sees them.
    """

    label: str
    animal: str
    matrix: tio.CellMatrix | None
    prior: float
    dropped: str | None = None
    note: str = ""
    diverged: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class HeadlineFamily:
    """One family block: its arms across animals and the labels the table shows."""

    family: str
    arms_label: str
    design: str
    dominant: str
    arms: list[HeadlineArm] = field(default_factory=list)
    note: str = ""


# ----------------------------------------------------------------------------------------------
# per-arm summaries
# ----------------------------------------------------------------------------------------------
def _quantiles(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {s: float("nan") for s in SUMMARY_STATS}
    return {"mean": float(x.mean()), "median": float(np.median(x)),
            "p75": float(np.percentile(x, 75)), "p95": float(np.percentile(x, 95)),
            "max": float(x.max())}


def arm_summary(matrix: tio.CellMatrix, prior: float) -> dict:
    """One arm's descriptive numbers: treatment runs, exceedances over the arm's control
    maximum, and the `SUMMARY_STATS` of the per-run ``p / prior`` and ``p / control`` (the
    matched control of the same cell) ratios.

    Args:
        matrix: The arm's `tio.CellMatrix`.
        prior: The receiver's test prior.

    Returns:
        ``{"runs", "exceed.", "xprior": {stat: value}, "xcontrol": {stat: value}}``.
    """
    ctrl = matrix.values[:, matrix.control_col]
    treat = np.delete(matrix.values, matrix.control_col, axis=1)
    ok = np.isfinite(treat)
    ctrl_max = np.nanmax(ctrl) if np.isfinite(ctrl).any() else np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        xctrl = treat / ctrl[:, None]
    return {"runs": int(ok.sum()), "exceed.": int((treat[ok] > ctrl_max).sum()),
            "xprior": _quantiles(treat[ok] / prior), "xcontrol": _quantiles(xctrl[ok])}


def triples(per_arm: Sequence[Mapping[str, float]]) -> dict[str, tuple[float, float, float]]:
    """``{stat: (min, median, max)}`` across arms of each per-arm `SUMMARY_STATS` value."""
    out = {}
    for s in SUMMARY_STATS:
        v = np.asarray([a[s] for a in per_arm], dtype=float)
        v = v[np.isfinite(v)]
        out[s] = ((float(v.min()), float(np.median(v)), float(v.max())) if v.size
                  else (np.nan, np.nan, np.nan))
    return out


# ----------------------------------------------------------------------------------------------
# arm loading (through the loaders significance_table uses)
# ----------------------------------------------------------------------------------------------
#: ``diverged="exclude"``: a student whose adapter went non-finite (`tio.DIVERGED_MARKER` in its
#: run leaf) counts as a COMPLETE run for the finished-set rule and is excluded from every
#: statistic -- its value stays NaN, which no test reads.
DIVERGED_EXCLUDE = "exclude"


def diverged_note(runs: Sequence[tuple[str, str]]) -> str:
    """The note naming the diverged runs excluded from an arm (``""`` when there are none)."""
    if not runs:
        return ""
    return (f"{len(runs)} diverged run{'s' if len(runs) > 1 else ''} excluded ("
            + "; ".join(f"teacher {t}, cell {c}" for t, c in runs) + ")")

def family_headline_arms(
    arms: Sequence[tio.FamilyArm] | Mapping[str, Sequence[tio.FamilyArm]],
    animals: Sequence[str],
    teachers_by_animal: Mapping[str, Sequence[str]],
    *,
    cells: Sequence[str] | str | Path,
    ftseed: int = 1,
    prompt_set: str = "test",
    output_root: Path | None = None,
    diverged: str = "drop",
    **kw,
) -> list[HeadlineArm]:
    """The Pythia seed-family arms (`tio.family_arm_set` / `tio.same_model_arms`) per animal,
    under the finished-set rule: a control on every design cell and every teacher finite on
    every cell, else the arm is dropped with the shortfall in `HeadlineArm.dropped`.

    Args:
        arms: The family's `tio.FamilyArm` list, or ``{animal: arms}`` when the animals carry
            different arm sets (cat / lion / pig have no ds-sender students).
        animals: Animals to read, in order.
        teachers_by_animal: ``{animal: param_tags}``.
        cells: The design (a cells JSON path or a list).
        diverged: ``"drop"`` (default) leaves a diverged student counting as a run that is not
            there, so its arm is short of a finished set; `DIVERGED_EXCLUDE` counts it as
            COMPLETE -- the arm stays finished, the run keeps its NaN (no statistic reads it)
            and it is named on `HeadlineArm.diverged` and in the arm's note.
        ftseed, prompt_set, output_root, **kw: Forwarded to `tio.load_cell_matrix`.
    """
    design = tio.resolve_cells(cells)
    root = resolve_output_root(output_root)
    out, priors = [], {}
    for animal in animals:
        teachers = list(teachers_by_animal[animal])
        for a in (arms[animal] if isinstance(arms, Mapping) else arms):
            m = tio.load_cell_matrix(animal, a.sender, a.receiver, teachers, cells=design,
                                     ftseed=ftseed, prompt_set=prompt_set, output_root=root,
                                     **kw)
            label = f"{a.sender}→{a.receiver}"
            key = (animal, tio.base_name(a.receiver))
            treat = np.delete(m.values, m.control_col, axis=1)
            n_students = len(design) * len(teachers)
            div: list[tuple[str, str]] = []
            if diverged == DIVERGED_EXCLUDE and m.missing and len(m.cells) == len(design):
                _, sender_tree, dtag, tsd = tio._arm_trees(root, animal, a.sender, a.receiver,
                                                           **kw)
                div = tio.diverged_missing(m, sender_tree, dataset_tag=dtag, tseed=tsd,
                                           ftseed=ftseed,
                                           teacher_dataset_tags=kw.get("teacher_dataset_tags"))
            short = [x for x in m.missing if x not in set(div)]
            if len(m.cells) < len(design) or short:
                n_run = int(np.isfinite(treat).sum())
                why = ("not started" if n_run == 0 else
                       f"students {n_run}/{n_students}, controls {len(m.cells)}/{len(design)}")
                out.append(HeadlineArm(label, animal, None, float("nan"), dropped=why))
                continue
            if key not in priors:
                priors[key] = tio.student_tree_prior(animal, a.receiver, prompt_set=prompt_set,
                                                     output_root=root)
            out.append(HeadlineArm(label, animal, m, priors[key], note=diverged_note(div),
                                   diverged=tuple(div)))
    return out


def registry_headline_arms(arms_json: str | Path, names: Sequence[str], *,
                           output_root: Path | None = None,
                           include_selection_cells: bool = False,
                           diverged: str = "drop") -> list[HeadlineArm]:
    """Registry arms by NAME, read through `tio.load_registry_arm` (the same design,
    ``exclude_cells`` / ``cells`` / ``control_dataset_tag`` handling and finished-set rule as
    `tio.significance_table`). An unfinished arm is dropped with its coverage counts; the
    auto-excluded on-disk strays, a pilot-selected set's excluded selection cells, and a
    ``held_out`` entry's selection cells, go to `HeadlineArm.note`.

    Args:
        arms_json: Registry path.
        names: The arm names to read, in the order they should be returned.
        output_root: Output tree root; defaults to the paths YAML's.
        include_selection_cells: Read every arm on its FULL design -- the registry's
            ``exclude_cells`` are ignored (for the referenced arms of a ``cells_from_arm`` too)
            and a ``held_out`` entry adds no note, so a pilot-selected teacher set is read on
            every design cell, selection cell included. No registry uses ``exclude_cells``
            any more, so the flag is a no-op on them; the registry files are never touched.
        diverged: As `family_headline_arms` -- `DIVERGED_EXCLUDE` counts a diverged student as
            a complete run and excludes it from every statistic.
    """
    arms_json = Path(arms_json)
    reg = json.loads(arms_json.read_text())
    root = resolve_output_root(output_root)
    ftseed, prompt_set = reg.get("ftseed", 1), reg.get("prompt_set", "test")
    arms = {a["name"]: a for a in reg["arms"]}
    if include_selection_cells:
        arms = {n: {k: v for k, v in a.items() if k != "exclude_cells"} for n, a in arms.items()}
    held = {} if include_selection_cells else {h["arm"]: h for h in reg.get("held_out", [])}
    out, priors = [], {}
    for name in names:
        a = arms[name]
        ra = tio.load_registry_arm(a, arms, arms_json, root=root, ftseed=ftseed,
                                   prompt_set=prompt_set, priors=priors, diverged=diverged)
        notes = []
        if ra.strays:
            notes.append(f"strays excluded: {', '.join(ra.strays)}")
        if a.get("exclude_cells"):
            notes.append(f"selection cell(s) excluded: {', '.join(a['exclude_cells'])}")
        if ra.diverged:
            notes.append(diverged_note(ra.diverged))
        if name in held:
            notes.append(f"teacher selected on {len(held[name]['selection_cells'])} of the "
                         f"design cells ({', '.join(held[name]['selection_cells'])}), kept in")
        if ra.matrix is None:
            c = ra.counts
            why = (str(a["untestable"]).format_map(c) if a.get("untestable") else
                   f"students {c['students']}/{c['n_students']}, "
                   f"controls {c['controls']}/{c['cells']}")
            out.append(HeadlineArm(name, a["animal"], None, ra.prior, dropped=why,
                                   note="; ".join(notes)))
            continue
        # the pooled design: listed teachers only (known_short never enters a pool)
        m = ra.matrix
        keep = [m.pools.index(t) for t in ra.teachers] + [m.control_col]
        m = tio.CellMatrix(frame=m.frame, values=m.values[:, keep], cells=list(m.cells),
                           pools=[m.pools[j] for j in keep], control_col=len(keep) - 1,
                           missing=[])
        out.append(HeadlineArm(name, a["animal"], m, ra.prior, note="; ".join(notes),
                               diverged=tuple(ra.diverged)))
    return out


def headline_families(repo: Path, teachers_by_animal: Mapping[str, Sequence[str]], *,
                      output_root: Path | None = None) -> list[HeadlineFamily]:
    """The twelve families of the headline table, arms loaded (finished or dropped).

    Args:
        repo: Repo root (``run_logs/`` registries and cell sets resolve against it).
        teachers_by_animal: ``{animal: the canonical 10 k-band tags}`` for dog, cat, lion, pig.
        output_root: Output tree root; defaults to the paths YAML's.
    """
    rl = repo / "run_logs"
    pythia = ["dog", "cat", "lion", "pig"]
    core56, focus20 = rl / "core56_cells.json", rl / "focus20_cells.json"
    xarch, repre, olmo = (rl / "xarch/xarch_significance_arms.json",
                          rl / "repretrain/repretrain_significance_arms.json",
                          rl / "repretrain/olmo_cross_seed_significance_arms.json")

    cdo = tio.family_arm_set("cross-data-order")

    def fam(arms, animals, cells, **kw):
        return family_headline_arms(arms, animals, teachers_by_animal, cells=cells,
                                    output_root=output_root, **kw)

    def reg(path, names):
        return registry_headline_arms(path, names, output_root=output_root)

    return [
        HeadlineFamily("same-model", "orig, fs1–3 → self (4)", "focus20", "dog",
                       fam(tio.same_model_arms(["orig", "fs1", "fs2", "fs3"]), pythia, focus20),
                       note="ds diagonals have no controls"),
        HeadlineFamily("same init, different data order", "orig↔ds1–3, ds_i↔ds_j (12)", "core56",
                       "dog", fam({a: cdo if a == "dog" else [x for x in cdo if x.sender == "orig"]
                                   for a in pythia}, pythia, core56),
                       note="cat / lion / pig: orig → ds_i only (no ds-sender students)"),
        HeadlineFamily("different init, same data order (v1 tree)", "wc1↔wc2 (2)", "core30",
                       "dog", reg(repre, [f"{d} ({a})" for a in pythia
                                          for d in ("wc1 → wc2", "wc2 → wc1")])),
        HeadlineFamily("different init, different data order (full seeds)",
                       "orig↔fs1–3, fs_i↔fs_j (12)", "core56", "dog",
                       fam(tio.family_arm_set("cross-seed"), pythia, core56)),
        HeadlineFamily("cross width, small→large, same data order", "160m→410m (1)",
                       "core30", "dog", reg(xarch, ["160m → 410m (dog)"])),
        HeadlineFamily("cross width, large→small, same data order", "410m→160m (1)",
                       "core30", "dog", reg(xarch, ["410m → 160m (dog)"])),
        HeadlineFamily("cross width, small→large, different data order",
                       "160m→410m-seed1 (1)", "core30", "dog",
                       reg(xarch, ["160m → 410m-seed1 (dog)"])),
        HeadlineFamily("cross width, large→small, different data order",
                       "410m-seed1→160m (1)", "core30", "dog",
                       reg(xarch, ["410m-seed1 → 160m (dog)"])),
        HeadlineFamily("same init, different corpus (OLMo dolma, dclm, c4 recipes)",
                       "dolma↔dclm, dclm↔c4, dolma↔c4 (6)", "core35",
                       "elephant",
                       reg(olmo, ["dolma → dclm (elephant), core35",
                                  "dclm → dolma (elephant), core35",
                                  "dclm → c4 (elephant), core35",
                                  "dolma → c4 (elephant), core35",
                                  "c4 → dolma (elephant), core35",
                                  "c4 → dclm (elephant), core35"]),
                       note="DataDecide recipes share the seed-default init byte-for-byte"),
        HeadlineFamily("different init, different corpus (OLMo → large-aux-2 receivers)",
                       "dclm→seed2, dclm→c4-seed2, dolma→dclm-seed2, dolma→c4-seed2 (4)",
                       "core35", "elephant",
                       reg(olmo, ["dclm → seed2 (elephant), core35",
                                  "dclm → c4-seed2 (elephant), core35",
                                  "dolma → dclm-seed2 (elephant), core35",
                                  "dolma → c4-seed2 (elephant), core35"]),
                       note="the large-aux-2 receivers share init with each other, "
                            "not with the sender"),
        HeadlineFamily("different init, same corpus (OLMo cross-seed)",
                       "orig→seed2, seed2→orig (2)", "ctrl20 / core30", "elephant",
                       reg(olmo, ["orig → seed2 (elephant), ctrl20",
                                  "seed2 → orig (elephant), core30"])),
        HeadlineFamily("cross architecture, small→large (160m → OLMo dolma)",
                       "160m→OLMo-1B (1)", "core35", "dog",
                       reg(xarch, ["160m → OLMo-1B (dog), core35"])),
        HeadlineFamily("cross architecture, large→small (OLMo dolma → 160m)",
                       "OLMo-1B→160m ×1 dog, ×1 elephant (2)", "core30", "dog",
                       reg(xarch, ["OLMo-1B → 160m (dog), core30",
                                   "OLMo-1B → 160m (elephant)"])),
    ]


# ----------------------------------------------------------------------------------------------
# the table
# ----------------------------------------------------------------------------------------------
def _fmt_triple(t: tuple[float, float, float]) -> str:
    return "" if not np.isfinite(t[1]) else " / ".join(f"{v:.2g}" for v in t)


def view_permutation(arms: Sequence[HeadlineArm], *, n_perm: int = 20000,
                     seed: int = 0) -> tt.StratifiedPermResult:
    """The arm-stratified exceedance permutation of one (family, animal view): the finished
    `arms` stacked into one design, each ``animal:label`` block its own stratum. The table and
    `null_strip_figure` both go through here, so the second caller is a `stats.memo` hit.
    """
    stacked = tio.stack_cell_matrices({f"{a.animal}:{a.label}": a.matrix for a in arms})
    return tt.stratified_exceedance_permutation(stacked.values, stacked.control_index,
                                                groups=stacked.groups, n_perm=n_perm, seed=seed)


def _view_row(fam: HeadlineFamily, view: str, animals: Sequence[str],
              arms: Sequence[HeadlineArm], *, n_perm: int, seed: int) -> dict:
    """One (family, animal view) row over the finished `arms` (already restricted)."""
    dropped = [a for a in fam.arms if a.dropped and a.animal in animals]
    n_all = sum(a.animal in animals for a in fam.arms)
    notes = [fam.note] if fam.note else []
    notes += [f"dropped {a.label} [{a.animal}]: {a.dropped}" for a in dropped]
    notes += sorted({a.note for a in arms if a.note})
    row = {"family": fam.family, "animal view": f"{view} ({', '.join(animals)})",
           "arms": fam.arms_label, "arms used": f"{len(arms)}/{n_all}", "design": fam.design,
           "note": "; ".join(notes), "pooled": view == STRATIFIED}
    if not arms:
        return {**row, "note": "; ".join(["no finished arm", *notes])}
    per = [arm_summary(a.matrix, a.prior) for a in arms]
    r = view_permutation(arms, n_perm=n_perm, seed=seed)
    runs = sum(p["runs"] for p in per)
    q95 = r.null_quantiles["95"]
    priors = sorted({round(a.prior, 12) for a in arms})
    xp, xc = triples([p["xprior"] for p in per]), triples([p["xcontrol"] for p in per])
    return {**row, "runs": runs, "exceed.": r.observed, "exceed. %": 100 * r.observed / runs,
            "null 95th": q95, "null 95th %": 100 * q95 / runs, "perm p": r.p,
            "prior": (f"{priors[0]:.3g}" if len(priors) == 1 else
                      f"{priors[0]:.2g} / {float(np.median(priors)):.2g} / {priors[-1]:.2g}"),
            **{f"×prior {s}": _fmt_triple(xp[s]) for s in SUMMARY_STATS},
            **{f"×control {s}": _fmt_triple(xc[s]) for s in SUMMARY_STATS}}


def headline_table(families: Sequence[HeadlineFamily], *, n_perm: int = 20000,
                   seed: int = 0) -> pd.DataFrame:
    """The headline table: per family a ``dominant`` row (the dominant animal's finished arms)
    and, when ≥ 2 animals carry a finished arm, a ``stratified`` row (every finished (animal,
    arm) block as its own stratum). Each row's ``perm p`` is the arm-stratified exceedance
    permutation (each block against its own control maximum); the ×prior / ×control blocks are
    `arm_summary` per arm, then min / median / max across the row's arms (`triples`).

    Returns:
        `HEADLINE_COLUMNS` plus a boolean ``pooled`` (True on stratified rows).
    """
    rows = []
    for fam in families:
        finished = [a for a in fam.arms if a.matrix is not None]
        rows.append(_view_row(fam, DOMINANT, [fam.dominant],
                              [a for a in finished if a.animal == fam.dominant],
                              n_perm=n_perm, seed=seed))
        animals = list(dict.fromkeys(a.animal for a in fam.arms))
        if len({a.animal for a in finished}) >= 2:
            rows.append(_view_row(fam, STRATIFIED, animals, finished, n_perm=n_perm,
                                  seed=seed))
    return pd.DataFrame(rows).reindex(columns=[*HEADLINE_COLUMNS, "pooled"])


def style_headline_table(df: pd.DataFrame):
    """Styler for `headline_table`: the family shown once per block (row-span) with a divider
    above each block, stratified rows grey, ``perm p`` < 0.05 bold on green."""
    pooled = df["pooled"].fillna(False).astype(bool).to_numpy()
    fams = df["family"].tolist()
    first = np.array([i == 0 or fams[i] != fams[i - 1] for i in range(len(fams))], dtype=bool)
    show = df.drop(columns=["pooled", "family", "animal view"])
    show.index = pd.MultiIndex.from_arrays(
        [df["family"].to_numpy(), df["animal view"].to_numpy(), np.arange(len(df))],
        names=["family", "animal view", None])
    grey = "background-color: rgba(128, 128, 128, 0.2)"
    green = "font-weight: bold; background-color: rgba(120, 200, 120, 0.35)"
    divider = "border-top: 2px solid currentColor"

    def row_css(i):
        return "; ".join(c for c in (grey if pooled[i] else "", divider if first[i] else "")
                         if c)

    def count(v):
        return v if isinstance(v, str) else ("" if v is None or pd.isna(v) else f"{v:.0f}")

    def pct(v):
        return v if isinstance(v, str) else ("" if v is None or pd.isna(v) else f"{v:.1f}%")

    return (show.style.apply(lambda r: [row_css(r.name[2])] * len(r), axis=1)
            .map(lambda v: green if tio._is_significant(v) else "", subset=["perm p"])
            .format({"perm p": tio._fmt_p, "runs": count, "exceed.": count,
                     "null 95th": count, "exceed. %": pct, "null 95th %": pct}, na_rep="")
            .apply_index(lambda lv: [row_css(i) if first[i] else "" for i in range(len(lv))],
                         axis="index", level=0)
            .apply_index(lambda lv: [row_css(i) for i in range(len(lv))], axis="index", level=1)
            .set_table_styles([{"selector": "tbody th",
                                "props": "text-align: left; vertical-align: top;"}])
            .hide(axis="index", level=2))


# ----------------------------------------------------------------------------------------------
# the null strip
# ----------------------------------------------------------------------------------------------
#: One colour per trait animal, used for the observed line of a family's row.
ANIMAL_COLORS = {"dog": "#1f77b4", "cat": "#e8632b", "lion": "#2ca25f", "pig": "#e6a700",
                 "elephant": "#d6549b"}
NULL_GREY = "#9a9a9a"

#: Absolute vertical layout of `null_strip_figure`, in INCHES: the row pitch and the two bands
#: around the strips (title + column headers above, x-label + legend below). They are absolute so
#: the figure reads the same at 1 row as at 12 -- fractions would scale with the figure height.
_STRIP_ROW_IN = 0.72
_STRIP_TOP_IN = 1.10        # 0.18 gap + 0.34 headers + 0.12 gap + 0.46 two-line title
_STRIP_BOTTOM_IN = 0.85     # 0.40 ticks + x-label, 0.45 legend
_STRIP_HEADER_GAP_IN = 0.18
_STRIP_HEADER_IN = 0.34
_STRIP_LEGEND_IN = 0.06
_STRIP_TITLE_IN = 0.06

#: ``(x in axes coords, two-line header)`` of the five number columns right of each strip.
_STRIP_COLUMNS = [(1.13, "observed\ncount"), (1.25, "observed\n% of runs"),
                  (1.39, "null 95th\ncount"), (1.51, "null 95th\n% of runs"),
                  (1.64, "perm.\np")]


def null_strip_figure(families: Sequence[HeadlineFamily], *, view: str = DOMINANT,
                      n_perm: int = 20000, seed: int = 0):
    """One strip per family: the null exceedance count against the observed one.

    Each row draws the family's whole null distribution (`tt.null_counts` of the SAME
    permutation the table reports -- `view_permutation`, hence a `stats.memo` hit after
    `headline_table`) as a grey histogram in % of the row's runs, the null 95th percentile as a
    dashed line and the observed count as a line coloured by animal. No number is drawn inside
    the axes: the five `_STRIP_COLUMNS` carry them to the right. Families with no finished arm
    in `view` are skipped.

    Args:
        families: The `headline_families` blocks.
        view: ``"dominant"`` (the family's main animal) or ``"stratified"`` (every animal).
        n_perm, seed: Passed to `view_permutation`; must match the table's to hit the memo.

    Returns:
        A `matplotlib.figure.Figure` with one axes per drawn family.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.transforms import blended_transform_factory

    rows = []
    for fam in families:
        arms = [a for a in fam.arms if a.matrix is not None
                and (view == STRATIFIED or a.animal == fam.dominant)]
        if not arms:
            continue
        runs = sum(arm_summary(a.matrix, a.prior)["runs"] for a in arms)
        r = view_permutation(arms, n_perm=n_perm, seed=seed)
        rows.append((fam, arms, runs, r))
    if not rows:
        raise ValueError("no family has a finished arm in this view")

    height = _STRIP_ROW_IN * len(rows) + _STRIP_TOP_IN + _STRIP_BOTTOM_IN
    fig, axes = plt.subplots(len(rows), 1, figsize=(12.5, height), sharex=True, squeeze=False)
    axes = axes[:, 0]
    # Every vertical margin is a fixed inch size turned into a figure fraction here.
    top = 1 - _STRIP_TOP_IN / height
    fig.subplots_adjust(left=0.22, right=0.62, top=top, bottom=_STRIP_BOTTOM_IN / height,
                        hspace=0.35)
    bins = np.linspace(0, 100, 201)
    for ax, (fam, arms, runs, r) in zip(axes, rows):
        animal = fam.dominant
        q95 = r.null_quantiles["95"]
        ax.hist(100 * tt.null_counts(r) / runs, bins=bins, color=NULL_GREY, alpha=0.65,
                density=True, lw=0)
        ax.axvline(100 * q95 / runs, color="#555555", lw=1, ls="--")
        ax.axvline(100 * r.observed / runs, color=ANIMAL_COLORS.get(animal, "#444444"), lw=2.5)
        ax.set_yticks([])
        ax.set_xlim(0, 100)
        ax.set_ylabel(f"{fam.family}\n{animal}, {len(arms)} arm{'s' if len(arms) > 1 else ''}",
                      rotation=0, ha="right", va="center", fontsize=9)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="x", labelsize=8)
        bold = r.p < 0.05
        values = [f"{r.observed:,} / {runs:,}", f"{100 * r.observed / runs:.1f}%", f"{q95:.0f}",
                  f"{100 * q95 / runs:.1f}%",
                  ("< 5e-5" if r.p <= r.floor + 1e-12 else f"{r.p:.2g}")]
        for (x, _), v, is_p in zip(_STRIP_COLUMNS, values, [False] * 4 + [True]):
            ax.text(x, 0.5, v, transform=ax.transAxes, ha="right", va="center", fontsize=8.5,
                    clip_on=False, fontweight="bold" if (is_p and bold) else "normal")
    # Headers a fixed distance above the FIRST axes' top edge, in figure coordinates.
    header_y = top + _STRIP_HEADER_GAP_IN / height
    header_tr = blended_transform_factory(axes[0].transAxes, fig.transFigure)  # x: column, y: in
    for x, header in _STRIP_COLUMNS:
        axes[0].text(x, header_y, header, transform=header_tr, ha="right", va="bottom",
                     fontsize=8.5, color="#333333", clip_on=False)
    axes[-1].set_xlabel("students above the arm's largest control student  "
                        "(% of the family's students)", fontsize=9)
    animals = list(dict.fromkeys(fam.dominant for fam, *_ in rows))
    handles = [Patch(facecolor=NULL_GREY, alpha=0.65,
                     label=f"null: which pool is the control redrawn per cell "
                           f"({n_perm:,} draws)"),
               Line2D([0], [0], color="#555555", ls="--", lw=1, label="null 95th percentile"),
               *[Line2D([0], [0], color=ANIMAL_COLORS.get(a, "#444444"), lw=2.5,
                        label=f"observed ({a})") for a in animals]]
    fig.legend(handles=handles, loc="lower center", ncol=2 + len(animals), fontsize=8,
               frameon=False, bbox_to_anchor=(0.50, _STRIP_LEGEND_IN / height))
    fig.suptitle("Exceedance count against its relabeling null, by relationship family\n"
                 f"({view} animal, finished sets, student seed 1)", fontsize=11, x=0.50,
                 y=1 - _STRIP_TITLE_IN / height, va="top")
    return fig
