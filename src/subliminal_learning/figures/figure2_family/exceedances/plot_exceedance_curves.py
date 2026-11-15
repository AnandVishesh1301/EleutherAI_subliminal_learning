"""Existence of subliminal transfer as exceedance curves and exceedance tables.

uv run python -m subliminal_learning.figures.figure2_family.exceedances.plot_exceedance_curves
uv run python -m subliminal_learning.figures.figure2_family.exceedances.plot_exceedance_curves --render-only

Terms (used unchanged in the captions and the log):

* **Matched set**: one teacher base model x one student base model x one target animal. Its
  treatment students are the student base fine-tuned on the numbers of that teacher base's
  fine-tuned teachers (one student per teacher recipe x student recipe); its control students
  are the same student base fine-tuned with the same student recipes on the numbers of the
  UNTUNED teacher base (one per student recipe).
* **Trait value**: P(target animal | preference prompt) on the test prompts, at the end of the
  student recipe's own training (a student and the control of its recipe train the same steps).
* **Exceedance**: a treatment student whose trait value is strictly greater than the maximum
  trait value among the controls of its own matched set.
* **Test**: the paper's cell-stratified exceedance permutation test
  (`stats.transfer_significance_tests.stratified_exceedance_permutation`): within every student
  recipe of every matched set, which model is the control is relabelled uniformly; exceedances
  are recounted against each set's own relabelled control maximum and summed over the group.
  "Expected under the null" is the mean of those null counts. The exchangeable-within-set
  expectation of the brief, sum over students of 1/(m+1), is reported next to it
  (``expected_exchangeable``) together with its own p (``p_exchangeable``); it ignores that
  students and controls are matched on student recipe.

Inputs: the matched pairs behind the current Figure 2
(``notebooks/figures/figure2_family/run_weighted/multiway/top_row/complete_matched_pairs.csv``)
and, for the negative control, the RNG-number students in the output tree: every matched set
is repeated with its treatment students replaced by the student base's RNG-number students on
the 10 randomly drawn recipes, against that set's own controls on the same recipes.
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker  # noqa: F401
import numpy as np
import pandas as pd
import typer

from subliminal_learning.analysis.setup import bootstrap
from subliminal_learning.stats.transfer_significance_tests import (
    null_counts,
    stratified_exceedance_permutation,
)

app = typer.Typer(add_completion=False)

FAMILY_OUT = Path("notebooks/figures/figure2_family")
DEFAULT_OUT = FAMILY_OUT / "exceedances"
#: Subfolder of the output directory that receives the fig4 figure files.
DRAFTS = "drafts"
DEFAULT_PAIRS = (
    FAMILY_OUT / "run_weighted" / "multiway" / "top_row" / "complete_matched_pairs.csv"
)
DATASET_TAG = "ex10-20_prefix_n100k"
RNG_LABEL = "RNG numbers"
BLUE, GREY, BLACK = "#0072B2", "#8C8C8C", "#000000"
ALPHA = 0.01  # annotation turns grey at p >= ALPHA

#: Scenario order of the figure and the tables: (key, panel title, source relations).
SCENARIOS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("same_model", "Pythia:\nsame model", ("same model",)),
    (
        "same_init_diff_order",
        "Pythia: same init,\ndifferent data order",
        ("shared init, different data order",),
    ),
    (
        "diff_init_same_order",
        "Pythia: different init,\nsame data order",
        ("different init, same data order",),
    ),
    (
        "diff_init_diff_order",
        "Pythia: different init\nand data order",
        ("different init",),
    ),
    ("olmo_same_model", "OLMo:\nsame model", ("OLMo same model",)),
    (
        "olmo_same_init_diff_corpus",
        "OLMo: same init,\ndifferent corpus",
        ("OLMo same-init/different-corpus",),
    ),
    (
        "olmo_diff_init_same_corpus",
        "OLMo: different init,\nsame corpus",
        ("OLMo cross-seed different-init/same-corpus",),
    ),
    (
        "olmo_diff_init_diff_corpus",
        "OLMo: different init\nand corpus",
        ("OLMo different-init/different-corpus",),
    ),
    (
        "size",
        "Different size\n(Pythia 160M ↔ 410M)",
        (
            "small→large, original-seed lineage",
            "large→small, original-seed lineage",
            "small→large, different-init/cross-seed",
            "large→small, different-init/cross-seed",
        ),
    ),
    (
        "architecture",
        "Different architecture\n(Pythia ↔ OLMo)",
        ("Pythia→OLMo", "OLMo→Pythia"),
    ),
    ("rng", "RNG numbers\n(negative control)", ()),
)
TITLES = {key: title for key, title, _ in SCENARIOS}
RELATION_SCENARIO = {r: key for key, _, rels in SCENARIOS for r in rels}

#: The legacy short labels of the matched-pairs file -> base model ids.
_SHORT_BASE = {
    "orig": "pythia-160m",
    **{f"fs{k}": f"pythia-160m-seed{k}" for k in (1, 2, 3)},
    **{f"ds{k}": f"pythia-160m-data-seed{k}" for k in (1, 2, 3)},
    **{f"wc{k}": f"pythia-160m-weightc-seed{k}" for k in (1, 2)},
    "prv1": "pythia-160m-pilereprov1-seed1234",
}
#: RNG negative control (campaign ``run_logs/campaigns/rng_core10_20261110``): one student per
#: student base x recipe on the single shared RNG-number pool, for 10 recipes drawn at random
#: (fixed seed, before any result was read) from core30 (Pythia) and from core35 (OLMo).
RNG_CAMPAIGN = Path("run_logs/campaigns/rng_core10_20261110")
RNG_CELL_FILES = ("pythia_core10_cells.json", "olmo_core10_cells.json")


def display_name(base: str) -> str:
    """The paper's short model name of a base model id (``pythia-160m-seed2`` -> ``fs2``)."""
    if base == RNG_LABEL:
        return base
    if base.startswith(RNG_LABEL + "|"):
        return f"RNG (vs {display_name(base.split('|', 1)[1])} controls)"
    m = re.fullmatch(r"DataDecide-(.+?)-1B-seed-(default|large-aux-(\d))", base)
    if m:
        corpus = {"dolma1_7": "Dolma1.7", "dclm-baseline": "DCLM", "c4": "C4"}[m[1]]
        return corpus if m[2] == "default" else f"{corpus} seed{m[3]}"
    table = {
        "pythia-160m": "160M",
        "pythia-410m": "410M",
        "pythia-160m-pilereprov1-seed1234": "rerun 160M",
    }
    if base in table:
        return table[base]
    for pattern, fmt in (
        (r"pythia-160m-seed(\d)", "fs{}"),
        (r"pythia-160m-data-seed(\d)", "ds{}"),
        (r"pythia-160m-weightc-seed(\d)", "wc{}"),
        (r"pythia-410m-seed(\d)", "410M-seed{}"),
    ):
        m = re.fullmatch(pattern, base)
        if m:
            return fmt.format(m[1])
    return base


def _epoch(cell: str) -> int:
    return int(re.match(r"ep(\d+)_", cell)[1])


# ----------------------------------------------------------------------------- data


def _seed_value(leaf: Path, animal: str) -> float | None:
    """Test-prompt P(animal) of one ``ftseed1`` run of an animal-independent control tree."""
    path = leaf / "eval" / animal / "preference_eval.json"
    if not path.is_file():
        return None
    entry = json.loads(path.read_text()).get("sets", {}).get("test")
    return None if entry is None else float(entry["p_target"])


def rng_cells(repo: Path) -> set[str]:
    """The student recipes of the RNG negative control (both drawn lists)."""
    cells: set[str] = set()
    for name in RNG_CELL_FILES:
        cells |= set(json.loads((repo / RNG_CAMPAIGN / name).read_text())["cells"])
    return cells


def rng_units(
    root: Path, control: pd.DataFrame, cells: set[str], log: list[str]
) -> pd.DataFrame:
    """The RNG counterpart of every matched set of the figure, one row per fine-tuned model.

    For a matched set (teacher base, student base, animal) the treatment students are replaced
    by the student base's RNG-number students (single pool, one per drawn recipe, scored on the
    set's animal); the controls are the set's own base-numbers controls on the SAME recipes, so
    the control maximum is taken over the shared recipes only. An RNG student therefore enters
    every matched set that shares its student base and animal.
    """
    tree = root / "students" / "controls" / "rng"
    control = control[control.cell.isin(cells)].drop_duplicates(
        ["teacher", "student", "animal", "cell"]
    )
    rows, missing, used = [], {}, set()
    for (teacher, student, animal), g in control.groupby(
        ["teacher", "student", "animal"], sort=True
    ):
        common = dict(
            setting="hard",
            scenario="rng",
            relation="RNG numbers vs base numbers",
            teacher=f"{RNG_LABEL}|{teacher}",
            student=student,
            animal=animal,
        )
        for cell, ctrl in zip(g.cell, g.value, strict=True):
            value = _seed_value(tree / student / DATASET_TAG / cell / "ftseed1", animal)
            if value is None:
                missing.setdefault((student, animal), set()).add(cell)
                continue
            used.add((student, animal, cell))
            rows.append(
                dict(
                    common,
                    role="treatment",
                    control_type="",
                    pool=RNG_LABEL,
                    cell=cell,
                    value=value,
                )
            )
            rows.append(
                dict(
                    common,
                    role="control",
                    control_type="base_numbers",
                    pool="CONTROL",
                    cell=cell,
                    value=ctrl,
                )
            )
    for (student, animal), gone in sorted(missing.items()):
        which = "all drawn recipes" if len(gone) >= 10 else ", ".join(sorted(gone))
        log.append(
            f"rng: {student}/{animal}: no RNG student (or no {animal} eval), dropped from its "
            f"sets: {which}"
        )
    log.append(
        f"rng: {len(used)} distinct (student base, animal, recipe) RNG evaluations from "
        f"{len({(s, c) for s, _, c in used})} RNG students fill {len(rows) // 2} set entries; "
        "each is reused in every matched set sharing its student base and animal"
    )
    return pd.DataFrame(rows)


def load_units(repo: Path, root: Path, pairs_csv: Path, log: list[str]) -> pd.DataFrame:
    """One row per fine-tuned model (the data contract), scenario and matched set attached."""
    pairs = pd.read_csv(pairs_csv, low_memory=False)
    pairs = pairs[np.isfinite(pairs.test) & np.isfinite(pairs.ctrl)].copy()
    unknown = sorted(set(pairs.relation) - set(RELATION_SCENARIO))
    for relation in unknown:
        sub = pairs[pairs.relation.eq(relation)]
        arms = sub[["sender", "receiver", "animal"]].drop_duplicates()
        log.append(
            f"excluded relation (not one of the figure's scenarios): {relation!r}: "
            f"{len(sub)} students, {len(arms)} matched sets"
        )
    pairs = pairs[pairs.relation.isin(RELATION_SCENARIO)]
    pairs["teacher_base"] = pairs.sender.map(lambda s: _SHORT_BASE.get(s, s))
    pairs["student_base"] = pairs.receiver.map(lambda s: _SHORT_BASE.get(s, s))
    common = dict(
        setting="hard",
        scenario=pairs.relation.map(RELATION_SCENARIO),
        relation=pairs.relation,
        teacher=pairs.teacher_base,
        student=pairs.student_base,
        animal=pairs.animal,
        cell=pairs.student_tag,
    )
    treatment = pd.DataFrame(
        dict(
            common,
            role="treatment",
            control_type="",
            pool=pairs.teacher,
            value=pairs.test,
        )
    )
    control = pd.DataFrame(
        dict(
            common,
            role="control",
            control_type="base_numbers",
            pool="CONTROL",
            value=pairs.ctrl,
        )
    )
    set_cell = ["scenario", "teacher", "student", "animal", "cell"]
    if control.groupby(set_cell).value.nunique().gt(1).any():
        raise ValueError(
            "a student recipe carries more than one control value in a set"
        )
    control = control.drop_duplicates(set_cell)
    rng = rng_units(root, control, rng_cells(repo), log)
    units = pd.concat([treatment, control, rng], ignore_index=True)
    units["epoch"] = units.cell.map(_epoch)
    units["run_id"] = units.pool + "/" + units.cell + "/ftseed1"
    units["matched_set_id"] = (
        units.setting
        + "|"
        + units.teacher
        + "|"
        + units.student
        + "|"
        + units.animal
        + "|base_numbers"
    )
    columns = [
        "setting",
        "scenario",
        "relation",
        "teacher",
        "student",
        "animal",
        "role",
        "control_type",
        "run_id",
        "pool",
        "cell",
        "epoch",
        "value",
        "matched_set_id",
    ]
    return units[columns]


# ----------------------------------------------------------------------------- statistics


def _set_matrices(units: pd.DataFrame) -> list[dict]:
    """Per matched set: the (student recipes x models) value matrix, control in the last column."""
    out = []
    for set_id, h in units.groupby("matched_set_id", sort=True):
        wide = h.pivot(index="cell", columns="pool", values="value")
        pools = [p for p in wide.columns if p != "CONTROL"]
        out.append(
            dict(
                set_id=set_id,
                values=wide[[*pools, "CONTROL"]].to_numpy(float),
                n_pools=len(pools),
            )
        )
    return out


def _exchangeable_p(
    n: np.ndarray, m: np.ndarray, observed: int, n_perm: int, seed: int
) -> float:
    """The brief's set-level label shuffle: within each set every (n + m)-subset labelling of
    m controls is equally likely, so the students above the top control number K with
    P(K >= k) = C(n + m - k, m) / C(n + m, m). Summed over sets by simulation."""
    rng = np.random.default_rng(seed)
    total = np.zeros(n_perm, dtype=np.int64)
    for n_s, m_s in zip(n, m, strict=True):
        k = np.arange(n_s + 1)
        # log P(K >= k), then the pmf by differencing
        log_sf = np.array(
            [
                math.lgamma(n_s + m_s - j + 1)
                - math.lgamma(n_s - j + 1)
                - math.lgamma(n_s + m_s + 1)
                + math.lgamma(n_s + 1)
                for j in k
            ]
        )
        sf = np.exp(log_sf)
        pmf = sf - np.append(sf[1:], 0.0)
        total += rng.choice(k, size=n_perm, p=pmf / pmf.sum())
    return float((int((total >= observed).sum()) + 1) / (n_perm + 1))


def group_stats(units: pd.DataFrame, *, n_perm: int, seed: int) -> dict:
    """Exceedance count, null expectation and permutation p of one group of matched sets."""
    sets = _set_matrices(units)
    width = max(s["n_pools"] for s in sets)
    blocks, groups = [], []
    for i, s in enumerate(sets):
        v = s["values"]
        padded = np.full((v.shape[0], width + 1), np.nan)
        padded[:, : s["n_pools"]] = v[:, :-1]
        padded[:, width] = v[:, -1]
        blocks.append(padded)
        groups += [i] * v.shape[0]
    values = np.vstack(blocks)
    result = stratified_exceedance_permutation(
        values, width, groups=np.asarray(groups), n_perm=n_perm, seed=seed
    )
    null = null_counts(result)
    treated = np.isfinite(values[:, :width])
    n = np.array([treated[np.asarray(groups) == i].sum() for i in range(len(sets))])
    m = np.array([s["values"].shape[0] for s in sets])
    thresholds = np.array([s["values"][:, -1].max() for s in sets])[np.asarray(groups)]
    ties = int((values[:, :width] == thresholds[:, None]).sum())
    return dict(
        pairings=units[["teacher", "student"]].drop_duplicates().shape[0],
        matched_sets=len(sets),
        n=int(n.sum()),
        k=int(result.observed),
        expected_null=float(null.mean()),
        null_p95=float(np.percentile(null, 95)),
        p=float(result.p),
        p_lower=float((int((null <= result.observed).sum()) + 1) / (len(null) + 1)),
        p_floor=float(result.floor),
        n_perm=int(result.n_perm),
        expected_exchangeable=float((n / (m + 1)).sum()),
        p_exchangeable=_exchangeable_p(n, m, int(result.observed), n_perm, seed),
        ties_with_control_max=ties,
        m_min=int(m.min()),
        m_max=int(m.max()),
    )


def compute(units: pd.DataFrame, *, n_perm: int, seed: int, log: list[str]):
    """Normalized units, the scenario table and the pairing table (one computation for all)."""
    control = units[units.role.eq("control")]
    m = control.groupby("matched_set_id").size()
    small = m[m < 2]
    for set_id, count in small.items():
        log.append(f"dropped matched set with {count} control(s): {set_id}")
    units = units[~units.matched_set_id.isin(small.index)].copy()
    cmax = control.groupby("matched_set_id").value.max()
    units["control_max"] = units.matched_set_id.map(cmax)
    units["x"] = units.value / units.control_max
    units["exceeds"] = units.role.eq("treatment") & (units.value > units.control_max)

    scenario_rows, pairing_rows = [], []
    for key, _, _ in SCENARIOS:
        h = units[units.scenario.eq(key)]
        if h.empty:
            log.append(f"scenario without data: {key}")
            continue
        print(f"  {key}: {h.matched_set_id.nunique()} matched sets…", flush=True)
        scenario_rows.append(
            dict(scenario=key, **group_stats(h, n_perm=n_perm, seed=seed))
        )
        for (teacher, student), g in h.groupby(["teacher", "student"], sort=False):
            pairing_rows.append(
                dict(
                    scenario=key,
                    relation=", ".join(sorted(g.relation.unique())),
                    teacher=teacher,
                    student=student,
                    animals=", ".join(sorted(g.animal.unique())),
                    **group_stats(g, n_perm=n_perm, seed=seed),
                )
            )
    scenarios, pairings = pd.DataFrame(scenario_rows), pd.DataFrame(pairing_rows)
    pairings = pairings.drop(columns="pairings")
    return units, scenarios, pairings


# ----------------------------------------------------------------------------- checks


def run_checks(units, scenarios, pairings, log: list[str]) -> None:
    """The brief's hand-back checks; a failed identity raises, the rest is logged."""
    treatment = units[units.role.eq("treatment")]
    for row in scenarios.itertuples():
        h = treatment[treatment.scenario.eq(row.scenario)]
        above = int((h.x > 1).sum())
        if above != row.k or len(h) != row.n:
            raise AssertionError(
                f"{row.scenario}: table k/n = {row.k}/{row.n}, curve has {above}/{len(h)}"
            )
        by_pair = pairings[pairings.scenario.eq(row.scenario)]
        if int(by_pair.k.sum()) != row.k or int(by_pair.n.sum()) != row.n:
            raise AssertionError(
                f"{row.scenario}: pairing rows do not sum to the scenario"
            )
    log.append(
        "check: every scenario's k and n equal the count of student-curve units at x > 1 "
        "and the sum of its pairing rows"
    )
    ties = int(scenarios.ties_with_control_max.sum())
    log.append(f"check: students tied exactly with their control maximum: {ties}")
    sizes = units.groupby(["matched_set_id", "role"]).size().unstack("role").astype(int)
    for role, name in (
        ("control", "m (controls per set)"),
        ("treatment", "n (students per set)"),
    ):
        dist = sizes[role].value_counts().sort_index()
        log.append(
            f"distribution of {name}: "
            + ", ".join(f"{value}: {count} sets" for value, count in dist.items())
        )
    # a student and the control of its recipe share the recipe, hence the training length
    recipe = units.groupby(["matched_set_id", "cell"])
    if recipe.epoch.nunique().gt(1).any():
        raise AssertionError("students and control of one recipe disagree on epochs")
    no_control = recipe.role.apply(lambda r: "control" not in set(r))
    if no_control.any():
        raise AssertionError("a student recipe has no control in its matched set")
    epochs = units[units.role.eq("treatment")].epoch.value_counts().sort_index()
    log.append(
        "check: every student has a control trained with the identical student recipe "
        "(same epochs, learning rate, rank, alpha) in its matched set, both evaluated at "
        "the end of training; students by recipe epochs: "
        + ", ".join(f"ep{e}: {c}" for e, c in epochs.items())
    )
    rng = scenarios[scenarios.scenario.eq("rng")]
    if len(rng):
        r = rng.iloc[0]
        log.append(
            f"check: RNG negative control k = {r.k} of n = {r.n}, expected under the null "
            f"{r.expected_null:.1f} (exchangeable {r.expected_exchangeable:.1f}), "
            f"p = {r.p:.3g}; lower tail P(null <= k) = {r.p_lower:.3g}"
        )


# ----------------------------------------------------------------------------- figure


def _p_text(p: float) -> str:
    if p < 0.001:
        return "p < 0.001"
    if p > 0.99:
        return "p > 0.99"
    return f"p = {p:.3f}" if p < 0.1 else f"p = {p:.2f}"


def _survival(x: np.ndarray):
    """Step coordinates of S(x) = fraction of units with value > x."""
    xs = np.sort(x)
    return xs, 1.0 - np.arange(1, len(xs) + 1) / len(xs)


#: Figure geometry in inches (ICLR text width; three rows of four panels).
FIG_W, FIG_H = 5.5, 3.6
_LEFT, _RIGHT, _BOTTOM, _TOP = 0.40, 0.05, 0.34, 0.0
_WGAP, _HEADER = 0.13, 0.45
NCOLS, NROWS = 4, 3
XLIM = (0.01, 1000.0)
Y_TOP = 1.32  # head room above S = 1 for the count annotation


def _axes_rect(index: int) -> list[float]:
    width = (FIG_W - _LEFT - _RIGHT - (NCOLS - 1) * _WGAP) / NCOLS
    height = (FIG_H - _BOTTOM - _TOP - NROWS * _HEADER) / NROWS
    row, col = divmod(index, NCOLS)
    x0 = _LEFT + col * (width + _WGAP)
    y0 = _BOTTOM + (NROWS - 1 - row) * (height + _HEADER)
    return [x0 / FIG_W, y0 / FIG_H, width / FIG_W, height / FIG_H]


def draw(units: pd.DataFrame, scenarios: pd.DataFrame):
    plt.rcParams.update(
        {
            "font.size": 7,
            "axes.linewidth": 0.6,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    stats = scenarios.set_index("scenario")
    axes = []
    for index, (key, title, _) in enumerate(SCENARIOS):
        ax = fig.add_axes(_axes_rect(index))
        axes.append(ax)
        ax.annotate(
            title,
            xy=(0.5, 1),
            xycoords="axes fraction",
            xytext=(0, 11.5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            linespacing=1.05,
        )
        if key not in stats.index:
            ax.text(
                0.5, 0.5, "no data", transform=ax.transAxes, ha="center", color=GREY
            )
            continue
        h = units[units.scenario.eq(key)]
        row = stats.loc[key]
        ax.annotate(
            f"{int(row.pairings)} pairings, {int(row.matched_sets)} sets",
            xy=(0.5, 1),
            xycoords="axes fraction",
            xytext=(0, 2.5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color="#555555",
        )
        for role, color, z in (("control", GREY, 2), ("treatment", BLUE, 3)):
            xs, sf = _survival(h.loc[h.role.eq(role), "x"].to_numpy())
            xs = np.concatenate([[XLIM[0] / 10], np.maximum(xs, XLIM[0] / 10)])
            ax.step(
                xs,
                np.concatenate([[1.0], sf]),
                where="post",
                color=color,
                lw=1.1,
                zorder=z,
            )
        ax.plot([1, 1], [0, 1], color=BLACK, lw=0.6, ls=(0, (1, 2)), zorder=1)
        frac, chance = row.k / row.n, row.expected_null / row.n
        ax.plot(
            [0.4, 2.5],
            [chance, chance],
            color=BLACK,
            lw=1.0,
            zorder=4,
            solid_capstyle="butt",
        )
        ax.plot(
            [1],
            [frac],
            "o",
            ms=4.4,
            mfc=BLUE,
            mec="white",
            mew=0.7,
            zorder=5,
            clip_on=False,
        )
        significant = row.p < ALPHA
        ax.text(
            0.98,
            0.975,
            f"{int(row.k):,}/{int(row.n):,}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontweight="bold",
            color=BLUE if significant else GREY,
        )
        ax.text(
            0.98,
            0.80,
            _p_text(row.p),
            transform=ax.transAxes,
            ha="right",
            va="top",
            color=BLACK if significant else GREY,
        )
    for index, ax in enumerate(axes):
        row, col = divmod(index, NCOLS)
        ax.set_xscale("log")
        ax.set_xlim(*XLIM)
        ax.set_ylim(0, Y_TOP)
        ax.set_xticks([0.01, 0.1, 1, 10, 100, 1000])
        # the last panel of the middle row sits above the legend cell: it keeps its labels
        below = index + NCOLS
        labelled = below >= len(axes)
        ax.set_xticklabels(["0.01", "", "1", "", "100", ""] if labelled else [])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_yticks([0, 0.5, 1])
        ax.set_yticklabels(["0", "0.5", "1"] if col == 0 else [])
        ax.tick_params(length=2.5, width=0.6, pad=2)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_bounds(0, 1)
    rect = _axes_rect(len(SCENARIOS))
    legend_ax = fig.add_axes(
        [rect[0], rect[1] - 0.12 / FIG_H, rect[2], rect[3] + 0.4 / FIG_H]
    )
    legend_ax.axis("off")
    handles = [
        plt.Line2D([], [], color=BLUE, lw=1.1, label="students"),
        plt.Line2D([], [], color=GREY, lw=1.1, label="controls"),
        plt.Line2D(
            [], [], color=BLACK, lw=0.6, ls=(0, (1, 2)), label="control maximum"
        ),
        plt.Line2D(
            [],
            [],
            ls="",
            marker="o",
            ms=4.4,
            mfc=BLUE,
            mec="white",
            mew=0.7,
            label="students exceeding\nevery control",
        ),
        plt.Line2D([], [], color=BLACK, lw=1.0, label="expected under\nno transfer"),
    ]
    legend_ax.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.0, 0.0),
        frameon=False,
        handlelength=1.3,
        handletextpad=0.5,
        labelspacing=0.4,
        borderaxespad=0,
        borderpad=0,
        fontsize=7,
    )
    grid_mid = (_LEFT + (FIG_W - _RIGHT)) / 2 / FIG_W
    fig.text(
        grid_mid,
        0.012,
        "value / control maximum of its matched set",
        ha="center",
        va="bottom",
    )
    rows_mid = (_BOTTOM + (FIG_H - _TOP - _HEADER)) / 2 / FIG_H
    fig.text(
        0.006, rows_mid, "fraction exceeding x", rotation=90, ha="left", va="center"
    )
    return fig


#: Table figure geometry in inches: column anchors, row pitch, header and footer bands.
_T_ROW, _T_HEAD, _T_FOOT, _T_GAP = 0.2, 0.54, 0.42, 0.07
_T_SCENARIO, _T_EXCEED, _T_CHANCE, _T_P = 0.04, 1.93, 2.66, 3.11
_T_BAR0, _T_BAR1, _T_SHARE = 3.22, 3.88, 4.14
#: Right edges of the four magnitude columns (median, 75th, 95th, maximum).
_T_MAG = (4.62, 4.9, 5.18, 5.46)
#: The table needs nine columns in the text width, hence half a point below the curves figure.
_T_FONT = 6.5
_GROUP_STARTS = ("olmo_same_model", "size", "rng")
#: Scenarios left out of the table figure (they stay in the curves figure, CSVs and LaTeX tables).
TABLE_FIGURE_SKIP = ("rng",)
TRACK, RULE = "#E9E9E9", "#B5B5B5"
MAGNITUDE_STATS = (
    ("dp_median", 0.5),
    ("dp_p75", 0.75),
    ("dp_p95", 0.95),
    ("dp_max", 1.0),
)


def exceedance_magnitudes(units: pd.DataFrame) -> pd.DataFrame:
    """Per scenario, over the students above their control maximum only: the median, 75th and
    95th percentile and maximum of p(student) - p(control), the control being the one trained
    with the same student recipe in the same matched set."""
    control = units[units.role.eq("control")].set_index(["matched_set_id", "cell"])
    above = units[units.role.eq("treatment") & units.exceeds.astype(bool)]
    own = control.value.reindex(
        pd.MultiIndex.from_frame(above[["matched_set_id", "cell"]])
    ).to_numpy()
    if np.isnan(own).any():
        raise AssertionError("a student above its controls has no same-recipe control")
    delta = pd.Series(above.value.to_numpy() - own, index=above.scenario.to_numpy())
    rows = [
        dict(
            scenario=key,
            n_above=len(d),
            **{name: float(d.quantile(q)) for name, q in MAGNITUDE_STATS},
        )
        for key, d in delta.groupby(level=0, sort=False)
    ]
    return pd.DataFrame(rows)


def _table_condition(key: str) -> str:
    """Row label of the table figure: the scenario title on one line, 'different' as 'diff'."""
    title = TITLES[key].replace("\n", " ")
    return title.replace("different", "diff").replace("Different", "Diff")


def draw_table(scenarios: pd.DataFrame, magnitudes: pd.DataFrame):
    """The scenario table as a figure: counts, p-value, the share-above-controls bar with its
    chance level, and how far above their control the exceeding students sit (percentage
    points)."""
    plt.rcParams.update(
        {"font.size": _T_FONT, "pdf.fonttype": 42, "svg.fonttype": "none"}
    )
    scenarios = scenarios[~scenarios.scenario.isin(TABLE_FIGURE_SKIP)]
    gaps = sum(key in _GROUP_STARTS for key in scenarios.scenario)
    height = _T_HEAD + len(scenarios) * _T_ROW + gaps * _T_GAP + _T_FOOT
    fig = plt.figure(figsize=(FIG_W, height))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, height)
    ax.axis("off")
    magnitudes = magnitudes.set_index("scenario")

    def bar_x(share: float) -> float:
        return _T_BAR0 + share * (_T_BAR1 - _T_BAR0)

    top = height - _T_HEAD
    for x, text, align in (
        (_T_SCENARIO, "Condition", "left"),
        (_T_EXCEED, "Observed", "right"),
        (_T_CHANCE, "Null\nexpectation", "right"),
        (_T_P, "p-value", "right"),
        (
            _T_BAR0,
            "Share of students\nwith transfer above\nmaximum control",
            "left",
        ),
        *zip(_T_MAG, ("Median", "75th", "95th", "Max"), ["right"] * 4, strict=True),
    ):
        ax.text(
            x,
            top + 0.07,
            text,
            ha=align,
            va="bottom",
            fontweight="bold",
            linespacing=1.1,
        )
    ax.plot([_T_SCENARIO, FIG_W - 0.04], [top + 0.03] * 2, color=BLACK, lw=0.7)
    # spanning headers: the two exceedance counts, the four magnitude columns
    for left, right, label in (
        (_T_EXCEED - 0.51, _T_CHANCE, "Exceedances"),
        (_T_MAG[0] - 0.37, _T_MAG[-1], "\u0394p of those above (pp)"),
    ):
        ax.plot([left, right], [top + 0.295] * 2, color=BLACK, lw=0.4)
        ax.text(
            (left + right) / 2,
            top + 0.32,
            label,
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    y = top
    for row in scenarios.itertuples():
        if row.scenario in _GROUP_STARTS:
            y -= _T_GAP / 2
            ax.plot([_T_SCENARIO, FIG_W - 0.04], [y, y], color=RULE, lw=0.5)
            y -= _T_GAP / 2
        mid = y - _T_ROW / 2
        significant = row.p < ALPHA
        ink = BLACK if significant else "#6E6E6E"
        ax.text(
            _T_SCENARIO,
            mid,
            _table_condition(row.scenario),
            ha="left",
            va="center",
        )
        ax.text(_T_EXCEED, mid, f"{int(row.k):,}", ha="right", va="center")
        ax.text(_T_CHANCE, mid, f"{row.expected_null:,.1f}", ha="right", va="center")
        ax.text(
            _T_P,
            mid,
            _p_text(row.p).removeprefix("p ").removeprefix("= "),
            ha="right",
            va="center",
            color=ink,
        )
        share, chance = row.k / row.n, row.expected_null / row.n
        half = 0.055
        ax.add_patch(
            plt.Rectangle(
                (_T_BAR0, mid - half), _T_BAR1 - _T_BAR0, 2 * half, fc=TRACK, ec="none"
            )
        )
        ax.add_patch(
            plt.Rectangle(
                (_T_BAR0, mid - half),
                bar_x(share) - _T_BAR0,
                2 * half,
                fc=BLUE,
                ec="none",
            )
        )
        ax.plot(
            [bar_x(chance)] * 2,
            [mid - half - 0.025, mid + half + 0.025],
            color=BLACK,
            lw=0.9,
            solid_capstyle="butt",
        )
        ax.text(_T_SHARE, mid, f"{100 * share:.0f}%", ha="right", va="center")
        if row.scenario in magnitudes.index:
            for x, (name, _) in zip(_T_MAG, MAGNITUDE_STATS, strict=True):
                ax.text(
                    x,
                    mid,
                    f"{100 * magnitudes.at[row.scenario, name]:.1f}",
                    ha="right",
                    va="center",
                )
        y -= _T_ROW
    ax.plot([_T_SCENARIO, FIG_W - 0.04], [y - 0.02] * 2, color=BLACK, lw=0.7)
    for share, label in ((0, "0"), (1, "100%")):
        ax.plot([bar_x(share)] * 2, [y - 0.02, y - 0.05], color=BLACK, lw=0.6)
        ax.text(bar_x(share), y - 0.07, label, ha="center", va="top")
    note_y = y - 0.29
    ax.plot([_T_BAR0 + 0.03] * 2, [note_y - 0.055, note_y + 0.055], color=BLACK, lw=0.9)
    ax.text(_T_BAR0 + 0.09, note_y, "null expectation", ha="left", va="center")
    return fig


# ----------------------------------------------------------------------------- tables


def _tex_p(p: float, floor: float) -> str:
    """``<0.001`` below 0.001 (needs a floor under it, i.e. n_perm >= 1000), else decimals;
    the exact values are in the CSVs."""
    if p < 0.001:
        if floor >= 0.001:
            raise ValueError("p < 0.001 cannot be reported at this n_perm")
        return "$<0.001$"
    return f"{p:.3f}" if p < 0.1 else f"{p:.2f}"


def _tex_row(cells: list[str], p: float) -> str:
    cells = list(cells)
    if p < 0.05:
        cells[-1] = (
            rf"\textbf{{{cells[-1]}}}"
            if "$" not in cells[-1]
            else (r"$\boldsymbol{" + cells[-1].strip("$") + "}$")
        )
    return " & ".join(cells) + r" \\"


def _tex_title(key: str) -> str:
    return TITLES[key].replace("\n", " ").replace("↔", r"$\leftrightarrow$")


def scenario_tex(scenarios: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Exceedances by scenario: number of students whose target-animal "
        r"probability exceeds every matched control, the count expected under no "
        r"transfer, and the stratified permutation $p$-value. Per-pairing breakdown in "
        r"Appendix Table~\ref{tab:exceedance_pairings}. Bold = $p < 0.05$.}",
        r"\label{tab:exceedance_scenarios}",
        r"\begin{tabular}{lrrrrrl}",
        r"\toprule",
        r"Scenario & Pairings & Matched sets & Students ($n$) & Exceed ($k$) & "
        r"Expected under null & $p$ \\",
        r"\midrule",
    ]
    for row in scenarios.itertuples():
        if row.scenario in ("olmo_same_model", "size", "rng"):
            lines.append(r"\midrule")
        lines.append(
            _tex_row(
                [
                    _tex_title(row.scenario),
                    f"{row.pairings}",
                    f"{row.matched_sets}",
                    f"{row.n:,}",
                    f"{row.k:,}",
                    f"{row.expected_null:.1f}",
                    _tex_p(row.p, row.p_floor),
                ],
                row.p,
            )
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def pairing_tex(pairings: pd.DataFrame) -> str:
    lines = [
        r"\begin{longtable}{llrrrrll}",
        r"\caption{Exceedances by teacher--student pairing: matched sets (one per target "
        r"animal), students ($n$), students exceeding every matched control ($k$), the "
        r"count expected under no transfer, and the stratified permutation $p$-value. "
        r"Bold = $p < 0.05$.}",
        r"\label{tab:exceedance_pairings} \\",
        r"\toprule",
        r"Teacher & Student & Sets & $n$ & $k$ & Expected & $p$ & Animals \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Teacher & Student & Sets & $n$ & $k$ & Expected & $p$ & Animals \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
    ]
    first = True
    for key, h in pairings.groupby("scenario", sort=False):
        if not first:
            lines.append(r"\midrule")
        first = False
        lines.append(rf"\multicolumn{{8}}{{l}}{{\textit{{{_tex_title(key)}}}}} \\")
        for row in h.itertuples():
            cells = [
                display_name(row.teacher),
                display_name(row.student),
                f"{row.matched_sets}",
                f"{row.n:,}",
                f"{row.k:,}",
                f"{row.expected_null:.1f}",
                _tex_p(row.p, row.p_floor),
            ]
            lines.append(
                _tex_row(cells, row.p)[: -len(r" \\")] + f" & {row.animals}" + r" \\"
            )
    lines += [r"\end{longtable}", ""]
    return "\n".join(lines)


def write_log(out, log, units, scenarios, pairs_csv, n_perm, seed) -> None:
    meta = (
        out.parent / pairs_csv.parent.relative_to(FAMILY_OUT) / "complete_snapshot.json"
        if pairs_csv.is_relative_to(FAMILY_OUT)
        else pairs_csv.parent / "complete_snapshot.json"
    )
    snapshot = (
        f" (run-tree snapshot of {json.loads(meta.read_text())['created_utc'][:16]}Z)"
        if meta.is_file()
        else ""
    )
    lines = [
        f"exceedance figure and tables, built {datetime.now(timezone.utc):%Y-%m-%d %H:%M}Z",
        f"matched pairs: {pairs_csv}" + snapshot,
        f"test: cell-stratified exceedance permutation, n_perm = {n_perm}, seed = {seed}, "
        f"p floor = {1 / (n_perm + 1):.2e}",
        "",
        "sample sizes by scenario "
        "(pairings / matched sets / students n / controls / exceed k / expected under null "
        "/ null 95th / p / exchangeable expectation / exchangeable p):",
    ]
    n_ctrl = units[units.role.eq("control")].groupby("scenario").size()
    for row in scenarios.itertuples():
        lines.append(
            f"  {row.scenario}: {row.pairings} / {row.matched_sets} / {row.n} / "
            f"{n_ctrl[row.scenario]} / {row.k} / {row.expected_null:.1f} / "
            f"{row.null_p95:.0f} / {row.p:.3g} / {row.expected_exchangeable:.1f} / "
            f"{row.p_exchangeable:.3g}"
        )
    lines += ["", *log, ""]
    (out / "exceedance_log.txt").write_text("\n".join(lines))


def render(out: Path) -> None:
    units = pd.read_csv(out / "exceedance_units.csv", keep_default_na=False)
    scenarios = pd.read_csv(out / "table_exceedance_scenarios.csv")
    pairings = pd.read_csv(out / "table_exceedance_pairings.csv")
    drafts = out / DRAFTS
    drafts.mkdir(exist_ok=True)
    fig = draw(units, scenarios)
    fig.savefig(drafts / "fig4_exceedance_curves.pdf")
    fig.savefig(drafts / "fig4_exceedance_curves.png", dpi=300)
    plt.close(fig)
    magnitudes = exceedance_magnitudes(units)
    magnitudes.to_csv(out / "table_exceedance_magnitudes.csv", index=False)
    fig = draw_table(scenarios, magnitudes)
    fig.savefig(drafts / "fig4_exceedance_table.pdf")
    fig.savefig(drafts / "fig4_exceedance_table.png", dpi=300)
    plt.close(fig)
    (out / "table_exceedance_scenarios.tex").write_text(scenario_tex(scenarios))
    (out / "table_exceedance_pairings.tex").write_text(pairing_tex(pairings))


@app.command()
def main(
    out_dir: Path = typer.Option(DEFAULT_OUT, help="output directory, repo-relative"),
    pairs_csv: Path = typer.Option(DEFAULT_PAIRS, help="matched pairs, repo-relative"),
    n_perm: int = typer.Option(20000, min=10000),
    seed: int = typer.Option(0),
    render_only: bool = False,
):
    repo, root = bootstrap()
    out = out_dir if out_dir.is_absolute() else repo / out_dir
    if render_only:
        render(out)
        print(f"Redrew the figure and tables in {out}", flush=True)
        return
    pairs_path = pairs_csv if pairs_csv.is_absolute() else repo / pairs_csv
    out.mkdir(parents=True, exist_ok=True)
    log: list[str] = []
    units = load_units(repo, root, pairs_path, log)
    print(
        f"Loaded {len(units)} fine-tuned models; running permutation tests…", flush=True
    )
    units, scenarios, pairings = compute(units, n_perm=n_perm, seed=seed, log=log)
    run_checks(units, scenarios, pairings, log)
    units.to_csv(out / "exceedance_units.csv", index=False)
    scenarios.to_csv(out / "table_exceedance_scenarios.csv", index=False)
    pairings.to_csv(out / "table_exceedance_pairings.csv", index=False)
    write_log(out, log, units, scenarios, pairs_csv, n_perm, seed)
    render(out)
    print(scenarios.drop(columns=["p_floor", "n_perm"]).to_string(index=False))
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    app()
