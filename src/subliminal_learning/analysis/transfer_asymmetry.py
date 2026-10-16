"""Is transfer SYMMETRIC? A -> B against B -> A, per model pair and per model.

Every family here is a set of pretraining-sibling bases (cross-seed, cross-data-order, the
re-pretraining tree, the OLMo recipes / seeds); every ordered off-diagonal pair of its bases is
an arm. A pair enters the comparison only when BOTH of its arms are FINISHED on a shared design
-- the `transfer_headline.family_headline_arms` rule: a matched control on every design cell and
every listed teacher finite on every cell, dead runs kept as measured. The pair is read at the
LARGEST design both directions finish at (`Family.designs`, largest first), and every ordered arm
of every family is reported in the census table whether it is used or not, so an absent direction
is a stated exclusion rather than a silently one-sided comparison.

Values are prior-scaled treatment effects, ``(student - matched control of the same cell) /
receiver test prior`` (`tio.student_tree_prior`), so the two directions -- which have different
receivers, hence different priors -- are on one scale. The tests are the two comparison
estimators of ``stats.transfer_significance_tests``, chosen by the design: identical teacher tag
sets on both sides make the comparison PAIRED (`paired_cluster_relabel_permutation`, the swapped
cluster is the teacher, the stratum the student cell), different tag sets make it unpaired
(`arm_relabel_cell_permutation`, the relabelled arm is direction+teacher). Both report the
two-sided p of a comparison, for the mean contrast and for the win probability.

Per MODEL the same machinery answers whether a base is a net receiver: ``in`` (partner -> model)
against ``out`` (model -> partner) over every partner whose pair was complete, stratified by
``partner|cell``.

Analysis layer: loading is `tio.load_cell_matrix` with pinned cells (memoized) and nothing else
expensive; no permutation code lives here and no figure -- `pairs_table_latex` / `roles_table_latex`
render the two tables, `scripts.transfer_asymmetry_readout` writes them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from subliminal_learning.analysis import transfer_conditions as tc
from subliminal_learning.analysis import transfer_significance_io as tio
from subliminal_learning.config import resolve_output_root
from subliminal_learning.stats import transfer_significance_tests as tt

__all__ = [
    "Family",
    "FAMILIES",
    "CONDITION_A",
    "CONDITION_B",
    "ROLE_IN",
    "ROLE_OUT",
    "PAIR_COLUMNS",
    "ROLE_COLUMNS",
    "CENSUS_COLUMNS",
    "EXCLUDED_COLUMNS",
    "VALUE_PRIOR",
    "VALUE_DELTA",
    "VALUE_MODES",
    "VALUE_UNITS",
    "VALUE_SHORT",
    "ArmLoad",
    "AsymmetryResult",
    "family_teachers",
    "family_designs",
    "load_arm",
    "contrast_row",
    "family_pairs",
    "family_roles",
    "analyze_family",
    "analyze",
    "pairs_table_latex",
    "roles_table_latex",
]

#: The two conditions of a pair contrast: A is the base listed FIRST in the family's base list,
#: so the orientation is fixed before any number is read and the SIGN says who sends more.
CONDITION_A, CONDITION_B = "A->B", "B->A"

#: The two conditions of a model's role contrast: what it receives against what it sends.
ROLE_IN, ROLE_OUT = "in", "out"

#: The two value modes. ``"prior"`` divides the matched gap by the receiver's test prior (the
#: two directions have different receivers, so their priors differ); ``"delta"`` reports the raw
#: gap in PERCENTAGE POINTS of trait preference, which no prior enters. The census reports the
#: prior either way.
VALUE_PRIOR, VALUE_DELTA = "prior", "delta"
VALUE_MODES = (VALUE_PRIOR, VALUE_DELTA)

#: How a mode's ``value`` column reads, for the tables and ``results.json``.
VALUE_UNITS = {VALUE_PRIOR: "receiver test prior units, (student - control) / prior",
               VALUE_DELTA: "percentage points, 100 x (student - control)"}

#: The short unit shown in a LaTeX column header.
VALUE_SHORT = {VALUE_PRIOR: "×prior", VALUE_DELTA: "pp"}

#: Test-kind labels of the two estimators.
TEST_PAIRED = "paired cluster relabel"
TEST_UNPAIRED = "arm relabel"


@dataclass(frozen=True)
class Family:
    """One family of pretraining siblings: every ordered off-diagonal pair of `bases` is an arm.

    Attributes:
        name: Family label.
        animals: Trait animals read, in order.
        bases: Base tokens, in the order that fixes each pair's A (first) and B (later).
        teachers: The sender's teacher param_tags -- a repo-relative manifest path (``{animal}``
            is substituted; a ``.tsv`` is read as first-column tags, a ``.json`` by its
            ``param_tags`` key) shared by every sender, or ``{sender token: path | tags}`` when
            the senders were selected separately.
        designs: ``(label, repo-relative cells JSON)`` LARGEST FIRST: a pair is read at the first
            design both its directions finish at.
        tseed: Teacher finetune seed in the tag dir (the OLMo arms use 0).
        ftseed: Student finetune seed read on both sides.
        control_source: Matched control arm, as `tio.load_cell_matrix`.
    """

    name: str
    animals: tuple[str, ...]
    bases: tuple[str, ...]
    teachers: str | Mapping[str, str | Sequence[str]]
    designs: tuple[tuple[str, str], ...]
    tseed: int = 1
    ftseed: int = 1
    control_source: str = "base"


_PYTHIA_DESIGNS = (("core56", "run_logs/registry/core56_cells.json"),
                   ("core30", "run_logs/registry/core30_cells.json"))
_OLMO_DESIGNS = (("core35", "run_logs/campaigns/dd1b/olmo_core35_cells.json"),
                 ("core30", "run_logs/campaigns/dd1b/olmo_core30_cells.json"))

#: The stride-3 core30 cells (10) of the full-length prv1 <-> pr1234 trajectory runs, the only
#: design that pair was run at.
_STRIDE3_DESIGN = ("core30 stride-3",
                   "run_logs/campaigns/prv1_pr1234_fulltraj_20261015/registry/prv12pr1234ft/"
                   "controls.json")

#: The five families of the readout. ``re-pretrain`` is core30 (the tree was never run at
#: core56), falling back to `_STRIDE3_DESIGN`; the OLMo families carry per-sender teacher sets,
#: so their tests are unpaired.
FAMILIES: dict[str, Family] = {
    "cross-seed": Family(
        "cross-seed", ("dog", "cat", "lion", "pig"), ("orig", "fs1", "fs2", "fs3"),
        "run_logs/registry/{animal}_top10_kband_configs.json", _PYTHIA_DESIGNS),
    "cross-data-order": Family(
        "cross-data-order", ("dog",), ("orig", "ds1", "ds2", "ds3"),
        "run_logs/registry/{animal}_top10_kband_configs.json", _PYTHIA_DESIGNS),
    "re-pretrain": Family(
        "re-pretrain", ("dog",), ("prv1", "wc1", "wc2", "pr1234", "orig"),
        "run_logs/registry/dog_top10_kband_configs.json", (*_PYTHIA_DESIGNS[1:], _STRIDE3_DESIGN)),
    "olmo-recipe": Family(
        "olmo-recipe", ("elephant",), ("dd-1b", "dd-1b-dclm", "dd-1b-c4"),
        {"dd-1b": "run_logs/registry/dolma_top10_senders.json",
         "dd-1b-c4": "run_logs/registry/c4_top10_senders.json",
         "dd-1b-dclm": "run_logs/campaigns/olmo_dclm_to_seed2_core35_20260921/teachers.tsv"},
        _OLMO_DESIGNS, tseed=0),
    "olmo-seed": Family(
        "olmo-seed", ("elephant",), ("dd-1b", "dd-1b-seed2"),
        {"dd-1b": "run_logs/registry/dolma_top10_senders.json",
         "dd-1b-seed2": ("ep10_lr4e-4_r64a128", "ep10_lr5e-4_r64a64", "ep10_lr5e-4_r64a128")},
        _OLMO_DESIGNS, tseed=0),
}

#: One row per (family, animal, pair): the two directions and their contrast.
PAIR_COLUMNS = ["family", "animal", "A", "B", "design", "n_cells", "runs A->B", "runs B->A",
                "mean A->B", "mean B->A", "contrast", "p_two_sided", "p_upper", "p_lower",
                "floor", "exact", "n_labelings", "test", "win prob", "win p_two_sided",
                "matched share"]

#: One row per (family, animal, model): what it receives against what it sends.
ROLE_COLUMNS = ["family", "animal", "model", "design", "n_cells", "runs in", "runs out",
                "mean in", "mean out", "contrast", "p_two_sided", "p_upper", "p_lower", "floor",
                "exact", "n_labelings", "test", "win prob", "win p_two_sided", "matched share"]

#: One row per (family, animal, ordered arm, design) -- the completeness census.
CENSUS_COLUMNS = ["family", "animal", "sender", "receiver", "design", "cells with control",
                  "design cells", "runs", "expected runs", "prior", "complete", "reason"]

#: One row per excluded pair, with the per-direction reason at every design tried.
EXCLUDED_COLUMNS = ["family", "animal", "A", "B", "reason A->B", "reason B->A"]


def family_designs(family: Family, repo: Path) -> list[tuple[str, list[str]]]:
    """``[(design label, cell names)]`` of a family, largest first."""
    return [(label, tio.resolve_cells(repo / rel)) for label, rel in family.designs]


def family_teachers(family: Family, animal: str, sender: str, repo: Path) -> list[str]:
    """The param_tags of one sender in `family`, read off its manifest (or listed inline)."""
    spec = family.teachers if isinstance(family.teachers, str) else family.teachers[sender]
    if not isinstance(spec, str):
        return list(spec)
    path = repo / spec.format(animal=animal)
    if path.suffix == ".tsv":
        return [line.split("\t")[0] for line in path.read_text().splitlines() if line.strip()]
    return list(json.loads(path.read_text())["param_tags"])


@dataclass(frozen=True)
class ArmLoad:
    """One ordered arm at one design: its prior-scaled runs, or why it is not usable.

    Attributes:
        frame: Tidy ``(sender, teacher, cell, value)`` rows, one per finite run, value in prior
            units.
        teachers: The param_tags asked for.
        prior: The receiver's test prior.
        complete: Whether the finished-set rule holds (used only then).
        reason: Why it is not complete (``""`` when it is).
        census: The arm's `CENSUS_COLUMNS` row.
    """

    frame: pd.DataFrame
    teachers: list[str]
    prior: float
    complete: bool
    reason: str
    census: dict


def load_arm(family: Family, animal: str, sender: str, receiver: str, design: str,
             cells: Sequence[str], *, repo: Path, output_root: Path | None = None,
             priors: dict | None = None, value: str = VALUE_PRIOR) -> ArmLoad:
    """One ordered arm at one design, under the finished-set rule.

    Args:
        family: The family the arm belongs to.
        animal: Trait animal (the teachers' and the students' subtree).
        sender, receiver: Base tokens.
        design: Design label (for the census row).
        cells: The design's cell names.
        repo: Repo root (the teacher manifests resolve against it).
        output_root: Output tree root; defaults to the paths YAML's.
        priors: Optional ``{(animal, base): prior}`` cache shared across arms.
        value: `VALUE_MODES` -- ``"prior"`` scales the matched gap by the receiver's test prior
            (a missing prior then excludes the arm), ``"delta"`` reports it in percentage points,
            which needs no prior.

    Returns:
        An `ArmLoad`; incomplete arms keep their (possibly empty) frame and carry the reason.
    """
    if value not in VALUE_MODES:
        raise ValueError(f"value must be one of {VALUE_MODES}, got {value!r}")
    root = resolve_output_root(output_root)
    teachers = family_teachers(family, animal, sender, repo)
    m = tio.load_cell_matrix(animal, sender, receiver, teachers, cells=list(cells),
                             ftseed=family.ftseed, tseed=family.tseed,
                             control_source=family.control_source, output_root=root)
    key = (animal, tio.base_name(receiver))
    priors = {} if priors is None else priors
    if key not in priors:
        priors[key] = tio.student_tree_prior(animal, receiver, output_root=root)
    prior = priors[key]

    treat = np.delete(m.values, m.control_col, axis=1) if m.cells else np.zeros((0, 0))
    n_run, expected = int(np.isfinite(treat).sum()), len(cells) * len(teachers)
    scale = (1.0 / prior) if value == VALUE_PRIOR else 100.0
    rows = []
    if m.cells and np.isfinite(scale):
        ctrl = m.values[:, m.control_col]
        for j, tag in enumerate(teachers):
            for i, cell in enumerate(m.cells):
                v = m.values[i, j]
                if np.isfinite(v) and np.isfinite(ctrl[i]):
                    rows.append({"sender": sender, "teacher": tag, "cell": cell,
                                 "value": float((v - ctrl[i]) * scale)})
    reason = ""
    if value == VALUE_PRIOR and not np.isfinite(prior):
        reason = "no prior"
    elif not m.cells:
        # no control cell at all: the matched design is empty, so the treatment runs are not
        # even read -- "not started" would claim more than is known about the students
        reason = f"no matched control (controls 0/{len(cells)})"
    elif len(m.cells) < len(cells) or m.missing:
        reason = ("not started" if n_run == 0 else
                  f"students {n_run}/{expected}, controls {len(m.cells)}/{len(cells)}")
    census = {"family": family.name, "animal": animal, "sender": sender, "receiver": receiver,
              "design": design, "cells with control": len(m.cells), "design cells": len(cells),
              "runs": n_run, "expected runs": expected, "prior": prior,
              "complete": not reason, "reason": reason}
    return ArmLoad(frame=pd.DataFrame(rows, columns=["sender", "teacher", "cell", "value"]),
                   teachers=teachers, prior=prior, complete=not reason, reason=reason,
                   census=census)


# ----------------------------------------------------------------------------------------------
# the contrast of two conditions
# ----------------------------------------------------------------------------------------------
def _condition_frame(a: pd.DataFrame, b: pd.DataFrame, label_a: str, label_b: str) -> pd.DataFrame:
    """The two sides as one tidy frame: ``condition``, ``teacher``, ``stratum``, ``arm``, ``value``.

    The relabelled arm of the unpaired test is ``condition|sender|teacher``: the SENDER belongs in
    the identity because a role contrast's ``in`` side pools several senders, and two of them
    reusing one tag string would otherwise be relabelled as a single arm. The swapped cluster of
    the paired test stays the teacher tag, which therefore carries all of a model's outbound runs
    at once.
    """
    parts = []
    for label, frame in ((label_a, a), (label_b, b)):
        part = frame.copy()
        part["condition"] = label
        part["arm"] = (label + "|" + part["sender"].astype(str) + "|"
                       + part["teacher"].astype(str))
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    return out[["condition", "sender", "teacher", "stratum", "arm", "value"]]


def _matched_share(frame: pd.DataFrame, label_a: str, label_b: str) -> float:
    """Share of matched ``(teacher, stratum)`` slots where the A run is above the B run."""
    wide = (frame.pivot_table(index=["teacher", "stratum"], columns="condition", values="value",
                              aggfunc="mean"))
    if label_a not in wide.columns or label_b not in wide.columns:
        return float("nan")
    both = wide[[label_a, label_b]].dropna()
    return float("nan") if both.empty else float((both[label_a] > both[label_b]).mean())


def contrast_row(frame: pd.DataFrame, label_a: str, label_b: str, *, paired: bool,
                 n_perm: int = 20000, seed: int = 0) -> dict:
    """The two permutation tests of one contrast plus its descriptives.

    Args:
        frame: A `_condition_frame`.
        label_a, label_b: The conditions; the contrast is A minus B.
        paired: True when both sides carry the SAME teacher tags, so the exchangeable unit is
            the teacher (`tt.paired_cluster_relabel_permutation`); else the arm
            (`tt.arm_relabel_cell_permutation`).
        n_perm, seed: Permutation settings.

    Returns:
        The shared numeric part of a `PAIR_COLUMNS` / `ROLE_COLUMNS` row, with the two
        conditions' counts and means under the ``a``/``b`` keys.
    """
    def run(statistic: str):
        if paired:
            return tt.paired_cluster_relabel_permutation(
                frame, cond_a=label_a, cond_b=label_b, statistic=statistic,
                condition_col="condition", cluster_col="teacher", stratum_col="stratum",
                value_col="value", n_perm=n_perm, seed=seed)
        return tt.arm_relabel_cell_permutation(
            frame, family_a=label_a, family_b=label_b, statistic=statistic,
            family_col="condition", arm_col="arm", stratum_col="stratum", value_col="value",
            n_perm=n_perm, seed=seed)

    mean, win = run("mean_contrast"), run("win_probability")
    side = frame.groupby("condition")["value"]
    return {"n_cells": int(mean.n_cells), "runs a": int((frame["condition"] == label_a).sum()),
            "runs b": int((frame["condition"] == label_b).sum()),
            "mean a": float(side.mean().get(label_a, float("nan"))),
            "mean b": float(side.mean().get(label_b, float("nan"))),
            "contrast": float(mean.observed), "p_two_sided": float(mean.p_two_sided),
            "p_upper": float(mean.p_upper), "p_lower": float(mean.p_lower),
            "floor": float(mean.floor), "exact": bool(mean.exact),
            "n_labelings": int(mean.n_labelings),
            "test": TEST_PAIRED if paired else TEST_UNPAIRED,
            "win prob": float(win.observed), "win p_two_sided": float(win.p_two_sided),
            "matched share": _matched_share(frame, label_a, label_b) if paired else float("nan")}


def _rename(row: dict, a: str, b: str) -> dict:
    """The ``a``/``b`` keys of `contrast_row` under a table's own condition names."""
    out = {}
    for k, v in row.items():
        out[k.replace(" a", f" {a}").replace(" b", f" {b}") if k.endswith((" a", " b")) else k] = v
    return out


# ----------------------------------------------------------------------------------------------
# pairs and roles
# ----------------------------------------------------------------------------------------------
@dataclass
class AsymmetryResult:
    """Everything the readout writes: the four tables."""

    pairs: pd.DataFrame
    roles: pd.DataFrame
    census: pd.DataFrame
    excluded: pd.DataFrame


def _pair_design(loads: dict, animal: str, a: str, b: str,
                 designs: Sequence[tuple[str, list[str]]]) -> str | None:
    """The first (largest) design at which BOTH directions of the pair are complete."""
    for label, _ in designs:
        if loads[(animal, a, b, label)].complete and loads[(animal, b, a, label)].complete:
            return label
    return None


def _reason(loads: dict, animal: str, sender: str, receiver: str,
            designs: Sequence[tuple[str, list[str]]]) -> str:
    """The per-direction exclusion reason, one clause per design tried."""
    return "; ".join(f"{label}: {loads[(animal, sender, receiver, label)].reason or 'complete'}"
                     for label, _ in designs)


def analyze_family(family: Family, *, repo: Path, output_root: Path | None = None,
                   n_perm: int = 20000, seed: int = 0,
                   value: str = VALUE_PRIOR) -> AsymmetryResult:
    """One family's pair table, role table, census and exclusions.

    Every ordered arm is loaded at every design in `Family.designs` (the loads are memoized in
    `tio.load_cell_matrix`), which is what the census reports; a pair is tested at the largest
    design both directions finish at, and a model's role pools every partner whose pair was
    tested. `value` picks the run value (`VALUE_MODES`); the census, which reports the prior, is
    the same either way.
    """
    root = resolve_output_root(output_root)
    designs = family_designs(family, repo)
    priors: dict = {}
    loads: dict = {}
    census, excluded, pairs, roles = [], [], [], []
    for animal in family.animals:
        for label, cells in designs:
            for sender in family.bases:
                for receiver in family.bases:
                    if sender == receiver:
                        continue
                    arm = load_arm(family, animal, sender, receiver, label, cells, repo=repo,
                                   output_root=root, priors=priors, value=value)
                    loads[(animal, sender, receiver, label)] = arm
                    census.append(arm.census)
        used: dict[str, list[tuple[str, str]]] = {}
        for i, a in enumerate(family.bases):
            for b in family.bases[i + 1:]:
                design = _pair_design(loads, animal, a, b, designs)
                if design is None:
                    excluded.append({"family": family.name, "animal": animal, "A": a, "B": b,
                                     "reason A->B": _reason(loads, animal, a, b, designs),
                                     "reason B->A": _reason(loads, animal, b, a, designs)})
                    continue
                ab, ba = loads[(animal, a, b, design)], loads[(animal, b, a, design)]
                frame = _condition_frame(ab.frame.assign(stratum=ab.frame["cell"]),
                                         ba.frame.assign(stratum=ba.frame["cell"]),
                                         CONDITION_A, CONDITION_B)
                paired = set(ab.teachers) == set(ba.teachers)
                row = contrast_row(frame, CONDITION_A, CONDITION_B, paired=paired,
                                   n_perm=n_perm, seed=seed)
                pairs.append({"family": family.name, "animal": animal, "A": a, "B": b,
                              "design": design, **_rename(row, CONDITION_A, CONDITION_B)})
                used.setdefault(a, []).append((b, design))
                used.setdefault(b, []).append((a, design))
        for model in family.bases:
            partners = used.get(model, [])
            if not partners:
                continue
            inbound, outbound, labels = [], [], []
            for partner, design in partners:
                labels.append(design)
                for side, (sender, receiver) in ((inbound, (partner, model)),
                                                 (outbound, (model, partner))):
                    arm = loads[(animal, sender, receiver, design)]
                    side.append(arm.frame.assign(stratum=partner + "|" + arm.frame["cell"]))
            paired = all(set(loads[(animal, p, model, d)].teachers)
                         == set(loads[(animal, model, p, d)].teachers) for p, d in partners)
            frame = _condition_frame(pd.concat(inbound, ignore_index=True),
                                     pd.concat(outbound, ignore_index=True), ROLE_IN, ROLE_OUT)
            row = contrast_row(frame, ROLE_IN, ROLE_OUT, paired=paired, n_perm=n_perm, seed=seed)
            roles.append({"family": family.name, "animal": animal, "model": model,
                          "design": ", ".join(sorted(set(labels))),
                          **_rename(row, ROLE_IN, ROLE_OUT)})
    return AsymmetryResult(
        pairs=pd.DataFrame(pairs).reindex(columns=PAIR_COLUMNS),
        roles=pd.DataFrame(roles).reindex(columns=ROLE_COLUMNS),
        census=pd.DataFrame(census).reindex(columns=CENSUS_COLUMNS),
        excluded=pd.DataFrame(excluded).reindex(columns=EXCLUDED_COLUMNS))


def family_pairs(family: Family, **kw) -> pd.DataFrame:
    """`analyze_family`'s pair table only."""
    return analyze_family(family, **kw).pairs


def family_roles(family: Family, **kw) -> pd.DataFrame:
    """`analyze_family`'s role table only."""
    return analyze_family(family, **kw).roles


def analyze(names: Sequence[str], *, repo: Path, output_root: Path | None = None,
            n_perm: int = 20000, seed: int = 0,
            value: str = VALUE_PRIOR) -> AsymmetryResult:
    """`analyze_family` over the named `FAMILIES` in `value` mode, the four tables concatenated."""
    parts = [analyze_family(FAMILIES[n], repo=repo, output_root=output_root, n_perm=n_perm,
                            seed=seed, value=value) for n in names]

    def cat(attr: str, columns: list[str]) -> pd.DataFrame:
        frames = [getattr(p, attr) for p in parts]
        return (pd.concat(frames, ignore_index=True) if frames
                else pd.DataFrame(columns=columns)).reindex(columns=columns)

    return AsymmetryResult(cat("pairs", PAIR_COLUMNS), cat("roles", ROLE_COLUMNS),
                           cat("census", CENSUS_COLUMNS), cat("excluded", EXCLUDED_COLUMNS))


# ----------------------------------------------------------------------------------------------
# LaTeX
# ----------------------------------------------------------------------------------------------
#: What the two value modes measure, spelled into a caption.
_CAPTION_VALUE = {
    VALUE_PRIOR: "Prior-scaled treatment effects (student minus matched control, divided by the "
                 "receiver's test prior)",
    VALUE_DELTA: "Absolute treatment effects (student minus matched control, in percentage "
                 "points of trait preference, no prior)",
}


def _pairs_caption(value: str) -> str:
    return (f"\\textbf{{Transfer asymmetry, model pair by model pair.}} {_CAPTION_VALUE[value]} of "
            "the two directions of each pair, with the permutation difference of means A$\\to$B "
            "minus B$\\to$A (per~\\ref{app:statistics}); the design each pair is read on is in "
            "the accompanying CSV. Bold = $p < 0.05$.")


def _roles_caption(value: str) -> str:
    return (f"\\textbf{{Net sender and receiver roles.}} {_CAPTION_VALUE[value]} of each model's "
            "inbound transfer (partner $\\to$ model) against its outbound transfer (model "
            "$\\to$ partner), over every partner whose pair is complete in both directions; the "
            "designs are in the accompanying CSV. Bold = $p < 0.05$.")


def _tex_table(df: pd.DataFrame, columns: Sequence[str], headers: Sequence[str], spec: str,
               caption: str, label: str) -> str:
    """A booktabs table of `columns`, the two p columns bold where significant."""
    lines = [r"% requires \usepackage{booktabs}", r"\begin{table}[!ht]", r"\centering",
             r"\footnotesize", rf"\begin{{tabular}}{{{spec}}}", r"\toprule",
             " & ".join(tc._tex(h) for h in headers) + r" \\", r"\midrule"]
    for _, row in df.iterrows():
        cells = []
        for c in columns:
            v = row[c]
            if c in ("p_two_sided", "win p_two_sided"):
                text = tc._tex(tc.fmt_p_compact(v))
                cells.append(rf"\textbf{{{text}}}" if tio._is_significant(v) else text)
            elif isinstance(v, (int, float, np.floating)) and not isinstance(v, bool):
                cells.append("" if v is None or pd.isna(v) else f"{float(v):.3g}")
            else:
                cells.append(tc._tex(v))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
              r"\end{table}"]
    return "\n".join(lines) + "\n"


def pairs_table_latex(df: pd.DataFrame, *, value: str = VALUE_PRIOR, caption: str | None = None,
                      label: str | None = None) -> str:
    """The booktabs rendering of the pair table: one row per (family, animal, pair); the design
    column stays in the CSV/JSON only.

    `value` names the mode the frame was built in (`VALUE_MODES`): it fixes the unit in the three
    mean/contrast headers and in the caption, and the ``delta`` table gets its own label.
    """
    unit = VALUE_SHORT[value]
    show = df.assign(pair=df["A"].astype(str) + " → " + df["B"].astype(str))
    columns = ["family", "animal", "pair", "mean A->B", "mean B->A", "contrast",
               "p_two_sided", "win p_two_sided"]
    headers = ["Family", "Animal", "Pair", f"Mean A→B ({unit})", f"Mean B→A ({unit})",
               f"Difference of means ({unit})", "P-value", "Win P-value"]
    return _tex_table(show, columns, headers, r"@{}l l l r r r r r@{}",
                      caption or _pairs_caption(value),
                      label or f"tab:transfer_asymmetry_pairs_{value}")


def roles_table_latex(df: pd.DataFrame, *, value: str = VALUE_PRIOR, caption: str | None = None,
                      label: str | None = None) -> str:
    """The booktabs rendering of the role table: one row per (family, animal, model); `value` as
    `pairs_table_latex`."""
    unit = VALUE_SHORT[value]
    columns = ["family", "animal", "model", "mean in", "mean out", "contrast",
               "p_two_sided", "win p_two_sided"]
    headers = ["Family", "Animal", "Model", f"Mean in ({unit})", f"Mean out ({unit})",
               f"Difference of means ({unit})", "P-value", "Win P-value"]
    return _tex_table(df, columns, headers, r"@{}l l l r r r r r@{}",
                      caption or _roles_caption(value),
                      label or f"tab:transfer_asymmetry_roles_{value}")
