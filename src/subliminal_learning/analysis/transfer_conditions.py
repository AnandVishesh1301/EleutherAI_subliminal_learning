"""Transfer existence by PRETRAINING CONDITION -- the table of notebooks/transfer_conditions.ipynb.

A re-cut of the headline table (`transfer_headline`, §0.0 of notebooks/transfer_significance.ipynb):
the same finished-set rule, the same arm-stratified exceedance permutation
(`transfer_headline.view_permutation`, 20 000 draws, seed 0, each (animal, arm) block against its
own matched-control maximum), regrouped into the fourteen pretraining conditions the paper argues
about and written with the abbreviated model names the paper uses (``160M``, ``wc1``,
``Dolma1.7``) instead of the raw on-disk tokens.

Rows: per condition one row per animal carrying a finished arm, then -- when at least two animals
do -- one stratified row (``Animal`` = ``All``) pooling every finished (animal, arm) block as its
own stratum. Columns: `CONDITION_COLUMNS`. `condition_table_latex` renders the same eleven
columns as a booktabs table.

Arm notation: ``{S}²`` is every ordered sender->receiver pair of distinct members of ``S``,
``\\`` subtracts the listed arms, a diagonal (same-model) block is written as the bare sender
set, and an index
range (``fs1--3``) folds three or more consecutive family members.

The module also carries the notebook's SECOND table -- which condition transfers MORE (a re-cut
of §3 of notebooks/transfer_significance.ipynb over the four Pythia conditions whose lineages are
160M / ds1-3 / fs1-3 / wc1-2 / rerun 160M): `COMPARISON_CONDITIONS`, `comparison_arms`,
`comparison_pairings`, `comparison_table`, `style_comparison_table`, `comparison_table_latex`,
plus the four DataDecide OLMo-1B conditions compared amongst themselves
(`OLMO_COMPARISON_CONDITIONS`, `compare_condition_matrices`) and `merge_comparison_tables`.
The statistics are `transfer_significance_io.compare_families`'s (arm-relabelling null within
(animal, cell) strata, every p two-sided by tail doubling); only the naming, the A-before-B
ordering and the columns are re-cut here. The OLMo pairings run those same tests on the
already-loaded existence matrices (`condition_matrix_frame`), because their registry arms carry
per-arm pilot-selected teacher sets `compare_families`' base-token loader cannot express.

The THIRD table (`direction_pairings`, `direction_table`, `direction_table_latex`) compares
registry arm families that share no cells -- Pythia → OLMo vs OLMo → Pythia, 160M → 410M vs
410M → 160M, same vs different data order across widths -- through
`transfer_headline.arm_family_test`: one scale-free summary per arm against its own matched
controls, the exact arm-relabelling permutation (and the exact sign-flip over model pairs when
the arms pair), every p two-sided; a pairing whose null has fewer than
`DIRECTION_MIN_LABELINGS` labellings is dropped and named in `direction_notes`.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from subliminal_learning.analysis import transfer_headline as th
from subliminal_learning.analysis import transfer_significance_io as tio

__all__ = [
    "CONDITION_COLUMNS",
    "CONDITION_LEAD",
    "ANIMAL_ORDER",
    "ALL_ANIMALS",
    "HP_GROUP",
    "HP_SUBLABELS",
    "OLMO_FAMILY",
    "NAME_MAP",
    "Condition",
    "arm_endpoints",
    "pretty_name",
    "compress_arms",
    "arms_text",
    "count_pct",
    "fmt_p_compact",
    "fmt_diff_compact",
    "count_text",
    "hp_cells",
    "comparison_hp_cells",
    "sparsify",
    "condition_blocks",
    "condition_table",
    "style_condition_table",
    "condition_notes",
    "condition_table_latex",
    "COMPARISON_COLUMNS",
    "COMPARISON_CONDITIONS",
    "comparison_arms",
    "unfinished_blocks",
    "comparison_pairings",
    "OLMO_COMPARISON_CONDITIONS",
    "olmo_comparison_arms",
    "finished_arms_by_condition",
    "condition_matrix_frame",
    "compare_condition_matrices",
    "comparison_table",
    "merge_comparison_tables",
    "style_comparison_table",
    "comparison_table_latex",
    "DIRECTION_COLUMNS",
    "DIRECTION_MIN_LABELINGS",
    "NO_PAIRED",
    "DirectionPairing",
    "direction_pairings",
    "direction_kept",
    "direction_table",
    "direction_notes",
    "style_direction_table",
    "direction_table_latex",
]

#: The eleven displayed columns, in order: the grouping levels first, the per-animal numbers last.
CONDITION_COLUMNS = ["Model Family", "Pretraining Condition", "Arms", "# Arms", "# Teacher HP",
                     "# Student HP", "# Total HP", "Animal", "Exceedances",
                     "Null 95th Perc. Exceedances", "P-value"]

#: The leading columns shown hierarchically (sparsified): a value appears only on the first row
#: of its run under equal higher levels, in the Styler and in the LaTeX alike.
CONDITION_LEAD = CONDITION_COLUMNS[:8]

#: The three run-count columns rendered under one ``# HP`` header group, and their sub-labels.
HP_GROUP = "# HP"
HP_SUBLABELS = {"# Teacher HP": "Teacher", "# Student HP": "Student", "# Total HP": "Total",
                "# Total HP A vs B": "Total A vs B"}

#: Animal row order inside a condition block.
ANIMAL_ORDER = ["dog", "cat", "lion", "pig", "elephant"]

#: The ``Animal`` cell of a stratified row: the fit over every animal with a finished arm.
ALL_ANIMALS = "All"

PRV1_BASE = "pythia-160m-pilereprov1-seed1234"
WC1_BASE = "pythia-160m-weightc-seed1"
WC2_BASE = "pythia-160m-weightc-seed2"

#: The Pythia ``Same-model`` condition's bases: EVERY lineage the table names, diagonal arms of
#: its own teachers, read for every animal -- which of them enter is the finished-set rule's
#: business, not the arm list's (as in every other block).
SAME_MODEL_BASES = ["orig", "fs1", "fs2", "fs3", "ds1", "ds2", "ds3", PRV1_BASE]

#: Short on-disk token -> the abbreviated model name the table prints.
NAME_MAP = {
    "orig": "160M", "original": "160M", "160m": "160M", "pythia-160m": "160M",
    "prv1": "rerun 160M", PRV1_BASE: "rerun 160M",
    "pythia-160m-weightc-seed1": "wc1", "pythia-160m-weightc-seed2": "wc2",
    # pythia-160m-seedN IS the full-seed-N base (``fsN``)
    "160m-seed1": "fs1", "160m-seed2": "fs2", "160m-seed3": "fs3",
    "410m": "410M", "410m-seed1": "410M-seed1", "410m-seed2": "410M-seed2",
    "410m-seed3": "410M-seed3",
    "OLMo-1B": "Dolma1.7", "dolma": "Dolma1.7", "dd-1b": "Dolma1.7",
    "OLMo-1B-seed2": "Dolma1.7 seed2", "OLMo-1B-seed3": "Dolma1.7 seed3",
    "dclm": "DCLM", "c4": "C4",
    "dolma-seed2": "Dolma1.7 seed2", "dclm-seed2": "DCLM seed2", "c4-seed2": "C4 seed2",
}

#: Token overrides inside the OLMo blocks, where ``orig``/``seed2`` mean the Dolma1.7 recipe.
OLMO_NAMES = {"orig": "Dolma1.7", "seed2": "Dolma1.7 seed2"}

#: The ``Model Family`` cell of the DataDecide blocks.
OLMO_FAMILY = "DataDecide OLMo-1B"

_INDEXED = re.compile(r"^(?P<prefix>.*?)(?P<index>\d+)$")


@dataclass(frozen=True)
class Condition:
    """One pretraining-condition block: its arms (finished or dropped) and how to name them.

    Attributes:
        family: ``Model Family`` cell.
        condition: ``Pretraining Condition`` cell.
        design: The design the arms were read on (``core30``, ``core56``, ...), for the notes.
        arms: The block's `transfer_headline.HeadlineArm`s across animals.
        names: Token -> full-name overrides layered over `NAME_MAP`.
        note: Anything qualifying the block, shown in the notes.
        alt_arms: ``{design: the SAME arms read on that design}`` for a design the block's rows
            are NOT read on (a block whose animals sit on different designs, e.g. dog on core56
            and cat / lion / pig on core30). They never make a row and never make a note: they
            only tell `unfinished_blocks` whether a block is finished on the OTHER design too.
    """

    family: str
    condition: str
    design: str
    arms: list[th.HeadlineArm] = field(default_factory=list)
    names: dict[str, str] = field(default_factory=dict)
    note: str = ""
    alt_arms: dict[str, list[th.HeadlineArm]] = field(default_factory=dict)


# ----------------------------------------------------------------------------------------------
# arm naming
# ----------------------------------------------------------------------------------------------
def arm_endpoints(label: str) -> tuple[str, str]:
    """``(sender token, receiver token)`` of an arm label.

    Handles both spellings the loaders produce: a family arm's ``sender→receiver`` and a registry
    name's ``sender → receiver (animal), design`` (everything from the animal parenthesis on is
    dropped). A registry diagonal spells its receiver ``X same-model``; the marker is dropped so
    the arm reads as the diagonal it is (``dolma → dolma``).

    Args:
        label: `transfer_headline.HeadlineArm.label`.

    Returns:
        The two endpoint tokens, stripped.

    Raises:
        ValueError: If the label carries no arrow.
    """
    core = label.split(" (")[0].split(", ")[0]
    parts = re.split(r"→|->", core)
    if len(parts) != 2:
        raise ValueError(f"cannot read endpoints from {label!r}")
    return parts[0].strip(), re.sub(r"\s+same-model$", "", parts[1].strip())


def pretty_name(token: str, names: Mapping[str, str] | None = None) -> str:
    """The displayed model name of a short token (`NAME_MAP`, `names` taking priority).

    ``orig`` -> ``160M``, ``dclm-seed2`` -> ``DCLM seed2``; the seed families (``fs2``, ``ds1``,
    ``wc1``) and any unknown token pass through unchanged.
    """
    if names and token in names:
        return names[token]
    return NAME_MAP.get(token, token)


def _families(nodes: Sequence[str]) -> dict[str, list[str]]:
    """``{prefix: members}`` of the indexed name families (``data-seed1/2/3``) among `nodes`."""
    groups: dict[str, list[str]] = {}
    for n in nodes:
        m = _INDEXED.match(n)
        if m and m.group("prefix"):
            groups.setdefault(m.group("prefix"), []).append(n)
    return {p: v for p, v in groups.items() if len(v) > 1}


def _index_of(node: str) -> int:
    """The trailing index of an indexed family member (``fs2`` -> ``2``)."""
    return int(_INDEXED.match(node).group("index"))


def _set_notation(nodes: Sequence[str]) -> list[str]:
    """`nodes` with every run of three or more consecutive family members folded into an index
    range, in first-appearance order.

    ``["160M", "fs1", "fs2", "fs3"]`` -> ``["160M", "fs1–3"]``; a two-member or non-contiguous
    family is left written out (``["wc1", "wc2"]``, ``["fs1", "fs3"]``).
    """
    seq = list(dict.fromkeys(nodes))
    label, skip = {}, set()
    for prefix, members in _families(seq).items():
        idx = sorted(members, key=_index_of)
        runs, run = [], [idx[0]]
        for n in idx[1:]:
            if _index_of(n) == _index_of(run[-1]) + 1:
                run.append(n)
            else:
                runs.append(run)
                run = [n]
        runs.append(run)
        for r in (x for x in runs if len(x) >= 3):
            head = min(r, key=seq.index)
            label[head] = f"{prefix}{_index_of(r[0])}–{_index_of(r[-1])}"
            skip.update(set(r) - {head})
    return [label.get(n, n) for n in seq if n not in skip]


def _set_text(nodes: Sequence[str]) -> str:
    """``"{160M, fs1–3}"`` -- the braced member list of a set, index ranges folded."""
    return "{" + ", ".join(_set_notation(nodes)) + "}"


def _ordered_pairs(nodes: Sequence[str]) -> list[tuple[str, str]]:
    """Every ordered pair of distinct `nodes`, sender-major in `nodes` order."""
    return [(a, b) for a in nodes for b in nodes if a != b]


def _hub_segments(edges: dict[tuple[str, str], int], nodes: Sequence[str]
                  ) -> list[tuple[int, str]]:
    """The hub / clique segments of `edges` (consumed in place), in first-arm order.

    A hub joined to every member of an indexed family becomes ``160M → {ds1–3}`` (or
    ``{wc1, wc2} → rerun 160M`` when the family is the sender side); a family every ordered pair
    of which is present becomes ``{wc1, wc2}²``.
    """
    segments: list[tuple[int, str]] = []

    def both(a: str, b: str) -> bool:
        return (a, b) in edges and (b, a) in edges

    for prefix, members in _families(nodes).items():
        idx = sorted(members, key=_index_of)
        setname = _set_text(idx)
        for hub in [n for n in nodes if n not in members]:
            for arrow, ok in (("↔", lambda h, m: both(h, m)),
                              ("→", lambda h, m: (h, m) in edges and (m, h) not in edges),
                              ("←", lambda h, m: (m, h) in edges and (h, m) not in edges)):
                if not all(ok(hub, m) for m in idx):
                    continue
                first = min(edges[(hub, m)] if (hub, m) in edges else edges[(m, hub)]
                            for m in idx)
                for m in idx:
                    edges.pop((hub, m), None)
                    edges.pop((m, hub), None)
                left, right = (setname, hub) if arrow == "←" else (hub, setname)
                segments.append((first, f"{left} {'↔' if arrow == '↔' else '→'} {right}"))
                break
        if len(idx) >= 2 and all(both(x, y) for x, y in _ordered_pairs(idx)):
            segments.append((min(edges[p] for p in _ordered_pairs(idx)), f"{_set_text(idx)}²"))
            for p in _ordered_pairs(idx):
                edges.pop(p, None)
    return segments


#: A greedy hub term (``ds1 → {160M, ds2, ds3}``) replaces the arms it covers only when it covers
#: at least this many: three arms written as one term always beats the ``a → b`` / ``a ↔ b`` list
#: they would otherwise make, two do not (a reciprocal pair is already the single ``a ↔ b``).
_HUB_MIN = 3


def _greedy_hubs(edges: dict[tuple[str, str], int]) -> list[tuple[int, str]]:
    """The hub segments of the arms `_hub_segments` left behind (consumed in place), chosen
    greedily: the sender or receiver hub covering the MOST remaining arms becomes one
    ``hub → {members}`` / ``{members} → hub`` term, and the pass repeats until no hub covers
    `_HUB_MIN` of them.

    ``{160M→ds1, 160M→ds2, 160M→ds3, ds1→160M, ds1→ds2, ds1→ds3}`` -- no complete square, no
    family-wide hub -- becomes ``160M → {ds1–3}; ds1 → {160M, ds2, ds3}`` instead of five
    written-out arms. Ties go to a sender hub, then to the hub whose members fold into an index
    range, then to the earlier arm; the members keep their arm order.
    """
    segments: list[tuple[int, str]] = []
    while True:
        best = None
        for side in (0, 1):
            hubs: dict[str, list[tuple[str, str]]] = {}
            for p in edges:
                hubs.setdefault(p[side], []).append(p)
            for hub, ps in hubs.items():
                members = [p[1 - side] for p in ps]
                key = (-len(ps), side, len(_set_notation(members)) == len(members),
                       min(edges[p] for p in ps))
                if best is None or key < best[0]:
                    best = (key, hub, side, members, ps)
        if best is None or -best[0][0] < _HUB_MIN:
            return segments
        _, hub, side, members, ps = best
        for p in ps:
            edges.pop(p)
        segments.append((best[0][3], f"{hub} → {_set_text(members)}" if side == 0
                         else f"{_set_text(members)} → {hub}"))


def _edge_segments(edges: dict[tuple[str, str], int]) -> list[tuple[int, str]]:
    """The segments of the non-diagonal arms `edges` (``{pair: arm order}``).

    A complete factorial over every endpoint is one ``{S}²`` segment; a factorial short of at
    most TWO arms -- and of fewer arms than it has -- is ``{S}² \\ {a→b, c→d}``; otherwise the
    hub / clique segments (the largest complete sub-squares first), then the greedy hubs of
    `_greedy_hubs`, then the remaining arms written out one by one. A complete grid between
    two disjoint endpoint sets (every sender to every receiver, no reverse arm) is the one
    product term ``{S} → {R}``.
    """
    if not edges:
        return []
    nodes = list(dict.fromkeys([n for p in edges for n in p]))
    missing = [p for p in _ordered_pairs(nodes) if p not in edges]
    first = min(edges.values())
    if not missing:
        return [(first, f"{_set_text(nodes)}²")]
    senders = list(dict.fromkeys(a for a, _ in edges))
    receivers = list(dict.fromkeys(b for _, b in edges))
    if (len(senders) >= 2 and len(receivers) >= 2 and not set(senders) & set(receivers)
            and len(edges) == len(senders) * len(receivers)):
        # a complete sender × receiver grid (a seed grid across families): one product term
        return [(first, f"{_set_text(senders)} → {_set_text(receivers)}")]
    left = dict(edges)
    segments = _hub_segments(left, nodes)
    if not left:
        return segments
    if len(nodes) >= 3 and len(missing) <= 2 and len(missing) < len(edges):
        gone = ", ".join(f"{a}→{b}" for a, b in missing)
        return [(first, f"{_set_text(nodes)}² \\ {{{gone}}}")]
    segments += _greedy_hubs(left)
    for (a, b), i in sorted(left.items(), key=lambda kv: kv[1]):
        if (a, b) not in left:
            continue
        left.pop((a, b))
        if (b, a) in left:
            left.pop((b, a))
            segments.append((i, f"{a} ↔ {b}"))
        else:
            segments.append((i, f"{a} → {b}"))
    return segments


def compress_arms(pairs: Sequence[tuple[str, str]]) -> str:
    """The arm text of one row: displayed names, factorials written ``{S}²``, hubs written
    against a braced set, index families folded into ranges, diagonals written as the bare
    sender set.

    Every ordered pair of ``160M, ds1, ds2, ds3`` becomes ``{160M, ds1–3}²``; the same set short
    of two arms becomes ``{160M, ds1–3}² \\ {160M→ds3, ds1→ds2}``; the diagonals of
    ``orig, fs1–3`` become the one segment ``{160M, fs1–3}`` (never ``X → X``); a lone diagonal
    is its sender name, ``160M``. What neither a square nor a set-minus covers is written as the
    greedy hubs of `_greedy_hubs` (``160M → {ds1–3}; ds1 → {160M, ds2, ds3}``) and, for the arms
    no hub covers, one ``a → b`` / ``a ↔ b`` term each.

    Args:
        pairs: ``(sender name, receiver name)`` in the order the arms should be read.

    Returns:
        The segments joined by ``"; "``.
    """
    order = list(dict.fromkeys(pairs))
    segments: list[tuple[int, str]] = []
    diagonals = [(i, p[0]) for i, p in enumerate(order) if p[0] == p[1]]
    if diagonals:
        names = _set_notation([n for _, n in diagonals])
        text = names[0] if len(names) == 1 else _set_text([n for _, n in diagonals])
        segments.append((diagonals[0][0], text))
    segments += _edge_segments({p: i for i, p in enumerate(order) if p[0] != p[1]})
    return "; ".join(s for _, s in sorted(segments))


def arms_text(arms: Sequence[th.HeadlineArm], names: Mapping[str, str] | None = None) -> str:
    """`compress_arms` of `arms`' endpoints, rendered through `pretty_name`."""
    return compress_arms([tuple(pretty_name(t, names) for t in arm_endpoints(a.label))
                          for a in arms])


# ----------------------------------------------------------------------------------------------
# cell formatting
# ----------------------------------------------------------------------------------------------
def sparsify(df: pd.DataFrame, lead: Sequence[str]) -> list[list[str]]:
    """The hierarchically sparsified `lead` cells of every row of `df`: a value is printed only
    on the first row of its run of equal values under equal higher levels, blank otherwise (what
    a pandas MultiIndex shows, done by hand for the LaTeX).

    Args:
        df: Any frame carrying `lead`.
        lead: The grouping columns, outermost first.

    Returns:
        One list of strings (blank where suppressed) per row of `df`.
    """
    keys = [tuple(str(df[c].iloc[i]) for c in lead) for i in range(len(df))]
    out = []
    for i, key in enumerate(keys):
        prev = keys[i - 1] if i else None
        shown, same = [], prev is not None
        for level, value in enumerate(key):
            same = same and prev[level] == value
            shown.append("" if same else value)
        out.append(shown)
    return out


def count_pct(count: float, total: float) -> str:
    """``"57 (19.0%)"`` -- a count and its share of `total` to one decimal."""
    if not total:
        return f"{count:.0f}"
    return f"{count:.0f} ({100 * count / total:.1f}%)"


def fmt_p_compact(p) -> str:
    """A p-value the way both tables print it: `transfer_significance_io._fmt_p` at or above
    1e-3 (``0.0011``, ``0.014``, ``0.91``), else scientific with at most two significant figures
    and no padding -- ``5e-5`` (the 20 000-draw permutation floor), ``1.1e-4``, ``8e-4``.

    `transfer_significance_io._fmt_p` itself is left alone: other notebooks print its ``5.0e-05``.
    """
    text = tio._fmt_p(p)
    if not text or isinstance(p, str) or p >= 1e-3:
        return text
    mantissa, _, exponent = f"{p:.1e}".partition("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    sign = "-" if exponent.startswith("-") else ""
    return f"{mantissa}e{sign}{exponent.lstrip('+-').lstrip('0') or '0'}"


def fmt_diff_compact(v) -> str:
    """A ``Diff. in Means`` cell: a tiny effect (``|v| < 0.01``) in one-significant-figure
    scientific notation with a compact exponent (``-0.000516`` -> ``-5e-4``, ``0.00347`` ->
    ``3e-3``), anything else the usual ``.3g`` (``0.0775``, ``41.6``, ``-0.914``). Zero is ``0``
    and a missing value is blank -- the long decimals were widening the column.
    """
    if v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v):
        return ""
    v = float(v)
    if v == 0:
        return "0"
    if abs(v) >= 0.01:
        return f"{v:.3g}"
    mantissa, _, exponent = f"{v:.0e}".partition("e")
    sign = "-" if exponent.startswith("-") else ""
    return f"{mantissa}e{sign}{exponent.lstrip('+-').lstrip('0') or '0'}"


def count_text(values: Sequence[int]) -> str:
    """The distinct `values` in first-appearance order, joined by ``" / "`` (``"34 / 35"``)."""
    return " / ".join(str(v) for v in dict.fromkeys(values))


def hp_cells(products: Sequence[tuple[int, int]], runs: int | None = None) -> dict[str, str]:
    """The three ``# HP`` cells of a row: teachers per arm, design cells per arm, total runs.

    Args:
        products: ``(teachers, cells)`` per finished arm, in row order.
        runs: The runs actually TESTED, when that is fewer than the design's teachers x cells
            (a diverged student is a complete run of its arm but enters no statistic).
            ``None`` takes the design product.

    Returns:
        ``{"# Teacher HP": "10", "# Student HP": "30", "# Total HP": "1200"}``; when the row's
        arms differ, the per-arm columns carry the distinct values, ``"34 / 35"``.
    """
    return {"# Teacher HP": count_text([t for t, _ in products]),
            "# Student HP": count_text([c for _, c in products]),
            "# Total HP": str(sum(t * c for t, c in products) if runs is None else runs)}


def comparison_hp_cells(products_a: Sequence[tuple[int, int]],
                        products_b: Sequence[tuple[int, int]]) -> dict[str, str]:
    """The comparison table's three ``# HP`` cells: the two sides' teachers and design cells
    (distinct values joined by ``" / "``) and their run counts, ``"1200 vs 3600"``.

    Args:
        products_a, products_b: ``(teachers, cells)`` per arm of side A / side B.

    Returns:
        ``{"# Teacher HP": ..., "# Student HP": ..., "# Total HP A vs B": ...}``.
    """
    both = [*products_a, *products_b]
    return {"# Teacher HP": count_text([t for t, _ in both]),
            "# Student HP": count_text([c for _, c in both]),
            "# Total HP A vs B": f"{sum(t * c for t, c in products_a)} vs "
                                 f"{sum(t * c for t, c in products_b)}"}


def _products(arms: Sequence[th.HeadlineArm]) -> list[tuple[int, int]]:
    """``(teachers, cells)`` of each finished arm: matrix pools minus the control, matrix rows."""
    return [(m.values.shape[1] - 1, len(m.cells)) for m in (a.matrix for a in arms)]


# ----------------------------------------------------------------------------------------------
# the table
# ----------------------------------------------------------------------------------------------
def _row(cond: Condition, animal: str, arms: Sequence[th.HeadlineArm], *, pooled: bool,
         n_perm: int, seed: int) -> dict:
    """One (condition, animal view) row over already-restricted finished `arms`."""
    r = th.view_permutation(arms, n_perm=n_perm, seed=seed)
    runs = sum(th.arm_summary(a.matrix, a.prior)["runs"] for a in arms)
    q95 = r.null_quantiles["95"]
    return {"Model Family": cond.family, "Pretraining Condition": cond.condition,
            "Animal": animal, "Arms": arms_text(arms, cond.names), "# Arms": len(arms),
            **hp_cells(_products(arms), runs),
            "Exceedances": count_pct(r.observed, runs),
            "Null 95th Perc. Exceedances": count_pct(q95, runs),
            "P-value": r.p, "p": r.p, "pooled": pooled}


def condition_table(conditions: Sequence[Condition], *, n_perm: int = 20000,
                    seed: int = 0) -> pd.DataFrame:
    """The condition table: per condition one row per animal with a finished arm (in
    `ANIMAL_ORDER`) and, when at least two animals have one, a stratified row pooling every
    finished (animal, arm) block as its own stratum.

    Args:
        conditions: The blocks, e.g. from `condition_blocks`.
        n_perm, seed: `transfer_headline.view_permutation` parameters (the memo key).

    Returns:
        `CONDITION_COLUMNS` plus the numeric ``p`` and a boolean ``pooled``.
    """
    rows = []
    for cond in conditions:
        finished = [a for a in cond.arms if a.matrix is not None]
        animals = [x for x in ANIMAL_ORDER if any(a.animal == x for a in finished)]
        for animal in animals:
            rows.append(_row(cond, animal, [a for a in finished if a.animal == animal],
                             pooled=False, n_perm=n_perm, seed=seed))
        if len(animals) >= 2:
            ordered = [a for x in animals for a in finished if a.animal == x]
            rows.append(_row(cond, ALL_ANIMALS, ordered, pooled=True, n_perm=n_perm, seed=seed))
    return pd.DataFrame(rows).reindex(columns=[*CONDITION_COLUMNS, "p", "pooled"])


def _grouped_columns(columns: Sequence[str]) -> pd.MultiIndex:
    """`columns` as a two-level header: the run-count columns under one `HP_GROUP` level with
    their `HP_SUBLABELS`, every other column under a blank top level."""
    return pd.MultiIndex.from_tuples([(HP_GROUP, HP_SUBLABELS[c]) if c in HP_SUBLABELS
                                      else ("", c) for c in columns])


def _shown(df: pd.DataFrame, lead: Sequence[str], index_depth: int,
           drop: Sequence[str]) -> pd.DataFrame:
    """The displayed frame: the first `index_depth` grouping levels as the (sparsified by pandas)
    index, the REST of `lead` as hand-sparsified leading columns so the run-count columns can
    carry the `HP_GROUP` header, `drop` dropped, and the columns grouped by `_grouped_columns`."""
    blanked = sparsify(df, lead)
    show = df.drop(columns=[*drop, *lead]).copy()
    for offset, column in enumerate(lead[index_depth:]):
        show.insert(offset, column, [row[index_depth + offset] for row in blanked])
    show.index = pd.MultiIndex.from_arrays(
        [*[df[c].to_numpy() for c in lead[:index_depth]], np.arange(len(df))],
        names=[*lead[:index_depth], None])
    show.columns = _grouped_columns(list(show.columns))
    return show


def style_condition_table(df: pd.DataFrame):
    """Styler for `condition_table`: the grouping levels sparsified (each label shown once per
    run) -- the first four as a MultiIndex, the run-count columns and ``Animal`` as blanked
    leading columns under a ``# HP`` header group -- with a divider above each condition,
    stratified rows grey, and a significant ``P-value`` bold on green."""
    pooled = df["pooled"].fillna(False).astype(bool).to_numpy()
    # a block is a (family, condition) pair: ``Same-model`` names a Pythia AND an OLMo block
    conds = list(zip(df["Model Family"], df["Pretraining Condition"]))
    first = np.array([i == 0 or conds[i] != conds[i - 1] for i in range(len(conds))], dtype=bool)
    depth = 4
    show = _shown(df, CONDITION_LEAD, depth, ["pooled", "p"])
    grey = "background-color: rgba(128, 128, 128, 0.2)"
    green = "font-weight: bold; background-color: rgba(120, 200, 120, 0.35)"
    divider = "border-top: 2px solid currentColor"

    def row_css(i):
        return "; ".join(c for c in (grey if pooled[i] else "", divider if first[i] else "")
                         if c)

    styler = (show.style.apply(lambda r: [row_css(r.name[depth])] * len(r), axis=1)
              .map(lambda v: green if tio._is_significant(v) else "",
                   subset=[("", "P-value")])
              .format({("", "P-value"): fmt_p_compact}, na_rep=""))
    for level in range(depth):
        styler = styler.apply_index(
            lambda lv: [row_css(i) if first[i] else "" for i in range(len(lv))],
            axis="index", level=level)
    return (styler.set_table_styles([{"selector": "tbody th",
                                      "props": "text-align: left; vertical-align: top;"}])
            .hide(axis="index", level=depth))


def condition_notes(conditions: Sequence[Condition]) -> list[str]:
    """One bullet per condition: the design it was read on, which animals its ``All`` row pools
    when they are fewer than the four Pythia animals, every dropped arm with its coverage
    shortfall, and any registry-provided note (strays, excluded selection cells).

    A condition name carried by more than one model family (``Same-model``) is labelled with its
    family, so the bullets stay one-to-one with the blocks."""
    shared = {c.condition for c in conditions
              if len({x.family for x in conditions if x.condition == c.condition}) > 1}
    out = []
    for cond in conditions:
        bits = [f"design {cond.design}"]
        animals = [x for x in ANIMAL_ORDER
                   if any(a.animal == x and a.matrix is not None for a in cond.arms)]
        if 2 <= len(animals) < 4:
            bits.append(f"{ALL_ANIMALS} = {' + '.join(animals)}")
        if cond.note:
            bits.append(cond.note)
        bits += [f"dropped {a.label} [{a.animal}]: {a.dropped}"
                 for a in cond.arms if a.dropped]
        bits += sorted({a.note for a in cond.arms if a.note and a.matrix is not None})
        label = (f"{cond.family}, {cond.condition}" if cond.condition in shared
                 else cond.condition)
        out.append(f"{label} — " + "; ".join(bits))
    return out


# ----------------------------------------------------------------------------------------------
# the blocks
# ----------------------------------------------------------------------------------------------
def condition_blocks(repo: Path, teachers_by_animal: Mapping[str, Sequence[str]], *,
                     output_root: Path | None = None,
                     diverged: str = "drop") -> list[Condition]:
    """The fourteen pretraining-condition blocks, arms loaded (finished or dropped).

    The same-model block is read on **core30** when every orig / full-seed diagonal that is
    finished on focus20 is finished on core30 too (the core30 wing-fill controls landed), else on
    focus20; which one was used is in the block's `Condition.design`.

    The cross-data-order block carries the full 12-arm set for every animal, per animal on the
    design its arms are finished on: dog on **core56**, cat / lion / pig on **core30** (their
    ``ds1 → ds2`` arms exist only there).

    Args:
        repo: Repo root (``run_logs/`` registries and cell sets resolve against it).
        teachers_by_animal: ``{animal: the canonical 10 k-band tags}``.
        output_root: Output tree root; defaults to the paths YAML's.
        diverged: `transfer_headline.DIVERGED_EXCLUDE` counts a student whose adapter went
            non-finite as a COMPLETE run of its arm and excludes it from every statistic (it
            stays NaN, which no test reads) instead of leaving its arm short of a finished set;
            ``"drop"`` (default) keeps the old rule.
    """
    rl = repo / "run_logs"
    pythia = ["dog", "cat", "lion", "pig"]
    core56, core30, focus20 = (rl / "registry" / "core56_cells.json", rl / "registry" / "core30_cells.json",
                               rl / "registry" / "focus20_cells.json")
    xarch, repre, olmo = (rl / "campaigns/xarch/xarch_significance_arms.json",
                          rl / "campaigns/repretrain/repretrain_significance_arms.json",
                          rl / "campaigns/repretrain/olmo_cross_seed_significance_arms.json")

    def fam(arms, animals, cells):
        return th.family_headline_arms(arms, animals, teachers_by_animal, cells=cells,
                                       output_root=output_root, diverged=diverged)

    def reg(path, names):
        # the conditions table reads every registry arm on its FULL design: the pilot selection
        # cells stay in (see `condition_notes` / the table legend)
        return th.registry_headline_arms(path, names, output_root=output_root,
                                         include_selection_cells=True, diverged=diverged)

    # same-model: EVERY lineage's own diagonal, read for every animal on the wider design when
    # the orig / full-seed ones are complete there; the finished-set rule decides which enter
    diag = tio.same_model_arms(["orig", "fs1", "fs2", "fs3"])
    done20 = {(a.animal, a.label) for a in fam(diag, pythia, focus20) if a.matrix is not None}
    on30 = fam(diag, pythia, core30)
    done30 = {(a.animal, a.label) for a in on30 if a.matrix is not None}
    use30 = done20 <= done30
    same_arms = fam(tio.same_model_arms(SAME_MODEL_BASES), pythia,
                    core30 if use30 else focus20)

    # cross-data-order: the FULL 12-arm set for every animal, each animal on the design its
    # finished arms live on -- dog on core56, cat / lion / pig on core30, where their ds1 → ds2
    # arms exist (on core56 that arm is part-run). The same arms are read on the other design
    # the core56 arms of the three moved animals are read too, as `alt_arms`, so the comparison
    # table can tell apart "unfinished on core56" from "never read on core56"
    # (see `unfinished_blocks`).
    cdo = tio.family_arm_set("cross-data-order")
    moved = [a for a in pythia if a != "dog"]
    cdo_cells = {a: core56 if a == "dog" else core30 for a in pythia}
    cdo_arms = [x for a in pythia for x in fam({a: cdo}, [a], cdo_cells[a])]
    cdo_alt = fam({a: cdo for a in moved}, moved, core56)
    prv1_cross = [f"{d} ({a})" for a in ("dog", "cat", "lion")
                  for d in ("prv1 → wc1", "wc1 → prv1", "prv1 → wc2", "wc2 → prv1")]
    # the three same-recipe cross-seed arms only: the reverse Dolma1.7 seed2 -> Dolma1.7
    # direction exists on disk as a 2-teacher core30 arm, too small to enter the table
    olmo_same_corpus = ["dolma → dolma-seed2 (elephant), core35",
                        "dclm → dclm-seed2 (elephant), core35",
                        "c4 → c4-seed2 (elephant), core35"]

    xcross = "Pythia/OLMo Cross Arch."
    return [
        Condition("Pythia", "Same-model", "core30" if use30 else "focus20", same_arms),
        Condition("Pythia", "Same init, diff. data order", "core56 (dog) / core30 (cat, lion, "
                  "pig)", cdo_arms,
                  note="cat / lion / pig are read on core30: their ds1 → ds2 arms exist only "
                       "there", alt_arms={"core56": cdo_alt}),
        Condition("Pythia", "Diff. init, same data order", "core30",
                  reg(repre, [f"{d} ({a})" for a in pythia
                              for d in ("wc1 → wc2", "wc2 → wc1")] + prv1_cross)),
        Condition("Pythia", "Diff. init, diff. data order", "core56",
                  fam(tio.family_arm_set("cross-seed"), pythia, core56)),
        Condition("Pythia", "Diff. model size, same data order, small to large", "core30",
                  reg(xarch, list(th.XWIDTH_ARMS["same", "up"]))),
        Condition("Pythia", "Diff. model size, same data order, large to small", "core30",
                  reg(xarch, list(th.XWIDTH_ARMS["same", "down"]))),
        Condition("Pythia", "Diff. model size, diff. data order, small to large",
                  "core30", reg(xarch, list(th.XWIDTH_ARMS["different", "up"]))),
        Condition("Pythia", "Diff. model size, diff. data order, large to small",
                  "core30", reg(xarch, list(th.XWIDTH_ARMS["different", "down"]))),
        Condition(xcross, "Diff. model arch., diff. data corpus, small to large",
                  "core35", reg(xarch, list(th.XARCH_DIRECTION_ARMS[0])),
                  note="the 3×3 Pythia-160M seed × Dolma1.7 seed grid; 160M → Dolma1.7 is the "
                       "complete 16-teacher arm (its top-10 re-cut is still running)"),
        Condition(xcross, "Diff. model arch., diff. data corpus, large to small",
                  "core30", reg(xarch, list(th.XARCH_DIRECTION_ARMS[1])),
                  note="the 3×3 Dolma1.7 seed × Pythia-160M seed grid; Dolma1.7 → 160M is the "
                       "seed-robust-8 arm (not the 11-teacher one), so every sender carries "
                       "the same 8 seed-robust teachers"),
        Condition(OLMO_FAMILY, "Same-model", "core35",
                  reg(olmo, ["dolma → dolma same-model (elephant), core35",
                             "dclm → dclm same-model (elephant), core35",
                             "c4 → c4 same-model (elephant), core35"]), names=OLMO_NAMES,
                  note="each recipe's own canonical teachers distilled back into itself "
                       "(teacher seed 0)"),
        Condition(OLMO_FAMILY, "Same init, diff. data corpus", "core35",
                  reg(olmo, ["dolma → dclm (elephant), core35",
                             "dclm → dolma (elephant), core35",
                             "dclm → c4 (elephant), core35",
                             "dolma → c4 (elephant), core35",
                             "c4 → dolma (elephant), core35",
                             "c4 → dclm (elephant), core35"]), names=OLMO_NAMES,
                  note="the DataDecide recipes share the seed-default init byte-for-byte"),
        Condition(OLMO_FAMILY, "Diff. init, diff. data corpus", "core35",
                  reg(olmo, ["dclm → seed2 (elephant), core35",
                             "dclm → c4-seed2 (elephant), core35",
                             "dolma → dclm-seed2 (elephant), core35",
                             "dolma → c4-seed2 (elephant), core35",
                             "c4 → seed2 (elephant), core35",
                             "c4 → dclm-seed2 (elephant), core35"]), names=OLMO_NAMES),
        Condition(OLMO_FAMILY, "Diff. init, same data corpus", "core35",
                  reg(olmo, olmo_same_corpus), names=OLMO_NAMES,
                  note="the reverse Dolma1.7 seed2 → Dolma1.7 direction exists only as a "
                       "2-teacher core30 arm and is left out until it is rerun at full scale"),
    ]


# ----------------------------------------------------------------------------------------------
# LaTeX
# ----------------------------------------------------------------------------------------------
_TEX_SUBS = [("×", r"$\times$"), ("↔", r"$\leftrightarrow$"), ("→", r"$\to$"),
             ("≥", r"$\geq$"), ("²", "$^{2}$"), ("–", "--"), ("—", "---")]

_TEX_CAPTION = (
    "\\textbf{Transfer existence by pretraining condition.} Stratified exceedance "
    "permutation test results, per~\\ref{app:statistics}. Bold = $p < 0.05$.")


#: A literal backslash in a cell (the set-minus of the arm notation) is held aside while the
#: LaTeX escapes -- which are themselves backslashes -- are inserted.
_SETMINUS = "\x00"


def _tex(s: str) -> str:
    """LaTeX-escape one table cell: braces, per cent, number signs, and the arrows / relations
    and set notation the arm texts use."""
    s = str(s).replace("\\", _SETMINUS)
    s = s.replace("{", r"\{").replace("}", r"\}")
    s = s.replace("%", r"\%").replace("#", r"\#")
    for a, b in _TEX_SUBS:
        s = s.replace(a, b)
    return s.replace(_SETMINUS, r"$\setminus$")


#: Header cells stacked over two lines so the narrow numeric columns stay narrow. Only the line
#: break is chosen here; the header WORDS are the displayed column names verbatim.
_TEX_HEADER_BREAKS = {
    "Model Family": ["Model", "Family"],
    "Pretraining Condition A": ["Pretraining", "Condition A"],
    "Pretraining Condition B": ["Pretraining", "Condition B"],
    "Treatment Effect Mode": ["Treatment", "Effect Mode"],
    "Null 95th Perc. Exceedances": ["Null 95th Perc.", "Exceedances"],
    "# Arms A vs B": ["# Arms", "A vs B"],
    "Diff. in Means": ["Diff. in", "Means"],
    "Mean P-value": ["Mean", "P-value"],
    "Win Prob.": ["Win", "Prob."],
    "Win P-value": ["Win", "P-value"],
}


#: Scoped to the table environment: tighter inter-column space, and a resizebox that only ever
#: SHRINKS (a table narrower than the text is left at its natural width).
_TABCOLSEP = r"\setlength{\tabcolsep}{3pt}"
_RESIZEBOX = r"\resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{%"


#: The columns whose numbers are right-aligned (their stacked header is right-aligned too).
_TEX_RIGHT = {"Null 95th Perc. Exceedances", "Diff. in Means", "Mean P-value", "Win Prob.",
              "Win P-value"}

#: The text columns whose stacked header is left-aligned, over their left-aligned body cells.
_TEX_LEFT = {"Model Family", "Pretraining Condition A", "Pretraining Condition B",
             "Treatment Effect Mode"}


def _tex_header(column: str, align: str = "c") -> str:
    """One header cell: the run-count columns abbreviated to their `HP_SUBLABELS` (the ``# HP``
    group line above carries the rest), otherwise stacked over two lines (a plain nested
    ``tabular``, no extra packages) when `column` is in `_TEX_HEADER_BREAKS`, so a numeric
    ``r``/``l`` column is no wider than its numbers; the ``p`` columns wrap on their own. The
    stack is centre-aligned (``[c]``), so its centre sits on the baseline the single-line
    labels sit on and every label in the row is vertically centred.
    """
    if column in HP_SUBLABELS:
        return _tex(HP_SUBLABELS[column])
    parts = _TEX_HEADER_BREAKS.get(column)
    if not parts:
        return _tex(column)
    body = r" \\ ".join(_tex(x) for x in parts)
    return rf"\begin{{tabular}}[c]{{@{{}}{align}@{{}}}}{body}\end{{tabular}}"


def _tex_header_lines(columns: Sequence[str]) -> list[str]:
    """The two header rows of a table: a spanning `HP_GROUP` ``\\multicolumn`` over the
    consecutive run-count columns with a ``\\cmidrule`` under it, then the column labels."""
    span = [i for i, c in enumerate(columns) if c in HP_SUBLABELS]
    top = ["" for _ in columns]
    top[span[0]] = rf"\multicolumn{{{len(span)}}}{{c}}{{{_tex(HP_GROUP)}}}"
    top = [c for i, c in enumerate(top) if i not in span[1:]]
    return [" & ".join(top) + r" \\",
            rf"\cmidrule(lr){{{span[0] + 1}-{span[-1] + 1}}}",
            " & ".join(_tex_header(c, "r" if c in _TEX_RIGHT else
                                    "l" if c in _TEX_LEFT else "c") for c in columns)
            + r" \\"]


def _tex_stack(parts: Sequence[str], align: str = "c") -> str:
    """`parts` stacked over as many lines as there are parts, as a plain nested ``tabular``.

    Used for the stacked HEADER cells only -- a body cell is never stacked: its parts are spread
    over the real rows of its block by `_spread_cells`.
    """
    parts = [x for x in parts if x]
    if len(parts) <= 1:
        return parts[0] if parts else ""
    body = r" \\ ".join(parts)
    return rf"\begin{{tabular}}[c]{{@{{}}{align}@{{}}}}{body}\end{{tabular}}"


#: An arms term longer than this (plain text, before the TeX escapes) is split at its arrow so
#: the column stays narrow; a term at or below it is left on one row.
_ARMS_WIDTH = 22


def _split_arrow(term: str) -> list[str]:
    """`term` split ONCE after its arrow when it is longer than `_ARMS_WIDTH`, the arrow ending
    the first part (``"Dolma1.7 → Dolma1.7 seed2"`` -> ``["Dolma1.7 →", "Dolma1.7 seed2"]``); a
    short term, or one carrying no spaced arrow (a ``{S}²`` factorial, a diagonal block's sender
    set, a set-minus tail), is one part."""
    if len(term) <= _ARMS_WIDTH:
        return [term]
    for arrow in ("→", "↔"):
        head, sep, tail = term.partition(f" {arrow} ")
        if sep:
            return [f"{head} {arrow}", tail]
    return [term]


#: A braced SET term longer than this (plain text) is split at its last top-level comma, so a
#: wide member list stays inside the column; ``{Dolma1.7, DCLM, C4}²`` (21) is left whole.
_SET_WIDTH = 21


def _split_set(term: str) -> list[str]:
    """`term` split ONCE at the LAST top-level comma of its braced member list when it is longer
    than `_SET_WIDTH`, the comma ending the first part (``"{160M, fs1–3, rerun 160M}"`` ->
    ``["{160M, fs1–3,", "rerun 160M}"]``, ``"{wc1, wc2, rerun 160M}²"`` -> ``["{wc1, wc2,",
    "rerun 160M}²"]``); a short term, or one that is no braced set, is one part.

    When that split still leaves a first part wider than `_ARMS_WIDTH` -- a long member list whose
    last member is short, ``"{160M, fs1–3, ds1, ds2, rerun 160M}"`` -- the term is split at the
    top-level comma nearest its MIDDLE instead (``["{160M, fs1–3, ds1,", "ds2, rerun 160M}"]``),
    so neither row overflows the column.
    """
    if len(term) <= _SET_WIDTH or not term.startswith("{"):
        return [term]
    depth, cuts = 0, []
    for i, ch in enumerate(term):
        depth += (ch == "{") - (ch == "}")
        if ch == "," and depth == 1:
            cuts.append(i)
    if not cuts:
        return [term]
    cut = cuts[-1]
    if cut + 1 > _ARMS_WIDTH:
        cut = min(cuts, key=lambda i: abs(i + 1 - len(term) / 2))
    return [term[:cut + 1], term[cut + 1:].lstrip()]


def _arms_parts(text: str) -> list[str]:
    """An ``Arms`` cell as one part per ``"; "``-separated term, a set-minus term split once more
    so the subtracted arms sit on their own row, and any part still wider than `_ARMS_WIDTH`
    split at its arrow (`_split_arrow`) and any braced set (a whole part, or one side of the
    arrow) still wider than `_SET_WIDTH` at its last top-level comma (`_split_set`).

    ``"{S}² \\ {a→b}; X → Y"`` -> ``["{S}²", "\\ {a→b}", "X → Y"]``.
    """
    parts = []
    for term in (t for t in str(text).split("; ") if t):
        head, sep, gone = term.partition(" \\ ")
        split = _split_arrow(head)
        parts += [p for piece in split for p in _split_set(piece)]
        if sep:
            parts.append(f"\\ {gone}")
    return parts


def _condition_parts(text: str) -> list[str]:
    """A ``Pretraining Condition`` cell as one part per comma, commas dropped and every part
    capitalised (``"Diff. model size, same data order"`` -> ``["Diff. model size",
    "Same data order"]``)."""
    parts = [x.strip() for x in str(text).split(",") if x.strip()]
    return [p[0].upper() + p[1:] for p in parts]


def _family_parts(text: str) -> list[str]:
    """A ``Model Family`` cell split over two rows: after the word carrying the ``/``
    (``Pythia/OLMo Cross Arch.`` -> ``["Pythia/OLMo", "Cross Arch."]``), else before the model
    name (``DataDecide OLMo-1B`` -> ``["DataDecide", "OLMo-1B"]``); a one-word name is one part."""
    text = str(text)
    head, _, tail = text.partition(" ") if "/" in text else text.rpartition(" ")
    return [head, tail] if head and tail else [text]


#: ``{column: how to split it}`` -- the body cells spread over the rows of their block.
_PART_SPLITTERS = {"Model Family": _family_parts, "Pretraining Condition": _condition_parts,
                   "Pretraining Condition A": _condition_parts,
                   "Pretraining Condition B": _condition_parts,
                   "Arms": _arms_parts, "Arms A": _arms_parts, "Arms B": _arms_parts}

#: ``{column: how many leading grouping levels its block is keyed on}`` per table.
_CONDITION_DEPTHS = {"Model Family": 1, "Pretraining Condition": 2, "Arms": 3}
_COMPARISON_DEPTHS = {"Model Family": 1, "Pretraining Condition A": 3,
                      "Pretraining Condition B": 3, "Arms A": 4, "Arms B": 5}


def _slot_keys(slots: Sequence[int | None], keys: Sequence[tuple]) -> list[tuple]:
    """The owning row key of every slot -- a padding slot inherits the row above it."""
    out, last = [], None
    for s in slots:
        if s is not None:
            last = keys[s]
        out.append(last)
    return out


def _runs(slot_keys: Sequence[tuple], depth: int) -> list[list[int]]:
    """The maximal runs of consecutive slots sharing their first `depth` key levels."""
    runs: list[list[int]] = []
    for i, key in enumerate(slot_keys):
        if runs and slot_keys[runs[-1][0]][:depth] == key[:depth]:
            runs[-1].append(i)
        else:
            runs.append([i])
    return runs


def _spread_cells(df: pd.DataFrame, lead: Sequence[str], depths: Mapping[str, int]
                  ) -> tuple[list[int | None], list[tuple], dict[tuple[int, str], str]]:
    """Spread the multi-part grouping cells over the REAL rows of their blocks.

    Part k of a cell goes on the k-th row of the block it groups (`depths` says on how many
    leading grouping levels that block is keyed); a block with fewer rows than parts gets blank
    padding rows appended after its last row, innermost block first, so an outer block sees the
    padding of its inner ones.

    Args:
        df: The table frame.
        lead: The grouping columns, outermost first.
        depths: ``{column: block depth}`` for the columns to spread.

    Returns:
        ``(slots, slot keys, {(slot, column): text})`` -- `slots` holds the frame row of each
        output row, ``None`` for a padding row.
    """
    keys = [tuple(str(df[c].iloc[i]) for c in lead) for i in range(len(df))]
    slots: list[int | None] = list(range(len(df)))
    for column in sorted(depths, key=lambda c: -depths[c]):
        depth = depths[column]
        runs = _runs(_slot_keys(slots, keys), depth)
        for run in reversed(runs):
            first = next(s for s in (slots[i] for i in run) if s is not None)
            short = len(_PART_SPLITTERS[column](df[column].iloc[first])) - len(run)
            if short > 0:
                slots[run[-1] + 1:run[-1] + 1] = [None] * short
    slot_keys = _slot_keys(slots, keys)
    cells: dict[tuple[int, str], str] = {}
    for column, depth in depths.items():
        for run in _runs(slot_keys, depth):
            first = next(s for s in (slots[i] for i in run) if s is not None)
            parts = _PART_SPLITTERS[column](df[column].iloc[first])
            for k, slot in enumerate(run):
                cells[(slot, column)] = parts[k] if k < len(parts) else ""
    return slots, slot_keys, cells


def _tex_lead_cells(columns: Sequence[str], values: Sequence[str], slot: int, row: int | None,
                    spread: Mapping[tuple[int, str], str]) -> list[str]:
    """One output row's grouping cells: the spread columns from `spread`, the others the
    hand-sparsified `values` (blank on a padding row)."""
    out = []
    for column, value in zip(columns, values):
        if (slot, column) in spread:
            out.append(_tex(spread[(slot, column)]))
        else:
            out.append(_tex(value) if row is not None and value else "")
    return out


def _block_label(values: list, i: int) -> str:
    """The block label for row `i`: the escaped value on the block's FIRST row, else empty."""
    return "" if i and values[i] == values[i - 1] else _tex(values[i])


def condition_table_latex(df: pd.DataFrame, *, caption: str = _TEX_CAPTION,
                          label: str = "tab:transfer_existence") -> str:
    """The booktabs rendering of `condition_table`: the same eleven columns (the three run-count
    ones under a spanning ``\\# HP`` header), `CONDITION_LEAD` sparsified by hand (each label on
    the FIRST row of its run), a ``\\cmidrule`` between condition blocks and a ``\\midrule`` where
    the Model Family changes, and a bold p-value where
    `transfer_significance_io._is_significant`. Needs only ``booktabs`` + ``graphicx``: every
    column is ``l`` / ``r`` (nothing wraps), the multi-line cells are hand-written nested
    ``tabular``s, and the whole table sits in a ``\\resizebox``.

    Args:
        df: A `condition_table` frame (needs `CONDITION_COLUMNS`; the numeric ``p`` is used for
            the bold test when present).
        caption, label: The table environment's caption and label.

    Returns:
        A complete ``table`` environment, newline-terminated.
    """
    lead = sparsify(df, CONDITION_LEAD)
    slots, keys, spread = _spread_cells(df, CONDITION_LEAD, _CONDITION_DEPTHS)
    spec = r"@{}l l l r r r r l r r r@{}"
    lines = [r"% requires \usepackage{booktabs,graphicx}", r"\begin{table}[!hp]", r"\centering",
             r"\footnotesize", _TABCOLSEP, _RESIZEBOX,
             rf"\begin{{tabular}}{{{spec}}}", r"\toprule",
             *_tex_header_lines(CONDITION_COLUMNS), r"\midrule"]
    for slot, i in enumerate(slots):
        if slot:
            if keys[slot][0] != keys[slot - 1][0]:
                lines.append(r"\midrule")
            elif keys[slot][1] != keys[slot - 1][1]:
                lines.append(rf"\cmidrule(l){{2-{len(CONDITION_COLUMNS)}}}")
        values = lead[i] if i is not None else [""] * len(CONDITION_LEAD)
        cells = _tex_lead_cells(CONDITION_LEAD, values, slot, i, spread)
        if i is None:
            cells += ["", "", ""]
        else:
            p = df["p"].iloc[i] if "p" in df.columns else df["P-value"].iloc[i]
            ptext = _tex(fmt_p_compact(df["P-value"].iloc[i]))
            cells += [_tex(df[c].iloc[i]) for c in CONDITION_COLUMNS[8:10]]
            cells.append(rf"\textbf{{{ptext}}}" if tio._is_significant(p) else ptext)
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}",
              rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table}"]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------------------------
# which condition transfers more: the comparison table
# ----------------------------------------------------------------------------------------------
#: The four Pythia conditions compared, in the existence table's order. A pairing's A is always
#: the condition listed FIRST here, so the orientation is fixed before the data is read; the
#: tests are two-sided and the SIGN of each contrast says which side is stronger.
COMPARISON_CONDITIONS = ["Same-model", "Same init, diff. data order",
                         "Diff. init, same data order",
                         "Diff. init, diff. data order"]

#: The fifteen displayed columns of `comparison_table`, in order: the grouping levels first,
#: the per-(animal, effect mode) numbers last. The design (core30 / core56) is carried by the
#: ``# Student HP`` column (30 = core30, 56 = core56), not by a column of its own.
COMPARISON_COLUMNS = ["Model Family", "Pretraining Condition A", "Pretraining Condition B",
                      "Arms A", "Arms B", "# Arms A vs B", "# Teacher HP", "# Student HP",
                      "# Total HP A vs B", "Animal",
                      "Treatment Effect Mode", "Diff. in Means", "Mean P-value", "Win Prob.",
                      "Win P-value"]

#: The leading columns shown hierarchically (sparsified), as `CONDITION_LEAD`.
COMPARISON_LEAD = COMPARISON_COLUMNS[:10]

#: The retired written-out spelling of the treatment effect, kept so an older frame (a pickled
#: one) still renders: every row now prints the plain `EFFECT_SHORT`, and the caption defines it.
EFFECT_LONG = "Δ (p_student − p_control)"
EFFECT_SHORT = "Δ"
EFFECT_RELATIVE = "Δ / prior"

#: The two effect modes shown: the absolute gap and (primary) the gap over the receiver's prior.
COMPARISON_MODES = {"absolute": EFFECT_SHORT, "relative": EFFECT_RELATIVE}

#: `transfer_significance_io.COMPARISON_COLUMNS` -> `COMPARISON_COLUMNS` name.
_COMPARISON_RENAME = {"animal": "Animal", "arms A vs B": "# Arms A vs B",
                      "mean contrast": "Diff. in Means", "mean p": "Mean P-value",
                      "win prob.": "Win Prob.", "win p": "Win P-value"}

#: The two p columns, bold + green below `transfer_significance_io.SIG_ALPHA`.
_COMPARISON_P = ["Mean P-value", "Win P-value"]


def comparison_arms() -> dict[str, list[tio.FamilyArm]]:
    """``{condition: its FamilyArm list}`` over `COMPARISON_CONDITIONS`, the same arm sets the
    existence table's finished blocks are built from.

    Same-model is `SAME_MODEL_BASES` -- every lineage's own diagonal, exactly the existence
    table's arm set; the ds and prv1 diagonals are finished for dog alone, and
    `unfinished_blocks` keeps them out of the other animals. The different-init,
    same-data-order condition is every ordered pair of ``wc1``, ``wc2`` and ``rerun 160M``
    (``{wc1, wc2, rerun 160M}²``, the existence table's arm set), written with the full on-disk
    base directory names -- `transfer_significance_io.compare_families`'s coverage rule drops the
    (animal, arm) blocks that are not finished.
    """
    same, cdo, cross, wc = COMPARISON_CONDITIONS[0], COMPARISON_CONDITIONS[1], \
        COMPARISON_CONDITIONS[3], COMPARISON_CONDITIONS[2]
    return {
        same: tio.same_model_arms(SAME_MODEL_BASES, family=same),
        cdo: tio.family_arm_set("cross-data-order", family_label=cdo),
        wc: [tio.FamilyArm(wc, sender, receiver)
             for sender, receiver in _ordered_pairs([WC1_BASE, WC2_BASE, PRV1_BASE])],
        cross: tio.family_arm_set("cross-seed", family_label=cross),
    }


def _design_label(arm: th.HeadlineArm) -> str | None:
    """``"core30"`` / ``"core56"`` -- the design a FINISHED arm was read on (its matrix rows)."""
    return None if arm.matrix is None else f"core{len(arm.matrix.cells)}"


def unfinished_blocks(conditions: Sequence[Condition],
                      arms: Mapping[str, Sequence[tio.FamilyArm]],
                      animals: Sequence[str], *,
                      design: str | None = None) -> set[tuple[str, str]]:
    """The ``(animal, arm)`` blocks of `arms` that the EXISTENCE table does not count as
    finished -- what `transfer_significance_io.compare_families` must be told to exclude so the
    comparison reads exactly the blocks the existence rows do.

    The existence rule is stricter than `normalized_effect_frame`'s full-design rule (every
    LISTED teacher on every cell, not just one run per cell), so e.g. lion
    ``wc1 → rerun 160M`` (7 of 10 teachers) enters the comparison but not the existence table.

    The rule is design-aware: a block is kept when the existence table has it finished on
    `design` itself, or when it is finished on some other design and was never READ on `design`
    (the cross-seed arms, read on core56 only, enter the core30 comparisons as they always did).
    A block READ on `design` -- through a block's `Condition.alt_arms` -- and unfinished there is
    excluded: cat / lion / pig ``ds1 → ds2`` is finished on core30 and part-run on core56, so it
    enters the core30 comparisons and not the core56 repeat.

    Args:
        conditions: The existence blocks (`condition_blocks` / `condition_table`'s input).
        arms: ``{condition: FamilyArm list}`` (`comparison_arms`).
        animals: The animals the comparison is read over.
        design: The design the comparison is read on (``core30`` / ``core56``); ``None`` asks
            only that the block be finished on SOME design.

    Returns:
        ``{(animal, FamilyArm.arm)}`` to pass as ``exclude_blocks``.
    """
    finished: dict[tuple[str, str, str], set[str]] = {}
    read: dict[tuple[str, str, str], set[str]] = {}
    for cond in conditions:
        blocks = [(_design_label(a), a) for a in cond.arms]
        blocks += [(d, a) for d, alt in cond.alt_arms.items() for a in alt]
        for d, a in blocks:
            key = (a.animal, *(pretty_name(t, cond.names) for t in arm_endpoints(a.label)))
            if d is not None:
                read.setdefault(key, set()).add(d)
                if a.matrix is not None:
                    finished.setdefault(key, set()).add(d)

    def keep(animal: str, arm: tio.FamilyArm) -> bool:
        key = (animal, pretty_name(arm.sender), pretty_name(arm.receiver))
        if key not in finished:
            return False
        return design is None or design in finished[key] or design not in read[key]

    return {(animal, arm.arm) for spec in arms.values() for arm in spec for animal in animals
            if not keep(animal, arm)}


def comparison_pairings(conditions: Sequence[str] = tuple(COMPARISON_CONDITIONS)
                        ) -> list[tuple[str, str]]:
    """The six unordered pairs of `conditions`, each oriented A-before-B by the LIST order (never
    by the data), and listed by A then B."""
    return [(conditions[i], conditions[j]) for i in range(len(conditions))
            for j in range(i + 1, len(conditions))]


def _side_arms_text(frame: pd.DataFrame, family: str, spec: Sequence[tio.FamilyArm],
                    names: Mapping[str, str] | None = None) -> str:
    """One side's `arms_text` on the arms of `spec` that actually entered `frame`."""
    present = set(frame.loc[frame["family"] == family, "arm"]) if len(frame) else set()
    return compress_arms([(pretty_name(a.sender, names), pretty_name(a.receiver, names))
                          for a in spec if a.arm in present])


def _side_products(frame: pd.DataFrame, family: str) -> list[tuple[int, int]]:
    """``(teachers, cells)`` of each arm of `family` in a `compare_families` effect frame."""
    side = frame[frame["family"] == family] if len(frame) else frame
    return [(block["teacher"].nunique(), block["cell"].nunique())
            for _, block in side.groupby("arm", sort=False)]


def _animal_text(animal: str) -> str:
    """``"dog+cat+lion+pig stratified"`` -> `ALL_ANIMALS`; a single animal passes through."""
    return ALL_ANIMALS if animal.endswith(" stratified") else animal


#: The four DataDecide OLMo-1B conditions compared, in the existence table's order (A is always
#: the earlier one here). They are compared amongst themselves only: no Pythia condition shares
#: their animal (elephant), their design (core35) or their receivers -- the OLMo ``Same-model``
#: condition is a different block from the Pythia one of the same name, so every lookup here is
#: keyed by (family, condition).
OLMO_COMPARISON_CONDITIONS = ["Same-model", "Same init, diff. data corpus",
                              "Diff. init, diff. data corpus", "Diff. init, same data corpus"]


def olmo_comparison_arms(conditions: Sequence[Condition]
                         ) -> dict[str, list[tio.FamilyArm]]:
    """``{condition: its FamilyArm list}`` over `OLMO_COMPARISON_CONDITIONS`, read off the
    FINISHED arms of the existence blocks themselves -- the registry arms carry per-arm
    pilot-selected teacher sets, so there is no base-token spelling to rebuild them from.

    Args:
        conditions: The existence blocks (`condition_blocks`).

    Returns:
        ``{condition: [FamilyArm(condition, sender token, receiver token)]}``, arm order kept.
    """
    # keyed by (family, condition): ``Same-model`` names a Pythia block AND an OLMo one
    by_name = {c.condition: c for c in conditions if c.family == OLMO_FAMILY}
    return {name: [tio.FamilyArm(name, *arm_endpoints(a.label))
                   for a in by_name[name].arms if a.matrix is not None]
            for name in OLMO_COMPARISON_CONDITIONS}


def finished_arms_by_condition(conditions: Sequence[Condition], *, family: str = OLMO_FAMILY
                               ) -> dict[str, list[th.HeadlineArm]]:
    """``{condition: its FINISHED arms}`` over the blocks of ONE model family.

    The family filter is what makes the mapping safe to key by condition name alone:
    ``Same-model`` names a Pythia block AND an OLMo one, so an unfiltered mapping would silently
    keep whichever came last.

    Args:
        conditions: The existence blocks (`condition_blocks`).
        family: The `Condition.family` to keep.

    Returns:
        ``{condition: [HeadlineArm]}`` in block order, arms with a loaded matrix only.
    """
    return {c.condition: [a for a in c.arms if a.matrix is not None]
            for c in conditions if c.family == family}


def condition_matrix_frame(arms: Sequence[th.HeadlineArm], family: str) -> pd.DataFrame:
    """The tidy per-run effect rows of already-loaded `arms`, in
    `transfer_significance_io.normalized_effect_frame`'s shape.

    One row per (arm, teacher pool, design cell): ``value_abs`` is the treatment value minus
    that cell's control value, ``value_rel`` the same gap over the arm's receiver prior
    (`transfer_headline.HeadlineArm.prior`) -- the same two treatment-effect modes
    `transfer_significance_io.compare_families` builds, taken straight off the finished
    `transfer_significance_io.CellMatrix` instead of re-reading the output tree. The retention
    value columns are carried as NaN: those modes need diagonal yardsticks this path does not
    load, and they are not shown in the comparison table.

    Args:
        arms: Finished `transfer_headline.HeadlineArm`s (``matrix`` set) of ONE condition.
        family: The `transfer_significance_io.FamilyArm` family label written into every row.

    Returns:
        Columns ``family, arm, animal, teacher, cell, stratum, value_abs, value_rel,
        value_recept, value_port, treatment, control``; ``stratum`` is the cell.
    """
    rows = []
    for a in arms:
        m = a.matrix
        sender, receiver = arm_endpoints(a.label)
        arm = f"{sender}->{receiver}"
        for i, cell in enumerate(m.cells):
            control = float(m.values[i, m.control_col])
            for j, pool in enumerate(m.pools):
                if j == m.control_col:
                    continue
                if not np.isfinite(m.values[i, j]):
                    continue                # a diverged run is no comparison observation
                gap = float(m.values[i, j]) - control
                rows.append({"family": family, "arm": arm, "animal": a.animal, "teacher": pool,
                             "cell": cell, "stratum": cell, "value_abs": gap,
                             "value_rel": gap / a.prior, "value_recept": np.nan,
                             "value_port": np.nan, "treatment": float(m.values[i, j]),
                             "control": control})
    return pd.DataFrame(rows).reindex(columns=[
        "family", "arm", "animal", "teacher", "cell", "stratum", "value_abs", "value_rel",
        "value_recept", "value_port", "treatment", "control"])


def compare_condition_matrices(arms_a: Sequence[th.HeadlineArm],
                               arms_b: Sequence[th.HeadlineArm], *, family_a: str,
                               family_b: str, n_perm: int = 20000, seed: int = 0
                               ) -> tio.MultiAnimalComparison:
    """`transfer_significance_io.compare_families`, run on ALREADY-LOADED existence matrices.

    Same effect frame (`condition_matrix_frame`), same strata ((animal, cell)), same unit (the
    arm, i.e. the animal-independent sender->receiver pair), same dead policy (``"keep"``: no
    dead filtering) and the SAME tests -- `stats.transfer_significance_tests`'
    `arm_relabel_cell_permutation` with the per-cell mean contrast and the per-cell win
    probability, plus `arm_set_permutation` on the arms' exceedance fractions, every p
    two-sided. Nothing is re-implemented here: the loading is replaced, the statistics are not.

    This is the path the DataDecide OLMo-1B pairings take, because their registry arms carry
    per-arm pilot-selected teacher sets and registry-specific control handling that
    `compare_families`' base-token loader cannot express; on a Pythia pairing it reproduces
    `compare_families`.

    Args:
        arms_a, arms_b: The two conditions' FINISHED `transfer_headline.HeadlineArm`s.
        family_a, family_b: Display labels (the condition names); A is the earlier condition.
        n_perm, seed: Permutation settings, as `compare_families`.

    Returns:
        A `transfer_significance_io.MultiAnimalComparison`, so `comparison_table` reads it
        exactly as it reads a `compare_families` result.

    Raises:
        ValueError: If either side has no finished arm.
    """
    if not arms_a or not arms_b:
        raise ValueError("both conditions need at least one finished arm")
    frame = pd.concat([condition_matrix_frame(arms_a, family_a),
                       condition_matrix_frame(arms_b, family_b)], ignore_index=True)
    animals = [x for x in ANIMAL_ORDER if (frame["animal"] == x).any()]
    per_animal = {}
    for animal in animals:
        block = frame[frame["animal"] == animal].reset_index(drop=True)
        per_animal[animal] = tio._family_comparison(
            block, family_a=family_a, family_b=family_b, missing={}, skipped=[],
            n_dead_treatment=None, n_dead_control=None, dead="keep", notes=[], n_perm=n_perm,
            seed=seed)
    stratified_frame = frame.copy()
    stratified_frame["stratum"] = stratified_frame["animal"] + "|" + stratified_frame["cell"]
    stratified = tio._family_comparison(
        stratified_frame, family_a=family_a, family_b=family_b, missing={}, skipped=[],
        n_dead_treatment=None, n_dead_control=None, dead="keep", notes=[], n_perm=n_perm,
        seed=seed)
    blocks = (stratified.frame.drop_duplicates(["arm", "animal"])
              .groupby("arm")["animal"].apply(list).to_dict())
    return tio.MultiAnimalComparison(family_a=family_a, family_b=family_b, animals=animals,
                                     per_animal=per_animal, stratified=stratified,
                                     blocks_present=blocks, missing_blocks=[])


def comparison_table(results: Mapping[tuple[str, str, str], object],
                     arms: Mapping[str, Sequence[tio.FamilyArm]], *, family: str = "Pythia",
                     names: Mapping[str, str] | None = None
                     ) -> tuple[pd.DataFrame, dict[tuple[str, str, str], str]]:
    """The condition-comparison table: `transfer_significance_io.compare_families` re-cut into
    `COMPARISON_COLUMNS`.

    Args:
        results: ``{(condition A, condition B, design): MultiAnimalComparison}``, in display
            order (A is the earlier condition of `COMPARISON_CONDITIONS`, see
            `comparison_pairings`).
        arms: ``{condition: FamilyArm list}`` (`comparison_arms`), for the ``Arms A`` / ``Arms B``
            cells.
        family: The constant ``Model Family`` cell.
        names: Token -> full-name overrides layered over `NAME_MAP` for the ``Arms`` cells (the
            OLMo blocks' `OLMO_NAMES`); every call is one model family, so one map is enough.

    Returns:
        ``(table, notes)``: `COMPARISON_COLUMNS` plus a boolean ``stratified``, and the
        ``(A, B, design) -> missing (animal, arm) blocks`` texts of the coverage rule.
    """
    rows, notes = [], {}
    for (cond_a, cond_b, design), res in results.items():
        raw = tio.family_comparison_table({(f"{cond_a} vs {cond_b}", design): res})
        raw = raw[raw["value mode"].isin(COMPARISON_MODES)]
        # one animal: its stratified fit repeats its own row, so only the animal row is shown
        if len(getattr(res, "animals", [])) < 2:
            raw = raw[~raw["stratified"].astype(bool)]
        frames = {**{a: c.frame for a, c in res.per_animal.items()},
                  f"{'+'.join(res.animals)} stratified": res.stratified.frame}
        for animal in dict.fromkeys(raw["animal"]):
            for mode in COMPARISON_MODES:
                block = raw[(raw["value mode"] == mode) & (raw["animal"] == animal)]
                for r in block.to_dict("records"):
                    frame = frames[r["animal"]]
                    rows.append({
                        **{k: r[v] for v, k in _COMPARISON_RENAME.items()},
                        "Model Family": family, "Pretraining Condition A": cond_a,
                        "Pretraining Condition B": cond_b,
                        **comparison_hp_cells(_side_products(frame, cond_a),
                                              _side_products(frame, cond_b)),
                        "Treatment Effect Mode": COMPARISON_MODES[mode],
                        "Animal": _animal_text(r["animal"]),
                        "Arms A": _side_arms_text(frame, cond_a, arms[cond_a], names),
                        "Arms B": _side_arms_text(frame, cond_b, arms[cond_b], names),
                        "stratified": bool(r["stratified"])})
        if res.missing_blocks:
            notes[(cond_a, cond_b, design)] = "; ".join(
                b.replace("excluded by the caller", "excluded: unfinished in the existence table")
                for b in res.missing_blocks)
    df = pd.DataFrame(rows).reindex(columns=[*COMPARISON_COLUMNS, "stratified"])
    df = df.reset_index(drop=True)
    return df, notes



def merge_comparison_tables(tables: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """`comparison_table` frames stacked in order, every absolute row carrying the plain
    `EFFECT_SHORT` label (the retired written-out `EFFECT_LONG` of an older frame is normalised
    away: the caption defines $\\Delta$)."""
    df = pd.concat(list(tables), ignore_index=True)
    df["Treatment Effect Mode"] = df["Treatment Effect Mode"].replace(EFFECT_LONG, EFFECT_SHORT)
    return df


def _comparison_blocks(df: pd.DataFrame) -> list[tuple[str, str, str]]:
    """The ``(family, A, B)`` block key of every row -- one block per condition pairing."""
    return list(zip(df["Model Family"], df["Pretraining Condition A"],
                    df["Pretraining Condition B"]))


def style_comparison_table(df: pd.DataFrame):
    """Styler for `comparison_table`: `COMPARISON_LEAD` sparsified (the first six levels as a
    MultiIndex, the run-count columns and ``Animal`` as blanked leading columns under a ``# HP``
    header group) with a divider above each pairing, stratified rows grey, and the two
    two-sided p's bold on green below `transfer_significance_io.SIG_ALPHA`."""
    strat = df["stratified"].fillna(False).astype(bool).to_numpy()
    keys = _comparison_blocks(df)
    first = np.array([i == 0 or keys[i] != keys[i - 1] for i in range(len(keys))], dtype=bool)
    depth = 6
    show = _shown(df, COMPARISON_LEAD, depth, ["stratified"])
    grey = "background-color: rgba(128, 128, 128, 0.2)"
    green = "font-weight: bold; background-color: rgba(120, 200, 120, 0.35)"
    divider = "border-top: 2px solid currentColor"

    def row_css(i):
        return "; ".join(c for c in (grey if strat[i] else "", divider if first[i] else "") if c)

    styler = (show.style.apply(lambda r: [row_css(r.name[depth])] * len(r), axis=1)
              .map(lambda v: green if tio._is_significant(v) else "",
                   subset=[("", c) for c in _COMPARISON_P])
              .format({**{("", c): fmt_p_compact for c in _COMPARISON_P},
                       ("", "Diff. in Means"): fmt_diff_compact,
                       ("", "Win Prob."): "{:.3f}"}, na_rep=""))
    for level in range(depth):
        styler = styler.apply_index(
            lambda lv: [row_css(i) if first[i] else "" for i in range(len(lv))],
            axis="index", level=level)
    return (styler.set_table_styles([{"selector": "tbody th",
                                      "props": "text-align: left; vertical-align: top;"}])
            .hide(axis="index", level=depth))


_COMPARISON_CAPTION = (
    "\\textbf{Relative Strength of transfer between pretraining conditions.} We compare "
    "pretraining conditions that vary weight initialization, data order, or data corpus in "
    "Pythia and DataDecide OLMo-1B families, computing both Difference in Means and Win "
    "Probability statistics on both ($\\Delta=$p(student) - p(control)) and $\\Delta/$prior. "
    "We compute two-sided $p$-values, and the sign of the Difference in Means (or the Win "
    "Probability above or below 0.5) determines which pretraining condition transfers more "
    "strongly: a positive difference (Win Probability above 0.5) means A is stronger, "
    "statistically significant or not. Bold = two-sided $p < 0.05$")


#: The three effect-mode cells in LaTeX (no amsmath).
_EFFECT_TEX = {
    EFFECT_LONG: r"$\Delta$",              # a pickled older frame: normalised on render
    EFFECT_SHORT: r"$\Delta$",
    EFFECT_RELATIVE: r"$\Delta$ / prior",
}


def _comparison_cell(df: pd.DataFrame, column: str, i: int) -> str:
    """One formatted body cell of `comparison_table_latex` (the numeric columns as the Styler)."""
    v = df[column].iloc[i]
    if column in _COMPARISON_P:
        text = _tex(fmt_p_compact(v))
        return rf"\textbf{{{text}}}" if tio._is_significant(v) else text
    if column == "Diff. in Means":
        return fmt_diff_compact(v)
    if column == "Win Prob.":
        return "" if pd.isna(v) else f"{v:.3f}"
    if column == "Treatment Effect Mode":
        return _EFFECT_TEX[v]
    return _tex(v)


def comparison_table_latex(df: pd.DataFrame, *, caption: str = _COMPARISON_CAPTION,
                           label: str = "tab:transfer_comparison") -> str:
    """The booktabs rendering of `comparison_table`: the same fifteen columns, `COMPARISON_LEAD`
    sparsified by hand (each label on the FIRST row of its run), a ``\\cmidrule`` between
    pairings and a full ``\\midrule`` where the Model Family changes, and a bold p where
    `transfer_significance_io._is_significant`. Needs only
    ``booktabs`` + ``graphicx``: every column is ``l`` / ``r`` (nothing wraps), the multi-line
    cells are hand-written nested ``tabular``s, and the whole table sits in a ``\\resizebox``.

    Args:
        df: A `comparison_table` frame (needs `COMPARISON_COLUMNS`).
        caption, label: The table environment's caption and label.

    Returns:
        A complete ``table`` environment, newline-terminated.
    """
    lead = sparsify(df, COMPARISON_LEAD)
    slots, keys, spread = _spread_cells(df, COMPARISON_LEAD, _COMPARISON_DEPTHS)
    spec = r"@{}l l l l l l r r l l l r r r r@{}"
    lines = [r"% requires \usepackage{booktabs,graphicx}", r"\begin{table}[!hp]", r"\centering",
             r"\scriptsize", _TABCOLSEP, _RESIZEBOX,
             rf"\begin{{tabular}}{{{spec}}}", r"\toprule",
             *_tex_header_lines(COMPARISON_COLUMNS), r"\midrule"]
    for slot, i in enumerate(slots):
        if slot and keys[slot][0] != keys[slot - 1][0]:
            lines.append(r"\midrule")
        elif slot and keys[slot][:3] != keys[slot - 1][:3]:
            lines.append(rf"\cmidrule(l){{4-{len(COMPARISON_COLUMNS)}}}")
        values = lead[i] if i is not None else [""] * len(COMPARISON_LEAD)
        cells = _tex_lead_cells(COMPARISON_LEAD, values, slot, i, spread)
        if i is None:
            cells += [""] * (len(COMPARISON_COLUMNS) - len(COMPARISON_LEAD))
        else:
            cells[-1] = _comparison_cell(df, "Animal", i) if lead[i][-1] else ""
            cells += [_comparison_cell(df, c, i) for c in COMPARISON_COLUMNS[10:]]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}",
              rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table}"]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------------------------
# which direction / data order transfers more: the direction table
# ----------------------------------------------------------------------------------------------
#: The fourteen displayed columns of `direction_table`, grouping levels first (as
#: `COMPARISON_COLUMNS`, with one arm-set p and one paired p in place of the four
#: comparison statistics).
DIRECTION_COLUMNS = ["Model Family", "Pretraining Condition A", "Pretraining Condition B",
                     "Arms A", "Arms B", "# Arms A vs B", "# Teacher HP", "# Student HP",
                     "# Total HP A vs B", "Animal", "Treatment Effect Mode", "Diff. in Means",
                     "P-value", "Paired P-value"]

#: The leading columns shown hierarchically (sparsified), as `COMPARISON_LEAD`.
DIRECTION_LEAD = DIRECTION_COLUMNS[:10]

#: A pairing enters the table only when its exact arm-relabelling null has at least this many
#: labellings: the two-sided floor ``2 / n`` must reach 0.05, else no outcome can be significant.
DIRECTION_MIN_LABELINGS = 40

#: `transfer_headline.arm_family_test` statistic -> its ``Treatment Effect Mode`` cell.
DIRECTION_MODES = {"exceed. frac": "Exceed. frac.", "rel. effect": EFFECT_RELATIVE}

#: The ``Paired P-value`` cell of an unpaired pairing.
NO_PAIRED = "–"

_DIRECTION_P = ["P-value", "Paired P-value"]
_TEX_HEADER_BREAKS["Paired P-value"] = ["Paired", "P-value"]
_TEX_RIGHT.add("Paired P-value")

_DIRECTION_CAPTION = (
    "\\textbf{Relative strength of transfer by direction and data order across model sizes and "
    "architectures.} Each arm is summarised against its own matched controls (exceedance "
    "fraction, and mean gap over the receiver prior); arm sets are compared with the exact "
    "arm-relabelling permutation test, two-sided; the paired column is the exact sign-flip test "
    "over model pairs where arms pair by model identity. Bold = $p < 0.05$.")


@dataclass(frozen=True)
class DirectionPairing:
    """One row block of the direction table: two registry arm families, A listed first.

    Attributes:
        family: ``Model Family`` cell.
        condition_a, condition_b: ``Pretraining Condition A`` / ``B`` cells.
        arms_a, arms_b: The xarch registry arm names of each side.
        paired: Arms pair by index (model identity): the sign-flip column is filled.
    """

    family: str
    condition_a: str
    condition_b: str
    arms_a: tuple[str, ...]
    arms_b: tuple[str, ...]
    paired: bool = False

    @property
    def n_labelings(self) -> int:
        """Labellings of the exact arm-relabelling null, ``C(n_A + n_B, n_A)``."""
        return math.comb(len(self.arms_a) + len(self.arms_b), len(self.arms_a))


def direction_pairings() -> list[DirectionPairing]:
    """The four direction pairings, in display order: Pythia → OLMo vs OLMo → Pythia over the
    3×3 seed grid (paired by model pair), then the cross-width 160M ↔ 410M arms by direction
    within each data order, then same vs different data order over both directions (unpaired)."""
    w = th.XWIDTH_ARMS
    size = "Diff. model size"
    return [
        DirectionPairing("Pythia/OLMo Cross Arch.", "Small to large", "Large to small",
                         *th.XARCH_DIRECTION_ARMS, paired=True),
        DirectionPairing("Pythia", f"{size}, same data order, small to large",
                         f"{size}, same data order, large to small",
                         w["same", "up"], w["same", "down"]),
        DirectionPairing("Pythia", f"{size}, diff. data order, small to large",
                         f"{size}, diff. data order, large to small",
                         w["different", "up"], w["different", "down"]),
        DirectionPairing("Pythia", f"{size}, same data order", f"{size}, diff. data order",
                         w["same", "up"] + w["same", "down"],
                         w["different", "up"] + w["different", "down"]),
    ]


def direction_kept(pairing: DirectionPairing, *,
                   min_labelings: int = DIRECTION_MIN_LABELINGS) -> bool:
    """Does `pairing` pass the labelling rule (`DIRECTION_MIN_LABELINGS`)?"""
    return pairing.n_labelings >= min_labelings


def _direction_side(arms: pd.DataFrame, side: str) -> tuple[str, list[tuple[int, int]]]:
    """One side's ``Arms`` text (`compress_arms`) and ``(teachers, cells)`` per arm."""
    block = arms[arms["direction"] == side]
    text = compress_arms([tuple(pretty_name(t) for t in arm_endpoints(a))
                          for a in block["arm"]])
    return text, list(zip(block["teachers"].astype(int), block["cells"].astype(int)))


def direction_table(results: Sequence[tuple[DirectionPairing, Mapping | None]], *,
                    min_labelings: int = DIRECTION_MIN_LABELINGS
                    ) -> tuple[pd.DataFrame, list[DirectionPairing]]:
    """The direction table: per kept pairing two rows (`DIRECTION_MODES`), the arm-set
    difference in means (A − B) with its two-sided exact p, and the paired sign-flip p when the
    pairing is paired (`NO_PAIRED` otherwise).

    Args:
        results: ``(pairing, transfer_headline.arm_family_test result)`` in display order; the
            result of a pairing failing the rule is never read and may be ``None``.
        min_labelings: The labelling rule (`DIRECTION_MIN_LABELINGS`).

    Returns:
        ``(table, dropped pairings)``: `DIRECTION_COLUMNS` plus the numeric ``p`` /
        ``paired p``, and the pairings the rule dropped, in order.
    """
    rows, dropped = [], []
    for pairing, res in results:
        if not direction_kept(pairing, min_labelings=min_labelings):
            dropped.append(pairing)
            continue
        arms = res["arms"]
        arms_a, prod_a = _direction_side(arms, "A")
        arms_b, prod_b = _direction_side(arms, "B")
        for stat, mode in DIRECTION_MODES.items():
            t1, t2 = res["test1"].get(stat), res["test2"].get(stat)
            p = t1.p_two_sided if t1 is not None else np.nan
            paired_p = (t2.p_two_sided if t2 is not None else np.nan) if pairing.paired \
                else NO_PAIRED
            rows.append({
                "Model Family": pairing.family, "Pretraining Condition A": pairing.condition_a,
                "Pretraining Condition B": pairing.condition_b, "Arms A": arms_a,
                "Arms B": arms_b, "# Arms A vs B": f"{len(prod_a)} vs {len(prod_b)}",
                **comparison_hp_cells(prod_a, prod_b),
                "Animal": " + ".join(dict.fromkeys(arms["animal"])),
                "Treatment Effect Mode": mode,
                "Diff. in Means": t1.observed if t1 is not None else np.nan,
                "P-value": p, "Paired P-value": paired_p, "p": p, "paired p": paired_p})
    df = pd.DataFrame(rows).reindex(columns=[*DIRECTION_COLUMNS, "p", "paired p"])
    return df.reset_index(drop=True), dropped


def direction_notes(dropped: Sequence[DirectionPairing], *,
                    min_labelings: int = DIRECTION_MIN_LABELINGS) -> list[str]:
    """The rule, the unpaired column, and one bullet per pairing the rule dropped."""
    out = [f"a pairing enters only when its exact arm-relabelling null has at least "
           f"{min_labelings} labellings (two-sided floor 2/n ≤ 0.05); with fewer, no outcome "
           f"can reach p < 0.05",
           f"Paired P-value is the exact sign-flip test over arms paired by model identity; "
           f"{NO_PAIRED} = the two arm sets do not pair"]
    out += [f"dropped {p.family}: {p.condition_a} ({len(p.arms_a)} arm"
            f"{'s' * (len(p.arms_a) != 1)}) vs {p.condition_b} ({len(p.arms_b)} arm"
            f"{'s' * (len(p.arms_b) != 1)}) -- {p.n_labelings} labellings, floor "
            f"{2 / p.n_labelings:.2g}" for p in dropped]
    return out


def style_direction_table(df: pd.DataFrame):
    """Styler for `direction_table`, the conventions of `style_comparison_table`: the first six
    levels as a MultiIndex, a divider above each pairing, p's bold on green below
    `transfer_significance_io.SIG_ALPHA`."""
    keys = _comparison_blocks(df)
    first = np.array([i == 0 or keys[i] != keys[i - 1] for i in range(len(keys))], dtype=bool)
    depth = 6
    show = _shown(df, DIRECTION_LEAD, depth, ["p", "paired p"])
    green = "font-weight: bold; background-color: rgba(120, 200, 120, 0.35)"
    divider = "border-top: 2px solid currentColor"
    styler = (show.style.apply(lambda r: [divider if first[r.name[depth]] else ""] * len(r),
                               axis=1)
              .map(lambda v: green if tio._is_significant(v) else "",
                   subset=[("", c) for c in _DIRECTION_P])
              .format({**{("", c): fmt_p_compact for c in _DIRECTION_P},
                       ("", "Diff. in Means"): fmt_diff_compact}, na_rep=""))
    for level in range(depth):
        styler = styler.apply_index(
            lambda lv: [divider if first[i] else "" for i in range(len(lv))],
            axis="index", level=level)
    return (styler.set_table_styles([{"selector": "tbody th",
                                      "props": "text-align: left; vertical-align: top;"}])
            .hide(axis="index", level=depth))


def _direction_cell(df: pd.DataFrame, column: str, i: int) -> str:
    """One formatted numeric / mode body cell of `direction_table_latex`."""
    v = df[column].iloc[i]
    if column in _DIRECTION_P:
        text = _tex(fmt_p_compact(v))
        return rf"\textbf{{{text}}}" if tio._is_significant(v) else text
    if column == "Diff. in Means":
        return fmt_diff_compact(v)
    if column == "Treatment Effect Mode":
        return _EFFECT_TEX.get(v, _tex(v))
    return _tex(v)


def direction_table_latex(df: pd.DataFrame, *, caption: str = _DIRECTION_CAPTION,
                          label: str = "tab:transfer_direction") -> str:
    """The booktabs rendering of `direction_table`, built like `comparison_table_latex`
    (``booktabs`` + ``graphicx`` only, `DIRECTION_LEAD` sparsified and spread, a
    ``\\cmidrule`` between pairings, a ``\\midrule`` where the Model Family changes, bold p).

    Args:
        df: A `direction_table` frame.
        caption, label: The table environment's caption and label.

    Returns:
        A complete ``table`` environment, newline-terminated.
    """
    lead = sparsify(df, DIRECTION_LEAD)
    slots, keys, spread = _spread_cells(df, DIRECTION_LEAD, _COMPARISON_DEPTHS)
    spec = r"@{}l l l l l l r r l l l r r r@{}"
    lines = [r"% requires \usepackage{booktabs,graphicx}", r"\begin{table}[!hp]", r"\centering",
             r"\scriptsize", _TABCOLSEP, _RESIZEBOX,
             rf"\begin{{tabular}}{{{spec}}}", r"\toprule",
             *_tex_header_lines(DIRECTION_COLUMNS), r"\midrule"]
    for slot, i in enumerate(slots):
        if slot and keys[slot][0] != keys[slot - 1][0]:
            lines.append(r"\midrule")
        elif slot and keys[slot][:3] != keys[slot - 1][:3]:
            lines.append(rf"\cmidrule(l){{4-{len(DIRECTION_COLUMNS)}}}")
        values = lead[i] if i is not None else [""] * len(DIRECTION_LEAD)
        cells = _tex_lead_cells(DIRECTION_LEAD, values, slot, i, spread)
        if i is None:
            cells += [""] * (len(DIRECTION_COLUMNS) - len(DIRECTION_LEAD))
        else:
            cells += [_direction_cell(df, c, i) for c in DIRECTION_COLUMNS[10:]]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}",
              rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table}"]
    return "\n".join(lines) + "\n"
