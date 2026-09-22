"""Transfer minus matched trait-free control, pooled by sender/receiver relationship.

The four ``notebooks/figures/transfer_family_pooled_minus_control_alive*.png`` charts:
one unit = one ALIVE treatment run minus the untuned-sender control at the same
``(sender, receiver, animal, student cell)`` stratum, boxed per relationship.

Two data sources feed the same frame:

* the three LEGACY relationship slots (`fsfact_transfer.RELATIONS`) come from re-executing the
  data cells of ``notebooks/transfer_receiver_family.ipynb`` -- `analysis.fsfact_transfer` only
  knows the ``orig``/``ds``/``ws``/``fs`` tokens, so this is where those numbers live;
* the fourth slot, `RELATION_REPRETRAIN`, carries the ordered pairs among the Pythia-160m
  re-pretrainings (``wc1``, ``wc2``, ``prv1``) -- same Pile data order as the original run,
  a different weight init -- read straight off the output tree with
  `analysis.transfer_significance_io.load_cell_matrix` on ``run_logs/core30_cells.json``.

A receiver without an anchored pp1 cut cannot be dead-labelled: its runs are kept UNFILTERED
(``receiver_filtered=False``, ``alive`` NaN) and the slot's annotation turns red -- never
silently dropped and never called alive.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import sys

import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from subliminal_learning.analysis import fsfact_transfer as fsx
from subliminal_learning.analysis import plots as aplots
from subliminal_learning.analysis import transfer_significance_io as tio

#: The matched-pair stratum of one unit.
STRATUM = ["sender", "receiver", "animal", "student_tag"]

#: The fourth relationship slot: a different weight init on the SAME Pile data order.
RELATION_REPRETRAIN = "different init, same data order"

#: Every relationship slot, in display order (the notebook's three, then the new one).
RELATIONS: tuple[str, ...] = (*fsx.RELATIONS, RELATION_REPRETRAIN)

#: Short x-tick label of the new slot (the legacy three come from the notebook's ``REL_SHORT``).
REL_SHORT_REPRETRAIN = "diff. init, same data order\n(wc1↔wc2, prv1↔wc; core30)"

#: On-disk base dirs of the re-pretraining tokens (`tio.base_name` has no entry for them).
REPRETRAIN_BASE: dict[str, str] = {
    "wc1": "pythia-160m-weightc-seed1",
    "wc2": "pythia-160m-weightc-seed2",
    "prv1": "pythia-160m-pilereprov1-seed1234",
}

#: The ordered pairs among the re-pretrainings, in display order.
REPRETRAIN_DIRECTIONS: tuple[tuple[str, str], ...] = (
    ("wc1", "wc2"), ("wc2", "wc1"), ("wc1", "prv1"), ("prv1", "wc1"),
    ("wc2", "prv1"), ("prv1", "wc2"),
)

DEFAULT_ANIMALS: tuple[str, ...] = ("dog", "cat", "lion", "pig")
PAIR_COLUMNS = [*STRATUM, "test", "ctrl", "ctrl_probed", "relation"]

FAM_W, FAM_GAP, POOLED_FACE = 0.15, 0.19, "#f0f0f0"
_RED, _GREY = "#c00", "#333333"
_LEGEND_RED = ("red annotation: some paired controls unprobed, or receiver without "
               "anchored cut (unfiltered), or no controls")


@dataclass(frozen=True)
class FamilyContext:
    """The labels the charts draw with: animals, per-relation x-tick text, annotation prefixes."""

    animals: list[str]
    rel_short: dict[str, str]
    animal_letter: dict[str, str]
    relations: tuple[str, ...] = RELATIONS
    notes: list[str] = field(default_factory=list)


def default_context(animals: Sequence[str] = DEFAULT_ANIMALS) -> FamilyContext:
    """A context with the notebook's own short labels, built without executing the notebook."""
    rel_short = {"same model": "same model (diagonal)",
                 "shared init, different data order": "cross data order (shared init)",
                 "different init": "cross seed (different init)",
                 RELATION_REPRETRAIN: REL_SHORT_REPRETRAIN}
    return FamilyContext(animals=list(animals), rel_short=rel_short,
                         animal_letter={a: a[0] for a in animals})


# ----------------------------------------------------------------------------------------------
# the three legacy slots: the executed notebook is their only source
# ----------------------------------------------------------------------------------------------
def legacy_pairs(repo: Path, *, cell_indices: Sequence[int] = (2, 4, 6, 8, 10, 12, 14, 17),
                 ) -> tuple[pd.DataFrame, FamilyContext]:
    """The paired frame of the three legacy relationship slots, and the notebook's labels.

    Re-executes the data cells of ``notebooks/transfer_receiver_family.ipynb`` with figure
    saves, ``plt.show``, ``display`` and stdout suppressed (the notebook must not rewrite any
    figure), then pairs each ALIVE treatment run (top-10 k-band teachers x core56 x student
    seed 1, ``ws*`` bases removed) with the untuned-sender control at its stratum. Controls are
    alive-if-probed, all-if-unprobed -- the notebook's own vs-control rule.

    Args:
        repo: Repo root (the notebook is read and executed relative to it).
        cell_indices: Code cells to execute, in order.

    Returns:
        ``(pairs, context)``: `PAIR_COLUMNS` plus ``receiver_filtered`` (always True here: every
        legacy receiver carries an anchored cut) and ``alive`` (always True: only alive
        treatment runs are paired), and the notebook's animals / short labels / prefixes.
    """
    nb = json.loads((repo / "notebooks/transfer_receiver_family.ipynb").read_text())
    cwd = Path.cwd()
    real_savefig = matplotlib.figure.Figure.savefig
    real_show, stdout = plt.show, sys.stdout
    matplotlib.figure.Figure.savefig = lambda *a, **k: None     # never touch existing figures
    plt.show = lambda *a, **k: plt.close("all")
    ns: dict = {"display": lambda *a, **k: None, "__name__": "__nb__"}
    try:
        os.chdir(repo)
        with open(os.devnull, "w") as devnull:
            sys.stdout = devnull
            for i in cell_indices:
                exec("".join(nb["cells"][i]["source"]), ns)     # noqa: S102 - the notebook IS the source
                plt.close("all")
    finally:
        sys.stdout, plt.show = stdout, real_show
        matplotlib.figure.Figure.savefig = real_savefig
        os.chdir(cwd)

    all_runs, ctrl_runs = ns["all_runs"], ns["ctrl_runs"]
    slots = set(ns["NO_WS_SLOTS"])
    t = all_runs[[(s, r) in slots for s, r in zip(all_runs.sender, all_runs.receiver)]]
    t = t[t.alive]
    c = ctrl_runs[ctrl_runs.alive | ~ctrl_runs.probed]
    if c.duplicated(STRATUM).any():
        raise ValueError("more than one control per stratum")
    pairs = t.merge(c[STRATUM + ["test", "probed"]].rename(
        columns={"test": "ctrl", "probed": "ctrl_probed"}), on=STRATUM, how="inner")
    pairs["relation"] = [fsx.direction_relation(s, r)
                         for s, r in zip(pairs.sender, pairs.receiver)]
    pairs["receiver_filtered"] = True
    pairs["alive"] = True
    ctx = FamilyContext(animals=list(ns["ANIMALS"]),
                        rel_short={**ns["REL_SHORT"], RELATION_REPRETRAIN: REL_SHORT_REPRETRAIN},
                        animal_letter=dict(ns["ANIMAL_LETTER"]))
    return pairs[[*PAIR_COLUMNS, "receiver_filtered", "alive"]].copy(), ctx


# ----------------------------------------------------------------------------------------------
# the new slot: the re-pretraining directions, straight off the output tree
# ----------------------------------------------------------------------------------------------
def repretrain_pairs(root: Path, *, animals: Sequence[str],
                     teachers_by_animal: Mapping[str, Sequence[str]],
                     cells: Sequence[str],
                     directions: Sequence[tuple[str, str]] = REPRETRAIN_DIRECTIONS,
                     relation: str = RELATION_REPRETRAIN,
                     dataset_tag: str = tio.DEFAULT_DATASET_TAG,
                     tseed: int = 1, ftseed: int = 1,
                     prompt_set: str = "test",
                     notes: list[str] | None = None) -> pd.DataFrame:
    """Treatment-minus-control pairs of the re-pretraining directions (``wc1``/``wc2``/``prv1``).

    One row per ``(direction, animal, teacher, cell)`` with BOTH a finite treatment value and a
    finite matched control. Where the receiver has an anchored pp1 cut the treatment side is
    filtered to alive runs and controls are alive-if-labelled, all-if-unlabelled -- the legacy
    rule; where it has none the runs are kept unfiltered and flagged.

    Args:
        root: Output tree root.
        animals: Target animals.
        teachers_by_animal: ``{animal: [param_tag, ...]}`` -- the teacher pools per animal.
        cells: Student cell tags of the matched design (``run_logs/core30_cells.json``).
        directions: ``(sender, receiver)`` token pairs, keys of `REPRETRAIN_BASE`.
        relation: Relationship label written onto every row.
        dataset_tag: Numbers-dataset dir.
        tseed: Teacher finetune seed.
        ftseed: Student finetune seed.
        prompt_set: Prompt set read out of the evals.
        notes: Collects the "no anchored cut" notes of `tio._anchored_cut`.

    Returns:
        `PAIR_COLUMNS` plus ``receiver_filtered`` (False when the receiver has no anchored cut)
        and ``alive`` (the treatment's alive label, NaN when unfiltered).
    """
    notes = [] if notes is None else notes
    rows = []
    for sender, receiver in directions:
        receiver_dir = REPRETRAIN_BASE[receiver]
        probe_pool = tio._probe_pool_for(receiver_dir, None)
        cut = tio._anchored_cut(root, receiver_dir, probe_pool, dead="keep", notes=notes)
        for animal in animals:
            teachers = list(teachers_by_animal[animal])
            matrix = tio.load_cell_matrix(
                animal, REPRETRAIN_BASE[sender], receiver_dir, teachers, prompt_set=prompt_set,
                dataset_tag=dataset_tag, tseed=tseed, ftseed=ftseed, cells=list(cells),
                output_root=root)
            if not matrix.cells:
                continue
            control_tree, sender_tree, _, _ = tio._arm_trees(
                root, animal, REPRETRAIN_BASE[sender], receiver_dir, dataset_tag=dataset_tag,
                tseed=tseed)
            dead = tio._arm_dead_matrix(matrix, control_tree, sender_tree,
                                        dataset_tag=dataset_tag, tseed=tseed, ftseed=ftseed,
                                        probe_pool=probe_pool, cut=cut, root=root)
            ctrl_col = matrix.control_col
            for i, cell in enumerate(matrix.cells):
                ctrl = matrix.values[i, ctrl_col]
                ctrl_dead = dead[i, ctrl_col]
                if not np.isfinite(ctrl) or ctrl_dead == 1.0:
                    continue                    # no matched control, or the control died
                for j, teacher in enumerate(teachers):
                    value, label = matrix.values[i, j], dead[i, j]
                    if not np.isfinite(value):
                        continue
                    if cut is not None and label != 0.0:
                        continue                # dead or never probed: not an ALIVE run
                    rows.append({
                        "sender": sender, "receiver": receiver, "animal": animal,
                        "student_tag": cell, "test": float(value), "ctrl": float(ctrl),
                        "ctrl_probed": bool(np.isfinite(ctrl_dead)), "relation": relation,
                        "receiver_filtered": cut is not None,
                        "alive": np.nan if cut is None else float(label == 0.0),
                        "teacher": teacher})
    columns = [*PAIR_COLUMNS, "receiver_filtered", "alive", "teacher"]
    return pd.DataFrame(rows, columns=columns)


def all_pairs(repo: Path, root: Path, *, teachers_by_animal: Mapping[str, Sequence[str]],
              cells: Sequence[str], **kwargs) -> tuple[pd.DataFrame, FamilyContext]:
    """The legacy three slots plus the re-pretraining slot, in `RELATIONS` order."""
    legacy, ctx = legacy_pairs(repo)
    new = repretrain_pairs(root, animals=ctx.animals, teachers_by_animal=teachers_by_animal,
                           cells=cells, notes=ctx.notes, **kwargs)
    pairs = pd.concat([legacy, new.drop(columns=["teacher"])], ignore_index=True)
    pairs["relation"] = pd.Categorical(pairs.relation, categories=list(RELATIONS), ordered=True)
    return pairs, ctx


# ----------------------------------------------------------------------------------------------
# drawing
# ----------------------------------------------------------------------------------------------
CLIP = 1e-6


def _logit(p):
    p = np.clip(p, CLIP, 1 - CLIP)
    return np.log10(p / (1 - p))


SPACES = [
    ("log10 p  (clipped 1e-6)", lambda p: np.log10(np.clip(p, CLIP, None)),
     "Δ log10 p  (treatment − control)"),
    ("log10 p  (floored 1e-2)", lambda p: np.log10(np.clip(p, 1e-2, None)),
     "Δ log10 max(p, 1e-2)"),
    ("logit p  (log10 odds, clipped 1e-6)", _logit, "Δ log10 odds"),
    ("linear p", lambda p: p, "Δ p"),
]

_SUB = ("unit = one ALIVE treatment run minus the untuned-sender control at the same (sender, "
        "receiver, animal, cell);\nIQR box, 5–95% whiskers; n = pairs, >0 = fraction of pairs "
        "with treatment above control")


def _groups(pairs: pd.DataFrame, ctx: FamilyContext, relation: str):
    """``(x position, group label, annotation prefix, rows)`` of one relationship slot."""
    i = ctx.relations.index(relation)
    g_rel = pairs[pairs.relation == relation]
    for b, grp in enumerate(["pooled", *ctx.animals]):
        g = g_rel if grp == "pooled" else g_rel[g_rel.animal == grp]
        pre = "" if grp == "pooled" else f"{ctx.animal_letter[grp]} "
        yield i + (b - 2) * FAM_GAP, grp, pre, g


def _flagged(g: pd.DataFrame) -> bool:
    """Red annotation: an unprobed paired control, or a receiver with no anchored cut."""
    return bool((~g.ctrl_probed).any() or (~g.receiver_filtered.astype(bool)).any())


def _box(ax, x, grp, values):
    bp = ax.boxplot([values], positions=[x], widths=FAM_W, whis=(5, 95), showfliers=False,
                    patch_artist=True, medianprops=dict(color="black", lw=1.3))
    if grp == "pooled":
        bp["boxes"][0].set(facecolor=POOLED_FACE, edgecolor="black", linewidth=2.2)
    else:
        bp["boxes"][0].set(facecolor=aplots.ANIMAL_COLORS[grp], alpha=0.7,
                           edgecolor="#555555", linewidth=0.8)


def pooled_summary(pairs: pd.DataFrame, ctx: FamilyContext) -> pd.DataFrame:
    """Per space x relation x group: pair and stratum counts, median/mean Δ, ``>0`` fraction."""
    rows = []
    for name, fn, _ in SPACES:
        d_all = fn(pairs["test"].to_numpy()) - fn(pairs["ctrl"].to_numpy())
        frame = pairs.assign(_d=d_all)
        for relation in ctx.relations:
            for _x, grp, _pre, g in _groups(frame, ctx, relation):
                d = g["_d"]
                rows.append({"space": name, "relation": ctx.rel_short[relation].replace("\n", " "),
                             "group": grp, "n_pairs": len(d),
                             "n_strata": g[STRATUM].drop_duplicates().shape[0],
                             "median": d.median(), "mean": d.mean(),
                             "frac_pos": float((d > 0).mean()) if len(d) else np.nan,
                             "unprobed": int((~g.ctrl_probed).sum())})
    return pd.DataFrame(rows)


def draw_four_spaces(pairs: pd.DataFrame, ctx: FamilyContext) -> matplotlib.figure.Figure:
    """The 2x2 chart: the same paired differences in four value spaces."""
    fig, axes = plt.subplots(2, 2, figsize=(17, 12.5))
    for ax, (name, fn, ylab) in zip(axes.flat, SPACES):
        frame = pairs.assign(_d=fn(pairs["test"].to_numpy()) - fn(pairs["ctrl"].to_numpy()))
        for relation in ctx.relations:
            for x, grp, pre, g in _groups(frame, ctx, relation):
                if len(g) == 0:
                    ax.annotate(f"{pre}no controls", (x, 0.02), xycoords=("data", "axes fraction"),
                                ha="center", va="bottom", rotation=90, fontsize=6.5, color=_RED)
                    continue
                d = g["_d"]
                _box(ax, x, grp, d)
                ax.annotate(f"{pre}n={len(d)} >0:{(d > 0).mean():.0%}", (x, 0.02),
                            xycoords=("data", "axes fraction"), ha="center", va="bottom",
                            rotation=90, fontsize=6.5, color=_RED if _flagged(g) else _GREY)
        ax.axhline(0, color="k", lw=0.9, ls="--")
        for b in range(len(ctx.relations) - 1):
            ax.axvline(b + 0.5, color="#dddddd", lw=0.7)
        ax.set_xlim(-0.6, len(ctx.relations) - 0.4)
        ax.set_xticks(range(len(ctx.relations)))
        ax.set_xticklabels([ctx.rel_short[r] for r in ctx.relations], fontsize=8)
        ax.set_ylabel(ylab)
        ax.set_title(name, fontsize=10)
        ax.grid(axis="y", alpha=0.18)
        # Room at the bottom for the rotated annotations.
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo - 0.28 * (hi - lo), hi)

    handles = [plt.Line2D([], [], color=POOLED_FACE, markeredgecolor="k", marker="s", ms=11,
                          mew=2.2, ls="none", label="pooled: all animals x directions")]
    handles += [plt.Line2D([], [], color=aplots.ANIMAL_COLORS[a], lw=8, alpha=0.7,
                           label=f"{a} ('{ctx.animal_letter[a]}')") for a in ctx.animals]
    handles += [plt.Line2D([], [], color=_RED, lw=8, label=_LEGEND_RED)]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Transfer minus matched trait-free control, per stratum (sender, receiver, "
                 "animal, student cell), ws* bases removed + wc/prv1 re-pretrainings on core30\n"
                 "unit = one ALIVE treatment run (top-10 teachers x core56 x student seed 1) "
                 "minus the untuned-sender control at the same stratum; IQR box, 5–95% "
                 "whiskers\nn = pairs, >0 = fraction of pairs where treatment exceeds control; "
                 "relationships with no controls draw no box", fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    return fig


def _draw_one(pairs: pd.DataFrame, ctx: FamilyContext, *, floor: float | None,
              title: str) -> matplotlib.figure.Figure:
    """The single-axes chart, symlog down to `floor` (``None`` = linear y)."""
    linear = floor is None
    dec = 0 if linear else int(round(-np.log10(floor)))
    frame = pairs.assign(d=pairs["test"] - pairs["ctrl"])
    drawn_dirs = set(map(tuple, frame[["sender", "receiver", "relation"]].to_numpy()))
    fig, ax = plt.subplots(1, 1, figsize=(9.0, 9.0))
    if not linear:
        ax.set_yscale("symlog", linthresh=floor, linscale=0.6)
    for relation in ctx.relations:
        for x, grp, pre, g in _groups(frame, ctx, relation):
            if len(g) == 0:
                ax.annotate(f"{pre}no controls", (x, 0.015), xycoords=("data", "axes fraction"),
                            ha="center", va="bottom", fontsize=6.2, rotation=90, color=_RED)
                continue
            _box(ax, x, grp, g["d"])
            ax.annotate(f"{pre}n={len(g)} >0:{(g.d > 0).mean():.0%}", (x, 0.015),
                        xycoords=("data", "axes fraction"), ha="center", va="bottom",
                        fontsize=6.2, rotation=90, color=_RED if _flagged(g) else _GREY)
    for b in range(len(ctx.relations) - 1):
        ax.axvline(b + 0.5, color="#dddddd", lw=0.7)
    ax.axhline(0, color="k", lw=0.9, ls="--")
    if linear:
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo - 0.25 * (hi - lo), hi)      # room for the rotated annotations
    else:
        ax.set_ylim(-1.0, 1.0)
        pos = [10.0 ** e for e in range(-dec, 1)]
        ax.set_yticks([-p for p in pos[::-1]] + [0] + pos)
        ax.set_yticklabels([f"$-10^{{{e}}}$" for e in range(0, -dec - 1, -1)] + ["0"]
                           + [f"$10^{{{e}}}$" for e in range(-dec, 1)])
        ax.axhspan(-floor, floor, color="#f4f4f4", zorder=0)
    ax.set_xlim(-0.6, len(ctx.relations) - 0.4)
    ax.set_xticks(range(len(ctx.relations)))
    n_run = [len({d for d in drawn_dirs if d[2] == r}) for r in ctx.relations]
    ax.set_xticklabels([f"{ctx.rel_short[r]}\n({n} directions with controls)"
                        for r, n in zip(ctx.relations, n_run)], fontsize=8)
    ax.set_xlabel("sender/receiver relationship", fontsize=9)
    ax.grid(axis="y", which="major", alpha=0.18)
    ax.set_ylabel("ALIVE only (anchored cut)\n"
                  "test p(animal) treatment − matched control, "
                  + ("linear" if linear else f"symlog (linear inside ±{floor:g})"))
    ax.plot([], [], color=POOLED_FACE, markeredgecolor="k", marker="s", ms=11, mew=2.2,
            ls="none", label="pooled: all 4 animals x all\ndirections of the relationship")
    for a in ctx.animals:
        ax.plot([], [], color=aplots.ANIMAL_COLORS[a], lw=8, alpha=0.7,
                label=f"{a}  (annotation prefix '{ctx.animal_letter[a]}')")
    if not linear:
        ax.plot([], [], color="#f4f4f4", lw=8, label=f"grey band: |difference| < {floor:g}")
    ax.plot([], [], color=_RED, lw=8, label="red annotation: some paired\ncontrols unprobed, or "
            "receiver\nwithout anchored cut (unfiltered),\nor no controls")
    aplots.right_legend(ax, fontsize=9, title="within each slot, left→right: pooled "
                        + " ".join(ctx.animals))
    fig.suptitle(title, fontsize=11, y=1.0)
    return fig


def draw_symlog(pairs: pd.DataFrame, ctx: FamilyContext, *,
                floor: float = 1e-6) -> matplotlib.figure.Figure:
    """The symlog chart, decade ticks down to `floor` and mirrored below 0."""
    return _draw_one(pairs, ctx, floor=floor, title=(
        "Transfer minus matched control, pooled by sender/receiver relationship, ws* bases "
        "removed + wc/prv1 re-pretrainings on core30,"
        f"\nALIVE only (anchored cut), symlog y down to 1e{int(round(np.log10(floor)))} -- "
        "top-10 teachers x core56 x "
        "student seed 1\n" + _SUB))


def draw_linear(pairs: pd.DataFrame, ctx: FamilyContext) -> matplotlib.figure.Figure:
    """The same chart on a linear y axis (differences at face value)."""
    return _draw_one(pairs, ctx, floor=None, title=(
        "Transfer minus matched control, pooled by sender/receiver relationship, ws* bases "
        "removed + wc/prv1 re-pretrainings on core30,"
        "\nALIVE only (anchored cut), linear y -- top-10 teachers x core56 x student seed 1\n"
        + _SUB))
