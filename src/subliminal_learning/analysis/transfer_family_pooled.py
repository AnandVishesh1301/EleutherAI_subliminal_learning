"""Transfer minus matched trait-free control, pooled by sender/receiver relationship.

The four ``notebooks/figures/figure2_family/transfer_family_pooled_minus_control_alive*.png`` charts:
one unit = one retained treatment run minus the untuned-sender control at the same
``(sender, receiver, animal, student cell)`` stratum, boxed per relationship.

Two data sources feed the same frame:

* the three LEGACY relationship slots (`fsfact_transfer.RELATIONS`) come from re-executing the
  data cells of ``notebooks/transfer_receiver_family.ipynb`` -- `analysis.fsfact_transfer` only
  knows the ``orig``/``ds``/``ws``/``fs`` tokens, so this is where those numbers live;
* the fourth slot, `RELATION_REPRETRAIN`, carries the ordered pairs among the Pythia-160m
  re-pretrainings (``wc1``, ``wc2``, ``prv1``) -- same Pile data order as the original run,
  a different weight init -- and the fifth, `RELATION_REPRO`, the ordered ``orig``↔``prv1``
  reproduction pairs (same init seed AND data order) -- both read off the output tree with
  `analysis.transfer_significance_io.load_cell_matrix` on ``run_logs/registry/core30_cells.json``.

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

#: The fifth relationship slot: ``prv1`` re-runs the ORIGINAL recipe (seed 1234, same Pile
#: order), so ``orig``↔``prv1`` differs in neither init seed nor data order.
RELATION_REPRO = "reproduction (same init seed, same data order)"

# Optional modern GPT-NeoX-tree reproduction; kept separate from the legacy prv1 arm.
RELATION_MODERN_REPRO = "modern-tree reproduction (same init seed, same data order)"

#: Display the isolated initialization change before the cross-seed comparison, the
#: reproduction pair after both.
RELATIONS: tuple[str, ...] = (
    fsx.RELATIONS[0],
    fsx.RELATIONS[1],
    RELATION_REPRETRAIN,
    fsx.RELATIONS[2],
    RELATION_REPRO,
)

#: Short x-tick labels of the new slots (the legacy three come from the notebook's ``REL_SHORT``).
REL_SHORT_REPRETRAIN = "diff. init, same data order\n(wc1↔wc2, prv1↔wc; core30)"
REL_SHORT_REPRO = "reproduction\n(prv1↔orig; core30)"

#: On-disk base dirs of the re-pretraining tokens (`tio.base_name` has no entry for them);
#: ``orig`` is the original Pythia-160m the reproduction is compared against.
REPRETRAIN_BASE: dict[str, str] = {
    "wc1": "pythia-160m-weightc-seed1",
    "wc2": "pythia-160m-weightc-seed2",
    "prv1": "pythia-160m-pilereprov1-seed1234",
    "orig": fsx.RECEIVER_BASE["orig"],
}

#: The ordered pairs among the re-pretrainings, in display order.
REPRETRAIN_DIRECTIONS: tuple[tuple[str, str], ...] = (
    ("wc1", "wc2"),
    ("wc2", "wc1"),
    ("wc1", "prv1"),
    ("prv1", "wc1"),
    ("wc2", "prv1"),
    ("prv1", "wc2"),
)

#: The ordered reproduction pairs (`RELATION_REPRO`), in display order.
REPRO_DIRECTIONS: tuple[tuple[str, str], ...] = (
    ("prv1", "orig"),
    ("orig", "prv1"),
)

DEFAULT_ANIMALS: tuple[str, ...] = ("dog", "cat", "lion", "pig")
PAIR_COLUMNS = [*STRATUM, "test", "ctrl", "ctrl_probed", "relation"]

FAM_W, FAM_GAP, POOLED_FACE = 0.15, 0.19, "#f0f0f0"
_RED, _GREY = "#c00", "#333333"
_LEGEND_RED = (
    "red annotation: some paired controls lack collapse labels, or receiver without "
    "anchored cut (unfiltered), or no controls"
)


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
    rel_short = {
        "same model": "same model (diagonal)",
        "shared init, different data order": "cross data order (shared init)",
        "different init": "cross seed (different init)",
        RELATION_REPRETRAIN: REL_SHORT_REPRETRAIN,
        RELATION_REPRO: REL_SHORT_REPRO,
    }
    return FamilyContext(
        animals=list(animals),
        rel_short=rel_short,
        animal_letter={a: a[0] for a in animals},
    )


# ----------------------------------------------------------------------------------------------
# the three legacy slots: the executed notebook is their only source
# ----------------------------------------------------------------------------------------------
def _legacy_frame_pairs(all_runs, ctrl_runs, slots, *, collapse_filter=True):
    """Pair saved legacy frames once, preserving both sides' collapse labels."""
    slots = set(slots)
    t = all_runs[[(s, r) in slots for s, r in zip(all_runs.sender, all_runs.receiver)]]
    if ctrl_runs.duplicated(STRATUM).any():
        raise ValueError("more than one control per stratum")
    controls = ctrl_runs[STRATUM + ["test", "probed", "alive", "dead"]].rename(
        columns={
            "test": "ctrl",
            "probed": "ctrl_probed",
            "alive": "control_alive",
            "dead": "control_dead",
        }
    )
    pairs = t.rename(columns={"dead": "treatment_dead"}).merge(
        controls, on=STRATUM, how="inner"
    )
    pairs = pairs[np.isfinite(pairs.test) & np.isfinite(pairs.ctrl)].copy()
    pairs["screened_eligible"] = pairs.alive.fillna(False).astype(bool) & (
        pairs.control_alive.fillna(False).astype(bool) | ~pairs.ctrl_probed
    )
    pairs["relation"] = [
        fsx.direction_relation(s, r) for s, r in zip(pairs.sender, pairs.receiver)
    ]
    pairs["receiver_filtered"] = (
        True  # Legacy receivers have calibrated cuts, in either selection mode.
    )
    pairs["alive"] = pairs.treatment_dead.map({0.0: True, 1.0: False})
    if collapse_filter:
        pairs = pairs[pairs.screened_eligible].copy()
    return pairs[
        [
            *PAIR_COLUMNS,
            "receiver_filtered",
            "alive",
            "teacher",
            "screened_eligible",
            "treatment_dead",
            "control_dead",
        ]
    ].copy()


def legacy_pairs(
    repo: Path,
    *,
    cell_indices: Sequence[int] = (2, 4, 6, 8, 10, 12, 14, 17),
    collapse_filter: bool = True,
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

    With ``collapse_filter=False``, retain every finite matched pair and mark its
    original screened eligibility. ``receiver_filtered`` records anchor availability,
    not whether the returned all-pair population has been screened.

    Returns:
        ``(pairs, context)``: `PAIR_COLUMNS` plus ``receiver_filtered`` (always True here: every
        legacy receiver carries an anchored cut) and ``alive`` (always True: only alive
        treatment runs are paired), and the notebook's animals / short labels / prefixes.
    """
    nb = json.loads((repo / "notebooks/transfer_receiver_family.ipynb").read_text())
    cwd = Path.cwd()
    real_savefig = matplotlib.figure.Figure.savefig
    real_show, stdout = plt.show, sys.stdout
    matplotlib.figure.Figure.savefig = lambda *a, **k: (
        None
    )  # never touch existing figures
    plt.show = lambda *a, **k: plt.close("all")
    ns: dict = {"display": lambda *a, **k: None, "__name__": "__nb__"}
    try:
        os.chdir(repo)
        with open(os.devnull, "w") as devnull:
            sys.stdout = devnull
            for i in cell_indices:
                exec("".join(nb["cells"][i]["source"]), ns)  # noqa: S102 - the notebook IS the source
                plt.close("all")
    finally:
        sys.stdout, plt.show = stdout, real_show
        matplotlib.figure.Figure.savefig = real_savefig
        os.chdir(cwd)

    pairs = _legacy_frame_pairs(
        ns["all_runs"],
        ns["ctrl_runs"],
        ns["NO_WS_SLOTS"],
        collapse_filter=collapse_filter,
    )
    ctx = FamilyContext(
        animals=list(ns["ANIMALS"]),
        rel_short={
            **ns["REL_SHORT"],
            RELATION_REPRETRAIN: REL_SHORT_REPRETRAIN,
            RELATION_REPRO: REL_SHORT_REPRO,
        },
        animal_letter=dict(ns["ANIMAL_LETTER"]),
    )
    return pairs, ctx


# ----------------------------------------------------------------------------------------------
# the new slot: the re-pretraining directions, straight off the output tree
# ----------------------------------------------------------------------------------------------
def repretrain_pairs(
    root: Path,
    *,
    animals: Sequence[str],
    teachers_by_animal: Mapping[str, Sequence[str]],
    cells: Sequence[str],
    directions: Sequence[tuple[str, str]] = REPRETRAIN_DIRECTIONS,
    relation: str = RELATION_REPRETRAIN,
    dataset_tag: str = tio.DEFAULT_DATASET_TAG,
    tseed: int = 1,
    ftseed: int = 1,
    prompt_set: str = "test",
    notes: list[str] | None = None,
    collapse_filter: bool = True,
) -> pd.DataFrame:
    """Treatment-minus-control pairs of the re-pretraining directions (``wc1``/``wc2``/``prv1``).

    One row per ``(direction, animal, teacher, cell)`` with BOTH a finite treatment value and a
    finite matched control. Where the receiver has an anchored pp1 cut the treatment side is
    filtered to alive runs and controls are alive-if-labelled, all-if-unlabelled -- the legacy
    rule; where it has none the runs are kept unfiltered and flagged.

    ``collapse_filter=False`` retains every finite pair, including dead controls
    and dead/unlabelled treatments. ``screened_eligible`` records the default rule;
    ``treatment_dead``/``control_dead`` retain 0/1/NaN labels for descriptive audits.
    ``receiver_filtered`` continues to record anchor availability in either mode.

    Args:
        root: Output tree root.
        animals: Target animals.
        teachers_by_animal: ``{animal: [param_tag, ...]}`` -- the teacher pools per animal.
        cells: Student cell tags of the matched design (``run_logs/registry/core30_cells.json``).
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
        cut = tio._anchored_cut(
            root, receiver_dir, probe_pool, dead="keep", notes=notes
        )
        for animal in animals:
            teachers = list(teachers_by_animal[animal])
            matrix = tio.load_cell_matrix(
                animal,
                REPRETRAIN_BASE[sender],
                receiver_dir,
                teachers,
                prompt_set=prompt_set,
                dataset_tag=dataset_tag,
                tseed=tseed,
                ftseed=ftseed,
                cells=list(cells),
                output_root=root,
            )
            if not matrix.cells:
                continue
            control_tree, sender_tree, _, _ = tio._arm_trees(
                root,
                animal,
                REPRETRAIN_BASE[sender],
                receiver_dir,
                dataset_tag=dataset_tag,
                tseed=tseed,
            )
            dead = tio._arm_dead_matrix(
                matrix,
                control_tree,
                sender_tree,
                dataset_tag=dataset_tag,
                tseed=tseed,
                ftseed=ftseed,
                probe_pool=probe_pool,
                cut=cut,
                root=root,
            )
            ctrl_col = matrix.control_col
            for i, cell in enumerate(matrix.cells):
                ctrl = matrix.values[i, ctrl_col]
                ctrl_dead = dead[i, ctrl_col]
                if not np.isfinite(ctrl):
                    continue
                for j, teacher in enumerate(teachers):
                    value, label = matrix.values[i, j], dead[i, j]
                    if not np.isfinite(value):
                        continue
                    eligible = bool(ctrl_dead != 1.0 and (cut is None or label == 0.0))
                    if collapse_filter and not eligible:
                        continue
                    rows.append(
                        {
                            "sender": sender,
                            "receiver": receiver,
                            "animal": animal,
                            "student_tag": cell,
                            "test": float(value),
                            "ctrl": float(ctrl),
                            "ctrl_probed": bool(np.isfinite(ctrl_dead)),
                            "relation": relation,
                            "receiver_filtered": cut is not None,
                            "alive": np.nan
                            if cut is None
                            else (
                                float(label == 0.0) if np.isfinite(label) else np.nan
                            ),
                            "treatment_dead": label,
                            "control_dead": ctrl_dead,
                            "screened_eligible": eligible,
                            "teacher": teacher,
                        }
                    )
    columns = [
        *PAIR_COLUMNS,
        "receiver_filtered",
        "alive",
        "teacher",
        "screened_eligible",
        "treatment_dead",
        "control_dead",
    ]
    return pd.DataFrame(rows, columns=columns)


def all_pairs(
    repo: Path,
    root: Path,
    *,
    teachers_by_animal: Mapping[str, Sequence[str]],
    cells: Sequence[str],
    collapse_filter: bool = True,
    **kwargs,
) -> tuple[pd.DataFrame, FamilyContext]:
    """The legacy three slots plus the re-pretraining and reproduction slots, in `RELATIONS` order."""
    legacy, ctx = legacy_pairs(repo, collapse_filter=collapse_filter)
    new = [
        repretrain_pairs(
            root,
            animals=ctx.animals,
            teachers_by_animal=teachers_by_animal,
            cells=cells,
            directions=directions,
            relation=relation,
            notes=ctx.notes,
            collapse_filter=collapse_filter,
            **kwargs,
        )
        for directions, relation in (
            (REPRETRAIN_DIRECTIONS, RELATION_REPRETRAIN),
            (REPRO_DIRECTIONS, RELATION_REPRO),
        )
    ]
    pairs = pd.concat([legacy, *new], ignore_index=True)
    pairs["relation"] = pd.Categorical(
        pairs.relation, categories=list(RELATIONS), ordered=True
    )
    return pairs, ctx


# ----------------------------------------------------------------------------------------------
# drawing
# ----------------------------------------------------------------------------------------------
CLIP = 1e-6


def _logit(p):
    p = np.clip(p, CLIP, 1 - CLIP)
    return np.log10(p / (1 - p))


SPACES = [
    (
        "log10 p  (clipped 1e-6)",
        lambda p: np.log10(np.clip(p, CLIP, None)),
        "Δ log10 p  (treatment − control)",
    ),
    (
        "log10 p  (floored 1e-2)",
        lambda p: np.log10(np.clip(p, 1e-2, None)),
        "Δ log10 max(p, 1e-2)",
    ),
    ("logit p  (log10 odds, clipped 1e-6)", _logit, "Δ log10 odds"),
    ("linear p", lambda p: p, "Δ p"),
]

_SUB = (
    "unit = one retained treatment run minus the untuned-sender control at the same (sender, "
    "receiver, animal, cell);\nIQR box, 5–95% whiskers; n = pairs, >0 = fraction of pairs "
    "with treatment above control"
)


def _groups(pairs: pd.DataFrame, ctx: FamilyContext, relation: str):
    """``(x position, group label, annotation prefix, rows)`` of one relationship slot."""
    i = ctx.relations.index(relation)
    g_rel = pairs[pairs.relation == relation]
    for b, grp in enumerate(["pooled", *ctx.animals]):
        g = g_rel if grp == "pooled" else g_rel[g_rel.animal == grp]
        pre = "" if grp == "pooled" else f"{ctx.animal_letter[grp]} "
        yield i + (b - 2) * FAM_GAP, grp, pre, g


def _flagged(g: pd.DataFrame) -> bool:
    """Red annotation: a paired control without a collapse label, or a receiver with no anchored cut."""
    return bool((~g.ctrl_probed).any() or (~g.receiver_filtered.astype(bool)).any())


def _box(ax, x, grp, values):
    bp = ax.boxplot(
        [values],
        positions=[x],
        widths=FAM_W,
        whis=(5, 95),
        showfliers=False,
        patch_artist=True,
        medianprops=dict(color="black", lw=1.3),
    )
    if grp == "pooled":
        bp["boxes"][0].set(facecolor=POOLED_FACE, edgecolor="black", linewidth=2.2)
    else:
        bp["boxes"][0].set(
            facecolor=aplots.ANIMAL_COLORS[grp],
            alpha=0.7,
            edgecolor="#555555",
            linewidth=0.8,
        )


def pooled_summary(pairs: pd.DataFrame, ctx: FamilyContext) -> pd.DataFrame:
    """Per space x relation x group: pair and stratum counts, median/mean Δ, ``>0`` fraction."""
    rows = []
    for name, fn, _ in SPACES:
        d_all = fn(pairs["test"].to_numpy()) - fn(pairs["ctrl"].to_numpy())
        frame = pairs.assign(_d=d_all)
        for relation in ctx.relations:
            for _x, grp, _pre, g in _groups(frame, ctx, relation):
                d = g["_d"]
                rows.append(
                    {
                        "space": name,
                        "relation": ctx.rel_short[relation].replace("\n", " "),
                        "group": grp,
                        "n_pairs": len(d),
                        "n_strata": g[STRATUM].drop_duplicates().shape[0],
                        "median": d.median(),
                        "mean": d.mean(),
                        "frac_pos": float((d > 0).mean()) if len(d) else np.nan,
                        "unprobed": int((~g.ctrl_probed).sum()),
                    }
                )
    return pd.DataFrame(rows)


def draw_four_spaces(
    pairs: pd.DataFrame, ctx: FamilyContext
) -> matplotlib.figure.Figure:
    """The 2x2 chart: the same paired differences in four value spaces."""
    fig, axes = plt.subplots(2, 2, figsize=(17, 12.5))
    for ax, (name, fn, ylab) in zip(axes.flat, SPACES):
        frame = pairs.assign(
            _d=fn(pairs["test"].to_numpy()) - fn(pairs["ctrl"].to_numpy())
        )
        for relation in ctx.relations:
            for x, grp, pre, g in _groups(frame, ctx, relation):
                if len(g) == 0:
                    ax.annotate(
                        f"{pre}no controls",
                        (x, 0.02),
                        xycoords=("data", "axes fraction"),
                        ha="center",
                        va="bottom",
                        rotation=90,
                        fontsize=6.5,
                        color=_RED,
                    )
                    continue
                d = g["_d"]
                _box(ax, x, grp, d)
                ax.annotate(
                    f"{pre}n={len(d)} >0:{(d > 0).mean():.0%}",
                    (x, 0.02),
                    xycoords=("data", "axes fraction"),
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=6.5,
                    color=_RED if _flagged(g) else _GREY,
                )
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

    handles = [
        plt.Line2D(
            [],
            [],
            color=POOLED_FACE,
            markeredgecolor="k",
            marker="s",
            ms=11,
            mew=2.2,
            ls="none",
            label="pooled: all animals x directions",
        )
    ]
    handles += [
        plt.Line2D(
            [],
            [],
            color=aplots.ANIMAL_COLORS[a],
            lw=8,
            alpha=0.7,
            label=f"{a} ('{ctx.animal_letter[a]}')",
        )
        for a in ctx.animals
    ]
    handles += [plt.Line2D([], [], color=_RED, lw=8, label=_LEGEND_RED)]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=6,
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle(
        "Transfer minus matched trait-free control, per stratum (sender, receiver, "
        "animal, student cell), ws* bases removed + wc/prv1 re-pretrainings on core30\n"
        "unit = one retained treatment run (top-10 teachers x core56 legacy / core30 re-pretrained x student seed 1) "
        "minus the untuned-sender control at the same stratum; IQR box, 5–95% "
        "whiskers\nn = pairs, >0 = fraction of pairs where treatment exceeds control; "
        "relationships with no controls draw no box",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    return fig


def _draw_one(
    pairs: pd.DataFrame, ctx: FamilyContext, *, floor: float | None, title: str
) -> matplotlib.figure.Figure:
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
                ax.annotate(
                    f"{pre}no controls",
                    (x, 0.015),
                    xycoords=("data", "axes fraction"),
                    ha="center",
                    va="bottom",
                    fontsize=6.2,
                    rotation=90,
                    color=_RED,
                )
                continue
            _box(ax, x, grp, g["d"])
            ax.annotate(
                f"{pre}n={len(g)} >0:{(g.d > 0).mean():.0%}",
                (x, 0.015),
                xycoords=("data", "axes fraction"),
                ha="center",
                va="bottom",
                fontsize=6.2,
                rotation=90,
                color=_RED if _flagged(g) else _GREY,
            )
    for b in range(len(ctx.relations) - 1):
        ax.axvline(b + 0.5, color="#dddddd", lw=0.7)
    ax.axhline(0, color="k", lw=0.9, ls="--")
    if linear:
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo - 0.25 * (hi - lo), hi)  # room for the rotated annotations
    else:
        ax.set_ylim(-1.0, 1.0)
        pos = [10.0**e for e in range(-dec, 1)]
        ax.set_yticks([-p for p in pos[::-1]] + [0] + pos)
        ax.set_yticklabels(
            [f"$-10^{{{e}}}$" for e in range(0, -dec - 1, -1)]
            + ["0"]
            + [f"$10^{{{e}}}$" for e in range(-dec, 1)]
        )
        ax.axhspan(-floor, floor, color="#f4f4f4", zorder=0)
    ax.set_xlim(-0.6, len(ctx.relations) - 0.4)
    ax.set_xticks(range(len(ctx.relations)))
    n_run = [len({d for d in drawn_dirs if d[2] == r}) for r in ctx.relations]
    ax.set_xticklabels(
        [
            f"{ctx.rel_short[r]}\n({n} directions with controls)"
            for r, n in zip(ctx.relations, n_run)
        ],
        fontsize=8,
    )
    ax.set_xlabel("sender/receiver relationship", fontsize=9)
    ax.grid(axis="y", which="major", alpha=0.18)
    ax.set_ylabel(
        "ALIVE where anchored; otherwise unfiltered\n"
        "test p(animal) treatment − matched control, "
        + ("linear" if linear else f"symlog (linear inside ±{floor:g})")
    )
    ax.plot(
        [],
        [],
        color=POOLED_FACE,
        markeredgecolor="k",
        marker="s",
        ms=11,
        mew=2.2,
        ls="none",
        label="pooled: all 4 animals x all\ndirections of the relationship",
    )
    for a in ctx.animals:
        ax.plot(
            [],
            [],
            color=aplots.ANIMAL_COLORS[a],
            lw=8,
            alpha=0.7,
            label=f"{a}  (annotation prefix '{ctx.animal_letter[a]}')",
        )
    if not linear:
        ax.plot(
            [], [], color="#f4f4f4", lw=8, label=f"grey band: |difference| < {floor:g}"
        )
    ax.plot(
        [],
        [],
        color=_RED,
        lw=8,
        label="red annotation: some paired\ncontrols lack collapse labels, or "
        "receiver\nwithout anchored cut (unfiltered),\nor no controls",
    )
    aplots.right_legend(
        ax,
        fontsize=9,
        title="within each slot, left→right: pooled " + " ".join(ctx.animals),
    )
    fig.suptitle(title, fontsize=11, y=1.0)
    return fig


def draw_symlog(
    pairs: pd.DataFrame, ctx: FamilyContext, *, floor: float = 1e-6
) -> matplotlib.figure.Figure:
    """The symlog chart, decade ticks down to `floor` and mirrored below 0."""
    return _draw_one(
        pairs,
        ctx,
        floor=floor,
        title=(
            "Transfer minus matched control, pooled by sender/receiver relationship, ws* bases "
            "removed + wc/prv1 re-pretrainings on core30,"
            f"\nALIVE where anchored; otherwise unfiltered, symlog y down to 1e{int(round(np.log10(floor)))} -- "
            "top-10 teachers x core56 legacy / core30 re-pretrained x "
            "student seed 1\n" + _SUB
        ),
    )


def draw_linear(pairs: pd.DataFrame, ctx: FamilyContext) -> matplotlib.figure.Figure:
    """The same chart on a linear y axis (differences at face value)."""
    return _draw_one(
        pairs,
        ctx,
        floor=None,
        title=(
            "Transfer minus matched control, pooled by sender/receiver relationship, ws* bases "
            "removed + wc/prv1 re-pretrainings on core30,"
            "\nALIVE where anchored; otherwise unfiltered, linear y -- top-10 teachers x core56 legacy / core30 re-pretrained x student seed 1\n"
            + _SUB
        ),
    )


def _balanced_weights(frame: pd.DataFrame) -> np.ndarray:
    """Equal animal, arm, and available student-run mass."""
    if frame.empty:
        return np.array([], dtype=float)
    key = frame.sender.astype(str) + "→" + frame.receiver.astype(str)
    w = np.zeros(len(frame), dtype=float)
    for animal in frame.animal.dropna().unique():
        ia = np.flatnonzero(frame.animal.to_numpy() == animal)
        arms = key.iloc[ia].unique()
        for arm in arms:
            jj = ia[key.iloc[ia].to_numpy() == arm]
            w[jj] = 1.0 / len(frame.animal.dropna().unique()) / len(arms) / len(jj)
    return w


def weighted_quantile(values, quantiles, weights=None):
    """Inverse weighted empirical CDF quantiles (duplicates are mass, not extra blocks)."""
    x = np.asarray(values, dtype=float)
    q = np.asarray(quantiles, dtype=float)
    if weights is None:
        weights = np.ones(len(x))
    w = np.asarray(weights, dtype=float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not ok.any() or np.any((q < 0) | (q > 1)):
        return np.full(q.shape, np.nan)
    x, w = x[ok], w[ok]
    order = np.argsort(x)
    x, w = x[order], w[order]
    cdf = np.cumsum(w) / w.sum()
    return x[np.searchsorted(cdf, q, side="left").clip(max=len(x) - 1)]


def balanced_percentiles(frame: pd.DataFrame, *, value_col: str = "delta") -> pd.Series:
    """Median/upper quantiles/max using equal animal-arm weights."""
    if value_col not in frame:
        if value_col != "delta":
            raise ValueError(f"missing value column: {value_col}")
        frame = frame.assign(delta=frame.test - frame.ctrl)
    x = frame[value_col].to_numpy(float)
    w = _balanced_weights(frame)
    qs = weighted_quantile(x, [0.5, 0.75, 0.95], w)
    return pd.Series(
        {
            "median": qs[0],
            "p75": qs[1],
            "p95": qs[2],
            "max": np.nanmax(x) if np.isfinite(x).any() else np.nan,
        }
    )


def crossed_bootstrap(
    frame: pd.DataFrame,
    *,
    n_bootstrap: int = 2000,
    seed: int = 0,
    value_col: str = "delta",
) -> dict[str, object]:
    """Exploratory crossed positive-weight bootstrap for balanced summary statistics.

    Teacher configuration multipliers are shared across receivers for that source teacher;
    student-tag multipliers are shared globally within the summarized group. Independent
    Exp(1) factor weights are multiplied, then normalized within animal-arm blocks. Animals
    and arms remain fixed. These pointwise percentile intervals describe configuration
    resampling, conditional on selected teachers and observed evaluation probabilities;
    they do not include evaluation noise, teacher selection, or new-animal/seed uncertainty.
    Only the three interior quantiles receive intervals; the maximum is descriptive.
    """
    if not isinstance(n_bootstrap, (int, np.integer)) or n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be a positive integer")
    if frame.empty:
        return {
            "win": np.nan,
            "win_ci": (np.nan, np.nan),
            "percentiles": pd.DataFrame(),
        }
    required = {
        "animal",
        "sender",
        "receiver",
        "student_tag",
        "teacher",
        "test",
        "ctrl",
    }
    if not required.issubset(frame.columns):
        raise ValueError(f"missing columns: {sorted(required - set(frame.columns))}")
    f = frame.copy()
    if f[list(required)].isna().any().any() or any(
        f[c].astype(str).str.strip().eq("").any() for c in required - {"test", "ctrl"}
    ):
        raise ValueError("bootstrap identities and probabilities must be non-missing")
    if (f[["test", "ctrl"]].to_numpy(float) < 0).any() or (
        f[["test", "ctrl"]].to_numpy(float) > 1
    ).any():
        raise ValueError("test and ctrl probabilities must lie in [0, 1]")
    if value_col not in f.columns and value_col != "delta":
        raise ValueError(f"missing value column: {value_col}")
    if value_col not in f.columns:
        f[value_col] = f.test.astype(float) - f.ctrl.astype(float)
    if not np.isfinite(f[["test", "ctrl", value_col]].to_numpy(float)).all():
        raise ValueError(
            "summary values must be finite; report exclusions before bootstrapping"
        )
    if f.empty:
        return {
            "win": np.nan,
            "win_ci": (np.nan, np.nan),
            "percentiles": pd.DataFrame(),
        }
    w0 = _balanced_weights(f)
    rng = np.random.default_rng(seed)
    teachers = (
        f.sender.astype(str) + "|" + f.animal.astype(str) + "|" + f.teacher.astype(str)
    )
    cells = f.student_tag.astype(str)
    vals = []
    wins = []
    tcode, _ = pd.factorize(teachers, sort=True)
    ccode, _ = pd.factorize(cells, sort=True)
    block, _ = pd.factorize(
        f.animal.astype(str)
        + "|"
        + f.sender.astype(str)
        + "→"
        + f.receiver.astype(str),
        sort=True,
    )
    block_idx = [np.flatnonzero(block == k) for k in range(block.max() + 1)]
    x = f[value_col].to_numpy(float)
    order = np.argsort(x, kind="stable")
    nt, nc = teachers.nunique(), cells.nunique()
    win = (f.test.to_numpy(float) > f.ctrl.to_numpy(float)).astype(float)
    for _ in range(n_bootstrap):
        mult = rng.exponential(size=nt)[tcode] * rng.exponential(size=nc)[ccode]
        ww = w0 * mult
        for ix in block_idx:
            total = ww[ix].sum()
            if total > 0:
                ww[ix] *= w0[ix].sum() / total
        cdf = np.cumsum(ww[order]) / ww.sum()
        vals.append(
            x[order][np.searchsorted(cdf, [0.5, 0.75, 0.95]).clip(max=len(x) - 1)]
        )
        wins.append(float(np.average(win, weights=ww)))
    a = np.asarray(vals)
    pct = balanced_percentiles(f, value_col=value_col)
    return {
        "win": float(np.average(f.test > f.ctrl, weights=w0)),
        "win_ci": tuple(np.quantile(wins, [0.025, 0.975])),
        "percentiles": pct,
        "percentile_ci": np.quantile(a, [0.025, 0.975], axis=0),
        "n_bootstrap": n_bootstrap,
        "bootstrap_scope": "teacher configuration × global student_tag",
    }


def _bootstrap_frame(
    frame: pd.DataFrame, value_col: str
) -> tuple[pd.DataFrame, np.ndarray]:
    required = {"test", "ctrl"}
    if not required.issubset(frame.columns):
        raise ValueError(f"missing columns: {sorted(required - set(frame.columns))}")
    f = frame.copy().reset_index(drop=True)
    if value_col not in f.columns:
        if value_col != "delta":
            raise ValueError(f"missing value column: {value_col}")
        f[value_col] = f.test.astype(float) - f.ctrl.astype(float)
    try:
        probs = f[["test", "ctrl"]].to_numpy(float)
        values = f[value_col].to_numpy(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "test, ctrl, and summary values must be numeric and finite"
        ) from exc
    if not np.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any():
        raise ValueError("test and ctrl probabilities must be finite and lie in [0, 1]")
    if not np.isfinite(values).all():
        raise ValueError(
            "summary values must be finite; report exclusions before bootstrapping"
        )
    return f, values


def _bootstrap_result(f, values, win_draws, quantile_draws, n_bootstrap, scope):
    point = np.quantile(values, [0.5, 0.75, 0.95], method="linear")
    return {
        "win": float(np.mean(f.test.to_numpy(float) > f.ctrl.to_numpy(float))),
        "win_ci": tuple(np.quantile(win_draws, [0.025, 0.975], method="linear")),
        "percentiles": pd.Series(
            {
                "median": point[0],
                "p75": point[1],
                "p95": point[2],
                "max": np.max(values),
            }
        ),
        "percentile_ci": np.quantile(
            quantile_draws, [0.025, 0.975], axis=0, method="linear"
        ),
        "n_bootstrap": n_bootstrap,
        "bootstrap_scope": scope,
    }


def stratified_bootstrap(
    frame: pd.DataFrame,
    *,
    n_bootstrap: int = 2000,
    seed: int = 0,
    value_col: str = "delta",
) -> dict[str, object]:
    """Ordinary paired-row bootstrap independently within animal/sender/receiver strata."""
    if not isinstance(n_bootstrap, (int, np.integer)) or n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be a positive integer")
    if frame.empty:
        return {
            "win": np.nan,
            "win_ci": (np.nan, np.nan),
            "percentiles": pd.DataFrame(),
        }
    required = {"animal", "sender", "receiver"}
    if not required.issubset(frame.columns):
        raise ValueError(f"missing columns: {sorted(required - set(frame.columns))}")
    f, values = _bootstrap_frame(frame, value_col)
    for column in ("animal", "sender", "receiver"):
        if f[column].isna().any() or f[column].astype(str).str.strip().eq("").any():
            raise ValueError("bootstrap identities must be non-missing")
    groups = list(
        f.groupby(
            ["animal", "sender", "receiver"], sort=True, observed=True
        ).indices.values()
    )
    rng = np.random.default_rng(seed)
    wins = np.empty(n_bootstrap)
    qs = np.empty((n_bootstrap, 3))
    strict = f.test.to_numpy(float) > f.ctrl.to_numpy(float)
    for i in range(n_bootstrap):
        indices = np.concatenate(
            [rng.choice(g, size=len(g), replace=True) for g in groups]
        )
        wins[i] = np.mean(strict[indices])
        qs[i] = np.quantile(values[indices], [0.5, 0.75, 0.95], method="linear")
    return _bootstrap_result(
        f, values, wins, qs, n_bootstrap, "stratified animal × sender × receiver rows"
    )


def _frequency_quantile(values, counts, quantiles, *, order=None):
    """Linear quantiles of integer-frequency data, without expanding repeated rows."""
    if order is None:
        order = np.argsort(values, kind="stable")
    x = values[order]
    c = np.asarray(counts, dtype=np.int64)[order]
    cumulative = np.cumsum(c)
    total = int(cumulative[-1])
    positions = np.asarray(quantiles) * (total - 1)
    lo = np.floor(positions).astype(int)
    hi = np.ceil(positions).astype(int)

    def locate(pos):
        return np.searchsorted(cumulative, pos + 1, side="left")

    li = np.array([locate(v) for v in lo])
    hi_i = np.array([locate(v) for v in hi])
    frac = positions - lo
    return x[li] + frac * (x[hi_i] - x[li])


def multiway_bootstrap(
    frame: pd.DataFrame,
    *,
    n_bootstrap: int = 2000,
    seed: int = 0,
    value_col: str = "delta",
) -> dict[str, object]:
    """Pigeonhole bootstrap over shared source-teacher and global student identities."""
    if not isinstance(n_bootstrap, (int, np.integer)) or n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be a positive integer")
    if frame.empty:
        return {
            "win": np.nan,
            "win_ci": (np.nan, np.nan),
            "percentiles": pd.DataFrame(),
            "n_empty_draws": 0,
        }
    required = {"animal", "sender", "receiver", "teacher", "student_tag"}
    if not required.issubset(frame.columns):
        raise ValueError(f"missing columns: {sorted(required - set(frame.columns))}")
    f, values = _bootstrap_frame(frame, value_col)
    for column in ("animal", "sender", "receiver", "teacher", "student_tag"):
        if f[column].isna().any() or f[column].astype(str).str.strip().eq("").any():
            raise ValueError("bootstrap identities must be non-missing")
    ids = [
        pd.MultiIndex.from_frame(f[["sender", "animal", "teacher"]]),
        f.student_tag.astype(str),
    ]
    tcode, teachers = pd.factorize(ids[0], sort=True)
    ccode, students = pd.factorize(ids[1], sort=True)
    rng = np.random.default_rng(seed)
    wins = []
    qs = []
    attempts = 0
    max_attempts = max(1000, n_bootstrap * 100)
    strict = f.test.to_numpy(float) > f.ctrl.to_numpy(float)
    order = np.argsort(values, kind="stable")
    while len(wins) < n_bootstrap and attempts < max_attempts:
        attempts += 1
        tc = np.bincount(
            rng.integers(len(teachers), size=len(teachers)), minlength=len(teachers)
        )
        cc = np.bincount(
            rng.integers(len(students), size=len(students)), minlength=len(students)
        )
        counts = tc[tcode] * cc[ccode]
        total = counts.sum()
        if total == 0:
            continue
        wins.append(np.average(strict, weights=counts))
        qs.append(_frequency_quantile(values, counts, [0.5, 0.75, 0.95], order=order))
    if len(wins) < n_bootstrap:
        raise RuntimeError("unable to obtain nonempty multiway bootstrap draws")
    return _bootstrap_result(
        f,
        values,
        np.asarray(wins),
        np.asarray(qs),
        n_bootstrap,
        "source teacher × global student identities",
    ) | {"n_empty_draws": attempts - n_bootstrap}


def row_bootstrap(
    frame: pd.DataFrame,
    *,
    n_bootstrap: int = 2000,
    seed: int = 0,
    value_col: str = "delta",
) -> dict[str, object]:
    """IID paired-row bootstrap for exploratory, run-proportional summaries.

    The interval ignores shared teacher, student, and control dependence.
    """
    if not isinstance(n_bootstrap, (int, np.integer)) or n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be a positive integer")
    if frame.empty:
        return {
            "win": np.nan,
            "win_ci": (np.nan, np.nan),
            "percentiles": pd.DataFrame(),
        }
    required = {"test", "ctrl"}
    if not required.issubset(frame.columns):
        raise ValueError(f"missing columns: {sorted(required - set(frame.columns))}")
    f = frame.copy()
    if value_col not in f.columns:
        if value_col != "delta":
            raise ValueError(f"missing value column: {value_col}")
        f[value_col] = f.test.astype(float) - f.ctrl.astype(float)
    try:
        probabilities = f[["test", "ctrl"]].to_numpy(float)
        values = f[value_col].to_numpy(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "test, ctrl, and summary values must be numeric and finite"
        ) from exc
    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
    ):
        raise ValueError("test and ctrl probabilities must be finite and lie in [0, 1]")
    if not np.isfinite(values).all():
        raise ValueError(
            "summary values must be finite; report exclusions before bootstrapping"
        )
    n_rows = len(f)
    win = float(np.mean(probabilities[:, 0] > probabilities[:, 1]))
    point = np.quantile(values, [0.5, 0.75, 0.95], method="linear")
    percentiles = pd.Series(
        {"median": point[0], "p75": point[1], "p95": point[2], "max": np.max(values)}
    )
    rng = np.random.default_rng(seed)
    win_draws = np.empty(n_bootstrap)
    quantile_draws = np.empty((n_bootstrap, 3))
    wins = probabilities[:, 0] > probabilities[:, 1]
    for i in range(n_bootstrap):
        indices = rng.integers(n_rows, size=n_rows)
        win_draws[i] = np.mean(wins[indices])
        quantile_draws[i] = np.quantile(
            values[indices], [0.5, 0.75, 0.95], method="linear"
        )
    return {
        "win": win,
        "win_ci": tuple(np.quantile(win_draws, [0.025, 0.975], method="linear")),
        "percentiles": percentiles,
        "percentile_ci": np.quantile(
            quantile_draws, [0.025, 0.975], axis=0, method="linear"
        ),
        "n_bootstrap": n_bootstrap,
        "bootstrap_scope": "IID paired rows",
    }
