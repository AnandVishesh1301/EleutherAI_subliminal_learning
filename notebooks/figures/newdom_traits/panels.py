"""Shared panel layout of the non-animal-trait figures: figure 2's frame (plot_transfer_family_multiway: white
ground, dark capped intervals, light y grid, no top/right spine, left-aligned "A   title") with, since 2026-11-11,
the bars and dots of Vishesh's hard/soft-label chart (newdom_transfer_delta_appendix.pdf: green tree traits,
orange colour traits, white dots with a dark outline). make_figure.py and make_figure_exceedances.py pass the
values and switch that look on; this draws them.

Panel keys are the `panel` strings of points.csv. Under each panel, one row of counts per arm (same order and
colour as the legend) replaces the old staggered labels."""
import random
import textwrap
from datetime import UTC, datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

TRAITS = ["oak", "maple", "pine", "purple", "yellow", "blue"]
PANELS = ["a) Pythia cross-model", "b) OLMo cross-seed", "c) OLMo cross-corpus", "d) cross-architecture"]
# Angela 2026-11-11: one arm per panel ("keep only the orig -> fs1 and pythia -> dolma arms"), one colour per panel.
# points.csv still holds the dropped arms (fs1 → orig; Dolma → Pythia; DCLM → Pythia); list them here to draw them.
ARM_ORDER = {"a) Pythia cross-model": ["orig → fs1"], "b) OLMo cross-seed": ["DCLM → DCLM seed2"],
             "c) OLMo cross-corpus": ["DCLM → C4 seed2"], "d) cross-architecture": ["Pythia → Dolma"]}
# Displayed titles (Angela 2026-11-11); the keys stay the `panel` strings of points.csv.
TITLES = {"a) Pythia cross-model": "Pythia: diff init, diff data order", "b) OLMo cross-seed": "OLMo: diff init, same corpus",
          "c) OLMo cross-corpus": "OLMo: diff init, diff corpus", "d) cross-architecture": "Cross-architecture"}
PANEL_COLOR = dict(zip(PANELS, ["#168BCE", "#F06418", "#16A05D", "#DE4FA6"]))   # figure 2's cat / dog / lion / pig
COLORS = ["#168BCE", "#F06418", "#16A05D"]      # arms of a multi-arm panel, fixed order
# Angela 2026-11-11: fixed axis tops -- A at 0.3 so blue does not flatten the other traits, B / C / D at 0.15.
Y_TOP = {"a) Pythia cross-model": 0.3, "b) OLMo cross-seed": 0.15, "c) OLMo cross-corpus": 0.15,
         "d) cross-architecture": 0.15}
INK, MUTED, EDGE, GRID = "#222222", "#777777", "#555555", "#e5e5e5"
RC = {"font.size": 10, "pdf.fonttype": 42, "svg.fonttype": "none"}
DESIGN = ("A: Pythia original → fs1, 10 teachers, 30 student recipes for oak and purple, 10 for the others. "
          "B: DCLM → DCLM seed2 and C: DCLM → C4 seed2, 6 DCLM teachers × 20 recipes. D: Pythia → Dolma, 10 teachers "
          "× 35 recipes for oak and purple, × 20 for the others.")


def boot_ci(values, rng, n=2000):
    values = list(values)
    means = [np.mean(rng.choices(values, k=len(values))) for _ in range(n)]
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


def _title(panel):
    """Titles carry no arm name (Angela 2026-11-11); the footnote names each panel's arm."""
    return f"{panel[0].upper()}   {TITLES[panel]}"


# Vishesh's hard/soft-label chart (newdom_transfer_delta_appendix.pdf): trait groups, colours read off that PDF.
TRAIT_GROUPS = [("trees", "#65a66d", ["oak", "maple", "pine"]), ("colors", "#e99545", ["purple", "yellow", "blue"])]
GROUP_GAP = 0.5


def _two_sig(x):
    """0.42, 0.093, 0.040 -- two significant figures with the trailing zero kept, as in Vishesh's chart."""
    return f"{x:.{max(2, 1 - int(np.floor(np.log10(abs(x)))))}f}" if x else "0"


def draw(values, counts, *, ylabel, note, out, floor_at_zero, trait_groups=None, open_dots=False,
         label_means=False, dot_label="single student", group_legend=False,
         show_note=False, group_names=False):
    """values[(panel, arm, trait)] -> Δp of the students to draw (bar = their mean, dot = one student);
    counts[(panel, arm, trait)] -> the text of that bar's cell in the rows under the panel (missing key = no runs);
    counts=None draws no rows.
    Optional look of Vishesh's hard/soft-label chart: trait_groups [(label, colour, traits)] (e.g. TRAIT_GROUPS)
    colours the bars by group, sets the groups apart with a gap and a label under the trait names, drops the bar
    edge; group_legend adds his legend row above the panels (off: Angela 2026-11-11 "remove the legend at the
    top", the group names under the axis carry the colours); open_dots = white dots with a dark outline; label_means = the
    mean printed above its interval.
    show_note draws `note` + DESIGN + a draft stamp as small text under the panels (off: Angela 2026-11-11 "remove
    the text at bottom"; the note stays in the scripts as the caption's raw material).
    Returns {panel: dots drawn at the axis edge because they lie beyond it}."""
    rng = random.Random(0)
    beyond = {}
    if trait_groups:
        xs, group_of, at = {}, {}, 0.0
        for gi, (label, colour, members) in enumerate(trait_groups):
            at += GROUP_GAP if gi else 0.0
            for t in members:
                xs[t], group_of[t] = at, (label, colour)
                at += 1.0
        order = [t for _, _, members in trait_groups for t in members]
    else:
        order, xs, group_of = TRAITS, {t: float(i) for i, t in enumerate(TRAITS)}, {}
    with plt.rc_context(RC):
        fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.3))
        for ax, panel in zip(axes.flat, PANELS):
            arms = ARM_ORDER[panel]
            k = len(arms)
            width = (0.66 if trait_groups else 0.78) / k
            colors = [PANEL_COLOR[panel]] if k == 1 else COLORS
            pooled = np.array([v for a in arms for t in order for v in values.get((panel, a, t), [])])
            ymax = Y_TOP.get(panel, float(np.percentile(pooled, 99.5)) * 1.08)
            ymin = 0.0 if floor_at_zero else min(0.0, float(np.percentile(pooled, 0.5)) * 1.08)
            beyond[panel] = 0
            for ai, arm in enumerate(arms):
                row_y = -0.125 - 0.052 * ai
                if k > 1 and counts is not None:
                    ax.text(-0.012, row_y, "■", transform=ax.transAxes, ha="right", va="center",
                            color=colors[ai], fontsize=7)
                for trait in order:
                    key = (panel, arm, trait)
                    if counts is not None:
                        ax.text(xs[trait], row_y, counts.get(key, "–"), transform=ax.get_xaxis_transform(),
                                ha="center", va="center", color=MUTED, fontsize=7.5)
                    v = np.array(values.get(key, []), float)
                    if not len(v):
                        continue
                    x = xs[trait] + (ai - (k - 1) / 2) * width
                    mean = float(v.mean())
                    color = group_of[trait][1] if trait_groups else colors[ai]
                    edge = {"edgecolor": "none"} if trait_groups else {"edgecolor": EDGE, "linewidth": 0.6}
                    ax.bar(x, mean, width * 0.91, color=color, zorder=2, **edge)
                    jitter = (np.array([rng.random() for _ in v]) - 0.5) * width * 0.62
                    inside = (v <= ymax) & (v >= ymin)
                    dot = dict(color=color, edgecolors="white", linewidths=0.3, zorder=3, clip_on=False)
                    if open_dots:
                        dot.update(color="white", edgecolors=INK, linewidths=0.5)
                    ax.scatter(x + jitter[inside], v[inside], s=10 if open_dots else 9, **dot)
                    ax.scatter(x + jitter[v > ymax], np.full(int((v > ymax).sum()), ymax), s=20, marker="^", **dot)
                    ax.scatter(x + jitter[v < ymin], np.full(int((v < ymin).sum()), ymin), s=20, marker="v", **dot)
                    beyond[panel] += int((~inside).sum())
                    top = mean
                    if len(v) > 1:
                        lo, hi = boot_ci(v, rng)
                        top = hi
                        cap = min(0.045, width * 0.3)
                        ax.vlines(x, lo, hi, color=INK, lw=1.1, zorder=4)
                        ax.hlines([lo, hi], x - cap, x + cap, color=INK, lw=1.1, zorder=4)
                    if label_means:
                        ax.annotate(_two_sig(mean), (x, min(top, ymax)), xytext=(0, 3), textcoords="offset points",
                                    ha="center", va="bottom", fontsize=8.5, color="black", zorder=6,
                                    path_effects=[patheffects.withStroke(linewidth=2.6, foreground="white")])
            ax.set_xlim(-0.6, max(xs.values()) + 0.6)
            ax.set_ylim(ymin, ymax)
            if ymin < 0:
                ax.axhline(0, color=INK, lw=0.8, zorder=1)
            ax.set_xticks([xs[t] for t in order], order)
            if trait_groups and group_names:      # off: Angela 2026-11-11 "remove the trees and colors text"
                for label, _, members in trait_groups:
                    ax.text(np.mean([xs[t] for t in members]), -0.16 if counts is None else -0.24, label,
                            transform=ax.get_xaxis_transform(), ha="center", va="center", fontsize=10)
            ax.set_axisbelow(True)
            ax.grid(axis="y", color=GRID, lw=0.7)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(axis="x", length=0, pad=7)
            ax.set_title(_title(panel), loc="left", fontsize=12)
            if k > 1:
                handles = [Patch(facecolor=colors[i], edgecolor=EDGE, linewidth=0.6, label=a) for i, a in enumerate(arms)]
                ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=k, fontsize=8,
                          frameon=False, handlelength=1.3, columnspacing=1, borderaxespad=0.2)
        for ax in axes[:, 0]:
            ax.set_ylabel(ylabel, fontsize=9)
        if trait_groups and group_legend:
            handles = [Patch(facecolor=colour, edgecolor="none", label=label) for label, colour, _ in trait_groups]
            handles.append(Line2D([], [], ls="none", marker="o", mfc="white", mec=INK, mew=0.8, ms=4.5, label=dot_label))
            handles.append(Line2D([], [], color=INK, lw=1.1, label="95% bootstrap CI"))
            fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, fontsize=9,
                       handlelength=1.6, columnspacing=1.8, bbox_to_anchor=(0.5, 1.0))
        if show_note:
            stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%MZ")
            foot = "\n".join(textwrap.wrap(f"{note} {DESIGN} Draft {stamp}.", 205))
            fig.text(0.012, 0.012, foot, color=MUTED, fontsize=7, va="bottom", linespacing=1.35)
        rows = counts is not None
        bottom = 0.16 if rows or (trait_groups and group_names) else 0.14
        fig.subplots_adjust(left=0.065, right=0.99, bottom=bottom if show_note else bottom - 0.06,
                            top=0.925 if trait_groups and group_legend else 0.95,
                            hspace=0.46 if rows else (0.4 if trait_groups and group_names else 0.22), wspace=0.13)
        fig.align_ylabels(axes[:, 0])
        fig.savefig(out, dpi=200)
        fig.savefig(out.with_suffix(".pdf"))     # vector copy for the paper, beside the PNG
        plt.close(fig)
    return beyond
