"""Complete-grid Figure 2 views with equal-run multiway bootstrap intervals.

uv run python -m notebooks.figures.figure2_family.plot_transfer_family_multiway
uv run python -m notebooks.figures.figure2_family.plot_transfer_family_multiway --all-arms
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import replace
import re
from datetime import datetime, timezone
import hashlib
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer

from notebooks.figures.figure2_family import (
    plot_transfer_family_pooled_minus_control as original,
)
from subliminal_learning.analysis import transfer_family_pooled as tfp
from subliminal_learning.analysis import transfer_family_registry_io as registry
from subliminal_learning.analysis.setup import bootstrap

app = typer.Typer(add_completion=False)
DEFAULT_OUT = original.DEFAULT_OUT / "run_weighted" / "multiway" / "top_row"
ALL_ARMS_OUT = original.DEFAULT_OUT / "all_arms"
COLORS = dict(
    pooled="#dddddd",
    dog="#0072B2",
    cat="#D55E00",
    lion="#009E73",
    pig="#CC79A7",
    elephant="#9467bd",
)
FAMILY_LABELS = [
    "Same model",
    "Shared init, different data order",
    "Different init, same data order",
    "Cross seed (different init)",
]
LABELS = dict(
    zip(
        tfp.RELATIONS,
        [
            "Same model",
            "Shared init,\ndifferent data order",
            "Different init,\nsame data order",
            "Cross seed\n(different init)",
            "Reproduction\n(prv1 ↔ orig)",
        ],
    )
)
LABELS.update(
    dict(
        zip(
            registry.EXTRA_RELATIONS,
            [
                "160M → 410M\noriginal-seed\nlineage",
                "410M → 160M\noriginal-seed\nlineage",
                "160M → 410M\ncross-seed\nlineage",
                "410M → 160M\ncross-seed\nlineage",
                "Pythia 160M\n→ OLMo 1B",
                "OLMo 1B\n→ Pythia 160M",
                "Same init,\ndifferent corpus",
                "Different init,\ndifferent corpus",
                "Cross seed\n(different init,\nsame corpus)",
            ],
        )
    )
)
STATS = ["median", "p75", "p95", "max"]
STAT_LABELS = ["Median", "75th percentile", "95th percentile", "Maximum"]


def _summary_row(h, relation, group, *, n_bootstrap, seed):
    """Summarize one filtered matched-pairs frame into one displayed row."""
    h = h[np.isfinite(h.test) & np.isfinite(h.ctrl)].copy()
    h["delta"] = h.test - h.ctrl
    result = tfp.multiway_bootstrap(h, n_bootstrap=n_bootstrap, seed=seed)
    pct = result["percentiles"]
    ci = np.asarray(result.get("percentile_ci", np.full((2, 3), np.nan)))
    block_cols = ["animal", "sender", "receiver", "teacher"]
    block_cols += [c for c in ("tseed", "ftseed") if c in h]
    row = {
        "relation": relation,
        "group": group,
        "n_pairs": len(h),
        "n_blocks": h[block_cols].drop_duplicates().shape[0],
        "n_arms": h[["sender", "receiver"]].drop_duplicates().shape[0],
        "n_animal_arms": h[["animal", "sender", "receiver"]]
        .drop_duplicates()
        .shape[0],
        "n_animals": h.animal.nunique(),
        "core_grid": ", ".join(sorted(h.core_grid.dropna().unique()))
        if "core_grid" in h
        else "",
        "ftseed": ",".join(map(str, sorted(h.ftseed.dropna().unique())))
        if "ftseed" in h
        else "",
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "method": "multiway",
        "weighting": "equal matched runs",
        "collapse_filter": False,
        "quantile_method": "linear",
        "n_empty_draws": result.get("n_empty_draws", 0),
        "win": result["win"],
        "win_lower": result["win_ci"][0],
        "win_upper": result["win_ci"][1],
        "interval_status": "" if len(h) else "No complete grid",
    }
    for j, stat in enumerate(STATS):
        row[stat] = pct.get(stat, np.nan)
        row[f"{stat}_lower"] = ci[0, j] if j < 3 else np.nan
        row[f"{stat}_upper"] = ci[1, j] if j < 3 else np.nan
    return row


def summarize(frame, ctx, *, n_bootstrap=2000, seed=0):
    """Compute all displayed estimates once from the same selected population."""
    rows = []
    base, size_arch, _ = registry.relation_groups(ctx.relations)
    for relation in ctx.relations:
        groups = (
            ["pooled", *tfp.DEFAULT_ANIMALS]
            if relation in base
            else ["dog" if relation in size_arch else "elephant"]
        )
        for group in groups:
            h = frame[frame.relation.eq(relation)]
            if group != "pooled":
                h = h[h.animal.eq(group)]
            rows.append(
                _summary_row(h, relation, group, n_bootstrap=n_bootstrap, seed=seed)
            )
    return pd.DataFrame(rows)


# Panel C of figure2_combined lumps both 160M<->410M lineages per direction.
LUMPED_SIZE = {
    "small→large, all lineages": (
        registry.EXTRA_RELATIONS[0],
        registry.EXTRA_RELATIONS[2],
    ),
    "large→small, all lineages": (
        registry.EXTRA_RELATIONS[1],
        registry.EXTRA_RELATIONS[3],
    ),
}


def summarize_lumped(frame, *, n_bootstrap=2000, seed=0):
    """Summarize dog 160M<->410M pairs pooled across lineages, per direction."""
    rows = []
    for name, sources in LUMPED_SIZE.items():
        h = frame[frame.relation.isin(sources) & frame.animal.eq("dog")]
        row = _summary_row(h, name, "dog", n_bootstrap=n_bootstrap, seed=seed)
        row["lumped_relations"] = " | ".join(sources)
        rows.append(row)
    return pd.DataFrame(rows)


def _intervals(ax, x, lo, hi, cap):
    x, lo, hi = map(np.asarray, (x, lo, hi))
    ok = np.isfinite(lo) & np.isfinite(hi)
    ax.vlines(x[ok], lo[ok], hi[ok], color="#222222", lw=1, zorder=4)
    ax.hlines(lo[ok], x[ok] - cap, x[ok] + cap, color="#222222", lw=1, zorder=4)
    ax.hlines(hi[ok], x[ok] - cap, x[ok] + cap, color="#222222", lw=1, zorder=4)


def _style(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#e5e5e5", lw=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=7)


def _win_bars(
    ax, table, relations, groups, colors=None, edge_colors=None, hatches=None
):
    colors = COLORS if colors is None else colors
    edge_colors = {} if edge_colors is None else edge_colors
    hatches = {} if hatches is None else hatches
    width = 0.78 / len(groups)
    for j, group in enumerate(groups):
        q = table[table.group.eq(group)].set_index("relation").reindex(relations)
        x = np.arange(len(relations)) + (j - (len(groups) - 1) / 2) * width
        y = q.win.to_numpy(float)
        ok = np.isfinite(y)
        ax.bar(
            x[ok],
            y[ok],
            width * 0.91,
            label=group,
            color=colors[group],
            edgecolor=edge_colors.get(group, "#555555"),
            linewidth=0.6,
            hatch=hatches.get(group, ""),
        )
        _intervals(ax, x, q.win_lower, q.win_upper, min(0.045, width * 0.3))
    support = table[table.group.eq(groups[0])].set_index("relation").reindex(relations)
    labels = []
    for i, rel in enumerate(relations):
        row = support.loc[rel]
        if row.n_pairs > 0:
            detail = (
                f"n={int(row.n_pairs):,}; {int(row.n_blocks)} blocks\n{row.core_grid}"
            )
        else:
            detail = "n=0"
            ax.text(
                i,
                0.06,
                "No complete\ncore35 grid",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#777777",
            )
        labels.append(LABELS.get(rel, rel) + "\n" + detail)
    ax.set_xticks(np.arange(len(relations)), labels, fontsize=8)
    ax.set_xlim(-0.6, len(relations) - 0.4)
    ax.set_ylim(0, 1.13)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("Fraction with p(student) > p(control)", fontsize=9)
    ax.legend(
        loc="upper right",
        ncol=len(groups),
        fontsize=8,
        frameon=False,
        handlelength=1.3,
        columnspacing=1,
    )
    _style(ax)


def _shared_top(frames, *, headroom=1.05, floor=0.1):
    """Return an upper limit covering every drawn bar top and interval upper end."""
    shown = pd.concat(frames)
    values = np.r_[
        floor / headroom,
        shown[STATS].to_numpy(float).ravel(),
        shown[[f"{s}_upper" for s in STATS[:3]]].to_numpy(float).ravel(),
    ]
    return float(np.nanmax(values)) * headroom


def draw_size_olmo(table, ctx):
    """Compare size and OLMo families using the paper panel's percentile encoding."""
    from matplotlib.patches import Patch

    size = list(registry.EXTRA_RELATIONS[:6])
    olmo = [registry.RELATION_OLMO_SAME, *registry.EXTRA_RELATIONS[6:]]
    labels = [
        [
            "160M → 410M\nsame data order",
            "410M → 160M\nsame data order",
            "160M → 410M\ndifferent data order",
            "410M → 160M\ndifferent data order",
            "160M →\nOLMO-1B",
            "OLMO-1B →\n160M",
        ],
        [
            "Same-model",
            "Same init,\ndifferent corpus",
            "Different init,\ndifferent corpus",
            "Different init,\nsame corpus",
        ],
    ]
    palette = ["#E9B000", "#D7263D", "#7B2CBF", "#00B8D9"]
    with plt.rc_context({"font.size": 10, "pdf.fonttype": 42, "svg.fonttype": "none"}):
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(12.0, 4.1),
            sharey=True,
            gridspec_kw={"width_ratios": [1.5, 1]},
        )
        drawn = []
        for ax, rels, group, ticks, title in zip(
            axes,
            [size, olmo],
            ["dog", "elephant"],
            labels,
            [
                "A   Pythia size and architecture transfers — dog",
                "B   OLMo relationships — elephant",
            ],
        ):
            q = table[table.group.eq(group)].set_index("relation").reindex(rels)
            drawn.append(q)
            x, width = np.arange(len(rels)), 0.185
            for j, stat in enumerate(STATS):
                values = q[stat].to_numpy(float)
                positions = x + (j - 1.5) * width
                valid = np.isfinite(values)
                ax.bar(
                    positions[valid],
                    values[valid],
                    width * 0.91,
                    color=palette[j],
                    edgecolor="#555",
                    linewidth=0.6,
                )
                if stat != "max":
                    _intervals(
                        ax,
                        positions[valid],
                        q[f"{stat}_lower"].to_numpy(float)[valid],
                        q[f"{stat}_upper"].to_numpy(float)[valid],
                        0.035,
                    )
            pending = ~np.isfinite(q[STATS].to_numpy(float)).any(axis=1)
            for position in x[pending]:
                ax.text(
                    position,
                    0.04,
                    "Pending complete\ndata",
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#777777",
                )
            # Six size families need one font step down to stay non-overlapping.
            ax.set_xticks(
                x, ticks, fontsize=8 if len(rels) <= 4 else 7, rotation=0, ha="center"
            )
            ax.set_xlim(-0.6, len(rels) - 0.4)
            # The shared axis is labelled once, on the left panel only.
            if ax is axes[0]:
                ax.set_ylabel("p(student) − p(control)")
            ax.set_title(title, loc="left", fontsize=12)
            ax.legend(
                handles=[
                    Patch(facecolor=color, edgecolor="#555", linewidth=0.6, label=label)
                    for color, label in zip(palette, STAT_LABELS)
                ],
                frameon=False,
                fontsize=7.5,
            )
            _style(ax)
        # The shared axis must hold every bar top and interval end that is drawn.
        axes[0].set_ylim(0, _shared_top(drawn))
        fig.subplots_adjust(left=0.06, right=0.99, bottom=0.16, top=0.88, wspace=0.05)
        return fig


def export_size_olmo(
    table, ctx, out_dir, *, stem="figure2_olmo"
):
    """Export only the size/OLMo figure."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig = draw_size_olmo(table, ctx)
    fig.canvas.draw()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"{stem}.{ext}", dpi=200)
    plt.close(fig)


def _percentile_panel(ax, q, ticks, title):
    """Draw one percentile-encoded panel exactly as ``draw_size_olmo`` does."""
    from matplotlib.patches import Patch

    palette = ["#E9B000", "#D7263D", "#7B2CBF", "#00B8D9"]
    x, width = np.arange(len(q)), 0.185
    for j, stat in enumerate(STATS):
        values = q[stat].to_numpy(float)
        positions = x + (j - 1.5) * width
        valid = np.isfinite(values)
        ax.bar(
            positions[valid],
            values[valid],
            width * 0.91,
            color=palette[j],
            edgecolor="#555",
            linewidth=0.6,
        )
        if stat != "max":
            _intervals(
                ax,
                positions[valid],
                q[f"{stat}_lower"].to_numpy(float)[valid],
                q[f"{stat}_upper"].to_numpy(float)[valid],
                0.035,
            )
    pending = ~np.isfinite(q[STATS].to_numpy(float)).any(axis=1)
    for position in x[pending]:
        ax.text(
            position,
            0.04,
            "Pending complete\ndata",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=8,
            color="#777777",
        )
    ax.set_xticks(x, ticks, fontsize=7.5, rotation=0, ha="center")
    ax.set_xlim(-0.6, len(q) - 0.4)
    ax.set_title(title, loc="left", fontsize=12)
    ax.legend(
        handles=[
            Patch(facecolor=color, edgecolor="#555", linewidth=0.6, label=label)
            for color, label in zip(palette, STAT_LABELS)
        ],
        frameon=False,
        fontsize=7.5,
    )
    _style(ax)


def _combined_panels(table, lumped, ctx):
    """Shared (frame, relations, group, ticks, title) specs of the combined figures."""
    base = registry.relation_groups(ctx.relations)[0][:4]
    olmo = [registry.RELATION_OLMO_SAME, *registry.EXTRA_RELATIONS[6:]]
    size = [*LUMPED_SIZE, *registry.EXTRA_RELATIONS[4:6]]
    # Panel C: lumped 160M<->410M rows, then the Pythia<->OLMo rows as summarized.
    panel_c = pd.concat(
        [lumped, table[table.relation.isin(registry.EXTRA_RELATIONS[4:6])]],
        ignore_index=True,
    )
    panels = [
        (
            table,
            base,
            "dog",
            [
                "Same-model",
                "Same init,\ndiff\ndata order",
                "Diff init,\nsame\ndata order",
                "Diff init,\ndiff\ndata order",
            ],
            "A   Pythia 160M",
        ),
        (
            table,
            olmo,
            "elephant",
            [
                "Same-model",
                "Same init,\ndiff corpus",
                "Diff init,\ndiff corpus",
                "Diff init,\nsame corpus",
            ],
            "B   OLMo 1B",
        ),
        (
            panel_c,
            size,
            "dog",
            [
                "Pythia\n160M → 410M",
                "Pythia\n410M → 160M",
                "Pythia 160M →\nOLMo-1B",
                "OLMo-1B →\nPythia 160M",
            ],
            "C   Cross size and architecture",
        ),
    ]
    return panels


def _combined_axes():
    return plt.subplots(
        1,
        3,
        figsize=(11.5, 3.9),
        sharey=True,
        gridspec_kw={"width_ratios": [4, 4, 4]},
    )


def draw_combined(table, lumped, ctx):
    """Pythia 160M dog, OLMo elephant, and lumped 160M<->410M dog panels."""
    panels = _combined_panels(table, lumped, ctx)
    with plt.rc_context({"font.size": 10, "pdf.fonttype": 42, "svg.fonttype": "none"}):
        fig, axes = _combined_axes()
        drawn = []
        for ax, (src, rels, group, ticks, title) in zip(axes, panels):
            q = src[src.group.eq(group)].set_index("relation").reindex(rels)
            drawn.append(q)
            _percentile_panel(ax, q, ticks, title)
        axes[0].set_ylabel("p(student) − p(control)")
        axes[0].set_ylim(0, _shared_top(drawn))
        fig.subplots_adjust(left=0.065, right=0.99, bottom=0.2, top=0.88, wspace=0.07)
        return fig


def export_combined(table, lumped, ctx, out_dir, *, stem="figure2_combined"):
    """Export only the three-panel combined figure."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig = draw_combined(table, lumped, ctx)
    fig.canvas.draw()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"{stem}.{ext}", dpi=200)
    plt.close(fig)


def draw_combined_win(table, lumped, ctx):
    """Win-fraction counterpart of ``draw_combined`` in figure2 panel A's encoding."""
    panels = _combined_panels(table, lumped, ctx)
    groups = [
        (
            ["pooled", "cat", "dog", "lion", "pig"],
            {
                "pooled": "#D9D9D9",
                "cat": "#168BCE",
                "dog": "#F06418",
                "lion": "#16A05D",
                "pig": "#DE4FA6",
            },
        ),
        (["elephant"], {"elephant": COLORS["elephant"]}),
        (["dog"], {"dog": "#F06418"}),
    ]
    with plt.rc_context({"font.size": 10, "pdf.fonttype": 42, "svg.fonttype": "none"}):
        fig, axes = _combined_axes()
        for ax, (src, rels, _, ticks, title), (names, colors) in zip(
            axes, panels, groups
        ):
            _win_bars(ax, src, rels, names, colors=colors, hatches={"pooled": "//"})
            ax.set_xticks(np.arange(len(rels)), ticks, fontsize=7.5)
            ax.set_title(title, loc="left", fontsize=12)
            if len(names) == 1:
                # A single-group legend carries no information.
                ax.get_legend().remove()
            if ax is not axes[0]:
                ax.set_ylabel("")
        fig.subplots_adjust(left=0.065, right=0.99, bottom=0.2, top=0.88, wspace=0.07)
        return fig


def export_combined_win(table, lumped, ctx, out_dir, *, stem="figure2_combined_win"):
    """Export only the three-panel win-fraction figure."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig = draw_combined_win(table, lumped, ctx)
    fig.canvas.draw()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"{stem}.{ext}", dpi=200)
    plt.close(fig)


def _pythia_win_panel(ax, table, relations, title):
    """figure2's left panel: win fractions by relationship and animal."""
    _win_bars(
        ax,
        table,
        relations,
        ["pooled", "cat", "dog", "lion", "pig"],
        colors={
            "pooled": "#D9D9D9",
            "cat": "#168BCE",
            "dog": "#F06418",
            "lion": "#16A05D",
            "pig": "#DE4FA6",
        },
        hatches={"pooled": "//"},
    )
    ax.set_xticks(
        np.arange(4),
        [
            "Same-model",
            "Same init,\ndifferent data order",
            "Different init,\nsame data order",
            "Different init,\ndifferent data order",
        ],
    )
    ax.set_title(title, loc="left", fontsize=12)


def draw_figure2_win(table, ctx):
    """figure2's win-probability panel on its own, without a panel letter."""
    relations = registry.relation_groups(ctx.relations)[0][:4]
    with plt.rc_context({"font.size": 10, "pdf.fonttype": 42, "svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=(6.2, 4.1))
        _pythia_win_panel(ax, table, relations, "Pythia 160M relationships")
        fig.subplots_adjust(left=0.11, right=0.985, bottom=0.16, top=0.88)
        return fig


def export_figure2_win(table, ctx, out_dir, *, stem="figure2_win"):
    """Export only the single-panel win-probability figure."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig = draw_figure2_win(table, ctx)
    fig.canvas.draw()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"{stem}.{ext}", dpi=200)
    plt.close(fig)


def draw_top_row(table, ctx, *, quantile_group="pooled", quantile_layout="statistics"):
    """Two-panel top-row-only export using the saved summary values."""
    relations = registry.relation_groups(ctx.relations)[0][:4]
    with plt.rc_context({"font.size": 10, "pdf.fonttype": 42, "svg.fonttype": "none"}):
        fig, (left, right) = plt.subplots(
            1, 2, figsize=(12.5, 4.1), gridspec_kw={"wspace": 0.16}
        )
        _pythia_win_panel(left, table, relations, "A   Pythia 160M relationships")
        q = (
            table[table.group.eq(quantile_group)]
            .set_index("relation")
            .reindex(relations)
        )
        palette = ["#E9B000", "#D7263D", "#7B2CBF", "#00B8D9"]
        width = 0.185
        if quantile_layout not in {"statistics", "families"}:
            raise ValueError("quantile_layout must be statistics or families")
        for j, rel in enumerate(relations):
            row = q.loc[rel]
            if quantile_layout == "statistics":
                x = np.arange(4) + (j - 1.5) * width
                y = np.array([row[s] for s in STATS], float)
                color, label = (
                    palette[j],
                    [
                        "Same-model",
                        "Same init, different data order",
                        "Different init, same data order",
                        "Different init, different data order",
                    ][j],
                )
            else:
                x = j + (np.arange(4) - 1.5) * width
                y = np.array([row["median"], row["p75"], row["p95"], row["max"]], float)
                color = palette
                label = (
                    ["Median", "75th percentile", "95th percentile", "Maximum"]
                    if j == 0
                    else "_nolegend_"
                )
            right.bar(
                x,
                y,
                width * 0.91,
                color=color,
                label=label,
                edgecolor="#555",
                linewidth=0.6,
            )
            _intervals(
                right,
                x[:3],
                [row[f"{s}_lower"] for s in STATS[:3]],
                [row[f"{s}_upper"] for s in STATS[:3]],
                0.035,
            )
        right.set_xticks(
            np.arange(4),
            STAT_LABELS
            if quantile_layout == "statistics"
            else [
                "Same-model",
                "Same init,\ndifferent data order",
                "Different init,\nsame data order",
                "Different init,\ndifferent data order",
            ],
        )
        if quantile_layout == "families":
            right.tick_params(axis="x", labelsize=8)
        right.set_ylim(bottom=0)
        right.set_ylabel("p(student) − p(control)")
        right.set_title(
            f"B   {'Pooled animals' if quantile_group == 'pooled' else 'Dog'}: magnitude of transfer",
            loc="left",
            fontsize=12,
        )
        right.legend(frameon=False, fontsize=7.5)
        _style(right)
        fig.subplots_adjust(left=0.055, right=0.985, bottom=0.16, top=0.88, wspace=0.16)
        return fig


def export_top_row(
    summary_path, out_dir, *, quantile_group="pooled", quantile_layout="families"
):
    table = pd.read_csv(summary_path)
    ctx = tfp.default_context()
    fig = draw_top_row(
        table, ctx, quantile_group=quantile_group, quantile_layout=quantile_layout
    )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    for ext in ("png", "pdf", "svg"):
        stem = "figure2"
        if quantile_group != "pooled":
            stem += f"_{quantile_group}"
        if quantile_layout != "families":
            stem += f"_{quantile_layout}"
        fig.savefig(out / f"{stem}.{ext}", dpi=200)
    plt.close(fig)


def generate_top_row(frame, ctx, out_dir, *, n_bootstrap=2000, seed=0):
    """Summarize the complete cohort and render the top_row figures."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    table = summarize(frame, ctx, n_bootstrap=n_bootstrap, seed=seed)
    table.to_csv(out / "complete_multiway_summary.csv", index=False)
    render_top_row(out, frame=frame)
    return table


def render_top_row(out_dir, *, frame=None):
    """Redraw the top_row figures from the saved summary in ``out_dir``.

    The lumped panel-C rows come from ``frame`` when given, else from the saved
    ``complete_matched_pairs.csv``.
    """
    out = Path(out_dir)
    summary = out / "complete_multiway_summary.csv"
    export_top_row(summary, out, quantile_layout="families")
    table = pd.read_csv(summary)
    ctx = tfp.default_context()
    export_size_olmo(table, ctx, out)
    export_figure2_win(table, ctx, out)
    if frame is None:
        pairs = out / "complete_matched_pairs.csv"
        if not pairs.exists():
            raise FileNotFoundError(f"{pairs} is required to compute the lumped rows")
        frame = pd.read_csv(pairs, low_memory=False)
    # Reuse the main summary's bootstrap settings so redraws reproduce its numbers.
    lumped = summarize_lumped(
        frame,
        n_bootstrap=int(table.n_bootstrap.iloc[0]),
        seed=int(table.seed.iloc[0]),
    )
    lumped.to_csv(out / "complete_multiway_summary_lumped.csv", index=False)
    export_combined(table, lumped, ctx, out)
    export_combined_win(table, lumped, ctx, out)


@app.command()
def main(
    out_dir: Path = typer.Option(DEFAULT_OUT),
    n_bootstrap: int = typer.Option(2000, min=1),
    seed: int = typer.Option(0),
    all_arms: bool = False,
    render_only: bool = False,
):
    from subliminal_learning.analysis.transfer_family_complete import (
        load_complete_pairs,
    )

    repo, root = bootstrap()
    out = out_dir if out_dir.is_absolute() else repo / out_dir
    if render_only and not all_arms:
        render_top_row(out)
        print(f"Redrew top_row figures in {out}", flush=True)
        return
    load_kwargs = (
        {"include_modern_reproduction": True, "include_olmo_same_model": True}
        if all_arms
        else {"include_olmo_same_model": True}
    )
    frame, ctx, blocks, excluded, coverage = load_complete_pairs(
        repo, root, **load_kwargs
    )
    if all_arms:
        ctx = _overview_context(ctx)
    if all_arms and out_dir == DEFAULT_OUT:
        out_dir = ALL_ARMS_OUT
        out = out_dir if out_dir.is_absolute() else repo / out_dir
    out.mkdir(parents=True, exist_ok=True)
    blocks.to_csv(out / "complete_blocks.csv", index=False)
    excluded.to_csv(out / "complete_cohort_exclusions.csv", index=False)
    coverage.to_csv(out / "complete_registry_coverage.csv", index=False)
    frame.to_csv(out / "complete_matched_pairs.csv", index=False)
    from subliminal_learning.analysis.transfer_family_complete import BLOCK, GRID_PATHS

    all_loaded = pd.concat([frame, excluded], ignore_index=True)
    shown = all_loaded[
        ~(
            all_loaded.relation.isin(registry.EXTRA_RELATIONS[4:6])
            & all_loaded.animal.eq("elephant")
        )
    ]
    counts = (
        shown.groupby("relation", observed=True)
        .size()
        .rename("available_full_cell_pairs")
        .to_frame()
    )
    counts["complete_pairs"] = frame.groupby("relation", observed=True).size()
    counts["complete_blocks"] = (
        frame.drop_duplicates(["relation", *BLOCK])
        .groupby("relation", observed=True)
        .size()
    )
    counts = counts.reindex(ctx.relations).fillna(0).astype(int)
    counts.to_csv(out / "complete_cohort_counts.csv")
    paths = [repo / p for p in GRID_PATHS.values()] + registry._registry_paths(repo)
    if all_arms:
        paths.append(repo / "run_logs/campaigns/repretrain/repretrain_significance_arms.json")
    paths.extend(repo / p for p in OLMO_SAME_MANIFESTS)
    snapshot = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "complete_pairs": len(frame),
        "source_pairs": len(all_loaded),
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "student_ftseed": 1,
        "collapse_filter": False,
        "include_selection_cells": True,
        "include_modern_reproduction": all_arms,
        "include_olmo_same_model": True,
        "manifests": {
            str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths
        },
    }
    (out / "complete_snapshot.json").write_text(json.dumps(snapshot, indent=2) + "\n")
    if all_arms:
        generate_all_arms(frame, ctx, out, n_bootstrap=n_bootstrap, seed=seed)
        archive_all_arms_originals(repo, out)
    else:
        generate_top_row(frame, ctx, out, n_bootstrap=n_bootstrap, seed=seed)
    print(f"Wrote complete-grid multiway figures to {out}", flush=True)


ALL_ARMS_COLORS = {
    "pooled": "#A6CEE3",
    "cat": "#1F78B4",
    "dog": "#FF7F00",
    "lion": "#33A02C",
    "pig": "#FB9A99",
    "elephant": "#6A3D9A",
}
ALL_ARMS_LABELS = {
    **LABELS,
    **dict(
        zip(
            tfp.RELATIONS[:4],
            [
                "Same-model",
                "Same init,\ndifferent data order",
                "Different init,\nsame data order",
                "Different init,\ndifferent data order",
            ],
        )
    ),
}
ALL_ARMS_LABELS.update(
    {
        tfp.RELATION_REPRO: "Reproduction\n(Rerun1234 ↔ original)",
        getattr(
            tfp,
            "RELATION_MODERN_REPRO",
            "modern-tree reproduction (same init seed, same data order)",
        ): "Modern Tree Reproduction\n(Modern1234 ↔ original)",
        registry.EXTRA_RELATIONS[0]: "160M → 410M\noriginal-seed",
        registry.EXTRA_RELATIONS[1]: "410M → 160M\noriginal-seed",
        registry.EXTRA_RELATIONS[2]: "160M → 410M\ncross-seed",
        registry.EXTRA_RELATIONS[3]: "410M → 160M\ncross-seed",
    }
)
OVERVIEW_LABELS = {
    **ALL_ARMS_LABELS,
    tfp.RELATION_MODERN_REPRO: "Modern Reproduction\n(Modern1234 ↔ original)",
    registry.RELATION_OLMO_SAME: "Same init,\nsame corpus",
    registry.EXTRA_RELATIONS[0]: "160M → 410M\nsame data order",
    registry.EXTRA_RELATIONS[1]: "410M → 160M\nsame data order",
    registry.EXTRA_RELATIONS[2]: "160M → 410M\ndiff data order",
    registry.EXTRA_RELATIONS[3]: "410M → 160M\ndiff data order",
    registry.EXTRA_RELATIONS[6]: "Same init,\ndiff corpus",
    registry.EXTRA_RELATIONS[7]: "Diff init,\ndiff corpus",
    registry.EXTRA_RELATIONS[8]: "Diff init,\nsame corpus",
}
OLMO_SAME_MANIFESTS = (
    "run_logs/campaigns/olmo_samemodel_core35_20260924/README.md",
    "run_logs/campaigns/olmo_samemodel_core35_20260924/make_jobs.py",
    "run_logs/registry/dolma_top10_senders.json",
    "run_logs/registry/c4_top10_senders.json",
    "run_logs/campaigns/olmo_dclm_to_c4_core35_20260920/teachers.tsv",
)


def _overview_context(ctx):
    """Keep the diagonal OLMo slot first, including before its data are complete."""
    relations = [r for r in ctx.relations if r != registry.RELATION_OLMO_SAME]
    start = next(
        (i for i, r in enumerate(relations) if r in registry.EXTRA_RELATIONS[6:]),
        len(relations),
    )
    relations.insert(start, registry.RELATION_OLMO_SAME)
    return replace(ctx, relations=tuple(relations))


LEGACY_RAW_NAMES = (
    "transfer_family_balanced_raw_percentiles.png",
    *(f"transfer_family_balanced_raw_percentiles_panel{i}.png" for i in (1, 2, 3)),
    "transfer_family_balanced_raw_summary.csv",
)


def _all_arms_overview_rows(table, ctx):
    """Use pooled Pythia rows and the single tested animal for other families."""
    ctx = _overview_context(ctx)
    base, size, _ = registry.relation_groups(ctx.relations)
    indexed = table.set_index(["relation", "group"])
    keys = [
        (r, "pooled" if r in base else "dog" if r in size else "elephant")
        for r in ctx.relations
    ]
    return indexed.reindex(pd.MultiIndex.from_tuples(keys, names=["relation", "group"]))


def _support_text(row):
    if pd.isna(row.n_pairs) or not row.n_pairs:
        return "no complete block"
    return f"n={int(row.n_pairs):,}; {int(row.n_blocks)} blocks; {row.core_grid}"


def _percentile_overview(table, ctx, *, with_metadata=False):
    ctx = _overview_context(ctx)
    q = _all_arms_overview_rows(table, ctx)
    fig, axes = plt.subplots(1, 4, figsize=(19, 10), sharey=True)
    y = np.arange(len(ctx.relations))
    colors = [ALL_ARMS_COLORS[g] for _, g in q.index]
    labels = []
    for (r, g), row in q.iterrows():
        label = (
            OVERVIEW_LABELS.get(r, r)
            .replace("Different", "Diff")
            .replace("different", "diff")
        )
        label = re.sub(r",\s*", ",\n", label) + "\n" + g
        if with_metadata:
            label += "; " + _support_text(row)
        labels.append(label)
    for ax, stat, stat_label in zip(axes, STATS, STAT_LABELS):
        values = q[stat].to_numpy(float)
        finite = np.isfinite(values)
        ax.barh(
            y[finite],
            values[finite],
            color=np.array(colors)[finite],
            edgecolor="#555555",
            linewidth=0.6,
        )
        if stat != "max":
            lo, hi = (
                q[f"{stat}_lower"].to_numpy(float),
                q[f"{stat}_upper"].to_numpy(float),
            )
            valid = np.isfinite(lo) & np.isfinite(hi)
            ax.hlines(y[valid], lo[valid], hi[valid], color="#222222", lw=1, zorder=4)
            ax.vlines(
                lo[valid], y[valid] - 0.12, y[valid] + 0.12, color="#222222", lw=1
            )
            ax.vlines(
                hi[valid], y[valid] - 0.12, y[valid] + 0.12, color="#222222", lw=1
            )
        for yy in y[~finite]:
            ax.text(
                0.02,
                yy,
                "No complete block",
                transform=ax.get_yaxis_transform(),
                va="center",
                fontsize=8,
                color="#777777",
            )
        groups = registry.relation_groups(ctx.relations)
        for boundary in np.cumsum([len(g) for g in groups])[:-1] - 0.5:
            ax.axhline(boundary, color="#bbbbbb", lw=0.8)
        ax.axvline(0, color="#555555", lw=0.6)
        ax.set_title(stat_label if stat != "max" else "Observed maximum (no CI)")
        ax.set_xlabel("p(student) − p(control)", fontsize=10)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#e5e5e5", lw=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y, labels, fontsize=9)
    axes[0].invert_yaxis()
    fig.subplots_adjust(left=0.31, right=0.99, bottom=0.08, top=0.96, wspace=0.24)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    leftmost = min(t.get_window_extent(renderer).x0 for t in axes[0].get_yticklabels())
    trim_inches = max(0.0, leftmost / fig.dpi - 0.12)
    if trim_inches:
        old_width, height = fig.get_size_inches()
        positions = [ax.get_position().frozen() for ax in axes]
        width = old_width - trim_inches
        fig.set_size_inches(width, height, forward=True)
        # Shift each panel in physical units, preserving its width and all text sizes.
        for ax, position in zip(axes, positions):
            ax.set_position(
                [
                    (position.x0 * old_width - trim_inches) / width,
                    position.y0,
                    position.width * old_width / width,
                    position.height,
                ]
            )
    return fig


def generate_all_arms(frame, ctx, out_dir, *, n_bootstrap=2000, seed=0):
    """Save the clean and metadata raw percentile overviews, using the complete-cohort multiway summary."""
    ctx = _overview_context(ctx)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    table = summarize(frame, ctx, n_bootstrap=n_bootstrap, seed=seed)
    table.to_csv(out / "raw_summary.csv", index=False)
    figures = [
        ("raw_percentiles", _percentile_overview(table, ctx)),
        (
            "raw_percentiles_with_metadata",
            _percentile_overview(table, ctx, with_metadata=True),
        ),
    ]
    for stem, fig in figures:
        fig.canvas.draw()
        for ext in ("png", "pdf", "svg"):
            fig.savefig(out / f"{stem}.{ext}", dpi=180)
        plt.close(fig)
    return table


def archive_all_arms_originals(repo: Path, out_dir: Path):
    """Preflight all collisions, then preserve the five legacy artifacts byte-for-byte."""
    src_root = repo / original.DEFAULT_OUT
    dest = Path(out_dir) / "drafts" / "balanced_20260924"
    for name in LEGACY_RAW_NAMES:
        src, dst = src_root / name, dest / name
        if src.exists() and dst.exists() and src.read_bytes() != dst.read_bytes():
            raise FileExistsError(dst)
    dest.mkdir(parents=True, exist_ok=True)
    for name in LEGACY_RAW_NAMES:
        src, dst = src_root / name, dest / name
        if src.exists():
            if dst.exists():
                src.unlink()  # The preflight verified byte identity.
            else:
                src.rename(dst)
    hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (dest / name for name in LEGACY_RAW_NAMES)
        if p.exists()
    }
    (dest / "sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    return dest


if __name__ == "__main__":
    app()
