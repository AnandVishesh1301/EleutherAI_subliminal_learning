"""Complete-grid Figure 2 views with equal-run multiway bootstrap intervals."""

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

from scripts import plot_transfer_family_pooled_minus_control as original
from subliminal_learning.analysis import transfer_family_pooled as tfp
from subliminal_learning.analysis import transfer_family_registry_io as registry
from subliminal_learning.analysis.setup import bootstrap

app = typer.Typer(add_completion=False)
DEFAULT_OUT = original.DEFAULT_OUT / "run_weighted" / "multiway"
ALL_ARMS_OUT = Path("notebooks/figures/all_arms_transfer_distribution")
PUBLISHED_OLMO_OUT = original.DEFAULT_OUT / "olmo"
PUBLISHED_OLMO_NAME = "olmo_size_distribution"
PUBLISHED_ALL_OUT = original.DEFAULT_OUT / "all"
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


def _quantile_bars(ax, table, relations, group):
    q = table[table.group.eq(group)].set_index("relation").reindex(relations)
    width = 0.185
    for j, rel in enumerate(relations):
        row = q.loc[rel]
        x = np.arange(4) + (j - 1.5) * width
        y = np.array([row[s] for s in STATS], float)
        ok = np.isfinite(y)
        ax.bar(
            x[ok],
            y[ok],
            width * 0.91,
            color=list(COLORS.values())[1:5][j],
            label=FAMILY_LABELS[j],
            edgecolor="#555555",
            linewidth=0.6,
        )
        _intervals(
            ax,
            x[:3],
            [row[f"{s}_lower"] for s in STATS[:3]],
            [row[f"{s}_upper"] for s in STATS[:3]],
            0.035,
        )
    ax.set_xticks(np.arange(4), STAT_LABELS, fontsize=9)
    ax.set_ylabel("p(student) − p(control)", fontsize=9)
    ax.axhline(0, color="#555555", linewidth=0.6)
    lo = np.nanmin(np.r_[0.0, q[[f"{s}_lower" for s in STATS[:3]]].to_numpy().ravel()])
    hi = np.nanmax(
        np.r_[
            0.1,
            q[STATS].to_numpy().ravel(),
            q[[f"{s}_upper" for s in STATS[:3]]].to_numpy().ravel(),
        ]
    )
    span = max(hi - lo, 0.1)
    ax.set_ylim(lo - 0.025 * span, hi + 0.29 * span)
    ax.legend(
        loc="upper left",
        ncol=1,
        fontsize=7.5,
        frameon=False,
        handlelength=1.2,
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
    table, ctx, out_dir, *, stem="transfer_family_complete_multiway_size_olmo"
):
    """Export only the size/OLMo figure, preserving the main top-row artifacts."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig = draw_size_olmo(table, ctx)
    fig.canvas.draw()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"{stem}.{ext}", dpi=200)
    plt.close(fig)


def draw(table, ctx, *, combined=True, quantile_group="pooled"):
    """Return a two-row figure; size/architecture and OLMo share the bottom row."""
    with plt.rc_context({"font.size": 10, "pdf.fonttype": 42, "svg.fonttype": "none"}):
        fig = plt.figure(figsize=(16, 9) if combined else (16, 8.4))
        gs = fig.add_gridspec(
            2,
            2,
            left=0.055,
            right=0.985,
            top=0.94,
            bottom=0.17,
            hspace=0.76,
            wspace=0.23,
        )
        ax = fig.add_subplot(gs[0, 0] if combined else gs[0, :])
        _win_bars(
            ax,
            table,
            registry.relation_groups(ctx.relations)[0],
            ["pooled", *tfp.DEFAULT_ANIMALS],
        )
        ax.set_title("A   Pythia 160M relationships", loc="left", fontsize=12, pad=10)
        if combined:
            ax = fig.add_subplot(gs[0, 1])
            # The magnitude panel keeps the ORIGINAL four families (fixed colors/labels).
            _quantile_bars(ax, table, list(ctx.relations[:4]), quantile_group)
            animal_label = "Pooled animals" if quantile_group == "pooled" else "Dog"
            ax.set_title(
                f"B   {animal_label}: magnitude of transfer",
                loc="left",
                fontsize=12,
                pad=10,
            )
        ax = fig.add_subplot(gs[1, 0])
        _win_bars(ax, table, registry.relation_groups(ctx.relations)[1], ["dog"])
        ax.set_title(
            f"{'C' if combined else 'B'}   Size and architecture — dog",
            loc="left",
            fontsize=12,
            pad=10,
        )
        ax = fig.add_subplot(gs[1, 1])
        _win_bars(ax, table, registry.relation_groups(ctx.relations)[2], ["elephant"])
        ax.set_title(
            f"{'D' if combined else 'C'}   OLMo corpus and seed — elephant",
            loc="left",
            fontsize=12,
            pad=10,
        )
        draws = int(table.n_bootstrap.iloc[0])
        fig.text(
            0.52,
            0.048,
            f"Equal weight per matched run · 95% multiway bootstrap intervals ({draws:,} draws) · All collapse states included\n"
            "Complete core30 / core56 / core35 teacher × arm blocks at student ftseed 1.\n"
            "n = matched runs. Maximum is descriptive, without a confidence interval.",
            ha="center",
            va="center",
            fontsize=8.5,
        )
        return fig


def generate(frame, ctx, out_dir, *, n_bootstrap=2000, seed=0):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    table = summarize(frame, ctx, n_bootstrap=n_bootstrap, seed=seed)
    table.to_csv(out / "complete_multiway_summary.csv", index=False)
    for stem, combined, group in [
        ("transfer_family_complete_multiway_win", False, "pooled"),
        ("transfer_family_complete_multiway_combined", True, "pooled"),
        ("transfer_family_complete_multiway_combined_dog", True, "dog"),
    ]:
        fig = draw(table, ctx, combined=combined, quantile_group=group)
        for ext in ("png", "pdf", "svg"):
            fig.savefig(out / f"{stem}.{ext}", dpi=200)
        plt.close(fig)
    return table


def publish_panels(
    repo,
    root,
    *,
    olmo_dir=PUBLISHED_OLMO_OUT,
    olmo_name=PUBLISHED_OLMO_NAME,
    all_dir=PUBLISHED_ALL_OUT,
    n_bootstrap=2000,
    seed=0,
    only="both",
    render_only=False,
):
    """Render the published OLMo/size and top-row panels from one fresh scan.

    The OLMo same-model diagonal is opted in so its reserved category can fill.
    ``only`` restricts the published figures; ``render_only`` redraws them from
    each directory's saved summary, leaving every CSV on disk untouched.
    """
    if only not in {"both", "olmo", "top-row"}:
        raise ValueError("only must be both, olmo or top-row")
    olmo = olmo_dir if Path(olmo_dir).is_absolute() else repo / olmo_dir
    every = all_dir if Path(all_dir).is_absolute() else repo / all_dir
    targets = {"both": (olmo, every), "olmo": (olmo,), "top-row": (every,)}[only]
    ctx, table = tfp.default_context(), None
    if not render_only:
        from subliminal_learning.analysis.transfer_family_complete import (
            load_complete_pairs,
        )

        frame, ctx, blocks, excluded, coverage = load_complete_pairs(
            repo, root, include_olmo_same_model=True
        )
        table = summarize(frame, ctx, n_bootstrap=n_bootstrap, seed=seed)
        for out in targets:
            out.mkdir(parents=True, exist_ok=True)
            table.to_csv(out / "complete_multiway_summary.csv", index=False)
        if olmo in targets:
            blocks.to_csv(olmo / "complete_blocks.csv", index=False)
            excluded.to_csv(olmo / "complete_cohort_exclusions.csv", index=False)
            coverage.to_csv(olmo / "complete_registry_coverage.csv", index=False)
            frame.to_csv(olmo / "complete_matched_pairs.csv", index=False)
    if olmo in targets:
        saved = (
            pd.read_csv(olmo / "complete_multiway_summary.csv")
            if table is None
            else table
        )
        export_size_olmo(saved, ctx, olmo, stem=olmo_name)
    if every in targets:
        summary = every / "complete_multiway_summary.csv"
        export_top_row(summary, every)
        export_top_row(summary, every, quantile_layout="families")
    print(f"Wrote published panels to {', '.join(map(str, targets))}", flush=True)
    return table


@app.command()
def main(
    out_dir: Path = typer.Option(DEFAULT_OUT),
    n_bootstrap: int = typer.Option(2000, min=1),
    seed: int = typer.Option(0),
    all_arms: bool = False,
    publish: bool = False,
    olmo_dir: Path = PUBLISHED_OLMO_OUT,
    olmo_name: str = PUBLISHED_OLMO_NAME,
    all_dir: Path = PUBLISHED_ALL_OUT,
    only: str = "both",
    render_only: bool = False,
):
    from subliminal_learning.analysis.transfer_family_complete import (
        load_complete_pairs,
    )

    repo, root = bootstrap()
    if publish:
        publish_panels(
            repo,
            root,
            olmo_dir=olmo_dir,
            olmo_name=olmo_name,
            all_dir=all_dir,
            n_bootstrap=n_bootstrap,
            seed=seed,
            only=only,
            render_only=render_only,
        )
        return
    load_kwargs = (
        {"include_modern_reproduction": True, "include_olmo_same_model": True}
        if all_arms
        else {}
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
        paths.append(repo / "run_logs/repretrain/repretrain_significance_arms.json")
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
        "include_olmo_same_model": all_arms,
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
        generate(frame, ctx, out, n_bootstrap=n_bootstrap, seed=seed)
    print(f"Wrote complete-grid multiway figures to {out}", flush=True)


def draw_top_row(table, ctx, *, quantile_group="pooled", quantile_layout="statistics"):
    """Two-panel top-row-only export using the saved summary values."""
    relations = registry.relation_groups(ctx.relations)[0][:4]
    with plt.rc_context({"font.size": 10, "pdf.fonttype": 42, "svg.fonttype": "none"}):
        fig, (left, right) = plt.subplots(
            1, 2, figsize=(12.5, 4.1), gridspec_kw={"wspace": 0.16}
        )
        _win_bars(
            left,
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
        left.set_xticks(
            np.arange(4),
            [
                "Same-model",
                "Same init,\ndifferent data order",
                "Different init,\nsame data order",
                "Different init,\ndifferent data order",
            ],
        )
        left.set_title("A   Pythia 160M relationships", loc="left", fontsize=12)
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
    summary_path, out_dir, *, quantile_group="pooled", quantile_layout="statistics"
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
        stem = "transfer_family_complete_multiway_top_row"
        if quantile_group != "pooled":
            stem += f"_{quantile_group}"
        if quantile_layout == "families":
            stem += "_families"
        fig.savefig(out / f"{stem}.{ext}", dpi=200)
    plt.close(fig)


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
    "run_logs/olmo_samemodel_core35_20260924/README.md",
    "run_logs/olmo_samemodel_core35_20260924/make_jobs.py",
    "run_logs/dolma_top10_senders.json",
    "run_logs/c4_top10_senders.json",
    "run_logs/olmo_dclm_to_c4_core35_20260920/teachers.tsv",
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


def _percentile_figure(table, relations, groups, title):
    fig, axes = plt.subplots(2, 2, figsize=(17, 10))
    width = 0.78 / len(groups)
    support = table[table.group.eq(groups[0])].set_index("relation").reindex(relations)
    labels = []
    for r in relations:
        label = ALL_ARMS_LABELS.get(r, r)
        if r == tfp.RELATION_REPRO:
            label = "Reproduction\n(Rerun1234 ↔\noriginal)"
        elif r == tfp.RELATION_MODERN_REPRO:
            label = "Modern Tree\nReproduction\n(Modern1234 ↔\noriginal)"
        labels.append(label + "\n" + _support_text(support.loc[r]).replace("; ", "\n"))
    for ax, stat, stat_label in zip(axes.flat, STATS, STAT_LABELS):
        for j, group in enumerate(groups):
            q = table[table.group.eq(group)].set_index("relation").reindex(relations)
            values = q[stat].to_numpy(float)
            finite = np.isfinite(values)
            x = np.arange(len(relations)) + (j - (len(groups) - 1) / 2) * width
            ax.bar(
                x[finite],
                values[finite],
                width * 0.91,
                color=ALL_ARMS_COLORS[group],
                edgecolor="#555555",
                linewidth=0.6,
                label=group,
            )
            if stat != "max":
                _intervals(
                    ax,
                    x,
                    q[f"{stat}_lower"],
                    q[f"{stat}_upper"],
                    min(0.045, width * 0.3),
                )
        for i, r in enumerate(relations):
            if not support.loc[r, "n_pairs"]:
                ax.text(
                    i,
                    0.04,
                    "No complete\nblock",
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#777777",
                )
        ax.set_title(stat_label if stat != "max" else "Observed maximum (no CI)")
        ax.set_ylabel("p(student) − p(control)")
        ax.axhline(0, color="#555555", lw=0.6)
        ax.set_xticks(np.arange(len(relations)), labels, fontsize=7.5)
        ax.set_xlim(-0.6, len(relations) - 0.4)
        _style(ax)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=len(groups),
        frameon=False,
    )
    fig.suptitle(title + " — raw difference", fontsize=15, y=0.99)
    fig.subplots_adjust(
        left=0.065, right=0.985, bottom=0.13, top=0.87, wspace=0.22, hspace=0.68
    )
    return fig


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
    """Save raw percentile figures only, using the complete-cohort multiway summary."""
    ctx = _overview_context(ctx)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    table = summarize(frame, ctx, n_bootstrap=n_bootstrap, seed=seed)
    table.to_csv(out / "transfer_family_multiway_raw_summary.csv", index=False)
    base, size, olmo = registry.relation_groups(ctx.relations)
    specs = [
        (base, ["pooled", "cat", "dog", "lion", "pig"], "Pythia 160M relationships"),
        (size, ["dog"], "Size and architecture — dog"),
        (olmo, ["elephant"], "OLMo corpus and seed — elephant"),
    ]
    figures = [
        ("transfer_family_multiway_raw_percentiles", _percentile_overview(table, ctx)),
        (
            "transfer_family_multiway_raw_percentiles_with_metadata",
            _percentile_overview(table, ctx, with_metadata=True),
        ),
    ]
    figures.extend(
        (
            f"transfer_family_multiway_raw_percentiles_panel{i}",
            _percentile_figure(table, relations, groups, title),
        )
        for i, (relations, groups, title) in enumerate(specs, 1)
    )
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
